#!/usr/bin/env python3
"""
Model training module for the Sand Mining Detection Tool.
Literature-upgraded: SMOTE class balancing, 5-fold stratified CV,
PR-AUC metric, RF preferred over SVM for generalisation.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── SMOTE for class imbalance (literature recommendation) ──────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. SMOTE disabled. Run: pip install imbalanced-learn")

from src import config
from src import features
from src.utils import load_labels, create_lat_lon_mapping


# ── Annotation-derived (leakage-prone) features ─────────────────────────────
# These features are computed from the polygons a human drew in the labeling
# GUI. Because the GUI auto-sets an image's label to 1 when a `sand_mining`
# region is drawn (see gui.py: on_mouse_release), these features encode the
# ANNOTATOR'S DECISION rather than independent image evidence — a classifier
# can score near-perfectly just by checking `num_sand_mining_areas > 0`.
#
# That is target leakage. It is legitimate for a human-in-the-loop workflow
# (a user marks regions, the model refines them), but it must NOT be used to
# claim detection accuracy on unlabeled imagery. Use
# exclude_annotation_features=True to train an honest imagery-only model.
ANNOTATION_FEATURE_PREFIXES = (
    'sand_mining_', 'equipment_', 'water_disturbance_', 'no_mining_',
    'num_', 'total_',
)
ANNOTATION_FEATURE_SUFFIXES = ('_area_ratio',)


def is_annotation_derived(feature_name):
    """True if a feature is derived from hand-drawn annotations (leakage-prone)."""
    if feature_name.startswith(ANNOTATION_FEATURE_PREFIXES):
        return True
    if feature_name.endswith(ANNOTATION_FEATURE_SUFFIXES):
        return True
    return False


def report_leakage_risk(features_df, feature_names):
    """
    Print a diagnostic showing how strongly annotation-derived features
    correlate with the label. |r| near 1.0 means the feature is effectively a
    copy of the target and any resulting accuracy is meaningless.
    """
    if 'label' not in features_df.columns:
        return

    ann = [f for f in feature_names if is_annotation_derived(f)]
    if not ann:
        return

    labeled = features_df[features_df['label'] != -1]
    if len(labeled) < 3:
        return

    y = labeled['label'].astype(float)
    risky = []
    for f in ann:
        if f not in labeled.columns:
            continue
        col = labeled[f].astype(float)
        if col.std() == 0:
            continue
        r = col.corr(y)
        if pd.notna(r) and abs(r) >= 0.7:
            risky.append((f, r))

    if risky:
        risky.sort(key=lambda x: abs(x[1]), reverse=True)
        print("\n" + "!"*80)
        print(" ⚠️  TARGET LEAKAGE WARNING")
        print("!"*80)
        print(f" {len(risky)} annotation-derived feature(s) correlate |r| >= 0.7 with the label.")
        print(" These come from polygons YOU drew, and the labeling GUI auto-sets")
        print(" label=1 when a sand_mining region is drawn — so they partly encode")
        print(" the label itself, not independent image evidence.")
        print("\n Worst offenders:")
        for f, r in risky[:8]:
            print(f"   r={r:+.3f}  {f}")
        print("\n Accuracy from a model using these is NOT a valid estimate of")
        print(" detection performance on unlabeled imagery.")
        print(" Re-run with --no-annotation-features for an honest number.")
        print("!"*80 + "\n")


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE oversampling to handle class imbalance.
    Literature: Li et al. (2024) — SMOTE+balanced accuracy for 0.85 F1 on Mekong.
    Falls back gracefully if imbalanced-learn not installed or dataset too small.
    """
    if not SMOTE_AVAILABLE:
        return X_train, y_train

    counts = np.bincount(y_train)
    minority = counts.min()

    if minority < 2:
        print("  SMOTE skipped: fewer than 2 samples in minority class.")
        return X_train, y_train

    # k_neighbors must be < minority class count
    k = min(5, minority - 1)
    try:
        sm = SMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"  SMOTE applied: {len(X_train)} → {len(X_res)} samples "
              f"(minority class: {minority} → {np.bincount(y_res).min()})")
        return X_res, y_res
    except Exception as e:
        print(f"  SMOTE failed ({e}), using original data.")
        return X_train, y_train


def build_spatial_groups(features_df, eps_km=2.0):
    """
    Assign a spatial group id to each labeled sample so that overlapping /
    near-duplicate images never straddle a cross-validation fold boundary.

    Training points are sampled every 0.2 km along a river while each image
    covers a `2 * DEFAULT_BUFFER_METERS` box (3 km by default). Two points a
    few hundred metres apart therefore yield images that overlap almost
    completely. Under plain random k-fold, one copy lands in training and the
    other in validation, and the model "predicts" an image it has effectively
    already seen — spatial autocorrelation leakage, the standard failure mode
    for remote-sensing ML.

    Points within `eps_km` of each other are merged into one group (greedy
    single-linkage over haversine distance), and CV is run with
    StratifiedGroupKFold over those groups.

    Returns:
        np.ndarray of group ids aligned with prepare_training_data's row order,
        or None if coordinates could not be parsed from the filenames.
    """
    from src.features import parse_lat_lon_from_filename

    labeled = features_df[features_df['label'] != -1]
    if 'filename' not in labeled.columns:
        return None

    coords = []
    for fn in labeled['filename']:
        lat, lon = parse_lat_lon_from_filename(str(fn))
        coords.append((lat, lon))

    if any(c[0] is None for c in coords):
        return None

    groups = np.full(len(coords), -1, dtype=int)
    next_group = 0
    for i, (lat_i, lon_i) in enumerate(coords):
        if groups[i] != -1:
            continue
        groups[i] = next_group
        # Greedy expansion: pull in every unassigned point within eps_km
        changed = True
        while changed:
            changed = False
            member_idx = np.where(groups == next_group)[0]
            for j, (lat_j, lon_j) in enumerate(coords):
                if groups[j] != -1:
                    continue
                for m in member_idx:
                    lat_m, lon_m = coords[m]
                    dlat = np.radians(lat_j - lat_m)
                    dlon = np.radians(lon_j - lon_m)
                    a = (np.sin(dlat / 2) ** 2 +
                         np.cos(np.radians(lat_m)) * np.cos(np.radians(lat_j)) *
                         np.sin(dlon / 2) ** 2)
                    dist_km = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                    if dist_km <= eps_km:
                        groups[j] = next_group
                        changed = True
                        break
        next_group += 1

    n_groups = len(np.unique(groups))
    n_merged = len(groups) - n_groups
    print(f"\n🗺️  Spatial blocking: {len(groups)} samples → {n_groups} spatial groups "
          f"(within {eps_km} km merged).")
    if n_merged > 0:
        print(f"   {n_merged} sample(s) share a location with another and would "
              f"otherwise leak across CV folds.")
    return groups


def build_cv_pipeline(model):
    """
    Wrap a model so that scaling (and SMOTE, when available) happen INSIDE
    each cross-validation fold, fit only on that fold's training portion.

    Doing these steps outside CV — as the original code did by scaling the
    full dataset up front — leaks information from the validation folds and
    inflates the reported scores.
    """
    from sklearn.pipeline import Pipeline as SkPipeline

    if SMOTE_AVAILABLE:
        try:
            from imblearn.pipeline import Pipeline as ImbPipeline
            return ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=config.RANDOM_STATE, k_neighbors=1)),
                ('model', model),
            ])
        except Exception:
            pass

    return SkPipeline([('scaler', StandardScaler()), ('model', model)])


def make_cv_splitter(y, groups=None, n_splits=5):
    """
    Build a CV splitter. When spatial `groups` are supplied, uses
    StratifiedGroupKFold so overlapping images stay within a single fold —
    this is what prevents spatial-autocorrelation leakage from inflating
    the scores. Falls back to StratifiedKFold when groups are unavailable.
    """
    counts = np.bincount(y)
    minority = int(counts[np.nonzero(counts)].min())
    n_splits = min(n_splits, minority)

    if groups is not None:
        n_splits = min(n_splits, len(np.unique(groups)))

    if n_splits < 2:
        return None, n_splits

    if groups is not None:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=config.RANDOM_STATE), n_splits
        except ImportError:
            print("  (StratifiedGroupKFold unavailable — needs scikit-learn >= 1.0; "
                  "falling back to non-spatial CV)")

    return StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=config.RANDOM_STATE), n_splits


def run_cross_validation(model, X, y, n_splits=5, groups=None):
    """
    Stratified k-fold CV with per-fold scaling/SMOTE, spatially blocked when
    `groups` is supplied. Returns {metric: (mean, std)} or {} if the dataset
    is too small to cross-validate.
    """
    counts = np.bincount(y)
    minority = counts[np.nonzero(counts)].min()

    skf, n_splits = make_cv_splitter(y, groups=groups, n_splits=n_splits)

    if skf is None:
        print(f"\n⚠️  Cannot cross-validate: minority class has only {minority} sample(s).")
        return {}

    if n_splits < 5:
        print(f"\n⚠️  Reduced to {n_splits}-fold CV "
              f"(minority class: {minority} samples"
              f"{', spatial groups: ' + str(len(np.unique(groups))) if groups is not None else ''}).")

    results = {}
    for metric in ['f1', 'roc_auc', 'average_precision', 'balanced_accuracy']:
        try:
            scores = cross_val_score(
                build_cv_pipeline(model), X, y, cv=skf, groups=groups,
                scoring=metric, n_jobs=1
            )
            label = {'average_precision': 'PR-AUC', 'roc_auc': 'ROC-AUC',
                     'balanced_accuracy': 'bal. acc', 'f1': 'F1'}[metric]
            results[label] = (float(np.mean(scores)), float(np.std(scores)))
        except Exception as e:
            print(f"  CV failed for {metric}: {e}")
    return results


def prepare_training_data(features_df, exclude_annotation_features=False):
    """
    Prepare the training data by filtering and processing.

    Args:
        features_df: DataFrame of extracted features with a 'label' column.
        exclude_annotation_features (bool): drop features derived from
            hand-drawn annotations (see ANNOTATION_FEATURE_PREFIXES). Set True
            to train a leakage-free, imagery-only model whose accuracy is a
            valid estimate of performance on unlabeled imagery.

    Returns (X, y, feature_names).
    """
    if 'label' not in features_df.columns:
        print("Error: No 'label' column found in features data.")
        return None, None, None

    labeled_df = features_df[features_df['label'] != -1].copy()

    if len(labeled_df) < 10:
        print(f"Error: Insufficient labeled data ({len(labeled_df)} samples). Need at least 10.")
        return None, None, None

    class_counts = labeled_df['label'].value_counts()
    print("Class distribution in training data:")
    print(class_counts)

    if len(class_counts) < 2:
        print("Error: Training data contains only one class. Cannot train model.")
        return None, None, None

    feature_columns = [col for col in labeled_df.columns if col not in ['label', 'filename']]

    if exclude_annotation_features:
        dropped = [c for c in feature_columns if is_annotation_derived(c)]
        feature_columns = [c for c in feature_columns if not is_annotation_derived(c)]
        print(f"\n🔒 Leakage-free mode: dropped {len(dropped)} annotation-derived features.")
        print(f"   Training on {len(feature_columns)} imagery-only features "
              f"(spectral, texture, GLCM, historical trends).")
        if not feature_columns:
            print("Error: no features left after excluding annotation-derived ones.")
            return None, None, None

    labeled_df[feature_columns] = labeled_df[feature_columns].fillna(0)

    X = labeled_df[feature_columns].values
    y = labeled_df['label'].values

    return X, y, feature_columns


def train_multiple_models(X, y, feature_names, test_size=0.25, random_state=42,
                          groups=None):
    """
    Train and evaluate multiple ML models with:
    - SMOTE for class imbalance          (Li et al. 2024)
    - 5-fold stratified CV               (EuroMineNet 2025)
    - PR-AUC + F1 primary metrics        (EuroMineNet 2025)
    - RF preferred over SVM as default   (Gallwey et al. 2020; prevents SVM overfitting on small datasets)
    """
    print("\nTraining and evaluating multiple models...")

    # BUGFIX (data leakage): scaler is now fit on the training split only,
    # not on the full dataset before splitting.
    scaler = StandardScaler()
    minority_class_count = min(np.bincount(y)[np.nonzero(np.bincount(y))[0]])

    run_evaluation = False
    X_train, X_test, y_train, y_test = None, None, y, None

    if X.shape[0] >= 4 and minority_class_count >= 2:
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            if len(np.unique(y_test)) == 2:
                X_train = scaler.fit_transform(X_train_raw)
                X_test = scaler.transform(X_test_raw)
                print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
                run_evaluation = True
            else:
                print("Warning: Test set has only one class. Training on full data.")
                X_train, y_train = scaler.fit_transform(X), y
                X_test, y_test = None, None
        except ValueError as e:
            print(f"Warning: Could not stratify split: {e}. Training on full data.")
            X_train, y_train = scaler.fit_transform(X), y

    if X_train is None:
        X_train, y_train = scaler.fit_transform(X), y

    # ── Apply SMOTE to training set only ────────────────────────────────────
    print("\nApplying SMOTE to handle class imbalance...")
    X_train_bal, y_train_bal = apply_smote(X_train, y_train, random_state)

    # ── Model definitions ────────────────────────────────────────────────────
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1,
            min_samples_leaf=config.MIN_SAMPLES_LEAF
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=random_state,
            learning_rate=0.1,
            max_depth=3
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            eval_metric='logloss',
            verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            random_state=random_state,
            verbose=-1
        ),
        "SVM": SVC(
            probability=True,
            random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )
    }

    results = pd.DataFrame()
    trained_models = {}
    best_model = None
    best_score = -1
    best_model_name = None

    viz_dir = os.path.join(config.MODELS_DIR, 'model_comparisons')
    os.makedirs(viz_dir, exist_ok=True)

    # ── 5-fold stratified CV ─────────────────────────────────────────────────
    # Spatially blocked CV when groups are available (prevents overlapping
    # images from straddling a fold boundary), else plain stratified CV.
    skf, _n_splits = make_cv_splitter(y, groups=groups, n_splits=5)
    if skf is None:
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
        _n_splits = 2
    print(f"  (using {_n_splits}-fold "
          f"{'spatially blocked' if groups is not None else 'stratified'} CV)")

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train_bal, y_train_bal)
        trained_models[model_name] = model

        result = {'Model': model_name}

        if run_evaluation and X_test is not None:
            print(f"  Evaluating {model_name}...")
            y_pred = model.predict(X_test)

            result['Accuracy']  = accuracy_score(y_test, y_pred)
            result['Precision'] = precision_score(y_test, y_pred, zero_division=0)
            result['Recall']    = recall_score(y_test, y_pred, zero_division=0)
            result['F1 Score']  = f1_score(y_test, y_pred, zero_division=0)

            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                try:
                    result['ROC AUC'] = roc_auc_score(y_test, y_proba)
                except Exception:
                    result['ROC AUC'] = 0.0
                # ── PR-AUC: key metric for rare-event / imbalanced classes ──
                try:
                    result['PR AUC'] = average_precision_score(y_test, y_proba)
                except Exception:
                    result['PR AUC'] = 0.0
            else:
                result['ROC AUC'] = 0.0
                result['PR AUC']  = 0.0

            # ── Stratified CV with per-fold scaling + SMOTE (no leakage) ──
            # Previously this cross-validated on X_scaled, which had been
            # scaled using the whole dataset — leaking validation-fold
            # statistics into every fold's training data.
            try:
                cv_f1 = cross_val_score(build_cv_pipeline(model), X, y,
                                        cv=skf, groups=groups,
                                        scoring='f1', n_jobs=1)
                result['CV F1 Mean'] = cv_f1.mean()
                result['CV F1 Std']  = cv_f1.std()
                try:
                    cv_pr = cross_val_score(build_cv_pipeline(model), X, y, cv=skf,
                                            groups=groups,
                                            scoring='average_precision', n_jobs=1)
                    result['CV PR-AUC'] = cv_pr.mean()
                except Exception:
                    result['CV PR-AUC'] = 0.0
                print(f"  CV F1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
            except Exception as cv_err:
                result['CV F1 Mean'] = 0.0
                result['CV F1 Std']  = 0.0
                print(f"  CV failed: {cv_err}")

        results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)

    # ── Best model selection: prefer RF/XGB/LGBM over SVM ───────────────────
    # Literature (Gallwey 2020, EuroMineNet 2025): SVM overfits small datasets.
    # Primary metric: CV F1 Mean if available, else holdout F1.
    print("\nModel Performance Comparison:")
    if 'CV F1 Mean' in results.columns:
        display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score',
                        'PR AUC', 'CV F1 Mean', 'CV F1 Std', 'CV PR-AUC']
        display_cols = [c for c in display_cols if c in results.columns]
        print(results[display_cols].to_string(index=False))
        score_col = 'CV F1 Mean'
    elif 'F1 Score' in results.columns:
        print(results.to_string(index=False))
        score_col = 'F1 Score'
    else:
        print(results.to_string(index=False))
        score_col = None

    ENSEMBLE_MODELS = {'Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting'}

    if score_col and score_col in results.columns:
        # First try ensemble models only
        ensemble_results = results[results['Model'].isin(ENSEMBLE_MODELS)]
        if not ensemble_results.empty:
            best_idx = ensemble_results[score_col].idxmax()
            best_model_name = results.loc[best_idx, 'Model']
            best_score = results.loc[best_idx, score_col]
        else:
            best_idx = results[score_col].idxmax()
            best_model_name = results.loc[best_idx, 'Model']
            best_score = results.loc[best_idx, score_col]
    else:
        # Fallback: pick Random Forest
        best_model_name = 'Random Forest'
        best_score = 0.0

    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name} ({score_col}: {best_score:.4f})")
    print("Note: Ensemble models preferred over SVM/LR to prevent overfitting on small datasets.")

    # ── Feature importance ───────────────────────────────────────────────────
    feature_importance_dict = {}
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance_dict = {
            feature_names[i]: float(importances[i]) for i in range(len(feature_names))
        }

    # ── Save model comparison CSV ────────────────────────────────────────────
    results_file = os.path.join(config.MODELS_DIR, 'model_comparison_results.csv')
    results.to_csv(results_file, index=False)
    print(f"Model comparison results saved to {results_file}")

    return best_model, results, feature_importance_dict, trained_models, scaler


def train_model(X, y, feature_names, model_type='random_forest',
                use_grid_search=False, groups=None):
    """
    Train a single model with SMOTE and proper evaluation.
    """
    print(f"\nTraining {model_type} model...")

    # BUGFIX (data leakage): the scaler used to be fit on the FULL dataset
    # before the train/test split, letting test-set statistics influence
    # training. The split now happens first and the scaler is fit on the
    # training portion only, then applied to the test set.
    scaler = StandardScaler()
    run_evaluation = False
    X_train, X_test, y_train, y_test = None, None, y, None

    minority_class_count = min(np.bincount(y)[np.nonzero(np.bincount(y))[0]])

    if X.shape[0] >= 4 and minority_class_count >= 2:
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE, stratify=y
            )
            if len(np.unique(y_test)) == 2:
                X_train = scaler.fit_transform(X_train_raw)
                X_test = scaler.transform(X_test_raw)
                print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
                run_evaluation = True
            else:
                X_train, y_train = scaler.fit_transform(X), y
                X_test, y_test = None, None
        except ValueError as e:
            print(f"Warning: Could not stratify: {e}")
            X_train, y_train = scaler.fit_transform(X), y

    if X_train is None:
        X_train, y_train = scaler.fit_transform(X), y

    # Apply SMOTE
    X_train, y_train = apply_smote(X_train, y_train, config.RANDOM_STATE)

    if model_type == 'random_forest':
        base_model = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1,
            min_samples_leaf=config.MIN_SAMPLES_LEAF
        )
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'gradient_boosting':
        base_model = GradientBoostingClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            learning_rate=0.1,
            max_depth=3
        )
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    else:
        print(f"Unknown model type: {model_type}. Using RandomForest.")
        base_model = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )
        param_grid = {}

    if use_grid_search and param_grid:
        print("Performing grid search...")
        gs = GridSearchCV(base_model, param_grid, cv=5,
                          scoring='f1', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        print(f"Best params: {gs.best_params_}")
        final_model = gs.best_estimator_
    else:
        final_model = base_model
        final_model.fit(X_train, y_train)

    if run_evaluation and X_test is not None:
        y_pred = final_model.predict(X_test)
        print(f"\n── Hold-out evaluation (n={len(y_test)}) ──")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        if hasattr(final_model, 'predict_proba'):
            y_proba = final_model.predict_proba(X_test)[:, 1]
            print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
            print(f"PR  AUC: {average_precision_score(y_test, y_proba):.4f}")
        if len(y_test) < 20:
            print(f"\n⚠️  Only {len(y_test)} test samples — these numbers are"
                  " statistically unreliable. Use the cross-validated scores below.")

    # ── Cross-validated scores (the number to actually report) ──────────────
    # Scaling and SMOTE are applied INSIDE each fold via a pipeline, so no
    # information leaks from validation folds into training.
    cv_scores = run_cross_validation(base_model, X, y, groups=groups)
    if cv_scores:
        label = "spatially blocked" if groups is not None else "stratified"
        print(f"\n── Cross-validation ({label}) — report these ──")
        for metric, (mean, std) in cv_scores.items():
            print(f"  {metric:<10} {mean:.3f} ± {std:.3f}")

    feature_importance_dict = {}
    if hasattr(final_model, 'feature_importances_'):
        feature_importance_dict = {
            feature_names[i]: float(final_model.feature_importances_[i])
            for i in range(len(feature_names))
        }

    return final_model, scaler, feature_importance_dict


def save_model_and_metadata(model, scaler, feature_importance_dict,
                             feature_names, all_models=None, model_results=None):
    """Save trained model, scaler, and metadata."""
    try:
        os.makedirs(config.MODELS_DIR, exist_ok=True)

        joblib.dump(model, config.DEFAULT_MODEL_FILE)
        print(f"Model saved to: {config.DEFAULT_MODEL_FILE}")

        joblib.dump(scaler, config.DEFAULT_SCALER_FILE)
        print(f"Feature scaler saved to: {config.DEFAULT_SCALER_FILE}")

        with open(config.DEFAULT_FEATURE_IMPORTANCE_FILE, 'w') as f:
            json.dump(feature_importance_dict, f, indent=2)
        print(f"Feature importance saved to: {config.DEFAULT_FEATURE_IMPORTANCE_FILE}")

        feature_names_file = os.path.join(config.MODELS_DIR, 'feature_names.json')
        with open(feature_names_file, 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"Feature names saved to: {feature_names_file}")

        if all_models is not None:
            model_name_map = {
                'Random Forest':      config.RF_MODEL_FILE,
                'Gradient Boosting':  config.GB_MODEL_FILE,
                'XGBoost':            config.XGB_MODEL_FILE,
                'LightGBM':           config.LGBM_MODEL_FILE,
                'SVM':                config.SVM_MODEL_FILE,
                'Logistic Regression':config.LR_MODEL_FILE,
            }
            for mname, mobj in all_models.items():
                mfile = model_name_map.get(mname,
                    os.path.join(config.MODELS_DIR,
                                 f"{mname.replace(' ', '_').lower()}_model.pkl"))
                joblib.dump(mobj, mfile)
                print(f"Saved {mname} model to: {mfile}")

        if model_results is not None and not model_results.empty:
            results_file = os.path.join(config.MODELS_DIR, 'model_comparison_results.csv')
            model_results.to_csv(results_file, index=False)
            print(f"Model comparison results saved to: {results_file}")

        return True
    except Exception as e:
        print(f"Error saving model and metadata: {e}")
        return False


def load_model_and_metadata():
    """Load trained model, scaler, and feature names."""
    try:
        if not os.path.exists(config.DEFAULT_MODEL_FILE):
            print(f"Error: Model file not found at {config.DEFAULT_MODEL_FILE}")
            return None, None, None
        if not os.path.exists(config.DEFAULT_SCALER_FILE):
            print(f"Error: Scaler file not found at {config.DEFAULT_SCALER_FILE}")
            return None, None, None

        model  = joblib.load(config.DEFAULT_MODEL_FILE)
        print(f"Model loaded from: {config.DEFAULT_MODEL_FILE}")

        scaler = joblib.load(config.DEFAULT_SCALER_FILE)
        print(f"Scaler loaded from: {config.DEFAULT_SCALER_FILE}")

        feature_names_file = os.path.join(config.MODELS_DIR, 'feature_names.json')
        if os.path.exists(feature_names_file):
            with open(feature_names_file, 'r') as f:
                feature_names = json.load(f)
            print(f"Feature names loaded from: {feature_names_file}")
        else:
            feature_names = None

        return model, scaler, feature_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def _discover_rivers():
    """Find every river that has labels saved, by scanning for training_labels_*.csv."""
    import glob
    rivers = []
    pattern = os.path.join(config.OUTPUT_DIR, 'training_labels_*.csv')
    for path in glob.glob(pattern):
        name = os.path.basename(path)[len('training_labels_'):-len('.csv')]
        rivers.append(name)
    return sorted(rivers)


def build_all_river_features(years_back=0):
    """
    Pool labeled data from EVERY river into one feature DataFrame.

    For each river it temporarily points config at that river's images, labels
    and annotations, extracts features (identically to single-river training),
    tags them with a 'river' column, then concatenates everything.
    """
    rivers = _discover_rivers()
    if not rivers:
        print("No per-river label files found (training_labels_<river>.csv).")
        return pd.DataFrame()

    print(f"\n[All-River] Found {len(rivers)} rivers: {', '.join(rivers)}")

    # Remember base paths so we can restore them
    base_images = os.path.join(config.OUTPUT_DIR, 'training_images')
    saved = (config.TRAINING_IMAGES_DIR, config.LABELS_FILE, config.ANNOTATIONS_FILE)

    frames = []
    for river in rivers:
        config.TRAINING_IMAGES_DIR = os.path.join(base_images, river)
        config.LABELS_FILE = os.path.join(config.OUTPUT_DIR, f'training_labels_{river}.csv')
        config.ANNOTATIONS_FILE = os.path.join(config.ANNOTATIONS_DIR,
                                                f'area_annotations_{river}.json')

        labels = load_labels()
        labeled = {k: v for k, v in labels.items() if v != -1}
        if not labeled:
            print(f"  {river}: no labeled images, skipping")
            continue

        labels_df = pd.DataFrame(list(labeled.items()), columns=['filename', 'label'])
        if years_back and years_back > 0:
            fdf = features.extract_features_from_df_with_history(
                config.TRAINING_IMAGES_DIR, labels_df,
                years_back=years_back,
                buffer_m=getattr(config, 'DEFAULT_BUFFER_METERS', 1500))
        else:
            fdf = features.extract_features_from_df(config.TRAINING_IMAGES_DIR, labels_df)

        if not fdf.empty:
            fdf['river'] = river
            frames.append(fdf)
            print(f"  {river}: {len(fdf)} samples")

    # Restore base paths
    config.TRAINING_IMAGES_DIR, config.LABELS_FILE, config.ANNOTATIONS_FILE = saved

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False).fillna(0)
    print(f"[All-River] Combined dataset: {len(combined)} samples, "
          f"{len(combined.columns) - 3} features")
    return combined


def run_all_river_training(use_grid_search=False, model_type='random_forest',
                           use_multiple_models=True, years_back=0,
                           exclude_annotation_features=False):
    """
    Train a SEPARATE global 'all-river' model on the pooled data of every river.
    Saved under MODELS_DIR/_ALL_RIVERS/ so it never overwrites per-river models.
    The good features (importances) from this global model are written alongside.
    """
    print("\n" + "="*80)
    print(" TRAINING ALL-RIVER (GLOBAL) MODEL")
    print("="*80 + "\n")

    combined = build_all_river_features(years_back=years_back)
    if combined.empty:
        print("❌ No data to train all-river model.")
        return False

    feature_df = combined.drop(columns=[c for c in ['river'] if c in combined.columns])
    report_leakage_risk(feature_df,
                        [c for c in feature_df.columns if c not in ['label', 'filename']])
    spatial_groups = build_spatial_groups(feature_df)
    X, y, feature_names = prepare_training_data(
        feature_df, exclude_annotation_features=exclude_annotation_features)
    if X is None:
        print("❌ Failed to prepare all-river training data.")
        return False

    if use_multiple_models:
        model, model_results, feature_importance, all_models, scaler = train_multiple_models(
            X, y, feature_names, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, groups=spatial_groups)
    else:
        model, scaler, feature_importance = train_model(
            X, y, feature_names, model_type=model_type,
            use_grid_search=use_grid_search, groups=spatial_groups)
        all_models, model_results = None, None

    # Save into a dedicated directory
    out_dir = os.path.join(config.MODELS_DIR, '_ALL_RIVERS')
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model,  os.path.join(out_dir, 'sand_mining_model.joblib'))
    joblib.dump(scaler, os.path.join(out_dir, 'feature_scaler.joblib'))
    with open(os.path.join(out_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    with open(os.path.join(out_dir, 'feature_importance.json'), 'w') as f:
        json.dump(feature_importance, f, indent=2)
    if model_results is not None and not model_results.empty:
        model_results.to_csv(os.path.join(out_dir, 'model_comparison_results.csv'), index=False)

    # Rank and persist the "good features" so the global model can be refined
    if feature_importance:
        good = {k: v for k, v in sorted(feature_importance.items(),
                                        key=lambda x: x[1], reverse=True) if v > 0}
        with open(os.path.join(out_dir, 'good_features.json'), 'w') as f:
            json.dump(good, f, indent=2)
        print(f"\nTop global features:")
        for k, v in list(good.items())[:12]:
            print(f"  {v:.4f}  {k}")

    print(f"\n✅ All-river global model saved to: {out_dir}")
    return True


def run_training_workflow(use_grid_search=False, model_type='random_forest',
                          use_multiple_models=False,
                          exclude_annotation_features=False):
    """Run the complete model training workflow."""
    print("\n" + "="*80)
    print(" TRAINING SAND MINING DETECTION MODEL")
    print(" Literature upgrades: SMOTE | 5-fold CV | PR-AUC | RF-preferred selection")
    print("="*80 + "\n")

    labels = load_labels()
    if not labels:
        print("No labels found. Please label images first.")
        return False

    labeled_count = sum(1 for label in labels.values() if label != -1)
    if labeled_count < 10:
        print(f"Only {labeled_count} labeled images. Please label at least 10.")
        return False

    print(f"Found {labeled_count} labeled images for training.")

    labels_df = pd.DataFrame(list(labels.items()), columns=['filename', 'label'])

    # Use historical trend features if configured (literature upgrade)
    use_history = getattr(config, 'HISTORICAL_YEARS_BACK', 0) > 0
    years_back  = getattr(config, 'HISTORICAL_YEARS_BACK', 0)

    if use_history:
        print(f"\nExtracting features + {years_back}-year historical trends...")
        print("Slower but produces a much stronger model (Li et al. 2024; Bendixen 2021).")
        features_df = features.extract_features_from_df_with_history(
            config.TRAINING_IMAGES_DIR, labels_df,
            years_back=years_back,
            buffer_m=getattr(config, 'DEFAULT_BUFFER_METERS', 1500)
        )
    else:
        print("\nExtracting features from labeled images (no historical trends)...")
        features_df = features.extract_features_from_df(config.TRAINING_IMAGES_DIR, labels_df)

    if features_df.empty:
        print("Failed to extract features.")
        return False

    print(f"Successfully extracted {len(features_df.columns) - 2} features "
          f"from {len(features_df)} images")

    print("\nAnalyzing feature correlations...")
    features.calculate_feature_correlation(
        features_df,
        output_file=os.path.join(config.MODELS_DIR, 'enhanced_feature_correlation.png')
    )

    # Diagnose target leakage from hand-drawn annotation features before training
    all_feature_cols = [c for c in features_df.columns if c not in ['label', 'filename']]
    report_leakage_risk(features_df, all_feature_cols)

    X, y, feature_names = prepare_training_data(
        features_df, exclude_annotation_features=exclude_annotation_features)
    if X is None:
        print("Failed to prepare training data.")
        return False

    # Spatial groups keep overlapping images inside one CV fold
    spatial_groups = build_spatial_groups(features_df)

    area_feats   = [f for f in feature_names if any(p in f for p in
                    ['sand_mining_', 'equipment_', 'water_disturbance_', 'num_', 'total_'])]
    global_feats = [f for f in feature_names if 'global_' in f]
    hist_feats   = [f for f in feature_names if '_trend' in f or 'historical' in f]

    print(f"\nFeature breakdown:")
    print(f"  Total:              {len(feature_names)}")
    print(f"  Area-specific:      {len(area_feats)}")
    print(f"  Global image:       {len(global_feats)}")
    print(f"  Historical trends:  {len(hist_feats)}")
    print(f"  Other:              {len(feature_names) - len(area_feats) - len(global_feats) - len(hist_feats)}")

    if use_multiple_models:
        print("\nTraining multiple model types for comparison...")
        model, model_results, feature_importance, all_models, scaler = train_multiple_models(
            X, y, feature_names,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            groups=spatial_groups
        )
        success = save_model_and_metadata(
            model, scaler, feature_importance, feature_names,
            all_models=all_models, model_results=model_results
        )
    else:
        model, scaler, feature_importance = train_model(
            X, y, feature_names,
            model_type=model_type,
            use_grid_search=use_grid_search,
            groups=spatial_groups
        )
        success = save_model_and_metadata(model, scaler, feature_importance, feature_names)

    if success:
        print("\n✅ Training workflow completed successfully!")
        return True
    else:
        print("\n❌ Training workflow failed.")
        return False