#!/usr/bin/env python3
"""
Model training module for the Sand Mining Detection Tool.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
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

from src import config
from src import features
from src.utils import load_labels, create_lat_lon_mapping

def prepare_training_data(features_df):
    """
    Prepare the training data by filtering and processing.
    
    Args:
        features_df (pd.DataFrame): DataFrame with features and labels
        
    Returns:
        tuple: (X, y, feature_names)
    """
    # Ensure we have a label column
    if 'label' not in features_df.columns:
        print("Error: No 'label' column found in features data.")
        return None, None, None
    
    # Filter out unlabeled data (-1)
    labeled_df = features_df[features_df['label'] != -1].copy()
    
    if len(labeled_df) < 10:
        print(f"Error: Insufficient labeled data ({len(labeled_df)} valid samples). Need at least 10 with labels 0 or 1.")
        return None, None, None
    
    # Check class balance
    class_counts = labeled_df['label'].value_counts()
    print("Class distribution in training data:")
    print(class_counts)
    
    if len(class_counts) < 2:
        print("Error: Training data contains only one class (0 or 1). Cannot train model.")
        print("Please ensure you have labeled examples for both 'Sand Mining' and 'No Sand Mining'.")
        return None, None, None
    
    # Keep a copy of feature names for later use
    feature_columns = [col for col in labeled_df.columns if col not in ['label', 'filename']]
    
    # Handle potential NaN values
    labeled_df[feature_columns] = labeled_df[feature_columns].fillna(0)
    
    X = labeled_df[feature_columns].values
    y = labeled_df['label'].values
    
    return X, y, feature_columns

def train_multiple_models(X, y, feature_names, test_size=0.25, random_state=42):
    """
    Train and evaluate multiple machine learning models.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        feature_names (list): List of feature names
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (best_model, all_models_results, feature_importance_dict, trained_models)
    """
    print("\nTraining and evaluating multiple models...")
    
    # Data scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data - ALWAYS KEEP TRAIN AND TEST SEPARATE
    minority_class_count = min(np.bincount(y)[np.nonzero(np.bincount(y))[0]])
    min_samples_split = 2  # sklearn needs at least 2 samples per class for stratification
    
    if X.shape[0] >= 4 and minority_class_count >= min_samples_split:
        try:
            # Use stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Verify test set has both classes
            if len(np.unique(y_test)) == 2:
                print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
                run_evaluation = True
            else:
                print("Warning: Stratified split resulted in only one class in the test set. Training on all data.")
                run_evaluation = False
                X_train, y_train = X_scaled, y  # Use all data for training
        except ValueError as e:
            print(f"Warning: Could not stratify data for train/test split (likely too few samples in one class): {e}")
            run_evaluation = False
            X_train, y_train = X_scaled, y  # Use all data for training
    else:
        print("Training final model on all data without evaluation split.")
        run_evaluation = False
        X_train, y_train = X_scaled, y  # Use all data for training
    
    # Define models to evaluate
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
            use_label_encoder=False,
            eval_metric='logloss'  # Suppress warnings
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            random_state=random_state
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
    
    # Metrics to evaluate
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': lambda y, y_pred: precision_score(y, y_pred, zero_division=0),
        'Recall': lambda y, y_pred: recall_score(y, y_pred, zero_division=0),
        'F1 Score': lambda y, y_pred: f1_score(y, y_pred, zero_division=0),
        'ROC AUC': lambda y, y_pred_proba: roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0,
        'PR AUC': lambda y, y_pred_proba: average_precision_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0
    }
    
    # Store results
    results = pd.DataFrame(columns=['Model'] + list(metrics.keys()))
    trained_models = {}
    best_model = None
    best_f1 = -1
    
    # Create output directory for visualizations
    viz_dir = os.path.join(config.MODELS_DIR, 'model_comparisons')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        # Only evaluate if we have a test set
        if run_evaluation:
            print(f"Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                except (AttributeError, IndexError):
                    y_pred_proba = y_pred
            else:
                y_pred_proba = y_pred
            
            # Calculate metrics
            result = {'Model': model_name}
            for metric_name, metric_func in metrics.items():
                try:
                    if metric_name in ['ROC AUC', 'PR AUC']:
                        score = metric_func(y_test, y_pred_proba)
                    else:
                        score = metric_func(y_test, y_pred)
                    result[metric_name] = score
                except Exception as e:
                    print(f"Warning: Could not calculate {metric_name} for {model_name}: {e}")
                    result[metric_name] = float('nan')
            
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
            
            # Create confusion matrix
            try:
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, interpolation='nearest', cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.colorbar()
                plt.xticks([0, 1], ['No Mining', 'Mining'])
                plt.yticks([0, 1], ['No Mining', 'Mining'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                
                # Add text annotations
                thresh = cm.max() / 2
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, format(cm[i, j], 'd'),
                                horizontalalignment="center",
                                color="white" if cm[i, j] > thresh else "black")
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f'{model_name.replace(" ", "_")}_confusion_matrix.png'))
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create confusion matrix for {model_name}: {e}")
            
            # Create ROC curve if applicable
            if hasattr(model, 'predict_proba'):
                try:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {model_name}')
                    plt.legend(loc="lower right")
                    plt.savefig(os.path.join(viz_dir, f'{model_name.replace(" ", "_")}_roc_curve.png'))
                    plt.close()
                    
                    # Create PR curve
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                    pr_auc = average_precision_score(y_test, y_pred_proba)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'Precision-Recall Curve - {model_name}')
                    plt.legend(loc="lower left")
                    plt.savefig(os.path.join(viz_dir, f'{model_name.replace(" ", "_")}_pr_curve.png'))
                    plt.close()
                except Exception as e:
                    print(f"Warning: Could not create ROC/PR curves for {model_name}: {e}")
            
            # Track the best model (by F1 score)
            if result.get('F1 Score', -1) > best_f1:
                best_f1 = result.get('F1 Score', -1)
                best_model = model
        else:
            # If no evaluation, just use the first model as "best"
            if best_model is None:
                best_model = model
    
    # If we have results, create comparison charts
    if not results.empty and run_evaluation:
        # Sort results by F1 score
        results = results.sort_values('F1 Score', ascending=False).reset_index(drop=True)
        
        # Save comparison results
        results_file = os.path.join(config.MODELS_DIR, 'model_comparison_results.csv')
        results.to_csv(results_file, index=False)
        print(f"Model comparison results saved to {results_file}")
        
        # Create comparative bar chart
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        results_subset = results[['Model'] + metrics_to_plot].set_index('Model')
        ax = results_subset.plot(kind='bar', figsize=(12, 8))
        plt.title('Model Comparison - Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'model_comparison.png'))
        plt.close()
        
        print("\nModel Performance Comparison:")
        print(results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']])
        
        best_model_name = results.iloc[0]['Model']
        print(f"\nBest model: {best_model_name} (F1 Score: {best_f1:.4f})")
    
    # Get feature importance from best model
    feature_importance_dict = {}
    if best_model is not None and hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        # Create dictionary mapping feature names to importance values
        feature_importance_dict = {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
        
        # Create feature importance visualization
        plt.figure(figsize=(10, 8))
        sorted_idx = np.argsort(importances)
        plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.title('Feature Importance (Best Model)')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
        plt.close()
        
        # Save feature importance
        with open(os.path.join(config.MODELS_DIR, 'feature_importance.json'), 'w') as f:
            json.dump(feature_importance_dict, f, indent=2)
    
    return best_model, results, feature_importance_dict, trained_models, scaler

def train_model(X, y, feature_names, model_type='random_forest', use_grid_search=False):
    """
    Train a single machine learning model.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        feature_names (list): List of feature names
        model_type (str): Type of model to train ('random_forest' or 'gradient_boosting')
        use_grid_search (bool): Whether to use grid search for hyperparameter tuning
        
    Returns:
        tuple: (model, scaler, feature_importance_dict)
    """
    print(f"\nTraining {model_type} model...")
    
    # Data scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    run_evaluation = False
    X_train, X_test, y_train, y_test = None, None, None, None
    
    # Check if enough samples per class exist for split
    minority_class_count = min(np.bincount(y)[np.nonzero(np.bincount(y))[0]])
    min_samples_split = 2  # sklearn needs at least 2 samples per class for stratification
    
    if X.shape[0] >= 4 and minority_class_count >= min_samples_split:
        try:
            # Use stratified split
            test_size = config.TEST_SIZE
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
            )
            
            # Verify test set has both classes if stratification succeeded
            if len(np.unique(y_test)) == 2:
                print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
                run_evaluation = True
            else:
                print("Warning: Stratified split resulted in only one class in the test set. Training on all data.")
                run_evaluation = False
        except ValueError as e:
            print(f"Warning: Could not stratify data for train/test split (likely too few samples in one class): {e}")
            run_evaluation = False
    
    if not run_evaluation:
        print("Training final model on all data without evaluation split.")
        X_train, y_train = X_scaled, y  # Use all data for training
    
    # Define the model
    if model_type == 'random_forest':
        base_model = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1,
            min_samples_leaf=config.MIN_SAMPLES_LEAF
        )
        
        # Grid search parameters if enabled
        if use_grid_search:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
    
    elif model_type == 'gradient_boosting':
        base_model = GradientBoostingClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            learning_rate=0.1,
            max_depth=3
        )
        
        # Grid search parameters if enabled
        if use_grid_search:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
    
    else:
        print(f"Unknown model type: {model_type}. Using RandomForest as default.")
        base_model = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1,
            min_samples_leaf=config.MIN_SAMPLES_LEAF
        )
    
    # Use grid search if enabled
    if use_grid_search:
        print("Performing grid search for hyperparameter tuning...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print("Best parameters found:")
        print(grid_search.best_params_)
        
        final_model = grid_search.best_estimator_
    else:
        # Train the model directly
        final_model = base_model
        final_model.fit(X_train, y_train)
    
    # Evaluate model if split was possible and successful
    if run_evaluation and X_test is not None and y_test is not None:
        print("Evaluating model on test set...")
        y_pred = final_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Specify zero_division=0 for report
        report = classification_report(y_test, y_pred, zero_division=0)
        
        print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Calculate ROC curve and AUC for probabilistic predictions
        if hasattr(final_model, "predict_proba"):
            try:
                y_prob = final_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                print(f"ROC AUC: {roc_auc:.4f}")
                
                # Plot ROC curve
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                
                # Save ROC curve
                roc_curve_file = os.path.join(config.MODELS_DIR, 'roc_curve.png')
                plt.savefig(roc_curve_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"ROC curve saved to: {roc_curve_file}")
            except Exception as e:
                print(f"Error generating ROC curve: {e}")
    
    # Check feature importance
    feature_importance_dict = {}
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        
        # Create dictionary mapping feature names to importance values
        feature_importance_dict = {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
        
        # Also compute permutation importance if evaluation data is available
        if run_evaluation and X_test is not None and y_test is not None:
            try:
                print("\nCalculating permutation importances for more robust feature ranking...")
                perm_importance = permutation_importance(
                    final_model, X_test, y_test, n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1
                )
                
                # Update with permutation importances (more reliable than model's built-in importances)
                perm_importance_dict = {
                    feature_names[i]: float(perm_importance.importances_mean[i]) 
                    for i in range(len(feature_names))
                }
                
                # Save permutation importance
                permutation_file = os.path.join(config.MODELS_DIR, 'permutation_importance.json')
                with open(permutation_file, 'w') as f:
                    json.dump(perm_importance_dict, f, indent=2)
                print(f"Permutation importance saved to: {permutation_file}")
                
                # Create combined feature importance visualization
                features.visualize_feature_importance(
                    final_model, feature_names, 
                    output_file=os.path.join(config.MODELS_DIR, 'feature_importance.png')
                )
            except Exception as e:
                print(f"Error calculating permutation importance: {e}")
    
    return final_model, scaler, feature_importance_dict

def save_model_and_metadata(model, scaler, feature_importance_dict, feature_names, all_models=None, model_results=None):
    """
    Save the trained model, scaler, and associated metadata.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_importance_dict (dict): Dictionary of feature importances
        feature_names (list): List of feature names
        all_models (dict, optional): Dictionary of all trained models
        model_results (pd.DataFrame, optional): Results of model evaluation
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create model directory if it doesn't exist
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        # Save model
        joblib.dump(model, config.DEFAULT_MODEL_FILE)
        print(f"Model saved to: {config.DEFAULT_MODEL_FILE}")
        
        # Save scaler
        joblib.dump(scaler, config.DEFAULT_SCALER_FILE)
        print(f"Feature scaler saved to: {config.DEFAULT_SCALER_FILE}")
        
        # Save feature importance
        with open(config.DEFAULT_FEATURE_IMPORTANCE_FILE, 'w') as f:
            json.dump(feature_importance_dict, f, indent=2)
        print(f"Feature importance saved to: {config.DEFAULT_FEATURE_IMPORTANCE_FILE}")
        
        # Save feature names
        feature_names_file = os.path.join(config.MODELS_DIR, 'feature_names.json')
        with open(feature_names_file, 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"Feature names saved to: {feature_names_file}")
        
        # Save all models if provided
        if all_models is not None:
            for model_name, model_obj in all_models.items():
                model_file = os.path.join(config.MODELS_DIR, f"{model_name.replace(' ', '_').lower()}_model.pkl")
                joblib.dump(model_obj, model_file)
                print(f"Saved {model_name} model to: {model_file}")
        
        # Save model results if provided
        if model_results is not None and not model_results.empty:
            results_file = os.path.join(config.MODELS_DIR, 'model_comparison_results.csv')
            model_results.to_csv(results_file, index=False)
            print(f"Model comparison results saved to: {results_file}")
        
        return True
    except Exception as e:
        print(f"Error saving model and metadata: {e}")
        return False

def load_model_and_metadata():
    """
    Load the trained model, scaler, and associated metadata.
    
    Returns:
        tuple: (model, scaler, feature_names) or (None, None, None) if error
    """
    try:
        if not os.path.exists(config.DEFAULT_MODEL_FILE):
            print(f"Error: Model file not found at {config.DEFAULT_MODEL_FILE}")
            return None, None, None
        
        if not os.path.exists(config.DEFAULT_SCALER_FILE):
            print(f"Error: Scaler file not found at {config.DEFAULT_SCALER_FILE}")
            return None, None, None
        
        # Load model
        model = joblib.load(config.DEFAULT_MODEL_FILE)
        print(f"Model loaded from: {config.DEFAULT_MODEL_FILE}")
        
        # Load scaler
        scaler = joblib.load(config.DEFAULT_SCALER_FILE)
        print(f"Scaler loaded from: {config.DEFAULT_SCALER_FILE}")
        
        # Load feature names if available
        feature_names_file = os.path.join(config.MODELS_DIR, 'feature_names.json')
        if os.path.exists(feature_names_file):
            with open(feature_names_file, 'r') as f:
                feature_names = json.load(f)
            print(f"Feature names loaded from: {feature_names_file}")
        else:
            feature_names = None
            print("Feature names file not found.")
        
        return model, scaler, feature_names
    
    except Exception as e:
        print(f"Error loading model and metadata: {e}")
        return None, None, None

def run_training_workflow(use_grid_search=False, model_type='random_forest', use_multiple_models=False):
    """
    Run the complete model training workflow with enhanced area features.
    
    Args:
        use_grid_search (bool): Whether to use grid search for hyperparameter tuning
        model_type (str): Type of model to train
        use_multiple_models (bool): Whether to train and evaluate multiple models
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*80)
    print(" TRAINING ENHANCED SAND MINING DETECTION MODEL")
    print(" (With Area-Specific Feature Extraction)")
    print("="*80 + "\n")
    
    # 1. Load labels
    labels = load_labels()
    if not labels:
        print("No labels found. Please label images first.")
        return False
    
    # Count labeled images
    labeled_count = sum(1 for label in labels.values() if label != -1)
    if labeled_count < 10:
        print(f"Only {labeled_count} labeled images found. Please label at least 10 images.")
        return False
    
    print(f"Found {labeled_count} labeled images for training.")
    
    # 2. Create DataFrame from labels
    labels_df = pd.DataFrame(list(labels.items()), columns=['filename', 'label'])
    
    # 3. Extract enhanced features (including area-specific features from annotations)
    print("\nExtracting enhanced features from labeled images...")
    print("This includes:")
    print("- Global image features (color, texture, etc.)")
    print("- Area-specific features from highlighted regions")
    print("- Enhanced texture and color analysis")
    
    features_df = features.extract_features_from_df(config.TRAINING_IMAGES_DIR, labels_df)
    
    if features_df.empty:
        print("Failed to extract features from images.")
        return False
    
    print(f"Successfully extracted {len(features_df.columns) - 2} features from {len(features_df)} images")
    
    # 4. Analyze feature correlations
    print("\nAnalyzing feature correlations...")
    features.calculate_feature_correlation(
        features_df, 
        output_file=os.path.join(config.MODELS_DIR, 'enhanced_feature_correlation.png')
    )
    
    # 5. Prepare data for training
    X, y, feature_names = prepare_training_data(features_df)
    
    if X is None or y is None:
        print("Failed to prepare training data.")
        return False
    
    # Show feature breakdown
    area_features = [f for f in feature_names if any(prefix in f for prefix in ['sand_mining_', 'equipment_', 'water_disturbance_', 'num_', 'total_'])]
    global_features = [f for f in feature_names if 'global_' in f]
    
    print(f"\nFeature breakdown:")
    print(f"- Total features: {len(feature_names)}")
    print(f"- Area-specific features: {len(area_features)}")
    print(f"- Global image features: {len(global_features)}")
    print(f"- Other features: {len(feature_names) - len(area_features) - len(global_features)}")
    
    # 6. Train models
    if use_multiple_models:
        # Train multiple models and get the best one
        print("\nTraining multiple model types for comparison...")
        model, model_results, feature_importance, all_models, scaler = train_multiple_models(
            X, y, feature_names, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        # Save model and metadata, including all trained models
        success = save_model_and_metadata(
            model, scaler, feature_importance, feature_names, 
            all_models=all_models, model_results=model_results
        )
    else:
        # Train a single model
        model, scaler, feature_importance = train_model(
            X, y, feature_names, model_type=model_type, use_grid_search=use_grid_search
        )
        # Save model and metadata
        success = save_model_and_metadata(model, scaler, feature_importance, feature_names)
    
    if success:
        print("\n✅ Enhanced training workflow completed successfully!")
        print("\nNext steps:")
        print("1. Test the model with probability mapping")
        print("2. Use area annotations to improve model performance")
        print("3. The model now focuses on highlighted sand mining areas")
        return True
    else:
        print("\n❌ Enhanced training workflow failed or was incomplete.")
        return False