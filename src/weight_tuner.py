#!/usr/bin/env python3
"""
Automated weight tuner for three-layer fusion.
Uses scraped ground truth + manual labels as validation anchors.
Runs scipy.optimize to find w1, w2, w3 that maximize F1 at known locations.
Saves weights and reloads them on next run — self-improving over time.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, roc_auc_score
from datetime import datetime

from src import config

WEIGHTS_FILE = os.path.join(config.MODELS_DIR, 'fusion_weights.json')


def _load_weights():
    """Load previously optimized weights, or return defaults."""
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE) as f:
                data = json.load(f)
            w = data.get('weights', [0.33, 0.33, 0.34])
            print(f"[Weights] Loaded from file: L1={w[0]:.3f} L2={w[1]:.3f} L3={w[2]:.3f} "
                  f"(optimized {data.get('timestamp','?')})")
            return w
        except Exception:
            pass
    print("[Weights] No saved weights — using equal weights (0.33, 0.33, 0.34)")
    return [0.33, 0.33, 0.34]


def _save_weights(weights, metrics):
    """Persist optimized weights to disk."""
    os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
    data = {
        'weights':   list(weights),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'metrics':   metrics,
    }
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[Weights] Saved optimized weights to {WEIGHTS_FILE}")


def optimize_weights(l1_probs, l2_probs, l3_probs, true_labels,
                     min_samples=10):
    """
    Find optimal w1, w2, w3 to maximize F1 at validation points.

    Args:
        l1_probs: array of Layer 1 probabilities at validation points
        l2_probs: array of Layer 2 probabilities at validation points
        l3_probs: array of Layer 3 probabilities at validation points
        true_labels: array of ground truth (0 or 1)
        min_samples: minimum samples needed to optimize

    Returns:
        list: [w1, w2, w3] optimized weights
    """
    l1 = np.array(l1_probs)
    l2 = np.array(l2_probs)
    l3 = np.array(l3_probs)
    y  = np.array(true_labels)

    # Remove uncertain labels
    valid = (y == 0) | (y == 1)
    l1, l2, l3, y = l1[valid], l2[valid], l3[valid], y[valid]

    if len(y) < min_samples:
        print(f"[Weights] Only {len(y)} validation points — need {min_samples}. "
              f"Using equal weights.")
        return _load_weights()

    print(f"\n[Weights] Optimizing on {len(y)} validation points "
          f"({y.sum()} mining, {(1-y).sum()} non-mining)...")

    def neg_f1(w):
        w = np.array(w)
        w = np.clip(w, 0.01, 0.98)
        w = w / w.sum()  # normalize to sum=1
        combined = w[0]*l1 + w[1]*l2 + w[2]*l3
        pred = (combined >= 0.5).astype(int)
        return -f1_score(y, pred, zero_division=0)

    # Multiple random starts to avoid local minima
    best_w   = [0.33, 0.33, 0.34]
    best_f1  = -neg_f1(best_w)
    best_auc = 0.0

    starts = [
        [0.33, 0.33, 0.34],
        [0.5,  0.3,  0.2],
        [0.2,  0.3,  0.5],
        [0.1,  0.1,  0.8],
        [0.6,  0.2,  0.2],
        [0.2,  0.6,  0.2],
    ]

    for start in starts:
        try:
            result = minimize(
                neg_f1,
                x0=start,
                method='SLSQP',
                bounds=[(0.01, 0.98)] * 3,
                constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1},
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            if result.success:
                w   = np.array(result.x)
                w   = np.clip(w, 0.01, 0.98)
                w   = w / w.sum()
                f1v = -result.fun
                if f1v > best_f1:
                    best_f1 = f1v
                    best_w  = list(w)
        except Exception:
            continue

    # Compute final metrics
    best_w = np.array(best_w)
    best_w = best_w / best_w.sum()
    combined = best_w[0]*l1 + best_w[1]*l2 + best_w[2]*l3
    pred     = (combined >= 0.5).astype(int)
    final_f1 = f1_score(y, pred, zero_division=0)
    try:
        final_auc = roc_auc_score(y, combined)
    except Exception:
        final_auc = 0.0

    metrics = {
        'f1':           round(float(final_f1), 4),
        'auc':          round(float(final_auc), 4),
        'n_validation': int(len(y)),
        'n_mining':     int(y.sum()),
    }

    print(f"[Weights] Optimized: L1={best_w[0]:.3f} L2={best_w[1]:.3f} L3={best_w[2]:.3f}")
    print(f"[Weights] Validation F1={final_f1:.3f}  AUC={final_auc:.3f}")

    _save_weights(list(best_w), metrics)
    return list(best_w)


def get_or_optimize_weights(l1_probs=None, l2_probs=None, l3_probs=None,
                             true_labels=None, force_reoptimize=False):
    """
    Main entry point:
    - If validation data provided and force_reoptimize=True → optimize
    - Otherwise load saved weights
    - Fall back to equal weights if nothing available

    Returns:
        list: [w1, w2, w3]
    """
    if (force_reoptimize and
            l1_probs is not None and
            l2_probs is not None and
            l3_probs is not None and
            true_labels is not None):
        return optimize_weights(l1_probs, l2_probs, l3_probs, true_labels)
    else:
        return _load_weights()
