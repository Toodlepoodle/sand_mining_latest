#!/usr/bin/env python3
"""
Fusion module: combines Layer 1 (unsupervised), Layer 2 (weak supervision),
and Layer 3 (supervised ML) into a single calibrated probability.
Integrates with existing mapper.py output.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from src import config
from src.weight_tuner import get_or_optimize_weights


def fuse_probabilities(l1_series, l2_series, l3_series, weights=None):
    """
    Weighted linear combination of three probability layers.

    Args:
        l1_series: pd.Series, Layer 1 unsupervised probs
        l2_series: pd.Series, Layer 2 weak supervised probs
        l3_series: pd.Series, Layer 3 ML probs
        weights: [w1, w2, w3] or None (auto-load)

    Returns:
        pd.Series: fused probabilities (0-1)
    """
    if weights is None:
        weights = get_or_optimize_weights()

    w1, w2, w3 = weights
    fused = w1 * l1_series + w2 * l2_series + w3 * l3_series
    return fused.clip(0, 1)


def classify(prob, low=0.4, high=0.65):
    """Convert probability to classification string."""
    if prob >= high:
        return 'Sand Mining Likely'
    elif prob >= low:
        return 'Possible Sand Mining'
    else:
        return 'No Sand Mining Likely'


def build_fused_results(coords, l1_probs, l2_probs, l3_results_df, weights=None):
    """
    Build final results DataFrame combining all three layers.

    Args:
        coords: list of (lat, lon)
        l1_probs: pd.Series from unsupervised layer
        l2_probs: pd.Series from weak supervision layer
        l3_results_df: existing mapper.py output DataFrame
        weights: [w1, w2, w3] or None

    Returns:
        pd.DataFrame with full results + fused probability
    """
    if weights is None:
        weights = get_or_optimize_weights()

    w1, w2, w3 = weights
    print(f"\n[Fusion] Combining layers: "
          f"L1(unsupervised)×{w1:.2f} + "
          f"L2(scraped)×{w2:.2f} + "
          f"L3(ML)×{w3:.2f}")

    df = l3_results_df.copy()

    # Align series to df index
    l1 = l1_probs.values[:len(df)] if len(l1_probs) >= len(df) else np.full(len(df), 0.5)
    l2 = l2_probs.values[:len(df)] if len(l2_probs) >= len(df) else np.full(len(df), 0.5)
    l3 = df['probability'].values

    fused = np.clip(w1*l1 + w2*l2 + w3*l3, 0, 1)

    df['prob_unsupervised'] = l1
    df['prob_scraped']      = l2
    df['prob_ml']           = l3
    df['probability']       = fused  # overwrite with fused
    df['classification']    = [classify(p) for p in fused]
    df['fusion_weights']    = f"L1={w1:.2f},L2={w2:.2f},L3={w3:.2f}"

    print(f"[Fusion] Done. "
          f"Sand Mining Likely: {(df['classification']=='Sand Mining Likely').sum()} | "
          f"Possible: {(df['classification']=='Possible Sand Mining').sum()} | "
          f"No Mining: {(df['classification']=='No Sand Mining Likely').sum()}")

    return df


def save_fused_results(df, shapefile_path):
    """Save fused results CSV."""
    date_str = datetime.now().strftime('%Y%m%d')
    base     = os.path.splitext(os.path.basename(shapefile_path))[0]
    out_path = os.path.join(
        config.PROBABILITY_MAPS_DIR,
        f'fused_sand_mining_{base}_{date_str}.csv'
    )
    df.to_csv(out_path, index=False)
    print(f"[Fusion] Saved fused results to: {out_path}")
    return out_path