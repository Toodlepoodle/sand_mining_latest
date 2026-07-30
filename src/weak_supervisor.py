#!/usr/bin/env python3
"""
Layer 2: Weakly supervised detection using scraped ground truth.
Uses distance to known mining sites + historical trend similarity.
No manual labeling required.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

from src import config


def compute_spatial_proximity_score(lat, lon, scraped_df, radius_km=5.0):
    """
    Distance-based score: how close is this point to a known mining site?
    Uses inverse-distance weighting within radius_km.

    Args:
        lat, lon: query point
        scraped_df: DataFrame with columns lat, lon, label
        radius_km: influence radius in km

    Returns:
        float: proximity score (0-1)
    """
    if scraped_df is None or len(scraped_df) == 0:
        return 0.5  # unknown

    mining_df = scraped_df[scraped_df['label'] == 1]
    if len(mining_df) == 0:
        return 0.5

    # Convert to radians for haversine
    lat_r  = np.radians(lat)
    lon_r  = np.radians(lon)
    lats_r = np.radians(mining_df['lat'].values)
    lons_r = np.radians(mining_df['lon'].values)

    # Haversine distances in km
    dlat = lats_r - lat_r
    dlon = lons_r - lon_r
    a    = np.sin(dlat/2)**2 + np.cos(lat_r)*np.cos(lats_r)*np.sin(dlon/2)**2
    dists_km = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # Gaussian kernel within radius
    sigma  = radius_km / 2
    within = dists_km[dists_km <= radius_km]

    if len(within) == 0:
        return 0.0

    weights = norm.pdf(within, 0, sigma)
    score   = float(np.sum(weights) / (np.sum(weights) + 1.0))
    return float(np.clip(score, 0, 1))


def run_weak_supervision_layer(coordinates, scraped_df, radius_km=10.0):
    """
    Run Layer 2 on a list of coordinates.

    Args:
        coordinates: list of (lat, lon) tuples
        scraped_df: output from scraper.run_scraper()
        radius_km: spatial influence radius

    Returns:
        pd.Series: proximity scores (0-1) per coordinate
    """
    print(f"\n[Layer 2] Weak supervision on {len(coordinates)} points "
          f"using {len(scraped_df) if scraped_df is not None else 0} scraped locations...")

    if scraped_df is None or len(scraped_df) == 0:
        print("  No scraped data — Layer 2 returning 0.5 (uncertain) for all points")
        return pd.Series([0.5] * len(coordinates))

    mining_locs = scraped_df[scraped_df['label'] == 1][['lat','lon']].values

    if len(mining_locs) == 0:
        return pd.Series([0.5] * len(coordinates))

    # Use KD-tree for fast nearest-neighbour lookup
    # Convert to approximate Cartesian (good enough for India's lat range)
    def to_xy(lats, lons):
        lat_c = np.radians(np.mean(lats))
        x = np.radians(lons) * np.cos(lat_c) * 6371
        y = np.radians(lats) * 6371
        return np.column_stack([x, y])

    all_lats = np.array([c[0] for c in coordinates])
    all_lons = np.array([c[1] for c in coordinates])

    query_xy  = to_xy(all_lats, all_lons)
    mining_xy = to_xy(mining_locs[:,0], mining_locs[:,1])

    tree = cKDTree(mining_xy)

    # Find all mining sites within radius for each point
    radius_approx = radius_km  # km, approximate
    indices = tree.query_ball_point(query_xy, r=radius_approx)

    scores = []
    for i, idx_list in enumerate(indices):
        if len(idx_list) == 0:
            scores.append(0.0)
        else:
            # Distance to each nearby mining site
            dists = np.linalg.norm(
                query_xy[i] - mining_xy[idx_list], axis=1
            )
            sigma   = radius_approx / 2
            weights = norm.pdf(dists, 0, sigma)
            score   = float(np.sum(weights) / (np.sum(weights) + 1.0))
            scores.append(float(np.clip(score, 0, 1)))

    result = pd.Series(scores)
    print(f"  Layer 2 complete. Mean score: {result.mean():.3f}, "
          f"Points with score>0: {(result>0).sum()}")
    return result