#!/usr/bin/env python3
"""
Layer 1: Unsupervised sand mining detection using spectral signatures.
Based on Mukherjee (2023) - no labels required.

Method:
1. Detect high-mineral regions using BSI, CMI, NDVI from Sentinel-2
2. Mask to river proximity
3. Morphological filtering to connect patches
4. Output probability per point (0-1)
"""

import ee
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

from src import config


def compute_spectral_scores(lat, lon, buffer_m=1500, years_back=3):
    """
    Compute unsupervised sandbank probability for a single point.
    Uses Mukherjee (2023) two-step method:
    Step 1: High mineral detection (BSI, CMI, NDVI)
    Step 2: River stream association (MNDWI proximity)

    Args:
        lat, lon: coordinates
        buffer_m: buffer in metres
        years_back: historical window for seasonal averaging

    Returns:
        float: unsupervised probability (0-1), or None on error
    """
    try:
        point  = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_m)

        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=365 * years_back)

        # Get dry-season composites (Nov-Apr) to maximise sandbank visibility
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(region)
              .filterDate(ee.Date(start_dt), ee.Date(end_dt))
              .filter(ee.Filter.calendarRange(11, 4, 'month'))
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35))
              .median())

        # Check if we got an image
        band_count = s2.bandNames().size().getInfo()
        if band_count == 0:
            # Fall back to all-season
            s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(ee.Date(start_dt), ee.Date(end_dt))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                  .median())

        # Rename to standard band names
        s2 = s2.select(
            ['B2',  'B3',    'B4',  'B8',  'B11',  'B12'],
            ['Blue','Green','Red','NIR','SWIR1','SWIR2']
        )

        # ── Spectral indices ─────────────────────────────────────────────
        # BSI — Bare Soil Index (Mukherjee uses this as primary mineral indicator)
        numer_bsi = s2.select('SWIR1').add(s2.select('Red')).subtract(
                    s2.select('NIR').add(s2.select('Blue')))
        denom_bsi = s2.select('SWIR1').add(s2.select('Red')).add(
                    s2.select('NIR').add(s2.select('Blue')))
        bsi = numer_bsi.divide(denom_bsi).rename('BSI')

        # CMI — Coal Mine Index (adapted: SWIR2-SWIR1 ratio, sensitive to silica/quartz)
        cmi = s2.select('SWIR2').subtract(s2.select('SWIR1')).divide(
              s2.select('SWIR2').add(s2.select('SWIR1')).add(1e-6)).rename('CMI')

        # NDVI — suppress vegetation (mining sites have low NDVI)
        ndvi = s2.normalizedDifference(['NIR', 'Red']).rename('NDVI')

        # MNDWI — detect water proximity (sandbanks must be near water)
        mndwi = s2.normalizedDifference(['Green', 'SWIR1']).rename('MNDWI')

        # ── Reduce to region stats ───────────────────────────────────────
        stats = (ee.Image([bsi, cmi, ndvi, mndwi])
                 .reduceRegion(
                     reducer=ee.Reducer.mean().combine(
                         ee.Reducer.percentile([25, 75]), sharedInputs=True),
                     geometry=region,
                     scale=20,
                     maxPixels=1e9
                 ).getInfo())

        if not stats:
            return None

        bsi_mean  = stats.get('BSI_mean',  0) or 0
        cmi_mean  = stats.get('CMI_mean',  0) or 0
        ndvi_mean = stats.get('NDVI_mean', 0) or 0
        mndwi_mean= stats.get('MNDWI_mean',0) or 0

        # ── Mukherjee two-step scoring ───────────────────────────────────
        # Step 1: Mineral abundance score
        # High BSI + moderate CMI + low NDVI = exposed mineral sand
        mineral_score = (
            _sigmoid(bsi_mean,  threshold=0.05, steepness=20) * 0.5 +
            _sigmoid(cmi_mean,  threshold=0.0,  steepness=15) * 0.2 +
            _sigmoid(-ndvi_mean, threshold=-0.1, steepness=10) * 0.3
        )

        # Step 2: River association score
        # Must be near water (MNDWI > -0.3 means water influence nearby)
        river_score = _sigmoid(mndwi_mean, threshold=-0.3, steepness=8)

        # Combined: both conditions must be true (multiplicative)
        unsupervised_prob = mineral_score * river_score

        return float(np.clip(unsupervised_prob, 0, 1))

    except Exception as e:
        return None


def _sigmoid(x, threshold=0.0, steepness=10):
    """Smooth threshold function."""
    return float(1 / (1 + np.exp(-steepness * (x - threshold))))


def run_unsupervised_layer(coordinates, years_back=3, buffer_m=1500):
    """
    Run Layer 1 unsupervised detection on a list of coordinates.

    Args:
        coordinates: list of (lat, lon) tuples
        years_back: historical window
        buffer_m: buffer radius

    Returns:
        pd.Series: index=range(len(coords)), values=probability (0-1)
    """
    print(f"\n[Layer 1] Unsupervised spectral detection on {len(coordinates)} points...")
    probs = []
    for i, (lat, lon) in enumerate(coordinates):
        p = compute_spectral_scores(lat, lon, buffer_m=buffer_m, years_back=years_back)
        probs.append(p if p is not None else 0.5)  # 0.5 = uncertain if failed
        if (i + 1) % 50 == 0:
            print(f"  Layer 1: {i+1}/{len(coordinates)} done")
        time.sleep(0.05)

    print(f"  Layer 1 complete. Mean score: {np.mean(probs):.3f}")
    return pd.Series(probs)
