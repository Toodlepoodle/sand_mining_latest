#!/usr/bin/env python3
"""
Single-image sand mining analysis.

Given one arbitrary satellite/aerial image (no shapefile, no river, no lat/lon
required), this module:
  1. Segments the image into superpixels (SLIC).
  2. Scores every superpixel with the existing trained model by treating each
     superpixel crop as its own miniature "image" and computing the same
     global spectral/texture/GLCM feature set (+ optional DiT deep features)
     that the model was trained on.
  3. Produces:
       - an overall 0-1 sand-mining probability score for the whole image
       - a heatmap overlay PNG highlighting suspected areas
       - a structured list of regions (centroid, bounding box, area, score)

This reuses the exact same model/scaler/feature pipeline as mapper.py and
model.py, so no separate training is needed — it is a new *inference mode*
over the existing trained model, not a new model.

Caveat (documented, not hidden): the model was trained on features computed
over whole training images (with optional area-annotation features that are
unavailable for an arbitrary unseen image, and so are zero-filled here). This
means the per-superpixel scores are an approximation — each superpixel is
scored using the model's response to that crop's own global-style features —
which surfaces relative regions of concern within the image rather than a
pixel-perfect segmentation mask. This is stated explicitly here and should be
stated in the JOSS paper / README as a known limitation.
"""

import os
import json
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.segmentation import slic, mark_boundaries

from src import config
from src.model import load_model_and_metadata
from src.features import (
    extract_basic_features, extract_texture_features, extract_advanced_features,
    convert_to_uint8,
)


def _build_feature_vector(all_features, feature_names):
    """Align an arbitrary features dict to the model's expected column order,
    filling any missing (e.g. area-annotation) columns with 0."""
    if not feature_names:
        return list(all_features.values())
    return [all_features.get(name, 0) for name in feature_names]


def _crop_features(crop_array, use_dit=False):
    """Compute the same 'global_*' feature set used at training time, but
    over a cropped superpixel region instead of a whole training image."""
    tmp_img = Image.fromarray(convert_to_uint8(crop_array))

    feats = {}
    # extract_basic_features / extract_texture_features / extract_advanced_features
    # all take an image *path*, so write the crop to a small temp file.
    tmp_path = os.path.join(config.TEMP_DIR, f'_crop_tmp_{os.getpid()}.png')
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    tmp_img.save(tmp_path)
    try:
        feats.update(extract_basic_features(tmp_path))
        feats.update(extract_texture_features(tmp_path))
        feats.update(extract_advanced_features(tmp_path))

        if use_dit:
            from src.dit_features import extract_dit_features
            feats.update(extract_dit_features(tmp_path))

        # Zero-fill area-annotation-derived columns the model may expect —
        # this image has no drawn annotations, so those features are unknown.
        for t in ['sand_mining', 'equipment', 'water_disturbance', 'no_mining']:
            feats.setdefault(f'num_{t}_areas', 0)
            feats.setdefault(f'total_{t}_area', 0)
            feats.setdefault(f'{t}_area_ratio', 0.0)
        for k in ['NDVI_trend', 'NDWI_trend', 'MNDWI_trend', 'BSI_trend', 'hist_periods']:
            feats.setdefault(k, 0.0)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    for k, v in feats.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            feats[k] = 0.0
    return feats


def analyze_image(image_path, model=None, scaler=None, feature_names=None,
                  n_segments=None, compactness=None, use_dit=None,
                  output_dir=None):
    """
    Run full single-image sand mining analysis.

    Args:
        image_path (str): Path to an arbitrary input image (PNG/JPG/etc).
        model, scaler, feature_names: pass an already-loaded model/scaler
            triple to avoid reloading from disk on repeated calls; if None,
            loads via model.load_model_and_metadata().
        n_segments (int): number of SLIC superpixels (default config.SLIC_N_SEGMENTS).
        compactness (float): SLIC compactness (default config.SLIC_COMPACTNESS).
        use_dit (bool): include DiT deep features (default config.USE_DIT_FEATURES).
        output_dir (str): where to write the heatmap PNG + region JSON
            (default config.SINGLE_IMAGE_OUTPUT_DIR).

    Returns:
        dict: {
            'overall_score': float (0-1),
            'classification': str,
            'regions': [ {region_id, score, bbox, centroid, area_pixels}, ... ],
            'heatmap_path': str,
            'regions_json_path': str,
        }
        or None on failure.
    """
    if model is None or scaler is None:
        model, scaler, feature_names = load_model_and_metadata()
        if model is None:
            print("No trained model found. Run --mode train first.")
            return None

    n_segments = n_segments or getattr(config, 'SLIC_N_SEGMENTS', 60)
    compactness = compactness or getattr(config, 'SLIC_COMPACTNESS', 12)
    if use_dit is None:
        use_dit = getattr(config, 'USE_DIT_FEATURES', False)
    output_dir = output_dir or config.SINGLE_IMAGE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        return None

    img = Image.open(image_path).convert('RGB')
    img_arr = np.array(img)
    H, W = img_arr.shape[:2]

    # ── Overall whole-image score (same feature set the model was trained on) ──
    whole_feats = _crop_features(img_arr, use_dit=use_dit)
    whole_vec = np.array([_build_feature_vector(whole_feats, feature_names)])
    whole_vec_scaled = scaler.transform(whole_vec)
    overall_score = float(model.predict_proba(whole_vec_scaled)[0][1])

    if overall_score >= 0.65:
        classification = 'Sand Mining Likely'
    elif overall_score >= 0.4:
        classification = 'Possible Sand Mining'
    else:
        classification = 'No Sand Mining Likely'

    # ── Superpixel segmentation + per-region scoring ─────────────────────────
    segments = slic(img_arr, n_segments=n_segments, compactness=compactness,
                    start_label=1)

    region_scores = np.zeros_like(segments, dtype=float)
    regions = []

    for seg_id in np.unique(segments):
        mask = segments == seg_id
        if mask.sum() < 9:
            continue

        ys, xs = np.where(mask)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        crop = img_arr[y1:y2, x1:x2]

        try:
            feats = _crop_features(crop, use_dit=use_dit)
            vec = np.array([_build_feature_vector(feats, feature_names)])
            vec_scaled = scaler.transform(vec)
            score = float(model.predict_proba(vec_scaled)[0][1])
        except Exception as e:
            print(f"  Warning: region {seg_id} scoring failed: {e}")
            score = 0.0

        region_scores[mask] = score
        regions.append({
            'region_id': int(seg_id),
            'score': round(score, 4),
            'bbox': [x1, y1, x2, y2],
            'centroid': [float(xs.mean()), float(ys.mean())],
            'area_pixels': int(mask.sum()),
        })

    regions.sort(key=lambda r: r['score'], reverse=True)

    # ── Heatmap overlay ──────────────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    heatmap_path = os.path.join(output_dir, f'{base}_heatmap_{timestamp}.png')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    boundary_img = mark_boundaries(img_arr, segments, color=(1, 1, 1))
    ax1.imshow(boundary_img)
    ax1.set_title('Input Image (superpixel boundaries)')
    ax1.axis('off')

    ax2.imshow(img_arr)
    heat = ax2.imshow(region_scores, cmap='hot', alpha=0.55, vmin=0, vmax=1)
    ax2.set_title(f'Sand Mining Probability Heatmap\nOverall score: {overall_score:.3f} '
                 f'({classification})')
    ax2.axis('off')
    plt.colorbar(heat, ax=ax2, fraction=0.046, pad=0.04, label='Probability')

    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap saved to: {heatmap_path}")

    # ── Structured region list (JSON) ────────────────────────────────────────
    regions_json_path = os.path.join(output_dir, f'{base}_regions_{timestamp}.json')
    output = {
        'image_path': image_path,
        'overall_score': round(overall_score, 4),
        'classification': classification,
        'n_regions': len(regions),
        'regions': regions,
    }
    with open(regions_json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Region scores saved to: {regions_json_path}")

    print(f"\nOverall sand mining probability: {overall_score:.3f} ({classification})")
    if regions:
        print("Top 5 suspected regions:")
        for r in regions[:5]:
            print(f"  region {r['region_id']}: score={r['score']:.3f}, "
                  f"bbox={r['bbox']}, area={r['area_pixels']}px")

    return {
        'overall_score': overall_score,
        'classification': classification,
        'regions': regions,
        'heatmap_path': heatmap_path,
        'regions_json_path': regions_json_path,
    }
