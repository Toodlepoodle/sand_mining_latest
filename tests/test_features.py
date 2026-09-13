"""Tests for src/features.py feature-extraction helpers (no EE dependency)."""
import numpy as np
from PIL import Image

from src.features import (
    convert_to_uint8, polygon_to_mask, build_type_masks, extract_basic_features,
)


def test_convert_to_uint8_from_float_0_1():
    arr = np.array([[0.0, 0.5], [1.0, 0.25]])
    out = convert_to_uint8(arr)
    assert out.dtype == np.uint8
    assert out.max() <= 255


def test_convert_to_uint8_already_uint8_passthrough():
    arr = np.array([[0, 255]], dtype=np.uint8)
    out = convert_to_uint8(arr)
    assert out is arr


def test_polygon_to_mask_valid_triangle():
    poly = [[0, 0], [10, 0], [5, 10]]
    mask = polygon_to_mask(poly, height=20, width=20)
    assert mask is not None
    assert mask.dtype == bool
    assert mask.sum() > 0


def test_polygon_to_mask_rejects_degenerate():
    assert polygon_to_mask([[0, 0], [1, 1]], height=10, width=10) is None
    assert polygon_to_mask(None, height=10, width=10) is None


def test_build_type_masks_pools_same_type_regions():
    annotations = [
        {'type': 'sand_mining', 'polygon': [[0, 0], [5, 0], [5, 5], [0, 5]]},
        {'type': 'sand_mining', 'polygon': [[10, 10], [15, 10], [15, 15], [10, 15]]},
        {'type': 'equipment', 'bbox': [1, 1, 3, 3]},
    ]
    masks = build_type_masks(annotations, height=20, width=20)
    assert set(masks.keys()) == {'sand_mining', 'equipment'}
    # Both sand_mining polygons should be OR-ed into a single pooled mask
    assert masks['sand_mining'].sum() > masks['equipment'].sum()


def test_extract_basic_features_returns_expected_keys(tmp_path):
    img_path = tmp_path / "test.png"
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(img_path)

    feats = extract_basic_features(str(img_path))
    for key in ['global_red_mean', 'global_green_mean', 'global_blue_mean',
               'global_gray_mean', 'global_gray_std']:
        assert key in feats


def test_extract_basic_features_missing_file_returns_empty():
    feats = extract_basic_features('/nonexistent/path.png')
    assert feats == {}
