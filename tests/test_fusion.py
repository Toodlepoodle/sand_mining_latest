"""Tests for src/fusion.py — pure math, no Earth Engine dependency."""
import pandas as pd
import numpy as np
import pytest

from src.fusion import fuse_probabilities, classify, build_fused_results


def test_fuse_probabilities_weighted_average():
    l1 = pd.Series([1.0, 0.0])
    l2 = pd.Series([0.0, 1.0])
    l3 = pd.Series([0.5, 0.5])
    fused = fuse_probabilities(l1, l2, l3, weights=[0.5, 0.3, 0.2])
    assert pytest.approx(fused.iloc[0], abs=1e-9) == 0.5 * 1.0 + 0.3 * 0.0 + 0.2 * 0.5
    assert pytest.approx(fused.iloc[1], abs=1e-9) == 0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 0.5


def test_fuse_probabilities_clips_to_0_1():
    l1 = pd.Series([2.0])
    l2 = pd.Series([2.0])
    l3 = pd.Series([2.0])
    fused = fuse_probabilities(l1, l2, l3, weights=[1, 1, 1])
    assert fused.iloc[0] == 1.0


@pytest.mark.parametrize("prob,expected", [
    (0.9, 'Sand Mining Likely'),
    (0.65, 'Sand Mining Likely'),
    (0.5, 'Possible Sand Mining'),
    (0.4, 'Possible Sand Mining'),
    (0.1, 'No Sand Mining Likely'),
])
def test_classify_thresholds(prob, expected):
    assert classify(prob) == expected


def test_build_fused_results_combines_three_layers():
    coords = [(0, 0), (1, 1)]
    l1 = pd.Series([0.8, 0.2])
    l2 = pd.Series([0.6, 0.1])
    l3_df = pd.DataFrame({'probability': [0.7, 0.3], 'latitude': [0, 1], 'longitude': [0, 1]})
    result = build_fused_results(coords, l1, l2, l3_df, weights=[0.34, 0.33, 0.33])
    assert 'prob_unsupervised' in result.columns
    assert 'prob_scraped' in result.columns
    assert 'prob_ml' in result.columns
    assert 'classification' in result.columns
    assert len(result) == 2
    assert (result['probability'] >= 0).all() and (result['probability'] <= 1).all()
