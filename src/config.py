#!/usr/bin/env python3
"""
Configuration settings for the Enhanced Sand Mining Detection Tool with Area Highlighting.
"""

import os
from datetime import datetime, timedelta

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Input data paths
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Output paths
TRAINING_IMAGES_DIR = os.path.join(OUTPUT_DIR, 'training_images')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
PROBABILITY_MAPS_DIR = os.path.join(OUTPUT_DIR, 'probability_maps')
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, 'annotations')
TEMP_DIR = os.path.join(OUTPUT_DIR, 'temp')

# ── Historical / temporal output paths ───────────────────────────────────────
HISTORICAL_DIR = os.path.join(OUTPUT_DIR, 'historical')
TIME_SERIES_DIR = os.path.join(HISTORICAL_DIR, 'time_series')
CHANGE_MAPS_DIR = os.path.join(HISTORICAL_DIR, 'change_maps')
TEMPORAL_IMAGES_DIR = os.path.join(HISTORICAL_DIR, 'images')
SSC_MAPS_DIR = os.path.join(HISTORICAL_DIR, 'ssc_maps')
SEASONAL_DIR = os.path.join(HISTORICAL_DIR, 'seasonal')
# ─────────────────────────────────────────────────────────────────────────────

# Default file paths
DEFAULT_MODEL_FILE = os.path.join(MODELS_DIR, 'sand_mining_model.joblib')
DEFAULT_SCALER_FILE = os.path.join(MODELS_DIR, 'feature_scaler.joblib')
DEFAULT_FEATURE_IMPORTANCE_FILE = os.path.join(MODELS_DIR, 'feature_importance.json')
LABELS_FILE = os.path.join(OUTPUT_DIR, 'training_labels.csv')
ANNOTATIONS_FILE = os.path.join(ANNOTATIONS_DIR, 'area_annotations.json')

# ── Historical results files ──────────────────────────────────────────────────
TEMPORAL_RESULTS_FILE = os.path.join(HISTORICAL_DIR, 'temporal_results.json')
CHANGE_REPORT_FILE = os.path.join(HISTORICAL_DIR, 'change_detection_report.csv')
SEASONAL_STATS_FILE = os.path.join(SEASONAL_DIR, 'seasonal_statistics.csv')
# ─────────────────────────────────────────────────────────────────────────────

# Earth Engine settings
EE_HIGH_VOLUME_URL = 'https://earthengine-highvolume.googleapis.com'

# Image parameters
DEFAULT_IMAGE_DIM = 1024
DEFAULT_MAP_IMAGE_DIM = 1024
DEFAULT_BUFFER_METERS = 1500

# ── Satellite collections ─────────────────────────────────────────────────────
# Sentinel-2 Harmonized (available from ~2017-03)
S2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
S2_BAND_NAMES = ['Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3',
                 'NIR', 'NarrowNIR', 'SWIR1', 'SWIR2']
S2_START_YEAR = 2017

# Landsat 8/9 (OLI)
L89_COLLECTION = 'LANDSAT/LC09/C02/T1_L2'
L8_COLLECTION  = 'LANDSAT/LC08/C02/T1_L2'
L89_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
L89_BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
L8_START_YEAR = 2013

# Landsat 4-7 (TM/ETM+)
L457_COLLECTION = 'LANDSAT/LE07/C02/T1_L2'
L5_COLLECTION   = 'LANDSAT/LT05/C02/T1_L2'
L457_BANDS = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
L457_BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
L457_START_YEAR = 1984

# VIIRS
VIIRS_COLLECTION = 'NOAA/VIIRS/001/VNP09GA'
VIIRS_BANDS = ['M3', 'M4', 'M5', 'M7', 'M8', 'M10', 'M11']
VIIRS_BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'NIR2', 'SWIR1', 'SWIR2']
VIIRS_START_YEAR = 2012
# ─────────────────────────────────────────────────────────────────────────────

# Spectral indices to compute
SPECTRAL_INDICES = {
    'NDVI':  ('NIR', 'Red'),
    'NDWI':  ('Green', 'NIR'),
    'MNDWI': ('Green', 'SWIR1'),
    'BSI':   ('SWIR1', 'Red', 'NIR', 'Blue'),
    'NDBI':  ('SWIR1', 'NIR'),
    'NDTI':  ('SWIR1', 'SWIR2'),
}

# Sampling and mapping settings
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_DISTANCE_KM = 0.2
MIN_CLOUD_COVER = 35

# Machine learning settings
TEST_SIZE = 0.25
RANDOM_STATE = 42
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 2
ENABLE_MULTIPLE_MODELS = True

# Area annotation settings
ANNOTATION_COLORS = {
    'sand_mining':       (255, 0,   0,   128),
    'no_mining':         (0,   255, 0,   128),
    'equipment':         (0,   0,   255, 128),
    'water_disturbance': (255, 255, 0,   128),
}

# ── Historical / temporal analysis settings ───────────────────────────────────
HISTORICAL_START_YEAR = 2017
HISTORICAL_END_YEAR = datetime.now().year

# Time interval between imagery samples
# Options: 'monthly', 'quarterly', 'biannual', 'annual'
HISTORICAL_INTERVAL = 'quarterly'

# Mapping of interval name → approximate days
INTERVAL_DAYS = {
    'monthly':   30,
    'quarterly': 91,
    'biannual':  182,
    'annual':    365,
}

# How many years back to look for the mapper (used by mapper.py)
# NOTE: This is overridden at runtime by --years-back CLI argument
HISTORICAL_YEARS_BACK = 3
HISTORICAL_INTERVAL_MONTHS = 3

# Seasonal definitions (month ranges, inclusive)
SEASONS = {
    'dry':        (11, 4),
    'monsoon':    (6,  9),
    'transition': (5,  5),
}

# Change-detection thresholds
CHANGE_DETECTION_THRESHOLD = 0.30
CONFIRMATION_PERIODS = 2
HIGH_PROB_THRESHOLD = 0.60

# SSC anomaly threshold
SSC_ANOMALY_ZSCORE = 2.0

# Maximum number of historical periods to keep in memory per location
MAX_PERIODS_PER_LOCATION = 40

# Whether to save individual historical images to disk
SAVE_HISTORICAL_IMAGES = False
# ─────────────────────────────────────────────────────────────────────────────

# Model files for different algorithms
RF_MODEL_FILE   = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
GB_MODEL_FILE   = os.path.join(MODELS_DIR, 'gradient_boosting_model.pkl')
XGB_MODEL_FILE  = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
LGBM_MODEL_FILE = os.path.join(MODELS_DIR, 'lightgbm_model.pkl')
SVM_MODEL_FILE  = os.path.join(MODELS_DIR, 'svm_model.pkl')
LR_MODEL_FILE   = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')

# Model visualization directory
MODEL_VIZ_DIR = os.path.join(MODELS_DIR, 'visualizations')

# ── Create full directory structure ──────────────────────────────────────────
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    OUTPUT_DIR, TRAINING_IMAGES_DIR, MODELS_DIR, PROBABILITY_MAPS_DIR,
    ANNOTATIONS_DIR, TEMP_DIR, MODEL_VIZ_DIR,
    # Historical directories
    HISTORICAL_DIR, TIME_SERIES_DIR, CHANGE_MAPS_DIR,
    TEMPORAL_IMAGES_DIR, SSC_MAPS_DIR, SEASONAL_DIR,
]:
    os.makedirs(directory, exist_ok=True)