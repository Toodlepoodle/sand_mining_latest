#!/usr/bin/env python3
"""
Configuration settings for the Sand Mining Detection Tool.
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
TEMP_DIR = os.path.join(OUTPUT_DIR, 'temp')

# Default file paths
DEFAULT_MODEL_FILE = os.path.join(MODELS_DIR, 'sand_mining_model.joblib')
DEFAULT_SCALER_FILE = os.path.join(MODELS_DIR, 'feature_scaler.joblib')
DEFAULT_FEATURE_IMPORTANCE_FILE = os.path.join(MODELS_DIR, 'feature_importance.json')
LABELS_FILE = os.path.join(OUTPUT_DIR, 'training_labels.csv')

# Earth Engine settings
EE_HIGH_VOLUME_URL = 'https://earthengine-highvolume.googleapis.com'

# Image parameters
DEFAULT_IMAGE_DIM = 512  # Dimension for downloaded training images
DEFAULT_MAP_IMAGE_DIM = 800  # Dimension for images used in mapping
DEFAULT_BUFFER_METERS = 1500  # Area around points for image download

# Satellite data settings
# Historical lookback duration for different sensors
SENTINEL_LOOKBACK_YEARS = 5  # Sentinel-2 data from ~2015
LANDSAT_LOOKBACK_YEARS = 40  # Landsat archive goes back to 1980s
VIIRS_LOOKBACK_YEARS = 10  # VIIRS from ~2011

# Bands and collections to use
# Sentinel-2 Harmonized
S2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
S2_BAND_NAMES = ['Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'NarrowNIR', 'SWIR1', 'SWIR2']

# Landsat 8/9 (OLI)
L89_COLLECTION = 'LANDSAT/LC09/C02/T1_L2'
L89_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
L89_BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

# Landsat 4-7 (TM/ETM+)
L457_COLLECTION = 'LANDSAT/LE07/C02/T1_L2'
L457_BANDS = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
L457_BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

# VIIRS
VIIRS_COLLECTION = 'NOAA/VIIRS/001/VNP09GA'
VIIRS_BANDS = ['M3', 'M4', 'M5', 'M7', 'M8', 'M10', 'M11']
VIIRS_BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR', 'NIR2', 'SWIR1', 'SWIR2']

# Spectral indices to compute
SPECTRAL_INDICES = {
    'NDVI': ('NIR', 'Red'),        # Normalized Difference Vegetation Index
    'NDWI': ('Green', 'NIR'),      # Normalized Difference Water Index
    'MNDWI': ('Green', 'SWIR1'),   # Modified NDWI
    'BSI': ('SWIR1', 'Red', 'NIR', 'Blue'),  # Bare Soil Index
    'NDBI': ('SWIR1', 'NIR'),      # Normalized Difference Built-up Index
    'NDTI': ('SWIR1', 'SWIR2'),    # Normalized Difference Turbidity Index
}

# Sampling and mapping settings
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_DISTANCE_KM = 0.2  # Changed from 1.0 to 0.2 for better coverage
MIN_CLOUD_COVER = 35  # Maximum acceptable cloud percentage

# Machine learning settings
TEST_SIZE = 0.25  # Proportion of data to use for testing
RANDOM_STATE = 42  # Random seed for reproducibility
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 2
ENABLE_MULTIPLE_MODELS = True  # Whether to train multiple models by default

# Historical imagery parameters
HISTORICAL_YEARS_BACK = 5  # Number of years to look back for historical imagery
HISTORICAL_INTERVAL_MONTHS = 6  # Frequency of historical imagery (smaller = more frequent)

# Model files for different algorithms
RF_MODEL_FILE = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
GB_MODEL_FILE = os.path.join(MODELS_DIR, 'gradient_boosting_model.pkl')
XGB_MODEL_FILE = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
LGBM_MODEL_FILE = os.path.join(MODELS_DIR, 'lightgbm_model.pkl') 
SVM_MODEL_FILE = os.path.join(MODELS_DIR, 'svm_model.pkl')
LR_MODEL_FILE = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')

# Model visualization directory
MODEL_VIZ_DIR = os.path.join(MODELS_DIR, 'visualizations')

# Create directory structure
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    OUTPUT_DIR, TRAINING_IMAGES_DIR, MODELS_DIR, PROBABILITY_MAPS_DIR, TEMP_DIR, MODEL_VIZ_DIR
]:
    os.makedirs(directory, exist_ok=True)