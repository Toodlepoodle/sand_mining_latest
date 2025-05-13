#!/usr/bin/env python3
"""
Feature extraction module for the Sand Mining Detection Tool.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
import joblib
from scipy import stats
import ee
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

from src import config
from src import ee_utils

def extract_basic_features(image_path):
    """
    Extract basic color and statistical features from an image.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: Dictionary of extracted features
    """
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return {}
        
        # Open image and convert to array
        img = Image.open(image_path).convert('RGB')
        img_arr = np.array(img)
        
        # Basic Color Stats
        mean_rgb = np.mean(img_arr, axis=(0, 1))
        std_rgb = np.std(img_arr, axis=(0, 1))
        median_rgb = np.median(img_arr, axis=(0, 1))
        
        # Convert to grayscale for texture analysis
        img_gray = rgb2gray(img_arr)
        
        # Basic Image Statistics
        gray_mean = np.mean(img_gray)
        gray_std = np.std(img_gray)
        gray_median = np.median(img_gray)
        
        # Create feature dictionary
        features = {
            'red_mean': mean_rgb[0],
            'green_mean': mean_rgb[1],
            'blue_mean': mean_rgb[2],
            'red_std': std_rgb[0],
            'green_std': std_rgb[1],
            'blue_std': std_rgb[2],
            'red_median': median_rgb[0],
            'green_median': median_rgb[1],
            'blue_median': median_rgb[2],
            'gray_mean': gray_mean,
            'gray_std': gray_std,
            'gray_median': gray_median,
        }
        
        return features
    
    except Exception as e:
        print(f"Error extracting basic features from {os.path.basename(image_path)}: {e}")
        return {}

def extract_texture_features(image_path):
    """
    Extract GLCM texture features from an image.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: Dictionary of extracted texture features
    """
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            return {}
        
        # Open image and convert to grayscale
        img = Image.open(image_path).convert('L')
        img_arr = np.array(img)
        
        # Normalize to 8-bit range if needed
        if img_arr.max() > 255 or img_arr.dtype != np.uint8:
            img_arr = (img_arr / img_arr.max() * 255).astype(np.uint8)
        
        # Parameters for GLCM
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Check if image has enough variation for meaningful GLCM
        if img_arr.std() < 1e-5:
            # Return default values for flat image
            return {
                'contrast_mean': 0,
                'dissimilarity_mean': 0,
                'homogeneity_mean': 1,
                'energy_mean': 1.0/max(1, img_arr.size),
                'correlation_mean': 0,
                'ASM_mean': 1.0/max(1, img_arr.size),
                'contrast_std': 0,
                'dissimilarity_std': 0,
                'homogeneity_std': 0,
                'energy_std': 0,
                'correlation_std': 0,
                'ASM_std': 0
            }
        
        # Calculate GLCM
        glcm = graycomatrix(
            img_arr, 
            distances=distances, 
            angles=angles, 
            levels=256, 
            symmetric=True, 
            normed=True
        )
        
        # Extract GLCM properties
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        texture_features = {}
        
        # Calculate mean and std for each property across distances and angles
        for prop in props:
            feature = graycoprops(glcm, prop)
            texture_features[f'{prop}_mean'] = np.mean(feature)
            texture_features[f'{prop}_std'] = np.std(feature)
        
        return texture_features
    
    except Exception as e:
        print(f"Error extracting texture features from {os.path.basename(image_path)}: {e}")
        return {}

def extract_advanced_features(image_path):
    """
    Extract advanced image features including entropy, LBP, and edge metrics.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: Dictionary of extracted advanced features
    """
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            return {}
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_arr = np.array(img)
        img_gray = rgb2gray(img_arr)
        
        features = {}
        
        # Entropy (measure of randomness/complexity)
        features['entropy'] = shannon_entropy(img_gray)
        
        # Local Binary Pattern for texture
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
        features['lbp_mean'] = np.mean(hist)
        features['lbp_std'] = np.std(hist)
        features['lbp_entropy'] = shannon_entropy(hist)
        
        # Edge detection using Sobel filter
        edge_sobel = sobel(img_gray)
        features['edge_mean'] = np.mean(edge_sobel)
        features['edge_std'] = np.std(edge_sobel)
        features['edge_max'] = np.max(edge_sobel)
        
        # Color ratios (useful for detecting water, vegetation, soil)
        if img_arr.shape[2] >= 3:  # Ensure image has RGB channels
            r = img_arr[:,:,0].astype(float)
            g = img_arr[:,:,1].astype(float)
            b = img_arr[:,:,2].astype(float)
            
            # Avoid division by zero
            epsilon = 1e-10
            
            # Red to Green ratio (soil indicator)
            rg_ratio = np.mean(r / (g + epsilon))
            features['red_green_ratio'] = rg_ratio
            
            # Blue to Red ratio (water indicator)
            br_ratio = np.mean(b / (r + epsilon))
            features['blue_red_ratio'] = br_ratio
            
            # Green to Red ratio (vegetation indicator)
            gr_ratio = np.mean(g / (r + epsilon))
            features['green_red_ratio'] = gr_ratio
        
        return features
    
    except Exception as e:
        print(f"Error extracting advanced features from {os.path.basename(image_path)}: {e}")
        return {}

def extract_all_features(image_path):
    """
    Extract all features from an image.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: Dictionary of all extracted features
    """
    # Combine all feature extraction methods
    basic_features = extract_basic_features(image_path)
    texture_features = extract_texture_features(image_path)
    advanced_features = extract_advanced_features(image_path)
    
    # Merge all features
    all_features = {
        **basic_features,
        **texture_features,
        **advanced_features
    }
    
    # Handle potential NaN/inf values
    for key, value in all_features.items():
        if np.isnan(value) or np.isinf(value):
            all_features[key] = 0.0
    
    return all_features

def extract_features_from_df(image_folder, dataframe):
    """
    Extract features for all images in a dataframe.
    
    Args:
        image_folder (str): Folder containing images
        dataframe (pd.DataFrame): DataFrame with image filenames
        
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    features_list = []
    
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Extracting Features"):
        if 'filename' not in row:
            continue
            
        img_path = os.path.join(image_folder, row['filename'])
        features = extract_all_features(img_path)
        
        if features:
            # Add label and filename to features
            features['filename'] = row['filename']
            if 'label' in row:
                features['label'] = row['label']
            
            features_list.append(features)
    
    if features_list:
        features_df = pd.DataFrame(features_list)
        return features_df
    else:
        return pd.DataFrame()

def add_historical_features(features_df, lat_lon_mapping, years_back=5):
    """
    Add historical time-series features for each point.
    
    Args:
        features_df (pd.DataFrame): DataFrame with extracted image features
        lat_lon_mapping (dict): Dictionary mapping filenames to (lat, lon) coordinates
        years_back (int): Number of years to look back for historical data
        
    Returns:
        pd.DataFrame: DataFrame with added historical features
    """
    # Initialize Earth Engine
    if not ee_utils.initialize_ee():
        print("Error: Could not initialize Earth Engine. Cannot add historical features.")
        return features_df
    
    enhanced_features = []
    
    for idx, row in tqdm(features_df.iterrows(), total=len(features_df), desc="Adding Historical Features"):
        filename = row['filename']
        
        if filename not in lat_lon_mapping:
            # Skip if coordinates not found
            enhanced_features.append(row.to_dict())
            continue
        
        lat, lon = lat_lon_mapping[filename]
        
        # Get historical data for this point
        historical_data = ee_utils.get_historical_images(lat, lon, years_back=years_back)
        
        if not historical_data:
            # Skip if no historical data found
            enhanced_features.append(row.to_dict())
            continue
        
        # Extract band statistics
        historical_stats = ee_utils.extract_historical_band_stats(historical_data, lat, lon)
        
        if historical_stats.empty:
            # Skip if no stats could be extracted
            enhanced_features.append(row.to_dict())
            continue
        
        # Add historical features to the row
        enhanced_row = row.to_dict()
        
        # Add trend features for key indicators (if enough historical points)
        if len(historical_stats) >= 3:
            # Calculate trend slopes for important indices
            for index in ['NDVI', 'NDWI', 'MNDWI', 'BSI']:
                col_name = f"{index}_mean"
                if col_name in historical_stats.columns:
                    # Calculate trend (slope) over time
                    y = historical_stats[col_name].values
                    x = np.arange(len(y))
                    
                    if len(x) > 1 and not np.all(np.isnan(y)):
                        # Remove NaN values
                        valid = ~np.isnan(y)
                        if sum(valid) > 1:
                            slope, _, _, _, _ = stats.linregress(x[valid], y[valid])
                            enhanced_row[f"{index}_trend"] = slope
            
            # Add variance and range metrics for key bands
            for band in ['Red', 'Green', 'Blue', 'NIR', 'SWIR1']:
                col_name = f"{band}_mean"  # Use standardized band names
                if col_name in historical_stats.columns:
                    values = historical_stats[col_name].values
                    if not np.all(np.isnan(values)):
                        valid_values = values[~np.isnan(values)]
                        if len(valid_values) > 0:
                            enhanced_row[f"{band}_historical_var"] = np.var(valid_values)
                            enhanced_row[f"{band}_historical_range"] = np.max(valid_values) - np.min(valid_values)
        
        enhanced_features.append(enhanced_row)
    
    # Convert back to DataFrame
    return pd.DataFrame(enhanced_features)

def visualize_feature_importance(model, feature_names, output_file=None):
    """
    Visualize feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        output_file (str, optional): Path to save the visualization
        
    Returns:
        None
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute. Cannot visualize.")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Select top 30 features to keep visualization readable
    top_n = min(30, len(indices))
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    plt.barh(range(top_n), importances[indices[:top_n]], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
    
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top Features for Sand Mining Detection')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Feature importance visualization saved to {output_file}")
    
    plt.close()
    
    # Also save feature importance as JSON for easier analysis
    if output_file:
        json_file = output_file.replace('.png', '.json')
        importance_dict = {feature_names[i]: float(importances[i]) for i in indices}
        
        with open(json_file, 'w') as f:
            json.dump(importance_dict, f, indent=2)
        
        print(f"Feature importance data saved to {json_file}")

def calculate_feature_correlation(features_df, output_file=None):
    """
    Calculate and visualize feature correlation matrix.
    
    Args:
        features_df (pd.DataFrame): DataFrame with features
        output_file (str, optional): Path to save the correlation matrix
        
    Returns:
        None
    """
    # Select only numeric columns
    numeric_df = features_df.select_dtypes(include=['float64', 'int64'])
    
    # Drop label column if present
    if 'label' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['label'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Visualize correlation matrix
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {output_file}")
    
    plt.close()
    
    # Identify highly correlated features
    threshold = 0.8
    high_corr = {}
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr[(corr_matrix.columns[i], corr_matrix.columns[j])] = corr_matrix.iloc[i, j]
    
    # Print highly correlated features
    if high_corr:
        print("\nHighly correlated features (|r| > 0.8):")
        for (f1, f2), corr in sorted(high_corr.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"{f1} <-> {f2}: {corr:.3f}")
    
    return high_corr