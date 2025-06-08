#!/usr/bin/env python3
"""
Enhanced feature extraction module focusing on highlighted areas for sand mining detection.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
from skimage.feature import local_binary_pattern
from skimage.filters import sobel, gaussian
from skimage.segmentation import slic
from skimage.measure import regionprops
import joblib
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import cv2

from src import config

def convert_to_uint8(image_array):
    """
    Convert any image array to proper uint8 format for texture analysis
    
    Args:
        image_array: numpy array of any dtype
    
    Returns:
        numpy array in uint8 format (0-255)
    """
    if image_array.dtype == np.uint8:
        return image_array
    
    # Handle different input ranges
    if image_array.max() <= 1.0:
        # Floating point 0-1 range
        return (image_array * 255).astype(np.uint8)
    elif image_array.max() <= 255:
        # Already in 0-255 range but wrong dtype
        return image_array.astype(np.uint8)
    else:
        # Normalize to 0-255 range
        normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        return (normalized * 255).astype(np.uint8)

def load_annotations(filename):
    """Load area annotations for an image."""
    try:
        if os.path.exists(config.ANNOTATIONS_FILE):
            with open(config.ANNOTATIONS_FILE, 'r') as f:
                all_annotations = json.load(f)
            return all_annotations.get(filename, [])
    except Exception as e:
        print(f"Error loading annotations: {e}")
    return []

def extract_area_features(image_path, bbox, feature_prefix="area"):
    """
    Extract enhanced features from a specific area of an image.
    
    Args:
        image_path (str): Path to image file
        bbox (list): Bounding box [x1, y1, x2, y2]
        feature_prefix (str): Prefix for feature names
        
    Returns:
        dict: Dictionary of extracted features from the area
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_arr = np.array(img)
        
        # Extract the area
        x1, y1, x2, y2 = bbox
        area_arr = img_arr[y1:y2, x1:x2]
        
        if area_arr.size == 0:
            return {}
        
        # Convert to grayscale for texture analysis
        area_gray = rgb2gray(area_arr)
        
        features = {}
        
        # Basic color statistics for the area
        for i, color in enumerate(['red', 'green', 'blue']):
            channel = area_arr[:,:,i]
            features[f'{feature_prefix}_{color}_mean'] = np.mean(channel)
            features[f'{feature_prefix}_{color}_std'] = np.std(channel)
            features[f'{feature_prefix}_{color}_median'] = np.median(channel)
            features[f'{feature_prefix}_{color}_range'] = np.max(channel) - np.min(channel)
            features[f'{feature_prefix}_{color}_skewness'] = stats.skew(channel.flatten())
            features[f'{feature_prefix}_{color}_kurtosis'] = stats.kurtosis(channel.flatten())
        
        # Enhanced color ratios
        r, g, b = area_arr[:,:,0], area_arr[:,:,1], area_arr[:,:,2]
        epsilon = 1e-10
        
        # Color ratios (important for sand/soil detection)
        features[f'{feature_prefix}_rg_ratio'] = np.mean(r / (g + epsilon))
        features[f'{feature_prefix}_rb_ratio'] = np.mean(r / (b + epsilon))
        features[f'{feature_prefix}_gb_ratio'] = np.mean(g / (b + epsilon))
        features[f'{feature_prefix}_br_ratio'] = np.mean(b / (r + epsilon))
        features[f'{feature_prefix}_gr_ratio'] = np.mean(g / (r + epsilon))
        features[f'{feature_prefix}_bg_ratio'] = np.mean(b / (g + epsilon))
        
        # Soil/sand color indices
        # Brown/soil index: higher red and green compared to blue
        features[f'{feature_prefix}_soil_index'] = np.mean((r + g) / (b + epsilon))
        # Water index: typically higher blue
        features[f'{feature_prefix}_water_index'] = np.mean(b / (r + g + epsilon))
        
        # Enhanced texture features
        if area_gray.std() > 1e-5:  # Only if area has variation
            # GLCM texture features
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            # Convert to uint8 for accurate GLCM computation
            area_gray_uint8 = convert_to_uint8(area_gray)
            
            glcm = graycomatrix(
                area_gray_uint8, 
                distances=distances, 
                angles=angles, 
                levels=256, 
                symmetric=True, 
                normed=True
            )
            
            # Extract texture properties
            props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            for prop in props:
                feature_values = graycoprops(glcm, prop)
                features[f'{feature_prefix}_{prop}_mean'] = np.mean(feature_values)
                features[f'{feature_prefix}_{prop}_std'] = np.std(feature_values)
        
        # Local Binary Pattern (LBP) for texture
        radius = 2
        n_points = 8 * radius
        # Convert to uint8 for accurate LBP computation
        area_gray_uint8 = convert_to_uint8(area_gray)
        lbp = local_binary_pattern(area_gray_uint8, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
        features[f'{feature_prefix}_lbp_uniformity'] = np.max(hist)  # Most frequent pattern
        features[f'{feature_prefix}_lbp_entropy'] = shannon_entropy(hist)
        features[f'{feature_prefix}_lbp_contrast'] = np.sum((np.arange(len(hist)) - np.mean(hist))**2 * hist)
        
        # Edge detection features
        edge_sobel = sobel(area_gray)
        features[f'{feature_prefix}_edge_density'] = np.mean(edge_sobel > 0.1)
        features[f'{feature_prefix}_edge_strength'] = np.mean(edge_sobel)
        features[f'{feature_prefix}_edge_max'] = np.max(edge_sobel)
        features[f'{feature_prefix}_edge_std'] = np.std(edge_sobel)
        
        # Shape and size features of the area
        area_height, area_width = area_arr.shape[:2]
        features[f'{feature_prefix}_area_pixels'] = area_height * area_width
        features[f'{feature_prefix}_aspect_ratio'] = area_width / max(area_height, 1)
        features[f'{feature_prefix}_compactness'] = (area_width * area_height) / max((area_width + area_height), 1)
        
        # Entropy (measure of randomness/disorder)
        features[f'{feature_prefix}_entropy'] = shannon_entropy(area_gray)
        
        # Spectral indices for the area (if we have RGB)
        if area_arr.shape[2] >= 3:
            # Simple vegetation index (green dominance)
            features[f'{feature_prefix}_vegetation_index'] = np.mean(g > r) * np.mean(g > b)
            
            # Brightness
            features[f'{feature_prefix}_brightness'] = np.mean(np.sum(area_arr, axis=2))
            
            # Color diversity (how many different colors)
            unique_colors = len(np.unique(area_arr.reshape(-1, area_arr.shape[2]), axis=0))
            max_possible_colors = area_height * area_width
            features[f'{feature_prefix}_color_diversity'] = unique_colors / max(max_possible_colors, 1)
        
        # Advanced texture: Local Standard Deviation
        # Indicates roughness/smoothness of surface
        from scipy.ndimage import generic_filter
        local_std = generic_filter(area_gray, np.std, size=5)
        features[f'{feature_prefix}_local_std_mean'] = np.mean(local_std)
        features[f'{feature_prefix}_local_std_max'] = np.max(local_std)
        features[f'{feature_prefix}_surface_roughness'] = np.std(local_std)
        
        return features
    
    except Exception as e:
        print(f"Error extracting area features: {e}")
        return {}

def extract_enhanced_features(image_path):
    """
    Extract enhanced features from an image, focusing on highlighted areas.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: Dictionary of all extracted features
    """
    try:
        # Get base filename for annotations
        filename = os.path.basename(image_path)
        
        # Load annotations for this image
        annotations = load_annotations(filename)
        
        # Start with global image features
        global_features = extract_basic_features(image_path)
        global_features.update(extract_texture_features(image_path))
        global_features.update(extract_advanced_features(image_path))
        
        # Extract features from highlighted areas
        area_features = {}
        
        if annotations:
            for i, annotation in enumerate(annotations):
                annotation_type = annotation.get('type', 'unknown')
                bbox = annotation.get('bbox', [])
                
                if len(bbox) == 4:
                    # Extract features from this specific area
                    prefix = f"{annotation_type}_{i}"
                    area_feats = extract_area_features(image_path, bbox, prefix)
                    area_features.update(area_feats)
            
            # Summary statistics across all sand mining areas
            sand_mining_areas = [ann for ann in annotations if ann['type'] == 'sand_mining']
            if sand_mining_areas:
                area_features['num_sand_mining_areas'] = len(sand_mining_areas)
                
                # Calculate total area of sand mining
                total_mining_area = 0
                for ann in sand_mining_areas:
                    bbox = ann['bbox']
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    total_mining_area += area
                area_features['total_mining_area'] = total_mining_area
                
                # Mining area density (relative to image size)
                img = Image.open(image_path)
                total_image_area = img.width * img.height
                area_features['mining_area_ratio'] = total_mining_area / total_image_area
            
            # Equipment areas
            equipment_areas = [ann for ann in annotations if ann['type'] == 'equipment']
            if equipment_areas:
                area_features['num_equipment_areas'] = len(equipment_areas)
                total_equipment_area = sum((ann['bbox'][2] - ann['bbox'][0]) * (ann['bbox'][3] - ann['bbox'][1]) 
                                         for ann in equipment_areas)
                area_features['total_equipment_area'] = total_equipment_area
            
            # Water disturbance areas
            water_areas = [ann for ann in annotations if ann['type'] == 'water_disturbance']
            if water_areas:
                area_features['num_water_disturbance_areas'] = len(water_areas)
                total_water_area = sum((ann['bbox'][2] - ann['bbox'][0]) * (ann['bbox'][3] - ann['bbox'][1]) 
                                     for ann in water_areas)
                area_features['total_water_disturbance_area'] = total_water_area
        else:
            # No annotations - set area features to zero
            area_features.update({
                'num_sand_mining_areas': 0,
                'total_mining_area': 0,
                'mining_area_ratio': 0,
                'num_equipment_areas': 0,
                'total_equipment_area': 0,
                'num_water_disturbance_areas': 0,
                'total_water_disturbance_area': 0
            })
        
        # Combine all features
        all_features = {**global_features, **area_features}
        
        # Handle NaN/inf values
        for key, value in all_features.items():
            if np.isnan(value) or np.isinf(value):
                all_features[key] = 0.0
        
        return all_features
    
    except Exception as e:
        print(f"Error in enhanced feature extraction for {os.path.basename(image_path)}: {e}")
        return {}

def extract_basic_features(image_path):
    """
    Extract basic color and statistical features from an image.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: Dictionary of extracted features
    """
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return {}
        
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
        
        features = {
            'global_red_mean': mean_rgb[0],
            'global_green_mean': mean_rgb[1],
            'global_blue_mean': mean_rgb[2],
            'global_red_std': std_rgb[0],
            'global_green_std': std_rgb[1],
            'global_blue_std': std_rgb[2],
            'global_red_median': median_rgb[0],
            'global_green_median': median_rgb[1],
            'global_blue_median': median_rgb[2],
            'global_gray_mean': gray_mean,
            'global_gray_std': gray_std,
            'global_gray_median': gray_median,
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
        if not os.path.exists(image_path):
            return {}
        
        img = Image.open(image_path).convert('L')
        img_arr = np.array(img)
        
        # Ensure proper uint8 conversion for GLCM
        img_arr = convert_to_uint8(img_arr)
        
        if img_arr.std() < 1e-5:
            return {
                'global_contrast_mean': 0,
                'global_dissimilarity_mean': 0,
                'global_homogeneity_mean': 1,
                'global_energy_mean': 1.0/max(1, img_arr.size),
                'global_correlation_mean': 0,
                'global_ASM_mean': 1.0/max(1, img_arr.size),
            }
        
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(
            img_arr, 
            distances=distances, 
            angles=angles, 
            levels=256, 
            symmetric=True, 
            normed=True
        )
        
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        texture_features = {}
        for prop in props:
            feature = graycoprops(glcm, prop)
            texture_features[f'global_{prop}_mean'] = np.mean(feature)
            texture_features[f'global_{prop}_std'] = np.std(feature)
        
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
        if not os.path.exists(image_path):
            return {}
        
        img = Image.open(image_path).convert('RGB')
        img_arr = np.array(img)
        img_gray = rgb2gray(img_arr)
        
        features = {}
        
        # Entropy
        features['global_entropy'] = shannon_entropy(img_gray)
        
        # Local Binary Pattern
        radius = 3
        n_points = 8 * radius
        # Convert to uint8 for accurate LBP computation
        img_gray_uint8 = convert_to_uint8(img_gray)
        lbp = local_binary_pattern(img_gray_uint8, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
        features['global_lbp_mean'] = np.mean(hist)
        features['global_lbp_std'] = np.std(hist)
        features['global_lbp_entropy'] = shannon_entropy(hist)
        
        # Edge detection
        edge_sobel = sobel(img_gray)
        features['global_edge_mean'] = np.mean(edge_sobel)
        features['global_edge_std'] = np.std(edge_sobel)
        features['global_edge_max'] = np.max(edge_sobel)
        
        # Color ratios
        if img_arr.shape[2] >= 3:
            r = img_arr[:,:,0].astype(float)
            g = img_arr[:,:,1].astype(float)
            b = img_arr[:,:,2].astype(float)
            
            epsilon = 1e-10
            
            features['global_red_green_ratio'] = np.mean(r / (g + epsilon))
            features['global_blue_red_ratio'] = np.mean(b / (r + epsilon))
            features['global_green_red_ratio'] = np.mean(g / (r + epsilon))
        
        return features
    
    except Exception as e:
        print(f"Error extracting advanced features from {os.path.basename(image_path)}: {e}")
        return {}

def extract_features_from_df(image_folder, dataframe):
    """
    Extract enhanced features for all images in a dataframe.
    
    Args:
        image_folder (str): Folder containing images
        dataframe (pd.DataFrame): DataFrame with image filenames
        
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    features_list = []
    
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Extracting Enhanced Features"):
        if 'filename' not in row:
            continue
            
        img_path = os.path.join(image_folder, row['filename'])
        features = extract_enhanced_features(img_path)
        
        if features:
            features['filename'] = row['filename']
            if 'label' in row:
                features['label'] = row['label']
            
            features_list.append(features)
    
    if features_list:
        features_df = pd.DataFrame(features_list)
        print(f"Extracted {len(features_df.columns) - 2} features from {len(features_df)} images")
        return features_df
    else:
        return pd.DataFrame()

def visualize_feature_importance(model, feature_names, output_file=None):
    """
    Visualize feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        output_file (str, optional): Path to save the visualization
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute. Cannot visualize.")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Separate area-specific features from global features
    area_features = []
    global_features = []
    
    for i in indices:
        if any(prefix in feature_names[i] for prefix in ['sand_mining_', 'equipment_', 'water_disturbance_', 'num_', 'total_']):
            area_features.append((feature_names[i], importances[i]))
        else:
            global_features.append((feature_names[i], importances[i]))
    
    # Plot top features with distinction between area and global
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Top area features
    if area_features:
        top_area = area_features[:15]
        names, values = zip(*top_area)
        ax1.barh(range(len(names)), values, color='red', alpha=0.7)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names)
        ax1.set_xlabel('Feature Importance')
        ax1.set_title('Top Area-Specific Features')
    
    # Top global features
    if global_features:
        top_global = global_features[:15]
        names, values = zip(*top_global)
        ax2.barh(range(len(names)), values, color='blue', alpha=0.7)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('Top Global Image Features')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Feature importance visualization saved to {output_file}")
    
    plt.close()
    
    # Save feature importance as JSON
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
    """
    numeric_df = features_df.select_dtypes(include=['float64', 'int64'])
    
    if 'label' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['label'])
    
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Enhanced Feature Correlation Matrix')
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
    
    if high_corr:
        print("\nHighly correlated features (|r| > 0.8):")
        for (f1, f2), corr in sorted(high_corr.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"{f1} <-> {f2}: {corr:.3f}")
    
    return high_corr

# Backward compatibility alias
def extract_all_features(image_path):
    """Alias for extract_enhanced_features"""
    return extract_enhanced_features(image_path)