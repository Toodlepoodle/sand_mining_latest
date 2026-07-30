#!/usr/bin/env python3
"""
Enhanced feature extraction module focusing on highlighted areas for sand mining detection.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
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

def polygon_to_mask(polygon, height, width):
    """
    Rasterize a free-hand polygon (list of [x, y] image-pixel points) into a
    boolean mask of shape (height, width).

    Returns a boolean numpy array, or None if the polygon is invalid.
    """
    try:
        if not polygon or len(polygon) < 3:
            return None
        mask_img = Image.new('L', (width, height), 0)
        pts = [(float(p[0]), float(p[1])) for p in polygon]
        ImageDraw.Draw(mask_img).polygon(pts, outline=1, fill=1)
        return np.array(mask_img, dtype=bool)
    except Exception:
        return None

def build_type_masks(annotations, height, width):
    """
    Pool ALL annotation polygons (or legacy bboxes) of each type into a single
    combined boolean mask per type.

    This is the key change: instead of extracting features separately for every
    individual annotation, every region of the same type on an image is merged
    into ONE mask, so a fixed, consistent feature set is produced per type.

    Returns: dict {type_name: boolean mask (H, W)}
    """
    masks = {}
    for ann in annotations:
        t = ann.get('type', 'unknown')
        m = None
        poly = ann.get('polygon')
        if poly and len(poly) >= 3:
            m = polygon_to_mask(poly, height, width)
        if m is None and ann.get('bbox'):       # legacy rectangle support
            x1, y1, x2, y2 = ann['bbox']
            m = np.zeros((height, width), dtype=bool)
            x1 = max(0, min(width,  int(x1)));  x2 = max(0, min(width,  int(x2)))
            y1 = max(0, min(height, int(y1)));  y2 = max(0, min(height, int(y2)))
            if x2 > x1 and y2 > y1:
                m[y1:y2, x1:x2] = True
        if m is None:
            continue
        if t in masks:
            masks[t] = masks[t] | m
        else:
            masks[t] = m
    return masks

def extract_region_features(image_path, mask, feature_prefix="region", _img_arr=None):
    """
    Extract features from an arbitrary masked region (pooled polygons of one
    type). Mirrors extract_area_features but operates on a boolean mask instead
    of a rectangular bbox, so free-hand shapes are supported.

    Args:
        image_path (str): path to image (used only if _img_arr not supplied)
        mask (np.ndarray bool): region mask, shape (H, W)
        feature_prefix (str): prefix for feature names (the annotation type)
        _img_arr (np.ndarray, optional): preloaded RGB array to avoid re-reading

    Returns:
        dict of features for this region.
    """
    try:
        if _img_arr is not None:
            img_arr = _img_arr
        else:
            img_arr = np.array(Image.open(image_path).convert('RGB'))

        if mask is None or mask.sum() < 9:   # too small to be meaningful
            return {}

        ys, xs = np.where(mask)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        # Crop to the mask's bounding box for texture ops, and a local mask
        crop      = img_arr[y1:y2, x1:x2]
        local_msk = mask[y1:y2, x1:x2]
        if crop.size == 0:
            return {}

        # Masked pixels (flat) for colour statistics
        sel = crop[local_msk]              # (N, 3)
        if sel.size == 0:
            return {}

        features = {}
        epsilon = 1e-10

        # ── Colour statistics over the exact masked pixels ──────────────────
        for i, color in enumerate(['red', 'green', 'blue']):
            channel = sel[:, i].astype(float)
            features[f'{feature_prefix}_{color}_mean']     = np.mean(channel)
            features[f'{feature_prefix}_{color}_std']      = np.std(channel)
            features[f'{feature_prefix}_{color}_median']   = np.median(channel)
            features[f'{feature_prefix}_{color}_range']    = np.max(channel) - np.min(channel)
            features[f'{feature_prefix}_{color}_skewness'] = stats.skew(channel)
            features[f'{feature_prefix}_{color}_kurtosis'] = stats.kurtosis(channel)

        r = sel[:, 0].astype(float); g = sel[:, 1].astype(float); b = sel[:, 2].astype(float)
        features[f'{feature_prefix}_rg_ratio']    = np.mean(r / (g + epsilon))
        features[f'{feature_prefix}_rb_ratio']    = np.mean(r / (b + epsilon))
        features[f'{feature_prefix}_gb_ratio']    = np.mean(g / (b + epsilon))
        features[f'{feature_prefix}_br_ratio']    = np.mean(b / (r + epsilon))
        features[f'{feature_prefix}_gr_ratio']    = np.mean(g / (r + epsilon))
        features[f'{feature_prefix}_bg_ratio']    = np.mean(b / (g + epsilon))
        features[f'{feature_prefix}_soil_index']  = np.mean((r + g) / (b + epsilon))
        features[f'{feature_prefix}_water_index'] = np.mean(b / (r + g + epsilon))

        # ── Texture over the cropped bbox (masked area dominates) ───────────
        crop_gray = rgb2gray(crop)
        if crop_gray.std() > 1e-5:
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            crop_gray_uint8 = convert_to_uint8(crop_gray)
            glcm = graycomatrix(crop_gray_uint8, distances=distances, angles=angles,
                                levels=256, symmetric=True, normed=True)
            for prop in ['contrast', 'dissimilarity', 'homogeneity',
                         'energy', 'correlation', 'ASM']:
                vals = graycoprops(glcm, prop)
                features[f'{feature_prefix}_{prop}_mean'] = np.mean(vals)
                features[f'{feature_prefix}_{prop}_std']  = np.std(vals)

        # LBP
        radius = 2; n_points = 8 * radius
        crop_gray_uint8 = convert_to_uint8(crop_gray)
        lbp = local_binary_pattern(crop_gray_uint8, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                               range=(0, n_points + 2), density=True)
        features[f'{feature_prefix}_lbp_uniformity'] = np.max(hist)
        features[f'{feature_prefix}_lbp_entropy']    = shannon_entropy(hist)
        features[f'{feature_prefix}_lbp_contrast']   = np.sum((np.arange(len(hist)) - np.mean(hist))**2 * hist)

        # Edges
        edge_sobel = sobel(crop_gray)
        features[f'{feature_prefix}_edge_density']  = np.mean(edge_sobel > 0.1)
        features[f'{feature_prefix}_edge_strength'] = np.mean(edge_sobel)
        features[f'{feature_prefix}_edge_max']      = np.max(edge_sobel)
        features[f'{feature_prefix}_edge_std']      = np.std(edge_sobel)

        # ── Shape / size of the pooled region (true polygon area, not bbox) ──
        region_pixels = int(mask.sum())
        bbox_h = int(y2 - y1); bbox_w = int(x2 - x1)
        features[f'{feature_prefix}_area_pixels']  = region_pixels
        features[f'{feature_prefix}_aspect_ratio'] = bbox_w / max(bbox_h, 1)
        features[f'{feature_prefix}_extent']       = region_pixels / max(bbox_w * bbox_h, 1)  # fill ratio
        features[f'{feature_prefix}_entropy']      = shannon_entropy(crop_gray)
        features[f'{feature_prefix}_brightness']   = float(np.mean(np.sum(sel, axis=1)))
        features[f'{feature_prefix}_vegetation_index'] = float(np.mean((g > r) & (g > b)))

        # Local roughness inside the crop
        from scipy.ndimage import generic_filter
        local_std = generic_filter(crop_gray, np.std, size=5)
        features[f'{feature_prefix}_local_std_mean']     = np.mean(local_std)
        features[f'{feature_prefix}_local_std_max']      = np.max(local_std)
        features[f'{feature_prefix}_surface_roughness']  = np.std(local_std)

        return features
    except Exception as e:
        print(f"Error extracting region features ({feature_prefix}): {e}")
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
        
        # Extract features from highlighted areas.
        # CHANGED: all regions of the same type are POOLED into a single mask,
        # producing ONE consistent feature set per type (e.g. sand_mining_*),
        # instead of separate per-annotation columns (sand_mining_0_*, _1_* ...).
        area_features = {}

        # Known annotation types — always emit the same columns so the feature
        # vector length is identical across every image (and at map time).
        KNOWN_TYPES = ['sand_mining', 'equipment', 'water_disturbance', 'no_mining']

        if annotations:
            img_arr_full = np.array(Image.open(image_path).convert('RGB'))
            H, W = img_arr_full.shape[:2]
            type_masks = build_type_masks(annotations, H, W)

            total_image_area = float(H * W)

            for t in KNOWN_TYPES:
                mask = type_masks.get(t)
                if mask is not None and mask.sum() >= 9:
                    feats = extract_region_features(
                        image_path, mask, feature_prefix=t, _img_arr=img_arr_full
                    )
                    area_features.update(feats)
                    n_regions = sum(1 for a in annotations if a.get('type') == t)
                    area_features[f'num_{t}_areas']   = n_regions
                    area_features[f'total_{t}_area']  = int(mask.sum())
                    area_features[f'{t}_area_ratio']  = mask.sum() / max(total_image_area, 1)
                else:
                    area_features[f'num_{t}_areas']   = 0
                    area_features[f'total_{t}_area']   = 0
                    area_features[f'{t}_area_ratio']   = 0.0

            # Backward-compatible aliases used elsewhere in the codebase
            area_features['num_sand_mining_areas'] = area_features.get('num_sand_mining_areas', 0)
            area_features['total_mining_area']     = area_features.get('total_sand_mining_area', 0)
            area_features['mining_area_ratio']     = area_features.get('sand_mining_area_ratio', 0.0)
            area_features['num_equipment_areas']   = area_features.get('num_equipment_areas', 0)
            area_features['total_equipment_area']  = area_features.get('total_equipment_area', 0)
            area_features['num_water_disturbance_areas']   = area_features.get('num_water_disturbance_areas', 0)
            area_features['total_water_disturbance_area']  = area_features.get('total_water_disturbance_area', 0)
        else:
            # No annotations - set summary area features to zero
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


def extract_historical_trend_features(lat, lon, years_back=3, buffer_m=1500):
    """
    Extract temporal trend features for a training point using its lat/lon.

    Literature basis:
    - Li et al. (2024): quarterly NDVI/BSI slopes detect sandbar dynamics (F1 0.85)
    - Bendixen et al. (2021): trends reveal 20-50% more extractions vs snapshots
    - EuroMineNet (2025): temporal BFAST > single-date classification

    For each labeled training image (which has lat/lon encoded in its filename),
    this fetches 3 years of quarterly Sentinel-2 composites and computes the
    linear slope of NDVI, NDWI, MNDWI, and BSI over time.

    Args:
        lat (float): Latitude of the training point
        lon (float): Longitude of the training point
        years_back (int): Number of years to look back
        buffer_m (int): Buffer radius in metres

    Returns:
        dict: Historical trend features (NDVI_trend, BSI_trend, etc.)
              Returns zeros on failure so training is never blocked.
    """
    trend_features = {
        'NDVI_trend':  0.0,
        'NDWI_trend':  0.0,
        'MNDWI_trend': 0.0,
        'BSI_trend':   0.0,
        'hist_periods': 0,
    }

    if lat is None or lon is None:
        return trend_features

    try:
        from src import ee_utils
        from scipy import stats as scipy_stats

        historical_data = ee_utils.get_historical_images(
            lat, lon, buffer_m=buffer_m, years_back=years_back, interval_months=3
        )

        if not historical_data:
            return trend_features

        historical_stats = ee_utils.extract_historical_band_stats(
            historical_data, lat, lon, buffer_m=buffer_m
        )

        if historical_stats.empty or len(historical_stats) < 3:
            return trend_features

        trend_features['hist_periods'] = len(historical_stats)

        x = np.arange(len(historical_stats))

        for index_name in ['NDVI', 'NDWI', 'MNDWI', 'BSI']:
            col = f'{index_name}_mean'
            if col in historical_stats.columns:
                y = historical_stats[col].values
                valid = ~np.isnan(y)
                if valid.sum() >= 3:
                    try:
                        slope, _, _, _, _ = scipy_stats.linregress(x[valid], y[valid])
                        trend_features[f'{index_name}_trend'] = float(slope)
                    except Exception:
                        pass

    except Exception as e:
        # Never crash training — return zeros silently
        pass

    return trend_features


def parse_lat_lon_from_filename(filename):
    """
    Parse lat/lon from training image filename.
    Expected format: train_image_N_LAT_LON.png
    e.g. train_image_5_23.170659_87.954341.png

    Returns (lat, lon) or (None, None) if parsing fails.
    """
    try:
        parts = os.path.splitext(filename)[0].split('_')
        # Last two parts should be lat and lon
        lon = float(parts[-1])
        lat = float(parts[-2])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except Exception:
        pass
    return None, None


def extract_features_with_history(image_path, years_back=3, buffer_m=1500):
    """
    Full feature extraction combining:
    1. Current image features (spectral, texture, area annotations)
    2. Historical trend features from GEE time-series

    This is the training-time feature extractor that ensures training and
    mapping use identical feature vectors.

    Args:
        image_path (str): Path to the training image
        years_back (int): Years of historical data to use
        buffer_m (int): Buffer in metres

    Returns:
        dict: Combined feature dictionary
    """
    # Current image features
    current_features = extract_enhanced_features(image_path)

    # Parse coordinates from filename
    filename = os.path.basename(image_path)
    lat, lon = parse_lat_lon_from_filename(filename)

    # Historical trend features
    if lat is not None and years_back > 0:
        hist_features = extract_historical_trend_features(
            lat, lon, years_back=years_back, buffer_m=buffer_m
        )
        current_features.update(hist_features)
    else:
        # Add zero-valued trend features so feature vector length is consistent
        current_features.update({
            'NDVI_trend':   0.0,
            'NDWI_trend':   0.0,
            'MNDWI_trend':  0.0,
            'BSI_trend':    0.0,
            'hist_periods': 0,
        })

    return current_features


def extract_features_from_df_with_history(image_folder, dataframe,
                                           years_back=3, buffer_m=1500):
    """
    Extract full features (current + historical) for all labeled images.

    Use this instead of extract_features_from_df when historical data
    should be included in training.

    Args:
        image_folder (str): Folder containing training images
        dataframe (pd.DataFrame): DataFrame with 'filename' and 'label' columns
        years_back (int): Years of historical imagery to use
        buffer_m (int): Buffer radius in metres

    Returns:
        pd.DataFrame: Feature DataFrame with historical trend columns included
    """
    features_list = []

    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe),
                         desc="Extracting Features + History"):
        if 'filename' not in row:
            continue

        img_path = os.path.join(image_folder, row['filename'])
        feat = extract_features_with_history(img_path, years_back=years_back,
                                              buffer_m=buffer_m)

        if feat:
            feat['filename'] = row['filename']
            if 'label' in row:
                feat['label'] = row['label']
            features_list.append(feat)

    if features_list:
        features_df = pd.DataFrame(features_list)
        n_feat = len(features_df.columns) - 2
        print(f"Extracted {n_feat} features (incl. historical trends) "
              f"from {len(features_df)} images")
        return features_df
    else:
        return pd.DataFrame()