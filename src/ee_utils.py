#!/usr/bin/env python3
"""
Earth Engine utilities for Sand Mining Detection Tool.
"""

import ee
import time
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from PIL import Image, ImageEnhance, ImageFilter
import io
import os

from src import config

def initialize_ee():
    """
    Initializes Earth Engine with robust error handling and authentication.
    
    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    # Configuration - REPLACE WITH YOUR PROJECT INFO
    YOUR_PROJECT_ID = "ed-sayandasgupta97"
    EE_HIGHVOLUME_URL = 'https://earthengine-highvolume.googleapis.com'
    
    # First check if already initialized
    if ee.data._credentials:  # pylint: disable=protected-access
        print("Earth Engine already initialized.")
        return True

    try:
        # First attempt: Initialize with explicit project ID
        ee.Initialize(
            project=YOUR_PROJECT_ID,
            opt_url=EE_HIGHVOLUME_URL
        )
        print(f"Successfully initialized with project: {YOUR_PROJECT_ID}")
        return True
    except Exception as e:
        if 'already initialized' in str(e).lower():
            print("Earth Engine already initialized.")
            return True
        print(f"Initialization failed: {e}")

    try:
        # Authentication flow
        print("\nAttempting Earth Engine Authentication...")
        ee.Authenticate(auth_mode='gcloud')
        
        # Second attempt with credentials
        ee.Initialize(
            project=YOUR_PROJECT_ID,
            opt_url=EE_HIGHVOLUME_URL
        )
        print(f"Successfully initialized after authentication with project: {YOUR_PROJECT_ID}")
        return True
    except Exception as auth_e:
        print(f"\nCritical initialization error: {auth_e}")
        print("\nREQUIRED SETUP STEPS:")
        print("1. Ensure YOU HAVE DONE THESE FIRST:")
        print("   a. Created Google Cloud Project: https://console.cloud.google.com")
        print("   b. Enabled Earth Engine API: https://console.cloud.google.com/apis/library/earthengine.googleapis.com")
        print("   c. Added 'Service Usage Consumer' role to your account")
        print("2. Run these commands:")
        print("   gcloud auth application-default login")
        print("   gcloud config set project ed-sayandasgupta97")
        
        # Final fallback
        try:
            print("\nManual recovery attempt...")
            ee.Initialize(
                project=YOUR_PROJECT_ID,
                opt_url=EE_HIGHVOLUME_URL,
                credentials=ee.oauth.get_credentials_package()
            )
            print("Manual initialization succeeded!")
            return True
        except Exception as final_e:
            print(f"Final initialization failed: {final_e}")
            print("\nTROUBLESHOOTING:")
            print("- Verify project ID 'ed-sayandasgupta97' exists in Google Cloud")
            print("- Check IAM permissions: https://console.cloud.google.com/iam-admin/iam?project=ed-sayandasgupta97")
            print("- Ensure Earth Engine API is enabled")
            return False

def enhance_image(img):
    """
    Apply enhancements to improve image clarity.
    
    Args:
        img: PIL Image object
        
    Returns:
        PIL Image: Enhanced image
    """
    try:
        # Subtle blur to reduce noise
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
        # Unsharp mask for detail
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        # Enhance color saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.2)
        # Adjust brightness slightly
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.05)
        # Final sharpen
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
        return img
    except Exception as e:
        print(f"  Warning: Error during image enhancement: {e}. Returning original image.")
        return img  # Return original if enhancement fails

def get_date_ranges(years_back=5, interval_months=12):
    """
    Generate a list of date ranges going back X years at specified intervals.
    
    Args:
        years_back (int): Number of years to look back
        interval_months (int): Interval between date ranges in months
        
    Returns:
        list: List of (start_date, end_date) tuples as EE Date objects
    """
    now = datetime.datetime.now()
    
    date_ranges = []
    for year in range(years_back):
        for month in range(0, 12, interval_months):
            # Start date is X years and Y months ago
            start_dt = now - datetime.timedelta(days=365*year + 30*month + 15)
            # End date is 1 month after start date (or whatever interval is more appropriate)
            end_dt = start_dt + datetime.timedelta(days=30)
            
            # Convert to EE Date objects
            start_ee = ee.Date(start_dt)
            end_ee = ee.Date(end_dt)
            
            date_ranges.append((start_ee, end_ee))
    
    return date_ranges

def get_spectral_indices(image, bands_dict=None):
    """
    Calculate spectral indices for a given image.
    
    Args:
        image (ee.Image): Input satellite image
        bands_dict (dict): Dictionary mapping standard band names to image-specific band names
        
    Returns:
        ee.Image: Original image with spectral indices added as bands
    """
    # First check which bands actually exist in the image
    try:
        band_list = image.bandNames().getInfo()
        
        # If bands_dict is empty or None, try to automatically map bands
        if not bands_dict:
            # Check for Sentinel-2 bands
            if 'B2' in band_list:  # Sentinel-2
                bands_dict = {
                    'Blue': 'B2',
                    'Green': 'B3',
                    'Red': 'B4',
                    'NIR': 'B8',
                    'SWIR1': 'B11',
                    'SWIR2': 'B12'
                }
            # Check for Landsat 8/9 bands
            elif 'SR_B2' in band_list:  # Landsat 8-9
                bands_dict = {
                    'Blue': 'SR_B2',
                    'Green': 'SR_B3',
                    'Red': 'SR_B4',
                    'NIR': 'SR_B5',
                    'SWIR1': 'SR_B6',
                    'SWIR2': 'SR_B7'
                }
            # Check for Landsat 4-7 bands
            elif 'SR_B1' in band_list:  # Landsat 4-7
                bands_dict = {
                    'Blue': 'SR_B1',
                    'Green': 'SR_B2',
                    'Red': 'SR_B3',
                    'NIR': 'SR_B4',
                    'SWIR1': 'SR_B5',
                    'SWIR2': 'SR_B7'
                }
        
        # Filter bands_dict to only include bands that exist in the image
        available_bands_dict = {}
        for std_name, img_band in bands_dict.items():
            if img_band in band_list:
                available_bands_dict[std_name] = img_band
        
        # If no bands match, return original image
        if not available_bands_dict:
            print(f"Warning: No matching bands found. Available bands: {band_list}")
            return image
            
        # Function to safely rename bands for calculations
        renamed_img = image
        for std_name, img_band in available_bands_dict.items():
            renamed_img = renamed_img.select([img_band], [std_name])
        
        # Calculate indices based on available bands
        result_img = image
        
        # NDVI = (NIR - Red) / (NIR + Red)
        if all(b in available_bands_dict for b in ['NIR', 'Red']):
            ndvi = renamed_img.normalizedDifference(['NIR', 'Red']).rename('NDVI')
            result_img = result_img.addBands(ndvi)
        
        # NDWI = (Green - NIR) / (Green + NIR)
        if all(b in available_bands_dict for b in ['Green', 'NIR']):
            ndwi = renamed_img.normalizedDifference(['Green', 'NIR']).rename('NDWI')
            result_img = result_img.addBands(ndwi)
        
        # MNDWI = (Green - SWIR1) / (Green + SWIR1)
        if all(b in available_bands_dict for b in ['Green', 'SWIR1']):
            mndwi = renamed_img.normalizedDifference(['Green', 'SWIR1']).rename('MNDWI')
            result_img = result_img.addBands(mndwi)
            
        # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        if all(b in available_bands_dict for b in ['SWIR1', 'NIR']):
            ndbi = renamed_img.normalizedDifference(['SWIR1', 'NIR']).rename('NDBI')
            result_img = result_img.addBands(ndbi)
        
        # BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
        if all(b in available_bands_dict for b in ['SWIR1', 'Red', 'NIR', 'Blue']):
            numerator = renamed_img.select('SWIR1').add(renamed_img.select('Red')).subtract(
                renamed_img.select('NIR').add(renamed_img.select('Blue')))
            denominator = renamed_img.select('SWIR1').add(renamed_img.select('Red')).add(
                renamed_img.select('NIR').add(renamed_img.select('Blue')))
            bsi = numerator.divide(denominator).rename('BSI')
            result_img = result_img.addBands(bsi)
        
        # NDTI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
        if all(b in available_bands_dict for b in ['SWIR1', 'SWIR2']):
            ndti = renamed_img.normalizedDifference(['SWIR1', 'SWIR2']).rename('NDTI')
            result_img = result_img.addBands(ndti)
        
        return result_img
    
    except Exception as e:
        print(f"Error in spectral indices calculation: {e}")
        # Return original image if calculation fails
        return image

def get_best_s2_image(region, start_date, end_date, max_cloud_cover=35):
    """
    Get the best Sentinel-2 image for a region within a date range.
    
    Args:
        region (ee.Geometry): Region of interest
        start_date (ee.Date): Start date
        end_date (ee.Date): End date
        max_cloud_cover (int): Maximum acceptable cloud cover percentage
        
    Returns:
        ee.Image: Best available Sentinel-2 image, or None if none found
    """
    # Filter Sentinel-2 collection
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE')
    
    # Check if any images meet the criteria
    image_count = s2_collection.size().getInfo()
    
    if image_count > 0:
        return ee.Image(s2_collection.first())
    else:
        # Try with higher cloud tolerance
        s2_collection_wider = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover + 25)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        
        image_count_wider = s2_collection_wider.size().getInfo()
        if image_count_wider > 0:
            return ee.Image(s2_collection_wider.first())
        
    return None

def get_best_landsat_image(region, start_date, end_date, max_cloud_cover=35):
    """
    Get the best Landsat image for a region within a date range.
    
    Args:
        region (ee.Geometry): Region of interest
        start_date (ee.Date): Start date
        end_date (ee.Date): End date
        max_cloud_cover (int): Maximum acceptable cloud cover percentage
        
    Returns:
        tuple: (ee.Image, str) - Best available Landsat image and its collection ID,
               or (None, None) if none found
    """
    # Check Landsat 8-9 first (newer sensors)
    l89_collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover)) \
        .sort('CLOUD_COVER')
    
    l89_count = l89_collection.size().getInfo()
    
    if l89_count > 0:
        return ee.Image(l89_collection.first()), 'LANDSAT/LC09/C02/T1_L2'
    
    # If no L8-9 images, check Landsat 7 (older but widely used)
    l7_collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover)) \
        .sort('CLOUD_COVER')
    
    l7_count = l7_collection.size().getInfo()
    
    if l7_count > 0:
        return ee.Image(l7_collection.first()), 'LANDSAT/LE07/C02/T1_L2'
    
    # Try with higher cloud tolerance for Landsat 8-9
    l89_wide_collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover + 25)) \
        .sort('CLOUD_COVER')
    
    l89_wide_count = l89_wide_collection.size().getInfo()
    
    if l89_wide_count > 0:
        return ee.Image(l89_wide_collection.first()), 'LANDSAT/LC09/C02/T1_L2'
    
    return None, None

def get_best_viirs_image(region, start_date, end_date):
    """
    Get the best VIIRS image for a region within a date range.
    
    Args:
        region (ee.Geometry): Region of interest
        start_date (ee.Date): Start date
        end_date (ee.Date): End date
        
    Returns:
        ee.Image: Best available VIIRS image, or None if none found
    """
    # VIIRS doesn't have a cloud percentage but has quality bands
    viirs_collection = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA') \
        .filterBounds(region) \
        .filterDate(start_date, end_date)
    
    viirs_count = viirs_collection.size().getInfo()
    
    if viirs_count > 0:
        # Sort by quality flags or just get the most recent
        return ee.Image(viirs_collection.sort('system:time_start', False).first())
    
    return None

def download_satellite_image(image, region, output_path, img_dim=512):
    """
    Download a satellite image for a given region.
    
    Args:
        image (ee.Image): Image to download
        region (ee.Geometry): Region to download
        output_path (str): Path to save the downloaded image
        img_dim (int): Dimension of the output image
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        # Define visualization parameters (True Color)
        vis_params = {
            'bands': ['B4', 'B3', 'B2'],  # RGB (for Sentinel-2)
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        }
        
        # Get region coordinates for download
        region_coords = region.getInfo()['coordinates']
        
        # Get download URL
        download_url = image.getThumbURL({
            **vis_params,
            'region': region_coords,
            'dimensions': img_dim,
            'format': 'png'
        })
        
        # Download the image
        response = requests.get(download_url, timeout=90)
        response.raise_for_status()
        
        # Open and enhance image
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img_enhanced = enhance_image(img)
        
        # Save the enhanced image
        img_enhanced.save(output_path, format='PNG', optimize=True, quality=95)
        
        return True
    
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

def download_training_image(lat, lon, i, folder, buffer_m=1500, img_dim=512):
    """
    Download a single Sentinel-2 training image for a specific point.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        i (int): Image index for naming
        folder (str): Folder to save the image
        buffer_m (int): Buffer around point in meters
        img_dim (int): Image dimension
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_m)
        
        # Get current date and 12 months prior
        current_ee_date = ee.Date(datetime.datetime.now())
        start_ee_date = current_ee_date.advance(-12, 'month')
        
        # Find the least cloudy Sentinel-2 image
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(start_ee_date, current_ee_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        
        image_count = s2_collection.size().getInfo()
        
        best_image = None
        if image_count > 0:
            best_image = ee.Image(s2_collection.first())
        else:
            # Try wider cloud tolerance
            s2_collection_wider = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(region) \
                .filterDate(start_ee_date, current_ee_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)) \
                .sort('CLOUDY_PIXEL_PERCENTAGE')
            
            image_count_wider = s2_collection_wider.size().getInfo()
            if image_count_wider > 0:
                best_image = ee.Image(s2_collection_wider.first())
            else:
                print(f"  Skipping point {i+1} ({lat:.4f}, {lon:.4f}): No suitable S2 image found.")
                return False
        
        # Define visualization parameters (True Color)
        vis_params = {
            'bands': ['B4', 'B3', 'B2'],  # RGB
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        }
        
        # Get the download URL
        region_coords = region.getInfo()['coordinates']
        download_url = best_image.getThumbURL({
            **vis_params,
            'region': region_coords,
            'dimensions': img_dim,
            'format': 'png'
        })
        
        # Download the image
        response = requests.get(download_url, timeout=90)
        response.raise_for_status()
        
        # Open image with PIL
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Enhance image quality
        img_enhanced = enhance_image(img)
        
        # Save the enhanced image
        filename = f'train_image_{i+1}_{lat:.6f}_{lon:.6f}.png'
        filepath = os.path.join(folder, filename)
        img_enhanced.save(filepath, format='PNG', optimize=True, quality=95)
        
        return True
    
    except ee.EEException as ee_err:
        print(f"  EE Error downloading image for point {i+1} ({lat:.4f}, {lon:.4f}): {ee_err}")
        return False
    except requests.exceptions.RequestException as req_err:
        print(f"  Download Error for point {i+1} ({lat:.4f}, {lon:.4f}): {req_err}")
        return False
    except Exception as e:
        print(f"  Unexpected Error for point {i+1} ({lat:.4f}, {lon:.4f}): {type(e).__name__} - {e}")
        return False

def get_historical_images(lat, lon, buffer_m=1500, years_back=5, interval_months=6):
    """
    Retrieve historical satellite images for a specific location.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        buffer_m (int): Buffer around point in meters
        years_back (int): Number of years to look back
        interval_months (int): Interval between images in months
        
    Returns:
        dict: Dictionary with dates as keys and image information as values
    """
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_m)
    
    date_ranges = get_date_ranges(years_back, interval_months)
    
    historical_data = {}
    
    for i, (start_date, end_date) in enumerate(date_ranges):
        print(f"Fetching data for timeframe {i+1}/{len(date_ranges)}")
        
        # Try Sentinel-2 first (highest resolution)
        s2_image = get_best_s2_image(region, start_date, end_date)
        
        if s2_image:
            try:
                date_millis = s2_image.get('system:time_start').getInfo()
                date_str = datetime.datetime.fromtimestamp(date_millis/1000).strftime('%Y-%m-%d')
                
                # Get available bands
                available_bands = s2_image.bandNames().getInfo()
                
                # Use available bands directly instead of assuming specific names
                # Create a mapping from standard names to actual band names
                band_map = {}
                
                # Look for bands by pattern rather than exact names
                for band in available_bands:
                    band_lower = band.lower()
                    # Look for bands by explicit name matching for Sentinel-2 (not pattern-based)
                    if 'B2' in available_bands:
                        band_map['Blue'] = 'B2'
                    if 'B3' in available_bands:
                        band_map['Green'] = 'B3'
                    if 'B4' in available_bands:
                        band_map['Red'] = 'B4'
                    if 'B8' in available_bands:
                        band_map['NIR'] = 'B8'
                    if 'B11' in available_bands:
                        band_map['SWIR1'] = 'B11'
                    if 'B12' in available_bands:
                        band_map['SWIR2'] = 'B12'
                
                # Calculate spectral indices
                indices = {}
                
                # NDVI = (NIR - Red) / (NIR + Red)
                if 'NIR' in band_map and 'Red' in band_map:
                    try:
                        ndvi_img = s2_image.normalizedDifference([band_map['NIR'], band_map['Red']])
                        indices['NDVI'] = ndvi_img.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=region,
                            scale=10
                        ).get('nd').getInfo()
                    except Exception as e:
                        print(f"Error calculating NDVI: {e}")
                
                # NDWI = (Green - NIR) / (Green + NIR)
                if 'Green' in band_map and 'NIR' in band_map:
                    try:
                        ndwi_img = s2_image.normalizedDifference([band_map['Green'], band_map['NIR']])
                        indices['NDWI'] = ndwi_img.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=region,
                            scale=10
                        ).get('nd').getInfo()
                    except Exception as e:
                        print(f"Error calculating NDWI: {e}")
                
                # MNDWI = (Green - SWIR1) / (Green + SWIR1)
                if 'Green' in band_map and 'SWIR1' in band_map:
                    try:
                        mndwi_img = s2_image.normalizedDifference([band_map['Green'], band_map['SWIR1']])
                        indices['MNDWI'] = mndwi_img.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=region,
                            scale=10
                        ).get('nd').getInfo()
                    except Exception as e:
                        print(f"Error calculating MNDWI: {e}")
                
                # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
                if 'SWIR1' in band_map and 'NIR' in band_map:
                    try:
                        ndbi_img = s2_image.normalizedDifference([band_map['SWIR1'], band_map['NIR']])
                        indices['NDBI'] = ndbi_img.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=region,
                            scale=10
                        ).get('nd').getInfo()
                    except Exception as e:
                        print(f"Error calculating NDBI: {e}")
                
                # BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
                if all(k in band_map for k in ['SWIR1', 'Red', 'NIR', 'Blue']):
                    try:
                        numerator = s2_image.select(band_map['SWIR1']).add(s2_image.select(band_map['Red'])).subtract(
                            s2_image.select(band_map['NIR']).add(s2_image.select(band_map['Blue']))
                        )
                        denominator = s2_image.select(band_map['SWIR1']).add(s2_image.select(band_map['Red'])).add(
                            s2_image.select(band_map['NIR']).add(s2_image.select(band_map['Blue']))
                        )
                        bsi_img = numerator.divide(denominator)
                        indices['BSI'] = bsi_img.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=region,
                            scale=10
                        ).get(band_map['SWIR1']).getInfo()
                    except Exception as e:
                        print(f"Error calculating BSI: {e}")
                
                historical_data[date_str] = {
                    'source': 'Sentinel-2',
                    'cloud_cover': s2_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo(),
                    'image': s2_image,
                    'bands': available_bands,
                    'band_map': band_map,
                    **indices  # Add all calculated indices
                }
                continue  # Skip to next date range if Sentinel-2 found
            except Exception as e:
                print(f"Error retrieving Sentinel-2 metadata: {e}")
        
        # Try Landsat if Sentinel-2 not available
        landsat_image, collection_id = get_best_landsat_image(region, start_date, end_date)
        
        if landsat_image:
            try:
                date_millis = landsat_image.get('system:time_start').getInfo()
                date_str = datetime.datetime.fromtimestamp(date_millis/1000).strftime('%Y-%m-%d')
                
                # Get available bands
                available_bands = landsat_image.bandNames().getInfo()
                
                # Create band mapping
                band_map = {}
                for band in available_bands:
                    band_lower = band.lower()
                    if 'blue' in band_lower:
                        band_map['Blue'] = band
                    elif 'green' in band_lower:
                        band_map['Green'] = band
                    elif 'red' in band_lower and 'edge' not in band_lower:
                        band_map['Red'] = band
                    elif 'nir' in band_lower:
                        band_map['NIR'] = band
                    elif 'swir1' in band_lower or ('swir' in band_lower and '1' in band):
                        band_map['SWIR1'] = band
                    elif 'swir2' in band_lower or ('swir' in band_lower and '2' in band):
                        band_map['SWIR2'] = band
                
                # Calculate spectral indices
                indices = {}
                
                # Calculate indices (same code as for Sentinel-2)
                if 'NIR' in band_map and 'Red' in band_map:
                    try:
                        ndvi_img = landsat_image.normalizedDifference([band_map['NIR'], band_map['Red']])
                        indices['NDVI'] = ndvi_img.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=region,
                            scale=30
                        ).get('nd').getInfo()
                    except Exception as e:
                        print(f"Error calculating NDVI: {e}")
                
                # Calculate other indices similarly
                
                historical_data[date_str] = {
                    'source': 'Landsat 8-9' if collection_id == 'LANDSAT/LC09/C02/T1_L2' else 'Landsat 4-7',
                    'cloud_cover': landsat_image.get('CLOUD_COVER').getInfo(),
                    'image': landsat_image,
                    'bands': available_bands,
                    'band_map': band_map,
                    **indices
                }
                continue  # Skip to next date range if Landsat found
            except Exception as e:
                print(f"Error retrieving Landsat metadata: {e}")
        
        # Try VIIRS as last resort (lower resolution)
        viirs_image = get_best_viirs_image(region, start_date, end_date)
        
        if viirs_image:
            try:
                date_millis = viirs_image.get('system:time_start').getInfo()
                date_str = datetime.datetime.fromtimestamp(date_millis/1000).strftime('%Y-%m-%d')
                
                # Get available bands
                available_bands = viirs_image.bandNames().getInfo()
                
                # Create band mapping (same approach)
                band_map = {}
                for band in available_bands:
                    band_lower = band.lower()
                    # Map VIIRS bands to standard names
                    # (similar pattern as before)
                
                # Calculate indices where possible
                indices = {}
                # (same calculation code as above)
                
                historical_data[date_str] = {
                    'source': 'VIIRS',
                    'cloud_cover': 'N/A',  # VIIRS doesn't have cloud cover metadata
                    'image': viirs_image,
                    'bands': available_bands,
                    'band_map': band_map,
                    **indices
                }
            except Exception as e:
                print(f"Error retrieving VIIRS metadata: {e}")
        
        # Add a small sleep to avoid hitting EE rate limits
        time.sleep(0.5)
    
    print(f"Retrieved {len(historical_data)} historical images for point ({lat}, {lon})")
    return historical_data
        

def extract_historical_band_stats(historical_data, lat, lon, buffer_m=1500):
    """
    Extract statistics from historical satellite images for a specific location.
    
    Args:
        historical_data (dict): Dictionary of historical image data
        lat (float): Latitude
        lon (float): Longitude
        buffer_m (int): Buffer around point in meters
        
    Returns:
        pd.DataFrame: DataFrame with band statistics for each date
    """
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_m)
    
    all_stats = []
    
    for date_str, data in tqdm(historical_data.items(), desc="Extracting band statistics"):
        try:
            image = data['image']
            source = data['source']
            
            # Create a dictionary to map standard band names to image-specific band names
            if 'band_map' in data:
                bands_dict = data['band_map']  # Use the existing band mapping
            elif 'band_names' in data and 'bands' in data:
                bands_dict = dict(zip(data['band_names'], data['bands']))
            else:
                # Try to create a band mapping from the image directly
                try:
                    image_bands = image.bandNames().getInfo()
                    bands_dict = {}
                    # Add explicit Sentinel-2 mapping
                    if 'B2' in image_bands:
                        bands_dict = {
                            'Blue': 'B2',
                            'Green': 'B3',
                            'Red': 'B4',
                            'NIR': 'B8',
                            'SWIR1': 'B11',
                            'SWIR2': 'B12'
                        }
                    # Add Landsat 8/9 mapping
                    elif 'SR_B2' in image_bands:
                        bands_dict = {
                            'Blue': 'SR_B2',
                            'Green': 'SR_B3',
                            'Red': 'SR_B4',
                            'NIR': 'SR_B5',
                            'SWIR1': 'SR_B6',
                            'SWIR2': 'SR_B7'
                        }
                    # Add Landsat 4-7 mapping
                    elif 'SR_B1' in image_bands:
                        bands_dict = {
                            'Blue': 'SR_B1',
                            'Green': 'SR_B2',
                            'Red': 'SR_B3',
                            'NIR': 'SR_B4',
                            'SWIR1': 'SR_B5',
                            'SWIR2': 'SR_B7'
                        }
                except Exception as band_err:
                    print(f"Warning: Could not get bands for date {date_str}: {band_err}")
                    bands_dict = {}  # Empty dict as fallback
            
            # Add spectral indices to the image
            image_with_indices = get_spectral_indices(image, bands_dict)
            
            # Get statistics from the region
            stats = image_with_indices.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.stdDev(), None, True
                ).combine(
                    ee.Reducer.minMax(), None, True
                ),
                geometry=region,
                scale=10 if source == 'Sentinel-2' else (30 if 'Landsat' in source else 500),
                maxPixels=1e9
            ).getInfo()
            
            # Create a record for this date
            record = {
                'date': date_str,
                'source': source
            }
            
            # Add band statistics to the record
            if 'bands' in data:
                for band_name in data['bands']:
                    if f"{band_name}_mean" in stats:
                        record[f"{band_name}_mean"] = stats[f"{band_name}_mean"]
                        record[f"{band_name}_stdDev"] = stats[f"{band_name}_stdDev"]
                        record[f"{band_name}_min"] = stats[f"{band_name}_min"]
                        record[f"{band_name}_max"] = stats[f"{band_name}_max"]
            else:
                # If 'bands' is missing, try to use the bands from bands_dict
                for std_name, band_name in bands_dict.items():
                    if f"{band_name}_mean" in stats:
                        record[f"{std_name}_mean"] = stats[f"{band_name}_mean"]
                        record[f"{std_name}_stdDev"] = stats[f"{band_name}_stdDev"]
                        record[f"{std_name}_min"] = stats[f"{band_name}_min"]
                        record[f"{std_name}_max"] = stats[f"{band_name}_max"]
            
            # Add spectral indices to the record
            for index_name in ['NDVI', 'NDWI', 'MNDWI', 'BSI', 'NDBI', 'NDTI']:
                if f"{index_name}_mean" in stats:
                    record[f"{index_name}_mean"] = stats[f"{index_name}_mean"]
                    record[f"{index_name}_stdDev"] = stats[f"{index_name}_stdDev"]
                    record[f"{index_name}_min"] = stats[f"{index_name}_min"]
                    record[f"{index_name}_max"] = stats[f"{index_name}_max"]
            
            all_stats.append(record)
            
        except Exception as e:
            print(f"Error extracting statistics for date {date_str}: {e}")
            continue
    
    # Create DataFrame from all records
    if all_stats:
        df = pd.DataFrame(all_stats)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no stats were extracted

def download_training_images(coordinates, output_folder, img_dim=512, buffer_m=1500):
    """
    Download Sentinel-2 images for the specified coordinates.
    
    Args:
        coordinates (list): List of (lat, lon) tuples
        output_folder (str): Folder to save images
        img_dim (int): Image dimension
        buffer_m (int): Buffer around points in meters
        
    Returns:
        bool: True if at least some images were downloaded, False otherwise
    """
    if not coordinates:
        print("Error: No coordinates provided for image download.")
        return False

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    print(f"\nDownloading {len(coordinates)} images for training/labeling...")
    print(f"Image dimensions: {img_dim}x{img_dim}, Buffer: {buffer_m}m")

    success_count = 0

    with tqdm(total=len(coordinates), desc="Downloading Images", unit="image") as pbar:
        for i, (lat, lon) in enumerate(coordinates):
            point_start_time = time.time()
            
            success = download_training_image(lat, lon, i, output_folder, buffer_m, img_dim)
            if success:
                success_count += 1
            
            # Dynamically adjust sleep based on request time
            elapsed_time = time.time() - point_start_time
            sleep_time = max(0, 1.2 - elapsed_time)  # Aim slightly slower than 1 req/sec
            time.sleep(sleep_time)
            
            pbar.update(1)  # Update progress bar

    print(f"\nSuccessfully downloaded {success_count} out of {len(coordinates)} images.")
    return success_count > 0  # Return True if any images were successfully downloaded

def calculate_temporal_features(historical_stats_df):
    """
    Calculate temporal features from a historical statistics DataFrame.
    
    Args:
        historical_stats_df (pd.DataFrame): DataFrame with historical band statistics
        
    Returns:
        dict: Dictionary of temporal features
    """
    if historical_stats_df.empty or len(historical_stats_df) < 2:
        return {}  # Need at least 2 time points for trends
    
    temporal_features = {}
    
    # Sort by date to ensure correct temporal order
    df = historical_stats_df.sort_values('date')
    
    # Calculate trends for key indices
    for index in ['NDVI', 'NDWI', 'MNDWI', 'BSI', 'NDBI']:
        col_name = f"{index}_mean"
        if col_name in df.columns:
            # Convert dates to ordinal values for regression
            date_ordinals = np.array([(d - df['date'].min()).days for d in df['date']])
            values = df[col_name].values
            
            # Skip if we have too many NaN values
            valid_mask = ~np.isnan(values)
            if np.sum(valid_mask) < 2:
                continue
                
            # Linear regression to get slope (trend)
            try:
                from scipy import stats as scipy_stats
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                    date_ordinals[valid_mask], values[valid_mask]
                )
                temporal_features[f"{index}_trend"] = slope
                temporal_features[f"{index}_trend_r_squared"] = r_value**2
                temporal_features[f"{index}_trend_p_value"] = p_value
            except Exception as e:
                print(f"Error calculating trend for {index}: {e}")
    
    # Calculate variability metrics
    for index in ['NDVI', 'NDWI', 'MNDWI', 'BSI', 'NDBI']:
        col_name = f"{index}_mean"
        if col_name in df.columns:
            values = df[col_name].dropna().values
            if len(values) >= 2:
                temporal_features[f"{index}_variability"] = np.std(values)
                temporal_features[f"{index}_range"] = np.max(values) - np.min(values)
    
    # Calculate seasonal metrics if we have enough data points
    if len(df) >= 6:
        # Look for seasonality in NDVI (applicable for natural vegetation)
        if 'NDVI_mean' in df.columns:
            try:
                # Simple approach: compare variability in different quarters/seasons
                df['quarter'] = df['date'].dt.quarter
                seasonal_stats = df.groupby('quarter')['NDVI_mean'].agg(['mean', 'std'])
                
                if not seasonal_stats.empty and not seasonal_stats['mean'].isna().all():
                    # Calculate seasonal variation
                    temporal_features['NDVI_seasonal_variation'] = seasonal_stats['mean'].max() - seasonal_stats['mean'].min()
                    # Calculate consistency of seasonality
                    temporal_features['NDVI_seasonal_consistency'] = seasonal_stats['std'].mean()
            except Exception as e:
                print(f"Error calculating seasonal metrics: {e}")
    
    # Calculate change detection metrics
    # First and last observation changes
    if len(df) >= 2:
        for index in ['NDVI', 'NDWI', 'MNDWI', 'BSI', 'NDBI']:
            col_name = f"{index}_mean"
            if col_name in df.columns:
                first_valid = df[col_name].first_valid_index()
                last_valid = df[col_name].last_valid_index()
                
                if first_valid is not None and last_valid is not None and first_valid != last_valid:
                    first_value = df.loc[first_valid, col_name]
                    last_value = df.loc[last_valid, col_name]
                    
                    if not np.isnan(first_value) and not np.isnan(last_value):
                        # Total change
                        temporal_features[f"{index}_total_change"] = last_value - first_value
                        # Percent change
                        if first_value != 0:
                            temporal_features[f"{index}_percent_change"] = (last_value - first_value) / abs(first_value) * 100
    
    # Calculate rapid change metrics - detect sudden changes that might indicate sand mining
    if len(df) >= 3:
        for index in ['NDVI', 'NDWI', 'MNDWI', 'BSI']:
            col_name = f"{index}_mean"
            if col_name in df.columns:
                # Get the differences between consecutive observations
                differences = df[col_name].diff().dropna().values
                
                if len(differences) > 0:
                    # Maximum rate of change (positive or negative)
                    temporal_features[f"{index}_max_change_rate"] = np.max(np.abs(differences))
                    
                    # Check for sudden drops (potentially indicating disturbances)
                    if index in ['NDVI', 'NDWI']:  # These typically decrease with disturbance
                        sudden_drops = np.where(differences < -0.1)[0]  # Threshold can be adjusted
                        temporal_features[f"{index}_sudden_drops"] = len(sudden_drops)
                    
                    # Check for sudden increases (potentially indicating disturbances)
                    if index in ['BSI', 'NDBI']:  # These typically increase with disturbance
                        sudden_increases = np.where(differences > 0.1)[0]  # Threshold can be adjusted
                        temporal_features[f"{index}_sudden_increases"] = len(sudden_increases)
    
    return temporal_features

def get_latest_imagery_date(region):
    """
    Determine the latest available imagery date with acceptable cloud coverage.
    
    Args:
        region (ee.Geometry): Region of interest
        
    Returns:
        tuple: (date_str, source_info)
    """
    try:
        # Use Python's datetime for current date calculation
        today_dt = datetime.datetime.now()
        today_ee = ee.Date(today_dt)  # Convert to EE Date object
        latest_date_info = {'date': None, 'source': 'None', 'cloud': 999}
        
        # Check Sentinel-2 Harmonized (last 6 months)
        start_s2_dt = today_dt - datetime.timedelta(days=180)
        start_s2_ee = ee.Date(start_s2_dt)
        
        s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(start_s2_ee, today_ee) \
            .sort('system:time_start', False)  # Sort descending by date
        
        # Iterate through images to find the best one (least cloudy recent)
        image_list_s2 = s2_col.toList(20)  # Check the latest 20 images first
        for i in range(image_list_s2.size().getInfo()):
            img = ee.Image(image_list_s2.get(i))
            cloud_cover = img.get('CLOUDY_PIXEL_PERCENTAGE')
            
            # Need to getInfo() to check the value client-side
            if cloud_cover is not None:
                # Add error handling for getInfo() calls
                try:
                    cloud_cover_val = cloud_cover.getInfo()
                    if cloud_cover_val is not None and cloud_cover_val < 35:
                        date_millis = img.get('system:time_start').getInfo()
                        latest_date_info = {
                            'date': datetime.datetime.fromtimestamp(date_millis/1000),
                            'source': 'Sentinel-2',
                            'cloud': cloud_cover_val
                        }
                        break  # Found a good recent one
                except ee.EEException as getinfo_err:
                    print(f"Warning: Error getting cloud cover info: {getinfo_err}")
                    continue  # Skip image if metadata cannot be retrieved
        
        # If no recent good S2, check Landsat 9 (last 12 months)
        if latest_date_info['date'] is None:
            start_l9_dt = today_dt - datetime.timedelta(days=365)
            start_l9_ee = ee.Date(start_l9_dt)
            
            l9_col = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
                .filterBounds(region) \
                .filterDate(start_l9_ee, today_ee) \
                .sort('system:time_start', False)
            
            image_list_l9 = l9_col.toList(10)
            for i in range(image_list_l9.size().getInfo()):
                img = ee.Image(image_list_l9.get(i))
                cloud_cover = img.get('CLOUD_COVER')  # Different metadata name
                if cloud_cover is not None:
                    try:
                        cloud_cover_val = cloud_cover.getInfo()
                        if cloud_cover_val is not None and cloud_cover_val < 40:
                            date_millis = img.get('system:time_start').getInfo()
                            ld_date = datetime.datetime.fromtimestamp(date_millis/1000)
                            latest_date_info = {
                                'date': ld_date,
                                'source': 'Landsat 9',
                                'cloud': cloud_cover_val
                            }
                            break  # Found a suitable Landsat image
                    except ee.EEException as getinfo_err:
                        print(f"Warning: Error getting cloud cover info for Landsat: {getinfo_err}")
                        continue
        
        if latest_date_info['date']:
            return latest_date_info['date'].strftime('%Y-%m-%d'), f"{latest_date_info['source']} ({latest_date_info['cloud']:.1f}% cloud)"
        else:
            # Fallback if absolutely nothing found
            return today_dt.strftime('%Y-%m-%d'), "No recent imagery found"  # Use Python date
    
    except ee.EEException as e:
        print(f"EE Error determining latest imagery date: {e}")
        return datetime.datetime.now().strftime('%Y-%m-%d'), "Error - EE Exception"
    except Exception as e:
        print(f"Error determining latest imagery date: {type(e).__name__} - {e}")
        return datetime.datetime.now().strftime('%Y-%m-%d'), "Error - General Exception"

def analyze_point(lat, lon, latest_date_str, model, scaler, feature_extractor, use_historical=True, 
                 years_back=5, img_dim=800, buffer_m=1500, temp_folder='temp'):
    """
    Download image, extract features, and predict probability for a single point.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        latest_date_str (str): Latest date for imagery search
        model: Trained model
        scaler: Feature scaler
        feature_extractor: Function to extract features from an image
        use_historical (bool): Whether to use historical features
        years_back (int): Number of years to look back for historical data
        img_dim (int): Image dimension
        buffer_m (int): Buffer around point in meters
        temp_folder (str): Temporary folder path
        
    Returns:
        dict: Result dictionary with probability and metadata
    """
    temp_file = None  # Initialize temp_file to None
    
    try:
        # Create geometry
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_m)
        
        # Define date range
        try:
            latest_dt = datetime.datetime.strptime(latest_date_str, '%Y-%m-%d')
        except ValueError:
            print(f"Warning: Invalid latest_date_str '{latest_date_str}'. Using current date.")
            latest_dt = datetime.datetime.now()
            latest_date_str = latest_dt.strftime('%Y-%m-%d')  # Update string
        
        start_dt_narrow = latest_dt - datetime.timedelta(days=15)
        end_dt = latest_dt + datetime.timedelta(days=1)  # Include target day
        
        latest_date_ee = ee.Date(latest_dt)
        start_date_ee_narrow = ee.Date(start_dt_narrow)
        end_date_ee = ee.Date(end_dt)
        
        # Get best image
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(start_date_ee_narrow, end_date_ee) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')  # Least cloudy first
        
        image_to_use = s2_collection.first()  # Get the best available image
        
        if image_to_use is None:
            # Try broader date range
            start_dt_broad = latest_dt - datetime.timedelta(days=45)
            start_date_ee_broad = ee.Date(start_dt_broad)
            
            s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(region) \
                .filterDate(start_date_ee_broad, end_date_ee) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
                .sort('CLOUDY_PIXEL_PERCENTAGE')
            image_to_use = s2_collection.first()
        
        # If still no S2 image, try Landsat
        if image_to_use is None:
            landsat_image, _ = get_best_landsat_image(region, start_date_ee_broad, end_date_ee, max_cloud_cover=60)
            
            if landsat_image is not None:
                image_to_use = landsat_image
                # Adjust visualization parameters for Landsat
                vis_params = {
                    'bands': ['SR_B4', 'SR_B3', 'SR_B2'],  # RGB for Landsat
                    'min': 0,
                    'max': 3000,
                    'gamma': 1.4
                }
            else:
                return None  # Skip this point if no image available
        else:
            # Default visualization for Sentinel-2
            vis_params = {
                'bands': ['B4', 'B3', 'B2'],  # RGB
                'min': 0,
                'max': 3000,
                'gamma': 1.4
            }
        
        # Get image date
        try:
            actual_img_date_millis = image_to_use.get('system:time_start').getInfo()
            actual_img_date = datetime.datetime.fromtimestamp(actual_img_date_millis/1000).strftime('%Y-%m-%d')
        except ee.EEException as date_err:
            print(f"Warning: Could not get image date: {date_err}")
            actual_img_date = "Unknown"  # Fallback date
        
        # Get download URL
        try:
            region_info = region.getInfo()
            if not region_info or 'coordinates' not in region_info:
                return None
            region_coords = region_info['coordinates']
        except ee.EEException as region_err:
            print(f"Warning: Error getting region info ({lat:.4f}, {lon:.4f}): {region_err}")
            return None
        
        download_url = image_to_use.getThumbURL({
            **vis_params,
            'region': region_coords,
            'dimensions': img_dim,
            'format': 'png'
        })
        
        # Download image
        response = requests.get(download_url, timeout=90)  # Extended timeout
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Save temporarily for feature extraction
        os.makedirs(temp_folder, exist_ok=True)
        temp_filename = f'map_temp_{lat:.6f}_{lon:.6f}.png'
        temp_file = os.path.join(temp_folder, temp_filename)
        img.save(temp_file)
        
        # Extract features from current image
        image_features = feature_extractor(temp_file)
        
        if not image_features:
            return None
        
        # Add historical features if enabled
        historical_features = {}
        if use_historical and years_back > 0:
            # Get historical data
            historical_data = get_historical_images(lat, lon, buffer_m=buffer_m, years_back=years_back)
            
            if historical_data:
                # Extract statistics from historical data
                historical_stats = extract_historical_band_stats(historical_data, lat, lon, buffer_m=buffer_m)
                
                if not historical_stats.empty:
                    # Calculate temporal features
                    historical_features = calculate_temporal_features(historical_stats)
        
        # Combine features
        all_features = {**image_features, **historical_features}
        
        # Convert to vector for prediction
        feature_vector = []
        
        if hasattr(model, 'feature_names_in_'):
            # Use model's feature names if available (scikit-learn 1.0+)
            for feature_name in model.feature_names_in_:
                if feature_name in all_features:
                    feature_vector.append(all_features[feature_name])
                else:
                    feature_vector.append(0)  # Use 0 for missing features
        else:
            # Fall back to all features in arbitrary order
            feature_vector = list(all_features.values())
        
        # Convert to numpy array
        X = np.array([feature_vector])
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict probability
        probability = model.predict_proba(X_scaled)[0][1]
        
        # Determine classification
        if probability >= 0.7:
            classification = 'Sand Mining Likely'
        elif probability >= 0.5:
            classification = 'Possible Sand Mining'
        else:
            classification = 'No Sand Mining Likely'
        
        # Get feature importance for explanation if model supports it
        top_features = []
        if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
            importances = model.feature_importances_
            feature_names = model.feature_names_in_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            # Get top 3 features that are present in this point
            for i in indices:
                feature_name = feature_names[i]
                if feature_name in all_features and len(top_features) < 3:
                    top_features.append((feature_name, importances[i]))
        
        # Build result
        result = {
            'latitude': lat,
            'longitude': lon,
            'probability': probability,
            'classification': classification,
            'image_date': actual_img_date
        }
        
        # Add top features info if available
        if top_features:
            result['top_features'] = ', '.join([f"{name} ({imp:.3f})" for name, imp in top_features])
        
        return result
    
    except ee.EEException as ee_err:
        ee_err_str = str(ee_err).lower()
        # Implement backoff for common errors
        if any(term in ee_err_str for term in ['quota', 'rate limit', 'user memory limit', 'computation timed out', 'backend error', 'too many requests']):
            wait_time = min(64, 2**(np.random.randint(1, 7)))  # Random backoff 2-64s
            time.sleep(wait_time)
        return None
    except requests.exceptions.RequestException:
        time.sleep(1)  # Shorter wait after download errors
        return None
    except Exception as e:
        print(f"Unexpected error analyzing point ({lat:.4f}, {lon:.4f}): {type(e).__name__} - {e}")
        return None
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass  # Ignore errors removing temp file