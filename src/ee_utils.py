#!/usr/bin/env python3
"""
Earth Engine utilities for Sand Mining Detection Tool - Simplified without historical imagery.
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

def analyze_point(lat, lon, latest_date_str, model, scaler, feature_extractor, 
                 img_dim=800, buffer_m=1500, temp_folder='temp'):
    """
    Download image, extract enhanced features (including from highlighted areas), and predict probability.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        latest_date_str (str): Latest date for imagery search
        model: Trained model
        scaler: Feature scaler
        feature_extractor: Function to extract enhanced features from an image
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
            landsat_image, landsat_collection = get_best_landsat_image(
                region, start_date_ee_broad, end_date_ee, max_cloud_cover=50
            )
            if landsat_image is not None:
                image_to_use = landsat_image
                # Update visualization parameters for Landsat
                vis_params = {
                    'bands': ['SR_B4', 'SR_B3', 'SR_B2'],  # RGB for Landsat
                    'min': 7000,
                    'max': 12000,
                    'gamma': 1.4
                }
            else:
                return {
                    'probability': 0.0,
                    'error': 'No suitable satellite imagery found',
                    'lat': lat,
                    'lon': lon,
                    'image_date': 'N/A',
                    'cloud_cover': 'N/A'
                }
        else:
            # Use Sentinel-2 visualization parameters
            vis_params = {
                'bands': ['B4', 'B3', 'B2'],  # RGB for Sentinel-2
                'min': 0,
                'max': 3000,
                'gamma': 1.4
            }
        
        # Get image metadata
        try:
            image_date_millis = image_to_use.get('system:time_start').getInfo()
            image_date = datetime.datetime.fromtimestamp(image_date_millis/1000).strftime('%Y-%m-%d')
            
            # Try to get cloud cover (different property names for different sensors)
            cloud_cover = None
            try:
                cloud_cover = image_to_use.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()  # Sentinel-2
            except:
                try:
                    cloud_cover = image_to_use.get('CLOUD_COVER').getInfo()  # Landsat
                except:
                    cloud_cover = 'Unknown'
        except Exception as meta_err:
            print(f"Warning: Error getting image metadata: {meta_err}")
            image_date = 'Unknown'
            cloud_cover = 'Unknown'
        
        # Ensure temp folder exists
        os.makedirs(temp_folder, exist_ok=True)
        
        # Create temporary file
        temp_file = os.path.join(temp_folder, f'temp_analysis_{lat:.6f}_{lon:.6f}.png')
        
        # Download image
        try:
            region_coords = region.getInfo()['coordinates']
            download_url = image_to_use.getThumbURL({
                **vis_params,
                'region': region_coords,
                'dimensions': img_dim,
                'format': 'png'
            })
            
            response = requests.get(download_url, timeout=90)
            response.raise_for_status()
            
            # Process image
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            img_enhanced = enhance_image(img)
            img_enhanced.save(temp_file, format='PNG', optimize=True, quality=95)
            
        except Exception as download_err:
            return {
                'probability': 0.0,
                'error': f'Image download failed: {download_err}',
                'lat': lat,
                'lon': lon,
                'image_date': image_date,
                'cloud_cover': cloud_cover
            }
        
        # Extract features using the provided feature extractor
        try:
            features = feature_extractor(temp_file)
            if features is None or len(features) == 0:
                return {
                    'probability': 0.0,
                    'error': 'Feature extraction failed',
                    'lat': lat,
                    'lon': lon,
                    'image_date': image_date,
                    'cloud_cover': cloud_cover
                }
            
            # Convert features dict to list if needed
            if isinstance(features, dict):
                feature_values = list(features.values())
            else:
                feature_values = features
            
            # Scale features
            features_scaled = scaler.transform([feature_values])
            
            # Predict probability
            probability = model.predict_proba(features_scaled)[0][1]  # Probability of positive class
            
            return {
                'probability': float(probability),
                'error': None,
                'latitude': lat,  # Changed from 'lat' to 'latitude' for consistency
                'longitude': lon,  # Changed from 'lon' to 'longitude' for consistency
                'image_date': image_date,
                'cloud_cover': cloud_cover,
                'features_count': len(feature_values)
            }
            
        except Exception as prediction_err:
            return {
                'probability': 0.0,
                'error': f'Prediction failed: {prediction_err}',
                'latitude': lat,
                'longitude': lon,
                'image_date': image_date,
                'cloud_cover': cloud_cover
            }
    
    except ee.EEException as ee_err:
        return {
            'probability': 0.0,
            'error': f'Earth Engine error: {ee_err}',
            'latitude': lat,
            'longitude': lon,
            'image_date': 'N/A',
            'cloud_cover': 'N/A'
        }
    
    except Exception as general_err:
        return {
            'probability': 0.0,
            'error': f'General error: {general_err}',
            'latitude': lat,
            'longitude': lon,
            'image_date': 'N/A',
            'cloud_cover': 'N/A'
        }
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as cleanup_err:
                print(f"Warning: Could not delete temporary file {temp_file}: {cleanup_err}")

def batch_analyze_points(coordinates_list, latest_date_str, model, scaler, 
                        feature_extractor, img_dim=800, buffer_m=1500, 
                        temp_folder='temp', max_workers=3):
    """
    Analyze multiple points in batch with progress tracking.
    
    Args:
        coordinates_list (list): List of (lat, lon) tuples
        latest_date_str (str): Latest date for imagery search
        model: Trained model
        scaler: Feature scaler
        feature_extractor: Function to extract features from images
        img_dim (int): Image dimension
        buffer_m (int): Buffer around points in meters
        temp_folder (str): Temporary folder path
        max_workers (int): Maximum number of concurrent workers
        
    Returns:
        list: List of result dictionaries
    """
    results = []
    
    print(f"\nAnalyzing {len(coordinates_list)} points...")
    print(f"Image dimensions: {img_dim}x{img_dim}, Buffer: {buffer_m}m")
    
    with tqdm(total=len(coordinates_list), desc="Analyzing Points", unit="point") as pbar:
        for lat, lon in coordinates_list:
            try:
                result = analyze_point(
                    lat, lon, latest_date_str, model, scaler, feature_extractor,
                    img_dim, buffer_m, temp_folder
                )
                results.append(result)
                
                # Add small delay to avoid overwhelming the API
                time.sleep(0.5)
                
            except Exception as e:
                error_result = {
                    'probability': 0.0,
                    'error': f'Analysis failed: {e}',
                    'latitude': lat,
                    'longitude': lon,
                    'image_date': 'N/A',
                    'cloud_cover': 'N/A'
                }
                results.append(error_result)
            
            pbar.update(1)
    
    return results

def cleanup_temp_files(temp_folder='temp'):
    """
    Clean up temporary files created during analysis.
    
    Args:
        temp_folder (str): Temporary folder path
    """
    try:
        if os.path.exists(temp_folder):
            for filename in os.listdir(temp_folder):
                if filename.startswith('temp_analysis_') and filename.endswith('.png'):
                    file_path = os.path.join(temp_folder, filename)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Warning: Could not delete {file_path}: {e}")
            
            # Remove temp folder if empty
            try:
                os.rmdir(temp_folder)
            except OSError:
                pass  # Folder not empty or doesn't exist
                
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def validate_coordinates(coordinates_list):
    """
    Validate a list of coordinates.
    
    Args:
        coordinates_list (list): List of (lat, lon) tuples
        
    Returns:
        tuple: (valid_coords, invalid_coords)
    """
    valid_coords = []
    invalid_coords = []
    
    for coord in coordinates_list:
        try:
            lat, lon = coord
            lat, lon = float(lat), float(lon)
            
            # Check if coordinates are within valid ranges
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                valid_coords.append((lat, lon))
            else:
                invalid_coords.append((lat, lon, "Out of valid range"))
                
        except (ValueError, TypeError):
            invalid_coords.append((coord, "Invalid format"))
        except Exception as e:
            invalid_coords.append((coord, f"Error: {e}"))
    
    return valid_coords, invalid_coords