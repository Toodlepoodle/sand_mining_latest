#!/usr/bin/env python3
"""
Probability mapping module for the Sand Mining Detection Tool.
"""

import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import requests
import io
from tqdm import tqdm
import joblib
import shutil
import json
from datetime import datetime, timedelta
import ee

from src import config
from src import ee_utils
from src import features
from src.model import load_model_and_metadata
from src.utils import load_shapefile_and_get_points, generate_interactive_map, create_summary_plot, clean_temp_dir

class SandMiningProbabilityMapper:
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize the probability mapper.
        
        Args:
            model_path (str, optional): Path to the model file
            scaler_path (str, optional): Path to the scaler file
        """
        # Ensure EE is initialized
        self.ee_initialized = ee_utils.initialize_ee()
        if not self.ee_initialized:
            raise Exception("Earth Engine could not be initialized for Mapper. Exiting.")
        
        # Set output directories
        self.output_folder = config.PROBABILITY_MAPS_DIR
        self.temp_folder = config.TEMP_DIR
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.temp_folder, exist_ok=True)
        
        print(f"Probability maps will be saved in: {self.output_folder}")
        print(f"Temporary processing files in: {self.temp_folder}")
        
        # Determine model and scaler paths
        model_to_use = model_path if model_path else config.DEFAULT_MODEL_FILE
        scaler_to_use = scaler_path if scaler_path else config.DEFAULT_SCALER_FILE
        
        # Load model and scaler
        model, scaler, feature_names = load_model_and_metadata()
        
        if model is None or scaler is None:
            raise Exception("Failed to load model or scaler. Please ensure training completed successfully.")
        
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        
        # Load feature importance for explanation if available
        self.feature_importance = {}
        if os.path.exists(config.DEFAULT_FEATURE_IMPORTANCE_FILE):
            try:
                with open(config.DEFAULT_FEATURE_IMPORTANCE_FILE, 'r') as f:
                    self.feature_importance = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load feature importance file: {e}")
    
    def get_latest_imagery_date(self, region):
        """
        Determine the latest available imagery date with acceptable cloud coverage.
        
        Args:
            region (ee.Geometry): Region of interest
            
        Returns:
            tuple: (date_str, source_info)
        """
        if not self.ee_initialized:
            return datetime.now().strftime('%Y-%m-%d'), "Error - EE Not Init"
        
        try:
            # Use Python's datetime for current date calculation
            today_dt = datetime.now()
            today_ee = ee.Date(today_dt)  # Convert to EE Date object
            latest_date_info = {'date': None, 'source': 'None', 'cloud': 999}
            
            # Check Sentinel-2 Harmonized (last 6 months)
            start_s2_dt = today_dt - timedelta(days=180)
            start_s2_ee = ee.Date(start_s2_dt)
            
            s2_col = ee.ImageCollection(config.S2_COLLECTION) \
                .filterBounds(region) \
                .filterDate(start_s2_ee, today_ee) \
                .sort('system:time_start', False)  # Sort descending by date
            
            # Iterate through images to find the best one (least cloudy recent)
            image_list_s2 = s2_col.toList(20)  # Check the latest 20 images first
            for i in range(image_list_s2.size().getInfo()):
                img = ee.Image(image_list_s2.get(i))
                cloud_cover = img.get('CLOUDY_PIXEL_PERCENTAGE')
                
                try:
                    cloud_cover_val = cloud_cover.getInfo()
                    if cloud_cover_val is not None and cloud_cover_val < 35:
                        date_millis = img.get('system:time_start').getInfo()
                        latest_date_info = {
                            'date': datetime.fromtimestamp(date_millis/1000),
                            'source': 'Sentinel-2',
                            'cloud': cloud_cover_val
                        }
                        break  # Found a good recent one
                except ee.EEException as getinfo_err:
                    print(f"Warning: Error getting cloud cover info: {getinfo_err}")
                    continue  # Skip image if metadata cannot be retrieved
            
            # If no recent good S2, check Landsat 9 (last 12 months)
            if latest_date_info['date'] is None:
                start_l9_dt = today_dt - timedelta(days=365)
                start_l9_ee = ee.Date(start_l9_dt)
                
                l9_col = ee.ImageCollection(config.L89_COLLECTION) \
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
                                ld_date = datetime.fromtimestamp(date_millis/1000)
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
            return datetime.now().strftime('%Y-%m-%d'), "Error - EE Exception"
        except Exception as e:
            print(f"Error determining latest imagery date: {type(e).__name__} - {e}")
            return datetime.now().strftime('%Y-%m-%d'), "Error - General Exception"
    
    def analyze_point(self, lat, lon, latest_date_str, use_historical=True, years_back=5, img_dim=config.DEFAULT_MAP_IMAGE_DIM, buffer_m=config.DEFAULT_BUFFER_METERS):
        """
        Download image, extract features, and predict probability for a single point.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            latest_date_str (str): Latest date for imagery search
            use_historical (bool): Whether to use historical features
            years_back (int): Number of years to look back for historical data
            img_dim (int): Dimension of downloaded images
            buffer_m (int): Buffer around point in meters
            
        Returns:
            dict: Analysis results including probability, or None if error
        """
        if not self.ee_initialized:
            return None
        
        temp_file = None  # Initialize temp_file to None
        
        try:
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(buffer_m)
            
            # Define date range around the latest identified date
            try:
                latest_dt = datetime.strptime(latest_date_str, '%Y-%m-%d')
            except ValueError:
                print(f"Warning: Invalid latest_date_str '{latest_date_str}'. Using current date.")
                latest_dt = datetime.now()
                latest_date_str = latest_dt.strftime('%Y-%m-%d')  # Update string
            
            start_dt_narrow = latest_dt - timedelta(days=15)
            end_dt = latest_dt + timedelta(days=1)  # Include target day
            
            latest_date_ee = ee.Date(latest_dt)
            start_date_ee_narrow = ee.Date(start_dt_narrow)
            end_date_ee = ee.Date(end_dt)
            
            # Get the best image within this narrow window
            best_image = ee_utils.get_best_s2_image(region, start_date_ee_narrow, end_date_ee)
            
            if best_image is None:
                # If nothing found in the narrow window, broaden search slightly
                start_dt_broad = latest_dt - timedelta(days=45)
                start_date_ee_broad = ee.Date(start_dt_broad)
                
                best_image = ee_utils.get_best_s2_image(region, start_date_ee_broad, end_date_ee, max_cloud_cover=50)
            
            # Try Landsat if Sentinel-2 not available
            if best_image is None:
                landsat_image, collection_id = ee_utils.get_best_landsat_image(
                    region, start_date_ee_broad, end_date_ee, max_cloud_cover=60
                )
                
                if landsat_image is not None:
                    best_image = landsat_image
                    # Adjust parameters for Landsat visualization
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
            
            # Get actual date of the image used
            try:
                actual_img_date_millis = best_image.get('system:time_start').getInfo()
                actual_img_date = datetime.fromtimestamp(actual_img_date_millis/1000).strftime('%Y-%m-%d')
            except ee.EEException as date_err:
                tqdm.write(f"Warning: Could not get image date: {date_err}")
                actual_img_date = "Unknown"  # Fallback date
            
            # Get download URL
            try:
                region_info = region.getInfo()
                if not region_info or 'coordinates' not in region_info:
                    return None
                region_coords = region_info['coordinates']
            except ee.EEException as region_err:
                tqdm.write(f"Warning: Error getting region info ({lat:.4f}, {lon:.4f}): {region_err}")
                return None
            
            download_url = best_image.getThumbURL({
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
            os.makedirs(self.temp_folder, exist_ok=True)
            temp_filename = f'map_temp_{lat:.6f}_{lon:.6f}.png'
            temp_file = os.path.join(self.temp_folder, temp_filename)
            img.save(temp_file)
            
            # Extract features from the image
            image_features = features.extract_all_features(temp_file)
            
            if not image_features:
                return None
            
            # Add historical features if enabled
            historical_features = {}
            if use_historical and years_back > 0:
                # Get historical data
                historical_data = ee_utils.get_historical_images(lat, lon, buffer_m=buffer_m, years_back=years_back)
                
                if historical_data:
                    # Extract statistics from historical data
                    historical_stats = ee_utils.extract_historical_band_stats(historical_data, lat, lon, buffer_m=buffer_m)
                    
                    if not historical_stats.empty:
                        # Calculate trend features for important indices
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
                                        try:
                                            from scipy import stats as scipy_stats
                                            slope, _, _, _, _ = scipy_stats.linregress(x[valid], y[valid])
                                            historical_features[f"{index}_trend"] = slope
                                        except Exception as e:
                                            print(f"Error calculating trend for {index}: {e}")
            
            # Combine features
            all_features = {**image_features, **historical_features}
            
            # Convert to vector matching training format
            if self.feature_names:
                # Use ordered feature names from training
                feature_vector = []
                for feature_name in self.feature_names:
                    if feature_name in all_features:
                        feature_vector.append(all_features[feature_name])
                    else:
                        feature_vector.append(0)  # Use 0 for missing features
            else:
                # Fall back to all features in arbitrary order if feature_names not available
                feature_vector = list(all_features.values())
            
            # Convert to numpy array
            X = np.array([feature_vector])
            
            # Scale features using the loaded scaler
            X_scaled = self.scaler.transform(X)
            
            # Predict probability
            probability = self.model.predict_proba(X_scaled)[0][1]
            
            # Determine classification
            if probability >= 0.7:
                classification = 'Sand Mining Likely'
            elif probability >= 0.5:
                classification = 'Possible Sand Mining'
            else:
                classification = 'No Sand Mining Likely'
            
            # Find top contributing features
            top_features = []
            if self.feature_importance and self.feature_names:
                # Get top 3 most important features that are present in this point
                sorted_features = sorted(
                    self.feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Only include features where we have an actual value
                valid_top_features = []
                for feature_name, importance in sorted_features:
                    if feature_name in all_features:
                        valid_top_features.append((feature_name, importance))
                        if len(valid_top_features) >= 3:
                            break
                
                # Format top features for display
                top_features = [f"{name} ({importance:.3f})" for name, importance in valid_top_features]
            
            # Build result
            result = {
                'latitude': lat,
                'longitude': lon,
                'probability': probability,
                'classification': classification,
                'image_date': actual_img_date
            }
            
            # Add historical change info if available
            if 'NDVI_trend' in historical_features:
                result['historical_change'] = historical_features['NDVI_trend']
            
            # Add top features info if available
            if top_features:
                result['top_features'] = ', '.join(top_features)
            
            return result
        
        except ee.EEException as ee_err:
            ee_err_str = str(ee_err).lower()
            # Implement backoff for common errors
            if any(term in ee_err_str for term in ['quota', 'rate limit', 'user memory limit', 'computation timed out', 'backend error', 'too many requests']):
                wait_time = min(64, 2**(np.random.randint(1, 7)))  # Random backoff 2-64s
                time.sleep(wait_time)
            return None
        except requests.exceptions.RequestException as req_err:
            time.sleep(1)  # Shorter wait after download errors
            return None
        except Exception as e:
            tqdm.write(f"  Unexpected error analyzing point ({lat:.4f}, {lon:.4f}): {type(e).__name__} - {e}")
            return None
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass  # Ignore errors removing temp file
    
    def create_probability_map(self, shapefile_path, distance_km=1.0, use_historical=True, years_back=5, output_file=None):
        """
        Generate the comprehensive probability map.
        
        Args:
            shapefile_path (str): Path to the shapefile
            distance_km (float): Sampling distance in km
            use_historical (bool): Whether to use historical features
            years_back (int): Number of years to look back for historical data
            output_file (str, optional): Path to save the map
            
        Returns:
            pd.DataFrame: Results DataFrame or None if error
        """
        print("\nStarting Sand Mining Probability Mapping...")
        
        # 1. Generate dense points along the river
        coords, river_gdf = load_shapefile_and_get_points(shapefile_path, distance_km=distance_km)
        
        # Check if generate_river_points returned valid data
        if river_gdf is None or not coords:
            print("Error: Failed to generate river points or load river geometry for mapping.")
            clean_temp_dir()
            return None
        
        # 2. Determine the overall latest imagery date for the area
        try:
            bounds = river_gdf.total_bounds  # [minx, miny, maxx, maxy]
            # Check if bounds are valid
            if not (np.all(np.isfinite(bounds)) and bounds[0] < bounds[2] and bounds[1] < bounds[3]):
                print(f"Error: Invalid bounds calculated from shapefile: {bounds}. Cannot determine AOI for imagery search.")
                clean_temp_dir()
                return None
            aoi_region = ee.Geometry.Rectangle(list(bounds))
        except Exception as bounds_err:
            print(f"Error creating AOI for imagery search from shapefile bounds: {bounds_err}")
            clean_temp_dir()
            return None
        
        latest_date_str, source_info = self.get_latest_imagery_date(aoi_region)
        if "Error" in source_info:
            print(f"Warning: Could not reliably determine latest imagery date ({source_info}). Using current date: {latest_date_str}")
        else:
            print(f"Using imagery baseline date: {latest_date_str} (Source: {source_info})")
        
        # 3. Analyze each point
        results = []
        print(f"\nAnalyzing {len(coords)} points along the river...")
        # Use tqdm for progress bar
        with tqdm(total=len(coords), desc="Mapping Points", unit="point", smoothing=0.1) as pbar:
            for lat, lon in coords:
                analysis_result = self.analyze_point(
                    lat, lon, latest_date_str, 
                    use_historical=use_historical,
                    years_back=years_back
                )
                
                if analysis_result is not None:
                    results.append(analysis_result)
                
                # Add a small sleep to avoid hitting EE limits too hard
                time.sleep(0.05)  # 50ms delay
                pbar.update(1)  # Update progress bar
        
        # 4. Process results and save
        if not results:
            print("\nError: No valid results generated from point analysis.")
            clean_temp_dir()
            return None
        
        results_df = pd.DataFrame(results)
        
        # Use the baseline date for the main filename, handle potential errors in date string
        try:
            map_filename_date = datetime.strptime(latest_date_str, '%Y-%m-%d').strftime('%Y%m%d')
        except ValueError:
            map_filename_date = datetime.now().strftime('%Y%m%d')  # Fallback
        
        csv_path = os.path.join(self.output_folder, f'sand_mining_probabilities_{map_filename_date}.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved probability results to: {csv_path}")
        
        # 5. Create interactive map
        if output_file is None:
            # Create default filename using shapefile name and date
            shapefile_name = os.path.splitext(os.path.basename(shapefile_path))[0]
            output_file = os.path.join(self.output_folder, f'sand_mining_map_{shapefile_name}_{map_filename_date}.html')
        
        # Find the actual date range of imagery used
        try:
            min_img_date = results_df['image_date'].min()
            max_img_date = results_df['image_date'].max()
            date_range_str = f"{min_img_date} to {max_img_date}" if min_img_date != max_img_date else min_img_date
            print(f"Actual imagery dates used in map range from: {date_range_str}")
        except Exception as date_err:
            print(f"Warning: Could not determine date range from results: {date_err}")
            date_range_str = "Unknown"
        
        # Generate interactive map
        generate_interactive_map(results_df, river_gdf, output_file, date_range_str)
        
        # Generate summary report
        create_summary_plot(results_df, shapefile_path, date_range_str)
        
        # Clean up temp folder
        clean_temp_dir()
        
        return results_df

def run_mapping_workflow(
    shapefile_path, 
    distance_km=0.2, 
    use_historical=True, 
    years_back=5, 
    interval_months=6,
    output_file=None
):
    """
    Run the complete mapping workflow.
    
    Args:
        shapefile_path (str): Path to river shapefile
        distance_km (float): Distance between points in km
        use_historical (bool): Whether to use historical data
        years_back (int): Number of years to look back for historical data
        interval_months (int): Interval between historical images in months
        output_file (str): Path to save output map
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Initialize Earth Engine
    if not ee_utils.initialize_ee():
        print("❌ Error: Earth Engine could not be initialized. Exiting.")
        return False
    
    # Load model
    model, scaler, _ = load_model_and_metadata()
    if model is None:
        print("❌ Error: Failed to load model. Cannot proceed with mapping.")
        return False
    
    # Create mapper instance
    mapper = SandMiningProbabilityMapper(model, scaler)
    
    # Generate points along the river for mapping
    print(f"\nGenerating points along the river with {distance_km} km spacing...")
    map_coords, river_gdf = load_shapefile_and_get_points(
        shapefile_path, distance_km=distance_km
    )
    
    if not map_coords:
        print("❌ Error: Failed to generate mapping points from shapefile.")
        return False
    
    print(f"Generated {len(map_coords)} points for mapping.")
    
    # Update config based on the passed parameters
    config.HISTORICAL_YEARS_BACK = years_back
    config.HISTORICAL_INTERVAL_MONTHS = interval_months
    
    # Generate probability map
    results_df = mapper.create_probability_map(
    shapefile_path, 
    distance_km=distance_km,
    use_historical=use_historical,
    years_back=years_back,
    output_file=output_file
)   
    
    if results_df is None or results_df.empty:
        print("❌ Error: Failed to generate probability map.")
        return False
    
    print("\n✅ Mapping workflow completed successfully!")
    return True