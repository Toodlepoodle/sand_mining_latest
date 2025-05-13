#!/usr/bin/env python3
"""
Main execution script for the Sand Mining Detection Tool.
"""

import ee
import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path to import local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from src import config
from src import ee_utils
from src.utils import ensure_directories, clean_temp_dir
from src.gui import start_labeling_gui
from src.model import run_training_workflow, load_model_and_metadata
from src.mapper import run_mapping_workflow

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Sand Mining Detection and Mapping Tool',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--mode', type=str, required=True, choices=['train', 'map', 'both', 'label'],
        help="Operating mode:\n"
             "  train - Download images, label them, and train a model.\n"
             "  map   - Create a probability map using an existing model.\n"
             "  both  - Run training first, then create a map.\n"
             "  label - Only run the image labeling interface."
    )
    parser.add_argument(
        '--shapefile', type=str, required=False,
        help='Path to the river shapefile (.shp).'
    )
    parser.add_argument(
        '--distance', type=float, default=0.2,
        help='Sampling distance in km for mapping points (default: 0.2km).\n'
             'For training, this sets minimum separation between random points.'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output filename for the interactive map (e.g., my_river_map.html).\n'
             'If not specified, a default name with date is used.'
    )
    parser.add_argument(
        '--model', type=str, default=config.DEFAULT_MODEL_FILE,
        help=f'Path to the trained model file (.joblib) for mode=map.\n'
             f'(Default: {config.DEFAULT_MODEL_FILE})'
    )
    parser.add_argument(
        '--scaler', type=str, default=config.DEFAULT_SCALER_FILE,
        help=f'Path to the feature scaler file (.joblib) for mode=map.\n'
              f'(Default: {config.DEFAULT_SCALER_FILE})'
    )
    parser.add_argument(
        '--sample-size', type=int, default=30,
        help='Number of random sample points to generate for training (default: 30).'
    )
    parser.add_argument(
        '--use-grid-search', action='store_true',
        help='Use grid search for hyperparameter tuning during model training.'
    )
    parser.add_argument(
        '--model-type', type=str, choices=['random_forest', 'gradient_boosting'], 
        default='random_forest',
        help='Type of model to use for training (default: random_forest).'
    )
    parser.add_argument(
        '--multiple-models', action='store_true',
        help='Train and evaluate multiple model types (Random Forest, XGBoost, etc.).'
    )
    parser.add_argument(
        '--historical', action='store_true', default=True,
        help='Use historical time-series data for feature extraction (default: True).'
    )
    parser.add_argument(
        '--years-back', type=int, default=5,
        help='Number of years to look back for historical data (default: 5).'
    )
    parser.add_argument(
        '--interval-months', type=int, default=6,
        help='Interval between historical images in months (default: 6). \n'
             'Smaller values provide more temporal data points.'
    )
    parser.add_argument(
        '-y', '--yes', action='store_true', 
        help='Automatically answer yes to confirmations.'
    )
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Validate required arguments based on mode
    if args.mode in ['train', 'map', 'both'] and not args.shapefile:
        parser.error(f"--shapefile is required for mode '{args.mode}'")
    
    return args

def download_training_images(args):
    """
    Download training images from a shapefile.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*80)
    print(" DOWNLOADING TRAINING IMAGES")
    print("="*80 + "\n")
    
    # Initialize Earth Engine
    if not ee_utils.initialize_ee():
        print("❌ Error: Earth Engine could not be initialized. Exiting.")
        return False
    
    # Load shapefile and get sampling points for training
    from src.utils import load_shapefile_and_get_points
    import random
    
    # Get coordinates from the shapefile
    coordinates, river_gdf = load_shapefile_and_get_points(
        args.shapefile,
        distance_km=args.distance
    )
    
    # Sample the coordinates if needed
    if coordinates and len(coordinates) > args.sample_size:
        print(f"Sampling {args.sample_size} points from {len(coordinates)} available points")
        training_coordinates = random.sample(coordinates, args.sample_size)
    else:
        training_coordinates = coordinates
    
    if not training_coordinates:
        print("❌ Error: Failed to get coordinates from shapefile. Cannot proceed.")
        return False
    
    # Create download function using the Sand Mining Detection code
    from PIL import Image
    import io
    import time
    import requests
    
    def download_training_image(lat, lon, img_idx):
        """Download a single training image."""
        try:
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(config.DEFAULT_BUFFER_METERS)
            
            # Get current date and 12 months prior
            current_ee_date = ee.Date(datetime.now())
            start_ee_date = current_ee_date.advance(-12, 'month')
            
            # Find the least cloudy Sentinel-2 image
            s2_collection = ee.ImageCollection(config.S2_COLLECTION) \
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
                s2_collection_wider = ee.ImageCollection(config.S2_COLLECTION) \
                    .filterBounds(region) \
                    .filterDate(start_ee_date, current_ee_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)) \
                    .sort('CLOUDY_PIXEL_PERCENTAGE')
                
                image_count_wider = s2_collection_wider.size().getInfo()
                if image_count_wider > 0:
                    best_image = ee.Image(s2_collection_wider.first())
                else:
                    print(f"  Skipping point {img_idx+1} ({lat:.4f}, {lon:.4f}): No suitable S2 image found.")
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
                'dimensions': config.DEFAULT_IMAGE_DIM,
                'format': 'png'
            })
            
            # Download the image
            response = requests.get(download_url, timeout=90)
            response.raise_for_status()
            
            # Open image with PIL
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            # Enhance image quality
            img_enhanced = ee_utils.enhance_image(img)
            
            # Save the enhanced image
            filename = f'train_image_{img_idx+1}_{lat:.6f}_{lon:.6f}.png'
            filepath = os.path.join(config.TRAINING_IMAGES_DIR, filename)
            img_enhanced.save(filepath, format='PNG', optimize=True, quality=95)
            
            return True
        
        except Exception as e:
            print(f"  Error downloading image for point {img_idx+1} ({lat:.4f}, {lon:.4f}): {e}")
            return False
    
    # Download images
    print(f"\nDownloading {len(training_coordinates)} images for training/labeling...")
    
    # Create training images directory if it doesn't exist
    os.makedirs(config.TRAINING_IMAGES_DIR, exist_ok=True)
    
    success_count = 0
    
    from tqdm import tqdm
    for i, (lat, lon) in enumerate(tqdm(training_coordinates, desc="Downloading Images", unit="image")):
        success = download_training_image(lat, lon, i)
        if success:
            success_count += 1
        
        # Add a small delay to avoid hitting EE rate limits
        time.sleep(1.2)
    
    print(f"\nSuccessfully downloaded {success_count} out of {len(training_coordinates)} images.")
    
    if success_count == 0:
        print("❌ Error: Failed to download any training images.")
        return False
    
    return True

def main():
    """Main execution function."""
    print(
        f"""
==========================================================
     SAND MINING DETECTION AND MAPPING TOOL v2.1
==========================================================
 Using Google Earth Engine and Machine Learning to identify
 potential sand mining activities along rivers.
----------------------------------------------------------
 Current directory: {os.getcwd()}
 Output directory: {config.OUTPUT_DIR}
 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
----------------------------------------------------------"""
    )
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set configuration based on arguments
    config.HISTORICAL_YEARS_BACK = args.years_back
    config.HISTORICAL_INTERVAL_MONTHS = args.interval_months
    
    # Ensure all directories exist
    ensure_directories()
    
    # Clean temporary directory
    clean_temp_dir()
    
    # Track success status for each step
    download_success = False
    training_success = False
    mapping_success = False
    
    # Execute based on mode
    if args.mode == 'label':
        # Just run the labeling GUI
        start_labeling_gui()
        return
    
    if args.mode in ['train', 'both']:
        # Download training images
        download_success = download_training_images(args)
        
        if not download_success:
            print("❌ Training image download failed. Cannot proceed with training.")
            sys.exit(1)
        
        # Launch labeling GUI
        print("\nLaunching image labeling GUI. Please label the images, then training will continue.")
        start_labeling_gui()
        
        # Run model training with enhanced options
        training_success = run_training_workflow(
            use_grid_search=args.use_grid_search,
            add_historical_features=args.historical,
            model_type=args.model_type,
            use_multiple_models=args.multiple_models
        )
        
        if not training_success and args.mode == 'both':
            print("\n❌ Training failed. Skipping mapping step.")
            sys.exit(1)
    
    if args.mode in ['map', 'both']:
        # In 'both' mode, only proceed if training was successful
        if args.mode == 'both' and not training_success:
            print("\n❌ Training failed. Skipping mapping step.")
            sys.exit(1)
        
        # Run mapping with improved point density
        mapping_success = run_mapping_workflow(
            shapefile_path=args.shapefile,
            distance_km=args.distance,  # Use smaller distance for better coverage
            use_historical=args.historical,
            years_back=args.years_back,
            interval_months=args.interval_months,
            output_file=args.output
        )
    
    # Final Summary
    print("\n" + "="*80)
    print(" TOOL EXECUTION SUMMARY")
    print("="*80)
    final_status = 0  # 0 for success, 1 for failure
    
    if args.mode in ['train', 'both']:
        status_msg = '✅ SUCCESS' if training_success else '❌ FAILED'
        print(f"Training attempt: {status_msg}")
        if not training_success:
            final_status = 1
    
    if args.mode in ['map', 'both']:
        # Check if mapping was skipped due to training failure
        if args.mode == 'both' and not training_success:
            print("Mapping attempt:  SKIPPED due to training failure")
        else:
            status_msg = '✅ SUCCESS' if mapping_success else '❌ FAILED'
            print(f"Mapping attempt:  {status_msg}")
            if not mapping_success:
                final_status = 1
    
    print("="*80)
    
    if final_status != 0:
        print("\nOne or more critical steps failed. Please review the logs above for errors.")
    else:
        print("\nTool finished successfully.")
    
    sys.exit(final_status)

if __name__ == "__main__":
    main()