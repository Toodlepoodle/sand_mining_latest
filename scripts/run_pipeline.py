#!/usr/bin/env python3
"""
Main execution script for the Enhanced Sand Mining Detection Tool.
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
from src.mapper import run_mapping_workflow  # FIXED: Changed from probability_mapping to mapper

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced Sand Mining Detection and Mapping Tool with Area Highlighting',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--mode', type=str, required=True, choices=['train', 'map', 'both', 'label'],
        help="Operating mode:\n"
             "  train - Download images, label them with area highlighting, and train model.\n"
             "  map   - Create a probability map using an existing model.\n"
             "  both  - Run training first, then create a map.\n"
             "  label - Only run the image labeling and area annotation interface."
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
        '--model-type', type=str, choices=['random_forest', 'gradient_boosting', 'xgboost'], 
        default='random_forest',
        help='Type of model to use for training (default: random_forest).'
    )
    parser.add_argument(
        '--multiple-models', action='store_true',
        help='Train and evaluate multiple model types (Random Forest, XGBoost, LightGBM, etc.).'
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
    
    # Download images using the enhanced ee_utils
    success = ee_utils.download_training_images(
        training_coordinates,
        config.TRAINING_IMAGES_DIR,
        img_dim=config.DEFAULT_IMAGE_DIM,
        buffer_m=config.DEFAULT_BUFFER_METERS
    )
    
    if success:
        print("✅ Training images downloaded successfully!")
        return True
    else:
        print("❌ Error: Failed to download training images.")
        return False

def run_enhanced_labeling():
    """
    Run the enhanced labeling GUI with area highlighting.
    """
    print("\n" + "="*80)
    print(" ENHANCED IMAGE LABELING & AREA HIGHLIGHTING")
    print("="*80 + "\n")
    
    print("Starting enhanced labeling GUI with area highlighting...")
    print("\n📋 Instructions:")
    print("1. LABELING MODE:")
    print("   - Use buttons or keys (0=No Mining, 1=Mining, ?=Skip) to label entire images")
    print("   - This provides overall image classification for training")
    print("\n2. AREA ANNOTATION MODE (Press 'a' to toggle):")
    print("   - Click and drag to highlight specific sand mining areas")
    print("   - Different highlight types:")
    print("     * sand_mining: Active sand mining areas")
    print("     * equipment: Heavy machinery/equipment")
    print("     * water_disturbance: Disturbed water patterns")
    print("     * no_mining: Clearly undisturbed areas")
    print("\n3. ENHANCED FEATURES:")
    print("   - Model learns from both global image features AND specific highlighted areas")
    print("   - Better precision by focusing on actual sand mining locations")
    print("   - Press 'h' for detailed help")
    print("\n⚠️  Important: Label at least 10+ images and highlight key areas for best results")
    print("="*80)
    
    # Launch the enhanced GUI
    start_labeling_gui()
    
    # Check if we have sufficient labels and annotations
    from src.utils import load_labels
    labels = load_labels()
    
    if not labels:
        print("\n❌ No labels found. Please label some images before training.")
        return False
    
    labeled_count = sum(1 for label in labels.values() if label != -1)
    
    if labeled_count < 5:
        print(f"\n⚠️  Warning: Only {labeled_count} images labeled. Recommend at least 10 for good results.")
        if not input("Continue anyway? (y/N): ").lower().startswith('y'):
            return False
    
    print(f"\n✅ Labeling completed! Found {labeled_count} labeled images.")
    
    # Check for area annotations
    if os.path.exists(config.ANNOTATIONS_FILE):
        try:
            import json
            with open(config.ANNOTATIONS_FILE, 'r') as f:
                annotations = json.load(f)
            
            total_annotations = sum(len(img_annotations) for img_annotations in annotations.values())
            print(f"✅ Found {total_annotations} area annotations across {len(annotations)} images.")
            
            if total_annotations == 0:
                print("⚠️  No area highlights found. Model will use only global features.")
            else:
                print("🎯 Enhanced training will use both global AND area-specific features!")
                
        except Exception as e:
            print(f"⚠️  Could not read annotations: {e}")
    
    return True

def main():
    """Main execution function."""
    print(
        f"""
==========================================================
   ENHANCED SAND MINING DETECTION TOOL v3.0
   With Area Highlighting & Spatial Feature Extraction
==========================================================
 Using Google Earth Engine, Machine Learning, and Enhanced
 Area-Specific Feature Extraction to precisely identify
 sand mining activities along rivers.
----------------------------------------------------------
 Current directory: {os.getcwd()}
 Output directory: {config.OUTPUT_DIR}
 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
----------------------------------------------------------"""
    )
    
    # Parse command line arguments
    args = parse_arguments()
    
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
        # Just run the enhanced labeling GUI
        run_enhanced_labeling()
        return
    
    if args.mode in ['train', 'both']:
        # Download training images
        download_success = download_training_images(args)
        
        if not download_success:
            print("❌ Training image download failed. Cannot proceed with training.")
            sys.exit(1)
        
        # Launch enhanced labeling GUI
        labeling_success = run_enhanced_labeling()
        
        if not labeling_success:
            print("❌ Labeling step failed or insufficient labels. Cannot proceed with training.")
            sys.exit(1)
        
        # Run enhanced model training
        print("\n" + "="*80)
        print(" ENHANCED MODEL TRAINING")
        print("="*80 + "\n")
        
        training_success = run_training_workflow(
            use_grid_search=args.use_grid_search,
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
        
        # Check if model exists for mapping mode
        if args.mode == 'map':
            model, scaler, feature_names = load_model_and_metadata()
            if model is None:
                print("❌ No trained model found. Please run training first or specify --model path.")
                sys.exit(1)
        
        # Run enhanced mapping
        print("\n" + "="*80)
        print(" ENHANCED PROBABILITY MAPPING")
        print("="*80 + "\n")
        
        mapping_success = run_mapping_workflow(
            shapefile_path=args.shapefile,
            distance_km=args.distance,
            use_historical=False,  # No historical features in enhanced version
            years_back=0,
            output_file=args.output
        )
    
    # Final Summary
    print("\n" + "="*80)
    print(" ENHANCED TOOL EXECUTION SUMMARY")
    print("="*80)
    final_status = 0  # 0 for success, 1 for failure
    
    if args.mode in ['train', 'both']:
        status_msg = '✅ SUCCESS' if training_success else '❌ FAILED'
        print(f"Enhanced Training: {status_msg}")
        if training_success:
            print("  ✓ Global image features extracted")
            print("  ✓ Area-specific features from highlights")
            print("  ✓ Enhanced model trained on spatial data")
        if not training_success:
            final_status = 1
    
    if args.mode in ['map', 'both']:
        # Check if mapping was skipped due to training failure
        if args.mode == 'both' and not training_success:
            print("Enhanced Mapping:  SKIPPED due to training failure")
        else:
            status_msg = '✅ SUCCESS' if mapping_success else '❌ FAILED'
            print(f"Enhanced Mapping:  {status_msg}")
            if mapping_success:
                print("  ✓ High-resolution point analysis")
                print("  ✓ Area-aware feature extraction")
                print("  ✓ Interactive probability map generated")
            if not mapping_success:
                final_status = 1
    
    print("="*80)
    
    if final_status != 0:
        print("\nOne or more critical steps failed. Please review the logs above for errors.")
    else:
        print("\n🎉 Enhanced Sand Mining Detection Tool finished successfully!")
        print("\nKey improvements in this version:")
        print("  • Area highlighting for precise training data")
        print("  • Enhanced feature extraction from highlighted regions")
        print("  • Better model accuracy through spatial awareness")
        print("  • Simplified workflow without complex historical analysis")
    
    sys.exit(final_status)

if __name__ == "__main__":
    main()