# Sand Mining Detection and Mapping Tool v2.0

A powerful tool for detecting and mapping potential sand mining activities along river systems using satellite imagery and machine learning. This enhanced version incorporates historical time-series analysis, multi-sensor integration, and feature importance visualization.

## Features

- **Historical Time-Series Analysis**: Analyzes changes in spectral indices over time to detect patterns of sand mining activity
- **Multi-Sensor Integration**: Combines data from Sentinel-2, Landsat series, and VIIRS to maximize coverage and historical depth
- **Advanced Feature Extraction**: Extracts texture, color, and spectral index features for improved detection accuracy
- **Interactive Visualization**: Creates heatmaps, cluster maps, and interactive visualizations of probability and risk
- **Feature Importance Analysis**: Explains model predictions through feature importance visualization
- **Modular Architecture**: Well-organized, maintainable code structure for easy extension

## Requirements

- Python 3.7+ 
- Google Earth Engine account
- Google Cloud SDK installed and configured
- Required libraries (see `requirements.txt`)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/sand-mining-detection.git
cd sand-mining-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Authenticate with Google Earth Engine:
```bash
earthengine authenticate
```

4. Ensure your shapefiles are placed in the `data/raw` directory.

## Folder Structure

```
sand-mining-detection/
├── data/
│   ├── raw/              # Store input shapefiles here
│   └── processed/        # Processed geographical data
├── outputs/
│   ├── training_images/  # Downloaded training images
│   ├── models/           # Saved models and scalers
│   ├── probability_maps/ # Generated heatmaps and reports
│   └── temp/             # Temporary processing files
├── src/
│   ├── config.py         # Configuration constants
│   ├── ee_utils.py       # Earth Engine helpers
│   ├── features.py       # Feature extraction
│   ├── gui.py            # Labeling interface
│   ├── mapper.py         # Probability mapping
│   ├── model.py          # ML model handling
│   └── utils.py          # Utility functions
└── scripts/
    └── run_pipeline.py   # Main execution script
```

## Usage

The tool has four main modes:

### 1. Training Mode

Downloads sample images along the river, allows for labeling, and trains a machine learning model:

```bash
python scripts/run_pipeline.py --mode train --shapefile data/raw/your_river.shp --sample-size 30
```

### 2. Mapping Mode

Creates a probability map using an existing trained model:

```bash
python scripts/run_pipeline.py --mode map --shapefile data/raw/your_river.shp --distance 0.5
```

### 3. Combined Mode (Train + Map)

Train a model and then immediately create a map:

```bash
python scripts/run_pipeline.py --mode both --shapefile data/raw/your_river.shp --sample-size 25 --distance 1.0
```

### 4. Labeling Mode

Just run the labeling interface without downloading new images or training:

```bash
python scripts/run_pipeline.py --mode label
```

## Advanced Options

- `--use-grid-search`: Enable hyperparameter tuning (slower but potentially more accurate)
- `--model-type`: Choose between 'random_forest' (default) or 'gradient_boosting' algorithms
- `--historical`: Use historical time-series data (default: True)
- `--years-back`: Number of years to look back for historical data (default: 5)
- `--output`: Specify custom output file name for the interactive map

## Workflow Example

Full workflow to detect sand mining along a river:

1. Place your river shapefile in `data/raw/`
2. Run the combined mode:
```bash
python scripts/run_pipeline.py --mode both --shapefile data/raw/ganges_river.shp --sample-size 40 --distance 0.8 --historical --years-back 10
```
3. Use the labeling GUI to classify training images
4. Once training is complete, a probability map will be automatically generated
5. View the interactive map in `outputs/probability_maps/`

## Output Files

The tool generates several output files:

- **Interactive Map**: HTML file with markers, heatmap, and classification clusters
- **Probability CSV**: Raw data with coordinates, probabilities, and classifications
- **Summary Plot**: PNG image with probability distribution and classifications
- **Feature Importance**: Visualization of the most important features for detection
- **Model Files**: Saved model and scaler for later use or analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This tool uses Google Earth Engine and various open-source libraries including scikit-learn, folium, and geopandas.