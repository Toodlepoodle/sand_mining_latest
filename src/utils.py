#!/usr/bin/env python3
"""
Utility functions for the Sand Mining Detection Tool.
"""

import os
import shutil
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import transform, unary_union
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime
import folium
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm

from src import config

def ensure_directories():
    """
    Ensure all required directories exist.
    """
    directories = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.OUTPUT_DIR,
        config.TRAINING_IMAGES_DIR,
        config.MODELS_DIR,
        config.PROBABILITY_MAPS_DIR,
        config.TEMP_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def clean_temp_dir():
    """
    Clean the temporary directory.
    """
    try:
        if os.path.exists(config.TEMP_DIR):
            shutil.rmtree(config.TEMP_DIR)
            os.makedirs(config.TEMP_DIR, exist_ok=True)
            print(f"Cleaned temporary directory: {config.TEMP_DIR}")
    except Exception as e:
        print(f"Warning: Could not clean temporary directory: {e}")

def load_labels():
    """
    Load the training labels file.
    
    Returns:
        dict: Dictionary mapping filenames to labels
    """
    labels = {}
    
    if os.path.exists(config.LABELS_FILE):
        try:
            df = pd.read_csv(config.LABELS_FILE)
            
            if 'filename' in df.columns and 'label' in df.columns:
                # Ensure filename is string, handle potential float conversion
                df['filename'] = df['filename'].astype(str)
                labels = pd.Series(df.label.values, index=df.filename).to_dict()
                
                # Convert labels explicitly to int, handling potential NaNs
                for img_file, label in labels.items():
                    try:
                        # Use pd.isna for robust NaN checking
                        labels[img_file] = int(label) if pd.notna(label) else -1
                    except (ValueError, TypeError):
                        labels[img_file] = -1  # Default to unlabeled if conversion fails
            else:
                print(f"Warning: Labels file {config.LABELS_FILE} missing 'filename' or 'label' column.")
        except pd.errors.EmptyDataError:
            print(f"Warning: Labels file {config.LABELS_FILE} is empty.")
        except Exception as e:
            print(f"Warning: Could not load labels from {config.LABELS_FILE}: {e}")
    
    return labels

def save_labels(labels):
    """
    Save labels to the labels file.
    
    Args:
        labels (dict): Dictionary mapping filenames to labels
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        df = pd.DataFrame(labels.items(), columns=['filename', 'label'])
        
        # Filter out rows where filename might be NaN or empty
        df = df.dropna(subset=['filename'])
        df = df[df['filename'] != '']
        
        # Convert label column to integer type before saving
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
        
        df.to_csv(config.LABELS_FILE, index=False)
        print(f"Labels saved to {config.LABELS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving labels: {e}")
        return False

def load_shapefile_and_get_points(shapefile_path, distance_km=0.2):
    """
    Load river shapefile and generate points along it.
    
    Args:
        shapefile_path (str): Path to the shapefile
        distance_km (float): Distance between points in km
        
    Returns:
        tuple: (coordinates, GeoDataFrame) where coordinates is a list of (lat, lon) points
    """
    print(f"Loading river shapefile from: {shapefile_path}")
    try:
        gdf = gpd.read_file(shapefile_path)
        
        # Ensure CRS is geographic (WGS84)
        if gdf.crs is None:
            print("Warning: Shapefile has no CRS defined. Assuming WGS84 (EPSG:4326).")
            gdf.crs = "EPSG:4326"
        elif gdf.crs.to_epsg() != 4326:
            print(f"Converting CRS from {gdf.crs.to_string()} to WGS84 (EPSG:4326)")
            gdf = gdf.to_crs("EPSG:4326")
        
        # Combine all geometries into a single LineString or MultiLineString
        geoms = []
        invalid_geom_count = 0
        
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
                
            if not geom.is_valid:
                invalid_geom_count += 1
                try:
                    geom = geom.buffer(0)  # Try to fix invalid geometry
                    if not geom.is_valid or geom.is_empty:
                        continue
                except Exception:
                    continue
            
            if isinstance(geom, (Polygon, MultiPolygon)):
                geoms.append(geom.boundary)
            elif isinstance(geom, (LineString, MultiLineString)):
                geoms.append(geom)
        
        if invalid_geom_count > 0:
            print(f"Warning: Found and attempted to fix {invalid_geom_count} invalid geometries.")
        
        if not geoms:
            print("Error: No valid LineString or Polygon geometries found in shapefile after cleaning.")
            return [], gdf
        
        river_geom = unary_union(geoms)
        
        if river_geom.is_empty:
            print("Error: Combined river geometry is empty.")
            return [], gdf
        
        print(f"Total river geometry type: {river_geom.geom_type}")
        
        # Convert distance in km to degrees (rough approximation)
        distance_degrees = distance_km / 111.0  # 1 degree ≈ 111 km
        
        # Generate points along the river at regular intervals
        points = []
        total_length = river_geom.length
        approx_length_km = total_length * 111  # Rough estimate
        
        print(f"Approximate river length: {approx_length_km:.2f} km")
        print(f"Generating points at {distance_km} km intervals...")
        
        # Calculate number of points based on river length and desired interval
        num_points = max(2, int(total_length / distance_degrees) + 1)
        
        if isinstance(river_geom, MultiLineString):
            # Handle each linestring in the multilinestring
            for line in river_geom.geoms:
                line_length = line.length
                # Calculate number of points for this line segment
                line_num_points = max(2, int(line_length / distance_degrees) + 1)
                for i in range(line_num_points):
                    # Interpolate point at regular interval
                    fraction = i / float(line_num_points - 1)
                    point = line.interpolate(fraction, normalized=True)
                    points.append((point.y, point.x))  # lat, lon
        else:
            # Handle single linestring
            for i in range(num_points):
                # Interpolate point at regular interval
                fraction = i / float(num_points - 1)
                point = river_geom.interpolate(fraction, normalized=True)
                points.append((point.y, point.x))  # lat, lon
        
        # Remove any potential duplicates
        unique_points = list(set(points))
        
        print(f"Generated {len(unique_points)} points along the river.")
        return unique_points, gdf
    
    except Exception as e:
        print(f"Error loading shapefile or generating points: {e}")
        import traceback
        traceback.print_exc()
        return [], None

def extract_coordinates_from_filename(filename):
    """
    Extract latitude and longitude from a filename pattern.
    
    Args:
        filename (str): Filename with pattern containing coordinates
        
    Returns:
        tuple: (lat, lon) or (None, None) if not found
    """
    try:
        # Expect pattern like 'train_image_X_LATITUDE_LONGITUDE.png'
        parts = filename.split('_')
        if len(parts) >= 5:
            lat_str = parts[-2]
            lon_str = parts[-1].split('.')[0]  # Remove file extension
            
            lat = float(lat_str)
            lon = float(lon_str)
            
            # Basic validation
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    except Exception:
        pass
    
    return None, None

def create_lat_lon_mapping(image_folder):
    """
    Create a mapping from filenames to (lat, lon) coordinates.
    
    Args:
        image_folder (str): Folder containing images
        
    Returns:
        dict: Dictionary mapping filenames to (lat, lon) coordinates
    """
    mapping = {}
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith('.png'):
            lat, lon = extract_coordinates_from_filename(filename)
            if lat is not None and lon is not None:
                mapping[filename] = (lat, lon)
    
    print(f"Created mapping for {len(mapping)} images with embedded coordinates.")
    return mapping

def generate_interactive_map(results_df, river_gdf, output_file, imagery_date_range):
    """
    Create an interactive Folium map with the results.
    
    Args:
        results_df (pd.DataFrame): DataFrame with results
        river_gdf (gpd.GeoDataFrame): GeoDataFrame with river geometry
        output_file (str): Path to save the map
        imagery_date_range (str): Date range of imagery used
        
    Returns:
        None
    """
    print("Generating interactive map...")
    
    if results_df.empty:
        print("Warning: No results to map.")
        return
    
    # Calculate map center and bounds
    try:
        # Drop rows with NaN lat/lon before calculating bounds/center
        results_df_valid = results_df.dropna(subset=['latitude', 'longitude'])
        
        if results_df_valid.empty:
            print("Warning: No valid coordinates in results to determine map center/bounds.")
            center_lat, center_lon = 0, 0  # Default fallback
            map_bounds = None
            zoom_start = 2
        else:
            center_lat = results_df_valid['latitude'].mean()
            center_lon = results_df_valid['longitude'].mean()
            min_lat = results_df_valid['latitude'].min()
            max_lat = results_df_valid['latitude'].max()
            min_lon = results_df_valid['longitude'].min()
            max_lon = results_df_valid['longitude'].max()
            map_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
            zoom_start = 11  # Default zoom if bounds exist
    except Exception as center_err:
        print(f"Warning: Error calculating map center/bounds: {center_err}. Using default view.")
        center_lat, center_lon = 0, 0
        map_bounds = None
        zoom_start = 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=None)
    
    # Add base layers
    folium.TileLayer('CartoDB positron', name='CartoDB Positron (Light)', show=True).add_to(m)
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap', show=False).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri World Imagery',
        overlay=False,
        control=True,
        show=False  # Start with light map active
    ).add_to(m)
    
    # River Geometry Layer
    if river_gdf is not None and not river_gdf.empty:
        try:
            # Attempt to get column names, handle if it's empty or invalid GDF
            tooltip_cols = list(river_gdf.columns) if not river_gdf.empty else None
            # Exclude geometry column itself from tooltip
            if tooltip_cols and 'geometry' in tooltip_cols:
                tooltip_cols.remove('geometry')
            # Only include a few relevant columns if too many exist
            max_tooltip_cols = 5
            if tooltip_cols and len(tooltip_cols) > max_tooltip_cols:
                tooltip_cols = tooltip_cols[:max_tooltip_cols]  # Take first few
            if not tooltip_cols:
                tooltip_cols = None  # Handle case where only geometry exists
            
            folium.GeoJson(
                river_gdf,
                name='River Outline',
                style_function=lambda x: {'color': 'blue', 'weight': 2, 'opacity': 0.6},
                tooltip=folium.GeoJsonTooltip(fields=tooltip_cols, aliases=tooltip_cols) if tooltip_cols else None,
                show=True  # Show river by default
            ).add_to(m)
        except Exception as geojson_err:
            print(f"Warning: Could not add river GeoJSON layer: {geojson_err}")
    else:
        print("Warning: River GeoDataFrame is None or empty, skipping GeoJSON layer.")
    
    # --- Probability Layers ---
    # Colormap
    color_scale = cm.LinearColormap(
        ['#00FF00', '#FFFF00', '#FFA500', '#FF0000'],  # Green, Yellow, Orange, Red
        vmin=0, vmax=1, caption='Sand Mining Probability'
    )
    m.add_child(color_scale)
    
    # Heatmap Layer
    heatmap_data = results_df[['latitude', 'longitude', 'probability']].dropna().values.tolist()
    if heatmap_data:  # Only add if data exists
        HeatMap(
            heatmap_data,
            name='Probability Heatmap',
            radius=12,  # Adjust radius
            blur=8,    # Adjust blur
            gradient={0.0: '#00FF00', 0.5: '#FFFF00', 0.7: '#FFA500', 1: '#FF0000'},
            show=False  # Initially hidden
        ).add_to(m)
    
    # Marker Clusters for different risk levels
    high_risk_df = results_df[results_df['probability'] >= 0.7].dropna(subset=['latitude', 'longitude'])
    medium_risk_df = results_df[(results_df['probability'] >= 0.5) & (results_df['probability'] < 0.7)].dropna(subset=['latitude', 'longitude'])
    low_risk_df = results_df[results_df['probability'] < 0.5].dropna(subset=['latitude', 'longitude'])
    
    # --- Add Markers ---
    # Create cluster groups even if empty initially, add markers if data exists
    mc_high = MarkerCluster(name=f"High Risk Points (p >= 0.7) [{len(high_risk_df)}]", show=True).add_to(m)
    mc_medium = MarkerCluster(name=f"Medium Risk Points (0.5 <= p < 0.7) [{len(medium_risk_df)}]", show=True).add_to(m)
    mc_low = MarkerCluster(name=f"Low Risk Points (p < 0.5) [{len(low_risk_df)}]", show=False).add_to(m)  # Initially hidden
    
    if not high_risk_df.empty:
        for idx, row in high_risk_df.iterrows():
            add_marker(mc_high, row, color_scale)
    
    if not medium_risk_df.empty:
        for idx, row in medium_risk_df.iterrows():
            add_marker(mc_medium, row, color_scale)
    
    if not low_risk_df.empty:
        for idx, row in low_risk_df.iterrows():
            add_marker(mc_low, row, color_scale)
    
    # Fit map bounds if they were calculated
    if map_bounds:
        m.fit_bounds(map_bounds, padding=(10, 10))  # Add a little padding
    
    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add Title and Info Box
    num_total = len(results_df)
    num_high = len(high_risk_df)
    num_medium = len(medium_risk_df)
    avg_prob = results_df['probability'].mean() if num_total > 0 else 0
    high_perc = (num_high / num_total * 100) if num_total > 0 else 0
    med_perc = (num_medium / num_total * 100) if num_total > 0 else 0
    
    title_html = f'''
        <div style="position: fixed;
                    top: 10px; left: 50px; width: 300px; height: auto;
                    background-color: rgba(255, 255, 255, 0.85); /* Semi-transparent white */
                    border:2px solid grey; z-index:9999; font-size:14px;
                    padding: 10px; border-radius: 5px; box-shadow: 3px 3px 5px rgba(0,0,0,0.3);">
            <h4 style="margin-top:0; text-align:center; margin-bottom: 8px;">Sand Mining Probability Map</h4>
            <p style="font-size:11px; margin-bottom:5px; border-bottom: 1px solid #ccc; padding-bottom: 4px;">
                <b>Imagery Dates:</b> {imagery_date_range}
            </p>
            <ul style="font-size:11px; list-style-type: none; padding-left: 0; margin-bottom: 5px;">
                <li>Total Points Analyzed: {num_total}</li>
                <li>Avg. Probability: {avg_prob:.3f}</li>
                <li style="color:red;">High Risk (p≥0.7): {num_high} ({high_perc:.1f}%)</li>
                <li style="color:orange;">Medium Risk (0.5≤p<0.7): {num_medium} ({med_perc:.1f}%)</li>
            </ul>
            <p style="font-size:10px; text-align:center; margin-bottom:0; color:#555;"><i>Toggle layers via control (top-right)</i></p>
        </div>
        '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    try:
        m.save(output_file)
        print(f"Successfully generated map: {output_file}")
    except Exception as save_map_err:
        print(f"Error saving map to {output_file}: {save_map_err}")

def add_marker(marker_cluster, data_row, color_scale):
    """
    Helper function to add a CircleMarker to a MarkerCluster.
    
    Args:
        marker_cluster: Folium MarkerCluster to add marker to
        data_row: DataFrame row with point data
        color_scale: Folium ColorMap for probability coloring
        
    Returns:
        None
    """
    prob = data_row['probability']
    color = color_scale(prob)
    
    # Ensure lat/lon are valid numbers before creating marker
    lat = data_row['latitude']
    lon = data_row['longitude']
    
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return
    
    # Create popup content with all available information
    popup_html = f"""
        <b>Lat:</b> {lat:.5f}, <b>Lon:</b> {lon:.5f}<br>
        <b>Probability:</b> {prob:.3f}<br>
        <b>Classification:</b> {data_row['classification']}<br>
    """
    
    # Add image date if available
    if 'image_date' in data_row and not pd.isna(data_row['image_date']):
        popup_html += f"<b>Image Date:</b> {data_row['image_date']}<br>"
    
    # Add historical data info if available
    if 'historical_change' in data_row and not pd.isna(data_row['historical_change']):
        popup_html += f"<b>Historical Change:</b> {data_row['historical_change']:.2f}<br>"
    
    # Add feature importance info if available
    if 'top_features' in data_row and not pd.isna(data_row['top_features']):
        popup_html += f"<b>Key Features:</b> {data_row['top_features']}<br>"
    
    popup = folium.Popup(popup_html, max_width=250)
    
    marker = folium.CircleMarker(
        location=[lat, lon],
        radius=6,  # Smaller radius for individual points
        color='black',  # Outline color
        weight=0.5,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=popup,
        tooltip=f"P: {prob:.2f}"  # Simple tooltip on hover
    )
    marker.add_to(marker_cluster)

def create_summary_plot(results_df, shapefile_path, imagery_date_range, output_file=None):
    """
    Generate a summary plot report.
    
    Args:
        results_df (pd.DataFrame): DataFrame with results
        shapefile_path (str): Path to the original shapefile
        imagery_date_range (str): Date range of imagery used
        output_file (str, optional): Path to save the plot
        
    Returns:
        str: Path to the saved plot
    """
    print("Generating summary report plot...")
    
    if results_df is None or results_df.empty:
        print("Warning: No results to create report from.")
        return None
    
    if output_file is None:
        # Try to extract start date, handle potential errors
        try:
            # Use first part of date range for filename
            map_filename_date = imagery_date_range.split(' ')[0].replace('-', '')
        except:
            map_filename_date = datetime.now().strftime('%Y%m%d')  # Fallback to current date
        
        # Add shapefile name to report filename
        shapefile_name = os.path.splitext(os.path.basename(shapefile_path))[0]
        output_file = os.path.join(config.PROBABILITY_MAPS_DIR, f'sand_mining_summary_{shapefile_name}_{map_filename_date}.png')
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')  # Use a clean style
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # 2x2 grid of plots
        
        # 1. Histogram of Probabilities
        valid_probs = results_df['probability'].dropna()  # Drop NaNs before plotting
        if not valid_probs.empty:
            axs[0, 0].hist(valid_probs, bins=20, color='skyblue', edgecolor='black')
            axs[0, 0].axvline(0.5, color='orange', linestyle='--', linewidth=1.5, label='Threshold 0.5')
            axs[0, 0].axvline(0.7, color='red', linestyle='--', linewidth=1.5, label='Threshold 0.7')
            axs[0, 0].legend()
        else:
            axs[0, 0].text(0.5, 0.5, "No valid probability data", ha='center', va='center')
        
        axs[0, 0].set_title('Distribution of Sand Mining Probabilities')
        axs[0, 0].set_xlabel('Probability')
        axs[0, 0].set_ylabel('Number of Points')
        axs[0, 0].grid(True, linestyle='--', alpha=0.6)
        
        # 2. Bar Chart of Classifications
        classification_counts = results_df['classification'].value_counts()
        # Define expected classification order for consistent coloring
        class_order = ['No Sand Mining Likely', 'Possible Sand Mining', 'Sand Mining Likely']
        # Define colors using a dictionary for direct mapping
        color_map = {
            'No Sand Mining Likely': '#90EE90',  # lightgreen
            'Possible Sand Mining': '#FFA500',  # orange
            'Sand Mining Likely': '#FF6347'   # tomato red
        }
        # Reindex counts based on expected order, fill missing with 0
        classification_counts = classification_counts.reindex(class_order, fill_value=0)
        # Get colors in the correct order, default to gray if classification is unexpected
        bar_colors = [color_map.get(cls, 'gray') for cls in classification_counts.index]
        
        if not classification_counts.empty:
            bars = classification_counts.plot(kind='bar', ax=axs[0, 1], color=bar_colors, edgecolor='black')
            axs[0, 1].tick_params(axis='x', rotation=15)  # Rotate labels slightly
            
            # Add counts on top of bars
            if hasattr(bars, 'patches'):  # Check if it's a bar plot with patches
                for bar in bars.patches:
                    if bar.get_height() > 0:  # Only label non-zero bars
                        axs[0, 1].text(bar.get_x() + bar.get_width() / 2,
                                      bar.get_height() + (axs[0, 1].get_ylim()[1] * 0.01),  # Adjust position
                                      f'{int(bar.get_height())}',  # Display integer count
                                      ha='center', va='bottom', fontsize=9)
        else:
            axs[0, 1].text(0.5, 0.5, "No classification data", ha='center', va='center')
        
        axs[0, 1].set_title('Point Classifications')
        axs[0, 1].set_xlabel('Classification')
        axs[0, 1].set_ylabel('Number of Points')
        axs[0, 1].grid(True, axis='y', linestyle='--', alpha=0.6)
        
        # 3. Hot Spots map (simplified)
        # Create a simple 2D histogram to show hotspots
        if 'latitude' in results_df.columns and 'longitude' in results_df.columns:
            lats = results_df['latitude'].dropna()
            lons = results_df['longitude'].dropna()
            if len(lats) > 0 and len(lons) > 0:
                heatmap, xedges, yedges = np.histogram2d(
                    lons, lats, 
                    bins=min(20, len(lats)//5 + 1),
                    weights=results_df['probability'].fillna(0)
                )
                
                # Plot heatmap
                im = axs[1, 0].imshow(
                    heatmap.T, 
                    origin='lower', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto',
                    cmap='hot'
                )
                axs[1, 0].set_title('Sand Mining Probability Hot Spots')
                axs[1, 0].set_xlabel('Longitude')
                axs[1, 0].set_ylabel('Latitude')
                plt.colorbar(im, ax=axs[1, 0], label='Probability Concentration')
            else:
                axs[1, 0].text(0.5, 0.5, "Insufficient coordinate data for heatmap", ha='center', va='center')
        else:
            axs[1, 0].text(0.5, 0.5, "No coordinate data available", ha='center', va='center')
        
        # 4. Top Contributing Features (if available)
        # Check if we have feature importance data
        try:
            feature_importance_file = config.DEFAULT_FEATURE_IMPORTANCE_FILE
            if os.path.exists(feature_importance_file):
                with open(feature_importance_file, 'r') as f:
                    importance_dict = json.load(f)
                
                # Get top 10 features
                top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                
                feature_names = [f"{name.split('_')[0]}..." if len(name) > 15 else name for name, _ in top_features]
                feature_values = [value for _, value in top_features]
                
                # Plot horizontal bar chart of feature importance
                bars = axs[1, 1].barh(feature_names, feature_values, color='teal')
                axs[1, 1].set_title('Top 10 Contributing Features')
                axs[1, 1].set_xlabel('Relative Importance')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    axs[1, 1].text(
                        bar.get_width() + 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{feature_values[i]:.3f}',
                        va='center',
                        fontsize=8
                    )
            else:
                axs[1, 1].text(0.5, 0.5, "Feature importance data not available", ha='center', va='center')
        except Exception as e:
            print(f"Error plotting feature importance: {e}")
            axs[1, 1].text(0.5, 0.5, "Error loading feature importance data", ha='center', va='center')
        
        # Overall Figure Title
        try:  # Safely get river name
            river_name = os.path.splitext(os.path.basename(shapefile_path))[0]
        except:
            river_name = "Unknown River"
        
        fig.suptitle(f'Sand Mining Analysis Summary - {river_name}\nImagery Date Range: {imagery_date_range}', fontsize=16)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close(fig)  # Close the plot figure to free memory
        
        print(f"Summary plot saved to: {output_file}")
        return output_file
    
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        import traceback
        traceback.print_exc()
        return None