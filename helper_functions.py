import pandas as pd
import glob
import os
import geopandas as gpd
from osgeo import gdal, osr
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib
import contextily as ctx
import seaborn as sns
import re  # Added import for regular expressions
import pickle
import sys  # Added import for system-specific parameters and functions

matplotlib.use('Agg')  


def standardize_headers(file_path):
    """
    Standardizes the headers of a CSV file to ensure consistency across datasets.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame with standardized headers, or None if an error occurs.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Define datetime formats for each time column
    datetime_formats = {
        'Acquisition Start Time': '%d/%m/%Y %H:%M',
        'Timestamp (GMT+2)': '%d/%m/%Y %H:%M',
        'Time Stamp UTC': '%d/%m/%Y %H:%M', 
        'Time Stamp UTC1': '%m/%d/%Y %H:%M',
        'Date': '%d/%m/%Y',
        'DATE (GMT+2)': '%Y-%m-%d',
        'Date1': '%m/%d/%y',
    }
    
    header_mapping = {
        "^Individual-local.*": "ID_Ind", 
        "^Individual-name.*": "ID_Ind", 
        "^Individual_Name.*": "ID_Ind", 
        "Location-long": "Longitude", 
        "Location-lat": "Latitude", 
        "^Longitude.*": "Longitude", 
        "^Latitude.*": "Latitude",
        "GPS Longitude": "Longitude", 
        "GPS Latitude": "Latitude",
    }
    
    parsed_datetime = False

    # Try to parse each datetime column with its specific format
    for col, fmt in datetime_formats.items():
        if col in df.columns:
            try:
                if 'Date' in df.columns and 'Time' in df.columns:
                    df['ct'] = df['Date'] + ' ' + df['Time']
                    df['t'] = pd.to_datetime(df['ct'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                    df['t'] = pd.to_datetime(df['ct'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
                    df['t'] = pd.to_datetime(df['ct'], dayfirst=True, errors='coerce')
                elif 'DATE (GMT+2)' in df.columns and 'TIME (GMT+2)' in df.columns:
                    df['ct'] = df['DATE (GMT+2)'] + ' ' + df['TIME (GMT+2)']
                    df['t'] = pd.to_datetime(df['ct'], yearfirst=True, format='%Y-%m-%d %H:%M:%S', errors='coerce')
                elif 'Date1' in df.columns and 'Time1' in df.columns:
                    df['ct'] = df['Date1'] + ' ' + df['Time1']
                    df['t'] = pd.to_datetime(df['ct'], format='%m/%d/%y %H:%M:%S', errors='coerce') 
                else:
                    df['t'] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                                
                df = df.dropna(subset=['t'])  # Drop rows where parsing failed
                df = df.set_index('t').tz_localize(None)  # Set index and remove timezone info
                parsed_datetime = True
                
                break
            except Exception as e:
                print(f"Error parsing {col}: {e}")

    if not parsed_datetime:
        print(f"Unable to parse datetime from any of the columns in {file_path}")

    # Function to rename columns based on the mapping
    def rename_column(col_name):
        for pattern, new_name in header_mapping.items():
            if re.match(pattern, col_name, re.IGNORECASE):
                return new_name
        return col_name

    # Rename the columns using the custom function
    df = df.rename(columns=rename_column)

    # Ensure the dataframe has only the standardized headers that match the mapping
    standard_headers = list(header_mapping.values())
    # Retain other necessary columns (e.g., 'TAG') if they exist
    required_columns = ['ID_Ind', 'Longitude', 'Latitude', 'TAG', 'Date', 'Time']
    # Include columns that are present in df to avoid KeyError
    standard_headers = [col for col in required_columns if col in df.columns] + [
        col for col in df.columns if col not in standard_headers and col not in required_columns
    ]
    df = df[[col for col in df.columns if col in standard_headers]]

    return df  # Return the DataFrame with 't' column set as index


def combine_csv_files(directory):
    """
    Reads all CSV files in a specified directory, standardizes their headers,
    visually inspects each DataFrame, combines the confirmed DataFrames into
    a single DataFrame, and returns it.

    Args:
        directory (str): Path to the directory containing CSV files.

    Returns:
        pd.DataFrame: A DataFrame containing the combined and standardized data from all confirmed CSV files in the directory.
    """
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {directory}")
    
    all_data_frames = []
    for file in csv_files:
        print(f"\nProcessing file: {file}")
        df = standardize_headers(file)
        if df is not None and not df.empty:
            print("Standardized DataFrame Headers:")
            print(df.columns.tolist())
            while True:
                user_input = input("Is the above header correct? [y/n]: ").strip().lower()
                if user_input == 'y':
                    all_data_frames.append(df)
                    print(f"File {file} has been accepted and will be included in the combined DataFrame.\n")
                    break
                elif user_input == 'n':
                    print(f"File {file} has been skipped based on user input.\n")
                    break
                else:
                    print("Invalid input. Please enter 'y' for yes or 'n' for no.")
        else:
            print(f"Skipping file {file} due to errors or no data after standardization.\n")
    
    if not all_data_frames:
        raise ValueError(f"No valid CSV files were confirmed after standardizing headers in directory: {directory}")
    
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    return combined_df


def convert_csv_to_geodataframe(read_csv):
    """
    Converts a DataFrame with 'Longitude' and 'Latitude' columns to a GeoDataFrame.

    Args:
        read_csv (pd.DataFrame): DataFrame containing at least 'Longitude' and 'Latitude' columns.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with points created from the 'Longitude' and 'Latitude' columns.
    """
    if 'Longitude' not in read_csv.columns or 'Latitude' not in read_csv.columns:
        raise ValueError("The CSV file does not contain 'Longitude' and 'Latitude' columns.")
    mygeometry_array = gpd.points_from_xy(read_csv['Longitude'], read_csv['Latitude'])
    return gpd.GeoDataFrame(read_csv, geometry=mygeometry_array, crs='EPSG:4326')


def write_gdf_to_csv(gdf, csv_output_file='output/output.csv'):
    """
    Writes a GeoDataFrame to a CSV file, excluding the geometry column.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to be saved.
        csv_output_file (str): Path where the CSV file will be saved.
    """
    os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
    gdf.drop('geometry', axis=1).to_csv(csv_output_file, index=False)


def export_gdf_to_gpkg(gdf, output_file='output/output.gpkg'):
    """
    Exports a GeoDataFrame to a GeoPackage file.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to be exported.
        output_file (str): Path where the GeoPackage file will be saved.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    gdf.to_file(filename=output_file, driver="GPKG")


def is_time_format_correct(time_str):
    """
    Check if the given time string is in the correct 24-hour format.

    Args:
        time_str (str): The time string to check, formatted as 'HH:MM:SS'.

    Returns:
        bool: True if the format is correct, False otherwise.
    """
    try:
        pd.to_datetime(time_str, format='%H:%M:%S', errors='raise')
        return True
    except ValueError:
        return False


def is_date_format_correct(date_str):
    """
    Check if the given date string is in the correct format.

    Args:
        date_str (str): The date string to check, formatted as 'DD/MM/YYYY'.

    Returns:
        bool: True if the format is correct, False otherwise.
    """
    try:
        pd.to_datetime(date_str, format='%d/%m/%Y', errors='raise')
        return True
    except ValueError:
        return False
    

def convert_time_date_formats(gdf):
    """
    Convert and verify the time and date formats within a GeoDataFrame.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame containing the 'Time' and 'Date' columns.
    
    Returns:
        GeoDataFrame: GeoDataFrame with converted 'Time' and 'Date' formats and a new 'DateTime' column.
    """
    gdf['Time'] = gdf['Time'].apply(
        lambda x: x if is_time_format_correct(x) else pd.to_datetime(x, format='%I:%M:%S %p', errors='coerce').strftime('%H:%M:%S') if pd.notnull(x) else x
    )
    gdf['Date'] = gdf['Date'].apply(
        lambda x: x if is_date_format_correct(x) else pd.to_datetime(x, errors='coerce').strftime('%d/%m/%Y') if pd.notnull(x) else x
    )
    datetime_format = '%d/%m/%Y %H:%M:%S'
    gdf['DateTime'] = pd.to_datetime(gdf['Date'] + ' ' + gdf['Time'], format=datetime_format, errors='coerce')
    return gdf


def filter_gdf(gdf, animal_id, time_start, time_end):
    """
    Filter GeoDataFrame based on animal ID and a time range.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame to filter.
        animal_id (str): Animal ID to filter by.
        time_start (str or datetime): Start time for filtering.
        time_end (str or datetime): End time for filtering.
    
    Returns:
        GeoDataFrame: Filtered GeoDataFrame.
    """
    time_start = pd.to_datetime(time_start)
    time_end = pd.to_datetime(time_end)
    
    gdf_filtered = gdf[(gdf['TAG'] == animal_id) & 
                       (gdf['DateTime'] >= time_start) &
                       (gdf['DateTime'] <= time_end)]
    if gdf_filtered.empty:
        raise ValueError("No data available for the specified animal and timeframe.")
    return gdf_filtered

def spatial_kernel_density(gdf, cutoff_list, bw_method=0.2):
    """
    Calculates Kernel Density Estimates for specified levels and extracts contour polygons.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame containing point geometries.
        cutoff_list (list): List of levels for KDE contour extraction.
        bw_method (float, optional): Bandwidth method for KDE. Defaults to 0.2.
    
    Returns:
        GeoDataFrame: GeoDataFrame containing contour polygons with associated levels.
    """
    import seaborn as sns  # Ensure seaborn is imported
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    import matplotlib

    level_polygons = []

    for level in cutoff_list:
        plt.figure()
        kde_plot = sns.kdeplot(
            x=gdf.geometry.x, 
            y=gdf.geometry.y, 
            levels=[level, 1],
            bw_adjust=bw_method,
            fill=True,
            thresh=0,
            cmap="Reds"
        )
        
        # Extract contours
        for collection in kde_plot.collections:
            for path in collection.get_paths():
                vertices = path.vertices
                if len(vertices) < 3:
                    continue  # Not a valid polygon
                polygon = Polygon(vertices)
                if polygon.is_valid:
                    level_polygons.append({"level": level, "geometry": polygon})
        
        plt.close()  # Close the figure to free memory

    if level_polygons:
        polygons_gdf = gpd.GeoDataFrame(level_polygons, geometry="geometry", crs=gdf.crs)
    else:
        polygons_gdf = gpd.GeoDataFrame(columns=["level", "geometry"], crs=gdf.crs)

    return polygons_gdf

def calculate_spatial_utilization(gdf_filtered, bw_method=0.2, cutoff_levels=[0.2, 0.4, 0.6, 0.8]):
    """
    Calculate spatial utilization metrics including convex hull and KDE polygons for a filtered GeoDataFrame.
    
    Args:
        gdf_filtered (GeoDataFrame): Filtered GeoDataFrame containing spatial data.
        bw_method (float, optional): Bandwidth method for KDE. Defaults to 0.2.
        cutoff_levels (list, optional): Levels for KDE contour extraction. Defaults to [0.2, 0.4, 0.6, 0.8].
    
    Returns:
        dict: Dictionary containing the convex hull, circumference, KDE values, grid, areas, and KDE polygons.
    """
    # Convex Hull and Circumference
    points = np.array(list(zip(gdf_filtered.geometry.x, gdf_filtered.geometry.y)))
    hull = ConvexHull(points)
    convex_hull_polygon = Polygon([points[vertex] for vertex in hull.vertices])
    circumference = convex_hull_polygon.length
    # Kernel Density Estimation using spatial_kernel_density
    polygons_gdf = spatial_kernel_density(gdf_filtered, cutoff_levels, bw_method=bw_method)

    # Calculating areas for each level
    area_dict = {}
    for level in cutoff_levels:
        polygons = polygons_gdf[polygons_gdf['level'] == level]
        if not polygons.empty:
            # Union of all polygons at this level
            union_polygon = polygons.unary_union
            area = union_polygon.area
            area_dict[f'area_{int(level*100)}'] = area
        else:
            area_dict[f'area_{int(level*100)}'] = 0.0

    return {
        'convex_hull_polygon': convex_hull_polygon, 
        'circumference': circumference,
        'kde_polygons': polygons_gdf,
        'areas': area_dict
    }


def plot_spatial_utilization(gdf_filtered, spatial_metrics, animal_id, time_start, time_end, output_file):
    """
    Plot spatial utilization with a heatmap, convex hull, and KDE polygons.
    
    Args:
        gdf_filtered (GeoDataFrame): Filtered GeoDataFrame containing spatial data.
        spatial_metrics (dict): Dictionary containing spatial metrics including the convex hull and KDE polygons.
        animal_id (str): Animal ID.
        time_start (str): Start time of the dataset.
        time_end (str): End time of the dataset.
        output_file (str): Path where the plot image will be saved.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot KDE polygons with transparency
    spatial_metrics['kde_polygons'].plot(ax=ax, column='level', cmap='Reds', alpha=0.3, legend=True, label='KDE Polygons')
    
    # Plot filtered points
    gdf_filtered.plot(ax=ax, color='blue', markersize=10, label='Filtered Points')
    
    # Plot convex hull
    gpd.GeoSeries([spatial_metrics['convex_hull_polygon']]).plot(ax=ax, color='none', edgecolor='green', linewidth=2, label='Convex Hull')
    
    # Add basemap
    buffer = 0.1
    xmin, xmax = gdf_filtered.geometry.x.min(), gdf_filtered.geometry.x.max()
    ymin, ymax = gdf_filtered.geometry.y.min(), gdf_filtered.geometry.y.max()
    ax.set_xlim([xmin - buffer * (xmax - xmin), xmax + buffer * (xmax - xmin)])
    ax.set_ylim([ymin - buffer * (ymax - ymin), ymax + buffer * (ymax - ymin)])
    ctx.add_basemap(ax, crs=gdf_filtered.crs.to_string(), zoom=12)
    
    # Set title and remove axes
    ax.set_title(f"Spatial Utilization for {animal_id} from {time_start} to {time_end}")
    ax.set_axis_off()
    
    # Create custom legend
    handles, labels = ax.get_legend_handles_labels()
    custom_handles = [
        plt.Line2D([0], [0], color='blue', marker='o', linestyle='', markersize=10, label='Filtered Points'),
        plt.Line2D([0], [0], color='green', lw=2, label='Convex Hull'),
        plt.Line2D([0], [0], color='red', alpha=0.3, lw=10, label='KDE Polygons')
    ]
    ax.legend(handles=custom_handles, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

