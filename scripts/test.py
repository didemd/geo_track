import pandas as pd
import numpy as np
import os
import re
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import contextily as ctx
from scipy.spatial import ConvexHull
import seaborn as sns
import geemap  # cartoee is part of geemap
import ee

# Initialize Google Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-didemdostt')

def standardize_headers(file_path):
    """
    Standardizes the headers of a CSV file to ensure consistency across datasets.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame with standardized headers, or None if an error occurs.
    """
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
    
    # Define header mappings using regex patterns
    header_mapping = {
        "^Individual-local.*": "ID_Ind", 
        "^Individual-name.*": "ID_Ind", 
        "^Individual_Name.*": "ID_Ind", 
        "Location-long": "LONGITUDE", 
        "Location-lat": "LATITUDE", 
        "^Longitude.*": "LONGITUDE", 
        "^Latitude.*": "LATITUDE",
        "GPS Longitude": "LONGITUDE", 
        "GPS Latitude": "LATITUDE",
    }
    
    parsed_datetime = False

    # Attempt to parse datetime columns
    for col, fmt in datetime_formats.items():
        if col in df.columns:
            try:
                df['t'] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                if df['t'].isna().all():
                    continue
                df = df.dropna(subset=['t'])  # Drop rows where parsing failed
                df['t'] = df['t'].dt.tz_localize(None)  # Remove timezone info if any
                parsed_datetime = True
                break
            except Exception as e:
                print(f"Error parsing {col}: {e}")

    # Additional Parsing if specific columns exist
    if not parsed_datetime and 'Date' in df.columns and 'Time' in df.columns:
        try:
            df['ct'] = df['Date'] + ' ' + df['Time']
            df['t'] = pd.to_datetime(df['ct'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['t'])
            df['t'] = df['t'].dt.tz_localize(None)
            parsed_datetime = True
        except Exception as e:
            print(f"Error parsing 'Date' and 'Time' columns: {e}")
    
    if not parsed_datetime:
        print(f"Unable to parse datetime from any of the columns in {file_path}")

    # Rename the columns using the mapping
    def rename_column(col_name):
        for pattern, new_name in header_mapping.items():
            if re.match(pattern, col_name, re.IGNORECASE):
                return new_name
        return col_name

    df = df.rename(columns=rename_column)
    print("DataFrame after header standardization:")
    print(df.head())

    # Ensure the DataFrame has the necessary columns
    required_columns = ['ID_Ind', 'LONGITUDE', 'LATITUDE', 't']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None

    df = df.dropna(subset=required_columns)
    return df

def convert_csv_to_geodataframe(df):
    """
    Converts a DataFrame with 'LONGITUDE' and 'LATITUDE' columns to a GeoDataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'LONGITUDE' and 'LATITUDE' columns.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with points created from the 'LONGITUDE' and 'LATITUDE' columns.
    """
    print("Starting conversion to GeoDataFrame...")
    
    try:
        geometry = gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE'])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        print("GeoDataFrame successfully created:")
        print(gdf.head())
        return gdf
    except Exception as e:
        print(f"Error during GeoDataFrame creation: {e}")
        return None

def write_gdf_to_csv(gdf, csv_output_file):
    """
    Writes a GeoDataFrame to a CSV file, excluding the geometry column.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to be saved.
        csv_output_file (str): Path where the CSV file will be saved.
    """
    print(f"Writing GeoDataFrame to CSV at: {csv_output_file}")
    try:
        gdf.drop(columns='geometry').to_csv(csv_output_file, index=False)
        print("GeoDataFrame successfully written to CSV.")
    except Exception as e:
        print(f"Failed to write GeoDataFrame to CSV: {e}")

def gdf_to_gpkg(gdf, output_file):
    """
    Exports a GeoDataFrame to a GeoPackage file.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to be exported.
        output_file (str): Path where the GeoPackage file will be saved.
    """
    print(f"Writing GeoDataFrame to GeoPackage at: {output_file}")
    try:
        gdf.to_file(filename=output_file, driver="GPKG")
        print("GeoDataFrame successfully written to GeoPackage.")
    except Exception as e:
        print(f"Failed to write GeoDataFrame to GeoPackage: {e}")

def filter_gdf(gdf, animal_id, time_start, time_end=None, exact=False):
    """
    Filter GeoDataFrame based on animal ID and a time range or exact time.

    Args:
        gdf (GeoDataFrame): GeoDataFrame to filter.
        animal_id (str): Animal ID to filter by.
        time_start (str or datetime): Start time for filtering.
        time_end (str or datetime, optional): End time for filtering.
        exact (bool, optional): If True, filter for exact timestamp.

    Returns:
        GeoDataFrame: Filtered GeoDataFrame.
    """
    time_start = pd.to_datetime(time_start)
    if time_end:
        time_end = pd.to_datetime(time_end)

    if exact:
        gdf_filtered = gdf[(gdf['ID_Ind'] == animal_id) & (gdf['t'] == time_start)]
    elif time_end:
        gdf_filtered = gdf[(gdf['ID_Ind'] == animal_id) & (gdf['t'] >= time_start) & (gdf['t'] <= time_end)]
    else:
        gdf_filtered = gdf[(gdf['ID_Ind'] == animal_id) & (gdf['t'] >= time_start)]
    
    if gdf_filtered.empty:
        raise ValueError(f"No data available for animal '{animal_id}' in the specified timeframe.")
    return gdf_filtered

def calculate_spatial_utilization(gdf_filtered, bw_method):
    """
    Calculate spatial utilization metrics such as convex hull and KDE for a filtered GeoDataFrame.

    Args:
        gdf_filtered (GeoDataFrame): Filtered GeoDataFrame containing spatial data.
        bw_method (float): Bandwidth method for Gaussian KDE.

    Returns:
        dict or None: Dictionary containing spatial metrics, or None if insufficient points.
    """
    points = np.vstack([gdf_filtered.geometry.x, gdf_filtered.geometry.y])

    if points.shape[1] < 3:
        print("Not enough points to calculate spatial utilization metrics.")
        return None

    try:
        kde = gaussian_kde(points, bw_method=bw_method)
        xmin, ymin = points.min(axis=1)
        xmax, ymax = points.max(axis=1)
        xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])
        kde_values = kde(grid_coords).reshape(xx.shape)

        level_95 = np.percentile(kde_values, 5)
        level_50 = np.percentile(kde_values, 50)

        hull = ConvexHull(points.T)
        convex_hull_polygon = Polygon(points.T[hull.vertices])

        return {
            'kde_values': kde_values,
            'xx': xx,
            'yy': yy,
            'level_95': level_95,
            'level_50': level_50,
            'convex_hull_polygon': convex_hull_polygon,
            'circumference': convex_hull_polygon.length
        }
    except Exception as e:
        print(f"Error calculating spatial utilization: {e}")
        return None

def spatial_kernel_density(gdf, cutoff_list):
    """
    Perform Spatial Kernel Density Estimation (Method 2) and extract contour polygons.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing spatial data.
        cutoff_list (list of float): List of density levels to extract contours.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing contour polygons with associated density levels.
    """
    level_polygons = []

    for level in cutoff_list:
        plt.figure(figsize=(8, 6))
        kde_plot = sns.kdeplot(x=gdf.geometry.x, y=gdf.geometry.y, levels=[level, 1], fill=True, cmap="Reds")
        plt.close()

        for collection in kde_plot.collections:
            for path in collection.get_paths():
                # Each path can consist of multiple polygons
                for polygon in path.to_polygons():
                    if len(polygon) < 3:
                        continue  # Not a valid polygon
                    new_shape = Polygon(polygon)
                    if not new_shape.is_valid:
                        new_shape = new_shape.buffer(0)
                        if not new_shape.is_valid:
                            continue  # Skip invalid geometries
                    level_polygons.append({"level": level, "geometry": new_shape})
    
    if not level_polygons:
        print("No contour polygons were generated.")
        return None

    contour_gdf = gpd.GeoDataFrame(level_polygons, geometry="geometry", crs=gdf.crs)
    return contour_gdf

def plot_spatial_utilization_method1(gdf_filtered, spatial_metrics, animal_id, time_start, time_end=None, output_dir="/Users/didemdost/Desktop/geo_track/output"):
    """
    Plot Spatial Utilization Metrics (Method 1).

    Args:
        gdf_filtered (GeoDataFrame): Filtered GeoDataFrame containing spatial data.
        spatial_metrics (dict): Dictionary containing spatial metrics.
        animal_id (str): Animal ID.
        time_start (datetime): Start time of the dataset.
        time_end (datetime, optional): End time of the dataset.
        output_dir (str): Directory where the plot will be saved.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.contourf(spatial_metrics['xx'], spatial_metrics['yy'], spatial_metrics['kde_values'],
                levels=[spatial_metrics['level_95'], spatial_metrics['level_50'], spatial_metrics['kde_values'].max()],
                colors=['orange', 'red', 'darkred'], alpha=0.5)

    gdf_filtered.plot(ax=ax, color='blue', markersize=5, label='Animal Positions')
    gpd.GeoSeries([spatial_metrics['convex_hull_polygon']]).plot(ax=ax, edgecolor='green', linewidth=2, label='Convex Hull', facecolor='none')

    buffer = 0.1
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min - buffer * abs(x_max - x_min), x_max + buffer * abs(x_max - x_min))
    ax.set_ylim(y_min - buffer * abs(y_max - y_min), y_max + buffer * abs(y_max - y_min))

    try:
        ctx.add_basemap(ax, crs=gdf_filtered.crs.to_string(), source=ctx.providers.Stamen.TonerLite)
    except Exception as e:
        print(f"Error adding basemap: {e}")
        print("Proceeding without basemap.")

    title = f"Spatial Utilization for {animal_id} from {time_start}"
    if time_end:
        title += f" to {time_end}"
    ax.set_title(title)
    ax.axis('off')
    ax.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, f'spatial_utilization_method1_{animal_id}.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Spatial utilization plot (Method 1) saved to {plot_file}")

def plot_spatial_kernel_density_method2(contour_gdf, animal_id, time_start, time_end=None, output_dir="/Users/didemdost/Desktop/geo_track/output"):
    """
    Plot Spatial Kernel Density Estimation Contours (Method 2) using cartoee.

    Args:
        contour_gdf (GeoDataFrame): GeoDataFrame containing contour polygons.
        animal_id (str): Animal ID.
        time_start (datetime): Start time of the dataset.
        time_end (datetime, optional): End time of the dataset.
        output_dir (str): Directory where the plot will be saved.
    """
    if contour_gdf is None or contour_gdf.empty:
        print("No contour polygons to plot for Method 2.")
        return

    # Initialize the map with Google Hybrid basemap
    center_x = contour_gdf.geometry.x.mean()
    center_y = contour_gdf.geometry.y.mean()
    Map = geemap.Map(center=[center_y, center_x], zoom=10, basemap='HYBRID')

    # Add the contour polygons
    Map.add_gdf(contour_gdf, layer_name='Kernel Density Contours', fill_color='red', stroke_color='black', opacity=0.5)

    # Add map elements
    geemap.add_scale_bar(Map, position=(0.1, 0.05))
    geemap.add_north_arrow(Map, position=(0.1, 0.1))
    
    # Create a legend manually
    legend_dict = {0.2: 'Low Density', 0.4: 'Medium Density', 0.6: 'High Density', 0.8: 'Very High Density'}
    geemap.add_legend(Map, legend_dict=legend_dict, title='Density Levels')

    # Save the map as HTML
    map_file = os.path.join(output_dir, f'kernel_density_map_method2_{animal_id}.html')
    Map.to_html(map_file)
    print(f"Kernel density map (Method 2) saved to {map_file}")

    # Optionally, display the map in a Jupyter Notebook
    # Map

def main():
    # Input CSV file
    file_path = "/Users/didemdost/Desktop/geo_track/data/NPL28.csv"

    # Step 1: Standardize Headers and Read Data
    df = standardize_headers(file_path)
    if df is None or df.empty:
        print("No data to process.")
        return

    # Step 2: Convert DataFrame to GeoDataFrame
    gdf = convert_csv_to_geodataframe(df)
    if gdf is None or gdf.empty:
        print("Failed to create GeoDataFrame.")
        return

    # Step 3: Reproject to EPSG:3857
    gdf = gdf.to_crs(epsg=3857)
    print("GeoDataFrame reprojected to EPSG:3857.")
    print("CRS after reprojection:", gdf.crs)

    # Step 4: Export GeoDataFrame to CSV and GeoPackage
    output_dir = "/Users/didemdost/Desktop/geo_track/output"
    os.makedirs(output_dir, exist_ok=True)
    csv_output_file = os.path.join(output_dir, 'output.csv')
    gpkg_output_file = os.path.join(output_dir, 'output.gpkg')
    write_gdf_to_csv(gdf, csv_output_file)
    gdf_to_gpkg(gdf, gpkg_output_file)

    # Step 5: Filter GeoDataFrame
    animal_id = 'NPL28_'  # Adjust as necessary

    time_start = '2023-04-01 12:00:20'  # Adjust as necessary
    time_end = '2023-04-30 22:00:32'    # Adjust as necessary

    try:
        gdf_filtered = filter_gdf(gdf, animal_id, time_start, time_end)
        print("Filtered GeoDataFrame:")
        print(gdf_filtered)
        print(f"Number of records after filtering: {len(gdf_filtered)}")
        print(f"CRS of filtered data: {gdf_filtered.crs}")
        print(f"Bounds of filtered data: {gdf_filtered.total_bounds}")
    except ValueError as e:
        print(e)
        return

    # Step 6: Calculate Spatial Utilization Metrics (Method 1)
    spatial_metrics = calculate_spatial_utilization(gdf_filtered, bw_method=0.2)
    if spatial_metrics is None:
        print("Could not calculate spatial utilization metrics.")
    else:
        # Step 7: Plot Spatial Utilization (Method 1)
        plot_spatial_utilization_method1(gdf_filtered, spatial_metrics, animal_id, time_start, time_end, output_dir)

    # Step 8: Spatial Kernel Density Estimation (Method 2)
    cutoff_levels = [0.2, 0.4, 0.6, 0.8]  # Adjust density levels as needed
    print("\nStarting Spatial Kernel Density Estimation (Method 2)...")
    contour_gdf = spatial_kernel_density(gdf_filtered, cutoff_levels)
    
    if contour_gdf is not None and not contour_gdf.empty:
        # Add ID and name columns for ArcGIS compatibility
        contour_gdf['ID'] = animal_id
        contour_gdf['name'] = animal_id  # Duplicate if required

        # Reproject to WGS84 for compatibility with most GIS software
        contour_gdf = contour_gdf.to_crs(epsg=4326)

        # Export to GeoPackage (instead of Shapefile for consistency)
        gpkg_contour_output = os.path.join(output_dir, 'all_kdes.gpkg')
        try:
            contour_gdf.to_file(gpkg_contour_output, driver="GPKG")
            print(f"Spatial Kernel Density contours successfully saved to {gpkg_contour_output}")
        except Exception as e:
            print(f"Failed to write Spatial Kernel Density contours to GeoPackage: {e}")
        
        # Step 9: Plot Spatial Kernel Density Estimation (Method 2) using cartoee
        # Reproject contour_gdf back to EPSG:3857 for plotting with cartoee
        #contour_gdf_plot = contour_gdf.to_crs(epsg=3857)
        #plot_spatial_kernel_density_method2(contour_gdf_plot, animal_id, time_start, time_end, output_dir)
    else:
        print("No contour polygons were generated in Spatial Kernel Density Estimation.")

if __name__ == "__main__":
    main()
