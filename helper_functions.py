import pandas as pd
import glob
import os
import geopandas as gpd
from osgeo import gdal, osr
import numpy as np
from scipy.stats import gaussian_kde
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
import contextily as ctx
from shapely.geometry import MultiPolygon, Polygon
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity
from shapely.ops import unary_union
import seaborn as sns
import numpy as np
import pickle

matplotlib.use('Agg')  


def combine_csv_files(directory):
    """
    Reads all CSV files in a specified directory, combines them into a single DataFrame, and returns it.

    Args:
    directory (str): Path to the directory containing CSV files.

    Returns:
    pd.DataFrame: A DataFrame containing the combined data from all CSV files in the directory.

    """
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {directory}")
    all_data_frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_data_frames.append(df)
    return pd.concat(all_data_frames, ignore_index=True)

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
    gdf['Time'] = gdf['Time'].apply(lambda x: x if is_time_format_correct(x) else pd.to_datetime(x, format='%I:%M:%S %p', errors='coerce').strftime('%H:%M:%S'))
    gdf['Date'] = gdf['Date'].apply(lambda x: x if is_date_format_correct(x) else pd.to_datetime(x, errors='coerce').strftime('%d/%m/%Y'))
    datetime_format = '%d/%m/%Y %H:%M:%S'
    gdf['DateTime'] = pd.to_datetime(gdf['Date'] + ' ' + gdf['Time'], format=datetime_format, errors='coerce')
    return gdf

def filter_gdf(gdf, animal_id, time_start, time_end):
    """
    Filter GeoDataFrame based on animal ID and a time range.
    
    Args:
    gdf (GeoDataFrame): GeoDataFrame to filter.
    animal_id (str): Animal ID to filter by.
    time_start (datetime): Start time for filtering.
    time_end (datetime): End time for filtering.
    
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

def calculate_spatial_utilization(gdf_filtered,bw_method):
    """
    Calculate spatial utilization metrics such as convex hull and KDE for a filtered GeoDataFrame.
    
    Args:
    gdf_filtered (GeoDataFrame): Filtered GeoDataFrame containing spatial data.
    
    Returns:
    dict: Dictionary containing the convex hull, KDE values, and other spatial metrics.
    """
    points = np.array(list(zip(gdf_filtered.geometry.x, gdf_filtered.geometry.y)))
    hull = ConvexHull(points)
    convex_hull_polygon = Polygon([points[vertex] for vertex in hull.vertices])
    circumference = convex_hull_polygon.length

    values = np.vstack((points[:, 0], points[:, 1]))
    kde = gaussian_kde(values, bw_method=bw_method)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kde_values = kde(positions).reshape(xx.shape)

    level_95 = np.percentile(kde_values, 5)
    level_50 = np.percentile(kde_values, 50)

    area_95 = np.sum(kde_values > level_95) * (xmax - xmin) * (ymax - ymin) / kde_values.size
    area_50 = np.sum(kde_values > level_50) * (xmax - xmin) * (ymax - ymin) / kde_values.size

    return {
        'convex_hull_polygon': convex_hull_polygon, 
        'circumference': circumference,
        'kde_values': kde_values,
        'xx': xx, 
        'yy': yy,
        'area_95': area_95,
        'area_50': area_50,
        'level_95': level_95,
        'level_50': level_50
    }

def plot_spatial_utilization(gdf_filtered, spatial_metrics, animal_id, time_start, time_end, output_file):
    """
    Plot spatial utilization with a heatmap and convex hull.

    Args:
        gdf_filtered (GeoDataFrame): Filtered GeoDataFrame containing spatial data.
        spatial_metrics (dict): Dictionary containing spatial metrics including the convex hull and KDE values.
        animal_id (str): Animal ID.
        time_start (str): Start time of the dataset.
        time_end (str): End time of the dataset.
        output_file (str): Path where the plot image will be saved.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.contourf(
        spatial_metrics['xx'], spatial_metrics['yy'], spatial_metrics['kde_values'],
        levels=[spatial_metrics['level_95'], spatial_metrics['level_50'], spatial_metrics['kde_values'].max()],
        colors=['orange', 'red', 'darkred'], alpha=0.5
    )
    gdf_filtered.plot(ax=ax, color='blue', markersize=10, label='Filtered Points')
    gpd.GeoSeries([spatial_metrics['convex_hull_polygon']]).plot(ax=ax, color='none', edgecolor='green', linewidth=2, label='Convex Hull')

    buffer = 0.1
    xmin, xmax = spatial_metrics['xx'].min(), spatial_metrics['xx'].max()
    ymin, ymax = spatial_metrics['yy'].min(), spatial_metrics['yy'].max()
    ax.set_xlim([xmin - buffer * (xmax - xmin), xmax + buffer * (xmax - xmin)])
    ax.set_ylim([ymin - buffer * (ymax - ymin), ymax + buffer * (ymax - ymin)])
    ctx.add_basemap(ax, crs=gdf_filtered.crs.to_string(), zoom=12)

    ax.set_title(f"Spatial Utilization for {animal_id} from {time_start} to {time_end}")
    ax.set_axis_off()
    ax.legend(handles=[
        plt.Line2D([0], [0], color='orange', lw=4, label=f'50% Density: {spatial_metrics["area_50"]:.2f} sq. units'),
        plt.Line2D([0], [0], color='red', lw=4, label=f'95% Density: {spatial_metrics["area_95"]:.2f} sq. units'),
        plt.Line2D([0], [0], color='green', lw=4, label=f'Circumference: {spatial_metrics["circumference"]:.2f} units')
    ])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)  
    plt.close(fig)
