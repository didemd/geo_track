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
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from shapely.geometry import Polygon, MultiPolygon
import contextily as ctx

matplotlib.use('Agg')  


def get_csv_heads(directory):
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {directory}")
    all_data_frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_data_frames.append(df)
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    print("Combined DataFrame:")
    print(combined_df.head())
    return combined_df

def convert_csv_to_geodataframe(read_csv):
    if 'Longitude' not in read_csv.columns or 'Latitude' not in read_csv.columns:
        raise ValueError("The CSV file does not contain 'Longitude' and 'Latitude' columns.")
    mygeometry_array = gpd.points_from_xy(read_csv['Longitude'], read_csv['Latitude'])
    df_gdf = gpd.GeoDataFrame(read_csv, geometry=mygeometry_array)
    df_gdf.crs = 'EPSG:4326'
    print("GeoDataFrame Head:")
    print(df_gdf.head())
    return df_gdf

def write_gdf_to_csv(gdf, csv_output_file='output/output.csv'):
    os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
    gdf.drop('geometry', axis=1).to_csv(csv_output_file, index=False)
    print(f"GeoDataFrame saved as CSV at {csv_output_file}")

def export_gdf_to_gpkg(gdf, output_file='output/output.gpkg'):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    gdf.to_file(filename=output_file, driver="GPKG")
    print(f"GeoDataFrame saved as {output_file}")

def is_time_format_correct(time_str):
    try:
        # Try parsing with the 24-hour format
        pd.to_datetime(time_str, format='%H:%M:%S', errors='raise')
        return True
    except ValueError:
        return False

def is_date_format_correct(date_str):
    try:
        # Try parsing with the desired date format
        pd.to_datetime(date_str, format='%d/%m/%Y', errors='raise')
        return True
    except ValueError:
        return False

def compute_and_visualize_spatial_utilization(gdf, animal_id, time_start, time_end):

    print(gdf['Individual_Name'].unique())
    print(gdf['Date'].min(), gdf['Date'].max())
    print("Converted Start Date:", time_start)
    print("Converted End Date:", time_end)
    print(gdf['Time'])

    # Correct format for 24-hour time without AM/PM
    gdf['Time'] = gdf['Time'].apply(lambda x: x if is_time_format_correct(x) else pd.to_datetime(x, format='%I:%M:%S %p', errors='coerce').strftime('%H:%M:%S'))
    # Check and convert Date if not in the correct format
    gdf['Date'] = gdf['Date'].apply(lambda x: x if is_date_format_correct(x) else pd.to_datetime(x, errors='coerce').strftime('%d/%m/%Y'))
    # Merge Time and Date into a new DateTime column
    datetime_format = '%d/%m/%Y %H:%M:%S'
    gdf['DateTime'] = pd.to_datetime(gdf['Date'] + ' ' + gdf['Time'], format=datetime_format, errors='coerce')
    print(gdf['Time'])
    print(gdf['DateTime'])

    gdf_filtered = gdf[(gdf['TAG'] == animal_id) & 
                        (gdf['DateTime'] >= time_start) &
                        (gdf['DateTime'] <= time_end)]
    
    if gdf_filtered.empty:
          raise ValueError("No data available for the specified animal and timeframe.")

    # Convex Hull
    points = np.array(list(zip(gdf_filtered.geometry.x, gdf_filtered.geometry.y)))
    hull = ConvexHull(points)
    convex_hull_polygon = Polygon([points[vertex] for vertex in hull.vertices])

    # Calculate circumference (perimeter) of the convex hull
    circumference = convex_hull_polygon.length

    # KDE calculation
    values = np.vstack((points[:, 0], points[:, 1]))  # Organizing x and y coordinates
    kde = gaussian_kde(values, bw_method=0.7)

    # Create grid for KDE evaluation
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kde_values = kde(positions).reshape(xx.shape)

    # Calculate density thresholds
    level_95 = np.percentile(kde_values, 5)
    level_50 = np.percentile(kde_values, 50)

    # Area calculations
    area_95 = np.sum(kde_values > level_95) * (xmax - xmin) * (ymax - ymin) / kde_values.size
    area_50 = np.sum(kde_values > level_50) * (xmax - xmin) * (ymax - ymin) / kde_values.size

    # Set up plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.contourf(xx, yy, kde_values, levels=[level_95, level_50, kde_values.max()], colors=['orange', 'red', 'darkred'], alpha=0.5)
    gdf_filtered.plot(ax=ax, color='blue', markersize=10, label='Filtered Points')
    gpd.GeoSeries([convex_hull_polygon]).plot(ax=ax, color='none', edgecolor='green', linewidth=2, label='Convex Hull')

    # Adjust zoom and add basemap
    buffer = 0.1
    ax.set_xlim([xmin - buffer * (xmax - xmin), xmax + buffer * (xmax - xmin)])
    ax.set_ylim([ymin - buffer * (ymax - ymin), ymax + buffer * (ymax - ymin)])
    ctx.add_basemap(ax, crs=gdf_filtered.crs.to_string(), zoom=12)

    # Title and legend with area and circumference
    ax.set_title(f"Spatial Utilization for {animal_id} from {time_start} to {time_end}")
    ax.set_axis_off()
    ax.legend(handles=[
        plt.Line2D([0], [0], color='orange', lw=4, label=f'50% Density: {area_50:.2f} sq. units'),
        plt.Line2D([0], [0], color='red', lw=4, label=f'95% Density: {area_95:.2f} sq. units'),
        plt.Line2D([0], [0], color='green', lw=4, label=f'Circumference: {circumference:.2f} units')
    ])

    # Save and close
    plt.tight_layout()
    plt.savefig('output/plot_spatial_utilization_heatmap_with_map.png', dpi=300)
    plt.close(fig)