import pandas as pd
import glob
import os
import geopandas as gpd
from osgeo import gdal, osr
import numpy as np
from scipy.stats import gaussian_kde
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import MultiPolygon, Polygon
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity
from shapely.ops import unary_union

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
    gdf['DateTime'] = pd.to_datetime(gdf['Date'] + ' ' + gdf['Time'])
    print(gdf['Time'])
    gdf['Date'] = pd.to_datetime(gdf['Date'], errors='coerce').dt.strftime('%d/%m/%Y')
    # gdf['DateTime'] = pd.to_datetime(gdf['Date'] + ' ' + gdf['Time'])
    # print(gdf['DateTime'])

    # gdf_filtered = gdf[(gdf['Individual_Name'] == animal_id) & 
    #                    (gdf['DateTime'] >= time_start) &
    #                    (gdf['DateTime'] <= time_end)]
    
    # if gdf_filtered.empty:
    #      raise ValueError("No data available for the specified animal and timeframe.")

    # # Convex Hull
    # points = np.array(list(zip(gdf_filtered.geometry.x, gdf_filtered.geometry.y)))
    # hull = ConvexHull(points)
    # convex_hull_polygon = Polygon([points[vertex] for vertex in hull.vertices])
    
    # # Kernel Density Estimation
    # bandwidth = 0.01  # bandwidth affects smoothness, adjust based on your dataset
    # kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(points)
    # scores = kde.score_samples(points)
    # threshold_95 = np.percentile(scores, 5)
    # threshold_50 = np.percentile(scores, 50)
    # high_density_area_95 = MultiPolygon([Polygon(points[scores > threshold_95])])
    # high_density_area_50 = MultiPolygon([Polygon(points[scores > threshold_50])])
    
    # # Plotting
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # gdf_filtered.plot(ax=ax, color='blue', markersize=10)
    # gpd.GeoSeries([convex_hull_polygon]).plot(ax=ax, color='none', edgecolor='red', linewidth=2)
    # gpd.GeoSeries(high_density_area_95).plot(ax=ax, color='red', alpha=0.5)
    # gpd.GeoSeries(high_density_area_50).plot(ax=ax, color='red', alpha=0.8)
    
    # ctx.add_basemap(ax, crs=gdf_filtered.crs.to_string())
    # ax.set_title(f"Spatial Utilization for {animal_id} from {time_start} to {time_end}")
    # plt.tight_layout()
    # plt.show()

    # # Compute metrics
    # area = convex_hull_polygon.area
    # perimeter = convex_hull_polygon.length
    # area_to_perimeter_ratio = area / perimeter if perimeter else 0

    # return {'Area': area, 'Perimeter': perimeter, 'Area_to_Perimeter_Ratio': area_to_perimeter_ratio}