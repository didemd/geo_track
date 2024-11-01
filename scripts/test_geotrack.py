import geopandas as gpd
import pandas as pd
import re
import pyproj
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import contextily as ctx
import seaborn as sns
import geemap
import ee
from ipyleaflet import ScaleControl
import cartoee
import cartoee.plotting as cplot
import cartopy.crs as ccrs
import os

# Set pyproj data directory (adjust path as necessary)
pyproj.datadir.set_data_dir('Users/didemdost/opt/anaconda3/envs/snakemake-env/lib/python3.12/site-packages/pyproj/proj_dir/share/proj')
os.environ['PROJ_LIB'] = '/Users/didemdost/opt/anaconda3/envs/snakemake-env/lib/python3.12/site-packages/pyproj/proj_dir/share/proj'

def standardize_headers(file_path):
    """
    Standardizes the headers of a CSV file to ensure consistency across datasets.
    Checks for unexpected headers and allows user to confirm correct parsing.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame or None: A DataFrame with standardized headers, or None if an error occurs or user aborts.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    print("\n--- Initial DataFrame ---")
    print(df.head())
    print(f"Columns: {df.columns.tolist()}\n")

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
                print(f"Parsed datetime using column: {col}")
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
            print("Parsed datetime using 'Date' and 'Time' columns.")
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
    print("\n--- DataFrame after Header Standardization ---")
    print(df.head())
    print(f"Columns: {df.columns.tolist()}\n")

    # Check for any unexpected headers that were not renamed
    expected_headers = set(header_mapping.values()).union({'t', 'ct', 'Index', 'LoadedBatV', 'UnloadedBatv', 'Mortality', 'ExtFixRequest'})
    actual_headers = set(df.columns)
    unexpected_headers = actual_headers - expected_headers

    if unexpected_headers:
        print(f"Warning: The following columns are unexpected and were not renamed: {unexpected_headers}")
    else:
        print("All headers are as expected.")

    # Visual Inspection: Prompt user to confirm DataFrame correctness
    while True:
        user_input = input("Is the DataFrame correctly read and headers standardized? [y/n]: ").strip().lower()
        if user_input == 'y':
            print("Proceeding with the standardized DataFrame.")
            break
        elif user_input == 'n':
            print("Aborting the process. Please check the CSV file and try again.")
            return None
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")

    # Ensure the DataFrame has the necessary columns
    required_columns = ['ID_Ind', 'LONGITUDE', 'LATITUDE', 't']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None

    # Format the 't' column to the desired string format
    if pd.api.types.is_datetime64_any_dtype(df['t']):
        df['t'] = df['t'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        print("'t' column is not in datetime format.")

    # Verify the conversion
    print("\n--- DataFrame after Formatting 't' Column ---")
    print(df[['t']].head())
    print("Data type of 't' column:", df['t'].dtype)

    # Drop rows with missing required columns after formatting
    df = df.dropna(subset=required_columns)
    print("\n--- Final DataFrame ---")
    print(df.head())

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
        print("Converting longitude and latitude to geometry...")
        geometry = gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE'])
        print("Geometry calculated.")
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')  # Changed CRS to EPSG:4326 for geospatial consistency
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
    print("Starting to filter GeoDataFrame (gdf)")

    # Convert time_start and time_end to datetime if they aren't already
    try:
        time_start = pd.to_datetime(time_start)
        print(f"Converted time_start to datetime: {time_start}")
    except Exception as e:
        raise ValueError(f"Invalid time_start format: {time_start}") from e

    if time_end:
        try:
            time_end = pd.to_datetime(time_end)
            print(f"Converted time_end to datetime: {time_end}")
        except Exception as e:
            raise ValueError(f"Invalid time_end format: {time_end}") from e

    # Ensure 't' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(gdf['t']):
        try:
            gdf['t'] = pd.to_datetime(gdf['t'])
            print("Converted 't' column to datetime.")
        except Exception as e:
            raise ValueError("Failed to convert 't' column to datetime.") from e
    else:
        print("'t' column is already in datetime format.")

    # Filtering logic
    if exact:
        gdf_filtered = gdf[
            (gdf['ID_Ind'] == animal_id) &
            (gdf['t'] == time_start)
        ]
    elif time_end:
        gdf_filtered = gdf[
            (gdf['ID_Ind'] == animal_id) &
            (gdf['t'] >= time_start) &
            (gdf['t'] <= time_end)
        ]
    else:
        gdf_filtered = gdf[
            (gdf['ID_Ind'] == animal_id) &
            (gdf['t'] == time_start)
        ]

    print("Filtered GeoDataFrame preview:")
    print(gdf_filtered.head())

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

# def plot_kernel_density_geemap(contour_gdf, gdf_filtered, output_filename='/Users/didemdost/Desktop/kernel_density_map_ee.html'):
#     import ee
#     import geemap
#     import geopandas as gpd

#     # Ensure Earth Engine is initialized
#     try:
#         ee.Initialize(project='ee-didemdostt')
#     except Exception as e:
#         print("Initializing Earth Engine...")
#         ee.Authenticate()
#         ee.Initialize(project='ee-didemdostt')

#     # Reproject to WGS84 (EPSG:4326) for compatibility
#     contour_gdf = contour_gdf.to_crs(epsg=4326)
#     gdf_filtered = gdf_filtered.to_crs(epsg=4326)


#     # Get the center of the map
#     center_lat = gdf_filtered.geometry.y.mean()
#     center_lon = gdf_filtered.geometry.x.mean()
#     center = [center_lat, center_lon]
#     print(f"Map center coordinates: Latitude={center_lat}, Longitude={center_lon}")

#     # Define the region of interest using the bounds with buffer
#     bounds = contour_gdf.total_bounds  # minx, miny, maxx, maxy
#     buffer_degree = 0.05  # Adjust as necessary
#     minx = bounds[0] - buffer_degree
#     miny = bounds[1] - buffer_degree
#     maxx = bounds[2] + buffer_degree
#     maxy = bounds[3] + buffer_degree
#     region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

#     print(f"Extended region bounds: [minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}]")

#     # Create a date range based on your data
#     date_start = '2023-01-01'  # Adjust as necessary
#     date_end = '2023-12-31'    # Adjust as necessary

#     # Use the correct Sentinel-2 dataset
#     s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR')
#                      .filterDate(date_start, date_end)
#                      .filterBounds(region)
#                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)))  # Adjusted cloud cover

#     # Check the number of images in the collection
#     collection_size = s2_collection.size().getInfo()
#     print(f"Number of images in the collection after filtering: {collection_size}")
#     if collection_size == 0:
#         print("No images found in the collection after filtering.")
#         return

#     # Take the median image
#     s2_image = s2_collection.median().clip(region)

#     # Check available bands
#     band_names = s2_image.bandNames().getInfo()
#     print(f"Available bands in the image: {band_names}")

#     # Define visualization parameters
#     vis_params = {
#         'bands': ['B4', 'B3', 'B2'],  # True color bands
#         'min': 0,
#         'max': 3000,
#         'gamma': [0.95, 1.1, 1],
#     }

#     # Verify if the specified bands are available
#     if not all(band in band_names for band in vis_params['bands']):
#         print("One or more specified bands are not available in the image.")
#         print("Please check the available bands and update vis_params accordingly.")
#         return

#     # Create a geemap map
#     Map = geemap.Map(center=center, zoom=10)

#     # Add the Earth Engine image layer
#     print("Adding Sentinel-2 Image layer to the map...")
#     Map.addLayer(s2_image, vis_params, 'Sentinel-2 Image')

#     # Add the contour polygons
#     print("Adding Kernel Density Contours to the map...")
#     Map.add_gdf(contour_gdf, layer_name='Kernel Density Contours')

#     # Add the animal movement points
#     print("Adding Animal Positions to the map...")
#     Map.add_gdf(gdf_filtered, layer_name='Animal Positions')

#     # Save the map as an HTML file
#     try:
#         print(f"Saving the map to {output_filename}...")
#         Map.to_html(outfile=output_filename)
#         print(f"Interactive map successfully saved to {output_filename}")
#     except Exception as e:
#         print(f"Failed to save the map: {e}")

def plot_kernel_density_geemap(contour_gdf, gdf_filtered, output_filename='/Users/didemdost/Desktop/kernel_density_map_ee.html'):
    import ee
    import geemap
    import geopandas as gpd

    # Ensure Earth Engine is initialized
    try:
        ee.Initialize(project='ee-didemdostt')
    except Exception as e:
        print("Initializing Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project='ee-didemdostt')

    # Reproject to WGS84 (EPSG:4326) for compatibility
    contour_gdf = contour_gdf.to_crs(epsg=4326)
    gdf_filtered = gdf_filtered.to_crs(epsg=4326)

# Convert datetime columns to strings
    datetime_cols = gdf_filtered.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    for col in datetime_cols:
        gdf_filtered[col] = gdf_filtered[col].astype(str)

    datetime_cols_contour = contour_gdf.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    for col in datetime_cols_contour:
        contour_gdf[col] = contour_gdf[col].astype(str)
    # Get the center of the map
    center_lat = gdf_filtered.geometry.y.mean()
    center_lon = gdf_filtered.geometry.x.mean()
    center = [center_lat, center_lon]
    print(f"Map center coordinates: Latitude={center_lat}, Longitude={center_lon}")

    # Define the region of interest using the bounds with buffer
    bounds = contour_gdf.total_bounds  # minx, miny, maxx, maxy
    buffer_degree = 0.05  # Adjust as necessary
    minx = bounds[0] - buffer_degree
    miny = bounds[1] - buffer_degree
    maxx = bounds[2] + buffer_degree
    maxy = bounds[3] + buffer_degree
    region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

    print(f"Extended region bounds: [minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}]")

    # Create a date range based on your data
    date_start = '2023-01-01'  # Adjust as necessary
    date_end = '2023-12-31'    # Adjust as necessary

    # Use the correct Sentinel-2 dataset
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterDate(date_start, date_end)
                     .filterBounds(region)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)))  # Adjusted cloud cover

    # Check the number of images in the collection
    collection_size = s2_collection.size().getInfo()
    print(f"Number of images in the collection after filtering: {collection_size}")
    if collection_size == 0:
        print("No images found in the collection after filtering.")
        return

    # Take the median image
    s2_image = s2_collection.median().clip(region)

    # Check available bands
    band_names = s2_image.bandNames().getInfo()
    print(f"Available bands in the image: {band_names}")

    # Define visualization parameters
    vis_params = {
        'bands': ['B4', 'B3', 'B2'],  # True color bands
        'min': 0,
        'max': 3000,
        'gamma': [0.95, 1.1, 1],
    }

    # Verify if the specified bands are available
    if not all(band in band_names for band in vis_params['bands']):
        print("One or more specified bands are not available in the image.")
        print("Please check the available bands and update vis_params accordingly.")
        return

    # Create a geemap map
    Map = geemap.Map(center=center, zoom=10)

    # Add the Earth Engine image layer
    print("Adding Sentinel-2 Image layer to the map...")
    Map.addLayer(s2_image, vis_params, 'Sentinel-2 Image')

    # Add the contour polygons
    print("Adding Kernel Density Contours to the map...")
    Map.add_gdf(contour_gdf, layer_name='Kernel Density Contours')

    # Add the animal movement points
    print("Adding Animal Positions to the map...")
    Map.add_gdf(gdf_filtered, layer_name='Animal Positions')

    # Save the map as an HTML file
    try:
        print(f"Saving the map to {output_filename}...")
        Map.save(output_filename)  # Correct usage
        print(f"Interactive map successfully saved to {output_filename}")
    except Exception as e:
        print(f"Failed to save the map: {e}")


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
    gdf = gdf.to_crs(epsg=4326)
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
    ID_Ind = 'NPL28_'  # Adjust as necessary

    time_start = '2023-04-01 12:00:20'  # Adjust as necessary
    time_end = '2023-04-30 22:00:32'    # Adjust as necessary

    try:
        gdf_filtered = filter_gdf(gdf, ID_Ind, time_start, time_end)
        print("Filtered GeoDataFrame:")
        print(gdf_filtered)
        print(f"Number of records after filtering: {len(gdf_filtered)}")
        print(f"CRS of filtered data: {gdf_filtered.crs}")
        print(f"Bounds of filtered data: {gdf_filtered.total_bounds}")
    except ValueError as e:
        print(e)
        return

    # Step 6: Compute Spatial Kernel Density
    cutoff_list = [0.25, 0.5, 0.75, 0.9]  # Adjust as necessary
    contour_gdf = spatial_kernel_density(gdf_filtered, cutoff_list)
    if contour_gdf is None:
        print("Failed to compute spatial kernel density.")
        return

    # Step 7: Plot and Save the Map using the Updated Function
    plot_kernel_density_geemap(contour_gdf, gdf_filtered,  output_filename='/Users/didemdost/Desktop/kernel_density_map_ee.html')

    # # Step 6: Calculate Spatial Utilization Metrics (Method 1)
    # spatial_metrics = calculate_spatial_utilization(gdf_filtered, bw_method=0.2)
    # if spatial_metrics is None:
    #     print("Could not calculate spatial utilization metrics.")
    # else:
    #     # Step 7: Plot Spatial Utilization (Method 1)
    #     plot_spatial_utilization_method1(gdf_filtered, spatial_metrics, ID_Ind, time_start, time_end, output_dir)

    # # Step 8: Spatial Kernel Density Estimation (Method 2)
    # cutoff_levels = [0.2, 0.4, 0.6, 0.8]  # Adjust density levels as needed
    # print("\nStarting Spatial Kernel Density Estimation (Method 2)...")
    # contour_gdf = spatial_kernel_density(gdf_filtered, cutoff_levels)
    
    # if contour_gdf is not None and not contour_gdf.empty:
    #     # Add ID and name columns for ArcGIS compatibility
    #     contour_gdf['ID'] = ID_Ind
    #     #contour_gdf['name'] = ID_Ind  # Duplicate if required

    #     # Reproject to WGS84 for compatibility with most GIS software
    #     contour_gdf = contour_gdf.to_crs(epsg=4326)

    #     # Export to GeoPackage (instead of Shapefile for consistency)
    #     gpkg_contour_output = os.path.join(output_dir, 'all_kdes.gpkg')
    #     try:
    #         contour_gdf.to_file(gpkg_contour_output, driver="GPKG")
    #         print(f"Spatial Kernel Density contours successfully saved to {gpkg_contour_output}")
    #     except Exception as e:
    #         print(f"Failed to write Spatial Kernel Density contours to GeoPackage: {e}")
        
    #     # Step 9: Plot Spatial Kernel Density Estimation (Method 2) using cartoee
    #     # Reproject contour_gdf back to EPSG:3857 for plotting with cartoee
    #     contour_gdf_plot = contour_gdf.to_crs(epsg=3857)
    #     plot_spatial_kernel_density_method2(contour_gdf_plot, ID_Ind, time_start, time_end, output_dir)
    # else:
    #     print("No contour polygons were generated in Spatial Kernel Density Estimation.")

if __name__ == "__main__":
    main()
