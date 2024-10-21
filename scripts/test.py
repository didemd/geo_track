
import pandas as pd
import numpy as np
import os
import re
import glob
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt  # *** Added for diagnostic plots ***
import seaborn as sns  # *** Added for advanced diagnostic plots ***
from scipy.stats import gaussian_kde
import hvplot.pandas
import holoviews as hv
import contextily as ctx
from IPython.display import display
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity
from shapely.geometry import MultiPolygon, Polygon


# Ensure that plots are displayed within the notebook
# Uncomment the following line if running in a Jupyter notebook
# %matplotlib inline

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
        "Location-long": "LONGITUDE", 
        "Location-lat": "LATITUDE", 
        "^Longitude.*": "LONGITUDE", 
        "^Latitude.*": "LATITUDE",
        "GPS Longitude": "LONGITUDE", 
        "GPS Latitude": "LATITUDE",
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
    print(df)

    # Ensure the dataframe has only the standardized headers that match the mapping
    standard_headers = list(header_mapping.values())
    # Retain other necessary columns (e.g., 'TAG') if they exist
    required_columns = ['ID_Ind', 'LONGITUDE', 'LATITUDE', 'TAG', 'Date', 'Time']
    # Include columns that are present in df to avoid KeyError
    standard_headers = [col for col in required_columns if col in df.columns] + [
        col for col in df.columns if col not in standard_headers and col not in required_columns
    ]
    df = df[[col for col in df.columns if col in standard_headers]]
    print("modified_df", df)

    # *** Added Diagnostic Plot: Missing Data Heatmap ***
    # This plot helps identify any missing values in the standardized DataFrame
    #plt.figure(figsize=(10, 6))
    #sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    #plt.title(f'Missing Data Heatmap for {os.path.basename(file_path)}')
    #plt.show()  # Display the plot
    # *** End of Added Diagnostic Plot ***

    # *** Added Preprocessing Step: Handling Missing Values ***
    # Depending on the analysis requirements, you can choose to drop or impute missing values
    # Here, we'll drop rows with any missing values for simplicity
    df = df.dropna()
    # Alternatively, to fill missing values, uncomment the following line:
    # df = df.fillna(method='ffill')  # Forward fill as an example
    # *** End of Added Preprocessing Step ***

    return df  # Return the DataFrame with 't' column set as index

def check_csv_file(file_path):
    """
    #Checks a single CSV file for headers, sample content, and observation count.

    #Args:
    #3    file_path (str): Path to the CSV file.

    #Returns:
    #    tuple or None: (headers, sample_content, length) if successful, else None.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    # Check the headers
    headers_tmp = df.columns.tolist()
    
    # Sample some content (first 5 rows)
    sample_content = df.head()

    # Detection count
    length = df.shape[0]
    
    return headers_tmp, sample_content, length

def check_all_csv_files(directory):
    """
    #Checks all standardized CSV files in a directory for headers, sample content, and observation count.

    #Args:
     #   directory (str): Path to the directory containing standardized CSV files.
    """
    # List all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, '*standardized.csv'))
    
    for file_path in csv_files:
        print(f"Checking file: {file_path}")
        
        result = check_csv_file(file_path)
        
        if result is not None:
            headers_tmp, sample_content, length = result
            print(f"Headers: {headers_tmp}")
            print(f"Sample content Observations {length}:\n{sample_content}")
        else:
            print("Could not read file or file is empty.")
        
        print("\n" + "="*50 + "\n")

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
    
    check_all_csv_files(directory)

    all_data_frames = []
    for file in csv_files:
        print(f"\nProcessing file: {file}")
        df = standardize_headers(file)
        if df is not None and not df.empty:
            print("Standardized DataFrame Headers:")
            print(df.columns.tolist())
            
            # *** Added Diagnostic Plot: Time Series Plot ***
            # Visualize the number of observations over time to detect any temporal patterns or anomalies
            plt.figure(figsize=(12, 6))
            df.resample('D').size().plot()
            plt.title(f'Time Series of Observations for {os.path.basename(file)}')
            plt.xlabel('Date')
            plt.ylabel('Number of Observations')
            #plt.show()
            # *** End of Added Diagnostic Plot ***

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

    print("Combined DataFrame Columns:", combined_df.columns.tolist())

    
    # *** Added Diagnostic Plot: Combined DataFrame Overview ***
    # Provides an overview of the combined dataset, including summary statistics and distributions
    print("Combined DataFrame Summary:")
    print(combined_df.describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(combined_df['LONGITUDE'], bins=50, kde=True)
    plt.title('Distribution of Longitude')
    plt.xlabel('Longitude')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(combined_df['LATITUDE'], bins=50, kde=True)
    plt.title('Distribution of Latitude')
    plt.xlabel('Latitude')
    plt.ylabel('Frequency')
    plt.show()
    # *** End of Added Diagnostic Plot ***
    
    return combined_df

def convert_csv_to_geodataframe(read_csv):
    """
    Converts a DataFrame with 'LONGITUDE' and 'LATITUDE' columns to a GeoDataFrame.

    Args:
        read_csv (pd.DataFrame): DataFrame containing at least 'LONGITUDE' and 'LATITUDE' columns.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with points created from the 'LONGITUDE' and 'LATITUDE' columns.
    """
    print("Starting conversion to GeoDataFrame...")
    
    missing_columns = []
    for col in ['LONGITUDE', 'LATITUDE']:
        if col not in read_csv.columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise ValueError(f"The DataFrame is missing the following required columns: {missing_columns}")
    
    print("All required columns are present. Proceeding with conversion...")
    
    try:
        mygeometry_array = gpd.points_from_xy(read_csv['LONGITUDE'], read_csv['LATITUDE'])
        gdf = gpd.GeoDataFrame(read_csv, geometry=mygeometry_array, crs='EPSG:4326')
        print("GeoDataFrame successfully created:")
        print(gdf.head())  # Print only the first few rows for brevity
        return gdf
    except Exception as e:
        print(f"Error during GeoDataFrame creation: {e}")
        return None

def write_gdf_to_csv(gdf, csv_output_file='output/output.csv'):
    """
    Writes a GeoDataFrame to a CSV file, excluding the geometry column.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to be saved.
        csv_output_file (str): Path where the CSV file will be saved.
    """
    print(f"Attempting to write GeoDataFrame to CSV at: {csv_output_file}")
    
    try:
        # Ensure the 'geometry' column exists before dropping
        if 'geometry' in gdf.columns:
            gdf_to_save = gdf.drop('geometry', axis=1)
        else:
            print("'geometry' column not found. Saving DataFrame without dropping geometry.")
            gdf_to_save = gdf.copy()
        
        # Create the output directory if it doesn't exist
        output_directory = os.path.dirname(csv_output_file)
        if output_directory and not os.path.exists(output_directory):
            print(f"Output directory {output_directory} does not exist. Creating it...")
            os.makedirs(output_directory, exist_ok=True)
            print(f"Output directory {output_directory} created successfully.")
        
        # Save to CSV
        gdf_to_save.to_csv(csv_output_file, index=False)
        print(f"GeoDataFrame successfully written to {csv_output_file}")
    except Exception as e:
        print(f"Failed to write GeoDataFrame to CSV: {e}")

def gdf_to_gpkg(gdf, output_file='output/output.gpkg'):
    """
    Exports a GeoDataFrame to a GeoPackage file.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to be exported.
        output_file (str): Path where the GeoPackage file will be saved.
    """
    print(f"Attempting to write GeoDataFrame to GPKG at: {output_file}")
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Save to GPKG
        gdf.to_file(filename=output_file, driver="GPKG")
        print(f"GeoDataFrame successfully written to {output_file}")
    except Exception as e:
        print(f"Failed to write GeoDataFrame to GPKG: {e}")

def filter_gdf(gdf, animal_id, time_start): #,time_end):
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
    #time_end = pd.to_datetime(time_end)
    
    gdf_filtered = gdf[(gdf['TAG'] == animal_id) & 
                       (gdf['t'] >= time_start)] #&
                       #(gdf['ct'] <= time_end)]
    if gdf_filtered.empty:
        raise ValueError("No data available for the specified animal and timeframe.")
    return gdf_filtered

def calculate_spatial_utilization(gdf_filtered, bw_method):
    """
    Calculate spatial utilization metrics such as convex hull and KDE for a filtered GeoDataFrame.
    
    Args:
    gdf_filtered (GeoDataFrame): Filtered GeoDataFrame containing spatial data.
    
    Returns:
    dict: Dictionary containing the convex hull, KDE values, and other spatial metrics.
    """
    points = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
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

def plot_spatial_utilization(gdf, spatial_metrics, animal_id, time_start):
    """
    Plot spatial utilization with a heatmap and convex hull.
    
    Args:
    gdf_filtered (GeoDataFrame): Filtered GeoDataFrame containing spatial data.
    spatial_metrics (dict): Dictionary containing spatial metrics including the convex hull and KDE values.
    animal_id (str): Animal ID.
    time_start (datetime): Start time of the dataset.
    time_end (datetime): End time of the dataset.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.contourf(spatial_metrics['xx'], spatial_metrics['yy'], spatial_metrics['kde_values'], levels=[spatial_metrics['level_95'], spatial_metrics['level_50'], spatial_metrics['kde_values'].max()], colors=['orange', 'red', 'darkred'], alpha=0.5)
    gdf.plot(ax=ax, color='blue', markersize=10, label='Filtered Points')
    gpd.GeoSeries([spatial_metrics['convex_hull_polygon']]).plot(ax=ax, color='none', edgecolor='green', linewidth=2, label='Convex Hull')

    buffer = 0.1
    xmin, xmax = spatial_metrics['xx'].min(), spatial_metrics['xx'].max()
    ymin, ymax = spatial_metrics['yy'].min(), spatial_metrics['yy'].max()
    ax.set_xlim([xmin - buffer * (xmax - xmin), xmax + buffer * (xmax - xmin)])
    ax.set_ylim([ymin - buffer * (ymax - ymin), ymax + buffer * (ymax - ymin)])
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), zoom=12)

    ax.set_title(f"Spatial Utilization for {animal_id} from {time_start} to {time_end}")
    ax.set_axis_off()
    ax.legend(handles=[
        plt.Line2D([0], [0], color='orange', lw=4, label=f'50% Density: {spatial_metrics['area_50']:.2f} sq. units'),
        plt.Line2D([0], [0], color='red', lw=4, label=f'95% Density: {spatial_metrics['area_95']:.2f} sq. units'),
        plt.Line2D([0], [0], color='green', lw=4, label=f'Circumference: {spatial_metrics['circumference']:.2f} units')
    ])

    plt.tight_layout()
    plt.savefig('output/plot_spatial_utilization_heatmap_with_map.png', dpi=300)
    plt.close(fig)



combined_df = combine_csv_files("/Users/didemdost/Desktop/geo_track/data")
gdf = convert_csv_to_geodataframe(combined_df)

if gdf is not None:
    display(gdf)  # This will render the GeoDataFrame in Jupyter notebooks
else:
    print("GeoDataFrame was not created due to errors.")

output_dir = "/Users/didemdost/Desktop/geo_track/output"
csv_output_file = os.path.join(output_dir, 'output.csv')
write_gdf_to_csv(gdf, csv_output_file=csv_output_file)
gpkg_output_file = os.path.join(output_dir, 'output.gpkg')
gdf_to_gpkg(gdf, gpkg_output_file)
gdf_filtered = filter_gdf(gdf, 'TAG221', '2022-11-01 00:00:32') 
print(gdf_filtered)
"""
spatial_metrics = calculate_spatial_utilization(gdf_filtered, bw_method=0.2, cutoff_levels=[0.2, 0.4, 0.6, 0.8])


#method 1
gdf_filtered = filter_gdf(gdf, 'TAG221', '2022-11-01 00:00:32') #, '2023-10-31 22:00:32')
print(gdf_filtered)
#calculate_spatial_utilization(gdf_filtered, 0.2)
#plot_spatial_utilization(gdf_filtered, spatial_metrics, 'TAG221', '2022-11-01 00:00:32', '2023-10-31 22:00:32')
"""

#currently not working