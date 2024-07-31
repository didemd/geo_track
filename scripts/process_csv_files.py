import os
import sys
import glob
import re
import logging
import hvplot.pandas
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import hvplot
import holoviews as hv

hv.extension('bokeh')  # Initialize Holoviews with Bokeh as the backend

# Configure logging to show info level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper functions for data standardization and conversion
def standardize_headers(df):
    # Dictionary to map various column names to standardized names
    header_mapping = {
        r'^Individual-local.*': 'ID_Ind',
        r'^Individual-name.*': 'ID_Ind',
        r'^Individual_Name.*': 'ID_Ind',
        'Location-long': 'LONGITUDE',
        'Location-lat': 'LATITUDE',
        r'^Longitude.*': 'LONGITUDE',
        r'^Latitude.*': 'LATITUDE',
        'GPS Longitude': 'LONGITUDE',
        'GPS Latitude': 'LATITUDE',
    }
    # Rename columns based on the header mapping
    df.columns = [next((new_name for pattern, new_name in header_mapping.items() if re.match(pattern, col)), col) for col in df.columns]
    return df

# Function to convert date columns to datetime and add geometry for geospatial data
def convert_dates_and_add_geometry(df):
    # Dictionary to map date column names to their respective formats
    datetime_formats = {
        'Acquisition Start Time': '%d/%m/%Y %H:%M',
        'Timestamp (GMT+2)': '%d/%m/%Y %H:%M',
        'Time Stamp UTC': '%d/%m/%Y %H:%M',
        'Time Stamp UTC1': '%m/%d/%Y %H:%M',
        'Date': '%d/%m/%Y',
        'DATE (GMT+2)': '%Y-%m-%d',
        'Date1': '%m/%d/%y',
    }
    # Convert each date column to datetime based on the format specified
    for col, fmt in datetime_formats.items():
        if col in df.columns:
            df['t'] = pd.to_datetime(df[col], format=fmt, errors='coerce')
    # Add a geometry column if latitude and longitude are present
    if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
        df['geometry'] = df.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)
    return df

# Function to read GeoPackage files, optionally reproject, plot, and save the plots
def read_and_plot_gpkg(file_pattern, output_dir, target_crs=None, default_crs=None):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it does not exist
    file_paths = glob.glob(file_pattern)  # Find all files matching the pattern
    gdfs = []  # List to store GeoDataFrames
    all_inds = []  # List to collect all individual IDs

    for file_path in file_paths:
        gdf = gpd.read_file(file_path)  # Read the GeoPackage file
        print(f"File: {file_path}")
        print(gdf.tail())  # Print the last few rows of the GeoDataFrame

        if gdf.crs is None:  # Check if GeoDataFrame has a CRS
            if default_crs is not None:
                gdf.set_crs(default_crs, inplace=True)  # Set default CRS if provided
                print(f"Set default CRS: {default_crs}")
            else:
                raise ValueError(f"GeoDataFrame from {file_path} has no CRS and no default CRS provided.")
        print("Original CRS:", gdf.crs)

        if target_crs is not None:  # Reproject if target CRS is provided
            gdf = gdf.to_crs(target_crs)
            print("Reprojected CRS:", gdf.crs)

        gdfs.append(gdf)  # Add the GeoDataFrame to the list

        # Generate the plot with latitude and longitude
        plot = gdf.hvplot.points('LONGITUDE', 'LATITUDE', geo=True, tiles='EsriImagery', frame_width=600, frame_height=400, title=os.path.basename(file_path))

        # Save the plot as an HTML file
        plot_file = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '_lat_lon.html')
        hvplot.save(plot, plot_file)
        print(f"Plot saved to {plot_file}")

        all_inds.extend(gdf['ID_Ind'])  # Collect individual IDs

    if all_inds:  # Plot histogram if there are individual IDs
        plt.figure(figsize=(10, 6))
        plt.hist(all_inds, bins=len(set(all_inds)), align='mid', rwidth=0.8)
        plt.xlabel('ID_Ind', loc='center')
        plt.ylabel('Count')
        plt.title('Histogram of detection counts across all individuals')
        plt.grid(axis='y', alpha=0.75)
        histogram_file = os.path.join(output_dir, 'histogram.png')
        plt.savefig(histogram_file)  # Save histogram as PNG file
        print(f"Histogram saved to {histogram_file}")
        plt.show()

    return gdfs

# Function to process CSV files: standardize headers, convert dates, add geometry, and save as GeoPackage
def process_csv_files(file_path, output_directory):
    try:
        df = pd.read_csv(file_path)  # Read the CSV file
        df = standardize_headers(df)  # Standardize column headers
        df = convert_dates_and_add_geometry(df)  # Convert dates and add geometry
        if df.empty:
            logging.error("Data processing failed or data is empty.")
            return
        id_name = os.path.splitext(os.path.basename(file_path))[0]  # Get the base name of the file
        csv_output = os.path.join(output_directory, f"{id_name}_processed.csv")  # Define the output CSV path
        gpkg_output = os.path.join(output_directory, f"{id_name}.gpkg")  # Define the output GeoPackage path
        df.to_csv(csv_output, index=False)  # Save the processed CSV
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')  # Convert to GeoDataFrame with CRS
        gdf.to_file(gpkg_output, driver='GPKG')  # Save as GeoPackage
        logging.info(f"Processed and saved: {csv_output} and {gpkg_output}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")

# Function to read, optionally reproject, combine GeoDataFrames, and save them
def read_and_combine_gpkg(file_pattern, target_crs=None, default_crs=None, output_path='./'):
    file_paths = glob.glob(file_pattern)  # Find all files matching the pattern
    
    # Initialize an empty dictionary to store combined GeoDataFrames grouped by prefix
    combined_gdfs = {}

    for file_path in file_paths:
        gdf = gpd.read_file(file_path)  # Read the GeoPackage file
        print(f"File: {file_path}")
        print(gdf.tail())  # Print the last few rows of the GeoDataFrame

        if gdf.crs is None:  # Check if GeoDataFrame has a CRS
            if default_crs is not None:
                gdf.set_crs(default_crs, inplace=True)  # Set default CRS if provided
                print(f"Set default CRS: {default_crs}")
            else:
                raise ValueError(f"GeoDataFrame from {file_path} has no CRS and no default CRS provided.")
        print("Original CRS:", gdf.crs)

        if target_crs is not None:  # Reproject if target CRS is provided
            gdf = gdf.to_crs(target_crs)
            print("Reprojected CRS:", gdf.crs)

        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(file_path))[0]

        # Extract prefix (before the first underscore)
        first_underscore_index = filename.find('_')
        if first_underscore_index != -1:
            prefix = filename[:first_underscore_index]
        else:
            prefix = filename

        # Combine GeoDataFrame with the same prefix
        if prefix in combined_gdfs:
            combined_gdfs[prefix] = pd.concat([combined_gdfs[prefix], gdf], ignore_index=True)
        else:
            combined_gdfs[prefix] = gdf.copy()

    # Save each combined GeoDataFrame to GeoPackage file
    for prefix, combined_gdf in combined_gdfs.items():
        output_file = os.path.join(output_path, f"{prefix}_cleaned.gpkg")
        combined_gdf.to_file(output_file, driver="GPKG")
        print(f"Saved combined GeoDataFrame for {prefix} to {output_file}")

        # Plot and display each combined GeoDataFrame
        plot = combined_gdf.hvplot(title=f"Combined {prefix}", tiles='EsriImagery', frame_width=600, frame_height=400, geo=True)
        hvplot.save(plot, os.path.join(output_path, f"{prefix}_combined.html"))
        print(f"Plot saved to {os.path.join(output_path, f'{prefix}_combined.html')}")

        # Count ID_Ind values and plot histogram for each combined GeoDataFrame
        if 'ID_Ind' in combined_gdf.columns:
            plt.figure(figsize=(6, 4))
            combined_gdf['ID_Ind'].value_counts().plot(kind='bar')
            plt.xlabel('ID_Ind')
            plt.ylabel('Count')
            plt.title(f'ID_Ind Counts for Combined {prefix}')
            plt_file = os.path.join(output_path, f"{prefix}_histogram.png")
            plt.savefig(plt_file)
            print(f"Histogram saved to {plt_file}")
            plt.show()
        else:
            print(f"No 'ID_Ind' column found in Combined {prefix}.")

    return combined_gdfs

# Main function to handle command-line arguments and run appropriate mode
def main():
    if len(sys.argv) < 4:
        logging.error("Usage: python script_name.py <mode> <input_file_or_pattern> <output_directory>")
        sys.exit(1)
    
    mode = sys.argv[1]  # First argument: mode ('process', 'plot', or 'combine')
    file_pattern_or_path = sys.argv[2]  # Second argument: input file path or pattern
    output_directory = sys.argv[3]  # Third argument: output directory
    os.makedirs(output_directory, exist_ok=True)  # Create output directory if it doesn't exist
    
    if mode == "process":
        process_csv_files(file_pattern_or_path, output_directory)  # Process CSV file
    elif mode == "plot":
        read_and_plot_gpkg(file_pattern_or_path, output_directory)  # Generate plots from GeoPackage
    elif mode == "combine":
        read_and_combine_gpkg(file_pattern_or_path, output_path=output_directory)  # Combine GeoPackages
    else:
        logging.error("Invalid mode. Use 'process', 'plot', or 'combine'.")
        sys.exit(1)

if __name__ == "__main__":
    main()  # Run the main function if the script is executed directly
