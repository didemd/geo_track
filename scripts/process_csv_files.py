import glob
import os
import sys
import re
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import hvplot.pandas
import holoviews as hv
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)

def standardize_headers(df):
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
    df.columns = [next((new_name for pattern, new_name in header_mapping.items() if re.match(pattern, col)), col) for col in df.columns]
    return df

def convert_dates_and_add_geometry(df):
    datetime_formats = {
        'Acquisition Start Time': '%d/%m/%Y %H:%M',
        'Timestamp (GMT+2)': '%d/%m/%Y %H:%M',
        'Time Stamp UTC': '%d/%m/%Y %H:%M',
        'Time Stamp UTC1': '%m/%d/%Y %H:%M',
        'Date': '%d/%m/%Y',
        'DATE (GMT+2)': '%Y-%m-%d',
        'Date1': '%m/%d/%y',
    }
    for col, fmt in datetime_formats.items():
        if col in df.columns:
            df['t'] = pd.to_datetime(df[col], format=fmt, errors='coerce')
    if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
        df['geometry'] = df.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)
    return df

def save_geopackage(gdf, filename):
    if os.path.exists(filename):
        logging.info(f"Warning: '{filename}' already exists. Overwriting.")
        os.remove(filename)
    gdf.to_file(filename, driver='GPKG')
    logging.info(f"Saved GeoPackage to {filename}")

def process_csv(file_path, output_directory):
    try:
        logging.info(f"Reading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        df = standardize_headers(df)
        df = convert_dates_and_add_geometry(df)
        logging.info("Standardized and converted dates and geometry")
        
        if df.empty:
            logging.error("Data processing failed or data is empty.")
            return

        id_name = os.path.splitext(os.path.basename(file_path))[0]
        csv_output = os.path.join(output_directory, f"{id_name}_processed.csv")
        gpkg_output = os.path.join(output_directory, f"{id_name}.gpkg")

        df.to_csv(csv_output, index=False)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        save_geopackage(gdf, gpkg_output)

        logging.info(f"Processed and saved: {csv_output} and {gpkg_output}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")

def combine_gpkgs(input_files, output_file):
    try:
        logging.info(f"Combining GPKG files: {input_files}")
        gdfs = [gpd.read_file(file) for file in input_files]
        combined_gdf = pd.concat(gdfs, ignore_index=True)
        save_geopackage(combined_gdf, output_file)
        logging.info(f"Saved combined GPKG: {output_file}")
    except Exception as e:
        logging.error(f"Error combining files {input_files}: {e}")

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "process":
        csv_file = sys.argv[2]
        output_directory = sys.argv[3]
        process_csv(csv_file, output_directory)
    elif mode == "combine":
        input_files = sys.argv[2:-1]
        output_file = sys.argv[-1]
        combine_gpkgs(input_files, output_file)
    else:
        logging.error("Invalid mode. Use 'process' or 'combine'.")
        sys.exit(1)
