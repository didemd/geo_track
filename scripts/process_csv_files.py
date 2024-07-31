import os
import sys
import glob
import re
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def process_csv_files(file_path, output_directory):
    try:
        df = pd.read_csv(file_path)
        df = standardize_headers(df)
        df = convert_dates_and_add_geometry(df)
        
        if df.empty:
            logging.error("Data processing failed or data is empty.")
            return

        id_name = os.path.splitext(os.path.basename(file_path))[0]
        csv_output = os.path.join(output_directory, f"{id_name}_processed.csv")
        gpkg_output = os.path.join(output_directory, f"{id_name}.gpkg")

        df.to_csv(csv_output, index=False)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        gdf.to_file(gpkg_output, driver='GPKG')

        logging.info(f"Processed and saved: {csv_output} and {gpkg_output}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        logging.error("Usage: python process_csv_files.py process <input_file> <output_directory>")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "process":
        file_path = sys.argv[2]
        output_directory = sys.argv[3]
        os.makedirs(output_directory, exist_ok=True)
        process_csv_files(file_path, output_directory)
    else:
        logging.error("Invalid mode. Use 'process'.")
        sys.exit(1)
