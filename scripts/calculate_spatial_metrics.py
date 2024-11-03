import argparse
import os
import pandas as pd
from helpers import convert_csv_to_geodataframe, filter_gdf, spatial_kernel_density, plot_kernel_density_geemap

def main():
    parser = argparse.ArgumentParser(description='Calculate spatial metrics and generate plots.')
    parser.add_argument('input_csvs', nargs='+', help='Input cleaned CSV files.')
    parser.add_argument('--gpkg_output', required=True, help='Output GPKG file path.')
    parser.add_argument('--plot_output', required=True, help='Output plot file path.')
    parser.add_argument('--ID', required=True, help='Animal ID.')
    parser.add_argument('--time_start', required=True, help='Start time for filtering.')
    parser.add_argument('--time_end', required=True, help='End time for filtering.')
    args = parser.parse_args()
    print("Arguments:", args)



    dfs = [pd.read_csv(f) for f in args.input_csvs]
    df = pd.concat(dfs, ignore_index=True)
    print(df.head())  # Check the first few rows

    df['t'] = pd.to_datetime(df['t'], errors='coerce')
    print(df['t'].head())  # Ensure dates are parsed correctly
    time_start = pd.to_datetime(args.time_start)
    time_end = pd.to_datetime(args.time_end)

    df_filtered = df[(df['t'] >= time_start) & (df['t'] <= time_end)]
    print(df_filtered.head())  # Ensure the data is correctly filtered

    gdf = convert_csv_to_geodataframe(df_filtered)
    contour_gdf = spatial_kernel_density(gdf, [0.5, 0.9])
    print(contour_gdf)  # Ensure contours are generated correctly

    gpkg_dir = os.path.dirname(args.gpkg_output)
    plot_dir = os.path.dirname(args.plot_output)

    try:
        os.makedirs(gpkg_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")


    # Save outputs
    if contour_gdf is not None:
        contour_gdf.to_file(args.gpkg_output, driver="GPKG")
        plot_kernel_density_geemap(contour_gdf, gdf, output_filename=args.plot_output)
    else:
        print(f"No spatial kernel density contours generated for ID {args.ID}")

if __name__ == "__main__":
    main()
