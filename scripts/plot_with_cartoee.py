import argparse
import cartoee
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs

def main():
    parser = argparse.ArgumentParser(description='Generate map using cartoee.')
    parser.add_argument('--gpkg_input', required=True, help='Input GPKG file path.')
    parser.add_argument('--plot_output', required=True, help='Output plot file path.')
    parser.add_argument('--ID', required=True, help='Animal ID.')
    args = parser.parse_args()

    gpkg_input = args.gpkg_input
    plot_output = args.plot_output
    ID = args.ID

    # Read the GPKG
    contour_gdf = gpd.read_file(gpkg_input)

    # Create a map using cartoee
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the contours
    contour_gdf.plot(ax=ax, column='level', cmap='viridis', legend=True)

    # Add north arrow, scale, and legend
    cartoee.add_scale_bar(ax, location=(0.1, 0.05))
    cartoee.add_north_arrow(ax, location=(0.1, 0.1))

    # Add title
    ax.set_title(f'Kernel Density Contours for {ID}')

    # Save the plot
    plt.savefig(plot_output, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
