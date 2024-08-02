import sys
import geopandas as gpd

def get_head_of_gpkg(input_gpkg, output_txt):
    # Read the GeoPackage file
    gdf = gpd.read_file(input_gpkg)
    
    # Get the head of the DataFrame
    head_df = gdf.head()
    
    # Write the head to a text file
    with open(output_txt, 'w') as f:
        f.write(head_df.to_string())

if __name__ == "__main__":
    input_gpkg = sys.argv[1]
    output_txt = sys.argv[2]
    
    get_head_of_gpkg(input_gpkg, output_txt)
