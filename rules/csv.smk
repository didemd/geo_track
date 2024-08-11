from snakemake.utils import validate

#validate(config, schema="config.yaml")

rule csv_to_gdf:
    input:
        directory = config["input_directory"]
    output:
        gdf_gpkg = "output/output.gpkg",
        csv_file = "output/output.csv",
    run:
        combined_df = get_csv_heads(input.directory)
        gdf = convert_csv_to_geodataframe(combined_df)
        csv_file =write_gdf_to_csv(gdf, output.csv_file)
        export_gdf_to_gpkg(gdf, output_file=output.gdf_gpkg)
        compute_and_visualize_spatial_utilization(gdf, animal_id= 'TAG221', time_start='01/11/2022', time_end='03/11/2022')



