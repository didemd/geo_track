rule combine_csv_files:
    input:
        directory = config["input_directory"]
    output:
        combined_csv = "intermediate/combined.csv"
    run:
        combined_df = combine_csv_files(input.directory)
        combined_df.to_csv(output.combined_csv, index=False)

rule convert_to_gdf:
    input:
        combined_csv = "intermediate/combined.csv"
    output:
        gdf = "intermediate/data.gpkg"
    run:
        df = pd.read_csv(input.combined_csv)
        gdf = convert_csv_to_geodataframe(df)
        gdf.to_file(output.gdf, driver='GPKG')


rule export_gdf:
    input:
        gdf = "intermediate/data.gpkg"
    output:
        csv_file = "output/output.csv",
        gdf_gpkg = "output/output.gpkg"
    run:
        gdf = gpd.read_file(input.gdf)
        write_gdf_to_csv(gdf, output.csv_file)
        export_gdf_to_gpkg(gdf, output_file=output.gdf_gpkg)

rule format_datetime:
    input:
        gdf = "output/output.gpkg"
    output:
        formatted_gdf = "intermediate/formatted.gdf"
    run:
        gdf = gpd.read_file(input.gdf)
        gdf = convert_time_date_formats(gdf)
        gdf.to_file(output.formatted_gdf, driver='GPKG')


rule process_gdf:
    input:
        gdf = "intermediate/formatted.gdf"
    output:
        spatial_metrics = "output/spatial_metrics.txt"
    run:
        gdf = gpd.read_file(input.gdf)
        gdf_filtered = filter_gdf(gdf, 'TAG221', '2022-11-01 00:00:32', '2023-10-31 22:00:32')
        spatial_metrics = calculate_spatial_utilization(gdf_filtered, bw_method=0.2)
        with open(output.spatial_metrics, 'w') as f:
            f.write(str(spatial_metrics))


rule plot_spatial_metrics:
    input:
        gdf = "intermediate/formatted.gdf",
        spatial_metrics = "output/spatial_metrics.txt"
    output:
        plot = "output/plot_spatial_utilization_heatmap_with_map.png"
    run:
        gdf = gpd.read_file(input.gdf)
        plot_spatial_utilization(gdf, input.spatial_metrics, 'TAG221', '2022-11-01 00:00:32', '2023-10-31 22:00:32')
        plt.savefig(output.plot)
