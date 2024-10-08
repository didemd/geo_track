rule combine_csv_files:
    input:
        directory = config["input_directory"]
    output:
        combined_csv = "intermediate/combined.csv"
    run:
        combined_df = combine_csv_files(input.directory)
        os.makedirs(os.path.dirname(output.combined_csv), exist_ok=True)
        combined_df.to_csv(output.combined_csv, index=False)

rule convert_to_gdf:
    input:
        combined_csv = "intermediate/combined.csv"
    output:
        gdf = "intermediate/data.gpkg"
    run:  # Removed the '/' here
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
        os.makedirs(os.path.dirname(output.formatted_gdf), exist_ok=True)
        gdf.to_file(output.formatted_gdf, driver='GPKG')

rule process_gdf:
    input:
        gdf = "intermediate/formatted.gdf"
    output:
        spatial_metrics = "output/spatial_metrics.pkl"  # Changed extension to .pkl for clarity
    run:
        import pickle  # Ensure pickle is imported within the rule
        gdf = gpd.read_file(input.gdf)
        gdf_filtered = filter_gdf(gdf, 'TAG221', '2022-11-01 00:00:32', '2023-10-31 22:00:32')
        spatial_metrics = calculate_spatial_utilization(gdf_filtered, bw_method=0.2)
        os.makedirs(os.path.dirname(output.spatial_metrics), exist_ok=True)
        with open(output.spatial_metrics, 'wb') as f:  # Use 'wb' mode for binary writing
            pickle.dump(spatial_metrics, f)

rule plot_spatial_metrics:
    input:
        gdf = "intermediate/formatted.gdf",
        spatial_metrics = "output/spatial_metrics.pkl"  
    output:
        plot = "output/plot_spatial_utilization_heatmap_with_map.png"
    run:
        import pickle  # Import pickle to deserialize spatial_metrics
        gdf = gpd.read_file(input.gdf)
        with open(input.spatial_metrics, 'rb') as f:  # Use 'rb' mode for binary reading
            spatial_metrics = pickle.load(f)
        plot_spatial_utilization(
            gdf,
            spatial_metrics,
            'TAG221',
            '2022-11-01 00:00:32',
            '2023-10-31 22:00:32',
            output.plot  
        )
