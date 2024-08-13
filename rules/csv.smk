rule csv_to_gdf:
    input:
        directory = config["input_directory"]  # input directory 
    output:
        gdf_gpkg = "output/output.gpkg",       # geopackage file.
        csv_file = "output/output.csv",        # csv file.
    run:
        combined_df = combine_csv_files(input.directory)  # combine csv files
        gdf = convert_csv_to_geodataframe(combined_df)    # Convert the combined DataFrame to a GeoDataFrame
        write_gdf_to_csv(gdf, output.csv_file)            # write the geodataframe to a csv file
        export_gdf_to_gpkg(gdf, output_file=output.gdf_gpkg)  # export the geodataframe to a geopackage
        gdf = convert_time_date_formats(gdf)  # convert datetime formats 
        gdf_filtered = filter_gdf(gdf, 'TAG221', '2022-11-01 00:00:32', '2023-10-31 22:00:32')  # filter gdf 
        spatial_metrics = calculate_spatial_utilization(gdf_filtered, bw_method=0.2)  # calculate kde and min convex hull
        plot_spatial_utilization(gdf_filtered, spatial_metrics, 'TAG221', '2022-11-01 00:00:32', '2023-10-31 22:00:32')  # plot 
