rule plot_with_cartoee:
    input:
        gpkg="output/spatial_metrics/{ID}_{time_start}_{time_end}.gpkg"
    params:
        ID="{ID}"
    output:
        "plots/{ID}_map.png"
    shell:
        """
        python scripts/plot_with_cartoee.py --gpkg_input {input.gpkg} --plot_output {output} --ID "{params.ID}"
        """