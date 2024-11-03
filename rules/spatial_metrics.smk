rule calculate_spatial_metrics:
    input:
        csvs=expand("output/cleaned/{sample}.csv", sample=samples)
    params:
        ID=lambda wildcards: wildcards.ID.split('__')[0] + '_',
        time_start=lambda wildcards: next(
            (a['time_start'] for a in config['animals'] if a['ID'] == wildcards.ID.split('__')[0] + '_'),
            "MISSING_TIME_START"
        ),
        time_end=lambda wildcards: next(
            (a['time_end'] for a in config['animals'] if a['ID'] == wildcards.ID.split('__')[0] + '_'),
            "MISSING_TIME_END"
        )
    output:
        gpkg="output/spatial_metrics/{ID}_{time_start}_{time_end}.gpkg",
        plot="plots/{ID}_{time_start}_{time_end}.html"
    shell:
        """
        if [[ "{params.time_start}" == "MISSING_TIME_START" || "{params.time_end}" == "MISSING_TIME_END" ]]; then
            echo "Error: Missing time_start or time_end for ID {params.ID}" >&2
            exit 1
        fi
        python scripts/calculate_spatial_metrics.py {input.csvs} --gpkg_output {output.gpkg} --plot_output {output.plot} --ID "{params.ID}" --time_start "{params.time_start}" --time_end "{params.time_end}"
        """
