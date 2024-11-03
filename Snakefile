configfile: "config.yaml"

import os
from snakemake.io import glob_wildcards

# Get the list of samples (CSV files)
samples = glob_wildcards("data/{sample}.csv").sample

# Prepare lists for IDs and time frames
animals = config['animals']
IDs = [a['ID'] for a in animals]
time_starts = [a['time_start'].replace(":", "").replace(" ", "_") for a in animals]
time_ends = [a['time_end'].replace(":", "").replace(" ", "_") for a in animals]

rule all:
    input:
        expand("output/cleaned/{sample}.csv", sample=samples),
        expand(
            "output/spatial_metrics/{ID}_{time_start}_{time_end}.gpkg",
            zip,
            ID=IDs,
            time_start=time_starts,
            time_end=time_ends
        ),
        expand(
            "plots/{ID}_{time_start}_{time_end}.html",
            zip,
            ID=IDs,
            time_start=time_starts,
            time_end=time_ends
        )

# Import rule definitions
include: "rules/csv.smk"
include: "rules/spatial_metrics.smk"

# include: "rules/plot.smk"
