# Import necessary helper functions
from helper_functions import *

# Specify the configuration file to be used by Snakemake
configfile: "config.yaml"

# Define the primary rule to specify the final expected outputs
rule all:
    input:
        "output/plot_spatial_utilization_heatmap_with_map.png"

# Include rules from other Snakefiles for better organization
include: "rules/csv.smk"
