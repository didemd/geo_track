
from helper_functions import *

configfile: "config.yaml"

# report: "report/workflow.rst"

rule all:
    input:
        gdf_gpkg = "output/output.gpkg",
        csv_file = "output/output.csv",

# import rule definitions
include: "rules/csv.smk"
