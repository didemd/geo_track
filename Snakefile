
from helper_functions import *

configfile: "config.yaml"

# report: "report/workflow.rst"
rule all:
    input:
        gdf_gpkg = "output/output.gpkg",
        csv_file = "output/output.csv",

# include rules
include: "rules/csv.smk"


# if config.output["trim_primers"]:

#     include: "rules/primerstrim.smk"
