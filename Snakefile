import glob
import os

# Define directories for input and output data
input_directory = "data"
output_directory = "output"
script_directory = "scripts"
plot_directory = os.path.join(output_directory, "plots")
combined_directory = os.path.join(output_directory, "combined")
log_directory = os.path.join(output_directory, "logs")

# Create directories if they don't exist
os.makedirs(log_directory, exist_ok=True)
os.makedirs(plot_directory, exist_ok=True)
os.makedirs(combined_directory, exist_ok=True)

# Gather all CSV files from the input directory
CSV_FILES = glob.glob(os.path.join(input_directory, '*.csv'))
CSV_BASENAMES = [os.path.splitext(os.path.basename(f))[0] for f in CSV_FILES]

# Filter only NPL28.csv
CSV_BASENAMES = [f for f in CSV_BASENAMES if f == "NPL28"]

# Define Snakemake rules
rule all:
    input:
        expand(os.path.join(output_directory, "{file}_processed.csv"), file=CSV_BASENAMES),
        expand(os.path.join(output_directory, "{file}.gpkg"), file=CSV_BASENAMES),
        os.path.join(combined_directory, "NPL28_cleaned.gpkg"),
        os.path.join(output_directory, "NPL28_cleaned_head.txt"),
        expand(os.path.join(plot_directory, "{file}_kde_plot.png"), file=CSV_BASENAMES),
        expand(os.path.join(output_directory, "{file}_density_polygons.gpkg"), file=CSV_BASENAMES)

rule process_csv_to_gpkg:
    input:
        csv_file = os.path.join(input_directory, "{file}.csv")
    output:
        processed_csv = os.path.join(output_directory, "{file}_processed.csv"),
        gpkg_file = os.path.join(output_directory, "{file}.gpkg")
    params:
        mode = "process"
    log:
        os.path.join(log_directory, "{file}_process.log")
    shell:
        """
        python {script_directory}/process_csv_files.py {params.mode} {input.csv_file} {output_directory} &> {log}
        """

rule combine_gpkg:
    input:
        gpkg_files = expand(os.path.join(output_directory, "{file}.gpkg"), file=CSV_BASENAMES)
    output:
        combined_file = os.path.join(combined_directory, "NPL28_cleaned.gpkg")
    log:
        os.path.join(log_directory, "combine.log")
    shell:
        """
        python {script_directory}/combine_gpkg_files.py {input.gpkg_files} {output.combined_file} &> {log}
        """

rule generate_kde_plot_and_polygons:
    input:
        gpkg_file = os.path.join(output_directory, "{file}.gpkg")
    output:
        plot_file = os.path.join(plot_directory, "{file}_kde_plot.png"),
        polygon_file = os.path.join(output_directory, "{file}_density_polygons.gpkg")
    log:
        os.path.join(log_directory, "{file}_kde_plot.log")
    shell:
        """
        python {script_directory}/generate_kde_plot.py {input.gpkg_file} {output.plot_file} {output.polygon_file} &> {log}
        """

rule get_head_of_gpkg:
    input:
        combined_file = os.path.join(combined_directory, "NPL28_cleaned.gpkg")
    output:
        head_file = os.path.join(output_directory, "NPL28_cleaned_head.txt")
    log:
        os.path.join(log_directory, "get_head.log")
    shell:
        """
        python {script_directory}/get_head_of_gpkg.py {input.combined_file} {output.head_file} &> {log}
        """
