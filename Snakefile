import glob
import os

# Define directories for input and output data
input_directory = "data"
output_directory = "output"
script_directory = "scripts"
plot_directory = os.path.join(output_directory, "plots")
combined_directory = os.path.join(output_directory, "combined")

# Gather all CSV files from the input directory
CSV_FILES = glob.glob(os.path.join(input_directory, '*.csv'))
CSV_BASENAMES = [os.path.splitext(os.path.basename(f))[0] for f in CSV_FILES]

# Define rules
rule all:
    input:
        expand(os.path.join(output_directory, "{file}_processed.csv"), file=CSV_BASENAMES),
        expand(os.path.join(output_directory, "{file}.gpkg"), file=CSV_BASENAMES),
        expand(os.path.join(plot_directory, "{file}_lat_lon.html"), file=CSV_BASENAMES),
        expand(os.path.join(combined_directory, "{prefix}_cleaned.gpkg"), prefix=["NPL28", "NPL42", "NPL35"])

rule process_csv_to_gpkg:
    input:
        csv_file = os.path.join(input_directory, "{file}.csv")
    output:
        processed_csv = os.path.join(output_directory, "{file}_processed.csv"),
        gpkg_file = os.path.join(output_directory, "{file}.gpkg")
    shell:
        """
        python {script_directory}/process_csv_files.py process {input.csv_file} {output_directory}
        """

rule generate_plots:
    input:
        gpkg_file = os.path.join(output_directory, "{file}.gpkg")
    output:
        plot_file = os.path.join(plot_directory, "{file}_lat_lon.html")
    shell:
        """
        python {script_directory}/process_csv_files.py plot {input.gpkg_file} {plot_directory}
        """

rule combine_gpkg:
    input:
        expand(os.path.join(output_directory, "{file}.gpkg"), file=CSV_BASENAMES)
    output:
        expand(os.path.join(combined_directory, "{prefix}_cleaned.gpkg"), prefix=["NPL28", "NPL42", "NPL35"])
    shell:
        """
        python {script_directory}/process_csv_files.py combine {output_directory}/*.gpkg {combined_directory}
        """
