rule clean_csv:
    input:
        "data/{sample}.csv"
    output:
        "output/cleaned/{sample}.csv"
    shell:
        """
        mkdir -p output/cleaned
        python /Users/didemdost/Desktop/geo_track/scripts/clean_csv.py {input} {output}
        """
