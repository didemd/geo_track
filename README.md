
# Automating Spatial Metrics Calculation and Visualization with Snakemake: geo_track

## 📌 Overview
This repository contains the workflow developed during my internship at the **Technical University of Munich (TUM)**.  
The project focuses on **automating the processing, analysis, and visualization of geospatial movement data** for wildlife tracking using **Snakemake**.  

The pipeline improves:
- ✅ **Reproducibility** of analyses  
- ✅ **Adaptability** to diverse datasets  
- ✅ **Clarity** of spatial insights for ecological research and conservation:contentReference[oaicite:0]{index=0}

---

## 🎯 Motivation
- Identify **critical habitats and migration routes** through geospatial data.  
- Detect **high-use areas and temporal patterns** to guide conservation.  
- Improve **workflow automation** for scalable, repeatable analyses.  

---

## 🗂️ Data
### Input  
- CSV files representing animal movement patterns:  
  - `NPL28_NCRST_Permit_Nov22-Oct23(WGS84).csv`  
  - `NPL35_NCRST_Permit_Nov22-Oct23(WGS84).csv`  
  - `NPL42_NCRST_Permit_Nov22-Oct23(WGS84).csv`  

### Output  
- Cleaned CSVs with standardized headers & datetime column.  
- **Kernel Density Estimation (KDE)** plots (50% core, 95% home range).  
- **Minimum Convex Hull** polygons.  
- Interactive **Google Earth Engine maps**.  
- GeoPackage (GPKG) files for downstream use.  

---

## ⚙️ Workflow
Implemented with **Snakemake**, the workflow is modular and reproducible.

## Repository Structure

├── Snakefile
├── rules/
│   ├── csv.smk                  # CSV cleaning rules
│   ├── calculate_spatial_metrics.smk  # Convex hull & KDE
├── config.yaml                  # Animal IDs & time intervals
├── scripts/
│   ├── clean_csv.py
│   ├── calculate_spatial_metrics.py
│   └── visualization.py
├── data/                        # Raw input CSV files
├── results/                     # Outputs: cleaned CSVs, GPKGs, plots
└── README.md

