# Chapter 4 — Spatially explicit microplastic retention pipeline

This folder contains the Python scripts that bring the Chapter 3 annual microplastic loads onto the 10,226 MARINA sub-basins of MARINA-Multi (Micella et al., 2024), compute the new sub-basin retention rate of this study (R_new), and compare it sub-basin-by-sub-basin with the MARINA-Multi baseline retention rate (R_MARINA).

The folder is organised by the three methodology steps defined in Section 4.2 of the thesis (Figure 4.1):

| Step | Folder | Scripts |
|---|---|---|
| Step 1 — Spatial harmonization onto the MARINA sub-basins (§4.2.1) | `01_spatial_harmonization/` | `01_create_base_grid.py` (overlay the 0.5° DDM30 grid on the MARINA sub-basin shapefile), `02_validate_data_coverage.py` (active-cell coverage filter), `03_aggregate_inside_load.py` (sum HydroBASINS Level 12 centroid loads to MP_inside), `04_process_outlet_load.py` (read MP_out at the mapped outlet of each MARINA sub-basin). |
| Step 2 — Retention rate calculation and imputation (§4.2.2) | `02_retention_calculation/` | `05_calculate_R_new.py` (within-sub-basin mass balance: R_new = 1 − MP_out / MP_inside), `06_impute_size_class.py` (size-class imputation for sub-basins that fail Step 1 filters), `12_create_deliverables.py` and `generate_final_deliverables.py` (consolidate the per-sub-basin master tables for Chapter 4 figures). |
| Step 3 — Comparison with MARINA-Multi baseline (§4.2.3) | `03_comparison/` | `01_run_comparison_analysis.py` (compute ΔR per sub-basin, decile-bin statistics, top-10 highest-load sub-basins), `02_build_figures.py` (Figures 4.10 to 4.14). |

## Inputs and outputs

- **Inputs**
  - The 10,226 MARINA sub-basin shapefile (from MARINA-Multi; not redistributed here — see Micella et al., 2024).
  - The Chapter 3 HydroBASINS Level 12 predicted annual loads (`../data/chapter_3/global_predictions_lev12.csv`).
  - The MARINA-Multi 2010 SSP1-RCP2.6 baseline columns `Lmip_dst_10_s1` (the binary in-river microplastic retention factor L_MIP) and `FQrem_10_s1` (the in-river water-removal fraction at the outlet FQrem).

- **Outputs (in `../data/chapter_4/`)**
  - `per_subbasin_master.csv` — the master per-MARINA-sub-basin table with all variables (MP_inside, MP_out, R_new, R_MARINA, ΔR, sub-basin size class, generation-load decile).
  - `R_new_per_subbasin.csv` — per-sub-basin R_new values (the headline output of this chapter).
  - `deltaR_per_subbasin.csv` — per-sub-basin ΔR = R_new − R_MARINA.
  - `analysis_size_class.csv` / `analysis_discharge_class.csv` / `analysis_latitudinal.csv` — per-class summary statistics used in Figures 4.12 and 4.13.
  - `chapter_4_summary.xlsx` — multi-sheet workbook with the figure-level data behind every figure in Chapter 4.

## Run

```bash
pip install -r ../requirements.txt
# Step 1
python 01_spatial_harmonization/01_create_base_grid.py
python 01_spatial_harmonization/02_validate_data_coverage.py
python 01_spatial_harmonization/03_aggregate_inside_load.py
python 01_spatial_harmonization/04_process_outlet_load.py
# Step 2
python 02_retention_calculation/05_calculate_R_new.py
python 02_retention_calculation/06_impute_size_class.py
python 02_retention_calculation/12_create_deliverables.py
# Step 3
python 03_comparison/01_run_comparison_analysis.py
python 03_comparison/02_build_figures.py
```

Most scripts read input/output paths from constants near the top. Set those constants to point to the local MARINA-Multi shapefile and the local Chapter 3 predictions directory before running.

## Variable naming convention

This pipeline uses the unified main-text symbols of the thesis throughout: `MP_inside` for the annual microplastic load produced inside a MARINA sub-basin, `MP_out` for the annual outlet load, `R_new` for the new sub-basin retention rate of this study (Equation 4.1), and `R_MARINA` for the MARINA-Multi baseline retention rate (Equation 4.2). Some intermediate-output filenames produced by older code revisions still mention `Genload` / `Outload` / `gen_flux` / `out_flux`; these are exactly the same quantities as `MP_inside` / `MP_out`.
