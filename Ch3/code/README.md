# Chapter 3 — LightGBM-with-SHAP machine-learning pipeline

This folder contains the Python scripts that, given the Chapter 2 harmonised observation database (`../data/chapter_3/training_observations.csv`, n = 3,081 records from 251 studies) and 38 candidate predictors from HydroATLAS v10, produce the global continuous map of microplastic abundance and annual load reported in Chapter 3 of the thesis.

The folder is organised by the five methodology phases defined in Section 3.2 of the thesis (Figure 3.1):

| Phase | Folder | Scripts |
|---|---|---|
| Phase 2 — Data preparation (§3.2.2) | `01_data_prep/` | `01_dataset_combine.py` (joins observations to HydroATLAS predictors at HydroBASINS Level 12), `02_sensitivity_data_sampling.py` (random subsets used in the data-size sensitivity analysis of §3.3.3.1). |
| Phase 3 — Baseline model and global apply (§3.2.3) | `02_training/` | `02_train_baseline.py` (Optuna 50 trials + 10-fold CV, all 38 predictors), `02_train_shap5.py` / `02_train_ct3.py` / `02_train_ct5.py` / `02_train_ct_cluster_helper.py` (the five reduced configurations: SHAP5, CT3, CT5, CT7, Jin5), `03_apply_global_*.py` (apply each trained model to the global HydroBASINS Level 12 sub-basins), `05_extract_optuna_artifacts.py` (export the chosen hyperparameters per trial). |
| Phase 4 — SHAP interpretation and knee-point detection (§3.2.4) | `03_shap_interpretation/` | `lightgbm_shap_pdp_kneepoint.py` (SHAP values + 95 % CI Partial Dependence Plots + structural knee-point detection), `01_shap_dependence_loess.py`, `02_calc_top6_contribution.py`, `02_shap_interaction_plot.py`, `03_combine_shap_top4.py`. |
| Phase 5 — Six-configuration robustness check (§3.2.5) | `04_model_comparison/` | `00_comparison_analysis.py`, `01_global_agreement_analysis.py`, `03_model_vis_comparison.py`, `04_agreement_statistics_table.py`, `05_statistical_comparison_groups.py`, `06_model_performance_scatter.py`, `07_cov_statistics.py` (per-sub-basin Coefficient of Variation across the six configurations), `08_abundance_vs_uncertainty.py`. |

## Inputs and outputs

- **Training data (`../data/chapter_3/training_observations.csv`)** — 3,081 records, columns: `Unique_ID`, `Study_ID`, `Title`, `Year`, `First_Author`, `Latitude`, `Longitude`, `Season_Calculated`, `Gear_Category`, `Std_Value_m3`, `Separation_Category`, shape and polymer composition fields, `Mesh_Size_um`, plus the 38 HydroATLAS-v10 predictor columns at HydroBASINS Level 12, plus `HYBAS_ID`.
- **Global predictor table** — the 38 HydroATLAS-v10 predictor columns for every HydroBASINS Level 12 sub-basin (n = 1,034,083). The full per-sub-basin table is too large for this repository (~290 MB compressed); a simplified version showing the predicted abundance and annual load is provided at `../data/chapter_3/global_predictions_lev12.csv` and is sufficient to reproduce the global maps.
- **Outputs** — predicted microplastic abundance (items m⁻³) and annual load (items y⁻¹) per HydroBASINS Level 12 sub-basin (and aggregated to Level 6 for visualisation), six-configuration uncertainty maps, SHAP-based feature importance, and structural knee points for the four highest-ranking predictors.

## Run

```bash
pip install -r ../requirements.txt
# Phase 2
python 01_data_prep/01_dataset_combine.py
# Phase 3 — train the baseline + apply to the global sub-basins
python 02_training/02_train_baseline.py
python 02_training/03_apply_global_baseline.py
# Phase 4 — SHAP, PDP, knee point
python 03_shap_interpretation/lightgbm_shap_pdp_kneepoint.py
# Phase 5 — robustness check across six configurations
python 02_training/02_train_shap5.py
python 02_training/02_train_ct3.py
python 02_training/02_train_ct5.py
python 04_model_comparison/00_comparison_analysis.py
python 04_model_comparison/07_cov_statistics.py
```

Most scripts read paths from constants near the top of each file. Adjust those constants to point to the local input data directory before running. All scripts print a one-line summary on success and write outputs to a per-script subfolder (typically named after the script).
