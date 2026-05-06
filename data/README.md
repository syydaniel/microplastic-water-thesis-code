# Data — figure-reproduction inputs and final outputs

The CSV / XLSX files in this directory are the **final outputs and figure-reproduction inputs** of the thesis. Together they let a third-party reader regenerate all figures in Chapters 2 to 4 without re-running the full machine-learning pipeline. They do **not** include intermediate per-trial tables or the full Optuna database (those are in the order of hundreds of megabytes and exceed GitHub's per-file size limit; the scripts in `ch3_machine_learning/` and `ch4_retention/` produce them locally).

## Chapter 2 — global microplastic-abundance database

| File | Rows | Contents |
|---|---|---|
| `chapter_2/retained_primary_studies_251.csv` | 251 | The full APA reference list of the 251 primary studies retained after QC (Chapter 2 Step 3). Columns: `study_id`, `first_author`, `year`, `title`, `author_full`, `journal`, `volume`, `pages`, `doi`, `apa_citation` (drop-in formatted). |
| `chapter_2/master_reference_table.csv` | 283 | The master EndNote-derived reference table that feeds the QC step (includes pre-QC entries and full abstracts). |
| `chapter_2/extracted_records_combined.csv` | 4,736 raw | Extracted-records table from the LLM data-mining stage of Step 2 (before the record-level QC filter that produces the 3,148-record clean dataset). |

## Chapter 3 — LightGBM training and global predictions

| File | Rows | Contents |
|---|---|---|
| `chapter_3/training_observations.csv` | 3,081 | Training-target file used as the input to the LightGBM workflow (Section 3.2.2). One row per observed sampling site. Columns: `Unique_ID`, `Study_ID`, location, sampling methodology, abundance (`Std_Value_m3`), shape and polymer composition, sampling year and mesh size, plus the 38 HydroATLAS-v10 predictor columns at HydroBASINS Level 12, plus `HYBAS_ID`. |
| `chapter_3/global_predictions_lev12.csv` | 1,034,083 | Predicted microplastic abundance and annual load at every HydroBASINS Level 12 sub-basin worldwide. The simplified (figure-reproduction) version of the full Phase-3 output table; carries the predicted log-abundance, the predicted abundance back-transformed to items m⁻³, the natural discharge in m³ s⁻¹, and the derived annual load in items y⁻¹. |
| `chapter_3/validation_comparison.csv` | 251 | Per-study extraction-vs-source comparison used in Section 2.3.1 / Appendix A.1.3 (study-level abundance match flag). |

## Chapter 4 — sub-basin retention pipeline outputs

| File | Rows | Contents |
|---|---|---|
| `chapter_4/per_subbasin_master.csv` | 10,226 | Master per-MARINA-sub-basin table with every variable used in the comparison: identifier, area, size class, MP_inside, MP_out, R_new, R_MARINA, ΔR, generation-load decile bin. |
| `chapter_4/R_new_per_subbasin.csv` | 10,226 | Just the per-sub-basin new retention rate of this study (R_new), with the imputation flag (Direct / Imputed / Not Available). |
| `chapter_4/deltaR_per_subbasin.csv` | 10,226 | Per-sub-basin ΔR = R_new − R_MARINA. |
| `chapter_4/analysis_size_class.csv` | 5 | Mean ΔR ± SD per sub-basin-size class (≤ 4, 5–10, 11–50, 51–200, > 200 0.5° grid cells). Source for Figure 4.12. |
| `chapter_4/analysis_discharge_class.csv` | 10 | Mean ΔR ± SD per generation-load decile (≈ 885 sub-basins per bin). Source for Figure 4.13. |
| `chapter_4/analysis_latitudinal.csv` | 18 | Mean R_new and R_MARINA per 10°-latitude band. |
| `chapter_4/discharge_and_area_analysis.csv` | 10,226 | Discharge × area scatter underlying the per-basin agreement plots. |
| `chapter_4/chapter_4_summary.xlsx` | multi-sheet | All Chapter 4 figure data in one workbook (one sheet per figure). |

## Provenance

All data files in this directory were produced by the scripts in `ch2_*`, `ch3_machine_learning/`, and `ch4_retention/` of this repository, on the input record set assembled in October–November 2025. Re-running the same scripts on the same inputs is expected to reproduce these files bit-for-bit; re-running with a fresh literature search will produce a fresh database.
