# Figure regeneration scripts (MSc Thesis — Yiyang Shen)

This folder contains the **stand-alone Python scripts** used to regenerate every figure in the thesis with a unified visual style (Times New Roman, bold axis labels, 300 dpi).

## Important: these scripts only **read existing model outputs** — they do NOT retrain anything

The scripts in this folder are visualization-only. They consume the CSV/Excel/shapefile outputs that the original Ch3 / Ch4 pipelines already produced. They do not call `lgb.train()`, do not run Optuna, and do not regenerate any of the global predictions or SHAP values. The only model-related call is `model.predict()` inside `build_ch3_figures.py` for Fig 3.10 (PDP curves of the top-4 drivers); this is a deterministic read of the trained model and produces no new training artefacts.

## File index

| File | Produces | Reads from |
|---|---|---|
| `thesis_style.py` | (no output — the shared matplotlib style; imported by all other scripts) | none |
| `build_ch2_figures.py` | Fig 2.2 – 2.8 (temporal trend, regional bars, abundance distribution, composition, gear/chemical, mesh-size scatter, gear×chemical×mesh facet) | `Thesis_Organized/03_Code_Final/Chapter_3/Export_Data/Training_Observation_Data.csv` |
| `build_fig_2_1_prisma.py` | Fig 2.1 (PRISMA flowchart) | none — fully constructed in matplotlib |
| `build_ch3_figures.py` | Fig 3.6 (frequency dist), 3.9 (SHAP summary), 3.10 (PDP top-4 with knees), 3.11 (six-config global stats), 3.12 (CoV vs abundance) | `01_Data_Prep/01_Data_Combine_Result/data_combine.csv`, `02_Training/02_Model_Results/{shap_values.csv, final_model.pkl}`, `02_Training/03_Global_Results_Refined/Global_Stats_Lev6.csv`, `04_Model_Comparison/Global_Results_Agreement_Levels.csv`, `04_Model_Comparison/01_Stats_Files/Global_Stats_Lev6_*.csv` |
| `build_ch4_figures.py` | Fig 4.10 (R_new histogram), 4.11 (cumulative R_new vs R_MARINA), 4.12 (size-class bars, **supervisor x-axis fix**), 4.13 (ΔR vs MP_inside, **supervisor x-axis fix**) | `Chapter_4/V7_Comparison_Analysis/outputs/Chapter_4_Summary.xlsx` (sheets: per_basin, by_size, top10_load) |
| `build_flowcharts.py` | Fig 1.1 (research framework), Fig 3.1 (Ch3 methodology, **supervisor RQ2 fix**), Fig 4.1 (Ch4 methodology), Fig 5.1 overview | none — fully constructed in matplotlib |
| `build_maps.py` | Fig 3.4 (mask), 3.5 (discharge), 3.7 + 3.7a/b/c (abundance global map + Maharashtra/Mississippi/Kaduna zooms, **separate 4:3 PNGs, May 2026**), 3.8 + 3.8a/b (annual load global map + North America/East Asia zooms, **separate 4:3 PNGs**), 3.13 (CoV map), 4.4 (filter outcome), 4.6 (calc/imputed/NA), 4.8 (MP_inside, **supervisor colorbar fix**), 4.9 (MP_out, **supervisor colorbar fix**), 4.14 (3-panel R_MARINA / R_new / ΔR, **supervisor panel-b fix**) | `BasinATLAS_v10_lev06.shp`, `Global_Stats_Lev6.csv`, `Global_Results_Agreement_Levels.csv`, `Top10_Subbasins_Detailed.csv`, `Shapefile_Retention_Clear.shp`, `Chapter_4_Summary.xlsx` |

## How to re-run

These paths are absolute. Run from anywhere:

```bash
python3 thesis_style.py                # smoke test (optional)
python3 build_ch2_figures.py           # ~10 s
python3 build_fig_2_1_prisma.py        # ~3 s
python3 build_ch3_figures.py           # ~30 s (PDP loop is the slow step)
python3 build_ch4_figures.py           # ~5 s
python3 build_flowcharts.py            # ~5 s
python3 build_maps.py                  # ~60 s (geopandas loading + drawing)
```

Each script writes its outputs to two places:

```
/Users/a1-6/Desktop/MASTER THESIS/Figures/Chapter X/Figure_X_Y.png
/Users/a1-6/Desktop/Thesis_Organized/02_Figures_Final/Chapter_X/Figure_X_Y.png
```

The first path is the user's working folder. The second path is what `build_thesis_v5.py` reads when it embeds figures into the consolidated thesis docx.

## Style invariants (every figure)

- Font: Times New Roman, 12 pt body, bold axis labels (per supervisor).
- Axis line width: 1.2 pt; tick width: 1.0 pt.
- Save: 300 dpi, tight bbox.
- Categorical palette: defined in `thesis_style.PALETTE` so the same gear / chemical / mesh-bin / sub-basin status uses the same colour across every figure.

## Supervisor edits applied

| Figure | Old text/style | New |
|---|---|---|
| 2.1 | "Cateria" typo, "in-land water system" | "Criteria", "inland water systems" |
| 3.1 | RQ2 wording mismatch with §1.4 | Canonical "How can the existing observations of microplastics in river basins be harmonised in a spatially explicit way through machine learning approaches?" |
| 4.3 | legend "DDM30 Sub-basin Boundary" | "MARINA Sub-basin Boundary" (script `build_maps.py` does NOT regenerate Fig 4.3 — see below) |
| 4.8 | colorbar "Genload" | "MP_inside" |
| 4.9 | colorbar "Outload" | "MP_out" |
| 4.12 | x-axis "Basin-size class (DDM30 grid cells)" | "MARINA sub-basin size class (number of 0.5° grid cells)" |
| 4.13 | x-axis "log₁₀(annual microplastic generation load, items y⁻¹)" | "log₁₀(MP_inside, items y⁻¹)" |
| 4.14 | panel (b) subscript "(Calculated)" | dropped (panel shows both calculated and imputed) |

## Figures NOT regenerated by these scripts

The following figures still rely on the existing PNGs in `MASTER THESIS/Figures/Chapter X/` and were not re-built:

- Fig 3.2 — sampling-site distribution map (3,148 dots on a basemap; can be reproduced from the same observation database with a 1-line scatter on cartopy)
- Fig 3.3 — Level-6 vs Level-12 example basins (illustrative, hand-curated)
- Fig 4.3 — active-cell coverage two-panel example (uses two specific MARINA sub-basins as illustration)

If you want any of these regenerated to match the unified style, let me know and I'll add them.

## What I would NOT recommend automating again

- The Optuna 50-trial hyperparameter sweep (Ch3 Section 3.2.3.1).
- The 6-configuration training runs.
- The HydroBASINS Level 12 → Level 6 spatial join.

These all live in your existing pipeline (`Thesis_Organized/03_Code_Final/Chapter_3/02_Training/...`) and the outputs are already saved as CSV/PKL. Re-running them would: (a) burn a lot of CPU, (b) introduce slightly different random-seed-dependent results, (c) not change the published numbers in the thesis. Stick with the saved outputs.
