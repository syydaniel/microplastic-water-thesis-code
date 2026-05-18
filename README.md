# Microplastic water-systems meta-analysis — thesis code companion

Code and figure-reproduction data accompanying the MSc thesis

> Shen, Y. (2026). *Developing a hybrid framework for estimating microplastic retention rates in global river systems* [MSc thesis, Wageningen University & Research].

This repository is reorganised so that each thesis chapter has its own top-level folder. Each chapter folder contains up to four sub-folders:

| Sub-folder | What it holds |
|---|---|
| `code/` | Python or R scripts to reproduce the chapter's pipeline |
| `data/` | Per-chapter data inputs and outputs (figure-reproduction data) |
| `figures/` | Figure-builder scripts for the chapter's main-text figures |
| `references/` | Appendix-only reference lists relocated from the thesis appendix |

## Repository layout

```
microplastic-water-thesis-code/
├── README.md                                        # this file
├── LICENSE
├── requirements.txt
├── .gitignore
├── Ch1/                                             # (no code; references only)
│   └── references/                                  # (placeholder)
├── Ch2/                                             # Chapter 2 — global microplastic-abundance database
│   ├── code/
│   │   ├── 01_rule_based/                           # (was ch2_01_rule_based_screening/)
│   │   ├── 02_asreview/                             # (was ch2_02_asreview_active_learning/)
│   │   ├── 03_llm_full_text/                        # (was ch2_03_llm_full_text_screening/)
│   │   └── 04_llm_extraction/                       # (was ch2_04_llm_data_extraction/)
│   ├── data/                                        # clean_dataset.csv, primary studies APA list
│   ├── figures/                                     # (build_ch2_figures.R if present)
│   └── references/                                  # appendix-only ref lists for Ch2
├── Ch3/                                             # Chapter 3 — LightGBM + SHAP pipeline
│   ├── code/                                        # (was ch3_machine_learning/)
│   ├── data/                                        # training observations + Level 12 prediction table
│   ├── figures/                                     # build_ch3_figures.py
│   └── references/
│       ├── appendix_a4_1_2_hydroatlas_sources.md    # 17 HydroATLAS source dataset refs
│       └── appendix_a4_1_2_hydroatlas_sources.bib
├── Ch4/                                             # Chapter 4 — spatially explicit retention pipeline
│   ├── code/                                        # (was ch4_retention/)
│   ├── data/                                        # per-MARINA-sub-basin master table
│   ├── figures/                                     # build_ch4_figures.py
│   └── references/
└── Ch5/                                             # (synthesis chapter; references only)
    └── references/                                  # (placeholder)
```

## What each chapter folder covers

- **Chapter 2 — global microplastic-abundance database**: a three-stage literature-screening pipeline (rule-based regex → ASReview active learning → Gemini-3-pro-preview LLM full-text screening) followed by an LLM data-extraction stage that produces 4,736 site-level records from the 329 retained papers, and a record-level QC filter that yields the final 3,148-record / 251-study database.
- **Chapter 3 — LightGBM-with-SHAP machine-learning pipeline**: takes the Chapter 2 database (3,081 records after zero-removal) and 38 HydroATLAS-v10 predictors at HydroBASINS Level 12, trains a baseline LightGBM model with Optuna (50 trials) and 10-fold cross-validation, applies the trained model to every Level 12 sub-basin worldwide (n = 1,034,083), interprets the model with SHAP and structural knee-point detection, and quantifies prediction uncertainty with a six-configuration robustness check.
- **Chapter 4 — spatially explicit retention pipeline**: brings the Chapter 3 annual loads onto the 10,226 MARINA sub-basins of MARINA-Multi (Micella et al., 2024), computes the new sub-basin retention rate of this study (R_new) from a within-sub-basin mass balance, and compares it sub-basin-by-sub-basin with the MARINA-Multi baseline retention rate (R_MARINA).

## References folder convention

Each chapter's `references/` folder holds the appendix-only reference lists that were moved out of the thesis docx to keep the appendix lean. Each list comes in two formats:

- `*.md` — human-readable markdown
- `*.bib` — BibTeX for citation managers

The main thesis bibliography stays inside the thesis docx; only appendix-only lists (typically per-database source citations) live here.

## Reproducing the figures

Each chapter's `figures/` folder contains a single `build_chN_figures.py` (or `.R` for Chapter 2) that regenerates every main-text figure for that chapter. All scripts use Times New Roman to match the thesis typography.
