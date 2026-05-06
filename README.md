# Microplastic water-systems meta-analysis — thesis code companion

Code and figure-reproduction data accompanying the MSc thesis

> Shen, Y. (2026). *Developing a hybrid framework for estimating microplastic retention rates in global river systems* [MSc thesis, Wageningen University & Research].

The repository covers Chapters 2 to 4 of the thesis end-to-end:

- **Chapter 2 — global microplastic-abundance database**: a three-stage literature-screening pipeline (rule-based regex → ASReview active learning → Gemini-3-pro-preview LLM full-text screening) followed by an LLM data-mining stage that extracts 4,736 site-level records from the 329 retained papers, and a record-level QC filter that yields the final 3,148-record / 251-study database.
- **Chapter 3 — LightGBM-with-SHAP machine-learning pipeline**: takes the Chapter 2 database (3,081 records after zero-removal) and 38 HydroATLAS-v10 predictors at HydroBASINS Level 12, trains a baseline LightGBM model with Optuna (50 trials) and 10-fold cross-validation, applies the trained model to every Level 12 sub-basin worldwide (n = 1,034,083), interprets the model with SHAP and structural knee-point detection, and quantifies prediction uncertainty with a six-configuration robustness check.
- **Chapter 4 — spatially explicit retention pipeline**: brings the Chapter 3 annual loads onto the 10,226 MARINA sub-basins of MARINA-Multi (Micella et al., 2024), computes the new sub-basin retention rate of this study (R_new) from a within-sub-basin mass balance, and compares it sub-basin-by-sub-basin with the MARINA-Multi baseline retention rate (R_MARINA).

## Repository layout

```
microplastic-water-thesis-code/
├── README.md                         # this file
├── LICENSE                           # MIT licence
├── requirements.txt                  # Python dependencies
├── .gitignore
│
├── ch2_01_rule_based_screening/      # Ch2 Step 1 stage (i): five-filter regex screen
├── ch2_02_asreview_active_learning/  # Ch2 Step 1 stage (ii): ASReview ranking
├── ch2_03_llm_full_text_screening/   # Ch2 Step 1 stage (iii): Gemini full-text screen
├── ch2_04_llm_data_extraction/       # Ch2 Step 2: Central Prompt + extraction wrapper
│
├── ch3_machine_learning/             # Ch3 LightGBM + SHAP + knee point + 6-config CoV
│   ├── 01_data_prep/
│   ├── 02_training/                  # baseline + SHAP5 + CT3 + CT5 + CT7 + Jin5 + global apply
│   ├── 03_shap_interpretation/
│   └── 04_model_comparison/          # 6-config CoV, agreement maps, statistics
│
├── ch4_retention/                    # Ch4 retention pipeline
│   ├── 01_spatial_harmonization/     # HydroBASINS Level 12 → MARINA sub-basin
│   ├── 02_retention_calculation/     # R_new + size-class imputation
│   └── 03_comparison/                # ΔR = R_new − R_MARINA
│
└── data/                             # figure-reproduction inputs and final outputs
    ├── README.md                     # data dictionary
    ├── chapter_2/                    # 251-study reference list, master extraction table
    ├── chapter_3/                    # training data, global predictions, validation
    └── chapter_4/                    # per-sub-basin master, R_new, ΔR, summary workbook
```

Each chapter folder has its own `README.md` with a phase/step map, run instructions, and the input/output file list.

## Quick start

```bash
git clone https://github.com/syydaniel/microplastic-water-thesis-code.git
cd microplastic-water-thesis-code
pip install -r requirements.txt
```

The pipelines are designed to run chapter-by-chapter; see the per-chapter README for the script order.

For the LLM stages (Ch2 Step 1.iii and Step 2) you also need a Gemini API key:

```bash
export GEMINI_API_KEY="<your-key>"
```

For ASReview (Ch2 Step 1.ii) install separately and run interactively:

```bash
pip install asreview
asreview lab
```

## Citation

If you use this code or any of the data files in this repository, please cite the thesis and the underlying tools:

> Shen, Y. (2026). *Developing a hybrid framework for estimating microplastic retention rates in global river systems* [MSc thesis, Wageningen University & Research].
>
> Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, *30*, 3146–3154.
>
> Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, *30*, 4765–4774.
>
> Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701
>
> van de Schoot, R., de Bruin, J., Schram, R., Zahedi, P., de Boer, J., Weijdema, F., Kramer, B., Huijts, M., Hoogerwerf, M., Ferdinands, G., Harkema, A., Willemsen, J., Ma, Y., Fang, Q., Hindriks, S., Tummers, L., & Oberski, D. L. (2021). An open source machine learning framework for efficient and transparent systematic reviews. *Nature Machine Intelligence*, *3*(2), 125–131. https://doi.org/10.1038/s42256-020-00287-7

The MARINA-Multi spatial unit (10,226 sub-basins) and the baseline retention parameters (L_MIP, FQrem) are from Micella et al. (2024) and must be cited in any downstream use of `ch4_retention/`.

## Licence

MIT licence (see `LICENSE`). The compiled microplastic abundance database is released for research and educational use; the original source publications and the HydroATLAS / MARINA-Multi underlying datasets retain their original licences.

## Reproducibility note

The scripts and prompts in this repository, run on the same input record set and the same Gemini model snapshot used in November 2025, are expected to reproduce the numbers reported in the thesis. Re-running on a fresh literature search (newer publications) will produce a fresh database; the pipeline is otherwise deterministic given fixed inputs.
