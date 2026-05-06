# Stage 1 — Rule-based exclusion screening

The rule-based script `primary_screen.py` (372 lines, pure-Python with `pandas`) loads the de-duplicated bibliographic record set, normalises the text (HTML stripped, units harmonised, lowercased), and applies five sequential filters to title and abstract. Records that pass all five filters survive to the ASReview ranking stage (`../02_asreview_active_learning/`).

## The five filters

| # | Filter | What it does |
|---|---|---|
| 1 | Microplastic-unit requirement | Requires the abstract to contain a numerator (`particles`, `items`, `MP`, `microplastics`, `fibres`, `fragments`, `beads`, `pellets`, `mg`, `ug`, …) paired with a denominator (`L`, `m3`, `km2`, `kg`, `ha`, …) in any of the conventional unit notations (`items / L`, `items per m3`, `items L^-1`, `items m-3`, `items m⁻³`, `particles kg-1`, etc.). |
| 2 | Non-marine focus | Drops records whose abstract contains marine or estuarine context terms (`marine`, `ocean`, `oceanic`, `sea`, `seawater`, `coastal`, `estuary`, `delta`, `lagoon`, `tidal`, `pelagic`, `saltwater`, …). |
| 3 | Biological / toxicology exclusions | Drops records that focus on laboratory toxicology or biochemistry: redox endpoints, heavy metals, vertebrate lab models (fish, mouse, rat, …), and organ/cell endpoints (liver, hepatocyte, enzyme, ATP, DNA, RNA, gene, cell, …). |
| 4 | Literature-type exclusions | Drops review, meta-analysis, commentary, perspective, editorial, opinion, protocol, conceptual framework, guideline, consensus, position-statement, research-proposal, "current status", and "state of the art" records. |
| 5 | Quantitative-evidence requirement | Requires at least one numeric expression (a number, a percentage, an `n =`, a *p*-value, an *R²*) AND at least one statistical / measurement cue within ±20 characters of the number (`mean`, `median`, `SD`, `SE`, `CI`, `correlation`, `regression`, `ANOVA`, `t-test`; or a measurement-action cue: `concentration`, `abundance`, `count`, `density`, `mass`, `load`, `flux`, `mesh`, `LOD`, `LOQ`, `measured`, `sampled`, `analyzed`, `quantified`, …). |

The exact regex categories are reproduced verbatim inside `primary_screen.py` and are used in both filters 3 (biological exclusions) and 5 (statistics + measurement proximity).

## Inputs and outputs

- **Input**: `data/processed/MIP_water_meta.xlsx` (Excel file with columns `title`, `abstract`, `doi`).
- **Output**: `data/processed/MIP_water_meta_screened.xlsx` (same three columns; one row per record that passes all five filters).

`convert_meta.py` and `convert_review_meta.py` are helper scripts to turn raw EndNote / RIS exports into the unified three-column Excel format expected by `primary_screen.py`.

## Run

```bash
pip install pandas openpyxl
python primary_screen.py
```

The script prints the per-filter survival count to stdout. The exact survival numbers from the thesis pipeline (October 2025 search, n = 329 papers retained at the end of Stage 3) are reported in Chapter 2 of the thesis.
