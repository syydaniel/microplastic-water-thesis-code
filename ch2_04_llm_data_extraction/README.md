# Chapter 2 Step 2 — LLM data extraction

The 329 papers retained after the three-stage screening (`ch2_01_*`, `ch2_02_*`, `ch2_03_*`) are mined with the Gemini-3-pro-preview LLM under a single Central Prompt that turns the unstructured text and supplementary tables of each paper into a structured per-site dataset of microplastic abundance (items m⁻³) together with study metadata, sampling methodology, and compositional information.

The Central Prompt is reproduced verbatim in `prompts/central_extraction_prompt.txt` and in Box A.1.3 of the thesis appendix. It instructs the LLM to:

1. Apply five exclusion criteria (review/meta-analysis, marine/estuarine/brackish water, sediment- or biota-only, mass-based-only, modelled-only).
2. Identify all unique sampling sites in the paper (Step A).
3. For each site, extract a JSON object with study metadata, sampling methodology, and the abundance value (Step B).
4. Cross-reference main text and Supplementary Information to fill missing fields, and read the exact panel of any graph that corresponds to the target water body.

The wrapper script `llm_extract.py` walks a directory tree (one sub-directory per paper, each containing the main-text PDF and any Supplementary-Information PDFs), runs the Central Prompt against the Gemini API, parses the JSON output, and writes one row per extracted site to a unified CSV.

## Run

```bash
pip install -r ../requirements.txt
export GEMINI_API_KEY="<your-key>"
python llm_extract.py --paper-dir /path/to/329_retained_papers/ --output records.csv
```

## Output schema

The output CSV has one row per extracted site with the following columns:

| Column | Description |
|---|---|
| `study_id` | Filesystem-safe paper directory name (e.g. `Tang_2021_Characteri`). |
| `first_author`, `publication_year`, `sampling_year` | Study metadata. |
| `site_id`, `country`, `longitude`, `latitude` | Sampling-site location. |
| `gear`, `mesh_size_um`, `separation_chemical`, `sampling_depth_m` | Sampling methodology. |
| `value_items_m3`, `raw_unit_in_paper` | Microplastic abundance (after unit conversion to items m⁻³) plus the verbatim original unit. |
| `shape_composition_pct`, `polymer_composition_pct` | JSON dictionaries of percentage shares per shape / polymer. |

The unified output of this stage feeds Chapter 2 Step 3 (record-level QC + study-level manual validation, Section 2.2.3 of the thesis), and the QC-passed subset (3,148 records from 251 studies) is the training input to Chapter 3 (`ch3_machine_learning/`).
