# Microplastic literature screening pipeline

A three-stage screening pipeline that turns a raw, de-duplicated record set of microplastic literature into a curated set of primary studies ready for full-text data mining. This repository is a companion to the MSc thesis *"Developing a hybrid framework for estimating microplastic retention rates in global river systems"* (Yiyang Shen, Wageningen University, 2026); the pipeline here produces the input to Step 2 of Chapter 2 of that thesis.

The three stages run in sequence:

1. **`01_rule_based_screening/`** — a rule-based exclusion script (`primary_screen.py`) that filters records on title and abstract using fixed regex categories. Records that pass all five filters survive to Stage 2.
2. **`02_asreview_active_learning/`** — an [ASReview](https://asreview.nl) (van de Schoot et al., 2021) active-learning ranking that orders the surviving records by predicted relevance. The reviewer labels the top-ranked records first; the underlying classifier (default naïve-Bayes-with-TF-IDF) retrains after every label and re-ranks the rest. The records ranked above the inclusion threshold form the input to Stage 3.
3. **`03_llm_full_text_screening/`** — a Gemini-3-pro-preview LLM full-text screening that returns an `INCLUDE / EXCLUDE / UNCERTAIN` decision per paper plus a one-sentence justification. Two prompts are provided in `prompts/`: one for the original-research workflow (primary studies, 2010–2025) and one for the parallel review workflow (review articles, 2020–2025).

## Quick start

```bash
# Clone and install dependencies
git clone https://github.com/syydaniel/microplastic-screening-pipeline.git
cd microplastic-screening-pipeline
pip install -r requirements.txt

# Stage 1 — rule-based screening
cd 01_rule_based_screening
# Place the de-duplicated record set at data/processed/MIP_water_meta.xlsx (columns: title, abstract, doi)
python primary_screen.py
# Output: data/processed/MIP_water_meta_screened.xlsx

# Stage 2 — ASReview ranking (interactive)
# Follow https://asreview.readthedocs.io/ to import the Stage-1 output, label ~50 seed records,
# and run the active-learning loop. Export the ranked records as CSV when finished.

# Stage 3 — LLM full-text screening
cd ../03_llm_full_text_screening
export GEMINI_API_KEY="<your-key>"
python llm_screen.py --input ../02_asreview_active_learning/asreview_ranked_records.csv \
                     --pdf-dir /path/to/full-text-pdfs/ \
                     --workflow primary \
                     --output decisions.csv
```

## Repository layout

```
microplastic-screening-pipeline/
├── README.md                              # this file
├── LICENSE                                # MIT licence
├── requirements.txt                       # Python dependencies
├── 01_rule_based_screening/
│   ├── README.md                          # stage 1 details
│   ├── primary_screen.py                  # the five-filter rule-based script
│   ├── convert_meta.py                    # helper: parse EndNote/RIS exports → unified CSV
│   └── convert_review_meta.py             # same helper, tuned for review-workflow input
├── 02_asreview_active_learning/
│   ├── README.md                          # stage 2 details and ASReview run protocol
│   └── asreview_ranked_records.csv        # example output (ranked records used in this thesis)
└── 03_llm_full_text_screening/
    ├── README.md                          # stage 3 details
    ├── llm_screen.py                      # wrapper around the Gemini API
    ├── prompts/
    │   ├── primary_screening_prompt.txt   # original-research workflow prompt (Box A.1.2)
    │   └── review_screening_prompt.txt    # review workflow prompt
    └── examples/
        └── decisions_example.csv          # example output schema
```

## Citation

If you use this pipeline, please cite both the thesis and the underlying tools (ASReview and the Gemini-3-pro-preview model):

> Shen, Y. (2026). *Developing a hybrid framework for estimating microplastic retention rates in global river systems* [MSc thesis, Wageningen University & Research].
>
> van de Schoot, R., et al. (2021). An open source machine learning framework for efficient and transparent systematic reviews. *Nature Machine Intelligence*, *3*(2), 125–131. https://doi.org/10.1038/s42256-020-00287-7

## Licence

This repository is released under the [MIT licence](LICENSE).

The example data file `02_asreview_active_learning/asreview_ranked_records.csv` contains bibliographic metadata (title, abstract, DOI) of records exported from Web of Science, Scopus, and PubMed; this metadata is freely usable for research purposes. No full-text PDFs are distributed in this repository.

## Reproducibility note

This repository contains the **scripts and prompts** used to produce the screening outcomes reported in Chapter 2 of the thesis. To reproduce the exact numbers (n = 329 papers retained for the data-mining stage in Step 2 of the thesis), the same input record set, the same ASReview seed labels, and the same Gemini model snapshot would be required; reproducing exactly is therefore expected to be approximate, but the pipeline itself is deterministic given fixed inputs.
