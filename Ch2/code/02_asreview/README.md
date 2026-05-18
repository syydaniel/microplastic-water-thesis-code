# Stage 2 — ASReview active-learning ranking

The records that pass all five filters in Stage 1 are imported into [ASReview](https://asreview.nl) (van de Schoot et al., 2021), an open-source active-learning tool for systematic reviews. ASReview iteratively ranks the candidate records by predicted relevance:

1. The reviewer manually labels a small seed set (about 50 records labelled *relevant* or *not relevant*).
2. The default classifier (naïve-Bayes-with-TF-IDF) trains on the seed set and re-ranks the remaining records by predicted relevance.
3. The reviewer labels the top-ranked records first; the classifier retrains after every label and re-ranks the rest.
4. The procedure stops when a sufficient share of the priority queue has been labelled non-relevant in a row, following the ASReview default heuristic.

The records ranked above the inclusion threshold at the stopping point form the input to Stage 3 (LLM full-text screening, see `../03_llm_full_text_screening/`).

## How to run

ASReview itself is run interactively in its web UI; this repository does not wrap it. Follow the official quickstart:

> https://asreview.readthedocs.io/en/latest/start.html

The settings used in the thesis pipeline were:

- **Classifier**: Naïve Bayes (default in ASReview).
- **Feature extractor**: TF-IDF (default in ASReview).
- **Query strategy**: Mixed (max + random), default in ASReview.
- **Seed set**: ~50 manually labelled records (about 25 relevant + 25 non-relevant).
- **Stopping rule**: ASReview default — stop after 50 consecutive non-relevant labels at the top of the queue.

## Inputs and outputs

- **Input**: the Stage-1 output `MIP_water_meta_screened.xlsx` converted to ASReview-compatible RIS or CSV.
- **Output**: a CSV export of the ranked records with the reviewer's INCLUDE / EXCLUDE label for the top-ranked portion.

This repository contains the reference output `asreview_ranked_records.csv` produced for the thesis pipeline. It carries the bibliographic metadata of the records that ASReview ranked above the inclusion threshold; these records were then forwarded to the LLM full-text screening stage.
