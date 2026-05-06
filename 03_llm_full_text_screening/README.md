# Stage 3 — LLM full-text screening

The records that ASReview ranks above the inclusion threshold in Stage 2 are full-text-screened by the Gemini-3-pro-preview large language model. For each record, the LLM is given the main text and any Supplementary Information of the paper as PDF input and is instructed to return an INCLUDE / EXCLUDE / UNCERTAIN decision plus a one-sentence justification.

Two prompts are provided in `prompts/`, one per workflow:

- **`prompts/primary_screening_prompt.txt`** — the *original-research workflow* prompt (primary studies, 2010–2025). Reproduced verbatim in Box A.1.2 of the thesis appendix.
- **`prompts/review_screening_prompt.txt`** — the *review workflow* prompt (review articles, 2020–2025). Same structure as the primary prompt with one extra criterion: the review article must list its primary-research sources by name so that those primary-research papers can be back-traced into the original-research workflow.

`llm_screen.py` is a thin wrapper around the `google-generativeai` Python SDK. It walks the input record set, finds the corresponding PDF in a local PDF directory (matching by DOI-derived filename stem first, then by a title-based fuzzy fallback), uploads the PDF together with the chosen prompt, and writes the parsed decisions to an output CSV.

## Inputs and outputs

- **Input**: a CSV with columns `title`, `abstract`, `doi` (typically the Stage-2 output, `02_asreview_active_learning/asreview_ranked_records.csv`).
- **PDFs**: a local directory containing the full-text PDFs of every record in the input CSV. PDFs themselves are not redistributed in this repository; they must be downloaded by the user from the original publishers.
- **Output**: a CSV with columns `doi`, `title`, `decision`, `reason`, `pdf_path`.

## Run

```bash
pip install -r ../requirements.txt
export GEMINI_API_KEY="<your-key>"

python llm_screen.py \
    --input ../02_asreview_active_learning/asreview_ranked_records.csv \
    --pdf-dir /path/to/full-text-pdfs \
    --workflow primary \
    --output decisions.csv
```

For the parallel review workflow, switch `--workflow primary` to `--workflow review`.

## Manual UNCERTAIN resolution

Records flagged as UNCERTAIN by the LLM are resolved by manual reading. After this three-stage screening (rule-based → ASReview → LLM full-text), 329 papers remained for the data-mining stage of Step 2 of the thesis (Section 2.2.2 of Chapter 2).
