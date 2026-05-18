"""LLM full-text screening wrapper for Stage 3 of the microplastic literature
screening pipeline.

Each row of the input CSV is matched to a PDF in the supplied PDF directory
(by DOI or by filename), the PDF is uploaded to the Gemini-3-pro-preview
model together with the screening prompt, and the model's INCLUDE / EXCLUDE /
UNCERTAIN decision plus one-sentence reason is appended to an output CSV.

Set the GEMINI_API_KEY environment variable before running.

Example:
    export GEMINI_API_KEY="<your-key>"
    python llm_screen.py \
        --input ../02_asreview_active_learning/asreview_ranked_records.csv \
        --pdf-dir /path/to/full-text-pdfs \
        --workflow primary \
        --output decisions.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "google-generativeai not installed. Run: pip install google-generativeai"
    ) from exc


PROMPTS = {
    "primary": Path(__file__).parent / "prompts" / "primary_screening_prompt.txt",
    "review": Path(__file__).parent / "prompts" / "review_screening_prompt.txt",
}

DECISION_RE = re.compile(r"DECISION:\s*(INCLUDE|EXCLUDE|UNCERTAIN)", re.IGNORECASE)
REASON_RE = re.compile(r"REASON:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def doi_to_filename_stem(doi: str) -> str:
    """Map a DOI to a filesystem-safe filename stem (mirrors the local PDF
    organisation convention used in the thesis project)."""
    if not doi:
        return ""
    return re.sub(r"[^a-zA-Z0-9]+", "_", doi.strip()).strip("_").lower()


def find_pdf(pdf_dir: Path, title: str, doi: str) -> Path | None:
    """Locate the PDF for a record under pdf_dir.

    Lookup strategy: first by DOI-derived filename stem; if no match, fall
    back to a simple title-based fuzzy match on the first 60 alphanumeric
    characters of the title.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        return None

    pdfs = list(pdf_dir.rglob("*.pdf"))

    stem = doi_to_filename_stem(doi)
    if stem:
        for pdf in pdfs:
            if stem in pdf.stem.lower():
                return pdf

    title_key = re.sub(r"[^a-zA-Z0-9]+", "", title or "").lower()[:60]
    if title_key:
        for pdf in pdfs:
            pdf_key = re.sub(r"[^a-zA-Z0-9]+", "", pdf.stem).lower()
            if title_key and title_key[:30] in pdf_key:
                return pdf

    return None


def parse_response(text: str) -> tuple[str, str]:
    decision_match = DECISION_RE.search(text or "")
    reason_match = REASON_RE.search(text or "")
    decision = decision_match.group(1).upper() if decision_match else "UNCERTAIN"
    reason = reason_match.group(1).strip() if reason_match else "(no reason returned)"
    return decision, reason


def screen_record(
    model: "genai.GenerativeModel",
    prompt: str,
    pdf_path: Path,
    retries: int = 3,
    sleep_seconds: int = 5,
) -> tuple[str, str]:
    """Send one PDF + prompt to the LLM and parse the decision."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            uploaded = genai.upload_file(path=str(pdf_path))
            response = model.generate_content([prompt, uploaded])
            return parse_response(response.text)
        except Exception as exc:  # noqa: BLE001 — network / API errors
            last_error = exc
            time.sleep(sleep_seconds)
    return "UNCERTAIN", f"(LLM call failed after {retries} attempts: {last_error})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with title, abstract, doi columns.")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing the full-text PDFs.")
    parser.add_argument(
        "--workflow",
        choices=("primary", "review"),
        default="primary",
        help="Which screening prompt to use (primary research or review article).",
    )
    parser.add_argument("--output", required=True, help="Output CSV with decisions.")
    parser.add_argument(
        "--model",
        default="gemini-3-pro-preview",
        help="Gemini model identifier (default: gemini-3-pro-preview).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        return 1

    prompt_path = PROMPTS[args.workflow]
    prompt = prompt_path.read_text(encoding="utf-8")

    df = pd.read_csv(args.input)
    required_cols = {"title", "abstract", "doi"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        print(f"ERROR: input CSV is missing required columns: {missing}", file=sys.stderr)
        return 1
    df.columns = df.columns.str.lower()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    rows_out = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM screening"):
        pdf = find_pdf(Path(args.pdf_dir), row.get("title", ""), row.get("doi", ""))
        if pdf is None:
            decision, reason = "UNCERTAIN", "(PDF not found in --pdf-dir)"
        else:
            decision, reason = screen_record(model, prompt, pdf)
        rows_out.append({
            "doi": row.get("doi", ""),
            "title": row.get("title", ""),
            "decision": decision,
            "reason": reason,
            "pdf_path": str(pdf) if pdf else "",
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doi", "title", "decision", "reason", "pdf_path"])
        writer.writeheader()
        writer.writerows(rows_out)

    n_include = sum(1 for r in rows_out if r["decision"] == "INCLUDE")
    n_exclude = sum(1 for r in rows_out if r["decision"] == "EXCLUDE")
    n_uncertain = sum(1 for r in rows_out if r["decision"] == "UNCERTAIN")
    print(
        f"\nDone. {len(rows_out)} records screened "
        f"(INCLUDE: {n_include}, EXCLUDE: {n_exclude}, UNCERTAIN: {n_uncertain}). "
        f"Output: {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
