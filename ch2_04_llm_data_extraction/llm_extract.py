"""LLM data-extraction wrapper for Chapter 2 Step 2 (Section 2.2.2).

For each paper in the input directory, the wrapper uploads the main-text PDF
and any Supplementary Information PDFs, runs the Central Prompt (see
prompts/central_extraction_prompt.txt) under Gemini-3-pro-preview, parses the
JSON output, and appends each site-level record to the unified output CSV.

Set the GEMINI_API_KEY environment variable before running.

Example:
    export GEMINI_API_KEY="<your-key>"
    python llm_extract.py \
        --paper-dir /path/to/329_retained_papers/ \
        --output records.csv
"""
from __future__ import annotations

import argparse
import csv
import json
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


PROMPT_PATH = Path(__file__).parent / "prompts" / "central_extraction_prompt.txt"

OUTPUT_COLUMNS = [
    "study_id",
    "first_author",
    "publication_year",
    "sampling_year",
    "site_id",
    "country",
    "longitude",
    "latitude",
    "gear",
    "mesh_size_um",
    "separation_chemical",
    "sampling_depth_m",
    "value_items_m3",
    "raw_unit_in_paper",
    "shape_composition_pct",
    "polymer_composition_pct",
]


def find_pdfs_for_paper(paper_dir: Path) -> list[Path]:
    """Return all PDFs in a paper sub-directory (main text + SI files)."""
    return sorted(paper_dir.glob("*.pdf"))


def parse_records(text: str) -> list[dict]:
    """Parse the LLM output. The Central Prompt asks for one JSON object per
    site, separated by newlines, ending with the literal token EOF."""
    records: list[dict] = []
    chunks = re.split(r"\n\s*\n", (text or "").split("EOF")[0])
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Strip ```json fences if present
        chunk = re.sub(r"^```(?:json)?\s*", "", chunk)
        chunk = re.sub(r"\s*```$", "", chunk)
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        records.append(obj)
    return records


def flatten_record(study_id: str, record: dict) -> dict:
    sm = record.get("study_metadata", {}) or {}
    sx = record.get("sampling_methodology", {}) or {}
    ab = record.get("abundance", {}) or {}
    return {
        "study_id": study_id,
        "first_author": sm.get("first_author"),
        "publication_year": sm.get("publication_year"),
        "sampling_year": sm.get("sampling_year"),
        "site_id": sm.get("site_id"),
        "country": sm.get("country"),
        "longitude": sm.get("longitude"),
        "latitude": sm.get("latitude"),
        "gear": sx.get("gear"),
        "mesh_size_um": sx.get("mesh_size_um"),
        "separation_chemical": sx.get("separation_chemical"),
        "sampling_depth_m": sx.get("sampling_depth_m"),
        "value_items_m3": ab.get("value_items_m3"),
        "raw_unit_in_paper": ab.get("raw_unit_in_paper"),
        "shape_composition_pct": json.dumps(ab.get("shape_composition_pct") or {}, ensure_ascii=False),
        "polymer_composition_pct": json.dumps(ab.get("polymer_composition_pct") or {}, ensure_ascii=False),
    }


def extract_one_paper(model, prompt: str, paper_dir: Path, retries: int = 3) -> tuple[str, list[dict]]:
    pdfs = find_pdfs_for_paper(paper_dir)
    if not pdfs:
        return "EXCLUDED: no PDFs found", []

    last_error: Exception | None = None
    for _ in range(retries):
        try:
            uploaded = [genai.upload_file(path=str(p)) for p in pdfs]
            response = model.generate_content([prompt, *uploaded])
            text = response.text or ""
            if "EXCLUDED" in text.split("\n")[0].upper():
                return text.split("\n")[0], []
            return "OK", parse_records(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(5)
    return f"ERROR: {last_error}", []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", required=True, help="Top-level directory; one sub-directory per paper, each with main text + SI PDFs.")
    parser.add_argument("--output", required=True, help="Output CSV (one row per extracted site).")
    parser.add_argument("--model", default="gemini-3-pro-preview")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        return 1

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    paper_root = Path(args.paper_dir)
    paper_dirs = sorted(p for p in paper_root.iterdir() if p.is_dir())
    if not paper_dirs:
        print(f"ERROR: no paper sub-directories found under {paper_root}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_excluded = 0
    n_records = 0
    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for paper_dir in tqdm(paper_dirs, desc="LLM extraction"):
            study_id = paper_dir.name
            status, records = extract_one_paper(model, prompt, paper_dir)
            if not records:
                n_excluded += 1
                continue
            for rec in records:
                writer.writerow(flatten_record(study_id, rec))
                n_records += 1
    print(f"\nDone. {len(paper_dirs)} papers processed: {n_records} records extracted, {n_excluded} papers excluded. Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
