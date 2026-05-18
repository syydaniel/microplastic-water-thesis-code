import html
import re
from pathlib import Path

import pandas as pd


ENTRY_PATTERN = re.compile(
    r"@article\s*\{(.*?)}\s*(?=@article|\Z)", flags=re.IGNORECASE | re.DOTALL
)

FIELD_PATTERN = {
    "title": re.compile(r"title\s*=\s*\{(.*?)\}", flags=re.IGNORECASE | re.DOTALL),
    "abstract": re.compile(r"abstract\s*=\s*\{(.*?)\}", flags=re.IGNORECASE | re.DOTALL),
    "doi": re.compile(r"doi\s*=\s*\{(.*?)\}", flags=re.IGNORECASE | re.DOTALL),
}


def extract_fields(entry: str) -> dict:
    # drop entry key (e.g. RN35550,)
    entry_body = entry.split(",", 1)[-1]
    values = {}
    for key, pattern in FIELD_PATTERN.items():
        match = pattern.search(entry_body)
        if match:
            value = match.group(1)
            value = html.unescape(value)
            value = value.replace("\n", " ")
            value = re.sub(r"\s+", " ", value)
            values[key] = value.strip()
        else:
            values[key] = ""
    return values


def parse_bibtex(raw_text: str) -> list:
    records = []

    for match in ENTRY_PATTERN.finditer(raw_text):
        entry = match.group(1)
        fields = extract_fields(entry)
        if any(value for value in fields.values()):
            records.append(fields)

    return records


def main() -> None:
    input_path = Path("data/raw/MIP_water_meta_rev.txt")
    output_path = Path("data/processed/MIP_water_meta_rev.xlsx")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_text = input_path.read_text(encoding="utf-8")
    records = parse_bibtex(raw_text)

    df = pd.DataFrame(records, columns=["title", "abstract", "doi"])
    df.to_excel(output_path, index=False)

    print(f"Extracted {len(df)} records to {output_path}")


if __name__ == "__main__":
    main()

