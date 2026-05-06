import html
import re
from pathlib import Path

import pandas as pd


def extract_field(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    value = match.group(1)
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return html.unescape(value.strip())


def main() -> None:
    input_path = Path("data/raw/MIP_water_meta.htm")
    output_path = Path("data/processed/MIP_water_meta.xlsx")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_text = input_path.read_text(encoding="utf-8")
    normalized = raw_text.replace("<p>", "\n")

    entries = re.split(r"@article\{", normalized)
    records = []

    for entry in entries[1:]:
        entry = entry.strip()
        if not entry:
            continue

        title = extract_field(r"title\s*=\s*\{(.*?)\},", entry)
        abstract = extract_field(r"abstract\s*=\s*\{(.*?)\},", entry)
        doi = extract_field(r"doi\s*=\s*\{(.*?)\},", entry)

        if not (title or abstract or doi):
            continue

        records.append(
            {
                "title": title,
                "abstract": abstract,
                "doi": doi,
            }
        )

    df = pd.DataFrame(records, columns=["title", "abstract", "doi"])
    df.to_excel(output_path, index=False)

    print(f"Extracted {len(df)} records to {output_path}")


if __name__ == "__main__":
    main()

