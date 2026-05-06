# download the packages first
    # pip install pandas
    # pip install html
    # pip install re
    # pip install pathlib
    # pip install typing


import html
import re
from pathlib import Path
from typing import List, Pattern

import pandas as pd


EXCLUDE_REGEX = re.compile(
    r"(?i)\b(review|systematic review|literature review|narrative review|scoping review|overview|"
    r"meta[-\s]?analysis|systematic evaluation|evidence synthesis|commentary|comment|perspective|"
    r"editorial|opinion|viewpoint|policy brief|white paper|protocol|study protocol|research design|"
    r"conceptual framework|guideline|consensus|position statement|research proposal|progress|update|"
    r"trend|current status|state of the art)\b"
) # this section means the abstract should not contain any of the following terms

BIOLOGICAL_EXCLUDE_REGEX = re.compile(
    r"(?i)\b("
    r"oxidation|reduction|redox|oxidative|antioxidant|"
    r"heavy metal|metals?|lead|mercury|cadmium|arsenic|chromium|nickel|copper|zinc|"
    r"fish|fishes|zebra fish|danio|tilapia|trout|salmon|carp|"
    r"mouse|mice|rat|rats|murine|rodent|"
    r"liver|hepatic|hepatocyte|"
    r"enzyme|enzymatic|"
    r"atp|adenosine triphosphate|"
    r"dna|rna|genes?|genetic|genomic|"
    r"cell|cells|cellular|cytotoxic"
    r")\b"
) # this section means the abstract should not contain any of the following terms

NUMERIC_REGEX = re.compile(
    r"(?i)(\b\d+(?:\.\d+)?\s*%\b|\b\d+(?:\.\d+)?\s*[×xEe]\s*-?\d+\b|\b\d+(?:\.\d+)?(?:\s*\(\s*\d+(?:\.\d+)?\s*\))?\b|"
    r"\bn\s*=\s*\d+\b|\bN\s*=\s*\d+\b|\bp\s*[<≤]\s*\d+(?:\.\d+)?\b|\bR\s*(?:\^?2|=)\s*\d+(?:\.\d+)?\b)"
) # this section means the abstract should contain a number

STATS_REGEX = re.compile(
    r"(?i)\b(mean|average|median|sd|standard deviation|standard error|se|ci|confidence interval|min|max|range|variance|"
    r"stdev|r2|r\s*=\s*|p\s*[<≤]\s*|slope|intercept|coefficient|significant|correlation|regression|sample size|"
    r"replicates?|duplicates?|triplicates?|anova|t-?test|mann-?whitney|kruskal-?wallis)\b|\bn\s*=\s*\d+\b|\bN\s*=\s*\d+\b|"
    r"\bR\s*(?:\^?2|=)\s*\d+(?:\.\d+)?\b" 
) # this section means the abstract should contain a number

MEASURE_REGEX = re.compile(
    r"(?i)\b(concentration|abundance|count|density|mass|load|flux|emission|removal|recovery|yield|size|diameter|length|"
    r"width|thickness|mesh|limit of detection|lod|loq|detection limit|measured|measuring|collected|sampled|analyzed|"
    r"quantified|reported)\b"
) # this section means the abstract should contain a number

NUMBER_SIMPLE_REGEX = re.compile(r"\d+(?:\.\d+)?")

PROXIMITY_WINDOW = 20

MARINE_TERMS = [
    "marine",
    "ocean",
    "oceanic",
    "sea",
    "seawater",
    "sea water",
    "coastal",
    "coast",
    "coastline",
    "shore",
    "shoreline",
    "estuaries",
    "estuary",
    "estuarine",
    "river mouth", # because later in the Marina model, river mouth is decribe in another section
    "delta",
    "deltaic",
    "bay",
    "bays",
    "fjord",
    "fjords",
    "gulf",
    "gulfs",
    "lagoon",
    "lagoons",
    "harbor",
    "harbour",
    "tidal",
    "pelagic",
    "saltwater",
    "salt water",
] # this section means the abstract should not contain any of the following terms

MARINE_EXCLUDE_REGEX = re.compile(
    r"\b(" + "|".join(sorted(set(MARINE_TERMS), key=len, reverse=True)) + r")\b", re.IGNORECASE
)

NUMERATOR_TOKENS = [
    "item",
    "items",
    "particle",
    "particles",
    "fiber",
    "fibers",
    "fibre",
    "fibres",
    "fragment",
    "fragments",
    "bead",
    "beads",
    "pellet",
    "pellets",
    "microplastic",
    "microplastics",
    "mp",
    "mps",
    "count",
    "counts",
    "abundance",
    "density",
    "concentration",
    "mg",
    "ug",
    "ng",
    "pg",
    "g",
    "kg",
]

DENOMINATOR_BASES = [
    "l",
    "liter",
    "litre",
    "ml",
    "ul",
    "cl",
    "dl",
    "u l",
    "u g",
    "m3",
    "cm3",
    "mm3",
    "km3",
    "m2",
    "cm2",
    "mm2",
    "km2",
    "m",
    "cm",
    "mm",
    "km",
    "kg",
    "g",
    "dw",
    "ww",
    "inds",
    "ind",
    "individual",
    "individuals",
    "organism",
    "organisms",
    "habitat",
    "ha",
]

NUMERATOR_PATTERN = r"(?:{})".format("|".join(sorted(set(NUMERATOR_TOKENS), key=len, reverse=True)))
DENOMINATOR_BASE_PATTERN = r"(?:{})".format("|".join(sorted(set(DENOMINATOR_BASES), key=len, reverse=True)))

DENOMINATOR_PATTERN = rf"{DENOMINATOR_BASE_PATTERN}(?:\s*(?:-|\u2212)?\s*-?\d+)?"

MICRO_UNIT_REGEX = re.compile(
    rf"""
    (?:
        {NUMERATOR_PATTERN}
        \s*/\s*
        (?:\d+\s*)?
        {DENOMINATOR_PATTERN}
    )
    |
    (?:
        {NUMERATOR_PATTERN}
        \s+
        {DENOMINATOR_PATTERN}
        (?=[\s,.;)\/-])
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

SUPERSCRIPT_MAP = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "₋": "-",
    }
)

NORMALIZATION_PATTERNS = [
    (r"μ", "u"),
    (r"µ", "u"),
    (r"\bum\b", "um"),
    (r"\bu\s*g\b", "ug"),
    (r"\bu\s*l\b", "ul"),
    (r"\bu\s*m\b", "um"),
    (r"\bmicrograms?\b", "ug"),
    (r"\bmicrogrammes?\b", "ug"),
    (r"\bmilligrams?\b", "mg"),
    (r"\bmilligrammes?\b", "mg"),
    (r"\bnanograms?\b", "ng"),
    (r"\bpicograms?\b", "pg"),
    (r"\bmicrograms?\b", "ug"),
    (r"\bmicrometers?\b", "um"),
    (r"\bmicrometres?\b", "um"),
    (r"\bsquare\s+kilomet(er|re)s?\b", "km2"),
    (r"\bsquare\s+met(er|re)s?\b", "m2"),
    (r"\bsquare\s+centimet(er|re)s?\b", "cm2"),
    (r"\bsquare\s+millimet(er|re)s?\b", "mm2"),
    (r"\bcubic\s+met(er|re)s?\b", "m3"),
    (r"\bcubic\s+kilomet(er|re)s?\b", "km3"),
    (r"\bcubic\s+centimet(er|re)s?\b", "cm3"),
    (r"\bcubic\s+millimet(er|re)s?\b", "mm3"),
    (r"\bper\b", "/"),
    (r"(?<=\d)\s*×\s*10", "e"),
    (r"−", "-"),
    (r"\s*-\s*1", "-1"),
    (r"\s*-\s*2", "-2"),
    (r"\s*-\s*3", "-3"),
    (r"\s*-\s*4", "-4"),
    (r"\s*-\s*5", "-5"),
    (r"\^", ""),
]


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def normalize(text: str) -> str:
    text = html.unescape(text or "")
    text = strip_html(text)
    text = text.translate(SUPERSCRIPT_MAP)
    for pattern, replacement in NORMALIZATION_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_proximity(text: str, number_matches: List[re.Match], target_pattern: Pattern[str]) -> bool:
    if not number_matches:
        return False

    targets = list(target_pattern.finditer(text))
    if not targets:
        return False

    for num_match in number_matches:
        num_pos = num_match.start()
        for target_match in targets:
            target_pos = target_match.start()
            if abs(num_pos - target_pos) <= PROXIMITY_WINDOW:
                return True
    return False


def contains_micro_units(abstract_text: str) -> bool:
    return bool(MICRO_UNIT_REGEX.search(abstract_text))


def should_keep(title_text: str, abstract_text: str) -> bool:
    if not abstract_text:
        return False

    if not contains_micro_units(abstract_text):
        return False

    combined_text = f"{title_text} {abstract_text}".strip()
    if not combined_text:
        return False

    if BIOLOGICAL_EXCLUDE_REGEX.search(combined_text):
        return False

    if MARINE_EXCLUDE_REGEX.search(combined_text):
        return False

    if EXCLUDE_REGEX.search(combined_text):
        return False

    if not NUMERIC_REGEX.search(combined_text):
        return False

    if STATS_REGEX.search(combined_text) or MEASURE_REGEX.search(combined_text):
        return True

    number_matches = list(NUMBER_SIMPLE_REGEX.finditer(combined_text))
    if has_proximity(combined_text, number_matches, STATS_REGEX) or has_proximity(
        combined_text, number_matches, MEASURE_REGEX
    ):
        return True

    return False


def screen_records(df: pd.DataFrame) -> pd.DataFrame:
    filtered_rows = []

    for _, row in df.iterrows():
        title = str(row.get("title", "") or "")
        abstract = str(row.get("abstract", "") or "")
        normalized_title = normalize(title)
        normalized_abstract = normalize(abstract)

        if should_keep(normalized_title, normalized_abstract):
            filtered_rows.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "doi": str(row.get("doi", "") or ""),
                }
            )

    return pd.DataFrame(filtered_rows, columns=["title", "abstract", "doi"])


def main() -> None:
    input_path = Path("data/processed/MIP_water_meta.xlsx")
    output_path = Path("data/processed/MIP_water_meta_screened.xlsx")

    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError as exc:
            fallback_path = output_path.with_stem(output_path.stem + "_latest")
            print(
                f"Warning: unable to delete existing output {output_path} "
                f"(likely open elsewhere). Writing to {fallback_path} instead."
            )
            output_path = fallback_path

    df = pd.read_excel(input_path, dtype=str)
    screened_df = screen_records(df)
    screened_df.to_excel(output_path, index=False)

    print(
        f"Screened {len(screened_df)} of {len(df)} records. Output saved to {output_path}"
    )


if __name__ == "__main__":
    main()

