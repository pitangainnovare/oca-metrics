from typing import Any

import unicodedata
import numpy as np
import pandas as pd

from oca_metrics.utils.constants import OPENALEX_URL_PREFIX


def stz_doi(value):
    if not value:
        return ""

    doi = str(value).strip().lower()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "").strip()

    return doi if doi else ""


def stz_title(value):
    if not value:
        return ""

    # Remove accents
    normalized = unicodedata.normalize("NFKD", str(value))
    without_accents = "".join(
        c for c in normalized if not unicodedata.combining(c)
    )

    # Remove spaces, transform to lowercase and strip leading/trailing whitespace
    title = without_accents.replace(" ", "").lower().strip()

    return title if title else ""


def extract_year(value):
    if not value:
        return None

    try:
        year = int(str(value).strip())
        return year
    except ValueError:
        pass

    return None


def stz_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def stz_binary_flag(value: Any) -> int:
    if pd.isna(value):
        return 0

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value != 0)

    txt = str(value).strip().lower()
    if txt in {"1", "true", "t", "yes", "y", "sim", "s"}:
        return 1

    return 0


def stz_openalex_source_id(value: Any, url_prefix: str = OPENALEX_URL_PREFIX) -> Any:
    if pd.isna(value):
        return None

    v = str(value).strip()
    if not v:
        return None

    if v.startswith(url_prefix):
        return v

    if v.startswith("S"):
        return f"{url_prefix}{v}"

    return v


def format_output_header_name(internal_key: str) -> str:
    return internal_key.replace("_", " ")


def shorten_openalex_id(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    v = value.strip()
    if v.startswith(OPENALEX_URL_PREFIX):
        return v[len(OPENALEX_URL_PREFIX):]

    return v
