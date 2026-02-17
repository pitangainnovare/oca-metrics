from typing import Any, List, Sequence, Set

import json
import pandas as pd
import re


YEARLY_CITATIONS_PATTERN = re.compile(r"^citations_\d{4}$")
SQL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def extract_yearly_citation_columns(columns: Sequence[str]) -> List[str]:
    yearly_cols = [c for c in columns if YEARLY_CITATIONS_PATTERN.match(c)]
    return sorted(yearly_cols, key=lambda c: int(c.split("_")[1]))


def get_valid_level_column(level: str, table_columns: Sequence[str]) -> str:
    if not SQL_IDENTIFIER_PATTERN.match(level):
        raise ValueError(f"Invalid level column name: {level}")

    if level not in table_columns:
        raise ValueError(f"Level column not found in parquet schema: {level}")

    return level


def parse_merged_languages(payload: Any) -> Set[str]:
    if payload is None or (isinstance(payload, float) and pd.isna(payload)):
        return set()

    try:
        data = payload if isinstance(payload, dict) else json.loads(payload)
    except Exception:
        return set()

    if not isinstance(data, dict):
        return set()

    langs = set()
    for value in data.values():
        if isinstance(value, dict):
            lang = value.get("language")
            if lang:
                langs.add(str(lang).strip().lower())

    return langs


def is_multilingual_scielo_merge_record(is_merged: Any, payload: Any) -> int:
    if not bool(is_merged):
        return 0

    return 1 if len(parse_merged_languages(payload)) > 1 else 0
