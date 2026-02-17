import json

import pytest

from oca_metrics.utils.parquet import (
    extract_yearly_citation_columns,
    get_valid_level_column,
    is_multilingual_scielo_merge_record,
    parse_merged_languages,
)


def test_extract_yearly_citation_columns():
    cols = [
        "citations_total",
        "citations_2025",
        "citations_2023",
        "source_id",
        "citations_2024",
    ]
    assert extract_yearly_citation_columns(cols) == ["citations_2023", "citations_2024", "citations_2025"]


def test_get_valid_level_column():
    assert get_valid_level_column("field", ["domain", "field", "topic"]) == "field"


def test_get_valid_level_column_invalid_identifier():
    with pytest.raises(ValueError):
        get_valid_level_column("field-name", ["field-name"])


def test_get_valid_level_column_missing():
    with pytest.raises(ValueError):
        get_valid_level_column("field", ["domain", "topic"])


def test_parse_merged_languages_and_multilingual_flag():
    payload = json.dumps(
        {
            "W1": {"language": "en"},
            "W2": {"language": "pt"},
            "W3": {"language": "en"},
        }
    )
    langs = parse_merged_languages(payload)
    assert langs == {"en", "pt"}
    assert is_multilingual_scielo_merge_record(1, payload) == 1
    assert is_multilingual_scielo_merge_record(0, payload) == 0


def test_parse_merged_languages_invalid_payload():
    assert parse_merged_languages("not-json") == set()
