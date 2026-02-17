from typing import Sequence

import logging
import os
import pandas as pd

from oca_metrics.utils.constants import (
    METADATA_FLAG_COLUMNS,
    METADATA_TEXT_COLUMNS,
    XLSX_TO_INTERNAL_COLUMN_MAP,
)
from oca_metrics.utils.normalization import (
    stz_binary_flag,
    stz_openalex_source_id,
    stz_text,
)


logger = logging.getLogger(__name__)


def _resolve_metadata_duplicates(df: pd.DataFrame, key_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    dup_mask = df.duplicated(subset=list(key_cols), keep=False)
    if not dup_mask.any():
        return df

    df_unique = df.loc[~dup_mask].copy()
    df_dups = df.loc[dup_mask].copy()

    resolved_rows = []
    conflicting_groups = []

    for key_values, group in df_dups.groupby(list(key_cols), sort=False):
        stable = group[value_cols].nunique(dropna=False).max() <= 1
        if stable:
            resolved_rows.append(group.iloc[0].to_dict())
        else:
            conflicting_groups.append((*key_values, len(group)))

    if resolved_rows:
        df_resolved = pd.DataFrame(resolved_rows)
        df_out = pd.concat([df_unique, df_resolved], ignore_index=True)
    else:
        df_out = df_unique

    duplicate_rows_total = len(df_dups)
    duplicate_rows_extra = int(df.duplicated(subset=list(key_cols)).sum())
    duplicate_pairs = df_dups.drop_duplicates(subset=list(key_cols)).shape[0]
    conflicting_rows = sum(g[2] for g in conflicting_groups)
    logger.warning(
        "Found %s duplicated source_id + publication_year rows in metadata "
        "(%s total duplicated rows across %s pairs): kept %s stable pairs and dropped %s conflicting pairs (%s rows).",
        duplicate_rows_extra,
        duplicate_rows_total,
        duplicate_pairs,
        len(resolved_rows),
        len(conflicting_groups),
        conflicting_rows,
    )

    if conflicting_groups:
        logger.warning(
            "Sample conflicting source_id + publication_year pairs (source_id, year, rows): %s",
            conflicting_groups[:5],
        )

    return df_out


def load_global_metadata(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning(f"Global metadata file not found at {path}")
        return pd.DataFrame()

    logger.info(f"Loading global metadata from {path}...")
    try:
        xlsx_cols = list(XLSX_TO_INTERNAL_COLUMN_MAP.keys())
        df = pd.read_excel(path, usecols=xlsx_cols)
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns=XLSX_TO_INTERNAL_COLUMN_MAP)
        df["source_id"] = df["openalex_id"].apply(stz_openalex_source_id)
        df = df.drop(columns=["openalex_id"])

        df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")

        for col in METADATA_TEXT_COLUMNS:
            df[col] = df[col].apply(stz_text)

        for col in METADATA_FLAG_COLUMNS:
            df[col] = df[col].apply(stz_binary_flag)

        df = df[df["source_id"].notna()].copy()
        df = df[df["publication_year"].notna()].copy()
        df["publication_year"] = df["publication_year"].astype(int)

        ordered_cols = ["source_id", "publication_year"] + METADATA_TEXT_COLUMNS + METADATA_FLAG_COLUMNS
        df = _resolve_metadata_duplicates(
            df[ordered_cols].copy(),
            key_cols=["source_id", "publication_year"],
            value_cols=METADATA_TEXT_COLUMNS + METADATA_FLAG_COLUMNS,
        )
        return df[ordered_cols]

    except Exception as e:
        logger.error(f"Error loading global metadata: {e}")
        return pd.DataFrame()
