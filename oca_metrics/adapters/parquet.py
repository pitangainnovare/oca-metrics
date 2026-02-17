from typing import Any, Dict, List, Optional, Sequence

import duckdb
import json
import logging
import pandas as pd
import re

from oca_metrics.adapters.base import BaseAdapter


logger = logging.getLogger(__name__)


class ParquetAdapter(BaseAdapter):
    """Adapter for extraction and computation of bibliometric indicators from Parquet data using DuckDB."""

    def __init__(self, parquet_path: str, table_name: str = "metrics"):
        self.con = duckdb.connect(database=':memory:')
        self.table_name = table_name

        try:
            self.con.execute(f"CREATE VIEW {self.table_name} AS SELECT * FROM read_parquet('{parquet_path}', union_by_name=True)")
            self.table_columns = self._get_table_columns()
            self.yearly_citation_cols = self._extract_yearly_citation_columns(self.table_columns)
        except Exception as e:
            logger.error(f"Failed to load parquet file at {parquet_path}: {e}")
            raise

    def _get_table_columns(self) -> List[str]:
        return [row[0] for row in self.con.execute(f"DESCRIBE {self.table_name}").fetchall()]

    @staticmethod
    def _extract_yearly_citation_columns(columns: Sequence[str]) -> List[str]:
        yearly_cols = [c for c in columns if re.match(r"^citations_\d{4}$", c)]
        return sorted(yearly_cols, key=lambda c: int(c.split("_")[1]))

    def _get_valid_level_column(self, level: str) -> str:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", level):
            raise ValueError(f"Invalid level column name: {level}")

        if level not in self.table_columns:
            raise ValueError(f"Level column not found in parquet schema: {level}")

        return level

    def get_yearly_citation_columns(self) -> List[str]:
        try:
            return list(self.yearly_citation_cols)
        except Exception as e:
            logger.warning(f"Could not infer yearly citation columns: {e}")
            return []

    @staticmethod
    def _parse_merged_languages(payload: Any) -> set:
        if payload is None or (isinstance(payload, float) and pd.isna(payload)):
            return set()

        try:
            data = payload if isinstance(payload, dict) else json.loads(payload)
        except Exception:
            return set()

        if not isinstance(data, dict):
            return set()

        langs = set()
        for v in data.values():
            if isinstance(v, dict):
                lang = v.get("language")
                if lang:
                    langs.add(str(lang).strip().lower())

        return langs

    @classmethod
    def _is_multilingual_scielo_merge_record(cls, is_merged: Any, payload: Any) -> int:
        if not bool(is_merged):
            return 0

        return 1 if len(cls._parse_merged_languages(payload)) > 1 else 0

    @staticmethod
    def _build_top_counts_sql(windows: Sequence[int], thresholds: Dict[str, Any]) -> List[str]:
        top_counts_sql = []
        percentiles = set()
        for key in thresholds.keys():
            if key.startswith("C_top") and "pct" in key:
                parts = key.split("top")[1].split("pct")
                percentiles.add(int(parts[0]))

        for pct_val in sorted(percentiles, reverse=True):
            t_all = thresholds.get(f"C_top{pct_val}pct", 0)
            top_counts_sql.append(
                f"SUM(CASE WHEN citations_total >= {t_all} THEN 1 ELSE 0 END) "
                f"as top_{pct_val}pct_all_time_publications_count"
            )

            for w in windows:
                t_w = thresholds.get(f"C_top{pct_val}pct_window_{w}y", 0)
                top_counts_sql.append(
                    f"SUM(CASE WHEN citations_window_{w}y >= {t_w} THEN 1 ELSE 0 END) "
                    f"as top_{pct_val}pct_window_{w}y_publications_count"
                )

        return top_counts_sql

    def _build_journal_select_columns(self, windows: Sequence[int], top_counts_sql: Sequence[str]) -> List[str]:
        select_cols = [
            "source_id as journal_id",
            "ANY_VALUE(source_issn_l) as journal_issn",
            "COUNT(*) as journal_publications_count",
            "SUM(citations_total) as journal_citations_total",
            "AVG(citations_total) as journal_citations_mean",
        ]
        select_cols.extend([f"SUM(citations_window_{w}y) as citations_window_{w}y" for w in windows])
        select_cols.extend(
            [f"SUM(CASE WHEN citations_window_{w}y >= 1 THEN 1 ELSE 0 END) as citations_window_{w}y_works" for w in windows]
        )
        select_cols.extend([f"SUM({c}) as {c}" for c in self.yearly_citation_cols])
        select_cols.extend([f"AVG(citations_window_{w}y) as journal_citations_mean_window_{w}y" for w in windows])
        select_cols.extend(top_counts_sql)
        return select_cols

    def _compute_multilingual_flag_by_scielo_merge(self, year: int, level: str, cat_id: str) -> pd.DataFrame:
        required_cols = {"is_merged", "oa_individual_works"}
        if not required_cols.issubset(set(self.table_columns)):
            return pd.DataFrame(columns=["journal_id", "is_journal_multilingual"])

        level_col = self._get_valid_level_column(level)

        query = f"""
        SELECT
            source_id as journal_id,
            is_merged,
            oa_individual_works
        FROM {self.table_name}
        WHERE publication_year = ? AND {level_col} = ? AND source_id IS NOT NULL
        """
        try:
            df = self.con.execute(query, [year, cat_id]).df()
        except Exception:
            return pd.DataFrame(columns=["journal_id", "is_journal_multilingual"])

        if df.empty:
            return pd.DataFrame(columns=["journal_id", "is_journal_multilingual"])

        df["is_journal_multilingual"] = [
            self._is_multilingual_scielo_merge_record(is_merged, payload)
            for is_merged, payload in zip(df["is_merged"], df["oa_individual_works"])
        ]
        return (
            df.groupby("journal_id", as_index=False)["is_journal_multilingual"]
            .max()
            .astype({"is_journal_multilingual": "int64"})
        )

    def get_categories(self, year: int, level: str, category_id: Optional[str] = None) -> List[str]:
        level_col = self._get_valid_level_column(level)
        query = f"SELECT DISTINCT {level_col} FROM {self.table_name} WHERE publication_year = ? AND {level_col} IS NOT NULL"
        params: List[Any] = [year]

        if category_id is not None:
            query += f" AND {level_col} = ?"
            params.append(category_id)

        try:
            categories = self.con.execute(query, params).fetchall()
            return [c[0] for c in categories]
        except Exception as e:
            logger.error(f"Error fetching categories: {e}")
            return []

    def compute_baseline(self, year: int, level: str, cat_id: str, windows: Sequence[int]) -> Optional[pd.Series]:
        level_col = self._get_valid_level_column(level)
        query = f"""
        SELECT 
            COUNT(*) as total_docs,
            SUM(citations_total) as total_citations,
            AVG(citations_total) as mean_citations,
            {", ".join([f"SUM(citations_window_{w}y) as total_citations_window_{w}y" for w in windows])},
            {", ".join([f"AVG(citations_window_{w}y) as mean_citations_window_{w}y" for w in windows])}
        FROM {self.table_name}
        WHERE publication_year = ? AND {level_col} = ?
        """
        try:
            res = self.con.execute(query, [year, cat_id]).df()
            if res.empty or res.iloc[0]['total_docs'] == 0:
                return None

            return res.iloc[0]

        except Exception as e:
            logger.error(f"Error computing baseline for {cat_id} in {year}: {e}")
            return None

    def compute_thresholds(self, year: int, level: str, cat_id: str, windows: Sequence[int], target_percentiles: Sequence[int]) -> Dict[str, Any]:
        level_col = self._get_valid_level_column(level)
        threshold_cols = []
        for p in target_percentiles:
            threshold_cols.append(f"CAST(quantile_cont(citations_total, {p/100.0}) AS INT) + 1 as C_top{100-p}pct")

            for w in windows:
                threshold_cols.append(f"CAST(quantile_cont(citations_window_{w}y, {p/100.0}) AS INT) + 1 as C_top{100-p}pct_window_{w}y")
        
        query = f"SELECT {', '.join(threshold_cols)} FROM {self.table_name} WHERE publication_year = ? AND {level_col} = ?"
        try:
            return self.con.execute(query, [year, cat_id]).df().iloc[0].to_dict()

        except Exception as e:
            logger.error(f"Error computing thresholds for {cat_id} in {year}: {e}")
            return {}

    def compute_journal_metrics(self, year: int, level: str, cat_id: str, windows: Sequence[int], thresholds: Dict[str, Any]) -> pd.DataFrame:
        level_col = self._get_valid_level_column(level)
        top_counts_sql = self._build_top_counts_sql(windows, thresholds)
        select_cols = self._build_journal_select_columns(windows, top_counts_sql)

        query = f"""
        SELECT 
            {", ".join(select_cols)}
        FROM {self.table_name}
        WHERE publication_year = ? AND {level_col} = ? AND source_id IS NOT NULL
        GROUP BY source_id
        """
        try:
            df_journals = self.con.execute(query, [year, cat_id]).df()
            if df_journals.empty:
                return df_journals

            df_multilingual = self._compute_multilingual_flag_by_scielo_merge(year, level, cat_id)
            if df_multilingual.empty:
                df_journals["is_journal_multilingual"] = 0
            else:
                df_journals = df_journals.merge(df_multilingual, on="journal_id", how="left")
                df_journals["is_journal_multilingual"] = (
                    pd.to_numeric(df_journals["is_journal_multilingual"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

            return df_journals

        except Exception as e:
            logger.error(f"Error computing journal metrics for {cat_id} in {year}: {e}")
            return pd.DataFrame()
