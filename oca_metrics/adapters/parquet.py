from typing import Any, Dict, List, Optional, Sequence

import duckdb
import logging
import pandas as pd

from oca_metrics.adapters.base import BaseAdapter
from oca_metrics.utils.metrics import (
    build_threshold_key,
    extract_threshold_pct_values,
)
from oca_metrics.utils.parquet import (
    extract_yearly_citation_columns,
    get_valid_level_column,
    is_multilingual_scielo_merge_record,
)


logger = logging.getLogger(__name__)


class ParquetAdapter(BaseAdapter):
    """Adapter for extraction and computation of bibliometric indicators from Parquet data using DuckDB."""

    def __init__(self, parquet_path: str, table_name: str = "metrics"):
        self.con = duckdb.connect(database=':memory:')
        self.table_name = table_name

        try:
            self.con.execute(f"CREATE VIEW {self.table_name} AS SELECT * FROM read_parquet('{parquet_path}', union_by_name=True)")
            self.table_columns = self._get_table_columns()
            self.yearly_citation_cols = extract_yearly_citation_columns(self.table_columns)
        except Exception as e:
            logger.error(f"Failed to load parquet file at {parquet_path}: {e}")
            raise

    def _get_table_columns(self) -> List[str]:
        return [row[0] for row in self.con.execute(f"DESCRIBE {self.table_name}").fetchall()]

    def get_yearly_citation_columns(self) -> List[str]:
        try:
            return list(self.yearly_citation_cols)
        except Exception as e:
            logger.warning(f"Could not infer yearly citation columns: {e}")
            return []

    @staticmethod
    def _build_top_counts_sql(windows: Sequence[int], thresholds: Dict[str, Any]) -> List[str]:
        top_counts_sql = []
        for pct_val in extract_threshold_pct_values(thresholds):
            t_all = thresholds.get(build_threshold_key(pct_val), 0)
            top_counts_sql.append(
                f"SUM(CASE WHEN citations_total >= {t_all} THEN 1 ELSE 0 END) "
                f"as top_{pct_val}pct_all_time_publications_count"
            )

            for w in windows:
                t_w = thresholds.get(build_threshold_key(pct_val, w), 0)
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
            "SUM(COALESCE(citations_total, 0)) as journal_citations_total",
            "AVG(COALESCE(citations_total, 0)) as journal_citations_mean",
        ]
        select_cols.extend([f"SUM(COALESCE(citations_window_{w}y, 0)) as citations_window_{w}y" for w in windows])
        select_cols.extend(
            [f"SUM(CASE WHEN COALESCE(citations_window_{w}y, 0) >= 1 THEN 1 ELSE 0 END) as citations_window_{w}y_works" for w in windows]
        )
        select_cols.extend([f"SUM(COALESCE({c}, 0)) as {c}" for c in self.yearly_citation_cols])
        select_cols.extend([f"AVG(COALESCE(citations_window_{w}y, 0)) as journal_citations_mean_window_{w}y" for w in windows])
        select_cols.extend(top_counts_sql)
        return select_cols

    def _compute_multilingual_flag_by_scielo_merge(self, year: int, level: str, cat_id: str) -> pd.DataFrame:
        required_cols = {"is_merged", "oa_individual_works"}
        if not required_cols.issubset(set(self.table_columns)):
            return pd.DataFrame(columns=["journal_id", "is_journal_multilingual"])

        level_col = get_valid_level_column(level, self.table_columns)

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
            is_multilingual_scielo_merge_record(is_merged, payload)
            for is_merged, payload in zip(df["is_merged"], df["oa_individual_works"])
        ]
        return (
            df.groupby("journal_id", as_index=False)["is_journal_multilingual"]
            .max()
            .astype({"is_journal_multilingual": "int64"})
        )

    def get_categories(self, year: int, level: str, category_id: Optional[str] = None) -> List[str]:
        level_col = get_valid_level_column(level, self.table_columns)
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
        level_col = get_valid_level_column(level, self.table_columns)
        query = f"""
        SELECT 
            COUNT(*) as total_docs,
            SUM(COALESCE(citations_total, 0)) as total_citations,
            AVG(COALESCE(citations_total, 0)) as mean_citations,
            {", ".join([f"SUM(COALESCE(citations_window_{w}y, 0)) as total_citations_window_{w}y" for w in windows])},
            {", ".join([f"AVG(COALESCE(citations_window_{w}y, 0)) as mean_citations_window_{w}y" for w in windows])}
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
        level_col = get_valid_level_column(level, self.table_columns)
        threshold_cols = []
        for p in target_percentiles:
            pct_val = 100 - p
            # top 1% -> p=99 -> quantile_disc(..., 0.99)
            # top 5% -> p=95 -> quantile_disc(..., 0.95)
            # ...
            # top 50% -> p=50 -> quantile_disc(..., 0.50)
            q = p / 100.0
            threshold_cols.append(
                f"quantile_disc(COALESCE(citations_total, 0), {q}) as {build_threshold_key(pct_val)}"
            )

            for w in windows:
                threshold_cols.append(
                    f"quantile_disc(COALESCE(citations_window_{w}y, 0), {q}) as {build_threshold_key(pct_val, w)}"
                )
        
        query = f"SELECT {', '.join(threshold_cols)} FROM {self.table_name} WHERE publication_year = ? AND {level_col} = ?"
        try:
            return self.con.execute(query, [year, cat_id]).df().iloc[0].to_dict()

        except Exception as e:
            logger.error(f"Error computing thresholds for {cat_id} in {year}: {e}")
            return {}

    def compute_journal_metrics(self, year: int, level: str, cat_id: str, windows: Sequence[int], thresholds: Dict[str, Any]) -> pd.DataFrame:
        level_col = get_valid_level_column(level, self.table_columns)
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
