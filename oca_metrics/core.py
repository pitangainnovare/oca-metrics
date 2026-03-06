from typing import (
    Optional,
    Sequence,
)

import logging
import pandas as pd

from oca_metrics.adapters.base import BaseAdapter
from oca_metrics.utils.constants import (
    METADATA_FLAG_COLUMNS,
    METADATA_TEXT_COLUMNS,
)
from oca_metrics.utils.metrics import (
    DEFAULT_IMPACT_MIN_PUBS_ABS,
    DEFAULT_IMPACT_MIN_PUBS_MEDIAN_MULTIPLIER,
    build_threshold_key,
    compute_share_pct,
    compute_cohort_impact,
    compute_impact_comparability_reference,
    compute_impact_is_comparable,
    resolve_impact_min_pubs_category_share,
)


logger = logging.getLogger(__name__)


TARGET_CITATION_PERCENTILES = [99, 95, 90, 50]


class MetricsEngine:
    """Metrics computation engine that uses a data adapter."""

    def __init__(
        self,
        adapter: BaseAdapter,
        target_percentiles: Sequence[int] = None,
        impact_min_pubs_abs: int = DEFAULT_IMPACT_MIN_PUBS_ABS,
        impact_min_pubs_category_share: Optional[float] = None,
        impact_min_pubs_median_multiplier: float = DEFAULT_IMPACT_MIN_PUBS_MEDIAN_MULTIPLIER,
    ):
        self.adapter = adapter
        self.target_percentiles = target_percentiles or TARGET_CITATION_PERCENTILES
        self.impact_min_pubs_abs = impact_min_pubs_abs
        self.impact_min_pubs_category_share = impact_min_pubs_category_share
        self.impact_min_pubs_median_multiplier = impact_min_pubs_median_multiplier

    def process_category(self, year: int, level: str, cat_id: str, windows: Sequence[int], df_meta: pd.DataFrame = None) -> Optional[pd.DataFrame]:
        """Processes a single category and returns enriched metrics per journal."""
        baseline_res = self.adapter.compute_baseline(year, level, cat_id, windows)
        if baseline_res is None:
            return None
        
        thresholds = self.adapter.compute_thresholds(year, level, cat_id, windows, self.target_percentiles)
        if not thresholds:
            return None
            
        df_journals = self.adapter.compute_journal_metrics(year, level, cat_id, windows, thresholds)
        if df_journals.empty:
            return None

        # Add category and year information
        df_journals['category_id'] = cat_id
        df_journals['category_level'] = level
        df_journals['publication_year'] = year
        df_journals['category_publications_count'] = baseline_res['total_docs']
        df_journals['category_citations_total'] = baseline_res['total_citations']
        df_journals['category_citations_mean'] = baseline_res['mean_citations']
        category_share = (
            self.impact_min_pubs_category_share
            if self.impact_min_pubs_category_share is not None
            else resolve_impact_min_pubs_category_share(level)
        )
        comparability_ref = compute_impact_comparability_reference(
            category_publications_count=baseline_res['total_docs'],
            publication_counts=df_journals['journal_publications_count'],
            min_publications_abs=self.impact_min_pubs_abs,
            category_share=category_share,
            median_multiplier=self.impact_min_pubs_median_multiplier,
        )
        min_required = int(comparability_ref['cohort_impact_min_pubs_required'])
        df_journals['cohort_impact_min_pubs_required'] = min_required
        df_journals['cohort_journal_publications_median'] = comparability_ref['cohort_journal_publications_median']
        df_journals['cohort_impact_min_pubs_category_share'] = comparability_ref['cohort_impact_min_pubs_category_share']
        df_journals['cohort_impact_min_pubs_median_multiplier'] = comparability_ref['cohort_impact_min_pubs_median_multiplier']
        df_journals['cohort_impact_is_comparable'] = compute_impact_is_comparable(
            df_journals['journal_publications_count'],
            min_required=min_required,
        )
        df_journals['journal_impact_cohort'] = df_journals['journal_citations_mean'].apply(
            lambda x: compute_cohort_impact(x, baseline_res['mean_citations'])
        )
        
        for w in windows:
            df_journals[f'category_citations_total_window_{w}y'] = baseline_res[f'total_citations_window_{w}y']
            df_journals[f'category_citations_mean_window_{w}y'] = baseline_res[f'mean_citations_window_{w}y']
            df_journals[f'journal_impact_cohort_window_{w}y'] = df_journals[f'journal_citations_mean_window_{w}y'].apply(
                lambda x: compute_cohort_impact(x, baseline_res[f'mean_citations_window_{w}y'])
            )
            df_journals[f'cohort_impact_window_{w}y_is_comparable'] = df_journals['cohort_impact_is_comparable']
            
        for p in self.target_percentiles:
            pct_val = 100 - p

            df_journals[f'top_{pct_val}pct_all_time_citations_threshold'] = thresholds.get(build_threshold_key(pct_val), 0)
            df_journals[f'top_{pct_val}pct_all_time_publications_share_pct'] = compute_share_pct(
                df_journals[f'top_{pct_val}pct_all_time_publications_count'],
                df_journals['journal_publications_count'],
            )

            for w in windows:
                df_journals[f'top_{pct_val}pct_window_{w}y_citations_threshold'] = thresholds.get(
                    build_threshold_key(pct_val, w),
                    0,
                )
                df_journals[f'top_{pct_val}pct_window_{w}y_publications_share_pct'] = compute_share_pct(
                    df_journals[f'top_{pct_val}pct_window_{w}y_publications_count'],
                    df_journals['journal_publications_count'],
                )

        if df_meta is not None and not df_meta.empty:
            meta_cols = ['journal_id', 'publication_year'] + METADATA_TEXT_COLUMNS + METADATA_FLAG_COLUMNS
            available_meta_cols = [c for c in meta_cols if c in df_meta.columns]

            if 'journal_id' not in available_meta_cols or 'publication_year' not in available_meta_cols:
                logger.warning(
                    "Metadata is missing required matching columns journal_id/publication_year. "
                    "Skipping metadata merge for this batch."
                )
                available_meta_cols = []

        else:
            available_meta_cols = []

        if available_meta_cols:
            df_journals = pd.merge(
                df_journals,
                df_meta[available_meta_cols],
                on=['journal_id', 'publication_year'],
                how='left',
                suffixes=('', '_meta'),
            )

        def _series_or_default(col_name: str, default_value):
            if col_name in df_journals.columns:
                return df_journals[col_name]

            return pd.Series(default_value, index=df_journals.index)

        df_journals['journal_title'] = (
            _series_or_default('journal_title', None)
            .replace("", pd.NA)
            .fillna(df_journals['journal_id'])
        )

        for col in METADATA_TEXT_COLUMNS:
            if col == 'journal_title':
                continue

            df_journals[col] = _series_or_default(col, "").fillna("")

        for col in METADATA_FLAG_COLUMNS:
            df_journals[col] = pd.to_numeric(_series_or_default(col, 0), errors='coerce').fillna(0).astype(int)

        has_scielo_collection = (
            (df_journals['is_scielo'] == 1)
            & (df_journals['scielo_active_valid'] == 1)
        )
        df_journals['scielo_collection'] = df_journals['scielo_collection_acronym'].where(has_scielo_collection, "")
        df_journals['is_journal_oa'] = pd.to_numeric(
            _series_or_default('is_journal_oa', 0), errors='coerce'
        ).fillna(0).astype(int)
        df_journals['is_journal_multilingual'] = pd.to_numeric(
            _series_or_default('is_journal_multilingual', 0), errors='coerce'
        ).fillna(0).astype(int)
            
        return df_journals
