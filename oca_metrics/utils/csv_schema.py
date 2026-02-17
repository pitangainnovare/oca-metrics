from typing import List, Optional, Sequence

from oca_metrics.utils.constants import CSV_METADATA_COLUMNS


def get_csv_schema_order(
    windows: Sequence[int],
    target_percentiles: Optional[Sequence[int]] = None,
    yearly_citation_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    if target_percentiles is None:
        target_percentiles = [99, 95, 90, 50]
    if yearly_citation_cols is None:
        yearly_citation_cols = []

    cols: List[str] = []
    cols += [
        "category_id",
        "category_level",
        "journal_id",
        "journal_issn",
        "journal_title",
        *CSV_METADATA_COLUMNS,
        "publication_year",
    ]
    cols += ["category_citations_mean"]
    cols += [f"category_citations_mean_window_{w}y" for w in windows]
    cols += ["category_citations_total"]
    cols += [f"category_citations_total_window_{w}y" for w in windows]
    cols += ["category_publications_count"]
    cols += [f"citations_window_{w}y" for w in windows]
    cols += [f"citations_window_{w}y_works" for w in windows]
    cols += list(yearly_citation_cols)
    cols += ["journal_citations_mean"]
    cols += [f"journal_citations_mean_window_{w}y" for w in windows]
    cols += ["journal_citations_total", "journal_impact_normalized"]
    cols += [f"journal_impact_normalized_window_{w}y" for w in windows]
    cols += ["journal_publications_count", "is_journal_multilingual"]

    for p in target_percentiles:
        pct = 100 - p

        cols += [
            f"top_{pct}pct_all_time_citations_threshold",
            f"top_{pct}pct_all_time_publications_count",
            f"top_{pct}pct_all_time_publications_share_pct",
        ]

        for w in windows:
            cols += [
                f"top_{pct}pct_window_{w}y_citations_threshold",
                f"top_{pct}pct_window_{w}y_publications_count",
                f"top_{pct}pct_window_{w}y_publications_share_pct",
            ]

    return cols
