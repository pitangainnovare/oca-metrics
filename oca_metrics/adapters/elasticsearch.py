from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
)

import logging
import pandas as pd

from oca_metrics.adapters.base import BaseAdapter


logger = logging.getLogger(__name__)


class ElasticsearchAdapter(BaseAdapter):
    """Adapter for extraction and computation of bibliometric indicators from Elasticsearch."""

    def __init__(self, hosts: List[str], index: str, **kwargs):
        self.index = index
        logger.warning("ElasticsearchAdapter is not fully implemented.")

    def get_yearly_citation_columns(self) -> List[str]:
        return []

    def get_categories(self, year: int, level: str, category_id: Optional[str] = None) -> List[str]:
        return []

    def compute_baseline(self, year: int, level: str, cat_id: str, windows: Sequence[int]) -> Optional[pd.Series]:
        return None

    def compute_thresholds(self, year: int, level: str, cat_id: str, windows: Sequence[int], target_percentiles: Sequence[int]) -> Dict[str, Any]:
        return {}

    def compute_journal_metrics(self, year: int, level: str, cat_id: str, windows: Sequence[int], thresholds: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame()
