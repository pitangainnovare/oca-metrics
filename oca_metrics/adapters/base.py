from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
)

import pandas as pd


class BaseAdapter(ABC):
    """Base interface for bibliometric data adapters."""

    @abstractmethod
    def get_yearly_citation_columns(self) -> List[str]:
        """Gets the available yearly citation columns (e.g., citations_2020)."""
        pass

    @abstractmethod
    def get_categories(self, year: int, level: str, category_id: Optional[str] = None) -> List[str]:
        """Gets the list of categories (cohorts) for a given year and level."""
        pass

    @abstractmethod
    def compute_baseline(self, year: int, level: str, cat_id: str, windows: Sequence[int]) -> Optional[pd.Series]:
        """Computes baseline metrics for a category."""
        pass

    @abstractmethod
    def compute_thresholds(self, year: int, level: str, cat_id: str, windows: Sequence[int], target_percentiles: Sequence[int]) -> Dict[str, Any]:
        """Computes citation thresholds for different percentiles."""
        pass

    @abstractmethod
    def compute_journal_metrics(self, year: int, level: str, cat_id: str, windows: Sequence[int], thresholds: Dict[str, Any]) -> pd.DataFrame:
        """Computes metrics per journal within a category."""
        pass
