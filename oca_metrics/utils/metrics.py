from typing import Any, Dict, List, Optional
import re

import numpy as np
import pandas as pd


THRESHOLD_KEY_PATTERN = re.compile(r"^C_top(\d+)pct(?:_window_(\d+)y)?$")


def compute_percentiles(citations: List[int], percentiles: List[float]) -> Dict[float, float]:
    """Compute citation thresholds for given percentiles."""
    if not isinstance(citations, (list, tuple)) or not isinstance(percentiles, (list, tuple)):
        raise ValueError("Inputs must be lists.")

    if not citations:
        return {p: 0 for p in percentiles}

    arr = np.array(citations)

    return {p: float(np.percentile(arr, p * 100)) for p in percentiles}


def compute_normalized_impact(journal_mean: float, category_mean: float) -> float:
    """Compute normalized impact (journal mean / category mean), returns 0 if denominator is 0."""
    if not category_mean:
        return 0.0

    return journal_mean / category_mean


def build_threshold_key(pct_val: int, window: Optional[int] = None) -> str:
    if window is None:
        return f"C_top{pct_val}pct"

    return f"C_top{pct_val}pct_window_{window}y"


def extract_threshold_pct_values(thresholds: Dict[str, Any]) -> List[int]:
    pct_values = set()
    for key in thresholds.keys():
        match = THRESHOLD_KEY_PATTERN.match(key)
        if match:
            pct_values.add(int(match.group(1)))

    return sorted(pct_values, reverse=True)


def compute_share_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0)
    den = pd.to_numeric(denominator, errors="coerce")
    return (num / den.replace(0, np.nan) * 100).fillna(0.0)
