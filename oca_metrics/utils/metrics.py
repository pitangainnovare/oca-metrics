from typing import Dict, List

import numpy as np


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
