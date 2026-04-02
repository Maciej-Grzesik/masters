from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


@dataclass(slots=True)
class StatisticalTestResult:
    method_a: str
    method_b: str
    metric_name: str
    n_folds: int
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float
    paired_t_statistic: float
    paired_t_pvalue: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        stat, pvalue = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(pvalue)
    except ValueError:
        return 0.0, 1.0


def run_paired_tests(
    method_a: str,
    method_b: str,
    metric_name: str,
    losses_a: np.ndarray,
    losses_b: np.ndarray,
) -> StatisticalTestResult:
    if losses_a.shape != losses_b.shape:
        raise ValueError("loss vectors must have the same shape")

    losses_a = np.asarray(losses_a, dtype=np.float64)
    losses_b = np.asarray(losses_b, dtype=np.float64)

    t_stat, t_p = ttest_rel(losses_a, losses_b, alternative="two-sided")
    w_stat, w_p = _safe_wilcoxon(losses_a, losses_b)

    return StatisticalTestResult(
        method_a=method_a,
        method_b=method_b,
        metric_name=metric_name,
        n_folds=int(losses_a.shape[0]),
        mean_a=float(np.mean(losses_a)),
        std_a=float(np.std(losses_a, ddof=1)) if losses_a.shape[0] > 1 else 0.0,
        mean_b=float(np.mean(losses_b)),
        std_b=float(np.std(losses_b, ddof=1)) if losses_b.shape[0] > 1 else 0.0,
        paired_t_statistic=float(t_stat),
        paired_t_pvalue=float(t_p),
        wilcoxon_statistic=w_stat,
        wilcoxon_pvalue=w_p,
    )
