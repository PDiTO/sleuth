"""Intervals: Wilson for plain proportions, a task-cluster bootstrap for everything else.

Episodes on the same task are not independent (the same rule, often at two budgets), so
the bootstrap resamples tasks, never single episodes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")

Z95 = 1.959963984540054


def wilson(k: float, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap(
    items: Sequence[T],
    cluster: Callable[[T], str],
    stat: Callable[[list[T]], float],
    *,
    draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile 95% interval, resampling whole clusters."""
    if not items:
        return (math.nan, math.nan, math.nan)
    groups: dict[str, list[T]] = {}
    for it in items:
        groups.setdefault(cluster(it), []).append(it)
    keys = sorted(groups)
    point = stat(list(items))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        pick = rng.integers(0, len(keys), size=len(keys))
        sample = [it for i in pick for it in groups[keys[i]]]
        v = stat(sample)
        if not math.isnan(v):
            values.append(v)
    if not values:
        return (point, math.nan, math.nan)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return (point, float(lo), float(hi))


def ratio(num: Callable[[T], float], den: Callable[[T], float]) -> Callable[[list[T]], float]:
    def f(items: list[T]) -> float:
        d = sum(den(i) for i in items)
        return sum(num(i) for i in items) / d if d else math.nan

    return f


def mean(value: Callable[[T], float]) -> Callable[[list[T]], float]:
    def f(items: list[T]) -> float:
        return sum(value(i) for i in items) / len(items) if items else math.nan

    return f
