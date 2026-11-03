"""SVG charts for the README. Static, white background, one axis each."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots in fixed order for models; references are neutral greys.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#e34948"]
ACTIVE = "#3b3b38"
RANDOM = "#9b9a94"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def colours(agents: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    for a in agents:
        if a == "active":
            out[a] = ACTIVE
        elif a == "random":
            out[a] = RANDOM
        else:
            out[a] = SLOTS[i % len(SLOTS)]
            i += 1
    return out


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def solve_curve(
    curves: dict[str, list[float]],
    points: dict[str, dict[int, tuple[float, float, float]]],
    labels: dict[str, str],
    path: Path,
) -> None:
    """Reference solve rate at every budget, model solve rate at the budgets run."""
    agents = list(curves) + [a for a in points if a not in curves]
    col = colours(agents)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for a, ys in curves.items():
        ax.plot(
            range(len(ys)),
            ys,
            color=col[a],
            lw=2,
            ls="-" if a == "active" else "--",
            label=labels[a],
        )
    offsets = {a: (i - (len(points) - 1) / 2) * 0.18 for i, a in enumerate(points)}
    for a, by_budget in points.items():
        xs = sorted(by_budget)
        ys = [by_budget[b][0] for b in xs]
        lo = [by_budget[b][0] - by_budget[b][1] for b in xs]
        hi = [by_budget[b][2] - by_budget[b][0] for b in xs]
        ax.errorbar(
            [x + offsets[a] for x in xs],
            ys,
            yerr=[lo, hi],
            color=col[a],
            fmt="o",
            ms=7,
            lw=1.5,
            capsize=3,
            label=labels[a],
            mec="white",
            mew=1,
        )
        ax.plot([x + offsets[a] for x in xs], ys, color=col[a], lw=1, alpha=0.5)
    ax.set_xlabel("query budget")
    ax.set_ylabel("solve rate (all 10 test inputs right)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.3, max(len(y) for y in curves.values()) - 0.7)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, path)


def grouped_bars(
    groups: list[str],
    series: dict[str, list[tuple[float, float, float]]],
    labels: dict[str, str],
    ylabel: str,
    path: Path,
    *,
    ylim: tuple[float, float] | None = (0, 1.02),
    zero_line: bool = False,
) -> None:
    col = colours(list(series))
    n = len(series)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, (a, vals) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        ys = [v[0] for v in vals]
        err = [[max(0.0, v[0] - v[1]) for v in vals], [max(0.0, v[2] - v[0]) for v in vals]]
        ax.bar(xs, ys, width * 0.88, color=col[a], label=labels[a], edgecolor="white", lw=1)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK2, lw=1, capsize=2)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if zero_line:
        ax.axhline(0, color=INK2, lw=1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5, ncol=4 if n == 4 else 3)
    _save(fig, path)


def hbars(
    rows: list[tuple[str, float, float, float, str]],
    xlabel: str,
    path: Path,
    *,
    reference_line: float | None = None,
    xlim: tuple[float, float] | None = None,
) -> None:
    """One bar per row: (label, value, lo, hi, colour)."""
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(rows) + 1.0))
    ys = list(range(len(rows)))[::-1]
    for y, (_label, v, lo, hi, c) in zip(ys, rows, strict=True):
        ax.barh(y, v, 0.62, color=c, edgecolor="white")
        ax.errorbar(
            v,
            y,
            xerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
            fmt="none",
            ecolor=INK2,
            lw=1,
            capsize=2,
        )
        ax.text(
            max(hi, v) + 0.01 * (xlim[1] if xlim else 1),
            y,
            f"{v:.2f}",
            va="center",
            fontsize=8.5,
            color=INK2,
        )
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    if reference_line is not None:
        ax.axvline(reference_line, color=INK2, lw=1, ls=":")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    _save(fig, path)
