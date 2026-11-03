"""`sleuth report`: rebuild every number and chart from the recordings, offline.

It replays each recorded episode through the environment (no model is called; a missing
reply is an error), runs the reference agents on the same tasks, and writes
`results/summary.json`, `results/episodes.json`, `results/report.md` and the charts in
`docs/`.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import plots, reference, runner, stats
from . import tasks as tasklib
from .analysis import InfoCache, Row, model_row, reference_rows

CURVE_MAX = 16
FAMILY_NAMES = {
    "int1": "f(x)",
    "int2": "f(x, y)",
    "triple": "triples",
    "string": "strings",
}


def label(agent: str) -> str:
    if agent == "active":
        return "active learner (max EIG)"
    if agent == "random":
        return "random queries + search"
    name = agent.split("/", 1)[-1].split("@", 1)[0]
    m = re.fullmatch(r"claude-(\w+)-(\d+)-(\d+)", name)
    if m:
        return f"{m.group(1).capitalize()} {m.group(2)}.{m.group(3)}"
    m = re.fullmatch(r"gpt-([\d.]+)-(\w+)", name)
    if m:
        return f"GPT-{m.group(1)} {m.group(2).capitalize()}"
    return name


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _ci(rows: list[Row], stat: Callable[[list[Row]], float], seed: int = 0) -> list[float]:
    point, lo, hi = stats.cluster_bootstrap(rows, lambda r: r.task, stat, seed=seed)
    return [round(point, 4), round(lo, 4), round(hi, 4)]


def _solve(rows: list[Row]) -> list[float]:
    """Solve rate with a Wilson interval when episodes are independent, else bootstrap."""
    k = sum(r.solved for r in rows)
    n = len(rows)
    if n == 0:
        return [math.nan, math.nan, math.nan]
    if len({r.task for r in rows}) == n:
        lo, hi = stats.wilson(k, n)
        return [round(k / n, 4), round(lo, 4), round(hi, 4)]
    return _ci(rows, stats.mean(lambda r: float(r.solved)))


def _efficiency(rows: list[Row]) -> float:
    vals = [e / b for r in rows for e, b in zip(r.eig, r.best, strict=True) if b > 1e-9]
    return sum(vals) / len(vals) if vals else math.nan


def _random_efficiency(rows: list[Row]) -> float:
    vals = [m / b for r in rows for m, b in zip(r.rand, r.best, strict=True) if b > 1e-9]
    return sum(vals) / len(vals) if vals else math.nan


def _edge(rows: list[Row]) -> float:
    """How much better than a random probe input from the same state, per query, as a
    share of the best query's EIG. Zero means no better than picking at random."""
    vals = [
        (e - m) / b for r in rows for e, m, b in zip(r.eig, r.rand, r.best, strict=True) if b > 1e-9
    ]
    return sum(vals) / len(vals) if vals else math.nan


def _zero_info(rows: list[Row]) -> float:
    live = [e for r in rows for e, b in zip(r.eig, r.best, strict=True) if b > 1e-9]
    return sum(e < 1e-9 for e in live) / len(live) if live else math.nan


def _per_query(attr: str) -> Callable[[list[Row]], float]:
    def f(rows: list[Row]) -> float:
        vals = [v for r in rows for v in getattr(r, attr)]
        return sum(vals) / len(vals) if vals else math.nan

    return f


def _share(num: Callable[[Row], float]) -> Callable[[list[Row]], float]:
    return stats.ratio(num, lambda r: float(r.queries))


def summarise(rows: list[Row], agents: list[str], budgets: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"agents": {}, "budgets": budgets}
    for a in agents:
        mine = [r for r in rows if r.agent == a]
        per_budget: dict[str, Any] = {}
        for b in budgets:
            rb = [r for r in mine if r.budget == b]
            if not rb:
                continue
            fams = sorted({r.family for r in rb})
            triple = [r for r in rb if r.family == "triple"]
            per_budget[str(b)] = {
                "episodes": len(rb),
                "solve_rate": _solve(rb),
                "accuracy": _ci(rb, stats.mean(lambda r: r.accuracy)),
                "queries": _ci(rb, stats.mean(lambda r: float(r.queries))),
                "stopped_early": round(sum(r.committed for r in rb) / len(rb), 4),
                "calls": sum(r.calls for r in rb),
                "invalid": sum(r.invalid for r in rb),
                "invalid_rate": round(
                    sum(r.invalid for r in rb) / max(1, sum(r.calls for r in rb)), 4
                ),
                "forfeited": sum(r.forfeited for r in rb),
                "tool_calls": sum(r.tool_calls for r in rb),
                "surprise_rate": _ci(rb, _share(lambda r: float(r.surprises))),
                "efficiency": _ci(rb, _efficiency),
                "random_efficiency": _ci(rb, _random_efficiency) if rb[0].rand else None,
                "edge_over_random": _ci(rb, _edge) if rb[0].rand else None,
                "zero_info": _ci(rb, _zero_info),
                "eig_per_query": _ci(rb, _per_query("eig")),
                "bits_per_query": _ci(rb, _per_query("bits")),
                "by_tier": {
                    str(t): {
                        "solve_rate": _solve([r for r in rb if r.tier == t]),
                        "accuracy": _ci(
                            [r for r in rb if r.tier == t], stats.mean(lambda r: r.accuracy)
                        ),
                        "queries": _ci(
                            [r for r in rb if r.tier == t],
                            stats.mean(lambda r: float(r.queries)),
                        ),
                    }
                    for t in (1, 2, 3)
                },
                "by_family": {
                    f: {
                        "solve_rate": _solve([r for r in rb if r.family == f]),
                        "accuracy": _ci(
                            [r for r in rb if r.family == f], stats.mean(lambda r: r.accuracy)
                        ),
                        "efficiency": _ci([r for r in rb if r.family == f], _efficiency),
                        "zero_info": _ci([r for r in rb if r.family == f], _zero_info),
                        "eig_per_query": _ci([r for r in rb if r.family == f], _per_query("eig")),
                    }
                    for f in fams
                },
                "positive_test_rate": _ci(triple, _share(lambda r: float(r.positive)))
                if triple
                else None,
            }
        out["agents"][a] = {"label": label(a), "by_budget": per_budget}
    return out


def build(
    *,
    tasks_path: Path,
    recordings: Path,
    out: Path,
    docs: Path,
    random_seeds: int = 10,
) -> dict[str, Any]:
    all_tasks = tasklib.load(tasks_path)
    world = runner.World(all_tasks)
    _log(f"replaying recordings in {recordings}")
    episodes = runner.replay(world, recordings)
    run_tasks = sorted({e.task_id for e in episodes})
    tasks = [world.by_id[t] for t in run_tasks] if run_tasks else all_tasks
    budgets = sorted({e.budget for e in episodes}) or [4, 12]
    _log(f"{len(episodes)} episodes on {len(tasks)} tasks at budgets {budgets}")

    caches: dict[str, InfoCache] = {}
    rows: list[Row] = []
    for i, ep in enumerate(sorted(episodes, key=lambda e: (e.task_id, e.provider, e.budget))):
        task = world.by_id[ep.task_id]
        cache = caches.setdefault(task.family, InfoCache(world.space(task.family)))
        rows.append(model_row(ep, task, cache))
        if (i + 1) % 50 == 0:
            _log(f"  information for {i + 1}/{len(episodes)} episodes")

    curves: dict[str, list[float]] = {
        "active": [0.0] * (CURVE_MAX + 1),
        "random": [0.0] * (CURVE_MAX + 1),
    }
    curve_weights = {"active": 0, "random": 0}
    ref_budgets = sorted(set(budgets))
    for j, task in enumerate(tasks):
        sp = world.space(task.family)
        runs = [("active", task.seed)] + [
            ("random", task.seed + 1 + s) for s in range(random_seeds)
        ]
        for agent, seed in runs:
            run = reference.run(
                sp,
                task.rule,
                task.test_inputs,
                task.test_outputs,
                agent=agent,
                budget=CURVE_MAX,
                seed=seed,
                given=task.given,
            )
            for k, acc in enumerate(run.accuracy):
                curves[agent][k] += float(acc == 1.0)
            curve_weights[agent] += 1
            for r in reference_rows(sp, task, run, ref_budgets, seed):
                rows.append(r)
        if (j + 1) % 10 == 0:
            _log(f"  reference agents on {j + 1}/{len(tasks)} tasks")
    for a, ys in curves.items():
        curves[a] = [round(v / max(1, curve_weights[a]), 4) for v in ys]

    models = sorted({e.provider for e in episodes}, key=_model_order)
    agents = [*models, "active", "random"]
    summary = summarise(rows, agents, budgets)
    summary["curves"] = curves
    summary["tasks"] = len(tasks)
    summary["model_calls"] = sum(r.calls for r in rows if r.kind == "model")
    summary["calibration"] = _calibration(tasks_path)

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "episodes.json").write_text(
        json.dumps([r.to_json() for r in rows if r.kind == "model"], indent=0) + "\n"
    )
    (out / "report.md").write_text(markdown(summary, agents))
    charts(summary, agents, models, docs)
    _log(f"wrote {out / 'summary.json'}, {out / 'report.md'} and charts in {docs}")
    return summary


def _model_order(agent: str) -> tuple[int, str]:
    order = ["opus", "sonnet", "haiku", "fable", "gpt"]
    for i, key in enumerate(order):
        if key in agent:
            return (i, agent)
    return (len(order), agent)


def _calibration(tasks_path: Path) -> Any:
    path = tasks_path.with_name(tasks_path.stem + "-calibration.json")
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(v: list[float] | None, pct: bool = True, digits: int = 0) -> str:
    if not v or v[0] is None or (isinstance(v[0], float) and math.isnan(v[0])):
        return "n/a"
    if pct:
        return f"{v[0] * 100:.{digits}f}% [{v[1] * 100:.{digits}f}, {v[2] * 100:.{digits}f}]"
    return f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"


def markdown(summary: dict[str, Any], agents: list[str]) -> str:
    lines = ["# sleuth results", ""]
    lines.append(
        f"{summary['tasks']} tasks, {summary['model_calls']} recorded model calls. "
        "Intervals are 95%: Wilson for solve rates with one episode per task, otherwise a "
        "bootstrap over tasks."
    )
    for b in summary["budgets"]:
        lines += ["", f"## Budget {b}", ""]
        lines.append(
            "| agent | solve rate | accuracy | queries | stopped early | invalid / calls | "
            "info efficiency | random query, same state | zero-info queries | surprise rate |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for a in agents:
            s = summary["agents"][a]["by_budget"].get(str(b))
            if not s:
                continue
            calls = f"{s['invalid']} / {s['calls']}" if s["calls"] else "-"
            lines.append(
                f"| {summary['agents'][a]['label']} | {_fmt(s['solve_rate'])} | "
                f"{_fmt(s['accuracy'])} | {_fmt(s['queries'], pct=False)} | "
                f"{s['stopped_early'] * 100:.0f}% | {calls} | {_fmt(s['efficiency'])} | "
                f"{_fmt(s['random_efficiency'])} | "
                f"{_fmt(s['zero_info'])} | {_fmt(s['surprise_rate'])} |"
            )
        lines += ["", "Solve rate by tier:", ""]
        lines.append("| agent | tier 1 | tier 2 | tier 3 |")
        lines.append("| :--- | ---: | ---: | ---: |")
        for a in agents:
            s = summary["agents"][a]["by_budget"].get(str(b))
            if s:
                cells = " | ".join(_fmt(s["by_tier"][str(t)]["solve_rate"]) for t in (1, 2, 3))
                lines.append(f"| {summary['agents'][a]['label']} | {cells} |")
        lines += ["", "Solve rate by family:", ""]
        fams = list(FAMILY_NAMES)
        lines.append("| agent | " + " | ".join(FAMILY_NAMES[f] for f in fams) + " |")
        lines.append("| :--- |" + " ---: |" * len(fams))
        for a in agents:
            s = summary["agents"][a]["by_budget"].get(str(b))
            if s:
                cells = " | ".join(
                    _fmt(s["by_family"][f]["solve_rate"]) if f in s["by_family"] else "n/a"
                    for f in fams
                )
                lines.append(f"| {summary['agents'][a]['label']} | {cells} |")
        lines += ["", "Information efficiency by family (EIG of the query / best EIG):", ""]
        lines.append("| agent | " + " | ".join(FAMILY_NAMES[f] for f in fams) + " |")
        lines.append("| :--- |" + " ---: |" * len(fams))
        for a in agents:
            s = summary["agents"][a]["by_budget"].get(str(b))
            if s:
                cells = " | ".join(
                    _fmt(s["by_family"][f]["efficiency"]) if f in s["by_family"] else "n/a"
                    for f in fams
                )
                lines.append(f"| {summary['agents'][a]['label']} | {cells} |")
        lines += ["", "Positive tests on triples (share of queries the agent predicted true):", ""]
        for a in agents:
            s = summary["agents"][a]["by_budget"].get(str(b))
            if s and s["positive_test_rate"]:
                lines.append(f"- {summary['agents'][a]['label']}: {_fmt(s['positive_test_rate'])}")
    return "\n".join(lines) + "\n"


def charts(summary: dict[str, Any], agents: list[str], models: list[str], docs: Path) -> None:
    labels = {a: summary["agents"][a]["label"] for a in agents}
    ab = summary["agents"]
    budgets = summary["budgets"]
    top = str(max(budgets))
    points = {
        m: {int(b): tuple(ab[m]["by_budget"][b]["solve_rate"]) for b in ab[m]["by_budget"]}
        for m in models
    }
    plots.solve_curve(summary["curves"], points, labels, docs / "solve-vs-budget.svg")

    have = [a for a in agents if top in ab[a]["by_budget"]]
    plots.grouped_bars(
        ["tier 1", "tier 2", "tier 3"],
        {
            a: [tuple(ab[a]["by_budget"][top]["by_tier"][str(t)]["solve_rate"]) for t in (1, 2, 3)]
            for a in have
        },
        labels,
        f"solve rate at budget {top}",
        docs / "solve-by-tier.svg",
    )
    fams = list(FAMILY_NAMES)
    plots.grouped_bars(
        [FAMILY_NAMES[f] for f in fams],
        {
            a: [
                tuple(ab[a]["by_budget"][top]["by_family"].get(f, {}).get("solve_rate", [0, 0, 0]))
                for f in fams
            ]
            for a in have
        },
        labels,
        f"solve rate at budget {top}",
        docs / "solve-by-family.svg",
    )
    plots.grouped_bars(
        [FAMILY_NAMES[f] for f in fams],
        {
            a: [
                tuple(ab[a]["by_budget"][top]["by_family"].get(f, {}).get("efficiency", [0, 0, 0]))
                for f in fams
            ]
            for a in have
        },
        labels,
        "EIG of each query / best available EIG",
        docs / "info-by-family.svg",
    )
    col = plots.colours(agents)
    rows = []
    for a in have:
        e = ab[a]["by_budget"][top]["efficiency"]
        rows.append((labels[a], e[0], e[1], e[2], col[a]))
    plots.hbars(
        rows,
        f"information efficiency at budget {top} (1 = always the best query)",
        docs / "info-efficiency.svg",
        xlim=(0, 1.1),
    )
    plots.grouped_bars(
        [f"budget {b}" for b in budgets],
        {
            m: [tuple(ab[m]["by_budget"][str(b)]["edge_over_random"]) for b in budgets]
            for m in models
        },
        labels,
        "EIG above a random query from the same state\n(share of the best query's EIG)",
        docs / "info-edge.svg",
        ylim=None,
        zero_line=True,
    )
    rows = []
    for a in have:
        p = ab[a]["by_budget"][top]["positive_test_rate"]
        if p:
            rows.append((labels[a], p[0], p[1], p[2], col[a]))
    if rows:
        plots.hbars(
            rows,
            "share of triple queries the agent expected to fit the rule",
            docs / "positive-tests.svg",
            reference_line=0.5,
            xlim=(0, 1.1),
        )
