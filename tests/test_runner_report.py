from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from sleuth import cli, report, runner, stats, tasks
from sleuth import space as spaces
from sleuth.episode import Episode
from sleuth.providers import Provider, Scripted


def _guesser(system: str, prompt: str) -> str:
    """A fake model: queries 0, 1, 2, ... then predicts the last output it saw."""
    seen = re.findall(r"-> (-?\d+|true|false|\"[a-z]*\")", prompt)
    if "# Test" in prompt:
        n = int(re.search(r"each of these (\d+) inputs", prompt).group(1))  # type: ignore[union-attr]
        last = seen[-1] if seen else "0"
        return json.dumps({"predictions": [json.loads(last)] * n})
    k = len(re.findall(r"^\d+\. ", prompt, re.M))
    if "triple" in prompt or "[a, b, c]" in prompt:
        q: object = [k, k + 1, k + 2]
        pred: object = True
    elif "f(x, y)" in prompt:
        q, pred = [k, -k], 0
    elif "lowercase" in prompt:
        q, pred = "abc"[: 1 + k % 3], "a"
    else:
        q, pred = k, k
    return json.dumps({"hypothesis": f"guess {k}", "query": q, "prediction": pred})


@pytest.fixture(scope="module")
def small_tasks() -> list[tasks.Task]:
    out: list[tasks.Task] = []
    for fam in ("int1", "triple", "string"):
        sp = spaces.build(fam)
        ts, _ = tasks.generate(sp, 1, 1, seed=3)
        out += ts
    return out


def test_run_replay_and_report(small_tasks: list[tasks.Task], tmp_path: Path) -> None:
    world = runner.World(small_tasks)
    rec = tmp_path / "recordings"
    made: list[Scripted] = []

    def make(spec: str) -> Provider:
        p = Scripted(_guesser, name=f"fake/{spec.split(':')[1]}")
        made.append(p)
        return p

    logs: list[str] = []
    eps = runner.run(world, ["claude:a", "codex:b"], [2, 3], root=rec, make=make, log=logs.append)
    assert len(eps) == len(small_tasks) * 4
    calls = sum(p.calls for p in made)
    assert calls == sum(e.calls for e in eps)

    # a second run reuses every recorded call
    again = runner.run(world, ["claude:a"], [2], root=rec, make=make, log=logs.append)
    assert made[-1].calls == 0
    assert len(again) == len(small_tasks)

    replayed = runner.replay(world, rec)

    def key(e: Episode) -> tuple[str, str, int]:
        return (e.task_id, e.provider, e.budget)

    by_key = {key(e): e for e in eps}
    for e in replayed:
        orig = by_key[key(e)]
        assert e.experiments == orig.experiments
        assert e.predictions == orig.predictions
        assert e.accuracy == orig.accuracy

    tasks_path = tmp_path / "tasks.json"
    tasks.save(small_tasks, tasks_path)
    summary = report.build(
        tasks_path=tasks_path,
        recordings=rec,
        out=tmp_path / "out",
        docs=tmp_path / "docs",
        random_seeds=2,
    )
    assert summary["model_calls"] == calls
    assert set(summary["agents"]) == {"fake/a", "fake/b", "active", "random"}
    assert (tmp_path / "docs" / "solve-vs-budget.svg").exists()
    assert (tmp_path / "docs" / "positive-tests.svg").exists()
    md = (tmp_path / "out" / "report.md").read_text()
    assert "## Budget 3" in md
    rows = json.loads((tmp_path / "out" / "episodes.json").read_text())
    assert len(rows) == len(eps)
    act = summary["agents"]["active"]["by_budget"]["3"]
    assert act["efficiency"][0] == pytest.approx(1.0, abs=0.05)

    # the cli wrapper around show
    path = next(rec.glob("*/*.json"))
    assert cli.main(["show", str(path), "--tasks", str(tasks_path)]) == 0


def test_world_rejects_changed_rules(small_tasks: list[tasks.Task]) -> None:
    t = small_tasks[0]
    bad = tasks.Task(**{**t.__dict__, "rule_text": "something else"})
    with pytest.raises(RuntimeError, match="no longer renders"):
        runner.World([bad]).truth(bad)


def test_labels() -> None:
    assert report.label("claude/claude-opus-5-5@medium") == "Opus 5.5"
    assert report.label("codex/gpt-6.1-sol@medium") == "GPT-6.1 Sol"
    assert report.label("active").startswith("active")
    assert report.label("random").startswith("random")
    assert report.label("x/other") == "other"


def test_stats() -> None:
    lo, hi = stats.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)
    assert all(math.isnan(v) for v in stats.wilson(0, 0))
    items = [("a", 1.0), ("a", 1.0), ("b", 0.0), ("c", 1.0)]
    point, lo2, hi2 = stats.cluster_bootstrap(items, lambda i: i[0], stats.mean(lambda i: i[1]))
    assert point == 0.75
    assert 0.0 <= lo2 <= point <= hi2 <= 1.0
    assert all(math.isnan(v) for v in stats.cluster_bootstrap([], lambda i: "", lambda x: 0.0))
    assert math.isnan(stats.ratio(lambda i: 1.0, lambda i: 0.0)([1]))


def test_cli_tasks_and_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "t.json"
    assert cli.main(["tasks", "--family", "string", "--per-tier", "1", "--out", str(out)]) == 0
    assert len(tasks.load(out)) == 3
    assert (tmp_path / "t-calibration.json").exists()

    seen: dict[str, object] = {}

    def fake_run(
        world: runner.World, specs: list[str], budgets: list[int], **kw: object
    ) -> list[object]:
        seen.update(tasks=len(world.tasks), specs=specs, budgets=budgets)
        return []

    monkeypatch.setattr(runner, "run", fake_run)
    code = cli.main(
        [
            "run",
            "--tasks",
            str(out),
            "--model",
            "claude:x",
            "--budget",
            "4",
            "--tier",
            "1",
            "--per-cell",
            "1",
            "--family",
            "string",
        ]
    )
    assert code == 0
    assert seen == {"tasks": 1, "specs": ["claude:x"], "budgets": [4]}
