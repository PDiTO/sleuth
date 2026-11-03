"""sleuth tasks | run | report | show | calibrate"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import families, report, runner, tasks
from . import space as spaces

DEFAULT_SEED = 20261103
DEFAULT_TASKS = Path("results/tasks.json")
DEFAULT_MODELS = (
    "claude:claude-opus-5-5",
    "claude:claude-sonnet-5-5",
    "claude:claude-haiku-5-5",
    "codex:gpt-6.1-sol",
)
DEFAULT_BUDGETS = (4, 12)


def cmd_tasks(args: argparse.Namespace) -> int:
    all_tasks: list[tasks.Task] = []
    calibration = []
    for name in args.family or list(families.FAMILIES):
        sp = spaces.build(name)
        for tier in (1, 2, 3):
            ts, cal = tasks.generate(sp, tier, args.per_tier, args.seed)
            all_tasks += ts
            calibration.append(
                {
                    "family": name,
                    "tier": tier,
                    "drawn": cal.drawn,
                    "accepted": cal.accepted,
                    "rejected": dict(cal.rejected),
                    "active_queries": cal.active_queries,
                }
            )
            print(
                f"{name} tier {tier}: {cal.accepted} accepted of {cal.drawn} drawn, "
                f"rejected {dict(cal.rejected)}",
                file=sys.stderr,
            )
    tasks.save(all_tasks, args.out)
    cal_path = args.out.with_name(args.out.stem + "-calibration.json")
    cal_path.write_text(json.dumps(calibration, indent=1) + "\n")
    print(f"wrote {len(all_tasks)} tasks to {args.out}", file=sys.stderr)
    return 0


def _select(all_tasks: list[tasks.Task], args: argparse.Namespace) -> list[tasks.Task]:
    out = all_tasks
    if args.family:
        out = [t for t in out if t.family in args.family]
    if args.tier:
        out = [t for t in out if t.tier in args.tier]
    if args.task:
        out = [t for t in out if t.id in args.task]
    if args.per_cell is not None:
        counts: dict[tuple[str, int], int] = {}
        kept = []
        for t in out:
            k = (t.family, t.tier)
            if counts.get(k, 0) < args.per_cell:
                kept.append(t)
                counts[k] = counts.get(k, 0) + 1
        out = kept
    return out


def cmd_run(args: argparse.Namespace) -> int:
    selected = _select(tasks.load(args.tasks), args)
    world = runner.World(selected)
    specs = args.model or list(DEFAULT_MODELS)
    budgets = args.budget or list(DEFAULT_BUDGETS)
    n = len(selected) * len(specs) * len(budgets)
    print(
        f"{len(selected)} tasks x {len(specs)} models x {len(budgets)} budgets = {n} episodes, "
        f"up to {len(selected) * len(specs) * sum(b + 1 for b in budgets)} calls "
        "before retries; recorded calls are reused",
        file=sys.stderr,
    )
    workers = {"claude": args.claude_workers, "codex": args.codex_workers}
    runner.run(world, specs, budgets, root=args.recordings, workers=workers)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report.build(
        tasks_path=args.tasks,
        recordings=args.recordings,
        out=args.out,
        docs=args.docs,
        random_seeds=args.random_seeds,
    )
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    world = runner.World(tasks.load(args.tasks))
    dropped = runner.prune(world, args.recordings)
    print(f"dropped {dropped} unused recorded calls", file=sys.stderr)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    all_tasks = {t.id: t for t in tasks.load(args.tasks)}
    path = args.path
    data = json.loads(path.read_text())
    task = all_tasks[data["task"]]
    world = runner.World([task])
    ep = runner.replay_one(world, path)
    fam = world.space(task.family).family
    print(f"{task.id}: {task.rule_text}  ({data['provider']}, budget {ep.budget})")
    for q, out in task.given:
        print(f"  given {fam.fmt_input(q)} -> {fam.fmt_output(out)}")
    for i, e in enumerate(ep.experiments, 1):
        mark = " " if e.prediction == e.output else "!"
        print(
            f"  {i:2d}{mark} {fam.fmt_input(e.query)} -> {fam.fmt_output(e.output)}"
            f"  (predicted {fam.fmt_output(e.prediction)})  {e.hypothesis}"
        )
    print(f"  final: {ep.final_hypothesis}")
    print(f"  test accuracy {ep.accuracy:.0%}, {ep.invalid} invalid replies")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sleuth", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tasks", help="generate and calibrate the task set")
    t.add_argument("--per-tier", type=int, default=5)
    t.add_argument("--seed", type=int, default=DEFAULT_SEED)
    t.add_argument("--family", action="append")
    t.add_argument("--out", type=Path, default=DEFAULT_TASKS)
    t.set_defaults(fn=cmd_tasks)

    r = sub.add_parser("run", help="run models on the tasks, recording every call")
    r.add_argument("--model", action="append", help="backend:model[@effort], repeatable")
    r.add_argument("--budget", action="append", type=int)
    r.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    r.add_argument("--recordings", type=Path, default=runner.RECORDINGS)
    r.add_argument("--family", action="append")
    r.add_argument("--tier", action="append", type=int)
    r.add_argument("--task", action="append")
    r.add_argument("--per-cell", type=int)
    r.add_argument("--claude-workers", type=int, default=4)
    r.add_argument("--codex-workers", type=int, default=4)
    r.set_defaults(fn=cmd_run)

    rep = sub.add_parser("report", help="rebuild every result offline from the recordings")
    rep.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    rep.add_argument("--recordings", type=Path, default=runner.RECORDINGS)
    rep.add_argument("--out", type=Path, default=Path("results"))
    rep.add_argument("--docs", type=Path, default=Path("docs"))
    rep.add_argument("--random-seeds", type=int, default=10)
    rep.set_defaults(fn=cmd_report)

    pr = sub.add_parser("prune", help="drop recorded calls that no replay uses")
    pr.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    pr.add_argument("--recordings", type=Path, default=runner.RECORDINGS)
    pr.set_defaults(fn=cmd_prune)

    s = sub.add_parser("show", help="print one recorded episode")
    s.add_argument("path", type=Path)
    s.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    s.set_defaults(fn=cmd_show)

    args = p.parse_args(argv)
    code: int = args.fn(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
