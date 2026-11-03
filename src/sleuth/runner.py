"""Run models over the task set, with resumable recordings, and replay them offline."""

from __future__ import annotations

import json
import sys
import threading
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import space as spaces
from .episode import Episode, play
from .families import Input, Output
from .providers import Provider, Recorder, from_spec, slug
from .space import Space
from .tasks import Task

RECORDINGS = Path("results/recordings")


class World:
    """Tasks plus the hypothesis spaces they were drawn from, built once."""

    def __init__(self, tasks: list[Task], *, cache: bool = True) -> None:
        self.tasks = tasks
        self.by_id = {t.id: t for t in tasks}
        self._spaces: dict[str, Space] = {}
        self._cache = cache
        self._lock = threading.Lock()

    def space(self, family: str) -> Space:
        with self._lock:
            if family not in self._spaces:
                self._spaces[family] = spaces.build(family, cache=self._cache)
            return self._spaces[family]

    def truth(self, task: Task) -> Callable[[Input], Output]:
        sp = self.space(task.family)
        program = sp.programs[task.rule]
        if sp.render(task.rule) != task.rule_text:
            raise RuntimeError(f"{task.id}: rule {task.rule} no longer renders as recorded")
        return lambda q: sp.family.evaluate(program, [q])[0]


def recording_path(root: Path, provider_id: str, task: Task, budget: int) -> Path:
    return root / slug(provider_id) / f"{task.id}-b{budget}.json"


def play_recorded(world: World, task: Task, budget: int, recorder: Recorder) -> Episode:
    recorder.data.setdefault("task", task.id)
    recorder.data.setdefault("budget", budget)
    sp = world.space(task.family)
    return play(task, sp.family, world.truth(task), recorder, budget)


@dataclass(frozen=True)
class Job:
    spec: str
    task: Task
    budget: int


def run(
    world: World,
    specs: Iterable[str],
    budgets: Iterable[int],
    *,
    root: Path = RECORDINGS,
    workers: dict[str, int] | None = None,
    make: Callable[[str], Provider] = from_spec,
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr, flush=True),
) -> list[Episode]:
    workers = workers or {"claude": 4, "codex": 4}
    providers = {spec: make(spec) for spec in specs}
    jobs = [Job(spec, task, b) for task in world.tasks for b in budgets for spec in providers]
    for fam in {t.family for t in world.tasks}:
        world.space(fam)
    pools = {
        backend: ThreadPoolExecutor(max_workers=n, thread_name_prefix=backend)
        for backend, n in workers.items()
    }
    futures: list[tuple[Job, Future[Episode]]] = []
    done_count = [0]
    lock = threading.Lock()

    def one(job: Job) -> Episode:
        provider = providers[job.spec]
        path = recording_path(root, provider.id, job.task, job.budget)
        rec = Recorder(path, provider.id, provider)
        ep = play_recorded(world, job.task, job.budget, rec)
        rec.data["complete"] = True
        rec.save()
        with lock:
            done_count[0] += 1
            log(
                f"[{done_count[0]}/{len(jobs)}] {provider.id} {job.task.id} b{job.budget}: "
                f"acc {ep.accuracy:.1f}, {len(ep.experiments)} queries, {ep.calls} calls "
                f"({rec.misses} new), {ep.invalid} invalid"
            )
        return ep

    for job in jobs:
        backend = job.spec.split(":", 1)[0]
        pool = pools.get(backend)
        if pool is None:
            raise ValueError(f"no worker pool for backend {backend!r}")
        futures.append((job, pool.submit(one, job)))
    episodes: list[Episode] = []
    try:
        for job, fut in futures:
            try:
                episodes.append(fut.result())
            except Exception as e:
                log(f"FAILED {job.spec} {job.task.id} b{job.budget}: {e}")
                log("".join(traceback.format_exception_only(e)).strip())
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
    return episodes


def replay(world: World, root: Path = RECORDINGS) -> list[Episode]:
    """Rebuild every recorded episode from disk, with no model calls."""
    episodes: list[Episode] = []
    for path in sorted(root.glob("*/*.json")):
        data = json.loads(path.read_text())
        if not data.get("complete"):
            continue
        if data["task"] not in world.by_id:
            continue
        episodes.append(replay_one(world, path))
    return episodes


def replay_one(world: World, path: Path, *, prune: bool = False) -> Episode:
    """Replay one recording. With `prune`, drop recorded calls the replay didn't use,
    which are left behind when a task's test set or a prompt changes between runs."""
    data = json.loads(path.read_text())
    task = world.by_id[data["task"]]
    rec = Recorder(path, data["provider"], None)
    ep = play_recorded(world, task, int(data["budget"]), rec)
    if prune:
        kept = [c for c in rec.data["calls"] if c["key"] in rec.used]
        if len(kept) != len(rec.data["calls"]):
            rec.data["calls"] = kept
            rec.save()
    return ep


def prune(world: World, root: Path = RECORDINGS) -> int:
    """Drop unused calls from every complete recording; returns how many were dropped."""
    dropped = 0
    for path in sorted(root.glob("*/*.json")):
        data = json.loads(path.read_text())
        if not data.get("complete") or data["task"] not in world.by_id:
            continue
        before = len(data["calls"])
        replay_one(world, path, prune=True)
        dropped += before - len(json.loads(path.read_text())["calls"])
    return dropped
