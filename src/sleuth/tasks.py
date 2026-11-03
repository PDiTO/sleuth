"""Procedural tasks: a hidden rule, an optional starting example, and a hidden test set.

A task is drawn from a seed. The rule is a behaviour in the family's enumerated space
whose smallest program falls in the tier's size range. Rules are rejected when they are
degenerate (constant, or true almost never or almost always), when the active reference
agent pins them down with a single query ("trivial"), or when it can't within the query
budget ("impossible"). Every rejection is counted, so the calibration is visible.

The test set is chosen to separate the rule from its near misses. Near misses are the
hypotheses that agree with the rule on a handful of easy inputs, plus the ones that agree
with it most often overall, ranked by how often they agree with it. Each greedy pick separates the closest near miss still standing,
choosing among the inputs that do so the one that separates the most others, and the set
is topped up at random. A model that settles on a plausible but wrong rule fails some of them.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import reference
from .families import Input, Output
from .space import Space

TEST_SIZE = 10
GREEDY = 7
NEAR_MISSES = 400
CALIBRATION_BUDGET = 12


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    tier: int
    rule: int
    rule_text: str
    size: int
    seed: int
    given: list[tuple[Input, Output]]
    test_inputs: list[Input]
    test_outputs: list[Output]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: dict[str, Any]) -> Task:
        return Task(
            id=d["id"],
            family=d["family"],
            tier=d["tier"],
            rule=d["rule"],
            rule_text=d["rule_text"],
            size=d["size"],
            seed=d["seed"],
            given=[(_input(q), o) for q, o in d["given"]],
            test_inputs=[_input(q) for q in d["test_inputs"]],
            test_outputs=list(d["test_outputs"]),
        )


def _input(q: Any) -> Input:
    return tuple(q) if isinstance(q, list) else q


@dataclass
class Calibration:
    family: str
    tier: int
    drawn: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    accepted: int = 0
    active_queries: list[int] = field(default_factory=list)


def degenerate(space: Space, h: int) -> bool:
    col = space.sig[h]
    if space.family.output_kind == "bool":
        rate = float(np.mean(col))
        return not 0.05 <= rate <= 0.85
    return len(np.unique(col)) < 2


def seed_example(space: Space, h: int) -> list[tuple[Input, Output]]:
    """For triples, the first prototype (2-4-6 style) that fits the rule."""
    fam = space.family
    if not fam.prototypes:
        return []
    fits = fam.evaluate(space.programs[h], list(fam.prototypes))
    for q, ok in zip(fam.prototypes, fits, strict=True):
        if ok:
            return [(q, True)]
    return []


def test_set(
    space: Space, h: int, given: list[tuple[Input, Output]], rng: random.Random
) -> list[Input]:
    fam = space.family
    truth = space.sig[h]
    easy_cols = [space.probe_index[q] for q in fam.easy] + [
        space.probe_index[q] for q, _ in given if q in space.probe_index
    ]
    closeness = (space.sig == truth).mean(axis=1)
    agree_easy = np.all(space.sig[:, easy_cols] == truth[easy_cols], axis=1)
    closeness[h] = -1.0
    # Hypotheses that agree with the rule on the easy inputs come first, then the ones
    # that agree with it most often anywhere: both are the wrong answers a model that
    # stopped early would plausibly hold.
    easy_first = np.argsort(-closeness - agree_easy.astype(float), kind="stable")
    near = np.concatenate(
        [easy_first[: NEAR_MISSES // 2], np.argsort(-closeness, kind="stable")[: NEAR_MISSES // 2]]
    )
    near = np.array(list(dict.fromkeys(int(i) for i in near if i != h)), dtype=np.int64)
    near = near[np.argsort(-closeness[near], kind="stable")]
    excluded = set(easy_cols)
    chosen: list[int] = []
    differs = space.sig[near] != truth if len(near) else np.zeros((0, len(truth)), bool)
    alive = np.ones(len(near), dtype=bool)
    for _ in range(GREEDY):
        if not alive.any():
            break
        # Separate the closest near miss still standing, and among the inputs that do,
        # take one that separates the most of the others too.
        target = int(np.flatnonzero(alive)[0])
        gain = differs[alive].sum(axis=0).astype(float)
        gain[~differs[target]] = -1
        gain[list(excluded)] = -1
        top = float(gain.max())
        if top <= 0:
            alive[target] = False
            continue
        ties = [int(c) for c in np.flatnonzero(gain == top)]
        col = rng.choice(ties)
        chosen.append(col)
        excluded.add(col)
        alive &= ~differs[:, col]
    pool = [c for c in range(len(fam.probe)) if c not in excluded]
    rng.shuffle(pool)
    if fam.output_kind == "bool":
        pos = sum(int(truth[c]) for c in chosen)
        want_pos = max(0, 3 - pos)
        want_neg = max(0, 3 - (len(chosen) - pos))
        for c in list(pool):
            if len(chosen) >= TEST_SIZE:
                break
            if truth[c] and want_pos:
                chosen.append(c)
                want_pos -= 1
                pool.remove(c)
            elif not truth[c] and want_neg:
                chosen.append(c)
                want_neg -= 1
                pool.remove(c)
    for c in pool:
        if len(chosen) >= TEST_SIZE:
            break
        chosen.append(c)
    rng.shuffle(chosen)
    return [fam.probe[c] for c in chosen]


def make_task(space: Space, h: int, tier: int, seed: int, task_id: str) -> Task:
    rng = random.Random(seed)
    given = seed_example(space, h)
    inputs = test_set(space, h, given, rng)
    outputs = space.family.evaluate(space.programs[h], inputs)
    return Task(
        id=task_id,
        family=space.family.name,
        tier=tier,
        rule=h,
        rule_text=space.render(h),
        size=int(space.sizes[h]),
        seed=seed,
        given=given,
        test_inputs=inputs,
        test_outputs=outputs,
    )


def generate(
    space: Space, tier: int, n: int, seed: int, budget: int = CALIBRATION_BUDGET
) -> tuple[list[Task], Calibration]:
    fam = space.family
    level = next(t for t in fam.tiers if t.level == tier)
    rng = random.Random(f"{seed}:{fam.name}:{tier}")
    candidates = [
        int(i)
        for i in np.flatnonzero((space.sizes >= level.min_size) & (space.sizes <= level.max_size))
    ]
    rng.shuffle(candidates)
    cal = Calibration(fam.name, tier)
    tasks: list[Task] = []
    for h in candidates:
        if len(tasks) >= n:
            break
        cal.drawn += 1
        if degenerate(space, h):
            cal.rejected["degenerate"] += 1
            continue
        if fam.prototypes and not seed_example(space, h):
            cal.rejected["no starting example"] += 1
            continue
        task_seed = rng.randrange(2**31)
        task = make_task(space, h, tier, task_seed, f"{fam.name}-t{tier}-{len(tasks):02d}")
        run = reference.run(
            space,
            h,
            task.test_inputs,
            task.test_outputs,
            agent="active",
            budget=budget,
            seed=task_seed,
            given=task.given,
        )
        solved = run.solved_at()
        if solved is not None and solved <= 1:
            cal.rejected["trivial"] += 1
            continue
        if solved is None:
            cal.rejected["impossible"] += 1
            continue
        cal.accepted += 1
        cal.active_queries.append(solved)
        tasks.append(task)
    return tasks, cal


def save(tasks: list[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([t.to_json() for t in tasks], indent=1) + "\n")


def load(path: Path) -> list[Task]:
    return [Task.from_json(d) for d in json.loads(path.read_text())]
