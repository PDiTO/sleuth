"""Turn episodes into per-episode rows: score, behaviour and information per query.

Information is measured in the family's enumerated hypothesis space, the same one the
reference agents search. Before each query there is a version space V: the hypotheses
consistent with everything seen so far. For a query q:

- expected information gain, EIG(q), is the entropy of how V splits on q's output. It is
  what the query was worth before the answer came back. An EIG of zero means every
  hypothesis still standing agreed on the answer: nothing could be learned.
- the best EIG is the largest EIG over the probe inputs, what the active learner would
  have got at that point. The random EIG is the mean over probe inputs, what a random
  query is worth on average.
- realised bits are log2(|V before| / |V after|), what the answer actually told you.

All of it uses a uniform prior over V. That is a choice: a model that believes simpler
rules are likelier would value queries differently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import reference
from .episode import Episode
from .families import Input, Output
from .space import Ids, Space
from .tasks import Task

EIG_SAMPLE = 2000


@dataclass
class Row:
    agent: str
    kind: str  # "model" | "reference"
    task: str
    family: str
    tier: int
    budget: int
    seed: int
    accuracy: float
    solved: bool
    queries: int
    calls: int = 0
    invalid: int = 0
    tool_calls: int = 0
    committed: bool = False
    forfeited: bool = False
    surprises: int = 0
    positive: int = 0
    repeats: int = 0
    eig: list[float] = field(default_factory=list)
    best: list[float] = field(default_factory=list)
    rand: list[float] = field(default_factory=list)
    bits: list[float] = field(default_factory=list)
    vs_final: int = 0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("eig", "best", "rand", "bits"):
            d[k] = [round(v, 4) for v in d[k]]
        return d


class InfoCache:
    """Version-space states keyed by observation history, so shared openings cost once."""

    def __init__(self, space: Space, seed: int = 0) -> None:
        self.space = space
        self.seed = seed
        self._vs: dict[tuple[Any, ...], Ids] = {}
        self._tables: dict[tuple[Any, ...], tuple[float, float]] = {}

    def vs(self, obs: tuple[tuple[Input, Output], ...]) -> Ids:
        if obs in self._vs:
            return self._vs[obs]
        if not obs:
            out = self.space.all
        else:
            prev = self.vs(obs[:-1])
            q, y = obs[-1]
            out = self.space.consistent(prev, q, y)
        self._vs[obs] = out
        return out

    def best_and_mean(self, obs: tuple[tuple[Input, Output], ...]) -> tuple[float, float]:
        if obs not in self._tables:
            vs = self.vs(obs)
            if len(vs) <= 1:
                self._tables[obs] = (0.0, 0.0)
            else:
                rng = np.random.default_rng(self.seed)
                table = reference.eig_table(self.space, vs, rng, EIG_SAMPLE)
                self._tables[obs] = (float(np.max(table)), float(np.mean(table)))
        return self._tables[obs]


def model_row(ep: Episode, task: Task, info: InfoCache) -> Row:
    fam = info.space.family
    row = Row(
        agent=ep.provider,
        kind="model",
        task=task.id,
        family=task.family,
        tier=task.tier,
        budget=ep.budget,
        seed=0,
        accuracy=ep.accuracy,
        solved=ep.solved,
        queries=len(ep.experiments),
        calls=ep.calls,
        invalid=ep.invalid,
        tool_calls=sum(t.tool_calls for t in ep.turns),
        committed=ep.committed,
        forfeited=ep.forfeited_experiments or ep.forfeited_test,
    )
    obs: tuple[tuple[Input, Output], ...] = tuple(task.given)
    seen = {q for q, _ in task.given}
    for e in ep.experiments:
        vs = info.vs(obs)
        best, mean = info.best_and_mean(obs)
        row.eig.append(reference.query_eig(info.space, vs, e.query))
        row.best.append(best)
        row.rand.append(mean)
        obs = (*obs, (e.query, e.output))
        after = info.vs(obs)
        row.bits.append(float(np.log2(len(vs) / len(after))) if len(after) else 0.0)
        row.surprises += e.prediction != e.output
        if fam.output_kind == "bool":
            row.positive += e.prediction is True
        row.repeats += e.query in seen
        seen.add(e.query)
    row.vs_final = len(info.vs(obs))
    return row


def reference_rows(
    space: Space, task: Task, run: reference.RefRun, budgets: list[int], seed: int
) -> list[Row]:
    rows = []
    for b in budgets:
        steps = run.steps[:b]
        row = Row(
            agent=run.agent,
            kind="reference",
            task=task.id,
            family=task.family,
            tier=task.tier,
            budget=b,
            seed=seed,
            accuracy=run.accuracy[min(b, len(run.accuracy) - 1)],
            solved=run.accuracy[min(b, len(run.accuracy) - 1)] == 1.0,
            queries=len(steps),
            committed=len(steps) < b,
            eig=[s.eig for s in steps],
            best=[s.best_eig for s in steps],
            bits=[s.bits for s in steps],
            surprises=sum(s.prediction != s.output for s in steps),
            positive=sum(s.prediction is True for s in steps)
            if space.family.output_kind == "bool"
            else 0,
            vs_final=run.vs_sizes[min(b, len(run.vs_sizes) - 1)],
        )
        rows.append(row)
    return rows
