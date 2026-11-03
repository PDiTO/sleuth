"""Reference agents that search the same hypothesis space the rules come from.

Both keep the version space, the hypotheses still consistent with every observation,
and predict with its smallest program. They differ only in how they pick queries:

- `random`: a uniformly random probe input it hasn't asked yet.
- `active`: the probe input with the largest expected information gain, which for a
  deterministic box is the entropy of how the version space splits on that input. With a
  uniform prior over the version space that is the greedy optimum, so it serves as an
  approximate upper bound on how much a query can tell you.

Both stop asking once the version space is a single behaviour.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .families import Input, Output
from .space import Ids, Space, entropy_by_column, partition_entropy

EIG_SAMPLE = 3000


@dataclass
class Step:
    query: Input
    output: Output
    vs_before: int
    vs_after: int
    eig: float  # expected bits of the chosen query
    best_eig: float  # expected bits of the best probe query at that point
    prediction: Output = None  # what the agent's current best hypothesis said

    @property
    def bits(self) -> float:
        return math.log2(self.vs_before / self.vs_after) if self.vs_after else 0.0


@dataclass
class RefRun:
    agent: str
    steps: list[Step] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)  # after k queries, k = 0..budget
    vs_sizes: list[int] = field(default_factory=list)

    def identified_at(self) -> int | None:
        for k, n in enumerate(self.vs_sizes):
            if n == 1:
                return k
        return None

    def solved_at(self) -> int | None:
        for k, acc in enumerate(self.accuracy):
            if acc == 1.0:
                return k
        return None


def predict(space: Space, vs: Ids, inputs: list[Input]) -> list[Output]:
    h = space.simplest(vs)
    return space.family.evaluate(space.programs[h], inputs)


def accuracy(predicted: list[Any], truth: list[Any]) -> float:
    hits = sum(bool(p == t) for p, t in zip(predicted, truth, strict=True))
    return hits / len(truth)


def eig_table(
    space: Space, vs: Ids, rng: np.random.Generator, sample: int = EIG_SAMPLE
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Expected information gain of every probe input, estimated on a subsample."""
    sub = vs if len(vs) <= sample else np.sort(rng.choice(vs, size=sample, replace=False))
    return entropy_by_column(space.sig[sub])


def best_eig(space: Space, vs: Ids, seed: int = 0) -> float:
    if len(vs) <= 1:
        return 0.0
    return float(np.max(eig_table(space, vs, np.random.default_rng(seed))))


def query_eig(space: Space, vs: Ids, query: Input) -> float:
    return partition_entropy(space.codes(vs, query))


def run(
    space: Space,
    truth: int,
    test_inputs: list[Input],
    test_outputs: list[Output],
    *,
    agent: str,
    budget: int,
    seed: int,
    given: list[tuple[Input, Output]] | None = None,
) -> RefRun:
    fam = space.family
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    vs = space.filter(given or [])
    asked: set[int] = set()
    out = RefRun(agent)
    out.vs_sizes.append(len(vs))
    out.accuracy.append(accuracy(predict(space, vs, test_inputs), test_outputs))
    order = list(range(len(fam.probe)))
    pyrng.shuffle(order)
    for _ in range(budget):
        if len(vs) <= 1:
            out.vs_sizes.append(len(vs))
            out.accuracy.append(out.accuracy[-1])
            continue
        table = eig_table(space, vs, rng)
        if agent == "active":
            masked = table.copy()
            if asked:
                masked[list(asked)] = -1.0
            col = int(np.argmax(masked))
        elif agent == "random":
            col = next(c for c in order if c not in asked)
        else:
            raise ValueError(f"unknown reference agent {agent!r}")
        asked.add(col)
        q = fam.probe[col]
        y = fam.evaluate(space.programs[truth], [q])[0]
        before = len(vs)
        guess = fam.evaluate(space.programs[space.simplest(vs)], [q])[0]
        eig = query_eig(space, vs, q) if len(vs) <= EIG_SAMPLE else float(table[col])
        vs = space.consistent(vs, q, y)
        out.steps.append(Step(q, y, before, len(vs), eig, float(np.max(table)), guess))
        out.vs_sizes.append(len(vs))
        out.accuracy.append(accuracy(predict(space, vs, test_inputs), test_outputs))
    return out
