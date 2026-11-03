"""One model playing one task: experiments, then the test.

The loop is deterministic given the model's replies. The prompt at each turn depends only
on the task and the history so far, and the box is a pure function, so replaying the
recorded replies reproduces the episode exactly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .families import Family, Input, Output
from .protocol import (
    SYSTEM,
    Experiment,
    InvalidAction,
    parse_action,
    parse_predictions,
    scoring_prompt,
    turn_prompt,
)
from .providers import Provider, ProviderError, ReplayMiss, Reply
from .tasks import Task

MAX_RETRIES = 2  # invalid replies allowed in a row before the phase is forfeited


@dataclass
class Turn:
    phase: str  # "experiment" | "test"
    text: str
    error: str | None
    tool_calls: int = 0

    @property
    def valid(self) -> bool:
        return self.error is None


@dataclass
class Episode:
    task_id: str
    provider: str
    budget: int
    experiments: list[Experiment] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    committed: bool = False
    forfeited_experiments: bool = False
    forfeited_test: bool = False
    final_hypothesis: str = ""
    predictions: list[Output] | None = None
    accuracy: float = 0.0

    @property
    def solved(self) -> bool:
        return self.accuracy == 1.0

    @property
    def invalid(self) -> int:
        return sum(not t.valid for t in self.turns)

    @property
    def calls(self) -> int:
        return len(self.turns)


def call_with_retry(
    provider: Provider,
    system: str,
    prompt: str,
    *,
    attempts: int = 3,
    wait: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Reply:
    for i in range(attempts):
        try:
            return provider.complete(system, prompt)
        except ReplayMiss:
            raise
        except ProviderError:
            if i == attempts - 1:
                raise
            sleep(wait * (3**i))
    raise AssertionError("unreachable")  # pragma: no cover


def play(
    task: Task,
    family: Family,
    truth: Callable[[Input], Output],
    provider: Provider,
    budget: int,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Episode:
    ep = Episode(task.id, provider.id, budget)
    error: str | None = None
    streak = 0
    while len(ep.experiments) < budget:
        prompt = turn_prompt(family, budget, task.given, ep.experiments, error)
        reply = call_with_retry(provider, SYSTEM, prompt, sleep=sleep)
        tools = int(reply.meta.get("tool_calls") or 0)
        try:
            action = parse_action(family, reply.text)
        except InvalidAction as e:
            ep.turns.append(Turn("experiment", reply.text, str(e), tools))
            error = str(e)
            streak += 1
            if streak > MAX_RETRIES:
                ep.forfeited_experiments = True
                break
            continue
        ep.turns.append(Turn("experiment", reply.text, None, tools))
        error, streak = None, 0
        ep.final_hypothesis = action.hypothesis
        if action.kind == "commit":
            ep.committed = True
            break
        ep.experiments.append(
            Experiment(action.hypothesis, action.query, action.prediction, truth(action.query))
        )

    error = None
    for _ in range(MAX_RETRIES + 1):
        prompt = scoring_prompt(
            family,
            task.given,
            ep.experiments,
            ep.final_hypothesis or "(none stated)",
            task.test_inputs,
            error,
        )
        reply = call_with_retry(provider, SYSTEM, prompt, sleep=sleep)
        tools = int(reply.meta.get("tool_calls") or 0)
        try:
            preds = parse_predictions(family, reply.text, len(task.test_inputs))
        except InvalidAction as e:
            ep.turns.append(Turn("test", reply.text, str(e), tools))
            error = str(e)
            continue
        ep.turns.append(Turn("test", reply.text, None, tools))
        ep.predictions = preds
        hits = sum(p == t for p, t in zip(preds, task.test_outputs, strict=True))
        ep.accuracy = hits / len(task.test_outputs)
        break
    else:
        ep.forfeited_test = True
    return ep
