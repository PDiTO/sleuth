from __future__ import annotations

import json
import re

import pytest

from sleuth import families
from sleuth.episode import MAX_RETRIES, call_with_retry, play
from sleuth.protocol import (
    Experiment,
    InvalidAction,
    parse_action,
    parse_predictions,
    scoring_prompt,
    turn_prompt,
)
from sleuth.providers import Provider, ProviderError, ReplayMiss, Reply, Scripted
from sleuth.tasks import Task

INT1 = families.int1()
TRIPLE = families.triple()


def _task() -> Task:
    inputs = [-7, -1, 0, 3, 8, 13, 22, 31, 40, 50]
    return Task(
        id="int1-t1-test",
        family="int1",
        tier=1,
        rule=0,
        rule_text="x mod 5",
        size=3,
        seed=1,
        given=[],
        test_inputs=inputs,
        test_outputs=[x % 5 for x in inputs],
    )


def test_parse_action_accepts_queries_and_commits() -> None:
    a = parse_action(INT1, '{"hypothesis": "x mod 5", "query": 7, "prediction": 2}')
    assert (a.kind, a.query, a.prediction) == ("query", 7, 2)
    c = parse_action(INT1, '  {"hypothesis": "x mod 5", "commit": true}\n')
    assert c.kind == "commit"
    t = parse_action(TRIPLE, '{"hypothesis": "h", "query": [1, 2, 3], "prediction": false}')
    assert t.query == (1, 2, 3)
    assert t.prediction is False


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ('```json\n{"hypothesis": "h", "commit": true}\n```', "code fence"),
        ('Sure! {"hypothesis": "h", "commit": true}', "JSON object and nothing else"),
        ('{"hypothesis": "h", "commit": true', "not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ("", "empty"),
        ('{"hypothesis": "h", "commit": false}', "must be true"),
        ('{"hypothesis": "", "commit": true}', "non-empty"),
        ('{"hypothesis": "h", "query": 7}', "exactly the keys"),
        ('{"hypothesis": "h", "query": 7, "prediction": 2, "why": "x"}', "exactly the keys"),
        ('{"hypothesis": "h", "query": 99, "prediction": 2}', '"query" must be'),
        ('{"hypothesis": "h", "query": 7, "prediction": 2.5}', '"prediction" must be'),
        ('{"hypothesis": "h", "query": 7, "prediction": true}', '"prediction" must be'),
    ],
)
def test_parse_action_rejects(text: str, reason: str) -> None:
    with pytest.raises(InvalidAction, match=re.escape(reason)):
        parse_action(INT1, text)


def test_hypothesis_length_cap() -> None:
    long = json.dumps({"hypothesis": "x" * 3000, "commit": True})
    with pytest.raises(InvalidAction, match="at most"):
        parse_action(INT1, long)


def test_parse_predictions() -> None:
    assert parse_predictions(INT1, '{"predictions": [1, 2]}', 2) == [1, 2]
    with pytest.raises(InvalidAction, match="exactly 2"):
        parse_predictions(INT1, '{"predictions": [1]}', 2)
    with pytest.raises(InvalidAction, match="prediction 2"):
        parse_predictions(INT1, '{"predictions": [1, "2"]}', 2)
    with pytest.raises(InvalidAction, match="exactly one key"):
        parse_predictions(INT1, '{"predictions": [1, 2], "note": 1}', 2)


def test_prompts_carry_the_whole_notebook() -> None:
    hist = [Experiment("x + 1", 3, 4, 3), Experiment("x", 5, 5, 0)]
    p = turn_prompt(INT1, 12, [], hist)
    assert "You have run 2, so 10 remain." in p
    assert 'f(3) -> 3 (you predicted 4). Your hypothesis then: "x + 1"' in p
    assert "f(5) -> 0 (you predicted 5)" in p
    assert "Reply again" not in p
    assert "Reply again" in turn_prompt(INT1, 12, [], hist, error="bad")
    assert "Nothing yet." in turn_prompt(INT1, 12, [], [])
    tp = scoring_prompt(TRIPLE, [((2, 4, 6), True)], [], "even", [(1, 2, 3)])
    assert "Given to you before you started: [2, 4, 6] -> true" in tp
    assert "1. [1, 2, 3]" in tp
    assert "not valid" in scoring_prompt(TRIPLE, [], [], "h", [(1, 2, 3)], error="x")


def _box(q: int) -> int:
    return q % 5


def test_play_scores_and_counts_invalid_actions() -> None:
    replies = iter(
        [
            "I think it's mod 5",  # invalid
            '{"hypothesis": "x mod 5", "query": 7, "prediction": 2}',
            '{"hypothesis": "x mod 5", "query": -3, "prediction": 2}',
            '{"hypothesis": "x mod 5", "commit": true}',
            '{"predictions": [3, 4, 0, 3, 3, 3, 2, 1, 0, 0]}',
        ]
    )
    prov = Scripted(lambda s, p: next(replies))
    ep = play(_task(), INT1, _box, prov, 12)
    assert ep.committed
    assert len(ep.experiments) == 2
    assert ep.invalid == 1
    assert ep.calls == 5
    assert ep.solved
    assert ep.final_hypothesis == "x mod 5"


def test_play_runs_out_of_budget_then_tests() -> None:
    def reply(system: str, prompt: str) -> str:
        if "# Test" in prompt:
            return '{"predictions": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}'
        return '{"hypothesis": "zero", "query": 5, "prediction": 0}'

    ep = play(_task(), INT1, _box, Scripted(reply), 3)
    assert len(ep.experiments) == 3
    assert not ep.committed
    assert ep.accuracy == pytest.approx(0.3)
    assert not ep.solved


def test_play_forfeits_after_repeated_invalid_replies() -> None:
    ep = play(_task(), INT1, _box, Scripted(lambda s, p: "nope"), 12)
    assert ep.forfeited_experiments
    assert ep.forfeited_test
    assert ep.invalid == (MAX_RETRIES + 1) * 2
    assert ep.accuracy == 0.0
    assert ep.predictions is None


class Flaky(Provider):
    def __init__(self, fails: int, exc: type[Exception] = ProviderError) -> None:
        self.fails = fails
        self.exc = exc

    @property
    def id(self) -> str:
        return "flaky"

    def complete(self, system: str, prompt: str) -> Reply:
        if self.fails:
            self.fails -= 1
            raise self.exc("boom")
        return Reply("ok")


def test_call_with_retry() -> None:
    waits: list[float] = []
    assert call_with_retry(Flaky(2), "s", "p", sleep=waits.append).text == "ok"
    assert waits == [20.0, 60.0]
    with pytest.raises(ProviderError):
        call_with_retry(Flaky(5), "s", "p", sleep=waits.append)
    with pytest.raises(ReplayMiss):
        call_with_retry(Flaky(1, ReplayMiss), "s", "p", sleep=waits.append)
