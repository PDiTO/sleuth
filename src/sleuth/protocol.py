"""The prompt a model sees each turn, and strict parsing of what it sends back.

The protocol is stateless. Every turn re-sends the whole lab notebook: the box's
description, the budget, and each earlier experiment with the hypothesis and prediction
the model stated before running it. The prompt is a pure function of the history, so a
recorded episode replays exactly.

A reply must be one JSON object and nothing else. No prose, no code fence, no extra
keys. Anything else is an invalid action: it is counted, the model is told why, and it
gets to try again without spending an experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .families import Family, Input, Output

MAX_HYPOTHESIS = 2000

SYSTEM = """\
You are the experimenter in a study of scientific reasoning. A black box hides a rule. \
You can learn about the rule only by choosing inputs and seeing what the box returns.

You have no tools. You cannot run code, search the web or read files, so do all of your \
reasoning in your head.

Each turn you either run one experiment or stop experimenting. Experiments are limited, so \
choose inputs that will tell you the most. When you stop, or when the budget runs out, you \
will be shown new inputs and asked to predict the box's output for each one. You are scored \
only on those predictions.

Every reply must be exactly one JSON object and nothing else: no text before or after it \
and no code fence. Before each experiment, write down your current best hypothesis and the \
output it predicts for the input you are about to try.\
"""

QUERY_SHAPE = (
    '{"hypothesis": "<your current best guess at the rule>", "query": <input>, '
    '"prediction": <the output your hypothesis predicts for that input>}'
)
COMMIT_SHAPE = '{"hypothesis": "<your final rule>", "commit": true}'


@dataclass(frozen=True)
class Experiment:
    hypothesis: str
    query: Input
    prediction: Output
    output: Output


@dataclass(frozen=True)
class Action:
    kind: str  # "query" | "commit"
    hypothesis: str
    query: Input = None
    prediction: Output = None


class InvalidAction(ValueError):
    pass


def _notebook(
    family: Family, given: list[tuple[Input, Output]], history: list[Experiment]
) -> list[str]:
    lines = ["# What you have seen", ""]
    if not given and not history:
        lines.append("Nothing yet.")
    for q, out in given:
        lines.append(
            f"Given to you before you started: {family.fmt_input(q)} -> {family.fmt_output(out)}"
        )
    for i, e in enumerate(history, 1):
        verdict = "as you predicted" if e.prediction == e.output else "you predicted "
        if e.prediction != e.output:
            verdict += family.fmt_output(e.prediction)
        lines.append(
            f"{i}. {family.fmt_input(e.query)} -> {family.fmt_output(e.output)} "
            f"({verdict}). Your hypothesis then: {json.dumps(e.hypothesis)}"
        )
    return lines


def turn_prompt(
    family: Family,
    budget: int,
    given: list[tuple[Input, Output]],
    history: list[Experiment],
    error: str | None = None,
) -> str:
    used = len(history)
    lines = ["# The box", "", family.describe, "", "# Budget", ""]
    lines.append(
        f"You may run up to {budget} experiments. You have run {used}, so {budget - used} remain."
    )
    lines += ["", *_notebook(family, given, history), "", "# Your move", ""]
    lines.append(f"An input is {family.query_format}. An output is {family.output_format}.")
    lines.append("Reply with exactly one JSON object. To run an experiment:")
    lines.append(QUERY_SHAPE)
    lines.append("To stop experimenting and move on to the test:")
    lines.append(COMMIT_SHAPE)
    if error:
        lines += [
            "",
            f"Your last reply was not a valid action: {error}. "
            "That did not use up an experiment. Reply again with one JSON object.",
        ]
    return "\n".join(lines)


def scoring_prompt(
    family: Family,
    given: list[tuple[Input, Output]],
    history: list[Experiment],
    hypothesis: str,
    inputs: list[Input],
    error: str | None = None,
) -> str:
    lines = ["# The box", "", family.describe, ""]
    lines += [*_notebook(family, given, history), "", "# Test", ""]
    lines.append(
        f"You have finished experimenting. Your final hypothesis was: {json.dumps(hypothesis)}"
    )
    lines.append(f"Predict the box's output for each of these {len(inputs)} inputs, in order:")
    lines += [f"{i}. {family.fmt_input(q)}" for i, q in enumerate(inputs, 1)]
    lines.append("")
    lines.append(
        'Reply with exactly one JSON object: {"predictions": [<output 1>, ..., '
        f"<output {len(inputs)}>]}}. Each output is {family.output_format}."
    )
    if error:
        lines += [
            "",
            f"Your last reply was not valid: {error}. Reply again with one JSON object.",
        ]
    return "\n".join(lines)


def _object(text: str) -> dict[str, Any]:
    body = text.strip()
    if body.startswith("```"):
        raise InvalidAction("the reply was wrapped in a code fence")
    if not body:
        raise InvalidAction("the reply was empty")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as e:
        if body.startswith("{"):
            raise InvalidAction(f"the reply is not valid JSON ({e.msg})") from None
        raise InvalidAction("the reply must be a JSON object and nothing else") from None
    if not isinstance(value, dict):
        raise InvalidAction("the reply must be a JSON object")
    return value


def _hypothesis(obj: dict[str, Any]) -> str:
    h = obj.get("hypothesis")
    if not isinstance(h, str) or not h.strip():
        raise InvalidAction('"hypothesis" must be a non-empty string')
    if len(h) > MAX_HYPOTHESIS:
        raise InvalidAction(f'"hypothesis" must be at most {MAX_HYPOTHESIS} characters')
    return h.strip()


def parse_action(family: Family, text: str) -> Action:
    obj = _object(text)
    keys = set(obj)
    if keys == {"hypothesis", "commit"}:
        if obj["commit"] is not True:
            raise InvalidAction('"commit" must be true')
        return Action("commit", _hypothesis(obj))
    if keys == {"hypothesis", "query", "prediction"}:
        hypothesis = _hypothesis(obj)
        q = family.parse_query(obj["query"])
        if q is None:
            raise InvalidAction(f'"query" must be {family.query_format}')
        pred = family.parse_output(obj["prediction"])
        if pred is None:
            raise InvalidAction(f'"prediction" must be {family.output_format}')
        return Action("query", hypothesis, q, pred)
    raise InvalidAction(
        'the object must have exactly the keys "hypothesis", "query" and "prediction", '
        'or exactly "hypothesis" and "commit"'
    )


def parse_predictions(family: Family, text: str, n: int) -> list[Output]:
    obj = _object(text)
    if set(obj) != {"predictions"}:
        raise InvalidAction('the object must have exactly one key, "predictions"')
    preds = obj["predictions"]
    if not isinstance(preds, list) or len(preds) != n:
        raise InvalidAction(f'"predictions" must be a list of exactly {n} outputs')
    out: list[Output] = []
    for i, raw in enumerate(preds, 1):
        v = family.parse_output(raw)
        if v is None:
            raise InvalidAction(f"prediction {i} must be {family.output_format}")
        out.append(v)
    return out
