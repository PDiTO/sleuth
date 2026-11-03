"""The four rule families: what the inputs are, which ops build rules, and how to talk
about them in a prompt.

- `int1`: integer functions f(x), x in [-50, 50].
- `int2`: integer functions f(x, y), x and y in [-20, 20].
- `triple`: true/false rules over triples (a, b, c), each in [0, 30]. A generalisation of
  Wason's 2-4-6 task. The model is shown one triple that fits, chosen to also fit a
  narrower, more obvious rule.
- `string`: transforms of a lowercase string of length 1 to 8.

Each family owns its probe set. Programs are told apart by their outputs on it, and the
reference agents choose their queries from it. For `int1` and `int2` the probe set is the
whole input domain, so two programs with the same signature really are the same rule.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dsl import Grammar, Node, Op

Input = Any  # int | tuple[int, int] | tuple[int, int, int] | str
Output = Any  # int | bool | str

MAX_ABS = 10_000_000


def _intv(vec: Any) -> bool:
    return bool(np.abs(vec).max() <= MAX_ABS)


def _int_key(vec: Any) -> bytes:
    return np.ascontiguousarray(vec, dtype=np.int64).tobytes()


def _bool_key(vec: Any) -> bytes:
    return np.packbits(np.asarray(vec, dtype=bool)).tobytes()


def _str_key(vec: Any) -> bytes:
    return "\x00".join(vec).encode()


@dataclass(frozen=True)
class Tier:
    level: int
    min_size: int
    max_size: int


@dataclass
class Family:
    name: str
    title: str
    grammar: Grammar
    max_size: int
    tiers: tuple[Tier, ...]
    probe: list[Input]
    batch: Callable[[Sequence[Input]], Any]
    output_kind: str  # "int" | "bool" | "str"
    describe: str
    query_format: str
    output_format: str
    easy: list[Input]
    parse_query: Callable[[Any], Input | None]
    fmt_input: Callable[[Input], str]
    prototypes: tuple[Input, ...] = ()  # triple only: candidate "2-4-6" examples

    def tier_of(self, size: int) -> int | None:
        for t in self.tiers:
            if t.min_size <= size <= t.max_size:
                return t.level
        return None

    def evaluate(self, node: Node, inputs: Sequence[Input]) -> list[Output]:
        vec = self.grammar.evaluate(node, self.batch(inputs))
        return [self.to_python(v) for v in _broadcast(vec, len(inputs))]

    def to_python(self, value: Any) -> Output:
        if self.output_kind == "int":
            return int(value)
        if self.output_kind == "bool":
            return bool(value)
        return str(value)

    def parse_output(self, raw: Any) -> Output | None:
        """Strict: JSON ints for ints, JSON booleans for booleans, strings for strings."""
        if self.output_kind == "int":
            return raw if isinstance(raw, int) and not isinstance(raw, bool) else None
        if self.output_kind == "bool":
            return raw if isinstance(raw, bool) else None
        return raw if isinstance(raw, str) else None

    def fmt_output(self, value: Output) -> str:
        if self.output_kind == "bool":
            return "true" if value else "false"
        if self.output_kind == "str":
            return f'"{value}"'
        return str(value)

    def render(self, node: Node) -> str:
        text = self.grammar.render(node)
        return text[1:-1] if text.startswith("(") and text.endswith(")") else text


def _broadcast(vec: Any, n: int) -> list[Any]:
    if isinstance(vec, list):
        return vec
    arr = np.asarray(vec)
    if arr.ndim == 0:
        return [arr.item()] * n
    return list(arr.tolist())


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ---------------------------------------------------------------- integer families


def _text(t: str) -> Callable[[], str]:
    return lambda: t


def _column(c: int) -> Callable[[Any], Any]:
    return lambda b: b[c]


def _constant(k: int) -> Callable[[Any], Any]:
    return lambda b: np.full(b.shape[1], k, dtype=np.int64)


def _mod(k: int) -> Callable[[Any], Any]:
    return lambda a: np.mod(a, k)


def _div(k: int) -> Callable[[Any], Any]:
    return lambda a: np.floor_divide(a, k)


def _suffix(k: int, word: str) -> Callable[[str], str]:
    return lambda a: f"({a} {word} {k})"


def _int_ops(variables: dict[str, int], consts: Sequence[int]) -> list[Op]:
    """Ops shared by int1 and int2. `variables` maps a name to its column in the batch."""
    ops: list[Op] = []
    for name, col in variables.items():
        ops.append(Op(name, 0, 1, _column(col), _text(name)))
    for c in consts:
        ops.append(Op(f"c{c}", 0, 1, _constant(c), _text(str(c))))

    def binop(name: str, f: Callable[[Any, Any], Any], sym: str, comm: bool) -> Op:
        return Op(name, 2, 1, f, lambda a, b: f"({a} {sym} {b})", comm)

    def fnop(name: str, f: Callable[[Any, Any], Any], comm: bool) -> Op:
        return Op(name, 2, 1, f, lambda a, b: f"{name}({a}, {b})", comm)

    ops += [
        binop("add", np.add, "+", True),
        binop("sub", np.subtract, "-", False),
        binop("mul", np.multiply, "*", True),
        fnop("min", np.minimum, True),
        fnop("max", np.maximum, True),
        binop("and", np.bitwise_and, "&", True),
        binop("or", np.bitwise_or, "|", True),
        binop("xor", np.bitwise_xor, "^", True),
        Op("abs", 1, 1, np.abs, lambda a: f"abs({a})"),
    ]
    for k in (2, 3, 5, 7, 10):
        ops.append(Op(f"mod{k}", 1, 2, _mod(k), _suffix(k, "mod")))
    for k in (2, 3):
        ops.append(Op(f"div{k}", 1, 2, _div(k), _suffix(k, "div")))
    return ops


def _ite(name: str, test: Callable[[Any], Any], text: str, batch_ref: list[Any]) -> Op:
    """`if <test> then A else B`. The test reads the raw inputs, so it is bound late."""

    def vec(a: Any, b: Any) -> Any:
        cond = test(batch_ref[0])
        return np.where(cond, a, b)

    return Op(name, 2, 3, vec, lambda a, b: f"(if {text} then {a} else {b})")


class _BatchRef:
    """Holds the current batch so that `ite` tests can see the raw inputs."""

    def __init__(self) -> None:
        self.cell: list[Any] = [None]

    def set(self, batch: Any) -> Any:
        self.cell[0] = batch
        return batch


SIZE_NOTE = (
    "Rules are small. Counting each variable, constant and operation as one piece, "
    '"mod k" and "div k" as two and an if-test as three, a rule has at most {n} pieces.'
)

INT_PRIMITIVES = (
    "integer constants {consts}; + - * ; min(a, b), max(a, b), abs(a); "
    "a mod k for k in {{2, 3, 5, 7, 10}} (the result has the sign of k, so it is in [0, k)); "
    "a div k for k in {{2, 3}} (floor division, rounding towards minus infinity); "
    "bitwise a & b, a | b, a ^ b on two's-complement integers, so negative numbers work "
    "as they do in Python; and piecewise rules 'if <test> then A else B' where the test is "
    "one of: {tests}"
)


def int1() -> Family:
    consts = (1, 2, 3, 5, 10)
    ref = _BatchRef()
    tests = {
        "x < 0": lambda b: b[0] < 0,
        "x < 5": lambda b: b[0] < 5,
        "x < 10": lambda b: b[0] < 10,
        "x is even": lambda b: b[0] % 2 == 0,
    }
    ops = _int_ops({"x": 0}, consts)
    ops += [_ite(f"if{i}", t, txt, ref.cell) for i, (txt, t) in enumerate(tests.items())]
    domain = list(range(-50, 51))
    return Family(
        name="int1",
        title="integer function f(x)",
        grammar=Grammar(ops, valid=_intv, key=_int_key),
        max_size=7,
        tiers=(Tier(1, 3, 3), Tier(2, 4, 5), Tier(3, 6, 7)),
        probe=domain,
        batch=lambda xs: ref.set(np.array([list(xs)], dtype=np.int64)),
        output_kind="int",
        describe=(
            "The box computes an integer function f(x) of one integer x, "
            "with -50 <= x <= 50. The rule is a small expression built from: x; "
            + INT_PRIMITIVES.format(consts=", ".join(map(str, consts)), tests="; ".join(tests))
            + ". "
            + SIZE_NOTE.format(n=7)
        ),
        query_format="an integer x with -50 <= x <= 50, for example 7",
        output_format="an integer",
        easy=[0, 1, 2, 3, 4, 5],
        parse_query=lambda q: q if _is_int(q) and -50 <= q <= 50 else None,
        fmt_input=lambda q: f"f({q})",
    )


def int2() -> Family:
    consts = (1, 2, 3)
    ref = _BatchRef()
    tests = {
        "x < y": lambda b: b[0] < b[1],
        "x == y": lambda b: b[0] == b[1],
        "x < 0": lambda b: b[0] < 0,
        "y < 0": lambda b: b[1] < 0,
        "x is even": lambda b: b[0] % 2 == 0,
    }
    ops = _int_ops({"x": 0, "y": 1}, consts)
    ops += [_ite(f"if{i}", t, txt, ref.cell) for i, (txt, t) in enumerate(tests.items())]
    domain = [(x, y) for x in range(-20, 21) for y in range(-20, 21)]

    def parse(q: Any) -> Input | None:
        if isinstance(q, list) and len(q) == 2 and all(_is_int(v) for v in q):
            x, y = q
            if -20 <= x <= 20 and -20 <= y <= 20:
                return (x, y)
        return None

    return Family(
        name="int2",
        title="integer function f(x, y)",
        grammar=Grammar(ops, valid=_intv, key=_int_key),
        max_size=6,
        tiers=(Tier(1, 3, 3), Tier(2, 4, 5), Tier(3, 6, 6)),
        probe=domain,
        batch=lambda qs: ref.set(np.array(list(zip(*qs, strict=True)), dtype=np.int64)),
        output_kind="int",
        describe=(
            "The box computes an integer function f(x, y) of two integers, "
            "with -20 <= x <= 20 and -20 <= y <= 20. The rule is a small expression built "
            "from: x, y; "
            + INT_PRIMITIVES.format(consts=", ".join(map(str, consts)), tests="; ".join(tests))
            + ". "
            + SIZE_NOTE.format(n=6)
        ),
        query_format="a pair [x, y] with each value between -20 and 20, for example [3, -4]",
        output_format="an integer",
        easy=[(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)],
        parse_query=parse,
        fmt_input=lambda q: f"f({q[0]}, {q[1]})",
    )


# ---------------------------------------------------------------- Wason triples

TRIPLE_PROTOTYPES: tuple[tuple[int, int, int], ...] = (
    (2, 4, 6),
    (1, 2, 3),
    (3, 6, 9),
    (2, 4, 8),
    (4, 8, 12),
    (1, 3, 5),
    (5, 10, 15),
    (10, 20, 30),
    (6, 4, 2),
    (3, 2, 1),
    (2, 2, 2),
    (5, 5, 5),
    (1, 1, 2),
    (3, 4, 7),
    (2, 3, 5),
    (6, 8, 10),
    (9, 6, 3),
    (1, 4, 9),
    (8, 4, 2),
    (12, 10, 8),
    (0, 0, 0),
    (7, 7, 14),
    (4, 6, 2),
    (5, 3, 1),
    (2, 6, 4),
    (10, 5, 0),
    (20, 10, 30),
    (9, 3, 6),
    (7, 14, 21),
    (3, 9, 27),
    (1, 5, 10),
    (8, 2, 6),
    (2, 5, 8),
    (11, 13, 17),
    (4, 4, 8),
    (6, 3, 9),
    (10, 1, 5),
)


def triple() -> Family:
    def a(b: Any) -> Any:
        return b[0]

    def bb(b: Any) -> Any:
        return b[1]

    def c(b: Any) -> Any:
        return b[2]

    atoms: dict[str, Callable[[Any], Any]] = {
        "a < b": lambda t: a(t) < bb(t),
        "b < c": lambda t: bb(t) < c(t),
        "a < c": lambda t: a(t) < c(t),
        "b < a": lambda t: bb(t) < a(t),
        "c < b": lambda t: c(t) < bb(t),
        "c < a": lambda t: c(t) < a(t),
        "a = b": lambda t: a(t) == bb(t),
        "b = c": lambda t: bb(t) == c(t),
        "a = c": lambda t: a(t) == c(t),
        "a is even": lambda t: a(t) % 2 == 0,
        "b is even": lambda t: bb(t) % 2 == 0,
        "c is even": lambda t: c(t) % 2 == 0,
        "a + b + c is even": lambda t: (a(t) + bb(t) + c(t)) % 2 == 0,
        "a + b = c": lambda t: a(t) + bb(t) == c(t),
        "b - a = c - b": lambda t: bb(t) - a(t) == c(t) - bb(t),
        "b = 2a": lambda t: bb(t) == 2 * a(t),
        "c = 2b": lambda t: c(t) == 2 * bb(t),
        "a + b + c > 20": lambda t: a(t) + bb(t) + c(t) > 20,
        "a + b + c > 40": lambda t: a(t) + bb(t) + c(t) > 40,
        "a < 10": lambda t: a(t) < 10,
        "b < 10": lambda t: bb(t) < 10,
        "c < 10": lambda t: c(t) < 10,
        "a > 20": lambda t: a(t) > 20,
        "c > 20": lambda t: c(t) > 20,
        "a is a multiple of 3": lambda t: a(t) % 3 == 0,
        "b is a multiple of 3": lambda t: bb(t) % 3 == 0,
        "c is a multiple of 3": lambda t: c(t) % 3 == 0,
        "b is the largest": lambda t: (bb(t) > a(t)) & (bb(t) > c(t)),
        "c is the largest": lambda t: (c(t) > a(t)) & (c(t) > bb(t)),
        "a is the largest": lambda t: (a(t) > bb(t)) & (a(t) > c(t)),
    }
    ops: list[Op] = [Op(f"p{i}", 0, 1, f, _text(s)) for i, (s, f) in enumerate(atoms.items())]
    ops += [
        Op("not", 1, 1, np.logical_not, lambda x: f"not ({x})"),
        Op("and", 2, 1, np.logical_and, lambda x, y: f"({x} and {y})", True),
        Op("or", 2, 1, np.logical_or, lambda x, y: f"({x} or {y})", True),
    ]
    rng = random.Random(20261103)
    probe: list[tuple[int, int, int]] = list(TRIPLE_PROTOTYPES)
    for start in range(0, 31, 2):
        for d in (-6, -3, -2, -1, 1, 2, 3, 5, 7):
            t = (start, start + d, start + 2 * d)
            if all(0 <= v <= 30 for v in t):
                probe.append(t)
    for x in range(0, 31, 3):
        probe += [(x, x, x), (x, x, (x + 7) % 31), (x, (x + 4) % 31, x), ((x + 9) % 31, x, x)]
        if 2 * x <= 30:
            probe.append((x, 2 * x, min(30, 4 * x)))
    while len(probe) < 2500:
        probe.append((rng.randint(0, 30), rng.randint(0, 30), rng.randint(0, 30)))
    probe = list(dict.fromkeys(probe))

    def parse(q: Any) -> Input | None:
        if (
            isinstance(q, list)
            and len(q) == 3
            and all(_is_int(v) for v in q)
            and all(0 <= v <= 30 for v in q)
        ):
            return (q[0], q[1], q[2])
        return None

    def valid(vec: Any) -> bool:
        rate = float(np.mean(vec))
        return 0.0 < rate < 1.0

    return Family(
        name="triple",
        title="rule over triples of numbers",
        grammar=Grammar(ops, valid=valid, key=_bool_key),
        max_size=5,
        tiers=(Tier(1, 1, 2), Tier(2, 3, 3), Tier(3, 4, 5)),
        probe=probe,
        batch=lambda qs: np.array(list(zip(*qs, strict=True)), dtype=np.int64),
        output_kind="bool",
        describe=(
            "The box takes a triple of integers [a, b, c], each between 0 and 30, and says "
            "true if the triple fits a hidden rule and false if it does not. The rule is "
            "built from these conditions, combined with and, or and not: "
            + "; ".join(atoms)
            + ". The rule is small: it uses at most three conditions, and the whole "
            "expression has at most five parts, counting each condition, and, or and not "
            "as one."
        ),
        query_format="a triple [a, b, c] with each value between 0 and 30, for example [1, 2, 3]",
        output_format="true or false",
        easy=list(TRIPLE_PROTOTYPES[:6]),
        parse_query=parse,
        fmt_input=lambda q: f"[{q[0]}, {q[1]}, {q[2]}]",
        prototypes=TRIPLE_PROTOTYPES,
    )


# ---------------------------------------------------------------- strings

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
VOWELS = set("aeiou")


def _shift(s: str, k: int) -> str:
    return "".join(ALPHABET[(ALPHABET.index(ch) + k) % 26] for ch in s)


def _dedup(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _swap_pairs(s: str) -> str:
    chars = list(s)
    for i in range(0, len(chars) - 1, 2):
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


STRING_UNARY: dict[str, tuple[Callable[[str], str], str]] = {
    "reverse": (lambda s: s[::-1], "reverse the string"),
    "sort": (lambda s: "".join(sorted(s)), "sort the letters alphabetically"),
    "rotl": (lambda s: s[1:] + s[:1], "move the first letter to the end"),
    "rotr": (lambda s: s[-1:] + s[:-1], "move the last letter to the front"),
    "shift1": (lambda s: _shift(s, 1), "replace each letter with the next one (z wraps to a)"),
    "shift2": (lambda s: _shift(s, 2), "shift each letter forward by 2 (y to a, z to b)"),
    "unshift1": (lambda s: _shift(s, -1), "replace each letter with the previous one (a to z)"),
    "dropfirst": (lambda s: s[1:], "drop the first letter"),
    "droplast": (lambda s: s[:-1], "drop the last letter"),
    "double": (lambda s: "".join(ch + ch for ch in s), "double every letter (ab to aabb)"),
    "evens": (lambda s: s[::2], "keep letters at positions 1, 3, 5, ... (abcde to ace)"),
    "dedup": (_dedup, "collapse runs of the same letter (aabccc to abc)"),
    "swappairs": (_swap_pairs, "swap letters in pairs (abcde to badce)"),
    "novowels": (lambda s: "".join(ch for ch in s if ch not in VOWELS), "remove vowels"),
    "unique": (lambda s: "".join(dict.fromkeys(s)), "keep only the first copy of each letter"),
}


def _lift(f: Callable[[str], str]) -> Callable[[list[str]], list[str]]:
    return lambda v: [f(x) for x in v]


def _call(name: str) -> Callable[[str], str]:
    return lambda a: f"{name}({a})"


def string() -> Family:
    ops: list[Op] = [Op("s", 0, 1, list, lambda: "s")]
    for name, (f, _desc) in STRING_UNARY.items():
        ops.append(Op(name, 1, 1, _lift(f), _call(name)))
    ops.append(
        Op(
            "concat",
            2,
            1,
            lambda u, v: [x + y for x, y in zip(u, v, strict=True)],
            lambda a, b: f"concat({a}, {b})",
        )
    )
    rng = random.Random(20261104)
    probe: list[str] = [
        "a",
        "b",
        "z",
        "ab",
        "ba",
        "aa",
        "az",
        "abc",
        "cba",
        "aab",
        "abb",
        "aba",
        "abba",
        "abcd",
        "dcba",
        "hello",
        "banana",
        "zebra",
        "apple",
        "aaaa",
        "xyz",
        "abcdefgh",
        "hgfedcba",
        "aabbcc",
        "mississi",
        "queue",
        "zzz",
        "yza",
        "eiou",
        "bcdfg",
    ]
    for _ in range(140):
        n = rng.randint(1, 8)
        probe.append("".join(rng.choice(ALPHABET) for _ in range(n)))
    for _ in range(80):
        n = rng.randint(2, 8)
        probe.append("".join(rng.choice("abce") for _ in range(n)))
    probe = list(dict.fromkeys(probe))

    def parse(q: Any) -> Input | None:
        if isinstance(q, str) and 1 <= len(q) <= 8 and all(ch in ALPHABET for ch in q):
            return q
        return None

    lines = "; ".join(f"{name}: {desc}" for name, (_f, desc) in STRING_UNARY.items())
    return Family(
        name="string",
        title="string transform",
        grammar=Grammar(ops, valid=lambda _v: True, key=_str_key),
        max_size=5,
        tiers=(Tier(1, 3, 3), Tier(2, 4, 4), Tier(3, 5, 5)),
        probe=probe,
        batch=list,
        output_kind="str",
        describe=(
            "The box takes a string s of 1 to 8 lowercase letters (a to z) and returns "
            "a string, possibly empty. The rule applies one or more of these operations in "
            "sequence: "
            + lines
            + ". It can also join two results end to end: concat(A, B), for example "
            "concat(s, reverse(s)) turns ab into abba. Counting s itself, each "
            "operation and each concat as one part, a rule has two to five parts, so at most "
            "four operations."
        ),
        query_format='a string of 1 to 8 lowercase letters, for example "cab"',
        output_format="a string",
        easy=["a", "ab", "abc", "abcd"],
        parse_query=parse,
        fmt_input=lambda q: f'f("{q}")',
    )


FAMILIES: dict[str, Callable[[], Family]] = {
    "int1": int1,
    "int2": int2,
    "triple": triple,
    "string": string,
}


def get(name: str) -> Family:
    try:
        return FAMILIES[name]()
    except KeyError:
        raise ValueError(f"unknown family {name!r}; choose from {', '.join(FAMILIES)}") from None
