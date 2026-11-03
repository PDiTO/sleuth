from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sleuth import families
from sleuth.dsl import Grammar, Op, TooManyPrograms, enumerate_programs


def _toy() -> Grammar:
    ops = [
        Op("x", 0, 1, lambda b: b, lambda: "x"),
        Op("one", 0, 1, np.ones_like, lambda: "1"),
        Op("add", 2, 1, np.add, lambda a, b: f"({a} + {b})", True),
        Op("neg", 1, 1, np.negative, lambda a: f"-{a}"),
    ]
    return Grammar(ops, key=lambda v: np.asarray(v).tobytes())


def test_enumeration_keeps_one_smallest_program_per_behaviour() -> None:
    g = _toy()
    batch = np.arange(-3, 4)
    programs, vecs = enumerate_programs(g, batch, 4)
    keys = [v.tobytes() for v in vecs]
    assert len(keys) == len(set(keys))
    sizes = [p.size for p in programs]
    assert sizes == sorted(sizes)
    # x + x and 1 + 1 are size 3; -(-x) is size 3 but equals x, so it is not kept
    rendered = {g.render(p) for p in programs}
    assert "(x + x)" in rendered
    assert "--x" not in rendered


def test_enumeration_limit() -> None:
    with pytest.raises(TooManyPrograms):
        enumerate_programs(_toy(), np.arange(-3, 4), 6, limit=5)


def test_grammar_node_checks_arity() -> None:
    g = _toy()
    with pytest.raises(ValueError, match="takes 2"):
        g.node("add", g.node("x"))
    n = g.node("add", g.node("x"), g.node("one"))
    assert n.size == 3
    assert list(g.evaluate(n, np.array([1, 2]))) == [2, 3]


def test_int_semantics_match_python() -> None:
    fam = families.int1()
    g = fam.grammar
    x = g.node("x")
    cases: list[tuple[Any, Any]] = [
        (g.node("mod3", x), lambda v: v % 3),
        (g.node("div2", x), lambda v: v // 2),
        (g.node("xor", x, g.node("c5")), lambda v: v ^ 5),
        (g.node("and", x, g.node("c10")), lambda v: v & 10),
        (g.node("if0", g.node("c1"), x), lambda v: 1 if v < 0 else v),
    ]
    xs = list(range(-50, 51))
    for node, f in cases:
        assert fam.evaluate(node, xs) == [f(v) for v in xs]


def test_int2_and_triple_and_string_evaluate() -> None:
    i2 = families.int2()
    g = i2.grammar
    node = g.node("sub", g.node("x"), g.node("y"))
    assert i2.evaluate(node, [(3, 5), (-2, -7)]) == [-2, 5]

    tr = families.triple()
    p = next(op.name for op in tr.grammar.ops if op.show() == "a < b")
    q = next(op.name for op in tr.grammar.ops if op.show() == "b < c")
    inc = tr.grammar.node("and", tr.grammar.node(p), tr.grammar.node(q))
    assert tr.evaluate(inc, [(1, 2, 3), (3, 2, 1), (1, 1, 2)]) == [True, False, False]
    assert tr.render(inc) == "a < b and b < c"

    st = families.string()
    sg = st.grammar
    s = sg.node("s")
    assert st.evaluate(sg.node("concat", s, sg.node("reverse", s)), ["ab"]) == ["abba"]
    assert st.evaluate(sg.node("shift1", s), ["az"]) == ["ba"]
    assert st.evaluate(sg.node("dedup", s), ["aabccc"]) == ["abc"]
    assert st.evaluate(sg.node("swappairs", s), ["abcde"]) == ["badce"]
    assert st.evaluate(sg.node("evens", s), ["abcde"]) == ["ace"]
    assert st.evaluate(sg.node("unique", s), ["banana"]) == ["ban"]
    assert st.evaluate(sg.node("novowels", s), ["queue"]) == ["q"]


@pytest.mark.parametrize(
    ("name", "good", "bad"),
    [
        ("int1", [7, -50, 50], [51, 7.0, True, "7", [7]]),
        ("int2", [[3, -4], [20, 20]], [[3], [3, 21], (3, 4), [3, True], [3.0, 4]]),
        ("triple", [[1, 2, 3], [0, 30, 0]], [[1, 2], [1, 2, 31], [-1, 2, 3], [1, 2, "3"]]),
        ("string", ["cab", "a", "abcdefgh"], ["", "Cab", "abcdefghi", "a b", 3]),
    ],
)
def test_parse_query_is_strict(name: str, good: list[Any], bad: list[Any]) -> None:
    fam = families.get(name)
    for q in good:
        assert fam.parse_query(q) is not None
    for q in bad:
        assert fam.parse_query(q) is None


def test_parse_output_is_strict() -> None:
    assert families.int1().parse_output(3) == 3
    assert families.int1().parse_output(True) is None
    assert families.int1().parse_output(3.0) is None
    assert families.triple().parse_output(True) is True
    assert families.triple().parse_output(1) is None
    assert families.string().parse_output("ab") == "ab"
    assert families.string().parse_output(None) is None


def test_formatting_and_unknown_family() -> None:
    assert families.triple().fmt_output(True) == "true"
    assert families.string().fmt_output("ab") == '"ab"'
    assert families.int2().fmt_input((1, 2)) == "f(1, 2)"
    assert families.int1().tier_of(3) == 1
    assert families.int1().tier_of(99) is None
    with pytest.raises(ValueError, match="unknown family"):
        families.get("nope")


def test_descriptions_name_every_primitive() -> None:
    st = families.string()
    for name in families.STRING_UNARY:
        assert name in st.describe
    tr = families.triple()
    for op in tr.grammar.ops:
        if op.arity == 0:
            assert op.show() in tr.describe
