"""A tiny expression DSL and a bottom-up enumerator with observational equivalence.

Every rule family is a set of `Op`s. A program is a tree of `Node`s, and its size is the
sum of its ops' costs. Evaluation is vectorised: a leaf turns a batch of inputs into a
vector of outputs, and every other op combines its children's vectors. The same code
evaluates one program on a probe set during enumeration and on a single query later.

`enumerate_programs` builds programs smallest first and keeps one program per distinct
behaviour on the probe set, so the first program kept for a behaviour is a smallest one.
That gives each behaviour a well-defined minimum size, which is what the tiers use.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

Vec = Any  # a numpy array, or a list of str for the string family


@dataclass(frozen=True)
class Op:
    name: str
    arity: int
    cost: int
    vec: Callable[..., Vec]
    show: Callable[..., str]
    commutative: bool = False


@dataclass(frozen=True)
class Node:
    op: str
    args: tuple[Node, ...] = ()
    size: int = 1

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.op if not self.args else f"{self.op}({', '.join(map(str, self.args))})"


@dataclass
class Grammar:
    ops: Sequence[Op]
    valid: Callable[[Vec], bool] = field(default=lambda _v: True)
    key: Callable[[Vec], bytes] = field(default=bytes)
    _by_name: dict[str, Op] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_name = {op.name: op for op in self.ops}

    def op(self, name: str) -> Op:
        return self._by_name[name]

    def evaluate(self, node: Node, batch: Any) -> Vec:
        op = self._by_name[node.op]
        if op.arity == 0:
            return op.vec(batch)
        return op.vec(*(self.evaluate(a, batch) for a in node.args))

    def render(self, node: Node) -> str:
        op = self._by_name[node.op]
        return op.show(*(self.render(a) for a in node.args))

    def node(self, name: str, *args: Node) -> Node:
        op = self._by_name[name]
        if len(args) != op.arity:
            raise ValueError(f"{name} takes {op.arity} arguments, got {len(args)}")
        return Node(name, tuple(args), op.cost + sum(a.size for a in args))


class TooManyPrograms(RuntimeError):
    pass


def digest(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=12).digest()


def enumerate_programs(
    grammar: Grammar, batch: Any, max_size: int, *, limit: int = 2_000_000
) -> tuple[list[Node], list[Vec]]:
    """All behaviours up to `max_size`, one smallest program each, in a fixed order."""
    programs: list[Node] = []
    vecs: list[Vec] = []
    by_size: dict[int, list[int]] = {}
    seen: set[bytes] = set()

    def add(node: Node, vec: Vec) -> None:
        if not grammar.valid(vec):
            return
        k = digest(grammar.key(vec))
        if k in seen:
            return
        seen.add(k)
        by_size.setdefault(node.size, []).append(len(programs))
        programs.append(node)
        vecs.append(vec)
        if len(programs) > limit:
            raise TooManyPrograms(f"more than {limit} programs by size {node.size}")

    for size in range(1, max_size + 1):
        for op in grammar.ops:
            rest = size - op.cost
            if op.arity == 0:
                if rest == 0:
                    add(Node(op.name, (), size), op.vec(batch))
            elif op.arity == 1:
                for i in list(by_size.get(rest, [])):
                    add(Node(op.name, (programs[i],), size), op.vec(vecs[i]))
            elif op.arity == 2:
                for left in range(1, rest):
                    right = rest - left
                    if op.commutative and left > right:
                        continue
                    lefts = list(by_size.get(left, []))
                    rights = list(by_size.get(right, []))
                    for i in lefts:
                        for j in rights:
                            if op.commutative and left == right and j < i:
                                continue
                            node = Node(op.name, (programs[i], programs[j]), size)
                            add(node, op.vec(vecs[i], vecs[j]))
            else:  # pragma: no cover - no family uses arity > 2
                raise ValueError(f"unsupported arity {op.arity}")
    return programs, vecs
