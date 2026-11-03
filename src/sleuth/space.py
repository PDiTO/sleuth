"""The enumerated hypothesis space of a family, and the version space inside it.

`Space` holds one smallest program per behaviour and a signature matrix: row h, column p
is hypothesis h's output on probe input p, encoded as an integer. Updating the version
space after an observation on a probe input is one column comparison. Inputs outside the
probe set (a model can ask anything in the domain) are evaluated program by program.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import families
from .dsl import Node, enumerate_programs
from .families import Family, Input, Output

Ids = npt.NDArray[np.int64]
CACHE_VERSION = 3


@dataclass
class Space:
    family: Family
    programs: list[Node]
    sizes: npt.NDArray[np.int16]
    sig: npt.NDArray[Any]
    strings: dict[str, int] = field(default_factory=dict)
    probe_index: dict[Input, int] = field(init=False)
    decode: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.probe_index = {q: i for i, q in enumerate(self.family.probe)}
        self.decode = [""] * len(self.strings)
        for text, code in self.strings.items():
            self.decode[code] = text

    def __len__(self) -> int:
        return len(self.programs)

    @property
    def all(self) -> Ids:
        return np.arange(len(self.programs), dtype=np.int64)

    def encode(self, value: Output) -> int:
        if self.family.output_kind == "str":
            return self.strings.get(str(value), -1)
        return int(value)

    def outputs(self, idx: Ids, query: Input) -> list[Output]:
        """Outputs of hypotheses `idx` on `query`, as Python values."""
        col = self.probe_index.get(query)
        if col is not None:
            values = self.sig[idx, col].tolist()
            if self.family.output_kind == "str":
                return [self.decode[v] for v in values]
            return [self.family.to_python(v) for v in values]
        return [self.family.evaluate(self.programs[int(i)], [query])[0] for i in idx]

    def codes(self, idx: Ids, query: Input) -> Ids:
        """Outputs of hypotheses `idx` on `query`, encoded, for partitioning."""
        col = self.probe_index.get(query)
        if col is not None:
            return np.asarray(self.sig[idx, col], dtype=np.int64)
        outs = self.outputs(idx, query)
        if self.family.output_kind == "str":
            table: dict[str, int] = {}
            return np.array([table.setdefault(o, len(table)) for o in outs], dtype=np.int64)
        return np.array([int(o) for o in outs], dtype=np.int64)

    def consistent(self, idx: Ids, query: Input, output: Output) -> Ids:
        col = self.probe_index.get(query)
        if col is not None:
            mask = np.asarray(self.sig[idx, col] == self.encode(output), dtype=bool)
        else:
            mask = np.array([o == output for o in self.outputs(idx, query)], dtype=bool)
        kept: Ids = idx[mask]
        return kept

    def filter(self, observations: Sequence[tuple[Input, Output]]) -> Ids:
        idx = self.all
        for q, out in observations:
            idx = self.consistent(idx, q, out)
        return idx

    def simplest(self, idx: Ids) -> int:
        """The smallest program in `idx`, ties to enumeration order (already sorted)."""
        return int(idx[0]) if len(idx) else -1

    def render(self, h: int) -> str:
        return self.family.render(self.programs[h])


def _cache_dir() -> Path:
    return Path.cwd() / ".cache" / "sleuth"


def build(name: str, max_size: int | None = None, *, cache: bool = True) -> Space:
    """Enumerate a family's space, or load it from `.cache/sleuth/`."""
    fam = families.get(name)
    size = max_size if max_size is not None else fam.max_size
    tag = hashlib.sha256(f"{CACHE_VERSION}:{name}:{size}".encode()).hexdigest()[:12]
    path = _cache_dir() / f"{name}-{size}-{tag}.pkl"
    if cache and path.exists():
        with path.open("rb") as f:
            programs, sig, strings = pickle.load(f)
    else:
        programs, vecs = enumerate_programs(fam.grammar, fam.batch(fam.probe), size)
        strings = {}
        if fam.output_kind == "str":
            sig = np.array(
                [[strings.setdefault(s, len(strings)) for s in v] for v in vecs], dtype=np.int32
            )
        elif fam.output_kind == "bool":
            sig = np.array(vecs, dtype=np.int8)
        else:
            sig = np.array(vecs, dtype=np.int32)
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump((programs, sig, strings), f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
    sizes = np.array([p.size for p in programs], dtype=np.int16)
    return Space(fam, programs, sizes, sig, strings)


def entropy_by_column(m: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Shannon entropy (bits) of the value distribution in each column of `m`."""
    n, cols = m.shape
    if n <= 1:
        return np.zeros(cols)
    s = np.sort(m, axis=0)
    starts = np.ones((n, cols), dtype=bool)
    starts[1:] = s[1:] != s[:-1]
    flat = np.flatnonzero(starts.T.ravel())
    lengths = np.diff(np.append(flat, n * cols))
    col = flat // n
    p = lengths / n
    counts = np.bincount(col, weights=-p * np.log2(p), minlength=cols)
    return np.asarray(counts, dtype=np.float64)


def partition_entropy(codes: Ids) -> float:
    if len(codes) <= 1:
        return 0.0
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())
