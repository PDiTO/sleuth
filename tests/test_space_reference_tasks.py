from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from sleuth import reference, tasks
from sleuth import space as spaces
from sleuth.space import entropy_by_column, partition_entropy


@pytest.fixture(scope="module")
def int1_small() -> spaces.Space:
    return spaces.build("int1", 4, cache=False)


@pytest.fixture(scope="module")
def triple_small() -> spaces.Space:
    return spaces.build("triple", 3, cache=False)


def test_entropy_by_column_matches_unique_counts() -> None:
    rng = np.random.default_rng(0)
    m = rng.integers(0, 4, size=(50, 7))
    got = entropy_by_column(m)
    want = [partition_entropy(m[:, j]) for j in range(7)]
    assert np.allclose(got, want)
    assert entropy_by_column(m[:1]).tolist() == [0.0] * 7
    assert partition_entropy(np.array([5])) == 0.0


def test_space_filtering(int1_small: spaces.Space) -> None:
    sp = int1_small
    x = next(i for i, p in enumerate(sp.programs) if p.op == "x")
    vs = sp.filter([(3, 3), (-4, -4)])
    assert x in vs.tolist()
    assert sp.simplest(vs) == x
    assert sp.outputs(vs[:1], 7) == [7]
    assert sp.simplest(np.array([], dtype=np.int64)) == -1


def test_space_off_probe_queries_are_evaluated(triple_small: spaces.Space) -> None:
    sp = triple_small
    q = (29, 1, 17)
    assert q not in sp.probe_index
    idx = sp.all[:50]
    outs = sp.outputs(idx, q)
    assert len(outs) == 50
    kept = sp.consistent(idx, q, True)
    assert all(sp.outputs(kept, q))
    assert len(sp.codes(idx, q)) == 50


def test_string_space_decodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    sp = spaces.build("string", 2)
    again = spaces.build("string", 2)  # from the on-disk cache
    assert len(sp) == len(again)
    rev = next(i for i, p in enumerate(sp.programs) if p.op == "reverse")
    assert sp.outputs(np.array([rev]), "abc") == ["cba"]
    assert sp.outputs(np.array([rev]), "zyxw") == ["wxyz"]
    assert sp.encode("not-a-seen-output") == -1


def test_active_learner_beats_random_on_triples(triple_small: spaces.Space) -> None:
    sp = triple_small
    ts, cal = tasks.generate(sp, 2, 3, seed=5)
    assert ts
    assert cal.accepted == len(ts)
    for t in ts:
        act = reference.run(
            sp,
            t.rule,
            t.test_inputs,
            t.test_outputs,
            agent="active",
            budget=12,
            seed=1,
            given=t.given,
        )
        assert act.steps[0].eig == pytest.approx(act.steps[0].best_eig, abs=0.05)
        assert act.solved_at() is not None
        assert act.vs_sizes[0] >= act.vs_sizes[-1]
        rnd = reference.run(
            sp,
            t.rule,
            t.test_inputs,
            t.test_outputs,
            agent="random",
            budget=12,
            seed=1,
            given=t.given,
        )
        assert sum(s.eig for s in act.steps[:3]) >= sum(s.eig for s in rnd.steps[:3]) - 1e-9
    with pytest.raises(ValueError, match="unknown reference agent"):
        reference.run(
            sp, ts[0].rule, ts[0].test_inputs, ts[0].test_outputs, agent="oracle", budget=1, seed=0
        )


def test_step_bits_and_identified() -> None:
    s = reference.Step(1, 1, 8, 2, 1.0, 1.0)
    assert s.bits == 2.0
    assert reference.Step(1, 1, 8, 0, 1.0, 1.0).bits == 0.0
    run = reference.RefRun("x", vs_sizes=[5, 3, 1], accuracy=[0.1, 0.5, 1.0])
    assert run.identified_at() == 2
    assert run.solved_at() == 2
    assert reference.RefRun("x", vs_sizes=[5], accuracy=[0.0]).identified_at() is None


def test_generated_tasks_are_deterministic_and_well_formed(
    triple_small: spaces.Space, tmp_path: Path
) -> None:
    a, _ = tasks.generate(triple_small, 1, 3, seed=9)
    b, _ = tasks.generate(triple_small, 1, 3, seed=9)
    assert [t.to_json() for t in a] == [t.to_json() for t in b]
    for t in a:
        assert len(t.test_inputs) == tasks.TEST_SIZE
        assert len(set(t.test_inputs)) == tasks.TEST_SIZE
        assert triple_small.family.tier_of(t.size) == 1
        assert t.given
        assert t.given[0][1] is True
        assert sum(t.test_outputs) >= 3
        assert sum(not o for o in t.test_outputs) >= 3
    path = tmp_path / "tasks.json"
    tasks.save(a, path)
    assert tasks.load(path) == a


def test_test_set_separates_near_misses(int1_small: spaces.Space) -> None:
    sp = int1_small
    h = next(i for i, p in enumerate(sp.programs) if sp.render(i) == "x mod 3")
    inputs = tasks.test_set(sp, h, [], random.Random(0))
    truth = sp.family.evaluate(sp.programs[h], inputs)
    # x mod 3 agrees with x on 0, 1, 2; the test set must catch that near miss
    x = next(i for i, p in enumerate(sp.programs) if p.op == "x")
    assert sp.family.evaluate(sp.programs[x], inputs) != truth
    assert tasks.degenerate(sp, next(i for i, p in enumerate(sp.programs) if p.op == "c1"))
