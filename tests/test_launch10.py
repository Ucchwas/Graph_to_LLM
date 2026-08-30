"""The Phase 10 launcher makes the chain's only decision, so its rule is pinned.

Per arm: the width with the highest validation mean over the selection seeds -- Phase 8H's rule,
applied identically to all six arms. It must refuse to submit anything if a row is missing or
incomplete, because a failed task has to be looked at rather than papered over.
"""
import json
import pathlib
import tempfile

import pytest
import yaml

from g2l.launch10 import select
from g2l.run_phase10 import ARMS, widths

CFG = yaml.safe_load(open("configs/phase10_molhiv.yaml"))


def write(rows: pathlib.Path, arm: str, d: int, vals):
    rows.joinpath(f"sel_{arm}_d{d}.json").write_text(json.dumps(
        {"arm": arm, "d": d, "seeds": list(range(len(vals))), "val_rocauc": vals,
         "val_mean": sum(vals) / len(vals)}))


def fill(rows: pathlib.Path, best: dict):
    """Every arm x width row present; `best` names the width that should win each arm."""
    for arm in ARMS:
        for d in widths(arm, CFG):
            write(rows, arm, d, [0.83, 0.83, 0.83] if d == best[arm] else [0.80, 0.80, 0.80])


def test_picks_each_arm_width_by_validation_mean_independently():
    best = {"lgm": 128, "lgm_rrwp": 256, "gin": 300, "gin_rwse": 100, "gcn": 200, "gcn_rwse": 300}
    with tempfile.TemporaryDirectory() as td:
        rows = pathlib.Path(td)
        fill(rows, best)
        w = select(rows, CFG)
        assert w["width"] == best
        assert set(w["val_by_width"]) == set(ARMS)
        for arm in ARMS:
            assert set(w["val_by_width"][arm]) == set(widths(arm, CFG))


def test_the_mean_decides_not_the_best_single_seed():
    with tempfile.TemporaryDirectory() as td:
        rows = pathlib.Path(td)
        fill(rows, {a: widths(a, CFG)[0] for a in ARMS})
        # give lgm d=256 the best single seed but a worse mean than d=64's flat 0.83
        write(rows, "lgm", 256, [0.90, 0.78, 0.79])
        assert select(rows, CFG)["width"]["lgm"] == widths("lgm", CFG)[0]


def test_refuses_on_a_missing_or_incomplete_row():
    with tempfile.TemporaryDirectory() as td:
        rows = pathlib.Path(td)
        fill(rows, {a: widths(a, CFG)[0] for a in ARMS})
        select(rows, CFG)                                   # complete: fine
        missing = rows / "sel_gcn_rwse_d200.json"
        body = missing.read_text()
        missing.unlink()
        with pytest.raises(SystemExit):
            select(rows, CFG)
        missing.write_text(body)
        select(rows, CFG)
        # a row with fewer results than seeds is also a refusal
        (rows / "sel_gin_d100.json").write_text(json.dumps(
            {"arm": "gin", "d": 100, "seeds": [0, 1, 2], "val_rocauc": [0.8], "val_mean": 0.8}))
        with pytest.raises(SystemExit):
            select(rows, CFG)


def test_the_grids_are_the_ones_the_slurm_array_walks():
    """18 stage-1/2 tasks = 6 arms x 3 widths, in the order the sbatch case statement assumes."""
    assert len(ARMS) == 6
    assert all(len(widths(a, CFG)) == 3 for a in ARMS)
    assert widths("lgm", CFG) == widths("lgm_rrwp", CFG) == CFG["lgm_widths"]
    assert widths("gin", CFG) == widths("gcn_rwse", CFG) == CFG["ogb_widths"]
    assert CFG["ogb_d"] in CFG["ogb_widths"], "OGB's reference width must stay reachable"
