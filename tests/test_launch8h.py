"""The 8H launcher is the one automatic decision-maker in the chain, so its rule is pinned."""
import json
import pathlib
import tempfile

import pytest

from g2l.launch8h import REQUIRED, select


def write(rows: pathlib.Path, name: str, vals):
    rows.joinpath(f"{name}.json").write_text(json.dumps(
        {"arm": name, "seeds": list(range(len(vals))), "val_rocauc": vals,
         "val_mean": sum(vals) / len(vals)}))


def test_picks_the_width_with_the_highest_validation_mean():
    with tempfile.TemporaryDirectory() as td:
        rows = pathlib.Path(td)
        write(rows, "lgm_d64_hlinear", [0.81, 0.82, 0.80])
        write(rows, "lgm_d128_hlinear", [0.83, 0.82, 0.84])     # best mean
        write(rows, "lgm_d256_hlinear", [0.85, 0.78, 0.80])     # best single seed, worse mean
        write(rows, "gin_pretrained", [0.83, 0.83, 0.84])
        write(rows, "gcn_pretrained", [0.83, 0.83, 0.83])
        w = select(rows)
        assert w["lgm_d"] == 128
        assert set(w["lgm_val_by_d"]) == {64, 128, 256}


def test_refuses_on_a_missing_or_incomplete_row():
    with tempfile.TemporaryDirectory() as td:
        rows = pathlib.Path(td)
        for name in REQUIRED[:-1]:
            write(rows, name, [0.8, 0.8, 0.8])
        with pytest.raises(SystemExit):
            select(rows)                                    # gcn_pretrained missing
        write(rows, REQUIRED[-1], [0.8, 0.8, 0.8])
        select(rows)                                        # now complete
        rows.joinpath("gin_pretrained.json").write_text(json.dumps(
            {"arm": "gin_pretrained", "seeds": [0, 1, 2], "val_rocauc": [0.8], "val_mean": 0.8}))
        with pytest.raises(SystemExit):
            select(rows)                                    # one seed of three
