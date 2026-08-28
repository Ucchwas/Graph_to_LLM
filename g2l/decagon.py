"""Decagon polypharmacy side effects (BioSNAP ChChSe-Decagon; Zitnik, Agrawal & Leskovec 2018):
645 drugs, 63,473 drug pairs, 1,317 side-effect types of which 963 have >= 500 pairs (the
paper's filter). One layer per side effect over the same 645 drug indices; layers are genuine
edge sets, not induced subgraphs (a layer holds ~10 % of the global pairs among its drugs).
Cached at <root>/Decagon/processed/decagon.pt; the raw csv is kept in raw/."""
import gzip
import os
import pathlib
import urllib.request

import torch

from g2l.datasets import root
from g2l.layers import split_layers

URL = "https://snap.stanford.edu/biodata/datasets/10017/files/ChChSe-Decagon_polypharmacy.csv.gz"
RAW = "ChChSe-Decagon_polypharmacy.csv.gz"


def load_decagon_raw() -> dict:
    d = pathlib.Path(root()) / "Decagon"
    out = d / "processed" / "decagon.pt"
    if out.exists():
        return torch.load(out)
    (d / "raw").mkdir(parents=True, exist_ok=True)
    if not (d / "raw" / RAW).exists():
        urllib.request.urlretrieve(URL, d / "raw" / RAW)
    drugs, by_se, names = {}, {}, {}
    with gzip.open(d / "raw" / RAW, "rt") as fh:
        next(fh)
        for line in fh:
            a, b, se, name = line.rstrip("\n").split(",", 3)
            if a == b:
                continue
            ia, ib = drugs.setdefault(a, len(drugs)), drugs.setdefault(b, len(drugs))
            by_se.setdefault(se, []).append((min(ia, ib), max(ia, ib)))
            names[se] = name
    # node index = rank of the STITCH id in sorted order, reproducible from the file alone
    ids = sorted(drugs)
    rank = {drugs[s]: k for k, s in enumerate(ids)}
    layers = {}
    for se, ps in by_se.items():
        e = torch.tensor([[rank[a] for a, _ in ps], [rank[b] for _, b in ps]], dtype=torch.int32)
        layers[se] = torch.unique(torch.stack([e.min(0).values, e.max(0).values]), dim=1)
    obj = {"drugs": ids, "names": names, "layers": layers}
    out.parent.mkdir(exist_ok=True)
    torch.save(obj, out)
    return obj


def load_decagon(min_pairs: int = 500, seed: int = 0, tau: float = 0.2, hidden_frac: float = 0.10):
    """(N, layers, hidden, info, names) with the Phase-6 splits applied."""
    o = load_decagon_raw()
    N = len(o["drugs"])
    pairs = {se: e.long() for se, e in o["layers"].items()}
    layers, hidden, info = split_layers(N, pairs, seed=seed, tau=tau, hidden_frac=hidden_frac, min_pairs=min_pairs)
    return N, layers, hidden, info, o["names"]
