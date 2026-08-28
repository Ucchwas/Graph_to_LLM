"""Phase 7: TCGA normal / tumour co-expression graph pairs from public Xena data.

Source: the UCSC Toil recompute `TcgaTargetGtex_rsem_gene_tpm` (log2(TPM+0.001), Ensembl gene
ids) with `TcgaTargetGTEX_phenotype.txt` -- one uniform pipeline over every TCGA cohort, which is
what makes cross-cancer comparison defensible. It is *uniformly processed*, NOT batch corrected:
cohort, donor and source effects remain, and nothing here claims otherwise.

Sample ids in every Xena TCGA hub are 15 characters (`TCGA-XX-XXXX-01`) -- participant plus
sample-type code, with the **vial letter dropped**. `11A` cannot be distinguished from `11B`
here; `-11` (Solid Tissue Normal) and `-01` (Primary Tumor) are the operational definitions, and
Xena already holds one column per (participant, sample type).

One aggregate graph per (cancer, condition). Individual patients are never graph pairs.
"""
import gzip
import hashlib
import os
import pathlib
import urllib.request
from collections import defaultdict

import numpy as np
import torch

HUB = "https://toil.xenahubs.net/download/"
EXPR = "TcgaTargetGtex_rsem_gene_tpm.gz"
PHENO = "TcgaTargetGTEX_phenotype.txt.gz"
TUMOUR_CODE, NORMAL_CODE = "01", "11"


def root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("G2L_PHASE7_ROOT", "data/phase7"))


def fetch(name: str) -> pathlib.Path:
    p = root() / "raw" / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HUB + name, p)
    return p


def sha256(p: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def phenotype() -> dict[str, dict]:
    """sample id -> {disease, sample_type, study}. TCGA rows only."""
    out = {}
    with gzip.open(fetch(PHENO), "rt", encoding="latin-1") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(hdr, line.rstrip("\n").split("\t")))
            if r.get("_study") == "TCGA":
                out[r["sample"]] = {"disease": r["detailed_category"], "sample_type": r["_sample_type"],
                                    "study": r["_study"]}
    return out


def cancer_table(min_normal: int = 30) -> list[dict]:
    """Per cancer: the `-11` normal and `-01` tumour samples and the participants having both.
    Included iff it has at least `min_normal` normals. Sorted by normal count, descending."""
    pheno = phenotype()
    by = defaultdict(lambda: {"normal": [], "tumour": []})
    for s, r in pheno.items():
        code = s[-2:]
        if code == NORMAL_CODE:
            by[r["disease"]]["normal"].append(s)
        elif code == TUMOUR_CODE:
            by[r["disease"]]["tumour"].append(s)
    rows = []
    for disease, d in by.items():
        n, t = sorted(d["normal"]), sorted(d["tumour"])
        matched = {s[:12] for s in n} & {s[:12] for s in t}
        rows.append({"disease": disease, "normal": n, "tumour": t, "n_normal": len(n),
                     "n_tumour": len(t), "n_matched": len(matched),
                     "included": len(n) >= min_normal})
    return sorted(rows, key=lambda r: -r["n_normal"])


def load_expression(samples: list[str], cache: str = "expr.pt") -> tuple[list[str], torch.Tensor]:
    """Stream the TPM matrix once, keeping only `samples` (columns). Returns (gene ids, [G, S]
    float32, log2(TPM+0.001) as published). Cached; the full file is never held in memory."""
    p = root() / "processed" / cache
    if p.exists():
        o = torch.load(p)
        if o["samples"] == samples:
            return o["genes"], o["expr"]
    want = {s: k for k, s in enumerate(samples)}
    genes, cols = [], []
    with gzip.open(fetch(EXPR), "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        idx = [(j, want[s]) for j, s in enumerate(hdr) if j and s in want]
        assert len(idx) == len(samples), f"{len(samples) - len(idx)} requested samples absent from the matrix"
        take = np.array([j for j, _ in idx])
        order = np.argsort([k for _, k in idx])
        for line in fh:
            f = line.rstrip("\n").split("\t")
            genes.append(f[0])
            v = np.array(f, dtype=object)[take].astype(np.float32)
            cols.append(v[order])
    expr = torch.from_numpy(np.vstack(cols))
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"genes": genes, "samples": samples, "expr": expr}, p)
    return genes, expr


def gene_universe(genes: list[str], expr: torch.Tensor, normal_cols: list[int], n_genes: int = 2000,
                  expr_floor: float = 1.0) -> list[int]:
    """Row indices of the fixed gene universe: expressed above `expr_floor` (median log2(TPM+0.001)
    across the pooled **normal** samples) then the top `n_genes` by variance.

    Uses normal expression only -- never tumour expression, and never the normal->tumour contrast
    that Phase 7 predicts. It does see the held-out cancer's normals, so the selection is mildly
    transductive; `docs/PLAN-PHASE7.md` records the per-fold alternative as a robustness check."""
    X = expr[:, normal_cols]
    keep = (X.median(dim=1).values > expr_floor).nonzero().flatten()
    var = X[keep].var(dim=1)
    return keep[var.argsort(descending=True)[:n_genes]].sort().values.tolist()


def spearman(X: torch.Tensor) -> torch.Tensor:
    """[G, S] -> [G, G] Spearman correlation across samples (rank transform + Pearson).
    numpy/torch only: scipy is unavailable on the laptop (blocked scipy.linalg DLL)."""
    r = X.argsort(dim=1).argsort(dim=1).float()
    r = r - r.mean(dim=1, keepdim=True)
    r = r / r.norm(dim=1, keepdim=True).clamp(min=1e-12)
    C = r @ r.T
    C.fill_diagonal_(0.0)
    return C


def binarise(C: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the `density` fraction of strict-upper-triangle pairs with the largest |correlation|.
    Density is matched between conditions by construction, so the normal->tumour task is edge
    rewiring rather than a change in edge count."""
    G = C.shape[0]
    iu = torch.triu_indices(G, G, offset=1)
    w = C[iu[0], iu[1]].abs()
    k = max(1, int(round(density * w.numel())))
    # exactly k edges: a |correlation| threshold would admit ties, and rank correlations over a
    # few dozen samples tie often -- unequal edge counts would break the matched-density design.
    sel = torch.argsort(w, descending=True, stable=True)[:k]
    A = torch.zeros(G, G)
    A[iu[0][sel], iu[1][sel]] = 1.0
    return A + A.T


def condition_graph(expr: torch.Tensor, gene_idx: list[int], cols: list[int], density: float) -> torch.Tensor:
    return binarise(spearman(expr[gene_idx][:, cols]), density)


def subsample(cols: list[str], n: int, seed: int) -> list[str]:
    """Seeded subsample of sample ids, order preserved -- used to match the tumour sample count to
    the normal one so the two correlation estimates carry the same sampling noise."""
    if len(cols) <= n:
        return list(cols)
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(cols), generator=g)[:n].sort().values.tolist()
    return [cols[i] for i in pick]


def jaccard(A: torch.Tensor, B: torch.Tensor) -> float:
    a, b = A.bool(), B.bool()
    u = (a | b).sum().item()
    return (a & b).sum().item() / u if u else float("nan")
