"""Per-phase gate walkthrough: `python -m g2l.walkthrough --phase N`.

Recomputes the phase's headline numbers live so a gate never means
"trust the logs". Phase 0: the metric protocol on a hand-checkable graph.
"""
import argparse

import torch

from g2l.bias import spd_matrix
from g2l.data import edge_split, load_cora, mask_matrix
from g2l.metrics import evaluate


def phase0():
    print("=" * 60)
    print("PHASE 0 WALKTHROUGH -- the evaluation protocol, by hand")
    print("=" * 60)

    # A 6-node path graph: 0-1-2-3-4-5. Mask decided by seed; scorer is 'distance-2
    # neighbours are likely edges' -- imperfect on purpose so no column is 1.0.
    A = torch.zeros(6, 6)
    for i in range(5):
        A[i, i + 1] = A[i + 1, i] = 1.0
    A_obs, sup = mask_matrix(A, frac=0.4, seed=4)
    hidden = [(int(i), int(j)) for i, j in zip(*torch.where(sup))]
    print(f"\npath graph 0-1-2-3-4-5; hidden cells (upper-tri): {hidden}")
    print(f"of which true edges: {[(i, j) for i, j in hidden if A[i, j] > 0]}")

    d = spd_matrix(A_obs, max_dist=8)
    logits = -d.float()  # score = negative observed-graph distance
    m = evaluate(logits, A, sup, seed=0)
    print("\nscorer: -shortest_path_distance on the OBSERVED graph")
    for k in ("auroc", "ap_balanced", "ap_sparse", "base_rate", "lift"):
        print(f"  {k:16s} {m[k]:.4f}")
    print(f"  scored cells     {m['n_scored']} ({m['n_pos']} positive)")
    print("\nCheck by hand: rank the hidden cells by observed distance; AUROC is the")
    print("probability a true edge outranks a non-edge; AP@sparse's floor is base_rate.")

    print("\n-- identity canary (leak check) --")
    g = torch.Generator().manual_seed(0)
    up = torch.triu((torch.rand(200, 200, generator=g) < 0.05).float(), diagonal=1)
    R = up + up.T
    R_obs, rsup = mask_matrix(R, frac=0.15, seed=0)
    mi = evaluate(R_obs * 100, R, rsup, seed=0)
    print(f"identity model: auroc={mi['auroc']:.3f} lift={mi['lift']:.3f}  (must be ~0.5 / ~1.0)")

    try:
        data = load_cora()
        print("\n-- Cora (corrected facts) --")
        print(f"nodes={data.num_nodes}  edge_index={list(data.edge_index.shape)}"
              f"  undirected_edges={data.edge_index.shape[1] // 2}  features={data.x.shape[1]}")
        tr, va, te = edge_split(data, seed=0)
        print(f"split: train={tr.pos_edge_label_index.shape[1]}"
              f" val={va.pos_edge_label_index.shape[1]}"
              f" test={te.pos_edge_label_index.shape[1]} (+{te.neg_edge_label_index.shape[1]} neg)")
        n_cells = data.num_nodes * (data.num_nodes - 1) // 2
        pos_rate = (data.edge_index.shape[1] // 2) / n_cells
        print(f"upper-tri positive rate: {pos_rate:.5f}  -> pos_weight ~ {(1 - pos_rate) / pos_rate:.1f}")
    except Exception as e:
        print(f"\n(Cora not downloaded yet: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True)
    args = ap.parse_args()
    {0: phase0}[args.phase]()
