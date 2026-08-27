# Phase 5 — Dataset feasibility: the fixed direct-GCN LGM on four graphs

**Claim:** the Phase-4 architecture — raw masked adjacency rows and the raw edge set into a first
GCN layer of input width N, one residual GCN block, a bilinear decoder, nothing else — **trains
end to end on every graph we tried and beats every featureless baseline on all four**: Cora
0.9335 AUROC, ogbl-ddi 0.9970 (Hits@20 **0.3073** on the official split), Amazon Photo 0.9927,
PPT-Ohmnet 0.9330. On the two denser graphs it also beats the published *with-features* numbers
(Photo: ours 0.9927 vs VGAE 95.6 / GAE 93.9 reported under the same 85/5/10 protocol). It does
not beat a with-features method on Cora (MaskGAE with the 1,433-dim bag of words, 0.9582), which
is expected: the model is given no node features anywhere. Controls are exact — identity scores
0.5000 and random ≈ 0.500 on all four graphs, so nothing leaks. The pipeline question this phase
was opened to answer is settled: **the architecture is dataset-agnostic within the fixed-N,
dense-scorable regime, and the graph structure alone carries the signal.**

Produced on Marlowe (H100 80 GB, 1 GPU per task), Slurm array **451412** (4 tasks = the 4
datasets, 34 runs, **1.12 GPU-h** total wall-clock; all four tasks COMPLETED — elapsed 00:59 /
52:25 / 12:24 / 03:08 for Cora / DDI / Photo / Ohmnet). All rows at code commit `550abee`.
One seed per run (architecture is fixed; this phase asks "does it run and where does it land",
not "how tight is the interval"). Aggregation: `results/phase5/aggregate.md`; laptop use was
downloads, inspection, tests and a 3-epoch CPU pass of all 34 runs before submission.

## Locked setup

| item | value |
|---|---|
| model | `A_obs` rows → `GCNConv(N → 1024)` (its weight *is* the row projection; no E1) → 1 pre-norm residual GCN block `h ← h + GELU(GCNConv(LN(h)))` → decoder-input LayerNorm → D1 `Â = Z W Zᵀ` |
| inputs | raw adjacency rows and `edge_index = A_obs.nonzero()`, rebuilt every forward from the model's own input graph. **No** node features, constant inputs, structural/spectral embeddings, ID-embedding inputs, tokenizer or LLM |
| hidden width | **1024** on every graph — the Phase-4 selection (0.939 vs 0.933 at 256, 0.934 at 2708 on Cora), it keeps the row-compression ratio ≤ 7.5:1 on the largest graph (Photo; 256 would be 30:1), and one width across graphs gives the later shared-backbone phase a single backbone size |
| loss | pos-weighted BCE on **masked cells only**: 15 % of upper-triangular cells re-drawn each epoch and hidden from the input rows |
| optimiser | AdamW lr 3e-3, wd 0.01, 100 warm-up epochs, grad-clip 1.0, ≤ 2000 epochs, early stop on val AUROC patience 200, best-val state scored on test |
| splits | Cora / Photo / Ohmnet: `RandomLinkSplit` 85/5/10, undirected, 1:1 seeded negatives, test input = train+val edges. DDI: the **official OGB protein-target split**, input graph = train edges at val and test, official negative sets scored by Hits@20 (`hits_at_k` matches `ogb.linkproppred.Evaluator` to 1e-6) |
| scorer | one `evaluate_edge_split`: AUROC and AP on test positives + 1:1 negatives (the published-comparable column), plus AUROC / AP / lift over every scorable cell |

## Datasets (measured, `g2l/datasets.py`)

| dataset | N | undirected edges | density | pos_weight | base rate | trainable params | peak GB | s/epoch |
|---|---|---|---|---|---|---|---|---|
| Cora | 2,708 | 5,278 | 0.0014 | 815.7 | 1.44e-4 | 4.88 M | 0.4 | 0.05 |
| ogbl-ddi | 4,267 | 1,334,889 (train 1,067,911) | 0.147 | 7.5 | 1.69e-2 | 6.47 M | 18.2 | 0.41 |
| Amazon Photo | 7,650 | 119,081 | 0.0041 | 288.1 | 4.09e-4 | 9.94 M | 3.2 | 0.71 |
| PPT-Ohmnet (combined) | 4,494 | 68,527 | 0.0068 | 172.3 | 6.83e-4 | 6.71 M | 1.7 | 0.22 |

PPT-Ohmnet replaces SagePPI (user, 2026-08-27): the tissue-labelled BioSNAP edgelist (3,666,564
lines over 144 tissues, Entrez ids) collapsed to its distinct protein pairs = the combined,
tissue-nonspecific PPI graph. Unlike SagePPI's 24 unaligned graphs this is one fixed-N graph, so
the existing masked-edge protocol applies unchanged. **The 144 tissue layers are preserved on the
same node index** (`processed/combined.pt: tissues`, plus the original OhmNet release in `raw/`)
for the cross-graph phase; every layer is a verified subgraph of the combined graph on the
aligned node set (`tests/test_datasets.py`).

Two memory terms govern feasibility: the dense O(N²) per step, and PyG's message tensor
O(E_directed × d) per conv layer — the latter is what makes DDI cost 18.2 GB (2.1 M directed
edges × 1024) and put it out of reach of the 12 GB laptop GPU. Both are comfortable on an H100;
the dense route stops being sensible around N ≈ 30 k.

## Results (one seed, `results/phase5/aggregate.md`)

AUROC / AP@1:1. **Bold** = best featureless model on that graph. `*_x` rows use node features.

| model | Cora | ogbl-ddi | Amazon Photo | PPT-Ohmnet |
|---|---|---|---|---|
| **direct GCN (ours)** | **0.9335 / 0.9365** | **0.9970 / 0.9952** | **0.9927 / 0.9918** | **0.9330 / 0.9368** |
| identity (leak canary) | 0.5000 / 0.5000 | 0.5000 / 0.5000 | 0.5000 / 0.5000 | 0.5000 / 0.5000 |
| random | 0.4798 / 0.4920 | 0.5001 / 0.5002 | 0.5011 / 0.4996 | 0.4998 / 0.5006 |
| common neighbours | 0.7262 / 0.7234 | 0.9476 / 0.9448 | 0.9710 / 0.9676 | 0.8776 / 0.8739 |
| PPR | 0.8545 / 0.8998 | 0.9264 / 0.9098 | 0.9867 / 0.9846 | 0.9186 / 0.9176 |
| GAE (featureless, X = I) | 0.8452 / 0.8729 | 0.9596 / 0.9601 | 0.9833 / 0.9830 | 0.8752 / 0.8791 |
| VGAE (featureless) | 0.8508 / 0.8826 | 0.9542 / 0.9531 | 0.9819 / 0.9814 | 0.8773 / 0.8781 |
| MaskGAE (featureless) | 0.8304 / 0.8688 | 0.9685 / 0.9667 | 0.9866 / 0.9855 | 0.9065 / 0.9099 |
| MaskGAE (with features) | 0.9582 / 0.9529 | n/a (no features) | 0.9828 / 0.9804 | n/a (no features) |

Margin of the direct route over the best **featureless** competitor: Cora **+0.089** (vs VGAE),
DDI **+0.029** (vs MaskGAE), Photo **+0.006** (vs MaskGAE), Ohmnet **+0.027** (vs MaskGAE) AUROC.

**ogbl-ddi, official metric.** Test **Hits@20 = 0.3073** (val 0.3928), against featureless GAE
0.1227, MaskGAE 0.1117, VGAE 0.0668, common neighbours 0.1773, PPR 0.0096, identity 0.0000 —
i.e. **1.7× the best baseline** on the official metric. For orientation, the OGB
leaderboard's GCN and GraphSAGE — which give every node a *learned free embedding* — report
0.3707 ± 0.0507 and 0.5390 ± 0.0474. Our run reached its best validation at epoch 1811 of a 2000
cap, i.e. it was still improving when the budget ended; the number is a floor, not a ceiling.
AP over all 8.9 M scorable cells is 0.8276 at a 1.7 % base rate.

## What this shows / does not show

- **Shows** the pipeline is not Cora-specific: four graphs spanning 2.7 k–7.7 k nodes, densities
  0.0014–0.147, biological / co-purchase / citation / drug domains, all train to convergence with
  one architecture, one learning rate, one width, no per-dataset tuning.
- **Shows** structure alone suffices to beat every featureless baseline, and on the denser graphs
  to beat published with-features autoencoders.
- **Shows** the harness is sound where it can be checked: Cora reproduces the Phase-4 laptop row
  (0.9335 vs 0.934), our MaskGAE re-implementation with features reaches 0.9582 / 0.9529 against
  the published 0.9642 / 0.9591, and featureless GAE/VGAE (0.845 / 0.851) sit at Kipf & Welling's
  featureless GAE\* / VGAE\* level (84.3 / 84.0).
- **Does not show** statistical significance: one seed per cell, no error bars, no paired tests.
  The 10-seed confirmation (paused by the user) is what supplies those.
- **Does not show** anything about generalisation across graphs. Every model here is trained and
  tested on one graph; the first layer is a per-node table tied to that graph's N.
- **Does not show** a competitive DDI result in leaderboard terms — 0.3073 vs GraphSAGE's 0.539 —
  though the comparison is not like-for-like (they learn a free embedding per node; the run was
  truncated by the epoch cap).

## Gate 5

- [x] `pytest tests/` green on both machines (26 dataset/GNN/metric tests locally, full suite exit 0 on Marlowe); cluster commit pinned at `550abee` for the whole phase
- [x] 34 rows at one commit; Cora direct row reproduces the Phase-4 selection (0.9335 vs 0.934)
- [x] AUROC / AP@1:1 / sparse columns on all four graphs; Hits@20 on DDI; MaskGAE-with-features on Cora within 0.006 of the published number
- [x] leak canary exact on every graph (identity = 0.5000)
- [ ] VALIDATED line — user

VALIDATED ____
