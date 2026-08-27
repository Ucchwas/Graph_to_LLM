# Phase 5 — dataset feasibility and smoke test

Written 2026-08-27. User direction: pause the Phase-4 10-seed confirmation; keep the architecture
exactly as selected in Phase 4; find out which graphs it can be trained on as is, what each costs,
and what the variable-size case (PPI) needs — before any seed sweep or GFM integration.

## 1. The model (fixed)

```
A_obs [N,N] (15 % of cells hidden per step) and edge_index = A_obs.nonzero()
  → GCNConv(N → 1024) on the raw rows            (its weight is the row projection: no E1)
  → one pre-norm residual GCN block  h ← h + GELU(GCNConv(LN(h)))
  → decoder-input LayerNorm → D1  Â = Z W Zᵀ
loss: pos-weighted BCE on the hidden cells; AdamW lr 3e-3, wd 0.01, warmup 100, patience 200
```

No node features, no constant inputs, no structural embeddings, no ID-embedding inputs, no
tokenizer, no LLM. Trained independently per graph, one seed. Note what the first layer is:
`x_i = Σ_j A_ij W[j]` — the weight matrix is a per-node table of vectors that node j hands to
each neighbour, read through the raw row. That is the reason the layer is tied to N (§5).

## 2. Datasets (downloaded and measured on the laptop, `g2l/datasets.py`)

| dataset | nodes | undirected edges | density | mean / max degree | features | dense A fp32 | first-layer W (×1024) | N² logits + mask per step |
|---|---|---|---|---|---|---|---|---|
| Cora (Planetoid) | 2,708 | 5,278 | 0.0014 | 3.9 / 168 | 1,433 (unused) | 29 MB | 11 MB | ~0.1 GB |
| ogbl-ddi | 4,267 | 1,334,889 (train 1,067,911) | 0.147 (train 0.117) | 500 / 2,234 (train) | none | 73 MB | 17 MB | ~0.3 GB |
| Amazon Photo | 7,650 | 119,081 | 0.0041 | 31.1 / 1,434; 115 isolated | 745 (unused) | 234 MB | 31 MB | ~1 GB |
| PPI (GraphSAGE) | 24 graphs, 591–3,480 each (56,944 total) | 3,854–53,377 each (793,632 total) | 0.009–0.022 | max 112–720 | 50 (unused) | 1–48 MB per graph; union 13 GB | per graph N_g × 1024 | per graph |

Two memory terms. The dense one is O(N²) per step (rows, logits, mask, their gradients): a
50 k-node graph would be 10 GB per N² fp32 tensor, so the dense route stops around N ≈ 30 k.
The other is PyG's message tensor, O(E_directed × d) per conv layer (x_j gathered per edge):
Cora 43 MB, Photo 1 GB, **DDI 8.7 GB per layer** (2.1 M directed edges × 1024 × 4 B; with the
backward of two layers ≈ 25 GB) — DDI is out of reach for the 12 GB laptop GPU and fits an H100.
If that ever binds, GCNConv accepts a sparse adjacency and then runs as an spmm at O(N × d);
not changed in this phase (it would alter the numerical path of the Phase-4 reproduction).
PPI is small per graph and only problematic because N varies.

Splits and metrics:

- **Cora, Photo** — RandomLinkSplit 85 / 5 / 10, is_undirected, 1:1 seeded uniform negatives
  (the Phase 1–4 protocol; published GAE / MaskGAE numbers use it). Test input = train + val
  edges. Columns: AUROC, AP@1:1, AUROC / AP over every scorable cell, lift.
- **ogbl-ddi** — the official split (by protein target: train 1,067,911 / valid 133,489 /
  test 133,489 positive pairs; official negatives valid 101,882 / test 95,599), official metric
  Hits@20 (fraction of test positives scored above the 20th-highest official negative;
  `hits_at_k` matches `ogb.linkproppred.Evaluator` to 1e-6 in `tests/test_datasets.py`). Input
  graph = the train edges at val and test time (the OGB convention). AUROC / AP@1:1 use seeded
  uniform non-edges as on the other graphs, so the columns stay comparable; Hits@20 is added.
  Selection stays on val AUROC (one rule for every graph); val Hits@20 is recorded at that epoch.
- **PPI** — no split defined yet (§5).

Compatibility with raw-row input: Cora, DDI and Photo are single fixed-N graphs, so the row
width equals the node count and the model is built per graph with `in_channels = N`. DDI is
dense (15 % of all pairs are edges, mean degree 500): the rows carry far more information than
Cora's, pos_weight ≈ 7.5 instead of 816, and the masked-cell objective labels 20 % of the true
edges (the held-out ones) as absent on the cells it draws — the same effect as Cora's 15 %, only
larger in absolute terms. Photo has 115 isolated nodes (all-zero rows: those tokens are the
bias of the first layer and can only be predicted as non-edges, which they are).

## 3. Baselines (same split, same scorer, one seed)

identity and random (leak canary, chance); common neighbours, PPR; featureless GAE / VGAE
(X = I, Kipf & Welling's GAE* / VGAE* rows: the first GCN weight is then also a per-node table);
MaskGAE re-implemented in the harness from the reference code's verified defaults
(`baselines/maskgae.py`: edge-wise masking p = 0.7 on undirected pairs, one GCN layer to 128
with BatchNorm / ELU / input dropout 0.8, 2-layer MLP decoder (64) on the Hadamard product,
degree decoder α = 0.003, negatives by `negative_sampling`, Adam 0.01, wd 5e-5, clip 1.0, 500
epochs, best validation AUC every 10 epochs, no early stopping) — featureless (X = I) as the
comparable row, and with the dataset's node features on Cora / Photo as the check of the
re-implementation: the published MaskGAE_edge Cora row is 96.42 / 95.91 AUC / AP (their Table 3;
verified). Two things the verification turned up: the reference masks *directed* entries and
re-symmetrises the remainder, so about 21 % of its targets stay visible to its encoder (we mask
pairs, so ours hides the full 70 %); and the paper has no Photo link-prediction row at all
(Photo is node classification only there). Featureless MaskGAE numbers are not published: ours
are new.

Reference points on Photo from later papers under the same 85 / 5 / 10 protocol (single-sourced
by the research pass, not re-verified): with features, VGAE 95.6 / 94.6 (Bandana, WWW '24) and
GAE 93.9 / 93.0 (Barlow GAE, AISTATS '23); structure only, DeepWalk 91.7 / 91.6 (GiGaMAE) and
common neighbours ≈ 0.96 AUC (TELP, 5-fold CV). No featureless GAE / VGAE number on Photo exists.
On DDI the leaderboard baselines with learned node embeddings (hidden 256, selection on val
Hits@20; verified) are GCN 0.371 ± 0.051 and GraphSAGE 0.539 ± 0.047 test Hits@20.

**What our model is, in these terms** (verified algebra): with rows as input the first layer
computes Â A W, which is the featureless GCN / GAE (X = I) with propagation operator Â A in
place of Â — one extra hop — followed by the residual block, the bilinear decoder and the
masked-cell loss. The featureless GAE / VGAE rows are therefore the nearest relatives of the
direct route, and the DDI leaderboard GCN (a free embedding table + GCN) is the same family.

## 4. Runs (Marlowe, `slurm/phase5_smoke.sbatch`, array 0-2 = cora / ddi / photo)

26 runs: 3 direct-GCN runs + 23 baseline rows, one GPU task per dataset, ≈ 1 GPU-h total
(Photo's direct run is the longest: ~0.5 s / epoch, ≤ 2,000 epochs). Rows under
`results/phase5/rows`, table by `python -m g2l.aggregate5`. Laptop use: downloads, inspection,
tests, and a 3-epoch CPU pass of every run (`--epochs 3`) before submission.

## 5. PPI: sharing one backbone across graph sizes without changing the input

The only size-tied parameter is the first layer's table W ∈ R^{N_g × 1024}. Everything after it
(the residual block, the decoder-input LayerNorm, the bilinear W) is size-free already. The
question is what to do with the first layer when N_g varies and node ids are not aligned across
graphs. In the released PPI data every node id belongs to exactly one tissue graph and no
protein identifiers ship with it, so the same protein in two tissues is two unrelated ids and
cannot be re-linked from the files (the 50 binary features have only 592 distinct rows in
56,944 nodes). The underlying OhmNet tissue layers do overlap heavily (a union of ≈ 4,300
proteins across the 24 graphs; 98–100 % of each val / test tissue's proteins also occur in the
training tissues), which is worth knowing: the benchmark is inductive over graphs, not over
proteins. Published link-prediction protocols on it: P-GNN (ICML 2019; graphs split 80 / 20,
10 % / 10 % of each graph's edges held out with 1:1 negatives, ROC-AUC: GCN 0.769, GraphSAGE
0.803, GAT 0.783, P-GNN 0.808) and DEAL (IJCAI 2020; one graph, the 1,767-node train graph 1:
GraphSAGE 0.811 / 0.813, SEAL 0.883 / 0.875). (Research-pass facts, measured on the files by
the agent; the independent re-verification of this group did not run.)

**What is impossible.** A first layer shared across graphs with no dependence on node index
must be permutation-equivariant in the column order. The permutation-equivariant linear maps from
an N×N matrix to N×d node features are spanned by five operations (Maron et al. 2019): row sum,
column sum, diagonal, total sum, trace. On a symmetric zero-diagonal A that leaves node i with
its degree and the edge count — i.e. a shared, index-free first layer on raw rows collapses to
a degree feature, the "constant node input" family already excluded. Non-linear variants (a
shared function applied to the row's entries and summed) are DeepSets over a multiset of 0/1
entries and reduce to the same count. This is not an engineering limit: raw rows shared across
unaligned graphs carry nothing but degree unless some parameter is attached to node identity.

**What remains** (raw rows in, no padding, no reindexing, no other representation):

| option | shared | per graph | unseen graph | params | verdict |
|---|---|---|---|---|---|
| **A. per-graph first layer, shared backbone** — one W_g per graph; residual block, LayerNorm, decoder shared; trained jointly on the 20 train graphs (one graph per step); a new graph gets its own W_g fitted on its observed edges with the shared part frozen (or co-trained) | block + decoder (~2 M) | W_g: N_g × 1024 | transductive: fit W_g first, no zero-shot | Σ N_g × 1024 = 58 M (233 MB) | the honest version of the user's architecture for many graphs; tests whether the backbone transfers |
| B. disjoint-union graph (block-diagonal A over the 24 graphs, N = 56,944, one first layer) | everything, formally | nothing | same as A | 58 M + shared | identical function to A (a row only touches its own block); needs a sparse first layer, sampled decoder / loss, and an index offset — no reason to prefer it |
| C. one shared table indexed by position (W[j] for j < N_max) | everything | nothing | "zero-shot" | N_max × 1024 | pads and ties unrelated proteins to the same vector: invalid for unaligned ids (and padding is excluded) |
| D. size-agnostic shared first layer | everything | nothing | zero-shot | ~1 k | collapses to degree (above): not an option |
| E. hypernetwork generating W_g from the graph | generator | W_g (computed) | zero-shot | generator | its input has to describe node j by something other than its index → a structural representation, which is excluded; from the raw row alone it collapses as D. The zero-shot GFMs that keep a "row × table" form obtain the table from the spectrum (OpenGraph, AnyGraph: SVD of A — spectral, excluded) |
| F. random, unlearned per-graph table (W_g = R_g Gaussian, resampled per graph; the row becomes a random sketch, E3) | everything learned | none learned | zero-shot | none | permutation-invariant only in expectation (random node initialisation, Abboud et al.); ⟨A_i R, A_j R⟩ estimates the common-neighbour count, so the bilinear decoder can recover that heuristic. Raw rows are still what is read, but the table is a random feature: flagged as bordering on "another representation" — the user's call |

Recommendation for PPI if it goes ahead: **A**, with the per-graph transductive protocol
(per-graph 85 / 5 / 10 split, per-graph W_g, shared backbone across the 20 train graphs; the
val / test graphs fit their W_g on their own train edges and are scored on their held-out
edges), paired against the same model trained on each graph alone (backbone not shared). The
paired difference is the only thing PPI can tell us: does a shared backbone help a new graph.

The general rule this exposes, relevant to the biomedical goal: a raw-adjacency first layer can
be shared across graphs exactly when node identities are aligned across them (the same genes
in every sample — then one gene table serves every graph and the model is zero-shot across
samples); it cannot be shared across unaligned graphs (molecules, PPI tissues as released)
without a structural node representation. The PI's translation setting is the aligned case.

Held for approval: nothing PPI-specific has been built beyond the download and the size table.

## 6. Gate 5

- [ ] `pytest tests/` green (new: `tests/test_datasets.py`); cluster commit pinned; datasets staged on Marlowe
- [ ] 26 rows at one commit; Cora direct row reproduces the Phase-4 width row (0.934 / 0.935 at lr 3e-3, seed 0) up to GPU nondeterminism
- [ ] table (`results/phase5/aggregate.md`) with AUROC / AP@1:1 / sparse columns on all three graphs, Hits@20 on DDI; MaskGAE-with-features on Cora near the published 96.4 / 95.9
- [ ] `results/phase5/RESULTS.md`: feasibility verdict per dataset, cost per epoch, peak memory, and the PPI decision request
