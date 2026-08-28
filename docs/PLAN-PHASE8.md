# Phase 8 — a standalone LGM: incidence-aware bidirectional Graph Transformer

Approved 2026-08-28. Gates 8.0–8.2 only; the full grid is a separate decision after gate results.

## 1. Why the architecture changed

Phases 4–7 ran `gnn_direct`: `GCNConv(N → 1024)`. Its first weight is `[N, d]` — one column per
node index, which is a node-ID table in disguise (57 % of the Cora model's parameters, 79 % of
Photo's). That ties a checkpoint to one N *and* one node ordering. It is why every earlier phase
had to retrain per graph, and why nothing transferred.

Phase 8 replaces it with a from-scratch model whose parameters are independent of N and M.

## 2. Architecture

External input is the raw graph and nothing else:

```
RawGraph(n, edge_index [2, M], edge_value [M, k])
```

Everything else is an internal hidden representation built by shared projections — not a supplied
embedding, not a node-ID lookup:

```
raw graph -> node states [N, d] + edge states [M_u, d]
          -> L pre-LN bidirectional layers, attention structured by incidence
          -> node outputs [N, d] -> D1Bilinear (W [d, d]) -> [N, N]
```

No GCN, no message passing, no pretrained LLM/GFM, no tokenizer, no node-ID table, no supplied node
features, no padding, no positional embeddings, no causal masking. GCN is an external baseline only.

**Attention.** One shared `W_q/W_k/W_v/W_o` over both state types; two learned type vectors
`u_node`, `u_edge` separate them.

| block | pattern | nnz |
|---|---|---|
| node←node | dense — the global channel | `N²` |
| node←edge | incidence | `2·M_u` |
| edge←node | incidence transpose, exactly 2 keys per query | `2·M_u` |
| edge←edge | line graph — **off**, redundant (E←N then N←E already reaches it) and `Σdeg²` is 1.8e9 on ogbl-ddi |

node←node and node←edge share **one** softmax, exponentiated against a common max so they combine
as their true joint normaliser, with a learned per-head offset `beta` on the edge block. Edge states
are keys **and** values in the node softmax, so edge content reaches node outputs as a `d`-vector
rather than as a scalar bias.

**Two symmetry rules, both load-bearing.** There is one state per *undirected* edge keyed by
`(min, max)`. A node permutation can flip which endpoint is `min`, so nothing may distinguish an
edge's two endpoints: edge←node scores both slots with the **same** projection (swap them and the
output is unchanged), and there is deliberately **no src/dst role tag anywhere**. The edge
featuriser is symmetric too — sum and |difference| of endpoint degrees.

**Node features** are structural and scale-free: Fourier encoding of `log1p(deg)`, plus
`deg/mean_deg`, `log1p(deg) − log1p(mean_deg)`, and an isolated-node flag. The scale-free columns
are what let one checkpoint span mean degree 2 (molecules) and 31 (Photo).

**Disclosed:** computing `deg` by counting incident edges is a 1-hop statistic of the raw input —
computed the same way `edge_index` is, not a learned propagation layer, and ablatable
(`degree_init=False`). Flagged rather than hidden.

**Batching** is a block-diagonal disjoint union under a node budget: no dummy node is created, so
this is masking, not padding.

## 3. Constraint reading, recorded

node←edge and edge←node are neighbourhood-restricted attention. The reading taken: *incidence-aware*
requires the model to know which nodes are incident to which edges, so incidence-structured
attention is the mechanism; what is forbidden is a **convolution** — a fixed normalised
sum-aggregation layer placed before or inside the model. The node update is never a pure
neighbourhood aggregate because the dense node←node block sits in the same softmax. A reviewer could
still call this an attentional MPNN with a global readout. The only construction that avoids the
objection entirely is fully dense attention over all `N + M_u` tokens (TokenGT), which is 1.6e10
entries per head per layer at Amazon Photo — infeasible.

## 4. Expressiveness, pre-registered

Restricting attention breaks TokenGT's 2-IGN precondition, so the default arm is **1-WL bounded**.
Separately, and unavoidable given a node-state-only decoder: a deterministic permutation-equivariant
model gives **identical scores to every pair in the same Aut(G) orbit**, and collapses to a constant
score matrix on any vertex-transitive graph. The labelling trick that would fix it needs one state
per scored pair, contradicting `node outputs [N,d] → shared decoder`. RNI breaks the symmetry
stochastically; whether that helps is measured, never asserted.

## 5. Gates

**8.0 — equivariance and size independence** (`tests/test_equivariance.py`,
`tests/test_size_independence.py`, CPU). Automorphism invariance is asserted over **pair** orbits on
multi-orbit graphs (P₆, K₂,₃), with Petersen kept as a separate *collapse* test — it is
vertex-transitive, so the theorem forces every node state equal and a broken constant-output model
would pass an orbit test identically. No metric is asserted: on a provably constant score matrix
`roc_auc_score` returns 0.5533 in fp32, because orbit-equivalent nodes reduce in a different order
and float non-associativity leaves a residual that is never exactly zero. Size independence is
behavioural — `strict=True` load of one state_dict into a differently-seeded model, run at
`N ∈ {7, 23, 101, d_model}` on Erdős–Rényi probes, state_dict shape map compared before and after
every forward, gradient-receiving parameter set required to match. No shape scan: it would flag
every `[d,d]` tensor the moment a probe has `N = d_model`.

**8.1 — batching and leakage** (`tests/test_lgm_batching.py`, CPU). A batch of 12/25/40 reproduces
three single-graph runs; perturbing one graph moves the others only at float-noise scale, asserted
as a ratio because changing a batch changes reduction lengths and bit-identity is unattainable.

**8.2 — does it transfer, and do edges matter** (`g2l/run_phase8.py`, Marlowe, array 0–1).
Corpus: `ogbg-molhiv`, 41,123 graphs after filtering, N = 4–222, with genuine 3-dim bond features —
chosen because every earlier benchmark in this repo is a single fixed-N graph with `k = 1`, so
neither Phase-8 claim can be tested on them. Train only on `N ≤ 30`, then score held-out graphs at
**seen** and **unseen** sizes. The no-edge control shares the parameter count exactly (`edges=False`
empties the edge states at run time and changes no shape), so a gap between arms cannot be capacity.

Non-learned references on identical held-out cells. Two run backwards on molecules and are declared
that way rather than sign-flipped after the fact: common neighbours **0.44** (atoms sharing a
neighbour sit at ring/bond-angle distance and are precisely the ones not bonded) and preferential
attachment **0.23** (a high-degree atom is valence-saturated), so **negated degree at 0.77** is the
bar to clear.

## 6. Deferred — not authorised

The full grid (backbone × node-input arms × seeds, SPD/RWSE/RNI/register ablations,
shuffled-adjacency control, external GCN baseline, evaluation on Cora / ogbl-ddi / Photo /
PPT-Ohmnet / Decagon / TCGA, and the graph-to-graph fine-tune with the separated task prior) is
presented for approval after gate results.

## 7. Pre-registered expectations

1. **A regression against Phase 5 on Cora/Photo is the price of the constraints, not a failure.**
   `gnn_direct` scores Cora 0.9335, Photo 0.9927, Ohmnet 0.9330, DDI Hits@20 0.3073 using an N-wide
   per-node signature table that this architecture deletes. The repo's own featureless
   GAE/VGAE/MaskGAE baselines also keep a per-node table (`X = I`).
2. **ogbl-ddi needs its own config** (M_u ≈ 1.07 M; d = 128 plus gradient checkpointing) and its
   checkpoint is not interchangeable with the others.
3. The universal backbone and the normal→tumour task prior stay structurally separate: any number
   labelled "LGM" is backbone-only; prior-using numbers are labelled "LGM + task prior" and never
   enter a universal table.
