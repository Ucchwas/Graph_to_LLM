# Phase 8 — preliminary: the LGM beats a capacity-matched size-independent GCN, and one checkpoint holds from N ≤ 30 to N = 213

**Preliminary result, one seed.** Gates 8.0–8.2 pass. On held-out `ogbg-molhiv` graphs the
incidence-aware Graph Transformer scores **+0.065 AUROC / +0.204 AP** above a capacity-matched
size-independent GCN at sizes it trained on and **+0.070 AUROC / +0.192 AP** at sizes it never saw,
and both architectures transfer across size essentially without decay. **That gap is confounded**:
the GCN cannot consume edge values, so it is not a clean architecture comparison — see
*The GCN comparison is confounded too*.

Marlowe array **452985**, 3 tasks, 1 GPU each, **25 min total (0.4 GPU-h)**, code commit `ac64f63`,
one seed, identical masks across arms. Rows: `results/phase8/rows/`.
Supersedes job 452918/452926, which ran before the decoder was symmetrised (LGM 0.9338/0.9280 then,
0.9307/0.9267 now); those rows are deleted, the numbers kept here.

## Setup

`g2l/lgm.py`: raw graph `(n, edge_index [2,M], edge_value [M,k])` in; node states `[N,d]` and edge
states `[M_u,d]` are internal; one transformer over both with shared QKV — node←node dense,
node←edge and edge←node on the incidence relation, sharing one softmax. `d=256, L=4, H=8`, 3.23 M
parameters. No GCN inside, no node-ID table, no supplied node features, no padding.

Data: `ogbg-molhiv`, 41,123 graphs (N = 4–222) with 3-dim bond features. **Trained only on N ≤ 30**
(25,589 graphs); held-out graphs scored at sizes seen (3,193) and never seen (920, N = 31–213).
Masked-cell reconstruction, 15 % of upper-triangle cells hidden per graph, `edge_index` rebuilt from
the observed graph only.

## Result

| arm | params | seen N ≤ 30 | | unseen N 31–213 | |
|---|---|---|---|---|---|
| | | **AUROC** | **AP** | **AUROC** | **AP** |
| **LGM** | 3,229,728 | **0.9307** | **0.6342** | **0.9267** | **0.4639** |
| GCN, capacity-matched | 3,238,400 | 0.8657 | 0.4302 | 0.8569 | 0.2724 |
| LGM, no edge states | 3,229,728 | 0.7948 | 0.2820 | 0.7927 | 0.1424 |
| negated degree (best prior) | — | 0.7697 | 0.2289 | 0.7705 | 0.1133 |
| common neighbours | — | 0.4426 | 0.1012 | 0.4757 | 0.0457 |
| random | — | 0.5010 | 0.1020 | 0.4862 | 0.0436 |

| paired | seen | unseen |
|---|---|---|
| LGM − GCN | **+0.0650** AUROC / +0.2040 AP | **+0.0698** AUROC / +0.1915 AP |
| LGM − no-edge | +0.1359 / +0.3521 | +0.1340 / +0.3215 |

Across unseen-size quartiles the LGM holds 0.9352 / 0.9304 / 0.9061 / 0.9175 (last bucket N 70–213,
7× the training cap); the GCN holds 0.8652 / 0.8646 / 0.8478 / 0.8513. **Size transfer is a property
of both size-independent architectures**, not of the LGM specifically — the gap between them is
accuracy, not transfer.

The GCN is given the LGM's full structural node featuriser rather than a bare scalar, and its width
(800) is chosen to match the LGM's parameter count to within 0.3 %. Both choices work against the
LGM deliberately; a handicapped baseline would have flattered it. It is still not a clean
architecture comparison, for the reason below.

**Two baselines run backwards on molecules** and were declared as their own arms before the run
rather than sign-flipped afterwards: common neighbours 0.44 (atoms sharing a neighbour sit at ring
or bond-angle distance and are exactly the pairs *not* bonded) and preferential attachment 0.23
(a high-degree atom is valence-saturated). Negated degree at 0.77 is the real non-learned bar.

## Correction to the earlier reading of the no-edge control

The previous write-up called the no-edge gap "what the first-class edge channel is worth". That
conflated two things and overstated the claim. **node←edge attention is the only path by which
graph structure reaches node states** — node←node is dense content-based attention over all nodes,
carrying no structure, and the node featuriser supplies only degree. So deleting edge states does
not remove *edge values*; it removes **all structural information except degree**, which is exactly
why that arm lands at 0.7948 against the degree prior's 0.7697.

The +0.134 is therefore structure-plus-edge-values combined, not edge values alone. Isolating edge
values needs an arm with edge states but constant (all-ones) values — a deferred ablation.

## The GCN comparison is confounded too — what it does and does not support

**`GCNConv` cannot consume edge values**, so the GCN never sees the 3-dim bond features the LGM
gets. The +0.065 / +0.070 therefore confounds three differences at once:

1. incidence attention vs message passing,
2. **bond features vs none**,
3. dense global attention vs a 4-hop receptive field.

Given how much the no-edge arm showed structural information is worth here, (2) could account for
much of the gap. **Supported: the LGM scores higher than a capacity-matched size-independent GCN.
Not supported: that incidence attention beats message passing.** No arm in this run isolates the
body, and the earlier claim in this file that the GCN comparison did so was wrong.

The decomposition needs one more arm — the LGM with constant all-ones edge values. Then
`lgm − lgm_flatedge` is the edge-value contribution and `lgm_flatedge − gcn` is the architecture
contribution, both at matched capacity and matched structural access. An edge-aware message-passing
baseline (GINEConv) would be the complementary check.

Two smaller caveats on the same comparison: the GCN's width is 800 against the LGM's 256 (parameters
are matched, width is not), and neither arm's hyperparameters were tuned separately — both use a
config written with the LGM in mind.

## Gate 8.0 / 8.1 and the decoder

Equivariance is asserted over **pair** orbits on multi-orbit graphs (P₆, K₂,₃), never on Petersen —
vertex-transitive graphs force every node state equal and collapse the score matrix to a constant, so
a broken model passes identically. No metric is asserted: `roc_auc_score` returns 0.5533 in fp32 on a
provably constant matrix, because float non-associativity leaves an ulp-scale residual between
orbit-equivalent nodes and AUROC swings across its full range on ties. Size independence is
behavioural (strict `state_dict` load into a differently-seeded model, run at
`N ∈ {7, 23, 101, d_model}`), not a parameter-shape scan.

The decoder is now `Z Ws Zᵀ` with `Ws = (W + Wᵀ)/2`. `W` is symmetric only at its `0.1·I` init and
training supervises the strict upper triangle only, so its antisymmetric half was near-unconstrained
while the Phase-8 path scored upper-triangle cells directly without the `(L + Lᵀ)/2` the older
phases' metrics applied. Symmetrising `W` equals symmetrising the output, so Phases 1–7 numbers are
unaffected. 40 tests green, locally and on the cluster build.

## What this does not show

- **One seed, no variance estimate.** Deferred.
- **Molecules are the friendliest domain**: bonds are valence-constrained and degree alone reaches
  0.77. Nothing here predicts the number on a citation or interaction graph.
- **The edge-value advantage is untested off molecules.** It rests on real 3-dim bond features;
  Cora, Photo, ogbl-ddi, PPT-Ohmnet and the binarised TCGA graphs all have `k = 1` and all-ones
  edge values. Not reproducible on any other dataset in this repo as they stand.
- **The LGM hit the 40-epoch cap with validation still rising** (0.9330), so it is undertrained and
  the number is a floor. The GCN also ran to the cap; the no-edge arm early-stopped at 27.
- Featureless GAE/VGAE/MaskGAE are **structurally disqualified** from this comparison: they use
  `X = I`, a per-node identity table, which cannot run at an unseen N at all.
- The default arm is 1-WL bounded with a proven automorphism ceiling, never measured on a real graph.

## Gate 8

- [x] 8.0 equivariance, pair-orbit invariance, behavioural size independence
- [x] 8.1 no-padding batch equivalence and cross-graph leakage
- [x] 8.2 one checkpoint transfers to unseen sizes; LGM beats the capacity-matched GCN and every
      non-learned prior on both splits
- [ ] VALIDATED line — user

VALIDATED ____
