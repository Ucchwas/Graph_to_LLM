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

---

# Phase 8C — ogbg-molhiv classification: on the one externally-calibrated task, the LGM loses clearly

**The result is negative and the margin is not close.** On the official OGB scaffold split with the
official Evaluator, `LGM-with-atom` reaches **test ROC-AUC 0.6817 / validation 0.7002**, against
OGB's weakest standard baseline (GCN) at **0.7606 ± 0.0097 test / 0.8204 ± 0.0141 validation** —
**−0.079 test and −0.120 validation**, with 6× the parameters. The gap is 5–8× the leaderboard's
own seed spread, so it is not seed noise.

Marlowe array **453025**, 2 tasks, 1 GPU each, 36 min total, one seed, code commit `3075749`.
Reference numbers read from the live leaderboard (`ogb.stanford.edu/docs/leader_graphprop/`), not
from recall.

| arm | val ROC-AUC | test ROC-AUC | params | best epoch |
|---|---|---|---|---|
| **LGM-with-atom** (comparable) | **0.7002** | **0.6817** | 3,208,993 | 19 / 40 |
| LGM-no-atom (**ablation**, not comparable) | 0.7076 | 0.6332 | 3,164,449 | 18 / 39 |
| **atom − no-atom** | **−0.0074** | **+0.0485** | +44,544 | |

Placement against the full 41-entry leaderboard, for `LGM-with-atom` only. **It would not rank: 0.6817
is below every ranked entry**, 0.073 under last place.

| rank | method | test ROC-AUC | params |
|---|---|---|---|
| 1 | Multi-RF Fusion + Multi-GNN | 0.8476 ± 0.0002 | 993,331,107 |
| 9 | Molecular FP + Random Forest | 0.8208 ± 0.0037 | **5,782** |
| 20 | GINE | 0.7921 ± 0.0128 | **33,217** |
| 21 | GIN | 0.7908 ± 0.0102 | **32,385** |
| 23 | GINE+HE | 0.7903 ± 0.0079 | **9,393** |
| 35 | GIN + virtual node | 0.7707 ± 0.0149 | 3,336,306 |
| 38 | GCN | 0.7606 ± 0.0097 | 527,701 |
| 41 | GCN (in Julia) | 0.7549 ± 0.0163 | 527,701 |
| — | **ours** | **0.6817** | **3,208,993** |

**The parameter column is the finding.** GIN reaches 0.7908 with 32,385 parameters against our
0.6817 with 3,208,993 — **99× the parameters for 0.11 less**. GINE+HE reaches 0.7903 with 9,393.
The competitive models on this dataset live at 10k–500k parameters; we ran 3.2 M with dropout 0.0 on
33k graphs at a 3.5 % positive rate, a capacity roughly two orders of magnitude too large,
inherited unchanged from a link-prediction config. The epoch-18 peak and subsequent collapse is
exactly what that predicts.

Context that is not an excuse: molhiv rewards chemistry features over graph structure — the top
entries are fingerprint fusions and a 5,782-parameter random forest outranks every GNN below rank 9.
That flatters domain features generally, but it does not rescue us, because we lose to plain GIN at
1/99th our size.

## What went wrong, stated as diagnosis rather than excuse

**Both arms overfit hard.** Validation peaked at epoch ~18–19 and then collapsed — the `atom` arm
ran 0.7002 down to 0.6118–0.6568 by epoch 38, the ablation 0.7076 down to 0.5212–0.5927. The config
(`configs/phase8c.yaml`) was written for link prediction and carries **dropout 0.0**, while OGB's
GCN baseline uses dropout 0.5 at **527k parameters**. We ran **3.2 M parameters with no dropout** on
33k graphs at a 3.5 % positive rate. That is a regularisation failure, and it is the most likely
explanation for most of the gap.

**So the honest statement has two halves.** As configured, our architecture substantially
underperforms standard MPNNs on a real benchmark — that is measured, and no amount of caveat
removes it. Whether the architecture is *inherently* weaker here is **not** established, because we
did zero hyperparameter tuning and are visibly overfitting. Until that work is done, the finding
stands as "we lose", not "we would lose after tuning".

**The two arms are inconclusive against each other**: the ablation is *better* on validation
(−0.0074 for atom) and worse on test (+0.0485 for atom), i.e. the sign flips between splits. With
one seed and this much overfitting, no conclusion is available. Worth noting on its own, though,
that atom features do not help on validation at all — a well-fit molecular model should benefit
substantially from knowing carbon from oxygen, which is further evidence the model is not fitting
properly rather than that atom identity is unimportant.

**Also not tuned:** the 3.5 % class imbalance is handled with plain unweighted BCE (matching the
official baselines), the head is a single linear layer on mean-pooled node states, and no virtual
node or readout variant was tried.

## What Phase 8C establishes

- The Phase-8 link-prediction numbers had **no external reference**, and this supplies one. It is
  unfavourable.
- A strong number on a self-defined task (0.9307 / 0.9267 on our masked-cell benchmark) did **not**
  translate into a competitive number on an established one. That is worth more than the earlier
  result was.
- The obvious next step, if Phase 8C is ever resumed, is regularisation and a hyperparameter sweep
  before any claim about the architecture is made from this task. Deferred as instructed.

## Gate 8

- [x] 8.0 equivariance, pair-orbit invariance, behavioural size independence
- [x] 8.1 no-padding batch equivalence and cross-graph leakage
- [x] 8.2 one checkpoint transfers to unseen sizes; LGM beats the capacity-matched GCN and every
      non-learned prior on both splits
- [ ] VALIDATED line — user

VALIDATED ____
