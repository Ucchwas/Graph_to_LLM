# Phase 8E — the training fix worked, the architecture fix backfired, both tests still fail

Two changes were made for two measured reasons. **One was right and one was wrong, and the wrong one
was mine.** Training hygiene gained +0.0166 on molhiv and cut the seed spread by 3.5×; the attention
rebalancing — the change I argued for hardest — made things **worse on both tasks**. Neither test is
passed: molhiv is 0.062 below GIN, link prediction 0.013 below the edge-aware GCN.

Marlowe arrays **453343** (7 tasks × 3 seeds) and **453344** (4 tasks × 3 seeds), ~5 GPU-h,
commit `9b8325c`. Test split never touched on either track.

## MolHIV — validation only, three seeds

| arm | val ROC-AUC | params |
|---|---|---|
| **GIN baseline** | **0.8315 ± 0.0044** | 1,885,506 |
| GIN + warmup/clip (control) | 0.8264 ± 0.0226 | 1,885,506 |
| GCN baseline | 0.8135 ± 0.0168 | 528,001 |
| **LGM d=64 p=0.5 `joint`** | **0.7691 ± 0.0116** | 212,425 |
| LGM d=128 p=0.3 `split` | 0.7581 ± 0.0321 | 818,081 |
| LGM d=64 p=0.5 `split` | 0.7486 ± 0.0243 | 212,433 |
| LGM d=128 p=0.3 `count` | 0.7434 ± 0.0282 | 818,065 |
| LGM d=128 p=0.3 `joint` | 0.7073 ± 0.0191 | 818,065 |
| LGM d=64 p=0.5 `count` | 0.6948 ± 0.0299 | 212,425 |

**Gate FAILS at −0.0624.**

## Link prediction — unseen sizes, three seeds

| arm | unseen AUROC | unseen AP | seen AUROC / AP |
|---|---|---|---|
| **edge-aware GCN** | **0.9427 ± 0.0009** | **0.5617** | 0.9478 / 0.7119 |
| lgmseq (`joint`) | 0.9295 ± 0.0027 | 0.4708 | 0.9339 / 0.6443 |
| lgmsplit | 0.9158 ± 0.0013 | 0.4113 | 0.9225 / 0.5972 |
| lgmcount | 0.9129 ± 0.0041 | 0.4078 | 0.9198 / 0.5886 |

The baseline is extremely stable at ±0.0009, so the −0.0132 gap is roughly 15 standard deviations.
Adding seeds narrowed it slightly (−0.0176 at one seed → −0.0132) but did not change the verdict.

## What worked: training hygiene

MolHIV's best configuration went **0.7525 → 0.7691 (+0.0166)** and its seed spread fell from
**±0.0410 to ±0.0116**. That was the diagnosed instability, and warm-up plus gradient clipping fixed
it. Phase 8D had given the LGM plain Adam to mirror OGB's reference recipe — correct for the
baselines, which reproduce faithfully under it, but it denied the transformer its standard recipe.

**The control settles whether that was a thumb on the scale.** `gin_warm` — the baseline *with*
warm-up and clipping — scored **0.8264 ± 0.0226** against plain GIN's 0.8315 ± 0.0044. The same
treatment slightly *hurt* the MPNN and tripled its variance. Giving it only to the transformer was
the correct call, and it is now measured rather than argued.

## What failed: the attention rebalancing, and why the reasoning was wrong

`count` and `split` both raise the edge channel's share of a node's attention mass from a measured
**6.8 % to ~47 %**. Both are **worse than `joint`** on both tasks — `count` by −0.074 on molhiv at
d=64 and by −0.017 on link prediction.

The argument for them was that the dense node←node block is dead weight, on the grounds that the
no-edge arm scored 0.7948 against the degree prior's 0.7697. **That inference was wrong.** The
no-edge arm has the dense block and *no* edge channel, so it measures what the dense block achieves
**alone** — not what it contributes in combination. Reaching 0.79 unaided is not nothing, and
down-weighting it costs real accuracy. The claim that "93 % of the attention budget goes to a
channel we measured as useless" does not follow from that arm, and this experiment refutes it.

The joint softmax's cardinality-driven balance is apparently doing something useful: letting the
dense block dominate when a graph is large and the degree low is, on this evidence, the right
default rather than a defect.

## Where the two tests stand

| test | best LGM | target | gap |
|---|---|---|---|
| molhiv (validation) | 0.7691 | GIN 0.8315 | **−0.0624** |
| link prediction (unseen sizes) | 0.9295 | edgegcn 0.9427 | **−0.0132** |

Both remain unpassed. Across Phase 8 the interventions tried are: capacity (15× sweep), dropout,
sequential edge-then-node update, warm-up and clipping, and two attention-balance schemes. Only
training hygiene and the sequential update helped, by +0.017 and +0.005 respectively, and the two
architectural interventions aimed at the attention mechanism made things worse.

## What has not been tried

Structural encodings (RWSE / SPD bias on the dense block), depth, virtual node, learning-rate
sweeps, and hybrid local-plus-global designs. The first of these is the standard remedy for a graph
transformer that loses to MPNNs on sparse graphs, and remains untested here.
