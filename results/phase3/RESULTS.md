# Phase 3 — Graph-aware attention: a learned per-head hop-distance bias on the frozen bodies

**Claim:** giving attention a structural channel — a zero-initialised, per-head table indexed by
shortest-path distance on the model's own input graph, added to the attention logits of every
layer — **helps every body** on Cora masked edge prediction: frozen pretrained Llama-3.2-1B
**+0.029 AUROC** (0.881 → 0.909, p < 0.001, 20 paired seeds), the same body at random init
**+0.033** (0.889 → 0.922), a 4-layer / 1-layer transformer trained from scratch **+0.024 /
+0.013**. The bias is the active ingredient: at its selected learning rate the table becomes a
hard routing mask (max |bias| 43–68 against unit-scale q·k), and the same architecture trained
and tested on a degree-preserving rewiring of the graph reaches only 0.688. **The language pretraining is not:** with
the bias tuned for both bodies, pretrained − random = **−0.013** (p < 0.001), and the best
frozen-LLM arm (random weights + bias, 0.922) only ties the body-less linear model (0.918,
+0.004 AUROC p 0.055, −0.004 AP p 0.024). No configuration with a frozen LLM in the loop beats
E1 + D1 alone. This is the evidence behind the 2026-08-27 redirection to a graph-native LGM
([docs/PLAN-LGM.md](../../docs/PLAN-LGM.md)): keep the structure channel, drop the frozen LLM as
the centre.

Produced on Marlowe (H100 80 GB, 1 GPU per run): probe at commit `8f3b78a`; main array
**449496** (370 runs; 9 tasks), pre-registered follow-up **449913** (380 runs: edge rule, seed
rule, data-fraction sweep), top-up **450168** (120 runs: bias LR 3.0 for every biased arm and
the high-bias cells for random + SPD, 20 seeds). Slurm accounting: all 25 array elements
COMPLETED; the ids 449312, 449391, 449403 and 449906 that appear in `docs/PLAN-PHASE3.md` and
commit messages were cancelled before running (queue placement / resubmission) and hold no rows. All **870 rows are at code commit `a60dc60`**,
which stayed checked out on the cluster for the whole phase; between stages only
`configs/phase3.yaml` changed. 26.8 GPU-hours of training wall-clock (frozen + bias 21.1,
frozen 2.2, scratch + bias 2.7, scratch 0.6, none 0.3). Recomputed on the laptop by
`python -m g2l.walkthrough --phase 3` (below). Aggregation: `results/phase3/aggregate.md`.

## Locked setup

| item | value |
|---|---|
| data | Cora, `RandomLinkSplit` 85/5/10, undirected, seeds 0–19; the seed fixes the split (positives and the 527 sampled test negatives) and every initialisation; masking pattern seeded by the epoch. Scorer `evaluate_edge_split` (Phase-1 protocol) |
| tokens | E1 `Linear(2708 → 2048)` + LayerNorm gain T/√d, T = 0.987007 (scratch arms: d 256, same gain rule) — Phase 2 unchanged |
| bodies | frozen `meta-llama/Llama-3.2-1B` @ `4e20de36…` pretrained / random-init (bf16, eager, RoPE = identity, bidirectional, RMSNorm gains trainable at 1e-4) — Phase 2 unchanged; scratch: pre-LN `nn.TransformerEncoder`, d 256, 8 heads, 4 or 1 layers, trained |
| **bias** | `SPDBias`: table `[heads, 10]`, bucket 0 = self (fixed 0), 1…8 = hop distance, 9 = farther or unreachable; **zero-init**; computed by `spd_matrix_fast` from the model's **own input graph** (the masked train graph in training, the test input at eval — never a held-out cell); the same `[1, heads, N, N]` float tensor is added to the logits of **every layer** (Llama: the 4-D `attention_mask` route, which transformers 5.15.1 adds untouched; scratch: the `nn.TransformerEncoder` float mask); own AdamW group `lr_bias`, no weight decay |
| decoder | D1 `Z W Zᵀ`, W = 0.1·I, decoder-input LayerNorm gain 1/√d — Phase 2 unchanged |
| loss | BCE-with-logits on **masked cells only**: 15 % of upper-triangular cells re-drawn each epoch, hidden from the input row, target = train graph, `pos_weight` = 815.68 |
| optimiser | AdamW, wd 0.01 (0 on the table), 100 warm-up epochs, grad-clip 1.0, ≤ 2000 epochs, early stop on val AUROC with patience 200, best-val state scored on test |
| grids | bias LR {3e-3, 1e-2, 3e-2, 1e-1} main; edge rule (follow-up) added {3e-1, 1.0} for pretrained and both scratch arms and {1e-3, 3e-4} for random; the top-up added 3.0 for every biased arm and {3e-1, 1.0} for random, so both frozen bodies were judged on identical grids; encoder LR at the Phase-2 selection with a ×⅓ / ×3 neighbourhood at the central bias LRs |
| selection | per arm, (encoder LR, bias LR) = best mean **val** AUROC over seeds 0–9; seeds 10–19 (run at every candidate cell of the arms whose gate delta fell below the MDE) enter only the paired deltas; controls exempt from the edge rule |
| memory / speed | frozen body + bias 30.9 GB peak, 0.33 s/epoch (no checkpointing); scratch L4 + bias 1.9 GB, 0.09 s/epoch |

The pre-array review caught one blocker: in eval mode `nn.TransformerEncoderLayer` takes a
fast path that **casts a float mask to bool**, which would have turned the scratch arms'
additive bias into a hard self-only mask at every validation and test pass. Fixed by
disabling the fast path at import (`g2l/model.py`) with a regression test
(`test_bias_survives_eval_mode`); the first submission (449312) was cancelled and its rows
deleted. This change is why the scratch d256 L4 reproduction below differs from Phase 2 by
0.007 while the three Llama / none arms reproduce to 0.0000.

## Table — Cora, mean ± std over seeds, each arm at its selected cell

AUROC / AP@1:1 on the 527 held-out test edges plus 527 seeded non-edges; AUROC(sparse) /
AP(sparse) on all 3.66 M masked cells at prevalence 0.000144; lift = AP(sparse) / prevalence.

| model | enc lr | bias lr | n | AUROC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift | trainable | best epoch | s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PPR (Phase 1) | – | – | 5 | 0.850±0.012 | 0.900±0.010 | 0.850±0.012 | 0.0308±0.0049 | 214 | – | – | – |
| GAE 600 ep (Phase 1) | – | – | 5 | 0.905±0.006 | 0.912±0.005 | 0.908±0.006 | 0.0078±0.0016 | 54 | – | – | – |
| GAT (Phase 1) | – | – | 5 | 0.910±0.009 | 0.911±0.008 | 0.910±0.007 | 0.0052±0.0007 | 36 | – | – | – |
| **none** (E1 + D1, no body) | 3e-4 | – | 20 | 0.918±0.006 | **0.932±0.006** | 0.917±0.007 | 0.0175±0.0044 | 121 | 9.75 M | 315 | 0.05 |
| pretrained frozen | 1e-2 | – | 20 | 0.881±0.014 | 0.895±0.014 | 0.881±0.015 | 0.0112±0.0033 | 78 | 9.82 M | 462 | 0.30 |
| **pretrained frozen + bias** | 1e-2 | 1.0 | 20 | 0.909±0.011 | 0.921±0.011 | 0.908±0.013 | 0.0121±0.0040 | 84 | 9.82 M | 324 | 0.33 |
| random frozen | 3e-3 | – | 20 | 0.889±0.008 | 0.902±0.008 | 0.890±0.009 | 0.0128±0.0028 | 89 | 9.82 M | 447 | 0.30 |
| **random frozen + bias** | 3e-3 | 3.0 (edge) | 20 | **0.922±0.006** | 0.929±0.007 | 0.923±0.007 | 0.0101±0.0027 | 70 | 9.82 M | 211 | 0.33 |
| scratch d256 L1 | 3e-3 | – | 20 | 0.877±0.014 | 0.883±0.016 | 0.876±0.015 | 0.0046±0.0021 | 32 | 1.55 M | 431 | 0.04 |
| scratch d256 L1 + bias | 1e-3 | 1.0 | 20 | 0.889±0.008 | 0.891±0.011 | 0.889±0.010 | 0.0045±0.0013 | 31 | 1.55 M | 259 | 0.07 |
| scratch d256 L4 | 1e-3 | – | 20 | 0.870±0.016 | 0.876±0.017 | 0.871±0.017 | 0.0037±0.0012 | 26 | 3.92 M | 410 | 0.05 |
| scratch d256 L4 + bias | 1e-3 | 1.0 | 20 | 0.895±0.010 | 0.895±0.013 | 0.895±0.009 | 0.0039±0.0013 | 27 | 3.92 M | 272 | 0.09 |
| pretrained + bias, **shuffled A** | 1e-2 | 3e-2 | 10 | 0.688±0.037 | 0.710±0.037 | 0.687±0.039 | 0.0008±0.0003 | 5 | 9.82 M | 346 | 0.32 |

The bias table adds 320 parameters (32 heads × 10 buckets; 80 for the 8-head scratch bodies).
"edge": random + bias selects the top of the bias-LR grid (val 1.0 → 3.0: 0.9178 → 0.9197,
+0.002; test 0.919 → 0.922). By the pre-registered rule one more step would have been run; the
phase was closed by the redirection instead and the cell is reported as an edge.

### Per-seed AUROC / AP@1:1, headline arms (selected cells)

| seed | none | pretrained + bias | random + bias | pretrained | random |
|---|---|---|---|---|---|
| 0 | 0.915/0.926 | 0.919/0.925 | 0.922/0.926 | 0.890/0.902 | 0.884/0.892 |
| 1 | 0.918/0.925 | 0.911/0.916 | 0.916/0.928 | 0.850/0.866 | 0.890/0.902 |
| 2 | 0.923/0.936 | 0.913/0.927 | 0.926/0.933 | 0.887/0.904 | 0.891/0.905 |
| 3 | 0.924/0.935 | 0.907/0.922 | 0.914/0.925 | 0.885/0.896 | 0.872/0.888 |
| 4 | 0.920/0.929 | 0.912/0.915 | 0.926/0.926 | 0.876/0.891 | 0.883/0.890 |
| 5 | 0.911/0.924 | 0.890/0.904 | 0.909/0.913 | 0.846/0.862 | 0.873/0.884 |
| 6 | 0.927/0.940 | 0.919/0.929 | 0.922/0.935 | 0.868/0.885 | 0.892/0.909 |
| 7 | 0.917/0.937 | 0.912/0.927 | 0.918/0.928 | 0.887/0.907 | 0.893/0.910 |
| 8 | 0.921/0.936 | 0.906/0.921 | 0.925/0.935 | 0.899/0.911 | 0.898/0.911 |
| 9 | 0.909/0.920 | 0.903/0.912 | 0.928/0.929 | 0.889/0.900 | 0.891/0.902 |
| 10 | 0.914/0.934 | 0.894/0.914 | 0.924/0.928 | 0.876/0.895 | 0.883/0.904 |
| 11 | 0.920/0.926 | 0.907/0.912 | 0.915/0.912 | 0.894/0.902 | 0.892/0.905 |
| 12 | 0.922/0.936 | 0.917/0.929 | 0.929/0.937 | 0.868/0.874 | 0.887/0.903 |
| 13 | 0.913/0.934 | 0.907/0.918 | 0.918/0.925 | 0.878/0.895 | 0.894/0.906 |
| 14 | 0.923/0.939 | 0.901/0.913 | 0.922/0.929 | 0.893/0.911 | 0.894/0.909 |
| 15 | 0.919/0.936 | 0.926/0.941 | 0.932/0.941 | 0.882/0.901 | 0.902/0.912 |
| 16 | 0.903/0.926 | 0.904/0.915 | 0.933/0.936 | 0.872/0.895 | 0.883/0.900 |
| 17 | 0.912/0.931 | 0.883/0.900 | 0.919/0.928 | 0.883/0.899 | 0.895/0.908 |
| 18 | 0.923/0.936 | 0.925/0.936 | 0.924/0.930 | 0.882/0.890 | 0.897/0.910 |
| 19 | 0.923/0.940 | 0.926/0.940 | 0.921/0.928 | 0.906/0.922 | 0.885/0.901 |

Seeds 0–9 vs 10–19 agree for every arm (pretrained + bias 0.909 / 0.909; random + bias
0.921 / 0.924; none 0.918 / 0.917): the extension moved nothing.

## Paired per-seed deltas — same split per seed; mean ± SE (paired t, p, MDE at α 0.05 / power 0.8); n = 20 unless noted

| Δ (A − B) | AUROC | AP@1:1 | AP(sparse) |
|---|---|---|---|
| **Q1** pretrained + bias − pretrained | **+0.0286 ± 0.0037** (t 7.76, p < 0.001, MDE 0.011) | +0.0253 ± 0.0036 (p < 0.001) | +0.0009 ± 0.0011 (p 0.41) |
| random + bias − random | **+0.0331 ± 0.0018** (t 18.74, p < 0.001, MDE 0.005) | +0.0261 ± 0.0017 (p < 0.001) | −0.0027 ± 0.0006 (p < 0.001) |
| scratch L4 + bias − scratch L4 | +0.0243 ± 0.0048 (t 5.07, p < 0.001, MDE 0.014) | +0.0190 ± 0.0048 (p 0.001) | +0.0001 ± 0.0003 (p 0.60) |
| scratch L1 + bias − scratch L1 | +0.0128 ± 0.0038 (t 3.33, p 0.004, MDE 0.011) | +0.0083 ± 0.0041 (p 0.059) | −0.0001 ± 0.0005 (p 0.83) |
| **Q2** pretrained + bias − none | **−0.0088 ± 0.0022** (t −4.02, p 0.001, MDE 0.006) | −0.0116 ± 0.0019 (p < 0.001) | −0.0054 ± 0.0009 (p < 0.001) |
| random + bias − none | +0.0041 ± 0.0020 (t 2.05, p 0.055, MDE 0.006) | −0.0037 ± 0.0015 (p 0.024) | −0.0074 ± 0.0011 (p < 0.001) |
| **Q3** pretrained + bias − random + bias | **−0.0129 ± 0.0025** (t −5.21, p < 0.001, MDE 0.007) | −0.0078 ± 0.0021 (p 0.001) | +0.0020 ± 0.0009 (p 0.044) |
| pretrained − random | −0.0085 ± 0.0032 (t −2.68, p 0.015, MDE 0.009) | −0.0071 ± 0.0031 (p 0.033) | −0.0016 ± 0.0010 (p 0.11) |
| pretrained − none | −0.0374 ± 0.0032 (t −11.73, p < 0.001) | −0.0369 ± 0.0030 (p < 0.001) | −0.0063 ± 0.0011 (p < 0.001) |
| pretrained + bias − scratch L4 + bias | +0.0145 ± 0.0035 (t 4.17, p 0.001, MDE 0.010) | +0.0258 ± 0.0034 (p < 0.001) | +0.0082 ± 0.0009 (p < 0.001) |
| pretrained + bias − scratch L1 + bias | +0.0198 ± 0.0031 (t 6.45, p < 0.001, MDE 0.009) | +0.0293 ± 0.0030 (p < 0.001) | +0.0076 ± 0.0009 (p < 0.001) |
| pretrained + bias − shuffled A (n = 10) | **+0.2210 ± 0.0135** (t 16.35, p < 0.001) | +0.2101 ± 0.0129 (p < 0.001) | +0.0098 ± 0.0010 (p < 0.001) |

## Validation-AUROC surfaces (mean over seeds 0–9; `*` = selected)

pretrained + bias (rows: encoder LR, columns: bias LR):

| enc lr | 3.0 | 1.0 | 3e-1 | 1e-1 | 3e-2 | 1e-2 | 3e-3 |
|---|---|---|---|---|---|---|---|
| 3e-2 | – | – | – | – | 0.8437 | 0.7875 | – |
| 1e-2 | 0.9042 | **0.9057 \*** | 0.8895 | 0.8893 | 0.8861 | 0.8813 | 0.8798 |
| 3e-3 | – | – | – | – | 0.8755 | 0.8672 | – |

random + bias:

| enc lr | 3.0 | 1.0 | 3e-1 | 1e-1 | 3e-2 | 1e-2 | 3e-3 | 1e-3 | 3e-4 |
|---|---|---|---|---|---|---|---|---|---|
| 1e-2 | – | – | – | – | 0.8844 | 0.8805 | – | – | – |
| 3e-3 | **0.9197 \*** | 0.9178 | 0.9073 | 0.8911 | 0.8899 | 0.8881 | 0.8968 | 0.8865 | 0.8929 |
| 1e-3 | – | – | – | – | 0.8768 | 0.8763 | – | – | – |

scratch L4 + bias (enc 1e-3): 3.0 0.9033 · 1.0 **0.9043 \*** · 3e-1 0.8978 · 1e-1 0.8981 · 3e-2 0.8810
· 1e-2 0.8811 · 3e-3 0.8784. scratch L1 + bias (enc 1e-3): 3.0 0.8913 · 1.0 **0.8924 \*** · 3e-1
0.8907 · 1e-1 0.8907 · 3e-2 0.8845 · 1e-2 0.8858 · 3e-3 0.8849. Every biased arm selects a bias LR
two orders of magnitude above GTLM's 5e-3–4e-2. The surface still rises from 0.3 to 1.0
(+0.016 pretrained, +0.011 random, +0.007 scratch L4, +0.002 scratch L1) and is flat within
0.002 between 1.0 and 3.0. The main-stage grid (≤ 1e-1) sat on the rising flank for pretrained
and both scratch arms, whose edge rule fired upward; random + bias peaked at the bottom of that
grid (3e-3) and its edge rule fired downward (1e-3, 3e-4) — its 0.3 / 1.0 / 3.0 cells, and the
3.0 cell of every arm, came from the top-up.

## The learned tables (mean over seeds and heads; bucket 1 … 9)

| arm | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 far | max \|b\| | mean \|b\| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pretrained + bias (lr 1.0) | −1.84 | −1.89 | −1.24 | −0.71 | −0.72 | −0.43 | +0.05 | +0.21 | +2.42 | 42.9 | 4.6 |
| random + bias (lr 3.0) | −5.41 | +1.15 | −2.87 | −3.70 | −3.34 | −1.72 | +0.21 | +0.44 | −0.86 | 68.1 | 9.2 |
| scratch L4 + bias (lr 1.0) | +0.10 | −2.82 | −3.81 | −4.40 | −2.12 | +0.53 | +1.24 | +1.17 | −3.34 | 32.7 | 5.5 |
| shuffled A (lr 3e-2) | +0.06 | +0.07 | −0.04 | −0.02 | +0.04 | +0.07 | +0.06 | 0.00 | −0.09 | 6.7 | 0.3 |

At the selected learning rates the tables are **hard routing masks**: entries of 20–70 on top
of unit-scale q·k logits make each head's attention a function of hop distance, not content.
Per head the pattern is stable across seeds for the pretrained body — 17 of 32 heads
consistently attend *away* from the ≤ 3-hop neighbourhood (d1–3 < far in ≥ 80 % of seeds),
none consistently towards it, 15 mixed. The random body's heads do not keep a role across
seeds (0 consistently near, 4 consistently far, 28 mixed); on average its table suppresses the
1-hop and 3–6-hop keys and leaves the 2-hop bucket (+1.15, the common-neighbour channel) and
the 7–8-hop buckets positive, but which head does what changes with the seed. The shuffled
control learns almost nothing (mean |b| 0.3): with structure destroyed
there is nothing for the table to route on.

## Controls and checks

- **Shuffled A** (one degree-preserving rewiring used as the input at train and test, so the
  bias sees the rewired graph too; run on its own two-point grid, bias LR {1e-2, 3e-2}, selected
  3e-2): 0.688 against 0.909 at the real arm's selected cell, Δ +0.221 (p < 0.001, n = 10).
  Structure is what the bias exploits.
- **Inertness at init**: 600 biased rows compared with their bias-off counterpart on five
  first-batch statistics (token norm, logit diagonal / off-diagonal std / train-edge mean,
  per-layer hidden RMS): 0 mismatches above 1e-2 relative. The zero-init table makes every
  biased model start at exactly the Phase-2 model.
- **Reproduction of Phase 2** at this commit, same seeds: none (5), pretrained (10), random
  (10) — mean |Δ AUROC| 0.0000; scratch d256 L4 (5): 0.0068 mean, 0.0124 max — the eval
  fast-path fix above.
- **Leak rule**: the bias is computed from the model's input only (`test_bias.py`: the bias
  tensor is identical under flips of held-out cells; `test_leaks.py`).
- **Data-fraction sweep (uninformative — design error, recorded).** Train edges kept at
  10 / 25 / 50 % for pretrained + bias, random + bias and none, 10 seeds each (90 rows). All
  arms collapse to 0.57–0.70 (lift 2–12, best epochs 24–136), and pretrained + bias sits below none at
  every fraction (−0.020 / −0.018 / −0.051). The cause is the loss target: dropped train edges
  are labelled **0** in the masked-cell supervision (`train.py` targets `A_train`), so "10 %"
  means 90 % of the positives are mislabelled — the sweep measures label-noise tolerance, not
  data efficiency. A valid version would exclude dropped edges from the supervised cells. Not
  re-run; the rows stay in `rows/` for the record and carry no claim.

## Probe (commit `8f3b78a`, before the array)

512-node subgraph, pretrained + bias, 300 steps: loss 1.229 → 0.573, table non-zero and
receiving gradient at step 0 (the encoder-gradient assertion runs in every array run, not in the
probe). Full Cora: 0.343 s/epoch, 30.9 GB peak.

## Walkthrough (laptop, RTX 5070, `python -m g2l.walkthrough --phase 3`, 2026-08-27)

`tests/test_bias.py` + `test_llm.py` + `test_llm_real.py` live: **23 passed**. Pretrained + bias
seed 0 rebuilt from the Marlowe checkpoint; full 2708 × 2708 logits recomputed under `no_grad`
and compared with the logits saved by the cluster job: **max |Δ logit| = 0.0078** (fp16 saved
logits; asserted < 0.02); AUROC 0.91884 / AP 0.92465 / AP(sparse) 0.00818 identical to the
stored row. Its learned table (mean over heads): d1 −1.20, d2 −1.41, d3 −1.17, d4 +0.42, …,
d9 +0.63; max |b| 19.5. Suite: 52 tests on the laptop (49 on Marlowe at `a60dc60`, before the
Phase-3B tests). Log: `results/phase3/walkthrough.log`.

## Gate checklist

- [x] probe recorded (memory, s/epoch, table moves, gradients)
- [x] 49 / 52 tests green on Marlowe / laptop; the eval fast-path regression test in place
- [x] bias-equality-under-held-out-changes (leak) test runs for real
- [x] inertness at init verified on 600 rows; Phase-2 rows reproduced (0.0000 on three arms)
- [x] every selected cell interior except random + bias (top of the bias-LR grid, flat, recorded)
- [x] paired deltas with SE / t / p / MDE, n = 20 on every gate question
- [x] shuffled-A control degrades (−0.22)
- [x] learned tables reported per distance and per head
- [x] seed-extension and fraction rules applied as written; the edge rule applied through the
      follow-up and the top-up, except that random + bias's final cell (top of the grid, +0.002
      val) was not extended once more — the phase was closed by the redirection (recorded above);
      fraction sweep declared uninformative with the cause
- [x] checkpoints pruned to the two selected seed-0 runs (`$SCR/runs/phase3/`)
- [ ] walkthrough run by the user and signed

## What the numbers show

1. **A structural channel into attention works, for any body.** +0.029 (pretrained), +0.033
   (random), +0.024 / +0.013 (scratch), all p ≤ 0.004, with MDEs of 0.005–0.014. The random
   body recovers all of what it cost in Phase 2 (random + bias − none +0.004, n.s.); the
   pretrained body recovers most of it (−0.037 → −0.009).
2. **It works by routing, not by nudging.** The selected bias LR is 1.0–3.0 and the tables
   reach |b| of 20–70: attention becomes a hop-distance mask. On average the random body's
   table keeps only the 2-hop (common-neighbour) bucket positive within six hops; the pretrained
   body's heads mostly route away from the local neighbourhood (17 of 32 consistently).
3. **The pretrained language weights are a cost.** Pretrained − random: −0.009 without the
   bias (p 0.015) and −0.013 with it (p < 0.001). The pretrained body is a specific geometry
   the encoder has to work against; random weights with structural routing do better.
4. **Nothing with a frozen LLM beats the linear model.** The best arm ties none on AUROC
   (+0.004, p 0.055) and loses on AP (−0.004, p 0.024) and AP(sparse) (−0.007). E1 + D1 —
   a bilinear form on adjacency rows — is at the ceiling of this protocol on Cora, above
   GAE / GAT, and a fixed-N encoder on one graph can memorise it. Cora can sanity-check a
   model here; it cannot rank bodies.
5. **The graph is everything.** Shuffled A: −0.22. The trained tables are ~0 on the rewired
   graph.

## Does not show

- Anything about a *trained* body: every body here is frozen (Llama) or small (scratch d256).
  Phase 4 trains a graph-native body and compares it with the same architecture frozen.
- Anything about LoRA (Phase 3B built, not run — `docs/PLAN-PHASE3B.md`).
- Data efficiency (the fraction sweep is void, above).
- Whether bias LR > 3.0 would add to random + bias (+0.011 val from 0.3 to 1.0, +0.002 from 1.0 to 3.0).
- Any number beyond Cora: single graph, N = 2708, fixed-N encoder.

## Locked going forward

`configs/phase3.yaml` as committed: the Phase-2 recipe plus `max_dist` 8 (10 buckets), bias
LR without weight decay, `selected` = {pretrained 1e-2 / 1.0, random 3e-3 / 3.0}. Kept on the
cluster: `$SCR/runs/phase3/{pretrained_spd_real_f1.0_lr1e-02_lb1e+00_s0, random_spd_real_f1.0_lr3e-03_lb3e+00_s0}.{pt,logits.pt}`.
Phase 4 (`docs/PLAN-LGM.md`; its grid is pre-registered in `docs/PLAN-PHASE4.md` before
submission) keeps the bias, the loss, the protocol and these rows as references, and replaces
the frozen LLM with a trainable graph transformer.

## One slide

**Phase 3 — graph structure into attention.** Cora masked edge prediction, 20 paired seeds.
A zero-init per-head hop-distance bias on every attention layer: frozen Llama-1B 0.881 → 0.909
(+0.029, p < 0.001); same body random-init 0.889 → 0.922 (+0.033); scratch transformer +0.024.
The learned table is a hard routing mask (|b| up to 68); on a rewired graph the model falls to
0.688. Pretrained − random = −0.013 (p < 0.001): the language weights hurt. Best LLM arm 0.922
= linear model 0.918 (tie). Structure is the ingredient, the LLM is not → we build the graph
model (LGM). Reproduced on the laptop from the cluster checkpoint to five decimals.

VALIDATED: ____________
