# Phase 2 — Frozen LLM core (no attention bias, no LoRA)

**Claim:** with adjacency-row tokens, the masked-cell loss and matched trainables (encoder +
decoder ≈ 9.8 M), inserting a **frozen Llama-3.2-1B body** between E1 and D1 **costs 0.043
AUROC** against no body at all (0.878 vs 0.920, paired p = 0.004); a **pretrained** and a
**random-init** frozen body are **indistinguishable** (Δ = −0.009, p = 0.11 over 10 paired
seeds); and the frozen
body sits **above every transformer trained from scratch** on the same tokens on point
estimates (0.818–0.873), significantly so from width 512 up (+0.038, p = 0.04).

The pre-bias delta table below is the Gate-2 deliverable; the graph-aware attention bias
(Phase 3) is the intervention meant to turn the body from a cost into a gain.

Produced on Marlowe (H100 80 GB, 1 GPU per run; probe 448175, D1 grid 448192, arm-4 sweep
448189, LR extensions 448961 / 448976 / 448989, D3 449017, controls 449018, seed extension
449155). All 230 training rows are at code commit `1f1f33a`, which stayed checked out on
the cluster for the whole phase; between stages only `configs/phase2.yaml` changed, and every
row records commit, config hash, arm, LR and seed. (The probe ran one commit earlier,
`706f41c`, before the `--chunk` runner change.) 10.3 GPU-hours of training wall-clock
(pretrained 3.7, random 3.6, none 1.9, scratch 1.1). Recomputed on the laptop by
`python -m g2l.walkthrough --phase 2` (below).

## Locked setup

| item | value |
|---|---|
| data | Cora, `RandomLinkSplit` 85/5/10, undirected, 5 seeds; the seed fixes the split (positives and the 527 sampled test negatives) and every initialisation. The per-epoch masking pattern is seeded by the epoch index, so it is identical across seeds. Scorer: `evaluate_edge_split` — the Phase-1 protocol |
| tokens | E1: `Linear(2708 → 2048)`, orthogonal init, then `LayerNorm` with gain **T/√d**, T = 0.987007 = mean L2 norm of the pretrained embedding rows (random body: T = 0.905) |
| body | `meta-llama/Llama-3.2-1B` @ `4e20de36…`, `AutoModel`, eager attention, bf16, `embed_tokens` stubbed, RoPE = identity (`position_ids = 0`), bidirectional via a 4-D float zero mask (Implementation A), `use_cache=False`; RMSNorm gains fp32 and trainable at lr 1e-4; everything else frozen. Random arm: same config, seeded `AutoModel.from_config` |
| parity procedure | `tests/test_llm_real.py`: on the untouched bf16 stock model, hidden states from the stub + 4-D zero bf16 mask + `position_ids = 0` path are `torch.equal` to those from `is_causal=False, attention_mask=None`; the causal mask gives a different result (negative control). The fp32-RMSNorm-under-autocast recipe drifts from the pure-bf16 model by 1.7e-3 relative Frobenius norm. `FrozenBody.canary()` (16 random tokens, zero mask vs no mask, atol 1e-2) runs at the start of every frozen-body training run |
| decoders | D1: `Z W Zᵀ`, W = 0.1·I init, `LayerNorm(gain 1/√d)` on the decoder input in every arm; D3: MLP(3d → 512 → 1) on `[z_i ‖ z_j ‖ z_i ⊙ z_j]`, chunked |
| loss | BCE-with-logits on **masked cells only**: 15 % of upper-triangular cells re-drawn every epoch, hidden from the input row, target = train graph, `pos_weight` = 815.68 |
| optimiser | AdamW, wd 0.01, 100 warm-up epochs, grad-clip 1.0, ≤ 2000 epochs, early stop on val AUROC with patience 200, best-val state scored on test |
| LR | grid {1e-3, 3e-4, 1e-4} per arm, **edge rule**: whenever the arg-max sat on a grid edge, one point further (×3) was added, up to three rounds; selection = best mean val AUROC over seeds 0–4 |
| memory | no gradient checkpointing (probe: 0.319 s/epoch, 30.8 GB peak vs 0.366 s/epoch, 6.3 GB with) |

Arms: (1) **pretrained** frozen body · (2) **none** — E1 → LayerNorm → D1, no body, the same
trainables minus the RMSNorm gains · (3) **random** frozen body · (4) **scratch** — a 4-layer
pre-LN transformer trained from scratch at widths 128 … 2048. Load asserts (eager,
`attention_scaling == 1`, Llama-3.2-1B geometry, bf16 body) run in every frozen-body job;
arms 2 and 4 have no body to assert on.

## Table — Cora, mean ± std over seeds, each arm at its selected LR

AUROC and AP@1:1 are scored on the 527 held-out test edges plus the split's 527 seeded
non-edges, exactly as in Phase 1; AUROC(sparse)/AP(sparse) score all 3.66 M masked cells at
the true prevalence 0.000144; lift = AP(sparse) / prevalence. Reference rows are the Phase-1
numbers on the same split objects and scorer.

| model | loss | lr | n | AUROC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift | trainable | best epoch | s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| identity | – | – | 5 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.0001±0.0000 | 1 | – | – | – |
| random scores | – | – | 5 | 0.497±0.023 | 0.503±0.018 | 0.498±0.010 | 0.0002±0.0000 | 1 | – | – | – |
| PPR (Phase 1) | – | – | 5 | 0.850±0.012 | 0.900±0.010 | 0.850±0.012 | 0.0308±0.0049 | 214 | – | – | – |
| GAE 600 ep (Phase 1) | recon | – | 5 | 0.905±0.006 | 0.912±0.005 | 0.908±0.006 | 0.0078±0.0016 | 54 | – | – | – |
| GAT (Phase 1) | recon | – | 5 | 0.910±0.009 | 0.911±0.008 | 0.910±0.007 | 0.0052±0.0007 | 36 | – | – | – |
| **none** (E1 + D1) | masked | 3e-4 | 5 | **0.920±0.003** | **0.930±0.005** | 0.920±0.002 | 0.0146±0.0039 | 101 | 9.75 M | 302 | 0.04 |
| **pretrained** frozen body + D1 | masked | 1e-2 | 10 | 0.878±0.017 | 0.892±0.016 | 0.877±0.018 | 0.0107±0.0035 | 74 | 9.82 M | 449 | 0.30 |
| **random** frozen body + D1 | masked | 3e-3 | 10 | 0.887±0.008 | 0.899±0.009 | 0.887±0.008 | 0.0116±0.0025 | 81 | 9.82 M | 409 | 0.44 |
| scratch d128 | masked | 3e-3 | 5 | 0.873±0.009 | 0.875±0.009 | 0.876±0.006 | 0.0033±0.0005 | 23 | 1.16 M | 286 | 0.05 |
| scratch d256 | masked | 1e-3 | 5 | 0.870±0.003 | 0.876±0.002 | 0.872±0.006 | 0.0035±0.0007 | 24 | 3.92 M | 432 | 0.05 |
| scratch d512 | masked | 3e-4 | 5 | 0.840±0.019 | 0.839±0.017 | 0.845±0.022 | 0.0022±0.0006 | 16 | 14.26 M | 392 | 0.06 |
| scratch d1024 | masked | 1e-4 | 5 | 0.822±0.011 | 0.823±0.010 | 0.826±0.009 | 0.0021±0.0004 | 15 | 54.21 M | 384 | 0.08 |
| scratch d2048 | masked | 3e-4 | 5 | 0.818±0.009 | 0.827±0.020 | 0.823±0.010 | 0.0023±0.0006 | 16 | 211.18 M | 451 | 0.18 |
| none + **D3** | masked | 3e-4 | 5 | 0.903±0.003 | 0.921±0.004 | 0.905±0.004 | 0.0212±0.0018 | 147 | 8.70 M | 413 | 1.85 |
| pretrained + **D3** | masked | 1e-2 | 5 | 0.891±0.008 | 0.906±0.010 | 0.895±0.010 | 0.0124±0.0026 | 86 | 8.77 M | 374 | 1.78 |
| random + **D3** | masked | 3e-3 | 5 | 0.886±0.006 | 0.901±0.005 | 0.891±0.003 | 0.0123±0.0007 | 85 | 8.77 M | 593 | 1.82 |
| none, **shuffled A** | masked | 3e-4 | 5 | 0.772±0.013 | 0.785±0.012 | 0.773±0.008 | 0.0037±0.0013 | 26 | 9.75 M | 496 | 0.05 |
| pretrained, **shuffled A** | masked | 1e-2 | 5 | 0.691±0.025 | 0.710±0.025 | 0.699±0.022 | 0.0010±0.0005 | 7 | 9.82 M | 296 | 0.30 |
| none, **recon_bce** | recon | 1e-4 | 5 | 0.841±0.014 | 0.888±0.010 | 0.839±0.014 | 0.0202±0.0046 | 140 | 9.75 M | 862 | 0.01 |
| none, **encoder gain 1** | masked | 3e-4 | 5 | 0.918±0.004 | 0.928±0.006 | 0.918±0.002 | 0.0129±0.0051 | 89 | 9.75 M | 299 | 0.05 |

Peak memory: frozen body + D1 30.8 GB, frozen body + D3 33.1 GB, none + D1 0.5 GB, none + D3
4.9 GB. The pretrained and random rows pool seeds 0–9 (seed extension below); every other
row is seeds 0–4. The random arm's seed 6 ran at 1.70 s/epoch (a slow allocation; the other
nine seeds at 0.30, same 30.8 GB peak), which is what lifts its mean s/epoch to 0.44.

### Per-seed AUROC / AP@1:1 (selected LR, seeds 0–4)

| model | s0 | s1 | s2 | s3 | s4 | best epoch |
|---|---|---|---|---|---|---|
| none | 0.915/0.926 | 0.918/0.925 | 0.923/0.936 | 0.924/0.935 | 0.920/0.929 | 419, 223, 246, 341, 283 |
| pretrained | 0.890/0.902 | 0.850/0.866 | 0.887/0.904 | 0.885/0.896 | 0.876/0.891 | 264, 282, 825, 339, 195 |
| random | 0.884/0.892 | 0.890/0.902 | 0.891/0.905 | 0.872/0.888 | 0.883/0.890 | 350, 356, 296, 481, 329 |
| scratch d128 | 0.858/0.858 | 0.876/0.872 | 0.882/0.884 | 0.870/0.876 | 0.880/0.883 | 311, 123, 384, 332, 279 |
| scratch d256 | 0.873/0.872 | 0.869/0.875 | 0.874/0.880 | 0.871/0.877 | 0.865/0.875 | 324, 532, 391, 571, 343 |
| scratch d512 | 0.868/0.862 | 0.854/0.851 | 0.818/0.826 | 0.840/0.842 | 0.819/0.815 | 342, 576, 380, 328, 332 |
| scratch d1024 | 0.833/0.833 | 0.807/0.810 | 0.829/0.819 | 0.829/0.836 | 0.810/0.815 | 379, 481, 437, 354, 269 |
| scratch d2048 | 0.826/0.815 | 0.813/0.841 | 0.803/0.798 | 0.827/0.856 | 0.820/0.823 | 450, 11, 1011, 10, 771 |
| none + D3 | 0.902/0.918 | 0.898/0.920 | 0.906/0.928 | 0.904/0.923 | 0.905/0.917 | 302, 549, 438, 489, 289 |
| pretrained + D3 | 0.893/0.909 | 0.893/0.912 | 0.895/0.913 | 0.898/0.911 | 0.875/0.886 | 268, 283, 266, 480, 572 |
| random + D3 | 0.877/0.894 | 0.882/0.898 | 0.892/0.904 | 0.894/0.909 | 0.886/0.901 | 441, 340, 694, 901, 588 |
| none, shuffled A | 0.788/0.796 | 0.762/0.769 | 0.783/0.794 | 0.773/0.793 | 0.753/0.771 | 459, 542, 413, 557, 511 |
| pretrained, shuffled A | 0.708/0.727 | 0.702/0.692 | 0.693/0.724 | 0.710/0.737 | 0.642/0.671 | 237, 350, 297, 465, 132 |
| none, recon_bce | 0.845/0.885 | 0.820/0.874 | 0.851/0.896 | 0.830/0.882 | 0.860/0.901 | 1167, 583, 688, 1066, 807 |
| none, gain 1 | 0.914/0.924 | 0.913/0.921 | 0.919/0.932 | 0.926/0.937 | 0.918/0.927 | 415, 223, 246, 328, 285 |

### Seed extension (pre-registered, plan §8 decision 3)

Trigger: |pretrained − random| = 0.007 < MDE 0.034 on seeds 0–4, so arms 1 and 3 were run
on seeds 5–9 at their selected LRs (10 runs, job 449155; LR selection itself used seeds 0–4
only).

| model | s5 | s6 | s7 | s8 | s9 | best epoch | mean AUROC seeds 0–4 / 5–9 |
|---|---|---|---|---|---|---|---|
| pretrained | 0.846/0.862 | 0.868/0.885 | 0.887/0.907 | 0.899/0.911 | 0.889/0.900 | 508, 479, 985, 292, 321 | 0.8777 / 0.8778 |
| random | 0.873/0.884 | 0.892/0.909 | 0.893/0.910 | 0.898/0.911 | 0.891/0.902 | 402, 643, 344, 398, 489 | 0.8842 / 0.8893 |

Over 10 paired seeds: pretrained − random = −0.0091 ± 0.0051 AUROC (t −1.77, p 0.11,
MDE 0.016), −0.0069 ± 0.0048 AP@1:1 (p 0.19), −0.0010 ± 0.0014 AP(sparse) (p 0.53). The
extension halved the MDE and moved nothing: a pretrained frozen body is not better than a
random one, and if anything slightly worse.

## Paired per-seed deltas — same split per seed; mean ± SE (paired t, p, MDE at α 0.05 / power 0.8); n = 5 except pretrained − random (n = 10)

| Δ (A − B) | AUROC | AP@1:1 | AP(sparse) |
|---|---|---|---|
| pretrained − none | **−0.0425 ± 0.0070** (t −6.05, p 0.004, MDE 0.0261) | −0.0382 ± 0.0058 (t −6.53, p 0.003, MDE 0.0217) | −0.0023 ± 0.0024 (t −0.97, p 0.386, MDE 0.0089) |
| pretrained − random (n = 10) | −0.0091 ± 0.0051 (t −1.77, p 0.111, MDE 0.0161) | −0.0069 ± 0.0048 (t −1.43, p 0.186, MDE 0.0151) | −0.0010 ± 0.0014 (t −0.66, p 0.526, MDE 0.0046) |
| random − none | −0.0359 ± 0.0043 (t −8.34, p 0.001, MDE 0.0160) | −0.0349 ± 0.0039 (t −8.90, p 0.001, MDE 0.0146) | −0.0035 ± 0.0021 (t −1.66, p 0.172, MDE 0.0079) |
| pretrained − scratch d128 | +0.0045 ± 0.0097 (t +0.47, p 0.662, MDE 0.0359) | +0.0172 ± 0.0082 (t +2.09, p 0.105, MDE 0.0306) | +0.0090 ± 0.0017 (t +5.15, p 0.007, MDE 0.0065) |
| pretrained − scratch d256 | +0.0075 ± 0.0065 (t +1.16, p 0.312, MDE 0.0242) | +0.0160 ± 0.0068 (t +2.35, p 0.078, MDE 0.0253) | +0.0087 ± 0.0019 (t +4.60, p 0.010, MDE 0.0071) |
| pretrained − scratch d512 | **+0.0380 ± 0.0129** (t +2.94, p 0.042, MDE 0.0480) | +0.0528 ± 0.0117 (t +4.50, p 0.011, MDE 0.0436) | +0.0100 ± 0.0016 (t +6.20, p 0.003, MDE 0.0060) |
| random − scratch d128 | +0.0111 ± 0.0044 (t +2.50, p 0.067, MDE 0.0165) | +0.0206 ± 0.0051 (t +4.05, p 0.015, MDE 0.0189) | +0.0078 ± 0.0010 (t +7.78, p 0.001, MDE 0.0037) |
| pretrained + D3 − none + D3 | −0.0121 ± 0.0046 (t −2.65, p 0.057, MDE 0.0169) | −0.0151 ± 0.0043 (t −3.50, p 0.025, MDE 0.0160) | −0.0088 ± 0.0013 (t −6.54, p 0.003, MDE 0.0050) |
| pretrained + D3 − random + D3 | +0.0047 ± 0.0046 (t +1.02, p 0.363, MDE 0.0170) | +0.0047 ± 0.0054 (t +0.88, p 0.429, MDE 0.0200) | +0.0001 ± 0.0014 (t +0.09, p 0.929, MDE 0.0052) |
| random + D3 − none + D3 | −0.0168 ± 0.0024 (t −6.93, p 0.002, MDE 0.0090) | −0.0198 ± 0.0019 (t −10.56, p < 0.001, MDE 0.0070) | −0.0089 ± 0.0010 (t −9.34, p 0.001, MDE 0.0035) |
| D3 − D1, pretrained | +0.0132 ± 0.0077 (t +1.72, p 0.160, MDE 0.0286) | +0.0141 ± 0.0086 (t +1.65, p 0.175, MDE 0.0318) | +0.0002 ± 0.0013 (t +0.16, p 0.880, MDE 0.0050) |
| D3 − D1, none | −0.0171 ± 0.0013 (t −13.00, p < 0.001, MDE 0.0049) | −0.0091 ± 0.0012 (t −7.52, p 0.002, MDE 0.0045) | +0.0066 ± 0.0023 (t +2.84, p 0.047, MDE 0.0087) |
| D3 − D1, random | +0.0020 ± 0.0054 (t +0.37, p 0.732, MDE 0.0202) | +0.0060 ± 0.0043 (t +1.39, p 0.237, MDE 0.0161) | +0.0013 ± 0.0011 (t +1.13, p 0.323, MDE 0.0042) |
| none − none gain 1 | +0.0021 ± 0.0010 (t +2.10, p 0.104, MDE 0.0036) | +0.0021 ± 0.0011 (t +1.93, p 0.126, MDE 0.0040) | +0.0017 ± 0.0007 (t +2.29, p 0.083, MDE 0.0028) |
| none − none shuffled A | **+0.1484 ± 0.0069** (t +21.52, p < 0.001, MDE 0.0256) | +0.1454 ± 0.0051 (t +28.43, p < 0.001, MDE 0.0190) | +0.0108 ± 0.0022 (t +4.98, p 0.008, MDE 0.0081) |
| pretrained − pretrained shuffled A | **+0.1867 ± 0.0141** (t +13.23, p < 0.001, MDE 0.0525) | +0.1816 ± 0.0101 (t +18.05, p < 0.001, MDE 0.0374) | +0.0113 ± 0.0017 (t +6.79, p 0.002, MDE 0.0062) |

## Validation-AUROC curves (mean over seeds 0–4; `*` = selected)

| model | 3e-2 | 1e-2 | 3e-3 | 1e-3 | 3e-4 | 1e-4 | 3e-5 |
|---|---|---|---|---|---|---|---|
| none | – | – | – | 0.9103 | 0.9105 * | 0.9076 | – |
| pretrained | 0.8393 | 0.8785 * | 0.8475 | 0.7999 | 0.7179 | 0.6826 | – |
| random | – | 0.8853 | 0.8892 * | 0.8501 | 0.7704 | 0.7610 | – |
| scratch d128 | – | 0.8802 | 0.8888 * | 0.8785 | 0.8610 | 0.8093 | – |
| scratch d256 | – | – | 0.8681 | 0.8840 * | 0.8691 | 0.8055 | – |
| scratch d512 | – | – | – | 0.8495 | 0.8651 * | 0.8472 | – |
| scratch d1024 | – | – | – | 0.8408 | 0.8300 | 0.8443 * | 0.7997 |
| scratch d2048 | – | – | – | 0.7928 | 0.8359 * | 0.8348 | – |
| none, recon_bce | – | – | – | 0.8280 | 0.8231 | 0.8316 * | – |
| none, gain 1 | – | – | – | 0.9078 | 0.9080 * | 0.9056 | – |

The edge rule fired three times: round 1 (3e-3 for pretrained, random, d128, d256; 3e-5 for
d1024) settled d256 and d1024; round 2 (1e-2) settled random and d128; round 3 (3e-2)
settled pretrained. All three rounds are in `configs/phase2.yaml` (`extend:`). **The frozen
arms need a 10–33× larger encoder LR than the body-less arm** (1e-2 / 3e-3 vs 3e-4). At
1e-3 three of five seeds already reach the 3e-3 level (pretrained 0.842–0.866, random
0.882–0.894) but two collapse; below 1e-3 every seed is poor. No selected LR sits on an
edge. (The `recon_bce` control row is flagged EDGE by the aggregator because its flat
validation curve — 0.828 / 0.823 / 0.832 — happens to peak at 1e-4; it is a control across
the base grid, not a tuned arm, and was not extended.)

### Frozen-arm training failures at low LR (the plateau diagnosis the gate asks for)

At 1e-3, two of five seeds per frozen arm collapse: validation AUROC peaks early and then
degrades until patience-200 stops the run — pretrained s1/s3 peak at epoch 138/58 and end
at 0.671/0.708 against 0.842–0.866 for the other three; random s1/s4 peak at epoch 10/9 and
end at 0.771/0.764 against 0.882–0.894. At 3e-4 and 1e-4 the random arm fails on all five
seeds (peak by epoch 7–26, 0.72–0.80) and the pretrained arm is slow and poor on all five
(0.62–0.77; one run, `pretrained@3e-4` s1, was censored at the 2000-epoch cap while still
improving). The extended grid resolved it: at 3e-3 all five random seeds converge
(0.872–0.891), and at 1e-2 four of five pretrained seeds converge (0.876–0.890) with seed 1
lower at 0.850. Encoder gradient norms were > 0 at every logged step in every run.

## Controls

- **Shuffled A** (one degree-preserving rewiring of the train graph, used as the input at
  train *and* test): none 0.920 → 0.772 (+0.148 for the real input, p < 0.001), pretrained
  0.878 → 0.691 (+0.187). A rewired row is still a unique node identity, and the body-less
  row lands where the bring-up measurement put identity-only performance (0.799 on seed 0,
  plan §1); the pretrained arm falls further, to 0.69. Only arms 1 and 2 have this control
  (plan §7). The neighbourhood structure is what the models learn from.
- **`recon_bce`** (the Phase-1 baselines' loss, supervising cells visible in the input row)
  on the body-less arm: 0.831–0.845 across the grid against 0.920 with the masked-cell
  loss — the copy-the-input shortcut measured at identical architecture, confirming the
  Phase-1 erratum. At matched LR it peaks *earlier* than the masked arm (epoch 39–52 vs
  123–182 at 1e-3): it fits the visible cells quickly and then stalls.
- **Encoder gain 1** instead of T/√d on the body-less arm: 0.918 vs 0.920 (Δ 0.002, p 0.10).
  The +0.05 recorded during bring-up (plan §1: 0.862 vs 0.915, arm 2, one seed, one LR)
  does not survive per-configuration LR selection — on the body-less arm the gain is a
  scale knob the LR sweep absorbs. Whether it matters for the frozen arms, whose RMSNorm
  expects inputs at the embedding scale, was not tested in this phase.

## Real-weight checks (step 3, laptop, `tests/test_llm_real.py`, commit e1a8ebc)

T = 0.987007. Exact (`torch.equal`) parity of the stub + zero-mask + `position_ids = 0` path
with the stock model under `is_causal=False, attention_mask=None`; the causal mask differs.
Precision recipe (fp32 RMSNorm gains under autocast vs pure bf16): relative Frobenius drift
1.7e-3. Canary passes.

## Probe (job 448175, commit 706f41c, before any array)

512-node subgraph, 300 steps at 1e-3: loss 1.229 → 0.324, encoder grad-norm 1.6–15.9 at every
logged step. Full Cora, arm 1: 0.319 s/epoch, 30.76 GB peak without checkpointing;
0.366 s/epoch, 6.31 GB with (16 layers checkpointed) → checkpointing off. Init statistics:
token norm 1.000 in both frozen arms; per-layer hidden RMS 0.034 → 1.14 (pretrained) vs
0.858 → 4.17 (random).

## Walkthrough (laptop, RTX 5070, `python -m g2l.walkthrough --phase 2`)

Runs `tests/test_llm.py` + `tests/test_llm_real.py` live, rebuilds arm 1 seed 0 from the
Marlowe checkpoint, recomputes the full 2708×2708 logits under `no_grad` and compares with
the logits saved by the cluster job: **max |Δ logit| = 0.0078** (saved logits are fp16),
AUROC 0.89024 / AP 0.90238 / AP(sparse) 0.01606 identical to the stored row to five
decimals. The tolerance is now asserted in the walkthrough at max |Δ| < 0.02. Test suite:
41 pass on the laptop, 41 pass on Marlowe.

## Gate checklist

- [x] probe log recorded (memory, s/epoch, checkpointing decision)
- [x] 41 tests green on both machines (22 Phase 0/1 + 19 new); load asserts and the
  4-D-mask canary run in every frozen-body job (arms 1, 3, their D3 and shuffled rows)
- [x] identity 0.500 / lift 1, random scores at chance
- [x] every arm converged or diagnosed (section above); one censored run at a non-selected LR
- [x] no selected LR on a grid edge (three extension rounds)
- [x] paired D1 delta table with SE / t / p / MDE on three columns
- [x] arm-4 (scratch) width curve under the masked-cell loss
- [x] D3 table and D3 − D1 deltas
- [x] both arms with a shuffled-A control beat it (by 0.15–0.19)
- [x] `recon_bce` and gain-1 rows recorded
- [x] T = 0.987007 and drift 1.7e-3 recorded
- [x] pre-registered seed extension applied (pretrained − random below MDE)
- [ ] walkthrough run by the user and signed

## What the numbers show

1. **The reference is strong.** E1 + D1 with the masked-cell loss — a linear graph
   auto-encoder with a bilinear decoder and no body — reaches 0.920 / 0.930, above GAE
   (0.905 / 0.912) and GAT (0.910 / 0.911) on the identical protocol, with every seed above
   both. This is the number every later configuration is measured against.
2. **A frozen LLM body is a cost of 0.043 AUROC before any graph-aware attention**
   (p = 0.004; the MDE of 0.026 means an effect 60 % of this size would still have been
   detected). Without a bias, the body receives node tokens with no way to relate them
   beyond what the encoder packs into each row; sixteen fixed layers then act as a
   transformation the encoder has to invert.
3. **Pretraining does not matter yet.** Pretrained vs random frozen body: −0.009 AUROC,
   p = 0.11 over 10 paired seeds (MDE 0.016), the same under D3 (+0.005, p 0.36, 5 seeds).
   Nothing language-specific is
   being used at this stage — which is the honest baseline for Phase 3, where the bias
   gives the pretrained attention something graph-shaped to act on.
4. **A frozen body is at least as good as training a transformer from scratch, and better
   from width 512 up.** Every scratch width trails both frozen arms on point estimates
   (best 0.873 at d128; 0.818 at d2048 with 211 M trained parameters); paired, the gap is
   +0.005 / +0.008 (n.s.) against d128 / d256 and +0.038 (p 0.04) against d512. The scratch
   curve gets *worse* with width under the masked-cell loss, as it did under recon in
   Phase 1.
5. **The decoder interacts with the body.** D3 (pair MLP) hurts the body-less arm
   (−0.017, p < 0.001) and helps the pretrained arm (+0.013, p 0.16, 4/5 seeds up): the
   linear model already has a bilinear structure that D1 matches exactly, while the body's
   states benefit from a nonlinear read-out. D3 also lifts AP(sparse) for `none`
   (0.015 → 0.021).
6. **The structure carries everything.** Shuffled-A costs 0.15–0.19 AUROC; `recon_bce`
   reproduces the 0.84 shortcut ceiling; the encoder gain is immaterial without a body.
7. **At true prevalence PPR still wins AP** (0.031 vs 0.021 best learned). Ranking the
   3.66 M masked cells is a different problem from the balanced test, and no learned model
   here beats the random-walk heuristic on it — recorded, not the Phase-2 question.

## Does not show

- Anything about the graph-aware attention bias, LoRA, or the encoders E2–E7 — none are in
  this phase.
- That pretraining is useless for graphs: only that without a structural channel into
  attention, a frozen pretrained body behaves like a frozen random one.
- That the frozen arms were under-tuned: pretrained was swept over {1e-4 … 3e-2} and random
  over {1e-4 … 1e-2}; the selected 1e-2 / 3e-3 are interior.
- A significant advantage of the frozen body over the *small* scratch models (d128, d256).
- Any number beyond Cora: single graph, N = 2708, fixed-N encoder.

## Locked going forward

`configs/phase2.yaml` as committed alongside this file: masked-cell loss, `mask_frac` 0.15,
warm-up 100, ≤ 2000 epochs / patience 200, `lr_norm` 1e-4, encoder gain T/√d with
T = 0.987007, `selected_lr` = {pretrained 1e-2, none 3e-4, random 3e-3}, D1 default, no
checkpointing, the full three-round `extend:` list. Phase 3 starts from arm 1 at these
settings (checkpoints for every selected-LR run are kept under `$SCR/runs/phase2/`), adds the
per-head SPD bias (zero-init, own LR), and is judged on the paired delta against **this**
table: the bias has to recover at least the 0.043 the body costs, and pretrained has to
separate from random.

## One slide

**Phase 2 — frozen LLM in the loop, no graph-aware attention yet.** Cora masked edge
prediction, paired seeds: no body 0.920 · frozen Llama-3.2-1B 0.878 (−0.043, p 0.004) ·
same body random-init 0.887 (pretrained = random, p 0.1 over 10 seeds) · transformer from scratch
0.82–0.87 (frozen body ≥ scratch; significant from width 512) · shuffled graph 0.69–0.77
(structure is everything). Pair-MLP decoder: +0.013 with the body, −0.017 without.
Reproduced on the laptop from the cluster checkpoint to 5 decimals. Next: the attention
bias (Phase 3) has 0.043 to recover.

VALIDATED: ____________
