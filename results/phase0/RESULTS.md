# Phase 0 — Protocol foundation

**Claim:** the data, masking, and evaluation protocol are correct, leak-free, and
externally anchored to the corrected Cora facts, before any model exists.

## Gate checklist
- [x] 18/18 tests green (`python -m pytest tests/`)
- [x] identity baseline scores exactly at chance on held-out cells (AUROC 0.500, lift 1.000)
- [x] SPD provably a function of the observed graph only (flip-held-out-cells test)
- [x] multi-graph [B,N,N] padding API works (Phase-4 molecular hedge)
- [x] Cora corrected facts verified live: 2708 nodes, 5278 undirected edges
      (edge_index [2,10556] — NOT the spec's 5,429), 1433 features,
      split 4488/263/527(+527 neg), pos_weight ≈ 693.4 computed at runtime
- [x] supervision edges disjoint from message-passing edges in the split
- [x] GPU sanity: RTX 5070 Ti Laptop, sm_120 initializes, bf16 matmul OK
- [x] stack pinned in requirements-local.txt (transformers==5.15.1, torch 2.13.0+cu130)
- [ ] user ran `python -m g2l.walkthrough --phase 0` and signed below

## The three metric columns (fixed for the project)
| column | meaning | comparable to |
|---|---|---|
| AUROC | balance-invariant ranking quality | everything |
| AP@1:1 | all positives + equal seeded negatives (≥5 draws) | published GAE 91.0/92.0, MaskGAE 96.45/95.95 |
| AP@true-prevalence (+lift) | honest sparsity number on all masked cells | nothing published — we generate the baseline |

## Shows / does not show
Shows: the measuring stick is trustworthy; a model that copies its input cannot
look good; the numbers we produce later will be comparable to the literature.
Does not show: anything about any model. No model exists yet.

## Locked going forward
`mask_matrix` masks the strict upper triangle and mirrors into the input;
diagonal excluded; `evaluate()` is the only scoring path; structural features
take A_observed only (tests/test_leaks.py enforces).

VALIDATED: ____________
