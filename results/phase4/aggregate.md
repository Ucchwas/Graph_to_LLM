27 Phase-4 rows @ 4638a4fe; Phase 3 rows @ a60dc60d (reproduction check only)

| model | enc lr | bias lr | LoRA lr | n | AUROC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift | best epoch | s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none (Phase 3) | 3e-04 | – | – | 20 | 0.918±0.006 | 0.932±0.006 | 0.917±0.007 | 0.0175±0.0044 | 121 | 315 | 0.05 |
| gnn gat d256 L2 | 3e-03 EDGE | – | – | 3 | 0.933±0.003 | 0.941±0.003 | 0.936±0.005 | 0.0132±0.0026 | 91 | 249 | 0.06 |
| gnn gcn d256 L2 | 3e-03 EDGE | – | – | 3 | 0.933±0.002 | 0.940±0.002 | 0.935±0.004 | 0.0160±0.0040 | 111 | 362 | 0.05 |
| gnn gin d256 L2 | 3e-03 EDGE | – | – | 3 | 0.924±0.007 | 0.930±0.007 | 0.929±0.007 | 0.0103±0.0034 | 72 | 432 | 0.05 |
| gnn sage d256 L2 | 3e-03 EDGE | – | – | 3 | 0.925±0.004 | 0.930±0.005 | 0.928±0.005 | 0.0085±0.0012 | 59 | 208 | 0.05 |
| none | 3e-04 | – | – | 3 | 0.919±0.003 | 0.929±0.005 | 0.921±0.002 | 0.0134±0.0044 | 93 | 296 | 0.07 | 

Same-seed reproduction of the Phase 3 rows at this commit:

| arm | seeds | previous AUROC | this phase AUROC | mean abs Δ | max abs Δ |
|---|---|---|---|---|---|
| none | 3 | 0.919±0.003 | 0.919±0.003 | 0.0000 | 0.0000 | 

inertness at init: 0 biased / LoRA rows compared with their plain counterpart (z_norm, logit_diag, logit_offdiag_std, logit_train_edge, layer_rms); 0 mismatch(es) above 1e-2 relative 

Paired per-seed deltas (all rows at this commit):

| Δ (A − B) | column | n | mean | SE | paired t | p | MDE |
|---|---|---|---|---|---|---|---|
| gnn gat d256 L2 − none | auc | 3 | +0.0149 | 0.0029 | +5.08 | 0.037 | 0.0157 |
| gnn gat d256 L2 − none | ap | 3 | +0.0116 | 0.0018 | +6.38 | 0.024 | 0.0098 |
| gnn gat d256 L2 − none | ap_sparse | 3 | -0.0003 | 0.0050 | -0.06 | 0.959 | 0.0266 |
| gnn gcn d256 L2 − none | auc | 3 | +0.0146 | 0.0026 | +5.63 | 0.030 | 0.0139 |
| gnn gcn d256 L2 − none | ap | 3 | +0.0106 | 0.0019 | +5.67 | 0.030 | 0.0100 |
| gnn gcn d256 L2 − none | ap_sparse | 3 | +0.0026 | 0.0059 | +0.44 | 0.706 | 0.0317 |
| gnn gin d256 L2 − none | auc | 3 | +0.0050 | 0.0051 | +0.98 | 0.432 | 0.0273 |
| gnn gin d256 L2 − none | ap | 3 | +0.0007 | 0.0047 | +0.15 | 0.896 | 0.0250 |
| gnn gin d256 L2 − none | ap_sparse | 3 | -0.0031 | 0.0017 | -1.78 | 0.218 | 0.0094 |
| gnn sage d256 L2 − none | auc | 3 | +0.0064 | 0.0028 | +2.30 | 0.149 | 0.0149 |
| gnn sage d256 L2 − none | ap | 3 | +0.0010 | 0.0011 | +0.90 | 0.461 | 0.0062 |
| gnn sage d256 L2 − none | ap_sparse | 3 | -0.0050 | 0.0022 | -2.22 | 0.156 | 0.0120 | 

LR surfaces (mean val AUROC):

gnn gat d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9321 *, 1e-03 0.9288

gnn gcn d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9325 *, 1e-03 0.9288

gnn gin d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9310 *, 1e-03 0.9283

gnn sage d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9313 *, 1e-03 0.9252

