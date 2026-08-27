note: results\phase4\direct\rows rows span 2 commits with identical model code (6b4c6d05, af03a478)
51 Phase-4D rows @ af03a478; Phase 4 smoke rows @ 4638a4fe (reproduction check only)

| model | enc lr | bias lr | LoRA lr | n | AUROC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift | best epoch | s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gnn gat d256 L2 (Phase 4 smoke) | 3e-03 | – | – | 3 | 0.933±0.003 | 0.941±0.003 | 0.936±0.005 | 0.0132±0.0026 | 91 | 249 | 0.06 |
| gnn gcn d256 L2 (Phase 4 smoke) | 3e-03 | – | – | 3 | 0.933±0.002 | 0.940±0.002 | 0.935±0.004 | 0.0160±0.0040 | 111 | 362 | 0.05 |
| gnn gin d256 L2 (Phase 4 smoke) | 3e-03 | – | – | 3 | 0.924±0.007 | 0.930±0.007 | 0.929±0.007 | 0.0103±0.0034 | 72 | 432 | 0.05 |
| gnn sage d256 L2 (Phase 4 smoke) | 3e-03 | – | – | 3 | 0.925±0.004 | 0.930±0.005 | 0.928±0.005 | 0.0085±0.0012 | 59 | 208 | 0.05 |
| none (Phase 4 smoke) | 3e-04 | – | – | 3 | 0.919±0.003 | 0.929±0.005 | 0.921±0.002 | 0.0134±0.0044 | 93 | 296 | 0.07 |
| gnn gat d256 L2 | 3e-03 EDGE | – | – | 3 | 0.931±0.004 | 0.938±0.003 | 0.933±0.007 | 0.0099±0.0015 | 68 | 149 | 0.06 |
| gnn gcn d256 L2 | 3e-03 EDGE | – | – | 3 | 0.933±0.002 | 0.940±0.002 | 0.935±0.004 | 0.0160±0.0040 | 111 | 362 | 0.06 |
| gnn gin d256 L2 | 3e-03 EDGE | – | – | 3 | 0.921±0.006 | 0.926±0.007 | 0.929±0.006 | 0.0109±0.0046 | 76 | 504 | 0.06 |
| gnn sage d256 L2 | 3e-03 EDGE | – | – | 3 | 0.925±0.004 | 0.930±0.005 | 0.928±0.005 | 0.0085±0.0012 | 59 | 208 | 0.06 |
| gnn-direct gat d256 L2 | 3e-03 EDGE | – | – | 3 | 0.928±0.006 | 0.935±0.005 | 0.932±0.004 | 0.0116±0.0028 | 81 | 264 | 0.07 |
| gnn-direct gcn d256 L2 | 3e-03 EDGE | – | – | 3 | 0.934±0.006 | 0.941±0.006 | 0.936±0.002 | 0.0133±0.0028 | 93 | 284 | 0.05 |
| gnn-direct gin d256 L2 | 3e-03 EDGE | – | – | 3 | 0.929±0.008 | 0.932±0.011 | 0.932±0.004 | 0.0088±0.0006 | 61 | 349 | 0.06 |
| gnn-direct sage d256 L2 | 3e-03 EDGE | – | – | 3 | 0.926±0.002 | 0.933±0.003 | 0.928±0.003 | 0.0106±0.0028 | 74 | 298 | 0.07 |
| none | 3e-04 | – | – | 3 | 0.919±0.003 | 0.929±0.005 | 0.921±0.002 | 0.0134±0.0044 | 93 | 296 | 0.07 | 

Same-seed reproduction of the Phase 4 smoke rows at this commit:

| arm | seeds | previous AUROC | this phase AUROC | mean abs Δ | max abs Δ |
|---|---|---|---|---|---|
| gnn gat d256 L2 | 3 | 0.933±0.003 | 0.931±0.004 | 0.0025 | 0.0038 |
| gnn gcn d256 L2 | 3 | 0.933±0.002 | 0.933±0.002 | 0.0000 | 0.0000 |
| gnn gin d256 L2 | 3 | 0.924±0.007 | 0.921±0.006 | 0.0021 | 0.0059 |
| gnn sage d256 L2 | 3 | 0.925±0.004 | 0.925±0.004 | 0.0000 | 0.0000 |
| none | 3 | 0.919±0.003 | 0.919±0.003 | 0.0000 | 0.0000 | 

inertness at init: 0 biased / LoRA rows compared with their plain counterpart (z_norm, logit_diag, logit_offdiag_std, logit_train_edge, layer_rms); 0 mismatch(es) above 1e-2 relative 

Paired per-seed deltas (all rows at this commit):

| Δ (A − B) | column | n | mean | SE | paired t | p | MDE |
|---|---|---|---|---|---|---|---|
| gnn gat d256 L2 − none | auc | 3 | +0.0124 | 0.0041 | +3.03 | 0.094 | 0.0220 |
| gnn gat d256 L2 − none | ap | 3 | +0.0086 | 0.0022 | +3.91 | 0.060 | 0.0117 |
| gnn gat d256 L2 − none | ap_sparse | 3 | -0.0036 | 0.0035 | -1.02 | 0.414 | 0.0188 |
| gnn gcn d256 L2 − none | auc | 3 | +0.0146 | 0.0026 | +5.63 | 0.030 | 0.0139 |
| gnn gcn d256 L2 − none | ap | 3 | +0.0106 | 0.0019 | +5.67 | 0.030 | 0.0100 |
| gnn gcn d256 L2 − none | ap_sparse | 3 | +0.0026 | 0.0059 | +0.44 | 0.706 | 0.0317 |
| gnn gin d256 L2 − none | auc | 3 | +0.0029 | 0.0037 | +0.78 | 0.517 | 0.0200 |
| gnn gin d256 L2 − none | ap | 3 | -0.0027 | 0.0022 | -1.22 | 0.346 | 0.0117 |
| gnn gin d256 L2 − none | ap_sparse | 3 | -0.0026 | 0.0012 | -2.18 | 0.161 | 0.0063 |
| gnn sage d256 L2 − none | auc | 3 | +0.0064 | 0.0028 | +2.30 | 0.149 | 0.0149 |
| gnn sage d256 L2 − none | ap | 3 | +0.0010 | 0.0011 | +0.90 | 0.461 | 0.0062 |
| gnn sage d256 L2 − none | ap_sparse | 3 | -0.0050 | 0.0022 | -2.22 | 0.156 | 0.0120 |
| gnn-direct gat d256 L2 − none | auc | 3 | +0.0094 | 0.0029 | +3.25 | 0.083 | 0.0155 |
| gnn-direct gat d256 L2 − none | ap | 3 | +0.0057 | 0.0010 | +5.60 | 0.030 | 0.0054 |
| gnn-direct gat d256 L2 − none | ap_sparse | 3 | -0.0018 | 0.0050 | -0.37 | 0.747 | 0.0266 |
| gnn-direct gcn d256 L2 − none | auc | 3 | +0.0152 | 0.0023 | +6.77 | 0.021 | 0.0121 |
| gnn-direct gcn d256 L2 − none | ap | 3 | +0.0123 | 0.0017 | +7.36 | 0.018 | 0.0089 |
| gnn-direct gcn d256 L2 − none | ap_sparse | 3 | -0.0001 | 0.0050 | -0.02 | 0.985 | 0.0267 |
| gnn-direct gin d256 L2 − none | auc | 3 | +0.0102 | 0.0058 | +1.77 | 0.219 | 0.0310 |
| gnn-direct gin d256 L2 − none | ap | 3 | +0.0033 | 0.0062 | +0.53 | 0.651 | 0.0334 |
| gnn-direct gin d256 L2 − none | ap_sparse | 3 | -0.0047 | 0.0031 | -1.52 | 0.268 | 0.0165 |
| gnn-direct sage d256 L2 − none | auc | 3 | +0.0076 | 0.0032 | +2.39 | 0.139 | 0.0171 |
| gnn-direct sage d256 L2 − none | ap | 3 | +0.0043 | 0.0039 | +1.11 | 0.384 | 0.0211 |
| gnn-direct sage d256 L2 − none | ap_sparse | 3 | -0.0028 | 0.0034 | -0.84 | 0.489 | 0.0182 |
| gnn-direct gat d256 L2 − gnn gat d256 L2 | auc | 3 | -0.0030 | 0.0038 | -0.79 | 0.511 | 0.0205 |
| gnn-direct gat d256 L2 − gnn gat d256 L2 | ap | 3 | -0.0029 | 0.0029 | -1.01 | 0.419 | 0.0153 |
| gnn-direct gat d256 L2 − gnn gat d256 L2 | ap_sparse | 3 | +0.0018 | 0.0016 | +1.12 | 0.378 | 0.0084 |
| gnn-direct gcn d256 L2 − gnn gcn d256 L2 | auc | 3 | +0.0006 | 0.0042 | +0.15 | 0.894 | 0.0226 |
| gnn-direct gcn d256 L2 − gnn gcn d256 L2 | ap | 3 | +0.0017 | 0.0026 | +0.65 | 0.581 | 0.0139 |
| gnn-direct gcn d256 L2 − gnn gcn d256 L2 | ap_sparse | 3 | -0.0027 | 0.0011 | -2.37 | 0.141 | 0.0061 |
| gnn-direct gin d256 L2 − gnn gin d256 L2 | auc | 3 | +0.0073 | 0.0022 | +3.32 | 0.080 | 0.0118 |
| gnn-direct gin d256 L2 − gnn gin d256 L2 | ap | 3 | +0.0060 | 0.0041 | +1.46 | 0.282 | 0.0219 |
| gnn-direct gin d256 L2 − gnn gin d256 L2 | ap_sparse | 3 | -0.0021 | 0.0031 | -0.69 | 0.561 | 0.0165 |
| gnn-direct sage d256 L2 − gnn sage d256 L2 | auc | 3 | +0.0013 | 0.0030 | +0.43 | 0.710 | 0.0159 |
| gnn-direct sage d256 L2 − gnn sage d256 L2 | ap | 3 | +0.0033 | 0.0033 | +0.99 | 0.426 | 0.0179 |
| gnn-direct sage d256 L2 − gnn sage d256 L2 | ap_sparse | 3 | +0.0021 | 0.0020 | +1.08 | 0.392 | 0.0106 | 

LR surfaces (mean val AUROC):

gnn gat d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9318 *, 1e-03 0.9278

gnn gcn d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9325 *, 1e-03 0.9288

gnn gin d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9324 *, 1e-03 0.9285

gnn sage d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9313 *, 1e-03 0.9252

gnn-direct gat d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9291 *, 1e-03 0.9267

gnn-direct gcn d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9357 *, 1e-03 0.9313

gnn-direct gin d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9268 *, 1e-03 0.9244

gnn-direct sage d256 L2 — mean val AUROC by encoder LR (* selected): 3e-03 0.9263 *, 1e-03 0.9179

