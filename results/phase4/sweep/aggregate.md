219 Phase-4S rows @ 69976dcf; Phase 4 smoke rows @ 4638a4fe (reproduction check only)

| model | enc lr | bias lr | LoRA lr | n | AUROC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift | best epoch | s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gnn gat d256 L2 (Phase 4 smoke) | 3e-03 | – | – | 3 | 0.933±0.003 | 0.941±0.003 | 0.936±0.005 | 0.0132±0.0026 | 91 | 249 | 0.06 |
| gnn sage d256 L2 (Phase 4 smoke) | 3e-03 | – | – | 3 | 0.925±0.004 | 0.930±0.005 | 0.928±0.005 | 0.0085±0.0012 | 59 | 208 | 0.05 |
| none (Phase 4 smoke) | 3e-04 | – | – | 3 | 0.919±0.003 | 0.929±0.005 | 0.921±0.002 | 0.0134±0.0044 | 93 | 296 | 0.07 |
| gnn gat d1024 L2 | 1e-02 EDGE | – | – | 3 | 0.930±0.005 | 0.936±0.007 | 0.930±0.001 | 0.0127±0.0043 | 88 | 163 | 0.05 |
| gnn gat d1024 L4 | 1e-02 EDGE | – | – | 3 | 0.924±0.008 | 0.928±0.011 | 0.929±0.005 | 0.0093±0.0023 | 65 | 111 | 0.06 |
| gnn gat d1024 L8 | 3e-03 | – | – | 3 | 0.932±0.002 | 0.939±0.003 | 0.935±0.005 | 0.0157±0.0017 | 109 | 373 | 0.06 |
| gnn gat d256 L2 | 3e-03 | – | – | 3 | 0.931±0.004 | 0.938±0.005 | 0.934±0.004 | 0.0115±0.0035 | 80 | 218 | 0.04 |
| gnn gat d256 L4 | 1e-02 EDGE | – | – | 3 | 0.926±0.012 | 0.930±0.013 | 0.930±0.009 | 0.0089±0.0034 | 62 | 148 | 0.05 |
| gnn gat d256 L8 | 1e-02 EDGE | – | – | 3 | 0.932±0.004 | 0.935±0.009 | 0.937±0.002 | 0.0110±0.0043 | 76 | 277 | 0.06 |
| gnn gcn d1024 L2 | 1e-02 EDGE | – | – | 3 | 0.926±0.002 | 0.937±0.004 | 0.930±0.002 | 0.0144±0.0026 | 100 | 192 | 0.05 |
| gnn gcn d1024 L4 | 1e-03 EDGE | – | – | 3 | 0.932±0.005 | 0.935±0.010 | 0.935±0.005 | 0.0115±0.0033 | 80 | 240 | 0.05 |
| gnn gcn d1024 L8 | 1e-03 EDGE | – | – | 3 | 0.926±0.007 | 0.930±0.011 | 0.930±0.003 | 0.0082±0.0028 | 57 | 169 | 0.06 |
| gnn gcn d256 L2 | 1e-02 EDGE | – | – | 3 | 0.930±0.005 | 0.938±0.005 | 0.932±0.005 | 0.0137±0.0036 | 95 | 220 | 0.05 |
| gnn gcn d256 L4 | 1e-02 EDGE | – | – | 3 | 0.928±0.005 | 0.935±0.008 | 0.932±0.004 | 0.0109±0.0036 | 76 | 157 | 0.05 |
| gnn gcn d256 L8 | 3e-03 | – | – | 3 | 0.933±0.006 | 0.936±0.011 | 0.936±0.003 | 0.0104±0.0040 | 73 | 236 | 0.05 |
| gnn gin d1024 L2 | 1e-02 EDGE | – | – | 3 | 0.931±0.009 | 0.934±0.013 | 0.934±0.004 | 0.0107±0.0031 | 74 | 257 | 0.05 |
| gnn gin d1024 L4 | 1e-03 EDGE | – | – | 3 | 0.926±0.009 | 0.927±0.015 | 0.931±0.005 | 0.0109±0.0044 | 76 | 482 | 0.06 |
| gnn gin d1024 L8 | 3e-03 | – | – | 3 | 0.933±0.002 | 0.938±0.003 | 0.936±0.001 | 0.0124±0.0018 | 86 | 283 | 0.06 |
| gnn gin d256 L2 | 1e-02 EDGE | – | – | 3 | 0.928±0.009 | 0.933±0.010 | 0.933±0.008 | 0.0134±0.0052 | 93 | 348 | 0.05 |
| gnn gin d256 L4 | 1e-02 EDGE | – | – | 3 | 0.921±0.008 | 0.924±0.010 | 0.929±0.003 | 0.0078±0.0024 | 54 | 214 | 0.05 |
| gnn gin d256 L8 | 1e-02 EDGE | – | – | 3 | 0.927±0.007 | 0.933±0.007 | 0.932±0.002 | 0.0097±0.0011 | 68 | 278 | 0.05 |
| gnn sage d1024 L2 | 1e-02 EDGE | – | – | 3 | 0.927±0.004 | 0.931±0.006 | 0.931±0.003 | 0.0123±0.0037 | 85 | 194 | 0.05 |
| gnn sage d1024 L4 | 1e-03 EDGE | – | – | 3 | 0.920±0.007 | 0.921±0.008 | 0.926±0.005 | 0.0081±0.0007 | 56 | 274 | 0.05 |
| gnn sage d1024 L8 | 1e-02 EDGE | – | – | 3 | 0.924±0.004 | 0.930±0.003 | 0.929±0.005 | 0.0093±0.0006 | 65 | 326 | 0.06 |
| gnn sage d256 L2 | 3e-03 | – | – | 3 | 0.925±0.004 | 0.930±0.005 | 0.928±0.005 | 0.0085±0.0012 | 59 | 208 | 0.05 |
| gnn sage d256 L4 | 1e-02 EDGE | – | – | 3 | 0.925±0.003 | 0.930±0.003 | 0.930±0.005 | 0.0097±0.0037 | 68 | 247 | 0.05 |
| gnn sage d256 L8 | 1e-02 EDGE | – | – | 3 | 0.924±0.006 | 0.926±0.007 | 0.931±0.003 | 0.0096±0.0002 | 67 | 282 | 0.05 |
| none | 3e-04 | – | – | 3 | 0.919±0.003 | 0.929±0.005 | 0.921±0.002 | 0.0134±0.0044 | 93 | 296 | 0.05 | 

Same-seed reproduction of the Phase 4 smoke rows at this commit:

| arm | seeds | previous AUROC | this phase AUROC | mean abs Δ | max abs Δ |
|---|---|---|---|---|---|
| gnn gat d256 L2 | 3 | 0.933±0.003 | 0.931±0.004 | 0.0023 | 0.0036 |
| gnn sage d256 L2 | 3 | 0.925±0.004 | 0.925±0.004 | 0.0000 | 0.0000 |
| none | 3 | 0.919±0.003 | 0.919±0.003 | 0.0000 | 0.0000 | 

inertness at init: 0 biased / LoRA rows compared with their plain counterpart (z_norm, logit_diag, logit_offdiag_std, logit_train_edge, layer_rms); 0 mismatch(es) above 1e-2 relative 

Paired per-seed deltas (all rows at this commit):

| Δ (A − B) | column | n | mean | SE | paired t | p | MDE |
|---|---|---|---|---|---|---|---|
| gnn gat d1024 L2 − none | auc | 3 | +0.0113 | 0.0027 | +4.16 | 0.053 | 0.0146 |
| gnn gat d1024 L2 − none | ap | 3 | +0.0070 | 0.0012 | +5.69 | 0.029 | 0.0066 |
| gnn gat d1024 L2 − none | ap_sparse | 3 | -0.0008 | 0.0054 | -0.15 | 0.898 | 0.0291 |
| gnn gat d1024 L4 − none | auc | 3 | +0.0059 | 0.0051 | +1.16 | 0.366 | 0.0272 |
| gnn gat d1024 L4 − none | ap | 3 | -0.0008 | 0.0052 | -0.15 | 0.892 | 0.0279 |
| gnn gat d1024 L4 − none | ap_sparse | 3 | -0.0041 | 0.0031 | -1.31 | 0.320 | 0.0168 |
| gnn gat d1024 L8 − none | auc | 3 | +0.0138 | 0.0032 | +4.35 | 0.049 | 0.0170 |
| gnn gat d1024 L8 − none | ap | 3 | +0.0102 | 0.0028 | +3.62 | 0.068 | 0.0151 |
| gnn gat d1024 L8 − none | ap_sparse | 3 | +0.0022 | 0.0024 | +0.93 | 0.449 | 0.0129 |
| gnn gat d256 L2 − none | auc | 3 | +0.0128 | 0.0026 | +5.01 | 0.038 | 0.0137 |
| gnn gat d256 L2 − none | ap | 3 | +0.0086 | 0.0004 | +20.44 | 0.002 | 0.0023 |
| gnn gat d256 L2 − none | ap_sparse | 3 | -0.0020 | 0.0046 | -0.43 | 0.708 | 0.0247 |
| gnn gat d256 L4 − none | auc | 3 | +0.0078 | 0.0077 | +1.02 | 0.415 | 0.0413 |
| gnn gat d256 L4 − none | ap | 3 | +0.0010 | 0.0067 | +0.15 | 0.893 | 0.0359 |
| gnn gat d256 L4 − none | ap_sparse | 3 | -0.0046 | 0.0032 | -1.43 | 0.289 | 0.0172 |
| gnn gat d256 L8 − none | auc | 3 | +0.0139 | 0.0020 | +6.80 | 0.021 | 0.0109 |
| gnn gat d256 L8 − none | ap | 3 | +0.0060 | 0.0037 | +1.62 | 0.246 | 0.0200 |
| gnn gat d256 L8 − none | ap_sparse | 3 | -0.0024 | 0.0050 | -0.49 | 0.672 | 0.0266 |
| gnn gcn d1024 L2 − none | auc | 3 | +0.0078 | 0.0017 | +4.48 | 0.046 | 0.0093 |
| gnn gcn d1024 L2 − none | ap | 3 | +0.0078 | 0.0016 | +4.98 | 0.038 | 0.0084 |
| gnn gcn d1024 L2 − none | ap_sparse | 3 | +0.0009 | 0.0026 | +0.35 | 0.759 | 0.0142 |
| gnn gcn d1024 L4 − none | auc | 3 | +0.0137 | 0.0028 | +4.98 | 0.038 | 0.0148 |
| gnn gcn d1024 L4 − none | ap | 3 | +0.0057 | 0.0039 | +1.49 | 0.276 | 0.0207 |
| gnn gcn d1024 L4 − none | ap_sparse | 3 | -0.0020 | 0.0027 | -0.73 | 0.541 | 0.0144 |
| gnn gcn d1024 L8 − none | auc | 3 | +0.0075 | 0.0037 | +2.01 | 0.182 | 0.0200 |
| gnn gcn d1024 L8 − none | ap | 3 | +0.0014 | 0.0046 | +0.30 | 0.793 | 0.0247 |
| gnn gcn d1024 L8 − none | ap_sparse | 3 | -0.0052 | 0.0038 | -1.36 | 0.306 | 0.0206 |
| gnn gcn d256 L2 − none | auc | 3 | +0.0114 | 0.0036 | +3.15 | 0.088 | 0.0194 |
| gnn gcn d256 L2 − none | ap | 3 | +0.0086 | 0.0028 | +3.07 | 0.092 | 0.0150 |
| gnn gcn d256 L2 − none | ap_sparse | 3 | +0.0003 | 0.0028 | +0.10 | 0.931 | 0.0149 |
| gnn gcn d256 L4 − none | auc | 3 | +0.0097 | 0.0023 | +4.32 | 0.050 | 0.0121 |
| gnn gcn d256 L4 − none | ap | 3 | +0.0057 | 0.0018 | +3.12 | 0.089 | 0.0098 |
| gnn gcn d256 L4 − none | ap_sparse | 3 | -0.0025 | 0.0038 | -0.68 | 0.569 | 0.0202 |
| gnn gcn d256 L8 − none | auc | 3 | +0.0145 | 0.0040 | +3.63 | 0.068 | 0.0215 |
| gnn gcn d256 L8 − none | ap | 3 | +0.0073 | 0.0062 | +1.18 | 0.359 | 0.0332 |
| gnn gcn d256 L8 − none | ap_sparse | 3 | -0.0030 | 0.0009 | -3.45 | 0.075 | 0.0047 |
| gnn gin d1024 L2 − none | auc | 3 | +0.0128 | 0.0056 | +2.31 | 0.147 | 0.0298 |
| gnn gin d1024 L2 − none | ap | 3 | +0.0052 | 0.0076 | +0.68 | 0.564 | 0.0409 |
| gnn gin d1024 L2 − none | ap_sparse | 3 | -0.0027 | 0.0038 | -0.73 | 0.543 | 0.0202 |
| gnn gin d1024 L4 − none | auc | 3 | +0.0070 | 0.0064 | +1.08 | 0.391 | 0.0344 |
| gnn gin d1024 L4 − none | ap | 3 | -0.0021 | 0.0095 | -0.22 | 0.849 | 0.0510 |
| gnn gin d1024 L4 − none | ap_sparse | 3 | -0.0026 | 0.0042 | -0.61 | 0.602 | 0.0223 |
| gnn gin d1024 L8 − none | auc | 3 | +0.0142 | 0.0005 | +28.93 | 0.001 | 0.0026 |
| gnn gin d1024 L8 − none | ap | 3 | +0.0093 | 0.0013 | +7.36 | 0.018 | 0.0068 |
| gnn gin d1024 L8 − none | ap_sparse | 3 | -0.0010 | 0.0027 | -0.36 | 0.750 | 0.0147 |
| gnn gin d256 L2 − none | auc | 3 | +0.0097 | 0.0062 | +1.57 | 0.257 | 0.0331 |
| gnn gin d256 L2 − none | ap | 3 | +0.0044 | 0.0051 | +0.86 | 0.479 | 0.0274 |
| gnn gin d256 L2 − none | ap_sparse | 3 | -0.0000 | 0.0038 | -0.00 | 0.998 | 0.0202 |
| gnn gin d256 L4 − none | auc | 3 | +0.0026 | 0.0038 | +0.68 | 0.568 | 0.0206 |
| gnn gin d256 L4 − none | ap | 3 | -0.0054 | 0.0037 | -1.46 | 0.281 | 0.0199 |
| gnn gin d256 L4 − none | ap_sparse | 3 | -0.0056 | 0.0039 | -1.43 | 0.290 | 0.0212 |
| gnn gin d256 L8 − none | auc | 3 | +0.0088 | 0.0030 | +2.91 | 0.101 | 0.0162 |
| gnn gin d256 L8 − none | ap | 3 | +0.0039 | 0.0012 | +3.23 | 0.084 | 0.0064 |
| gnn gin d256 L8 − none | ap_sparse | 3 | -0.0037 | 0.0037 | -0.99 | 0.427 | 0.0200 |
| gnn sage d1024 L2 − none | auc | 3 | +0.0081 | 0.0022 | +3.69 | 0.066 | 0.0118 |
| gnn sage d1024 L2 − none | ap | 3 | +0.0018 | 0.0010 | +1.78 | 0.218 | 0.0055 |
| gnn sage d1024 L2 − none | ap_sparse | 3 | -0.0011 | 0.0045 | -0.25 | 0.824 | 0.0243 |
| gnn sage d1024 L4 − none | auc | 3 | +0.0018 | 0.0049 | +0.36 | 0.754 | 0.0264 |
| gnn sage d1024 L4 − none | ap | 3 | -0.0076 | 0.0034 | -2.25 | 0.153 | 0.0180 |
| gnn sage d1024 L4 − none | ap_sparse | 3 | -0.0054 | 0.0026 | -2.08 | 0.174 | 0.0138 |
| gnn sage d1024 L8 − none | auc | 3 | +0.0052 | 0.0049 | +1.07 | 0.396 | 0.0262 |
| gnn sage d1024 L8 − none | ap | 3 | +0.0006 | 0.0047 | +0.14 | 0.904 | 0.0253 |
| gnn sage d1024 L8 − none | ap_sparse | 3 | -0.0041 | 0.0028 | -1.46 | 0.282 | 0.0151 |
| gnn sage d256 L2 − none | auc | 3 | +0.0064 | 0.0028 | +2.30 | 0.149 | 0.0149 |
| gnn sage d256 L2 − none | ap | 3 | +0.0010 | 0.0011 | +0.90 | 0.461 | 0.0062 |
| gnn sage d256 L2 − none | ap_sparse | 3 | -0.0050 | 0.0022 | -2.22 | 0.156 | 0.0120 |
| gnn sage d256 L4 − none | auc | 3 | +0.0065 | 0.0023 | +2.89 | 0.102 | 0.0121 |
| gnn sage d256 L4 − none | ap | 3 | +0.0011 | 0.0013 | +0.87 | 0.477 | 0.0070 |
| gnn sage d256 L4 − none | ap_sparse | 3 | -0.0037 | 0.0019 | -1.94 | 0.192 | 0.0102 |
| gnn sage d256 L8 − none | auc | 3 | +0.0059 | 0.0026 | +2.22 | 0.157 | 0.0142 |
| gnn sage d256 L8 − none | ap | 3 | -0.0028 | 0.0012 | -2.33 | 0.145 | 0.0065 |
| gnn sage d256 L8 − none | ap_sparse | 3 | -0.0038 | 0.0031 | -1.24 | 0.339 | 0.0164 | 

LR surfaces (mean val AUROC):

gnn gat d1024 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9342 *, 3e-03 0.9338, 1e-03 0.9338

gnn gat d1024 L4 — mean val AUROC by encoder LR (* selected): 1e-02 0.9357 *, 3e-03 0.9348, 1e-03 0.9324

gnn gat d1024 L8 — mean val AUROC by encoder LR (* selected): 1e-02 0.9347, 3e-03 0.9354 *, 1e-03 0.9346

gnn gat d256 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9329, 3e-03 0.9340 *, 1e-03 0.9275

gnn gat d256 L4 — mean val AUROC by encoder LR (* selected): 1e-02 0.9387 *, 3e-03 0.9351, 1e-03 0.9318

gnn gat d256 L8 — mean val AUROC by encoder LR (* selected): 1e-02 0.9373 *, 3e-03 0.9365, 1e-03 0.9323

gnn gcn d1024 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9366 *, 3e-03 0.9308, 1e-03 0.9337

gnn gcn d1024 L4 — mean val AUROC by encoder LR (* selected): 1e-02 0.9328, 3e-03 0.9328, 1e-03 0.9331 *

gnn gcn d1024 L8 — mean val AUROC by encoder LR (* selected): 1e-02 0.9307, 3e-03 0.9317, 1e-03 0.9325 *

gnn gcn d256 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9346 *, 3e-03 0.9325, 1e-03 0.9288

gnn gcn d256 L4 — mean val AUROC by encoder LR (* selected): 1e-02 0.9346 *, 3e-03 0.9332, 1e-03 0.9319

gnn gcn d256 L8 — mean val AUROC by encoder LR (* selected): 1e-02 0.9315, 3e-03 0.9328 *, 1e-03 0.9302

gnn gin d1024 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9340 *, 3e-03 0.9333, 1e-03 0.9330

gnn gin d1024 L4 — mean val AUROC by encoder LR (* selected): 1e-02 0.9326, 3e-03 0.9319, 1e-03 0.9345 *

gnn gin d1024 L8 — mean val AUROC by encoder LR (* selected): 1e-02 0.9321, 3e-03 0.9371 *, 1e-03 0.9316

gnn gin d256 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9360 *, 3e-03 0.9318, 1e-03 0.9285

gnn gin d256 L4 — mean val AUROC by encoder LR (* selected): 1e-02 0.9349 *, 3e-03 0.9340, 1e-03 0.9297

gnn gin d256 L8 — mean val AUROC by encoder LR (* selected): 1e-02 0.9356 *, 3e-03 0.9348, 1e-03 0.9309

gnn sage d1024 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9361 *, 3e-03 0.9306, 1e-03 0.9283

gnn sage d1024 L4 — mean val AUROC by encoder LR (* selected): 1e-02 0.9288, 3e-03 0.9283, 1e-03 0.9322 *

gnn sage d1024 L8 — mean val AUROC by encoder LR (* selected): 1e-02 0.9337 *, 3e-03 0.9319, 1e-03 0.9321

gnn sage d256 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9297, 3e-03 0.9313 *, 1e-03 0.9252

gnn sage d256 L4 — mean val AUROC by encoder LR (* selected): 1e-02 0.9311 *, 3e-03 0.9279, 1e-03 0.9181

gnn sage d256 L8 — mean val AUROC by encoder LR (* selected): 1e-02 0.9296 *, 3e-03 0.9244, 1e-03 0.9065

