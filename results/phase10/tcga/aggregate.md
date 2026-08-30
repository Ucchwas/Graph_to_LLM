| body | changed-edge AUROC | changed-edge AP | overall AUROC | direction | #Params | seeds |
|---|---|---|---|---|---|---|
| LGM | 0.7034 ± 0.0208 | 0.2791 ± 0.0287 | 0.7463 ± 0.0191 | 0.4279 ± 0.0653 | 812,176 | 5 |
| **LGM + RRWP** | 0.7314 ± 0.0185 | 0.3086 ± 0.0321 | 0.7854 ± 0.0156 | 0.3888 ± 0.0525 | 812,432 | 5 |
| edge-aware GCN | 0.7284 ± 0.0153 | 0.3001 ± 0.0250 | 0.7771 ± 0.0225 | 0.3468 ± 0.0517 | 810,810 | 5 |
| edge-aware GCN + RWSE | 0.7261 ± 0.0171 | 0.2997 ± 0.0269 | 0.7728 ± 0.0211 | 0.3552 ± 0.0492 | 815,562 | 5 |

Paired over the held-out cancers (seeds averaged within a cancer first):

| contrast | metric | Δ ± SE | t | p | wins |
|---|---|---|---|---|---|
| lgm_rrwp − lgm | changed-edge AUROC | +0.0280 ± 0.0153 | +1.83 | 9.74e-02 | 7/11 |
| lgm_rrwp − lgm | changed-edge AP | +0.0296 ± 0.0126 | +2.35 | 4.10e-02 | 9/11 |
| lgm_rrwp − lgm | overall AUROC | +0.0391 ± 0.0178 | +2.19 | 5.33e-02 | 8/11 |
| lgm_rrwp − lgm | direction | -0.0391 ± 0.0324 | -1.21 | 2.55e-01 | 4/11 |
| edgegcn_rwse − edgegcn | changed-edge AUROC | -0.0022 ± 0.0055 | -0.41 | 6.92e-01 | 6/11 |
| edgegcn_rwse − edgegcn | changed-edge AP | -0.0005 ± 0.0110 | -0.04 | 9.65e-01 | 6/11 |
| edgegcn_rwse − edgegcn | overall AUROC | -0.0043 ± 0.0059 | -0.74 | 4.78e-01 | 6/11 |
| edgegcn_rwse − edgegcn | direction | +0.0084 ± 0.0264 | +0.32 | 7.58e-01 | 5/11 |
| lgm_rrwp − edgegcn_rwse | changed-edge AUROC | +0.0053 ± 0.0159 | +0.33 | 7.47e-01 | 5/11 |
| lgm_rrwp − edgegcn_rwse | changed-edge AP | +0.0090 ± 0.0119 | +0.75 | 4.68e-01 | 6/11 |
| lgm_rrwp − edgegcn_rwse | overall AUROC | +0.0126 ± 0.0184 | +0.69 | 5.08e-01 | 6/11 |
| lgm_rrwp − edgegcn_rwse | direction | +0.0336 ± 0.0380 | +0.88 | 3.97e-01 | 7/11 |
| lgm − edgegcn | changed-edge AUROC | -0.0249 ± 0.0122 | -2.05 | 6.75e-02 | 3/11 |
| lgm − edgegcn | changed-edge AP | -0.0211 ± 0.0163 | -1.29 | 2.26e-01 | 5/11 |
| lgm − edgegcn | overall AUROC | -0.0308 ± 0.0102 | -3.02 | 1.29e-02 | 2/11 |
| lgm − edgegcn | direction | +0.0811 ± 0.0518 | +1.57 | 1.49e-01 | 8/11 |

Pairing every (cancer, seed) for lgm_rrwp − edgegcn_rwse, n = 55: changed-edge AUROC Δ +0.0053, t = +0.51, p = 6.09e-01, wins 28/55.

Per cancer (changed-edge AUROC, mean over seeds):

| held-out cancer | LGM | **LGM + RRWP** | edge-aware GCN | edge-aware GCN + RWSE |
|---|---|---|---|---|
| Breast Invasive Carcinoma | 0.8118 | 0.7916 | 0.8138 | 0.8253 |
| Head & Neck Squamous Cell Carcinoma | 0.7741 | 0.7670 | 0.8164 | 0.8186 |
| Lung Adenocarcinoma | 0.7673 | 0.8067 | 0.7587 | 0.7757 |
| Lung Squamous Cell Carcinoma | 0.7258 | 0.7785 | 0.7539 | 0.7430 |
| Prostate Adenocarcinoma | 0.7252 | 0.7530 | 0.7196 | 0.6987 |
| Kidney Clear Cell Carcinoma | 0.7180 | 0.6953 | 0.6888 | 0.7154 |
| Colon Adenocarcinoma | 0.6825 | 0.6370 | 0.6907 | 0.6967 |
| Liver Hepatocellular Carcinoma | 0.6599 | 0.6694 | 0.6875 | 0.6910 |
| Stomach Adenocarcinoma | 0.6530 | 0.7484 | 0.7168 | 0.6802 |
| Kidney Papillary Cell Carcinoma | 0.6501 | 0.7628 | 0.6778 | 0.6700 |
| Thyroid Carcinoma | 0.5699 | 0.6357 | 0.6879 | 0.6727 |
