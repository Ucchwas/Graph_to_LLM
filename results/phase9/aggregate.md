| body | changed-edge AUROC | changed-edge AP | overall AUROC | direction | #Params | seeds |
|---|---|---|---|---|---|---|
| **LGM** | 0.7196 ± 0.0213 | 0.2998 ± 0.0273 | 0.7543 ± 0.0253 | 0.4185 ± 0.0728 | 812,176 | 3 |
| edge-aware GCN | 0.7135 ± 0.0166 | 0.2837 ± 0.0275 | 0.7735 ± 0.0237 | 0.2917 ± 0.0576 | 810,810 | 3 |

Paired LGM − edge-GCN over the held-out cancers (seeds averaged within a cancer first):

| metric | Δ mean ± SE | t | p | wins |
|---|---|---|---|---|
| changed-edge AUROC | +0.0062 ± 0.0165 | +0.37 | 7.16e-01 | 6/11 |
| changed-edge AP | +0.0161 ± 0.0202 | +0.80 | 4.43e-01 | 6/11 |
| overall AUROC | -0.0193 ± 0.0180 | -1.07 | 3.09e-01 | 4/11 |
| direction | +0.1268 ± 0.0604 | +2.10 | 6.20e-02 | 8/11 |

Pairing every (cancer, seed) instead, n = 33: changed-edge AUROC Δ +0.0062, t = +0.55, p = 5.85e-01, wins 18/33.

Per cancer (changed-edge AUROC, mean over seeds):

| held-out cancer | LGM | edge-GCN | Δ |
|---|---|---|---|
| Breast Invasive Carcinoma | 0.8176 | 0.8275 | -0.0100 |
| Lung Adenocarcinoma | 0.8095 | 0.7632 | +0.0462 |
| Head & Neck Squamous Cell Carcinoma | 0.7642 | 0.7583 | +0.0060 |
| Lung Squamous Cell Carcinoma | 0.7384 | 0.7488 | -0.0104 |
| Kidney Clear Cell Carcinoma | 0.7368 | 0.6504 | +0.0864 |
| Liver Hepatocellular Carcinoma | 0.7322 | 0.6888 | +0.0433 |
| Prostate Adenocarcinoma | 0.7231 | 0.6929 | +0.0302 |
| Kidney Papillary Cell Carcinoma | 0.6890 | 0.6423 | +0.0467 |
| Colon Adenocarcinoma | 0.6709 | 0.6921 | -0.0212 |
| Stomach Adenocarcinoma | 0.6682 | 0.6995 | -0.0314 |
| Thyroid Carcinoma | 0.5662 | 0.6844 | -0.1182 |
