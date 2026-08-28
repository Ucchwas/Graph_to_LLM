# Phase 6 -- Decagon aligned multi-layer completion (5875 rows @ f33eaa49)

shared model: val-layer AUROC 0.9783 at epoch 66 of 97, 3.9 s/epoch, 2,763,776 params, peak 0.445553664 GB; split {'n_layers': 963, 'n_union_pairs': 63472, 'n_hidden_pairs': 6308, 'groups': {'train': 770, 'val': 96, 'test': 97}, 'max_cross_jaccard': 0.19841346144676208, 'n_components': 715}

## E2 held-out layers (completion from the layer's own partial input; shared never trained on them)

| arm | n layers | AUROC | AP@1:1 | AUROC hidden pairs | AP hidden pairs | AUROC sparse | AP sparse |
|---|---|---|---|---|---|---|---|
| shared | 97 | 0.9770 +- 0.0018 | 0.9727 +- 0.0023 | 0.9321 +- 0.0050 | 0.9273 +- 0.0055 | 0.9769 +- 0.0017 | 0.1468 +- 0.0081 |
| independent | 97 | 0.9711 +- 0.0020 | 0.9684 +- 0.0023 | 0.9226 +- 0.0046 | 0.9166 +- 0.0051 | 0.9709 +- 0.0018 | 0.1105 +- 0.0064 |
| gae0 | 97 | 0.9577 +- 0.0022 | 0.9526 +- 0.0028 | 0.8948 +- 0.0058 | 0.8821 +- 0.0064 | 0.9574 +- 0.0022 | 0.0501 +- 0.0024 |
| knn | 97 | 0.9500 +- 0.0018 | 0.9318 +- 0.0028 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.9498 +- 0.0012 | 0.0248 +- 0.0014 |
| pair_frequency | 97 | 0.9475 +- 0.0018 | 0.9278 +- 0.0028 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.9473 +- 0.0012 | 0.0230 +- 0.0013 |
| common_neighbors | 97 | 0.9520 +- 0.0022 | 0.9470 +- 0.0025 | 0.9187 +- 0.0045 | 0.9102 +- 0.0050 | 0.9516 +- 0.0022 | 0.2001 +- 0.0119 |
| identity | 97 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.0013 +- 0.0001 |

paired shared - independent over 97 layers:

- AUROC: delta = +0.0059 +- 0.0012 (t = 4.79, p = 6.2e-06, n = 97; shared better on 68 / 97)
- AP@1:1: delta = +0.0043 +- 0.0011 (t = 3.80, p = 0.00026, n = 97; shared better on 62 / 97)
- AUROC hidden pairs: delta = +0.0095 +- 0.0017 (t = 5.71, p = 1.3e-07, n = 97; shared better on 71 / 97)
- AP hidden pairs: delta = +0.0107 +- 0.0023 (t = 4.69, p = 9.1e-06, n = 97; shared better on 66 / 97)

## val layers (selection set of the shared model)

| arm | n layers | AUROC | AP@1:1 | AUROC hidden pairs | AP hidden pairs | AUROC sparse | AP sparse |
|---|---|---|---|---|---|---|---|
| shared | 96 | 0.9757 +- 0.0015 | 0.9714 +- 0.0020 | 0.9318 +- 0.0051 | 0.9271 +- 0.0057 | 0.9758 +- 0.0015 | 0.1514 +- 0.0095 |
| independent | 96 | 0.9678 +- 0.0019 | 0.9659 +- 0.0020 | 0.9246 +- 0.0046 | 0.9206 +- 0.0052 | 0.9675 +- 0.0018 | 0.1121 +- 0.0062 |
| knn | 96 | 0.9472 +- 0.0016 | 0.9290 +- 0.0026 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.9478 +- 0.0012 | 0.0258 +- 0.0016 |
| pair_frequency | 96 | 0.9445 +- 0.0017 | 0.9247 +- 0.0027 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.9450 +- 0.0012 | 0.0238 +- 0.0015 |
| common_neighbors | 96 | 0.9500 +- 0.0023 | 0.9452 +- 0.0025 | 0.9196 +- 0.0044 | 0.9112 +- 0.0048 | 0.9500 +- 0.0023 | 0.1968 +- 0.0113 |
| identity | 96 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.0013 +- 0.0001 |

paired shared - independent over 96 layers:

- AUROC: delta = +0.0079 +- 0.0016 (t = 4.99, p = 2.8e-06, n = 96; shared better on 76 / 96)
- AP@1:1: delta = +0.0055 +- 0.0013 (t = 4.24, p = 5.2e-05, n = 96; shared better on 64 / 96)
- AUROC hidden pairs: delta = +0.0072 +- 0.0013 (t = 5.72, p = 1.3e-07, n = 96; shared better on 69 / 96)
- AP hidden pairs: delta = +0.0064 +- 0.0018 (t = 3.66, p = 0.00041, n = 96; shared better on 62 / 96)

## E1 train layers (in-distribution)

| arm | n layers | AUROC | AP@1:1 | AUROC hidden pairs | AP hidden pairs | AUROC sparse | AP sparse |
|---|---|---|---|---|---|---|---|
| shared | 770 | 0.9690 +- 0.0008 | 0.9633 +- 0.0010 | 0.9033 +- 0.0029 | 0.8981 +- 0.0030 | 0.9692 +- 0.0008 | 0.1583 +- 0.0032 |
| independent | 770 | 0.9633 +- 0.0008 | 0.9585 +- 0.0010 | 0.8962 +- 0.0026 | 0.8897 +- 0.0027 | 0.9634 +- 0.0008 | 0.1137 +- 0.0022 |
| knn | 770 | 0.9470 +- 0.0006 | 0.9279 +- 0.0010 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.9475 +- 0.0005 | 0.0400 +- 0.0013 |
| pair_frequency | 770 | 0.9442 +- 0.0006 | 0.9232 +- 0.0010 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.9448 +- 0.0005 | 0.0362 +- 0.0011 |
| common_neighbors | 770 | 0.9448 +- 0.0010 | 0.9380 +- 0.0011 | 0.8909 +- 0.0026 | 0.8812 +- 0.0027 | 0.9447 +- 0.0010 | 0.1817 +- 0.0039 |
| identity | 770 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.0024 +- 0.0001 |

paired shared - independent over 770 layers:

- AUROC: delta = +0.0057 +- 0.0004 (t = 15.88, p = 2.6e-49, n = 770; shared better on 587 / 770)
- AP@1:1: delta = +0.0048 +- 0.0004 (t = 13.30, p = 1.8e-36, n = 770; shared better on 517 / 770)
- AUROC hidden pairs: delta = +0.0070 +- 0.0006 (t = 12.39, p = 2.9e-32, n = 770; shared better on 506 / 770)
- AP hidden pairs: delta = +0.0084 +- 0.0007 (t = 11.75, p = 2e-29, n = 770; shared better on 495 / 770)

## E4 shared model on the test layers from reduced input (fraction of the layer's train + val pairs kept)

| input | AUROC | AP@1:1 |
|---|---|---|
| 1.0 (all) | 0.9770 +- 0.0018 | 0.9727 +- 0.0023 |
| 0.5 | 0.9662 +- 0.0020 | 0.9626 +- 0.0025 |
| 0.25 | 0.9416 +- 0.0024 | 0.9377 +- 0.0028 |

## Where the gain lives (test layers, n = 97)

corr(log layer train size, shared - independent AUROC) = **-0.350**

| quartile by layer size | train pairs | shared | independent | delta |
|---|---|---|---|---|
| 1 | 391-871 | 0.9912 | 0.9782 | +0.0129 |
| 2 | 893-1580 | 0.9864 | 0.9807 | +0.0057 |
| 3 | 1617-2902 | 0.9746 | 0.9707 | +0.0039 |
| 4 | 3261-6962 | 0.9551 | 0.9543 | +0.0008 |

delta spread: min -0.0077, median +0.0029, max +0.0818; **shared is worse on 29 / 97 layers**

- shared - common neighbours on test pairs: +0.0250 +- 0.0017 (shared better on 96/97)
- shared - common neighbours on hidden pairs: +0.0134 +- 0.0019 (shared better on 78/97)

