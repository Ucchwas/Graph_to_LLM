# Phase 7 -- normal graph -> tumour graph translation (264 rows @ 6d804123, 9e558168, f75392da)

11 held-out cancers, [2000] genes, leave-one-cancer-out. Identity is exactly 0.5 on the changed-edge strata and 0.0 on direction by construction.

Gate 7.0: **PASS** (3/3 probe cancers, 20 bootstraps)

| cancer | bootstrap Jaccard normal (p5) | tumour (p5) | normal vs tumour | verdict |
|---|---|---|---|---|
| Breast Invasive Carcinoma | 0.4659 (0.3973) | 0.5461 (0.4857) | 0.0907 | pass |
| Kidney Clear Cell Carcinoma | 0.4134 (0.3422) | 0.4057 (0.3470) | 0.1302 | pass |
| Thyroid Carcinoma | 0.5108 (0.4338) | 0.4073 (0.3369) | 0.3234 | pass |

## density 0.005 (sensitivity)

| arm | n | changed-edge AUROC | gained | lost | direction | overall AUROC | overall AP@1:1 |
|---|---|---|---|---|---|---|---|
| shared_stratified | 11 | 0.7917 +- 0.0279 | 0.7974 +- 0.0212 | 0.7859 +- 0.0490 | 0.5770 +- 0.0883 | 0.8448 +- 0.0115 | 0.8573 +- 0.0102 |
| shared_weighted | 11 | 0.8084 +- 0.0261 | 0.8097 +- 0.0233 | 0.8071 +- 0.0428 | 0.5206 +- 0.0728 | 0.8604 +- 0.0122 | 0.8794 +- 0.0107 |
| shared | 11 | 0.7988 +- 0.0272 | 0.8064 +- 0.0237 | 0.7911 +- 0.0479 | 0.4600 +- 0.0782 | 0.8578 +- 0.0141 | 0.8774 +- 0.0119 |
| linear_prior | 11 | 0.8476 +- 0.0216 | 0.8201 +- 0.0187 | 0.8752 +- 0.0297 | 0.9476 +- 0.0141 | 0.7868 +- 0.0251 | 0.8231 +- 0.0190 |
| mean_tumour | 11 | 0.8266 +- 0.0240 | 0.7901 +- 0.0215 | 0.8631 +- 0.0320 | 0.6859 +- 0.0412 | 0.8347 +- 0.0151 | 0.8340 +- 0.0153 |
| mean_change | 11 | 0.6978 +- 0.0241 | 0.6831 +- 0.0213 | 0.7125 +- 0.0323 | 0.0003 +- 0.0001 | 0.7732 +- 0.0134 | 0.8000 +- 0.0122 |
| common_neighbors | 11 | 0.6420 +- 0.0230 | 0.6378 +- 0.0158 | 0.6462 +- 0.0363 | 0.0823 +- 0.0122 | 0.7323 +- 0.0206 | 0.7313 +- 0.0209 |
| identity | 11 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.0000 +- 0.0000 | 0.6361 +- 0.0189 | 0.6342 +- 0.0190 |

- paired shared - shared_weighted over 11 cancers: changed-edge AUROC -0.0167 (p 0.096, 3/11); gained -0.0123 (p 0.34, 6/11); lost -0.0211 (p 0.073, 1/11); direction +0.0565 (p 0.078, 9/11)
- paired shared - shared over 11 cancers: changed-edge AUROC -0.0071 (p 0.59, 5/11); gained -0.0091 (p 0.44, 4/11); lost -0.0051 (p 0.77, 5/11); direction +0.1171 (p 0.008, 11/11)
- paired shared - linear_prior over 11 cancers: changed-edge AUROC -0.0560 (p 0.0013, 1/11); gained -0.0227 (p 0.24, 3/11); lost -0.0892 (p 0.0023, 0/11); direction -0.3706 (p 0.0011, 0/11)
- paired shared - mean_tumour over 11 cancers: changed-edge AUROC -0.0349 (p 0.024, 2/11); gained +0.0073 (p 0.73, 4/11); lost -0.0771 (p 0.0038, 1/11); direction -0.1088 (p 0.12, 4/11)
- paired shared - mean_change over 11 cancers: changed-edge AUROC +0.0939 (p 0.00021, 11/11); gained +0.1143 (p 0.00045, 11/11); lost +0.0735 (p 0.011, 9/11); direction +0.5767 (p 6.6e-05, 11/11)
- paired shared - common_neighbors over 11 cancers: changed-edge AUROC +0.1496 (p 0.0016, 9/11); gained +0.1596 (p 4e-05, 11/11); lost +0.1397 (p 0.034, 8/11); direction +0.4948 (p 0.00028, 11/11)
- paired shared - identity over 11 cancers: changed-edge AUROC +0.2917 (p 1.1e-06, 11/11); gained +0.2974 (p 6.6e-08, 11/11); lost +0.2859 (p 0.00017, 11/11); direction +0.5770 (p 6.6e-05, 11/11)

## density 0.01 (main protocol)

| arm | n | changed-edge AUROC | gained | lost | direction | overall AUROC | overall AP@1:1 |
|---|---|---|---|---|---|---|---|
| shared_stratified | 11 | 0.7869 +- 0.0239 | 0.7901 +- 0.0211 | 0.7838 +- 0.0486 | 0.6004 +- 0.0680 | 0.8291 +- 0.0148 | 0.8379 +- 0.0144 |
| shared_weighted | 11 | 0.7939 +- 0.0209 | 0.8013 +- 0.0202 | 0.7866 +- 0.0403 | 0.5284 +- 0.0694 | 0.8449 +- 0.0136 | 0.8608 +- 0.0101 |
| shared | 11 | 0.7888 +- 0.0221 | 0.7868 +- 0.0237 | 0.7908 +- 0.0435 | 0.3925 +- 0.0565 | 0.8367 +- 0.0171 | 0.8577 +- 0.0132 |
| linear_prior | 11 | 0.8348 +- 0.0185 | 0.8052 +- 0.0173 | 0.8645 +- 0.0273 | 0.9424 +- 0.0142 | 0.7701 +- 0.0225 | 0.8071 +- 0.0163 |
| mean_tumour | 11 | 0.8130 +- 0.0201 | 0.7726 +- 0.0190 | 0.8534 +- 0.0289 | 0.6672 +- 0.0356 | 0.8143 +- 0.0111 | 0.8130 +- 0.0112 |
| mean_change | 11 | 0.6815 +- 0.0191 | 0.6618 +- 0.0179 | 0.7012 +- 0.0272 | 0.0004 +- 0.0001 | 0.7502 +- 0.0085 | 0.7776 +- 0.0074 |
| common_neighbors | 11 | 0.6321 +- 0.0244 | 0.6402 +- 0.0201 | 0.6240 +- 0.0348 | 0.0655 +- 0.0114 | 0.7273 +- 0.0209 | 0.7256 +- 0.0213 |
| identity | 11 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.0000 +- 0.0000 | 0.6272 +- 0.0148 | 0.6237 +- 0.0149 |

- paired shared - shared_weighted over 11 cancers: changed-edge AUROC -0.0070 (p 0.24, 6/11); gained -0.0112 (p 0.034, 2/11); lost -0.0028 (p 0.81, 7/11); direction +0.0720 (p 0.0016, 11/11)
- paired shared - shared over 11 cancers: changed-edge AUROC -0.0019 (p 0.84, 6/11); gained +0.0033 (p 0.71, 7/11); lost -0.0071 (p 0.55, 7/11); direction +0.2079 (p 0.00087, 11/11)
- paired shared - linear_prior over 11 cancers: changed-edge AUROC -0.0479 (p 0.00045, 1/11); gained -0.0151 (p 0.43, 5/11); lost -0.0807 (p 0.0047, 2/11); direction -0.3420 (p 0.00021, 0/11)
- paired shared - mean_tumour over 11 cancers: changed-edge AUROC -0.0261 (p 0.023, 1/11); gained +0.0175 (p 0.4, 6/11); lost -0.0696 (p 0.0085, 2/11); direction -0.0668 (p 0.21, 3/11)
- paired shared - mean_change over 11 cancers: changed-edge AUROC +0.1054 (p 6.3e-05, 11/11); gained +0.1283 (p 0.0002, 11/11); lost +0.0826 (p 0.015, 10/11); direction +0.6000 (p 4.9e-06, 11/11)
- paired shared - common_neighbors over 11 cancers: changed-edge AUROC +0.1548 (p 0.0028, 10/11); gained +0.1499 (p 1.5e-05, 11/11); lost +0.1598 (p 0.037, 8/11); direction +0.5349 (p 2.8e-05, 11/11)
- paired shared - identity over 11 cancers: changed-edge AUROC +0.2869 (p 2.9e-07, 11/11); gained +0.2901 (p 8.2e-08, 11/11); lost +0.2838 (p 0.00016, 11/11); direction +0.6004 (p 4.9e-06, 11/11)

## density 0.02 (sensitivity)

| arm | n | changed-edge AUROC | gained | lost | direction | overall AUROC | overall AP@1:1 |
|---|---|---|---|---|---|---|---|
| shared_stratified | 11 | 0.7576 +- 0.0280 | 0.7746 +- 0.0229 | 0.7406 +- 0.0496 | 0.5671 +- 0.0505 | 0.8024 +- 0.0213 | 0.8052 +- 0.0200 |
| shared_weighted | 11 | 0.7514 +- 0.0255 | 0.7729 +- 0.0232 | 0.7299 +- 0.0472 | 0.5104 +- 0.0403 | 0.8056 +- 0.0203 | 0.8096 +- 0.0164 |
| shared | 11 | 0.7685 +- 0.0171 | 0.7767 +- 0.0240 | 0.7603 +- 0.0330 | 0.3730 +- 0.0342 | 0.8201 +- 0.0197 | 0.8362 +- 0.0153 |
| linear_prior | 11 | 0.8189 +- 0.0149 | 0.7928 +- 0.0163 | 0.8450 +- 0.0215 | 0.9284 +- 0.0155 | 0.7566 +- 0.0177 | 0.7930 +- 0.0136 |
| mean_tumour | 11 | 0.7977 +- 0.0158 | 0.7606 +- 0.0173 | 0.8348 +- 0.0222 | 0.6493 +- 0.0307 | 0.7983 +- 0.0102 | 0.7954 +- 0.0099 |
| mean_change | 11 | 0.6625 +- 0.0137 | 0.6414 +- 0.0146 | 0.6836 +- 0.0209 | 0.0007 +- 0.0002 | 0.7271 +- 0.0080 | 0.7534 +- 0.0076 |
| common_neighbors | 11 | 0.6149 +- 0.0304 | 0.6465 +- 0.0229 | 0.5833 +- 0.0426 | 0.0594 +- 0.0115 | 0.7234 +- 0.0245 | 0.7178 +- 0.0258 |
| identity | 11 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.0000 +- 0.0000 | 0.6200 +- 0.0135 | 0.6134 +- 0.0136 |

- paired shared - shared_weighted over 11 cancers: changed-edge AUROC +0.0062 (p 0.44, 4/11); gained +0.0017 (p 0.79, 5/11); lost +0.0107 (p 0.43, 8/11); direction +0.0567 (p 0.25, 7/11)
- paired shared - shared over 11 cancers: changed-edge AUROC -0.0109 (p 0.49, 5/11); gained -0.0021 (p 0.77, 4/11); lost -0.0198 (p 0.47, 6/11); direction +0.1941 (p 0.0015, 10/11)
- paired shared - linear_prior over 11 cancers: changed-edge AUROC -0.0613 (p 0.0079, 0/11); gained -0.0182 (p 0.34, 5/11); lost -0.1044 (p 0.015, 1/11); direction -0.3613 (p 4.3e-06, 0/11)
- paired shared - mean_tumour over 11 cancers: changed-edge AUROC -0.0402 (p 0.056, 2/11); gained +0.0139 (p 0.48, 6/11); lost -0.0943 (p 0.026, 1/11); direction -0.0822 (p 0.041, 2/11)
- paired shared - mean_change over 11 cancers: changed-edge AUROC +0.0951 (p 0.0026, 10/11); gained +0.1332 (p 9.6e-05, 11/11); lost +0.0569 (p 0.21, 9/11); direction +0.5664 (p 5.5e-07, 11/11)
- paired shared - common_neighbors over 11 cancers: changed-edge AUROC +0.1426 (p 0.017, 10/11); gained +0.1280 (p 5.8e-05, 11/11); lost +0.1572 (p 0.097, 8/11); direction +0.5077 (p 5.7e-06, 11/11)
- paired shared - identity over 11 cancers: changed-edge AUROC +0.2576 (p 3.4e-06, 11/11); gained +0.2746 (p 3e-07, 11/11); lost +0.2406 (p 0.00067, 10/11); direction +0.5671 (p 5.5e-07, 11/11)

## Per-cancer detail, main protocol (shared model)

| held-out cancer | changed AUROC | gained | lost | direction | overall AUROC | edges | gained/lost cells |
|---|---|---|---|---|---|---|---|
| Head & Neck Squamous Cell Carcinoma | 0.8883 | 0.8804 | 0.8963 | 0.7710 | 0.8988 | 19990 | 14431/14431 |
| Breast Invasive Carcinoma | 0.8864 | 0.8230 | 0.9497 | 0.8006 | 0.8459 | 19990 | 16666/16666 |
| Kidney Clear Cell Carcinoma | 0.8424 | 0.7175 | 0.9672 | 0.8885 | 0.7689 | 19990 | 15385/15385 |
| Kidney Papillary Cell Carcinoma | 0.8343 | 0.7370 | 0.9316 | 0.8830 | 0.7622 | 19990 | 16620/16620 |
| Lung Adenocarcinoma | 0.8248 | 0.7986 | 0.8510 | 0.6924 | 0.8238 | 19990 | 15310/15310 |
| Lung Squamous Cell Carcinoma | 0.8224 | 0.7557 | 0.8891 | 0.6309 | 0.7974 | 19990 | 15452/15452 |
| Liver Hepatocellular Carcinoma | 0.7660 | 0.8044 | 0.7276 | 0.3828 | 0.8458 | 19990 | 14225/14225 |
| Stomach Adenocarcinoma | 0.7380 | 0.8955 | 0.5804 | 0.5604 | 0.9045 | 19990 | 15399/15399 |
| Prostate Adenocarcinoma | 0.6966 | 0.7824 | 0.6108 | 0.3514 | 0.8395 | 19990 | 12359/12359 |
| Colon Adenocarcinoma | 0.6833 | 0.8368 | 0.5298 | 0.2960 | 0.8581 | 19990 | 16251/16251 |
| Thyroid Carcinoma | 0.6737 | 0.6596 | 0.6878 | 0.3472 | 0.7750 | 19990 | 10221/10221 |

