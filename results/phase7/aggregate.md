# Phase 7 -- normal graph -> tumour graph translation (165 rows @ 82975113, f75392da)

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
| shared | 11 | 0.7988 +- 0.0272 | 0.8064 +- 0.0237 | 0.7911 +- 0.0479 | 0.4600 +- 0.0782 | 0.8578 +- 0.0141 | 0.8774 +- 0.0119 |
| mean_change | 11 | 0.6978 +- 0.0241 | 0.6831 +- 0.0213 | 0.7125 +- 0.0323 | 0.0003 +- 0.0001 | 0.7732 +- 0.0134 | 0.8000 +- 0.0122 |
| mean_tumour | 11 | 0.8266 +- 0.0240 | 0.7901 +- 0.0215 | 0.8631 +- 0.0320 | 0.6859 +- 0.0412 | 0.8347 +- 0.0151 | 0.8340 +- 0.0153 |
| common_neighbors | 11 | 0.6420 +- 0.0230 | 0.6378 +- 0.0158 | 0.6462 +- 0.0363 | 0.0823 +- 0.0122 | 0.7323 +- 0.0206 | 0.7313 +- 0.0209 |
| identity | 11 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.0000 +- 0.0000 | 0.6361 +- 0.0189 | 0.6342 +- 0.0190 |

- paired shared - mean_change over 11 cancers: changed-edge AUROC +0.1010 (p 0.0013, 9/11); gained +0.1234 (p 0.0011, 11/11); lost +0.0786 (p 0.026, 9/11); direction +0.4596 (p 0.00016, 11/11)
- paired shared - mean_tumour over 11 cancers: changed-edge AUROC -0.0278 (p 0.17, 3/11); gained +0.0163 (p 0.54, 4/11); lost -0.0720 (p 0.02, 0/11); direction -0.2259 (p 0.0025, 1/11)
- paired shared - common_neighbors over 11 cancers: changed-edge AUROC +0.1567 (p 0.00076, 11/11); gained +0.1686 (p 9.6e-06, 11/11); lost +0.1449 (p 0.026, 8/11); direction +0.3777 (p 0.00075, 11/11)
- paired shared - identity over 11 cancers: changed-edge AUROC +0.2988 (p 6.6e-07, 11/11); gained +0.3064 (p 1.4e-07, 11/11); lost +0.2911 (p 0.00012, 10/11); direction +0.4600 (p 0.00015, 11/11)

## density 0.01 (main protocol)

| arm | n | changed-edge AUROC | gained | lost | direction | overall AUROC | overall AP@1:1 |
|---|---|---|---|---|---|---|---|
| shared | 11 | 0.7888 +- 0.0221 | 0.7868 +- 0.0237 | 0.7908 +- 0.0435 | 0.3925 +- 0.0565 | 0.8367 +- 0.0171 | 0.8577 +- 0.0132 |
| mean_change | 11 | 0.6815 +- 0.0191 | 0.6618 +- 0.0179 | 0.7012 +- 0.0272 | 0.0004 +- 0.0001 | 0.7502 +- 0.0085 | 0.7776 +- 0.0074 |
| mean_tumour | 11 | 0.8130 +- 0.0201 | 0.7726 +- 0.0190 | 0.8534 +- 0.0289 | 0.6672 +- 0.0356 | 0.8143 +- 0.0111 | 0.8130 +- 0.0112 |
| common_neighbors | 11 | 0.6321 +- 0.0244 | 0.6402 +- 0.0201 | 0.6240 +- 0.0348 | 0.0655 +- 0.0114 | 0.7273 +- 0.0209 | 0.7256 +- 0.0213 |
| identity | 11 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.0000 +- 0.0000 | 0.6272 +- 0.0148 | 0.6237 +- 0.0149 |

- paired shared - mean_change over 11 cancers: changed-edge AUROC +0.1073 (p 0.00033, 11/11); gained +0.1250 (p 0.00083, 10/11); lost +0.0896 (p 0.0088, 9/11); direction +0.3922 (p 3.9e-05, 11/11)
- paired shared - mean_tumour over 11 cancers: changed-edge AUROC -0.0242 (p 0.13, 3/11); gained +0.0142 (p 0.56, 6/11); lost -0.0626 (p 0.012, 2/11); direction -0.2747 (p 0.00011, 0/11)
- paired shared - common_neighbors over 11 cancers: changed-edge AUROC +0.1567 (p 0.0011, 11/11); gained +0.1466 (p 8.3e-06, 11/11); lost +0.1669 (p 0.019, 7/11); direction +0.3270 (p 0.00021, 11/11)
- paired shared - identity over 11 cancers: changed-edge AUROC +0.2888 (p 1.3e-07, 11/11); gained +0.2868 (p 2.7e-07, 11/11); lost +0.2908 (p 5.5e-05, 11/11); direction +0.3925 (p 3.9e-05, 11/11)

## density 0.02 (sensitivity)

| arm | n | changed-edge AUROC | gained | lost | direction | overall AUROC | overall AP@1:1 |
|---|---|---|---|---|---|---|---|
| shared | 11 | 0.7685 +- 0.0171 | 0.7767 +- 0.0240 | 0.7603 +- 0.0330 | 0.3730 +- 0.0342 | 0.8201 +- 0.0197 | 0.8362 +- 0.0153 |
| mean_change | 11 | 0.6625 +- 0.0137 | 0.6414 +- 0.0146 | 0.6836 +- 0.0209 | 0.0007 +- 0.0002 | 0.7271 +- 0.0080 | 0.7534 +- 0.0076 |
| mean_tumour | 11 | 0.7977 +- 0.0158 | 0.7606 +- 0.0173 | 0.8348 +- 0.0222 | 0.6493 +- 0.0307 | 0.7983 +- 0.0102 | 0.7954 +- 0.0099 |
| common_neighbors | 11 | 0.6149 +- 0.0304 | 0.6465 +- 0.0229 | 0.5833 +- 0.0426 | 0.0594 +- 0.0115 | 0.7234 +- 0.0245 | 0.7178 +- 0.0258 |
| identity | 11 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.5000 +- 0.0000 | 0.0000 +- 0.0000 | 0.6200 +- 0.0135 | 0.6134 +- 0.0136 |

- paired shared - mean_change over 11 cancers: changed-edge AUROC +0.1060 (p 0.00018, 11/11); gained +0.1353 (p 0.00021, 11/11); lost +0.0767 (p 0.014, 9/11); direction +0.3723 (p 7.3e-07, 11/11)
- paired shared - mean_tumour over 11 cancers: changed-edge AUROC -0.0292 (p 0.072, 3/11); gained +0.0160 (p 0.5, 7/11); lost -0.0745 (p 0.0038, 1/11); direction -0.2762 (p 1.2e-05, 0/11)
- paired shared - common_neighbors over 11 cancers: changed-edge AUROC +0.1536 (p 0.003, 10/11); gained +0.1301 (p 7.6e-06, 11/11); lost +0.1770 (p 0.03, 9/11); direction +0.3137 (p 6.7e-06, 11/11)
- paired shared - identity over 11 cancers: changed-edge AUROC +0.2685 (p 2.2e-08, 11/11); gained +0.2767 (p 4.3e-07, 11/11); lost +0.2603 (p 1.3e-05, 11/11); direction +0.3730 (p 7.2e-07, 11/11)

## Per-cancer detail, main protocol (shared model)

| held-out cancer | changed AUROC | gained | lost | direction | overall AUROC | edges | gained/lost cells |
|---|---|---|---|---|---|---|---|
| Head & Neck Squamous Cell Carcinoma | 0.8858 | 0.8742 | 0.8974 | 0.6873 | 0.9000 | 19990 | 14431/14431 |
| Breast Invasive Carcinoma | 0.8759 | 0.8145 | 0.9374 | 0.6240 | 0.8412 | 19990 | 16666/16666 |
| Kidney Clear Cell Carcinoma | 0.8481 | 0.7297 | 0.9664 | 0.4706 | 0.7877 | 19990 | 15385/15385 |
| Lung Adenocarcinoma | 0.8334 | 0.8186 | 0.8481 | 0.5357 | 0.8523 | 19990 | 15310/15310 |
| Lung Squamous Cell Carcinoma | 0.8135 | 0.7512 | 0.8758 | 0.4483 | 0.7983 | 19990 | 15452/15452 |
| Kidney Papillary Cell Carcinoma | 0.7878 | 0.6726 | 0.9030 | 0.3662 | 0.7214 | 19990 | 16620/16620 |
| Stomach Adenocarcinoma | 0.7859 | 0.9111 | 0.6608 | 0.3227 | 0.9252 | 19990 | 15399/15399 |
| Prostate Adenocarcinoma | 0.7573 | 0.8307 | 0.6838 | 0.2880 | 0.8875 | 19990 | 12359/12359 |
| Liver Hepatocellular Carcinoma | 0.7461 | 0.7901 | 0.7020 | 0.3610 | 0.8357 | 19990 | 14225/14225 |
| Thyroid Carcinoma | 0.6859 | 0.6555 | 0.7162 | 0.1046 | 0.8194 | 19990 | 10221/10221 |
| Colon Adenocarcinoma | 0.6574 | 0.8064 | 0.5084 | 0.1094 | 0.8356 | 19990 | 16251/16251 |

