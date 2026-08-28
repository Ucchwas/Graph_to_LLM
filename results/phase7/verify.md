# Phase 7C -- strict-protocol verification (rho = 0.01)

### normaliser: znorm (11 folds)

| arm | changed-edge AUROC | direction | overall AUROC |
|---|---|---|---|
| combo_strict | 0.8686 +- 0.0154 | 0.9154 | 0.8283 |
| combo_published | 0.8670 +- 0.0158 | 0.9252 | 0.8305 |
| prior_loo | 0.8348 +- 0.0185 | 0.9424 | 0.7701 |
| prior_val | 0.8327 +- 0.0181 | 0.9129 | 0.7699 |
| model | 0.7869 +- 0.0239 | 0.6004 | 0.8291 |

- paired combo_strict - prior_loo: +0.0337 (p 0.0001, 11/11)
- paired combo_strict - model: +0.0816 (p 1.3e-05, 11/11)
- paired combo_strict - combo_published: +0.0015 (p 0.24, 5/11)
- paired combo_published - prior_loo: +0.0322 (p 0.00029, 11/11)

selected (lambda, alpha) per fold: [(-1.0, 0.5), (-0.6, 0.3), (-0.4, 0.2), (-0.4, 0.4), (-0.4, 0.5), (-0.4, 0.6), (-0.3, 0.1), (-0.3, 0.2), (-0.3, 0.3), (-0.1, 0.2), (0.0, 0.6)]
lambda by train-LOO: [-0.4, -0.3]

### normaliser: ranknorm (11 folds)

| arm | changed-edge AUROC | direction | overall AUROC |
|---|---|---|---|
| combo_strict | 0.8645 +- 0.0164 | 0.8389 | 0.8486 |
| combo_published | 0.8485 +- 0.0194 | 0.8946 | 0.8462 |
| prior_loo | 0.8348 +- 0.0185 | 0.9424 | 0.7701 |
| prior_val | 0.8316 +- 0.0178 | 0.8437 | 0.7981 |
| model | 0.7869 +- 0.0239 | 0.6004 | 0.8291 |

- paired combo_strict - prior_loo: +0.0297 (p 0.00013, 11/11)
- paired combo_strict - model: +0.0776 (p 9.7e-06, 11/11)
- paired combo_strict - combo_published: +0.0160 (p 0.0049, 11/11)
- paired combo_published - prior_loo: +0.0136 (p 0.015, 9/11)

selected (lambda, alpha) per fold: [(-1.0, 0.1), (-0.4, 0.1), (-0.3, 0.1), (-0.2, 0.1), (-0.1, 0.1), (-0.1, 0.4), (0.0, 0.1)]
lambda by train-LOO: [-0.4, -0.3]

