note: results\phase4\direct\rows rows span 2 commits with identical model code (6b4c6d05, af03a478)
12 Phase-4W rows @ af03a478; Phase 4 direct rows @ af03a478 (reproduction check only)

INCOMPLETE — cells with fewer rows than the 3 seeds: {'gnn-direct gat d1024 L2': {'1e-02': 1, '3e-03': 1}, 'gnn-direct gat d256 L2': {'1e-02': 1, '3e-03': 1}, 'gnn-direct gat d2708 L2': {'1e-02': 1, '3e-03': 1}, 'gnn-direct gcn d1024 L2': {'1e-02': 1, '3e-03': 1}, 'gnn-direct gcn d256 L2': {'1e-02': 1, '3e-03': 1}, 'gnn-direct gcn d2708 L2': {'1e-02': 1, '3e-03': 1}}

| model | enc lr | bias lr | LoRA lr | n | AUROC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift | best epoch | s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gnn-direct gat d256 L2 (Phase 4 direct) | 3e-03 | – | – | 3 | 0.928±0.006 | 0.935±0.005 | 0.932±0.004 | 0.0116±0.0028 | 81 | 264 | 0.07 |
| gnn-direct gcn d256 L2 (Phase 4 direct) | 3e-03 | – | – | 3 | 0.934±0.006 | 0.941±0.006 | 0.936±0.002 | 0.0133±0.0028 | 93 | 284 | 0.05 |
| gnn-direct gat d1024 L2 | 1e-02 EDGE | – | – | 1 | 0.923±0.000 | 0.934±0.000 | 0.929±0.000 | 0.0089±0.0000 | 62 | 101 | 0.10 |
| gnn-direct gat d256 L2 | 3e-03 EDGE | – | – | 1 | 0.927±0.000 | 0.930±0.000 | 0.936±0.000 | 0.0081±0.0000 | 56 | 123 | 0.07 |
| gnn-direct gat d2708 L2 | 3e-03 EDGE | – | – | 1 | 0.930±0.000 | 0.935±0.000 | 0.935±0.000 | 0.0108±0.0000 | 75 | 264 | 0.15 |
| gnn-direct gcn d1024 L2 | 1e-02 EDGE | – | – | 1 | 0.939±0.000 | 0.944±0.000 | 0.943±0.000 | 0.0109±0.0000 | 75 | 122 | 0.08 |
| gnn-direct gcn d256 L2 | 3e-03 EDGE | – | – | 1 | 0.929±0.000 | 0.935±0.000 | 0.936±0.000 | 0.0097±0.0000 | 68 | 129 | 0.09 |
| gnn-direct gcn d2708 L2 | 3e-03 EDGE | – | – | 1 | 0.934±0.000 | 0.940±0.000 | 0.940±0.000 | 0.0122±0.0000 | 85 | 123 | 0.14 | 

Same-seed reproduction of the Phase 4 direct rows at this commit:

| arm | seeds | previous AUROC | this phase AUROC | mean abs Δ | max abs Δ |
|---|---|---|---|---|---|
| gnn-direct gat d256 L2 | 1 | 0.927±0.000 | 0.927±0.000 | 0.0000 | 0.0000 |
| gnn-direct gcn d256 L2 | 1 | 0.929±0.000 | 0.929±0.000 | 0.0000 | 0.0000 | 

inertness at init: 0 biased / LoRA rows compared with their plain counterpart (z_norm, logit_diag, logit_offdiag_std, logit_train_edge, layer_rms); 0 mismatch(es) above 1e-2 relative 

Paired per-seed deltas (all rows at this commit):

| Δ (A − B) | column | n | mean | SE | paired t | p | MDE |
|---|---|---|---|---|---|---|---| 

LR surfaces (mean val AUROC):

gnn-direct gat d1024 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9235 * (1), 3e-03 0.9216 (1)

gnn-direct gat d256 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9191 (1), 3e-03 0.9234 * (1)

gnn-direct gat d2708 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9175 (1), 3e-03 0.9176 * (1)

gnn-direct gcn d1024 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9245 * (1), 3e-03 0.9224 (1)

gnn-direct gcn d256 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9261 (1), 3e-03 0.9263 * (1)

gnn-direct gcn d2708 L2 — mean val AUROC by encoder LR (* selected): 1e-02 0.9221 (1), 3e-03 0.9272 * (1)

