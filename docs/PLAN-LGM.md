# Plan after the 2026-08-27 redirection: we are building an LGM

## 1. What Dr. Islam asked for

Slack, 2026-08-27. After Sakib suggested feeding the lab GFM's embeddings to the LLM through a
small adapter, Dr. Islam wrote: *"We are not building LLM, we are building LGM. So the graph
input should not be changed to embedding, but we might integrate GFM architecture into the end
to end network to train LGM."* Confirmed the next morning: **raw adjacency / edges → GFM/LGM
architecture directly → graph output. "Yes."**

Vocabulary, so the thread reads unambiguously:
- **GFM** (graph foundation model): a graph network (GNN or graph transformer) pretrained on
  many graphs so that it transfers. The lab has one (Sakib's). Its details are unknown to us.
- **LGM** (large graph model, Zhang et al. arXiv 2308.14522): a large network whose native
  input *and output* are graphs, trained end to end. That is the deliverable.
- Rejected: any design where a separate model turns the graph into vectors that a language
  model then consumes (GraphToken / LLaGA / "GFM embeddings + adapter"). The graph enters the
  trained network as a graph. Also rejected by implication: the frozen pretrained LLM as the
  centre of the project.

## 2. What Phase 3 already showed (Cora, masked edge prediction, AUROC, 20 seeds)

| body between the row tokenizer and the graph decoder | AUROC |
|---|---|
| frozen **random-weight** Llama-1B + structure bias | **0.922 ± 0.006** |
| no body (linear tokenizer + decoder) | 0.918 ± 0.006 |
| frozen **pretrained** Llama-1B + structure bias | 0.909 ± 0.011 |
| GAT / GAE (Phase 1) | 0.910 / 0.905 |
| small graph transformer trained end to end (d 256, 4 layers) + bias | 0.895 ± 0.010 |
| frozen random / pretrained Llama, no bias | 0.889 / 0.881 |
| structure bias on a degree-preserving rewired graph (control) | 0.688 |

Reading in LGM terms: the graph-structure channel into attention is the ingredient (+0.03 for
every body, p < 0.001; the learned table becomes a hop-distance routing mask). The language
pretraining is not (pretrained − random = −0.013, p < 0.001). Cora itself is saturated near
0.92–0.93 for anything with a bilinear decoder, so it can sanity-check an LGM but cannot rank
LGMs; the ranking has to come from multi-graph data (§5, Phase 6).

Two facts that shape the design: the trained small transformer (0.895) lost to the frozen
random large one (0.922), so on a *single* graph a trained body needs width and regularisation
(one graph is one training example); and "no body" already matches GNNs, so every LGM result
must be paired against that control or the body's contribution cannot be claimed.

## 3. The LGM, as we will build it

```
A_obs [N,N] (15 % of cells hidden; edges only, no text, no external embeddings)
   │
   ├─► node tokens                              structure channel
   │   E1 row tokenizer  (a layer of the net)   per-head hop-distance bias b_h(spd(i,j)),
   │   or the GFM's own node input layer        computed on A_obs, added to attention logits
   ▼                                            in every layer (Graphormer / GTLM style)
BODY, trainable, graph-native, one of
   gt   our graph transformer: pre-LN encoder, width d, depth L, bias in every layer
   gfm  the lab GFM architecture (Sakib), from its checkpoint or from scratch
   (baselines kept as rows: frozen pretrained / random Llama, no body, GAE, GAT)
   ▼  H [N, d]
DECODER  Â = σ(H W Hᵀ)   (unchanged, permutation-equivariant, graph out)
LOSS     BCE with pos_weight on the hidden cells only (unchanged); one optimizer, LR groups
         {tokenizer, body, bias table, decoder}; everything trains.
```

Task stays **masked edge prediction**: it is the self-supervised way to train a graph-in /
graph-out model without labels, and graph-to-graph translation (normal → tumour) is the same
code path with a different target tensor (Phase 7).

What survives from Phases 0–3, unchanged: data/masking/splits ([data.py](../g2l/data.py),
multi-graph `DenseBatch` already there), the three-column metric protocol, the masked-cell
loss and training loop, the SPD bias module, the D1 decoder, the encoder interface, the
baselines and their tables, the Slurm harness, the paired-delta aggregation, the walkthrough
discipline. The frozen-LLM path stays as baseline rows; it is no longer developed.

Superseded rules in CLAUDE.md (they were ours, written for the LLM framing): "frozen LLM
body / never unfreeze", "no GNN encoder", "LoRA only". A short amendment at the top of
CLAUDE.md records this; nothing else there changes.

## 4. What we need from Sakib (critical path for Phase 5)

1. Input / output contract: dense adjacency or `edge_index`; required node features (Cora has
   1433-dim bag-of-words, molecules have atom types; "raw adjacency" means it must also run
   with none); per-node output width; one forward on 2.7 k nodes or subgraphs only.
2. Architecture and size: message passing or attention; layers, width, parameter count; any
   structural encodings it computes (SPD, random walk, Laplacian) that our bias path could share.
3. Code and checkpoint (path on Marlowe, torch / PyG versions), and whether training it
   unfrozen with our loss is acceptable.
4. Pretraining data. If it saw Cora, PubMed, ogbn-arxiv or the OGB molecular sets, masked-edge
   numbers on those are leakage and we use a scratch-initialised copy there.

## 5. Phases

**Phase 3 close-out (now, laptop, no jobs).** `results/phase3/RESULTS.md` in LGM framing,
selections recorded (random + SPD at bias LR 3.0 sits on the grid edge; noted, not extended),
the data-fraction sweep declared uninformative (dropped edges were labelled negative), cluster
checkpoints pruned, commit. **Phase 3B (LoRA on the frozen LLM) is cancelled, not run**: it
only answers a question about the pretrained LLM.

**Phase 4 — LGM v0 on Cora: our graph transformer, trained end to end (~1 week, ≤ 25 GPU-h).**
No dependency on Sakib. Body `gt` = `ScratchBody` generalised (width, depth, dropout, weight
decay, `frozen` flag). Grid, pre-registered in PLAN-PHASE4 before submission: width
{512, 2048} × depth {1, 4} × body LR (3) × bias LR {0.1, 1} × 10 seeds, plus the same
architectures **frozen at random init** (the reservoir control that separates "trained" from
"large"), plus shuffled-A and no-bias controls. Selection on val, paired deltas against the
0.918 / 0.922 / GAE / GAT rows. Gate 4: does training the body beat freezing it at equal size;
does width or depth matter on one graph; where the LGM v0 sits against the linear control.
Whatever the answer, it is the control every GFM result in Phase 5 is paired against.

**Phase 5 — GFM integrated end to end (starts when §4 is answered; runs beside Phase 4).**
`gfm` body (or tokenizer + `gt` on top: the "integrate GFM architecture into the end-to-end
network" configuration) behind the existing interfaces, own LR group, leakage check, three
arms: GFM from checkpoint, GFM from scratch, GFM + structure-biased `gt`. Controls: GFM +
decoder alone is a GAE-class model, so it is a row, not the claim. Gate 5: GFM-based LGM vs
`gt`-based LGM vs GAE/GAT, paired, same protocol.

**Phase 6 — the "large" claim: multi-graph LGM (prepare on the laptop during Phase 4).**
OGB molecular (ogbg-molbbbp 2 k graphs for sweeps, ogbg-molhiv 41 k for confirmation):
masked edge prediction on held-out graphs (inductive), then cross-dataset transfer. Needs
per-graph masking and padding (exists), the size-agnostic tokenizers E3/E6/E7 (E1 is fixed-N),
and **random atom re-indexing per graph** (SMILES order makes adjacent indices bonded; a
fixed-N tokenizer would exploit it). Width / depth / data scaling curves of the best body from
Phases 4–5. This is where an LGM is ranked. ~40–80 GPU-h.

**Phase 7 — graph-to-graph translation, biomedical.** Same code path, target tensor changes;
directed graphs need a directed structure channel (magnetic Laplacian); data-classification
review first (both clusters are Low/Moderate-risk only).

## 6. Rules that do not change

One pre-registered chunked array per phase, sized so no extension is expected; selection on
base seeds, paired deltas with SE / p / MDE; identity, shuffled-A and no-body controls on every
table; rows stamped with the pinned cluster commit; RESULTS.md + live walkthrough + VALIDATED
line before the next phase's first commit; ask before any submission.
