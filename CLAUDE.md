# CLAUDE.md — Graph-In / Graph-Out LLM

Context and build plan for this repository. Read this fully before writing code.

---

## 1. What we are building

A model that takes a **raw graph** in and produces a **graph** out, with a pretrained LLM in the middle.

```
raw adjacency A [N,N]  →  [our encoder]  →  tokens [N,d_model]
                       →  frozen pretrained LLM (attention biased by the graph)
                       →  hidden states H [N,d_model]
                       →  [our decoder]  →  Â [N,N]
```

Parts of the input graph are hidden. The model predicts what is missing.

### The one thing that makes this different

Every existing method converts the graph into something else before the LLM sees it:

- **Verbalization** — the graph is written as sentences ("node 1 is connected to nodes 2 and 3"). Used by *Talk Like a Graph* (Fatemi, Halcrow & Perozzi, ICLR 2024, arXiv 2310.04560).
- **GNN soft prompts** — a GNN compresses each node into one vector, prepended to the prompt. Used by *GraphToken* (Perozzi et al. 2024, arXiv 2402.05862), GraphGPT, LLaGA.

We do neither. **We design and train our own encoders that map raw adjacency rows into LLM token space**, and we decode a graph back out instead of text. The LLM's tokenizer and embedding table are unused.

### Scope

Phase 1 is standard graph benchmarks (Cora, molecular datasets). Biomedical data — gene regulatory networks, normal→tumour translation — is a later extension. **Build the architecture so the data source is swappable.** Nothing in the model should assume a particular domain.

---

## 2. Architecture specification

```
INPUT
  A            [N, N]          adjacency, 0/1, possibly with entries masked
  X            [N, F]          optional node features (bag-of-words, expression, etc.)

ENCODER  (ours — trainable, this is a core research contribution)
  tokens       [N, d_model]    one token per node

LLM BODY  (frozen weights + LoRA)
  attention:   score(i,j) = (q_i·k_j)/sqrt(d) + bias(i,j)
  bias comes from graph structure, added at every layer
  RoPE disabled / reset per node
  bidirectional (non-causal) mask over graph tokens
  H            [N, d_model]

DECODER  (ours — trainable)
  Â            [N, N]          predicted adjacency (logits, then sigmoid)
```

**Trainable:** encoder, attention-bias parameters, LoRA adapters, decoder.
**Frozen:** the LLM body.

---

## 3. Encoders to design and implement

This is the part the PI specifically asked for. Implement all of these behind a common interface so they can be swapped and compared. **The comparative study of encoders is a deliverable in itself, not a preliminary step.**

Common interface:

```python
class GraphEncoder(nn.Module):
    """Maps a raw graph to LLM token embeddings.
    A: [B, N, N] float  (masked entries are 0, with a separate mask tensor)
    X: [B, N, F] float or None
    returns: [B, N, d_model]
    """
    def forward(self, A, X=None, mask=None) -> torch.Tensor: ...
```

### E1 — Linear
`nn.Linear(N, d_model)` applied to each adjacency row. The simplest possible thing and the reference point everything else must beat. Ties the model to a fixed N.

### E2 — MLP
Two or three layers with nonlinearity. Tests whether depth in the encoder helps or whether the LLM should do all the work.

### E3 — Sketch
Project the adjacency row through a fixed random matrix `R [N, k]` (k ≈ 256–1024), then `Linear(k, d_model)`. Motivated by linear sketching of adjacency rows (Ahn, Guha & McGregor 2012), which the Depth-Width paper (below) cites as sufficient to solve connectivity despite being lossy compression. Decouples encoder size from N.

### E4 — Row + features
Concatenate node features to the adjacency row before projection: `[A_i,: ‖ x_i]`, giving width N+F. This is exactly what the Depth-Width paper does in practice (their Appendix E), and they zero-pad to the largest graph in the dataset.

### E5 — Row + structural
`[A_i,: ‖ laplacian_eigvecs_i ‖ degree_i ‖ x_i]`. **Important — see the warning in §11 about adjacency rows being local-biased.** Laplacian eigenvectors supply the global information adjacency rows are bad at.

### E6 — Set-based / DeepSets
Treat node i's neighbours as a *set* of node embeddings, aggregate permutation-invariantly, project. Size-agnostic and permutation-aware. This is **not** a GNN — there is no message passing, no multi-hop propagation, one aggregation step over the immediate neighbour set only. Keep that distinction clear in comments and in the paper.

### E7 — Sparse / bucketed
For large sparse graphs, encode only nonzero indices via an embedding table plus positional hashing. Avoids materialising an N-wide dense row.

Each encoder gets a config entry, a unit test proving output shape `[B, N, d_model]`, and a parameter count logged at init.

---

## 4. Decoders to implement

### D1 — Inner product (default)
```python
Â = sigmoid(H @ W @ H.T)          # W: [d_model, d_model], or low-rank [d_model, r] @ [r, d_model]
```
Every entry is an inner product between two node states. **Permutation-equivariant by construction** — permute the input nodes and the output permutes identically. No node ordering to choose. Lineage: Kipf & Welling, *Variational Graph Auto-Encoders*, 2016.

### D2 — Bilinear low-rank
Same as D1 with `r` small (32–128). Cheaper, regularising.

### D3 — MLP on pairs
`MLP([h_i ‖ h_j ‖ h_i * h_j])`. More expressive, O(N²) MLP calls, only feasible on small N or sampled pairs.

### D4 — Autoregressive head
Predict row *t* conditioned on rows `< t`. Needed for the autoregressive task (§5.3). **Requires a node ordering** — use BFS canonical ordering and record the choice; this is a known weakness and reviewers ask about it.

---

## 5. Tasks and losses

Implement all three. They share the encoder, LLM body and decoder.

### 5.1 Masked edge prediction (PRIMARY — build this first)

Hide a random 15% of edge entries. Predict them.

```python
loss = F.binary_cross_entropy_with_logits(
    logits[mask], A_true[mask], pos_weight=w
)
```

Self-supervised, no labels needed, parallel, and **no node-ordering problem**. This is the task to get working end to end before anything else.

### 5.2 Masked node prediction

Hide a node's entire adjacency row. Predict the whole row. Harder than 5.1 and closer to "a token is missing."

### 5.3 Autoregressive

Reveal nodes one at a time in a canonical (BFS) order. At step *t*, predict edges from node *t* to all nodes `< t`.

```
L = -Σ_t Σ_{j<t} [ A_tj·log p_tj + (1-A_tj)·log(1-p_tj) ]
```

This is next-token prediction with a Bernoulli head instead of a softmax over vocabulary. Precedents: GraphRNN (You et al., ICML 2018), GRAN (Liao et al., NeurIPS 2019). Implement it, benchmark it against 5.1, and report both.

### 5.4 Graph-to-graph translation (later)

Input graph in condition A, target graph in condition B, same nodes. Same code path — only the `target` tensor changes. Keep the data loader abstract enough that this needs no model changes.

### Auxiliary losses (add once the base trains)

```
L_total = w·BCE  +  λ1·L_degree  +  λ2·L_spectral
```

`L_degree` = MSE between predicted and true degree per node.
`L_spectral` = MSE between top-k Laplacian eigenvalues of Â and A.

Without these the model produces locally plausible but globally implausible graphs.

---

## 6. Datasets, in order

**Phase 1 — Cora.** 2,708 nodes, 5,429 edges, 7 classes, 1,433-dim bag-of-words node features. Available in PyTorch Geometric as `Planetoid(root, 'Cora')`. Use it for masked edge prediction and node classification.

**Phase 2 — OGB molecular.** `ogbg-molhiv`, `ogbg-molbbbp`, `ogbg-molbace`. Small graphs (average 24–34 nodes) so they train fast, and the Depth-Width paper reports adjacency-row numbers on exactly these — direct comparison available.

**Phase 3 — PubMed, ogbn-arxiv, Reddit.** Larger. GTLM reports numbers on all of these.

**Phase 4 — biomedical.** Later. Keep the loader interface generic: anything that yields `(A_input, A_target, X, mask)` should work without touching the model.

---

## 7. Baselines — run these BEFORE the LLM model

Do not skip. If a cheap baseline already solves the task, we need to know in week one.

1. **Identity** — output the input unchanged.
2. **Random** — predict edges at the base rate.
3. **Feature-only** — logistic regression / XGBoost on node features, no graph at all.
4. **GAE / VGAE** — PyTorch Geometric `GAE` with a 2-layer GCN encoder and the same inner-product decoder. This is the number to beat.
5. **GAT** — PyG `GATConv`.
6. **Transformer from scratch** — adjacency-row tokens into a randomly-initialised transformer, no pretrained weights. **This isolates what the pretrained LLM actually contributes.** Arguably the single most important baseline in the whole project.

---

## 8. Ablations and controls

Run all of these. They are the spine of any writeup.

```
No attention bias            plain frozen LLM, graph enters only via encoder
Soft hop bias                the proposed model
Hard 1-hop mask              equivalent to GAT-style attention
Shuffled adjacency           edges rewired at random, degree preserved
Random-init LLM              same architecture, pretrained weights discarded
Encoder sweep                E1 … E7
Decoder sweep                D1 … D4
```

**Shuffled adjacency is the decisive control.** Same graph statistics, edges randomised. If performance holds up, the graph was never doing anything and the model is exploiting something else. Implement this from day one, not at the end.

---

## 9. Reference codebases — study these, take what applies

### GTLM — `github.com/DarioVajda/graph_model`
*"Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning"*, Dario Vajda, arXiv 2605.10247.

**The closest existing work and the most useful repo.** It puts graph structure into a frozen LLM's attention. It feeds node *text* and outputs *text* — we replace both ends.

Take from it:
- **Attention bias injection.** Three learned biases added to pre-softmax logits at every layer: SPD (learned lookup on shortest-path distance), RRWP (K-step random walk probabilities through an MLP), Magnetic Laplacian (directed spectral). Reported hyperparameters: SPD max distance 8, RRWP max steps 16, Magnetic Laplacian q = 0.25, dim 32. Total 173,056 bias parameters = 0.015% of Llama-3.2-1B.
- **RoPE reset per node** and **bidirectional (non-causal) prefix mask.** Together these give node permutation equivariance, which they prove.
- **Intra-node zero bias.** Bias is forced to 0 for token pairs inside the same node, which guarantees a single-node graph runs identically to the base LLM. They verified numerically (difference ~2×10⁻⁵, floating-point noise). Preserves pretrained behaviour.
- **Differential learning rate.** Bias parameters get a much higher LR than LoRA — reported 5e-3 vs 3e-5 on GraphQA, and 0.01–0.04 vs 6e-5–3e-4 on node-classification benchmarks. Reason: bias params start random, LoRA starts from refined pretrained weights. **Copy this or the bias parameters will not converge.**
- LoRA rank 32 or 64, alpha = 2r.

Their reported limitation: custom attention biases are incompatible with FlashAttention, so O(N²) attention and roughly 3× slower training than a plain LLM. They ran on a single A100/H100 80GB at ~1,000 token sequences.

Also note their Future Work section explicitly names autoregressive graph generation with `<ADD_EDGE>` control tokens as an open problem. That is our project.

### Depth-Width Tradeoffs — arXiv 2503.01805
Yehudai, Sanford, Bechler-Speicher, Fischer, Gilad-Bachrach, Globerson. Code in supplementary material.

**The precedent for our input format.** They define *node-adjacency tokenization*: the i-th token to the transformer is the i-th row of A. No GNN, no verbalization. Standard PyTorch `TransformerEncoderLayer`. Not a pretrained LLM, and the output is a label or count — never a graph.

Take from it:
- The tokenization itself, plus their practical detail: concatenate node features to the row (width N+F) and zero-pad to the dataset's largest graph.
- **Theorem 4.3** — an O(L)-layer transformer with width O(N) can output, at token i, the i-th row of A^L. Proof that adjacency-row tokens can yield adjacency-like output.
- **Theorem 4.4** — for graphs with max degree d, width O(d·log N) suffices to detect 2-cycles, optimal up to log factors. **Width scales with degree, not node count.** For a sparse graph with degree ~20 and N=1500 that is ~210 dimensions; d_model 4096 clears it comfortably.
- **Theorem 4.2** — worst case, `m·p·H·L = Ω(N)` for adjacency-row tokenization. Check d_model against N.
- Empirically, adjacency rows beat edge lists on all three OGB molecular datasets (ROC-AUC 61.87 vs 54.01 on MOLHIV, 67.63 vs 64.73 on MOLBBBP, 68.64 vs 66.06 on MOLBACE) and beat Laplacian eigenvectors on two of three.
- Efficiency: edge-list needs O(N²) tokens, adjacency rows need O(N).

### PyTorch Geometric
`torch_geometric.nn.GAE`, `VGAE`, `InnerProductDecoder`, `GCNConv`, `GATConv`, `Planetoid`.

Our decoder D1 is `InnerProductDecoder` with a learned W inserted. **Import it, do not reimplement.** GAE and GAT baselines come free.

### Graphormer — Ying et al., NeurIPS 2021
Its `spatial_pos_encoder` is a learnable embedding indexed by shortest-path distance, added to attention logits. That is our hop-bias table. Origin of the mechanism GTLM and GaLA both use.

### GaLA — arXiv 2606.15633 (KDD '26)
Loveland, Trivedi, Weinstein, Huang, Koutra. Frozen LLM, inference-time bias `B(i,j) = 1/shortest_path_distance`, applied only in the first half of layers.

Take one idea: **per-head bias strength `λ_h`**, chosen by a one-time calibration — either an attention-entropy heuristic (no labels) or a single gradient pass (with labels). Forcing the bias onto every head damages the language machinery. Our bias table should be per-head, not shared.

### FPT — `github.com/kzl/universal-computation`
Lu, Grover, Abbeel, Mordatch, arXiv 2103.05247. Frozen language-pretrained GPT-2, new linear input and output layers, layernorm and positional params finetuned, transferred to bit strings, XOR, MNIST, CIFAR and protein folding.

**This is the justification that a frozen text LLM can process a non-text modality entering through a linear projection.** Cite it when asked why this should work at all.

### G2PT — `github.com/tufts-ml/G2PT`
Chen et al., arXiv 2501.01073. Autoregressive graph generation over edge-sequence tokens, transformer trained from scratch. Reference for the autoregressive task (§5.3), not for the architecture.

---

## 10. Suggested repo structure

```
configs/               yaml per experiment
data/
  loaders.py           yields (A_input, A_target, X, mask); domain-agnostic
  masking.py           edge masking, node masking, BFS ordering
models/
  encoders/            e1_linear.py … e7_sparse.py, base.py
  decoders/            d1_inner_product.py … d4_autoregressive.py
  bias/                spd.py, rrwp.py, laplacian.py, per_head.py
  llm_wrapper.py       frozen LLM + bias injection + LoRA + RoPE control
  model.py             assembles encoder + llm + decoder
losses/                weighted_bce.py, degree.py, spectral.py
baselines/             identity.py, gae.py, gat.py, feature_only.py, scratch_transformer.py
train.py
eval.py                AUPRC on masked entries, degree/spectral stats
ablate.py              runs the §8 grid
tests/
```

---

## 11. Pitfalls — read before writing the training loop

### Class imbalance will silently kill this
Cora has ~5,429 edges out of 2,708² ≈ 7.3M possible — about 0.1% positive. Unweighted BCE learns to predict zero everywhere, reports 99.9% accuracy, and has learned nothing.

```python
pos_weight = num_negatives / num_positives
```

Use `pos_weight` or focal loss. **Report AUPRC, not accuracy and not AUROC.** AUROC is misleading at this sparsity.

### Score only what was masked
Never compute the metric over the whole matrix. Evaluate on held-out masked entries plus an equal number of sampled true non-edges. A model that copies its input will look excellent on the full matrix.

### Adjacency rows are local-biased
From the Depth-Width paper's Appendix A: node degree is computable with width 1 from adjacency rows, but **connectivity needs depth Ω(log N) or width Ω(N)**, while Laplacian eigenvectors make it trivial (a graph is disconnected iff its second-smallest Laplacian eigenvalue is 0). Both schemes become universal only when width is Ω(N²) — we are far below that.

**Consequence: implement E5 (row + Laplacian eigenvectors) early.** It costs little and covers the blind spot. Note GTLM independently reached the same conclusion, using RRWP and Magnetic Laplacian alongside SPD.

### Fixed N
E1/E2/E4/E5 tie the model to one N. Acceptable for a fixed node set. E3, E6 and E7 exist to break that dependence — implement at least one of them.

### Permutation
Raw adjacency-row tokenization is **not** permutation invariant on its own; the Depth-Width paper says so explicitly and accepts it for convenience. GTLM's fix — RoPE reset plus bidirectional prefix mask plus structure-derived (not sequence-derived) bias — restores equivariance. Adopt it, and write a unit test: permute the input nodes, check the output permutes identically to within floating-point tolerance.

### Memory
Attention is O(N²) and the custom bias blocks FlashAttention. Cora at N=2,708 is ~7.3M attention entries per head per layer. **Measure this before building anything elaborate.** Fallbacks: sample k-hop subgraphs, cap at 500–1,000 nodes per batch, or use a k-hop attention mask (GTLM lists this as their own suggested fix for quadratic scaling).

### Do not use FlashAttention
It cannot take the custom bias. Use the eager attention path.

---

## 12. Build order

1. Data loader + masking + metrics (AUPRC on masked entries). No model.
2. Baselines from §7, including the from-scratch transformer.
3. E1 + D1 + frozen LLM, **no attention bias yet**, masked edge prediction on Cora. Get it training end to end.
4. Add the attention bias. Measure the delta.
5. Encoder sweep E1–E7.
6. Decoder sweep, add autoregressive.
7. Ablations and controls from §8.
8. Scale up datasets.
9. Swap in biomedical loader.

Every stage: log parameter counts, wall-clock, peak memory, and AUPRC on masked entries.

---

## 13. What NOT to do

- **No GNN encoder.** No message passing, no multi-hop neighbour aggregation as a preprocessing step feeding soft prompts into the LLM. That is the approach we are replacing. E6 aggregates over the immediate neighbour set once — keep it that way and say so explicitly in comments.
- **No verbalization.** The graph never becomes text. No `"node 1 is connected to node 2"` anywhere.
- **No tokenizer, no embedding table.** They are unused. Our encoder produces `[N, d_model]` directly.
- **No unweighted BCE.**
- **No accuracy as a metric.**
- **Do not unfreeze the LLM body.** LoRA only.
