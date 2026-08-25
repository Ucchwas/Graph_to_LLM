# Autoregressive graph generation — research notes for §5.3 and decoder D4

Author: research subagent (topic: `autoregressive`)
Date: 2026-08-25
Status: every claim below is tagged with the source I actually fetched. Where I could not verify
something I say so explicitly. Where CLAUDE.md is wrong I say so bluntly in
[Corrections](#corrections-to-claudemd).

Method note: I did not trust HTML-summarisation for numbers. All paper numbers below were
extracted with `pdftotext -layout` on the arXiv PDF I downloaded myself, and cross-checked
against a second rendering (ar5iv / arxiv HTML) where the PDF column layout was ambiguous.
All code blocks marked **REAL** are verbatim from `raw.githubusercontent.com`.

---

## Verified facts (with source next to each)

### Identifiers — all four arXiv ids in CLAUDE.md that touch this topic resolve

| Claim in CLAUDE.md | Status | Source |
|---|---|---|
| GraphRNN, You/Ying/Ren/Hamilton/Leskovec, ICML 2018, arXiv 1802.08773 | **CONFIRMED** — exact title *"GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models"* | `arxiv.org/abs/1802.08773`, PDF |
| GRAN, Liao et al., NeurIPS 2019, arXiv 1910.00760 | **CONFIRMED** — *"Efficient Graph Generation with Graph Recurrent Attention Networks"*, Liao, Li, Song, Wang, Nash, Hamilton, Duvenaud, Urtasun, Zemel | `arxiv.org/abs/1910.00760`, PDF |
| G2PT, Chen et al., arXiv 2501.01073, `github.com/tufts-ml/G2PT` | **CONFIRMED** — *"Graph Generative Pre-trained Transformer"*, Xiaohui Chen, Yinkai Wang, Jiaxing He, Yuanqi Du, Soha Hassoun, Xiaolin Xu, Li-Ping Liu. v1 2025-01-02, v2 2025-06-03, ICML 2025. Repo exists, 40 files listed via GitHub API. | `arxiv.org/abs/2501.01073`, `api.github.com/repos/tufts-ml/G2PT` |

I also spot-checked every other arXiv id in CLAUDE.md — `2605.10247`, `2606.15633`, `2503.01805`,
`2103.05247`, `2402.05862`, `2310.04560` — all return HTTP 200 with matching titles. **No dead ids
in the spec.** (One naming nit: `2606.15633` is titled *"Formalizing and Mitigating Structural
Distortion in LLM Attention for Graph Reasoning"*, which CLAUDE.md calls "GaLA". Not my topic; flagging only.)

---

### 1. GraphRNN (arXiv 1802.08773)

**The sequence representation.** For graph `G` with `n` nodes under node ordering `π`:

- Eq. (1): `S^π = f_S(G, π) = (S_1^π, …, S_n^π)`, each `S_i^π ∈ {0,1}^{i-1}`.
- Eq. (2): `S_i^π = (A^π_{1,i}, …, A^π_{i-1,i})^T,  i ∈ {2,…,n}`. `S_1` is defined as the empty vector; self-loops prohibited (footnote 3).
- Eq. (3): `p(G) = Σ_S p(S^π) · 1[f_G(S^π) = G]`.
- Eq. (4): **`p(S^π) = ∏_{i=1}^{n+1} p(S_i^π | S_1^π, …, S_{i-1}^π)`**, with `S_{n+1}` = EOS token.
- Eq. (5): `h_i = f_trans(h_{i-1}, S^π_{i-1})` — graph-level RNN (GRU).
- Eq. (6): `θ_i = f_out(h_i)`.
- Eq. (7): `p(S_i^π | S^π_{<i}) = ∏_{j=1}^{i-1} p(S_{i,j}^π | S_{i,<j}^π, S^π_{<i})` — the **full GraphRNN** ("dependent Bernoulli sequence"), each factor from an *edge-level* GRU whose hidden state is initialised from `h_i`.

Source: 1802.08773 PDF, §2.3.1–2.3.3.

**GraphRNN-S** ("simplified"): `p(S_i^π|S^π_{<i})` is a **multivariate Bernoulli**, `f_out` a
single-layer MLP with sigmoid, weights shared across timesteps, edges within a row sampled
independently. Source: 1802.08773 §2.3.3.

**BFS ordering — the exact argument.** Eq. (8): `S^π = f_S(G, BFS(G, π))`. The paper gives *two*
reasons, and CLAUDE.md only knows the second one:

> "Using BFS to specify the node ordering during generation has two essential benefits. The first
> is that we only need to train on all possible BFS orderings, rather than all possible node
> permutations, i.e., multiple node permutations map to the same BFS ordering, providing a
> reduction in the overall number of sequences we need to consider. The second is that the BFS
> ordering makes learning easier by reducing the number of edge predictions we need to make in the
> edge-level RNN; in particular, when we are adding a new node under a BFS ordering, the only
> possible edges for this new node are those connecting to nodes that are in the 'frontier' of the
> BFS (i.e., nodes that are still in the BFS queue)"

(verbatim, 1802.08773 §2.3.4). Footnote 4 is the honest caveat: *"In the worst case (e.g., star
graphs), the number of BFS orderings is n!, but we observe substantial reductions on many
real-world graphs."*

**Proposition 1** (1802.08773 §2.3.4, proof in appendix):

> Suppose `v_1, …, v_n` is a BFS ordering of `n` nodes in graph `G`, and `(v_i, v_{j-1}) ∈ E` but
> `(v_i, v_j) ∉ E` for some `i < j ≤ n`, then `(v_{i'}, v_{j'}) ∉ E, ∀ 1 ≤ i' ≤ i` and `j ≤ j' ≤ n`.

*(pdftotext dropped the `∈/∉/≤` glyphs; the ar5iv rendering gives the last quantifier as
`j ≤ j′ < n`. The `i'`-side quantifier `∀1 ≤ i′ ≤ i` is unambiguous in both renderings. Treat the
final bound as `j ≤ j′ ≤ n` up to an off-by-one in the paper's own typesetting — it does not
change the consequence.)*

**Why this bounds the window, in plain language.** Under BFS, node indices are non-decreasing in
distance from the root. If node `v_i` has already "closed" — its last BFS child `v_{j-1}` has been
emitted and `v_j` is not a neighbour — then no node with index `≤ i` can have any neighbour with
index `≥ j`. Contrapositive: the set of nodes a *new* node can attach to is a contiguous suffix of
the already-emitted nodes, namely the current BFS frontier. Nodes that have fallen off the back of
the frontier are provably unreachable. So the variable-length `S_i ∈ {0,1}^{i-1}` can be replaced
by a **fixed `M`-dimensional** window, Eq. (9):

```
S_i = (A_{max(1, i-M), i}, …, A_{i-1, i})^T ,   i ∈ {2, …, n}
```

**Corollary 1** (verbatim modulo glyphs): with a BFS ordering the maximum number of entries
GraphRNN needs to predict for `S_i`, `1 ≤ i ≤ n`, is

```
O( max_{d=1..diam(G)} |{ v_i : dist(v_i, v_1) = d }| )
```

i.e. **the widest BFS level (frontier width)**, not `n`. Overall GraphRNN time complexity is
therefore **`O(M·n)`** rather than `O(n²)`. "In practice, we estimate an empirical upper bound for
M." Source: 1802.08773 §2.3.4.

**How they actually estimate `M` (REAL code).** They do *not* compute the diameter bound — they
Monte-Carlo it over random permutations + random BFS roots and take the top-`k` largest observed
frontier width (`topk=10` out of `iter=20000` samples):

```python
# REAL — github.com/JiaxuanYou/graph-generation/blob/master/data.py
def calc_max_prev_node(self, iter=20000, topk=10):
    max_prev_node = []
    for i in range(iter):
        adj_idx = np.random.randint(len(self.adj_all))
        adj_copy = self.adj_all[adj_idx].copy()
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        G = nx.from_numpy_matrix(np.asmatrix(adj_copy))
        start_idx = np.random.randint(adj_copy.shape[0])
        x_idx = np.array(bfs_seq(G, start_idx))
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj_flexible(adj_copy.copy())
        max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
        max_prev_node.append(max_encoded_len)
    max_prev_node = sorted(max_prev_node)[-1*topk:]
    return max_prev_node        # caller does max(...)
```

`encode_adj_flexible` is the routine that measures the actual frontier width per row:

```python
# REAL — data.py
def encode_adj_flexible(adj):
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]
    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end - len(adj_slice) + np.amin(non_zero)
    return adj_output
```

**Empirical `M` values they hard-coded** (REAL, `create_graphs.py`) — note these are what
Corollary 1 actually buys you in practice:

| dataset | max &#124;V&#124; | `max_prev_node` (M) | M / N |
|---|---|---|---|
| grid (2-D lattice) | 361 | **40** | 0.11 |
| community4 | ~160 | **80** | 0.5 |
| barabasi (BA, m=4) | 200 | **130** | 0.65 |
| protein (D&D-style) | 500 | **80** | 0.16 |
| DD | 500 | **230** | 0.46 |
| citeseer 3-hop ego | 400 | **250** | 0.63 |
| enzymes | ~125 | **25** | 0.2 |
| tree (balanced) | — | **256** | — |

Read: BFS truncation is a large win on lattice-like graphs (grid: 40 of 361) and a *modest* win
on citation/social ego graphs (citeseer: 250 of 400). Do not assume it will save you on Cora.

**Loss actually used (REAL).** Plain, **unweighted** BCE on sigmoid probabilities:

```python
# REAL — model.py
def binary_cross_entropy_weight(y_pred, y, has_weight=False, weight_length=1, weight_max=10):
    if has_weight:
        ...
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss
```

and in `train_rnn_epoch`: `y_pred = F.sigmoid(y_pred); loss = binary_cross_entropy_weight(y_pred, output_y)`.
Training is **teacher-forced** (ground-truth `S_{i-1}` fed as input at step `i`), packed with
`pack_padded_sequence` so batching is over variable-length graphs.

**GraphRNN hyperparameters (REAL, `args.py`)** — the reference point:

```
hidden_size_rnn            128   (64 for '*_small' datasets: parameter_shrink=2)
hidden_size_rnn_output      16
embedding_size_rnn          64   (32 small)
embedding_size_rnn_output    8
embedding_size_output       64   (32 small)
num_layers                   4
batch_size                  32   ; batch_ratio 32 (1 epoch = 32 batches)
epochs                    3000
lr                       0.003 ; milestones [400, 1000] ; lr_rate (gamma) 0.3
sample_time                  2
GRU init: bias -> constant 0.25 ; weight -> xavier_uniform(gain=calculate_gain('sigmoid'))
Linear init: xavier_uniform(gain=calculate_gain('relu'))
```

**GraphRNN evaluation metrics.** MMD (Eq. 10) over three graph statistics: degree distribution,
clustering-coefficient distribution, and **average 4-node orbit counts** (via ORCA). They use the
first Wasserstein / EMD distance (Eq. 11) inside a Gaussian kernel, and claim (Proposition 2) that
`k_W(p,q) = exp(W(p,q)/(2σ²))` induces a unique RKHS. Exact kernel parameters, from **REAL** code
in `eval/stats.py` + `eval/mmd.py`:

| statistic | descriptor | kernel | σ | other |
|---|---|---|---|---|
| degree | `nx.degree_histogram(G)`, normalised to a pmf | `gaussian_emd` | 1.0 (default) | `distance_scaling=1.0` |
| clustering | `np.histogram(nx.clustering(G).values(), bins=100, range=(0,1))`, normalised | `gaussian_emd` | **0.1** (`1.0/10`) | `distance_scaling=bins=100` |
| orbit | ORCA `node 4` counts, `sum(axis=0)/G.number_of_nodes()` | `gaussian` (plain L2 RBF) | **30.0** | `is_hist=False` |

`compute_mmd(X,Y,k) = disc(X,X,k) + disc(Y,Y,k) − 2·disc(X,Y,k)` (biased V-statistic, not the
unbiased U-statistic).

**GraphRNN reported MMD (Table 1, full datasets, gaussian-EMD kernel):**

| model | Community Deg/Clus/Orbit | Ego Deg/Clus/Orbit | Grid Deg/Clus/Orbit | Protein Deg/Clus/Orbit |
|---|---|---|---|---|
| E-R | 0.021 / 1.243 / 0.049 | 0.508 / 1.288 / 0.232 | 1.011 / 0.018 / 0.900 | 0.145 / 1.779 / 1.135 |
| B-A | 0.268 / 0.322 / 0.047 | 0.275 / 0.973 / 0.095 | 1.860 / 0 / 0.720 | 1.401 / 1.706 / 0.920 |
| Kronecker | 0.259 / 1.685 / 0.069 | 0.108 / 0.975 / 0.052 | 1.074 / 0.008 / 0.080 | 0.084 / 0.441 / 0.288 |
| MMSB | 0.166 / 1.59 / 0.054 | 0.304 / 0.245 / 0.048 | 1.881 / 0.131 / 1.239 | 0.236 / 0.495 / 0.775 |
| **GraphRNN-S** | 0.055 / 0.016 / 0.041 | 0.090 / 0.006 / 0.043 | 0.029 / 1e-5 / 0.011 | 0.057 / 0.102 / 0.037 |
| **GraphRNN** | **0.014 / 0.002 / 0.039** | 0.077 / 0.316 / 0.030 | **1e-5 / 0 / 1e-4** | 0.034 / 0.935 / 0.217 |

Dataset sizes: Community (max &#124;V&#124;=160, &#124;E&#124;=1945), Ego (399, 1071), Grid (361, 684), Protein (500, 1575).

**GraphRNN Table 2** (small datasets, adds NLL — the only likelihood numbers in the paper):

| model | Community-small Deg/Clus/Orbit + train/test NLL | Ego-small Deg/Clus/Orbit + train/test NLL |
|---|---|---|
| GraphVAE | 0.35 / 0.98 / 0.54 · 13.55 / 25.48 | 0.13 / 0.17 / 0.05 · 12.45 / 14.28 |
| DeepGMG | 0.22 / 0.95 / 0.40 · 106.09 / 112.19 | 0.04 / 0.10 / 0.02 · 21.17 / 22.40 |
| GraphRNN-S | 0.02 / 0.15 / 0.01 · 31.24 / 35.94 | 0.002 / 0.05 / 0.0009 · 8.51 / 9.88 |
| GraphRNN | 0.03 / 0.03 / 0.01 · **28.95 / 35.10** | 0.0003 / 0.05 / 0.0009 · **9.05 / 10.61** |

They emphasise the *train↔test NLL gap* as the generalisation measure ("22% smaller average NLL gap").

---

### 2. GRAN (arXiv 1910.00760)

**Block-wise factorisation.** Only the strictly-lower triangle `L` is modelled; `A = L + Lᵀ`.
Block `t` covers row indices `b_t = {B(t−1)+1, …, Bt}`, `T = ⌈N/B⌉`. Eq. (1):

```
p(L^π) = ∏_{t=1}^{T}  p( L^π_{b_t} | L^π_{b_1}, …, L^π_{b_{t-1}} )
```

"This significantly shortens the sequence of auto-regressive graph generation decisions by a factor
of `O(N)`" — i.e. **`O(N²)` decision steps for GraphRNN → `O(N)` for GRAN**. Source: 1910.00760 §2.1–2.2.

**Node representation.** Eq. (2): `h⁰_{b_i} = W L_{b_i} + b, i < t`. The block being generated gets
**`h⁰_{b_t} = 0`** — this is exactly how GRAN prevents the row it is predicting from leaking into its
own conditioning. (Directly relevant to our D4; see §5 below.)

**GNN with attentive messages** (Eqs. 3–6), on an *augmented graph* `G_t` = already-generated
`B(t−1)` nodes + their real edges + `B` new nodes fully connected to each other and to all previous
nodes ("augmented edges"):

```
m^r_ij  = f( h^r_i − h^r_j )                                    (3)
h̃^r_i   = [ h^r_i , x_i ]                                       (4)   x_i = B-dim binary mask:
                                                                       0 for existing nodes,
                                                                       one-of-B for new block nodes
a^r_ij  = Sigmoid( g( h̃^r_i − h̃^r_j ) )                         (5)
h^{r+1}_i = GRU( h^r_i , Σ_{j∈N(i)} a^r_ij · m^r_ij )            (6)
```

`f` and `g` are 2-layer MLPs with ReLU. **GNN hidden states are NOT carried across generation
steps** — that is what makes training parallel (PixelCNN-style) rather than sequential.

**Mixture-of-Bernoullis output** (Eqs. 7–9) — the mechanism that captures within-block edge
correlation without giving up parallel computation:

```
p(L_{b_t} | L_{b_1..b_{t-1}})  =  Σ_{k=1}^{K}  α_k  ∏_{i∈b_t} ∏_{1≤j≤i}  θ_{k,i,j}          (7)
α_1,…,α_K       = Softmax( Σ_{i∈b_t, 1≤j≤i}  MLP_α( h^R_i − h^R_j ) )                       (8)
θ_{1,i,j},…,θ_{K,i,j} = Sigmoid( MLP_θ( h^R_i − h^R_j ) )                                    (9)
```

"When K = 1, the distribution degenerates to Bernoulli which assumes the independence of each
potential edge conditioned on the existing graph. This is a strong assumption and may compromise
the model capacity." "within each mixture component the distribution is fully factorial, and all the
mixture components can be computed in parallel."

**Block size and stride.** Trained with block size `B`, stride `1`; at *test* time stride
`S ∈ [1,B]` can be varied **without retraining** ("strided sampling"). `T = ⌈(N−B)/S⌉ + 1`.

**Ordering marginalisation — the exact derivation.** §2.3, verbatim structure:

```
maximise   log p(G) = log Σ_{π ∈ Π} p(G, π)          (intractable, |Π| = N!)
restrict   Q = {π_i | π_i ∈ Ω, i = 1..M},  Q̃ = {π | u(π) = π̃, π̃ ∈ Q}
Eq. (10)   log p(G)  ≥  log Σ_{π ∈ Q̃} p(G, π)  =  log Σ_{π ∈ Q} p(A^π)
```

with the constraint that no two orderings `π₁, π₂ ∈ Q` give `A^{π₁} = A^{π₂}` (they de-duplicate
graph automorphisms via the surjection `u`). **There is no `−log|Q|` term** — I checked the PDF text
directly. Their words: the RHS "is a valid lower bound of the true log-likelihood, and it is a
tighter bound than any single term `log p(G, π)`, which is the objective for picking a single
arbitrary ordering… increasing the size of Q can make the bound tighter."

Variational interpretation, Eqs. (11)–(12):

```
Eq. (11)   log p(G)  ≥  E_{q(π|G)}[ log p(G,π) ]  +  H( q(π|G) )
Eq. (12)   q*(π|G)   =  p(G,π) / Σ_{π∈Q̃} p(G,π)          (categorical over |Q| items,
                                                            solved analytically w/ Lagrange multipliers)
```

Substituting (12) into (11) recovers (10). "In other words, by optimizing the objective in Eq. 10,
we are implicitly picking the optimal (combination of) node orderings from the set Q̃."

**The canonical orderings.** Five, all "solely based on graph properties":
1. **DFS** tree rooted at the largest-degree node
2. **BFS** tree rooted at the largest-degree node
3. **k-core descending** ("core descending", their novel one)
4. **node degree descending**
5. **default** ordering used by NetworkX

Confirmed in **REAL** code (`dataset/gran_data.py::_get_graph_data`), where `adj_0..adj_5` are
default / degree-descent / degree-ascent / BFS / DFS / k-core and `num_canonical_order == 5`
selects `[adj_0, adj_1, adj_3, adj_4, adj_5]` — note **degree-ascent is built but excluded**.

```python
# REAL — GRAN/dataset/gran_data.py
CGs = [G.subgraph(c) for c in nx.connected_components(G)]
CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)   # big components first
node_list_bfs, node_list_dfs = [], []
for ii in range(len(CGs)):
    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
    degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)
    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])   # root = max-degree node
    dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])
    node_list_bfs += list(bfs_tree.nodes())
    node_list_dfs += list(dfs_tree.nodes())
adj_3 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_bfs))
adj_4 = np.array(nx.to_numpy_matrix(G, nodelist=node_list_dfs))
# k-core: group nodes by core number descending, then by degree descending inside each core
num_core = nx.core_number(G)
core_order_list = sorted(list(set(num_core.values())), reverse=True)
```

**The ordering ablation — the single most important table for us** (1910.00760 Appendix Table 4,
grid graphs, validation set, all with stride 1; `1`=DFS `2`=BFS `3`=k-core `4`=degree-descent `5`=default):

| B | K | Q | Deg. | Clus. | Orbit | Spec. | Runtime (s) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | {1} | 1.51e-5 | 0 | 2.66e-5 | 1.57e-2 | – |
| 1 | 20 | **{1} DFS** | **1.54e-5** | **0** | **4.27e-6** | **1.32e-2** | – |
| 1 | 50 | {1} | 1.70e-5 | 0 | 9.56e-7 | 1.35e-2 | – |
| 1 | 20 | **{2} BFS** | **0.16** | **0.29** | **0.16** | **0.15** | – |
| 1 | 20 | {3} k-core | 1.43e-2 | 3.38e-3 | 1.34e-2 | 3.97e-2 | – |
| 1 | 20 | {4} degree-desc | 2.50e-3 | 1.46e-2 | 5.23e-3 | 1.64e-2 | – |
| 1 | 20 | {5} default | 7.12e-2 | 1.10e-3 | 7.81e-2 | 8.43e-2 | – |
| 1 | 20 | {1,2} | 6.00e-2 | 0.16 | 2.48e-2 | 4.79e-2 | – |
| 1 | 20 | {1,2,3} | 8.99e-3 | 7.37e-3 | 1.69e-2 | 2.84e-2 | – |
| 1 | 20 | {1,2,3,4} | 2.34e-2 | 5.95e-2 | 5.21e-2 | 4.49e-2 | – |
| 1 | 20 | {1,2,3,4,5} (all) | 4.11e-4 | 9.43e-3 | 6.66e-4 | 1.57e-2 | – |
| 4 | 20 | {1} | 1.69e-4 | 0 | 5.04e-4 | 2.02e-2 | – |
| 8 | 20 | {1} | 7.01e-5 | 4.89e-5 | 8.57e-5 | 2.20e-2 | – |
| 16 | 20 | {1} | 1.30e-3 | 6.65e-3 | 9.32e-3 | 2.01e-2 | 33.1 |
| 16(S=4) | 20 | {1} | 1.70e-2 | 0.11 | 1.95e-2 | 2.39e-2 | 8.4 |
| 16(S=8) | 20 | {1} | 8.45e-2 | 0.55 | 2.77e-2 | 2.61e-2 | 4.3 |
| 16(S=16) | 20 | {1} | 5.48e-2 | 0.48 | 1.03e-2 | 2.50e-2 | 2.2 |

Three things jump out, and all three matter to us:

1. **BFS is the *worst* single ordering on grid graphs — by four orders of magnitude vs DFS**
   (Deg. 0.16 vs 1.54e-5). CLAUDE.md's "use BFS canonical ordering" is not a safe default.
2. **Marginalising over more orderings is not free.** `Q={1,2}` is ~4000× *worse* on Deg. than
   `Q={1}`. Even the full 5-ordering set is 27× worse than DFS alone. Their conclusion in §4.4:
   *"We found that using all orderings and DFS ordering alone are similarly good on grid graphs.
   Therefore, we choose to use DFS ordering due to its lower computational cost."* — that is a
   generous reading of their own table; DFS alone is clearly better.
3. **Larger block size is monotonically worse.** B=1 → B=16 costs ~2 orders of magnitude on Deg.
   Block size buys speed, nothing else.

**Their released configs use exactly what the ablation says** — a *single* ordering, B=1, S=1
(REAL, `config/gran_grid.yaml` and `config/gran_DD.yaml`):

```yaml
dataset:
  node_order: DFS              # k_core/BFS/degree_decent
model:
  name: GRANMixtureBernoulli
  num_mix_component: 20
  is_sym: true
  block_size: 1
  sample_stride: 1
  max_num_nodes: 361           # 500 for DD
  hidden_dim: 128              # 512 for DD, 256 for point cloud (paper §4.2)
  embedding_dim: 128
  num_GNN_layers: 7
  num_GNN_prop: 1
  num_canonical_order: 1       # <-- single ordering in the released config!
  dimension_reduce: true
  has_attention: true
  edge_weight: 1.0e+0          # pos_weight for BCEWithLogits; UNWEIGHTED by default
train:
  optimizer: Adam ; lr: 1.0e-4 ; wd: 0 ; lr_decay_epoch: [100000000]  # i.e. no decay
  max_epoch: 3000 (grid) / 50000 (DD) ; batch_size: 1 (grid) / 16 (DD)
```

**GRAN benchmark results (Table 1, gaussian-TV kernel, test set):**

| model | Grid Deg/Clus/Orbit/Spec | Protein Deg/Clus/Orbit/Spec | Point cloud Deg/Clus/Orbit/Spec |
|---|---|---|---|
| Erdős–Rényi | 0.79 / 2.00 / 1.08 / 0.68 | 5.64e-2 / 1.00 / 1.54 / 9.13e-2 | 0.31 / 1.22 / 1.27 / 4.26e-2 |
| GraphVAE* | 7.07e-2 / 7.33e-2 / 0.12 / 1.44e-2 | 0.48 / 7.14e-2 / 0.74 / 0.11 | OOM |
| GraphRNN-S | 0.13 / 3.73e-2 / 0.18 / 0.19 | 4.02e-2 / 4.79e-2 / 0.23 / 0.21 | OOM |
| GraphRNN | 1.12e-2 / 7.73e-5 / 1.03e-3 / 1.18e-2 | 1.06e-2 / 0.14 / 0.88 / 1.88e-2 | OOM |
| **GRAN** | **8.23e-4 / 3.79e-3 / 1.59e-3 / 1.62e-2** | **1.98e-3 / 4.86e-2 / 0.13 / 5.13e-3** | **1.75e-2 / 0.51 / 0.21 / 7.45e-3** |

Datasets: Grid `|V|max=361, |E|max=684, |V|avg≈210, |E|avg≈392`; Protein `500/1575/258/646`;
3-D point cloud (FirstMM-DB, 41 graphs) `5037/10886/1377/3074`.
Two independent extractions (PDF layout + ar5iv HTML) agree on these numbers.

**Speed.** "GraphRNN takes around 9.5 seconds to generate one grid graph on average. Our best
performing model with stride 1 is about 6 times as fast as GraphRNN… With a stride of 16, our
model is more than 80x faster than GraphRNN, but the model quality is also noticeably worse."
(single GTX 1080Ti).

**GRAN changed the MMD kernel.** §4.1 verbatim: *"In practice, we found computing this MMD with the
Gaussian EMD kernel to be very slow for moderately large graphs. Therefore, we use the total
variation (TV) distance, which greatly speeds up the evaluation and is still consistent with EMD."*
They also **added a 4th statistic**: the **normalized-Laplacian eigenvalue histogram**.

```python
# REAL — GRAN/utils/dist_helper.py
def gaussian_tv(x, y, sigma=1.0):
    support_size = max(len(x), len(y))
    x = x.astype(np.float); y = y.astype(np.float)
    if len(x) < len(y):  x = np.hstack((x, [0.0]*(support_size-len(x))))
    elif len(y) < len(x): y = np.hstack((y, [0.0]*(support_size-len(y))))
    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))

# REAL — GRAN/utils/eval_helper.py
def spectral_worker(G):
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf
```

GRAN's active kernel choices: degree → `gaussian_tv` (σ=1); spectral → `gaussian_tv` (σ=1, 200
bins on [−1e-5, 2]); clustering → `gaussian_tv`, σ=0.1, 100 bins on [0,1]; orbit → `gaussian_tv`,
σ=30, `is_hist=False`.

**⚠ GraphRNN numbers and GRAN numbers are computed with DIFFERENT kernels and are not comparable.**
Compare GraphRNN's own Table 1 (grid Deg = 1e-5) with GRAN's Table 1 (GraphRNN grid Deg = 1.12e-2)
— three orders of magnitude apart, same model, same dataset, different kernel + different split +
GRAN's own retraining. Never quote across papers.

**GRAN loss (REAL) — this is the code to adapt for D4:**

```python
# REAL — GRAN/model/gran_mixture_bernoulli.py
pos_weight = torch.ones([1]) * self.edge_weight
self.adj_loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

def mixture_bernoulli_loss(label, log_theta, log_alpha, adj_loss_func,
                           subgraph_idx, subgraph_idx_base, num_canonical_order,
                           sum_order_log_prob=False, return_neg_log_prob=False, reduction="mean"):
  num_subgraph = subgraph_idx_base[-1]
  B = subgraph_idx_base.shape[0] - 1
  C = num_canonical_order
  E = log_theta.shape[0]
  K = log_theta.shape[1]
  assert E % C == 0

  # [E, K] per-edge BCE for every mixture component
  adj_loss = torch.stack([adj_loss_func(log_theta[:, kk], label) for kk in range(K)], dim=1)

  const = torch.zeros(num_subgraph).to(label.device)
  const = const.scatter_add(0, subgraph_idx, torch.ones_like(subgraph_idx).float())

  reduce_adj_loss = torch.zeros(num_subgraph, K).to(label.device)
  reduce_adj_loss = reduce_adj_loss.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), adj_loss)          # sum_j BCE per block

  reduce_log_alpha = torch.zeros(num_subgraph, K).to(label.device)
  reduce_log_alpha = reduce_log_alpha.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), log_alpha)
  reduce_log_alpha = reduce_log_alpha / const.view(-1, 1)            # MEAN of alpha logits
  reduce_log_alpha = F.log_softmax(reduce_log_alpha, -1)

  log_prob = -reduce_adj_loss + reduce_log_alpha
  log_prob = torch.logsumexp(log_prob, dim=1)                        # <-- mixture over K

  # ... scatter into [B*C], then:
  bc_loss = (bc_log_prob / bc_const)                                 # length-NORMALISED
  bc_log_prob = bc_log_prob.reshape(B, C); bc_loss = bc_loss.reshape(B, C)
  if sum_order_log_prob:
    b_log_prob = torch.sum(bc_log_prob, dim=1); b_loss = torch.sum(bc_loss, dim=1)
  else:
    b_log_prob = torch.logsumexp(bc_log_prob, dim=1)                 # <-- Eq. 10 marginalisation
    b_loss     = torch.logsumexp(bc_loss, dim=1)
  b_neg_log_prob = -2*b_log_prob
  b_loss = -b_loss
  ...
```

Two implementation subtleties worth copying (or at least knowing about):
- The mixture weight `α` is obtained by **averaging** the per-edge `log_alpha` over the block
  (`/ const`) before `log_softmax` — not summing, despite Eq. (8) writing a sum.
- They backprop the **length-normalised** `b_loss` (logsumexp of *per-edge-averaged* log-probs),
  while `b_neg_log_prob = −2·b_log_prob` (the un-normalised one) is only reported. The factor 2
  accounts for `A = L + Lᵀ`.

---

### 3. G2PT (arXiv 2501.01073) — ID VERIFIED

**Representation (§3.1).** A graph is a sequence that first lists **all nodes**, then **all edges**:

```
[ v_1^c, v_1^id, …, v_n^c, v_n^id,  a_Δ,  v^id_src, v^id_dst, e_1^c, …, v^id_src, v^id_dst, e_m^c ]
```

- node = tuple `v := (v^c, v^id)`; `v^id ∈ Z+` is the node index, `v^c ∈ {1..K_v}` the node type.
- edge = triple `(v^id_src, v^id_dst, e^c)`.
- `a_Δ` is the single separator token between the node section and the edge section.

**Tokenisation (§3.2), verbatim:**

```
tokenize(v^id) = v^id,                       v^id ∈ {1, …, n_max}
tokenize(v^c)  = v^c + n_max,                v^c  ∈ {1, …, K_v}
tokenize(e^c)  = e^c + n_max + K_v,          e^c  ∈ {1, …, K_e}
tokenize(a_Δ)  = n_max + K_v + K_e + 1
plus special tokens [SOG], [EOG]
```

The design point is stated explicitly in Appendix C.3: *"by introducing node index into the
vocabulary, G2PT easily implements the edge tokenizations and node identifications."* No positional
embedding hack, no separate type embedding — node identity **is** a vocabulary entry.

**Training objective, Eq. (1):**

```
L_pt(s; θ) := −log p_θ(s) = − Σ_{l=1}^{L} log p_θ(s_l | s_{<l})
```

Plain next-token cross-entropy. On legality masking: *"Illegal tokens can be avoided by masking out
their logits in the model output. However, our experiments show that unconstrained logits also
yield superior performance thanks to the learning power of Transformers."* — i.e. **they do not use
constrained decoding.**

**Ordering.** *"Nodes are ordered randomly"*, and `v^id_i = i`. The **edge** order is the one that
matters, and it is *"determined by reversing a degree-based edge-removal process"* (Alg. 1): remove
edges incident to low-degree nodes first; reversing this "constructs a compact, relatively dense
core first, followed by the addition of edges with fewer connections."

**Released code disagrees slightly with the paper's notation** (REAL, `datasets_utils.py`) — the
actual token stream is 3 tokens per node and **4 tokens per edge** (a `<sepg>` marker + src + dst +
type), with separate begin/end markers for the node ("context") and edge ("graph") sections:

```python
# REAL — github.com/tufts-ml/G2PT/blob/main/datasets_utils.py
def to_seq_by_bfs(data, atom_type, bond_type):
    x, edge_index, edge_attr = data['x'], data['edge_index'], data['edge_attr']
    x, edge_index = randperm_node(x, edge_index)          # <-- "nodes are ordered randomly"
    ctx = [['<sepc>', atom_type[node_type.item()], f'IDX_{node_idx}']
           for node_idx, node_type in enumerate(x.argmax(-1))]
    ctx = sum(ctx, [])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    outputs = []
    G = to_networkx(data)
    edges_order_bfs = bfs_with_all_edges(G, 0)            # BFS from node 0 (already random)
    for src, dst in edges_order_bfs:
        ... # look up edge type
        outputs.append(['<sepg>', f'IDX_{src}', f'IDX_{dst}', bond_type[removed_edge_type-1]])
    ctx[0] = '<boc>'; ctx.append('<eoc>')
    outputs = sum(outputs, []); outputs[0] = '<bog>'; outputs.append('<eog>')
    return {"text": [" ".join(ctx + outputs)]}

def to_seq_by_deg(data, atom_type, bond_type):
    ...
    while True:
        node_degrees_t = degree(data_t.edge_index[0], num_nodes=num_nodes)
        if torch.all(node_degrees_t == 0): break
        node_degrees_t[node_degrees_t == 0] = INF                     # INF = 100
        candidate_source_nodes = torch.where(node_degrees_t == node_degrees_t.min())[0]
        selected_source_node_idx = candidate_source_nodes[randint].item()      # random tie-break
        candidate_dest_nodes = data_t.edge_index[1][source_node_mask].unique()
        # pick the destination with minimum degree, random tie-break
        data_tminus1, removed_edge_type = remove_edge_with_attr(data_t, (src, dst))
        outputs.append(['<sepg>', f'IDX_{src}', f'IDX_{dst}', bond_type[removed_edge_type-1]])
        data_t = data_tminus1
    outputs = outputs[::-1]        # <-- REVERSE the removal process
```

Note `bfs_with_all_edges` emits **every** edge, not just tree edges: after a BFS tree edge it also
emits back-edges to already-visited neighbours. That is what makes the representation lossless.

Actual vocabulary (REAL, `tokenizers/planar/vocab.json`, **72 entries**):
`<boc>=0, <eoc>=1, <sepc>=2, <bog>=3, <eog>=4, <sepg>=5, IDX_0=6 … IDX_63=69, NODE=70, EDGE=71`.

**Architecture (Table 7 + REAL `configs/networks/*.py`):**

| | small (~10M) | base (~85M) | large (~300M) |
|---|---|---|---|
| n_layer | 6 | 12 | 24 |
| n_head | 6 | 12 | 16 |
| d_model | 384 | 768 | 1024 |
| dropout | 0.0 | 0.0 | 0.0 |
| bias | False | False | False |

Training (Table 7): AdamW, lr **1e-4**, cosine schedule, weight decay **1e-1**, **300 000
iterations**, batch size 60/60/30, gradient accumulation 8/8/16, grad clip 1.0, warmup 2000 iters.
`configs/default.py`: `batch_size = 60`, `gradient_accumulation_steps = 8`, `ordering = 'bfs'`
(the repo's *default* is BFS, not the degree-based ordering the paper headlines).

**Datasets (Table 6).** `n_max`: QM9 9, MOSES 27, GuacaMol 88, Planar 64, Tree 64, Lobster 100,
SBM 187. Max sequence length: QM9 85, MOSES 207, GuacaMol 614, Planar 737, Tree 383, Lobster 599,
SBM 3950. Training sequences: 9.77M / 142M / 112M / 12.8M / 10M / 12.8M / 12.8M. *(Table 6's column
alignment in the PDF is broken; I reconstructed it and sanity-checked it against `n_max`, the
`3·|E|` token estimate, and the repo's actual vocab files. The vocab size row disagrees with the
repo by 1 for planar (73 vs 72) and the repo config uses `block_size = 918` where the paper says
737 — treat these two rows as **approximate**, everything else as solid.)*

**Generic-graph results (Table 2, extracted from arXiv HTML — clean):**

| model | Planar Deg/Clus/Orbit/Spec/Wav/**VUN↑** | Tree | Lobster | SBM |
|---|---|---|---|---|
| GRAN | 7e-4 / 4.3e-2 / 9e-4 / 7.5e-3 / 1.9e-3 / **0** | 1.9e-1 / 8e-3 / 2e-2 / 2.8e-1 / 3.3e-1 / **0** | 3.8e-2 / 0 / 1e-3 / 2.7e-2 / – / – | 1.1e-2 / 5.5e-2 / 5.4e-2 / 5.4e-3 / 2.1e-2 / **25** |
| BiGG | 7e-4 / 5.7e-2 / 3.7e-2 / 1.1e-2 / 5.2e-3 / **5** | 1.4e-3 / 0 / 0 / 1.2e-2 / 5.8e-3 / **75** | 0 / 0 / 0 / 9e-3 / – / – | 1.2e-3 / 6.0e-2 / 6.7e-2 / 5.9e-3 / 3.7e-2 / **10** |
| DiGress | 7e-4 / 7.8e-2 / 7.9e-3 / 9.8e-3 / 3.1e-3 / **77.5** | 2e-4 / 0 / 0 / 1.1e-2 / 4.3e-3 / **90** | 2.1e-2 / 0 / 4e-3 / – / – / – | 1.8e-3 / 4.9e-2 / 4.2e-2 / 4.5e-3 / 1.4e-3 / **60** |
| HSpectre | 5e-4 / 6.3e-2 / 1.7e-3 / 7.5e-3 / 1.3e-3 / **95** | 1e-4 / 0 / 0 / 1.2e-2 / 4.7e-3 / **100** | – | 1.2e-2 / 5.2e-2 / 6.7e-2 / 6.7e-3 / 2.2e-2 / **45** |
| DeFoG | 5e-4 / 5e-2 / 6e-4 / 7.2e-3 / 1.4e-3 / **99.5** | 2e-4 / 0 / 0 / 1.1e-2 / 4.6e-3 / **96.5** | – | 6e-4 / 5.2e-2 / 5.6e-2 / 5.4e-3 / 8e-3 / **90** |
| **G2PT_small** | 4.7e-3 / 2.4e-3 / 0 / 1.6e-2 / 1.4e-2 / **95** | 2e-3 / 0 / 0 / 7.4e-3 / 3.9e-3 / **99** | 2e-3 / 0 / 0 / 5e-3 / 8.5e-3 / **100** | 3.5e-3 / 1.2e-2 / 7e-4 / 7.6e-3 / 9.8e-3 / **100** |
| **G2PT_base** | 1.8e-3 / 4.7e-3 / 0 / 8.1e-3 / 5.1e-3 / **100** | 4.3e-3 / 0 / 1e-4 / 7.3e-3 / 5.7e-3 / **99** | 1e-3 / 0 / 0 / 4e-3 / 1e-2 / **100** | 4.2e-3 / 5.3e-3 / 3e-4 / 6.1e-3 / 6.9e-3 / **100** |

Headline: **GRAN scores V.U.N. = 0 on Planar and Tree** despite competitive MMDs. That is the single
best argument in the literature that MMD alone is not a sufficient evaluation — a model can match
degree/clustering/orbit statistics and still never produce a valid instance of the target class.

**Edge-sequence vs adjacency-matrix ablation (Table 3, planar):**

| representation | #tokens ↓ | Deg. | Clus. | Orbit | Spec. | Wavelet | V.U.N. ↑ |
|---|---|---|---|---|---|---|---|
| `A` (flattened strictly-lower triangle, BFS-permuted, vocab size 4) | **2018** | 8.6e-3 | 1e-1 | 8e-3 | 3.2e-2 | 6.1e-2 | **94** |
| `s` (G2PT edge sequence) | **737** | 4.7e-3 | 2.4e-3 | 0.00 | 1.6e-2 | 1.4e-2 | **95** |

**Honest reading:** the paper's prose claims *"the model trained with adjacency matrices struggles
to capture the topological rule of planar graphs"*, but V.U.N. is **94 vs 95** — statistically
indistinguishable. The real, defensible win is **2.7× shorter sequences** and 2–40× better MMDs on
clustering/wavelet. This matters to us directly: **the adjacency-row representation is not the
bottleneck the G2PT prose implies.**

**Edge-ordering sensitivity (Table 9, MOSES) — the strongest single-paper evidence that ordering matters:**

| model | edge ordering | Validity | Unique. | Novelty | Filters | FCD ↓ | SNN | Scaf |
|---|---|---|---|---|---|---|---|---|
| G2PT_small | Degree-based | 95.1 | 100 | 91.7 | 97.4 | 1.1 | 0.52 | 5.0 |
| | DFS | 91.6 | 100 | 87.1 | 98.0 | 1.2 | 0.55 | 8.9 |
| | **BFS** | **96.2** | 100 | 86.8 | 98.3 | **1.0** | 0.55 | 10.6 |
| | **Uniform (random)** | **62.9** | 100 | 99.4 | **52.0** | **7.0** | 0.38 | 9.5 |
| G2PT_base | Degree-based | 96.4 | 100 | 86.0 | 98.3 | 0.97 | 0.55 | 3.3 |
| | DFS | 91.9 | 100 | 83.7 | 98.1 | 1.13 | 0.55 | 7.5 |
| | **BFS** | **96.9** | 100 | 84.6 | 98.7 | 0.98 | 0.55 | 11.1 |
| | **Uniform (random)** | **80.9** | 100 | 97.0 | **83.9** | **2.14** | 0.46 | 10.3 |

Random ordering costs **33 validity points** at small scale and **16** at base scale, and 7× on FCD.
Their conclusion: *"This result highlights the importance of choosing the right ordering families
for generating sequences."* Note that **scaling the model partially compensates** for a bad ordering
(62.9 → 80.9 going small → base) but does not fix it.

---

### 4. The node-ordering problem — state of the art and how to defend it

#### 4a. The problem statement
`p(G) = Σ_{π∈Π} p(G, π)` with `|Π| = N!`. Picking one `π` is not MLE. Worse, orderings with
non-trivial automorphisms double-count: multiple orderings give the same adjacency matrix
(GRAN §2.1 handles this with the surjection `u`; GraphRNN's footnote-4 star-graph example is the
same issue).

#### 4b. Ordering choice is empirically decisive AND dataset-dependent — with hard numbers

| source | dataset | best ordering | worst ordering | gap |
|---|---|---|---|---|
| GRAN Table 4 | grid | **DFS** (Deg 1.54e-5) | **BFS** (Deg 0.16) | **10⁴×** |
| G2PT Table 9 | MOSES (molecules) | **BFS** ≈ degree-based (Val 96.9) | uniform random (Val 80.9) | 16 pts validity |
| G2PT Table 9 | MOSES | BFS (96.9) | **DFS** (91.9) | 5 pts |
| Order Matters Table 2 | Cora random-walk subgraphs, GraphRNN | uniform (Deg 0.188) | **BFS** (Deg 1.125) | **6×, BFS is worse** |
| Order Matters Table 2 | Community-small, GraphRNN | BFS (0.034) | uniform (0.096) | 2.8×, BFS is better |

**There is no ordering that wins everywhere.** DFS wins on grids, BFS wins on molecules, uniform
beats BFS on Cora subgraphs. Any paper that hard-codes one ordering without a sweep is inviting the
obvious reviewer question.

#### 4c. Defence 1 — marginalise over a fixed family (GRAN)
Eq. (10) above. Cheap (one forward per ordering, logsumexp at the end), valid lower bound, tighter
than any single term. **But GRAN's own Table 4 shows it can be worse in practice than the best
single ordering** — averaging in bad orderings dilutes the good one. Use it as a *robustness*
argument, not a *performance* argument.

#### 4d. Defence 2 — learn the ordering posterior (Order Matters, arXiv 2106.06189)
Chen, Han, Hu, Ruiz, Liu, ICML 2021. (Same first author + senior author as G2PT — Tufts.)
Code: `github.com/tufts-ml/graph-generation-vi`. Verified via PDF.

ELBO, Eq. (12): `L(θ, φ, G) = E_{q_φ(π|G)}[ log p_θ(G, π) − log q_φ(π|G) ]`.
`q_φ` is autoregressive over the ordering, Eq. (13): `q(π|G) = ∏_t q(π_t | G, π_{1:(t-1)})`, each
factor a categorical over unselected nodes, parameterised by a **GAT (3 layers, 6 heads, residual)**
run as a recurrent unit with a positional encoding marking already-selected nodes, Eq. (15):
`h_j^t = h^0 + PE(t')` for `j = π_{t'}`, `t' < t`. Gradients: reparameterisation for θ, **score-function
(REINFORCE) estimator** for φ, `S = 4` samples suffices.

Their Table 1 (test log-likelihood / ELBO), showing how much a learned ordering distribution buys
over uniform:

| model | Community-small | Citeseer-small | Enzymes | Lung | Yeast | Cora |
|---|---|---|---|---|---|---|
| DeepGMG uniform | −206.2/−303.9 | −60.9/−67 | −281.9/−290.8 | −146.7/−225.7 | −115.1/−128.9 | −283.7/−295.2 |
| DeepGMG **VI** | −124.8/−131.8 | −59.6/−65.6 | −145.8/−156.2 | −146.1/−224.6 | −105.4/−115.7 | −227/−247.2 |
| GraphRNN uniform | −154.6/−157.6 | −101.9/−105.7 | −340.3/−349.1 | −232.4/−242.2 | −189.3/−200.1 | −380.6/−401.8 |
| GraphRNN **VI** | **−53.7**/−59.9 | −89.6/−93.2 | −274.9/−282.8 | −155.9/−175.8 | −109.1/−133.7 | −345.3/−358.3 |
| GraphGEN DFS | −263.74/NA | −73.0/NA | −574.2/NA | −140.1/NA | −66.46/NA | −199.5/NA |
| GraphGEN **VI** | **−26.6**/−35.0 | −64.3/−71.1 | −189.7/−213.8 | −117.3/−125.5 | −64.98/−72.39 | −143.6/−152.3 |

and Table 2 (MMD Deg/Clus/Orbit) for GraphRNN, the **direct BFS-vs-uniform-vs-VI comparison**:

| dataset | BFS | uniform | VI |
|---|---|---|---|
| Community-small | 0.034 / 0.11 / 0.009 | 0.096 / 0.091 / 0.021 | **0.018 / 0.01 / 0.008** |
| Citeseer-small | 0.016 / 0.05 / 0.004 | **0.009** / 0.09 / 0.003 | 0.08 / 0.05 / **0.002** |
| Enzymes | 0.03 / 0.085 / 0.043 | 0.042 / 0.104 / 0.074 | **0.015 / 0.067 / 0.02** |
| Lung | 0.103 / 0.301 / 0.043 | 1.213 / **0.002** / 0.081 | **0.074** / 0.060 / **0.004** |
| Yeast | 0.512 / 0.153 / 0.026 | 0.746 / 0.351 / 0.070 | **0.097 / 0.092 / 0.005** |
| **Cora** | **1.125 / 1.002 / 0.427** | 0.188 / 0.206 / 0.200 | **0.066 / 0.171 / 0.052** |

⚠ **Their "Cora" is 400 random-walk subgraphs with 9 ≤ |V| ≤ 97**, *not* the 2708-node Cora in
CLAUDE.md §6. Do not quote these numbers as full-Cora baselines.

#### 4e. Defence 3 — sample random orderings and let scale absorb it (AutoGraph, NeurIPS 2025)
*"Flatten Graphs as Sequences: Transformers are Scalable Graph Generators"*, Dexiong Chen, Markus
Krimmel, Karsten Borgwardt, **arXiv 2502.02216** (v3, 2025-12-08), NeurIPS 2025.
Code: `github.com/BorgwardtLab/AutoGraph`. Verified by downloading the PDF.

Idea: flatten a graph into a **segmented Eulerian neighbourhood trail (SENT)** — a random,
reversible token sequence. Key theorem (Thm 2.15): a SENT is causal + Hamiltonian iff every tuple
`w := (v, A_v)` satisfies `A_v = N_G(v) ∩ V_s(w)` — i.e. you traverse and at each node emit exactly
its already-visited neighbours. Consequence (Thm 2.14): **every prefix of a subgraph-induced SENT
generates an induced subgraph** — the direct analogue of "a prefix of a sentence is a sentence".
Complexity: sequence length and sampling time are `O(m)` (edges), optimal for sparse graphs. Claimed
"up to 100x faster generation and 3x faster training than leading diffusion models."

For us the takeaway is the *strategy*: rather than committing to a canonical order, sample a fresh
random valid order every epoch as **data augmentation**, which is a Monte-Carlo estimate of the
same marginalisation GRAN does exhaustively over a small `Q`. G2PT does a weaker version of this
(random node relabelling + fixed edge ordering), and their §3.2 explicitly justifies it: providing
more sequences from the same `G` reduces the conditional KL term in
`L(θ) + H[p_d(G)] = −KL(p_d(G)‖p_θ(G)) ≥ −KL(p_d(G)‖p_θ(G)) − E_{p_d} KL(p_d(s|G)‖p_θ(s|G))`.

#### 4f. Defence 4 — sidestep ordering entirely
- Permutation-**equivariant** decoders (our D1/D2/D3): no ordering to choose, which is why
  CLAUDE.md is right to make masked edge prediction (§5.1) the primary task.
- Diffusion (DiGress, DeFoG) and iterative local expansion (HSpectre): permutation-equivariant by
  construction, and they dominate Planar/SBM V.U.N. in G2PT's Table 2 above.
- BiGG (Dai, Nazi, Li, Dai, Schuurmans, **arXiv 2006.15502**, ICML 2020) — verified. Recursive
  binary-tree edge decomposition, still ordered but `O((n+m) log n)` generation, `O(log n)`
  synchronisation stages during training, `O(m log n)` training memory. This is the escape hatch if
  N gets large.

#### 4g. What reviewers actually attack (and the pre-emptive answer for each)

| attack | pre-emptive defence |
|---|---|
| "Your likelihood is not a likelihood — you fixed `π`." | State it as a lower bound (GRAN Eq. 10). Report `log Σ_{π∈Q} p(A^π)` over ≥2 orderings. |
| "Why BFS?" | Sweep {BFS, DFS, degree-desc, k-core, random} and report the table. GRAN Table 4 is your justification for why the sweep is mandatory. |
| "Your metric is MMD with an indefinite kernel." | See §6 — report RBF alongside TV, and report PGD. |
| "Your ordering leaks the labels." | **See the leakage warning in §5 — this is the one that would actually sink us.** |
| "You never compare against a permutation-equivariant model." | We already have D1 (§5.1) as the in-house control. Say so. |
| "You report MMD but never validity." | Report V.U.N. for any dataset with a checkable predicate. GRAN scores 0 V.U.N. on Planar — that is what MMD hides. |

---

## Corrections to CLAUDE.md

### C1 — "**use BFS canonical ordering**" (§4, D4) is not defensible as a default. `confidence: certain`
CLAUDE.md D4: *"Requires a node ordering — use BFS canonical ordering and record the choice."*
GRAN's own Appendix Table 4 puts BFS **last** among five single orderings on grid graphs
(Deg MMD 0.16 vs DFS 1.54e-5 — four orders of magnitude), and *Order Matters* Table 2 shows BFS
worse than **uniform random** on Cora subgraphs for GraphRNN (Deg 1.125 vs 0.188).
**Fix:** make the ordering a first-class config field with at least {BFS, DFS, degree-descending,
k-core, random} implemented, sweep it, and report the whole table. Copy GRAN's
`_get_graph_data` verbatim — it computes all five in ~40 lines.

### C2 — §5.3's loss omits the `M` truncation that is GraphRNN's actual contribution. `confidence: certain`
CLAUDE.md writes `L = −Σ_t Σ_{j<t}[…]`, which is `O(N²)` terms — that is the *pre*-BFS GraphRNN
(their Eq. 2/4/7). GraphRNN's Corollary 1 replaces it with an `M`-wide band (Eq. 9), `O(M·N)` terms.
The spec cites GraphRNN as the precedent but drops the one thing GraphRNN is famous for. For Cora
(`N=2708`) the difference is 3.67M vs ~646k loss terms per graph, plus a ~5.7× better positive class
balance. **Fix:** implement `max_prev_node` as an optional band and default it to `None` on small
graphs, `M` on large ones. Estimate `M` with GraphRNN's `calc_max_prev_node` procedure (code above).

### C3 — §5.3 is architecturally incompatible with §2 as written. `confidence: certain`
CLAUDE.md §2 specifies *"RoPE disabled / reset per node"* and *"bidirectional (non-causal) mask over
graph tokens."* Both are correct for §5.1/§5.2 and **both break §5.3**:
- With a bidirectional mask, `h_t` sees every node including `t+1…N`, so predicting row `t` is
  trivial and the autoregressive loss is meaningless.
- With RoPE reset per node there is no inter-node positional signal, so the model cannot know
  *which* step it is on — precisely the information an ordered factorisation depends on.

Also, GTLM's permutation-equivariance property (which the spec adopts from §9 and asks for a unit
test in §11) **is deliberately given up** under a causal mask. That is fine — the task defines an
order — but the unit test must be conditioned on `attn_mode`.
**Fix:** `llm_wrapper.py` needs `attn_mode ∈ {'bidirectional', 'causal'}` and
`inter_node_pos ∈ {'none', 'rope', 'learned'}`, wired from the task config. Add a test asserting
`grad(logits[t], tokens[t':]) == 0` for `t' > t` under `attn_mode='causal'`.

### C4 — the BFS/DFS ordering is computed from `A_true`, which leaks held-out edges. `confidence: certain`
Not stated anywhere in CLAUDE.md, and it is the most likely way this project produces a
too-good-to-be-true number. If we hide 15% of edges and then compute a BFS ordering on the
**complete** graph, the *ordering itself* is a function of the hidden edges. Node `t`'s position in
the sequence tells the model about its hidden neighbours. §11's "Score only what was masked" rule
does not catch this because the leak is in the permutation, not in the input tensor.
**Fix (pick one, and write it in the code comment):**
1. Compute the ordering from `A_observed` only (the masked adjacency), or
2. use an ordering independent of edges (random, or feature/ID-derived), or
3. for §5.3 specifically, do *not* mask — the AR task hides the future by construction, so use the
   complete graph and let causality do the hiding. **This is the clean answer for §5.3.**
Add a unit test: two graphs differing only in held-out edges must produce the same ordering.

### C5 — "**No unweighted BCE**" (§13) is stated as universal but GraphRNN and GRAN both use it. `confidence: certain`
Verified: GraphRNN `binary_cross_entropy_weight(..., has_weight=False)` → `F.binary_cross_entropy`;
GRAN `edge_weight: 1.0e+0` in every released config → `pos_weight = 1`.
This is not a contradiction — they score inside an `M`-band / block where the positive rate is far
higher — but the rule as written would make us diverge from the reference implementations we are
benchmarking against. **Fix:** phrase it as *"weighted BCE for §5.1/§5.2 on the full matrix;
`pos_weight` for §5.3 computed over the actual scored index set (band or full triangle), and report
what it was."* For Cora full triangle: `pos_weight = (2708·2707/2 − 5429)/5429 ≈ 674`. Banded at
M=250: `≈ 118`.

### C6 — §5.3 says "benchmark it against 5.1" but the two tasks are not directly comparable as specced. `confidence: likely`
§5.1 scores AUPRC on a random 15% of entries drawn from the whole matrix. §5.3 scores every
strictly-lower-triangular entry under an ordering. The prior over which entries are scored is
different, and §5.3's model has strictly less information (only nodes `<t`). A naive AUPRC
comparison will make §5.3 look worse for reasons that have nothing to do with the model.
**Fix:** define one shared evaluation index set — e.g. score both tasks on the same held-out 15%
of *strictly-lower-triangular* entries, with §5.3 evaluated in teacher-forced mode so both see the
same conditioning budget. Report the AR NLL separately as its own number.

### C7 — §10's `eval.py` ("AUPRC on masked entries, degree/spectral stats") is insufficient for §5.3. `confidence: certain`
To be comparable to GraphRNN/GRAN/G2PT we need MMD over **four** statistics (degree, clustering,
4-node orbit, normalized-Laplacian spectrum) with **specified kernels**, plus V.U.N. where a
validity predicate exists. Also note the spec's `L_spectral` auxiliary loss (MSE on top-k Laplacian
eigenvalues of `Â` vs `A`) is a *different object* from GRAN's spectral MMD (histogram of the
**normalized** Laplacian spectrum over 200 bins on `[−1e-5, 2]`, compared with a TV kernel). Both
are worth having; do not conflate them in the results table.

### C8 — G2PT is cited as "reference for the autoregressive task" but its representation is orthogonal to ours. `confidence: certain`
G2PT tokenises **edge lists**; we tokenise **adjacency rows** (per §9, Depth-Width). G2PT's own
Table 3 compares the two on planar graphs and finds V.U.N. 95 (edge seq) vs **94** (adjacency
matrix) — i.e. essentially tied on validity, with the edge sequence winning on token count (737 vs
2018) and on clustering/wavelet MMD. **G2PT is not evidence that our representation is wrong.** It
is evidence that sequence length is the axis to worry about. Worth quoting in our related-work
section as a *point in our favour*, not against.

### C9 — Minor: the G2PT paper's own prose overstates its Table 3. `confidence: certain`
Paper text: *"the model trained with adjacency matrices struggles to capture the topological rule of
planar graphs."* Table 3: V.U.N. 94 vs 95. Do not repeat that sentence as if it were a result.

### C10 — Unverifiable / not checked
- I did **not** verify the GTLM (`2605.10247`) or GaLA (`2606.15633`) technical claims in §9 —
  outside my topic. I only confirmed the ids resolve and the titles match (except the GaLA nickname).
- I could **not** verify GraphRNN's Table 1 column alignment beyond what the PDF layout gives; the
  12 numbers per row parse cleanly into 4 datasets × 3 metrics and are self-consistent, so I
  consider them solid, but they were not cross-checked against a second rendering the way GRAN's
  Table 1 was.
- G2PT Table 6's "Vocabulary Size" and "Max Sequence Length" rows disagree with the repo by small
  amounts (planar: 73 vs 72 tokens; 737 vs `block_size = 918`). Flagged, not resolved.

---

## Mechanism / math (exact formulas, tensor shapes)

### The task, precisely
Fix an ordering `π: {1..N} → V`. Let `A^π` be the permuted adjacency, `L = tril(A^π, −1)`.

```
L_AR(θ)  =  − Σ_{t=2}^{N}  Σ_{j ∈ P(t)}  [ A^π_{t,j} log σ(z_{t,j}) + (1 − A^π_{t,j}) log(1 − σ(z_{t,j})) ]

P(t) = {1, …, t−1}                       full  (CLAUDE.md §5.3 as written), |Σ| = N(N−1)/2
P(t) = {max(1, t−M), …, t−1}             band  (GraphRNN Eq. 9),            |Σ| = MN − M²/2
```

with the **hard causality constraint**

```
z_{t,·}  =  f( A^π_{1:t−1, 1:t−1} )        and NOTHING about row t.
```

### Shapes through the pipeline

```
A            [B, N, N]  float, 0/1, symmetric
order        [B, N]     long, the canonical permutation
A_ord        [B, N, N]  = A[order][:, order]
L            [B, N, N]  = tril(A_ord, -1)
X_in         [B, N, N]  = right-shift(L, 1)      # position t carries row t-1
tokens       [B, N, d]  = encoder(X_in, X_feat)  # E1..E7, unchanged
H            [B, N, d]  = LLM(tokens, attn_mode='causal')
Q, K         [B, N, r]  = D4.q(H), D4.k(H)       # r = 32..128 (matches D2)
Z            [B, N, N]  = Q @ Kᵀ / sqrt(r)       # logits; only j < t is scored
pair_mask    [B, N, N]  bool, strict lower triangle ∧ band ∧ padding
```

Memory: `Z` at `N=2708`, `B=1`, fp32 is **29.3 MB** forward; with autograd budget ~3× that. `B=8` →
235 MB forward, ~700 MB with activations. Feasible on 80 GB but chunk if `N > 4000` (see code below).

### Why the right-shift is the correct leakage fix
Under a standard causal mask, position `t` attends to positions `0..t` inclusive. With
`X_in[t] = L[t−1]`, the set of rows visible at position `t` is `{L[0], …, L[t−1]}`, which is exactly
the edge set of the induced subgraph on nodes `1..t−1`. Row `t` is not in that set. This is
GraphRNN Eq. (5) (`h_i = f_trans(h_{i−1}, S_{i−1})`) transposed into transformer form, and it is the
same trick GRAN uses when it sets `h⁰_{b_t} = 0` for the block being generated.

The alternative — a *strictly* causal mask that excludes the diagonal — is worse in practice:
position 0 would attend to nothing and softmax produces NaN, and most fused attention kernels do not
expose that mask shape. Use the shift.

### Positive-class balance, computed

| scoring set | #pairs (Cora, N=2708) | #positives | `pos_weight` |
|---|---|---|---|
| full matrix `N²` | 7,333,264 | 10,858 (directed) | 674 |
| strict lower triangle | 3,665,278 | 5,429 | **674.1** |
| band, M=250 | 645,625 | ≤5,429 | **≥118** |
| band, M=40 (grid-like) | 107,500 | — | — |

(Band count = `Σ_{t=1}^{N} min(t−1, M)` = `M·N − M(M+1)/2` for `M < N`. The positive count in the
band is `≤ |E|` and equals `|E|` only if the ordering keeps every edge inside the window — which is
exactly what GraphRNN's Corollary 1 asserts for BFS and what `calc_max_prev_node` measures.)

### Multi-ordering marginalisation, our version of GRAN Eq. (10)

```
per_order_nll[b, c]  =  − log p( A^{π_c} )                       [B, C]
loss                 =  − mean_b  logsumexp_c ( − per_order_nll[b, c] )
```

Cost: `C` forward passes (or one forward with the batch dimension expanded `C×`). GRAN's code
applies logsumexp to the *length-normalised* per-order loss and backprops that; the unnormalised
version is only reported. Match them if you want comparable NLL numbers.

### Sampling cost
Generation is `N` sequential forward passes (`O(N)` decision steps, matching GRAN; GraphRNN is
`O(N²)` because of the inner edge-level RNN). With a KV cache each step is `O(N·d)` incremental
attention, so total sampling is `O(N²·d)`. For `N=2708` on a frozen 1B LLM this is ~2700 decode
steps per graph — measure it before promising it in the paper.

---

## Code we can reuse (real snippets, real signatures)

### R1 — Node orderings (adapt GRAN's `_get_graph_data`, which computes all five)
Verbatim source above in §2. Signature for our repo:

```python
# data/masking.py
def canonical_orders(G: nx.Graph, which=('bfs','dfs','degree_desc','k_core','default')) -> dict[str, np.ndarray]:
    """Returns {name: node_list}.  BFS/DFS are rooted at the largest-degree node of each
    connected component, components ordered large->small (GRAN, dataset/gran_data.py)."""
```

### R2 — BFS sequence + M-band encoding (GraphRNN, REAL)

```python
# REAL — github.com/JiaxuanYou/graph-generation/blob/master/data.py
def bfs_seq(G, start_id):
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]; output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output

def encode_adj(adj, max_prev_node=10, is_full=False):
    if is_full: max_prev_node = adj.shape[0]-1
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start  = max(0, i - max_prev_node + 1)
        input_end    = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end   = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i, :] = adj_output[i, :][::-1]     # NOTE: reversed -> col 0 is the NEAREST predecessor
    return adj_output
```

The final `[::-1]` matters: GraphRNN's `M`-vector is indexed from the **nearest** predecessor
outwards, which is what makes a fixed-width head meaningful across different `t`. `decode_adj` is
the inverse and is in the same file.

Note the sampler's teacher-forcing layout (REAL, `Graph_sequence_sampler_pytorch.__getitem__`):
`x_batch[0,:] = 1` (a SOS row of all ones), `x_batch[1:L+1] = adj_encoded`, `y_batch[0:L] = adj_encoded`
— i.e. **input is the target shifted right by one**, exactly the shift described above. It also
re-randomises the permutation and the BFS root **on every `__getitem__` call**, which is a cheap
form of ordering augmentation we should copy.

### R3 — D4 autoregressive head (ours; new)

```python
# models/decoders/d4_autoregressive.py
import torch, torch.nn as nn, torch.nn.functional as F


class D4Autoregressive(nn.Module):
    """Autoregressive edge head.  At node-step t predict A[t, j] for all j in P(t).

    Reuses the D1/D2 bilinear form so the decoder sweep is apples-to-apples:
        z_{t,j} = <W_q h_t, W_k h_j> / sqrt(r) + b
    h_j is causally valid at step t because j < t and h_j depends only on nodes < j.

    IMPORTANT: this module is causal ONLY IF the caller
      (a) right-shifts the encoder inputs by one row, and
      (b) runs the LLM body with attn_mode='causal'.
    See build_ar_inputs() and the test in tests/test_d4_causality.py.
    """

    def __init__(self, d_model: int, rank: int = 128, num_mix: int = 1, use_bias: bool = True):
        super().__init__()
        self.rank, self.num_mix = rank, num_mix
        if num_mix == 1:
            self.q = nn.Linear(d_model, rank, bias=False)
            self.k = nn.Linear(d_model, rank, bias=False)
        else:                                   # GRAN Eqs. (7)-(9): K-component mixture
            self.q = nn.Linear(d_model, rank * num_mix, bias=False)
            self.k = nn.Linear(d_model, rank * num_mix, bias=False)
            self.alpha = nn.Sequential(         # MLP_alpha on (h_t - h_j)
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, num_mix))
        self.scale = rank ** -0.5
        self.bias = nn.Parameter(torch.zeros(())) if use_bias else None

    def forward(self, H):                       # H: [B, N, d_model]
        B, N, _ = H.shape
        if self.num_mix == 1:
            Q, K = self.q(H), self.k(H)                              # [B,N,r]
            Z = torch.einsum('btr,bjr->btj', Q, K) * self.scale      # [B,N,N]
            if self.bias is not None:
                Z = Z + self.bias
            return Z                                                 # logits
        Q = self.q(H).view(B, N, self.num_mix, self.rank)
        K = self.k(H).view(B, N, self.num_mix, self.rank)
        log_theta = torch.einsum('btkr,bjkr->btjk', Q, K) * self.scale   # [B,N,N,K]
        if self.bias is not None:
            log_theta = log_theta + self.bias
        # alpha logits per pair, from the difference feature (GRAN uses h_i - h_j)
        diff = H.unsqueeze(2) - H.unsqueeze(1)                           # [B,N,N,d]  (O(N^2 d)! )
        log_alpha = self.alpha(diff)                                     # [B,N,N,K]
        return log_theta, log_alpha
```

⚠ The `num_mix > 1` path materialises `[B, N, N, d]` — **only usable for small `N`** (molecules,
`N ≲ 100`). For Cora use `num_mix = 1`, or replace the difference feature with the bilinear form
`alpha_{t,j} = <W_a h_t, W_b h_j>` which is `O(N²K)` instead of `O(N²d)`.

### R4 — Input construction and the causal shift

```python
# data/masking.py
def build_ar_inputs(A, order):
    """
    A     [N, N] symmetric float 0/1 (the COMPLETE graph; see leakage note C4)
    order [N]    LongTensor canonical ordering
    returns (A_ord, X_in) where
      A_ord [N,N] : A permuted into `order`  (the loss target)
      X_in  [N,N] : encoder input; row t carries L[t-1], i.e. shifted right by one.
                    Row 0 is all-zeros = the "start of graph" token.
    """
    A_ord = A[order][:, order]
    L = torch.tril(A_ord, diagonal=-1)          # row i holds only edges to j < i
    X_in = torch.zeros_like(L)
    X_in[1:] = L[:-1]
    return A_ord, X_in


def ar_pair_mask(N, device, max_prev=None, node_mask=None, B=1):
    """[B, N, N] bool: which (t, j) entries the AR loss scores."""
    idx = torch.arange(N, device=device)
    t, j = idx.view(N, 1), idx.view(1, N)
    m = t > j                                            # strict lower triangle
    if max_prev is not None:
        m = m & ((t - j) <= max_prev)                    # GraphRNN Eq. (9) band
    m = m.unsqueeze(0).expand(B, N, N)
    if node_mask is not None:                            # [B, N] padding-safe
        m = m & (node_mask.unsqueeze(1) & node_mask.unsqueeze(2))
    return m
```

### R5 — The loss (single forward pass, all rows at once)

```python
# losses/autoregressive.py
def ar_edge_loss(Z, A_ord, pair_mask, pos_weight=None, reduction='mean_per_entry'):
    """
    Z          [B, N, N] logits from D4Autoregressive (num_mix=1)
    A_ord      [B, N, N] target adjacency in the SAME ordering
    pair_mask  [B, N, N] bool from ar_pair_mask
    pos_weight scalar tensor, shape [1] or ()

    reduction:
      'mean_per_entry' -> comparable across N and across band widths (use for training)
      'sum_per_graph'  -> the actual -log p(A^pi); use for NLL reporting / marginalisation
    """
    per = F.binary_cross_entropy_with_logits(
        Z, A_ord, pos_weight=pos_weight, reduction='none')          # [B,N,N]
    per = per * pair_mask
    if reduction == 'sum_per_graph':
        return per.flatten(1).sum(-1)                                # [B]
    denom = pair_mask.sum().clamp(min=1)
    return per.sum() / denom
```

**One forward pass, every row scored simultaneously.** There is no loop over `t` at training time —
the causal mask inside the LLM body does all the work. The `N` sequential steps exist *only* at
sampling time. This is precisely GRAN's "parallel training similar to PixelCNN" argument, and it is
also why GRAN deliberately does **not** carry GNN hidden state across steps.

### R6 — Chunked variant for large N (never materialises `[B,N,N]`)

```python
def ar_edge_loss_chunked(Q, K, A_ord, chunk=256, pos_weight=None,
                         max_prev=None, node_mask=None):
    """Q, K: [B, N, r] from D4.q(H), D4.k(H).  Peak memory O(B * chunk * N)."""
    B, N, r = Q.shape
    scale = r ** -0.5
    total = Q.new_zeros(())
    count = 0
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        lo = 0 if max_prev is None else max(0, s - max_prev)
        Zc = torch.einsum('btr,bjr->btj', Q[:, s:e], K[:, lo:e]) * scale     # [B, c, e-lo]
        t = torch.arange(s, e, device=Q.device).view(-1, 1)
        j = torch.arange(lo, e, device=Q.device).view(1, -1)
        m = t > j
        if max_prev is not None:
            m = m & ((t - j) <= max_prev)
        m = m.unsqueeze(0).expand(B, -1, -1)
        if node_mask is not None:
            m = m & (node_mask[:, s:e].unsqueeze(2) & node_mask[:, lo:e].unsqueeze(1))
        per = F.binary_cross_entropy_with_logits(
            Zc, A_ord[:, s:e, lo:e], pos_weight=pos_weight, reduction='none')
        total = total + (per * m).sum()
        count += int(m.sum())
    return total / max(count, 1)
```

### R7 — Mixture-of-Bernoullis row loss (dense port of GRAN's `mixture_bernoulli_loss`)

```python
def mixture_bernoulli_row_loss(log_theta, log_alpha, A_ord, pair_mask):
    """
    log_theta [B, N, N, K] per-pair logits for each mixture component
    log_alpha [B, N, N, K] per-pair mixture logits
    A_ord     [B, N, N]
    pair_mask [B, N, N] bool

    Follows GRAN Eqs. (7)-(9), with the "block" = one row t.
    Returns mean over rows of  -log sum_k alpha_k prod_j theta_{k,t,j}.
    """
    B, N, _, K = log_theta.shape
    m = pair_mask.unsqueeze(-1).to(log_theta.dtype)                   # [B,N,N,1]
    tgt = A_ord.unsqueeze(-1).expand_as(log_theta)
    bce = F.binary_cross_entropy_with_logits(log_theta, tgt, reduction='none') * m
    row_bce = bce.sum(dim=2)                                          # [B,N,K]  sum_j
    cnt = m.sum(dim=2).clamp(min=1)                                   # [B,N,1]
    row_alpha = (log_alpha * m).sum(dim=2) / cnt                      # MEAN, matching GRAN's code
    row_alpha = F.log_softmax(row_alpha, dim=-1)                      # [B,N,K]
    row_logp = torch.logsumexp(-row_bce + row_alpha, dim=-1)          # [B,N]
    valid = (cnt.squeeze(-1) > 0).to(row_logp.dtype)
    return -(row_logp * valid).sum() / valid.sum().clamp(min=1)
```

### R8 — Multi-ordering marginalisation (GRAN Eq. 10)

```python
def marginalised_ar_loss(model, A, orders, **kw):
    """
    orders: [B, C, N] long — C canonical orderings per graph.
    Returns -mean_b logsumexp_c log p(A^{pi_c}), the GRAN Eq. (10) lower bound.
    C forward passes (or fold C into the batch dim).
    """
    B, C, N = orders.shape
    nll = []
    for c in range(C):
        A_ord, X_in = build_ar_inputs_batched(A, orders[:, c])
        H = model.body(model.encoder(X_in), attn_mode='causal')
        Z = model.decoder(H)
        pm = ar_pair_mask(N, A.device, B=B, **kw)
        nll.append(ar_edge_loss(Z, A_ord, pm, reduction='sum_per_graph'))   # [B]
    nll = torch.stack(nll, dim=-1)                                          # [B, C]
    return -torch.logsumexp(-nll, dim=-1).mean()
```

### R9 — Sampling loop

```python
@torch.no_grad()
def ar_sample(model, N, device, temperature=1.0, threshold=None, max_prev=None):
    """O(N) sequential steps.  Use a KV cache in model.body or this is O(N^2) full forwards."""
    A = torch.zeros(1, N, N, device=device)
    X = torch.zeros(1, N, N, device=device)      # X[t] = row t-1 of the strict lower triangle
    for t in range(1, N):
        H = model.body(model.encoder(X[:, :t + 1]), attn_mode='causal')     # [1, t+1, d]
        Q, K = model.decoder.q(H), model.decoder.k(H)
        z = torch.einsum('br,bjr->bj', Q[:, t], K[:, :t]) * model.decoder.scale   # [1, t]
        if max_prev is not None and t > max_prev:
            z[:, :t - max_prev] = -1e4                                       # outside the band
        p = torch.sigmoid(z / temperature)
        row = torch.bernoulli(p) if threshold is None else (p > threshold).float()
        A[0, t, :t] = row
        A[0, :t, t] = row
        if t + 1 < N:
            X[0, t + 1, :t] = row               # feed back the row we just produced
    return A
```

### R10 — Causality unit test (write this before anything else)

```python
# tests/test_d4_causality.py
def test_no_future_leakage():
    """Row t's logits must not depend on any input row >= t."""
    model.eval()
    X = torch.randn(1, N, N, requires_grad=True)
    H = model.body(model.encoder(X), attn_mode='causal')
    Z = model.decoder(H)
    t = N // 2
    Z[0, t, :t].sum().backward()
    assert X.grad[0, t + 1:].abs().max() == 0, "future rows leaked into row t"
    assert X.grad[0, :t].abs().max() > 0,      "past rows are being ignored"


def test_ordering_does_not_leak_labels():
    """The canonical ordering must be computable from the OBSERVED graph only."""
    A_obs, A_full = make_masked_pair()
    assert (canonical_orders(A_obs)['bfs'] == canonical_orders(A_obs)['bfs']).all()
    # and: the ordering function must never be handed A_full in the 5.1/5.2 code path
```

### R11 — Evaluation, MMD (GRAN's exact settings) + PolyGraph

```python
# eval/mmd_graph.py  — mirror GRAN/utils/eval_helper.py exactly so numbers are comparable
from utils.dist_helper import compute_mmd, gaussian_tv, gaussian_emd, gaussian

degree_mmd    = compute_mmd(ref_deg_hists,  gen_deg_hists,  kernel=gaussian_tv)              # sigma=1
spectral_mmd  = compute_mmd(ref_spec_pmfs,  gen_spec_pmfs,  kernel=gaussian_tv)              # sigma=1
clustering_mmd= compute_mmd(ref_clus_hists, gen_clus_hists, kernel=gaussian_tv, sigma=1.0/10)
orbit_mmd     = compute_mmd(ref_orbits, gen_orbits, kernel=gaussian_tv, is_hist=False, sigma=30.0)
# degree hist:  np.array(nx.degree_histogram(G))
# clustering:   np.histogram(list(nx.clustering(G).values()), bins=100, range=(0.0,1.0))
# spectral:     np.histogram(eigvalsh(nx.normalized_laplacian_matrix(G).todense()),
#                            bins=200, range=(-1e-5, 2)) ; then / sum
# orbit:        ORCA `node 4`, np.sum(counts, axis=0) / G.number_of_nodes()
```

```python
# REAL — from BorgwardtLab/polygraph-benchmark README (pip install polygraph-benchmark)
from polygraph.metrics import GaussianTVMMD2Benchmark, StandardPGD, VUN

gtv = GaussianTVMMD2Benchmark(reference)
print(gtv.compute(generated))      # {'orbit':…, 'clustering':…, 'degree':…, 'spectral':…}

pgd = StandardPGD(reference)
print(pgd.compute(generated))      # {'pgd':…, 'pgd_descriptor':…, 'subscores': {'orbit':…, …}}

vun = VUN(reference, validity_fn=reference_ds.is_valid, confidence_level=0.95)
print(vun.compute(generated))      # {'valid':…, 'valid_unique_novel':…, 'valid_novel':…, 'valid_unique':…}
```

Lightweight PGD without the gated TabPFN-2.5 weights:
```python
from sklearn.linear_model import LogisticRegression
pgd = StandardPGD(reference, classifier=LogisticRegression())   # looser bound, no HF login
```

---

## Numbers to beat / hyperparameters to copy

### Copy these directly

| knob | value | source |
|---|---|---|
| mixture components `K` | **20** (K=1→20 improves orbit MMD 6×; K=50 marginal) | GRAN §4.4 + Table 4 |
| block size `B` | **1** (larger is monotonically worse) | GRAN Table 4 |
| GNN/decoder depth | 7 layers, 1 propagation step each | GRAN `config/*.yaml` |
| hidden dim | 128 (N≈361), 512 (N≈500), 256 (N≈5000) | GRAN §4.2 + configs |
| optimiser | Adam, lr **1e-4**, no decay, wd 0 | GRAN configs |
| `edge_weight` / `pos_weight` | 1.0 in GRAN; **compute ours** (Cora full-triangle ≈ 674) | GRAN configs; CLAUDE.md §11 |
| transformer size (from-scratch AR baseline) | 6L/6H/384 ≈10M, 12L/12H/768 ≈85M, 24L/16H/1024 ≈300M | G2PT Table 7 + repo configs |
| AR transformer training | AdamW, lr 1e-4, cosine, wd **1e-1**, 300k iters, bs 60, grad-accum 8, clip 1.0, warmup 2000 | G2PT Table 7 |
| dropout for the AR transformer | **0.0** | G2PT `configs/networks/*.py` |
| GraphRNN-style RNN reference | 4 layers, hidden 128, lr 3e-3, milestones [400,1000], γ=0.3, bs 32, 3000 epochs | GraphRNN `args.py` |
| ordering augmentation | re-draw the permutation **and** the BFS root every `__getitem__` | GraphRNN `Graph_sequence_sampler_pytorch` |
| `M` (band width) starting guess | grid-like 0.11·N; ego/citation 0.6·N; estimate with `calc_max_prev_node(iter=20000, topk=10)` | GraphRNN `create_graphs.py` |

### Numbers to beat

**If we run grid / protein (GRAN's split, gaussian-TV kernel, Deg/Clus/Orbit/Spec):**
- Grid: GRAN **8.23e-4 / 3.79e-3 / 1.59e-3 / 1.62e-2**; GraphRNN 1.12e-2 / 7.73e-5 / 1.03e-3 / 1.18e-2.
- Protein (DD): GRAN **1.98e-3 / 4.86e-2 / 0.13 / 5.13e-3**; GraphRNN 1.06e-2 / 0.14 / 0.88 / 1.88e-2.

**If we run Planar / Tree / Lobster / SBM (G2PT's protocol, adds Wavelet + V.U.N.):**
- The bar is `G2PT_base` V.U.N. **100 / 99 / 100 / 100** with Deg 1.8e-3 / 4.3e-3 / 1e-3 / 4.2e-3.
- GRAN scores V.U.N. **0** on Planar and Tree — beating GRAN on MMD is easy, beating it on V.U.N. is
  the meaningful claim.

**If we run Cora subgraphs (Order Matters protocol, 400 random-walk subgraphs 9≤|V|≤97, gaussian-EMD):**
- GraphRNN BFS: 1.125 / 1.002 / 0.427. GraphRNN uniform: 0.188 / 0.206 / 0.200.
  GraphRNN + learned ordering (VI): **0.066 / 0.171 / 0.052**.

**NLL reference points (GraphRNN Table 2, train/test):** Community-small 28.95/35.10;
Ego-small 9.05/10.61. Their headline generalisation claim is a "22% smaller average NLL gap."

**Speed reference:** GraphRNN ≈ **9.5 s per grid graph** on a GTX 1080Ti. GRAN stride-1 ≈ 6× faster;
stride-16 ≈ 80× faster (with clearly worse quality). Report seconds/graph and #sequential steps.

---

## Evaluation — what to report for §5.3 so it is comparable

Report **all six blocks**. The first two are internal (comparable to §5.1); the rest are external
(comparable to the literature).

**1. Per-edge discrimination (internal, comparable to §5.1).**
AUPRC + AUROC on a **shared held-out index set** of strictly-lower-triangular entries, evaluated in
teacher-forced mode so §5.1 and §5.3 see the same conditioning. Also report the positive rate of that
index set so the AUPRC is interpretable. Never accuracy (CLAUDE.md §13 is right).

**2. Autoregressive NLL (internal).**
`−log p(A^π)` in **nats per scored entry** and per graph, for each ordering in the sweep. If using
`C` orderings, also report the GRAN Eq. (10) bound `−logsumexp_c(−nll_c)`. Report train and test and
the **gap** (GraphRNN's generalisation criterion).

**3. Sample-quality MMD (external).**
Four statistics — degree, clustering, 4-node orbit (ORCA), normalized-Laplacian spectrum — with
**both** kernels, in two columns:
- `gaussian_tv` with GRAN's exact σ (1.0 / 0.1 / 30 / 1.0) → comparable to GRAN, G2PT, everything post-2019.
- `RBF` → because O'Bray et al. (ICLR 2022, arXiv 2106.01098) show the EMD kernel is **indefinite**
  and recommend avoiding the TV kernel too "for its non-p.s.d nature". Their positive recommendation:
  RBF (universal), Laplacian, or linear kernel; and descriptors = degree, clustering, **Laplacian
  spectrum histograms**.

**4. The scale row that almost nobody reports.** O'Bray et al.'s single most actionable
recommendation: *"practitioners should calculate MMD between the test and training graphs, and then
include this in the results table alongside the other MMD results. This will provide a meaningful
bound on what two 'indistinguishable' sets of graphs look like."* **Add a `train vs test` row to
every MMD table.** MMD has no intrinsic scale; without that row a "0.003" is meaningless.

**5. V.U.N. + PGD (external, modern).**
- **V.U.N.** — valid / unique / novel, per Vignac et al.'s protocol as used in G2PT Table 2. This is
  what exposes GRAN's failure on Planar. Only applicable where a validity predicate exists (planar,
  tree, lobster, SBM, molecules).
- **PGD (PolyGraph Discrepancy)** — Krimmel, Hartout, Borgwardt, Chen, **arXiv 2510.06122**, ICLR
  2026, `pip install polygraph-benchmark`, repo `BorgwardtLab/polygraph-benchmark` (verified via the
  GitHub API; default branch `master`). Fits a binary classifier (TabPFN v2.5, or logistic
  regression) to separate real from generated graphs under each descriptor; the classifier's data
  log-likelihood is a variational lower bound on the Jensen–Shannon divergence (their Eq. 2:
  `JS(p‖q) = sup_{d:X→[0,1]} ½E_p[log₂ d] + ½E_q[log₂(1−d)] + 1`), and the square root is the JS
  *distance*, a true metric. Scores lie in **[0,1]**, are comparable across descriptors, and the
  best descriptor is selected by **4-fold stratified CV on the fit split only** (no test leakage).
  Their warning we should heed: `PlanarGraphDataset`, `SBMGraphDataset`, `LobsterGraphDataset`
  "should not be used for benchmarking, due to unreliable metric estimates" — use the `-L` (large)
  variants.

**6. Ordering robustness + cost.**
- MMD/V.U.N./NLL under **≥3 orderings** (BFS, DFS, degree-descending, random). Given GRAN's 10⁴×
  spread, a single-ordering result is not a result.
- Seconds per generated graph, number of sequential decision steps, peak memory. GraphRNN's 9.5 s/graph
  is the anchor.

**⚠ Cross-paper comparability rules.**
1. Never compare a `gaussian_emd` number to a `gaussian_tv` number. GraphRNN's own grid Deg is
   `1e-5`; GRAN's re-evaluation of GraphRNN on grid Deg is `1.12e-2`. Same model, same dataset.
2. Never compare across data splits. GRAN explicitly re-trained GraphRNN on their split "for a fair
   comparison."
3. Cora-the-node-classification-dataset (2708 nodes, CLAUDE.md §6) is **not** Cora-the-graph-
   generation-dataset (400 random-walk subgraphs, 9–97 nodes). If we want GraphRNN/GRAN
   comparability on Cora we must build the subgraph version.
4. Our Phase-1 Cora is a **single graph**. MMD over graph statistics needs a *set* of graphs. For
   §5.3 either (a) use ego/random-walk subgraphs of Cora, or (b) use the OGB molecular sets from
   CLAUDE.md Phase 2 (avg 24–34 nodes — perfect for the `num_mix > 1` path), or (c) accept that on
   single-graph Cora the only meaningful §5.3 metrics are blocks 1 and 2 above.

---

## Open questions

1. **Does the frozen LLM help at all under a causal mask?** Every §9 precedent (GTLM, GaLA) uses a
   *bidirectional* mask over graph tokens. Switching to causal makes the setup much closer to the
   LLM's pretraining regime, which could help — or the graph tokens could be so far out of
   distribution that it does not matter. The from-scratch transformer baseline (CLAUDE.md §7.6) is
   the decisive control and should be run for §5.3 specifically, not just §5.1.

2. **Should the attention bias be causal too?** GTLM's SPD/RRWP/Magnetic-Laplacian biases are
   computed from the full graph. Under an AR factorisation, `bias(t, j)` for `j < t` must be
   computable from the induced subgraph on `1..t−1` — a shortest-path distance computed on the full
   graph leaks the future. This is the same class of bug as C4 and I have not seen it addressed
   anywhere in the literature. Either recompute the bias per prefix (expensive: `O(N)` SPD
   computations) or restrict to biases that are prefix-monotone.

3. **Does `M`-truncation survive a transformer?** GraphRNN's `M` bound is a statement about the
   *edge set*, and it holds regardless of architecture. But the fixed-width reversed-index encoding
   (`encode_adj`'s `[::-1]`) is an RNN convenience. With a transformer we can just band the attention
   mask and the pair mask, which is cleaner. Untested whether the two are equivalent in practice.

4. **Which ordering for `ogbg-mol*`?** G2PT says BFS ≈ degree-based ≫ DFS on molecules; GRAN says
   DFS ≫ BFS on grids. Molecules are neither. Needs a sweep. Note G2PT's repo default is
   `ordering = 'bfs'` even though the paper headlines the degree-based one.

5. **Is the AR task worth it at all for our thesis?** §5.1 is permutation-equivariant, parallel, and
   has no ordering problem; §5.3 buys a likelihood and generative sampling at the cost of an
   arbitrary ordering. If the paper's claim is "adjacency rows into a frozen LLM", §5.3 may be a
   distraction. Recommend: build D4 because CLAUDE.md §12 asks for it, run it on the *molecular*
   datasets where generation is meaningful, and keep Cora on §5.1.

6. **Unverified:** GRAN's Table 4 "Run Time(s)" column is blank for every B=1 row, so I cannot state
   the wall-clock cost of the multi-ordering marginalisation. Their §4.3 numbers are all for B=16.

7. **Unverified:** I did not read G2PT's `model.py` or `train.py`, so I cannot confirm whether their
   transformer is vanilla nanoGPT (learned positional embeddings, pre-LN) or something custom. The
   config fields (`n_layer/n_head/n_embd/dropout/bias`, `block_size`, `vocab_size`,
   `gradient_accumulation_steps`) are exactly nanoGPT's, so it almost certainly is — but I did not
   verify it.

---

## Sources fetched

Primary papers (downloaded PDF + `pdftotext -layout`, read directly):
- `arxiv.org/abs/1802.08773` + `arxiv.org/pdf/1802.08773` — GraphRNN (You, Ying, Ren, Hamilton, Leskovec, ICML 2018)
- `arxiv.org/abs/1910.00760` + `arxiv.org/pdf/1910.00760` — GRAN (Liao et al., NeurIPS 2019)
- `arxiv.org/abs/2501.01073` + `arxiv.org/pdf/2501.01073` + `arxiv.org/html/2501.01073v2` — G2PT (Chen et al., ICML 2025)
- `arxiv.org/pdf/2106.06189` — Order Matters (Chen, Han, Hu, Ruiz, Liu, ICML 2021)
- `arxiv.org/pdf/2106.01098` — Evaluation Metrics for Graph Generative Models (O'Bray, Horn, Rieck, Borgwardt, ICLR 2022)
- `arxiv.org/pdf/2510.06122` — PolyGraph Discrepancy (Krimmel, Hartout, Borgwardt, Chen, ICLR 2026)
- `arxiv.org/pdf/2502.02216` — AutoGraph / Flatten Graphs as Sequences (Chen, Krimmel, Borgwardt, NeurIPS 2025)
- `arxiv.org/pdf/2006.15502` — BiGG (Dai, Nazi, Li, Dai, Schuurmans, ICML 2020)

Secondary rendering used only for cross-checking table layout:
- `ar5iv.labs.arxiv.org/html/1802.08773`, `ar5iv.labs.arxiv.org/html/1910.00760`

Source code read verbatim (raw.githubusercontent.com / GitHub API):
- `JiaxuanYou/graph-generation` — `data.py`, `model.py`, `train.py`, `args.py`, `create_graphs.py`, `eval/stats.py`, `eval/mmd.py`
- `lrjconan/GRAN` — `model/gran_mixture_bernoulli.py`, `dataset/gran_data.py`, `utils/dist_helper.py`, `utils/eval_helper.py`, `config/gran_grid.yaml`, `config/gran_DD.yaml`
- `tufts-ml/G2PT` — full file tree via GitHub API, `datasets_utils.py`, `configs/networks/{small,base,large}.py`, `configs/datasets/planar.py`, `configs/default.py`, `tokenizers/planar/vocab.json`
- `BorgwardtLab/polygraph-benchmark` — repo metadata via GitHub API, `README.md` (master branch)
- `BorgwardtLab/AutoGraph` — README

arXiv id existence checks (HTTP 200 + title match): `2605.10247`, `2606.15633`, `2503.01805`,
`2103.05247`, `2402.05862`, `2310.04560`.
