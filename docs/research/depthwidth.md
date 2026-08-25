# Depth-Width Tradeoffs (arXiv 2503.01805)

Deep dive on **"Depth-Width tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers"** —
the precedent for our input format (node-adjacency tokenization). Plus its direct follow-up,
**"Lost in Tokenization"** (arXiv 2605.22471), which sharpens several of the conclusions we depend on
and which CLAUDE.md does not yet reference.

Everything below was read from primary sources (arXiv HTML of v1 and v3, downloaded and stripped to
text; HuggingFace `config.json` files; the raw Planetoid `ind.cora.graph` file). Where I could **not**
verify something I say so explicitly.

---

## Verified facts (with source next to each)

### Identity of the paper

| Field | Value | Source |
|---|---|---|
| arXiv id | **2503.01805** — resolves, correct | `arxiv.org/abs/2503.01805` |
| Title (arXiv) | *Depth-Width tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers* | `arxiv.org/abs/2503.01805` |
| Title (NeurIPS) | *Depth-Width Tradeoffs for Transformers on Graph Tasks* (shorter) | `neurips.cc/virtual/2025/poster/119487` |
| Authors | Gilad Yehudai, Clayton Sanford, Maya Bechler-Speicher, Orr Fischer, Ran Gilad-Bachrach, Amir Globerson | `arxiv.org/abs/2503.01805` |
| Venue | **NeurIPS 2025, Spotlight Poster** | `neurips.cc/virtual/2025/poster/119487` |
| OpenReview | `openreview.net/forum?id=A2pmNL7L1E` | NeurIPS poster page |
| Versions | v1 2025-03-03, v2 2026-01-28, v3 2026-06-17 | `arxiv.org/abs/2503.01805` |

CLAUDE.md's author list and arXiv id are **exactly right**. Nothing to fix there.

**Version note that matters:** Section-4 theorem numbering is **identical in v1 and v3** (4.1 / 4.2 / 4.3
/ 4.4), and the Table 1 numbers are byte-identical across versions. But v3 **adds** a paragraph on
permutation invariance that does not exist in v1 (see below), and v3 renumbers Section 5
(v1 has Thm 5.1 subgraph counting + Thm 5.3 Eulerian; v3 has Thm 5.1 Eulerian + Thm 5.2 subgraph
counting, sections swapped). If you cite Section 5, **pin the version**.

---

### 1. Node-adjacency tokenization — the definition

Verbatim, Section 3.1 "Graph Inputs" (v1 and v3 identical):

> In this work, we focus primarily on the node-adjacency tokenization scheme, where each token
> corresponds to a node and encodes its incident edges.
>
> Given a graph $G$ with $n$ nodes, let $A \in \mathbb{R}^{n\times n}$ be its adjacency matrix.
> The $i$th token input $\mathbf{x}_i \in \mathbb{R}^n$ to the transformer is defined as the $i$th row
> of $A$. Thus, $X \in \mathbb{R}^{n\times n}$.

**CONFIRMED: token i = row i of A.** Exactly what CLAUDE.md says.

Their two comparison schemes, also verbatim:

> The **edge-list tokenization** of Sanford et al. (2024a) converts the graph into a sequence of
> discrete edge tokens. For a unique identifier assigned to each $v\in V$, this scheme encodes each
> edge $(v_i,v_j)\in E$ as a token $\mathbf{x}=(i,j)\in\mathbb{R}^2$.

Efficiency claim (Section 6.3), verbatim:

> This tokenization offers significant efficiency advantages for dense graphs, as the edge-list
> representation requires $O(n^2)$ tokens, whereas the adjacency-row representation reduces this to
> $O(n)$.

**CONFIRMED** — CLAUDE.md's "edge-list needs O(N²) tokens, adjacency rows need O(N)" is verbatim theirs.

### 1b. Node features — Appendix E (CLAUDE.md's E4). CONFIRMED verbatim

> **Adjacency Rows tokenization.** Tokenization of the graph as an adjacency rows is done as follows.
> Assume a graph over $n$ nodes and adjacency matrix $A$. Each node is associated with a vector of
> features $x_i \in \mathbb{R}^d$. We concatenate to each row of $A$ the node's corresponding feature
> vector. **This results in a vector of size $n+d$ for each node.** As graphs vary in size, we
> **pad each node vector with zeros to match the maximal graph size in the dataset.** Each such
> vector is used as an input token to the transformer.

CLAUDE.md §3 E4 ("Concatenate node features to the adjacency row before projection: `[A_i,: ‖ x_i]`,
giving width N+F ... they zero-pad to the largest graph in the dataset") is **exactly correct**,
right down to the padding detail. This is the one claim in CLAUDE.md about this paper that is
word-for-word accurate.

Their other two, for completeness (same appendix, verbatim):

> **Edge List tokenization.** ... Each node is represented by a one-hot encoding vector
> $b_i\in\mathbb{R}^n$ concatenated with the node input features $x_i\in\mathbb{R}^d$. Then, each edge
> is represented by concatenating its node representations. Each edge representation is fed as an
> independent token. ... pad each node representation with zeros to match the maximal graph size.
> → token width $2(n+d)$, and $|E|$ tokens.

> **Laplacian Eigenvectors tokenization.** ... We compute the eigenvector decomposition of the
> graph's Laplacian. Then, each node is associated an eigenvector and an eigenvalue. We concatenate
> these two for each node, resulting in a vector of size $n+1$. We then concatenate to each of these
> spectral vectors the node's corresponding feature vector $x_i$. This results in a vector of size
> $n+d+1$ for each node.

(The LE description is loosely worded in the paper — "each node is associated an eigenvector" almost
certainly means node $i$ gets row $i$ of the eigenvector matrix, i.e. $n$ numbers, plus one
eigenvalue. Their *theory* Appendix A uses the clean definition
$\mathbf{y}_i=(\mathbf{v}_{1,i},\dots,\mathbf{v}_{m,i})$ with a separate token
$\mathbf{y}_0=(\lambda_1,\dots,\lambda_m)$. The follow-up paper uses yet a third, cleaner convention
— see §"Lost in Tokenization" below. **Do not copy their Appendix-E LE recipe; copy the follow-up's.**)

---

### 2. The theorems — each stated verbatim and checked against CLAUDE.md

#### Theorem 4.1 (CLAUDE.md does not mention this one; it should)

> There exists a transformer with $2$ layers of self-attention, and embedding dimension $O(n)$ that
> solves the $1$ vs. $2$ cycle problem.

Mechanism: sparse graph ⇒ all $n$ edges fit in one token ⇒ offload to an unbounded MLP. Section 4.1
also states the sketching corollary that CLAUDE.md's **E3** leans on, verbatim:

> Ahn et al. (2012) uses linear sketching to solve the connectivity problem on general graphs with
> $O(n\log^3(n))$ total memory. The idea is to use linear sketching, which is a linear projection of
> the adjacency rows into vectors of dimension $O(\text{poly}\log(n))$. Although this is a lossy
> compression, it still allows to solve the connectivity problem.

**CONFIRMED** — CLAUDE.md's E3 motivation is a real, correctly-attributed claim of this paper.
(Caveat about what it does *not* license: see Corrections.)

#### Theorem 4.2 — VERBATIM (v1)

> Let $T$ be a transformer with embedding dimension $m$, depth $L$, bit-precision $p$ and $H$
> attention heads in each layer. Also, assume that the input graphs to $T$ are embedded such that
> each token is equal to a row of the adjacency matrix. Then, if $T$ can detect $2$-cycles on
> directed graphs we must have that:
> 1. If $T$ has residual connections then $mpHL=\Omega(n)$.
> 2. If $T$ doesn't have residual connections then $mpH=\Omega(n)$.

**CONFIRMED as stated.** CLAUDE.md's "worst case `m·p·H·L = Ω(N)`" is right. CLAUDE.md's *gloss*
— "Check d_model against N" — is **wrong** and is corrected below. It is a bound on the **product**,
not on $m$.

Proof: reduction from **set disjointness** with $s=n^2$; two-party protocol where Alice owns tokens
$1..n$ (edges $V_1\!\to\!V_2$) and Bob owns tokens $n{+}1..2n$ (edges $V_2\!\to\!V_1$); each layer
costs $O(nmpH)$ transmitted bits; $L$ layers ⇒ $O(nmpHL) \ge s = n^2$ ⇒ $mpHL=\Omega(n)$.
Without residuals Alice has everything after one layer, so no $L$ factor.

**Theorem B.2 (bounded-degree version of 4.2)** — CLAUDE.md omits this, and it is the one that
actually binds for us:

> Let $T$ be a transformer with embedding dimension $m$, depth $L$, bit-precision $p$ and $H$
> attention heads. If $T$ can detect $2$-cycles on $d$-degree directed graphs, then:
> 1. If $T$ has residual connections then $mpHL=\Omega(d)$.
> 2. If $T$ doesn't have residual connections then $mpH=\Omega(d)$.

#### Theorem 4.3 — VERBATIM

> There exists an $O(L)$-layer transformer with embedding dimension $m=O(n)$ such that, for any graph
> embedded as rows of an adjacency matrix $A$, the output of the transformer in the $i$-th token is
> the $i$-th row of $A^L$.

**CONFIRMED** — CLAUDE.md's statement is accurate. This is the strongest single justification in the
literature for "adjacency-row tokens in ⇒ adjacency-like structure out", i.e. for our decoder.

(v1 says the softmax "equals $A$"; v3 softens it to "approximately equals $A$" — a fix, since a
finite temperature $c$ gives an approximation. Same theorem.)

#### Theorem 4.4 — VERBATIM

> For any $n\in\mathbb{N}$ and $d\leq n$, there exists a single-layer transformer with embedding
> dimension $O(d\log n)$ that detects 2-cycles in any graph with node degree at most $d$.
> This embedding dimension is optimal up to logarithmic factors.

**CONFIRMED**, including the optimality clause (optimality comes from Theorem B.2 above).
CLAUDE.md's arithmetic example ("degree ~20, N=1500 ⇒ ~210 dimensions") checks out under
$\log = \log_2$: $20\cdot\log_2 1500 = 211$. But see Corrections — the *constructive* width in
Appendix B.4 is $m = 2p+2$ with $p=\Omega(d\log n)$, i.e. **roughly twice** the headline number.

#### Section 5 (CLAUDE.md ignores these; worth knowing)

- **v1 Thm 5.1 / v3 Thm 5.2** — subgraph counting: $O(1)$ layers and width
  $O(n^{2-1/k})$ counts occurrences of any fixed $k$-node graph $G'$. (Implements the
  "tri-tri-again" algorithm, Dolev et al. 2012.)
- **v1 Thm 5.3 / v3 Thm 5.1** — Eulerian cycle verification on multigraphs with self loops cannot be
  solved with $m=O(n^{2-\epsilon})$ unless $L=\Omega(\log n)$, conditional on Conjecture 2.4 of
  Sanford et al. 2024b. This is the "quadratic width is sometimes necessary" result in the abstract.

---

### 3. Appendix A — the local/global tradeoff. All four CLAUDE.md claims checked

Appendix A is titled **"Alternative tokenization approaches"** and is a short, informal,
**proof-free** section (no numbered propositions — it is prose with one footnote-length argument).
That matters: CLAUDE.md §11 quotes it as if it were a theorem.

| CLAUDE.md §11 claim | Status | Evidence |
|---|---|---|
| "node degree is computable with width 1 from adjacency rows" | **CONFIRMED** | Verbatim: *"the degree of each node can be computed in a sequence-wise manner with node-adjacency tokenization with embedding dimension $m=1$ by simply computing the inner products $\langle \mathbf{1}_n,\mathbf{x}_i\rangle$."* |
| "connectivity needs depth Ω(log N) or width Ω(N)" | **CONFIRMED as a sentence, but it is asserted, not proved, and it is CONDITIONAL** | Verbatim: *"transformers with the node-adjacency tokenization require either depth $\Omega(\log n)$ or width $\Omega(n)$ to solve the same problem."* No proof is given in Appendix A; it inherits from the 1-vs-2-cycle conjecture (Conjecture 13, Sanford et al. 2024a) invoked in §4.1. See Corrections. |
| "Laplacian eigenvectors make connectivity trivial (disconnected iff λ₂=0)" | **CONFIRMED** | Verbatim: *"the tokenization trivializes the connectivity task because a graph is disconnected if and only if its second-smallest eigenvalue is zero"* |
| "Both schemes become universal only when width is Ω(N²)" | **CONFIRMED** | Verbatim: *"In the regime where $m=\Omega(n^2)$ and MLPs are universal approximators, both tokenization schemes are universal. The entire graph can be encoded in a single token, which can then convert between $A$ and the spectrum of $\mathcal{L}$."* |

Bonus fact CLAUDE.md omits, and which **argues against a pure-spectral encoder**: with the *smallest*
Laplacian eigenvectors, you cannot even compute node degree unless $m$ grows linearly in $n$.
Their construction: $n/3$ disconnected 3-node paths ⇒ eigenvalue 0 has multiplicity $n/3$ ⇒ the first
$n/3$ eigenvectors are pure component indicators ⇒ *"if nodes $i,j,k$ comprise a cluster and
$m\le n/3$, then their embeddings $\mathbf{y}_i,\mathbf{y}_j,\mathbf{y}_k$ are identical."*
**Degree-1 endpoints and the degree-2 middle node get the same token.** This is why E5 must be
`[row ‖ eigvecs ‖ ...]`, never `[eigvecs]` alone.

---

### 4. The empirical table — every number verified

Table 1, verbatim from both v1 and v3 (ROC-AUC, mean ± std over **3 random seeds**):

| Tokenization | molhiv | molbbbp | molbace |
|---|---|---|---|
| EdgeList | 54.01 ± 1.38 | 64.73 ± 1.66 | 66.06 ± 3.89 |
| **AdjRows** | **61.87 ± 1.10** | **67.63 ± 2.57** | **68.64 ± 2.34** |
| LE (Laplacian eigenvectors) | **68.11 ± 1.52** | 55.31 ± 4.79 | 63.61 ± 2.31 |

**Every one of the nine numbers in CLAUDE.md's line matches exactly.** AdjRows beats EdgeList on all
three; AdjRows beats LE on molbbbp and molbace (2 of 3) and loses badly on molhiv. CLAUDE.md is
correct on both counts.

(Cosmetic: the v1 table header misspells the third column "molbeca".)

Dataset stats, Appendix E Table 2:

| Dataset | # Graphs | Avg # Nodes | Avg # Edges | # Node Features | # Classes |
|---|---|---|---|---|---|
| ogbg-molhiv | 41,127 | 25.5 | 27.5 | 9 | 2 |
| ogbg-molbace | 1,513 | 34.1 | 36.9 | 9 | 2 |
| ogbg-molbbbp | 2,039 | 24.1 | 26.0 | 9 | 2 |

CLAUDE.md §6 says "average 24-34 nodes" — matches (24.1 / 25.5 / 34.1).

**Important framing point:** these are *weak* absolute numbers. A plain GIN on molhiv is ~77-78
ROC-AUC in the OGB leaderboard regime; their AdjRows gets 61.87. Their point is the *relative*
ordering of tokenizations under a fixed, deliberately plain transformer — **not** state of the art.
Do not treat 61.87 as "the number to beat" (see Numbers section).

---

### 5. Exact model setup

From Section 6 (verbatim):

> Code is provided in the Supplementary Material. In our experiments, we used a standard transformer
> architecture using **Pytorch's transformer encoder layers** (Paszke et al., 2019). Specifically,
> each layer is composed of Multi-Head Self-Attention, Feedforward Neural Network, Layer
> Normalization and Residual Connections.

**CONFIRMED: PyTorch `TransformerEncoderLayer`.** CLAUDE.md is right.

Appendix E "Hyper-Parameters", verbatim, complete:

> For all experiments, we use a fixed drouput rate of **0.1** and **Relu** activations.
> In Section 6.1 we tuned the learning rate in $\{10^{-4}, 5\cdot 10^{-5}\}$, batch size in
> $\{32,64\}$.
> In Section 6.3 we tuned the learning rate in $\{10^{-3}, 5\cdot 10^{-3}\}$, number of layers in
> $\{3,5,6,10,12\}$, hidden dimensions in $\{32,64\}$. We used batch size of size 64.

Depth/width sweep for §6.1 (fixed ~100k params): **(depth, width) ∈ {(1,125), (2,89), (4,63),
(8,45), (10,40)}**, 100 epochs, 3 seeds, graph sizes 50 and 100.

Critical-width experiment (§6.2): 1 layer, **2 attention heads**, graph sizes 50→400 step 50,
widths 100→800 step 100. "Critical width" = the width at which training loss plateaus above 0.05.
Result: critical width grows **roughly linearly** with graph size.

**NOT SPECIFIED anywhere in the paper (v1 or v3):**
- the **optimizer** (no mention of Adam/AdamW/SGD)
- the **number of attention heads** for the §6.3 OGB experiments (only §6.2 says 2)
- the **pooling / readout** method
- the **output head**
- epochs / schedule for §6.3
- weight decay, gradient clipping

The NeurIPS reproducibility checklist in v3 answers "[Yes]" to "does the paper specify ... type of
optimizer" with an **empty justification field**. It does not.

### 5b. Released code — NOT PUBLIC as far as I can verify

- Paper footnote: *"Code is provided in the Supplementary Material."*
- v3 NeurIPS checklist, verbatim: *"Justification: **We will provide in the final version a Github
  project with the code.**"*
- I found **no GitHub repository**. Web searches for it return only the paper itself.
- OpenReview is behind a Cloudflare challenge from this environment: `api2.openreview.net` returns
  `ChallengeRequiredError` (403), and
  `openreview.net/attachment?id=A2pmNL7L1E&name=supplementary_material` returns **HTTP 403**.

**Conclusion: there is no code we can read.** CLAUDE.md's "Code in supplementary material" repeats
the paper's own footnote accurately, but you should not plan on obtaining it. Reimplement — it is
~40 lines (below).

---

## Corrections to CLAUDE.md

### C1. "Theorem 4.2 — worst case, `m·p·H·L = Ω(N)`. **Check d_model against N.**" — the gloss is wrong

The theorem constrains the **product** $m\cdot p\cdot H\cdot L$, not $m$. For every LLM we are
considering the product exceeds $n$ by three to four orders of magnitude:

| Model | $m$ | $L$ | $H$ | $p$ (bf16) | $mpHL$ | vs $n{=}2708$ |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | 2048 | 16 | 32 | 16 | 16,777,216 | **6,196×** |
| Qwen3-1.7B | 2048 | 28 | 16 | 16 | 14,680,064 | 5,421× |
| Qwen3-4B | 2560 | 36 | 32 | 16 | 47,185,920 | 17,425× |
| Llama-3.1-8B | 4096 | 32 | 32 | 16 | 67,108,864 | 24,782× |
| Qwen3-8B | 4096 | 36 | 32 | 16 | 75,497,472 | 27,879× |

Even the no-residual form $mpH$ is $\ge 10^6$. **Theorem 4.2 imposes no constraint on us whatsoever.**
Rewrite the CLAUDE.md bullet as: *"Theorem 4.2 gives $mpHL=\Omega(N)$; with a real LLM this is
satisfied by ~4 orders of magnitude and is not the binding constraint. The binding constraints are
Theorem 4.3's construction width (~3N) and Theorem 4.4's $\Theta(d\log N)$."*
Confidence: **certain** (theorem text verified verbatim; model dims fetched from `config.json`).

### C2. Theorem 4.3's "width O(N)" is **3N in the construction**, and it needs a one-hot node ID

From Appendix B.3, verbatim (note their notation slip: they write $d$ for $n$):

> We define the input for the transformer as $X=\begin{pmatrix}A\\ I\\ \mathbf{0}\\ \mathbf{0}\end{pmatrix}\in\mathbb{R}^{3d\times d}$,
> namely, there are $d$ tokens, each token contains a column of the adjacency matrix concatenated
> with a positional embedding.

So the residual stream carries three $n$-wide blocks: `[A row ‖ one-hot node id ‖ A^ℓ accumulator]`.
$m = 3n$. For Cora that is $3\times 2708 = \mathbf{8124}$ — **larger than Llama-3.1-8B's 4096 and
4× Llama-3.2-1B's 2048.** CLAUDE.md's "width O(N)" hides this. Also note the construction *requires*
an explicit identity/one-hot node-ID block, which is exactly the thing that destroys permutation
equivariance. Confidence: **certain** (read the proof).

### C3. Theorem 4.4's constructive width is $2p+2$ with $p=\Omega(d\log n)$ — i.e. ~2× the headline

From Appendix B.4, verbatim: *"produce the tokens $\tilde X=(\tilde{\mathbf{x}}_1,\dots\tilde{\mathbf{x}}_n)\in\mathbb{R}^{m\times n}$
for $m=2p+2$"*, with Lemma B.4 requiring $p=\Omega(d\log n)$. Each token is
$\tilde{\mathbf{x}}_i=(\phi(\mathbf{x}_i),\,\mathbf{y}_i,\,1,\,0)$ — a sketch of the row **and** the
node's own RIP codeword.

Applied to Cora, with $\log_2$ and the $\Omega$ constant taken as 1 (optimistic):

| $d$ (max degree) | $d\log_2 N$ | $2d\log_2 N + 2$ (construction) |
|---|---|---|
| Cora max degree **168** | 1,916 | **3,833** |
| Cora p99 degree 19 | 217 | 435 |
| Cora mean degree 3.9 | 44 | 91 |

So Llama-3.2-1B's 2048 does **not** clear Theorem 4.4 at Cora's *max* degree; Llama-3.1-8B / Qwen3-8B
at 4096 barely does. It clears it comfortably at the 99th-percentile degree. CLAUDE.md's blanket
"d_model 4096 clears it comfortably" is true for its own example (N=1500, d=20) but not for Cora's
hub nodes. Confidence: **certain** on the formula, **likely** on the practical implication (big-O
constants are unspecified).

### C4. Appendix A's connectivity bound is **conditional and unproved in that appendix**

CLAUDE.md §11 states it flatly: *"connectivity needs depth Ω(log N) or width Ω(N)"*. The paper's
Appendix A asserts it in one sentence with no proof. Its provenance is Conjecture 13 of Sanford et
al. 2024a (the 1-vs-2-cycle MPC conjecture), which §4.1 explicitly labels a **conjecture**, and which
the paper's own Theorem 4.1 shows is broken by linear width. Contrast with Theorem 4.2, which the
paper is careful to call *"unconditional"*.

The **follow-up paper gives a cleaner version** (see below): Theorem 4 of arXiv 2605.22471 —
*"Assuming $\mathsf{TC}^0\subsetneq\mathsf{L}$, any Transformer requires $\Omega(\log n)$ depth to
solve global graph connectivity, even when provided with full $\Theta(n)$ Adjacency rows."*
Still conditional, but on a much weaker and more standard separation, and **width does not save you**
— note "even when provided with full $\Theta(n)$ adjacency rows". Cite that one instead.
Confidence: **certain** that Appendix A gives no proof; **certain** on the follow-up's statement.

### C5. Permutation invariance — the quote CLAUDE.md relies on exists only in **v2/v3**, and our encoder breaks equivariance regardless of what the LLM body does

CLAUDE.md §11: *"the Depth-Width paper says so explicitly and accepts it for convenience."*

**In v1 the word "permutation" appears nowhere in this context.** The statement was added later.
v3, verbatim:

> The adjacency node embedding is not permutation invariant, similar to the edge-list embedding used
> in Sanford et al. (2024a). Namely, if the tokens contain positional embeddings, then changing the
> order of the tokens may also change the output. These embeddings are used for technical
> convenience, especially since spectral embeddings that are permutation invariant are difficult to
> analyze for combinatorial problems (such as connectivity, subgraph counting, etc.). In section 6 we
> provide experiments that justify the use of such embedding on real-world datasets.

So: **CONFIRMED for v3, FALSE for v1.** Update the citation to "(v2 onward)".

**The more serious issue — a real bug waiting to happen in our repo.** CLAUDE.md §11 then says
GTLM's fix (RoPE reset + bidirectional prefix mask + structure-derived bias) *"restores
equivariance. Adopt it, and write a unit test: permute the input nodes, check the output permutes
identically."*

**That unit test will fail for E1, E2, E3, E4 and E5, no matter what the LLM body does.** The
non-equivariance lives in the *encoder*, upstream of the LLM. Under a node permutation $P$,
$A \mapsto PAP^\top$, so token $i$'s **contents** are permuted, not just the token order:

```
E1:  t_i        = W · A[i, :]              W ∈ R^{d_model × N}, fixed
     t'_{π(i)}  = W · (PAPᵀ)[π(i), :] = W · Pᵀ A[i, :]  ≠  W · A[i, :] = t_i
```

A fixed $[d,N]$ weight matrix assigns a **different learned direction to each column index**, i.e. to
each node id. Same argument kills E3 (`A @ R` with fixed `R ∈ R^{N×k}`), E4 and E5 (they contain the
row). Yehudai et al.'s v3 paragraph blames "positional embeddings", but the adjacency row **is
itself** a node-indexed encoding — the follow-up paper states it without the hedge:
*"adjacency tokenization is not permutation-equivariant, unlike the spectral and random-walk
variants."* (arXiv 2605.22471 §3.2)

**Fix the spec, not the test.** Options, in order of how much I'd trust them:
1. **Accept order-dependence for the fixed-node-set setting** (Cora, gene networks) and *say so in
   the paper*. This is what Yehudai et al. do and they justify it empirically. Delete the
   equivariance unit test for row-based encoders.
2. If you want a genuinely equivariant model, the graph must enter **only through the attention
   bias**, with tokens built from permutation-equivariant features (degree, RW return
   probabilities, sign-invariant Laplacian eigvecs, node features $X$). Then E1-E5 are out as the
   *sole* structure path and the bias carries $A$.
3. Keep the equivariance test but scope it to **E6/E7 + bias-only** configurations, and add a
   separate, weaker test for E1-E5: *"permuting nodes and applying the same permutation to the
   encoder's input weight columns leaves the output equivariant"* — that at least catches indexing
   bugs.

Confidence: **certain** on the math; **certain** that v1 lacks the quote.

### C6. E3's theoretical warrant does not cover a plain Gaussian projection

CLAUDE.md E3: *"Project the adjacency row through a fixed random matrix `R [N,k]` (k ≈ 256-1024)...
Motivated by linear sketching of adjacency rows (Ahn, Guha & McGregor 2012), which the Depth-Width
paper cites as sufficient to solve connectivity despite being lossy compression."*

The *citation* is accurate. The *inference* is not. AGM's sketch is a specific $\ell_0$-sampling /
$\ell_0$-sampler-over-incidence-vectors construction with a specific multi-round Boruvka-style
**decoding algorithm**; the DepthWidth construction (§4.1) works only because it then hands the
sketch to an **arbitrarily large, unbounded MLP** that runs AGM's decoder. A `Linear(k, d_model)` on
`A @ R` with `R ~ N(0,1)` is a Johnson-Lindenstrauss projection, which preserves *inner products*,
not connectivity-decodability. **E3 is a fine empirical encoder — the follow-up paper uses exactly
it — but do not claim in the writeup that AGM proves it retains connectivity.**

Sizing sanity check for Cora if you do want to gesture at AGM: $\log^3 n$ with $\log_2$ is
$11.4^3 = 1482$; with $\ln$ it is $7.9^3 = 494$. So CLAUDE.md's `k ≈ 256-1024` is in the right
ballpark for the natural-log reading. Confidence: **certain** on the AGM/JL distinction,
**likely** on the sizing being "right ballpark".

### C7. Cora has **5,278** undirected edges in Planetoid/PyG, not 5,429 — and `pos_weight ≈ 694`

CLAUDE.md §6 and §11 say "5,429 edges" and "about 0.1% positive". I loaded the raw Planetoid
adjacency (`ind.cora.graph`, the file PyG's `Planetoid` builds from) and counted:

```
N = 2708
undirected simple edges (self-loops removed) = 5278
edge_index columns (2E)                      = 10556      <- matches PyG docs "Edges 10,556"
max degree = 168   (node 1358)
min degree = 1     isolated nodes = 0
mean degree = 3.8981
density (symmetric full matrix) = 0.143947%
pos_weight = neg/pos = 693.70   (full symmetric matrix)
pos_weight = neg/pos = 693.44   (upper triangle only)
degree percentiles: p50=3  p90=7  p95=9  p99=19  p99.9=65  max=168
```

5,429 is the **raw LINQS `cora.cites` line count**, before deduplication and self-loop removal. The
number you will actually see in code is 5,278 / 10,556. PyG's own docs table says Cora has
**10,556 edges** (= $2\times5278$), 1,433 features, 7 classes, 2,708 nodes — CLAUDE.md's feature and
class counts are right. Confidence: **certain** (computed from the primary file).

Actionable consequence: **`pos_weight ≈ 694`**, not the ~1350 you'd get from 5429 positives over
$N^2$. And 15% masking of the upper triangle = 549,791 masked pairs of which only **~792 are real
edges** — that is the AUPRC denominator, and it is why CLAUDE.md's "score only what was masked +
an equal number of sampled non-edges" rule matters so much.

### C8. Pooling / output head / optimizer — CLAUDE.md doesn't claim these, but the task asked, and the answer is "the paper does not say"

I searched both v1 and v3 for `pool`, `readout`, `CLS`, `Adam`, `optimiz`. **Zero hits** for pooling
and optimizer. Do not fabricate these. Use the follow-up paper's setup (below), which is by an
overlapping author group and is fully specified.

---

## The follow-up paper you should also be reading — "Lost in Tokenization" (arXiv 2605.22471)

**Not in CLAUDE.md. It should be.** Maya Bechler-Speicher (Meta AI) et al., v1 submitted 2026-05-21.
Same tokenization vocabulary, sharper theorems, a fully specified experimental setup, and — critically
— **a theorem about edge prediction, which is our primary task.**

### Its tokenizations (verbatim definitions)

```
Spectral:      x_v = (u_1(v),...,u_n(v), λ_1,...,λ_n) ∈ R^{2n}        L = D - A = UΛUᵀ
  truncated:   x_v = (u_1(v),...,u_k(v), λ_1,...,λ_k) ∈ R^{2k}
  T_pre = Õ(n²) full / Õ(kn) truncated ; lossless at k = n

Random-walk:   x_v = ((P^1)_vv, ..., (P^t)_vv) ∈ R^{t(n)}             P = D^{-1} A
  T_pre = O(t(n)|E|) ; LOSSY for every t(n) (Theorem 2)

Adjacency:     x_v = A_v = (A_v1,...,A_vn) ∈ {0,1}^n
  LOSSLESS. T_pre = O(|E|), N_tok = n, d_tok = Θ(n)
  truncated:   P_adj,tr(G) = A R ∈ R^{n × d_tr},  R ∈ R^{n × d_tr} sampled ONCE, shared
               across all nodes AND all graphs.  R_ij ~ N(0,1).
```

That truncated variant **is CLAUDE.md's E3, exactly**, with a real empirical precedent.

### Its theorems, and what each one means for us

- **Theorem 1** ($\mathsf{TC}^0\subsetneq\mathsf{NC}^1$): *"any Transformer utilizing Adjacency
  tokenization requires $\Omega(\log k)$ depth to solve $k$-closed-walk detection."* Adjacency rows
  make you *simulate* path traversal. Random-walk tokens hand you the answer at depth 0.
  → **Argues for adding RRWP-style features to E5**, which is also what GTLM does.
- **Theorem 2:** Random-walk tokenizations of **any** length are strictly insufficient to determine
  planarity (Godsil-McKay switching gives identical RW distributions for a planar and a non-planar
  graph). → RW features alone are not a substitute for the row.
- **Theorem 3 (truncation is brittle)**, verbatim: *"(1) **Laplacian:** If $T$ uses a truncated
  spectrum retaining only $k \ll n$ eigenvalues (either largest or smallest), it cannot accurately
  count triangles. (2) **Adjacency:** If $T$ uses tokens with hidden dimension $m$, precision $p$,
  and $H$ attention heads across $L$ layers, then $mpHL=\Omega(n)$ with residual connections, and
  $mpH=\Omega(n)$ without."* — the Theorem-4.2 bound, re-derived for triangle counting.
- **Theorem 4:** connectivity needs $\Omega(\log n)$ depth **even with full $\Theta(n)$ adjacency
  rows**, assuming $\mathsf{TC}^0\subsetneq\mathsf{L}$ (undirected connectivity is $\mathsf{L}$-complete).
  → **Width does not buy you connectivity. Depth does.** Good news for us: a 16-36 layer LLM has
  depth to spare (see below).
- **Theorem 5 — THE ONE THAT MATTERS FOR OUR PRIMARY TASK.** A 1-layer transformer with **full
  Laplacian tokenization** predicting whether edge $(u,v)$ exists must satisfy

  ```
  Lip_MLP · ||W_V||_2 · (1 + γ)  ≥  d_max        where γ = ||W_QK||_2 · ||X||_2²
  ```

  and, verbatim: *"Because Laplacian tokens $X$ contain unnormalized eigenvalues, their norm scales
  with graph size ($\|X\|_2^2=\Omega(n^2)$ for dense graphs...). ... It must either allow the logit
  energy $\gamma$ to scale to $\Omega(n)$, causing **Softmax saturation and vanishing gradients**, or
  aggressively shrink $\|W_{QK}\|_2$ ... forcing the remaining weights to diverge linearly with $n$,
  causing **catastrophic weight explosion and ill-conditioning**."* Remark 1: depth only spreads the
  problem, giving $\Omega(d_{\max}^{1/L})$ per layer.

  **Direct consequence for our E5:** concatenating raw Laplacian eigenvectors **and raw eigenvalues**
  to the adjacency row, feeding it into a frozen LLM whose LayerNorm/attention scale we do not
  control, is an ill-conditioning trap. Concatenating the row is what keeps us safe (the row makes
  edge prediction trivially local). **Normalize the spectral block** — symmetric normalized
  Laplacian, drop the trivial eigenvector, random sign flips, and per-feature standardization — and
  consider dropping the raw eigenvalues entirely or passing them through a log/rank transform.

### Its empirical table (Table 1) — much stronger numbers than the DepthWidth paper

Graph transformer with each tokenization, "Pad." = full width zero-padded to $n_{\max}$,
"Trunc." = fixed width 8, "Comb." = all three tokenizations concatenated. 3 seeds.

| Model | Tok. | Mode | BBBP ↑ | BACE ↑ | HIV ↑ | Tox21 ↑ | ZINC MAE ↓ | EC-5 RSE ↓ | MaxClq F1 ↑ | TopoOrd MAE ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSet | – | – | 61.37 | 75.22 | 74.82 | 73.79 | 0.701 | 0.766 | 9.805 | 0.252 |
| GIN | – | – | 66.46 | 72.73 | 73.20 | 69.78 | 0.441 | **0.035** | 20.04 | 0.322 |
| GT | Lap | Pad. | 65.98 | **75.94** | 75.42 | **81.36** | 0.678 | 0.098 | 9.805 | 0.246 |
| GT | Lap | Trunc. | 68.45 | 74.91 | 73.15 | 76.10 | 0.650 | 0.077 | 3.937 | 0.239 |
| GT | RW | Pad. | 67.05 | 70.61 | 72.75 | 76.30 | **0.342** | 0.279 | 10.78 | 0.252 |
| GT | RW | Trunc. | 67.04 | 70.87 | 74.08 | 75.63 | 0.524 | 0.316 | 4.807 | 0.252 |
| GT | **Adj** | Pad. | 67.90 | 74.88 | 71.31 | 76.70 | 0.562 | 0.051 | **25.20** | **0.196** |
| GT | **Adj** | Trunc. | **70.14** | 75.26 | 71.30 | 74.43 | 0.689 | 0.051 | 6.303 | 0.230 |
| GT | Comb | Pad. | 68.52 | 72.49 | 73.10 | 76.89 | 0.392 | 0.051 | **26.73** | **0.192** |
| GT | Comb | Trunc. | 68.99 | 74.78 | **76.44** | 74.63 | 0.590 | 0.052 | 2.760 | 0.219 |

Three things to take from this:

1. **AdjRows on molhiv is 71.31 here vs 61.87 in the DepthWidth paper.** Same tokenization, ~10
   ROC-AUC points apart. The difference is tuning (their §6.3 grid was width ∈ {32,64} only). **Use
   71.31 / 67.90 / 74.88 as the real bar, not 61.87 / 67.63 / 68.64.**
2. **`Comb` (all tokenizations concatenated) wins on MaxClq and TopoOrd** — direct empirical support
   for our **E5** (row ‖ structural), and for going further to `row ‖ Lap ‖ RW`.
3. **Adjacency dominates on the two tasks that are about exact edge constraints** (MaxClique
   membership, topological order), verbatim: *"On tasks governed by explicit local constraints,
   adjacency-based tokenization is favored ... Both tasks depend on exact edge constraints ...
   Adjacency exposes these constraints directly, whereas spectral and random-walk tokens provide
   only global or diffusive summaries."* **Masked edge prediction is exactly such a task.** This is
   the strongest single justification in the literature for our input format on our primary task.

### Its experimental setup — fully specified, copy this

> The graph transformer is implemented as a dense-token Transformer Encoder: node features are first
> projected linearly to hidden dimension $H$, and the resulting node tokens are processed by $L$
> standard Transformer Encoder blocks with **four attention heads**, **feed-forward dimension $4H$**,
> and **dropout rate 0.1**. For graph-level tasks, Transformer token embeddings are
> **masked-mean-pooled and passed through a single linear output layer**; for node-level tasks, the
> same linear layer is applied independently to each non-padded node token.

Masking protocol, verbatim (relevant to our batching):

> Missing tokens are zero padded and excluded by the padding mask during transformer attention and
> mean pooling; node-level losses and metrics also ignore padded target positions. **The transformer
> uses only a padding mask, not an edge-based attention mask**, i.e. all real nodes in a graph may
> attend to one another, while padded tokens are excluded from attention, pooling, loss, and metrics.

Tokenization implementation notes, verbatim:

> Laplacian tokens use the **symmetric normalized graph Laplacian**, **sorted eigenvectors after
> dropping the trivial eigenvector**, and **random sign flips**. Random-walk tokens use return
> probabilities from powers of the row-normalized transition matrix. Adjacency tokens use the dataset
> node order directly: padded mode stores the binary adjacency row, while truncated mode uses a
> **Gaussian random projection of outgoing adjacency rows**. In the truncated setting, the random
> projection matrix is sampled $R_{ij}\sim\mathcal{N}(0,1)$ **once at initialization**. For directed
> graphs, the respective adjacency matrix is **not symmetrized**.

Hyperparameters (Table 3, GT column): layers ∈ {1,2,5,10}; hidden ∈ {64,128,256}; dropout 0.1;
**gradient norm clip 1.0**; heads 4; **LR 1e-3**; **weight decay 1e-4**; seeds {42, 123, 456}.
Truncation dimension **8** for all datasets. Hardware: V100 32GB and A100 80GB.
(Table 4) BBBP/BACE/HIV/Tox21: batch 64, 200 epochs, **ReduceLROnPlateau**, warmup 50.
ZINC: batch 8, 1000 epochs, CosineAnnealing, warmup 50. **Optimizer still not named** — weight decay
1e-4 + warmup + cosine implies AdamW, but that is my inference, not their statement.

---

## Mechanism / math (exact formulas, tensor shapes)

### Their transformer (Section 3.1, verbatim formula)

Tokens are **columns**, not rows, in their notation: $X^{(0)}\in\mathbb{R}^{d_{\text{in}}\times N}$.

$$Z^{(\ell)}=\sum_{h=1}^{H}V^{(\ell)}_{h}X^{(\ell-1)}\,\text{sm}\!\left(X^{(\ell-1)\top}K_h^{(\ell)\top}Q^{(\ell)}_{h}X^{(\ell-1)}\right)$$

with $K,Q,V\in\mathbb{R}^{m_{\ell-1}\times m_{\ell-1}}$, `sm` = row-wise softmax, residual
$\tilde X^{(\ell)}=Z^{(\ell)}+X^{(\ell-1)}$, then a per-token ReLU MLP
$\mathcal{N}^{(\ell)}:\mathbb{R}^{m_{\ell-1}}\to\mathbb{R}^{m_\ell}$. Bit precision $p=O(\log n)$.
$m=\max(m_0,\dots,m_L)$. **Note: no $1/\sqrt{d}$ scaling in their definition** — the temperature is
folded into $K$.

### Theorem 4.3's construction, spelled out — this is the blueprint for our attention bias

Input, $L=1$ case (Appendix B.3, their $d \equiv n$):

```
X = [ A      ]   ∈ R^{3n × n}     block 0: adjacency (as columns)
    [ I      ]                    block 1: one-hot node id  (positional)
    [ 0_{n×n}]                    block 2: A^ℓ accumulator
```

Single head:

```
K = c · [ I  0  0 ]     Q = [ 0  I  0 ]     V = [ I  0  0 ]
        [ 0  0  0 ]         [ 0  0  0 ]         [ 0  I  0 ]
        [ 0  0  0 ]         [ 0  0  0 ]         [ 0  0  0 ]
```

Then, verbatim: $X^\top K^\top Q X = A$, and *"Since all the values of $A$ are either $0$ or $1$, for
a sufficiently large $c>0$, the softmax behave similarly to the hardmax"*, giving

$$V X\,\text{sm}(A)=\begin{pmatrix}A^{2}\deg(A)^{-1}\\ A\deg(A)^{-1}\\ \mathbf{0}\end{pmatrix}$$

and then a **2-layer MLP of width $n$** (Lemma B.3) recovers $\deg$ from
$A\deg(A)^{-1}$ and multiplies it back out, leaving $A^2$ in block 2. For general $L$, swap
$V$ to $\text{diag}(0,I,I)$ and induct: input to layer $L$ is $[A;\,I;\,A^L]$, output
$[0;\,A\deg^{-1};\,A^{L+1}\deg^{-1}]$, MLP cleans up.

**The single most important observation in this whole document for our architecture:**

> The entire purpose of the $O(n)$-wide token block is to make
> $\text{softmax}(X^\top K^\top Q X) \approx A$. **Our architecture hands the model that for free.**

Our design is `score(i,j) = (q_i·k_j)/√d + bias(i,j)` with `bias` derived from graph structure. Set
`bias(i,j) = c · A_ij` (or a learned SPD-lookup whose distance-1 bucket is large) and you obtain
row-normalized $A$ as the attention pattern **without spending a single dimension of $d_{model}$ on
storing the row**. The lower bounds in Theorem 4.2 / B.2 are proved under the explicit hypothesis
*"the input graphs to $T$ are embedded such that each token is equal to a row of the adjacency
matrix"* — i.e. the graph enters **only** through tokens. Our bias path is outside that hypothesis.

⚠️ **This is my analysis, not the paper's.** The paper says nothing about attention-bias injection.
I have *not* verified that the set-disjointness argument fails to extend: in their reduction Alice
owns rows $1..n$ and Bob rows $n{+}1..2n$, and a bias $c\cdot A_{ij}$ for a cross-partition pair is
computable by exactly one of them, so the bias is not obviously "free information". Detecting a
2-cycle still requires token $i$ to learn $A_{ji}$, which lives in Bob's row. **A careful version of
this claim would need its own proof.** Treat "the bias exempts us from the width bound" as a
plausible, unproven, and *paper-worthy* conjecture — not as an established fact. Do not write it
into a paper as settled.

### Theorem 4.4's construction (sparse attention unit)

Lemma B.4 (from RIP / Candès-Tao): for $p=\Omega(d\log n)$ there exist $\mathbf{y}_1..\mathbf{y}_n\in\mathbb{R}^p$
such that for any $\mathbf{x}\in\{0,1\}^n$ with $\|\mathbf{x}\|_1\le d$ there is $\phi(\mathbf{x})\in\mathbb{R}^p$ with
$\langle\phi(\mathbf{x}),\mathbf{y}_i\rangle = 1$ if $\mathbf{x}_i=1$ and $\le\tfrac12$ if $\mathbf{x}_i=0$.

Tokens, $m=2p+2$:

```
x̃_i     = [ φ(A_i,:) ; y_i ; 1 ; 0 ]        real node i
x̃_{n+1} = [ 0_p      ; 0_p ; 0 ; 1 ]        dummy sink node
Q x̃_i   = c·[ φ(A_i,:) ; y_i ; 7/4 ; 0 ]
K x̃_i   =   [ y_i      ; φ(A_i,:) ; 0 ; 0 ]
K x̃_{n+1}=  [ 0_p      ; 0_p ; 1 ; 0 ]
V x̃_i   =   [ 0_p      ; 0_p ; 0 ; 1 ]      V x̃_{n+1} = 0
```

giving $(\tilde X^\top K^\top Q\tilde X)_{j,i} = c(\langle\phi(\mathbf{x}_i),\mathbf{y}_j\rangle+\langle\phi(\mathbf{x}_j),\mathbf{y}_i\rangle)$
which is $2c$ iff both $A_{ij}=1$ and $A_{ji}=1$, and $\le \tfrac32 c$ otherwise; the dummy sits at
$\tfrac74 c$ and absorbs all attention when no 2-cycle exists.

**Design lesson for E3:** the symmetric $\langle\phi(\mathbf{x}_i),\mathbf{y}_j\rangle+\langle\phi(\mathbf{x}_j),\mathbf{y}_i\rangle$
structure means a sketch encoder should emit **two** blocks per token — a compressed row $\phi(A_i)$
*and* the node's own codeword $\mathbf{y}_i$ — not just the projected row. CLAUDE.md's E3
(`Linear(k, d_model) ∘ (A @ R)`) has only the first. **Add the node codeword.** Cheap: a learned
`nn.Embedding(N, k)` or a second fixed random codebook.

---

## Code we can reuse (real snippets, real signatures)

No repo exists (see §5b). These are reimplementations from the papers' prose, with the exact
constants they specify.

### Their model, verbatim-faithful (`baselines/scratch_transformer.py`)

```python
import torch, torch.nn as nn

class DepthWidthTransformer(nn.Module):
    """Yehudai et al. 2503.01805 §6 + Bechler-Speicher et al. 2605.22471 §A.7.

    Paper-specified: PyTorch TransformerEncoderLayer, ReLU, dropout 0.1, 4 heads,
    ff_dim = 4*d, masked-mean-pool -> single Linear.
    NOT specified by 2503.01805: optimizer, pooling, head. Taken from 2605.22471.
    """
    def __init__(self, d_in, d_model=128, n_layers=5, n_heads=4,
                 dropout=0.1, n_out=1):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)          # "node features are first
                                                     #  projected linearly to hidden dim H"
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,                           # 4
            dim_feedforward=4 * d_model,             # "feed-forward dimension 4H"
            dropout=dropout,                         # 0.1
            activation="relu",                       # "Relu activations"
            batch_first=True,
            norm_first=False,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_out)        # "a single linear output layer"

    def forward(self, tokens, pad_mask):
        # tokens:   [B, N_max, d_in]
        # pad_mask: [B, N_max] bool, True = PAD  (src_key_padding_mask convention)
        h = self.enc(self.inp(tokens), src_key_padding_mask=pad_mask)   # [B,N,d]
        keep = (~pad_mask).unsqueeze(-1).float()                        # [B,N,1]
        g = (h * keep).sum(1) / keep.sum(1).clamp(min=1)                # masked mean pool
        return self.head(g)                                             # graph-level
        # node-level: return self.head(h)  -> [B,N,n_out], ignore padded positions
```

Their §6.3 grid, exactly: `lr ∈ {1e-3, 5e-3}`, `n_layers ∈ {3,5,6,10,12}`,
`d_model ∈ {32,64}`, `batch_size = 64`, `dropout = 0.1`.
Follow-up's grid: `n_layers ∈ {1,2,5,10}`, `d_model ∈ {64,128,256}`, `lr = 1e-3`,
`weight_decay = 1e-4`, `clip_grad_norm_(1.0)`, seeds `{42,123,456}`.

### The three tokenizers, exactly as Appendix E / A.7 specify them

```python
import torch

def tok_adj_rows(A, X=None, n_max=None):
    """Yehudai et al. Appendix E: [row_i || x_i], zero-padded to the dataset's
    largest graph.  A: [n,n] float.  X: [n,F] or None.  -> [n_max, n_max + F]"""
    n = A.shape[0]
    n_max = n_max or n
    row = torch.zeros(n_max, n_max, dtype=A.dtype, device=A.device)
    row[:n, :n] = A                                        # zero-pad BOTH dims
    parts = [row]
    if X is not None:
        f = torch.zeros(n_max, X.shape[1], dtype=A.dtype, device=A.device)
        f[:n] = X
        parts.append(f)
    return torch.cat(parts, dim=-1)                        # width n_max + F


def tok_adj_trunc(A, R, X=None):
    """Bechler-Speicher et al. §A.7 truncated adjacency == CLAUDE.md E3.
    R: [n_max, d_tr], R_ij ~ N(0,1), sampled ONCE at init, shared across all
    nodes and all graphs.  Directed graphs are NOT symmetrized."""
    z = A @ R[:A.shape[0]]                                 # [n, d_tr]
    return z if X is None else torch.cat([z, X], -1)
# their d_tr = 8 for every dataset (!). Ours should be larger: see Numbers below.


def tok_laplacian(A, k=None, X=None, generator=None):
    """Bechler-Speicher et al. §A.7: SYMMETRIC NORMALIZED Laplacian, eigenvectors
    sorted by eigenvalue, TRIVIAL (constant) eigenvector DROPPED, RANDOM SIGN FLIPS.
    Do NOT use Yehudai et al.'s Appendix-E recipe (unnormalized L, raw eigenvalues)
    -- see 2605.22471 Theorem 5 on ill-conditioning."""
    n = A.shape[0]
    deg = A.sum(-1).clamp(min=1)
    dinv = deg.pow(-0.5)
    L = torch.eye(n, device=A.device) - dinv[:, None] * A * dinv[None, :]
    evals, evecs = torch.linalg.eigh(L)                    # ascending
    evecs = evecs[:, 1:]                                   # drop trivial
    evals = evals[1:]
    if k is not None:
        evecs, evals = evecs[:, :k], evals[:k]
    sign = torch.randint(0, 2, (evecs.shape[1],), generator=generator,
                         device=A.device).float() * 2 - 1  # random sign flip
    U = evecs * sign                                       # [n, k]
    out = [U]
    if X is not None:
        out.append(X)
    return torch.cat(out, -1)
```

### Theorem 4.3 as an attention bias (our `models/bias/`)

The construction's whole job is `softmax(logits) ≈ row_normalize(A)`. Do it directly:

```python
class AdjacencyBias(nn.Module):
    """score(i,j) = q_i·k_j/sqrt(d) + bias(i,j), with bias built from A.

    Setting c large recovers Theorem 4.3's attention pattern:
        softmax(c*A)_i  ->  A_i / deg(i)   as c -> inf
    which is exactly what their O(n)-wide token block was constructed to produce.
    Per-head lambda_h follows GaLA (CLAUDE.md sec 9): do not force the bias on
    every head."""
    def __init__(self, n_heads, init_c=4.0):
        super().__init__()
        self.log_c = nn.Parameter(torch.full((n_heads,), float(init_c)).log())

    def forward(self, A):                     # A: [B,N,N] float (masked entries 0)
        c = self.log_c.exp().view(1, -1, 1, 1)
        return c * A.unsqueeze(1)             # [B,H,N,N], added pre-softmax
```

Sanity assertion worth putting in `tests/`:

```python
# Theorem 4.3, L=1: attention with bias c*A and V=I should reproduce D^{-1}A.
c, A = 50.0, adj.float()
P = torch.softmax(c * A + (A == 0) * -1e9 * 0, dim=-1)      # no qk term
assert torch.allclose(P, A / A.sum(-1, keepdim=True).clamp(min=1), atol=1e-4)
```

### Class-imbalance constants for Cora, computed not guessed

```python
N, E = 2708, 5278                      # undirected simple edges, self-loops removed
pos = 2 * E                            # 10556 entries equal 1 in the full symmetric matrix
neg = N * N - pos                      # 7322708
pos_weight = neg / pos                 # 693.70   <-- use this
# upper-triangle-only variant gives 693.44 -- same number, use either consistently.
# 15% mask of the upper triangle -> 549791 masked pairs, of which ~792 are true edges.
```

---

## Numbers to beat / hyperparameters to copy

### Numbers

| What | Value | Where from | Use |
|---|---|---|---|
| AdjRows ROC-AUC, molhiv | 61.87 ± 1.10 | 2503.01805 T1 | *Weak* baseline. Cite for tokenization ordering only. |
| AdjRows ROC-AUC, molbbbp | 67.63 ± 2.57 | 2503.01805 T1 | ditto |
| AdjRows ROC-AUC, molbace | 68.64 ± 2.34 | 2503.01805 T1 | ditto |
| **Adj/Pad ROC-AUC, HIV** | **71.31 ± 1.26** | 2605.22471 T1 | **the real bar** |
| **Adj/Pad ROC-AUC, BBBP** | **67.90 ± 2.35** | 2605.22471 T1 | **the real bar** |
| **Adj/Pad ROC-AUC, BACE** | **74.88 ± 2.21** | 2605.22471 T1 | **the real bar** |
| GIN on BBBP / BACE / HIV | 66.46 / 72.73 / 73.20 | 2605.22471 T1 | GNN reference in the same harness |
| DeepSet on BBBP / BACE / HIV | 61.37 / 75.22 / 74.82 | 2605.22471 T1 | **no-graph** reference — note it beats GIN on 2 of 3! |
| Best combined tokenization, HIV | 76.44 ± 1.70 (Comb/Trunc) | 2605.22471 T1 | ceiling for E5-style concatenation |

**The DeepSet row deserves a comment in our writeup.** On BACE and HIV a model that *ignores edges
entirely* beats GIN and beats adjacency tokenization. That is CLAUDE.md §7's "feature-only baseline"
already proving its point in the literature. Run it first.

### Hyperparameters to copy

```yaml
# from 2605.22471 Table 3 (fully specified) -- use for baselines/scratch_transformer
n_layers:        [1, 2, 5, 10]
d_model:         [64, 128, 256]
n_heads:         4
ff_dim:          4 * d_model
dropout:         0.1
lr:              1.0e-3
weight_decay:    1.0e-4
grad_clip_norm:  1.0
seeds:           [42, 123, 456]
batch_size:      64          # molecular; 8 for ZINC, 256 for GraphBench tasks
epochs:          200         # molecular
scheduler:       ReduceLROnPlateau   # molecular; CosineAnnealing for ZINC/GraphBench
warmup:          50
trunc_dim:       8           # their choice; too small for us, see below
# optimizer:     NOT STATED in either paper. AdamW is the obvious inference.

# from 2503.01805 App. E sec 6.3 (their molecular grid, for exact replication)
lr:              [1.0e-3, 5.0e-3]
n_layers:        [3, 5, 6, 10, 12]
d_model:         [32, 64]
batch_size:      64
dropout:         0.1
activation:      relu
```

### Sizing for Cora (N = 2708) — computed

```
N          = 2708
N^2        = 7,333,264
log2(N)    = 11.403        ln(N) = 7.904
max degree = 168           mean = 3.898   p99 = 19   p99.9 = 65
```

| Requirement | Formula | Cora value | Llama-3.2-1B (2048) | Llama-3.1-8B / Qwen3-8B (4096) |
|---|---|---|---|---|
| Thm 4.2 lower bound | $mpHL \ge \Omega(N)$ | 2,708 | 16.8M ✅ 6196× | 67-75M ✅ |
| Thm 4.3 headline | $m = O(N)$ | 2,708 | 2048 ✳ marginal | 4096 ✅ |
| **Thm 4.3 construction** | $m = 3N$ | **8,124** | ❌ 0.25× | ❌ 0.50× |
| Thm 4.4 headline | $d\log_2 N$ @ d=168 | 1,916 | ✳ marginal | ✅ |
| **Thm 4.4 construction** | $2d\log_2 N + 2$ @ d=168 | **3,833** | ❌ | ✅ (just) |
| Thm 4.4 @ p99 degree 19 | $2d\log_2 N + 2$ | 435 | ✅ | ✅ |
| App. A universality | $m = \Omega(N^2)$ | 7,333,264 | ❌ | ❌ |
| Connectivity depth (2605.22471 Thm 4) | $L = \Omega(\log_2 N)$ | 11.4 → **12 layers** | **16 ✅** | 32 / 36 ✅ |
| AGM sketch dim (E3) | $O(\log^3 N)$ | 494 (ln) / 1482 (log2) | — | — |

**Bottom line on sizing: depth is fine, width is the problem, and only at Cora's hub nodes.**

### Recommended E3 sketch dimension

The follow-up paper's `d_tr = 8` is far too aggressive for our task (they were testing brittleness,
and their truncated adjacency collapses on MaxClique: F1 6.30 vs 25.20 padded). For Cora:
- $\ln^3 N \approx 494$ → **k = 512** is the AGM-flavoured choice, matching CLAUDE.md's `256-1024`.
- Also emit the node codeword (Thm 4.4's $\mathbf{y}_i$): total token width $2k = 1024$, comfortably
  inside $d_{model}=2048$.

---

## Does $d_{model} < N$ sink us? — the answer for our project

**Short answer: no, but three of CLAUDE.md's mitigations are load-bearing and one of its claims is
misleading.**

### What each theorem actually says about Cora + Llama

1. **Theorem 4.2 is a non-issue.** $mpHL$ for Llama-3.2-1B is $\approx 1.7\times10^7$ vs $N=2708$.
   Off by 6,196×. Delete "Check d_model against N" from CLAUDE.md.

2. **Theorem 4.3 is where $d_{model} < N$ bites.** Its construction needs a $3N$-wide residual
   stream ($8124$ for Cora) purely to hold `[row ‖ id ‖ accumulator]`. No LLM we are considering has
   that. **But we do not need that construction**, because the entire purpose of its first block is
   to make the attention matrix equal $A$ — and our architecture injects $A$ into the attention
   logits directly. That is the architectural answer, and it is genuinely novel relative to this
   paper. (See the ⚠️ caveat above: this is a conjecture, not a proved exemption.)

3. **Theorem 4.4 is the one to actually respect.** $2d\log_2 N + 2$ with Cora's max degree 168 is
   **3,833** — above Llama-3.2-1B's 2048, below Llama-3.1-8B's 4096. At the 99th-percentile degree
   (19) it is 435, trivially satisfied. So a $d_{model}=2048$ model can represent 99% of Cora's
   nodes' neighbourhoods losslessly-enough and will be lossy on the ~27 hub nodes. **Log per-node
   degree against prediction error and check whether hub nodes are where the model fails** — that is
   a cheap, publishable diagnostic that falls straight out of Theorem 4.4.

4. **Depth is comfortably sufficient.** The binding depth result (2605.22471 Thm 4) is
   $\Omega(\log n)$ for connectivity *even at full width*. $\lceil\log_2 2708\rceil = 12$.
   Llama-3.2-1B has 16 layers; Qwen3-1.7B has 28; Llama-3.1-8B has 32; Qwen3-8B has 36. **Every
   candidate clears it.** This is a genuinely reassuring result and worth stating in the paper:
   we are trading a width deficit for a depth surplus, which is precisely the tradeoff this paper
   is about — just read in the opposite direction from theirs.

### The mitigations, ranked

1. **Subgraph sampling — the cleanest fix, and CLAUDE.md already mandates it for memory.**
   §11 says "cap at 500-1,000 nodes per batch". At $N \le 1000$: $3N = 3000$ ✳ still above 2048 but
   below 4096; at $N \le 682$, $3N \le 2048$ and **Llama-3.2-1B is in Theorem 4.3's linear-width
   regime outright**. Recommendation: **sample k-hop subgraphs of ≤ 680 nodes for the 1B model,
   ≤ 1360 for the 8B**, and say in the paper that this puts us in the regime the theory covers.
   This single decision converts "we are theoretically under-width" into "we are theoretically
   in-regime", at zero cost, and it is required anyway — full Cora attention is
   $2708^2 = 7.3\text{M}$ entries per head per layer.

2. **Attention bias (our core contribution).** Carries $A$ outside the token budget. Do the
   Theorem-4.3 sanity test above so the mechanism is verified, not asserted.

3. **E5 with a *normalized* spectral block.** Confirmed necessary by Appendix A (adjacency rows are
   blind to connectivity) and confirmed *dangerous if done naively* by 2605.22471 Theorem 5 (raw
   eigenvalues ⇒ $\|X\|^2=\Omega(n^2)$ ⇒ softmax saturation or weight explosion). Use symmetric
   normalized $\mathcal{L}$, drop the trivial eigenvector, random sign flips, standardize.
   **Feeding an un-normalized spectral block into a frozen LLM is asking for a dead run.**

4. **E3 sketch at $k=512$ + node codeword.** Decouples from $N$ entirely. Do not oversell the AGM
   connection.

### One more consequence CLAUDE.md should absorb

Their **Section 6.1** finding — at a fixed ~100k parameter budget, (depth, width) = (1,125) matches
(10,40) on loss and accuracy while training and running **much faster** — is an argument for
**not** reaching for the 8B model first. If width is what we need and depth is already surplus,
Qwen3-4B ($d{=}2560$, $L{=}36$) is a better width-per-FLOP trade than Llama-3.2-1B ($d{=}2048$,
$L{=}16$), and gets us to $2d\log_2 N$ coverage at degree ~110 instead of ~89.

---

## Open questions

1. **Does the attention-bias path actually escape Theorem 4.2's communication lower bound?** My
   argument above is suggestive, not a proof, and the set-disjointness reduction may well extend. If
   it does escape, that is a theorem worth writing; if it does not, we need to know before we claim
   the architecture "solves" the width problem. Someone should try to construct the two-party
   protocol with a bias term.
2. **Is there a version of Theorem 4.3 for bias-injected transformers?** i.e. "an $O(L)$-layer
   transformer with $\text{bias}(i,j)=cA_{ij}$ and embedding dimension $m=O(?)$ outputs row $i$ of
   $A^L$ at token $i$". Their proof needs width $n$ only to *hold* $A$ and the accumulator; with the
   bias supplying $A$, the accumulator is still $n$-wide. So probably $m=O(N)$ not $3N$ — a 3×
   saving, still not $\ll N$. Worth working out.
3. **Their code.** Genuinely unobtainable from here (OpenReview 403). If someone has an OpenReview
   login, pull `openreview.net/attachment?id=A2pmNL7L1E&name=supplementary_material` and diff it
   against the reimplementation above — chiefly to learn the optimizer, the §6.3 head count, and the
   pooling.
4. **Why is molhiv the outlier?** LE beats AdjRows by 6.2 points on molhiv but loses by 12.3 on
   molbbbp. Neither paper explains it. If we run the mol datasets, this is a cheap ablation with a
   real finding in it.
5. **Does the hub-node hypothesis hold?** Theorem 4.4 predicts error should concentrate on
   high-degree nodes when $d_{model} < 2d\log_2 N$. Testable on Cora in an afternoon: bucket AUPRC by
   node degree, compare Llama-3.2-1B (2048) vs Llama-3.1-8B (4096). If the prediction holds it is a
   direct empirical validation of a theory paper — a nice thing to have in a results section.
6. **Zero-padding to $n_{\max}$ for Cora is meaningless** (single graph, fixed $N$) but matters for
   the molecular phase and for the biomedical phase. Their padding convention pads *both* the row
   dimension and the token count; the follow-up excludes pads from attention via
   `src_key_padding_mask`. Make sure our loader does both, or padded columns will leak into the
   encoder as real zeros.
7. **`Comb` tokenization.** The follow-up's best results on 3 of 8 tasks come from concatenating all
   three tokenizations. Our E5 is a 2-way version. Should there be an E5b = `row ‖ Lap ‖ RW ‖ X`?
   The evidence says yes.

---

## Sources fetched

Primary, read in full as text:

- `https://arxiv.org/abs/2503.01805` — abstract, authors, version history
- `https://arxiv.org/html/2503.01805v1` — downloaded, HTML-stripped, read in full (Sections 3-7,
  Appendices A-E, proofs B.1-B.4)
- `https://arxiv.org/html/2503.01805v3` — downloaded, HTML-stripped, read (Sections 4-6, Appendix E,
  NeurIPS checklist); diffed against v1
- `https://ar5iv.labs.arxiv.org/html/2503.01805` — cross-check of theorem statements
- `https://arxiv.org/pdf/2503.01805v1` — fetched (4.9 MB) but unreadable via WebFetch (compressed
  PDF streams); superseded by the HTML
- `https://arxiv.org/html/2605.22471v1` — "Lost in Tokenization", downloaded, HTML-stripped, read
  (Sections 3-5, Appendix A.6-A.7, Tables 1, 3, 4, 5)
- `https://arxiv.org/abs/2405.18512` — Sanford, Fatemi, Hall, Tsitsulin, Kazemi, Halcrow, Perozzi,
  Mirrokni, "Understanding Transformer Reasoning Capabilities via Graph Algorithms" (the edge-list /
  depth-hierarchy predecessor)
- `https://neurips.cc/virtual/2025/poster/119487` — venue: NeurIPS 2025 Spotlight Poster
- `https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/ind.cora.graph` — the raw Planetoid
  Cora adjacency; **degree statistics and `pos_weight` above were computed from this file**, not
  quoted from anywhere
- `https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json` (and `-1.7B`, `-4B`, `-8B`)
- `https://huggingface.co/unsloth/Llama-3.2-1B/resolve/main/config.json`
- `https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/resolve/main/config.json`
  (meta-llama/* are gated → 401; the mirrors carry byte-identical configs)
- `https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html`
  — Cora 2708 nodes / 10,556 edges / 1433 features / 7 classes; `Planetoid` signature

Attempted and **blocked** (recorded so nobody repeats the work):

- `https://openreview.net/forum?id=A2pmNL7L1E` — Cloudflare browser-verification interstitial
- `https://api2.openreview.net/notes?forum=A2pmNL7L1E` — `ChallengeRequiredError`, HTTP 403
- `https://openreview.net/attachment?id=A2pmNL7L1E&name=supplementary_material` — **HTTP 403**
- `https://openreview.net/attachment?id=A2pmNL7L1E&name=pdf` — HTTP 403
- GitHub search for the authors' code — **no repository found**

Secondary (used only where flagged, never as the basis for a "verified" claim):

- Web search for Cora max degree (168 vs 169 across preprocessing variants) — **superseded** by the
  direct computation from `ind.cora.graph`, which gives **168**
- Web search for Ahn, Guha & McGregor, SODA 2012, "Analyzing graph structure via linear
  measurements" — confirms $d=O(n\,\mathrm{polylog}\,n)$ sketch space for connectivity,
  $k$-connectivity, bipartiteness, MST approximation; consistent with the DepthWidth paper's own
  characterization, which is what I quoted
