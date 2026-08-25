# Competitors — what we reject, and what we must beat

Research notes for the "Graph-In / Graph-Out LLM" project. Every claim below was checked
against a primary source that I fetched myself. Where I could not verify something, it says so.

Date of research: 2026-08-25.
Scope: (A) verbalization, (B) GNN soft prompts, (C) work closer to us than CLAUDE.md admits,
(D) published numbers on Cora.

---

## Verified facts (with source next to each)

### A. VERBALIZATION — "Talk like a Graph" (TLAG)

| Fact | Source |
|---|---|
| Title *Talk like a Graph: Encoding Graphs for Large Language Models*; Bahare Fatemi, Jonathan Halcrow, Bryan Perozzi; arXiv **2310.04560**, submitted 2023-10-06 | fetched `arxiv.org/abs/2310.04560` |
| It **is** an ICLR 2024 conference paper ("Published as a conference paper at ICLR 2024") | `proceedings.iclr.cc/paper_files/paper/2024/file/bf72f65f30eedf5d48da6980ee02b589-Paper-Conference.pdf` |
| Primary LLM is **PaLM 62B**; a capacity study uses **PaLM 2 XXS / XS / S / L** | TLAG PDF p.5 Table 1 caption, p.6 Table 2, p.7 §3.4 |
| **9 graph encoder functions**: `adjacency`, `incident`, `friendship`, `co-authorship`, `social network`, `politician`, `SP` (South Park), `GOT` (Game of Thrones), `expert` | TLAG PDF Fig. 2 p.4; confirmed in code `talk_like_a_graph/graph_text_encoders.py` `EDGE_ENCODER_FN` |
| **6 tasks in the main table**: edge existence, node degree, node count, edge count, connected nodes, cycle check. A 7th, **disconnected nodes**, is studied separately in §3.5 | TLAG PDF p.4 §3.1, p.7 §3.5 |
| **5 prompting heuristics**: ZERO-SHOT, ZERO-COT, FEW-SHOT, COT, COT-BAG | TLAG PDF p.3 §2.2, Table 1 |
| Graphs come from **ER, BA, SBM, star, path, SFN (scale-free), complete** generators via NetworkX | TLAG PDF p.8 §4.1, Table 4 |
| Abstract's "**4.8% to 61.8%**" is the spread attributable to *choice of encoder*, not an absolute score | TLAG abstract; consistent with Table 1 δ column |

**The killer numbers for our positioning** (TLAG PDF p.5, Table 1, PaLM 62B, format `overall mean μ / spread δ`):

```
                   Edge Exist.  Node deg   Node count  Edge count  Conn. nodes  Cycle chk
ZERO-SHOT overall   44.5/ 9.4   14.0/16.0  21.73/ 8.6  12.4/ 4.8   14.7/11.0   76.0/13.2
  best encoder      49.0 (GOT)  25.0 (Inc) 24.2 (Pol)  16.4 (Inc)  53.8 (Inc)  82.0 (Frnd)
ZERO-COT overall    33.5/11.6   10.4/22.4  14.6/ 9.4    9.4/ 4.8    8.8/ 9.2   32.3/23.2
FEW-SHOT overall    36.8/13.8   17.4/23.4  25.3/35.6   12.0/ 9.0   12.4/15.2   37.4/24.0
COT overall         42.8/ 7.0   29.2/60.4  27.6/42.4   12.8/17.4   13.1/18.0   58.0/16.4
  best encoder      46.6 (Frnd) 75.0 (Inc) 57.6 (Inc)  25.2 (Adj)  22.4 (Adj)  62.6 (Inc)
COT-BAG overall     37.3/16.6   28.0/61.8  26.9/37.8   12.5/17.8   15.8/31.8   52.1/26.0
  best encoder                                                     41.0 (Inc)
```

Three documented failure modes, quoted from the paper:

1. **Worse than the majority-class baseline on the task closest to ours.**
   > "there is not an edge 53.96% of the time for the edge existence task and there is a cycle
   > 81.96% of the time for the cycle check task. Therefore, LLMs perform worse than the majority
   > baseline." — TLAG PDF p.4 §3.1.1

   Best edge-existence accuracy anywhere in Table 1 is **49.0%** (GOT, zero-shot). Majority baseline
   is 53.96%. **Verbalization cannot beat "always say no" on edge existence.** Edge existence is
   exactly our masked-edge-prediction task. This is the single most useful sentence in the paper for us.

2. **No global model of the graph.**
   > "The ZERO-SHOT prompting method achieved an accuracy of 0.5%, while the ZERO-COT, FEW-SHOT, COT,
   > and COT-BAG methods achieved close to 0.0% accuracy" on *disconnected nodes*. "LLMs lack a global
   > model of a graph." — TLAG PDF p.7 §3.5

   Directly relevant to CLAUDE.md §11's "adjacency rows are local-biased" warning: verbalization has
   the *same* blind spot, and worse.

3. **Node naming and graph shape swing the answer.** Integer node names help integer-output tasks
   (degree/count); real-person or fictional-character names help non-integer tasks (GOT best for edge
   existence, Friendship best for cycle check). Structure matters too: cycle check is 91.7% on complete
   graphs and 5.9% on path graphs (TLAG PDF p.8 Table 4) — i.e. the model answers from a prior, not
   from the graph. "Distractive statements in the graph encoding function disrupt the performance."

**Graph size.** GraphQA graphs are tiny. The GraphToken follow-up evaluates on "all 8-node graphs" and
"all tree graphs with 15 nodes". *Caveat:* I did not find an explicit "between 5 and 20 nodes" sentence
in the PDF pages I read (pp. 3–8); an ar5iv render reported that string but I could not confirm it in
the PDF, so treat the exact range as unverified. **What is verified is that the graphs are single- to
low-double-digit node counts, i.e. ~100× smaller than Cora.**

**Token cost — computed, not quoted.** TLAG never reports token counts (verified: no such discussion
in pp. 3–8). I reconstructed it by running their own `adjacency_encoder` / `incident_encoder` string
formats on a Cora-sized graph (N=2708, E=5278):

```
adjacency (edge-list) encoding : ~79,500 chars ~= 22,700-31,800 tokens
incident encoding              : ~155,000 chars ~= 44,300-62,000 tokens
our scheme (1 token per node)  :                     2,708 tokens
                                 -> 8.4x - 16.4x blowup, and that is for a SPARSE graph
full dense N^2 verbalization   : 7,333,264 entries  (hopeless)
```
(script at `/tmp/blowup.py`; token estimate at 2.5–3.5 chars/token for digit-heavy text — **this is an
estimate, not a measured tokenizer count**. Re-measure with the real Llama tokenizer before quoting it
in a paper.)

Asymptotics to state in the writeup: edge-list verbalization is **O(E)** tokens, incident-list is
**O(N + E)**, dense-matrix verbalization is **O(N²)**; adjacency-row tokenization is **O(N)** tokens of
width N. On Cora specifically the constant factor is ~8–16×, not the "O(N²)" that CLAUDE.md implies for
the *sparse* case — see Corrections.

**Honest caveat on "beating TLAG":** TLAG trains nothing. It is a prompting study on a frozen PaLM.
Beating its numbers is close to free and reviewers know it. Use it to establish that
*verbalization + no training* fails, not as a competitive baseline.

---

### B. GNN SOFT PROMPTS

#### B1. GraphToken — "Let Your Graph Do the Talking" (arXiv 2402.05862)

| Fact | Source |
|---|---|
| Authors Perozzi, Fatemi, Zelle, Tsitsulin, Kazemi, Al-Rfou, Halcrow; submitted 2024-02-08 | `arxiv.org/abs/2402.05862` |
| **LLM is fully frozen**: "we freeze its parameters and teach the graph encoder to align"; "keeping all LLM parameters frozen" | arXiv HTML v1 §3, §3.2 |
| LLM is the **instruction-fine-tuned Flan checkpoint of PaLM 2 S** | arXiv HTML v1 |
| Augmented query: **Q = E(G) ‖ T(T)** — graph encoding concatenated *before* the text prompt embedding | arXiv HTML v1 §3 |
| Loss: **optimise LLM perplexity L(A\|Q)** of the expected answer A given the augmented query | arXiv HTML v1 §3 |
| **Latent size 128**; optimizer **Lion**, learning rate **α = 0.05** | arXiv HTML v1 |
| Readouts: graph-level = mean/sum global pool; **node-level = one representation per node**; edge-level = global rep or the two node reps concatenated | arXiv HTML v1 §3 |
| **7 encoders, and two of them are NOT GNNs**: GCN, GIN, MPNN, HGT, MHA (graph transformer), **NodeSet**, **EdgeSet** | arXiv HTML v1 Tables 2, 3 |

Parameter counts (Table 3, verbatim):

```
Model      Body      Head
GCN        17,152    1.1e7
GIN        17,152    1.1e7
MPNN       83,968    1.1e7
HGT       198,788    1.1e7
MHA       101,376    1.1e7
Node Set        0    4.1e5
Edge Set        0    7.4e5
```

Main results (Table 1, GraphQA test set — **accuracy**):

```
Method       NodeCnt EdgeCnt Cycle  Triangle NodeDeg ConnNodes Reach  EdgeExist ShortPath
Zero-shot     0.217   0.124  0.760   0.015    0.140    0.147   0.849    0.445     0.115
Few-shot      0.253   0.120  0.374   0.030    0.174    0.124   0.794    0.368     0.227
CoT           0.276   0.128  0.580   0.081    0.292    0.131   0.452    0.428     0.386
Soft-prompt   0.056   0.018  0.832   0.162    0.098    0.068   0.838    0.544     0.462
GraphToken    0.996   0.426  0.956   0.348    0.962    0.264   0.932    0.738     0.638
```

Per-encoder (Table 2) — note the non-GNN encoders are competitive:

```
Encoder   NodeCnt EdgeCnt Cycle  Triangle NodeDeg ConnNodes Reach  EdgeExist ShortPath
GCN        0.746   0.056  0.964   0.208    0.264    0.264   0.918    0.680     0.604
GIN        0.704   0.052  0.898   0.194    0.252    0.180   0.902    0.650     0.586
MPNN       0.792   0.368  0.956   0.348    0.962    0.250   0.934    0.648     0.638
HGT        0.252   0.084  0.934   0.234    0.266    0.184   0.944    0.718     0.600
MHA        0.912   0.264  0.962   0.266    0.552    0.244   0.932    0.738     0.608
NodeSet    0.996   0.080  0.948   0.198    0.190    0.118   0.942    0.596     0.568   <- 0 body params
EdgeSet    0.618   0.426  0.964   0.228    0.220    0.096   0.904    0.592     0.568   <- 0 body params
```

**Read this carefully before writing our novelty claim.** NodeSet (zero message passing, zero body
parameters, a pure linear set encoder) gets the *best* node-count score in the whole paper (0.996) and
EdgeSet the *best* edge-count score (0.426). Our E6 "DeepSets, not a GNN" encoder is very close to
GraphToken's NodeSet. GraphToken already ran that ablation. We must not claim "no one has tried a
non-message-passing encoder into a frozen LLM" — they have, in this exact paper.

**Number of soft tokens: NOT stated.** The paper says only "outputs a fixed number of token embeddings".
For node-level tasks the readout is per-node, so it is effectively one token per node — the same budget
as ours. **I could not find an explicit count anywhere in the HTML.** Do not cite a number.

#### B2. GraphGPT (Tang et al., SIGIR 2024, arXiv 2310.13023)

| Fact | Source |
|---|---|
| Graph tokens are inserted as `{<graph_begin>, <graph_token>_1, …, <graph_token>_n, <graph_end>}`, where **n = number of nodes in the sampled subgraph** | GraphGPT PDF p.4 §3.2.1 |
| Projector `f_P` "can be as simple as a **single linear layer**" | GraphGPT PDF p.4 |
| Graph encoder = **pretrained graph transformer** (message passing, eq. 2 `H^(l) = σ(Ã H^(l-1) W)`); text encoder = vanilla transformer / BERT | GraphGPT PDF p.3 §3.1 |
| **LLM is frozen in BOTH stages.** Stage 1: "keeping the parameters of both the LLM and the graph encoder fixed". Stage 2: "we keep the parameters of the language model (LLM) and graph encoder fixed, focusing solely on optimizing the parameters of the projector". Ablation §4.4: "this improvement in performance was achieved **without altering the original parameters of the LLM**. Instead, it was solely accomplished through our lightweight alignment projector … through the 1-linear projection operation." | GraphGPT PDF pp.4, 7 |
| Base LLM **Vicuna-7B-v1.1 / v1.5**; batch 2/GPU, **lr 2e-3**, warmup 3e-2, max input 2048, 3 epochs (stage 1), 2 epochs (stage 2) | GraphGPT PDF p.6 §4.1.4 |
| Cora here is an **expanded 70-class, 25,120-paper version**, split 3:1:1 — *not* standard Planetoid Cora | GraphGPT PDF p.5 §4.1.1 |
| Metrics: Accuracy + Macro-F1 for node classification, **AUC for link prediction** | GraphGPT PDF p.5 §4.1.2 |

Text-graph grounding contrastive objective (eqs. 3–4, p.3):

```
H = f_G(X),  T = f_T(C),  Ĥ = norm(H),  T̂ = norm(T)
Γ1 = (Ĥ T̂ᵀ)·exp(τ)    Γ2 = (Ĥ T̂'ᵀ)·exp(τ)    Γ3 = (T̂ T̂'ᵀ)·exp(τ)
L  = Σ_{i=1..3} ½ λ_i [ CE(Γ_i, y) + CE(Γ_iᵀ, y) ]
where T̂'_i = (1/|N_i|) Σ_{j∈N_i} T̂_j  and  y = (0,1,…,n-1)ᵀ
```

Generation objective (eq. 5): `p(X_O | X_G, X_T) = Π_{i=1..L} p_θ(x_i | X_G, X_{T,<i}, X_{O,<i})`

Node classification (Table 1, accuracy / Macro-F1):

```
                          Arxiv-Arxiv      Arxiv-PubMed     Arxiv-Cora(70cls)
MLP                     0.5179 / 0.2536   0.3940 / 0.1885   0.0258 / 0.0037
GCN                     0.5267 / 0.3202   0.3940 / 0.1884   0.0214 / 0.0088
GAT                     0.5332 / 0.3118   0.3940 / 0.1884   0.0167 / 0.0110
GraphSAGE               0.5480 / 0.3290   0.3950 / 0.1939   0.0328 / 0.0132
NodeFormer              0.5922 / 0.3328   0.2064 / 0.1678   0.0152 / 0.0065
vicuna-7b-v1.5          0.4962 / 0.1853   0.6351 / 0.5231   0.1489 / 0.1213
GraphGPT-7B-v1.5-stage2 0.7511 / 0.5600   0.6484 / 0.5634   0.0813 / 0.0713
GraphGPT-7B-v1.5-std    0.6258 / 0.2622   0.7011 / 0.6491   0.1256 / 0.0819
GraphGPT-7B-v1.5-cot    0.5759 / 0.2276   0.5213 / 0.4816   0.1813 / 0.1272
```

**Link prediction on PubMed with real AUC/AP** (Table 3, p.7) — the only genuine AUC numbers I found
from a soft-prompt LLM method:

```
                                     AUC      AP
MLP                                 0.5583  0.5833
GAT                                 0.5606  0.6373
GraphSAGE                           0.5041  0.5813
RevGNN                              0.4538  0.5083
Node2Vec                            0.6535  0.6885
w/o GS (no graph structure)         0.5010  0.5005
only Link                           0.6704  0.6087
Arxiv-std + PubMed-std + Link       0.8246  0.8026   <- GraphGPT best
Arxiv-mix + PubMed-mix + Link       0.6451  0.5886
```

**Compare to GAE on PubMed: 96.4 AUC / 96.5 AP (Kipf & Welling 2016).** The best LLM-soft-prompt link
predictor in the literature is **~14 AUC points behind a 2016 two-layer GCN autoencoder.** This is the
strongest single argument in the whole competitor landscape for why the graph-out side is unsolved.

#### B3. LLaGA (Chen, Zhao, Jaiswal, Shah, Wang — arXiv 2402.08170, v3 2024-04-11)

| Fact | Source |
|---|---|
| **No GNN in the Neighborhood Detail template**: "this transformation is **parameter-free**, ensuring the preservation of the original structural integrity without necessitating further distillation" | LLaGA PDF p.2 §2 |
| **LLM is frozen**: "the parameters θ of the projector are the only parameters subject to tuning during the training process of LLaGA"; Fig. 1 labels the LLM "Freezed LLM", the Projector "Tuned" | LLaGA PDF pp.3–4, Fig. 1 |
| Projector is "a **simple MLP**" | LLaGA PDF p.4 |
| Base LLM **Vicuna-7B-v1.5-16K**; text encoder **SimTeG**; **lr 2e-5**, batch 16, **1 epoch**, Cora replicated 3× | LLaGA PDF p.5 §3.1 |
| ND template: **2 hops, sample size 10 per hop** → tree of 1 + 10 + 100 = **111 tokens per centre node**. HO template: **4 hop embeddings** → 4 tokens | LLaGA PDF p.5 + Fig. 1 |
| Link prediction metric is **Accuracy on a yes/no question**, NOT AUC | LLaGA PDF p.5 "Evaluation Metrics" |

Laplacian positional embedding (eq. 1) and node embedding (eq. 2):

```
L = I − D^{-1/2} A_tree D^{-1/2} = Uᵀ Λ U          # computed ONCE, tree shape is fixed
h_{v_i} = 0 ‖ U_i                  if v_i = [pad]
        = φ(x_{v_i}) ‖ U_i         otherwise
```
Hop-field embedding (eq. 3): `h_v^i = (1/|N_v^1|) Σ_{v'∈N_v^1} h_{v'}^{i-1}`, `h_x^0 = φ(x_v)`
Projection (eq. 4): `e_i = f_θ(h_i)`.  Objective (eq. 5): `max_θ p(X_answer | X_graph, X_question, X_system)`

**Table 1 (accuracy %) — the numbers we will be compared against on Cora:**

```
Setting          Model          NC-Arxiv NC-Prod NC-PubMed NC-Cora | LP-Arxiv LP-Prod LP-PubMed LP-Cora
Single Focus     GCN             73.72   80.75    92.96    88.93   |  91.43   93.95    90.91    81.59
                 GraphSage       76.29   82.87    94.87    88.89   |  91.64   94.96    90.64    79.15
                 GAT             74.06   83.06    92.33    88.97   |  85.99   93.85    83.96    80.06
                 SGC             71.77   75.47    87.35    87.97   |  87.99   88.51    83.60    80.94
                 SAGN            75.70   82.58    95.17    89.19   |  90.62   94.85    90.48    79.88
                 NodeFormer      74.85   83.72    94.90    88.23   |  91.84   90.93    77.69    77.26
                 LLaGA-ND-7B     75.98   84.60    95.03    88.86   |  91.24   97.36    91.41    83.79
                 LLaGA-HO-7B     76.66   84.67    95.03    89.22   |  94.15   95.56    89.18    86.82
Task Expert      LLaGA-ND-7B     76.41   84.60    94.78    88.19   |  94.36   97.38    93.27    89.41
                 LLaGA-HO-7B     76.40   84.18    95.06    89.85   |  94.36   95.85    88.88    87.50
Classif. Expert  LLaGA-ND-7B     75.85   83.58    95.06    87.64   |  90.81   96.56    92.36    87.35
                 LLaGA-HO-7B     75.99   83.32    94.80    89.30   |  94.30   96.05    88.64    88.53
General Model    GPT3.5-TURBO    55.00   75.25    88.00    71.75   |  63.80   60.30    68.70    65.74
                 LLaGA-ND-7B     74.29   82.21    92.42    87.82   |  90.53   96.82    86.31    81.91
                 LLaGA-HO-7B     75.01   82.07    94.45    87.82   |  92.04   86.80    89.81    84.41
```
Templates ablation (Table 6): no template → Cora NC 84.50 / LP 83.97; ND → 87.64 / 87.35; HO → 89.30 / 88.53.
Zero-shot LP (Table 5): Arxiv+PubMed → Cora: GCN 58.97, GraphSage 67.68, GraphGPT-7B 50.74,
LLaGA-ND 86.47, LLaGA-HO 87.35.

#### B4. GraphPrompter — "Can we Soft Prompt LLMs for Graph Learning Tasks?" (arXiv 2402.10359)

- GNN = **GAT** over a **3-hop subgraph**; the node embedding `X_i` is projected by an **MLP** into the
  LLM vector space and concatenated as `[X̂_i, T_emb]`.
- Backbone **LLaMA-2-7B**, frozen; LoRA in the enhanced variants; GNN + projection trained end-to-end.
- **Link prediction reported as accuracy, not AUC.**

```
Node classification acc (%)     Cora   Citeseer PubMed  Arxiv  Products
GAT                             84.69   70.78   84.09   71.82   70.52
Zero-Shot                       43.31   29.22   91.39   44.23   15.05
Soft Prompt Tuning              70.31   70.97   91.45   71.99   75.14
Fine-tuning + LoRA              75.97   73.45   94.68   74.58   78.99
GraphPrompter + LoRA            80.26   73.61   94.80   75.61   79.54

Link prediction acc (%)         Cora   Citeseer PubMed  Arxiv  Products
GAT                             90.71   87.40   86.18   72.93   65.67
Fine-tuning + LoRA              87.58   87.91   81.33   70.10   67.31
GraphPrompter + LoRA            90.10   91.67   86.49   73.21   69.55
Subgraph Prompt Tuning          89.15   93.49   87.20   75.28   68.18
```
Note **GraphPrompter does not beat plain GAT on Cora link prediction (90.10 vs 90.71) or on Cora node
classification (80.26 vs 84.69).** Another argument that the LLM is not currently earning its keep.

#### B5. InstructGLM — "Language is All a Graph Needs" (Ye, Zhang, Wang, Xu, Zhang; arXiv 2308.07134, EACL 2024 Findings)

**This is verbalization, not a soft prompt, and its LLM is NOT frozen.** CLAUDE.md does not name it,
but the task brief did, and it matters (see Corrections).

- Prompts are natural language: *"`<node_1>`, Title_1 is connected to `<node_2>`, Title_2 … within one
  hop. Which category should `<node_1>` be classified as?"* (Fig. 2, p.5).
- Backbones **Flan-T5-base / Flan-T5-large (full instruction tuning)** and **Llama-7B (LoRA)** — trained,
  not frozen (p.8 §4.2.2).
- **It expands the LLM vocabulary with one new token per node and initialises those token embeddings
  with the graph's raw node feature vectors (BoW / OGB features)** — Fig. 2 caption, p.5. This is the
  closest thing in the literature to putting non-text vectors into an LLM's embedding table.
- Objective: `L_θ = −Σ_j log P_θ(y_j | x, y_<j)`, `x = Concatenate(P; I; Q)` (p.6).
- Link prediction is only an **auxiliary self-supervised task**; no standalone LP metric is reported.

```
Cora & PubMed node classification accuracy (Table 2, 60/20/20 random split)
                           Cora           PubMed
MixHop                   75.65 ± 1.31   90.04 ± 1.41
GAT                      76.70 ± 0.42   83.28 ± 0.12
Geom-GCN                 85.27 ± 1.48   90.05 ± 0.14
GraphSAGE                86.58 ± 0.26   86.85 ± 0.11
GCN                      87.78 ± 0.96   88.90 ± 0.32
GCNII                    88.93 ± 1.37   89.80 ± 0.30
RevGAT                   89.11 ± 0.00   88.50 ± 0.05
ACM-GCN+                 89.75 ± 1.16   90.96 ± 0.62
Graphormer               80.41 ± 0.30   88.24 ± 1.50
GT                       86.42 ± 0.82   88.75 ± 0.16
CoarFormer               88.69 ± 0.82   89.75 ± 0.31
InstructGLM Llama-7b     87.08 ± 0.32   93.84 ± 0.25
InstructGLM Flan-T5-base 90.77 ± 0.52   94.45 ± 0.12
InstructGLM Flan-T5-large 88.93 ± 1.06  94.62 ± 0.13

ogbn-arxiv (Table 1, OGB split 54/18/28, OGB features)
GAT 73.66 ± 0.11 | RevGAT 74.02 ± 0.07 | DRGAT 74.16 ± 0.07 | Graphormer 72.81 ± 0.23
InstructGLM Flan-T5-base 73.51 ± 0.16 | Flan-T5-large 74.67 ± 0.08 | Llama-7b 75.70 ± 0.12
```
**Beware the split.** GAT scores 76.70 on Cora here (60/20/20 random) versus 83.0 in the GAT paper
(Planetoid 20-labels-per-class). Numbers across these papers are **not** mutually comparable.

---

### C. Work CLOSER to us than CLAUDE.md admits

This is the section our novelty claim lives or dies on. I ran targeted arXiv full-text API queries as
well as web search. **Negative results are informative and I list them.**

#### C1. What I could NOT find (good news, stated honestly)

arXiv full-text API queries returning **zero hits**:
```
all:"adjacency rows"   AND all:"language model"          -> 0
all:"adjacency matrix" AND all:"frozen LLM"              -> 0
all:"adjacency matrix" AND all:"frozen language model"   -> 0
all:"node-adjacency tokenization"                        -> 0
abs:"adjacency" AND abs:"token" AND abs:"pretrained LLM" -> 0
abs:"adjacency" AND abs:"LoRA" AND abs:"attention bias"  -> 0
```
Two independent 2026 sources also come up empty on this:
- *Revisiting Graph-Tokenizing LLMs* (arXiv 2605.03514; Zhang, Yu, Zhang, Du, Wang, Shi; 2026-05-05)
  surveys LLaGA, InstructGLM, GraphGPT, GraphTranslator, TEA-GLM, GOFA — **all use either verbalization
  or a GNN/encoder-produced embedding. None feeds raw adjacency.**
- *Are LLMs Suitable for Graph Computation? Progress and Prospects* (arXiv 2606.06865) taxonomy:
  textual/verbalization, GNN-encoder soft prompts, structural encodings, attention-mask modification.
  Asked directly, it lists **no** method that feeds unverbalized raw matrix rows to a frozen LLM, and
  **no** method that decodes an adjacency matrix out of an LLM.

**Conclusion: the combination "raw adjacency row → learned projection → pretrained LLM → adjacency
matrix out" appears genuinely unoccupied as of Aug 2026.** But the two halves separately are not.

#### C2. What eats part of our novelty (the honest accounting)

**(i) The attention-bias half is well-trodden.** CLAUDE.md already names GTLM and GaLA. There are more:

| Work | What it does | How much it eats |
|---|---|---|
| **Graph Language Models** (Plenz & Frank, ACL 2024, arXiv 2401.07105) | Takes pretrained **T5** and turns it into a graph transformer: `softmax(QKᵀ/√d + B_P + M)V` where **M** is a graph-derived attention mask and **B_P** is T5's relative-position bias table re-indexed by *graph* distance instead of sequence distance. ℓGLM masks to within-triplet; gGLM allows all-pairs. **Keeps T5's tokenizer and 32,128-entry embedding table. Finetuned, not frozen. Outputs classifications, never a graph.** ConceptNet relation classification: gGLM linear-probe 59.5%, finetuned 65.3% vs T5 ~55%. Wikidata relation F1 85.28 vs T5 85.04. | Eats the "reuse the pretrained position-bias machinery for graph distance" idea. **Predates GTLM by 16 months.** We must cite it — a reviewer who knows the ACL literature will. |
| **UniGTE** (Wang, Zuo, Lu, Wu — NeurIPS 2025, arXiv 2510.16885) | `Â_ij = [(x_i W_Q) R(i−j) (x_j W_K)ᵀ]/√d_k + b(i,j)` with `b(i,j) = 1_{i,j≤n}(b^PE_ij + b^Edge_ij) + b^M_ij`; `b^PE` is a **shortest-path lookup table** (= our hop bias), `b^Edge` an MLP over edge descriptions along the SP, `b^M` a masking bias making graph tokens bidirectional and blocking text→graph attention. Encoder = **Vicuna-7B + LoRA**; decoder = **frozen Vicuna-7B**. **64 learnable alignment tokens** as the only interface. Claims **permutation invariance to node order**. Tokens are node-*attribute text* encoded by BERT. Decoder "reconstructs" the graph — **as a natural-language paraphrase, not an adjacency matrix.** | Eats: (a) SPD-lookup attention bias in a LoRA'd LLM, (b) permutation-invariance claim, (c) the *reconstruction-as-regularizer* framing, (d) the encoder→frozen-LLM→decoder shape. **This is the single biggest novelty threat and CLAUDE.md does not mention it.** What it does NOT do: raw adjacency in, adjacency out. Its 64-token bottleneck is exactly the "compressive bottleneck" GTLM criticises. |
| **GL-Fusion** (arXiv 2412.06849) | "Structure-Aware Transformer Layers" inside **Llama-3-8B**: graph-aware attention mask (nodes see all nodes in the graph + prior text tokens, text stays causal) + in-layer **message passing** `COMBINE(h_u, AGGREGATE({h_v ⊙ e_uv : v ∈ N(u)}))` with mean/max/std aggregators, gated by `tanh(W t_i) h_u` initialised to zero. LoRA rank 4–64; "only about 10% of the parameters in the entire model are optimized". Each node gets a `<node>` token from its **text attributes**; Graph-Text Cross-Attention reads the uncompressed node text. Outputs text (LLM head) and classifications (GNN head) — **not a graph**. ogbn-arxiv 78.20%, Cora 84.3%. | Eats "modify a pretrained LLM's attention with graph structure and keep most weights frozen". But it is explicitly a **message-passing** design — the thing CLAUDE.md §13 forbids. Good contrast for us. |
| **HLM-G** (arXiv 2410.22372) | Two-block hierarchical **attention masking** in an LLM: a local block over per-node descriptions, a global block over interactions. Reduces cost from O((Σn_i)²) to O(Σn_i²). Still operates on **verbalized** node descriptions. | Minor. Cite as prior art for "structured attention masks over graph text". |

**(ii) The frozen-LLM-for-a-non-text-modality half is old.** CLAUDE.md already cites FPT
(Lu et al. 2103.05247), which is the right citation. Note the time-series literature is a second,
independent precedent for the *exact* recipe "linear projection in → frozen pretrained transformer →
linear projection out, only the two projections + layernorm trained": **GPT4TS / "One Fits All"** and
**Time-LLM** (ICLR 2024). Use them to pre-empt "why would a text LLM process a matrix row?" — the
answer is that a whole subfield already does exactly this for numeric sequences.

**(iii) One-token-per-node is not new; a *raw-adjacency*-derived token is.** GraphToken's node-level
readout emits one soft token per node; LLaGA emits 111 (ND) or 4 (HO) per centre node; GraphGPT emits
n tokens for an n-node subgraph. **Token budget is not our differentiator. The provenance of the token
is.** Frame the claim as: *what goes into the token is the raw adjacency row, not a message-passing
summary and not a text embedding.*

**(iv) Graph generation from token streams exists — but from scratch, not from a pretrained LLM.**
G2PT (already in CLAUDE.md) and **AutoGraph** ("Flatten Graphs as Sequences: Transformers are Scalable
Graph Generators", NeurIPS 2025) both train decoder-only transformers from scratch on flattened graph
token sequences. Neither uses pretrained language weights. **No published work decodes an adjacency
matrix out of a pretrained LLM's hidden states.** That half of our claim survives intact.

**(v) Near-miss checked and cleared: `<SOG_k>`** (Wu et al., arXiv 2602.01771, 2026-02-02) maps a whole
graph structure to **one discrete vocabulary token** via a GNN + vector quantization over the LLM's
codebook, then LoRA-tunes only the new token embeddings while freezing pre-existing text tokens.
Molecular graph classification (BACE 98.4 AUC-ROC with LLaMA2-7B), Cora/PubMed node classification
91.58 acc at 3B. **It is a GNN encoder, it uses the vocabulary/embedding table, and it never decodes a
graph.** Not a threat, but it shows the "one token per graph" space is being actively claimed.

#### C3. Ammunition: two 2026 papers that say the incumbents don't actually work

- **"Revisiting Graph-Tokenizing LLMs" (arXiv 2605.03514):** "existing GTokenLLMs do not fully
  understand graph tokens"; they show "over-sensitivity or over-insensitivity to instruction changes"
  and "rely heavily on text for reasoning". Accuracy drops **3–50%** under semantics-preserving
  rephrasing; under content-level relabeling/reversing, most models fall **below 50% and some under
  10%**. Reference points on original instructions: LLaGA 87.45 (Cora), 92.13 (PubMed), 74.36 (Arxiv);
  GraphGPT 18.66 (Cora-70), 80.71 (PubMed), 62.50 (Arxiv).
- **"When Graph Tokens Sink" (Zhang, Zhou, Zheng, Fathony, Bruss, Agarwal — arXiv 2606.03712,
  2026-06-02):** graph tokens become "sink tokens" with massive activations at early positions, but
  there is a **"severe decoupling between activation-level saliency and graph-semantic utility"**.
  Pruning the top-2 sink tokens costs **0–1%** accuracy; removing *random non-sink* tokens sometimes
  hurts more. Sink tokens are predominantly `[PAD]`. Logit lens shows they decode to generic domain
  terms, not structure. Analysed: LLaGA, TEA-GLM, InstructGLM.

**Use these two in the introduction.** They are the empirical case that soft-prompt graph tokens are
not carrying structure, which is precisely the gap our attention-bias + adjacency-row design targets.

---

## Corrections to CLAUDE.md

### 1. "GNN soft prompts — a GNN compresses each node into one vector … Used by GraphToken, GraphGPT, LLaGA" (§1)
**Partially wrong, in a way that weakens our novelty claim if left uncorrected.** Confidence: **certain**.
- **GraphToken is not only GNNs.** Two of its seven encoders — **NodeSet** and **EdgeSet** — have
  **0 body parameters** and do no message passing (Table 3). NodeSet gets the best node-count score in
  the paper (0.996) and EdgeSet the best edge-count score (0.426). Our E6 ("DeepSets, one aggregation,
  not a GNN") is very close to NodeSet. **Do not claim non-message-passing encoders into a frozen LLM
  are unexplored.**
- **LLaGA has no GNN at all** in its Neighborhood Detail template: "this transformation is
  **parameter-free**" (p.2). Structure enters as a fixed-shape computation tree plus a precomputed
  Laplacian eigenvector positional embedding, and the *only* trained module is an MLP projector.
  Calling it a "GNN soft prompt" is inaccurate.
- **Only GraphGPT** among the three actually uses a pretrained message-passing graph transformer.

Suggested rewrite: *"**Compressed soft prompts** — a graph encoder (a GNN, a set encoder, or a
parameter-free structural template) compresses each node's neighbourhood into one or more vectors
prepended to the prompt. GraphToken, GraphGPT, LLaGA, GraphPrompter."*

### 2. "Verbalization … Used by Talk Like a Graph" — needs a caveat about what beating it means
**Not wrong, but misleading as a baseline.** Confidence: **certain**.
TLAG trains nothing; it is a prompting study on a frozen PaLM 62B over synthetic graphs with node counts
in the single to low double digits. It is not a competitive method and beating it is not evidence of
anything. **The useful fact is the failure mode, not the score:** best edge-existence accuracy 49.0%
against a 53.96% majority baseline, and ~0% on disconnected nodes. Cite those two, not "we beat TLAG".

### 3. "Every existing method converts the graph into something else before the LLM sees it" (§1)
**Overstated.** Confidence: **certain**.
GTLM, GaLA, UniGTE, GL-Fusion, HLM-G and Graph Language Models (Plenz & Frank, ACL 2024) all inject
graph structure into a pretrained LM's **attention computation** without converting it — that half of
our architecture has substantial prior art, including one paper (Plenz & Frank) that predates GTLM by
16 months and modifies T5's own relative-position bias table to index graph distance.
Our defensible novelty is narrower and should be stated as **two** claims:
  (a) **input**: the LLM's per-node token is a learned projection of the *raw adjacency row*, not text
      and not a message-passing summary;
  (b) **output**: a decoder head produces an **adjacency matrix** from LLM hidden states, rather than
      text or a label.
Claim (b) is the stronger one — I found no counterexample at all.

### 4. **UniGTE (arXiv 2510.16885, NeurIPS 2025) is missing from §9 and is the biggest threat**
Confidence: **certain** that it exists and does what I describe; the paper was fetched.
It is an encoder–decoder where a **LoRA-tuned Vicuna-7B encoder** carries an additive attention bias
`b(i,j)` built from a **shortest-path-distance lookup table** plus an edge MLP plus a masking bias, and
a **frozen Vicuna-7B decoder** both answers the task and **reconstructs the input graph** as a
regularizer, with **permutation invariance to node order** claimed. That is: SPD attention bias +
frozen LLM + reconstruction objective + permutation invariance — four of our selling points in one
NeurIPS 2025 paper. Differences we can defend on: their tokens are BERT-encoded **node attribute text**,
their reconstruction target is a **natural-language paraphrase** (not an adjacency matrix), and they
funnel everything through a **64-token bottleneck**. **Add it to §9 and to the related-work section.**

### 5. GraphToken's projection parameter count is self-contradictory in the source
Confidence: **certain** that the contradiction exists.
§4.2.3 says "we then project this into a prompt embedding with approximately **80,000 parameters** in
GraphToken", while Table 3 lists the Head at **1.1e7** for every GNN encoder (and 4.1e5 / 7.4e5 for
Node Set / Edge Set). These cannot both be right. **Do not cite a projection parameter count for
GraphToken without resolving this.** Cite the body counts (17,152 – 198,788), which are consistent.

### 6. GraphToken's soft-token count is not stated anywhere
Confidence: **certain**. The paper says only "a fixed number of token embeddings". CLAUDE.md's phrasing
"one vector [per node], prepended to the prompt" is a reasonable reading of the node-level readout but
is **not** something the paper states. Don't put a number in the writeup.

### 7. "Cora … 5,429 edges" (§6)
Confidence: **uncertain** — flagging, not asserting. 5,429 is the figure from the original Sen et al.
description. PyTorch Geometric's `Planetoid('Cora')` is commonly reported with `edge_index` of 10,556
entries, i.e. **5,278 undirected edges** after de-duplication and self-loop removal. Since `pos_weight =
num_negatives / num_positives` depends on this, **compute it from the loaded tensor at runtime rather
than hardcoding either number.** I did not run PyG in this session, so verify before relying on it.

### 8. arXiv IDs — all four check out
Confidence: **certain**. I fetched each abstract page myself:
- `2310.04560` → *Talk like a Graph* (Fatemi, Halcrow, Perozzi). ICLR 2024 confirmed via
  `proceedings.iclr.cc`. **Correct.**
- `2402.05862` → *Let Your Graph Do the Talking* / GraphToken (Perozzi, Fatemi, Zelle, Tsitsulin,
  Kazemi, Al-Rfou, Halcrow). **Correct.**
- `2605.10247` → *Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning*, Dario Vajda,
  submitted 2026-05-11. **Correct.**
- `2606.15633` → *Formalizing and Mitigating Structural Distortion in LLM Attention for Graph
  Reasoning* (Loveland, Trivedi, Weinstein, Huang, Koutra), KDD 2026, submitted 2026-06-14, revised
  2026-06-17. The method is named **GaLA (Graph-aligned Language Attention)** — CLAUDE.md calls the
  *paper* "GaLA", which is the method name, not the title. Minor; fix in the bibliography.

### 9. GTLM's headline claims all verified — no correction needed
Confidence: **certain**, from `arxiv.org/html/2605.10247`. SPD max distance 8; RRWP max steps 16;
Magnetic Laplacian q=0.25, dim 32; **173,056 bias parameters (4,096 SPD + 50,688 RRWP + 118,272
MagLap) ≈ 0.015% of a 1B base LLM**; LoRA rank 32–64, alpha = 2r; bias LR 5e-3 vs LoRA 3e-5;
RoPE position ids reset per node; non-causal prefix mask with causal target generation; intra-node bias
zeroed; FlashAttention incompatible; ~3× slower training. **Also confirmed: GTLM uses the base Llama
tokenizer and embedding table for node text, and has no link prediction and no graph output** — its
Future Work names autoregressive graph generation as open. CLAUDE.md §9 is accurate.
GTLM node classification: **Cora 90.04 acc / 92.14 F1**, PubMed 94.70 / 94.84, ogbn-arxiv 76.53 / 70.02,
Reddit 68.09 / 68.00. GraphQA: node count 100%, node degree 99.7%, edge existence 99.8%, cycle 96.7%,
connected nodes 90.7%, reachability 99.0%, shortest path 90.1%, edge count 56.9%, triangle count 30.9%.

### 10. "No verbalization … no tokenizer, no embedding table" — one existing partial violation
Confidence: **certain**. **InstructGLM** adds one new vocabulary token per node and **initialises those
token embeddings with the graph's raw node feature vectors (BoW / OGB)** (Fig. 2 caption, p.5). So
"non-text vectors have never been placed in an LLM's embedding space for graphs" would be false.
What remains true and defensible: *no one puts a projected **adjacency row** there, and no one bypasses
the tokenizer entirely.*

### 11. Token-blowup framing: "O(N²)" is right for dense verbalization, wrong for sparse
Confidence: **certain** (arithmetic, not a source claim). The task brief said "token blowup as
O(N²)/O(E)". For a *sparse* graph like Cora, the realistic verbalizations are edge-list = **O(E)** and
incident-list = **O(N + E)**, giving ~8–16× more tokens than our N-token scheme — not N². O(N²) only
applies if you verbalize the dense matrix, which nobody does. **Say "8–16× on Cora, and O(N²) in the
dense limit"** rather than claiming quadratic blowup on sparse benchmarks; a reviewer will catch it.

---

## Mechanism / math (exact formulas, tensor shapes)

### Verbalization cost model (derive this in the paper)
```
Let N = |V|, E = |edges|, avg degree d = 2E/N.
edge-list ("adjacency" encoder):  chars ~ c_pre + Σ_{i<N}(len(str(i))+2) + E*(2*len(str(N))+4)  = O(E log N)
incident encoder:                 chars ~ ...   + N*(c_node + d*(len(str(N))+2))                = O((N+E) log N)
dense matrix verbalization:                                                                     = O(N^2)
ours (adjacency-row tokens):      exactly N tokens, each of width N (or k after sketching)      = O(N) tokens
```
Measured on Cora (N=2708, E=5278): edge-list ≈ 79.5k chars, incident ≈ 155k chars, ours = 2708 tokens.

### GraphToken (arXiv 2402.05862)
```
Q = E(G) || T(T)                     # graph encoding concatenated before text prompt embedding
minimise L(A | Q)                    # LLM perplexity of the gold answer A
readout: graph-level  -> mean/sum pool over node reps
         node-level   -> one representation per node       (~ our token budget)
         edge-level   -> global rep, or concat of the two node reps
latent dim 128; Lion optimizer; lr alpha = 0.05
```

### LLaGA (arXiv 2402.08170)
```
(1) L = I - D^{-1/2} A_tree D^{-1/2} = U^T Lambda U    # tree shape fixed => computed ONCE for all graphs
(2) h_{v_i} = 0 || U_i             if v_i = [pad]
            = phi(x_{v_i}) || U_i  otherwise           # phi = SBERT / RoBERTa / SimTeG
(3) h_v^i = (1/|N_v^1|) * sum_{v' in N_v^1} h_{v'}^{i-1},   h_x^0 = phi(x_v)   # Hop-Field, parameter-free
(4) e_i = f_theta(h_i)                                 # f_theta = simple MLP; ONLY trained module
(5) max_theta p(X_answer | X_graph, X_question, X_system)
ND token count = 1 + n1 + n1*n2 = 1 + 10 + 100 = 111 per centre node
HO token count = 4
```

### GraphGPT (arXiv 2310.13023)
```
(2) H^(l) = sigma( A_tilde H^(l-1) W ),  A_tilde = A + I
(3) H = f_G(X), T = f_T(C), H_hat = norm(H), T_hat = norm(T)
    G1 = (H_hat T_hat^T)exp(tau), G2 = (H_hat T_hat'^T)exp(tau), G3 = (T_hat T_hat'^T)exp(tau)
    T_hat'_i = (1/|N_i|) sum_{j in N_i} T_hat_j
(4) L = sum_{i=1..3} 0.5 * lambda_i [ CE(G_i, y) + CE(G_i^T, y) ],  y = (0,1,...,n-1)^T
(5) p(X_O | X_G, X_T) = prod_{i=1..L} p_theta(x_i | X_G, X_{T,<i}, X_{O,<i})
graph token sequence: {<graph_begin>, <graph_token>_1, ..., <graph_token>_n, <graph_end>}   # n = |V_subgraph|
X_G = f_P(H_hat),  f_P = single nn.Linear
```

### UniGTE (arXiv 2510.16885) — closest attention formulation to ours
```
A_hat_ij = [ (x_i W_Q) R(i-j) (x_j W_K)^T ] / sqrt(d_k)  +  b(i,j)
b(i,j) = 1_{i,j <= n} ( b^PE_ij + b^Edge_ij ) + b^M_ij
   b^PE_ij   = e( dist_G(i,j) )                    # learned shortest-path lookup   <- our hop bias
   b^Edge_ij = MLP-aggregate(edge descriptions on the shortest path i->j)
   b^M_ij    = masking bias: graph tokens bidirectional; text never attends to graph
m = 64 learnable alignment tokens; encoder = Vicuna-7B + LoRA; decoder = frozen Vicuna-7B
```

### Graph Language Models (Plenz & Frank, ACL 2024)
```
Attention(Q,K,V) = softmax( Q K^T / sqrt(d) + B_P + M ) V
  M   : -inf outside the allowed graph neighbourhood (lGLM), 0 everywhere (gGLM)
  B_P : T5's BUCKETED relative-position bias table, re-indexed by GRAPH distance, not sequence distance
        cross-triplet / T2G / G2T bias parameters initialised from the +inf bucket
```
*Note for us:* re-using the frozen LLM's **existing** relative-position bias table, indexed by hop
distance, is a cheaper alternative to GTLM's fresh SPD table and it is already published. If we use a
fresh table we should say why (their table is bucketed and tuned for sequence distance).

### GAE / VGAE (Kipf & Welling 2016) — the decoder we are importing
```
(1) q(Z|X,A) = prod_i N(z_i | mu_i, diag(sigma_i^2)),  mu = GCN_mu(X,A),  log sigma = GCN_sigma(X,A)
    GCN(X,A) = A_tilde ReLU(A_tilde X W_0) W_1,   A_tilde = D^{-1/2} A D^{-1/2}
(2) p(A|Z) = prod_i prod_j p(A_ij | z_i, z_j),   p(A_ij = 1 | z_i, z_j) = sigmoid(z_i^T z_j)
(3) L = E_{q(Z|X,A)}[ log p(A|Z) ] - KL[ q(Z|X,A) || p(Z) ]
(4) GAE (non-probabilistic):  A_hat = sigmoid(Z Z^T),  Z = GCN(X,A)
```
Our D1 is (4) with a learned `W` inserted: `A_hat = sigmoid(H W H^T)`.
Kipf's own note: "For very sparse A, it can be beneficial to re-weight terms with A_ij = 1 in L or
alternatively sub-sample terms with A_ij = 0. We choose the former" — i.e. **`pos_weight` is what the
original GAE paper does too.** Cite it for CLAUDE.md §11.

---

## Code we can reuse (real snippets, real signatures)

### 1. `MaskGAE` — masking, decoders, degree loss, and the exact metric protocol
Repo `github.com/EdisonLeeeee/MaskGAE` (branch **`master`**). **This is the closest existing codebase
to our §5.1 primary task and it already implements our §5 auxiliary degree loss.**

`maskgae/mask.py` — edge masking (their `p=0.7`; our spec says 15%):
```python
def mask_edge(edge_index: Tensor, p: float = 0.7):
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 (got {p}')
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]   # (remaining, masked)

class MaskEdge(nn.Module):
    def __init__(self, p: float = 0.7, undirected: bool = True): ...
    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges
```
There is also `MaskPath` (random-walk masking via `torch_cluster.random_walk`), which gives their best
number. Worth an ablation slot: **random-walk masking beat i.i.d. edge masking** (Cora 96.45 vs 96.42
AUC — marginal there; PubMed 98.84 vs 98.75).

`maskgae/model.py` — our D1 and D3 already written:
```python
class DotEdgeDecoder(nn.Module):          # our D1 without the learned W
    def forward(self, z, edge, sigmoid=True):
        x = (z[edge[0]] * z[edge[1]]).sum(-1)
        return x.sigmoid() if sigmoid else x

class EdgeDecoder(nn.Module):             # our D3, but on the Hadamard product only
    def __init__(self, in_channels, hidden_channels, out_channels=1,
                 num_layers=2, dropout=0.5, activation='relu'): ...
    def forward(self, z, edge, sigmoid=True, reduction=False):
        x = z[edge[0]] * z[edge[1]]
        for mlp in self.mlps[:-1]:
            x = self.activation(mlp(self.dropout(x)))
        x = self.mlps[-1](x)
        return x.sigmoid() if sigmoid else x

class DegreeDecoder(nn.Module):           # our L_degree auxiliary head, already written
    def __init__(self, in_channels, hidden_channels, out_channels=1,
                 num_layers=2, dropout=0.5, activation='relu'): ...
```
Degree loss weighting, from `MaskGAE.train_step` — **note alpha is tiny**:
```python
if self.degree_decoder is not None and alpha:
    deg = degree(masked_edges[1].flatten(), data.num_nodes).float()
    loss += alpha * F.mse_loss(self.degree_decoder(z).squeeze(), deg)
# default alpha = 0.003  (CLI: --alpha, "loss weight for degree prediction. (default: 2e-3)")
```
**Copy that scale.** CLAUDE.md's `L_total = w*BCE + lambda1*L_degree + lambda2*L_spectral` gives no
lambda values; MaskGAE's tuned answer for lambda1 on Cora link prediction is **3e-3**, not 0.1 or 1.0.

Their evaluation protocol — this is exactly CLAUDE.md §11 "score only what was masked":
```python
@torch.no_grad()
def test_step(self, data, pos_edge_index, neg_edge_index, batch_size=2**16):
    self.eval()
    z = self(data.x, data.edge_index)
    pos_pred = self.batch_predict(z, pos_edge_index)
    neg_pred = self.batch_predict(z, neg_edge_index)
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    y = torch.cat([pos_pred.new_ones(pos_pred.size(0)),
                   neg_pred.new_zeros(neg_pred.size(0))], dim=0)
    y, pred = y.cpu().numpy(), pred.cpu().numpy()
    return roc_auc_score(y, pred), average_precision_score(y, pred)
```
**AP = `average_precision_score` = AUPRC.** So MaskGAE's "AP" column *is* the metric CLAUDE.md §11
demands, computed on a balanced pos/neg set. Their Cora AP of **95.95** is our real target.
(Careful: on a *balanced* eval set AUPRC and AUROC are close; on the true 0.1%-positive matrix AUPRC
collapses. Report both protocols and say which is which.)

`maskgae/loss.py` — four ranking losses, all one-liners; `ce_loss` is the pos/neg BCE we need:
```python
def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def auc_loss(pos_out, neg_out):        # squared hinge on the margin
    return torch.square(1 - (pos_out - neg_out)).sum()

def hinge_auc_loss(pos_out, neg_out):
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()

def log_rank_loss(pos_out, neg_out, num_neg=1):
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()
```
Worth trying `auc_loss` / `hinge_auc_loss` as an alternative to `pos_weight`-ed BCE — they sidestep the
class-imbalance calibration problem entirely by only ever comparing a positive to a negative.

Negative sampler they use (deliberately crude — uniform random pairs, not `negative_sampling`):
```python
def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges
```

### 2. `LLaGA` — the canonical "project graph vectors into LLM token space" pattern
Repo `github.com/VITA-Group/LLaGA` (branch **`master`**, not `main`), `model/llaga_arch.py`:
```python
def build_graph_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    hidden_dim = getattr(config, 'word_embed_proj_dim',
                         getattr(config, 'hidden_size', 'linear'))
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, hidden_dim)      # <- our E1, verbatim
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)                            # <- our E2, verbatim
    raise ValueError(f'Unknown projector type: {projector_type}')

class LlagaMetaForCausalLM(ABC):
    def encode_graphs(self, graph, graph_emb):
        graph_features = self.get_model().mm_projector(graph_emb)
        graph_features[graph == DEFAULT_GRAPH_PAD_ID] = 0.        # pad handling -- copy this
        return graph_features
```
Two things to lift: (a) `mm_hidden_size` → `hidden_size` naming and the `mm_projector_type` config
string (`"linear"` / `"2-layer-mlp"`) is a clean interface for our E1/E2 swap; (b) **zeroing padded
node slots after projection** — we will need the same for variable-N batches.

Also `build_special_tokens`, which initialises new special-token embeddings from the **mean of the
existing embedding table** rather than randomly — cheap trick if we ever add control tokens
(GTLM's Future Work names `<ADD_EDGE>`):
```python
input_embeddings = self.get_input_embeddings().weight.data
input_embeddings_avg = input_embeddings.mean(dim=0, keepdim=True).unsqueeze(1).detach()
special_token_emb = torch.nn.parameter.Parameter(
    data=input_embeddings_avg.repeat(num_token, 1, 1), requires_grad=True)
```

### 3. `talk-like-a-graph` — for the verbalization ABLATION only
Repo `github.com/google-research/talk-like-a-graph` (branch `main`),
`talk_like_a_graph/graph_text_encoders.py`. If a reviewer asks "did you actually try verbalizing Cora
into your frozen LLM?", this is the code that answers it in an afternoon:
```python
def adjacency_encoder(graph: nx.Graph, name_dict: dict[int, str]) -> str:
  output = ("In an undirected graph, (i,j) means that node i and node j are "
            "connected with an undirected edge. ")
  output += "G describes a graph among nodes %s.\n" % create_node_string(name_dict, len(graph.nodes()))
  if graph.edges():
    output += "The edges in G are: "
  for i, j in graph.edges():
    output += "(%s, %s) " % (name_dict[i], name_dict[j])
  return output.strip() + ".\n"

def incident_encoder(graph: nx.Graph, name_dict: dict[int, str]) -> str:
  ...
  for source_node in graph.nodes():
    target_nodes = graph.neighbors(source_node)
    ...
    output += "Node %s is connected to nodes %s.\n" % (source_node, target_nodes_str[:-2])
  return output

EDGE_ENCODER_FN = {"adjacency": adjacency_encoder, "incident": incident_encoder,
                   "friendship": friendship_encoder, "south_park": friendship_encoder,
                   "got": friendship_encoder, "politician": social_network_encoder,
                   "social_network": social_network_encoder, "expert": expert_encoder,
                   "coauthorship": coauthorship_encoder, "random": adjacency_encoder,
                   "nx_edge_encoder": nx_encoder}

# public entry point:
encode_graph(graph, graph_encoder=None, node_encoder=None, edge_encoder=None) -> str
```
`incident` is the strongest encoder under CoT in their Table 1 (node degree 75.0, node count 57.6,
connected nodes 30.2) and is the one to use if we run the verbalization ablation.
Node-name dictionaries live in `talk_like_a_graph/name_dictionaries.py` with keys
`integer | popular | alphabet | got | south_park | politician | random_integer | nx_node_name`.

### 4. PyTorch Geometric (already in CLAUDE.md §9)
`torch_geometric.nn.GAE`, `VGAE`, `InnerProductDecoder`, `GCNConv`, `GATConv`,
`torch_geometric.datasets.Planetoid`, `torch_geometric.transforms.RandomLinkSplit`.
To reproduce the Kipf 5%/10% protocol exactly:
```python
RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True,
                split_labels=True, add_negative_train_samples=False)
```

---

## Numbers to beat / hyperparameters to copy

### D. Cora — LINK PREDICTION

**Protocol matters more than the number.** There are four incompatible protocols in this table.
Say which one we use, and report under it.

| Method | Cora AUC | Cora AP (=AUPRC) | Protocol | Source (fetched) |
|---|---|---|---|---|
| SC (spectral clustering) | 84.6 ± 0.01 | 88.5 ± 0.00 | P1 | VGAE PDF Table 1 |
| DeepWalk | 83.1 ± 0.01 | 85.0 ± 0.00 | P1 | VGAE PDF Table 1 |
| GAE* (no features) | 84.3 ± 0.02 | 88.1 ± 0.01 | P1 | VGAE PDF Table 1 |
| VGAE* (no features) | 84.0 ± 0.02 | 87.7 ± 0.01 | P1 | VGAE PDF Table 1 |
| **GAE** | **91.0 ± 0.02** | **92.0 ± 0.03** | P1 | VGAE PDF Table 1 |
| **VGAE** | **91.4 ± 0.01** | **92.6 ± 0.01** | P1 | VGAE PDF Table 1 |
| GAE (re-run by MaskGAE) | 91.09 | 92.83 | P2 | MaskGAE ar5iv Table 3 |
| VGAE (re-run by MaskGAE) | 91.40 | 92.60 | P2 | MaskGAE ar5iv Table 3 |
| ARGA | 92.40 | 93.23 | P2 | MaskGAE ar5iv Table 3 |
| SEAL | 92.22 | 93.12 | P2 | MaskGAE ar5iv Table 3 |
| **MaskGAE (edge mask)** | **96.42** | **95.91** | P2 | MaskGAE ar5iv Table 3 |
| **MaskGAE (path mask)** | **96.45** | **95.95** | P2 | MaskGAE ar5iv Table 3 |
| GAT | 90.71 (**accuracy**) | — | P3 | GraphPrompter arXiv HTML |
| GraphPrompter + LoRA (LLaMA-2-7B frozen) | 90.10 (**accuracy**) | — | P3 | GraphPrompter arXiv HTML |
| Subgraph Prompt Tuning | 89.15 (**accuracy**) | — | P3 | GraphPrompter arXiv HTML |
| GCN | 81.59 (**accuracy**) | — | P4 | LLaGA PDF Table 1 |
| GAT | 80.06 (**accuracy**) | — | P4 | LLaGA PDF Table 1 |
| GraphSage | 79.15 (**accuracy**) | — | P4 | LLaGA PDF Table 1 |
| **LLaGA-HO-7B (Vicuna-7B frozen)** | **86.82** (**accuracy**) | — | P4 | LLaGA PDF Table 1 |
| LLaGA-ND-7B, Task Expert | 89.41 (**accuracy**) | — | P4 | LLaGA PDF Table 1 |
| GPT-3.5-turbo | 65.74 (**accuracy**) | — | P4 | LLaGA PDF Table 1 |

```
P1  Kipf & Welling: 5% of citation links for val, 10% for test, plus an EQUAL number of randomly
    sampled non-edges. Metrics: roc_auc_score and average_precision_score. 10 runs, fixed splits.
P2  MaskGAE: 85% / 5% / 10% edge split, same balanced-negatives AUC/AP metric.
P3  GraphPrompter: binary yes/no question to the LLM, ACCURACY on a balanced set. NOT AUC.
P4  LLaGA: binary yes/no question to the LLM, ACCURACY. NOT AUC. Different Cora preprocessing again.
```

**Bottom line for our claim:** `MaskGAE path-mask, Cora AUC 96.45 / AP 95.95` under P2 is the number a
reviewer will point at. `GAE 91.0 / 92.0` under P1 is what CLAUDE.md §7 calls "the number to beat" —
it is the *floor*, not the ceiling. **Reaching 92 is not a result. Reaching 96+ is.**

**On PubMed** the equivalent contrast is brutal and worth putting in the abstract:
`GAE 96.4 AUC / 96.5 AP` (2016 GCN autoencoder) vs `GraphGPT best 0.8246 AUC / 0.8026 AP`
(2024 LLM soft prompts, the only genuine AUC from an LLM method). **~14 AUC points.**

Full VGAE Table 1 for the other datasets (P1), since we will need Citeseer/PubMed too:
```
            Cora AUC / AP    Citeseer AUC / AP   Pubmed AUC / AP
SC          84.6 / 88.5      80.5 / 85.0         84.2 / 87.8
DW          83.1 / 85.0      80.5 / 83.6         84.4 / 84.1
GAE*        84.3 / 88.1      78.7 / 84.1         82.2 / 87.4
VGAE*       84.0 / 87.7      78.9 / 84.1         82.7 / 87.5
GAE         91.0 / 92.0      89.5 / 89.9         96.4 / 96.5
VGAE        91.4 / 92.6      90.8 / 92.0         94.4 / 94.7
```
MaskGAE (P2): Citeseer 98.02 / 98.18 (edge), 97.87 / 98.09 (path); PubMed 98.75 / 98.66 (edge),
98.84 / 98.78 (path).

### D. Cora — NODE CLASSIFICATION

| Method | Cora acc (%) | Split | Source (fetched) |
|---|---|---|---|
| GCN | 81.5 | Planetoid (20/class, 500 val, 1000 test) | GAT ar5iv Table 2 |
| GCN-64 | 81.4 ± 0.5 | Planetoid | GAT ar5iv Table 2 |
| **GAT** | **83.0 ± 0.7** | Planetoid | GAT ar5iv Table 2 |
| GAT | 76.70 ± 0.42 | 60/20/20 random | InstructGLM PDF Table 2 |
| GCN | 87.78 ± 0.96 | 60/20/20 random | InstructGLM PDF Table 2 |
| GCNII | 88.93 ± 1.37 | 60/20/20 random | InstructGLM PDF Table 2 |
| RevGAT | 89.11 ± 0.00 | 60/20/20 random | InstructGLM PDF Table 2 |
| ACM-GCN+ | 89.75 ± 1.16 | 60/20/20 random | InstructGLM PDF Table 2 |
| Graphormer | 80.41 ± 0.30 | 60/20/20 random | InstructGLM PDF Table 2 |
| **InstructGLM Flan-T5-base (tuned LLM)** | **90.77 ± 0.52** | 60/20/20 random | InstructGLM PDF Table 2 |
| InstructGLM Llama-7b (LoRA) | 87.08 ± 0.32 | 60/20/20 random | InstructGLM PDF Table 2 |
| GAT | 84.69 | GraphPrompter's own | GraphPrompter arXiv HTML |
| GraphPrompter + LoRA | 80.26 | GraphPrompter's own | GraphPrompter arXiv HTML |
| GCN | 88.93 | LLaGA's own | LLaGA PDF Table 1 |
| LLaGA-HO-7B | 89.22 | LLaGA's own | LLaGA PDF Table 1 |
| **GTLM (Llama-3.2-1B, frozen + bias + LoRA)** | **90.04** (F1 92.14) | GTLM's own | `arxiv.org/html/2605.10247` |
| GraphGPT-7B-v1.5-cot (zero-shot, **70-class** Cora) | 18.13 | Arxiv→Cora zero-shot | GraphGPT PDF Table 1 |
| LLaGA (as measured by 2605.03514) | 87.45 | that paper's harness | arXiv 2605.03514 |
| GL-Fusion (Llama-3-8B, ~10% params trained) | 84.3 | its own | ar5iv 2412.06849 |

**Rule for our tables: never compare across split protocols without saying so.** GAT is 76.70, 83.0
and 84.69 in three different papers on the same dataset.

### Hyperparameters worth copying verbatim

| Source | Setting | Value |
|---|---|---|
| **GTLM** | bias LR vs LoRA LR (GraphQA) | **5e-3 vs 3e-5** |
| GTLM | bias LR vs LoRA LR (node classification) | 0.01–0.04 vs 6e-5–3e-4 |
| GTLM | LoRA rank / alpha | r = 32 or 64, alpha = 2r |
| GTLM | SPD max distance / RRWP steps / MagLap q, dim | 8 / 16 / 0.25, 32 |
| GTLM | total bias params | 173,056 = 4,096 (SPD) + 50,688 (RRWP) + 118,272 (MagLap) ≈ 0.015% of 1B |
| **MaskGAE** | degree-loss weight alpha | **0.003** (CLI default; docstring says 2e-3) |
| MaskGAE | mask ratio p | 0.7 (much higher than our 15% — worth an ablation) |
| MaskGAE | encoder / hidden / decoder channels | 128 / 128 / 64 |
| MaskGAE | encoder layers / decoder layers | 1 / 2 |
| MaskGAE | encoder dropout / decoder dropout | 0.8 / 0.2 |
| MaskGAE | lr / weight decay / grad_norm | 1e-2 / 5e-5 / 1.0 |
| MaskGAE | epochs / runs / batch | 500 / 10 / 2**16 |
| **VGAE** | hidden / latent, epochs, optimizer | 32 / 16, 200 iters, Adam lr 0.01, Glorot init |
| **GraphToken** | optimizer / lr / latent | Lion / 0.05 / 128 |
| **LLaGA** | lr / batch / epochs | 2e-5 / 16 / 1 (Cora replicated 3×) |
| **GraphGPT** | lr / batch / max len / epochs | 2e-3 / 2 per GPU / 2048 / 3 then 2 |
| **UniGTE** | alignment tokens | 64 |
| **GraphPrompter** | subgraph hops | 3 |

---

## Open questions

1. **Does UniGTE's `b^PE` beat GTLM's SPD table?** Both are learned shortest-path lookups but UniGTE
   builds on top of RoPE's `R(i−j)` term while GTLM *resets* RoPE per node. These are contradictory
   design choices and neither paper ablates against the other. Worth a direct experiment — cheap, and
   a publishable ablation on its own.
2. **How many soft tokens does GraphToken actually emit?** Not stated. If it is one per node for
   node-level tasks, then our token budget is identical to a Feb-2024 paper and we should stop
   mentioning token efficiency as a contribution *against GraphToken* (it still holds against
   verbalization and against LLaGA-ND's 111 tokens/node).
3. **Which projection parameter count is right for GraphToken, 80k or 1.1e7?** The paper contradicts
   itself. Resolve before citing.
4. **Is MaskGAE's 96.45 reproducible under our masking rate?** They mask 70% of edges; CLAUDE.md
   specifies 15%. Their number may not transfer. Re-run MaskGAE at p=0.15 to get an apples-to-apples
   floor before claiming anything.
5. **Cora edge count: 5,429 or 5,278?** Affects `pos_weight`. Compute at runtime.
6. **Nobody reports AUPRC on the *true* sparse matrix.** Every AP number in the literature is on a
   balanced pos/neg set. If we report AUPRC on the real 0.1%-positive matrix (as CLAUDE.md §11
   implies), our numbers will look far worse than everyone else's and it will not be a fair
   comparison. **Decide the protocol now** and report both: (a) balanced, comparable to MaskGAE/GAE,
   (b) true-sparsity, honest but incomparable.
7. **Has anyone run "verbalize Cora into a frozen LLM"?** TLAG only used tiny synthetic graphs. Nobody
   I found verbalized a 2,708-node graph, presumably because it does not fit. That absence is itself a
   citable point but I could not find a paper stating it — so state it as our own measurement, not as
   a literature claim.
8. **Does the "graph tokens sink" finding (2606.03712) apply to our tokens too?** They found soft
   graph tokens become high-activation, low-utility sinks. If our adjacency-row tokens do the same,
   the whole architecture is in trouble. **Add a diagnostic:** measure activation norms and attention
   mass on our graph tokens early, and run their "prune the top-2 sink tokens" test. If pruning our
   tokens costs 0–1% like theirs, we have the same disease.
9. **Is GaLA's per-head `λ_h` calibration compatible with a *learned* bias table?** GaLA is
   inference-time with a fixed `1/SPD` bias. We would be learning the bias, so the entropy heuristic
   may not apply. Unresolved.
10. **Should we reuse the base LLM's existing relative-position bias table (Plenz & Frank style)
    instead of a fresh SPD table (GTLM style)?** For a RoPE model like Llama there is no additive
    table to reuse, so probably not — but this needs a sentence in the paper, because a T5-family
    reviewer will ask.

---

## Sources fetched

Primary papers (abstract and/or full text read by me in this session):
- arXiv **2310.04560** — *Talk like a Graph* (abs page + full PDF pp. 3–8 incl. Tables 1–4). ICLR 2024
  confirmed via `proceedings.iclr.cc/paper_files/paper/2024/file/bf72f65f30eedf5d48da6980ee02b589-Paper-Conference.pdf`
- arXiv **2402.05862** — *Let Your Graph Do the Talking* / GraphToken (abs page + `arxiv.org/html/2402.05862v1`, Tables 1–3)
- arXiv **2402.08170** — *LLaGA* (full PDF pp. 1–8 incl. Tables 1–6)
- arXiv **2310.13023** — *GraphGPT* (full PDF pp. 3–7 incl. Tables 1–4)
- arXiv **2402.10359** — *Can we Soft Prompt LLMs for Graph Learning Tasks?* / GraphPrompter (`arxiv.org/html/2402.10359v2`)
- arXiv **2308.07134** — *Language is All a Graph Needs* / InstructGLM (full PDF pp. 5–8 incl. Tables 1–2)
- arXiv **2401.07105** — *Graph Language Models* (Plenz & Frank, ACL 2024) (`arxiv.org/html/2401.07105`)
- arXiv **2510.16885** — *UniGTE* (abs page + `arxiv.org/html/2510.16885v1`)
- arXiv **2412.06849** — *GL-Fusion* (`ar5iv.labs.arxiv.org/html/2412.06849`)
- arXiv **2605.03514** — *Revisiting Graph-Tokenizing LLMs* (`arxiv.org/html/2605.03514v1`)
- arXiv **2606.03712** — *When Graph Tokens Sink* (`arxiv.org/html/2606.03712`)
- arXiv **2606.06865** — *Are LLMs Suitable for Graph Computation?* survey (`arxiv.org/html/2606.06865`)
- arXiv **2602.01771** — *`<SOG_k>`: One LLM Token for Explicit Graph Structural Understanding* (`arxiv.org/html/2602.01771v1`)
- arXiv **2605.10247** — *GTLM / Teaching LLMs to See Graphs* (abs page + `arxiv.org/html/2605.10247`)
- arXiv **2606.15633** — *Formalizing and Mitigating Structural Distortion…* / GaLA (abs page)
- arXiv **1611.07308** — *Variational Graph Auto-Encoders* (full PDF pp. 1–3 incl. Table 1)
- arXiv **1710.10903** — *Graph Attention Networks* (`ar5iv.labs.arxiv.org/html/1710.10903`, Table 2)
- arXiv **2205.10053** — *MaskGAE / What's Behind the Mask* (`ar5iv.labs.arxiv.org/html/2205.10053`, Table 3)
- arXiv **2605.06239** — *When Graph Language Models Go Beyond Memorization* (PDF; partial extraction only)

Code read directly (raw.githubusercontent.com):
- `google-research/talk-like-a-graph@main : talk_like_a_graph/graph_text_encoders.py` (all 299 lines)
- `EdisonLeeeee/MaskGAE@master : maskgae/mask.py` (all 110 lines), `maskgae/model.py` (decoders,
  `train_step`, `test_step`, `random_negative_sampler`), `maskgae/loss.py` (all), `train_linkpred.py`
  (argparse defaults)
- `VITA-Group/LLaGA@master : model/llaga_arch.py` (`build_graph_projector`, `LlagaMetaModel`,
  `encode_graphs`, `build_special_tokens`)
- Repo file listings via GitHub API for `VITA-Group/LLaGA`, `HKUDS/GraphGPT`,
  `EdisonLeeeee/MaskGAE`, `google-research/talk-like-a-graph`

Negative-result queries (arXiv full-text API, all returning zero hits) — listed in §C1.

Not verifiable in this session:
- **GraphToken has no public code repository that I could find.**
- `openreview.net` was behind a browser-verification wall; ICLR 2024 status for TLAG was confirmed via
  the ICLR proceedings PDF URL instead.
- GL-Fusion and *Beyond Memorization* were read via secondary renderings (ar5iv / partial PDF text);
  their numbers are lower-confidence than the ones I read in a rendered PDF or the paper's own HTML.
