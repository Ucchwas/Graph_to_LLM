# PyG Baselines, Data Protocol and Metrics

Domain reference for: Planetoid/Cora loading, the canonical link-prediction split,
GAE/VGAE/GCNConv/GATConv APIs, AUPRC done correctly on an extremely imbalanced
matrix, `pos_weight` vs focal loss, baseline recipes, and the N=2708 memory reality
check.

Everything below was fetched from primary sources on **2026-08-25**. Where I could
not verify something with my own fetch, it is marked **UNVERIFIED** and says so.

**Versions in force at time of writing**

- PyTorch Geometric **2.8.0**, released **2026-06-05** (previous: 2.7.0, 2025-10-14;
  2.6.0, 2024-09-13). Source: `CHANGELOG.md` on `master`.
- All PyG source quoted below is from the `master` branch as of 2026-08-25.
- **No PyTorch and no PyG is installed in this working directory** — `import torch`
  raises `ModuleNotFoundError`. Nothing here was executed against a live PyG; every
  API detail is quoted from the current source on GitHub. The arithmetic (density,
  `pos_weight`, split sizes, memory, the AP-vs-balance simulation) *was* executed,
  in pure Python.

---

## Verified facts (with source next to each)

### Cora dataset numbers

| Quantity | Value | Source |
|---|---|---|
| `data.num_nodes` | **2708** | PyG `Planetoid` docstring STATS table |
| `data.edge_index.shape` | **[2, 10556]** | `Planetoid` docstring; "Introduction by Example" prints `Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])` |
| unique undirected edges | **5278** (= 10556 / 2) | arithmetic on the above |
| `dataset.num_node_features` | **1433** | Introduction by Example |
| `dataset.num_classes` | **7** | Introduction by Example |
| `data.is_undirected()` | **True** | Introduction by Example |
| `data.train_mask.sum()` | **140** | Introduction by Example (node-classification public split) |
| `data.val_mask.sum()` | **500** | Introduction by Example |
| `data.test_mask.sum()` | **1000** | Introduction by Example |

Derived, computed here in pure Python:

```
N               = 2708
N^2             = 7,333,264
N^2 - N         = 7,330,556        (off-diagonal cells)
upper triangle  = 3,665,278        (= N(N-1)/2)

directed edge entries (PyG)   = 10,556
unique undirected edges       =  5,278

density including diagonal    = 10556 / 7,333,264 = 0.0014395  = 0.14395 %
density excluding diagonal    = 10556 / 7,330,556 = 0.0014400  = 0.14400 %
upper-tri positive rate       =  5278 / 3,665,278 = 0.0014400  = 0.14400 %

negative : positive ratio (full matrix)      = 693.70 : 1
negative : positive ratio (off-diagonal)     = 693.44 : 1
```

**The positive rate is 0.144 %, not 0.1 %.** Close enough that the spec's conclusion
holds, but use 0.144 % and `pos_weight ≈ 693` in code and in the writeup.

### The 5,429 vs 10,556 discrepancy

The original Cora source (Yang, Cohen & Salakhutdinov, *Revisiting Semi-Supervised
Learning with Graph Embeddings*, arXiv 1603.08861) reports 5,429 citation links. PyG
reports 10,556 directed entries, i.e. 5,278 undirected edges — **not** 5,429.

PyG maintainer `EdisonLeeeee` in pyg-team/pytorch_geometric discussion #6203:

> there are edges that are inherently undirected, e.g., (1,2) and (2, 1) both exist
> in the dataset. PyG will remove duplicate edges from a graph when converting it to
> an undirected one, so the number of edges will be less than 5429 * 2.

`5429 × 2 = 10,858`; after `coalesce` this becomes 10,556, so **302 directed entries
(= 151 undirected links) of the raw 5,429 are duplicates or self-citations**, leaving
5,278 distinct undirected edges.

### `Planetoid` — exact current signature

```python
# torch_geometric/datasets/planetoid.py  (master, 2026-08-25)
def __init__(
    self,
    root: str,
    name: str,
    split: str = "public",
    num_train_per_class: int = 20,
    num_val: int = 500,
    num_test: int = 1000,
    transform: Optional[Callable] = None,
    pre_transform: Optional[Callable] = None,
    force_reload: bool = False,
) -> None:
```

`split` ∈ `{"public", "full", "geom-gcn", "random"}`; `"public"` is the fixed split
from arXiv 1603.08861 and is the default. `num_train_per_class` / `num_val` /
`num_test` **only apply when `split="random"`**.

The `Data` object Cora yields has exactly six attributes:
`x [2708, 1433] float`, `edge_index [2, 10556] int64`, `y [2708] int64`,
`train_mask [2708] bool`, `val_mask [2708] bool`, `test_mask [2708] bool`.
There is no `edge_attr` and no `edge_weight`.

**The public split is a node-classification split (140/500/1000 nodes) and is
irrelevant to link prediction.** For link prediction you split *edges*, with
`RandomLinkSplit`.

### `RandomLinkSplit` — exact current signature

```python
# torch_geometric/transforms/random_link_split.py  (master, 2026-08-25)
def __init__(
    self,
    num_val: Union[int, float] = 0.1,
    num_test: Union[int, float] = 0.2,
    is_undirected: bool = False,
    key: str = 'edge_label',
    split_labels: bool = False,
    add_negative_train_samples: bool = True,
    neg_sampling_ratio: float = 1.0,
    disjoint_train_ratio: Union[int, float] = 0.0,
    edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
    rev_edge_types: Optional[Union[
        EdgeType,
        List[Optional[EdgeType]],
    ]] = None,
) -> None:
```

Docstring verbatim, on the parameters that matter:

> **num_val** (int or float, optional): The number of validation edges. If set to a
> floating-point value in [0, 1], it represents the ratio of edges to include in the
> validation set. (default: `0.1`)
>
> **is_undirected** (bool): If set to `True`, the graph is assumed to be undirected,
> and positive and negative samples will not leak (reverse) edge connectivity across
> different splits. **This only affects the graph split, label data will not be
> returned undirected.**
>
> **split_labels** (bool, optional): If set to `True`, will split positive and
> negative labels and save them in distinct attributes `"pos_edge_label"` and
> `"neg_edge_label"`, respectively. (default: `False`)
>
> **add_negative_train_samples** (bool, optional): Whether to add negative training
> samples for link prediction. If the model already performs negative sampling, then
> the option should be set to `False`. Otherwise, the added negative samples will be
> the same across training iterations unless negative sampling is performed again.
> (default: `True`)
>
> **neg_sampling_ratio** (float, optional): The ratio of sampled negative edges to
> the number of positive edges. (default: `1.0`)
>
> **disjoint_train_ratio** (int or float, optional): If set to a value greater than
> `0.0`, training edges will not be shared for message passing and supervision.

### `RandomLinkSplit` internals — what actually happens

Verbatim from `forward()`:

```python
if is_undirected:
    mask = edge_index[0] <= edge_index[1]
    perm = mask.nonzero(as_tuple=False).view(-1)
    perm = perm[torch.randperm(perm.size(0), device=perm.device)]
else:
    device = edge_index.device
    perm = torch.randperm(edge_index.size(1), device=device)

num_val = self.num_val
if isinstance(num_val, float):
    num_val = int(num_val * perm.numel())
num_test = self.num_test
if isinstance(num_test, float):
    num_test = int(num_test * perm.numel())
```

**Critical:** with `is_undirected=True`, `perm.numel()` is the count of edges with
`row <= col` — **5,278 for Cora, not 10,556**. The ratios are ratios of the *unique
undirected* edge count.

Negative sample counts, verbatim:

```python
num_neg_train = 0
if self.add_negative_train_samples:
    if num_disjoint > 0:
        num_neg_train = int(num_disjoint * self.neg_sampling_ratio)
    else:
        num_neg_train = int(num_train * self.neg_sampling_ratio)
num_neg_val = int(num_val * self.neg_sampling_ratio)
num_neg_test = int(num_test * self.neg_sampling_ratio)

neg_edge_index = negative_sampling(edge_index, size,
                                   num_neg_samples=num_neg,
                                   method='sparse')
```

The **full original `edge_index`** is the forbidden set, so a sampled "negative" is
guaranteed not to be a real edge anywhere in the graph, in any split.
`num_neg = num_neg_train + num_neg_val + num_neg_test`; it is sampled **once** and
sliced:

- val gets `neg_edge_index[:, :num_neg_val]`
- test gets `neg_edge_index[:, num_neg_val : num_neg_val + num_neg_test]`
- train gets `neg_edge_index[:, num_neg_val + num_neg_test:]`

Which message-passing graph each split sees, verbatim from `forward()`:

```python
self._split(train_store, train_edges[num_disjoint:], is_undirected, rev_edge_type)
self._split(val_store,   train_edges,                is_undirected, rev_edge_type)
self._split(test_store,  train_val_edges,            is_undirected, rev_edge_type)
```

and `_split` symmetrises:

```python
edge_index = store.edge_index[:, index]
if is_undirected:
    edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)
store.edge_index = edge_index
```

So: `train_data.edge_index` = train edges only (both directions).
`val_data.edge_index` = **train edges only** (both directions). `test_data.edge_index`
= train + val edges (both directions). No target edge is ever visible in the graph you
encode from. This is correct link-prediction hygiene and it is why the old
`examples/autoencoder.py` leakage complaint (issue #264, against pre-`RandomLinkSplit`
code) no longer applies.

`negative_sampling` signature, verbatim:

```python
def negative_sampling(
    edge_index: Tensor,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    num_neg_samples: Optional[Union[int, float]] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> Tensor:
```

Sampled negatives are guaranteed not to be existing edges (filtered against the
encoded edge set) and cannot be self-loops (`edge_index_to_vector` removes the
diagonal from the sample space).

### Exact split sizes for Cora with the canonical settings

`RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True,
add_negative_train_samples=False)`:

```
perm.numel()                        = 5278       (row <= col edges)
num_val   = int(0.05 * 5278)        =  263
num_test  = int(0.10 * 5278)        =  527
num_train = 5278 - 263 - 527        = 4488

train_data.edge_index               = [2, 8976]   (4488 * 2)
train_data.pos_edge_label_index     = [2, 4488]
train_data.neg_edge_label_index     = ABSENT      (add_negative_train_samples=False)

val_data.edge_index                 = [2, 8976]   (train edges only)
val_data.pos_edge_label_index       = [2,  263]
val_data.neg_edge_label_index       = [2,  263]

test_data.edge_index                = [2, 9502]   ((4488+263) * 2)
test_data.pos_edge_label_index      = [2,  527]
test_data.neg_edge_label_index      = [2,  527]
```

**The canonical Cora link-prediction test set is 527 positives and 527 negatives.
Base rate 0.5.** This is the single most important number on this page; §Metrics
explains why.

### `train_test_split_edges` is deprecated — confirmed

```python
# torch_geometric/utils/_train_test_split_edges.py
@deprecated("use 'transforms.RandomLinkSplit' instead")
```

The docstring adds that it "is deprecated and will be removed in a future release."
Do not use it.

### GAE / VGAE / InnerProductDecoder — exact current source

Verbatim from `torch_geometric/nn/models/autoencoder.py` (master, 2026-08-25):

```python
EPS = 1e-15
MAX_LOGSTD = 10


class InnerProductDecoder(torch.nn.Module):
    def forward(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True) -> Tensor:
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GAE(torch.nn.Module):
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def encode(self, *args, **kwargs) -> Tensor:
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(GAE):
    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
```

Facts that follow directly from this source and that people routinely get wrong:

1. **`GAE.test()` returns `(roc_auc_score, average_precision_score)`** — AUC first,
   AP second. The AP is sklearn's **non-interpolated** average precision.
2. **`GAE.test()` builds its own label vector** from the two edge-index tensors you
   hand it. Its "AP" reflects whatever balance `pos_edge_index : neg_edge_index`
   happens to have. Hand it 527/527 and you get an AP with a 0.5 baseline. Not a bug —
   it is exactly the published protocol — but it means **`GAE.test()`'s AP is not
   comparable to an AUPRC over masked N² entries.**
3. **`recon_loss` is not `pos_weight`-ed BCE.** It is `mean(-log p)` over positives
   plus `mean(-log(1-p))` over an equal-sized negative sample. Balancing is done by
   *sampling*, not weighting. This differs from Kipf's original TF code (below).
4. `InnerProductDecoder` has **no parameters**; `reset_parameters` on it is a no-op.
   Our D1 (`H W Hᵀ`) needs a `W`, so we cannot literally use PyG's class for D1 — we
   subclass or write our own. The spec's "import it, do not reimplement" is half-right:
   import it for the *baseline*, write our own for D1.
5. `VGAE` stores `__mu__` / `__logstd__` as side effects of `encode()`. `kl_loss()`
   with no args reads the last `encode()`. Call `encode` then `kl_loss`, in that order.
6. `EPS = 1e-15` is added inside the log, so `recon_loss` is finite but its gradient
   near `p=0` is enormous. If you write your own, use `binary_cross_entropy_with_logits`
   on the logits instead — numerically stable, no EPS hack.

### `GCNConv` and `GATConv` — exact current signatures

```python
# torch_geometric/nn/conv/gcn_conv.py
def __init__(
    self,
    in_channels: int,
    out_channels: int,
    improved: bool = False,
    cached: bool = False,
    add_self_loops: Optional[bool] = None,
    normalize: bool = True,
    bias: bool = True,
    **kwargs,
):

def forward(self, x: Tensor, edge_index: Adj,
            edge_weight: OptTensor = None) -> Tensor:
```

Normalisation, from the docstring: `X' = D̂^{-1/2} Â D̂^{-1/2} X Θ`, where `Â = A + I`.

```python
# torch_geometric/nn/conv/gat_conv.py
def __init__(
    self,
    in_channels: Union[int, Tuple[int, int]],
    out_channels: int,
    heads: int = 1,
    concat: bool = True,
    negative_slope: float = 0.2,
    dropout: float = 0.0,
    add_self_loops: bool = True,
    edge_dim: Optional[int] = None,
    fill_value: Union[float, Tensor, str] = 'mean',
    bias: bool = True,
    residual: bool = False,
    **kwargs,
):
```

Output width: `heads * out_channels` when `concat=True` (default), `out_channels` when
`concat=False`. **This trips people up when stacking GAT layers** — layer 2's
`in_channels` must be `heads * out_channels` of layer 1.

### Published GAE/VGAE reference numbers — verified against the paper

Kipf & Welling, *Variational Graph Auto-Encoders*, arXiv 1611.07308. Table 1, fetched
verbatim from ar5iv:

| Method | Cora AUC | Cora AP | Citeseer AUC | Citeseer AP | Pubmed AUC | Pubmed AP |
|---|---|---|---|---|---|---|
| SC | 84.6±0.01 | 88.5±0.00 | 80.5±0.01 | 85.0±0.01 | 84.2±0.02 | 87.8±0.01 |
| DW | 83.1±0.01 | 85.0±0.00 | 80.5±0.02 | 83.6±0.01 | 84.4±0.00 | 84.1±0.00 |
| GAE* | 84.3±0.02 | 88.1±0.01 | 78.7±0.02 | 84.1±0.02 | 82.2±0.01 | 87.4±0.00 |
| VGAE* | 84.0±0.02 | 87.7±0.01 | 78.9±0.03 | 84.1±0.02 | 82.7±0.01 | 87.5±0.01 |
| **GAE** | **91.0±0.02** | **92.0±0.03** | 89.5±0.04 | 89.9±0.05 | 96.4±0.00 | 96.5±0.00 |
| **VGAE** | **91.4±0.01** | **92.6±0.01** | 90.8±0.02 | 92.0±0.02 | 94.4±0.02 | 94.7±0.02 |

(`*` rows are the featureless variants — no `X`. The gap between GAE* 84.3 and GAE
91.0 on Cora is *entirely* the node features. Keep that in mind when reading our own
"feature-only" baseline.)

**The spec's numbers (AUC ~91.0/91.4, AP ~92.0/92.6) are CORRECT**, verified against
the paper's own Table 1.

Protocol, quoted verbatim from the paper:

> "The validation and test sets contain 5% and 10% of citation links, respectively."
>
> "We form validation and test sets from previously removed edges **and the same
> number of randomly sampled pairs of unconnected nodes**."

Hyperparameters, quoted: "32-dim hidden layer and 16-dim latent variables", Adam, "a
learning rate of 0.01", "200 iterations", Glorot init.

### Kipf's original training loss — the detail nobody quotes

From `github.com/tkipf/gae/gae/train.py`, verbatim:

```python
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
```

For Cora: `pos_weight = 693.70`, `norm = 0.50072`.

And from `gae/preprocessing.py::mask_test_edges`:
`num_test = int(np.floor(edges.shape[0] / 10.))`, `num_val = int(np.floor(edges.shape[0] / 20.))`,
with `test_edges_false` / `val_edges_false` sampled 1:1 against the positives.

**This is the central asymmetry of the entire GAE literature:**

- Kipf **trains** with weighted BCE over the **entire N² adjacency matrix**
  (`pos_weight ≈ 694`).
- Kipf **evaluates** on **527 positives + 527 sampled negatives** — base rate 0.5.
- PyG's `GAE.recon_loss` **trains** on a 1:1 sampled pos/neg set instead — a different
  training objective, reaching similar test numbers.

So "GAE gets 92.0 AP on Cora" means "92.0 AP **on a balanced 527/527 test set**". It
does **not** mean 0.92 AUPRC on the sparse matrix.

### Independently reproduced GAE/VGAE numbers using PyG's own code

*On Generalization of Graph Autoencoders with Adversarial Training*, arXiv 2107.02658,
Table 2, fetched from ar5iv. They state they used "the official Pytorch geometric
code", "30 repeat experiments with random splitting datasets into 85%, 5% and 10% for
training sets, validation sets and test sets respectively", 600 epochs, Adam, lr 0.01:

| Model | Cora AUC (%) | Cora AP (%) |
|---|---|---|
| GAE | **90.6 ± 0.9** | **91.2 ± 1.0** |
| VGAE | **89.8 ± 0.9** | **90.3 ± 1.0** |

**PyG's implementation reproduces 0.4–2.3 points BELOW the paper's published table.**
Use 90.6 / 91.2 as the realistic number to beat when you run PyG's `GAE` yourself, and
91.0 / 92.0 as the *published* number to cite. Note also that VGAE lands *below* GAE in
this reproduction, reversing the paper's ordering. The gap between the two is inside one
standard deviation in both studies — **do not claim VGAE > GAE.**

### sklearn / torchmetrics AP semantics

`sklearn.metrics.average_precision_score`, verbatim signature and formula:

```python
sklearn.metrics.average_precision_score(
    y_true, y_score, *, average='macro', pos_label=1, sample_weight=None)
```

> AP = Σₙ (Rₙ − Rₙ₋₁) Pₙ
>
> where Pₙ and Rₙ are the precision and recall at the nth threshold.

And the critical caveat, verbatim from the docs:

> **This implementation is not interpolated and is different from computing the area
> under the precision-recall curve with the trapezoidal rule, which uses linear
> interpolation and can be too optimistic.**

`torchmetrics.classification.BinaryAveragePrecision` inherits from
`BinaryPrecisionRecallCurve`. Its docstring gives the identical formula and states the
value "is equivalent to the area under the precision-recall curve (AUPRC)". Inputs:
`preds` a float tensor `(N, ...)` of probabilities **or logits**; `target` an int tensor
`(N, ...)` in `{0,1}`. `thresholds`:

- `None` → non-binned, exact, memory `O(n_samples)`
- `int` → that many thresholds linearly spaced in `[0,1]`, memory `O(n_thresholds)`,
  **approximate**
- `list`/`Tensor` → those thresholds as bins

Class attrs: `is_differentiable=False`, `higher_is_better=True`.

### Focal loss — verified formulas

Lin et al., *Focal Loss for Dense Object Detection*, arXiv 1708.02002, via ar5iv:

> FL(p_t) = −(1−p_t)^γ log(p_t)

alpha-balanced variant:

> FL(p_t) = −α_t (1−p_t)^γ log(p_t)

Best setting reported: **γ = 2.0 with α = 0.25**. The paper describes the regime it
targets as "an extreme imbalance between foreground and background classes during
training (e.g., 1:1000)". Its argument against α-balancing alone: α weights positives
vs negatives but "cannot differentiate between easily-classified versus hard-to-classify
examples"; the modulating factor `(1−p_t)^γ` down-weights easy examples so training
focuses on hard ones.

### `BCEWithLogitsLoss` `pos_weight` — verified semantics

```python
torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None,
                           reduction='mean', pos_weight=None)
```

Loss, verbatim:

> ℓ_{n,c} = −w_{n,c} [ p_c · y_{n,c} · log σ(x_{n,c}) + (1 − y_{n,c}) · log(1 − σ(x_{n,c})) ]

and the docs' own worked example, verbatim:

> "For example, if a dataset contains 100 positive and 300 negative examples of a
> single class, then `pos_weight` for the class should be equal to 300/100 = 3. The
> loss would act as if the dataset contains 3×100 = 300 positive examples."

### Attention backends and custom bias (bears on §Memory)

From PyTorch `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`, the constraint
arrays, verbatim:

```cpp
// can_use_flash_attention
constexpr auto general_constraints = std::to_array<bool (*)(sdp_params const&, bool)>({
    check_runtime_disabled_flash,
    check_all_tensors_on_device,
    check_tensor_shapes,
    check_for_attn_mask,          // <-- flash rejects ANY non-null attn_mask
    check_fa4_constraints,
    check_head_dim_size_flash<false /*caller_is_meff*/>,
    check_flash_attention_hardware_support,
    check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120,
    check_flash_causal_non_square_seqlens,
    check_dtypes_flash_attention});

// can_use_mem_efficient_attention
constexpr auto general_constraints = std::to_array<bool (*)(sdp_params const&, bool)>({
    check_runtime_disabled_mem_efficient,
    check_all_tensors_on_device,
    check_mem_efficient_hardware_support,
    check_tensor_shapes,
    check_head_dim_size_mem_efficient,
    check_data_ptr_alignment_mem_efficient});   // <-- no check_for_attn_mask
```

`check_for_attn_mask` warns "Flash Attention does not support non-null attn_mask." It
is **absent from the memory-efficient list**, so the mem-efficient (cutlass) backend
accepts a non-null additive float `attn_mask`.

`torch.nn.functional.scaled_dot_product_attention` docs on `attn_mask`, verbatim:

> "A boolean mask where a value of True indicates that the element should take part in
> attention. A float mask of the same type as query, key, value that is added to the
> attention score."

`torch.nn.attention.flex_attention` signature:

```python
flex_attention(query, key, value, score_mod=None, block_mask=None, scale=None,
               enable_gqa=False, return_lse=False, kernel_options=None,
               *, return_aux=None) -> Tensor

def score_mod(score: Tensor, batch: Tensor, head: Tensor,
              q_idx: Tensor, k_idx: Tensor) -> Tensor
```

The PyTorch FlexAttention blog, verbatim, on the relative-position example:

> "Note that unlike typical implementations, this does *not* need to materialize a SxS
> tensor. Instead, FlexAttention computes the bias values 'on the fly' within the
> kernel, leading to significant memory and performance improvements."

and on capturing an external tensor:

```python
bias = torch.randn(1024, 1024)
def score_mod(score, b, h, q_idx, kv_idx):
    return score + bias[q_idx][kv_idx] # The bias tensor can change!
```

**Does gradient flow to that captured bias?** Yes for `score_mod`, no for `mask_mod`.
From `torch/_higher_order_ops/flex_attention.py` on current `main`, verbatim comment:

```python
# We have asserted that mask_mod_other_buffers do not require grad,
# but score_mod_other_buffers can require grad.
```

with the guard:

```python
any_buffer_requires_grad = any(
    buffer.requires_grad
    for buffer in mask_mod_other_buffers
    if isinstance(buffer, torch.Tensor)
)
if any_buffer_requires_grad:
    raise AssertionError(
        "Captured buffers from mask mod that require grad are not supported."
    )
```

and the backward returning `*grad_score_mod_captured`. pytorch/pytorch issue #145460
("Flex Attention not support score_mod with gradients", `AssertionError: Captured
buffers that require grad are not yet supported.`) is **stale for `score_mod`** — the
current restriction on `main` is scoped to `mask_mod` only.

### HF Llama eager attention path (bears on §Memory)

`transformers/models/llama/modeling_llama.py` on `main`, verbatim:

```python
def eager_attention_forward(module, query, key, value, attention_mask,
                            scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1,
        dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout,
        training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

Two consequences we must budget for:

1. `attention_mask` is **added pre-softmax**. Our structural bias can be injected with
   zero surgery by passing a `[B, H, N, N]` float tensor as `attention_mask`. But HF
   builds that tensor itself from the 2-D mask you pass to `forward`, so we must patch
   `_update_causal_mask` / pass a 4-D mask explicitly.
2. `softmax(..., dtype=torch.float32)` **upcasts the N×N tensor to fp32** even in a
   bf16 model. The eager path therefore costs ~10 bytes per attention element, not 2.

### Llama-3.2-1B config (verified)

`meta-llama/Llama-3.2-1B/config.json` is gated (HTTP 401); fetched an unmodified mirror
at `unsloth/Llama-3.2-1B`:

```json
{
  "hidden_size": 2048,
  "intermediate_size": 8192,
  "num_hidden_layers": 16,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 64,
  "max_position_embeddings": 131072,
  "vocab_size": 128256,
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16"
}
```

**The spec's "16 layers, 32 heads" is CORRECT.** Additional facts the spec omits and
that matter: `head_dim = 64`, `num_key_value_heads = 8` (GQA, 4× KV compression — the
*query* head count 32 is what drives attention-matrix memory, not 8), and
`tie_word_embeddings = true`, so dropping the embedding table drops the LM head too.

---

## Corrections to CLAUDE.md

### C1. "Cora ... 5,429 edges" — wrong for PyG. **Confidence: certain.**

§6 and §11 both say 5,429. PyG gives 10,556 directed entries = **5,278 unique
undirected edges**. 151 of the raw 5,429 links are duplicates or self-citations and are
removed by `coalesce`. Every `pos_weight` and density computed from 5,429 will be off by
~2.9 %. Use 5,278 / 10,556.

### C2. "about 0.1% positive" — imprecise, and derived two different ways. **Confidence: certain.**

§11 says "~5,429 edges out of 2,708² ≈ 7.3M possible — about 0.1% positive". `5429 /
7,333,264 = 0.074 %`, which is not 0.1 %. The correct figure, using the symmetric
adjacency PyG actually gives you, is `10,556 / 7,333,264 = **0.144 %**`. The spec's
arithmetic silently mixes an undirected edge count with a full (symmetric) matrix
denominator. Use 0.144 %.

### C3. `pos_weight = num_negatives / num_positives` — right formula, no number given. **Confidence: certain.**

§11's formula is correct and matches both PyTorch's docs and Kipf's original code. The
concrete value for Cora over the full matrix is **693.70** (or 693.44 excluding the
diagonal). Note that Kipf pairs it with `norm = 0.50072` as an outer scale factor; if
you copy `pos_weight` without `norm` your loss magnitude is ~2× Kipf's, which changes
the effective learning rate. Not wrong, but log it.

### C4. "Evaluate on held-out masked entries plus an equal number of sampled true non-edges" — this is a real trap and the spec does not flag it. **Confidence: certain.**

This sentence describes two mutually incompatible protocols in one breath, and the
choice between them changes the headline number by **50×**.

- "held-out masked entries" = ~1.1 M cells at 0.144 % positive if you mask 15 % of N².
- "plus an equal number of sampled true non-edges" = balance it to 50 %.

You cannot do both. And AUPRC is **not** invariant to which you pick. Measured here
(pure-Python AP, sklearn's exact non-interpolated formula, two Gaussians tuned to
AUROC ≈ 0.91, 527 positives — i.e. a simulated GAE-quality scorer):

| neg:pos | #neg | base rate | AUROC | AP | AP / base |
|---|---|---|---|---|---|
| 1:1 | 527 | 0.5000 | 0.9124 | **0.9103** | 1.8× |
| 2:1 | 1,054 | 0.3333 | 0.9091 | 0.8409 | 2.5× |
| 5:1 | 2,635 | 0.1667 | 0.9073 | 0.7246 | 4.3× |
| 10:1 | 5,270 | 0.0909 | 0.9208 | 0.6446 | 7.1× |
| 100:1 | 52,700 | 0.0099 | 0.9187 | 0.2562 | 25.9× |
| 1000:1 | 527,000 | 0.0010 | 0.9054 | 0.0414 | 41.4× |
| 6952:1 | 3,663,704 | 0.00014 | 0.9006 | **0.0174** | 121.1× |

**AUROC stays at ~0.91 across five orders of magnitude of class balance. AP falls from
0.910 to 0.017.** The last row is the real full-upper-triangle setting: 527 held-out
positives against all 3,664,751 non-edges. This is the empirical version of Saito &
Rehmsmeier's claim, which I verified directly:

> "the baseline of PRC is determined by the ratio of positives (P) and negatives (N) as
> y = P / (P + N)" ... "ROC plots are unchanged between balanced and imbalanced
> datasets" whereas "the PRC plots are changed between balanced and imbalanced dataset".

**Consequence for us:** the spec's §11 says "Report AUPRC, not accuracy and not AUROC"
and §7 says GAE's 92.0 AP is "the number to beat". Those two instructions are
inconsistent as written. If we report AUPRC on masked N² entries we will get ~0.02–0.30
and it will look like we lost to GAE by 60 points, when in fact we never ran the same
evaluation. **Report both, always, and label the denominator.** See §Metrics for the
exact protocol.

### C5. "Attention is O(N²) and the custom bias blocks FlashAttention ... Do not use FlashAttention. Use the eager attention path." — outdated. **Confidence: likely.**

§11 and §13 are half right and the practical conclusion is wrong.

- Correct: FlashAttention rejects any non-null `attn_mask` (`check_for_attn_mask` is in
  its constraint list, verbatim above).
- **Missing:** the memory-efficient (cutlass) SDPA backend has *no* such check and does
  accept an additive float mask.
- **Missing and more important:** `torch.nn.attention.flex_attention` exists precisely
  for this. `score_mod` adds an arbitrary bias inside a fused FlashAttention-style
  kernel and, per the blog, "does *not* need to materialize a SxS tensor". Gradients
  **do** flow to a learned tensor captured in `score_mod` on current PyTorch `main`
  (verbatim comment quoted above); only `mask_mod` captures are barred.

Following the spec literally (eager everywhere) costs us ~35 GB of attention
activations at N=2708 and rules out a 40 GB card. FlexAttention is the correct target.
I mark this **likely** rather than certain because I verified the code paths but did not
run FlexAttention with a learned per-head SPD bias table — the `q_idx/kv_idx` gather
pattern `bias_table[h, spd[q_idx, kv_idx]]` needs a real benchmark before we commit.

### C6. "GTLM ... reported limitation: custom attention biases are incompatible with FlashAttention, so O(N²) attention" — inherited, not independently checked. **Confidence: uncertain.**

§9 attributes this to GTLM (arXiv 2605.10247). **I did not fetch that paper** (it is
another researcher's assignment) and cannot confirm what it says. But if the claim is
"custom bias ⇒ must be O(N²)", it is false as a general statement about 2026 PyTorch,
for the reasons in C5. It may well have been true for the specific HF version they used.
Do not repeat it as a general fact in our writeup.

### C7. "Our decoder D1 is `InnerProductDecoder` with a learned W inserted. Import it, do not reimplement." — not literally possible. **Confidence: certain.**

§9. `InnerProductDecoder.forward` is `(z[i] * z[j]).sum(dim=1)` with **no parameters and
no hook** for a `W`. There is nowhere to "insert" a learned `W` without rewriting the
method. Import it for the *GAE baseline*; write our own ~10-line module for D1.

### C8. "GAE / VGAE ... This is the number to beat" — needs a caveat. **Confidence: certain.**

§7. Two problems. (a) The published 91.0/92.0 is on a **balanced 527/527 test set**, not
on our masked-N² task — see C4. (b) PyG's own implementation reproduces at **90.6 ± 0.9
AUC / 91.2 ± 1.0 AP** (arXiv 2107.02658, 30 runs), below the published table. Beating
"92.0" as printed in the paper is not the same as beating the GAE you can actually run.
Report our own GAE run as the baseline, with its seed variance, and cite the paper
number separately.

### C9. VGAE > GAE is not a safe claim. **Confidence: likely.**

The spec lists them together without ordering, which is fine, but for the writeup: Kipf
reports VGAE above GAE on Cora (91.4 vs 91.0 AUC); arXiv 2107.02658 reports VGAE
*below* GAE (89.8 vs 90.6). Both gaps are within one std. Treat them as tied.

### C10. `train_test_split_edges` deprecation — the spec never mentions it, which is correct by omission. **Confidence: certain.**

No correction needed; recorded here so nobody reintroduces it. It carries
`@deprecated("use 'transforms.RandomLinkSplit' instead")`.

### C11. Llama-3.2-1B "16 layers, 32 heads" — confirmed correct. **Confidence: certain.**

No correction. Adding the missing numbers that matter for our budget: `head_dim=64`,
`hidden_size=2048`, `num_key_value_heads=8` (GQA), `intermediate_size=8192`,
`tie_word_embeddings=true`.

---

## Mechanism / math (exact formulas, tensor shapes)

### Why AUPRC moves with class balance and AUROC does not

Let `P` = number of positives, `N` = number of negatives in the evaluation set. At any
score threshold with `TP` true positives and `FP` false positives:

```
Recall    = TP / P                  <- does not involve N
Precision = TP / (TP + FP)          <- involves N through FP
TPR       = TP / P
FPR       = FP / N                  <- normalised by N
```

`TPR` and `FPR` are each normalised **within** their own class, so the ROC curve is
invariant if you subsample negatives uniformly. Precision is **not** normalised: scale
`N` by `k` and, in expectation, `FP → k·FP`, so

```
Precision_k = TP / (TP + k·FP)
```

which decreases monotonically in `k`. Hence AUPRC decreases as you add negatives, with
the random-classifier baseline sitting at `P / (P + N)`.

Numerically for Cora, held-out test positives = 527:

```
balanced (PyG / Kipf):     P=527,  N=527        baseline AP = 0.5000
full upper triangle:       P=527,  N=3,664,751  baseline AP = 0.000144
ratio of baselines:                             3475x
```

### Non-interpolated AP vs interpolated PR-AUC — and which to report

sklearn's `average_precision_score`:

```
AP = Σₙ (Rₙ − Rₙ₋₁) · Pₙ
```

This is a right-hand Riemann sum over the step-wise PR curve: no interpolation between
operating points. **This is what you should report.** It is also what PyG's
`GAE.test()` and torchmetrics' `BinaryAveragePrecision` compute, so it is
apples-to-apples with the published literature.

The alternative, "PR-AUC" computed as `sklearn.metrics.auc(recall, precision)`, applies
the **trapezoidal rule**, i.e. straight lines between PR points. Davis & Goadrich 2006
(ICML) show this is wrong. Verbatim from §4 "Interpolation and AUC":

> "However, in Precision-Recall space, interpolation is more complicated. As the level
> of Recall varies, the Precision does not necessarily change linearly due to the fact
> that FP replaces FN in the denominator of the Precision metric. In these cases, linear
> interpolation is a mistake that yields an overly-optimistic estimate of performance."

The correct interpolation between two PR points `A = (TP_A, FP_A)` and `B = (TP_B, FP_B)`
is nonlinear — the "local skew" construction. Verbatim:

> "We find out how many negative examples it takes to equal one positive, or the local
> skew, defined by (FP_B − FP_A)/(TP_B − TP_A). Now we can create new points TP_A + x
> for all integer values of x such that 1 ≤ x ≤ TP_B − TP_A ... Our resulting
> intermediate Precision-Recall points will be
>
> ( (TP_A + x) / TotalPos ,  (TP_A + x) / (TP_A + x + FP_A + ((FP_B − FP_A)/(TP_B − TP_A))·x) )."

And the magnitude of the error, verbatim, from their own worked example:

> "Consider a curve (Figure 6) constructed from a single point of (0.02, 1), and
> extended to the endpoints of (0, 1) and (1, 0.008) as described above (for this
> example, our dataset contains 433 positives and 56,164 negatives). Interpolating as we
> have described would have an AUC-PR of 0.031; a linear connection would severely
> overestimate with an AUC-PR of 0.50."

**0.031 vs 0.50 — a 16× overstatement, in exactly our regime** (433 positives against
56,164 negatives is a 130:1 imbalance; ours is 694:1 or worse). Never compute PR-AUC by
trapezoid on a sparse graph.

Related, from the same paper — Theorem 3.2 and Corollary 3.1, verbatim:

> **Theorem 3.2.** *For a fixed number of positive and negative examples, one curve
> dominates a second curve in ROC space if and only if the first dominates the second in
> Precision-Recall space.*
>
> **Corollary 3.1.** *Given a set of points in PR space, there exists an achievable PR
> curve that dominates the other valid curves that could be constructed with these
> points.*

and:

> "In PR space, there exists an analogous curve to the convex hull in ROC space, which
> we call the achievable PR curve, although it cannot be achieved by linear
> interpolation."

Also worth knowing for the writeup: "Algorithms that optimize the area under the ROC
curve are not guaranteed to optimize the area under the PR curve."

### `pos_weight` does not change the optimal ranking — derivation

This settles whether `pos_weight` interacts with AUPRC/AUROC. Let `q = σ(x)` be the
model's output for a cell whose true label is Bernoulli(`p`). The weighted BCE with
`pos_weight = w` has expected loss

```
L(q) = − [ w · p · log q  +  (1 − p) · log(1 − q) ]
```

Differentiate and set to zero:

```
dL/dq = − w·p/q + (1 − p)/(1 − q) = 0
    ⇒  w·p·(1 − q) = (1 − p)·q
    ⇒  w·p = q·(1 − p + w·p)
    ⇒  q* = w·p / (1 − p + w·p)
```

`q*` is a strictly increasing function of `p` for any `w > 0`
(`dq*/dp = w / (1 − p + w·p)² > 0`). Therefore:

- **Ranking is preserved.** The Bayes-optimal scores under weighted BCE are a monotone
  transform of the Bayes-optimal scores under plain BCE. **AUROC and AUPRC of the
  optimum are unchanged by `pos_weight`.**
- **Calibration is destroyed.** At `w = 694` and `p = 0.001`, `q* = 0.694/1.693 = 0.41`.
  The model outputs 0.41 for a cell that is a 1-in-1000 edge. **Never threshold at 0.5
  on a `pos_weight`-trained model** and never report accuracy, precision or F1 at a
  fixed 0.5 threshold. If you need a hard threshold, pick it on validation.
- `pos_weight` matters for **optimisation**, not for the metric's definition: without
  it, the gradient from 693 negatives per positive swamps the positive signal and the
  model collapses to predicting 0. Its job is to keep the positive gradient alive.

Sanity check on the loss scale: with `reduction='mean'` and `pos_weight=w`, the mean
loss is dominated by the `w`-scaled positive terms. Kipf's `norm` factor
(`N² / (2·(N² − ΣA))` = 0.50072) rescales the whole thing back so the loss magnitude is
comparable to unweighted BCE. If you copy `pos_weight` but not `norm`, your effective
learning rate is ~2× Kipf's.

### Memory arithmetic — exact

`N = 2708`, `N² = 7,333,264`. One `N×N` matrix:

```
fp32: 7,333,264 × 4 B = 28.0 MB
bf16: 7,333,264 × 2 B = 14.0 MB
```

Llama-3.2-1B: `L = 16` layers, `H = 32` **query** heads (GQA's 8 KV heads do not reduce
the attention-matrix count — `repeat_kv` expands K/V back to 32 before the matmul).

```
                                  fp32        bf16
1 head, 1 layer                  28.0 MB     14.0 MB
32 heads, 1 layer               895.2 MB    447.6 MB
32 heads × 16 layers, 1 tensor   13.99 GB     6.99 GB
```

That "1 tensor" figure is the theoretical floor — only the softmax output kept for
backward. The **actual** HF eager path allocates four `N×N` tensors per head-layer, and
one of them is fp32 because of the `dtype=torch.float32` upcast in `softmax`:

| tensor | dtype | bytes/elem | all 32 heads × 16 layers |
|---|---|---|---|
| `q @ kᵀ * scaling` | bf16 | 2 | 6.99 GB |
| `+ attention_mask` | bf16 | 2 | 6.99 GB |
| `softmax(..., dtype=float32)` | **fp32** | **4** | **13.99 GB** |
| `.to(query.dtype)` | bf16 | 2 | 6.99 GB |
| **total** | | **10** | **34.97 GB** |

Plus the bias tensor itself, `[1, H, N, N]`, which is shared across layers if you build
it once: 447.6 MB in bf16, plus 447.6 MB for its gradient = **895 MB**.

Full-stack budget, eager, bf16 model, N=2708, batch 1, no gradient checkpointing:

```
attention activations                 34.97 GB
bias tensor + its grad                 0.87 GB
frozen weights (1.24 B params, bf16)   2.30 GB
other activations (qkvo + MLP, 16 L)  ~2.64 GB
--------------------------------------------
TOTAL                                ~40.78 GB
```

**Verdict: does full-graph Cora attention fit?**

| GPU | Fits? | Notes |
|---|---|---|
| 24 GB (4090, A10G) | **NO** | off by 17× on the attention term alone |
| 40 GB (A100-40) | **NO** | 40.78 GB vs 40 GB before any fragmentation headroom |
| 80 GB (A100-80, H100) | **YES**, ~51 % utilisation | batch size 1 only; no room to grow |

With **gradient checkpointing** (recompute each layer's attention in backward, so only
one layer's `N×N` tensors are live at a time):

```
attention activations, 1 layer         2.19 GB
+ bias + weights + other acts
--------------------------------------------
TOTAL                                 ~8.00 GB     -> fits 24 GB comfortably
```

Max `N` that fits, solving `bytes_per_elem × H × L_live × N² ≤ 0.60 × budget`:

| Budget | eager, no checkpointing | eager + grad checkpointing |
|---|---|---|
| 24 GB | N ≈ **1,737** | N ≈ 6,951 |
| 40 GB | N ≈ **2,243** | N ≈ 8,973 |
| 80 GB | N ≈ **3,172** | N ≈ 12,690 |

Note that N ≈ 2,243 on a 40 GB card lands *just below* Cora's 2,708 — which is why the
spec's "measure this before building anything elaborate" is the right instinct and its
"use the eager attention path" is the wrong conclusion.

### Subgraph-sampling fallback — concrete recipe

If we stay on the eager path or on a 24 GB card, the fallback the spec gestures at
("sample k-hop subgraphs, cap at 500–1,000 nodes") is sound and here is what it costs:

```
N = 1024:  eager attention activations = 32 heads × 16 layers × 1024² × 10 B = 5.00 GB
N =  512:                                                                     1.25 GB
N =  256:                                                                     0.31 GB
```

Sampler: `torch_geometric.loader.NeighborLoader` (k-hop) or
`torch_geometric.utils.k_hop_subgraph`. For masked edge prediction the sampled subgraph
must be **induced** — every edge among the sampled nodes must be present, or the labels
are wrong. `k_hop_subgraph(..., relabel_nodes=True)` returns the induced edge set;
`NeighborLoader` does **not** by default (it samples a fixed fan-out) — pass
`subgraph_type='induced'` or post-process. **UNVERIFIED:** I did not fetch the current
`NeighborLoader` signature to confirm the `subgraph_type` argument name; check it before
relying on it.

Ordering of preference, cheapest fix first:

1. **FlexAttention with a `score_mod` bias** — removes the `N²` activation term entirely.
   Full-graph Cora then fits in 24 GB. This is the right answer; validate it in week one.
2. **Gradient checkpointing on the eager path** — 8 GB, no kernel work, ~1.3–1.5× slower.
   The safe fallback if FlexAttention's `score_mod` gather is too slow.
3. **SDPA with the mem-efficient backend and a `[B,H,N,N]` float mask** — you still pay
   895 MB for the mask, but not the 35 GB of activations. Middle ground.
4. **k-hop subgraph sampling at N ≤ 1024** — last resort, because it changes the task
   (the model never sees the whole graph) and makes comparison to full-graph GAE unfair.

---

## Code we can reuse (real snippets, real signatures)

### 1. Canonical Cora link-prediction loader — full working snippet

```python
import os.path as osp
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = T.Compose([
    T.NormalizeFeatures(),          # row-normalise the 1433-dim bag of words
    T.ToDevice(device),
    T.RandomLinkSplit(
        num_val=0.05,               # 263 edges
        num_test=0.10,              # 527 edges
        is_undirected=True,         # ratios are of the 5278 row<=col edges
        split_labels=True,          # -> pos_edge_label_index / neg_edge_label_index
        add_negative_train_samples=False,  # we resample negatives every step
    ),
])

path = osp.join('data', 'Planetoid')
dataset = Planetoid(path, 'Cora', transform=transform)
train_data, val_data, test_data = dataset[0]

# Shapes you will actually get (verified by reading RandomLinkSplit.forward):
#   train_data.edge_index            [2, 8976]
#   train_data.pos_edge_label_index  [2, 4488]
#   val_data.edge_index              [2, 8976]    <- TRAIN edges only
#   val_data.pos_edge_label_index    [2,  263]
#   val_data.neg_edge_label_index    [2,  263]
#   test_data.edge_index             [2, 9502]    <- train + val edges
#   test_data.pos_edge_label_index   [2,  527]
#   test_data.neg_edge_label_index   [2,  527]
```

Gotchas, in order of how often they bite:

- **`split_labels=True` vs `False` changes the attribute names.** `True` gives
  `pos_edge_label_index` / `neg_edge_label_index`. `False` gives one concatenated
  `edge_label_index` plus a `edge_label` vector of 1s and 0s. PyG's
  `examples/autoencoder.py` uses `True`; `examples/link_pred.py` uses `False`. Pick one
  and be consistent.
- **`add_negative_train_samples=False` means `train_data` has no `neg_*` attributes at
  all** — not empty tensors, absent. Guard with `hasattr`.
- **`is_undirected=True` does not make the labels undirected.** Only one direction of
  each held-out edge appears in `pos_edge_label_index`. If your decoder is symmetric this
  is what you want; if you score `A[i,j]` and `A[j,i]` separately, you will double-count.
- **`T.ToDevice(device)` before `RandomLinkSplit`** means `negative_sampling` runs on GPU.
  Fine, but it makes the split non-reproducible across devices; set `torch.manual_seed`
  and record the device.
- `transform` runs **on every access** to `dataset[0]`. Calling `dataset[0]` twice gives
  you two *different* random splits. Assign once.

### 2. GAE / VGAE baseline — PyG's `examples/autoencoder.py`, verbatim

This is the file to copy. Fetched verbatim from `master`:

```python
import argparse
import os.path as osp
import time

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--epochs', type=int, default=400)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=transform)
train_data, val_data, test_data = dataset[0]


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


in_channels, out_channels = dataset.num_features, 16

if not args.variational and not args.linear:
    model = GAE(GCNEncoder(in_channels, out_channels))
elif not args.variational and args.linear:
    model = GAE(LinearEncoder(in_channels, out_channels))
elif args.variational and not args.linear:
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
elif args.variational and args.linear:
    model = VGAE(VariationalLinearEncoder(in_channels, out_channels))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    auc, ap = test(test_data)
    print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
```

Key details worth stating explicitly:

- **Latent dim is 16, hidden is 32** (`2 * out_channels`). Matches Kipf's 32-16.
- **Adam, lr = 0.01, 400 epochs** (Kipf used 200). Dropout 0.
- **KL weight is `1 / num_nodes` = 1/2708 ≈ 3.7e-4**, not 1. Forgetting this makes VGAE
  collapse to the prior.
- This script **selects no model on validation** — it prints test metrics every epoch and
  the last line is whatever epoch 400 gave. For our baseline table, add early stopping on
  `val` AUC and report the corresponding test number, as `examples/link_pred.py` does.

### 3. A correct evaluation module for our task

The one thing we must get right. Both denominators, always, side by side.

```python
"""eval.py -- AUPRC on masked entries, reported two ways.

The published GAE/VGAE number (AP 92.0 on Cora) is measured on a BALANCED
test set: 527 held-out positives against 527 sampled non-edges.  An AUPRC
over all masked N^2 cells is a different quantity by ~50x.  We report both
and label the denominator, every time.
"""
from dataclasses import dataclass

import torch
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class LinkPredMetrics:
    # sparse: every masked cell, base rate ~0.00144 on Cora
    auprc_sparse: float
    auroc_sparse: float
    base_rate_sparse: float
    lift_sparse: float          # auprc_sparse / base_rate_sparse

    # balanced: all positives + an equal number of sampled negatives.
    # THIS is the column comparable to Kipf's 92.0.
    auprc_balanced: float
    auroc_balanced: float

    n_pos: int
    n_neg_sparse: int


@torch.no_grad()
def evaluate(logits: torch.Tensor,
             target: torch.Tensor,
             mask: torch.Tensor,
             generator: torch.Generator | None = None) -> LinkPredMetrics:
    """
    logits  [N, N]  raw scores, pre-sigmoid
    target  [N, N]  0/1 ground truth
    mask    [N, N]  bool, True where the entry was held out

    Metrics are computed ONLY on mask==True cells.  Sigmoid is a monotone
    transform, so AUPRC/AUROC are identical on logits and on probabilities.
    Skip the sigmoid: it costs time and loses precision in the tails.
    """
    y = target[mask].to(torch.int8).cpu().numpy()
    s = logits[mask].float().cpu().numpy()

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    assert n_pos > 0, "no positives in the masked set -- widen the mask or reseed"

    base = n_pos / len(y)
    ap_sparse = average_precision_score(y, s)
    auc_sparse = roc_auc_score(y, s)

    # --- balanced view, matching the published protocol ---
    pos_idx = (target[mask] == 1).nonzero(as_tuple=True)[0]
    neg_idx = (target[mask] == 0).nonzero(as_tuple=True)[0]
    pick = torch.randperm(neg_idx.numel(), generator=generator,
                          device=neg_idx.device)[:n_pos]
    sel = torch.cat([pos_idx, neg_idx[pick]])

    y_bal = target[mask][sel].to(torch.int8).cpu().numpy()
    s_bal = logits[mask][sel].float().cpu().numpy()

    return LinkPredMetrics(
        auprc_sparse=float(ap_sparse),
        auroc_sparse=float(auc_sparse),
        base_rate_sparse=base,
        lift_sparse=float(ap_sparse) / base,
        auprc_balanced=float(average_precision_score(y_bal, s_bal)),
        auroc_balanced=float(roc_auc_score(y_bal, s_bal)),
        n_pos=n_pos,
        n_neg_sparse=n_neg,
    )
```

Rules this encodes, each one a real trap:

1. **`average_precision_score`, never `auc(recall, precision)`.** The trapezoid
   overstates by up to 16× at our imbalance (Davis & Goadrich's own example: 0.031 vs
   0.50). sklearn's AP is non-interpolated by design.
2. **Feed logits, not probabilities.** Both metrics are rank-based; sigmoid is monotone,
   so the values are identical, and in float32 a logit of −40 and −60 both sigmoid to
   exactly 0.0 and become a tie, which *does* change AP.
3. **Report `lift = AUPRC / base_rate`.** On the sparse view, an AUPRC of 0.25 sounds
   terrible and is actually a 174× lift over random. Without the lift column the sparse
   number is uninterpretable.
4. **Always report the balanced column too**, and say in the table caption that it is the
   Kipf-comparable one.
5. Seed the balanced negative sample and report the mean over ≥5 draws — a single
   527-negative draw has visible variance.

For a streaming / GPU-side version, `torchmetrics.classification.BinaryAveragePrecision`
with `thresholds=None` gives the identical number:

```python
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC

ap = BinaryAveragePrecision(thresholds=None).to(device)   # exact, O(n_samples) memory
auc = BinaryAUROC(thresholds=None).to(device)
ap.update(logits[mask], target[mask].long())
```

Do **not** set `thresholds=int` here. Binning to e.g. 100 thresholds in `[0,1]` on a
problem where almost all the useful signal lives in the top 0.14 % of scores will
quantise away the part of the curve you care about. Use `thresholds=None` and eat the
memory (1.1 M floats = 4.4 MB, trivial).

### 4. Loss with the right `pos_weight`

```python
import torch
import torch.nn.functional as F

def masked_weighted_bce(logits, target, mask, pos_weight=None, norm=True):
    """
    logits/target/mask : [N, N]
    pos_weight : scalar tensor.  If None, computed from the masked set.
    norm : apply Kipf's outer rescale so the loss magnitude matches
           unweighted BCE (keeps the effective LR comparable).
    """
    y = target[mask]
    x = logits[mask]

    if pos_weight is None:
        n_pos = y.sum().clamp(min=1.0)
        n_neg = y.numel() - n_pos
        pos_weight = n_neg / n_pos              # 693.70 on full-matrix Cora

    loss = F.binary_cross_entropy_with_logits(
        x, y, pos_weight=pos_weight, reduction='mean')

    if norm:
        # Kipf: norm = N^2 / (2 * (N^2 - sum(A))) = 0.50072 for Cora
        n_pos = y.sum().clamp(min=1.0)
        n_neg = y.numel() - n_pos
        loss = loss * (y.numel() / (2.0 * n_neg))

    return loss
```

Compute `pos_weight` **from the training split only**, and recompute it per dataset —
hardcoding 693.7 will silently break the moment we move to PubMed or a molecular graph.

### 5. FlexAttention structural bias (the memory fix) — sketch

Not yet run. Included so the shape of the solution is on record.

```python
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

flex_attention = torch.compile(flex_attention, dynamic=False)

# Learned per-head SPD bias table, exactly the Graphormer / GTLM mechanism.
# [H, max_spd + 2]  -- +2 for "unreachable" and "self"
spd_bias = torch.nn.Parameter(torch.zeros(32, 10, device='cuda'))
spd = shortest_path_distance(A).clamp(max=8)          # [N, N] int64, 9 = unreachable

def graph_bias(score, b, h, q_idx, kv_idx):
    return score + spd_bias[h, spd[q_idx, kv_idx]]

out = flex_attention(q, k, v, score_mod=graph_bias)
out.sum().backward()      # spd_bias.grad is populated
```

Why this should work, with the receipts:

- `score_mod` receives `h`, so the bias is **per-head** — which is what GaLA's per-head
  `lambda_h` argument says we want, without a separate calibration step.
- Gradients reach `spd_bias`. Verbatim from `torch/_higher_order_ops/flex_attention.py`
  on `main`: *"We have asserted that mask_mod_other_buffers do not require grad, but
  score_mod_other_buffers can require grad."* The backward returns
  `*grad_score_mod_captured`.
- No `N×N` activation is materialised (blog, verbatim: *"this does not need to
  materialize a SxS tensor"*), which removes the entire 35 GB term from §Memory.

Risks to test first, in order:

1. The gather `spd[q_idx, kv_idx]` is a **2-D indexed load inside the kernel**. The blog
   only demonstrates a 1-D capture (`alibi_bias[h]`) and a 2-D one
   (`bias[q_idx][kv_idx]`) with no learned parameter. Benchmark before committing.
2. `spd` itself is `[N, N]` int64 = 58.7 MB at N=2708. Store it as `uint8` (7.3 MB) since
   values are ≤ 9.
3. FlexAttention historically required sequence lengths to be a multiple of 128 (fixed,
   per the blog's "Limitations"), and there is an open NaN bug (#153799) when a
   `block_mask` **and** a `score_mod` are combined at a non-multiple-of-128 length.
   N=2708 is not a multiple of 128 (2708 = 21×128 + 20). **Pad to 2816 and mask, or
   verify on the installed version.**
4. `torch.compile` on `flex_attention` must not recompile every step. The blog states
   captured tensors changing value does **not** trigger recompilation, which is what we
   need for a bias table under gradient descent.

### 6. From-scratch transformer over adjacency-row tokens

The spec's single most important baseline. This isolates the pretrained weights.

```python
import torch
import torch.nn as nn

class ScratchAdjTransformer(nn.Module):
    """Depth-Width-paper node-adjacency tokenization, no pretrained weights.
    Token i = row i of A, optionally concatenated with x_i.
    """
    def __init__(self, n_nodes, n_feats=0, d_model=256, nhead=8,
                 num_layers=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_nodes + n_feats, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.W = nn.Parameter(torch.eye(d_model) * 0.1)   # our D1 decoder

    def forward(self, A, X=None):
        # A: [B, N, N] float (masked entries zeroed), X: [B, N, F] or None
        tok = A if X is None else torch.cat([A, X], dim=-1)
        h = self.enc(self.inp(tok))                       # [B, N, d_model]
        return h @ self.W @ h.transpose(1, 2)             # [B, N, N] logits
```

Match `d_model`, `num_layers` and `nhead` as closely as the LLM allows when you compare —
otherwise the ablation measures capacity, not pretraining. Llama-3.2-1B is
`d_model=2048, L=16, H=32`; a from-scratch model at that size on Cora will overfit
instantly, so run a **capacity sweep** (d_model ∈ {128, 256, 512, 1024, 2048}) and report
the from-scratch *curve*, not a single point. That is the honest version of the
comparison and it is much harder to attack in review.

---

## Numbers to beat / hyperparameters to copy

### Reference numbers, Cora link prediction, BALANCED 527/527 test set

| Model | AUC | AP | Source |
|---|---|---|---|
| Spectral Clustering (SC) | 84.6 | 88.5 | Kipf & Welling 2016, Table 1 |
| DeepWalk (DW) | 83.1 | 85.0 | same |
| GAE* (no features) | 84.3 | 88.1 | same |
| VGAE* (no features) | 84.0 | 87.7 | same |
| **GAE (published)** | **91.0** | **92.0** | same |
| **VGAE (published)** | **91.4** | **92.6** | same |
| GAE (PyG reproduction, 30 runs) | 90.6 ± 0.9 | 91.2 ± 1.0 | arXiv 2107.02658 Table 2 |
| VGAE (PyG reproduction, 30 runs) | 89.8 ± 0.9 | 90.3 ± 1.0 | same |
| Random | 50.0 | 50.0 | analytic (base rate 0.5) |

**Beat 91.2 AP to beat the GAE we can actually run. Beat 92.0 to beat the paper.**
Report both, with our own seed variance over ≥10 runs.

### Reference numbers on the SPARSE view — nobody publishes these

There is no published AUPRC for GAE over all N² Cora cells. **We must generate it
ourselves**, by taking the trained PyG GAE, scoring every masked cell, and running the
sparse column of our `evaluate()`. This is a genuinely useful contribution to the
baseline table and takes about twenty minutes. Expected magnitude, extrapolating from
the simulation in C4 (AUROC 0.91 → AP ≈ 0.017 at 6952:1): somewhere in **0.02–0.30**
depending on how much better than my Gaussian toy the real GAE ranking is. **Do not
quote that range as a result** — measure it.

### Hyperparameters to copy verbatim

**GAE / VGAE (from Kipf's `train.py` and PyG's `examples/autoencoder.py`):**

| Param | Value | Source |
|---|---|---|
| hidden1 (GCN layer 1 out) | 32 | Kipf `flags` / PyG `2 * out_channels` |
| hidden2 (latent) | 16 | both |
| optimizer | Adam | both |
| learning rate | 0.01 | both |
| epochs | 200 (Kipf) / 400 (PyG) / 600 (arXiv 2107.02658) | all three |
| dropout | 0.0 | Kipf `flags` |
| init | Glorot | Kipf, paper text |
| VGAE KL weight | `1 / num_nodes` = 3.69e-4 | PyG example |
| `pos_weight` (Kipf full-matrix loss) | 693.70 | computed from his formula |
| `norm` (Kipf outer scale) | 0.50072 | computed from his formula |

**GAT (spec §7 baseline), following the original Veličković setup for Cora:**
`GATConv(1433, 8, heads=8, dropout=0.6)` → ELU → `GATConv(64, out, heads=1, concat=False,
dropout=0.6)`, Adam lr 0.005, weight_decay 5e-4, dropout 0.6 on inputs and attention.
**UNVERIFIED** — I did not fetch the GAT paper or PyG's `examples/gat.py` this session.
These are the widely-repeated Cora numbers and should be checked before use. Note the
layer-2 `in_channels=64` follows from `heads * out_channels = 8 * 8`.

**Focal loss, if we try it:** γ = 2.0, α = 0.25 (Lin et al., verified). But see below —
try `pos_weight` first.

**Splits:** `num_val=0.05, num_test=0.10, is_undirected=True`. Everyone uses this;
deviating makes our numbers incomparable for no gain.

### `pos_weight` vs focal loss — the evidence, not an opinion

The spec says "Use `pos_weight` or focal loss." Here is what the sources actually
support:

**In favour of `pos_weight` as the default:**

1. **It is what the reference implementation does.** Kipf's GAE trains with exactly
   `pos_weight = (N² − ΣA)/ΣA` on the full matrix and reaches the published 91.0/92.0.
   That is direct evidence it is sufficient for this task at this imbalance.
2. **It provably does not distort the ranking metric.** The derivation in §Mechanism
   shows `q* = wp/(1−p+wp)` is a monotone transform of `p`, so the Bayes-optimal AUPRC
   and AUROC are unchanged. No hidden interaction with the metric we report.
3. **One scalar, no tuning.** `n_neg/n_pos` is determined by the data.

**In favour of focal loss:**

1. Lin et al. show, in the 1:1000 regime, that α-balancing alone "cannot differentiate
   between easily-classified versus hard-to-classify examples", and that the modulating
   factor `(1−p_t)^γ` is what closes the gap to two-stage detectors. Our regime is
   1:694, squarely inside theirs.
2. On a 7.3 M-cell matrix, the overwhelming majority of negatives are trivially easy
   (two random papers in a citation network). `pos_weight` keeps re-spending gradient on
   them; focal loss stops.

**Honest verdict.** The evidence supports `pos_weight` as the default because it is the
protocol under which the number we are trying to beat was produced, and because it is
metric-neutral by construction. Focal loss is a well-motivated *second* experiment, not a
replacement. **I found no paper measuring focal loss vs `pos_weight` for graph link
prediction specifically** — Lin et al. is object detection, and I did not find a graph
paper making the comparison. Anyone claiming focal loss is better *for this task* is
extrapolating. Treat it as an ablation with a hypothesis, run both, report both.

Two practical notes:

- Focal loss with `reduction='mean'` shrinks the loss scale a lot (the `(1−p_t)^γ` factor
  is ≪1 for most terms), so the effective learning rate drops. Retune the LR when you
  switch, or you will conclude focal loss is worse when you have only lowered the LR.
- Both are training-time choices. Neither changes what we report. Do not let a loss
  ablation leak into the metrics table.

### Baseline recipes — concrete

| # | Baseline | Recipe | Expected balanced AP | What it tells us |
|---|---|---|---|---|
| 1 | **Identity** | `logits = A_input * BIG` (masked cells → 0 → sigmoid 0.5). Since masked cells are zeroed in the input, this predicts "no edge" on every masked cell. | **≈ base rate** (0.5 balanced / 0.00144 sparse) | Confirms masking actually hides the answer. **If this scores above base rate, we have leakage.** Run it first, every time the loader changes. |
| 2 | **Random at base rate** | `logits = torch.rand_like(A)`, or the constant `logit(0.00144)`. | 0.500 balanced, 0.00144 sparse | The floor. Also the sanity check that `evaluate()` is not inverted. |
| 3 | **Feature-only, logreg** | `sklearn.linear_model.LogisticRegression(class_weight='balanced', max_iter=1000)` on `[x_i ‖ x_j ‖ x_i*x_j]`, 1433×3 = 4299 dims. Train on the 4488 train positives + an equal random negative sample. **No graph.** | see note | Isolates how much of link prediction is just feature similarity. Kipf's GAE* (no features) 88.1 vs GAE 92.0 says features carry ~4 AP points; this measures the other direction. |
| 3b | **Feature-only, XGBoost** | `XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, scale_pos_weight=n_neg/n_pos, tree_method='hist')` on the same pairs. Consider reducing 4299 dims with TruncatedSVD(128) first — 4299 sparse binary features × 9k rows is fine, but the N² inference is not. | see note | Nonlinear feature-only ceiling. |
| 4 | **GAE** | PyG `examples/autoencoder.py` verbatim, 32-16, Adam 0.01, 400 epochs, early stop on val AUC. | **91.2 ± 1.0** | The number to beat. |
| 4b | **VGAE** | same, `--variational`, KL weight `1/2708`. | 90.3 ± 1.0 | Tied with GAE; report both. |
| 5 | **GAT** | `GATConv(1433, 8, heads=8, dropout=0.6)` → ELU → `GATConv(64, 16, heads=1, concat=False)`, wrapped in `GAE(...)` so the loss and `test()` are identical to #4. Adam lr 0.005, wd 5e-4. | **UNVERIFIED**, expect ≈ GAE | Isolates attention-vs-convolution *inside* the GNN family. Wrapping in `GAE` is the trick that keeps it comparable — same decoder, same loss, same `test()`. |
| 6 | **Scratch transformer** | §Code #6. Capacity sweep d_model ∈ {128, 256, 512, 1024, 2048}, 4 layers, AdamW 3e-4, cosine, dropout 0.1. Same `evaluate()`. | unknown | **The most important baseline in the project.** Report the whole curve. |

Note on 3/3b: I have **no verified reference number** for feature-only logistic regression
on Cora link prediction. I did not find one. Measure it; do not guess. The one anchor we
do have is that removing features costs GAE 3.9 AP points (92.0 → 88.1), which bounds
neither direction of the feature-only baseline.

Every baseline must be scored by the **same `evaluate()`** on the **same split object**,
and every row of the table must carry both the balanced and the sparse column. A baseline
table where different rows used different negative sampling is worse than no table.

---

## Open questions

1. **Which masking scheme do we actually use?** The spec's §5.1 says "hide a random 15 %
   of edge entries", which reads as 15 % of all N² cells (1,099,990 cells, ~1,583
   positives). But every published Cora link-prediction number holds out 10 % of *edges*
   (527 positives) and samples matching negatives. These are different tasks. My
   recommendation: implement `mask_fraction_of_matrix` as the primary (it is the honest
   version of "predict the missing structure") and *also* emit the 527/527 edge-split
   view so the GAE column means something. Needs a decision before `masking.py` is written.
2. **Do we mask symmetrically?** If cell `(i,j)` is masked but `(j,i)` is not, the model
   reads the answer off the transpose. Almost certainly a bug. Mask upper-triangle cells
   and mirror. Not addressed anywhere in the spec.
3. **Does the diagonal count?** `A[i,i] = 0` for all i in Cora (`negative_sampling`
   excludes self-loops, and Planetoid has none after coalesce). 2,708 guaranteed-negative
   cells in the denominator inflates the sparse AUPRC denominator by 0.037 %. Negligible,
   but exclude the diagonal from both the mask and the metric so the number is exactly
   defined.
4. **Does FlexAttention's `score_mod` handle a 2-D learned gather at acceptable speed?**
   The code path supports gradients (verified); the performance of
   `spd_bias[h, spd[q_idx, kv_idx]]` inside the Triton kernel is unmeasured. This
   single benchmark decides whether full-graph Cora fits on a 24 GB card or whether we
   are subgraph-sampling for the rest of the project. **Do it in week one.**
5. **Is FlexAttention's non-multiple-of-128 NaN bug (pytorch#153799) live in the version
   we install?** N=2708 is not a multiple of 128. Pad to 2816 or verify.
6. **GAT baseline hyperparameters are UNVERIFIED.** I quoted the widely-repeated Cora GAT
   setup from memory of the community consensus, not from a fetch. Pull PyG's
   `examples/gat.py` and the Veličković paper before running it.
7. **`NeighborLoader`'s induced-subgraph argument name is UNVERIFIED.** If we go the
   sampling route, check the current signature.
8. **No published AUPRC-on-N² baseline exists for GAE.** We will be the first to report
   it, which means no external check on our number. Mitigate by also reporting the
   balanced column, where we *can* be checked against 91.2.
9. **Do we ever need `disjoint_train_ratio`?** `RandomLinkSplit` supports splitting
   training edges into message-passing vs supervision sets. For a GNN this prevents the
   model trivially reading the supervision edge off its own input. **Our architecture has
   the same problem** — the encoder sees `A_input` and the decoder predicts cells of `A`.
   If a training-supervision cell is also visible in `A_input`, the task is trivial for
   that cell. The masking scheme must make them disjoint. Verify this explicitly with
   baseline #1 (identity): if identity beats base rate, they are not disjoint.

---

## Sources fetched

All fetched 2026-08-25.

**PyTorch Geometric source (branch `master`, raw.githubusercontent.com)**

- `CHANGELOG.md` — version 2.8.0, released 2026-06-05
- `torch_geometric/transforms/random_link_split.py` — full docstring, `__init__`,
  `forward`, `_split`, `_create_label` (4 separate fetches)
- `torch_geometric/nn/models/autoencoder.py` — full file verbatim
- `torch_geometric/utils/_negative_sampling.py` — `negative_sampling` signature/docstring
- `torch_geometric/utils/_train_test_split_edges.py` — `@deprecated` decorator
- `torch_geometric/datasets/planetoid.py` — `__init__` + full docstring + STATS table
- `torch_geometric/nn/conv/gcn_conv.py` — `__init__` and `forward` signatures
- `torch_geometric/nn/conv/gat_conv.py` — `__init__` signature + Args docstring
- `examples/autoencoder.py` — full file verbatim
- `examples/link_pred.py` — full file verbatim

**PyTorch Geometric docs / discussions**

- https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
- https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html
- https://github.com/pyg-team/pytorch_geometric/discussions/6203 (Cora 5429 vs 10556)

**Papers**

- arXiv 1611.07308 — Kipf & Welling, *Variational Graph Auto-Encoders*. Abstract page
  plus full text via ar5iv (Table 1 + protocol, verbatim)
- arXiv 1708.02002 — Lin et al., *Focal Loss for Dense Object Detection*, via ar5iv
- arXiv 2107.02658 — *On Generalization of Graph Autoencoders with Adversarial Training*,
  via ar5iv (Table 2, GAE/VGAE reproduced with PyG code)
- Davis & Goadrich 2006, *The Relationship Between Precision-Recall and ROC Curves*,
  ICML '06, DOI 10.1145/1143844.1143874 — PDF from mark.goadrich.com, pages 3–6 read
  directly (Theorem 3.2, Corollary 3.1, §4 Interpolation and AUC, Table 1, Figure 6)
- Saito & Rehmsmeier 2015, PLoS ONE 10(3):e0118432, *The Precision-Recall Plot Is More
  Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced
  Datasets* — journals.plos.org

**Reference implementations**

- https://raw.githubusercontent.com/tkipf/gae/master/gae/train.py — `pos_weight`, `norm`,
  flags
- https://raw.githubusercontent.com/tkipf/gae/master/gae/preprocessing.py —
  `mask_test_edges`, 10 %/5 % split, 1:1 negatives

**Metrics docs**

- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- https://raw.githubusercontent.com/Lightning-AI/torchmetrics/master/src/torchmetrics/classification/average_precision.py

**PyTorch / attention**

- https://docs.pytorch.org/docs/2.13/generated/torch.nn.BCEWithLogitsLoss.html
- https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://docs.pytorch.org/docs/2.13/nn.attention.flex_attention.html
- `aten/src/ATen/native/transformers/sdp_utils_cpp.h` (raw)
- `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp` (raw) — constraint arrays
- `torch/_higher_order_ops/flex_attention.py` (raw, `main`) — captured-buffer grad rules
- `torch/_inductor/kernel/flex/flex_attention.py` (raw, `main`)
- https://pytorch.org/blog/flexattention/
- https://github.com/pytorch/pytorch/issues/145460

**Model config**

- https://huggingface.co/meta-llama/Llama-3.2-1B/raw/main/config.json — **HTTP 401,
  gated**
- https://huggingface.co/unsloth/Llama-3.2-1B/raw/main/config.json — mirror, used instead

**HuggingFace transformers**

- `src/transformers/models/llama/modeling_llama.py` (raw, `main`) —
  `eager_attention_forward`

**Could not fetch**

- dl.acm.org/doi/10.1145/1143844.1143874 — HTTP 403 (obtained the paper elsewhere)
- researchgate.net copy of Davis & Goadrich — HTTP 403
- pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf — HTTP 404
- biostat.wisc.edu/~page/rocpr.pdf — returned an empty document
