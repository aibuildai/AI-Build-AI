# Cancer Gene Identification on Biological Networks

Autonomous solution to *Cancer Gene Identification on Biological Networks*,
produced end-to-end by **AIBuildAI** — an autonomous system that builds AI
models, designing, implementing, and evaluating candidate solutions with no
human in the loop, and able to recursively self-evolve. The task is distilled
from the *Nature Biomedical Engineering* study of Su et al. [1]: its networks,
features, and splits come from that paper, and its **TREE** model is the
published state of the art we compare against. This solution's method is *not*
TREE and does not reuse TREE's architecture — it is an independent multi-task
Chebyshev-spectral GNN (MTGCN-style; see *Final model* below). Given a biological
interaction network, 64-d multi-omics node features, and labeled train/validation
nodes, it prioritizes **cancer-associated genes** among the held-out test nodes,
across eight networks. The headline metric is **AUPRC**.

📝 **Read the blog:** https://www.aibuildai.io/blog-cancer-gene-identification

## Result

Across the eight networks, AIBuildAI reaches a **mean AUPRC of 0.774**, against
**~0.716** for the published SOTA (TREE [1]), and surpasses TREE on **6 of the 8**
networks.

| | Mean AUPRC | Networks won |
|---|---|---|
| **AIBuildAI** | **0.774** | **6 / 8** |
| Published SOTA — TREE [1] | ~0.716 | — |

Per-network AUPRC versus the published SOTA (sorted by margin) — above it on six
of the eight networks, behind only on `cpdb` and `iref_v15`:

| Network | Our AUPRC | TREE (SOTA) | Δ AUPRC |
|---|---|---|---|
| mtg | 0.773 | 0.540 | +0.233 |
| ltg | 0.843 | 0.731 | +0.112 |
| pcnet | 0.768 | 0.672 | +0.096 |
| multinet | 0.777 | 0.686 | +0.091 |
| stringdb | 0.793 | 0.765 | +0.028 |
| iref_v9 | 0.694 | 0.681 | +0.013 |
| cpdb | 0.779 | 0.791 | −0.012 |
| iref_v15 | 0.754 | 0.815 | −0.061 |

## Background

Identifying which genes drive cancer is a foundation for understanding
tumorigenesis and for precision oncology. Genes do not act alone: they operate
inside networks of protein–protein and regulatory interactions, and cancer genes
tend to occupy characteristic positions in those networks while also carrying
recurrent molecular alterations. Combining **network topology** with **multi-omics
molecular profiles** is the modern approach to prioritizing cancer genes across
the full diversity of interaction databases.

## Task

Each of the eight instances is a **transductive node-classification** problem on
one biological network.

- **Input**: an `N×N` adjacency matrix, an `N×64` multi-omics feature matrix
  (mutation, methylation, expression, copy-number across 16 cancer types), and
  labeled train/validation masks; test-node labels are hidden.
- **Output**: a probability in `[0,1]` for every test node.
- **Metric**: per-network AUPRC (primary), compared against the paper's published SOTA.
- **Instances**: six PPI networks (cpdb, stringdb, pcnet, iref_v15, iref_v9,
  multinet) and two heterogeneous networks (mtg, ltg); N ≈ 12k–26k, positives 14–29%.

## Final model

A **multi-task Chebyshev-spectral GNN (MTGCN)**, trained independently per network:

- **Encoder.** Two `ChebConv` layers (K=2, 300→100), with DropEdge and dropout.
- **Features.** The 64-d multi-omics vector plus **five label-free structural
  descriptors** (degree, PageRank, k-core number, local clustering, eigenvector
  centrality), each z-scored.
- **Multi-task heads.** A `ChebConv` cancer-gene classifier and a self-supervised
  **link-prediction** head share the encoder, balanced by **learned uncertainty
  weights** (Kendall et al.).
- **Training.** Adam with class-imbalance `pos_weight`, gradient clipping, and an
  **EMA of the weights**; an adaptive per-network step budget keeps the full run
  under its time cap (≈ **15 minutes** on one A100).

## Files

| File | What it is |
|---|---|
| `run.py` | End-to-end pipeline: builds features, trains the MTGCN, writes predictions per network |
| `model.py` | The MTGCN (encoder + classifier + link head + uncertainty weights) and edge helpers |
| `graph_features.py` | The five label-free structural features |
| `config.py` | All hyperparameters |
| `predictions/` | `predictions.npy` for all eight networks + `score.json` — the exact submission (mean AUPRC 0.774) |

Note: like the two-stage genomics solution, this method **trains at inference
time** — the transductive GNN is fit fresh per network — so there is no persisted
neural checkpoint to ship.

## Reproduce inference

```bash
python run.py --data-dir /path/to/data --output ./output
```

`--data-dir` holds one `<network>/data.h5` per network (keys: `network`,
`features`, `mask_train/val/test`, `y_train`, `y_val`). Requirements: `torch`,
`torch_geometric`, `numpy`, `h5py`, `scipy`, `scikit-learn`. The full eight-network
run takes ≈ 15 minutes on one A100. The `predictions/` files are the exact scored
submission; re-running reproduces the score to within GPU/training nondeterminism.

## References

[1] Su, X., Hu, P., Li, D., Zhao, B., Niu, Z., Herget, T., Yu, P. S. & Hu, L.
Interpretable identification of cancer genes across biological networks via
transformer-powered graph representation learning. *Nature Biomedical
Engineering* **9**, 371–389 (2025). DOI: 10.1038/s41551-024-01312-5 — the source
paper this task is distilled from; its TREE model is the published SOTA this
solution is compared against.
