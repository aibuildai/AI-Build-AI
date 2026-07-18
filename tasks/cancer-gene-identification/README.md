# Cancer Gene Identification on Biological Networks

AI-generated solution for the NatureBench [1] task *Cancer Gene Identification on
Biological Networks*, produced end-to-end by AIBuildAI. NatureBench **distilled
this task** from the *Nature Biomedical Engineering* study of Su et al. [2]:
the task's networks, features, and splits come from that paper, and its **TREE**
model is the published SOTA we are scored against. This solution's own method is
*not* TREE and does not reuse TREE's architecture — it is an independent
multi-task Chebyshev GNN (MTGCN-style; see *Final model* below). Given a
biological interaction network, 64-d multi-omics node features, and labeled
train/validation nodes, the method must prioritize **cancer-associated genes**
among the held-out test nodes, across eight networks. The headline metric is
AUPRC.

## Result

NatureBench reports the mean per-network improvement relative to the source
paper's published SOTA as a normalized gap `g`. Across the eight networks:

| | Mean AUPRC | `g` vs SOTA | Networks won |
|---|---|---|---|
| **AIBuildAI** | **0.774** | **+0.103** (surpass-SOTA) | **6 / 8** |
| Published SOTA — TREE [2] | ~0.716 | 0.000 (reference) | — |

Per-network AUPRC versus the published SOTA — above it on six of the eight
networks; behind on `cpdb` and `iref_v15`:

| Network | Our AUPRC | TREE SOTA | Δ (g) |
|---|---|---|---|
| cpdb | 0.779 | 0.791 | −0.015 |
| stringdb | 0.793 | 0.765 | +0.038 |
| pcnet | 0.768 | 0.672 | +0.143 |
| iref_v15 | 0.754 | 0.815 | −0.075 |
| iref_v9 | 0.694 | 0.681 | +0.018 |
| multinet | 0.777 | 0.686 | +0.132 |
| mtg | 0.773 | 0.540 | +0.431 |
| ltg | 0.843 | 0.731 | +0.153 |
| | | **mean g** | **+0.103** |

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

- **Input** per network: an `N×N` adjacency matrix, an `N×64` multi-omics feature
  matrix (mutation frequency, methylation, gene expression, copy-number across 16
  cancer types), and boolean masks + labels for train / validation nodes. Test
  nodes are given as a mask; their labels are hidden.
- **Output** per network: a probability in `[0,1]` for every test node.
- **Metric**: AUPRC (primary) and AUROC per network, aggregated as the mean
  improvement `g` relative to the paper's per-network SOTA.
- **Instances**: six PPI networks (cpdb, stringdb, pcnet, iref_v15, iref_v9,
  multinet) and two heterogeneous networks (mtg, ltg); N ranges 12k–26k nodes,
  cancer-gene positives 14–29%.

## Final model

A **multi-task Chebyshev-spectral graph neural network (MTGCN)** trained independently per
network:

- **Encoder.** Two `ChebConv` layers (K=2 Chebyshev order, 300→100) over the node
  features, with DropEdge and dropout.
- **Features.** The 64-d multi-omics vector is augmented with **five structural
  descriptors computed from the adjacency alone** — degree, PageRank, k-core
  number (an O(E) Batagelj–Zaversnik decomposition), local clustering
  coefficient, and eigenvector centrality — each z-scored.
- **Multi-task heads.** A `ChebConv` cancer-gene classifier plus a self-supervised
  **link-prediction** head (inner product of embeddings on real vs random edges)
  that share the encoder, combined with **learned homoscedastic uncertainty
  weights** (the Kendall et al. loss).
- **Training.** Adam with class-imbalance `pos_weight`, gradient clipping, and an
  **EMA of the weights**; an adaptive per-network step budget keeps the full
  eight-network run inside its wall-clock cap. Total run time ≈ **15 minutes** on
  one A100.

## Files

| File | What it is |
|---|---|
| `run.py` | End-to-end pipeline — per network, builds features, trains the MTGCN, writes `output/<network>/predictions.npy` |
| `model.py` | The MTGCN (ChebConv encoder + classifier + link head + uncertainty weights) and edge helpers |
| `graph_features.py` | The five label-free structural features (degree, PageRank, k-core, clustering, eigenvector centrality) |
| `config.py` | Every hyperparameter, in one place |
| `predictions/` | The produced `predictions.npy` for all eight networks + `score.json` — the exact submission that scored mean AUPRC 0.774 (`g` +0.103) |

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

[1] Wang, Y. et al. NatureBench: Can Coding Agents Match the Published SOTA of
Nature-Family Papers? arXiv:2606.24530 (2026).

[2] Su, X., Hu, P., Li, D., Zhao, B., Niu, Z., Herget, T., Yu, P. S. & Hu, L.
Interpretable identification of cancer genes across biological networks via
transformer-powered graph representation learning. *Nature Biomedical
Engineering* **9**, 371–389 (2025). DOI: 10.1038/s41551-024-01312-5 — **the source
paper this NatureBench task is distilled from**; its TREE model is the published
SOTA this solution is scored against.
