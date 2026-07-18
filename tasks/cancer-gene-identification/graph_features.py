"""Structural graph features — computed from adjacency ONLY, no labels.

For each node these add five topology descriptors (degree, PageRank, k-core
number, local clustering coefficient, eigenvector centrality) to the 64-d
multi-omics features. They are z-scored and derived purely from the graph
structure, so they carry no label information. (See the README's validity note:
these centrality features are also what make the well-known network *study-bias*
confound in cancer-gene prediction possible — a caveat shared with the source
paper's method, not a leak of the answer.)
"""
import numpy as np
import scipy.sparse as sp


def kcore_numbers(A, deg):
    """Batagelj-Zaversnik O(E) k-core decomposition on a CSR adjacency."""
    N = A.shape[0]
    deg = deg.copy()
    md = int(deg.max()) if N > 0 else 0
    order = np.argsort(deg, kind="stable")
    pos = np.empty(N, dtype=np.int64)
    pos[order] = np.arange(N)
    order = list(order)
    bin_boundaries = [0] * (md + 2)
    for dv in deg:
        bin_boundaries[dv + 1] += 1
    for i in range(1, len(bin_boundaries)):
        bin_boundaries[i] += bin_boundaries[i - 1]
    bin_start = bin_boundaries[:]
    core = deg.copy()
    indptr = A.indptr
    indices = A.indices
    processed = np.zeros(N, dtype=bool)
    for i in range(N):
        v = order[i]
        processed[v] = True
        for k in range(indptr[v], indptr[v + 1]):
            u = indices[k]
            if core[u] > core[v] and not processed[u]:
                du = core[u]
                pu = pos[u]
                pw = bin_start[du]
                w = order[pw]
                if u != w:
                    order[pu], order[pw] = w, u
                    pos[u], pos[w] = pw, pu
                bin_start[du] += 1
                core[u] -= 1
    return core


def compute_structural_features(rows, cols, N, drop_eigenvector=False):
    """Return (N, 5) z-scored structural features from an undirected adjacency.

    rows/cols are undirected edge endpoints (both directions present).
    `drop_eigenvector` is a safety valve for very large graphs (replaces the
    power-iteration eigenvector centrality with a degree proxy)."""
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    A.data[:] = 1.0
    A = A.maximum(A.T)   # ensure symmetric
    A.setdiag(0)
    A.eliminate_zeros()

    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
    deg_safe = np.maximum(deg, 1.0)

    # --- PageRank via power iteration (sparse) ---
    d = 0.85
    inv_deg = 1.0 / deg_safe
    M = A.tocsc().multiply(inv_deg[np.newaxis, :]).tocsr()
    pr = np.full(N, 1.0 / N)
    dangling = (deg == 0)
    for _ in range(60):
        prev = pr
        dangle_sum = pr[dangling].sum()
        pr = (1 - d) / N + d * (M.dot(pr) + dangle_sum / N)
        if np.abs(pr - prev).sum() < 1e-8:
            break

    # --- eigenvector centrality via power iteration ---
    if drop_eigenvector:
        eigc = deg / deg.max()
    else:
        v = np.full(N, 1.0 / np.sqrt(N))
        for _ in range(80):
            v_new = A.dot(v)
            nrm = np.linalg.norm(v_new)
            if nrm < 1e-12:
                break
            v_new = v_new / nrm
            if np.abs(v_new - v).sum() < 1e-8:
                v = v_new
                break
            v = v_new
        eigc = np.abs(v)

    # --- local clustering coefficient: triangles / possible triangles ---
    A2 = A.dot(A)
    tri = np.asarray(A.multiply(A2).sum(axis=1)).ravel() / 2.0
    denom = deg * (deg - 1.0) / 2.0
    clustering = np.zeros(N)
    nz = denom > 0
    clustering[nz] = tri[nz] / denom[nz]

    # --- k-core number ---
    kcore = kcore_numbers(A, deg.astype(np.int64))

    feats = np.stack([deg, pr, kcore.astype(np.float64), clustering, eigc], axis=1)
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((feats - mu) / sd).astype(np.float32)
