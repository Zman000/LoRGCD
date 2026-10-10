from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


@dataclass
class LocalGeometryStats:
    mean_condition: float
    mean_anisotropy: float
    mean_neighbor_cosine: float
    mean_tangent_mismatch: float = 0.0
    mean_tangent_rank: float = 0.0


def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    return x @ x.T


@torch.no_grad()
def local_riemannian_affinity(
    z: torch.Tensor,
    k: int = 16,
    rank: int = 8,
    tau_geo: float = 1.0,
    shrinkage: float = 0.10,
    eps: float = 1e-4,
    labels: Optional[torch.Tensor] = None,
    labeled_mask: Optional[torch.Tensor] = None,
    supervised_anchor_strength: float = 1.0,
    use_local_metric: bool = True,
    use_tangent_alignment: bool = True,
    adaptive_rank: bool = True,
) -> Tuple[torch.Tensor, LocalGeometryStats]:
    """Build a row-stochastic anisotropic affinity graph for one mini-batch.

    The local metric at sample i is estimated from the top-r right singular vectors of
    its k-neighbour offset matrix. Tangential directions are scaled by inverse local
    variance; normal directions use a smaller variance floor, which makes travel across
    a locally thin direction expensive while allowing motion along high-variance
    nuisance/tangent directions.

    Args:
        z: [B, D] L2-normalized (or normalizable) embeddings.
        labels: ground-truth labels only for labeled points; ignored elsewhere.
        labeled_mask: boolean [B]. When supplied, labeled must-link/cannot-link anchors
            overwrite graph edges among labeled points.
    """
    z = F.normalize(z.detach().float(), dim=-1)
    b, d = z.shape
    if b < 3:
        eye = torch.eye(b, device=z.device)
        return eye, LocalGeometryStats(1.0, 0.0, 1.0)

    k = min(max(2, k), b - 1)
    rank = min(max(1, rank), k, d)

    sim = z @ z.T
    sim.fill_diagonal_(-2.0)
    nn_idx = sim.topk(k=k, dim=1, largest=True).indices  # [B, k]

    if not use_local_metric:
        # Euclidean/spherical local graph ablation.
        dist2 = 2.0 - 2.0 * (z @ z.T).clamp(-1.0, 1.0)
        mask = torch.zeros_like(dist2, dtype=torch.bool)
        mask.scatter_(1, nn_idx, True)
        mask = mask | mask.T
        aff = torch.exp(-dist2 / max(tau_geo, eps)) * mask.float()
        aff.fill_diagonal_(0.0)
        aff = _inject_supervision(aff, labels, labeled_mask, supervised_anchor_strength)
        aff = _row_normalize_with_self_fallback(aff)
        mean_cos = float(sim.gather(1, nn_idx).mean().item())
        return aff, LocalGeometryStats(1.0, 0.0, mean_cos)

    neigh = z[nn_idx]  # [B, k, D]
    offsets = neigh - z[:, None, :]
    # Remove local translation and scale. SVD is detached by design: geometry acts as
    # a slowly changing teacher, avoiding second-order gradients through eigensystems.
    offsets = offsets - offsets.mean(dim=1, keepdim=True)

    # Batched thin SVD: offsets = U S Vh. Right singular vectors span local tangent.
    _, s, vh = torch.linalg.svd(offsets, full_matrices=False)
    basis = vh[:, :rank, :]  # [B, r_max, D]
    lam = (s[:, :rank] ** 2) / max(k - 1, 1) + eps  # local tangent variances

    # Parameter-light local intrinsic dimension. The participation ratio
    #   r_eff = (sum lambda)^2 / sum lambda^2
    # is basis-invariant and lies in [1, r_max]. We round it to obtain a per-sample
    # tangent rank, avoiding one fixed intrinsic dimension for every category/location.
    if adaptive_rank and rank > 1:
        participation = (lam.sum(dim=1) ** 2) / lam.square().sum(dim=1).clamp_min(eps)
        local_rank = participation.round().long().clamp(1, rank)
    else:
        local_rank = torch.full((b,), rank, device=z.device, dtype=torch.long)
    rank_mask = torch.arange(rank, device=z.device)[None, :] < local_rank[:, None]
    rank_mask_f = rank_mask.float()

    # Robust normal variance floor based only on active local tangent dimensions.
    mean_lam = (lam * rank_mask_f).sum(dim=1, keepdim=True) / local_rank[:, None].float()
    normal_var = (shrinkage * mean_lam).clamp_min(eps)  # [B,1]

    # Pairwise deltas from i to j.
    delta = z[None, :, :] - z[:, None, :]  # [B,B,D]
    norm2 = (delta ** 2).sum(dim=-1)  # [B,B]
    proj = torch.einsum("ijd,ird->ijr", delta, basis)  # projection onto T_i
    proj2 = proj ** 2

    inv_normal = 1.0 / normal_var  # [B,1]
    # d_i^2 = ||delta_perp||^2/sigma_perp^2 + sum_r proj_r^2/lambda_r
    active_proj2 = proj2 * rank_mask_f[:, None, :]
    tangent_norm2 = active_proj2.sum(dim=-1)
    d_i = (norm2 - tangent_norm2).clamp_min(0.0) * inv_normal
    d_i = d_i + (active_proj2 / lam[:, None, :]).sum(dim=-1)
    dist2 = 0.5 * (d_i + d_i.T)

    # Scale invariance: normalize the local-Mahalanobis energy by its robust local scale.
    edge_d = dist2.gather(1, nn_idx)
    positive = edge_d[edge_d > 0]
    scale = positive.median() if positive.numel() else torch.tensor(1.0, device=z.device)
    scale = scale.clamp_min(eps)
    dist2 = dist2 / scale

    # Tangent-bundle consistency. For orthonormal U_i,U_j,
    #   d_T(i,j)=r-||U_i U_j^T||_F^2 = sum_l sin^2(theta_l),
    # the squared chordal distance on the Grassmannian. It is zero for aligned tangent
    # spaces and remains positive when two manifolds intersect but have different tangents.
    tangent_mismatch = torch.zeros_like(dist2)
    if use_tangent_alignment and rank > 0:
        basis_eff = basis * rank_mask_f[:, :, None]
        cross = torch.einsum("ird,jsd->ijrs", basis_eff, basis_eff)
        # Extended projection/Grassmann chordal distance supports unequal local dimensions:
        # 0.5 ||P_i-P_j||_F^2 = 0.5(r_i+r_j) - ||U_i U_j^T||_F^2.
        tangent_mismatch = (
            0.5 * (local_rank[:, None].float() + local_rank[None, :].float())
            - cross.square().sum(dim=(-1, -2))
        ).clamp_min(0.0)
        tangent_mismatch.fill_diagonal_(0.0)
        edge_t = tangent_mismatch.gather(1, nn_idx)
        positive_t = edge_t[edge_t > eps]
        # Median normalization removes a free mixing coefficient between positional and
        # tangent terms. If all local tangents are effectively identical, the term vanishes.
        if positive_t.numel():
            tangent_scale = positive_t.median().clamp_min(eps)
            dist2 = dist2 + tangent_mismatch / tangent_scale

    # Re-select graph neighbors under the learned tangent-bundle metric. The cosine kNN above
    # is only used to estimate each tangent space; discovery edges themselves are
    # allowed to change under d_LR, which is essential for testing the geometric claim.
    graph_dist = dist2.clone()
    graph_dist.fill_diagonal_(float("inf"))
    geo_idx = graph_dist.topk(k=k, dim=1, largest=False).indices
    mask = torch.zeros_like(dist2, dtype=torch.bool)
    mask.scatter_(1, geo_idx, True)
    mask = mask | mask.T
    aff = torch.exp(-dist2 / max(tau_geo, eps)) * mask.float()
    aff.fill_diagonal_(0.0)

    aff = _inject_supervision(aff, labels, labeled_mask, supervised_anchor_strength)
    aff = _row_normalize_with_self_fallback(aff)

    active_min = lam.masked_fill(~rank_mask, float("inf")).min(dim=1).values
    active_max = lam.masked_fill(~rank_mask, 0.0).max(dim=1).values.clamp_min(eps)
    cond = (active_max / active_min.clamp_min(eps)).mean()
    anis = (1.0 - active_min / active_max).mean()
    mean_cos = sim.gather(1, nn_idx).mean()
    mean_tmis = tangent_mismatch.gather(1, nn_idx).mean() if use_tangent_alignment else z.new_tensor(0.0)
    stats = LocalGeometryStats(
        float(cond.item()), float(anis.item()), float(mean_cos.item()), float(mean_tmis.item()),
        float(local_rank.float().mean().item()),
    )
    return aff, stats


def _inject_supervision(
    aff: torch.Tensor,
    labels: Optional[torch.Tensor],
    labeled_mask: Optional[torch.Tensor],
    strength: float,
) -> torch.Tensor:
    if labels is None or labeled_mask is None or labeled_mask.sum() < 2:
        return aff
    labels = labels.to(aff.device)
    lm = labeled_mask.to(aff.device).bool()
    both = lm[:, None] & lm[None, :]
    same = labels[:, None].eq(labels[None, :]) & both
    diff = (~labels[:, None].eq(labels[None, :])) & both
    same.fill_diagonal_(False)
    # Labeled must-links are high-confidence manifold edges; labeled cannot-links are cut.
    if same.any():
        base = aff[aff > 0].max() if (aff > 0).any() else torch.tensor(1.0, device=aff.device)
        aff = torch.where(same, torch.maximum(aff, strength * base), aff)
    aff = torch.where(diff, torch.zeros_like(aff), aff)
    return aff


def _row_normalize_with_self_fallback(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    row = a.sum(dim=1, keepdim=True)
    bad = row.squeeze(1) <= eps
    if bad.any():
        a = a.clone()
        a[bad, bad] = 1.0
        row = a.sum(dim=1, keepdim=True)
    return a / row.clamp_min(eps)


def geometry_aware_mean_shift(z: torch.Tensor, affinity: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """One/few steps of anisotropic graph diffusion; not an exact geodesic solver."""
    out = F.normalize(z, dim=-1)
    for _ in range(max(1, steps)):
        out = F.normalize(affinity @ out, dim=-1)
    return out


# Backward-compatible alias for early experiments. New code should use the precise name.
geodesic_mean_shift = geometry_aware_mean_shift


@torch.no_grad()
def local_graph_diagnostics(
    affinity: torch.Tensor,
    labels: torch.Tensor,
    labeled_mask: Optional[torch.Tensor] = None,
) -> dict:
    """Evaluation-only diagnostics. Uses ground truth only to measure graph quality."""
    b = affinity.shape[0]
    if b <= 1:
        return {"neighbor_purity": 1.0, "cross_class_mass": 0.0}
    same = labels[:, None].eq(labels[None, :]).to(affinity.device)
    purity = (affinity * same.float()).sum(dim=1).mean().item()
    cross = (affinity * (~same).float()).sum(dim=1).mean().item()
    out = {"neighbor_purity": float(purity), "cross_class_mass": float(cross)}
    if labeled_mask is not None and labeled_mask.any():
        lm = labeled_mask.bool().to(affinity.device)
        out["labeled_neighbor_purity"] = float(
            (affinity[lm] * same[lm].float()).sum(dim=1).mean().item()
        )
    return out


def _numpy_local_edge_weights(
    x: np.ndarray,
    n_neighbors: int = 20,
    rank: int = 8,
    shrinkage: float = 0.10,
    candidate_factor: int = 4,
    use_tangent_alignment: bool = True,
    adaptive_rank: bool = True,
    eps: float = 1e-6,
):
    """Sparse CPU tangent-bundle graph for unknown-K estimation.

    Cosine neighbors are *candidates* used to estimate each local tangent and to avoid
    an O(N^2) all-pairs graph. Final graph edges are re-selected under the same local
    standardized displacement + tangent-alignment energy used during mini-batch training.
    """
    x = x.astype(np.float64, copy=False)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + eps
    n, d = x.shape
    k = min(max(2, n_neighbors), n - 1)
    cand_k = min(n - 1, max(k, int(candidate_factor) * k))
    nbrs = NearestNeighbors(n_neighbors=cand_k + 1, metric="cosine").fit(x)
    _, all_inds = nbrs.kneighbors(x)
    candidates = all_inds[:, 1:]
    tangent_inds = candidates[:, :k]

    bases, vals, normal, ranks = [], [], [], []
    r = min(rank, k, d)
    for i in range(n):
        off = x[tangent_inds[i]] - x[i]
        off -= off.mean(axis=0, keepdims=True)
        _, s, vh = np.linalg.svd(off, full_matrices=False)
        lam = (s[:r] ** 2) / max(k - 1, 1) + eps
        if adaptive_rank and r > 1:
            participation = float((lam.sum() ** 2) / max(float(np.sum(lam * lam)), eps))
            ri = int(np.clip(np.rint(participation), 1, r))
        else:
            ri = r
        bases.append(vh[:r])
        vals.append(lam)
        ranks.append(ri)
        normal.append(max(shrinkage * float(lam[:ri].mean()), eps))

    # First pass: collect dimensionless scales on local cosine candidates.
    raw_metric = []
    raw_tangent = []
    pair_cache = []
    for i in range(n):
        for j0 in candidates[i]:
            j = int(j0)
            delta = x[j] - x[i]
            energies = []
            for a in (i, j):
                ra = ranks[a]
                p = bases[a][:ra] @ delta
                tang = float(np.dot(p * p, 1.0 / vals[a][:ra]))
                perp = max(float(np.dot(delta, delta) - np.dot(p, p)), 0.0) / normal[a]
                energies.append(tang + perp)
            dm = 0.5 * (energies[0] + energies[1])
            if use_tangent_alignment:
                cross = bases[i][:ranks[i]] @ bases[j][:ranks[j]].T
                dt = max(float(0.5 * (ranks[i] + ranks[j]) - np.sum(cross * cross)), 0.0)
            else:
                dt = 0.0
            pair_cache.append((i, j, dm, dt))
            if dm > eps:
                raw_metric.append(dm)
            if dt > eps:
                raw_tangent.append(dt)

    metric_scale = float(np.median(raw_metric)) if raw_metric else 1.0
    tangent_scale = float(np.median(raw_tangent)) if raw_tangent else 1.0
    metric_scale = max(metric_scale, eps)
    tangent_scale = max(tangent_scale, eps)

    # Re-select k graph neighbors per row under the tangent-bundle energy.
    per_row = [[] for _ in range(n)]
    for i, j, dm, dt in pair_cache:
        energy = dm / metric_scale
        if use_tangent_alignment and raw_tangent:
            energy += dt / tangent_scale
        per_row[i].append((energy, j))

    rows, cols, energies = [], [], []
    for i in range(n):
        chosen = sorted(per_row[i], key=lambda t: t[0])[:k]
        for energy, j in chosen:
            rows.append(i)
            cols.append(j)
            energies.append(energy)
    energies = np.asarray(energies, dtype=np.float64)
    weights = np.exp(-energies)
    return np.asarray(rows), np.asarray(cols), weights


def estimate_num_classes_spectral(
    features: np.ndarray,
    min_k: int,
    max_k: int,
    n_neighbors: int = 20,
    rank: int = 8,
    shrinkage: float = 0.10,
    candidate_factor: int = 4,
    use_tangent_alignment: bool = True,
    adaptive_rank: bool = True,
) -> Tuple[int, np.ndarray]:
    """Estimate total class count using an eigengap of the local-geometry graph.

    Returns (estimated_K, smallest_normalized_laplacian_eigenvalues).
    The search is restricted to [min_k, max_k]. Use number of known classes as min_k.
    """
    n = len(features)
    if n <= 3:
        return max(1, min_k), np.array([0.0])
    max_k = int(min(max_k, n - 1))
    min_k = int(max(1, min(min_k, max_k)))

    rows, cols, weights = _numpy_local_edge_weights(
        features, n_neighbors=n_neighbors, rank=rank, shrinkage=shrinkage,
        candidate_factor=candidate_factor, use_tangent_alignment=use_tangent_alignment,
        adaptive_rank=adaptive_rank
    )
    w = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    w = 0.5 * (w + w.T)
    deg = np.asarray(w.sum(axis=1)).ravel()
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    dmat = sparse.diags(inv_sqrt)
    lap = sparse.eye(n, format="csr") - dmat @ w @ dmat

    nev = min(max_k + 2, n - 1)
    vals = eigsh(lap, k=nev, which="SM", return_eigenvectors=False)
    vals = np.sort(np.real(vals))
    # eigengap after the K-th near-zero/low eigenvalue: gap[K] = λ_{K+1}-λ_K.
    gaps = vals[1:] - vals[:-1]
    lo = min_k - 1
    hi = min(max_k - 1, len(gaps) - 1)
    if hi < lo:
        return min_k, vals
    k_hat = int(lo + np.argmax(gaps[lo : hi + 1]) + 1)
    return k_hat, vals

@torch.no_grad()
def global_mahalanobis_affinity(
    z: torch.Tensor,
    k: int = 16,
    tau_geo: float = 1.0,
    covariance_shrinkage: float = 0.10,
    eps: float = 1e-4,
    labels: Optional[torch.Tensor] = None,
    labeled_mask: Optional[torch.Tensor] = None,
    supervised_anchor_strength: float = 1.0,
) -> Tuple[torch.Tensor, LocalGeometryStats]:
    """Control ablation: one global covariance metric for the entire mini-batch."""
    z = F.normalize(z.detach().float(), dim=-1)
    b, d = z.shape
    if b < 3:
        eye = torch.eye(b, device=z.device)
        return eye, LocalGeometryStats(1.0, 0.0, 1.0)
    k = min(max(2, k), b - 1)
    xc = z - z.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / max(1, b - 1)
    iso = torch.trace(cov) / max(1, d)
    cov = (1.0 - covariance_shrinkage) * cov + covariance_shrinkage * iso * torch.eye(d, device=z.device)
    vals, vecs = torch.linalg.eigh(cov)
    vals = vals.clamp_min(eps)
    inv = (vecs * (1.0 / vals)[None, :]) @ vecs.T
    delta = z[None, :, :] - z[:, None, :]
    dist2 = torch.einsum("ijd,de,ije->ij", delta, inv, delta)
    positive = dist2[dist2 > 0]
    scale = positive.median() if positive.numel() else torch.tensor(1.0, device=z.device)
    dist2 = dist2 / scale.clamp_min(eps)
    gd = dist2.clone(); gd.fill_diagonal_(float("inf"))
    idx = gd.topk(k=k, dim=1, largest=False).indices
    mask = torch.zeros_like(dist2, dtype=torch.bool); mask.scatter_(1, idx, True); mask = mask | mask.T
    aff = torch.exp(-dist2 / max(tau_geo, eps)) * mask.float(); aff.fill_diagonal_(0.0)
    aff = _inject_supervision(aff, labels, labeled_mask, supervised_anchor_strength)
    aff = _row_normalize_with_self_fallback(aff)
    cond = float((vals.max() / vals.min()).item())
    anis = float((1.0 - vals.min() / vals.max()).item())
    sim = z @ z.T; sim.fill_diagonal_(-2.0)
    nn_idx = sim.topk(k=k, dim=1).indices
    mean_cos = float(sim.gather(1, nn_idx).mean().item())
    return aff, LocalGeometryStats(cond, anis, mean_cos)
