import numpy as np
import torch

from geometry import local_riemannian_affinity, geometry_aware_mean_shift, estimate_num_classes_spectral


def test_affinity_is_stochastic():
    torch.manual_seed(0)
    z = torch.randn(32, 16)
    a, stats = local_riemannian_affinity(z, k=6, rank=3)
    assert a.shape == (32, 32)
    assert torch.all(a >= 0)
    assert torch.allclose(a.sum(1), torch.ones(32), atol=1e-5)
    assert stats.mean_condition >= 1.0


def test_mean_shift_shape_and_norm():
    torch.manual_seed(1)
    z = torch.randn(20, 8)
    a, _ = local_riemannian_affinity(z, k=5, rank=3)
    m = geometry_aware_mean_shift(z, a, steps=2)
    assert m.shape == z.shape
    assert torch.allclose(m.norm(dim=1), torch.ones(20), atol=1e-5)


def test_spectral_estimator_recovers_four_clusters():
    rng = np.random.default_rng(0)
    centers = np.eye(4, 12) * 4.0
    x = np.concatenate([c + 0.15 * rng.normal(size=(45, 12)) for c in centers], axis=0)
    k, vals = estimate_num_classes_spectral(x, min_k=2, max_k=7, n_neighbors=12, rank=3)
    assert k == 4
    assert len(vals) >= 5
    assert vals[4] - vals[3] > 0.1


def test_tangent_bundle_changes_local_graph():
    rng = np.random.default_rng(3)
    dim, n = 12, 100
    base = np.zeros(dim); base[0] = 4.0
    d1 = np.zeros(dim); d1[1] = 1.0
    d2 = np.zeros(dim); d2[1] = np.cos(0.8); d2[2] = np.sin(0.8)
    t = np.linspace(-1, 1, n)
    x1 = base + t[:, None] * d1 + 0.01 * rng.normal(size=(n, dim))
    x2 = base + t[:, None] * d2 + 0.01 * rng.normal(size=(n, dim))
    x2[:, 3] += 0.03
    x = np.concatenate([x1, x2]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    z = torch.from_numpy(x)
    a0, _ = local_riemannian_affinity(z, k=12, rank=1, use_tangent_alignment=False)
    a1, stats = local_riemannian_affinity(z, k=12, rank=1, use_tangent_alignment=True)
    assert not torch.allclose(a0, a1)
    assert stats.mean_tangent_mismatch > 0.0


def test_adaptive_rank_detects_near_one_dimensional_neighborhoods():
    torch.manual_seed(7)
    t = torch.linspace(-2.0, 2.0, 80)
    direction = torch.randn(12); direction = direction / direction.norm()
    base = torch.randn(12)
    z = base[None, :] + 0.8 * t[:, None] * direction[None, :] + 0.005 * torch.randn(80, 12)
    z = torch.nn.functional.normalize(z, dim=-1)
    _, adaptive = local_riemannian_affinity(z, k=12, rank=5, adaptive_rank=True)
    _, fixed = local_riemannian_affinity(z, k=12, rank=5, adaptive_rank=False)
    assert adaptive.mean_tangent_rank < fixed.mean_tangent_rank
    assert 1.0 <= adaptive.mean_tangent_rank <= 5.0
    assert fixed.mean_tangent_rank == 5.0
