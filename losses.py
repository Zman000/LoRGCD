from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_cross_entropy(student_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs.detach() * F.log_softmax(student_logits, dim=-1)).sum(dim=-1).mean()


def entropy_of_mean(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = logits.softmax(dim=-1).mean(dim=0)
    return -(p * (p + eps).log()).sum()


def instance_nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / temperature
    mask = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    pos = torch.arange(b, device=z.device)
    targets = torch.cat([pos + b, pos], dim=0)
    return F.cross_entropy(sim, targets)


def supervised_contrastive(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    labeled_mask: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if labeled_mask.sum() < 2:
        return z1.new_tensor(0.0)
    z = torch.cat([z1[labeled_mask], z2[labeled_mask]], dim=0)
    y = torch.cat([labels[labeled_mask], labels[labeled_mask]], dim=0)
    z = F.normalize(z, dim=-1)
    logits = z @ z.T / temperature
    n = logits.size(0)
    self_mask = torch.eye(n, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, -1e9)
    same = y[:, None].eq(y[None, :]) & (~self_mask)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    count = same.sum(dim=1)
    valid = count > 0
    if not valid.any():
        return z.new_tensor(0.0)
    loss = -(log_prob * same.float()).sum(dim=1)[valid] / count[valid].float()
    return loss.mean()


def riemannian_affinity_distillation(
    z_student: torch.Tensor,
    target_affinity: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Cross-entropy from a Riemannian graph distribution to student cosine affinities."""
    z = F.normalize(z_student, dim=-1)
    logits = z @ z.T / temperature
    b = z.size(0)
    self_mask = torch.eye(b, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, -1e9)

    target = target_affinity.detach().clone()
    target = target.masked_fill(self_mask, 0.0)
    row = target.sum(dim=1, keepdim=True)
    valid = row.squeeze(1) > 1e-8
    target = target / row.clamp_min(1e-8)
    logp = F.log_softmax(logits, dim=-1)
    if not valid.any():
        return z.new_tensor(0.0)
    return -(target[valid] * logp[valid]).sum(dim=-1).mean()


def symmetric_distillation(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    student_temp: float,
    teacher_temp: float,
) -> torch.Tensor:
    p1 = logits1 / student_temp
    p2 = logits2 / student_temp
    q1 = F.softmax(logits1.detach() / teacher_temp, dim=-1)
    q2 = F.softmax(logits2.detach() / teacher_temp, dim=-1)
    return 0.5 * (soft_cross_entropy(p1, q2) + soft_cross_entropy(p2, q1))
