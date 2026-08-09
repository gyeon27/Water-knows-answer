"""Differentiable physics-informed losses for residual water graphs."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _graph_sum(value: torch.Tensor, graph_index: torch.Tensor, graphs: int) -> torch.Tensor:
    output = torch.zeros((graphs,) + value.shape[1:], dtype=value.dtype, device=value.device)
    output.index_add_(0, graph_index, value)
    return output


def physics_losses(
    predicted_delta_v: torch.Tensor,
    target_delta_v: torch.Tensor,
    batch: dict[str, torch.Tensor],
    dt: float = 1.0 / 30.0,
    radius: float = 0.32,
) -> dict[str, torch.Tensor]:
    """Return supervised and four physical losses in float32."""
    node = batch["node_features"].float()
    predicted_delta_v = predicted_delta_v.float()
    target_delta_v = target_delta_v.float()
    current_v = node[:, 15:18]
    normal = node[:, 19:22]
    clearance = node[:, 18]
    gravity = node[:, 30:33] * 9.81
    base_v = (current_v + gravity * dt) * torch.exp(torch.tensor(-0.08 * dt, device=node.device))
    predicted_v = base_v + predicted_delta_v
    teacher_v = base_v + target_delta_v
    predicted_x = batch["positions"].float() + predicted_v * dt
    teacher_x = batch["positions"].float() + teacher_v * dt
    mask = batch["splash_mask"]
    supervised = F.mse_loss(predicted_delta_v[mask], target_delta_v[mask])

    predicted_clearance = clearance + dt * torch.sum(predicted_v * normal, dim=1)
    penetration = torch.mean(torch.relu(-predicted_clearance) ** 2)

    graph_index = batch["graph_index"]
    graphs = int(graph_index.max().item()) + 1
    mass = batch["particle_mass"].float().unsqueeze(1)
    predicted_momentum = _graph_sum(mass * predicted_v, graph_index, graphs)
    teacher_momentum = _graph_sum(mass * teacher_v, graph_index, graphs)
    mass_total = _graph_sum(mass, graph_index, graphs).clamp_min(1e-6)
    momentum = torch.mean(((predicted_momentum - teacher_momentum) / mass_total) ** 2)

    edge_index = batch["edge_index"]
    if edge_index.shape[1]:
        sender, receiver = edge_index
        predicted_distance = torch.linalg.vector_norm(predicted_x[sender] - predicted_x[receiver], dim=1)
        teacher_distance = torch.linalg.vector_norm(teacher_x[sender] - teacher_x[receiver], dim=1)
        predicted_kernel = torch.relu(1.0 - predicted_distance / radius) ** 3
        teacher_kernel = torch.relu(1.0 - teacher_distance / radius) ** 3
        predicted_density = torch.zeros(predicted_x.shape[0], device=node.device).index_add_(0, receiver, predicted_kernel * mass[sender, 0])
        teacher_density = torch.zeros(teacher_x.shape[0], device=node.device).index_add_(0, receiver, teacher_kernel * mass[sender, 0])
        density = F.mse_loss(predicted_density, teacher_density)
    else:
        density = predicted_delta_v.sum() * 0.0

    predicted_energy = mass[:, 0] * (0.5 * torch.sum(predicted_v**2, dim=1) + 9.81 * predicted_x[:, 1])
    teacher_energy = mass[:, 0] * (0.5 * torch.sum(teacher_v**2, dim=1) + 9.81 * teacher_x[:, 1])
    pe = _graph_sum(predicted_energy.unsqueeze(1), graph_index, graphs)[:, 0]
    te = _graph_sum(teacher_energy.unsqueeze(1), graph_index, graphs)[:, 0]
    energy = torch.mean((torch.relu(pe - te * 1.02) / mass_total[:, 0]) ** 2)
    return {"supervised": supervised, "penetration": penetration, "momentum": momentum, "density": density, "energy": energy}
