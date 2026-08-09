"""Dependency-light Encode–Process–Decode residual graph network."""

from __future__ import annotations

import torch
from torch import nn


def mlp(input_size: int, hidden_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, output_size))


class MessageBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.edge_model = mlp(hidden_size * 3, hidden_size, hidden_size)
        self.node_model = mlp(hidden_size * 2, hidden_size, hidden_size)
        self.edge_norm = nn.LayerNorm(hidden_size)
        self.node_norm = nn.LayerNorm(hidden_size)

    def forward(self, node: torch.Tensor, edge: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.shape[1] == 0:
            return node, edge
        sender, receiver = edge_index
        message = self.edge_model(torch.cat((node[sender], node[receiver], edge), dim=-1))
        edge = self.edge_norm(edge + message)
        aggregate = torch.zeros_like(node)
        # LayerNorm may promote its output to float32 under CUDA autocast.
        aggregate.index_add_(0, receiver, edge.to(aggregate.dtype))
        degree = torch.bincount(receiver, minlength=node.shape[0]).clamp_min(1).to(node.dtype).unsqueeze(1)
        aggregate = aggregate / degree
        update = self.node_model(torch.cat((node, aggregate), dim=-1))
        return self.node_norm(node + update), edge


class ResidualGNS(nn.Module):
    def __init__(self, node_size: int = 33, edge_size: int = 8, hidden_size: int = 64, blocks: int = 3):
        super().__init__()
        self.node_encoder = mlp(node_size, hidden_size, hidden_size)
        self.edge_encoder = mlp(edge_size, hidden_size, hidden_size)
        self.processor = nn.ModuleList(MessageBlock(hidden_size) for _ in range(blocks))
        self.decoder = mlp(hidden_size, hidden_size, 3)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node = self.node_encoder(node_features)
        edge = self.edge_encoder(edge_features)
        for block in self.processor:
            node, edge = block(node, edge, edge_index)
        return self.decoder(node)
