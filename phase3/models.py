"""Architecture-matched PyTorch GNS used by every learned Phase 3 condition."""

from __future__ import annotations

import torch
from torch import nn


def _mlp(input_size: int, hidden: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_size, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, output_size))


class ProcessorBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = _mlp(hidden * 3, hidden, hidden)
        self.node_mlp = _mlp(hidden * 2, hidden, hidden)
        self.edge_norm = nn.LayerNorm(hidden)
        self.node_norm = nn.LayerNorm(hidden)

    def forward(self, node: torch.Tensor, edge: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.shape[1] == 0:
            return node, edge
        sender, receiver = edge_index
        edge = self.edge_norm(edge + self.edge_mlp(torch.cat((node[sender], node[receiver], edge), dim=1)))
        # Functional scatter_add exports to ONNX ScatterElements(reduction=add)
        # and correctly handles many edges sharing the same receiver. Legacy
        # index_add export can silently overwrite duplicated receiver indices.
        expanded_receiver = receiver[:, None].expand(-1, node.shape[1])
        aggregate = torch.zeros_like(node).scatter_add(
            0, expanded_receiver, edge.to(node.dtype)
        )
        node = self.node_norm(node + self.node_mlp(torch.cat((node, aggregate), dim=1)))
        return node, edge


class UnifiedGNS(nn.Module):
    def __init__(self, numeric_size: int = 27, edge_size: int = 4, hidden: int = 128, blocks: int = 10, type_embedding: int = 16, particle_types: int = 9):
        super().__init__()
        self.type_embedding = nn.Embedding(particle_types, type_embedding)
        self.node_encoder = _mlp(numeric_size + type_embedding, hidden, hidden)
        self.edge_encoder = _mlp(edge_size, hidden, hidden)
        self.processor = nn.ModuleList(ProcessorBlock(hidden) for _ in range(blocks))
        self.decoder = _mlp(hidden, hidden, 3)

    def forward(self, numeric: torch.Tensor, particle_type: torch.Tensor, edge_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        particle_type = particle_type.clamp(0, self.type_embedding.num_embeddings - 1)
        node = self.node_encoder(torch.cat((numeric, self.type_embedding(particle_type)), dim=1))
        edge = self.edge_encoder(edge_features)
        for block in self.processor:
            node, edge = block(node, edge, edge_index)
        return self.decoder(node)
