"""Reusable local NN blocks for semi-synthetic experiments."""

from __future__ import annotations

from torch import Tensor, nn


class MLPEncoder(nn.Module):
    """Configurable MLP encoder over tabular inputs."""

    def __init__(self, *, in_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers: list[nn.Module] = []
        current_dim = int(in_dim)
        for _ in range(int(depth)):
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            current_dim = int(hidden_dim)
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x.float())


class LinearProjection(nn.Module):
    """Simple linear projection helper."""

    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_dim), int(out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.float())


class BinaryLogitHead(nn.Module):
    """Binary prediction head producing logits shaped ``(B,)``."""

    def __init__(self, *, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x.float()).squeeze(-1)
