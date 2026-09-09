"""Neural network state-value function V(s)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class NeuralValueFunction(nn.Module):
    """A minimal, inspectable PyTorch neural network estimating state-value V_phi(s).

    Maps state representation (one-hot encoded discrete state or continuous vector)
    to a scalar expected return estimate V_phi(s) in R.
    """

    def __init__(
        self,
        num_states: int,
        hidden_dims: tuple[int, ...] = (32,),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.num_states = num_states

        layers: list[nn.Module] = []
        in_dim = num_states
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def _state_to_tensor(self, state: int | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(state, (int, np.integer)):
            one_hot = torch.zeros(self.num_states, dtype=torch.float32)
            one_hot[int(state)] = 1.0
            return one_hot
        elif isinstance(state, np.ndarray):
            if state.ndim == 0 or (
                state.ndim == 1 and len(state) == 1 and state.dtype.kind in "iu"
            ):
                one_hot = torch.zeros(self.num_states, dtype=torch.float32)
                one_hot[int(state.item())] = 1.0
                return one_hot
            return torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            if state.ndim == 0 or (
                state.ndim == 1 and state.numel() == 1 and state.dtype in (torch.int32, torch.int64)
            ):
                one_hot = torch.zeros(self.num_states, dtype=torch.float32, device=state.device)
                one_hot[state.long()] = 1.0
                return one_hot
            return state.float()
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass returning scalar values of shape (batch_size,)."""
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
            return self.network(state_tensor).squeeze(-1).squeeze(0)
        return self.network(state_tensor).squeeze(-1)

    def predict(self, state: int | np.ndarray | torch.Tensor) -> float:
        """Compute V_phi(state) as a Python float without tracking gradients."""
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            val = self.forward(state_tensor)
            return float(val.item())

    def inspect_state(self, state: int | np.ndarray | torch.Tensor) -> float:
        """Alias for predict() to inspect state value."""
        return self.predict(state)
