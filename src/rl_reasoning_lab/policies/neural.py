"""Neural network policies for discrete action spaces."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rl_reasoning_lab.policies.base import Policy


class CategoricalPolicy(nn.Module, Policy):
    """A minimal, inspectable PyTorch categorical policy for discrete state and action spaces.

    Accepts either an integer state (encoded as one-hot) or a continuous state vector.
    Computes logits via a multi-layer perceptron (MLP), defines a Categorical distribution,
    and exposes every intermediate quantity (logits, probabilities, log_probs, entropy).
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (32,),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.is_discrete_state = True

        layers: list[nn.Module] = []
        in_dim = num_states
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def _state_to_tensor(self, state: int | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert state into a float torch.Tensor.

        If state is an integer, it is one-hot encoded.
        """
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
        """Compute action logits for the given state tensor."""
        return self.network(state_tensor)

    def get_distribution(self, state: int | np.ndarray | torch.Tensor) -> Categorical:
        """Construct the Categorical action distribution for a given state."""
        state_tensor = self._state_to_tensor(state)
        logits = self.forward(state_tensor)
        return Categorical(logits=logits)

    def sample_action(self, state: int, *, rng: np.random.Generator | None = None) -> int:
        """Sample an action as an integer (conforms to the base Policy protocol)."""
        dist = self.get_distribution(state)
        action = dist.sample()
        return int(action.item())

    def sample_action_with_log_prob(
        self, state: int | np.ndarray | torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        """Sample an action and return both the integer action and its log probability tensor."""
        dist = self.get_distribution(state)
        action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)
        return int(action_tensor.item()), log_prob

    def action_probabilities(self, state: int) -> np.ndarray:
        """Return the probability vector pi(a | state) as a numpy array."""
        with torch.no_grad():
            dist = self.get_distribution(state)
            probs = dist.probs.cpu().numpy()
            return probs.astype(np.float64)

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch evaluate actions under current policy.

        Args:
            states: Tensor of state inputs, shape (B, num_states).
            actions: Tensor of integer actions, shape (B,).

        Returns:
            log_probs: Log probabilities log pi(a_t | s_t), shape (B,).
            entropy: Policy entropy H(pi(· | s_t)), shape (B,).
        """
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def inspect_state(self, state: int | np.ndarray | torch.Tensor) -> dict[str, Any]:
        """Inspect all internal policy quantities for a given state.

        Returns a dictionary of raw numpy values:
            - 'logits': raw unnormalized scores
            - 'probabilities': softmax normalized probabilities
            - 'log_probabilities': log(probabilities)
            - 'entropy': Shannon entropy -sum(p * log p)
        """
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            logits = self.forward(state_tensor)
            dist = Categorical(logits=logits)
            probs = dist.probs
            log_probs = torch.log(probs + 1e-12)
            entropy = dist.entropy()

            return {
                "logits": logits.cpu().numpy(),
                "probabilities": probs.cpu().numpy(),
                "log_probabilities": log_probs.cpu().numpy(),
                "entropy": float(entropy.item()),
            }
