"""Algorithms module."""

from rl_reasoning_lab.algorithms.advantage import (
    compute_full_returns,
    compute_gae,
    compute_reward_to_go,
    normalize_advantages,
)
from rl_reasoning_lab.algorithms.dynamic_programming import (
    DPResult,
    compute_q_from_v,
    policy_evaluation,
    value_iteration,
)
from rl_reasoning_lab.algorithms.lm_policy_gradient import (
    LMRLResult,
    train_lm_reinforce,
)
from rl_reasoning_lab.algorithms.lm_ppo import (
    LMPPOResult,
    train_lm_ppo,
)
from rl_reasoning_lab.algorithms.policy_gradient import (
    PolicyGradientResult,
    actor_critic,
    reinforce,
)
from rl_reasoning_lab.algorithms.ppo import (
    PPOResult,
    compute_ppo_policy_loss,
    compute_ppo_value_loss,
    train_ppo,
)
from rl_reasoning_lab.algorithms.temporal_difference import (
    TDResult,
    q_learning,
    sarsa,
)

__all__ = [
    "DPResult",
    "policy_evaluation",
    "value_iteration",
    "compute_q_from_v",
    "TDResult",
    "q_learning",
    "sarsa",
    "compute_full_returns",
    "compute_reward_to_go",
    "compute_gae",
    "normalize_advantages",
    "reinforce",
    "actor_critic",
    "PolicyGradientResult",
    "train_ppo",
    "compute_ppo_policy_loss",
    "compute_ppo_value_loss",
    "PPOResult",
    "train_lm_reinforce",
    "train_lm_ppo",
    "LMRLResult",
    "LMPPOResult",
]
