"""REINFORCE algorithm adapted for autoregressive language models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.distributions import Categorical

from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.trainers.lm_rollout import LMRolloutBuffer


@dataclass
class LMRLResult:
    """Diagnostic training records for language model RL."""

    mean_rewards: list[float] = field(default_factory=list)
    exact_match_rates: list[float] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    kl_divergences: list[float] = field(default_factory=list)


def train_lm_reinforce(
    model: TinyTransformer,
    prompts: list[str],
    tokenizer: CharTokenizer,
    reward_fn: LanguageRewardFunction,
    num_iterations: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    temperature: float = 1.0,
    entropy_coef: float = 0.01,
    seed: int | None = None,
) -> LMRLResult:
    """Train language model using REINFORCE with masked action tokens."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    result = LMRLResult()

    for _it in range(num_iterations):
        # Sample batch of prompts
        sampled_prompts = [prompts[i] for i in np.random.choice(len(prompts), size=batch_size)]

        buffer = LMRolloutBuffer.collect_rollouts(
            model=model,
            prompts=sampled_prompts,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            temperature=temperature,
        )

        rewards = [s.reward for s in buffer.samples]
        exact_matches = [1.0 if s.is_correct else 0.0 for s in buffer.samples]

        # Optimization step
        model.train()
        total_loss = torch.tensor(0.0)
        iter_entropies = []

        for batch in buffer.get_minibatch_generator(batch_size=batch_size, shuffle=False):
            inputs = batch["input_ids"]
            targets = batch["target_ids"]
            masks = batch["action_masks"]
            advantages = batch["advantages"]

            logits = model(inputs)  # (B, T, V)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(targets)
            entropy = dist.entropy()

            # Masked policy loss: only on generated actions
            mask_sum = torch.clamp(masks.sum(), min=1.0)
            policy_loss = -(advantages * log_probs * masks).sum() / mask_sum
            entropy_loss = -entropy_coef * (entropy * masks).sum() / mask_sum

            loss = policy_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss = policy_loss.detach()
            iter_entropies.append(((entropy * masks).sum() / mask_sum).item())

        result.mean_rewards.append(float(np.mean(rewards)))
        result.exact_match_rates.append(float(np.mean(exact_matches)))
        result.policy_losses.append(float(total_loss.item()))
        result.entropies.append(float(np.mean(iter_entropies)) if iter_entropies else 0.0)

    return result
