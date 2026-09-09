"""Proximal Policy Optimization (PPO) for autoregressive language models."""

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
class LMPPOResult:
    """Diagnostic training records for language model PPO."""

    mean_rewards: list[float] = field(default_factory=list)
    exact_match_rates: list[float] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    clip_fractions: list[float] = field(default_factory=list)
    approx_kls: list[float] = field(default_factory=list)
    ref_kls: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)


def train_lm_ppo(
    model: TinyTransformer,
    ref_model: TinyTransformer | None,
    prompts: list[str],
    tokenizer: CharTokenizer,
    reward_fn: LanguageRewardFunction,
    num_iterations: int = 50,
    rollout_batch_size: int = 16,
    epochs_per_rollout: int = 4,
    minibatch_size: int = 8,
    clip_eps: float = 0.2,
    lr: float = 1e-3,
    ref_kl_coef: float = 0.05,
    entropy_coef: float = 0.01,
    temperature: float = 1.0,
    seed: int | None = None,
) -> LMPPOResult:
    """Train language model with PPO over token-level actions and reference KL regularization."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    result = LMPPOResult()

    for _it in range(num_iterations):
        # 1. Collect Rollouts with current policy
        sampled_prompts = [
            prompts[i] for i in np.random.choice(len(prompts), size=rollout_batch_size)
        ]
        buffer = LMRolloutBuffer.collect_rollouts(
            model=model,
            prompts=sampled_prompts,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            temperature=temperature,
        )

        rewards = [s.reward for s in buffer.samples]
        exact_matches = [1.0 if s.is_correct else 0.0 for s in buffer.samples]

        # 2. PPO Optimization Epochs
        model.train()
        epoch_losses = []
        epoch_clips = []
        epoch_approx_kls = []
        epoch_ref_kls = []
        epoch_entropies = []

        for _epoch in range(epochs_per_rollout):
            for batch in buffer.get_minibatch_generator(batch_size=minibatch_size, shuffle=True):
                inputs = batch["input_ids"]
                targets = batch["target_ids"]
                masks = batch["action_masks"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]

                logits = model(inputs)  # (B, T, V)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(targets)  # (B, T)
                entropy = dist.entropy()

                # Compute PPO ratio: exp(log_p - old_log_p)
                log_ratio = log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * advantages
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                surr2 = clipped_ratio * advantages

                mask_sum = torch.clamp(masks.sum(), min=1.0)
                policy_loss = -(torch.min(surr1, surr2) * masks).sum() / mask_sum

                # Reference Model KL penalty
                ref_kl_val = 0.0
                if ref_model is not None:
                    with torch.no_grad():
                        ref_logits = ref_model(inputs)
                        ref_dist = Categorical(logits=ref_logits)
                        ref_log_probs = ref_dist.log_prob(targets)

                    # Sampled KL from reference model: log pi_theta - log pi_ref
                    token_ref_kl = (log_probs - ref_log_probs) * masks
                    ref_kl_loss = ref_kl_coef * (token_ref_kl.sum() / mask_sum)
                    ref_kl_val = float((token_ref_kl.sum() / mask_sum).item())
                else:
                    ref_kl_loss = torch.tensor(0.0)

                # Entropy Bonus
                entropy_loss = -entropy_coef * (entropy * masks).sum() / mask_sum

                total_loss = policy_loss + ref_kl_loss + entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                with torch.no_grad():
                    clipped_steps = (torch.abs(ratio - 1.0) > clip_eps).float() * masks
                    clip_fraction = (clipped_steps.sum() / mask_sum).item()
                    approx_kl = (((old_log_probs - log_probs) * masks).sum() / mask_sum).item()

                epoch_losses.append(policy_loss.item())
                epoch_clips.append(clip_fraction)
                epoch_approx_kls.append(approx_kl)
                epoch_ref_kls.append(ref_kl_val)
                epoch_entropies.append(((entropy * masks).sum() / mask_sum).item())

        result.mean_rewards.append(float(np.mean(rewards)))
        result.exact_match_rates.append(float(np.mean(exact_matches)))
        result.policy_losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        result.clip_fractions.append(float(np.mean(epoch_clips)) if epoch_clips else 0.0)
        result.approx_kls.append(float(np.mean(epoch_approx_kls)) if epoch_approx_kls else 0.0)
        result.ref_kls.append(float(np.mean(epoch_ref_kls)) if epoch_ref_kls else 0.0)
        result.entropies.append(float(np.mean(epoch_entropies)) if epoch_entropies else 0.0)

    return result
