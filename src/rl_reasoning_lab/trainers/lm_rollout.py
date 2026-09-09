"""Rollout generation and buffering for language model reinforcement learning."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction


@dataclass
class LMSequenceSample:
    """A single complete trajectory rollout for a language model."""

    prompt_ids: list[int]
    generated_ids: list[int]
    full_ids: list[int]
    old_log_probs: list[float]
    reward: float
    is_correct: bool
    is_malformed: bool


class LMRolloutBuffer:
    """Buffer storing autoregressively generated sequences and frozen action log probabilities."""

    def __init__(self, tokenizer: CharTokenizer) -> None:
        self.tokenizer = tokenizer
        self.samples: list[LMSequenceSample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def clear(self) -> None:
        self.samples.clear()

    def add(self, sample: LMSequenceSample) -> None:
        self.samples.append(sample)

    @classmethod
    def collect_rollouts(
        cls,
        model: TinyTransformer,
        prompts: list[str],
        tokenizer: CharTokenizer,
        reward_fn: LanguageRewardFunction,
        max_new_tokens: int = 8,
        temperature: float = 1.0,
    ) -> LMRolloutBuffer:
        """Sample autoregressive trajectories from the current policy model."""
        buffer = cls(tokenizer)
        model.eval()

        with torch.no_grad():
            for prompt_str in prompts:
                prompt_ids = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
                full_ids_tensor, log_probs_tensor = model.generate(
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    greedy=False,
                    eos_token_id=tokenizer.eos_token_id,
                )

                full_ids = full_ids_tensor[0].tolist()
                gen_ids = full_ids[len(prompt_ids) :]
                old_lps = [float(lp.item()) for lp in log_probs_tensor]

                # Decode completion
                comp_str = tokenizer.decode(gen_ids, skip_special=True)
                outcome = reward_fn(prompt_str, comp_str)

                sample = LMSequenceSample(
                    prompt_ids=prompt_ids,
                    generated_ids=gen_ids,
                    full_ids=full_ids,
                    old_log_probs=old_lps,
                    reward=outcome.reward,
                    is_correct=outcome.is_correct,
                    is_malformed=outcome.is_malformed,
                )
                buffer.add(sample)

        return buffer

    def get_minibatch_generator(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield padded, action-masked minibatches for policy gradient / PPO optimization."""
        n = len(self.samples)
        indices = torch.randperm(n) if shuffle else torch.arange(n)

        max_seq_len = max(len(s.full_ids) for s in self.samples)

        # Allocate full padded arrays
        padded_ids = torch.full((n, max_seq_len), self.tokenizer.pad_token_id, dtype=torch.long)
        action_masks = torch.zeros((n, max_seq_len - 1), dtype=torch.float32)
        old_log_probs = torch.zeros((n, max_seq_len - 1), dtype=torch.float32)
        rewards = torch.zeros(n, dtype=torch.float32)

        for i, s in enumerate(self.samples):
            seq = s.full_ids
            padded_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            rewards[i] = s.reward

            prompt_len = len(s.prompt_ids)
            gen_len = len(s.generated_ids)

            # In causal shift: token full_ids[t+1] is predicted from prefix full_ids[:t+1].
            # Actions are generated tokens at positions [prompt_len, ..., prompt_len + gen_len - 1].
            # Therefore, in the target sequence (full_ids[1:]), the action indices are:
            # target position index = (t + 1) - 1 = t.
            # When full_ids[prompt_len] is the first action token, its target position is prompt_len - 1.
            start_target_idx = prompt_len - 1
            for k in range(gen_len):
                tgt_idx = start_target_idx + k
                if tgt_idx < max_seq_len - 1:
                    action_masks[i, tgt_idx] = 1.0
                    if k < len(s.old_log_probs):
                        old_log_probs[i, tgt_idx] = s.old_log_probs[k]

        input_ids = padded_ids[:, :-1].contiguous()
        target_ids = padded_ids[:, 1:].contiguous()

        # Compute advantages: standardize sequence rewards across batch
        if len(rewards) > 1:
            r_std = torch.std(rewards, unbiased=False)
            if r_std > 1e-6:
                norm_rewards = (rewards - rewards.mean()) / (r_std + 1e-8)
            else:
                norm_rewards = rewards - rewards.mean()
        else:
            norm_rewards = torch.zeros_like(rewards)

        # Broadcast scalar sequence advantage to action tokens
        advantages = torch.zeros_like(action_masks)
        for i in range(n):
            advantages[i] = norm_rewards[i] * action_masks[i]

        for start_idx in range(0, n, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield {
                "input_ids": input_ids[batch_idx],
                "target_ids": target_ids[batch_idx],
                "action_masks": action_masks[batch_idx],
                "old_log_probs": old_log_probs[batch_idx],
                "advantages": advantages[batch_idx],
                "rewards": rewards[batch_idx],
            }
