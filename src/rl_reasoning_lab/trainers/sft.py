"""Supervised Fine-Tuning (SFT) trainer for causal language models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rl_reasoning_lab.environments.language_tasks import Example
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer


@dataclass
class SFTResult:
    """Diagnostic training records for Supervised Fine-Tuning."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    token_accuracies: list[float] = field(default_factory=list)
    exact_match_accuracies: list[float] = field(default_factory=list)


def prepare_sft_batch(
    examples: list[Example],
    tokenizer: CharTokenizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare teacher-forced input, target, and loss mask tensors for a batch.

    Causal Autoregressive Shift:
        input_ids:  [BOS, x_1, x_2, ..., x_{T-1}]
        target_ids: [x_1, x_2, ..., x_T, EOS]
        loss_mask:  1.0 on completion tokens, 0.0 on prompt tokens and padding.
    """
    full_seqs: list[list[int]] = []
    masks: list[list[float]] = []

    for ex in examples:
        prompt_ids = tokenizer.encode(ex.prompt, add_bos=True, add_eos=False)
        target_ids = tokenizer.encode(ex.target, add_bos=False, add_eos=True)

        seq = prompt_ids + target_ids
        full_seqs.append(seq)

        # Mask: 0 for prompt positions, 1 for target positions
        mask = [0.0] * len(prompt_ids) + [1.0] * len(target_ids)
        masks.append(mask)

    max_len = max(len(s) for s in full_seqs)
    batch_size = len(examples)

    padded_seqs = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
    loss_masks = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for i, (seq, m) in enumerate(zip(full_seqs, masks, strict=True)):
        padded_seqs[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        loss_masks[i, : len(m)] = torch.tensor(m, dtype=torch.float32)

    # Shift for autoregressive prediction:
    # Inputs: all tokens except the last
    input_ids = padded_seqs[:, :-1].contiguous()
    # Targets: all tokens except the first
    target_ids = padded_seqs[:, 1:].contiguous()
    # Mask aligns with target_ids
    loss_mask = loss_masks[:, 1:].contiguous()

    return input_ids, target_ids, loss_mask


def evaluate_sft(
    model: TinyTransformer,
    examples: list[Example],
    tokenizer: CharTokenizer,
    batch_size: int = 32,
) -> tuple[float, float, float]:
    """Evaluate SFT model on validation examples.

    Returns:
        (val_loss, token_acc, exact_match_acc)
    """
    model.eval()
    losses = []
    correct_tokens = 0
    total_tokens = 0
    exact_matches = 0

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch_ex = examples[i : i + batch_size]
            inputs, targets, masks = prepare_sft_batch(batch_ex, tokenizer)

            logits = model(inputs)  # (B, T, V)
            # Flatten for cross-entropy
            flat_logits = logits.view(-1, model.vocab_size)
            flat_targets = targets.view(-1)
            flat_masks = masks.view(-1)

            loss_per_token = nn.functional.cross_entropy(
                flat_logits, flat_targets, reduction="none"
            )
            mask_sum = flat_masks.sum().item()
            if mask_sum > 0:
                loss = (loss_per_token * flat_masks).sum() / mask_sum
                losses.append(loss.item())

            # Token accuracy on masked targets
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == targets).float() * masks
            correct_tokens += int(correct.sum().item())
            total_tokens += int(mask_sum)

        # Exact match via autoregressive generation
        for ex in examples:
            prompt_ids = tokenizer.encode(ex.prompt, add_bos=True, add_eos=False)
            gen_ids, _ = model.generate(
                prompt_ids,
                max_new_tokens=len(ex.target) + 2,
                greedy=True,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Decode generated part
            prompt_len = len(prompt_ids)
            gen_tokens = gen_ids[0, prompt_len:].tolist()
            gen_str = tokenizer.decode(gen_tokens, skip_special=True)
            if gen_str.strip() == ex.target.strip():
                exact_matches += 1

    val_loss = float(np.mean(losses)) if losses else 0.0
    token_acc = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0
    em_acc = exact_matches / len(examples) if examples else 0.0

    return val_loss, token_acc, em_acc


def train_sft(
    model: TinyTransformer,
    train_examples: list[Example],
    val_examples: list[Example],
    tokenizer: CharTokenizer,
    num_epochs: int = 15,
    batch_size: int = 16,
    lr: float = 3e-3,
    seed: int | None = None,
) -> SFTResult:
    """Train language model with Supervised Fine-Tuning."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    result = SFTResult()

    for _epoch in range(1, num_epochs + 1):
        model.train()
        indices = np.random.permutation(len(train_examples))
        epoch_losses = []

        for start_idx in range(0, len(train_examples), batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            batch_ex = [train_examples[i] for i in batch_idx]

            inputs, targets, masks = prepare_sft_batch(batch_ex, tokenizer)

            logits = model(inputs)
            flat_logits = logits.view(-1, model.vocab_size)
            flat_targets = targets.view(-1)
            flat_masks = masks.view(-1)

            loss_per_token = nn.functional.cross_entropy(
                flat_logits, flat_targets, reduction="none"
            )
            mask_sum = flat_masks.sum()
            loss = (loss_per_token * flat_masks).sum() / torch.clamp(mask_sum, min=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        val_loss, token_acc, em_acc = evaluate_sft(model, val_examples, tokenizer)
        result.train_losses.append(float(np.mean(epoch_losses)))
        result.val_losses.append(val_loss)
        result.token_accuracies.append(token_acc)
        result.exact_match_accuracies.append(em_acc)

    return result
