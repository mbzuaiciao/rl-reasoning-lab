"""A minimal, inspectable causal autoregressive transformer in PyTorch."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with explicit upper-triangular masking."""

    def __init__(
        self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Causal mask: lower triangular matrix of ones (attend only to past and present)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
            1, 1, max_seq_len, max_seq_len
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        enable_dropout: bool = False,
    ) -> torch.Tensor:
        b, t, c = x.size()

        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask: mask positions where causal_mask == 0 with -inf
        mask = self.causal_mask[:, :, :t, :t] == 0
        scores = scores.masked_fill(mask, float("-inf"))

        # Optional key padding mask (shape B, T) -> (B, 1, 1, T)
        if attention_mask is not None:
            pad_mask = (attention_mask == 0.0).view(b, 1, 1, t)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        # In case entire row is masked out (e.g. pure padding)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        if self.dropout_p > 0.0:
            attn_weights = F.dropout(
                attn_weights, p=self.dropout_p, training=self.training or enable_dropout
            )

        out = attn_weights @ v  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.out_proj(out)
        if self.dropout_p > 0.0:
            out = F.dropout(out, p=self.dropout_p, training=self.training or enable_dropout)
        return out


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer decoder block."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.dropout_p = dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_seq_len, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        enable_dropout: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask, enable_dropout=enable_dropout)
        ffn_out = self.ffn(self.ln2(x))
        if self.dropout_p > 0.0:
            ffn_out = F.dropout(ffn_out, p=self.dropout_p, training=self.training or enable_dropout)
        x = x + ffn_out
        return x


class TinyTransformer(nn.Module):
    """A small, inspectable causal language model designed for local RL experiments.

    Architecture:
        - Token embeddings + learned positional embeddings
        - N Transformer decoder blocks with pre-LayerNorm
        - Final LayerNorm + linear language modeling head
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, max_seq_len, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights between embedding and head for sample efficiency
        self.head.weight = self.tok_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        enable_dropout: bool = False,
    ) -> torch.Tensor:
        """Compute logits for all token positions.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len).
            attention_mask: Optional FloatTensor of shape (batch_size, seq_len).
            enable_dropout: If True, activates dropout even during evaluation (for MC Dropout).

        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, vocab_size).
        """
        b, t = input_ids.size()
        if t > self.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len {self.max_seq_len}.")

        positions = torch.arange(0, t, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, enable_dropout=enable_dropout)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    def inspect_next_token(self, prefix_ids: list[int] | torch.Tensor) -> dict[str, Any]:
        """Inspect next-token logits, probabilities, and entropy given a prefix."""
        with torch.no_grad():
            if not isinstance(prefix_ids, torch.Tensor):
                input_ids = torch.tensor([prefix_ids], dtype=torch.long)
            elif prefix_ids.ndim == 1:
                input_ids = prefix_ids.unsqueeze(0)
            else:
                input_ids = prefix_ids

            logits = self.forward(input_ids)[:, -1, :]  # (1, V)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs=probs)

            return {
                "logits": logits.squeeze(0).cpu().numpy(),
                "probabilities": probs.squeeze(0).cpu().numpy(),
                "entropy": float(dist.entropy().item()),
                "argmax_token": int(torch.argmax(logits, dim=-1).item()),
            }

    def generate(
        self,
        prompt_ids: list[int] | torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        greedy: bool = False,
        eos_token_id: int | None = None,
        enable_dropout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively sample tokens conditioned on prompt_ids.

        Returns:
            full_ids: Tensor of shape (1, len(prompt) + num_generated)
            action_log_probs: Tensor of shape (num_generated,)
        """
        if not isinstance(prompt_ids, torch.Tensor):
            current_ids = torch.tensor([prompt_ids], dtype=torch.long)
        elif prompt_ids.ndim == 1:
            current_ids = prompt_ids.unsqueeze(0)
        else:
            current_ids = prompt_ids

        generated_log_probs: list[torch.Tensor] = []

        for _ in range(max_new_tokens):
            logits = self.forward(current_ids, enable_dropout=enable_dropout)[:, -1, :]  # (1, V)

            if greedy or temperature <= 1e-4:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                dist = Categorical(logits=logits)
                log_p = dist.log_prob(next_token.squeeze(-1))
            else:
                scaled_logits = logits / temperature
                dist = Categorical(logits=scaled_logits)
                next_token = dist.sample().unsqueeze(-1)
                log_p = dist.log_prob(next_token.squeeze(-1))

            generated_log_probs.append(log_p.squeeze(0))
            current_ids = torch.cat([current_ids, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        action_log_probs = (
            torch.stack(generated_log_probs) if generated_log_probs else torch.tensor([])
        )
        return current_ids, action_log_probs
