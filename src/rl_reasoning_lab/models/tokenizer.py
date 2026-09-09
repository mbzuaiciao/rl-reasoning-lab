"""A transparent, inspectable character-level tokenizer for symbolic reasoning tasks."""

from __future__ import annotations

import torch


class CharTokenizer:
    """Character-level tokenizer with special tokens for arithmetic and reasoning tasks.

    Vocabulary:
        0: <PAD> - Padding token
        1: <BOS> - Beginning of sequence
        2: <EOS> - End of sequence
        3..12: '0'..'9' - Digits
        13: '+'
        14: '-'
        15: '*'
        16: '='
        17: ' ' (space)
        18: ';'
        19: '?'
        20..N: Letters/symbols if needed
    """

    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, extra_chars: str = "") -> None:
        import string

        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        base_chars = (
            list("0123456789+-*= ;?:><_") + list(string.ascii_uppercase) + list(extra_chars)
        )
        # Deduplicate while preserving order
        unique_chars = []
        for ch in base_chars:
            if ch not in unique_chars:
                unique_chars.append(ch)

        self.vocab = special_tokens + unique_chars
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.vocab)}

        self.pad_token_id = self.char_to_id[self.PAD_TOKEN]
        self.bos_token_id = self.char_to_id[self.BOS_TOKEN]
        self.eos_token_id = self.char_to_id[self.EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Convert a string into a list of token integers."""
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.bos_token_id)
        for char in text:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                raise ValueError(f"Unknown character '{char}' not in tokenizer vocabulary.")
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special: bool = False) -> str:
        """Convert a list or tensor of token integers back into text."""
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.detach().cpu().view(-1).tolist()
        else:
            ids = list(token_ids)

        chars = []
        for tid in ids:
            if tid in self.id_to_char:
                ch = self.id_to_char[tid]
                if skip_special and ch in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN):
                    continue
                chars.append(ch)
            else:
                chars.append(f"<UNK_{tid}>")
        return "".join(chars)

    def pad_sequences(
        self,
        sequences: list[list[int]],
        pad_to_length: int | None = None,
        padding_side: str = "right",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a batch of sequences to a uniform length.

        Returns:
            padded_ids: Tensor of shape (batch_size, max_len).
            attention_mask: Tensor of shape (batch_size, max_len) with 1.0 for valid tokens and 0.0 for PAD.
        """
        max_len = pad_to_length if pad_to_length is not None else max(len(s) for s in sequences)
        batch_size = len(sequences)

        padded_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)

        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_len)
            if padding_side == "right":
                padded_ids[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)
                attention_mask[i, :seq_len] = 1.0
            elif padding_side == "left":
                padded_ids[i, max_len - seq_len :] = torch.tensor(seq[:seq_len], dtype=torch.long)
                attention_mask[i, max_len - seq_len :] = 1.0
            else:
                raise ValueError(f"Unknown padding_side: {padding_side}")

        return padded_ids, attention_mask
