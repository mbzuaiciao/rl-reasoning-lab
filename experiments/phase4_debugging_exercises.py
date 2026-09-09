"""Phase 4 Debugging Exercises: 8 Common Bugs in Language Model RL.

Demonstrates 8 silent, subtle, or catastrophic bugs in LM RL pipelines:
1. Reinforcing prompt tokens (violates likelihood ratio identity).
2. Padding tokens included in loss calculation (dilutes gradients and distorts loss).
3. Continuing to reinforce tokens after <eos> (spurious post-terminal training).
4. Off-by-one logit-to-target alignment (predicting current token instead of next).
5. Mutating or updating the reference policy weights during PPO (destroys KL anchor).
6. Recomputing old log-probs dynamically rather than caching frozen rollouts (ratio collapses to 1.0).
7. Assigning outcome reward directly to prompt tokens (spurious credit assignment).
8. Generating or evaluating without causal masking (future token leakage).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer


def bug_1_reinforce_prompt_tokens() -> tuple[float, float, str]:
    """Bug 1: Applying policy gradients to prompt tokens instead of only generated tokens."""
    torch.manual_seed(42)
    tok = CharTokenizer()
    model = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)

    prompt = "2 + 2 = "
    completion = " 4"
    full_seq = tok.encode(prompt + completion, add_bos=True, add_eos=True)
    prompt_len = len(tok.encode(prompt, add_bos=True, add_eos=False))

    seq_t = torch.tensor([full_seq], dtype=torch.long)
    logits = model(seq_t[:, :-1])
    log_probs = F.log_softmax(logits, dim=-1)
    target_tokens = seq_t[:, 1:].unsqueeze(-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_tokens).squeeze(-1)

    # Correct mask: 0 on prompt, 1 on completion
    correct_mask = torch.zeros_like(token_log_probs)
    correct_mask[0, prompt_len - 1 :] = 1.0

    # Buggy mask: 1 everywhere including prompt
    buggy_mask = torch.ones_like(token_log_probs)

    reward = -1.0  # Wrong answer penalty
    loss_correct = -(token_log_probs * correct_mask * reward).sum()
    loss_buggy = -(token_log_probs * buggy_mask * reward).sum()

    diagnosis = (
        "Buggy loss updates prompt tokens based on completion reward. "
        "Prompts are fixed conditioning context, not policy actions. "
        "Reinforcing prompt tokens destroys the model's ability to process questions."
    )
    return float(loss_correct.item()), float(loss_buggy.item()), diagnosis


def bug_2_padding_in_loss() -> tuple[float, float, str]:
    """Bug 2: Unmasked padding tokens included in loss denominator and reduction."""
    torch.manual_seed(42)
    logits = torch.randn(2, 10, 20)  # batch 2, seq 10, vocab 20
    targets = torch.randint(0, 20, (2, 10))
    # Batch element 0 has 4 valid tokens; element 1 has 9 valid tokens
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )

    loss_raw = F.cross_entropy(logits.view(-1, 20), targets.view(-1), reduction="none").view(2, 10)

    # Correct: sum over valid tokens divided by valid token count
    correct_loss = (loss_raw * mask).sum() / mask.sum()

    # Buggy: mean over all tokens including padding
    buggy_loss = (loss_raw * mask).mean()

    diagnosis = (
        f"Correct loss normalized by {int(mask.sum())} valid tokens: {correct_loss.item():.4f}. "
        f"Buggy loss averages across all 20 tokens: {buggy_loss.item():.4f} "
        "(dilutes loss by 35% and makes effective learning rate batch-padding dependent)."
    )
    return float(correct_loss.item()), float(buggy_loss.item()), diagnosis


def bug_3_post_eos_tokens() -> tuple[float, float, str]:
    """Bug 3: Continuing to compute policy gradient on garbage tokens generated after EOS."""
    torch.manual_seed(42)
    # Suppose EOS occurred at index 3, but sequence rolled out to index 7
    token_rewards = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    correct_mask = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Stops at EOS
    buggy_mask = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Runs past EOS

    correct_credit = (token_rewards * correct_mask).sum().item()
    buggy_credit = (token_rewards * buggy_mask).sum().item()

    diagnosis = (
        f"Correct credit stopped at EOS: {correct_credit:.1f}. "
        f"Buggy credit accumulates {buggy_credit:.1f} on post-EOS hallucinated tokens, "
        "teaching the policy that rambling past termination yields extra reward."
    )
    return correct_credit, buggy_credit, diagnosis


def bug_4_off_by_one_alignment() -> tuple[float, float, str]:
    """Bug 4: Aligning logits at step t with token t instead of next token t+1."""
    torch.manual_seed(42)
    tok = CharTokenizer()
    model = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)
    seq = tok.encode("1 + 2 = 3", add_bos=True, add_eos=True)
    t_seq = torch.tensor([seq], dtype=torch.long)

    logits = model(t_seq[:, :-1])

    # Correct alignment: logits[:, t] predicts t_seq[:, t+1]
    correct_targets = t_seq[:, 1:]
    correct_loss = F.cross_entropy(logits.transpose(1, 2), correct_targets).item()

    # Buggy alignment: logits[:, t] compared against t_seq[:, t] (current token!)
    buggy_targets = t_seq[:, :-1]
    buggy_loss = F.cross_entropy(logits.transpose(1, 2), buggy_targets).item()

    diagnosis = (
        f"Correct next-token loss: {correct_loss:.4f}. "
        f"Buggy auto-encoding loss: {buggy_loss:.4f}. "
        "Off-by-one trains the model to copy the current input token rather than predicting the next one."
    )
    return correct_loss, buggy_loss, diagnosis


def bug_5_ref_policy_updated() -> tuple[float, float, str]:
    """Bug 5: Accidentally leaving reference policy trainable and updating its parameters."""
    torch.manual_seed(42)
    tok = CharTokenizer()
    ref_correct = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)
    ref_buggy = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)

    # Correct: Explicitly freeze all reference parameters
    for p in ref_correct.parameters():
        p.requires_grad = False
    correct_trainable = sum(p.numel() for p in ref_correct.parameters() if p.requires_grad)

    # Buggy: Reference parameters left un-frozen (requires_grad=True) and passed to optimizer
    buggy_trainable = sum(p.numel() for p in ref_buggy.parameters() if p.requires_grad)

    opt = torch.optim.SGD(ref_buggy.parameters(), lr=0.1)
    loss = ref_buggy.tok_embeddings.weight.sum()
    loss.backward()
    opt.step()

    weight_diff = (
        (ref_buggy.tok_embeddings.weight - ref_correct.tok_embeddings.weight).abs().sum().item()
    )

    diagnosis = (
        f"Correct reference trainable params: {correct_trainable}. "
        f"Buggy reference trainable params: {buggy_trainable} (weight delta after step: {weight_diff:.4f}). "
        "When pi_ref is updated, the reference distribution shifts towards the policy, "
        "rendering the KL penalty ineffective at preventing catastrophic distribution drift."
    )
    return float(correct_trainable), float(buggy_trainable), diagnosis


def bug_6_recomputing_old_log_probs() -> tuple[float, float, str]:
    """Bug 6: Recomputing old log probs with current weights instead of freezing rollout cache."""
    # When old log probs are recomputed using current weights theta:
    # ratio = exp(log pi_theta(a|s) - log pi_theta(a|s)) = exp(0) = 1.0 identically!
    curr_log_prob = torch.tensor([-1.2, -0.8, -2.1])

    # Correct: old_log_prob frozen at rollout time
    cached_old_log_prob = torch.tensor([-1.5, -0.7, -1.9])
    correct_ratio = torch.exp(curr_log_prob - cached_old_log_prob)

    # Buggy: old_log_prob recomputed with current weights
    buggy_ratio = torch.exp(curr_log_prob - curr_log_prob)

    diagnosis = (
        f"Correct ratios reflect policy movement: {[round(x, 3) for x in correct_ratio.tolist()]}. "
        f"Buggy ratio is frozen at exactly 1.0: {[round(x, 3) for x in buggy_ratio.tolist()]}. "
        "The PPO clipping mechanism never triggers and multi-epoch training reduces to unscaled first-order steps."
    )
    return float(correct_ratio.mean().item()), float(buggy_ratio.mean().item()), diagnosis


def bug_7_prompt_token_reward() -> tuple[float, float, str]:
    """Bug 7: Assigning outcome reward directly to prompt tokens during trajectory credit assignment."""
    # Prompt has length 4, completion has length 3. Outcome reward is +1.0.
    prompt_len = 4
    comp_len = 3
    total_len = prompt_len + comp_len

    # Correct: Reward placed at final token or distributed only across completion tokens
    correct_rewards = torch.zeros(total_len)
    correct_rewards[-1] = 1.0

    # Buggy: Outcome reward assigned to every token including prompt
    buggy_rewards = torch.ones(total_len)

    diagnosis = (
        f"Correct total completion reward sum: {correct_rewards.sum().item():.1f}. "
        f"Buggy reward assigns credit to prompts: {buggy_rewards.sum().item():.1f}. "
        "This credits the prompt tokens for the solution, creating high variance in value and advantage estimation."
    )
    return float(correct_rewards.sum().item()), float(buggy_rewards.sum().item()), diagnosis


def bug_8_generation_without_causal_mask() -> tuple[bool, bool, str]:
    """Bug 8: Forward pass or evaluation performed without causal masking, leaking future tokens."""
    torch.manual_seed(42)
    tok = CharTokenizer()
    model = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)
    seq = tok.encode("1 + 2 = 3", add_bos=True, add_eos=True)
    t_seq = torch.tensor([seq], dtype=torch.long)

    # 1. Forward with causal mask (standard)
    logits_causal = model(t_seq)

    # 2. Modify final token to an unrelated token
    t_seq_corrupted = t_seq.clone()
    t_seq_corrupted[0, -1] = tok.encode("9", add_bos=False, add_eos=False)[0]

    logits_corrupted = model(t_seq_corrupted)

    # Check whether token at position 2 changed
    causal_preserved = torch.allclose(logits_causal[0, 2], logits_corrupted[0, 2], atol=1e-5)

    diagnosis = (
        f"Causal invariant verified: position 2 unaffected by position 8 change: {causal_preserved}. "
        "If a non-causal (bidirectional) attention mask is mistakenly used in generation or SFT loss, "
        "tokens attend to future answers, yielding 100% training accuracy but complete inference failure."
    )
    return causal_preserved, not causal_preserved, diagnosis


def run_all_exercises() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold red]Phase 4 Debugging Exercises: 8 Common Bugs in LM RL[/bold red]\n"
            "Detailed inspection of silent bugs that compromise LM policy optimization.",
            title="RL Reasoning Lab — Diagnostics",
        )
    )

    bugs = [
        ("1. Reinforcing Prompt Tokens", bug_1_reinforce_prompt_tokens),
        ("2. Padding Included in Loss Normalizer", bug_2_padding_in_loss),
        ("3. Continuing to Reinforce Post-EOS", bug_3_post_eos_tokens),
        ("4. Off-by-One Logit-to-Target Alignment", bug_4_off_by_one_alignment),
        ("5. Reference Policy Weights Updated", bug_5_ref_policy_updated),
        ("6. Recomputing Old Log-Probs Dynamically", bug_6_recomputing_old_log_probs),
        ("7. Reward Assigned to Prompt Tokens", bug_7_prompt_token_reward),
        ("8. Generation/Loss Without Causal Mask", bug_8_generation_without_causal_mask),
    ]

    table = Table(title="Language Model RL Failure Modes & Invariant Checks")
    table.add_column("Bug Name", style="bold yellow", justify="left")
    table.add_column("Correct Value", justify="right")
    table.add_column("Buggy Value", justify="right")
    table.add_column("Mechanism & Consequence", justify="left")

    for name, func in bugs:
        val_c, val_b, diag = func()
        str_c = f"{val_c:.3f}" if isinstance(val_c, float) else str(val_c)
        str_b = f"{val_b:.3f}" if isinstance(val_b, float) else str(val_b)
        table.add_row(name, str_c, str_b, diag)

    console.print(table)


if __name__ == "__main__":
    run_all_exercises()
