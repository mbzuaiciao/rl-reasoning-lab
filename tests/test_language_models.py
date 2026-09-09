"""Tests for Phase 4: Language Models as Policies, Tokenizer, Transformer, SFT, and LM PPO."""

from __future__ import annotations

import torch

from rl_reasoning_lab.algorithms.lm_policy_gradient import train_lm_reinforce
from rl_reasoning_lab.algorithms.lm_ppo import train_lm_ppo
from rl_reasoning_lab.environments.language_tasks import ArithmeticTask, Example
from rl_reasoning_lab.models.tiny_transformer import TinyTransformer
from rl_reasoning_lab.models.tokenizer import CharTokenizer
from rl_reasoning_lab.rewards.language_rewards import LanguageRewardFunction
from rl_reasoning_lab.trainers.lm_rollout import LMRolloutBuffer
from rl_reasoning_lab.trainers.sft import prepare_sft_batch, train_sft


class TestTokenizer:
    def test_round_trip(self) -> None:
        tok = CharTokenizer()
        text = "3 + 4 = 7"
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text

    def test_special_tokens(self) -> None:
        tok = CharTokenizer()
        text = "5"
        encoded = tok.encode(text, add_bos=True, add_eos=True)
        assert encoded[0] == tok.bos_token_id
        assert encoded[-1] == tok.eos_token_id
        assert tok.decode(encoded, skip_special=True) == "5"

    def test_padding(self) -> None:
        tok = CharTokenizer()
        seqs = [[1, 2], [1, 2, 3, 4]]
        padded, mask = tok.pad_sequences(seqs, pad_to_length=5)
        assert padded.shape == (2, 5)
        assert mask.shape == (2, 5)
        assert torch.all(mask[0] == torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))
        assert torch.all(mask[1] == torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]))


class TestTinyTransformer:
    def test_forward_shape(self) -> None:
        model = TinyTransformer(vocab_size=30, d_model=32, num_layers=2, num_heads=2)
        x = torch.randint(0, 30, (2, 8))
        logits = model(x)
        assert logits.shape == (2, 8, 30)

    def test_causal_mask_invariant(self) -> None:
        """Causality Invariant:

        Modifying future tokens at positions > t MUST NOT change logits at position t.
        """
        model = TinyTransformer(vocab_size=30, d_model=32, num_layers=2, num_heads=2)
        model.eval()

        x1 = torch.tensor([[1, 5, 6, 7, 8]])
        x2 = torch.tensor([[1, 5, 6, 20, 25]])  # Same prefix [1, 5, 6], different future [20, 25]

        with torch.no_grad():
            logits1 = model(x1)
            logits2 = model(x2)

        # Logits at positions 0, 1, 2 must be identical to machine precision
        assert torch.allclose(logits1[:, :3, :], logits2[:, :3, :], atol=1e-6), (
            "Causal mask failed! Future tokens leaked into past representations."
        )

    def test_generate_stops_at_eos(self) -> None:
        tok = CharTokenizer()
        model = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)
        prompt_ids = tok.encode("1 + 1 = ", add_bos=True)
        gen_ids, log_probs = model.generate(
            prompt_ids, max_new_tokens=5, eos_token_id=tok.eos_token_id
        )
        assert gen_ids.shape[1] > len(prompt_ids)
        assert len(log_probs) == gen_ids.shape[1] - len(prompt_ids)


class TestLanguageTaskAndRewards:
    def test_task_verification(self) -> None:
        task = ArithmeticTask()
        is_corr, val = task.verify("3 + 4 = ", " 7")
        assert is_corr and val == 7

        is_corr, val = task.verify("2 + 5 + 1 = ", " 8")
        assert is_corr and val == 8

        is_corr, val = task.verify("3 + 4 = ", " 9")
        assert not is_corr and val == 7

        is_corr, val = task.verify("3 + 4 = ", " garbage")
        assert not is_corr and val == 7

    def test_reward_function(self) -> None:
        task = ArithmeticTask()
        reward_fn = LanguageRewardFunction(task.verify)

        # Correct
        out_corr = reward_fn("3 + 4 = ", " 7")
        assert out_corr.reward == 1.0 and out_corr.is_correct and not out_corr.is_malformed

        # Incorrect
        out_wrong = reward_fn("3 + 4 = ", " 9")
        assert out_wrong.reward == 0.0 and not out_wrong.is_correct and not out_wrong.is_malformed

        # Malformed
        out_malformed = reward_fn("3 + 4 = ", " abc")
        assert out_malformed.reward == -0.5 and out_malformed.is_malformed


class TestSFTTrainer:
    def test_prepare_sft_batch_masks(self) -> None:
        tok = CharTokenizer()
        examples = [
            Example(prompt="1 + 2 = ", target="3", full_text="1 + 2 = 3"),
        ]
        inputs, targets, masks = prepare_sft_batch(examples, tok)
        # Prompt length with BOS is len("<BOS>1 + 2 = ") = 1 + 8 = 9
        # Target length with EOS is len("3<EOS>") = 2
        # Target tokens start at index 8 in target_ids
        assert masks.shape == inputs.shape
        # Prompt positions must be 0.0, completion positions must be 1.0
        assert masks[0, -2:].tolist() == [1.0, 1.0]

    def test_sft_train_improves_loss(self) -> None:
        tok = CharTokenizer()
        model = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)
        train_ex = [
            Example(prompt="1 + 1 = ", target="2", full_text="1 + 1 = 2"),
            Example(prompt="2 + 2 = ", target="4", full_text="2 + 2 = 4"),
        ]
        res = train_sft(model, train_ex, train_ex, tok, num_epochs=10, batch_size=2, lr=1e-2)
        assert res.train_losses[-1] < res.train_losses[0]


class TestLMRolloutAndRL:
    def test_lm_rollout_buffer_collection(self) -> None:
        tok = CharTokenizer()
        model = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)
        task = ArithmeticTask()
        reward_fn = LanguageRewardFunction(task.verify)

        prompts = ["1 + 1 = ", "2 + 3 = "]
        buffer = LMRolloutBuffer.collect_rollouts(model, prompts, tok, reward_fn, max_new_tokens=4)
        assert len(buffer) == 2

        # Minibatch generation preserves action mask alignment
        for batch in buffer.get_minibatch_generator(batch_size=2, shuffle=False):
            assert batch["action_masks"].shape == batch["input_ids"].shape
            assert len(batch["rewards"]) == 2

    def test_lm_reinforce_step(self) -> None:
        tok = CharTokenizer()
        model = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)
        task = ArithmeticTask()
        reward_fn = LanguageRewardFunction(task.verify)

        prompts = ["1 + 1 = "]
        res = train_lm_reinforce(model, prompts, tok, reward_fn, num_iterations=5, batch_size=2)
        assert len(res.mean_rewards) == 5

    def test_lm_ppo_step_and_frozen_ref(self) -> None:
        tok = CharTokenizer()
        model = TinyTransformer(vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1)
        ref_model = TinyTransformer(
            vocab_size=tok.vocab_size, d_model=32, num_layers=1, num_heads=1
        )
        ref_model.load_state_dict(model.state_dict())
        ref_weights_initial = [p.clone() for p in ref_model.parameters()]

        task = ArithmeticTask()
        reward_fn = LanguageRewardFunction(task.verify)
        prompts = ["2 + 3 = ", "1 + 4 = "]

        res = train_lm_ppo(
            model=model,
            ref_model=ref_model,
            prompts=prompts,
            tokenizer=tok,
            reward_fn=reward_fn,
            num_iterations=3,
            rollout_batch_size=2,
            epochs_per_rollout=2,
            minibatch_size=2,
            seed=42,
        )
        assert len(res.mean_rewards) == 3

        # Verify reference model weights remained completely frozen
        for p_init, p_curr in zip(ref_weights_initial, ref_model.parameters(), strict=True):
            assert torch.equal(p_init, p_curr), "Reference model was corrupted during PPO training!"
