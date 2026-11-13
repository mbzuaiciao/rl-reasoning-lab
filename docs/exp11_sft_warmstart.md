# Experiment 11: Local supervised warm-start

[Issue #7](https://github.com/mbzuaiciao/rl-reasoning-lab/issues/7) follows the [preflight in PR #6](https://github.com/mbzuaiciao/rl-reasoning-lab/pull/6). That preflight found effectively identical standardized outcome and hybrid advantages on one sampled training batch. This pilot asks whether a small supervised warm-start produces verifiable partial paths that distinguish the rewards. It runs no reinforcement learning.

## Protocol

- Reuse the [preflight's frozen splits, letter prompt, graph generator and verifier](exp11_preflight.md). The script checks the five original split fingerprints and the nine verifier controls. Held-out identities are reserved by the generator and hashed mechanically; their labels are never exposed to the researcher, shown to the model, scored, or used for model selection.
- Use Python's RNG seeded with 1100 to draw 64 depth-2 and 64 depth-3 examples without replacement from the frozen training split. Sort by frozen index, render each known path as a two-line completion, and require correct answer, complete path and full prefix credit from the original verifier. The script saves the training rows and hashes their serialization. Exact graphs and query paths cannot cross the frozen splits; shorter subpaths, labels and graph structure can recur.
- Use the pinned Qwen2.5-0.5B-Instruct revision and chat template. Train one FP32 LoRA adapter on CPU with four Torch threads, rank 8, alpha 16, zero adapter dropout and targets `q_proj` and `v_proj`. Freeze the backbone. Optimize mean assistant response cross-entropy including EOS, excluding all prompt tokens, with AdamW (learning rate 3e-4, zero weight decay), gradient norm cap 1, one sequence per microbatch and four per update. Plan three seeded shuffled passes (96 updates); save the final adapter only. A 30-minute wall timer stops optimization if the cap is reached.
- On each checkpoint, decode one greedy completion on each of the 8 IID and 9 deeper development graphs. Draw 48 distinct unused examples from the frozen training split with the continued selection RNG and divide them into three batches of 16. Use one full-vocabulary sample per graph at temperature 1, resetting the Torch RNG to 1100, 1101 and 1102 at the start of the respective batches, with the same prompt order and seeds for both models. All decodes share the 48-token cap and pinned EOS settings. Development is used only after the last checkpoint is saved.
- Score outcome as 1 for a correct final `A: X` line and 0 otherwise, process as the fraction of the valid contiguous prefix from the queried start (zero for malformed traces), and hybrid as outcome + 0.5 × process. A complete path requires the full valid trace and a matching terminal answer. Standardize sequence rewards separately within each batch with population SD, epsilon 1e-8, and center only if SD is at most 1e-6, as in the repository rollout trainer. Compare the two resulting vectors on the same completions. Correlation is undefined when either vector is constant.
- For this pilot, count a wrong-answer rollout with a nonempty but incomplete valid prefix as direct evidence of distinct partial credit. Require at least two such rollouts and an advantage RMSE of at least 0.10 in at least two of the three batches, and more complete paths across the 17 development examples without a decline on deeper development. These are descriptive feasibility gates, not significance tests. Formatting alone cannot clear them. If the time cap or an evidence gate is missed, stop before RL. Any subsequent authorized PPO comparison would start both arms from the same saved adapter.

## Reproduce

From the repository root (Python 3.13.12):

```bash
python -m pip install -e '.[dev]' 'torch==2.15.0' 'transformers==4.57.6' 'peft==0.18.1'
python experiments/exp11_sft_warmstart.py --output-dir /tmp/exp11-sft-pilot
pytest -q
ruff check experiments/exp11_sft_warmstart.py
```

Use a new output directory. Add `--local-snapshot /path/to/pinned/snapshot` to use an existing copy of the exact revision without downloading. The directory contains `sft_train.jsonl`, the PEFT `adapter/`, and `results.json` with all dev and sampled train completions, per-item scores, and per-batch advantage vectors. No test completions are generated.

The measured run used the same local snapshot as the preflight. Its weight SHA-256 matches the pinned [Hugging Face file](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors): `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`. The generated training set SHA-256 is `9115ecea180601dde3060ed1655e1f7b81119ceb746d0f7180abfe84cca8d04b`.

## Measured results

**No-go for PPO from this adapter.** SFT improved graph traversal on development, but made sampled training rollouts nearly uniformly correct. The single partially correct training trajectory received process credit with zero outcome reward, yet the standardized advantages were effectively identical in every batch. The unchanged baseline actually had distinguishable advantages in two of these three batches; PR #6's original single-batch result did not characterize all baseline samples.

| Checkpoint | Split | N | Correct answer | Exact format | Any valid prefix | Valid steps / possible | Full prefix | Complete path | Partial prefix, wrong answer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | IID dev | 8 | 3 | 8 | 4 | 8/20 | 2 | 1 | 2 |
| Base | Deeper dev | 9 | 0 | 9 | 1 | 1/45 | 0 | 0 | 1 |
| SFT | IID dev | 8 | 8 | 8 | 8 | 20/20 | 8 | 8 | 0 |
| SFT | Deeper dev | 9 | 0 | 6 | 6 | 19/45 | 0 | 0 | 6 |
| Base | Train 1 | 16 | 5 | 13 | 8 | 15/40 | 6 | 4 | 2 |
| Base | Train 2 | 16 | 4 | 12 | 3 | 6/42 | 1 | 1 | 2 |
| Base | Train 3 | 16 | 2 | 11 | 0 | 0/45 | 0 | 0 | 0 |
| SFT | Train 1 | 16 | 15 | 16 | 16 | 38/40 | 15 | 15 | 1 |
| SFT | Train 2 | 16 | 16 | 16 | 16 | 42/42 | 16 | 16 | 0 |
| SFT | Train 3 | 16 | 16 | 16 | 16 | 45/45 | 16 | 16 | 0 |

Rewards below are shown as `reward: count`. The [raw results](../experiments/results/exp11_sft_warmstart.json) include development histograms, matched completions, invalid edges, cap counts, and the advantage vectors.

| Checkpoint | Batch | Outcome | Process | Hybrid | Pearson r | Advantage RMSE | Max abs. difference |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| Base | 1 | 0: 11, 1: 5 | 0: 8, 1/3: 2, 1: 6 | 0: 7, 1/6: 2, 1/2: 2, 1: 1, 1.5: 4 | 0.951 | 0.313 | 0.714 |
| Base | 2 | 0: 12, 1: 4 | 0: 13, 2/3: 2, 1: 1 | 0: 10, 1/3: 2, 1: 3, 1.5: 1 | 0.950 | 0.317 | 0.682 |
| Base | 3 | 0: 14, 1: 2 | 0: 16 | 0: 14, 1: 2 | 1 | 0 | 0 |
| SFT | 1 | 0: 1, 1: 15 | 1/3: 1, 1: 15 | 1/6: 1, 1.5: 15 | 1 | 1.03e-8 | 4.00e-8 |
| SFT | 2 | 1: 16 | 1: 16 | 1.5: 16 | undefined | 0 | 0 |
| SFT | 3 | 1: 16 | 1: 16 | 1.5: 16 | undefined | 0 | 0 |

On train item 933 the adapter followed `G->Z`, invented `Z->X`, then followed `X->Y` and answered `Y`. The correct path was `G->Z; Z->A; A->S`. It earned process reward 1/3 despite zero outcome reward. Within that batch, however, hybrid was an affine function of outcome: `1/6 + (4/3) × outcome`. Standardization erased the difference except for the numerical epsilon. The other two batches had constant rewards and zero advantages.

On deeper development item 0, the adapter emitted `K->Z; Z->I; I->D` and answered `D`, omitting the final `D->J` step. It correctly began six of nine deeper paths under the strict verifier, versus one for the baseline, while completing none. Five of the six parseable SFT traces stopped after three steps; one emitted four. The three malformed responses put an arrow in the answer line, so they earned no process reward. This is evidence of improved partial traversal alongside a tendency to stop early, with no observed longer-depth success. Both checkpoints already had perfect IID development formatting.

Training took **185.74 s** for 96 updates to 540,672 adapter parameters. Mean response losses were 0.05947, 0.00404 and 0.00018. End-to-end wall time was **359.69 s** and peak process RSS was **2.92 GiB** on Apple M2 CPU (macOS arm64; NumPy 2.5.4). The total includes a PEFT metadata lookup that retried for about 23 seconds in the measured run; the script now skips the lookup because the vocabulary is unchanged. The saved adapter weights (2.1 MiB) had SHA-256 `b3062f6b4c98d07f564c85fddb99354197eaf1ae9c41fc609810ec438a7ad975`. Saving and reloading offline produced identical adapter files.

The evidence comes from one trained seed, 17 greedy development completions and 48 sampled training completions per model. Neither the observed training ceiling nor the deeper partial-credit rate is a population estimate. Changing the training distribution, sampler or reward after seeing these results would answer a different question. The held-out tests were not opened for evaluation. No cloud resources or RL training were used.

All 128 existing tests and nine verifier controls passed. Ruff passed on the new script. PR #6 and Phase 10 artifacts are unchanged.
