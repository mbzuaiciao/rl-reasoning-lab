# EXP-014 · Local coding-model selection

[Issue #15](https://github.com/mbzuaiciao/rl-reasoning-lab/issues/15) · [EXP-013 protocol](exp13_code_repair_protocol.md) · [EXP-013 results](exp13_code_repair_report.md) · [EXP-014 raw results](../experiments/results/exp14_model_selection.json)

## Local inventory

MacBook Air M2, 24 GB unified memory and about 54 GiB of free workspace-volume storage at preflight. The inventory covered the Hugging Face, LM Studio and Ollama caches, the repository, and the temporary directory containing the EXP-013 snapshot. All three shortlisted artifacts use Apache-2.0. Sizes below cover text weights; the two GGUF caches also hold separate vision projectors, which the text-only comparison leaves unloaded.

| Policy and artifact | Pinned revision and precision | Storage and CPU RAM | Runtime and inference cost |
| --- | --- | --- | --- |
| Baseline: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) | `7ae557604adf67be50417f59c2c2f167def9a775`, BF16 safetensors, loaded as FP32 | 0.99 GB weights, about 2 GB FP32 weights in memory before runtime overhead | PyTorch 2.15.0 and Transformers 4.57.6 on four CPU threads. EXP-013 measured 151 s for generation plus grading and 37 s for preflight. |
| First candidate: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523) | `e87f176479d0855a907a41277aca2f8ee7a09523`, `UD-Q4_K_XL` | 2.91 GB weights, 6.05 GB peak child-process RSS measured | Local llama.cpp on four CPU threads and a 1,024-token context. Measured 225 s of prefill and decode, 281 s with load and grading. |
| Reserve candidate: [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), [Unsloth GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) | `ce152932ac27bc40bc9c727386760424d50bb456`, `UD-Q4_K_XL` | 5.10 GB weights, provisionally budget 6–8 GB of unified memory for a single short-context process | Publisher documents llama.cpp support. A 10-minute run needs at least 6.4 tokens/s for 24 full-length completions before load, prefill, or grading. A rough weight-bandwidth extrapolation from Qwen gives about 6 tokens/s, or about 11 minutes of decode at the full cap. Different architectures and output lengths make this a planning estimate. Gemma remained in reserve. |

The upstream Qwen and Gemma model cards report coding benchmark results, which motivated the shortlist. The local artifacts use 4-bit weights and the comparison uses short direct answers. Other cached chat weights included Qwen3-14B (7.8 GiB, 4-bit MLX), gpt-oss-20b (11 GiB, MXFP4-Q8 MLX), Gemma 3 12B (8.15 GB, Ollama), and a separate Gemma 4 E4B 4-bit MLX copy (6.4 GiB). The SmolLM3-3B Hugging Face snapshot had only the second of two weight shards. All five baseline files matched the EXP-013 digests. No model downloads, paid inference, or cloud resources were used.

## Comparison method

The eight frozen Python repair tasks have three public diagnostics and four local acceptance cases each. They support a paired development comparison by task and sample index. The acceptance fixtures and reference solutions are committed, so the results measure performance on visible development data. Generation receives only the frozen spec, buggy function and public examples. The runner reuses EXP-013's parser, scoring and fresh-container evaluation. It checks isolation and limits, reproduces the buggy-function outcomes and verifies all 56 reference outcomes before generation.

The comparison reuses the recorded EXP-013 samples and runs Qwen3.5 in direct-answer mode with its own chat template. Both use seeds 13001–13024, three independent samples per task, temperature 0.8 followed by top-p 0.95, no top-k cutoff and a 160-token cap. The same seed labels align sample slots. The tokenizer, template, random-number implementation, quantization and backend differ, so this compares two deployable configurations. The 600-second candidate budget starts before server load and includes the smoke decode and intervening grading. The first launch spent 7.55 seconds before a readiness error, with zero samples generated. The successful readiness check waits for the model to load, and the retry had a 592-second deadline.

## Results

**Go for a separate feasibility study with Qwen3.5-4B.** Applicability rose from 10/24 to 24/24, and acceptance on the local fixtures rose from 2/24 to 22/24. Two tasks, `label` and `rotate`, produced both passing and failing applicable repairs, meeting Issue #15’s multiple-task criterion. Six tasks passed on all three draws. Keep the pinned Qwen3.5 configuration as the stronger candidate and Gemma as the reserve. The original EXP-013 RL gate still fails on the counts of mixed tasks and public-improving acceptance failures.

| Metric | Qwen2.5-0.5B (recorded) | Qwen3.5-4B |
| --- | ---: | ---: |
| Applicable / sampled | 10/24 | 24/24 |
| Applicable with public improvement | 4 | 23 |
| Applicable with lower public score | 1 | 0 |
| Accepted on all four local cases | 2/24 | 22/24 |
| Public improvement with acceptance failure | 2 | 1 |
| Output-format failures | 14 | 0 |
| Reached the token cap | 8 | 0 |
| Tasks with both outcomes among applicable patches | 1 | 2 |

Each row pairs the same three sample slots. Public deltas count tests gained or lost versus the buggy function. A dash marks an inapplicable patch with no public delta.

| Task | Qwen2.5 public deltas | Qwen3.5 public deltas | Accepted out of 3, Qwen2.5 → Qwen3.5 |
| --- | --- | --- | ---: |
| clamp | 0, —, +2 | +2, +2, +2 | 1 → 3 |
| chunk | —, 0, — | +2, +2, +2 | 0 → 3 |
| dedupe | 0, —, — | +2, +2, +2 | 0 → 3 |
| overlap | —, 0, −1 | +2, +2, +2 | 0 → 3 |
| label | +1, —, — | +2, +2, +2 | 0 → 2 |
| left_index | +2, —, — | +2, +2, +2 | 1 → 3 |
| rotate | —, —, — | +2, +2, 0 | 0 → 2 |
| totals | 0, +1, — | +2, +2, +2 | 0 → 3 |

On `label`, the first sample discarded punctuation-only words and still passed all three public cases, tying both accepted repairs. On `rotate`, the third sample rotated in the wrong direction and scored below both accepted repairs. Review found no case-specific hardcoding or grader manipulation in the 24 outputs. The four within-task pairs yielded two public-score ties and two correct rankings. This small development sample supports choosing a model for feasibility work. Reward effectiveness and generalization remain open.

Qwen3.5 evaluated 1,139 generated tokens in 103 s (about 11 tokens/s), plus 122 s of prompt processing. The retry took 316 s wall time, including 281 s with the model server running. Peak child-process RSS was 6.05 GB on macOS, excluding the Docker VM. EXP-013 did not record peak RSS.

The Docker daemon disconnected while grading the final six patches. On relaunch, the backend log showed the per-user socket listening at 20:04:31 PST on November 12, followed by the same `config --user … set-docker-socket` helper as EXP-013. The log records cancellation at 20:06:21 PST. The helper sets up the optional default `/var/run/docker.sock` link. Regrading used the existing `desktop-linux` context and its per-user socket after the isolation and affected fixture controls passed. Regrading took 41 s with zero extra samples and left zero unresolved infrastructure errors. [Docker Desktop permission requirements](https://docs.docker.com/desktop/setup/install/mac-permission-requirements/#installing-symlinks)

The raw results preserve all 24 original completions and token sequences, with SHA-256 digests of the UTF-8 completion and the extracted function for each sample. They retain the six patches’ original infrastructure statuses separately from the verified scores. The runner stops on Docker connection errors and retains the current patch with an unset acceptance result if grading is interrupted.

## Reproduce

Use the cached GGUF file and the already installed llama.cpp server build `2a66703` (LM Studio backend 2.60.0 on this host). Supply a new output path. The runner verifies the GGUF SHA-256, starts an offline server on a private Unix socket, and writes the raw rows and a separate server log. Docker must already have the pinned EXP-013 image.

```bash
python experiments/exp14_model_selection.py \
  --server /path/to/llama-server \
  --gguf /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --context desktop-linux --output /tmp/exp14-reproduction.json
python -m pytest -q
```
