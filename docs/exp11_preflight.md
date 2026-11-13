# Experiment 11 · Pretrained graph traversal preflight

[Issue #5](https://github.com/mbzuaiciao/rl-reasoning-lab/issues/5) asks whether verified intermediate transitions improve longer-depth graph traversal after RL from a shared pretrained checkpoint. The primary comparison is longer-depth terminal-node accuracy for outcome-plus-process RL versus outcome-only RL.

## Starting point

Phase 10 motivates this comparison. Its capstone driver simulates seed-level outcomes from treatment profiles calibrated on earlier phases. Phase 5 contains a runnable tiny-model training example. Experiment 11 would directly measure the effect on a pretrained model in this graph family. [Phase 10 report](phase10_capstone_report.md) · [Capstone driver](../experiments/phase10_capstone.py) · [Phase 5 example](../experiments/phase5_process_vs_outcome_rl.py)

The preflight reuses `GraphTraversalTask`, `TraceEvaluator`, and `compute_ppo_policy_loss`. The existing LM trainer, rollout buffer, and model-level evaluation tie generation to `TinyTransformer` and `CharTokenizer`. Their response masking, batch reward normalization, and evaluation breakdown inform the proposed comparison. The Experiment 11 wrapper anchors the trace grammar and requires a trace for complete-path credit. The legacy reward's answer-only fallback would otherwise award process credit for a correct final answer. Use the existing paired bootstrap for the final analysis.

Use Qwen2.5-0.5B-Instruct (0.49B parameters, Apache-2.0), revision `7ae557604adf67be50417f59c2c2f167def9a775`, with its tokenizer and chat template from that same revision. Instruction tuning gives this small checkpoint a chance to follow the output contract without a warm start. The BF16 weights occupy about 0.99 GB on disk, and CPU FP32 loading gives a portable local feasibility check. [Pinned checkpoint and tokenizer](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) · [License](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE)

## Frozen graph splits

Each example contains three disjoint directed chains of the same depth, with node labels sampled without replacement from A–Z. Shuffle all premises and query the first node of one chain. Equal-length distractors make component length alone uninformative about which sink to choose. The original preflight used the same prompt and fixed worked example on digits across splits. OOD varies chain depth, which also increases the number of premises and response tokens.

| Split | Depths, balanced | Size | Seed |
| --- | --- | ---: | ---: |
| IID test | 2, 3 | 256 | 1101 |
| Depth OOD test | 4, 5, 6 | 384 | 1102 |
| IID development | 2, 3 | 8 | 1103 |
| Depth OOD development | 4, 5, 6 | 9 | 1104 |
| Training | 2, 3 | 1,024 | 1105 |

Generate in table order and reject duplicate sorted directed edge sets or full ordered query paths across all splits, including within each split. Shared labels, local transitions and isomorphic shapes are part of the task distribution. The preflight checks prompt lengths and prints split hashes. It decodes only the two development sets and a 16-prompt training sample. Lock the test hashes before any approved training. Keep test instances and labels out of prompt development, optimization, reward callbacks, early stopping and checkpoint selection. Decode them once after the final checkpoint is fixed.

## Controlled comparison, pending approval

- Initialize two identical LoRA adapters from the pinned model for each paired seed (1100 for a pilot). Freeze the backbone. Use rank 8, alpha 16, `q_proj` and `v_proj`, zero adapter dropout, AdamW at 1e-5 and gradient clipping at 1.0. Evaluate the unchanged checkpoint as a zero-training baseline. A train-only supervised or rejection-sampling comparator would require a separately accounted budget.
- Arm A receives 1 for a correct terminal `A: X` line and 0 otherwise. Arm B receives the identical outcome reward plus 0.5 times the fraction of valid contiguous steps from the queried start, divided by depth. The prefix ends at the first missing, disconnected or invalid transition. A malformed or absent trace earns zero process reward. Score after completion, using only graph edges and the queried start for transition checks. The environment supplies the scalar reward to the optimizer. The policy gets no reference path, target, per-step correction, or verifier calls during decoding.
- Use clipped token PPO with clip 0.2, response-token loss only, and sequence rewards standardized over each 16-rollout batch (population SD, epsilon 1e-8, center only for near-zero SD). Broadcast the scalar advantage over that response. This tests supervision derived from intermediate steps using sequence-level credit. Use no critic or reference KL in either arm. Log reward variance, the sampled old/new KL, clip fractions, and generated lengths.
- Pair the exact train prompt order and sampling seeds. Use one on-policy completion per prompt at temperature 1, full vocabulary, and a 48-token response cap. Stop scoring at EOS, including EOS as an action. For equal scheduled rollout compute, fill any remaining decode slots with masked dummy work after EOS. Run two PPO epochs per rollout batch, one sequence per microbatch padded to 256 tokens, accumulating over 16 sequences per optimizer step. Each arm has 64 rollout batches, 49,152 scheduled completion slots, 2,048 update sequences and 128 optimizer steps. Log actual response actions, padding, verifier calls, and measured time for both arms. Score both reward components for both arms and use the designated reward for optimization.
- Evaluate both arms and the frozen baseline with one greedy response per held-out prompt and the same cap. Report accuracy overall and by depth, complete-path rate, invalid edge count per emitted parsed step, trace format rate, cap rate and response length. Report invalid edge rate as undefined when there are zero parsed steps. Show representative failures such as template copying, wrong start, invalid edge, broken chain or truncation. Bootstrap paired differences on the same held-out instances, stratified by depth, and use seed-paired estimates for any later multi-seed replication. Single-seed intervals quantify item variation conditional on that seed.
- The added reward changes the supervision density and scalar reward distribution. Both arms use the same normalization, but the comparison cannot isolate information content from easier credit assignment. Other confounders include format compliance, pretraining and instruction-tuning priors, length, and shortcut use. Complete-path and format metrics help distinguish these within the tested graph family. This experiment supports claims about synthetic graph-depth transfer only.

## Reproduce the preflight

From the repository root, in a Python 3.13.12 environment:

```bash
python -m pip install -e '.[dev]' 'torch==2.15.0' 'transformers==4.57.6' 'peft==0.18.1'
python experiments/exp11_preflight.py --checks-only
python experiments/exp11_preflight.py
pytest -q
```

The model command downloads the pinned public weights on first use. For a predownloaded copy of that exact revision, use `--local-snapshot /path/to/snapshot`. The command runs 33 decodes and one disposable LoRA optimizer step on CPU using four Torch threads. The step uses a fixed positive advantage to test sampled action log-probability alignment, backpropagation, and an actual adapter change even if the real reward has no variance. It saves no weights. The measured run used a local snapshot.

## Observed results and budget

Measured on an Apple M2 MacBook Air with 24 GB unified memory, Python 3.13.12 and PyTorch 2.15.0. Torch exposed CPU only in the execution environment. The run took **66.84 s** with **2.91 GiB peak process RSS**, excluding download time. Model and tokenizer loading took 2.53 s. The longest prompt was 182 tokens. All nine verifier controls passed, covering exact answers, answer-only output, wrong answers, invalid and disconnected steps, repeats, malformed syntax, and duplicate answers. The LoRA step updated 540,672 trainable parameters in 0.90 s with gradient norm 3.62, maximum parameter change 1e-5 and maximum action log-probability difference 0.000255.

| Preflight sample | Correct answers | Parseable traces | Complete paths | Nonzero process rewards | Generated tokens | Capped | Decode time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy IID dev | 0/8 | 0/8 | 0/8 | 0/8 | 197 | 1 | 24.20 s |
| Greedy deeper dev | 0/9 | 0/9 | 0/9 | 0/9 | 173 | 0 | 12.90 s |
| Sampled train | 0/16 | 0/16 | 0/16 | 0/16 | 338 | 1 | 23.01 s |

Invalid edge rate was undefined across these samples. On the first IID failure, for instance, the query started at `M` and the correct path was `M->V; V->P`. The model emitted `R: 0->1; 1->2; 2->3; 3->4` and `A: 4`. The first deeper failure repeated the same digit trace. The first sampled training failure emitted `R: 0->1; 1->3; 3->0` and `A: 0`. These small samples characterize the model and prompt combination.

Frozen split SHA-256 fingerprints (canonical serialization in the script):

```text
iid       33fa91e8c7cbd9468f6e0eb9eeb9764c3b62b2bdcbfce6567a55056e330a3c9c
ood       5a8469056a3e843cc6806e48b08c30a281e0b942615de01432921213fc169b4c
dev_iid   9a744b2f87e82890e61f6891395adb29f2933db95c1b7542e4e918c99d39bde2
dev_ood   a20adfad80aee8d57a17787e54be66e9f71e9969f9dff0540f1c3189bc033c58
train     7b8f7878014804708a29dfef8d58a1156a68c299f5d2cda2780d6c8e272b5ad9
```

For one hypothetical paired seed, allow roughly **3–6 CPU hours** including final evaluation. The estimate uses 98,304 scheduled training decode slots at approximately 15 tokens/s on the training sample (which includes prefill), 4,096 padded forward/backward sequences at up to 0.90 s each, and up to 92,160 evaluation slots over both adapters and baseline. Actual EOS usage, thermal throttling, and implementation overhead can move this estimate. A dense compute approximation is on the order of **4 × 10¹⁵ FLOPs** using 2 FLOPs per parameter per forward token and up to three forward equivalents per update token. It omits attention and systems overhead. FP32 weights are about 1.84 GiB and FP32 adapter parameters, gradients and Adam moments total about 8 MiB. Allow 4–6 GiB peak process RSS for serial CPU microbatches with activation and logit workspace. The 256-token FP32 logit tensor alone is about 148 MiB. Only the smoke-step memory was measured. At 20–40 W system draw, 3–6 hours would use about 0.06–0.24 kWh (about $0.02–$0.07 at an illustrative $0.30/kWh). There were no cloud resources or paid compute charges.

**No-go for the controlled comparison with this checkpoint and prompt as configured.** Both reward streams were constant at zero on the sampled rollouts. The optimizer result shows that the local machine can change an adapter, while the reward floor makes this comparison uninformative. Any renewed feasibility check should establish usable reward variation on training and development data before opening the frozen test sets.

## Prompt diagnostic

```bash
python experiments/exp11_preflight.py --prompt-diagnostic --local-snapshot /path/to/snapshot
```

Compare the original numeric demonstration, the same example graph relabeled with uppercase letters, and the original instructions with the demonstration removed. Every variant sees the same 8 IID and 9 deeper development graphs, one greedy decode each. Rank variants by how many of the two reward streams vary on these completions, then by strict format compliance, independently extracted correct answers, and total valid prefix steps. Break ties by the order above. Sample the selected prompt once on each of the first 16 frozen training graphs at temperature 1 with full-vocabulary sampling and Torch seed 1100. All decodes use the same chat template, EOS handling and 48-token cap. The diagnostic mode exits before the optimizer code. Frozen test identities are reserved during split construction, and only development and training examples are sent to the model.

Strict format means two lines matching the existing verifier's trace and answer grammar. The independent answer check reads a single label on the last line, accepting a bare label or a label introduced by `A`, `Answer`, `Final answer` or `Final node` with a colon or “is.” It tolerates lowercase, simple Markdown emphasis and trailing punctuation. The independent prefix check reads arrows on the first explicit trace line or the first line of an unlabeled output, accepts ASCII or Unicode arrows, and counts matching directed edges from the queried start up to the first mismatch. A missing or unrecognized output scores zero for the corresponding diagnostic. These extraction rules are diagnostic only. The original verifier determines both reward streams.

| Prompt / sample | Strict format | Correct answer, independent | Any valid prefix | Valid prefix steps | Complete paths | Nonzero process | Nonzero outcome | Nonzero hybrid | Capped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Numeric / greedy dev | 0/17 | 0/17 | 0/17 | 0/65 | 0/17 | 0/17 | 0/17 | 0/17 | 1/17 |
| Letters / greedy dev | 17/17 | 3/17 | 5/17 | 9/65 | 1/17 | 5/17 | 3/17 | 7/17 | 0/17 |
| Instruction only / greedy dev | 0/17 | 0/17 | 2/17 | 2/65 | 0/17 | 0/17 | 0/17 | 0/17 | 2/17 |
| Letters / sampled train | 12/16 | 2/16 | 3/16 | 5/40 | 2/16 | 2/16 | 2/16 | 2/16 | 0/16 |

The greedy letter prompt had two full valid prefixes, including one with an incorrect terminal answer. Its hybrid rewards were zero on 10 items and positive on seven. Several wrong answers earned partial process credit, while two correct answers earned no process credit. In the sampled training batch, outcome rewards were 0 on 14 items and 1 on two. Hybrid rewards were 0 on those same 14 items and 1.5 on those same two. The 67 decodes took 88.07 s at 2.10 GiB peak process RSS on the same M2 CPU, with the local checkpoint and no optimizer step.

On the same development graph with start `M` and true path `M->V; V->P`, the numeric prompt elicited a digit chain ending in `4`. The instruction-only prompt emitted fragments such as `R: MG; AV; PF` and ran into the token cap. The letter prompt emitted `R: M->G; G->F; F->P` with the correct `A: P`, but `M->G` and `F->P` are absent from the graph. A sampled letter response started with the invented edge `Y->X` where the true first edge was `Y->O`. Another had the valid starting transition under the independent diagnostic but failed the strict grammar, as reflected by three independent positive prefixes and two positive process rewards.

**Retain no-go for a training pilot on this evidence.** The numerical demonstration explains the original formatting collapse. Relabeling it restores format, but valid traversal remains rare even with format compliance. The letter prompt clears the raw format and reward-variance checks in the sampled training batch. However, the hybrid reward there is exactly 1.5 times the outcome reward for every rollout, so batch standardization makes the two advantage vectors effectively identical. The batch therefore gives no evidence of additional process credit under the planned training sampler. The greedy development batch does contain distinct process credit, so this small diagnostic cannot establish how often it would occur in more stochastic batches, or whether the checkpoint could solve the task under other prompts.

## Stopping criteria

Stop before a controlled comparison if verifier controls or split integrity fail, the optimizer smoke test fails, target hardware cannot fit the update with margin, or the projected paired run exceeds the approved time or compute cap. On development data, stop for a greedy accuracy ceiling (at least 90% at longer depth) or a format/reward floor (fewer than half of sampled rollouts have a parseable trace, or no variation in either reward stream). If the preflight clears those gates and a paired pilot is approved, use the fixed last checkpoint. Stop with an inconclusive or negative result if the process arm's longer-depth accuracy gain is below 5 percentage points or its paired item interval includes zero. One seed is a pilot, and any stronger claim requires replication across seeds.
