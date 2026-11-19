# EXP-019 · Progress feedback with hidden graph topology

[Issue #25](https://github.com/mbzuaiciao/rl-reasoning-lab/issues/25) · [Frozen protocol](exp19_partial_navigation_protocol.md) · [Raw trajectories](../experiments/results/exp19_partial_navigation.json) · [Replay evaluation](../experiments/results/exp19_replay.json) · [Artifact hashes](../experiments/exp19/artifacts.sha256)

**No-go.** Control solved 6/8 episodes and feedback solved 5/8. There were zero feedback-only successes, one control-only success (`p07`), and seven success ties. The frozen descriptive gate required eight complete pairs, at least three feedback-only successes, and at most one control-only success. All eight pairs finished in **194.181 seconds**, following 8.075 seconds of runtime preflight. Stop this line; no additional sampling, tuning or expanded experiment followed the result.

| Metric (eight complete pairs) | State-only | Feedback |
| --- | ---: | ---: |
| Goal successes | 6/8 | 5/8 |
| Total attempted moves | 34 | 38 |
| Attempts to goal among successes, fixture order | 5, 4, 3, 3, 4, 3 | 6, 5, 3, 3, 3 |
| Invalid actions | 4 | 5 |
| Sum of final shortest-path distances | 6 | 7 |
| Mean final distance | 0.750 | 0.875 |
| Distance-increasing valid moves | 4 | 6 |
| Wrong-turn events returning to pre-turn distance or closer | 4/4 | 6/6 |
| Episodes reaching goal after a wrong turn | 1 | 1 |
| Prompt tokens | 4,149 | 5,205 |
| Generated tokens | 68 | 76 |

For the five pairs both arms solved, feedback took one extra attempt on `p02` and `p03`, and the same attempts on `p04`, `p06`, and `p08` (mean 4.0 versus 3.6 attempts). Both extra attempts were invalid moves. On the common failures, final distance tied at two on `p01`; feedback ended closer on `p05` (two versus four). On `p07`, control reached the goal in four moves while feedback exhausted six attempts at distance three.

Both arms recovered from the initial wrong turn on `p02` and subsequently reached the goal. Every recorded distance-increasing event eventually returned to its prior distance, including loops on failed episodes; returning to a prior distance did not imply finding the goal. Feedback had more wrong-turn events and no additional goal recovery. Recovery counts are conditional on each arm's actual trajectory, include overlapping events, and do not establish a causal recovery advantage.

## Design and preflight

The protocol, exact fixture, runner, tests and reused implementation hashes were committed **before inference**, in `af89dc841c403a19a95e16f0fad30fa5662f346b`. The sampling sources and `freeze.json` remain unchanged. EXP-018's deterministic transitions and pinned local Qwen3.5-4B runtime were reused, with full adjacency removed from prompts. Both arms received only local state and the same observed-history format. Feedback added one truthful distance-direction line after each action. Numerical distances and hidden topology occur only in evaluation artifacts.

The shortest-path oracle solved all eight episodes in alternating four and three moves. The fixed-first-neighbor baseline solved 3/8. Independent forward search checked all distances; preflight checked 128 legal and invalid transitions in both the reused environment and partial-observation wrapper. Every episode had a recoverable distance-increasing choice at the start.

Each episode's stored hidden-world counterpart admits the same initial observation and the same observed first action/history, yet gives opposite progress signals and disjoint optimal next actions. For example, in `p01` and `p02`, start `G`, goal `H`, and move `G -> E` produce the same displayed state/history at `E`, including legal neighbors `G, C`. The distance decreased in `p01`, where the optimal next move is `C`; it increased in `p02`, where the optimal next move is `G`. Feedback therefore supplies information unavailable in the control observation. No counterpart identity or witness was displayed to the model.

The cached GGUF digest and llama.cpp build `2a66703` matched the EXP-014 pins. The run used four CPU threads, zero GPU layers, operation offload disabled, offline mode, and the frozen sampling settings. All 72 action responses stopped at EOS. No downloads, Docker, cloud services, paid APIs, training, generated-code execution, privilege elevation, or system changes were used. The private local socket ran as the regular user outside the desktop filesystem sandbox.

## Replay correction and artifact integrity

The sampling command completed all pairs and saved all raw data, then exited with a replay assertion. Its frozen replay check incorrectly expected the observation's final newline to occur inside the rendered checkpoint prompt. The actual checkpoint template trims surrounding user-content whitespace. This was a verifier defect; model prompts and raw responses were preserved.

To retain the exact implementation used for sampling, the frozen runner was left intact. A separate [replay-only verifier](../experiments/exp19_replay.py) checks the exact checkpoint template with its whitespace trimming, original freeze hashes, all observation/output/rendered-prompt hashes, seeds, token limits, trajectory order, stops, independently reconstructed transitions and forward distances, preflight and evaluation summaries. It verified **16 trajectories and 72 turns** without further inference. [Replay tests](../tests/test_exp19_replay.py) cover the actual template behavior and reject changed prompts, outputs and transitions. The raw JSON is unchanged; its original `stop_reason: complete` refers to sampling, and the separate replay artifact records successful verification.

The raw trajectory SHA-256 is `6d8c214b41a0c690759c04dc2c48fc69ab62b480fa0d905664d00cc82044e729`. The [artifact manifest](../experiments/exp19/artifacts.sha256) also covers the server log, replay evaluation, protocol, frozen source hashes and report. All **162 tests pass** and **Ruff passes** after adding replay verification.

## Limits and reproduction

This bounded inference-time pilot covers two designed topologies under four relabelings, eight paired rollouts, one checkpoint, one seed per episode/turn and short action budgets. Relabelings are not independent topology samples. Alternating arm order also aligns with the two alternating topology variants. Invalid actions, label choices and histories affect recovery opportunities. The result supplies no evidence about training or broader navigation performance.

With the existing pinned local backend and cached weights, use a fresh output path. The historical frozen runner's final replay check has the whitespace defect described above; after sampling, use the separate verifier. Do not rerun to improve this pilot's result.

```bash
python experiments/exp19_partial_navigation.py \
  --server /path/to/llama-server \
  --gguf /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --output /tmp/exp19-reproduction.json
python experiments/exp19_replay.py /tmp/exp19-reproduction.json \
  --output /tmp/exp19-replay.json
python -m pytest -q
ruff check .
shasum -a 256 -c experiments/exp19/artifacts.sha256
```
