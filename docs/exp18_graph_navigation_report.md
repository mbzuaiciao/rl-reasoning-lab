# EXP-018 · Stepwise feedback in graph navigation

[Issue #23](https://github.com/mbzuaiciao/rl-reasoning-lab/issues/23) · [Frozen protocol](exp18_graph_navigation_protocol.md) · [Raw trajectories](../experiments/results/exp18_graph_navigation.json)

**No-go under the frozen criterion.** Both arms solved 2/10 episodes (20%). All ten pairs tied on success. The gate required at least three feedback-only successes and at most one state-only success across at least eight complete pairs. The pilot ran for 273.0 seconds after 6.4 seconds of preflight.

| Metric | State-only | Intermediate feedback |
| --- | ---: | ---: |
| Episodes solved | 2/10 | 2/10 |
| Total attempts | 54 | 54 |
| Attempts to goal on `g04` and `g07` | 4, 4 | 4, 4 |
| Invalid attempts | 0 | 1 |
| Attempts that reduced goal distance | 28 | 32 |

Both arms took identical routes on the two successes. The oracle needed three moves on `g04` and four on `g07`. Among the eight pairs that exhausted their budgets, feedback ended closer in four (`g02`, `g03`, `g06`, `g10`), farther in three (`g01`, `g05`, `g08`), and at the same distance in one (`g09`). On `g10`, feedback ended one edge from the goal and state-only ended five edges away. The single invalid attempt was a non-neighbor choice on `g08`.

All 108 model responses stopped at EOS. An independent replay verified the first-turn match in every pair, each observation and transition, and all reported counts. The raw trajectories have SHA-256 `5934024b6ed10cc70651fce12ece356c2cd96dc68579cdb5cfae24429ff6ddc8`.

This result covers one checkpoint, ten designed graphs, and one paired rollout per graph. It measures inference-time use of immediate feedback with the full graph already visible and provides no evidence about the effects of training.

## Preflight

The shortest-path oracle solved all ten frozen episodes in three to six moves. The fixed-first-neighbor baseline solved none. Independent forward search confirmed the distance table and the feedback on 227 legal and invalid attempts. Both arms saw the full directed graph. On turn one they received identical observations, and on later turns the feedback arm received one additional line about the previous attempt. The runtime used the pinned cached Qwen3.5-4B GGUF on four local CPU threads.

The first launch stopped before sampling because the desktop sandbox blocked the private Unix socket. The retry ran as the regular macOS user with local socket access and used the frozen settings.

## Reproduce

Use the cached `Qwen3.5-4B-UD-Q4_K_XL.gguf` from EXP-014 and the installed llama.cpp build `2a66703`. Use a fresh output path.

```bash
python experiments/exp18_graph_navigation.py \
  --server /path/to/llama-server \
  --gguf /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --output /tmp/exp18-reproduction.json
python -m pytest -q
ruff check .
```
