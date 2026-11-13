# Experiment 11: Why the SFT adapter stops short

[PR #8](https://github.com/mbzuaiciao/rl-reasoning-lab/pull/8) reported 8/8 complete paths at depths 2–3 and 0/9 correct answers at depths 4–6. This diagnostic uses the same saved adapter, pinned Qwen checkpoint, letter prompt, and nine deeper development graphs from [PR #6](https://github.com/mbzuaiciao/rl-reasoning-lab/pull/6).

- **A — Original:** greedy generation, 48 new tokens.
- **B — Extended:** same prompt and greedy settings, 96 new tokens.
- **C — Verified prefix:** same 48-new-token cap as A. Prefill the open assistant turn with `R:` and the first two verified transitions, without an EOS or a trailing semicolon. The original prompt asks for the path and answer; the model must choose how to continue the partial response.

Terminal accuracy and complete paths use the unchanged verifier. For C, the verifier sees the reconstructed text to check grammar and connectivity, but a complete continuation requires every remaining edge and the answer to be generated. The supplied edges are excluded from all generated transition counts and token counts. C is an oracle intervention on development, so its complete paths are conditional on the supplied prefix.

For diagnosis, parse the `R:` line independently of the `A:` line and count consecutive correct edges until the first invalid transition. The first invalid position is one-based within generated transitions; the raw output also gives the position in the full path. A missing next edge is not an invalid emitted edge. An EOS with a valid but incomplete trace line is a **premature trace stop**. We separate a valid scalar answer emitted at an intermediate node from arrows placed on the answer line. Those misplaced arrows can show evidence of continued traversal but never earn trace credit. Token counts include EOS and exclude the prompt and any supplied prefix.

## Reproduce

Use the local pinned snapshot and the exact adapter saved by [PR #8's command](exp11_sft_warmstart.md#reproduce). The script checks both against the recorded hashes and needs no network. Run from the repository root in the same Python 3.13.12 environment with Torch 2.15.0, Transformers 4.57.6 and PEFT 0.18.1:

```bash
HF_HUB_OFFLINE=1 python experiments/exp11_depth_diagnostic.py \
  --local-snapshot /tmp/exp11-qwen \
  --adapter /tmp/exp11-sft-issue7/adapter \
  --output /tmp/exp11-depth-diagnostic.json
pytest -q
```

Use a new output file. The [raw results](../experiments/results/exp11_depth_diagnostic.json) retain each continuation, token IDs, termination reason, and per-graph scores. The command is inference only, with a 10-minute wall timer and no held-out evaluation.

## Results

**Stop this experimental direction with the current adapter and protocol.** A reproduced all nine PR #8 responses and token counts. B emitted exactly the same token IDs as A on every graph. C generated the exact suffix of A's response on every graph; it did not change where the model left the trace line or what it answered. All 27 responses ended at EOS and none reached its token cap.

| Condition | Depth | Correct answers | Complete paths | Valid generated trace steps / required | Premature trace stops (scalar answers) | New tokens, median (range) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A and B, each | 4 | 0/3 | 0/3 | 9/12 | 3 (2) | 18 (18–20) |
| A and B, each | 5 | 0/3 | 0/3 | 10/15 | 3 (3) | 18 (18–22) |
| A and B, each | 6 | 0/3 | 0/3 | 11/18 | 3 (1) | 36 (18–40) |
| A and B, each | All | 0/9 | 0/9 | 30/45 | 9 (6) | 18 (18–40) |
| C | 4 | 0/3 | 0/3 | 3/6 | 3 (2) | 9 (9–11) |
| C | 5 | 0/3 | 0/3 | 4/9 | 3 (3) | 9 (9–13) |
| C | 6 | 0/3 | 0/3 | 5/12 | 3 (1) | 27 (9–31) |
| C | All | 0/9 | 0/9 | 12/27 | 9 (6) | 9 (9–31) |

Every emitted edge on the trace line was valid and contiguous, so the first invalid trace transition is absent for all 27 responses. The independent trace-line count includes three responses per condition that earned zero process reward because their answer line contained arrows. The unchanged strict verifier accepted the format of the other six.

- **Early scalar answer, graph 0 (depth 4):** A and B emitted `R: K->Z; Z->I; I->D` then `A: D`, missing `D->J`. C generated only `; I->D` and `A: D` after the two supplied edges. The first missing edge was path transition 4.
- **Continued traversal on the wrong line, graph 3 (depth 4):** The trace stopped at `R->I`, then `A: I->F` contained the correct last transition where a single `F` was required.
- **Continued traversal past the sink, graph 2 (depth 6):** After ending the trace line at `F->Q`, the model placed the two correct remaining edges `Q->M; M->H` on the answer line, then emitted `H->W` and further edges. If the answer line is read as a path extension, the first invalid edge is path position 7 (the fifth model-generated edge in C), after reaching the true sink. Graph 5 also reaches its sink and invents an outgoing edge at position 7.

The strongest evidence is for **leaving the trace too early**, not running out of tokens or failing to find a valid starting prefix. Six responses ended with a well-formed answer at an intermediate node; three entered answer mode early but still generated the remaining correct edges there. Those cases rule out a blanket inability to continue a trajectory. The test does not establish whether the six early answers reflect a learned length preference, a failure to recognize that the current node has an outgoing edge, or both. C supplied edges the model already got right, so it does not test recovery from a wrong prefix or whether forcing a later continuation would help. This is a single checkpoint and nine greedy examples.

PR #8 also found effectively identical standardized advantages for this adapter on all three sampled training batches. Together, the experiments give no basis for a targeted PPO follow-up with the current setup. No training or checkpoint changes were made.

The 27 decodes and setup took **35.58 s**, with **2.82 GiB** peak process RSS on Apple M2 CPU. All 128 existing tests, nine verifier controls, and Ruff checks passed.
