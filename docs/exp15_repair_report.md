# EXP-015 · Fresh code-repair feasibility

[Issue #17](https://github.com/mbzuaiciao/rl-reasoning-lab/issues/17) · [Frozen protocol](exp15_repair_protocol.md) · [Raw results](../experiments/results/exp15_repair_feasibility.json)

**No-go for a matched-budget RL comparison under the frozen gate.** Only 1 of 8 tasks produced both an accepted and a rejected applicable patch, below the required four. Only 2 applicable rejected patches improved their public score, below the required four. The only accepted patch passed all three public diagnostics. The result applies to the pinned Qwen3.5-4B configuration and its 160-token budget on this development set.

The pilot used eight original Python tasks covering escaped fields, versioned records, integer allocation, dependency ordering, interval loads, recursive aliases, segment selection, and scheduling. The three public cases per task supplied diagnostic feedback. Four additional committed cases per task determined local acceptance. The two splits used distinct inputs. The acceptance fixtures are development data.

## Results

The pilot finished in 416.2 seconds after 42.6 seconds of preflight. The Docker probes and all 56 reference outcomes passed, and each buggy control failed at least one case in each split. Docker ran through the existing per-user socket as macOS UID 501. There were no unresolved infrastructure errors, ungraded applicable patches, or repeated inference slots.

Of 24 completions, 10 were applicable, 3 improved the public score, and 1 passed all four acceptance cases. Twenty completions reached the token cap. Fourteen of those were inapplicable, including all three outputs for each of `parse_fields` and `water_peak`. The six capped outputs that parsed as applicable all failed acceptance. Several ended with syntactically valid but unfinished function bodies. Longer tasks frequently spent their available tokens on comments before reaching the return statement.

| Task | Applicable / 3 | Accepted / 3 | Improved public, failed acceptance | Public / acceptance regressions |
| --- | ---: | ---: | ---: | ---: |
| parse_fields | 0 | 0 | 0 | — |
| latest_records | 2 | 0 | 2 | 0 / 3 |
| allocate_units | 2 | 0 | 0 | 4 / 2 |
| dependency_order | 2 | 0 | 0 | 2 / 4 |
| water_peak | 0 | 0 | 0 | — |
| expand_alias | 2 | 1 | 0 | 1 / 1 |
| best_segment | 1 | 0 | 0 | 1 / 1 |
| next_slot | 1 | 0 | 0 | 1 / 1 |
| **Total** | **10** | **1** | **2** | **9 / 12** |

- `latest_records` sample 1 fixed the inclusive cutoff and raised the public score from 1/3 to 2/3. It compared timestamps with stored payloads, causing a type error on a string payload. Sample 3 passed 3/3 public diagnostics, yet passed only 1/4 acceptance cases. Its strict timestamp comparison mishandled equal-time updates and deletions. The higher public score came with fewer acceptance passes and two acceptance regressions.
- `expand_alias` sample 3 tracked cycle detection along the current expansion path and passed all seven cases. Sample 1 was cut off after a comment inside the function and failed all seven. The public score correctly ordered the only comparable accepted/rejected pair. The pair count was one correct ordering, zero ties and zero inversions.
- Review of all 24 completions found no case-specific hardcoding, grader manipulation, or evidence of acceptance answers reaching the model. The full-public failure shows the three diagnostics leave some required behavior untested. Public scores were terminal diagnostic measurements for each independently sampled patch. No test feedback entered generation.

The raw results preserve all original completions, token IDs, completion SHA-256 digests and applicable replacement digests. The per-case rows were independently checked against the aggregate scores, regressions, rankings and hashes. All 144 repository tests and Ruff passed.

## Reproduce

Use the cached EXP-014 `Qwen3.5-4B-UD-Q4_K_XL.gguf`, llama.cpp build `2a66703`, and the existing Docker Desktop `desktop-linux` context with its per-user socket and cached evaluator image. Run as your regular user. Use a new path for each result file.

```bash
python experiments/exp15_repair_feasibility.py \
  --server /path/to/llama-server \
  --gguf /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --output /tmp/exp15-pilot.json
python -m pytest -q
ruff check .
```

If Docker interrupts grading, first confirm the existing per-user daemon is available. Use `--regrade /tmp/exp15-pilot.json --output /tmp/exp15-regraded.json` in place of the model arguments. This reruns isolation and fixture controls, verifies the completion and extracted-function hashes, and grades saved applicable patches that lack an acceptance result.

The freeze was committed before sampling. The first invocation stopped on a transcribed digest mismatch. The digest was corrected against the frozen file before any inference. Fixture contents, sampling settings, and scoring stayed fixed.
