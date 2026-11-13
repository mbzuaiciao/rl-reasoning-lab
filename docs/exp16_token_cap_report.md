# EXP-016 · Qwen3.5-4B output-cap diagnostic

[Issue #19](https://github.com/mbzuaiciao/rl-reasoning-lab/issues/19) · [Frozen protocol](exp16_token_cap_protocol.md) · [160-token results](../experiments/results/exp15_repair_feasibility.json) · [512-token results](../experiments/results/exp16_token_cap.json)

**Incomplete at the time limit.** The 512-token arm returned 13 of 24 completions in 600.4 seconds after 52.5 seconds of preflight. The request for seed 15014 timed out at the deadline. Seeds 15015–15024 were never submitted. All ten applicable patches received all seven case outcomes. The unchanged 1,024-token context accommodated the longest prompt (291 tokens) and the full 512-token allowance. Docker isolation, resource probes and all fixture controls passed through the existing per-user socket as UID 501.

## Paired results

| Metric | 160 tokens, all 24 | 160 tokens, matched 13 | 512 tokens, observed 13 |
| --- | ---: | ---: | ---: |
| Cap hits | 20 | 11 | 5 |
| Generated tokens, mean / median | 153.5 / 160 | 152.1 / 160 | 329.8 / 301 |
| Applicable | 10 | 6 | 10 |
| Improved public score | 3 | 2 | 6 |
| Passed all acceptance cases | 1 | 0 | 3 |
| Improved public score and failed acceptance | 2 | 2 | 3 |
| Tasks with accepted and rejected applicable patches | 1 | 0 | 2 |

Among graded applicable patches, the matched 160-token arm gained one public case once and two cases once. It lost one case twice and two cases twice. The 512-token arm gained one case three times and two cases three times, and lost one case four times. Each public case is worth one third of the score. The full 160-token pilot took 416.2 seconds after 42.6 seconds of preflight.

| Task | Matched seeds | Cap hits, 160 → 512 | Applicable, 160 → 512 | Public improvements, 160 → 512 | Accepted, 160 → 512 |
| --- | --- | ---: | ---: | ---: | ---: |
| `parse_fields` | 15001–15003 | 3 → 1 | 0 → 2 | 0 → 2 | 0 → 1 |
| `latest_records` | 15004–15006 | 1 → 0 | 2 → 3 | 2 → 2 | 0 → 0 |
| `allocate_units` | 15007–15009 | 3 → 0 | 2 → 3 | 0 → 2 | 0 → 2 |
| `dependency_order` | 15010–15012 | 3 → 3 | 2 → 2 | 0 → 0 | 0 → 0 |
| `water_peak` | 15013 | 1 → 1 | 0 → 0 | 0 → 0 | 0 → 0 |

Every matched 160-token sequence was a prefix of its 512-token counterpart, including two identical responses that ended below either cap. This was checked against the recorded token IDs. A shared seed alone does not guarantee a matching draw.

## Truncation and repair errors

Truncation was a material source of failure. At seed 15003, the 160-token `parse_fields` response ended before the final field was appended and was inapplicable. The continuation passed all seven cases. At seed 15007, the 160-token `allocate_units` response compiled but stopped before distributing the remaining units. The continuation passed acceptance. Three `dependency_order` responses still reached 512 tokens. Two compiled with unfinished bodies and failed every case.

Substantive code-repair errors remained. Five of the eight responses that ended naturally failed acceptance. `parse_fields` at seed 15001 passed all three public tests but returned an empty list for empty input. `latest_records` at seed 15005 mixed scalar and tuple dictionary entries and indexed a two-element tuple at index 2. At seed 15006 it passed all public tests but mishandled equal-time updates. Static review of the saved completions found no fixture-specific hardcoding or grader manipulation.

The observed prefix had two mixed-outcome tasks (`parse_fields` and `allocate_units`) and three public-improving acceptance failures. Both counts are below the EXP-015 yardsticks of four, with `expand_alias`, `best_segment`, `next_slot` and two `water_peak` slots missing. The run establishes clear cutoff effects in specific pairs and persistent behavioral failures after natural stops. The incomplete coverage leaves the primary failure mode across all eight tasks unresolved. Acceptance uses published development fixtures.

## Reproduce

Use the cached Qwen3.5-4B GGUF, llama.cpp build `2a66703`, and the existing `desktop-linux` context with its per-user socket and cached evaluator image. Run as a regular macOS user and choose a fresh results path.

```bash
python experiments/exp16_token_cap.py \
  --server /path/to/llama-server \
  --gguf /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --output /tmp/exp16-pilot.json
python -m pytest -q
ruff check .
```

After a Docker interruption, `--regrade /tmp/exp16-pilot.json --output /tmp/exp16-regraded.json` reruns the isolation and fixture controls and verifies the saved hashes before grading any remaining applicable patches. Regrading uses the saved output and leaves any unfinished generation slots missing.
