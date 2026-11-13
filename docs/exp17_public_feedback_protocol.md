# EXP-017 · Frozen public-feedback protocol

[Issue #21](https://github.com/mbzuaiciao/rl-reasoning-lab/issues/21) · [EXP-015 protocol](exp15_repair_protocol.md) · [EXP-016 report](exp16_token_cap_report.md)

Use the first three EXP-015 tasks in fixture order, `parse_fields`, `latest_records` and `allocate_units`. These cover strings, records and arithmetic. Each produced naturally stopped responses at 512 tokens in EXP-016, making nine generations plausible on the local CPU. Freeze this subset before sampling. Use the original fixture digests, direct-answer prompts, Qwen3.5-4B GGUF digest, llama.cpp build, four CPU threads, temperature 0.8, top-p 0.95, no top-k or prompt cache, and a 512-token cap for every generation. Use a 2,048-token context for all turns to accommodate the original prompt, up to 512 tokens of initial output, the revision instructions and feedback, and up to 512 revision tokens.

| Task | Initial seed | Shared revision seed | Revision order |
| --- | ---: | ---: | --- |
| `parse_fields` | 17001 | 17101 | Feedback, control |
| `latest_records` | 17002 | 17102 | Control, feedback |
| `allocate_units` | 17003 | 17103 | Feedback, control |

Generate one initial completion per task and run its three public cases if applicable. Both revisions receive the unchanged initial prompt, the exact same raw initial completion and the same revision instruction. The feedback block lists each public case's pass/fail, with the expected result for a mismatch or the sandbox status for an execution failure. For an inapplicable initial response, the block states that no public cases could run. The control block says “No results were provided.” A shared seed does not guarantee equal stochastic draws under different prompts. Acceptance inputs, outcomes, and references stay out of generation. See the frozen [runner](../experiments/exp17_public_feedback.py) for the exact wording.

Use the existing `desktop-linux` per-user socket and EXP-015 isolation, resource and fixture controls for the selected tasks. Check the cached GGUF, backend and local server readiness before starting the 600-second deadline. Run one case per offline Docker container with the existing limits. Grade each revision on public and acceptance cases before sampling the next. Stop on infrastructure failure or when the deadline prevents another operation. Preserve available outcomes and leave interrupted evaluations missing. Attempt each slot once.

Record exact prompts and feedback, original text, token IDs, stop reasons, timings, completion and applicable replacement SHA-256 hashes, and per-case outcomes. Inapplicable revisions fail acceptance and have no public score. Report applicability, acceptance, public passes and case-level fixes and regressions relative to the initial attempt. Compare complete pairs by acceptance and identify cases fixed by feedback alone or passing only without feedback. Leave comparisons involving missing evaluations pending. Go only with at least two complete pairs where feedback passes acceptance and control fails, and at most one showing the reverse. The small, selected sample and published development acceptance fixtures support a descriptive feasibility comparison.
