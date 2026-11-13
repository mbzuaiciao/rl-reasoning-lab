# EXP-017 · Public feedback on a second repair attempt

[Issue #21](https://github.com/mbzuaiciao/rl-reasoning-lab/issues/21) · [Frozen protocol](exp17_public_feedback_protocol.md) · [Raw results](../experiments/results/exp17_public_feedback.json)

**No-go under the frozen gate.** All nine completions returned and all three pairs had acceptance decisions. Neither arm passed acceptance on any task. The gate required at least two tasks where feedback alone passed acceptance, with at most one task showing the reverse. The pilot took 270.1 seconds after 22.4 seconds of preflight. The per-user Docker isolation and fixture controls passed. The longest revision prompt occupied 820 tokens in the 2,048-token context.

| Task | Initial public | Feedback public (change) | Control public (change) | Feedback acceptance | Control acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `parse_fields` | 1/3 | 1/3 (0) | 1/3 (0) | 1/4 | 2/4 |
| `latest_records` | 0/3 | 3/3 (+3) | 3/3 (+3) | 2/4 | 2/4 |
| `allocate_units` | 0/3 | Inapplicable | Inapplicable | Fail (inapplicable) | Fail (inapplicable) |

All three initial patches and two revisions per arm were applicable. Among applicable revisions, both arms fixed the same three initial public failures on `latest_records`. Neither arm fixed a public case the other missed, and neither regressed on an initially passing public case. The `allocate_units` initial response and both revisions hit the 512-token cap. Both revisions ended in unclosed code fences and lost applicability. The other six completions ended naturally.

- `parse_fields` kept the same two public failures in both arms. The control alone passed acceptance case 4.
- `latest_records` passed all public cases in both arms. Feedback alone passed acceptance case 4, which checks the later of two equal-time updates. Control alone passed acceptance case 1, which checks equal-time deletion. The feedback revision skipped deletion records entirely.

The acceptance comparison was three ties, with no feedback wins or losses on the all-cases outcome. Acceptance was measured only on revisions, so case-level acceptance differences compare the two arms. These three selected tasks and their published development fixtures provide a feasibility result for this setup. The run used shared revision seeds with different prompts, which can produce different draws.

## Reproduce

Use the cached EXP-015 GGUF, llama.cpp build `2a66703`, and the existing `desktop-linux` context with its per-user socket and cached evaluator image. Run as a regular macOS user and choose a fresh output path.

```bash
python experiments/exp17_public_feedback.py \
  --server /path/to/llama-server \
  --gguf /path/to/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --output /tmp/exp17-pilot.json
python -m pytest -q
ruff check .
```
