# EXP-013 · Test feedback for code repair

[Issue #13](https://github.com/mbzuaiciao/rl-reasoning-lab/issues/13) · [Frozen protocol](exp13_code_repair_protocol.md) · [Raw run](../experiments/results/exp13_code_repair.json)

## Reproduce

Use a local Linux Docker daemon reachable over a Unix socket (the run used Docker Desktop's `desktop-linux` context), Python 3.13.12, PyTorch 2.15.0 and Transformers 4.57.6. Supply an existing local copy of the exact EXP-012 checkpoint; the runner verifies all five model and tokenizer files and never downloads weights. Cache the pinned official Python runtime image if it is missing. The evaluator itself uses `--pull=never`.

```bash
docker --context desktop-linux pull python@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
python experiments/exp13_code_repair.py --context desktop-linux \
  --local-snapshot /path/to/cached/snapshot --output /tmp/exp13-reproduction.json
python -m pytest -q
```

Use a new output path. Add `--preflight-only` to check the runtime and checkpoint without sampling patches. The runner writes an explicit stop reason if preflight fails and never substitutes a host evaluator.

## Preflight

On macOS arm64, launching the existing Docker Desktop prompted for administrator authorization. The Docker backend log recorded its bundled install helper running `config --user … set-docker-socket`, then recorded the user's cancellation. That helper configures the [optional default socket at `/var/run/docker.sock`](https://docs.docker.com/desktop/setup/install/mac-permission-requirements/#installing-symlinks). The experiment instead used the existing `desktop-linux` context and per-user socket. No administrator authorization, Docker reinstall, or system setting changes were needed.

Trusted probes verified the active network interface was loopback only, an external connection was unreachable, the root filesystem was read-only, and there was no host workspace or Docker socket in the container. They also checked UID/GID 65534, zero effective and bounding capabilities, seccomp, no new privileges, a cleared environment, and cgroup limits (128 MiB memory, no swap, 16 PIDs, half a CPU). Separate probes confirmed CPU exhaustion killed the worker, excess output was cut off, and an idle worker was removed after five seconds. A fresh container ran each case. All eight trusted reference functions passed all 56 cases, each buggy function failed at least one public and hidden case, and the cached model completed an unrelated offline smoke decode.

The public score is a count over three diagnostics, so it has only four levels before baseline subtraction. Hidden acceptance checks four additional cases per task and still cannot prove correctness. Each candidate was sampled independently: this pilot measures whether scores could help select candidates, and does not test whether a model improves when fed test results. The tasks and model are too few to estimate a general reward effect. No training, paid inference, or cloud jobs were run.

## Pilot results

**No-go for a separate RL proposal under the frozen gate.** Only **1 of the required 4** tasks produced both a successful and a failed applicable patch. Only **2 of the required 4** applicable patches improved the public score while failing hidden acceptance. Review found no evidence of hidden test leakage.

The full 24-sample pilot took **151.38 seconds** after **37.28 seconds** of preflight. **10/24** samples applied and **2/24** passed hidden acceptance (**2/10** among applicable patches). Both hidden successes passed every public case. Eight samples reached the token cap; all eight were inapplicable. Other format failures included extra prose, an import, test calls, and a changed function name. Some rejected completions contained plausible repairs, so the applicability floor limits what this pilot says about the model's repair ability.

| Task | Applicable / sampled | Hidden successes | Improved public, failed hidden | Public / hidden regressions |
| --- | ---: | ---: | ---: | ---: |
| clamp | 2/3 | 1 | 0 | 0 / 1 |
| chunk | 1/3 | 0 | 0 | 1 / 1 |
| dedupe | 1/3 | 0 | 0 | 0 / 0 |
| overlap | 2/3 | 0 | 0 | 1 / 0 |
| label | 1/3 | 0 | 1 | 0 / 0 |
| left_index | 1/3 | 1 | 0 | 0 / 0 |
| rotate | 0/3 | 0 | 0 | — |
| totals | 2/3 | 0 | 1 | 0 / 0 |
| **Total** | **10/24** | **2** | **2** | **2 / 2** |

- **Useful partial credit:** `label` sample 1 collapsed repeated whitespace, and `totals` sample 2 summed duplicate keys. Each moved from 1/3 to 2/3 public cases. Both retained a bug: `label` skipped lowercasing, and `totals` still merged keys that differed in case.
- **Misleading aggregate:** `chunk` sample 2 gained the public empty-input case but broke the previously passing exact-chunk case, leaving its public score unchanged. It also regressed on a hidden case. No applicable patch passed all public cases and failed hidden acceptance.
- **Within-task ranking:** The only comparable success/failure pair was on `clamp`: the successful patch gained two public passes and the failed patch gained none. That is one concordant pair, with zero ties or inversions. Neither partially improving patch had an applicable successful patch on the same task, so the run did not test whether the score could rank those partial repairs below a full repair.
- **Reward hacking and leakage:** Review of all completions found no exact-input branching, hardcoded case answers, or grader manipulation. Some rejected samples copied public examples into tests or docstrings. The model saw those examples by design; the prompt and container boundaries kept hidden answers out of generation. The public cases would still be easy to memorize, and the hidden cases are visible to future users of the published fixtures.

The raw file contains all sampled completions, token IDs, baseline results, per-case outcomes, and aggregate scores. The public deltas, regressions, hidden acceptance, and pair counts were independently recomputed from the case rows. All 139 repository tests passed.
