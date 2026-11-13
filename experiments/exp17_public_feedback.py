"""Compare paired revisions with and without intermediate public-test feedback."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from time import monotonic

if __package__:
    from . import exp15_repair_feasibility as pilot
    from .exp16_token_cap import CAP
else:
    import exp15_repair_feasibility as pilot
    from exp16_token_cap import CAP

model = pilot.model
repair = pilot.repair
PLAN = (
    ("parse_fields", 17001, 17101, "feedback", "control"),
    ("latest_records", 17002, 17102, "control", "feedback"),
    ("allocate_units", 17003, 17103, "feedback", "control"),
)
CONTEXT = 2048
FLAGS = pilot.FLAGS.copy()
FLAGS[FLAGS.index("--ctx-size") + 1] = str(CONTEXT)
REVISION = """

Revise the first attempt below using the specification above. Return only one complete replacement function with the original name and parameters.

First attempt:
{initial}

Public-test feedback:
{feedback}
"""
NO_FEEDBACK = "No results were provided."


def public_feedback(results, cases):
    if results is None:
        return "The first attempt was inapplicable. No public cases could be run."
    lines = []
    for row, case in zip(results, cases, strict=True):
        prefix = f"Public case {row['case']}"
        if row["passed"]:
            lines.append(f"{prefix}: passed.")
        elif row["status"] == "ok":
            lines.append(f"{prefix}: failed. Result differed from expected {case['expected']!r}.")
        else:
            lines.append(f"{prefix}: failed. Sandbox status: {row['status']}.")
    return "\n".join(lines)


def grade(sandbox, row, source, cases, split, deadline):
    if source is None:
        return
    results = row[split] = []
    for index, case in enumerate(cases, 1):
        # Leave time for the sandbox's per-case timeouts and cleanup.
        pilot.check_budget(deadline - 16)
        (result,) = sandbox.cases(row["task"], source, [case])
        results.append(dict(result, case=index))


def generate(socket, report, name, params, arm, prompt, seed, deadline):
    pilot.check_budget(deadline)
    report["pending"] = dict(task=name, arm=arm, seed=seed, prompt=prompt)
    result = model.generate(socket, prompt, seed, CAP, deadline)
    raw = result["content"]
    source = repair.replacement(raw, name, params)
    row = dict(
        **report.pop("pending"),
        completion=raw,
        completion_sha256=pilot.digest(raw),
        patch_sha256=pilot.digest(source) if source is not None else None,
        token_ids=result["tokens"],
        stop=result["stop_type"],
        timings=result["timings"],
        applicable=source is not None,
        public=None,
        acceptance=None,
        accepted=None if arm == "initial" or source is not None else False,
    )
    report["candidates"].append(row)
    print(f"sampled {name} {arm}", flush=True)
    return row, source


def run(socket, sandbox, report, public, hidden, template, deadline):
    for name, initial_seed, revision_seed, *order in PLAN:
        task = public[name]
        cases = task["cases"]
        initial_prompt = repair.prompt_for(template, task, name)
        initial, source = generate(
            socket, report, name, task["params"], "initial", initial_prompt, initial_seed, deadline
        )
        grade(sandbox, initial, source, cases, "public", deadline)
        feedback = public_feedback(initial["public"], cases)
        report["feedback"][name] = feedback
        for arm in order:
            prompt = initial_prompt + REVISION.format(
                initial=initial["completion"],
                feedback=feedback if arm == "feedback" else NO_FEEDBACK,
            )
            row, source = generate(
                socket, report, name, task["params"], arm, prompt, revision_seed, deadline
            )
            grade(sandbox, row, source, cases, "public", deadline)
            grade(sandbox, row, source, hidden[name]["cases"], "acceptance", deadline)
            if row["applicable"]:
                row["accepted"] = all(case["passed"] for case in row["acceptance"])
            print(f"finished {name} {arm}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = dict(
        phase="incomplete",
        plan=PLAN,
        fixtures=pilot.FROZEN,
        model=model.MODEL,
        revision=model.REVISION,
        gguf_sha256=model.SHA256,
        sampling=dict(model.SAMPLING, cap=CAP),
        server_flags=FLAGS,
        image=repair.IMAGE,
        docker_flags=repair.FLAGS,
        host_uid=os.getuid(),
        feedback={},
        candidates=[],
    )
    command_started = monotonic()
    started = None
    with args.output.open("x") as out:
        try:
            public, hidden, template = pilot.fixtures()
            selected = {name: public[name] for name, *_ in PLAN}
            sandbox = pilot.docker_preflight(selected, hidden, report)
            with args.output.with_suffix(".log").open("x") as log:
                with pilot.local_server(args, report, log, flags=FLAGS) as (socket, _):
                    started = monotonic()
                    report["preflight_seconds"] = round(started - command_started, 3)
                    run(socket, sandbox, report, selected, hidden, template, started + repair.LIMIT)
            report["phase"] = "complete"
            return 0
        except TimeoutError as exc:
            report["stop_reason"] = str(exc)
            print(report["stop_reason"], flush=True)
            return 1
        except subprocess.TimeoutExpired as exc:
            at_deadline = started is not None and monotonic() >= started + repair.LIMIT
            report["stop_reason"] = "Pilot wall-clock budget reached" if at_deadline else str(exc)
            print(report["stop_reason"], flush=True)
            return 1
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            report["stop_reason"] = f"{type(exc).__name__}: {exc}"
            print(report["stop_reason"], flush=True)
            return 1
        finally:
            if started is not None:
                report["pilot_seconds"] = round(monotonic() - started, 3)
            json.dump(report, out, indent=2)
            out.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
