"""Replay EXP-019 without inference, accounting for checkpoint template whitespace.

The unchanged frozen sampling runner expected a trailing newline inside the rendered
prompt. Qwen's template trims it. This verifier checks the exact observed template
and all frozen hashes; it neither changes observations nor resamples trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from . import exp19_partial_navigation as exp
else:
    import exp19_partial_navigation as exp


def verify(report):
    if report["frozen"] != exp.verify_freeze():
        raise ValueError("Frozen implementation mismatch")
    episodes = exp.load_episodes()
    if report["preflight"] != exp.preflight(episodes):
        raise ValueError("Preflight mismatch")
    for key, value in dict(
        model=exp.model.MODEL,
        revision=exp.model.REVISION,
        gguf_sha256=exp.model.SHA256,
        episodes_sha256=exp.EPISODES_SHA256,
        sampling=dict(exp.model.SAMPLING, cap=exp.CAP),
        server_flags=exp.runtime.FLAGS,
        budget_seconds=exp.SECONDS,
    ).items():
        if report[key] != value:
            raise ValueError(f"Configuration mismatch: {key}")
    plan = [
        (episode, arm, index)
        for index, episode in enumerate(episodes)
        for arm in (exp.ARMS if index % 2 == 0 else exp.ARMS[::-1])
    ]
    rows = report["rows"]
    if len(rows) > len(plan):
        raise ValueError("Extra trajectories")
    turns_checked = 0
    for slot, row in enumerate(rows):
        episode, arm, index = plan[slot]
        if (row["episode"], row["arm"]) != (episode["id"], arm):
            raise ValueError("Trajectory order mismatch")
        env = exp.environment(episode)
        for turn_index, turn in enumerate(row["turns"]):
            observation = env.observe(arm == "feedback")
            rendered = (
                "<|im_start|>user\n"
                + observation.strip()
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
            if turn["observation"] != observation or turn["rendered_prompt"] != rendered:
                raise ValueError("Observation or checkpoint template mismatch")
            for field in ("observation", "completion", "rendered_prompt"):
                if turn[field + "_sha256"] != exp.digest(turn[field]):
                    raise ValueError("Saved hash mismatch")
            if (
                turn["seed"] != 190000 + 10 * (index + 1) + turn_index + 1
                or not 0 <= turn["completion_tokens"] <= exp.CAP
                or turn["model_stop"] not in {"eos", "limit"}
            ):
                raise ValueError("Seed, tokens or model stop mismatch")
            # Independently reconstruct parsing, transitions and forward distances.
            before = env.node
            text = turn["completion"].strip()
            action = text if len(text) == 1 and "A" <= text <= "Z" else None
            valid = action in env.graph[before]
            after = action if valid else before
            distance_before = exp.graph.forward_distance(env.graph, before, env.goal)
            distance_after = exp.graph.forward_distance(env.graph, after, env.goal)
            label = (
                "decreased"
                if distance_after < distance_before
                else "increased"
                if distance_after > distance_before
                else "stayed the same"
            )
            expected = dict(
                action=action,
                valid=valid,
                before=before,
                after=after,
                distance_before=distance_before,
                distance_after=distance_after,
                feedback=exp.graph.FEEDBACK.format(label),
            )
            if any(turn[key] != value for key, value in expected.items()):
                raise ValueError("Independent transition mismatch")
            env.step(turn["completion"])
            turns_checked += 1
        if row["stop_reason"] != env.stop_reason:
            raise ValueError("Trajectory stop mismatch")
        if slot < len(rows) - 1 and row["stop_reason"] is None:
            raise ValueError("Incomplete nonfinal trajectory")
    summary = exp.summarize(rows)
    if report["summary"] != summary:
        raise ValueError("Evaluation summary mismatch")
    if report["stop_reason"] == "complete" and len(rows) != len(plan):
        raise ValueError("Missing completed trajectories")
    return dict(verified=True, trajectories=len(rows), turns=turns_checked, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.input.read_text()))
    result["raw_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(f"Verified {result['trajectories']} trajectories and {result['turns']} turns")


if __name__ == "__main__":
    main()
