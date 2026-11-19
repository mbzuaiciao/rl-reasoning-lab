"""EXP-019: frozen paired navigation with hidden topology and progress feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from time import monotonic

if __package__:
    from . import exp14_model_selection as model
    from . import exp15_repair_feasibility as runtime
    from . import exp18_graph_navigation as graph
else:
    import exp14_model_selection as model
    import exp15_repair_feasibility as runtime
    import exp18_graph_navigation as graph

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "experiments/exp19/episodes.json"
EPISODES_SHA256 = "4e4ea0ca4554d908cdb415bdb34a1eeac1f24063cd134e6b78d36a5b07d37096"
CAP = 8
SECONDS = 600
ARMS = graph.ARMS
PROMPT = (
    "Navigate a hidden directed graph to reach the goal within the step limit. "
    "Only outgoing neighbors at your current node are visible. "
    "Choose one legal next node. Reply with its single uppercase letter only.\n"
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def verify_freeze():
    frozen = json.loads((ROOT / "experiments/exp19/freeze.json").read_text())
    for name, expected in frozen.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen file mismatch: {name}")
    return frozen


def load_episodes():
    raw = EPISODES.read_text()
    if digest(raw) != EPISODES_SHA256:
        raise ValueError("Frozen episode digest mismatch")
    return json.loads(raw)


class Navigation(graph.Navigation):
    """Use EXP-018 transitions and strict parsing; expose only local observations."""

    def __post_init__(self):
        super().__post_init__()
        self.history = []

    def observe(self, with_feedback):
        prompt = (
            f"{PROMPT}\nGoal: {self.goal}\nCurrent node: {self.node}\n"
            f"Legal next nodes: {', '.join(self.graph[self.node])}\n"
            f"Remaining steps: {self.step_limit - self.steps}\n"
            "Observed history (node, legal neighbors, attempted output, valid, next node):\n"
            + ("\n".join(self.history) if self.history else "(none)")
            + "\n"
        )
        if with_feedback and self.feedback is not None:
            prompt += self.feedback + "\n"
        return prompt

    def step(self, completion):
        node, legal = self.node, list(self.graph[self.node])
        turn = super().step(completion)
        self.history.append(json.dumps([node, legal, completion, turn["valid"], self.node]))
        return turn


def environment(episode):
    return Navigation(episode["graph"], episode["start"], episode["goal"], episode["step_limit"])


def preflight(episodes):
    checks = graph.preflight(episodes)
    by_id = {e["id"]: e for e in episodes}
    for episode in episodes:
        for node, neighbors in episode["graph"].items():
            if node == episode["goal"]:
                continue
            for action in [*neighbors, "?"]:
                probe = Navigation(episode["graph"], node, episode["goal"], 6)
                turn = probe.step(action)
                before = graph.forward_distance(probe.graph, node, probe.goal)
                after = graph.forward_distance(probe.graph, probe.node, probe.goal)
                label = (
                    "decreased"
                    if after < before
                    else "increased"
                    if after > before
                    else "stayed the same"
                )
                expected = graph.FEEDBACK.format(label)
                if (
                    turn["feedback"] != expected
                    or probe.observe(True) != probe.observe(False) + expected + "\n"
                    or "shortest-path" in probe.observe(False)
                ):
                    raise ValueError("Partial observation or feedback mismatch")
        env = environment(episode)
        other = environment(by_id[episode["counterpart"]])
        if env.observe(False) != other.observe(False):
            raise ValueError("Witness initial observations differ")
        action = episode["witness_action"]
        before = env.distances[env.node]
        env.step(action)
        other.step(action)

        def optimal(e):
            return {n for n in e.graph[e.node] if e.distances[n] == e.distances[e.node] - 1}

        if (
            env.observe(False) != other.observe(False)
            or env.feedback == other.feedback
            or optimal(env) & optimal(other)
        ):
            raise ValueError("Witness does not establish informative hidden uncertainty")
        if env.steps + env.distances[env.node] > env.step_limit:
            raise ValueError("Witness wrong turn is not recoverable")
        checks[episode["id"]]["information_witness"] = dict(
            counterpart=episode["counterpart"],
            action=action,
            shared_observation=env.observe(False),
            feedback=env.feedback,
            counterpart_feedback=other.feedback,
            optimal_next=sorted(optimal(env)),
            counterpart_optimal_next=sorted(optimal(other)),
        )
        # Every episode must offer a recoverable distance-increasing move at start.
        if not any(
            env.distances[n] > before and 1 + env.distances[n] <= env.step_limit
            for n in env.graph[env.start]
        ):
            raise ValueError("No recoverable wrong turn")
    return checks


def generate_model(socket, prompt, seed, cap, deadline):
    rendered = model.request(
        socket, "apply-template", {"messages": [{"role": "user", "content": prompt}]}
    )["prompt"]
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("Inference deadline reached")
    result = model.request(
        socket,
        "completion",
        dict(model.SAMPLING, prompt=rendered, seed=seed, n_predict=cap),
        timeout=remaining,
    )
    if result["truncated"] or result["stop_type"] not in {"eos", "limit"}:
        raise ValueError("Unexpected stop or context truncation")
    return dict(result, rendered_prompt=rendered)


def sample(socket, episodes, rows, deadline, *, generate=generate_model):
    for index, episode in enumerate(episodes):
        order = ARMS if index % 2 == 0 else ARMS[::-1]
        for arm in order:
            env = environment(episode)
            row = dict(episode=episode["id"], arm=arm, turns=[], stop_reason=None)
            rows.append(row)
            while env.stop_reason is None:
                if monotonic() >= deadline:
                    raise TimeoutError("Pilot wall-clock budget reached")
                observation = env.observe(arm == "feedback")
                seed = 190000 + 10 * (index + 1) + env.steps + 1
                result = generate(socket, observation, seed, CAP, deadline)
                transition = env.step(result["content"])
                row["turns"].append(
                    dict(
                        observation=observation,
                        seed=seed,
                        observation_sha256=digest(observation),
                        completion_sha256=digest(result["content"]),
                        rendered_prompt=result["rendered_prompt"],
                        rendered_prompt_sha256=digest(result["rendered_prompt"]),
                        completion=result["content"],
                        token_ids=result["tokens"],
                        prompt_tokens=result["tokens_evaluated"],
                        completion_tokens=result["tokens_predicted"],
                        model_stop=result["stop_type"],
                        **transition,
                    )
                )
            row["stop_reason"] = env.stop_reason
        print(f"{episode['id']} pair complete", flush=True)


def recovery(turns):
    wrong = [
        i for i, t in enumerate(turns) if t["valid"] and t["distance_after"] > t["distance_before"]
    ]
    return dict(
        wrong_turns=len(wrong),
        recovered_turns=sum(
            any(t["distance_after"] <= turns[i]["distance_before"] for t in turns[i + 1 :])
            for i in wrong
        ),
        goal_after_wrong_turn=bool(wrong and turns[-1]["distance_after"] == 0),
    )


def summarize(rows):
    summary = graph.summarize(rows)
    completed = {}
    for row in rows:
        if row["stop_reason"] in {"goal", "step_limit"}:
            completed.setdefault(row["episode"], {})[row["arm"]] = row
    pairs = {k: v for k, v in completed.items() if set(v) == set(ARMS)}
    summary["paired_metrics"] = {}
    for name, arms in pairs.items():
        summary["paired_metrics"][name] = {
            arm: dict(
                final_distance=row["turns"][-1]["distance_after"],
                steps=len(row["turns"]),
                steps_to_goal=len(row["turns"]) if row["stop_reason"] == "goal" else None,
                invalid_actions=sum(not t["valid"] for t in row["turns"]),
                **recovery(row["turns"]),
            )
            for arm, row in arms.items()
        }
    for arm in ARMS:
        metrics = [v[arm] for v in summary["paired_metrics"].values()]
        summary["arms"][arm].update(
            {
                key: sum(m[key] for m in metrics)
                for key in (
                    "final_distance",
                    "wrong_turns",
                    "recovered_turns",
                    "goal_after_wrong_turn",
                )
            }
        )
    return summary


def replay(report, episodes):
    """Reconstruct observations/transitions and verify hashes without model inference."""
    if report["frozen"] != verify_freeze():
        raise ValueError("Run freeze mismatch")
    plan = [
        (e, arm, i) for i, e in enumerate(episodes) for arm in (ARMS if i % 2 == 0 else ARMS[::-1])
    ]
    rows = report["rows"]
    if len(rows) > len(plan):
        raise ValueError("Extra rows")
    for j, row in enumerate(rows):
        episode, arm, index = plan[j]
        if (row["episode"], row["arm"]) != (episode["id"], arm):
            raise ValueError("Run order mismatch")
        env = environment(episode)
        for i, turn in enumerate(row["turns"]):
            observation = env.observe(arm == "feedback")
            if (
                turn["observation"] != observation
                or turn["observation_sha256"] != digest(observation)
                or turn["completion_sha256"] != digest(turn["completion"])
                or turn["rendered_prompt_sha256"] != digest(turn["rendered_prompt"])
                or observation not in turn["rendered_prompt"]
                or turn["seed"] != 190000 + 10 * (index + 1) + i + 1
                or not 0 <= turn["completion_tokens"] <= CAP
            ):
                raise ValueError("Prompt, hash, seed or token mismatch")
            expected = env.step(turn["completion"])
            independent = graph.forward_distance(env.graph, env.node, env.goal)
            if (
                any(turn[k] != v for k, v in expected.items())
                or independent != turn["distance_after"]
            ):
                raise ValueError("Transition replay mismatch")
        if row["stop_reason"] != env.stop_reason and not (
            j == len(rows) - 1 and row["stop_reason"] is None
        ):
            raise ValueError("Stop reason mismatch")
        if j < len(rows) - 1 and row["stop_reason"] is None:
            raise ValueError("Incomplete nonfinal row")
    if report["summary"] != summarize(rows):
        raise ValueError("Summary mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replay", action="store_true", help="verify saved output without inference"
    )
    args = parser.parse_args()
    if args.replay:
        replay(json.loads(args.output.read_text()), load_episodes())
        print("Replay verified")
        return 0
    if not args.server or not args.gguf:
        parser.error("inference requires --server and --gguf")
    report = dict(
        frozen=verify_freeze(),
        episodes_sha256=EPISODES_SHA256,
        model=model.MODEL,
        revision=model.REVISION,
        gguf_sha256=model.SHA256,
        sampling=dict(model.SAMPLING, cap=CAP),
        server_flags=runtime.FLAGS,
        budget_seconds=SECONDS,
        rows=[],
        stop_reason="interrupted",
    )
    started = monotonic()
    with args.output.open("x") as output:
        try:
            episodes = load_episodes()
            report["preflight"] = preflight(episodes)
            with args.output.with_suffix(".log").open("x") as log:
                with runtime.local_server(args, report, log) as (socket, ready_by):
                    smoke = generate_model(
                        socket, "Reply with the single letter K.", 19000, CAP, ready_by
                    )
                    if not smoke["tokens"]:
                        raise ValueError("Empty offline smoke decode")
                    report["smoke"] = smoke["content"]
                    report["preflight_seconds"] = round(monotonic() - started, 3)
                    pilot_started = monotonic()
                    try:
                        sample(socket, episodes, report["rows"], pilot_started + SECONDS)
                    finally:
                        report["pilot_seconds"] = round(monotonic() - pilot_started, 3)
            report["stop_reason"] = "complete"
            return 0
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            report["stop_reason"] = f"timeout: {exc}"
            print(report["stop_reason"], flush=True)
            return 1
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            report["stop_reason"] = f"{type(exc).__name__}: {exc}"
            print(report["stop_reason"], flush=True)
            return 1
        finally:
            report["summary"] = summarize(report["rows"])
            try:
                replay(report, load_episodes())
                report["replay_verified"] = True
            finally:
                json.dump(report, output, indent=2)
                output.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
