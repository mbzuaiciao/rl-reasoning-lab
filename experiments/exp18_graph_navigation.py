"""Compare paired graph-navigation rollouts with and without progress feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

if __package__:
    from . import exp14_model_selection as model
    from . import exp15_repair_feasibility as runtime
else:
    import exp14_model_selection as model
    import exp15_repair_feasibility as runtime

EPISODES = Path(__file__).with_name("exp18") / "episodes.json"
EPISODES_SHA256 = "18ce5ddba522e5cfe40b41f7b78f21c5e96a908d0509cb1954dc99b0799d08b4"
CAP = 8
SECONDS = 600
ARMS = ("state_only", "feedback")
PROMPT = (
    "Navigate the directed graph to reach the goal within the step limit. "
    "Each row lists the outgoing neighbors of that node. "
    "Choose one legal next node. Reply with its single uppercase letter only.\n"
)
FEEDBACK = "Previous action: shortest-path distance to the goal {}."


def load_episodes():
    raw = EPISODES.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EPISODES_SHA256:
        raise ValueError("Frozen episode digest mismatch")
    return json.loads(raw)


def goal_distances(graph, goal):
    incoming = {node: [] for node in graph}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            incoming[neighbor].append(node)
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        node = queue.popleft()
        for previous in incoming[node]:
            if previous not in distances:
                distances[previous] = distances[node] + 1
                queue.append(previous)
    return distances


@dataclass
class Navigation:
    graph: dict[str, list[str]]
    start: str
    goal: str
    step_limit: int
    node: str = field(init=False)
    distances: dict[str, int] = field(init=False)
    steps: int = field(init=False, default=0)
    feedback: str | None = field(init=False, default=None)

    def __post_init__(self):
        self.node = self.start
        self.distances = goal_distances(self.graph, self.goal)
        if self.distances.keys() != self.graph.keys():
            raise ValueError("Every node must have a path to the goal")

    @property
    def stop_reason(self):
        if self.node == self.goal:
            return "goal"
        if self.steps == self.step_limit:
            return "step_limit"
        return None

    def observe(self, with_feedback):
        edges = "\n".join(
            f"{node} -> {', '.join(neighbors) or '(none)'}"
            for node, neighbors in self.graph.items()
        )
        prompt = (
            f"{PROMPT}\nGraph:\n{edges}\n"
            f"Goal: {self.goal}\nCurrent node: {self.node}\n"
            f"Legal next nodes: {', '.join(self.graph[self.node])}\n"
            f"Remaining steps: {self.step_limit - self.steps}\n"
        )
        if with_feedback and self.feedback is not None:
            prompt += self.feedback + "\n"
        return prompt

    def step(self, completion):
        if self.stop_reason is not None:
            raise ValueError("Episode already ended")
        previous = self.node
        label = completion.strip()
        action = label if re.fullmatch(r"[A-Z]", label) else None
        valid = action in self.graph[previous]
        if valid:
            self.node = action
        self.steps += 1
        before, after = self.distances[previous], self.distances[self.node]
        change = {-1: "decreased", 0: "stayed the same", 1: "increased"}
        direction = (after > before) - (after < before)
        self.feedback = FEEDBACK.format(change[direction])
        return dict(
            action=action,
            valid=valid,
            before=previous,
            after=self.node,
            distance_before=before,
            distance_after=after,
            feedback=self.feedback,
        )


def environment(episode):
    return Navigation(episode["graph"], episode["start"], episode["goal"], episode["step_limit"])


def forward_distance(graph, start, goal):
    """Independent forward search for the deterministic preflight controls."""
    frontier = {start}
    visited = {start}
    for depth in range(len(graph)):
        if goal in frontier:
            return depth
        frontier = {neighbor for node in frontier for neighbor in graph[node]} - visited
        visited.update(frontier)
    raise ValueError(f"No path from {start} to {goal}")


def preflight(episodes):
    checks = {}
    for episode in episodes:
        oracle = environment(episode)
        graph, goal = oracle.graph, oracle.goal
        expected = {node: forward_distance(graph, node, goal) for node in graph}
        optimum = expected[oracle.start]
        if oracle.distances != expected or not 2 <= optimum <= oracle.step_limit <= 6:
            raise ValueError("Wrong distances, trivial start, or invalid budget")
        if not any(len({expected[neighbor] for neighbor in ns}) > 1 for ns in graph.values()):
            raise ValueError("Graph must offer choices with different goal distances")

        checked = 0
        for node, neighbors in graph.items():
            if node == goal:
                continue
            for action in [*neighbors, "?"]:
                probe = Navigation(graph, node, goal, oracle.step_limit)
                initial = probe.observe(False)
                turn = probe.step(action)
                after = action if action in neighbors else node
                delta = expected[after] - expected[node]
                label = (
                    "decreased" if delta < 0 else "increased" if delta > 0 else "stayed the same"
                )
                state = probe.observe(False)
                expected_line = f"Previous action: shortest-path distance to the goal {label}.\n"
                if (
                    turn["after"] != after
                    or "Previous action:" in initial + state
                    or probe.observe(True) != state + expected_line
                ):
                    raise ValueError("Transition or feedback check failed")
                checked += 1

        while oracle.stop_reason is None:
            oracle.step(min(graph[oracle.node], key=oracle.distances.__getitem__))
        baseline = environment(episode)
        while baseline.stop_reason is None:
            baseline.step(graph[baseline.node][0])
        if oracle.stop_reason != "goal" or oracle.steps != optimum:
            raise ValueError("Oracle failed")
        checks[episode["id"]] = dict(
            oracle_steps=optimum,
            fixed_first_success=baseline.stop_reason == "goal",
            fixed_first_steps=baseline.steps,
            transitions_checked=checked,
        )
    if all(check["fixed_first_success"] for check in checks.values()):
        raise ValueError("Fixed-first baseline saturated the episode set")
    return checks


def sample(socket, episodes, rows, deadline, *, generate=model.generate):
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
                seed = 180000 + 10 * (index + 1) + env.steps + 1
                result = generate(socket, observation, seed, CAP, deadline)
                transition = env.step(result["content"])
                row["turns"].append(
                    dict(
                        observation=observation,
                        seed=seed,
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


def summarize(rows):
    completed = {}
    for row in rows:
        if row["stop_reason"] in {"goal", "step_limit"}:
            completed.setdefault(row["episode"], {})[row["arm"]] = row
    pairs = {name: arms for name, arms in completed.items() if len(arms) == len(ARMS)}
    counts = {arm: dict(successes=0, steps=0, steps_to_goal=[], invalid_actions=0) for arm in ARMS}
    outcomes = {"helpful": [], "harmful": [], "tied": []}
    for name, arms in pairs.items():
        for arm, row in arms.items():
            stats = counts[arm]
            steps = len(row["turns"])
            stats["steps"] += steps
            stats["invalid_actions"] += sum(not turn["valid"] for turn in row["turns"])
            if row["stop_reason"] == "goal":
                stats["successes"] += 1
                stats["steps_to_goal"].append(steps)
        state, feedback = (arms[arm]["stop_reason"] == "goal" for arm in ARMS)
        outcome = (
            "helpful" if feedback and not state else "harmful" if state and not feedback else "tied"
        )
        outcomes[outcome].append(name)
    return dict(
        complete_pairs=len(pairs),
        arms=counts,
        **outcomes,
        go=len(pairs) >= 8 and len(outcomes["helpful"]) >= 3 and len(outcomes["harmful"]) <= 1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(
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
                    smoke = model.generate(
                        socket, "Reply with the single letter K.", 18000, CAP, ready_by
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
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
