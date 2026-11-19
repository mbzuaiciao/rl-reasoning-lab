import copy
import json

import pytest

from experiments import exp19_partial_navigation as exp


def test_frozen_preflight_proves_uncertainty_and_nonsaturated_controls():
    episodes = exp.load_episodes()
    checks = exp.preflight(episodes)
    assert len(checks) == 8
    assert [c["oracle_steps"] for c in checks.values()] == [4, 3] * 4
    assert sum(c["fixed_first_success"] for c in checks.values()) == 3
    for c in checks.values():
        witness = c["information_witness"]
        assert set(witness["optimal_next"]).isdisjoint(witness["counterpart_optimal_next"])
        assert witness["feedback"] != witness["counterpart_feedback"]


def test_partial_observations_and_invalid_history():
    episode = exp.load_episodes()[0]
    env = exp.environment(episode)
    assert env.observe(True) == env.observe(False)
    assert "Graph:" not in env.observe(False)
    # Hidden edge edits leave both current observations and visible history unchanged.
    other = copy.deepcopy(episode)
    hidden = next(n for n in other["graph"] if n not in {env.node, env.goal})
    other["graph"][hidden] = [env.goal]
    assert exp.environment(other).observe(False) == env.observe(False)
    env.step("?")
    assert env.steps == 1 and env.node == env.start
    assert json.dumps([env.start, env.graph[env.start], "?", False, env.start]) in env.observe(
        False
    )
    assert "distance" not in env.observe(False)
    assert env.observe(True) == env.observe(False) + env.feedback + "\n"


def oracle_generate(socket, prompt, seed, cap, deadline):
    episode = exp.load_episodes()[(seed - 190011) // 10]
    env = exp.environment(episode)
    node = prompt.split("Current node: ")[1][0]
    action = min(env.graph[node], key=env.distances.__getitem__)
    return dict(
        content=action,
        tokens=[1],
        tokens_evaluated=80,
        tokens_predicted=1,
        stop_type="eos",
        rendered_prompt="USER\n" + prompt + "ASSISTANT\n",
    )


def test_paired_sample_replay_and_tampering():
    episodes = exp.load_episodes()
    rows = []
    exp.sample(None, episodes, rows, float("inf"), generate=oracle_generate)
    report = dict(frozen=exp.verify_freeze(), rows=rows, summary=exp.summarize(rows))
    exp.replay(report, episodes)
    assert report["summary"]["complete_pairs"] == 8
    assert not report["summary"]["go"]
    assert [r["arm"] for r in rows[:4]] == ["state_only", "feedback", "feedback", "state_only"]
    assert rows[0]["turns"][0] == rows[1]["turns"][0]
    report["rows"][0]["turns"][0]["distance_after"] += 1
    with pytest.raises(ValueError, match="Transition replay"):
        exp.replay(report, episodes)


def test_interruption_preserves_complete_pairs_and_partial_row():
    rows = []

    def interrupted(*args):
        if args[2] == 190022:
            raise TimeoutError("bounded")
        return oracle_generate(*args)

    with pytest.raises(TimeoutError):
        exp.sample(None, exp.load_episodes(), rows, float("inf"), generate=interrupted)
    assert exp.summarize(rows)["complete_pairs"] == 1
    assert len(rows[-1]["turns"]) == 1 and rows[-1]["stop_reason"] is None
    exp.replay(
        dict(frozen=exp.verify_freeze(), rows=rows, summary=exp.summarize(rows)),
        exp.load_episodes(),
    )


def test_recovery_requires_return_to_pre_wrong_turn_distance():
    turns = [
        dict(valid=True, distance_before=a, distance_after=b)
        for a, b in [(3, 4), (4, 5), (5, 4), (4, 3), (3, 2)]
    ]
    assert exp.recovery(turns) == dict(
        wrong_turns=2, recovered_turns=2, goal_after_wrong_turn=False
    )
    assert exp.recovery(turns[:3])["recovered_turns"] == 1
