import pytest

from experiments import exp18_graph_navigation as exp


def test_frozen_episode_controls():
    episodes = exp.load_episodes()
    assert len(episodes) == 10
    checks = exp.preflight(episodes)
    assert [row["oracle_steps"] for row in checks.values()] == [4, 4, 5, 3, 6, 4, 4, 5, 5, 5]
    assert not any(row["fixed_first_success"] for row in checks.values())


def test_state_and_feedback_have_identical_observations_except_progress():
    graph = {"A": ["B", "C"], "B": ["D"], "C": ["A", "B"], "D": []}
    env = exp.Navigation(graph, "A", "D", 5)
    assert env.observe(False) == env.observe(True)
    for action, node, outcome, valid in [
        ("b", "A", "stayed the same", False),
        ("A -> B", "A", "stayed the same", False),
        ("C", "C", "stayed the same", True),
        (" B\n", "B", "decreased", True),
        ("D", "D", "decreased", True),
    ]:
        turn = env.step(action)
        assert turn["after"] == node
        assert turn["valid"] is valid
        expected = f"Previous action: shortest-path distance to the goal {outcome}."
        assert turn["feedback"] == expected
        assert env.observe(True) == env.observe(False) + expected + "\n"
    assert env.stop_reason == "goal"
    with pytest.raises(ValueError, match="already ended"):
        env.step("D")


def test_distance_can_increase_and_invalid_attempt_can_exhaust_budget():
    graph = {"A": ["D", "B"], "B": ["A"], "D": []}
    env = exp.Navigation(graph, "A", "D", 2)
    assert env.step("B")["feedback"].endswith("increased.")
    turn = env.step("D")
    assert turn["action"] == "D" and not turn["valid"]
    assert env.node == "B" and env.stop_reason == "step_limit"


def test_paired_runner_keeps_completed_pairs_after_interruption():
    episodes = exp.load_episodes()[:2]
    rows = []

    def generate(socket, prompt, seed, cap, deadline):
        if seed == 180022:
            raise TimeoutError("test interruption")
        episode = episodes[(seed - 180011) // 10]
        env = exp.environment(episode)
        node = prompt.split("Current node: ")[1][0]
        action = min(env.graph[node], key=env.distances.__getitem__)
        return dict(
            content=action, tokens=[1], tokens_evaluated=50, tokens_predicted=1, stop_type="eos"
        )

    with pytest.raises(TimeoutError, match="test interruption"):
        exp.sample(None, episodes, rows, float("inf"), generate=generate)
    assert [row["arm"] for row in rows] == ["state_only", "feedback", "feedback"]
    assert rows[0]["turns"][0] == rows[1]["turns"][0]
    assert [turn["seed"] for turn in rows[0]["turns"]] == [180011, 180012, 180013, 180014]
    assert rows[-1]["stop_reason"] is None and len(rows[-1]["turns"]) == 1
    summary = exp.summarize(rows)
    assert summary["complete_pairs"] == 1 and summary["tied"] == ["g01"] and not summary["go"]
    assert summary["arms"]["feedback"] == dict(
        successes=1,
        steps=4,
        steps_to_goal=[4],
        invalid_actions=0,
    )


def test_go_gate_and_helpful_harmful_counts():
    rows = []
    for index in range(8):
        for arm in exp.ARMS:
            success = index < 3 if arm == "feedback" else index == 3
            rows.append(
                dict(
                    episode=str(index),
                    arm=arm,
                    turns=[dict(valid=False)],
                    stop_reason="goal" if success else "step_limit",
                )
            )
    summary = exp.summarize(rows)
    assert summary["go"] and summary["helpful"] == ["0", "1", "2"] and summary["harmful"] == ["3"]
    assert not exp.summarize(rows[:-1])["go"]
    rows[-2]["stop_reason"] = "goal"
    assert not exp.summarize(rows)["go"]
