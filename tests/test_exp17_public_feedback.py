import json
from pathlib import Path

from experiments import exp17_public_feedback as exp


def test_saved_run_matches_frozen_prompts_and_scores():
    path = Path(__file__).resolve().parents[1] / "experiments/results/exp17_public_feedback.json"
    report = json.loads(path.read_text())
    public, hidden, template = exp.pilot.fixtures()
    rows = report["candidates"]
    expected_slots = [
        (name, arm, initial_seed if arm == "initial" else revision_seed)
        for name, initial_seed, revision_seed, *order in exp.PLAN
        for arm in ("initial", *order)
    ]
    assert [(r["task"], r["arm"], r["seed"]) for r in rows] == expected_slots[: len(rows)]
    assert report["sampling"] == dict(exp.model.SAMPLING, cap=exp.CAP)
    assert report["image"] == exp.repair.IMAGE
    assert report["docker_flags"] == exp.repair.FLAGS
    if report["phase"] == "complete":
        assert len(rows) == len(expected_slots)
        assert "pending" not in report
    else:
        assert report["stop_reason"]

    by_slot = {(r["task"], r["arm"]): r for r in rows}
    for row in rows:
        name, arm, raw = row["task"], row["arm"], row["completion"]
        task = public[name]
        source = exp.repair.replacement(raw, name, task["params"])
        assert row["completion_sha256"] == exp.pilot.digest(raw)
        assert row["patch_sha256"] == (exp.pilot.digest(source) if source is not None else None)
        assert row["applicable"] == (source is not None)
        assert 0 < len(row["token_ids"]) <= exp.CAP
        assert row["timings"]["prompt_n"] + exp.CAP <= exp.CONTEXT
        prompt = exp.repair.prompt_for(template, task, name)
        if arm == "initial":
            assert row["acceptance"] is None and row["accepted"] is None
        else:
            initial = by_slot[name, "initial"]
            feedback = exp.public_feedback(initial["public"], task["cases"])
            assert report["feedback"][name] == feedback
            prompt += exp.REVISION.format(
                initial=initial["completion"],
                feedback=feedback if arm == "feedback" else exp.NO_FEEDBACK,
            )
            if row["applicable"] and row["accepted"] is not None:
                outcomes = row["acceptance"]
                assert len(outcomes) == len(hidden[name]["cases"])
                assert row["accepted"] == all(case["passed"] for case in outcomes)
            if not row["applicable"]:
                assert row["accepted"] is False and row["acceptance"] is None
        assert row["prompt"] == prompt


def test_feedback_uses_public_outcomes_and_only_fixed_failure_details():
    rows = [
        {"case": 1, "passed": True, "status": "ok"},
        {"case": 2, "passed": False, "status": "ok"},
        {"case": 3, "passed": False, "status": "exit_1"},
    ]
    cases = [{"expected": "a"}, {"expected": "b"}, {"expected": "c"}]
    assert exp.public_feedback(rows, cases).splitlines() == [
        "Public case 1: passed.",
        "Public case 2: failed. Result differed from expected 'b'.",
        "Public case 3: failed. Sandbox status: exit_1.",
    ]
    assert "inapplicable" in exp.public_feedback(None, cases)
