import json
from pathlib import Path

from experiments import exp15_repair_feasibility as pilot
from experiments.exp16_token_cap import CAP

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


def test_frozen_output_cap_comparison():
    baseline = json.loads((RESULTS / "exp15_repair_feasibility.json").read_text())
    treatment = json.loads((RESULTS / "exp16_token_cap.json").read_text())
    for key in (
        "fixtures",
        "model",
        "revision",
        "gguf_sha256",
        "server_flags",
        "image",
        "docker_flags",
        "backend",
        "baselines",
    ):
        assert treatment[key] == baseline[key]
    assert treatment["sampling"] == dict(baseline["sampling"], cap=CAP)
    context_size = int(pilot.FLAGS[pilot.FLAGS.index("--ctx-size") + 1])
    assert max(row["timings"]["prompt_n"] for row in baseline["candidates"]) + CAP < context_size

    public, _, _ = pilot.fixtures()
    rows = treatment["candidates"]
    slots = [(row["task"], row["sample"], row["seed"]) for row in rows]
    assert slots == list(pilot.sample_plan(public))[: len(rows)]
    if treatment["phase"] == "complete":
        assert len(slots) == 24
    else:
        assert treatment["stop_reason"]

    for row in rows:
        assert (pilot.patch_for(row, public) is not None) == row["applicable"]
        assert 0 < len(row["token_ids"]) <= CAP
        if row["stop"] == "limit":
            assert len(row["token_ids"]) == CAP
        if row["applicable"] and row["hidden_success"] is not None:
            score = pilot.repair.score(
                row["public"], row["hidden"], treatment["baselines"][row["task"]]
            )
            assert all(row[key] == value for key, value in score.items())
    graded = [row for row in rows if row["hidden_success"] is not None]
    assert treatment["summary"] == pilot.repair.summarize(graded)
    assert treatment["ungraded_applicable"] == len(rows) - len(graded)
