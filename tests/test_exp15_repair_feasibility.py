import pytest

from experiments.exp15_repair_feasibility import digest, fixtures, grade, patch_for, sample_plan


def test_frozen_slots_and_acceptance_cases():
    public, hidden, _ = fixtures()
    assert len(public) == 8 and public.keys() == hidden.keys()
    assert [seed for _, _, seed in sample_plan(public)] == list(range(15001, 15025))
    for name, task in public.items():
        assert len(task["cases"]) == 3 and len(hidden[name]["cases"]) == 4
        assert all(
            diagnostic["args"] != acceptance["args"]
            for diagnostic in task["cases"]
            for acceptance in hidden[name]["cases"]
        )


def test_saved_patch_stays_ungraded_when_docker_disconnects():
    public, hidden, _ = fixtures()
    raw = public["parse_fields"]["buggy"].strip()
    row = dict(
        task="parse_fields",
        sample=1,
        completion=raw,
        completion_sha256=digest(raw),
        patch_sha256=digest(raw),
        applicable=True,
        hidden_success=None,
    )

    class DisconnectedSandbox:
        def cases(self, name, source, cases):
            raise RuntimeError("Docker daemon connection failed")

    with pytest.raises(RuntimeError, match="Docker daemon connection failed"):
        grade(DisconnectedSandbox(), {"candidates": [row]}, public, hidden, float("inf"))
    assert row["hidden_success"] is None
    assert "public_delta" not in row
    row["completion"] += "\n"
    with pytest.raises(ValueError, match="Saved hash mismatch"):
        patch_for(row, public)
