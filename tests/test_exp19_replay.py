import copy
import json
from pathlib import Path

import pytest

from experiments import exp19_replay as replay

RAW = Path(__file__).resolve().parents[1] / "experiments/results/exp19_partial_navigation.json"


def test_real_checkpoint_template_replay():
    report = json.loads(RAW.read_text())
    result = replay.verify(report)
    assert result["trajectories"] == 16 and result["summary"]["complete_pairs"] == 8
    assert not result["summary"]["go"]
    assert report["rows"][0]["turns"][0]["observation"].endswith("\n")


@pytest.mark.parametrize(
    "field", ["rendered_prompt", "observation", "completion", "distance_after"]
)
def test_replay_rejects_tampering(field):
    report = copy.deepcopy(json.loads(RAW.read_text()))
    turn = report["rows"][0]["turns"][0]
    turn[field] = turn[field] + ("?" if isinstance(turn[field], str) else 1)
    with pytest.raises(ValueError):
        replay.verify(report)
