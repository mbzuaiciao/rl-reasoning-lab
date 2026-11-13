import json
from itertools import product

import pytest

from experiments.exp13_code_repair import fixtures, prompt_for, replacement, score, summarize


def test_frozen_public_and_hidden_inputs_are_disjoint_and_only_public_is_prompted():
    public, hidden, template = fixtures()
    assert len(public) == 8 and public.keys() == hidden.keys()
    for name, task in public.items():
        prompt = prompt_for(template, task, name)
        assert task["buggy"] in prompt and hidden[name]["reference"] not in prompt
        assert len(task["cases"]) == 3 and len(hidden[name]["cases"]) == 4
        for first, second in product(task["cases"], hidden[name]["cases"]):
            assert first["args"] != second["args"]
        for case in hidden[name]["cases"]:
            call = f"{name}({', '.join(map(repr, case['args']))})"
            assert call not in prompt
        for case in task["cases"]:
            call = f"{name}({', '.join(map(repr, case['args']))})"
            assert f"{call} -> {case['expected']!r}" in prompt


@pytest.mark.parametrize(
    ("text", "applies"),
    [
        ("def f(x):\n    return x", True),
        ("```python\ndef f(x):\n    return x\n```", True),
        ("Here is the patch:\ndef f(x): return x", False),
        ("def f(x): return x\nprint(f(1))", False),
        ("@decorate\ndef f(x): return x", False),
        ("def f(x=call()): return x", False),
        ("def f(y): return y", False),
        ("def f(x):\n    break", False),  # parses as AST, but cannot compile
    ],
)
def test_applicability(text, applies):
    assert (replacement(text, "f", ["x"]) is not None) == applies


def cases(*passes):
    return [{"case": i, "passed": passed} for i, passed in enumerate(passes, 1)]


def test_zero_net_improvement_can_hide_a_regression():
    baseline = {"public": cases(True, False, False), "hidden": cases(False, True)}
    result = score(cases(False, True, False), cases(False, False), baseline)
    assert result["public_delta"] == 0
    assert result["regressions"] == {"public": [1], "hidden": [2]}
    assert not result["hidden_success"]


def test_rankings_and_gate_exclude_invalid_patches_and_include_ties():
    baseline = {"public": cases(True, False, False), "hidden": cases(False)}
    rows = [{"task": "invalid", "applicable": False, "hidden_success": False}]
    for task, good, bad in [
        ("a", (True, True, True), (True, True, False)),
        ("b", (True, True, True), (True, True, True)),
        ("c", (True, True, False), (True, True, True)),
        ("d", (True, True, True), (True, True, False)),
    ]:
        for public, accepted in ((good, True), (bad, False)):
            rows.append(
                dict(task=task, applicable=True, **score(cases(*public), cases(accepted), baseline))
            )
    summary = summarize(json.loads(json.dumps(rows)))
    assert summary["mixed_tasks"] == ["a", "b", "c", "d"]
    assert summary["improving_hidden_failures"] == 4
    assert summary["full_public_hidden_failures"] == 2
    assert summary["pairs"] == {"concordant": 2, "tied": 1, "inverted": 1}
    assert summary["numeric_gate"]
    assert not summarize(rows[:-1])["numeric_gate"]
