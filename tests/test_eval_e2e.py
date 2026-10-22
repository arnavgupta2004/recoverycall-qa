"""eval_e2e.py: abort rules and per-mode flag selection."""

import pytest

from eval_e2e import Abort, Guard, mode_types


def test_guard_stops_on_403():
    with pytest.raises(Abort):
        Guard().check("STT", 403, False)


def test_guard_stops_on_two_consecutive_failures_only():
    g = Guard()
    g.check("STT", 503, False)
    g.check("STT", 200, True)          # a success resets the streak
    g.check("LLM", None, False)
    with pytest.raises(Abort):
        g.check("STT", 429, False)


def test_guard_ignores_a_single_400():
    g = Guard()
    g.check("STT", 400, False)
    g.check("STT", 503, False)          # 400 is not a retryable failure, so this is the first in the streak


def test_mode_types_split_hybrid_flags_by_detector():
    res = {"flags": [{"concern_type": "a", "detector": "rules"}, {"concern_type": "b", "detector": "LLM"},
                     {"concern_type": "c", "detector": "both"}]}
    assert mode_types(res, "rules") == {"a", "c"}
    assert mode_types(res, "llm") == {"b", "c"}
    assert mode_types(res, "hybrid") == {"a", "b", "c"}
