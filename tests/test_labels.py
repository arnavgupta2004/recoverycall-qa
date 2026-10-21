"""Ground-truth consistency for data/dataset_manifest.json and the dev scripts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))
DEV = {s["id"]: s for s in json.loads((ROOT / "data" / "scripts_dev.json").read_text(encoding="utf-8"))["scenarios"]}
AUTHORITY = "authority_or_legal_action_claim"


def test_every_dev_call_has_a_script_and_vice_versa():
    dev_ids = {c["id"] for c in MANIFEST["calls"] if c["split"] in ("dev", "dev_v2")}
    assert dev_ids == set(DEV)
    assert not [c for c in MANIFEST["calls"] if c["split"] == "held_out"]  # held-out v2 not written yet


def test_spans_are_verbatim_in_agent_turns():
    for c in MANIFEST["calls"]:
        turns = DEV[c["id"]]["turns"]
        for s in c["concern_spans"]:
            assert s["text"] in turns[s["turn"]]["text"], (c["id"], s)
            assert turns[s["turn"]]["speaker"] == "agent", (c["id"], s)


def test_hard_negatives_are_compliant_and_authority_flag_matches_spans():
    for c in MANIFEST["calls"]:
        assert not (c["hard_negative"] and not c["compliant"]), c["id"]
        assert c["authority_mention"] == any(s["type"] == AUTHORITY for s in c["concern_spans"]), c["id"]


def test_dev_v2_types_equal_span_types():
    for c in MANIFEST["calls"]:
        if c["split"] != "dev_v2":
            continue
        span_types = {s["type"] for s in c["concern_spans"]} - ({AUTHORITY} if c["compliant"] else set())
        assert set(c["concern_types"]) == span_types, c["id"]
        assert c["compliant"] == (not c["concern_types"]), c["id"]


def test_017_and_003_labelled_consistently():
    calls = {c["id"]: c for c in MANIFEST["calls"]}
    for cid in ("script_003", "script_017"):
        assert {"intimidation_or_threats", "third_party_contact_threat"} <= set(calls[cid]["concern_types"]), cid
