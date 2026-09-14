import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "qualification_receipt_core_v1.py"
spec = importlib.util.spec_from_file_location("qualification_receipt_core_v1", SCRIPT)
assert spec and spec.loader
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
sys.modules["qualification_receipt_core_v1"] = core

GOLDEN_FRAME_HEX = "260000000000000073796d74686165612e7175616c696669636174696f6e2d726563656970742d636f72652e7631260000000000000073796d74686165612e7175616c696669636174696f6e2d726563656970742d636f72652e763147000000000000007368613235363a3131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313131313147000000000000007368613235363a3232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323247000000000000007368613235363a3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333347000000000000007368613235363a34343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434020000000000000036000000000000006769742d626c6f622d736861313a3535353535353535353535353535353535353535353535353535353535353535353535353535353547000000000000007368613235363a3636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363647000000000000007368613235363a3737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373737373747000000000000007368613235363a38383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838383838270000000000000053656c65637465645265717569726564526563697065735265706f7274656450617373656456311c000000000000004e6f47656e6572696343726f737343757474696e6752756c6573563107000000000000002a00000000000000417474656d7074486973746f72794578636c7564656446726f6d53656d616e7469634964656e7469747912000000000000004e6f43757272656e7441646d697373696f6e15000000000000004e6f44657461636865644174746573746174696f6e1b000000000000004e6f4d657267654f72457865637574696f6e417574686f7269747916000000000000004e6f50726f766964657241757468656e74696369747914000000000000004e6f536369656e746966696356616c696469747918000000000000004e6f547275737465645061737345737461626c6973686564"
GOLDEN_ID = "blake3:10b18b9ab46ae0c75b4cfb322ff6dfed48f7b45541cf3de2a4128812e28cf69d"


def semantic_core(reverse=False):
    recipes = [
        {"recipe_id": "git-blob-sha1:" + "5" * 40, "attempt_subject_id": "sha256:" + "6" * 64},
        {"recipe_id": "sha256:" + "7" * 64, "attempt_subject_id": "sha256:" + "8" * 64},
    ]
    if reverse:
        recipes.reverse()
    return {
        "schema": core.SCHEMA,
        "qualification_subject_id": "sha256:" + "1" * 64,
        "qualification_profile_id": "sha256:" + "2" * 64,
        "input_closure_id": "sha256:" + "3" * 64,
        "qualification_environment_id": "sha256:" + "4" * 64,
        "recipes": recipes,
        "selection_disposition": core.SELECTION_DISPOSITION,
        "cross_cutting_disposition": core.CROSS_CUTTING_DISPOSITION,
        "non_claims": list(core.NON_CLAIM_TAGS),
    }


def test_reference_blake3_matches_official_short_vectors():
    assert core.blake3_256(b"").hex() == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    assert core.blake3_256(b"abc").hex() == "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"


def test_python_reproduces_canonical_golden_frame_and_receipt_id():
    frame = core.frame_core(semantic_core())
    assert len(frame) == 1036
    assert frame.hex() == GOLDEN_FRAME_HEX
    assert core.receipt_id(semantic_core()) == GOLDEN_ID


def test_receipt_core_itself_does_not_claim_trusted_pass():
    value = semantic_core()
    assert value["selection_disposition"] == "SelectedRequiredRecipesReportedPassedV1"
    assert "NoTrustedPassEstablished" in value["non_claims"]


def test_recipe_order_is_canonical():
    assert core.frame_core(semantic_core(False)) == core.frame_core(semantic_core(True))
    assert core.receipt_id(semantic_core(False)) == core.receipt_id(semantic_core(True))


def test_semantic_drift_changes_id():
    first = semantic_core()
    second = semantic_core()
    second["input_closure_id"] = "sha256:" + "9" * 64
    assert core.receipt_id(first) != core.receipt_id(second)


def test_duplicate_recipe_fails_closed():
    value = semantic_core()
    value["recipes"].append(dict(value["recipes"][0]))
    with pytest.raises(core.ReceiptCoreError, match="duplicate recipe"):
        core.frame_core(value)


def _selection_and_gate():
    selection = {
        "schema": core.PASS_SELECTION_SCHEMA,
        "admission_subject_id": "sha256:" + "a" * 64,
        "qualification_subject_id": "sha256:" + "1" * 64,
        "qualification_profile_id": "sha256:" + "2" * 64,
        "input_closure_id": "sha256:" + "3" * 64,
        "qualification_environment_id": "sha256:" + "4" * 64,
        "attempt_history_id": "sha256:" + "b" * 64,
        "selected_recipe_attempts": [{
            "recipe_id": "sha256:" + "7" * 64,
            "attempt_subject_id": "sha256:" + "8" * 64,
            "attempt_registration_id": "sha256:" + "c" * 64,
            "attempt_observation_id": "sha256:" + "d" * 64,
            "evidence_content_ids": ["sha256:" + "e" * 64],
        }],
        "cross_cutting_evidence": [],
        "non_claims": ["occurrence evidence retained elsewhere"],
        "pass_selection_id": "sha256:" + "f" * 64,
    }
    gate = {
        "schema": core.GATE_SCHEMA,
        "qualification_subject_id": selection["qualification_subject_id"],
        "qualification_profile_id": selection["qualification_profile_id"],
        "attempt_history_id": selection["attempt_history_id"],
        "pass_selection_id": selection["pass_selection_id"],
        "support_closure_id": "sha256:" + "0" * 64,
        "required_recipe_ids": ["sha256:" + "7" * 64],
        "disposition": "WitnessRequired",
        "non_claims": ["still not qualification PASS"],
        "receipt_candidate_id": "sha256:" + "9" * 64,
    }
    return selection, gate


def test_projection_requires_witness_required_gate_and_matching_selection():
    selection, gate = _selection_and_gate()
    projected = core.project_core(selection, gate)
    assert projected["qualification_subject_id"] == selection["qualification_subject_id"]
    assert projected["recipes"] == [{"recipe_id": "sha256:" + "7" * 64, "attempt_subject_id": "sha256:" + "8" * 64}]
    assert projected["selection_disposition"] == "SelectedRequiredRecipesReportedPassedV1"
    assert "NoTrustedPassEstablished" in projected["non_claims"]
    assert projected["qualification_receipt_id"].startswith("blake3:")

    bad = dict(gate)
    bad["disposition"] = "Passed"
    with pytest.raises(core.ReceiptCoreError, match="WitnessRequired"):
        core.project_core(selection, bad)


def test_occurrence_ids_do_not_enter_semantic_core():
    selection, gate = _selection_and_gate()
    first = core.project_core(selection, gate)

    selection2 = dict(selection)
    selection2["attempt_history_id"] = "sha256:" + "1" * 64
    selection2["pass_selection_id"] = "sha256:" + "2" * 64
    selection2["selected_recipe_attempts"] = [dict(selection["selected_recipe_attempts"][0])]
    selection2["selected_recipe_attempts"][0]["attempt_registration_id"] = "sha256:" + "3" * 64
    selection2["selected_recipe_attempts"][0]["attempt_observation_id"] = "sha256:" + "4" * 64
    gate2 = dict(gate)
    gate2["attempt_history_id"] = selection2["attempt_history_id"]
    gate2["pass_selection_id"] = selection2["pass_selection_id"]
    second = core.project_core(selection2, gate2)

    assert first["qualification_receipt_id"] == second["qualification_receipt_id"]
