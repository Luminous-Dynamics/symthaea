import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("verify_vart_world_creative_001.py")
spec = importlib.util.spec_from_file_location("vart_verify", SCRIPT)
v = importlib.util.module_from_spec(spec)
sys.modules["vart_verify"] = v
assert spec.loader is not None
spec.loader.exec_module(v)


def dump(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path) -> None:
    analysis_sha = dump(
        root / "analysis_contract.json", {"schema": "analysis-v1", "cluster_unit": "world"}
    )
    metric_sha = dump(
        root / "metric_definitions.json",
        {"schema": "metrics-v1", "metrics": [{"id": "goal", "direction": "higher_is_better"}]},
    )
    trial_ids = []

    for policy in ["full_symthaea", "random_valid", "heuristic"]:
        trial_id = f"blockA:{policy}"
        trial_ids.append(trial_id)
        d = root / "trials" / trial_id
        d.mkdir(parents=True, exist_ok=True)

        experience_sha = dump(d / "experience.json", {"kind": "experience"})
        hypothesis_sha = dump(
            d / "hypothesis.json", {"prospective": True, "predicted": {"goal": 0.1}}
        )
        p0 = dump(d / "candidate0.json", {"candidate_id": "c0", "edit": {"x": 1}})
        p1 = dump(d / "candidate1.json", {"candidate_id": "c1", "edit": {"x": 2}})
        p2 = dump(d / "candidate2.json", {"candidate_id": "c2", "edit": {"x": 3}})
        candidate_set = {
            "schema": "candidate-set-v1",
            "candidates": [
                {"proposal_sha256": p0, "physically_admitted": True},
                {"proposal_sha256": p1, "physically_admitted": True},
                {"proposal_sha256": p2, "physically_admitted": False},
            ],
        }
        candidate_set_sha = dump(d / "candidate_set.json", candidate_set)

        if policy == "random_valid":
            selection_index, counter, digest_hex = v.sha256_counter_draw(
                42, "blockA", candidate_set_sha, 2
            )
        else:
            selection_index = 0 if policy == "full_symthaea" else 1
        selected_sha = [p0, p1][selection_index]
        selected_file = ["candidate0.json", "candidate1.json"][selection_index]

        receipt_sha = dump(
            d / "receipt.json",
            {
                "selected_proposal_sha256": selected_sha,
                "world_version_before": "w0",
                "world_version_after": "w1",
            },
        )
        revisit_sha = dump(
            d / "revisit.json",
            {"world_version": "w1", "provenance_domain": "digital_committed"},
        )
        outcome_sha = dump(d / "outcome.json", {"actual": {"goal": 0.2}})

        files = {
            "experience_episode": "experience.json",
            "revision_hypothesis": "hypothesis.json",
            "candidate_set": "candidate_set.json",
            "selected_proposal": selected_file,
            "applied_receipt": "receipt.json",
            "revisit_observation": "revisit.json",
            "revision_outcome": "outcome.json",
        }
        draw_sha = None
        if policy == "random_valid":
            draw_sha = dump(
                d / "draw.json",
                {
                    "algorithm": "sha256-counter-v1",
                    "seed": 42,
                    "paired_block_id": "blockA",
                    "candidate_set_sha256": candidate_set_sha,
                    "admissible_candidate_count": 2,
                    "counter": counter,
                    "accepted_digest_sha256": digest_hex,
                    "selected_index": selection_index,
                },
            )
            files["random_draw_receipt"] = "draw.json"

        evidence_sha = dump(
            d / "evidence_index.json",
            {
                "trial_id": trial_id,
                "files": files,
                "timestamps_ns": {
                    "hypothesis_closed": 10,
                    "selection_closed": 20,
                    "applied_receipt": 30,
                    "revisit_closed": 40,
                    "outcome_closed": 50,
                },
                "cross_policy_outcome_observed_before_selection": False,
                "prospective_exclusion_reason_classes": [],
            },
        )

        dump(
            d / "manifest.json",
            {
                "schema": "symthaea.vart-world-creative-001.trial-manifest.v1",
                "experiment_id": "VART-WORLD-CREATIVE-001",
                "campaign": "confirmatory_longitudinal",
                "trial_id": trial_id,
                "paired_block_id": "blockA",
                "policy": policy,
                "policy_sha256": "a" * 64,
                "world_fixture_sha256": "b" * 64,
                "seed": 42,
                "revision_index": 0,
                "world_version_before": "w0",
                "experience_episode_sha256": experience_sha,
                "revision_hypothesis_sha256": hypothesis_sha,
                "candidate_set_sha256": candidate_set_sha,
                "admissible_candidate_count": 2,
                "selected_proposal_sha256": selected_sha,
                "selection_index": selection_index,
                "random_draw_receipt_sha256": draw_sha,
                "applied_receipt_sha256": receipt_sha,
                "world_version_after": "w1",
                "revisit_observation_sha256": revisit_sha,
                "revision_outcome_sha256": outcome_sha,
                "included_in_confirmatory_analysis": True,
                "exclusion_reason": None,
                "trial_state": "complete",
                "abort_stage": None,
                "integrity_violations": [],
                "metric_definition_set_sha256": metric_sha,
                "analysis_contract_sha256": analysis_sha,
                "evidence_bundle_sha256": evidence_sha,
            },
        )

    inventory_sha = dump(
        root / "trial_inventory.json",
        {"trial_ids": trial_ids, "expected_trial_count": len(trial_ids)},
    )
    dump(
        root / "confirmatory_freeze.json",
        {
            "analysis_contract_sha256": analysis_sha,
            "metric_definition_set_sha256": metric_sha,
            "trial_inventory_sha256": inventory_sha,
            "forbidden_primary_aggregates": ["world_quality", "creative_score"],
        },
    )
    dump(root / "primary_results.json", {"channels": {"goal": {"full": 0.2}}})


def clone(src: Path) -> Path:
    dst = Path(tempfile.mkdtemp()) / "bundle"
    shutil.copytree(src, dst)
    return dst


def expect_reject(bundle: Path, code: str) -> None:
    try:
        v.verify_bundle(bundle)
    except v.Reject as exc:
        assert exc.code == code, (exc.code, code)
        return
    raise AssertionError(f"expected {code}")


with tempfile.TemporaryDirectory() as td:
    base = Path(td) / "base"
    base.mkdir()
    build(base)
    assert v.verify_bundle(base)["verdict"] == "ACCEPT"

    # N2 — paired candidate-set substitution while the mutated trial remains internally coherent.
    b = clone(base)
    d = b / "trials" / "blockA:heuristic"
    cset = json.loads((d / "candidate_set.json").read_text())
    cset["candidates"].append({"proposal_sha256": "f" * 64, "physically_admitted": False})
    csha = dump(d / "candidate_set.json", cset)
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["candidate_set_sha256"] = csha
    dump(d / "manifest.json", manifest)
    expect_reject(b, "PAIRED_CANDIDATE_SET_MISMATCH")

    # N3 — coherent receipt tamper that no longer matches sha256-counter-v1.
    b = clone(base)
    d = b / "trials" / "blockA:random_valid"
    draw = json.loads((d / "draw.json").read_text())
    draw["selected_index"] = (draw["selected_index"] + 1) % 2
    draw_sha = dump(d / "draw.json", draw)
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["random_draw_receipt_sha256"] = draw_sha
    dump(d / "manifest.json", manifest)
    expect_reject(b, "RANDOM_VALID_DRAW_MISMATCH")

    # N12 — forbidden scalar aggregate.
    b = clone(base)
    dump(b / "primary_results.json", {"world_quality": 0.9})
    expect_reject(b, "FORBIDDEN_AGGREGATE")

    # N10 — selective omission of a preregistered trial.
    b = clone(base)
    shutil.rmtree(b / "trials" / "blockA:heuristic")
    expect_reject(b, "PREREGISTERED_TRIAL_MISSING")

print("PASS: VART verifier synthetic acceptance + N2/N3/N10/N12")
