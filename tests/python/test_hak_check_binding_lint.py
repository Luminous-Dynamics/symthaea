import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "hak_check_binding_lint", SCRIPTS / "hak_check_binding_lint.py"
)
assert spec and spec.loader
hak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hak)

fixture_spec = importlib.util.spec_from_file_location(
    "hak_check_evidence_fixtures", ROOT / "tests/python/test_hak_check_evidence_lint.py"
)
assert fixture_spec and fixture_spec.loader
check_fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(check_fixtures)

HAK010_PLAN_PATH = ROOT / "docs/architecture/hak/plans/hak010-precommitted-check-binding-e5-v1.plan.json"
HAK010_POLICY_PATH = ROOT / "docs/architecture/hak/policies/hak010-precommitted-check-binding-policy-v1.json"


def load_hak010_plan():
    return json.loads(HAK010_PLAN_PATH.read_text())


def load_hak010_policy():
    return json.loads(HAK010_POLICY_PATH.read_text())


def synthetic_policy(plan):
    job = "HAK Evidence Linter"
    bindings = []
    counter = 1
    for item in plan["required_checks"]:
        bindings.append({
            "obligation_kind": "RequiredCheck",
            "obligation_id": item["check_id"],
            "job_name": job,
            "step_name": "synthetic obligation step",
            "step_number": counter,
            "accepted_job_conclusions": ["success"],
            "accepted_step_conclusions": ["success"],
        })
        counter += 1
    for case in plan["required_negative_cases"]:
        bindings.append({
            "obligation_kind": "NegativeCase",
            "obligation_id": case,
            "job_name": job,
            "step_name": "synthetic obligation step",
            "step_number": counter,
            "accepted_job_conclusions": ["success"],
            "accepted_step_conclusions": ["success"],
        })
        counter += 1
    policy = {
        "schema_version": "hak.check-evidence-binding-policy.v1",
        "policy_id": "synthetic-binding-policy",
        "qualification_plan": {
            "plan_id": plan["plan_id"],
            "plan_digest": hak.interpretation.compute_plan_digest(plan),
        },
        "workflow_path": plan["scope"]["workflow_path"],
        "bindings": bindings,
    }
    policy["policy_digest"] = hak.compute_binding_policy_digest(policy)
    return policy


def binding_for(policy, kind, ident):
    return next(
        item for item in policy["bindings"]
        if item["obligation_kind"] == kind and item["obligation_id"] == ident
    )


def test_static_hak010_policy_is_valid_and_plan_bound():
    hak.validate_binding_policy(load_hak010_plan(), load_hak010_policy())


def test_plan_id_mismatch_rejected():
    plan = check_fixtures.fixtures.load_plan()
    policy = synthetic_policy(plan)
    policy["qualification_plan"]["plan_id"] = "other-plan"
    policy["policy_digest"] = hak.compute_binding_policy_digest(policy)
    with pytest.raises(hak.CheckBindingLintError):
        hak.validate_binding_policy(plan, policy)


def test_plan_digest_mismatch_rejected():
    plan = check_fixtures.fixtures.load_plan()
    policy = synthetic_policy(plan)
    policy["qualification_plan"]["plan_digest"] = "sha256:" + "0" * 64
    policy["policy_digest"] = hak.compute_binding_policy_digest(policy)
    with pytest.raises(hak.CheckBindingLintError):
        hak.validate_binding_policy(plan, policy)


def test_workflow_mismatch_rejected():
    plan = check_fixtures.fixtures.load_plan()
    policy = synthetic_policy(plan)
    policy["workflow_path"] = ".github/workflows/other.yml"
    policy["policy_digest"] = hak.compute_binding_policy_digest(policy)
    with pytest.raises(hak.CheckBindingLintError):
        hak.validate_binding_policy(plan, policy)


def test_missing_required_obligation_rejected():
    plan = check_fixtures.fixtures.load_plan()
    policy = synthetic_policy(plan)
    policy["bindings"].pop()
    policy["policy_digest"] = hak.compute_binding_policy_digest(policy)
    with pytest.raises(hak.CheckBindingLintError):
        hak.validate_binding_policy(plan, policy)


def test_duplicate_obligation_rejected():
    plan = check_fixtures.fixtures.load_plan()
    policy = synthetic_policy(plan)
    policy["bindings"].append(deepcopy(policy["bindings"][0]))
    policy["policy_digest"] = hak.compute_binding_policy_digest(policy)
    with pytest.raises(hak.CheckBindingLintError):
        hak.validate_binding_policy(plan, policy)


def test_unknown_obligation_rejected():
    plan = check_fixtures.fixtures.load_plan()
    policy = synthetic_policy(plan)
    policy["bindings"][0]["obligation_id"] = "not-in-plan"
    policy["policy_digest"] = hak.compute_binding_policy_digest(policy)
    with pytest.raises(hak.CheckBindingLintError):
        hak.validate_binding_policy(plan, policy)


def setup_provider_record():
    plan = check_fixtures.fixtures.load_plan()
    receipt = check_fixtures.fixtures.receipt(plan)
    record = check_fixtures.record(plan, receipt)
    policy = synthetic_policy(plan)
    return plan, receipt, record, policy


def validate_record(plan, receipt, record, policy):
    hak.validate_provider_evidence_against_binding_policy(
        plan, receipt, record, policy,
        plan_repo_path=check_fixtures.fixtures.PLAN_REPO_PATH,
    )


def test_provider_evidence_matches_precommitted_selector():
    plan, receipt, record, policy = setup_provider_record()
    validate_record(plan, receipt, record, policy)


def test_posthoc_job_name_selection_rejected():
    plan, receipt, record, policy = setup_provider_record()
    record["provider_binding"]["job_name"] = "other successful job"
    record["evidence_digest"] = hak.check_evidence.compute_check_evidence_digest(record)
    with pytest.raises((hak.CheckBindingLintError, hak.check_evidence.CheckEvidenceLintError)):
        validate_record(plan, receipt, record, policy)


def test_posthoc_step_name_selection_rejected():
    plan, receipt, record, policy = setup_provider_record()
    record["provider_binding"]["step_name"] = "different successful step"
    record["evidence_digest"] = hak.check_evidence.compute_check_evidence_digest(record)
    with pytest.raises(hak.CheckBindingLintError):
        validate_record(plan, receipt, record, policy)


def test_posthoc_step_number_selection_rejected():
    plan, receipt, record, policy = setup_provider_record()
    record["provider_binding"]["step_number"] = 99
    record["evidence_digest"] = hak.check_evidence.compute_check_evidence_digest(record)
    with pytest.raises(hak.CheckBindingLintError):
        validate_record(plan, receipt, record, policy)


def test_disallowed_job_conclusion_rejected():
    plan, receipt, record, policy = setup_provider_record()
    binding = binding_for(policy, "RequiredCheck", plan["required_checks"][0]["check_id"])
    binding["accepted_job_conclusions"] = ["failure"]
    policy["policy_digest"] = hak.compute_binding_policy_digest(policy)
    with pytest.raises(hak.CheckBindingLintError):
        validate_record(plan, receipt, record, policy)


def test_disallowed_step_conclusion_rejected():
    plan, receipt, record, policy = setup_provider_record()
    binding = binding_for(policy, "RequiredCheck", plan["required_checks"][0]["check_id"])
    binding["accepted_step_conclusions"] = ["failure"]
    policy["policy_digest"] = hak.compute_binding_policy_digest(policy)
    with pytest.raises(hak.CheckBindingLintError):
        validate_record(plan, receipt, record, policy)


def test_policy_digest_tampering_rejected():
    plan = check_fixtures.fixtures.load_plan()
    policy = synthetic_policy(plan)
    policy["bindings"][0]["step_name"] = "tampered after materialization"
    with pytest.raises(hak.CheckBindingLintError):
        hak.validate_binding_policy(plan, policy)


def test_strict_conformance_rejects_provider_evidence_not_matching_precommit():
    plan = check_fixtures.fixtures.load_plan()
    receipt = check_fixtures.fixtures.receipt(plan)
    conformance = check_fixtures.fixtures.conformance(plan, receipt)
    records = check_fixtures.evidence_for_all_passed_obligations(plan, receipt, conformance)
    policy = synthetic_policy(plan)
    policy["bindings"][0]["step_name"] = "precommitted different step"
    policy["policy_digest"] = hak.compute_binding_policy_digest(policy)
    with pytest.raises(hak.CheckBindingLintError):
        hak.validate_strict_conformance_with_precommitted_bindings(
            plan, receipt, conformance, records, policy,
            plan_repo_path=check_fixtures.fixtures.PLAN_REPO_PATH,
        )


def test_cli_validates_static_policy():
    rc = hak.main([
        "--plan", str(HAK010_PLAN_PATH),
        "--policy", str(HAK010_POLICY_PATH),
    ])
    assert rc == 0
