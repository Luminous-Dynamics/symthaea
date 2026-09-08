import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LINTER_PATH = ROOT / "scripts" / "hak_conformance_lint.py"

spec = importlib.util.spec_from_file_location("hak_conformance_lint", LINTER_PATH)
assert spec is not None and spec.loader is not None
lint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lint)


def obligation(
    oid="O1",
    *,
    kind="Qualified",
    tier="E2",
    criticality="critical",
    evidence=True,
    policy_sensitive=False,
    policy_lineage=None,
    depends_on=None,
):
    status = {"kind": kind}
    if kind == "Qualified":
        status["evidence_tier"] = tier
    if evidence and kind in lint.EVIDENCE_REQUIRED:
        status["evidence_refs"] = ["ci://example/run/1"]
    return {
        "id": oid,
        "statement": f"statement for {oid}",
        "criticality": criticality,
        "policy_sensitive": policy_sensitive,
        "policy_lineage": policy_lineage,
        "depends_on": depends_on or [],
        "status": status,
    }


def manifest(obligations=None, claims=None):
    return {
        "schema_version": lint.SCHEMA_VERSION,
        "profile_id": "test-profile",
        "domain": "test-domain",
        "implementation_lineage": {
            "repository": "Luminous-Dynamics/example",
            "commit": "a" * 40,
        },
        "policy_lineage": None,
        "obligations": obligations or [obligation()],
        "end_to_end_claims": claims or [],
    }


def test_valid_manifest_passes():
    assert lint.lint_manifest(manifest()) == []


def test_not_applicable_requires_reason():
    o = obligation(kind="NotApplicable", evidence=False)
    errors = lint.lint_manifest(manifest([o]))
    assert any("NotApplicable requires a reason" in error for error in errors)


def test_e5_qualified_requires_evidence_refs():
    o = obligation(kind="Qualified", tier="E5", evidence=False)
    errors = lint.lint_manifest(manifest([o]))
    assert any("requires at least one exact evidence ref" in error for error in errors)


def test_policy_sensitive_requires_policy_lineage():
    o = obligation(policy_sensitive=True)
    errors = lint.lint_manifest(manifest([o]))
    assert any("requires a policy_lineage" in error for error in errors)


def test_qualified_end_to_end_claim_rejects_unknown_critical_obligation():
    o = obligation(kind="Unknown", evidence=False)
    claim = {
        "id": "C1",
        "statement": "end to end",
        "required_evidence_tier": "E2",
        "critical_obligations": ["O1"],
        "status": {
            "kind": "Qualified",
            "evidence_tier": "E2",
            "evidence_refs": ["ci://claim/1"],
        },
    }
    errors = lint.lint_manifest(manifest([o], [claim]))
    assert any("critical obligation 'O1' is not Qualified" in error for error in errors)


def test_test_source_observed_does_not_satisfy_e2_claim():
    o = obligation(kind="TestSourceObserved", evidence=True)
    claim = {
        "id": "C1",
        "statement": "end to end",
        "required_evidence_tier": "E2",
        "critical_obligations": ["O1"],
        "status": {
            "kind": "Qualified",
            "evidence_tier": "E2",
            "evidence_refs": ["ci://claim/1"],
        },
    }
    errors = lint.lint_manifest(manifest([o], [claim]))
    assert any("critical obligation 'O1' is not Qualified" in error for error in errors)


def test_unknown_dependency_fails():
    o = obligation(depends_on=["MISSING"])
    errors = lint.lint_manifest(manifest([o]))
    assert any("unknown dependency 'MISSING'" in error for error in errors)


def test_dependency_cycle_fails():
    a = obligation("A", depends_on=["B"])
    b = obligation("B", depends_on=["A"])
    errors = lint.lint_manifest(manifest([a, b]))
    assert any("dependency cycle" in error for error in errors)


def test_open_finding_requires_reference():
    o = obligation(kind="OpenFinding", evidence=False)
    errors = lint.lint_manifest(manifest([o]))
    assert any("OpenFinding requires a reference" in error for error in errors)


def test_nonqualified_status_cannot_claim_evidence_tier():
    o = obligation(kind="Unknown", evidence=False)
    o["status"]["evidence_tier"] = "E2"
    errors = lint.lint_manifest(manifest([o]))
    assert any("only Qualified status may claim an evidence tier" in error for error in errors)


def test_cli_json_output(tmp_path, capsys):
    p = tmp_path / "profile.json"
    p.write_text(json.dumps(manifest()), encoding="utf-8")
    rc = lint.main([str(p), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["valid"] is True
    assert out["disclaimer"] == "LintPass != HAKQualification"
