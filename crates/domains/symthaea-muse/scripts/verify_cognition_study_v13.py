#!/usr/bin/env python3
"""Independent standard-library verifier for V13 replication/stewardship releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
Z95 = 1.959963984540054

FILES = {
    "source_final_release": "confirmatory_final_release.json",
    "protocol": "replication_protocol.json",
    "registry": "replication_site_registry.json",
    "packages": "replication_packages.json",
    "executions": "replication_executions.json",
    "synthesis": "replication_synthesis.json",
    "orchestration": "replication_orchestration.json",
    "charter": "stewardship_charter.json",
    "archive": "research_archive.json",
    "promotion": "research_release_promotion.json",
    "release": "stewardship_release.json",
}

SELF_FIELDS = {
    "source_final_release": "bundle_sha256",
    "protocol": "protocol_sha256",
    "registry": "registry_sha256",
    "synthesis": "synthesis_sha256",
    "orchestration": "log_sha256",
    "charter": "charter_sha256",
    "archive": "archive_sha256",
    "promotion": "promotion_sha256",
    "release": "bundle_sha256",
}

REQUIRED_PACKAGE_ROLES = {
    "SourceFinalRelease",
    "SourceCodeSnapshot",
    "FrozenReplicationProtocol",
    "AnalysisPlan",
    "ArtifactGenerationPlan",
    "ParticipantRunnerSource",
    "EnvironmentLock",
    "PublicStudyMaterials",
    "SyntheticDryRun",
}
FORBIDDEN_PACKAGE_ROLES = {
    "OriginalParticipantEvidence",
    "OriginalUnblindedDataset",
    "OriginalBlindingCodebook",
    "OriginalRandomizationKey",
}
REQUIRED_STEWARDSHIP_ROLES = {
    "ReleaseMaintainer",
    "ReproducibilityCustodian",
    "ArchiveCustodian",
    "SecurityContact",
    "ParticipantProtectionOfficer",
    "IndependentMethodsReviewer",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_self(label: str, value: dict[str, Any], field: str, errors: list[str]) -> None:
    observed = value.get(field)
    if not is_sha256(observed):
        errors.append(f"{label}: invalid {field}")
        return
    payload = dict(value)
    payload.pop(field, None)
    if digest(payload) != observed:
        errors.append(f"{label}: {field} mismatch")


def package_root(packages: list[dict[str, Any]]) -> str:
    bindings = sorted((item.get("site_id"), item.get("package_sha256")) for item in packages)
    return digest(bindings)


def execution_root(records: list[dict[str, Any]]) -> str:
    bindings = sorted((item.get("site_id"), item.get("record_sha256")) for item in records)
    return digest(bindings)


def verify_protocol(
    source: dict[str, Any], protocol: dict[str, Any], errors: list[str]
) -> None:
    if protocol.get("protocol_version") != "symthaea-muse-replication-protocol-v1":
        errors.append("protocol: wrong version")
    if protocol.get("source_final_release_sha256") != source.get("bundle_sha256"):
        errors.append("protocol: source final release mismatch")
    if protocol.get("source_study_id") != source.get("study_id"):
        errors.append("protocol: source study mismatch")
    if protocol.get("required_site_count", 0) < 2:
        errors.append("protocol: fewer than two sites")
    if protocol.get("minimum_independent_organizations", 0) < 2:
        errors.append("protocol: fewer than two independent organizations")
    endpoint = protocol.get("primary_endpoint", {})
    if endpoint.get("confidence_level") != 0.95:
        errors.append("protocol: V13 requires a frozen 95% confidence level")
    if endpoint.get("alpha", 1.0) <= 0 or endpoint.get("alpha", 1.0) > 0.05:
        errors.append("protocol: invalid alpha")


def verify_registry(
    protocol: dict[str, Any], registry: dict[str, Any], errors: list[str]
) -> set[str]:
    if registry.get("protocol_sha256") != protocol.get("protocol_sha256"):
        errors.append("registry: protocol mismatch")
    active: set[str] = set()
    organizations: set[str] = set()
    all_sites: set[str] = set()
    for site in registry.get("sites", []):
        site_id = site.get("site_id")
        if site_id in all_sites:
            errors.append(f"registry: duplicate site {site_id}")
        all_sites.add(site_id)
        if site.get("site_status") == "Registered":
            active.add(site_id)
            org = site.get("organization_id")
            if org in organizations:
                errors.append(f"registry: duplicate active organization {org}")
            organizations.add(org)
        if not site.get("independent_of_source_authors"):
            errors.append(f"registry: site {site_id} is not independent")
        roles = [
            site.get("principal_investigator_id"),
            site.get("data_custodian_id"),
            site.get("analyst_id"),
        ]
        if len(set(roles)) != len(roles):
            errors.append(f"registry: site {site_id} has colliding roles")
    if len(active) < protocol.get("required_site_count", 0):
        errors.append("registry: insufficient active sites")
    if len(organizations) < protocol.get("minimum_independent_organizations", 0):
        errors.append("registry: insufficient independent organizations")
    return active


def verify_packages(
    protocol: dict[str, Any],
    registry: dict[str, Any],
    packages: list[dict[str, Any]],
    active_sites: set[str],
    errors: list[str],
) -> None:
    seen: set[str] = set()
    for package in packages:
        verify_self(f"package:{package.get('site_id')}", package, "package_sha256", errors)
        site_id = package.get("site_id")
        if site_id in seen:
            errors.append(f"packages: duplicate site {site_id}")
        seen.add(site_id)
        if site_id not in active_sites:
            errors.append(f"packages: inactive or unknown site {site_id}")
        if package.get("protocol_sha256") != protocol.get("protocol_sha256"):
            errors.append(f"packages: protocol mismatch for {site_id}")
        if package.get("site_registry_sha256") != registry.get("registry_sha256"):
            errors.append(f"packages: registry mismatch for {site_id}")
        roles = [entry.get("role") for entry in package.get("entries", [])]
        if len(roles) != len(set(roles)):
            errors.append(f"packages: duplicate role for {site_id}")
        missing = REQUIRED_PACKAGE_ROLES - set(roles)
        if missing:
            errors.append(f"packages: {site_id} missing roles {sorted(missing)}")
        forbidden = FORBIDDEN_PACKAGE_ROLES & set(roles)
        if forbidden:
            errors.append(f"packages: {site_id} includes forbidden roles {sorted(forbidden)}")
    if seen != active_sites:
        errors.append("packages: active-site coverage mismatch")


def derive_site_conclusion(protocol: dict[str, Any], record: dict[str, Any]) -> str:
    if (
        any(item.get("material_to_primary_claim") for item in record.get("deviations", []))
        or not record.get("all_frozen_commands_succeeded")
        or not record.get("collection_blinded_until_close")
        or not record.get("source_outcomes_withheld_until_close")
    ):
        return "DescriptiveOnly"
    result = record.get("primary_result")
    if not isinstance(result, dict):
        return "NonEstimable"
    try:
        estimate = float(result["estimate"])
        lower = float(result["confidence_lower"])
        upper = float(result["confidence_upper"])
        se = float(result["standard_error"])
    except (KeyError, TypeError, ValueError):
        return "NonEstimable"
    if not all(math.isfinite(value) for value in (estimate, lower, upper, se)) or se <= 0:
        return "NonEstimable"
    if (
        int(result.get("participant_count", 0)) < int(protocol.get("participant_target_per_site", 0))
        or int(result.get("family_count", 0)) < int(protocol.get("family_target_per_site", 0))
    ):
        return "DescriptiveOnly"
    margin = float(protocol["primary_endpoint"]["practical_margin"])
    if protocol["primary_endpoint"]["favorable_direction"] == "Higher":
        if lower >= margin:
            return "SupportsReplication"
        if upper < margin:
            return "DoesNotSupportReplication"
    else:
        threshold = -margin
        if upper <= threshold:
            return "SupportsReplication"
        if lower > threshold:
            return "DoesNotSupportReplication"
    return "Inconclusive"


def verify_executions(
    protocol: dict[str, Any],
    registry: dict[str, Any],
    packages: list[dict[str, Any]],
    records: list[dict[str, Any]],
    active_sites: set[str],
    errors: list[str],
) -> None:
    package_by_site = {item.get("site_id"): item for item in packages}
    seen: set[str] = set()
    for record in records:
        site_id = record.get("site_id")
        verify_self(f"execution:{site_id}", record, "record_sha256", errors)
        if site_id in seen:
            errors.append(f"executions: duplicate site {site_id}")
        seen.add(site_id)
        if site_id not in active_sites:
            errors.append(f"executions: inactive or unknown site {site_id}")
        package = package_by_site.get(site_id, {})
        if record.get("site_package_sha256") != package.get("package_sha256"):
            errors.append(f"executions: package mismatch for {site_id}")
        if record.get("protocol_sha256") != protocol.get("protocol_sha256"):
            errors.append(f"executions: protocol mismatch for {site_id}")
        if record.get("site_registry_sha256") != registry.get("registry_sha256"):
            errors.append(f"executions: registry mismatch for {site_id}")
        expected = derive_site_conclusion(protocol, record)
        if record.get("conclusion") != expected:
            errors.append(f"executions: conclusion mismatch for {site_id}")
    if seen != active_sites:
        errors.append("executions: active-site coverage mismatch")


def compute_meta(
    source: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    eligible = [
        record
        for record in records
        if record.get("conclusion")
        in {"SupportsReplication", "DoesNotSupportReplication", "Inconclusive"}
        and isinstance(record.get("primary_result"), dict)
    ]
    if not eligible:
        return None
    values: list[tuple[float, float, int, int]] = []
    for record in eligible:
        result = record["primary_result"]
        values.append(
            (
                float(result["estimate"]),
                float(result["standard_error"]),
                int(result["participant_count"]),
                int(result["family_count"]),
            )
        )
    weights = [1.0 / (se * se) for _, se, _, _ in values]
    sum_w = sum(weights)
    fixed = sum(value[0] * weight for value, weight in zip(values, weights)) / sum_w
    q = sum(weight * (value[0] - fixed) ** 2 for value, weight in zip(values, weights))
    df = max(len(values) - 1, 0)
    c = sum_w - sum(weight * weight for weight in weights) / sum_w
    tau2 = max((q - df) / c, 0.0) if c > 0 else 0.0
    random_weights = [1.0 / (se * se + tau2) for _, se, _, _ in values]
    random_sum = sum(random_weights)
    estimate = sum(value[0] * weight for value, weight in zip(values, random_weights)) / random_sum
    se = math.sqrt(1.0 / random_sum)
    i2 = max((q - df) / q, 0.0) * 100.0 if q > 0 else 0.0
    source_estimate = float(source["estimate"])
    ratio = estimate / source_estimate if abs(source_estimate) > 2.220446049250313e-16 else None
    return {
        "site_count": len(values),
        "participant_count": sum(value[2] for value in values),
        "family_count": sum(value[3] for value in values),
        "fixed_effect_estimate": fixed,
        "random_effect_estimate": estimate,
        "random_effect_standard_error": se,
        "confidence_lower": estimate - Z95 * se,
        "confidence_upper": estimate + Z95 * se,
        "cochran_q": q,
        "tau_squared": tau2,
        "i_squared_percent": i2,
        "source_attenuation_ratio": ratio,
        "direction_concordant_with_source": math.copysign(1.0, estimate)
        == math.copysign(1.0, source_estimate),
    }


def near(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def compare_meta(expected: dict[str, Any] | None, found: Any, errors: list[str]) -> None:
    if expected is None:
        if found is not None:
            errors.append("synthesis: unexpected meta-analysis")
        return
    if not isinstance(found, dict):
        errors.append("synthesis: missing meta-analysis")
        return
    for field, expected_value in expected.items():
        found_value = found.get(field)
        if isinstance(expected_value, (float, type(None))):
            if not near(expected_value, found_value):
                errors.append(f"synthesis: {field} mismatch")
        elif found_value != expected_value:
            errors.append(f"synthesis: {field} mismatch")


def derive_synthesis_conclusion(
    protocol: dict[str, Any], records: list[dict[str, Any]], meta: dict[str, Any] | None
) -> str:
    if len(records) < int(protocol.get("required_site_count", 0)):
        return "InsufficientEvidence"
    if any(record.get("conclusion") in {"DescriptiveOnly", "NonEstimable"} for record in records):
        return "DescriptiveOnly"
    if meta is None:
        return "InsufficientEvidence"
    support = sum(record.get("conclusion") == "SupportsReplication" for record in records)
    margin = float(protocol["primary_endpoint"]["practical_margin"])
    if protocol["primary_endpoint"]["favorable_direction"] == "Higher":
        if meta["confidence_lower"] >= margin and support >= 2:
            return "IndependentlyReplicated"
        if meta["confidence_upper"] < margin and support == 0:
            return "DidNotReplicate"
    else:
        threshold = -margin
        if meta["confidence_upper"] <= threshold and support >= 2:
            return "IndependentlyReplicated"
        if meta["confidence_lower"] > threshold and support == 0:
            return "DidNotReplicate"
    return "MixedEvidence"


def verify_synthesis(
    protocol: dict[str, Any],
    registry: dict[str, Any],
    records: list[dict[str, Any]],
    synthesis: dict[str, Any],
    errors: list[str],
) -> None:
    if synthesis.get("protocol_sha256") != protocol.get("protocol_sha256"):
        errors.append("synthesis: protocol mismatch")
    if synthesis.get("site_registry_sha256") != registry.get("registry_sha256"):
        errors.append("synthesis: registry mismatch")
    ordered = sorted(records, key=lambda item: item.get("site_id"))
    expected_digests = [item.get("record_sha256") for item in ordered]
    if synthesis.get("site_execution_sha256") != expected_digests:
        errors.append("synthesis: execution digest list mismatch")
    meta = compute_meta(synthesis.get("source_result", {}), ordered)
    compare_meta(meta, synthesis.get("meta_analysis"), errors)
    expected = derive_synthesis_conclusion(protocol, ordered, meta)
    if synthesis.get("conclusion") != expected:
        errors.append("synthesis: conclusion mismatch")


def verify_orchestration(value: dict[str, Any], errors: list[str]) -> None:
    if value.get("current_phase") != "StewardshipReleased":
        errors.append("orchestration: not StewardshipReleased")
    previous = GENESIS
    phase = "Draft"
    replication_id = value.get("replication_id")
    for index, event in enumerate(value.get("events", []), start=1):
        if event.get("sequence") != index:
            errors.append(f"orchestration: sequence mismatch at {index}")
        if event.get("from") != phase:
            errors.append(f"orchestration: phase mismatch at {index}")
        if event.get("previous_event_sha256") != previous:
            errors.append(f"orchestration: chain broken at {index}")
        observed = event.get("event_sha256")
        payload = dict(event)
        payload.pop("event_sha256", None)
        payload = {"replication_id": replication_id, **payload}
        if not is_sha256(observed) or digest(payload) != observed:
            errors.append(f"orchestration: event digest mismatch at {index}")
        previous = observed
        phase = event.get("to")


def verify_charter(value: dict[str, Any], errors: list[str]) -> None:
    members = value.get("members", [])
    if len(members) < 3:
        errors.append("charter: fewer than three members")
    if len({item.get("organization_id") for item in members}) < 2:
        errors.append("charter: fewer than two organizations")
    roles = {role for item in members for role in item.get("roles", [])}
    missing = REQUIRED_STEWARDSHIP_ROLES - roles
    if missing:
        errors.append(f"charter: missing roles {sorted(missing)}")
    for item in members:
        critical = {
            "ReleaseMaintainer",
            "ReproducibilityCustodian",
            "ArchiveCustodian",
            "SecurityContact",
        } & set(item.get("roles", []))
        if len(critical) > 2:
            errors.append(f"charter: critical role concentration at {item.get('member_id')}")


def verify_archive(value: dict[str, Any], errors: list[str], root: Path | None = None) -> None:
    files_root = digest(value.get("files", []))
    if value.get("files_root_sha256") != files_root:
        errors.append("archive: files root mismatch")
    if value.get("recovery_drill", {}).get("restored_root_sha256") != files_root:
        errors.append("archive: recovery root mismatch")
    if not value.get("recovery_drill", {}).get("succeeded"):
        errors.append("archive: recovery drill failed")
    locations = value.get("locations", [])
    if len(locations) < 2:
        errors.append("archive: fewer than two locations")
    if len({item.get("provider") for item in locations}) != len(locations):
        errors.append("archive: duplicate provider")
    for item in locations:
        if item.get("object_root_sha256") != files_root:
            errors.append(f"archive: object root mismatch for {item.get('provider')}")
    if root is not None:
        for item in value.get("files", []):
            relative = item.get("relative_path")
            path = root / relative if isinstance(relative, str) else None
            if path is None or not path.is_file():
                errors.append(f"archive: missing archived file {relative}")
                continue
            payload = path.read_bytes()
            if len(payload) != item.get("size_bytes"):
                errors.append(f"archive: size mismatch for {relative}")
            if hashlib.sha256(payload).hexdigest() != item.get("sha256"):
                errors.append(f"archive: digest mismatch for {relative}")


def verify_release(values: dict[str, Any], errors: list[str]) -> None:
    source = values["source_final_release"]
    protocol = values["protocol"]
    registry = values["registry"]
    packages = values["packages"]
    executions = values["executions"]
    synthesis = values["synthesis"]
    orchestration = values["orchestration"]
    charter = values["charter"]
    archive = values["archive"]
    promotion = values["promotion"]
    release = values["release"]

    expected = {
        "source_final_release_sha256": source.get("bundle_sha256"),
        "replication_protocol_sha256": protocol.get("protocol_sha256"),
        "site_registry_sha256": registry.get("registry_sha256"),
        "site_packages_root_sha256": package_root(packages),
        "site_executions_root_sha256": execution_root(executions),
        "replication_synthesis_sha256": synthesis.get("synthesis_sha256"),
        "replication_orchestration_sha256": orchestration.get("log_sha256"),
        "stewardship_charter_sha256": charter.get("charter_sha256"),
        "research_archive_sha256": archive.get("archive_sha256"),
        "release_promotion_sha256": promotion.get("promotion_sha256"),
    }
    for field, value in expected.items():
        if release.get(field) != value:
            errors.append(f"stewardship release: {field} mismatch")
    if synthesis.get("conclusion") != "IndependentlyReplicated":
        errors.append("stewardship release: replication not established")
    if promotion.get("target_stage") != "StableResearchRelease" or not promotion.get("promoted"):
        errors.append("stewardship release: stable promotion not established")
    if any(gate.get("status") != "Passed" for gate in promotion.get("gates", [])):
        errors.append("stewardship release: failed promotion gate")
    if protocol.get("source_final_release_sha256") != source.get("bundle_sha256"):
        errors.append("stewardship release: protocol lineage mismatch")
    if registry.get("protocol_sha256") != protocol.get("protocol_sha256"):
        errors.append("stewardship release: registry lineage mismatch")
    if synthesis.get("protocol_sha256") != protocol.get("protocol_sha256"):
        errors.append("stewardship release: synthesis protocol mismatch")
    if synthesis.get("site_registry_sha256") != registry.get("registry_sha256"):
        errors.append("stewardship release: synthesis registry mismatch")
    if charter.get("source_final_release_sha256") != source.get("bundle_sha256"):
        errors.append("stewardship release: charter lineage mismatch")
    if archive.get("authority_root_sha256") != synthesis.get("synthesis_sha256"):
        errors.append("stewardship release: archive authority mismatch")
    if archive.get("stewardship_id") != charter.get("stewardship_id"):
        errors.append("stewardship release: archive stewardship mismatch")


def verify_root(root: Path) -> list[str]:
    errors: list[str] = []
    values: dict[str, Any] = {}
    for key, filename in FILES.items():
        path = root / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
            continue
        values[key] = read_json(path)
    if errors:
        return errors
    for key, field in SELF_FIELDS.items():
        verify_self(key, values[key], field, errors)
    for item in values["packages"]:
        verify_self(f"package:{item.get('site_id')}", item, "package_sha256", errors)
    for item in values["executions"]:
        verify_self(f"execution:{item.get('site_id')}", item, "record_sha256", errors)
    source = values["source_final_release"]
    protocol = values["protocol"]
    registry = values["registry"]
    packages = values["packages"]
    executions = values["executions"]
    verify_protocol(source, protocol, errors)
    active = verify_registry(protocol, registry, errors)
    verify_packages(protocol, registry, packages, active, errors)
    verify_executions(protocol, registry, packages, executions, active, errors)
    verify_synthesis(protocol, registry, executions, values["synthesis"], errors)
    verify_orchestration(values["orchestration"], errors)
    verify_charter(values["charter"], errors)
    verify_archive(values["archive"], errors, root)
    verify_release(values, errors)
    return errors


def self_test() -> None:
    assert digest({"b": 2, "a": 1}) == hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    source = {
        "source_final_release_sha256": "a" * 64,
        "endpoint_id": "primary",
        "estimate": 0.1,
        "standard_error": 0.02,
        "confidence_lower": 0.06,
        "confidence_upper": 0.14,
    }
    records = []
    for site in ("a", "b"):
        records.append(
            {
                "site_id": site,
                "conclusion": "SupportsReplication",
                "primary_result": {
                    "estimate": 0.1,
                    "standard_error": 0.02,
                    "participant_count": 48,
                    "family_count": 24,
                },
            }
        )
    meta = compute_meta(source, records)
    assert meta is not None
    assert abs(meta["random_effect_estimate"] - 0.1) < 1e-12
    assert abs(meta["tau_squared"]) < 1e-12
    sealed = {"value": 1}
    sealed["sha256"] = digest(sealed)
    errors: list[str] = []
    verify_self("self-test", sealed, "sha256", errors)
    assert not errors
    sealed["value"] = 2
    verify_self("self-test", sealed, "sha256", errors)
    assert errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("V13 verifier self-test passed")
        return 0
    if args.root is None:
        parser.error("ROOT is required unless --self-test is used")
    errors = verify_root(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("V13 replication and stewardship release verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
