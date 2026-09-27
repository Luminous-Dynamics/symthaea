#!/usr/bin/env python3
import copy
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_DIGEST = "428f7ab81d95973b3a324a1d88c9c2d29ecaded5a1b6914bacba00e7849a45b0"
EXPECTED_PARENT = {
    "source_head": "9e00c54f5f5c319df2c25d4af4a2381a21ffc4e9",
    "tree": "ff87b6915904e7d4a82862c0e4e101dc0d51e099",
    "registry_sha256": "87cbe6e38b84aaeecfa188997774b64aec209b314f158b2f7867ee9f34b29665",
    "schema": "transport-bench-ext-001a-registry-v1",
}
TOP_KEYS = {
    "schema", "issue", "source_kind", "snapshot_date", "parent_registry",
    "claim_ceiling", "common_mode_rule", "entries",
}
ENTRY_KEYS = {
    "id", "mode", "benchmark", "benchmark_class", "external_form", "version",
    "task", "status", "rights", "claim_ceiling", "adapter_disposition",
    "common_mode_roots", "notes",
}
RIGHT_KEYS = {"training", "local_evaluation", "redistribution", "commercial"}
COMMON_MODE_KEYS = {"dataset_family", "simulator", "sensor_model", "evaluator", "exposure"}


def canonical_bytes(data):
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()


def validate(data, *, check_digest=True):
    if set(data) != TOP_KEYS:
        raise ValueError("top-level closed-world schema violation")
    if data["schema"] != "transport-bench-ext-001b-extension-v1":
        raise ValueError("schema")
    if data["issue"] != 6266:
        raise ValueError("issue")
    if data["source_kind"] != "external-benchmark-registry-extension":
        raise ValueError("source_kind")
    if data["snapshot_date"] != "2026-09-27":
        raise ValueError("snapshot_date")
    if data["parent_registry"] != EXPECTED_PARENT:
        raise ValueError("parent_registry")
    if data["claim_ceiling"] != "ArchitectureRegistryExtensionOnly":
        raise ValueError("claim_ceiling")
    if data["common_mode_rule"] != (
        "Different benchmark names do not imply independent evidence when dataset, "
        "simulator, sensor assumptions, evaluator, or tuning exposure share a root."
    ):
        raise ValueError("common_mode_rule")

    entries = data["entries"]
    ids = [entry["id"] for entry in entries]
    if ids != [f"B{i}" for i in range(24, 34)] or len(set(ids)) != 10:
        raise ValueError("ordered census")

    for entry in entries:
        if set(entry) != ENTRY_KEYS:
            raise ValueError(f"{entry.get('id')}: entry closed-world schema")
        for key in (
            "id", "mode", "benchmark", "benchmark_class", "external_form", "version",
            "task", "status", "claim_ceiling", "adapter_disposition", "notes",
        ):
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise ValueError(f"{entry['id']}: empty {key}")
        if set(entry["rights"]) != RIGHT_KEYS:
            raise ValueError(f"{entry['id']}: rights shape")
        if not all(isinstance(v, str) and v.strip() for v in entry["rights"].values()):
            raise ValueError(f"{entry['id']}: empty rights")
        if set(entry["common_mode_roots"]) != COMMON_MODE_KEYS:
            raise ValueError(f"{entry['id']}: common-mode shape")
        if not all(isinstance(v, str) and v.strip() for v in entry["common_mode_roots"].values()):
            raise ValueError(f"{entry['id']}: empty common-mode root")

    by = {entry["id"]: entry for entry in entries}

    b = by["B24"]
    if not (
        b["benchmark"] == "BARN Challenge 2026"
        and b["external_form"] == "ChallengeWithHiddenTestSetAndStandardizedPhysicalFinal"
        and b["rights"]["commercial"] == "UnknownNeedsReview"
        and b["claim_ceiling"] == "NavigationBenchmarkOnly"
        and b["adapter_disposition"] == "PreferredFirst"
        and b["common_mode_roots"]["dataset_family"] == "BARN"
        and "does not establish" in b["notes"]
    ):
        raise ValueError("B24 semantic boundary")

    b = by["B25"]
    if not (
        b["benchmark"] == "TartanGround"
        and b["rights"]["redistribution"] == "CC-BY-4.0 conditions apply"
        and b["rights"]["commercial"] == "AllowedWithAttribution"
        and b["claim_ceiling"] == "PerceptionNavigationSubproblemEvidenceOnly"
        and b["common_mode_roots"]["dataset_family"] == "TartanGround/TartanAir ecosystem"
    ):
        raise ValueError("B25 semantic boundary")

    b = by["B26"]
    if not (
        b["benchmark"] == "RELLIS-3D"
        and b["version"] == "VersionSelectionDeferred"
        and set(b["rights"].values()) == {"UnknownNeedsReview"}
        and b["claim_ceiling"] == "OffroadPerceptionOnly"
    ):
        raise ValueError("B26 semantic boundary")

    b = by["B27"]
    if not (
        b["version"] == "DOI-10.3929/ethz-b-000690084"
        and b["rights"]["training"] == "NonCommercialUsePermitted"
        and b["rights"]["local_evaluation"] == "NonCommercialUsePermitted"
        and b["rights"]["commercial"] == "RestrictedByKnownTerms"
        and b["claim_ceiling"] == "EstimatorNavigationEvidenceOnly"
        and "ground truth" in b["notes"].lower()
    ):
        raise ValueError("B27 semantic boundary")

    b = by["B28"]
    if not (
        b["benchmark"] == "TartanAir V2"
        and b["rights"]["redistribution"] == "CC-BY-4.0 conditions apply"
        and b["rights"]["commercial"] == "AllowedWithAttribution"
        and b["claim_ceiling"] == "SyntheticPerceptionNavigationEvidenceOnly"
    ):
        raise ValueError("B28 semantic boundary")

    for ident in ("B29", "B30"):
        b = by[ident]
        if not (
            b["benchmark_class"] == "PhysicalTestbedBenchmark"
            and b["external_form"] == "ExternalPhysicalTestPrecedent"
            and b["adapter_disposition"] == "ReferenceOnly"
            and "PrecedentOnly" in b["claim_ceiling"]
        ):
            raise ValueError(f"{ident} precedent boundary")

    for ident in ("B31", "B32"):
        b = by[ident]
        if not (
            b["external_form"] == "BenchmarkGap"
            and b["status"] == "BenchmarkGap"
            and b["adapter_disposition"] == "BenchmarkGap"
            and b["claim_ceiling"] == "BenchmarkGapOnly"
        ):
            raise ValueError(f"{ident} gap boundary")

    b = by["B33"]
    if not (
        b["external_form"] == "BenchmarkGap"
        and b["status"] == "BenchmarkGap"
        and b["adapter_disposition"] == "Deferred"
        and b["claim_ceiling"] == "ModeSpecificAuditRequired"
    ):
        raise ValueError("B33 deferred-audit boundary")

    if check_digest:
        digest = hashlib.sha256(canonical_bytes(data)).hexdigest()
        if digest != EXPECTED_DIGEST:
            raise ValueError(f"canonical digest mismatch: {digest}")

    return True


def expect_reject(base, name, mutator):
    candidate = copy.deepcopy(base)
    mutator(candidate)
    try:
        validate(candidate, check_digest=False)
    except (AssertionError, KeyError, TypeError, ValueError):
        return
    raise AssertionError(f"mutation unexpectedly accepted: {name}")


def mutation_suite(base):
    cases = []

    def add(name, fn):
        cases.append((name, fn))

    add("schema", lambda x: x.__setitem__("schema", "x"))
    add("issue", lambda x: x.__setitem__("issue", 1))
    add("source kind", lambda x: x.__setitem__("source_kind", "x"))
    add("snapshot", lambda x: x.__setitem__("snapshot_date", "2026-09-28"))
    add("parent head", lambda x: x["parent_registry"].__setitem__("source_head", "0" * 40))
    add("parent tree", lambda x: x["parent_registry"].__setitem__("tree", "0" * 40))
    add("parent digest", lambda x: x["parent_registry"].__setitem__("registry_sha256", "0" * 64))
    add("parent schema", lambda x: x["parent_registry"].__setitem__("schema", "x"))
    add("claim escalation", lambda x: x.__setitem__("claim_ceiling", "TransportAuthority"))
    add("independence rule", lambda x: x.__setitem__("common_mode_rule", "names imply independence"))
    add("drop entry", lambda x: x["entries"].pop())
    add("reorder entries", lambda x: x["entries"].reverse())
    add("duplicate id", lambda x: x["entries"].__setitem__(1, copy.deepcopy(x["entries"][0])))
    add("invent independence count", lambda x: x.__setitem__("evidence_independence_count", 10))
    add("invent transport ok", lambda x: x["entries"][0].__setitem__("transport_ok", True))
    add("remove rights dimension", lambda x: x["entries"][0]["rights"].pop("commercial"))
    add("remove common-mode root", lambda x: x["entries"][0]["common_mode_roots"].pop("evaluator"))
    add("blank common-mode root", lambda x: x["entries"][0]["common_mode_roots"].__setitem__("dataset_family", ""))
    add("BARN commercial", lambda x: x["entries"][0]["rights"].__setitem__("commercial", "Allowed"))
    add("BARN fleet authority", lambda x: x["entries"][0].__setitem__("claim_ceiling", "WarehouseFleetCapability"))
    add("BARN operational reclass", lambda x: x["entries"][0].__setitem__("external_form", "OperationalFleetBenchmark"))
    add("TartanGround license", lambda x: x["entries"][1]["rights"].__setitem__("redistribution", "Unrestricted"))
    add("RELLIS commercial", lambda x: x["entries"][2]["rights"].__setitem__("commercial", "Allowed"))
    add("RELLIS floating latest", lambda x: x["entries"][2].__setitem__("version", "Latest"))
    add("EuRoC commercial", lambda x: x["entries"][3]["rights"].__setitem__("commercial", "Allowed"))
    add("EuRoC airworthiness", lambda x: x["entries"][3].__setitem__("claim_ceiling", "Airworthiness"))
    add("EuRoC floating latest", lambda x: x["entries"][3].__setitem__("version", "Latest"))
    add("TartanAir license", lambda x: x["entries"][4]["rights"].__setitem__("commercial", "UnknownNeedsReview"))
    add("Mars Yard preferred benchmark", lambda x: x["entries"][5].__setitem__("adapter_disposition", "PreferredFirst"))
    add("Mars Yard NASA qualified", lambda x: x["entries"][5].__setitem__("claim_ceiling", "NASAQualified"))
    add("ERNEST leaderboard", lambda x: x["entries"][6].__setitem__("external_form", "PublicLeaderboardBenchmark"))
    add("ERNEST mission ready", lambda x: x["entries"][6].__setitem__("claim_ceiling", "MissionReady"))
    add("fill corridor gap", lambda x: (x["entries"][7].__setitem__("status", "PublishedBenchmark"), x["entries"][7].__setitem__("adapter_disposition", "PreferredFirst")))
    add("fill heavy gap", lambda x: (x["entries"][8].__setitem__("status", "PublishedBenchmark"), x["entries"][8].__setitem__("adapter_disposition", "Candidate")))
    add("universal novel coverage", lambda x: (x["entries"][9].__setitem__("status", "UniversalCoverage"), x["entries"][9].__setitem__("claim_ceiling", "AllNovelModesQualified")))
    add("fake independence root", lambda x: x["entries"][1]["common_mode_roots"].__setitem__("dataset_family", "IndependentByBenchmarkName"))
    add("operation authority", lambda x: x["entries"][1].__setitem__("claim_ceiling", "OperationAuthority"))
    add("BARN positive authority prose", lambda x: x["entries"][0].__setitem__("notes", "PASS establishes deployment authority"))
    add("extra rights key", lambda x: x["entries"][0]["rights"].__setitem__("license_ok", "true"))
    add("extra common-mode score", lambda x: x["entries"][0]["common_mode_roots"].__setitem__("independence_score", "1.0"))

    for name, fn in cases:
        expect_reject(base, name, fn)
    return len(cases)


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/engineering/data/transport_bench_ext_001b_extension_v1.json")
    data = json.loads(path.read_bytes())
    validate(data, check_digest=True)
    count = mutation_suite(data)
    print(f"PASS: {len(data['entries'])} entries; {count} hostile mutations rejected; digest {EXPECTED_DIGEST}")


if __name__ == "__main__":
    main()
