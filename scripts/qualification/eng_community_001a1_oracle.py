#!/usr/bin/env python3
"""Independent ENG-COMMUNITY-001A semantic oracle.

Stdlib-only. This oracle deliberately does not import production/community
projection code and does not trust the source corpus's embedded `expect` labels.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

SOURCE_HEAD = "f3e87f5ff7ca823c0860de16b22eb4666189c2ee"
SOURCE_JSON_SHA256 = "1ef1548759ba8c81c93e7c6189042da50757a4af0d40bf69aa8163866046230f"
SOURCE_DOC_BLOB = "c3ce167575a1e1fd910eda4f657afeb12acd9b50"
SOURCE_JSON_BLOB = "5cd16591f0ffa56da3f469847e99844cbda79e79"

DOC_PATH = pathlib.Path(
    "docs/engineering/community/ENG_COMMUNITY_001A_SYNTHETIC_COMMUNITY_INFRASTRUCTURE_V1.md"
)
JSON_PATH = pathlib.Path(
    "docs/engineering/community/eng-community-001a-synthetic-community-infrastructure-v1.json"
)
ORACLE_PATH = pathlib.Path("scripts/qualification/eng_community_001a1_oracle.py")

EXPECTED_DIMENSIONS = [
    "StewardshipState",
    "EconomicInterestState",
    "LegalTitleState",
    "CapitalClaimState",
    "OperatorConcessionState",
    "OperatorQualificationState",
    "ExternalLicenceState",
    "ServiceCovenantState",
    "LifecycleReserveState",
    "HandbackState",
    "EngineeringConfigurationState",
    "PublicProjectionState",
]

EXPECTED_MODULES = {
    "land-public-realm": ("Stewarded", "CommunityFacilitiesOperator"),
    "thermal-network": ("CooperativeStewarded", "ThermalUtilityCooperative"),
    "water-wellness": ("Stewarded", "CommunityWaterWellnessOperator"),
    "greenhouse": ("CommunityEnterprise", "CommunityEnterprise"),
    "research-facility": ("Partnership", "ResearchPartnership"),
    "compute-datacenter": (
        "PrivateConcessionOnStewardedAsset",
        "PrivateComputeConcessionaire",
    ),
    "energy-project-spv": ("MixedCapitalBoundedClaims", "QualifiedEnergyOperator"),
}

CLAIM_CEILING_KEYS = {
    "democratic_legitimacy_established",
    "engineering_safety_established",
    "financial_viability_established",
    "legal_title_established",
    "licensing_established",
    "physical_operating_authority_established",
}

# Independent case rules. Derive the disposition from the case ID plus required
# semantic cues, then compare it with the source's embedded `expect` field.
CASE_RULES = {
    "C01": (("land/public realm", "energy SPV mixed"), "MixedOwnershipPreserved"),
    "C02": (("economic interest", "qualification absent"), "NoOperationAuthority"),
    "C03": (("qualified operator", "stewardship ref unknown"), "CommunityOwnershipNotEstablished"),
    "C04": (("investor", "majority capex"), "NoGovernanceAuthorityFromCapital"),
    "C05": (("investor claim sold",), "StewardshipUnchanged"),
    "C06": (("claim reaches zero",), "NoAutomaticTitleOrHandback"),
    "C07": (("service covenant fails",), "DistributionBlockedOwnershipPreserved"),
    "C08": (("reserve deficient",), "ReserveBlockedOwnershipPreserved"),
    "C09": (("benefit payment",), "NoOverallFairnessClaim"),
    "C10": (("private compute concession",), "NoCampusPrivatizationInference"),
    "C11": (("private concession expires",), "ModuleOperationChangesStewardshipIndependent"),
    "C12": (("replaces operator", "qualification/licence absent"), "ReplacementBlocked"),
    "C13": (("operator changes", "configuration unchanged"), "OperatorGenerationChangesPhysicalMayRemain"),
    "C14": (("material physical configuration changes",), "EngineeringCurrentnessReopenedGovernanceHistoryPreserved"),
    "C15": (("mandatory external constraint",), "ImplementationBlocked"),
    "C16": (("regulator/licensor ref changes",), "ApplicabilityReviewRequired"),
    "C17": (("omits restricted dependency",), "StructurallyIncomplete"),
    "C18": (("RestrictedPresent",), "ValidPublicAbstraction"),
    "C19": (("synthetic fixture", "real deployment"), "RejectSyntheticAsReal"),
    "C20": (("community_owned=true",), "RejectCollapsedOwnershipBoolean"),
    "C21": (("benefit improves", "worsens"), "MultidimensionalDisagreementPreserved"),
    "C22": (("legal title unknown", "economic-interest ref current"), "IndependentStatesPreserved"),
    "C23": (("module removed",), "DependentRefsReconsideredUnrelatedPreserved"),
    "C24": (("all synthetic dimensions close",), "SyntheticCompositionOnlyNoAuthorityPromotion"),
}

FORBIDDEN_KEY_TOKENS = {
    "private_key",
    "password",
    "credential",
    "access_token",
    "api_token",
    "security_sensor_coordinate",
    "security_sensor_location",
    "response_tactic",
    "reactor_core_geometry",
    "fuel_enrichment",
    "criticality",
    "protection_setpoint",
    "trip_logic",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def git_blob(path: pathlib.Path) -> str:
    return git("hash-object", str(path))


def canonical_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def derive_case_disposition(case: dict[str, Any]) -> str:
    case_id = case.get("id")
    scenario = case.get("scenario")
    if case_id not in CASE_RULES:
        raise ValueError(f"unknown case id: {case_id!r}")
    if not isinstance(scenario, str):
        raise ValueError(f"{case_id}: scenario must be a string")
    required_cues, disposition = CASE_RULES[case_id]
    lower = scenario.lower()
    for cue in required_cues:
        if cue.lower() not in lower:
            raise ValueError(f"{case_id}: missing independent semantic cue {cue!r}")
    return disposition


def validate(doc: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["top-level document is not an object"]

    exact_top = {
        "campus",
        "cases",
        "claim_ceiling",
        "dimensions",
        "profile_id",
        "schema",
        "synthetic_only",
    }
    if set(doc) != exact_top:
        errors.append(f"top-level keys differ: {sorted(doc)}")

    if doc.get("schema") != "eng-community-001a-reference-v1":
        errors.append("unsupported schema")
    if doc.get("profile_id") != "ENG-COMMUNITY-001A":
        errors.append("wrong profile_id")
    if doc.get("synthetic_only") is not True:
        errors.append("synthetic_only must be true")

    if any(key == "community_owned" for key in walk_keys(doc)):
        errors.append("collapsed community_owned authority boolean is forbidden")
    bad_keys = sorted({key for key in walk_keys(doc) if key.lower() in FORBIDDEN_KEY_TOKENS})
    if bad_keys:
        errors.append(f"forbidden real/security/credential fields present: {bad_keys}")

    dims = doc.get("dimensions")
    if dims != EXPECTED_DIMENSIONS:
        errors.append("dimension vector differs from frozen 12-dimension profile")
    if isinstance(dims, list) and len(set(dims)) != len(dims):
        errors.append("duplicate dimensions")

    ceiling = doc.get("claim_ceiling")
    if not isinstance(ceiling, dict):
        errors.append("claim_ceiling missing/not object")
    else:
        if set(ceiling) != CLAIM_CEILING_KEYS:
            errors.append("claim_ceiling keys differ")
        for key in CLAIM_CEILING_KEYS:
            if ceiling.get(key) is not False:
                errors.append(f"authority promotion forbidden: {key}")

    campus = doc.get("campus")
    if not isinstance(campus, dict):
        errors.append("campus missing/not object")
    else:
        if campus.get("campus_subject") != "synthetic:community-campus:v1":
            errors.append("campus subject is not exact synthetic subject")
        if campus.get("engineering_configuration_generation") != "SYNTH_CAMPUS_G1":
            errors.append("engineering configuration generation differs")
        modules = campus.get("modules")
        if not isinstance(modules, list):
            errors.append("modules missing/not list")
        else:
            if len(modules) != 7:
                errors.append("module census must contain exactly 7 modules")
            by_id: dict[str, dict[str, Any]] = {}
            for module in modules:
                if not isinstance(module, dict):
                    errors.append("module is not object")
                    continue
                module_id = module.get("id")
                if not isinstance(module_id, str):
                    errors.append("module id missing/not string")
                    continue
                if module_id in by_id:
                    errors.append(f"duplicate module id {module_id}")
                by_id[module_id] = module
                if module.get("stewardship") != "CommunityPublicBenefitSteward":
                    errors.append(f"{module_id}: stewardship drift")
            if set(by_id) != set(EXPECTED_MODULES):
                errors.append("module identity census differs")
            for module_id, (ownership, operator) in EXPECTED_MODULES.items():
                module = by_id.get(module_id)
                if module is None:
                    continue
                if module.get("ownership_class") != ownership:
                    errors.append(f"{module_id}: ownership class differs")
                if module.get("operator") != operator:
                    errors.append(f"{module_id}: operator differs")

    cases = doc.get("cases")
    if not isinstance(cases, list):
        errors.append("cases missing/not list")
    else:
        ids = [case.get("id") for case in cases if isinstance(case, dict)]
        expected_ids = [f"C{i:02d}" for i in range(1, 25)]
        if ids != expected_ids:
            errors.append("case IDs/order must be exactly C01..C24")
        if len(set(ids)) != len(ids):
            errors.append("duplicate case IDs")
        for case in cases:
            if not isinstance(case, dict):
                errors.append("case is not object")
                continue
            try:
                derived = derive_case_disposition(case)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if case.get("expect") != derived:
                errors.append(
                    f"{case.get('id')}: embedded expect {case.get('expect')!r} "
                    f"does not match independently derived {derived!r}"
                )
    return errors


def require_invalid(name: str, candidate: Any) -> None:
    errors = validate(candidate)
    if not errors:
        raise AssertionError(f"hostile mutation unexpectedly accepted: {name}")


def mutation_suite(doc: dict[str, Any]) -> None:
    m = copy.deepcopy(doc)
    m["schema"] = "eng-community-001a-reference-v2"
    require_invalid("unknown schema", m)

    m = copy.deepcopy(doc)
    m["cases"][1]["id"] = "C01"
    require_invalid("duplicate case id", m)

    m = copy.deepcopy(doc)
    del m["cases"][-1]
    require_invalid("missing case", m)

    m = copy.deepcopy(doc)
    del m["dimensions"][-1]
    require_invalid("missing dimension", m)

    m = copy.deepcopy(doc)
    m["claim_ceiling"]["physical_operating_authority_established"] = True
    require_invalid("authority promotion", m)

    m = copy.deepcopy(doc)
    m["synthetic_only"] = False
    require_invalid("synthetic_only false", m)

    m = copy.deepcopy(doc)
    m["private_key"] = "do-not-accept"
    require_invalid("credential/security field", m)

    m = copy.deepcopy(doc)
    m["community_owned"] = True
    require_invalid("collapsed ownership boolean", m)

    m = copy.deepcopy(doc)
    m["cases"][3]["expect"] = "GovernanceAuthorityGranted"
    require_invalid("changed embedded disposition", m)

    m = copy.deepcopy(doc)
    m["campus"]["campus_subject"] = "deployment:real-site:001"
    require_invalid("synthetic promoted to real", m)

    m = copy.deepcopy(doc)
    m["campus"]["modules"].pop()
    require_invalid("module census changed", m)

    m = copy.deepcopy(doc)
    m["cases"][17]["scenario"] = "restricted dependency omitted"
    require_invalid("RestrictedPresent semantics erased", m)


def verify_git_envelope() -> None:
    head = git("rev-parse", "HEAD")
    parent = git("rev-parse", "HEAD^")
    if parent != SOURCE_HEAD:
        raise AssertionError(f"qualifier parent {parent} != frozen source {SOURCE_HEAD}")
    count = git("rev-list", "--count", f"{SOURCE_HEAD}..{head}")
    if count != "1":
        raise AssertionError(f"qualifier must be exactly one commit over source; got {count}")
    diff_files = set(git("diff", "--name-only", f"{SOURCE_HEAD}..{head}").splitlines())
    if diff_files != {str(ORACLE_PATH)}:
        raise AssertionError(f"unexpected qualifier diff: {sorted(diff_files)}")


def main() -> int:
    verify_git_envelope()

    if not DOC_PATH.is_file() or not JSON_PATH.is_file():
        raise AssertionError("frozen source files missing")
    if git_blob(DOC_PATH) != SOURCE_DOC_BLOB:
        raise AssertionError("contract source blob differs")
    if git_blob(JSON_PATH) != SOURCE_JSON_BLOB:
        raise AssertionError("JSON source blob differs")
    if canonical_sha256(JSON_PATH) != SOURCE_JSON_SHA256:
        raise AssertionError("canonical JSON SHA-256 differs")

    doc = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    errors = validate(doc)
    if errors:
        raise AssertionError("source validation failed:\n- " + "\n- ".join(errors))

    ownership_classes = {module["ownership_class"] for module in doc["campus"]["modules"]}
    if len(ownership_classes) < 5:
        raise AssertionError("mixed ownership fixture collapsed unexpectedly")

    mutation_suite(doc)

    result = {
        "profile": "ENG-COMMUNITY-001A1",
        "source_head": SOURCE_HEAD,
        "source_json_sha256": SOURCE_JSON_SHA256,
        "module_count": len(doc["campus"]["modules"]),
        "dimension_count": len(doc["dimensions"]),
        "case_count": len(doc["cases"]),
        "independently_derived_case_count": len(CASE_RULES),
        "hostile_mutation_count": 12,
        "authority_promotion": False,
        "disposition": "PASS",
        "nonclaims": [
            "no legal title established",
            "no democratic legitimacy established",
            "no financial viability established",
            "no engineering safety established",
            "no licensing established",
            "no physical operating authority established",
        ],
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"ENG-COMMUNITY-001A1 FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
