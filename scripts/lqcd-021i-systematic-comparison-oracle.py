#!/usr/bin/env python3
"""Independent systematic-error / comparison oracle for LQCD-021I."""
import copy
import hashlib
import json
import math

ORACLE_ID = "lqcd_beta6_systematic_comparison_oracle_v1"

SEALED_2840 = {
    "sealed_digest": "a6e9235bd001fe4b90d736ec6e502c0d92eb7e89dbae5b57618077c8c94fe3b7",
    "source": "LQCD-021F#2840",
    "coordinates": {
        "a_sqrt_sigma": {"estimate": 0.228178735476, "stat_se": None},
        "r0_over_a": {"estimate": 5.163580437350, "stat_se": 0.001017988705},
        "r4_over_a": {"estimate": 8.473369264395, "stat_se": 0.001670506407},
        "r6_over_a": {"estimate": 10.498148088743, "stat_se": 0.002069687169},
    },
}

EHK = {
    "benchmark_id": "EHK_beta6_16x32_published_v1",
    "coordinates": {
        "a_sqrt_sigma": {"estimate": 0.2189, "source_se": 0.0009},
        "r0_over_a": {"estimate": 5.369, "source_se": 0.009},
        "r4_over_a": {"estimate": 8.831, "source_se": 0.021},
        "r6_over_a": {"estimate": 10.89, "source_se": 0.03},
    },
}

BASE_LEDGER = {
    "schema": "lqcd_systematic_ledger_v1",
    "ledger_id": "synthetic_systematic_ledger_v1",
    "components": [
        {"id": "autocorrelation_block_adequacy", "status": "ResolvedAdequate",
         "quantitative_sigma": None, "denominator_authority": "ExcludedEvidenceOnly"},
        {"id": "plateau_excited_state", "status": "BoundedDiagnosticOnly",
         "quantitative_sigma": 0.002, "denominator_authority": "NotAuthorizedUnknownCorrelation"},
        {"id": "fit_family_r_range", "status": "BoundedDiagnosticOnly",
         "quantitative_sigma": 0.003, "denominator_authority": "NotAuthorizedUnknownCorrelation"},
        {"id": "operator_convention", "status": "NotApplicableSynthetic",
         "quantitative_sigma": None, "denominator_authority": "ExcludedEvidenceOnly"},
        {"id": "ape_historical_ambiguity", "status": "NotApplicableSynthetic",
         "quantitative_sigma": None, "denominator_authority": "ExcludedEvidenceOnly"},
        {"id": "lattice_coulomb_numerical", "status": "ResolvedNegligibleSynthetic",
         "quantitative_sigma": None, "denominator_authority": "ExcludedEvidenceOnly"},
        {"id": "finite_volume", "status": "NotApplicableSynthetic",
         "quantitative_sigma": None, "denominator_authority": "ExcludedEvidenceOnly"},
        {"id": "historical_underdetermination", "status": "NotApplicableSynthetic",
         "quantitative_sigma": None, "denominator_authority": "ExcludedEvidenceOnly"},
    ],
    "denominator_policy": {
        "allowed_terms": ["sealed_statistical_se", "benchmark_source_se"],
        "combination": "quadrature",
        "justification": (
            "Synthetic fixture explicitly treats estimator sampling uncertainty and "
            "benchmark-source uncertainty as independent. No systematic component "
            "is authorized for denominator combination."
        ),
    },
    "claim_policy": {
        "support_abs_z_lte": 2.0,
        "not_support_abs_z_gte": 5.0,
        "primary_coordinates": ["a_sqrt_sigma", "r0_over_a", "r4_over_a", "r6_over_a"],
        "unresolved_statuses": ["Unresolved", "Unknown", "InsufficientEvidence"],
    },
}

LEDGER_TAG = b"symthaea.lqcd.systematic-ledger.v1\0"
RECEIPT_TAG = b"symthaea.lqcd.comparison-receipt.v1\0"
SEALED_TAG = b"symthaea.lqcd.synthetic-sealed-result.v1\0"
RESULT_TAG = b"symthaea.lqcd.021i.result.v1\0"

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def digest(tag, value):
    return hashlib.sha256(tag + canonical(value)).hexdigest()

def validate_ledger(ledger):
    expected = {"sealed_statistical_se", "benchmark_source_se"}
    if set(ledger["denominator_policy"]["allowed_terms"]) != expected:
        raise ValueError("v1 denominator terms changed")
    if ledger["denominator_policy"]["combination"] != "quadrature":
        raise ValueError("v1 denominator combination changed")
    seen = set()
    for component in ledger["components"]:
        if component["id"] in seen:
            raise ValueError("duplicate systematic component")
        seen.add(component["id"])
        sigma = component["quantitative_sigma"]
        if sigma is not None and sigma < 0:
            raise ValueError("negative systematic sigma")
        if component["denominator_authority"].startswith("Authorized"):
            raise ValueError("v1 authorizes no systematic component in denominator")

def compare(sealed, benchmark, ledger):
    validate_ledger(ledger)
    policy = ledger["claim_policy"]
    unresolved = [
        c["id"] for c in ledger["components"]
        if c["status"] in policy["unresolved_statuses"]
    ]
    coordinates = {}
    abs_z = []
    missing = []
    for key in policy["primary_coordinates"]:
        source = sealed["coordinates"][key]
        target = benchmark["coordinates"][key]
        delta = source["estimate"] - target["estimate"]
        z = None
        unavailable = None
        if source["stat_se"] is None:
            unavailable = "MissingSealedStatisticalUncertainty"
            missing.append(key)
        elif target["source_se"] is None:
            unavailable = "MissingBenchmarkSourceUncertainty"
            missing.append(key)
        else:
            denominator = math.sqrt(
                source["stat_se"] ** 2 + target["source_se"] ** 2
            )
            z = delta / denominator
            abs_z.append(abs(z))
        coordinates[key] = {
            "sealed_estimate": source["estimate"],
            "sealed_stat_se": source["stat_se"],
            "benchmark_estimate": target["estimate"],
            "benchmark_source_se": target["source_se"],
            "signed_difference": delta,
            "standardized_discrepancy": z,
            "unavailable_reason": unavailable,
        }

    if unresolved:
        disposition = "Inconclusive"
        rationale = ["UnresolvedSystematicEvidence"]
    elif any(z >= policy["not_support_abs_z_gte"] for z in abs_z):
        disposition = "ExactVolumeNumericalReproductionNotSupported"
        rationale = ["LargeDeclaredStandardizedDiscrepancy"]
    elif missing:
        disposition = "Inconclusive"
        rationale = ["MissingRequiredCoordinateUncertainty"]
    elif abs_z and all(z <= policy["support_abs_z_lte"] for z in abs_z):
        disposition = "ExactVolumeNumericalReproductionSupported"
        rationale = ["AllPrimaryCoordinatesWithinFrozenSupportBand"]
    else:
        disposition = "Inconclusive"
        rationale = ["IntermediateDiscrepancy"]

    receipt = {
        "schema": "lqcd_beta6_comparison_receipt_v1",
        "sealed_digest": sealed["sealed_digest"],
        "benchmark_id": benchmark["benchmark_id"],
        "systematic_ledger_digest": digest(LEDGER_TAG, ledger),
        "coordinates": coordinates,
        "disposition": disposition,
        "rationale": rationale,
        "unresolved_components": unresolved,
        "claim_boundary": "synthetic_comparison_semantics_only_no_real_beta6_claim",
    }
    receipt["receipt_digest"] = digest(RECEIPT_TAG, receipt)
    return receipt

def main():
    main_receipt = compare(SEALED_2840, EHK, BASE_LEDGER)

    supported_sealed = copy.deepcopy(SEALED_2840)
    supported_sealed["source"] = "independent_supported_fixture"
    supported_sealed["coordinates"]["a_sqrt_sigma"]["stat_se"] = 0.001
    supported_sealed["sealed_digest"] = digest(SEALED_TAG, supported_sealed)
    close_benchmark = {
        "benchmark_id": "synthetic_close_benchmark",
        "coordinates": {
            "a_sqrt_sigma": {"estimate": 0.2279, "source_se": 0.001},
            "r0_over_a": {"estimate": 5.164, "source_se": 0.002},
            "r4_over_a": {"estimate": 8.474, "source_se": 0.003},
            "r6_over_a": {"estimate": 10.499, "source_se": 0.004},
        },
    }
    supported_receipt = compare(supported_sealed, close_benchmark, BASE_LEDGER)

    unresolved_ledger = copy.deepcopy(BASE_LEDGER)
    for component in unresolved_ledger["components"]:
        if component["id"] == "operator_convention":
            component["status"] = "Unresolved"
            component["quantitative_sigma"] = 0.05
            component["denominator_authority"] = "NotAuthorizedUnknownCorrelation"
    inconclusive_receipt = compare(SEALED_2840, EHK, unresolved_ledger)

    blind_quadrature = copy.deepcopy(BASE_LEDGER)
    blind_quadrature["denominator_policy"]["allowed_terms"].append("plateau_excited_state")
    blind_systematic_quadrature_rejected = False
    try:
        validate_ledger(blind_quadrature)
    except ValueError:
        blind_systematic_quadrature_rejected = True

    mutated_benchmark = copy.deepcopy(EHK)
    mutated_benchmark["benchmark_id"] = "mutated_benchmark_fixture"
    mutated_benchmark["coordinates"]["r0_over_a"]["estimate"] += 0.5
    mutated_receipt = compare(SEALED_2840, mutated_benchmark, BASE_LEDGER)

    result = {
        "oracle_id": ORACLE_ID,
        "sealed_2840_digest": SEALED_2840["sealed_digest"],
        "ehk_fixture_disposition": main_receipt["disposition"],
        "ehk_fixture_receipt_digest": main_receipt["receipt_digest"],
        "ehk_standardized_discrepancies": {
            key: value["standardized_discrepancy"]
            for key, value in main_receipt["coordinates"].items()
        },
        "a_sqrt_sigma_missing_uncertainty_fails_closed": (
            main_receipt["coordinates"]["a_sqrt_sigma"]["standardized_discrepancy"] is None
        ),
        "supported_fixture_disposition": supported_receipt["disposition"],
        "inconclusive_fixture_disposition": inconclusive_receipt["disposition"],
        "blind_systematic_quadrature_rejected": blind_systematic_quadrature_rejected,
        "benchmark_mutation_preserves_sealed_digest": (
            mutated_receipt["sealed_digest"]
            == main_receipt["sealed_digest"]
            == SEALED_2840["sealed_digest"]
        ),
        "benchmark_mutation_changes_comparison_receipt": (
            mutated_receipt["receipt_digest"] != main_receipt["receipt_digest"]
        ),
        "systematic_sigmas_preserved_but_not_combined": [
            component["id"] for component in BASE_LEDGER["components"]
            if component["quantitative_sigma"] is not None
        ],
        "scientific_boundary": (
            "synthetic_comparison_semantics_only_no_real_beta6_claim"
        ),
    }

    if result["ehk_fixture_disposition"] != "ExactVolumeNumericalReproductionNotSupported":
        raise AssertionError("synthetic #2840 / EHK fixture must exercise NotSupported")
    if result["supported_fixture_disposition"] != "ExactVolumeNumericalReproductionSupported":
        raise AssertionError("close synthetic fixture must exercise Supported")
    if result["inconclusive_fixture_disposition"] != "Inconclusive":
        raise AssertionError("unresolved systematic must force Inconclusive")
    if not result["blind_systematic_quadrature_rejected"]:
        raise AssertionError("unauthorized systematic quadrature must fail")
    if not result["a_sqrt_sigma_missing_uncertainty_fails_closed"]:
        raise AssertionError("missing sealed uncertainty must remain unavailable")
    if not result["benchmark_mutation_preserves_sealed_digest"]:
        raise AssertionError("comparison must not mutate sealed estimator identity")
    if not result["benchmark_mutation_changes_comparison_receipt"]:
        raise AssertionError("benchmark mutation should change only downstream comparison")

    result_sha = digest(RESULT_TAG, result)
    print("ok")
    print("result_sha256=" + result_sha)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))

if __name__ == "__main__":
    main()
