#!/usr/bin/env python3
"""Independent validator for the V18 SCIP text-to-claim benchmark preregistration."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea.scip-text-claim-extraction-preregistration/v1"
AUTHORITY = "preregistration-only"
SEMANTIC_PROFILE = "symthaea.scip-semantic-realization-contract/v1"
ORACLE_SCHEMA = "symthaea.scip-semantic-claim-comparison-result/v1"
EXPECTED_SEMANTIC_SHA256 = "96e2ec5e1fad2e4a955f15f1b0ec58d0ab748dc582cb5ed172f6b55ee42bacb3"

DIMENSIONS = (
    "entity-reference", "relation-direction", "numeric-value-and-unit",
    "polarity-and-negation", "quantifier-and-cardinality", "temporal-scope",
    "epistemic-modality", "attribution-and-source", "causal-strength",
    "unsupported-additions", "required-detail-coverage",
)
OUTCOMES = ["supported", "contradicted", "indeterminate"]

TOP_FIELDS = {
    "schema", "authority", "subject", "semantic_contract_profile",
    "comparison_oracle_schema", "dimensions", "corpus", "case_families",
    "human_adjudication", "candidate_protocol", "outcomes", "abstention",
    "confirmatory_metrics", "confirmatory_gates", "negative_controls",
    "reporting", "promotion_boundary", "evaluation_binding",
}
CORPUS_FIELDS = {
    "unit", "source_claims_per_case_min", "source_claims_per_case_max",
    "calibration_cases_per_dimension", "confirmatory_cases_per_dimension",
    "calibration_positive_fraction", "confirmatory_positive_fraction",
    "cross_dimension_discourse_calibration_cases",
    "cross_dimension_discourse_confirmatory_cases",
    "minimum_total_calibration_cases", "minimum_total_confirmatory_cases",
    "case_id_policy", "confirmatory_labels_sealed_before_candidate_evaluation",
    "confirmatory_surface_text_sealed_before_candidate_evaluation",
    "no_case_reuse_across_splits", "no_template_reuse_across_splits",
    "no_named_entity_tuple_reuse_across_splits", "no_numeric_tuple_reuse_across_splits",
    "no_exact_sentence_reuse_across_splits", "negative_cases_change_exactly_one_semantic_factor",
    "positive_cases_preserve_all_semantic_factors", "case_assignment_seed_committed_before_generation",
    "case_assignment_seed_revealed_only_after_corpus_manifest_seal",
    "within_dimension_positive_negative_balance",
}
CASE_FAMILY_FIELDS = {"positive", "negative_single_factor", "discourse", "negative_family_dimension_map"}
ADJUDICATION_FIELDS = {
    "independent_annotators_per_case", "adjudicator_required_on_disagreement",
    "annotators_blind_to_candidate_extractor_output", "adjudicator_blind_to_candidate_extractor_identity",
    "canonical_claim_schema_required", "raw_annotation_and_resolution_history_retained",
}
CANDIDATE_FIELDS = {
    "extractor_identity_must_bind", "candidate_freeze_before_confirmatory_execution",
    "calibration_may_change_candidate", "confirmatory_results_may_not_change_candidate",
    "one_confirmatory_attempt_per_frozen_candidate", "surface_text_is_only_semantic_input",
    "ground_truth_inventory_hidden_during_extraction", "source_graph_hidden_during_extraction",
    "comparison_oracle_may_run_only_after_extraction_output_is_frozen",
    "candidate_must_not_emit_source_claim_ids", "candidate_output_schema",
}
ABSTENTION_FIELDS = {
    "explicit_abstention_maps_to", "parse_or_runtime_failure_maps_to",
    "indeterminate_never_counts_as_supported", "indeterminate_rate_reported_per_dimension",
}
METRIC_FIELDS = {
    "per_dimension_positive_preservation", "per_dimension_negative_sensitivity",
    "per_dimension_indeterminate_rate", "unsupported_addition_false_positive_rate",
    "required_detail_omission_rate", "macro_average_is_diagnostic_only",
    "no_weighted_scalar_authority", "per_dimension_exact_claim_precision",
    "per_dimension_exact_claim_recall", "per_dimension_expected_verdict_accuracy",
}
GATE_FIELDS = {
    "minimum_positive_preservation_each_dimension", "minimum_negative_sensitivity_each_dimension",
    "maximum_indeterminate_rate_each_dimension", "maximum_unsupported_addition_false_positive_rate",
    "maximum_required_detail_omission_rate", "zero_allowed_catastrophic_misses",
    "catastrophic_miss_definition", "all_gates_required", "minimum_exact_claim_precision_each_dimension",
    "minimum_exact_claim_recall_each_dimension", "minimum_expected_verdict_accuracy_each_dimension",
}
CONTROL_FIELDS = {
    "surface_permutation_control", "source_inventory_permutation_control", "unrelated_surface_control",
    "empty_surface_control", "instruction_injection_string_as_semantic_data_control",
    "same_tokens_different_scope_control",
}
REPORTING_FIELDS = {
    "all_case_level_results_retained", "all_dimension_confusion_counts_reported",
    "calibration_and_confirmatory_results_separate", "failed_cases_reported_without_cherry_picking",
    "candidate_runtime_failures_reported", "human_adjudication_disagreements_reported",
}
PROMOTION_FIELDS = {
    "benchmark_pass_does_not_establish_surface_fidelity", "benchmark_pass_does_not_establish_factual_truth",
    "benchmark_pass_does_not_establish_backend_identity", "benchmark_pass_does_not_grant_action_authority",
    "positive_runtime_capability_requires_separate_typed_composition",
}
EVALUATION_FIELDS = {
    "candidate_output_contains_source_claim_ids", "candidate_output_frozen_before_ground_truth_access",
    "candidate_claims_match_expected_by_exact_canonical_claim_content",
    "matched_candidate_claim_inherits_hidden_expected_source_alignment_only_after_freeze",
    "unmatched_candidate_claim_classification", "unmatched_expected_claim_classification",
    "ambiguous_duplicate_canonical_claims_rejected",
    "no_model_or_human_semantic_aligner_in_confirmatory_evaluation",
    "v17_comparison_runs_only_on_post-freeze_exactly_matched_alignment",
}

class PreregError(ValueError):
    pass


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise PreregError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"), object_pairs_hook=strict_object)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise PreregError(str(exc)) from exc
    if not isinstance(value, dict):
        raise PreregError("top-level value must be an object")
    return value


def fields(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreregError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        raise PreregError(f"{where} fields mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    return value


def require_true(value: Any, where: str) -> None:
    if value is not True:
        raise PreregError(f"{where} must be true")


def fraction(value: Any, expected: str, where: str) -> None:
    if value != expected:
        raise PreregError(f"{where} must remain {expected}")


def validate(document: dict[str, Any]) -> dict[str, Any]:
    fields(document, TOP_FIELDS, "top-level")
    if document["schema"] != SCHEMA or document["authority"] != AUTHORITY:
        raise PreregError("schema/authority drift")
    if document["subject"] != "natural-language-surface-to-structured-claim-extraction":
        raise PreregError("subject drift")
    if document["semantic_contract_profile"] != SEMANTIC_PROFILE or document["comparison_oracle_schema"] != ORACLE_SCHEMA:
        raise PreregError("upstream profile drift")

    dims = document["dimensions"]
    if not isinstance(dims, list) or len(dims) != 11:
        raise PreregError("exactly eleven dimensions are required")
    for code, name in enumerate(DIMENSIONS):
        if dims[code] != {"code": code, "name": name}:
            raise PreregError(f"dimension {code} drifted")
    if document["outcomes"] != OUTCOMES:
        raise PreregError("outcome taxonomy drift")

    corpus = fields(document["corpus"], CORPUS_FIELDS, "corpus")
    if corpus["source_claims_per_case_min"] != 1 or corpus["source_claims_per_case_max"] != 6:
        raise PreregError("claim-count bounds drift")
    if corpus["calibration_cases_per_dimension"] != 40 or corpus["confirmatory_cases_per_dimension"] != 64:
        raise PreregError("per-dimension corpus floor drift")
    if corpus["cross_dimension_discourse_calibration_cases"] != 88 or corpus["cross_dimension_discourse_confirmatory_cases"] != 128:
        raise PreregError("discourse corpus floor drift")
    if corpus["minimum_total_calibration_cases"] != 528 or corpus["minimum_total_confirmatory_cases"] != 832:
        raise PreregError("total corpus floor drift")
    fraction(corpus["calibration_positive_fraction"], "1/2", "calibration positive fraction")
    fraction(corpus["confirmatory_positive_fraction"], "1/2", "confirmatory positive fraction")
    if corpus["within_dimension_positive_negative_balance"] != "1:1":
        raise PreregError("within-dimension balance drift")
    if corpus["case_id_policy"] != "content-addressed-after-human-adjudication":
        raise PreregError("case-id policy drift")
    for key in CORPUS_FIELDS - {
        "unit", "source_claims_per_case_min", "source_claims_per_case_max",
        "calibration_cases_per_dimension", "confirmatory_cases_per_dimension",
        "calibration_positive_fraction", "confirmatory_positive_fraction",
        "cross_dimension_discourse_calibration_cases", "cross_dimension_discourse_confirmatory_cases",
        "minimum_total_calibration_cases", "minimum_total_confirmatory_cases", "case_id_policy",
        "within_dimension_positive_negative_balance",
    }:
        require_true(corpus[key], f"corpus.{key}")

    families = fields(document["case_families"], CASE_FAMILY_FIELDS, "case_families")
    for name in ("positive", "negative_single_factor", "discourse"):
        values = families[name]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise PreregError(f"{name} families must be a unique list")
    family_map = families["negative_family_dimension_map"]
    if not isinstance(family_map, dict) or set(family_map) != set(families["negative_single_factor"]):
        raise PreregError("negative-family map key drift")
    if set(family_map.values()) != set(DIMENSIONS):
        raise PreregError("negative families must cover every dimension")

    adjudication = fields(document["human_adjudication"], ADJUDICATION_FIELDS, "human_adjudication")
    if adjudication["independent_annotators_per_case"] != 2:
        raise PreregError("two independent annotators are required")
    for key in ADJUDICATION_FIELDS - {"independent_annotators_per_case"}:
        require_true(adjudication[key], f"human_adjudication.{key}")

    candidate = fields(document["candidate_protocol"], CANDIDATE_FIELDS, "candidate_protocol")
    identity = candidate["extractor_identity_must_bind"]
    if set(identity) != {"implementation_digest", "model_or_ruleset_identity", "prompt_or_configuration_digest", "normalization_profile", "runtime_or_environment_identity"} or len(identity) != 5:
        raise PreregError("extractor identity binding drift")
    if candidate["candidate_output_schema"] != "surface-claim-inventory-without-source-alignment/v1":
        raise PreregError("candidate output schema drift")
    for key in CANDIDATE_FIELDS - {"extractor_identity_must_bind", "candidate_output_schema"}:
        require_true(candidate[key], f"candidate_protocol.{key}")

    abstention = fields(document["abstention"], ABSTENTION_FIELDS, "abstention")
    if abstention["explicit_abstention_maps_to"] != "indeterminate" or abstention["parse_or_runtime_failure_maps_to"] != "indeterminate":
        raise PreregError("abstention mapping drift")
    require_true(abstention["indeterminate_never_counts_as_supported"], "abstention.indeterminate_never_counts_as_supported")
    require_true(abstention["indeterminate_rate_reported_per_dimension"], "abstention.indeterminate_rate_reported_per_dimension")

    metrics = fields(document["confirmatory_metrics"], METRIC_FIELDS, "confirmatory_metrics")
    require_true(metrics["macro_average_is_diagnostic_only"], "confirmatory_metrics.macro_average_is_diagnostic_only")
    require_true(metrics["no_weighted_scalar_authority"], "confirmatory_metrics.no_weighted_scalar_authority")

    gates = fields(document["confirmatory_gates"], GATE_FIELDS, "confirmatory_gates")
    for key in ("minimum_positive_preservation_each_dimension", "minimum_negative_sensitivity_each_dimension", "minimum_exact_claim_precision_each_dimension", "minimum_exact_claim_recall_each_dimension", "minimum_expected_verdict_accuracy_each_dimension"):
        fraction(gates[key], "0.95", key)
    fraction(gates["maximum_indeterminate_rate_each_dimension"], "0.05", "maximum_indeterminate_rate_each_dimension")
    fraction(gates["maximum_unsupported_addition_false_positive_rate"], "0.02", "maximum_unsupported_addition_false_positive_rate")
    fraction(gates["maximum_required_detail_omission_rate"], "0.02", "maximum_required_detail_omission_rate")
    if set(gates["zero_allowed_catastrophic_misses"]) != {"polarity-and-negation", "numeric-value-and-unit", "attribution-and-source", "causal-strength"}:
        raise PreregError("catastrophic-miss set drift")
    if gates["catastrophic_miss_definition"] != "targeted negative case reported supported":
        raise PreregError("catastrophic-miss definition drift")
    require_true(gates["all_gates_required"], "confirmatory_gates.all_gates_required")

    for section, expected in (
        ("negative_controls", CONTROL_FIELDS), ("reporting", REPORTING_FIELDS), ("promotion_boundary", PROMOTION_FIELDS)
    ):
        obj = fields(document[section], expected, section)
        for key in expected:
            require_true(obj[key], f"{section}.{key}")

    evaluation = fields(document["evaluation_binding"], EVALUATION_FIELDS, "evaluation_binding")
    if evaluation["candidate_output_contains_source_claim_ids"] is not False:
        raise PreregError("source-claim-id leakage is forbidden")
    if evaluation["unmatched_candidate_claim_classification"] != "unsupported-addition" or evaluation["unmatched_expected_claim_classification"] != "required-detail-omission":
        raise PreregError("unmatched-claim classification drift")
    for key in EVALUATION_FIELDS - {"candidate_output_contains_source_claim_ids", "unmatched_candidate_claim_classification", "unmatched_expected_claim_classification"}:
        require_true(evaluation[key], f"evaluation_binding.{key}")

    canonical = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    semantic_sha = hashlib.sha256(canonical).hexdigest()
    if semantic_sha != EXPECTED_SEMANTIC_SHA256:
        raise PreregError(f"semantic preregistration identity drifted: {semantic_sha}")
    return {
        "schema": "symthaea.scip-text-claim-extraction-preregistration-validation/v1",
        "authority": AUTHORITY,
        "preregistration_sha256": semantic_sha,
        "dimension_count": 11,
        "minimum_total_calibration_cases": 528,
        "minimum_total_confirmatory_cases": 832,
        "confirmatory_execution_authorized": False,
        "surface_fidelity_established": False,
        "runtime_capability_authorized": False,
    }


def self_test() -> None:
    path = Path(__file__).resolve().parent / "qualification" / "scip_text_claim_extraction_preregistration_v1.json"
    document = load(path)
    result = validate(document)
    assert result["preregistration_sha256"] == EXPECTED_SEMANTIC_SHA256
    for section, key in (("candidate_protocol", "ground_truth_inventory_hidden_during_extraction"), ("promotion_boundary", "benchmark_pass_does_not_establish_surface_fidelity")):
        mutated = json.loads(json.dumps(document))
        mutated[section][key] = False
        try:
            validate(mutated)
        except PreregError:
            pass
        else:
            raise AssertionError(f"{section}.{key} mutation was accepted")
    print(f"PASS_PREREGISTRATION_SELF_TEST preregistration_sha256={EXPECTED_SEMANTIC_SHA256}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if not args.input:
                parser.error("input is required unless --self-test is used")
            print(json.dumps(validate(load(Path(args.input))), sort_keys=True, separators=(",", ":")))
        return 0
    except PreregError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
