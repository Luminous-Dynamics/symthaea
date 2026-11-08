#!/usr/bin/env python3
"""Validate CEB's normalized external comparator observation registry.

This is a deterministic, offline data-contract check. It does not fetch sources,
execute benchmark tasks, run candidate code, or qualify a benchmark runner.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {path}: {error}") from error


def semantic_errors(payload: dict) -> list[str]:
    errors: list[str] = []
    sources = {row["source_id"]: row for row in payload["sources"]}
    comparators = {row["comparator_id"]: row for row in payload["comparators"]}
    observations = payload["observations"]

    if len(sources) != len(payload["sources"]):
        errors.append("duplicate source_id")
    if len({row["source_url"] for row in payload["sources"]}) != len(payload["sources"]):
        errors.append("duplicate source_url")
    if len(comparators) != len(payload["comparators"]):
        errors.append("duplicate comparator_id")
    if len({row["observation_id"] for row in observations}) != len(observations):
        errors.append("duplicate observation_id")

    used_sources: set[str] = set()
    used_comparators: set[str] = set()
    metric_system_keys: set[tuple[str, str, str]] = set()

    for comparator in payload["comparators"]:
        source_ids = [comparator["source_id"], *comparator["supporting_source_ids"]]
        for source_id in source_ids:
            source = sources.get(source_id)
            if source is None:
                errors.append(f"{comparator['comparator_id']}: unknown source {source_id}")
                continue
            used_sources.add(source_id)
            if comparator["lane_id"] not in source["supports"]:
                errors.append(
                    f"{comparator['comparator_id']}: source {source_id} does not support "
                    f"{comparator['lane_id']}"
                )
        denom = comparator["sample_denominator"]
        if denom["status"] == "reported" and denom["value"] is None:
            errors.append(f"{comparator['comparator_id']}: reported denominator has no value")
        if denom["status"] == "not_reported" and denom["value"] is not None:
            errors.append(f"{comparator['comparator_id']}: not_reported denominator has a value")

    for observation in observations:
        comparator = comparators.get(observation["comparator_id"])
        if comparator is None:
            errors.append(
                f"{observation['observation_id']}: unknown comparator "
                f"{observation['comparator_id']}"
            )
            continue
        used_comparators.add(comparator["comparator_id"])
        metric = observation["metric"]
        uncertainty = metric["uncertainty"]
        if uncertainty["applies_to_metric_id"] != metric["metric_id"]:
            errors.append(
                f"{observation['observation_id']}: uncertainty is attached to a different metric"
            )
        if uncertainty["applies_to_task_count"] != comparator["sample_denominator"]["value"]:
            errors.append(
                f"{observation['observation_id']}: uncertainty denominator differs from comparator denominator"
            )
        if uncertainty["status"] == "reported":
            if uncertainty["margin"] is None:
                errors.append(f"{observation['observation_id']}: reported uncertainty has no margin")
        elif uncertainty["margin"] is not None:
            errors.append(f"{observation['observation_id']}: unreported uncertainty has a margin")

        key = (
            comparator["comparator_id"],
            metric["metric_id"],
            observation["system"]["system_id"],
        )
        if key in metric_system_keys:
            errors.append(
                f"{observation['observation_id']}: duplicate metric/system observation within comparator"
            )
        metric_system_keys.add(key)

        if metric["value_status"] == "unlabeled" and metric["metric_id"] != "unlabeled_leaderboard_score":
            errors.append(f"{observation['observation_id']}: unlabeled value assigned a named metric")
        if metric["value_status"] == "qualitative" and metric["metric_id"] != "verifier_lifecycle_policy":
            errors.append(f"{observation['observation_id']}: qualitative finding assigned a numeric metric")

    if used_comparators != set(comparators):
        missing = sorted(set(comparators) - used_comparators)
        errors.append("comparators without observations: " + ", ".join(missing))
    if used_sources != set(sources):
        missing = sorted(set(sources) - used_sources)
        errors.append("registered sources not used by any comparator: " + ", ".join(missing))

    # Preserve the November 7 causal-method and current ReviewBench refresh.
    required_comparators = {
        "reviewbench-high-severity-recheck-2014": ("CEB-REVIEW", "reviewbench-leaderboard", "reported", 219),
        "sdr-ca-runtime-root-cause": ("CEB-DEBUG", "sdr-ca-paper", "not_reported", None),
        "unival-causal-fault-localization": ("CEB-DEBUG", "unival-paper", "reported", 835),
        "aacr-bench-repository-context": ("CEB-REVIEW", "aacr-bench-paper", "reported", 200),
    }
    for comparator_id, expected in required_comparators.items():
        comparator = comparators.get(comparator_id)
        if comparator is None:
            errors.append(f"missing current causal-method comparator {comparator_id}")
            continue
        lane, source_id, denominator_status, denominator_value = expected
        if comparator["lane_id"] != lane or comparator["source_id"] != source_id:
            errors.append(f"{comparator_id}: lane/source binding changed")
        denominator = comparator["sample_denominator"]
        if denominator["status"] != denominator_status or denominator["value"] != denominator_value:
            errors.append(f"{comparator_id}: denominator contract changed")

    required_observations = {
        "reviewbench-intent-agent-grounded-f1": ("reviewbench-high-severity-recheck-2014", "grounded_f1", 82.2),
        "reviewbench-intent-agent-grounded-precision": ("reviewbench-high-severity-recheck-2014", "grounded_precision", 92.0),
        "reviewbench-intent-agent-grounded-recall": ("reviewbench-high-severity-recheck-2014", "grounded_recall", 74.4),
        "sdr-ca-public-dataset-count": ("sdr-ca-runtime-root-cause", "evaluation_dataset_count", 3),
        "unival-defects4j-program-versions": ("unival-causal-fault-localization", "evaluated_program_versions", 835),
        "aacr-bench-annotation-coverage-increase": ("aacr-bench-repository-context", "annotation_coverage_increase", 285),
    }
    for observation_id, expected in required_observations.items():
        observation = next((row for row in observations if row["observation_id"] == observation_id), None)
        if observation is None:
            errors.append(f"missing required scoped observation {observation_id}")
            continue
        comparator_id, metric_id, value = expected
        if (
            observation["comparator_id"] != comparator_id
            or observation["metric"]["metric_id"] != metric_id
            or observation["metric"]["value"] != value
        ):
            errors.append(f"{observation_id}: metric/comparator/value binding changed")

    current_reviewbench = [
        row for row in observations
        if row["comparator_id"] == "reviewbench-high-severity-recheck-2014"
    ]
    if len(current_reviewbench) != 3:
        errors.append("current ReviewBench snapshot must preserve exactly F1, precision and recall rows")
    if any(
        "20:14 SAST" not in row["source_locator"]
        or "2026-10-20" not in row["system"]["configuration"]
        for row in current_reviewbench
    ):
        errors.append("current ReviewBench snapshot lost its observation time or model run date")

    aacr = next(
        (row for row in observations if row["observation_id"] == "aacr-bench-annotation-coverage-increase"),
        None
    )
    if aacr and not any("not model performance" in note for note in aacr["notes"]):
        errors.append("AACR-Bench 285% annotation coverage must not be represented as model performance")

    # Verifier generation and benchmark lifecycle integrity are distinct
    # evaluation comparators, not CEB review-performance scores.
    required_integrity_comparators = {
        "verifierbench-executable-verifier-generation": ("CEB-EVAL", "verifierbench-arxiv-2610-28679", "reported", 222),
        "benchmark-maintenance-pass-audit": ("CEB-EVAL", "unearned-pass-maintenance-arxiv-2609-34262", "reported", 3810),
        "benchmark-maintenance-unearned-pass-rates": ("CEB-EVAL", "unearned-pass-maintenance-arxiv-2609-34262", "not_reported", None),
    }
    for comparator_id, expected in required_integrity_comparators.items():
        comparator = comparators.get(comparator_id)
        if comparator is None:
            errors.append(f"missing verifier/benchmark-integrity comparator {comparator_id}")
            continue
        lane, source_id, denominator_status, denominator_value = expected
        denominator = comparator["sample_denominator"]
        if comparator["lane_id"] != lane or comparator["source_id"] != source_id:
            errors.append(f"{comparator_id}: lane/source binding changed")
        if denominator["status"] != denominator_status or denominator["value"] != denominator_value:
            errors.append(f"{comparator_id}: denominator contract changed")

    required_integrity_observations = {
        "verifierbench-task-count": ("verifierbench-executable-verifier-generation", "task_count", 222),
        "verifierbench-hidden-case-count": ("verifierbench-executable-verifier-generation", "hidden_case_count", 4993),
        "verifierbench-source-benchmark-count": ("verifierbench-executable-verifier-generation", "source_benchmark_count", 6),
        "verifierbench-paper-common-harness-model-count": ("verifierbench-executable-verifier-generation", "common_harness_model_count", 19),
        "verifierbench-project-model-harness-config-count": ("verifierbench-executable-verifier-generation", "model_harness_configuration_count", 36),
        "verifierbench-project-source-defect-exclusions": ("verifierbench-executable-verifier-generation", "source_defect_exclusion_count", 45),
        "verifierbench-attack-case-count": ("verifierbench-executable-verifier-generation", "attack_case_count", 798),
        "verifierbench-v4v-check-count": ("verifierbench-executable-verifier-generation", "v4v_check_count", 6),
        "verifierbench-positive-case-count": ("verifierbench-executable-verifier-generation", "positive_case_count", 1566),
        "verifierbench-negative-case-count": ("verifierbench-executable-verifier-generation", "negative_case_count", 3427),
        "verifierbench-alternative-positive-count": ("verifierbench-executable-verifier-generation", "alternate_positive_case_count", 1344),
        "verifierbench-reference-mutation-configuration-count": ("verifierbench-executable-verifier-generation", "mutation_test_configuration_count", 1410),
        "benchmark-maintenance-audited-trajectory-count": ("benchmark-maintenance-pass-audit", "passing_trajectory_count", 3810),
        "benchmark-maintenance-cohort-count": ("benchmark-maintenance-pass-audit", "model_benchmark_cohort_count", 29),
        "benchmark-maintenance-opus47-unearned-pass-rate": ("benchmark-maintenance-unearned-pass-rates", "confirmed_unearned_pass_rate", 24),
        "benchmark-maintenance-fable5-unearned-pass-rate": ("benchmark-maintenance-unearned-pass-rates", "confirmed_unearned_pass_rate", 73),
        "benchmark-maintenance-fable51-unearned-pass-rate": ("benchmark-maintenance-unearned-pass-rates", "confirmed_unearned_pass_rate", 11),
        "benchmark-maintenance-gpt6astra-unearned-pass-rate": ("benchmark-maintenance-unearned-pass-rates", "confirmed_unearned_pass_rate", 0),
    }
    for observation_id, expected in required_integrity_observations.items():
        row = next((item for item in observations if item["observation_id"] == observation_id), None)
        if row is None:
            errors.append(f"missing verifier/benchmark-integrity observation {observation_id}")
            continue
        comparator_id, metric_id, value = expected
        if (
            row["comparator_id"] != comparator_id
            or row["metric"]["metric_id"] != metric_id
            or row["metric"]["value"] != value
        ):
            errors.append(f"{observation_id}: comparator/metric/value binding changed")

    verifier_asymmetry = next(
        (row for row in observations if row["observation_id"] == "verifierbench-validity-asymmetry"),
        None,
    )
    if verifier_asymmetry is None or (
        verifier_asymmetry["metric"]["value_status"] != "qualitative"
        or verifier_asymmetry["metric"]["metric_id"] != "verifier_lifecycle_policy"
        or "10.6" not in verifier_asymmetry["metric"]["raw_display"]
        or "62.1" not in verifier_asymmetry["metric"]["raw_display"]
        or not any("reject every case" in note for note in verifier_asymmetry["notes"])
    ):
        errors.append("VerifierBench class-rate and all-reject findings must remain qualitative, not single-model scores")

    verifier_case_rows = {
        row["observation_id"]: row
        for row in observations
        if row["observation_id"] in {
            "verifierbench-hidden-case-count",
            "verifierbench-positive-case-count",
            "verifierbench-negative-case-count",
            "verifierbench-alternative-positive-count",
            "verifierbench-attack-case-count",
        }
    }
    if len(verifier_case_rows) == 5:
        hidden = verifier_case_rows["verifierbench-hidden-case-count"]["metric"]["value"]
        positive = verifier_case_rows["verifierbench-positive-case-count"]["metric"]["value"]
        negative = verifier_case_rows["verifierbench-negative-case-count"]["metric"]["value"]
        alternatives = verifier_case_rows["verifierbench-alternative-positive-count"]["metric"]["value"]
        attacks = verifier_case_rows["verifierbench-attack-case-count"]["metric"]["value"]
        if positive + negative != hidden:
            errors.append("VerifierBench PASS and FAIL labels do not sum to the hidden-case count")
        if alternatives > positive or attacks > negative:
            errors.append("VerifierBench alternative positives or attacks exceed their parent class")

    if "verifierbench-executable-verifier-generation" in comparators:
        caveats = " ".join(comparators["verifierbench-executable-verifier-generation"]["caveats"]).lower()
        if "not a pr-review" not in caveats or "36 model" not in caveats or "19 models" not in caveats:
            errors.append("VerifierBench must retain task-scope and 19-vs-36 configuration caveats")
        if "accept all valid alternatives" not in caveats:
            errors.append("VerifierBench must retain positive-validity acceptance caveat")
    if "benchmark-maintenance-pass-audit" in comparators:
        caveats = " ".join(comparators["benchmark-maintenance-pass-audit"]["caveats"]).lower()
        if "not independent tasks" not in caveats or "fresh-agent evaluation" not in caveats:
            errors.append("unearned-pass audit must retain cohort dependence and fresh-agent replay caveats")
    if "benchmark-maintenance-unearned-pass-rates" in comparators:
        caveats = " ".join(comparators["benchmark-maintenance-unearned-pass-rates"]["caveats"]).lower()
        if "not normalized" not in caveats or "not stated in the abstract" not in caveats:
            errors.append("unearned-pass rates must retain descriptive and denominator caveats")

    # Guard the current actual-causality / repair / partial-identifiability comparators.
    required_method_comparators = {
        "cbfl-softrex-defects4j": ("CEB-DEBUG", "cbfl-softrex-paper", 827),
        "causalrepair-defects4j": ("CEB-DEBUG", "causalrepair-paper", 829),
        "causalrepair-defects4j-trans": ("CEB-DEBUG", "causalrepair-paper", 483),
        "poscm-uai2026": ("CEB-DEBUG", "poscm-uai2026-paper", 2),
    }
    for comparator_id, expected in required_method_comparators.items():
        comparator = comparators.get(comparator_id)
        if comparator is None:
            errors.append(f"missing current causal method comparator {comparator_id}")
            continue
        lane, source_id, denominator = expected
        if comparator["lane_id"] != lane or comparator["source_id"] != source_id:
            errors.append(f"{comparator_id}: method comparator lane/source binding changed")
        if comparator["sample_denominator"]["status"] != "reported" or comparator["sample_denominator"]["value"] != denominator:
            errors.append(f"{comparator_id}: method comparator denominator changed")

    required_method_observations = {
        "cbfl-softrex-top1": ("cbfl-softrex-defects4j", "top_1_localizations", 103),
        "cbfl-blmu-top1": ("cbfl-softrex-defects4j", "top_1_localizations", 89),
        "cbfl-softrex-top10": ("cbfl-softrex-defects4j", "top_10_localizations", 301),
        "cbfl-blmu-top10": ("cbfl-softrex-defects4j", "top_10_localizations", 270),
        "cbfl-softrex-mrr": ("cbfl-softrex-defects4j", "mean_reciprocal_rank", 0.349),
        "cbfl-blmu-mrr": ("cbfl-softrex-defects4j", "mean_reciprocal_rank", 0.301),
        "causalrepair-d4j-correct-fixes": ("causalrepair-defects4j", "correct_patches", 313),
        "reinfix-d4j-correct-fixes": ("causalrepair-defects4j", "correct_patches", 236),
        "causalrepair-d4j-trans-correct-fixes": ("causalrepair-defects4j-trans", "correct_patches", 289),
        "poscm-uai26-simulators": ("poscm-uai2026", "external_simulator_count", 2),
    }
    for observation_id, expected in required_method_observations.items():
        observation = next((row for row in observations if row["observation_id"] == observation_id), None)
        if observation is None:
            errors.append(f"missing required causal-method observation {observation_id}")
            continue
        comparator_id, metric_id, value = expected
        if (
            observation["comparator_id"] != comparator_id
            or observation["metric"]["metric_id"] != metric_id
            or observation["metric"]["value"] != value
        ):
            errors.append(f"{observation_id}: comparator/metric/value binding changed")

    # Preserve the method/task boundary: execution and repair results do not
    # become source-only review precision or Recall@5.
    if "causalrepair-defects4j" in comparators:
        caveats = " ".join(comparators["causalrepair-defects4j"]["caveats"]).lower()
        if "not ceb" not in caveats and "not a source-only" not in caveats:
            errors.append("CausalRepair comparator must retain repair-vs-review scope caveat")
    if "cbfl-softrex-defects4j" in comparators:
        caveats = " ".join(comparators["cbfl-softrex-defects4j"]["caveats"]).lower()
        if "not ceb defect recall@5" not in caveats:
            errors.append("SoftReX comparator must retain Top-N-vs-CEB Recall@5 scope caveat")

    # Preserve causal-guided repair and secure repository completion as distinct
    # contextual comparators; neither becomes a direct code-review equivalence.
    required_adjacent_comparators = {
        "cgarf-swebench-repair": ("CEB-DEBUG", "cgarf-paper", "not_reported", None),
        "secrepobench-secure-completion": ("CEB-SECURE", "secrepobench-leaderboard", "reported", 318),
    }
    for comparator_id, expected in required_adjacent_comparators.items():
        comparator = comparators.get(comparator_id)
        if comparator is None:
            errors.append(f"missing adjacent research comparator {comparator_id}")
            continue
        lane, source_id, denominator_status, denominator_value = expected
        denom = comparator["sample_denominator"]
        if comparator["lane_id"] != lane or comparator["source_id"] != source_id:
            errors.append(f"{comparator_id}: lane/source binding changed")
        if denom["status"] != denominator_status or denom["value"] != denominator_value:
            errors.append(f"{comparator_id}: denominator contract changed")

    required_adjacent_observations = {
        "cgarf-repair-success-gain-lower": ("cgarf-swebench-repair", "repair_success_gain_lower_bound", 2),
        "cgarf-repair-success-gain-upper": ("cgarf-swebench-repair", "repair_success_gain_upper_bound", 4),
        "cgarf-patch-applicability-gain": ("cgarf-swebench-repair", "patch_applicability_gain", 3),
        "secrepobench-top-setup-secure-pass-at-1": ("secrepobench-secure-completion", "secure_pass_at_1", 60.1),
        "secrepobench-top-setup-pass-at-1": ("secrepobench-secure-completion", "pass_at_1", 76.4),
    }
    for observation_id, expected in required_adjacent_observations.items():
        observation = next((row for row in observations if row["observation_id"] == observation_id), None)
        if observation is None:
            errors.append(f"missing adjacent research observation {observation_id}")
            continue
        comparator_id, metric_id, value = expected
        if (
            observation["comparator_id"] != comparator_id
            or observation["metric"]["metric_id"] != metric_id
            or observation["metric"]["value"] != value
        ):
            errors.append(f"{observation_id}: comparator/metric/value binding changed")

    if "cgarf-swebench-repair" in comparators:
        caveats = " ".join(comparators["cgarf-swebench-repair"]["caveats"]).lower()
        if "not ceb review" not in caveats and "not reviewer" not in caveats:
            errors.append("CGARF must retain its repair-vs-review caveat")
    if "secrepobench-secure-completion" in comparators:
        caveats = " ".join(comparators["secrepobench-secure-completion"]["caveats"]).lower()
        if "not code review" not in caveats or "not ceb-secure" not in caveats:
            errors.append("SecRepoBench must retain its secure-completion-vs-CEB caveat")

    # Requirement-conformance over-correction, academic-review grounding and
    # selective pairwise judging stay distinct from direct CEB review scores.
    required_claim_reliability_comparators = {
        "llm-overcorrection-paired-conformance": ("CEB-REVIEW", "llm-overcorrection-review-paper", "not_reported", None),
        "reviewgrounder-academic-peer-review": ("CEB-REVIEW", "reviewgrounder-acl2026-paper", "not_reported", None),
        "scope-selective-pairwise-judging": ("CEB-REVIEW", "scope-selective-judge-summary", "not_reported", None),
    }
    for comparator_id, expected in required_claim_reliability_comparators.items():
        comparator = comparators.get(comparator_id)
        if comparator is None:
            errors.append(f"missing claim-reliability comparator {comparator_id}")
            continue
        lane, source_id, denominator_status, denominator_value = expected
        denom = comparator["sample_denominator"]
        if comparator["lane_id"] != lane or comparator["source_id"] != source_id:
            errors.append(f"{comparator_id}: lane/source binding changed")
        if denom["status"] != denominator_status or denom["value"] != denominator_value:
            errors.append(f"{comparator_id}: denominator contract changed")

    required_claim_reliability_observations = {
        "overcorr-humaneval-full-fnr": ("llm-overcorrection-paired-conformance", "false_negative_rate_humaneval_full", 54.8),
        "overcorr-humaneval-filter-fnr": ("llm-overcorrection-paired-conformance", "false_negative_rate_humaneval_full_filter", 16.3),
        "overcorr-mbpp-full-fnr": ("llm-overcorrection-paired-conformance", "false_negative_rate_mbpp_full", 69.0),
        "overcorr-mbpp-filter-fnr": ("llm-overcorrection-paired-conformance", "false_negative_rate_mbpp_full_filter", 28.9),
        "overcorr-quixbugs-full-fnr": ("llm-overcorrection-paired-conformance", "false_negative_rate_quixbugs_full", 51.0),
        "overcorr-quixbugs-filter-fnr": ("llm-overcorrection-paired-conformance", "false_negative_rate_quixbugs_full_filter", 24.0),
        "overcorr-gpt4o-humaneval-full-fpr": ("llm-overcorrection-paired-conformance", "false_positive_rate_gpt4o_humaneval_full", 0.0),
        "overcorr-gpt4o-humaneval-filter-fpr": ("llm-overcorrection-paired-conformance", "false_positive_rate_gpt4o_humaneval_full_filter", 1.83),
        "reviewgrounder-rubric-dimensions": ("reviewgrounder-academic-peer-review", "rubric_dimension_count", 8),
        "scope-fdr-lower-bound": ("scope-selective-pairwise-judging", "accepted_judgement_fdr_lower_bound", 0.097),
        "scope-fdr-upper-bound": ("scope-selective-pairwise-judging", "accepted_judgement_fdr_upper_bound", 0.099),
        "scope-acceptance-multiplier": ("scope-selective-pairwise-judging", "accepted_judgement_coverage_multiplier_upper_bound", 2.4),
    }
    for observation_id, expected in required_claim_reliability_observations.items():
        observation = next((row for row in observations if row["observation_id"] == observation_id), None)
        if observation is None:
            errors.append(f"missing claim-reliability observation {observation_id}")
            continue
        comparator_id, metric_id, value = expected
        if (
            observation["comparator_id"] != comparator_id
            or observation["metric"]["metric_id"] != metric_id
            or observation["metric"]["value"] != value
        ):
            errors.append(f"{observation_id}: comparator/metric/value binding changed")

    if "llm-overcorrection-paired-conformance" in comparators:
        caveats = " ".join(comparators["llm-overcorrection-paired-conformance"]["caveats"]).lower()
        if "not pr review" not in caveats or "not ceb-review" not in caveats:
            errors.append("over-correction comparator must retain conformance-vs-PR-review metric caveat")
        if "two-sided error trade-off" not in caveats:
            errors.append("over-correction comparator must retain FNR/FPR trade-off caveat")
    if "reviewgrounder-academic-peer-review" in comparators:
        caveats = " ".join(comparators["reviewgrounder-academic-peer-review"]["caveats"]).lower()
        if "academic-paper review" not in caveats or "not source-code" not in caveats:
            errors.append("ReviewGrounder must retain academic-peer-review scope caveat")
    if "scope-selective-pairwise-judging" in comparators:
        caveats = " ".join(comparators["scope-selective-pairwise-judging"]["caveats"]).lower()
        if "not software code review" not in caveats or "exchangeability" not in caveats:
            errors.append("SCOPE must retain pairwise-judging and calibration-assumption caveats")

    # Continuity-adjacent incremental-analysis and change-aware test-selection
    # studies are contextual/methodological comparators, not CEB outcome scores.
    required_continuity_comparators = {
        "incremental-program-analysis-ase2025": ("CEB-CONTINUITY", "ase-ipa-2025-paper", "reported", 4084),
        "diffci-change-aware-test-selection-2026": ("CEB-CONTINUITY", "diffci-open-evidence-study", "reported", 2000),
    }
    for comparator_id, expected in required_continuity_comparators.items():
        comparator = comparators.get(comparator_id)
        if comparator is None:
            errors.append(f"missing continuity-adjacent comparator {comparator_id}")
            continue
        lane, source_id, denominator_status, denominator_value = expected
        denominator = comparator["sample_denominator"]
        if comparator["lane_id"] != lane or comparator["source_id"] != source_id:
            errors.append(f"{comparator_id}: lane/source binding changed")
        if denominator["status"] != denominator_status or denominator["value"] != denominator_value:
            errors.append(f"{comparator_id}: denominator contract changed")

    required_continuity_observations = {
        "ase-ipa-real-change-commit-count": ("incremental-program-analysis-ase2025", "program_change_count", 4084),
        "ase-ipa-project-count": ("incremental-program-analysis-ase2025", "project_count", 20),
        "ase-ipa-tool-count": ("incremental-program-analysis-ase2025", "evaluated_tool_count", 2),
        "diffci-historical-delta-count": ("diffci-change-aware-test-selection-2026", "historical_commit_delta_count", 2000),
        "diffci-repository-count": ("diffci-change-aware-test-selection-2026", "repository_count", 20),
        "diffci-selective-opportunity-rate": ("diffci-change-aware-test-selection-2026", "selective_test_opportunity_rate", 29.8),
        "diffci-whole-corpus-test-reduction": ("diffci-change-aware-test-selection-2026", "whole_corpus_test_reduction_vs_path_rule", 4.2),
        "diffci-job-recall": ("diffci-change-aware-test-selection-2026", "historical_ci_failure_job_recall", 96.4),
        "diffci-path-rule-job-recall": ("diffci-change-aware-test-selection-2026", "historical_ci_failure_job_recall", 70.7),
        "diffci-historical-failure-count": ("diffci-change-aware-test-selection-2026", "historical_ci_failure_count", 280),
        "diffci-measurable-mutation-count": ("diffci-change-aware-test-selection-2026", "measurable_mutation_count", 32),
    }
    for observation_id, expected in required_continuity_observations.items():
        observation = next((row for row in observations if row["observation_id"] == observation_id), None)
        if observation is None:
            errors.append(f"missing continuity-adjacent observation {observation_id}")
            continue
        comparator_id, metric_id, value = expected
        if (
            observation["comparator_id"] != comparator_id
            or observation["metric"]["metric_id"] != metric_id
            or observation["metric"]["value"] != value
        ):
            errors.append(f"{observation_id}: comparator/metric/value binding changed")

    if "incremental-program-analysis-ase2025" in comparators:
        caveats = " ".join(comparators["incremental-program-analysis-ase2025"]["caveats"]).lower()
        if "not ceb evidence-disposition" not in caveats or "peak memory" not in caveats:
            errors.append("incremental program analysis must retain scope and peak-memory caveats")
    if "diffci-change-aware-test-selection-2026" in comparators:
        caveats = " ".join(comparators["diffci-change-aware-test-selection-2026"]["caveats"]).lower()
        if "self-published" not in caveats or "not a reliability estimate" not in caveats:
            errors.append("DiffCI must retain self-published and mutation-count limitations")
        if "no production ci runs were skipped" not in caveats:
            errors.append("DiffCI must retain no-production-skip caveat")

    # Provenance, research-reporting and repository-evolution studies are
    # methodology comparators, not direct code-review performance rows.
    required_provenance_comparators = {
        "agent-evidence-tracing-provenance-survey": ("CEB-REVIEW", "agent-evidence-tracing-survey-2026", "not_reported", None),
        "llm-se-empirical-reproducibility-guidelines": ("CEB-REVIEW", "llm-se-empirical-guidelines-2026", "not_reported", None),
        "pr-change-impact-callgraph-history": ("CEB-REVIEW", "pr-change-impact-analysis-2025", "not_reported", None),
        "history-mined-change-propagation": ("CEB-CONTINUITY", "history-change-propagation-channels", "not_reported", None),
        "microservice-evolution-formal-verification-slr": ("CEB-CONTINUITY", "microservice-evolution-verification-slr-2026", "reported", 18),
    }
    for comparator_id, expected in required_provenance_comparators.items():
        comparator = comparators.get(comparator_id)
        if comparator is None:
            errors.append(f"missing provenance/evolution comparator {comparator_id}")
            continue
        lane, source_id, denominator_status, denominator_value = expected
        denominator = comparator["sample_denominator"]
        if comparator["lane_id"] != lane or comparator["source_id"] != source_id:
            errors.append(f"{comparator_id}: lane/source binding changed")
        if denominator["status"] != denominator_status or denominator["value"] != denominator_value:
            errors.append(f"{comparator_id}: denominator contract changed")

    required_provenance_observations = {
        "agent-trace-provenance-taxonomy-dimensions": ("agent-evidence-tracing-provenance-survey", "provenance_taxonomy_dimension_count", 6),
        "agent-trace-open-challenge-categories": ("agent-evidence-tracing-provenance-survey", "open_challenge_category_count", 6),
        "llm-se-guidelines-contributor-count": ("llm-se-empirical-reproducibility-guidelines", "contributor_count", 22),
        "llm-se-guidelines-study-type-count": ("llm-se-empirical-reproducibility-guidelines", "study_type_taxonomy_count", 7),
        "llm-se-guidelines-recommendation-count": ("llm-se-empirical-reproducibility-guidelines", "reporting_guideline_count", 8),
        "pr-impact-analysis-time-min": ("pr-change-impact-callgraph-history", "analysis_time_lower_bound", 7.4),
        "pr-impact-analysis-time-max": ("pr-change-impact-callgraph-history", "analysis_time_upper_bound", 22.43),
        "pr-impact-focus-group-enhancement-rating": ("pr-change-impact-callgraph-history", "review_enhancement_rating", 3.66),
        "pr-impact-risk-score-perceived-accuracy": ("pr-change-impact-callgraph-history", "risk_score_perceived_accuracy_rating", 3.2),
        "pr-impact-metric-agreement": ("pr-change-impact-callgraph-history", "selected_metric_agreement", 71),
        "microservice-evolution-initial-records-screened": ("microservice-evolution-formal-verification-slr", "records_screened", 1187),
        "microservice-evolution-primary-studies": ("microservice-evolution-formal-verification-slr", "primary_study_count", 13),
        "microservice-evolution-total-included-studies": ("microservice-evolution-formal-verification-slr", "included_study_count", 18),
    }
    for observation_id, expected in required_provenance_observations.items():
        observation = next((row for row in observations if row["observation_id"] == observation_id), None)
        if observation is None:
            errors.append(f"missing provenance/evolution observation {observation_id}")
            continue
        comparator_id, metric_id, value = expected
        if (
            observation["comparator_id"] != comparator_id
            or observation["metric"]["metric_id"] != metric_id
            or observation["metric"]["value"] != value
        ):
            errors.append(f"{observation_id}: comparator/metric/value binding changed")

    if "agent-evidence-tracing-provenance-survey" in comparators:
        caveats = " ".join(comparators["agent-evidence-tracing-provenance-survey"]["caveats"]).lower()
        if "not a software code-review benchmark" not in caveats or "private chain-of-thought" not in caveats:
            errors.append("agent provenance survey must retain scope and private-reasoning boundary")
    if "llm-se-empirical-reproducibility-guidelines" in comparators:
        caveats = " ".join(comparators["llm-se-empirical-reproducibility-guidelines"]["caveats"]).lower()
        if "not a code-review system" not in caveats or "reporting checklist" not in caveats:
            errors.append("LLM-SE guidelines must retain methodological-only scope")
    if "pr-change-impact-callgraph-history" in comparators:
        caveats = " ".join(comparators["pr-change-impact-callgraph-history"]["caveats"]).lower()
        if "not a large-scale code-review accuracy benchmark" not in caveats or "not proof of semantic claim-support completeness" not in caveats:
            errors.append("PR change-impact study must retain scope and graph-completeness caveats")
    if "microservice-evolution-formal-verification-slr" in comparators:
        caveats = " ".join(comparators["microservice-evolution-formal-verification-slr"]["caveats"]).lower()
        if "not a software-agent benchmark" not in caveats or "few shared datasets" not in caveats:
            errors.append("microservice SLR must retain review-scope and literature-gap caveats")

    # SWE-CI is a long-horizon maintenance comparator, not a review claim
    # disposition benchmark.
    sweci = comparators.get("swe-ci-long-horizon-maintenance")
    if sweci is None:
        errors.append("missing SWE-CI long-horizon maintenance comparator")
    else:
        if (
            sweci["lane_id"] != "CEB-CONTINUITY"
            or sweci["source_id"] != "swe-ci-maintenance-paper-2026"
            or sweci["sample_denominator"]["status"] != "reported"
            or sweci["sample_denominator"]["value"] != 100
        ):
            errors.append("SWE-CI lane/source/100-task denominator changed")
        caveats = " ".join(sweci["caveats"]).lower()
        if "not code-review claim-support adjudication" not in caveats or "not evidence-disposition macro-f1" not in caveats:
            errors.append("SWE-CI must retain maintenance-vs-evidence-continuity scope caveat")
    required_sweci_observations = {
        "sweci-maintenance-task-count": ("repository_maintenance_task_count", 100),
        "sweci-average-history-days": ("average_history_duration", 233),
        "sweci-average-commit-count": ("average_consecutive_commit_count", 71),
    }
    for observation_id, expected in required_sweci_observations.items():
        row = next((item for item in observations if item["observation_id"] == observation_id), None)
        if row is None:
            errors.append(f"missing SWE-CI observation {observation_id}")
            continue
        metric_id, value = expected
        if row["comparator_id"] != "swe-ci-long-horizon-maintenance" or row["metric"]["metric_id"] != metric_id or row["metric"]["value"] != value:
            errors.append(f"{observation_id}: SWE-CI metric/value binding changed")

    # These two splits must remain separate benchmark populations.
    full = comparators.get("swe-pro-v2-full")
    hard = comparators.get("swe-pro-v2-hard")
    if full and hard:
        if full["sample_denominator"]["value"] != 642:
            errors.append("SWE-Bench Pro V2 Full denominator drifted from 642")
        if hard["sample_denominator"]["value"] != 51:
            errors.append("SWE-Bench Pro V2 HARD denominator drifted from 51")
        if full["sample_denominator"]["value"] == hard["sample_denominator"]["value"]:
            errors.append("SWE-Bench Pro V2 Full and HARD denominators were collapsed")

    # Do not allow the best functional and best secure scores to be synthesized
    # from different configurations. Both real configurations must remain explicit.
    als = [
        row for row in observations
        if row["comparator_id"] == "agent-security-league"
    ]
    by_system: dict[str, dict[str, float]] = {}
    for row in als:
        by_system.setdefault(row["system"]["system_id"], {})[
            row["metric"]["metric_id"]
        ] = row["metric"]["value"]
    muse = by_system.get("muse-wonder-muse-code", {})
    fable = by_system.get("fable51-claude-code", {})
    if muse.get("functional_pass") != 91.6 or muse.get("secure_pass") != 25.1:
        errors.append("Muse Code/Muse Wonder functional and secure observations drifted or were mixed")
    if fable.get("functional_pass") != 86.6 or fable.get("secure_pass") != 36.9:
        errors.append("Claude Code/Fable 5.1 functional and secure observations drifted or were mixed")

    agent = [
        row for row in observations
        if row["comparator_id"] == "agentabstain"
        and row["system"]["system_id"] == "gemini31-google-adk-agentabstain"
    ]
    agent_metrics = {row["metric"]["metric_id"]: row["metric"]["value"] for row in agent}
    if agent_metrics != {
        "paired_accuracy": 59.5,
        "act_accuracy": 90.5,
        "abstain_accuracy": 65.4,
        "conditioned_abstention_rate": 65.7,
    }:
        errors.append("AgentAbstain paired/act/abstain/CAR observations drifted or were conflated")

    hil = [
        row for row in observations
        if row["comparator_id"] == "hil-bench"
        and row["observation_id"] == "hil-muse-unlabeled-score"
    ]
    if hil and (
        hil[0]["metric"]["value_status"] != "unlabeled"
        or hil[0]["metric"]["metric_id"] != "unlabeled_leaderboard_score"
    ):
        errors.append("HiL-Bench unlabeled score was assigned named metric semantics")

    # HiL-Bench has four separate headline observations; none may be
    # reconstructed from the unlabeled consolidated Muse Wonder row.
    expected_hil = {
        "hil-best-ask-f1": ("hil-bench", "ask_f1", 44.0, 300),
        "hil-opus-pass3": ("hil-bench", "pass_at_3", 24.0, 300),
        "hil-swe-blocker-recall": ("hil-bench-swe", "blocker_recall", 36.0, 150),
        "hil-sql-blocker-recall": ("hil-bench-sql", "blocker_recall", 61.0, 150),
    }
    for observation_id, (comparator_id, metric_id, value, denominator) in expected_hil.items():
        row = next((x for x in observations if x["observation_id"] == observation_id), None)
        if row is None:
            errors.append(f"missing HiL-Bench observation {observation_id}")
            continue
        comparator = comparators.get(comparator_id)
        if row["comparator_id"] != comparator_id or row["metric"]["metric_id"] != metric_id or row["metric"]["value"] != value:
            errors.append(f"HiL-Bench observation {observation_id} has incorrect metric identity or value")
        if comparator is None or comparator["sample_denominator"]["value"] != denominator:
            errors.append(f"HiL-Bench observation {observation_id} has incorrect task denominator")
    for comparator_id in ("hil-bench-swe", "hil-bench-sql"):
        comparator = comparators.get(comparator_id)
        if comparator is None or comparator["lane_id"] != "CEB-HUMAN":
            errors.append(f"HiL-Bench domain comparator {comparator_id} missing or assigned to wrong lane")

    return errors


def validate_payload(payload: dict, schema: dict, format_checker: FormatChecker) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=format_checker)
    schema_errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))
    if schema_errors:
        return [
            f"schema: {'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in schema_errors
        ]
    return semantic_errors(payload)


def self_test(payload: dict, schema: dict, format_checker: FormatChecker) -> None:
    def must_reject(mutant: dict, name: str) -> None:
        if not validate_payload(mutant, schema, format_checker):
            raise SystemExit(f"negative control unexpectedly passed: {name}")

    # A score row cannot carry multiple system configurations.
    mutant = copy.deepcopy(payload)
    mutant["observations"][0]["systems"] = [mutant["observations"][0]["system"]]
    must_reject(mutant, "mixed system configurations")

    # A source reference must resolve and support the comparator's lane.
    mutant = copy.deepcopy(payload)
    mutant["comparators"][0]["source_id"] = "missing-source"
    must_reject(mutant, "unknown comparator source")

    # A benchmark split cannot borrow another split's denominator.
    mutant = copy.deepcopy(payload)
    mutant["comparators"][1]["sample_denominator"]["value"] = 642
    must_reject(mutant, "collapsed SWE-Bench Pro Full/HARD denominator")

    # A reported interval must bind to its own metric and exact denominator.
    mutant = copy.deepcopy(payload)
    interval = next(row for row in mutant["observations"] if row["observation_id"] == "pdb-single-opus-precision")
    interval["metric"]["uncertainty"]["applies_to_metric_id"] = "bug_recall"
    must_reject(mutant, "confidence interval attached to a different metric")

    mutant = copy.deepcopy(payload)
    interval = next(row for row in mutant["observations"] if row["observation_id"] == "pdb-single-opus-precision")
    interval["metric"]["uncertainty"]["applies_to_task_count"] = 484
    must_reject(mutant, "confidence interval attached to a different denominator")

    # The 79.67 leaderboard value remains unlabeled even after ASK-F1/Pass@3 were normalized.
    mutant = copy.deepcopy(payload)
    hil_row = next(row for row in mutant["observations"] if row["observation_id"] == "hil-muse-unlabeled-score")
    hil_row["metric"]["metric_id"] = "ask_f1"
    must_reject(mutant, "HiL unlabeled consolidated score mapped to ASK-F1")

    # The separate HiL task metrics must not be confused.
    mutant = copy.deepcopy(payload)
    ask_row = next(row for row in mutant["observations"] if row["observation_id"] == "hil-best-ask-f1")
    ask_row["metric"]["metric_id"] = "pass_at_3"
    must_reject(mutant, "HiL ASK-F1 relabeled as Pass@3")

    # An unlabeled number cannot be relabeled as ASK-F1 or Pass@3.
    mutant = copy.deepcopy(payload)
    hil = next(row for row in mutant["observations"] if row["observation_id"] == "hil-muse-unlabeled-score")
    hil["metric"]["metric_id"] = "ask_f1"
    must_reject(mutant, "unlabeled HiL score assigned a named metric")

    # A synthetic composite of unrelated maxima must not be introduced.
    mutant = copy.deepcopy(payload)
    mutant["observations"][0]["derived_from_observation_ids"] = [
        "swepro-full-opus-resolve", "contextbench-openjiuwen-f1"
    ]
    must_reject(mutant, "composite score synthesized from unrelated maxima")

    # Observation identities and source dates are part of the evidence record.
    mutant = copy.deepcopy(payload)
    mutant["observations"][1]["observation_id"] = mutant["observations"][0]["observation_id"]
    must_reject(mutant, "duplicate observation identity")

    mutant = copy.deepcopy(payload)
    mutant["sources"][0]["accessed_on"] = "not-a-date"
    must_reject(mutant, "missing/invalid source retrieval date")

    # The security benchmark's functional and secure maxima must stay tied to
    # their actual, different configurations.
    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "als-fable-secpass")
    row["system"]["system_id"] = "muse-wonder-muse-code"
    must_reject(mutant, "functional/security maxima mixed across configurations")

    # A matched/SOTA claim is not available from a contextual external record.
    mutant = copy.deepcopy(payload)
    mutant["comparators"][0]["comparison_class"] = "beats_sota"
    must_reject(mutant, "unsupported direct superiority claim")

    mutant = copy.deepcopy(payload)
    mutant["observations"] = [
        row for row in mutant["observations"]
        if row["observation_id"] != "reviewbench-intent-agent-grounded-recall"
    ]
    must_reject(mutant, "current ReviewBench snapshot observation removed")

    mutant = copy.deepcopy(payload)
    mutant["observations"] = [
        row for row in mutant["observations"]
        if row["observation_id"] != "sdr-ca-public-dataset-count"
    ]
    must_reject(mutant, "causal runtime comparator observation removed")

    mutant = copy.deepcopy(payload)
    mutant["comparators"] = [
        row for row in mutant["comparators"] if row["comparator_id"] != "cbfl-softrex-defects4j"
    ]
    must_reject(mutant, "actual-causality comparator removed")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "cbfl-softrex-top1")
    row["metric"]["value"] = 89
    must_reject(mutant, "SoftReX Top-1 conflated with BLMu")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "causalrepair-d4j-trans-correct-fixes")
    row["comparator_id"] = "causalrepair-defects4j"
    must_reject(mutant, "Defects4J-Trans task population collapsed into original Defects4J")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "poscm-uai26-simulators")
    row["metric"]["value"] = 95
    must_reject(mutant, "POSCM simulator count misrepresented as a performance score")

    mutant = copy.deepcopy(payload)
    mutant["observations"] = [
        row for row in mutant["observations"]
        if row["observation_id"] != "cgarf-repair-success-gain-upper"
    ]
    must_reject(mutant, "CGARF repair gain range upper bound removed")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "secrepobench-top-setup-secure-pass-at-1")
    row["metric"]["value"] = 76.4
    must_reject(mutant, "SecRepoBench secure-pass conflated with functional pass")

    mutant = copy.deepcopy(payload)
    mutant["observations"] = [
        row for row in mutant["observations"]
        if row["observation_id"] != "overcorr-humaneval-filter-fnr"
    ]
    must_reject(mutant, "Fix-guided filter HumanEval over-correction observation removed")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "overcorr-gpt4o-humaneval-filter-fpr")
    row["metric"]["value"] = 0.0
    must_reject(mutant, "Fix-guided filter FPR trade-off erased")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "reviewgrounder-rubric-dimensions")
    row["metric"]["value"] = 12
    must_reject(mutant, "ReviewGrounder rubric dimension descriptor changed")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "scope-fdr-upper-bound")
    row["metric"]["value"] = 0.25
    must_reject(mutant, "SCOPE FDR bound inflated")

    mutant = copy.deepcopy(payload)
    mutant["observations"] = [
        row for row in mutant["observations"]
        if row["observation_id"] != "ase-ipa-real-change-commit-count"
    ]
    must_reject(mutant, "incremental program-analysis corpus count removed")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "diffci-whole-corpus-test-reduction")
    row["metric"]["value"] = 96.4
    must_reject(mutant, "DiffCI whole-corpus test reduction conflated with job recall")

    mutant = copy.deepcopy(payload)
    comparator = next(row for row in mutant["comparators"] if row["comparator_id"] == "diffci-change-aware-test-selection-2026")
    comparator["sample_denominator"]["value"] = 280
    must_reject(mutant, "DiffCI historical commit-delta denominator collapsed into CI failure count")

    mutant = copy.deepcopy(payload)
    comparator = next(row for row in mutant["comparators"] if row["comparator_id"] == "diffci-change-aware-test-selection-2026")
    comparator["caveats"] = [x for x in comparator["caveats"] if "not a reliability estimate" not in x]
    must_reject(mutant, "DiffCI mutation caveat removed")

    mutant = copy.deepcopy(payload)
    mutant["observations"] = [
        row for row in mutant["observations"]
        if row["observation_id"] != "agent-trace-provenance-taxonomy-dimensions"
    ]
    must_reject(mutant, "agent provenance taxonomy observation removed")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "pr-impact-analysis-time-max")
    row["metric"]["value"] = 7.4
    must_reject(mutant, "PR impact analysis runtime upper/lower bounds conflated")

    mutant = copy.deepcopy(payload)
    comparator = next(row for row in mutant["comparators"] if row["comparator_id"] == "microservice-evolution-formal-verification-slr")
    comparator["sample_denominator"]["value"] = 1187
    must_reject(mutant, "systematic review search yield conflated with included-study denominator")

    mutant = copy.deepcopy(payload)
    comparator = next(row for row in mutant["comparators"] if row["comparator_id"] == "history-mined-change-propagation")
    comparator["sample_denominator"] = {
        "status": "reported",
        "value": 5032,
        "unit": "project versions",
    }
    must_reject(mutant, "evolution-history dataset size conflated with task denominator")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "sweci-average-history-days")
    row["metric"]["value"] = 71
    must_reject(mutant, "SWE-CI history duration conflated with commit count")

    mutant = copy.deepcopy(payload)
    comparator = next(row for row in mutant["comparators"] if row["comparator_id"] == "swe-ci-long-horizon-maintenance")
    comparator["sample_denominator"]["value"] = 71
    must_reject(mutant, "SWE-CI commit-history depth conflated with task count")

    mutant = copy.deepcopy(payload)
    mutant["observations"] = [
        row for row in mutant["observations"]
        if row["observation_id"] != "verifierbench-v4v-check-count"
    ]
    must_reject(mutant, "VerifierBench V4V audit-size observation removed")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "verifierbench-positive-case-count")
    row["metric"]["value"] = 1565
    must_reject(mutant, "VerifierBench positive-case denominator changed")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "verifierbench-project-model-harness-config-count")
    row["metric"]["value"] = 19
    must_reject(mutant, "VerifierBench project configuration count conflated with paper common-harness model count")

    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "benchmark-maintenance-fable5-unearned-pass-rate")
    row["metric"]["value"] = 24
    must_reject(mutant, "unearned-pass model cohort rates conflated")

    mutant = copy.deepcopy(payload)
    comparator = next(row for row in mutant["comparators"] if row["comparator_id"] == "benchmark-maintenance-pass-audit")
    comparator["sample_denominator"]["value"] = 29
    must_reject(mutant, "unearned-pass trajectory count conflated with cohort count")

    print("CEB_EXTERNAL_OBSERVATION_SELF_TEST_PASS")


def validate_pilot_payload(pilot: dict, schema: dict, format_checker: FormatChecker) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=format_checker)
    schema_errors = sorted(validator.iter_errors(pilot), key=lambda error: list(error.absolute_path))
    if schema_errors:
        return [
            f"pilot schema: {'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in schema_errors
        ]
    errors: list[str] = []
    families = pilot["population"]["root_cause_families"]
    if len({row["id"] for row in families}) != len(families):
        errors.append("pilot root-cause family IDs are not unique")
    if sum(row["target_tasks"] for row in families) != pilot["population"]["target_task_count"]:
        errors.append("pilot root-cause family task targets do not sum to the target task count")
    custody = pilot["task_custody"]
    if custody["development_partition_tasks"] + custody["heldout_partition_tasks"] != pilot["population"]["target_task_count"]:
        errors.append("pilot development and holdout partitions do not sum to the target task count")
    if pilot["statistics"]["independent_seeds_minimum"] < 3:
        errors.append("pilot minimum independent seed count is below three")
    if pilot["status"] != "design_only_not_executed":
        errors.append("pilot must remain design_only_not_executed")
    if pilot["execution_gate"]["candidate_controlled_execution"] != "blocked_until_issue_10064_qualified":
        errors.append("candidate-controlled execution gate was weakened")
    if pilot["execution_gate"]["independent_regrade_claim"] != "blocked_until_issue_10369_qualified":
        errors.append("independent regrade trust gate was weakened")
    return errors


def pilot_self_test(pilot: dict, schema: dict, format_checker: FormatChecker) -> None:
    def must_reject(mutant: dict, label: str) -> None:
        if not validate_pilot_payload(mutant, schema, format_checker):
            raise SystemExit(f"pilot negative control unexpectedly passed: {label}")

    mutant = copy.deepcopy(pilot)
    mutant["population"]["root_cause_families"][0]["target_tasks"] += 1
    must_reject(mutant, "root-cause strata do not sum to 48")

    mutant = copy.deepcopy(pilot)
    mutant["task_custody"]["heldout_partition_tasks"] = 15
    must_reject(mutant, "development/holdout partition is incomplete")

    mutant = copy.deepcopy(pilot)
    mutant["status"] = "qualified"
    must_reject(mutant, "design-only pilot marked qualified")

    mutant = copy.deepcopy(pilot)
    mutant["execution_gate"]["candidate_controlled_execution"] = "allowed"
    must_reject(mutant, "OS isolation gate weakened")

    mutant = copy.deepcopy(pilot)
    mutant["statistics"]["independent_seeds_minimum"] = 2
    must_reject(mutant, "insufficient independent seed count")

    mutant = copy.deepcopy(pilot)
    mutant["statistics"]["pilot_is_exploratory"] = False
    must_reject(mutant, "pilot converted into an unsupported superiority study")

    print("CEB_RUST_PILOT_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    base = root / "benchmarks" / "ceb"
    try:
        payload = load_json(base / "external-observations.v0.json")
        schema = load_json(base / "external-observations.schema.json")
        pilot = load_json(base / "rust-pilot-preregistration.v0.json")
        pilot_schema = load_json(base / "rust-pilot-preregistration.schema.json")
        Draft202012Validator.check_schema(schema)
        Draft202012Validator.check_schema(pilot_schema)
    except Exception as error:
        print(f"CEB_EXTERNAL_OBSERVATION_ERROR: {error}", file=sys.stderr)
        return 2
    format_checker = FormatChecker()
    if args.self_test:
        self_test(payload, schema, format_checker)
        pilot_self_test(pilot, pilot_schema, format_checker)
        return 0
    errors = validate_payload(payload, schema, format_checker)
    pilot_errors = validate_pilot_payload(pilot, pilot_schema, format_checker)
    if errors or pilot_errors:
        for error in errors:
            print(f"CEB_EXTERNAL_OBSERVATION_FAIL: {error}", file=sys.stderr)
        for error in pilot_errors:
            print(f"CEB_RUST_PILOT_FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "CEB_EXTERNAL_OBSERVATION_PASS: "
        f"sources={len(payload['sources'])} "
        f"comparators={len(payload['comparators'])} "
        f"observations={len(payload['observations'])}"
    )
    print(
        "CEB_RUST_PILOT_PASS: "
        f"tasks={pilot['population']['target_task_count']} "
        f"families={len(pilot['population']['root_cause_families'])} "
        f"arms={len(pilot['baseline_arms'])} "
        f"holdout={pilot['task_custody']['heldout_partition_tasks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
