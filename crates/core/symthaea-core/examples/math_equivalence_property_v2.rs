// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MATH-REP-001D1 staged evaluator.
//!
//! Default execution consumes only frozen development seeds. The separate
//! evaluation split requires explicit `--evaluation` or an ignored test.

use std::collections::BTreeMap;

use serde::Serialize;

mod adapter_v1 {
    include!("support/math_exact_polynomial_adapter_v1.rs");
}
mod adapter_v2 {
    include!("support/math_exact_polynomial_adapter_v2.rs");
}
mod property_generator {
    include!("support/math_equivalence_property_q0_v2.rs");
}

use adapter_v2::{normalize_term_uniform_domain, Disposition, NORMALIZER_ID};
use property_generator::{
    generate_pairs, generate_refusals, PairExpectation, AUTHORITY, DEV_SEEDS, EVAL_SEEDS,
    GENERATOR_ID,
};

pub const EVALUATOR_ID: &str = "math-equivalence-property-evaluator-v1";
pub const REPORT_VERSION: &str = "math-equivalence-property-report-v1";

#[derive(Debug, Clone, Default, Serialize)]
struct FamilyStats {
    total: usize,
    passed: usize,
    unexpected_normalization_errors: usize,
}

impl FamilyStats {
    fn record(&mut self, passed: bool, unexpected_normalization_error: bool) {
        self.total += 1;
        if passed {
            self.passed += 1;
        }
        if unexpected_normalization_error {
            self.unexpected_normalization_errors += 1;
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
struct EvaluationReport {
    same_normal_form: FamilyStats,
    different_normal_form: FamilyStats,
    refusal_contract: FamilyStats,
    pair_families: BTreeMap<String, FamilyStats>,
    refusal_families: BTreeMap<String, FamilyStats>,
}

impl EvaluationReport {
    fn all_passed(&self) -> bool {
        self.same_normal_form.total == self.same_normal_form.passed
            && self.different_normal_form.total == self.different_normal_form.passed
            && self.refusal_contract.total == self.refusal_contract.passed
    }

    fn total_cases(&self) -> usize {
        self.same_normal_form.total
            + self.different_normal_form.total
            + self.refusal_contract.total
    }

    fn passed_cases(&self) -> usize {
        self.same_normal_form.passed
            + self.different_normal_form.passed
            + self.refusal_contract.passed
    }
}

#[derive(Serialize)]
struct ReportEnvelope<'a> {
    version: &'static str,
    evaluator_id: &'static str,
    generator_id: &'static str,
    normalizer_id: &'static str,
    authority: &'static str,
    split: &'a str,
    seeds_hex: Vec<String>,
    total_cases: usize,
    passed_cases: usize,
    all_passed: bool,
    report: &'a EvaluationReport,
}

fn evaluate(seeds: &[u64]) -> EvaluationReport {
    let mut report = EvaluationReport::default();

    for &seed in seeds {
        for case in generate_pairs(seed) {
            let family = format!("{:?}", case.family);
            let lhs = normalize_term_uniform_domain(&case.lhs, case.lhs_domain);
            let rhs = normalize_term_uniform_domain(&case.rhs, case.rhs_domain);

            let (passed, unexpected_error) = match (lhs, rhs) {
                (Ok(lhs), Ok(rhs)) => {
                    let observed_same = lhs.canonical_serialization == rhs.canonical_serialization;
                    let expected_same = case.expectation == PairExpectation::SameNormalForm;
                    (observed_same == expected_same, false)
                }
                _ => (false, true),
            };

            match case.expectation {
                PairExpectation::SameNormalForm => {
                    report.same_normal_form.record(passed, unexpected_error)
                }
                PairExpectation::DifferentNormalForm => {
                    report
                        .different_normal_form
                        .record(passed, unexpected_error)
                }
            }
            report
                .pair_families
                .entry(family)
                .or_default()
                .record(passed, unexpected_error);
        }

        for case in generate_refusals(seed) {
            let family = format!("{:?}", case.family);
            let passed = match normalize_term_uniform_domain(&case.term, case.domain) {
                Ok(_) => false,
                Err(err) => {
                    let disposition = match err.disposition() {
                        Disposition::Unsupported => "Unsupported",
                        Disposition::Rejected => "Rejected",
                    };
                    disposition == case.expected_disposition
                        && err.receipt_rejection_reason() == case.expected_receipt_reason
                }
            };
            report.refusal_contract.record(passed, false);
            report
                .refusal_families
                .entry(family)
                .or_default()
                .record(passed, false);
        }
    }

    report
}

fn report_json(split: &str, seeds: &[u64], report: &EvaluationReport) -> String {
    let envelope = ReportEnvelope {
        version: REPORT_VERSION,
        evaluator_id: EVALUATOR_ID,
        generator_id: GENERATOR_ID,
        normalizer_id: NORMALIZER_ID,
        authority: AUTHORITY,
        split,
        seeds_hex: seeds.iter().map(|seed| format!("{seed:016x}")).collect(),
        total_cases: report.total_cases(),
        passed_cases: report.passed_cases(),
        all_passed: report.all_passed(),
        report,
    };
    serde_json::to_string_pretty(&envelope).expect("evaluation report must serialize")
}

fn main() {
    let evaluation = std::env::args().skip(1).any(|arg| arg == "--evaluation");
    let (split, seeds): (&str, &[u64]) = if evaluation {
        ("evaluation", &EVAL_SEEDS)
    } else {
        ("development", &DEV_SEEDS)
    };

    let report = evaluate(seeds);
    println!("{}", report_json(split, seeds, &report));
    if !report.all_passed() {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn evaluator_generator_and_normalizer_identity_are_frozen() {
        assert_eq!(EVALUATOR_ID, "math-equivalence-property-evaluator-v1");
        assert_eq!(GENERATOR_ID, "math-equivalence-property-q0-v2");
        assert_eq!(NORMALIZER_ID, "symthaea-exact-polynomial-term-v2");
        assert_eq!(AUTHORITY, "MeasurementOnly");
    }

    #[test]
    fn development_and_evaluation_seeds_are_disjoint() {
        let dev: BTreeSet<u64> = DEV_SEEDS.into_iter().collect();
        let eval: BTreeSet<u64> = EVAL_SEEDS.into_iter().collect();
        assert!(dev.is_disjoint(&eval));
    }

    #[test]
    fn development_split_matches_frozen_oracle() {
        let report = evaluate(&DEV_SEEDS);
        assert_eq!(report.same_normal_form.total, 392);
        assert_eq!(report.different_normal_form.total, 120);
        assert_eq!(report.refusal_contract.total, 96);
        assert_eq!(report.total_cases(), 608);
        assert!(
            report.all_passed(),
            "development split mismatch: {report:#?}"
        );
    }

    #[test]
    fn development_report_is_structured_and_identified() {
        let report = evaluate(&DEV_SEEDS);
        let encoded = report_json("development", &DEV_SEEDS, &report);
        assert!(encoded.contains(REPORT_VERSION));
        assert!(encoded.contains(EVALUATOR_ID));
        assert!(encoded.contains(GENERATOR_ID));
        assert!(encoded.contains(NORMALIZER_ID));
        assert!(encoded.contains("\"authority\": \"MeasurementOnly\""));
        assert!(encoded.contains("\"split\": \"development\""));
    }

    #[test]
    #[ignore = "explicit preregistered evaluation split; do not use for v2 tuning"]
    fn evaluation_split_matches_frozen_oracle() {
        let report = evaluate(&EVAL_SEEDS);
        assert_eq!(report.same_normal_form.total, 392);
        assert_eq!(report.different_normal_form.total, 120);
        assert_eq!(report.refusal_contract.total, 96);
        assert_eq!(report.total_cases(), 608);
        assert!(
            report.all_passed(),
            "evaluation split mismatch: {report:#?}"
        );
    }
}
