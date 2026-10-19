// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic-first target selection with non-compensatory gates and robust interval Pareto.
//! Every fixture is invented; this module does not establish biological, regulatory, safety,
//! economic, or lifecycle validity.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub const RUNNER_VERSION: &str = "0.1.0";
pub const CORPUS_ID: &str = "BIOPROTEIN-TARGET-SELECTION-CORPUS-V1";
pub const SCHEMA_VERSION: &str = "1.0.0";
const CLAIM_CEILING: &str = "Synthetic target-selection and interval-Pareto contract only; all candidate data are fabricated fixtures. It does not establish biological performance, regulatory approval, food safety, commercial viability, TEA/LCA accuracy or environmental superiority.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateState {
    Pass,
    Fail,
    Unknown,
    NotAssessed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceLabel {
    Measured,
    Literature,
    Estimated,
    Simulated,
    Synthetic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    Valid,
    Incomparable,
    InvalidInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetDisposition {
    Eligible,
    Blocked,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateBundle {
    pub food_safety: GateState,
    pub authorization: GateState,
    pub source_rights: GateState,
    pub market_evidence: GateState,
    pub biology_function: GateState,
    pub process_closure: GateState,
    pub scale_transfer: GateState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetBrief {
    pub ingredient_class: String,
    pub intended_application: String,
    pub functional_job: String,
    pub product_spec_digest: String,
    pub quality_purity_basis: String,
    pub functional_unit: String,
    pub target_volume_low: i64,
    pub target_volume_high: i64,
    pub volume_unit: String,
    pub value_basis_id: String,
    pub evidence_vintage: String,
    pub biology_endpoint: String,
    pub process_closure_digest: String,
    pub scale_transfer_basis: String,
    pub tea_lca_boundary_digest: String,
    pub region_scenario_digest: String,
    pub feedstock_scenario_digest: String,
    pub electricity_scenario_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveInterval {
    pub objective_id: String,
    pub direction: ObjectiveDirection,
    pub unit: String,
    pub low: i64,
    pub high: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCandidate {
    pub target_id: String,
    pub brief: TargetBrief,
    pub evidence_label: EvidenceLabel,
    pub claim_basis: EvidenceLabel,
    pub gates: GateBundle,
    pub objectives: Vec<ObjectiveInterval>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedTargetEvaluation {
    pub target_id: String,
    pub disposition: TargetDisposition,
    pub reason_codes: Vec<String>,
    pub dominates: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCaseExpected {
    pub status: CaseStatus,
    pub reason_codes: Vec<String>,
    pub targets: Vec<ExpectedTargetEvaluation>,
    pub pareto_frontier_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCase {
    pub case_id: String,
    pub targets: Vec<TargetCandidate>,
    pub expected: TargetCaseExpected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCorpus {
    pub corpus_id: String,
    pub schema_version: String,
    pub case_count: usize,
    pub claim_ceiling: String,
    pub cases: Vec<TargetCase>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCaseEvaluation {
    pub case_id: String,
    pub status: CaseStatus,
    pub reason_codes: Vec<String>,
    pub targets: Vec<ExpectedTargetEvaluation>,
    pub pareto_frontier_ids: Vec<String>,
}

impl TargetCaseEvaluation {
    pub fn matches_expected(&self, expected: &TargetCaseExpected) -> bool {
        self.status == expected.status
            && self.reason_codes == expected.reason_codes
            && self.targets == expected.targets
            && self.pareto_frontier_ids == expected.pareto_frontier_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCaseReceipt {
    pub case_id: String,
    pub status: CaseStatus,
    pub reason_codes: Vec<String>,
    pub targets: Vec<ExpectedTargetEvaluation>,
    pub pareto_frontier_ids: Vec<String>,
    pub expected_contract_match: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetRunReceipt {
    pub runner_version: String,
    pub corpus_id: String,
    pub schema_version: String,
    pub corpus_sha256: String,
    pub declared_head: String,
    pub actual_head: String,
    pub exact_head_match: bool,
    pub case_count: usize,
    pub matched_case_count: usize,
    pub failed_case_ids: Vec<String>,
    pub cases: Vec<TargetCaseReceipt>,
    pub synthetic_only_claim_ceiling: String,
    pub contract_passed: bool,
}

fn is_nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_case(case: &TargetCase) -> Vec<String> {
    let mut errors = BTreeSet::<String>::new();
    if !is_nonempty(&case.case_id) {
        errors.insert("case_id_missing".to_owned());
    }
    if case.targets.is_empty() {
        errors.insert("targets_missing".to_owned());
    }
    let mut ids = BTreeSet::<String>::new();

    for target in &case.targets {
        if !is_nonempty(&target.target_id) || !ids.insert(target.target_id.clone()) {
            errors.insert("duplicate_or_empty_target_id".to_owned());
        }
        let b = &target.brief;
        let required_fields = [
            b.ingredient_class.as_str(),
            b.intended_application.as_str(),
            b.functional_job.as_str(),
            b.product_spec_digest.as_str(),
            b.quality_purity_basis.as_str(),
            b.functional_unit.as_str(),
            b.volume_unit.as_str(),
            b.value_basis_id.as_str(),
            b.evidence_vintage.as_str(),
            b.biology_endpoint.as_str(),
            b.process_closure_digest.as_str(),
            b.scale_transfer_basis.as_str(),
            b.tea_lca_boundary_digest.as_str(),
            b.region_scenario_digest.as_str(),
            b.feedstock_scenario_digest.as_str(),
            b.electricity_scenario_digest.as_str(),
        ];
        if required_fields.iter().any(|field| !is_nonempty(field)) {
            errors.insert("target_brief_field_missing".to_owned());
        }
        if b.target_volume_low < 0 || b.target_volume_high < b.target_volume_low {
            errors.insert("invalid_target_volume_interval".to_owned());
        }
        if target.evidence_label == EvidenceLabel::Synthetic
            && target.claim_basis == EvidenceLabel::Measured
        {
            errors.insert("synthetic_as_measured_claim".to_owned());
        } else if target.evidence_label != target.claim_basis {
            errors.insert("claim_basis_evidence_mismatch".to_owned());
        }
        if target.objectives.is_empty() {
            errors.insert("objectives_missing".to_owned());
        }
        let mut objective_ids = BTreeSet::<String>::new();
        for objective in &target.objectives {
            if !is_nonempty(&objective.objective_id)
                || !objective_ids.insert(objective.objective_id.clone())
            {
                errors.insert("duplicate_or_empty_objective_id".to_owned());
            }
            if !is_nonempty(&objective.unit) {
                errors.insert("objective_unit_missing".to_owned());
            }
            if objective.low > objective.high {
                errors.insert("invalid_objective_interval".to_owned());
            }
        }
        let required_objectives = [
            ("functional_fit", ObjectiveDirection::Maximize),
            ("compliant_capacity", ObjectiveDirection::Maximize),
            ("contribution_margin_per_kg", ObjectiveDirection::Maximize),
            ("cost_per_kg", ObjectiveDirection::Minimize),
            ("energy_per_kg", ObjectiveDirection::Minimize),
            ("water_per_kg", ObjectiveDirection::Minimize),
            ("days_to_ready", ObjectiveDirection::Minimize),
        ];
        for (objective_id, expected_direction) in required_objectives {
            match target
                .objectives
                .iter()
                .find(|objective| objective.objective_id == objective_id)
            {
                Some(objective) if objective.direction != expected_direction => {
                    errors.insert(format!(
                        "required_objective_direction_mismatch:{objective_id}"
                    ));
                }
                Some(_) => {}
                None => {
                    errors.insert(format!("required_objective_missing:{objective_id}"));
                }
            }
        }
        if let Some(capacity) = target
            .objectives
            .iter()
            .find(|objective| objective.objective_id == "compliant_capacity")
        {
            if capacity.unit != b.volume_unit {
                errors.insert("compliant_capacity_unit_mismatch".to_owned());
            }
        }
    }
    errors.into_iter().collect()
}

fn target_disposition(target: &TargetCandidate) -> ExpectedTargetEvaluation {
    let gates = [
        ("food_safety", target.gates.food_safety),
        ("authorization", target.gates.authorization),
        ("source_rights", target.gates.source_rights),
        ("market_evidence", target.gates.market_evidence),
        ("biology_function", target.gates.biology_function),
        ("process_closure", target.gates.process_closure),
        ("scale_transfer", target.gates.scale_transfer),
    ];
    let mut reasons = Vec::new();
    for (gate, state) in gates {
        match state {
            GateState::Pass => {}
            GateState::Fail => reasons.push(format!("{gate}_failed")),
            GateState::Unknown => reasons.push(format!("{gate}_unknown")),
            GateState::NotAssessed => reasons.push(format!("{gate}_not_assessed")),
        }
    }
    reasons.sort();
    let disposition = if reasons.iter().any(|reason| reason.ends_with("_failed")) {
        TargetDisposition::Blocked
    } else if !reasons.is_empty() {
        TargetDisposition::Indeterminate
    } else {
        TargetDisposition::Eligible
    };
    ExpectedTargetEvaluation {
        target_id: target.target_id.clone(),
        disposition,
        reason_codes: reasons,
        dominates: Vec::new(),
    }
}

fn same_comparison_context(a: &TargetCandidate, b: &TargetCandidate) -> bool {
    let x = &a.brief;
    let y = &b.brief;
    x.intended_application == y.intended_application
        && x.functional_job == y.functional_job
        && x.product_spec_digest == y.product_spec_digest
        && x.quality_purity_basis == y.quality_purity_basis
        && x.functional_unit == y.functional_unit
        && x.target_volume_low == y.target_volume_low
        && x.target_volume_high == y.target_volume_high
        && x.volume_unit == y.volume_unit
        && x.value_basis_id == y.value_basis_id
        && x.evidence_vintage == y.evidence_vintage
        && x.tea_lca_boundary_digest == y.tea_lca_boundary_digest
        && x.region_scenario_digest == y.region_scenario_digest
        && x.feedstock_scenario_digest == y.feedstock_scenario_digest
        && x.electricity_scenario_digest == y.electricity_scenario_digest
}

fn objective_signature(target: &TargetCandidate) -> Vec<(String, ObjectiveDirection, String)> {
    let mut signature: Vec<(String, ObjectiveDirection, String)> = target
        .objectives
        .iter()
        .map(|o| (o.objective_id.clone(), o.direction, o.unit.clone()))
        .collect();
    signature.sort();
    signature
}

fn robustly_dominates(a: &TargetCandidate, b: &TargetCandidate) -> bool {
    let b_by_id: std::collections::BTreeMap<&str, &ObjectiveInterval> = b
        .objectives
        .iter()
        .map(|objective| (objective.objective_id.as_str(), objective))
        .collect();
    let mut strictly_better = false;
    for objective_a in &a.objectives {
        let Some(objective_b) = b_by_id.get(objective_a.objective_id.as_str()) else {
            return false;
        };
        match objective_a.direction {
            ObjectiveDirection::Maximize => {
                if objective_a.low < objective_b.high {
                    return false;
                }
                strictly_better |= objective_a.low > objective_b.high;
            }
            ObjectiveDirection::Minimize => {
                if objective_a.high > objective_b.low {
                    return false;
                }
                strictly_better |= objective_a.high < objective_b.low;
            }
        }
    }
    strictly_better
}

pub fn evaluate_case(case: &TargetCase) -> TargetCaseEvaluation {
    let errors = validate_case(case);
    if !errors.is_empty() {
        return TargetCaseEvaluation {
            case_id: case.case_id.clone(),
            status: CaseStatus::InvalidInput,
            reason_codes: errors,
            targets: Vec::new(),
            pareto_frontier_ids: Vec::new(),
        };
    }

    let mut targets: Vec<ExpectedTargetEvaluation> =
        case.targets.iter().map(target_disposition).collect();
    targets.sort_by(|a, b| a.target_id.cmp(&b.target_id));

    let context_mismatch = case.targets.iter().enumerate().any(|(i, a)| {
        case.targets
            .iter()
            .skip(i + 1)
            .any(|b| !same_comparison_context(a, b))
    });
    let objective_mismatch = case.targets.iter().enumerate().any(|(i, a)| {
        case.targets
            .iter()
            .skip(i + 1)
            .any(|b| objective_signature(a) != objective_signature(b))
    });
    if context_mismatch || objective_mismatch {
        let mut reasons = Vec::new();
        if context_mismatch {
            reasons.push("comparison_context_mismatch".to_owned());
        }
        if objective_mismatch {
            reasons.push("objective_contract_mismatch".to_owned());
        }
        reasons.sort();
        return TargetCaseEvaluation {
            case_id: case.case_id.clone(),
            status: CaseStatus::Incomparable,
            reason_codes: reasons,
            targets,
            pareto_frontier_ids: Vec::new(),
        };
    }

    for left_index in 0..targets.len() {
        if targets[left_index].disposition != TargetDisposition::Eligible {
            continue;
        }
        let left_id = targets[left_index].target_id.clone();
        let left = case
            .targets
            .iter()
            .find(|target| target.target_id == left_id)
            .expect("validated candidate IDs exist");
        let mut dominated = Vec::new();
        for right in &case.targets {
            if right.target_id == left_id {
                continue;
            }
            let right_eval = targets
                .iter()
                .find(|evaluation| evaluation.target_id == right.target_id)
                .expect("all candidates have an evaluation");
            if right_eval.disposition == TargetDisposition::Eligible
                && robustly_dominates(left, right)
            {
                dominated.push(right.target_id.clone());
            }
        }
        dominated.sort();
        targets[left_index].dominates = dominated;
    }

    let pareto_frontier_ids: Vec<String> = targets
        .iter()
        .filter(|target| {
            target.disposition == TargetDisposition::Eligible
                && !targets
                    .iter()
                    .any(|other| other.dominates.contains(&target.target_id))
        })
        .map(|target| target.target_id.clone())
        .collect();
    TargetCaseEvaluation {
        case_id: case.case_id.clone(),
        status: CaseStatus::Valid,
        reason_codes: Vec::new(),
        targets,
        pareto_frontier_ids,
    }
}

pub fn evaluate_corpus(corpus: &TargetCorpus) -> Vec<TargetCaseEvaluation> {
    corpus.cases.iter().map(evaluate_case).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn run_cli() -> i32 {
    let mut expected_head: Option<String> = None;
    let mut corpus_path: Option<PathBuf> = None;
    let mut receipt_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--expected-head" => expected_head = args.next(),
            "--corpus" => corpus_path = args.next().map(PathBuf::from),
            "--receipt" => receipt_path = args.next().map(PathBuf::from),
            _ => {
                eprintln!("unknown argument: {arg}");
                return 2;
            }
        }
    }
    let Some(expected_head) = expected_head else {
        eprintln!("missing --expected-head");
        return 2;
    };
    let Some(corpus_path) = corpus_path else {
        eprintln!("missing --corpus");
        return 2;
    };
    let Some(receipt_path) = receipt_path else {
        eprintln!("missing --receipt");
        return 2;
    };

    let actual_head = match Command::new("git").args(["rev-parse", "HEAD"]).output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }
        _ => {
            eprintln!("could not establish actual Git HEAD");
            return 2;
        }
    };
    if actual_head != expected_head {
        eprintln!("exact-head mismatch: expected={expected_head} actual={actual_head}");
        return 2;
    }
    let corpus_bytes = match fs::read(&corpus_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read corpus {}: {error}", corpus_path.display());
            return 2;
        }
    };
    let corpus: TargetCorpus = match serde_json::from_slice(&corpus_bytes) {
        Ok(corpus) => corpus,
        Err(error) => {
            eprintln!("cannot parse target-selection corpus: {error}");
            return 2;
        }
    };
    if corpus.corpus_id != CORPUS_ID
        || corpus.schema_version != SCHEMA_VERSION
        || corpus.case_count != corpus.cases.len()
        || corpus.cases.is_empty()
    {
        eprintln!("unexpected corpus ID/schema or inconsistent case count");
        return 2;
    }

    let evaluations = evaluate_corpus(&corpus);
    let cases: Vec<TargetCaseReceipt> = corpus
        .cases
        .iter()
        .zip(evaluations.iter())
        .map(|(case, evaluation)| TargetCaseReceipt {
            case_id: evaluation.case_id.clone(),
            status: evaluation.status,
            reason_codes: evaluation.reason_codes.clone(),
            targets: evaluation.targets.clone(),
            pareto_frontier_ids: evaluation.pareto_frontier_ids.clone(),
            expected_contract_match: evaluation.matches_expected(&case.expected),
        })
        .collect();
    let failed_case_ids: Vec<String> = cases
        .iter()
        .filter(|receipt| !receipt.expected_contract_match)
        .map(|receipt| receipt.case_id.clone())
        .collect();
    let matched_case_count = cases.len() - failed_case_ids.len();
    let contract_passed = failed_case_ids.is_empty();
    let receipt = TargetRunReceipt {
        runner_version: RUNNER_VERSION.to_owned(),
        corpus_id: corpus.corpus_id,
        schema_version: corpus.schema_version,
        corpus_sha256: sha256_hex(&corpus_bytes),
        declared_head: expected_head.clone(),
        exact_head_match: actual_head == expected_head,
        actual_head,
        case_count: cases.len(),
        matched_case_count,
        failed_case_ids,
        cases,
        synthetic_only_claim_ceiling: CLAIM_CEILING.to_owned(),
        contract_passed,
    };
    if let Some(parent) = receipt_path.parent()
        && let Err(error) = fs::create_dir_all(parent)
    {
        eprintln!("cannot create receipt directory: {error}");
        return 2;
    }
    let receipt_bytes = match serde_json::to_vec_pretty(&receipt) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot serialize receipt: {error}");
            return 2;
        }
    };
    if let Err(error) = fs::write(&receipt_path, &receipt_bytes) {
        eprintln!("cannot write receipt {}: {error}", receipt_path.display());
        return 2;
    }
    println!("{}", String::from_utf8_lossy(&receipt_bytes));
    if contract_passed { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    const CORPUS_JSON: &str = include_str!(
        "../../../docs/research/bioprotein-experiments/target-selection-corpus-v1.json"
    );
    fn fixture() -> TargetCorpus {
        serde_json::from_str(CORPUS_JSON).expect("frozen target corpus parses")
    }

    #[test]
    fn frozen_corpus_matches_all_dispositions_and_frontiers() {
        let corpus = fixture();
        assert_eq!(corpus.corpus_id, CORPUS_ID);
        assert_eq!(corpus.schema_version, SCHEMA_VERSION);
        assert_eq!(corpus.case_count, corpus.cases.len());
        for (case, actual) in corpus.cases.iter().zip(evaluate_corpus(&corpus)) {
            assert!(
                actual.matches_expected(&case.expected),
                "case {} mismatch: expected {:?}, actual {:?}",
                case.case_id,
                case.expected,
                actual
            );
        }
    }

    #[test]
    fn candidate_input_order_does_not_change_evaluation() {
        let mut case = fixture().cases[0].clone();
        let before = evaluate_case(&case);
        case.targets.reverse();
        assert_eq!(before, evaluate_case(&case));
    }

    #[test]
    fn overlapping_intervals_do_not_create_robust_dominance() {
        let case = fixture()
            .cases
            .into_iter()
            .find(|case| case.case_id == "MVP-008-overlapping-intervals")
            .unwrap();
        let a = case
            .targets
            .iter()
            .find(|t| t.target_id == "interval-a")
            .unwrap();
        let b = case
            .targets
            .iter()
            .find(|t| t.target_id == "interval-b")
            .unwrap();
        assert!(!robustly_dominates(a, b));
        assert!(!robustly_dominates(b, a));
    }

    #[test]
    fn identical_intervals_are_not_strict_dominance() {
        let case = fixture().cases[0].clone();
        let mut a = case.targets[0].clone();
        let b = case.targets[1].clone();
        a.objectives = b.objectives.clone();
        assert!(!robustly_dominates(&a, &b));
    }

    #[test]
    fn food_safety_biology_and_process_failures_are_noncompensatory() {
        let case = fixture()
            .cases
            .iter()
            .find(|case| case.case_id == "MVP-016-hard-gate-fail-coverage")
            .unwrap();
        let actual = evaluate_case(case);
        assert_eq!(actual.status, CaseStatus::Valid);
        let blocked = actual
            .targets
            .iter()
            .find(|target| target.target_id == "multiple-hard-gate-failures")
            .unwrap();
        assert_eq!(blocked.disposition, TargetDisposition::Blocked);
        for reason in [
            "food_safety_failed",
            "biology_function_failed",
            "process_closure_failed",
        ] {
            assert!(
                blocked.reason_codes.contains(&reason.to_owned()),
                "{reason}"
            );
        }
        assert!(actual.pareto_frontier_ids.is_empty());
    }

    #[test]
    fn failed_authorization_cannot_be_compensated_by_metrics() {
        let case = fixture()
            .cases
            .into_iter()
            .find(|case| case.case_id == "MVP-002-hard-gate-over-market")
            .unwrap();
        let target = case
            .targets
            .iter()
            .find(|t| t.target_id == "market-winner")
            .unwrap();
        let result = target_disposition(target);
        assert_eq!(result.disposition, TargetDisposition::Blocked);
        assert!(
            result
                .reason_codes
                .contains(&"authorization_failed".to_owned())
        );
    }

    #[test]
    fn unknown_scale_transfer_is_indeterminate_and_not_frontier_eligible() {
        let case = fixture()
            .cases
            .into_iter()
            .find(|case| case.case_id == "MVP-006-scale-transfer-unknown")
            .unwrap();
        let actual = evaluate_case(&case);
        let uncertain = actual
            .targets
            .iter()
            .find(|t| t.target_id == "scale-unknown")
            .unwrap();
        assert_eq!(uncertain.disposition, TargetDisposition::Indeterminate);
        assert!(
            !actual
                .pareto_frontier_ids
                .contains(&"scale-unknown".to_owned())
        );
    }

    #[test]
    fn scenario_and_scope_mismatch_prevent_any_pareto_claim() {
        let corpus = fixture();
        for id in [
            "MVP-004-functional-unit-mismatch",
            "MVP-007-region-feedstock-scenario-shift",
            "MVP-011-objective-unit-mismatch",
            "MVP-012-demand-volume-band-mismatch",
        ] {
            let case = corpus.cases.iter().find(|case| case.case_id == id).unwrap();
            let actual = evaluate_case(case);
            assert_eq!(actual.status, CaseStatus::Incomparable, "{id}");
            assert!(actual.pareto_frontier_ids.is_empty(), "{id}");
            assert!(
                actual
                    .targets
                    .iter()
                    .all(|target| target.dominates.is_empty()),
                "{id}"
            );
        }
    }

    #[test]
    fn core_objectives_cannot_be_omitted_or_reversed() {
        let corpus = fixture();
        for (case_id, reason) in [
            (
                "MVP-013-missing-core-objectives-negative-control",
                "required_objective_missing:compliant_capacity",
            ),
            (
                "MVP-014-reversed-objective-direction-negative-control",
                "required_objective_direction_mismatch:functional_fit",
            ),
            (
                "MVP-015-compliant-capacity-unit-mismatch-negative-control",
                "compliant_capacity_unit_mismatch",
            ),
        ] {
            let case = corpus
                .cases
                .iter()
                .find(|case| case.case_id == case_id)
                .unwrap();
            let actual = evaluate_case(case);
            assert_eq!(actual.status, CaseStatus::InvalidInput, "{case_id}");
            assert!(
                actual.reason_codes.contains(&reason.to_owned()),
                "{case_id}"
            );
            assert!(actual.targets.is_empty(), "{case_id}");
        }
    }

    #[test]
    fn synthetic_data_cannot_be_promoted_to_measured_claims() {
        let case = fixture()
            .cases
            .into_iter()
            .find(|case| case.case_id == "MVP-009-synthetic-as-measured-negative-control")
            .unwrap();
        let actual = evaluate_case(&case);
        assert_eq!(actual.status, CaseStatus::InvalidInput);
        assert!(
            actual
                .reason_codes
                .contains(&"synthetic_as_measured_claim".to_owned())
        );
        assert!(actual.targets.is_empty());
    }

    #[test]
    fn downstream_usable_capacity_not_nominal_titer_drives_frontier() {
        let case = fixture()
            .cases
            .into_iter()
            .find(|case| case.case_id == "MVP-003-usable-output-over-titer")
            .unwrap();
        let actual = evaluate_case(&case);
        assert_eq!(
            actual.pareto_frontier_ids,
            vec!["lower-titer-high-recovery".to_owned()]
        );
        let recovered = actual
            .targets
            .iter()
            .find(|t| t.target_id == "lower-titer-high-recovery")
            .unwrap();
        assert!(
            recovered
                .dominates
                .contains(&"high-titer-low-recovery".to_owned())
        );
    }
}
