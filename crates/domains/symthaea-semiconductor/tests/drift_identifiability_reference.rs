#![forbid(unsafe_code)]

use serde::Deserialize;
use symthaea_semiconductor::benchmark::{
    DRIFT_IDENTIFIABILITY_INPUT_SCHEMA_V1, DriftActionEvidenceStateV1, DriftAttributionTargetV1,
    DriftIdentifiabilityInputV1, DriftIdentifiabilityPlanStatusV1, DriftIdentifiabilityReportV1,
    DriftObservationActionV1, plan_drift_identifiability_v1,
};

const CORPUS: &str = include_str!(
    "../../../../benchmarks/semi-bench-001/reference/fixtures/drift-identifiability-known-answer-v1.json"
);
const DIMENSIONS: usize = 4;
const ORACLE_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Deserialize)]
struct CorpusV1 {
    schema_version: String,
    status: String,
    claim_ceiling: String,
    dimension_order: Vec<String>,
    cases: Vec<FixtureCaseV1>,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureCaseV1 {
    id: String,
    actions: Vec<DriftObservationActionV1>,
    required_targets: Vec<DriftAttributionTargetV1>,
    maximum_incremental_cost_units: u64,
    maximum_candidate_actions: usize,
    expected: FixtureExpectedV1,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureExpectedV1 {
    observed_design_rank: usize,
    plan_status: DriftIdentifiabilityPlanStatusV1,
    selected_action_ids: Vec<String>,
    incremental_cost_units: u64,
    conservative_sequential_delay_units: u64,
    target_status: Vec<FixtureTargetStatusV1>,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureTargetStatusV1 {
    target: DriftAttributionTargetV1,
    identifiable_from_observed: bool,
    identifiable_after_plan: bool,
}

#[derive(Debug)]
struct OraclePlanV1 {
    status: DriftIdentifiabilityPlanStatusV1,
    selected_action_ids: Vec<String>,
    cost: u64,
    delay: u64,
    final_rows: Vec<[f64; DIMENSIONS]>,
}

#[derive(Debug)]
struct OracleCandidatePlanV1 {
    cost: u64,
    delay: u64,
    action_ids: Vec<String>,
    rows: Vec<[f64; DIMENSIONS]>,
}

fn dot(left: &[f64; DIMENSIONS], right: &[f64; DIMENSIONS]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn target_contrast(target: DriftAttributionTargetV1) -> [f64; DIMENSIONS] {
    match target {
        DriftAttributionTargetV1::ProcessShift => [1.0, 0.0, 0.0, 0.0],
        DriftAttributionTargetV1::InstrumentABias => [0.0, 1.0, 0.0, 0.0],
        DriftAttributionTargetV1::InstrumentBBias => [0.0, 0.0, 1.0, 0.0],
        DriftAttributionTargetV1::SharedCommonMode => [0.0, 0.0, 0.0, 1.0],
    }
}

/// Independent modified Gram-Schmidt oracle. This intentionally does not call
/// the production Gaussian-elimination rank implementation.
fn orthonormal_row_basis(rows: &[[f64; DIMENSIONS]]) -> Vec<[f64; DIMENSIONS]> {
    let mut basis: Vec<[f64; DIMENSIONS]> = Vec::new();

    for row in rows {
        let mut residual = *row;
        for basis_row in &basis {
            let projection = dot(&residual, basis_row);
            for (value, basis_value) in residual.iter_mut().zip(basis_row) {
                *value -= projection * basis_value;
            }
        }

        let norm = dot(&residual, &residual).sqrt();
        if norm > ORACLE_TOLERANCE {
            for value in &mut residual {
                *value /= norm;
            }
            basis.push(residual);
        }
    }

    basis
}

fn oracle_estimable(rows: &[[f64; DIMENSIONS]], target: DriftAttributionTargetV1) -> bool {
    let basis = orthonormal_row_basis(rows);
    let mut residual = target_contrast(target);
    for basis_row in &basis {
        let projection = dot(&residual, basis_row);
        for (value, basis_value) in residual.iter_mut().zip(basis_row) {
            *value -= projection * basis_value;
        }
    }
    dot(&residual, &residual).sqrt() <= ORACLE_TOLERANCE
}

fn oracle_all_targets_estimable(
    rows: &[[f64; DIMENSIONS]],
    targets: &[DriftAttributionTargetV1],
) -> bool {
    targets.iter().all(|target| oracle_estimable(rows, *target))
}

fn oracle_plan(case: &FixtureCaseV1) -> OraclePlanV1 {
    let mut observed_actions: Vec<&DriftObservationActionV1> = case
        .actions
        .iter()
        .filter(|action| action.evidence_state == DriftActionEvidenceStateV1::ObservedQualified)
        .collect();
    observed_actions.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    let observed_rows: Vec<[f64; DIMENSIONS]> = observed_actions
        .iter()
        .map(|action| action.sensitivity)
        .collect();

    if oracle_all_targets_estimable(&observed_rows, &case.required_targets) {
        return OraclePlanV1 {
            status: DriftIdentifiabilityPlanStatusV1::AlreadyIdentifiable,
            selected_action_ids: Vec::new(),
            cost: 0,
            delay: 0,
            final_rows: observed_rows,
        };
    }

    let mut candidates: Vec<&DriftObservationActionV1> = case
        .actions
        .iter()
        .filter(|action| action.evidence_state == DriftActionEvidenceStateV1::AvailableQualified)
        .collect();
    candidates.sort_by(|left, right| left.action_id.cmp(&right.action_id));

    let mut best: Option<OracleCandidatePlanV1> = None;
    for mask in 0_u64..(1_u64 << candidates.len()) {
        let mut rows = observed_rows.clone();
        let mut action_ids = Vec::new();
        let mut cost = 0_u64;
        let mut delay = 0_u64;

        for (index, action) in candidates.iter().enumerate() {
            if (mask & (1_u64 << index)) == 0 {
                continue;
            }
            cost = cost
                .checked_add(action.cost_units)
                .expect("fixture cost does not overflow");
            delay = delay
                .checked_add(action.delay_units)
                .expect("fixture delay does not overflow");
            action_ids.push(action.action_id.clone());
            rows.push(action.sensitivity);
        }

        if cost > case.maximum_incremental_cost_units
            || !oracle_all_targets_estimable(&rows, &case.required_targets)
        {
            continue;
        }

        let candidate = OracleCandidatePlanV1 {
            cost,
            delay,
            action_ids,
            rows,
        };
        let should_replace = best.as_ref().is_none_or(|current| {
            (candidate.cost, candidate.delay, &candidate.action_ids)
                < (current.cost, current.delay, &current.action_ids)
        });
        if should_replace {
            best = Some(candidate);
        }
    }

    match best {
        Some(plan) => OraclePlanV1 {
            status: DriftIdentifiabilityPlanStatusV1::DiscriminatingPlanFound,
            selected_action_ids: plan.action_ids,
            cost: plan.cost,
            delay: plan.delay,
            final_rows: plan.rows,
        },
        None => OraclePlanV1 {
            status: DriftIdentifiabilityPlanStatusV1::NotIdentifiableWithinBudget,
            selected_action_ids: Vec::new(),
            cost: 0,
            delay: 0,
            final_rows: observed_rows,
        },
    }
}

fn production_input(case: &FixtureCaseV1) -> DriftIdentifiabilityInputV1 {
    DriftIdentifiabilityInputV1 {
        schema: DRIFT_IDENTIFIABILITY_INPUT_SCHEMA_V1.into(),
        input_id: case.id.clone(),
        actions: case.actions.clone(),
        required_targets: case.required_targets.clone(),
        maximum_incremental_cost_units: case.maximum_incremental_cost_units,
        maximum_candidate_actions: case.maximum_candidate_actions,
    }
}

fn assert_report_matches_independent_oracle(
    case: &FixtureCaseV1,
    report: &DriftIdentifiabilityReportV1,
) {
    let oracle = oracle_plan(case);
    let observed_rows: Vec<[f64; DIMENSIONS]> = case
        .actions
        .iter()
        .filter(|action| action.evidence_state == DriftActionEvidenceStateV1::ObservedQualified)
        .map(|action| action.sensitivity)
        .collect();

    assert_eq!(
        report.observed_design_rank,
        orthonormal_row_basis(&observed_rows).len(),
        "{}",
        case.id
    );
    assert_eq!(report.plan_status, oracle.status, "{}", case.id);
    assert_eq!(
        report.selected_action_ids, oracle.selected_action_ids,
        "{}",
        case.id
    );
    assert_eq!(report.incremental_cost_units, oracle.cost, "{}", case.id);
    assert_eq!(
        report.conservative_sequential_delay_units, oracle.delay,
        "{}",
        case.id
    );
    assert_eq!(
        report.final_design_rank,
        orthonormal_row_basis(&oracle.final_rows).len(),
        "{}",
        case.id
    );

    assert_eq!(
        report.target_status.len(),
        case.required_targets.len(),
        "{}",
        case.id
    );
    for (actual, expected) in report
        .target_status
        .iter()
        .zip(&case.expected.target_status)
    {
        assert_eq!(actual.target, expected.target, "{}", case.id);
        assert_eq!(
            actual.identifiable_from_observed,
            oracle_estimable(&observed_rows, actual.target),
            "{} observed target {:?}",
            case.id,
            actual.target
        );
        assert_eq!(
            actual.identifiable_after_plan,
            oracle_estimable(&oracle.final_rows, actual.target),
            "{} target after plan {:?}",
            case.id,
            actual.target
        );
        assert_eq!(
            actual.identifiable_from_observed, expected.identifiable_from_observed,
            "{} fixture observed target",
            case.id
        );
        assert_eq!(
            actual.identifiable_after_plan, expected.identifiable_after_plan,
            "{} fixture planned target",
            case.id
        );
    }

    assert_eq!(report.plan_status, case.expected.plan_status, "{}", case.id);
    assert_eq!(
        report.selected_action_ids, case.expected.selected_action_ids,
        "{}",
        case.id
    );
    assert_eq!(
        report.incremental_cost_units, case.expected.incremental_cost_units,
        "{}",
        case.id
    );
    assert_eq!(
        report.conservative_sequential_delay_units,
        case.expected.conservative_sequential_delay_units,
        "{}",
        case.id
    );
    assert_eq!(
        report.observed_design_rank, case.expected.observed_design_rank,
        "{}",
        case.id
    );
    assert!(!report.measurement_execution_authority);
}

#[test]
fn frozen_cases_match_known_answers_and_independent_rank_plan_oracle() {
    let corpus: CorpusV1 =
        serde_json::from_str(CORPUS).expect("valid frozen source-attribution corpus");
    assert_eq!(
        corpus.schema_version,
        "semi-cause-identifiability-known-answer-v0.1"
    );
    assert_eq!(
        corpus.status,
        "frozen_synthetic_structural_identifiability_cases"
    );
    assert!(
        corpus
            .claim_ceiling
            .contains("no physical fault attribution")
    );
    assert_eq!(
        corpus.dimension_order,
        vec![
            "process_shift",
            "instrument_a_bias",
            "instrument_b_bias",
            "shared_common_mode",
        ]
    );
    assert_eq!(corpus.cases.len(), 10);

    for case in &corpus.cases {
        let input = production_input(case);
        let report = plan_drift_identifiability_v1(&input)
            .unwrap_or_else(|error| panic!("{}: {error}", case.id));
        assert_report_matches_independent_oracle(case, &report);
    }
}

#[test]
fn reordering_actions_preserves_input_and_report_digests() {
    let corpus: CorpusV1 =
        serde_json::from_str(CORPUS).expect("valid frozen source-attribution corpus");
    for case in &corpus.cases {
        let original = plan_drift_identifiability_v1(&production_input(case)).unwrap();
        let mut permuted = case.clone();
        permuted.actions.reverse();
        let reordered = plan_drift_identifiability_v1(&production_input(&permuted)).unwrap();

        assert_eq!(original.input_sha256, reordered.input_sha256, "{}", case.id);
        assert_eq!(
            original.report_sha256, reordered.report_sha256,
            "{}",
            case.id
        );
    }
}
