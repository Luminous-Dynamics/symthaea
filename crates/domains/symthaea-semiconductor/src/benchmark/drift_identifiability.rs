// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Small-instance source-identifiability and discriminating-measurement planner.
//!
//! This benchmark primitive distinguishes structural identifiability from
//! model fit. It does not ingest equipment data, authenticate evidence, execute
//! measurements, estimate a covariance model, or authorize process actions.
//! Known dependence must be represented as explicit nuisance-factor columns;
//! stochastic likelihood/covariance qualification remains with the metrology
//! likelihood owners.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const DRIFT_IDENTIFIABILITY_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.drift-identifiability-input.v1";
pub const DRIFT_IDENTIFIABILITY_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.drift-identifiability-report.v1";
pub const DRIFT_IDENTIFIABILITY_DIMENSIONS_V1: usize = 4;
const RANK_TOLERANCE_V1: f64 = 1.0e-10;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftAttributionTargetV1 {
    ProcessShift,
    InstrumentABias,
    InstrumentBBias,
    SharedCommonMode,
}

impl DriftAttributionTargetV1 {
    fn contrast(self) -> [f64; DRIFT_IDENTIFIABILITY_DIMENSIONS_V1] {
        match self {
            Self::ProcessShift => [1.0, 0.0, 0.0, 0.0],
            Self::InstrumentABias => [0.0, 1.0, 0.0, 0.0],
            Self::InstrumentBBias => [0.0, 0.0, 1.0, 0.0],
            Self::SharedCommonMode => [0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftActionEvidenceStateV1 {
    /// The action has already produced an owner-qualified observation.
    ObservedQualified,
    /// The action is an admissible candidate under the declared evidence contract.
    AvailableQualified,
    /// Evidence is missing or unresolved; the action cannot be counted or planned.
    Unknown,
    /// The action is explicitly ineligible.
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftObservationActionV1 {
    pub action_id: String,
    pub evidence_origin_id: String,
    /// Sensitivity of this observation to [process, gauge A, gauge B, shared cause].
    /// Shared causes must appear as explicit nuisance columns, not as assumed independence.
    pub sensitivity: [f64; DRIFT_IDENTIFIABILITY_DIMENSIONS_V1],
    pub evidence_state: DriftActionEvidenceStateV1,
    /// Positive integer cost units for one execution of this action.
    pub cost_units: u64,
    /// Conservative sequential delay units; this planner does not assume parallel execution.
    pub delay_units: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftIdentifiabilityInputV1 {
    pub schema: String,
    pub input_id: String,
    pub actions: Vec<DriftObservationActionV1>,
    pub required_targets: Vec<DriftAttributionTargetV1>,
    pub maximum_incremental_cost_units: u64,
    /// Exhaustive subset planning is deliberately bounded.
    pub maximum_candidate_actions: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftIdentifiabilityPlanStatusV1 {
    AlreadyIdentifiable,
    DiscriminatingPlanFound,
    NotIdentifiableWithinBudget,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftTargetIdentifiabilityV1 {
    pub target: DriftAttributionTargetV1,
    pub identifiable_from_observed: bool,
    pub identifiable_after_plan: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftIdentifiabilityReportV1 {
    pub schema: String,
    pub input_id: String,
    pub input_sha256: String,
    pub report_sha256: String,
    pub structural_parameter_dimension: usize,
    pub observed_design_rank: usize,
    pub final_design_rank: usize,
    pub target_status: Vec<DriftTargetIdentifiabilityV1>,
    pub plan_status: DriftIdentifiabilityPlanStatusV1,
    pub selected_action_ids: Vec<String>,
    pub incremental_cost_units: u64,
    pub conservative_sequential_delay_units: u64,
    /// This is a planning result, not permission to execute an action.
    pub measurement_execution_authority: bool,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct DriftIdentifiabilityErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), DriftIdentifiabilityErrorV1> {
    if value.trim().is_empty() {
        return Err(DriftIdentifiabilityErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, DriftIdentifiabilityErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        DriftIdentifiabilityErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn validate_action(action: &DriftObservationActionV1) -> Result<(), DriftIdentifiabilityErrorV1> {
    require_text(&action.action_id, "action.action_id")?;
    require_text(&action.evidence_origin_id, "action.evidence_origin_id")?;
    if action.cost_units == 0 {
        return Err(DriftIdentifiabilityErrorV1(format!(
            "action {} cost_units must be positive",
            action.action_id
        )));
    }
    if action.sensitivity.iter().any(|value| !value.is_finite()) {
        return Err(DriftIdentifiabilityErrorV1(format!(
            "action {} contains a non-finite sensitivity",
            action.action_id
        )));
    }
    if action
        .sensitivity
        .iter()
        .all(|value| value.abs() <= RANK_TOLERANCE_V1)
    {
        return Err(DriftIdentifiabilityErrorV1(format!(
            "action {} has a zero sensitivity row",
            action.action_id
        )));
    }
    Ok(())
}

fn validate_input(input: &DriftIdentifiabilityInputV1) -> Result<(), DriftIdentifiabilityErrorV1> {
    if input.schema != DRIFT_IDENTIFIABILITY_INPUT_SCHEMA_V1 {
        return Err(DriftIdentifiabilityErrorV1(
            "input schema is unsupported".into(),
        ));
    }
    require_text(&input.input_id, "input.input_id")?;
    if input.required_targets.is_empty() {
        return Err(DriftIdentifiabilityErrorV1(
            "at least one attribution target is required".into(),
        ));
    }
    if input.maximum_candidate_actions > 20 {
        return Err(DriftIdentifiabilityErrorV1(
            "exact planner supports at most 20 candidate actions".into(),
        ));
    }

    let mut action_ids = BTreeSet::new();
    for action in &input.actions {
        validate_action(action)?;
        if !action_ids.insert(action.action_id.as_str()) {
            return Err(DriftIdentifiabilityErrorV1("duplicate action ID".into()));
        }
    }

    let mut targets = BTreeSet::new();
    for target in &input.required_targets {
        if !targets.insert(*target) {
            return Err(DriftIdentifiabilityErrorV1(
                "duplicate attribution target".into(),
            ));
        }
    }
    Ok(())
}

/// Computes the row rank of a small sensitivity/design matrix with partial pivoting.
/// The matrix is a structural identifiability object, not a stochastic covariance model.
fn matrix_rank(rows: &[[f64; DRIFT_IDENTIFIABILITY_DIMENSIONS_V1]]) -> usize {
    if rows.is_empty() {
        return 0;
    }

    let mut matrix = rows.to_vec();
    let row_count = matrix.len();
    let mut pivot_row = 0;

    for column in 0..DRIFT_IDENTIFIABILITY_DIMENSIONS_V1 {
        if pivot_row >= row_count {
            break;
        }

        let pivot = (pivot_row..row_count)
            .max_by(|left, right| {
                matrix[*left][column]
                    .abs()
                    .total_cmp(&matrix[*right][column].abs())
            })
            .expect("non-empty pivot range");

        if matrix[pivot][column].abs() <= RANK_TOLERANCE_V1 {
            continue;
        }

        matrix.swap(pivot_row, pivot);
        let pivot_value = matrix[pivot_row][column];
        for value in &mut matrix[pivot_row][column..] {
            *value /= pivot_value;
        }
        let normalized_pivot_tail = matrix[pivot_row][column..].to_vec();

        for row in 0..row_count {
            if row == pivot_row {
                continue;
            }
            let factor = matrix[row][column];
            if factor.abs() <= RANK_TOLERANCE_V1 {
                continue;
            }
            for (value, pivot_value) in matrix[row][column..].iter_mut().zip(&normalized_pivot_tail)
            {
                *value -= factor * pivot_value;
            }
        }
        pivot_row += 1;
    }

    pivot_row
}

/// A linear contrast is estimable exactly when appending its row does not increase rank.
fn contrast_is_estimable(
    rows: &[[f64; DRIFT_IDENTIFIABILITY_DIMENSIONS_V1]],
    target: DriftAttributionTargetV1,
) -> bool {
    let rank = matrix_rank(rows);
    let mut augmented = rows.to_vec();
    augmented.push(target.contrast());
    matrix_rank(&augmented) == rank
}

fn all_targets_estimable(
    rows: &[[f64; DRIFT_IDENTIFIABILITY_DIMENSIONS_V1]],
    targets: &[DriftAttributionTargetV1],
) -> bool {
    targets
        .iter()
        .all(|target| contrast_is_estimable(rows, *target))
}

fn canonical_input_sha256(
    input: &DriftIdentifiabilityInputV1,
) -> Result<String, DriftIdentifiabilityErrorV1> {
    let mut canonical = input.clone();
    canonical
        .actions
        .sort_by(|left, right| left.action_id.cmp(&right.action_id));
    canonical.required_targets.sort();
    json_sha256(&canonical)
}

fn report_digest(
    report: &DriftIdentifiabilityReportV1,
) -> Result<String, DriftIdentifiabilityErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

#[derive(Clone, Debug)]
struct CandidatePlanV1 {
    cost: u64,
    delay: u64,
    action_ids: Vec<String>,
    rows: Vec<[f64; DRIFT_IDENTIFIABILITY_DIMENSIONS_V1]>,
}

/// Plans the least-cost qualified set of discriminating measurements for the
/// requested source contrasts. It never executes measurements or infers causes
/// from observation values. Likelihood and dependence qualification live upstream.
pub fn plan_drift_identifiability_v1(
    input: &DriftIdentifiabilityInputV1,
) -> Result<DriftIdentifiabilityReportV1, DriftIdentifiabilityErrorV1> {
    validate_input(input)?;

    let mut canonical_actions = input.actions.clone();
    canonical_actions.sort_by(|left, right| left.action_id.cmp(&right.action_id));

    let observed: Vec<&DriftObservationActionV1> = canonical_actions
        .iter()
        .filter(|action| action.evidence_state == DriftActionEvidenceStateV1::ObservedQualified)
        .collect();
    let candidates: Vec<&DriftObservationActionV1> = canonical_actions
        .iter()
        .filter(|action| action.evidence_state == DriftActionEvidenceStateV1::AvailableQualified)
        .collect();

    if candidates.len() > input.maximum_candidate_actions {
        return Err(DriftIdentifiabilityErrorV1(format!(
            "candidate action count {} exceeds declared exact limit {}",
            candidates.len(),
            input.maximum_candidate_actions
        )));
    }

    let observed_rows: Vec<[f64; DRIFT_IDENTIFIABILITY_DIMENSIONS_V1]> =
        observed.iter().map(|action| action.sensitivity).collect();
    let observed_rank = matrix_rank(&observed_rows);
    let observed_target_status: Vec<DriftTargetIdentifiabilityV1> = input
        .required_targets
        .iter()
        .map(|target| DriftTargetIdentifiabilityV1 {
            target: *target,
            identifiable_from_observed: contrast_is_estimable(&observed_rows, *target),
            identifiable_after_plan: false,
        })
        .collect();

    let (plan_status, selected_actions, final_rows, plan_cost, plan_delay) =
        if all_targets_estimable(&observed_rows, &input.required_targets) {
            (
                DriftIdentifiabilityPlanStatusV1::AlreadyIdentifiable,
                Vec::new(),
                observed_rows.clone(),
                0,
                0,
            )
        } else {
            let mut best: Option<CandidatePlanV1> = None;
            let subset_count = 1_u64 << candidates.len();

            for mask in 0..subset_count {
                let mut rows = observed_rows.clone();
                let mut ids = Vec::new();
                let mut cost = 0_u64;
                let mut delay = 0_u64;
                let mut overflowed = false;

                for (index, action) in candidates.iter().enumerate() {
                    if (mask & (1_u64 << index)) == 0 {
                        continue;
                    }
                    let Some(next_cost) = cost.checked_add(action.cost_units) else {
                        overflowed = true;
                        break;
                    };
                    let Some(next_delay) = delay.checked_add(action.delay_units) else {
                        overflowed = true;
                        break;
                    };
                    cost = next_cost;
                    delay = next_delay;
                    ids.push(action.action_id.clone());
                    rows.push(action.sensitivity);
                }

                if overflowed || cost > input.maximum_incremental_cost_units {
                    continue;
                }
                if !all_targets_estimable(&rows, &input.required_targets) {
                    continue;
                }

                ids.sort();
                let candidate = CandidatePlanV1 {
                    cost,
                    delay,
                    action_ids: ids,
                    rows,
                };

                let replace = best.as_ref().is_none_or(|current| {
                    (candidate.cost, candidate.delay, &candidate.action_ids)
                        < (current.cost, current.delay, &current.action_ids)
                });
                if replace {
                    best = Some(candidate);
                }
            }

            match best {
                Some(plan) => (
                    DriftIdentifiabilityPlanStatusV1::DiscriminatingPlanFound,
                    plan.action_ids,
                    plan.rows,
                    plan.cost,
                    plan.delay,
                ),
                None => (
                    DriftIdentifiabilityPlanStatusV1::NotIdentifiableWithinBudget,
                    Vec::new(),
                    observed_rows.clone(),
                    0,
                    0,
                ),
            }
        };

    let final_rank = matrix_rank(&final_rows);
    let target_status = observed_target_status
        .into_iter()
        .map(|mut target| {
            target.identifiable_after_plan = contrast_is_estimable(&final_rows, target.target);
            target
        })
        .collect();

    let mut report = DriftIdentifiabilityReportV1 {
        schema: DRIFT_IDENTIFIABILITY_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        input_sha256: canonical_input_sha256(input)?,
        report_sha256: String::new(),
        structural_parameter_dimension: DRIFT_IDENTIFIABILITY_DIMENSIONS_V1,
        observed_design_rank: observed_rank,
        final_design_rank: final_rank,
        target_status,
        plan_status,
        selected_action_ids: selected_actions,
        incremental_cost_units: plan_cost,
        conservative_sequential_delay_units: plan_delay,
        measurement_execution_authority: false,
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(
        action_id: &str,
        sensitivity: [f64; DRIFT_IDENTIFIABILITY_DIMENSIONS_V1],
        evidence_state: DriftActionEvidenceStateV1,
        cost_units: u64,
        delay_units: u64,
    ) -> DriftObservationActionV1 {
        DriftObservationActionV1 {
            action_id: action_id.into(),
            evidence_origin_id: format!("origin-{action_id}"),
            sensitivity,
            evidence_state,
            cost_units,
            delay_units,
        }
    }

    fn input(
        actions: Vec<DriftObservationActionV1>,
        targets: Vec<DriftAttributionTargetV1>,
        budget: u64,
    ) -> DriftIdentifiabilityInputV1 {
        DriftIdentifiabilityInputV1 {
            schema: DRIFT_IDENTIFIABILITY_INPUT_SCHEMA_V1.into(),
            input_id: "semi-cause-small-known-answer-v1".into(),
            actions,
            required_targets: targets,
            maximum_incremental_cost_units: budget,
            maximum_candidate_actions: 20,
        }
    }

    fn wafer_a(state: DriftActionEvidenceStateV1, cost: u64) -> DriftObservationActionV1 {
        action("wafer-a", [1.0, 1.0, 0.0, 1.0], state, cost, 2)
    }

    fn witness_a(state: DriftActionEvidenceStateV1, cost: u64) -> DriftObservationActionV1 {
        action("witness-a", [0.0, 1.0, 0.0, 1.0], state, cost, 1)
    }

    fn wafer_b(state: DriftActionEvidenceStateV1, cost: u64) -> DriftObservationActionV1 {
        action("wafer-b", [1.0, 0.0, 1.0, 1.0], state, cost, 2)
    }

    fn witness_b(state: DriftActionEvidenceStateV1, cost: u64) -> DriftObservationActionV1 {
        action("witness-b", [0.0, 0.0, 1.0, 1.0], state, cost, 1)
    }

    fn common_monitor(state: DriftActionEvidenceStateV1, cost: u64) -> DriftObservationActionV1 {
        action("common-monitor", [0.0, 0.0, 0.0, 1.0], state, cost, 1)
    }

    #[test]
    fn process_and_gauge_are_not_separable_from_one_product_measurement() {
        let value = input(
            vec![wafer_a(DriftActionEvidenceStateV1::ObservedQualified, 1)],
            vec![
                DriftAttributionTargetV1::ProcessShift,
                DriftAttributionTargetV1::InstrumentABias,
            ],
            10,
        );

        let report = plan_drift_identifiability_v1(&value).unwrap();
        assert_eq!(
            report.plan_status,
            DriftIdentifiabilityPlanStatusV1::NotIdentifiableWithinBudget
        );
        assert_eq!(report.observed_design_rank, 1);
        assert!(
            !report.target_status[0].identifiable_from_observed
                || !report.target_status[1].identifiable_from_observed
        );
        assert!(!report.measurement_execution_authority);
    }

    #[test]
    fn reference_crossover_identifies_process_but_not_gauge_common_mode_split() {
        let value = input(
            vec![
                wafer_a(DriftActionEvidenceStateV1::ObservedQualified, 1),
                witness_a(DriftActionEvidenceStateV1::ObservedQualified, 1),
            ],
            vec![
                DriftAttributionTargetV1::ProcessShift,
                DriftAttributionTargetV1::InstrumentABias,
                DriftAttributionTargetV1::SharedCommonMode,
            ],
            10,
        );

        let report = plan_drift_identifiability_v1(&value).unwrap();
        assert_eq!(
            report.plan_status,
            DriftIdentifiabilityPlanStatusV1::NotIdentifiableWithinBudget
        );
        assert!(report.target_status[0].identifiable_from_observed);
        assert!(!report.target_status[1].identifiable_from_observed);
        assert!(!report.target_status[2].identifiable_from_observed);
    }

    #[test]
    fn independent_common_mode_monitor_separates_gauge_from_shared_cause() {
        let value = input(
            vec![
                wafer_a(DriftActionEvidenceStateV1::ObservedQualified, 1),
                witness_a(DriftActionEvidenceStateV1::ObservedQualified, 1),
                common_monitor(DriftActionEvidenceStateV1::ObservedQualified, 1),
            ],
            vec![
                DriftAttributionTargetV1::ProcessShift,
                DriftAttributionTargetV1::InstrumentABias,
                DriftAttributionTargetV1::SharedCommonMode,
            ],
            10,
        );

        let report = plan_drift_identifiability_v1(&value).unwrap();
        assert_eq!(
            report.plan_status,
            DriftIdentifiabilityPlanStatusV1::AlreadyIdentifiable
        );
        assert!(
            report
                .target_status
                .iter()
                .all(|target| target.identifiable_from_observed)
        );
    }

    #[test]
    fn planner_selects_minimum_cost_discriminating_action() {
        let value = input(
            vec![
                wafer_a(DriftActionEvidenceStateV1::ObservedQualified, 1),
                action(
                    "cheap-witness",
                    [0.0, 1.0, 0.0, 1.0],
                    DriftActionEvidenceStateV1::AvailableQualified,
                    2,
                    1,
                ),
                action(
                    "expensive-witness",
                    [0.0, 1.0, 0.0, 1.0],
                    DriftActionEvidenceStateV1::AvailableQualified,
                    5,
                    1,
                ),
            ],
            vec![DriftAttributionTargetV1::ProcessShift],
            10,
        );

        let report = plan_drift_identifiability_v1(&value).unwrap();
        assert_eq!(
            report.plan_status,
            DriftIdentifiabilityPlanStatusV1::DiscriminatingPlanFound
        );
        assert_eq!(report.selected_action_ids, vec!["cheap-witness"]);
        assert_eq!(report.incremental_cost_units, 2);
        assert!(report.target_status[0].identifiable_after_plan);
        assert!(!report.measurement_execution_authority);
    }

    #[test]
    fn ineligible_actions_never_enter_the_discriminating_plan() {
        let value = input(
            vec![
                wafer_a(DriftActionEvidenceStateV1::ObservedQualified, 1),
                action(
                    "unqualified-witness",
                    [0.0, 1.0, 0.0, 1.0],
                    DriftActionEvidenceStateV1::Unknown,
                    1,
                    1,
                ),
            ],
            vec![DriftAttributionTargetV1::ProcessShift],
            10,
        );

        let report = plan_drift_identifiability_v1(&value).unwrap();
        assert_eq!(
            report.plan_status,
            DriftIdentifiabilityPlanStatusV1::NotIdentifiableWithinBudget
        );
        assert!(report.selected_action_ids.is_empty());
    }

    #[test]
    fn budget_prevents_an_otherwise_identifying_measurement_plan() {
        let value = input(
            vec![
                wafer_a(DriftActionEvidenceStateV1::ObservedQualified, 1),
                witness_a(DriftActionEvidenceStateV1::AvailableQualified, 4),
            ],
            vec![DriftAttributionTargetV1::ProcessShift],
            3,
        );

        let report = plan_drift_identifiability_v1(&value).unwrap();
        assert_eq!(
            report.plan_status,
            DriftIdentifiabilityPlanStatusV1::NotIdentifiableWithinBudget
        );
    }

    #[test]
    fn duplicate_action_ids_and_non_finite_sensitivities_reject() {
        let duplicate = input(
            vec![
                wafer_a(DriftActionEvidenceStateV1::ObservedQualified, 1),
                wafer_a(DriftActionEvidenceStateV1::AvailableQualified, 1),
            ],
            vec![DriftAttributionTargetV1::ProcessShift],
            10,
        );
        assert!(plan_drift_identifiability_v1(&duplicate).is_err());

        let mut non_finite = input(
            vec![wafer_a(DriftActionEvidenceStateV1::ObservedQualified, 1)],
            vec![DriftAttributionTargetV1::ProcessShift],
            10,
        );
        non_finite.actions.push(action(
            "bad",
            [f64::NAN, 0.0, 0.0, 0.0],
            DriftActionEvidenceStateV1::AvailableQualified,
            1,
            1,
        ));
        assert!(plan_drift_identifiability_v1(&non_finite).is_err());
    }

    #[test]
    fn event_ordering_and_candidate_input_order_do_not_change_receipt() {
        let actions_a = vec![
            wafer_a(DriftActionEvidenceStateV1::ObservedQualified, 1),
            witness_a(DriftActionEvidenceStateV1::AvailableQualified, 3),
            action(
                "cheaper-witness",
                [0.0, 1.0, 0.0, 1.0],
                DriftActionEvidenceStateV1::AvailableQualified,
                2,
                1,
            ),
        ];
        let mut actions_b = actions_a.clone();
        actions_b.reverse();

        let target = vec![DriftAttributionTargetV1::ProcessShift];
        let report_a =
            plan_drift_identifiability_v1(&input(actions_a, target.clone(), 10)).unwrap();
        let report_b = plan_drift_identifiability_v1(&input(actions_b, target, 10)).unwrap();
        assert_eq!(report_a.input_sha256, report_b.input_sha256);
        assert_eq!(report_a.report_sha256, report_b.report_sha256);
    }

    #[test]
    fn gauge_b_crossover_requires_independent_common_mode_monitor() {
        let value = input(
            vec![
                wafer_b(DriftActionEvidenceStateV1::ObservedQualified, 1),
                witness_b(DriftActionEvidenceStateV1::ObservedQualified, 1),
                common_monitor(DriftActionEvidenceStateV1::AvailableQualified, 2),
            ],
            vec![
                DriftAttributionTargetV1::ProcessShift,
                DriftAttributionTargetV1::InstrumentBBias,
                DriftAttributionTargetV1::SharedCommonMode,
            ],
            10,
        );

        let report = plan_drift_identifiability_v1(&value).unwrap();
        assert_eq!(
            report.plan_status,
            DriftIdentifiabilityPlanStatusV1::DiscriminatingPlanFound
        );
        assert_eq!(report.selected_action_ids, vec!["common-monitor"]);
        assert!(
            report
                .target_status
                .iter()
                .all(|target| target.identifiable_after_plan)
        );
    }

    #[test]
    fn multiple_targets_require_all_declared_contrasts_to_be_estimable() {
        let value = input(
            vec![
                wafer_a(DriftActionEvidenceStateV1::ObservedQualified, 1),
                witness_a(DriftActionEvidenceStateV1::ObservedQualified, 1),
                common_monitor(DriftActionEvidenceStateV1::AvailableQualified, 2),
            ],
            vec![
                DriftAttributionTargetV1::ProcessShift,
                DriftAttributionTargetV1::InstrumentABias,
                DriftAttributionTargetV1::SharedCommonMode,
            ],
            10,
        );

        let report = plan_drift_identifiability_v1(&value).unwrap();
        assert_eq!(
            report.plan_status,
            DriftIdentifiabilityPlanStatusV1::DiscriminatingPlanFound
        );
        assert_eq!(report.selected_action_ids, vec!["common-monitor"]);
        assert!(
            report
                .target_status
                .iter()
                .all(|target| target.identifiable_after_plan)
        );
    }
}
