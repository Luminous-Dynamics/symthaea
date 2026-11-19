//! Stage-aware validation for online plasma-closure history buffers.
//!
//! Physical stage time and computational availability order are distinct.
//! Explicit Runge-Kutta tableaux may evaluate stages in a non-monotone order
//! of abscissae (for example c = 0, 1, 1/2). A later-computed provisional
//! stage can therefore represent a later physical time than the current query.
//!
//! This validator preserves both facts:
//! - a slot must already have been computed before the query stage;
//! - a same-step slot whose physical abscissa is later than the query must be
//!   explicitly declared as an integrator dependency, unless PaperExact mode
//!   is reproducing the published buffer protocol unchanged.
//!
//! This crate validates buffer metadata and causal availability only. It does
//! not validate a PDE solver, numerical solution, plasma physics, labels,
//! source data, or model quality.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalAbscissa {
    pub numerator: u64,
    pub denominator: u64,
}

impl RationalAbscissa {
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };

    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    pub fn is_valid(self) -> bool {
        self.denominator > 0 && self.numerator <= self.denominator
    }

    fn compare(self, other: Self) -> std::cmp::Ordering {
        let left = u128::from(self.numerator) * u128::from(other.denominator);
        let right = u128::from(other.numerator) * u128::from(self.denominator);
        left.cmp(&right)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    AcceptedPhysicalState,
    ProvisionalStageState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferPolicy {
    /// Faithfully reproduce the paper's deployed buffer semantics.
    /// Earlier-computed provisional stages are allowed, including stages whose
    /// abscissa is later than the query. Such slots are surfaced separately.
    PaperExact,
    /// Only committed physical states may persist in the history buffer.
    AcceptedStateOnly,
    /// Provisional states require an exact tableau digest and an explicit
    /// same-step dependency declaration when their abscissa is later.
    StageExplicit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageStamp {
    pub sample_id: String,
    pub solver_run_id: String,
    pub trajectory_id: String,
    pub solver_scheme_id: String,
    pub tableau_sha256: Option<String>,
    pub step_index: u64,
    /// Order in which stages were actually computed within this step.
    /// Accepted physical states use 0; provisional stages use 1 or greater.
    pub evaluation_order: u32,
    pub stage_id: String,
    /// Rational Runge-Kutta abscissa c, not a wall-clock timestamp.
    pub abscissa: RationalAbscissa,
    pub source_state_id: String,
    pub kind: SlotKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryStamp {
    pub solver_run_id: String,
    pub trajectory_id: String,
    pub solver_scheme_id: String,
    pub tableau_sha256: Option<String>,
    pub step_index: u64,
    pub evaluation_order: u32,
    pub stage_id: String,
    pub abscissa: RationalAbscissa,
    /// Provisional stages from the same step that the declared tableau says
    /// are legal inputs to this query, even if their c value is later.
    pub allowed_prior_stage_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferContract {
    pub policy: BufferPolicy,
    pub max_age_steps: u64,
    pub minimum_history_slots: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferDisposition {
    Valid,
    Invalid,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferIssue {
    EmptyQueryRunId,
    EmptyQueryTrajectoryId,
    EmptyQuerySchemeId,
    EmptyQueryStageId,
    InvalidQueryAbscissa,
    QueryMustBeProvisionalStage,
    DuplicateAllowedDependency(String),
    EmptyAllowedDependency,
    EmptySampleId(usize),
    EmptyRunId(String),
    EmptyTrajectoryId(String),
    EmptySchemeId(String),
    EmptyStageId(String),
    EmptySourceStateId(String),
    InvalidSlotAbscissa(String),
    RunMismatch(String),
    TrajectoryMismatch(String),
    SchemeMismatch(String),
    TableauMissingForStageExplicit(String),
    TableauMismatch(String),
    DuplicateSampleId(String),
    DuplicateEvaluationPosition(u64, u32),
    UnsortedWindow(usize),
    FutureStep(String),
    StageNotYetComputed(String),
    StaleSlot(String),
    AcceptedSlotHasStageOrder(String),
    AcceptedSlotHasWrongStageId(String),
    AcceptedSlotHasWrongAbscissa(String),
    ProvisionalSlotHasNoStageOrder(String),
    ProvisionalSlotUsesAcceptedStageId(String),
    ProvisionalSlotDisallowed(String),
    FutureAbscissaWithoutDeclaredDependency(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferReport {
    pub disposition: BufferDisposition,
    pub accepted_slot_ids: Vec<String>,
    pub provisional_slot_ids: Vec<String>,
    /// Earlier-computed stages whose c value exceeds the query c. This is
    /// expected in some explicit RK tableaux and is never hidden in the report.
    pub non_monotone_stage_slot_ids: Vec<String>,
    pub issues: Vec<BufferIssue>,
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn push_unique(issues: &mut Vec<BufferIssue>, issue: BufferIssue) {
    if !issues.contains(&issue) {
        issues.push(issue);
    }
}

fn is_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn position_is_before(slot: &StageStamp, query: &QueryStamp) -> bool {
    slot.step_index < query.step_index
        || (slot.step_index == query.step_index
            && slot.evaluation_order < query.evaluation_order)
}

fn validate_slot_shape(slot: &StageStamp, index: usize, issues: &mut Vec<BufferIssue>) {
    if !non_empty(&slot.sample_id) {
        push_unique(issues, BufferIssue::EmptySampleId(index));
    }
    if !non_empty(&slot.solver_run_id) {
        push_unique(
            issues,
            BufferIssue::EmptyRunId(slot.sample_id.clone()),
        );
    }
    if !non_empty(&slot.trajectory_id) {
        push_unique(
            issues,
            BufferIssue::EmptyTrajectoryId(slot.sample_id.clone()),
        );
    }
    if !non_empty(&slot.solver_scheme_id) {
        push_unique(
            issues,
            BufferIssue::EmptySchemeId(slot.sample_id.clone()),
        );
    }
    if !non_empty(&slot.stage_id) {
        push_unique(
            issues,
            BufferIssue::EmptyStageId(slot.sample_id.clone()),
        );
    }
    if !non_empty(&slot.source_state_id) {
        push_unique(
            issues,
            BufferIssue::EmptySourceStateId(slot.sample_id.clone()),
        );
    }
    if !slot.abscissa.is_valid() {
        push_unique(
            issues,
            BufferIssue::InvalidSlotAbscissa(slot.sample_id.clone()),
        );
    }

    match slot.kind {
        SlotKind::AcceptedPhysicalState => {
            if slot.evaluation_order != 0 {
                push_unique(
                    issues,
                    BufferIssue::AcceptedSlotHasStageOrder(slot.sample_id.clone()),
                );
            }
            if slot.stage_id != "accepted" {
                push_unique(
                    issues,
                    BufferIssue::AcceptedSlotHasWrongStageId(slot.sample_id.clone()),
                );
            }
            if slot.abscissa != RationalAbscissa::ONE {
                push_unique(
                    issues,
                    BufferIssue::AcceptedSlotHasWrongAbscissa(slot.sample_id.clone()),
                );
            }
        }
        SlotKind::ProvisionalStageState => {
            if slot.evaluation_order == 0 {
                push_unique(
                    issues,
                    BufferIssue::ProvisionalSlotHasNoStageOrder(slot.sample_id.clone()),
                );
            }
            if slot.stage_id == "accepted" {
                push_unique(
                    issues,
                    BufferIssue::ProvisionalSlotUsesAcceptedStageId(slot.sample_id.clone()),
                );
            }
        }
    }
}

/// Validate a history window against the exact solver/query position.
///
/// The input order must already be chronological in *computation order*
/// (step index, then evaluation order). We do not sort or rewrite the source
/// window. Runge-Kutta abscissae are checked separately and may be non-monotone.
pub fn validate_buffer(
    query: &QueryStamp,
    slots: &[StageStamp],
    contract: BufferContract,
) -> BufferReport {
    let mut issues = Vec::new();
    let mut accepted_slot_ids = Vec::new();
    let mut provisional_slot_ids = Vec::new();
    let mut non_monotone_stage_slot_ids = Vec::new();

    if !non_empty(&query.solver_run_id) {
        push_unique(&mut issues, BufferIssue::EmptyQueryRunId);
    }
    if !non_empty(&query.trajectory_id) {
        push_unique(&mut issues, BufferIssue::EmptyQueryTrajectoryId);
    }
    if !non_empty(&query.solver_scheme_id) {
        push_unique(&mut issues, BufferIssue::EmptyQuerySchemeId);
    }
    if !non_empty(&query.stage_id) {
        push_unique(&mut issues, BufferIssue::EmptyQueryStageId);
    }
    if !query.abscissa.is_valid() {
        push_unique(&mut issues, BufferIssue::InvalidQueryAbscissa);
    }
    if query.evaluation_order == 0 {
        push_unique(&mut issues, BufferIssue::QueryMustBeProvisionalStage);
    }

    let mut allowed_dependencies = BTreeSet::new();
    for dependency in &query.allowed_prior_stage_ids {
        if !non_empty(dependency) {
            push_unique(&mut issues, BufferIssue::EmptyAllowedDependency);
        }
        if !allowed_dependencies.insert(dependency.as_str()) {
            push_unique(
                &mut issues,
                BufferIssue::DuplicateAllowedDependency(dependency.clone()),
            );
        }
    }

    if contract.policy == BufferPolicy::StageExplicit {
        match &query.tableau_sha256 {
            Some(digest) if is_hex_digest(digest) => {}
            _ => push_unique(
                &mut issues,
                BufferIssue::TableauMissingForStageExplicit("query".into()),
            ),
        }
    }

    let mut sample_ids = BTreeSet::new();
    let mut positions = BTreeSet::new();
    let mut previous_position: Option<(u64, u32)> = None;

    for (index, slot) in slots.iter().enumerate() {
        validate_slot_shape(slot, index, &mut issues);

        if !sample_ids.insert(slot.sample_id.as_str()) {
            push_unique(
                &mut issues,
                BufferIssue::DuplicateSampleId(slot.sample_id.clone()),
            );
        }
        if !positions.insert((slot.step_index, slot.evaluation_order)) {
            push_unique(
                &mut issues,
                BufferIssue::DuplicateEvaluationPosition(
                    slot.step_index,
                    slot.evaluation_order,
                ),
            );
        }
        let current_position = (slot.step_index, slot.evaluation_order);
        if previous_position.is_some_and(|previous| previous >= current_position) {
            push_unique(&mut issues, BufferIssue::UnsortedWindow(index));
        }
        previous_position = Some(current_position);

        if slot.solver_run_id != query.solver_run_id {
            push_unique(
                &mut issues,
                BufferIssue::RunMismatch(slot.sample_id.clone()),
            );
        }
        if slot.trajectory_id != query.trajectory_id {
            push_unique(
                &mut issues,
                BufferIssue::TrajectoryMismatch(slot.sample_id.clone()),
            );
        }
        if slot.solver_scheme_id != query.solver_scheme_id {
            push_unique(
                &mut issues,
                BufferIssue::SchemeMismatch(slot.sample_id.clone()),
            );
        }

        if contract.policy == BufferPolicy::StageExplicit {
            match (&query.tableau_sha256, &slot.tableau_sha256) {
                (Some(expected), Some(actual))
                    if is_hex_digest(actual) && expected == actual => {}
                _ => push_unique(
                    &mut issues,
                    BufferIssue::TableauMismatch(slot.sample_id.clone()),
                ),
            }
        }

        if slot.step_index > query.step_index {
            push_unique(
                &mut issues,
                BufferIssue::FutureStep(slot.sample_id.clone()),
            );
        } else if slot.step_index == query.step_index
            && slot.evaluation_order >= query.evaluation_order
        {
            push_unique(
                &mut issues,
                BufferIssue::StageNotYetComputed(slot.sample_id.clone()),
            );
        }

        if slot.step_index < query.step_index {
            let age = query.step_index - slot.step_index;
            if age > contract.max_age_steps {
                push_unique(
                    &mut issues,
                    BufferIssue::StaleSlot(slot.sample_id.clone()),
                );
            }
        }

        match slot.kind {
            SlotKind::AcceptedPhysicalState => {
                if contract.policy == BufferPolicy::AcceptedStateOnly
                    || contract.policy == BufferPolicy::PaperExact
                    || contract.policy == BufferPolicy::StageExplicit
                {
                    accepted_slot_ids.push(slot.sample_id.clone());
                }
            }
            SlotKind::ProvisionalStageState => {
                if contract.policy == BufferPolicy::AcceptedStateOnly {
                    push_unique(
                        &mut issues,
                        BufferIssue::ProvisionalSlotDisallowed(slot.sample_id.clone()),
                    );
                }
                provisional_slot_ids.push(slot.sample_id.clone());

                if slot.step_index == query.step_index
                    && slot.abscissa.compare(query.abscissa).is_gt()
                {
                    match contract.policy {
                        BufferPolicy::PaperExact => {
                            non_monotone_stage_slot_ids.push(slot.sample_id.clone());
                        }
                        BufferPolicy::AcceptedStateOnly => {}
                        BufferPolicy::StageExplicit => {
                            if !allowed_dependencies.contains(slot.stage_id.as_str()) {
                                push_unique(
                                    &mut issues,
                                    BufferIssue::FutureAbscissaWithoutDeclaredDependency(
                                        slot.sample_id.clone(),
                                    ),
                                );
                            } else {
                                non_monotone_stage_slot_ids.push(slot.sample_id.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    let disposition = if !issues.is_empty() {
        BufferDisposition::Invalid
    } else if slots.len() < contract.minimum_history_slots || slots.is_empty() {
        BufferDisposition::Unknown
    } else {
        BufferDisposition::Valid
    };

    BufferReport {
        disposition,
        accepted_slot_ids,
        provisional_slot_ids,
        non_monotone_stage_slot_ids,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLEAU: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn accepted(step: u64) -> StageStamp {
        StageStamp {
            sample_id: format!("accepted-{step}"),
            solver_run_id: "run-1".into(),
            trajectory_id: "trajectory-1".into(),
            solver_scheme_id: "ssp-rk3".into(),
            tableau_sha256: Some(TABLEAU.into()),
            step_index: step,
            evaluation_order: 0,
            stage_id: "accepted".into(),
            abscissa: RationalAbscissa::ONE,
            source_state_id: format!("state-{step}"),
            kind: SlotKind::AcceptedPhysicalState,
        }
    }

    fn provisional(step: u64, order: u32, stage_id: &str, c: RationalAbscissa) -> StageStamp {
        StageStamp {
            sample_id: format!("sample-{step}-{order}"),
            solver_run_id: "run-1".into(),
            trajectory_id: "trajectory-1".into(),
            solver_scheme_id: "ssp-rk3".into(),
            tableau_sha256: Some(TABLEAU.into()),
            step_index: step,
            evaluation_order: order,
            stage_id: stage_id.into(),
            abscissa: c,
            source_state_id: format!("predictor-{step}-{order}"),
            kind: SlotKind::ProvisionalStageState,
        }
    }

    fn query(order: u32, stage_id: &str, c: RationalAbscissa) -> QueryStamp {
        QueryStamp {
            solver_run_id: "run-1".into(),
            trajectory_id: "trajectory-1".into(),
            solver_scheme_id: "ssp-rk3".into(),
            tableau_sha256: Some(TABLEAU.into()),
            step_index: 7,
            evaluation_order: order,
            stage_id: stage_id.into(),
            abscissa: c,
            allowed_prior_stage_ids: Vec::new(),
        }
    }

    fn contract(policy: BufferPolicy) -> BufferContract {
        BufferContract {
            policy,
            max_age_steps: 4,
            minimum_history_slots: 1,
        }
    }

    #[test]
    fn paper_exact_preserves_non_monotone_ssprk_stage_abscissae() {
        // SSPRK-style stages: c = 0, 1, 1/2, computed in that order.
        let slots = vec![
            accepted(6),
            provisional(7, 1, "stage-1", RationalAbscissa::ZERO),
            provisional(7, 2, "stage-2", RationalAbscissa::ONE),
        ];
        let report = validate_buffer(
            &query(3, "stage-3", RationalAbscissa { numerator: 1, denominator: 2 }),
            &slots,
            contract(BufferPolicy::PaperExact),
        );
        assert_eq!(report.disposition, BufferDisposition::Valid);
        assert_eq!(report.non_monotone_stage_slot_ids, vec!["sample-7-2"]);
        assert_eq!(report.provisional_slot_ids.len(), 2);
    }

    #[test]
    fn accepted_state_only_rejects_provisional_slots() {
        let report = validate_buffer(
            &query(2, "stage-2", RationalAbscissa::ONE),
            &vec![accepted(6), provisional(7, 1, "stage-1", RationalAbscissa::ZERO)],
            contract(BufferPolicy::AcceptedStateOnly),
        );
        assert_eq!(report.disposition, BufferDisposition::Invalid);
        assert!(report.issues.contains(&BufferIssue::ProvisionalSlotDisallowed(
            "sample-7-1".into()
        )));
    }

    #[test]
    fn stage_explicit_rejects_future_abscissa_without_tableau_dependency() {
        let report = validate_buffer(
            &query(3, "stage-3", RationalAbscissa { numerator: 1, denominator: 2 }),
            &vec![provisional(7, 2, "stage-2", RationalAbscissa::ONE)],
            contract(BufferPolicy::StageExplicit),
        );
        assert_eq!(report.disposition, BufferDisposition::Invalid);
        assert!(report.issues.contains(
            &BufferIssue::FutureAbscissaWithoutDeclaredDependency("sample-7-2".into())
        ));
    }

    #[test]
    fn stage_explicit_accepts_declared_runge_kutta_dependency() {
        let mut q = query(3, "stage-3", RationalAbscissa { numerator: 1, denominator: 2 });
        q.allowed_prior_stage_ids = vec!["stage-2".into()];
        let report = validate_buffer(
            &q,
            &vec![
                provisional(7, 1, "stage-1", RationalAbscissa::ZERO),
                provisional(7, 2, "stage-2", RationalAbscissa::ONE),
            ],
            contract(BufferPolicy::StageExplicit),
        );
        assert_eq!(report.disposition, BufferDisposition::Valid);
        assert_eq!(report.non_monotone_stage_slot_ids, vec!["sample-7-2"]);
    }

    #[test]
    fn future_computation_order_is_rejected_even_when_abscissa_is_earlier() {
        let report = validate_buffer(
            &query(2, "stage-2", RationalAbscissa::ONE),
            &vec![provisional(7, 3, "stage-3", RationalAbscissa::ZERO)],
            contract(BufferPolicy::PaperExact),
        );
        assert_eq!(report.disposition, BufferDisposition::Invalid);
        assert!(report.issues.contains(
            &BufferIssue::StageNotYetComputed("sample-7-3".into())
        ));
    }

    #[test]
    fn future_step_is_rejected() {
        let report = validate_buffer(
            &query(1, "stage-1", RationalAbscissa::ZERO),
            &vec![provisional(8, 1, "stage-1", RationalAbscissa::ZERO)],
            contract(BufferPolicy::PaperExact),
        );
        assert_eq!(report.disposition, BufferDisposition::Invalid);
        assert!(report.issues.contains(&BufferIssue::FutureStep("sample-8-1".into())));
    }

    #[test]
    fn wrong_trajectory_and_solver_run_are_rejected() {
        let mut slot = accepted(6);
        slot.trajectory_id = "other-trajectory".into();
        slot.solver_run_id = "other-run".into();
        let report = validate_buffer(
            &query(1, "stage-1", RationalAbscissa::ZERO),
            &[slot],
            contract(BufferPolicy::AcceptedStateOnly),
        );
        assert_eq!(report.disposition, BufferDisposition::Invalid);
        assert!(report.issues.contains(&BufferIssue::TrajectoryMismatch("accepted-6".into())));
        assert!(report.issues.contains(&BufferIssue::RunMismatch("accepted-6".into())));
    }

    #[test]
    fn stale_history_is_rejected() {
        let report = validate_buffer(
            &query(1, "stage-1", RationalAbscissa::ZERO),
            &[accepted(1)],
            BufferContract {
                max_age_steps: 2,
                minimum_history_slots: 1,
                policy: BufferPolicy::AcceptedStateOnly,
            },
        );
        assert_eq!(report.disposition, BufferDisposition::Invalid);
        assert!(report.issues.contains(&BufferIssue::StaleSlot("accepted-1".into())));
    }

    #[test]
    fn malformed_rational_abscissa_is_rejected() {
        let report = validate_buffer(
            &query(2, "stage-2", RationalAbscissa::ONE),
            &[provisional(7, 1, "stage-1", RationalAbscissa { numerator: 2, denominator: 1 })],
            contract(BufferPolicy::PaperExact),
        );
        assert_eq!(report.disposition, BufferDisposition::Invalid);
        assert!(report.issues.contains(&BufferIssue::InvalidSlotAbscissa("sample-7-1".into())));
    }

    #[test]
    fn empty_history_is_unknown_not_a_valid_empty_result() {
        let report = validate_buffer(
            &query(1, "stage-1", RationalAbscissa::ZERO),
            &[],
            contract(BufferPolicy::PaperExact),
        );
        assert_eq!(report.disposition, BufferDisposition::Unknown);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn unsorted_input_window_is_rejected_not_silently_reordered() {
        let report = validate_buffer(
            &query(3, "stage-3", RationalAbscissa { numerator: 1, denominator: 2 }),
            &vec![
                provisional(7, 2, "stage-2", RationalAbscissa::ONE),
                provisional(7, 1, "stage-1", RationalAbscissa::ZERO),
            ],
            contract(BufferPolicy::PaperExact),
        );
        assert_eq!(report.disposition, BufferDisposition::Invalid);
        assert!(report.issues.contains(&BufferIssue::UnsortedWindow(1)));
    }

    #[test]
    fn stage_explicit_requires_exact_tableau_identity() {
        let mut slot = provisional(7, 1, "stage-1", RationalAbscissa::ZERO);
        slot.tableau_sha256 = Some("f".repeat(64));
        let report = validate_buffer(
            &query(2, "stage-2", RationalAbscissa::ONE),
            &[slot],
            contract(BufferPolicy::StageExplicit),
        );
        assert_eq!(report.disposition, BufferDisposition::Invalid);
        assert!(report.issues.contains(&BufferIssue::TableauMismatch("sample-7-1".into())));
    }

    #[test]
    fn duplicate_sample_and_evaluation_identity_are_rejected() {
        let first = provisional(7, 1, "stage-1", RationalAbscissa::ZERO);
        let mut duplicate = first.clone();
        duplicate.stage_id = "renamed-stage".into();
        duplicate.source_state_id = "different-state".into();
        let report = validate_buffer(
            &query(3, "stage-3", RationalAbscissa { numerator: 1, denominator: 2 }),
            &[first, duplicate],
            contract(BufferPolicy::PaperExact),
        );
        assert_eq!(report.disposition, BufferDisposition::Invalid);
        assert!(report.issues.contains(&BufferIssue::DuplicateSampleId("sample-7-1".into())));
        assert!(report.issues.contains(&BufferIssue::DuplicateEvaluationPosition(7, 1)));
    }
}
