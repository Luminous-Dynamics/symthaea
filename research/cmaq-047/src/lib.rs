#![forbid(unsafe_code)]

//! Controlled Blackwell betting with a frozen, null-safe diagnostic action catalog.
//!
//! The OCO normal is selected by CMAQ-046 using only prior outcomes. The
//! controller then chooses an action whose authenticated action-specific null
//! mean set S_a satisfies h_(S_a)(w_t) <= h_S(w_t). Because the action is
//! selected predictably after w_t, the support residual against S retains
//! nonpositive conditional expectation under the action-conditional null.
//!
//! This crate checks the declared support inequalities and structural receipt
//! bindings. It does not verify signatures or prove that a physical sensor or
//! actuator actually satisfies its declared conditional-mean envelope.
//!
//! Operationally, one action may be outstanding at a time. An unknown outcome
//! blocks new actions until the original plan is reconciled with a result.

use std::collections::HashSet;

use cmaq_046_convex_target_approachability::{
    ApproachabilityError, ApproachabilityUpdate, BoxTarget, ConvexTargetApproachabilityMonitor,
};

const DIGEST_HEX_LEN: usize = 64;
const SUPPORT_TOLERANCE: f64 = 1e-12;
const MAX_ACTIONS: usize = 4_096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlledError {
    InvalidDigest(&'static str),
    TargetBindingMismatch,
    EmptyActionId,
    EmptyActionScope,
    EmptyCatalog,
    TooManyActions,
    DuplicateActionId,
    DimensionMismatch,
    InvalidDiagnosticSignature,
    NoNullSafeAction,
    ActionOutcomePending,
    NoPendingAction,
    PlanMismatch,
    ActionMismatch,
    SequenceOverflow,
    ArithmeticOverflow,
    MonitorQuarantined,
    Monitor(ApproachabilityError),
}

impl From<ApproachabilityError> for ControlledError {
    fn from(value: ApproachabilityError) -> Self {
        Self::Monitor(value)
    }
}

/// Externally attested action contract.
///
/// null_mean_box asserts that, under the relevant null and conditional on
/// prior history plus this action, the mean residual vector belongs to the
/// supplied box. authority_receipt_sha256 is a structural reference only:
/// this crate does not cryptographically authenticate it.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedDiagnosticAction {
    pub action_id: String,
    pub action_scope_id: String,
    pub null_mean_box: BoxTarget,
    /// Pre-registered alternative signature used only for action ranking.
    pub diagnostic_signature: Vec<f64>,
    pub action_contract_sha256: String,
    pub authority_receipt_sha256: String,
    pub action_scope_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionCatalog {
    target_sha256: String,
    catalog_sha256: String,
    authority_inventory_sha256: String,
    actions: Vec<CertifiedDiagnosticAction>,
}

impl ActionCatalog {
    pub fn new(
        target_sha256: String,
        catalog_sha256: String,
        authority_inventory_sha256: String,
        mut actions: Vec<CertifiedDiagnosticAction>,
        dimension: usize,
    ) -> Result<Self, ControlledError> {
        if actions.is_empty() {
            return Err(ControlledError::EmptyCatalog);
        }
        if actions.len() > MAX_ACTIONS {
            return Err(ControlledError::TooManyActions);
        }
        validate_digest(&target_sha256, "target_sha256")?;
        validate_digest(&catalog_sha256, "catalog_sha256")?;
        validate_digest(
            &authority_inventory_sha256,
            "authority_inventory_sha256",
        )?;

        let mut action_ids = HashSet::with_capacity(actions.len());
        for action in &actions {
            if action.action_id.trim().is_empty() {
                return Err(ControlledError::EmptyActionId);
            }
            if action.action_scope_id.trim().is_empty() {
                return Err(ControlledError::EmptyActionScope);
            }
            if !action_ids.insert(action.action_id.clone()) {
                return Err(ControlledError::DuplicateActionId);
            }
            if action.null_mean_box.dimension() != dimension
                || action.diagnostic_signature.len() != dimension
            {
                return Err(ControlledError::DimensionMismatch);
            }
            if action
                .diagnostic_signature
                .iter()
                .any(|value| !value.is_finite())
                || l2_norm(&action.diagnostic_signature) > 1.0 + SUPPORT_TOLERANCE
            {
                return Err(ControlledError::InvalidDiagnosticSignature);
            }
            validate_digest(
                &action.action_contract_sha256,
                "action_contract_sha256",
            )?;
            validate_digest(
                &action.authority_receipt_sha256,
                "authority_receipt_sha256",
            )?;
            validate_digest(&action.action_scope_sha256, "action_scope_sha256")?;
        }

        // Sorting once gives stable tie-breaking independent of caller order.
        actions.sort_by(|left, right| left.action_id.cmp(&right.action_id));

        Ok(Self {
            target_sha256,
            catalog_sha256,
            authority_inventory_sha256,
            actions,
        })
    }

    pub fn target_sha256(&self) -> &str {
        &self.target_sha256
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub fn authority_inventory_sha256(&self) -> &str {
        &self.authority_inventory_sha256
    }

    pub fn actions(&self) -> &[CertifiedDiagnosticAction] {
        &self.actions
    }
}

/// One-use proposal emitted before the selected diagnostic action is executed.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionPlan {
    pub sequence: u64,
    pub action_id: String,
    pub action_contract_sha256: String,
    pub authority_receipt_sha256: String,
    pub action_scope_sha256: String,
    pub target_sha256: String,
    pub catalog_sha256: String,
    pub normal: Vec<f64>,
    pub action_support_value: f64,
    pub target_support_value: f64,
    /// Nonnegative slack in the supporting-halfspace condition.
    pub null_safety_slack: f64,
    pub diagnostic_alignment: f64,
}

/// Result of reconciling exactly one action plan with its bounded observation.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledUpdate {
    pub action_plan: ActionPlan,
    pub evidence: ApproachabilityUpdate,
}

/// Controlled approachability monitor with one outstanding action plan.
///
/// The monitor deliberately has no cancel_pending operation. When an action
/// may have executed but its outcome is unknown, the caller must reconcile the
/// original plan rather than obtaining a second action that could double-spend
/// the same physical transition or invalidate the action-conditional model.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledBlackwellMonitor {
    monitor: ConvexTargetApproachabilityMonitor,
    catalog: ActionCatalog,
    pending: Option<ActionPlan>,
}

impl ControlledBlackwellMonitor {
    pub fn new(
        target: BoxTarget,
        observation_norm_bound: f64,
        alpha_numerator: u32,
        alpha_denominator: u32,
        target_sha256: String,
        catalog: ActionCatalog,
    ) -> Result<Self, ControlledError> {
        if target.dimension() != catalog.actions.first().map_or(0, |a| a.null_mean_box.dimension())
        {
            return Err(ControlledError::DimensionMismatch);
        }
        if catalog.target_sha256 != target_sha256 {
            return Err(ControlledError::TargetBindingMismatch);
        }
        validate_digest(&target_sha256, "target_sha256")?;
        let monitor = ConvexTargetApproachabilityMonitor::new(
            target,
            observation_norm_bound,
            alpha_numerator,
            alpha_denominator,
        )?;
        Ok(Self {
            monitor,
            catalog,
            pending: None,
        })
    }

    /// Selects an action using the current predictable OCO normal.
    ///
    /// Action a is eligible only when its certified conditional-mean set
    /// satisfies h_(S_a)(w) <= h_S(w). Among eligible actions, choose the one
    /// maximizing <w, diagnostic_signature_a>; ties are broken lexicographically
    /// by the action ID. The plan becomes pending before it is returned.
    pub fn propose_action(&mut self) -> Result<ActionPlan, ControlledError> {
        if self.pending.is_some() {
            return Err(ControlledError::ActionOutcomePending);
        }
        if self.monitor.is_quarantined() {
            return Err(ControlledError::MonitorQuarantined);
        }

        let sequence = self
            .monitor
            .observations_seen()
            .checked_add(1)
            .ok_or(ControlledError::SequenceOverflow)?;
        let normal = self.monitor.normal_for_next_observation().to_vec();
        let target_support = self.monitor.target().support(&normal)?;
        let mut best: Option<(&CertifiedDiagnosticAction, f64, f64, f64)> = None;

        for action in &self.catalog.actions {
            let action_support = action.null_mean_box.support(&normal)?;
            let slack = target_support - action_support;
            if slack < -SUPPORT_TOLERANCE {
                continue;
            }
            let alignment = dot(&normal, &action.diagnostic_signature)?;
            let replace = best.as_ref().is_none_or(
                |(best_action, _, _, best_alignment)| {
                    alignment > *best_alignment + SUPPORT_TOLERANCE
                        || ((alignment - *best_alignment).abs() <= SUPPORT_TOLERANCE
                            && action.action_id < best_action.action_id)
                },
            );
            if replace {
                best = Some((action, action_support, slack.max(0.0), alignment));
            }
        }

        let Some((action, action_support_value, null_safety_slack, diagnostic_alignment)) = best
        else {
            return Err(ControlledError::NoNullSafeAction);
        };

        let plan = ActionPlan {
            sequence,
            action_id: action.action_id.clone(),
            action_contract_sha256: action.action_contract_sha256.clone(),
            authority_receipt_sha256: action.authority_receipt_sha256.clone(),
            action_scope_sha256: action.action_scope_sha256.clone(),
            target_sha256: self.catalog.target_sha256.clone(),
            catalog_sha256: self.catalog.catalog_sha256.clone(),
            normal,
            action_support_value,
            target_support_value: target_support,
            null_safety_slack,
            diagnostic_alignment,
        };
        self.pending = Some(plan.clone());
        Ok(plan)
    }

    /// Reconciles one action result and feeds it into the CMAQ-046 e-process.
    ///
    /// A mismatch or invalid observation leaves the exact plan pending. This is
    /// fail-closed: do not request a second action until the real result for the
    /// outstanding action has been independently reconciled.
    pub fn reconcile_action_result(
        &mut self,
        plan: &ActionPlan,
        returned_action_id: &str,
        observation: &[f64],
    ) -> Result<ControlledUpdate, ControlledError> {
        let pending = self.pending.as_ref().ok_or(ControlledError::NoPendingAction)?;
        if pending != plan {
            return Err(ControlledError::PlanMismatch);
        }
        if returned_action_id != pending.action_id {
            return Err(ControlledError::ActionMismatch);
        }
        if pending.sequence != self.monitor.observations_seen() + 1 {
            return Err(ControlledError::SequenceOverflow);
        }

        // CMAQ-046 is transactional. On failure, leave the plan pending and do
        // not increment sequence or create any new action permission.
        let evidence = self.monitor.observe(observation)?;
        let action_plan = self
            .pending
            .take()
            .ok_or(ControlledError::NoPendingAction)?;
        Ok(ControlledUpdate {
            action_plan,
            evidence,
        })
    }

    pub fn pending_action(&self) -> Option<&ActionPlan> {
        self.pending.as_ref()
    }

    pub fn observations_seen(&self) -> u64 {
        self.monitor.observations_seen()
    }

    pub fn normal_for_next_action(&self) -> &[f64] {
        self.monitor.normal_for_next_observation()
    }

    pub fn log_e_value(&self) -> f64 {
        self.monitor.log_e_value()
    }

    pub fn is_quarantined(&self) -> bool {
        self.monitor.is_quarantined()
    }

    pub fn catalog(&self) -> &ActionCatalog {
        &self.catalog
    }
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), ControlledError> {
    let valid = value.len() == DIGEST_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid {
        Ok(())
    } else {
        Err(ControlledError::InvalidDigest(label))
    }
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, ControlledError> {
    if left.len() != right.len() {
        return Err(ControlledError::DimensionMismatch);
    }
    let mut result = 0.0;
    for (a, b) in left.iter().zip(right) {
        result += a * b;
    }
    if result.is_finite() {
        Ok(result)
    } else {
        Err(ControlledError::ArithmeticOverflow)
    }
}

fn l2_norm(vector: &[f64]) -> f64 {
    vector.iter().fold(0.0f64, |norm, value| norm.hypot(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn action(
        action_id: &str,
        lower: Vec<f64>,
        upper: Vec<f64>,
        signature: Vec<f64>,
        contract: &str,
    ) -> CertifiedDiagnosticAction {
        CertifiedDiagnosticAction {
            action_id: action_id.into(),
            action_scope_id: format!("scope-{action_id}"),
            null_mean_box: BoxTarget::new(lower, upper).unwrap(),
            diagnostic_signature: signature,
            action_contract_sha256: contract.into(),
            authority_receipt_sha256: SHA_D.into(),
            action_scope_sha256: SHA_E.into(),
        }
    }

    fn catalog() -> ActionCatalog {
        ActionCatalog::new(
            SHA_A.into(),
            SHA_B.into(),
            SHA_C.into(),
            vec![
                action(
                    "sensor-x-positive",
                    vec![-0.2, -0.05],
                    vec![0.04, 0.05],
                    vec![1.0, 0.0],
                    SHA_D,
                ),
                action(
                    "sensor-x-negative",
                    vec![-0.04, -0.05],
                    vec![0.2, 0.05],
                    vec![-1.0, 0.0],
                    SHA_E,
                ),
            ],
            2,
        )
        .unwrap()
    }

    fn controlled() -> ControlledBlackwellMonitor {
        ControlledBlackwellMonitor::new(
            BoxTarget::new(vec![-0.05, -0.05], vec![0.05, 0.05]).unwrap(),
            1.0,
            1,
            20,
            SHA_A.into(),
            catalog(),
        )
        .unwrap()
    }

    #[test]
    fn only_actions_satisfying_the_current_supporting_halfspace_are_eligible() {
        let mut monitor = controlled();
        let first = monitor.propose_action().unwrap();
        assert_eq!(first.normal, vec![0.0, 0.0]);
        // At w=0 both actions are admissible and lexical tie-breaking is stable.
        assert_eq!(first.action_id, "sensor-x-negative");
        assert!(first.null_safety_slack >= 0.0);
    }

    #[test]
    fn adaptive_witness_selects_directional_action_after_prior_observation() {
        let mut monitor = controlled();
        let first = monitor.propose_action().unwrap();
        monitor
            .reconcile_action_result(&first, &first.action_id, &[0.5, 0.0])
            .unwrap();

        let second = monitor.propose_action().unwrap();
        assert!(second.normal[0] > 0.0);
        assert_eq!(second.action_id, "sensor-x-positive");
        assert!(second.action_support_value <= second.target_support_value + SUPPORT_TOLERANCE);
    }

    #[test]
    fn no_second_action_is_allowed_while_first_result_is_unknown() {
        let mut monitor = controlled();
        monitor.propose_action().unwrap();
        assert_eq!(
            monitor.propose_action(),
            Err(ControlledError::ActionOutcomePending)
        );
        assert_eq!(monitor.observations_seen(), 0);
    }

    #[test]
    fn wrong_action_or_plan_does_not_consume_pending_result() {
        let mut monitor = controlled();
        let plan = monitor.propose_action().unwrap();
        assert_eq!(
            monitor.reconcile_action_result(&plan, "forged-action", &[0.0, 0.0]),
            Err(ControlledError::ActionMismatch)
        );
        let mut modified = plan.clone();
        modified.sequence += 1;
        assert_eq!(
            monitor.reconcile_action_result(&modified, &plan.action_id, &[0.0, 0.0]),
            Err(ControlledError::PlanMismatch)
        );
        assert_eq!(monitor.observations_seen(), 0);
        assert_eq!(monitor.pending_action(), Some(&plan));
    }

    #[test]
    fn invalid_observation_keeps_pending_plan_for_reconciliation() {
        let mut monitor = controlled();
        let plan = monitor.propose_action().unwrap();
        assert!(matches!(
            monitor.reconcile_action_result(&plan, &plan.action_id, &[2.0, 0.0]),
            Err(ControlledError::Monitor(
                ApproachabilityError::ObservationNormExceeded
            ))
        ));
        assert_eq!(monitor.observations_seen(), 0);
        assert_eq!(monitor.pending_action(), Some(&plan));

        monitor
            .reconcile_action_result(&plan, &plan.action_id, &[0.0, 0.0])
            .unwrap();
        assert_eq!(monitor.observations_seen(), 1);
        assert_eq!(monitor.pending_action(), None);
    }

    #[test]
    fn successful_result_updates_evidence_exactly_once_and_replay_fails() {
        let mut monitor = controlled();
        let plan = monitor.propose_action().unwrap();
        let result = monitor
            .reconcile_action_result(&plan, &plan.action_id, &[0.0, 0.0])
            .unwrap();
        assert_eq!(result.evidence.observations_seen, 1);
        assert_eq!(monitor.log_e_value(), result.evidence.log_e_value);
        assert_eq!(
            monitor.reconcile_action_result(&plan, &plan.action_id, &[0.0, 0.0]),
            Err(ControlledError::NoPendingAction)
        );
        assert_eq!(monitor.observations_seen(), 1);
    }

    #[test]
    fn action_catalog_rejects_duplicate_ids_and_malformed_receipts() {
        let duplicate = action(
            "sensor-x",
            vec![-0.1, -0.1],
            vec![0.1, 0.1],
            vec![1.0, 0.0],
            SHA_D,
        );
        assert_eq!(
            ActionCatalog::new(
                SHA_A.into(),
                SHA_B.into(),
                SHA_C.into(),
                vec![duplicate.clone(), duplicate],
                2,
            )
            .err(),
            Some(ControlledError::DuplicateActionId)
        );

        let mut malformed = action(
            "sensor-y",
            vec![-0.1, -0.1],
            vec![0.1, 0.1],
            vec![0.0, 1.0],
            SHA_D,
        );
        malformed.authority_receipt_sha256 = "bad".into();
        assert_eq!(
            ActionCatalog::new(
                SHA_A.into(),
                SHA_B.into(),
                SHA_C.into(),
                vec![malformed],
                2,
            )
            .err(),
            Some(ControlledError::InvalidDigest("authority_receipt_sha256"))
        );
    }

    #[test]
    fn action_catalog_must_bind_exact_target_and_dimension() {
        assert_eq!(
            ControlledBlackwellMonitor::new(
                BoxTarget::new(vec![-0.05, -0.05], vec![0.05, 0.05]).unwrap(),
                1.0,
                1,
                20,
                SHA_E.into(),
                catalog(),
            )
            .err(),
            Some(ControlledError::TargetBindingMismatch)
        );
        assert_eq!(
            ActionCatalog::new(
                SHA_A.into(),
                SHA_B.into(),
                SHA_C.into(),
                vec![action(
                    "wrong-dimension",
                    vec![-0.1],
                    vec![0.1],
                    vec![1.0, 0.0],
                    SHA_D
                )],
                2
            )
            .err(),
            Some(ControlledError::DimensionMismatch)
        );
    }

    #[test]
    fn no_admissible_action_fails_closed_without_pending_state() {
        let unsafe_action = action(
            "always-unsafe-for-positive-x",
            vec![-0.2, -0.05],
            vec![0.2, 0.05],
            vec![1.0, 0.0],
            SHA_D,
        );
        let action_catalog = ActionCatalog::new(
            SHA_A.into(),
            SHA_B.into(),
            SHA_C.into(),
            vec![unsafe_action],
            2,
        )
        .unwrap();
        let mut monitor = ControlledBlackwellMonitor::new(
            BoxTarget::new(vec![-0.05, -0.05], vec![0.05, 0.05]).unwrap(),
            1.0,
            1,
            20,
            SHA_A.into(),
            action_catalog,
        )
        .unwrap();

        // A zero normal is safe at t=1; first result creates a positive-x witness.
        let initial = monitor.propose_action().unwrap();
        monitor
            .reconcile_action_result(&initial, &initial.action_id, &[0.5, 0.0])
            .unwrap();
        assert_eq!(
            monitor.propose_action(),
            Err(ControlledError::NoNullSafeAction)
        );
        assert_eq!(monitor.observations_seen(), 1);
        assert_eq!(monitor.pending_action(), None);
    }

    #[test]
    fn catalog_actions_are_immutable_after_monitor_construction() {
        let monitor = controlled();
        assert_eq!(monitor.catalog().actions().len(), 2);
        assert_eq!(monitor.catalog().target_sha256(), SHA_A);
        assert_eq!(monitor.catalog().catalog_sha256(), SHA_B);
        assert_eq!(monitor.catalog().authority_inventory_sha256(), SHA_C);
    }
}
