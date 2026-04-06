// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Saga pattern for multi-cluster workflow orchestration.
//!
//! Provides a pure state machine for executing multi-step cross-cluster
//! transactions with compensating actions on failure. No HDK dependency —
//! bridge coordinators call [`advance`] and execute the returned [`SagaAction`].
//!
//! ## Example
//!
//! ```rust,ignore
//! let mut saga = SagaDefinition::new("property-sale", steps, 300_000_000); // 5min timeout
//! loop {
//!     match saga::advance(&mut saga, now_us) {
//!         SagaAction::Dispatch(input) => {
//!             let result = dispatch_call_cross_cluster(&input, allowed);
//!             if result.success {
//!                 saga::record_success(&mut saga, result.response);
//!             } else {
//!                 saga::record_failure(&mut saga, result.error.unwrap_or_default());
//!             }
//!         }
//!         SagaAction::Complete => break,
//!         SagaAction::Compensate(actions) => { /* execute compensations */ break; }
//!         SagaAction::Timeout => break,
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Status of a single saga step.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SagaStepStatus {
    Pending,
    Executing,
    Completed,
    Failed,
    Compensating,
    Compensated,
    CompensationFailed,
}

/// A single step in a saga workflow.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SagaStep {
    /// Target cluster role name (e.g., "finance", "commons")
    pub cluster: String,
    /// Target zome name
    pub zome: String,
    /// Function to call for forward execution
    pub execute_fn: String,
    /// Function to call for compensation (rollback). None = no compensation needed.
    pub compensate_fn: Option<String>,
    /// MessagePack-encoded payload for execute_fn
    pub payload: Vec<u8>,
    /// Current status
    pub status: SagaStepStatus,
    /// Response from successful execution (MessagePack bytes)
    pub result: Option<Vec<u8>>,
    /// Error message on failure
    pub error: Option<String>,
    /// Timestamp when executed (microseconds since epoch)
    pub executed_at: Option<u64>,
}

/// Overall saga status.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SagaStatus {
    Created,
    Running,
    Completed,
    Failed,
    Compensating,
    Compensated,
    CompensationFailed,
}

/// A multi-step cross-cluster workflow definition.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SagaDefinition {
    /// Unique saga ID
    pub id: String,
    /// Human-readable name (e.g., "property-sale")
    pub name: String,
    /// Ordered steps to execute
    pub steps: Vec<SagaStep>,
    /// Index of the current step (0-based)
    pub current_step: usize,
    /// Overall saga status
    pub status: SagaStatus,
    /// Creation timestamp (microseconds)
    pub created_at: u64,
    /// Timeout duration (microseconds). 0 = no timeout.
    pub timeout_us: u64,
}

/// Action returned by [`advance`] telling the bridge coordinator what to do next.
#[derive(Debug, Clone)]
pub enum SagaAction {
    /// Dispatch this cross-cluster call and report result via [`record_success`]/[`record_failure`].
    Dispatch {
        role: String,
        zome: String,
        fn_name: String,
        payload: Vec<u8>,
    },
    /// Saga completed successfully.
    Complete,
    /// Saga failed — execute these compensation calls in order.
    Compensate(Vec<CompensationAction>),
    /// Saga timed out.
    Timeout,
}

/// A compensation action to undo a completed step.
#[derive(Debug, Clone)]
pub struct CompensationAction {
    pub role: String,
    pub zome: String,
    pub fn_name: String,
    pub payload: Vec<u8>,
    /// Index of the step being compensated
    pub step_index: usize,
}

impl SagaDefinition {
    /// Create a new saga with the given steps and timeout.
    pub fn new(name: impl Into<String>, steps: Vec<SagaStep>, now_us: u64, timeout_us: u64) -> Self {
        let name = name.into();
        let id = format!("saga-{}-{}", &name, now_us);
        Self {
            id,
            name,
            steps,
            current_step: 0,
            status: SagaStatus::Created,
            created_at: now_us,
            timeout_us,
        }
    }

    /// Number of completed steps.
    pub fn completed_count(&self) -> usize {
        self.steps.iter().filter(|s| s.status == SagaStepStatus::Completed).count()
    }

    /// Whether the saga is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            SagaStatus::Completed | SagaStatus::Compensated | SagaStatus::CompensationFailed
        )
    }
}

impl SagaStep {
    /// Create a new pending step.
    pub fn new(
        cluster: impl Into<String>,
        zome: impl Into<String>,
        execute_fn: impl Into<String>,
        compensate_fn: Option<String>,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            cluster: cluster.into(),
            zome: zome.into(),
            execute_fn: execute_fn.into(),
            compensate_fn,
            payload,
            status: SagaStepStatus::Pending,
            result: None,
            error: None,
            executed_at: None,
        }
    }
}

/// Advance the saga to its next action.
///
/// Returns `SagaAction::Dispatch` if there's a step to execute,
/// `SagaAction::Complete` if all steps succeeded,
/// `SagaAction::Timeout` if the saga has exceeded its timeout.
pub fn advance(saga: &mut SagaDefinition, now_us: u64) -> SagaAction {
    // Check timeout
    if saga.timeout_us > 0 && now_us > saga.created_at + saga.timeout_us {
        saga.status = SagaStatus::Failed;
        if let Some(step) = saga.steps.get_mut(saga.current_step) {
            if step.status == SagaStepStatus::Executing {
                step.status = SagaStepStatus::Failed;
                step.error = Some("Saga timeout".into());
            }
        }
        return SagaAction::Timeout;
    }

    // If already terminal, no-op
    if saga.is_terminal() {
        return SagaAction::Complete;
    }

    // If compensating, return compensation actions
    if saga.status == SagaStatus::Compensating {
        return build_compensations(saga);
    }

    // If all steps done, mark complete
    if saga.current_step >= saga.steps.len() {
        saga.status = SagaStatus::Completed;
        return SagaAction::Complete;
    }

    // Mark saga as running
    if saga.status == SagaStatus::Created {
        saga.status = SagaStatus::Running;
    }

    let step = &mut saga.steps[saga.current_step];

    match step.status {
        SagaStepStatus::Pending => {
            step.status = SagaStepStatus::Executing;
            step.executed_at = Some(now_us);
            SagaAction::Dispatch {
                role: step.cluster.clone(),
                zome: step.zome.clone(),
                fn_name: step.execute_fn.clone(),
                payload: step.payload.clone(),
            }
        }
        SagaStepStatus::Completed => {
            // Move to next step
            saga.current_step += 1;
            advance(saga, now_us)
        }
        SagaStepStatus::Failed => {
            // Start compensation
            saga.status = SagaStatus::Compensating;
            build_compensations(saga)
        }
        // Executing — waiting for result, caller should call record_success/record_failure
        _ => SagaAction::Dispatch {
            role: step.cluster.clone(),
            zome: step.zome.clone(),
            fn_name: step.execute_fn.clone(),
            payload: step.payload.clone(),
        },
    }
}

/// Record a successful step execution.
pub fn record_success(saga: &mut SagaDefinition, response: Option<Vec<u8>>) {
    if let Some(step) = saga.steps.get_mut(saga.current_step) {
        step.status = SagaStepStatus::Completed;
        step.result = response;
    }
    saga.current_step += 1;
}

/// Record a failed step execution.
pub fn record_failure(saga: &mut SagaDefinition, error: String) {
    if let Some(step) = saga.steps.get_mut(saga.current_step) {
        step.status = SagaStepStatus::Failed;
        step.error = Some(error);
    }
    saga.status = SagaStatus::Failed;
}

/// Build compensation actions for all completed steps in reverse order.
fn build_compensations(saga: &mut SagaDefinition) -> SagaAction {
    let mut actions = Vec::new();

    // Walk backwards through completed steps
    for i in (0..saga.steps.len()).rev() {
        let step = &saga.steps[i];
        if step.status == SagaStepStatus::Completed {
            if let Some(ref compensate_fn) = step.compensate_fn {
                actions.push(CompensationAction {
                    role: step.cluster.clone(),
                    zome: step.zome.clone(),
                    fn_name: compensate_fn.clone(),
                    payload: step.result.clone().unwrap_or_default(),
                    step_index: i,
                });
            }
        }
    }

    if actions.is_empty() {
        saga.status = SagaStatus::Compensated;
        SagaAction::Complete
    } else {
        SagaAction::Compensate(actions)
    }
}

/// Mark compensation as complete for a saga.
pub fn mark_compensated(saga: &mut SagaDefinition) {
    for step in &mut saga.steps {
        if step.status == SagaStepStatus::Completed {
            if step.compensate_fn.is_some() {
                step.status = SagaStepStatus::Compensated;
            }
        }
    }
    saga.status = SagaStatus::Compensated;
}

/// Mark compensation as failed for a saga.
pub fn mark_compensation_failed(saga: &mut SagaDefinition, error: String) {
    saga.status = SagaStatus::CompensationFailed;
    // Record error on the first compensating step
    for step in &mut saga.steps {
        if step.status == SagaStepStatus::Compensating {
            step.status = SagaStepStatus::CompensationFailed;
            step.error = Some(error);
            return;
        }
    }
}

// ============================================================================
// Saga Templates
// ============================================================================

/// Build a simple JSON payload as bytes from key-value pairs.
/// This avoids a runtime dependency on `serde_json` (which is dev-only).
fn json_kv(pairs: &[(&str, &str)]) -> Vec<u8> {
    let mut s = String::from("{");
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push('"');
        s.push_str(k);
        s.push_str("\":\"");
        s.push_str(v);
        s.push('"');
    }
    s.push('}');
    s.into_bytes()
}

/// Create a property sale saga template.
pub fn property_sale_saga(
    property_hash: String,
    buyer_did: String,
    _price_cents: u64,
    now_us: u64,
) -> SagaDefinition {
    let steps = vec![
        SagaStep::new(
            "commons", "property_transfer", "initiate_transfer",
            Some("cancel_transfer".into()),
            json_kv(&[("property_hash", &property_hash), ("buyer_did", &buyer_did)]),
        ),
        SagaStep::new(
            "finance", "payments", "escrow_payment",
            Some("release_escrow".into()),
            json_kv(&[("buyer_did", &buyer_did)]),
        ),
        SagaStep::new(
            "governance", "proposals", "verify_approval",
            None,
            json_kv(&[("property_hash", &property_hash)]),
        ),
        SagaStep::new(
            "commons", "property_transfer", "finalize_transfer",
            Some("revert_transfer".into()),
            json_kv(&[("property_hash", &property_hash), ("buyer_did", &buyer_did)]),
        ),
    ];
    SagaDefinition::new("property-sale", steps, now_us, 300_000_000)
}

/// Create an emergency response saga template.
pub fn emergency_response_saga(
    incident_id: String,
    _latitude: f64,
    _longitude: f64,
    now_us: u64,
) -> SagaDefinition {
    let steps = vec![
        SagaStep::new(
            "civic", "emergency_incidents", "declare_disaster",
            Some("end_disaster".into()),
            json_kv(&[("incident_id", &incident_id)]),
        ),
        SagaStep::new(
            "commons", "resource_mesh", "emergency_allocate",
            Some("emergency_deallocate".into()),
            json_kv(&[("incident_id", &incident_id)]),
        ),
        SagaStep::new(
            "finance", "treasury", "emergency_fund_release",
            Some("emergency_fund_reclaim".into()),
            json_kv(&[("incident_id", &incident_id)]),
        ),
    ];
    SagaDefinition::new("emergency-response", steps, now_us, 120_000_000)
}

/// Create a course completion saga template (Praxis -> Identity -> Finance).
pub fn course_completion_saga(
    course_id: String,
    student_did: String,
    now_us: u64,
) -> SagaDefinition {
    let steps = vec![
        SagaStep::new(
            "identity", "verifiable_credential", "issue_credential",
            Some("revoke_credential".into()),
            json_kv(&[("course_id", &course_id), ("student_did", &student_did), ("credential_type", "course_completion")]),
        ),
        SagaStep::new(
            "finance", "recognition", "award_recognition",
            None,
            json_kv(&[("student_did", &student_did)]),
        ),
    ];
    SagaDefinition::new("course-completion", steps, now_us, 60_000_000)
}

/// Create a justice enforcement saga template.
pub fn justice_enforcement_saga(
    case_id: String,
    defendant_did: String,
    _fine_cents: u64,
    now_us: u64,
) -> SagaDefinition {
    let steps = vec![
        SagaStep::new(
            "finance", "payments", "collect_fine",
            Some("refund_fine".into()),
            json_kv(&[("case_id", &case_id), ("defendant_did", &defendant_did)]),
        ),
        SagaStep::new(
            "commons", "property_registry", "apply_lien",
            Some("remove_lien".into()),
            json_kv(&[("case_id", &case_id), ("defendant_did", &defendant_did)]),
        ),
    ];
    SagaDefinition::new("justice-enforcement", steps, now_us, 180_000_000)
}

/// Create a carbon credit cycle saga template.
pub fn carbon_credit_saga(
    route_id: String,
    _distance_km: f64,
    now_us: u64,
) -> SagaDefinition {
    let steps = vec![
        SagaStep::new(
            "commons", "transport_impact", "record_transport",
            None,
            json_kv(&[("route_id", &route_id)]),
        ),
        SagaStep::new(
            "climate", "carbon", "mint_credit",
            Some("burn_credit".into()),
            json_kv(&[("route_id", &route_id)]),
        ),
    ];
    SagaDefinition::new("carbon-credit", steps, now_us, 120_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_700_000_000_000_000;

    fn test_steps() -> Vec<SagaStep> {
        vec![
            SagaStep::new("commons", "property_transfer", "initiate", Some("cancel".into()), vec![1]),
            SagaStep::new("finance", "payments", "escrow", Some("release".into()), vec![2]),
            SagaStep::new("governance", "proposals", "verify", None, vec![3]),
        ]
    }

    #[test]
    fn new_saga_is_created() {
        let saga = SagaDefinition::new("test", test_steps(), NOW, 60_000_000);
        assert_eq!(saga.status, SagaStatus::Created);
        assert_eq!(saga.current_step, 0);
        assert_eq!(saga.steps.len(), 3);
        assert!(!saga.is_terminal());
    }

    #[test]
    fn advance_dispatches_first_step() {
        let mut saga = SagaDefinition::new("test", test_steps(), NOW, 0);
        match advance(&mut saga, NOW) {
            SagaAction::Dispatch { role, zome, fn_name, .. } => {
                assert_eq!(role, "commons");
                assert_eq!(zome, "property_transfer");
                assert_eq!(fn_name, "initiate");
            }
            other => panic!("Expected Dispatch, got {:?}", other),
        }
        assert_eq!(saga.status, SagaStatus::Running);
        assert_eq!(saga.steps[0].status, SagaStepStatus::Executing);
    }

    #[test]
    fn full_success_flow() {
        let mut saga = SagaDefinition::new("test", test_steps(), NOW, 0);

        // Step 1: dispatch
        let _action = advance(&mut saga, NOW);
        record_success(&mut saga, Some(vec![10]));

        // Step 2: dispatch
        let _action = advance(&mut saga, NOW);
        record_success(&mut saga, Some(vec![20]));

        // Step 3: dispatch
        let _action = advance(&mut saga, NOW);
        record_success(&mut saga, Some(vec![30]));

        // Complete
        match advance(&mut saga, NOW) {
            SagaAction::Complete => {}
            other => panic!("Expected Complete, got {:?}", other),
        }
        assert_eq!(saga.status, SagaStatus::Completed);
        assert_eq!(saga.completed_count(), 3);
        assert!(saga.is_terminal());
    }

    #[test]
    fn failure_triggers_compensation() {
        let mut saga = SagaDefinition::new("test", test_steps(), NOW, 0);

        // Step 1: success
        advance(&mut saga, NOW);
        record_success(&mut saga, Some(vec![10]));

        // Step 2: failure
        advance(&mut saga, NOW);
        record_failure(&mut saga, "Payment failed".into());

        assert_eq!(saga.status, SagaStatus::Failed);

        // Advance should produce compensation
        match advance(&mut saga, NOW) {
            SagaAction::Compensate(actions) => {
                // Only step 1 needs compensation (step 2 failed, step 3 not started)
                assert_eq!(actions.len(), 1);
                assert_eq!(actions[0].fn_name, "cancel");
                assert_eq!(actions[0].step_index, 0);
                assert_eq!(actions[0].payload, vec![10]); // result from step 1
            }
            other => panic!("Expected Compensate, got {:?}", other),
        }
    }

    #[test]
    fn timeout_stops_saga() {
        let mut saga = SagaDefinition::new("test", test_steps(), NOW, 1_000_000); // 1s timeout
        advance(&mut saga, NOW);

        // Advance past timeout
        match advance(&mut saga, NOW + 2_000_000) {
            SagaAction::Timeout => {}
            other => panic!("Expected Timeout, got {:?}", other),
        }
        assert_eq!(saga.status, SagaStatus::Failed);
    }

    #[test]
    fn no_timeout_when_zero() {
        let mut saga = SagaDefinition::new("test", test_steps(), NOW, 0);
        // Even far in the future, no timeout
        match advance(&mut saga, NOW + 999_999_999_999) {
            SagaAction::Dispatch { .. } => {}
            other => panic!("Expected Dispatch, got {:?}", other),
        }
    }

    #[test]
    fn mark_compensated_updates_all_steps() {
        let mut saga = SagaDefinition::new("test", test_steps(), NOW, 0);
        advance(&mut saga, NOW);
        record_success(&mut saga, Some(vec![1]));
        advance(&mut saga, NOW);
        record_failure(&mut saga, "fail".into());

        mark_compensated(&mut saga);
        assert_eq!(saga.status, SagaStatus::Compensated);
        assert!(saga.is_terminal());
        assert_eq!(saga.steps[0].status, SagaStepStatus::Compensated);
    }

    #[test]
    fn step_without_compensate_fn_is_skipped() {
        let steps = vec![
            SagaStep::new("a", "z1", "exec", None, vec![]),
            SagaStep::new("b", "z2", "exec", Some("comp".into()), vec![]),
        ];
        let mut saga = SagaDefinition::new("test", steps, NOW, 0);

        // Complete both steps
        advance(&mut saga, NOW);
        record_success(&mut saga, Some(vec![1]));
        advance(&mut saga, NOW);
        record_success(&mut saga, Some(vec![2]));

        // Force failure to trigger compensation
        saga.status = SagaStatus::Compensating;
        match advance(&mut saga, NOW) {
            SagaAction::Compensate(actions) => {
                // Only step 2 has compensate_fn
                assert_eq!(actions.len(), 1);
                assert_eq!(actions[0].step_index, 1);
            }
            other => panic!("Expected Compensate, got {:?}", other),
        }
    }

    #[test]
    fn empty_saga_completes_immediately() {
        let mut saga = SagaDefinition::new("empty", vec![], NOW, 0);
        match advance(&mut saga, NOW) {
            SagaAction::Complete => {}
            other => panic!("Expected Complete, got {:?}", other),
        }
        assert_eq!(saga.status, SagaStatus::Completed);
    }

    // Template tests

    #[test]
    fn property_sale_template_has_correct_shape() {
        let saga = property_sale_saga("hash123".into(), "did:buyer".into(), 50000, NOW);
        assert_eq!(saga.name, "property-sale");
        assert_eq!(saga.steps.len(), 4);
        assert_eq!(saga.steps[0].cluster, "commons");
        assert_eq!(saga.steps[1].cluster, "finance");
        assert_eq!(saga.steps[2].cluster, "governance");
        assert_eq!(saga.steps[3].cluster, "commons");
        assert!(saga.steps[2].compensate_fn.is_none()); // verify is read-only
    }

    #[test]
    fn emergency_response_template_has_correct_shape() {
        let saga = emergency_response_saga("inc-001".into(), 32.9, -96.7, NOW);
        assert_eq!(saga.name, "emergency-response");
        assert_eq!(saga.steps.len(), 3);
        assert_eq!(saga.steps[0].cluster, "civic");
        assert_eq!(saga.steps[1].cluster, "commons");
        assert_eq!(saga.steps[2].cluster, "finance");
    }

    #[test]
    fn course_completion_template_has_correct_shape() {
        let saga = course_completion_saga("CS101".into(), "did:student".into(), NOW);
        assert_eq!(saga.name, "course-completion");
        assert_eq!(saga.steps.len(), 2);
        assert_eq!(saga.steps[0].cluster, "identity");
        assert_eq!(saga.steps[1].cluster, "finance");
    }

    #[test]
    fn justice_enforcement_template_has_correct_shape() {
        let saga = justice_enforcement_saga("case-42".into(), "did:defendant".into(), 10000, NOW);
        assert_eq!(saga.name, "justice-enforcement");
        assert_eq!(saga.steps.len(), 2);
    }

    #[test]
    fn carbon_credit_template_has_correct_shape() {
        let saga = carbon_credit_saga("route-1".into(), 42.5, NOW);
        assert_eq!(saga.name, "carbon-credit");
        assert_eq!(saga.steps.len(), 2);
        assert_eq!(saga.steps[0].cluster, "commons");
        assert_eq!(saga.steps[1].cluster, "climate");
    }

    #[test]
    fn completed_count_tracks_correctly() {
        let mut saga = SagaDefinition::new("test", test_steps(), NOW, 0);
        assert_eq!(saga.completed_count(), 0);

        advance(&mut saga, NOW);
        record_success(&mut saga, None);
        assert_eq!(saga.completed_count(), 1);

        advance(&mut saga, NOW);
        record_success(&mut saga, None);
        assert_eq!(saga.completed_count(), 2);
    }
}
