// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-conservative runtime bridge for cyber-physical effects.
//!
//! The critical ordering is deliberately stronger than ordinary request/response
//! execution:
//!
//! ```text
//! bound execution admission
//!   -> reserve one use + exact risk charge
//!   -> transition reservation to OutcomeUnknown
//!   -> construct anti-rollback checkpoint
//!   -> host durably persists that checkpoint
//!   -> only then mint an affine DispatchPermit
//!   -> external effect attempt
//!   -> reconcile Applied | proven NotDispatched | remain Unknown
//! ```
//!
//! Persisting `OutcomeUnknown` *before* dispatch closes the dangerous crash window
//! where a command could reach the device but the durable account still looked
//! cancellable. A crash before actual dispatch is intentionally conservative: the
//! reservation remains charged until independent evidence proves no effect occurred.
//!
//! This crate does not perform persistence or actuator I/O. The persistence owner
//! must retain/authenticate the latest [`CheckpointHead`] outside the checkpoint
//! itself. Matching a supplied head is an ordering/type-state check, not proof that
//! storage hardware actually honored a durability request.

#![deny(unsafe_code)]

use std::fmt::Write as _;

use symthaea_action_checkpoint::{
    CheckpointError, CheckpointHead, GrantAccountCheckpoint,
};
use symthaea_action_runtime::{
    ExecutionId, GrantAccount, ReservationId, ReservationState, RuntimeAccountingError,
};
use symthaea_authority::{CapabilityGrant, Digest32};
use symthaea_iot_execution::BoundExecutionAdmission;
use thiserror::Error;

/// Error at the IoT/runtime composition boundary.
#[derive(Debug, Error)]
pub enum IoTRuntimeError {
    #[error("execution admission does not bind the supplied capability grant")]
    AdmissionGrantMismatch,
    #[error("runtime account does not bind the supplied capability grant")]
    AccountGrantMismatch,
    #[error("generation-zero preparation requires a fresh grant account")]
    GenesisAccountNotFresh,
    #[error("checkpoint and external trusted head must be supplied together")]
    IncompleteCheckpointBase,
    #[error("supplied checkpoint is not the externally trusted checkpoint head")]
    TrustedHeadMismatch,
    #[error("runtime account state diverged from the trusted checkpoint snapshot")]
    AccountCheckpointDiverged,
    #[error("reservation is not in the expected outcome-unknown state")]
    ReservationNotOutcomeUnknown,
    #[error("runtime accounting failed: {0}")]
    Runtime(#[from] RuntimeAccountingError),
    #[error("checkpoint operation failed: {0}")]
    Checkpoint(#[from] CheckpointError),
}

/// Reservation state waiting for an external persistence owner to confirm the
/// exact anti-rollback checkpoint before dispatch authority is exposed.
///
/// Intentionally not `Clone`: the normal path consumes this value into exactly
/// one [`DispatchPermit`].
#[derive(Debug)]
pub struct PendingDispatchPersistence {
    reservation_id: ReservationId,
    execution_id: ExecutionId,
    proposal_digest: Digest32,
    checkpoint: GrantAccountCheckpoint,
    expected_head: CheckpointHead,
}

impl PendingDispatchPersistence {
    /// Exact checkpoint that must be made durable before dispatch.
    pub fn checkpoint(&self) -> &GrantAccountCheckpoint {
        &self.checkpoint
    }

    /// Head the persistence owner must retain/authenticate externally.
    pub fn expected_head(&self) -> CheckpointHead {
        self.expected_head
    }

    pub fn reservation_id(&self) -> &ReservationId {
        &self.reservation_id
    }

    pub fn execution_id(&self) -> &ExecutionId {
        &self.execution_id
    }

    pub fn proposal_digest(&self) -> Digest32 {
        self.proposal_digest
    }

    /// Consume the pending state only when the persistence owner returns the exact
    /// checkpoint head it claims is now durable.
    ///
    /// On mismatch the original pending value is returned unchanged, so the caller
    /// may repair/retry persistence without minting a dispatch permit.
    pub fn confirm_persisted(
        self,
        durable_head: CheckpointHead,
    ) -> Result<DispatchPermit, PendingDispatchPersistence> {
        if durable_head != self.expected_head {
            return Err(self);
        }
        Ok(DispatchPermit {
            reservation_id: self.reservation_id,
            execution_id: self.execution_id,
            proposal_digest: self.proposal_digest,
            armed_checkpoint: self.checkpoint,
            armed_head: self.expected_head,
        })
    }
}

/// Affine authorization to attempt one already-reserved physical effect.
///
/// The reservation is already persisted as `OutcomeUnknown` before this type can
/// exist. Dropping the permit therefore cannot restore authority; capacity remains
/// charged until explicit reconciliation.
#[derive(Debug)]
pub struct DispatchPermit {
    reservation_id: ReservationId,
    execution_id: ExecutionId,
    proposal_digest: Digest32,
    armed_checkpoint: GrantAccountCheckpoint,
    armed_head: CheckpointHead,
}

impl DispatchPermit {
    pub fn reservation_id(&self) -> &ReservationId {
        &self.reservation_id
    }

    pub fn execution_id(&self) -> &ExecutionId {
        &self.execution_id
    }

    pub fn proposal_digest(&self) -> Digest32 {
        self.proposal_digest
    }

    pub fn armed_head(&self) -> CheckpointHead {
        self.armed_head
    }

    /// Device result is ambiguous. No runtime mutation is needed: the durable
    /// pre-dispatch state already says `OutcomeUnknown` and remains fully charged.
    pub fn into_unknown(self) -> UnknownPhysicalEffect {
        UnknownPhysicalEffect {
            reservation_id: self.reservation_id,
            execution_id: self.execution_id,
            proposal_digest: self.proposal_digest,
            checkpoint: self.armed_checkpoint,
            head: self.armed_head,
        }
    }

    /// The effect is positively observed as applied. Commit the charged use/risk
    /// and produce the exact successor checkpoint that must be persisted.
    pub fn observed_applied(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<EffectTransition, IoTRuntimeError> {
        ensure_account_matches_checkpoint(account, &self.armed_checkpoint)?;
        account.reconcile_applied(&self.reservation_id)?;
        make_transition(
            account,
            grant,
            self.armed_checkpoint,
            self.reservation_id,
            self.execution_id,
            self.proposal_digest,
            EffectOutcome::Applied,
        )
    }

    /// Host can independently prove the effect was never dispatched. Release the
    /// precharged unknown reservation and produce a successor checkpoint.
    ///
    /// A timeout, missing acknowledgement, or "probably not sent" is insufficient
    /// evidence for this path; those cases must remain [`UnknownPhysicalEffect`].
    pub fn proven_not_dispatched(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<EffectTransition, IoTRuntimeError> {
        ensure_account_matches_checkpoint(account, &self.armed_checkpoint)?;
        account.reconcile_not_applied(&self.reservation_id)?;
        make_transition(
            account,
            grant,
            self.armed_checkpoint,
            self.reservation_id,
            self.execution_id,
            self.proposal_digest,
            EffectOutcome::NotDispatched,
        )
    }
}

/// Persisted ambiguous physical effect. The reservation remains charged.
#[derive(Debug)]
pub struct UnknownPhysicalEffect {
    reservation_id: ReservationId,
    execution_id: ExecutionId,
    proposal_digest: Digest32,
    checkpoint: GrantAccountCheckpoint,
    head: CheckpointHead,
}

impl UnknownPhysicalEffect {
    pub fn reservation_id(&self) -> &ReservationId {
        &self.reservation_id
    }

    pub fn head(&self) -> CheckpointHead {
        self.head
    }

    /// Later evidence proves the ambiguous effect was applied.
    pub fn reconcile_applied(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<EffectTransition, IoTRuntimeError> {
        ensure_account_matches_checkpoint(account, &self.checkpoint)?;
        account.reconcile_applied(&self.reservation_id)?;
        make_transition(
            account,
            grant,
            self.checkpoint,
            self.reservation_id,
            self.execution_id,
            self.proposal_digest,
            EffectOutcome::Applied,
        )
    }

    /// Later evidence proves the ambiguous effect was not applied.
    pub fn reconcile_not_applied(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<EffectTransition, IoTRuntimeError> {
        ensure_account_matches_checkpoint(account, &self.checkpoint)?;
        account.reconcile_not_applied(&self.reservation_id)?;
        make_transition(
            account,
            grant,
            self.checkpoint,
            self.reservation_id,
            self.execution_id,
            self.proposal_digest,
            EffectOutcome::NotDispatched,
        )
    }
}

/// Reconciled outcome recorded in a successor checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectOutcome {
    Applied,
    NotDispatched,
}

/// Successor checkpoint produced after effect reconciliation.
#[derive(Debug)]
pub struct EffectTransition {
    pub outcome: EffectOutcome,
    pub reservation_id: ReservationId,
    pub execution_id: ExecutionId,
    pub proposal_digest: Digest32,
    pub checkpoint: GrantAccountCheckpoint,
    pub head: CheckpointHead,
}

/// Precharge and checkpoint an admitted physical effect before dispatch.
///
/// The reservation is moved to `OutcomeUnknown` *before* the checkpoint is
/// constructed. Therefore the returned checkpoint, once durably persisted, is
/// already conservative against a crash immediately after external dispatch.
pub fn prepare_dispatch(
    grant: &CapabilityGrant,
    account: &mut GrantAccount,
    admission: &BoundExecutionAdmission,
    previous_checkpoint: Option<&GrantAccountCheckpoint>,
    trusted_previous_head: Option<CheckpointHead>,
) -> Result<PendingDispatchPersistence, IoTRuntimeError> {
    if admission.cyber_physical.grant_digest != grant.digest() {
        return Err(IoTRuntimeError::AdmissionGrantMismatch);
    }
    if account.snapshot().grant_digest != grant.digest() {
        return Err(IoTRuntimeError::AccountGrantMismatch);
    }

    validate_checkpoint_base(
        grant,
        account,
        previous_checkpoint,
        trusted_previous_head,
    )?;

    let reservation_id = ReservationId(format!(
        "iot-reservation:{}",
        digest_hex(admission.proposal_digest)
    ));
    let execution_id = ExecutionId(format!(
        "iot-execution:{}",
        digest_hex(admission.proposal_digest)
    ));

    account.reserve_execution(
        reservation_id.clone(),
        execution_id.clone(),
        admission.risk_charge,
    )?;
    // Arm uncertainty before dispatch. If anything fails after this point, the
    // reservation remains charged rather than being silently released.
    account.mark_outcome_unknown(&reservation_id)?;

    let checkpoint = match previous_checkpoint {
        Some(previous) => GrantAccountCheckpoint::successor(previous, grant, account.snapshot())?,
        None => GrantAccountCheckpoint::first(grant, account.snapshot())?,
    };
    let expected_head = checkpoint.head()?;

    debug_assert_eq!(
        checkpoint
            .snapshot
            .reservations
            .get(&reservation_id)
            .map(|reservation| reservation.state),
        Some(ReservationState::OutcomeUnknown)
    );

    Ok(PendingDispatchPersistence {
        reservation_id,
        execution_id,
        proposal_digest: admission.proposal_digest,
        checkpoint,
        expected_head,
    })
}

fn validate_checkpoint_base(
    grant: &CapabilityGrant,
    account: &GrantAccount,
    previous_checkpoint: Option<&GrantAccountCheckpoint>,
    trusted_previous_head: Option<CheckpointHead>,
) -> Result<(), IoTRuntimeError> {
    match (previous_checkpoint, trusted_previous_head) {
        (None, None) => {
            if account.snapshot() != GrantAccount::new(grant).snapshot() {
                return Err(IoTRuntimeError::GenesisAccountNotFresh);
            }
        }
        (Some(previous), Some(trusted_head)) => {
            previous.verify_payload(grant)?;
            if previous.head()? != trusted_head {
                return Err(IoTRuntimeError::TrustedHeadMismatch);
            }
            if account.snapshot() != previous.snapshot {
                return Err(IoTRuntimeError::AccountCheckpointDiverged);
            }
        }
        _ => return Err(IoTRuntimeError::IncompleteCheckpointBase),
    }
    Ok(())
}

fn ensure_account_matches_checkpoint(
    account: &GrantAccount,
    checkpoint: &GrantAccountCheckpoint,
) -> Result<(), IoTRuntimeError> {
    if account.snapshot() != checkpoint.snapshot {
        return Err(IoTRuntimeError::AccountCheckpointDiverged);
    }
    Ok(())
}

fn make_transition(
    account: &GrantAccount,
    grant: &CapabilityGrant,
    previous: GrantAccountCheckpoint,
    reservation_id: ReservationId,
    execution_id: ExecutionId,
    proposal_digest: Digest32,
    outcome: EffectOutcome,
) -> Result<EffectTransition, IoTRuntimeError> {
    let checkpoint = GrantAccountCheckpoint::successor(&previous, grant, account.snapshot())?;
    let head = checkpoint.head()?;
    Ok(EffectTransition {
        outcome,
        reservation_id,
        execution_id,
        proposal_digest,
        checkpoint,
        head,
    })
}

fn digest_hex(Digest32(bytes): Digest32) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use symthaea_action_runtime::ReservationState;
    use symthaea_authority::{
        AuthorityContext, AuthorityEpoch, GrantUseState, Operation, PrincipalId, ResourceRef,
        RiskBudget, TaskId,
    };
    use symthaea_iot_authority::{
        DEVICE_COMMAND_SCHEMA_VERSION, DeviceCommand, DeviceRuntimeState, InclusiveRangeI64,
        SAFETY_ENVELOPE_SCHEMA_VERSION, SafetyEnvelope,
    };
    use symthaea_iot_execution::{
        BoundExecutionDecision, PhysicalExecutionProposal, evaluate_bound_execution,
        safety_world_digest,
    };

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn risk(units: u64) -> RiskBudget {
        RiskBudget {
            mutation_units: units,
            ..RiskBudget::default()
        }
    }

    fn grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "iot-g1",
            PrincipalId("human:operator".into()),
            PrincipalId("agent:irrigation".into()),
            AuthorityEpoch(12),
        );
        grant.audience = Some(PrincipalId("gateway:field-a".into()));
        grant.task = Some(TaskId("irrigate:zone-7".into()));
        grant.resources = BTreeSet::from([ResourceRef("iot:valve:72".into())]);
        grant.operations = BTreeSet::from([Operation("valve.open".into())]);
        grant.expires_at_unix_s = Some(10_000);
        grant.max_uses = 3;
        grant.risk_budget = risk(3);
        grant
    }

    fn context(grant: &CapabilityGrant, account: &GrantAccount) -> AuthorityContext {
        AuthorityContext {
            now_unix_s: 5_000,
            current_epoch: grant.authority_epoch,
            use_state: account.authority_use_state(),
        }
    }

    fn command(sequence: u64) -> DeviceCommand {
        DeviceCommand {
            schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
            command_id: format!("cmd-{sequence}"),
            actor: PrincipalId("agent:irrigation".into()),
            executor: PrincipalId("gateway:field-a".into()),
            task: Some(TaskId("irrigate:zone-7".into())),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            expected_firmware: digest(7),
            sequence,
            issued_at_unix_s: 4_990,
            expires_at_unix_s: 5_030,
            parameters: BTreeMap::from([("duration_ms".into(), 60_000)]),
        }
    }

    fn safety() -> SafetyEnvelope {
        SafetyEnvelope {
            schema_version: SAFETY_ENVELOPE_SCHEMA_VERSION,
            policy_id: "safe-open-v1".into(),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            allowed_firmware: BTreeSet::from([digest(7)]),
            parameter_ranges: BTreeMap::from([(
                "duration_ms".into(),
                InclusiveRangeI64 {
                    min: 1_000,
                    max: 120_000,
                },
            )]),
            required_observations: BTreeMap::from([(
                "pressure_x100".into(),
                InclusiveRangeI64 {
                    min: 100,
                    max: 350_000,
                },
            )]),
        }
    }

    fn runtime(sequence: u64) -> DeviceRuntimeState {
        DeviceRuntimeState {
            running_firmware: digest(7),
            last_accepted_sequence: Some(sequence - 1),
            observations: BTreeMap::from([("pressure_x100".into(), 210_000)]),
        }
    }

    fn admission(
        grant: &CapabilityGrant,
        account: &GrantAccount,
        sequence: u64,
    ) -> BoundExecutionAdmission {
        let runtime = runtime(sequence);
        let safety = safety();
        let proposal = PhysicalExecutionProposal {
            command: command(sequence),
            plan_digest: Some(digest(3)),
            world_digest: safety_world_digest(&runtime, &safety).unwrap(),
            risk_charge: risk(1),
        };
        let decision = evaluate_bound_execution(
            grant,
            context(grant, account),
            &[],
            &proposal,
            &runtime,
            &safety,
        );
        let BoundExecutionDecision::Allow(admission) = decision else {
            panic!("expected bound execution admission");
        };
        admission
    }

    #[test]
    fn dispatch_is_armed_unknown_before_permit_exists() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let admission = admission(&grant, &account, 1);
        let pending = prepare_dispatch(&grant, &mut account, &admission, None, None).unwrap();

        let reservation = pending
            .checkpoint()
            .snapshot
            .reservations
            .get(pending.reservation_id())
            .unwrap();
        assert_eq!(reservation.state, ReservationState::OutcomeUnknown);
        assert_eq!(account.authority_use_state().reserved, 1);
    }

    #[test]
    fn wrong_persistence_head_cannot_mint_dispatch_permit() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let admission = admission(&grant, &account, 1);
        let pending = prepare_dispatch(&grant, &mut account, &admission, None, None).unwrap();
        let wrong = CheckpointHead {
            sequence: pending.expected_head().sequence,
            digest: digest(99),
        };
        let pending = pending.confirm_persisted(wrong).unwrap_err();
        assert_eq!(account.authority_use_state().reserved, 1);
        assert_eq!(pending.expected_head().sequence, 0);
    }

    #[test]
    fn ambiguous_result_remains_fully_charged() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let admission = admission(&grant, &account, 1);
        let pending = prepare_dispatch(&grant, &mut account, &admission, None, None).unwrap();
        let permit = pending
            .confirm_persisted(pending.expected_head())
            .expect("exact durable head");
        let unknown = permit.into_unknown();
        assert_eq!(account.authority_use_state().reserved, 1);
        assert_eq!(unknown.head().sequence, 0);
    }

    #[test]
    fn observed_effect_commits_and_chains_successor_checkpoint() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let admission = admission(&grant, &account, 1);
        let pending = prepare_dispatch(&grant, &mut account, &admission, None, None).unwrap();
        let armed_head = pending.expected_head();
        let permit = pending.confirm_persisted(armed_head).unwrap();
        let transition = permit.observed_applied(&mut account, &grant).unwrap();

        assert_eq!(transition.outcome, EffectOutcome::Applied);
        assert_eq!(account.authority_use_state().committed, 1);
        assert_eq!(account.authority_use_state().reserved, 0);
        assert!(
            transition
                .checkpoint
                .verify_against_head(&grant, Some(armed_head))
                .is_ok()
        );
    }

    #[test]
    fn proven_not_dispatched_is_the_only_release_path_from_permit() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let admission = admission(&grant, &account, 1);
        let pending = prepare_dispatch(&grant, &mut account, &admission, None, None).unwrap();
        let permit = pending
            .confirm_persisted(pending.expected_head())
            .expect("exact durable head");
        let transition = permit.proven_not_dispatched(&mut account, &grant).unwrap();

        assert_eq!(transition.outcome, EffectOutcome::NotDispatched);
        assert_eq!(account.authority_use_state(), GrantUseState::default());
    }

    #[test]
    fn same_execution_proposal_cannot_multiply_reservations() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let admission = admission(&grant, &account, 1);
        let pending = prepare_dispatch(&grant, &mut account, &admission, None, None).unwrap();
        let checkpoint = pending.checkpoint().clone();
        let head = pending.expected_head();

        let error = prepare_dispatch(
            &grant,
            &mut account,
            &admission,
            Some(&checkpoint),
            Some(head),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            IoTRuntimeError::Runtime(RuntimeAccountingError::DuplicateReservation)
        ));
    }

    #[test]
    fn continuation_requires_exact_external_trusted_head() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let first_admission = admission(&grant, &account, 1);
        let pending = prepare_dispatch(&grant, &mut account, &first_admission, None, None).unwrap();
        let armed_head = pending.expected_head();
        let permit = pending.confirm_persisted(armed_head).unwrap();
        let transition = permit.observed_applied(&mut account, &grant).unwrap();

        let second_admission = admission(&grant, &account, 2);
        let wrong_head = CheckpointHead {
            sequence: transition.head.sequence,
            digest: digest(88),
        };
        let error = prepare_dispatch(
            &grant,
            &mut account,
            &second_admission,
            Some(&transition.checkpoint),
            Some(wrong_head),
        )
        .unwrap_err();
        assert!(matches!(error, IoTRuntimeError::TrustedHeadMismatch));
    }
}
