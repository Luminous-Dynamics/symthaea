// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Product-facing type-state boundary for cyber-physical actuation.
//!
//! The lower IoT crates intentionally expose serializable policy decisions and
//! checkpoints because those objects are useful for audit, persistence, testing,
//! and independent verification. Serializable decision records must not become
//! ambient physical authority merely because their fields describe an allowed
//! evaluation.
//!
//! This crate therefore adds an opaque, non-deserializable path for code that may
//! eventually transmit a physical command:
//!
//! ```text
//! real authority + firmware + safety + exact plan/world evaluation
//!   -> ValidatedActuation                  (opaque, owns exact proposal)
//!   -> durable device-sequence + use/risk reservation
//!   -> persist exact combined checkpoint
//!   -> ArmedActuationPermit                (still not send-ready)
//!   -> fresh authority/firmware/safety/world revalidation
//!   -> ReadyActuationPermit                (opaque, affine)
//!   -> authenticated transport / device boundary
//! ```
//!
//! A valid transport session is deliberately not an input to validation here.
//! Authentication proves who carried bytes; it does not create actuation authority.
//! A future Xenia egress adapter should require both a [`ReadyActuationPermit`] and
//! Xenia's independently authenticated session evidence.
//!
//! Fresh preflight also distinguishes policy/safety rejection from evidence that
//! the device sequence may already have been accepted. If fresh runtime state says
//! `last_accepted_sequence >= command.sequence`, this crate keeps the reservation
//! charged as an ambiguous physical outcome rather than claiming the command was
//! never dispatched.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use symthaea_action_runtime::{GrantAccount, ReservationId, ReservationState};
use symthaea_authority::{
    AuthorityContext, AuthorityEpoch, CapabilityGrant, Digest32, GrantUseState,
    NegativeAuthorityFact,
};
use symthaea_iot_authority::{DeviceCommand, DeviceRuntimeState, SafetyEnvelope};
use symthaea_iot_durable_runtime::{
    DurableDispatchPermit, DurableEffectTransition, DurableIoTCheckpoint, DurableIoTError,
    DurableIoTHead, DurableUnknownPhysicalEffect, PendingDurableDispatch, prepare_durable_dispatch,
};
use symthaea_iot_execution::{
    BoundExecutionAdmission, BoundExecutionDecision, BoundExecutionDenyReason,
    PhysicalExecutionProposal, evaluate_bound_execution,
};
use thiserror::Error;

/// Opaque proof-in-process that the exact owned proposal passed the real
/// cyber-physical evaluator.
///
/// This type intentionally does **not** implement `Serialize`, `Deserialize`, or
/// `Clone`, has no public constructor, and owns the proposal that was evaluated.
/// Callers cannot pair a successful receipt with a substituted proposal later.
///
/// This token is still not a dispatch permit: no use/risk has been reserved and no
/// checkpoint has been made durable yet.
#[derive(Debug)]
pub struct ValidatedActuation {
    proposal: PhysicalExecutionProposal,
    admission: BoundExecutionAdmission,
}

impl ValidatedActuation {
    /// Exact execution-proposal commitment validated by the evaluator.
    pub fn proposal_digest(&self) -> Digest32 {
        self.admission.proposal_digest
    }

    /// Exact command commitment validated by the evaluator.
    pub fn command_digest(&self) -> Digest32 {
        self.admission.cyber_physical.command_digest
    }

    /// Device-domain command sequence validated by the evaluator.
    pub fn sequence(&self) -> u64 {
        self.admission.cyber_physical.accepted_sequence
    }

    /// Read-only proposal view for operator/audit UI before durable preparation.
    /// Possessing these bytes is not authority; physical I/O should require the
    /// later opaque permit types from this crate.
    pub fn proposal(&self) -> &PhysicalExecutionProposal {
        &self.proposal
    }
}

/// Run the real bound cyber-physical evaluator and, only on success, mint an
/// opaque [`ValidatedActuation`] that owns that exact proposal.
pub fn validate_actuation(
    grant: &CapabilityGrant,
    authority_context: AuthorityContext,
    negative_facts: &[NegativeAuthorityFact],
    proposal: PhysicalExecutionProposal,
    runtime: &DeviceRuntimeState,
    safety: &SafetyEnvelope,
) -> Result<ValidatedActuation, BoundExecutionDenyReason> {
    match evaluate_bound_execution(
        grant,
        authority_context,
        negative_facts,
        &proposal,
        runtime,
        safety,
    ) {
        BoundExecutionDecision::Allow(admission) => Ok(ValidatedActuation {
            proposal,
            admission,
        }),
        BoundExecutionDecision::Deny(reason) => Err(reason),
    }
}

/// Opaque validated dispatch waiting for its complete combined checkpoint to be
/// made durable.
///
/// The set of pre-existing reservation IDs is captured before lower-layer
/// preparation. It is later used to identify *this permit's* one new
/// `OutcomeUnknown` reservation without duplicating the lower runtime's private
/// reservation-ID format.
#[derive(Debug)]
pub struct PendingActuationPersistence {
    inner: PendingDurableDispatch,
    proposal: PhysicalExecutionProposal,
    prior_reservation_ids: BTreeSet<ReservationId>,
    checkpoint: DurableIoTCheckpoint,
}

impl PendingActuationPersistence {
    /// Exact combined checkpoint that must be persisted before this command can
    /// advance to an armed permit.
    pub fn checkpoint(&self) -> &DurableIoTCheckpoint {
        &self.checkpoint
    }

    /// Exact combined anti-rollback head the persistence owner must retain.
    pub const fn expected_head(&self) -> DurableIoTHead {
        self.inner.expected_head()
    }

    /// Exact validated proposal commitment.
    pub fn proposal_digest(&self) -> Digest32 {
        self.inner.proposal_digest()
    }

    /// Consume the pending state only when the persistence owner returns the exact
    /// combined head. A mismatched head returns this value unchanged.
    pub fn confirm_persisted(
        self,
        durable_head: DurableIoTHead,
    ) -> Result<ArmedActuationPermit, PendingActuationPersistence> {
        let PendingActuationPersistence {
            inner,
            proposal,
            prior_reservation_ids,
            checkpoint,
        } = self;

        match inner.confirm_persisted(durable_head) {
            Ok(inner) => Ok(ArmedActuationPermit {
                inner,
                proposal,
                prior_reservation_ids,
                checkpoint,
            }),
            Err(inner) => Err(PendingActuationPersistence {
                inner,
                proposal,
                prior_reservation_ids,
                checkpoint,
            }),
        }
    }
}

/// Prepare a real validated actuation for crash-safe dispatch.
///
/// This function consumes [`ValidatedActuation`], so the secure path cannot reuse
/// the same opaque validation token to mint multiple durable permits. The lower
/// runtime burns the device sequence and moves the action reservation to
/// `OutcomeUnknown` before returning its pending persistence object.
pub fn prepare_actuation_dispatch(
    grant: &CapabilityGrant,
    account: &mut GrantAccount,
    validated: ValidatedActuation,
    previous_checkpoint: Option<&DurableIoTCheckpoint>,
    trusted_previous_head: Option<DurableIoTHead>,
) -> Result<PendingActuationPersistence, ActuationError> {
    let ValidatedActuation {
        proposal,
        admission,
    } = validated;

    let prior_reservation_ids = account
        .snapshot()
        .reservations
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();

    let inner = prepare_durable_dispatch(
        grant,
        account,
        &admission,
        &proposal,
        previous_checkpoint,
        trusted_previous_head,
    )?;
    let checkpoint = inner.checkpoint().clone();

    Ok(PendingActuationPersistence {
        inner,
        proposal,
        prior_reservation_ids,
        checkpoint,
    })
}

/// A durably armed physical effect.
///
/// Persistence has happened, but this type intentionally exposes no command
/// getter: it is **not send-ready** until current authority and physical state are
/// checked again by [`Self::revalidate_before_send`].
#[derive(Debug)]
pub struct ArmedActuationPermit {
    inner: DurableDispatchPermit,
    proposal: PhysicalExecutionProposal,
    prior_reservation_ids: BTreeSet<ReservationId>,
    checkpoint: DurableIoTCheckpoint,
}

impl ArmedActuationPermit {
    /// Combined checkpoint head that armed this effect.
    pub const fn armed_head(&self) -> DurableIoTHead {
        self.inner.armed_head()
    }

    /// Proposal commitment retained across validation, persistence and preflight.
    pub fn proposal_digest(&self) -> Digest32 {
        self.inner.proposal_digest()
    }

    /// Revalidate current authority and safety immediately before transport egress.
    ///
    /// The authority kernel normally counts all reserved uses. Because this permit
    /// has already reserved exactly one use for itself, preflight first proves the
    /// exact newly-added reservation exists in `OutcomeUnknown`, then subtracts
    /// **only that one use** from the evaluator's temporary use-state view. This
    /// prevents a one-use grant from denying its own already-reserved permit while
    /// still counting every other reservation and delegation escrow.
    ///
    /// If fresh device state already reports this sequence or a later one as
    /// accepted, the result is [`PreflightOutcome::SequenceAmbiguous`]: capacity
    /// remains charged because the system no longer has proof of non-dispatch.
    pub fn revalidate_before_send(
        self,
        grant: &CapabilityGrant,
        account: &mut GrantAccount,
        now_unix_s: u64,
        current_epoch: AuthorityEpoch,
        negative_facts: &[NegativeAuthorityFact],
        runtime: &DeviceRuntimeState,
        safety: &SafetyEnvelope,
    ) -> Result<PreflightOutcome, ActuationError> {
        let ArmedActuationPermit {
            inner,
            proposal,
            prior_reservation_ids,
            checkpoint,
        } = self;

        if checkpoint.head() != inner.armed_head() {
            return Err(ActuationError::ArmedCheckpointMismatch);
        }
        if inner.proposal_digest() != proposal.digest() {
            return Err(ActuationError::ProposalBindingMismatch);
        }

        let restored = checkpoint.verify_as_trusted_head(grant, inner.armed_head())?;
        let snapshot = account.snapshot();
        if restored.snapshot() != snapshot {
            return Err(ActuationError::AccountCheckpointDiverged);
        }
        if checkpoint.device_sequence(&proposal.command.device) != Some(proposal.command.sequence) {
            return Err(ActuationError::DeviceSequenceBindingMismatch);
        }

        let mut new_reservations = snapshot
            .reservations
            .values()
            .filter(|reservation| !prior_reservation_ids.contains(&reservation.reservation_id));
        let reservation = new_reservations
            .next()
            .ok_or(ActuationError::MissingPermitReservation)?;
        if new_reservations.next().is_some() {
            return Err(ActuationError::AmbiguousPermitReservation);
        }
        if reservation.state != ReservationState::OutcomeUnknown {
            return Err(ActuationError::PermitReservationNotOutcomeUnknown);
        }
        if reservation.risk_charge != proposal.risk_charge {
            return Err(ActuationError::PermitRiskBindingMismatch);
        }

        // A device/gateway report at or beyond this sequence means the command may
        // already have crossed some dispatch boundary. Do not claim non-dispatch.
        if let Some(last_accepted_sequence) = runtime.last_accepted_sequence {
            if last_accepted_sequence >= proposal.command.sequence {
                return Ok(PreflightOutcome::SequenceAmbiguous(
                    PreflightSequenceAmbiguity {
                        last_accepted_sequence,
                        effect: inner.into_unknown(),
                    },
                ));
            }
        }

        let GrantUseState {
            committed,
            reserved,
        } = account.authority_use_state();
        let reserved_without_self = reserved
            .checked_sub(1)
            .ok_or(ActuationError::ReservedUseUnderflow)?;
        let authority_context = AuthorityContext {
            now_unix_s,
            current_epoch,
            use_state: GrantUseState {
                committed,
                reserved: reserved_without_self,
            },
        };

        match evaluate_bound_execution(
            grant,
            authority_context,
            negative_facts,
            &proposal,
            runtime,
            safety,
        ) {
            BoundExecutionDecision::Allow(fresh) => {
                if fresh.proposal_digest != inner.proposal_digest()
                    || fresh.cyber_physical.command_digest != proposal.command.digest()
                    || fresh.cyber_physical.accepted_sequence != proposal.command.sequence
                    || fresh.risk_charge != proposal.risk_charge
                {
                    return Err(ActuationError::FreshAdmissionBindingMismatch);
                }
                Ok(PreflightOutcome::Ready(ReadyActuationPermit {
                    inner,
                    proposal,
                    validated_at_unix_s: now_unix_s,
                }))
            }
            BoundExecutionDecision::Deny(reason) => {
                // No network I/O has occurred through this permit. With no device
                // sequence-acceptance evidence above, the reservation can be
                // reconciled as not dispatched; the durable device sequence remains
                // burned in the successor checkpoint.
                let transition = inner.proven_not_dispatched(account, grant)?;
                Ok(PreflightOutcome::Rejected(PreflightRejection {
                    reason,
                    transition,
                }))
            }
        }
    }
}

/// Result of the final fresh pre-send gate.
#[derive(Debug)]
pub enum PreflightOutcome {
    /// Current authority, firmware, safety and exact world commitment still match.
    Ready(ReadyActuationPermit),
    /// Fresh policy/safety denied before network I/O; use/risk was released in a
    /// successor checkpoint while the device sequence remained burned.
    Rejected(PreflightRejection),
    /// Fresh runtime state says this sequence (or a later one) was already accepted;
    /// outcome remains charged and must be reconciled from evidence.
    SequenceAmbiguous(PreflightSequenceAmbiguity),
}

/// Clean pre-send rejection and its crash-safe not-dispatched transition.
#[derive(Debug)]
pub struct PreflightRejection {
    /// Stable evaluator denial reason.
    pub reason: BoundExecutionDenyReason,
    /// Successor checkpoint that must be persisted before its released capacity is
    /// treated as the new durable truth.
    pub transition: DurableEffectTransition,
}

/// Device-sequence evidence made the effect ambiguous before this permit sent bytes.
#[derive(Debug)]
pub struct PreflightSequenceAmbiguity {
    /// Fresh device/gateway sequence observed at preflight.
    pub last_accepted_sequence: u64,
    /// Still-charged durable effect requiring later reconciliation.
    pub effect: DurableUnknownPhysicalEffect,
}

/// Final opaque permit for one exact command after durable preparation **and** fresh
/// pre-send revalidation.
///
/// This type is affine, non-serializable and non-deserializable. A future physical
/// egress API should consume it together with independently authenticated transport
/// evidence. It does not itself prove that a transport is live or that a device-side
/// hardware interlock remains satisfied after transmission.
#[derive(Debug)]
pub struct ReadyActuationPermit {
    inner: DurableDispatchPermit,
    proposal: PhysicalExecutionProposal,
    validated_at_unix_s: u64,
}

impl ReadyActuationPermit {
    /// Exact command that the egress adapter may transmit for this permit.
    pub fn command(&self) -> &DeviceCommand {
        &self.proposal.command
    }

    /// Exact proposal commitment carried through every security layer.
    pub fn proposal_digest(&self) -> Digest32 {
        self.inner.proposal_digest()
    }

    /// Combined durable head that armed this command.
    pub const fn armed_head(&self) -> DurableIoTHead {
        self.inner.armed_head()
    }

    /// Trusted wall-clock value used by the final pre-send evaluator.
    pub const fn validated_at_unix_s(&self) -> u64 {
        self.validated_at_unix_s
    }

    /// Transport/device result is ambiguous. Capacity remains charged.
    pub fn into_unknown(self) -> DurableUnknownPhysicalEffect {
        self.inner.into_unknown()
    }

    /// Positive evidence shows the physical effect was applied.
    pub fn observed_applied(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<DurableEffectTransition, ActuationError> {
        Ok(self.inner.observed_applied(account, grant)?)
    }

    /// Independent evidence proves the ready command never crossed the physical
    /// dispatch boundary.
    pub fn proven_not_dispatched(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<DurableEffectTransition, ActuationError> {
        Ok(self.inner.proven_not_dispatched(account, grant)?)
    }
}

/// Failure in the opaque validated-actuation composition layer.
#[derive(Debug, Error)]
pub enum ActuationError {
    /// Lower durable action/checkpoint layer failed.
    #[error("durable IoT runtime failed: {0}")]
    Durable(#[from] DurableIoTError),
    /// Stored combined checkpoint does not match the permit's armed head.
    #[error("armed permit and retained combined checkpoint diverged")]
    ArmedCheckpointMismatch,
    /// Stored proposal does not match the lower durable permit commitment.
    #[error("armed permit no longer binds the exact validated proposal")]
    ProposalBindingMismatch,
    /// In-memory account is not the exact state committed by the armed checkpoint.
    #[error("action account diverged from the armed durable checkpoint")]
    AccountCheckpointDiverged,
    /// Combined checkpoint did not burn the exact command sequence.
    #[error("armed checkpoint does not bind the exact device command sequence")]
    DeviceSequenceBindingMismatch,
    /// The lower preparation did not add the expected reservation.
    #[error("armed checkpoint is missing this permit's reservation")]
    MissingPermitReservation,
    /// More than one reservation appeared during one affine preparation.
    #[error("cannot uniquely identify this permit's reservation")]
    AmbiguousPermitReservation,
    /// The permit reservation is not conservatively charged as OutcomeUnknown.
    #[error("permit reservation is not in OutcomeUnknown state")]
    PermitReservationNotOutcomeUnknown,
    /// Reservation risk differs from the validated proposal's exact charge.
    #[error("permit reservation risk charge does not match validated proposal")]
    PermitRiskBindingMismatch,
    /// Exact permit reservation was proven, but aggregate reserved-use accounting
    /// could not subtract that one use.
    #[error("reserved-use accounting underflow during pre-send revalidation")]
    ReservedUseUnderflow,
    /// Fresh evaluator output did not reproduce the exact persisted proposal/command.
    #[error("fresh evaluator admission does not bind the armed command exactly")]
    FreshAdmissionBindingMismatch,
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use symthaea_authority::{Operation, PrincipalId, ResourceRef, RiskBudget, TaskId};
    use symthaea_iot_authority::{
        DEVICE_COMMAND_SCHEMA_VERSION, InclusiveRangeI64, SAFETY_ENVELOPE_SCHEMA_VERSION,
    };
    use symthaea_iot_execution::safety_world_digest;

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn risk(units: u64) -> RiskBudget {
        RiskBudget {
            mutation_units: units,
            ..RiskBudget::default()
        }
    }

    fn grant(max_uses: u32) -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "g-actuation",
            PrincipalId("human:operator".into()),
            PrincipalId("agent:irrigation".into()),
            AuthorityEpoch(21),
        );
        grant.audience = Some(PrincipalId("gateway:field-a".into()));
        grant.task = Some(TaskId("irrigate:zone-7".into()));
        grant.resources = BTreeSet::from([ResourceRef("iot:valve:72".into())]);
        grant.operations = BTreeSet::from([Operation("valve.open".into())]);
        grant.expires_at_unix_s = Some(10_000);
        grant.max_uses = max_uses;
        grant.risk_budget = risk(5);
        grant
    }

    fn safety() -> SafetyEnvelope {
        SafetyEnvelope {
            schema_version: SAFETY_ENVELOPE_SCHEMA_VERSION,
            policy_id: "safe-valve-open".into(),
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
                "tank_pressure_kpa_x100".into(),
                InclusiveRangeI64 {
                    min: 100,
                    max: 350_000,
                },
            )]),
        }
    }

    fn runtime() -> DeviceRuntimeState {
        DeviceRuntimeState {
            running_firmware: digest(7),
            last_accepted_sequence: Some(42),
            observations: BTreeMap::from([("tank_pressure_kpa_x100".into(), 210_000)]),
        }
    }

    fn proposal(runtime: &DeviceRuntimeState, safety: &SafetyEnvelope) -> PhysicalExecutionProposal {
        PhysicalExecutionProposal {
            command: DeviceCommand {
                schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
                command_id: "cmd-43".into(),
                actor: PrincipalId("agent:irrigation".into()),
                executor: PrincipalId("gateway:field-a".into()),
                task: Some(TaskId("irrigate:zone-7".into())),
                device: ResourceRef("iot:valve:72".into()),
                operation: Operation("valve.open".into()),
                expected_firmware: digest(7),
                sequence: 43,
                issued_at_unix_s: 4_995,
                expires_at_unix_s: 5_020,
                parameters: BTreeMap::from([("duration_ms".into(), 60_000)]),
            },
            plan_digest: Some(digest(3)),
            world_digest: safety_world_digest(runtime, safety).unwrap(),
            risk_charge: risk(1),
        }
    }

    fn context(grant: &CapabilityGrant, account: &GrantAccount) -> AuthorityContext {
        AuthorityContext {
            now_unix_s: 5_000,
            current_epoch: grant.authority_epoch,
            use_state: account.authority_use_state(),
        }
    }

    fn validated(
        grant: &CapabilityGrant,
        account: &GrantAccount,
        runtime: &DeviceRuntimeState,
        safety: &SafetyEnvelope,
    ) -> ValidatedActuation {
        validate_actuation(
            grant,
            context(grant, account),
            &[],
            proposal(runtime, safety),
            runtime,
            safety,
        )
        .expect("valid actuation")
    }

    fn armed(
        grant: &CapabilityGrant,
        account: &mut GrantAccount,
        runtime: &DeviceRuntimeState,
        safety: &SafetyEnvelope,
    ) -> ArmedActuationPermit {
        let validated = validated(grant, account, runtime, safety);
        let pending = prepare_actuation_dispatch(grant, account, validated, None, None).unwrap();
        let head = pending.expected_head();
        pending.confirm_persisted(head).expect("exact durable head")
    }

    fn bound_grant(max_uses: u32) -> (CapabilityGrant, DeviceRuntimeState, SafetyEnvelope) {
        let runtime = runtime();
        let safety = safety();
        let proposal = proposal(&runtime, &safety);
        let mut grant = grant(max_uses);
        grant.plan_digest = proposal.plan_digest;
        grant.world_digest = Some(proposal.world_digest);
        (grant, runtime, safety)
    }

    #[test]
    fn opaque_validation_is_minted_only_after_real_evaluator_success() {
        let (grant, runtime, safety) = bound_grant(1);
        let account = GrantAccount::new(&grant);
        let token = validated(&grant, &account, &runtime, &safety);
        assert_eq!(token.proposal_digest(), token.proposal().digest());
        assert_eq!(token.command_digest(), token.proposal().command.digest());

        let mut changed = proposal(&runtime, &safety);
        changed.world_digest = digest(99);
        assert!(
            validate_actuation(
                &grant,
                context(&grant, &account),
                &[],
                changed,
                &runtime,
                &safety,
            )
            .is_err()
        );
    }

    #[test]
    fn one_use_grant_does_not_deny_its_own_reserved_preflight() {
        let (grant, runtime, safety) = bound_grant(1);
        let mut account = GrantAccount::new(&grant);
        let permit = armed(&grant, &mut account, &runtime, &safety);
        assert_eq!(account.authority_use_state().reserved, 1);

        let outcome = permit
            .revalidate_before_send(
                &grant,
                &mut account,
                5_001,
                grant.authority_epoch,
                &[],
                &runtime,
                &safety,
            )
            .unwrap();
        let PreflightOutcome::Ready(ready) = outcome else {
            panic!("expected ready permit");
        };
        assert_eq!(ready.command().sequence, 43);
        assert_eq!(ready.validated_at_unix_s(), 5_001);
        assert_eq!(account.authority_use_state().reserved, 1);
    }

    #[test]
    fn changed_world_rejects_before_send_releases_capacity_but_keeps_sequence() {
        let (grant, runtime, safety) = bound_grant(1);
        let mut account = GrantAccount::new(&grant);
        let permit = armed(&grant, &mut account, &runtime, &safety);

        let mut changed = runtime.clone();
        changed
            .observations
            .insert("tank_pressure_kpa_x100".into(), 220_000);
        let outcome = permit
            .revalidate_before_send(
                &grant,
                &mut account,
                5_001,
                grant.authority_epoch,
                &[],
                &changed,
                &safety,
            )
            .unwrap();
        let PreflightOutcome::Rejected(rejected) = outcome else {
            panic!("expected fresh-world rejection");
        };
        assert_eq!(
            rejected.reason,
            BoundExecutionDenyReason::RuntimeWorldMismatch
        );
        assert_eq!(account.authority_use_state(), GrantUseState::default());
        assert_eq!(
            rejected
                .transition
                .checkpoint
                .device_sequence(&ResourceRef("iot:valve:72".into())),
            Some(43)
        );
    }

    #[test]
    fn revocation_after_persistence_blocks_send_and_releases_this_reservation() {
        let (grant, runtime, safety) = bound_grant(1);
        let mut account = GrantAccount::new(&grant);
        let permit = armed(&grant, &mut account, &runtime, &safety);
        let facts = [NegativeAuthorityFact::RevokeGrant {
            grant_digest: grant.digest(),
        }];

        let outcome = permit
            .revalidate_before_send(
                &grant,
                &mut account,
                5_001,
                grant.authority_epoch,
                &facts,
                &runtime,
                &safety,
            )
            .unwrap();
        assert!(matches!(outcome, PreflightOutcome::Rejected(_)));
        assert_eq!(account.authority_use_state(), GrantUseState::default());
    }

    #[test]
    fn already_accepted_sequence_stays_charged_as_ambiguous() {
        let (grant, runtime, safety) = bound_grant(1);
        let mut account = GrantAccount::new(&grant);
        let permit = armed(&grant, &mut account, &runtime, &safety);
        let head = permit.armed_head();
        let mut advanced = runtime.clone();
        advanced.last_accepted_sequence = Some(43);

        let outcome = permit
            .revalidate_before_send(
                &grant,
                &mut account,
                5_001,
                grant.authority_epoch,
                &[],
                &advanced,
                &safety,
            )
            .unwrap();
        let PreflightOutcome::SequenceAmbiguous(ambiguous) = outcome else {
            panic!("expected ambiguous sequence outcome");
        };
        assert_eq!(ambiguous.last_accepted_sequence, 43);
        assert_eq!(account.authority_use_state().reserved, 1);
        assert_eq!(ambiguous.effect.head(), head);
    }

    #[test]
    fn ready_permit_preserves_exact_durable_head_and_proposal() {
        let (grant, runtime, safety) = bound_grant(2);
        let mut account = GrantAccount::new(&grant);
        let permit = armed(&grant, &mut account, &runtime, &safety);
        let head = permit.armed_head();
        let proposal_digest = permit.proposal_digest();
        let outcome = permit
            .revalidate_before_send(
                &grant,
                &mut account,
                5_001,
                grant.authority_epoch,
                &[],
                &runtime,
                &safety,
            )
            .unwrap();
        let PreflightOutcome::Ready(ready) = outcome else {
            panic!("expected ready permit");
        };
        assert_eq!(ready.armed_head(), head);
        assert_eq!(ready.proposal_digest(), proposal_digest);
    }
}
