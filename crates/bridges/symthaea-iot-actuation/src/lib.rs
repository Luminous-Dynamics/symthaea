// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Product-facing type-state boundary for cyber-physical actuation.
//!
//! Serializable lower-layer authority/safety decisions are useful audit records,
//! but they are not capabilities. The consequential path in this crate requires
//! two opaque inputs before a command can eventually reach transport:
//!
//! 1. an [`ActuationPolicyHandle`] selected from one anti-rollback policy registry;
//! 2. successful execution of the real authority + firmware + safety + world
//!    evaluator over the exact owned proposal.
//!
//! The resulting path is:
//!
//! ```text
//! current policy-registry handle
//!   + real authority/firmware/safety/world evaluation
//!   -> ValidatedActuation              (opaque; owns exact proposal + policy binding)
//!   -> durable device sequence + use/risk reservation
//!   -> persist exact combined checkpoint
//!   -> ArmedActuationPermit            (still exposes no command bytes)
//!   -> fresh policy-registry + authority + firmware + safety/world preflight
//!   -> ReadyActuationPermit            (opaque; affine; exact command exposed)
//!   -> authenticated transport / device boundary
//! ```
//!
//! The policy registry, Xenia transport authentication, device attestation and
//! physical authority remain deliberately distinct. A future egress adapter should
//! require this crate's [`ReadyActuationPermit`] plus independent authenticated
//! session/device evidence. Connectivity never creates physical authority.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use symthaea_action_runtime::{GrantAccount, ReservationId, ReservationState};
use symthaea_authority::{
    AuthorityContext, AuthorityEpoch, CapabilityGrant, Digest32, GrantUseState,
    NegativeAuthorityFact,
};
use symthaea_iot_authority::{DeviceCommand, DeviceRuntimeState};
use symthaea_iot_durable_runtime::{
    DurableDispatchPermit, DurableEffectTransition, DurableIoTCheckpoint, DurableIoTError,
    DurableIoTHead, DurableUnknownPhysicalEffect, PendingDurableDispatch, prepare_durable_dispatch,
};
use symthaea_iot_execution::{
    BoundExecutionAdmission, BoundExecutionDecision, BoundExecutionDenyReason,
    PhysicalExecutionProposal, evaluate_bound_execution,
};
use symthaea_iot_policy::{
    ActuationPolicyError, ActuationPolicyHandle, ActuationPolicyHead, ActuationPolicyRegistry,
};
use thiserror::Error;

/// Exact trusted-policy selection retained across validation, persistence and
/// pre-send revalidation.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PolicyBinding {
    policy_id: String,
    revision: u64,
    policy_digest: Digest32,
    registry_head: ActuationPolicyHead,
}

impl PolicyBinding {
    fn from_handle(handle: &ActuationPolicyHandle<'_>) -> Self {
        Self {
            policy_id: handle.policy().policy_id.clone(),
            revision: handle.policy().revision,
            policy_digest: handle.policy_digest(),
            registry_head: handle.registry_head(),
        }
    }
}

/// Opaque proof-in-process that the exact owned proposal passed the real
/// cyber-physical evaluator under one exact registry-issued policy.
///
/// This type intentionally does **not** implement `Serialize`, `Deserialize`, or
/// `Clone`, has no public constructor, and owns the exact proposal that was
/// evaluated. A successful serializable lower-layer receipt therefore cannot be
/// paired with another proposal or another safety policy later.
#[derive(Debug)]
pub struct ValidatedActuation {
    proposal: PhysicalExecutionProposal,
    admission: BoundExecutionAdmission,
    policy: PolicyBinding,
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

    /// Exact policy commitment used for admission.
    pub fn policy_digest(&self) -> Digest32 {
        self.policy.policy_digest
    }

    /// Exact anti-rollback policy-registry generation used for admission.
    pub fn policy_registry_head(&self) -> ActuationPolicyHead {
        self.policy.registry_head
    }

    /// Read-only proposal view for operator/audit UI before durable preparation.
    /// Possessing these bytes is not authority; physical I/O should require the
    /// later opaque permit types from this crate.
    pub fn proposal(&self) -> &PhysicalExecutionProposal {
        &self.proposal
    }
}

/// Validate one exact proposal using a registry-issued policy handle.
///
/// The policy handle must have been selected at the same trusted time carried by
/// `authority_context`; this prevents a handle minted under an old-but-once-valid
/// policy window from being replayed into a later admission. Risk charge and
/// command lifetime are policy-derived constraints rather than free caller choices.
pub fn validate_actuation(
    grant: &CapabilityGrant,
    authority_context: AuthorityContext,
    negative_facts: &[NegativeAuthorityFact],
    proposal: PhysicalExecutionProposal,
    runtime: &DeviceRuntimeState,
    policy: &ActuationPolicyHandle<'_>,
) -> Result<ValidatedActuation, ActuationValidationError> {
    validate_initial_policy_binding(&proposal, authority_context, policy)?;

    match evaluate_bound_execution(
        grant,
        authority_context,
        negative_facts,
        &proposal,
        runtime,
        policy.safety(),
    ) {
        BoundExecutionDecision::Allow(admission) => Ok(ValidatedActuation {
            proposal,
            admission,
            policy: PolicyBinding::from_handle(policy),
        }),
        BoundExecutionDecision::Deny(reason) => {
            Err(ActuationValidationError::ExecutionDenied(reason))
        }
    }
}

fn validate_initial_policy_binding(
    proposal: &PhysicalExecutionProposal,
    authority_context: AuthorityContext,
    policy: &ActuationPolicyHandle<'_>,
) -> Result<(), ActuationValidationError> {
    if policy.selected_at_unix_s() != authority_context.now_unix_s {
        return Err(ActuationValidationError::PolicySelectionTimeMismatch {
            selected: policy.selected_at_unix_s(),
            evaluation: authority_context.now_unix_s,
        });
    }
    validate_proposal_against_policy(proposal, policy).map_err(ActuationValidationError::Policy)
}

fn validate_proposal_against_policy(
    proposal: &PhysicalExecutionProposal,
    policy: &ActuationPolicyHandle<'_>,
) -> Result<(), ActuationPolicyBindingDenyReason> {
    if proposal.command.device != policy.policy().device {
        return Err(ActuationPolicyBindingDenyReason::DeviceMismatch);
    }
    if proposal.command.operation != policy.policy().operation {
        return Err(ActuationPolicyBindingDenyReason::OperationMismatch);
    }
    if proposal.risk_charge != policy.risk_charge() {
        return Err(ActuationPolicyBindingDenyReason::RiskChargeMismatch);
    }
    let command_lifetime = proposal
        .command
        .expires_at_unix_s
        .checked_sub(proposal.command.issued_at_unix_s)
        .ok_or(ActuationPolicyBindingDenyReason::MalformedCommandLifetime)?;
    if command_lifetime > policy.max_command_lifetime_s() {
        return Err(ActuationPolicyBindingDenyReason::CommandLifetimeExceeded {
            proposed: command_lifetime,
            maximum: policy.max_command_lifetime_s(),
        });
    }
    Ok(())
}

/// Failure before the first opaque validation token is minted.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ActuationValidationError {
    /// Policy handle was selected for another trusted wall-clock instant.
    #[error("policy selection time {selected} does not match evaluation time {evaluation}")]
    PolicySelectionTimeMismatch { selected: u64, evaluation: u64 },
    /// Proposal does not obey the selected configured policy.
    #[error("proposal violates selected actuation policy: {0:?}")]
    Policy(ActuationPolicyBindingDenyReason),
    /// The lower authority/firmware/safety/world evaluator denied the proposal.
    #[error("cyber-physical execution denied: {0:?}")]
    ExecutionDenied(BoundExecutionDenyReason),
}

/// Stable reason a proposal failed the configured-policy binding layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActuationPolicyBindingDenyReason {
    /// Selected policy protects another resource.
    DeviceMismatch,
    /// Selected policy protects another operation.
    OperationMismatch,
    /// Proposal tried to reserve a consequence charge other than the configured one.
    RiskChargeMismatch,
    /// Command expiry precedes issue time.
    MalformedCommandLifetime,
    /// Command validity interval is broader than policy permits.
    CommandLifetimeExceeded { proposed: u64, maximum: u64 },
}

/// Opaque validated dispatch waiting for its complete combined checkpoint to be
/// made durable.
#[derive(Debug)]
pub struct PendingActuationPersistence {
    inner: PendingDurableDispatch,
    proposal: PhysicalExecutionProposal,
    policy: PolicyBinding,
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

    /// Exact configured policy commitment retained by this pending effect.
    pub fn policy_digest(&self) -> Digest32 {
        self.policy.policy_digest
    }

    /// Consume pending state only when the persistence owner returns the exact
    /// combined head. A mismatched head returns this value unchanged in a box so
    /// the normal success-path `Result` remains compact.
    pub fn confirm_persisted(
        self,
        durable_head: DurableIoTHead,
    ) -> Result<ArmedActuationPermit, Box<PendingActuationPersistence>> {
        let PendingActuationPersistence {
            inner,
            proposal,
            policy,
            prior_reservation_ids,
            checkpoint,
        } = self;

        match inner.confirm_persisted(durable_head) {
            Ok(inner) => Ok(ArmedActuationPermit {
                inner,
                proposal,
                policy,
                prior_reservation_ids,
                checkpoint,
            }),
            Err(inner) => Err(Box::new(PendingActuationPersistence {
                inner,
                proposal,
                policy,
                prior_reservation_ids,
                checkpoint,
            })),
        }
    }
}

/// Prepare a real policy-bound validated actuation for crash-safe dispatch.
///
/// This consumes [`ValidatedActuation`], so one opaque validation token cannot
/// mint multiple durable permits. The lower runtime burns the device sequence and
/// moves the action reservation to `OutcomeUnknown` before returning.
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
        policy,
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
        policy,
        prior_reservation_ids,
        checkpoint,
    })
}

/// A durably armed physical effect.
///
/// Persistence has happened, but this type deliberately exposes no command getter:
/// it is not send-ready until current policy and current physical authority/safety
/// are checked again by [`Self::revalidate_before_send`].
#[derive(Debug)]
pub struct ArmedActuationPermit {
    inner: DurableDispatchPermit,
    proposal: PhysicalExecutionProposal,
    policy: PolicyBinding,
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

    /// Policy commitment retained across validation, persistence and preflight.
    pub fn policy_digest(&self) -> Digest32 {
        self.policy.policy_digest
    }

    /// Policy-registry generation retained by this armed effect.
    pub fn policy_registry_head(&self) -> ActuationPolicyHead {
        self.policy.registry_head
    }

    /// Revalidate policy, authority and safety immediately before transport egress.
    ///
    /// The authority kernel normally counts every reserved use. Because this permit
    /// has already reserved exactly one use for itself, preflight first proves the
    /// exact newly-added reservation exists in `OutcomeUnknown`, then subtracts
    /// **only that one use** from the temporary evaluator view. Every other
    /// reservation and delegation escrow remains charged.
    ///
    /// Device sequence evidence dominates clean rejection: if fresh gateway/device
    /// state already reports this sequence or a later one accepted, the effect stays
    /// charged as ambiguous rather than being released.
    pub fn revalidate_before_send(
        self,
        grant: &CapabilityGrant,
        account: &mut GrantAccount,
        now_unix_s: u64,
        current_epoch: AuthorityEpoch,
        negative_facts: &[NegativeAuthorityFact],
        runtime: &DeviceRuntimeState,
        policy_registry: &ActuationPolicyRegistry,
    ) -> Result<PreflightOutcome, ActuationError> {
        let ArmedActuationPermit {
            inner,
            proposal,
            policy,
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

        if let Some(last_accepted_sequence) = runtime
            .last_accepted_sequence
            .filter(|last| *last >= proposal.command.sequence)
        {
            return Ok(PreflightOutcome::SequenceAmbiguous(Box::new(
                PreflightSequenceAmbiguity {
                    last_accepted_sequence,
                    effect: inner.into_unknown(),
                },
            )));
        }

        let fresh_policy = match fresh_policy_handle(&policy, &proposal, now_unix_s, policy_registry)
        {
            Ok(handle) => handle,
            Err(reason) => {
                let transition = inner.proven_not_dispatched(account, grant)?;
                return Ok(PreflightOutcome::Rejected(Box::new(PreflightRejection {
                    reason: PreflightRejectionReason::Policy(reason),
                    transition,
                })));
            }
        };

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
            fresh_policy.safety(),
        ) {
            BoundExecutionDecision::Allow(fresh) => {
                if fresh.proposal_digest != inner.proposal_digest()
                    || fresh.cyber_physical.command_digest != proposal.command.digest()
                    || fresh.cyber_physical.accepted_sequence != proposal.command.sequence
                    || fresh.risk_charge != proposal.risk_charge
                {
                    return Err(ActuationError::FreshAdmissionBindingMismatch);
                }
                Ok(PreflightOutcome::Ready(Box::new(ReadyActuationPermit {
                    inner,
                    proposal,
                    policy,
                    validated_at_unix_s: now_unix_s,
                })))
            }
            BoundExecutionDecision::Deny(reason) => {
                let transition = inner.proven_not_dispatched(account, grant)?;
                Ok(PreflightOutcome::Rejected(Box::new(PreflightRejection {
                    reason: PreflightRejectionReason::Execution(reason),
                    transition,
                })))
            }
        }
    }
}

fn fresh_policy_handle<'a>(
    binding: &PolicyBinding,
    proposal: &PhysicalExecutionProposal,
    now_unix_s: u64,
    registry: &'a ActuationPolicyRegistry,
) -> Result<ActuationPolicyHandle<'a>, PolicyPreflightDenyReason> {
    if registry.head() != binding.registry_head {
        return Err(PolicyPreflightDenyReason::RegistryGenerationChanged {
            admitted: binding.registry_head,
            current: registry.head(),
        });
    }
    let handle = registry
        .policy(&binding.policy_id, now_unix_s)
        .map_err(PolicyPreflightDenyReason::PolicyUnavailable)?;
    if handle.policy_digest() != binding.policy_digest
        || handle.policy().revision != binding.revision
    {
        return Err(PolicyPreflightDenyReason::PolicyBindingChanged);
    }
    validate_proposal_against_policy(proposal, &handle)
        .map_err(PolicyPreflightDenyReason::ProposalNoLongerMatchesPolicy)?;
    Ok(handle)
}

/// Result of the final fresh pre-send gate.
#[derive(Debug)]
pub enum PreflightOutcome {
    /// Current policy, authority, firmware, safety and world commitment still match.
    Ready(Box<ReadyActuationPermit>),
    /// Fresh policy or execution checks denied before network I/O. Use/risk was
    /// released in a successor checkpoint while device sequence remained burned.
    Rejected(Box<PreflightRejection>),
    /// Fresh runtime state says this sequence (or a later one) was already accepted;
    /// outcome remains charged and must be reconciled from independent evidence.
    SequenceAmbiguous(Box<PreflightSequenceAmbiguity>),
}

/// Clean pre-send rejection and its crash-safe not-dispatched transition.
#[derive(Debug)]
pub struct PreflightRejection {
    /// Stable reason the current pre-send gate denied the effect.
    pub reason: PreflightRejectionReason,
    /// Successor checkpoint that must be persisted before released capacity is
    /// treated as the new durable truth.
    pub transition: DurableEffectTransition,
}

/// Distinguishes configured-policy withdrawal/change from ordinary execution denial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreflightRejectionReason {
    /// Current registry/policy no longer authorizes the original policy binding.
    Policy(PolicyPreflightDenyReason),
    /// Current authority/firmware/safety/world evaluator denied the exact proposal.
    Execution(BoundExecutionDenyReason),
}

/// Stable configured-policy reason for a clean pre-send rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyPreflightDenyReason {
    /// Policy registry generation changed after the effect was armed. Conservatively
    /// re-admit under the new generation instead of carrying an old permit across it.
    RegistryGenerationChanged {
        admitted: ActuationPolicyHead,
        current: ActuationPolicyHead,
    },
    /// Exact policy cannot currently be selected (expired/retired/revoked/snapshot stale).
    PolicyUnavailable(ActuationPolicyError),
    /// Exact policy bytes/revision no longer match the admitted commitment.
    PolicyBindingChanged,
    /// The retained proposal no longer satisfies the selected configured policy.
    ProposalNoLongerMatchesPolicy(ActuationPolicyBindingDenyReason),
}

/// Device-sequence evidence made the effect ambiguous before this permit sent bytes.
#[derive(Debug)]
pub struct PreflightSequenceAmbiguity {
    /// Fresh device/gateway sequence observed at preflight.
    pub last_accepted_sequence: u64,
    /// Still-charged durable effect requiring later reconciliation.
    pub effect: DurableUnknownPhysicalEffect,
}

/// Final opaque permit for one exact command after durable preparation and fresh
/// policy + authority/safety revalidation.
///
/// This type is affine, non-serializable and non-deserializable. A future physical
/// egress API should consume it together with independently authenticated transport
/// and device-posture evidence.
#[derive(Debug)]
pub struct ReadyActuationPermit {
    inner: DurableDispatchPermit,
    proposal: PhysicalExecutionProposal,
    policy: PolicyBinding,
    validated_at_unix_s: u64,
}

impl ReadyActuationPermit {
    /// Exact command that an egress adapter may transmit for this permit.
    pub fn command(&self) -> &DeviceCommand {
        &self.proposal.command
    }

    /// Exact proposal commitment carried through every security layer.
    pub fn proposal_digest(&self) -> Digest32 {
        self.inner.proposal_digest()
    }

    /// Exact configured policy commitment carried through every security layer.
    pub fn policy_digest(&self) -> Digest32 {
        self.policy.policy_digest
    }

    /// Policy registry generation used by both initial admission and fresh preflight.
    pub fn policy_registry_head(&self) -> ActuationPolicyHead {
        self.policy.registry_head
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
        self.inner.observed_applied(account, grant).map_err(Into::into)
    }

    /// Independent evidence proves the ready command never crossed the physical
    /// dispatch boundary.
    pub fn proven_not_dispatched(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<DurableEffectTransition, ActuationError> {
        self.inner
            .proven_not_dispatched(account, grant)
            .map_err(Into::into)
    }
}

/// Failure in the opaque policy-bound actuation composition layer.
#[derive(Debug, Error)]
pub enum ActuationError {
    /// Lower durable action/checkpoint layer failed.
    #[error("durable IoT runtime failed: {0}")]
    Durable(#[from] DurableIoTError),
    #[error("armed permit and retained combined checkpoint diverged")]
    ArmedCheckpointMismatch,
    #[error("armed permit no longer binds the exact validated proposal")]
    ProposalBindingMismatch,
    #[error("action account diverged from the armed durable checkpoint")]
    AccountCheckpointDiverged,
    #[error("armed checkpoint does not bind the exact device command sequence")]
    DeviceSequenceBindingMismatch,
    #[error("armed checkpoint is missing this permit's reservation")]
    MissingPermitReservation,
    #[error("cannot uniquely identify this permit's reservation")]
    AmbiguousPermitReservation,
    #[error("permit reservation is not in OutcomeUnknown state")]
    PermitReservationNotOutcomeUnknown,
    #[error("permit reservation risk charge does not match validated proposal")]
    PermitRiskBindingMismatch,
    #[error("reserved-use accounting underflow during pre-send revalidation")]
    ReservedUseUnderflow,
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
        SafetyEnvelope,
    };
    use symthaea_iot_execution::safety_world_digest;
    use symthaea_iot_policy::{
        ACTUATION_POLICY_SCHEMA_VERSION, ACTUATION_POLICY_SNAPSHOT_SCHEMA_VERSION,
        ActuationPolicySnapshotV1, ActuationPolicyStatus, ActuationPolicyV1,
    };

    const POLICY_ID: &str = "policy-valve-open";

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

    fn policy_record(safety: SafetyEnvelope, revision: u64) -> ActuationPolicyV1 {
        ActuationPolicyV1 {
            schema_version: ACTUATION_POLICY_SCHEMA_VERSION,
            policy_id: POLICY_ID.into(),
            revision,
            status: ActuationPolicyStatus::Active,
            device: safety.device.clone(),
            operation: safety.operation.clone(),
            safety,
            risk_charge: risk(1),
            max_command_lifetime_s: 30,
            not_before_unix_s: 1_000,
            not_after_unix_s: Some(9_000),
        }
    }

    fn registry(safety: SafetyEnvelope) -> ActuationPolicyRegistry {
        ActuationPolicyRegistry::genesis(ActuationPolicySnapshotV1 {
            schema_version: ACTUATION_POLICY_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_s: 1_000,
            expires_at_unix_s: 9_000,
            previous_snapshot_digest: None,
            policies: vec![policy_record(safety, 1)],
        })
        .unwrap()
    }

    fn context(grant: &CapabilityGrant, account: &GrantAccount, now: u64) -> AuthorityContext {
        AuthorityContext {
            now_unix_s: now,
            current_epoch: grant.authority_epoch,
            use_state: account.authority_use_state(),
        }
    }

    fn bound_fixture(
        max_uses: u32,
    ) -> (CapabilityGrant, DeviceRuntimeState, ActuationPolicyRegistry) {
        let runtime = runtime();
        let safety = safety();
        let proposal = proposal(&runtime, &safety);
        let mut grant = grant(max_uses);
        grant.plan_digest = proposal.plan_digest;
        grant.world_digest = Some(proposal.world_digest);
        (grant, runtime, registry(safety))
    }

    fn validated(
        grant: &CapabilityGrant,
        account: &GrantAccount,
        runtime: &DeviceRuntimeState,
        registry: &ActuationPolicyRegistry,
    ) -> ValidatedActuation {
        let policy = registry.policy(POLICY_ID, 5_000).unwrap();
        validate_actuation(
            grant,
            context(grant, account, 5_000),
            &[],
            proposal(runtime, policy.safety()),
            runtime,
            &policy,
        )
        .expect("valid policy-bound actuation")
    }

    fn armed(
        grant: &CapabilityGrant,
        account: &mut GrantAccount,
        runtime: &DeviceRuntimeState,
        registry: &ActuationPolicyRegistry,
    ) -> ArmedActuationPermit {
        let validated = validated(grant, account, runtime, registry);
        let pending = prepare_actuation_dispatch(grant, account, validated, None, None).unwrap();
        let head = pending.expected_head();
        pending.confirm_persisted(head).expect("exact durable head")
    }

    #[test]
    fn policy_handle_is_required_and_binds_exact_risk() {
        let (grant, runtime, registry) = bound_fixture(1);
        let account = GrantAccount::new(&grant);
        let policy = registry.policy(POLICY_ID, 5_000).unwrap();
        let mut undercharged = proposal(&runtime, policy.safety());
        undercharged.risk_charge = RiskBudget::default();
        assert!(matches!(
            validate_actuation(
                &grant,
                context(&grant, &account, 5_000),
                &[],
                undercharged,
                &runtime,
                &policy,
            ),
            Err(ActuationValidationError::Policy(
                ActuationPolicyBindingDenyReason::RiskChargeMismatch
            ))
        ));
    }

    #[test]
    fn policy_handle_cannot_be_replayed_at_another_evaluation_time() {
        let (grant, runtime, registry) = bound_fixture(1);
        let account = GrantAccount::new(&grant);
        let policy = registry.policy(POLICY_ID, 5_000).unwrap();
        assert!(matches!(
            validate_actuation(
                &grant,
                context(&grant, &account, 5_001),
                &[],
                proposal(&runtime, policy.safety()),
                &runtime,
                &policy,
            ),
            Err(ActuationValidationError::PolicySelectionTimeMismatch { .. })
        ));
    }

    #[test]
    fn one_use_grant_does_not_deny_its_own_reserved_preflight() {
        let (grant, runtime, registry) = bound_fixture(1);
        let mut account = GrantAccount::new(&grant);
        let permit = armed(&grant, &mut account, &runtime, &registry);
        assert_eq!(account.authority_use_state().reserved, 1);

        let outcome = permit
            .revalidate_before_send(
                &grant,
                &mut account,
                5_001,
                grant.authority_epoch,
                &[],
                &runtime,
                &registry,
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
        let (grant, runtime, registry) = bound_fixture(1);
        let mut account = GrantAccount::new(&grant);
        let permit = armed(&grant, &mut account, &runtime, &registry);

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
                &registry,
            )
            .unwrap();
        let PreflightOutcome::Rejected(rejected) = outcome else {
            panic!("expected fresh-world rejection");
        };
        assert_eq!(
            rejected.reason,
            PreflightRejectionReason::Execution(BoundExecutionDenyReason::RuntimeWorldMismatch)
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
    fn policy_registry_generation_change_blocks_send_and_releases_capacity() {
        let (grant, runtime, registry) = bound_fixture(1);
        let mut account = GrantAccount::new(&grant);
        let permit = armed(&grant, &mut account, &runtime, &registry);

        let next = ActuationPolicySnapshotV1 {
            schema_version: ACTUATION_POLICY_SNAPSHOT_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_s: 5_001,
            expires_at_unix_s: 9_100,
            previous_snapshot_digest: Some(registry.head().digest),
            policies: vec![policy_record(safety(), 2)],
        };
        let new_registry = registry.successor(next).unwrap();
        let outcome = permit
            .revalidate_before_send(
                &grant,
                &mut account,
                5_001,
                grant.authority_epoch,
                &[],
                &runtime,
                &new_registry,
            )
            .unwrap();
        let PreflightOutcome::Rejected(rejected) = outcome else {
            panic!("expected policy-generation rejection");
        };
        assert!(matches!(
            rejected.reason,
            PreflightRejectionReason::Policy(
                PolicyPreflightDenyReason::RegistryGenerationChanged { .. }
            )
        ));
        assert_eq!(account.authority_use_state(), GrantUseState::default());
    }

    #[test]
    fn revocation_after_persistence_blocks_send_and_releases_reservation() {
        let (grant, runtime, registry) = bound_fixture(1);
        let mut account = GrantAccount::new(&grant);
        let permit = armed(&grant, &mut account, &runtime, &registry);
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
                &registry,
            )
            .unwrap();
        assert!(matches!(
            outcome,
            PreflightOutcome::Rejected(rejected)
                if matches!(rejected.reason, PreflightRejectionReason::Execution(_))
        ));
        assert_eq!(account.authority_use_state(), GrantUseState::default());
    }

    #[test]
    fn already_accepted_sequence_stays_charged_as_ambiguous() {
        let (grant, runtime, registry) = bound_fixture(1);
        let mut account = GrantAccount::new(&grant);
        let permit = armed(&grant, &mut account, &runtime, &registry);
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
                &registry,
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
    fn ready_permit_preserves_policy_durable_head_and_proposal() {
        let (grant, runtime, registry) = bound_fixture(2);
        let mut account = GrantAccount::new(&grant);
        let permit = armed(&grant, &mut account, &runtime, &registry);
        let head = permit.armed_head();
        let proposal_digest = permit.proposal_digest();
        let policy_digest = permit.policy_digest();
        let policy_head = permit.policy_registry_head();
        let outcome = permit
            .revalidate_before_send(
                &grant,
                &mut account,
                5_001,
                grant.authority_epoch,
                &[],
                &runtime,
                &registry,
            )
            .unwrap();
        let PreflightOutcome::Ready(ready) = outcome else {
            panic!("expected ready permit");
        };
        assert_eq!(ready.armed_head(), head);
        assert_eq!(ready.proposal_digest(), proposal_digest);
        assert_eq!(ready.policy_digest(), policy_digest);
        assert_eq!(ready.policy_registry_head(), policy_head);
    }
}
