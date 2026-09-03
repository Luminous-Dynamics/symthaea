// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable cyber-physical checkpointing above `symthaea-iot-runtime`.
//!
//! Crash-conservative use/risk accounting is not sufficient on its own for a
//! physical device protocol: the monotonic command sequence consumed by the
//! device domain must advance in the same durable generation. Otherwise a host
//! can remember that "some effect may have happened" while forgetting which
//! device generation was burned.
//!
//! The ordering enforced here is:
//!
//! ```text
//! exact bound execution admission + exact proposal
//!   -> verify prior combined trusted head
//!   -> burn the device-domain command sequence
//!   -> reserve use/risk and mark OutcomeUnknown
//!   -> bind both states into one DurableIoTCheckpoint
//!   -> host persists checkpoint + retains DurableIoTHead externally
//!   -> only then mint a DurableDispatchPermit
//! ```
//!
//! Device sequences intentionally never roll back, including when later evidence
//! proves a command was not dispatched. Sequence numbers are cheap; reusing one
//! after crash/reconciliation creates ambiguity that is not worth recovering.
//!
//! This crate still does not perform storage, fsync, networking, Xenia
//! authentication, actuator I/O, hardware attestation, or proof of a physical
//! effect. A matching returned head is a type/order contract with the persistence
//! owner, not evidence that storage hardware actually made bytes durable.

#![deny(unsafe_code)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use symthaea_action_checkpoint::{CheckpointError, CheckpointHead, GrantAccountCheckpoint};
use symthaea_action_runtime::GrantAccount;
use symthaea_authority::{CapabilityGrant, Digest32, ResourceRef};
use symthaea_iot_execution::{BoundExecutionAdmission, PhysicalExecutionProposal};
use symthaea_iot_runtime::{
    DispatchPermit, EffectOutcome, EffectTransition, IoTRuntimeError, PendingDispatchPersistence,
    UnknownPhysicalEffect, prepare_dispatch,
};
use thiserror::Error;

/// Current schema generation for [`DurableIoTCheckpoint`].
pub const IOT_DURABLE_CHECKPOINT_SCHEMA_VERSION: u16 = 1;
/// Domain separator for combined IoT checkpoint commitments.
pub const IOT_DURABLE_CHECKPOINT_DOMAIN: &[u8] = b"symthaea-iot-durable-checkpoint-v1\0";

/// Externally retainable anti-rollback anchor for one combined IoT generation.
///
/// `action_head` is included explicitly so the outer checkpoint cannot be
/// detached from the exact inner use/risk generation it commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableIoTHead {
    /// Inner crash-conservative action-accounting head.
    pub action_head: CheckpointHead,
    /// Commitment to the complete combined checkpoint, including device sequences.
    pub digest: Digest32,
}

/// One atomic durable generation of action accounting plus device replay state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableIoTCheckpoint {
    /// Fail-closed schema version.
    pub schema_version: u16,
    /// Previous combined checkpoint digest, absent only at generation zero.
    pub previous_checkpoint_digest: Option<Digest32>,
    /// Inner action-accounting checkpoint.
    pub action_checkpoint: GrantAccountCheckpoint,
    /// Exact head of `action_checkpoint`, duplicated for infallible outer hashing.
    pub action_head: CheckpointHead,
    /// Highest device-domain command sequence ever burned for each resource.
    pub device_sequences: BTreeMap<ResourceRef, u64>,
}

impl DurableIoTCheckpoint {
    /// Construct generation zero from an already prepared action checkpoint.
    pub fn first(
        grant: &CapabilityGrant,
        action_checkpoint: GrantAccountCheckpoint,
        device_sequences: BTreeMap<ResourceRef, u64>,
    ) -> Result<Self, DurableIoTError> {
        action_checkpoint.verify_against_head(grant, None)?;
        let action_head = action_checkpoint.head()?;
        Ok(Self {
            schema_version: IOT_DURABLE_CHECKPOINT_SCHEMA_VERSION,
            previous_checkpoint_digest: None,
            action_checkpoint,
            action_head,
            device_sequences,
        })
    }

    /// Construct the exact successor of a verified combined checkpoint.
    pub fn successor(
        previous: &DurableIoTCheckpoint,
        grant: &CapabilityGrant,
        action_checkpoint: GrantAccountCheckpoint,
        device_sequences: BTreeMap<ResourceRef, u64>,
    ) -> Result<Self, DurableIoTError> {
        previous.verify_payload(grant)?;
        action_checkpoint.verify_against_head(grant, Some(previous.action_head))?;
        ensure_sequence_map_monotonic(&previous.device_sequences, &device_sequences)?;
        let action_head = action_checkpoint.head()?;
        Ok(Self {
            schema_version: IOT_DURABLE_CHECKPOINT_SCHEMA_VERSION,
            previous_checkpoint_digest: Some(previous.digest()),
            action_checkpoint,
            action_head,
            device_sequences,
        })
    }

    /// Infallible domain-separated commitment to the combined checkpoint.
    ///
    /// The inner `action_head.digest` already commits the complete serialized action
    /// checkpoint. The outer transcript therefore binds that head plus the complete
    /// canonical `BTreeMap` of device sequences and the predecessor commitment.
    pub fn digest(&self) -> Digest32 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(IOT_DURABLE_CHECKPOINT_DOMAIN);
        hasher.update(&self.schema_version.to_be_bytes());
        match self.previous_checkpoint_digest {
            Some(Digest32(bytes)) => {
                hasher.update(&[1]);
                hasher.update(&bytes);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hasher.update(&self.action_head.sequence.to_be_bytes());
        let Digest32(action_digest) = self.action_head.digest;
        hasher.update(&action_digest);
        hasher.update(&(self.device_sequences.len() as u64).to_be_bytes());
        for (resource, sequence) in &self.device_sequences {
            let bytes = resource.0.as_bytes();
            hasher.update(&(bytes.len() as u64).to_be_bytes());
            hasher.update(bytes);
            hasher.update(&sequence.to_be_bytes());
        }
        Digest32(*hasher.finalize().as_bytes())
    }

    /// Externally retainable head for this exact combined state.
    pub fn head(&self) -> DurableIoTHead {
        DurableIoTHead {
            action_head: self.action_head,
            digest: self.digest(),
        }
    }

    /// Verify the checkpoint's internal payload against the exact capability grant.
    pub fn verify_payload(&self, grant: &CapabilityGrant) -> Result<GrantAccount, DurableIoTError> {
        if self.schema_version != IOT_DURABLE_CHECKPOINT_SCHEMA_VERSION {
            return Err(DurableIoTError::UnsupportedSchema);
        }
        let account = self.action_checkpoint.verify_payload(grant)?;
        if self.action_checkpoint.head()? != self.action_head {
            return Err(DurableIoTError::ActionHeadMismatch);
        }
        Ok(account)
    }

    /// Verify this checkpoint against an externally authenticated latest head.
    pub fn verify_as_trusted_head(
        &self,
        grant: &CapabilityGrant,
        trusted: DurableIoTHead,
    ) -> Result<GrantAccount, DurableIoTError> {
        let account = self.verify_payload(grant)?;
        if self.head() != trusted {
            return Err(DurableIoTError::TrustedHeadMismatch);
        }
        Ok(account)
    }

    /// Highest sequence burned for `device`, if this lineage has seen it.
    pub fn device_sequence(&self, device: &ResourceRef) -> Option<u64> {
        self.device_sequences.get(device).copied()
    }

    /// Complete monotonic device sequence map.
    pub fn device_sequences(&self) -> &BTreeMap<ResourceRef, u64> {
        &self.device_sequences
    }
}

/// Durable state waiting for the complete combined checkpoint to be persisted.
///
/// Intentionally not `Clone`; the normal path consumes it into exactly one permit.
#[derive(Debug)]
pub struct PendingDurableDispatch {
    inner: PendingDispatchPersistence,
    checkpoint: DurableIoTCheckpoint,
    expected_head: DurableIoTHead,
}

impl PendingDurableDispatch {
    /// Exact combined checkpoint that must be durable before dispatch.
    pub fn checkpoint(&self) -> &DurableIoTCheckpoint {
        &self.checkpoint
    }

    /// Combined head the persistence owner must retain/authenticate externally.
    pub const fn expected_head(&self) -> DurableIoTHead {
        self.expected_head
    }

    /// Exact proposal commitment associated with this pending dispatch.
    pub fn proposal_digest(&self) -> Digest32 {
        self.inner.proposal_digest()
    }

    /// Consume pending state only for the exact combined head.
    ///
    /// A mismatch returns the original pending state unchanged and cannot mint a
    /// dispatch permit.
    pub fn confirm_persisted(
        self,
        durable_head: DurableIoTHead,
    ) -> Result<DurableDispatchPermit, PendingDurableDispatch> {
        if durable_head != self.expected_head {
            return Err(self);
        }

        let PendingDurableDispatch {
            inner,
            checkpoint,
            expected_head,
        } = self;
        match inner.confirm_persisted(checkpoint.action_head) {
            Ok(inner) => Ok(DurableDispatchPermit {
                inner,
                armed_checkpoint: checkpoint,
                armed_head: expected_head,
            }),
            Err(inner) => Err(PendingDurableDispatch {
                inner,
                checkpoint,
                expected_head,
            }),
        }
    }
}

/// Affine physical dispatch permit whose use/risk and device sequence are already
/// represented by one externally persisted combined checkpoint.
#[derive(Debug)]
pub struct DurableDispatchPermit {
    inner: DispatchPermit,
    armed_checkpoint: DurableIoTCheckpoint,
    armed_head: DurableIoTHead,
}

impl DurableDispatchPermit {
    /// Combined checkpoint head that armed this permit.
    pub const fn armed_head(&self) -> DurableIoTHead {
        self.armed_head
    }

    /// Exact proposal commitment this permit may attempt.
    pub fn proposal_digest(&self) -> Digest32 {
        self.inner.proposal_digest()
    }

    /// Preserve an ambiguous external result. No state is released: both risk/use
    /// and device sequence remain durably consumed.
    pub fn into_unknown(self) -> DurableUnknownPhysicalEffect {
        DurableUnknownPhysicalEffect {
            inner: self.inner.into_unknown(),
            checkpoint: self.armed_checkpoint,
            head: self.armed_head,
        }
    }

    /// Positive evidence shows the physical effect was applied.
    pub fn observed_applied(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<DurableEffectTransition, DurableIoTError> {
        let action = self.inner.observed_applied(account, grant)?;
        Ok(wrap_effect_transition(
            self.armed_checkpoint,
            action,
        ))
    }

    /// Independent evidence proves the effect was never dispatched.
    ///
    /// Use/risk may be released by the inner runtime, but the device sequence stays
    /// burned forever in this lineage.
    pub fn proven_not_dispatched(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<DurableEffectTransition, DurableIoTError> {
        let action = self.inner.proven_not_dispatched(account, grant)?;
        Ok(wrap_effect_transition(
            self.armed_checkpoint,
            action,
        ))
    }
}

/// Ambiguous physical effect retained across recovery with its exact sequence map.
#[derive(Debug)]
pub struct DurableUnknownPhysicalEffect {
    inner: UnknownPhysicalEffect,
    checkpoint: DurableIoTCheckpoint,
    head: DurableIoTHead,
}

impl DurableUnknownPhysicalEffect {
    /// Combined trusted head for the still-unknown effect.
    pub const fn head(&self) -> DurableIoTHead {
        self.head
    }

    /// Later evidence proves the effect was applied.
    pub fn reconcile_applied(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<DurableEffectTransition, DurableIoTError> {
        let action = self.inner.reconcile_applied(account, grant)?;
        Ok(wrap_effect_transition(self.checkpoint, action))
    }

    /// Later evidence proves the effect was not applied. Device sequence still does
    /// not roll back.
    pub fn reconcile_not_applied(
        self,
        account: &mut GrantAccount,
        grant: &CapabilityGrant,
    ) -> Result<DurableEffectTransition, DurableIoTError> {
        let action = self.inner.reconcile_not_applied(account, grant)?;
        Ok(wrap_effect_transition(self.checkpoint, action))
    }
}

/// Reconciled action transition plus the combined successor checkpoint.
#[derive(Debug)]
pub struct DurableEffectTransition {
    /// Applied vs independently proven not dispatched.
    pub outcome: EffectOutcome,
    /// Original inner action-accounting transition.
    pub action: EffectTransition,
    /// Combined successor retaining monotonic device sequences.
    pub checkpoint: DurableIoTCheckpoint,
    /// Head that must replace the prior external trusted anchor after persistence.
    pub head: DurableIoTHead,
}

/// Prepare one exact physical effect with atomic action-accounting and device-sequence
/// state.
pub fn prepare_durable_dispatch(
    grant: &CapabilityGrant,
    account: &mut GrantAccount,
    admission: &BoundExecutionAdmission,
    proposal: &PhysicalExecutionProposal,
    previous_checkpoint: Option<&DurableIoTCheckpoint>,
    trusted_previous_head: Option<DurableIoTHead>,
) -> Result<PendingDurableDispatch, DurableIoTError> {
    validate_admission_proposal_pair(admission, proposal)?;

    let mut device_sequences = match (previous_checkpoint, trusted_previous_head) {
        (None, None) => BTreeMap::new(),
        (Some(previous), Some(trusted)) => {
            let restored = previous.verify_as_trusted_head(grant, trusted)?;
            if restored.snapshot() != account.snapshot() {
                return Err(DurableIoTError::AccountCheckpointDiverged);
            }
            previous.device_sequences.clone()
        }
        _ => return Err(DurableIoTError::IncompleteCheckpointBase),
    };

    let device = proposal.command.device.clone();
    let proposed_sequence = proposal.command.sequence;
    if let Some(previous) = device_sequences.get(&device).copied() {
        if proposed_sequence <= previous {
            return Err(DurableIoTError::DeviceSequenceReplay {
                device,
                previous,
                proposed: proposed_sequence,
            });
        }
    }
    // Burn before the inner permit exists. This value is never rolled back by a
    // later NotDispatched reconciliation.
    device_sequences.insert(device, proposed_sequence);

    let (inner_previous, inner_head) = match previous_checkpoint {
        Some(previous) => (Some(&previous.action_checkpoint), Some(previous.action_head)),
        None => (None, None),
    };
    let inner = prepare_dispatch(
        grant,
        account,
        admission,
        inner_previous,
        inner_head,
    )?;

    // From here construction is infallible: the inner runtime already produced and
    // verified the action checkpoint/head. This avoids a failure path that could
    // mutate accounting to OutcomeUnknown and then lose the checkpoint object.
    let checkpoint = DurableIoTCheckpoint {
        schema_version: IOT_DURABLE_CHECKPOINT_SCHEMA_VERSION,
        previous_checkpoint_digest: previous_checkpoint.map(DurableIoTCheckpoint::digest),
        action_checkpoint: inner.checkpoint().clone(),
        action_head: inner.expected_head(),
        device_sequences,
    };
    let expected_head = checkpoint.head();

    Ok(PendingDurableDispatch {
        inner,
        checkpoint,
        expected_head,
    })
}

fn validate_admission_proposal_pair(
    admission: &BoundExecutionAdmission,
    proposal: &PhysicalExecutionProposal,
) -> Result<(), DurableIoTError> {
    if admission.proposal_digest != proposal.digest() {
        return Err(DurableIoTError::ProposalBindingMismatch);
    }
    if admission.cyber_physical.command_digest != proposal.command.digest() {
        return Err(DurableIoTError::AdmissionCommandMismatch);
    }
    if admission.cyber_physical.accepted_sequence != proposal.command.sequence {
        return Err(DurableIoTError::AdmissionSequenceMismatch);
    }
    Ok(())
}

fn wrap_effect_transition(
    previous: DurableIoTCheckpoint,
    action: EffectTransition,
) -> DurableEffectTransition {
    let checkpoint = DurableIoTCheckpoint {
        schema_version: IOT_DURABLE_CHECKPOINT_SCHEMA_VERSION,
        previous_checkpoint_digest: Some(previous.digest()),
        action_checkpoint: action.checkpoint.clone(),
        action_head: action.head,
        device_sequences: previous.device_sequences.clone(),
    };
    let head = checkpoint.head();
    DurableEffectTransition {
        outcome: action.outcome,
        action,
        checkpoint,
        head,
    }
}

fn ensure_sequence_map_monotonic(
    previous: &BTreeMap<ResourceRef, u64>,
    current: &BTreeMap<ResourceRef, u64>,
) -> Result<(), DurableIoTError> {
    for (device, previous_sequence) in previous {
        match current.get(device) {
            Some(current_sequence) if current_sequence >= previous_sequence => {}
            _ => {
                return Err(DurableIoTError::SequenceStateRollback {
                    device: device.clone(),
                    previous: *previous_sequence,
                    current: current.get(device).copied(),
                });
            }
        }
    }
    Ok(())
}

/// Failure at the combined durable cyber-physical boundary.
#[derive(Debug, Error)]
pub enum DurableIoTError {
    /// Unknown outer checkpoint schema.
    #[error("unsupported durable IoT checkpoint schema")]
    UnsupportedSchema,
    /// Inner action checkpoint and its committed head disagree.
    #[error("durable IoT checkpoint action head does not match its action checkpoint")]
    ActionHeadMismatch,
    /// Previous combined checkpoint and trusted head must be supplied together.
    #[error("durable IoT checkpoint and trusted head must be supplied together")]
    IncompleteCheckpointBase,
    /// Persisted checkpoint is not the externally retained combined head.
    #[error("durable IoT checkpoint does not match the externally trusted head")]
    TrustedHeadMismatch,
    /// In-memory action account is not the state committed by the trusted checkpoint.
    #[error("action account diverged from durable IoT checkpoint")]
    AccountCheckpointDiverged,
    /// Lower-layer admission was paired with a different execution proposal.
    #[error("bound execution admission does not commit the supplied proposal")]
    ProposalBindingMismatch,
    /// Lower-layer cyber-physical admission commits a different device command.
    #[error("cyber-physical admission does not commit the supplied device command")]
    AdmissionCommandMismatch,
    /// Admission sequence differs from the exact command sequence.
    #[error("cyber-physical admission sequence does not match the supplied command")]
    AdmissionSequenceMismatch,
    /// Proposed device sequence is not strictly newer than durable replay state.
    #[error("device sequence replay/rollback for {device:?}: proposed {proposed} <= durable {previous}")]
    DeviceSequenceReplay {
        /// Physical resource whose sequence was stale.
        device: ResourceRef,
        /// Highest durable sequence already burned.
        previous: u64,
        /// Proposed stale/replayed sequence.
        proposed: u64,
    },
    /// A successor checkpoint removed or lowered existing device replay state.
    #[error("durable device sequence state rolled back for {device:?}: prior {previous}, current {current:?}")]
    SequenceStateRollback {
        /// Physical resource whose replay state regressed.
        device: ResourceRef,
        /// Prior durable sequence.
        previous: u64,
        /// New sequence, or `None` if the key disappeared.
        current: Option<u64>,
    },
    /// Inner action-accounting/checkpoint runtime failed.
    #[error("IoT action runtime failed: {0}")]
    Runtime(#[from] IoTRuntimeError),
    /// Inner action checkpoint validation failed.
    #[error("action checkpoint validation failed: {0}")]
    Checkpoint(#[from] CheckpointError),
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use symthaea_action_runtime::{GrantAccount, ReservationState};
    use symthaea_authority::{
        AuthorityEpoch, Operation, PrincipalId, RiskBudget, TaskId,
    };
    use symthaea_iot_authority::{
        CyberPhysicalAdmission, DEVICE_COMMAND_SCHEMA_VERSION, DeviceCommand,
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
            "iot-durable-g1",
            PrincipalId("human:operator".into()),
            PrincipalId("agent:irrigation".into()),
            AuthorityEpoch(15),
        );
        grant.audience = Some(PrincipalId("gateway:field-a".into()));
        grant.task = Some(TaskId("irrigate:zone-7".into()));
        grant.resources = BTreeSet::from([ResourceRef("iot:valve:72".into())]);
        grant.operations = BTreeSet::from([Operation("valve.open".into())]);
        grant.expires_at_unix_s = Some(10_000);
        grant.max_uses = 4;
        grant.risk_budget = risk(4);
        grant
    }

    fn proposal(sequence: u64, duration_ms: i64) -> PhysicalExecutionProposal {
        PhysicalExecutionProposal {
            command: DeviceCommand {
                schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
                command_id: format!("cmd-{sequence}-{duration_ms}"),
                actor: PrincipalId("agent:irrigation".into()),
                executor: PrincipalId("gateway:field-a".into()),
                task: Some(TaskId("irrigate:zone-7".into())),
                device: ResourceRef("iot:valve:72".into()),
                operation: Operation("valve.open".into()),
                expected_firmware: digest(7),
                sequence,
                issued_at_unix_s: 4_990,
                expires_at_unix_s: 5_030,
                parameters: BTreeMap::from([("duration_ms".into(), duration_ms)]),
            },
            plan_digest: Some(digest(3)),
            world_digest: digest(4),
            risk_charge: risk(1),
        }
    }

    fn admission(grant: &CapabilityGrant, proposal: &PhysicalExecutionProposal) -> BoundExecutionAdmission {
        BoundExecutionAdmission {
            cyber_physical: CyberPhysicalAdmission {
                command_digest: proposal.command.digest(),
                grant_digest: grant.digest(),
                safety_envelope_digest: digest(5),
                accepted_sequence: proposal.command.sequence,
            },
            proposal_digest: proposal.digest(),
            runtime_world_digest: proposal.world_digest,
            risk_charge: proposal.risk_charge,
        }
    }

    #[test]
    fn sequence_is_persisted_before_dispatch_permit() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let proposal = proposal(7, 60_000);
        let admission = admission(&grant, &proposal);
        let pending = prepare_durable_dispatch(
            &grant,
            &mut account,
            &admission,
            &proposal,
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            pending.checkpoint().device_sequence(&proposal.command.device),
            Some(7)
        );
        let reservation = pending
            .checkpoint()
            .action_checkpoint
            .snapshot
            .reservations
            .values()
            .next()
            .unwrap();
        assert_eq!(reservation.state, ReservationState::OutcomeUnknown);
        assert_eq!(account.authority_use_state().reserved, 1);
    }

    #[test]
    fn wrong_combined_head_cannot_mint_permit() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let proposal = proposal(7, 60_000);
        let admission = admission(&grant, &proposal);
        let pending = prepare_durable_dispatch(
            &grant,
            &mut account,
            &admission,
            &proposal,
            None,
            None,
        )
        .unwrap();
        let expected = pending.expected_head();
        let wrong = DurableIoTHead {
            action_head: expected.action_head,
            digest: digest(99),
        };
        let pending = pending.confirm_persisted(wrong).unwrap_err();
        assert_eq!(pending.expected_head(), expected);
        assert_eq!(account.authority_use_state().reserved, 1);
    }

    #[test]
    fn sequence_stays_burned_when_proven_not_dispatched() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let proposal = proposal(7, 60_000);
        let admission = admission(&grant, &proposal);
        let pending = prepare_durable_dispatch(
            &grant,
            &mut account,
            &admission,
            &proposal,
            None,
            None,
        )
        .unwrap();
        let head = pending.expected_head();
        let permit = pending.confirm_persisted(head).unwrap();
        let transition = permit.proven_not_dispatched(&mut account, &grant).unwrap();

        assert_eq!(account.authority_use_state(), Default::default());
        assert_eq!(
            transition.checkpoint.device_sequence(&proposal.command.device),
            Some(7)
        );
        assert_eq!(transition.outcome, EffectOutcome::NotDispatched);
    }

    #[test]
    fn stale_sequence_is_rejected_even_after_inner_capacity_was_released() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let first = proposal(7, 60_000);
        let first_admission = admission(&grant, &first);
        let pending = prepare_durable_dispatch(
            &grant,
            &mut account,
            &first_admission,
            &first,
            None,
            None,
        )
        .unwrap();
        let permit = pending
            .confirm_persisted(pending.expected_head())
            .expect("exact head");
        let transition = permit.proven_not_dispatched(&mut account, &grant).unwrap();
        assert_eq!(account.authority_use_state(), Default::default());

        // Change another command field so the proposal digest/reservation id differs,
        // while deliberately reusing the already-burned device generation.
        let replay = proposal(7, 61_000);
        let replay_admission = admission(&grant, &replay);
        let error = prepare_durable_dispatch(
            &grant,
            &mut account,
            &replay_admission,
            &replay,
            Some(&transition.checkpoint),
            Some(transition.head),
        )
        .unwrap_err();
        assert!(matches!(error, DurableIoTError::DeviceSequenceReplay { .. }));
    }

    #[test]
    fn admission_cannot_be_paired_with_a_different_proposal() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let admitted = proposal(7, 60_000);
        let admission = admission(&grant, &admitted);
        let substituted = proposal(8, 60_000);
        let error = prepare_durable_dispatch(
            &grant,
            &mut account,
            &admission,
            &substituted,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(error, DurableIoTError::ProposalBindingMismatch));
        assert_eq!(account.authority_use_state(), Default::default());
    }

    #[test]
    fn higher_sequence_advances_durable_state() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let first = proposal(7, 60_000);
        let first_admission = admission(&grant, &first);
        let pending = prepare_durable_dispatch(
            &grant,
            &mut account,
            &first_admission,
            &first,
            None,
            None,
        )
        .unwrap();
        let head = pending.expected_head();
        let permit = pending.confirm_persisted(head).unwrap();
        let first_done = permit.proven_not_dispatched(&mut account, &grant).unwrap();

        let second = proposal(8, 60_000);
        let second_admission = admission(&grant, &second);
        let pending = prepare_durable_dispatch(
            &grant,
            &mut account,
            &second_admission,
            &second,
            Some(&first_done.checkpoint),
            Some(first_done.head),
        )
        .unwrap();
        assert_eq!(
            pending.checkpoint().device_sequence(&second.command.device),
            Some(8)
        );
    }

    #[test]
    fn wrong_external_trusted_head_rejects_continuation() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let first = proposal(7, 60_000);
        let first_admission = admission(&grant, &first);
        let pending = prepare_durable_dispatch(
            &grant,
            &mut account,
            &first_admission,
            &first,
            None,
            None,
        )
        .unwrap();
        let head = pending.expected_head();
        let permit = pending.confirm_persisted(head).unwrap();
        let first_done = permit.proven_not_dispatched(&mut account, &grant).unwrap();

        let second = proposal(8, 60_000);
        let second_admission = admission(&grant, &second);
        let wrong = DurableIoTHead {
            action_head: first_done.head.action_head,
            digest: digest(44),
        };
        let error = prepare_durable_dispatch(
            &grant,
            &mut account,
            &second_admission,
            &second,
            Some(&first_done.checkpoint),
            Some(wrong),
        )
        .unwrap_err();
        assert!(matches!(error, DurableIoTError::TrustedHeadMismatch));
    }

    #[test]
    fn checkpoint_digest_binds_device_sequence_map() {
        let grant = grant();
        let mut account = GrantAccount::new(&grant);
        let proposal = proposal(7, 60_000);
        let admission = admission(&grant, &proposal);
        let pending = prepare_durable_dispatch(
            &grant,
            &mut account,
            &admission,
            &proposal,
            None,
            None,
        )
        .unwrap();
        let mut changed = pending.checkpoint().clone();
        changed
            .device_sequences
            .insert(proposal.command.device.clone(), 8);
        assert_ne!(pending.checkpoint().digest(), changed.digest());
    }
}
