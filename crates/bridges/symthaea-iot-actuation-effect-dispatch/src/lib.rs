// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-shot privileged physical-effect dispatch from a globally linearized actuation attempt.
//!
//! The production boundary distinguishes two persistence contracts:
//!
//! - [`DurablePhysicalEffectAttemptJournal`] is a **local crash-durable** journal interface;
//! - [`RollbackProtectedPhysicalEffectAttemptJournal`] additionally proves that each local journal
//!   transition reached an independently retained anti-rollback anchor.
//!
//! Only the rollback-protected interface is accepted by production dispatch. This prevents a crash
//! plus adversarial storage rollback from erasing a `Prepared` generation after the hardware
//! boundary may have been entered. The dispatcher still accepts no raw receipts, trust registries,
//! reusable final permits or JIT leases. Its sole authority-bearing input is a
//! `CurrentActuationAttempt` consumed by value.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_authority::{Digest32, Operation, PrincipalId, ResourceRef};
use symthaea_iot_actuation_linearization::{
    ActuationLinearizationError, CurrentActuationAttempt,
};
use symthaea_iot_authority::DeviceCommand;
use thiserror::Error;

#[cfg(feature = "qualification-legacy-unjournaled")]
mod qualification_legacy;
#[cfg(feature = "qualification-legacy-unjournaled")]
pub use qualification_legacy::{PhysicalEffectDispatchError, dispatch_current_attempt};

pub const MAX_PRIVILEGED_ADAPTER_ID_BYTES: usize = 128;

/// Authority-free commitment to one exact physical-attempt journal generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalEffectAttemptJournalHead {
    generation: u64,
    digest: Digest32,
}

impl PhysicalEffectAttemptJournalHead {
    pub fn new(generation: u64, digest: Digest32) -> Result<Self, AttemptJournalHeadError> {
        if digest == Digest32([0; 32]) {
            return Err(AttemptJournalHeadError);
        }
        Ok(Self { generation, digest })
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn digest(self) -> Digest32 {
        self.digest
    }
}

/// Opaque proof that an exact attempt correlation reached a journal's `Prepared` state.
pub trait DurablePreparedPhysicalEffectAttempt {
    fn journal_head(&self) -> PhysicalEffectAttemptJournalHead;
}

/// Local crash-durable attempt journal. Implementing this trait alone does not satisfy the
/// adversarial-rollback requirement for physical I/O.
pub trait DurablePhysicalEffectAttemptJournal {
    type Error: StdError + Send + Sync + 'static;
    type Prepared: DurablePreparedPhysicalEffectAttempt;

    fn persist_prepared(
        &mut self,
        correlation: &PhysicalEffectAttemptCorrelation,
    ) -> Result<Self::Prepared, Self::Error>;

    fn persist_abandoned_before_port(
        &mut self,
        prepared: &Self::Prepared,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error>;

    fn persist_adapter_acknowledged(
        &mut self,
        prepared: &Self::Prepared,
        adapter_evidence_digest: Digest32,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error>;

    fn persist_adapter_indeterminate(
        &mut self,
        prepared: &Self::Prepared,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error>;
}

/// Crash-durable attempt journal whose returned heads are also independently retained against
/// rollback. Production physical dispatch requires this stronger interface.
pub trait RollbackProtectedPhysicalEffectAttemptJournal {
    type Error: StdError + Send + Sync + 'static;
    type Prepared: DurablePreparedPhysicalEffectAttempt;

    /// Persist exact `Prepared` state locally, independently advance the anti-rollback anchor to
    /// that exact head, verify the anchor confirmation, and only then return the proof.
    fn persist_prepared_anchored(
        &mut self,
        correlation: &PhysicalEffectAttemptCorrelation,
    ) -> Result<Self::Prepared, Self::Error>;

    /// Persist and independently anchor a proven pre-port abandonment.
    fn persist_abandoned_before_port_anchored(
        &mut self,
        prepared: &Self::Prepared,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error>;

    /// Persist and independently anchor the adapter acknowledgement state.
    fn persist_adapter_acknowledged_anchored(
        &mut self,
        prepared: &Self::Prepared,
        adapter_evidence_digest: Digest32,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error>;

    /// Persist and independently anchor an indeterminate adapter return.
    fn persist_adapter_indeterminate_anchored(
        &mut self,
        prepared: &Self::Prepared,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error>;
}

#[derive(Debug)]
pub struct AuthorizedPhysicalEffectRequest<'a> {
    command: &'a DeviceCommand,
    command_digest: Digest32,
    envelope_digest: Digest32,
    composition_digest: Digest32,
}

impl<'a> AuthorizedPhysicalEffectRequest<'a> {
    pub const fn command(&self) -> &'a DeviceCommand {
        self.command
    }

    pub const fn command_digest(&self) -> Digest32 {
        self.command_digest
    }

    pub const fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    pub const fn composition_digest(&self) -> Digest32 {
        self.composition_digest
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct AdapterAttemptAcknowledgement {
    evidence_digest: Digest32,
}

impl AdapterAttemptAcknowledgement {
    pub fn new(evidence_digest: Digest32) -> Result<Self, AdapterAcknowledgementError> {
        if evidence_digest == Digest32([0; 32]) {
            return Err(AdapterAcknowledgementError::ZeroEvidenceDigest);
        }
        Ok(Self { evidence_digest })
    }

    pub const fn evidence_digest(&self) -> Digest32 {
        self.evidence_digest
    }
}

pub trait PrivilegedPhysicalEffectPort {
    type Error: StdError + Send + Sync + 'static;

    fn adapter_id(&self) -> &str;
    fn device(&self) -> &ResourceRef;
    fn operation(&self) -> &Operation;
    fn executor(&self) -> &PrincipalId;

    fn attempt_effect(
        &mut self,
        request: AuthorizedPhysicalEffectRequest<'_>,
    ) -> Result<AdapterAttemptAcknowledgement, Self::Error>;
}

#[derive(Debug, PartialEq, Eq)]
pub struct PhysicalEffectAttemptCorrelation {
    command_digest: Digest32,
    envelope_digest: Digest32,
    composition_digest: Digest32,
    device: ResourceRef,
    operation: Operation,
    executor: PrincipalId,
    sequence: u64,
    adapter_id: String,
    common_fenced_at_unix_ms: u64,
    wall_valid_until_unix_ms: u64,
}

impl PhysicalEffectAttemptCorrelation {
    pub const fn command_digest(&self) -> Digest32 {
        self.command_digest
    }

    pub const fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    pub const fn composition_digest(&self) -> Digest32 {
        self.composition_digest
    }

    pub fn device(&self) -> &ResourceRef {
        &self.device
    }

    pub fn operation(&self) -> &Operation {
        &self.operation
    }

    pub fn executor(&self) -> &PrincipalId {
        &self.executor
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn adapter_id(&self) -> &str {
        &self.adapter_id
    }

    pub const fn common_fenced_at_unix_ms(&self) -> u64 {
        self.common_fenced_at_unix_ms
    }

    pub const fn wall_valid_until_unix_ms(&self) -> u64 {
        self.wall_valid_until_unix_ms
    }
}

/// Adapter acknowledgement whose journal transition has also reached its independent anchor.
#[derive(Debug, PartialEq, Eq)]
pub struct PhysicalEffectAttemptRecord {
    correlation: PhysicalEffectAttemptCorrelation,
    adapter_evidence_digest: Digest32,
    journal_head: PhysicalEffectAttemptJournalHead,
}

impl PhysicalEffectAttemptRecord {
    pub fn correlation(&self) -> &PhysicalEffectAttemptCorrelation {
        &self.correlation
    }

    pub const fn adapter_evidence_digest(&self) -> Digest32 {
        self.adapter_evidence_digest
    }

    pub const fn journal_head(&self) -> PhysicalEffectAttemptJournalHead {
        self.journal_head
    }
}

/// Terminal production dispatch boundary.
///
/// `persist_prepared_anchored` must complete before the final time check. The privileged port call
/// is then immediately adjacent to that final check. Every post-call journal transition is likewise
/// required to be independently anchored before a stable result is returned.
pub fn dispatch_current_attempt_durable<P, J>(
    attempt: CurrentActuationAttempt<'_>,
    port: &mut P,
    journal: &mut J,
) -> Result<PhysicalEffectAttemptRecord, DurablePhysicalEffectDispatchError<P::Error, J::Error>>
where
    P: PrivilegedPhysicalEffectPort,
    J: RollbackProtectedPhysicalEffectAttemptJournal,
{
    let adapter_id = validate_adapter_id(port.adapter_id())
        .map_err(|_| DurablePhysicalEffectDispatchError::InvalidAdapterIdentity)?
        .to_owned();

    let command = attempt.command();
    if port.device() != &command.device {
        return Err(DurablePhysicalEffectDispatchError::PortDeviceMismatch);
    }
    if port.operation() != &command.operation {
        return Err(DurablePhysicalEffectDispatchError::PortOperationMismatch);
    }
    if port.executor() != &command.executor {
        return Err(DurablePhysicalEffectDispatchError::PortExecutorMismatch);
    }

    let command_digest = command.digest();
    let envelope_digest = attempt.envelope_digest();
    let composition_digest = attempt.composition_digest();
    let correlation = PhysicalEffectAttemptCorrelation {
        command_digest,
        envelope_digest,
        composition_digest,
        device: command.device.clone(),
        operation: command.operation.clone(),
        executor: command.executor.clone(),
        sequence: command.sequence,
        adapter_id,
        common_fenced_at_unix_ms: attempt.common_fenced_at_unix_ms(),
        wall_valid_until_unix_ms: attempt.wall_valid_until_unix_ms(),
    };
    let request = AuthorizedPhysicalEffectRequest {
        command,
        command_digest,
        envelope_digest,
        composition_digest,
    };

    let prepared = journal
        .persist_prepared_anchored(&correlation)
        .map_err(DurablePhysicalEffectDispatchError::ProtectedPreparation)?;
    let prepared_head = prepared.journal_head();

    // NORMATIVE LAST SOFTWARE CHECK. Keep the privileged call immediately adjacent.
    if let Err(source) = attempt.validate_dispatch_window_now() {
        return match journal.persist_abandoned_before_port_anchored(&prepared) {
            Ok(abandoned_head) => Err(
                DurablePhysicalEffectDispatchError::LinearizationAfterProtectedPreparation {
                    source,
                    abandoned_head,
                },
            ),
            Err(journal_source) => Err(
                DurablePhysicalEffectDispatchError::PrePortProtectionIndeterminate {
                    source,
                    prepared_head,
                    journal_source,
                },
            ),
        };
    }
    let adapter_result = port.attempt_effect(request);

    match adapter_result {
        Ok(acknowledgement) => {
            let adapter_evidence_digest = acknowledgement.evidence_digest();
            let journal_head = match journal
                .persist_adapter_acknowledged_anchored(&prepared, adapter_evidence_digest)
            {
                Ok(head) => head,
                Err(journal_source) => {
                    return Err(
                        DurablePhysicalEffectDispatchError::AdapterAcknowledgedButProtectionIndeterminate {
                            correlation,
                            prepared_head,
                            adapter_evidence_digest,
                            journal_source,
                        },
                    );
                }
            };
            Ok(PhysicalEffectAttemptRecord {
                correlation,
                adapter_evidence_digest,
                journal_head,
            })
        }
        Err(source) => match journal.persist_adapter_indeterminate_anchored(&prepared) {
            Ok(journal_head) => Err(DurablePhysicalEffectDispatchError::AdapterAttemptIndeterminate {
                correlation,
                journal_head,
                source,
            }),
            Err(journal_source) => Err(
                DurablePhysicalEffectDispatchError::AdapterAndProtectionIndeterminate {
                    correlation,
                    prepared_head,
                    adapter_source: source,
                    journal_source,
                },
            ),
        },
    }
}

fn validate_adapter_id(adapter_id: &str) -> Result<&str, AdapterIdentityError> {
    if adapter_id.is_empty()
        || adapter_id.len() > MAX_PRIVILEGED_ADAPTER_ID_BYTES
        || adapter_id.trim() != adapter_id
        || adapter_id.chars().any(char::is_control)
    {
        return Err(AdapterIdentityError);
    }
    Ok(adapter_id)
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("privileged adapter identity is empty, oversized, padded or contains control characters")]
struct AdapterIdentityError;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("physical-effect attempt journal head has a zero digest")]
pub struct AttemptJournalHeadError;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum AdapterAcknowledgementError {
    #[error("privileged adapter acknowledgement contains a zero evidence digest")]
    ZeroEvidenceDigest,
}

#[derive(Debug, Error)]
pub enum DurablePhysicalEffectDispatchError<E, J>
where
    E: StdError + Send + Sync + 'static,
    J: StdError + Send + Sync + 'static,
{
    #[error("privileged adapter identity is invalid")]
    InvalidAdapterIdentity,
    #[error("privileged effect port targets another device")]
    PortDeviceMismatch,
    #[error("privileged effect port implements another operation")]
    PortOperationMismatch,
    #[error("privileged effect port belongs to another executor")]
    PortExecutorMismatch,
    #[error("failed to persist and independently anchor physical-attempt preparation: {0}")]
    ProtectedPreparation(#[source] J),
    #[error("linearized attempt became invalid after rollback-protected preparation but before privileged dispatch: {source}")]
    LinearizationAfterProtectedPreparation {
        #[source]
        source: ActuationLinearizationError,
        abandoned_head: PhysicalEffectAttemptJournalHead,
    },
    #[error("linearized attempt failed before port invocation and the protected abandonment transition was not confirmed; anchored Prepared remains unresolved: linearization={source}; protection={journal_source}")]
    PrePortProtectionIndeterminate {
        source: ActuationLinearizationError,
        prepared_head: PhysicalEffectAttemptJournalHead,
        journal_source: J,
    },
    #[error("privileged adapter acknowledged the attempt but the rollback-protected acknowledgement transition was not confirmed; anchored Prepared remains unresolved: {journal_source}")]
    AdapterAcknowledgedButProtectionIndeterminate {
        correlation: PhysicalEffectAttemptCorrelation,
        prepared_head: PhysicalEffectAttemptJournalHead,
        adapter_evidence_digest: Digest32,
        journal_source: J,
    },
    #[error("privileged adapter returned an error after invocation; the indeterminate outcome is crash-durable and rollback-protected: {source}")]
    AdapterAttemptIndeterminate {
        correlation: PhysicalEffectAttemptCorrelation,
        journal_head: PhysicalEffectAttemptJournalHead,
        #[source]
        source: E,
    },
    #[error("privileged adapter errored and the rollback-protected indeterminate transition was not confirmed; anchored Prepared remains unresolved: adapter={adapter_source}; protection={journal_source}")]
    AdapterAndProtectionIndeterminate {
        correlation: PhysicalEffectAttemptCorrelation,
        prepared_head: PhysicalEffectAttemptJournalHead,
        adapter_source: E,
        journal_source: J,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_acknowledgement_rejects_zero_evidence() {
        assert_eq!(
            AdapterAttemptAcknowledgement::new(Digest32([0; 32])),
            Err(AdapterAcknowledgementError::ZeroEvidenceDigest)
        );
        assert_eq!(
            AdapterAttemptAcknowledgement::new(Digest32([0xA5; 32]))
                .unwrap()
                .evidence_digest(),
            Digest32([0xA5; 32])
        );
    }

    #[test]
    fn adapter_identity_is_bounded_and_canonical() {
        assert_eq!(validate_adapter_id("hal:valve-72").unwrap(), "hal:valve-72");
        assert!(validate_adapter_id("").is_err());
        assert!(validate_adapter_id(" padded").is_err());
        assert!(validate_adapter_id("line\nbreak").is_err());
        assert!(validate_adapter_id(&"x".repeat(MAX_PRIVILEGED_ADAPTER_ID_BYTES + 1)).is_err());
    }

    #[test]
    fn journal_heads_allow_genesis_but_reject_zero_digest() {
        assert!(PhysicalEffectAttemptJournalHead::new(0, Digest32([1; 32])).is_ok());
        assert!(PhysicalEffectAttemptJournalHead::new(1, Digest32([0; 32])).is_err());
        let head = PhysicalEffectAttemptJournalHead::new(7, Digest32([2; 32])).unwrap();
        assert_eq!(head.generation(), 7);
        assert_eq!(head.digest(), Digest32([2; 32]));
    }
}
