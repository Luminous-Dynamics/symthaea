// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only compatibility for the immediately previous #514 theorem.
//!
//! This module is compiled only with the non-default `qualification-legacy-unjournaled` feature.
//! It supplies an in-memory rollback-protected journal shim so the previous exact full-chain tests
//! can execute unchanged. Production callers must use `dispatch_current_attempt_durable` with a
//! real rollback-protected journal implementation.

use super::*;

impl PhysicalEffectAttemptCorrelation {
    /// Deterministic qualification-only constructor for downstream journal regression tests.
    #[doc(hidden)]
    pub fn qualification_fixture(device: ResourceRef, sequence: u64) -> Self {
        Self {
            command_digest: Digest32([0x11; 32]),
            envelope_digest: Digest32([0x22; 32]),
            composition_digest: Digest32([0x33; 32]),
            device,
            operation: Operation("qualification.effect".into()),
            executor: PrincipalId("qualification:executor".into()),
            sequence,
            adapter_id: "qualification:adapter".into(),
            common_fenced_at_unix_ms: 1_000,
            wall_valid_until_unix_ms: 2_000,
        }
    }
}

#[derive(Debug, Error)]
pub enum PhysicalEffectDispatchError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("linearized attempt expired or became temporally invalid before privileged dispatch: {0}")]
    Linearization(#[source] ActuationLinearizationError),
    #[error("privileged adapter identity is invalid")]
    InvalidAdapterIdentity,
    #[error("privileged effect port targets another device")]
    PortDeviceMismatch,
    #[error("privileged effect port implements another operation")]
    PortOperationMismatch,
    #[error("privileged effect port belongs to another executor")]
    PortExecutorMismatch,
    #[error("privileged adapter returned an error after the physical-attempt boundary was invoked: {source}")]
    AdapterAttemptIndeterminate {
        correlation: PhysicalEffectAttemptCorrelation,
        #[source]
        source: E,
    },
    #[error("qualification-only in-memory rollback-protection shim unexpectedly failed")]
    QualificationDurabilityFailure,
}

#[derive(Debug, Error)]
#[error("qualification-only in-memory rollback-protection shim failed")]
struct QualificationDurabilityError;

#[derive(Debug, Clone, Copy)]
struct QualificationPrepared {
    head: PhysicalEffectAttemptJournalHead,
}

impl DurablePreparedPhysicalEffectAttempt for QualificationPrepared {
    fn journal_head(&self) -> PhysicalEffectAttemptJournalHead {
        self.head
    }
}

struct QualificationProtectedJournal {
    head: PhysicalEffectAttemptJournalHead,
}

impl Default for QualificationProtectedJournal {
    fn default() -> Self {
        Self {
            head: PhysicalEffectAttemptJournalHead::new(0, Digest32([0x51; 32])).unwrap(),
        }
    }
}

impl QualificationProtectedJournal {
    fn advance(&mut self) -> Result<PhysicalEffectAttemptJournalHead, QualificationDurabilityError> {
        let generation = self
            .head
            .generation()
            .checked_add(1)
            .ok_or(QualificationDurabilityError)?;
        let mut bytes = [0xA5; 32];
        bytes[..8].copy_from_slice(&generation.to_be_bytes());
        let next = PhysicalEffectAttemptJournalHead::new(generation, Digest32(bytes))
            .map_err(|_| QualificationDurabilityError)?;
        self.head = next;
        Ok(next)
    }
}

impl RollbackProtectedPhysicalEffectAttemptJournal for QualificationProtectedJournal {
    type Error = QualificationDurabilityError;
    type Prepared = QualificationPrepared;

    fn persist_prepared_anchored(
        &mut self,
        _correlation: &PhysicalEffectAttemptCorrelation,
    ) -> Result<Self::Prepared, Self::Error> {
        let head = self.advance()?;
        Ok(QualificationPrepared { head })
    }

    fn persist_abandoned_before_port_anchored(
        &mut self,
        _prepared: &Self::Prepared,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error> {
        self.advance()
    }

    fn persist_adapter_acknowledged_anchored(
        &mut self,
        _prepared: &Self::Prepared,
        _adapter_evidence_digest: Digest32,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error> {
        self.advance()
    }

    fn persist_adapter_indeterminate_anchored(
        &mut self,
        _prepared: &Self::Prepared,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error> {
        self.advance()
    }
}

#[doc(hidden)]
pub fn dispatch_current_attempt<P>(
    attempt: CurrentActuationAttempt<'_>,
    port: &mut P,
) -> Result<PhysicalEffectAttemptRecord, PhysicalEffectDispatchError<P::Error>>
where
    P: PrivilegedPhysicalEffectPort,
{
    let mut journal = QualificationProtectedJournal::default();
    match dispatch_current_attempt_durable(attempt, port, &mut journal) {
        Ok(record) => Ok(record),
        Err(DurablePhysicalEffectDispatchError::InvalidAdapterIdentity) => {
            Err(PhysicalEffectDispatchError::InvalidAdapterIdentity)
        }
        Err(DurablePhysicalEffectDispatchError::PortDeviceMismatch) => {
            Err(PhysicalEffectDispatchError::PortDeviceMismatch)
        }
        Err(DurablePhysicalEffectDispatchError::PortOperationMismatch) => {
            Err(PhysicalEffectDispatchError::PortOperationMismatch)
        }
        Err(DurablePhysicalEffectDispatchError::PortExecutorMismatch) => {
            Err(PhysicalEffectDispatchError::PortExecutorMismatch)
        }
        Err(DurablePhysicalEffectDispatchError::LinearizationAfterProtectedPreparation {
            source,
            ..
        })
        | Err(DurablePhysicalEffectDispatchError::PrePortProtectionIndeterminate {
            source,
            ..
        }) => Err(PhysicalEffectDispatchError::Linearization(source)),
        Err(DurablePhysicalEffectDispatchError::AdapterAttemptIndeterminate {
            correlation,
            source,
            ..
        })
        | Err(DurablePhysicalEffectDispatchError::AdapterAndProtectionIndeterminate {
            correlation,
            adapter_source: source,
            ..
        }) => Err(PhysicalEffectDispatchError::AdapterAttemptIndeterminate {
            correlation,
            source,
        }),
        Err(DurablePhysicalEffectDispatchError::ProtectedPreparation(_))
        | Err(DurablePhysicalEffectDispatchError::AdapterAcknowledgedButProtectionIndeterminate {
            ..
        }) => Err(PhysicalEffectDispatchError::QualificationDurabilityFailure),
    }
}
