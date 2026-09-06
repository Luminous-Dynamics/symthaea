// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-shot privileged physical-effect dispatch from a globally linearized actuation attempt.
//!
//! This crate is intentionally narrower than the historical final-permit/JIT-lease path. It does
//! not parse transport receipts, select trust keys, verify signatures, mint a reusable permit or
//! create a HAL lease. Those currentness questions are already answered by
//! `CurrentActuationAttempt`, which retains every owner-local current fence and mutation barrier.
//!
//! The only authority-bearing input accepted here is a `CurrentActuationAttempt` **by value**. The
//! dispatcher binds the privileged port to the exact device, operation and executor already present
//! in that attempt, builds a request that borrows the exact bound command, performs the attempt's
//! wall+monotonic dispatch-window check as the final software operation, and immediately invokes one
//! privileged port method. Whether the port succeeds or returns an error, the attempt is consumed.
//!
//! A successful return proves only that the privileged adapter boundary returned an acknowledgement
//! for the exact request. It does **not** prove that the requested physical state transition was
//! realized. Likewise, once the port method has been invoked, an adapter error is classified as an
//! indeterminate physical outcome: callers must reconcile trusted device observations rather than
//! retry the same command.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_authority::{Digest32, Operation, PrincipalId, ResourceRef};
use symthaea_iot_actuation_linearization::{
    ActuationLinearizationError, CurrentActuationAttempt,
};
use symthaea_iot_authority::DeviceCommand;
use thiserror::Error;

/// Bound adapter identifier size. This is audit/correlation metadata, not authority.
pub const MAX_PRIVILEGED_ADAPTER_ID_BYTES: usize = 128;

/// Request available only after a caller has supplied a live `CurrentActuationAttempt`.
///
/// Fields are private and this type has no public constructor. A privileged adapter therefore
/// receives the exact command already bound by the linearized evidence rather than caller-selected
/// command data.
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

/// Minimal acknowledgement returned by a privileged effect adapter.
///
/// The digest should commit to adapter-local evidence such as a bus/controller acknowledgement,
/// transaction receipt or other device-class-specific attempt evidence. It is an adapter claim,
/// not proof that the physical world reached the requested state.
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

/// Narrow privileged effect boundary.
///
/// Implementations are trusted adapters owned by the minimal privileged guard process. The
/// identity methods must describe the exact physical sink reached by `attempt_effect`. The generic
/// dispatcher checks those identities against the already-linearized command before invoking the
/// effect method.
pub trait PrivilegedPhysicalEffectPort {
    type Error: StdError + Send + Sync + 'static;

    fn adapter_id(&self) -> &str;
    fn device(&self) -> &ResourceRef;
    fn operation(&self) -> &Operation;
    fn executor(&self) -> &PrincipalId;

    /// Attempt exactly one physical effect for the supplied linearized request.
    ///
    /// Returning `Err` does not establish that no physical effect occurred. The method may have
    /// crossed the external-effect boundary before the adapter detected or reported the error.
    fn attempt_effect(
        &mut self,
        request: AuthorizedPhysicalEffectRequest<'_>,
    ) -> Result<AdapterAttemptAcknowledgement, Self::Error>;
}

/// Owned audit correlation captured before the privileged port is invoked.
///
/// This record is intentionally non-serializable and confers no authority. On a port error it is
/// returned inside `AdapterAttemptIndeterminate` so recovery logic can reconcile the exact command
/// whose physical outcome became uncertain.
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

/// Evidence that the exact privileged adapter boundary acknowledged one consumed attempt.
///
/// This remains evidence of an **adapter attempt**, not a certificate of physical realization.
#[derive(Debug, PartialEq, Eq)]
pub struct PhysicalEffectAttemptRecord {
    correlation: PhysicalEffectAttemptCorrelation,
    adapter_evidence_digest: Digest32,
}

impl PhysicalEffectAttemptRecord {
    pub fn correlation(&self) -> &PhysicalEffectAttemptCorrelation {
        &self.correlation
    }

    pub const fn adapter_evidence_digest(&self) -> Digest32 {
        self.adapter_evidence_digest
    }
}

/// Consume one globally current attempt and invoke exactly one matching privileged effect port.
///
/// The function deliberately has no raw-receipt, registry, verifier, caller-selected time, permit
/// or lease inputs. Port binding and request construction happen before the final time check. The
/// final two operations are normative and intentionally adjacent:
///
/// 1. `attempt.validate_dispatch_window_now()`;
/// 2. `port.attempt_effect(request)`.
///
/// After step 2 has been invoked, any adapter error is epistemically indeterminate. The consumed
/// attempt cannot be retried; callers should obtain fresh trusted device observations and reconcile
/// the durable command state.
pub fn dispatch_current_attempt<P>(
    attempt: CurrentActuationAttempt<'_>,
    port: &mut P,
) -> Result<PhysicalEffectAttemptRecord, PhysicalEffectDispatchError<P::Error>>
where
    P: PrivilegedPhysicalEffectPort,
{
    let adapter_id = validate_adapter_id(port.adapter_id())
        .map_err(|_| PhysicalEffectDispatchError::InvalidAdapterIdentity)?
        .to_owned();

    let command = attempt.command();
    if port.device() != &command.device {
        return Err(PhysicalEffectDispatchError::PortDeviceMismatch);
    }
    if port.operation() != &command.operation {
        return Err(PhysicalEffectDispatchError::PortOperationMismatch);
    }
    if port.executor() != &command.executor {
        return Err(PhysicalEffectDispatchError::PortExecutorMismatch);
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

    // NORMATIVE LAST SOFTWARE CHECK. Keep the privileged call immediately adjacent.
    attempt
        .validate_dispatch_window_now()
        .map_err(PhysicalEffectDispatchError::Linearization)?;
    let acknowledgement = match port.attempt_effect(request) {
        Ok(acknowledgement) => acknowledgement,
        Err(source) => {
            return Err(PhysicalEffectDispatchError::AdapterAttemptIndeterminate {
                correlation,
                source,
            });
        }
    };

    Ok(PhysicalEffectAttemptRecord {
        correlation,
        adapter_evidence_digest: acknowledgement.evidence_digest(),
    })
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
pub enum AdapterAcknowledgementError {
    #[error("privileged adapter acknowledgement contains a zero evidence digest")]
    ZeroEvidenceDigest,
}

/// Fail-closed dispatch result.
///
/// Variants before `AdapterAttemptIndeterminate` prove the privileged port was not invoked through
/// this function. `AdapterAttemptIndeterminate` means the port **was invoked** and therefore the
/// physical result must be reconciled rather than blindly retried.
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
}
