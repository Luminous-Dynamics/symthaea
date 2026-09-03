// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Just-in-time trust fencing for the final cyber-physical HAL handoff.
//!
//! `TrustBoundFinalActuatorPermit` proves that controller-key trust was current when
//! the permit was upgraded. Trust can still advance or revoke a key before the actual
//! hardware attempt. This crate closes that small TOCTOU window for a trusted guard
//! process by requiring the *current externally anchored* `InterlockTrustHead` again
//! immediately before HAL use.
//!
//! The resulting [`JustInTimeHalLease`] borrows the exact current registry object. In
//! ordinary Rust ownership this prevents that registry value from being replaced while
//! the lease exists. This is **not** a sandbox against malicious code in the same
//! process: real actuator authority still requires a minimal privileged guard process
//! that owns the authenticated trust anchor and exclusive HAL/device handle.

#![deny(unsafe_code)]

use symthaea_authority::Digest32;
use symthaea_iot_authority::DeviceCommand;
use symthaea_iot_interlock_trust::{
    InterlockControllerKeyStatus, InterlockTrustHead, InterlockTrustRegistry,
    TrustBoundFinalActuatorPermit,
};
use thiserror::Error;

/// Maximum time between the JIT trust fence and the actual HAL attempt.
///
/// The inherited physical-effect deadline may be shorter; the lease always uses the
/// earliest applicable deadline.
pub const MAX_JIT_FENCE_TO_HAL_MS: u64 = 250;

/// Borrowed final lease intended to be consumed immediately by a privileged HAL guard.
///
/// It is intentionally non-clone and non-serializable. Holding the borrow keeps the
/// exact registry object used by the fence alive for the lease lifetime.
#[derive(Debug)]
pub struct JustInTimeHalLease<'a> {
    permit: TrustBoundFinalActuatorPermit,
    _current_registry: &'a InterlockTrustRegistry,
    current_trust_head: InterlockTrustHead,
    fenced_at_unix_ms: u64,
    must_attempt_by_unix_ms: u64,
}

impl JustInTimeHalLease<'_> {
    /// Exact physical command whose authority lineage survived the JIT fence.
    pub fn command(&self) -> &DeviceCommand {
        self.permit.command()
    }

    /// Exact physical-effect envelope commitment.
    pub const fn envelope_digest(&self) -> Digest32 {
        self.permit.envelope_digest()
    }

    /// Current externally anchored interlock-trust generation at the fence.
    pub const fn current_trust_head(&self) -> InterlockTrustHead {
        self.current_trust_head
    }

    /// Local relying-party time at which the final trust fence ran.
    pub const fn fenced_at_unix_ms(&self) -> u64 {
        self.fenced_at_unix_ms
    }

    /// Inclusive latest millisecond at which a privileged guard may begin the HAL attempt.
    pub const fn must_attempt_by_unix_ms(&self) -> u64 {
        self.must_attempt_by_unix_ms
    }
}

/// Revalidate controller trust immediately before handing authority to a privileged HAL.
///
/// `externally_anchored_current_head` must come from the guard's independently
/// authenticated anti-rollback state, not from the permit or untrusted request. The
/// function deliberately invalidates *all* outstanding permits whenever the interlock
/// trust generation changes, even if the change concerned another controller. That is
/// conservative but makes revocation semantics simple and fail closed.
pub fn fence_for_hal<'a>(
    permit: TrustBoundFinalActuatorPermit,
    current_registry: &'a InterlockTrustRegistry,
    externally_anchored_current_head: InterlockTrustHead,
    now_unix_ms: u64,
) -> Result<JustInTimeHalLease<'a>, JitFenceError> {
    validate_head_fence(
        current_registry.head(),
        externally_anchored_current_head,
        permit.interlock_trust_head(),
    )?;

    let snapshot = current_registry.snapshot();
    if now_unix_ms < snapshot.issued_at_unix_ms || now_unix_ms >= snapshot.expires_at_unix_ms {
        return Err(JitFenceError::CurrentTrustSnapshotNotFresh);
    }
    if now_unix_ms < permit.trust_verified_at_unix_ms() {
        return Err(JitFenceError::FencePredatesPriorTrustVerification);
    }
    if now_unix_ms > permit.must_dispatch_by_unix_ms() {
        return Err(JitFenceError::PhysicalDispatchWindowElapsed);
    }

    let key = find_exact_current_key(
        current_registry,
        permit.interlock_key_id(),
        permit.interlock_key_digest(),
    )?;
    if key.status != InterlockControllerKeyStatus::Active
        || permit.trust_verified_at_unix_ms() < key.not_before_unix_ms
        || permit.trust_verified_at_unix_ms() >= key.not_after_unix_ms
        || now_unix_ms < key.not_before_unix_ms
        || now_unix_ms >= key.not_after_unix_ms
    {
        return Err(JitFenceError::ControllerKeyNotCurrentlyActive);
    }

    let jit_deadline = now_unix_ms
        .checked_add(MAX_JIT_FENCE_TO_HAL_MS)
        .ok_or(JitFenceError::TimeOverflow)?;
    let trust_deadline = snapshot
        .expires_at_unix_ms
        .checked_sub(1)
        .ok_or(JitFenceError::TimeOverflow)?;
    let key_deadline = key
        .not_after_unix_ms
        .checked_sub(1)
        .ok_or(JitFenceError::TimeOverflow)?;
    let must_attempt_by_unix_ms = permit
        .must_dispatch_by_unix_ms()
        .min(jit_deadline)
        .min(trust_deadline)
        .min(key_deadline);
    if must_attempt_by_unix_ms < now_unix_ms {
        return Err(JitFenceError::NoLiveHalAttemptWindow);
    }

    Ok(JustInTimeHalLease {
        permit,
        _current_registry: current_registry,
        current_trust_head: externally_anchored_current_head,
        fenced_at_unix_ms: now_unix_ms,
        must_attempt_by_unix_ms,
    })
}

fn validate_head_fence(
    registry_head: InterlockTrustHead,
    externally_anchored_current_head: InterlockTrustHead,
    permit_head: InterlockTrustHead,
) -> Result<(), JitFenceError> {
    if registry_head != externally_anchored_current_head {
        return Err(JitFenceError::ExternallyAnchoredHeadMismatch);
    }
    if permit_head != externally_anchored_current_head {
        return Err(JitFenceError::TrustAdvancedSincePermit);
    }
    Ok(())
}

fn find_exact_current_key<'a>(
    registry: &'a InterlockTrustRegistry,
    key_id: &str,
    expected_digest: Digest32,
) -> Result<&'a symthaea_iot_interlock_trust::InterlockControllerKeyV1, JitFenceError> {
    let mut matches = registry
        .snapshot()
        .keys
        .iter()
        .filter(|key| key.key_id == key_id)
        .filter_map(|key| match key.digest() {
            Ok(digest) if digest == expected_digest => Some(key),
            _ => None,
        });
    let key = matches.next().ok_or(JitFenceError::CurrentKeyRecordMismatch)?;
    if matches.next().is_some() {
        return Err(JitFenceError::AmbiguousCurrentKeyRecord);
    }
    Ok(key)
}

/// Fail-closed errors for the last trust check before HAL handoff.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum JitFenceError {
    /// The in-memory registry does not match the independently retained current head.
    #[error("current interlock registry does not match externally anchored trust head")]
    ExternallyAnchoredHeadMismatch,
    /// Interlock trust advanced after the permit was verified.
    #[error("interlock trust generation advanced after final permit verification")]
    TrustAdvancedSincePermit,
    /// Current trust snapshot is outside its validity interval.
    #[error("current interlock trust snapshot is not fresh")]
    CurrentTrustSnapshotNotFresh,
    /// Local time regressed behind the earlier key-verification time.
    #[error("HAL trust fence predates prior controller-key verification")]
    FencePredatesPriorTrustVerification,
    /// Physical-effect dispatch deadline already elapsed.
    #[error("physical dispatch window elapsed before HAL trust fence")]
    PhysicalDispatchWindowElapsed,
    /// Current registry no longer contains the exact trusted key record.
    #[error("current interlock controller key record no longer matches the permit")]
    CurrentKeyRecordMismatch,
    /// More than one current key record matched the supposedly exact key identity/digest.
    #[error("current interlock controller key record is ambiguous")]
    AmbiguousCurrentKeyRecord,
    /// Exact key is no longer active at both required times.
    #[error("interlock controller key is not active at HAL handoff")]
    ControllerKeyNotCurrentlyActive,
    /// Millisecond arithmetic overflowed.
    #[error("HAL trust-fence time arithmetic overflow")]
    TimeOverflow,
    /// All inherited/current deadlines collapsed before the HAL attempt could begin.
    #[error("no live HAL attempt window remains after current trust fence")]
    NoLiveHalAttemptWindow,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_iot_interlock_trust::{
        INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION, InterlockControllerKeyStatus,
        InterlockControllerKeyV1, InterlockTrustSnapshotV1,
    };

    fn key(status: InterlockControllerKeyStatus) -> InterlockControllerKeyV1 {
        InterlockControllerKeyV1 {
            controller_id: "safety-plc:field-a".into(),
            key_id: "plc-key-1".into(),
            algorithm: "vendor-signature-v1".into(),
            public_key: vec![0x42; 64],
            status,
            not_before_unix_ms: 100_000,
            not_after_unix_ms: 130_000,
        }
    }

    fn registry() -> InterlockTrustRegistry {
        InterlockTrustRegistry::genesis(InterlockTrustSnapshotV1 {
            schema_version: INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms: 100_000,
            expires_at_unix_ms: 130_000,
            previous_snapshot_digest: None,
            keys: vec![key(InterlockControllerKeyStatus::Active)],
        })
        .unwrap()
    }

    #[test]
    fn unchanged_current_head_passes_head_fence() {
        let registry = registry();
        let head = registry.head();
        assert_eq!(validate_head_fence(head, head, head), Ok(()));
    }

    #[test]
    fn any_trust_generation_change_invalidates_old_permit_head() {
        let registry = registry();
        let old = registry.head();
        let next_snapshot = InterlockTrustSnapshotV1 {
            schema_version: INTERLOCK_TRUST_SNAPSHOT_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_ms: 101_000,
            expires_at_unix_ms: 129_000,
            previous_snapshot_digest: Some(old.digest),
            keys: vec![InterlockControllerKeyV1 {
                not_after_unix_ms: 129_000,
                ..key(InterlockControllerKeyStatus::Active)
            }],
        };
        let current = registry.successor(next_snapshot).unwrap();
        assert_eq!(
            validate_head_fence(current.head(), current.head(), old),
            Err(JitFenceError::TrustAdvancedSincePermit)
        );
    }

    #[test]
    fn externally_anchored_head_must_match_registry() {
        let registry = registry();
        let fake = InterlockTrustHead {
            sequence: registry.head().sequence,
            digest: Digest32([0xAA; 32]),
        };
        assert_eq!(
            validate_head_fence(registry.head(), fake, registry.head()),
            Err(JitFenceError::ExternallyAnchoredHeadMismatch)
        );
    }

    #[test]
    fn exact_key_lookup_is_digest_bound() {
        let registry = registry();
        let expected = registry.snapshot().keys[0].digest().unwrap();
        assert_eq!(
            find_exact_current_key(&registry, "plc-key-1", expected)
                .unwrap()
                .controller_id,
            "safety-plc:field-a"
        );
        assert!(matches!(
            find_exact_current_key(&registry, "plc-key-1", Digest32([0x11; 32])),
            Err(JitFenceError::CurrentKeyRecordMismatch)
        ));
    }
}
