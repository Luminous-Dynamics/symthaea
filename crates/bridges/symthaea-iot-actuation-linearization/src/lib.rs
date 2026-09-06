// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Closure-scoped linearization of one exact composed IoT actuation lineage.
//!
//! This crate is the first boundary at which all independently mutable roots are held stable at
//! the same time. It deliberately performs no HAL/device I/O and mints no portable permit or lease.
//! Instead, it consumes one affine `ComposedActuationEvidence`, acquires every owner-local mutation
//! barrier in a fixed order, re-establishes all three live cryptographic fences, checks one common
//! wall-clock horizon, starts a short monotonic handoff ceiling, and exposes the resulting attempt
//! only inside a higher-ranked `FnOnce` scope. The attempt therefore cannot escape with references
//! to the consumed composition or retained currentness locks.

#![deny(unsafe_code)]

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_actuation_guard_admission_reservation::{
    CurrentAdmissionReservationFence, CurrentAdmissionReservationFenceError,
    DurableAdmissionReservationStore,
};
use symthaea_iot_actuation_guard_device_reality::{
    CurrentAdmissionDeviceRealityFence, CurrentAdmissionDeviceRealityGuard, DeviceRealityError,
};
use symthaea_iot_actuation_guard_interlock::{
    CurrentPostSemanticInterlockError, CurrentPostSemanticInterlockFence,
    CurrentPostSemanticInterlockGuard,
};
use symthaea_iot_actuation_guard_semantic_persistence::{
    CurrentSemanticHeadFence, CurrentSemanticHeadFenceError, DurableSemanticAcceptanceStore,
};
use symthaea_iot_actuation_trust_publication::{
    ActuationTrustPublicationError, CurrentActuationTrustFence,
    DurableActuationTrustPublicationStore,
};
use symthaea_iot_composed_actuation_evidence::ComposedActuationEvidence;
use symthaea_iot_transport_current_fence::{
    CurrentXeniaTransportFence, CurrentXeniaTransportFenceError, CurrentXeniaTransportFenceGuard,
};
use thiserror::Error;

/// Maximum monotonic time allowed between successful global convergence and the eventual HAL
/// boundary. A later HAL adapter must re-check the attempt immediately before the physical effect.
pub const MAX_LINEARIZED_HANDOFF_MS: u64 = 250;

/// Immutable references to the six owner-local currentness authorities needed for one attempt.
///
/// This object itself grants no authority. It is reusable only as a coordinator; every invocation
/// consumes a distinct `ComposedActuationEvidence` and obtains fresh held/current fences.
pub struct ActuationLinearizer<'a> {
    trust_store: &'a DurableActuationTrustPublicationStore,
    admission_store: &'a DurableAdmissionReservationStore,
    semantic_store: &'a DurableSemanticAcceptanceStore,
    transport_guard: &'a CurrentXeniaTransportFenceGuard,
    device_reality_guard: &'a CurrentAdmissionDeviceRealityGuard,
    interlock_guard: &'a CurrentPostSemanticInterlockGuard,
}

impl<'a> ActuationLinearizer<'a> {
    pub const fn new(
        trust_store: &'a DurableActuationTrustPublicationStore,
        admission_store: &'a DurableAdmissionReservationStore,
        semantic_store: &'a DurableSemanticAcceptanceStore,
        transport_guard: &'a CurrentXeniaTransportFenceGuard,
        device_reality_guard: &'a CurrentAdmissionDeviceRealityGuard,
        interlock_guard: &'a CurrentPostSemanticInterlockGuard,
    ) -> Self {
        Self {
            trust_store,
            admission_store,
            semantic_store,
            transport_guard,
            device_reality_guard,
            interlock_guard,
        }
    }

    /// Consume one exact composed evidence object and expose a globally current attempt only inside
    /// a non-escaping `FnOnce` scope.
    ///
    /// Lock/currentness acquisition order is normative:
    ///
    /// 1. atomic actuation-trust publication;
    /// 2. durable admission reservation;
    /// 3. durable semantic head;
    /// 4. current Xenia transport;
    /// 5. current device reality; and
    /// 6. current controller/interlock evidence.
    ///
    /// The common wall-clock read and monotonic ceiling occur only after every mutable root has
    /// converged. `R` is outside the higher-ranked attempt lifetime, preventing the returned value
    /// from borrowing the scoped attempt or its locks.
    pub fn with_current_attempt<R, F>(
        &self,
        composed: ComposedActuationEvidence,
        use_attempt: F,
    ) -> Result<R, ActuationLinearizationError>
    where
        F: for<'attempt> FnOnce(CurrentActuationAttempt<'attempt>) -> R,
    {
        // The publication fence is acquired first and remains live through the closure. Therefore
        // no new trust/policy bundle can become authoritative while the lower owner roots converge.
        let trust_fence = self.trust_store.fence_current()?;
        validate_publication_bindings(
            &composed,
            &trust_fence,
            self.transport_guard,
            self.device_reality_guard,
            self.interlock_guard,
        )?;

        let semantic = composed.semantic_acceptance();
        let admission = semantic.admission_reservation();

        // These two fences retain the exact kernel locks used by their corresponding mutation
        // paths. Exact checkpoint equality is checked in addition to head equality so the final
        // boundary never treats a cached head as a substitute for the retained durable object.
        let admission_fence = self.admission_store.fence_current(admission.head())?;
        if admission_fence.checkpoint() != admission.checkpoint() {
            return Err(ActuationLinearizationError::AdmissionCheckpointMismatch);
        }

        let semantic_fence = self.semantic_store.fence_current(semantic.device_head())?;
        if semantic_fence.checkpoint() != semantic.checkpoint() {
            return Err(ActuationLinearizationError::SemanticCheckpointMismatch);
        }

        // Cryptographic/current-key verification remains owner-local. The generic linearizer does
        // not learn or duplicate key selection, lifecycle, signature or interlock-policy logic.
        let transport_fence = self.transport_guard.fence_current(composed.transport())?;
        let device_reality_fence = self
            .device_reality_guard
            .fence_current(semantic.device_reality())?;
        let interlock_fence = self
            .interlock_guard
            .fence_current(composed.post_semantic_interlock())?;

        // One final wall-clock read establishes a common horizon after all sequential owner-local
        // currentness checks. Any clock regression behind one of those checks fails closed.
        let common_fenced_at_unix_ms = system_unix_ms()?;
        let effect_deadline_unix_ms = semantic
            .admission_reservation()
            .envelope()
            .send_not_after_unix_s
            .checked_mul(1_000)
            .ok_or(ActuationLinearizationError::TimeOverflow)?;
        let wall_valid_until_unix_ms = validate_common_wall_window(
            common_fenced_at_unix_ms,
            [
                transport_fence.fenced_at_unix_ms(),
                device_reality_fence.fenced_at_unix_ms(),
                interlock_fence.fenced_at_unix_ms(),
            ],
            [
                transport_fence.valid_until_unix_ms(),
                device_reality_fence.valid_until_unix_ms(),
                interlock_fence.valid_until_unix_ms(),
                effect_deadline_unix_ms,
            ],
        )?;

        let monotonic_started_at = Instant::now();
        let monotonic_deadline = monotonic_started_at
            .checked_add(Duration::from_millis(MAX_LINEARIZED_HANDOFF_MS))
            .ok_or(ActuationLinearizationError::MonotonicDeadlineOverflow)?;

        let attempt = CurrentActuationAttempt {
            composition: &composed,
            _trust_fence: trust_fence,
            _admission_fence: admission_fence,
            _semantic_fence: semantic_fence,
            _transport_fence: transport_fence,
            _device_reality_fence: device_reality_fence,
            _interlock_fence: interlock_fence,
            common_fenced_at_unix_ms,
            wall_valid_until_unix_ms,
            _monotonic_started_at: monotonic_started_at,
            monotonic_deadline,
        };

        Ok(use_attempt(attempt))
    }
}

/// Non-clone, non-serializable, non-escaping evidence that all mutable actuation roots converged
/// under retained mutation barriers and a common final time horizon.
///
/// This type performs no physical effect. A future HAL adapter should accept it by value and call
/// `validate_dispatch_window_now` immediately before the device-side effect operation.
#[derive(Debug)]
#[must_use = "dropping a current actuation attempt safely abandons the physical attempt"]
pub struct CurrentActuationAttempt<'a> {
    composition: &'a ComposedActuationEvidence,
    _trust_fence: CurrentActuationTrustFence<'a>,
    _admission_fence: CurrentAdmissionReservationFence<'a>,
    _semantic_fence: CurrentSemanticHeadFence<'a>,
    _transport_fence: CurrentXeniaTransportFence<'a>,
    _device_reality_fence: CurrentAdmissionDeviceRealityFence<'a>,
    _interlock_fence: CurrentPostSemanticInterlockFence<'a>,
    common_fenced_at_unix_ms: u64,
    wall_valid_until_unix_ms: u64,
    _monotonic_started_at: Instant,
    monotonic_deadline: Instant,
}

impl CurrentActuationAttempt<'_> {
    pub const fn composition_digest(&self) -> Digest32 {
        self.composition.composition_digest()
    }

    pub fn device(&self) -> &ResourceRef {
        &self
            .composition
            .semantic_acceptance()
            .admission_reservation()
            .envelope()
            .command
            .device
    }

    pub const fn common_fenced_at_unix_ms(&self) -> u64 {
        self.common_fenced_at_unix_ms
    }

    pub const fn wall_valid_until_unix_ms(&self) -> u64 {
        self.wall_valid_until_unix_ms
    }

    /// Re-check both independent time domains immediately before a later HAL effect.
    ///
    /// This method deliberately does not consume the attempt or perform I/O; a later HAL adapter
    /// must take the attempt by value, invoke this as its final pre-effect check, then execute at
    /// most one physical operation.
    pub fn validate_dispatch_window_now(&self) -> Result<(), ActuationLinearizationError> {
        let now_unix_ms = system_unix_ms()?;
        if now_unix_ms < self.common_fenced_at_unix_ms {
            return Err(ActuationLinearizationError::WallClockRegressed);
        }
        if now_unix_ms >= self.wall_valid_until_unix_ms {
            return Err(ActuationLinearizationError::CommonWallWindowElapsed);
        }
        if Instant::now() >= self.monotonic_deadline {
            return Err(ActuationLinearizationError::MonotonicHandoffElapsed);
        }
        Ok(())
    }
}

fn validate_publication_bindings(
    composed: &ComposedActuationEvidence,
    trust_fence: &CurrentActuationTrustFence<'_>,
    transport_guard: &CurrentXeniaTransportFenceGuard,
    device_reality_guard: &CurrentAdmissionDeviceRealityGuard,
    interlock_guard: &CurrentPostSemanticInterlockGuard,
) -> Result<(), ActuationLinearizationError> {
    let roots = trust_fence.roots();
    let semantic = composed.semantic_acceptance();
    let admission = semantic.admission_reservation();
    let reality = semantic.device_reality();
    let interlock = composed.post_semantic_interlock();
    let device = &admission.envelope().command.device;

    if roots.device != *device {
        return Err(ActuationLinearizationError::PublicationDeviceMismatch);
    }
    if roots.transport_trust_head != transport_guard.anchored_current_head()
        || roots.transport_trust_head != composed.transport().transport_trust_head()
    {
        return Err(ActuationLinearizationError::PublicationTransportTrustMismatch);
    }
    if roots.device_reality_trust_head != device_reality_guard.anchored_trust_head()
        || roots.device_reality_trust_head != reality.trust_head()
    {
        return Err(ActuationLinearizationError::PublicationDeviceRealityTrustMismatch);
    }
    if roots.device_reality_policy.digest != device_reality_guard.anchored_policy_digest()
        || roots.device_reality_policy.digest != reality.policy_digest()
    {
        return Err(ActuationLinearizationError::PublicationDeviceRealityPolicyMismatch);
    }
    if roots.interlock_trust_head != interlock_guard.anchored_trust_head()
        || roots.interlock_trust_head != interlock.interlock_trust_head()
    {
        return Err(ActuationLinearizationError::PublicationInterlockTrustMismatch);
    }
    if roots.interlock_policy.digest != interlock_guard.anchored_policy_digest()
        || roots.interlock_policy.digest != interlock.policy_digest()
    {
        return Err(ActuationLinearizationError::PublicationInterlockPolicyMismatch);
    }
    Ok(())
}

fn validate_common_wall_window(
    common_now_unix_ms: u64,
    fenced_at_unix_ms: [u64; 3],
    valid_until_unix_ms: [u64; 4],
) -> Result<u64, ActuationLinearizationError> {
    let latest_owner_fence = fenced_at_unix_ms
        .into_iter()
        .max()
        .ok_or(ActuationLinearizationError::InvalidCurrentnessSet)?;
    if common_now_unix_ms < latest_owner_fence {
        return Err(ActuationLinearizationError::WallClockRegressed);
    }
    let wall_valid_until_unix_ms = valid_until_unix_ms
        .into_iter()
        .min()
        .ok_or(ActuationLinearizationError::InvalidCurrentnessSet)?;
    if common_now_unix_ms >= wall_valid_until_unix_ms {
        return Err(ActuationLinearizationError::CommonWallWindowElapsed);
    }
    Ok(wall_valid_until_unix_ms)
}

fn system_unix_ms() -> Result<u64, ActuationLinearizationError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ActuationLinearizationError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| ActuationLinearizationError::TimeOverflow)
}

#[derive(Debug, Error)]
pub enum ActuationLinearizationError {
    #[error("current actuation-trust publication failed: {0}")]
    TrustPublication(#[from] ActuationTrustPublicationError),
    #[error("current durable admission reservation failed: {0}")]
    AdmissionCurrent(#[from] CurrentAdmissionReservationFenceError),
    #[error("current durable semantic head failed: {0}")]
    SemanticCurrent(#[from] CurrentSemanticHeadFenceError),
    #[error("current Xenia transport fence failed: {0}")]
    TransportCurrent(#[from] CurrentXeniaTransportFenceError),
    #[error("current device-reality fence failed: {0}")]
    DeviceRealityCurrent(#[from] DeviceRealityError),
    #[error("current controller/interlock fence failed: {0}")]
    InterlockCurrent(#[from] CurrentPostSemanticInterlockError),
    #[error("authoritative trust publication targets another device")]
    PublicationDeviceMismatch,
    #[error("published Xenia transport trust does not match the current guard and exact proof")]
    PublicationTransportTrustMismatch,
    #[error("published device-reality trust does not match the current guard and exact proof")]
    PublicationDeviceRealityTrustMismatch,
    #[error("published device-reality policy does not match the current guard and exact proof")]
    PublicationDeviceRealityPolicyMismatch,
    #[error("published controller trust does not match the current guard and exact proof")]
    PublicationInterlockTrustMismatch,
    #[error("published interlock policy does not match the current guard and exact proof")]
    PublicationInterlockPolicyMismatch,
    #[error("durable admission checkpoint differs from the exact checkpoint retained by composition")]
    AdmissionCheckpointMismatch,
    #[error("durable semantic checkpoint differs from the exact checkpoint retained by composition")]
    SemanticCheckpointMismatch,
    #[error("currentness set is unexpectedly empty")]
    InvalidCurrentnessSet,
    #[error("system wall clock regressed behind an owner-local currentness check")]
    WallClockRegressed,
    #[error("the common wall-clock actuation window has elapsed")]
    CommonWallWindowElapsed,
    #[error("the short monotonic handoff window has elapsed")]
    MonotonicHandoffElapsed,
    #[error("system wall clock predates Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("time conversion overflow")]
    TimeOverflow,
    #[error("monotonic handoff deadline overflow")]
    MonotonicDeadlineOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_window_uses_latest_fence_and_earliest_deadline() {
        assert_eq!(
            validate_common_wall_window(1_250, [1_000, 1_100, 1_200], [2_000, 1_700, 1_900, 1_800])
                .unwrap(),
            1_700
        );
    }

    #[test]
    fn common_window_rejects_clock_regression() {
        assert!(matches!(
            validate_common_wall_window(1_150, [1_000, 1_200, 1_100], [2_000; 4]),
            Err(ActuationLinearizationError::WallClockRegressed)
        ));
    }

    #[test]
    fn common_window_is_exclusive_at_earliest_deadline() {
        assert!(matches!(
            validate_common_wall_window(1_700, [1_000, 1_100, 1_200], [2_000, 1_700, 1_900, 1_800]),
            Err(ActuationLinearizationError::CommonWallWindowElapsed)
        ));
    }

    #[test]
    fn monotonic_handoff_ceiling_is_strictly_future() {
        let start = Instant::now();
        let deadline = start
            .checked_add(Duration::from_millis(MAX_LINEARIZED_HANDOFF_MS))
            .unwrap();
        assert!(deadline > start);
    }
}
