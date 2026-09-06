// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fixed privileged verification of challenge-bound post-reservation device reality.
//!
//! The generic `symthaea-iot-posture` layer correctly treats raw `DeviceRuntimeState`
//! as non-attested input, but its v0.1 signature-provider trait and metadata-only key
//! registry remain too ambient for the minimal privileged actuation guard TCB.
//!
//! This crate closes that guard-facing seam by owning:
//!
//! - an immutable, independently anchored local device-reality policy;
//! - an anti-rollback verifier trust registry that commits the **actual Ed25519 public
//!   key bytes**; and
//! - one fixed RFC 8032 Ed25519 verification path with no provider argument, fallback
//!   or caller-selected algorithm.
//!
//! The historical semantic-reservation verifier remains for lineage compatibility. The
//! current privileged path is [`GuardAdmissionDeviceRealityState`], which consumes the
//! reservation-bound evidence introduced after durable admission reservation.
//!
//! A later physical attempt must not assume that historical verification remains current.
//! [`CurrentAdmissionDeviceRealityGuard`] re-fences the exact proof under independently anchored
//! current policy/trust, the exact verifier key, fixed Ed25519, and natural expiry while borrowing
//! the checked current state for the attempt.
//!
//! Successful verification and current fencing are still **not actuator authority**. Durable
//! semantic state, controller interlock, multi-root linearization and HAL/device I/O remain
//! separate later stages.

#![deny(unsafe_code)]

mod admission;
mod current;
mod error;
mod policy;
mod trust;
mod verifier;

pub use admission::{GuardAdmissionDeviceRealityState, VerifiedAdmissionDeviceReality};
pub use current::{CurrentAdmissionDeviceRealityFence, CurrentAdmissionDeviceRealityGuard};
pub use error::DeviceRealityError;
pub use policy::{
    DEVICE_REALITY_POLICY_SCHEMA_VERSION, DeviceRealityPolicyV1,
    MAX_DEVICE_REALITY_RESULT_LIFETIME_MS,
};
pub use trust::{
    DEVICE_REALITY_TRUST_SCHEMA_VERSION, DeviceRealityTrustHead, DeviceRealityTrustRegistry,
    DeviceRealityTrustSnapshotV1, DeviceRealityVerifierKeyStatus, DeviceRealityVerifierKeyV1,
};
pub use verifier::{GuardDeviceRealityState, VerifiedPostReservationDeviceReality};

/// Exact algorithm/profile accepted by the privileged device-reality verifier.
pub const DEVICE_REALITY_ED25519_ALGORITHM: &str = "ed25519-rfc8032";
/// Exact RFC 8032 Ed25519 public-key length committed in trust state.
pub const DEVICE_REALITY_ED25519_PUBLIC_KEY_LEN: usize = 32;
/// Exact RFC 8032 Ed25519 signature length accepted from posture results.
pub const DEVICE_REALITY_ED25519_SIGNATURE_LEN: usize = 64;
/// Shared bound for verifier/key identity labels.
pub const MAX_DEVICE_REALITY_ID_BYTES: usize = 256;

pub(crate) fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_DEVICE_REALITY_ID_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}
