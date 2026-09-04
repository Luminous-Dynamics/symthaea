// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reservation-bound device-reality challenge for privileged cyber-physical actuation.
//!
//! This crate corrects the first challenge in the earlier two-phase sketch: the state
//! that exists before authenticated device reality is a crash-durable **admission
//! reservation**, not a semantic device checkpoint. Challenge construction therefore
//! requires opaque persisted-reservation proof.
//!
//! Decoded device responses establish only canonical/correlation facts. Signature trust,
//! semantic acceptance, controller interlocks, final authority, and HAL remain later.

#![deny(unsafe_code)]

mod challenge;
mod error;
mod response;

pub use challenge::*;
pub use error::*;
pub use response::*;

use symthaea_authority::{Digest32, ResourceRef};

pub const ADMISSION_REALITY_CHALLENGE_SCHEMA_VERSION: u16 = 1;
pub const ADMISSION_DEVICE_REALITY_RESPONSE_SCHEMA_VERSION: u16 = 1;
pub const MAX_ADMISSION_REALITY_CHALLENGE_LIFETIME_MS: u64 = 5_000;
pub const MAX_ADMISSION_DEVICE_ATTESTATION_BYTES: usize = 96 * 1024;
pub const MAX_ADMISSION_DEVICE_RESPONSE_BYTES: usize = 128 * 1024;

pub(crate) const CHALLENGE_DOMAIN: &[u8] = b"symthaea-iot-admission-reality-challenge-v1\0";
pub(crate) const RESPONSE_DOMAIN: &[u8] = b"symthaea-iot-admission-device-response-v1\0";
pub(crate) const ATTESTATION_OBJECT_DOMAIN: &[u8] =
    b"symthaea-iot-admission-device-attestation-object-v1\0";

pub(crate) fn digest_bytes(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    h.update(&(bytes.len() as u64).to_be_bytes());
    h.update(bytes);
    Digest32(*h.finalize().as_bytes())
}

pub(crate) fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u32).to_be_bytes());
    h.update(value.as_bytes());
}

pub(crate) fn update_digest(h: &mut blake3::Hasher, Digest32(value): Digest32) {
    h.update(&value);
}

pub(crate) fn valid_device(device: &ResourceRef) -> bool {
    !device.0.is_empty() && device.0.trim() == device.0
}
