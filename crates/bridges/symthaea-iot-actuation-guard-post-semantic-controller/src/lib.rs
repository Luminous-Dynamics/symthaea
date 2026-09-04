// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Post-semantic controller challenge for privileged cyber-physical actuation.
//!
//! This stage begins only after semantic safety has reached crash-durable storage.
//! The outbound challenge is constructor-gated by `PersistedSemanticAcceptance`, uses
//! guard-owned entropy/time, and cannot outlive either authenticated device reality or
//! the original physical-effect deadline.
//!
//! The portable response carries only controller report + controller evidence. It is
//! canonical/correlation evidence, not authority: fixed current controller-key trust,
//! interlock policy, final composition, JIT fencing and HAL remain later stages.

#![deny(unsafe_code)]

mod challenge;
mod error;
mod response;
#[cfg(test)]
mod tests;

pub use challenge::PostSemanticControllerChallengeV1;
pub use error::PostSemanticControllerError;
pub use response::{
    DecodedPostSemanticControllerEvidence, PostSemanticControllerResponseV1,
    decode_post_semantic_controller_response,
};

use symthaea_authority::{Digest32, ResourceRef};

pub const POST_SEMANTIC_CONTROLLER_CHALLENGE_SCHEMA_VERSION: u16 = 1;
pub const POST_SEMANTIC_CONTROLLER_RESPONSE_SCHEMA_VERSION: u16 = 1;
/// Short round-trip ceiling after durable semantic safety state exists.
pub const MAX_POST_SEMANTIC_CONTROLLER_CHALLENGE_LIFETIME_MS: u64 = 2_000;
/// Current controller profile is compact Ed25519 evidence; keep the wire bound narrow.
pub const MAX_POST_SEMANTIC_CONTROLLER_EVIDENCE_BYTES: usize = 4 * 1024;
pub const MAX_POST_SEMANTIC_CONTROLLER_RESPONSE_BYTES: usize = 24 * 1024;
pub const MAX_POST_SEMANTIC_DEVICE_ID_BYTES: usize = 512;

pub(crate) const POST_SEMANTIC_CONTROLLER_CHALLENGE_DOMAIN: &[u8] =
    b"symthaea-iot-post-semantic-controller-challenge-v1\0";
pub(crate) const POST_SEMANTIC_CONTROLLER_RESPONSE_DOMAIN: &[u8] =
    b"symthaea-iot-post-semantic-controller-response-v1\0";

pub(crate) fn digest_frame(domain: &[u8], bytes: &[u8]) -> Digest32 {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    h.update(&(bytes.len() as u64).to_be_bytes());
    h.update(bytes);
    Digest32(*h.finalize().as_bytes())
}

pub(crate) fn update_digest(h: &mut blake3::Hasher, Digest32(value): Digest32) {
    h.update(&value);
}

pub(crate) fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u32).to_be_bytes());
    h.update(value.as_bytes());
}

pub(crate) fn valid_device(device: &ResourceRef) -> bool {
    !device.0.is_empty()
        && device.0.len() <= MAX_POST_SEMANTIC_DEVICE_ID_BYTES
        && device.0.trim() == device.0
        && !device.0.chars().any(char::is_control)
}
