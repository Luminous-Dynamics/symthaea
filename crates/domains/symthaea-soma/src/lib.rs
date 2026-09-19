// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symthaea Soma
//!
//! Mobile-embodied consciousness engine wrapping the Spore kernel.
//!
//! Spore is the pure consciousness kernel (~500KB WASM) — HDC, CfC, IIT, neuromod,
//! harmonies, substrate independence, and epistemic honesty. Soma extends it with
//! phone-body embodiment: sensors, haptics, sleep/wake metabolism, BLE mesh,
//! device pairing, holon desktop sync, and screen vision.
//!
//! ## Architecture
//!
//! ```text
//! SomaEngine wraps SporeEngine + adds:
//!   SensorBridge    (accel/gyro/light → neuromod nudges)
//!   HapticManager   (consciousness-gated vibration events)
//!   Metabolism      (Sleep/Drowsy/Alert/Focused state machine)
//!   BleMesh         (BLE peer discovery & consciousness sharing)
//!   HolonBridge     (desktop ↔ phone sync)
//!   PairingManager  (Ed25519 trust establishment)
//!   ScreenVisionBridge  (screen framebuffer → visual perception)
//!   TouchBody       (touch events → proprioceptive signals)
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use symthaea_soma::{SomaEngine, SomaConfig};
//!
//! let config = SomaConfig::default();
//! let mut engine = SomaEngine::new(config);
//! let result = engine.cycle("hello world");
//! println!("Consciousness: {}", result.consciousness_level);
//! ```

#![cfg_attr(
    not(any(feature = "native-ffi", feature = "litert", feature = "prism-search")),
    deny(unsafe_code)
)]

// Re-export Spore kernel types for downstream consumers
pub use symthaea_spore::broca;
pub use symthaea_spore::config;
pub use symthaea_spore::engine::SporeEngine;
pub use symthaea_spore::engine::{CycleResult, EpistemicStatus};
pub use symthaea_spore::persistence;

// Mobile embodiment modules (moved from spore)
pub mod ble_mesh;
pub mod haptic;
pub mod holon_bridge;
pub mod metabolism;
pub mod sensor_bridge;

// Always compiled: pairing.rs internally selects real Ed25519 signatures
// (`pairing` feature) vs. an X25519-DH-backed fallback (feature off) --
// gating the whole module on `pairing` would make the documented fallback
// path permanently dead code and leave non-`pairing` builds with no device
// pairing at all.
pub mod pairing;

// Strict Ed25519 verifier adapter for the presentation-assurance attestation trait.
#[cfg(feature = "pairing")]
pub mod assurance_ed25519;

// Screen embodiment (Phase 3)
#[cfg(feature = "screen-vision")]
pub mod screen_vision;

#[cfg(feature = "screen-vision")]
pub mod touch_body;

// Exact framebuffer/touch observations for measured presentation interaction evidence.
#[cfg(feature = "screen-vision")]
pub mod assurance_soma_interaction;

// Recompute measured interaction evidence from exact raw framebuffer/touch inputs.
#[cfg(feature = "screen-vision")]
pub mod assurance_soma_interaction_verify;

// Crypto-agile authentication of measured framebuffer/touch observation identities.
#[cfg(feature = "screen-vision")]
pub mod assurance_measured_attestation;

// Strict reject-not-normalize platform ingress semantics for assurance-bearing input.
#[cfg(feature = "screen-vision")]
pub mod assurance_platform_ingress;

// Current-policy admission for exact strict ingress profiles and observations.
#[cfg(feature = "screen-vision")]
pub mod assurance_ingress_policy;

// Android trusted-attention state and policy requirements remain a vector of
// independently observed platform protections rather than one "secure" bit.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_attention;

// Current policy must admit the exact Android attention requirement rather than
// merely sharing its policy context/generation/root.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_attention_policy;

// Bind each Android MotionEvent's obscuration/pointer facts to the current
// attention theorem and the exact measured touch observation.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_motion_event;

// Current policy must also admit the exact MotionEvent requirement; a caller may
// not retain valid attention policy while weakening obscuration/pointer rules.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_motion_policy;

// Join policy-admitted Android MotionEvent evidence to the exact policy-admitted
// native touch ingress evidence before any platform adapter claims one touch.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_touch_join;

// Freeze the exact 523/526/527 policy closure before event-time Android code can
// reference it; installation/currentness authority remains a separate boundary.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_policy_bundle;

// Install, replace, revoke, and verify one exact 529 policy bundle in a monotonic
// Rust-owned session before event-time JNI can reference it.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_policy_session;

// Own the active 529 bundle and 530 session state behind one Rust boundary so
// event-time code cannot supply alternate policy evidence during verification.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_policy_store;

// Append-only, replay-verified recovery history for the 531 policy store. This
// closes full-session authorization replay and defines rollback/fork checkpoints;
// durable storage and rollback-resistant anchoring remain platform boundaries.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_policy_journal;

// Canonical, strict, versioned persistence bytes for 532 journal/checkpoint state.
// Decoding is bounded and must survive semantic replay plus exact re-encoding.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_policy_codec;

// Exact journal equality is exact ordered-entry equality. This is useful for
// canonical codec roundtrips and does not imply authority or currentness.
#[cfg(feature = "screen-vision")]
impl PartialEq for assurance_android_policy_journal::AndroidTouchPolicyJournal {
    fn eq(&self, other: &Self) -> bool {
        self.entries() == other.entries()
    }
}

#[cfg(feature = "screen-vision")]
impl Eq for assurance_android_policy_journal::AndroidTouchPolicyJournal {}

// Receipt decoding exposes 530 validation failures directly. Keep the codec's
// public error surface compact by preserving those failures through the existing
// 532 journal error chain rather than flattening them into a generic parse error.
#[cfg(feature = "screen-vision")]
impl From<assurance_android_policy_session::AndroidTouchPolicySessionError>
    for assurance_android_policy_codec::AndroidTouchPolicyCodecError
{
    fn from(value: assurance_android_policy_session::AndroidTouchPolicySessionError) -> Self {
        Self::Journal(assurance_android_policy_journal::AndroidTouchPolicyJournalError::Session(
            value,
        ))
    }
}

// Two-plane durability: canonical journal bytes must be durably/atomically
// published and the resulting checkpoint must advance in a distinct anchor domain.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_policy_durability;

// Test-only exact reconstruction of 534's journal commit identity. Keeping this
// helper outside the production 534 module avoids widening its internal certifier
// solely to build successor fixtures in higher-level recovery tests.
#[cfg(all(test, feature = "screen-vision"))]
impl assurance_android_policy_durability::AndroidTouchPolicyJournalDurabilityObservation {
    pub(crate) fn certify_for_test(
        &self,
        expected_digest: assurance_android_policy_durability::AndroidTouchPolicyJournalBytesDigest,
        checkpoint_id: assurance_android_policy_journal::AndroidTouchPolicyCheckpointId,
        _previous: Option<&assurance_android_policy_durability::AndroidTouchPolicyDurableState>,
    ) -> Result<
        assurance_android_policy_durability::AndroidTouchPolicyJournalDurableCommitId,
        assurance_android_policy_durability::AndroidTouchPolicyDurabilityError,
    > {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.soma.presentation.v1/android-policy-journal-durable-commit\0");
        hasher.update(self.storage_domain_id.as_bytes());
        hasher.update(self.transaction_id.as_bytes());
        hasher.update(&self.previous_storage_generation.to_le_bytes());
        hasher.update(&self.next_storage_generation.to_le_bytes());
        hasher.update(expected_digest.as_bytes());
        hasher.update(checkpoint_id.as_bytes());
        hasher.update(self.durable_prepare_evidence_id.as_bytes());
        hasher.update(self.atomic_publish_evidence_id.as_bytes());
        hasher.update(self.metadata_durability_evidence_id.as_bytes());
        hasher.update(self.exact_readback_evidence_id.as_bytes());
        Ok(
            assurance_android_policy_durability::AndroidTouchPolicyJournalDurableCommitId(
                *hasher.finalize().as_bytes(),
            ),
        )
    }
}

// Android-specific capability admission for 534. AtomicFile, Keystore, StrongBox,
// and external monotonic anchors keep their actual guarantees distinct.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_storage_capabilities;

// Bind restart recovery to the exact 534 durable state before event-time policy
// currentness can be re-established. Revoked state recovers but stays revoked.
#[cfg(feature = "screen-vision")]
pub mod assurance_android_policy_recovery;

// Test-only equality is exact recovery receipt + exact recovered snapshot. This
// exists only so negative Result assertions can remain precise in recovery tests.
#[cfg(all(test, feature = "screen-vision"))]
impl PartialEq for assurance_android_policy_recovery::AndroidTouchPolicyRecoveredStore {
    fn eq(&self, other: &Self) -> bool {
        self.recovery_receipt() == other.recovery_receipt()
            && self.snapshot().ok() == other.snapshot().ok()
    }
}

#[cfg(all(test, feature = "screen-vision"))]
impl Eq for assurance_android_policy_recovery::AndroidTouchPolicyRecoveredStore {}

// Versioned checked C ABI that routes accepted native input through the strict ingress contract.
#[cfg(all(feature = "native-ffi", feature = "screen-vision"))]
pub mod assurance_native_ingress;

// Decentralized positioning (GPS-independent)
#[cfg(feature = "positioning")]
pub mod positioning_bridge;

// Full Broca language center (replaces BrocaLite)
#[cfg(feature = "broca-full")]
pub mod broca_soma;

// On-device LLM via LiteRT-LM (Gemma 4 E2B)
#[cfg(feature = "litert")]
pub mod litert_bridge;

// Tool-use framework for LLM function calling
#[cfg(feature = "litert")]
pub mod tool_use;

// Broca + LiteRT fusion engine
#[cfg(feature = "fusion")]
pub mod fusion;

// Native FFI for Android/iOS
#[cfg(feature = "native-ffi")]
pub mod native_ffi;

// SomaEngine — the mobile consciousness engine
pub mod engine;

pub use engine::{SomaConfig, SomaEngine, SomaEngineHandle};
