// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Guard-owned verification of challenge-bound physical-effect outcome evidence.
//!
//! This crate deliberately separates two proof classes that are easy to conflate during recovery:
//!
//! - `ExecutionAndPostcondition`: a tamper-evident execution record for the exact command inside
//!   the original actuation window plus a fresh postcondition observation;
//! - `NonExecution`: a tamper-evident non-execution proof whose certified log coverage spans the
//!   complete original actuation window.
//!
//! A current state observation alone can never produce `NonExecution`. Likewise, adapter
//! acknowledgement, command-sequence movement and generic telemetry are insufficient inputs.
//!
//! V1 verification remains available for compatibility. The V2 policy-bound surface signs the
//! authoritative outcome-policy generation/digest inside the evidence itself and makes the
//! V1-compatibility decision part of the V2 policy commitment rather than a later terminal switch.
//!
//! Verification uses fixed RFC 8032 Ed25519, guard-owned policy, exact concrete verifier keys and
//! independently anchored trust heads. Returned proofs remain non-authorizing and cannot close the
//! durable effect-attempt journal. A later reconciliation writer must re-fence the exact proof under
//! current outcome-verifier trust and independently re-check the protected journal head.

#![deny(unsafe_code)]

mod current;
mod error;
mod evidence;
mod policy;
mod provenance_v2;
mod trust;
mod verifier;

#[cfg(test)]
mod tests;

pub use current::{CurrentPhysicalEffectOutcomeFence, CurrentPhysicalEffectOutcomeGuard};
pub use error::EffectOutcomeError;
pub use evidence::{
    EFFECT_OUTCOME_EVIDENCE_SCHEMA_VERSION, EffectOutcomeClaimKindV1, EffectOutcomeClaimV1,
    PhysicalEffectOutcomeEvidenceBodyV1, PhysicalEffectOutcomeEvidenceV1,
};
pub use policy::{
    EFFECT_OUTCOME_POLICY_SCHEMA_VERSION, EffectOutcomePolicyV1,
    MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS,
};
pub use provenance_v2::{
    EFFECT_OUTCOME_EVIDENCE_V2_SCHEMA_VERSION, EFFECT_OUTCOME_EVIDENCE_V2_WIRE_MAGIC,
    EFFECT_OUTCOME_POLICY_V2_SCHEMA_VERSION, CurrentPhysicalEffectOutcomeFenceV2,
    CurrentPhysicalEffectOutcomeGuardV2, EffectOutcomeEvidenceProvenanceModeV2,
    EffectOutcomePolicyV2, GuardPhysicalEffectOutcomeStateV2, PhysicalEffectOutcomeEvidenceBodyV2,
    PhysicalEffectOutcomeEvidenceV2, PolicyBoundOutcomeError,
    VerifiedPhysicalEffectOutcomeEvidenceV2,
};
pub use trust::{
    EFFECT_OUTCOME_TRUST_SCHEMA_VERSION, EffectOutcomeTrustHead, EffectOutcomeTrustRegistry,
    EffectOutcomeTrustSnapshotV1, EffectOutcomeVerifierKeyStatus, EffectOutcomeVerifierKeyV1,
};
pub use verifier::{GuardPhysicalEffectOutcomeState, VerifiedPhysicalEffectOutcomeEvidence};

pub const EFFECT_OUTCOME_ED25519_ALGORITHM: &str = "ed25519-rfc8032";
pub const EFFECT_OUTCOME_ED25519_PUBLIC_KEY_LEN: usize = 32;
pub const EFFECT_OUTCOME_ED25519_SIGNATURE_LEN: usize = 64;
pub const MAX_EFFECT_OUTCOME_ID_BYTES: usize = 256;
pub const MAX_EFFECT_OUTCOME_DEVICE_ID_BYTES: usize = 512;

pub(crate) fn valid_id(value: &str, max: usize) -> bool {
    !value.is_empty()
        && value.len() <= max
        && value.trim() == value
        && !value.chars().any(char::is_control)
}
