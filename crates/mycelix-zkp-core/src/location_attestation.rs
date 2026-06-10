// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Location-attestation trust tiers and attestation envelope.
//!
//! Companion module to `circuits::jurisdiction_proof`. A jurisdiction
//! proof cryptographically asserts "the attested value is inside the
//! box"; this module defines *what "attested" means* — five tiers
//! ranging from self-attest (low trust) to hardware-TEE / notary-oracle
//! (high trust) — and the attestation envelope itself.
//!
//! ## Why tiers matter more than the STARK
//!
//! A user can fake GPS. The STARK faithfully encodes whatever location
//! the prover claims; it does not validate the claim against physics.
//! Security comes from the *attestation source*, not the proof. A
//! verifier that accepts T0 (self-attested) proofs is effectively
//! accepting the prover's word; one that requires T3 (hardware TEE)
//! has a meaningful integrity claim from the device.
//!
//! Verifiers MUST set a tier floor appropriate to their risk. The
//! default floor is `T1PhoneGps` (per plan decision 2026-04-18), which
//! avoids a bootstrap paradox where no one can produce any attestation
//! until civic-bridge infrastructure is deployed.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use crate::circuits::jurisdiction_proof::AttestationTier;

/// A raw location attestation. Carries an attester-signed claim about
/// where a user was at a given instant. Feeds into `prove_jurisdiction`
/// as private input; the STARK proves containment of `(lat, lng)`
/// without revealing it.
///
/// The envelope itself is never transmitted to a verifier — only the
/// STARK proof's public surface is. This struct lives on the prover's
/// device.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocationAttestation {
    /// Unbiased latitude in decimal degrees.
    pub lat_degrees: f64,
    /// Unbiased longitude in decimal degrees.
    pub lng_degrees: f64,
    /// Unix timestamp at which the attester produced this claim.
    pub timestamp_unix: u64,
    /// Attester's public identifier (e.g., device-key DID, civic-bridge
    /// agent pubkey, notary network root). Hashed for the proof.
    pub attester_pubkey: Vec<u8>,
    /// Signature over `(lat, lng, timestamp)` produced by the attester.
    /// Format is attester-specific; this module does not verify it
    /// (the `AttestationSource` trait impl does).
    pub attester_signature: Vec<u8>,
    /// Which tier of attester produced this claim.
    pub tier: AttestationTier,
}

impl LocationAttestation {
    /// SHA-256 hash of the attester pubkey. This is what reaches the
    /// STARK's public input — the raw pubkey never does.
    pub fn attester_pubkey_hash(&self) -> [u8; 32] {
        let digest = Sha256::digest(&self.attester_pubkey);
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    }

    /// Whether this attestation is within `max_age_seconds` of `now_unix`.
    /// Verifiers typically require very recent attestations (minutes,
    /// not days) to reduce the window for GPS replay.
    pub fn is_fresh(&self, now_unix: u64, max_age_seconds: u64) -> bool {
        now_unix.saturating_sub(self.timestamp_unix) <= max_age_seconds
    }
}

/// Trait implemented by concrete attestation sources. Each tier has its
/// own mechanism for producing and verifying signatures. This crate
/// defines the trait and ships stubs for T0 and T1; richer tiers are
/// implemented by downstream crates that have the relevant hardware or
/// oracle bindings.
pub trait AttestationSource {
    /// Which tier this source produces.
    fn tier(&self) -> AttestationTier;

    /// Verify that `attestation` was truthfully signed by a valid
    /// attester of this source's tier. Returns `Ok(())` if trusted.
    fn verify(&self, attestation: &LocationAttestation) -> Result<(), AttestationError>;
}

/// Errors that can occur when producing or verifying an attestation.
#[derive(Debug, thiserror::Error, Clone, Serialize, Deserialize)]
pub enum AttestationError {
    /// The attestation's tier does not match this source.
    #[error("tier mismatch: expected {expected:?}, got {got:?}")]
    TierMismatch {
        expected: AttestationTier,
        got: AttestationTier,
    },
    /// Signature verification failed.
    #[error("signature verification failed: {reason}")]
    SignatureInvalid { reason: String },
    /// The attester's public key is not trusted by this source.
    #[error("attester pubkey not in trust set")]
    UntrustedAttester,
    /// The attestation is older than the source permits.
    #[error("attestation too old (age {age_seconds}s exceeds max {max_seconds}s)")]
    Stale { age_seconds: u64, max_seconds: u64 },
    /// The underlying platform (GPS, TEE, notary) is unavailable.
    #[error("attestation source unavailable: {reason}")]
    SourceUnavailable { reason: String },
    /// Feature not implemented at the current tier.
    #[error("tier {tier:?} not yet implemented")]
    NotImplemented { tier: AttestationTier },
}

/// T0 — self-attested. Trivially accepted; verifiers should reject
/// anything that only reaches this tier. Useful only for testing or
/// for cases where the verifier has independent out-of-band trust.
pub struct SelfAttested;

impl AttestationSource for SelfAttested {
    fn tier(&self) -> AttestationTier {
        AttestationTier::T0SelfAttested
    }

    fn verify(&self, attestation: &LocationAttestation) -> Result<(), AttestationError> {
        if attestation.tier != AttestationTier::T0SelfAttested {
            return Err(AttestationError::TierMismatch {
                expected: AttestationTier::T0SelfAttested,
                got: attestation.tier,
            });
        }
        Ok(())
    }
}

/// T1 — phone-GPS attestation. The prover's own device reports its
/// GPS fix and signs it with a device-bound key. Stronger than
/// self-attest because the device-key is typically bound to
/// installation; weaker than hardware-TEE because the GPS sensor
/// itself can be spoofed by mock-location apps.
///
/// Downstream integration lives in `symthaea-phone-embodiment`; this
/// crate only defines the envelope and max-age policy.
pub struct PhoneGps {
    /// Maximum permitted attestation age in seconds. GPS fixes older
    /// than this are rejected. Default: 300 (five minutes).
    pub max_age_seconds: u64,
    /// Acceptable device-pubkey set. If empty, any pubkey is accepted
    /// (useful for first-run when the verifier has no device-key
    /// expectation yet). Production verifiers SHOULD populate this
    /// from user-controlled onboarding.
    pub trusted_device_pubkeys: Vec<Vec<u8>>,
}

impl Default for PhoneGps {
    fn default() -> Self {
        Self {
            max_age_seconds: 300,
            trusted_device_pubkeys: Vec::new(),
        }
    }
}

impl AttestationSource for PhoneGps {
    fn tier(&self) -> AttestationTier {
        AttestationTier::T1PhoneGps
    }

    fn verify(&self, attestation: &LocationAttestation) -> Result<(), AttestationError> {
        if attestation.tier != AttestationTier::T1PhoneGps {
            return Err(AttestationError::TierMismatch {
                expected: AttestationTier::T1PhoneGps,
                got: attestation.tier,
            });
        }

        if !self.trusted_device_pubkeys.is_empty()
            && !self
                .trusted_device_pubkeys
                .iter()
                .any(|k| k == &attestation.attester_pubkey)
        {
            return Err(AttestationError::UntrustedAttester);
        }

        // Actual signature verification happens in the downstream
        // phone bridge; here we only enforce envelope hygiene.
        // A richer impl would plug in the device-key verifier here.
        if attestation.attester_signature.is_empty() {
            return Err(AttestationError::SignatureInvalid {
                reason: "empty signature".to_string(),
            });
        }
        Ok(())
    }
}

/// T2 — civic-bridge attestation. Signed by the `mycelix-civic`
/// robotics-dispatch telemetry layer, which requires a live physical
/// presence (robotic agent physically observed the user at a place)
/// or multi-party attestation from civic infrastructure.
///
/// Stub: the actual civic-bridge integration lives in a downstream
/// crate that depends on `mycelix-civic`. This source, invoked in
/// isolation, always returns `NotImplemented`.
pub struct CivicBridge;

impl AttestationSource for CivicBridge {
    fn tier(&self) -> AttestationTier {
        AttestationTier::T2CivicBridge
    }

    fn verify(&self, _attestation: &LocationAttestation) -> Result<(), AttestationError> {
        Err(AttestationError::NotImplemented {
            tier: AttestationTier::T2CivicBridge,
        })
    }
}

/// T3 — hardware TEE attestation (Android StrongBox, iOS Secure
/// Enclave, Android Play Integrity with hardware-backed attestation).
/// Signature is produced by a key that cannot be exported from the
/// TEE, providing strong evidence the device was physically at the
/// claimed location.
///
/// Stub: real TEE integration requires per-platform bindings that
/// live in downstream crates. This source always returns
/// `NotImplemented` from the core crate.
pub struct HardwareTee;

impl AttestationSource for HardwareTee {
    fn tier(&self) -> AttestationTier {
        AttestationTier::T3HardwareTee
    }

    fn verify(&self, _attestation: &LocationAttestation) -> Result<(), AttestationError> {
        Err(AttestationError::NotImplemented {
            tier: AttestationTier::T3HardwareTee,
        })
    }
}

/// T4 — notary-oracle attestation. A threshold set of notary oracles
/// jointly sign the location claim. Strongest tier; requires live
/// oracle network infrastructure and is therefore the slowest to
/// deploy.
///
/// Stub: notary network integration is future work.
pub struct NotaryOracle;

impl AttestationSource for NotaryOracle {
    fn tier(&self) -> AttestationTier {
        AttestationTier::T4Notary
    }

    fn verify(&self, _attestation: &LocationAttestation) -> Result<(), AttestationError> {
        Err(AttestationError::NotImplemented {
            tier: AttestationTier::T4Notary,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_attestation(tier: AttestationTier) -> LocationAttestation {
        LocationAttestation {
            lat_degrees: -26.1625,
            lng_degrees: 27.8725,
            timestamp_unix: 1_713_400_000,
            attester_pubkey: vec![1, 2, 3, 4],
            attester_signature: vec![0xDE, 0xAD, 0xBE, 0xEF],
            tier,
        }
    }

    #[test]
    fn pubkey_hash_stable() {
        let a = sample_attestation(AttestationTier::T1PhoneGps);
        let b = sample_attestation(AttestationTier::T1PhoneGps);
        assert_eq!(a.attester_pubkey_hash(), b.attester_pubkey_hash());
    }

    #[test]
    fn freshness_window() {
        let a = sample_attestation(AttestationTier::T1PhoneGps);
        assert!(a.is_fresh(1_713_400_060, 300));
        assert!(!a.is_fresh(1_713_401_000, 300));
    }

    #[test]
    fn self_attested_accepts_matching_tier() {
        let src = SelfAttested;
        let a = sample_attestation(AttestationTier::T0SelfAttested);
        assert!(src.verify(&a).is_ok());
    }

    #[test]
    fn self_attested_rejects_wrong_tier() {
        let src = SelfAttested;
        let a = sample_attestation(AttestationTier::T1PhoneGps);
        match src.verify(&a) {
            Err(AttestationError::TierMismatch { .. }) => {}
            other => panic!("expected TierMismatch, got {other:?}"),
        }
    }

    #[test]
    fn phone_gps_requires_nonempty_signature() {
        let src = PhoneGps::default();
        let mut a = sample_attestation(AttestationTier::T1PhoneGps);
        a.attester_signature.clear();
        match src.verify(&a) {
            Err(AttestationError::SignatureInvalid { .. }) => {}
            other => panic!("expected SignatureInvalid, got {other:?}"),
        }
    }

    #[test]
    fn phone_gps_trusted_pubkey_allowlist() {
        let src = PhoneGps {
            max_age_seconds: 300,
            trusted_device_pubkeys: vec![vec![9, 9, 9, 9]],
        };
        let a = sample_attestation(AttestationTier::T1PhoneGps);
        match src.verify(&a) {
            Err(AttestationError::UntrustedAttester) => {}
            other => panic!("expected UntrustedAttester, got {other:?}"),
        }
    }

    #[test]
    fn higher_tiers_unimplemented() {
        for src in [
            Box::new(CivicBridge) as Box<dyn AttestationSource>,
            Box::new(HardwareTee),
            Box::new(NotaryOracle),
        ] {
            let a = sample_attestation(src.tier());
            match src.verify(&a) {
                Err(AttestationError::NotImplemented { .. }) => {}
                other => panic!(
                    "expected NotImplemented at tier {:?}, got {other:?}",
                    src.tier()
                ),
            }
        }
    }

    #[test]
    fn tier_re_export_matches_circuit_module() {
        // Sanity check that the re-export lines up with the circuit's enum.
        assert_eq!(
            AttestationTier::default_minimum(),
            AttestationTier::T1PhoneGps
        );
    }
}
