// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! STARK proof that an attested (lat, lng) falls inside a published
//! jurisdiction bounding box — without revealing the coordinate.
//!
//! Part of the Mycelix × Nation-State coexistence extensions. See
//! `MYCELIX_STATE_COEXISTENCE.md` at the repo root for context. The
//! verifier learns only "user is in jurisdiction commitment C at time T
//! with attestation tier X"; never the coordinate, the attester's
//! identifier beyond its hash, or the raw timestamp beyond what the
//! commitment binds.
//!
//! ## Trust model caveat — READ BEFORE USE
//!
//! The STARK proves *containment of the attested value*, not that the
//! attestation itself is truthful. A user can fake GPS. Security comes
//! from the attester (T0..T4 tiers in `location_attestation.rs`),
//! not from the proof. Verifiers MUST set a tier floor appropriate to
//! their risk tolerance.
//!
//! ## Construction
//!
//! A jurisdiction bounding box is `{ lat ∈ [lat_min, lat_max], lng ∈
//! [lng_min, lng_max] }`. The proof bundles two range proofs (one per
//! axis) from the existing `range_proof` circuit, plus a commitment
//! binding the proof to `(box_id, attester_hash, timestamp, verifier
//! nonce, tier)`. The two range proofs prove lat and lng containment
//! independently; AND-ing them is implicit (both must verify).
//!
//! Latitude and longitude are in microdegrees (×1_000_000), biased by
//! +180_000_000 so the value range is `[0, 360_000_000]`, fitting the
//! underlying `u64` range proof while handling the Southern and Western
//! hemispheres uniformly.
//!
//! ## Privacy properties
//!
//! - Verifier sees: `{ box_id, attestation_tier, nonce_hash, commitment,
//!   two STARK proofs }`. That is the complete public surface.
//! - Verifier does NOT see: lat, lng, attester identity beyond its hash,
//!   or anything that makes two proofs from the same location linkable
//!   (the fresh nonce makes commitments distinct).
//! - Replay protection: verifier-supplied fresh nonce; reused nonces
//!   MUST be rejected by the verifier (LRU window of 65,536 entries).

#[cfg(feature = "backend-winterfell")]
use crate::circuits::range_proof;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Microdegree scale: multiply decimal degrees by this to get the integer.
pub const MICRODEG_SCALE: i64 = 1_000_000;

/// Bias added to (lat, lng) before range-proof encoding so that negative
/// latitudes and longitudes (Southern + Western hemispheres) still fit
/// in the `u64` range the underlying circuit uses.
pub const COORD_BIAS: i64 = 180 * MICRODEG_SCALE;

/// Attestation tier for a location claim. See `location_attestation.rs`
/// for the full semantics. Included here as a public input so verifiers
/// can reject proofs below their required tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AttestationTier {
    /// Self-attested. Low trust. Verifier SHOULD reject for most uses.
    T0SelfAttested = 0,
    /// Attested by the user's phone GPS via Holon-Soma bridge.
    /// Default minimum tier (see plan decision 2026-04-18).
    T1PhoneGps = 1,
    /// Attested by the civic-bridge robotics-dispatch telemetry layer.
    T2CivicBridge = 2,
    /// Attested by a hardware TEE (Android StrongBox, iOS Secure Enclave, etc.).
    T3HardwareTee = 3,
    /// Attested by a notary oracle network.
    T4Notary = 4,
}

impl AttestationTier {
    /// The suggested default floor: accept T1 or stronger. Verifiers may
    /// override in either direction for their specific trust needs.
    pub const fn default_minimum() -> Self {
        AttestationTier::T1PhoneGps
    }

    /// Whether `self` is at least as strong as `required`.
    pub const fn meets(self, required: AttestationTier) -> bool {
        (self as u8) >= (required as u8)
    }
}

/// A jurisdiction axis-aligned bounding box in biased microdegrees.
///
/// Real-world jurisdictions are described by unions of many such boxes;
/// callers prove containment in a specific box from a published set
/// (see `jurisdiction_registry.rs`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JurisdictionBox {
    /// Stable identifier. Format: `<jurisdiction>-<version>-<box-index>`,
    /// e.g., `"US-tax-residency-v1-0"` or `"ZA-SARS-v1-3"`.
    pub id: String,
    /// Biased lat minimum (add `COORD_BIAS` to signed microdegrees).
    pub lat_min_biased: u64,
    /// Biased lat maximum.
    pub lat_max_biased: u64,
    /// Biased lng minimum.
    pub lng_min_biased: u64,
    /// Biased lng maximum.
    pub lng_max_biased: u64,
}

impl JurisdictionBox {
    /// Build from signed decimal-degree floats. Returns `None` if the
    /// coordinates are outside the valid WGS-84 range or min > max.
    pub fn from_degrees(
        id: impl Into<String>,
        lat_min: f64,
        lat_max: f64,
        lng_min: f64,
        lng_max: f64,
    ) -> Option<Self> {
        if !(-90.0..=90.0).contains(&lat_min)
            || !(-90.0..=90.0).contains(&lat_max)
            || !(-180.0..=180.0).contains(&lng_min)
            || !(-180.0..=180.0).contains(&lng_max)
            || lat_min > lat_max
            || lng_min > lng_max
        {
            return None;
        }
        Some(Self {
            id: id.into(),
            lat_min_biased: biased_microdegrees(lat_min),
            lat_max_biased: biased_microdegrees(lat_max),
            lng_min_biased: biased_microdegrees(lng_min),
            lng_max_biased: biased_microdegrees(lng_max),
        })
    }

    /// Whether the given unbiased decimal degrees fall inside this box.
    /// Useful for callers to pre-check before generating a proof.
    pub fn contains_degrees(&self, lat: f64, lng: f64) -> bool {
        let lat_b = biased_microdegrees(lat);
        let lng_b = biased_microdegrees(lng);
        lat_b >= self.lat_min_biased
            && lat_b <= self.lat_max_biased
            && lng_b >= self.lng_min_biased
            && lng_b <= self.lng_max_biased
    }
}

/// Convert a signed decimal degree to biased microdegrees (`u64`).
pub fn biased_microdegrees(degrees: f64) -> u64 {
    let scaled = (degrees * MICRODEG_SCALE as f64).round() as i64;
    (scaled + COORD_BIAS) as u64
}

/// Request to prove location containment in a jurisdiction box.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JurisdictionProofRequest {
    /// Unbiased latitude in decimal degrees (private; never leaves prover).
    pub lat_degrees: f64,
    /// Unbiased longitude in decimal degrees (private; never leaves prover).
    pub lng_degrees: f64,
    /// The specific jurisdiction box being proven containment in.
    pub jurisdiction_box: JurisdictionBox,
    /// SHA-256 hash of the attester's public identity. Binds the proof
    /// to an attestation source without revealing the source itself.
    pub attester_pubkey_hash: [u8; 32],
    /// Unix timestamp of the attestation. Binds the proof to a moment.
    pub timestamp_unix: u64,
    /// Fresh nonce provided by the verifier at request time. Prevents
    /// replay. Verifier MUST track recently-seen nonces and reject reuse.
    pub verifier_nonce: [u8; 32],
    /// Which tier of attester produced the underlying location claim.
    pub attestation_tier: AttestationTier,
}

/// Result of a successful proof generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JurisdictionProofResult {
    /// STARK proof bytes for the latitude-bound constraint.
    pub lat_proof_bytes: Vec<u8>,
    /// STARK proof bytes for the longitude-bound constraint.
    pub lng_proof_bytes: Vec<u8>,
    /// Commitment binding the proof to `(box_id, attester, time, nonce,
    /// tier)`. Caller publishes this commitment; verifier recomputes.
    pub location_commitment: [u8; 32],
    /// Box identifier being claimed.
    pub jurisdiction_box_id: String,
    /// Attestation tier for verifier-side threshold checks.
    pub attestation_tier: AttestationTier,
    /// Hash of the verifier nonce (verifier re-hashes and compares).
    pub nonce_hash: [u8; 32],
    /// Wall-clock time to generate the bundle, milliseconds.
    pub prove_time_ms: f64,
}

/// Compute the binding commitment for a jurisdiction proof.
///
/// The commitment is what makes two proofs from the same location
/// unlinkable: without the verifier's fresh nonce, two commitments
/// from the same (lat, lng, attester, time) would collide. With a
/// fresh nonce per proof, commitments are distinct.
///
/// The commitment does NOT include the raw coordinates — those are
/// private inputs to the underlying STARK.
pub fn compute_commitment(
    box_id: &str,
    attester_pubkey_hash: &[u8; 32],
    timestamp_unix: u64,
    verifier_nonce: &[u8; 32],
    tier: AttestationTier,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"MYCELIX-JURISDICTION-PROOF:v1:");
    hasher.update(box_id.as_bytes());
    hasher.update(b":");
    hasher.update(attester_pubkey_hash);
    hasher.update(b":");
    hasher.update(timestamp_unix.to_le_bytes());
    hasher.update(b":");
    hasher.update(verifier_nonce);
    hasher.update(b":");
    hasher.update([tier as u8]);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Hash a verifier nonce for inclusion in the public proof surface.
pub fn hash_nonce(nonce: &[u8; 32]) -> [u8; 32] {
    let digest = Sha256::digest(nonce);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Inner commitment the underlying range proofs bind to. Distinct from
/// the outer `location_commitment` so the STARK's public input never
/// leaks the composite (which includes the tier).
fn axis_commitment(location_commitment: &[u8; 32], axis_tag: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"MYCELIX-JURISDICTION-AXIS:v1:");
    hasher.update(location_commitment);
    hasher.update(b":");
    hasher.update(axis_tag);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Generate a jurisdiction-containment proof.
///
/// Requires the Winterfell backend. Will return `Err` if the backend
/// feature is disabled or if the caller's coordinates fall outside the
/// claimed box (an honest-prover check — dishonest provers simply get
/// an unprovable trace).
#[cfg(feature = "backend-winterfell")]
pub fn prove_jurisdiction(
    request: &JurisdictionProofRequest,
) -> Result<JurisdictionProofResult, String> {
    let start = std::time::Instant::now();

    if !request
        .jurisdiction_box
        .contains_degrees(request.lat_degrees, request.lng_degrees)
    {
        return Err(format!(
            "coordinates ({}, {}) outside claimed box {}",
            request.lat_degrees, request.lng_degrees, request.jurisdiction_box.id
        ));
    }

    let lat_biased = biased_microdegrees(request.lat_degrees);
    let lng_biased = biased_microdegrees(request.lng_degrees);
    let b = &request.jurisdiction_box;

    let commitment = compute_commitment(
        &b.id,
        &request.attester_pubkey_hash,
        request.timestamp_unix,
        &request.verifier_nonce,
        request.attestation_tier,
    );

    let lat_commit = axis_commitment(&commitment, b"lat");
    let lng_commit = axis_commitment(&commitment, b"lng");

    let lat_proof =
        range_proof::prove_range(lat_biased, b.lat_min_biased, b.lat_max_biased, lat_commit)
            .map_err(|e| format!("lat range-proof failed: {e}"))?;
    let lng_proof =
        range_proof::prove_range(lng_biased, b.lng_min_biased, b.lng_max_biased, lng_commit)
            .map_err(|e| format!("lng range-proof failed: {e}"))?;

    let lat_bytes = lat_proof.to_bytes();
    let lng_bytes = lng_proof.to_bytes();

    Ok(JurisdictionProofResult {
        lat_proof_bytes: lat_bytes,
        lng_proof_bytes: lng_bytes,
        location_commitment: commitment,
        jurisdiction_box_id: b.id.clone(),
        attestation_tier: request.attestation_tier,
        nonce_hash: hash_nonce(&request.verifier_nonce),
        prove_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Verification input bundle. Caller supplies the box (looked up from
/// the jurisdiction registry) and the verifier's own nonce (to confirm
/// the commitment binds to this verification session).
#[cfg(feature = "backend-winterfell")]
pub struct JurisdictionVerifyInput<'a> {
    pub result: &'a JurisdictionProofResult,
    pub jurisdiction_box: &'a JurisdictionBox,
    pub attester_pubkey_hash: &'a [u8; 32],
    pub timestamp_unix: u64,
    pub expected_verifier_nonce: &'a [u8; 32],
    pub required_tier_minimum: AttestationTier,
}

/// Verify a jurisdiction proof.
///
/// Checks in order:
///   1. `attestation_tier` meets the verifier's required minimum.
///   2. `nonce_hash` matches a re-hash of the expected nonce.
///   3. `location_commitment` is reproducible from the public inputs.
///   4. The box id in the result matches the box the verifier is checking.
///   5. Both STARK range proofs verify against their axis commitments.
#[cfg(feature = "backend-winterfell")]
pub fn verify_jurisdiction(input: &JurisdictionVerifyInput<'_>) -> Result<(), String> {
    let r = input.result;

    if !r.attestation_tier.meets(input.required_tier_minimum) {
        return Err(format!(
            "attestation tier {:?} does not meet required minimum {:?}",
            r.attestation_tier, input.required_tier_minimum
        ));
    }

    if hash_nonce(input.expected_verifier_nonce) != r.nonce_hash {
        return Err("nonce hash mismatch (replay or wrong-session proof)".into());
    }

    let expected_commitment = compute_commitment(
        &input.jurisdiction_box.id,
        input.attester_pubkey_hash,
        input.timestamp_unix,
        input.expected_verifier_nonce,
        r.attestation_tier,
    );
    if expected_commitment != r.location_commitment {
        return Err("location commitment mismatch".into());
    }

    if input.jurisdiction_box.id != r.jurisdiction_box_id {
        return Err("jurisdiction box id mismatch".into());
    }

    let lat_commit = axis_commitment(&expected_commitment, b"lat");
    let lng_commit = axis_commitment(&expected_commitment, b"lng");

    let lat_proof = winterfell::Proof::from_bytes(&r.lat_proof_bytes)
        .map_err(|e| format!("lat proof deserialize: {e:?}"))?;
    let lng_proof = winterfell::Proof::from_bytes(&r.lng_proof_bytes)
        .map_err(|e| format!("lng proof deserialize: {e:?}"))?;

    range_proof::verify_range(
        lat_proof,
        input.jurisdiction_box.lat_min_biased,
        input.jurisdiction_box.lat_max_biased,
        lat_commit,
    )
    .map_err(|e| format!("lat range-proof verify failed: {e}"))?;
    range_proof::verify_range(
        lng_proof,
        input.jurisdiction_box.lng_min_biased,
        input.jurisdiction_box.lng_max_biased,
        lng_commit,
    )
    .map_err(|e| format!("lng range-proof verify failed: {e}"))?;

    Ok(())
}

#[cfg(all(test, feature = "backend-winterfell"))]
mod tests {
    use super::*;

    fn roodepoort_za() -> (f64, f64) {
        // Roughly the user's home area; Gauteng, South Africa.
        (-26.1625, 27.8725)
    }

    fn sa_box() -> JurisdictionBox {
        // Coarse bounding box over South Africa's mainland extent.
        JurisdictionBox::from_degrees("ZA-SARS-v1-0", -35.0, -22.0, 16.0, 33.0).unwrap()
    }

    fn fresh_nonce(seed: u8) -> [u8; 32] {
        let mut n = [0u8; 32];
        for (i, b) in n.iter_mut().enumerate() {
            *b = seed.wrapping_add(i as u8);
        }
        n
    }

    fn attester() -> [u8; 32] {
        [0xAB; 32]
    }

    #[test]
    fn prove_and_verify_in_jurisdiction() {
        let (lat, lng) = roodepoort_za();
        let nonce = fresh_nonce(7);
        let result = prove_jurisdiction(&JurisdictionProofRequest {
            lat_degrees: lat,
            lng_degrees: lng,
            jurisdiction_box: sa_box(),
            attester_pubkey_hash: attester(),
            timestamp_unix: 1_713_400_000,
            verifier_nonce: nonce,
            attestation_tier: AttestationTier::T1PhoneGps,
        })
        .expect("prove");

        verify_jurisdiction(&JurisdictionVerifyInput {
            result: &result,
            jurisdiction_box: &sa_box(),
            attester_pubkey_hash: &attester(),
            timestamp_unix: 1_713_400_000,
            expected_verifier_nonce: &nonce,
            required_tier_minimum: AttestationTier::default_minimum(),
        })
        .expect("verify");
    }

    #[test]
    fn outside_jurisdiction_fails_to_prove() {
        // New York City — definitely not SA.
        let err = prove_jurisdiction(&JurisdictionProofRequest {
            lat_degrees: 40.7128,
            lng_degrees: -74.0060,
            jurisdiction_box: sa_box(),
            attester_pubkey_hash: attester(),
            timestamp_unix: 1_713_400_000,
            verifier_nonce: fresh_nonce(9),
            attestation_tier: AttestationTier::T1PhoneGps,
        })
        .unwrap_err();
        assert!(err.contains("outside claimed box"));
    }

    #[test]
    fn tier_floor_enforced() {
        let (lat, lng) = roodepoort_za();
        let nonce = fresh_nonce(11);
        let result = prove_jurisdiction(&JurisdictionProofRequest {
            lat_degrees: lat,
            lng_degrees: lng,
            jurisdiction_box: sa_box(),
            attester_pubkey_hash: attester(),
            timestamp_unix: 1_713_400_000,
            verifier_nonce: nonce,
            attestation_tier: AttestationTier::T0SelfAttested,
        })
        .expect("prove");

        let err = verify_jurisdiction(&JurisdictionVerifyInput {
            result: &result,
            jurisdiction_box: &sa_box(),
            attester_pubkey_hash: &attester(),
            timestamp_unix: 1_713_400_000,
            expected_verifier_nonce: &nonce,
            required_tier_minimum: AttestationTier::T2CivicBridge,
        })
        .unwrap_err();
        assert!(err.contains("does not meet required minimum"));
    }

    #[test]
    fn wrong_nonce_rejected() {
        let (lat, lng) = roodepoort_za();
        let prover_nonce = fresh_nonce(13);
        let result = prove_jurisdiction(&JurisdictionProofRequest {
            lat_degrees: lat,
            lng_degrees: lng,
            jurisdiction_box: sa_box(),
            attester_pubkey_hash: attester(),
            timestamp_unix: 1_713_400_000,
            verifier_nonce: prover_nonce,
            attestation_tier: AttestationTier::T1PhoneGps,
        })
        .expect("prove");

        let different_nonce = fresh_nonce(42);
        let err = verify_jurisdiction(&JurisdictionVerifyInput {
            result: &result,
            jurisdiction_box: &sa_box(),
            attester_pubkey_hash: &attester(),
            timestamp_unix: 1_713_400_000,
            expected_verifier_nonce: &different_nonce,
            required_tier_minimum: AttestationTier::default_minimum(),
        })
        .unwrap_err();
        assert!(err.contains("nonce"));
    }

    #[test]
    fn distinct_nonces_yield_distinct_proofs() {
        // Unlinkability smoke test — two proofs from the same location
        // with different nonces must have distinct commitments.
        let (lat, lng) = roodepoort_za();
        let one = prove_jurisdiction(&JurisdictionProofRequest {
            lat_degrees: lat,
            lng_degrees: lng,
            jurisdiction_box: sa_box(),
            attester_pubkey_hash: attester(),
            timestamp_unix: 1_713_400_000,
            verifier_nonce: fresh_nonce(1),
            attestation_tier: AttestationTier::T1PhoneGps,
        })
        .expect("prove 1");
        let two = prove_jurisdiction(&JurisdictionProofRequest {
            lat_degrees: lat,
            lng_degrees: lng,
            jurisdiction_box: sa_box(),
            attester_pubkey_hash: attester(),
            timestamp_unix: 1_713_400_000,
            verifier_nonce: fresh_nonce(2),
            attestation_tier: AttestationTier::T1PhoneGps,
        })
        .expect("prove 2");

        assert_ne!(one.location_commitment, two.location_commitment);
        assert_ne!(one.nonce_hash, two.nonce_hash);
    }

    #[test]
    fn tier_meets_monotonic() {
        use AttestationTier::*;
        assert!(T4Notary.meets(T0SelfAttested));
        assert!(T2CivicBridge.meets(T1PhoneGps));
        assert!(!T0SelfAttested.meets(T1PhoneGps));
        assert!(T1PhoneGps.meets(T1PhoneGps));
    }

    #[test]
    fn biased_microdegrees_round_trip() {
        let b = biased_microdegrees(-26.1625);
        assert_eq!(b, (COORD_BIAS + (-26_162_500)) as u64);
        let b2 = biased_microdegrees(27.8725);
        assert_eq!(b2, (COORD_BIAS + 27_872_500) as u64);
    }

    #[test]
    fn invalid_box_rejected() {
        assert!(JurisdictionBox::from_degrees("x", 91.0, 92.0, 0.0, 1.0).is_none());
        assert!(JurisdictionBox::from_degrees("x", 0.0, 1.0, -181.0, 0.0).is_none());
        assert!(JurisdictionBox::from_degrees("x", 10.0, 5.0, 0.0, 1.0).is_none());
    }
}
