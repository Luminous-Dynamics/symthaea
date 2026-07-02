// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared AIR (Algebraic Intermediate Representation) for Mycelix Attribution ZK-STARK proofs.
//!
//! This crate defines the common AIR used by both the prover and verifier,
//! preventing divergence between proof generation and verification.
//!
//! ## Trace Layout
//!
//! 9 columns × 16 rows (sized for S128 FRI domain requirements):
//! - Column 0: `scale` — usage scale (1=Small, 2=Medium, 3=Large, 4=Enterprise)
//! - Column 1: `dep_hash_lo` — lower 64 bits of Blake3(dependency_id)
//! - Column 2: `dep_hash_hi` — upper 64 bits of Blake3(dependency_id)
//! - Column 3: `did_hash_lo` — lower 64 bits of Blake3(user_did)
//! - Column 4: `did_hash_hi` — upper 64 bits of Blake3(user_did)
//! - Columns 5–8: `witness_commitment[0..3]` — 4 u64 LE limbs of Blake3 commitment
//!
//! ## Constraints
//!
//! - C0: `(scale-1)(scale-2)(scale-3)(scale-4) == 0` — range check (degree 4)
//! - C1–C8: `col[next] - col[curr] == 0` — constant columns
//!
//! ## Public Inputs
//!
//! 8 field elements: dep_hash (2) + did_hash (2) + witness_commitment (4)

use winter_math::fields::f128::BaseElement;
use winter_math::FieldElement;
use winterfell::{
    math::ToElements, Air, AirContext, Assertion, EvaluationFrame, ProofOptions, TraceInfo,
    TransitionConstraintDegree,
};

pub const TRACE_WIDTH: usize = 9;
pub const TRACE_LENGTH: usize = 16;

// ── Public Inputs ───────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PublicInputs {
    /// dep_hash lower 64 bits
    pub dep_hash_lo: u64,
    /// dep_hash upper 64 bits
    pub dep_hash_hi: u64,
    /// did_hash lower 64 bits
    pub did_hash_lo: u64,
    /// did_hash upper 64 bits
    pub did_hash_hi: u64,
    /// witness_commitment as 4×u64 LE limbs
    pub witness_commitment: [u64; 4],
}

impl ToElements<BaseElement> for PublicInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        vec![
            BaseElement::from(self.dep_hash_lo),
            BaseElement::from(self.dep_hash_hi),
            BaseElement::from(self.did_hash_lo),
            BaseElement::from(self.did_hash_hi),
            BaseElement::from(self.witness_commitment[0]),
            BaseElement::from(self.witness_commitment[1]),
            BaseElement::from(self.witness_commitment[2]),
            BaseElement::from(self.witness_commitment[3]),
        ]
    }
}

/// Wrapper for Winterfell's verify() which expects a reference.
pub struct PublicInputsWrapper(pub PublicInputs);

impl ToElements<BaseElement> for PublicInputsWrapper {
    fn to_elements(&self) -> Vec<BaseElement> {
        self.0.to_elements()
    }
}

// ── Helper Functions ────────────────────────────────────────────────

/// Hash arbitrary bytes to a pair of u64 values (lower, upper).
/// Uses first 16 bytes of Blake3 hash as 2 little-endian u64.
pub fn hash_to_u64_pair(data: &[u8]) -> (u64, u64) {
    let hash = blake3::hash(data);
    let bytes = hash.as_bytes();
    let lo = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let hi = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    (lo, hi)
}

/// Compute the witness commitment: blake3(scale_byte ++ dep_id_bytes ++ user_did_bytes ++ org_bytes)
pub fn compute_witness_commitment(
    scale: u8,
    dep_id: &str,
    user_did: &str,
    organization: &str,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[scale]);
    hasher.update(dep_id.as_bytes());
    hasher.update(user_did.as_bytes());
    hasher.update(organization.as_bytes());
    *hasher.finalize().as_bytes()
}

/// Convert a 32-byte hash to 4 u64 LE limbs.
pub fn commitment_to_limbs(commitment: &[u8; 32]) -> [u64; 4] {
    [
        u64::from_le_bytes(commitment[0..8].try_into().unwrap()),
        u64::from_le_bytes(commitment[8..16].try_into().unwrap()),
        u64::from_le_bytes(commitment[16..24].try_into().unwrap()),
        u64::from_le_bytes(commitment[24..32].try_into().unwrap()),
    ]
}

/// Map usage scale string to u8.
pub fn scale_to_u8(scale: &str) -> Option<u8> {
    match scale.to_lowercase().as_str() {
        "small" => Some(1),
        "medium" => Some(2),
        "large" => Some(3),
        "enterprise" => Some(4),
        _ => None,
    }
}

// ── AIR Definition ──────────────────────────────────────────────────

pub struct UsageAttestationAir {
    context: AirContext<BaseElement>,
    pub_inputs: PublicInputs,
}

impl Air for UsageAttestationAir {
    type BaseField = BaseElement;
    type PublicInputs = PublicInputs;

    fn new(trace_info: TraceInfo, pub_inputs: PublicInputs, options: ProofOptions) -> Self {
        // 9 transition constraints:
        // C0: degree 4 (range check on scale)
        // C1-C8: degree 1 (constant columns: hashes + commitment)
        let degrees = vec![
            TransitionConstraintDegree::new(4), // (scale-1)(scale-2)(scale-3)(scale-4)
            TransitionConstraintDegree::new(1), // dep_hash_lo constant
            TransitionConstraintDegree::new(1), // dep_hash_hi constant
            TransitionConstraintDegree::new(1), // did_hash_lo constant
            TransitionConstraintDegree::new(1), // did_hash_hi constant
            TransitionConstraintDegree::new(1), // witness_commitment[0] constant
            TransitionConstraintDegree::new(1), // witness_commitment[1] constant
            TransitionConstraintDegree::new(1), // witness_commitment[2] constant
            TransitionConstraintDegree::new(1), // witness_commitment[3] constant
        ];

        let num_assertions = 8; // boundary assertions at row 0 (columns 1-8)

        UsageAttestationAir {
            context: AirContext::new(trace_info, degrees, num_assertions, options),
            pub_inputs,
        }
    }

    fn context(&self) -> &AirContext<BaseElement> {
        &self.context
    }

    fn evaluate_transition<E: FieldElement + From<BaseElement>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();

        let one = E::ONE;
        let two = one + one;
        let three = two + one;
        let four = three + one;

        let scale = current[0];

        // C0: (scale-1)(scale-2)(scale-3)(scale-4) == 0
        result[0] = (scale - one) * (scale - two) * (scale - three) * (scale - four);

        // C1-C8: columns are constant across rows
        result[1] = next[1] - current[1]; // dep_hash_lo
        result[2] = next[2] - current[2]; // dep_hash_hi
        result[3] = next[3] - current[3]; // did_hash_lo
        result[4] = next[4] - current[4]; // did_hash_hi
        result[5] = next[5] - current[5]; // witness_commitment[0]
        result[6] = next[6] - current[6]; // witness_commitment[1]
        result[7] = next[7] - current[7]; // witness_commitment[2]
        result[8] = next[8] - current[8]; // witness_commitment[3]
    }

    fn get_assertions(&self) -> Vec<Assertion<BaseElement>> {
        let pi = &self.pub_inputs;
        vec![
            // Boundary assertions at row 0: columns 1-8 match public inputs.
            // Column 0 (scale) has NO boundary assertion — it is a private witness
            // proven valid by the degree-4 range check constraint C0.
            // Columns 5-8 bind the witness_commitment (which includes the scale
            // in its blake3 preimage) to the proof, preventing commitment swaps.
            Assertion::single(1, 0, BaseElement::from(pi.dep_hash_lo)),
            Assertion::single(2, 0, BaseElement::from(pi.dep_hash_hi)),
            Assertion::single(3, 0, BaseElement::from(pi.did_hash_lo)),
            Assertion::single(4, 0, BaseElement::from(pi.did_hash_hi)),
            Assertion::single(5, 0, BaseElement::from(pi.witness_commitment[0])),
            Assertion::single(6, 0, BaseElement::from(pi.witness_commitment[1])),
            Assertion::single(7, 0, BaseElement::from(pi.witness_commitment[2])),
            Assertion::single(8, 0, BaseElement::from(pi.witness_commitment[3])),
        ]
    }
}

/// Create standard proof options for S128 security.
///
/// 128 queries + grinding_factor=20 + blowup=16 provides ≥128-bit security.
pub fn default_proof_options() -> ProofOptions {
    ProofOptions::new(
        128, // num_queries (S128)
        16,  // blowup_factor
        20,  // grinding_factor (2^20 PoW)
        winterfell::FieldExtension::None,
        8,                                  // fri_folding_factor
        31,                                 // fri_max_remainder_poly_degree
        winterfell::BatchingMethod::Linear, // constraint batching
        winterfell::BatchingMethod::Linear, // deep poly batching
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_to_u64_pair() {
        let (lo, hi) = hash_to_u64_pair(b"crate:serde:1.0");
        assert_ne!(lo, 0);
        assert_ne!(hi, 0);
        // Deterministic
        let (lo2, hi2) = hash_to_u64_pair(b"crate:serde:1.0");
        assert_eq!(lo, lo2);
        assert_eq!(hi, hi2);
    }

    #[test]
    fn test_compute_witness_commitment() {
        let c1 = compute_witness_commitment(2, "crate:serde:1.0", "did:mycelix:abc", "Acme");
        let c2 = compute_witness_commitment(2, "crate:serde:1.0", "did:mycelix:abc", "Acme");
        assert_eq!(c1, c2); // deterministic

        let c3 = compute_witness_commitment(3, "crate:serde:1.0", "did:mycelix:abc", "Acme");
        assert_ne!(c1, c3); // different scale = different commitment
    }

    #[test]
    fn test_scale_to_u8() {
        assert_eq!(scale_to_u8("small"), Some(1));
        assert_eq!(scale_to_u8("Medium"), Some(2));
        assert_eq!(scale_to_u8("LARGE"), Some(3));
        assert_eq!(scale_to_u8("enterprise"), Some(4));
        assert_eq!(scale_to_u8("huge"), None);
    }

    #[test]
    fn test_commitment_to_limbs() {
        let commitment = [0u8; 32];
        let limbs = commitment_to_limbs(&commitment);
        assert_eq!(limbs, [0, 0, 0, 0]);

        let mut commitment2 = [0u8; 32];
        commitment2[0] = 1;
        let limbs2 = commitment_to_limbs(&commitment2);
        assert_eq!(limbs2[0], 1);
    }

    #[test]
    fn test_public_inputs_to_elements() {
        let pi = PublicInputs {
            dep_hash_lo: 42,
            dep_hash_hi: 43,
            did_hash_lo: 44,
            did_hash_hi: 45,
            witness_commitment: [1, 2, 3, 4],
        };
        let elements = pi.to_elements();
        assert_eq!(elements.len(), 8);
    }
}
