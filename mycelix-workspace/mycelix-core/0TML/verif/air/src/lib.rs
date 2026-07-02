// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// VSV-STARK: Verifiable PoGQ Decision Logic (v0)
//
// Proves per-round decision logic (EMA + warm-up + hysteresis + conformal)
// for Byzantine detection in federated learning.
//
// Architecture:
// - air.rs: AIR trait implementation (constraints)
// - prover.rs: Proof generation logic
// - verifier.rs: Standalone verification
// - public.rs: Public input parsing

mod air;
mod public;

pub use air::PoGQAir;
pub use public::{PublicInputs, PrivateWitness};

use winterfell::{
    ProofOptions, Prover, VerifierError,
    math::fields::f64::BaseElement,
    proof::StarkProof,
    TraceTable,
    crypto::hashers::Blake3_256,
};

/// Fixed-point scale factor: S = 2^16 = 65536
pub const SCALE: u64 = 65536;

/// Convert float to fixed-point
pub fn to_fixed(value: f64) -> u64 {
    (value * SCALE as f64).floor() as u64
}

/// Convert fixed-point to float
pub fn from_fixed(value_fp: u64) -> f64 {
    value_fp as f64 / SCALE as f64
}

/// Generate a STARK proof for one round of PoGQ decision logic
///
/// # Arguments
/// * `public` - Public inputs (commitments, params, previous state)
/// * `witness` - Private witness (hybrid score, flags)
///
/// # Returns
/// Serialized STARK proof bytes
pub fn generate_proof(
    public: &PublicInputs,
    witness: &PrivateWitness,
) -> Result<Vec<u8>, String> {
    // Validate inputs
    public.validate()?;
    witness.validate()?;

    // Build execution trace
    let trace = air::build_trace(public, witness)?;

    // Create prover
    let prover = PoGQProver::new(public.clone());

    // Generate proof
    let proof = prover.prove(trace).map_err(|e| format!("Proof generation failed: {:?}", e))?;

    // Serialize proof
    let mut bytes = Vec::new();
    proof.write_into(&mut bytes);

    Ok(bytes)
}

/// Verify a STARK proof
///
/// # Arguments
/// * `proof_bytes` - Serialized STARK proof
/// * `public` - Public inputs to verify against
///
/// # Returns
/// true if proof is valid, false otherwise
pub fn verify_proof(
    proof_bytes: &[u8],
    public: &PublicInputs,
) -> Result<bool, String> {
    // Deserialize proof
    let proof = StarkProof::from_bytes(proof_bytes)
        .map_err(|e| format!("Failed to deserialize proof: {:?}", e))?;

    // Verify
    let pub_inputs = public.to_field_elements();

    match winterfell::verify::<PoGQAir>(proof, pub_inputs) {
        Ok(security_level) => {
            // Check security level is acceptable (>= 96 bits for research)
            if security_level >= 96 {
                Ok(true)
            } else {
                Ok(false)
            }
        }
        Err(_) => Ok(false),
    }
}

/// Prover implementation for PoGQ AIR
struct PoGQProver {
    public: PublicInputs,
}

impl PoGQProver {
    fn new(public: PublicInputs) -> Self {
        Self { public }
    }
}

impl Prover for PoGQProver {
    type BaseField = BaseElement;
    type Air = PoGQAir;
    type Trace = TraceTable<Self::BaseField>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> Self::BaseField {
        // Pack public inputs into single field element
        // For now, return a placeholder (will use hash of inputs)
        BaseElement::new(0)
    }

    fn options(&self) -> ProofOptions {
        // Default proof options (32 queries, 8 blowup, 96-bit security)
        ProofOptions::new(
            32,  // num_queries
            8,   // blowup_factor
            0,   // grinding_factor
            winterfell::FieldExtension::None,
            4,   // fri_folding_factor
            31,  // fri_max_remainder_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_conversion() {
        assert_eq!(to_fixed(0.85), 55705);
        assert_eq!(to_fixed(0.90), 58982);
        assert_eq!(to_fixed(0.75), 49152);
        assert_eq!(to_fixed(1.00), 65536);

        assert!((from_fixed(55705) - 0.85).abs() < 1e-4);
        assert!((from_fixed(58982) - 0.90).abs() < 1e-4);
    }

    #[test]
    fn test_public_inputs_validation() {
        let public = PublicInputs {
            h_calib: "a".repeat(64),
            h_model: "b".repeat(64),
            h_grad: "c".repeat(64),
            beta_fp: 55705,
            w: 3,
            k: 2,
            m: 3,
            egregious_cap_fp: 65535,
            threshold_fp: 58982,
            ema_prev_fp: 49152,
            consec_viol_prev: 1,
            consec_clear_prev: 0,
            quarantined_prev: 0,
            current_round: 5,
            quarantine_out: 0,
        };

        assert!(public.validate().is_ok());
    }

    #[test]
    fn test_witness_validation() {
        let witness = PrivateWitness {
            x_t_fp: 52428,
            in_warmup: 0,
            violation_t: 1,
            release_t: 0,
        };

        assert!(witness.validate().is_ok());
    }
}
