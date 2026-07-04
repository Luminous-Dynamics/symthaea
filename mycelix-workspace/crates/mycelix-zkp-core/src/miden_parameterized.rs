// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Parameterized Miden ZK-STARK — Model-Agnostic Governance Proofs
//!
//! Proves that a citizen's weighted civic score (after decay) exceeds
//! a governance threshold — without revealing ANY dimension values.
//!
//! Unlike the simple threshold circuit in `miden_consciousness.rs`, this
//! circuit takes the scoring model's weights as PUBLIC inputs, making it
//! work with ANY [`ScoringModel`](mycelix_bridge_common::scoring_model).
//!
//! ## Fixed-Point Arithmetic
//!
//! All values are scaled to permille (0–1000) to avoid floating-point
//! operations inside the STARK circuit:
//! - Weight 0.25 → 250 permille
//! - Dimension 0.8 → 800 permille
//! - Threshold 0.4 → 400 permille
//! - Decay multiplier 0.94 → 940 permille
//!
//! ## Pre-Computed Decay
//!
//! The decay multiplier `M = e^(-λΔt)` is computed OUTSIDE the circuit
//! and passed as a private input. This avoids expensive in-circuit
//! exponentiation while preserving zero-knowledge (the verifier never
//! learns M, only that the decayed score passes the threshold).
//!
//! ## Commitment
//!
//! The score commitment includes the model_id to prevent cross-model
//! proof reuse: `SHA-256(model_id || dims_concat || decay_m || domain_tag)`

use crate::error::ZkpError;
use sha2::{Digest, Sha256};

pub type ParamResult<T> = Result<T, ZkpError>;

/// Maximum number of dimensions the circuit supports.
pub const MAX_CIRCUIT_DIMENSIONS: usize = 16;

/// Fixed-point scale factor: 1.0 = 1000 permille.
pub const PERMILLE_SCALE: u64 = 1000;

// ============================================================================
// Fixed-Point Conversion
// ============================================================================

/// Convert a float [0.0, 1.0] to permille [0, 1000].
pub fn to_permille(value: f64) -> u32 {
    if !value.is_finite() {
        return 0;
    }
    (value.clamp(0.0, 1.0) * PERMILLE_SCALE as f64).round() as u32
}

/// Convert permille [0, 1000] back to float [0.0, 1.0].
pub fn from_permille(permille: u32) -> f64 {
    permille.min(1000) as f64 / PERMILLE_SCALE as f64
}

// ============================================================================
// Proof Types
// ============================================================================

/// Input for generating a parameterized governance proof.
#[derive(Debug, Clone)]
pub struct ParameterizedProofInput {
    /// Scoring model identifier (included in commitment).
    pub model_id: String,
    /// Per-dimension weights in permille (PUBLIC — from community config).
    /// Must sum to 1000.
    pub weights_permille: Vec<u32>,
    /// Per-dimension values in permille (PRIVATE — the citizen's scores).
    pub dimensions_permille: Vec<u32>,
    /// Pre-computed decay multiplier in permille (PRIVATE).
    /// M = to_permille(e^(-λΔt))
    pub decay_multiplier_permille: u32,
    /// Governance threshold in permille (PUBLIC).
    pub threshold_permille: u32,
}

/// Result of generating a parameterized proof.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParameterizedProof {
    /// STARK proof bytes.
    pub proof_bytes: Vec<u8>,
    /// Commitment: SHA-256(model_id || dimensions || decay || domain_tag).
    pub commitment: [u8; 32],
    /// Model identifier (public — verifier checks this matches community config).
    pub model_id: String,
    /// Number of dimensions in this proof.
    pub n_dims: usize,
    /// The threshold that was proven against (public).
    pub threshold_permille: u32,
    /// Stack outputs from the Miden program.
    pub stack_outputs: Vec<u64>,
}

// ============================================================================
// Commitment
// ============================================================================

/// Compute the proof commitment binding the proof to specific inputs.
///
/// Includes model_id to prevent cross-model proof reuse.
pub fn compute_parameterized_commitment(
    model_id: &str,
    dimensions_permille: &[u32],
    decay_multiplier_permille: u32,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(model_id.as_bytes());
    for &d in dimensions_permille {
        hasher.update(d.to_le_bytes());
    }
    hasher.update(decay_multiplier_permille.to_le_bytes());
    hasher.update(b"MYCELIX:ParameterizedGovProof:v1");
    let result = hasher.finalize();
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(&result);
    commitment
}

// ============================================================================
// Circuit Math (pure Rust — mirrors what the Miden program computes)
// ============================================================================

/// Compute the weighted score in permille-squared units.
///
/// result = sum(w_i × d_i) for all dimensions
///
/// In permille: if w=250 and d=800, contribution = 200,000.
/// Sum of all contributions for weights summing to 1000 and dims ≤ 1000
/// gives max = 1,000,000 (= 1.0 × 1.0 in permille²).
pub fn weighted_sum_permille_sq(weights: &[u32], dimensions: &[u32]) -> u64 {
    weights
        .iter()
        .zip(dimensions.iter())
        .map(|(&w, &d)| w as u64 * d as u64)
        .sum()
}

/// Apply decay: score_sq × M_permille.
///
/// Result is in permille³ units.  The threshold comparison is:
/// score_sq × M >= threshold × PERMILLE_SCALE²
pub fn apply_decay_permille(score_sq: u64, decay_m_permille: u32) -> u64 {
    score_sq * decay_m_permille as u64
}

/// Check if decayed score passes threshold.
///
/// Compares: (sum(w_i × d_i) × M) >= (threshold × 1000 × 1000)
/// All in integer arithmetic, no division needed.
pub fn passes_threshold(
    weights: &[u32],
    dimensions: &[u32],
    decay_m_permille: u32,
    threshold_permille: u32,
) -> bool {
    let score_sq = weighted_sum_permille_sq(weights, dimensions);
    let decayed = apply_decay_permille(score_sq, decay_m_permille);
    let threshold_scaled = threshold_permille as u64 * PERMILLE_SCALE * PERMILLE_SCALE;
    decayed >= threshold_scaled
}

// ============================================================================
// Structural Validation
// ============================================================================

/// Validate proof input before proving.
pub fn validate_input(input: &ParameterizedProofInput) -> ParamResult<()> {
    if input.weights_permille.is_empty() || input.dimensions_permille.is_empty() {
        return Err(ZkpError::ProvingError("Empty weights or dimensions".into()));
    }
    if input.weights_permille.len() != input.dimensions_permille.len() {
        return Err(ZkpError::ProvingError(format!(
            "Weight count {} != dimension count {}",
            input.weights_permille.len(),
            input.dimensions_permille.len()
        )));
    }
    if input.weights_permille.len() > MAX_CIRCUIT_DIMENSIONS {
        return Err(ZkpError::ProvingError(format!(
            "Too many dimensions: {} > {}",
            input.weights_permille.len(),
            MAX_CIRCUIT_DIMENSIONS
        )));
    }

    let weight_sum: u32 = input.weights_permille.iter().sum();
    if weight_sum != 1000 {
        return Err(ZkpError::ProvingError(format!(
            "Weights must sum to 1000 permille, got {}",
            weight_sum
        )));
    }

    for (i, &w) in input.weights_permille.iter().enumerate() {
        if w > 500 {
            return Err(ZkpError::ProvingError(format!(
                "Weight[{}] = {} > 500 (50% max)",
                i, w
            )));
        }
    }

    for (i, &d) in input.dimensions_permille.iter().enumerate() {
        if d > 1000 {
            return Err(ZkpError::ProvingError(format!(
                "Dimension[{}] = {} > 1000",
                i, d
            )));
        }
    }

    if input.decay_multiplier_permille > 1000 {
        return Err(ZkpError::ProvingError(format!(
            "Decay multiplier {} > 1000",
            input.decay_multiplier_permille
        )));
    }

    if input.threshold_permille > 1000 {
        return Err(ZkpError::ProvingError(format!(
            "Threshold {} > 1000",
            input.threshold_permille
        )));
    }

    // Pre-check: would this proof succeed?
    if !passes_threshold(
        &input.weights_permille,
        &input.dimensions_permille,
        input.decay_multiplier_permille,
        input.threshold_permille,
    ) {
        return Err(ZkpError::ProvingError(
            "Decayed score below threshold — cannot prove false statement".into(),
        ));
    }

    Ok(())
}

/// Validate a proof structurally (without Miden — runs in WASM).
pub fn validate_proof_structure(proof: &ParameterizedProof) -> ParamResult<()> {
    if proof.proof_bytes.is_empty() {
        return Err(ZkpError::ProvingError("Empty proof bytes".into()));
    }
    if proof.proof_bytes.len() > 200_000 {
        return Err(ZkpError::ProvingError("Proof exceeds 200KB".into()));
    }
    if proof.commitment == [0u8; 32] {
        return Err(ZkpError::ProvingError("Zero commitment".into()));
    }
    if proof.model_id.is_empty() {
        return Err(ZkpError::ProvingError("Empty model_id".into()));
    }
    if proof.n_dims == 0 || proof.n_dims > MAX_CIRCUIT_DIMENSIONS {
        return Err(ZkpError::ProvingError(format!(
            "Invalid dimension count: {}",
            proof.n_dims
        )));
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Fixed-point conversion ----

    #[test]
    fn to_permille_basic() {
        assert_eq!(to_permille(0.0), 0);
        assert_eq!(to_permille(0.25), 250);
        assert_eq!(to_permille(0.5), 500);
        assert_eq!(to_permille(1.0), 1000);
    }

    #[test]
    fn to_permille_clamps() {
        assert_eq!(to_permille(1.5), 1000);
        assert_eq!(to_permille(-0.5), 0);
    }

    #[test]
    fn to_permille_nan() {
        assert_eq!(to_permille(f64::NAN), 0);
    }

    #[test]
    fn from_permille_roundtrip() {
        for v in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] {
            let p = to_permille(v);
            let back = from_permille(p);
            assert!(
                (back - v).abs() < 0.002,
                "Roundtrip failed: {} -> {} -> {}",
                v,
                p,
                back
            );
        }
    }

    // ---- Weighted sum ----

    #[test]
    fn weighted_sum_canonical_4d() {
        // I=0.8(800), R=0.6(600), C=0.7(700), E=0.5(500)
        // W: 250, 250, 300, 200
        let w = [250, 250, 300, 200];
        let d = [800, 600, 700, 500];
        let sum = weighted_sum_permille_sq(&w, &d);
        // 250*800 + 250*600 + 300*700 + 200*500 = 200000+150000+210000+100000 = 660000
        assert_eq!(sum, 660_000);
        // This equals 0.66 in original scale: 660000 / 1000000 = 0.66 ✓
    }

    #[test]
    fn weighted_sum_max() {
        // All weights 250 (4D), all dims 1000
        let w = [250, 250, 250, 250];
        let d = [1000, 1000, 1000, 1000];
        let sum = weighted_sum_permille_sq(&w, &d);
        assert_eq!(sum, 1_000_000); // = 1.0 × 1.0
    }

    #[test]
    fn weighted_sum_zero() {
        let w = [250, 250, 250, 250];
        let d = [0, 0, 0, 0];
        assert_eq!(weighted_sum_permille_sq(&w, &d), 0);
    }

    // ---- Decay ----

    #[test]
    fn apply_decay_no_decay() {
        let score_sq = 660_000u64;
        let decayed = apply_decay_permille(score_sq, 1000); // M=1.0
        assert_eq!(decayed, 660_000_000); // score_sq * 1000
    }

    #[test]
    fn apply_decay_half() {
        let score_sq = 1_000_000u64;
        let decayed = apply_decay_permille(score_sq, 500); // M=0.5
        assert_eq!(decayed, 500_000_000);
    }

    // ---- Threshold check ----

    #[test]
    fn passes_threshold_citizen() {
        // Score 0.66, no decay, threshold 0.4 → should pass
        let w = [250, 250, 300, 200];
        let d = [800, 600, 700, 500];
        assert!(passes_threshold(&w, &d, 1000, 400));
    }

    #[test]
    fn fails_threshold_when_low() {
        // Score 0.20, no decay, threshold 0.4 → should fail
        let w = [250, 250, 300, 200];
        let d = [200, 200, 200, 200];
        assert!(!passes_threshold(&w, &d, 1000, 400));
    }

    #[test]
    fn passes_threshold_with_decay() {
        // Score 1.0, decay M=0.5, threshold 0.4 → 0.5 >= 0.4 → pass
        let w = [250, 250, 250, 250];
        let d = [1000, 1000, 1000, 1000];
        assert!(passes_threshold(&w, &d, 500, 400));
    }

    #[test]
    fn fails_threshold_after_heavy_decay() {
        // Score 1.0, decay M=0.3, threshold 0.4 → 0.3 < 0.4 → fail
        let w = [250, 250, 250, 250];
        let d = [1000, 1000, 1000, 1000];
        assert!(!passes_threshold(&w, &d, 300, 400));
    }

    #[test]
    fn passes_threshold_exact_boundary() {
        // Score 0.4, no decay, threshold 0.4 → should pass (>=)
        let w = [250, 250, 250, 250];
        let d = [400, 400, 400, 400];
        assert!(passes_threshold(&w, &d, 1000, 400));
    }

    #[test]
    fn passes_with_8d_sovereign_weights() {
        // Sovereign governance preset
        let w: Vec<u32> = [0.15, 0.10, 0.10, 0.12, 0.18, 0.13, 0.12, 0.10]
            .iter()
            .map(|&f| to_permille(f))
            .collect();
        let d = [800, 500, 600, 700, 900, 600, 700, 500];
        // Score ≈ 0.688
        assert!(passes_threshold(&w, &d, 1000, 400));
        assert!(passes_threshold(&w, &d, 1000, 600));
        assert!(!passes_threshold(&w, &d, 1000, 700));
    }

    // ---- Commitment ----

    #[test]
    fn commitment_deterministic() {
        let c1 = compute_parameterized_commitment("4d-v1", &[800, 600], 940);
        let c2 = compute_parameterized_commitment("4d-v1", &[800, 600], 940);
        assert_eq!(c1, c2);
    }

    #[test]
    fn commitment_differs_by_model_id() {
        let c1 = compute_parameterized_commitment("4d-v1", &[800, 600], 940);
        let c2 = compute_parameterized_commitment("8d-v1", &[800, 600], 940);
        assert_ne!(
            c1, c2,
            "Different model IDs should produce different commitments"
        );
    }

    #[test]
    fn commitment_differs_by_dimensions() {
        let c1 = compute_parameterized_commitment("4d-v1", &[800, 600], 940);
        let c2 = compute_parameterized_commitment("4d-v1", &[800, 700], 940);
        assert_ne!(c1, c2);
    }

    #[test]
    fn commitment_differs_by_decay() {
        let c1 = compute_parameterized_commitment("4d-v1", &[800, 600], 940);
        let c2 = compute_parameterized_commitment("4d-v1", &[800, 600], 500);
        assert_ne!(c1, c2);
    }

    // ---- Input validation ----

    #[test]
    fn valid_input_passes() {
        let input = ParameterizedProofInput {
            model_id: "canonical-4d-v1".into(),
            weights_permille: vec![250, 250, 300, 200],
            dimensions_permille: vec![800, 600, 700, 500],
            decay_multiplier_permille: 1000,
            threshold_permille: 400,
        };
        assert!(validate_input(&input).is_ok());
    }

    #[test]
    fn weights_not_summing_to_1000_rejected() {
        let input = ParameterizedProofInput {
            model_id: "test".into(),
            weights_permille: vec![300, 300, 300], // 900 != 1000
            dimensions_permille: vec![500, 500, 500],
            decay_multiplier_permille: 1000,
            threshold_permille: 400,
        };
        assert!(validate_input(&input).is_err());
    }

    #[test]
    fn weight_exceeding_500_rejected() {
        let input = ParameterizedProofInput {
            model_id: "test".into(),
            weights_permille: vec![600, 400], // 600 > 500
            dimensions_permille: vec![500, 500],
            decay_multiplier_permille: 1000,
            threshold_permille: 400,
        };
        assert!(validate_input(&input).is_err());
    }

    #[test]
    fn dimension_exceeding_1000_rejected() {
        let input = ParameterizedProofInput {
            model_id: "test".into(),
            weights_permille: vec![500, 500],
            dimensions_permille: vec![1001, 500],
            decay_multiplier_permille: 1000,
            threshold_permille: 400,
        };
        assert!(validate_input(&input).is_err());
    }

    #[test]
    fn mismatched_counts_rejected() {
        let input = ParameterizedProofInput {
            model_id: "test".into(),
            weights_permille: vec![500, 500],
            dimensions_permille: vec![500, 500, 500],
            decay_multiplier_permille: 1000,
            threshold_permille: 400,
        };
        assert!(validate_input(&input).is_err());
    }

    #[test]
    fn false_statement_rejected() {
        let input = ParameterizedProofInput {
            model_id: "test".into(),
            weights_permille: vec![500, 500],
            dimensions_permille: vec![100, 100], // score 0.1 < threshold 0.4
            decay_multiplier_permille: 1000,
            threshold_permille: 400,
        };
        assert!(validate_input(&input).is_err());
    }

    // ---- Proof structure validation ----

    #[test]
    fn valid_proof_structure_passes() {
        let proof = ParameterizedProof {
            proof_bytes: vec![1u8; 80_000],
            commitment: [0xAA; 32],
            model_id: "canonical-4d-v1".into(),
            n_dims: 4,
            threshold_permille: 400,
            stack_outputs: vec![1],
        };
        assert!(validate_proof_structure(&proof).is_ok());
    }

    #[test]
    fn empty_proof_bytes_rejected() {
        let proof = ParameterizedProof {
            proof_bytes: vec![],
            commitment: [0xAA; 32],
            model_id: "test".into(),
            n_dims: 4,
            threshold_permille: 400,
            stack_outputs: vec![],
        };
        assert!(validate_proof_structure(&proof).is_err());
    }

    #[test]
    fn zero_commitment_rejected() {
        let proof = ParameterizedProof {
            proof_bytes: vec![1],
            commitment: [0u8; 32],
            model_id: "test".into(),
            n_dims: 4,
            threshold_permille: 400,
            stack_outputs: vec![],
        };
        assert!(validate_proof_structure(&proof).is_err());
    }

    #[test]
    fn proof_serde_roundtrip() {
        let proof = ParameterizedProof {
            proof_bytes: vec![1, 2, 3],
            commitment: compute_parameterized_commitment("test", &[500], 1000),
            model_id: "canonical-4d-v1".into(),
            n_dims: 4,
            threshold_permille: 400,
            stack_outputs: vec![1],
        };
        let json = serde_json::to_string(&proof).unwrap();
        let back: ParameterizedProof = serde_json::from_str(&json).unwrap();
        assert_eq!(proof.proof_bytes, back.proof_bytes);
        assert_eq!(proof.commitment, back.commitment);
        assert_eq!(proof.model_id, back.model_id);
    }
}
