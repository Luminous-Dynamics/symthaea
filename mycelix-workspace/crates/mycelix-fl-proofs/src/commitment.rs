// Ported from fl-aggregator/src/proofs/gradient_circuit.rs (commitment portion)
//! Blake3 commitments for gradient vectors.
//!
//! Provides deterministic commitments to gradient data that can be used
//! as public inputs to zkSTARK proofs.

use crate::types::Commitment;

/// Compute a Blake3 commitment to a gradient vector.
///
/// The commitment covers the raw byte representation of the f32 values,
/// ensuring any modification to the gradient is detectable.
pub fn commit_gradient(gradient: &[f32]) -> Commitment {
    let mut hasher = blake3::Hasher::new();
    for &value in gradient {
        hasher.update(&value.to_le_bytes());
    }
    let hash = hasher.finalize();
    Commitment::new(*hash.as_bytes())
}

/// Compute a Blake3 commitment to gradient data with round metadata.
///
/// Includes the round number in the commitment to prevent cross-round replay.
pub fn commit_gradient_with_round(gradient: &[f32], round: u32) -> Commitment {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&round.to_le_bytes());
    for &value in gradient {
        hasher.update(&value.to_le_bytes());
    }
    let hash = hasher.finalize();
    Commitment::new(*hash.as_bytes())
}

/// Verify a gradient commitment matches the expected data.
pub fn verify_commitment(gradient: &[f32], expected: &Commitment) -> bool {
    let actual = commit_gradient(gradient);
    actual == *expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_deterministic() {
        let gradient = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let c1 = commit_gradient(&gradient);
        let c2 = commit_gradient(&gradient);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_commit_different_gradients() {
        let g1 = vec![0.1, 0.2, 0.3];
        let g2 = vec![0.1, 0.2, 0.4]; // Differs in last element
        assert_ne!(commit_gradient(&g1), commit_gradient(&g2));
    }

    #[test]
    fn test_commit_with_round() {
        let gradient = vec![1.0, 2.0, 3.0];
        let c1 = commit_gradient_with_round(&gradient, 1);
        let c2 = commit_gradient_with_round(&gradient, 2);
        // Same gradient, different rounds → different commitments
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_verify_commitment() {
        let gradient = vec![0.5, -0.3, 0.8];
        let commitment = commit_gradient(&gradient);
        assert!(verify_commitment(&gradient, &commitment));

        let wrong_gradient = vec![0.5, -0.3, 0.9];
        assert!(!verify_commitment(&wrong_gradient, &commitment));
    }

    #[test]
    fn test_empty_gradient_commitment() {
        let c1 = commit_gradient(&[]);
        let c2 = commit_gradient(&[]);
        assert_eq!(c1, c2);
    }
}
