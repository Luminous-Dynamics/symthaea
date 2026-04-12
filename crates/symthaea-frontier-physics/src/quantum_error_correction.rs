// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quantum Error Correction — how biology might protect quantum coherence.
//!
//! If consciousness involves quantum coherence in warm biological systems,
//! some form of error correction or decoherence-free subspace must exist.
//!
//! References:
//! - Shor, P. W. (1995). Phys. Rev. A 52, R2493.
//! - Zanardi, P. & Rasetti, M. (1997). Phys. Rev. Lett. 79, 3306 (DFS).
//! - Lidar, D. A. (2014). Quantum Error Correction. Cambridge UP.

/// Quantum error syndrome for a 3-qubit bit-flip code.
///
/// The code encodes |0⟩ → |000⟩, |1⟩ → |111⟩.
/// Given a noisy 3-bit state, determines which qubit (if any) flipped.
/// Returns: (corrected_bit, error_syndrome)
pub fn three_qubit_bit_flip_syndrome(bits: [bool; 3]) -> (bool, usize) {
    // Syndrome: XOR of pairs
    let s1 = bits[0] ^ bits[1]; // First two qubits disagree?
    let s2 = bits[1] ^ bits[2]; // Last two qubits disagree?

    let error_qubit = match (s1, s2) {
        (false, false) => 0, // No error
        (true, false) => 1,  // Qubit 0 flipped
        (true, true) => 2,   // Qubit 1 flipped
        (false, true) => 3,  // Qubit 2 flipped
    };

    // Majority vote for the logical bit value
    let ones = bits.iter().filter(|&&b| b).count();
    let logical = ones >= 2;

    (logical, error_qubit)
}

/// Decoherence-free subspace (DFS) dimension for collective dephasing.
///
/// For n qubits under collective dephasing (H_error = Σ_i σ_z^i),
/// the DFS consists of states with total spin projection m = 0.
/// dim(DFS) = C(n, n/2) for even n.
pub fn dfs_dimension_collective_dephasing(n_qubits: usize) -> usize {
    if n_qubits % 2 != 0 {
        return 0; // Odd number of qubits has no m=0 subspace in strict sense
    }
    // C(n, n/2) = n! / ((n/2)!)^2
    binomial_exact(n_qubits, n_qubits / 2)
}

fn binomial_exact(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Quantum channel fidelity: F = ⟨ψ|ε(|ψ⟩⟨ψ|)|ψ⟩
/// For a depolarizing channel: F = 1 - p + p/d
/// where p is the error probability and d is the Hilbert space dimension.
pub fn depolarizing_channel_fidelity(error_prob: f64, dimension: usize) -> f64 {
    1.0 - error_prob + error_prob / dimension as f64
}

/// Threshold theorem: the error rate below which quantum error correction succeeds.
/// For the surface code: p_threshold ≈ 1.1%
pub const SURFACE_CODE_THRESHOLD: f64 = 0.011;

/// Number of physical qubits needed for one logical qubit at distance d.
/// Surface code: n_physical = d² (for one logical qubit at distance d).
pub fn surface_code_physical_qubits(distance: usize) -> usize {
    distance * distance
}

/// Logical error rate for surface code: p_L ≈ (p/p_th)^{(d+1)/2}
pub fn surface_code_logical_error_rate(physical_error_rate: f64, distance: usize) -> f64 {
    (physical_error_rate / SURFACE_CODE_THRESHOLD).powf((distance as f64 + 1.0) / 2.0)
}

/// Quantum Hamming bound: the maximum number of errors a code can correct.
/// [[n, k, d]] code corrects t = ⌊(d-1)/2⌋ errors.
pub fn correctable_errors(distance: usize) -> usize {
    (distance - 1) / 2
}

/// Knill-Laflamme condition: necessary and sufficient for error correction.
/// ⟨ψ_i|E_a†E_b|ψ_j⟩ = C_ab δ_ij for all code words |ψ_i⟩, |ψ_j⟩ and errors E_a, E_b.
/// Returns true if the condition is satisfied (simplified check).
pub fn knill_laflamme_check(overlap_matrix: &[f64], n_codewords: usize, n_errors: usize) -> bool {
    // Check that off-diagonal blocks are zero
    for a in 0..n_errors {
        for i in 0..n_codewords {
            for j in 0..n_codewords {
                if i != j {
                    let idx = a * n_codewords * n_codewords + i * n_codewords + j;
                    if idx < overlap_matrix.len() && overlap_matrix[idx].abs() > 1e-10 {
                        return false;
                    }
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_flip_no_error() {
        let (logical, syndrome) = three_qubit_bit_flip_syndrome([false, false, false]);
        assert_eq!(logical, false);
        assert_eq!(syndrome, 0); // No error
    }

    #[test]
    fn test_bit_flip_single_error() {
        // |000⟩ with qubit 1 flipped → |010⟩
        let (logical, syndrome) = three_qubit_bit_flip_syndrome([false, true, false]);
        assert_eq!(logical, false); // Majority says 0
        assert_eq!(syndrome, 2); // Error on qubit 1
    }

    #[test]
    fn test_bit_flip_encodes_one() {
        let (logical, syndrome) = three_qubit_bit_flip_syndrome([true, true, true]);
        assert_eq!(logical, true);
        assert_eq!(syndrome, 0);
    }

    #[test]
    fn test_dfs_dimension() {
        // 2 qubits: DFS dim = C(2,1) = 2 (singlet + triplet m=0)
        assert_eq!(dfs_dimension_collective_dephasing(2), 2);
        // 4 qubits: C(4,2) = 6
        assert_eq!(dfs_dimension_collective_dephasing(4), 6);
    }

    #[test]
    fn test_depolarizing_fidelity() {
        // No error: F = 1
        assert!((depolarizing_channel_fidelity(0.0, 2) - 1.0).abs() < 1e-14);
        // Full depolarization: F = 1/d
        let f = depolarizing_channel_fidelity(1.0, 2);
        assert!((f - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_surface_code_scaling() {
        // Higher distance → exponentially lower error rate
        let p = 0.005; // Below threshold
        let e3 = surface_code_logical_error_rate(p, 3);
        let e5 = surface_code_logical_error_rate(p, 5);
        assert!(
            e5 < e3,
            "Distance 5 should have lower error: e3={:.2e}, e5={:.2e}",
            e3,
            e5
        );
    }

    #[test]
    fn test_correctable_errors_distance() {
        assert_eq!(correctable_errors(3), 1);
        assert_eq!(correctable_errors(5), 2);
        assert_eq!(correctable_errors(7), 3);
    }
}
