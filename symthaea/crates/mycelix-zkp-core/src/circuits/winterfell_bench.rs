// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Measured Winterfell STARK baseline benchmark.
//!
//! Uses the REAL HealthRangeAir circuit (16-bit bit decomposition + verification)
//! to establish a measured per-bit proving cost, then extrapolates to 16,384-bit
//! HDC XOR binding.

use std::time::Instant;

use super::range_proof;

/// Measured Winterfell baseline for the paper.
pub struct WinterfellBaseline {
    /// Bits proven in this benchmark
    pub bits_proven: usize,
    /// Number of constraints (transition constraints × trace rows)
    pub transition_constraints: usize,
    /// Measured prover time in ms
    pub prove_time_ms: f64,
    /// Measured verifier time in ms
    pub verify_time_ms: f64,
    /// Measured proof size in bytes
    pub proof_size_bytes: usize,
    /// Extrapolated prover time for 16,384-bit XOR (ms)
    pub extrapolated_16k_prove_ms: f64,
    /// Extrapolated constraint count for 16,384-bit XOR
    pub extrapolated_16k_constraints: usize,
}

/// Run measured Winterfell benchmark.
///
/// Proves a 16-bit range proof (bit decomposition + verification),
/// times it precisely, and extrapolates to 16,384 bits.
pub fn bench_winterfell_baseline() -> WinterfellBaseline {
    use sha2::{Digest, Sha256};

    println!("\n=== WINTERFELL (MEASURED): Bit Decomposition Baseline ===\n");

    // Prove multiple range proofs to get stable timing
    let iterations = 10;
    let mut total_prove = std::time::Duration::ZERO;
    let mut total_verify = std::time::Duration::ZERO;
    let mut proof_size = 0;

    for i in 0..iterations {
        let value = 35u64 + i as u64; // Vary the value slightly
        let min = 10u64;
        let max = 100u64;
        let commit = {
            let h = Sha256::digest(value.to_le_bytes());
            let mut c = [0u8; 32];
            c.copy_from_slice(&h);
            c
        };

        let prove_start = Instant::now();
        let proof =
            range_proof::prove_range(value, min, max, commit).expect("Winterfell prove failed");
        total_prove += prove_start.elapsed();

        proof_size = proof.to_bytes().len();

        let verify_start = Instant::now();
        range_proof::verify_range(proof, min, max, commit).expect("Winterfell verify failed");
        total_verify += verify_start.elapsed();
    }

    let avg_prove_ms = total_prove.as_secs_f64() * 1000.0 / iterations as f64;
    let avg_verify_ms = total_verify.as_secs_f64() * 1000.0 / iterations as f64;

    // HealthRangeAir: 2 phases × 16 bits = 32 rows, 1 transition constraint (binary check)
    let bits_proven = 32; // 2 × 16 bits (value-min and max-value)
    let transition_constraints = 32; // 1 constraint per row

    println!("  Bits decomposed: {} (2 phases × 16 bits)", bits_proven);
    println!("  Transition constraints: {}", transition_constraints);
    println!(
        "  Avg prover time: {:.2} ms ({} iterations)",
        avg_prove_ms, iterations
    );
    println!("  Avg verifier time: {:.3} ms", avg_verify_ms);
    println!(
        "  Proof size: {} bytes ({:.1} KB)",
        proof_size,
        proof_size as f64 / 1024.0
    );

    // Extrapolate to 16,384-bit XOR binding:
    // XOR of 16,384 bits requires:
    // - 16,384 binary checks for vector A (1 constraint each)
    // - 16,384 binary checks for vector B (1 constraint each)
    // - 16,384 XOR constraints (c = a + b - 2*a*b, 1 multiplication each)
    // Total: 49,152 constraints
    // vs our measured 32 constraints
    let scale = 49_152.0 / transition_constraints as f64;
    let extrapolated_prove = avg_prove_ms * scale;
    let extrapolated_constraints = 49_152;

    println!("\n  Extrapolation to 16,384-bit XOR ({:.0}× scale):", scale);
    println!("    Constraints: {}", extrapolated_constraints);
    println!("    Estimated prover time: {:.0} ms", extrapolated_prove);
    println!(
        "    (Based on measured {:.2} ms / {} constraints)",
        avg_prove_ms, transition_constraints
    );

    WinterfellBaseline {
        bits_proven,
        transition_constraints,
        prove_time_ms: avg_prove_ms,
        verify_time_ms: avg_verify_ms,
        proof_size_bytes: proof_size,
        extrapolated_16k_prove_ms: extrapolated_prove,
        extrapolated_16k_constraints: extrapolated_constraints,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winterfell_baseline_runs() {
        let result = bench_winterfell_baseline();
        assert!(result.prove_time_ms > 0.0);
        assert!(result.verify_time_ms > 0.0);
        assert!(result.proof_size_bytes > 0);
        assert_eq!(result.extrapolated_16k_constraints, 49_152);
    }
}
