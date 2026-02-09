//! Property-based conservation law tests for physics modules.
//!
//! Uses proptest to verify physical invariants hold across randomized parameters:
//! trace preservation, positivity, unitarity, purity bounds, entropy non-negativity,
//! and eigenvalue sum conservation.

use proptest::prelude::*;
use super::decoherence::{Complex64, DecoherenceChannel, DensityMatrix, LindbladEvolution, simulate_decoherence};
use super::quantum_tunneling::TunnelingCalculator;

/// Generate a random pure state on the Bloch sphere for a 2-level system.
fn arb_pure_state_2d() -> impl Strategy<Value = DensityMatrix> {
    (0.0..std::f64::consts::PI, 0.0..2.0 * std::f64::consts::PI).prop_map(|(theta, phi)| {
        let c0 = Complex64::new((theta / 2.0).cos(), 0.0);
        let c1 = Complex64::new(
            (theta / 2.0).sin() * phi.cos(),
            (theta / 2.0).sin() * phi.sin(),
        );
        DensityMatrix::pure_state(&[c0, c1])
    })
}

/// Generate a random decoherence channel with γ ∈ [0.01, 10].
fn arb_channel() -> impl Strategy<Value = DecoherenceChannel> {
    (0..3u8, 0.01_f64..10.0).prop_map(|(kind, gamma)| match kind {
        0 => DecoherenceChannel::Dephasing { gamma },
        1 => DecoherenceChannel::AmplitudeDamping { gamma },
        _ => DecoherenceChannel::Depolarizing { p: gamma },
    })
}

/// Generate random tunneling parameters: E < V₀, with width in [0.5, 5] Å.
fn arb_tunneling_params() -> impl Strategy<Value = (f64, f64, f64)> {
    (0.1_f64..5.0, 0.5_f64..5.0).prop_flat_map(|(e_ev, width_angstrom)| {
        // V₀ must be > E
        let e_j = e_ev * 1.6e-19;
        let v_min_ev = e_ev + 0.1;
        (Just(e_j), (v_min_ev..20.0_f64).prop_map(|v| v * 1.6e-19), Just(width_angstrom * 1e-10))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Lindblad evolution preserves trace: |Tr(ρ(t)) - 1| < 1e-6
    #[test]
    fn prop_lindblad_trace_preservation(
        rho in arb_pure_state_2d(),
        channel in arb_channel(),
        dt_exp in -4.0_f64..-2.0,
    ) {
        let dt = 10.0_f64.powf(dt_exp);
        let steps = 100;

        let result = simulate_decoherence(&rho, channel, dt * steps as f64, steps);
        for (i, p) in result.purity.iter().enumerate() {
            // Purity should be finite and in valid range
            prop_assert!(p.is_finite(), "Purity is not finite at step {i}: {p}");
        }

        // Trace preservation: simulate_decoherence uses evolve() which is RK4.
        // The trace of the final state should remain ~1.
        // Reconstruct and check trace via the coherence + diagonal constraint:
        // For 2-level, Tr(ρ) = ρ₀₀ + ρ₁₁. Purity being finite implies trace is preserved.
        // Actually, let's directly evolve and check trace.
        let hamiltonian = DensityMatrix::new(2);
        let mut evolution = LindbladEvolution::new(hamiltonian);
        for (op, gamma) in channel.to_lindblad_operators() {
            evolution.add_lindblad(op, gamma);
        }
        let rho_final = evolution.evolve(&rho, dt);
        let tr = rho_final.trace();
        let tr_err = (tr.re - 1.0).abs() + tr.im.abs();
        prop_assert!(tr_err < 1e-6, "Trace not preserved: Tr = {} + {}i", tr.re, tr.im);
    }

    /// All eigenvalues of ρ(t) ≥ -1e-10 (positivity)
    #[test]
    fn prop_lindblad_positivity(
        rho in arb_pure_state_2d(),
        channel in arb_channel(),
    ) {
        let dt = 0.001;
        let hamiltonian = DensityMatrix::new(2);
        let mut evolution = LindbladEvolution::new(hamiltonian);
        for (op, gamma) in channel.to_lindblad_operators() {
            evolution.add_lindblad(op, gamma);
        }

        let mut current = rho;
        for step in 0..50 {
            current = evolution.evolve(&current, dt);
            let eigenvalues = current.hermitian_eigenvalues();
            for lambda in &eigenvalues {
                prop_assert!(
                    *lambda >= -1e-10,
                    "Negative eigenvalue {lambda} at step {step}"
                );
            }
        }
    }

    /// Tunneling unitarity: T + R = 1 for any E, V₀, width
    #[test]
    fn prop_tunneling_unitarity(
        (energy, barrier, width) in arb_tunneling_params(),
    ) {
        let calc = TunnelingCalculator::electron();
        let result = calc.rectangular_barrier(energy, barrier, width);
        let sum = result.transmission + result.reflection;
        prop_assert!(
            (sum - 1.0).abs() < 1e-10,
            "T + R = {sum} != 1 (E={energy:.2e}, V={barrier:.2e}, w={width:.2e})"
        );
    }

    /// Purity bounded: 1/d ≤ Tr(ρ²) ≤ 1 after any Lindblad step
    #[test]
    fn prop_purity_bounded(
        rho in arb_pure_state_2d(),
        channel in arb_channel(),
    ) {
        let dt = 0.001;
        let hamiltonian = DensityMatrix::new(2);
        let mut evolution = LindbladEvolution::new(hamiltonian);
        for (op, gamma) in channel.to_lindblad_operators() {
            evolution.add_lindblad(op, gamma);
        }

        let mut current = rho;
        for step in 0..50 {
            current = evolution.evolve(&current, dt);
            let p = current.purity();
            prop_assert!(
                p >= 0.5 - 1e-6 && p <= 1.0 + 1e-6,
                "Purity {p} out of [0.5, 1.0] at step {step}"
            );
        }
    }

    /// Entropy non-negative: S ≥ 0 for any density matrix
    #[test]
    fn prop_entropy_nonneg(
        rho in arb_pure_state_2d(),
        channel in arb_channel(),
    ) {
        let dt = 0.01;
        let hamiltonian = DensityMatrix::new(2);
        let mut evolution = LindbladEvolution::new(hamiltonian);
        for (op, gamma) in channel.to_lindblad_operators() {
            evolution.add_lindblad(op, gamma);
        }

        let evolved = evolution.evolve(&rho, dt);
        let s = evolved.von_neumann_entropy();
        prop_assert!(s >= -1e-10, "Entropy should be non-negative, got {s}");
    }

    /// Eigenvalue sum = Tr(ρ) = 1 for any density matrix
    #[test]
    fn prop_eigenvalue_sum(
        rho in arb_pure_state_2d(),
        channel in arb_channel(),
    ) {
        let dt = 0.01;
        let hamiltonian = DensityMatrix::new(2);
        let mut evolution = LindbladEvolution::new(hamiltonian);
        for (op, gamma) in channel.to_lindblad_operators() {
            evolution.add_lindblad(op, gamma);
        }

        let evolved = evolution.evolve(&rho, dt);
        let eigenvalues = evolved.hermitian_eigenvalues();
        let sum: f64 = eigenvalues.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-6,
            "Eigenvalue sum = {sum}, expected 1.0"
        );
    }
}
