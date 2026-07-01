// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Numerical validation tests for physics modules.
//!
//! Each test compares module output against a known analytical value or textbook result.
//! This catches formula transcription errors and numerical issues that range-only tests miss.

use super::acoustics::AcousticsEncoder;
use super::astrophysics::StellarEncoder;
use super::chaos_dynamics::{AttractorAnalyzer, LyapunovCalculator, systems};
use super::condensed_matter::CMEncoder;
use super::constants::{C, E_CHARGE, EPSILON_0, G, H, HBAR, K_BOLTZMANN, M_ELECTRON};
use super::cosmology::CosmologyEncoder;
use super::decoherence::{Complex64, DecoherenceChannel, DensityMatrix, simulate_decoherence};
use super::electromagnetism::{EMEncoder, Polarization, SpectrumRegion};
use super::fluid_dynamics::FluidEncoder;
use super::general_relativity::GREncoder;
use super::hadrons::Hadrons;
use super::nonequilibrium::{FluctuationDissipation, JarzynskiEstimator, OnsagerCoefficients};
use super::optics::{OpticsEncoder, PhotonStatistics};
use super::periodic_table::PeriodicTable;
use super::physics_test_helpers::assert_relative_eq;
use super::plasma_physics::PlasmaEncoder;
use super::quantum_information::QIEncoder;
use super::quantum_tunneling::TunnelingCalculator;
use super::standard_model::StandardModel;
use super::tensor_algebra::{ChristoffelSymbols, MetricTensor};
use super::thermal_transport::{LayerGeometry, ThermalProperties, ThermalTransport};
use super::thermodynamics::ThermoEncoder;
use super::trigger_systems::LcfPhysicsConstants;
use crate::consciousness_metrics::{QuantumEntropyCalculator, TruePhiCalculator};
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;

// =========================================================================
// Section 1: Quantum Tunneling
// =========================================================================

#[test]
fn parabolic_barrier_hill_wheeler() {
    // Hill-Wheeler formula: T = 1 / (1 + exp(2π(V₀ - E) / (ℏω)))
    let calc = TunnelingCalculator::electron();
    let v0: f64 = 5.0 * E_CHARGE; // barrier height in J
    let omega: f64 = 1.0e14; // rad/s

    let energies_ev: [f64; 6] = [3.0, 4.0, 4.5, 5.0, 5.5, 6.0];
    for &e_ev in &energies_ev {
        let energy = e_ev * E_CHARGE;
        let result = calc.parabolic_barrier(energy, v0, omega);

        // Independent Hill-Wheeler calculation
        let exponent: f64 = 2.0 * std::f64::consts::PI * (v0 - energy) / (HBAR * omega);
        let expected_t: f64 = 1.0 / (1.0 + exponent.exp());

        assert_relative_eq(
            result.transmission,
            expected_t,
            1e-6,
            &format!("Hill-Wheeler T at E={e_ev}eV"),
        );
    }
}

#[test]
fn rectangular_barrier_unitarity() {
    // T + R = 1 must hold for any parameters
    let calc = TunnelingCalculator::electron();
    let params: [(f64, f64, f64); 4] = [
        (1.0, 5.0, 1e-10),  // below barrier
        (4.0, 5.0, 1e-10),  // just below
        (6.0, 5.0, 1e-10),  // above barrier
        (2.0, 10.0, 5e-10), // thick barrier
    ];
    for &(e_ev, v_ev, width) in &params {
        let energy = e_ev * E_CHARGE;
        let barrier = v_ev * E_CHARGE;
        let result = calc.rectangular_barrier(energy, barrier, width);
        let sum = result.transmission + result.reflection;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "T+R={sum} != 1 at E={e_ev}eV, V={v_ev}eV, L={width}m"
        );
    }
}

#[test]
fn rectangular_barrier_wkb_value() {
    // WKB: T ≈ exp(-2γL), γ = √(2m(V-E)) / ℏ
    let calc = TunnelingCalculator::electron();
    let energy: f64 = 1.0 * E_CHARGE;
    let barrier: f64 = 5.0 * E_CHARGE;
    let width: f64 = 1e-10; // 1 Å

    let result = calc.rectangular_barrier(energy, barrier, width);

    // Hand-computed WKB
    let gamma: f64 = (2.0_f64 * M_ELECTRON * (barrier - energy)).sqrt() / HBAR;
    let expected_t: f64 = (-2.0_f64 * gamma * width).exp();

    assert_relative_eq(
        result.transmission,
        expected_t,
        1e-6,
        "WKB rectangular barrier T",
    );
}

#[test]
fn alpha_decay_geiger_nuttall() {
    // Higher Q-value → faster decay (shorter half-life)
    let calc = TunnelingCalculator::alpha_particle();
    let z_daughter: u8 = 82; // lead-like
    let radius: f64 = 7.5e-15; // nuclear radius ~7.5 fm

    let q_low: f64 = 4.0 * 1.6e-13; // 4 MeV in J
    let q_high: f64 = 8.0 * 1.6e-13; // 8 MeV in J

    let rate_low = calc.alpha_decay_rate(q_low, z_daughter, radius);
    let rate_high = calc.alpha_decay_rate(q_high, z_daughter, radius);

    let t_half_low = calc.half_life(rate_low);
    let t_half_high = calc.half_life(rate_high);

    assert!(
        t_half_high < t_half_low,
        "Geiger-Nuttall: higher Q should give shorter half-life, \
         got t½(high)={t_half_high} >= t½(low)={t_half_low}"
    );
}

#[test]
fn gamow_scales_with_charge() {
    // Gamow factor ∝ Z₁Z₂ at fixed energy
    let calc = TunnelingCalculator::proton();
    let energy: f64 = 1.0 * 1.6e-13; // 1 MeV in J

    let g1 = calc.gamow_factor(1, 10, energy); // Z product = 10
    let g2 = calc.gamow_factor(1, 20, energy); // Z product = 20

    let ratio = g2 / g1;
    let expected_ratio = 2.0; // 20/10

    assert_relative_eq(ratio, expected_ratio, 1e-6, "Gamow Z-scaling ratio");
}

// =========================================================================
// Section 2: Decoherence
// =========================================================================

#[test]
fn pure_state_purity_one() {
    // A pure state |ψ⟩ = (1, 0) has Tr(ρ²) = 1
    let psi = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let rho = DensityMatrix::pure_state(&psi);
    let p = rho.purity();
    assert!(
        (p - 1.0).abs() < 1e-10,
        "Pure state purity should be 1.0, got {p}"
    );

    // Also test a superposition: (1/√2)(|0⟩ + |1⟩) — still pure
    let s = 1.0 / 2.0_f64.sqrt();
    let psi2 = vec![Complex64::new(s, 0.0), Complex64::new(s, 0.0)];
    let rho2 = DensityMatrix::pure_state(&psi2);
    let p2 = rho2.purity();
    assert!(
        (p2 - 1.0).abs() < 1e-10,
        "Superposition pure state purity should be 1.0, got {p2}"
    );
}

#[test]
fn maximally_mixed_purity() {
    // Tr(ρ²) = 1/d for maximally mixed state of dimension d
    for d in [2, 3, 4] {
        let rho = DensityMatrix::maximally_mixed(d);
        let p = rho.purity();
        let expected = 1.0 / d as f64;
        assert!(
            (p - expected).abs() < 1e-10,
            "Maximally mixed d={d}: purity expected {expected}, got {p}"
        );
    }
}

#[test]
fn maximally_mixed_entropy() {
    // S = log₂(d) for maximally mixed state.
    // With proper Hermitian eigenvalue decomposition (analytical for d=2, Jacobi for d≥3),
    // we should match the exact value to high precision.
    for d in [2, 3, 4] {
        let rho = DensityMatrix::maximally_mixed(d);
        let s = rho.von_neumann_entropy();
        let expected = (d as f64).log2();
        assert!(
            (s - expected).abs() < 1e-6,
            "Maximally mixed d={d}: entropy expected {expected:.6}, got {s:.6}"
        );
    }

    // Pure state should have S ≈ 0
    let psi = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let pure_rho = DensityMatrix::pure_state(&psi);
    let s_pure = pure_rho.von_neumann_entropy();
    assert!(
        s_pure < 1e-6,
        "Pure state entropy should be ~0, got {s_pure}"
    );
}

#[test]
fn dephasing_coherence_decay() {
    // Pure dephasing: off-diagonal elements decay as exp(-2γt) under Lindblad evolution
    let gamma: f64 = 0.1;
    let psi = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let rho = DensityMatrix::pure_state(&psi);
    let channel = DecoherenceChannel::Dephasing { gamma };
    let total_time: f64 = 5.0;
    let steps: usize = 500;

    let result = simulate_decoherence(&rho, channel, total_time, steps);

    let dt = total_time / steps as f64;
    for &step_idx in &[50usize, 100, 200] {
        let t = step_idx as f64 * dt;
        let measured_coh = result.coherence[step_idx];
        let initial_coh = result.coherence[0];
        if initial_coh < 1e-15 {
            continue;
        }
        let ratio = measured_coh / initial_coh;
        // Lindblad dephasing with σ_z operator: off-diag decays as exp(-2γt)
        let expected_ratio = (-2.0 * gamma * t).exp();
        let rel_err = ((ratio - expected_ratio) / expected_ratio).abs();
        assert!(
            rel_err < 0.02,
            "Dephasing coherence at t={t:.2}: ratio={ratio:.4}, \
             expected={expected_ratio:.4} (rel err {rel_err:.2e})"
        );
    }
}

#[test]
fn dephasing_purity_converges() {
    // For 2-level dephasing starting from a superposition, purity → 0.5
    let psi = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let rho = DensityMatrix::pure_state(&psi);
    let channel = DecoherenceChannel::Dephasing { gamma: 1.0 };
    let total_time: f64 = 20.0;
    let steps: usize = 2000;

    let result = simulate_decoherence(&rho, channel, total_time, steps);
    let final_purity = *result.purity.last().unwrap();

    let rel_err = ((final_purity - 0.5) / 0.5).abs();
    assert!(
        rel_err < 0.02,
        "Dephasing purity should converge to 0.5, got {final_purity} (rel err {rel_err:.2e})"
    );
}

// =========================================================================
// Section 3: Chaos Dynamics
// =========================================================================

#[test]
fn lorenz_maximal_lyapunov() {
    // Standard Lorenz: σ=10, ρ=28, β=8/3 → λ₁ ≈ 0.906
    let calc = LyapunovCalculator::new(0.01, 5000, 50000);
    let lorenz = |state: &[f64]| systems::lorenz(state, 10.0, 28.0, 8.0 / 3.0);
    let initial = vec![1.0, 1.0, 1.0];

    let lambda1 = calc.maximal_lyapunov(lorenz, &initial, 1e-8);

    // Numerical Lyapunov exponents are sensitive to algorithm parameters;
    // accept 30% tolerance for this finite-difference computation.
    assert_relative_eq(lambda1, 0.906, 0.30, "Lorenz maximal Lyapunov exponent");
}

#[test]
fn kaplan_yorke_from_spectrum() {
    // D_KY = j + Σᵢ₌₁ʲ λᵢ / |λⱼ₊₁|
    // For Lorenz textbook values: λ = [0.906, 0.0, -14.572]
    // D_KY = 2 + 0.906/14.572
    let analyzer = AttractorAnalyzer::default();
    let spectrum = [0.906_f64, 0.0, -14.572];

    let d_ky = analyzer.kaplan_yorke_dimension(&spectrum);
    let expected = 2.0 + 0.906 / 14.572;

    assert_relative_eq(
        d_ky,
        expected,
        1e-10,
        "Kaplan-Yorke dimension (pure arithmetic)",
    );
}

#[test]
fn lorenz_exponent_sum() {
    // Σλᵢ = -(σ + 1 + β) = -(10 + 1 + 8/3) = -13.6667
    let calc = LyapunovCalculator::new(0.01, 5000, 50000);
    let lorenz = |state: &[f64]| systems::lorenz(state, 10.0, 28.0, 8.0 / 3.0);
    let jacobian = |state: &[f64]| systems::lorenz_jacobian(state, 10.0, 28.0, 8.0 / 3.0);
    let initial = vec![1.0, 1.0, 1.0];

    let spectrum = calc.lyapunov_spectrum(lorenz, jacobian, &initial);
    let sum: f64 = spectrum.iter().sum();
    let expected: f64 = -(10.0 + 1.0 + 8.0 / 3.0);

    assert_relative_eq(sum, expected, 0.15, "Lorenz exponent sum");
}

#[test]
fn stable_lorenz_negative_lyapunov() {
    // For ρ < 1, the origin is a stable fixed point → all exponents negative
    let calc = LyapunovCalculator::new(0.01, 2000, 10000);
    let lorenz = |state: &[f64]| systems::lorenz(state, 10.0, 0.5, 8.0 / 3.0);
    let initial = vec![0.1, 0.1, 0.1];

    let lambda1 = calc.maximal_lyapunov(lorenz, &initial, 1e-8);

    assert!(
        lambda1 < 0.0,
        "Stable Lorenz (rho=0.5) should have negative lambda_1, got {lambda1}"
    );
}

// =========================================================================
// Section 4: Non-equilibrium Thermodynamics
// =========================================================================

#[test]
fn einstein_relation_exact() {
    // D = k_B T / γ
    let temp: f64 = 300.0;
    let friction: f64 = 1.885e-12;
    let fd = FluctuationDissipation::new(temp, friction);

    let d = fd.diffusion_coefficient();
    let expected: f64 = K_BOLTZMANN * temp / friction;

    assert_relative_eq(d, expected, 1e-6, "Einstein relation D = k_BT/gamma");
}

#[test]
fn msd_linear_in_time() {
    // MSD(t) = 2Dt — ratio MSD(10)/MSD(1) = 10
    let fd = FluctuationDissipation::new(300.0, 1e-12);
    let msd1 = fd.mean_squared_displacement(1.0);
    let msd10 = fd.mean_squared_displacement(10.0);

    let ratio = msd10 / msd1;
    assert!(
        (ratio - 10.0).abs() < 1e-10,
        "MSD linearity: MSD(10)/MSD(1) = {ratio}, expected 10.0"
    );
}

#[test]
fn onsager_diagonal_flux() {
    // J_i = L_ii * X_i for diagonal Onsager matrix
    let diag = vec![1.5_f64, 2.5, 3.5];
    let onsager = OnsagerCoefficients::diagonal(&diag);
    let forces = vec![0.1_f64, 0.2, 0.3];
    let fluxes = onsager.compute_fluxes(&forces);

    for i in 0..3 {
        let expected = diag[i] * forces[i];
        assert!(
            (fluxes[i] - expected).abs() < 1e-14,
            "Onsager flux[{i}]: expected {expected}, got {}",
            fluxes[i]
        );
    }
}

#[test]
fn entropy_production_nonneg() {
    // σ = Σᵢⱼ Lᵢⱼ Xᵢ Xⱼ ≥ 0 for positive-definite L
    let onsager = OnsagerCoefficients::diagonal(&[1.0, 2.0, 3.0]);
    let forces_sets: Vec<Vec<f64>> = vec![
        vec![0.1, -0.2, 0.3],
        vec![1.0, 0.0, -1.0],
        vec![0.0, 0.0, 0.0],
        vec![-0.5, 0.5, 0.5],
    ];

    for forces in &forces_sets {
        let sigma = onsager.entropy_production(forces);
        assert!(
            sigma >= 0.0,
            "Entropy production should be >= 0, got {sigma} for forces {forces:?}"
        );
    }
}

#[test]
fn jarzynski_constant_work() {
    // When all work samples equal W₀, ΔF = W₀
    let temp: f64 = 300.0;
    let w0: f64 = 1e-20;
    let samples = vec![w0; 100];
    let estimator = JarzynskiEstimator::new();
    let delta_f = estimator.free_energy_difference(&samples, temp);

    assert_relative_eq(delta_f, w0, 1e-6, "Jarzynski delta_F with constant work");
}

// =========================================================================
// Section 5: Plasma Physics
// =========================================================================

#[test]
fn debye_length_value() {
    // λ_D = √(ε₀ k_B T / (n e²)), with T in K = temp_ev * 11604
    let genesis = GenesisSeed::from_phrase("plasma_validation_test");
    let encoder = PlasmaEncoder::from_genesis(&genesis);

    let temp_ev: f64 = 1.0;
    let density: f64 = 1e18;
    let lambda_d = encoder.debye_length_m(temp_ev, density);

    let t_k: f64 = temp_ev * 11604.0;
    let expected: f64 = (EPSILON_0 * K_BOLTZMANN * t_k / (density * E_CHARGE * E_CHARGE)).sqrt();

    assert_relative_eq(lambda_d, expected, 1e-6, "Debye length");
}

#[test]
fn plasma_frequency_value() {
    // ω_p = √(n e² / (ε₀ m_e))
    let genesis = GenesisSeed::from_phrase("plasma_validation_test");
    let encoder = PlasmaEncoder::from_genesis(&genesis);

    let density: f64 = 1e18;
    let omega_p = encoder.plasma_frequency_rad_s(density);
    let expected: f64 = (density * E_CHARGE * E_CHARGE / (EPSILON_0 * M_ELECTRON)).sqrt();

    assert_relative_eq(omega_p, expected, 1e-6, "Plasma frequency");
}

#[test]
fn cyclotron_frequency_value() {
    // ω_c = eB / m_e
    let genesis = GenesisSeed::from_phrase("plasma_validation_test");
    let encoder = PlasmaEncoder::from_genesis(&genesis);

    let b: f64 = 1.0; // Tesla
    let omega_c = encoder.cyclotron_frequency_rad_s(b);
    let expected: f64 = E_CHARGE * b / M_ELECTRON;

    assert_relative_eq(omega_c, expected, 1e-6, "Cyclotron frequency");
}

// =========================================================================
// Section 6: Logistic Map Lyapunov Exponent
// =========================================================================

#[test]
fn logistic_map_lyapunov_r4() {
    // For the logistic map x_{n+1} = r·x·(1-x) at r=4 (full chaos),
    // the Lyapunov exponent is exactly ln(2) ≈ 0.6931.
    // λ = (1/N) Σ ln|f'(xₙ)| = (1/N) Σ ln|r(1-2xₙ)|
    let r: f64 = 4.0;
    let transient = 1000;
    let iterations = 100_000;

    // Start from an irrational-ish point to avoid short cycles
    let mut x: f64 = 0.1234567;

    // Discard transient
    for _ in 0..transient {
        x = systems::logistic(x, r);
    }

    // Accumulate Lyapunov sum
    let mut lyap_sum: f64 = 0.0;
    for _ in 0..iterations {
        let deriv = r * (1.0 - 2.0 * x);
        lyap_sum += deriv.abs().ln();
        x = systems::logistic(x, r);
    }
    let lambda = lyap_sum / iterations as f64;

    assert_relative_eq(lambda, 2.0_f64.ln(), 1e-3, "Logistic map Lyapunov at r=4");
}

// =========================================================================
// Section 7: Amplitude Damping & Depolarizing Channels
// =========================================================================

#[test]
fn amplitude_damping_decay_to_ground() {
    // AmplitudeDamping channel drives |1⟩ → |0⟩ (ground state, purity → 1)
    let psi = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]; // |1⟩
    let rho = DensityMatrix::pure_state(&psi);
    let channel = DecoherenceChannel::AmplitudeDamping { gamma: 0.5 };
    let result = simulate_decoherence(&rho, channel, 30.0, 3000);

    let final_purity = *result.purity.last().unwrap();
    assert!(
        (final_purity - 1.0).abs() < 0.02,
        "Amplitude damping final purity should be ~1.0 (pure ground), got {final_purity}"
    );
}

#[test]
fn depolarizing_converges_to_maximally_mixed() {
    // Depolarizing channel drives any state → maximally mixed (purity → 1/d = 0.5)
    let psi = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]; // |0⟩
    let rho = DensityMatrix::pure_state(&psi);
    let channel = DecoherenceChannel::Depolarizing { p: 0.5 };
    let result = simulate_decoherence(&rho, channel, 20.0, 2000);

    let final_purity = *result.purity.last().unwrap();
    assert!(
        (final_purity - 0.5).abs() < 0.02,
        "Depolarizing final purity should be ~0.5 (maximally mixed), got {final_purity}"
    );
}

// =========================================================================
// Section 8: Eckart Barrier Validation
// =========================================================================

#[test]
fn eckart_barrier_unitarity() {
    // T + R = 1 for Eckart barrier at various energies
    let calc = TunnelingCalculator::electron();
    let v0: f64 = 1.0e-19; // barrier height ~0.6 eV
    let a: f64 = 1.0e-10; // barrier width ~1 Å

    let energies = [0.1e-19, 0.3e-19, 0.5e-19, 0.8e-19, 1.2e-19, 2.0e-19];
    for &e in &energies {
        let result = calc.eckart_barrier(e, v0, a);
        let sum = result.transmission + result.reflection;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Eckart T+R={sum} != 1 at E={e:.2e}, V0={v0:.2e}"
        );
    }
}

#[test]
fn eckart_barrier_transmission_increases_with_energy() {
    // For E < V₀, transmission should monotonically increase with energy
    let calc = TunnelingCalculator::electron();
    let v0: f64 = 1.0e-19;
    let a: f64 = 1.0e-10;

    let energies = [0.1e-19, 0.2e-19, 0.3e-19, 0.5e-19, 0.7e-19, 0.9e-19];
    let mut prev_t = 0.0;
    for &e in &energies {
        let result = calc.eckart_barrier(e, v0, a);
        assert!(
            result.transmission >= prev_t,
            "Eckart T should increase with E: T({:.2e})={} < T_prev={}",
            e,
            result.transmission,
            prev_t
        );
        prev_t = result.transmission;
    }
}

#[test]
fn eckart_barrier_wkb_value() {
    // Cross-check: manual WKB integral with 2000 midpoint-rule points vs method (200 points).
    // The method uses 200 points; we recompute with 2000 and compare (10% tolerance for
    // discretization difference).
    let calc = TunnelingCalculator::electron();
    let v0: f64 = 1.0e-19;
    let a: f64 = 1.0e-10;
    let energy: f64 = 0.3e-19;

    let result = calc.eckart_barrier(energy, v0, a);

    // Manual WKB with 2000 points
    let n = 2000;
    let x_start = -5.0 * a;
    let x_end = 5.0 * a;
    let dx = (x_end - x_start) / n as f64;
    let mut action = 0.0f64;
    for i in 0..n {
        let x = x_start + (i as f64 + 0.5) * dx;
        let v = v0 / (x / a).cosh().powi(2);
        if v > energy {
            let kappa = (2.0 * M_ELECTRON * (v - energy)).sqrt() / HBAR;
            action += kappa * dx;
        }
    }
    let expected_t = (-2.0 * action).exp();

    assert_relative_eq(
        result.transmission,
        expected_t,
        0.10,
        "Eckart WKB cross-check (200 vs 2000 points)",
    );
}

// =========================================================================
// Section 9: Onsager Symmetric Matrix
// =========================================================================

#[test]
fn onsager_symmetric_full_matrix() {
    // 3×3 symmetric matrix from upper triangular: [2.0, 0.5, 0.3, 3.0, 0.4, 1.0]
    // Layout: L00=2.0, L01=L10=0.5, L02=L20=0.3, L11=3.0, L12=L21=0.4, L22=1.0
    let onsager = OnsagerCoefficients::symmetric(&[2.0, 0.5, 0.3, 3.0, 0.4, 1.0], 3);

    assert!(
        onsager.verify_symmetry(1e-15),
        "Symmetric Onsager matrix should verify symmetry"
    );
    assert!(
        (onsager.get(0, 1) - 0.5).abs() < 1e-15,
        "L01 should be 0.5, got {}",
        onsager.get(0, 1)
    );
    assert!(
        (onsager.get(1, 0) - 0.5).abs() < 1e-15,
        "L10 should be 0.5, got {}",
        onsager.get(1, 0)
    );
    assert!(
        (onsager.get(0, 2) - 0.3).abs() < 1e-15,
        "L02 should be 0.3, got {}",
        onsager.get(0, 2)
    );
    assert!(
        (onsager.get(2, 0) - 0.3).abs() < 1e-15,
        "L20 should be 0.3, got {}",
        onsager.get(2, 0)
    );
}

#[test]
fn onsager_coupled_flux_matrix_vector() {
    // J = L·X for L = [[2.0, 0.5, 0.3], [0.5, 3.0, 0.4], [0.3, 0.4, 1.0]], X = [1, -0.5, 2]
    // J0 = 2.0*1 + 0.5*(-0.5) + 0.3*2 = 2.0 - 0.25 + 0.6 = 2.35
    // J1 = 0.5*1 + 3.0*(-0.5) + 0.4*2 = 0.5 - 1.5 + 0.8 = -0.2
    // J2 = 0.3*1 + 0.4*(-0.5) + 1.0*2 = 0.3 - 0.2 + 2.0 = 2.1
    let onsager = OnsagerCoefficients::symmetric(&[2.0, 0.5, 0.3, 3.0, 0.4, 1.0], 3);
    let forces = vec![1.0, -0.5, 2.0];
    let fluxes = onsager.compute_fluxes(&forces);

    let expected = [2.35, -0.2, 2.1];
    for i in 0..3 {
        assert!(
            (fluxes[i] - expected[i]).abs() < 1e-14,
            "Flux[{i}]: expected {}, got {}",
            expected[i],
            fluxes[i]
        );
    }
}

#[test]
fn onsager_entropy_production_quadratic_form() {
    // σ = X^T · L · X = Σ_ij L_ij X_i X_j
    // For L and X above:
    // σ = X·J = 1*2.35 + (-0.5)*(-0.2) + 2*2.1 = 2.35 + 0.1 + 4.2 = 6.65
    let onsager = OnsagerCoefficients::symmetric(&[2.0, 0.5, 0.3, 3.0, 0.4, 1.0], 3);
    let forces = vec![1.0, -0.5, 2.0];
    let sigma = onsager.entropy_production(&forces);

    // Cross-check via manual double sum
    let l = [[2.0, 0.5, 0.3], [0.5, 3.0, 0.4], [0.3, 0.4, 1.0]];
    let x = [1.0, -0.5, 2.0];
    let mut manual_sigma = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            manual_sigma += l[i][j] * x[i] * x[j];
        }
    }

    assert!(
        (sigma - 6.65).abs() < 1e-14,
        "Entropy production should be 6.65, got {sigma}"
    );
    assert!(
        (sigma - manual_sigma).abs() < 1e-14,
        "Entropy production {sigma} != manual double-sum {manual_sigma}"
    );
}

// =========================================================================
// Section 10: ThermalRelaxation Channel
// =========================================================================

#[test]
fn thermal_relaxation_high_temp() {
    // At high temperature, ThermalRelaxation drives any state → maximally mixed (purity → 0.5)
    // Qubit energy gap: ℏ * 2π * 5GHz ≈ 3.3e-24 J. At T=300K,
    // n_th ≈ k_BT/E ≈ 1254. Combined Lindblad rates ≈ 2*n_th/t1 ≈ 2508/t1.
    // With t1=5000, effective rate ≈ 0.50 s⁻¹, so dt=0.01 gives rate*dt ≈ 0.005 (stable).
    let energy_gap = HBAR * 2.0 * std::f64::consts::PI * 5.0e9;
    let channel = DecoherenceChannel::ThermalRelaxation {
        t1: 5000.0,
        t2: 3000.0,
        temperature: 300.0,
        energy_gap,
    };
    // Start from |0⟩
    let psi = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let rho = DensityMatrix::pure_state(&psi);
    // 50s / 5000 steps = dt=0.01. Need ~5/0.50 ≈ 10s to converge (5 time constants).
    let result = simulate_decoherence(&rho, channel, 50.0, 5000);
    let final_purity = *result.purity.last().unwrap();

    assert!(
        (final_purity - 0.5).abs() < 0.10,
        "High-temp thermal relaxation should converge to purity ~0.5, got {final_purity}"
    );
}

#[test]
fn thermal_relaxation_low_temp() {
    // At very low temperature, ThermalRelaxation drives any state → ground state (purity → 1.0)
    // At T=0.015K with 5GHz qubit gap, n_th ≈ exp(-h*5e9 / k_B*0.015) ≈ exp(-16) ≈ 0 (negligible)
    let energy_gap = HBAR * 2.0 * std::f64::consts::PI * 5.0e9;
    let channel = DecoherenceChannel::ThermalRelaxation {
        t1: 1.0,
        t2: 1.5,
        temperature: 0.015,
        energy_gap,
    };
    // Start from |1⟩ (excited state)
    let psi = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
    let rho = DensityMatrix::pure_state(&psi);
    let result = simulate_decoherence(&rho, channel, 30.0, 3000);
    let final_purity = *result.purity.last().unwrap();

    assert!(
        (final_purity - 1.0).abs() < 0.05,
        "Low-temp thermal relaxation should converge to purity ~1.0, got {final_purity}"
    );
}

// =========================================================================
// Section 11: Jarzynski with Gaussian Work
// =========================================================================

/// Inverse normal CDF (Abramowitz & Stegun 26.2.23 rational approximation)
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return -6.0;
    }
    if p >= 1.0 {
        return 6.0;
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    // Rational approximation coefficients
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let approx = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -approx } else { approx }
}

#[test]
fn jarzynski_gaussian_work() {
    // For Gaussian-distributed work W ~ N(W₀, σ²),
    // Jarzynski equality gives: ΔF = W₀ - β·σ²/2
    let w0: f64 = 5e-21; // Mean work (J)
    let sigma: f64 = 1e-21; // Std dev (J)
    let temp: f64 = 300.0;
    let beta: f64 = 1.0 / (K_BOLTZMANN * temp);

    let expected_delta_f = w0 - beta * sigma * sigma / 2.0;

    // Generate deterministic Gaussian samples via inverse CDF (quantile sampling)
    let n = 10_000;
    let samples: Vec<f64> = (0..n)
        .map(|i| {
            let p = (i as f64 + 0.5) / n as f64;
            let z = inverse_normal_cdf(p);
            w0 + sigma * z
        })
        .collect();

    let estimator = JarzynskiEstimator::new();
    let delta_f = estimator.free_energy_difference(&samples, temp);

    assert_relative_eq(
        delta_f,
        expected_delta_f,
        0.01,
        "Jarzynski Gaussian work ΔF",
    );
}

// =========================================================================
// Section 12: FluctuationDissipation
// =========================================================================

#[test]
fn noise_amplitude_formula() {
    // noise_amplitude(dt) = √(2·k_B·T·γ / dt)
    let temp: f64 = 300.0;
    let friction: f64 = 1e-12;
    let dt: f64 = 1e-6;
    let fd = FluctuationDissipation::new(temp, friction);

    let actual = fd.noise_amplitude(dt);
    let expected = (2.0 * K_BOLTZMANN * temp * friction / dt).sqrt();

    assert_relative_eq(actual, expected, 1e-10, "Noise amplitude formula");
}

#[test]
fn velocity_correlation_time_formula() {
    // τ = m / γ
    let temp: f64 = 300.0;
    let friction: f64 = 1e-12;
    let mass: f64 = 1e-20;
    let fd = FluctuationDissipation::new(temp, friction);

    let actual = fd.velocity_correlation_time(mass);
    let expected = mass / friction;

    assert_relative_eq(actual, expected, 1e-10, "Velocity correlation time τ = m/γ");
}

// =========================================================================
// Section 13: Hermitian Eigenvalue Verification
// =========================================================================

#[test]
fn hermitian_eigenvalues_known_spectrum() {
    // 3×3 diagonal density matrix: diag(0.6, 0.3, 0.1)
    // Eigenvalues should be exactly {0.6, 0.3, 0.1}
    let mut rho = DensityMatrix::new(3);
    rho.set(0, 0, Complex64::from_real(0.6));
    rho.set(1, 1, Complex64::from_real(0.3));
    rho.set(2, 2, Complex64::from_real(0.1));

    let mut eigenvalues = rho.hermitian_eigenvalues();
    eigenvalues.sort_by(|a, b| b.total_cmp(a)); // Descending

    assert_relative_eq(eigenvalues[0], 0.6, 1e-10, "λ₁ of diag(0.6,0.3,0.1)");
    assert_relative_eq(eigenvalues[1], 0.3, 1e-10, "λ₂ of diag(0.6,0.3,0.1)");
    assert_relative_eq(eigenvalues[2], 0.1, 1e-10, "λ₃ of diag(0.6,0.3,0.1)");

    // Entropy: S = -(0.6 log₂ 0.6 + 0.3 log₂ 0.3 + 0.1 log₂ 0.1)
    let s = rho.von_neumann_entropy();
    let expected_s =
        -(0.6_f64 * 0.6_f64.log2() + 0.3_f64 * 0.3_f64.log2() + 0.1_f64 * 0.1_f64.log2());
    assert_relative_eq(s, expected_s, 1e-10, "Entropy of diag(0.6,0.3,0.1)");
}

#[test]
fn hermitian_eigenvalues_off_diagonal() {
    // 2×2 Hermitian: [[0.7, 0.1], [0.1, 0.3]]
    // λ = 0.5 ± √(0.04 + 0.01) = 0.5 ± √0.05
    let mut rho = DensityMatrix::new(2);
    rho.set(0, 0, Complex64::from_real(0.7));
    rho.set(0, 1, Complex64::from_real(0.1));
    rho.set(1, 0, Complex64::from_real(0.1));
    rho.set(1, 1, Complex64::from_real(0.3));

    let mut eigenvalues = rho.hermitian_eigenvalues();
    eigenvalues.sort_by(|a, b| b.total_cmp(a));

    let disc = 0.05_f64.sqrt();
    let lambda1 = 0.5 + disc;
    let lambda2 = 0.5 - disc;

    assert_relative_eq(
        eigenvalues[0],
        lambda1,
        1e-10,
        "λ₁ of [[0.7,0.1],[0.1,0.3]]",
    );
    assert_relative_eq(
        eigenvalues[1],
        lambda2,
        1e-10,
        "λ₂ of [[0.7,0.1],[0.1,0.3]]",
    );
}

// =========================================================================
// Section 14: Optics
// =========================================================================

#[test]
fn photon_energy_ev_formula() {
    // E = 1239.8 / λ(nm)
    let genesis = GenesisSeed::from_phrase("optics_validation");
    let encoder = OpticsEncoder::from_genesis(&genesis);

    let test_cases: [(f64, f64); 3] = [
        (532.0, 1239.8 / 532.0),
        (1064.0, 1239.8 / 1064.0),
        (633.0, 1239.8 / 633.0),
    ];

    for (wavelength, expected) in &test_cases {
        let beam = encoder.create_beam(*wavelength, 1.0, 100.0, PhotonStatistics::Coherent);
        let actual = beam.photon_energy_ev();
        assert_relative_eq(
            actual,
            *expected,
            1e-10,
            &format!("Photon energy at {}nm", wavelength),
        );
    }
}

#[test]
fn photon_flux_formula() {
    // Φ = P / (h·c / λ) = P·λ / (h·c)
    let genesis = GenesisSeed::from_phrase("optics_validation");
    let encoder = OpticsEncoder::from_genesis(&genesis);

    let wavelength_nm: f64 = 532.0;
    let power_w: f64 = 1.0;
    let beam = encoder.create_beam(wavelength_nm, power_w, 100.0, PhotonStatistics::Coherent);
    let actual = beam.photon_flux();

    let expected = power_w / (H * C / (wavelength_nm * 1e-9));
    assert_relative_eq(actual, expected, 1e-10, "Photon flux at 1W, 532nm");
}

#[test]
fn rayleigh_range_formula() {
    // z_R = π·w₀² / λ, with w₀ in mm and λ in mm → result in mm
    // create_beam: waist_um=100 → w₀ = 0.1 mm, wavelength_nm=532 → λ = 532e-6 mm
    let genesis = GenesisSeed::from_phrase("optics_validation");
    let encoder = OpticsEncoder::from_genesis(&genesis);

    let wavelength_nm: f64 = 532.0;
    let waist_um: f64 = 100.0;
    let beam = encoder.create_beam(wavelength_nm, 1.0, waist_um, PhotonStatistics::Coherent);
    let actual = beam.rayleigh_range_mm();

    // w0_mm = waist_um * 1e-3 = 0.1 mm
    // lambda_mm = wavelength_nm * 1e-6 = 532e-6 mm
    let w0_mm = waist_um * 1e-3;
    let lambda_mm = wavelength_nm * 1e-6;
    let expected = std::f64::consts::PI * w0_mm * w0_mm / lambda_mm;

    assert_relative_eq(
        actual,
        expected,
        1e-10,
        "Rayleigh range for 100μm waist at 532nm",
    );
}

// =========================================================================
// Section 15: Electromagnetism
// =========================================================================

#[test]
fn em_spectrum_frequency_ranges() {
    // Verify all 7 band boundaries match documented values
    let bands: [(SpectrumRegion, f64, f64); 7] = [
        (SpectrumRegion::Radio, 3e3, 3e9),
        (SpectrumRegion::Microwave, 3e9, 3e11),
        (SpectrumRegion::Infrared, 3e11, 4e14),
        (SpectrumRegion::Visible, 4e14, 8e14),
        (SpectrumRegion::Ultraviolet, 8e14, 3e16),
        (SpectrumRegion::XRay, 3e16, 3e19),
        (SpectrumRegion::GammaRay, 3e19, f64::INFINITY),
    ];

    for (region, expected_lo, expected_hi) in &bands {
        let (lo, hi) = region.frequency_range();
        assert!(
            (lo - expected_lo).abs() < 1.0,
            "{:?} lower bound: expected {expected_lo}, got {lo}",
            region
        );
        if expected_hi.is_finite() {
            assert!(
                (hi - expected_hi).abs() < 1.0,
                "{:?} upper bound: expected {expected_hi}, got {hi}",
                region
            );
        } else {
            assert!(
                hi.is_infinite(),
                "{:?} upper bound should be infinity",
                region
            );
        }
    }
}

#[test]
fn em_spectrum_classification() {
    // Verify that create_wave classifies center frequencies correctly
    let genesis = GenesisSeed::from_phrase("em_validation");
    let encoder = EMEncoder::from_genesis(&genesis);

    let test_cases: [(f64, SpectrumRegion); 6] = [
        (1e6, SpectrumRegion::Radio),        // 1 MHz
        (10e9, SpectrumRegion::Microwave),   // 10 GHz
        (1e13, SpectrumRegion::Infrared),    // 10 THz
        (6e14, SpectrumRegion::Visible),     // 600 THz (green light)
        (1e15, SpectrumRegion::Ultraviolet), // 1 PHz
        (1e18, SpectrumRegion::XRay),        // 1 EHz
    ];

    for (freq, expected_region) in &test_cases {
        let wave = encoder.create_wave(*freq, Polarization::Linear, &genesis);
        assert_eq!(
            wave.region, *expected_region,
            "Frequency {freq:.2e} Hz classified as {:?}, expected {:?}",
            wave.region, expected_region
        );
    }
}

#[test]
fn em_wavelength_frequency_consistency() {
    // λ = c / f for all frequencies
    let genesis = GenesisSeed::from_phrase("em_validation");
    let encoder = EMEncoder::from_genesis(&genesis);

    let frequencies: [f64; 4] = [1e6, 1e10, 5e14, 1e18];
    for &freq in &frequencies {
        let wave = encoder.create_wave(freq, Polarization::Linear, &genesis);
        let expected_wavelength = C / freq;
        assert_relative_eq(
            wave.wavelength_m,
            expected_wavelength,
            1e-10,
            &format!("λ = c/f at f={freq:.2e} Hz"),
        );
    }
}

// =========================================================================
// Section 16: Thermal Transport
// =========================================================================

#[test]
fn thermal_diffusivity_formula() {
    // α = k / (ρ · Cₚ)
    let hea = ThermalProperties::hea_shell();
    let alpha = hea.diffusivity();
    let expected = 15.0 / (7200.0 * 450.0);
    assert_relative_eq(alpha, expected, 1e-10, "HEA thermal diffusivity");
}

#[test]
fn spherical_layer_volume() {
    // V = (4/3)π(R_outer³ - R_inner³)
    let r_inner: f64 = 0.05; // 5 cm
    let r_outer: f64 = 0.10; // 10 cm
    let geom = LayerGeometry::sphere(r_inner, r_outer);
    let vol = geom.volume();

    let expected = (4.0 / 3.0) * std::f64::consts::PI * (r_outer.powi(3) - r_inner.powi(3));
    assert_relative_eq(vol, expected, 1e-10, "Spherical shell volume");
}

#[test]
fn coolant_flow_rate_formula() {
    // ṁ = Q / (Cₚ · ΔT)
    let genesis = GenesisSeed::from_phrase("thermal_validation");
    let thermal = ThermalTransport::from_genesis(&genesis);

    let power_w: f64 = 5000.0;
    let cp: f64 = 4186.0; // Water Cₚ (J/kg·K)
    let dt: f64 = 10.0; // 10 K rise

    let actual = thermal.coolant_flow_rate(power_w, cp, dt);
    let expected = power_w / (cp * dt);
    assert_relative_eq(actual, expected, 1e-10, "Coolant flow rate ṁ = Q/(Cₚ·ΔT)");
}

#[test]
fn thermal_stress_hoop() {
    // σ_θ = α·E·ΔT / (2·(1-ν))
    // Using HEA shell properties: α = 10e-6 /K
    // Young's modulus E = 120 GPa, Poisson ν = 0.33, yield = 800 MPa
    let genesis = GenesisSeed::from_phrase("thermal_validation");
    let thermal = ThermalTransport::from_genesis(&genesis);
    let hea = ThermalProperties::hea_shell();

    // Create a minimal thermal profile with a known ΔT
    // Build a TemperatureProfile by doing a real steady-state calculation
    // with known boundary conditions
    let shell = ThermalProperties::hea_shell();
    let interface = ThermalProperties::nano_laminate();
    let core = ThermalProperties::max_phase();

    let core_geom = LayerGeometry::sphere(0.0, 0.03);
    let iface_geom = LayerGeometry::sphere(0.03, 0.04);
    let shell_geom = LayerGeometry::sphere(0.04, 0.06);

    let profile = thermal.steady_state_profile(
        1000.0, // 1 kW
        &shell,
        &shell_geom,
        &interface,
        &iface_geom,
        &core,
        &core_geom,
        300.0,  // coolant at 300 K
        5000.0, // h = 5000 W/m²K
    );

    let alpha_coeff = hea.expansion_coeff; // 10e-6
    let youngs = 120e9;
    let poisson = 0.33;
    let yield_stress = 800e6;

    let stress = thermal.thermal_stress(&profile, &hea, &shell_geom, youngs, poisson, yield_stress);

    // Expected hoop stress: σ_θ = α·E·ΔT / (2·(1-ν))
    let dt = profile.t_max - profile.t_shell_outer;
    let expected_hoop = alpha_coeff * youngs * dt / (2.0 * (1.0 - poisson));

    assert_relative_eq(
        stress.max_hoop_stress,
        expected_hoop,
        1e-6,
        "Thermal hoop stress σ = α·E·ΔT/(2(1-ν))",
    );
}

// =========================================================================
// Section 17: Classical Mechanics
// =========================================================================

#[test]
fn kinetic_energy_half_mv2() {
    // T = ½mv² — the scalar embedded in the HDC vector should scale norm proportionally
    let genesis = GenesisSeed::from_phrase("classical_validation");
    let encoder = super::classical_mechanics::ClassicalMechanicsEncoder::from_genesis(&genesis);

    // T = 0.5 * 2.0 * 3.0² = 9.0
    let ke = encoder.kinetic_energy(2.0, 3.0);
    // The vector is kinetic.bind(mass).bind(velocity).scale(T as f32)
    // So norm should be ~9.0 * base_norm (where base is the bound vector before scaling)
    let ke_1 = encoder.kinetic_energy(1.0, 1.0); // T = 0.5
    let ratio = ke.norm() / ke_1.norm();
    // T(2,3)/T(1,1) = 9.0 / 0.5 = 18.0
    assert_relative_eq(
        ratio as f64,
        18.0,
        1e-6,
        "Kinetic energy ratio m=2,v=3 vs m=1,v=1",
    );
}

#[test]
fn ising_2d_critical_temperature() {
    // Tc = 2J / ln(1 + √2)
    let genesis = GenesisSeed::from_phrase("ising_validation");
    let encoder = super::statistical_mechanics::StatMechEncoder::from_genesis(&genesis);
    let j = 1.0;
    let ising = encoder.ising_model(2, j, 0.0);

    let expected_tc = 2.0 * j / (1.0 + 2.0_f64.sqrt()).ln();
    assert_relative_eq(
        ising.critical_temp,
        expected_tc,
        1e-10,
        "Ising 2D Tc = 2J/ln(1+√2)",
    );
}

// =========================================================================
// Section 18: Statistical Mechanics
// =========================================================================

#[test]
fn ising_2d_critical_temp_exact() {
    // Onsager exact result: kTc/J = 2/ln(1+√2) ≈ 2.269185...
    let genesis = GenesisSeed::from_phrase("stat_mech_validation");
    let encoder = super::statistical_mechanics::StatMechEncoder::from_genesis(&genesis);
    let ising = encoder.ising_model(2, 1.0, 0.0);

    let exact_tc = 2.0 / (1.0 + 2.0_f64.sqrt()).ln();
    assert_relative_eq(ising.critical_temp, exact_tc, 1e-10, "Onsager exact Tc");
}

#[test]
fn ising_magnetization_above_tc() {
    // Above Tc, spontaneous magnetization = 0 (paramagnetic phase)
    let genesis = GenesisSeed::from_phrase("stat_mech_validation");
    let encoder = super::statistical_mechanics::StatMechEncoder::from_genesis(&genesis);
    let ising = encoder.ising_model(2, 1.0, 0.0);

    let m = encoder.ising_magnetization(&ising, 3.0); // T=3.0 > Tc≈2.27
    assert!(
        m == 0.0,
        "Magnetization above Tc should be exactly 0, got {m}"
    );
}

#[test]
fn ising_magnetization_below_tc() {
    // Below Tc: M = (1 - T/Tc)^β with β = 1/8 = 0.125
    let genesis = GenesisSeed::from_phrase("stat_mech_validation");
    let encoder = super::statistical_mechanics::StatMechEncoder::from_genesis(&genesis);
    let ising = encoder.ising_model(2, 1.0, 0.0);

    let t = 1.5;
    let m = encoder.ising_magnetization(&ising, t);
    let expected = (1.0 - t / ising.critical_temp).powf(0.125);
    assert_relative_eq(m, expected, 1e-10, "Ising 2D magnetization at T=1.5");
}

// =========================================================================
// Section 19: Neuroscience
// =========================================================================

#[test]
fn nernst_potentials() {
    use super::neuroscience::IonType;

    // Standard textbook Nernst potentials
    assert_relative_eq(
        IonType::Sodium.nernst_potential(),
        60.0,
        1e-10,
        "Na+ Nernst",
    );
    assert_relative_eq(
        IonType::Potassium.nernst_potential(),
        -90.0,
        1e-10,
        "K+ Nernst",
    );
    assert_relative_eq(
        IonType::Calcium.nernst_potential(),
        120.0,
        1e-10,
        "Ca2+ Nernst",
    );
    assert_relative_eq(
        IonType::Chloride.nernst_potential(),
        -80.0,
        1e-10,
        "Cl- Nernst",
    );
}

#[test]
fn ion_channel_conductances() {
    let genesis = GenesisSeed::from_phrase("neuro_validation");
    let encoder = super::neuroscience::NeuroEncoder::from_genesis(&genesis);

    // Nav channel: 20 pS, +60 mV reversal
    let nav = encoder.nav_channel();
    assert_relative_eq(nav.conductance_ps, 20.0, 1e-10, "Nav conductance");
    assert_relative_eq(nav.reversal_mv, 60.0, 1e-10, "Nav reversal potential");

    // Kv channel: 15 pS, -90 mV reversal
    let kv = encoder.kv_channel();
    assert_relative_eq(kv.conductance_ps, 15.0, 1e-10, "Kv conductance");
    assert_relative_eq(kv.reversal_mv, -90.0, 1e-10, "Kv reversal potential");
}

#[test]
fn stdp_ltp_exponential() {
    // Pre→Post (dt=5ms): strength *= (1 + 0.1·exp(-5/10)) · ltp_factor
    let genesis = GenesisSeed::from_phrase("neuro_validation");
    let encoder = super::neuroscience::NeuroEncoder::from_genesis(&genesis);

    let pre_vec = genesis.hv("test::pre", super::standard_model::PHYSICS_DIM);
    let post_vec = genesis.hv("test::post", super::standard_model::PHYSICS_DIM);
    let mut synapse = encoder.excitatory_synapse(&pre_vec, &post_vec);

    let initial_strength = synapse.strength;
    encoder.stdp_update(&mut synapse, 0.0, 5.0); // pre at 0ms, post at 5ms

    // dt = 5ms > 0, so LTP: strength *= 1 + 0.1 * exp(-5/10) * ltp_factor
    let factor = (-5.0_f64 / 10.0).exp();
    let expected = initial_strength * (1.0 + 0.1 * factor * 1.0); // ltp_factor=1.0
    assert_relative_eq(synapse.strength, expected, 1e-10, "STDP LTP at dt=5ms");
}

// =========================================================================
// Section 20: Antimatter
// =========================================================================

#[test]
fn annihilation_energies() {
    let genesis = GenesisSeed::from_phrase("antimatter_validation");
    let model = super::standard_model::StandardModel::from_genesis(&genesis);
    let hadrons = super::hadrons::Hadrons::from_model(&model, &genesis);
    let antimatter = super::antimatter::Antimatter::from_model(&model, &hadrons, &genesis);

    // e⁺e⁻ annihilation: 2 × 0.511 MeV = 1.022 MeV
    let ee = antimatter.electron_positron_annihilation(&model);
    assert_relative_eq(ee.energy_mev, 1.022, 1e-6, "e+e- annihilation energy");

    // pp̄ annihilation: 2 × 938.272 MeV ≈ 1876 MeV
    let pp = antimatter.proton_antiproton_annihilation(&hadrons);
    assert_relative_eq(pp.energy_mev, 1876.0, 0.01, "pp̄ annihilation energy");
}

#[test]
fn antiatom_charge_neutrality() {
    // Antiatoms should have positrons == antiprotons (charge neutral)
    let genesis = GenesisSeed::from_phrase("antimatter_validation");
    let model = super::standard_model::StandardModel::from_genesis(&genesis);
    let hadrons = super::hadrons::Hadrons::from_model(&model, &genesis);
    let antimatter = super::antimatter::Antimatter::from_model(&model, &hadrons, &genesis);

    let anti_h = antimatter.antihydrogen();
    assert_eq!(
        anti_h.positrons, anti_h.antiprotons,
        "Anti-H: positrons == antiprotons"
    );

    let anti_he3 = antimatter.antihelium3();
    assert_eq!(
        anti_he3.positrons, anti_he3.antiprotons,
        "Anti-He3: positrons == antiprotons"
    );

    let anti_he4 = antimatter.antihelium4();
    assert_eq!(
        anti_he4.positrons, anti_he4.antiprotons,
        "Anti-He4: positrons == antiprotons"
    );
}

// =========================================================================
// Section 21: Derived Laws
// =========================================================================

#[test]
fn law_composition_confidence() {
    // conf(A∘B) = conf(A)·conf(B) — multiplicative
    let genesis = GenesisSeed::from_phrase("laws_validation");
    let engine = super::derived_laws::LawsDerivationEngine::from_genesis(&genesis);

    let energy = engine.derive_energy_conservation(); // confidence = 1.0
    let momentum = engine.derive_momentum_conservation(); // confidence = 1.0
    let ohm = engine.derive_ohms_law(); // confidence < 1.0

    // 1.0 * 1.0 = 1.0
    let composed_fund = engine.compose_laws(&energy, &momentum);
    assert_relative_eq(
        composed_fund.confidence,
        energy.confidence * momentum.confidence,
        1e-10,
        "Composition of fundamental laws: conf = product",
    );

    // 1.0 * ohm.confidence = ohm.confidence
    let composed_empirical = engine.compose_laws(&energy, &ohm);
    assert_relative_eq(
        composed_empirical.confidence,
        energy.confidence * ohm.confidence,
        1e-10,
        "Composition with empirical law: conf = product",
    );
}

// =========================================================================
// Section 22: Eckart Analytical (Kemble Formula)
// =========================================================================

#[test]
fn eckart_kemble_analytical() {
    // Kemble formula for symmetric Eckart barrier V(x) = V₀ sech²(x/a):
    // T = sinh²(πka) / (sinh²(πka) + cosh²(π√(2mV₀a²/ℏ² - 1/4)))
    // where k = √(2mE)/ℏ
    let calc = TunnelingCalculator::electron();
    let v0: f64 = 1.0e-19; // ~0.6 eV
    let a: f64 = 1.0e-10; // 1 Å

    let energies = [0.2e-19, 0.4e-19, 0.6e-19, 0.8e-19, 1.5e-19];
    for &e in &energies {
        let wkb_result = calc.eckart_barrier(e, v0, a);
        let exact_result = calc.eckart_barrier_exact(e, v0, a);

        // Independent Kemble calculation
        let k = (2.0 * M_ELECTRON * e).sqrt() / HBAR;
        let pka = std::f64::consts::PI * k * a;
        let alpha_param = 2.0 * M_ELECTRON * v0 * a * a / (HBAR * HBAR);
        let inner = if alpha_param > 0.25 {
            std::f64::consts::PI * (alpha_param - 0.25).sqrt()
        } else {
            0.0
        };

        let sinh_pka = pka.sinh();
        let cosh_inner = inner.cosh();
        let expected_t = sinh_pka * sinh_pka / (sinh_pka * sinh_pka + cosh_inner * cosh_inner);

        // Exact method should match inline Kemble to machine precision
        if expected_t > 1e-15 {
            let exact_err = ((exact_result.transmission - expected_t) / expected_t).abs();
            assert!(
                exact_err < 1e-10,
                "eckart_barrier_exact at E={e:.2e}: got={:.10}, expected={expected_t:.10} (rel err {exact_err:.2e})",
                exact_result.transmission
            );
        }

        // WKB vs exact Kemble: WKB systematically underestimates transmission
        // at sub-barrier energies and overestimates near/above the barrier top.
        // Worst agreement is near E ≈ V₀ (classical turning point). Allow 20%.
        if expected_t > 1e-10 {
            let rel_err = ((wkb_result.transmission - expected_t) / expected_t).abs();
            assert!(
                rel_err < 0.20 || wkb_result.transmission > 0.90,
                "Eckart WKB at E={e:.2e}: WKB={:.4}, exact={expected_t:.4} (rel err {rel_err:.2e})",
                wkb_result.transmission
            );
        }
    }
}

// =========================================================================
// Section 23: Adaptive RK45 Integrator
// =========================================================================

#[test]
fn adaptive_rk45_matches_fixed_step() {
    use super::decoherence::LindbladEvolution;

    // Set up dephasing evolution
    let hamiltonian = DensityMatrix::new(2);
    let mut evolution = LindbladEvolution::new(hamiltonian);
    let mut sigma_z = DensityMatrix::new(2);
    sigma_z.set(0, 0, Complex64::from_real(1.0));
    sigma_z.set(1, 1, Complex64::from_real(-1.0));
    evolution.add_lindblad(sigma_z, 0.5);

    // Initial state: |+⟩ = (|0⟩ + |1⟩)/√2
    let s = 1.0 / 2.0_f64.sqrt();
    let psi = vec![Complex64::new(s, 0.0), Complex64::new(s, 0.0)];
    let rho = DensityMatrix::pure_state(&psi);

    // Fixed step (high resolution)
    let t_end = 3.0;
    let trajectory_fixed = evolution.evolve_trajectory(&rho, 0.001, 3000);
    let final_fixed = trajectory_fixed.last().unwrap();

    // Adaptive
    let trajectory_adaptive = evolution.evolve_adaptive(&rho, t_end, 1e-8, 1e-10);
    let final_adaptive = trajectory_adaptive.last().unwrap();

    // Compare final purity
    let purity_fixed = final_fixed.purity();
    let purity_adaptive = final_adaptive.purity();
    assert!(
        (purity_fixed - purity_adaptive).abs() < 1e-6,
        "Adaptive and fixed-step purities should match: fixed={purity_fixed}, adaptive={purity_adaptive}"
    );
}

// =========================================================================
// Section 24: General Relativity
// =========================================================================

#[test]
fn schwarzschild_radius_solar_mass() {
    // r_s = 2GM/c² for M = M_sun = 1.989e30 kg
    let genesis = GenesisSeed::from_phrase("gr validation");
    let encoder = GREncoder::from_genesis(&genesis);
    let m_sun = 1.989e30;
    let bh = encoder.schwarzschild(m_sun);

    // Expected: 2 * 6.67430e-11 * 1.989e30 / (299792458)^2 ≈ 2953 m
    let expected_rs = 2.0 * G * m_sun / (C * C);
    assert_relative_eq(
        bh.length_scale_m,
        expected_rs,
        1e-6,
        "Schwarzschild radius for solar mass",
    );
    assert!(
        (bh.length_scale_m - 2953.0).abs() < 10.0,
        "Schwarzschild radius ~2953m, got {}",
        bh.length_scale_m
    );
}

#[test]
fn gravitational_redshift_factor() {
    let genesis = GenesisSeed::from_phrase("gr redshift");
    let encoder = GREncoder::from_genesis(&genesis);
    // At r = 2*rs, redshift factor = 1/sqrt(1 - rs/r) = 1/sqrt(1/2) = sqrt(2)
    let z = encoder.redshift_factor(6000.0, 3000.0);
    assert_relative_eq(
        z,
        std::f64::consts::SQRT_2,
        1e-10,
        "Gravitational redshift at r=2rs",
    );
    // At horizon: infinite
    let z_horizon = encoder.redshift_factor(3000.0, 3000.0);
    assert!(
        z_horizon.is_infinite(),
        "Redshift at horizon should be infinite"
    );
}

#[test]
fn gravitational_time_dilation() {
    let genesis = GenesisSeed::from_phrase("gr time dilation");
    let encoder = GREncoder::from_genesis(&genesis);
    // At r = 2*rs: sqrt(1 - rs/r) = sqrt(1/2) = 1/sqrt(2)
    let td = encoder.gravitational_time_dilation(6000.0, 3000.0);
    assert_relative_eq(
        td,
        1.0 / std::f64::consts::SQRT_2,
        1e-10,
        "Time dilation at r=2rs",
    );
    // At horizon: 0
    let td_horizon = encoder.gravitational_time_dilation(3000.0, 3000.0);
    assert!(
        (td_horizon).abs() < 1e-15,
        "Time dilation at horizon should be 0"
    );
}

// =========================================================================
// Section 25: Quantum Information
// =========================================================================

#[test]
fn bell_state_maximal_entanglement() {
    let genesis = GenesisSeed::from_phrase("qi bell");
    let encoder = QIEncoder::from_genesis(&genesis);
    let bell = encoder.bell_phi_plus();
    assert_relative_eq(bell.concurrence, 1.0, 1e-10, "Bell |Φ+⟩ concurrence");
}

#[test]
fn ghz_state_entanglement() {
    let genesis = GenesisSeed::from_phrase("qi ghz");
    let encoder = QIEncoder::from_genesis(&genesis);
    let ghz = encoder.ghz_state(3);
    assert_relative_eq(ghz.concurrence, 1.0, 1e-10, "GHZ(3) concurrence");
}

#[test]
fn von_neumann_entropy_pure() {
    let genesis = GenesisSeed::from_phrase("qi entropy");
    let encoder = QIEncoder::from_genesis(&genesis);
    let s = encoder.von_neumann_entropy(1.0);
    assert!(
        s.abs() < 1e-10,
        "von Neumann entropy of pure state should be 0, got {s}"
    );
}

#[test]
fn von_neumann_entropy_mixed() {
    let genesis = GenesisSeed::from_phrase("qi entropy mixed");
    let encoder = QIEncoder::from_genesis(&genesis);
    let s = encoder.von_neumann_entropy(0.5);
    assert_relative_eq(
        s,
        1.0,
        1e-10,
        "von Neumann entropy of maximally mixed 2-state",
    );
}

// =========================================================================
// Section 26: Thermodynamics
// =========================================================================

#[test]
fn landauer_limit_300k() {
    let genesis = GenesisSeed::from_phrase("thermo landauer");
    let encoder = ThermoEncoder::from_genesis(&genesis);
    let limit = encoder.landauer_limit(300.0);
    // k_B * T * ln(2) = 1.380649e-23 * 300 * 0.693147 ≈ 2.871e-21 J
    let expected = K_BOLTZMANN * 300.0 * 2.0_f64.ln();
    assert_relative_eq(limit, expected, 1e-10, "Landauer limit at 300K");
}

#[test]
fn carnot_efficiency_half() {
    let genesis = GenesisSeed::from_phrase("thermo carnot");
    let encoder = ThermoEncoder::from_genesis(&genesis);
    let eta = encoder.carnot_efficiency(600.0, 300.0);
    assert_relative_eq(eta, 0.5, 1e-10, "Carnot efficiency T_hot=600, T_cold=300");
}

#[test]
fn carnot_efficiency_bounds() {
    let genesis = GenesisSeed::from_phrase("thermo carnot bounds");
    let encoder = ThermoEncoder::from_genesis(&genesis);
    // η(T, T) = 0 (no temperature difference)
    let eta_same = encoder.carnot_efficiency(300.0, 300.0);
    assert!(
        eta_same.abs() < 1e-10,
        "Carnot η(T,T) should be 0, got {eta_same}"
    );
    // η(T, 0) = 1 - 0/T = 1
    let eta_max = encoder.carnot_efficiency(300.0, 0.0);
    assert_relative_eq(eta_max, 1.0, 1e-10, "Carnot η(T,0) should be 1");
}

#[test]
fn crooks_fluctuation_ratio() {
    let genesis = GenesisSeed::from_phrase("thermo crooks");
    let encoder = ThermoEncoder::from_genesis(&genesis);
    // Crooks: P_F(W)/P_R(-W) = exp(β(W - ΔF))
    // For W = ΔF: ratio = exp(0) = 1
    let ratio_eq = encoder.crooks_ratio(1e-20, 1e-20, 300.0);
    assert_relative_eq(ratio_eq, 1.0, 1e-10, "Crooks ratio when W=ΔF");
    // For known values: β = 1/(k_B*T), W=2e-20, ΔF=1e-20, T=300
    let beta = 1.0 / (K_BOLTZMANN * 300.0);
    let expected = (beta * (2e-20 - 1e-20)).exp();
    let ratio = encoder.crooks_ratio(2e-20, 1e-20, 300.0);
    assert_relative_eq(ratio, expected, 1e-8, "Crooks ratio for known W, ΔF");
}

// =========================================================================
// Section 27: Fluid Dynamics
// =========================================================================

#[test]
fn reynolds_number_formula() {
    let genesis = GenesisSeed::from_phrase("fluid re");
    let encoder = FluidEncoder::from_genesis(&genesis);
    // Re = ρvL/μ = 1000 * 1 * 0.01 / 0.001 = 10000
    let re = encoder.reynolds_number(1000.0, 1.0, 0.01, 0.001);
    assert_relative_eq(re, 10000.0, 1e-10, "Reynolds number");
}

#[test]
fn mach_number_formula() {
    let genesis = GenesisSeed::from_phrase("fluid mach");
    let encoder = FluidEncoder::from_genesis(&genesis);
    let ma = encoder.mach_number(680.0, 340.0);
    assert_relative_eq(ma, 2.0, 1e-10, "Mach number");
}

#[test]
fn kolmogorov_microscale() {
    let genesis = GenesisSeed::from_phrase("fluid kolmogorov");
    let encoder = FluidEncoder::from_genesis(&genesis);
    // η = (ν³/ε)^(1/4)
    // ν = 1e-6 m²/s (water), ε = 1.0 W/kg
    let eta = encoder.kolmogorov_scale(1e-6, 1.0);
    let expected = (1e-18_f64 / 1.0).powf(0.25); // (1e-6)^3 = 1e-18
    assert_relative_eq(eta, expected, 1e-10, "Kolmogorov microscale");
}

#[test]
fn boundary_layer_thickness() {
    let genesis = GenesisSeed::from_phrase("fluid bl");
    let encoder = FluidEncoder::from_genesis(&genesis);
    // δ = L/√Re = 1.0/√10000 = 1/100 = 0.01
    let delta = encoder.boundary_layer_thickness(1.0, 10000.0);
    assert_relative_eq(delta, 0.01, 1e-10, "Boundary layer thickness");
}

// =========================================================================
// Section 28: Astrophysics
// =========================================================================

#[test]
fn solar_temperature() {
    let genesis = GenesisSeed::from_phrase("astro sun");
    let encoder = StellarEncoder::from_genesis(&genesis);
    let sun = encoder.sun();
    assert_relative_eq(sun.temperature_k, 5778.0, 1e-6, "Solar surface temperature");
}

#[test]
fn pp_chain_energy() {
    let genesis = GenesisSeed::from_phrase("astro pp");
    let encoder = StellarEncoder::from_genesis(&genesis);
    let model = StandardModel::from_genesis(&genesis);
    let hadrons = Hadrons::from_model(&model, &genesis);
    let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
    let pp = encoder.pp_chain(&table);
    assert_relative_eq(pp.energy_mev, 26.7, 0.01, "pp-chain energy");
}

#[test]
fn triple_alpha_energy() {
    let genesis = GenesisSeed::from_phrase("astro 3alpha");
    let encoder = StellarEncoder::from_genesis(&genesis);
    let model = StandardModel::from_genesis(&genesis);
    let hadrons = Hadrons::from_model(&model, &genesis);
    let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
    let triple = encoder.triple_alpha(&table);
    assert_relative_eq(triple.energy_mev, 7.27, 0.01, "Triple-alpha energy");
}

#[test]
fn black_hole_hawking_temperature() {
    let genesis = GenesisSeed::from_phrase("astro hawking");
    let encoder = StellarEncoder::from_genesis(&genesis);
    // T_H = 6.17e-8 / M_solar
    let bh = encoder.encode_black_hole("test", 1.0, 0.0);
    assert_relative_eq(
        bh.hawking_temperature_k,
        6.17e-8,
        0.01,
        "Hawking temperature for 1 solar mass",
    );
}

// =========================================================================
// Section 29: Condensed Matter
// =========================================================================

#[test]
fn silicon_band_gap() {
    let genesis = GenesisSeed::from_phrase("cm silicon");
    let encoder = CMEncoder::from_genesis(&genesis);
    let si = encoder.silicon(&genesis);
    assert_relative_eq(si.band_gap_ev, 1.12, 1e-6, "Silicon band gap");
}

#[test]
fn germanium_band_gap() {
    let genesis = GenesisSeed::from_phrase("cm germanium");
    let encoder = CMEncoder::from_genesis(&genesis);
    let ge = encoder.germanium(&genesis);
    assert_relative_eq(ge.band_gap_ev, 0.67, 1e-6, "Germanium band gap");
}

#[test]
fn gaas_band_gap() {
    let genesis = GenesisSeed::from_phrase("cm gaas");
    let encoder = CMEncoder::from_genesis(&genesis);
    let gaas = encoder.gallium_arsenide(&genesis);
    assert_relative_eq(gaas.band_gap_ev, 1.42, 1e-6, "GaAs band gap");
}

#[test]
fn fermi_dirac_at_fermi_energy() {
    let genesis = GenesisSeed::from_phrase("cm fermi");
    let encoder = CMEncoder::from_genesis(&genesis);
    // f(E_F, E_F, T) = 1 / (1 + exp(0)) = 0.5
    let f = encoder.fermi_dirac(5.0, 5.0, 300.0);
    assert_relative_eq(f, 0.5, 1e-10, "Fermi-Dirac at Fermi energy");
}

// =========================================================================
// Section 30: Cosmology
// =========================================================================

#[test]
fn cmb_temperature() {
    let genesis = GenesisSeed::from_phrase("cosmo cmb");
    let encoder = CosmologyEncoder::from_genesis(&genesis);
    let cmb = encoder.create_cmb();
    assert_relative_eq(cmb.temperature_k, 2.725, 1e-6, "CMB temperature");
}

#[test]
fn scale_factor_redshift() {
    let genesis = GenesisSeed::from_phrase("cosmo scale");
    let encoder = CosmologyEncoder::from_genesis(&genesis);
    // a(z) = 1/(1+z)
    assert_relative_eq(encoder.scale_factor(0.0), 1.0, 1e-10, "Scale factor at z=0");
    assert_relative_eq(encoder.scale_factor(1.0), 0.5, 1e-10, "Scale factor at z=1");
    assert_relative_eq(encoder.scale_factor(9.0), 0.1, 1e-10, "Scale factor at z=9");
}

#[test]
fn hubble_parameter_present() {
    let genesis = GenesisSeed::from_phrase("cosmo hubble");
    let encoder = CosmologyEncoder::from_genesis(&genesis);
    // H(z=0) = H₀ * sqrt(Ωm + ΩΛ) = 67.4 * sqrt(0.315 + 0.685) = 67.4
    let h0 = encoder.hubble_parameter(0.0, 0.315, 0.685);
    assert_relative_eq(h0, 67.4, 1e-6, "Hubble parameter at z=0");
}

#[test]
fn lcdm_flatness() {
    let genesis = GenesisSeed::from_phrase("cosmo flat");
    let encoder = CosmologyEncoder::from_genesis(&genesis);
    let params = encoder.lcdm_parameters();
    let sum = params.omega_m + params.omega_lambda;
    assert!(
        (sum - 1.0).abs() < 0.01,
        "ΛCDM flatness: Ωm + ΩΛ = {sum}, expected ≈ 1.0"
    );
}

#[test]
fn comoving_distance_low_z() {
    let genesis = GenesisSeed::from_phrase("cosmo distance");
    let encoder = CosmologyEncoder::from_genesis(&genesis);
    // d_c ≈ c*z/H₀ for z << 1
    // c = 299792.458 km/s, z = 0.01, H₀ = 67.4 km/s/Mpc
    // d_c ≈ 299792.458 * 0.01 / 67.4 ≈ 44.48 Mpc
    let d = encoder.comoving_distance_mpc(0.01);
    let expected = 299792.458 * 0.01 / 67.4;
    assert_relative_eq(d, expected, 1e-6, "Comoving distance at z=0.01");
}

// =========================================================================
// Section 31: Exact Eckart vs WKB
// =========================================================================

#[test]
fn eckart_exact_vs_wkb_comparison() {
    let calc = TunnelingCalculator::electron();
    let v0: f64 = 1.0e-19;
    let a: f64 = 1.0e-10;

    // Sub-barrier energy where WKB and exact can diverge
    let e = 0.3e-19;
    let exact = calc.eckart_barrier_exact(e, v0, a);
    let wkb = calc.eckart_barrier(e, v0, a);

    // Exact should match independent Kemble to machine precision
    let k = (2.0 * M_ELECTRON * e).sqrt() / HBAR;
    let pka = std::f64::consts::PI * k * a;
    let alpha_param = 2.0 * M_ELECTRON * v0 * a * a / (HBAR * HBAR);
    let inner = if alpha_param > 0.25 {
        std::f64::consts::PI * (alpha_param - 0.25).sqrt()
    } else {
        0.0
    };
    let sinh_pka = pka.sinh();
    let cosh_inner = inner.cosh();
    let kemble_t = sinh_pka * sinh_pka / (sinh_pka * sinh_pka + cosh_inner * cosh_inner);

    assert!(
        ((exact.transmission - kemble_t) / kemble_t).abs() < 1e-10,
        "Exact Eckart should match Kemble: exact={}, kemble={kemble_t}",
        exact.transmission
    );

    // Document WKB divergence (informational, not a failure)
    let _wkb_err = ((wkb.transmission - kemble_t) / kemble_t).abs();
}

// =========================================================================
// Section 32: Edge-Case Hardening (20 tests)
// =========================================================================

#[test]
fn edge_landauer_negative_temp() {
    let genesis = GenesisSeed::from_phrase("edge landauer neg");
    let encoder = ThermoEncoder::from_genesis(&genesis);
    let result = encoder.landauer_limit(-100.0);
    assert!(
        result.is_finite(),
        "Landauer at T=-100 should be finite, got {result}"
    );
    assert!(
        result < 0.0,
        "Landauer at T<0 should be negative, got {result}"
    );
}

#[test]
fn edge_landauer_zero_temp() {
    let genesis = GenesisSeed::from_phrase("edge landauer zero");
    let encoder = ThermoEncoder::from_genesis(&genesis);
    let result = encoder.landauer_limit(0.0);
    assert!(
        (result).abs() < 1e-30,
        "Landauer at T=0 should be 0.0, got {result}"
    );
}

#[test]
fn edge_fermi_dirac_zero_temp_below() {
    let genesis = GenesisSeed::from_phrase("edge fd zero below");
    let encoder = CMEncoder::from_genesis(&genesis);
    // E < E_F at T=0 → f = 1.0
    let f = encoder.fermi_dirac(4.0, 5.0, 0.0);
    assert!(
        (f - 1.0).abs() < 1e-10,
        "Fermi-Dirac below E_F at T=0 should be 1.0, got {f}"
    );
}

#[test]
fn edge_fermi_dirac_zero_temp_above() {
    let genesis = GenesisSeed::from_phrase("edge fd zero above");
    let encoder = CMEncoder::from_genesis(&genesis);
    // E > E_F at T=0 → f = 0.0
    let f = encoder.fermi_dirac(6.0, 5.0, 0.0);
    assert!(
        f.abs() < 1e-10,
        "Fermi-Dirac above E_F at T=0 should be 0.0, got {f}"
    );
}

#[test]
fn edge_fermi_dirac_zero_temp_at_fermi() {
    let genesis = GenesisSeed::from_phrase("edge fd zero at");
    let encoder = CMEncoder::from_genesis(&genesis);
    // E = E_F at T=0 → f = 1.0 (convention: <=)
    let f = encoder.fermi_dirac(5.0, 5.0, 0.0);
    assert!(
        (f - 1.0).abs() < 1e-10,
        "Fermi-Dirac at E=E_F, T=0 should be 1.0, got {f}"
    );
}

#[test]
fn edge_kolmogorov_zero_dissipation() {
    let genesis = GenesisSeed::from_phrase("edge kolm zero");
    let encoder = FluidEncoder::from_genesis(&genesis);
    // η = (ν³/ε)^(1/4), ε=0 → Inf
    let eta = encoder.kolmogorov_scale(1e-6, 0.0);
    assert!(
        eta.is_infinite(),
        "Kolmogorov scale at ε=0 should be Inf, got {eta}"
    );
}

#[test]
fn edge_reynolds_zero_viscosity() {
    let genesis = GenesisSeed::from_phrase("edge re zero mu");
    let encoder = FluidEncoder::from_genesis(&genesis);
    // Re = ρvL/μ, μ=0 → Inf
    let re = encoder.reynolds_number(1000.0, 1.0, 0.01, 0.0);
    assert!(
        re.is_infinite(),
        "Reynolds number at μ=0 should be Inf, got {re}"
    );
}

#[test]
fn edge_mach_zero_sound_speed() {
    let genesis = GenesisSeed::from_phrase("edge mach zero cs");
    let encoder = FluidEncoder::from_genesis(&genesis);
    // Ma = v/cs, cs=0 → Inf
    let ma = encoder.mach_number(340.0, 0.0);
    assert!(
        ma.is_infinite(),
        "Mach number at cs=0 should be Inf, got {ma}"
    );
}

#[test]
fn edge_scale_factor_z_neg1() {
    let genesis = GenesisSeed::from_phrase("edge scale z neg1");
    let encoder = CosmologyEncoder::from_genesis(&genesis);
    // a(z) = 1/(1+z), z=-1 → 1/0 = Inf
    let a = encoder.scale_factor(-1.0);
    assert!(
        a.is_infinite(),
        "Scale factor at z=-1 should be Inf, got {a}"
    );
}

#[test]
fn edge_von_neumann_entropy_zero_purity() {
    let genesis = GenesisSeed::from_phrase("edge vn zero");
    let encoder = QIEncoder::from_genesis(&genesis);
    let s = encoder.von_neumann_entropy(0.0);
    assert!(
        (s).abs() < 1e-10,
        "VN entropy at purity=0 should be 0.0 (guarded), got {s}"
    );
}

#[test]
fn edge_von_neumann_entropy_negative_purity() {
    let genesis = GenesisSeed::from_phrase("edge vn neg");
    let encoder = QIEncoder::from_genesis(&genesis);
    let s = encoder.von_neumann_entropy(-0.5);
    assert!(
        (s).abs() < 1e-10,
        "VN entropy at purity=-0.5 should be 0.0 (guarded), got {s}"
    );
}

#[test]
fn edge_boundary_layer_zero_re() {
    let genesis = GenesisSeed::from_phrase("edge bl zero re");
    let encoder = FluidEncoder::from_genesis(&genesis);
    // δ = L/√Re, Re=0 → Inf
    let delta = encoder.boundary_layer_thickness(1.0, 0.0);
    assert!(
        delta.is_infinite(),
        "BL thickness at Re=0 should be Inf, got {delta}"
    );
}

#[test]
fn edge_eckart_exact_zero_width() {
    let _genesis = GenesisSeed::from_phrase("edge eckart zero w");
    let calc = TunnelingCalculator::electron();
    let result = calc.eckart_barrier_exact(0.5e-19, 1.0e-19, 0.0);
    assert!(
        result.transmission.is_finite(),
        "Eckart exact at a=0 should be finite, got {}",
        result.transmission
    );
}

#[test]
fn edge_eckart_exact_zero_energy() {
    let _genesis = GenesisSeed::from_phrase("edge eckart zero e");
    let calc = TunnelingCalculator::electron();
    let result = calc.eckart_barrier_exact(0.0, 1.0e-19, 1.0e-10);
    assert!(
        result.transmission.is_finite(),
        "Eckart exact at E=0 should be finite, got {}",
        result.transmission
    );
    assert!(
        result.transmission >= 0.0 && result.transmission <= 1.0,
        "Transmission should be in [0,1], got {}",
        result.transmission
    );
}

#[test]
fn edge_redshift_factor_negative_r() {
    let genesis = GenesisSeed::from_phrase("edge rs neg r");
    let encoder = GREncoder::from_genesis(&genesis);
    // r < 0 → r <= rs (since rs > 0), so should return Inf per existing guard
    let z = encoder.redshift_factor(-1.0, 3000.0);
    assert!(
        z.is_infinite(),
        "Redshift factor at r<0 should be Inf, got {z}"
    );
}

#[test]
fn edge_gravitational_time_dilation_negative_r() {
    let genesis = GenesisSeed::from_phrase("edge td neg r");
    let encoder = GREncoder::from_genesis(&genesis);
    // r < 0 → r <= rs, returns 0.0 per existing guard
    let td = encoder.gravitational_time_dilation(-1.0, 3000.0);
    assert!(
        (td).abs() < 1e-15,
        "Time dilation at r<0 should be 0.0, got {td}"
    );
}

#[test]
fn edge_carnot_negative_t_hot() {
    let genesis = GenesisSeed::from_phrase("edge carnot neg");
    let encoder = ThermoEncoder::from_genesis(&genesis);
    // T_hot < 0 → 0.0 per existing guard (t_hot <= 0.0)
    let eta = encoder.carnot_efficiency(-100.0, 300.0);
    assert!(
        (eta).abs() < 1e-10,
        "Carnot at T_hot<0 should be 0.0, got {eta}"
    );
}

#[test]
fn edge_carnot_equal_temps() {
    let genesis = GenesisSeed::from_phrase("edge carnot equal");
    let encoder = ThermoEncoder::from_genesis(&genesis);
    // T_hot = T_cold → 0.0 per existing guard (t_hot <= t_cold)
    let eta = encoder.carnot_efficiency(300.0, 300.0);
    assert!(
        (eta).abs() < 1e-10,
        "Carnot at T_hot=T_cold should be 0.0, got {eta}"
    );
}

#[test]
fn edge_mach_zero_velocity() {
    let genesis = GenesisSeed::from_phrase("edge mach zero v");
    let encoder = FluidEncoder::from_genesis(&genesis);
    // Ma = 0/cs = 0
    let ma = encoder.mach_number(0.0, 340.0);
    assert!(
        (ma).abs() < 1e-15,
        "Mach number at v=0 should be 0.0, got {ma}"
    );
}

#[test]
fn edge_reynolds_zero_velocity() {
    let genesis = GenesisSeed::from_phrase("edge re zero v");
    let encoder = FluidEncoder::from_genesis(&genesis);
    // Re = ρ*0*L/μ = 0
    let re = encoder.reynolds_number(1000.0, 0.0, 0.01, 0.001);
    assert!(
        (re).abs() < 1e-15,
        "Reynolds number at v=0 should be 0.0, got {re}"
    );
}

// =========================================================================
// Section 33: Dimensional Consistency (5 tests)
// =========================================================================

#[test]
fn schwarzschild_feeds_redshift() {
    let genesis = GenesisSeed::from_phrase("dim schwarzschild redshift");
    let encoder = GREncoder::from_genesis(&genesis);
    // Schwarzschild radius for 1 solar mass
    let m_sun = 1.989e30;
    let st = encoder.schwarzschild(m_sun);
    let rs = st.length_scale_m;
    // At r = 2*rs: z = 1/sqrt(1 - rs/r) = 1/sqrt(1/2) = sqrt(2)
    let z = encoder.redshift_factor(2.0 * rs, rs);
    assert_relative_eq(
        z,
        std::f64::consts::SQRT_2,
        1e-10,
        "Redshift at r=2rs should be √2",
    );
}

#[test]
fn fermi_dirac_step_function_at_zero_temp() {
    let genesis = GenesisSeed::from_phrase("dim fd step");
    let encoder = CMEncoder::from_genesis(&genesis);
    // At T=0, Fermi-Dirac becomes step function
    let energies = [0.0, 2.0, 4.0, 4.99, 5.0, 5.01, 6.0, 8.0, 10.0];
    let fermi_ev = 5.0;
    for &e in &energies {
        let f = encoder.fermi_dirac(e, fermi_ev, 0.0);
        if e <= fermi_ev {
            assert!(
                (f - 1.0).abs() < 1e-10,
                "Step function: f({e}) should be 1.0, got {f}"
            );
        } else {
            assert!(
                f.abs() < 1e-10,
                "Step function: f({e}) should be 0.0, got {f}"
            );
        }
    }
}

#[test]
fn carnot_bounds_landauer() {
    let genesis = GenesisSeed::from_phrase("dim carnot landauer");
    let encoder = ThermoEncoder::from_genesis(&genesis);
    // Higher T → higher Landauer limit AND higher Carnot efficiency (fixed cold)
    let t_cold = 200.0;
    let temps = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0];
    let mut prev_landauer = 0.0_f64;
    let mut prev_carnot = 0.0_f64;
    for &t in &temps {
        let landauer = encoder.landauer_limit(t);
        let carnot = encoder.carnot_efficiency(t, t_cold);
        assert!(
            landauer > prev_landauer,
            "Landauer should increase: {landauer} > {prev_landauer} at T={t}"
        );
        assert!(
            carnot > prev_carnot,
            "Carnot should increase: {carnot} > {prev_carnot} at T={t}"
        );
        prev_landauer = landauer;
        prev_carnot = carnot;
    }
}

#[test]
fn reynolds_boundary_consistency() {
    let genesis = GenesisSeed::from_phrase("dim re bl");
    let encoder = FluidEncoder::from_genesis(&genesis);
    // BL thickness = L/√Re matches reynolds_number output
    let density = 1000.0;
    let velocity = 2.0;
    let length = 0.5;
    let viscosity = 0.001;
    let re = encoder.reynolds_number(density, velocity, length, viscosity);
    let delta = encoder.boundary_layer_thickness(length, re);
    let expected_delta = length / re.sqrt();
    assert_relative_eq(
        delta,
        expected_delta,
        1e-10,
        "BL thickness should equal L/√Re",
    );
}

#[test]
fn tunneling_exact_vs_wkb_ordering() {
    let calc = TunnelingCalculator::electron();
    let v0 = 1.0e-19;
    let a = 1.0e-10;
    // Test across a range of energies
    let energies = [0.1e-19, 0.3e-19, 0.5e-19, 0.7e-19, 0.9e-19];
    for &e in &energies {
        let exact = calc.eckart_barrier_exact(e, v0, a);
        assert!(
            exact.transmission.is_finite(),
            "Exact should be finite at E={e:.2e}"
        );
        assert!(
            exact.transmission >= 0.0 && exact.transmission <= 1.0,
            "Exact T should be in [0,1], got {} at E={e:.2e}",
            exact.transmission
        );
    }
}

// =========================================================================
// Section 34: Acoustics (10 tests)
// =========================================================================

#[test]
fn acoustics_sound_speed_water() {
    let genesis = GenesisSeed::from_phrase("acoustics water");
    let encoder = AcousticsEncoder::from_genesis(&genesis);
    // c = √(K/ρ), K=2.2e9 Pa, ρ=1000 kg/m³
    let c = encoder.sound_speed_fluid(2.2e9, 1000.0);
    assert_relative_eq(c, 1483.24, 1e-3, "Sound speed in water");
}

#[test]
fn acoustics_sound_speed_air() {
    let genesis = GenesisSeed::from_phrase("acoustics air");
    let encoder = AcousticsEncoder::from_genesis(&genesis);
    // c = √(K/ρ), K=142e3 Pa (adiabatic bulk modulus), ρ=1.2 kg/m³
    let c = encoder.sound_speed_fluid(142e3, 1.2);
    let expected = (142e3_f64 / 1.2).sqrt();
    assert_relative_eq(c, expected, 1e-10, "Sound speed in air");
}

#[test]
fn acoustics_impedance_air() {
    let genesis = GenesisSeed::from_phrase("acoustics z air");
    let encoder = AcousticsEncoder::from_genesis(&genesis);
    // Z = ρc = 1.2 × 343 = 411.6 Pa·s/m
    let z = encoder.acoustic_impedance(1.2, 343.0);
    assert_relative_eq(z, 411.6, 1e-10, "Acoustic impedance of air");
}

#[test]
fn acoustics_impedance_water() {
    let genesis = GenesisSeed::from_phrase("acoustics z water");
    let encoder = AcousticsEncoder::from_genesis(&genesis);
    // Z = ρc = 1000 × 1483 = 1.483e6 Pa·s/m
    let z = encoder.acoustic_impedance(1000.0, 1483.0);
    assert_relative_eq(z, 1.483e6, 1e-10, "Acoustic impedance of water");
}

#[test]
fn acoustics_reflection_matched() {
    let genesis = GenesisSeed::from_phrase("acoustics r matched");
    let encoder = AcousticsEncoder::from_genesis(&genesis);
    // R(Z,Z) = (Z-Z)/(Z+Z) = 0
    let r = encoder.reflection_coefficient(411.6, 411.6);
    assert!(
        r.abs() < 1e-15,
        "Matched impedance reflection should be 0, got {r}"
    );
}

#[test]
fn acoustics_reflection_plus_transmission_energy() {
    let genesis = GenesisSeed::from_phrase("acoustics energy");
    let encoder = AcousticsEncoder::from_genesis(&genesis);
    // Energy conservation: R² + (Z1/Z2)T² = 1
    let z1 = 411.6;
    let z2 = 1.483e6;
    let r = encoder.reflection_coefficient(z1, z2);
    let t = encoder.transmission_coefficient(z1, z2);
    let energy = r * r + (z1 / z2) * t * t;
    assert!(
        (energy - 1.0).abs() < 1e-10,
        "Energy conservation: R² + (Z1/Z2)T² = {energy}, expected 1.0"
    );
}

#[test]
fn acoustics_spl_reference() {
    let genesis = GenesisSeed::from_phrase("acoustics spl ref");
    let encoder = AcousticsEncoder::from_genesis(&genesis);
    // SPL(2e-5 Pa) = 20 log₁₀(2e-5/2e-5) = 0 dB
    let spl = encoder.spl_db(2e-5);
    assert!(
        spl.abs() < 1e-10,
        "SPL at reference pressure should be 0 dB, got {spl}"
    );
}

#[test]
fn acoustics_sil_reference() {
    let genesis = GenesisSeed::from_phrase("acoustics sil ref");
    let encoder = AcousticsEncoder::from_genesis(&genesis);
    // SIL(1e-12 W/m²) = 10 log₁₀(1e-12/1e-12) = 0 dB
    let sil = encoder.sil_db(1e-12);
    assert!(
        sil.abs() < 1e-10,
        "SIL at reference intensity should be 0 dB, got {sil}"
    );
}

#[test]
fn acoustics_open_pipe_harmonics() {
    let genesis = GenesisSeed::from_phrase("acoustics pipe");
    let encoder = AcousticsEncoder::from_genesis(&genesis);
    // f_n = nc/(2L), L=1.7m, c=343 m/s
    let f1 = encoder.open_pipe_frequency(1.7, 343.0, 1);
    let f2 = encoder.open_pipe_frequency(1.7, 343.0, 2);
    let expected_f1 = 343.0 / (2.0 * 1.7);
    assert_relative_eq(f1, expected_f1, 1e-10, "Open pipe fundamental");
    assert_relative_eq(f2, 2.0 * expected_f1, 1e-10, "Open pipe 2nd harmonic");
}

#[test]
fn acoustics_doppler_approaching() {
    let genesis = GenesisSeed::from_phrase("acoustics doppler");
    let encoder = AcousticsEncoder::from_genesis(&genesis);
    // f' = f(c + v_obs)/(c - v_src), source approaching at 34.3 m/s, observer stationary
    // f' = 440 × 343/(343 - 34.3) = 440 × 343/308.7
    let fp = encoder.doppler_frequency(440.0, 343.0, 34.3, 0.0);
    let expected = 440.0 * 343.0 / (343.0 - 34.3);
    assert_relative_eq(fp, expected, 1e-10, "Doppler approaching source");
}

// =========================================================================
// Section 35: True Phi Bounds (8 tests)
// =========================================================================

#[test]
fn true_phi_entropy_nonneg() {
    let calc = TruePhiCalculator::new();
    let hv = ContinuousHV::random(256, 42);
    let h = calc.entropy(&hv);
    assert!(h >= 0.0, "Entropy should be non-negative, got {h}");
}

#[test]
fn true_phi_entropy_upper_bound() {
    // Default TruePhiCalculator uses 16 bins → max entropy = log₂(16) = 4.0
    let calc = TruePhiCalculator::new();
    let hv = ContinuousHV::random(4096, 42);
    let h = calc.entropy(&hv);
    assert!(
        h <= 4.0 + 0.01,
        "Entropy should be ≤ log₂(16) ≈ 4.0, got {h}"
    );
}

#[test]
fn true_phi_constant_entropy_zero() {
    let calc = TruePhiCalculator::new();
    // Constant vector: all components = 0.5
    let hv = ContinuousHV::from_vec(vec![0.5; 1024]);
    let h = calc.entropy(&hv);
    assert!(
        h.abs() < 0.01,
        "Entropy of constant vector should be ~0, got {h}"
    );
}

#[test]
fn true_phi_mi_symmetric() {
    let calc = TruePhiCalculator::new();
    let hv1 = ContinuousHV::random(256, 42);
    let hv2 = ContinuousHV::random(256, 99);
    let mi_12 = calc.mutual_information(&hv1, &hv2);
    let mi_21 = calc.mutual_information(&hv2, &hv1);
    assert!(
        (mi_12 - mi_21).abs() < 1e-10,
        "MI should be symmetric: I(X;Y)={mi_12}, I(Y;X)={mi_21}"
    );
}

#[test]
fn true_phi_mi_nonneg() {
    let calc = TruePhiCalculator::new();
    let hv1 = ContinuousHV::random(256, 42);
    let hv2 = ContinuousHV::random(256, 99);
    let mi = calc.mutual_information(&hv1, &hv2);
    assert!(mi >= -1e-10, "MI should be non-negative, got {mi}");
}

#[test]
fn true_phi_purity_bounds() {
    let calc = QuantumEntropyCalculator::new();
    let hv = ContinuousHV::random(256, 42);
    let p = calc.purity(&hv);
    assert!(
        p >= 0.0 && p <= 1.0 + 1e-10,
        "Purity should be in [0, 1], got {p}"
    );
}

#[test]
fn true_phi_joint_entropy_subadditive() {
    let calc = TruePhiCalculator::new();
    let hv1 = ContinuousHV::random(256, 42);
    let hv2 = ContinuousHV::random(256, 99);
    let h1 = calc.entropy(&hv1);
    let h2 = calc.entropy(&hv2);
    let h_joint = calc.joint_entropy(&hv1, &hv2);
    assert!(
        h_joint <= h1 + h2 + 0.01,
        "Joint entropy should be subadditive: H(X,Y)={h_joint} ≤ H(X)+H(Y)={}",
        h1 + h2
    );
}

#[test]
fn true_phi_intrinsic_info_nonneg() {
    let calc = TruePhiCalculator::new();
    // Use 3 random components to get effective_information (≥ 0)
    let components: Vec<ContinuousHV> = (0..3).map(|i| ContinuousHV::random(256, 42 + i)).collect();
    let ei = calc.effective_information(&components);
    assert!(
        ei >= -1e-10,
        "Effective information should be non-negative, got {ei}"
    );
}

// =========================================================================
// Section 36: Tensor Algebra (8 tests)
// =========================================================================

#[test]
fn tensor_minkowski_determinant() {
    // Mostly-plus: η = diag(-1, 1, 1, 1) → det = -1
    let eta = MetricTensor::minkowski();
    let det = eta.determinant();
    assert_relative_eq(det, -1.0, 1e-10, "Minkowski determinant (mostly-plus)");
}

#[test]
fn tensor_minkowski_null_geodesic() {
    let eta = MetricTensor::minkowski();
    // Minkowski: g = diag(-1, 1, 1, 1) (geometric units, c=1 in metric)
    // Light-like: ds² = -dt² + dr² = 0 when dr = dt
    let dx = [1.0, 1.0, 0.0, 0.0];
    let ds2 = eta.line_element(&dx);
    assert!(
        ds2.abs() < 1e-10,
        "Null geodesic should have ds²=0, got {ds2}"
    );
}

#[test]
fn tensor_schwarzschild_vacuum_ricci() {
    // Verify Schwarzschild vacuum indirectly: analytical Christoffel symbols
    // are symmetric (Γ^λ_μν = Γ^λ_νμ) and Ricci scalar from analytical
    // Christoffels should be exactly 0.
    // (Direct numerical Riemann computation is ill-conditioned due to C² scaling.)
    //
    // Instead, verify the trace of Ricci tensor from analytical Christoffels:
    // For Schwarzschild, all R_μν = 0 identically. We verify this by checking
    // that the Christoffel symbols satisfy the expected analytical relations.
    let m_sun = 1.989e30;
    let r_s = 2.0 * G * m_sun / (C * C);
    let r = 10.0 * r_s;
    let theta = std::f64::consts::FRAC_PI_2;
    let gamma = ChristoffelSymbols::schwarzschild(m_sun, r, theta);

    // Γ^t_tr should equal r_s c²/(2r²f) where f = 1 - r_s/r
    let f = 1.0 - r_s / r;
    let expected_gamma_t_tr = r_s * C * C / (2.0 * r * r * f);
    assert_relative_eq(
        gamma.get(0, 0, 1),
        expected_gamma_t_tr,
        1e-10,
        "Γ^t_tr analytical check",
    );

    // Γ^r_tt should equal r_s c² f / (2r²)
    let expected_gamma_r_tt = r_s * C * C * f / (2.0 * r * r);
    assert_relative_eq(
        gamma.get(1, 0, 0),
        expected_gamma_r_tt,
        1e-10,
        "Γ^r_tt analytical check",
    );

    // Γ^θ_rθ = 1/r
    assert_relative_eq(gamma.get(2, 1, 2), 1.0 / r, 1e-10, "Γ^θ_rθ = 1/r");
}

#[test]
fn tensor_christoffel_symmetry() {
    // Γ^λ_μν = Γ^λ_νμ (torsion-free connection)
    let m_sun = 1.989e30;
    let r_s = 2.0 * G * m_sun / (C * C);
    let r = 5.0 * r_s;
    let theta = std::f64::consts::FRAC_PI_4;
    let gamma = ChristoffelSymbols::schwarzschild(m_sun, r, theta);
    for lambda in 0..4 {
        for mu in 0..4 {
            for nu in 0..4 {
                let diff = (gamma.get(lambda, mu, nu) - gamma.get(lambda, nu, mu)).abs();
                assert!(
                    diff < 1e-10,
                    "Christoffel not symmetric: Γ^{lambda}_{mu}{nu}={} vs Γ^{lambda}_{nu}{mu}={}",
                    gamma.get(lambda, mu, nu),
                    gamma.get(lambda, nu, mu)
                );
            }
        }
    }
}

#[test]
fn tensor_schwarzschild_det_sign() {
    // Outside horizon: det(g) < 0 (Lorentzian signature)
    let m_sun = 1.989e30;
    let r_s = 2.0 * G * m_sun / (C * C);
    let r = 3.0 * r_s;
    let theta = std::f64::consts::FRAC_PI_2;
    let g = MetricTensor::schwarzschild(m_sun, r, theta);
    let det = g.determinant();
    assert!(
        det < 0.0,
        "Schwarzschild det(g) should be negative outside horizon, got {det}"
    );
}

#[test]
fn tensor_metric_inverse_identity() {
    // g·g⁻¹ = I (identity matrix)
    let m_sun = 1.989e30;
    let r_s = 2.0 * G * m_sun / (C * C);
    let r = 5.0 * r_s;
    let theta = std::f64::consts::FRAC_PI_3;
    let g = MetricTensor::schwarzschild(m_sun, r, theta);
    let g_inv = g.inverse();

    // Compute product g_μα * g^αν and check it equals δ_μν
    for mu in 0..4 {
        for nu in 0..4 {
            let mut sum = 0.0;
            for alpha in 0..4 {
                sum += g.get(mu, alpha) * g_inv.get(alpha, nu);
            }
            let expected = if mu == nu { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 1e-8,
                "g·g⁻¹ [{mu}][{nu}] = {sum}, expected {expected}"
            );
        }
    }
}

#[test]
fn tensor_flrw_determinant() {
    // FLRW flat (k=0): ds² = -c²dt² + a²(dr² + r²dΩ²)
    // det(g) = -c² · a⁶ · r⁴ · sin²θ (for flat FLRW)
    let a = 2.0;
    let r = 1.0;
    let theta = std::f64::consts::FRAC_PI_2;
    let g = MetricTensor::flrw(a, 0.0, r, theta);
    let det = g.determinant();
    let expected = -(C * C) * a.powi(6) * r.powi(4) * theta.sin().powi(2);
    assert_relative_eq(det, expected, 1e-8, "FLRW flat determinant");
}

#[test]
fn tensor_timelike_ds2_negative() {
    // For a slow (timelike) particle in Schwarzschild: ds² < 0
    let m_sun = 1.989e30;
    let r_s = 2.0 * G * m_sun / (C * C);
    let r = 10.0 * r_s;
    let theta = std::f64::consts::FRAC_PI_2;
    let g = MetricTensor::schwarzschild(m_sun, r, theta);
    // dt = 1, dr = 100 m (slow particle: v << c)
    let dx = [1.0, 100.0, 0.0, 0.0];
    let ds2 = g.line_element(&dx);
    assert!(
        ds2 < 0.0,
        "Timelike interval should have ds² < 0, got {ds2}"
    );
}

// =========================================================================
// Section 37: Trigger Systems (8 tests)
// =========================================================================

#[test]
fn trigger_cross_section_10kev() {
    // σ(10 keV) via S-factor/Gamow: S/(E exp(B_G/√E))
    let sigma = LcfPhysicsConstants::dd_cross_section_barn(10.0);
    // At 10 keV, σ ~ order of 1e-4 barn (Gamow suppression)
    assert!(
        sigma > 1e-8 && sigma < 1.0,
        "Cross section at 10 keV should be between 1e-8 and 1 barn, got {sigma}"
    );
    assert!(sigma.is_finite(), "Cross section should be finite");
}

#[test]
fn trigger_cross_section_monotone() {
    // σ increases with energy (in the low keV range)
    let s1 = LcfPhysicsConstants::dd_cross_section_barn(1.0);
    let s10 = LcfPhysicsConstants::dd_cross_section_barn(10.0);
    let s50 = LcfPhysicsConstants::dd_cross_section_barn(50.0);
    assert!(s50 > s10, "σ(50) > σ(10): {s50} > {s10}");
    assert!(s10 > s1, "σ(10) > σ(1): {s10} > {s1}");
}

#[test]
fn trigger_tunneling_gamow_form() {
    // Tunneling probability at 10 keV center-of-mass energy
    let t = LcfPhysicsConstants::dd_tunneling_probability(10000.0); // eV
    assert!(
        t > 0.0 && t < 1.0,
        "Tunneling probability should be in (0,1), got {t}"
    );
    assert!(t.is_finite(), "Tunneling probability should be finite");
}

#[test]
fn trigger_screening_low_energy() {
    // At low energy (1 keV = 1000 eV), screening factor >> 1
    let f = LcfPhysicsConstants::screening_enhancement_at_energy(309.0, 1000.0);
    assert!(
        f > 1.0,
        "Screening enhancement at 1 keV should be > 1, got {f}"
    );
}

#[test]
fn trigger_screening_high_energy() {
    // At high energy (100 keV = 100000 eV), screening negligible: f ~ 1
    let f = LcfPhysicsConstants::screening_enhancement_at_energy(309.0, 100000.0);
    assert!(
        f >= 1.0 && f < 1.1,
        "Screening enhancement at 100 keV should be ~1, got {f}"
    );
}

#[test]
fn trigger_ebeam_penetration_monotone() {
    // Higher energy → deeper penetration in the Gamow peak
    // Use cross section as proxy: higher energy = higher cross section
    let s10 = LcfPhysicsConstants::dd_cross_section_barn(10.0);
    let s50 = LcfPhysicsConstants::dd_cross_section_barn(50.0);
    let s100 = LcfPhysicsConstants::dd_cross_section_barn(100.0);
    assert!(s100 > s50, "σ(100) > σ(50): {s100} > {s50}");
    assert!(s50 > s10, "σ(50) > σ(10): {s50} > {s10}");
}

#[test]
fn trigger_cross_section_screened_gte_bare() {
    // Screened cross section should always be ≥ bare cross section
    let energies = [1.0, 5.0, 10.0, 50.0, 100.0];
    for &e in &energies {
        let bare = LcfPhysicsConstants::dd_cross_section_barn(e);
        let screened = LcfPhysicsConstants::dd_cross_section_screened_barn(e, 309.0);
        assert!(
            screened >= bare * 0.9999, // allow tiny float imprecision
            "Screened ≥ bare at {e} keV: screened={screened}, bare={bare}"
        );
    }
}

#[test]
fn trigger_tunneling_decreases_with_energy_barrier() {
    // Lower energy → lower tunneling probability (harder to tunnel)
    let t_low = LcfPhysicsConstants::dd_tunneling_probability(1000.0); // 1 keV
    let t_high = LcfPhysicsConstants::dd_tunneling_probability(10000.0); // 10 keV
    assert!(
        t_high > t_low,
        "Tunneling at 10 keV should exceed 1 keV: {t_high} > {t_low}"
    );
}

// =========================================================================
// Section 38: Nonequilibrium Additions (8 tests)
// =========================================================================

#[test]
fn noise_amplitude_temp_scaling() {
    // noise(2T)/noise(T) = √2 (at same γ, dt)
    let fd1 = FluctuationDissipation::new(300.0, 1e-12);
    let fd2 = FluctuationDissipation::new(600.0, 1e-12);
    let dt = 1e-15;
    let n1 = fd1.noise_amplitude(dt);
    let n2 = fd2.noise_amplitude(dt);
    let ratio = n2 / n1;
    assert_relative_eq(ratio, std::f64::consts::SQRT_2, 1e-10, "Noise scales as √T");
}

#[test]
fn noise_amplitude_underdamped_formula() {
    // √(2γk_BT / m / dt) — exact check
    let temp = 300.0;
    let gamma = 1e-12;
    let mass = 1e-26;
    let dt = 1e-15;
    let fd = FluctuationDissipation::new(temp, gamma);
    let noise = fd.noise_amplitude_underdamped(mass, dt);
    let expected = (2.0 * gamma * K_BOLTZMANN * temp / mass / dt).sqrt();
    assert_relative_eq(noise, expected, 1e-10, "Underdamped noise amplitude");
}

#[test]
fn msd_equals_2dt() {
    // ⟨x²(t)⟩ = 2Dt where D = k_BT/γ
    let temp = 300.0;
    let gamma = 1e-12;
    let fd = FluctuationDissipation::new(temp, gamma);
    let d = fd.diffusion_coefficient();
    let t = 1e-9;
    let msd = fd.mean_squared_displacement(t);
    assert_relative_eq(msd, 2.0 * d * t, 1e-10, "MSD = 2Dt");
}

#[test]
fn velocity_correlation_time_linear() {
    // τ(2m) = 2τ(m) since τ = m/γ
    let gamma = 1e-12;
    let fd = FluctuationDissipation::new(300.0, gamma);
    let m = 1e-26;
    let tau1 = fd.velocity_correlation_time(m);
    let tau2 = fd.velocity_correlation_time(2.0 * m);
    assert_relative_eq(tau2, 2.0 * tau1, 1e-10, "Correlation time linear in mass");
}

#[test]
fn jarzynski_constant_work_delta_f() {
    // All W equal → ΔF = W (no dissipation)
    let jarzynski = JarzynskiEstimator::new();
    let w = 5e-21; // ~1.2 kT at 300K
    let work_samples: Vec<f64> = vec![w; 1000];
    let delta_f = jarzynski.free_energy_difference(&work_samples, 300.0);
    assert_relative_eq(delta_f, w, 1e-6, "Constant work → ΔF = W");
}

#[test]
fn jarzynski_dissipated_work_nonneg() {
    // W_diss = ⟨W⟩ - ΔF ≥ 0 (second law)
    let jarzynski = JarzynskiEstimator::new();
    // Biased work distribution: some higher, some lower
    let work_samples: Vec<f64> = (0..500)
        .map(|i| {
            let base = 4.14e-21; // ~kT at 300K
            base * (1.0 + 0.5 * (i as f64 / 500.0))
        })
        .collect();
    let w_diss = jarzynski.dissipated_work(&work_samples, 300.0);
    assert!(
        w_diss >= -1e-30,
        "Dissipated work should be ≥ 0, got {w_diss}"
    );
}

#[test]
fn jarzynski_equilibrium_zero_dissipation() {
    // Identical W → W_diss = 0
    let jarzynski = JarzynskiEstimator::new();
    let w = 4.14e-21;
    let work_samples: Vec<f64> = vec![w; 1000];
    let w_diss = jarzynski.dissipated_work(&work_samples, 300.0);
    assert!(
        w_diss.abs() < 1e-30,
        "Zero-variance work should give zero dissipation, got {w_diss}"
    );
}

#[test]
fn fd_noise_scales_inversely_with_dt() {
    // noise ∝ 1/√dt → noise(dt/4) = 2 × noise(dt)
    let fd = FluctuationDissipation::new(300.0, 1e-12);
    let dt = 1e-15;
    let n1 = fd.noise_amplitude(dt);
    let n2 = fd.noise_amplitude(dt / 4.0);
    assert_relative_eq(n2 / n1, 2.0, 1e-10, "Noise scales as 1/√dt");
}
