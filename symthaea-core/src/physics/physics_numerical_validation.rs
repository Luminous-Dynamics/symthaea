//! Numerical validation tests for physics modules.
//!
//! Each test compares module output against a known analytical value or textbook result.
//! This catches formula transcription errors and numerical issues that range-only tests miss.

use super::chaos_dynamics::{systems, AttractorAnalyzer, LyapunovCalculator};
use super::constants::{E_CHARGE, EPSILON_0, HBAR, K_BOLTZMANN, M_ELECTRON};
use super::decoherence::{Complex64, DecoherenceChannel, DensityMatrix, simulate_decoherence};
use super::nonequilibrium::{FluctuationDissipation, JarzynskiEstimator, OnsagerCoefficients};
use super::plasma_physics::PlasmaEncoder;
use super::quantum_tunneling::TunnelingCalculator;
use crate::genesis::GenesisSeed;

/// Assert that `actual` is within `tolerance` relative error of `expected`.
fn assert_relative_eq(actual: f64, expected: f64, tolerance: f64, context: &str) {
    if expected == 0.0 {
        assert!(
            actual.abs() < tolerance,
            "{context}: expected ~0, got {actual} (abs tol {tolerance})"
        );
        return;
    }
    let rel = ((actual - expected) / expected).abs();
    assert!(
        rel <= tolerance,
        "{context}: expected {expected}, got {actual} (rel err {rel:.2e}, tol {tolerance:.2e})"
    );
}

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
        (1.0, 5.0, 1e-10), // below barrier
        (4.0, 5.0, 1e-10), // just below
        (6.0, 5.0, 1e-10), // above barrier
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
            rel_err < 0.10,
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
        rel_err < 0.10,
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

    assert_relative_eq(d_ky, expected, 1e-10, "Kaplan-Yorke dimension (pure arithmetic)");
}

#[test]
fn lorenz_exponent_sum() {
    // Σλᵢ = -(σ + 1 + β) = -(10 + 1 + 8/3) = -13.6667
    let calc = LyapunovCalculator::new(0.01, 5000, 50000);
    let lorenz = |state: &[f64]| systems::lorenz(state, 10.0, 28.0, 8.0 / 3.0);
    let jacobian =
        |state: &[f64]| systems::lorenz_jacobian(state, 10.0, 28.0, 8.0 / 3.0);
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
    let expected: f64 =
        (EPSILON_0 * K_BOLTZMANN * t_k / (density * E_CHARGE * E_CHARGE)).sqrt();

    assert_relative_eq(lambda_d, expected, 1e-6, "Debye length");
}

#[test]
fn plasma_frequency_value() {
    // ω_p = √(n e² / (ε₀ m_e))
    let genesis = GenesisSeed::from_phrase("plasma_validation_test");
    let encoder = PlasmaEncoder::from_genesis(&genesis);

    let density: f64 = 1e18;
    let omega_p = encoder.plasma_frequency_rad_s(density);
    let expected: f64 =
        (density * E_CHARGE * E_CHARGE / (EPSILON_0 * M_ELECTRON)).sqrt();

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
        (final_purity - 1.0).abs() < 0.05,
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
        (final_purity - 0.5).abs() < 0.10,
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
    let v0: f64 = 1.0e-19;   // barrier height ~0.6 eV
    let a: f64 = 1.0e-10;    // barrier width ~1 Å

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
            e, result.transmission, prev_t
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
    let l = [
        [2.0, 0.5, 0.3],
        [0.5, 3.0, 0.4],
        [0.3, 0.4, 1.0],
    ];
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
