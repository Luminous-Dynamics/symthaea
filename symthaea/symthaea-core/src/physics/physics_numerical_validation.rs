//! Numerical validation tests for physics modules.
//!
//! Each test compares module output against a known analytical value or textbook result.
//! This catches formula transcription errors and numerical issues that range-only tests miss.

use super::chaos_dynamics::{systems, AttractorAnalyzer, LyapunovCalculator};
use super::constants::{C, E_CHARGE, EPSILON_0, H, HBAR, K_BOLTZMANN, M_ELECTRON};
use super::decoherence::{Complex64, DecoherenceChannel, DensityMatrix, simulate_decoherence};
use super::electromagnetism::{EMEncoder, Polarization, SpectrumRegion};
use super::nonequilibrium::{FluctuationDissipation, JarzynskiEstimator, OnsagerCoefficients};
use super::optics::{OpticsEncoder, PhotonStatistics};
use super::plasma_physics::PlasmaEncoder;
use super::quantum_tunneling::TunnelingCalculator;
use super::thermal_transport::{ThermalProperties, LayerGeometry, ThermalTransport};
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
    if p <= 0.0 { return -6.0; }
    if p >= 1.0 { return 6.0; }

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
    let w0: f64 = 5e-21;     // Mean work (J)
    let sigma: f64 = 1e-21;  // Std dev (J)
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

    assert_relative_eq(delta_f, expected_delta_f, 0.01, "Jarzynski Gaussian work ΔF");
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
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Descending

    assert_relative_eq(eigenvalues[0], 0.6, 1e-10, "λ₁ of diag(0.6,0.3,0.1)");
    assert_relative_eq(eigenvalues[1], 0.3, 1e-10, "λ₂ of diag(0.6,0.3,0.1)");
    assert_relative_eq(eigenvalues[2], 0.1, 1e-10, "λ₃ of diag(0.6,0.3,0.1)");

    // Entropy: S = -(0.6 log₂ 0.6 + 0.3 log₂ 0.3 + 0.1 log₂ 0.1)
    let s = rho.von_neumann_entropy();
    let expected_s = -(0.6_f64 * 0.6_f64.log2()
        + 0.3_f64 * 0.3_f64.log2()
        + 0.1_f64 * 0.1_f64.log2());
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
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let disc = 0.05_f64.sqrt();
    let lambda1 = 0.5 + disc;
    let lambda2 = 0.5 - disc;

    assert_relative_eq(eigenvalues[0], lambda1, 1e-10, "λ₁ of [[0.7,0.1],[0.1,0.3]]");
    assert_relative_eq(eigenvalues[1], lambda2, 1e-10, "λ₂ of [[0.7,0.1],[0.1,0.3]]");
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
        assert_relative_eq(actual, *expected, 1e-10,
            &format!("Photon energy at {}nm", wavelength));
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

    assert_relative_eq(actual, expected, 1e-10, "Rayleigh range for 100μm waist at 532nm");
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
            assert!(hi.is_infinite(), "{:?} upper bound should be infinity", region);
        }
    }
}

#[test]
fn em_spectrum_classification() {
    // Verify that create_wave classifies center frequencies correctly
    let genesis = GenesisSeed::from_phrase("em_validation");
    let encoder = EMEncoder::from_genesis(&genesis);

    let test_cases: [(f64, SpectrumRegion); 6] = [
        (1e6, SpectrumRegion::Radio),           // 1 MHz
        (10e9, SpectrumRegion::Microwave),       // 10 GHz
        (1e13, SpectrumRegion::Infrared),        // 10 THz
        (6e14, SpectrumRegion::Visible),         // 600 THz (green light)
        (1e15, SpectrumRegion::Ultraviolet),     // 1 PHz
        (1e18, SpectrumRegion::XRay),            // 1 EHz
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
    let r_inner: f64 = 0.05;  // 5 cm
    let r_outer: f64 = 0.10;  // 10 cm
    let geom = LayerGeometry::sphere(r_inner, r_outer);
    let vol = geom.volume();

    let expected = (4.0 / 3.0) * std::f64::consts::PI
        * (r_outer.powi(3) - r_inner.powi(3));
    assert_relative_eq(vol, expected, 1e-10, "Spherical shell volume");
}

#[test]
fn coolant_flow_rate_formula() {
    // ṁ = Q / (Cₚ · ΔT)
    let genesis = GenesisSeed::from_phrase("thermal_validation");
    let thermal = ThermalTransport::from_genesis(&genesis);

    let power_w: f64 = 5000.0;
    let cp: f64 = 4186.0;   // Water Cₚ (J/kg·K)
    let dt: f64 = 10.0;     // 10 K rise

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
        1000.0,     // 1 kW
        &shell,
        &shell_geom,
        &interface,
        &iface_geom,
        &core,
        &core_geom,
        300.0,      // coolant at 300 K
        5000.0,     // h = 5000 W/m²K
    );

    let alpha_coeff = hea.expansion_coeff; // 10e-6
    let youngs = 120e9;
    let poisson = 0.33;
    let yield_stress = 800e6;

    let stress = thermal.thermal_stress(
        &profile,
        &hea,
        &shell_geom,
        youngs,
        poisson,
        yield_stress,
    );

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
    assert_relative_eq(ratio as f64, 18.0, 1e-6, "Kinetic energy ratio m=2,v=3 vs m=1,v=1");
}

#[test]
fn ising_2d_critical_temperature() {
    // Tc = 2J / ln(1 + √2)
    let genesis = GenesisSeed::from_phrase("ising_validation");
    let encoder = super::statistical_mechanics::StatMechEncoder::from_genesis(&genesis);
    let j = 1.0;
    let ising = encoder.ising_model(2, j, 0.0);

    let expected_tc = 2.0 * j / (1.0 + 2.0_f64.sqrt()).ln();
    assert_relative_eq(ising.critical_temp, expected_tc, 1e-10, "Ising 2D Tc = 2J/ln(1+√2)");
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
    assert_relative_eq(IonType::Sodium.nernst_potential(), 60.0, 1e-10, "Na+ Nernst");
    assert_relative_eq(IonType::Potassium.nernst_potential(), -90.0, 1e-10, "K+ Nernst");
    assert_relative_eq(IonType::Calcium.nernst_potential(), 120.0, 1e-10, "Ca2+ Nernst");
    assert_relative_eq(IonType::Chloride.nernst_potential(), -80.0, 1e-10, "Cl- Nernst");
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
    assert_relative_eq(
        synapse.strength,
        expected,
        1e-10,
        "STDP LTP at dt=5ms",
    );
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
    assert_eq!(anti_h.positrons, anti_h.antiprotons, "Anti-H: positrons == antiprotons");

    let anti_he3 = antimatter.antihelium3();
    assert_eq!(anti_he3.positrons, anti_he3.antiprotons, "Anti-He3: positrons == antiprotons");

    let anti_he4 = antimatter.antihelium4();
    assert_eq!(anti_he4.positrons, anti_he4.antiprotons, "Anti-He4: positrons == antiprotons");
}

// =========================================================================
// Section 21: Derived Laws
// =========================================================================

#[test]
fn law_composition_confidence() {
    // conf(A∘B) = conf(A)·conf(B) — multiplicative
    let genesis = GenesisSeed::from_phrase("laws_validation");
    let engine = super::derived_laws::LawsDerivationEngine::from_genesis(&genesis);

    let energy = engine.derive_energy_conservation();   // confidence = 1.0
    let momentum = engine.derive_momentum_conservation(); // confidence = 1.0
    let ohm = engine.derive_ohms_law();                   // confidence < 1.0

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
    let v0: f64 = 1.0e-19;   // ~0.6 eV
    let a: f64 = 1.0e-10;    // 1 Å

    let energies = [0.2e-19, 0.4e-19, 0.6e-19, 0.8e-19, 1.5e-19];
    for &e in &energies {
        let result = calc.eckart_barrier(e, v0, a);

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

        // WKB vs exact Kemble: WKB systematically underestimates transmission
        // at sub-barrier energies and overestimates near/above the barrier top.
        // Worst agreement is near E ≈ V₀ (classical turning point). Allow 20%.
        if expected_t > 1e-10 {
            let rel_err = ((result.transmission - expected_t) / expected_t).abs();
            assert!(
                rel_err < 0.20 || result.transmission > 0.90,
                "Eckart Kemble at E={e:.2e}: WKB={:.4}, exact={expected_t:.4} (rel err {rel_err:.2e})",
                result.transmission
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
