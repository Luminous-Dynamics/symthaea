//! Property-based conservation law tests for physics modules.
//!
//! Uses proptest to verify physical invariants hold across randomized parameters:
//! trace preservation, positivity, unitarity, purity bounds, entropy non-negativity,
//! and eigenvalue sum conservation.

use proptest::prelude::*;
use super::decoherence::{Complex64, DecoherenceChannel, DensityMatrix, LindbladEvolution, simulate_decoherence};
use super::quantum_tunneling::TunnelingCalculator;
use super::thermodynamics::ThermoEncoder;
use super::fluid_dynamics::FluidEncoder;
use super::general_relativity::GREncoder;
use super::condensed_matter::CMEncoder;
use super::cosmology::CosmologyEncoder;
use super::chaos_dynamics::systems;
use super::constants::{M_ELECTRON, HBAR};
use super::electromagnetism::EMEncoder;
use super::nonequilibrium::{FluctuationDissipation, JarzynskiEstimator, OnsagerCoefficients};
use super::optics::{OpticsEncoder, PhotonStatistics};
use super::plasma_physics::PlasmaEncoder;
use crate::genesis::GenesisSeed;

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
    #![proptest_config(ProptestConfig::with_cases(128))]

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

    /// Adaptive RK45 matches fixed-step RK4 within 1e-3 for random γ
    #[test]
    fn prop_adaptive_rk45_bounded_error(
        gamma in 0.1_f64..5.0,
    ) {
        let hamiltonian = DensityMatrix::new(2);
        let mut evolution = LindbladEvolution::new(hamiltonian);
        let mut sigma_z = DensityMatrix::new(2);
        sigma_z.set(0, 0, Complex64::from_real(1.0));
        sigma_z.set(1, 1, Complex64::from_real(-1.0));
        evolution.add_lindblad(sigma_z, gamma);

        let s = 1.0 / 2.0_f64.sqrt();
        let psi = vec![Complex64::new(s, 0.0), Complex64::new(s, 0.0)];
        let rho = DensityMatrix::pure_state(&psi);

        let t_end = 2.0;
        let traj_fixed = evolution.evolve_trajectory(&rho, 0.001, 2000);
        let traj_adaptive = evolution.evolve_adaptive(&rho, t_end, 1e-8, 1e-10);

        let p_fixed = traj_fixed.last().unwrap().purity();
        let p_adaptive = traj_adaptive.last().unwrap().purity();

        prop_assert!(
            (p_fixed - p_adaptive).abs() < 1e-3,
            "Adaptive vs fixed purity mismatch at γ={gamma}: fixed={p_fixed}, adaptive={p_adaptive}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Carnot efficiency bounded: 0 ≤ η ≤ 1 for T_hot > T_cold > 0
    #[test]
    fn prop_carnot_efficiency_bounded(
        t_hot in 301.0_f64..10000.0,
        t_cold in 1.0_f64..300.0,
    ) {
        let genesis = GenesisSeed::from_phrase("proptest carnot");
        let encoder = ThermoEncoder::from_genesis(&genesis);
        let eta = encoder.carnot_efficiency(t_hot, t_cold);
        prop_assert!(eta >= 0.0 && eta <= 1.0,
            "Carnot efficiency should be in [0,1], got {eta} (T_hot={t_hot}, T_cold={t_cold})");
    }

    /// Reynolds scaling linearity: Re(ρ, αv, L, μ) = α·Re(ρ, v, L, μ)
    #[test]
    fn prop_reynolds_mach_linear(
        v in 0.1_f64..100.0,
        alpha in 0.1_f64..10.0,
    ) {
        let genesis = GenesisSeed::from_phrase("proptest re linear");
        let encoder = FluidEncoder::from_genesis(&genesis);
        let rho = 1000.0;
        let length = 0.01;
        let mu = 0.001;
        let re1 = encoder.reynolds_number(rho, alpha * v, length, mu);
        let re2 = alpha * encoder.reynolds_number(rho, v, length, mu);
        prop_assert!(
            ((re1 - re2) / re2).abs() < 1e-10,
            "Re should scale linearly: Re(αv)={re1}, α·Re(v)={re2}"
        );
    }

    /// Redshift monotone decreasing: z(r1) > z(r2) when rs < r1 < r2
    #[test]
    fn prop_redshift_monotone_decreasing(
        r_ratio_1 in 1.01_f64..5.0,
        r_ratio_2 in 5.01_f64..100.0,
    ) {
        let genesis = GenesisSeed::from_phrase("proptest redshift mono");
        let encoder = GREncoder::from_genesis(&genesis);
        let rs = 3000.0;
        let r1 = rs * r_ratio_1;
        let r2 = rs * r_ratio_2;
        let z1 = encoder.redshift_factor(r1, rs);
        let z2 = encoder.redshift_factor(r2, rs);
        prop_assert!(z1 > z2,
            "Redshift should decrease with r: z(r1={r1})={z1} should > z(r2={r2})={z2}");
    }

    /// Fermi-Dirac bounded: 0 ≤ f(E, E_F, T) ≤ 1 for T > 0
    #[test]
    fn prop_fermi_dirac_bounded(
        e_ev in -10.0_f64..20.0,
        fermi_ev in 0.0_f64..10.0,
        temp_k in 1.0_f64..10000.0,
    ) {
        let genesis = GenesisSeed::from_phrase("proptest fd bounded");
        let encoder = CMEncoder::from_genesis(&genesis);
        let f = encoder.fermi_dirac(e_ev, fermi_ev, temp_k);
        prop_assert!(f >= 0.0 && f <= 1.0,
            "Fermi-Dirac should be in [0,1], got {f} (E={e_ev}, E_F={fermi_ev}, T={temp_k})");
    }

    /// Scale factor monotone: a(z1) > a(z2) when z1 < z2 ≥ 0
    #[test]
    fn prop_scale_factor_monotone(
        z1 in 0.0_f64..5.0,
        z_delta in 0.01_f64..10.0,
    ) {
        let genesis = GenesisSeed::from_phrase("proptest scale mono");
        let encoder = CosmologyEncoder::from_genesis(&genesis);
        let z2 = z1 + z_delta;
        let a1 = encoder.scale_factor(z1);
        let a2 = encoder.scale_factor(z2);
        prop_assert!(a1 > a2,
            "Scale factor should decrease: a(z1={z1})={a1} should > a(z2={z2})={a2}");
    }

    /// Exact Eckart matches inline Kemble formula to 1e-10
    #[test]
    fn prop_tunneling_exact_matches_kemble(
        e_ev in 0.1_f64..0.9,
    ) {
        let calc = TunnelingCalculator::electron();
        let v0 = 1.0e-19;
        let a = 1.0e-10;
        let energy = e_ev * 1.6e-19;

        let exact = calc.eckart_barrier_exact(energy, v0, a);

        // Independent Kemble calculation
        let k = (2.0 * M_ELECTRON * energy).sqrt() / HBAR;
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

        prop_assert!(
            ((exact.transmission - kemble_t) / kemble_t).abs() < 1e-10,
            "Exact should match Kemble: exact={}, kemble={kemble_t}, E={e_ev}eV",
            exact.transmission
        );
    }

    /// Landauer monotone: L(T1) > L(T2) when T1 > T2 > 0
    #[test]
    fn prop_landauer_monotone(
        t1 in 100.1_f64..10000.0,
        t_delta in 0.1_f64..100.0,
    ) {
        let genesis = GenesisSeed::from_phrase("proptest landauer mono");
        let encoder = ThermoEncoder::from_genesis(&genesis);
        let t2 = t1 - t_delta.min(t1 - 0.01);
        let l1 = encoder.landauer_limit(t1);
        let l2 = encoder.landauer_limit(t2);
        prop_assert!(l1 > l2,
            "Landauer should increase with T: L({t1})={l1} should > L({t2})={l2}");
    }

    /// Kolmogorov scale decreasing with dissipation: η(ν, ε1) > η(ν, ε2) when ε1 < ε2
    #[test]
    fn prop_kolmogorov_decreasing_dissipation(
        eps1 in 0.001_f64..1.0,
        eps_factor in 1.01_f64..100.0,
    ) {
        let genesis = GenesisSeed::from_phrase("proptest kolm");
        let encoder = FluidEncoder::from_genesis(&genesis);
        let nu = 1e-6;
        let eps2 = eps1 * eps_factor;
        let eta1 = encoder.kolmogorov_scale(nu, eps1);
        let eta2 = encoder.kolmogorov_scale(nu, eps2);
        prop_assert!(eta1 > eta2,
            "Kolmogorov should decrease: η(ε1={eps1})={eta1} should > η(ε2={eps2})={eta2}");
    }
}

// =========================================================================
// Round 6: Coverage expansion properties (P16-P25)
// =========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Entropy production ≥ 0 for PSD Onsager matrix (second law of thermodynamics)
    #[test]
    fn prop_entropy_production_nonneg(
        l11 in 0.01_f64..10.0,
        l22 in 0.01_f64..10.0,
        x1 in -10.0_f64..10.0,
        x2 in -10.0_f64..10.0,
    ) {
        // Diagonal matrix is always PSD
        let onsager = OnsagerCoefficients::diagonal(&[l11, l22]);
        let sigma = onsager.entropy_production(&[x1, x2]);
        prop_assert!(sigma >= -1e-15,
            "Entropy production should be ≥ 0 for PSD Onsager matrix, got {sigma}");
    }

    /// Dissipated work ≥ 0 (second law via Jarzynski equality)
    #[test]
    fn prop_dissipated_work_nonneg(
        w_base in 1e-22_f64..1e-19,
        spread in 0.0_f64..2.0,
    ) {
        let jarzynski = JarzynskiEstimator::new();
        let work_samples: Vec<f64> = (0..100).map(|i| {
            w_base * (1.0 + spread * (i as f64 / 100.0))
        }).collect();
        let w_diss = jarzynski.dissipated_work(&work_samples, 300.0);
        prop_assert!(w_diss >= -1e-30,
            "Dissipated work should be ≥ 0, got {w_diss} (base={w_base}, spread={spread})");
    }

    /// g²(0) bounded: 0 ≤ g²(0) ≤ 2 for all PhotonStatistics variants
    #[test]
    fn prop_g2_zero_bounded(
        _dummy in 0..5u32, // iterate; we check all 5 variants inside
    ) {
        let genesis = GenesisSeed::from_phrase("proptest g2 bounded");
        let encoder = OpticsEncoder::from_genesis(&genesis);
        let variants = [
            PhotonStatistics::Coherent,
            PhotonStatistics::Thermal,
            PhotonStatistics::Squeezed,
            PhotonStatistics::Fock,
            PhotonStatistics::Entangled,
        ];
        for &stats in &variants {
            let g2 = encoder.g2_zero(stats);
            prop_assert!(g2 >= 0.0 && g2 <= 2.0 + 1e-10,
                "g²(0) should be in [0, 2], got {g2} for {:?}", stats);
        }
    }

    /// Snell's law reciprocity: forward + reverse recovers θ₁
    #[test]
    fn prop_snells_law_reciprocity(
        n1 in 1.0_f64..2.5,
        n2 in 1.0_f64..2.5,
        theta1_deg in 0.1_f64..40.0, // small angles to avoid TIR
    ) {
        let genesis = GenesisSeed::from_phrase("proptest snell reciprocity");
        let encoder = EMEncoder::from_genesis(&genesis);
        let theta1 = theta1_deg.to_radians();

        if let Some(theta2) = encoder.snells_law(n1, n2, theta1) {
            if let Some(theta1_recovered) = encoder.snells_law(n2, n1, theta2) {
                prop_assert!(
                    (theta1_recovered - theta1).abs() < 1e-10,
                    "Snell reciprocity: θ₁={theta1}, recovered={theta1_recovered}"
                );
            }
        }
    }

    /// Refractive index: n = √(ε_r × μ_r)
    #[test]
    fn prop_refractive_index_formula(
        epsilon_r in 1.0_f64..20.0,
        mu_r in 0.5_f64..5.0,
    ) {
        let genesis = GenesisSeed::from_phrase("proptest refractive");
        let encoder = EMEncoder::from_genesis(&genesis);
        let n = encoder.refractive_index(epsilon_r, mu_r);
        let expected = (epsilon_r * mu_r).sqrt();
        prop_assert!(
            ((n - expected) / expected).abs() < 1e-10,
            "n should equal √(εμ): got {n}, expected {expected}"
        );
    }

    /// Debye length monotone in temperature: λ_D(T2) > λ_D(T1) for T2 > T1
    #[test]
    fn prop_debye_length_monotone_temp(
        t1_ev in 0.1_f64..10.0,
        t_delta in 0.01_f64..10.0,
        density in 1e16_f64..1e20,
    ) {
        let genesis = GenesisSeed::from_phrase("proptest debye mono");
        let encoder = PlasmaEncoder::from_genesis(&genesis);
        let t2_ev = t1_ev + t_delta;
        let ld1 = encoder.debye_length_m(t1_ev, density);
        let ld2 = encoder.debye_length_m(t2_ev, density);
        prop_assert!(ld2 > ld1,
            "Debye length should increase with T: λ(T2={t2_ev})={ld2} > λ(T1={t1_ev})={ld1}");
    }

    /// Plasma frequency positive for positive density
    #[test]
    fn prop_plasma_frequency_positive(
        density in 1e10_f64..1e25,
    ) {
        let genesis = GenesisSeed::from_phrase("proptest wp positive");
        let encoder = PlasmaEncoder::from_genesis(&genesis);
        let wp = encoder.plasma_frequency_rad_s(density);
        prop_assert!(wp > 0.0, "Plasma frequency should be > 0, got {wp}");
    }

    /// Logistic map Lyapunov exponent at r=4 is exactly ln(2) ≈ 0.693
    /// Test near r=4 where the map is conjugate to the tent map (always chaotic)
    #[test]
    fn prop_logistic_lyapunov_positive_chaos(
        r in 3.999_f64..4.0,
    ) {
        // Compute Lyapunov exponent analytically: λ = (1/N) Σ ln|f'(x_n)|
        // f'(x) = r(1 - 2x)
        let n_transient = 1000;
        let n_calc = 10000;
        let mut x = 0.4; // initial condition

        // Skip transient
        for _ in 0..n_transient {
            x = systems::logistic(x, r);
        }

        // Accumulate Lyapunov
        let mut lambda_sum = 0.0;
        for _ in 0..n_calc {
            let derivative = (r * (1.0 - 2.0 * x)).abs();
            if derivative > 1e-15 {
                lambda_sum += derivative.ln();
            }
            x = systems::logistic(x, r);
        }

        let lambda = lambda_sum / n_calc as f64;
        // At r=4: λ = ln(2) ≈ 0.693. Near r=4, λ should be close.
        prop_assert!(lambda > 0.5,
            "Logistic Lyapunov near r=4 should be > 0.5, got {lambda} at r={r}");
    }

    /// Noise amplitude positive for T > 0, γ > 0, dt > 0
    #[test]
    fn prop_noise_amplitude_positive(
        temp in 1.0_f64..10000.0,
        gamma in 1e-15_f64..1e-8,
        dt_exp in -18.0_f64..-12.0,
    ) {
        let dt = 10.0_f64.powf(dt_exp);
        let fd = FluctuationDissipation::new(temp, gamma);
        let noise = fd.noise_amplitude(dt);
        prop_assert!(noise > 0.0,
            "Noise amplitude should be > 0, got {noise} (T={temp}, γ={gamma}, dt={dt})");
    }

    /// MSD monotone in time: MSD(t2) > MSD(t1) for t2 > t1
    #[test]
    fn prop_msd_monotone_time(
        t1 in 1e-12_f64..1e-6,
        t_delta in 1e-12_f64..1e-6,
    ) {
        let fd = FluctuationDissipation::new(300.0, 1e-12);
        let t2 = t1 + t_delta;
        let msd1 = fd.mean_squared_displacement(t1);
        let msd2 = fd.mean_squared_displacement(t2);
        prop_assert!(msd2 > msd1,
            "MSD should increase with time: MSD(t2={t2})={msd2} > MSD(t1={t1})={msd1}");
    }
}
