// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Causal Consciousness Experiments
//!
//! Five experiments that go beyond correlation to test whether consciousness
//! CAUSES behavioral differences, not just correlates with them.
//!
//! ## The Scientific Question
//! Experiment 10 showed r=0.455 (Phi↔proxy) and r=-0.275 (Phi↔error).
//! But correlation ≠ causation. These experiments test causality.
//!
//! Run: `cargo test --features humanoid --test causal_consciousness_experiments -- --test-threads=1 --nocapture`

use symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn conscious_config() -> CognitiveLoopConfig {
    CognitiveLoopConfig {
        embodiment_platform: EmbodimentPlatform::Humanoid,
        embodiment_blend_weight: 0.2,
        embodiment_step_interval: 1,
        enable_consciousness_thermodynamics: true,
        enable_phenomenal_binding: true,
        enable_primitive_consciousness: true,
        enable_phi_tau_feedback: true,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    }
}

fn zombie_config() -> CognitiveLoopConfig {
    CognitiveLoopConfig {
        embodiment_platform: EmbodimentPlatform::Humanoid,
        embodiment_blend_weight: 0.0,
        embodiment_step_interval: 1,
        enable_consciousness_thermodynamics: false,
        enable_phenomenal_binding: false,
        enable_primitive_consciousness: false,
        async_training: false,
        learning_threshold: 1.0,
        ..Default::default()
    }
}

fn inputs() -> Vec<&'static str> {
    vec![
        "calm observation of a still lake",
        "complex reasoning about ethical dilemmas",
        "sudden emergency requiring immediate action",
        "quiet introspective self-monitoring",
        "creative synthesis of contradictory information",
    ]
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let cov: f64 = x
        .iter()
        .zip(y)
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f64>()
        / n;
    let sx = (x.iter().map(|a| (a - mx).powi(2)).sum::<f64>() / n).sqrt();
    let sy = (y.iter().map(|b| (b - my).powi(2)).sum::<f64>() / n).sqrt();
    if sx < 1e-15 || sy < 1e-15 {
        return 0.0;
    }
    cov / (sx * sy)
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT C1: CONSCIOUS vs ZOMBIE — REAL PHI COMPARISON
//
// Does the consciousness engine (SpectralMIP) produce different Phi values
// when consciousness components are disabled? If Phi is the same for
// conscious and zombie configs, the components don't affect integration.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_c1_phi_differs_conscious_vs_zombie() {
    let input_list = inputs();

    // Conscious agent
    let mut conscious = CognitiveLoopService::new(conscious_config()).expect("conscious");
    let mut conscious_phi = Vec::new();
    for cycle in 0..60 {
        let r = conscious.cycle(input_list[cycle % input_list.len()]);
        if let Some(phi) = r.metadata.structural.spectral_mip_phi {
            conscious_phi.push(phi);
        }
    }

    // Zombie agent (same inputs)
    let mut zombie = CognitiveLoopService::new(zombie_config()).expect("zombie");
    let mut zombie_phi = Vec::new();
    for cycle in 0..60 {
        let r = zombie.cycle(input_list[cycle % input_list.len()]);
        if let Some(phi) = r.metadata.structural.spectral_mip_phi {
            zombie_phi.push(phi);
        }
    }

    let c_mean = if conscious_phi.is_empty() {
        0.0
    } else {
        conscious_phi.iter().sum::<f64>() / conscious_phi.len() as f64
    };
    let z_mean = if zombie_phi.is_empty() {
        0.0
    } else {
        zombie_phi.iter().sum::<f64>() / zombie_phi.len() as f64
    };
    let phi_diff = (c_mean - z_mean).abs();

    eprintln!("\n═══ C1: CONSCIOUS vs ZOMBIE — REAL PHI ═══");
    eprintln!(
        "  Conscious: {}/{} cycles with Phi, mean={c_mean:.4}",
        conscious_phi.len(),
        60
    );
    eprintln!(
        "  Zombie:    {}/{} cycles with Phi, mean={z_mean:.4}",
        zombie_phi.len(),
        60
    );
    eprintln!("  Phi difference: {phi_diff:.4}");
    if phi_diff > 0.1 {
        eprintln!("  VERDICT: Consciousness components CHANGE integrated information.");
    } else {
        eprintln!("  FINDING: Phi is similar for both — components don't affect integration.");
    }
    eprintln!("═══════════════════════════════════════════\n");

    // Both should produce finite values
    for &p in &conscious_phi {
        assert!(p.is_finite());
    }
    for &p in &zombie_phi {
        assert!(p.is_finite());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT C2: TEMPORAL STRUCTURE MATTERS
//
// Does the ORDER of Phi values matter, or just the average?
// Compare: real Phi sequence vs reverse-order sequence.
// If the Phi-error correlation is the same for reversed Phi,
// then temporal structure doesn't matter — only mean level.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_c2_temporal_structure_of_phi() {
    let mut service = CognitiveLoopService::new(conscious_config()).expect("service");
    let input_list = inputs();

    let mut phi_sequence = Vec::new();
    let mut error_sequence = Vec::new();

    for cycle in 0..80 {
        let r = service.cycle(input_list[cycle % input_list.len()]);
        if let Some(phi) = r.metadata.structural.spectral_mip_phi {
            phi_sequence.push(phi);
            error_sequence.push(r.prediction_error as f64);
        }
    }

    let n = phi_sequence.len();
    if n < 10 {
        eprintln!("Not enough Phi values for temporal analysis: {n}");
        return;
    }

    // Forward correlation (real temporal order)
    let r_forward = pearson(&phi_sequence, &error_sequence);

    // Reversed Phi (same values, reversed order)
    let mut phi_reversed = phi_sequence.clone();
    phi_reversed.reverse();
    let r_reversed = pearson(&phi_reversed, &error_sequence);

    // Shuffled Phi (destroy temporal structure)
    let mut phi_shuffled = phi_sequence.clone();
    let mut rng = 42u64;
    for i in (1..phi_shuffled.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng >> 33) as usize % (i + 1);
        phi_shuffled.swap(i, j);
    }
    let r_shuffled = pearson(&phi_shuffled, &error_sequence);

    eprintln!("\n═══ C2: TEMPORAL STRUCTURE OF PHI ═══");
    eprintln!("  {n} paired (Phi, error) observations");
    eprintln!("  Forward r  (real order):    {r_forward:.4}");
    eprintln!("  Reversed r (time-flipped):  {r_reversed:.4}");
    eprintln!("  Shuffled r (random order):  {r_shuffled:.4}");
    let forward_stronger = r_forward.abs() > r_reversed.abs().max(r_shuffled.abs());
    if forward_stronger {
        eprintln!("  VERDICT: Temporal structure matters — real order has strongest correlation.");
    } else {
        eprintln!(
            "  FINDING: Temporal structure may not matter — disrupted order has similar correlation."
        );
    }
    eprintln!("═════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT C3: DOSE-RESPONSE — DOES MORE CONSCIOUSNESS HELP MORE?
//
// Compare consciousness at different "doses": full, partial (no thermo),
// and minimal (zombie). If behavioral outcomes scale with consciousness
// dose, the relationship is monotonic and dose-dependent.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_c3_dose_response() {
    let input_list = inputs();

    // Dose 1: Full consciousness (all modules + phi feedback)
    let config_full = conscious_config();

    // Dose 2: Partial — consciousness modules ON but phi feedback OFF.
    // Phi is measured but doesn't feed back into dynamics or learning.
    let mut config_partial = conscious_config();
    config_partial.enable_phi_tau_feedback = false;

    // Dose 3: Minimal (zombie) — no consciousness modules, no feedback
    let config_minimal = zombie_config();

    let configs = [
        ("full+fb", config_full),
        ("no-fb", config_partial),
        ("zombie", config_minimal),
    ];

    let mut dose_results = Vec::new();

    for (label, config) in &configs {
        let mut service = CognitiveLoopService::new(config.clone()).expect("service");
        let mut phi_sum = 0.0f64;
        let mut phi_count = 0;
        let mut error_sum = 0.0f64;
        let mut proxy_sum = 0.0f64;
        let mut phi_tau_sum = 0.0f64;
        let total = 500;

        for cycle in 0..total {
            let r = service.cycle(input_list[cycle % input_list.len()]);
            if let Some(phi) = r.metadata.structural.spectral_mip_phi {
                phi_sum += phi;
                phi_count += 1;
            }
            error_sum += r.prediction_error as f64;
            proxy_sum += r.metadata.consciousness.consciousness_level;
            phi_tau_sum += r.metadata.phi_tau_factor as f64;
        }

        let phi_mean = if phi_count > 0 {
            phi_sum / phi_count as f64
        } else {
            0.0
        };
        let error_mean = error_sum / total as f64;
        let proxy_mean = proxy_sum / total as f64;
        let phi_tau_mean = phi_tau_sum / total as f64;

        dose_results.push((
            *label,
            phi_mean,
            error_mean,
            proxy_mean,
            phi_count,
            phi_tau_mean,
        ));
    }

    eprintln!("\n═══ C3: DOSE-RESPONSE (500 cycles) ═══");
    eprintln!(
        "  {:10} | {:>8} | {:>8} | {:>8} | {:>8} | Phi cycles",
        "Dose", "Phi", "Error", "Proxy", "PhiTau"
    );
    for (label, phi, error, proxy, count, phi_tau) in &dose_results {
        eprintln!(
            "  {label:10} | {phi:8.3} | {error:8.4} | {proxy:8.4} | {phi_tau:8.4} | {count}/500"
        );
    }

    // Check monotonicity: full > partial > minimal for Phi
    let phi_monotonic =
        dose_results[0].1 >= dose_results[1].1 && dose_results[1].1 >= dose_results[2].1;
    // Check monotonicity: full < partial < minimal for error (lower is better)
    let error_monotonic =
        dose_results[0].2 <= dose_results[1].2 && dose_results[1].2 <= dose_results[2].2;

    // Check monotonicity: full > partial > minimal for phi_tau (feedback active)
    let phi_tau_monotonic =
        dose_results[0].5 >= dose_results[1].5 && dose_results[1].5 >= dose_results[2].5;

    if phi_monotonic {
        eprintln!("  Phi is dose-monotonic: more consciousness → more integration.");
    }
    if error_monotonic {
        eprintln!("  Error is dose-monotonic: more consciousness → better predictions.");
    }
    if phi_tau_monotonic {
        eprintln!("  PhiTau is dose-monotonic: consciousness feedback active in higher doses.");
    }
    if !phi_monotonic && !error_monotonic && !phi_tau_monotonic {
        eprintln!("  No dose-response detected.");
    }
    eprintln!("═════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT C4: CROSS-VALIDATED PREDICTION
//
// Train linear regression on first 30 (Phi, error) pairs.
// Test on next 30 pairs. If the model generalizes, the relationship
// is stable (not an artifact of non-stationarity).
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_c4_cross_validated_prediction() {
    let mut service = CognitiveLoopService::new(conscious_config()).expect("service");
    let input_list = inputs();

    let mut phi_all = Vec::new();
    let mut error_all = Vec::new();

    for cycle in 0..80 {
        let r = service.cycle(input_list[cycle % input_list.len()]);
        if let Some(phi) = r.metadata.structural.spectral_mip_phi {
            phi_all.push(phi);
            error_all.push(r.prediction_error as f64);
        }
    }

    let n = phi_all.len();
    if n < 20 {
        eprintln!("Not enough data for cross-validation: {n}");
        return;
    }

    let split = n / 2;
    let (train_phi, test_phi) = phi_all.split_at(split);
    let (train_error, test_error) = error_all.split_at(split);

    // Simple linear regression: error = a + b * phi
    let train_r = pearson(train_phi, train_error);
    let test_r = pearson(test_phi, test_error);

    // Fit on train
    let phi_mean_train = train_phi.iter().sum::<f64>() / train_phi.len() as f64;
    let err_mean_train = train_error.iter().sum::<f64>() / train_error.len() as f64;
    let phi_var_train = train_phi
        .iter()
        .map(|p| (p - phi_mean_train).powi(2))
        .sum::<f64>()
        / train_phi.len() as f64;
    let cov_train = train_phi
        .iter()
        .zip(train_error)
        .map(|(p, e)| (p - phi_mean_train) * (e - err_mean_train))
        .sum::<f64>()
        / train_phi.len() as f64;
    let b = if phi_var_train > 1e-15 {
        cov_train / phi_var_train
    } else {
        0.0
    };
    let a = err_mean_train - b * phi_mean_train;

    // Predict on test
    let predictions: Vec<f64> = test_phi.iter().map(|p| a + b * p).collect();
    let test_mse: f64 = predictions
        .iter()
        .zip(test_error)
        .map(|(pred, actual)| (pred - actual).powi(2))
        .sum::<f64>()
        / test_error.len() as f64;
    let baseline_mse: f64 = test_error
        .iter()
        .map(|e| (e - err_mean_train).powi(2))
        .sum::<f64>()
        / test_error.len() as f64;

    let r_squared_oos = 1.0 - test_mse / baseline_mse.max(1e-15);

    eprintln!("\n═══ C4: CROSS-VALIDATED PREDICTION ═══");
    eprintln!("  Train: {split} samples, r = {train_r:.4}");
    eprintln!("  Test:  {} samples, r = {test_r:.4}", n - split);
    eprintln!("  Linear model: error = {a:.4} + {b:.6} * phi");
    eprintln!("  Out-of-sample R²: {r_squared_oos:.4}");
    if r_squared_oos > 0.0 {
        eprintln!("  VERDICT: Model generalizes — Phi-error relationship is stable.");
    } else {
        eprintln!("  FINDING: Model does not generalize (R²<0) — relationship is non-stationary.");
    }
    eprintln!("══════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT C5: MULTI-SEED PHI-BEHAVIOR CORRELATION
//
// Is the r=0.455 (Phi↔proxy) correlation robust across seeds?
// Run Experiment 10 with 3 seeds and compare r values.
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_c5_multi_seed_phi_correlation() {
    let seeds = ["causal-alpha", "causal-beta", "causal-gamma"];
    let input_list = inputs();

    let mut seed_results = Vec::new();

    for seed in &seeds {
        let mut config = conscious_config();
        config.genesis_phrase = Some(seed.to_string());
        let mut service = CognitiveLoopService::new(config).expect("service");

        let mut phi_vals = Vec::new();
        let mut proxy_vals = Vec::new();
        let mut error_vals = Vec::new();

        for cycle in 0..60 {
            let r = service.cycle(input_list[cycle % input_list.len()]);
            if let Some(phi) = r.metadata.structural.spectral_mip_phi {
                phi_vals.push(phi);
                proxy_vals.push(r.metadata.consciousness.consciousness_level);
                error_vals.push(r.prediction_error as f64);
            }
        }

        let r_proxy = pearson(&phi_vals, &proxy_vals);
        let r_error = pearson(&phi_vals, &error_vals);
        let phi_mean = if phi_vals.is_empty() {
            0.0
        } else {
            phi_vals.iter().sum::<f64>() / phi_vals.len() as f64
        };

        seed_results.push((*seed, phi_mean, r_proxy, r_error, phi_vals.len()));
    }

    // HARD ASSERT: All seeds should produce Phi values
    for (seed, _, _, _, count) in &seed_results {
        assert!(*count > 5, "Seed '{seed}' produced only {count} Phi values");
    }

    let r_proxy_values: Vec<f64> = seed_results.iter().map(|s| s.2).collect();
    let r_proxy_mean = r_proxy_values.iter().sum::<f64>() / r_proxy_values.len() as f64;
    let r_proxy_std = (r_proxy_values
        .iter()
        .map(|r| (r - r_proxy_mean).powi(2))
        .sum::<f64>()
        / r_proxy_values.len() as f64)
        .sqrt();

    eprintln!("\n═══ C5: MULTI-SEED PHI CORRELATION ═══");
    for (seed, phi_mean, r_proxy, r_error, count) in &seed_results {
        eprintln!(
            "  {seed:15} | Phi={phi_mean:7.2} | r(proxy)={r_proxy:.4} | r(error)={r_error:.4} | n={count}"
        );
    }
    eprintln!("  r(proxy) mean: {r_proxy_mean:.4} ± {r_proxy_std:.4}");
    if r_proxy_std < 0.15 {
        eprintln!("  VERDICT: Phi-behavior correlation is ROBUST across seeds.");
    } else {
        eprintln!("  FINDING: Phi-behavior correlation is SEED-SENSITIVE (std={r_proxy_std:.4}).");
    }
    eprintln!("══════════════════════════════════════\n");
}
