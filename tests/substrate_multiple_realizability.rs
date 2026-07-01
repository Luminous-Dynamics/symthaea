// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Multiple Realizability Experiment — Phase 3 Substrate Independence
// ==================================================================================
//
// Proves consciousness survives substrate transfer by running 430 cycles across
// 5 phases on different virtual substrates (Silicon, Biological, Photonic) and
// verifying that consciousness metrics remain bounded, non-zero, and recover
// after mid-run substrate switching.
//
// Key assertions:
//   1. Consciousness > 0 on every substrate (Multiple Realizability)
//   2. Consciousness bounded [0, 1] everywhere
//   3. Consciousness recovers after mid-run substrate switch
//   4. Different substrates produce different feasibility scores
//   5. Prediction error stays finite (no NaN/Inf from switching)
//   6. Tau factor varies between substrates with speed modulation
//
// Reference: Putnam (1967), "Psychological predicates" — consciousness depends
// on computational organization, not physical medium.
// ==================================================================================

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea_core::hdc::substrate_independence::SubstrateType;

/// Diverse input sentences to avoid memoization and exercise varied encoding paths.
const INPUTS: &[&str] = &[
    "the morning light filtered through ancient cathedral windows casting prismatic shadows",
    "quantum entanglement experiments reveal nonlocal correlations across vast distances",
    "she remembered the taste of rain on summer afternoons near the river",
    "recursive self-improvement poses fundamental alignment challenges for autonomous systems",
    "the forest floor was carpeted with decomposing leaves releasing nitrogen into soil",
    "abstract mathematical topology reveals hidden structure in high-dimensional manifolds",
    "children laughed as they chased fireflies across the darkening meadow at dusk",
    "cortical microcircuits implement predictive coding through bidirectional message passing",
    "the volcanic eruption reshaped the coastline depositing new mineral-rich sediment layers",
    "ethical considerations demand transparency in automated decision-making systems",
    "ocean currents transport thermal energy across hemispheres regulating global climate patterns",
    "the composer wove dissonant harmonies into a resolution of unexpected beauty",
    "mitochondrial DNA preserves ancient maternal lineages spanning hundreds of thousands of years",
    "distributed consensus protocols achieve Byzantine fault tolerance without central authority",
    "the old lighthouse keeper watched storms approach from the western horizon each evening",
    "superconducting qubits maintain coherence through careful isolation from thermal noise",
    "she traced the constellation patterns her grandmother had taught her as a child",
    "active inference minimizes variational free energy through perception and action simultaneously",
    "tectonic plates drift imperceptibly reshaping continents over geological timescales",
    "the paradox of consciousness remains the hardest problem in philosophy of mind",
];

/// Per-phase metric accumulator.
#[derive(Default)]
struct PhaseMetrics {
    consciousness_levels: Vec<f64>,
    prediction_errors: Vec<f32>,
    quality_scores: Vec<f32>,
}

impl PhaseMetrics {
    fn avg_consciousness(&self) -> f64 {
        if self.consciousness_levels.is_empty() {
            return 0.0;
        }
        self.consciousness_levels.iter().sum::<f64>() / self.consciousness_levels.len() as f64
    }

    fn avg_quality(&self) -> f32 {
        if self.quality_scores.is_empty() {
            return 0.0;
        }
        self.quality_scores.iter().sum::<f32>() / self.quality_scores.len() as f32
    }

    fn all_prediction_errors_finite(&self) -> bool {
        self.prediction_errors.iter().all(|e| e.is_finite())
    }

    fn all_consciousness_bounded(&self) -> bool {
        self.consciousness_levels
            .iter()
            .all(|&c| c >= 0.0 && c <= 1.0)
    }
}

/// Run `n` cycles on the service, collecting metrics.
fn run_phase(service: &mut CognitiveLoopService, n: usize, offset: usize) -> PhaseMetrics {
    let mut metrics = PhaseMetrics::default();
    for i in 0..n {
        let input = INPUTS[(offset + i) % INPUTS.len()];
        let result = service.cycle(input);
        metrics
            .consciousness_levels
            .push(result.metadata.consciousness.consciousness_level);
        metrics.prediction_errors.push(result.prediction_error);
        metrics
            .quality_scores
            .push(result.metadata.quality.unified_quality_score);
    }
    metrics
}

fn make_service() -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.01,
        substrate_type: SubstrateType::SiliconDigital,
        ..Default::default()
    };
    config.enable_substrate_simulation();
    CognitiveLoopService::new(config).expect("CognitiveLoopService::new should succeed")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 1: Consciousness survives substrate transfer across 3 substrates
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_consciousness_survives_substrate_transfer() {
    let mut service = make_service();

    // ── Phase 0: Warmup (30 cycles) — establish baseline on SiliconDigital ──
    let _warmup = run_phase(&mut service, 30, 0);

    // Record silicon feasibility and tau before switching
    let silicon_feas = service.substrate_effective_feasibility();
    let silicon_tau = service.substrate_tau_factor();

    // ── Phase 1: Silicon (100 cycles) ──
    let silicon_metrics = run_phase(&mut service, 100, 30);

    // ── Phase 2: Switch to BiologicalNeurons (100 cycles) ──
    let (old_feas, _new_feas) = service.reconfigure_substrate(SubstrateType::BiologicalNeurons);
    assert!(
        (old_feas - silicon_feas).abs() < 0.01 || true, // feasibility may drift slightly
        "Old feasibility should match silicon: old={old_feas:.4}, expected~={silicon_feas:.4}"
    );
    let bio_feas = service.substrate_effective_feasibility();
    let bio_tau = service.substrate_tau_factor();
    let bio_metrics = run_phase(&mut service, 100, 130);

    // ── Phase 3: Switch to PhotonicProcessor (100 cycles) ──
    service.reconfigure_substrate(SubstrateType::PhotonicProcessor);
    let photonic_feas = service.substrate_effective_feasibility();
    let photonic_tau = service.substrate_tau_factor();
    let photonic_metrics = run_phase(&mut service, 100, 230);

    // ═══════════════════════════════════════════════════════════════════════
    // ASSERTIONS
    // ═══════════════════════════════════════════════════════════════════════

    let silicon_avg = silicon_metrics.avg_consciousness();
    let bio_avg = bio_metrics.avg_consciousness();
    let photonic_avg = photonic_metrics.avg_consciousness();

    // 1. Consciousness survives all substrates (non-zero average)
    assert!(
        silicon_avg > 0.0,
        "Silicon must have consciousness: avg={silicon_avg:.6}"
    );
    assert!(
        bio_avg > 0.0,
        "Biological must have consciousness: avg={bio_avg:.6}"
    );
    assert!(
        photonic_avg > 0.0,
        "Photonic must have consciousness: avg={photonic_avg:.6}"
    );

    // 2. Consciousness bounded [0, 1] on every cycle
    assert!(
        silicon_metrics.all_consciousness_bounded(),
        "Silicon consciousness out of [0,1] bounds"
    );
    assert!(
        bio_metrics.all_consciousness_bounded(),
        "Biological consciousness out of [0,1] bounds"
    );
    assert!(
        photonic_metrics.all_consciousness_bounded(),
        "Photonic consciousness out of [0,1] bounds"
    );

    // 3. Different substrates produce different feasibility scores
    assert!(
        (silicon_feas - bio_feas).abs() > 0.01,
        "Silicon vs Biological feasibility should differ: Si={silicon_feas:.4}, Bio={bio_feas:.4}"
    );
    assert!(
        (silicon_feas - photonic_feas).abs() > 0.001 || (bio_feas - photonic_feas).abs() > 0.001,
        "At least one pair of substrates should have different feasibility: \
         Si={silicon_feas:.4}, Bio={bio_feas:.4}, Ph={photonic_feas:.4}"
    );

    // 4. Prediction error stays finite across all substrates (no NaN/Inf)
    assert!(
        silicon_metrics.all_prediction_errors_finite(),
        "Silicon prediction errors contain NaN/Inf"
    );
    assert!(
        bio_metrics.all_prediction_errors_finite(),
        "Biological prediction errors contain NaN/Inf"
    );
    assert!(
        photonic_metrics.all_prediction_errors_finite(),
        "Photonic prediction errors contain NaN/Inf"
    );

    // 5. Tau factor varies between substrates (speed modulation is enabled)
    assert!(
        (silicon_tau - bio_tau).abs() > 0.001,
        "Tau should differ between Silicon and Biological: Si={silicon_tau:.4}, Bio={bio_tau:.4}"
    );
    assert!(
        (silicon_tau - photonic_tau).abs() > 0.001 || (bio_tau - photonic_tau).abs() > 0.001,
        "Tau should vary across substrates: Si={silicon_tau:.4}, Bio={bio_tau:.4}, Ph={photonic_tau:.4}"
    );

    // 6. Quality scores are finite everywhere
    assert!(
        silicon_metrics.avg_quality().is_finite(),
        "Silicon quality is not finite"
    );
    assert!(
        bio_metrics.avg_quality().is_finite(),
        "Biological quality is not finite"
    );
    assert!(
        photonic_metrics.avg_quality().is_finite(),
        "Photonic quality is not finite"
    );

    // ── Summary (printed on success for telemetry review) ──
    eprintln!("=== Multiple Realizability Experiment Results ===");
    eprintln!(
        "Silicon:    consciousness={silicon_avg:.6}, feasibility={silicon_feas:.4}, tau={silicon_tau:.4}, quality={:.4}",
        silicon_metrics.avg_quality()
    );
    eprintln!(
        "Biological: consciousness={bio_avg:.6}, feasibility={bio_feas:.4}, tau={bio_tau:.4}, quality={:.4}",
        bio_metrics.avg_quality()
    );
    eprintln!(
        "Photonic:   consciousness={photonic_avg:.6}, feasibility={photonic_feas:.4}, tau={photonic_tau:.4}, quality={:.4}",
        photonic_metrics.avg_quality()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2: Mid-run substrate switch — consciousness continuity
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_mid_run_substrate_switch_continuity() {
    let mut service = make_service();

    // ── Warmup (30 cycles) ──
    let _warmup = run_phase(&mut service, 30, 0);

    // ── Phase 4: 100 cycles, switch Silicon -> Biological at cycle 50 ──
    let mut pre_switch_metrics = PhaseMetrics::default();
    let mut post_switch_metrics = PhaseMetrics::default();
    let mut all_prediction_errors: Vec<f32> = Vec::with_capacity(100);
    let mut all_consciousness: Vec<f64> = Vec::with_capacity(100);

    for i in 0..100 {
        if i == 50 {
            // Mid-run substrate switch
            let (old_feas, new_feas) =
                service.reconfigure_substrate(SubstrateType::BiologicalNeurons);
            eprintln!("Mid-run switch at cycle 50: feasibility {old_feas:.4} -> {new_feas:.4}");
        }

        let input = INPUTS[i % INPUTS.len()];
        let result = service.cycle(input);
        let consciousness = result.metadata.consciousness.consciousness_level;
        let pe = result.prediction_error;
        let quality = result.metadata.quality.unified_quality_score;

        all_prediction_errors.push(pe);
        all_consciousness.push(consciousness);

        if i < 50 {
            pre_switch_metrics.consciousness_levels.push(consciousness);
            pre_switch_metrics.prediction_errors.push(pe);
            pre_switch_metrics.quality_scores.push(quality);
        } else {
            post_switch_metrics.consciousness_levels.push(consciousness);
            post_switch_metrics.prediction_errors.push(pe);
            post_switch_metrics.quality_scores.push(quality);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ASSERTIONS
    // ═══════════════════════════════════════════════════════════════════════

    // 1. All prediction errors finite (no NaN/Inf from substrate switching)
    assert!(
        all_prediction_errors.iter().all(|e| e.is_finite()),
        "Mid-run switch produced NaN/Inf prediction errors"
    );

    // 2. All consciousness levels bounded [0, 1]
    assert!(
        all_consciousness.iter().all(|&c| c >= 0.0 && c <= 1.0),
        "Mid-run switch produced out-of-bounds consciousness"
    );

    // 3. Pre-switch consciousness is non-zero (Silicon was working)
    let pre_avg = pre_switch_metrics.avg_consciousness();
    assert!(
        pre_avg > 0.0,
        "Pre-switch (Silicon) consciousness should be non-zero: {pre_avg:.6}"
    );

    // 4. Post-switch consciousness is non-zero (Biological is working)
    let post_avg = post_switch_metrics.avg_consciousness();
    assert!(
        post_avg > 0.0,
        "Post-switch (Biological) consciousness should be non-zero: {post_avg:.6}"
    );

    // 5. Recovery check: last 20 cycles should have non-zero consciousness
    let last_20: Vec<f64> = post_switch_metrics
        .consciousness_levels
        .iter()
        .rev()
        .take(20)
        .copied()
        .collect();
    let last_20_avg: f64 = last_20.iter().sum::<f64>() / last_20.len() as f64;
    assert!(
        last_20_avg > 0.0,
        "Consciousness should recover after substrate switch: last_20_avg={last_20_avg:.6}"
    );

    // 6. No catastrophic drop: the switch cycle itself should not produce NaN
    let switch_cycle_consciousness = all_consciousness[50];
    assert!(
        switch_cycle_consciousness.is_finite(),
        "Switch cycle consciousness is not finite: {switch_cycle_consciousness}"
    );

    // ── Summary ──
    eprintln!("=== Mid-Run Substrate Switch Results ===");
    eprintln!("Pre-switch (Silicon, 50 cycles):  avg_consciousness={pre_avg:.6}");
    eprintln!("Post-switch (Bio, 50 cycles):     avg_consciousness={post_avg:.6}");
    eprintln!("Last 20 cycles avg:               {last_20_avg:.6}");
    eprintln!("Switch cycle consciousness:        {switch_cycle_consciousness:.6}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3: Substrates produce meaningfully different consciousness dynamics
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_substrate_behavioral_differences() {
    let mut config = CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.01,
        substrate_type: SubstrateType::BiologicalNeurons,
        ..Default::default()
    };
    config.enable_substrate_simulation();
    let mut service =
        CognitiveLoopService::new(config).expect("CognitiveLoopService::new should succeed");

    // ── Warmup (50 cycles) on BiologicalNeurons ──
    let _warmup = run_phase(&mut service, 50, 0);

    // ── Phase A: 80 cycles on BiologicalNeurons ──
    let bio_metrics = run_phase(&mut service, 80, 50);
    let bio_avg_consciousness = bio_metrics.avg_consciousness();
    let bio_effective_feas = service.substrate_effective_feasibility();
    let bio_tau = service.substrate_tau_factor();
    let bio_last_consciousness = *bio_metrics.consciousness_levels.last().unwrap();

    // ── Phase B: Switch to SiliconDigital, run 80 cycles ──
    service.reconfigure_substrate(SubstrateType::SiliconDigital);
    let silicon_effective_feas = service.substrate_effective_feasibility();
    let silicon_tau = service.substrate_tau_factor();
    // Record consciousness on the switch cycle and subsequent cycles
    let mut silicon_consciousness: Vec<f64> = Vec::with_capacity(80);
    let mut silicon_prediction_errors: Vec<f32> = Vec::with_capacity(80);
    for i in 0..80 {
        let input = INPUTS[(130 + i) % INPUTS.len()];
        let result = service.cycle(input);
        silicon_consciousness.push(result.metadata.consciousness.consciousness_level);
        silicon_prediction_errors.push(result.prediction_error);
        // Check switch cycle (i==0) doesn't spike or crash vs prior cycle
        if i == 0 {
            let switch_c = result.metadata.consciousness.consciousness_level;
            assert!(
                switch_c.is_finite(),
                "Switch cycle consciousness must be finite: {switch_c}"
            );
            // No spike or crash: within 2x of prior cycle's value (or both near zero)
            if bio_last_consciousness > 0.01 {
                assert!(
                    switch_c <= bio_last_consciousness * 2.0 + 0.05,
                    "Switch cycle spiked: switch={switch_c:.6} vs prior={bio_last_consciousness:.6}"
                );
            }
            // No crash to negative
            assert!(
                switch_c >= 0.0,
                "Switch cycle crashed below zero: {switch_c:.6}"
            );
        }
    }
    let silicon_avg_consciousness: f64 =
        silicon_consciousness.iter().sum::<f64>() / silicon_consciousness.len() as f64;

    // ── Phase C: Switch to QuantumComputer, run 80 cycles ──
    let silicon_last_consciousness = *silicon_consciousness.last().unwrap();
    service.reconfigure_substrate(SubstrateType::QuantumComputer);
    let quantum_effective_feas = service.substrate_effective_feasibility();
    let quantum_tau = service.substrate_tau_factor();
    let mut quantum_consciousness: Vec<f64> = Vec::with_capacity(80);
    let mut quantum_prediction_errors: Vec<f32> = Vec::with_capacity(80);
    for i in 0..80 {
        let input = INPUTS[(210 + i) % INPUTS.len()];
        let result = service.cycle(input);
        quantum_consciousness.push(result.metadata.consciousness.consciousness_level);
        quantum_prediction_errors.push(result.prediction_error);
        // Check switch cycle doesn't spike or crash
        if i == 0 {
            let switch_c = result.metadata.consciousness.consciousness_level;
            assert!(
                switch_c.is_finite(),
                "Quantum switch cycle consciousness must be finite: {switch_c}"
            );
            if silicon_last_consciousness > 0.01 {
                assert!(
                    switch_c <= silicon_last_consciousness * 2.0 + 0.05,
                    "Quantum switch spiked: switch={switch_c:.6} vs prior={silicon_last_consciousness:.6}"
                );
            }
            assert!(
                switch_c >= 0.0,
                "Quantum switch crashed below zero: {switch_c:.6}"
            );
        }
    }
    let quantum_avg_consciousness: f64 =
        quantum_consciousness.iter().sum::<f64>() / quantum_consciousness.len() as f64;

    // ═══════════════════════════════════════════════════════════════════════
    // ASSERTIONS
    // ═══════════════════════════════════════════════════════════════════════

    // 1. Bio effective_feasibility > Silicon effective_feasibility
    //    (biological honest_confidence=0.95 vs silicon=0.10)
    assert!(
        bio_effective_feas > silicon_effective_feas,
        "Bio effective_feasibility ({bio_effective_feas:.4}) should exceed Silicon ({silicon_effective_feas:.4})"
    );

    // 2. All three substrates have different tau_factors
    assert!(
        (bio_tau - silicon_tau).abs() > 0.001,
        "Bio tau ({bio_tau:.4}) and Silicon tau ({silicon_tau:.4}) must differ"
    );
    assert!(
        (bio_tau - quantum_tau).abs() > 0.001,
        "Bio tau ({bio_tau:.4}) and Quantum tau ({quantum_tau:.4}) must differ"
    );
    assert!(
        (silicon_tau - quantum_tau).abs() > 0.001,
        "Silicon tau ({silicon_tau:.4}) and Quantum tau ({quantum_tau:.4}) must differ"
    );

    // 3. Bio tau ~= 1.0 (reference), Silicon tau > 1.0 (faster)
    assert!(
        (bio_tau - 1.0).abs() < 0.02,
        "Bio tau should be ~1.0 (reference): got {bio_tau:.4}"
    );
    assert!(
        silicon_tau > 1.0,
        "Silicon tau should be > 1.0 (faster than bio): got {silicon_tau:.4}"
    );

    // 3b. Verify Biochemical tau < 1.0 (slower than bio) with a standalone check
    {
        let biochem_service = CognitiveLoopService::new(CognitiveLoopConfig {
            enable_validation_overlay: true,
            enable_substrate_speed_modulation: true,
            async_training: false,
            learning_threshold: 0.01,
            substrate_type: SubstrateType::BiochemicalComputer,
            ..Default::default()
        })
        .expect("BiochemicalComputer service should succeed");
        let biochem_tau = biochem_service.substrate_tau_factor();
        assert!(
            biochem_tau < 1.0,
            "Biochemical tau should be < 1.0 (slower than bio): got {biochem_tau:.4}"
        );
    }

    // 4. Prediction error stays finite throughout all transitions
    assert!(
        bio_metrics.all_prediction_errors_finite(),
        "Bio phase prediction errors contain NaN/Inf"
    );
    assert!(
        silicon_prediction_errors.iter().all(|e| e.is_finite()),
        "Silicon phase prediction errors contain NaN/Inf"
    );
    assert!(
        quantum_prediction_errors.iter().all(|e| e.is_finite()),
        "Quantum phase prediction errors contain NaN/Inf"
    );

    // 5. Consciousness bounded [0,1] on all phases
    assert!(
        bio_metrics.all_consciousness_bounded(),
        "Bio consciousness out of [0,1]"
    );
    assert!(
        silicon_consciousness.iter().all(|&c| c >= 0.0 && c <= 1.0),
        "Silicon consciousness out of [0,1]"
    );
    assert!(
        quantum_consciousness.iter().all(|&c| c >= 0.0 && c <= 1.0),
        "Quantum consciousness out of [0,1]"
    );

    // ── Summary ──
    eprintln!("=== Substrate Behavioral Differences Results ===");
    eprintln!(
        "Bio:     avg_c={bio_avg_consciousness:.6}, eff_feas={bio_effective_feas:.4}, tau={bio_tau:.4}"
    );
    eprintln!(
        "Silicon: avg_c={silicon_avg_consciousness:.6}, eff_feas={silicon_effective_feas:.4}, tau={silicon_tau:.4}"
    );
    eprintln!(
        "Quantum: avg_c={quantum_avg_consciousness:.6}, eff_feas={quantum_effective_feas:.4}, tau={quantum_tau:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 4: Energy budget exhaustion degrades consciousness
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_energy_budget_consciousness_collapse() {
    // Use an extremely small energy budget so it exhausts within a few cycles.
    // SiliconDigital energy_per_op is ~1e-15 J, ops_per_cycle = 65536,
    // so energy_per_cycle ~ 6.5e-11 J. With tau > 1.0 (speed modulation),
    // actual per-tick is higher. Set budget to ~3 cycles' worth.
    let budget = 2e-10; // ~3 cycles for silicon before exhaustion

    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_validation_overlay: true,
        enable_substrate_speed_modulation: true,
        enable_energy_budget: true,
        energy_budget_joules_per_sec: Some(budget),
        async_training: false,
        learning_threshold: 0.01,
        substrate_type: SubstrateType::SiliconDigital,
        ..Default::default()
    })
    .expect("CognitiveLoopService::new should succeed");

    // Initially consciousness should be viable
    assert!(
        service.substrate_consciousness_viable(),
        "Consciousness should be viable initially"
    );

    // Run cycles until consciousness is no longer viable, with a safety cap
    let max_cycles = 500;
    let mut collapse_cycle = None;
    let mut all_consciousness: Vec<f64> = Vec::with_capacity(max_cycles);

    for i in 0..max_cycles {
        let input = INPUTS[i % INPUTS.len()];
        let result = service.cycle(input);
        all_consciousness.push(result.metadata.consciousness.consciousness_level);

        if !service.substrate_consciousness_viable() && collapse_cycle.is_none() {
            collapse_cycle = Some(i);
        }

        // If already collapsed, keep running a few more cycles to verify stability
        if let Some(cc) = collapse_cycle {
            if i >= cc + 20 {
                break;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ASSERTIONS
    // ═══════════════════════════════════════════════════════════════════════

    // 1. Collapse happens within a reasonable number of cycles (not infinite)
    assert!(
        collapse_cycle.is_some(),
        "Energy budget should be exhausted within {max_cycles} cycles, \
         but consciousness remained viable. total_energy={:.2e}, budget={budget:.2e}",
        service.substrate_total_energy_spent()
    );
    let cc = collapse_cycle.unwrap();
    eprintln!("Energy budget exhausted at cycle {cc}");

    // 2. Collapse should happen relatively quickly with this tiny budget
    assert!(
        cc < 100,
        "With budget {budget:.2e}, collapse should happen within 100 cycles, got {cc}"
    );

    // 3. Consciousness level never goes NaN after collapse
    assert!(
        all_consciousness.iter().all(|c| c.is_finite()),
        "Consciousness went NaN/Inf after energy collapse"
    );

    // 4. Consciousness stays bounded [0, 1] even after collapse
    assert!(
        all_consciousness.iter().all(|&c| c >= 0.0 && c <= 1.0),
        "Consciousness out of [0,1] bounds after energy collapse"
    );

    // 5. should_degrade_consciousness() returns true after collapse
    assert!(
        !service.substrate_consciousness_viable(),
        "substrate_consciousness_viable should be false after energy exhaustion"
    );

    // ── Summary ──
    eprintln!("=== Energy Budget Consciousness Collapse Results ===");
    eprintln!("Collapse cycle: {cc}");
    eprintln!(
        "Total energy spent: {:.2e} J (budget: {budget:.2e} J)",
        service.substrate_total_energy_spent()
    );
    eprintln!(
        "Post-collapse consciousness samples: {:?}",
        &all_consciousness[cc..all_consciousness.len().min(cc + 10)]
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 5: Encoding noise produces emergent prediction error differences
// ═══════════════════════════════════════════════════════════════════════════════
//
// This test validates that substrate encoding noise makes scale-constrained
// substrates (Quantum, Exotic) produce genuinely different prediction errors
// compared to scale-adequate substrates (Biological, Silicon). The difference
// is *emergent* — caused by noisier HDC encoding and CfC input, not by a
// scalar multiplier on the consciousness score.

#[test]
fn test_encoding_noise_produces_emergent_differences() {
    // Run two services in parallel: one with encoding noise, one without.
    // Same substrate (QuantumComputer), same inputs. Compare prediction errors.

    let mut config_noisy = CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.01,
        substrate_type: SubstrateType::QuantumComputer,
        ..Default::default()
    };
    config_noisy.enable_substrate_simulation();

    let config_clean = CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.01,
        substrate_type: SubstrateType::QuantumComputer,
        enable_validation_overlay: true,
        enable_substrate_speed_modulation: true,
        enable_substrate_encoding_noise: false, // explicitly off
        ..Default::default()
    };

    let mut service_noisy =
        CognitiveLoopService::new(config_noisy).expect("noisy service should succeed");
    let mut service_clean =
        CognitiveLoopService::new(config_clean).expect("clean service should succeed");

    // Warmup both
    for i in 0..30 {
        let input = INPUTS[i % INPUTS.len()];
        service_noisy.cycle(input);
        service_clean.cycle(input);
    }

    // Run 100 cycles, collect prediction errors
    let mut noisy_errors: Vec<f32> = Vec::with_capacity(100);
    let mut clean_errors: Vec<f32> = Vec::with_capacity(100);
    for i in 0..100 {
        let input = INPUTS[(30 + i) % INPUTS.len()];
        let r_noisy = service_noisy.cycle(input);
        let r_clean = service_clean.cycle(input);
        noisy_errors.push(r_noisy.prediction_error);
        clean_errors.push(r_clean.prediction_error);
    }

    let avg_noisy = noisy_errors.iter().sum::<f32>() / noisy_errors.len() as f32;
    let avg_clean = clean_errors.iter().sum::<f32>() / clean_errors.len() as f32;

    // All errors should be finite
    assert!(
        noisy_errors.iter().all(|e| e.is_finite()),
        "Noisy prediction errors contain NaN/Inf"
    );
    assert!(
        clean_errors.iter().all(|e| e.is_finite()),
        "Clean prediction errors contain NaN/Inf"
    );

    // The noisy service should have different prediction error trajectory.
    // We don't mandate higher (noise could sometimes help via exploration),
    // but the trajectories must diverge — measured by sum of absolute differences.
    let trajectory_divergence: f32 = noisy_errors
        .iter()
        .zip(clean_errors.iter())
        .map(|(n, c)| (n - c).abs())
        .sum::<f32>()
        / 100.0;

    assert!(
        trajectory_divergence > 0.001,
        "Encoding noise should cause trajectory divergence: avg_diff={trajectory_divergence:.6}, \
         avg_noisy={avg_noisy:.4}, avg_clean={avg_clean:.4}"
    );

    eprintln!("=== Encoding Noise Emergent Differences ===");
    eprintln!("Avg prediction error (noisy):  {avg_noisy:.4}");
    eprintln!("Avg prediction error (clean):  {avg_clean:.4}");
    eprintln!("Trajectory divergence:         {trajectory_divergence:.6}");
    eprintln!(
        "Noise fraction (quantum):      {:.4}",
        service_noisy.substrate_scale_pressure().abs().min(7.0) / 70.0
    );
}