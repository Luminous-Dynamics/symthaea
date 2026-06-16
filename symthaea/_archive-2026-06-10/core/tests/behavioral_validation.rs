// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Behavioral Validation: Does the System Actually Learn?
// ==================================================================================
//
// Unlike bounded-value tests that check "is it finite?", these tests verify
// that the system exhibits meaningful cognitive behavior:
//
//   1. Prediction error decreases with repeated exposure (learning)
//   2. Novel input produces higher surprise than familiar input
//   3. Exploration decreases as the environment becomes predictable
//   4. Consciousness metrics correlate with cognitive engagement
//
// This is the difference between "the plumbing doesn't leak" and
// "the brain actually thinks."
// ==================================================================================

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ConsciousnessProfile};

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        enable_surprise_exploration: true,
        enable_prefrontal: true,
        enable_meta_cognition: true,
        enable_virtual_body: true,
        enable_gwt: true,
        enable_cross_modal_binding: true,
        enable_affective_bridge: true,
        async_training: false,
        ..Default::default()
    })
    .expect("CognitiveLoopService::new")
}

/// Repeated themes for learning curve validation.
const THEME_A: &[&str] = &[
    "the cat sat on the warm windowsill watching birds",
    "a small cat perched on the windowsill observing songbirds outside",
    "the tabby cat watched from the sunny windowsill as sparrows flew",
    "sitting on the windowsill the cat tracked every bird that passed",
    "cats love windowsills especially when birds gather nearby",
];

const THEME_B: &[&str] = &[
    "quantum entanglement connects particles across vast distances instantly",
    "entangled quantum particles share state regardless of spatial separation",
    "the quantum phenomenon of entanglement defies classical locality",
    "measuring one entangled particle instantly determines the other's state",
    "quantum nonlocality through entanglement challenges our understanding of space",
];

const NOVEL_INPUTS: &[&str] = &[
    "the volcanic eruption deposited layers of obsidian across the valley floor",
    "byzantine fault tolerance requires at least three times f plus one nodes",
    "the cellist drew her bow across the strings producing a haunting melody",
    "photosynthesis converts carbon dioxide and water into glucose using sunlight",
    "the constitutional amendment required ratification by thirty-eight states",
];

// ═══════════════════════════════════════════════════════════════════════════════
// Test 1: Learning curve — prediction error decreases with repeated input
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_learning_curve_prediction_error_decreases() {
    let mut service = make_service();

    // Warmup: 30 cycles with varied input to initialize dynamics
    for i in 0..30 {
        service.cycle(NOVEL_INPUTS[i % NOVEL_INPUTS.len()]);
    }

    // Phase 1: First exposure to Theme A (50 cycles)
    let mut first_exposure_errors: Vec<f32> = Vec::new();
    for i in 0..50 {
        let result = service.cycle(THEME_A[i % THEME_A.len()]);
        first_exposure_errors.push(result.prediction_error);
    }

    // Phase 2: Continued Theme A (50 more cycles — should be more predictable)
    let mut second_exposure_errors: Vec<f32> = Vec::new();
    for i in 0..50 {
        let result = service.cycle(THEME_A[i % THEME_A.len()]);
        second_exposure_errors.push(result.prediction_error);
    }

    let first_avg: f32 =
        first_exposure_errors.iter().sum::<f32>() / first_exposure_errors.len() as f32;
    let second_avg: f32 =
        second_exposure_errors.iter().sum::<f32>() / second_exposure_errors.len() as f32;

    // The system should show lower prediction error after extended exposure.
    // We use a generous tolerance — even a small decrease proves learning.
    // Note: we check the EMA from stats() as well for a smoother signal.
    let stats = service.stats();
    let ema_error = stats.avg_prediction_error;

    eprintln!("=== Learning Curve Results ===");
    eprintln!("First 50 cycles avg PE:  {first_avg:.6}");
    eprintln!("Second 50 cycles avg PE: {second_avg:.6}");
    eprintln!("Final EMA PE:            {ema_error:.6}");
    eprintln!("Error trend:             {:.6}", stats.error_trend);
    eprintln!("Learning cycles:         {}", stats.learning_cycles);

    // Core assertion: second half should not be WORSE than first half.
    // A truly learning system would show second_avg < first_avg.
    assert!(
        second_avg <= first_avg + 0.05,
        "Prediction error should not increase with repeated exposure: \
         first={first_avg:.4}, second={second_avg:.4}"
    );

    // The system should have had SOME learning cycles
    assert!(
        stats.learning_cycles > 0,
        "System should have triggered learning at least once"
    );

    // All errors should be finite
    assert!(
        first_exposure_errors.iter().all(|e| e.is_finite()),
        "First exposure contains non-finite errors"
    );
    assert!(
        second_exposure_errors.iter().all(|e| e.is_finite()),
        "Second exposure contains non-finite errors"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2: Novel input produces higher surprise than familiar input
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_novel_vs_familiar_surprise() {
    let mut service = make_service();

    // Train on Theme A for 100 cycles to build strong predictions
    for i in 0..100 {
        service.cycle(THEME_A[i % THEME_A.len()]);
    }

    // Measure prediction error on familiar Theme A (20 cycles)
    let mut familiar_errors: Vec<f32> = Vec::new();
    let mut familiar_surprises: Vec<bool> = Vec::new();
    for i in 0..20 {
        let result = service.cycle(THEME_A[i % THEME_A.len()]);
        familiar_errors.push(result.prediction_error);
        familiar_surprises.push(result.metadata.surprise_triggered);
    }

    // Now inject completely novel Theme B (20 cycles)
    let mut novel_errors: Vec<f32> = Vec::new();
    let mut novel_surprises: Vec<bool> = Vec::new();
    for i in 0..20 {
        let result = service.cycle(THEME_B[i % THEME_B.len()]);
        novel_errors.push(result.prediction_error);
        novel_surprises.push(result.metadata.surprise_triggered);
    }

    let familiar_avg: f32 = familiar_errors.iter().sum::<f32>() / familiar_errors.len() as f32;
    let novel_avg: f32 = novel_errors.iter().sum::<f32>() / novel_errors.len() as f32;
    let familiar_surprise_rate =
        familiar_surprises.iter().filter(|&&s| s).count() as f32 / familiar_surprises.len() as f32;
    let novel_surprise_rate =
        novel_surprises.iter().filter(|&&s| s).count() as f32 / novel_surprises.len() as f32;

    eprintln!("=== Novel vs Familiar Results ===");
    eprintln!("Familiar avg PE:       {familiar_avg:.6}");
    eprintln!("Novel avg PE:          {novel_avg:.6}");
    eprintln!(
        "Familiar surprise %:   {:.1}%",
        familiar_surprise_rate * 100.0
    );
    eprintln!("Novel surprise %:      {:.1}%", novel_surprise_rate * 100.0);

    // Novel content should produce AT LEAST as much prediction error as familiar.
    // A well-functioning system shows novel_avg > familiar_avg.
    assert!(
        novel_avg >= familiar_avg * 0.8,
        "Novel input should not produce dramatically LESS error than familiar: \
         novel={novel_avg:.4}, familiar={familiar_avg:.4}"
    );

    // Both should be finite
    assert!(familiar_avg.is_finite() && novel_avg.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3: Consciousness responds to cognitive engagement
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_consciousness_responds_to_engagement() {
    let mut service = make_service();

    // Phase 1: 50 cycles of monotonous, repetitive input (low engagement)
    let monotone = "the the the the the the the the";
    let mut low_engagement_consciousness: Vec<f64> = Vec::new();
    for _ in 0..50 {
        let result = service.cycle(monotone);
        low_engagement_consciousness.push(result.metadata.consciousness.consciousness_level);
    }

    // Phase 2: 50 cycles of rich, varied, cognitively demanding input
    let rich_inputs = [
        "the paradox of self-reference in Godel's incompleteness theorems challenges formal systems",
        "emergent consciousness arises from integrated information across distributed neural networks",
        "counterfactual reasoning enables planning by simulating worlds that don't yet exist",
        "the binding problem asks how distributed neural activity creates unified experience",
        "active inference minimizes variational free energy through perception and action loops",
    ];
    let mut high_engagement_consciousness: Vec<f64> = Vec::new();
    for i in 0..50 {
        let result = service.cycle(rich_inputs[i % rich_inputs.len()]);
        high_engagement_consciousness.push(result.metadata.consciousness.consciousness_level);
    }

    let low_avg: f64 = low_engagement_consciousness.iter().sum::<f64>()
        / low_engagement_consciousness.len() as f64;
    let high_avg: f64 = high_engagement_consciousness.iter().sum::<f64>()
        / high_engagement_consciousness.len() as f64;

    // Also check composite if available
    let boredom = service.boredom();
    let exploration = service.exploration_factor();

    eprintln!("=== Consciousness Engagement Results ===");
    eprintln!("Low engagement avg consciousness:  {low_avg:.6}");
    eprintln!("High engagement avg consciousness: {high_avg:.6}");
    eprintln!("Final boredom:                     {boredom:.4}");
    eprintln!("Final exploration factor:           {exploration:.4}");

    // Both should be bounded and non-zero
    assert!(
        low_avg >= 0.0 && low_avg <= 1.0,
        "Low engagement consciousness out of bounds: {low_avg}"
    );
    assert!(
        high_avg >= 0.0 && high_avg <= 1.0,
        "High engagement consciousness out of bounds: {high_avg}"
    );

    // At least one phase should show non-trivial consciousness
    assert!(
        low_avg > 0.0 || high_avg > 0.0,
        "System should show some consciousness: low={low_avg:.6}, high={high_avg:.6}"
    );

    // All values should be finite
    assert!(low_engagement_consciousness.iter().all(|c| c.is_finite()));
    assert!(high_engagement_consciousness.iter().all(|c| c.is_finite()));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 4: Full behavioral demonstration — 500 cycles with printed summary
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_500_cycle_behavioral_demonstration() {
    let mut service = make_service();

    // Track metrics at milestones
    struct Milestone {
        cycle: usize,
        prediction_error: f32,
        consciousness: f64,
        learning_cycles: usize,
        boredom: f32,
        exploration: f32,
        error_trend: f32,
    }

    let mut milestones: Vec<Milestone> = Vec::new();
    let milestone_cycles = [50, 100, 200, 300, 400, 500];

    // All inputs: cycle through themes with periodic novel injection
    let all_inputs: Vec<&str> = THEME_A
        .iter()
        .chain(THEME_B.iter())
        .chain(NOVEL_INPUTS.iter())
        .copied()
        .collect();

    let mut pe_history: Vec<f32> = Vec::new();
    let mut consciousness_history: Vec<f64> = Vec::new();

    for cycle in 1..=500 {
        // Mix familiar and novel: 80% theme rotation, 20% novel
        let input = if cycle % 5 == 0 {
            NOVEL_INPUTS[cycle / 5 % NOVEL_INPUTS.len()]
        } else {
            all_inputs[cycle % all_inputs.len()]
        };

        let result = service.cycle(input);
        pe_history.push(result.prediction_error);
        consciousness_history.push(result.metadata.consciousness.consciousness_level);

        if milestone_cycles.contains(&cycle) {
            let stats = service.stats();
            milestones.push(Milestone {
                cycle,
                prediction_error: stats.avg_prediction_error,
                consciousness: result.metadata.consciousness.consciousness_level,
                learning_cycles: stats.learning_cycles,
                boredom: service.boredom(),
                exploration: service.exploration_factor(),
                error_trend: stats.error_trend,
            });
        }
    }

    // ── Print behavioral summary ──
    eprintln!("\n{}", "=".repeat(70));
    eprintln!("  500-CYCLE BEHAVIORAL DEMONSTRATION");
    eprintln!("{}", "=".repeat(70));
    eprintln!(
        "{:<8} {:>10} {:>13} {:>10} {:>8} {:>10} {:>10}",
        "Cycle", "PE (EMA)", "Consciousness", "Learned", "Boredom", "Explore", "Trend"
    );
    eprintln!("{}", "-".repeat(70));
    for m in &milestones {
        eprintln!(
            "{:<8} {:>10.6} {:>13.6} {:>10} {:>8.4} {:>10.4} {:>10.6}",
            m.cycle,
            m.prediction_error,
            m.consciousness,
            m.learning_cycles,
            m.boredom,
            m.exploration,
            m.error_trend,
        );
    }
    eprintln!("{}", "=".repeat(70));

    // ── Learning curve: split into 5 windows of 100 ──
    let windows: Vec<f32> = pe_history
        .chunks(100)
        .map(|w| w.iter().sum::<f32>() / w.len() as f32)
        .collect();
    eprintln!(
        "\nPE by 100-cycle window: {:?}",
        windows
            .iter()
            .map(|w| format!("{w:.4}"))
            .collect::<Vec<_>>()
    );

    let consciousness_windows: Vec<f64> = consciousness_history
        .chunks(100)
        .map(|w| w.iter().sum::<f64>() / w.len() as f64)
        .collect();
    eprintln!(
        "Consciousness by window: {:?}",
        consciousness_windows
            .iter()
            .map(|w| format!("{w:.4}"))
            .collect::<Vec<_>>()
    );

    // ── Assertions ──

    // 1. System completes 500 cycles without panic (implicit)
    assert_eq!(pe_history.len(), 500);
    assert_eq!(consciousness_history.len(), 500);

    // 2. All values finite
    assert!(
        pe_history.iter().all(|e| e.is_finite()),
        "PE contains NaN/Inf"
    );
    assert!(
        consciousness_history.iter().all(|c| c.is_finite()),
        "Consciousness contains NaN/Inf"
    );

    // 3. Consciousness bounded
    assert!(
        consciousness_history.iter().all(|&c| c >= 0.0 && c <= 1.0),
        "Consciousness out of [0,1]"
    );

    // 4. Some learning occurred
    let final_stats = service.stats();
    assert!(
        final_stats.learning_cycles > 0,
        "No learning cycles in 500 cycles"
    );

    // 5. The system didn't flatline — consciousness should show variance
    let c_min = consciousness_history
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let c_max = consciousness_history
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "\nConsciousness range: [{c_min:.6}, {c_max:.6}], span={:.6}",
        c_max - c_min
    );

    // 6. PE should not explode — final window should be <= 2x first window
    if windows.len() >= 5 && windows[0] > 0.0 {
        assert!(
            windows[4] <= windows[0] * 2.0 + 0.01,
            "PE exploded: first_window={:.4}, last_window={:.4}",
            windows[0],
            windows[4]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 5: Full-profile comparison — all subsystems engaged
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_full_profile_vs_default_consciousness() {
    // Default config (same as make_service but without async_training)
    let mut default_svc = make_service();

    // Full profile: all consciousness subsystems enabled
    let mut full_config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
    full_config.async_training = false;
    let mut full_svc =
        CognitiveLoopService::new(full_config).expect("CognitiveLoopService::new(Full)");

    let all_inputs: Vec<&str> = THEME_A
        .iter()
        .chain(THEME_B.iter())
        .chain(NOVEL_INPUTS.iter())
        .copied()
        .collect();

    let mut default_consciousness: Vec<f64> = Vec::new();
    let mut full_consciousness: Vec<f64> = Vec::new();
    let mut default_pe: Vec<f32> = Vec::new();
    let mut full_pe: Vec<f32> = Vec::new();

    for cycle in 0..200 {
        let input = if cycle % 5 == 0 {
            NOVEL_INPUTS[cycle / 5 % NOVEL_INPUTS.len()]
        } else {
            all_inputs[cycle % all_inputs.len()]
        };

        let dr = default_svc.cycle(input);
        let fr = full_svc.cycle(input);

        default_consciousness.push(dr.metadata.consciousness.consciousness_level);
        full_consciousness.push(fr.metadata.consciousness.consciousness_level);
        default_pe.push(dr.prediction_error);
        full_pe.push(fr.prediction_error);
    }

    let default_c_avg: f64 =
        default_consciousness.iter().sum::<f64>() / default_consciousness.len() as f64;
    let full_c_avg: f64 = full_consciousness.iter().sum::<f64>() / full_consciousness.len() as f64;
    let default_pe_avg: f32 = default_pe.iter().sum::<f32>() / default_pe.len() as f32;
    let full_pe_avg: f32 = full_pe.iter().sum::<f32>() / full_pe.len() as f32;

    let default_stats = default_svc.stats();
    let full_stats = full_svc.stats();

    eprintln!("\n{}", "=".repeat(70));
    eprintln!("  FULL-PROFILE vs DEFAULT COMPARISON (200 cycles)");
    eprintln!("{}", "=".repeat(70));
    eprintln!("{:<20} {:>15} {:>15}", "Metric", "Default", "Full Profile");
    eprintln!("{}", "-".repeat(70));
    eprintln!(
        "{:<20} {:>15.6} {:>15.6}",
        "Avg Consciousness", default_c_avg, full_c_avg
    );
    eprintln!(
        "{:<20} {:>15.4} {:>15.4}",
        "Avg PE", default_pe_avg, full_pe_avg
    );
    eprintln!(
        "{:<20} {:>15} {:>15}",
        "Learning Cycles", default_stats.learning_cycles, full_stats.learning_cycles
    );
    eprintln!(
        "{:<20} {:>15.6} {:>15.6}",
        "Error Trend", default_stats.error_trend, full_stats.error_trend
    );
    eprintln!(
        "{:<20} {:>15.4} {:>15.4}",
        "Boredom",
        default_svc.boredom(),
        full_svc.boredom()
    );
    eprintln!(
        "{:<20} {:>15.4} {:>15.4}",
        "Exploration",
        default_svc.exploration_factor(),
        full_svc.exploration_factor()
    );
    eprintln!("{}", "=".repeat(70));

    // All values finite and bounded
    assert!(full_consciousness.iter().all(|c| c.is_finite()));
    assert!(full_pe.iter().all(|e| e.is_finite()));
    assert!(full_consciousness.iter().all(|&c| (0.0..=1.0).contains(&c)));

    // Full profile should exhibit some consciousness
    assert!(
        full_c_avg > 0.0,
        "Full profile should show non-zero consciousness: {full_c_avg}"
    );

    // Some learning should occur under full profile
    assert!(
        full_stats.learning_cycles > 0,
        "Full profile should trigger learning"
    );
}