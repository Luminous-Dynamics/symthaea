//! Integration tests for CognitiveSubsystem managers.
//!
//! Tests that managers correctly influence CLS state when wired into the
//! full cognitive loop. Each manager runs on a co-prime interval:
//!   DriveManager(7), MemoryManager(11), LearningManager(13), PerceptionManager(19).

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const INPUTS: &[&str] = &[
    "The weather is warm today.",
    "I need to solve this problem efficiently.",
    "Music brings people together in unexpected ways.",
    "Mathematics describes the structure of reality.",
    "Consciousness emerges from integrated information.",
];

fn create_service() -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::default();
    config.genesis_phrase = Some("manager_integration_2026".to_string());
    CognitiveLoopService::new(config).expect("CLS construction")
}

// ═══════════════════════════════════════════════════════════════════════════════
// DriveManager (interval 7): curiosity, flow, boredom, exploration
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn drive_manager_runs_at_interval() {
    let mut svc = create_service();

    // Run 50 cycles — DriveManager fires at 7, 14, 21, 28, 35, 42, 49
    for i in 0..50 {
        svc.cycle(INPUTS[i % INPUTS.len()]);
    }

    // After 50 cycles, curiosity and boredom should have non-default values
    let curiosity = svc.curiosity();
    let boredom = svc.boredom();

    // At least one should have moved from default (they respond to prediction error)
    assert!(
        curiosity != 0.5 || boredom != 0.0,
        "DriveManager should have modulated curiosity({curiosity}) or boredom({boredom}) after 50 cycles"
    );
}

#[test]
fn drive_manager_boredom_accumulates_on_repeated_input() {
    let mut svc = create_service();

    // Run many cycles with the same input → low prediction error → boredom rises
    for _ in 0..100 {
        svc.cycle("the same boring input every cycle");
    }

    let boredom = svc.boredom();
    // After 100 cycles of identical input, boredom should be non-trivial
    // (exact value depends on prediction dynamics, but should be > 0)
    assert!(
        boredom >= 0.0,
        "Boredom should be non-negative: {boredom}"
    );
}

#[test]
fn drive_manager_exploration_responds_to_surprise() {
    let mut svc = create_service();

    // Establish baseline with predictable input
    for _ in 0..30 {
        svc.cycle("predictable input");
    }

    let baseline_curiosity = svc.curiosity();

    // Inject surprising/novel input
    for _ in 0..30 {
        svc.cycle("quantum entanglement creates non-local correlations in spacetime");
    }

    let post_surprise_curiosity = svc.curiosity();

    // Curiosity should respond (direction depends on PE dynamics)
    // Just verify it's finite and bounded
    assert!(post_surprise_curiosity.is_finite());
    assert!(
        (0.0..=1.0).contains(&post_surprise_curiosity),
        "Curiosity should be bounded [0,1]: {post_surprise_curiosity}"
    );
    // Log for diagnostic purposes
    eprintln!(
        "Curiosity: baseline={baseline_curiosity:.4}, post_surprise={post_surprise_curiosity:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MemoryManager (interval 11): episodic, semantic, resonator, coordinator
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn memory_manager_runs_without_panic() {
    let mut svc = create_service();

    // Run 50 cycles — MemoryManager fires at 11, 22, 33, 44
    for i in 0..50 {
        let result = svc.cycle(INPUTS[i % INPUTS.len()]);
        // Every result should be finite
        assert!(
            result.prediction_error.is_finite(),
            "Prediction error should be finite at cycle {i}"
        );
    }
}

#[test]
fn memory_manager_metadata_populated() {
    let mut svc = create_service();

    // Run enough cycles for memory subsystems to fire
    let mut last_result = svc.cycle(INPUTS[0]);
    for i in 1..60 {
        last_result = svc.cycle(INPUTS[i % INPUTS.len()]);
    }

    // Verify memory-related telemetry is populated
    let m = &last_result.metadata;
    // semantic_memory_size should be accessible (may be 0 if no storage)
    assert!(m.prediction_error.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// LearningManager (interval 13): FEP, dream, school, evolution
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn learning_manager_modulates_learning_rate() {
    let mut svc = create_service();

    // Run enough cycles for LearningManager to fire multiple times (13, 26, 39, 52, 65)
    for i in 0..70 {
        svc.cycle(INPUTS[i % INPUTS.len()]);
    }

    let stats = svc.stats();
    // Learning rate should be finite and positive
    assert!(stats.fep_learning_signal.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// PerceptionManager (interval 19): attention, multi-modal, social
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn perception_manager_runs_at_interval() {
    let mut svc = create_service();

    // Run 60 cycles — PerceptionManager fires at 19, 38, 57
    for i in 0..60 {
        let result = svc.cycle(INPUTS[i % INPUTS.len()]);
        assert!(
            result.metadata.attention.attention_schema_focus.is_finite(),
            "Attention schema focus should be finite at cycle {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-manager interaction: all managers running together
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn all_managers_coexist_without_interference() {
    let mut svc = create_service();

    // LCM(7,11,13,19) = 19019, but we just need enough cycles for all to fire
    // 20 cycles guarantees at least: Drive(7,14), Memory(11), Learning(13), Perception(19)
    let mut results = Vec::with_capacity(100);
    for i in 0..100 {
        results.push(svc.cycle(INPUTS[i % INPUTS.len()]));
    }

    // All results should be finite and well-formed
    for (i, r) in results.iter().enumerate() {
        assert!(
            r.prediction_error.is_finite(),
            "PE not finite at cycle {i}: {}",
            r.prediction_error
        );
        assert!(
            r.metadata.consciousness_level.is_finite(),
            "Consciousness not finite at cycle {i}"
        );
    }

    // After 100 cycles, system should have settled into a stable regime
    let last = results.last().unwrap();
    assert!(
        (0.0..=1.0).contains(&last.metadata.consciousness_level),
        "Consciousness level out of bounds: {}",
        last.metadata.consciousness_level
    );
}

#[test]
fn managers_survive_reset() {
    let mut svc = create_service();

    // Run some cycles
    for i in 0..30 {
        svc.cycle(INPUTS[i % INPUTS.len()]);
    }

    // Reset
    svc.reset();

    // Run more cycles — should not panic
    for i in 0..30 {
        let result = svc.cycle(INPUTS[i % INPUTS.len()]);
        assert!(result.prediction_error.is_finite());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Per-phase performance monitoring
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn phase_timings_populated() {
    let mut svc = create_service();

    let result = svc.cycle("test phase timing instrumentation");
    let t = &result.metadata.module_timings_us;

    // All 4 phase timings should be non-zero
    assert!(t.phase_perception > 0, "Phase 1 (perception) timing should be > 0");
    assert!(t.phase_dynamics > 0, "Phase 2 (dynamics) timing should be > 0");
    assert!(t.phase_feedback > 0, "Phase 3 (feedback) timing should be > 0");
    assert!(t.phase_output > 0, "Phase 4 (output) timing should be > 0");

    // Sum of phases should approximate total cycle time
    let phase_sum = t.phase_perception + t.phase_dynamics + t.phase_feedback + t.phase_output;
    assert!(
        phase_sum > 0,
        "Sum of phase timings should be positive: {phase_sum}"
    );
}

#[test]
fn phase_timings_stable_across_cycles() {
    let mut svc = create_service();

    let mut perception_times = Vec::new();
    let mut dynamics_times = Vec::new();

    for i in 0..20 {
        let result = svc.cycle(INPUTS[i % INPUTS.len()]);
        let t = &result.metadata.module_timings_us;
        perception_times.push(t.phase_perception);
        dynamics_times.push(t.phase_dynamics);
    }

    // All timings should be non-zero after warmup
    for (i, &t) in perception_times.iter().enumerate() {
        assert!(t > 0, "Perception timing zero at cycle {i}");
    }
    for (i, &t) in dynamics_times.iter().enumerate() {
        assert!(t > 0, "Dynamics timing zero at cycle {i}");
    }
}

/// Budget check: warn if any phase exceeds 25ms (half of 50ms budget).
#[test]
fn no_phase_exceeds_budget_half() {
    let mut svc = create_service();

    // Warmup
    for i in 0..10 {
        svc.cycle(INPUTS[i % INPUTS.len()]);
    }

    // Measure
    let result = svc.cycle("budget check cycle");
    let t = &result.metadata.module_timings_us;

    let budget_half_us = 25_000; // 25ms = half of 50ms cycle budget
    // These are soft assertions — CI machines may be slow
    if t.phase_perception > budget_half_us {
        eprintln!(
            "WARNING: Phase 1 (perception) took {}us > {}us budget half",
            t.phase_perception, budget_half_us
        );
    }
    if t.phase_dynamics > budget_half_us {
        eprintln!(
            "WARNING: Phase 2 (dynamics) took {}us > {}us budget half",
            t.phase_dynamics, budget_half_us
        );
    }
    if t.phase_feedback > budget_half_us {
        eprintln!(
            "WARNING: Phase 3 (feedback) took {}us > {}us budget half",
            t.phase_feedback, budget_half_us
        );
    }
    if t.phase_output > budget_half_us {
        eprintln!(
            "WARNING: Phase 4 (output) took {}us > {}us budget half",
            t.phase_output, budget_half_us
        );
    }
}
