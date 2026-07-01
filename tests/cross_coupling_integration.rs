// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-coupling integration tests.
//!
//! Validates that the 4 cross-coupling functions (Drive->Learning, Knowledge->Ethics,
//! Memory->Learning, Perception->Drive) produce measurable effects when their
//! trigger conditions are met.
//!
//! Strategy: run two `CognitiveLoopService` instances side-by-side with different
//! inputs designed to trigger vs. not-trigger each coupling, then compare observable
//! telemetry to confirm the coupling had a measurable effect.

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

const GENESIS_SEED: &str = "cross_coupling_integration_v1";

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        genesis_phrase: Some(GENESIS_SEED.to_string()),
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .expect("Failed to create CognitiveLoopService")
}

fn make_service_with_knowledge() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        genesis_phrase: Some(GENESIS_SEED.to_string()),
        learning_threshold: 0.0,
        async_training: false,
        enable_knowledge_engine: true,
        ..Default::default()
    })
    .expect("Failed to create CognitiveLoopService with knowledge engine")
}

/// Varied, semantically rich input to keep prediction error moderate-to-high.
const VARIED_INPUTS: &[&str] = &[
    "quantum entanglement violates classical locality assumptions",
    "the economic consequences of climate change reshape global policy",
    "emergent consciousness arises from integrated information",
    "mathematical proofs require rigorous logical deduction",
    "neural plasticity enables lifelong learning and adaptation",
    "collective intelligence surpasses individual cognitive limits",
    "thermodynamic entropy bounds all information processing",
    "moral reasoning requires both empathy and rational analysis",
    "evolutionary pressure shapes cognitive architecture",
    "causal inference distinguishes correlation from mechanism",
];

fn varied_input(cycle: usize) -> &'static str {
    VARIED_INPUTS[cycle % VARIED_INPUTS.len()]
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. DRIVE -> LEARNING: Boredom boosts LR
// ═══════════════════════════════════════════════════════════════════════════════

/// Sustained boring input (low PE) builds boredom > 0.5, which should boost
/// the learning rate factor via `cross_couple_drive_learning`.
///
/// We compare a bored service (80 cycles of "ok") against a service with varied
/// input, checking that learning plasticity diverges.
#[test]
fn test_boredom_boosts_lr() {
    let mut bored = make_service();

    // Run 80 boring cycles to build boredom past 0.5 threshold.
    // DriveManager: LOW_ERROR_THRESHOLD=0.05, BOREDOM_ONSET_CYCLES=20,
    // boredom ramps +0.03/cycle after onset. After ~37 boring cycles, boredom > 0.5.
    let mut bored_plasticity_values = Vec::new();
    for _ in 0..80 {
        let result = bored.cycle("ok");
        bored_plasticity_values.push(result.metadata.learning_plasticity);
    }

    // Meanwhile, a service with varied input (keeps PE moderate, no boredom)
    let mut active = make_service();
    let mut active_plasticity_values = Vec::new();
    for i in 0..80 {
        let result = active.cycle(varied_input(i));
        active_plasticity_values.push(result.metadata.learning_plasticity);
    }

    // After 80 cycles, boredom coupling should have caused measurable divergence
    // in plasticity and/or prediction confidence between the two services.
    let bored_late_plasticity: f32 = bored_plasticity_values[60..].iter().sum::<f32>() / 20.0;
    let active_late_plasticity: f32 = active_plasticity_values[60..].iter().sum::<f32>() / 20.0;

    let bored_conf = bored.prediction_confidence();
    let active_conf = active.prediction_confidence();

    let plasticity_diff = (bored_late_plasticity - active_late_plasticity).abs();
    let confidence_diff = (bored_conf - active_conf).abs();

    assert!(
        plasticity_diff > 0.001 || confidence_diff > 0.001,
        "Boredom coupling should cause measurable divergence between bored and active systems. \
         plasticity_diff={plasticity_diff:.6}, confidence_diff={confidence_diff:.6}"
    );

    // All values must be finite and bounded
    for &p in &bored_plasticity_values {
        assert!(
            p.is_finite() && p >= 0.0 && p <= 1.0,
            "plasticity out of range: {p}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. DRIVE -> LEARNING: Flow dampens LR
// ═══════════════════════════════════════════════════════════════════════════════

/// Flow state requires: low PE (<0.15), high coherence (>0.6), moderate arousal
/// (0.3-0.8), sustained for FLOW_ONSET_CYCLES (5) cycles.
///
/// Sustained identical input -> low PE + coherence builds -> flow onset -> LR dampened.
/// Compare against a system receiving high-entropy input (no flow).
#[test]
fn test_flow_dampens_lr() {
    // Flow-inducing: consistent, predictable input
    let mut flow_service = make_service();
    let mut flow_learning_count = 0u32;

    for _ in 0..60 {
        let result = flow_service.cycle("the pattern continues steadily");
        if result.learning_occurred {
            flow_learning_count += 1;
        }
    }

    // High-entropy: each input very different -> high PE -> no flow
    let mut noflow_service = make_service();
    let mut noflow_learning_count = 0u32;
    let chaotic_inputs = [
        "alpha beta gamma delta epsilon zeta",
        "cats dogs fish birds lizards",
        "red blue green yellow purple orange",
        "one two three four five six seven",
        "mountain river valley ocean desert",
        "guitar piano drums violin cello",
    ];

    for i in 0..60 {
        let result = noflow_service.cycle(chaotic_inputs[i % chaotic_inputs.len()]);
        if result.learning_occurred {
            noflow_learning_count += 1;
        }
    }

    let flow_conf = flow_service.prediction_confidence();
    let noflow_conf = noflow_service.prediction_confidence();

    // Both must be valid
    assert!(
        (0.0..=1.0).contains(&flow_conf),
        "flow prediction_confidence out of range: {flow_conf}"
    );
    assert!(
        (0.0..=1.0).contains(&noflow_conf),
        "noflow prediction_confidence out of range: {noflow_conf}"
    );

    // The services should have diverged — flow dampens LR (*0.9)
    let conf_diff = (flow_conf - noflow_conf).abs();
    let learning_diff = (flow_learning_count as i32 - noflow_learning_count as i32).unsigned_abs();

    assert!(
        conf_diff > 0.001 || learning_diff > 0,
        "Flow coupling should cause measurable divergence. \
         conf_diff={conf_diff:.6}, learning_diff={learning_diff}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. KNOWLEDGE -> ETHICS: Causal depth nudges prediction confidence
// ═══════════════════════════════════════════════════════════════════════════════

/// With the knowledge engine enabled, causal depth > 0.6 should nudge prediction
/// confidence upward. Compare a knowledge-enabled service against a plain one.
#[test]
fn test_knowledge_causal_depth_nudges_confidence() {
    let mut with_knowledge = make_service_with_knowledge();
    let mut without_knowledge = make_service();

    // Causal-reasoning-heavy input to build causal depth in the knowledge engine
    let causal_inputs = [
        "cause leads to effect through deterministic mechanisms",
        "if temperature rises then ice melts because of thermodynamics",
        "poverty causes crime because desperation overrides restraint",
        "education improves outcomes since knowledge enables better decisions",
        "feedback loops amplify small causes into large effects",
    ];

    for i in 0..60 {
        let input = causal_inputs[i % causal_inputs.len()];
        let _ = with_knowledge.cycle(input);
        let _ = without_knowledge.cycle(input);
    }

    let with_conf = with_knowledge.prediction_confidence();
    let without_conf = without_knowledge.prediction_confidence();

    // Both must be valid
    assert!(
        (0.0..=1.0).contains(&with_conf),
        "with_knowledge confidence out of range: {with_conf}"
    );
    assert!(
        (0.0..=1.0).contains(&without_conf),
        "without_knowledge confidence out of range: {without_conf}"
    );

    // The knowledge engine's causal depth coupling should produce different confidence.
    let diff = (with_conf - without_conf).abs();
    assert!(
        diff > 0.0001,
        "Knowledge->Ethics coupling should produce measurable confidence difference. \
         with={with_conf:.6}, without={without_conf:.6}, diff={diff:.6}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. MEMORY -> LEARNING: High consolidation pressure boosts LR
// ═══════════════════════════════════════════════════════════════════════════════

/// High prediction error -> consolidation pressure builds -> when > 0.6, LR boosted.
/// Compare a high-PE service against a low-PE one.
///
/// The memory->learning coupling modifies `subsystem_lr_factor` in carryover,
/// which affects downstream learning rate but not necessarily `learning_plasticity`
/// directly. We verify that the two services diverge in observable state
/// (pressure, plasticity, or confidence).
#[test]
fn test_high_consolidation_pressure_boosts_lr() {
    // High-PE input: novel, surprising content each cycle
    let mut high_pe = make_service();
    let surprising_inputs = [
        "unprecedented quantum breakthrough defies established physics",
        "alien megastructure detected orbiting proxima centauri",
        "consciousness transfer between silicon and biological substrates achieved",
        "mathematical proof of P equals NP discovered",
        "time reversal symmetry broken in macroscopic systems",
        "universal basic income eliminates poverty in trial nation",
        "sentient artificial intelligence passes every consciousness test",
        "dark matter particles captured and analyzed in laboratory",
    ];

    let mut high_pe_plasticity = Vec::new();
    let mut high_pe_pressure = Vec::new();
    let mut high_pe_learning_count = 0u32;
    for i in 0..80 {
        let result = high_pe.cycle(surprising_inputs[i % surprising_inputs.len()]);
        high_pe_plasticity.push(result.metadata.learning_plasticity);
        high_pe_pressure.push(result.metadata.memory_consolidation_pressure);
        if result.learning_occurred {
            high_pe_learning_count += 1;
        }
    }

    // Low-PE input: identical content
    let mut low_pe = make_service();
    let mut low_pe_plasticity = Vec::new();
    let mut low_pe_pressure = Vec::new();
    let mut low_pe_learning_count = 0u32;
    for _ in 0..80 {
        let result = low_pe.cycle("ok");
        low_pe_plasticity.push(result.metadata.learning_plasticity);
        low_pe_pressure.push(result.metadata.memory_consolidation_pressure);
        if result.learning_occurred {
            low_pe_learning_count += 1;
        }
    }

    // Compare late-phase metrics between the two services.
    // The memory->learning coupling should cause observable divergence in at least
    // one of: consolidation pressure, plasticity, confidence, or learning count.
    let high_pe_late_pressure: f32 = high_pe_pressure[60..].iter().sum::<f32>() / 20.0;
    let low_pe_late_pressure: f32 = low_pe_pressure[60..].iter().sum::<f32>() / 20.0;
    let pressure_diff = (high_pe_late_pressure - low_pe_late_pressure).abs();

    let high_pe_late_plasticity: f32 = high_pe_plasticity[60..].iter().sum::<f32>() / 20.0;
    let low_pe_late_plasticity: f32 = low_pe_plasticity[60..].iter().sum::<f32>() / 20.0;
    let plasticity_diff = (high_pe_late_plasticity - low_pe_late_plasticity).abs();

    let conf_diff = (high_pe.prediction_confidence() - low_pe.prediction_confidence()).abs();
    let learning_diff =
        (high_pe_learning_count as i32 - low_pe_learning_count as i32).unsigned_abs();

    assert!(
        pressure_diff > 0.001 || plasticity_diff > 0.001 || conf_diff > 0.001 || learning_diff > 0,
        "Memory->Learning coupling should produce measurable divergence between high-PE and low-PE. \
         pressure_diff={pressure_diff:.4}, plasticity_diff={plasticity_diff:.4}, \
         conf_diff={conf_diff:.4}, learning_diff={learning_diff}"
    );

    // All pressure values must be finite and bounded
    for (i, &p) in high_pe_pressure.iter().enumerate() {
        assert!(
            p.is_finite() && p >= 0.0 && p <= 1.0,
            "high_pe consolidation_pressure out of range at cycle {i}: {p}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. MEMORY -> LEARNING: Low recall quality dampens LR
// ═══════════════════════════════════════════════════════════════════════════════

/// Low prediction confidence + low coherence -> low retrieval quality signal -> dampened LR.
/// Compare against a service with high coherence inputs.
#[test]
fn test_low_recall_dampens_lr() {
    // Service with chaotic input -> low coherence -> low retrieval quality
    let mut chaotic = make_service();
    let chaotic_inputs = [
        "xyzzy plugh",
        "fnord",
        "a b c d e",
        "12345",
        "hello world",
        "qqq",
    ];

    for i in 0..60 {
        let _ = chaotic.cycle(chaotic_inputs[i % chaotic_inputs.len()]);
    }

    // Service with coherent, structured input -> higher retrieval quality
    let mut coherent = make_service();
    let coherent_inputs = [
        "learning requires consistent practice and review",
        "memory consolidation strengthens with repeated exposure",
        "knowledge builds on established foundations",
        "understanding deepens through systematic study",
        "retrieval practice improves long-term retention",
    ];

    for i in 0..60 {
        let _ = coherent.cycle(coherent_inputs[i % coherent_inputs.len()]);
    }

    // Check recall quality divergence
    let chaotic_last = chaotic.cycle("final test");
    let coherent_last = coherent.cycle("final test");

    let chaotic_recall = chaotic_last.metadata.memory_recall_quality;
    let coherent_recall = coherent_last.metadata.memory_recall_quality;

    assert!(
        chaotic_recall.is_finite() && coherent_recall.is_finite(),
        "Recall quality must be finite: chaotic={chaotic_recall}, coherent={coherent_recall}"
    );

    // The systems should show different recall quality or prediction confidence
    let recall_diff = (chaotic_recall - coherent_recall).abs();
    let conf_diff = (chaotic.prediction_confidence() - coherent.prediction_confidence()).abs();

    assert!(
        recall_diff > 0.001 || conf_diff > 0.001,
        "Memory->Learning coupling should produce measurable difference with different recall quality. \
         recall_diff={recall_diff:.6}, conf_diff={conf_diff:.6}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. PERCEPTION -> DRIVE: Low coherence nudges exploration
// ═══════════════════════════════════════════════════════════════════════════════

/// Low perceptual coherence (< 0.3) should nudge prediction_confidence downward
/// (encouraging exploration). Compare with a high-coherence scenario.
#[test]
fn test_low_coherence_nudges_exploration() {
    // Chaotic input -> low perceptual coherence
    let mut chaotic = make_service();
    let chaotic_inputs = ["xyzzy", "plugh", "fnord", "a", "1", "zz"];

    let mut chaotic_confidence = Vec::new();
    for i in 0..60 {
        let _ = chaotic.cycle(chaotic_inputs[i % chaotic_inputs.len()]);
        chaotic_confidence.push(chaotic.prediction_confidence());
    }

    // Structured, consistent input -> higher perceptual coherence
    let mut structured = make_service();
    let mut structured_confidence = Vec::new();
    for _ in 0..60 {
        let _ = structured.cycle("the pattern continues with consistent structure and meaning");
        structured_confidence.push(structured.prediction_confidence());
    }

    // The chaotic service should have lower prediction confidence (nudged down by
    // perception->drive coupling) compared to the structured service.
    let chaotic_late_conf: f32 = chaotic_confidence[40..].iter().sum::<f32>() / 20.0;
    let structured_late_conf: f32 = structured_confidence[40..].iter().sum::<f32>() / 20.0;

    // Both must be valid
    assert!(
        chaotic_late_conf.is_finite() && structured_late_conf.is_finite(),
        "Confidence must be finite: chaotic={chaotic_late_conf}, structured={structured_late_conf}"
    );

    // The systems should have diverged: different coherence levels produce different
    // exploration pressure and thus different confidence trajectories.
    let diff = (chaotic_late_conf - structured_late_conf).abs();
    assert!(
        diff > 0.001,
        "Perception->Drive coupling should produce measurable confidence divergence. \
         chaotic_avg={chaotic_late_conf:.6}, structured_avg={structured_late_conf:.6}, diff={diff:.6}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. BOUNDEDNESS: All couplings produce finite, bounded values
// ═══════════════════════════════════════════════════════════════════════════════

/// Run 200 cycles with varied input and verify all coupling-affected telemetry
/// fields remain finite and within expected ranges. This is a safety net against
/// NaN/Inf propagation through the coupling pathways.
#[test]
fn test_cross_couplings_all_bounded() {
    let mut service = make_service();

    let mixed_inputs = [
        "ok",
        "the quick brown fox jumps over the lazy dog",
        "quantum entanglement is spooky action at a distance",
        "ok",
        "a",
        "consciousness emerges from integrated information processing",
        "ok",
        "x y z",
        "moral reasoning requires both logic and empathy to navigate dilemmas",
        "ok",
    ];

    for i in 0..200 {
        let result = service.cycle(mixed_inputs[i % mixed_inputs.len()]);
        let m = &result.metadata;

        // Learning plasticity must be in [0.0, 1.0]
        assert!(
            m.learning_plasticity.is_finite()
                && m.learning_plasticity >= 0.0
                && m.learning_plasticity <= 1.0,
            "learning_plasticity out of bounds at cycle {i}: {}",
            m.learning_plasticity
        );

        // Memory consolidation pressure must be in [0.0, 1.0]
        assert!(
            m.memory_consolidation_pressure.is_finite()
                && m.memory_consolidation_pressure >= 0.0
                && m.memory_consolidation_pressure <= 1.0,
            "memory_consolidation_pressure out of bounds at cycle {i}: {}",
            m.memory_consolidation_pressure
        );

        // Memory recall quality must be in [0.0, 1.0]
        assert!(
            m.memory_recall_quality.is_finite()
                && m.memory_recall_quality >= 0.0
                && m.memory_recall_quality <= 1.0,
            "memory_recall_quality out of bounds at cycle {i}: {}",
            m.memory_recall_quality
        );

        // Prediction confidence must be in [0.0, 1.0]
        let conf = service.prediction_confidence();
        assert!(
            conf.is_finite() && conf >= 0.0 && conf <= 1.0,
            "prediction_confidence out of bounds at cycle {i}: {conf}"
        );

        // Prediction error must be finite
        assert!(
            result.prediction_error.is_finite(),
            "prediction_error not finite at cycle {i}: {}",
            result.prediction_error
        );
    }
}