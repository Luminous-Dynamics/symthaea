// ==================================================================================
// Module Ablation Test: Prove multi-module integration doesn't degrade baseline
// ==================================================================================
//
// Tests each consciousness module individually against baseline (all modules off)
// and verifies that multi-module synergy produces meaningful consciousness output.
//
// Protocol: 100 cycles, deterministic seed, repeating 4-word pattern, sync training
// Metrics: final_10_avg_error, coherence at cycle 100
// ==================================================================================

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const GENESIS_SEED: &str = "ablation_test_deterministic_seed_v061";
const PATTERN: &[&str] = &["alpha beta", "gamma delta", "epsilon zeta", "eta theta"];
const ABLATION_CYCLES: usize = 100;

/// Run N cycles of the repeating pattern and return (final_10_avg_error, coherence).
fn run_config(config: CognitiveLoopConfig, num_cycles: usize) -> (f32, f32) {
    let mut service =
        CognitiveLoopService::new(config).expect("Failed to create CognitiveLoopService");

    let mut errors = Vec::with_capacity(num_cycles);
    for i in 0..num_cycles {
        let result = service.cycle(PATTERN[i % PATTERN.len()]);
        errors.push(result.prediction_error);
    }

    let window = 10.min(errors.len());
    let final_avg = errors[errors.len() - window..].iter().sum::<f32>() / window as f32;
    let coherence = service.stats().temporal_coherence;

    (final_avg, coherence)
}

fn baseline_config() -> CognitiveLoopConfig {
    CognitiveLoopConfig {
        genesis_phrase: Some(GENESIS_SEED.to_string()),
        learning_threshold: 0.0,
        async_training: false,
        enable_virtual_body: false,
        enable_surprise_exploration: false,
        enable_prefrontal: false,
        enable_meta_cognition: false,
        enable_narrative_self: false,
        enable_predictive_self: false,
        enable_attention_schema: false,
        enable_gwt: false,
        enable_resonance: false,
        enable_quantum_coherence: false,
        enable_temporal_consciousness: false,
        enable_embodied_cognition: false,
        enable_narrative_gwt: false,
        ..Default::default()
    }
}

// ── Single-Module Ablation Tests ─────────────────────────────────

#[test]
fn test_ablation_baseline() {
    let (error, coherence) = run_config(baseline_config(), ABLATION_CYCLES);
    println!("Baseline: error={error:.4}, coherence={coherence:.4}");
    assert!(error.is_finite());
    assert!(coherence.is_finite());
}

#[test]
fn test_ablation_surprise_exploration() {
    let baseline = run_config(baseline_config(), ABLATION_CYCLES).0;
    let mut config = baseline_config();
    config.enable_surprise_exploration = true;
    let (error, _) = run_config(config, ABLATION_CYCLES);
    println!("Surprise: error={error:.4}, baseline={baseline:.4}");
    assert!(
        error <= baseline * 1.05,
        "Surprise module should not degrade baseline by >5%: {error:.4} vs {baseline:.4}"
    );
}

#[test]
fn test_ablation_prefrontal() {
    let baseline = run_config(baseline_config(), ABLATION_CYCLES).0;
    let mut config = baseline_config();
    config.enable_prefrontal = true;
    let (error, _) = run_config(config, ABLATION_CYCLES);
    println!("Prefrontal: error={error:.4}, baseline={baseline:.4}");
    assert!(
        error <= baseline * 1.05,
        "Prefrontal module should not degrade baseline by >5%: {error:.4} vs {baseline:.4}"
    );
}

#[test]
fn test_ablation_meta_cognition() {
    let baseline = run_config(baseline_config(), ABLATION_CYCLES).0;
    let mut config = baseline_config();
    config.enable_meta_cognition = true;
    let (error, _) = run_config(config, ABLATION_CYCLES);
    println!("Meta-cognition: error={error:.4}, baseline={baseline:.4}");
    assert!(
        error <= baseline * 1.05,
        "Meta-cognition module should not degrade baseline by >5%: {error:.4} vs {baseline:.4}"
    );
}

#[test]
fn test_ablation_virtual_body() {
    let baseline = run_config(baseline_config(), ABLATION_CYCLES).0;
    let mut config = baseline_config();
    config.enable_virtual_body = true;
    let (error, _) = run_config(config, ABLATION_CYCLES);
    println!("Virtual body: error={error:.4}, baseline={baseline:.4}");
    assert!(
        error <= baseline * 1.05,
        "Virtual body module should not degrade baseline by >5%: {error:.4} vs {baseline:.4}"
    );
}

#[test]
fn test_ablation_predictive_self() {
    let baseline = run_config(baseline_config(), ABLATION_CYCLES).0;
    let mut config = baseline_config();
    config.enable_predictive_self = true;
    config.enable_narrative_self = true; // Dependency
    let (error, _) = run_config(config, ABLATION_CYCLES);
    println!("Predictive self: error={error:.4}, baseline={baseline:.4}");
    assert!(
        error <= baseline * 1.05,
        "Predictive self module should not degrade baseline by >5%: {error:.4} vs {baseline:.4}"
    );
}

#[test]
fn test_ablation_attention_schema() {
    let baseline = run_config(baseline_config(), ABLATION_CYCLES).0;
    let mut config = baseline_config();
    config.enable_attention_schema = true;
    let (error, _) = run_config(config, ABLATION_CYCLES);
    println!("Attention schema: error={error:.4}, baseline={baseline:.4}");
    assert!(
        error <= baseline * 1.05,
        "Attention schema module should not degrade baseline by >5%: {error:.4} vs {baseline:.4}"
    );
}

#[test]
fn test_ablation_gwt() {
    let baseline = run_config(baseline_config(), ABLATION_CYCLES).0;
    let mut config = baseline_config();
    config.enable_gwt = true;
    let (error, _) = run_config(config, ABLATION_CYCLES);
    println!("GWT: error={error:.4}, baseline={baseline:.4}");
    assert!(
        error <= baseline * 1.05,
        "GWT module should not degrade baseline by >5%: {error:.4} vs {baseline:.4}"
    );
}

#[test]
fn test_ablation_temporal_consciousness() {
    let baseline = run_config(baseline_config(), ABLATION_CYCLES).0;
    let mut config = baseline_config();
    config.enable_temporal_consciousness = true;
    config.enable_narrative_self = true; // Dependency
    config.enable_predictive_self = true; // Dependency
    let (error, _) = run_config(config, ABLATION_CYCLES);
    println!("Temporal consciousness: error={error:.4}, baseline={baseline:.4}");
    assert!(
        error <= baseline * 1.05,
        "Temporal consciousness module should not degrade baseline by >5%: {error:.4} vs {baseline:.4}"
    );
}

#[test]
fn test_ablation_embodied_cognition() {
    let baseline = run_config(baseline_config(), ABLATION_CYCLES).0;
    let mut config = baseline_config();
    config.enable_embodied_cognition = true;
    config.enable_virtual_body = true; // Dependency
    let (error, _) = run_config(config, ABLATION_CYCLES);
    println!("Embodied cognition: error={error:.4}, baseline={baseline:.4}");
    assert!(
        error <= baseline * 1.05,
        "Embodied cognition module should not degrade baseline by >5%: {error:.4} vs {baseline:.4}"
    );
}

#[test]
fn test_ablation_narrative_gwt() {
    let baseline = run_config(baseline_config(), ABLATION_CYCLES).0;
    let mut config = baseline_config();
    config.enable_narrative_gwt = true;
    let (error, _) = run_config(config, ABLATION_CYCLES);
    println!("Narrative-GWT: error={error:.4}, baseline={baseline:.4}");
    assert!(
        error <= baseline * 1.05,
        "Narrative-GWT module should not degrade baseline by >5%: {error:.4} vs {baseline:.4}"
    );
}

// ── Multi-Module Synergy Test ────────────────────────────────────

#[test]
fn test_all_modules_synergy() {
    let baseline = run_config(baseline_config(), ABLATION_CYCLES);

    let all_config = CognitiveLoopConfig {
        genesis_phrase: Some(GENESIS_SEED.to_string()),
        learning_threshold: 0.0,
        async_training: false,
        enable_virtual_body: true,
        enable_surprise_exploration: true,
        enable_prefrontal: true,
        enable_meta_cognition: true,
        enable_narrative_self: true,
        enable_predictive_self: true,
        enable_attention_schema: true,
        enable_gwt: true,
        enable_resonance: true,
        enable_quantum_coherence: true,
        enable_temporal_consciousness: true,
        enable_embodied_cognition: true,
        enable_narrative_gwt: true,
        ..Default::default()
    };

    let mut service =
        CognitiveLoopService::new(all_config).expect("Failed to create all-modules service");

    let mut errors = Vec::with_capacity(200);
    let mut consciousness_levels = Vec::new();
    let mut metadata_populated = 0u32;

    for i in 0..200 {
        let result = service.cycle(PATTERN[i % PATTERN.len()]);
        errors.push(result.prediction_error);

        if result.metadata.consciousness_level > 0.0 {
            consciousness_levels.push(result.metadata.consciousness_level);
        }

        // Count non-default metadata fields
        if result.metadata.prefrontal_veto {
            metadata_populated |= 1;
        }
        if result.metadata.gwt_broadcast {
            metadata_populated |= 2;
        }
        if result.metadata.meta_cognitive_accuracy > 0.0 {
            metadata_populated |= 4;
        }
        if result.metadata.narrative_self_phi > 0.0 {
            metadata_populated |= 8;
        }
        if (result.metadata.body_phi_modulation - 1.0).abs() > 0.001 {
            metadata_populated |= 16;
        }
        if result.metadata.temporal_coherence_score > 0.0 {
            metadata_populated |= 32;
        }
        if result.metadata.attention_schema_focus > 0.0 {
            metadata_populated |= 64;
        }
    }

    let final_10_avg = errors[190..].iter().sum::<f32>() / 10.0;

    println!(
        "All modules: final_error={final_10_avg:.4}, baseline_error={:.4}",
        baseline.0
    );
    println!("Consciousness levels computed: {}", consciousness_levels.len());
    println!("Metadata populated bits: {metadata_populated:#010b}");

    // Assert: consciousness_level > 0 (MCE produces meaningful output)
    assert!(
        !consciousness_levels.is_empty(),
        "MCE should have computed consciousness_level at least once in 200 cycles"
    );
    assert!(
        consciousness_levels.iter().any(|&c| c > 0.0),
        "At least one consciousness_level should be > 0"
    );

    // Assert: all-modules doesn't degrade badly (within 20%)
    // 11 modules add feedback loops that can temporarily increase error before convergence
    assert!(
        final_10_avg <= baseline.0 * 1.20,
        "All modules should not degrade baseline by >20%: {final_10_avg:.4} vs {:.4}",
        baseline.0
    );

    // Assert: at least 3 metadata fields populated with non-default values
    let populated_count = metadata_populated.count_ones();
    assert!(
        populated_count >= 3,
        "At least 3 metadata fields should be populated, got {populated_count} (bits: {metadata_populated:#010b})"
    );
}
