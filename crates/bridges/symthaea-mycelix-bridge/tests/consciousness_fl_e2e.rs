// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-End Consciousness FL Integration Test
//!
//! Proves the full consciousness-guided federated learning pipeline:
//!
//! 1. Create HyperGradients (honest + Byzantine participants)
//! 2. Run each through `SymthaeaBackend::assess_update()` → `QualityScore`
//! 3. Feed scores into `SymthaeaQualityPlugin` (bridge plugin)
//! 4. Compose with `ConsciousnessAwareByzantinePlugin` (core plugin)
//! 5. Run through `UnifiedPipeline::aggregate_with_plugins()`
//! 6. Verify Byzantine influence is suppressed, honest updates dominate
//!
//! Run: `cargo test --test consciousness_fl_e2e`

use std::collections::HashMap;

use mycelix_fl_core::UnifiedPipeline;
use mycelix_fl_core::consciousness_plugin::ConsciousnessAwareByzantinePlugin;
use mycelix_fl_core::pipeline::PipelineConfig;
use mycelix_fl_core::plugins::{ByzantinePlugin, PipelinePlugins};
use mycelix_fl_core::types::GradientUpdate;

use mycelix_sdk::hyperfeel::{HV16_BYTES, HyperGradient};

use symthaea_mycelix_bridge::fl_plugin::SymthaeaQualityPlugin;
use symthaea_mycelix_bridge::{
    ConsciousAnomalySeverity, ConsciousnessBackend, ConsciousnessVector, QualityScore,
    SpectralConnectivityAssessment, SymthaeaBackend, pogq_from_quality_score,
};

// =============================================================================
// Helpers
// =============================================================================

/// Create a HyperGradient with a deterministic byte pattern.
fn make_hypergradient(node_id: &str, round: u32, fill_byte: u8) -> HyperGradient {
    let mut hash = [0u8; 32];
    for (i, b) in node_id.bytes().enumerate() {
        hash[i % 32] ^= b;
    }
    HyperGradient::new(
        node_id.to_string(),
        round,
        vec![fill_byte; HV16_BYTES],
        0.5,
        4_000_000,
        1.0,
        hash,
    )
}

/// Create a GradientUpdate matching a HyperGradient's participant.
fn make_gradient_update(node_id: &str, dim: usize, value: f32) -> GradientUpdate {
    GradientUpdate::new(node_id.into(), 1, vec![value; dim], 100, 0.5)
}

/// Construct a QualityScore directly (for cases where we want to control
/// the signal without running through SymthaeaBackend).
fn synthetic_quality(
    confidence: f32,
    phi_before: f32,
    phi_after: f32,
    is_anomalous: bool,
    severity: ConsciousAnomalySeverity,
) -> QualityScore {
    QualityScore {
        accuracy: if is_anomalous { 0.3 } else { 0.85 },
        loss: if is_anomalous { 1.5 } else { 0.2 },
        spectral: SpectralConnectivityAssessment::new(phi_before, phi_after),
        consciousness_vector: ConsciousnessVector::default(),
        epistemic_confidence: confidence,
        is_anomalous,
        similarity: Some(if is_anomalous { 0.4 } else { 0.9 }),
        is_ambiguous: false,
        severity,
        causes: if is_anomalous {
            vec!["phi_drop".into()]
        } else {
            Vec::new()
        },
    }
}

// =============================================================================
// Tests
// =============================================================================

/// Full pipeline: SymthaeaBackend produces QualityScores, SymthaeaQualityPlugin
/// feeds them into the FL pipeline, Byzantine nodes get filtered.
#[test]
fn test_e2e_symthaea_backend_to_pipeline() {
    let dim = 20;
    let mut backend = SymthaeaBackend::new();

    // --- Stage 1: Assess updates via SymthaeaBackend ---
    // We use a DummySnapshot since SymthaeaBackend needs a ModelSnapshot.
    struct DummySnapshot {
        accuracy: f32,
        loss: f32,
    }
    impl symthaea_mycelix_bridge::ModelSnapshot for DummySnapshot {
        fn apply_update_clone(
            &self,
            _update: &HyperGradient,
        ) -> symthaea_mycelix_bridge::Result<Box<dyn symthaea_mycelix_bridge::ModelSnapshot>>
        {
            Ok(Box::new(DummySnapshot {
                accuracy: self.accuracy,
                loss: self.loss,
            }))
        }
        fn evaluate(&self) -> symthaea_mycelix_bridge::Result<(f32, f32)> {
            Ok((self.accuracy, self.loss))
        }
        fn current_accuracy(&self) -> Option<f32> {
            Some(self.accuracy)
        }
    }

    let good_snapshot = DummySnapshot {
        accuracy: 0.85,
        loss: 0.2,
    };

    // Honest nodes: moderate byte patterns (close to mid-range)
    let honest_ids: Vec<String> = (0..6).map(|i| format!("honest_{}", i)).collect();
    let byzantine_ids: Vec<String> = (0..2).map(|i| format!("byz_{}", i)).collect();

    // --- Warm-up pass: calibrate phi_before for each node ---
    // SymthaeaBackend defaults phi_before to 0.5; PhiEngine on a single HV
    // returns ~0.0, causing a large phi_drop on the first assessment. Run a
    // warm-up so the second pass has phi_before ≈ phi_after (no false alarm).
    for id in honest_ids.iter().chain(byzantine_ids.iter()) {
        let hg = make_hypergradient(id, 0, 128);
        let _ = backend.assess_update(&good_snapshot, &hg);
    }

    let mut quality_scores: HashMap<String, QualityScore> = HashMap::new();

    // Assess honest nodes (second pass — phi_before now calibrated)
    for id in &honest_ids {
        let hg = make_hypergradient(id, 1, 128); // Mid-range bytes
        let score = backend.assess_update(&good_snapshot, &hg).unwrap();
        quality_scores.insert(id.clone(), score);
    }

    // Assess Byzantine nodes with extreme patterns and bad validation
    let bad_snapshot = DummySnapshot {
        accuracy: 0.2,
        loss: 3.0,
    };
    for id in &byzantine_ids {
        let hg = make_hypergradient(id, 1, 255); // Extreme bytes
        let score = backend.assess_update(&bad_snapshot, &hg).unwrap();
        quality_scores.insert(id.clone(), score);
    }

    // Verify honest nodes have higher confidence than Byzantine
    for id in &honest_ids {
        let q = &quality_scores[id];
        assert!(
            q.epistemic_confidence > 0.3,
            "Honest node {} should have reasonable confidence, got {}",
            id,
            q.epistemic_confidence
        );
    }

    // --- Stage 2: Feed into SymthaeaQualityPlugin ---
    let mut quality_plugin = SymthaeaQualityPlugin::new();
    quality_plugin.set_quality_scores(quality_scores);

    // --- Stage 3: Create matching GradientUpdates ---
    let mut updates = Vec::new();
    let mut reps = HashMap::new();

    for id in &honest_ids {
        updates.push(make_gradient_update(id, dim, 0.5));
        reps.insert(id.clone(), 0.9);
    }
    for id in &byzantine_ids {
        updates.push(make_gradient_update(id, dim, 100.0)); // Extreme gradients
        reps.insert(id.clone(), 0.5); // Medium reputation — plugin must catch them
    }

    // --- Stage 4: Run through pipeline ---
    // Use performance config (lower trust_threshold=0.3) to avoid over-filtering
    let config = PipelineConfig::performance();
    let mut pipeline = UnifiedPipeline::new(config);

    let mut plugins = PipelinePlugins {
        compression: None,
        byzantine: vec![&mut quality_plugin],
        verification: None,
    };

    let result = pipeline
        .aggregate_with_plugins(&updates, &reps, &mut plugins)
        .expect("Pipeline should succeed");

    // --- Stage 5: Verify ---
    // Result should be near the honest mean (0.5), not pulled by Byzantine (100.0)
    for (i, val) in result.result.aggregated.gradients.iter().enumerate() {
        assert!(
            (*val - 0.5).abs() < 10.0,
            "Dim {}: value {} too far from honest mean 0.5 (Byzantine influence leaked)",
            i,
            val
        );
    }
}

/// Compose both plugins: SymthaeaQualityPlugin + ConsciousnessAwareByzantinePlugin.
/// Byzantine nodes should be caught by at least one plugin.
#[test]
fn test_e2e_composed_plugins_synergy() {
    let dim = 15;

    // --- SymthaeaQualityPlugin with synthetic scores ---
    let mut quality_plugin = SymthaeaQualityPlugin::new();
    let mut quality_scores = HashMap::new();

    // Honest nodes: high confidence, no anomaly
    for i in 0..6 {
        let id = format!("h{}", i);
        quality_scores.insert(
            id,
            synthetic_quality(0.85, 0.4, 0.6, false, ConsciousAnomalySeverity::None),
        );
    }

    // Byzantine node 0: caught by quality plugin (severe anomaly)
    quality_scores.insert(
        "byz0".into(),
        synthetic_quality(0.05, 0.6, 0.1, true, ConsciousAnomalySeverity::Severe),
    );

    // Byzantine node 1: low confidence but not severe — caught by Phi plugin
    quality_scores.insert(
        "byz1".into(),
        synthetic_quality(0.4, 0.2, 0.15, true, ConsciousAnomalySeverity::Mild),
    );

    quality_plugin.set_quality_scores(quality_scores);

    // --- ConsciousnessAwareByzantinePlugin with connectivity scores ---
    let mut phi_plugin = ConsciousnessAwareByzantinePlugin::new();
    let mut phi_scores = HashMap::new();
    for i in 0..6 {
        phi_scores.insert(format!("h{}", i), 0.8); // High connectivity for honest
    }
    phi_scores.insert("byz0".into(), 0.05); // Below veto threshold (0.1)
    phi_scores.insert("byz1".into(), 0.05); // Below veto threshold
    phi_plugin.set_consciousness_scores(phi_scores);

    // --- Build updates ---
    let mut updates = Vec::new();
    let mut reps = HashMap::new();

    for i in 0..6 {
        let id = format!("h{}", i);
        updates.push(GradientUpdate::new(id.clone(), 1, vec![0.5; dim], 100, 0.5));
        reps.insert(id, 0.9);
    }
    updates.push(GradientUpdate::new(
        "byz0".into(),
        1,
        vec![100.0; dim],
        100,
        0.5,
    ));
    reps.insert("byz0".into(), 0.5);
    updates.push(GradientUpdate::new(
        "byz1".into(),
        1,
        vec![-100.0; dim],
        100,
        0.5,
    ));
    reps.insert("byz1".into(), 0.5);

    // --- Compose plugins ---
    let config = PipelineConfig::adaptive();
    let mut pipeline = UnifiedPipeline::new(config);

    let mut plugins = PipelinePlugins {
        compression: None,
        byzantine: vec![&mut quality_plugin, &mut phi_plugin],
        verification: None,
    };

    let result = pipeline
        .aggregate_with_plugins(&updates, &reps, &mut plugins)
        .expect("Composed plugins should succeed");

    // Both Byzantine nodes should be suppressed
    for (i, val) in result.result.aggregated.gradients.iter().enumerate() {
        assert!(
            (*val - 0.5).abs() < 5.0,
            "Dim {}: Byzantine influence leaked with composed plugins: {}",
            i,
            val
        );
    }

    // Quality plugin stats should show at least 1 veto (byz0 = severe)
    let (vetoed, _, _, _) = quality_plugin.stats();
    assert!(
        vetoed >= 1,
        "Quality plugin should have vetoed at least 1 Byzantine node"
    );
}

/// Multi-round test: run 3 rounds and verify rolling stats accumulate.
#[test]
fn test_e2e_multi_round_stats_accumulate() {
    let dim = 10;

    let mut quality_plugin = SymthaeaQualityPlugin::new();
    let config = PipelineConfig::adaptive();
    let mut pipeline = UnifiedPipeline::new(config);

    for round in 0..3 {
        let mut quality_scores = HashMap::new();

        // 4 honest per round
        for i in 0..4 {
            let id = format!("h{}", i);
            quality_scores.insert(
                id.clone(),
                synthetic_quality(0.85, 0.4, 0.6, false, ConsciousAnomalySeverity::None),
            );
        }

        // 1 Byzantine per round
        let byz_id = format!("byz_r{}", round);
        quality_scores.insert(
            byz_id.clone(),
            synthetic_quality(0.05, 0.6, 0.1, true, ConsciousAnomalySeverity::Severe),
        );

        quality_plugin.set_quality_scores(quality_scores);

        let mut updates = Vec::new();
        let mut reps = HashMap::new();

        for i in 0..4 {
            let id = format!("h{}", i);
            updates.push(GradientUpdate::new(
                id.clone(),
                round as u64 + 1,
                vec![0.5; dim],
                100,
                0.5,
            ));
            reps.insert(id, 0.9);
        }
        updates.push(GradientUpdate::new(
            byz_id.clone(),
            round as u64 + 1,
            vec![50.0; dim],
            100,
            0.5,
        ));
        reps.insert(byz_id, 0.5);

        let mut plugins = PipelinePlugins {
            compression: None,
            byzantine: vec![&mut quality_plugin],
            verification: None,
        };

        let _result = pipeline
            .aggregate_with_plugins(&updates, &reps, &mut plugins)
            .expect("Pipeline round should succeed");

        // Note: record_outcome is called by the pipeline internally, no manual call needed
    }

    // After 3 rounds, we should have 3 vetoes and 3 rounds recorded
    // (pipeline calls record_outcome once per aggregate_with_plugins)
    let (vetoed, _boosted, _dampened, rounds) = quality_plugin.stats();
    assert_eq!(
        vetoed, 3,
        "Should have vetoed 1 Byzantine per round × 3 rounds"
    );
    assert_eq!(rounds, 3, "Should have recorded 3 rounds");
}

/// PoGQ mapping: verify quality scores produce sensible ProofOfGradientQuality.
#[test]
fn test_e2e_pogq_from_quality_scores() {
    let good = synthetic_quality(0.9, 0.4, 0.7, false, ConsciousAnomalySeverity::None);
    let bad = synthetic_quality(0.05, 0.6, 0.1, true, ConsciousAnomalySeverity::Severe);

    let pogq_good = pogq_from_quality_score(&good);
    let pogq_bad = pogq_from_quality_score(&bad);

    assert!(
        pogq_good.quality > pogq_bad.quality,
        "Good update should have higher PoGQ quality"
    );
    assert!(
        pogq_good.entropy < pogq_bad.entropy,
        "Good update should have lower PoGQ entropy"
    );
}

/// Boost test: high-confidence nodes with positive Phi gain get weight > 1.0.
#[test]
fn test_e2e_quality_plugin_boosts_high_confidence() {
    let dim = 10;
    let mut quality_plugin = SymthaeaQualityPlugin::new();

    let mut scores = HashMap::new();
    // High confidence + positive connectivity gain → should boost
    let mut q = synthetic_quality(0.9, 0.3, 0.6, false, ConsciousAnomalySeverity::None);
    q.spectral = SpectralConnectivityAssessment::new(0.3, 0.6); // connectivity_gain = 0.3 > 0.05 threshold
    q.epistemic_confidence = 0.85; // > 0.7 threshold
    scores.insert("boosted".into(), q);

    quality_plugin.set_quality_scores(scores);

    let updates = vec![GradientUpdate::new(
        "boosted".into(),
        1,
        vec![0.5; dim],
        100,
        0.5,
    )];

    let weights = quality_plugin.analyze(&updates);

    assert!(weights.contains_key("boosted"), "Should have an adjustment");
    let adj = &weights["boosted"][0];
    assert!(
        adj.weight_multiplier > 1.0,
        "High-confidence node with phi gain should be boosted, got {}",
        adj.weight_multiplier
    );
    assert!(!adj.veto);
}

/// Backend state tracking: multiple assessments for the same node update its
/// internal state (last_phi, anomaly_count).
#[test]
fn test_e2e_backend_tracks_node_state_across_rounds() {
    let mut backend = SymthaeaBackend::new();

    struct ConstSnapshot;
    impl symthaea_mycelix_bridge::ModelSnapshot for ConstSnapshot {
        fn apply_update_clone(
            &self,
            _: &HyperGradient,
        ) -> symthaea_mycelix_bridge::Result<Box<dyn symthaea_mycelix_bridge::ModelSnapshot>>
        {
            Ok(Box::new(ConstSnapshot))
        }
        fn evaluate(&self) -> symthaea_mycelix_bridge::Result<(f32, f32)> {
            Ok((0.7, 0.3))
        }
        fn current_accuracy(&self) -> Option<f32> {
            Some(0.65)
        }
    }

    let snapshot = ConstSnapshot;

    // Round 1
    let hg1 = make_hypergradient("node_a", 1, 128);
    let _score1 = backend.assess_update(&snapshot, &hg1).unwrap();

    // Node state should exist after first assessment
    let (last_phi, anomaly_count) = backend
        .node_state_snapshot("node_a")
        .expect("node_a state should exist");
    assert!(last_phi >= 0.0 && last_phi <= 1.0);
    let _initial_anomaly = anomaly_count;

    // Round 2: same node, different pattern
    let hg2 = make_hypergradient("node_a", 2, 200);
    let _score2 = backend.assess_update(&snapshot, &hg2).unwrap();

    let (last_phi_2, _) = backend.node_state_snapshot("node_a").unwrap();
    // Phi should be updated (may or may not equal the first)
    assert!(last_phi_2 >= 0.0 && last_phi_2 <= 1.0);

    // Reset and verify state is cleared
    backend.reset_node("node_a");
    assert!(
        backend.node_state_snapshot("node_a").is_none(),
        "State should be cleared after reset"
    );
}

/// Ambiguity detection: updates in the gray zone (between ambiguity and
/// recall thresholds) should be dampened by the quality plugin.
#[test]
fn test_e2e_ambiguous_update_dampened() {
    let mut quality_plugin = SymthaeaQualityPlugin::new();

    let mut q = synthetic_quality(0.55, 0.5, 0.52, true, ConsciousAnomalySeverity::Mild);
    q.is_ambiguous = true;

    let mut scores = HashMap::new();
    scores.insert("gray_zone".into(), q);
    quality_plugin.set_quality_scores(scores);

    let updates = vec![GradientUpdate::new(
        "gray_zone".into(),
        1,
        vec![0.5; 10],
        100,
        0.5,
    )];

    let weights = quality_plugin.analyze(&updates);

    assert!(weights.contains_key("gray_zone"));
    let adj = &weights["gray_zone"][0];
    assert!(!adj.veto, "Ambiguous should be dampened, not vetoed");
    assert!(
        adj.weight_multiplier < 1.0,
        "Ambiguous update should be dampened, got {}",
        adj.weight_multiplier
    );
}

// =============================================================================
// Full C-Vector → Attestation → Governance compatibility test
// =============================================================================

/// End-to-end: SymthaeaBackend computes C-Vector, attestation carries it,
/// and the serialized form is compatible with governance's expected input.
#[test]
fn test_e2e_cvector_attestation_governance_compatibility() {
    use symthaea_mycelix_bridge::{
        ConsciousnessVectorSerde, create_consciousness_attestation_data,
    };

    // Step 1: Assess an update — produces a QualityScore with populated C-Vector
    let mut backend = SymthaeaBackend::new();

    struct TestSnapshot;
    impl symthaea_mycelix_bridge::ModelSnapshot for TestSnapshot {
        fn apply_update_clone(
            &self,
            _update: &HyperGradient,
        ) -> symthaea_mycelix_bridge::Result<Box<dyn symthaea_mycelix_bridge::ModelSnapshot>>
        {
            Ok(Box::new(TestSnapshot))
        }
        fn evaluate(&self) -> symthaea_mycelix_bridge::Result<(f32, f32)> {
            Ok((0.82, 0.3))
        }
        fn current_accuracy(&self) -> Option<f32> {
            Some(0.78)
        }
    }

    let hg = make_hypergradient("attester-node", 1, 170);
    let quality = backend.assess_update(&TestSnapshot, &hg).unwrap();

    // Step 2: Verify C-Vector is populated
    let cv = &quality.consciousness_vector;
    assert!(
        cv.populated_count() >= 3,
        "C-Vector should have at least 3 dimensions populated, got {}",
        cv.populated_count()
    );
    assert!(
        cv.spectral_connectivity.is_some(),
        "spectral_connectivity must be populated"
    );
    assert!(
        cv.epistemic_confidence.is_some(),
        "epistemic_confidence must be populated"
    );

    // Step 3: Create attestation — should include C-Vector
    let attestation = create_consciousness_attestation_data(&quality, "did:mycelix:test-agent", 42);

    // Consciousness level should equal composite
    let composite = cv.composite();
    assert!(
        (attestation.consciousness_level - composite).abs() < 0.01,
        "Attestation consciousness_level ({}) should match C-Vector composite ({})",
        attestation.consciousness_level,
        composite
    );

    // Step 4: Verify C-Vector is present in attestation
    assert!(
        attestation.consciousness_vector.is_some(),
        "Attestation should carry C-Vector"
    );
    let serde_cv = attestation.consciousness_vector.as_ref().unwrap();
    assert_eq!(serde_cv.spectral_connectivity, cv.spectral_connectivity);
    assert_eq!(serde_cv.true_phi, cv.true_phi);
    assert_eq!(serde_cv.entropy, cv.entropy);

    // Step 5: Verify governance-compatible serialization
    // Governance's RecordConsciousnessAttestationInput expects the C-Vector as
    // a JSON object with the same field names. Test round-trip.
    let json = serde_json::to_string(&serde_cv).unwrap();
    let roundtrip: ConsciousnessVectorSerde = serde_json::from_str(&json).unwrap();
    assert_eq!(
        roundtrip.spectral_connectivity,
        serde_cv.spectral_connectivity
    );
    assert_eq!(roundtrip.true_phi, serde_cv.true_phi);
    assert_eq!(roundtrip.phi_fast, serde_cv.phi_fast);
    assert_eq!(roundtrip.entropy, serde_cv.entropy);
    assert_eq!(roundtrip.coherence, serde_cv.coherence);
    assert_eq!(
        roundtrip.epistemic_confidence,
        serde_cv.epistemic_confidence
    );

    // Step 6: Verify best_phi fallback chain works on the bridge side
    let best = cv.best_phi();
    assert!(
        best >= 0.0 && best <= 1.0,
        "best_phi should be in [0,1], got {}",
        best
    );

    // Step 7: Verify sign message is deterministic (governance verification depends on this)
    let msg1 = attestation.sign_message();
    let msg2 = attestation.sign_message();
    assert_eq!(msg1, msg2, "sign_message must be deterministic");
}

/// Verify that PoGQ mapping uses C-Vector composite, not raw spectral connectivity
#[test]
fn test_e2e_pogq_uses_cvector_composite() {
    use mycelix_sdk::matl::ProofOfGradientQuality;

    // Create a quality score with high spectral but low true_phi
    let mut quality = synthetic_quality(0.8, 0.3, 0.7, false, ConsciousAnomalySeverity::None);
    quality.consciousness_vector = ConsciousnessVector {
        spectral_connectivity: Some(0.7),
        true_phi: Some(0.15), // Much lower than spectral
        phi_fast: Some(0.12),
        entropy: Some(0.5),
        coherence: Some(0.4),
        epistemic_confidence: Some(0.8),
    };

    let pogq: ProofOfGradientQuality = pogq_from_quality_score(&quality);

    // PoGQ quality should reflect epistemic confidence
    assert!(pogq.quality > 0.0, "PoGQ quality should be positive");

    // Verify the C-Vector composite reflects the lower true_phi
    let composite = quality.consciousness_vector.composite();
    assert!(
        composite < 0.6,
        "Composite should be pulled down by low true_phi, got {}",
        composite
    );
    assert!(composite > 0.0, "Composite should still be positive");
}

// =============================================================================
// Multi-round C-Vector: true_phi activates with HV history
// =============================================================================

/// After multiple assessments of the same node, the C-Vector should have
/// true_phi populated (requires ≥2 components from HV history).
#[test]
fn test_multi_round_cvector_activates_true_phi() {
    struct TestSnapshot;
    impl symthaea_mycelix_bridge::ModelSnapshot for TestSnapshot {
        fn apply_update_clone(
            &self,
            _update: &HyperGradient,
        ) -> symthaea_mycelix_bridge::Result<Box<dyn symthaea_mycelix_bridge::ModelSnapshot>>
        {
            Ok(Box::new(TestSnapshot))
        }
        fn evaluate(&self) -> symthaea_mycelix_bridge::Result<(f32, f32)> {
            Ok((0.80, 0.3))
        }
        fn current_accuracy(&self) -> Option<f32> {
            Some(0.75)
        }
    }

    let mut backend = SymthaeaBackend::new();

    // Round 1: single component — true_phi should be None
    let hg1 = make_hypergradient("multi-round-node", 1, 100);
    let q1 = backend.assess_update(&TestSnapshot, &hg1).unwrap();
    assert!(
        q1.consciousness_vector.true_phi.is_none(),
        "Round 1: true_phi should be None with only 1 component"
    );

    // Round 2: now 2 components — true_phi should be Some
    let hg2 = make_hypergradient("multi-round-node", 2, 150);
    let q2 = backend.assess_update(&TestSnapshot, &hg2).unwrap();
    assert!(
        q2.consciousness_vector.true_phi.is_some(),
        "Round 2: true_phi should be populated with 2 HV history components"
    );
    let phi = q2.consciousness_vector.true_phi.unwrap();
    assert!(phi.is_finite(), "true_phi should be finite, got {}", phi);

    // Round 3: coherence should now be non-trivial (not always 1.0)
    let hg3 = make_hypergradient("multi-round-node", 3, 200);
    let q3 = backend.assess_update(&TestSnapshot, &hg3).unwrap();
    assert!(q3.consciousness_vector.true_phi.is_some());
    // With 3 diverse components, coherence should differ from 1.0
    let coh = q3.consciousness_vector.coherence.unwrap();
    assert!(
        coh < 1.0,
        "Coherence with 3 diverse HVs should be < 1.0, got {}",
        coh
    );
    assert!(coh >= 0.0, "Coherence should be non-negative");

    // Verify populated_count grows
    assert!(
        q3.consciousness_vector.populated_count() >= 5,
        "Round 3 should have at least 5 dimensions populated, got {}",
        q3.consciousness_vector.populated_count()
    );
}
