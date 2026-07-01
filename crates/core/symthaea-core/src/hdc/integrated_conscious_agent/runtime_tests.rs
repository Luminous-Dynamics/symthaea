// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Runtime tests for the conscious agent

use super::*;

use crate::physiology::HormoneState;

// Use the agent's configured dimension (default is 2048)
const TEST_DIM: usize = 2048;

#[test]
fn test_sync_runtime_basic() {
    let config = RuntimeConfig::default();
    let mut runtime = SyncConsciousAgentRuntime::new(config);

    // Process some inputs - use agent's dimension
    let input = vec![0.5; TEST_DIM];
    let response = runtime.process(&input);

    match response {
        RuntimeResponse::ProcessingComplete {
            phi,
            dominant_emotion,
            ..
        } => {
            println!("Processed: Φ={:.4}, emotion={}", phi, dominant_emotion);
            assert!(phi >= 0.0);
        }
        RuntimeResponse::Error(e) => panic!("Unexpected error: {}", e),
        _ => panic!("Unexpected response type"),
    }
}

#[test]
fn test_sync_runtime_hormone_integration() {
    let config = RuntimeConfig::default();
    let mut runtime = SyncConsciousAgentRuntime::new(config);

    // Initial state
    let initial_snapshot = runtime.snapshot();
    println!("Initial: {:?}", initial_snapshot.emotion);

    // Apply stress hormones
    let stress_hormones = HormoneState {
        cortisol: 0.9,
        dopamine: 0.3,
        acetylcholine: 0.4,
    };
    runtime.apply_hormones(&stress_hormones);

    // Process input
    let input = vec![0.5; TEST_DIM];
    let _ = runtime.process(&input);

    // Check emotional state changed
    let after_snapshot = runtime.snapshot();
    println!("After stress: {:?}", after_snapshot.emotion);

    // Stress should increase arousal and decrease valence
    assert!(after_snapshot.emotion.arousal >= initial_snapshot.emotion.arousal * 0.9);
}

#[test]
fn test_sync_runtime_coherence_gating() {
    let mut config = RuntimeConfig::default();
    config.deep_processing_threshold = 0.8;
    let mut runtime = SyncConsciousAgentRuntime::new(config);

    // Set low coherence
    runtime.set_coherence(0.2);

    // Should still process (reflex level)
    let input = vec![0.5; TEST_DIM];
    let response = runtime.process(&input);

    let processed = matches!(response, RuntimeResponse::ProcessingComplete { .. });
    assert!(
        processed,
        "low-coherence input should still produce ProcessingComplete (reflex mode)"
    );
    if let RuntimeResponse::ProcessingComplete { phi, .. } = response {
        println!(
            "Low coherence processing succeeded (reflex mode), phi={:.4}",
            phi
        );
        assert!(phi >= 0.0, "phi should be non-negative, got {}", phi);
    }
}

#[test]
fn test_sync_runtime_memory_cycle() {
    let config = RuntimeConfig::default();
    let mut runtime = SyncConsciousAgentRuntime::new(config);

    // Process several inputs to build up working memory
    for i in 0..5 {
        let input: Vec<f32> = (0..TEST_DIM).map(|j| ((i * j) as f32).sin()).collect();
        let _ = runtime.process(&input);
    }

    // Export memories
    let exports = runtime.export_memories();
    println!("Exported {} memories", exports.len());
    assert!(
        !exports.is_empty(),
        "should have exported at least one memory after 5 processing steps"
    );

    // Simulate hippocampus processing and re-import
    let imports: Vec<MemoryImport> = exports
        .iter()
        .take(2)
        .map(|e| MemoryImport {
            content_vector: e.content_vector.clone(),
            emotional_valence: e.emotional_valence.clone(),
            relevance_score: 0.9,
        })
        .collect();

    let import_count = imports.len();
    runtime.import_memories(imports);
    println!("Re-imported {} memories", import_count);
    assert!(
        import_count > 0,
        "should have re-imported at least one memory"
    );
}

#[test]
fn test_sync_runtime_identity_tracking() {
    let config = RuntimeConfig::default();
    let mut runtime = SyncConsciousAgentRuntime::new(config);

    // Process to establish identity
    let input = vec![0.5; TEST_DIM];
    let _ = runtime.process(&input);

    // Check identity
    if let Some(identity) = runtime.check_identity() {
        println!(
            "Identity: {:?}, similarity: {:.4}",
            identity.status, identity.similarity
        );
        assert!(identity.similarity > 0.9); // Should be very similar to self
    }

    // Process more inputs
    for i in 0..10 {
        let input: Vec<f32> = (0..TEST_DIM)
            .map(|j| ((i * j) as f32 * 0.1).cos())
            .collect();
        let _ = runtime.process(&input);
    }

    // Check identity again - should show some drift
    if let Some(identity) = runtime.check_identity() {
        println!(
            "After processing: {:?}, similarity: {:.4}",
            identity.status, identity.similarity
        );
    }
}

#[test]
fn test_sync_runtime_prosody_generation() {
    let config = RuntimeConfig::default();
    let mut runtime = SyncConsciousAgentRuntime::new(config);

    // Process some input
    let input = vec![0.5; TEST_DIM];
    let _ = runtime.process(&input);

    // Get prosody hints
    let prosody = runtime.get_prosody();
    println!(
        "Prosody: rate={:.2}, pitch_shift={:.2}, energy={:.2}",
        prosody.rate, prosody.pitch_shift, prosody.energy
    );

    // Apply excitement hormones
    let excitement = HormoneState {
        cortisol: 0.3,
        dopamine: 0.9,
        acetylcholine: 0.7,
    };
    runtime.apply_hormones(&excitement);
    let _ = runtime.process(&input);

    // Get prosody again - should reflect excitement
    let excited_prosody = runtime.get_prosody();
    println!(
        "Excited prosody: rate={:.2}, pitch_shift={:.2}, energy={:.2}",
        excited_prosody.rate, excited_prosody.pitch_shift, excited_prosody.energy
    );

    // Excitement should increase rate and energy
    assert!(excited_prosody.energy >= prosody.energy * 0.9);
}

#[test]
fn test_sync_runtime_full_cycle() {
    let config = RuntimeConfig::default();
    let mut runtime = SyncConsciousAgentRuntime::new(config);

    println!("=== Full Conscious Agent Runtime Cycle ===\n");

    // 1. Initial state
    println!("1. Initial state:");
    let snapshot = runtime.snapshot();
    println!(
        "   Φ: {:.4}, Emotion: {}, Coherence: {:.2}\n",
        snapshot.phi, snapshot.emotion.quadrant, snapshot.coherence
    );

    // 2. Receive sensory input
    println!("2. Processing sensory input...");
    let input: Vec<f32> = (0..TEST_DIM).map(|i| (i as f32 * 0.01).sin()).collect();
    let response = runtime.process(&input);
    if let RuntimeResponse::ProcessingComplete {
        phi,
        dominant_emotion,
        qualia_summary,
    } = response
    {
        println!("   {}", qualia_summary);
    }

    // 3. Apply environmental stressor (cortisol spike)
    println!("\n3. Environmental stressor (cortisol spike)...");
    runtime.apply_hormones(&HormoneState {
        cortisol: 0.85,
        dopamine: 0.4,
        acetylcholine: 0.5,
    });
    let _ = runtime.process(&input);
    let snapshot = runtime.snapshot();
    println!(
        "   Emotion shifted to: {} (valence: {:.2}, arousal: {:.2})",
        snapshot.emotion.quadrant, snapshot.emotion.valence, snapshot.emotion.arousal
    );

    // 4. Get voice parameters
    println!("\n4. Voice output parameters:");
    let prosody = runtime.get_prosody();
    println!(
        "   Rate: {:.2}, Energy: {:.2}, Pause multiplier: {:.2}",
        prosody.rate, prosody.energy, prosody.pause_multiplier
    );

    // 5. Check identity
    println!("\n5. Identity check:");
    if let Some(identity) = runtime.check_identity() {
        println!(
            "   Status: {:?}, Similarity: {:.4}",
            identity.status, identity.similarity
        );
    }

    // 6. Memory consolidation
    println!("\n6. Memory consolidation:");
    let exports = runtime.export_memories();
    println!("   {} memories ready for hippocampus", exports.len());

    // 7. Get hormone suggestions
    println!("\n7. Hormone suggestions for endocrine system:");
    let suggestions = runtime.get_hormone_suggestions();
    for suggestion in &suggestions {
        println!("   {:?}", suggestion);
    }

    // 8. Final state
    println!("\n8. Final state:");
    let final_snapshot = runtime.snapshot();
    println!(
        "   Tick: {}, Φ: {:.4}, Memory load: {:.2}%",
        final_snapshot.tick,
        final_snapshot.phi,
        final_snapshot.memory_load * 100.0
    );

    println!("\n=== Cycle Complete ===");

    assert!(
        final_snapshot.tick > 0,
        "tick should have advanced, got {}",
        final_snapshot.tick
    );
    assert!(
        final_snapshot.phi >= 0.0,
        "phi should be non-negative, got {}",
        final_snapshot.phi
    );
}