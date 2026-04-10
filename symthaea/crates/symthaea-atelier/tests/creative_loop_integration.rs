// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end creative loop integration test.
//!
//! Verifies the full cycle: state → generate → evaluate → feedback → evolve → generate again.
//! The aesthetic score should improve (or at least not degrade) over iterations.

use symthaea_aesthetic::{AestheticConfig, AestheticTracker};
use symthaea_atelier::{AtelierConfig, AtelierStyle};
use symthaea_canvas::CognitiveSnapshot;

/// Test the full creative feedback loop over multiple cycles.
#[test]
fn creative_loop_score_trajectory() {
    let config = AtelierConfig {
        style: AtelierStyle::Composite,
        iteration_budget: 3,
        ..AtelierConfig::default()
    };

    let mut tracker = AestheticTracker::new(AestheticConfig::default());

    let mut harmony_activations = [0.5f32; 8];
    let mut valence = 0.3f32;
    let mut arousal = 0.4f32;
    let mut dopamine = 0.5f32;
    let mut consciousness = 0.6f32;

    let mut scores: Vec<f32> = Vec::new();

    for cycle in 0..15 {
        let snapshot = CognitiveSnapshot {
            consciousness_level: consciousness as f64,
            valence,
            arousal,
            dopamine,
            serotonin: 0.5,
            noradrenaline: arousal * 0.5,
            harmony_activations,
            betti_0: 3,
            betti_1: 1,
            persistence_components: vec![[0.0, 0.6]],
            persistence_cycles: vec![[0.1, 0.5]],
            thought_vector: vec![valence * 0.5, arousal * 0.3, consciousness * 0.4, 0.1],
            cycle_count: cycle,
            ..CognitiveSnapshot::dormant()
        };

        // Generate
        let artwork = symthaea_atelier::create_artwork_iterative(&config, &snapshot, 42 + cycle);

        // Evaluate
        let score = artwork.aesthetic_score;
        scores.push(score.composite);

        // Feedback
        let feedback = tracker.process(&score, &harmony_activations);

        // Evolve state based on feedback
        if feedback.dopamine_delta > 0.01 {
            // Reward: boost dominant harmony
            let dominant = harmony_activations
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            harmony_activations[dominant] = (harmony_activations[dominant] + 0.05).min(0.95);
            dopamine = (dopamine + feedback.dopamine_delta).min(1.0);
        }

        if feedback.serotonin_delta > 0.01 {
            consciousness = (consciousness + feedback.serotonin_delta * 0.3).min(0.95);
            valence = (valence + feedback.serotonin_delta * 0.2).min(0.8);
        }

        // Gentle decay
        arousal *= 0.98;
        dopamine *= 0.98;
    }

    // Verify: system should not degrade
    assert!(!scores.is_empty());
    let first_3_avg: f32 = scores[..3].iter().sum::<f32>() / 3.0;
    let last_3_avg: f32 = scores[scores.len() - 3..].iter().sum::<f32>() / 3.0;

    // Last 3 should be at least as good as first 3 (feedback prevents degradation)
    assert!(
        last_3_avg >= first_3_avg * 0.95,
        "scores degraded: first_3={first_3_avg:.3}, last_3={last_3_avg:.3}"
    );

    // EMA should have converged to a reasonable value
    let ema = tracker.expectation();
    assert!(ema > 0.3, "EMA too low: {ema}");
}

/// Test that Neural mode produces valid art.
#[test]
fn neural_mode_produces_valid_art() {
    let config = AtelierConfig {
        style: AtelierStyle::Neural,
        iteration_budget: 1,
        ..AtelierConfig::default()
    };

    let snapshot = CognitiveSnapshot {
        consciousness_level: 0.7,
        harmony_activations: [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2],
        dopamine: 0.6,
        serotonin: 0.5,
        valence: 0.3,
        arousal: 0.5,
        thought_vector: vec![0.3, -0.2],
        ..CognitiveSnapshot::dormant()
    };

    let artwork = symthaea_atelier::create_artwork(&config, &snapshot, 42);

    assert!(artwork.svg.contains("<svg"), "should produce valid SVG");
    assert!(
        artwork.scene.node_count() >= 5,
        "should have multiple elements"
    );
    assert!(
        artwork.aesthetic_score.composite >= 0.0,
        "score should be non-negative"
    );
    assert!(
        artwork.aesthetic_score.composite <= 1.0,
        "score should be bounded"
    );
}

/// Test that different cognitive states produce measurably different art.
#[test]
fn cognitive_states_differentiate_art() {
    let config = AtelierConfig {
        style: AtelierStyle::Composite,
        iteration_budget: 3,
        ..AtelierConfig::default()
    };

    let serene = CognitiveSnapshot {
        consciousness_level: 0.9,
        valence: 0.7,
        arousal: 0.2,
        harmony_activations: [0.8, 0.8, 0.7, 0.2, 0.7, 0.7, 0.5, 0.9],
        dopamine: 0.5,
        serotonin: 0.8,
        noradrenaline: 0.1,
        thought_vector: vec![0.1, 0.0, 0.1, -0.1],
        ..CognitiveSnapshot::dormant()
    };

    let turbulent = CognitiveSnapshot {
        consciousness_level: 0.5,
        valence: -0.7,
        arousal: 0.9,
        harmony_activations: [0.2, 0.1, 0.3, 0.9, 0.5, 0.1, 0.8, 0.1],
        dopamine: 0.3,
        serotonin: 0.2,
        noradrenaline: 0.9,
        thought_vector: vec![0.8, -0.7, 0.9, -0.5],
        ..CognitiveSnapshot::dormant()
    };

    let art_serene = symthaea_atelier::create_artwork_iterative(&config, &serene, 42);
    let art_turbulent = symthaea_atelier::create_artwork_iterative(&config, &turbulent, 42);

    // Scores should differ
    let score_diff =
        (art_serene.aesthetic_score.composite - art_turbulent.aesthetic_score.composite).abs();
    assert!(
        score_diff > 0.01,
        "states should produce different scores: serene={:.3}, turbulent={:.3}",
        art_serene.aesthetic_score.composite,
        art_turbulent.aesthetic_score.composite
    );

    // SVG content should differ
    assert_ne!(art_serene.svg, art_turbulent.svg, "SVGs should differ");

    // Element counts should differ (high consciousness = more layers)
    assert!(
        art_serene.scene.node_count() != art_turbulent.scene.node_count()
            || art_serene.svg.len() != art_turbulent.svg.len(),
        "art should be structurally different"
    );
}

/// Test the aesthetic tracker feedback mechanics.
#[test]
fn aesthetic_tracker_rewards_improvement() {
    let mut tracker = AestheticTracker::new(AestheticConfig::default());
    let harmonies = [0.5; 8];

    // Set low baseline
    for _ in 0..5 {
        let low = symthaea_aesthetic::AestheticScore::uniform(0.3);
        tracker.process(&low, &harmonies);
    }

    // Submit high score — should get positive dopamine
    let high = symthaea_aesthetic::AestheticScore::uniform(0.8);
    let feedback = tracker.process(&high, &harmonies);

    assert!(
        feedback.dopamine_delta > 0.0,
        "should reward improvement: DA={}",
        feedback.dopamine_delta
    );
    assert!(
        feedback.serotonin_delta > 0.0,
        "should have harmony satisfaction: 5-HT={}",
        feedback.serotonin_delta
    );
}
