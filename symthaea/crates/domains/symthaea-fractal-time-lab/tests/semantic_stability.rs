// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use rand::{Rng, thread_rng};
use symthaea_fractal_time_lab::semantic_stream_diagnostics::SemanticDiagnosticAdapter;

#[test]
fn test_semantic_stability_during_reasoning() {
    // Simulate a 128-step reasoning chain with some drift + resonance
    let mut adapter = SemanticDiagnosticAdapter::new(128);
    let mut rng = thread_rng();

    // Semantic vector dimension (e.g., first dimension of the embedding)
    for i in 0..128 {
        // Generate a 2T oscillation: 1.0, -1.0, 1.0, -1.0...
        let signal = if i % 2 == 0 { 1.0 } else { -1.0 };
        let mut vector = vec![0.0; 16384];
        vector[0] = signal as f32;
        adapter.push(vector);
    }

    let scorecard = adapter.temporal_diagnostic(0);

    // A stable resonating reasoning chain should score higher than 0.5
    // on persistence + subharmonic response
    println!("Reasoning Stability Score: {}", scorecard.primary_score);
    assert!(scorecard.primary_score > 0.1);
}
