// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Glyph Codex integration tests.
//!
//! Verifies the full glyph pipeline: input text → GlyphManager encodes →
//! projects onto 11 Field Modality axes → consciousness modulation →
//! telemetry populated in CycleMetadata.

#![cfg(feature = "glyph_codex")]

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ConsciousnessProfile};

const N_CYCLES: usize = 100;

const INPUTS: &[&str] = &[
    "The weather is warm today.",
    "I need to solve this problem efficiently.",
    "Music brings people together in unexpected ways.",
    "How does photosynthesis convert light into energy?",
    "The architecture of this building is remarkable.",
    "We should consider the ethical implications carefully.",
    "Water flows downhill following the path of least resistance.",
    "Learning happens best through deliberate practice.",
    "The stars are beautiful tonight in the clear sky.",
    "Complex systems often exhibit emergent behavior.",
];

fn build_service() -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
    config.async_training = false;
    config.genesis_phrase = Some("glyph_codex_test_2026".to_string());
    CognitiveLoopService::new(config).expect("CognitiveLoopService should construct")
}

#[test]
fn glyph_telemetry_populates_after_warmup() {
    let mut svc = build_service();

    // Run enough cycles for the GlyphManager (interval 43) to fire at least twice
    let results: Vec<_> = (0..N_CYCLES)
        .map(|i| svc.cycle(INPUTS[i % INPUTS.len()]))
        .collect();

    let last = results.last().unwrap();
    let m = &last.metadata;

    // After 100 cycles, the glyph manager should have run at least once (cycle 43, 86)
    // and populated telemetry fields
    assert!(
        !m.glyph_dominant_modality.is_empty(),
        "glyph_dominant_modality should be populated after {} cycles, got empty",
        N_CYCLES
    );
    assert!(
        m.glyph_coherence >= 0.0 && m.glyph_coherence <= 0.95,
        "glyph_coherence should be in [0, 0.95], got {}",
        m.glyph_coherence
    );
    assert!(
        m.glyph_spiral_position >= 0.0 && m.glyph_spiral_position <= 56.0,
        "glyph_spiral_position should be in [0, 56], got {}",
        m.glyph_spiral_position
    );
}

#[test]
fn glyph_dominant_modality_is_valid() {
    let mut svc = build_service();

    // Run past interval 43
    for i in 0..50 {
        let _ = svc.cycle(INPUTS[i % INPUTS.len()]);
    }
    let result = svc.cycle("root ground earth body anchor stabilize presence");
    let modality = &result.metadata.glyph_dominant_modality;

    // Should be one of the 11 valid modalities
    let valid = [
        "Metaharmonic",
        "Threshold",
        "Rooting",
        "Resonant",
        "Transitional",
        "Igniting",
        "Witnessing",
        "Bridging",
        "Reflective",
        "Revealing",
        "Integrated",
    ];
    assert!(
        modality.is_empty() || valid.contains(&modality.as_str()),
        "glyph_dominant_modality '{}' is not a valid FieldModality",
        modality
    );
}

#[test]
fn glyph_consciousness_modulation_bounded() {
    let mut svc = build_service();

    // Run 100 cycles and verify consciousness level stays bounded
    for i in 0..N_CYCLES {
        let result = svc.cycle(INPUTS[i % INPUTS.len()]);
        let psi = result.metadata.consciousness.consciousness_level;
        assert!(
            psi >= 0.0 && psi <= 1.0,
            "consciousness_level out of bounds at cycle {}: {}",
            i,
            psi
        );
    }
}

#[test]
fn glyph_spiral_does_not_exceed_bounds() {
    let mut svc = build_service();

    // Run 200 cycles with varied input
    for i in 0..200 {
        let result = svc.cycle(INPUTS[i % INPUTS.len()]);
        let sp = result.metadata.glyph_spiral_position;
        assert!(
            sp >= 0.0 && sp <= 56.0,
            "glyph_spiral_position out of bounds at cycle {}: {}",
            i,
            sp
        );
    }
}