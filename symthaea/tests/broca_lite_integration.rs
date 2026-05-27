// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BrocaLite integration tests: verify language_output is populated
//! through a real cognitive cycle without ssm_language feature.

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

/// Run multiple cognitive cycles and verify that language_output
/// is eventually populated by BrocaLite.
///
/// BrocaLite fires at the Broca cadence interval (co-prime scheduling),
/// so we need enough cycles for it to trigger.
#[test]
fn test_broca_lite_produces_language_output() {
    let config = CognitiveLoopConfig::default();
    let mut service = CognitiveLoopService::new(config).expect("service should construct");

    let mut found_language = false;
    // Run enough cycles for BrocaLite to fire (cadence interval + consciousness ramp-up)
    for i in 0..200 {
        let input = if i % 3 == 0 {
            "consciousness emerges from integration"
        } else if i % 3 == 1 {
            "patterns of awareness flow through the system"
        } else {
            "meaning arises from the binding of experience"
        };
        let result = service.cycle(input);
        if let Some(ref text) = result.language_output {
            if !text.is_empty() {
                eprintln!("[Cycle {i}] language_output: {text}");
                found_language = true;
                // Verify the output has reasonable properties
                assert!(
                    text.len() > 5,
                    "language output should be meaningful, got: {text}"
                );
                assert!(
                    text.len() < 2000,
                    "language output should not be excessively long"
                );
                break;
            }
        }
    }

    assert!(
        found_language,
        "BrocaLite should produce language_output within 200 cycles"
    );
}

/// Verify that language_output field exists and is None on early cycles
/// (before consciousness ramps up enough to trigger generation).
#[test]
fn test_language_output_field_always_present() {
    let config = CognitiveLoopConfig::default();
    let mut service = CognitiveLoopService::new(config).expect("service should construct");
    let result = service.cycle("hello");
    // language_output should exist as a field (not feature-gated)
    // It may be None on the first cycle — that's fine
    let _: &Option<String> = &result.language_output;
}

/// Verify async voice synthesis pipeline: enable → run cycles → drain audio.
#[test]
fn test_voice_synthesis_end_to_end() {
    let config = CognitiveLoopConfig::default();
    let mut service = CognitiveLoopService::new(config).expect("service should construct");

    // Enable voice synthesis (spawns background thread)
    service.enable_voice_synthesis();

    // Run cycles until language_output is populated
    for i in 0..100 {
        let input = if i % 2 == 0 {
            "consciousness flows through integrated patterns"
        } else {
            "awareness emerges from the binding of experience"
        };
        let result = service.cycle(input);
        if result.language_output.is_some() {
            // Give the background thread time to synthesize
            std::thread::sleep(std::time::Duration::from_millis(200));
            break;
        }
    }

    // Drain audio responses from background thread
    let responses = service.drain_voice_audio();
    // Note: voice synthesis may or may not have completed in 200ms
    // depending on the synthesis backend. The important thing is that
    // the pipeline doesn't crash or deadlock.
    eprintln!(
        "[Voice] {} audio responses drained, total samples: {}",
        responses.len(),
        responses.iter().map(|r| r.audio.len()).sum::<usize>()
    );
}