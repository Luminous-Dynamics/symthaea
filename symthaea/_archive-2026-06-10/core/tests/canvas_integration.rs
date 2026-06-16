// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canvas Living Topology Integration Tests
//!
//! Verifies the canvas pipeline is wired end-to-end:
//! 1. CanvasManager is initialized when canvas feature is enabled
//! 2. CycleResult.canvas_svg is populated after enough cycles (generation_interval)
//! 3. Canvas telemetry appears in CycleMetadata
//! 4. Aesthetic score is computed and bounded
//!
//! Run with:
//! ```bash
//! cargo test --test canvas_integration --features canvas
//! ```

#![cfg(feature = "canvas")]

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_canvas_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .expect("CognitiveLoopService::new should succeed")
}

#[test]
fn test_canvas_manager_active() {
    let service = make_canvas_service();
    assert!(service.has_canvas(), "canvas manager should be active");
}

#[test]
fn test_canvas_svg_generated_after_interval() {
    let mut service = make_canvas_service();
    let mut found_svg = false;

    // Default generation_interval is 5, so run 10 cycles to guarantee at least one SVG.
    for i in 0..10 {
        let input = format!("canvas test cycle {i}");
        let result = service.cycle(&input);
        if result.canvas_svg.is_some() {
            found_svg = true;
            let svg = result.canvas_svg.as_ref().unwrap();
            assert!(svg.starts_with("<svg"), "SVG should start with <svg tag");
            assert!(svg.contains("</svg>"), "SVG should be well-formed");
            break;
        }
    }
    assert!(found_svg, "canvas SVG should be generated within 10 cycles");
}

#[test]
fn test_canvas_telemetry_in_metadata() {
    let mut service = make_canvas_service();
    let mut found_telemetry = false;

    for i in 0..10 {
        let input = format!("telemetry test {i}");
        let result = service.cycle(&input);
        if let Some(ref canvas) = result.metadata.canvas {
            if canvas.generated {
                found_telemetry = true;
                assert!(canvas.aesthetic_score >= 0.0 && canvas.aesthetic_score <= 1.0);
                assert!(canvas.node_count > 0, "generated frame should have nodes");
                assert!(canvas.svg_bytes > 0, "generated frame should have bytes");
                break;
            }
        }
    }
    assert!(
        found_telemetry,
        "canvas telemetry should report generation within 10 cycles"
    );
}

#[test]
fn test_canvas_aesthetic_score_accessor() {
    let mut service = make_canvas_service();
    // Run enough cycles to generate at least one frame
    for i in 0..10 {
        service.cycle(&format!("accessor test {i}"));
    }
    let score = service.canvas_aesthetic_score();
    assert!(
        score >= 0.0 && score <= 1.0,
        "aesthetic score should be in [0,1], got {score}"
    );
}

#[test]
fn test_canvas_deterministic_same_seed() {
    let mut s1 = make_canvas_service();
    let mut s2 = make_canvas_service();

    let mut svg1 = None;
    let mut svg2 = None;

    for i in 0..10 {
        let input = format!("deterministic {i}");
        let r1 = s1.cycle(&input);
        let r2 = s2.cycle(&input);
        if r1.canvas_svg.is_some() {
            svg1 = r1.canvas_svg;
            svg2 = r2.canvas_svg;
            break;
        }
    }

    assert!(svg1.is_some(), "should have generated SVG");
    assert_eq!(svg1, svg2, "same seed + input → same SVG");
}