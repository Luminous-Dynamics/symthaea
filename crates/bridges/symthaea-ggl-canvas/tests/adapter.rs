// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use sha2::{Digest, Sha256};
use symthaea_canvas::render_svg;
use symthaea_canvas::scene_graph::NodeKind;
use symthaea_ggl_canvas::adapt_compiled_scene;

const GGL_PROGRAM: &str =
    include_str!("../../../domains/symthaea-canvas/tests/fixtures/ggl-program.ggl");
const COMPILED: &str =
    include_str!("../../../domains/symthaea-canvas/tests/fixtures/ggl-source.json");

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("formatting into String");
    }
    output
}

#[test]
fn adapts_supported_primitives_and_renders_stable_ids() {
    let output = adapt_compiled_scene(COMPILED, "ggl-rust-conformance")
        .expect("valid compiled scene");
    assert_eq!(output.manifest.primitive_count, 4);
    assert_eq!(output.manifest.units.as_deref(), Some("mm"));
    assert_eq!(output.identity_map.entries.len(), 4);
    assert_eq!(
        output.manifest.canvas_scene_digest,
        output.identity_map.canvas_scene_digest
    );
    assert_eq!(
        output.manifest.source_sha256,
        sha256_hex(GGL_PROGRAM.as_bytes())
    );
    assert_eq!(
        output.manifest.digest_encoding,
        "serde-json-compact-struct-order-v1"
    );

    let canvas_ids: std::collections::BTreeSet<_> = output
        .identity_map
        .entries
        .iter()
        .map(|entry| entry.canvas_group_id.as_str())
        .collect();
    assert_eq!(canvas_ids.len(), output.identity_map.entries.len());

    let leaves: Vec<_> = output
        .scene
        .children
        .iter()
        .map(|wrapper| {
            assert!(matches!(&wrapper.kind, NodeKind::Group { id: Some(_) }));
            assert_eq!(wrapper.children.len(), 1);
            &wrapper.children[0]
        })
        .collect();

    match &leaves[0].kind {
        NodeKind::Circle { cx, cy, r } => {
            assert!((*cx - 1.0).abs() < 1.0e-6);
            assert!((*cy + 2.0).abs() < 1.0e-6, "GGL y-up must map to Canvas y-down");
            assert!((*r - 4.75).abs() < 1.0e-6);
        }
        other => panic!("circle mapping mismatch: {other:?}"),
    }
    match &leaves[1].kind {
        NodeKind::Line { x1, y1, x2, y2 } => {
            assert!((*x1 + 2.0).abs() < 1.0e-6);
            assert!((*y1 + 1.25).abs() < 1.0e-6);
            assert!((*x2 - 5.5).abs() < 1.0e-6);
            assert!((*y2 - 4.0).abs() < 1.0e-6);
        }
        other => panic!("segment mapping mismatch: {other:?}"),
    }
    match &leaves[2].kind {
        NodeKind::Polygon { points, closed } => {
            assert!(*closed);
            assert_eq!(points.len(), 3);
            assert!((points[0].0 - 3.9344428).abs() < 1.0e-5);
            assert!((points[0].1 + 2.623735).abs() < 1.0e-5);
        }
        other => panic!("polygon mapping mismatch: {other:?}"),
    }
    match &leaves[3].kind {
        NodeKind::Path { d } => {
            assert!(d.contains(" A 3.000000000000 3.000000000000 0 0 0"));
        }
        other => panic!("arc must become a Canvas path, got {other:?}"),
    }

    let svg = render_svg(&output.scene, 0.5);
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>\n"));
    for entry in &output.identity_map.entries {
        let needle = format!(r#"id="{}""#, entry.canvas_group_id);
        assert_eq!(svg.matches(needle.as_str()).count(), 1);
    }
}

#[test]
fn repeated_adaptation_is_deterministic() {
    let first = adapt_compiled_scene(COMPILED, "ggl-rust-conformance").unwrap();
    let second = adapt_compiled_scene(COMPILED, "ggl-rust-conformance").unwrap();
    assert_eq!(first.manifest, second.manifest);
    assert_eq!(first.identity_map, second.identity_map);
    assert_eq!(
        serde_json::to_vec(&first.scene).unwrap(),
        serde_json::to_vec(&second.scene).unwrap()
    );
}

#[test]
fn rejects_duplicate_ids_unknown_primitives_and_invalid_arcs() {
    let mut duplicate: serde_json::Value = serde_json::from_str(COMPILED).unwrap();
    duplicate["primitives"][3]["id"] = "circle.primary".into();
    assert!(adapt_compiled_scene(&duplicate.to_string(), "fixture").is_err());

    let mut unsupported: serde_json::Value = serde_json::from_str(COMPILED).unwrap();
    unsupported["primitives"][3]["type"] = "bezier".into();
    assert!(adapt_compiled_scene(&unsupported.to_string(), "fixture").is_err());

    let mut invalid_arc: serde_json::Value = serde_json::from_str(COMPILED).unwrap();
    invalid_arc["primitives"][3]["sweep_rad"] = 9.0.into();
    assert!(adapt_compiled_scene(&invalid_arc.to_string(), "fixture").is_err());
}

#[test]
fn rejects_unsafe_namespaces_non_finite_geometry_and_oversized_input() {
    assert!(adapt_compiled_scene(COMPILED, "bad namespace").is_err());

    let invalid_number = COMPILED.replace("\"radius\":4.75", "\"radius\":1e999");
    assert!(adapt_compiled_scene(&invalid_number, "fixture").is_err());

    let oversized = " ".repeat(32 * 1024 * 1024 + 1);
    assert!(adapt_compiled_scene(&oversized, "fixture").is_err());
}

#[test]
fn requires_consistent_counts_and_well_formed_units() {
    let mut missing_count: serde_json::Value = serde_json::from_str(COMPILED).unwrap();
    missing_count.as_object_mut().unwrap().remove("primitive_count");
    assert!(adapt_compiled_scene(&missing_count.to_string(), "fixture").is_err());

    let mut mismatched_count: serde_json::Value = serde_json::from_str(COMPILED).unwrap();
    mismatched_count["primitive_count"] = 99.into();
    assert!(adapt_compiled_scene(&mismatched_count.to_string(), "fixture").is_err());

    let mut invalid_units: serde_json::Value = serde_json::from_str(COMPILED).unwrap();
    invalid_units["units"] = 123.into();
    assert!(adapt_compiled_scene(&invalid_units.to_string(), "fixture").is_err());

    let mut oversized_units: serde_json::Value = serde_json::from_str(COMPILED).unwrap();
    oversized_units["units"] = "m".repeat(65).into();
    assert!(adapt_compiled_scene(&oversized_units.to_string(), "fixture").is_err());
}

#[test]
fn supports_full_circle_arcs_as_two_svg_arc_segments() {
    let mut full_circle: serde_json::Value = serde_json::from_str(COMPILED).unwrap();
    full_circle["primitives"][3]["sweep_rad"] = std::f64::consts::TAU.into();
    let output = adapt_compiled_scene(&full_circle.to_string(), "fixture").unwrap();
    match &output.scene.children[3].children[0].kind {
        NodeKind::Path { d } => assert_eq!(d.matches(" A ").count(), 2),
        other => panic!("full circular arc must remain a path: {other:?}"),
    }
}

#[test]
fn normalizes_large_finite_arc_angles() {
    let mut large_start: serde_json::Value = serde_json::from_str(COMPILED).unwrap();
    large_start["primitives"][3]["start_rad"] = 1.0e100.into();
    let output = adapt_compiled_scene(&large_start.to_string(), "fixture").unwrap();
    assert!(matches!(
        &output.scene.children[3].children[0].kind,
        NodeKind::Path { d } if d.starts_with("M ")
    ));
}
