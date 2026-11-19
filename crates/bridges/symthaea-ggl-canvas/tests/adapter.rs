// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_canvas::render_svg;
use symthaea_canvas::scene_graph::NodeKind;
use symthaea_ggl_canvas::adapt_compiled_scene;

const GGL_PROGRAM: &str = include_str!("../../../domains/symthaea-canvas/tests/fixtures/ggl-program.ggl");
const COMPILED: &str = include_str!("../../../domains/symthaea-canvas/tests/fixtures/ggl-source.json");

#[test]
fn adapts_all_supported_v01_primitives_and_renders_stable_ids() {
    let output = adapt_compiled_scene(COMPILED, "ggl-rust-conformance").expect("valid compiled scene");
    assert_eq!(output.manifest.primitive_count, 4);
    assert_eq!(output.manifest.units.as_deref(), Some("mm"));
    assert_eq!(output.identity_map.entries.len(), 4);
    assert_eq!(output.manifest.canvas_scene_digest, output.identity_map.canvas_scene_digest);
    assert_eq!(output.manifest.source_sha256.len(), 64);
    assert!(!GGL_PROGRAM.is_empty());

    let leaves: Vec<_> = output.scene.children.iter().map(|wrapper| {
        assert!(matches!(&wrapper.kind, NodeKind::Group { id: Some(_) }));
        assert_eq!(wrapper.children.len(), 1);
        &wrapper.children[0]
    }).collect();
    assert!(matches!(&leaves[0].kind, NodeKind::Circle { .. }));
    assert!(matches!(&leaves[1].kind, NodeKind::Line { .. }));
    assert!(matches!(&leaves[2].kind, NodeKind::Polygon { closed: true, .. }));
    match &leaves[3].kind {
        NodeKind::Path { d } => assert!(d.contains(" A 3.000000000000 3.000000000000 0 0 0")),
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
fn rejects_unsafe_namespaces_and_non_finite_geometry() {
    assert!(adapt_compiled_scene(COMPILED, "bad namespace").is_err());
    let invalid = COMPILED.replace("\"radius\":4.75", "\"radius\":1e999");
    assert!(adapt_compiled_scene(&invalid, "fixture").is_err());
}
