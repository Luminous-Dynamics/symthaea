// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! GGL reference-compiler -> Canvas serde/render conformance.

use serde_json::Value;
use sha2::{Digest, Sha256};
use symthaea_canvas::render_svg;
use symthaea_canvas::scene_graph::{NodeKind, SceneNode};

const GGL_PROGRAM: &str = include_str!("fixtures/ggl-program.ggl");
const GGL_SOURCE: &str = include_str!("fixtures/ggl-source.json");
const CANVAS_JSON: &str = include_str!("fixtures/canvas-scene.json");
const IDENTITY_JSON: &str = include_str!("fixtures/identity-map.json");
const MANIFEST_JSON: &str = include_str!("fixtures/adapter-manifest.json");

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("String formatting");
    }
    out
}

fn assert_digest(value: &Value, field: &str) {
    let digest = value[field].as_str().expect("digest string");
    assert_eq!(digest.len(), 64, "{field} length");
    assert!(digest.bytes().all(|b| b.is_ascii_hexdigit()), "{field} must be hex");
}

fn close(actual: f32, expected: f64, field: &str) {
    let actual = f64::from(actual);
    let tolerance = 1.0e-5 * expected.abs().max(1.0);
    assert!((actual - expected).abs() <= tolerance, "{field}: {actual} != {expected}");
}

fn group_id(node: &SceneNode) -> &str {
    match &node.kind {
        NodeKind::Group { id: Some(id) } => id,
        other => panic!("expected stable-ID Group, got {other:?}"),
    }
}

#[test]
fn ggl_adapter_deserializes_preserves_geometry_and_renders_stable_ids() {
    let source: Value = serde_json::from_str(GGL_SOURCE).expect("compiled GGL fixture");
    let identity: Value = serde_json::from_str(IDENTITY_JSON).expect("identity map fixture");
    let manifest: Value = serde_json::from_str(MANIFEST_JSON).expect("adapter manifest fixture");
    let canvas: SceneNode = serde_json::from_str(CANVAS_JSON).expect("actual Canvas SceneNode serde");

    let primitives = source["primitives"].as_array().expect("primitive list");
    let entries = identity["entries"].as_array().expect("identity entries");
    assert_eq!(source["language"].as_str(), Some("GGL"));
    assert_eq!(source["version"].as_str(), Some("0.1"));
    assert_eq!(primitives.len(), 4);
    assert_eq!(entries.len(), 4);
    assert_eq!(canvas.children.len(), 4);
    assert_eq!(manifest["primitive_count"].as_u64(), Some(4));
    assert_eq!(manifest["units"], source["units"]);
    assert_eq!(manifest["source_sha256"].as_str(), Some(sha256_hex(GGL_PROGRAM.as_bytes()).as_str()));
    for field in ["program_sha256", "geometry_sha256", "scene_sha256"] {
        assert_eq!(manifest[field], source[field], "{field} binding");
    }
    assert_eq!(identity["canvas_scene_digest"], manifest["canvas_scene_digest"]);
    assert_eq!(identity["namespace"], manifest["namespace"]);
    assert_eq!(manifest["unsupported_or_opaque_count"].as_u64(), Some(0));
    for field in ["source_sha256", "program_sha256", "geometry_sha256", "scene_sha256", "canvas_scene_digest", "identity_map_digest"] {
        assert_digest(&manifest, field);
    }

    let serialized = serde_json::to_value(&canvas).expect("Canvas SceneNode reserializes");
    assert_eq!(serialized["children"].as_array().map(Vec::len), Some(4));

    let expected_types = ["circle", "segment", "polygon", "arc"];
    let mut seen = std::collections::BTreeSet::new();
    let mut ids = Vec::new();
    for (index, ((primitive, entry), wrapper)) in primitives.iter().zip(entries).zip(&canvas.children).enumerate() {
        let semantic_id = primitive["id"].as_str().expect("semantic ID");
        let canvas_id = group_id(wrapper);
        assert_eq!(primitive["type"].as_str(), Some(expected_types[index]));
        assert_eq!(entry["semantic_id"].as_str(), Some(semantic_id));
        assert_eq!(entry["canvas_group_id"].as_str(), Some(canvas_id));
        assert!(seen.insert(canvas_id.to_owned()), "duplicate group ID");
        assert!(canvas_id.starts_with("ggl-"));
        assert_eq!(wrapper.children.len(), 1);
        assert_eq!(entry["scene_node_path"][0].as_u64(), Some(index as u64));
        assert_eq!(entry["leaf_node_path"][0].as_u64(), Some(index as u64));
        assert_eq!(entry["leaf_node_path"][1].as_u64(), Some(0));
        let leaf = &wrapper.children[0];
        match expected_types[index] {
            "circle" => match &leaf.kind {
                NodeKind::Circle { cx, cy, r } => {
                    close(*cx, primitive["cx"].as_f64().unwrap(), "circle.x");
                    close(*cy, -primitive["cy"].as_f64().unwrap(), "circle.y");
                    close(*r, primitive["radius"].as_f64().unwrap(), "circle.radius");
                }
                other => panic!("circle kind mismatch: {other:?}"),
            },
            "segment" => match &leaf.kind {
                NodeKind::Line { x1, y1, x2, y2 } => {
                    close(*x1, primitive["x1"].as_f64().unwrap(), "segment.x1");
                    close(*y1, -primitive["y1"].as_f64().unwrap(), "segment.y1");
                    close(*x2, primitive["x2"].as_f64().unwrap(), "segment.x2");
                    close(*y2, -primitive["y2"].as_f64().unwrap(), "segment.y2");
                }
                other => panic!("segment kind mismatch: {other:?}"),
            },
            "polygon" => match &leaf.kind {
                NodeKind::Polygon { points, closed } => {
                    assert!(*closed);
                    let vertices = primitive["vertices"].as_array().unwrap();
                    assert_eq!(points.len(), vertices.len());
                    for (actual, expected) in points.iter().zip(vertices) {
                        close(actual.0, expected[0].as_f64().unwrap(), "polygon.x");
                        close(actual.1, -expected[1].as_f64().unwrap(), "polygon.y");
                    }
                }
                other => panic!("polygon kind mismatch: {other:?}"),
            },
            "arc" => match &leaf.kind {
                NodeKind::Path { d } => {
                    assert!(d.starts_with("M "));
                    assert_eq!(d.matches(" A ").count(), 1);
                    assert!(d.contains("A 3 3 0 0 0"));
                }
                other => panic!("arc kind mismatch: {other:?}"),
            },
            _ => unreachable!(),
        }
        ids.push(canvas_id.to_owned());
    }

    let svg = render_svg(&canvas, 0.5);
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>\n"));
    for id in ids {
        assert_eq!(svg.matches(&format!(r#"id="{id}""#)).count(), 1, "SVG must retain {id}");
    }
    assert!(!svg.contains("NaN"));
    assert!(!svg.contains("inf"));
}
