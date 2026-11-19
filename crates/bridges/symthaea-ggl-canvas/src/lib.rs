// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![deny(unsafe_code)]

//! Adapter from compiled GGL 2-D geometry to the existing Symthaea Canvas IR.
//!
//! This crate does not parse GGL source and does not render SVG. It consumes
//! versioned compiled-scene JSON, creates symthaea_canvas::SceneNode values,
//! and emits a semantic-ID sidecar and deterministic adapter manifest.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use symthaea_canvas::scene_graph::{NodeKind, SceneNode, Style, Transform};

const ADAPTER_VERSION: &str = "0.2.0";
const CANVAS_SCHEMA: &str = "symthaea-canvas-scene-node-serde-v1";
const DIGEST_ENCODING: &str = "serde-json-compact-struct-order-v1";
const MAX_JSON_BYTES: usize = 32 * 1024 * 1024;
const MAX_UNITS_BYTES: usize = 64;
const MAX_PRIMITIVES: usize = 100_000;
const MAX_POLYGON_VERTICES: usize = 256;
const MAX_ID_BYTES: usize = 256;
const EPSILON: f64 = 1.0e-12;
const TAU: f64 = std::f64::consts::TAU;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterError(pub String);

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AdapterError {}

pub type AdapterResult<T> = Result<T, AdapterError>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IdentityEntry {
    pub semantic_id: String,
    pub canvas_group_id: String,
    pub scene_node_path: Vec<usize>,
    pub leaf_node_path: Vec<usize>,
    pub primitive_type: String,
    pub source_line: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IdentityMap {
    pub schema: String,
    pub adapter_version: String,
    pub namespace: String,
    pub canvas_schema: String,
    pub canvas_scene_digest: String,
    pub entries: Vec<IdentityEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdapterManifest {
    pub schema: String,
    pub adapter_version: String,
    pub canvas_schema: String,
    pub namespace: String,
    pub units: Option<String>,
    pub coordinate_conversion: String,
    pub unit_semantics: String,
    pub primitive_count: usize,
    pub source_sha256: String,
    pub program_sha256: String,
    pub geometry_sha256: String,
    pub scene_sha256: String,
    pub canvas_scene_digest: String,
    pub identity_map_digest: String,
    pub digest_encoding: String,
    pub unsupported_or_opaque_count: usize,
}

#[derive(Debug, Clone)]
pub struct AdapterOutput {
    pub scene: SceneNode,
    pub identity_map: IdentityMap,
    pub manifest: AdapterManifest,
}

fn fail<T>(message: impl Into<String>) -> AdapterResult<T> {
    Err(AdapterError(message.into()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("formatting into String");
    }
    output
}

fn digest_json<T: Serialize>(value: &T) -> AdapterResult<String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| AdapterError(format!("cannot serialize digest input: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn string_field<'a>(value: &'a Value, field: &str) -> AdapterResult<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| AdapterError(format!("{field} must be a non-empty string")))
}

fn number_field(value: &Value, field: &str) -> AdapterResult<f64> {
    let number = value
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| AdapterError(format!("{field} must be numeric")))?;
    if !number.is_finite() {
        return fail(format!("{field} must be finite"));
    }
    Ok(number)
}

fn to_f32(value: f64, field: &str) -> AdapterResult<f32> {
    let result = value as f32;
    if !result.is_finite() {
        return fail(format!("{field} is outside Canvas f32 range"));
    }
    Ok(result)
}

fn validate_sha256(field: &str, value: &str) -> AdapterResult<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return fail(format!("{field} must be a 64-character SHA-256 hex digest"));
    }
    Ok(())
}

fn validate_namespace(namespace: &str) -> AdapterResult<()> {
    if namespace.is_empty()
        || namespace.len() > 80
        || !namespace
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.:-".contains(&byte))
    {
        return fail("namespace must be 1-80 ASCII characters from [A-Za-z0-9_.:-]");
    }
    Ok(())
}

fn safe_group_id(namespace: &str, semantic_id: &str) -> String {
    let mut material = Vec::with_capacity(namespace.len() + semantic_id.len() + 1);
    material.extend_from_slice(namespace.as_bytes());
    material.push(0);
    material.extend_from_slice(semantic_id.as_bytes());
    format!("ggl-{}", &sha256_hex(&material)[..24])
}

fn empty_style() -> Style {
    Style {
        fill: None,
        fill_url: None,
        stroke: None,
        stroke_width: None,
        opacity: None,
        filter: None,
        css_class: Some("ggl-primitive".to_owned()),
    }
}

fn make_node(kind: NodeKind, style: Style) -> SceneNode {
    SceneNode {
        kind,
        transform: Transform::identity(),
        style,
        children: Vec::new(),
    }
}

fn svg_arc_path(primitive: &Value) -> AdapterResult<String> {
    let cx = number_field(primitive, "cx")?;
    let cy = number_field(primitive, "cy")?;
    let radius = number_field(primitive, "radius")?;
    let start = number_field(primitive, "start_rad")?.rem_euclid(TAU);
    let sweep = number_field(primitive, "sweep_rad")?;
    if radius <= 0.0 {
        return fail("arc radius must be positive");
    }
    if sweep.abs() <= EPSILON {
        return fail("arc sweep must be non-zero");
    }
    if sweep.abs() > TAU + 1.0e-9 {
        return fail("arc sweep greater than one revolution is unsupported");
    }

    // GGL uses y-up coordinates; SVG uses y-down, reversing orientation.
    let point = |angle: f64| (cx + radius * angle.cos(), -(cy + radius * angle.sin()));
    let format = |value: f64| format!("{value:.12}");
    let (x0, y0) = point(start);
    let sweep_flag = if sweep < 0.0 { 1 } else { 0 };

    if (sweep.abs() - TAU).abs() <= 1.0e-9 {
        let (xm, ym) = point(start + sweep / 2.0);
        return Ok(format!(
            "M {} {} A {} {} 0 0 {} {} {} A {} {} 0 0 {} {} {}",
            format(x0), format(y0), format(radius), format(radius), sweep_flag,
            format(xm), format(ym), format(radius), format(radius), sweep_flag,
            format(x0), format(y0)
        ));
    }

    let (x1, y1) = point(start + sweep);
    let large_arc = if sweep.abs() > std::f64::consts::PI { 1 } else { 0 };
    Ok(format!(
        "M {} {} A {} {} 0 {} {} {} {}",
        format(x0), format(y0), format(radius), format(radius), large_arc,
        sweep_flag, format(x1), format(y1)
    ))
}

fn compile_primitive(primitive: &Value) -> AdapterResult<(String, NodeKind)> {
    let primitive_type = string_field(primitive, "type")?;
    let semantic_id = string_field(primitive, "id")?;
    if semantic_id.len() > MAX_ID_BYTES {
        return fail(format!("primitive ID exceeds {MAX_ID_BYTES} bytes"));
    }

    let kind = match primitive_type {
        "circle" => {
            let cx = to_f32(number_field(primitive, "cx")?, "circle.cx")?;
            let cy = to_f32(-number_field(primitive, "cy")?, "circle.cy")?;
            let radius = number_field(primitive, "radius")?;
            if radius <= 0.0 {
                return fail(format!("circle {semantic_id} radius must be positive"));
            }
            NodeKind::Circle { cx, cy, r: to_f32(radius, "circle.radius")? }
        }
        "segment" => {
            let x1 = number_field(primitive, "x1")?;
            let y1 = number_field(primitive, "y1")?;
            let x2 = number_field(primitive, "x2")?;
            let y2 = number_field(primitive, "y2")?;
            if (x2 - x1).hypot(y2 - y1) <= EPSILON {
                return fail(format!("segment {semantic_id} has zero length"));
            }
            NodeKind::Line {
                x1: to_f32(x1, "segment.x1")?,
                y1: to_f32(-y1, "segment.y1")?,
                x2: to_f32(x2, "segment.x2")?,
                y2: to_f32(-y2, "segment.y2")?,
            }
        }
        "polygon" => {
            let vertices = primitive
                .get("vertices")
                .and_then(Value::as_array)
                .ok_or_else(|| AdapterError(format!("polygon {semantic_id} requires vertices")))?;
            if !(3..=MAX_POLYGON_VERTICES).contains(&vertices.len()) {
                return fail(format!("polygon {semantic_id} must have 3-{MAX_POLYGON_VERTICES} vertices"));
            }
            let mut points = Vec::with_capacity(vertices.len());
            for (index, vertex) in vertices.iter().enumerate() {
                let pair = vertex
                    .as_array()
                    .filter(|pair| pair.len() == 2)
                    .ok_or_else(|| AdapterError(format!("polygon {semantic_id} vertex {index} must be [x,y]")))?;
                let x = pair[0].as_f64().ok_or_else(|| AdapterError("polygon x must be numeric".into()))?;
                let y = pair[1].as_f64().ok_or_else(|| AdapterError("polygon y must be numeric".into()))?;
                if !x.is_finite() || !y.is_finite() {
                    return fail(format!("polygon {semantic_id} vertex {index} must be finite"));
                }
                points.push((to_f32(x, "polygon.x")?, to_f32(-y, "polygon.y")?));
            }
            NodeKind::Polygon { points, closed: true }
        }
        "arc" => NodeKind::Path { d: svg_arc_path(primitive)? },
        unsupported => return fail(format!("unsupported GGL primitive type: {unsupported}")),
    };
    Ok((primitive_type.to_owned(), kind))
}

/// Convert a compiled GGL 0.1 scene JSON document to the existing Canvas IR.
///
/// No units are converted: the declared unit is retained in the manifest. The
/// caller must supply an explicit viewport/unit conversion when needed.
pub fn adapt_compiled_scene(json: &str, namespace: &str) -> AdapterResult<AdapterOutput> {
    validate_namespace(namespace)?;
    if json.len() > MAX_JSON_BYTES {
        return fail(format!("compiled scene exceeds {MAX_JSON_BYTES} byte input limit"));
    }
    let source: Value = serde_json::from_str(json)
        .map_err(|error| AdapterError(format!("invalid compiled GGL JSON: {error}")))?;
    if string_field(&source, "language")? != "GGL" {
        return fail("input language must be GGL");
    }
    if string_field(&source, "version")? != "0.1" {
        return fail("only compiled GGL version 0.1 is supported");
    }

    let primitives = source
        .get("primitives")
        .and_then(Value::as_array)
        .ok_or_else(|| AdapterError("compiled GGL scene must contain a primitives array".into()))?;
    if primitives.len() > MAX_PRIMITIVES {
        return fail(format!("primitive count exceeds limit {MAX_PRIMITIVES}"));
    }
    let declared_count = source
        .get("primitive_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| AdapterError("primitive_count must be an unsigned integer".into()))?;
    if declared_count != primitives.len() as u64 {
        return fail("primitive_count does not match the primitives array");
    }

    let source_sha = string_field(&source, "source_sha256")?.to_owned();
    let program_sha = string_field(&source, "program_sha256")?.to_owned();
    let geometry_sha = string_field(&source, "geometry_sha256")?.to_owned();
    let scene_sha = string_field(&source, "scene_sha256")?.to_owned();
    for (name, digest) in [
        ("source_sha256", source_sha.as_str()),
        ("program_sha256", program_sha.as_str()),
        ("geometry_sha256", geometry_sha.as_str()),
        ("scene_sha256", scene_sha.as_str()),
    ] {
        validate_sha256(name, digest)?;
    }

    let units = match source.get("units") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if !value.is_empty() && value.len() <= MAX_UNITS_BYTES => {
            Some(value.clone())
        }
        Some(_) => {
            return fail(format!(
                "units must be a non-empty string of at most {MAX_UNITS_BYTES} bytes"
            ));
        }
    };
    let mut ids = BTreeSet::new();
    let mut wrappers = Vec::with_capacity(primitives.len());
    let mut entries = Vec::with_capacity(primitives.len());

    for (index, primitive) in primitives.iter().enumerate() {
        let semantic_id = string_field(primitive, "id")?.to_owned();
        if !ids.insert(semantic_id.clone()) {
            return fail(format!("duplicate primitive ID: {semantic_id}"));
        }
        let (primitive_type, kind) = compile_primitive(primitive)?;
        let canvas_id = safe_group_id(namespace, &semantic_id);
        let leaf = make_node(kind, empty_style());
        let wrapper = SceneNode::group(Some(&canvas_id)).with_child(leaf);
        wrappers.push(wrapper);
        entries.push(IdentityEntry {
            semantic_id,
            canvas_group_id: canvas_id,
            scene_node_path: vec![index],
            leaf_node_path: vec![index, 0],
            primitive_type,
            source_line: primitive.get("source_line").and_then(Value::as_u64).map(|line| line as usize),
        });
    }

    let root_material = format!("{namespace}\0{source_sha}\0{program_sha}");
    let root_id = format!("ggl-root-{}", &sha256_hex(root_material.as_bytes())[..16]);
    let mut scene = SceneNode::group(Some(&root_id));
    scene.children = wrappers;
    let canvas_scene_digest = digest_json(&scene)?;

    let identity_map = IdentityMap {
        schema: "ggl-canvas-identity-map-v1".into(),
        adapter_version: ADAPTER_VERSION.into(),
        namespace: namespace.into(),
        canvas_schema: CANVAS_SCHEMA.into(),
        canvas_scene_digest: canvas_scene_digest.clone(),
        entries,
    };
    let identity_map_digest = digest_json(&identity_map)?;

    let manifest = AdapterManifest {
        schema: "ggl-canvas-adapter-manifest-v1".into(),
        adapter_version: ADAPTER_VERSION.into(),
        canvas_schema: CANVAS_SCHEMA.into(),
        namespace: namespace.into(),
        units,
        coordinate_conversion: "GGL y-up -> Canvas/SVG y-down; x unchanged".into(),
        unit_semantics: "preserved as metadata; no implicit physical-unit conversion".into(),
        primitive_count: primitives.len(),
        source_sha256: source_sha,
        program_sha256: program_sha,
        geometry_sha256: geometry_sha,
        scene_sha256: scene_sha,
        canvas_scene_digest,
        identity_map_digest,
        digest_encoding: DIGEST_ENCODING.into(),
        unsupported_or_opaque_count: 0,
    };

    Ok(AdapterOutput { scene, identity_map, manifest })
}
