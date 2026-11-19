// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded, seeded calls into the existing Symthaea Atelier generators.
//!
//! This module is an adapter, not a generator implementation. It deliberately
//! calls `symthaea_atelier::generate` and returns its existing Canvas scene.
//! It does not call Atelier's aesthetic scoring or selection loop.

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use symthaea_atelier::{AtelierConfig, AtelierStyle};
use symthaea_canvas::color::Color;
use symthaea_canvas::scene_graph::{FilterType, NodeKind, SceneNode};
use symthaea_canvas::{CognitiveSnapshot, SnapshotLimits};

use crate::{digest_json, safe_group_id, validate_namespace, AdapterError, AdapterResult};

const MAX_VIEWPORT_DIMENSION: f32 = 8_192.0;
const MAX_GENERATOR_ELEMENTS: usize = 10_000;
const MAX_SCENE_NODES: usize = 10_000;
const MAX_SCENE_DEPTH: usize = 128;
const MAX_POLYGON_POINTS: usize = 256;
const MAX_GRADIENT_STOPS: usize = 256;
const MAX_PATH_BYTES: usize = 512 * 1024;
const MAX_SCENE_TEXT_BYTES: usize = 512;

/// Request one deterministic, registered Atelier production.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AtelierProductionRequest {
    /// Namespace shared by a GGL document or compilation unit.
    pub namespace: String,
    /// Stable GGL call-site identity, unique within the namespace.
    pub instance_id: String,
    /// Existing allowlisted Atelier production family.
    pub style: AtelierStyle,
    /// Canvas/SVG viewport width in scene units.
    pub width: f32,
    /// Canvas/SVG viewport height in scene units.
    pub height: f32,
    /// Existing Atelier generator element cap.
    pub max_elements: usize,
    /// Hard cap on the complete returned Canvas scene subtree, including wrapper.
    pub max_scene_nodes: usize,
    /// Explicit deterministic RNG seed.
    pub seed: u64,
}

/// Receipt binding the request to the generated Canvas scene.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AtelierProductionReceipt {
    pub schema: String,
    pub adapter_version: String,
    pub namespace: String,
    pub instance_id: String,
    pub canvas_group_id: String,
    pub style: AtelierStyle,
    pub seed: u64,
    pub width: f32,
    pub height: f32,
    pub max_elements: usize,
    pub max_scene_nodes: usize,
    pub generated_scene_nodes: usize,
    pub request_digest: String,
    pub scene_digest: String,
}

/// Existing Atelier output plus the GGL-to-Atelier invocation receipt.
#[derive(Debug, Clone)]
pub struct AtelierProductionOutput {
    /// The generated scene wrapped in a stable GGL instance-ID group.
    pub scene: SceneNode,
    pub receipt: AtelierProductionReceipt,
}

fn finite(value: f32) -> bool {
    value.is_finite()
}

fn valid_unit(value: f32) -> bool {
    finite(value) && (0.0..=1.0).contains(&value)
}

fn valid_color(color: &Color) -> bool {
    valid_unit(color.r)
        && valid_unit(color.g)
        && valid_unit(color.b)
        && valid_unit(color.a)
}

fn valid_style(node: &SceneNode) -> bool {
    let style = &node.style;
    style.fill.as_ref().is_none_or(valid_color)
        && style.stroke.as_ref().is_none_or(valid_color)
        && style
            .stroke_width
            .is_none_or(|value| finite(value) && value >= 0.0)
        && style.opacity.is_none_or(valid_unit)
        && style
            .css_class
            .as_ref()
            .is_none_or(|value| value.len() <= MAX_SCENE_TEXT_BYTES)
        && style
            .fill_url
            .as_ref()
            .is_none_or(|value| value.len() <= MAX_SCENE_TEXT_BYTES)
        && style
            .filter
            .as_ref()
            .is_none_or(|value| value.len() <= MAX_SCENE_TEXT_BYTES)
}

fn valid_kind(kind: &NodeKind) -> bool {
    match kind {
        NodeKind::Group { id } => id
            .as_ref()
            .is_none_or(|value| value.len() <= MAX_SCENE_TEXT_BYTES),
        NodeKind::Circle { cx, cy, r } => {
            finite(*cx) && finite(*cy) && finite(*r) && *r >= 0.0
        }
        NodeKind::Ellipse { cx, cy, rx, ry } => {
            finite(*cx)
                && finite(*cy)
                && finite(*rx)
                && finite(*ry)
                && *rx >= 0.0
                && *ry >= 0.0
        }
        NodeKind::Line { x1, y1, x2, y2 } => {
            finite(*x1) && finite(*y1) && finite(*x2) && finite(*y2)
        }
        NodeKind::Polygon { points, .. } => {
            points.len() <= MAX_POLYGON_POINTS
                && points.iter().all(|(x, y)| finite(*x) && finite(*y))
        }
        NodeKind::Rect { x, y, w, h, rx } => {
            finite(*x)
                && finite(*y)
                && finite(*w)
                && finite(*h)
                && finite(*rx)
                && *w >= 0.0
                && *h >= 0.0
                && *rx >= 0.0
        }
        NodeKind::RadialGradient { id, stops } => {
            id.len() <= MAX_SCENE_TEXT_BYTES
                && stops.len() <= MAX_GRADIENT_STOPS
                && stops
                    .iter()
                    .all(|stop| valid_unit(stop.offset) && valid_color(&stop.color))
        }
        NodeKind::Filter { id, filter_type } => {
            if id.len() > MAX_SCENE_TEXT_BYTES {
                return false;
            }
            match filter_type {
                FilterType::Turbulence {
                    base_frequency,
                    num_octaves,
                    scale,
                } => {
                    finite(*base_frequency)
                        && *base_frequency >= 0.0
                        && *num_octaves > 0
                        && *num_octaves <= 16
                        && finite(*scale)
                        && *scale >= 0.0
                }
                FilterType::Blur { std_dev } => finite(*std_dev) && *std_dev >= 0.0,
            }
        }
        NodeKind::UseFilter { filter_id } => filter_id.len() <= MAX_SCENE_TEXT_BYTES,
        NodeKind::Path { d } => {
            let lower = d.to_ascii_lowercase();
            d.len() <= MAX_PATH_BYTES && !lower.contains("nan") && !lower.contains("inf")
        }
    }
}

fn validate_scene(node: &SceneNode, depth: usize, visited: &mut usize) -> AdapterResult<()> {
    if depth > MAX_SCENE_DEPTH {
        return Err(AdapterError(format!(
            "Atelier scene depth exceeds {MAX_SCENE_DEPTH}"
        )));
    }
    *visited = (*visited).saturating_add(1);
    if *visited > MAX_SCENE_NODES {
        return Err(AdapterError(format!(
            "Atelier scene node count exceeds {MAX_SCENE_NODES}"
        )));
    }
    if !valid_kind(&node.kind) || !valid_style(node) {
        return Err(AdapterError(
            "Atelier generated invalid or out-of-bounds Canvas data".into(),
        ));
    }

    let transform = &node.transform;
    if !finite(transform.translate_x)
        || !finite(transform.translate_y)
        || !finite(transform.rotate_deg)
        || !finite(transform.scale)
        || transform.scale.abs() > 10_000.0
        || transform.scale.abs() <= f32::EPSILON
        || transform.translate_x.abs() > 1.0e7
        || transform.translate_y.abs() > 1.0e7
        || transform.rotate_deg.abs() > 1.0e9
    {
        return Err(AdapterError(
            "Atelier generated invalid Canvas transform".into(),
        ));
    }

    for child in &node.children {
        validate_scene(child, depth + 1, visited)?;
    }
    Ok(())
}

/// Generate one allowlisted Atelier production with explicit seed and bounds.
///
/// This calls Atelier's public generator directly. It does not compute or use
/// Atelier's aesthetic score, rank variants, or claim that a generated scene
/// is beautiful. Human and QD selection remain with the existing owner systems.
pub fn generate_atelier_production(
    request: &AtelierProductionRequest,
    snapshot: &CognitiveSnapshot,
) -> AdapterResult<AtelierProductionOutput> {
    validate_namespace(&request.namespace)?;
    if request.instance_id.is_empty()
        || request.instance_id.len() > 256
        || !request
            .instance_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.:-".contains(&byte))
    {
        return Err(AdapterError(
            "instance_id must be 1-256 ASCII characters from [A-Za-z0-9_.:-]".into(),
        ));
    }
    if !finite(request.width)
        || !finite(request.height)
        || request.width <= 0.0
        || request.height <= 0.0
        || request.width > MAX_VIEWPORT_DIMENSION
        || request.height > MAX_VIEWPORT_DIMENSION
    {
        return Err(AdapterError(
            "Atelier viewport dimensions are outside the supported range".into(),
        ));
    }
    if request.max_elements == 0 || request.max_elements > MAX_GENERATOR_ELEMENTS {
        return Err(AdapterError(format!(
            "max_elements must be within 1..={MAX_GENERATOR_ELEMENTS}"
        )));
    }
    if request.max_scene_nodes < 2 || request.max_scene_nodes > MAX_SCENE_NODES {
        return Err(AdapterError(format!(
            "max_scene_nodes must be within 2..={MAX_SCENE_NODES}"
        )));
    }

    // Composite invokes multiple generators. Keep it out of the registry until
    // its aggregate compute budget has a separately qualified bound.
    match request.style {
        AtelierStyle::LSystem
        | AtelierStyle::ParametricCurve
        | AtelierStyle::PersistenceTexture
        | AtelierStyle::ColorField
        | AtelierStyle::ReactionDiffusion
        | AtelierStyle::StrangeAttractor
        | AtelierStyle::HofstadterButterfly
        | AtelierStyle::PersistenceDiagram => {}
        AtelierStyle::Composite => {
            return Err(AdapterError(
                "Atelier Composite is not registered: use explicit bounded productions".into(),
            ));
        }
    }

    // Do not silently accept a snapshot that Canvas would need to clamp or
    // truncate. The caller must repair and resubmit the actual source state.
    let (_, sanitization) = snapshot.sanitize_with_limits(SnapshotLimits::default());
    if sanitization.changed() {
        return Err(AdapterError(
            "Atelier snapshot requires sanitization; repair it before production".into(),
        ));
    }

    let config = AtelierConfig {
        width: request.width,
        height: request.height,
        max_elements: request.max_elements,
        style: request.style,
        iteration_budget: 1,
    };
    let mut rng = StdRng::seed_from_u64(request.seed);
    let generated = symthaea_atelier::generate(&config, snapshot, &mut rng);

    let mut visited = 0;
    validate_scene(&generated, 0, &mut visited)?;
    let wrapped_count = visited.saturating_add(1);
    if wrapped_count > request.max_scene_nodes {
        return Err(AdapterError(format!(
            "Atelier production emitted {wrapped_count} nodes, exceeding request cap {}",
            request.max_scene_nodes
        )));
    }

    let canvas_group_id = safe_group_id(&request.namespace, &request.instance_id);
    let scene = SceneNode::group(Some(canvas_group_id.as_str())).with_child(generated);
    let receipt = AtelierProductionReceipt {
        schema: "ggl-atelier-production-receipt-v1".into(),
        adapter_version: "0.2.0".into(),
        namespace: request.namespace.clone(),
        instance_id: request.instance_id.clone(),
        canvas_group_id,
        style: request.style,
        seed: request.seed,
        width: request.width,
        height: request.height,
        max_elements: request.max_elements,
        max_scene_nodes: request.max_scene_nodes,
        generated_scene_nodes: visited,
        request_digest: digest_json(request)?,
        scene_digest: digest_json(&scene)?,
    };

    Ok(AtelierProductionOutput { scene, receipt })
}
