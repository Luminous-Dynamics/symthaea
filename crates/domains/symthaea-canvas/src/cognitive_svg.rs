// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Static, accessible SVG projection of the evidence-bearing cognitive graph.
//!
//! This renderer is intentionally non-animated. It presents the same
//! renderer-neutral projection used by native/browser renderers and preserves
//! semantic IDs, epistemic status, producer provenance and event references.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::Write as _;

use crate::cognitive_layout::{CognitiveAnchor, CognitiveSpatialLink, CognitiveSpatialProjection};
use crate::cognitive_morphology::{CognitiveHypha, CognitiveMorphologyProjection};
use crate::cognitive_topology::{
    CognitiveNodeKind, CognitiveProvenance, CognitiveRelation, EpistemicStatus,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CognitiveSvgOptions {
    pub width: u32,
    pub height: u32,
    pub padding: f32,
    pub show_labels: bool,
}

impl Default for CognitiveSvgOptions {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 900,
            padding: 72.0,
            show_labels: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveSvgError {
    InvalidOptions,
    InvalidProjection,
}

impl fmt::Display for CognitiveSvgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions => write!(f, "cognitive SVG dimensions or padding are invalid"),
            Self::InvalidProjection => {
                write!(f, "cognitive SVG requires a valid spatial projection")
            }
        }
    }
}

impl Error for CognitiveSvgError {}

/// Render a static, accessible 2D projection. The source projection is not
/// modified. SVG coordinates are a presentation transform, never semantic
/// model state. No animation or synthetic cognitive events are generated.
pub fn render_cognitive_projection_svg(
    projection: &CognitiveSpatialProjection,
    options: CognitiveSvgOptions,
) -> Result<String, CognitiveSvgError> {
    if !(320..=4096).contains(&options.width)
        || !(240..=4096).contains(&options.height)
        || !options.padding.is_finite()
        || !(16.0..=256.0).contains(&options.padding)
        || options.padding * 2.0 >= options.width as f32
        || options.padding * 2.0 >= options.height as f32
    {
        return Err(CognitiveSvgError::InvalidOptions);
    }
    let geometry_digest = projection
        .geometry_digest_hex()
        .map_err(|_| CognitiveSvgError::InvalidProjection)?;

    let mut anchors: Vec<_> = projection.anchors.iter().collect();
    anchors.sort_by(|left, right| left.node_id.cmp(&right.node_id));

    let (min_x, max_x, min_y, max_y) = if anchors.is_empty() {
        (-1.0f32, 1.0f32, -1.0f32, 1.0f32)
    } else {
        let min_x = anchors
            .iter()
            .map(|anchor| anchor.position.x)
            .fold(f32::INFINITY, f32::min);
        let max_x = anchors
            .iter()
            .map(|anchor| anchor.position.x)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = anchors
            .iter()
            .map(|anchor| anchor.position.y)
            .fold(f32::INFINITY, f32::min);
        let max_y = anchors
            .iter()
            .map(|anchor| anchor.position.y)
            .fold(f32::NEG_INFINITY, f32::max);
        (min_x, max_x, min_y, max_y)
    };

    let available_width = options.width as f32 - options.padding * 2.0;
    let available_height = options.height as f32 - options.padding * 2.0;
    let span_x = (max_x - min_x).max(1.0);
    let span_y = (max_y - min_y).max(1.0);
    let scale = (available_width / span_x)
        .min(available_height / span_y)
        .max(0.01);
    let content_width = span_x * scale;
    let content_height = span_y * scale;
    let offset_x = (options.width as f32 - content_width) * 0.5;
    let offset_y = (options.height as f32 - content_height) * 0.5;

    let mut positions = HashMap::with_capacity(anchors.len());
    for anchor in &anchors {
        let x = offset_x + (anchor.position.x - min_x) * scale;
        let y = offset_y + (anchor.position.y - min_y) * scale;
        positions.insert(anchor.node_id.as_str(), (x, y));
    }

    let has_synthetic = projection
        .anchors
        .iter()
        .any(|anchor| anchor.provenance == CognitiveProvenance::SyntheticFixture)
        || projection
            .links
            .iter()
            .any(|link| link.provenance == CognitiveProvenance::SyntheticFixture);

    let mut output = String::with_capacity(
        projection.anchors.len() * 320 + projection.links.len() * 260 + 1_200,
    );
    writeln!(
        &mut output,
        r#"<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mycelial cognitive topology" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        options.width, options.height, options.width, options.height
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut output,
        "<title>Mycelial Presence — Cognitive Instrument</title>"
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut output,
        "<desc>Static evidence-bearing graph. Node shape identifies kind; outline pattern identifies epistemic status. Selectable labels retain producer, event ID, sequence and provenance as data attributes.</desc>"
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut output,
        r#"<metadata data-session="{}" data-topology-digest="{}" data-geometry-digest="{}"/>"#,
        escape_xml(&projection.session_id),
        escape_xml(&projection.topology_digest),
        geometry_digest
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut output,
        r##"<rect x="0" y="0" width="100%" height="100%" fill="#071321"/>"##
    )
    .expect("writing to String cannot fail");

    if has_synthetic {
        writeln!(
            &mut output,
            r##"<g class="fixture-warning"><text x="{}" y="34" text-anchor="middle" fill="#ffd18a" font-family="sans-serif" font-size="15" letter-spacing="1.8">SYNTHETIC FIXTURE — NOT LIVE COGNITIVE TELEMETRY</text></g>"##,
            options.width / 2
        )
        .expect("writing to String cannot fail");
    }

    let mut links: Vec<_> = projection.links.iter().collect();
    links.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    for link in links {
        render_link(&mut output, link, &positions);
    }

    for anchor in anchors {
        let Some(&(x, y)) = positions.get(anchor.node_id.as_str()) else {
            continue;
        };
        render_anchor(&mut output, anchor, x, y, options.show_labels);
    }

    render_legend(&mut output, options.width, options.height, has_synthetic);
    output.push_str("</svg>\n");
    Ok(output)
}

/// Render the direct CognitiveMorphologyProjection as a static cubic-hypha SVG.
/// Unlike the overview graph view, each semantic edge is represented by the
/// exact Bézier curve produced by the morphology adapter.
pub fn render_cognitive_morphology_svg(
    projection: &CognitiveMorphologyProjection,
    options: CognitiveSvgOptions,
) -> Result<String, CognitiveSvgError> {
    if !(320..=4096).contains(&options.width)
        || !(240..=4096).contains(&options.height)
        || !options.padding.is_finite()
        || !(16.0..=256.0).contains(&options.padding)
        || options.padding * 2.0 >= options.width as f32
        || options.padding * 2.0 >= options.height as f32
    {
        return Err(CognitiveSvgError::InvalidOptions);
    }
    let geometry_digest = projection
        .geometry_digest_hex()
        .map_err(|_| CognitiveSvgError::InvalidProjection)?;

    let mut points = Vec::with_capacity(projection.nodes.len() + projection.hyphae.len() * 4);
    points.extend(projection.nodes.iter().map(|node| node.position));
    for hypha in &projection.hyphae {
        points.extend([hypha.start, hypha.control_a, hypha.control_b, hypha.end]);
    }
    let (min_x, max_x, min_y, max_y) = if points.is_empty() {
        (-1.0f32, 1.0f32, -1.0f32, 1.0f32)
    } else {
        (
            points
                .iter()
                .map(|point| point.x)
                .fold(f32::INFINITY, f32::min),
            points
                .iter()
                .map(|point| point.x)
                .fold(f32::NEG_INFINITY, f32::max),
            points
                .iter()
                .map(|point| point.y)
                .fold(f32::INFINITY, f32::min),
            points
                .iter()
                .map(|point| point.y)
                .fold(f32::NEG_INFINITY, f32::max),
        )
    };
    let available_width = options.width as f32 - options.padding * 2.0;
    let available_height = options.height as f32 - options.padding * 2.0;
    let span_x = (max_x - min_x).max(1.0);
    let span_y = (max_y - min_y).max(1.0);
    let scale = (available_width / span_x).min(available_height / span_y);
    let content_width = span_x * scale;
    let content_height = span_y * scale;
    let offset_x = (options.width as f32 - content_width) * 0.5;
    let offset_y = (options.height as f32 - content_height) * 0.5;
    let project_point = |point: crate::mycelial_model::SpatialPoint3| {
        (
            offset_x + (point.x - min_x) * scale,
            offset_y + (point.y - min_y) * scale,
        )
    };

    let has_synthetic = projection
        .nodes
        .iter()
        .any(|node| node.evidence.provenance == CognitiveProvenance::SyntheticFixture)
        || projection
            .hyphae
            .iter()
            .any(|hypha| hypha.evidence.provenance == CognitiveProvenance::SyntheticFixture);

    let mut output =
        String::with_capacity(projection.nodes.len() * 320 + projection.hyphae.len() * 520 + 1_400);
    writeln!(
        &mut output,
        r#"<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mycelial evidence morphology" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        options.width, options.height, options.width, options.height
    )
    .expect("writing to String cannot fail");
    output.push_str("<title>Mycelial Presence — Evidence Morphology</title>\n");
    output.push_str("<desc>Static cubic-hypha projection. Every curve corresponds to one validated semantic edge and retains its source event and epistemic status.</desc>\n");
    writeln!(
        &mut output,
        r#"<metadata data-session="{}" data-topology-digest="{}" data-geometry-digest="{}"/>"#,
        escape_xml(&projection.session_id),
        escape_xml(&projection.topology_digest),
        geometry_digest
    )
    .expect("writing to String cannot fail");
    output.push_str(r##"<rect x="0" y="0" width="100%" height="100%" fill="#071321"/>"##);
    output.push('\n');

    if has_synthetic {
        writeln!(
            &mut output,
            r##"<g class="fixture-warning"><text x="{}" y="34" text-anchor="middle" fill="#ffd18a" font-family="sans-serif" font-size="15">SYNTHETIC FIXTURE — NOT LIVE COGNITIVE TELEMETRY</text></g>"##,
            options.width / 2
        )
        .expect("writing to String cannot fail");
    }

    let mut hyphae: Vec<_> = projection.hyphae.iter().collect();
    hyphae.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    for hypha in hyphae {
        render_cognitive_hypha(&mut output, hypha, &project_point);
    }

    let mut nodes: Vec<_> = projection.nodes.iter().collect();
    nodes.sort_by(|left, right| left.node_id.cmp(&right.node_id));
    for node in nodes {
        let anchor = CognitiveAnchor {
            node_id: node.node_id.clone(),
            node_kind: node.kind,
            label: node.label.clone(),
            status: node.status,
            position: node.position,
            source_event_id: node.evidence.event_id.clone(),
            source_producer: node.evidence.producer.clone(),
            source_sequence: node.evidence.sequence,
            provenance: node.evidence.provenance,
        };
        let (x, y) = project_point(node.position);
        render_anchor(&mut output, &anchor, x, y, options.show_labels);
    }
    render_legend(&mut output, options.width, options.height, has_synthetic);
    output.push_str("</svg>\n");
    Ok(output)
}

fn render_cognitive_hypha(
    output: &mut String,
    hypha: &CognitiveHypha,
    project_point: &impl Fn(crate::mycelial_model::SpatialPoint3) -> (f32, f32),
) {
    let (sx, sy) = project_point(hypha.start);
    let (a_x, a_y) = project_point(hypha.control_a);
    let (b_x, b_y) = project_point(hypha.control_b);
    let (tx, ty) = project_point(hypha.end);
    let (color, width) = relation_style(hypha.relation);
    let opacity = status_opacity(hypha.status);
    let dash = status_dash(hypha.status);
    let title = format!(
        "{}: {:?} {} → {}; status={:?}; event={}; producer={}; sequence={}; provenance={:?}",
        hypha.edge_id,
        hypha.relation,
        hypha.source_node_id,
        hypha.target_node_id,
        hypha.status,
        hypha.evidence.event_id,
        hypha.evidence.producer,
        hypha.evidence.sequence,
        hypha.evidence.provenance
    );
    let _ = writeln!(
        output,
        r#"<g class="cognitive-hypha" data-edge-id="{}" data-source-node="{}" data-target-node="{}" data-event-id="{}" data-producer="{}" data-sequence="{}" data-provenance="{}" data-status="{:?}" data-relation="{:?}"><title>{}</title><path d="M {:.2} {:.2} C {:.2} {:.2}, {:.2} {:.2}, {:.2} {:.2}" fill="none" stroke="{}" stroke-width="{:.2}" stroke-linecap="round" stroke-opacity="{:.2}"{} /></g>"#,
        escape_xml(&hypha.edge_id),
        escape_xml(&hypha.source_node_id),
        escape_xml(&hypha.target_node_id),
        escape_xml(&hypha.evidence.event_id),
        escape_xml(&hypha.evidence.producer),
        hypha.evidence.sequence,
        provenance_name(hypha.evidence.provenance),
        hypha.status,
        hypha.relation,
        escape_xml(&title),
        sx,
        sy,
        a_x,
        a_y,
        b_x,
        b_y,
        tx,
        ty,
        color,
        width,
        opacity,
        dash
    );
}

fn relation_style(relation: CognitiveRelation) -> (&'static str, f32) {
    match relation {
        CognitiveRelation::Predicts => ("#ffc778", 2.4),
        CognitiveRelation::Contradicts => ("#ff827e", 2.6),
        CognitiveRelation::RetrievedFrom => ("#62d9ce", 2.8),
        CognitiveRelation::Supports => ("#8dc8ff", 2.3),
        CognitiveRelation::DependsOn => ("#a9b9d4", 2.0),
        CognitiveRelation::ExecutedBy => ("#d2a0f5", 2.2),
        CognitiveRelation::AssociatedWith => ("#79b6e8", 2.0),
        CognitiveRelation::CandidateRelation => ("#ffc778", 1.9),
    }
}

fn render_link(
    output: &mut String,
    link: &CognitiveSpatialLink,
    positions: &HashMap<&str, (f32, f32)>,
) {
    let (Some(&(sx, sy)), Some(&(tx, ty))) = (
        positions.get(link.source_node_id.as_str()),
        positions.get(link.target_node_id.as_str()),
    ) else {
        return;
    };

    let dx = tx - sx;
    let dy = ty - sy;
    let length = (dx * dx + dy * dy).sqrt().max(1.0);
    let bend = (length * 0.12).clamp(10.0, 42.0);
    let direction = if stable_sign(&link.edge_id) {
        1.0
    } else {
        -1.0
    };
    let control_x = (sx + tx) * 0.5 - (dy / length) * bend * direction;
    let control_y = (sy + ty) * 0.5 + (dx / length) * bend * direction;
    let (color, width) = match link.relation {
        CognitiveRelation::Predicts => ("#ffc778", 2.4),
        CognitiveRelation::Contradicts => ("#ff827e", 2.6),
        CognitiveRelation::RetrievedFrom => ("#62d9ce", 2.8),
        CognitiveRelation::Supports => ("#8dc8ff", 2.3),
        CognitiveRelation::DependsOn => ("#a9b9d4", 2.0),
        CognitiveRelation::ExecutedBy => ("#d2a0f5", 2.2),
        CognitiveRelation::AssociatedWith => ("#79b6e8", 2.0),
        CognitiveRelation::CandidateRelation => ("#ffc778", 1.9),
    };
    let dash = status_dash(link.status);
    let opacity = status_opacity(link.status);
    let title = format!(
        "{}: {} — {:?} → {}; status={:?}; event={}; sequence={}; provenance={:?}",
        link.edge_id,
        link.source_node_id,
        link.relation,
        link.target_node_id,
        link.status,
        link.source_event_id,
        link.source_sequence,
        link.provenance
    );

    let _ = writeln!(
        output,
        r#"<g class="relation" data-edge-id="{}" data-event-id="{}" data-sequence="{}" data-provenance="{}" data-status="{:?}" data-relation="{:?}"><title>{}</title><path d="M {:.2} {:.2} Q {:.2} {:.2} {:.2} {:.2}" fill="none" stroke="{}" stroke-width="{:.1}" stroke-linecap="round" stroke-opacity="{:.2}"{} /></g>"#,
        escape_xml(&link.edge_id),
        escape_xml(&link.source_event_id),
        link.source_sequence,
        provenance_name(link.provenance),
        link.status,
        link.relation,
        escape_xml(&title),
        sx,
        sy,
        control_x,
        control_y,
        tx,
        ty,
        color,
        width,
        opacity,
        dash
    );
}

fn render_anchor(
    output: &mut String,
    anchor: &crate::cognitive_layout::CognitiveAnchor,
    x: f32,
    y: f32,
    show_label: bool,
) {
    let z = anchor.position.z.clamp(-6.0, 6.0);
    let radius = (9.0 + z * 0.35).clamp(6.5, 11.5);
    let color = node_color(anchor.node_kind);
    let dash = status_dash(anchor.status);
    let opacity = status_opacity(anchor.status);
    let title = format!(
        "{} — {}; status={:?}; event={}; producer={}; sequence={}; provenance={:?}",
        anchor.node_id,
        anchor.label,
        anchor.status,
        anchor.source_event_id,
        anchor.source_producer,
        anchor.source_sequence,
        anchor.provenance
    );
    let label = escape_xml(&anchor.label);
    let node_id = escape_xml(&anchor.node_id);

    writeln!(
        output,
        r#"<g class="cognitive-node" role="group" tabindex="0" aria-label="{}" data-node-id="{}" data-event-id="{}" data-sequence="{}" data-provenance="{}" data-status="{:?}" data-kind="{:?}"><title>{}</title>"#,
        escape_xml(&title),
        node_id,
        escape_xml(&anchor.source_event_id),
        anchor.source_sequence,
        provenance_name(anchor.provenance),
        anchor.status,
        anchor.node_kind,
        escape_xml(&title)
    )
    .expect("writing to String cannot fail");

    match anchor.node_kind {
        CognitiveNodeKind::Memory => {
            let _ = writeln!(
                output,
                r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" rx="{:.2}" fill="{}" fill-opacity=".16" stroke="{}" stroke-width="2.2" stroke-opacity="{:.2}"{} />"#,
                x - radius * 1.15,
                y - radius * 0.75,
                radius * 2.3,
                radius * 1.5,
                radius * 0.35,
                color,
                color,
                opacity,
                dash
            );
        }
        CognitiveNodeKind::Prediction => {
            let _ = writeln!(
                output,
                r#"<polygon points="{:.2},{:.2} {:.2},{:.2} {:.2},{:.2} {:.2},{:.2}" fill="{}" fill-opacity=".18" stroke="{}" stroke-width="2.2" stroke-opacity="{:.2}"{} />"#,
                x,
                y - radius * 1.35,
                x + radius * 1.2,
                y,
                x,
                y + radius * 1.35,
                x - radius * 1.2,
                y,
                color,
                color,
                opacity,
                dash
            );
        }
        CognitiveNodeKind::Observation => {
            let points = polygon_points(x, y, radius, 3);
            let _ = writeln!(
                output,
                r#"<polygon points="{}" fill="{}" fill-opacity=".16" stroke="{}" stroke-width="2.2" stroke-opacity="{:.2}"{} />"#,
                points, color, color, opacity, dash
            );
        }
        CognitiveNodeKind::Tool => {
            let _ = writeln!(
                output,
                r#"<polygon points="{}" fill="{}" fill-opacity=".16" stroke="{}" stroke-width="2.2" stroke-opacity="{:.2}"{} />"#,
                polygon_points(x, y, radius, 6),
                color,
                color,
                opacity,
                dash
            );
        }
        CognitiveNodeKind::Task | CognitiveNodeKind::Plan => {
            let rx = if anchor.node_kind == CognitiveNodeKind::Plan {
                radius
            } else {
                radius * 0.15
            };
            let _ = writeln!(
                output,
                r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" rx="{:.2}" fill="{}" fill-opacity=".16" stroke="{}" stroke-width="2.2" stroke-opacity="{:.2}"{} />"#,
                x - radius,
                y - radius,
                radius * 2.0,
                radius * 2.0,
                rx,
                color,
                color,
                opacity,
                dash
            );
        }
        CognitiveNodeKind::Concept | CognitiveNodeKind::Entity => {
            let _ = writeln!(
                output,
                r#"<circle cx="{x:.2}" cy="{y:.2}" r="{radius:.2}" fill="{}" fill-opacity=".17" stroke="{}" stroke-width="2.2" stroke-opacity="{:.2}"{} />"#,
                color, color, opacity, dash
            );
        }
    }

    let status_marker = match anchor.status {
        EpistemicStatus::Observed => "•",
        EpistemicStatus::Derived => "◇",
        EpistemicStatus::Candidate => "?",
        EpistemicStatus::Resolved => "✓",
        EpistemicStatus::Rejected => "×",
        EpistemicStatus::Stale => "…",
        EpistemicStatus::Unavailable => "–",
    };
    let _ = writeln!(
        output,
        r##"<text x="{x:.2}" y="{:.2}" text-anchor="middle" dominant-baseline="central" fill="#edf7ff" font-size="10" font-family="sans-serif">{}</text>"##,
        y + 3.5,
        status_marker
    );

    if show_label {
        let short_label = truncate_label(&label, 36);
        let _ = writeln!(
            output,
            r##"<text x="{x:.2}" y="{:.2}" text-anchor="middle" fill="#d4e2f1" font-size="12" font-family="sans-serif">{short_label}</text>"##,
            y + radius + 18.0
        );
    }
    output.push_str("</g>\n");
}

fn render_legend(output: &mut String, width: u32, height: u32, has_synthetic: bool) {
    let legend_y = height as f32 - 24.0;
    let warning = if has_synthetic {
        "Fixture data is synthetic; no live cognitive activity is claimed."
    } else {
        "Every node and edge carries an inspectable source-event reference."
    };
    let _ = writeln!(
        output,
        r##"<g class="legend"><text x="24" y="{legend_y:.1}" fill="#91a9c1" font-family="sans-serif" font-size="11">{}</text><text x="{}" y="{legend_y:.1}" text-anchor="end" fill="#91a9c1" font-family="sans-serif" font-size="11">Outline: observed / derived / candidate / resolved / rejected / stale</text></g>"##,
        escape_xml(warning),
        width - 24
    );
}

fn node_color(kind: CognitiveNodeKind) -> &'static str {
    match kind {
        CognitiveNodeKind::Memory => "#40d4c9",
        CognitiveNodeKind::Concept => "#7fbfff",
        CognitiveNodeKind::Observation => "#8edaa9",
        CognitiveNodeKind::Task => "#c4a4ff",
        CognitiveNodeKind::Tool => "#f2b36e",
        CognitiveNodeKind::Plan => "#bd95ec",
        CognitiveNodeKind::Prediction => "#ffca7a",
        CognitiveNodeKind::Entity => "#b9c8d9",
    }
}

fn status_dash(status: EpistemicStatus) -> &'static str {
    match status {
        EpistemicStatus::Observed | EpistemicStatus::Resolved => "",
        EpistemicStatus::Derived => r#" stroke-dasharray="7 3""#,
        EpistemicStatus::Candidate => r#" stroke-dasharray="3 4""#,
        EpistemicStatus::Rejected => r#" stroke-dasharray="2 5""#,
        EpistemicStatus::Stale => r#" stroke-dasharray="1 4""#,
        EpistemicStatus::Unavailable => r#" stroke-dasharray="1 3""#,
    }
}

fn status_opacity(status: EpistemicStatus) -> f32 {
    match status {
        EpistemicStatus::Observed | EpistemicStatus::Resolved => 0.96,
        EpistemicStatus::Derived => 0.84,
        EpistemicStatus::Candidate => 0.9,
        EpistemicStatus::Rejected => 0.56,
        EpistemicStatus::Stale => 0.42,
        EpistemicStatus::Unavailable => 0.36,
    }
}

fn provenance_name(provenance: CognitiveProvenance) -> &'static str {
    match provenance {
        CognitiveProvenance::RuntimeEvent => "runtime_event",
        CognitiveProvenance::RuntimeDerived => "runtime_derived",
        CognitiveProvenance::SyntheticFixture => "synthetic_fixture",
    }
}

fn stable_sign(id: &str) -> bool {
    blake3::hash(id.as_bytes()).as_bytes()[0] & 1 == 0
}

fn polygon_points(x: f32, y: f32, radius: f32, sides: usize) -> String {
    let mut points = String::with_capacity(sides * 18);
    for index in 0..sides {
        let angle =
            -std::f32::consts::FRAC_PI_2 + index as f32 * std::f32::consts::TAU / sides as f32;
        let px = x + radius * angle.cos();
        let py = y + radius * angle.sin();
        let _ = write!(&mut points, "{px:.2},{py:.2} ");
    }
    points
}

fn truncate_label(label: &str, max_chars: usize) -> String {
    let mut chars = label.chars();
    let prefix: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_layout::{CognitiveLayoutConfig, layout_cognitive_topology};
    use crate::cognitive_topology::CognitiveTopology;

    fn fixture_projection() -> CognitiveSpatialProjection {
        let topology: CognitiveTopology =
            serde_json::from_str(include_str!("../tests/fixtures/cognitive_topology_v1.json"))
                .expect("checked-in cognitive topology fixture");
        layout_cognitive_topology(&topology, None, CognitiveLayoutConfig::default())
            .expect("valid topology layout")
    }

    #[test]
    fn static_svg_distinguishes_candidate_links_and_marks_synthetic_fixture() {
        let svg =
            render_cognitive_projection_svg(&fixture_projection(), CognitiveSvgOptions::default())
                .expect("valid SVG");
        assert!(svg.contains(r#"role="img""#));
        assert!(svg.contains("SYNTHETIC FIXTURE"));
        assert!(svg.contains(r#"data-provenance="synthetic_fixture""#));
        assert!(svg.contains(r#"data-status="Candidate""#));
        assert!(svg.contains(r#"class="relation""#));
        assert!(!svg.contains("<animate"));
        assert!(!svg.contains("@keyframes"));
    }

    #[test]
    fn morphology_svg_renders_cubic_hyphae_and_preserves_event_metadata() {
        let topology: crate::cognitive_topology::CognitiveTopology =
            serde_json::from_str(include_str!("../tests/fixtures/cognitive_topology_v1.json"))
                .expect("checked-in cognitive topology fixture");
        let spatial = crate::cognitive_layout::layout_cognitive_topology(
            &topology,
            None,
            crate::cognitive_layout::CognitiveLayoutConfig::default(),
        )
        .expect("valid spatial projection");
        let morphology = crate::cognitive_morphology::build_cognitive_morphology(
            &topology,
            &spatial,
            crate::cognitive_morphology::CognitiveMorphologyConfig::default(),
        )
        .expect("valid morphology");
        let svg = render_cognitive_morphology_svg(&morphology, CognitiveSvgOptions::default())
            .expect("morphology SVG");
        assert!(svg.contains(" C "));
        assert!(svg.contains(r#"class="cognitive-hypha""#));
        assert!(svg.contains(r#"data-provenance="synthetic_fixture""#));
        assert!(svg.contains(r#"data-status="Candidate""#));
        assert!(svg.contains("SYNTHETIC FIXTURE"));
        assert!(!svg.contains("<animate"));
    }

    #[test]
    fn labels_are_xml_escaped_and_full_text_remains_in_title() {
        let mut projection = fixture_projection();
        projection.anchors[0].label = "<script>& unsafe".to_string();
        let svg = render_cognitive_projection_svg(&projection, CognitiveSvgOptions::default())
            .expect("valid escaped SVG");
        assert!(svg.contains("&lt;script&gt;&amp; unsafe"));
        assert!(!svg.contains("<script>"));
    }

    #[test]
    fn malformed_geometry_and_options_are_rejected() {
        let mut projection = fixture_projection();
        projection.anchors[0].position.x = f32::NAN;
        assert_eq!(
            render_cognitive_projection_svg(&projection, CognitiveSvgOptions::default()),
            Err(CognitiveSvgError::InvalidProjection)
        );
        assert_eq!(
            render_cognitive_projection_svg(
                &fixture_projection(),
                CognitiveSvgOptions {
                    width: 100,
                    ..CognitiveSvgOptions::default()
                }
            ),
            Err(CognitiveSvgError::InvalidOptions)
        );
    }
}
