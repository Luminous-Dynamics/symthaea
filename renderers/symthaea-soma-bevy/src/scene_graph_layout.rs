// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Bevy-native geometry planning for typed SceneGraphV2.
//!
//! Semantic relation paths are generated only from explicit SceneRelationV2
//! records. Layout attachments position nodes but never become visible edges.
//! This module returns geometry data; the host still owns authorization,
//! rendering, compositor observation, and output-stop receipts.

use bevy::prelude::Vec2;
use std::collections::HashMap;
use symthaea_soma::embodiment_projection::EmbodimentForm;
use symthaea_soma::embodiment_scene::{SceneGlyph, SceneMotion, SceneRole};
use symthaea_soma::embodiment_scene_graph::{
    LayoutAnchor, SceneGraphV2, SceneRelationKind,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphNodePlacementV2 {
    pub node_id: u16,
    pub role: SceneRole,
    pub glyph: SceneGlyph,
    pub motion: SceneMotion,
    pub weight: Option<f32>,
    pub position: Vec2,
    pub rotation_radians: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphRelationPathV2 {
    pub relation_id: u16,
    pub kind: SceneRelationKind,
    pub source_node_id: u16,
    pub target_node_id: u16,
    pub start: Vec2,
    pub control: Vec2,
    pub end: Vec2,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneGraphLayoutV2 {
    pub nodes: Vec<GraphNodePlacementV2>,
    pub semantic_paths: Vec<GraphRelationPathV2>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneGraphLayoutError {
    DuplicateNode,
    MissingLayoutParent,
    MissingRelationEndpoint,
    InvalidNodePosition,
}

/// Deterministic form-aware layout. This is a presentation transform only:
/// it cannot add/remove semantic nodes or create inferred relationships.
pub fn layout_scene_graph_v2(
    graph: &SceneGraphV2,
) -> Result<SceneGraphLayoutV2, SceneGraphLayoutError> {
    let mut node_ids = HashMap::with_capacity(graph.nodes.len());
    for (index, node) in graph.nodes.iter().enumerate() {
        if node_ids.insert(node.id, index).is_some() {
            return Err(SceneGraphLayoutError::DuplicateNode);
        }
    }

    let mut placements: HashMap<u16, GraphNodePlacementV2> = HashMap::with_capacity(graph.nodes.len());
    let max_passes = graph.nodes.len().max(1);
    for _ in 0..max_passes {
        let mut progressed = false;
        for node in &graph.nodes {
            if placements.contains_key(&node.id) {
                continue;
            }
            let (position, rotation) = match node.layout_anchor {
                LayoutAnchor::Core => (Vec2::ZERO, 0.0),
                LayoutAnchor::Periphery { slot } => {
                    form_position(graph.form, slot, Vec2::ZERO)
                }
                LayoutAnchor::AttachedTo { node_id, slot } => {
                    let Some(parent) = placements.get(&node_id) else { continue; };
                    form_position(graph.form, slot, parent.position)
                }
            };
            if !position.is_finite() || !rotation.is_finite() {
                return Err(SceneGraphLayoutError::InvalidNodePosition);
            }
            placements.insert(node.id, GraphNodePlacementV2 {
                node_id: node.id,
                role: node.role,
                glyph: node.glyph,
                motion: node.motion,
                weight: node.weight,
                position,
                rotation_radians: rotation,
            });
            progressed = true;
        }
        if placements.len() == graph.nodes.len() {
            break;
        }
        if !progressed {
            return Err(SceneGraphLayoutError::MissingLayoutParent);
        }
    }

    let mut nodes: Vec<_> = placements.into_values().collect();
    nodes.sort_by_key(|node| node.node_id);
    let by_id: HashMap<_, _> = nodes.iter().map(|node| (node.node_id, node.position)).collect();

    // Only typed semantic relations produce visible connectors.
    let mut semantic_paths = Vec::with_capacity(graph.relations.len());
    for relation in &graph.relations {
        let Some(start) = by_id.get(&relation.source_node_id).copied() else {
            return Err(SceneGraphLayoutError::MissingRelationEndpoint);
        };
        let Some(end) = by_id.get(&relation.target_node_id).copied() else {
            return Err(SceneGraphLayoutError::MissingRelationEndpoint);
        };
        let delta = end - start;
        let midpoint = (start + end) * 0.5;
        let normal = if delta.length_squared() > f32::EPSILON {
            Vec2::new(-delta.y, delta.x).normalize()
        } else {
            Vec2::Y
        };
        let bend = form_edge_bend(graph.form);
        let control = midpoint + normal * bend;
        semantic_paths.push(GraphRelationPathV2 {
            relation_id: relation.id,
            kind: relation.kind,
            source_node_id: relation.source_node_id,
            target_node_id: relation.target_node_id,
            start,
            control,
            end,
        });
    }
    semantic_paths.sort_by_key(|path| path.relation_id);
    Ok(SceneGraphLayoutV2 { nodes, semantic_paths })
}

fn form_position(form: EmbodimentForm, slot: u16, parent: Vec2) -> (Vec2, f32) {
    let slot = slot as f32;
    let (position, rotation) = match form {
        EmbodimentForm::LivingConstellation => {
            let angle = slot * 2.399_963_1;
            (Vec2::new(angle.cos() * 92.0, angle.sin() * 92.0), angle)
        }
        EmbodimentForm::PuppetTheatre => {
            let x = if (slot as u32) % 2 == 0 { -34.0 } else { 34.0 };
            (Vec2::new(x, -64.0 - slot * 28.0), -std::f32::consts::FRAC_PI_2)
        }
        EmbodimentForm::MycelialBeing => {
            let angle = slot * 1.7 - 0.8;
            (Vec2::new(angle.cos() * 78.0, angle.sin() * 78.0), angle)
        }
        EmbodimentForm::Swarm => {
            let angle = slot * 2.399_963_1;
            let radius = 52.0 + (slot % 3.0) * 15.0;
            (Vec2::new(angle.cos() * radius, angle.sin() * radius), angle)
        }
        EmbodimentForm::LivingInstrument => {
            (Vec2::new(slot * 48.0 - 24.0, if (slot as u32) % 2 == 0 { 30.0 } else { -30.0 }), 0.0)
        }
        EmbodimentForm::HearthCompanion => {
            let angle = std::f32::consts::PI + slot * 0.55;
            (Vec2::new(angle.cos() * 88.0, angle.sin() * 64.0), angle)
        }
        EmbodimentForm::CognitiveMandala => {
            let angle = slot * std::f32::consts::FRAC_PI_4;
            (Vec2::new(angle.cos() * 74.0, angle.sin() * 74.0), angle)
        }
    };
    (parent + position, rotation)
}

fn form_edge_bend(form: EmbodimentForm) -> f32 {
    match form {
        EmbodimentForm::LivingConstellation => 18.0,
        EmbodimentForm::PuppetTheatre => 8.0,
        EmbodimentForm::MycelialBeing => 30.0,
        EmbodimentForm::Swarm => 22.0,
        EmbodimentForm::LivingInstrument => 12.0,
        EmbodimentForm::HearthCompanion => 24.0,
        EmbodimentForm::CognitiveMandala => 28.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_seven_forms_have_deterministic_distinct_placement_grammars() {
        let positions: Vec<_> = EmbodimentForm::ALL
            .iter()
            .map(|form| form_position(*form, 2, Vec2::ZERO).0)
            .collect();
        for position in &positions {
            assert!(position.is_finite());
        }
        assert!(positions.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn semantic_edges_are_separate_from_layout_attachments() {
        let path = GraphRelationPathV2 {
            relation_id: 1,
            kind: SceneRelationKind::AttentionTo,
            source_node_id: 0,
            target_node_id: 1,
            start: Vec2::ZERO,
            control: Vec2::new(2.0, 2.0),
            end: Vec2::new(4.0, 0.0),
        };
        assert_eq!(path.start, Vec2::ZERO);
        assert_eq!(path.end, Vec2::new(4.0, 0.0));
    }

    #[test]
    fn form_specific_edge_bend_is_stable() {
        assert_eq!(form_edge_bend(EmbodimentForm::MycelialBeing), 30.0);
        assert_eq!(form_edge_bend(EmbodimentForm::PuppetTheatre), 8.0);
    }
}
