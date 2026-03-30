// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared visualization state — the bridge between Leptos signals and
//! the Canvas2D renderer (or future Bevy ECS).

use std::cell::RefCell;
use std::rc::Rc;

/// A node in the kinship web.
#[derive(Clone, Debug)]
pub struct VisNode {
    pub id: String,
    pub label: String,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub radius: f64,
    pub color: [f32; 3],
    pub is_guardian: bool,
    pub presence_home: bool,
}

/// An edge (bond) in the kinship web.
#[derive(Clone, Debug)]
pub struct VisEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub strength_bp: u32,
    pub bond_type_label: String,
    /// Breathing phase (0..2*PI), oscillates per frame.
    pub breath_phase: f64,
}

/// A gratitude particle traveling along an edge.
#[derive(Clone, Debug)]
pub struct VisParticle {
    pub from: String,
    pub to: String,
    /// Progress along the edge (0.0 = source, 1.0 = target).
    pub t: f64,
    /// Remaining lifetime in seconds.
    pub ttl: f64,
}

/// Theme colors for the canvas renderer.
#[derive(Clone, Debug)]
pub struct CanvasTheme {
    pub bg: (f64, f64, f64),
    pub primary: (f64, f64, f64),
    pub text: (f64, f64, f64),
    pub glow: (f64, f64, f64),
}

impl Default for CanvasTheme {
    fn default() -> Self {
        // Ember
        Self {
            bg: (10.0 / 255.0, 10.0 / 255.0, 8.0 / 255.0),
            primary: (212.0 / 255.0, 165.0 / 255.0, 116.0 / 255.0),
            text: (232.0 / 255.0, 224.0 / 255.0, 212.0 / 255.0),
            glow: (251.0 / 255.0, 191.0 / 255.0, 36.0 / 255.0),
        }
    }
}

/// The full visualization state, shared between Leptos and the renderer.
#[derive(Clone, Debug)]
pub struct VisualizationState {
    pub nodes: Vec<VisNode>,
    pub edges: Vec<VisEdge>,
    pub particles: Vec<VisParticle>,
    pub homeostasis: bool,
    pub torpor: f64,
    pub canvas_width: f64,
    pub canvas_height: f64,
    pub theme: CanvasTheme,
    /// Edge ID that was just tended (flashes bright, then clears).
    pub flash_edge: Option<String>,
    /// Remaining flash duration in seconds.
    pub flash_ttl: f64,
    /// Edge ID that should be tended (set by click handler, consumed by sync).
    pub pending_tend: Option<String>,
}

impl VisualizationState {
    pub fn new(width: f64, height: f64) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            particles: Vec::new(),
            homeostasis: false,
            torpor: 0.0,
            canvas_width: width,
            canvas_height: height,
            theme: CanvasTheme::default(),
            flash_edge: None,
            flash_ttl: 0.0,
            pending_tend: None,
        }
    }

    pub fn node_pos(&self, id: &str) -> Option<(f64, f64)> {
        self.nodes.iter().find(|n| n.id == id).map(|n| (n.x, n.y))
    }
}

impl VisualizationState {
    /// Find the nearest edge to a click point. Returns edge ID if within threshold.
    pub fn edge_at_point(&self, px: f64, py: f64, threshold: f64) -> Option<String> {
        let mut best: Option<(f64, String)> = None;

        for edge in &self.edges {
            let src = self.node_pos(&edge.source);
            let tgt = self.node_pos(&edge.target);
            if let (Some((x1, y1)), Some((x2, y2))) = (src, tgt) {
                let dist = point_to_segment_dist(px, py, x1, y1, x2, y2);
                if dist < threshold {
                    if best.as_ref().map_or(true, |(d, _)| dist < *d) {
                        best = Some((dist, edge.id.clone()));
                    }
                }
            }
        }

        best.map(|(_, id)| id)
    }
}

/// Distance from point (px, py) to line segment (x1,y1)-(x2,y2).
fn point_to_segment_dist(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 0.001 {
        return ((px - x1).powi(2) + (py - y1).powi(2)).sqrt();
    }
    let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let cx = x1 + t * dx;
    let cy = y1 + t * dy;
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

pub type SharedVizState = Rc<RefCell<VisualizationState>>;

pub fn new_shared_viz_state(width: f64, height: f64) -> SharedVizState {
    Rc::new(RefCell::new(VisualizationState::new(width, height)))
}
