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
        }
    }

    pub fn node_pos(&self, id: &str) -> Option<(f64, f64)> {
        self.nodes.iter().find(|n| n.id == id).map(|n| (n.x, n.y))
    }
}

pub type SharedVizState = Rc<RefCell<VisualizationState>>;

pub fn new_shared_viz_state(width: f64, height: f64) -> SharedVizState {
    Rc::new(RefCell::new(VisualizationState::new(width, height)))
}
