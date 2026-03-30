// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Force-directed graph visualization — Leptos SVG replacement for D3.js force layout.
//!
//! Uses Fruchterman-Reingold-style forces:
//! - Repulsion between all nodes (Coulomb's law)
//! - Attraction along edges (Hooke's law)
//! - Center gravity (prevents drift)
//! - Velocity damping (stabilization)
//!
//! Layout runs in Rust/WASM (faster than D3's JS force simulation),
//! rendering is reactive SVG via Leptos.

use leptos::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// A node in the force-directed graph.
#[derive(Clone, Debug)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub color: String,
    /// Visual size multiplier (1.0 = default)
    pub size: f64,
    /// Optional group for clustering
    pub group: Option<String>,
}

/// An edge between two nodes.
#[derive(Clone, Debug)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub weight: f64,
    pub color: Option<String>,
}

// Internal physics state (pub(crate) for testing)
#[derive(Clone)]
pub(crate) struct PhysicsNode {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
}

pub(crate) fn force_step(
    nodes: &mut [PhysicsNode],
    edges: &[(usize, usize, f64)],
    cx: f64,
    cy: f64,
    repulsion: f64,
    attraction: f64,
) {
    let n = nodes.len();
    let damping = 0.85;
    let center_pull = 0.003;
    let mut fx = vec![0.0_f64; n];
    let mut fy = vec![0.0_f64; n];

    // Repulsion (all pairs)
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = nodes[i].x - nodes[j].x;
            let dy = nodes[i].y - nodes[j].y;
            let dist_sq = dx * dx + dy * dy + 1.0;
            let force = repulsion / dist_sq;
            let dist = dist_sq.sqrt();
            let fdx = force * dx / dist;
            let fdy = force * dy / dist;
            fx[i] += fdx;
            fy[i] += fdy;
            fx[j] -= fdx;
            fy[j] -= fdy;
        }
    }

    // Attraction (edges)
    for &(from, to, weight) in edges {
        let dx = nodes[to].x - nodes[from].x;
        let dy = nodes[to].y - nodes[from].y;
        let dist = (dx * dx + dy * dy).sqrt().max(1.0);
        let force = attraction * dist * weight;
        let fdx = force * dx / dist;
        let fdy = force * dy / dist;
        fx[from] += fdx;
        fy[from] += fdy;
        fx[to] -= fdx;
        fy[to] -= fdy;
    }

    // Center gravity
    for i in 0..n {
        fx[i] += (cx - nodes[i].x) * center_pull;
        fy[i] += (cy - nodes[i].y) * center_pull;
    }

    // Apply forces
    for i in 0..n {
        nodes[i].vx = (nodes[i].vx + fx[i]) * damping;
        nodes[i].vy = (nodes[i].vy + fy[i]) * damping;
        nodes[i].x += nodes[i].vx;
        nodes[i].y += nodes[i].vy;
    }
}

/// Reactive force-directed graph visualization.
///
/// Runs physics simulation in Rust/WASM via requestAnimationFrame,
/// renders nodes and edges as SVG elements.
///
/// # Props
/// - `nodes` — Graph nodes with id, label, color, size
/// - `edges` — Edges between node IDs with weight
/// - `width`/`height` — SVG dimensions
/// - `repulsion` — Repulsion force strength (default 3000)
/// - `attraction` — Attraction force strength (default 0.004)
#[component]
pub fn ForceGraph(
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    #[prop(default = 500.0)] width: f64,
    #[prop(default = 400.0)] height: f64,
    #[prop(default = 3000.0)] repulsion: f64,
    #[prop(default = 0.004)] attraction: f64,
) -> impl IntoView {
    let cx = width / 2.0;
    let cy = height / 2.0;
    let radius = height.min(width) * 0.35;

    // Build edge index (source_idx, target_idx, weight)
    let node_ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
    let edge_indices: Vec<(usize, usize, f64)> = edges.iter().filter_map(|e| {
        let from = node_ids.iter().position(|id| id == &e.source)?;
        let to = node_ids.iter().position(|id| id == &e.target)?;
        Some((from, to, e.weight))
    }).collect();

    // Initialize physics nodes in a circle
    let n = nodes.len();
    let physics: Rc<RefCell<Vec<PhysicsNode>>> = Rc::new(RefCell::new(
        (0..n).map(|i| {
            let angle = (i as f64 / n.max(1) as f64) * std::f64::consts::TAU;
            PhysicsNode {
                x: cx + angle.cos() * radius,
                y: cy + angle.sin() * radius,
                vx: 0.0,
                vy: 0.0,
            }
        }).collect()
    ));

    // Pre-settle (80 iterations)
    {
        let mut phys = physics.borrow_mut();
        for _ in 0..80 {
            force_step(&mut phys, &edge_indices, cx, cy, repulsion, attraction);
        }
    }

    // Positions signal — updated each frame
    let positions: RwSignal<Vec<(f64, f64)>> = RwSignal::new(
        physics.borrow().iter().map(|p| (p.x, p.y)).collect()
    );

    // Animation loop
    let physics_for_loop = physics.clone();
    let edge_indices_rc = Rc::new(edge_indices);
    Effect::new(move |_| {
        let physics = physics_for_loop.clone();
        let edges = edge_indices_rc.clone();

        let frame_ref: Rc<RefCell<Option<Closure<dyn FnMut()>>>> =
            Rc::new(RefCell::new(None));
        let frame_ref_clone = frame_ref.clone();

        let closure = Closure::wrap(Box::new(move || {
            let mut phys = physics.borrow_mut();
            force_step(&mut phys, &edges, cx, cy, repulsion, attraction);
            positions.set(phys.iter().map(|p| (p.x, p.y)).collect());

            if let Some(window) = web_sys::window() {
                if let Some(ref cb) = *frame_ref_clone.borrow() {
                    let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
                }
            }
        }) as Box<dyn FnMut()>);

        if let Some(window) = web_sys::window() {
            let _ = window.request_animation_frame(closure.as_ref().unchecked_ref());
        }

        *frame_ref.borrow_mut() = Some(closure);
    });

    // Edge data for rendering (capture colors)
    let edge_render_data: Vec<(usize, usize, String)> = edges.iter().filter_map(|e| {
        let from = node_ids.iter().position(|id| id == &e.source)?;
        let to = node_ids.iter().position(|id| id == &e.target)?;
        let color = e.color.clone().unwrap_or_else(|| "rgba(126, 200, 160, 0.25)".into());
        Some((from, to, color))
    }).collect();

    view! {
        <svg
            viewBox=format!("0 0 {width} {height}")
            style="width: 100%; height: auto;"
            xmlns="http://www.w3.org/2000/svg"
        >
            // Edges
            {edge_render_data.into_iter().map(|(from, to, color)| {
                view! {
                    <line
                        x1=move || positions.get().get(from).map(|p| p.0).unwrap_or(0.0)
                        y1=move || positions.get().get(from).map(|p| p.1).unwrap_or(0.0)
                        x2=move || positions.get().get(to).map(|p| p.0).unwrap_or(0.0)
                        y2=move || positions.get().get(to).map(|p| p.1).unwrap_or(0.0)
                        stroke=color
                        stroke-width="1"
                    />
                }
            }).collect::<Vec<_>>()}

            // Nodes
            {nodes.into_iter().enumerate().map(|(i, node)| {
                let r = 6.0 * node.size;
                let color = node.color.clone();
                let label = node.label.clone();
                view! {
                    <circle
                        cx=move || positions.get().get(i).map(|p| p.0).unwrap_or(0.0)
                        cy=move || positions.get().get(i).map(|p| p.1).unwrap_or(0.0)
                        r=r
                        fill=color.clone()
                        opacity="0.85"
                    />
                    <text
                        x=move || positions.get().get(i).map(|p| p.0).unwrap_or(0.0)
                        y=move || positions.get().get(i).map(|p| p.1 + r + 12.0).unwrap_or(0.0)
                        text-anchor="middle"
                        fill="rgba(126, 200, 160, 0.7)"
                        font-size="9"
                        font-family="monospace"
                    >
                        {label}
                    </text>
                }
            }).collect::<Vec<_>>()}
        </svg>
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(x: f64, y: f64) -> PhysicsNode {
        PhysicsNode { x, y, vx: 0.0, vy: 0.0 }
    }

    #[test]
    fn two_connected_nodes_converge() {
        let mut nodes = vec![node(0.0, 0.0), node(100.0, 0.0)];
        let edges = vec![(0, 1, 1.0)];
        for _ in 0..200 {
            force_step(&mut nodes, &edges, 50.0, 0.0, 3000.0, 0.004);
        }
        // After 200 iterations, nodes should be near equilibrium
        let dist = ((nodes[0].x - nodes[1].x).powi(2) + (nodes[0].y - nodes[1].y).powi(2)).sqrt();
        assert!(dist > 10.0, "nodes too close: {dist}");
        assert!(dist < 200.0, "nodes too far: {dist}");
    }

    #[test]
    fn triangle_stabilizes() {
        let mut nodes = vec![node(0.0, 0.0), node(100.0, 0.0), node(50.0, 87.0)];
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        for _ in 0..200 {
            force_step(&mut nodes, &edges, 50.0, 30.0, 3000.0, 0.004);
        }
        // All velocities should be near zero (stable)
        for n in &nodes {
            assert!(n.vx.abs() < 1.0, "vx not stable: {}", n.vx);
            assert!(n.vy.abs() < 1.0, "vy not stable: {}", n.vy);
        }
    }

    #[test]
    fn single_node_stays_centered() {
        let mut nodes = vec![node(50.0, 50.0)];
        let edges = vec![];
        for _ in 0..100 {
            force_step(&mut nodes, &edges, 50.0, 50.0, 3000.0, 0.004);
        }
        assert!((nodes[0].x - 50.0).abs() < 5.0);
        assert!((nodes[0].y - 50.0).abs() < 5.0);
    }

    #[test]
    fn nodes_stay_bounded() {
        let mut nodes = vec![node(0.0, 0.0), node(1000.0, 1000.0), node(-500.0, 500.0)];
        let edges = vec![(0, 1, 0.5), (1, 2, 0.5)];
        for _ in 0..500 {
            force_step(&mut nodes, &edges, 250.0, 250.0, 3000.0, 0.004);
        }
        for n in &nodes {
            assert!(n.x.abs() < 2000.0, "x out of bounds: {}", n.x);
            assert!(n.y.abs() < 2000.0, "y out of bounds: {}", n.y);
        }
    }
}
