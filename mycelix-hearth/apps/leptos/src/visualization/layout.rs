// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Force-directed layout for the kinship web.
//!
//! Simple spring-charge simulation: nodes repel each other,
//! edges act as springs pulling connected nodes together.

use super::shared_state::VisualizationState;

const REPULSION: f64 = 8000.0;
const SPRING_K: f64 = 0.005;
const SPRING_REST: f64 = 120.0;
const DAMPING: f64 = 0.85;
const CENTER_GRAVITY: f64 = 0.01;
const MIN_DIST: f64 = 20.0;

/// Run one step of the force-directed layout.
/// Call this each animation frame (~60Hz).
pub fn step_layout(state: &mut VisualizationState, dt: f64) {
    let n = state.nodes.len();
    if n == 0 { return; }

    let cx = state.canvas_width / 2.0;
    let cy = state.canvas_height / 2.0;

    // Accumulate forces
    let mut fx = vec![0.0_f64; n];
    let mut fy = vec![0.0_f64; n];

    // Node-node repulsion (Coulomb)
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = state.nodes[i].x - state.nodes[j].x;
            let dy = state.nodes[i].y - state.nodes[j].y;
            let dist = (dx * dx + dy * dy).sqrt().max(MIN_DIST);
            let force = REPULSION / (dist * dist);
            let ux = dx / dist;
            let uy = dy / dist;
            fx[i] += ux * force;
            fy[i] += uy * force;
            fx[j] -= ux * force;
            fy[j] -= uy * force;
        }
    }

    // Edge springs (Hooke)
    for edge in &state.edges {
        let si = state.nodes.iter().position(|n| n.id == edge.source);
        let ti = state.nodes.iter().position(|n| n.id == edge.target);
        if let (Some(si), Some(ti)) = (si, ti) {
            let dx = state.nodes[ti].x - state.nodes[si].x;
            let dy = state.nodes[ti].y - state.nodes[si].y;
            let dist = (dx * dx + dy * dy).sqrt().max(MIN_DIST);
            // Stronger bonds = shorter rest length
            let bond_factor = edge.strength_bp as f64 / 10000.0;
            let rest = SPRING_REST * (1.5 - bond_factor);
            let displacement = dist - rest;
            let force = SPRING_K * displacement;
            let ux = dx / dist;
            let uy = dy / dist;
            fx[si] += ux * force;
            fy[si] += uy * force;
            fx[ti] -= ux * force;
            fy[ti] -= uy * force;
        }
    }

    // Center gravity
    for i in 0..n {
        let dx = cx - state.nodes[i].x;
        let dy = cy - state.nodes[i].y;
        fx[i] += dx * CENTER_GRAVITY;
        fy[i] += dy * CENTER_GRAVITY;
    }

    // Integrate
    for i in 0..n {
        state.nodes[i].vx = (state.nodes[i].vx + fx[i] * dt) * DAMPING;
        state.nodes[i].vy = (state.nodes[i].vy + fy[i] * dt) * DAMPING;
        state.nodes[i].x += state.nodes[i].vx * dt;
        state.nodes[i].y += state.nodes[i].vy * dt;

        // Keep in bounds with padding
        let pad = 40.0;
        state.nodes[i].x = state.nodes[i].x.clamp(pad, state.canvas_width - pad);
        state.nodes[i].y = state.nodes[i].y.clamp(pad, state.canvas_height - pad);
    }
}

/// Advance bond breathing phases.
pub fn step_breathing(state: &mut VisualizationState, dt: f64) {
    for edge in &mut state.edges {
        let health = edge.strength_bp as f64 / 10000.0;
        // Healthy bonds breathe faster (period 2s), neglected slower (period 6s)
        let period = 2.0 + (1.0 - health) * 4.0;
        let speed = std::f64::consts::TAU / period;
        edge.breath_phase = (edge.breath_phase + speed * dt) % std::f64::consts::TAU;
    }
}

/// Advance gratitude particles along their edges.
pub fn step_particles(state: &mut VisualizationState, dt: f64) {
    for particle in &mut state.particles {
        particle.t += dt * 0.5; // travel speed
        particle.ttl -= dt;
    }
    state.particles.retain(|p| p.ttl > 0.0 && p.t <= 1.0);
}
