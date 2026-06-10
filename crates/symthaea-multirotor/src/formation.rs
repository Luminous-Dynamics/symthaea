// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-agent formation flight: pure geometry + coordination types.
//!
//! Defines formation shapes, setpoint computation, and neighbor state tracking.
//! The actual AsyncMind wiring lives in the example — this module is dependency-free.

use serde::{Deserialize, Serialize};

/// A formation shape for N agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormationShape {
    /// Agents in a line along the X axis with given spacing (meters).
    Line { spacing: f64 },
    /// Agents equally spaced on a circle of given radius (meters).
    Circle { radius: f64 },
    /// V-formation (like geese) with given angle (radians) and arm spacing.
    Vee { angle: f64, spacing: f64 },
    /// Custom positions relative to formation center.
    Custom(Vec<[f64; 3]>),
}

/// A setpoint for one agent in the formation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormationSetpoint {
    /// Target position relative to formation center.
    pub offset: [f64; 3],
    /// Agent index.
    pub agent_id: usize,
}

/// State shared between formation agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormationState {
    /// Agent identifier.
    pub agent_id: usize,
    /// Encoded state as float vector (for HDC social relay).
    pub encoded_state: Vec<f32>,
    /// Current world-frame position.
    pub position: [f64; 3],
    /// Timestamp of this state.
    pub timestamp: f64,
}

/// Per-agent formation controller.
///
/// Tracks neighbor states and computes adjusted setpoints to maintain formation.
pub struct FormationController {
    agent_id: usize,
    shape: FormationShape,
    n_agents: usize,
    neighbor_states: Vec<Option<FormationState>>,
    formation_center: [f64; 3],
}

impl FormationController {
    /// Create a new formation controller.
    pub fn new(agent_id: usize, shape: FormationShape, n_agents: usize) -> Self {
        Self {
            agent_id,
            shape,
            n_agents,
            neighbor_states: vec![None; n_agents],
            formation_center: [0.0, 0.0, 0.1],
        }
    }

    /// Update a neighbor's state.
    pub fn update_neighbor(&mut self, state: FormationState) {
        let id = state.agent_id;
        if id < self.n_agents {
            self.neighbor_states[id] = Some(state);
        }
    }

    /// Set the formation center position.
    pub fn set_center(&mut self, center: [f64; 3]) {
        self.formation_center = center;
    }

    /// Compute the setpoint for this agent.
    pub fn compute_setpoint(&self) -> FormationSetpoint {
        let setpoints = self.shape.compute_setpoints(self.n_agents);
        if self.agent_id < setpoints.len() {
            let offset = setpoints[self.agent_id];
            FormationSetpoint {
                offset: [
                    self.formation_center[0] + offset[0],
                    self.formation_center[1] + offset[1],
                    self.formation_center[2] + offset[2],
                ],
                agent_id: self.agent_id,
            }
        } else {
            FormationSetpoint {
                offset: self.formation_center,
                agent_id: self.agent_id,
            }
        }
    }

    /// Compute the formation error for this agent (distance from setpoint).
    pub fn formation_error(&self, current_position: &[f64; 3]) -> f64 {
        let sp = self.compute_setpoint();
        let dx = current_position[0] - sp.offset[0];
        let dy = current_position[1] - sp.offset[1];
        let dz = current_position[2] - sp.offset[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl FormationShape {
    /// Compute offset positions for N agents in this formation.
    ///
    /// Returns offsets relative to the formation center.
    pub fn compute_setpoints(&self, n_agents: usize) -> Vec<[f64; 3]> {
        match self {
            FormationShape::Line { spacing } => {
                let half = (n_agents as f64 - 1.0) * spacing / 2.0;
                (0..n_agents)
                    .map(|i| [i as f64 * spacing - half, 0.0, 0.0])
                    .collect()
            }
            FormationShape::Circle { radius } => (0..n_agents)
                .map(|i| {
                    let angle = 2.0 * std::f64::consts::PI * i as f64 / n_agents as f64;
                    [radius * angle.cos(), radius * angle.sin(), 0.0]
                })
                .collect(),
            FormationShape::Vee { angle, spacing } => {
                let mut setpoints = vec![[0.0, 0.0, 0.0]]; // Leader at center
                for i in 1..n_agents {
                    let side = if i % 2 == 1 { 1.0 } else { -1.0 };
                    let arm_idx = i.div_ceil(2) as f64;
                    let x = -arm_idx * spacing * angle.cos();
                    let y = side * arm_idx * spacing * angle.sin();
                    setpoints.push([x, y, 0.0]);
                }
                setpoints
            }
            FormationShape::Custom(positions) => {
                let mut result = positions.clone();
                // Pad with origin if fewer positions than agents
                while result.len() < n_agents {
                    result.push([0.0, 0.0, 0.0]);
                }
                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_geometry() {
        let shape = FormationShape::Line { spacing: 1.0 };
        let setpoints = shape.compute_setpoints(3);
        assert_eq!(setpoints.len(), 3);
        // Center agent at origin, others at ±1.0
        assert!((setpoints[0][0] - (-1.0)).abs() < 1e-10);
        assert!((setpoints[1][0] - 0.0).abs() < 1e-10);
        assert!((setpoints[2][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_circle_geometry() {
        let shape = FormationShape::Circle { radius: 1.0 };
        let setpoints = shape.compute_setpoints(4);
        assert_eq!(setpoints.len(), 4);
        // All at distance 1.0 from origin
        for sp in &setpoints {
            let dist = (sp[0] * sp[0] + sp[1] * sp[1]).sqrt();
            assert!(
                (dist - 1.0).abs() < 1e-10,
                "Agent at distance {dist} from center"
            );
        }
    }

    #[test]
    fn test_vee_geometry() {
        let shape = FormationShape::Vee {
            angle: std::f64::consts::FRAC_PI_4,
            spacing: 1.0,
        };
        let setpoints = shape.compute_setpoints(5);
        assert_eq!(setpoints.len(), 5);
        // Leader at origin
        assert!((setpoints[0][0]).abs() < 1e-10);
        assert!((setpoints[0][1]).abs() < 1e-10);
    }

    #[test]
    fn test_setpoint_computation() {
        let shape = FormationShape::Circle { radius: 0.5 };
        let ctrl = FormationController::new(0, shape, 4);
        let sp = ctrl.compute_setpoint();
        assert_eq!(sp.agent_id, 0);
        // Agent 0 should be at radius + center
        assert!((sp.offset[0] - 0.5).abs() < 1e-10); // cos(0) * 0.5
        assert!((sp.offset[2] - 0.1).abs() < 1e-10); // formation center z
    }

    #[test]
    fn test_formation_error_zero_when_aligned() {
        let shape = FormationShape::Line { spacing: 1.0 };
        let ctrl = FormationController::new(1, shape, 3);
        let sp = ctrl.compute_setpoint();
        let err = ctrl.formation_error(&sp.offset);
        assert!(err < 1e-10, "Error should be zero when at setpoint: {err}");
    }

    #[test]
    fn test_formation_state_serde() {
        let state = FormationState {
            agent_id: 0,
            encoded_state: vec![1.0, 2.0, 3.0],
            position: [0.0, 0.0, 0.1],
            timestamp: 1.5,
        };
        let json = serde_json::to_string(&state).unwrap();
        let restored: FormationState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.agent_id, 0);
        assert_eq!(restored.position, [0.0, 0.0, 0.1]);
    }

    #[test]
    fn test_custom_shape() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let shape = FormationShape::Custom(positions.clone());
        let setpoints = shape.compute_setpoints(3);
        assert_eq!(setpoints, positions);
    }

    #[test]
    fn test_line_half_spacing() {
        let shape = FormationShape::Line { spacing: 0.5 };
        let setpoints = shape.compute_setpoints(3);
        assert_eq!(setpoints.len(), 3);
        // With spacing 0.5, 3 agents: half-width = (2 * 0.5) / 2 = 0.5
        assert!((setpoints[0][0] - (-0.5)).abs() < 1e-10);
        assert!((setpoints[1][0] - 0.0).abs() < 1e-10);
        assert!((setpoints[2][0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_circle_angular_positions() {
        let shape = FormationShape::Circle { radius: 1.0 };
        let sp = shape.compute_setpoints(4);
        // Agent 0: angle=0 → (1, 0)
        assert!((sp[0][0] - 1.0).abs() < 1e-10);
        assert!((sp[0][1] - 0.0).abs() < 1e-10);
        // Agent 1: angle=π/2 → (0, 1)
        assert!((sp[1][0] - 0.0).abs() < 1e-10);
        assert!((sp[1][1] - 1.0).abs() < 1e-10);
        // Agent 2: angle=π → (-1, 0)
        assert!((sp[2][0] - (-1.0)).abs() < 1e-10);
        assert!((sp[2][1] - 0.0).abs() < 1e-10);
        // Agent 3: angle=3π/2 → (0, -1)
        assert!((sp[3][0] - 0.0).abs() < 1e-10);
        assert!((sp[3][1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_vee_symmetry() {
        let shape = FormationShape::Vee {
            angle: std::f64::consts::FRAC_PI_6, // 30 degrees
            spacing: 0.5,
        };
        let sp = shape.compute_setpoints(5);
        // Leader at origin
        assert!((sp[0][0]).abs() < 1e-10);
        assert!((sp[0][1]).abs() < 1e-10);
        // Agents 1 & 2 should be symmetric about y=0 (same x, opposite y)
        assert!((sp[1][0] - sp[2][0]).abs() < 1e-10, "Same x offset");
        assert!((sp[1][1] + sp[2][1]).abs() < 1e-10, "Symmetric y");
        // Agents 3 & 4 should be symmetric too, and farther back
        assert!((sp[3][0] - sp[4][0]).abs() < 1e-10);
        assert!((sp[3][1] + sp[4][1]).abs() < 1e-10);
        assert!(sp[3][0] < sp[1][0], "Farther agents are farther back");
    }

    #[test]
    fn test_controller_center_offset() {
        let shape = FormationShape::Line { spacing: 1.0 };
        let mut ctrl = FormationController::new(0, shape, 3);
        ctrl.set_center([5.0, 3.0, 1.0]);
        let sp = ctrl.compute_setpoint();
        // Agent 0 in 3-agent line: offset is -1.0 on x
        assert!((sp.offset[0] - 4.0).abs() < 1e-10);
        assert!((sp.offset[1] - 3.0).abs() < 1e-10);
        assert!((sp.offset[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_custom_padding_extends() {
        let positions = vec![[1.0, 0.0, 0.0]];
        let shape = FormationShape::Custom(positions);
        let sp = shape.compute_setpoints(3);
        assert_eq!(sp.len(), 3);
        assert_eq!(sp[0], [1.0, 0.0, 0.0]);
        // Padded positions should be at origin
        assert_eq!(sp[1], [0.0, 0.0, 0.0]);
        assert_eq!(sp[2], [0.0, 0.0, 0.0]);
    }
}
