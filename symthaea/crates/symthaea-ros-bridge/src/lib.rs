// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-ros-bridge
//!
//! Bridges Symthaea's holographic liquid brain (HLB) to the ROS2 ecosystem.
//!
//! ## Capabilities
//!
//! - **Proprioception**: Subscribes to `/joint_states` and transforms them into HDC vectors.
//! - **Consciousness Telemetry**: Publishes Φ (Phi), neuromodulator levels, and Eight Harmonies activations.
//! - **Motor Output**: Translates `RobotActuatorCommand` into `trajectory_msgs`.
//!
//! ## ROS2 Setup
//!
//! This crate requires a ROS2 environment (e.g., Humble). If using Nix, ensure
//! `nix-ros-overlay` is enabled in your `flake.nix`.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;
use thiserror::Error;

/// Errors reported by the ROS2 bridge.
#[derive(Debug, Error)]
pub enum RosBridgeError {
    #[error("ROS2 context not initialized")]
    NotInitialized,
    #[error("Communication error: {0}")]
    Communication(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Normalized consciousness metrics for ROS2 broadcasting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessStatusMsg {
    /// Integrated Information Theory (IIT) Φ value [0, 1].
    pub phi: f64,
    /// Eight Harmonies activations [0, 1].
    pub harmonies: [f32; 8],
    /// Neuromodulator levels: [dopamine, serotonin, noradrenaline, acetylcholine].
    pub neuromodulators: [f32; 4],
    /// Global arousal level [0, 1].
    pub arousal: f32,
    /// Epistemic uncertainty [0, 1].
    pub uncertainty: f32,
    /// Current cycle timestamp (ISO 8601).
    pub timestamp: String,
}

/// Configuration for the ROS2 bridge node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosBridgeConfig {
    /// Node name in the ROS graph.
    pub node_name: String,
    /// Namespace for all topics.
    pub namespace: String,
    /// Update rate for telemetry publishing (Hz).
    pub telemetry_rate_hz: f32,
}

impl Default for RosBridgeConfig {
    fn default() -> Self {
        Self {
            node_name: "symthaea_brain_bridge".to_string(),
            namespace: "symthaea".to_string(),
            telemetry_rate_hz: 10.0,
        }
    }
}

/// Placeholder for the ROS2 bridge node.
///
/// In a real implementation with `rclrs`, this would own the `rclrs::Node`
/// and various publishers/subscribers.
pub struct RosBridgeNode {
    config: RosBridgeConfig,
    is_running: bool,
}

/// Command to control robot joint trajectories via ROS2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosTrajectoryCommand {
    pub joint_names: Vec<String>,
    pub points: Vec<RosTrajectoryPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosTrajectoryPoint {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub time_from_start_ns: u64,
}

impl RosBridgeNode {
    /// Create a new bridge node with the given configuration.
    pub fn new(config: RosBridgeConfig) -> Self {
        Self {
            config,
            is_running: false,
        }
    }

    /// Initialize the ROS2 context and create publishers/subscribers.
    pub fn init(&mut self) -> Result<(), RosBridgeError> {
        #[cfg(feature = "ros2")]
        {
            // Real initialization with rclrs would happen here.
            // Example: let context = rclrs::Context::new(std::env::args())?;
        }
        self.is_running = true;
        Ok(())
    }

    /// Publish consciousness status to the `/symthaea/status` topic.
    pub fn publish_status(&self, status: ConsciousnessStatusMsg) -> Result<(), RosBridgeError> {
        if !self.is_running {
            return Err(RosBridgeError::NotInitialized);
        }
        #[cfg(feature = "ros2")]
        {
            // Publish logic using rclrs::Publisher<ConsciousnessStatusMsg>
        }
        tracing::debug!(
            "ROS2 [{}] Publish Status: Φ={:.4}, Arousal={:.2}",
            self.config.node_name,
            status.phi,
            status.arousal
        );
        Ok(())
    }

    /// Publish a joint trajectory command to the robot.
    pub fn publish_trajectory(&self, command: RosTrajectoryCommand) -> Result<(), RosBridgeError> {
        if !self.is_running {
            return Err(RosBridgeError::NotInitialized);
        }
        tracing::debug!(
            "ROS2 [{}] Publish Trajectory: {} joints, {} points",
            self.config.node_name,
            command.joint_names.len(),
            command.points.len()
        );
        Ok(())
    }

    /// Process a proprioception update and return an HDC-encoded sensor vector.
    pub fn process_joint_states(&self, _joint_angles: &[f32]) -> ContinuousHV {
        // TODO: Map ROS2 JointState message to HDC space via learned projection
        ContinuousHV::zero(16384)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = RosBridgeConfig::default();
        assert_eq!(config.node_name, "symthaea_brain_bridge");
    }

    #[test]
    fn test_node_lifecycle() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        assert!(
            node.publish_status(ConsciousnessStatusMsg {
                phi: 0.8,
                harmonies: [0.5; 8],
                neuromodulators: [0.5; 4],
                arousal: 0.5,
                uncertainty: 0.1,
                timestamp: "now".to_string(),
            })
            .is_err()
        );

        node.init().unwrap();
        assert!(
            node.publish_status(ConsciousnessStatusMsg {
                phi: 0.8,
                harmonies: [0.5; 8],
                neuromodulators: [0.5; 4],
                arousal: 0.5,
                uncertainty: 0.1,
                timestamp: "now".to_string(),
            })
            .is_ok()
        );
    }
}
