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
    pub phi: std::sync::Mutex<f64>,
    pub interlock: std::sync::Mutex<symthaea_sim_bridge::AmygdalaInterlock>,
    pub surprise_monitor: std::sync::Mutex<symthaea_sim_bridge::SurpriseMonitor>,
    pub predicted_state_hv: std::sync::Mutex<Option<ContinuousHV>>,
    pub telemetry_broadcaster: std::sync::Arc<symthaea_telemetry_grpc::TelemetryBroadcaster>,
    #[cfg(feature = "ros2")]
    pub latest_joint_state: std::sync::Arc<std::sync::Mutex<Option<sensor_msgs::msg::JointState>>>,
    #[cfg(feature = "ros2")]
    pub latest_imu: std::sync::Arc<std::sync::Mutex<Option<sensor_msgs::msg::Imu>>>,
    #[cfg(feature = "ros2")]
    context: Option<std::sync::Arc<rclrs::Context>>,
    #[cfg(feature = "ros2")]
    node: Option<std::sync::Arc<rclrs::Node>>,
    #[cfg(feature = "ros2")]
    trajectory_pub: Option<std::sync::Arc<rclrs::Publisher<trajectory_msgs::msg::JointTrajectory>>>,
    #[cfg(feature = "ros2")]
    status_pub: Option<std::sync::Arc<rclrs::Publisher<std_msgs::msg::String>>>,
    #[cfg(feature = "ros2")]
    joint_state_sub: Option<std::sync::Arc<rclrs::Subscription<sensor_msgs::msg::JointState>>>,
    #[cfg(feature = "ros2")]
    imu_sub: Option<std::sync::Arc<rclrs::Subscription<sensor_msgs::msg::Imu>>>,
    #[cfg(feature = "ros2")]
    scheduler: symthaea_control::ConsciousnessGainScheduler,
    #[cfg(feature = "ros2")]
    spin_thread: Option<std::thread::JoinHandle<()>>,
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
    pub efforts: Vec<f64>,
    pub time_from_start_ns: u64,
}

impl RosBridgeNode {
    /// Create a new bridge node with the given configuration.
    pub fn new(config: RosBridgeConfig) -> Self {
        Self {
            config,
            is_running: false,
            phi: std::sync::Mutex::new(0.0),
            interlock: std::sync::Mutex::new(symthaea_sim_bridge::AmygdalaInterlock::default()),
            surprise_monitor: std::sync::Mutex::new(symthaea_sim_bridge::SurpriseMonitor::default()),
            predicted_state_hv: std::sync::Mutex::new(None),
            telemetry_broadcaster: std::sync::Arc::new(
                symthaea_telemetry_grpc::TelemetryBroadcaster::new(),
            ),
            #[cfg(feature = "ros2")]
            latest_joint_state: std::sync::Arc::new(std::sync::Mutex::new(None)),
            #[cfg(feature = "ros2")]
            latest_imu: std::sync::Arc::new(std::sync::Mutex::new(None)),
            #[cfg(feature = "ros2")]
            context: None,
            #[cfg(feature = "ros2")]
            node: None,
            #[cfg(feature = "ros2")]
            trajectory_pub: None,
            #[cfg(feature = "ros2")]
            status_pub: None,
            #[cfg(feature = "ros2")]
            joint_state_sub: None,
            #[cfg(feature = "ros2")]
            imu_sub: None,
            #[cfg(feature = "ros2")]
            scheduler: symthaea_control::ConsciousnessGainScheduler::default(),
            #[cfg(feature = "ros2")]
            spin_thread: None,
        }
    }

    /// Initialize the ROS2 context, publishers/subscribers, and spin up the background executor thread.
    pub fn init(&mut self) -> Result<(), RosBridgeError> {
        #[cfg(feature = "ros2")]
        {
            let context = std::sync::Arc::new(
                rclrs::Context::new(std::env::args())
                    .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?,
            );
            let node = rclrs::create_node(&context, &self.config.node_name)
                .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;

            let trajectory_topic = format!("/{}/joint_trajectory", self.config.namespace);
            let trajectory_pub = node
                .create_publisher::<trajectory_msgs::msg::JointTrajectory>(
                    &trajectory_topic,
                    rclrs::QOS_PROFILE_DEFAULT,
                )
                .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;

            let status_topic = format!("/{}/status", self.config.namespace);
            let status_pub = node
                .create_publisher::<std_msgs::msg::String>(
                    &status_topic,
                    rclrs::QOS_PROFILE_DEFAULT,
                )
                .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;

            // 1. Subscribe to joint states and IMU feedback
            let latest_joint_state_clone = self.latest_joint_state.clone();
            let joint_state_sub = node
                .create_subscription::<sensor_msgs::msg::JointState, _>(
                    "/joint_states",
                    rclrs::QOS_PROFILE_DEFAULT,
                    move |msg: sensor_msgs::msg::JointState| {
                        let mut lock = latest_joint_state_clone.lock().unwrap();
                        *lock = Some(msg);
                    },
                )
                .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;

            let latest_imu_clone = self.latest_imu.clone();
            let imu_sub = node
                .create_subscription::<sensor_msgs::msg::Imu, _>(
                    "/imu",
                    rclrs::QOS_PROFILE_DEFAULT,
                    move |msg: sensor_msgs::msg::Imu| {
                        let mut lock = latest_imu_clone.lock().unwrap();
                        *lock = Some(msg);
                    },
                )
                .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;

            // 2. Spawn the background spinning executor thread
            let node_clone = node.clone();
            let spin_thread = std::thread::spawn(move || {
                if let Err(e) = rclrs::spin(node_clone) {
                    tracing::error!("ROS2 spin thread error: {:?}", e);
                }
            });

            self.context = Some(context);
            self.node = Some(node);
            self.trajectory_pub = Some(trajectory_pub);
            self.status_pub = Some(status_pub);
            self.joint_state_sub = Some(joint_state_sub);
            self.imu_sub = Some(imu_sub);
            self.spin_thread = Some(spin_thread);
        }
        let broadcaster_clone = self.telemetry_broadcaster.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let addr = "127.0.0.1:50051".parse().unwrap();
                if let Err(e) = broadcaster_clone.run(addr).await {
                    tracing::error!("gRPC Telemetry server error: {:?}", e);
                }
            });
        });
        self.is_running = true;
        Ok(())
    }

    /// Update the surprise monitor based on similarity between actual and expected proprioception vectors.
    #[cfg(feature = "ros2")]
    pub fn update_surprise_monitor(&self) -> Result<(), RosBridgeError> {
        let latest_joint_state_opt = self.latest_joint_state.lock().unwrap().clone();
        if let Some(joint_state) = latest_joint_state_opt {
            let joint_angles_f32: Vec<f32> =
                joint_state.position.iter().map(|&p| p as f32).collect();
            let current_hv = self.process_joint_states(&joint_angles_f32);
            let predicted_opt = self.predicted_state_hv.lock().unwrap().clone();
            if let Some(predicted_hv) = predicted_opt {
                let similarity = current_hv.similarity(&predicted_hv);
                let surprise = (1.0 - similarity as f64).clamp(0.0, 1.0);
                let mut surprise_monitor = self.surprise_monitor.lock().unwrap();
                surprise_monitor.update(surprise);
            }
        }
        Ok(())
    }

    /// Estimate and return the current `HumanoidState` using sensor fusion over JointState and IMU.
    #[cfg(feature = "ros2")]
    pub fn get_current_humanoid_state(
        &self,
    ) -> Result<symthaea_humanoid::types::HumanoidState, RosBridgeError> {
        let joint_state_opt = self.latest_joint_state.lock().unwrap().clone();
        let imu_opt = self.latest_imu.lock().unwrap().clone();

        let joint_state = joint_state_opt.ok_or(RosBridgeError::NotInitialized)?;

        // Torso orientation and angular velocity from IMU
        let root_quat = if let Some(ref imu) = imu_opt {
            [
                imu.orientation.w,
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
            ]
        } else {
            [1.0, 0.0, 0.0, 0.0]
        };

        let root_ang_vel = if let Some(ref imu) = imu_opt {
            [
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };

        // Torso vertical vector from IMU orientation
        let torso_vertical = if let Some(ref imu) = imu_opt {
            let w = imu.orientation.w;
            let x = imu.orientation.x;
            let y = imu.orientation.y;
            let z = imu.orientation.z;
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ]
        } else {
            [0.0, 0.0, 1.0]
        };

        // Linear velocity estimated from IMU linear acceleration as a simple proxy
        let root_lin_vel = if let Some(ref imu) = imu_opt {
            [
                imu.linear_acceleration.x * 0.025,
                imu.linear_acceleration.y * 0.025,
                imu.linear_acceleration.z * 0.025,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };

        let timestamp =
            joint_state.header.stamp.sec as f64 + (joint_state.header.stamp.nanosec as f64 * 1e-9);

        // Standard DMC humanoid heights: root at ~1.3m, head at ~1.4m
        let root_height = 1.3;
        let head_height = 1.4;

        let mut state = self.ros_joint_states_to_humanoid(
            &joint_state.position,
            &joint_state.velocity,
            head_height,
            torso_vertical,
            root_lin_vel,
            timestamp,
        );

        // Update root pose metrics with actual IMU orientation & height
        state.root_height = root_height;
        state.root_quaternion = root_quat;
        state.root_angular_velocity = root_ang_vel;

        Ok(state)
    }

    /// Log status and trajectory states to a local CSV file for offline replay and IIT diagnostics.
    pub fn log_telemetry(
        &self,
        status: &ConsciousnessStatusMsg,
        joint_names: &[String],
        positions: &[f64],
    ) -> Result<(), RosBridgeError> {
        let log_dir = "logs/telemetry";
        std::fs::create_dir_all(log_dir)
            .map_err(|e| RosBridgeError::Serialization(e.to_string()))?;

        let path = format!("{}/episode_log.csv", log_dir);
        let file_exists = std::path::Path::new(&path).exists();

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| RosBridgeError::Serialization(e.to_string()))?;

        use std::io::Write;
        if !file_exists {
            writeln!(
                file,
                "timestamp,phi,arousal,uncertainty,joint_count,first_joint_pos"
            )
            .map_err(|e| RosBridgeError::Serialization(e.to_string()))?;
        }

        let first_pos = positions.first().copied().unwrap_or(0.0);
        writeln!(
            file,
            "{},{},{},{},{},{}",
            status.timestamp,
            status.phi,
            status.arousal,
            status.uncertainty,
            joint_names.len(),
            first_pos
        )
        .map_err(|e| RosBridgeError::Serialization(e.to_string()))?;

        Ok(())
    }

    /// Publish consciousness status to the `/symthaea/status` topic.
    pub fn publish_status(&self, status: ConsciousnessStatusMsg) -> Result<(), RosBridgeError> {
        if !self.is_running {
            return Err(RosBridgeError::NotInitialized);
        }

        // Store the latest Φ value for gain scheduling
        {
            let mut stored_phi = self.phi.lock().unwrap();
            *stored_phi = status.phi;
        }

        #[cfg(feature = "ros2")]
        {
            if let Some(pub_status) = &self.status_pub {
                let serialized = serde_json::to_string(&status)
                    .map_err(|e| RosBridgeError::Serialization(e.to_string()))?;
                let mut msg = std_msgs::msg::String::default();
                msg.data = serialized;
                pub_status
                    .publish(&msg)
                    .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;
            }
        }

        // Broadcast live telemetry via gRPC stream
        let surprise = self.surprise_monitor.lock().unwrap().current_surprise;
        let frame = symthaea_telemetry_grpc::TelemetryFrame {
            phi: status.phi,
            harmonies: status.harmonies.to_vec(),
            neuromodulators: status.neuromodulators.to_vec(),
            arousal: status.arousal,
            uncertainty: status.uncertainty,
            surprise,
            timestamp: status.timestamp.clone(),
            mental_movie: None,
            gwt_broadcast: 0.0,
            hot_metacognitive: 0.0,
            ast_temporal: 0.0,
            knowledge_coherence: 0.0,
            embodiment_level: 0.0,
            self_awareness: 0.0,
            topological_unity: 1.0,
        };
        self.telemetry_broadcaster.broadcast(frame);
        tracing::debug!(
            "ROS2 [{}] Publish Status: Φ={:.4}, Arousal={:.2}",
            self.config.node_name,
            status.phi,
            status.arousal
        );
        Ok(())
    }

    /// Publish a joint trajectory command to the robot, applying gain scheduling and safety interlocks.
    pub fn publish_trajectory(
        &self,
        mut command: RosTrajectoryCommand,
    ) -> Result<(), RosBridgeError> {
        if !self.is_running {
            return Err(RosBridgeError::NotInitialized);
        }

        // 1. Amygdala Interlock safety checks
        let mut peak_torque = 0.0f32;
        for pt in &command.points {
            for &eff in &pt.efforts {
                if eff.abs() as f32 > peak_torque {
                    peak_torque = eff.abs() as f32;
                }
            }
        }

        let mut interlock = self.interlock.lock().unwrap();
        let safety_status = interlock.monitor(peak_torque, 0.0);

        if safety_status == symthaea_sim_bridge::SafetyStatus::Red {
            tracing::warn!(
                "ROS2 Amygdala Interlock E-STOP triggered! Clamping all motor command efforts to zero."
            );
        }

        // Apply any dampening or clamping overrides
        for pt in &mut command.points {
            for eff in &mut pt.efforts {
                *eff = interlock.apply_override(*eff as f32) as f64;
            }
        }

        // 2. Consciousness-Gated Gain Scheduling
        #[cfg(feature = "ros2")]
        let phi = { *self.phi.lock().unwrap() };
        #[cfg(feature = "ros2")]
        {
            let gain_scale = self.scheduler.alpha(phi);
            for pt in &mut command.points {
                for eff in &mut pt.efforts {
                    *eff *= gain_scale;
                }
            }
        }

        let joint_count = command.joint_names.len();
        let point_count = command.points.len();

        #[cfg(feature = "ros2")]
        {
            if let Some(pub_traj) = &self.trajectory_pub {
                let mut msg = trajectory_msgs::msg::JointTrajectory::default();
                msg.joint_names = command.joint_names;

                for pt in command.points {
                    let mut point = trajectory_msgs::msg::JointTrajectoryPoint::default();
                    point.positions = pt.positions;
                    point.velocities = pt.velocities;
                    point.effort = pt.efforts;

                    let sec = (pt.time_from_start_ns / 1_000_000_000) as i32;
                    let nanosec = (pt.time_from_start_ns % 1_000_000_000) as u32;
                    point.time_from_start = builtin_interfaces::msg::Duration { sec, nanosec };

                    msg.points.push(point);
                }

                pub_traj
                    .publish(&msg)
                    .map_err(|e| RosBridgeError::Communication(format!("{:?}", e)))?;
            }
        }
        let gain_scale = {
            #[cfg(feature = "ros2")]
            {
                self.scheduler.alpha(phi)
            }
            #[cfg(not(feature = "ros2"))]
            {
                1.0
            }
        };
        tracing::debug!(
            "ROS2 [{}] Publish Trajectory: {} joints, {} points, Φ scale={:.2}",
            self.config.node_name,
            joint_count,
            point_count,
            gain_scale
        );
        Ok(())
    }

    /// Process a proprioception update and return an HDC-encoded sensor vector.
    pub fn process_joint_states(&self, joint_angles: &[f32]) -> ContinuousHV {
        let dimension = 16384;
        if joint_angles.is_empty() {
            return ContinuousHV::zero(dimension);
        }

        let mut hv = ContinuousHV::zero(dimension);
        let mut first = true;

        for (i, &angle) in joint_angles.iter().enumerate() {
            let joint_base = ContinuousHV::random(dimension, (i as u64) ^ 0x5E5C0DE);
            let sensation = joint_base.scale(angle);

            if first {
                hv = sensation;
                first = false;
            } else {
                hv = ContinuousHV::bundle(&[&hv, &sensation]);
            }
        }

        hv.normalize()
    }

    /// Converts a Symthaea HumanoidCommand into a ROS2 JointTrajectory format, using torque/effort control.
    pub fn command_to_ros_trajectory(
        &self,
        command: &symthaea_humanoid::types::HumanoidCommand,
        joint_names: Vec<String>,
        time_from_start_ns: u64,
    ) -> RosTrajectoryCommand {
        let efforts = command.torques.iter().map(|&t| t as f64).collect();
        let point = RosTrajectoryPoint {
            positions: vec![],
            velocities: vec![],
            efforts,
            time_from_start_ns,
        };
        RosTrajectoryCommand {
            joint_names,
            points: vec![point],
        }
    }

    /// Converts ROS2 joint state arrays back into Symthaea's HumanoidState.
    /// Since ROS2 JointState lacks root pose/IMU, those defaults are injected or assumed.
    pub fn ros_joint_states_to_humanoid(
        &self,
        positions: &[f64],
        velocities: &[f64],
        head_height: f64,
        torso_vertical: [f64; 3],
        com_velocity: [f64; 3],
        timestamp: f64,
    ) -> symthaea_humanoid::types::HumanoidState {
        // HumanoidState::from_mujoco expects qpos (root 7 + joints) and qvel (root 6 + joints)
        let mut qpos = vec![0.0; 7]; // Root pose stub: [x, y, z, qw, qx, qy, qz]
        qpos[2] = 1.0; // Assume upright root height
        qpos[3] = 1.0; // Quaternion w
        qpos.extend_from_slice(positions);

        let mut qvel = vec![0.0; 6]; // Root vel stub: [vx, vy, vz, wx, wy, wz]
        qvel.extend_from_slice(velocities);

        symthaea_humanoid::types::HumanoidState::from_mujoco(
            &qpos,
            &qvel,
            head_height,
            torso_vertical,
            &[0.0, 0.0, 0.0, 0.0], // Extremities stub
            com_velocity,
            timestamp,
        )
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

    #[test]
    fn test_amygdala_safety_and_gain_scaling() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        node.init().unwrap();

        let normal_cmd = RosTrajectoryCommand {
            joint_names: vec!["joint_1".to_string()],
            points: vec![RosTrajectoryPoint {
                positions: vec![0.0],
                velocities: vec![0.0],
                efforts: vec![10.0],
                time_from_start_ns: 1000000,
            }],
        };

        node.publish_status(ConsciousnessStatusMsg {
            phi: 1.0,
            harmonies: [0.5; 8],
            neuromodulators: [0.5; 4],
            arousal: 0.5,
            uncertainty: 0.1,
            timestamp: "now".to_string(),
        })
        .unwrap();

        node.publish_trajectory(normal_cmd.clone()).unwrap();

        {
            let mut interlock = node.interlock.lock().unwrap();
            assert_eq!(interlock.status, symthaea_sim_bridge::SafetyStatus::Green);

            // Peak torque exceeding limit * 1.5 triggers RED
            let status = interlock.monitor(200.0, 0.0);
            assert_eq!(status, symthaea_sim_bridge::SafetyStatus::Red);
            assert_eq!(interlock.apply_override(10.0), 0.0);
        }

        #[cfg(feature = "ros2")]
        {
            node.publish_status(ConsciousnessStatusMsg {
                phi: 0.0,
                harmonies: [0.5; 8],
                neuromodulators: [0.5; 4],
                arousal: 0.5,
                uncertainty: 0.1,
                timestamp: "now".to_string(),
            })
            .unwrap();
            let cmd = normal_cmd.clone();
            node.publish_trajectory(cmd).unwrap();
        }
    }

    #[test]
    fn test_hdc_encoding_and_surprise() {
        let node = RosBridgeNode::new(RosBridgeConfig::default());
        let joints = vec![0.1f32, -0.2f32, 0.3f32];
        let hv = node.process_joint_states(&joints);
        assert_eq!(hv.dim(), 16384);

        {
            let mut predicted = node.predicted_state_hv.lock().unwrap();
            *predicted = Some(hv.clone());
        }

        #[cfg(feature = "ros2")]
        {
            assert!(node.get_current_humanoid_state().is_err());

            let mut js = sensor_msgs::msg::JointState::default();
            js.position = vec![0.1, -0.2, 0.3];
            js.velocity = vec![0.0, 0.0, 0.0];
            *node.latest_joint_state.lock().unwrap() = Some(js);

            node.update_surprise_monitor().unwrap();
            let surprise = node.surprise_monitor.lock().unwrap().current_surprise;
            assert!(
                surprise < 0.1,
                "Surprise should be low when predicted matches current"
            );
        }
    }

    #[test]
    fn test_telemetry_logging() {
        let node = RosBridgeNode::new(RosBridgeConfig::default());
        let status = ConsciousnessStatusMsg {
            phi: 0.95,
            harmonies: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            neuromodulators: [0.1, 0.2, 0.3, 0.4],
            arousal: 0.85,
            uncertainty: 0.05,
            timestamp: "2026-06-25T09:44:00Z".to_string(),
        };
        let joint_names = vec!["joint_1".to_string()];
        let positions = vec![0.123];

        node.log_telemetry(&status, &joint_names, &positions)
            .unwrap();

        let path = "logs/telemetry/episode_log.csv";
        assert!(std::path::Path::new(path).exists());
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("2026-06-25T09:44:00Z"));
        assert!(content.contains("0.95"));

        let _ = std::fs::remove_file(path);
    }
}
