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
//! Enable the `ros2` Cargo feature to compile the ROS transport and qualify it
//! against an exact ROS distro/OS/RMW/architecture profile. Without that feature,
//! telemetry may still run, but trajectory publication returns
//! `RosTransportUnavailable`; it never simulates a successful robot command.
//! The ROS 2 support matrix and migration gates are tracked in issue #1287.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;
use thiserror::Error;

/// Errors reported by the ROS2 bridge.
#[derive(Debug, Error)]
pub enum RosBridgeError {
    #[error("ROS2 context not initialized")]
    NotInitialized,
    #[error("ROS2 trajectory publisher unavailable; command was not published")]
    RosTransportUnavailable,
    #[error("Communication error: {0}")]
    Communication(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Invalid consciousness status: {0}")]
    InvalidStatus(String),
    #[error("Invalid trajectory command: {0}")]
    InvalidTrajectory(String),
    #[error("Invalid ROS JointState: {0}")]
    InvalidJointState(String),
    #[error("Sensor data unavailable: {0}")]
    SensorDataUnavailable(String),
    #[error("Invalid sensor data: {0}")]
    InvalidSensorData(String),
    #[error("Safety interlock blocked trajectory publication: {0}")]
    SafetyInterlock(String),
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

/// Reorder one ROS JointState value array into the bridge's explicit model order.
///
/// ROS message array position is not joint identity. This helper requires an exact
/// set of unique names and finite values; missing or unknown joints remain errors
/// rather than being silently filled, dropped, or shifted into another joint slot.
fn canonicalize_joint_values(
    observed_names: &[String],
    values: &[f64],
    expected_names: &[&str],
    field: &str,
) -> Result<Vec<f64>, RosBridgeError> {
    use std::collections::{HashMap, HashSet};

    if expected_names.is_empty() {
        return Err(RosBridgeError::InvalidJointState(
            "expected joint profile must not be empty".into(),
        ));
    }
    let expected_set: HashSet<&str> = expected_names.iter().copied().collect();
    if expected_set.len() != expected_names.len()
        || expected_names.iter().any(|name| name.trim().is_empty())
    {
        return Err(RosBridgeError::InvalidJointState(
            "expected joint profile contains empty or duplicate names".into(),
        ));
    }
    if observed_names.len() != values.len() {
        return Err(RosBridgeError::InvalidJointState(format!(
            "name/value cardinality mismatch: {} names but {} {field} values",
            observed_names.len(),
            values.len()
        )));
    }
    if observed_names.len() != expected_names.len() {
        return Err(RosBridgeError::InvalidJointState(format!(
            "expected exactly {} joints, received {}",
            expected_names.len(),
            observed_names.len()
        )));
    }

    let mut indices = HashMap::with_capacity(observed_names.len());
    for (index, name) in observed_names.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(RosBridgeError::InvalidJointState(
                "joint names must not be empty".into(),
            ));
        }
        if !expected_set.contains(name.as_str()) {
            return Err(RosBridgeError::InvalidJointState(format!(
                "unrecognized joint name: {name}"
            )));
        }
        if indices.insert(name.as_str(), index).is_some() {
            return Err(RosBridgeError::InvalidJointState(format!(
                "duplicate joint name: {name}"
            )));
        }
        if !values[index].is_finite() {
            return Err(RosBridgeError::InvalidJointState(format!(
                "{field} for joint {name} is not finite"
            )));
        }
    }

    expected_names
        .iter()
        .map(|name| {
            indices
                .get(name)
                .map(|&index| values[index])
                .ok_or_else(|| {
                    RosBridgeError::InvalidJointState(format!(
                        "required joint is missing: {name}"
                    ))
                })
        })
        .collect()
}

/// Validate and normalize an IMU quaternion in ROS [w, x, y, z] order.
fn normalize_imu_quaternion(quaternion: [f64; 4]) -> Result<[f64; 4], RosBridgeError> {
    if quaternion.iter().any(|component| !component.is_finite()) {
        return Err(RosBridgeError::InvalidSensorData(
            "IMU orientation quaternion contains a non-finite component".into(),
        ));
    }

    let norm_squared: f64 = quaternion.iter().map(|component| component * component).sum();
    if !norm_squared.is_finite() || norm_squared <= 1.0e-24 {
        return Err(RosBridgeError::InvalidSensorData(
            "IMU orientation quaternion is degenerate or cannot be normalized".into(),
        ));
    }

    let norm = norm_squared.sqrt();
    Ok(quaternion.map(|component| component / norm))
}

/// Reject non-finite 3D sensor vectors before they enter physical state.
fn validate_sensor_vector(field: &str, values: [f64; 3]) -> Result<[f64; 3], RosBridgeError> {
    if values.iter().any(|component| !component.is_finite()) {
        return Err(RosBridgeError::InvalidSensorData(format!(
            "{field} contains a non-finite component"
        )));
    }
    Ok(values)
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

impl ConsciousnessStatusMsg {
    fn validate(&self) -> Result<(), RosBridgeError> {
        fn unit(value: f64, name: &str) -> Result<(), RosBridgeError> {
            if value.is_finite() && (0.0..=1.0).contains(&value) {
                Ok(())
            } else {
                Err(RosBridgeError::InvalidStatus(format!(
                    "{name} must be finite and in [0, 1]"
                )))
            }
        }

        unit(self.phi, "phi")?;
        unit(self.arousal as f64, "arousal")?;
        unit(self.uncertainty as f64, "uncertainty")?;
        for (index, value) in self.harmonies.iter().enumerate() {
            unit(*value as f64, &format!("harmonies[{index}]"))?;
        }
        for (index, value) in self.neuromodulators.iter().enumerate() {
            unit(*value as f64, &format!("neuromodulators[{index}]"))?;
        }
        if self.timestamp.trim().is_empty() {
            return Err(RosBridgeError::InvalidStatus(
                "timestamp must not be empty".into(),
            ));
        }
        Ok(())
    }
}

impl RosTrajectoryCommand {
    fn validate(&self) -> Result<(), RosBridgeError> {
        use std::collections::HashSet;

        if self.joint_names.is_empty() {
            return Err(RosBridgeError::InvalidTrajectory(
                "joint_names must not be empty".into(),
            ));
        }
        if self.points.is_empty() {
            return Err(RosBridgeError::InvalidTrajectory(
                "trajectory must contain at least one point".into(),
            ));
        }
        let mut unique = HashSet::with_capacity(self.joint_names.len());
        for name in &self.joint_names {
            if name.trim().is_empty() || !unique.insert(name) {
                return Err(RosBridgeError::InvalidTrajectory(
                    "joint names must be non-empty and unique".into(),
                ));
            }
        }

        let joint_count = self.joint_names.len();
        let mut previous_time = None;
        for (point_index, point) in self.points.iter().enumerate() {
            if previous_time.is_some_and(|time| point.time_from_start_ns <= time) {
                return Err(RosBridgeError::InvalidTrajectory(format!(
                    "point {point_index} time_from_start must be strictly increasing"
                )));
            }
            if point.time_from_start_ns / 1_000_000_000 > i32::MAX as u64 {
                return Err(RosBridgeError::InvalidTrajectory(format!(
                    "point {point_index} time_from_start exceeds ROS Duration range"
                )));
            }
            previous_time = Some(point.time_from_start_ns);

            if point.positions.is_empty() && point.velocities.is_empty() && point.efforts.is_empty()
            {
                return Err(RosBridgeError::InvalidTrajectory(format!(
                    "point {point_index} has no command values"
                )));
            }
            for (field, values) in [
                ("positions", &point.positions),
                ("velocities", &point.velocities),
                ("efforts", &point.efforts),
            ] {
                if !values.is_empty() && values.len() != joint_count {
                    return Err(RosBridgeError::InvalidTrajectory(format!(
                        "point {point_index} {field} length {} does not match {joint_count} joints",
                        values.len()
                    )));
                }
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(RosBridgeError::InvalidTrajectory(format!(
                        "point {point_index} {field} contains a non-finite value"
                    )));
                }
            }
            if point
                .efforts
                .iter()
                .any(|value| value.abs() > f32::MAX as f64)
            {
                return Err(RosBridgeError::InvalidTrajectory(format!(
                    "point {point_index} effort exceeds interlock numeric range"
                )));
            }
        }
        Ok(())
    }
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

    /// Initialize ROS endpoints when the `ros2` feature is enabled and start telemetry.
    ///
    /// Without `ros2`, initialization is telemetry-only; robot trajectory
    /// publication remains unavailable and fails explicitly.
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
            let canonical_positions = canonicalize_joint_values(
                &joint_state.name,
                &joint_state.position,
                &symthaea_humanoid::types::JOINT_NAMES,
                "position",
            )?;
            let joint_angles_f32: Vec<f32> =
                canonical_positions.iter().map(|&p| p as f32).collect();
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

    /// Return a humanoid state only when the subscribed evidence can support its fields.
    ///
    /// Joint positions and velocities are mapped by canonical joint names. IMU
    /// orientation and angular velocity are validated, but this bridge currently
    /// has no odometry or qualified state-estimator source for root/COM linear
    /// velocity. A single acceleration sample multiplied by a fixed timestep is
    /// not velocity. Until a typed velocity source is wired in, this method fails
    /// closed instead of constructing a plausible-looking state with fabricated
    /// neutral values.
    #[cfg(feature = "ros2")]
    pub fn get_current_humanoid_state(
        &self,
    ) -> Result<symthaea_humanoid::types::HumanoidState, RosBridgeError> {
        let joint_state_opt = self.latest_joint_state.lock().unwrap().clone();
        let imu_opt = self.latest_imu.lock().unwrap().clone();

        let joint_state = joint_state_opt.ok_or(RosBridgeError::NotInitialized)?;
        // DMC21 is the current explicit bridge profile. Resolve semantic identity
        // by JointState.name; never assume transport array position is model order.
        let _canonical_positions = canonicalize_joint_values(
            &joint_state.name,
            &joint_state.position,
            &symthaea_humanoid::types::JOINT_NAMES,
            "position",
        )?;
        let _canonical_velocities = canonicalize_joint_values(
            &joint_state.name,
            &joint_state.velocity,
            &symthaea_humanoid::types::JOINT_NAMES,
            "velocity",
        )?;

        let imu = imu_opt.ok_or_else(|| {
            RosBridgeError::SensorDataUnavailable(
                "IMU orientation/angular velocity is absent; no neutral physical state was substituted".into(),
            )
        })?;
        let _orientation = normalize_imu_quaternion([
            imu.orientation.w,
            imu.orientation.x,
            imu.orientation.y,
            imu.orientation.z,
        ])?;
        let _angular_velocity = validate_sensor_vector(
            "IMU angular velocity",
            [
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z,
            ],
        )?;

        Err(RosBridgeError::SensorDataUnavailable(
            "root linear velocity and center-of-mass velocity are not subscribed or estimated by this bridge; IMU linear acceleration cannot substitute for velocity. Add a qualified odometry/state-estimator source before constructing HumanoidState".into(),
        ))
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
        status.validate()?;

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

    /// Submit a joint trajectory through the live ROS publisher, applying gain scheduling and safety interlocks.
    ///
    /// A successful return means only that the local ROS publisher accepted the
    /// message. It does not prove subscriber delivery, controller acceptance,
    /// actuator latch, or physical movement.
    pub fn publish_trajectory(
        &self,
        mut command: RosTrajectoryCommand,
    ) -> Result<(), RosBridgeError> {
        if !self.is_running {
            return Err(RosBridgeError::NotInitialized);
        }
        command.validate()?;

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
                "ROS2 Amygdala Interlock E-STOP triggered; trajectory was not published"
            );
            return Err(RosBridgeError::SafetyInterlock(
                "emergency stop is latched; explicitly reset it before sending commands".into(),
            ));
        }
        if safety_status == symthaea_sim_bridge::SafetyStatus::Yellow
            && command
                .points
                .iter()
                .any(|point| !point.positions.is_empty() || !point.velocities.is_empty())
        {
            return Err(RosBridgeError::SafetyInterlock(
                "yellow state cannot safely damp position/velocity trajectories".into(),
            ));
        }

        // Apply any dampening or clamping overrides
        for pt in &mut command.points {
            for eff in &mut pt.efforts {
                *eff = interlock.apply_override(*eff as f32) as f64;
            }
        }

        // Publish only through a live ROS publisher. Telemetry-only builds and
        // partially initialized ROS nodes must never report command success.
        #[cfg(feature = "ros2")]
        {
            let phi = *self.phi.lock().unwrap();
            let gain_scale = self.scheduler.alpha(phi);
            for pt in &mut command.points {
                for eff in &mut pt.efforts {
                    *eff *= gain_scale;
                }
            }

            let pub_traj = self
                .trajectory_pub
                .as_ref()
                .ok_or(RosBridgeError::RosTransportUnavailable)?;

            let joint_count = command.joint_names.len();
            let point_count = command.points.len();
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

            tracing::debug!(
                "ROS2 [{}] Publish Trajectory: {} joints, {} points, Φ scale={:.2}",
                self.config.node_name,
                joint_count,
                point_count,
                gain_scale
            );
            Ok(())
        }

        #[cfg(not(feature = "ros2"))]
        {
            // All local validation/interlock checks above still run, but no
            // ROS command transport exists in this build.
            Err(RosBridgeError::RosTransportUnavailable)
        }
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

    // No JointState-only -> HumanoidState converter is exposed here: the
    // available ROS topics do not provide a complete, qualified root pose and
    // linear-velocity observation. See get_current_humanoid_state(), which fails
    // closed until those evidence sources exist.

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

        #[cfg(feature = "ros2")]
        node.publish_trajectory(normal_cmd.clone()).unwrap();
        #[cfg(not(feature = "ros2"))]
        assert!(matches!(
            node.publish_trajectory(normal_cmd.clone()),
            Err(RosBridgeError::RosTransportUnavailable)
        ));

        {
            let mut interlock = node.interlock.lock().unwrap();
            assert_eq!(interlock.status(), symthaea_sim_bridge::SafetyStatus::Green);

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
            // A lower gain must not clear an already latched emergency stop.
            let cmd = normal_cmd.clone();
            assert!(matches!(
                node.publish_trajectory(cmd),
                Err(RosBridgeError::SafetyInterlock(_))
            ));
        }
    }

    #[cfg(feature = "ros2")]
    #[test]
    fn ros2_build_without_live_publisher_rejects_trajectory() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        // Simulate inconsistent lifecycle state without constructing ROS context.
        node.is_running = true;
        node.trajectory_pub = None;

        let command = RosTrajectoryCommand {
            joint_names: vec!["joint_1".into()],
            points: vec![RosTrajectoryPoint {
                positions: vec![0.0],
                velocities: Vec::new(),
                efforts: vec![0.0],
                time_from_start_ns: 1_000_000,
            }],
        };

        assert!(matches!(
            node.publish_trajectory(command),
            Err(RosBridgeError::RosTransportUnavailable)
        ));
    }

    #[cfg(not(feature = "ros2"))]
    #[test]
    fn telemetry_only_build_rejects_robot_trajectory_publication() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        // Isolate command-path behavior without starting a telemetry server.
        node.is_running = true;

        let command = RosTrajectoryCommand {
            joint_names: vec!["joint_1".into()],
            points: vec![RosTrajectoryPoint {
                positions: vec![0.0],
                velocities: Vec::new(),
                efforts: vec![0.0],
                time_from_start_ns: 1_000_000,
            }],
        };

        assert!(matches!(
            node.publish_trajectory(command),
            Err(RosBridgeError::RosTransportUnavailable)
        ));
    }

    #[test]
    fn invalid_and_estopped_trajectories_fail_closed() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        node.init().unwrap();

        let invalid = RosTrajectoryCommand {
            joint_names: vec!["joint_1".into(), "joint_2".into()],
            points: vec![RosTrajectoryPoint {
                positions: vec![f64::NAN, 0.0],
                velocities: Vec::new(),
                efforts: Vec::new(),
                time_from_start_ns: 1_000_000,
            }],
        };
        assert!(matches!(
            node.publish_trajectory(invalid),
            Err(RosBridgeError::InvalidTrajectory(_))
        ));

        node.interlock.lock().unwrap().trigger_emergency_stop();
        let command = RosTrajectoryCommand {
            joint_names: vec!["joint_1".into()],
            points: vec![RosTrajectoryPoint {
                positions: Vec::new(),
                velocities: Vec::new(),
                efforts: vec![0.0],
                time_from_start_ns: 1_000_000,
            }],
        };
        assert!(matches!(
            node.publish_trajectory(command),
            Err(RosBridgeError::SafetyInterlock(_))
        ));
    }

    #[test]
    fn non_finite_status_is_rejected_before_gain_scheduling() {
        let mut node = RosBridgeNode::new(RosBridgeConfig::default());
        node.init().unwrap();
        let result = node.publish_status(ConsciousnessStatusMsg {
            phi: f64::NAN,
            harmonies: [0.5; 8],
            neuromodulators: [0.5; 4],
            arousal: 0.5,
            uncertainty: 0.1,
            timestamp: "now".into(),
        });
        assert!(matches!(result, Err(RosBridgeError::InvalidStatus(_))));
    }

    #[test]
    fn joint_state_values_are_canonicalized_by_name_not_array_order() {
        let expected = ["hip", "knee", "ankle"];
        let first_names = vec!["hip".into(), "knee".into(), "ankle".into()];
        let first_values = vec![0.1, -0.2, 0.3];
        let shuffled_names = vec!["ankle".into(), "hip".into(), "knee".into()];
        let shuffled_values = vec![0.3, 0.1, -0.2];

        let first =
            canonicalize_joint_values(&first_names, &first_values, &expected, "position")
                .unwrap();
        let shuffled =
            canonicalize_joint_values(&shuffled_names, &shuffled_values, &expected, "position")
                .unwrap();

        assert_eq!(first, vec![0.1, -0.2, 0.3]);
        assert_eq!(shuffled, first);
    }

    #[test]
    fn joint_state_canonicalization_rejects_duplicate_missing_unknown_and_nonfinite_data() {
        let expected = ["hip", "knee", "ankle"];
        let values = vec![0.1, -0.2, 0.3];

        let duplicate = vec!["hip".into(), "hip".into(), "ankle".into()];
        assert!(matches!(
            canonicalize_joint_values(&duplicate, &values, &expected, "position"),
            Err(RosBridgeError::InvalidJointState(_))
        ));

        let missing = vec!["hip".into(), "knee".into()];
        assert!(matches!(
            canonicalize_joint_values(&missing, &values[..2], &expected, "position"),
            Err(RosBridgeError::InvalidJointState(_))
        ));

        let unknown = vec!["hip".into(), "knee".into(), "finger".into()];
        assert!(matches!(
            canonicalize_joint_values(&unknown, &values, &expected, "position"),
            Err(RosBridgeError::InvalidJointState(_))
        ));

        let nonfinite = vec![0.1, f64::NAN, 0.3];
        let names = vec!["hip".into(), "knee".into(), "ankle".into()];
        assert!(matches!(
            canonicalize_joint_values(&names, &nonfinite, &expected, "position"),
            Err(RosBridgeError::InvalidJointState(_))
        ));

        assert!(matches!(
            canonicalize_joint_values(&names, &values[..2], &expected, "position"),
            Err(RosBridgeError::InvalidJointState(_))
        ));
    }

    #[test]
    fn imu_quaternion_is_normalized_and_rejects_degenerate_or_nonfinite_values() {
        let normalized = normalize_imu_quaternion([2.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(normalized, [1.0, 0.0, 0.0, 0.0]);

        assert!(matches!(
            normalize_imu_quaternion([0.0, 0.0, 0.0, 0.0]),
            Err(RosBridgeError::InvalidSensorData(_))
        ));
        assert!(matches!(
            normalize_imu_quaternion([f64::NAN, 0.0, 0.0, 1.0]),
            Err(RosBridgeError::InvalidSensorData(_))
        ));
    }

    #[test]
    fn sensor_vectors_reject_nonfinite_components() {
        assert_eq!(
            validate_sensor_vector("angular velocity", [1.0, -2.0, 3.0]).unwrap(),
            [1.0, -2.0, 3.0]
        );
        assert!(matches!(
            validate_sensor_vector("angular velocity", [0.0, f64::INFINITY, 0.0]),
            Err(RosBridgeError::InvalidSensorData(_))
        ));
    }

    #[test]
    fn test_hdc_encoding_and_surprise() {
        let node = RosBridgeNode::new(RosBridgeConfig::default());
        let mut joints = vec![0.0f32; symthaea_humanoid::types::JOINT_NAMES.len()];
        joints[0] = 0.1;
        joints[1] = -0.2;
        joints[2] = 0.3;
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
            js.name = symthaea_humanoid::types::JOINT_NAMES
                .iter()
                .map(|name| (*name).to_string())
                .collect();
            js.position = joints.iter().map(|&value| value as f64).collect();
            js.velocity = vec![0.0; js.name.len()];
            *node.latest_joint_state.lock().unwrap() = Some(js);

            assert!(matches!(
                node.get_current_humanoid_state(),
                Err(RosBridgeError::SensorDataUnavailable(reason)) if reason.contains("IMU")
            ));

            let mut imu = sensor_msgs::msg::Imu::default();
            imu.orientation.w = 1.0;
            *node.latest_imu.lock().unwrap() = Some(imu);
            assert!(matches!(
                node.get_current_humanoid_state(),
                Err(RosBridgeError::SensorDataUnavailable(reason)) if reason.contains("linear velocity")
            ));

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
