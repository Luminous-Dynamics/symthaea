// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned one-shot protocol for an independently built MuJoCo oracle worker.
//!
//! The candidate process never imports an oracle snapshot from its own runtime.
//! Instead it launches a separately identified executable, supplies an exact
//! generalized state, and independently revalidates the returned structure and
//! state fingerprint before dataset admission.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::floating_base::FloatingBaseDynamicsSnapshot;
use crate::morphology::HumanoidMorphology;
use crate::oracle_dataset::fingerprint_state;

pub const MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MujocoOracleWorkerRequest {
    pub protocol_version: u32,
    pub request_id: u64,
    pub generator_id: String,
    pub generator_build_id: String,
    pub candidate_build_id: String,
    pub engine_id: String,
    pub morphology: HumanoidMorphology,
    pub generalized_position: Vec<f64>,
    pub generalized_velocity: Vec<f64>,
    pub actuator_command: Vec<f64>,
    pub state_fingerprint: u64,
}

impl MujocoOracleWorkerRequest {
    pub fn validate(&self) -> bool {
        let actuators = self.morphology.num_actuators();
        self.protocol_version == MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION
            && self.request_id != 0
            && !self.generator_id.trim().is_empty()
            && !self.generator_build_id.trim().is_empty()
            && !self.candidate_build_id.trim().is_empty()
            && self.generator_build_id != self.candidate_build_id
            && !self.engine_id.trim().is_empty()
            && self.generalized_position.len() == 7 + actuators
            && self.generalized_velocity.len() == 6 + actuators
            && self.actuator_command.len() == actuators
            && self
                .generalized_position
                .iter()
                .chain(self.generalized_velocity.iter())
                .chain(self.actuator_command.iter())
                .all(|value| value.is_finite())
            && fingerprint_state(
                &self.generalized_position,
                &self.generalized_velocity,
                &self.actuator_command,
            ) == self.state_fingerprint
            && root_quaternion_is_unit(&self.generalized_position)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MujocoOracleWorkerResponse {
    pub protocol_version: u32,
    pub request_id: u64,
    pub generator_id: String,
    pub generator_build_id: String,
    pub engine_id: String,
    pub state_fingerprint: u64,
    pub oracle: Option<FloatingBaseDynamicsSnapshot>,
    pub error: Option<String>,
}

impl MujocoOracleWorkerResponse {
    pub fn success(
        request: &MujocoOracleWorkerRequest,
        oracle: FloatingBaseDynamicsSnapshot,
    ) -> Self {
        Self {
            protocol_version: MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION,
            request_id: request.request_id,
            generator_id: request.generator_id.clone(),
            generator_build_id: request.generator_build_id.clone(),
            engine_id: request.engine_id.clone(),
            state_fingerprint: request.state_fingerprint,
            oracle: Some(oracle),
            error: None,
        }
    }

    pub fn failure(request: &MujocoOracleWorkerRequest, error: impl Into<String>) -> Self {
        Self {
            protocol_version: MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION,
            request_id: request.request_id,
            generator_id: request.generator_id.clone(),
            generator_build_id: request.generator_build_id.clone(),
            engine_id: request.engine_id.clone(),
            state_fingerprint: request.state_fingerprint,
            oracle: None,
            error: Some(error.into()),
        }
    }

    pub fn validate_against(&self, request: &MujocoOracleWorkerRequest) -> bool {
        if !request.validate()
            || self.protocol_version != MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION
            || self.request_id != request.request_id
            || self.generator_id != request.generator_id
            || self.generator_build_id != request.generator_build_id
            || self.engine_id != request.engine_id
            || self.state_fingerprint != request.state_fingerprint
            || self.error.is_some()
        {
            return false;
        }
        self.oracle
            .as_ref()
            .map(|oracle| {
                oracle.validate()
                    && oracle.morphology == request.morphology
                    && oracle.generalized_velocity_count == request.generalized_velocity.len()
            })
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone)]
pub struct ProcessMujocoOracleGenerator {
    executable: PathBuf,
    arguments: Vec<String>,
    maximum_response_bytes: usize,
}

impl ProcessMujocoOracleGenerator {
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            arguments: Vec::new(),
            maximum_response_bytes: 64 * 1024 * 1024,
        }
    }

    pub fn with_arguments(mut self, arguments: impl IntoIterator<Item = String>) -> Self {
        self.arguments = arguments.into_iter().collect();
        self
    }

    pub fn with_maximum_response_bytes(mut self, maximum_response_bytes: usize) -> Self {
        self.maximum_response_bytes = maximum_response_bytes.max(1024);
        self
    }

    pub fn executable(&self) -> &Path {
        &self.executable
    }

    pub fn generate(
        &self,
        request: &MujocoOracleWorkerRequest,
    ) -> Option<MujocoOracleWorkerResponse> {
        if self.executable.as_os_str().is_empty() || !request.validate() {
            return None;
        }
        let payload = serde_json::to_vec(request).ok()?;
        let mut child = Command::new(&self.executable)
            .args(&self.arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .ok()?;
        {
            let mut stdin = child.stdin.take()?;
            stdin.write_all(&payload).ok()?;
            stdin.write_all(b"\n").ok()?;
        }
        let output = child.wait_with_output().ok()?;
        if !output.status.success()
            || output.stdout.is_empty()
            || output.stdout.len() > self.maximum_response_bytes
        {
            return None;
        }
        let response: MujocoOracleWorkerResponse = serde_json::from_slice(&output.stdout).ok()?;
        response.validate_against(request).then_some(response)
    }
}

fn root_quaternion_is_unit(position: &[f64]) -> bool {
    if position.len() < 7 {
        return false;
    }
    let norm_squared = position[3..7]
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    norm_squared.is_finite() && (norm_squared - 1.0).abs() <= 1.0e-6
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> MujocoOracleWorkerRequest {
        let morphology = HumanoidMorphology::Dmc21;
        let mut generalized_position = vec![0.0; 7 + morphology.num_actuators()];
        generalized_position[2] = 1.35;
        generalized_position[3] = 1.0;
        let generalized_velocity = vec![0.0; 6 + morphology.num_actuators()];
        let actuator_command = vec![0.0; morphology.num_actuators()];
        let state_fingerprint = fingerprint_state(
            &generalized_position,
            &generalized_velocity,
            &actuator_command,
        );
        MujocoOracleWorkerRequest {
            protocol_version: MUJOCO_ORACLE_WORKER_PROTOCOL_VERSION,
            request_id: 1,
            generator_id: "independent-mujoco-oracle".to_string(),
            generator_build_id: "oracle-build".to_string(),
            candidate_build_id: "candidate-build".to_string(),
            engine_id: "mujoco-3.x".to_string(),
            morphology,
            generalized_position,
            generalized_velocity,
            actuator_command,
            state_fingerprint,
        }
    }

    #[test]
    fn self_oracle_identity_is_rejected() {
        let mut request = request();
        request.candidate_build_id = request.generator_build_id.clone();
        assert!(!request.validate());
    }

    #[test]
    fn non_unit_root_quaternion_is_rejected() {
        let mut request = request();
        request.generalized_position[3] = 2.0;
        request.state_fingerprint = fingerprint_state(
            &request.generalized_position,
            &request.generalized_velocity,
            &request.actuator_command,
        );
        assert!(!request.validate());
    }
}
