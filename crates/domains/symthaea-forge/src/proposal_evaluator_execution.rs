// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Execution-evidence contract for the label-blind Forge evaluator protocol.
//!
//! This module deliberately separates a *precommitted launch policy* from a *self-reported success
//! record*. It does not itself launch a process and therefore does not prove that any OS sandbox,
//! namespace, seccomp profile, VM, cgroup, or network policy was actually enforced.
//!
//! The contract is still useful: it freezes resource bounds and transport semantics before launch,
//! binds exact serialized request/response bytes, and makes "runtime isolation not established" an
//! explicit field that cannot be silently upgraded by callers.

use crate::proposal_evaluator_protocol::{
    ForgeProposalEvaluationRequest, ForgeProposalEvaluationResponse,
    ForgeProposalEvaluatorProtocolError, ForgeProposalEvaluatorProtocolSpec,
};
use serde::Serialize;
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalEvaluatorExecutionError {
    #[error(transparent)]
    Protocol(#[from] ForgeProposalEvaluatorProtocolError),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
    #[error("evaluator launch policy resource bounds must all be greater than zero")]
    InvalidResourceBounds,
    #[error("evaluator launch policy identity does not match canonical fields")]
    PolicyIdentityMismatch,
    #[error("evaluator launch policy does not bind the supplied evaluator protocol")]
    PolicyProtocolMismatch,
    #[error("evaluator request or response exceeds the frozen transport-size limits")]
    TransportLimitExceeded,
    #[error("evaluator execution exceeded the frozen wall-time or stderr-size limit")]
    ExecutionLimitExceeded,
    #[error("evaluator execution record does not bind the supplied policy/request/response")]
    RecordScopeMismatch,
    #[error("evaluator execution record identity does not match canonical fields")]
    RecordIdentityMismatch,
}

/// Fixed semantics of the first execution-policy version.
///
/// These are requirements a later launcher must enforce. This module does not claim that they were
/// enforced merely because a policy object exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeProposalEvaluatorLaunchMode {
    /// Direct executable invocation; no shell command string is permitted.
    DirectExecNoShellV1,
}

impl ForgeProposalEvaluatorLaunchMode {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::DirectExecNoShellV1 => b"direct-exec-no-shell-v1",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeProposalEvaluatorEnvironmentMode {
    /// Start from an empty environment; a later launcher may add only explicitly precommitted values.
    ClearInheritedEnvironmentV1,
}

impl ForgeProposalEvaluatorEnvironmentMode {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::ClearInheritedEnvironmentV1 => b"clear-inherited-environment-v1",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeProposalEvaluatorWorkingDirectoryMode {
    /// Execute from a newly-created empty working directory.
    FreshEmptyDirectoryV1,
}

impl ForgeProposalEvaluatorWorkingDirectoryMode {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::FreshEmptyDirectoryV1 => b"fresh-empty-directory-v1",
        }
    }
}

/// Runtime-isolation conclusion carried by v1 execution evidence.
///
/// V1 has only one legal value. A later launcher-backed theorem must introduce a new version rather
/// than mutating this proposition into a stronger claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeProposalEvaluatorRuntimeIsolationStatus {
    NotEstablishedV1,
}

impl ForgeProposalEvaluatorRuntimeIsolationStatus {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::NotEstablishedV1 => b"not-established-v1",
        }
    }
}

/// Resource and process semantics frozen before an evaluator is launched.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalEvaluatorLaunchPolicy {
    id: ContentId,
    protocol_id: ContentId,
    launcher_implementation_id: ContentId,
    launcher_configuration_id: ContentId,
    launch_mode: ForgeProposalEvaluatorLaunchMode,
    environment_mode: ForgeProposalEvaluatorEnvironmentMode,
    working_directory_mode: ForgeProposalEvaluatorWorkingDirectoryMode,
    max_request_bytes: u64,
    max_response_bytes: u64,
    max_stderr_bytes: u64,
    max_wall_time_ms: u64,
}

impl ForgeProposalEvaluatorLaunchPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn precommit(
        protocol: &ForgeProposalEvaluatorProtocolSpec,
        launcher_implementation_id: ContentId,
        launcher_configuration_id: ContentId,
        max_request_bytes: u64,
        max_response_bytes: u64,
        max_stderr_bytes: u64,
        max_wall_time_ms: u64,
    ) -> Result<Self, ForgeProposalEvaluatorExecutionError> {
        protocol.validate()?;
        if max_request_bytes == 0
            || max_response_bytes == 0
            || max_stderr_bytes == 0
            || max_wall_time_ms == 0
        {
            return Err(ForgeProposalEvaluatorExecutionError::InvalidResourceBounds);
        }
        let launch_mode = ForgeProposalEvaluatorLaunchMode::DirectExecNoShellV1;
        let environment_mode = ForgeProposalEvaluatorEnvironmentMode::ClearInheritedEnvironmentV1;
        let working_directory_mode = ForgeProposalEvaluatorWorkingDirectoryMode::FreshEmptyDirectoryV1;
        let id = derive_policy_id(
            protocol.id(),
            &launcher_implementation_id,
            &launcher_configuration_id,
            launch_mode,
            environment_mode,
            working_directory_mode,
            max_request_bytes,
            max_response_bytes,
            max_stderr_bytes,
            max_wall_time_ms,
        );
        Ok(Self {
            id,
            protocol_id: protocol.id().clone(),
            launcher_implementation_id,
            launcher_configuration_id,
            launch_mode,
            environment_mode,
            working_directory_mode,
            max_request_bytes,
            max_response_bytes,
            max_stderr_bytes,
            max_wall_time_ms,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn protocol_id(&self) -> &ContentId { &self.protocol_id }
    pub fn launcher_implementation_id(&self) -> &ContentId { &self.launcher_implementation_id }
    pub fn launcher_configuration_id(&self) -> &ContentId { &self.launcher_configuration_id }
    pub fn launch_mode(&self) -> ForgeProposalEvaluatorLaunchMode { self.launch_mode }
    pub fn environment_mode(&self) -> ForgeProposalEvaluatorEnvironmentMode { self.environment_mode }
    pub fn working_directory_mode(&self) -> ForgeProposalEvaluatorWorkingDirectoryMode {
        self.working_directory_mode
    }
    pub fn max_request_bytes(&self) -> u64 { self.max_request_bytes }
    pub fn max_response_bytes(&self) -> u64 { self.max_response_bytes }
    pub fn max_stderr_bytes(&self) -> u64 { self.max_stderr_bytes }
    pub fn max_wall_time_ms(&self) -> u64 { self.max_wall_time_ms }

    pub fn validate_for(
        &self,
        protocol: &ForgeProposalEvaluatorProtocolSpec,
    ) -> Result<(), ForgeProposalEvaluatorExecutionError> {
        protocol.validate()?;
        if self.protocol_id != *protocol.id()
            || self.max_request_bytes == 0
            || self.max_response_bytes == 0
            || self.max_stderr_bytes == 0
            || self.max_wall_time_ms == 0
        {
            return Err(ForgeProposalEvaluatorExecutionError::PolicyProtocolMismatch);
        }
        let expected = derive_policy_id(
            &self.protocol_id,
            &self.launcher_implementation_id,
            &self.launcher_configuration_id,
            self.launch_mode,
            self.environment_mode,
            self.working_directory_mode,
            self.max_request_bytes,
            self.max_response_bytes,
            self.max_stderr_bytes,
            self.max_wall_time_ms,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalEvaluatorExecutionError::PolicyIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_policy_id(
    protocol_id: &ContentId,
    launcher_implementation_id: &ContentId,
    launcher_configuration_id: &ContentId,
    launch_mode: ForgeProposalEvaluatorLaunchMode,
    environment_mode: ForgeProposalEvaluatorEnvironmentMode,
    working_directory_mode: ForgeProposalEvaluatorWorkingDirectoryMode,
    max_request_bytes: u64,
    max_response_bytes: u64,
    max_stderr_bytes: u64,
    max_wall_time_ms: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-evaluator-launch-policy.v1",
        [
            protocol_id.as_str().as_bytes(),
            launcher_implementation_id.as_str().as_bytes(),
            launcher_configuration_id.as_str().as_bytes(),
            launch_mode.tag(),
            environment_mode.tag(),
            working_directory_mode.tag(),
            max_request_bytes.to_be_bytes().as_slice(),
            max_response_bytes.to_be_bytes().as_slice(),
            max_stderr_bytes.to_be_bytes().as_slice(),
            max_wall_time_ms.to_be_bytes().as_slice(),
        ],
    )
}

/// Self-reported successful execution record under one frozen launch policy.
///
/// This record is descriptive evidence only. In particular, `runtime_isolation_status` is fixed to
/// `NotEstablishedV1`; callers cannot use v1 to assert stronger isolation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalEvaluatorExecutionRecord {
    id: ContentId,
    policy_id: ContentId,
    protocol_id: ContentId,
    request_id: ContentId,
    response_id: ContentId,
    request_wire_id: ContentId,
    response_wire_id: ContentId,
    execution_context_id: ContentId,
    launcher_evidence_id: ContentId,
    stderr_artifact_id: ContentId,
    stderr_bytes: u64,
    wall_time_ms: u64,
    runtime_isolation_status: ForgeProposalEvaluatorRuntimeIsolationStatus,
}

impl ForgeProposalEvaluatorExecutionRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn record_success(
        policy: &ForgeProposalEvaluatorLaunchPolicy,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
        launcher_evidence_id: ContentId,
        stderr_artifact_id: ContentId,
        stderr_bytes: u64,
        wall_time_ms: u64,
    ) -> Result<Self, ForgeProposalEvaluatorExecutionError> {
        policy.validate_for(request.protocol())?;
        response.validate_for(request)?;
        let request_wire = serde_json::to_vec(request)?;
        let response_wire = serde_json::to_vec(response)?;
        let request_len = u64::try_from(request_wire.len())
            .map_err(|_| ForgeProposalEvaluatorExecutionError::TransportLimitExceeded)?;
        let response_len = u64::try_from(response_wire.len())
            .map_err(|_| ForgeProposalEvaluatorExecutionError::TransportLimitExceeded)?;
        if request_len > policy.max_request_bytes() || response_len > policy.max_response_bytes() {
            return Err(ForgeProposalEvaluatorExecutionError::TransportLimitExceeded);
        }
        if stderr_bytes > policy.max_stderr_bytes() || wall_time_ms > policy.max_wall_time_ms() {
            return Err(ForgeProposalEvaluatorExecutionError::ExecutionLimitExceeded);
        }
        let request_wire_id = ContentId::derive(
            "symthaea.forge-proposal-evaluator-request-wire.v1",
            [request_wire.as_slice()],
        );
        let response_wire_id = ContentId::derive(
            "symthaea.forge-proposal-evaluator-response-wire.v1",
            [response_wire.as_slice()],
        );
        let runtime_isolation_status = ForgeProposalEvaluatorRuntimeIsolationStatus::NotEstablishedV1;
        let id = derive_record_id(
            policy.id(),
            request.protocol().id(),
            request.id(),
            response.id(),
            &request_wire_id,
            &response_wire_id,
            response.execution_context_id(),
            &launcher_evidence_id,
            &stderr_artifact_id,
            stderr_bytes,
            wall_time_ms,
            runtime_isolation_status,
        );
        Ok(Self {
            id,
            policy_id: policy.id().clone(),
            protocol_id: request.protocol().id().clone(),
            request_id: request.id().clone(),
            response_id: response.id().clone(),
            request_wire_id,
            response_wire_id,
            execution_context_id: response.execution_context_id().clone(),
            launcher_evidence_id,
            stderr_artifact_id,
            stderr_bytes,
            wall_time_ms,
            runtime_isolation_status,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn policy_id(&self) -> &ContentId { &self.policy_id }
    pub fn request_id(&self) -> &ContentId { &self.request_id }
    pub fn response_id(&self) -> &ContentId { &self.response_id }
    pub fn request_wire_id(&self) -> &ContentId { &self.request_wire_id }
    pub fn response_wire_id(&self) -> &ContentId { &self.response_wire_id }
    pub fn execution_context_id(&self) -> &ContentId { &self.execution_context_id }
    pub fn launcher_evidence_id(&self) -> &ContentId { &self.launcher_evidence_id }
    pub fn stderr_artifact_id(&self) -> &ContentId { &self.stderr_artifact_id }
    pub fn stderr_bytes(&self) -> u64 { self.stderr_bytes }
    pub fn wall_time_ms(&self) -> u64 { self.wall_time_ms }
    pub fn runtime_isolation_status(&self) -> ForgeProposalEvaluatorRuntimeIsolationStatus {
        self.runtime_isolation_status
    }

    pub fn validate_for(
        &self,
        policy: &ForgeProposalEvaluatorLaunchPolicy,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
    ) -> Result<(), ForgeProposalEvaluatorExecutionError> {
        policy.validate_for(request.protocol())?;
        response.validate_for(request)?;
        if self.policy_id != *policy.id()
            || self.protocol_id != *request.protocol().id()
            || self.request_id != *request.id()
            || self.response_id != *response.id()
            || self.execution_context_id != *response.execution_context_id()
            || self.stderr_bytes > policy.max_stderr_bytes()
            || self.wall_time_ms > policy.max_wall_time_ms()
            || self.runtime_isolation_status
                != ForgeProposalEvaluatorRuntimeIsolationStatus::NotEstablishedV1
        {
            return Err(ForgeProposalEvaluatorExecutionError::RecordScopeMismatch);
        }
        let request_wire = serde_json::to_vec(request)?;
        let response_wire = serde_json::to_vec(response)?;
        let request_len = u64::try_from(request_wire.len())
            .map_err(|_| ForgeProposalEvaluatorExecutionError::TransportLimitExceeded)?;
        let response_len = u64::try_from(response_wire.len())
            .map_err(|_| ForgeProposalEvaluatorExecutionError::TransportLimitExceeded)?;
        if request_len > policy.max_request_bytes() || response_len > policy.max_response_bytes() {
            return Err(ForgeProposalEvaluatorExecutionError::TransportLimitExceeded);
        }
        let expected_request_wire_id = ContentId::derive(
            "symthaea.forge-proposal-evaluator-request-wire.v1",
            [request_wire.as_slice()],
        );
        let expected_response_wire_id = ContentId::derive(
            "symthaea.forge-proposal-evaluator-response-wire.v1",
            [response_wire.as_slice()],
        );
        if self.request_wire_id != expected_request_wire_id
            || self.response_wire_id != expected_response_wire_id
        {
            return Err(ForgeProposalEvaluatorExecutionError::RecordScopeMismatch);
        }
        let expected = derive_record_id(
            &self.policy_id,
            &self.protocol_id,
            &self.request_id,
            &self.response_id,
            &self.request_wire_id,
            &self.response_wire_id,
            &self.execution_context_id,
            &self.launcher_evidence_id,
            &self.stderr_artifact_id,
            self.stderr_bytes,
            self.wall_time_ms,
            self.runtime_isolation_status,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalEvaluatorExecutionError::RecordIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_record_id(
    policy_id: &ContentId,
    protocol_id: &ContentId,
    request_id: &ContentId,
    response_id: &ContentId,
    request_wire_id: &ContentId,
    response_wire_id: &ContentId,
    execution_context_id: &ContentId,
    launcher_evidence_id: &ContentId,
    stderr_artifact_id: &ContentId,
    stderr_bytes: u64,
    wall_time_ms: u64,
    isolation_status: ForgeProposalEvaluatorRuntimeIsolationStatus,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-evaluator-execution-record.v1",
        [
            policy_id.as_str().as_bytes(),
            protocol_id.as_str().as_bytes(),
            request_id.as_str().as_bytes(),
            response_id.as_str().as_bytes(),
            request_wire_id.as_str().as_bytes(),
            response_wire_id.as_str().as_bytes(),
            execution_context_id.as_str().as_bytes(),
            launcher_evidence_id.as_str().as_bytes(),
            stderr_artifact_id.as_str().as_bytes(),
            stderr_bytes.to_be_bytes().as_slice(),
            wall_time_ms.to_be_bytes().as_slice(),
            isolation_status.tag(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    #[test]
    fn launch_policy_identity_binds_resource_limits() {
        let protocol = ForgeProposalEvaluatorProtocolSpec::new(
            cid("runner", "runner"),
            cid("runner-config", "config"),
            cid("transport", "schema"),
        )
        .unwrap();
        let a = ForgeProposalEvaluatorLaunchPolicy::precommit(
            &protocol,
            cid("launcher", "impl"),
            cid("launcher-config", "config"),
            1024,
            2048,
            512,
            10_000,
        )
        .unwrap();
        let b = ForgeProposalEvaluatorLaunchPolicy::precommit(
            &protocol,
            cid("launcher", "impl"),
            cid("launcher-config", "config"),
            1024,
            2048,
            512,
            20_000,
        )
        .unwrap();
        assert_ne!(a.id(), b.id());
        a.validate_for(&protocol).unwrap();
    }

    #[test]
    fn v1_cannot_claim_runtime_isolation() {
        assert_eq!(
            ForgeProposalEvaluatorRuntimeIsolationStatus::NotEstablishedV1.tag(),
            b"not-established-v1"
        );
    }
}
