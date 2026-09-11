// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Linux bubblewrap isolation backend for label-blind Forge evaluator execution.
//!
//! This is a stronger theorem than the direct-process launcher. On Linux, the backend executes the
//! exact frozen executable-model payload inside a bubblewrap-created mount/user/PID/IPC/UTS/network
//! namespace set. The sandbox sees a read-only `/nix/store`, the exact model executable mounted
//! read-only at `/runner`, a fresh tmpfs `/tmp`, isolated `/proc`, and bubblewrap's minimal `/dev`.
//! The inherited environment is cleared both by the parent process and inside bubblewrap.
//!
//! The backend deliberately does **not** claim seccomp filtering, Landlock policy, or cgroup CPU /
//! memory enforcement. Those remain explicit `NotEstablishedV1` fields in the receipt. The parent
//! still enforces bounded request/response/stderr sizes and monotonic wall time.

use crate::proposal_evaluator_launcher::{
    forge_direct_evaluator_transport_schema_id, forge_evaluator_runner_argv_id,
    forge_evaluator_runner_artifact_id,
};
use crate::proposal_evaluator_protocol::{
    ForgeProposalEvaluationRequest, ForgeProposalEvaluationResponse,
    ForgeProposalEvaluatorPrediction, ForgeProposalEvaluatorProtocolError,
};
use crate::proposal_executable_model::{
    ForgeProposalExecutableModelBinding, ForgeProposalExecutableModelError,
};
use crate::proposal_model::ForgeProposalFrozenModel;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use symthaea_algorithms::ContentId;
use thiserror::Error;

const WIRE_RESPONSE_SCHEMA_V1: &str = "symthaea-forge-evaluator-response-v1";
const NIX_STORE: &str = "/nix/store";
const SANDBOX_RUNNER: &str = "/runner";
const SANDBOX_TMP: &str = "/tmp";
const HARD_MAX_REQUEST_BYTES_V1: u64 = 256 * 1024 * 1024;
const HARD_MAX_RESPONSE_BYTES_V1: u64 = 256 * 1024 * 1024;
const HARD_MAX_STDERR_BYTES_V1: u64 = 64 * 1024 * 1024;
const HARD_MAX_WALL_TIME_MS_V1: u64 = 60 * 60 * 1000;
const HARD_MAX_ARTIFACT_BYTES_V1: u64 = 512 * 1024 * 1024;
const POLL_INTERVAL_MS: u64 = 10;

static WORKDIR_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum ForgeProposalLinuxIsolationError {
    #[error(transparent)]
    Protocol(#[from] ForgeProposalEvaluatorProtocolError),
    #[error(transparent)]
    ExecutableModel(#[from] ForgeProposalExecutableModelError),
    #[error("bubblewrap evaluator isolation is supported only on Linux")]
    UnsupportedPlatform,
    #[error("bubblewrap isolation IO error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("bubblewrap or model executable is not a regular file")]
    InvalidExecutable,
    #[error("bubblewrap or model artifact exceeds the hard v1 artifact-size ceiling")]
    ArtifactTooLarge,
    #[error("bubblewrap/model executable path is not valid UTF-8")]
    NonUtf8Path,
    #[error("bubblewrap artifact bytes do not match the frozen isolation policy")]
    BubblewrapArtifactMismatch,
    #[error("model executable bytes do not match the frozen executable-model payload")]
    ModelArtifactMismatch,
    #[error("model argv does not match the frozen evaluator protocol runner configuration")]
    ModelArgvMismatch,
    #[error("evaluator protocol does not use the required v1 label-blind transport schema")]
    TransportSchemaMismatch,
    #[error("bubblewrap isolation policy exceeds hard v1 safety ceilings")]
    UnsafePolicyLimits,
    #[error("evaluator request exceeds the frozen request-size limit")]
    RequestTooLarge,
    #[error("isolated evaluator process could not expose all required stdio pipes")]
    MissingPipe,
    #[error("isolated evaluator stdin writer failed or panicked")]
    StdinWriterFailed,
    #[error("isolated evaluator stdout/stderr reader thread panicked")]
    ReaderThreadPanicked,
    #[error("isolated evaluator exceeded the frozen wall-time limit")]
    TimedOut,
    #[error("isolated evaluator stdout exceeded the frozen response-size limit")]
    StdoutLimitExceeded,
    #[error("isolated evaluator stderr exceeded the frozen stderr-size limit")]
    StderrLimitExceeded,
    #[error("bubblewrap/evaluator exited unsuccessfully with code {code:?}")]
    RunnerFailed { code: Option<i32> },
    #[error("isolated evaluator stdout is not valid v1 response JSON: {0}")]
    InvalidResponseJson(#[from] serde_json::Error),
    #[error("isolated evaluator wire response does not match request/protocol identity")]
    WireScopeMismatch,
    #[error("isolated evaluator response contains duplicate, missing, or unknown targets")]
    WirePredictionCoverageMismatch,
    #[error("isolated evaluator execution-context string must be non-empty")]
    EmptyExecutionContext,
    #[error("isolation measurement cannot be represented in u64")]
    MeasurementOverflow,
    #[error("Linux isolation receipt does not bind the supplied policy/model/request/response")]
    ReceiptScopeMismatch,
    #[error("Linux isolation receipt identity does not match canonical fields")]
    ReceiptIdentityMismatch,
}

/// Exact artifact identity for the bubblewrap binary used by this backend.
pub fn forge_bubblewrap_artifact_id(bytes: &[u8]) -> ContentId {
    ContentId::derive("symthaea.forge-bubblewrap-artifact.v1", [bytes])
}

/// Fixed semantic identity of the namespace/filesystem recipe implemented by this module.
pub fn forge_bubblewrap_isolation_recipe_id() -> ContentId {
    ContentId::derive(
        "symthaea.forge-bubblewrap-isolation-recipe.v1",
        [b"user+pid+ipc+uts+net namespaces;ro-nix-store;ro-runner;proc;minimal-dev;tmpfs-tmp;clearenv;new-session;die-with-parent".as_slice()],
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ForgeProposalIsolationState {
    EstablishedV1,
    NotEstablishedV1,
}

/// Granular claims produced by a successful bubblewrap-backed evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ForgeProposalLinuxIsolationStatus {
    user_namespace: ForgeProposalIsolationState,
    pid_namespace: ForgeProposalIsolationState,
    ipc_namespace: ForgeProposalIsolationState,
    uts_namespace: ForgeProposalIsolationState,
    network_namespace: ForgeProposalIsolationState,
    restricted_mount_namespace: ForgeProposalIsolationState,
    seccomp_filter: ForgeProposalIsolationState,
    landlock: ForgeProposalIsolationState,
    cgroup_resource_enforcement: ForgeProposalIsolationState,
}

impl ForgeProposalLinuxIsolationStatus {
    fn bubblewrap_v1() -> Self {
        Self {
            user_namespace: ForgeProposalIsolationState::EstablishedV1,
            pid_namespace: ForgeProposalIsolationState::EstablishedV1,
            ipc_namespace: ForgeProposalIsolationState::EstablishedV1,
            uts_namespace: ForgeProposalIsolationState::EstablishedV1,
            network_namespace: ForgeProposalIsolationState::EstablishedV1,
            restricted_mount_namespace: ForgeProposalIsolationState::EstablishedV1,
            seccomp_filter: ForgeProposalIsolationState::NotEstablishedV1,
            landlock: ForgeProposalIsolationState::NotEstablishedV1,
            cgroup_resource_enforcement: ForgeProposalIsolationState::NotEstablishedV1,
        }
    }

    pub fn user_namespace(&self) -> ForgeProposalIsolationState { self.user_namespace }
    pub fn pid_namespace(&self) -> ForgeProposalIsolationState { self.pid_namespace }
    pub fn ipc_namespace(&self) -> ForgeProposalIsolationState { self.ipc_namespace }
    pub fn uts_namespace(&self) -> ForgeProposalIsolationState { self.uts_namespace }
    pub fn network_namespace(&self) -> ForgeProposalIsolationState { self.network_namespace }
    pub fn restricted_mount_namespace(&self) -> ForgeProposalIsolationState {
        self.restricted_mount_namespace
    }
    pub fn seccomp_filter(&self) -> ForgeProposalIsolationState { self.seccomp_filter }
    pub fn landlock(&self) -> ForgeProposalIsolationState { self.landlock }
    pub fn cgroup_resource_enforcement(&self) -> ForgeProposalIsolationState {
        self.cgroup_resource_enforcement
    }
}

/// Frozen Linux isolation policy. The bubblewrap artifact is exact; the namespace/mount recipe is
/// fixed by `forge_bubblewrap_isolation_recipe_id()` rather than supplied by callers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalBubblewrapIsolationPolicy {
    id: ContentId,
    bubblewrap_artifact_id: ContentId,
    isolation_recipe_id: ContentId,
    transport_schema_id: ContentId,
    max_request_bytes: u64,
    max_response_bytes: u64,
    max_stderr_bytes: u64,
    max_wall_time_ms: u64,
}

impl ForgeProposalBubblewrapIsolationPolicy {
    pub fn new(
        bubblewrap_artifact_id: ContentId,
        max_request_bytes: u64,
        max_response_bytes: u64,
        max_stderr_bytes: u64,
        max_wall_time_ms: u64,
    ) -> Result<Self, ForgeProposalLinuxIsolationError> {
        if max_request_bytes == 0
            || max_response_bytes == 0
            || max_stderr_bytes == 0
            || max_wall_time_ms == 0
            || max_request_bytes > HARD_MAX_REQUEST_BYTES_V1
            || max_response_bytes > HARD_MAX_RESPONSE_BYTES_V1
            || max_stderr_bytes > HARD_MAX_STDERR_BYTES_V1
            || max_wall_time_ms > HARD_MAX_WALL_TIME_MS_V1
        {
            return Err(ForgeProposalLinuxIsolationError::UnsafePolicyLimits);
        }
        let isolation_recipe_id = forge_bubblewrap_isolation_recipe_id();
        let transport_schema_id = forge_direct_evaluator_transport_schema_id();
        let id = derive_policy_id(
            &bubblewrap_artifact_id,
            &isolation_recipe_id,
            &transport_schema_id,
            max_request_bytes,
            max_response_bytes,
            max_stderr_bytes,
            max_wall_time_ms,
        );
        Ok(Self {
            id,
            bubblewrap_artifact_id,
            isolation_recipe_id,
            transport_schema_id,
            max_request_bytes,
            max_response_bytes,
            max_stderr_bytes,
            max_wall_time_ms,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn bubblewrap_artifact_id(&self) -> &ContentId { &self.bubblewrap_artifact_id }
    pub fn isolation_recipe_id(&self) -> &ContentId { &self.isolation_recipe_id }
    pub fn transport_schema_id(&self) -> &ContentId { &self.transport_schema_id }
    pub fn max_request_bytes(&self) -> u64 { self.max_request_bytes }
    pub fn max_response_bytes(&self) -> u64 { self.max_response_bytes }
    pub fn max_stderr_bytes(&self) -> u64 { self.max_stderr_bytes }
    pub fn max_wall_time_ms(&self) -> u64 { self.max_wall_time_ms }

    pub fn validate(&self) -> Result<(), ForgeProposalLinuxIsolationError> {
        let rebuilt = Self::new(
            self.bubblewrap_artifact_id.clone(),
            self.max_request_bytes,
            self.max_response_bytes,
            self.max_stderr_bytes,
            self.max_wall_time_ms,
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ForgeProposalLinuxIsolationError::ReceiptIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_policy_id(
    bubblewrap_artifact_id: &ContentId,
    isolation_recipe_id: &ContentId,
    transport_schema_id: &ContentId,
    max_request_bytes: u64,
    max_response_bytes: u64,
    max_stderr_bytes: u64,
    max_wall_time_ms: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-bubblewrap-isolation-policy.v1",
        [
            bubblewrap_artifact_id.as_str().as_bytes(),
            isolation_recipe_id.as_str().as_bytes(),
            transport_schema_id.as_str().as_bytes(),
            max_request_bytes.to_be_bytes().as_slice(),
            max_response_bytes.to_be_bytes().as_slice(),
            max_stderr_bytes.to_be_bytes().as_slice(),
            max_wall_time_ms.to_be_bytes().as_slice(),
        ],
    )
}

#[derive(Debug, Deserialize)]
struct WirePredictionV1 {
    target_id: String,
    probability_scaled: u64,
}

#[derive(Debug, Deserialize)]
struct WireResponseV1 {
    schema: String,
    request_id: String,
    protocol_id: String,
    execution_context: String,
    predictions: Vec<WirePredictionV1>,
}

#[derive(Debug)]
struct CappedOutput {
    bytes: Vec<u8>,
    total_bytes: u64,
    exceeded: bool,
}

fn read_capped_and_drain<R: Read>(mut reader: R, cap: u64) -> Result<CappedOutput, std::io::Error> {
    let mut bytes = Vec::new();
    let mut total_bytes = 0u64;
    let mut exceeded = false;
    let mut buffer = [0u8; 8192];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        total_bytes = total_bytes.saturating_add(read as u64);
        let remaining = cap.saturating_sub(bytes.len() as u64);
        if remaining > 0 {
            let keep = usize::try_from(remaining.min(read as u64)).unwrap_or(read);
            bytes.extend_from_slice(&buffer[..keep]);
        }
        if total_bytes > cap {
            exceeded = true;
        }
    }
    Ok(CappedOutput { bytes, total_bytes, exceeded })
}

struct FreshWorkDir { path: PathBuf }

impl FreshWorkDir {
    fn create(request_id: &ContentId) -> Result<Self, ForgeProposalLinuxIsolationError> {
        let epoch_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let nonce = WORKDIR_NONCE.fetch_add(1, Ordering::Relaxed);
        let request_tag = request_id
            .as_str()
            .bytes()
            .filter(|byte| byte.is_ascii_alphanumeric())
            .take(16)
            .map(char::from)
            .collect::<String>();
        for attempt in 0u64..64 {
            let path = std::env::temp_dir().join(format!(
                "symthaea-forge-bwrap-{}-{epoch_nanos}-{nonce}-{attempt}-{request_tag}",
                std::process::id(),
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(source) => return Err(ForgeProposalLinuxIsolationError::Io { path, source }),
            }
        }
        Err(ForgeProposalLinuxIsolationError::Io {
            path: std::env::temp_dir(),
            source: std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not allocate unique bubblewrap evaluator working directory",
            ),
        })
    }
}

impl Drop for FreshWorkDir {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.path); }
}

fn canonical_file_bytes(path: &Path) -> Result<(PathBuf, Vec<u8>), ForgeProposalLinuxIsolationError> {
    let canonical = path.canonicalize().map_err(|source| ForgeProposalLinuxIsolationError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = fs::metadata(&canonical).map_err(|source| ForgeProposalLinuxIsolationError::Io {
        path: canonical.clone(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(ForgeProposalLinuxIsolationError::InvalidExecutable);
    }
    if metadata.len() > HARD_MAX_ARTIFACT_BYTES_V1 {
        return Err(ForgeProposalLinuxIsolationError::ArtifactTooLarge);
    }
    let bytes = fs::read(&canonical).map_err(|source| ForgeProposalLinuxIsolationError::Io {
        path: canonical.clone(),
        source,
    })?;
    Ok((canonical, bytes))
}

fn build_bwrap_args(
    runner_path: &Path,
    model_args: &[String],
) -> Result<Vec<String>, ForgeProposalLinuxIsolationError> {
    let runner_source = runner_path.to_str().ok_or(ForgeProposalLinuxIsolationError::NonUtf8Path)?;
    let mut args = vec![
        "--die-with-parent".to_string(),
        "--new-session".to_string(),
        "--unshare-user".to_string(),
        "--unshare-pid".to_string(),
        "--unshare-uts".to_string(),
        "--unshare-ipc".to_string(),
        "--unshare-net".to_string(),
        "--proc".to_string(),
        "/proc".to_string(),
        "--dev".to_string(),
        "/dev".to_string(),
        "--tmpfs".to_string(),
        SANDBOX_TMP.to_string(),
        "--ro-bind".to_string(),
        NIX_STORE.to_string(),
        NIX_STORE.to_string(),
        "--ro-bind".to_string(),
        runner_source.to_string(),
        SANDBOX_RUNNER.to_string(),
        "--chdir".to_string(),
        SANDBOX_TMP.to_string(),
        "--clearenv".to_string(),
        "--".to_string(),
        SANDBOX_RUNNER.to_string(),
    ];
    args.extend(model_args.iter().cloned());
    Ok(args)
}

fn derive_bwrap_argv_id(args: &[String]) -> ContentId {
    let count = (args.len() as u64).to_be_bytes();
    let mut parts = vec![count.to_vec()];
    for arg in args {
        parts.push((arg.len() as u64).to_be_bytes().to_vec());
        parts.push(arg.as_bytes().to_vec());
    }
    ContentId::derive(
        "symthaea.forge-bubblewrap-argv.v1",
        parts.iter().map(Vec::as_slice),
    )
}

enum WaitOutcome {
    Exited(ExitStatus, u64),
    TimedOut(u64),
}

fn wait_bounded(
    child: &mut std::process::Child,
    max_wall_time_ms: u64,
) -> Result<WaitOutcome, ForgeProposalLinuxIsolationError> {
    let start = Instant::now();
    let timeout = Duration::from_millis(max_wall_time_ms);
    loop {
        if let Some(status) = child.try_wait().map_err(|source| ForgeProposalLinuxIsolationError::Io {
            path: PathBuf::from("<bubblewrap-evaluator-process>"),
            source,
        })? {
            let millis = u64::try_from(start.elapsed().as_millis())
                .map_err(|_| ForgeProposalLinuxIsolationError::MeasurementOverflow)?;
            return Ok(WaitOutcome::Exited(status, millis));
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            let millis = u64::try_from(start.elapsed().as_millis())
                .map_err(|_| ForgeProposalLinuxIsolationError::MeasurementOverflow)?;
            return Ok(WaitOutcome::TimedOut(millis));
        }
        thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }
}

fn parse_wire_response(
    request: &ForgeProposalEvaluationRequest,
    stdout: &[u8],
) -> Result<(ContentId, Vec<ForgeProposalEvaluatorPrediction>), ForgeProposalLinuxIsolationError> {
    let wire: WireResponseV1 = serde_json::from_slice(stdout)?;
    if wire.schema != WIRE_RESPONSE_SCHEMA_V1
        || wire.request_id != request.id().as_str()
        || wire.protocol_id != request.protocol().id().as_str()
    {
        return Err(ForgeProposalLinuxIsolationError::WireScopeMismatch);
    }
    if wire.execution_context.is_empty() {
        return Err(ForgeProposalLinuxIsolationError::EmptyExecutionContext);
    }
    let rows = request
        .feature_rows()
        .iter()
        .map(|row| (row.target_id().as_str().to_string(), row))
        .collect::<BTreeMap<_, _>>();
    if rows.len() != request.feature_rows().len() || wire.predictions.len() != rows.len() {
        return Err(ForgeProposalLinuxIsolationError::WirePredictionCoverageMismatch);
    }
    let mut seen = BTreeSet::new();
    let mut predictions = Vec::with_capacity(wire.predictions.len());
    for prediction in wire.predictions {
        if !seen.insert(prediction.target_id.clone()) {
            return Err(ForgeProposalLinuxIsolationError::WirePredictionCoverageMismatch);
        }
        let row = rows
            .get(&prediction.target_id)
            .ok_or(ForgeProposalLinuxIsolationError::WirePredictionCoverageMismatch)?;
        predictions.push(ForgeProposalEvaluatorPrediction::for_feature_row(
            request,
            row,
            prediction.probability_scaled,
        )?);
    }
    if seen.len() != rows.len() || !rows.keys().all(|target| seen.contains(target)) {
        return Err(ForgeProposalLinuxIsolationError::WirePredictionCoverageMismatch);
    }
    let execution_context_id = ContentId::derive(
        "symthaea.forge-bubblewrap-evaluator-execution-context.v1",
        [wire.execution_context.as_bytes()],
    );
    Ok((execution_context_id, predictions))
}

/// Evidence emitted by a successful bubblewrap-isolated executable-model evaluation.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalBubblewrapExecutionReceipt {
    id: ContentId,
    policy_id: ContentId,
    executable_model_binding_id: ContentId,
    model_id: ContentId,
    model_payload_id: ContentId,
    bubblewrap_artifact_id: ContentId,
    bubblewrap_argv_id: ContentId,
    request_id: ContentId,
    response_id: ContentId,
    request_wire_id: ContentId,
    stdout_wire_id: ContentId,
    stderr_artifact_id: ContentId,
    execution_context_id: ContentId,
    stderr_bytes: u64,
    wall_time_ms: u64,
    isolation_status: ForgeProposalLinuxIsolationStatus,
}

impl ForgeProposalBubblewrapExecutionReceipt {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn policy_id(&self) -> &ContentId { &self.policy_id }
    pub fn executable_model_binding_id(&self) -> &ContentId { &self.executable_model_binding_id }
    pub fn model_id(&self) -> &ContentId { &self.model_id }
    pub fn model_payload_id(&self) -> &ContentId { &self.model_payload_id }
    pub fn bubblewrap_artifact_id(&self) -> &ContentId { &self.bubblewrap_artifact_id }
    pub fn bubblewrap_argv_id(&self) -> &ContentId { &self.bubblewrap_argv_id }
    pub fn request_id(&self) -> &ContentId { &self.request_id }
    pub fn response_id(&self) -> &ContentId { &self.response_id }
    pub fn request_wire_id(&self) -> &ContentId { &self.request_wire_id }
    pub fn stdout_wire_id(&self) -> &ContentId { &self.stdout_wire_id }
    pub fn stderr_artifact_id(&self) -> &ContentId { &self.stderr_artifact_id }
    pub fn execution_context_id(&self) -> &ContentId { &self.execution_context_id }
    pub fn stderr_bytes(&self) -> u64 { self.stderr_bytes }
    pub fn wall_time_ms(&self) -> u64 { self.wall_time_ms }
    pub fn isolation_status(&self) -> ForgeProposalLinuxIsolationStatus { self.isolation_status }

    pub fn validate_for(
        &self,
        policy: &ForgeProposalBubblewrapIsolationPolicy,
        binding: &ForgeProposalExecutableModelBinding,
        model: &ForgeProposalFrozenModel,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
    ) -> Result<(), ForgeProposalLinuxIsolationError> {
        policy.validate()?;
        binding.validate_for(model, request.protocol())?;
        response.validate_for(request)?;
        if self.policy_id != *policy.id()
            || self.executable_model_binding_id != *binding.id()
            || self.model_id != *model.id()
            || self.model_payload_id != *model.model_payload_id()
            || self.bubblewrap_artifact_id != *policy.bubblewrap_artifact_id()
            || self.request_id != *request.id()
            || self.response_id != *response.id()
            || self.execution_context_id != *response.execution_context_id()
            || self.stderr_bytes > policy.max_stderr_bytes()
            || self.wall_time_ms > policy.max_wall_time_ms()
            || self.isolation_status != ForgeProposalLinuxIsolationStatus::bubblewrap_v1()
        {
            return Err(ForgeProposalLinuxIsolationError::ReceiptScopeMismatch);
        }
        let request_wire = serde_json::to_vec(request)?;
        let expected_request_wire_id = ContentId::derive(
            "symthaea.forge-bubblewrap-observed-stdin.v1",
            [request_wire.as_slice()],
        );
        if self.request_wire_id != expected_request_wire_id {
            return Err(ForgeProposalLinuxIsolationError::ReceiptScopeMismatch);
        }
        let expected = derive_receipt_id(
            &self.policy_id,
            &self.executable_model_binding_id,
            &self.model_id,
            &self.model_payload_id,
            &self.bubblewrap_artifact_id,
            &self.bubblewrap_argv_id,
            &self.request_id,
            &self.response_id,
            &self.request_wire_id,
            &self.stdout_wire_id,
            &self.stderr_artifact_id,
            &self.execution_context_id,
            self.stderr_bytes,
            self.wall_time_ms,
            self.isolation_status,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalLinuxIsolationError::ReceiptIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_receipt_id(
    policy_id: &ContentId,
    binding_id: &ContentId,
    model_id: &ContentId,
    model_payload_id: &ContentId,
    bubblewrap_artifact_id: &ContentId,
    bubblewrap_argv_id: &ContentId,
    request_id: &ContentId,
    response_id: &ContentId,
    request_wire_id: &ContentId,
    stdout_wire_id: &ContentId,
    stderr_artifact_id: &ContentId,
    execution_context_id: &ContentId,
    stderr_bytes: u64,
    wall_time_ms: u64,
    isolation_status: ForgeProposalLinuxIsolationStatus,
) -> ContentId {
    let status = serde_json::to_vec(&isolation_status).unwrap_or_default();
    ContentId::derive(
        "symthaea.forge-bubblewrap-execution-receipt.v1",
        [
            policy_id.as_str().as_bytes(),
            binding_id.as_str().as_bytes(),
            model_id.as_str().as_bytes(),
            model_payload_id.as_str().as_bytes(),
            bubblewrap_artifact_id.as_str().as_bytes(),
            bubblewrap_argv_id.as_str().as_bytes(),
            request_id.as_str().as_bytes(),
            response_id.as_str().as_bytes(),
            request_wire_id.as_str().as_bytes(),
            stdout_wire_id.as_str().as_bytes(),
            stderr_artifact_id.as_str().as_bytes(),
            execution_context_id.as_str().as_bytes(),
            stderr_bytes.to_be_bytes().as_slice(),
            wall_time_ms.to_be_bytes().as_slice(),
            status.as_slice(),
        ],
    )
}

/// Execute the exact frozen executable-model payload inside the fixed bubblewrap v1 sandbox.
#[allow(clippy::too_many_arguments)]
pub fn run_bubblewrap_executable_model_evaluator(
    policy: &ForgeProposalBubblewrapIsolationPolicy,
    binding: &ForgeProposalExecutableModelBinding,
    request: &ForgeProposalEvaluationRequest,
    model: &ForgeProposalFrozenModel,
    bubblewrap_executable: impl AsRef<Path>,
    model_executable: impl AsRef<Path>,
    model_args: &[String],
) -> Result<(ForgeProposalEvaluationResponse, ForgeProposalBubblewrapExecutionReceipt), ForgeProposalLinuxIsolationError> {
    if !cfg!(target_os = "linux") {
        return Err(ForgeProposalLinuxIsolationError::UnsupportedPlatform);
    }
    policy.validate()?;
    request.validate_identity()?;
    binding.validate_for(model, request.protocol())?;
    if request.model_id() != model.id()
        || request.protocol().transport_schema_id() != policy.transport_schema_id()
        || request.protocol().transport_schema_id() != &forge_direct_evaluator_transport_schema_id()
    {
        return Err(ForgeProposalLinuxIsolationError::TransportSchemaMismatch);
    }

    let (canonical_bwrap, bwrap_bytes) = canonical_file_bytes(bubblewrap_executable.as_ref())?;
    if forge_bubblewrap_artifact_id(&bwrap_bytes) != *policy.bubblewrap_artifact_id() {
        return Err(ForgeProposalLinuxIsolationError::BubblewrapArtifactMismatch);
    }
    let (canonical_model, model_bytes) = canonical_file_bytes(model_executable.as_ref())?;
    if forge_evaluator_runner_artifact_id(&model_bytes) != *model.model_payload_id() {
        return Err(ForgeProposalLinuxIsolationError::ModelArtifactMismatch);
    }
    if request.protocol().runner_configuration_id() != &forge_evaluator_runner_argv_id(model_args) {
        return Err(ForgeProposalLinuxIsolationError::ModelArgvMismatch);
    }

    let request_wire = serde_json::to_vec(request)?;
    let request_len = u64::try_from(request_wire.len())
        .map_err(|_| ForgeProposalLinuxIsolationError::MeasurementOverflow)?;
    if request_len > policy.max_request_bytes() {
        return Err(ForgeProposalLinuxIsolationError::RequestTooLarge);
    }

    let bwrap_args = build_bwrap_args(&canonical_model, model_args)?;
    let bwrap_argv_id = derive_bwrap_argv_id(&bwrap_args);
    let workdir = FreshWorkDir::create(request.id())?;
    let mut command = Command::new(&canonical_bwrap);
    command
        .args(&bwrap_args)
        .env_clear()
        .current_dir(&workdir.path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|source| ForgeProposalLinuxIsolationError::Io {
        path: canonical_bwrap.clone(),
        source,
    })?;

    let mut stdin = child.stdin.take().ok_or(ForgeProposalLinuxIsolationError::MissingPipe)?;
    let stdout = child.stdout.take().ok_or(ForgeProposalLinuxIsolationError::MissingPipe)?;
    let stderr = child.stderr.take().ok_or(ForgeProposalLinuxIsolationError::MissingPipe)?;

    let stdin_bytes = request_wire.clone();
    let writer = thread::spawn(move || -> Result<(), std::io::Error> {
        stdin.write_all(&stdin_bytes)?;
        stdin.flush()?;
        Ok(())
    });
    let stdout_cap = policy.max_response_bytes();
    let stderr_cap = policy.max_stderr_bytes();
    let stdout_reader = thread::spawn(move || read_capped_and_drain(stdout, stdout_cap));
    let stderr_reader = thread::spawn(move || read_capped_and_drain(stderr, stderr_cap));

    let wait = wait_bounded(&mut child, policy.max_wall_time_ms())?;
    let writer_result = writer.join().map_err(|_| ForgeProposalLinuxIsolationError::StdinWriterFailed)?;
    writer_result.map_err(|_| ForgeProposalLinuxIsolationError::StdinWriterFailed)?;
    let stdout_result = stdout_reader
        .join()
        .map_err(|_| ForgeProposalLinuxIsolationError::ReaderThreadPanicked)?
        .map_err(|source| ForgeProposalLinuxIsolationError::Io {
            path: PathBuf::from("<bubblewrap-evaluator-stdout>"),
            source,
        })?;
    let stderr_result = stderr_reader
        .join()
        .map_err(|_| ForgeProposalLinuxIsolationError::ReaderThreadPanicked)?
        .map_err(|source| ForgeProposalLinuxIsolationError::Io {
            path: PathBuf::from("<bubblewrap-evaluator-stderr>"),
            source,
        })?;

    let (status, wall_time_ms) = match wait {
        WaitOutcome::TimedOut(_) => return Err(ForgeProposalLinuxIsolationError::TimedOut),
        WaitOutcome::Exited(status, wall_time_ms) => (status, wall_time_ms),
    };
    if stdout_result.exceeded {
        return Err(ForgeProposalLinuxIsolationError::StdoutLimitExceeded);
    }
    if stderr_result.exceeded {
        return Err(ForgeProposalLinuxIsolationError::StderrLimitExceeded);
    }
    if !status.success() {
        return Err(ForgeProposalLinuxIsolationError::RunnerFailed { code: status.code() });
    }

    let (execution_context_id, predictions) = parse_wire_response(request, &stdout_result.bytes)?;
    let response = ForgeProposalEvaluationResponse::freeze(request, execution_context_id, predictions)?;

    let request_wire_id = ContentId::derive(
        "symthaea.forge-bubblewrap-observed-stdin.v1",
        [request_wire.as_slice()],
    );
    let stdout_wire_id = ContentId::derive(
        "symthaea.forge-bubblewrap-observed-stdout.v1",
        [stdout_result.bytes.as_slice()],
    );
    let stderr_artifact_id = ContentId::derive(
        "symthaea.forge-bubblewrap-stderr.v1",
        [stderr_result.bytes.as_slice()],
    );
    let isolation_status = ForgeProposalLinuxIsolationStatus::bubblewrap_v1();
    let id = derive_receipt_id(
        policy.id(),
        binding.id(),
        model.id(),
        model.model_payload_id(),
        policy.bubblewrap_artifact_id(),
        &bwrap_argv_id,
        request.id(),
        response.id(),
        &request_wire_id,
        &stdout_wire_id,
        &stderr_artifact_id,
        response.execution_context_id(),
        stderr_result.total_bytes,
        wall_time_ms,
        isolation_status,
    );
    let receipt = ForgeProposalBubblewrapExecutionReceipt {
        id,
        policy_id: policy.id().clone(),
        executable_model_binding_id: binding.id().clone(),
        model_id: model.id().clone(),
        model_payload_id: model.model_payload_id().clone(),
        bubblewrap_artifact_id: policy.bubblewrap_artifact_id().clone(),
        bubblewrap_argv_id,
        request_id: request.id().clone(),
        response_id: response.id().clone(),
        request_wire_id,
        stdout_wire_id,
        stderr_artifact_id,
        execution_context_id: response.execution_context_id().clone(),
        stderr_bytes: stderr_result.total_bytes,
        wall_time_ms,
        isolation_status,
    };
    receipt.validate_for(policy, binding, model, request, &response)?;
    Ok((response, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolation_status_does_not_overclaim_seccomp_or_cgroups() {
        let status = ForgeProposalLinuxIsolationStatus::bubblewrap_v1();
        assert_eq!(status.network_namespace(), ForgeProposalIsolationState::EstablishedV1);
        assert_eq!(status.restricted_mount_namespace(), ForgeProposalIsolationState::EstablishedV1);
        assert_eq!(status.seccomp_filter(), ForgeProposalIsolationState::NotEstablishedV1);
        assert_eq!(status.landlock(), ForgeProposalIsolationState::NotEstablishedV1);
        assert_eq!(
            status.cgroup_resource_enforcement(),
            ForgeProposalIsolationState::NotEstablishedV1
        );
    }

    #[test]
    fn bubblewrap_argv_identity_is_order_sensitive() {
        let a = derive_bwrap_argv_id(&["--unshare-net".into(), "--clearenv".into()]);
        let b = derive_bwrap_argv_id(&["--clearenv".into(), "--unshare-net".into()]);
        assert_ne!(a, b);
    }
}