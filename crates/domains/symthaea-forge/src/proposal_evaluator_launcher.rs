// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded direct-process launcher for the label-blind Forge evaluator protocol.
//!
//! This is the first layer in the evaluator stack that actually executes a child process. It
//! enforces direct execution (no shell), clears the inherited environment, creates a fresh empty
//! working directory, pipes the exact request JSON on stdin, captures stdout/stderr concurrently,
//! enforces a monotonic wall-time limit, drains output under hard size caps, validates a narrow JSON
//! response schema, and produces both a semantic evaluator response and execution evidence.
//!
//! Important nonclaim: this launcher does **not** isolate the child filesystem or network. Clearing
//! environment variables and changing the working directory are useful hygiene but are not a
//! security sandbox. The execution record therefore remains `NotEstablishedV1` for runtime
//! isolation. A future backend should add namespace/seccomp/cgroup/VM isolation under a new theorem.

use crate::proposal_evaluator_execution::{
    ForgeProposalEvaluatorExecutionError, ForgeProposalEvaluatorExecutionRecord,
    ForgeProposalEvaluatorLaunchPolicy,
};
use crate::proposal_evaluator_protocol::{
    ForgeProposalEvaluationRequest, ForgeProposalEvaluationResponse,
    ForgeProposalEvaluatorPrediction, ForgeProposalEvaluatorProtocolError,
};
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
const HARD_MAX_REQUEST_BYTES_V1: u64 = 256 * 1024 * 1024;
const HARD_MAX_RESPONSE_BYTES_V1: u64 = 256 * 1024 * 1024;
const HARD_MAX_STDERR_BYTES_V1: u64 = 64 * 1024 * 1024;
const HARD_MAX_WALL_TIME_MS_V1: u64 = 60 * 60 * 1000;
const HARD_MAX_RUNNER_BYTES_V1: u64 = 512 * 1024 * 1024;
const POLL_INTERVAL_MS: u64 = 10;

static WORKDIR_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum ForgeProposalEvaluatorLauncherError {
    #[error(transparent)]
    Protocol(#[from] ForgeProposalEvaluatorProtocolError),
    #[error(transparent)]
    Execution(#[from] ForgeProposalEvaluatorExecutionError),
    #[error("evaluator launcher IO error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("evaluator runner executable is not a regular file")]
    InvalidExecutable,
    #[error("evaluator runner binary exceeds the hard v1 artifact-size ceiling")]
    RunnerTooLarge,
    #[error("evaluator protocol runner artifact does not match the exact executable bytes")]
    RunnerArtifactMismatch,
    #[error("evaluator protocol runner configuration does not match the exact ordered argv")]
    RunnerConfigurationMismatch,
    #[error("evaluator protocol transport schema is not the direct-launcher v1 schema")]
    TransportSchemaMismatch,
    #[error("evaluator launch policy is not bound to the direct-launcher v1 implementation/configuration")]
    LauncherPolicyMismatch,
    #[error("evaluator launch policy exceeds hard v1 safety ceilings")]
    UnsafePolicyLimits,
    #[error("evaluator request exceeds the frozen request-size limit")]
    RequestTooLarge,
    #[error("evaluator process could not expose all required stdio pipes")]
    MissingPipe,
    #[error("evaluator stdin writer thread failed or panicked")]
    StdinWriterFailed,
    #[error("evaluator stdout/stderr reader thread panicked")]
    ReaderThreadPanicked,
    #[error("evaluator process exceeded the frozen wall-time limit")]
    TimedOut,
    #[error("evaluator stdout exceeded the frozen response-size limit")]
    StdoutLimitExceeded,
    #[error("evaluator stderr exceeded the frozen stderr-size limit")]
    StderrLimitExceeded,
    #[error("evaluator process exited unsuccessfully with code {code:?}")]
    RunnerFailed { code: Option<i32> },
    #[error("evaluator stdout is not valid v1 response JSON: {0}")]
    InvalidResponseJson(#[from] serde_json::Error),
    #[error("evaluator wire response schema/request/protocol identity does not match the request")]
    WireScopeMismatch,
    #[error("evaluator wire response contains duplicate, missing, or unknown targets")]
    WirePredictionCoverageMismatch,
    #[error("evaluator execution-context string must be non-empty")]
    EmptyExecutionContext,
    #[error("evaluator launcher measurement cannot be represented in u64")]
    MeasurementOverflow,
}

/// Semantic identity for the actual direct-launcher implementation in this module.
pub fn forge_direct_evaluator_launcher_implementation_id() -> ContentId {
    ContentId::derive(
        "symthaea.forge-direct-evaluator-launcher-implementation.v1",
        [b"direct-exec;env-clear;fresh-cwd;piped-stdio;concurrent-drain;monotonic-timeout".as_slice()],
    )
}

/// Semantic configuration identity for fixed v1 launcher mechanics.
pub fn forge_direct_evaluator_launcher_configuration_id() -> ContentId {
    ContentId::derive(
        "symthaea.forge-direct-evaluator-launcher-configuration.v1",
        [b"poll-10ms;hard-caps-v1;stdout-json-v1;stderr-captured;network-fs-isolation-not-established".as_slice()],
    )
}

/// Transport-schema identity expected by the direct launcher.
pub fn forge_direct_evaluator_transport_schema_id() -> ContentId {
    ContentId::derive(
        "symthaea.forge-direct-evaluator-transport-schema.v1",
        [b"stdin=ForgeProposalEvaluationRequest-json;stdout=symthaea-forge-evaluator-response-v1".as_slice()],
    )
}

/// Exact runner artifact identity from executable bytes.
pub fn forge_evaluator_runner_artifact_id(bytes: &[u8]) -> ContentId {
    ContentId::derive(
        "symthaea.forge-evaluator-runner-artifact.v1",
        [bytes],
    )
}

/// Exact ordered argv configuration identity used by the direct launcher.
pub fn forge_evaluator_runner_argv_id(args: &[String]) -> ContentId {
    let count = (args.len() as u64).to_be_bytes();
    let mut parts = vec![count.to_vec()];
    for arg in args {
        parts.push((arg.len() as u64).to_be_bytes().to_vec());
        parts.push(arg.as_bytes().to_vec());
    }
    ContentId::derive(
        "symthaea.forge-evaluator-runner-argv.v1",
        parts.iter().map(Vec::as_slice),
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

fn read_capped_and_drain<R: Read>(
    mut reader: R,
    cap: u64,
) -> Result<CappedOutput, std::io::Error> {
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
    Ok(CappedOutput {
        bytes,
        total_bytes,
        exceeded,
    })
}

struct FreshWorkDir {
    path: PathBuf,
}

impl FreshWorkDir {
    fn create(request_id: &ContentId) -> Result<Self, ForgeProposalEvaluatorLauncherError> {
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
                "symthaea-forge-evaluator-{}-{epoch_nanos}-{nonce}-{attempt}-{request_tag}",
                std::process::id(),
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(source) => {
                    return Err(ForgeProposalEvaluatorLauncherError::Io { path, source });
                }
            }
        }
        Err(ForgeProposalEvaluatorLauncherError::Io {
            path: std::env::temp_dir(),
            source: std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not allocate unique evaluator working directory",
            ),
        })
    }
}

impl Drop for FreshWorkDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Evidence emitted by the real direct-process launcher.
///
/// It binds the actual runner bytes/argv plus the exact stdin/stdout byte streams observed by the
/// launcher. The embedded execution record still reports runtime isolation as not established.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalDirectLaunchReceipt {
    id: ContentId,
    execution_record: ForgeProposalEvaluatorExecutionRecord,
    runner_artifact_id: ContentId,
    runner_argv_id: ContentId,
    transport_schema_id: ContentId,
    stdin_wire_id: ContentId,
    stdout_wire_id: ContentId,
    exit_code: i32,
    wall_time_ms: u64,
}

impl ForgeProposalDirectLaunchReceipt {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn execution_record(&self) -> &ForgeProposalEvaluatorExecutionRecord {
        &self.execution_record
    }
    pub fn runner_artifact_id(&self) -> &ContentId { &self.runner_artifact_id }
    pub fn runner_argv_id(&self) -> &ContentId { &self.runner_argv_id }
    pub fn transport_schema_id(&self) -> &ContentId { &self.transport_schema_id }
    pub fn stdin_wire_id(&self) -> &ContentId { &self.stdin_wire_id }
    pub fn stdout_wire_id(&self) -> &ContentId { &self.stdout_wire_id }
    pub fn exit_code(&self) -> i32 { self.exit_code }
    pub fn wall_time_ms(&self) -> u64 { self.wall_time_ms }
}

fn validate_policy_for_direct_launcher(
    policy: &ForgeProposalEvaluatorLaunchPolicy,
    request: &ForgeProposalEvaluationRequest,
) -> Result<(), ForgeProposalEvaluatorLauncherError> {
    policy.validate_for(request.protocol())?;
    if policy.launcher_implementation_id() != &forge_direct_evaluator_launcher_implementation_id()
        || policy.launcher_configuration_id()
            != &forge_direct_evaluator_launcher_configuration_id()
    {
        return Err(ForgeProposalEvaluatorLauncherError::LauncherPolicyMismatch);
    }
    if policy.max_request_bytes() > HARD_MAX_REQUEST_BYTES_V1
        || policy.max_response_bytes() > HARD_MAX_RESPONSE_BYTES_V1
        || policy.max_stderr_bytes() > HARD_MAX_STDERR_BYTES_V1
        || policy.max_wall_time_ms() > HARD_MAX_WALL_TIME_MS_V1
    {
        return Err(ForgeProposalEvaluatorLauncherError::UnsafePolicyLimits);
    }
    if request.protocol().transport_schema_id() != &forge_direct_evaluator_transport_schema_id() {
        return Err(ForgeProposalEvaluatorLauncherError::TransportSchemaMismatch);
    }
    Ok(())
}

fn canonical_runner_bytes(
    executable: &Path,
) -> Result<(PathBuf, Vec<u8>), ForgeProposalEvaluatorLauncherError> {
    let canonical = executable
        .canonicalize()
        .map_err(|source| ForgeProposalEvaluatorLauncherError::Io {
            path: executable.to_path_buf(),
            source,
        })?;
    let metadata = fs::metadata(&canonical).map_err(|source| {
        ForgeProposalEvaluatorLauncherError::Io {
            path: canonical.clone(),
            source,
        }
    })?;
    if !metadata.is_file() {
        return Err(ForgeProposalEvaluatorLauncherError::InvalidExecutable);
    }
    if metadata.len() > HARD_MAX_RUNNER_BYTES_V1 {
        return Err(ForgeProposalEvaluatorLauncherError::RunnerTooLarge);
    }
    let bytes = fs::read(&canonical).map_err(|source| ForgeProposalEvaluatorLauncherError::Io {
        path: canonical.clone(),
        source,
    })?;
    Ok((canonical, bytes))
}

fn wait_bounded(
    child: &mut std::process::Child,
    max_wall_time_ms: u64,
) -> Result<(ExitStatus, u64), ForgeProposalEvaluatorLauncherError> {
    let start = Instant::now();
    let timeout = Duration::from_millis(max_wall_time_ms);
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|source| ForgeProposalEvaluatorLauncherError::Io {
                path: PathBuf::from("<evaluator-process>"),
                source,
            })?
        {
            let millis = u64::try_from(start.elapsed().as_millis())
                .map_err(|_| ForgeProposalEvaluatorLauncherError::MeasurementOverflow)?;
            return Ok((status, millis));
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ForgeProposalEvaluatorLauncherError::TimedOut);
        }
        thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }
}

fn parse_wire_response(
    request: &ForgeProposalEvaluationRequest,
    stdout: &[u8],
) -> Result<(ContentId, Vec<ForgeProposalEvaluatorPrediction>), ForgeProposalEvaluatorLauncherError> {
    let wire: WireResponseV1 = serde_json::from_slice(stdout)?;
    if wire.schema != WIRE_RESPONSE_SCHEMA_V1
        || wire.request_id != request.id().as_str()
        || wire.protocol_id != request.protocol().id().as_str()
    {
        return Err(ForgeProposalEvaluatorLauncherError::WireScopeMismatch);
    }
    if wire.execution_context.is_empty() {
        return Err(ForgeProposalEvaluatorLauncherError::EmptyExecutionContext);
    }
    let rows = request
        .feature_rows()
        .iter()
        .map(|row| (row.target_id().as_str().to_string(), row))
        .collect::<BTreeMap<_, _>>();
    if rows.len() != request.feature_rows().len() || wire.predictions.len() != rows.len() {
        return Err(ForgeProposalEvaluatorLauncherError::WirePredictionCoverageMismatch);
    }
    let mut seen = BTreeSet::new();
    let mut predictions = Vec::with_capacity(wire.predictions.len());
    for prediction in wire.predictions {
        if !seen.insert(prediction.target_id.clone()) {
            return Err(ForgeProposalEvaluatorLauncherError::WirePredictionCoverageMismatch);
        }
        let row = rows
            .get(&prediction.target_id)
            .ok_or(ForgeProposalEvaluatorLauncherError::WirePredictionCoverageMismatch)?;
        predictions.push(ForgeProposalEvaluatorPrediction::for_feature_row(
            request,
            row,
            prediction.probability_scaled,
        )?);
    }
    if seen.len() != rows.len() || !rows.keys().all(|target| seen.contains(target)) {
        return Err(ForgeProposalEvaluatorLauncherError::WirePredictionCoverageMismatch);
    }
    let execution_context_id = ContentId::derive(
        "symthaea.forge-direct-evaluator-execution-context.v1",
        [wire.execution_context.as_bytes()],
    );
    Ok((execution_context_id, predictions))
}

/// Execute one exact evaluator request through the bounded direct-process backend.
///
/// The runner protocol must bind the exact executable bytes, ordered argv, and transport schema via
/// `runner_implementation_id`, `runner_configuration_id`, and `transport_schema_id` respectively.
pub fn run_direct_evaluator(
    policy: &ForgeProposalEvaluatorLaunchPolicy,
    request: &ForgeProposalEvaluationRequest,
    executable: impl AsRef<Path>,
    args: &[String],
) -> Result<
    (ForgeProposalEvaluationResponse, ForgeProposalDirectLaunchReceipt),
    ForgeProposalEvaluatorLauncherError,
> {
    request.validate_identity()?;
    validate_policy_for_direct_launcher(policy, request)?;

    let (canonical_executable, runner_bytes) = canonical_runner_bytes(executable.as_ref())?;
    let runner_artifact_id = forge_evaluator_runner_artifact_id(&runner_bytes);
    if request.protocol().runner_implementation_id() != &runner_artifact_id {
        return Err(ForgeProposalEvaluatorLauncherError::RunnerArtifactMismatch);
    }
    let runner_argv_id = forge_evaluator_runner_argv_id(args);
    if request.protocol().runner_configuration_id() != &runner_argv_id {
        return Err(ForgeProposalEvaluatorLauncherError::RunnerConfigurationMismatch);
    }

    let request_wire = serde_json::to_vec(request)?;
    let request_len = u64::try_from(request_wire.len())
        .map_err(|_| ForgeProposalEvaluatorLauncherError::MeasurementOverflow)?;
    if request_len > policy.max_request_bytes() {
        return Err(ForgeProposalEvaluatorLauncherError::RequestTooLarge);
    }

    let workdir = FreshWorkDir::create(request.id())?;
    let mut command = Command::new(&canonical_executable);
    command
        .args(args)
        .env_clear()
        .current_dir(&workdir.path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|source| ForgeProposalEvaluatorLauncherError::Io {
            path: canonical_executable.clone(),
            source,
        })?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or(ForgeProposalEvaluatorLauncherError::MissingPipe)?;
    let stdout = child
        .stdout
        .take()
        .ok_or(ForgeProposalEvaluatorLauncherError::MissingPipe)?;
    let stderr = child
        .stderr
        .take()
        .ok_or(ForgeProposalEvaluatorLauncherError::MissingPipe)?;

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

    let (status, wall_time_ms) = wait_bounded(&mut child, policy.max_wall_time_ms())?;
    let writer_result = writer
        .join()
        .map_err(|_| ForgeProposalEvaluatorLauncherError::StdinWriterFailed)?;
    writer_result.map_err(|_| ForgeProposalEvaluatorLauncherError::StdinWriterFailed)?;
    let stdout_result = stdout_reader
        .join()
        .map_err(|_| ForgeProposalEvaluatorLauncherError::ReaderThreadPanicked)?
        .map_err(|source| ForgeProposalEvaluatorLauncherError::Io {
            path: PathBuf::from("<evaluator-stdout>"),
            source,
        })?;
    let stderr_result = stderr_reader
        .join()
        .map_err(|_| ForgeProposalEvaluatorLauncherError::ReaderThreadPanicked)?
        .map_err(|source| ForgeProposalEvaluatorLauncherError::Io {
            path: PathBuf::from("<evaluator-stderr>"),
            source,
        })?;

    if stdout_result.exceeded {
        return Err(ForgeProposalEvaluatorLauncherError::StdoutLimitExceeded);
    }
    if stderr_result.exceeded {
        return Err(ForgeProposalEvaluatorLauncherError::StderrLimitExceeded);
    }
    if !status.success() {
        return Err(ForgeProposalEvaluatorLauncherError::RunnerFailed {
            code: status.code(),
        });
    }
    let exit_code = status
        .code()
        .ok_or(ForgeProposalEvaluatorLauncherError::RunnerFailed { code: None })?;
    if exit_code != 0 {
        return Err(ForgeProposalEvaluatorLauncherError::RunnerFailed {
            code: Some(exit_code),
        });
    }

    let (execution_context_id, predictions) = parse_wire_response(request, &stdout_result.bytes)?;
    let response = ForgeProposalEvaluationResponse::freeze(
        request,
        execution_context_id,
        predictions,
    )?;

    let stderr_artifact_id = ContentId::derive(
        "symthaea.forge-direct-evaluator-stderr.v1",
        [stderr_result.bytes.as_slice()],
    );
    let stdin_wire_id = ContentId::derive(
        "symthaea.forge-direct-evaluator-observed-stdin.v1",
        [request_wire.as_slice()],
    );
    let stdout_wire_id = ContentId::derive(
        "symthaea.forge-direct-evaluator-observed-stdout.v1",
        [stdout_result.bytes.as_slice()],
    );
    let launcher_evidence_id = ContentId::derive(
        "symthaea.forge-direct-evaluator-launch-evidence.v1",
        [
            policy.id().as_str().as_bytes(),
            runner_artifact_id.as_str().as_bytes(),
            runner_argv_id.as_str().as_bytes(),
            stdin_wire_id.as_str().as_bytes(),
            stdout_wire_id.as_str().as_bytes(),
            stderr_artifact_id.as_str().as_bytes(),
            wall_time_ms.to_be_bytes().as_slice(),
        ],
    );
    let execution_record = ForgeProposalEvaluatorExecutionRecord::record_success(
        policy,
        request,
        &response,
        launcher_evidence_id,
        stderr_artifact_id.clone(),
        stderr_result.total_bytes,
        wall_time_ms,
    )?;
    let id = derive_launch_receipt_id(
        execution_record.id(),
        &runner_artifact_id,
        &runner_argv_id,
        request.protocol().transport_schema_id(),
        &stdin_wire_id,
        &stdout_wire_id,
        exit_code,
        wall_time_ms,
    );
    let receipt = ForgeProposalDirectLaunchReceipt {
        id,
        execution_record,
        runner_artifact_id,
        runner_argv_id,
        transport_schema_id: request.protocol().transport_schema_id().clone(),
        stdin_wire_id,
        stdout_wire_id,
        exit_code,
        wall_time_ms,
    };
    Ok((response, receipt))
}

#[allow(clippy::too_many_arguments)]
fn derive_launch_receipt_id(
    execution_record_id: &ContentId,
    runner_artifact_id: &ContentId,
    runner_argv_id: &ContentId,
    transport_schema_id: &ContentId,
    stdin_wire_id: &ContentId,
    stdout_wire_id: &ContentId,
    exit_code: i32,
    wall_time_ms: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-proposal-direct-launch-receipt.v1",
        [
            execution_record_id.as_str().as_bytes(),
            runner_artifact_id.as_str().as_bytes(),
            runner_argv_id.as_str().as_bytes(),
            transport_schema_id.as_str().as_bytes(),
            stdin_wire_id.as_str().as_bytes(),
            stdout_wire_id.as_str().as_bytes(),
            exit_code.to_be_bytes().as_slice(),
            wall_time_ms.to_be_bytes().as_slice(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_argv_changes_configuration_identity() {
        let a = forge_evaluator_runner_argv_id(&["--alpha".to_string(), "1".to_string()]);
        let b = forge_evaluator_runner_argv_id(&["1".to_string(), "--alpha".to_string()]);
        assert_ne!(a, b);
    }

    #[test]
    fn transport_schema_is_versioned() {
        assert_ne!(
            forge_direct_evaluator_transport_schema_id(),
            ContentId::derive("other-transport", [b"v1".as_slice()])
        );
    }
}
