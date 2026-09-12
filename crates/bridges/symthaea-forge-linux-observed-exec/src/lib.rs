// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(target_os = "linux")]
//! Kernel-gated Bubblewrap execution for Forge's label-blind evaluator.
//!
//! The Bubblewrap monitor reports the host-visible sandbox PID through `--json-status-fd`. The
//! sandbox itself is held at `--block-fd` after privileged setup and capability dropping. The
//! parent independently observes the exact sandbox process and releases one byte only after the
//! parent-side `KernelIsolationGate` succeeds.
//!
//! Before `child-pid` is known, Bubblewrap and its setup child live in a dedicated process group.
//! Every pre-gate failure SIGKILLs that group and verifies it is empty before control FDs are
//! allowed to close. After `child-pid` is known, a Linux pidfd becomes the stable process handle;
//! it is kill-on-drop until exact sandbox exit is verified. The frozen gate timeout is one total
//! launch-to-release budget rather than independent status and observation budgets.
//!
//! This establishes an observed-process-isolation proposition. It still does not establish
//! seccomp filtering, Landlock, cgroup CPU/memory limits, VM isolation, kernel-exploit resistance,
//! or independent semantic correctness of Bubblewrap itself.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use symthaea_algorithms::ContentId;
use symthaea_forge::{
    forge_bubblewrap_artifact_id, forge_direct_evaluator_transport_schema_id,
    forge_evaluator_runner_argv_id, forge_evaluator_runner_artifact_id,
    ForgeProposalEvaluationRequest, ForgeProposalEvaluationResponse, ForgeProposalEvaluatorPrediction,
    ForgeProposalEvaluatorProtocolError, ForgeProposalExecutableModelBinding,
    ForgeProposalExecutableModelError, ForgeProposalFrozenModel,
};
use symthaea_forge_linux_isolation::BubblewrapV2Policy;
use symthaea_forge_linux_kernel_attestation::{
    observe_sandbox_process, KernelAttestationError, KernelIsolationGate, KernelSandboxObservation,
};
use thiserror::Error;

const WIRE_RESPONSE_SCHEMA_V1: &str = "symthaea-forge-evaluator-response-v1";
const SANDBOX_RUNNER: &str = "/runner";
const SANDBOX_TMP: &str = "/tmp";
const SANDBOX_HOSTNAME: &str = "symthaea-forge-evaluator";
const NIX_STORE: &str = "/nix/store";
const STATUS_FD: RawFd = 3;
const BLOCK_FD: RawFd = 4;
const HIGH_FD_MIN: RawFd = 10;
const MAX_STATUS_BYTES: u64 = 1024 * 1024;
const MAX_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const POLL_INTERVAL_MS: u64 = 10;
const MAX_GATE_TIMEOUT_MS: u64 = 60_000;
const MAX_TEARDOWN_TIMEOUT_MS: u64 = 30_000;

static WORKDIR_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum ObservedEvaluatorError {
    #[error(transparent)]
    Protocol(#[from] ForgeProposalEvaluatorProtocolError),
    #[error(transparent)]
    ExecutableModel(#[from] ForgeProposalExecutableModelError),
    #[error(transparent)]
    Kernel(#[from] KernelAttestationError),
    #[error("observed evaluator IO error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Bubblewrap or model executable is not a regular file")]
    InvalidExecutable,
    #[error("Bubblewrap or model executable exceeds the hard artifact-size ceiling")]
    ArtifactTooLarge,
    #[error("model executable path is not valid UTF-8")]
    NonUtf8Path,
    #[error("Bubblewrap bytes do not match the frozen Bubblewrap v2 policy")]
    BubblewrapArtifactMismatch,
    #[error("model executable bytes do not match the frozen model payload")]
    ModelArtifactMismatch,
    #[error("model argv does not match the frozen evaluator protocol configuration")]
    ModelArgvMismatch,
    #[error("evaluator protocol does not use the required label-blind transport schema")]
    TransportSchemaMismatch,
    #[error("kernel-gated policy exceeds hard v1 safety ceilings")]
    UnsafePolicyLimits,
    #[error("evaluator request exceeds the frozen request-size limit")]
    RequestTooLarge,
    #[error("could not create, duplicate, poll, or query a Linux control primitive: {0}")]
    Control(std::io::Error),
    #[error("Bubblewrap process did not expose required stdio")]
    MissingPipe,
    #[error("pre-gate Bubblewrap process-group teardown could not be verified")]
    PreGateTeardownUnverified,
    #[error("Bubblewrap JSON status stream timed out before reporting child-pid")]
    StatusTimeout,
    #[error("Bubblewrap JSON status stream exceeded its hard size ceiling")]
    StatusTooLarge,
    #[error("Bubblewrap JSON status line is invalid: {0}")]
    InvalidStatusJson(serde_json::Error),
    #[error("Bubblewrap JSON status transcript is not UTF-8")]
    StatusNotUtf8,
    #[error("Bubblewrap JSON status stream ended without reporting child-pid")]
    MissingChildPid,
    #[error("could not open a pidfd for the exact sandbox process: {0}")]
    PidfdOpen(std::io::Error),
    #[error("could not signal the exact sandbox process through pidfd: {0}")]
    PidfdSignal(std::io::Error),
    #[error("pre-exec gate budget was exhausted before the kernel isolation gate was established; last observation error: {last_error}")]
    KernelGateTimeout { last_error: String },
    #[error("sandbox exited before the pre-exec kernel gate was established with code {code:?}")]
    SandboxExitedBeforeGate { code: Option<i32> },
    #[error("kernel gate release byte could not be written")]
    GateReleaseFailed,
    #[error("evaluator stdin writer failed or panicked")]
    StdinWriterFailed,
    #[error("evaluator output/status reader thread panicked")]
    ReaderThreadPanicked,
    #[error("evaluator exceeded the frozen wall-time limit and exact pidfd teardown was verified")]
    TimedOutAfterVerifiedTeardown,
    #[error("sandbox teardown could not be verified through the exact pidfd")]
    TeardownUnverified,
    #[error("evaluator stdout exceeded the frozen response-size limit")]
    StdoutLimitExceeded,
    #[error("evaluator stderr exceeded the frozen stderr-size limit")]
    StderrLimitExceeded,
    #[error("Bubblewrap/evaluator exited unsuccessfully with code {code:?}")]
    RunnerFailed { code: Option<i32> },
    #[error("Bubblewrap status stream did not report the final exit code")]
    MissingStatusExit,
    #[error("Bubblewrap status transcript does not bind the reported sandbox PID")]
    StatusChildPidMismatch,
    #[error("Bubblewrap status exit code does not match the wrapper exit status")]
    StatusExitMismatch,
    #[error("evaluator stdout is not valid response JSON: {0}")]
    InvalidResponseJson(serde_json::Error),
    #[error("evaluator wire response does not match request/protocol identity")]
    WireScopeMismatch,
    #[error("evaluator response contains duplicate, missing, or unknown targets")]
    WirePredictionCoverageMismatch,
    #[error("evaluator execution-context string must be non-empty")]
    EmptyExecutionContext,
    #[error("measurement cannot be represented in u64")]
    MeasurementOverflow,
    #[error("kernel-gated execution receipt does not bind the supplied evidence")]
    ReceiptScopeMismatch,
    #[error("kernel-gated execution receipt identity does not match canonical fields")]
    ReceiptIdentityMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelGatedEvaluatorPolicy {
    id: ContentId,
    bubblewrap_policy: BubblewrapV2Policy,
    kernel_gate_timeout_ms: u64,
    teardown_timeout_ms: u64,
}

impl KernelGatedEvaluatorPolicy {
    pub fn new(
        bubblewrap_policy: BubblewrapV2Policy,
        kernel_gate_timeout_ms: u64,
        teardown_timeout_ms: u64,
    ) -> Result<Self, ObservedEvaluatorError> {
        bubblewrap_policy
            .validate()
            .map_err(|_| ObservedEvaluatorError::ReceiptScopeMismatch)?;
        if kernel_gate_timeout_ms == 0
            || teardown_timeout_ms == 0
            || kernel_gate_timeout_ms > MAX_GATE_TIMEOUT_MS
            || teardown_timeout_ms > MAX_TEARDOWN_TIMEOUT_MS
            || kernel_gate_timeout_ms >= bubblewrap_policy.max_wall_time_ms()
        {
            return Err(ObservedEvaluatorError::UnsafePolicyLimits);
        }
        let id = derive_policy_id(
            bubblewrap_policy.id(),
            kernel_gate_timeout_ms,
            teardown_timeout_ms,
        );
        Ok(Self {
            id,
            bubblewrap_policy,
            kernel_gate_timeout_ms,
            teardown_timeout_ms,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn bubblewrap_policy(&self) -> &BubblewrapV2Policy { &self.bubblewrap_policy }
    pub fn kernel_gate_timeout_ms(&self) -> u64 { self.kernel_gate_timeout_ms }
    pub fn teardown_timeout_ms(&self) -> u64 { self.teardown_timeout_ms }

    pub fn validate(&self) -> Result<(), ObservedEvaluatorError> {
        let rebuilt = Self::new(
            self.bubblewrap_policy.clone(),
            self.kernel_gate_timeout_ms,
            self.teardown_timeout_ms,
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(ObservedEvaluatorError::ReceiptIdentityMismatch)
        }
    }
}

fn derive_policy_id(base: &ContentId, gate_timeout_ms: u64, teardown_timeout_ms: u64) -> ContentId {
    ContentId::derive(
        "symthaea.forge-kernel-gated-evaluator-policy.v1",
        [
            base.as_str().as_bytes(),
            gate_timeout_ms.to_be_bytes().as_slice(),
            teardown_timeout_ms.to_be_bytes().as_slice(),
            b"json-status-fd+block-fd+single-preexec-budget+verified-process-group+kernel-gate+pidfd-teardown",
        ],
    )
}

#[derive(Debug, Deserialize)]
struct StatusEvent {
    #[serde(rename = "child-pid")]
    child_pid: Option<u32>,
    #[serde(rename = "exit-code")]
    exit_code: Option<i32>,
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

struct FreshWorkDir {
    path: PathBuf,
}

impl FreshWorkDir {
    fn create(request_id: &ContentId) -> Result<Self, ObservedEvaluatorError> {
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
                "symthaea-forge-observed-{}-{epoch_nanos}-{nonce}-{attempt}-{request_tag}",
                std::process::id(),
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(source) => return Err(ObservedEvaluatorError::Io { path, source }),
            }
        }
        Err(ObservedEvaluatorError::Io {
            path: std::env::temp_dir(),
            source: std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not allocate unique observed evaluator work directory",
            ),
        })
    }
}

impl Drop for FreshWorkDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn canonical_file_bytes(path: &Path) -> Result<(PathBuf, Vec<u8>), ObservedEvaluatorError> {
    let canonical = path.canonicalize().map_err(|source| ObservedEvaluatorError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = fs::metadata(&canonical).map_err(|source| ObservedEvaluatorError::Io {
        path: canonical.clone(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(ObservedEvaluatorError::InvalidExecutable);
    }
    if metadata.len() > MAX_ARTIFACT_BYTES {
        return Err(ObservedEvaluatorError::ArtifactTooLarge);
    }
    let bytes = fs::read(&canonical).map_err(|source| ObservedEvaluatorError::Io {
        path: canonical.clone(),
        source,
    })?;
    Ok((canonical, bytes))
}

fn pipe_cloexec() -> Result<(OwnedFd, OwnedFd), ObservedEvaluatorError> {
    let mut fds = [-1; 2];
    // SAFETY: `fds` points to two writable integers and pipe2 initializes both on success.
    let result = unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) };
    if result != 0 {
        return Err(ObservedEvaluatorError::Control(std::io::Error::last_os_error()));
    }
    // SAFETY: successful pipe2 returned two fresh descriptors owned by the caller.
    let read = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    // SAFETY: same as above and the two descriptors are distinct.
    let write = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    Ok((read, write))
}

fn duplicate_high(fd: RawFd) -> Result<OwnedFd, ObservedEvaluatorError> {
    // SAFETY: fd is open and fcntl returns a fresh descriptor on success.
    let duplicated = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, HIGH_FD_MIN) };
    if duplicated < 0 {
        return Err(ObservedEvaluatorError::Control(std::io::Error::last_os_error()));
    }
    // SAFETY: duplicated is a fresh descriptor returned by fcntl.
    Ok(unsafe { OwnedFd::from_raw_fd(duplicated) })
}

fn process_group_exists(pgid: u32) -> Result<bool, ObservedEvaluatorError> {
    let pgid = i32::try_from(pgid).map_err(|_| ObservedEvaluatorError::MeasurementOverflow)?;
    // SAFETY: signal 0 performs an existence/permission check without delivering a signal.
    let result = unsafe { libc::kill(-pgid, 0) };
    if result == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(ObservedEvaluatorError::Control(error)),
    }
}

fn terminate_pre_gate_group(
    pgid: u32,
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<(), ObservedEvaluatorError> {
    let pgid_i32 = i32::try_from(pgid).map_err(|_| ObservedEvaluatorError::MeasurementOverflow)?;
    // SAFETY: setpgid(0,0) in pre_exec established the Bubblewrap monitor as this group leader;
    // before release, the sandbox child has not called setsid and remains in the same group.
    let result = unsafe { libc::kill(-pgid_i32, libc::SIGKILL) };
    if result < 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) {
        return Err(ObservedEvaluatorError::Control(std::io::Error::last_os_error()));
    }
    let _ = child.kill();
    let _ = child.wait();
    let deadline = Instant::now() + timeout;
    loop {
        if !process_group_exists(pgid)? {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(ObservedEvaluatorError::PreGateTeardownUnverified);
        }
        thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }
}

fn hardened_observed_args(
    runner_path: &Path,
    model_args: &[String],
) -> Result<Vec<String>, ObservedEvaluatorError> {
    let runner_source = runner_path.to_str().ok_or(ObservedEvaluatorError::NonUtf8Path)?;
    let mut args = vec![
        "--die-with-parent".to_string(),
        "--json-status-fd".to_string(),
        STATUS_FD.to_string(),
        "--block-fd".to_string(),
        BLOCK_FD.to_string(),
        "--new-session".to_string(),
        "--unshare-user".to_string(),
        "--disable-userns".to_string(),
        "--assert-userns-disabled".to_string(),
        "--unshare-pid".to_string(),
        "--unshare-uts".to_string(),
        "--unshare-ipc".to_string(),
        "--unshare-net".to_string(),
        "--unshare-cgroup".to_string(),
        "--hostname".to_string(),
        SANDBOX_HOSTNAME.to_string(),
        "--cap-drop".to_string(),
        "ALL".to_string(),
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
        "--remount-ro".to_string(),
        "/".to_string(),
        "--chdir".to_string(),
        SANDBOX_TMP.to_string(),
        "--clearenv".to_string(),
        "--setenv".to_string(),
        "HOME".to_string(),
        SANDBOX_TMP.to_string(),
        "--setenv".to_string(),
        "TMPDIR".to_string(),
        SANDBOX_TMP.to_string(),
        "--".to_string(),
        SANDBOX_RUNNER.to_string(),
    ];
    args.extend(model_args.iter().cloned());
    Ok(args)
}

fn ordered_argv_id(args: &[String]) -> ContentId {
    let mut parts = vec![(args.len() as u64).to_be_bytes().to_vec()];
    for arg in args {
        parts.push((arg.len() as u64).to_be_bytes().to_vec());
        parts.push(arg.as_bytes().to_vec());
    }
    ContentId::derive(
        "symthaea.forge-kernel-gated-bubblewrap-argv.v1",
        parts.iter().map(Vec::as_slice),
    )
}

fn poll_readable(fd: RawFd, timeout: Duration) -> Result<bool, ObservedEvaluatorError> {
    let millis = timeout.as_millis().min(i32::MAX as u128) as i32;
    let mut pollfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: pollfd is valid for one element for the duration of the call.
    let result = unsafe { libc::poll(&mut pollfd, 1, millis) };
    if result < 0 {
        return Err(ObservedEvaluatorError::Control(std::io::Error::last_os_error()));
    }
    Ok(result > 0 && (pollfd.revents & (libc::POLLIN | libc::POLLHUP)) != 0)
}

fn read_status_until_child_pid(
    status: &mut File,
    deadline: Instant,
) -> Result<(u32, Vec<u8>), ObservedEvaluatorError> {
    let mut raw = Vec::new();
    let mut parsed_offset = 0usize;
    loop {
        while let Some(relative_newline) = raw[parsed_offset..].iter().position(|byte| *byte == b'\n') {
            let end = parsed_offset + relative_newline + 1;
            let line = &raw[parsed_offset..end];
            parsed_offset = end;
            let event: StatusEvent = serde_json::from_slice(line)
                .map_err(ObservedEvaluatorError::InvalidStatusJson)?;
            if let Some(child_pid) = event.child_pid {
                return Ok((child_pid, raw));
            }
        }
        let now = Instant::now();
        if now >= deadline {
            return Err(ObservedEvaluatorError::StatusTimeout);
        }
        if !poll_readable(status.as_raw_fd(), deadline.saturating_duration_since(now))? {
            return Err(ObservedEvaluatorError::StatusTimeout);
        }
        let mut buffer = [0u8; 4096];
        let read = status.read(&mut buffer).map_err(|source| ObservedEvaluatorError::Io {
            path: PathBuf::from("<bubblewrap-json-status>"),
            source,
        })?;
        if read == 0 {
            return Err(ObservedEvaluatorError::MissingChildPid);
        }
        raw.extend_from_slice(&buffer[..read]);
        if raw.len() as u64 > MAX_STATUS_BYTES {
            return Err(ObservedEvaluatorError::StatusTooLarge);
        }
    }
}

struct PidFd {
    fd: OwnedFd,
    armed: bool,
}

impl PidFd {
    fn open(pid: u32) -> Result<Self, ObservedEvaluatorError> {
        let pid = i32::try_from(pid).map_err(|_| ObservedEvaluatorError::MeasurementOverflow)?;
        // SAFETY: direct Linux syscall with scalar arguments; success returns a fresh owned fd.
        let result = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0) };
        if result < 0 {
            return Err(ObservedEvaluatorError::PidfdOpen(std::io::Error::last_os_error()));
        }
        // SAFETY: positive syscall return is a fresh pidfd owned by the caller.
        let fd = unsafe { OwnedFd::from_raw_fd(result as RawFd) };
        Ok(Self { fd, armed: true })
    }

    fn kill(&self) -> Result<(), ObservedEvaluatorError> {
        // SAFETY: pidfd identifies the exact process; null siginfo and flags=0 are permitted.
        let result = unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                self.fd.as_raw_fd(),
                libc::SIGKILL,
                std::ptr::null::<libc::siginfo_t>(),
                0,
            )
        };
        if result < 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ESRCH) {
                return Ok(());
            }
            return Err(ObservedEvaluatorError::PidfdSignal(error));
        }
        Ok(())
    }

    fn wait_exited(&self, timeout: Duration) -> Result<bool, ObservedEvaluatorError> {
        poll_readable(self.fd.as_raw_fd(), timeout)
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PidFd {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // SAFETY: best-effort fail-closed cleanup for the exact process handle during unwinding or
        // an unexpected early return. Explicit terminal paths still perform verified teardown.
        let _ = unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                self.fd.as_raw_fd(),
                libc::SIGKILL,
                std::ptr::null::<libc::siginfo_t>(),
                0,
            )
        };
    }
}

fn wait_for_kernel_gate(
    child: &mut std::process::Child,
    sandbox_pid: u32,
    deadline: Instant,
) -> Result<(KernelSandboxObservation, KernelIsolationGate), ObservedEvaluatorError> {
    loop {
        if let Some(status) = child.try_wait().map_err(|source| ObservedEvaluatorError::Io {
            path: PathBuf::from("<bubblewrap-monitor>"),
            source,
        })? {
            return Err(ObservedEvaluatorError::SandboxExitedBeforeGate { code: status.code() });
        }
        let last_error = match observe_sandbox_process(sandbox_pid) {
            Ok(observation) => match KernelIsolationGate::issue(&observation) {
                Ok(gate) => return Ok((observation, gate)),
                Err(error) => error.to_string(),
            },
            Err(error) => error.to_string(),
        };
        if Instant::now() >= deadline {
            return Err(ObservedEvaluatorError::KernelGateTimeout { last_error });
        }
        thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }
}

enum WaitOutcome {
    Exited(ExitStatus, u64),
    TimedOut,
}

fn wait_bounded_from(
    child: &mut std::process::Child,
    started: Instant,
    max_wall_time_ms: u64,
) -> Result<WaitOutcome, ObservedEvaluatorError> {
    let timeout = Duration::from_millis(max_wall_time_ms);
    loop {
        if let Some(status) = child.try_wait().map_err(|source| ObservedEvaluatorError::Io {
            path: PathBuf::from("<bubblewrap-monitor>"),
            source,
        })? {
            let elapsed = u64::try_from(started.elapsed().as_millis())
                .map_err(|_| ObservedEvaluatorError::MeasurementOverflow)?;
            return Ok(WaitOutcome::Exited(status, elapsed));
        }
        if started.elapsed() >= timeout {
            return Ok(WaitOutcome::TimedOut);
        }
        thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }
}

fn terminate_exact_sandbox(
    pidfd: &mut PidFd,
    child: &mut std::process::Child,
    teardown_timeout: Duration,
) -> Result<(), ObservedEvaluatorError> {
    let signal_result = pidfd.kill();
    let _ = child.kill();
    let _ = child.wait();
    if !pidfd.wait_exited(teardown_timeout)? {
        return Err(ObservedEvaluatorError::TeardownUnverified);
    }
    pidfd.disarm();
    signal_result
}

fn shell_exit_code(status: &ExitStatus) -> Option<i32> {
    status.code().or_else(|| status.signal().map(|signal| 128 + signal))
}

fn parse_status_summary(transcript: &[u8]) -> Result<(u32, i32), ObservedEvaluatorError> {
    let text = std::str::from_utf8(transcript).map_err(|_| ObservedEvaluatorError::StatusNotUtf8)?;
    let mut child_pid = None;
    let mut exit_code = None;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let event: StatusEvent = serde_json::from_str(line)
            .map_err(ObservedEvaluatorError::InvalidStatusJson)?;
        if let Some(pid) = event.child_pid {
            child_pid = Some(pid);
        }
        if let Some(code) = event.exit_code {
            exit_code = Some(code);
        }
    }
    Ok((
        child_pid.ok_or(ObservedEvaluatorError::MissingChildPid)?,
        exit_code.ok_or(ObservedEvaluatorError::MissingStatusExit)?,
    ))
}

fn parse_wire_response(
    request: &ForgeProposalEvaluationRequest,
    stdout: &[u8],
) -> Result<(ContentId, Vec<ForgeProposalEvaluatorPrediction>), ObservedEvaluatorError> {
    let wire: WireResponseV1 = serde_json::from_slice(stdout)
        .map_err(ObservedEvaluatorError::InvalidResponseJson)?;
    if wire.schema != WIRE_RESPONSE_SCHEMA_V1
        || wire.request_id != request.id().as_str()
        || wire.protocol_id != request.protocol().id().as_str()
    {
        return Err(ObservedEvaluatorError::WireScopeMismatch);
    }
    if wire.execution_context.is_empty() {
        return Err(ObservedEvaluatorError::EmptyExecutionContext);
    }
    let rows = request
        .feature_rows()
        .iter()
        .map(|row| (row.target_id().as_str().to_string(), row))
        .collect::<BTreeMap<_, _>>();
    if rows.len() != request.feature_rows().len() || wire.predictions.len() != rows.len() {
        return Err(ObservedEvaluatorError::WirePredictionCoverageMismatch);
    }
    let mut seen = BTreeSet::new();
    let mut predictions = Vec::with_capacity(wire.predictions.len());
    for prediction in wire.predictions {
        if !seen.insert(prediction.target_id.clone()) {
            return Err(ObservedEvaluatorError::WirePredictionCoverageMismatch);
        }
        let row = rows
            .get(&prediction.target_id)
            .ok_or(ObservedEvaluatorError::WirePredictionCoverageMismatch)?;
        predictions.push(ForgeProposalEvaluatorPrediction::for_feature_row(
            request,
            row,
            prediction.probability_scaled,
        )?);
    }
    if seen.len() != rows.len() || !rows.keys().all(|target| seen.contains(target)) {
        return Err(ObservedEvaluatorError::WirePredictionCoverageMismatch);
    }
    let execution_context_id = ContentId::derive(
        "symthaea.forge-kernel-gated-evaluator-execution-context.v1",
        [wire.execution_context.as_bytes()],
    );
    Ok((execution_context_id, predictions))
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelGatedExecutionReceipt {
    id: ContentId,
    policy_id: ContentId,
    bubblewrap_policy_id: ContentId,
    executable_model_binding_id: ContentId,
    model_id: ContentId,
    model_payload_id: ContentId,
    bubblewrap_artifact_id: ContentId,
    bubblewrap_argv: Vec<String>,
    bubblewrap_argv_id: ContentId,
    runner_source_path: String,
    model_args: Vec<String>,
    request_id: ContentId,
    response_id: ContentId,
    request_wire_id: ContentId,
    stdout_wire_id: ContentId,
    stderr_artifact_id: ContentId,
    status_transcript: String,
    status_transcript_id: ContentId,
    execution_context_id: ContentId,
    sandbox_host_pid: u32,
    kernel_observation_id: ContentId,
    kernel_gate_id: ContentId,
    gate_wait_ms: u64,
    wall_time_ms: u64,
    stderr_bytes: u64,
    pidfd_exit_verified: bool,
}

impl KernelGatedExecutionReceipt {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn policy_id(&self) -> &ContentId { &self.policy_id }
    pub fn sandbox_host_pid(&self) -> u32 { self.sandbox_host_pid }
    pub fn kernel_observation_id(&self) -> &ContentId { &self.kernel_observation_id }
    pub fn kernel_gate_id(&self) -> &ContentId { &self.kernel_gate_id }
    pub fn gate_wait_ms(&self) -> u64 { self.gate_wait_ms }
    pub fn wall_time_ms(&self) -> u64 { self.wall_time_ms }
    pub fn pidfd_exit_verified(&self) -> bool { self.pidfd_exit_verified }
    pub fn status_transcript(&self) -> &str { &self.status_transcript }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_for(
        &self,
        policy: &KernelGatedEvaluatorPolicy,
        binding: &ForgeProposalExecutableModelBinding,
        model: &ForgeProposalFrozenModel,
        request: &ForgeProposalEvaluationRequest,
        response: &ForgeProposalEvaluationResponse,
        observation: &KernelSandboxObservation,
        gate: &KernelIsolationGate,
    ) -> Result<(), ObservedEvaluatorError> {
        policy.validate()?;
        binding.validate_for(model, request.protocol())?;
        response.validate_for(request)?;
        gate.validate_for(observation)?;
        if self.policy_id != *policy.id()
            || self.bubblewrap_policy_id != *policy.bubblewrap_policy().id()
            || self.executable_model_binding_id != *binding.id()
            || self.model_id != *model.id()
            || self.model_payload_id != *model.model_payload_id()
            || self.bubblewrap_artifact_id != *policy.bubblewrap_policy().bubblewrap_artifact_id()
            || self.request_id != *request.id()
            || self.response_id != *response.id()
            || self.execution_context_id != *response.execution_context_id()
            || self.sandbox_host_pid != observation.host_pid()
            || self.kernel_observation_id != *observation.id()
            || self.kernel_gate_id != *gate.id()
            || self.gate_wait_ms > policy.kernel_gate_timeout_ms()
            || self.wall_time_ms > policy.bubblewrap_policy().max_wall_time_ms()
            || self.stderr_bytes > policy.bubblewrap_policy().max_stderr_bytes()
            || !self.pidfd_exit_verified
        {
            return Err(ObservedEvaluatorError::ReceiptScopeMismatch);
        }
        if request.protocol().runner_configuration_id() != &forge_evaluator_runner_argv_id(&self.model_args) {
            return Err(ObservedEvaluatorError::ReceiptScopeMismatch);
        }
        let expected_argv = hardened_observed_args(Path::new(&self.runner_source_path), &self.model_args)?;
        if self.bubblewrap_argv != expected_argv
            || self.bubblewrap_argv_id != ordered_argv_id(&expected_argv)
        {
            return Err(ObservedEvaluatorError::ReceiptScopeMismatch);
        }
        let request_wire = serde_json::to_vec(request)
            .map_err(ObservedEvaluatorError::InvalidResponseJson)?;
        let expected_request_wire_id = ContentId::derive(
            "symthaea.forge-kernel-gated-observed-stdin.v1",
            [request_wire.as_slice()],
        );
        let expected_status_id = ContentId::derive(
            "symthaea.forge-kernel-gated-status-transcript.v1",
            [self.status_transcript.as_bytes()],
        );
        let (status_pid, status_exit) = parse_status_summary(self.status_transcript.as_bytes())?;
        if self.request_wire_id != expected_request_wire_id
            || self.status_transcript_id != expected_status_id
            || status_pid != self.sandbox_host_pid
            || status_exit != 0
        {
            return Err(ObservedEvaluatorError::ReceiptScopeMismatch);
        }
        let expected = derive_receipt_id(
            &self.policy_id,
            &self.bubblewrap_policy_id,
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
            &self.status_transcript_id,
            &self.execution_context_id,
            self.sandbox_host_pid,
            &self.kernel_observation_id,
            &self.kernel_gate_id,
            self.gate_wait_ms,
            self.wall_time_ms,
            self.stderr_bytes,
            self.pidfd_exit_verified,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ObservedEvaluatorError::ReceiptIdentityMismatch)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_receipt_id(
    policy_id: &ContentId,
    bubblewrap_policy_id: &ContentId,
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
    status_transcript_id: &ContentId,
    execution_context_id: &ContentId,
    sandbox_host_pid: u32,
    kernel_observation_id: &ContentId,
    kernel_gate_id: &ContentId,
    gate_wait_ms: u64,
    wall_time_ms: u64,
    stderr_bytes: u64,
    pidfd_exit_verified: bool,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-kernel-gated-execution-receipt.v1",
        [
            policy_id.as_str().as_bytes(),
            bubblewrap_policy_id.as_str().as_bytes(),
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
            status_transcript_id.as_str().as_bytes(),
            execution_context_id.as_str().as_bytes(),
            sandbox_host_pid.to_be_bytes().as_slice(),
            kernel_observation_id.as_str().as_bytes(),
            kernel_gate_id.as_str().as_bytes(),
            gate_wait_ms.to_be_bytes().as_slice(),
            wall_time_ms.to_be_bytes().as_slice(),
            stderr_bytes.to_be_bytes().as_slice(),
            &[u8::from(pidfd_exit_verified)],
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_kernel_gated_evaluator(
    policy: &KernelGatedEvaluatorPolicy,
    binding: &ForgeProposalExecutableModelBinding,
    request: &ForgeProposalEvaluationRequest,
    model: &ForgeProposalFrozenModel,
    bubblewrap_executable: impl AsRef<Path>,
    model_executable: impl AsRef<Path>,
    model_args: &[String],
) -> Result<(
    ForgeProposalEvaluationResponse,
    KernelSandboxObservation,
    KernelIsolationGate,
    KernelGatedExecutionReceipt,
), ObservedEvaluatorError> {
    policy.validate()?;
    request.validate_identity()?;
    binding.validate_for(model, request.protocol())?;
    if request.model_id() != model.id()
        || request.protocol().transport_schema_id() != policy.bubblewrap_policy().transport_schema_id()
        || request.protocol().transport_schema_id() != &forge_direct_evaluator_transport_schema_id()
    {
        return Err(ObservedEvaluatorError::TransportSchemaMismatch);
    }

    let (canonical_bwrap, bwrap_bytes) = canonical_file_bytes(bubblewrap_executable.as_ref())?;
    if forge_bubblewrap_artifact_id(&bwrap_bytes) != *policy.bubblewrap_policy().bubblewrap_artifact_id() {
        return Err(ObservedEvaluatorError::BubblewrapArtifactMismatch);
    }
    let (canonical_model, model_bytes) = canonical_file_bytes(model_executable.as_ref())?;
    if forge_evaluator_runner_artifact_id(&model_bytes) != *model.model_payload_id() {
        return Err(ObservedEvaluatorError::ModelArtifactMismatch);
    }
    if request.protocol().runner_configuration_id() != &forge_evaluator_runner_argv_id(model_args) {
        return Err(ObservedEvaluatorError::ModelArgvMismatch);
    }

    let request_wire = serde_json::to_vec(request)
        .map_err(ObservedEvaluatorError::InvalidResponseJson)?;
    if request_wire.len() as u64 > policy.bubblewrap_policy().max_request_bytes() {
        return Err(ObservedEvaluatorError::RequestTooLarge);
    }

    let (status_read, status_write) = pipe_cloexec()?;
    let (block_read, block_write) = pipe_cloexec()?;
    let child_status = duplicate_high(status_write.as_raw_fd())?;
    let child_block = duplicate_high(block_read.as_raw_fd())?;
    drop(status_write);
    drop(block_read);

    let bwrap_args = hardened_observed_args(&canonical_model, model_args)?;
    let bwrap_argv_id = ordered_argv_id(&bwrap_args);
    let runner_source_path = canonical_model
        .to_str()
        .ok_or(ObservedEvaluatorError::NonUtf8Path)?
        .to_string();
    let workdir = FreshWorkDir::create(request.id())?;
    let launch_started = Instant::now();
    let gate_deadline = launch_started + Duration::from_millis(policy.kernel_gate_timeout_ms());
    let status_src = child_status.as_raw_fd();
    let block_src = child_block.as_raw_fd();
    let mut command = Command::new(&canonical_bwrap);
    command
        .args(&bwrap_args)
        .env_clear()
        .current_dir(&workdir.path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // SAFETY: the closure performs only setpgid/dup2/close operations between fork and exec.
    unsafe {
        command.pre_exec(move || {
            if libc::setpgid(0, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::dup2(status_src, STATUS_FD) < 0 || libc::dup2(block_src, BLOCK_FD) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if status_src != STATUS_FD {
                libc::close(status_src);
            }
            if block_src != BLOCK_FD {
                libc::close(block_src);
            }
            Ok(())
        });
    }
    let mut child = command.spawn().map_err(|source| ObservedEvaluatorError::Io {
        path: canonical_bwrap.clone(),
        source,
    })?;
    let process_group = child.id();
    drop(child_status);
    drop(child_block);

    let (mut stdin, stdout, stderr) = match (child.stdin.take(), child.stdout.take(), child.stderr.take()) {
        (Some(stdin), Some(stdout), Some(stderr)) => (stdin, stdout, stderr),
        _ => {
            terminate_pre_gate_group(
                process_group,
                &mut child,
                Duration::from_millis(policy.teardown_timeout_ms()),
            )?;
            return Err(ObservedEvaluatorError::MissingPipe);
        }
    };
    let stdout_cap = policy.bubblewrap_policy().max_response_bytes();
    let stderr_cap = policy.bubblewrap_policy().max_stderr_bytes();
    let stdout_reader = thread::spawn(move || read_capped_and_drain(stdout, stdout_cap));
    let stderr_reader = thread::spawn(move || read_capped_and_drain(stderr, stderr_cap));

    let mut status_file = File::from(status_read);
    let (sandbox_pid, status_prefix) = match read_status_until_child_pid(&mut status_file, gate_deadline) {
        Ok(value) => value,
        Err(error) => {
            let teardown = terminate_pre_gate_group(
                process_group,
                &mut child,
                Duration::from_millis(policy.teardown_timeout_ms()),
            );
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            teardown?;
            return Err(error);
        }
    };
    let mut pidfd = match PidFd::open(sandbox_pid) {
        Ok(pidfd) => pidfd,
        Err(error) => {
            let teardown = terminate_pre_gate_group(
                process_group,
                &mut child,
                Duration::from_millis(policy.teardown_timeout_ms()),
            );
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            teardown?;
            return Err(error);
        }
    };
    let status_reader = thread::spawn(move || read_capped_and_drain(status_file, MAX_STATUS_BYTES));

    if Instant::now() >= gate_deadline {
        let teardown = terminate_exact_sandbox(
            &mut pidfd,
            &mut child,
            Duration::from_millis(policy.teardown_timeout_ms()),
        );
        let _ = stdout_reader.join();
        let _ = stderr_reader.join();
        let _ = status_reader.join();
        teardown?;
        return Err(ObservedEvaluatorError::KernelGateTimeout {
            last_error: "status handshake consumed the entire pre-exec gate budget".to_string(),
        });
    }

    let (observation, gate) = match wait_for_kernel_gate(&mut child, sandbox_pid, gate_deadline) {
        Ok(value) => value,
        Err(error) => {
            let teardown = terminate_exact_sandbox(
                &mut pidfd,
                &mut child,
                Duration::from_millis(policy.teardown_timeout_ms()),
            );
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            let _ = status_reader.join();
            teardown?;
            return Err(error);
        }
    };
    let gate_wait_ms = u64::try_from(launch_started.elapsed().as_millis())
        .map_err(|_| ObservedEvaluatorError::MeasurementOverflow)?;

    let mut release = File::from(block_write);
    if release.write_all(&[1]).and_then(|_| release.flush()).is_err() {
        let teardown = terminate_exact_sandbox(
            &mut pidfd,
            &mut child,
            Duration::from_millis(policy.teardown_timeout_ms()),
        );
        let _ = stdout_reader.join();
        let _ = stderr_reader.join();
        let _ = status_reader.join();
        teardown?;
        return Err(ObservedEvaluatorError::GateReleaseFailed);
    }
    drop(release);

    let stdin_bytes = request_wire.clone();
    let writer = thread::spawn(move || -> Result<(), std::io::Error> {
        stdin.write_all(&stdin_bytes)?;
        stdin.flush()?;
        Ok(())
    });

    let wait = match wait_bounded_from(
        &mut child,
        launch_started,
        policy.bubblewrap_policy().max_wall_time_ms(),
    ) {
        Ok(wait) => wait,
        Err(error) => {
            let teardown = terminate_exact_sandbox(
                &mut pidfd,
                &mut child,
                Duration::from_millis(policy.teardown_timeout_ms()),
            );
            let _ = writer.join();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            let _ = status_reader.join();
            teardown?;
            return Err(error);
        }
    };

    if matches!(wait, WaitOutcome::TimedOut) {
        let teardown = terminate_exact_sandbox(
            &mut pidfd,
            &mut child,
            Duration::from_millis(policy.teardown_timeout_ms()),
        );
        let _ = writer.join();
        let _ = stdout_reader.join();
        let _ = stderr_reader.join();
        let _ = status_reader.join();
        teardown?;
        return Err(ObservedEvaluatorError::TimedOutAfterVerifiedTeardown);
    }

    if !pidfd.wait_exited(Duration::from_millis(policy.teardown_timeout_ms()))? {
        return Err(ObservedEvaluatorError::TeardownUnverified);
    }
    pidfd.disarm();

    let writer_result = writer.join().map_err(|_| ObservedEvaluatorError::StdinWriterFailed)?;
    writer_result.map_err(|_| ObservedEvaluatorError::StdinWriterFailed)?;
    let stdout_result = stdout_reader
        .join()
        .map_err(|_| ObservedEvaluatorError::ReaderThreadPanicked)?
        .map_err(|source| ObservedEvaluatorError::Io {
            path: PathBuf::from("<kernel-gated-evaluator-stdout>"),
            source,
        })?;
    let stderr_result = stderr_reader
        .join()
        .map_err(|_| ObservedEvaluatorError::ReaderThreadPanicked)?
        .map_err(|source| ObservedEvaluatorError::Io {
            path: PathBuf::from("<kernel-gated-evaluator-stderr>"),
            source,
        })?;
    let status_suffix = status_reader
        .join()
        .map_err(|_| ObservedEvaluatorError::ReaderThreadPanicked)?
        .map_err(|source| ObservedEvaluatorError::Io {
            path: PathBuf::from("<bubblewrap-json-status>"),
            source,
        })?;

    let (status, wall_time_ms) = match wait {
        WaitOutcome::Exited(status, wall_time_ms) => (status, wall_time_ms),
        WaitOutcome::TimedOut => unreachable!("timeout returned before output interpretation"),
    };
    if stdout_result.exceeded {
        return Err(ObservedEvaluatorError::StdoutLimitExceeded);
    }
    if stderr_result.exceeded {
        return Err(ObservedEvaluatorError::StderrLimitExceeded);
    }
    if status_suffix.exceeded {
        return Err(ObservedEvaluatorError::StatusTooLarge);
    }

    let mut status_transcript_bytes = status_prefix;
    status_transcript_bytes.extend_from_slice(&status_suffix.bytes);
    if status_transcript_bytes.len() as u64 > MAX_STATUS_BYTES {
        return Err(ObservedEvaluatorError::StatusTooLarge);
    }
    let (status_pid, status_exit) = parse_status_summary(&status_transcript_bytes)?;
    if status_pid != sandbox_pid {
        return Err(ObservedEvaluatorError::StatusChildPidMismatch);
    }
    if Some(status_exit) != shell_exit_code(&status) {
        return Err(ObservedEvaluatorError::StatusExitMismatch);
    }
    if !status.success() {
        return Err(ObservedEvaluatorError::RunnerFailed { code: status.code() });
    }

    let (execution_context_id, predictions) = parse_wire_response(request, &stdout_result.bytes)?;
    let response = ForgeProposalEvaluationResponse::freeze(request, execution_context_id, predictions)?;
    let request_wire_id = ContentId::derive(
        "symthaea.forge-kernel-gated-observed-stdin.v1",
        [request_wire.as_slice()],
    );
    let stdout_wire_id = ContentId::derive(
        "symthaea.forge-kernel-gated-observed-stdout.v1",
        [stdout_result.bytes.as_slice()],
    );
    let stderr_artifact_id = ContentId::derive(
        "symthaea.forge-kernel-gated-stderr.v1",
        [stderr_result.bytes.as_slice()],
    );
    let status_transcript = String::from_utf8(status_transcript_bytes)
        .map_err(|_| ObservedEvaluatorError::StatusNotUtf8)?;
    let status_transcript_id = ContentId::derive(
        "symthaea.forge-kernel-gated-status-transcript.v1",
        [status_transcript.as_bytes()],
    );
    let id = derive_receipt_id(
        policy.id(),
        policy.bubblewrap_policy().id(),
        binding.id(),
        model.id(),
        model.model_payload_id(),
        policy.bubblewrap_policy().bubblewrap_artifact_id(),
        &bwrap_argv_id,
        request.id(),
        response.id(),
        &request_wire_id,
        &stdout_wire_id,
        &stderr_artifact_id,
        &status_transcript_id,
        response.execution_context_id(),
        sandbox_pid,
        observation.id(),
        gate.id(),
        gate_wait_ms,
        wall_time_ms,
        stderr_result.total_bytes,
        true,
    );
    let receipt = KernelGatedExecutionReceipt {
        id,
        policy_id: policy.id().clone(),
        bubblewrap_policy_id: policy.bubblewrap_policy().id().clone(),
        executable_model_binding_id: binding.id().clone(),
        model_id: model.id().clone(),
        model_payload_id: model.model_payload_id().clone(),
        bubblewrap_artifact_id: policy.bubblewrap_policy().bubblewrap_artifact_id().clone(),
        bubblewrap_argv: bwrap_args,
        bubblewrap_argv_id,
        runner_source_path,
        model_args: model_args.to_vec(),
        request_id: request.id().clone(),
        response_id: response.id().clone(),
        request_wire_id,
        stdout_wire_id,
        stderr_artifact_id,
        status_transcript,
        status_transcript_id,
        execution_context_id: response.execution_context_id().clone(),
        sandbox_host_pid: sandbox_pid,
        kernel_observation_id: observation.id().clone(),
        kernel_gate_id: gate.id().clone(),
        gate_wait_ms,
        wall_time_ms,
        stderr_bytes: stderr_result.total_bytes,
        pidfd_exit_verified: true,
    };
    receipt.validate_for(policy, binding, model, request, &response, &observation, &gate)?;
    Ok((response, observation, gate, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_recipe_contains_status_block_and_strict_isolation_before_command() {
        let args = hardened_observed_args(Path::new("/nix/store/example-runner/bin/runner"), &[]).unwrap();
        let status = args.iter().position(|value| value == "--json-status-fd").unwrap();
        let block = args.iter().position(|value| value == "--block-fd").unwrap();
        let cap_drop = args.iter().position(|value| value == "--cap-drop").unwrap();
        let command = args.iter().position(|value| value == "--").unwrap();
        assert!(status < command);
        assert!(block < command);
        assert!(cap_drop < command);
        assert_eq!(args[status + 1], STATUS_FD.to_string());
        assert_eq!(args[block + 1], BLOCK_FD.to_string());
        assert_eq!(args[cap_drop + 1], "ALL");
    }

    #[test]
    fn observed_argv_identity_is_order_sensitive() {
        let a = ordered_argv_id(&[
            "--json-status-fd".into(),
            "3".into(),
            "--block-fd".into(),
            "4".into(),
        ]);
        let b = ordered_argv_id(&[
            "--block-fd".into(),
            "4".into(),
            "--json-status-fd".into(),
            "3".into(),
        ]);
        assert_ne!(a, b);
    }

    #[test]
    fn status_summary_requires_both_child_and_exit() {
        assert!(matches!(
            parse_status_summary(b"{ \"child-pid\": 42 }\n"),
            Err(ObservedEvaluatorError::MissingStatusExit)
        ));
        assert_eq!(
            parse_status_summary(b"{ \"child-pid\": 42 }\n{ \"exit-code\": 0 }\n").unwrap(),
            (42, 0)
        );
    }
}
