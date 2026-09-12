// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(target_os = "linux")]
//! Parent-observed, kernel-gated Bubblewrap execution for Forge evaluators.
//!
//! The model executable is held behind Bubblewrap's `--block-fd`. The parent receives the host
//! sandbox PID through `--json-status-fd`, opens an exact pidfd, independently observes the
//! process through `/proc`, and releases the block only if `KernelIsolationGate` succeeds inside
//! one frozen launch-to-release budget. Before the sandbox PID is known, failures tear down a
//! dedicated process group and verify that group is empty. After the PID is known, pidfd becomes
//! the stable kill/exit handle and remains armed until exact exit is verified.
//!
//! This is stronger than a command-line recipe receipt, but remains deliberately narrower than a
//! full hostile-code sandbox theorem: seccomp filtering, Landlock, cgroup resource limits, VM
//! isolation, kernel-exploit resistance, and independent Bubblewrap semantic correctness remain
//! unestablished here.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
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
const NIX_STORE: &str = "/nix/store";
const SANDBOX_RUNNER: &str = "/runner";
const SANDBOX_TMP: &str = "/tmp";
const SANDBOX_HOSTNAME: &str = "symthaea-forge-evaluator";
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
    #[error("kernel-gated policy exceeds hard safety ceilings")]
    UnsafePolicyLimits,
    #[error("evaluator request exceeds the frozen request-size limit")]
    RequestTooLarge,
    #[error("Linux control primitive failed: {0}")]
    Control(std::io::Error),
    #[error("Bubblewrap process did not expose required stdio")]
    MissingPipe,
    #[error("pre-gate Bubblewrap process-group teardown could not be verified")]
    PreGateTeardownUnverified,
    #[error("Bubblewrap status stream timed out before reporting child-pid")]
    StatusTimeout,
    #[error("Bubblewrap status stream exceeded its hard size ceiling")]
    StatusTooLarge,
    #[error("Bubblewrap status JSON is invalid: {0}")]
    InvalidStatusJson(serde_json::Error),
    #[error("Bubblewrap status transcript is not UTF-8")]
    StatusNotUtf8,
    #[error("Bubblewrap status stream ended without child-pid")]
    MissingChildPid,
    #[error("Bubblewrap status stream did not report terminal exit-code")]
    MissingStatusExit,
    #[error("Bubblewrap status transcript does not bind the reported sandbox PID")]
    StatusChildPidMismatch,
    #[error("Bubblewrap status exit-code disagrees with wrapper exit status")]
    StatusExitMismatch,
    #[error("could not open pidfd for exact sandbox process: {0}")]
    PidfdOpen(std::io::Error),
    #[error("could not signal exact sandbox process through pidfd: {0}")]
    PidfdSignal(std::io::Error),
    #[error("pre-exec gate budget expired before release; last observation error: {last_error}")]
    KernelGateTimeout { last_error: String },
    #[error("sandbox exited before the pre-exec kernel gate with code {code:?}")]
    SandboxExitedBeforeGate { code: Option<i32> },
    #[error("kernel-gate release byte could not be written")]
    GateReleaseFailed,
    #[error("evaluator stdin writer failed or panicked")]
    StdinWriterFailed,
    #[error("evaluator output/status reader thread panicked")]
    ReaderThreadPanicked,
    #[error("evaluator exceeded frozen wall time after verified pidfd teardown")]
    TimedOutAfterVerifiedTeardown,
    #[error("sandbox teardown could not be verified through exact pidfd")]
    TeardownUnverified,
    #[error("evaluator stdout exceeded frozen response-size limit")]
    StdoutLimitExceeded,
    #[error("evaluator stderr exceeded frozen stderr-size limit")]
    StderrLimitExceeded,
    #[error("Bubblewrap/evaluator exited unsuccessfully with code {code:?}")]
    RunnerFailed { code: Option<i32> },
    #[error("evaluator stdout is not valid response JSON: {0}")]
    InvalidResponseJson(serde_json::Error),
    #[error("evaluator wire response does not match request/protocol identity")]
    WireScopeMismatch,
    #[error("evaluator response has duplicate, missing, or unknown targets")]
    WirePredictionCoverageMismatch,
    #[error("evaluator execution-context string must be non-empty")]
    EmptyExecutionContext,
    #[error("measurement cannot be represented in u64")]
    MeasurementOverflow,
    #[error("kernel-gated execution receipt does not bind supplied evidence")]
    ReceiptScopeMismatch,
    #[error("kernel-gated execution receipt identity is non-canonical")]
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
        let id = ContentId::derive(
            "symthaea.forge-kernel-gated-evaluator-policy.v1",
            [
                bubblewrap_policy.id().as_str().as_bytes(),
                kernel_gate_timeout_ms.to_be_bytes().as_slice(),
                teardown_timeout_ms.to_be_bytes().as_slice(),
                b"status-fd+block-fd+single-preexec-budget+verified-pgroup+kernel-gate+pidfd",
            ],
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

type ReaderHandle = JoinHandle<Result<CappedOutput, std::io::Error>>;

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
        exceeded |= total_bytes > cap;
    }
    Ok(CappedOutput { bytes, total_bytes, exceeded })
}

struct FreshWorkDir { path: PathBuf }

impl FreshWorkDir {
    fn create(request_id: &ContentId) -> Result<Self, ObservedEvaluatorError> {
        let epoch_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let nonce = WORKDIR_NONCE.fetch_add(1, Ordering::Relaxed);
        let tag = request_id
            .as_str()
            .bytes()
            .filter(u8::is_ascii_alphanumeric)
            .take(16)
            .map(char::from)
            .collect::<String>();
        for attempt in 0u64..64 {
            let path = std::env::temp_dir().join(format!(
                "symthaea-forge-observed-{}-{epoch_nanos}-{nonce}-{attempt}-{tag}",
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
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.path); }
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
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(ObservedEvaluatorError::Control(std::io::Error::last_os_error()));
    }
    // SAFETY: successful pipe2 returned two fresh, distinct owned descriptors.
    let read = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let write = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    Ok((read, write))
}

fn duplicate_high(fd: RawFd) -> Result<OwnedFd, ObservedEvaluatorError> {
    // SAFETY: `fd` is open; successful fcntl returns a fresh owned descriptor.
    let duplicated = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, HIGH_FD_MIN) };
    if duplicated < 0 {
        return Err(ObservedEvaluatorError::Control(std::io::Error::last_os_error()));
    }
    // SAFETY: `duplicated` is fresh and now owned here.
    Ok(unsafe { OwnedFd::from_raw_fd(duplicated) })
}

fn process_group_exists(pgid: u32) -> Result<bool, ObservedEvaluatorError> {
    let pgid = i32::try_from(pgid).map_err(|_| ObservedEvaluatorError::MeasurementOverflow)?;
    // SAFETY: signal 0 performs only an existence/permission check.
    if unsafe { libc::kill(-pgid, 0) } == 0 {
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
    // SAFETY: pre_exec made Bubblewrap this process-group leader; before block release its setup
    // child has not called setsid, so group SIGKILL covers the pre-gate tree.
    if unsafe { libc::kill(-pgid_i32, libc::SIGKILL) } < 0 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            return Err(ObservedEvaluatorError::Control(error));
        }
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
    let runner = runner_path.to_str().ok_or(ObservedEvaluatorError::NonUtf8Path)?;
    let mut args = vec![
        "--die-with-parent".into(), "--json-status-fd".into(), STATUS_FD.to_string(),
        "--block-fd".into(), BLOCK_FD.to_string(), "--new-session".into(),
        "--unshare-user".into(), "--disable-userns".into(), "--assert-userns-disabled".into(),
        "--unshare-pid".into(), "--unshare-uts".into(), "--unshare-ipc".into(),
        "--unshare-net".into(), "--unshare-cgroup".into(),
        "--hostname".into(), SANDBOX_HOSTNAME.into(), "--cap-drop".into(), "ALL".into(),
        "--proc".into(), "/proc".into(), "--dev".into(), "/dev".into(),
        "--tmpfs".into(), SANDBOX_TMP.into(),
        "--ro-bind".into(), NIX_STORE.into(), NIX_STORE.into(),
        "--ro-bind".into(), runner.into(), SANDBOX_RUNNER.into(),
        "--remount-ro".into(), "/".into(), "--chdir".into(), SANDBOX_TMP.into(),
        "--clearenv".into(), "--setenv".into(), "HOME".into(), SANDBOX_TMP.into(),
        "--setenv".into(), "TMPDIR".into(), SANDBOX_TMP.into(),
        "--".into(), SANDBOX_RUNNER.into(),
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
    let millis = i32::try_from(timeout.as_millis().min(i32::MAX as u128))
        .map_err(|_| ObservedEvaluatorError::MeasurementOverflow)?;
    let mut descriptor = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
    // SAFETY: `descriptor` is valid for one element for the duration of poll.
    let result = unsafe { libc::poll(&mut descriptor, 1, millis) };
    if result < 0 {
        return Err(ObservedEvaluatorError::Control(std::io::Error::last_os_error()));
    }
    Ok(result > 0 && (descriptor.revents & (libc::POLLIN | libc::POLLHUP)) != 0)
}

fn read_status_until_child_pid(
    status: &mut File,
    deadline: Instant,
) -> Result<(u32, Vec<u8>), ObservedEvaluatorError> {
    let mut raw = Vec::new();
    let mut parsed = 0usize;
    loop {
        while let Some(relative) = raw[parsed..].iter().position(|byte| *byte == b'\n') {
            let end = parsed + relative + 1;
            let event: StatusEvent = serde_json::from_slice(&raw[parsed..end])
                .map_err(ObservedEvaluatorError::InvalidStatusJson)?;
            parsed = end;
            if let Some(pid) = event.child_pid {
                return Ok((pid, raw));
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

struct PidFd { fd: OwnedFd, armed: bool }

impl PidFd {
    fn open(pid: u32) -> Result<Self, ObservedEvaluatorError> {
        let pid = i32::try_from(pid).map_err(|_| ObservedEvaluatorError::MeasurementOverflow)?;
        // SAFETY: direct Linux syscall with scalar arguments; success returns a fresh pidfd.
        let result = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0) };
        if result < 0 {
            return Err(ObservedEvaluatorError::PidfdOpen(std::io::Error::last_os_error()));
        }
        // SAFETY: successful pidfd_open returned a fresh descriptor.
        Ok(Self { fd: unsafe { OwnedFd::from_raw_fd(result as RawFd) }, armed: true })
    }

    fn kill(&self) -> Result<(), ObservedEvaluatorError> {
        // SAFETY: pidfd identifies the exact process; null siginfo and flags=0 are allowed.
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
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(ObservedEvaluatorError::PidfdSignal(error));
            }
        }
        Ok(())
    }

    fn wait_exited(&self, timeout: Duration) -> Result<bool, ObservedEvaluatorError> {
        poll_readable(self.fd.as_raw_fd(), timeout)
    }

    fn disarm(&mut self) { self.armed = false; }
}

impl Drop for PidFd {
    fn drop(&mut self) {
        if !self.armed { return; }
        // SAFETY: best-effort fail-closed kill for this exact process handle during unexpected return.
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
        if Instant::now() >= deadline {
            return Err(ObservedEvaluatorError::KernelGateTimeout {
                last_error: "gate deadline reached before another kernel observation".into(),
            });
        }
        if let Some(status) = child.try_wait().map_err(|source| ObservedEvaluatorError::Io {
            path: PathBuf::from("<bubblewrap-monitor>"), source,
        })? {
            return Err(ObservedEvaluatorError::SandboxExitedBeforeGate { code: status.code() });
        }
        let last_error = match observe_sandbox_process(sandbox_pid) {
            Ok(observation) => match KernelIsolationGate::issue(&observation) {
                Ok(gate) => {
                    if Instant::now() >= deadline {
                        return Err(ObservedEvaluatorError::KernelGateTimeout {
                            last_error: "final kernel observation crossed the frozen gate deadline".into(),
                        });
                    }
                    return Ok((observation, gate));
                }
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

enum WaitOutcome { Exited(ExitStatus, u64), TimedOut }

fn wait_bounded_from(
    child: &mut std::process::Child,
    started: Instant,
    max_wall_time_ms: u64,
) -> Result<WaitOutcome, ObservedEvaluatorError> {
    let timeout = Duration::from_millis(max_wall_time_ms);
    loop {
        if let Some(status) = child.try_wait().map_err(|source| ObservedEvaluatorError::Io {
            path: PathBuf::from("<bubblewrap-monitor>"), source,
        })? {
            let elapsed = u64::try_from(started.elapsed().as_millis())
                .map_err(|_| ObservedEvaluatorError::MeasurementOverflow)?;
            return Ok(WaitOutcome::Exited(status, elapsed));
        }
        if started.elapsed() >= timeout { return Ok(WaitOutcome::TimedOut); }
        thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }
}

fn terminate_exact_sandbox(
    pidfd: &mut PidFd,
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<(), ObservedEvaluatorError> {
    let signal = pidfd.kill();
    let _ = child.kill();
    let _ = child.wait();
    if !pidfd.wait_exited(timeout)? {
        return Err(ObservedEvaluatorError::TeardownUnverified);
    }
    pidfd.disarm();
    signal
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
            if child_pid.replace(pid).is_some() {
                return Err(ObservedEvaluatorError::StatusChildPidMismatch);
            }
        }
        if let Some(code) = event.exit_code {
            if exit_code.replace(code).is_some() {
                return Err(ObservedEvaluatorError::StatusExitMismatch);
            }
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
    let context = ContentId::derive(
        "symthaea.forge-kernel-gated-evaluator-execution-context.v1",
        [wire.execution_context.as_bytes()],
    );
    Ok((context, predictions))
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
    pub fn sandbox_host_pid(&self) -> u32 { self.sandbox_host_pid }
    pub fn kernel_observation_id(&self) -> &ContentId { &self.kernel_observation_id }
    pub fn kernel_gate_id(&self) -> &ContentId { &self.kernel_gate_id }
    pub fn gate_wait_ms(&self) -> u64 { self.gate_wait_ms }
    pub fn wall_time_ms(&self) -> u64 { self.wall_time_ms }
    pub fn pidfd_exit_verified(&self) -> bool { self.pidfd_exit_verified }

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
        if self.bubblewrap_argv != expected_argv || self.bubblewrap_argv_id != ordered_argv_id(&expected_argv) {
            return Err(ObservedEvaluatorError::ReceiptScopeMismatch);
        }
        let request_wire = serde_json::to_vec(request)
            .map_err(ObservedEvaluatorError::InvalidResponseJson)?;
        let expected_request_id = ContentId::derive(
            "symthaea.forge-kernel-gated-observed-stdin.v1",
            [request_wire.as_slice()],
        );
        let expected_status_id = ContentId::derive(
            "symthaea.forge-kernel-gated-status-transcript.v1",
            [self.status_transcript.as_bytes()],
        );
        let (status_pid, status_exit) = parse_status_summary(self.status_transcript.as_bytes())?;
        if self.request_wire_id != expected_request_id
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
        if expected == self.id { Ok(()) } else { Err(ObservedEvaluatorError::ReceiptIdentityMismatch) }
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

fn join_reader(
    handle: ReaderHandle,
    label: &'static str,
) -> Result<CappedOutput, ObservedEvaluatorError> {
    handle
        .join()
        .map_err(|_| ObservedEvaluatorError::ReaderThreadPanicked)?
        .map_err(|source| ObservedEvaluatorError::Io { path: PathBuf::from(label), source })
}

fn join_best_effort(handle: ReaderHandle) { let _ = handle.join(); }

fn teardown_after_pid(
    pidfd: &mut PidFd,
    child: &mut std::process::Child,
    timeout_ms: u64,
    stdout: ReaderHandle,
    stderr: ReaderHandle,
    status: ReaderHandle,
) -> Result<(), ObservedEvaluatorError> {
    let teardown = terminate_exact_sandbox(pidfd, child, Duration::from_millis(timeout_ms));
    join_best_effort(stdout);
    join_best_effort(stderr);
    join_best_effort(status);
    teardown
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
    let runner_source_path = canonical_model.to_str().ok_or(ObservedEvaluatorError::NonUtf8Path)?.to_string();
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
    // SAFETY: closure performs only setpgid/dup2/close between fork and exec.
    unsafe {
        command.pre_exec(move || {
            if libc::setpgid(0, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::dup2(status_src, STATUS_FD) < 0 || libc::dup2(block_src, BLOCK_FD) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if status_src != STATUS_FD { libc::close(status_src); }
            if block_src != BLOCK_FD { libc::close(block_src); }
            Ok(())
        });
    }
    let mut child = command.spawn().map_err(|source| ObservedEvaluatorError::Io {
        path: canonical_bwrap.clone(), source,
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
    let stdout_reader = thread::spawn({
        let cap = policy.bubblewrap_policy().max_response_bytes();
        move || read_capped_and_drain(stdout, cap)
    });
    let stderr_reader = thread::spawn({
        let cap = policy.bubblewrap_policy().max_stderr_bytes();
        move || read_capped_and_drain(stderr, cap)
    });

    let mut status_file = File::from(status_read);
    let (sandbox_pid, status_prefix) = match read_status_until_child_pid(&mut status_file, gate_deadline) {
        Ok(value) => value,
        Err(error) => {
            let teardown = terminate_pre_gate_group(
                process_group,
                &mut child,
                Duration::from_millis(policy.teardown_timeout_ms()),
            );
            join_best_effort(stdout_reader);
            join_best_effort(stderr_reader);
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
            join_best_effort(stdout_reader);
            join_best_effort(stderr_reader);
            teardown?;
            return Err(error);
        }
    };
    let status_reader = thread::spawn(move || read_capped_and_drain(status_file, MAX_STATUS_BYTES));

    if Instant::now() >= gate_deadline {
        teardown_after_pid(
            &mut pidfd, &mut child, policy.teardown_timeout_ms(),
            stdout_reader, stderr_reader, status_reader,
        )?;
        return Err(ObservedEvaluatorError::KernelGateTimeout {
            last_error: "status handshake consumed the pre-exec budget".into(),
        });
    }

    let (observation, gate) = match wait_for_kernel_gate(&mut child, sandbox_pid, gate_deadline) {
        Ok(value) => value,
        Err(error) => {
            teardown_after_pid(
                &mut pidfd, &mut child, policy.teardown_timeout_ms(),
                stdout_reader, stderr_reader, status_reader,
            )?;
            return Err(error);
        }
    };
    let gate_wait_ms = u64::try_from(launch_started.elapsed().as_millis())
        .map_err(|_| ObservedEvaluatorError::MeasurementOverflow)?;

    // Critical release-boundary theorem: a slow final observation must never become a post-hoc
    // receipt failure after model execution. Budget expiry is checked again immediately before the
    // release byte exists, and exact pidfd teardown happens instead of release on overrun.
    if Instant::now() >= gate_deadline || gate_wait_ms > policy.kernel_gate_timeout_ms() {
        teardown_after_pid(
            &mut pidfd, &mut child, policy.teardown_timeout_ms(),
            stdout_reader, stderr_reader, status_reader,
        )?;
        return Err(ObservedEvaluatorError::KernelGateTimeout {
            last_error: "final kernel observation crossed the frozen pre-exec budget".into(),
        });
    }

    let mut release = File::from(block_write);
    if release.write_all(&[1]).and_then(|_| release.flush()).is_err() {
        teardown_after_pid(
            &mut pidfd, &mut child, policy.teardown_timeout_ms(),
            stdout_reader, stderr_reader, status_reader,
        )?;
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
            join_best_effort(stdout_reader);
            join_best_effort(stderr_reader);
            join_best_effort(status_reader);
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
        join_best_effort(stdout_reader);
        join_best_effort(stderr_reader);
        join_best_effort(status_reader);
        teardown?;
        return Err(ObservedEvaluatorError::TimedOutAfterVerifiedTeardown);
    }

    if !pidfd.wait_exited(Duration::from_millis(policy.teardown_timeout_ms()))? {
        return Err(ObservedEvaluatorError::TeardownUnverified);
    }
    pidfd.disarm();

    let writer_result = writer.join().map_err(|_| ObservedEvaluatorError::StdinWriterFailed)?;
    writer_result.map_err(|_| ObservedEvaluatorError::StdinWriterFailed)?;
    let stdout_result = join_reader(stdout_reader, "<kernel-gated-evaluator-stdout>")?;
    let stderr_result = join_reader(stderr_reader, "<kernel-gated-evaluator-stderr>")?;
    let status_suffix = join_reader(status_reader, "<bubblewrap-json-status>")?;

    let (status, wall_time_ms) = match wait {
        WaitOutcome::Exited(status, wall) => (status, wall),
        WaitOutcome::TimedOut => unreachable!("timeout returned before output interpretation"),
    };
    if stdout_result.exceeded { return Err(ObservedEvaluatorError::StdoutLimitExceeded); }
    if stderr_result.exceeded { return Err(ObservedEvaluatorError::StderrLimitExceeded); }
    if status_suffix.exceeded { return Err(ObservedEvaluatorError::StatusTooLarge); }

    let mut status_bytes = status_prefix;
    status_bytes.extend_from_slice(&status_suffix.bytes);
    if status_bytes.len() as u64 > MAX_STATUS_BYTES {
        return Err(ObservedEvaluatorError::StatusTooLarge);
    }
    let (status_pid, status_exit) = parse_status_summary(&status_bytes)?;
    if status_pid != sandbox_pid { return Err(ObservedEvaluatorError::StatusChildPidMismatch); }
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
    let status_transcript = String::from_utf8(status_bytes)
        .map_err(|_| ObservedEvaluatorError::StatusNotUtf8)?;
    let status_transcript_id = ContentId::derive(
        "symthaea.forge-kernel-gated-status-transcript.v1",
        [status_transcript.as_bytes()],
    );
    let id = derive_receipt_id(
        policy.id(), policy.bubblewrap_policy().id(), binding.id(), model.id(),
        model.model_payload_id(), policy.bubblewrap_policy().bubblewrap_artifact_id(),
        &bwrap_argv_id, request.id(), response.id(), &request_wire_id, &stdout_wire_id,
        &stderr_artifact_id, &status_transcript_id, response.execution_context_id(), sandbox_pid,
        observation.id(), gate.id(), gate_wait_ms, wall_time_ms, stderr_result.total_bytes, true,
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
    fn observed_recipe_contains_gate_controls_before_command() {
        let args = hardened_observed_args(Path::new("/nix/store/example/bin/runner"), &[]).unwrap();
        let status = args.iter().position(|v| v == "--json-status-fd").unwrap();
        let block = args.iter().position(|v| v == "--block-fd").unwrap();
        let cap = args.iter().position(|v| v == "--cap-drop").unwrap();
        let command = args.iter().position(|v| v == "--").unwrap();
        assert!(status < command && block < command && cap < command);
        assert_eq!(args[status + 1], STATUS_FD.to_string());
        assert_eq!(args[block + 1], BLOCK_FD.to_string());
        assert_eq!(args[cap + 1], "ALL");
    }

    #[test]
    fn argv_identity_is_order_sensitive() {
        let a = ordered_argv_id(&["--json-status-fd".into(), "3".into(), "--block-fd".into(), "4".into()]);
        let b = ordered_argv_id(&["--block-fd".into(), "4".into(), "--json-status-fd".into(), "3".into()]);
        assert_ne!(a, b);
    }

    #[test]
    fn status_summary_rejects_incomplete_and_duplicate_identity() {
        assert!(matches!(
            parse_status_summary(b"{ \"child-pid\": 42 }\n"),
            Err(ObservedEvaluatorError::MissingStatusExit)
        ));
        assert_eq!(
            parse_status_summary(b"{ \"child-pid\": 42 }\n{ \"exit-code\": 0 }\n").unwrap(),
            (42, 0)
        );
        assert!(matches!(
            parse_status_summary(b"{ \"child-pid\": 42 }\n{ \"child-pid\": 43 }\n{ \"exit-code\": 0 }\n"),
            Err(ObservedEvaluatorError::StatusChildPidMismatch)
        ));
    }
}
