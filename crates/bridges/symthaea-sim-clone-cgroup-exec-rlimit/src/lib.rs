// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent first-instruction cgroup launcher with exact pre-exec rlimits.
//!
//! This is a strict superset theorem over `symthaea-sim-clone-cgroup-exec`:
//! the child is born directly in the target cgroup, receives exact legacy-style
//! kernel rlimits plus a new session/no-new-privs/umask before `execveat`, and
//! then executes the exact immutable sealed image. The older v1 launcher remains
//! unchanged; this crate reconstructs its evidence independently and requires
//! the older verifier to accept it before stronger evidence can be minted.

#![deny(unsafe_op_in_unsafe_fn)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::mem::size_of;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
#[cfg(target_os = "linux")]
use std::os::unix::fs::MetadataExt;
use std::os::unix::process::ExitStatusExt;
use std::process::ExitStatus;
use symthaea_sim_clone_cgroup_exec::{
    SEALED_EXECVEAT_CGROUP_PROFILE_V1, SealedExecveatCgroupError,
    SealedExecveatCgroupEvidence,
};
use symthaea_sim_worker::{SUPERVISOR_PROFILE_V1, SupervisorError, SupervisorLimits};
use symthaea_sim_worker_cgroup::{CgroupV2Error, CgroupV2Lease};
use symthaea_sim_worker_image::{SEALED_WORKER_IMAGE_PROFILE_V1, SealedWorkerImage};
use thiserror::Error;

pub const SEALED_EXECVEAT_CGROUP_RLIMIT_PROFILE_V1: &str =
    "symthaea.simulation.clone-into-cgroup.sealed-execveat-rlimit-v1";

const EVIDENCE_DOMAIN_V1: &[u8] =
    b"symthaea.simulation.clone-into-cgroup.sealed-execveat-rlimit-v1\0";
const BASE_EVIDENCE_DOMAIN_V1: &[u8] =
    b"symthaea.simulation.clone-into-cgroup.sealed-execveat-v1\0";
const CLONE_INTO_CGROUP_FLAG: u64 = 0x2000_0000_0;
const REQUIRED_SEALS: libc::c_int =
    libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
const EMPTY_PATH: &[u8] = b"\0";
const ARGV0: &[u8] = b"symthaea-sealed-cgroup-rlimit-exec\0";

const STAGE_STDIO: u32 = 1;
const STAGE_RLIMIT_AS: u32 = 2;
const STAGE_RLIMIT_CPU: u32 = 3;
const STAGE_RLIMIT_FSIZE: u32 = 4;
const STAGE_RLIMIT_NOFILE: u32 = 5;
const STAGE_RLIMIT_NPROC: u32 = 6;
const STAGE_RLIMIT_CORE: u32 = 7;
const STAGE_SETSID: u32 = 8;
const STAGE_NO_NEW_PRIVS: u32 = 9;
const STAGE_EXEC_FD_CLOEXEC: u32 = 10;
const STAGE_EXECVEAT: u32 = 11;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct CloneArgs {
    flags: u64,
    pidfd: u64,
    child_tid: u64,
    parent_tid: u64,
    exit_signal: u64,
    stack: u64,
    stack_size: u64,
    tls: u64,
    set_tid: u64,
    set_tid_size: u64,
    cgroup: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct KernelRlimit64 {
    current: u64,
    maximum: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreExecSupervisorEnvelope {
    pub address_space_bytes: u64,
    pub cpu_seconds: u64,
    pub file_size_bytes: u64,
    pub open_files: u64,
    pub process_count: u64,
    pub core_bytes: u64,
    pub expected_parent_wall_time_ms: u64,
    pub expected_parent_max_stdout_bytes: u64,
    pub expected_parent_max_stderr_bytes: u64,
}

impl PreExecSupervisorEnvelope {
    fn from_limits(limits: SupervisorLimits) -> Self {
        Self {
            address_space_bytes: limits.address_space_bytes,
            cpu_seconds: limits.cpu_seconds,
            file_size_bytes: limits.file_size_bytes,
            open_files: limits.open_files,
            process_count: limits.process_count,
            core_bytes: 0,
            expected_parent_wall_time_ms: limits.wall_time_ms,
            expected_parent_max_stdout_bytes: limits.max_stdout_bytes,
            expected_parent_max_stderr_bytes: limits.max_stderr_bytes,
        }
    }

    fn as_limits(self) -> SupervisorLimits {
        SupervisorLimits {
            address_space_bytes: self.address_space_bytes,
            cpu_seconds: self.cpu_seconds,
            file_size_bytes: self.file_size_bytes,
            open_files: self.open_files,
            process_count: self.process_count,
            wall_time_ms: self.expected_parent_wall_time_ms,
            max_stdout_bytes: self.expected_parent_max_stdout_bytes,
            max_stderr_bytes: self.expected_parent_max_stderr_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedExecveatCgroupRlimitEvidence {
    pub profile: String,
    pub base: SealedExecveatCgroupEvidence,
    pub supervisor_profile: String,
    pub limits: PreExecSupervisorEnvelope,
    pub legacy_preexec_rlimits_applied: bool,
    pub setsid_before_exec: bool,
    pub no_new_privs_before_exec: bool,
    pub umask_0077_before_exec: bool,
    pub parent_wall_time_enforced_by_this_launcher: bool,
    pub parent_output_limits_enforced_by_this_launcher: bool,
    pub evidence_sha256: String,
}

impl SealedExecveatCgroupRlimitEvidence {
    pub fn verify(&self) -> Result<(), RlimitExecveatCgroupError> {
        if self.profile != SEALED_EXECVEAT_CGROUP_RLIMIT_PROFILE_V1
            || self.supervisor_profile != SUPERVISOR_PROFILE_V1
            || !self.legacy_preexec_rlimits_applied
            || !self.setsid_before_exec
            || !self.no_new_privs_before_exec
            || !self.umask_0077_before_exec
            || self.parent_wall_time_enforced_by_this_launcher
            || self.parent_output_limits_enforced_by_this_launcher
            || self.limits.core_bytes != 0
        {
            return Err(RlimitExecveatCgroupError::InvalidEvidence);
        }
        self.limits.as_limits().validate()?;
        self.base.verify()?;
        if self.base.profile != SEALED_EXECVEAT_CGROUP_PROFILE_V1
            || !self.base.birth_membership_observed
            || !self.base.exec_status_eof_observed
            || !self.base.no_new_privs_before_exec
            || !self.base.executable_fd_closed_on_exec
        {
            return Err(RlimitExecveatCgroupError::InvalidEvidence);
        }
        if self.evidence_sha256 != hex_digest(evidence_sha256_v1(self)?) {
            return Err(RlimitExecveatCgroupError::EvidenceDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct RlimitSealedExecveatCgroupChild {
    pid: u32,
    stdin: Option<File>,
    stdout: File,
    stderr: File,
    evidence: SealedExecveatCgroupRlimitEvidence,
    reaped: bool,
}

impl RlimitSealedExecveatCgroupChild {
    pub const fn pid(&self) -> u32 { self.pid }
    pub fn evidence(&self) -> &SealedExecveatCgroupRlimitEvidence { &self.evidence }
    pub fn stdin_mut(&mut self) -> Option<&mut File> { self.stdin.as_mut() }
    pub fn stdout_mut(&mut self) -> &mut File { &mut self.stdout }
    pub fn stderr_mut(&mut self) -> &mut File { &mut self.stderr }
    pub fn close_stdin(&mut self) { self.stdin.take(); }

    pub fn wait(mut self) -> Result<(ExitStatus, SealedExecveatCgroupRlimitEvidence), RlimitExecveatCgroupError> {
        self.stdin.take();
        let status = wait_pid(self.pid)?;
        self.reaped = true;
        Ok((status, self.evidence.clone()))
    }

    pub fn terminate_and_wait(mut self) -> Result<SealedExecveatCgroupRlimitEvidence, RlimitExecveatCgroupError> {
        self.stdin.take();
        terminate_pid(self.pid);
        let _ = wait_pid(self.pid)?;
        self.reaped = true;
        Ok(self.evidence.clone())
    }
}

impl Drop for RlimitSealedExecveatCgroupChild {
    fn drop(&mut self) {
        if !self.reaped {
            self.stdin.take();
            terminate_pid(self.pid);
            let _ = wait_pid(self.pid);
            self.reaped = true;
        }
    }
}

struct PostCloneReapGuard { pid: u32, armed: bool }
impl PostCloneReapGuard {
    const fn new(pid: u32) -> Self { Self { pid, armed: true } }
    fn disarm(&mut self) { self.armed = false; }
}
impl Drop for PostCloneReapGuard {
    fn drop(&mut self) {
        if self.armed {
            terminate_pid(self.pid);
            let _ = wait_pid(self.pid);
        }
    }
}

pub fn spawn_sealed_image_in_cgroup_with_rlimits(
    image: &SealedWorkerImage,
    lease: &CgroupV2Lease,
    limits: SupervisorLimits,
) -> Result<RlimitSealedExecveatCgroupChild, RlimitExecveatCgroupError> {
    let limits = limits.validate()?;
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = (image, lease, limits);
        return Err(RlimitExecveatCgroupError::UnsupportedPlatform);
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    spawn_linux(image, lease, limits)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn spawn_linux(
    image: &SealedWorkerImage,
    lease: &CgroupV2Lease,
    limits: SupervisorLimits,
) -> Result<RlimitSealedExecveatCgroupChild, RlimitExecveatCgroupError> {
    lease.verify_configuration()?;
    if lease.populated()? { return Err(RlimitExecveatCgroupError::TargetAlreadyPopulated); }
    let target = lease.evidence()?;
    target.verify()?;
    verify_live_image(image)?;

    let cgroup = File::open(lease.leaf()).map_err(RlimitExecveatCgroupError::OpenCgroup)?;
    let metadata = cgroup.metadata().map_err(RlimitExecveatCgroupError::CgroupMetadata)?;
    if metadata.dev() != target.leaf_device || metadata.ino() != target.leaf_inode {
        return Err(RlimitExecveatCgroupError::CgroupIdentityMismatch);
    }
    let executable_fd = image_fd(image)?;
    if executable_fd < 3 { return Err(RlimitExecveatCgroupError::InvalidExecutableFd); }

    let (stdin_read, stdin_write) = pipe_cloexec()?;
    let (stdout_read, stdout_write) = pipe_cloexec()?;
    let (stderr_read, stderr_write) = pipe_cloexec()?;
    let (exec_status_read, exec_status_write) = pipe_cloexec()?;
    let args = CloneArgs {
        flags: CLONE_INTO_CGROUP_FLAG,
        exit_signal: libc::SIGCHLD as u64,
        cgroup: u64::try_from(cgroup.as_raw_fd()).map_err(|_| RlimitExecveatCgroupError::InvalidCgroupFd)?,
        ..CloneArgs::default()
    };
    let result = unsafe { libc::syscall(libc::SYS_clone3, &args as *const CloneArgs, size_of::<CloneArgs>()) };
    if result < 0 { return Err(classify_clone3_error(io::Error::last_os_error())); }
    if result == 0 {
        unsafe {
            child_execveat_with_rlimits(
                executable_fd, cgroup.as_raw_fd(), stdin_read.as_raw_fd(), stdin_write.as_raw_fd(),
                stdout_read.as_raw_fd(), stdout_write.as_raw_fd(), stderr_read.as_raw_fd(),
                stderr_write.as_raw_fd(), exec_status_read.as_raw_fd(), exec_status_write.as_raw_fd(), limits,
            )
        }
    }
    let pid = u32::try_from(result).map_err(|_| RlimitExecveatCgroupError::InvalidChildPid)?;
    let mut guard = PostCloneReapGuard::new(pid);
    drop(stdin_read); drop(stdout_write); drop(stderr_write); drop(exec_status_write); drop(cgroup);
    observe_exec_success(exec_status_read)?;
    if !lease.contains_pid(pid)? { return Err(RlimitExecveatCgroupError::BirthMembershipNotObserved(pid)); }
    lease.verify_configuration()?;
    let proc_exe_sha256 = sha256_path(format!("/proc/{pid}/exe")).map_err(RlimitExecveatCgroupError::ProcExe)?;
    if proc_exe_sha256 != image.image_sha256() { return Err(RlimitExecveatCgroupError::ExecutableDigestMismatch); }
    verify_live_image(image)?;

    let mut base = SealedExecveatCgroupEvidence {
        profile: SEALED_EXECVEAT_CGROUP_PROFILE_V1.into(), child_pid: pid, clone_flags: CLONE_INTO_CGROUP_FLAG,
        target: lease.evidence()?, sealed_image_profile: SEALED_WORKER_IMAGE_PROFILE_V1.into(),
        source_sha256: hex_digest(image.source_sha256()), image_sha256: hex_digest(image.image_sha256()),
        proc_exe_sha256: hex_digest(proc_exe_sha256), image_size_bytes: image.image_size_bytes(),
        image_seal_mask: image.seal_mask(), birth_membership_observed: true, exec_status_eof_observed: true,
        no_new_privs_before_exec: true, executable_fd_closed_on_exec: true, evidence_sha256: String::new(),
    };
    base.evidence_sha256 = hex_digest(base_evidence_sha256_v1(&base)?);
    base.verify()?;

    let mut evidence = SealedExecveatCgroupRlimitEvidence {
        profile: SEALED_EXECVEAT_CGROUP_RLIMIT_PROFILE_V1.into(), base,
        supervisor_profile: SUPERVISOR_PROFILE_V1.into(), limits: PreExecSupervisorEnvelope::from_limits(limits),
        legacy_preexec_rlimits_applied: true, setsid_before_exec: true, no_new_privs_before_exec: true,
        umask_0077_before_exec: true, parent_wall_time_enforced_by_this_launcher: false,
        parent_output_limits_enforced_by_this_launcher: false, evidence_sha256: String::new(),
    };
    evidence.evidence_sha256 = hex_digest(evidence_sha256_v1(&evidence)?);
    evidence.verify()?;
    guard.disarm();
    Ok(RlimitSealedExecveatCgroupChild {
        pid, stdin: Some(File::from(stdin_write)), stdout: File::from(stdout_read),
        stderr: File::from(stderr_read), evidence, reaped: false,
    })
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn child_execveat_with_rlimits(
    executable_fd: RawFd, cgroup_fd: RawFd, stdin_read: RawFd, stdin_write: RawFd,
    stdout_read: RawFd, stdout_write: RawFd, stderr_read: RawFd, stderr_write: RawFd,
    exec_status_read: RawFd, exec_status_write: RawFd, limits: SupervisorLimits,
) -> ! {
    unsafe { raw_close(stdin_write); raw_close(stdout_read); raw_close(stderr_read); raw_close(exec_status_read); raw_close(cgroup_fd); }
    if unsafe { raw_dup2(stdin_read, libc::STDIN_FILENO) } < 0
        || unsafe { raw_dup2(stdout_write, libc::STDOUT_FILENO) } < 0
        || unsafe { raw_dup2(stderr_write, libc::STDERR_FILENO) } < 0
    { unsafe { child_fail(exec_status_write, STAGE_STDIO, 121) } }
    for fd in [stdin_read, stdout_write, stderr_write] { if fd > libc::STDERR_FILENO { unsafe { raw_close(fd) }; } }
    unsafe {
        set_limit_or_fail(exec_status_write, STAGE_RLIMIT_AS, libc::RLIMIT_AS as i32, limits.address_space_bytes, 122);
        set_limit_or_fail(exec_status_write, STAGE_RLIMIT_CPU, libc::RLIMIT_CPU as i32, limits.cpu_seconds, 123);
        set_limit_or_fail(exec_status_write, STAGE_RLIMIT_FSIZE, libc::RLIMIT_FSIZE as i32, limits.file_size_bytes, 124);
        set_limit_or_fail(exec_status_write, STAGE_RLIMIT_NOFILE, libc::RLIMIT_NOFILE as i32, limits.open_files, 125);
        set_limit_or_fail(exec_status_write, STAGE_RLIMIT_NPROC, libc::RLIMIT_NPROC as i32, limits.process_count, 126);
        set_limit_or_fail(exec_status_write, STAGE_RLIMIT_CORE, libc::RLIMIT_CORE as i32, 0, 127);
    }
    if unsafe { libc::syscall(libc::SYS_setsid) } < 0 { unsafe { child_fail(exec_status_write, STAGE_SETSID, 128) } }
    if unsafe { libc::syscall(libc::SYS_prctl, libc::PR_SET_NO_NEW_PRIVS, 1usize, 0usize, 0usize, 0usize) } != 0 {
        unsafe { child_fail(exec_status_write, STAGE_NO_NEW_PRIVS, 129) }
    }
    unsafe { libc::syscall(libc::SYS_umask, 0o077usize); }
    if unsafe { libc::syscall(libc::SYS_fcntl, executable_fd, libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
        unsafe { child_fail(exec_status_write, STAGE_EXEC_FD_CLOEXEC, 130) }
    }
    let argv = [ARGV0.as_ptr().cast::<libc::c_char>(), std::ptr::null()];
    let envp: [*const libc::c_char; 1] = [std::ptr::null()];
    unsafe { libc::syscall(libc::SYS_execveat, executable_fd, EMPTY_PATH.as_ptr().cast::<libc::c_char>(), argv.as_ptr(), envp.as_ptr(), libc::AT_EMPTY_PATH); }
    unsafe { child_fail(exec_status_write, STAGE_EXECVEAT, 131) }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn set_limit_or_fail(status_fd: RawFd, stage: u32, resource: i32, value: u64, exit_code: i32) {
    let limit = KernelRlimit64 { current: value, maximum: value };
    let result = unsafe { libc::syscall(libc::SYS_prlimit64, 0i32, resource, &limit as *const KernelRlimit64, std::ptr::null_mut::<KernelRlimit64>()) };
    if result != 0 { unsafe { child_fail(status_fd, stage, exit_code) } }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn raw_close(fd: RawFd) { unsafe { libc::syscall(libc::SYS_close, fd); } }
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn raw_dup2(old: RawFd, new: RawFd) -> libc::c_long { unsafe { libc::syscall(libc::SYS_dup2, old, new) } }
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn child_fail(status_fd: RawFd, stage: u32, exit_code: i32) -> ! {
    let errno = unsafe { *libc::__errno_location() };
    let mut bytes = [0u8; 8];
    bytes[..4].copy_from_slice(&stage.to_le_bytes());
    bytes[4..].copy_from_slice(&errno.to_le_bytes());
    unsafe {
        libc::syscall(libc::SYS_write, status_fd, bytes.as_ptr().cast::<libc::c_void>(), bytes.len());
        libc::syscall(libc::SYS_exit, exit_code);
        std::hint::unreachable_unchecked();
    }
}

fn observe_exec_success(status: OwnedFd) -> Result<(), RlimitExecveatCgroupError> {
    let mut file = File::from(status);
    let mut bytes = [0u8; 8];
    let mut offset = 0usize;
    loop {
        match file.read(&mut bytes[offset..]) {
            Ok(0) if offset == 0 => return Ok(()),
            Ok(0) => return Err(RlimitExecveatCgroupError::TruncatedChildFailure),
            Ok(read) => {
                offset += read;
                if offset == bytes.len() {
                    let stage = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                    let errno = i32::from_le_bytes(bytes[4..].try_into().unwrap());
                    return Err(RlimitExecveatCgroupError::ChildSetupFailed { stage, errno });
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(RlimitExecveatCgroupError::ExecStatus(error)),
        }
    }
}

fn verify_live_image(image: &SealedWorkerImage) -> Result<(), RlimitExecveatCgroupError> {
    if image.source_sha256() != image.image_sha256() || image.image_size_bytes() == 0 { return Err(RlimitExecveatCgroupError::ExecutableDigestMismatch); }
    let digest = sha256_path(image.exec_path()).map_err(RlimitExecveatCgroupError::ImageRead)?;
    if digest != image.image_sha256() { return Err(RlimitExecveatCgroupError::ExecutableDigestMismatch); }
    let fd = image_fd(image)?;
    let seals = unsafe { libc::fcntl(fd, libc::F_GET_SEALS) };
    if seals < 0 { return Err(RlimitExecveatCgroupError::SealOperation(io::Error::last_os_error())); }
    if seals as u32 != image.seal_mask() || seals & REQUIRED_SEALS != REQUIRED_SEALS {
        return Err(RlimitExecveatCgroupError::SealMismatch { expected: image.seal_mask(), actual: seals as u32 });
    }
    Ok(())
}

fn image_fd(image: &SealedWorkerImage) -> Result<RawFd, RlimitExecveatCgroupError> {
    image.exec_path().file_name().and_then(|value| value.to_str()).and_then(|value| value.parse::<RawFd>().ok())
        .filter(|fd| *fd >= 0).ok_or(RlimitExecveatCgroupError::InvalidExecutableFd)
}

fn pipe_cloexec() -> Result<(OwnedFd, OwnedFd), RlimitExecveatCgroupError> {
    let mut fds = [-1; 2];
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 { return Err(RlimitExecveatCgroupError::Pipe(io::Error::last_os_error())); }
    let read = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let write = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    Ok((read, write))
}

fn sha256_path(path: impl AsRef<std::path::Path>) -> Result<[u8; 32], io::Error> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop { let read = file.read(&mut buffer)?; if read == 0 { break; } hasher.update(&buffer[..read]); }
    Ok(hasher.finalize().into())
}

fn wait_pid(pid: u32) -> Result<ExitStatus, RlimitExecveatCgroupError> {
    let pid = i32::try_from(pid).map_err(|_| RlimitExecveatCgroupError::InvalidChildPid)?;
    let mut raw_status = 0;
    loop {
        let result = unsafe { libc::waitpid(pid, &mut raw_status, 0) };
        if result == pid { return Ok(ExitStatus::from_raw(raw_status)); }
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted { continue; }
            return Err(RlimitExecveatCgroupError::Wait(error));
        }
    }
}
fn terminate_pid(pid: u32) { if let Ok(pid) = i32::try_from(pid) { unsafe { libc::kill(pid, libc::SIGKILL); } } }
fn classify_clone3_error(error: io::Error) -> RlimitExecveatCgroupError {
    match error.raw_os_error() {
        Some(libc::ENOSYS) => RlimitExecveatCgroupError::Clone3Unavailable,
        Some(libc::EACCES) => RlimitExecveatCgroupError::PlacementDenied,
        Some(libc::EBUSY) => RlimitExecveatCgroupError::TargetNotLeafDomain,
        Some(libc::EOPNOTSUPP) => RlimitExecveatCgroupError::TargetDomainInvalid,
        _ => RlimitExecveatCgroupError::Clone3(error),
    }
}

fn base_evidence_sha256_v1(evidence: &SealedExecveatCgroupEvidence) -> Result<[u8; 32], RlimitExecveatCgroupError> {
    evidence.target.verify()?;
    let target = parse_hex_32(&evidence.target.evidence_sha256)?;
    let source = parse_hex_32(&evidence.source_sha256)?;
    let image = parse_hex_32(&evidence.image_sha256)?;
    let proc_exe = parse_hex_32(&evidence.proc_exe_sha256)?;
    let mut hasher = Sha256::new();
    hasher.update(BASE_EVIDENCE_DOMAIN_V1); hasher.update(evidence.child_pid.to_le_bytes()); hasher.update(evidence.clone_flags.to_le_bytes());
    hasher.update(target); hasher.update(source); hasher.update(image); hasher.update(proc_exe);
    hasher.update(evidence.image_size_bytes.to_le_bytes()); hasher.update(evidence.image_seal_mask.to_le_bytes());
    hasher.update([u8::from(evidence.birth_membership_observed)]); hasher.update([u8::from(evidence.exec_status_eof_observed)]);
    hasher.update([u8::from(evidence.no_new_privs_before_exec)]); hasher.update([u8::from(evidence.executable_fd_closed_on_exec)]);
    Ok(hasher.finalize().into())
}

fn evidence_sha256_v1(evidence: &SealedExecveatCgroupRlimitEvidence) -> Result<[u8; 32], RlimitExecveatCgroupError> {
    evidence.base.verify()?;
    let base = parse_hex_32(&evidence.base.evidence_sha256)?;
    let mut hasher = Sha256::new();
    hasher.update(EVIDENCE_DOMAIN_V1); hasher.update(base); hash_string(&mut hasher, &evidence.supervisor_profile)?;
    for value in [evidence.limits.address_space_bytes, evidence.limits.cpu_seconds, evidence.limits.file_size_bytes,
        evidence.limits.open_files, evidence.limits.process_count, evidence.limits.core_bytes,
        evidence.limits.expected_parent_wall_time_ms, evidence.limits.expected_parent_max_stdout_bytes,
        evidence.limits.expected_parent_max_stderr_bytes] { hasher.update(value.to_le_bytes()); }
    hasher.update([u8::from(evidence.legacy_preexec_rlimits_applied)]); hasher.update([u8::from(evidence.setsid_before_exec)]);
    hasher.update([u8::from(evidence.no_new_privs_before_exec)]); hasher.update([u8::from(evidence.umask_0077_before_exec)]);
    hasher.update([u8::from(evidence.parent_wall_time_enforced_by_this_launcher)]); hasher.update([u8::from(evidence.parent_output_limits_enforced_by_this_launcher)]);
    Ok(hasher.finalize().into())
}
fn hash_string(hasher: &mut Sha256, value: &str) -> Result<(), RlimitExecveatCgroupError> {
    let len = u64::try_from(value.len()).map_err(|_| RlimitExecveatCgroupError::LengthOverflow)?;
    hasher.update(len.to_le_bytes()); hasher.update(value.as_bytes()); Ok(())
}
fn parse_hex_32(value: &str) -> Result<[u8; 32], RlimitExecveatCgroupError> {
    if value.len() != 64 { return Err(RlimitExecveatCgroupError::InvalidDigest); }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(value.as_bytes().chunks_exact(2)) { *slot = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?; }
    Ok(out)
}
fn hex_nibble(value: u8) -> Result<u8, RlimitExecveatCgroupError> {
    match value { b'0'..=b'9' => Ok(value - b'0'), b'a'..=b'f' => Ok(value - b'a' + 10), _ => Err(RlimitExecveatCgroupError::InvalidDigest) }
}
fn hex_digest(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes { out.push(TABLE[(byte >> 4) as usize] as char); out.push(TABLE[(byte & 0x0f) as usize] as char); }
    out
}

#[derive(Debug, Error)]
pub enum RlimitExecveatCgroupError {
    #[error("rlimit clone3 execveat launcher is supported only on Linux x86_64")] UnsupportedPlatform,
    #[error("target cgroup must be empty before direct child creation")] TargetAlreadyPopulated,
    #[error("failed to open target cgroup: {0}")] OpenCgroup(#[source] io::Error),
    #[error("failed to inspect target cgroup: {0}")] CgroupMetadata(#[source] io::Error),
    #[error("target cgroup identity changed")] CgroupIdentityMismatch,
    #[error("invalid cgroup descriptor")] InvalidCgroupFd,
    #[error("invalid sealed executable descriptor")] InvalidExecutableFd,
    #[error("failed creating launcher pipe: {0}")] Pipe(#[source] io::Error),
    #[error("clone3 is unavailable on this kernel")] Clone3Unavailable,
    #[error("kernel denied CLONE_INTO_CGROUP placement")] PlacementDenied,
    #[error("target cgroup is not a valid leaf domain")] TargetNotLeafDomain,
    #[error("target cgroup domain is invalid")] TargetDomainInvalid,
    #[error("clone3 failed: {0}")] Clone3(#[source] io::Error),
    #[error("clone3 returned an invalid child PID")] InvalidChildPid,
    #[error("child pre-exec setup failed at stage {stage} with errno {errno}")] ChildSetupFailed { stage: u32, errno: i32 },
    #[error("child pre-exec status was truncated")] TruncatedChildFailure,
    #[error("failed reading child pre-exec status: {0}")] ExecStatus(#[source] io::Error),
    #[error("child PID {0} was not observed in the target cgroup")] BirthMembershipNotObserved(u32),
    #[error("failed to inspect /proc child executable: {0}")] ProcExe(#[source] io::Error),
    #[error("failed to read sealed image: {0}")] ImageRead(#[source] io::Error),
    #[error("sealed/executed executable digest does not match")] ExecutableDigestMismatch,
    #[error("failed to inspect sealed image seals: {0}")] SealOperation(#[source] io::Error),
    #[error("sealed image seal mask differs: expected={expected:#x}, actual={actual:#x}")] SealMismatch { expected: u32, actual: u32 },
    #[error("failed waiting for launched child: {0}")] Wait(#[source] io::Error),
    #[error("serialized rlimit execveat evidence is structurally invalid")] InvalidEvidence,
    #[error("serialized rlimit execveat evidence digest does not match")] EvidenceDigestMismatch,
    #[error("invalid canonical lowercase SHA-256 digest")] InvalidDigest,
    #[error("string length cannot be represented in commitment")] LengthOverflow,
    #[error(transparent)] BaseEvidence(#[from] SealedExecveatCgroupError),
    #[error(transparent)] Supervisor(#[from] SupervisorError),
    #[error(transparent)] Cgroup(#[from] CgroupV2Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clone_args_layout_and_profile_are_frozen() {
        assert_eq!(size_of::<CloneArgs>(), 88);
        assert_eq!(CLONE_INTO_CGROUP_FLAG, 0x2000_0000_0);
        assert_eq!(SEALED_EXECVEAT_CGROUP_RLIMIT_PROFILE_V1, "symthaea.simulation.clone-into-cgroup.sealed-execveat-rlimit-v1");
    }
    #[test]
    fn envelope_preserves_legacy_preexec_and_parent_owned_fields() {
        let limits = SupervisorLimits::default();
        let envelope = PreExecSupervisorEnvelope::from_limits(limits);
        assert_eq!(envelope.address_space_bytes, limits.address_space_bytes);
        assert_eq!(envelope.cpu_seconds, limits.cpu_seconds);
        assert_eq!(envelope.file_size_bytes, limits.file_size_bytes);
        assert_eq!(envelope.open_files, limits.open_files);
        assert_eq!(envelope.process_count, limits.process_count);
        assert_eq!(envelope.core_bytes, 0);
        assert_eq!(envelope.expected_parent_wall_time_ms, limits.wall_time_ms);
        assert_eq!(envelope.expected_parent_max_stdout_bytes, limits.max_stdout_bytes);
        assert_eq!(envelope.expected_parent_max_stderr_bytes, limits.max_stderr_bytes);
    }
}
