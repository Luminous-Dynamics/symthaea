// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent `clone3(CLONE_INTO_CGROUP)` + `execveat(AT_EMPTY_PATH)` launcher.
//!
//! This intentionally re-expresses the public Linux ABI rather than reaching
//! into the private implementation of `symthaea-sim-clone-cgroup`. The parent
//! proves that one child was born in an exact cgroup-v2 lease, that exec
//! succeeded through a CLOEXEC status-pipe handshake, and that `/proc/<pid>/exe`
//! hashes to the exact immutable sealed-image digest while the child remains
//! alive. The post-clone/pre-exec child path uses raw libc syscalls only.

#![deny(unsafe_op_in_unsafe_fn)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read, Write};
use std::mem::size_of;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
#[cfg(target_os = "linux")]
use std::os::unix::fs::MetadataExt;
use std::os::unix::process::ExitStatusExt;
use std::process::ExitStatus;
use symthaea_sim_worker_cgroup::{CgroupV2Error, CgroupV2Evidence, CgroupV2Lease};
use symthaea_sim_worker_image::{SealedWorkerImage, SEALED_WORKER_IMAGE_PROFILE_V1};
use thiserror::Error;

pub const SEALED_EXECVEAT_CGROUP_PROFILE_V1: &str =
    "symthaea.simulation.clone-into-cgroup.sealed-execveat-v1";
const EVIDENCE_DOMAIN_V1: &[u8] =
    b"symthaea.simulation.clone-into-cgroup.sealed-execveat-v1\0";
const CLONE_INTO_CGROUP_FLAG: u64 = 0x2000_0000_0;
const REQUIRED_SEALS: libc::c_int =
    libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
const EMPTY_PATH: &[u8] = b"\0";
const ARGV0: &[u8] = b"symthaea-sealed-cgroup-exec\0";

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedExecveatCgroupEvidence {
    pub profile: String,
    pub child_pid: u32,
    pub clone_flags: u64,
    pub target: CgroupV2Evidence,
    pub sealed_image_profile: String,
    pub source_sha256: String,
    pub image_sha256: String,
    pub proc_exe_sha256: String,
    pub image_size_bytes: u64,
    pub image_seal_mask: u32,
    pub birth_membership_observed: bool,
    pub exec_status_eof_observed: bool,
    pub no_new_privs_before_exec: bool,
    pub executable_fd_closed_on_exec: bool,
    pub evidence_sha256: String,
}

impl SealedExecveatCgroupEvidence {
    pub fn verify(&self) -> Result<(), SealedExecveatCgroupError> {
        if self.profile != SEALED_EXECVEAT_CGROUP_PROFILE_V1
            || self.clone_flags != CLONE_INTO_CGROUP_FLAG
            || self.child_pid == 0
            || self.sealed_image_profile != SEALED_WORKER_IMAGE_PROFILE_V1
            || !self.birth_membership_observed
            || !self.exec_status_eof_observed
            || !self.no_new_privs_before_exec
            || !self.executable_fd_closed_on_exec
            || self.image_size_bytes == 0
            || self.image_seal_mask & REQUIRED_SEALS as u32 != REQUIRED_SEALS as u32
        {
            return Err(SealedExecveatCgroupError::InvalidEvidence);
        }
        self.target.verify()?;
        let source = parse_hex_32(&self.source_sha256)?;
        let image = parse_hex_32(&self.image_sha256)?;
        let proc_exe = parse_hex_32(&self.proc_exe_sha256)?;
        if source != image || image != proc_exe {
            return Err(SealedExecveatCgroupError::ExecutableDigestMismatch);
        }
        let expected = hex_digest(evidence_sha256_v1(self)?);
        if self.evidence_sha256 != expected {
            return Err(SealedExecveatCgroupError::EvidenceDigestMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct SealedExecveatCgroupChild {
    pid: u32,
    stdin: Option<File>,
    stdout: File,
    stderr: File,
    evidence: SealedExecveatCgroupEvidence,
    reaped: bool,
}

impl SealedExecveatCgroupChild {
    pub const fn pid(&self) -> u32 {
        self.pid
    }

    pub fn evidence(&self) -> &SealedExecveatCgroupEvidence {
        &self.evidence
    }

    pub fn stdin_mut(&mut self) -> Option<&mut File> {
        self.stdin.as_mut()
    }

    pub fn stdout_mut(&mut self) -> &mut File {
        &mut self.stdout
    }

    pub fn stderr_mut(&mut self) -> &mut File {
        &mut self.stderr
    }

    pub fn close_stdin(&mut self) {
        self.stdin.take();
    }

    pub fn wait(mut self) -> Result<(ExitStatus, SealedExecveatCgroupEvidence), SealedExecveatCgroupError> {
        self.stdin.take();
        let status = wait_pid(self.pid)?;
        self.reaped = true;
        Ok((status, self.evidence.clone()))
    }

    pub fn terminate_and_wait(
        mut self,
    ) -> Result<SealedExecveatCgroupEvidence, SealedExecveatCgroupError> {
        self.stdin.take();
        terminate_pid(self.pid);
        let _ = wait_pid(self.pid)?;
        self.reaped = true;
        Ok(self.evidence.clone())
    }
}

impl Drop for SealedExecveatCgroupChild {
    fn drop(&mut self) {
        if !self.reaped {
            self.stdin.take();
            terminate_pid(self.pid);
            let _ = wait_pid(self.pid);
            self.reaped = true;
        }
    }
}

struct PostCloneReapGuard {
    pid: u32,
    armed: bool,
}

impl PostCloneReapGuard {
    const fn new(pid: u32) -> Self {
        Self { pid, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PostCloneReapGuard {
    fn drop(&mut self) {
        if self.armed {
            terminate_pid(self.pid);
            let _ = wait_pid(self.pid);
        }
    }
}

/// Create a process directly in `lease`, then atomically exec the exact sealed
/// image through `execveat(AT_EMPTY_PATH)`. The executed program receives an
/// empty environment and ordinary piped stdin/stdout/stderr.
pub fn spawn_sealed_image_in_cgroup(
    image: &SealedWorkerImage,
    lease: &CgroupV2Lease,
) -> Result<SealedExecveatCgroupChild, SealedExecveatCgroupError> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (image, lease);
        return Err(SealedExecveatCgroupError::UnsupportedPlatform);
    }

    #[cfg(target_os = "linux")]
    spawn_linux(image, lease)
}

#[cfg(target_os = "linux")]
fn spawn_linux(
    image: &SealedWorkerImage,
    lease: &CgroupV2Lease,
) -> Result<SealedExecveatCgroupChild, SealedExecveatCgroupError> {
    lease.verify_configuration()?;
    if lease.populated()? {
        return Err(SealedExecveatCgroupError::TargetAlreadyPopulated);
    }
    let target = lease.evidence()?;
    target.verify()?;
    verify_live_image(image)?;

    let cgroup = File::open(lease.leaf()).map_err(SealedExecveatCgroupError::OpenCgroup)?;
    let cgroup_metadata = cgroup
        .metadata()
        .map_err(SealedExecveatCgroupError::CgroupMetadata)?;
    if cgroup_metadata.dev() != target.leaf_device || cgroup_metadata.ino() != target.leaf_inode {
        return Err(SealedExecveatCgroupError::CgroupIdentityMismatch);
    }

    let executable_fd = image_fd(image)?;
    if executable_fd < 3 {
        return Err(SealedExecveatCgroupError::InvalidExecutableFd);
    }

    let (stdin_read, stdin_write) = pipe_cloexec()?;
    let (stdout_read, stdout_write) = pipe_cloexec()?;
    let (stderr_read, stderr_write) = pipe_cloexec()?;
    let (exec_status_read, exec_status_write) = pipe_cloexec()?;

    let args = CloneArgs {
        flags: CLONE_INTO_CGROUP_FLAG,
        exit_signal: libc::SIGCHLD as u64,
        cgroup: u64::try_from(cgroup.as_raw_fd())
            .map_err(|_| SealedExecveatCgroupError::InvalidCgroupFd)?,
        ..CloneArgs::default()
    };

    // SAFETY: public Linux clone3 ABI, fork-like semantics, exact verified
    // cgroup-v2 fd, and a syscall-only child path ending in execveat/_exit.
    let result = unsafe {
        libc::syscall(
            libc::SYS_clone3,
            &args as *const CloneArgs,
            size_of::<CloneArgs>(),
        )
    };
    if result < 0 {
        return Err(classify_clone3_error(io::Error::last_os_error()));
    }
    if result == 0 {
        // SAFETY: post-clone child path is raw-syscall only and never unwinds.
        unsafe {
            child_execveat(
                executable_fd,
                cgroup.as_raw_fd(),
                stdin_read.as_raw_fd(),
                stdin_write.as_raw_fd(),
                stdout_read.as_raw_fd(),
                stdout_write.as_raw_fd(),
                stderr_read.as_raw_fd(),
                stderr_write.as_raw_fd(),
                exec_status_read.as_raw_fd(),
                exec_status_write.as_raw_fd(),
            )
        }
    }

    let pid = u32::try_from(result).map_err(|_| SealedExecveatCgroupError::InvalidChildPid)?;
    let mut guard = PostCloneReapGuard::new(pid);

    drop(stdin_read);
    drop(stdout_write);
    drop(stderr_write);
    drop(exec_status_write);
    drop(cgroup);

    observe_exec_success(exec_status_read)?;
    if !lease.contains_pid(pid)? {
        return Err(SealedExecveatCgroupError::BirthMembershipNotObserved(pid));
    }
    lease.verify_configuration()?;

    let proc_exe_sha256 = sha256_path(&format!("/proc/{pid}/exe"))
        .map_err(SealedExecveatCgroupError::ProcExe)?;
    if proc_exe_sha256 != image.image_sha256() {
        return Err(SealedExecveatCgroupError::ExecutableDigestMismatch);
    }
    verify_live_image(image)?;

    let mut evidence = SealedExecveatCgroupEvidence {
        profile: SEALED_EXECVEAT_CGROUP_PROFILE_V1.into(),
        child_pid: pid,
        clone_flags: CLONE_INTO_CGROUP_FLAG,
        target: lease.evidence()?,
        sealed_image_profile: SEALED_WORKER_IMAGE_PROFILE_V1.into(),
        source_sha256: hex_digest(image.source_sha256()),
        image_sha256: hex_digest(image.image_sha256()),
        proc_exe_sha256: hex_digest(proc_exe_sha256),
        image_size_bytes: image.image_size_bytes(),
        image_seal_mask: image.seal_mask(),
        birth_membership_observed: true,
        exec_status_eof_observed: true,
        no_new_privs_before_exec: true,
        executable_fd_closed_on_exec: true,
        evidence_sha256: String::new(),
    };
    evidence.evidence_sha256 = hex_digest(evidence_sha256_v1(&evidence)?);
    evidence.verify()?;

    guard.disarm();
    Ok(SealedExecveatCgroupChild {
        pid,
        stdin: Some(File::from(stdin_write)),
        stdout: File::from(stdout_read),
        stderr: File::from(stderr_read),
        evidence,
        reaped: false,
    })
}

#[cfg(target_os = "linux")]
unsafe fn child_execveat(
    executable_fd: RawFd,
    cgroup_fd: RawFd,
    stdin_read: RawFd,
    stdin_write: RawFd,
    stdout_read: RawFd,
    stdout_write: RawFd,
    stderr_read: RawFd,
    stderr_write: RawFd,
    exec_status_read: RawFd,
    exec_status_write: RawFd,
) -> ! {
    // Close parent-side pipe ends first.
    unsafe {
        libc::close(stdin_write);
        libc::close(stdout_read);
        libc::close(stderr_read);
        libc::close(exec_status_read);
        libc::close(cgroup_fd);
    }

    if unsafe { libc::dup2(stdin_read, libc::STDIN_FILENO) } < 0
        || unsafe { libc::dup2(stdout_write, libc::STDOUT_FILENO) } < 0
        || unsafe { libc::dup2(stderr_write, libc::STDERR_FILENO) } < 0
    {
        unsafe { child_exec_fail(exec_status_write, 124) }
    }

    for fd in [stdin_read, stdout_write, stderr_write] {
        if fd > libc::STDERR_FILENO {
            unsafe { libc::close(fd) };
        }
    }

    if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
        unsafe { child_exec_fail(exec_status_write, 125) }
    }
    unsafe { libc::umask(0o077) };

    // The sealed memfd is needed by execveat itself but should not remain open
    // in the executed ELF. FD_CLOEXEC is per descriptor table entry and this
    // fork-like clone does not share the parent's descriptor table.
    if unsafe { libc::fcntl(executable_fd, libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
        unsafe { child_exec_fail(exec_status_write, 126) }
    }

    let argv = [ARGV0.as_ptr().cast::<libc::c_char>(), std::ptr::null()];
    let envp: [*const libc::c_char; 1] = [std::ptr::null()];
    // SAFETY: executable_fd names the exact inherited sealed ELF; EMPTY_PATH,
    // argv and envp are valid NUL-terminated/static pointers. On success this
    // does not return and CLOEXEC closes both the status fd and executable fd.
    unsafe {
        libc::syscall(
            libc::SYS_execveat,
            executable_fd,
            EMPTY_PATH.as_ptr().cast::<libc::c_char>(),
            argv.as_ptr(),
            envp.as_ptr(),
            libc::AT_EMPTY_PATH,
        );
    }
    unsafe { child_exec_fail(exec_status_write, 127) }
}

#[cfg(target_os = "linux")]
unsafe fn child_exec_fail(exec_status_write: RawFd, exit_code: i32) -> ! {
    // SAFETY: errno is thread-local libc state; the four bytes are a stack-local
    // integer and write is async-signal-safe.
    let errno = unsafe { *libc::__errno_location() };
    let bytes = errno.to_le_bytes();
    unsafe {
        libc::write(exec_status_write, bytes.as_ptr().cast(), bytes.len());
        libc::_exit(exit_code);
    }
}

fn observe_exec_success(exec_status_read: OwnedFd) -> Result<(), SealedExecveatCgroupError> {
    let mut file = File::from(exec_status_read);
    let mut bytes = [0u8; 4];
    let mut offset = 0usize;
    loop {
        match file.read(&mut bytes[offset..]) {
            Ok(0) if offset == 0 => return Ok(()),
            Ok(0) => return Err(SealedExecveatCgroupError::TruncatedExecFailure),
            Ok(read) => {
                offset += read;
                if offset == bytes.len() {
                    let errno = i32::from_le_bytes(bytes);
                    return Err(SealedExecveatCgroupError::ExecveatFailed(errno));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(SealedExecveatCgroupError::ExecStatus(error)),
        }
    }
}

fn verify_live_image(image: &SealedWorkerImage) -> Result<(), SealedExecveatCgroupError> {
    if image.source_sha256() != image.image_sha256() || image.image_size_bytes() == 0 {
        return Err(SealedExecveatCgroupError::ExecutableDigestMismatch);
    }
    let digest = sha256_path(image.exec_path()).map_err(SealedExecveatCgroupError::ImageRead)?;
    if digest != image.image_sha256() {
        return Err(SealedExecveatCgroupError::ExecutableDigestMismatch);
    }
    let fd = image_fd(image)?;
    // SAFETY: fd is the live memfd descriptor encoded by SealedWorkerImage.
    let seals = unsafe { libc::fcntl(fd, libc::F_GET_SEALS) };
    if seals < 0 {
        return Err(SealedExecveatCgroupError::SealOperation(
            io::Error::last_os_error(),
        ));
    }
    if seals as u32 != image.seal_mask() || seals & REQUIRED_SEALS != REQUIRED_SEALS {
        return Err(SealedExecveatCgroupError::SealMismatch {
            expected: image.seal_mask(),
            actual: seals as u32,
        });
    }
    Ok(())
}

fn image_fd(image: &SealedWorkerImage) -> Result<RawFd, SealedExecveatCgroupError> {
    image
        .exec_path()
        .file_name()
        .and_then(|value| value.to_str())
        .and_then(|value| value.parse::<RawFd>().ok())
        .filter(|fd| *fd >= 0)
        .ok_or(SealedExecveatCgroupError::InvalidExecutableFd)
}

fn pipe_cloexec() -> Result<(OwnedFd, OwnedFd), SealedExecveatCgroupError> {
    let mut fds = [-1; 2];
    // SAFETY: storage is valid for exactly two descriptors.
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(SealedExecveatCgroupError::Pipe(io::Error::last_os_error()));
    }
    // SAFETY: successful pipe2 returned two newly owned descriptors.
    let read = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let write = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    Ok((read, write))
}

fn sha256_path(path: impl AsRef<std::path::Path>) -> Result<[u8; 32], io::Error> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

fn wait_pid(pid: u32) -> Result<ExitStatus, SealedExecveatCgroupError> {
    let pid = i32::try_from(pid).map_err(|_| SealedExecveatCgroupError::InvalidChildPid)?;
    let mut raw_status = 0;
    loop {
        // SAFETY: status storage is valid and pid belongs to this parent.
        let result = unsafe { libc::waitpid(pid, &mut raw_status, 0) };
        if result == pid {
            return Ok(ExitStatus::from_raw(raw_status));
        }
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(SealedExecveatCgroupError::Wait(error));
        }
    }
}

fn terminate_pid(pid: u32) {
    if let Ok(pid) = i32::try_from(pid) {
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
    }
}

fn classify_clone3_error(error: io::Error) -> SealedExecveatCgroupError {
    match error.raw_os_error() {
        Some(libc::ENOSYS) => SealedExecveatCgroupError::Clone3Unavailable,
        Some(libc::EACCES) => SealedExecveatCgroupError::PlacementDenied,
        Some(libc::EBUSY) => SealedExecveatCgroupError::TargetNotLeafDomain,
        Some(libc::EOPNOTSUPP) => SealedExecveatCgroupError::TargetDomainInvalid,
        _ => SealedExecveatCgroupError::Clone3(error),
    }
}

fn evidence_sha256_v1(
    evidence: &SealedExecveatCgroupEvidence,
) -> Result<[u8; 32], SealedExecveatCgroupError> {
    evidence.target.verify()?;
    let target = parse_hex_32(&evidence.target.evidence_sha256)?;
    let source = parse_hex_32(&evidence.source_sha256)?;
    let image = parse_hex_32(&evidence.image_sha256)?;
    let proc_exe = parse_hex_32(&evidence.proc_exe_sha256)?;
    let mut hasher = Sha256::new();
    hasher.update(EVIDENCE_DOMAIN_V1);
    hasher.update(evidence.child_pid.to_le_bytes());
    hasher.update(evidence.clone_flags.to_le_bytes());
    hasher.update(target);
    hasher.update(source);
    hasher.update(image);
    hasher.update(proc_exe);
    hasher.update(evidence.image_size_bytes.to_le_bytes());
    hasher.update(evidence.image_seal_mask.to_le_bytes());
    hasher.update([u8::from(evidence.birth_membership_observed)]);
    hasher.update([u8::from(evidence.exec_status_eof_observed)]);
    hasher.update([u8::from(evidence.no_new_privs_before_exec)]);
    hasher.update([u8::from(evidence.executable_fd_closed_on_exec)]);
    Ok(hasher.finalize().into())
}

fn parse_hex_32(value: &str) -> Result<[u8; 32], SealedExecveatCgroupError> {
    if value.len() != 64 {
        return Err(SealedExecveatCgroupError::InvalidDigest);
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        *slot = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(out)
}

fn hex_nibble(value: u8) -> Result<u8, SealedExecveatCgroupError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(SealedExecveatCgroupError::InvalidDigest),
    }
}

fn hex_digest(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    out
}

#[derive(Debug, Error)]
pub enum SealedExecveatCgroupError {
    #[error("sealed clone3 execveat launcher is supported only on Linux")]
    UnsupportedPlatform,
    #[error("target cgroup must be empty before direct child creation")]
    TargetAlreadyPopulated,
    #[error("failed to open target cgroup: {0}")]
    OpenCgroup(#[source] io::Error),
    #[error("failed to inspect target cgroup: {0}")]
    CgroupMetadata(#[source] io::Error),
    #[error("target cgroup identity changed")]
    CgroupIdentityMismatch,
    #[error("invalid cgroup descriptor")]
    InvalidCgroupFd,
    #[error("invalid sealed executable descriptor")]
    InvalidExecutableFd,
    #[error("failed creating launcher pipe: {0}")]
    Pipe(#[source] io::Error),
    #[error("clone3 is unavailable on this kernel")]
    Clone3Unavailable,
    #[error("kernel denied CLONE_INTO_CGROUP placement")]
    PlacementDenied,
    #[error("target cgroup is not a valid leaf domain")]
    TargetNotLeafDomain,
    #[error("target cgroup domain is invalid")]
    TargetDomainInvalid,
    #[error("clone3 failed: {0}")]
    Clone3(#[source] io::Error),
    #[error("clone3 returned an invalid child PID")]
    InvalidChildPid,
    #[error("failed reading exec-status pipe: {0}")]
    ExecStatus(#[source] io::Error),
    #[error("exec failure status was truncated")]
    TruncatedExecFailure,
    #[error("execveat failed in child with errno {0}")]
    ExecveatFailed(i32),
    #[error("child PID {0} was not observed in the target cgroup")]
    BirthMembershipNotObserved(u32),
    #[error("failed to inspect /proc child executable: {0}")]
    ProcExe(#[source] io::Error),
    #[error("failed to read sealed image: {0}")]
    ImageRead(#[source] io::Error),
    #[error("sealed/executed executable digest does not match")]
    ExecutableDigestMismatch,
    #[error("failed to inspect sealed image seals: {0}")]
    SealOperation(#[source] io::Error),
    #[error("sealed image seal mask differs: expected={expected:#x}, actual={actual:#x}")]
    SealMismatch { expected: u32, actual: u32 },
    #[error("failed waiting for launched child: {0}")]
    Wait(#[source] io::Error),
    #[error("serialized execveat evidence is structurally invalid")]
    InvalidEvidence,
    #[error("serialized execveat evidence digest does not match")]
    EvidenceDigestMismatch,
    #[error("invalid canonical lowercase SHA-256 digest")]
    InvalidDigest,
    #[error(transparent)]
    Cgroup(#[from] CgroupV2Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_args_layout_is_frozen() {
        assert_eq!(size_of::<CloneArgs>(), 88);
        assert_eq!(CLONE_INTO_CGROUP_FLAG, 0x2000_0000_0);
    }

    #[test]
    fn evidence_profile_is_frozen() {
        assert_eq!(
            SEALED_EXECVEAT_CGROUP_PROFILE_V1,
            "symthaea.simulation.clone-into-cgroup.sealed-execveat-v1"
        );
    }
}
