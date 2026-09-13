// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal Linux `clone3(CLONE_INTO_CGROUP)` primitive.
//!
//! This crate establishes one narrow theorem: a fork-like child can be created
//! directly in one exact live cgroup-v2 lease, rather than being created in the
//! parent's cgroup and migrated later. The child performs only raw, allocation-
//! free libc syscalls after `clone3`; it signals readiness and blocks until the
//! parent releases it. No untrusted code, Rust allocator, Wasmtime, or `exec` is
//! involved in this tranche.
//!
//! A later launcher may compose this primitive with immediate `execveat` of an
//! exact sealed worker image. Persisted evidence here is audit-only and cannot
//! recreate process or cgroup authority.

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
use symthaea_sim_worker_cgroup::{CgroupV2Error, CgroupV2Evidence, CgroupV2Lease};
use thiserror::Error;

pub const CLONE_INTO_CGROUP_PROFILE_V1: &str =
    "symthaea.simulation.clone-into-cgroup.first-instruction-v1";

const EVIDENCE_DOMAIN_V1: &[u8] =
    b"symthaea.simulation.clone-into-cgroup.first-instruction-v1\0";
const CLONE_INTO_CGROUP_FLAG: u64 = 0x2000_0000_0;
const READY_BYTE: u8 = 0xA5;
const RELEASE_BYTE: u8 = 0x5A;

/// Linux's public `struct clone_args` ABI through the cgroup field.
///
/// The structure is versioned by the size passed to `clone3`. All fields not
/// needed by this fork-like primitive are deliberately zero.
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
pub struct CloneIntoCgroupEvidence {
    pub profile: String,
    pub child_pid: u32,
    pub clone_flags: u64,
    pub exit_signal: i32,
    pub target: CgroupV2Evidence,
    pub child_ready_observed: bool,
    pub birth_membership_observed: bool,
    pub evidence_sha256: String,
}

impl CloneIntoCgroupEvidence {
    /// Verify serialized structure and commitments only. This cannot prove the
    /// child is still alive or recreate either process/cgroup authority.
    pub fn verify(&self) -> Result<(), CloneIntoCgroupError> {
        if self.profile != CLONE_INTO_CGROUP_PROFILE_V1
            || self.child_pid == 0
            || self.clone_flags != CLONE_INTO_CGROUP_FLAG
            || self.exit_signal != libc::SIGCHLD
            || !self.child_ready_observed
            || !self.birth_membership_observed
        {
            return Err(CloneIntoCgroupError::InvalidEvidence);
        }
        self.target.verify()?;
        let expected = hex_digest(evidence_sha256_v1(self)?);
        if self.evidence_sha256 != expected {
            return Err(CloneIntoCgroupError::EvidenceDigestMismatch);
        }
        Ok(())
    }
}

/// Live authority over one blocked child born directly in an exact cgroup.
///
/// Deliberately non-cloneable and non-serializable. Dropping an unreaped handle
/// kills and waits the child best-effort so this primitive cannot orphan the
/// process merely because a caller abandoned the handle.
#[derive(Debug)]
pub struct CloneIntoCgroupChild {
    pid: u32,
    release_write: Option<OwnedFd>,
    evidence: CloneIntoCgroupEvidence,
    reaped: bool,
}

impl CloneIntoCgroupChild {
    pub const fn pid(&self) -> u32 {
        self.pid
    }

    pub fn evidence(&self) -> &CloneIntoCgroupEvidence {
        &self.evidence
    }

    /// Release the blocked child and require a clean zero exit.
    pub fn release_and_wait(mut self) -> Result<CloneIntoCgroupEvidence, CloneIntoCgroupError> {
        if let Some(fd) = self.release_write.take() {
            write_byte(fd.as_raw_fd(), RELEASE_BYTE)?;
            drop(fd);
        }
        let status = wait_pid(self.pid)?;
        self.reaped = true;
        if !status.success() {
            return Err(CloneIntoCgroupError::UnexpectedChildExit(status));
        }
        Ok(self.evidence.clone())
    }

    /// Explicitly terminate the child and wait for it to be reaped.
    pub fn terminate_and_wait(mut self) -> Result<CloneIntoCgroupEvidence, CloneIntoCgroupError> {
        terminate_pid(self.pid);
        let _ = wait_pid(self.pid)?;
        self.reaped = true;
        Ok(self.evidence.clone())
    }
}

impl Drop for CloneIntoCgroupChild {
    fn drop(&mut self) {
        if !self.reaped {
            self.release_write.take();
            terminate_pid(self.pid);
            let _ = wait_pid(self.pid);
            self.reaped = true;
        }
    }
}

/// Armed immediately after clone3 returns in the parent so every subsequent
/// failure path reaps the exact child. Disarmed only when ownership transfers
/// into `CloneIntoCgroupChild`.
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

/// Create one fork-like child directly in `lease` using the Linux clone3 cgroup
/// field. The returned child is blocked before any caller-controlled workload
/// can run.
pub fn spawn_blocked_in_cgroup(
    lease: &CgroupV2Lease,
) -> Result<CloneIntoCgroupChild, CloneIntoCgroupError> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = lease;
        return Err(CloneIntoCgroupError::UnsupportedPlatform);
    }

    #[cfg(target_os = "linux")]
    spawn_blocked_linux(lease)
}

#[cfg(target_os = "linux")]
fn spawn_blocked_linux(
    lease: &CgroupV2Lease,
) -> Result<CloneIntoCgroupChild, CloneIntoCgroupError> {
    lease.verify_configuration()?;
    if lease.populated()? {
        return Err(CloneIntoCgroupError::TargetAlreadyPopulated);
    }
    let target = lease.evidence()?;
    target.verify()?;

    let cgroup = File::open(lease.leaf()).map_err(CloneIntoCgroupError::OpenCgroup)?;
    let metadata = cgroup.metadata().map_err(CloneIntoCgroupError::CgroupMetadata)?;
    if metadata.dev() != target.leaf_device || metadata.ino() != target.leaf_inode {
        return Err(CloneIntoCgroupError::CgroupIdentityMismatch);
    }

    let (ready_read, ready_write) = pipe_cloexec()?;
    let (release_read, release_write) = pipe_cloexec()?;

    let args = CloneArgs {
        flags: CLONE_INTO_CGROUP_FLAG,
        exit_signal: libc::SIGCHLD as u64,
        cgroup: u64::try_from(cgroup.as_raw_fd())
            .map_err(|_| CloneIntoCgroupError::InvalidCgroupFd)?,
        ..CloneArgs::default()
    };

    // SAFETY: `args` is the public Linux clone3 ABI, is initialized for a
    // fork-like child (no CLONE_VM/custom stack), and the target fd refers to
    // the already-verified exact cgroup-v2 leaf. The child path immediately
    // enters `child_block_loop`, which uses only raw libc syscalls and `_exit`.
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
        // SAFETY: this is the post-clone child. No Rust allocation, locking,
        // unwinding, or destructor execution occurs before `_exit`.
        unsafe {
            child_block_loop(
                ready_read.as_raw_fd(),
                ready_write.as_raw_fd(),
                release_read.as_raw_fd(),
                release_write.as_raw_fd(),
                cgroup.as_raw_fd(),
            )
        }
    }

    let pid = u32::try_from(result).map_err(|_| CloneIntoCgroupError::InvalidChildPid)?;
    let mut reap_guard = PostCloneReapGuard::new(pid);
    drop(ready_write);
    drop(release_read);
    drop(cgroup);

    let mut ready_file = File::from(ready_read);
    let mut marker = [0u8; 1];
    ready_file
        .read_exact(&mut marker)
        .map_err(CloneIntoCgroupError::ChildReady)?;
    if marker[0] != READY_BYTE {
        return Err(CloneIntoCgroupError::InvalidReadyMarker);
    }
    if !lease.contains_pid(pid)? {
        return Err(CloneIntoCgroupError::BirthMembershipNotObserved(pid));
    }
    lease.verify_configuration()?;

    let mut evidence = CloneIntoCgroupEvidence {
        profile: CLONE_INTO_CGROUP_PROFILE_V1.into(),
        child_pid: pid,
        clone_flags: CLONE_INTO_CGROUP_FLAG,
        exit_signal: libc::SIGCHLD,
        target: lease.evidence()?,
        child_ready_observed: true,
        birth_membership_observed: true,
        evidence_sha256: String::new(),
    };
    evidence.evidence_sha256 = hex_digest(evidence_sha256_v1(&evidence)?);
    evidence.verify()?;

    reap_guard.disarm();
    Ok(CloneIntoCgroupChild {
        pid,
        release_write: Some(release_write),
        evidence,
        reaped: false,
    })
}

/// Child-side syscall-only rendezvous. This must remain allocation-free.
#[cfg(target_os = "linux")]
unsafe fn child_block_loop(
    ready_read: RawFd,
    ready_write: RawFd,
    release_read: RawFd,
    release_write: RawFd,
    cgroup_fd: RawFd,
) -> ! {
    // SAFETY: all descriptors originate from successful pipe/open calls before
    // clone. Close failures are irrelevant to the one-shot child protocol.
    unsafe {
        libc::close(ready_read);
        libc::close(release_write);
        libc::close(cgroup_fd);
    }

    let ready = [READY_BYTE; 1];
    // SAFETY: pointer refers to one initialized stack byte for this syscall.
    let wrote = unsafe { libc::write(ready_write, ready.as_ptr().cast(), 1) };
    if wrote != 1 {
        // SAFETY: immediate process termination; no destructors are required.
        unsafe { libc::_exit(120) }
    }
    // SAFETY: descriptor is no longer needed after readiness notification.
    unsafe { libc::close(ready_write) };

    let mut release = [0u8; 1];
    loop {
        // SAFETY: pointer refers to one writable stack byte.
        let read = unsafe { libc::read(release_read, release.as_mut_ptr().cast(), 1) };
        if read == 1 {
            break;
        }
        if read == 0 {
            // Parent disappeared without authorizing a clean release.
            unsafe { libc::_exit(121) }
        }
        // SAFETY: errno location is thread-local process state exposed by libc.
        let errno = unsafe { *libc::__errno_location() };
        if errno != libc::EINTR {
            unsafe { libc::_exit(122) }
        }
    }
    unsafe { libc::close(release_read) };
    if release[0] != RELEASE_BYTE {
        unsafe { libc::_exit(123) }
    }
    unsafe { libc::_exit(0) }
}

fn pipe_cloexec() -> Result<(OwnedFd, OwnedFd), CloneIntoCgroupError> {
    let mut fds = [-1; 2];
    // SAFETY: `fds` points to storage for exactly two file descriptors.
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(CloneIntoCgroupError::Pipe(io::Error::last_os_error()));
    }
    // SAFETY: successful pipe2 returned two newly owned descriptors.
    let read = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let write = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    Ok((read, write))
}

fn write_byte(fd: RawFd, value: u8) -> Result<(), CloneIntoCgroupError> {
    let bytes = [value; 1];
    loop {
        // SAFETY: `bytes` is initialized and valid for one byte.
        let written = unsafe { libc::write(fd, bytes.as_ptr().cast(), 1) };
        if written == 1 {
            return Ok(());
        }
        if written == 0 {
            return Err(CloneIntoCgroupError::ReleaseWrite(io::Error::new(
                io::ErrorKind::WriteZero,
                "zero-byte release write",
            )));
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(CloneIntoCgroupError::ReleaseWrite(error));
        }
    }
}

fn wait_pid(pid: u32) -> Result<ExitStatus, CloneIntoCgroupError> {
    let pid = i32::try_from(pid).map_err(|_| CloneIntoCgroupError::InvalidChildPid)?;
    let mut raw_status = 0;
    loop {
        // SAFETY: raw_status points to writable storage and pid is a child PID
        // returned by clone3 for this process.
        let result = unsafe { libc::waitpid(pid, &mut raw_status, 0) };
        if result == pid {
            return Ok(ExitStatus::from_raw(raw_status));
        }
        if result < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(CloneIntoCgroupError::Wait(error));
        }
    }
}

fn terminate_pid(pid: u32) {
    if let Ok(pid) = i32::try_from(pid) {
        // SAFETY: sending SIGKILL to the exact unreaped child PID is the
        // best-effort cleanup path for this process-authority handle.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
    }
}

fn classify_clone3_error(error: io::Error) -> CloneIntoCgroupError {
    match error.raw_os_error() {
        Some(libc::ENOSYS) => CloneIntoCgroupError::Clone3Unavailable,
        Some(libc::EACCES) => CloneIntoCgroupError::PlacementDenied,
        Some(libc::EBUSY) => CloneIntoCgroupError::TargetNotLeafDomain,
        Some(libc::EOPNOTSUPP) => CloneIntoCgroupError::TargetDomainInvalid,
        Some(libc::EINVAL) => CloneIntoCgroupError::Clone3Invalid(error),
        _ => CloneIntoCgroupError::Clone3(error),
    }
}

fn evidence_sha256_v1(
    evidence: &CloneIntoCgroupEvidence,
) -> Result<[u8; 32], CloneIntoCgroupError> {
    evidence.target.verify()?;
    let target = parse_hex_32(&evidence.target.evidence_sha256)?;
    let mut hasher = Sha256::new();
    hasher.update(EVIDENCE_DOMAIN_V1);
    hasher.update(evidence.child_pid.to_le_bytes());
    hasher.update(evidence.clone_flags.to_le_bytes());
    hasher.update(evidence.exit_signal.to_le_bytes());
    hasher.update(target);
    hasher.update([u8::from(evidence.child_ready_observed)]);
    hasher.update([u8::from(evidence.birth_membership_observed)]);
    Ok(hasher.finalize().into())
}

fn parse_hex_32(value: &str) -> Result<[u8; 32], CloneIntoCgroupError> {
    if value.len() != 64 {
        return Err(CloneIntoCgroupError::InvalidDigest);
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        *slot = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(out)
}

fn hex_nibble(value: u8) -> Result<u8, CloneIntoCgroupError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(CloneIntoCgroupError::InvalidDigest),
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
pub enum CloneIntoCgroupError {
    #[error("clone-into-cgroup is currently supported only on Linux")]
    UnsupportedPlatform,
    #[error("target cgroup must be empty before first-instruction child creation")]
    TargetAlreadyPopulated,
    #[error("failed to open exact target cgroup: {0}")]
    OpenCgroup(#[source] io::Error),
    #[error("failed to inspect exact target cgroup: {0}")]
    CgroupMetadata(#[source] io::Error),
    #[error("opened target cgroup does not match the live lease identity")]
    CgroupIdentityMismatch,
    #[error("cgroup file descriptor cannot be represented in clone_args")]
    InvalidCgroupFd,
    #[error("failed to create synchronization pipe: {0}")]
    Pipe(#[source] io::Error),
    #[error("clone3 is unavailable on this kernel")]
    Clone3Unavailable,
    #[error("kernel denied CLONE_INTO_CGROUP placement")]
    PlacementDenied,
    #[error("target cgroup is not a valid leaf for CLONE_INTO_CGROUP")]
    TargetNotLeafDomain,
    #[error("target cgroup domain is invalid for CLONE_INTO_CGROUP")]
    TargetDomainInvalid,
    #[error("kernel rejected clone3 arguments: {0}")]
    Clone3Invalid(#[source] io::Error),
    #[error("clone3 failed: {0}")]
    Clone3(#[source] io::Error),
    #[error("clone3 returned an invalid child PID")]
    InvalidChildPid,
    #[error("failed waiting for child readiness: {0}")]
    ChildReady(#[source] io::Error),
    #[error("child readiness marker is invalid")]
    InvalidReadyMarker,
    #[error("child PID {0} was not observed in the exact target cgroup")]
    BirthMembershipNotObserved(u32),
    #[error("failed to release blocked child: {0}")]
    ReleaseWrite(#[source] io::Error),
    #[error("failed waiting for clone3 child: {0}")]
    Wait(#[source] io::Error),
    #[error("clone3 child exited unexpectedly: {0}")]
    UnexpectedChildExit(ExitStatus),
    #[error("serialized first-instruction evidence is structurally invalid")]
    InvalidEvidence,
    #[error("serialized first-instruction evidence digest does not match")]
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
    fn clone_args_layout_reaches_cgroup_field() {
        assert_eq!(size_of::<CloneArgs>(), 88);
    }

    #[test]
    fn profile_and_flag_are_frozen() {
        assert_eq!(
            CLONE_INTO_CGROUP_PROFILE_V1,
            "symthaea.simulation.clone-into-cgroup.first-instruction-v1"
        );
        assert_eq!(CLONE_INTO_CGROUP_FLAG, 0x2000_0000_0);
    }

    #[test]
    fn digest_parser_rejects_noncanonical_hex() {
        let lower = "ab".repeat(32);
        assert_eq!(parse_hex_32(&lower).unwrap(), [0xab; 32]);
        assert!(parse_hex_32(&lower.to_uppercase()).is_err());
    }
}
