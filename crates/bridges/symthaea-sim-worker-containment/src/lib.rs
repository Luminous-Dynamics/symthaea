// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Linux syscall/process boundary for the supervised simulation worker.
//!
//! This profile is installed inside the worker as the first action in `main`,
//! after the kernel/dynamic loader have entered the exact sealed executable but
//! before any untrusted request frame is parsed or Wasmtime sees Component
//! bytes. It closes inherited descriptors above stderr, sets `NO_NEW_PRIVS`,
//! then installs a classic seccomp-BPF filter.
//!
//! The v1 filter deliberately permits normal memory management, `memfd_create`,
//! and pthread-style threads needed by Wasmtime and the host deadline timer.
//! `clone3` is reported as `ENOSYS` so libc falls back to legacy `clone`; legacy
//! `clone` is permitted only when `CLONE_THREAD` is present. New processes,
//! later exec, networking, io_uring, ptrace/process-memory APIs,
//! namespace/mount manipulation, BPF/perf, keyring and several privileged
//! kernel surfaces fail closed.
//!
//! This is not filesystem confinement: ordinary file-open syscalls remain
//! available in v1. A Landlock/mount-namespace layer is a separate future claim.

#![deny(unsafe_op_in_unsafe_fn)]

use std::io;
use thiserror::Error;

pub const WORKER_CONTAINMENT_PROFILE_V1: &str =
    "symthaea.simulation.worker-containment.seccomp-x86_64-v1";

const SECCOMP_SET_MODE_FILTER: libc::c_uint = 1;
const SECCOMP_MODE_FILTER: libc::c_int = 2;
const AUDIT_ARCH_X86_64: u32 = 0xc000_003e;
const SECCOMP_RET_KILL_PROCESS: u32 = 0x8000_0000;
const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
const CLONE_THREAD_FLAG: u32 = 0x0001_0000;

const BPF_LD: u16 = 0x00;
const BPF_W: u16 = 0x00;
const BPF_ABS: u16 = 0x20;
const BPF_JMP: u16 = 0x05;
const BPF_JEQ: u16 = 0x10;
const BPF_JSET: u16 = 0x40;
const BPF_K: u16 = 0x00;
const BPF_RET: u16 = 0x06;

const SECCOMP_NR_OFFSET: u32 = 0;
const SECCOMP_ARCH_OFFSET: u32 = 4;
const SECCOMP_ARG0_OFFSET: u32 = 16;

// Stable Linux x86_64 syscall numbers. This profile intentionally refuses to
// install on another architecture rather than silently applying wrong numbers.
const SYS_SOCKET: u32 = 41;
const SYS_CONNECT: u32 = 42;
const SYS_ACCEPT: u32 = 43;
const SYS_SENDTO: u32 = 44;
const SYS_RECVFROM: u32 = 45;
const SYS_SENDMSG: u32 = 46;
const SYS_RECVMSG: u32 = 47;
const SYS_SHUTDOWN: u32 = 48;
const SYS_BIND: u32 = 49;
const SYS_LISTEN: u32 = 50;
const SYS_GETSOCKNAME: u32 = 51;
const SYS_GETPEERNAME: u32 = 52;
const SYS_SOCKETPAIR: u32 = 53;
const SYS_SETSOCKOPT: u32 = 54;
const SYS_GETSOCKOPT: u32 = 55;
const SYS_CLONE: u32 = 56;
const SYS_FORK: u32 = 57;
const SYS_VFORK: u32 = 58;
const SYS_EXECVE: u32 = 59;
const SYS_KILL: u32 = 62;
const SYS_PTRACE: u32 = 101;
const SYS_RT_SIGQUEUEINFO: u32 = 129;
const SYS_USELIB: u32 = 134;
const SYS_PERSONALITY: u32 = 135;
const SYS_PIVOT_ROOT: u32 = 155;
const SYS_ADJTIMEX: u32 = 159;
const SYS_CHROOT: u32 = 161;
const SYS_ACCT: u32 = 163;
const SYS_SETTIMEOFDAY: u32 = 164;
const SYS_MOUNT: u32 = 165;
const SYS_UMOUNT2: u32 = 166;
const SYS_SWAPON: u32 = 167;
const SYS_SWAPOFF: u32 = 168;
const SYS_REBOOT: u32 = 169;
const SYS_SETHOSTNAME: u32 = 170;
const SYS_SETDOMAINNAME: u32 = 171;
const SYS_IOPL: u32 = 172;
const SYS_IOPERM: u32 = 173;
const SYS_INIT_MODULE: u32 = 175;
const SYS_DELETE_MODULE: u32 = 176;
const SYS_QUOTACTL: u32 = 179;
const SYS_TKILL: u32 = 200;
const SYS_TGKILL: u32 = 234;
const SYS_KEXEC_LOAD: u32 = 246;
const SYS_ADD_KEY: u32 = 248;
const SYS_REQUEST_KEY: u32 = 249;
const SYS_KEYCTL: u32 = 250;
const SYS_UNSHARE: u32 = 272;
const SYS_ACCEPT4: u32 = 288;
const SYS_RT_TGSIGQUEUEINFO: u32 = 297;
const SYS_PERF_EVENT_OPEN: u32 = 298;
const SYS_RECVMMSG: u32 = 299;
const SYS_FANOTIFY_INIT: u32 = 300;
const SYS_NAME_TO_HANDLE_AT: u32 = 303;
const SYS_OPEN_BY_HANDLE_AT: u32 = 304;
const SYS_CLOCK_ADJTIME: u32 = 305;
const SYS_SENDMMSG: u32 = 307;
const SYS_SETNS: u32 = 308;
const SYS_PROCESS_VM_READV: u32 = 310;
const SYS_PROCESS_VM_WRITEV: u32 = 311;
const SYS_KCMP: u32 = 312;
const SYS_FINIT_MODULE: u32 = 313;
const SYS_SECCOMP: u32 = 317;
const SYS_KEXEC_FILE_LOAD: u32 = 320;
const SYS_BPF: u32 = 321;
const SYS_EXECVEAT: u32 = 322;
const SYS_USERFAULTFD: u32 = 323;
const SYS_PIDFD_SEND_SIGNAL: u32 = 424;
const SYS_IO_URING_SETUP: u32 = 425;
const SYS_IO_URING_ENTER: u32 = 426;
const SYS_IO_URING_REGISTER: u32 = 427;
const SYS_OPEN_TREE: u32 = 428;
const SYS_MOVE_MOUNT: u32 = 429;
const SYS_FSOPEN: u32 = 430;
const SYS_FSCONFIG: u32 = 431;
const SYS_FSMOUNT: u32 = 432;
const SYS_FSPICK: u32 = 433;
const SYS_PIDFD_OPEN: u32 = 434;
const SYS_CLONE3: u32 = 435;
const SYS_PIDFD_GETFD: u32 = 438;
const SYS_PROCESS_MADVISE: u32 = 440;
const SYS_MOUNT_SETATTR: u32 = 442;
const SYS_MEMFD_SECRET: u32 = 447;
const SYS_PROCESS_MRELEASE: u32 = 448;

const DENY_EPERM: &[u32] = &[
    SYS_SOCKET,
    SYS_CONNECT,
    SYS_ACCEPT,
    SYS_SENDTO,
    SYS_RECVFROM,
    SYS_SENDMSG,
    SYS_RECVMSG,
    SYS_SHUTDOWN,
    SYS_BIND,
    SYS_LISTEN,
    SYS_GETSOCKNAME,
    SYS_GETPEERNAME,
    SYS_SOCKETPAIR,
    SYS_SETSOCKOPT,
    SYS_GETSOCKOPT,
    SYS_FORK,
    SYS_VFORK,
    SYS_EXECVE,
    SYS_KILL,
    SYS_PTRACE,
    SYS_RT_SIGQUEUEINFO,
    SYS_USELIB,
    SYS_PERSONALITY,
    SYS_PIVOT_ROOT,
    SYS_ADJTIMEX,
    SYS_CHROOT,
    SYS_ACCT,
    SYS_SETTIMEOFDAY,
    SYS_MOUNT,
    SYS_UMOUNT2,
    SYS_SWAPON,
    SYS_SWAPOFF,
    SYS_REBOOT,
    SYS_SETHOSTNAME,
    SYS_SETDOMAINNAME,
    SYS_IOPL,
    SYS_IOPERM,
    SYS_INIT_MODULE,
    SYS_DELETE_MODULE,
    SYS_QUOTACTL,
    SYS_TKILL,
    SYS_TGKILL,
    SYS_KEXEC_LOAD,
    SYS_ADD_KEY,
    SYS_REQUEST_KEY,
    SYS_KEYCTL,
    SYS_UNSHARE,
    SYS_ACCEPT4,
    SYS_RT_TGSIGQUEUEINFO,
    SYS_PERF_EVENT_OPEN,
    SYS_RECVMMSG,
    SYS_FANOTIFY_INIT,
    SYS_NAME_TO_HANDLE_AT,
    SYS_OPEN_BY_HANDLE_AT,
    SYS_CLOCK_ADJTIME,
    SYS_SENDMMSG,
    SYS_SETNS,
    SYS_PROCESS_VM_READV,
    SYS_PROCESS_VM_WRITEV,
    SYS_KCMP,
    SYS_FINIT_MODULE,
    SYS_SECCOMP,
    SYS_KEXEC_FILE_LOAD,
    SYS_BPF,
    SYS_EXECVEAT,
    SYS_USERFAULTFD,
    SYS_PIDFD_SEND_SIGNAL,
    SYS_IO_URING_SETUP,
    SYS_IO_URING_ENTER,
    SYS_IO_URING_REGISTER,
    SYS_OPEN_TREE,
    SYS_MOVE_MOUNT,
    SYS_FSOPEN,
    SYS_FSCONFIG,
    SYS_FSMOUNT,
    SYS_FSPICK,
    SYS_PIDFD_OPEN,
    SYS_PIDFD_GETFD,
    SYS_PROCESS_MADVISE,
    SYS_MOUNT_SETATTR,
    SYS_MEMFD_SECRET,
    SYS_PROCESS_MRELEASE,
];

#[derive(Debug, Error)]
pub enum WorkerContainmentError {
    #[error("worker containment v1 currently supports only Linux x86_64")]
    UnsupportedPlatform,
    #[error("failed to close inherited worker descriptors: {0}")]
    CloseInherited(#[source] io::Error),
    #[error("failed to set PR_SET_NO_NEW_PRIVS: {0}")]
    NoNewPrivs(#[source] io::Error),
    #[error("seccomp program is too large")]
    ProgramTooLarge,
    #[error("failed to install seccomp filter: {0}")]
    SeccompInstall(#[source] io::Error),
    #[error("seccomp mode is not active after installation")]
    SeccompInactive,
}

/// Close inherited non-stdio descriptors, set no-new-privs, and install the v1
/// seccomp filter. Call this before parsing any untrusted worker frame.
pub fn install_worker_containment() -> Result<(), WorkerContainmentError> {
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        return Err(WorkerContainmentError::UnsupportedPlatform);
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        close_inherited_descriptors()?;
        set_no_new_privs()?;
        install_seccomp_filter()?;
        if !seccomp_filter_active()? {
            return Err(WorkerContainmentError::SeccompInactive);
        }
        Ok(())
    }
}

/// Report whether Linux says the current thread is in seccomp filter mode.
pub fn seccomp_filter_active() -> Result<bool, WorkerContainmentError> {
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        return Err(WorkerContainmentError::UnsupportedPlatform);
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let mode = unsafe { libc::prctl(libc::PR_GET_SECCOMP, 0, 0, 0, 0) };
        Ok(mode == SECCOMP_MODE_FILTER)
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn close_inherited_descriptors() -> Result<(), WorkerContainmentError> {
    // close_range(2) is available on current qualification kernels. Fall back to
    // the inherited RLIMIT_NOFILE ceiling for older kernels.
    let rc = unsafe { libc::syscall(436, 3u32, u32::MAX, 0u32) };
    if rc == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() != Some(libc::ENOSYS) {
        return Err(WorkerContainmentError::CloseInherited(error));
    }

    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return Err(WorkerContainmentError::CloseInherited(
            io::Error::last_os_error(),
        ));
    }
    let maximum = limit.rlim_cur.min(1_048_576);
    for fd in 3..maximum {
        unsafe {
            libc::close(fd as libc::c_int);
        }
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn set_no_new_privs() -> Result<(), WorkerContainmentError> {
    if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
        return Err(WorkerContainmentError::NoNewPrivs(
            io::Error::last_os_error(),
        ));
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn install_seccomp_filter() -> Result<(), WorkerContainmentError> {
    let mut filters = build_filter();
    let length = u16::try_from(filters.len()).map_err(|_| WorkerContainmentError::ProgramTooLarge)?;
    let program = libc::sock_fprog {
        len: length,
        filter: filters.as_mut_ptr(),
    };
    let rc = unsafe {
        libc::syscall(
            317,
            SECCOMP_SET_MODE_FILTER,
            0u32,
            &program as *const libc::sock_fprog,
        )
    };
    if rc != 0 {
        return Err(WorkerContainmentError::SeccompInstall(
            io::Error::last_os_error(),
        ));
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn build_filter() -> Vec<libc::sock_filter> {
    let eperm = SECCOMP_RET_ERRNO | libc::EPERM as u32;
    let enosys = SECCOMP_RET_ERRNO | libc::ENOSYS as u32;
    let mut filter = vec![
        stmt(BPF_LD | BPF_W | BPF_ABS, SECCOMP_ARCH_OFFSET),
        jump(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        stmt(BPF_LD | BPF_W | BPF_ABS, SECCOMP_NR_OFFSET),
        // Legacy clone is permitted only for pthread-style CLONE_THREAD calls.
        jump(BPF_JMP | BPF_JEQ | BPF_K, SYS_CLONE, 0, 4),
        stmt(BPF_LD | BPF_W | BPF_ABS, SECCOMP_ARG0_OFFSET),
        jump(BPF_JMP | BPF_JSET | BPF_K, CLONE_THREAD_FLAG, 1, 0),
        stmt(BPF_RET | BPF_K, eperm),
        stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        // Make libc see clone3 as unavailable so pthread_create falls back to
        // legacy clone, whose flags we can inspect directly in classic BPF.
        jump(BPF_JMP | BPF_JEQ | BPF_K, SYS_CLONE3, 0, 1),
        stmt(BPF_RET | BPF_K, enosys),
    ];

    for syscall in DENY_EPERM {
        filter.push(jump(BPF_JMP | BPF_JEQ | BPF_K, *syscall, 0, 1));
        filter.push(stmt(BPF_RET | BPF_K, eperm));
    }
    filter.push(stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
    filter
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const fn stmt(code: u16, k: u32) -> libc::sock_filter {
    libc::sock_filter {
        code,
        jt: 0,
        jf: 0,
        k,
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const fn jump(code: u16, k: u32, jt: u8, jf: u8) -> libc::sock_filter {
    libc::sock_filter { code, jt, jf, k }
}
