// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Landlock filesystem content/mutation confinement for simulation workers.
//!
//! This layer composes with the seccomp/process profile rather than replacing
//! it. The contained worker first closes inherited descriptors, sets
//! `NO_NEW_PRIVS`, and enters the seccomp domain; this crate then creates a
//! Landlock ruleset that handles every filesystem content/mutation right known
//! through Landlock ABI v8 and adds no allow rules. Consequently those handled
//! accesses are denied by default before any untrusted worker frame is parsed.
//!
//! Landlock intentionally does not restrict every metadata operation. In
//! particular, operations such as `stat(2)` and `chdir(2)` are outside the v1
//! claim. This profile establishes path-content/mutation confinement, not full
//! filesystem invisibility.

#![deny(unsafe_op_in_unsafe_fn)]

use std::io;
use std::mem::size_of;
use symthaea_sim_worker_containment::{WorkerContainmentError, install_worker_containment};
use thiserror::Error;

pub const WORKER_FILESYSTEM_PROFILE_V1: &str =
    "symthaea.simulation.worker-filesystem.landlock-deny-content-v1";

const LANDLOCK_CREATE_RULESET_VERSION: u32 = 1;
const LANDLOCK_MIN_ABI: i32 = 1;
const LANDLOCK_MAX_TESTED_ABI: i32 = 8;

// Linux x86_64 syscall numbers. This stack is already x86_64-specific because
// its seccomp parent profile is x86_64-specific.
const SYS_LANDLOCK_CREATE_RULESET: libc::c_long = 444;
const SYS_LANDLOCK_RESTRICT_SELF: libc::c_long = 446;

const LANDLOCK_ACCESS_FS_EXECUTE: u64 = 1 << 0;
const LANDLOCK_ACCESS_FS_WRITE_FILE: u64 = 1 << 1;
const LANDLOCK_ACCESS_FS_READ_FILE: u64 = 1 << 2;
const LANDLOCK_ACCESS_FS_READ_DIR: u64 = 1 << 3;
const LANDLOCK_ACCESS_FS_REMOVE_DIR: u64 = 1 << 4;
const LANDLOCK_ACCESS_FS_REMOVE_FILE: u64 = 1 << 5;
const LANDLOCK_ACCESS_FS_MAKE_CHAR: u64 = 1 << 6;
const LANDLOCK_ACCESS_FS_MAKE_DIR: u64 = 1 << 7;
const LANDLOCK_ACCESS_FS_MAKE_REG: u64 = 1 << 8;
const LANDLOCK_ACCESS_FS_MAKE_SOCK: u64 = 1 << 9;
const LANDLOCK_ACCESS_FS_MAKE_FIFO: u64 = 1 << 10;
const LANDLOCK_ACCESS_FS_MAKE_BLOCK: u64 = 1 << 11;
const LANDLOCK_ACCESS_FS_MAKE_SYM: u64 = 1 << 12;
const LANDLOCK_ACCESS_FS_REFER: u64 = 1 << 13;
const LANDLOCK_ACCESS_FS_TRUNCATE: u64 = 1 << 14;
const LANDLOCK_ACCESS_FS_IOCTL_DEV: u64 = 1 << 15;

const ABI1_RIGHTS: u64 = LANDLOCK_ACCESS_FS_EXECUTE
    | LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_READ_DIR
    | LANDLOCK_ACCESS_FS_REMOVE_DIR
    | LANDLOCK_ACCESS_FS_REMOVE_FILE
    | LANDLOCK_ACCESS_FS_MAKE_CHAR
    | LANDLOCK_ACCESS_FS_MAKE_DIR
    | LANDLOCK_ACCESS_FS_MAKE_REG
    | LANDLOCK_ACCESS_FS_MAKE_SOCK
    | LANDLOCK_ACCESS_FS_MAKE_FIFO
    | LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | LANDLOCK_ACCESS_FS_MAKE_SYM;

#[repr(C)]
struct LandlockRulesetAttrV1 {
    handled_access_fs: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilesystemContainmentStatus {
    pub landlock_abi: i32,
    pub handled_access_fs: u64,
}

#[derive(Debug, Error)]
pub enum WorkerFilesystemError {
    #[error("worker filesystem profile currently supports only Linux x86_64")]
    UnsupportedPlatform,
    #[error(transparent)]
    ProcessContainment(#[from] WorkerContainmentError),
    #[error("NO_NEW_PRIVS is not active before Landlock restriction")]
    NoNewPrivsMissing,
    #[error("Landlock ABI query failed: {0}")]
    AbiQuery(#[source] io::Error),
    #[error("Landlock ABI {0} is older than the required ABI 1")]
    AbiTooOld(i32),
    #[error("Landlock ABI {0} is newer than the highest profile tested ABI 8")]
    AbiUntested(i32),
    #[error("Landlock ruleset creation failed: {0}")]
    RulesetCreate(#[source] io::Error),
    #[error("Landlock restriction failed: {0}")]
    RestrictSelf(#[source] io::Error),
    #[error("failed to close the Landlock ruleset fd after restriction: {0}")]
    RulesetClose(#[source] io::Error),
}

/// Install the parent seccomp/fd profile followed by a deny-by-default Landlock
/// filesystem ruleset. Call this before reading any untrusted worker frame.
pub fn install_worker_filesystem_containment(
) -> Result<FilesystemContainmentStatus, WorkerFilesystemError> {
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        return Err(WorkerFilesystemError::UnsupportedPlatform);
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        install_worker_containment()?;
        require_no_new_privs()?;
        install_landlock_deny_all_content()
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn require_no_new_privs() -> Result<(), WorkerFilesystemError> {
    let value = unsafe { libc::prctl(libc::PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) };
    if value != 1 {
        return Err(WorkerFilesystemError::NoNewPrivsMissing);
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn install_landlock_deny_all_content(
) -> Result<FilesystemContainmentStatus, WorkerFilesystemError> {
    let abi_raw = unsafe {
        libc::syscall(
            SYS_LANDLOCK_CREATE_RULESET,
            std::ptr::null::<libc::c_void>(),
            0usize,
            LANDLOCK_CREATE_RULESET_VERSION,
        )
    };
    if abi_raw < 0 {
        return Err(WorkerFilesystemError::AbiQuery(io::Error::last_os_error()));
    }
    let abi = i32::try_from(abi_raw).unwrap_or(i32::MAX);
    if abi < LANDLOCK_MIN_ABI {
        return Err(WorkerFilesystemError::AbiTooOld(abi));
    }
    if abi > LANDLOCK_MAX_TESTED_ABI {
        return Err(WorkerFilesystemError::AbiUntested(abi));
    }

    let handled_access_fs = rights_for_abi(abi);
    let attr = LandlockRulesetAttrV1 { handled_access_fs };
    let ruleset_fd = unsafe {
        libc::syscall(
            SYS_LANDLOCK_CREATE_RULESET,
            &attr as *const LandlockRulesetAttrV1,
            size_of::<LandlockRulesetAttrV1>(),
            0u32,
        )
    };
    if ruleset_fd < 0 {
        return Err(WorkerFilesystemError::RulesetCreate(
            io::Error::last_os_error(),
        ));
    }

    let restrict_result = unsafe {
        libc::syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd as libc::c_int, 0u32)
    };
    let restrict_error = if restrict_result != 0 {
        Some(io::Error::last_os_error())
    } else {
        None
    };
    let close_result = unsafe { libc::close(ruleset_fd as libc::c_int) };
    let close_error = if close_result != 0 {
        Some(io::Error::last_os_error())
    } else {
        None
    };

    if let Some(error) = restrict_error {
        return Err(WorkerFilesystemError::RestrictSelf(error));
    }
    if let Some(error) = close_error {
        return Err(WorkerFilesystemError::RulesetClose(error));
    }

    Ok(FilesystemContainmentStatus {
        landlock_abi: abi,
        handled_access_fs,
    })
}

const fn rights_for_abi(abi: i32) -> u64 {
    let mut rights = ABI1_RIGHTS;
    if abi >= 2 {
        rights |= LANDLOCK_ACCESS_FS_REFER;
    }
    if abi >= 3 {
        rights |= LANDLOCK_ACCESS_FS_TRUNCATE;
    }
    if abi >= 5 {
        rights |= LANDLOCK_ACCESS_FS_IOCTL_DEV;
    }
    rights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rights_expand_only_at_known_filesystem_abi_boundaries() {
        assert_eq!(rights_for_abi(1), ABI1_RIGHTS);
        assert_eq!(rights_for_abi(2), ABI1_RIGHTS | LANDLOCK_ACCESS_FS_REFER);
        assert_eq!(
            rights_for_abi(3),
            ABI1_RIGHTS | LANDLOCK_ACCESS_FS_REFER | LANDLOCK_ACCESS_FS_TRUNCATE
        );
        assert_eq!(rights_for_abi(4), rights_for_abi(3));
        assert_eq!(
            rights_for_abi(5),
            rights_for_abi(3) | LANDLOCK_ACCESS_FS_IOCTL_DEV
        );
        assert_eq!(rights_for_abi(8), rights_for_abi(5));
    }
}
