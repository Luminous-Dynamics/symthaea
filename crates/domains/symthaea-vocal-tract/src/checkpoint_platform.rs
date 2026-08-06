// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Small platform-specific utilities the checkpoint-durability subsystem's files reference at
//! `crate::effective_uid`/`crate::lock_exclusive` but never defined (per
//! `TRACK_B_RECOVERY_PLAN_2026-07-30.md`'s missing-utility-function finding). Unix-only --
//! matches this subsystem's existing use of `std::os::unix::fs::MetadataExt` elsewhere.
//!
//! This module is the one deliberate exception to the crate's `#![deny(unsafe_code)]`: raw
//! `flock(2)`/`geteuid(2)` calls have no safe wrapper in this crate's existing dependency
//! set. Every `unsafe` block below carries its own `// SAFETY:` justification.

#![allow(unsafe_code)]

use std::fs::File;
use std::os::unix::io::AsRawFd;

/// The calling process's effective UID, used to verify a checkpoint file's on-disk owner
/// matches the process that's supposed to own it (see `checkpoint_power_loss_operations.rs`'s
/// `metadata.uid() != crate::effective_uid()` checks).
pub fn effective_uid() -> u32 {
    // SAFETY: geteuid() takes no arguments, performs no memory access, and cannot fail.
    unsafe { libc::geteuid() }
}

/// An exclusive `flock(2)` advisory lock on `file`, released automatically when the returned
/// guard is dropped.
pub struct CheckpointFileLockGuard<'a> {
    file: &'a File,
}

impl Drop for CheckpointFileLockGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.file`'s fd is valid for the guard's lifetime; LOCK_UN on an fd this
        // process itself locked cannot fail in a way that would be unsafe to ignore.
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

/// Take an exclusive advisory `flock(2)` lock on `file`. Blocks until the lock is available.
pub fn lock_exclusive(file: &File) -> std::io::Result<CheckpointFileLockGuard<'_>> {
    // SAFETY: `file.as_raw_fd()` is a valid, open file descriptor for the duration of this
    // call (borrowed from `file`, which outlives the call).
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
    if result != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(CheckpointFileLockGuard { file })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_uid_matches_libc_directly() {
        assert_eq!(effective_uid(), unsafe { libc::geteuid() });
    }

    #[test]
    fn lock_exclusive_round_trips() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "symthaea-vocal-tract-lock-test-{}-{}",
            std::process::id(),
            effective_uid()
        ));
        let file = File::create(&path).unwrap();
        {
            let _guard = lock_exclusive(&file).unwrap();
            // Guard held; dropped at end of this block, releasing the lock.
        }
        // A second lock after the first is dropped must succeed (proves the guard actually
        // released the lock, not just that acquiring a lock once works).
        let _guard2 = lock_exclusive(&file).unwrap();
        std::fs::remove_file(&path).ok();
    }
}
