//! Real-time memory residency helpers.
//!
//! ## What `mlockall` does and does not buy you
//!
//! Locking pages prevents the kernel from swapping them out or lazily faulting them in, which
//! removes *one* source of nondeterministic latency. It does **not**, on its own, establish
//! hard real-time behaviour: allocation, stack growth, scheduler policy and priority, driver and
//! interrupt latency, lock contention, and every other runtime path remain unconstrained.
//!
//! This module previously claimed, in a doc comment, that locking "guarantees absolute zero
//! page-fault latency during 500Hz control loops", and printed
//! `"Zero-page-fault hard real-time execution active"` on success. Both overstated what the
//! syscall provides. Worse, the function returned `()`: `mlockall` requires `CAP_IPC_LOCK` and
//! fails for any ordinary unprivileged process, in which case the old code printed a warning to
//! stderr and returned normally — leaving **no way for a caller to detect that locking had not
//! happened**. A control loop presented as real-time could run fully swappable and nothing
//! downstream would know.
//!
//! [`lock_memory_pages`] now returns [`std::io::Result`], reports nothing itself, and fails
//! closed on platforms where it cannot lock at all. Presentation and policy belong to the
//! caller, which must choose explicitly between aborting startup and continuing in a declared,
//! visible degraded mode.
//!
//! ## Current call sites
//!
//! None. As of this writing nothing in the workspace calls [`lock_memory_pages`]; it is public
//! API of a crate with many dependents, but no live real-time path depends on it today. The
//! signature change is therefore a latent-defect repair, not an incident fix.

use std::io;

/// Seam over the `mlockall(2)` syscall so the return-code-to-[`io::Error`] propagation is
/// testable without `CAP_IPC_LOCK` and without perturbing the process's real `errno`.
trait MlockAll {
    /// Raw syscall result: `0` on success, `-1` on failure with `errno` set.
    fn mlockall(&self, flags: i32) -> i32;
    /// The error corresponding to a failed [`Self::mlockall`] call.
    fn last_error(&self) -> io::Error;
}

#[cfg(target_os = "linux")]
struct SystemMlockAll;

#[cfg(target_os = "linux")]
impl MlockAll for SystemMlockAll {
    fn mlockall(&self, flags: i32) -> i32 {
        // SAFETY: `mlockall` takes an int of flag bits and has no memory-safety preconditions;
        // it only affects the residency of this process's own mappings. The return value is
        // checked by the sole caller below.
        unsafe { libc::mlockall(flags) }
    }

    fn last_error(&self) -> io::Error {
        io::Error::last_os_error()
    }
}

/// Applies the syscall result, converting failure into a real error rather than a printed
/// warning. Split out from [`lock_memory_pages`] purely so it can be exercised in tests.
fn lock_memory_pages_with(locker: &dyn MlockAll, flags: i32) -> io::Result<()> {
    if locker.mlockall(flags) == 0 {
        Ok(())
    } else {
        Err(locker.last_error())
    }
}

/// Requests that all current and future pages of this process stay resident in RAM
/// (`MCL_CURRENT | MCL_FUTURE`).
///
/// On success, current and future mappings were successfully locked. That is the whole of the
/// claim — see the module docs for why it is not by itself a hard real-time guarantee.
///
/// # Errors
///
/// Returns the OS error if `mlockall` fails. The common cause is missing `CAP_IPC_LOCK`
/// (`EPERM`), which is the default for an unprivileged process; `ENOMEM` indicates the request
/// exceeded the `RLIMIT_MEMLOCK` limit.
///
/// **Callers on a real-time path must not discard this result.** Either abort startup, or enter
/// an explicitly configured degraded mode and surface that state wherever the system reports its
/// own operating status. Silently ignoring the error reproduces exactly the defect this
/// signature exists to prevent.
#[cfg(target_os = "linux")]
pub fn lock_memory_pages() -> io::Result<()> {
    lock_memory_pages_with(&SystemMlockAll, libc::MCL_CURRENT | libc::MCL_FUTURE)
}

/// Non-Linux platforms: fails closed.
///
/// Returns [`io::ErrorKind::Unsupported`] rather than `Ok(())` so a caller cannot mistake
/// "this platform has no implementation" for "pages are locked". The previous implementation
/// printed a warning and returned `()`, which read as success.
#[cfg(not(target_os = "linux"))]
pub fn lock_memory_pages() -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "memory page locking is only implemented for Linux targets",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeMlock {
        rc: i32,
        errno: i32,
    }

    impl MlockAll for FakeMlock {
        fn mlockall(&self, _flags: i32) -> i32 {
            self.rc
        }
        fn last_error(&self) -> io::Error {
            io::Error::from_raw_os_error(self.errno)
        }
    }

    #[test]
    fn success_is_reported_as_ok() {
        let locker = FakeMlock { rc: 0, errno: 0 };
        assert!(lock_memory_pages_with(&locker, 3).is_ok());
    }

    #[test]
    fn eperm_propagates_as_a_real_error() {
        // The realistic failure: no CAP_IPC_LOCK. Previously this printed a warning and the
        // caller received nothing at all.
        let locker = FakeMlock { rc: -1, errno: 1 };
        let err = lock_memory_pages_with(&locker, 3).expect_err("EPERM must surface");
        assert_eq!(err.raw_os_error(), Some(1));
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn enomem_propagates_with_its_own_errno() {
        // RLIMIT_MEMLOCK exceeded — a different operational fix than EPERM, so the distinction
        // has to survive to the caller.
        let locker = FakeMlock { rc: -1, errno: 12 };
        let err = lock_memory_pages_with(&locker, 3).expect_err("ENOMEM must surface");
        assert_eq!(err.raw_os_error(), Some(12));
    }

    #[test]
    fn any_nonzero_return_code_is_a_failure() {
        // mlockall is specified to return -1 on error, but the check must not be `== -1`:
        // treating any other nonzero value as success would fail open.
        let locker = FakeMlock { rc: 7, errno: 1 };
        assert!(lock_memory_pages_with(&locker, 3).is_err());
    }
}
