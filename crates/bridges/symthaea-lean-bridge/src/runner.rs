// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Optional subprocess wrapper around `lean4 --check`.
//!
//! This module is deliberately skinny. Week-3 milestone: 1 emitted Lean
//! script accepted by `lean4 --check` with zero errors. Most CI work uses
//! this wrapper; unit tests in sibling modules do not require Lean to be
//! installed.

use std::path::Path;

use crate::subprocess::{RunError, run_bounded_on_file};

/// Outcome of running `lean4 --check` on a single `.lean` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckOutcome {
    /// Lean accepted the file with zero errors.
    Accepted,
    /// Lean emitted errors. String is captured stderr for triage.
    Rejected(String),
    /// `lean4` binary not found on PATH.
    LeanNotInstalled,
    /// Other process error (IO, signal, etc.).
    ProcessError(String),
}

impl CheckOutcome {
    pub fn is_accepted(&self) -> bool {
        matches!(self, CheckOutcome::Accepted)
    }
}

/// Resolve the `lean` executable to invoke. Honors `LEAN_PATH_BIN` env var
/// for CI customization, otherwise falls back to `lean` on `PATH`.
fn resolve_lean_binary() -> String {
    std::env::var("LEAN_PATH_BIN").unwrap_or_else(|_| "lean".to_string())
}

/// Run `lean <path>` and classify the outcome.
///
/// In Lean 4, invoking `lean` on a source file type-checks it; there is no
/// separate `--check` flag (that was a Lean 3 convention). A zero exit code
/// means the file type-checked with no errors. Any stderr output from a
/// successful run is informational (e.g. #check statements), not errors.
///
/// Bounded by [`crate::subprocess::run_bounded_on_file`]: a hanging `lean`
/// process is killed after a wall-clock timeout (default 300s, override via
/// `LEAN_CHECK_TIMEOUT_SECS`) rather than blocking the caller forever, and
/// captured output is capped rather than buffered without limit.
pub fn check_with_lean4<P: AsRef<Path>>(path: P) -> CheckOutcome {
    let bin = resolve_lean_binary();
    let output = match run_bounded_on_file(&bin, path.as_ref()) {
        Ok(o) => o,
        Err(RunError::NotFound) => return CheckOutcome::LeanNotInstalled,
        Err(RunError::Io(e)) => return CheckOutcome::ProcessError(e),
    };

    if output.timed_out {
        return CheckOutcome::ProcessError(
            "lean did not finish within the configured timeout and was killed".to_string(),
        );
    }

    if output.success() {
        CheckOutcome::Accepted
    } else {
        // Lean writes errors to stdout in its default output mode, not stderr.
        // Merge both so the rejection string is useful for triage.
        let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
        if output.truncated {
            combined.push_str("\n[... output truncated at the capture limit ...]");
        }
        CheckOutcome::Rejected(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_is_accepted_flag() {
        assert!(CheckOutcome::Accepted.is_accepted());
        assert!(!CheckOutcome::LeanNotInstalled.is_accepted());
        assert!(!CheckOutcome::Rejected("err".into()).is_accepted());
    }
}
