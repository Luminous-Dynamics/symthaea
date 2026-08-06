// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded-time, bounded-memory subprocess execution for invoking the Lean
//! toolchain.
//!
//! `Command::output()` -- used by `runner::check_with_lean4` and
//! `axiom_gate::run_lean_capture` before this module existed -- blocks with
//! no timeout and buffers stdout/stderr with no cap. A hanging Lean process
//! (or a pathological `.lean` file that makes Lean emit unbounded
//! diagnostic output) would hang or memory-exhaust the caller indefinitely.
//! This wraps spawn + reader threads + a wall-clock deadline, killing the
//! child if either bound is exceeded, mirroring the pattern already used
//! (with `Stdio::null()`, so it didn't need the output cap) by
//! `examples/ingest_minif2f_baseline.rs::run_lake_check`.

use std::ffi::OsStr;
use std::io::Read;
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// Default wall-clock budget for a single Lean invocation. Generous:
/// Mathlib-heavy tactic cascades (`nlinarith`/`polyrith`) can legitimately
/// take tens of seconds even though Lean's own heartbeat limit bounds
/// individual tactic elaboration; this wall clock is a backstop against the
/// whole process hanging (e.g. on an unresponsive resource or a tactic that
/// sidesteps the heartbeat budget via `#eval`), not a policer of normal
/// elaboration time. Overridable via `LEAN_CHECK_TIMEOUT_SECS`.
const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// Cap on captured stdout/stderr bytes, applied independently to each
/// stream. Generous for Lean diagnostics; exists to bound memory if a
/// pathological file makes Lean emit unbounded output.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 8 * 1024 * 1024;

/// The default timeout, honoring `LEAN_CHECK_TIMEOUT_SECS` if set and
/// parseable, else [`DEFAULT_TIMEOUT_SECS`].
pub fn default_timeout() -> Duration {
    std::env::var("LEAN_CHECK_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
}

/// Result of a bounded subprocess run.
#[derive(Debug)]
pub struct BoundedOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// `None` if the process was killed for exceeding the timeout.
    pub status: Option<ExitStatus>,
    /// The process was killed for running past `timeout`.
    pub timed_out: bool,
    /// Stdout and/or stderr hit `max_output_bytes` and further output from
    /// that stream was discarded (the pipe is still drained so the child
    /// doesn't block on a full buffer).
    pub truncated: bool,
}

impl BoundedOutput {
    pub fn success(&self) -> bool {
        self.status.map(|s| s.success()).unwrap_or(false)
    }
}

#[derive(Debug)]
pub enum RunError {
    NotFound,
    Io(String),
}

/// Run `bin` with `args`, bounded by `timeout` (wall clock) and
/// `max_output_bytes` (per stream). Never blocks past `timeout` and never
/// buffers more than `max_output_bytes` per stream, regardless of how the
/// child behaves.
pub fn run_bounded(
    bin: &str,
    args: &[&OsStr],
    timeout: Duration,
    max_output_bytes: usize,
) -> Result<BoundedOutput, RunError> {
    let mut child = match Command::new(bin)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(RunError::NotFound),
        Err(e) => return Err(RunError::Io(e.to_string())),
    };

    // Read stdout/stderr on their own threads so the child never blocks on
    // a full pipe while this thread is polling `try_wait` -- the same
    // reason `Command::output()` itself uses reader threads internally.
    let mut stdout_pipe = child.stdout.take().expect("stdout was piped");
    let mut stderr_pipe = child.stderr.take().expect("stderr was piped");
    let stdout_thread = std::thread::spawn(move || read_capped(&mut stdout_pipe, max_output_bytes));
    let stderr_thread = std::thread::spawn(move || read_capped(&mut stderr_pipe, max_output_bytes));

    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(RunError::Io(e.to_string())),
        }
    };

    // Killing the child closes its ends of the pipes, so the reader
    // threads see EOF and exit shortly after -- these joins don't add an
    // unbounded wait on top of the timeout above.
    let (stdout, stdout_truncated) = stdout_thread.join().unwrap_or_default();
    let (stderr, stderr_truncated) = stderr_thread.join().unwrap_or_default();

    Ok(BoundedOutput {
        stdout,
        stderr,
        status,
        timed_out,
        truncated: stdout_truncated || stderr_truncated,
    })
}

/// Convenience wrapper for the common case: a single path argument, default
/// timeout/output cap.
pub fn run_bounded_on_file(bin: &str, path: &Path) -> Result<BoundedOutput, RunError> {
    run_bounded(
        bin,
        &[path.as_os_str()],
        default_timeout(),
        DEFAULT_MAX_OUTPUT_BYTES,
    )
}

fn read_capped<R: Read>(r: &mut R, cap: usize) -> (Vec<u8>, bool) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        match r.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                let remaining = cap.saturating_sub(buf.len());
                if remaining > 0 {
                    let take = n.min(remaining);
                    buf.extend_from_slice(&chunk[..take]);
                }
                // If remaining == 0, keep looping (draining the pipe
                // without retaining bytes) so the child never blocks on a
                // full pipe just because we stopped storing its output.
            }
            Err(_) => break,
        }
    }
    let truncated = buf.len() >= cap;
    (buf, truncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonexistent_binary_reports_not_found() {
        let result = run_bounded(
            "symthaea-no-such-binary-xyzzy",
            &[],
            Duration::from_secs(5),
            1024,
        );
        assert!(matches!(result, Err(RunError::NotFound)));
    }

    #[test]
    fn output_is_captured_and_status_reflects_exit_code() {
        let out = run_bounded(
            "sh",
            &[OsStr::new("-c"), OsStr::new("echo hello; exit 0")],
            Duration::from_secs(5),
            1024,
        )
        .unwrap();
        assert!(out.success());
        assert!(!out.timed_out);
        assert!(!out.truncated);
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hello");
    }

    #[test]
    fn oversized_output_is_truncated_not_buffered_unboundedly() {
        // Emits far more than the 16-byte cap; must not hang or panic, and
        // must report truncation.
        let out = run_bounded(
            "sh",
            &[
                OsStr::new("-c"),
                OsStr::new("i=0; while [ $i -lt 100000 ]; do printf 'x'; i=$((i+1)); done"),
            ],
            Duration::from_secs(10),
            16,
        )
        .unwrap();
        assert!(out.truncated);
        assert_eq!(out.stdout.len(), 16);
    }

    #[test]
    fn hanging_process_is_killed_at_timeout() {
        let start = Instant::now();
        let out = run_bounded(
            "sh",
            &[OsStr::new("-c"), OsStr::new("sleep 30")],
            Duration::from_millis(200),
            1024,
        )
        .unwrap();
        assert!(out.timed_out);
        assert!(out.status.is_none());
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "must not wait anywhere near the full 30s sleep"
        );
    }
}
