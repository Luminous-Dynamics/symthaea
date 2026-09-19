// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compatibility bridge from the legacy `CommandSolver` API to raw process evidence.
//!
//! This crate deliberately does not change `symthaea-sim-bridge` yet. It proves that
//! the existing public `CommandSolver` configuration can be mapped deterministically
//! into `symthaea-process-capture`, while retaining the historical `execute()`-style
//! success/error semantics through [`CommandSolverCaptureExt::execute_via_capture`].
//! Once this boundary is qualified, `CommandSolver` can delegate internally in a
//! later mechanical simplification without changing its public behavior.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use std::collections::BTreeMap;
use symthaea_process_capture::{
    EnvironmentPolicy, ProcessCapture, ProcessCaptureError, ProcessSpec, ProcessTermination,
    capture_process,
};
use symthaea_sim_bridge::{CommandSolver, SimulationError};

/// Raw-process execution extensions for the existing simulation `CommandSolver`.
pub trait CommandSolverCaptureExt {
    /// Deterministically translate the legacy solver configuration into a process spec.
    ///
    /// The legacy solver inherits unspecified parent environment variables, so this
    /// translation uses [`EnvironmentPolicy::InheritAndOverride`]. Scientific runs
    /// needing a closed environment should construct a `ProcessSpec` directly or use
    /// a later policy-bearing solver API rather than silently changing legacy behavior.
    fn process_spec(&self) -> ProcessSpec;

    /// Execute the solver and preserve bounded raw process evidence.
    ///
    /// Non-zero exit, timeout, and output-limit termination are returned as successful
    /// captures. They are process outcomes, not I/O failures and not convergence claims.
    fn execute_capture(&self) -> Result<ProcessCapture, ProcessCaptureError>;

    /// Execute through the raw-capture layer while reproducing legacy `execute()` semantics.
    ///
    /// A successful direct-child exit returns lossy UTF-8 stdout. Non-zero exit,
    /// timeout, and output overflow become `SimulationError::Adapter`, matching the
    /// historical public contract. This remains a process-level success predicate only;
    /// numerical convergence still belongs to the domain parser.
    fn execute_via_capture(&self) -> Result<String, SimulationError>;
}

impl CommandSolverCaptureExt for CommandSolver {
    fn process_spec(&self) -> ProcessSpec {
        let environment = self
            .env
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>();

        ProcessSpec {
            command: self.cmd.clone(),
            args: self.args.clone(),
            environment,
            environment_policy: EnvironmentPolicy::InheritAndOverride,
            timeout_ms: self.timeout_ms,
            max_output_bytes: self.max_output_bytes,
        }
    }

    fn execute_capture(&self) -> Result<ProcessCapture, ProcessCaptureError> {
        capture_process(&self.process_spec())
    }

    fn execute_via_capture(&self) -> Result<String, SimulationError> {
        let capture = self
            .execute_capture()
            .map_err(|error| SimulationError::Adapter(error.to_string()))?;

        match &capture.termination {
            ProcessTermination::Exited { success: true, .. } => Ok(capture.stdout_lossy()),
            ProcessTermination::Exited {
                exit_code,
                success: false,
            } => Err(SimulationError::Adapter(format!(
                "'{}' exited with code {:?}: {}",
                self.cmd,
                exit_code,
                capture.stderr_lossy()
            ))),
            ProcessTermination::TimedOut => Err(SimulationError::Adapter(format!(
                "'{}' exceeded its {} ms timeout",
                self.cmd, self.timeout_ms
            ))),
            ProcessTermination::OutputLimitExceeded => Err(SimulationError::Adapter(format!(
                "'{}' exceeded the {} byte per-stream output limit",
                self.cmd, self.max_output_bytes
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn mapping_preserves_legacy_command_configuration() {
        let mut solver = CommandSolver::new("printf")
            .arg("%s")
            .arg("hello")
            .timeout(Duration::from_millis(321))
            .max_output_bytes(654);
        solver.env.insert("SYMT_TEST".into(), "value".into());

        let spec = solver.process_spec();
        assert_eq!(spec.command, "printf");
        assert_eq!(
            spec.args,
            vec!["%s".to_string(), "hello".to_string()]
        );
        assert_eq!(
            spec.environment.get("SYMT_TEST").map(String::as_str),
            Some("value")
        );
        assert_eq!(spec.environment_policy, EnvironmentPolicy::InheritAndOverride);
        assert_eq!(spec.timeout_ms, 321);
        assert_eq!(spec.max_output_bytes, 654);
    }

    #[test]
    fn environment_insertion_order_does_not_change_process_identity() {
        let mut left = CommandSolver::new("true");
        left.env.insert("B".into(), "2".into());
        left.env.insert("A".into(), "1".into());

        let mut right = CommandSolver::new("true");
        right.env.insert("A".into(), "1".into());
        right.env.insert("B".into(), "2".into());

        let left_sha = left.process_spec().manifest_sha256().unwrap();
        let right_sha = right.process_spec().manifest_sha256().unwrap();
        assert_eq!(left_sha, right_sha);
    }

    #[test]
    fn successful_capture_and_legacy_adapter_agree() {
        let solver = CommandSolver::new("printf").arg("hello-from-capture");
        let capture = solver.execute_capture().expect("capture should complete");
        assert!(capture.process_success());
        assert_eq!(capture.stdout_lossy(), "hello-from-capture");

        let legacy = solver
            .execute_via_capture()
            .expect("legacy compatibility path should succeed");
        assert_eq!(legacy, "hello-from-capture");
    }

    #[test]
    fn nonzero_exit_is_evidence_but_legacy_path_is_error() {
        let solver = CommandSolver::new("sh").arg("-c").arg("echo failure >&2; exit 7");
        let capture = solver.execute_capture().expect("nonzero exit should be captured");
        assert!(matches!(
            &capture.termination,
            ProcessTermination::Exited {
                exit_code: Some(7),
                success: false
            }
        ));
        assert!(capture.stderr_lossy().contains("failure"));
        assert!(matches!(
            solver.execute_via_capture(),
            Err(SimulationError::Adapter(_))
        ));
    }

    #[test]
    fn timeout_is_evidence_but_legacy_path_is_error() {
        let solver = CommandSolver::new("sleep")
            .arg("1")
            .timeout(Duration::from_millis(20));
        let capture = solver.execute_capture().expect("timeout should be captured");
        assert!(matches!(
            &capture.termination,
            ProcessTermination::TimedOut
        ));
        assert!(matches!(
            solver.execute_via_capture(),
            Err(SimulationError::Adapter(_))
        ));
    }

    #[test]
    fn output_overflow_is_evidence_but_legacy_path_is_error() {
        let solver = CommandSolver::new("printf")
            .arg("0123456789abcdef")
            .max_output_bytes(8);
        let capture = solver
            .execute_capture()
            .expect("output overflow should be captured");
        assert!(matches!(
            &capture.termination,
            ProcessTermination::OutputLimitExceeded
        ));
        assert!(capture.stdout_truncated);
        assert!(matches!(
            solver.execute_via_capture(),
            Err(SimulationError::Adapter(_))
        ));
    }
}
