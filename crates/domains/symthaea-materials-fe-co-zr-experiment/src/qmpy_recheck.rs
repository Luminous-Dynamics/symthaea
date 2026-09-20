// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh qmpy importability recheck for every runtime-readiness consumption.
//!
//! The stored runtime gate proves that the exact selected Python/qmpy path imported
//! successfully when the gate was minted. This module closes the later TOCTOU gap by
//! executing the same isolated import probe again immediately before a stored gate is
//! consumed. It deliberately does not claim that the complete Python closure has been
//! independently enumerated or content-addressed.

use crate::readiness_gate::QmpyImportProbeReceipt;
use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use symthaea_materials_fe_co_zr_experiment::LocalExperimentConfig;
use thiserror::Error;

const QMPY_IMPORT_MARKER: &str =
    "QMPY_IMPORT_OK:qmpy.materials.formation_energy.FormationEnergy";

// Keep byte-identical with the probe frozen by MAG-EXP-002. The stored capture's exact
// argv is checked before execution, so a future change cannot silently reinterpret an
// older readiness receipt.
const QMPY_IMPORT_SCRIPT: &str = r#"import logging
import logging.handlers
import os
os.environ.setdefault('qmdb_v1_1_pswd', '')
_original = logging.handlers.WatchedFileHandler
logging.handlers.WatchedFileHandler = lambda *args, **kwargs: logging.NullHandler()
try:
    from qmpy.materials.formation_energy import FormationEnergy
    print('QMPY_IMPORT_OK:' + FormationEnergy.__module__ + '.' + FormationEnergy.__name__)
finally:
    logging.handlers.WatchedFileHandler = _original
"#;

/// Re-execute the exact frozen qmpy import probe and require the same observable result.
pub fn require_current_qmpy_import(
    config: &LocalExperimentConfig,
    stored: &QmpyImportProbeReceipt,
) -> Result<(), QmpyFreshnessError> {
    let expected_args = probe_args();
    if stored.capture.executable_path != config.python_path
        || stored.capture.args != expected_args
        || stored.capture.exit_code != 0
        || !stored
            .capture
            .stdout
            .lines()
            .any(|line| line.trim() == QMPY_IMPORT_MARKER)
    {
        return Err(QmpyFreshnessError::StoredProbeContractMismatch);
    }

    let current = execute_probe(config, &expected_args)?;
    if current.exit_code != 0
        || !current
            .stdout
            .lines()
            .any(|line| line.trim() == QMPY_IMPORT_MARKER)
    {
        return Err(QmpyFreshnessError::CurrentImportFailed {
            exit_code: current.exit_code,
            stdout: current.stdout,
            stderr: current.stderr,
        });
    }

    if current.stdout != stored.capture.stdout || current.stderr != stored.capture.stderr {
        return Err(QmpyFreshnessError::CurrentObservationChanged);
    }

    Ok(())
}

fn probe_args() -> Vec<String> {
    vec![
        "-I".to_string(),
        "-c".to_string(),
        QMPY_IMPORT_SCRIPT.to_string(),
    ]
}

struct CurrentCapture {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

fn execute_probe(
    config: &LocalExperimentConfig,
    args: &[String],
) -> Result<CurrentCapture, QmpyFreshnessError> {
    let mut command = Command::new(&config.python_path);
    command.args(args).env_clear();
    command
        .env("LC_ALL", "C")
        .env("LANG", "C")
        .env("TZ", "UTC")
        .env("qmdb_v1_1_pswd", "")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|source| QmpyFreshnessError::Io {
            context: format!("spawn {}", config.python_path),
            source,
        })?;

    let deadline = Instant::now() + Duration::from_millis(config.probe_timeout_ms);
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|source| QmpyFreshnessError::Io {
                context: format!("wait for {}", config.python_path),
                source,
            })?
        {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(QmpyFreshnessError::TimedOut(config.python_path.clone()));
        }
        thread::sleep(Duration::from_millis(10));
    };

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_end(&mut stdout)
            .map_err(|source| QmpyFreshnessError::Io {
                context: format!("read stdout for {}", config.python_path),
                source,
            })?;
    }
    if let Some(mut pipe) = child.stderr.take() {
        pipe.read_to_end(&mut stderr)
            .map_err(|source| QmpyFreshnessError::Io {
                context: format!("read stderr for {}", config.python_path),
                source,
            })?;
    }
    if stdout.len() > config.max_output_bytes || stderr.len() > config.max_output_bytes {
        return Err(QmpyFreshnessError::DiagnosticLimitExceeded);
    }

    Ok(CurrentCapture {
        exit_code: status.code().unwrap_or(-1),
        stdout: String::from_utf8(stdout).map_err(|_| QmpyFreshnessError::NonUtf8Output)?,
        stderr: String::from_utf8(stderr).map_err(|_| QmpyFreshnessError::NonUtf8Output)?,
    })
}

/// Failure while proving that qmpy remains freshly importable under the frozen probe.
#[derive(Debug, Error)]
pub enum QmpyFreshnessError {
    /// Stored readiness evidence no longer matches the exact probe contract consumed here.
    #[error("stored qmpy readiness capture does not match the frozen import-probe contract")]
    StoredProbeContractMismatch,
    /// The current runtime no longer reproduces the stored successful observation.
    #[error("current qmpy import observation differs from the stored readiness observation")]
    CurrentObservationChanged,
    /// The current executable failed the fresh FormationEnergy import.
    #[error("current qmpy import failed (exit {exit_code}); stdout={stdout:?}; stderr={stderr:?}")]
    CurrentImportFailed {
        /// Process exit code, or -1 when unavailable.
        exit_code: i32,
        /// Captured stdout.
        stdout: String,
        /// Captured stderr.
        stderr: String,
    },
    /// The fresh probe exceeded the configured runtime-readiness timeout.
    #[error("current qmpy import probe timed out: {0}")]
    TimedOut(String),
    /// Probe output exceeded the configured diagnostic retention bound.
    #[error("current qmpy import probe output exceeded the configured diagnostic bound")]
    DiagnosticLimitExceeded,
    /// Probe output was not valid UTF-8.
    #[error("current qmpy import probe output was not valid UTF-8")]
    NonUtf8Output,
    /// Operating-system I/O failure while executing the probe.
    #[error("I/O while {context}: {source}")]
    Io {
        /// Operation being attempted.
        context: String,
        /// Underlying operating-system error.
        #[source]
        source: std::io::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_contract_is_isolated_and_exact() {
        let args = probe_args();
        assert_eq!(args[0], "-I");
        assert_eq!(args[1], "-c");
        assert_eq!(args[2], QMPY_IMPORT_SCRIPT);
        assert!(QMPY_IMPORT_SCRIPT.contains("FormationEnergy"));
        assert!(QMPY_IMPORT_SCRIPT.contains("WatchedFileHandler"));
    }
}
