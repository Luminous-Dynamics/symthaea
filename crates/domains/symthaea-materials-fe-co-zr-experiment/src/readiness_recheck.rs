// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cheap current-state rechecks for a previously issued runtime readiness gate.

use crate::readiness_gate::MysqlLiveObservation;
use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use symthaea_materials_fe_co_zr_experiment::LocalExperimentConfig;
use thiserror::Error;

const LIVE_QUERY: &str = "SELECT @@version,@@version_comment,@@character_set_server,@@collation_server,@@sql_mode,@@lower_case_table_names,@@time_zone,@@max_allowed_packet,@@innodb_strict_mode,@@socket,@@pid_file";

pub fn require_current_mysql_observation(
    config: &LocalExperimentConfig,
    expected: &MysqlLiveObservation,
) -> Result<(), CurrentReadinessError> {
    let observed = observe_current_mysql(config)?;
    if &observed != expected {
        return Err(CurrentReadinessError::LiveObservationChanged {
            expected: expected.clone(),
            observed,
        });
    }
    Ok(())
}

fn observe_current_mysql(
    config: &LocalExperimentConfig,
) -> Result<MysqlLiveObservation, CurrentReadinessError> {
    let args = vec![
        "--no-defaults".to_string(),
        "--protocol=SOCKET".to_string(),
        format!("--socket={}", config.unix_socket_path),
        format!("--user={}", config.mysql_user),
        "--batch".to_string(),
        "--raw".to_string(),
        "--skip-column-names".to_string(),
        format!("--default-character-set={}", config.character_set_server),
        format!("--execute={LIVE_QUERY}"),
        config.database_name.clone(),
    ];
    let stdout = run_bounded_query(
        &config.mysql_client_path,
        &args,
        config.probe_timeout_ms,
        config.max_output_bytes,
    )?;
    parse_live_observation(&stdout)
}

fn run_bounded_query(
    executable: &str,
    args: &[String],
    timeout_ms: u64,
    max_output_bytes: usize,
) -> Result<String, CurrentReadinessError> {
    let mut child = Command::new(executable)
        .args(args)
        .env_clear()
        .env("LC_ALL", "C")
        .env("LANG", "C")
        .env("TZ", "UTC")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(CurrentReadinessError::Io)?;

    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let status = loop {
        if let Some(status) = child.try_wait().map_err(CurrentReadinessError::Io)? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(CurrentReadinessError::TimedOut);
        }
        thread::sleep(Duration::from_millis(10));
    };

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_end(&mut stdout).map_err(CurrentReadinessError::Io)?;
    }
    if let Some(mut pipe) = child.stderr.take() {
        pipe.read_to_end(&mut stderr).map_err(CurrentReadinessError::Io)?;
    }
    if stdout.len() > max_output_bytes || stderr.len() > max_output_bytes {
        return Err(CurrentReadinessError::DiagnosticLimitExceeded);
    }
    let stdout = String::from_utf8(stdout).map_err(|_| CurrentReadinessError::NonUtf8Output)?;
    let stderr = String::from_utf8(stderr).map_err(|_| CurrentReadinessError::NonUtf8Output)?;
    if !status.success() {
        return Err(CurrentReadinessError::QueryFailed {
            exit_code: status.code().unwrap_or(-1),
            stdout,
            stderr,
        });
    }
    Ok(stdout)
}

fn parse_live_observation(stdout: &str) -> Result<MysqlLiveObservation, CurrentReadinessError> {
    let columns: Vec<&str> = stdout.trim().split('\t').collect();
    if columns.len() != 11 {
        return Err(CurrentReadinessError::UnexpectedOutput(format!(
            "expected 11 columns, got {}",
            columns.len()
        )));
    }
    let mut sql_mode: Vec<String> = columns[4]
        .split(',')
        .filter(|mode| !mode.is_empty())
        .map(str::to_string)
        .collect();
    sql_mode.sort();
    sql_mode.dedup();
    Ok(MysqlLiveObservation {
        runtime_version: columns[0].to_string(),
        version_comment: columns[1].to_string(),
        character_set_server: columns[2].to_string(),
        collation_server: columns[3].to_string(),
        sql_mode,
        lower_case_table_names: columns[5]
            .parse()
            .map_err(|_| CurrentReadinessError::UnexpectedOutput("lower_case_table_names".to_string()))?,
        time_zone: columns[6].to_string(),
        max_allowed_packet_bytes: columns[7]
            .parse()
            .map_err(|_| CurrentReadinessError::UnexpectedOutput("max_allowed_packet".to_string()))?,
        innodb_strict_mode: matches!(columns[8], "1" | "ON" | "on"),
        socket_path: columns[9].to_string(),
        pid_file: columns[10].to_string(),
    })
}

#[derive(Debug, Error)]
pub enum CurrentReadinessError {
    #[error("live MySQL observation changed after runtime-readiness evidence; expected={expected:?}; observed={observed:?}")]
    LiveObservationChanged {
        expected: MysqlLiveObservation,
        observed: MysqlLiveObservation,
    },
    #[error("current MySQL readiness query timed out")]
    TimedOut,
    #[error("current MySQL readiness query failed with exit {exit_code}; stdout={stdout:?}; stderr={stderr:?}")]
    QueryFailed {
        exit_code: i32,
        stdout: String,
        stderr: String,
    },
    #[error("current MySQL readiness output exceeded the configured diagnostic bound")]
    DiagnosticLimitExceeded,
    #[error("current MySQL readiness output was not UTF-8")]
    NonUtf8Output,
    #[error("unexpected current MySQL readiness output: {0}")]
    UnexpectedOutput(String),
    #[error("I/O during current MySQL readiness query: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_canonicalizes_sql_mode_before_comparison() {
        let parsed = parse_live_observation(
            "8.4.0\tPercona\tutf8mb4\tutf8mb4_bin\tSTRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION\t0\t+00:00\t67108864\tON\t/run/mysql.sock\t/run/mysql.pid\n",
        )
        .unwrap();
        assert_eq!(
            parsed.sql_mode,
            vec!["NO_ENGINE_SUBSTITUTION".to_string(), "STRICT_TRANS_TABLES".to_string()]
        );
        assert!(parsed.innodb_strict_mode);
    }
}
