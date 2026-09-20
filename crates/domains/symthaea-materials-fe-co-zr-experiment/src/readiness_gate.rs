// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Executable pre-acquisition readiness evidence for the Fe-Co-Zr retrospective run.
//!
//! This module deliberately proves only cheap runtime prerequisites. It does not
//! manufacture OQMD acquisition/import/database-state authority. The real archive
//! still has to pass the MAG-DATA-013/015/016 evidence chain after acquisition.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use symthaea_materials_fe_co_zr_experiment::{LocalExperimentConfig, PreparedExperiment};
use symthaea_materials_oqmd_extraction_executor::{QMPY_SOURCE_COMMIT, QMPY_VERSION};
use thiserror::Error;

pub const RUNTIME_READINESS_FILE: &str = "runtime-readiness.json";
const READINESS_SCHEMA_VERSION: u32 = 1;
const QMPY_PROBE_PROFILE: &str = "qmpy-formation-energy-import-probe-v1";
const MYSQL_FIXTURE_PROFILE: &str = "mysql-live-roundtrip-inventory-v1";
const QMPY_IMPORT_MARKER: &str = "QMPY_IMPORT_OK:qmpy.materials.formation_energy.FormationEnergy";

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactObservation {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeCapture {
    pub executable_path: String,
    pub args: Vec<String>,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub stdout_sha256: String,
    pub stderr_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QmpyImportProbeReceipt {
    pub profile: String,
    pub qmpy_version: String,
    pub qmpy_source_commit: String,
    pub python: ArtifactObservation,
    pub qmpy_artifact_manifest: ArtifactObservation,
    pub adapter_script: ArtifactObservation,
    pub probe_script_sha256: String,
    pub capture: ProbeCapture,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MysqlLiveObservation {
    pub runtime_version: String,
    pub version_comment: String,
    pub character_set_server: String,
    pub collation_server: String,
    pub sql_mode: Vec<String>,
    pub lower_case_table_names: u8,
    pub time_zone: String,
    pub max_allowed_packet_bytes: u64,
    pub innodb_strict_mode: bool,
    pub socket_path: String,
    pub pid_file: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MysqlFixtureReceipt {
    pub profile: String,
    pub mysql_server: ArtifactObservation,
    pub mysql_client: ArtifactObservation,
    pub server_pid: u32,
    pub live_server_executable_sha256: String,
    pub server_version_capture: ProbeCapture,
    pub client_version_capture: ProbeCapture,
    pub live_probe_capture: ProbeCapture,
    pub live_observation: MysqlLiveObservation,
    pub fixture_table: String,
    pub fixture_sql_sha256: String,
    pub fixture_capture: ProbeCapture,
    pub row_count: u64,
    pub inventory_columns: Vec<String>,
    pub payload_identity: String,
    pub post_drop_table_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReadinessGate {
    pub schema_version: u32,
    pub config_sha256: String,
    pub prepared_sha256: String,
    pub qmpy_import: QmpyImportProbeReceipt,
    pub mysql_fixture: MysqlFixtureReceipt,
    pub claim_ceiling: String,
}

impl RuntimeReadinessGate {
    pub fn validate_against(
        &self,
        config: &LocalExperimentConfig,
        prepared: &PreparedExperiment,
    ) -> Result<(), ReadinessError> {
        prepared
            .validate_against(config)
            .map_err(|error| ReadinessError::Upstream(error.to_string()))?;
        if self.schema_version != READINESS_SCHEMA_VERSION
            || self.config_sha256 != config.config_sha256().map_err(|e| ReadinessError::Upstream(e.to_string()))?
            || self.prepared_sha256 != prepared.prepared_sha256().map_err(|e| ReadinessError::Upstream(e.to_string()))?
            || self.claim_ceiling != "runtime-prerequisites-only-no-scientific-authority"
        {
            return Err(ReadinessError::GateIdentityMismatch);
        }
        self.qmpy_import.validate_against(config)?;
        self.mysql_fixture.validate_against(config, &self.prepared_sha256)?;
        Ok(())
    }

    pub fn gate_sha256(&self) -> Result<String, ReadinessError> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

impl QmpyImportProbeReceipt {
    fn validate_against(&self, config: &LocalExperimentConfig) -> Result<(), ReadinessError> {
        if self.profile != QMPY_PROBE_PROFILE
            || self.qmpy_version != QMPY_VERSION
            || self.qmpy_source_commit != QMPY_SOURCE_COMMIT
            || self.probe_script_sha256 != sha256_hex(QMPY_IMPORT_SCRIPT.as_bytes())
            || self.capture.exit_code != 0
            || !self.capture.stdout.lines().any(|line| line.trim() == QMPY_IMPORT_MARKER)
        {
            return Err(ReadinessError::QmpyProbeMismatch);
        }
        require_same_artifact(&self.python, &config.python_path)?;
        require_same_artifact(&self.qmpy_artifact_manifest, &config.qmpy_artifact_manifest_path)?;
        require_same_artifact(&self.adapter_script, &config.qmpy_adapter_script_path)?;
        self.capture.validate()?;
        if self.capture.executable_path != config.python_path {
            return Err(ReadinessError::QmpyProbeMismatch);
        }
        Ok(())
    }
}

impl MysqlFixtureReceipt {
    fn validate_against(
        &self,
        config: &LocalExperimentConfig,
        prepared_sha256: &str,
    ) -> Result<(), ReadinessError> {
        if self.profile != MYSQL_FIXTURE_PROFILE
            || self.fixture_table != fixture_table_name(prepared_sha256)
            || self.row_count != 2
            || self.inventory_columns != ["id:1".to_string(), "payload:2".to_string()]
            || self.payload_identity != "1:alpha,2:beta"
            || self.post_drop_table_count != 0
        {
            return Err(ReadinessError::MysqlFixtureMismatch);
        }
        require_same_artifact(&self.mysql_server, &config.mysql_server_path)?;
        require_same_artifact(&self.mysql_client, &config.mysql_client_path)?;
        self.server_version_capture.validate()?;
        self.client_version_capture.validate()?;
        self.live_probe_capture.validate()?;
        self.fixture_capture.validate()?;
        require_trimmed_stdout(&self.server_version_capture, &config.expected_server_version, "server version")?;
        require_trimmed_stdout(&self.client_version_capture, &config.expected_client_version, "client version")?;
        validate_live_observation(&self.live_observation, config)?;
        let current_pid = read_server_pid(&config.server_pid_file_path)?;
        if current_pid != self.server_pid {
            return Err(ReadinessError::ServerPidChanged { expected: self.server_pid, actual: current_pid });
        }
        let current_exe_sha = hash_live_server_executable(current_pid)?;
        if current_exe_sha != self.live_server_executable_sha256
            || current_exe_sha != self.mysql_server.sha256
        {
            return Err(ReadinessError::LiveServerExecutableChanged);
        }
        Ok(())
    }
}

impl ProbeCapture {
    fn validate(&self) -> Result<(), ReadinessError> {
        if self.exit_code != 0
            || self.stdout_sha256 != sha256_hex(self.stdout.as_bytes())
            || self.stderr_sha256 != sha256_hex(self.stderr.as_bytes())
        {
            return Err(ReadinessError::CaptureMismatch);
        }
        Ok(())
    }
}

pub fn execute_runtime_readiness(
    config: &LocalExperimentConfig,
    prepared: &PreparedExperiment,
) -> Result<RuntimeReadinessGate, ReadinessError> {
    prepared
        .validate_against(config)
        .map_err(|error| ReadinessError::Upstream(error.to_string()))?;
    let prepared_sha256 = prepared
        .prepared_sha256()
        .map_err(|error| ReadinessError::Upstream(error.to_string()))?;
    let qmpy_import = execute_qmpy_import_probe(config)?;
    let mysql_fixture = execute_mysql_fixture(config, &prepared_sha256)?;
    let gate = RuntimeReadinessGate {
        schema_version: READINESS_SCHEMA_VERSION,
        config_sha256: config
            .config_sha256()
            .map_err(|error| ReadinessError::Upstream(error.to_string()))?,
        prepared_sha256,
        qmpy_import,
        mysql_fixture,
        claim_ceiling: "runtime-prerequisites-only-no-scientific-authority".to_string(),
    };
    gate.validate_against(config, prepared)?;
    gate.gate_sha256()?;
    Ok(gate)
}

fn execute_qmpy_import_probe(config: &LocalExperimentConfig) -> Result<QmpyImportProbeReceipt, ReadinessError> {
    let python = observe_artifact(&config.python_path)?;
    let qmpy_artifact_manifest = observe_artifact(&config.qmpy_artifact_manifest_path)?;
    let adapter_script = observe_artifact(&config.qmpy_adapter_script_path)?;
    let args = vec!["-I".to_string(), "-c".to_string(), QMPY_IMPORT_SCRIPT.to_string()];
    let capture = run_capture(
        &config.python_path,
        &args,
        None,
        config.probe_timeout_ms,
        config.max_output_bytes,
        &[("qmdb_v1_1_pswd", ""), ("PYTHONDONTWRITEBYTECODE", "1"), ("PYTHONNOUSERSITE", "1")],
    )?;
    if capture.exit_code != 0 || !capture.stdout.lines().any(|line| line.trim() == QMPY_IMPORT_MARKER) {
        return Err(ReadinessError::QmpyImportFailed {
            stdout: capture.stdout,
            stderr: capture.stderr,
        });
    }
    let receipt = QmpyImportProbeReceipt {
        profile: QMPY_PROBE_PROFILE.to_string(),
        qmpy_version: QMPY_VERSION.to_string(),
        qmpy_source_commit: QMPY_SOURCE_COMMIT.to_string(),
        python,
        qmpy_artifact_manifest,
        adapter_script,
        probe_script_sha256: sha256_hex(QMPY_IMPORT_SCRIPT.as_bytes()),
        capture,
    };
    receipt.validate_against(config)?;
    Ok(receipt)
}

fn execute_mysql_fixture(
    config: &LocalExperimentConfig,
    prepared_sha256: &str,
) -> Result<MysqlFixtureReceipt, ReadinessError> {
    let mysql_server = observe_artifact(&config.mysql_server_path)?;
    let mysql_client = observe_artifact(&config.mysql_client_path)?;
    let version_args = vec!["--version".to_string()];
    let server_version_capture = run_capture(
        &config.mysql_server_path,
        &version_args,
        None,
        config.probe_timeout_ms,
        config.max_output_bytes,
        &[],
    )?;
    require_trimmed_stdout(&server_version_capture, &config.expected_server_version, "server version")?;
    let client_version_capture = run_capture(
        &config.mysql_client_path,
        &version_args,
        None,
        config.probe_timeout_ms,
        config.max_output_bytes,
        &[],
    )?;
    require_trimmed_stdout(&client_version_capture, &config.expected_client_version, "client version")?;

    let server_pid = read_server_pid(&config.server_pid_file_path)?;
    let live_server_executable_sha256 = hash_live_server_executable(server_pid)?;
    if live_server_executable_sha256 != mysql_server.sha256 {
        return Err(ReadinessError::LiveServerExecutableChanged);
    }

    let live_query = "SELECT @@version,@@version_comment,@@character_set_server,@@collation_server,@@sql_mode,@@lower_case_table_names,@@time_zone,@@max_allowed_packet,@@innodb_strict_mode,@@socket,@@pid_file";
    let live_probe_capture = run_mysql_query(config, live_query)?;
    let live_observation = parse_live_observation(&live_probe_capture.stdout)?;
    validate_live_observation(&live_observation, config)?;

    let fixture_table = fixture_table_name(prepared_sha256);
    let collision_query = format!(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name='{}'",
        fixture_table
    );
    let collision = run_mysql_query(config, &collision_query)?;
    let existing = collision.stdout.trim().parse::<u64>().map_err(|_| ReadinessError::UnexpectedMysqlOutput("fixture collision count".to_string()))?;
    if existing != 0 {
        return Err(ReadinessError::FixtureTableAlreadyExists(fixture_table));
    }

    let fixture_sql = format!(
        "CREATE TABLE `{table}` (id INT NOT NULL PRIMARY KEY,payload VARCHAR(16) NOT NULL) ENGINE=InnoDB;\n\
INSERT INTO `{table}` (id,payload) VALUES (1,'alpha'),(2,'beta');\n\
SELECT CONCAT('rows=',COUNT(*)) FROM `{table}`;\n\
SELECT CONCAT('columns=',GROUP_CONCAT(CONCAT(column_name,':',ordinal_position) ORDER BY ordinal_position SEPARATOR ',')) FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='{table}';\n\
SELECT CONCAT('payload=',GROUP_CONCAT(CONCAT(id,':',payload) ORDER BY id SEPARATOR ',')) FROM `{table}`;\n\
DROP TABLE `{table}`;\n\
SELECT CONCAT('post_drop=',COUNT(*)) FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name='{table}';\n",
        table = fixture_table
    );
    let fixture_capture = run_mysql_stdin(config, fixture_sql.as_bytes())?;
    let markers = parse_fixture_markers(&fixture_capture.stdout)?;
    if markers.row_count != 2
        || markers.inventory_columns != ["id:1".to_string(), "payload:2".to_string()]
        || markers.payload_identity != "1:alpha,2:beta"
        || markers.post_drop_table_count != 0
    {
        return Err(ReadinessError::MysqlFixtureMismatch);
    }

    let receipt = MysqlFixtureReceipt {
        profile: MYSQL_FIXTURE_PROFILE.to_string(),
        mysql_server,
        mysql_client,
        server_pid,
        live_server_executable_sha256,
        server_version_capture,
        client_version_capture,
        live_probe_capture,
        live_observation,
        fixture_table,
        fixture_sql_sha256: sha256_hex(fixture_sql.as_bytes()),
        fixture_capture,
        row_count: markers.row_count,
        inventory_columns: markers.inventory_columns,
        payload_identity: markers.payload_identity,
        post_drop_table_count: markers.post_drop_table_count,
    };
    receipt.validate_against(config, prepared_sha256)?;
    Ok(receipt)
}

fn run_mysql_query(config: &LocalExperimentConfig, query: &str) -> Result<ProbeCapture, ReadinessError> {
    let mut args = mysql_common_args(config);
    args.push(format!("--execute={query}"));
    args.push(config.database_name.clone());
    let capture = run_capture(
        &config.mysql_client_path,
        &args,
        None,
        config.probe_timeout_ms,
        config.max_output_bytes,
        &[],
    )?;
    require_success(&capture, "mysql query")?;
    Ok(capture)
}

fn run_mysql_stdin(config: &LocalExperimentConfig, stdin: &[u8]) -> Result<ProbeCapture, ReadinessError> {
    let mut args = mysql_common_args(config);
    args.push(config.database_name.clone());
    let capture = run_capture(
        &config.mysql_client_path,
        &args,
        Some(stdin),
        config.probe_timeout_ms,
        config.max_output_bytes,
        &[],
    )?;
    require_success(&capture, "mysql fixture")?;
    Ok(capture)
}

fn mysql_common_args(config: &LocalExperimentConfig) -> Vec<String> {
    vec![
        "--no-defaults".to_string(),
        "--protocol=SOCKET".to_string(),
        format!("--socket={}", config.unix_socket_path),
        format!("--user={}", config.mysql_user),
        "--batch".to_string(),
        "--raw".to_string(),
        "--skip-column-names".to_string(),
        format!("--default-character-set={}", config.character_set_server),
    ]
}

fn parse_live_observation(stdout: &str) -> Result<MysqlLiveObservation, ReadinessError> {
    let line = stdout.trim();
    let columns: Vec<&str> = line.split('\t').collect();
    if columns.len() != 11 {
        return Err(ReadinessError::UnexpectedMysqlOutput(format!("live probe expected 11 columns, got {}", columns.len())));
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
        lower_case_table_names: columns[5].parse().map_err(|_| ReadinessError::UnexpectedMysqlOutput("lower_case_table_names".to_string()))?,
        time_zone: columns[6].to_string(),
        max_allowed_packet_bytes: columns[7].parse().map_err(|_| ReadinessError::UnexpectedMysqlOutput("max_allowed_packet".to_string()))?,
        innodb_strict_mode: matches!(columns[8], "1" | "ON" | "on"),
        socket_path: columns[9].to_string(),
        pid_file: columns[10].to_string(),
    })
}

fn validate_live_observation(
    observation: &MysqlLiveObservation,
    config: &LocalExperimentConfig,
) -> Result<(), ReadinessError> {
    if observation.character_set_server != config.character_set_server
        || observation.collation_server != config.collation_server
        || observation.sql_mode != config.sql_mode
        || observation.lower_case_table_names != config.lower_case_table_names
        || observation.time_zone != config.time_zone
        || observation.max_allowed_packet_bytes != config.max_allowed_packet_bytes
        || observation.innodb_strict_mode != config.innodb_strict_mode
        || observation.socket_path != config.unix_socket_path
        || observation.pid_file != config.server_pid_file_path
    {
        return Err(ReadinessError::LiveDatabasePolicyMismatch);
    }
    Ok(())
}

struct FixtureMarkers {
    row_count: u64,
    inventory_columns: Vec<String>,
    payload_identity: String,
    post_drop_table_count: u64,
}

fn parse_fixture_markers(stdout: &str) -> Result<FixtureMarkers, ReadinessError> {
    let mut rows = None;
    let mut columns = None;
    let mut payload = None;
    let mut post_drop = None;
    for line in stdout.lines().map(str::trim) {
        if let Some(value) = line.strip_prefix("rows=") {
            rows = Some(value.parse::<u64>().map_err(|_| ReadinessError::UnexpectedMysqlOutput("rows".to_string()))?);
        } else if let Some(value) = line.strip_prefix("columns=") {
            columns = Some(value.split(',').map(str::to_string).collect::<Vec<_>>());
        } else if let Some(value) = line.strip_prefix("payload=") {
            payload = Some(value.to_string());
        } else if let Some(value) = line.strip_prefix("post_drop=") {
            post_drop = Some(value.parse::<u64>().map_err(|_| ReadinessError::UnexpectedMysqlOutput("post_drop".to_string()))?);
        }
    }
    Ok(FixtureMarkers {
        row_count: rows.ok_or_else(|| ReadinessError::UnexpectedMysqlOutput("missing rows marker".to_string()))?,
        inventory_columns: columns.ok_or_else(|| ReadinessError::UnexpectedMysqlOutput("missing columns marker".to_string()))?,
        payload_identity: payload.ok_or_else(|| ReadinessError::UnexpectedMysqlOutput("missing payload marker".to_string()))?,
        post_drop_table_count: post_drop.ok_or_else(|| ReadinessError::UnexpectedMysqlOutput("missing post_drop marker".to_string()))?,
    })
}

fn run_capture(
    executable: &str,
    args: &[String],
    stdin: Option<&[u8]>,
    timeout_ms: u64,
    max_output_bytes: usize,
    extra_environment: &[(&str, &str)],
) -> Result<ProbeCapture, ReadinessError> {
    let mut command = Command::new(executable);
    command.args(args).env_clear();
    command.env("LC_ALL", "C").env("LANG", "C").env("TZ", "UTC");
    for (key, value) in extra_environment {
        command.env(key, value);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    if stdin.is_some() {
        command.stdin(Stdio::piped());
    } else {
        command.stdin(Stdio::null());
    }
    let mut child = command.spawn().map_err(|source| ReadinessError::Io {
        context: format!("spawn {executable}"),
        source,
    })?;
    if let Some(input) = stdin {
        let mut child_stdin = child.stdin.take().ok_or(ReadinessError::MissingChildStdin)?;
        child_stdin.write_all(input).map_err(|source| ReadinessError::Io {
            context: format!("write stdin for {executable}"),
            source,
        })?;
    }
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|source| ReadinessError::Io {
            context: format!("wait for {executable}"),
            source,
        })? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ReadinessError::ProcessTimedOut(executable.to_string()));
        }
        thread::sleep(Duration::from_millis(10));
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_end(&mut stdout).map_err(|source| ReadinessError::Io {
            context: format!("read stdout for {executable}"),
            source,
        })?;
    }
    if let Some(mut pipe) = child.stderr.take() {
        pipe.read_to_end(&mut stderr).map_err(|source| ReadinessError::Io {
            context: format!("read stderr for {executable}"),
            source,
        })?;
    }
    if stdout.len() > max_output_bytes || stderr.len() > max_output_bytes {
        return Err(ReadinessError::DiagnosticLimitExceeded);
    }
    let stdout = String::from_utf8(stdout).map_err(|_| ReadinessError::NonUtf8ProcessOutput)?;
    let stderr = String::from_utf8(stderr).map_err(|_| ReadinessError::NonUtf8ProcessOutput)?;
    Ok(ProbeCapture {
        executable_path: executable.to_string(),
        args: args.to_vec(),
        exit_code: status.code().unwrap_or(-1),
        stdout_sha256: sha256_hex(stdout.as_bytes()),
        stderr_sha256: sha256_hex(stderr.as_bytes()),
        stdout,
        stderr,
    })
}

fn require_success(capture: &ProbeCapture, label: &str) -> Result<(), ReadinessError> {
    if capture.exit_code != 0 {
        return Err(ReadinessError::ProcessFailed {
            label: label.to_string(),
            exit_code: capture.exit_code,
            stdout: capture.stdout.clone(),
            stderr: capture.stderr.clone(),
        });
    }
    capture.validate()
}

fn require_trimmed_stdout(
    capture: &ProbeCapture,
    expected: &str,
    label: &str,
) -> Result<(), ReadinessError> {
    require_success(capture, label)?;
    if capture.stdout.trim_end() != expected.trim_end() {
        return Err(ReadinessError::VersionMismatch(label.to_string()));
    }
    Ok(())
}

fn fixture_table_name(prepared_sha256: &str) -> String {
    let prefix = prepared_sha256.get(..16).unwrap_or(prepared_sha256);
    format!("__symthaea_mag_readiness_{prefix}")
}

fn observe_artifact(path: &str) -> Result<ArtifactObservation, ReadinessError> {
    if !Path::new(path).is_absolute() {
        return Err(ReadinessError::PathNotAbsolute(path.to_string()));
    }
    Ok(ArtifactObservation {
        path: path.to_string(),
        sha256: hash_file(Path::new(path))?,
    })
}

fn require_same_artifact(observed: &ArtifactObservation, expected_path: &str) -> Result<(), ReadinessError> {
    if observed.path != expected_path || observed.sha256 != hash_file(Path::new(expected_path))? {
        return Err(ReadinessError::ArtifactChanged(expected_path.to_string()));
    }
    Ok(())
}

fn read_server_pid(path: &str) -> Result<u32, ReadinessError> {
    let mut text = String::new();
    File::open(path)
        .and_then(|mut file| file.read_to_string(&mut text))
        .map_err(|source| ReadinessError::Io { context: format!("read server pid file {path}"), source })?;
    let pid = text.trim().parse::<u32>().map_err(|_| ReadinessError::InvalidServerPidFile(path.to_string()))?;
    if pid == 0 {
        return Err(ReadinessError::InvalidServerPidFile(path.to_string()));
    }
    Ok(pid)
}

#[cfg(target_os = "linux")]
fn hash_live_server_executable(pid: u32) -> Result<String, ReadinessError> {
    hash_file(Path::new(&format!("/proc/{pid}/exe")))
}

#[cfg(not(target_os = "linux"))]
fn hash_live_server_executable(_pid: u32) -> Result<String, ReadinessError> {
    Err(ReadinessError::UnsupportedPlatform)
}

fn hash_file(path: &Path) -> Result<String, ReadinessError> {
    let mut file = File::open(path).map_err(|source| ReadinessError::Io {
        context: format!("open {}", path.display()),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|source| ReadinessError::Io {
            context: format!("read {}", path.display()),
            source,
        })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Error)]
pub enum ReadinessError {
    #[error("upstream contract rejected readiness input: {0}")]
    Upstream(String),
    #[error("runtime readiness identity does not match the current config/prepared subject")]
    GateIdentityMismatch,
    #[error("qmpy import receipt no longer matches the current runtime")]
    QmpyProbeMismatch,
    #[error("qmpy import failed; stdout={stdout:?}; stderr={stderr:?}")]
    QmpyImportFailed { stdout: String, stderr: String },
    #[error("MySQL live fixture receipt is malformed or inconsistent")]
    MysqlFixtureMismatch,
    #[error("stored process capture is malformed or did not exit successfully")]
    CaptureMismatch,
    #[error("runtime artifact changed: {0}")]
    ArtifactChanged(String),
    #[error("path is not absolute: {0}")]
    PathNotAbsolute(String),
    #[error("server PID changed after readiness evidence: expected {expected}, found {actual}")]
    ServerPidChanged { expected: u32, actual: u32 },
    #[error("live server executable is not the exact configured mysqld artifact")]
    LiveServerExecutableChanged,
    #[error("live database policy differs from the frozen experiment policy")]
    LiveDatabasePolicyMismatch,
    #[error("fixture table already exists before readiness probe: {0}")]
    FixtureTableAlreadyExists(String),
    #[error("unexpected MySQL probe output: {0}")]
    UnexpectedMysqlOutput(String),
    #[error("process failed for {label} (exit {exit_code}); stdout={stdout:?}; stderr={stderr:?}")]
    ProcessFailed { label: String, exit_code: i32, stdout: String, stderr: String },
    #[error("process timed out: {0}")]
    ProcessTimedOut(String),
    #[error("process output exceeded configured diagnostic retention bound")]
    DiagnosticLimitExceeded,
    #[error("process output was not valid UTF-8")]
    NonUtf8ProcessOutput,
    #[error("child stdin was unavailable")]
    MissingChildStdin,
    #[error("version output differs from frozen value: {0}")]
    VersionMismatch(String),
    #[error("invalid server PID file: {0}")]
    InvalidServerPidFile(String),
    #[error("runtime readiness live-server binding currently requires Linux /proc")]
    UnsupportedPlatform,
    #[error("I/O while {context}: {source}")]
    Io { context: String, #[source] source: std::io::Error },
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_table_is_subject_bound_and_bounded() {
        let a = fixture_table_name(&"a".repeat(64));
        let b = fixture_table_name(&"b".repeat(64));
        assert_ne!(a, b);
        assert!(a.len() < 64);
        assert!(a.ends_with("aaaaaaaaaaaaaaaa"));
    }

    #[test]
    fn fixture_markers_require_all_four_observations() {
        let parsed = parse_fixture_markers(
            "rows=2\ncolumns=id:1,payload:2\npayload=1:alpha,2:beta\npost_drop=0\n",
        )
        .unwrap();
        assert_eq!(parsed.row_count, 2);
        assert_eq!(parsed.inventory_columns, ["id:1", "payload:2"]);
        assert_eq!(parsed.payload_identity, "1:alpha,2:beta");
        assert_eq!(parsed.post_drop_table_count, 0);
        assert!(parse_fixture_markers("rows=2\n").is_err());
    }

    #[test]
    fn live_probe_parser_canonicalizes_sql_modes() {
        let parsed = parse_live_observation(
            "8.4.0\tPercona\tutf8mb4\tutf8mb4_bin\tSTRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION\t0\t+00:00\t67108864\tON\t/run/mysql.sock\t/run/mysql.pid\n",
        )
        .unwrap();
        assert_eq!(parsed.sql_mode, ["NO_ENGINE_SUBSTITUTION", "STRICT_TRANS_TABLES"]);
        assert!(parsed.innodb_strict_mode);
    }
}
