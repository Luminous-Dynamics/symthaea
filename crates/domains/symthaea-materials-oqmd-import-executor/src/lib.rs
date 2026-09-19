// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing OQMD decompression and direct MySQL import execution.
//!
//! This crate establishes process/input/server facts only. Successful mysql exit is
//! not database-state authority; a later canonical schema/row inventory must bind the
//! resulting database before MAG-DATA-008 may issue a successful import receipt.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use symthaea_materials_historical_extraction::OqmdExtractionProtocol;
use symthaea_materials_historical_import::HistoricalImportProfile;
use symthaea_materials_snapshot_acquisition::HistoricalSnapshotAcquisitionReceipt;
use symthaea_process_capture::{EnvironmentPolicy, ProcessCapture, ProcessSpec, capture_process};
use symthaea_process_file_io::{
    BoundFileIoProcessCapture, BoundFileIoProcessRequest, FileIoLauncherArtifact, NewStdoutFile,
    capture_process_with_file_io,
};
use symthaea_process_stdin::{
    BoundStdinProcessCapture, BoundStdinProcessRequest, ContentAddressedStdinFile,
    StdinLauncherArtifact, capture_process_with_stdin_file,
};
use thiserror::Error;

const MAX_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;
const MAX_DIAGNOSTIC_BYTES: usize = 64 * 1024 * 1024;

/// Exact executable or local evidence artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeArtifact {
    /// Absolute UTF-8 path.
    pub path: String,
    /// Exact SHA-256 of file bytes.
    pub sha256: String,
}

impl RuntimeArtifact {
    /// Observe one local artifact.
    pub fn observe(path: &Path) -> Result<Self, OqmdImportExecutionError> {
        validate_absolute_path(path, "runtime_artifact")?;
        let path_text = path.to_str().ok_or(OqmdImportExecutionError::NonUtf8Path)?;
        let (sha256, _) = hash_file(path)?;
        Ok(Self { path: path_text.to_string(), sha256 })
    }

    fn revalidate(&self) -> Result<(), OqmdImportExecutionError> {
        validate_absolute_text_path(&self.path, "runtime_artifact")?;
        validate_sha256(&self.sha256)?;
        let (actual, _) = hash_file(Path::new(&self.path))?;
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(OqmdImportExecutionError::RuntimeArtifactDigestMismatch(self.path.clone()));
        }
        Ok(())
    }
}

/// Frozen local plan for archive decompression and import into an isolated live server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OqmdLocalImportPlan {
    /// Plan schema version.
    pub schema_version: u32,
    /// Frozen historical protocol identity.
    pub protocol_sha256: String,
    /// Frozen MAG-DATA-008 import-profile identity.
    pub import_profile_sha256: String,
    /// Frozen MAG-DATA-006 acquisition-receipt identity.
    pub acquisition_receipt_sha256: String,
    /// Absolute compressed OQMD archive path.
    pub compressed_snapshot_path: String,
    /// Absolute create-new decompressed SQL path.
    pub decompressed_sql_path: String,
    /// Exact gzip-compatible decompressor.
    pub decompressor: RuntimeArtifact,
    /// Exact SIM-PROC-004 launcher.
    pub file_io_launcher: FileIoLauncherArtifact,
    /// Exact mysql client.
    pub mysql_client: RuntimeArtifact,
    /// Exact SIM-PROC-003 launcher.
    pub stdin_launcher: StdinLauncherArtifact,
    /// Exact mysqld/server binary selected for the isolated instance.
    pub mysql_server: RuntimeArtifact,
    /// Expected live server PID.
    pub server_pid: u32,
    /// Exact Unix socket path.
    pub unix_socket_path: String,
    /// Destination database/schema.
    pub database_name: String,
    /// Local isolated MySQL user; passwords are intentionally unsupported here.
    pub mysql_user: String,
    /// Exact Nix/container environment manifest.
    pub execution_environment_manifest: RuntimeArtifact,
    /// Decompression wall-clock bound.
    pub decompression_timeout_ms: u64,
    /// Server-probe wall-clock bound.
    pub probe_timeout_ms: u64,
    /// Import wall-clock bound.
    pub import_timeout_ms: u64,
    /// Per-stream diagnostic retention bound.
    pub max_output_bytes: usize,
}

impl OqmdLocalImportPlan {
    /// Validate the plan against protocol/profile/acquisition contracts.
    pub fn validate_contract(
        &self,
        protocol: &OqmdExtractionProtocol,
        profile: &HistoricalImportProfile,
        acquisition: &HistoricalSnapshotAcquisitionReceipt,
    ) -> Result<(), OqmdImportExecutionError> {
        protocol.validate().map_err(|e| OqmdImportExecutionError::Protocol(e.to_string()))?;
        profile.validate_against(protocol).map_err(|e| OqmdImportExecutionError::ImportProfile(e.to_string()))?;
        acquisition.validate_against(protocol).map_err(|e| OqmdImportExecutionError::Acquisition(e.to_string()))?;
        if self.schema_version != 1 {
            return Err(OqmdImportExecutionError::UnsupportedPlanSchema(self.schema_version));
        }
        let protocol_sha = protocol.protocol_sha256().map_err(|e| OqmdImportExecutionError::Protocol(e.to_string()))?;
        if !self.protocol_sha256.eq_ignore_ascii_case(&protocol_sha) {
            return Err(OqmdImportExecutionError::ProtocolDigestMismatch);
        }
        let profile_sha = profile.profile_sha256(protocol).map_err(|e| OqmdImportExecutionError::ImportProfile(e.to_string()))?;
        if !self.import_profile_sha256.eq_ignore_ascii_case(&profile_sha) {
            return Err(OqmdImportExecutionError::ImportProfileDigestMismatch);
        }
        let acquisition_sha = acquisition.receipt_sha256(protocol).map_err(|e| OqmdImportExecutionError::Acquisition(e.to_string()))?;
        if !self.acquisition_receipt_sha256.eq_ignore_ascii_case(&acquisition_sha) {
            return Err(OqmdImportExecutionError::AcquisitionReceiptDigestMismatch);
        }
        if !self.decompressor.sha256.eq_ignore_ascii_case(&profile.decompressor_artifact_sha256) {
            return Err(OqmdImportExecutionError::DecompressorDigestMismatch);
        }
        if !self.mysql_client.sha256.eq_ignore_ascii_case(&profile.client_artifact_sha256) {
            return Err(OqmdImportExecutionError::MysqlClientDigestMismatch);
        }
        if !self.mysql_server.sha256.eq_ignore_ascii_case(&profile.server_artifact_sha256) {
            return Err(OqmdImportExecutionError::MysqlServerDigestMismatch);
        }
        if !self.execution_environment_manifest.sha256.eq_ignore_ascii_case(&profile.import_environment_manifest_sha256) {
            return Err(OqmdImportExecutionError::EnvironmentDigestMismatch);
        }
        validate_absolute_text_path(&self.compressed_snapshot_path, "compressed_snapshot_path")?;
        validate_absolute_text_path(&self.decompressed_sql_path, "decompressed_sql_path")?;
        validate_absolute_text_path(&self.unix_socket_path, "unix_socket_path")?;
        if self.compressed_snapshot_path == self.decompressed_sql_path {
            return Err(OqmdImportExecutionError::InputOutputPathAlias);
        }
        validate_identifier(&self.database_name, "database_name")?;
        validate_identifier(&self.mysql_user, "mysql_user")?;
        if self.server_pid == 0 {
            return Err(OqmdImportExecutionError::InvalidServerPid);
        }
        for timeout in [self.decompression_timeout_ms, self.probe_timeout_ms, self.import_timeout_ms] {
            if timeout == 0 || timeout > MAX_TIMEOUT_MS {
                return Err(OqmdImportExecutionError::InvalidTimeout(timeout));
            }
        }
        if self.max_output_bytes == 0 || self.max_output_bytes > MAX_DIAGNOSTIC_BYTES {
            return Err(OqmdImportExecutionError::InvalidOutputLimit(self.max_output_bytes));
        }
        let command_sha = self.mysql_command_contract(profile)?.command_sha256()?;
        if !command_sha.eq_ignore_ascii_case(&profile.import_command_sha256) {
            return Err(OqmdImportExecutionError::ImportCommandDigestMismatch);
        }
        Ok(())
    }

    /// Rehash every local runtime artifact before side effects.
    pub fn validate_local_artifacts(&self) -> Result<(), OqmdImportExecutionError> {
        self.decompressor.revalidate()?;
        self.mysql_client.revalidate()?;
        self.mysql_server.revalidate()?;
        self.execution_environment_manifest.revalidate()?;
        Ok(())
    }

    /// Deterministic plan identity.
    pub fn plan_sha256(
        &self,
        protocol: &OqmdExtractionProtocol,
        profile: &HistoricalImportProfile,
        acquisition: &HistoricalSnapshotAcquisitionReceipt,
    ) -> Result<String, OqmdImportExecutionError> {
        self.validate_contract(protocol, profile, acquisition)?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// Derive exact mysql import semantics bound by MAG-DATA-008.
    pub fn mysql_command_contract(
        &self,
        profile: &HistoricalImportProfile,
    ) -> Result<MysqlImportCommandContract, OqmdImportExecutionError> {
        let args = vec![
            "--no-defaults".to_string(),
            "--protocol=SOCKET".to_string(),
            format!("--socket={}", self.unix_socket_path),
            format!("--user={}", self.mysql_user),
            "--batch".to_string(),
            "--raw".to_string(),
            "--skip-column-names".to_string(),
            "--binary-mode".to_string(),
            format!("--default-character-set={}", profile.character_set_server),
            format!("--max_allowed_packet={}", profile.max_allowed_packet_bytes),
            self.database_name.clone(),
        ];
        Ok(MysqlImportCommandContract {
            schema_version: 1,
            client_artifact_sha256: self.mysql_client.sha256.to_ascii_lowercase(),
            client_path: self.mysql_client.path.clone(),
            args,
            environment: deterministic_environment(),
            environment_policy: EnvironmentPolicy::ClearAndSet,
        })
    }
}

/// Exact mysql command semantics, excluding timeout/output policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MysqlImportCommandContract {
    /// Contract schema version.
    pub schema_version: u32,
    /// Exact mysql client artifact.
    pub client_artifact_sha256: String,
    /// Exact client path.
    pub client_path: String,
    /// Ordered arguments.
    pub args: Vec<String>,
    /// Closed deterministic environment.
    pub environment: BTreeMap<String, String>,
    /// Environment inheritance policy.
    pub environment_policy: EnvironmentPolicy,
}

impl MysqlImportCommandContract {
    /// Deterministic command artifact identity.
    pub fn command_sha256(&self) -> Result<String, OqmdImportExecutionError> {
        if self.schema_version != 1 {
            return Err(OqmdImportExecutionError::UnsupportedCommandSchema(self.schema_version));
        }
        validate_sha256(&self.client_artifact_sha256)?;
        validate_absolute_text_path(&self.client_path, "mysql_client_path")?;
        if self.args.first().map(String::as_str) != Some("--no-defaults") {
            return Err(OqmdImportExecutionError::MysqlDefaultsNotDisabled);
        }
        if self.environment_policy != EnvironmentPolicy::ClearAndSet {
            return Err(OqmdImportExecutionError::MysqlEnvironmentNotClosed);
        }
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Exact archive-to-SQL process evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OqmdDecompressionEvidence {
    /// Plan identity.
    pub plan_sha256: String,
    /// Acquisition receipt identity.
    pub acquisition_receipt_sha256: String,
    /// File-bound decompression capture.
    pub process: BoundFileIoProcessCapture,
}

impl OqmdDecompressionEvidence {
    /// Whether a successful non-empty SQL artifact exists.
    pub fn complete_sql(&self) -> bool {
        self.process.complete_output()
            && self.process.observed_stdout.as_ref().is_some_and(|output| output.bytes > 0)
    }

    /// Deterministic evidence identity.
    pub fn evidence_sha256(&self) -> Result<String, OqmdImportExecutionError> {
        validate_sha256(&self.plan_sha256)?;
        validate_sha256(&self.acquisition_receipt_sha256)?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Live isolated MySQL server observation captured before import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MysqlServerObservation {
    /// Runtime server version.
    pub runtime_version: String,
    /// Runtime version comment.
    pub version_comment: String,
    /// Server character set.
    pub character_set_server: String,
    /// Server collation.
    pub collation_server: String,
    /// Canonically sorted SQL-mode entries.
    pub sql_mode: Vec<String>,
    /// lower_case_table_names.
    pub lower_case_table_names: u8,
    /// Runtime timezone.
    pub time_zone: String,
    /// Runtime max packet.
    pub max_allowed_packet_bytes: u64,
    /// InnoDB strict mode.
    pub innodb_strict_mode: bool,
    /// Server-reported socket path.
    pub socket_path: String,
    /// Server-reported PID-file path.
    pub pid_file: String,
}

/// Evidence binding selected binaries to the live isolated server configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MysqlServerPreflightEvidence {
    /// Plan identity.
    pub plan_sha256: String,
    /// Exact server-binary `--version` capture.
    pub server_version_capture: ProcessCapture,
    /// Exact client-binary `--version` capture.
    pub client_version_capture: ProcessCapture,
    /// Live server-query capture.
    pub server_probe_capture: ProcessCapture,
    /// Parsed live server state.
    pub observation: MysqlServerObservation,
    /// PID read from server-reported PID file.
    pub observed_server_pid: u32,
    /// SHA-256 of `/proc/<pid>/exe` bytes.
    pub observed_server_executable_sha256: String,
}

impl MysqlServerPreflightEvidence {
    /// Deterministic preflight identity.
    pub fn evidence_sha256(&self) -> Result<String, OqmdImportExecutionError> {
        validate_sha256(&self.plan_sha256)?;
        validate_sha256(&self.observed_server_executable_sha256)?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Evidence from feeding exact SQL bytes to the exact mysql client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OqmdMysqlImportEvidence {
    /// Plan identity.
    pub plan_sha256: String,
    /// Decompression evidence identity.
    pub decompression_evidence_sha256: String,
    /// Server preflight evidence identity.
    pub server_preflight_evidence_sha256: String,
    /// Content-addressed stdin process capture.
    pub process: BoundStdinProcessCapture,
}

impl OqmdMysqlImportEvidence {
    /// Whether the mysql client exited successfully. This is not DB-state authority.
    pub fn process_success(&self) -> bool {
        self.process.process_success()
    }

    /// Deterministic process-evidence identity.
    pub fn evidence_sha256(&self) -> Result<String, OqmdImportExecutionError> {
        validate_sha256(&self.plan_sha256)?;
        validate_sha256(&self.decompression_evidence_sha256)?;
        validate_sha256(&self.server_preflight_evidence_sha256)?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// Deterministic import diagnostic-log identity.
    pub fn import_log_sha256(&self) -> Result<String, OqmdImportExecutionError> {
        #[derive(Serialize)]
        struct Log<'a> {
            termination: &'a symthaea_process_capture::ProcessTermination,
            stdout: &'a [u8],
            stderr: &'a [u8],
            stdout_truncated: bool,
            stderr_truncated: bool,
        }
        let capture = &self.process.process_capture;
        Ok(sha256_hex(&serde_json::to_vec(&Log {
            termination: &capture.termination,
            stdout: &capture.stdout,
            stderr: &capture.stderr,
            stdout_truncated: capture.stdout_truncated,
            stderr_truncated: capture.stderr_truncated,
        })?))
    }
}

/// Execute deterministic gzip decompression into a create-new SQL artifact.
pub fn execute_decompression(
    protocol: &OqmdExtractionProtocol,
    profile: &HistoricalImportProfile,
    acquisition: &HistoricalSnapshotAcquisitionReceipt,
    plan: &OqmdLocalImportPlan,
) -> Result<OqmdDecompressionEvidence, OqmdImportExecutionError> {
    plan.validate_contract(protocol, profile, acquisition)?;
    plan.validate_local_artifacts()?;
    let request = BoundFileIoProcessRequest {
        launcher: plan.file_io_launcher.clone(),
        stdin: ContentAddressedStdinFile {
            path: plan.compressed_snapshot_path.clone(),
            sha256: acquisition.compressed_snapshot_sha256.clone(),
            bytes: acquisition.compressed_snapshot_bytes,
        },
        stdout: NewStdoutFile::new(Path::new(&plan.decompressed_sql_path))
            .map_err(|e| OqmdImportExecutionError::FileIo(e.to_string()))?,
        target: ProcessSpec {
            command: plan.decompressor.path.clone(),
            args: vec!["--decompress".to_string(), "--stdout".to_string()],
            environment: deterministic_environment(),
            environment_policy: EnvironmentPolicy::ClearAndSet,
            timeout_ms: plan.decompression_timeout_ms,
            max_output_bytes: plan.max_output_bytes,
        },
    };
    let process = capture_process_with_file_io(&request)
        .map_err(|e| OqmdImportExecutionError::FileIo(e.to_string()))?;
    let evidence = OqmdDecompressionEvidence {
        plan_sha256: plan.plan_sha256(protocol, profile, acquisition)?,
        acquisition_receipt_sha256: acquisition.receipt_sha256(protocol)
            .map_err(|e| OqmdImportExecutionError::Acquisition(e.to_string()))?,
        process,
    };
    evidence.evidence_sha256()?;
    Ok(evidence)
}

/// Validate exact binaries and live isolated server state before import.
pub fn execute_server_preflight(
    protocol: &OqmdExtractionProtocol,
    profile: &HistoricalImportProfile,
    acquisition: &HistoricalSnapshotAcquisitionReceipt,
    plan: &OqmdLocalImportPlan,
) -> Result<MysqlServerPreflightEvidence, OqmdImportExecutionError> {
    plan.validate_contract(protocol, profile, acquisition)?;
    plan.validate_local_artifacts()?;
    let plan_sha = plan.plan_sha256(protocol, profile, acquisition)?;

    let server_version_capture = capture_process(&version_spec(&plan.mysql_server.path, plan))
        .map_err(|e| OqmdImportExecutionError::Process(e.to_string()))?;
    require_success(&server_version_capture, "mysqld --version")?;
    require_exact_trimmed_stdout(&server_version_capture, &profile.server_version, "server_version")?;

    let client_version_capture = capture_process(&version_spec(&plan.mysql_client.path, plan))
        .map_err(|e| OqmdImportExecutionError::Process(e.to_string()))?;
    require_success(&client_version_capture, "mysql --version")?;
    require_exact_trimmed_stdout(&client_version_capture, &profile.client_version, "client_version")?;

    let server_probe_capture = capture_process(&server_probe_spec(plan)?)
        .map_err(|e| OqmdImportExecutionError::Process(e.to_string()))?;
    require_success(&server_probe_capture, "mysql server probe")?;
    let observation = parse_server_observation(&server_probe_capture.stdout)?;
    validate_server_observation(&observation, profile, plan)?;

    let observed_server_pid = read_pid_file(Path::new(&observation.pid_file))?;
    if observed_server_pid != plan.server_pid {
        return Err(OqmdImportExecutionError::ServerPidMismatch { expected: plan.server_pid, observed: observed_server_pid });
    }
    let observed_server_executable_sha256 = hash_running_executable(plan.server_pid)?;
    if !observed_server_executable_sha256.eq_ignore_ascii_case(&plan.mysql_server.sha256) {
        return Err(OqmdImportExecutionError::RunningServerDigestMismatch);
    }

    let evidence = MysqlServerPreflightEvidence {
        plan_sha256: plan_sha,
        server_version_capture,
        client_version_capture,
        server_probe_capture,
        observation,
        observed_server_pid,
        observed_server_executable_sha256,
    };
    evidence.evidence_sha256()?;
    Ok(evidence)
}

/// Feed exact decompressed SQL bytes into the exact mysql client.
pub fn execute_mysql_import(
    protocol: &OqmdExtractionProtocol,
    profile: &HistoricalImportProfile,
    acquisition: &HistoricalSnapshotAcquisitionReceipt,
    plan: &OqmdLocalImportPlan,
    decompression: &OqmdDecompressionEvidence,
    preflight: &MysqlServerPreflightEvidence,
) -> Result<OqmdMysqlImportEvidence, OqmdImportExecutionError> {
    plan.validate_contract(protocol, profile, acquisition)?;
    plan.validate_local_artifacts()?;
    let plan_sha = plan.plan_sha256(protocol, profile, acquisition)?;
    if !decompression.plan_sha256.eq_ignore_ascii_case(&plan_sha) || !decompression.complete_sql() {
        return Err(OqmdImportExecutionError::DecompressionNotQualified);
    }
    if !preflight.plan_sha256.eq_ignore_ascii_case(&plan_sha) {
        return Err(OqmdImportExecutionError::PreflightPlanMismatch);
    }
    let output = decompression.process.observed_stdout.as_ref()
        .ok_or(OqmdImportExecutionError::DecompressionNotQualified)?;
    if output.path != plan.decompressed_sql_path || output.bytes == 0 {
        return Err(OqmdImportExecutionError::DecompressedSqlIdentityMismatch);
    }
    let command = plan.mysql_command_contract(profile)?;
    let request = BoundStdinProcessRequest {
        launcher: plan.stdin_launcher.clone(),
        stdin: ContentAddressedStdinFile {
            path: output.path.clone(),
            sha256: output.sha256.clone(),
            bytes: output.bytes,
        },
        target: ProcessSpec {
            command: command.client_path,
            args: command.args,
            environment: command.environment,
            environment_policy: command.environment_policy,
            timeout_ms: plan.import_timeout_ms,
            max_output_bytes: plan.max_output_bytes,
        },
    };
    let process = capture_process_with_stdin_file(&request)
        .map_err(|e| OqmdImportExecutionError::Stdin(e.to_string()))?;
    let evidence = OqmdMysqlImportEvidence {
        plan_sha256: plan_sha,
        decompression_evidence_sha256: decompression.evidence_sha256()?,
        server_preflight_evidence_sha256: preflight.evidence_sha256()?,
        process,
    };
    evidence.evidence_sha256()?;
    Ok(evidence)
}

fn version_spec(path: &str, plan: &OqmdLocalImportPlan) -> ProcessSpec {
    ProcessSpec {
        command: path.to_string(),
        args: vec!["--version".to_string()],
        environment: deterministic_environment(),
        environment_policy: EnvironmentPolicy::ClearAndSet,
        timeout_ms: plan.probe_timeout_ms,
        max_output_bytes: plan.max_output_bytes,
    }
}

fn server_probe_spec(plan: &OqmdLocalImportPlan) -> Result<ProcessSpec, OqmdImportExecutionError> {
    let query = "SELECT @@version,@@version_comment,@@character_set_server,@@collation_server,@@sql_mode,@@lower_case_table_names,@@time_zone,@@max_allowed_packet,@@innodb_strict_mode,@@socket,@@pid_file";
    let spec = ProcessSpec {
        command: plan.mysql_client.path.clone(),
        args: vec![
            "--no-defaults".to_string(),
            "--protocol=SOCKET".to_string(),
            format!("--socket={}", plan.unix_socket_path),
            format!("--user={}", plan.mysql_user),
            "--batch".to_string(),
            "--raw".to_string(),
            "--skip-column-names".to_string(),
            format!("--execute={query}"),
        ],
        environment: deterministic_environment(),
        environment_policy: EnvironmentPolicy::ClearAndSet,
        timeout_ms: plan.probe_timeout_ms,
        max_output_bytes: plan.max_output_bytes,
    };
    spec.validate().map_err(|e| OqmdImportExecutionError::Process(e.to_string()))?;
    Ok(spec)
}

fn parse_server_observation(bytes: &[u8]) -> Result<MysqlServerObservation, OqmdImportExecutionError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| OqmdImportExecutionError::ServerProbeNotUtf8)?
        .trim_end_matches(|ch| ch == '\r' || ch == '\n');
    if text.contains('\n') || text.contains('\r') {
        return Err(OqmdImportExecutionError::UnexpectedServerProbeShape);
    }
    let fields: Vec<&str> = text.split('\t').collect();
    if fields.len() != 11 {
        return Err(OqmdImportExecutionError::UnexpectedServerProbeShape);
    }
    let mut sql_mode: Vec<String> = fields[4]
        .split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.trim().to_string())
        .collect();
    sql_mode.sort();
    sql_mode.dedup();
    let lower_case_table_names = fields[5].parse::<u8>()
        .map_err(|_| OqmdImportExecutionError::UnexpectedServerProbeShape)?;
    let max_allowed_packet_bytes = fields[7].parse::<u64>()
        .map_err(|_| OqmdImportExecutionError::UnexpectedServerProbeShape)?;
    let innodb_strict_mode = match fields[8].to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => true,
        "0" | "off" | "false" => false,
        _ => return Err(OqmdImportExecutionError::UnexpectedServerProbeShape),
    };
    Ok(MysqlServerObservation {
        runtime_version: fields[0].to_string(),
        version_comment: fields[1].to_string(),
        character_set_server: fields[2].to_string(),
        collation_server: fields[3].to_string(),
        sql_mode,
        lower_case_table_names,
        time_zone: fields[6].to_string(),
        max_allowed_packet_bytes,
        innodb_strict_mode,
        socket_path: fields[9].to_string(),
        pid_file: fields[10].to_string(),
    })
}

fn validate_server_observation(
    observed: &MysqlServerObservation,
    profile: &HistoricalImportProfile,
    plan: &OqmdLocalImportPlan,
) -> Result<(), OqmdImportExecutionError> {
    if observed.character_set_server != profile.character_set_server
        || observed.collation_server != profile.collation_server
        || observed.sql_mode != profile.sql_mode
        || observed.lower_case_table_names != profile.lower_case_table_names
        || observed.time_zone != profile.time_zone
        || observed.max_allowed_packet_bytes != profile.max_allowed_packet_bytes
        || observed.innodb_strict_mode != profile.innodb_strict_mode
    {
        return Err(OqmdImportExecutionError::ServerConfigurationMismatch);
    }
    if observed.socket_path != plan.unix_socket_path {
        return Err(OqmdImportExecutionError::ServerSocketMismatch);
    }
    validate_absolute_text_path(&observed.pid_file, "server_pid_file")?;
    Ok(())
}

fn require_success(capture: &ProcessCapture, label: &'static str) -> Result<(), OqmdImportExecutionError> {
    if !capture.process_success() {
        return Err(OqmdImportExecutionError::ProcessDidNotSucceed(label));
    }
    if capture.stdout_truncated || capture.stderr_truncated {
        return Err(OqmdImportExecutionError::ProbeOutputTruncated(label));
    }
    Ok(())
}

fn require_exact_trimmed_stdout(
    capture: &ProcessCapture,
    expected: &str,
    label: &'static str,
) -> Result<(), OqmdImportExecutionError> {
    let observed = std::str::from_utf8(&capture.stdout)
        .map_err(|_| OqmdImportExecutionError::VersionOutputNotUtf8(label))?
        .trim();
    if observed != expected.trim() {
        return Err(OqmdImportExecutionError::VersionStringMismatch(label));
    }
    Ok(())
}

fn read_pid_file(path: &Path) -> Result<u32, OqmdImportExecutionError> {
    validate_absolute_path(path, "server_pid_file")?;
    let text = std::fs::read_to_string(path).map_err(OqmdImportExecutionError::Io)?;
    text.trim().parse::<u32>().map_err(|_| OqmdImportExecutionError::InvalidPidFile)
}

#[cfg(unix)]
fn hash_running_executable(pid: u32) -> Result<String, OqmdImportExecutionError> {
    let path = PathBuf::from(format!("/proc/{pid}/exe"));
    Ok(hash_file(&path)?.0)
}

#[cfg(not(unix))]
fn hash_running_executable(_pid: u32) -> Result<String, OqmdImportExecutionError> {
    Err(OqmdImportExecutionError::UnsupportedPlatform)
}

fn deterministic_environment() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("LANG".to_string(), "C".to_string()),
        ("LC_ALL".to_string(), "C".to_string()),
        ("TZ".to_string(), "UTC".to_string()),
    ])
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), OqmdImportExecutionError> {
    if value.is_empty() || value.len() > 64
        || !value.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(OqmdImportExecutionError::InvalidIdentifier(field));
    }
    Ok(())
}

fn validate_absolute_text_path(value: &str, field: &'static str) -> Result<(), OqmdImportExecutionError> {
    if value.trim().is_empty() || !Path::new(value).is_absolute() || value.contains('\0') {
        return Err(OqmdImportExecutionError::InvalidAbsolutePath(field));
    }
    Ok(())
}

fn validate_absolute_path(path: &Path, field: &'static str) -> Result<(), OqmdImportExecutionError> {
    if !path.is_absolute() {
        return Err(OqmdImportExecutionError::InvalidAbsolutePath(field));
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<(String, u64), OqmdImportExecutionError> {
    let file = File::open(path).map_err(OqmdImportExecutionError::Io)?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut chunk).map_err(OqmdImportExecutionError::Io)?;
        if read == 0 { break; }
        digest.update(&chunk[..read]);
        bytes = bytes.checked_add(read as u64).ok_or(OqmdImportExecutionError::ByteCountOverflow)?;
    }
    Ok((format!("{:x}", digest.finalize()), bytes))
}

fn validate_sha256(value: &str) -> Result<(), OqmdImportExecutionError> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(OqmdImportExecutionError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// OQMD decompression/import execution failure.
#[derive(Debug, Error)]
pub enum OqmdImportExecutionError {
    /// Historical protocol invalid.
    #[error("historical protocol rejected input: {0}")]
    Protocol(String),
    /// Import profile invalid.
    #[error("historical import profile rejected input: {0}")]
    ImportProfile(String),
    /// Acquisition receipt invalid.
    #[error("historical acquisition receipt rejected input: {0}")]
    Acquisition(String),
    /// SIM-PROC file-I/O layer failure.
    #[error("file-I/O process binding failed: {0}")]
    FileIo(String),
    /// SIM-PROC stdin layer failure.
    #[error("stdin process binding failed: {0}")]
    Stdin(String),
    /// Raw process layer failure.
    #[error("raw process capture failed: {0}")]
    Process(String),
    /// Unsupported plan schema.
    #[error("unsupported local import plan schema {0}")]
    UnsupportedPlanSchema(u32),
    /// Unsupported command schema.
    #[error("unsupported mysql command contract schema {0}")]
    UnsupportedCommandSchema(u32),
    /// Protocol identity mismatch.
    #[error("local import plan protocol digest mismatch")]
    ProtocolDigestMismatch,
    /// Import profile identity mismatch.
    #[error("local import plan profile digest mismatch")]
    ImportProfileDigestMismatch,
    /// Acquisition receipt identity mismatch.
    #[error("local import plan acquisition receipt digest mismatch")]
    AcquisitionReceiptDigestMismatch,
    /// Decompressor identity mismatch.
    #[error("decompressor artifact digest differs from import profile")]
    DecompressorDigestMismatch,
    /// Client identity mismatch.
    #[error("mysql client artifact digest differs from import profile")]
    MysqlClientDigestMismatch,
    /// Server identity mismatch.
    #[error("mysql server artifact digest differs from import profile")]
    MysqlServerDigestMismatch,
    /// Environment identity mismatch.
    #[error("execution environment manifest digest differs from import profile")]
    EnvironmentDigestMismatch,
    /// Command contract differs from preregistration.
    #[error("derived mysql import command digest differs from import profile")]
    ImportCommandDigestMismatch,
    /// Compressed/decompressed paths alias.
    #[error("compressed archive and decompressed SQL paths must differ")]
    InputOutputPathAlias,
    /// Required path invalid.
    #[error("invalid absolute path: {0}")]
    InvalidAbsolutePath(&'static str),
    /// Path not UTF-8.
    #[error("runtime artifact path is not UTF-8")]
    NonUtf8Path,
    /// Local artifact changed.
    #[error("runtime artifact digest mismatch: {0}")]
    RuntimeArtifactDigestMismatch(String),
    /// Identifier invalid.
    #[error("invalid MySQL identifier: {0}")]
    InvalidIdentifier(&'static str),
    /// Server PID invalid.
    #[error("server PID must be non-zero")]
    InvalidServerPid,
    /// Timeout invalid.
    #[error("execution timeout outside supported range: {0} ms")]
    InvalidTimeout(u64),
    /// Output limit invalid.
    #[error("diagnostic output bound outside supported range: {0} bytes")]
    InvalidOutputLimit(usize),
    /// Defaults were not disabled first.
    #[error("mysql command must place --no-defaults first")]
    MysqlDefaultsNotDisabled,
    /// Environment not closed.
    #[error("mysql import command must use a closed environment")]
    MysqlEnvironmentNotClosed,
    /// Required process failed.
    #[error("required process did not succeed: {0}")]
    ProcessDidNotSucceed(&'static str),
    /// Probe output truncated.
    #[error("required probe output was truncated: {0}")]
    ProbeOutputTruncated(&'static str),
    /// Version output invalid UTF-8.
    #[error("version output is not UTF-8: {0}")]
    VersionOutputNotUtf8(&'static str),
    /// Version string mismatch.
    #[error("version string differs from import profile: {0}")]
    VersionStringMismatch(&'static str),
    /// Probe output invalid UTF-8.
    #[error("live MySQL server probe output is not UTF-8")]
    ServerProbeNotUtf8,
    /// Probe output malformed.
    #[error("live MySQL server probe output has unexpected shape")]
    UnexpectedServerProbeShape,
    /// Live server settings differ.
    #[error("live MySQL server configuration differs from import profile")]
    ServerConfigurationMismatch,
    /// Live socket differs.
    #[error("live MySQL server reports another Unix socket")]
    ServerSocketMismatch,
    /// PID file malformed.
    #[error("server PID file does not contain one valid PID")]
    InvalidPidFile,
    /// Server PID mismatch.
    #[error("server PID mismatch: expected {expected}, observed {observed}")]
    ServerPidMismatch { expected: u32, observed: u32 },
    /// Running server executable differs.
    #[error("/proc/<pid>/exe differs from selected MySQL server artifact")]
    RunningServerDigestMismatch,
    /// Platform unsupported for executable identity.
    #[error("running server executable identity currently requires Unix /proc")]
    UnsupportedPlatform,
    /// Decompression not complete.
    #[error("decompression evidence has not established a complete SQL artifact")]
    DecompressionNotQualified,
    /// Preflight belongs to another plan.
    #[error("server preflight belongs to another local import plan")]
    PreflightPlanMismatch,
    /// SQL output inconsistent.
    #[error("decompressed SQL identity is inconsistent with local import plan")]
    DecompressedSqlIdentityMismatch,
    /// SHA malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Byte-count overflow.
    #[error("file byte count overflowed u64")]
    ByteCountOverflow,
    /// File I/O failure.
    #[error("local file I/O failure: {0}")]
    Io(#[source] std::io::Error),
    /// Serialization failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_probe_parser_canonicalizes_sql_modes() {
        let bytes = b"8.4.0\tMySQL Community\tutf8mb4\tutf8mb4_bin\tSTRICT_TRANS_TABLES,NO_ZERO_DATE\t0\t+00:00\t67108864\tON\t/tmp/mysql.sock\t/tmp/mysql.pid\n";
        let observed = parse_server_observation(bytes).unwrap();
        assert_eq!(observed.sql_mode, vec!["NO_ZERO_DATE".to_string(), "STRICT_TRANS_TABLES".to_string()]);
        assert!(observed.innodb_strict_mode);
    }

    #[test]
    fn mysql_command_contract_is_closed_and_disables_defaults() {
        use symthaea_materials_historical_import::HistoricalDatabaseEngine;
        let plan = OqmdLocalImportPlan {
            schema_version: 1,
            protocol_sha256: "1".repeat(64),
            import_profile_sha256: "2".repeat(64),
            acquisition_receipt_sha256: "3".repeat(64),
            compressed_snapshot_path: "/tmp/oqmd.sql.gz".to_string(),
            decompressed_sql_path: "/tmp/oqmd.sql".to_string(),
            decompressor: RuntimeArtifact { path: "/nix/store/gzip/bin/gzip".to_string(), sha256: "4".repeat(64) },
            file_io_launcher: FileIoLauncherArtifact { path: "/nix/store/fileio/bin/launcher".to_string(), sha256: "5".repeat(64) },
            mysql_client: RuntimeArtifact { path: "/nix/store/mysql/bin/mysql".to_string(), sha256: "6".repeat(64) },
            stdin_launcher: StdinLauncherArtifact { path: "/nix/store/stdin/bin/launcher".to_string(), sha256: "7".repeat(64) },
            mysql_server: RuntimeArtifact { path: "/nix/store/mysql/bin/mysqld".to_string(), sha256: "8".repeat(64) },
            server_pid: 123,
            unix_socket_path: "/tmp/oqmd/mysql.sock".to_string(),
            database_name: "oqmd_v17".to_string(),
            mysql_user: "root".to_string(),
            execution_environment_manifest: RuntimeArtifact { path: "/tmp/env.json".to_string(), sha256: "9".repeat(64) },
            decompression_timeout_ms: 1000,
            probe_timeout_ms: 1000,
            import_timeout_ms: 1000,
            max_output_bytes: 4096,
        };
        let profile = HistoricalImportProfile {
            schema_version: 1,
            protocol_sha256: plan.protocol_sha256.clone(),
            database_engine: HistoricalDatabaseEngine::MySql,
            server_artifact_sha256: plan.mysql_server.sha256.clone(),
            client_artifact_sha256: plan.mysql_client.sha256.clone(),
            decompressor_artifact_sha256: plan.decompressor.sha256.clone(),
            import_environment_manifest_sha256: plan.execution_environment_manifest.sha256.clone(),
            server_version: "server".to_string(),
            client_version: "client".to_string(),
            character_set_server: "utf8mb4".to_string(),
            collation_server: "utf8mb4_bin".to_string(),
            sql_mode: vec!["STRICT_TRANS_TABLES".to_string()],
            lower_case_table_names: 0,
            time_zone: "+00:00".to_string(),
            max_allowed_packet_bytes: 67_108_864,
            innodb_strict_mode: true,
            import_command_sha256: "a".repeat(64),
        };
        let contract = plan.mysql_command_contract(&profile).unwrap();
        assert_eq!(contract.args.first().map(String::as_str), Some("--no-defaults"));
        assert_eq!(contract.environment_policy, EnvironmentPolicy::ClearAndSet);
        assert!(contract.args.iter().any(|arg| arg == "--protocol=SOCKET"));
        assert!(contract.args.iter().any(|arg| arg == "oqmd_v17"));
        assert_eq!(contract.command_sha256().unwrap().len(), 64);
    }
}
