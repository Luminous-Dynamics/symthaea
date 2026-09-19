// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistered historical-database import profiles and execution receipts.
//!
//! OQMD v1.7 is distributed as a MySQL dump. Reproducible historical benchmarking
//! therefore needs to bind not only the dump bytes but the exact import environment
//! and semantics used to materialize them before extraction.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_materials_historical_extraction::{
    HistoricalExtractionReceipt, OqmdExtractionProtocol,
};
use symthaea_materials_snapshot_acquisition::HistoricalSnapshotAcquisitionReceipt;
use thiserror::Error;

/// Database engine expected by the historical dump profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalDatabaseEngine {
    /// MySQL-compatible import semantics.
    MySql,
}

/// Import semantics committed before the historical dump is materialized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalImportProfile {
    /// Profile schema version.
    pub schema_version: u32,
    /// Exact extraction protocol digest.
    pub protocol_sha256: String,
    /// Database engine family.
    pub database_engine: HistoricalDatabaseEngine,
    /// Exact server binary/package artifact digest.
    pub server_artifact_sha256: String,
    /// Exact client binary/package artifact digest.
    pub client_artifact_sha256: String,
    /// Exact decompressor artifact digest.
    pub decompressor_artifact_sha256: String,
    /// Exact Nix/container environment manifest digest used by extraction receipts.
    pub import_environment_manifest_sha256: String,
    /// Server version string captured from the exact binary.
    pub server_version: String,
    /// Client version string captured from the exact binary.
    pub client_version: String,
    /// Server character set.
    pub character_set_server: String,
    /// Server collation.
    pub collation_server: String,
    /// Exact SQL mode entries, lexically sorted and unique.
    pub sql_mode: Vec<String>,
    /// MySQL lower-case-table-names setting (0, 1, or 2).
    pub lower_case_table_names: u8,
    /// Server time zone. Historical qualification requires `+00:00`.
    pub time_zone: String,
    /// Bound maximum packet size used during import.
    pub max_allowed_packet_bytes: u64,
    /// Whether strict InnoDB behavior is enabled.
    pub innodb_strict_mode: bool,
    /// Exact import command/script artifact digest.
    pub import_command_sha256: String,
}

impl HistoricalImportProfile {
    /// Validate this profile against the frozen extraction protocol.
    pub fn validate_against(
        &self,
        protocol: &OqmdExtractionProtocol,
    ) -> Result<(), HistoricalImportError> {
        protocol
            .validate()
            .map_err(|error| HistoricalImportError::Protocol(error.to_string()))?;
        if self.schema_version != 1 {
            return Err(HistoricalImportError::UnsupportedProfileSchema(
                self.schema_version,
            ));
        }
        let protocol_sha = protocol
            .protocol_sha256()
            .map_err(|error| HistoricalImportError::Protocol(error.to_string()))?;
        if self.protocol_sha256 != protocol_sha {
            return Err(HistoricalImportError::ProtocolDigestMismatch);
        }
        if !protocol.database_engine.eq_ignore_ascii_case("mysql")
            || self.database_engine != HistoricalDatabaseEngine::MySql
        {
            return Err(HistoricalImportError::DatabaseEngineMismatch);
        }
        for digest in [
            &self.server_artifact_sha256,
            &self.client_artifact_sha256,
            &self.decompressor_artifact_sha256,
            &self.import_environment_manifest_sha256,
            &self.import_command_sha256,
        ] {
            sha256(digest)?;
        }
        for (field, value) in [
            ("server_version", &self.server_version),
            ("client_version", &self.client_version),
            ("character_set_server", &self.character_set_server),
            ("collation_server", &self.collation_server),
            ("time_zone", &self.time_zone),
        ] {
            nonempty(field, value)?;
        }
        if self.time_zone != "+00:00" {
            return Err(HistoricalImportError::NonUtcImportTimezone(
                self.time_zone.clone(),
            ));
        }
        if self.lower_case_table_names > 2 {
            return Err(HistoricalImportError::InvalidLowerCaseTableNames(
                self.lower_case_table_names,
            ));
        }
        if self.max_allowed_packet_bytes == 0 {
            return Err(HistoricalImportError::InvalidMaxAllowedPacket);
        }
        validate_sorted_unique_nonempty(&self.sql_mode)?;
        Ok(())
    }

    /// Deterministic identity of the preregistered import semantics.
    pub fn profile_sha256(
        &self,
        protocol: &OqmdExtractionProtocol,
    ) -> Result<String, HistoricalImportError> {
        self.validate_against(protocol)?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Evidence from one completed historical dump import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalImportExecutionReceipt {
    /// Receipt schema version.
    pub schema_version: u32,
    /// Exact import profile digest.
    pub profile_sha256: String,
    /// Exact acquisition receipt digest.
    pub acquisition_receipt_sha256: String,
    /// Exact compressed snapshot digest imported.
    pub compressed_snapshot_sha256: String,
    /// Exact raw process evidence artifact digest for the import command.
    pub import_process_evidence_sha256: String,
    /// Import process reported OS-level success.
    pub process_exit_success: bool,
    /// Exact import log artifact digest.
    pub import_log_sha256: String,
    /// Exact schema inventory artifact digest.
    pub schema_inventory_sha256: String,
    /// Exact table/row-count inventory artifact digest.
    pub row_count_inventory_sha256: String,
    /// Exact post-import database-state manifest digest.
    pub database_state_manifest_sha256: String,
}

impl HistoricalImportExecutionReceipt {
    /// Validate receipt identity against profile/acquisition evidence.
    pub fn validate_against(
        &self,
        protocol: &OqmdExtractionProtocol,
        profile: &HistoricalImportProfile,
        acquisition: &HistoricalSnapshotAcquisitionReceipt,
    ) -> Result<(), HistoricalImportError> {
        profile.validate_against(protocol)?;
        acquisition
            .validate_against(protocol)
            .map_err(|error| HistoricalImportError::Acquisition(error.to_string()))?;
        if self.schema_version != 1 {
            return Err(HistoricalImportError::UnsupportedReceiptSchema(
                self.schema_version,
            ));
        }
        if self.profile_sha256 != profile.profile_sha256(protocol)? {
            return Err(HistoricalImportError::ProfileDigestMismatch);
        }
        let acquisition_sha = acquisition
            .receipt_sha256(protocol)
            .map_err(|error| HistoricalImportError::Acquisition(error.to_string()))?;
        if self.acquisition_receipt_sha256 != acquisition_sha {
            return Err(HistoricalImportError::AcquisitionReceiptMismatch);
        }
        if !self
            .compressed_snapshot_sha256
            .eq_ignore_ascii_case(&acquisition.compressed_snapshot_sha256)
        {
            return Err(HistoricalImportError::CompressedSnapshotMismatch);
        }
        if !self.process_exit_success {
            return Err(HistoricalImportError::ImportProcessDidNotSucceed);
        }
        for digest in [
            &self.import_process_evidence_sha256,
            &self.import_log_sha256,
            &self.schema_inventory_sha256,
            &self.row_count_inventory_sha256,
            &self.database_state_manifest_sha256,
        ] {
            sha256(digest)?;
        }
        Ok(())
    }

    /// Deterministic digest of the validated import receipt.
    pub fn receipt_sha256(
        &self,
        protocol: &OqmdExtractionProtocol,
        profile: &HistoricalImportProfile,
        acquisition: &HistoricalSnapshotAcquisitionReceipt,
    ) -> Result<String, HistoricalImportError> {
        self.validate_against(protocol, profile, acquisition)?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Build a successful import receipt from already-captured execution artifacts.
#[allow(clippy::too_many_arguments)]
pub fn bind_successful_import(
    protocol: &OqmdExtractionProtocol,
    profile: &HistoricalImportProfile,
    acquisition: &HistoricalSnapshotAcquisitionReceipt,
    import_process_evidence_sha256: &str,
    import_log_sha256: &str,
    schema_inventory_sha256: &str,
    row_count_inventory_sha256: &str,
    database_state_manifest_sha256: &str,
) -> Result<HistoricalImportExecutionReceipt, HistoricalImportError> {
    profile.validate_against(protocol)?;
    acquisition
        .validate_against(protocol)
        .map_err(|error| HistoricalImportError::Acquisition(error.to_string()))?;
    for digest in [
        import_process_evidence_sha256,
        import_log_sha256,
        schema_inventory_sha256,
        row_count_inventory_sha256,
        database_state_manifest_sha256,
    ] {
        sha256(digest)?;
    }
    let receipt = HistoricalImportExecutionReceipt {
        schema_version: 1,
        profile_sha256: profile.profile_sha256(protocol)?,
        acquisition_receipt_sha256: acquisition
            .receipt_sha256(protocol)
            .map_err(|error| HistoricalImportError::Acquisition(error.to_string()))?,
        compressed_snapshot_sha256: acquisition.compressed_snapshot_sha256.to_ascii_lowercase(),
        import_process_evidence_sha256: import_process_evidence_sha256.to_ascii_lowercase(),
        process_exit_success: true,
        import_log_sha256: import_log_sha256.to_ascii_lowercase(),
        schema_inventory_sha256: schema_inventory_sha256.to_ascii_lowercase(),
        row_count_inventory_sha256: row_count_inventory_sha256.to_ascii_lowercase(),
        database_state_manifest_sha256: database_state_manifest_sha256.to_ascii_lowercase(),
    };
    receipt.validate_against(protocol, profile, acquisition)?;
    Ok(receipt)
}

/// Verified bridge from the import evidence into MAG-DATA-003 extraction evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportExtractionBinding {
    /// Exact import profile digest.
    pub profile_sha256: String,
    /// Exact import receipt digest.
    pub import_receipt_sha256: String,
    /// Exact extraction receipt digest.
    pub extraction_receipt_sha256: String,
    /// Shared compressed snapshot identity.
    pub compressed_snapshot_sha256: String,
    /// Shared schema inventory identity.
    pub schema_inventory_sha256: String,
}

impl ImportExtractionBinding {
    /// Deterministic binding identity.
    pub fn binding_sha256(&self) -> Result<String, HistoricalImportError> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Require extraction evidence to use the exact preregistered/imported database state.
pub fn bind_import_to_extraction(
    protocol: &OqmdExtractionProtocol,
    profile: &HistoricalImportProfile,
    acquisition: &HistoricalSnapshotAcquisitionReceipt,
    import: &HistoricalImportExecutionReceipt,
    extraction: &HistoricalExtractionReceipt,
) -> Result<ImportExtractionBinding, HistoricalImportError> {
    import.validate_against(protocol, profile, acquisition)?;
    let import_sha = import.receipt_sha256(protocol, profile, acquisition)?;
    let extraction_sha = extraction
        .receipt_sha256()
        .map_err(|error| HistoricalImportError::Extraction(error.to_string()))?;

    if !extraction
        .protocol_sha256
        .eq_ignore_ascii_case(&profile.protocol_sha256)
    {
        return Err(HistoricalImportError::ExtractionProtocolMismatch);
    }
    if !extraction
        .compressed_dump_sha256
        .eq_ignore_ascii_case(&import.compressed_snapshot_sha256)
    {
        return Err(HistoricalImportError::ExtractionSnapshotMismatch);
    }
    if !extraction
        .import_environment_sha256
        .eq_ignore_ascii_case(&profile.import_environment_manifest_sha256)
    {
        return Err(HistoricalImportError::ExtractionEnvironmentMismatch);
    }
    if !extraction
        .schema_inventory_sha256
        .eq_ignore_ascii_case(&import.schema_inventory_sha256)
    {
        return Err(HistoricalImportError::ExtractionSchemaMismatch);
    }
    if !extraction.import_receipt_sha256.eq_ignore_ascii_case(&import_sha) {
        return Err(HistoricalImportError::ExtractionImportReceiptMismatch);
    }

    Ok(ImportExtractionBinding {
        profile_sha256: profile.profile_sha256(protocol)?,
        import_receipt_sha256: import_sha,
        extraction_receipt_sha256: extraction_sha,
        compressed_snapshot_sha256: import.compressed_snapshot_sha256.to_ascii_lowercase(),
        schema_inventory_sha256: import.schema_inventory_sha256.to_ascii_lowercase(),
    })
}

fn validate_sorted_unique_nonempty(values: &[String]) -> Result<(), HistoricalImportError> {
    let mut previous: Option<&str> = None;
    let mut seen = BTreeSet::new();
    for value in values {
        nonempty("sql_mode", value)?;
        if previous.is_some_and(|prior| value.as_str() <= prior) || !seen.insert(value.as_str()) {
            return Err(HistoricalImportError::NonCanonicalSqlMode);
        }
        previous = Some(value);
    }
    Ok(())
}

fn nonempty(name: &'static str, value: &str) -> Result<(), HistoricalImportError> {
    if value.trim().is_empty() {
        return Err(HistoricalImportError::EmptyField(name));
    }
    Ok(())
}

fn sha256(value: &str) -> Result<(), HistoricalImportError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(HistoricalImportError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Historical import/profile verification failure.
#[derive(Debug, Error)]
pub enum HistoricalImportError {
    /// Extraction protocol invalid.
    #[error("invalid historical extraction protocol: {0}")]
    Protocol(String),
    /// Profile schema unsupported.
    #[error("unsupported import profile schema {0}")]
    UnsupportedProfileSchema(u32),
    /// Receipt schema unsupported.
    #[error("unsupported import receipt schema {0}")]
    UnsupportedReceiptSchema(u32),
    /// Profile protocol digest mismatch.
    #[error("import profile protocol digest mismatch")]
    ProtocolDigestMismatch,
    /// Database engine inconsistent with protocol/profile.
    #[error("historical import database engine mismatch")]
    DatabaseEngineMismatch,
    /// Required field empty.
    #[error("required field {0} is empty")]
    EmptyField(&'static str),
    /// SHA malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Import timezone is not UTC.
    #[error("historical import timezone must be +00:00, got {0}")]
    NonUtcImportTimezone(String),
    /// lower_case_table_names outside MySQL range.
    #[error("invalid lower_case_table_names value {0}")]
    InvalidLowerCaseTableNames(u8),
    /// max_allowed_packet not positive.
    #[error("max_allowed_packet_bytes must be positive")]
    InvalidMaxAllowedPacket,
    /// SQL-mode list not strictly sorted/unique/non-empty.
    #[error("SQL mode list is not canonical sorted unique text")]
    NonCanonicalSqlMode,
    /// Acquisition evidence invalid.
    #[error("invalid acquisition evidence: {0}")]
    Acquisition(String),
    /// Receipt profile identity differs.
    #[error("import receipt profile digest mismatch")]
    ProfileDigestMismatch,
    /// Receipt acquisition identity differs.
    #[error("import receipt acquisition digest mismatch")]
    AcquisitionReceiptMismatch,
    /// Receipt/import archive differs from acquisition.
    #[error("import receipt compressed snapshot differs from acquisition")]
    CompressedSnapshotMismatch,
    /// Process evidence reports import failure.
    #[error("historical import process did not report OS-level success")]
    ImportProcessDidNotSucceed,
    /// Extraction receipt invalid.
    #[error("invalid extraction receipt: {0}")]
    Extraction(String),
    /// Extraction names another protocol.
    #[error("extraction protocol differs from import profile")]
    ExtractionProtocolMismatch,
    /// Extraction names another archive.
    #[error("extraction compressed dump differs from imported snapshot")]
    ExtractionSnapshotMismatch,
    /// Extraction names another import environment.
    #[error("extraction import environment differs from preregistered profile")]
    ExtractionEnvironmentMismatch,
    /// Extraction names another schema inventory.
    #[error("extraction schema inventory differs from import receipt")]
    ExtractionSchemaMismatch,
    /// Extraction names another import receipt.
    #[error("extraction import receipt differs from executed import")]
    ExtractionImportReceiptMismatch,
    /// Serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use symthaea_materials_historical_extraction::{
        CompactHistoricalCorpus, NormalizedOqmdRecord, bind_extraction_receipt,
        oqmd_v17_fe_co_zr_protocol,
    };
    use symthaea_materials_snapshot_acquisition::{
        AcquisitionRoute, hash_snapshot_stream,
    };

    fn hex(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn profile(protocol: &OqmdExtractionProtocol) -> HistoricalImportProfile {
        HistoricalImportProfile {
            schema_version: 1,
            protocol_sha256: protocol.protocol_sha256().unwrap(),
            database_engine: HistoricalDatabaseEngine::MySql,
            server_artifact_sha256: hex('1'),
            client_artifact_sha256: hex('2'),
            decompressor_artifact_sha256: hex('3'),
            import_environment_manifest_sha256: hex('4'),
            server_version: "fixture-server".to_string(),
            client_version: "fixture-client".to_string(),
            character_set_server: "utf8mb4".to_string(),
            collation_server: "utf8mb4_bin".to_string(),
            sql_mode: vec!["STRICT_ALL_TABLES".to_string()],
            lower_case_table_names: 0,
            time_zone: "+00:00".to_string(),
            max_allowed_packet_bytes: 1_073_741_824,
            innodb_strict_mode: true,
            import_command_sha256: hex('5'),
        }
    }

    fn acquisition(
        protocol: &OqmdExtractionProtocol,
    ) -> HistoricalSnapshotAcquisitionReceipt {
        let identity = hash_snapshot_stream(Cursor::new(b"oqmd-fixture")).unwrap();
        HistoricalSnapshotAcquisitionReceipt {
            schema_version: 1,
            protocol_sha256: protocol.protocol_sha256().unwrap(),
            provider: protocol.provider.clone(),
            database_version: protocol.database_version.clone(),
            dump_filename: protocol.dump_filename.clone(),
            source_license: protocol.source_license.clone(),
            acquired_at_utc: "2026-09-19T20:00:00Z".to_string(),
            compressed_snapshot_sha256: identity.sha256,
            compressed_snapshot_bytes: identity.bytes,
            acquisition_tool_sha256: hex('6'),
            execution_environment_sha256: hex('7'),
            transfer_log_sha256: hex('8'),
            route: AcquisitionRoute::LocalMirror {
                mirror_locator: "fixture".to_string(),
                mirror_manifest_sha256: hex('9'),
            },
        }
    }

    fn corpus(protocol: &OqmdExtractionProtocol) -> CompactHistoricalCorpus {
        CompactHistoricalCorpus::from_records(
            protocol,
            vec![NormalizedOqmdRecord {
                entry_id: 1,
                name: "Fe".to_string(),
                element_set: vec!["Fe".to_string()],
                composition_sha256: hex('a'),
                structure_sha256: Some(hex('b')),
                duplicate_entry_id: None,
                spacegroup: None,
                prototype: None,
                natoms: 1,
                ntypes: 1,
                delta_e_ev_atom: Some("0".to_string()),
                stability_ev_atom: Some("0".to_string()),
                band_gap_ev: None,
                calculation_label: None,
                fit: None,
                icsd_id: None,
                property_condition_signature: "oqmd-v1.7-dft".to_string(),
            }],
        )
        .unwrap()
    }

    #[test]
    fn import_profile_requires_canonical_utc_semantics() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let mut profile = profile(&protocol);
        profile.validate_against(&protocol).unwrap();
        profile.time_zone = "SYSTEM".to_string();
        assert!(matches!(
            profile.validate_against(&protocol),
            Err(HistoricalImportError::NonUtcImportTimezone(_))
        ));
    }

    #[test]
    fn executed_import_binds_acquisition_and_extraction() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let profile = profile(&protocol);
        let acquisition = acquisition(&protocol);
        let import = bind_successful_import(
            &protocol,
            &profile,
            &acquisition,
            &hex('c'),
            &hex('d'),
            &hex('e'),
            &hex('f'),
            &hex('1'),
        )
        .unwrap();
        let corpus = corpus(&protocol);
        let import_sha = import
            .receipt_sha256(&protocol, &profile, &acquisition)
            .unwrap();
        let extraction = bind_extraction_receipt(
            &protocol,
            &corpus,
            &acquisition.compressed_snapshot_sha256,
            &profile.import_environment_manifest_sha256,
            &import.schema_inventory_sha256,
            &import_sha,
            &hex('2'),
            &hex('3'),
            &hex('4'),
            &hex('5'),
            &hex('6'),
        )
        .unwrap();
        let binding = bind_import_to_extraction(
            &protocol,
            &profile,
            &acquisition,
            &import,
            &extraction,
        )
        .unwrap();
        assert_eq!(binding.import_receipt_sha256, import_sha);
        assert_eq!(binding.binding_sha256().unwrap().len(), 64);
    }

    #[test]
    fn extraction_cannot_substitute_another_schema_inventory() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let profile = profile(&protocol);
        let acquisition = acquisition(&protocol);
        let import = bind_successful_import(
            &protocol,
            &profile,
            &acquisition,
            &hex('c'),
            &hex('d'),
            &hex('e'),
            &hex('f'),
            &hex('1'),
        )
        .unwrap();
        let corpus = corpus(&protocol);
        let import_sha = import
            .receipt_sha256(&protocol, &profile, &acquisition)
            .unwrap();
        let mut extraction = bind_extraction_receipt(
            &protocol,
            &corpus,
            &acquisition.compressed_snapshot_sha256,
            &profile.import_environment_manifest_sha256,
            &import.schema_inventory_sha256,
            &import_sha,
            &hex('2'),
            &hex('3'),
            &hex('4'),
            &hex('5'),
            &hex('6'),
        )
        .unwrap();
        extraction.schema_inventory_sha256 = hex('0');
        assert!(matches!(
            bind_import_to_extraction(
                &protocol,
                &profile,
                &acquisition,
                &import,
                &extraction,
            ),
            Err(HistoricalImportError::ExtractionSchemaMismatch)
        ));
    }
}
