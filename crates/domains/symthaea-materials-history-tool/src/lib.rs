// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end verification for one frozen historical materials benchmark run.
//!
//! This crate does not download OQMD or execute MySQL. It composes the already-frozen
//! acquisition, import, schema, extraction, and contamination authorities over actual
//! artifacts. External process execution is intentionally deferred to a separately
//! qualified process-capture adapter.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use symthaea_materials_corpus_audit::{
    CorpusContaminationAudit, HistoricalBenchmarkTarget, audit_historical_corpus,
};
use symthaea_materials_historical_extraction::{
    CompactHistoricalCorpus, HistoricalExtractionReceipt, OqmdExtractionProtocol,
};
use symthaea_materials_historical_import::{
    HistoricalImportExecutionReceipt, HistoricalImportProfile, bind_import_to_extraction,
};
use symthaea_materials_schema_inventory::{
    MySqlSchemaInventory, bind_schema_inventory_to_import,
};
use symthaea_materials_snapshot_acquisition::{
    HistoricalSnapshotAcquisitionReceipt, verify_snapshot_stream,
};
use thiserror::Error;

/// Immutable identity of one fully cross-checked historical benchmark input bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalRunBundle {
    /// Bundle schema version.
    pub schema_version: u32,
    /// Exact verifier executable artifact used to produce this bundle.
    pub verifier_artifact_sha256: String,
    /// Exact frozen historical-extraction protocol.
    pub protocol_sha256: String,
    /// Exact compressed historical archive bytes.
    pub compressed_snapshot_sha256: String,
    /// Exact compressed byte count re-observed by this verifier.
    pub compressed_snapshot_bytes: u64,
    /// Acquisition receipt identity.
    pub acquisition_receipt_sha256: String,
    /// Preregistered MySQL import profile identity.
    pub import_profile_sha256: String,
    /// Executed import receipt identity.
    pub import_receipt_sha256: String,
    /// Canonical schema inventory identity.
    pub schema_inventory_sha256: String,
    /// Canonical `(table, exact_row_count)` inventory identity.
    pub row_count_inventory_sha256: String,
    /// Cross-object import-to-extraction binding identity.
    pub import_extraction_binding_sha256: String,
    /// Extraction receipt identity.
    pub extraction_receipt_sha256: String,
    /// Frozen compact Fe/Co/Zr corpus identity.
    pub compact_corpus_sha256: String,
    /// Exact preserved record count.
    pub compact_corpus_record_count: u64,
    /// Canonical benchmark target-set identity computed by this verifier.
    pub target_set_sha256: String,
    /// Number of targets audited.
    pub target_count: u64,
    /// Historical-corpus contamination audit identity.
    pub contamination_audit_sha256: String,
}

impl HistoricalRunBundle {
    /// Deterministic digest of this complete verified bundle.
    pub fn bundle_sha256(&self) -> Result<String, HistoryToolError> {
        if self.schema_version != 1 {
            return Err(HistoryToolError::UnsupportedBundleSchema(self.schema_version));
        }
        validate_sha256(&self.verifier_artifact_sha256)?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Bundle plus the complete per-target contamination classifications.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedHistoricalRun {
    /// Content-addressed input/evidence bundle.
    pub bundle: HistoricalRunBundle,
    /// Full contamination audit; targets are never silently dropped.
    pub contamination_audit: CorpusContaminationAudit,
}

impl VerifiedHistoricalRun {
    /// Deterministic digest over both bundle and full audit detail.
    pub fn verified_run_sha256(&self) -> Result<String, HistoryToolError> {
        self.bundle.bundle_sha256()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Canonicalize target order and property-label order before hashing/scoring.
pub fn canonicalize_targets(mut targets: Vec<HistoricalBenchmarkTarget>) -> Vec<HistoricalBenchmarkTarget> {
    for target in &mut targets {
        target.scored_property_labels.sort();
    }
    targets.sort_by(|a, b| a.target_id.cmp(&b.target_id));
    targets
}

/// Deterministic target-set identity after semantic ordering normalization.
pub fn canonical_target_set_sha256(
    targets: Vec<HistoricalBenchmarkTarget>,
) -> Result<String, HistoryToolError> {
    Ok(sha256_hex(&serde_json::to_vec(&canonicalize_targets(targets))?))
}

/// Verify one historical benchmark run from actual frozen artifacts.
#[allow(clippy::too_many_arguments)]
pub fn verify_historical_run<R: Read>(
    verifier_artifact_sha256: &str,
    protocol: &OqmdExtractionProtocol,
    snapshot_reader: R,
    acquisition: &HistoricalSnapshotAcquisitionReceipt,
    import_profile: &HistoricalImportProfile,
    import_receipt: &HistoricalImportExecutionReceipt,
    schema_inventory: &MySqlSchemaInventory,
    extraction_receipt: &HistoricalExtractionReceipt,
    compact_corpus: &CompactHistoricalCorpus,
    targets: &[HistoricalBenchmarkTarget],
) -> Result<VerifiedHistoricalRun, HistoryToolError> {
    validate_sha256(verifier_artifact_sha256)?;
    protocol
        .validate()
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;

    let streamed = verify_snapshot_stream(protocol, acquisition, snapshot_reader)
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;
    import_profile
        .validate_against(protocol)
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;
    import_receipt
        .validate_against(protocol, import_profile, acquisition)
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;

    let schema_binding = bind_schema_inventory_to_import(schema_inventory, import_receipt)
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;
    let import_extraction_binding = bind_import_to_extraction(
        protocol,
        import_profile,
        acquisition,
        import_receipt,
        extraction_receipt,
    )
    .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;

    compact_corpus
        .validate_for(protocol)
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;
    let corpus_sha = compact_corpus
        .corpus_sha256(protocol)
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;
    if !corpus_sha.eq_ignore_ascii_case(&extraction_receipt.compact_corpus_sha256) {
        return Err(HistoryToolError::ExtractionCorpusMismatch);
    }
    let record_count = compact_corpus.records.len() as u64;
    if record_count != extraction_receipt.record_count {
        return Err(HistoryToolError::ExtractionRecordCountMismatch {
            extraction: extraction_receipt.record_count,
            corpus: record_count,
        });
    }

    let canonical_targets = canonicalize_targets(targets.to_vec());
    let target_set_sha = sha256_hex(&serde_json::to_vec(&canonical_targets)?);
    let audit_records = compact_corpus
        .audit_records(protocol)
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;
    let contamination_audit = audit_historical_corpus(
        &corpus_sha,
        &target_set_sha,
        &audit_records,
        &canonical_targets,
    )
    .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;
    let audit_sha = contamination_audit
        .audit_sha256()
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;

    let acquisition_sha = acquisition
        .receipt_sha256(protocol)
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;
    let import_profile_sha = import_profile
        .profile_sha256(protocol)
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;
    let import_receipt_sha = import_receipt
        .receipt_sha256(protocol, import_profile, acquisition)
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;
    let extraction_receipt_sha = extraction_receipt
        .receipt_sha256()
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;
    let import_extraction_binding_sha = import_extraction_binding
        .binding_sha256()
        .map_err(|error| HistoryToolError::Upstream(error.to_string()))?;

    let bundle = HistoricalRunBundle {
        schema_version: 1,
        verifier_artifact_sha256: verifier_artifact_sha256.to_ascii_lowercase(),
        protocol_sha256: protocol
            .protocol_sha256()
            .map_err(|error| HistoryToolError::Upstream(error.to_string()))?,
        compressed_snapshot_sha256: streamed.sha256,
        compressed_snapshot_bytes: streamed.bytes,
        acquisition_receipt_sha256: acquisition_sha,
        import_profile_sha256: import_profile_sha,
        import_receipt_sha256: import_receipt_sha,
        schema_inventory_sha256: schema_binding.schema_inventory_sha256,
        row_count_inventory_sha256: schema_binding.row_count_inventory_sha256,
        import_extraction_binding_sha256,
        extraction_receipt_sha256,
        compact_corpus_sha256: corpus_sha,
        compact_corpus_record_count: record_count,
        target_set_sha256: target_set_sha,
        target_count: canonical_targets.len() as u64,
        contamination_audit_sha256: audit_sha,
    };
    bundle.bundle_sha256()?;

    Ok(VerifiedHistoricalRun {
        bundle,
        contamination_audit,
    })
}

fn validate_sha256(value: &str) -> Result<(), HistoryToolError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(HistoryToolError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// End-to-end history-tool verification failure.
#[derive(Debug, Error)]
pub enum HistoryToolError {
    /// One frozen upstream authority rejected its input.
    #[error("upstream historical-data authority rejected input: {0}")]
    Upstream(String),
    /// SHA-256 text is malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Extraction receipt names another compact corpus.
    #[error("extraction receipt compact-corpus SHA differs from actual corpus")]
    ExtractionCorpusMismatch,
    /// Extraction receipt record count differs from actual compact corpus.
    #[error("extraction record-count mismatch: receipt={extraction}, corpus={corpus}")]
    ExtractionRecordCountMismatch {
        /// Count named by extraction receipt.
        extraction: u64,
        /// Count present in actual compact corpus.
        corpus: u64,
    },
    /// Bundle schema unsupported.
    #[error("unsupported historical-run bundle schema {0}")]
    UnsupportedBundleSchema(u32),
    /// JSON serialization/deserialization failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use symthaea_materials_corpus_audit::PropertyLabelRef;
    use symthaea_materials_historical_extraction::{
        NormalizedOqmdRecord, bind_extraction_receipt, oqmd_v17_fe_co_zr_protocol,
    };
    use symthaea_materials_historical_import::{HistoricalDatabaseEngine, bind_successful_import};
    use symthaea_materials_schema_inventory::{
        ColumnInventory, IndexColumnInventory, IndexInventory, TableInventory,
    };
    use symthaea_materials_snapshot_acquisition::{AcquisitionRoute, hash_snapshot_stream};

    fn hex(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn target(id: &str, labels: Vec<PropertyLabelRef>) -> HistoricalBenchmarkTarget {
        HistoricalBenchmarkTarget {
            target_id: id.to_string(),
            composition_sha256: hex('e'),
            structure_sha256: Some(hex('f')),
            scored_property_labels: labels,
        }
    }

    #[test]
    fn target_and_label_order_do_not_change_target_set_identity() {
        let k1 = PropertyLabelRef {
            property_id: "k1".to_string(),
            condition_signature: "0K|SOC".to_string(),
        };
        let js = PropertyLabelRef {
            property_id: "js".to_string(),
            condition_signature: "0K|spin-dft".to_string(),
        };
        let one = vec![target("b", vec![k1.clone(), js.clone()]), target("a", vec![js.clone()])];
        let two = vec![target("a", vec![js.clone()]), target("b", vec![js, k1])];
        assert_eq!(
            canonical_target_set_sha256(one).unwrap(),
            canonical_target_set_sha256(two).unwrap()
        );
    }

    #[test]
    fn verifies_complete_fixture_and_emits_content_addressed_bundle() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let snapshot = b"oqmd-v1.7-test-archive";
        let identity = hash_snapshot_stream(Cursor::new(snapshot)).unwrap();
        let acquisition = HistoricalSnapshotAcquisitionReceipt {
            schema_version: 1,
            protocol_sha256: protocol.protocol_sha256().unwrap(),
            provider: protocol.provider.clone(),
            database_version: protocol.database_version.clone(),
            dump_filename: protocol.dump_filename.clone(),
            source_license: protocol.source_license.clone(),
            acquired_at_utc: "2026-09-19T20:00:00Z".to_string(),
            compressed_snapshot_sha256: identity.sha256.clone(),
            compressed_snapshot_bytes: identity.bytes,
            acquisition_tool_sha256: hex('a'),
            execution_environment_sha256: hex('b'),
            transfer_log_sha256: hex('c'),
            route: AcquisitionRoute::LocalMirror {
                mirror_locator: "fixture".to_string(),
                mirror_manifest_sha256: hex('d'),
            },
        };
        let profile = HistoricalImportProfile {
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
            sql_mode: vec!["STRICT_TRANS_TABLES".to_string()],
            lower_case_table_names: 0,
            time_zone: "+00:00".to_string(),
            max_allowed_packet_bytes: 64 * 1024 * 1024,
            innodb_strict_mode: true,
            import_command_sha256: hex('5'),
        };
        let inventory = MySqlSchemaInventory {
            schema_version: 1,
            database_name: "oqmd_v17".to_string(),
            tables: vec![TableInventory {
                name: "entries".to_string(),
                engine: "InnoDB".to_string(),
                show_create_table_sha256: hex('6'),
                exact_row_count: 1,
                columns: vec![ColumnInventory {
                    ordinal_position: 1,
                    name: "id".to_string(),
                    column_type: "bigint".to_string(),
                    nullable: false,
                    default_repr: None,
                    collation: None,
                    extra: String::new(),
                }],
                indexes: vec![IndexInventory {
                    name: "PRIMARY".to_string(),
                    unique: true,
                    index_type: "BTREE".to_string(),
                    columns: vec![IndexColumnInventory {
                        sequence: 1,
                        column_name: "id".to_string(),
                        sub_part: None,
                        descending: false,
                    }],
                }],
                foreign_keys: Vec::new(),
            }],
        };
        let import = bind_successful_import(
            &protocol,
            &profile,
            &acquisition,
            &hex('7'),
            &hex('8'),
            &inventory.inventory_sha256().unwrap(),
            &inventory.row_count_projection_sha256().unwrap(),
            &hex('9'),
        )
        .unwrap();
        let corpus = CompactHistoricalCorpus::from_records(
            &protocol,
            vec![NormalizedOqmdRecord {
                entry_id: 1,
                name: "Fe".to_string(),
                element_set: vec!["Fe".to_string()],
                composition_sha256: hex('e'),
                structure_sha256: Some(hex('f')),
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
        .unwrap();
        let import_sha = import.receipt_sha256(&protocol, &profile, &acquisition).unwrap();
        let extraction = bind_extraction_receipt(
            &protocol,
            &corpus,
            &identity.sha256,
            &profile.import_environment_manifest_sha256,
            &inventory.inventory_sha256().unwrap(),
            &import_sha,
            &hex('a'),
            &hex('b'),
            &hex('c'),
            &hex('d'),
            &hex('1'),
        )
        .unwrap();
        let target = target(
            "held-out-fe",
            vec![PropertyLabelRef {
                property_id: "k1_mj_m3".to_string(),
                condition_signature: "0K|SOC".to_string(),
            }],
        );

        let verified = verify_historical_run(
            &hex('9'),
            &protocol,
            Cursor::new(snapshot),
            &acquisition,
            &profile,
            &import,
            &inventory,
            &extraction,
            &corpus,
            &[target],
        )
        .unwrap();

        assert_eq!(verified.bundle.compact_corpus_record_count, 1);
        assert_eq!(verified.bundle.target_count, 1);
        assert_eq!(verified.bundle.bundle_sha256().unwrap().len(), 64);
        assert_eq!(verified.verified_run_sha256().unwrap().len(), 64);
    }
}
