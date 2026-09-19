// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-object identity between historical snapshot acquisition and extraction.
//!
//! A historical benchmark must not acquire one archive and extract another while
//! retaining independently well-formed receipts. This crate proves that the exact
//! compressed snapshot receipted by the acquisition layer is the same compressed
//! dump claimed by the extraction layer under the same preregistered protocol.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_materials_historical_extraction::{
    HistoricalExtractionReceipt, OqmdExtractionProtocol,
};
use symthaea_materials_snapshot_acquisition::HistoricalSnapshotAcquisitionReceipt;
use thiserror::Error;

/// Verified identity binding acquisition and extraction receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcquisitionExtractionBinding {
    /// Exact preregistered protocol digest.
    pub protocol_sha256: String,
    /// Exact acquisition receipt digest.
    pub acquisition_receipt_sha256: String,
    /// Exact extraction receipt digest.
    pub extraction_receipt_sha256: String,
    /// Exact compressed historical snapshot/dump digest shared by both receipts.
    pub compressed_snapshot_sha256: String,
}

impl AcquisitionExtractionBinding {
    /// Deterministic digest of the verified cross-object binding.
    pub fn binding_sha256(&self) -> Result<String, AcquisitionExtractionError> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Verify that acquisition and extraction refer to the same historical bytes.
pub fn bind_acquisition_to_extraction(
    protocol: &OqmdExtractionProtocol,
    acquisition: &HistoricalSnapshotAcquisitionReceipt,
    extraction: &HistoricalExtractionReceipt,
) -> Result<AcquisitionExtractionBinding, AcquisitionExtractionError> {
    acquisition
        .validate_against(protocol)
        .map_err(|error| AcquisitionExtractionError::Acquisition(error.to_string()))?;
    let protocol_sha = protocol
        .protocol_sha256()
        .map_err(|error| AcquisitionExtractionError::Protocol(error.to_string()))?;

    if !extraction.protocol_sha256.eq_ignore_ascii_case(&protocol_sha) {
        return Err(AcquisitionExtractionError::ExtractionProtocolMismatch);
    }
    let extraction_receipt_sha = extraction
        .receipt_sha256()
        .map_err(|error| AcquisitionExtractionError::Extraction(error.to_string()))?;

    if !acquisition
        .compressed_snapshot_sha256
        .eq_ignore_ascii_case(&extraction.compressed_dump_sha256)
    {
        return Err(AcquisitionExtractionError::CompressedSnapshotMismatch);
    }

    let acquisition_receipt_sha = acquisition
        .receipt_sha256(protocol)
        .map_err(|error| AcquisitionExtractionError::Acquisition(error.to_string()))?;

    Ok(AcquisitionExtractionBinding {
        protocol_sha256: protocol_sha.to_ascii_lowercase(),
        acquisition_receipt_sha256: acquisition_receipt_sha,
        extraction_receipt_sha256: extraction_receipt_sha,
        compressed_snapshot_sha256: acquisition
            .compressed_snapshot_sha256
            .to_ascii_lowercase(),
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Cross-object acquisition/extraction verification failure.
#[derive(Debug, Error)]
pub enum AcquisitionExtractionError {
    /// Protocol could not be validated/hashed.
    #[error("invalid historical extraction protocol: {0}")]
    Protocol(String),
    /// Acquisition receipt invalid.
    #[error("invalid acquisition receipt: {0}")]
    Acquisition(String),
    /// Extraction receipt invalid.
    #[error("invalid extraction receipt: {0}")]
    Extraction(String),
    /// Extraction receipt names a different protocol.
    #[error("extraction receipt protocol does not match supplied/acquisition protocol")]
    ExtractionProtocolMismatch,
    /// Acquisition and extraction name different compressed archive bytes.
    #[error("acquisition and extraction receipts bind different compressed snapshot bytes")]
    CompressedSnapshotMismatch,
    /// Binding serialization failed.
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
        AcquisitionRoute, HistoricalSnapshotAcquisitionReceipt, hash_snapshot_stream,
    };

    fn hex(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn corpus(protocol: &OqmdExtractionProtocol) -> CompactHistoricalCorpus {
        CompactHistoricalCorpus::from_records(
            protocol,
            vec![NormalizedOqmdRecord {
                entry_id: 1,
                name: "Fe".to_string(),
                element_set: vec!["Fe".to_string()],
                composition_sha256: hex('1'),
                structure_sha256: Some(hex('2')),
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

    fn acquisition(
        protocol: &OqmdExtractionProtocol,
        dump_bytes: &[u8],
    ) -> HistoricalSnapshotAcquisitionReceipt {
        let identity = hash_snapshot_stream(Cursor::new(dump_bytes)).unwrap();
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
            acquisition_tool_sha256: hex('a'),
            execution_environment_sha256: hex('b'),
            transfer_log_sha256: hex('c'),
            route: AcquisitionRoute::LocalMirror {
                mirror_locator: "fixture".to_string(),
                mirror_manifest_sha256: hex('d'),
            },
        }
    }

    fn extraction(
        protocol: &OqmdExtractionProtocol,
        corpus: &CompactHistoricalCorpus,
        dump_sha: &str,
    ) -> HistoricalExtractionReceipt {
        bind_extraction_receipt(
            protocol,
            corpus,
            dump_sha,
            &hex('3'),
            &hex('4'),
            &hex('5'),
            &hex('6'),
            &hex('7'),
            &hex('8'),
            &hex('9'),
            &hex('a'),
        )
        .unwrap()
    }

    #[test]
    fn exact_same_compressed_bytes_bind() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let corpus = corpus(&protocol);
        let acquisition = acquisition(&protocol, b"oqmd-v1.7-fixture");
        let extraction = extraction(
            &protocol,
            &corpus,
            &acquisition.compressed_snapshot_sha256,
        );
        let binding = bind_acquisition_to_extraction(&protocol, &acquisition, &extraction).unwrap();
        assert_eq!(
            binding.compressed_snapshot_sha256,
            acquisition.compressed_snapshot_sha256
        );
        assert_eq!(binding.binding_sha256().unwrap().len(), 64);
    }

    #[test]
    fn different_dump_bytes_fail_even_when_both_receipts_are_well_formed() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let corpus = corpus(&protocol);
        let acquisition = acquisition(&protocol, b"archive-a");
        let other = hash_snapshot_stream(Cursor::new(b"archive-b")).unwrap();
        let extraction = extraction(&protocol, &corpus, &other.sha256);
        assert!(matches!(
            bind_acquisition_to_extraction(&protocol, &acquisition, &extraction),
            Err(AcquisitionExtractionError::CompressedSnapshotMismatch)
        ));
    }

    #[test]
    fn extraction_from_another_protocol_fails() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let corpus = corpus(&protocol);
        let acquisition = acquisition(&protocol, b"archive");
        let mut extraction = extraction(
            &protocol,
            &corpus,
            &acquisition.compressed_snapshot_sha256,
        );
        extraction.protocol_sha256 = hex('f');
        assert!(matches!(
            bind_acquisition_to_extraction(&protocol, &acquisition, &extraction),
            Err(AcquisitionExtractionError::ExtractionProtocolMismatch)
        ));
    }
}
