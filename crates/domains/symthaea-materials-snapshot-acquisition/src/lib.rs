// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming acquisition receipts for historical materials snapshots.
//!
//! This crate binds exact compressed snapshot bytes to a preregistered historical
//! extraction protocol plus transfer/tool/environment evidence. It deliberately
//! stops short of declaring successfully transferred bytes to be authentic provider
//! data; source authenticity requires separate evidence.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use symthaea_materials_historical_extraction::OqmdExtractionProtocol;
use thiserror::Error;

const STREAM_BUFFER_BYTES: usize = 1024 * 1024;

/// How the snapshot bytes entered the acquisition environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcquisitionRoute {
    /// Provider network transfer over HTTPS.
    ProviderHttps {
        /// Locator originally requested.
        requested_url: String,
        /// Final locator after redirects.
        final_url: String,
        /// Final HTTP response status.
        http_status: u16,
        /// SHA-256 of canonical response-header evidence.
        response_headers_sha256: String,
    },
    /// Legacy provider transfer over unauthenticated HTTP.
    ProviderHttpLegacy {
        /// Locator originally requested.
        requested_url: String,
        /// Final locator after redirects.
        final_url: String,
        /// Final HTTP response status.
        http_status: u16,
        /// SHA-256 of canonical response-header evidence.
        response_headers_sha256: String,
    },
    /// Snapshot obtained from a local or mirrored archive.
    LocalMirror {
        /// Stable locator meaningful to the acquisition environment.
        mirror_locator: String,
        /// SHA-256 of the mirror provenance manifest/receipt.
        mirror_manifest_sha256: String,
    },
}

impl AcquisitionRoute {
    fn validate(&self) -> Result<(), SnapshotAcquisitionError> {
        match self {
            Self::ProviderHttps {
                requested_url,
                final_url,
                http_status,
                response_headers_sha256,
            } => {
                validate_http_status(*http_status)?;
                validate_network_locator(requested_url, "https://")?;
                validate_network_locator(final_url, "https://")?;
                sha256(response_headers_sha256)?;
            }
            Self::ProviderHttpLegacy {
                requested_url,
                final_url,
                http_status,
                response_headers_sha256,
            } => {
                validate_http_status(*http_status)?;
                validate_network_locator(requested_url, "http://")?;
                validate_network_locator(final_url, "http://")?;
                sha256(response_headers_sha256)?;
            }
            Self::LocalMirror {
                mirror_locator,
                mirror_manifest_sha256,
            } => {
                nonempty("mirror_locator", mirror_locator)?;
                sha256(mirror_manifest_sha256)?;
            }
        }
        Ok(())
    }

    /// Descriptive transfer-security class; never an authenticity verdict.
    pub fn transport_class(&self) -> TransportClass {
        match self {
            Self::ProviderHttps { .. } => TransportClass::AuthenticatedNetworkTransport,
            Self::ProviderHttpLegacy { .. } => TransportClass::UnauthenticatedNetworkTransport,
            Self::LocalMirror { .. } => TransportClass::MirrorOrLocalCopy,
        }
    }
}

/// Transfer-security class weaker than provider/source authenticity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportClass {
    /// HTTPS transport was used.
    AuthenticatedNetworkTransport,
    /// Network transport lacked HTTPS authentication.
    UnauthenticatedNetworkTransport,
    /// Bytes came from a separately receipted mirror/local copy.
    MirrorOrLocalCopy,
}

/// Content-addressed acquisition receipt for one compressed historical snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalSnapshotAcquisitionReceipt {
    /// Receipt schema version.
    pub schema_version: u32,
    /// Exact preregistered extraction protocol digest.
    pub protocol_sha256: String,
    /// Provider copied from the bound protocol.
    pub provider: String,
    /// Historical database version copied from the bound protocol.
    pub database_version: String,
    /// Expected source dump filename copied from the bound protocol.
    pub dump_filename: String,
    /// Source license identifier copied from the bound protocol.
    pub source_license: String,
    /// UTC acquisition timestamp in `YYYY-MM-DDTHH:MM:SSZ` form.
    pub acquired_at_utc: String,
    /// SHA-256 over exact compressed bytes.
    pub compressed_snapshot_sha256: String,
    /// Exact compressed byte count.
    pub compressed_snapshot_bytes: u64,
    /// Acquisition/downloader implementation artifact digest.
    pub acquisition_tool_sha256: String,
    /// Exact Nix/container/execution environment artifact digest.
    pub execution_environment_sha256: String,
    /// Canonical transfer log artifact digest.
    pub transfer_log_sha256: String,
    /// Transfer/mirror evidence.
    pub route: AcquisitionRoute,
}

impl HistoricalSnapshotAcquisitionReceipt {
    /// Validate this receipt against the exact extraction protocol.
    pub fn validate_against(
        &self,
        protocol: &OqmdExtractionProtocol,
    ) -> Result<(), SnapshotAcquisitionError> {
        protocol
            .validate()
            .map_err(|error| SnapshotAcquisitionError::Protocol(error.to_string()))?;
        if self.schema_version != 1 {
            return Err(SnapshotAcquisitionError::UnsupportedReceiptSchema(
                self.schema_version,
            ));
        }
        let protocol_sha = protocol
            .protocol_sha256()
            .map_err(|error| SnapshotAcquisitionError::Protocol(error.to_string()))?;
        if self.protocol_sha256 != protocol_sha {
            return Err(SnapshotAcquisitionError::ProtocolDigestMismatch);
        }
        for (field, actual, expected) in [
            ("provider", self.provider.as_str(), protocol.provider.as_str()),
            (
                "database_version",
                self.database_version.as_str(),
                protocol.database_version.as_str(),
            ),
            (
                "dump_filename",
                self.dump_filename.as_str(),
                protocol.dump_filename.as_str(),
            ),
            (
                "source_license",
                self.source_license.as_str(),
                protocol.source_license.as_str(),
            ),
        ] {
            if actual != expected {
                return Err(SnapshotAcquisitionError::ProtocolFieldMismatch(field));
            }
        }
        validate_utc_timestamp(&self.acquired_at_utc)?;
        sha256(&self.compressed_snapshot_sha256)?;
        if self.compressed_snapshot_bytes == 0 {
            return Err(SnapshotAcquisitionError::EmptySnapshot);
        }
        sha256(&self.acquisition_tool_sha256)?;
        sha256(&self.execution_environment_sha256)?;
        sha256(&self.transfer_log_sha256)?;
        self.route.validate()
    }

    /// Deterministic digest of the entire validated receipt.
    pub fn receipt_sha256(
        &self,
        protocol: &OqmdExtractionProtocol,
    ) -> Result<String, SnapshotAcquisitionError> {
        self.validate_against(protocol)?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Identity measured directly from a snapshot byte stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamedSnapshotIdentity {
    /// SHA-256 over the exact bytes observed.
    pub sha256: String,
    /// Exact bytes observed.
    pub bytes: u64,
}

/// Stream a snapshot through SHA-256 without loading it wholly into memory.
pub fn hash_snapshot_stream<R: Read>(
    mut reader: R,
) -> Result<StreamedSnapshotIdentity, SnapshotAcquisitionError> {
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; STREAM_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        total = total
            .checked_add(read as u64)
            .ok_or(SnapshotAcquisitionError::ByteCountOverflow)?;
    }
    if total == 0 {
        return Err(SnapshotAcquisitionError::EmptySnapshot);
    }
    Ok(StreamedSnapshotIdentity {
        sha256: format!("{:x}", hasher.finalize()),
        bytes: total,
    })
}

/// Re-hash actual snapshot bytes and require identity with the receipt.
pub fn verify_snapshot_stream<R: Read>(
    protocol: &OqmdExtractionProtocol,
    receipt: &HistoricalSnapshotAcquisitionReceipt,
    reader: R,
) -> Result<StreamedSnapshotIdentity, SnapshotAcquisitionError> {
    receipt.validate_against(protocol)?;
    let identity = hash_snapshot_stream(reader)?;
    if !identity
        .sha256
        .eq_ignore_ascii_case(&receipt.compressed_snapshot_sha256)
    {
        return Err(SnapshotAcquisitionError::SnapshotDigestMismatch);
    }
    if identity.bytes != receipt.compressed_snapshot_bytes {
        return Err(SnapshotAcquisitionError::SnapshotByteCountMismatch {
            expected: receipt.compressed_snapshot_bytes,
            observed: identity.bytes,
        });
    }
    Ok(identity)
}

fn validate_http_status(status: u16) -> Result<(), SnapshotAcquisitionError> {
    if !(200..=299).contains(&status) {
        return Err(SnapshotAcquisitionError::UnexpectedHttpStatus(status));
    }
    Ok(())
}

fn validate_network_locator(
    value: &str,
    required_prefix: &str,
) -> Result<(), SnapshotAcquisitionError> {
    nonempty("network_locator", value)?;
    if !value.is_ascii()
        || value
            .chars()
            .any(|ch| matches!(ch, '\n' | '\r' | '\0'))
        || !value.starts_with(required_prefix)
    {
        return Err(SnapshotAcquisitionError::InvalidNetworkLocator(
            value.to_string(),
        ));
    }
    let remainder = &value[required_prefix.len()..];
    let authority = remainder.split('/').next().unwrap_or_default();
    if authority.is_empty()
        || authority.contains('@')
        || value.contains('?')
        || value.contains('#')
    {
        return Err(SnapshotAcquisitionError::InvalidNetworkLocator(
            value.to_string(),
        ));
    }
    Ok(())
}

fn validate_utc_timestamp(value: &str) -> Result<(), SnapshotAcquisitionError> {
    let bytes = value.as_bytes();
    if !value.is_ascii()
        || bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return Err(SnapshotAcquisitionError::InvalidTimestamp(value.to_string()));
    }
    let year = parse_decimal(&bytes[0..4])? as u16;
    let month = parse_decimal(&bytes[5..7])? as u8;
    let day = parse_decimal(&bytes[8..10])? as u8;
    let hour = parse_decimal(&bytes[11..13])? as u8;
    let minute = parse_decimal(&bytes[14..16])? as u8;
    let second = parse_decimal(&bytes[17..19])? as u8;
    if year < 1900
        || !(1..=12).contains(&month)
        || day == 0
        || day > days_in_month(year, month)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(SnapshotAcquisitionError::InvalidTimestamp(value.to_string()));
    }
    Ok(())
}

fn parse_decimal(bytes: &[u8]) -> Result<u32, SnapshotAcquisitionError> {
    if bytes.is_empty() || !bytes.iter().all(|byte| byte.is_ascii_digit()) {
        return Err(SnapshotAcquisitionError::InvalidTimestamp(
            "non-decimal timestamp field".to_string(),
        ));
    }
    let mut value = 0_u32;
    for digit in bytes {
        value = value * 10 + u32::from(*digit - b'0');
    }
    Ok(value)
}

fn days_in_month(year: u16, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: u16) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn nonempty(name: &'static str, value: &str) -> Result<(), SnapshotAcquisitionError> {
    if value.trim().is_empty() {
        return Err(SnapshotAcquisitionError::EmptyField(name));
    }
    Ok(())
}

fn sha256(value: &str) -> Result<(), SnapshotAcquisitionError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(SnapshotAcquisitionError::InvalidSha256(value.to_string()));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Snapshot acquisition/verification failure.
#[derive(Debug, Error)]
pub enum SnapshotAcquisitionError {
    /// Extraction protocol invalid.
    #[error("invalid extraction protocol: {0}")]
    Protocol(String),
    /// Receipt schema unsupported.
    #[error("unsupported acquisition receipt schema {0}")]
    UnsupportedReceiptSchema(u32),
    /// Protocol digest differs.
    #[error("acquisition receipt protocol digest does not match supplied protocol")]
    ProtocolDigestMismatch,
    /// Human-readable protocol field differs.
    #[error("acquisition receipt field {0} does not match supplied protocol")]
    ProtocolFieldMismatch(&'static str),
    /// Required string empty.
    #[error("required field {0} is empty")]
    EmptyField(&'static str),
    /// SHA malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Snapshot contained no bytes.
    #[error("historical snapshot stream is empty")]
    EmptySnapshot,
    /// Exact byte counter overflowed.
    #[error("snapshot byte counter overflowed")]
    ByteCountOverflow,
    /// Network locator malformed or embeds userinfo/query/fragment data.
    #[error("invalid acquisition network locator: {0}")]
    InvalidNetworkLocator(String),
    /// Final response was not a successful 2xx transfer.
    #[error("unexpected acquisition HTTP status {0}")]
    UnexpectedHttpStatus(u16),
    /// UTC timestamp malformed.
    #[error("invalid acquisition UTC timestamp: {0}")]
    InvalidTimestamp(String),
    /// Snapshot digest differs from receipt.
    #[error("snapshot bytes do not match acquisition receipt SHA-256")]
    SnapshotDigestMismatch,
    /// Snapshot byte count differs from receipt.
    #[error("snapshot byte count mismatch: expected {expected}, observed {observed}")]
    SnapshotByteCountMismatch {
        /// Receipt count.
        expected: u64,
        /// Re-observed count.
        observed: u64,
    },
    /// Stream read failed.
    #[error("failed while reading snapshot stream: {0}")]
    Io(#[from] std::io::Error),
    /// Receipt serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use symthaea_materials_historical_extraction::oqmd_v17_fe_co_zr_protocol;

    fn hex(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn receipt_for(bytes: &[u8]) -> HistoricalSnapshotAcquisitionReceipt {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let identity = hash_snapshot_stream(Cursor::new(bytes)).unwrap();
        HistoricalSnapshotAcquisitionReceipt {
            schema_version: 1,
            protocol_sha256: protocol.protocol_sha256().unwrap(),
            provider: protocol.provider.clone(),
            database_version: protocol.database_version.clone(),
            dump_filename: protocol.dump_filename.clone(),
            source_license: protocol.source_license.clone(),
            acquired_at_utc: "2026-09-19T18:00:00Z".to_string(),
            compressed_snapshot_sha256: identity.sha256,
            compressed_snapshot_bytes: identity.bytes,
            acquisition_tool_sha256: hex('a'),
            execution_environment_sha256: hex('b'),
            transfer_log_sha256: hex('c'),
            route: AcquisitionRoute::ProviderHttps {
                requested_url: "https://oqmd.org/download/qmdb.sql.gz".to_string(),
                final_url: "https://oqmd.org/download/qmdb.sql.gz".to_string(),
                http_status: 200,
                response_headers_sha256: hex('d'),
            },
        }
    }

    #[test]
    fn streaming_identity_and_receipt_round_trip() {
        let bytes = b"historical-snapshot-fixture";
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let receipt = receipt_for(bytes);
        let observed = verify_snapshot_stream(&protocol, &receipt, Cursor::new(bytes)).unwrap();
        assert_eq!(observed.bytes, bytes.len() as u64);
        assert_eq!(observed.sha256, receipt.compressed_snapshot_sha256);
        assert_eq!(receipt.receipt_sha256(&protocol).unwrap().len(), 64);
    }

    #[test]
    fn altered_snapshot_bytes_fail_digest_check() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let receipt = receipt_for(b"expected");
        assert!(matches!(
            verify_snapshot_stream(&protocol, &receipt, Cursor::new(b"tampered")),
            Err(SnapshotAcquisitionError::SnapshotDigestMismatch)
        ));
    }

    #[test]
    fn correct_digest_with_wrong_byte_count_fails_independently() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let bytes = b"same-bytes";
        let mut receipt = receipt_for(bytes);
        receipt.compressed_snapshot_bytes += 1;
        assert!(matches!(
            verify_snapshot_stream(&protocol, &receipt, Cursor::new(bytes)),
            Err(SnapshotAcquisitionError::SnapshotByteCountMismatch { .. })
        ));
    }

    #[test]
    fn copied_receipt_for_wrong_protocol_is_rejected() {
        let mut protocol = oqmd_v17_fe_co_zr_protocol();
        let receipt = receipt_for(b"bytes");
        protocol.database_version = "1.8".to_string();
        assert!(matches!(
            receipt.validate_against(&protocol),
            Err(SnapshotAcquisitionError::ProtocolDigestMismatch)
        ));
    }

    #[test]
    fn locator_rejects_userinfo_queries_and_unresolved_redirects() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let mut receipt = receipt_for(b"bytes");
        receipt.route = AcquisitionRoute::ProviderHttps {
            requested_url: "https://user:token@oqmd.org/file".to_string(),
            final_url: "https://oqmd.org/file?token=secret".to_string(),
            http_status: 302,
            response_headers_sha256: hex('d'),
        };
        assert!(receipt.validate_against(&protocol).is_err());
    }

    #[test]
    fn timestamp_validation_handles_calendar_edges() {
        assert!(validate_utc_timestamp("2024-02-29T23:59:59Z").is_ok());
        assert!(validate_utc_timestamp("2025-02-29T00:00:00Z").is_err());
        assert!(validate_utc_timestamp("2026-02-30T00:00:00Z").is_err());
        assert!(validate_utc_timestamp("２０２６-09-19T18:00:00Z").is_err());
    }

    #[test]
    fn mirror_transport_is_not_upgraded_to_network_authenticity() {
        let protocol = oqmd_v17_fe_co_zr_protocol();
        let mut receipt = receipt_for(b"bytes");
        receipt.route = AcquisitionRoute::LocalMirror {
            mirror_locator: "offline-archive/oqmd-v1.7".to_string(),
            mirror_manifest_sha256: hex('e'),
        };
        receipt.validate_against(&protocol).unwrap();
        assert_eq!(
            receipt.route.transport_class(),
            TransportClass::MirrorOrLocalCopy
        );
    }
}
