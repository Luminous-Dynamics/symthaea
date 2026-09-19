// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! External materials-data provider adapters with offline-reproducible provenance.
//!
//! The core `symthaea-materials` crate stays deterministic and network-free. This
//! crate owns provider schemas, raw-response capture metadata, normalization, and
//! data-license provenance. Provider output may populate materials evidence
//! contracts, but it never promotes scientific authority by itself.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod oqmd;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Stable data-license metadata attached to imported provider evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataLicenseRef {
    /// SPDX-like or provider-defined identifier, for example `CC-BY-4.0`.
    pub identifier: String,
    /// Human-readable canonical license URL or provider terms URL.
    pub url: String,
    /// Attribution/source text that must survive export.
    pub attribution: String,
}

/// Exact provider query that produced a captured response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderQueryManifest {
    /// Stable endpoint/profile identifier.
    pub endpoint_id: String,
    /// Exact provider query/path representation after secret redaction.
    pub representation: String,
}

/// Provenance for one immutable raw provider response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderRawCapture {
    /// Stable provider identifier.
    pub provider_id: String,
    /// Provider API/schema version reported or bound by the adapter.
    pub provider_schema_version: String,
    /// Exact query that produced this capture.
    pub query: ProviderQueryManifest,
    /// Retrieval timestamp as supplied by the capture process/provider.
    pub fetched_at_utc: String,
    /// SHA-256 of the exact raw response bytes.
    pub raw_sha256: String,
    /// Source URL with sensitive query values redacted.
    pub source_url: String,
    /// License/terms governing the imported data.
    pub license: DataLicenseRef,
}

impl ProviderRawCapture {
    /// Construct capture metadata and bind it to exact raw response bytes.
    pub fn new(
        provider_id: String,
        provider_schema_version: String,
        endpoint_id: String,
        query_representation: String,
        fetched_at_utc: String,
        source_url: String,
        license: DataLicenseRef,
        raw: &[u8],
    ) -> Result<Self, ProviderError> {
        nonempty("provider_id", &provider_id)?;
        nonempty("provider_schema_version", &provider_schema_version)?;
        nonempty("endpoint_id", &endpoint_id)?;
        nonempty("query_representation", &query_representation)?;
        nonempty("fetched_at_utc", &fetched_at_utc)?;
        validate_license(&license)?;
        Ok(Self {
            provider_id,
            provider_schema_version,
            query: ProviderQueryManifest {
                endpoint_id,
                representation: redact_query_secrets(&query_representation),
            },
            fetched_at_utc,
            raw_sha256: sha256_hex(raw),
            source_url: redact_query_secrets(&source_url),
            license,
        })
    }

    /// Verify that supplied raw bytes are exactly the captured response.
    pub fn verify_raw(&self, raw: &[u8]) -> Result<(), ProviderError> {
        if self.raw_sha256 != sha256_hex(raw) {
            return Err(ProviderError::RawDigestMismatch);
        }
        Ok(())
    }
}

/// Normalize common credential-bearing query values before provenance/logging.
pub fn redact_query_secrets(input: &str) -> String {
    let Some((prefix, query)) = input.split_once('?') else {
        return input.to_string();
    };
    let redacted = query
        .split('&')
        .map(|pair| {
            let Some((key, value)) = pair.split_once('=') else {
                return pair.to_string();
            };
            if sensitive_key(key) {
                format!("{key}=[REDACTED]")
            } else {
                format!("{key}={value}")
            }
        })
        .collect::<Vec<_>>()
        .join("&");
    format!("{prefix}?{redacted}")
}

fn sensitive_key(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "api_key"
            | "apikey"
            | "key"
            | "token"
            | "access_token"
            | "secret"
            | "password"
            | "authorization"
    )
}

/// SHA-256 as lowercase hexadecimal.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

fn validate_license(value: &DataLicenseRef) -> Result<(), ProviderError> {
    nonempty("license identifier", &value.identifier)?;
    nonempty("license url", &value.url)?;
    nonempty("license attribution", &value.attribution)
}

fn nonempty(field: &'static str, value: &str) -> Result<(), ProviderError> {
    if value.trim().is_empty() {
        Err(ProviderError::EmptyField(field))
    } else {
        Ok(())
    }
}

/// Provider provenance/normalization failure.
#[derive(Debug, Error)]
pub enum ProviderError {
    /// Required field was empty.
    #[error("required provider field is empty: {0}")]
    EmptyField(&'static str),
    /// Raw bytes no longer match the bound capture digest.
    #[error("raw response digest does not match capture metadata")]
    RawDigestMismatch,
    /// Provider schema changed or fixture/response does not satisfy this adapter profile.
    #[error("provider schema/profile mismatch: {0}")]
    SchemaMismatch(String),
    /// Provider metadata disagrees with capture metadata.
    #[error("provider capture metadata mismatch: {0}")]
    CaptureMetadataMismatch(&'static str),
    /// Provider-specific normalization failed.
    #[error("normalization failed: {0}")]
    Normalization(String),
    /// JSON serialization/deserialization failed.
    #[error("provider JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn license() -> DataLicenseRef {
        DataLicenseRef {
            identifier: "CC-BY-4.0".to_string(),
            url: "https://creativecommons.org/licenses/by/4.0/".to_string(),
            attribution: "OQMD / Wolverton Group, Northwestern University".to_string(),
        }
    }

    #[test]
    fn raw_capture_binds_exact_bytes() {
        let raw = br#"{\"ok\":true}"#;
        let capture = ProviderRawCapture::new(
            "oqmd".to_string(),
            "1.0".to_string(),
            "formationenergy-v1".to_string(),
            "/formationenergy?fields=name,entry_id,delta_e".to_string(),
            "2026-09-19T00:00:00Z".to_string(),
            "https://oqmd.org/oqmdapi/formationenergy?fields=name,entry_id,delta_e".to_string(),
            license(),
            raw,
        )
        .unwrap();
        capture.verify_raw(raw).unwrap();
        assert_eq!(capture.raw_sha256.len(), 64);
        assert!(matches!(
            capture.verify_raw(b"different"),
            Err(ProviderError::RawDigestMismatch)
        ));
    }

    #[test]
    fn query_secrets_are_redacted_before_provenance() {
        let input = "https://example.invalid/api?api_key=secret123&formula=Fe2O3&token=abc";
        let redacted = redact_query_secrets(input);
        assert!(!redacted.contains("secret123"));
        assert!(!redacted.contains("token=abc"));
        assert!(redacted.contains("api_key=[REDACTED]"));
        assert!(redacted.contains("formula=Fe2O3"));
    }
}
