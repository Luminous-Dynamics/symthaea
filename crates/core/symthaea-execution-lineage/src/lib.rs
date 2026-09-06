// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Authoritative execution-lineage identity for Recursive Cognitive Architecture v1.
//!
//! This crate answers one narrow question:
//!
//! ```text
//! which exact cognitive execution produced this observation?
//! ```
//!
//! It deliberately does **not** decide whether the identified source generation
//! is qualified, whether an observation is epistemically admissible, or whether
//! any downstream action is authorized.
//!
//! The issuer requires an externally supplied source-generation commitment,
//! commits the exact caller-provided cognitive configuration bytes and genesis
//! material, and mints a fresh execution-instance nonce from operating-system
//! entropy. Entropy failure is fatal: there is no deterministic, timestamp,
//! PID, package-version, or genesis-derived fallback for execution identity.

#![deny(unsafe_code)]

use serde::{Deserialize, Deserializer, Serialize};

pub const EXECUTION_LINEAGE_SCHEMA_VERSION: u16 = 1;
pub const EXECUTION_LINEAGE_ISSUER_PROFILE_V1: &str = "symthaea-cognitive-execution-lineage-v1";

const DOMAIN_CONFIG: &[u8] = b"symthaea:execution-lineage:config:v1\0";
const DOMAIN_GENESIS: &[u8] = b"symthaea:execution-lineage:genesis:v1\0";
const DOMAIN_NONCE: &[u8] = b"symthaea:execution-lineage:nonce:v1\0";
const DOMAIN_LINEAGE: &[u8] = b"symthaea:execution-lineage:record:v1\0";

/// Committed fields that define one cognitive execution lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionLineageBodyV1 {
    schema_version: u16,
    issuer_profile: String,
    /// Exact source/build/tree generation supplied by an outer evidence boundary.
    /// This crate validates its commitment shape but does not self-certify its authority.
    source_generation_digest: String,
    /// Named serialization/projection profile used to produce `config_bytes`.
    config_profile: String,
    /// Commitment to the exact adapter/schema contract for `config_profile`.
    config_profile_digest: String,
    /// Commitment to the exact caller-provided configuration bytes.
    config_digest: String,
    /// Domain-separated commitment to explicit genesis material, including
    /// a distinction between no genesis and an explicitly empty genesis value.
    genesis_commitment: String,
    /// Commitment to a fresh 256-bit OS-entropy execution nonce.
    execution_nonce_commitment: String,
}

/// Immutable, persistence-safe identity for one concrete cognitive execution.
///
/// The raw execution nonce is intentionally not retained. Persisted records
/// revalidate all commitment shapes and recompute `lineage_digest` on load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveExecutionLineageV1 {
    body: ExecutionLineageBodyV1,
    lineage_digest: String,
}

impl CognitiveExecutionLineageV1 {
    /// Issue a fresh execution lineage.
    ///
    /// `source_generation_digest` must come from the outer build/evidence
    /// boundary. It is never synthesized from package version, wall time, or
    /// genesis material here.
    pub fn issue(
        source_generation_digest: &str,
        config_profile: &str,
        config_profile_digest: &str,
        config_bytes: &[u8],
        genesis_material: Option<&[u8]>,
    ) -> Result<Self, ExecutionLineageError> {
        let mut nonce = [0_u8; 32];
        getrandom::getrandom(&mut nonce)
            .map_err(|error| ExecutionLineageError::EntropyUnavailable(error.to_string()))?;

        Self::issue_with_nonce(
            source_generation_digest,
            config_profile,
            config_profile_digest,
            config_bytes,
            genesis_material,
            nonce,
        )
    }

    fn issue_with_nonce(
        source_generation_digest: &str,
        config_profile: &str,
        config_profile_digest: &str,
        config_bytes: &[u8],
        genesis_material: Option<&[u8]>,
        nonce: [u8; 32],
    ) -> Result<Self, ExecutionLineageError> {
        validate_digest("source_generation_digest", source_generation_digest)?;
        validate_digest("config_profile_digest", config_profile_digest)?;
        require_nonempty_config_profile(config_profile)?;

        let config_digest = domain_hash(DOMAIN_CONFIG, config_bytes);
        let genesis_commitment = genesis_commitment(genesis_material);
        let execution_nonce_commitment = domain_hash(DOMAIN_NONCE, &nonce);

        let body = ExecutionLineageBodyV1 {
            schema_version: EXECUTION_LINEAGE_SCHEMA_VERSION,
            issuer_profile: EXECUTION_LINEAGE_ISSUER_PROFILE_V1.to_string(),
            source_generation_digest: source_generation_digest.to_string(),
            config_profile: config_profile.to_string(),
            config_profile_digest: config_profile_digest.to_string(),
            config_digest,
            genesis_commitment,
            execution_nonce_commitment,
        };
        let lineage_digest = compute_lineage_digest(&body);
        let lineage = Self {
            body,
            lineage_digest,
        };
        lineage.validate_integrity()?;
        Ok(lineage)
    }

    pub fn lineage_digest(&self) -> &str {
        &self.lineage_digest
    }

    pub fn source_generation_digest(&self) -> &str {
        &self.body.source_generation_digest
    }

    pub fn config_profile(&self) -> &str {
        &self.body.config_profile
    }

    pub fn config_profile_digest(&self) -> &str {
        &self.body.config_profile_digest
    }

    pub fn config_digest(&self) -> &str {
        &self.body.config_digest
    }

    pub fn genesis_commitment(&self) -> &str {
        &self.body.genesis_commitment
    }

    pub fn execution_nonce_commitment(&self) -> &str {
        &self.body.execution_nonce_commitment
    }

    fn validate_integrity(&self) -> Result<(), ExecutionLineageError> {
        if self.body.schema_version != EXECUTION_LINEAGE_SCHEMA_VERSION {
            return Err(ExecutionLineageError::UnsupportedSchemaVersion {
                found: self.body.schema_version,
            });
        }
        if self.body.issuer_profile != EXECUTION_LINEAGE_ISSUER_PROFILE_V1 {
            return Err(ExecutionLineageError::UnexpectedIssuerProfile {
                found: self.body.issuer_profile.clone(),
            });
        }
        require_nonempty_config_profile(&self.body.config_profile)?;

        for (field, digest) in [
            (
                "source_generation_digest",
                self.body.source_generation_digest.as_str(),
            ),
            (
                "config_profile_digest",
                self.body.config_profile_digest.as_str(),
            ),
            ("config_digest", self.body.config_digest.as_str()),
            (
                "genesis_commitment",
                self.body.genesis_commitment.as_str(),
            ),
            (
                "execution_nonce_commitment",
                self.body.execution_nonce_commitment.as_str(),
            ),
            ("lineage_digest", self.lineage_digest.as_str()),
        ] {
            validate_digest(field, digest)?;
        }

        let expected = compute_lineage_digest(&self.body);
        if self.lineage_digest != expected {
            return Err(ExecutionLineageError::IntegrityMismatch);
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for CognitiveExecutionLineageV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct PersistedLineageV1 {
            body: ExecutionLineageBodyV1,
            lineage_digest: String,
        }

        let persisted = PersistedLineageV1::deserialize(deserializer)?;
        let lineage = CognitiveExecutionLineageV1 {
            body: persisted.body,
            lineage_digest: persisted.lineage_digest,
        };
        lineage
            .validate_integrity()
            .map_err(serde::de::Error::custom)?;
        Ok(lineage)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionLineageError {
    UnsupportedSchemaVersion { found: u16 },
    UnexpectedIssuerProfile { found: String },
    MissingConfigProfile,
    MalformedDigest { field: &'static str },
    EntropyUnavailable(String),
    IntegrityMismatch,
}

impl std::fmt::Display for ExecutionLineageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported execution-lineage schema version {found}; expected {EXECUTION_LINEAGE_SCHEMA_VERSION}"
            ),
            Self::UnexpectedIssuerProfile { found } => write!(
                f,
                "unexpected execution-lineage issuer profile {found:?}; expected {EXECUTION_LINEAGE_ISSUER_PROFILE_V1:?}"
            ),
            Self::MissingConfigProfile => {
                f.write_str("execution lineage requires a non-empty config profile")
            }
            Self::MalformedDigest { field } => write!(
                f,
                "{field} must be sha256:<64 hex> or blake3:<64 hex>"
            ),
            Self::EntropyUnavailable(error) => {
                write!(f, "cannot issue execution lineage without OS entropy: {error}")
            }
            Self::IntegrityMismatch => {
                f.write_str("execution-lineage commitment does not match its committed body")
            }
        }
    }
}

impl std::error::Error for ExecutionLineageError {}

fn require_nonempty_config_profile(profile: &str) -> Result<(), ExecutionLineageError> {
    if profile.trim().is_empty() {
        Err(ExecutionLineageError::MissingConfigProfile)
    } else {
        Ok(())
    }
}

fn validate_digest(field: &'static str, digest: &str) -> Result<(), ExecutionLineageError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(ExecutionLineageError::MalformedDigest { field });
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ExecutionLineageError::MalformedDigest { field });
    }
    Ok(())
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn genesis_commitment(genesis_material: Option<&[u8]>) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DOMAIN_GENESIS);
    match genesis_material {
        None => hasher.update(b"none\0"),
        Some(bytes) => {
            hasher.update(b"some\0");
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        }
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn compute_lineage_digest(body: &ExecutionLineageBodyV1) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DOMAIN_LINEAGE);
    hasher.update(&body.schema_version.to_le_bytes());
    hash_text_field(&mut hasher, b"issuer_profile", &body.issuer_profile);
    hash_text_field(
        &mut hasher,
        b"source_generation_digest",
        &body.source_generation_digest,
    );
    hash_text_field(&mut hasher, b"config_profile", &body.config_profile);
    hash_text_field(
        &mut hasher,
        b"config_profile_digest",
        &body.config_profile_digest,
    );
    hash_text_field(&mut hasher, b"config_digest", &body.config_digest);
    hash_text_field(
        &mut hasher,
        b"genesis_commitment",
        &body.genesis_commitment,
    );
    hash_text_field(
        &mut hasher,
        b"execution_nonce_commitment",
        &body.execution_nonce_commitment,
    );
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_text_field(hasher: &mut blake3::Hasher, label: &[u8], value: &str) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_A: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SOURCE_B: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const PROFILE_A: &str =
        "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const PROFILE_B: &str =
        "blake3:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn deterministic(
        source: &str,
        profile_digest: &str,
        config: &[u8],
        genesis: Option<&[u8]>,
        nonce: [u8; 32],
    ) -> CognitiveExecutionLineageV1 {
        CognitiveExecutionLineageV1::issue_with_nonce(
            source,
            "cognitive-loop-config-serde-json-v1",
            profile_digest,
            config,
            genesis,
            nonce,
        )
        .unwrap()
    }

    #[test]
    fn identical_committed_inputs_and_nonce_are_stable() {
        let a = deterministic(SOURCE_A, PROFILE_A, b"{\"a\":1}", Some(b"genesis"), [7; 32]);
        let b = deterministic(SOURCE_A, PROFILE_A, b"{\"a\":1}", Some(b"genesis"), [7; 32]);
        assert_eq!(a, b);
    }

    #[test]
    fn fresh_execution_nonce_separates_otherwise_identical_runs() {
        let a = deterministic(SOURCE_A, PROFILE_A, b"same-config", Some(b"same"), [1; 32]);
        let b = deterministic(SOURCE_A, PROFILE_A, b"same-config", Some(b"same"), [2; 32]);
        assert_ne!(a.execution_nonce_commitment(), b.execution_nonce_commitment());
        assert_ne!(a.lineage_digest(), b.lineage_digest());
    }

    #[test]
    fn source_generation_is_part_of_execution_identity() {
        let a = deterministic(SOURCE_A, PROFILE_A, b"same", None, [9; 32]);
        let b = deterministic(SOURCE_B, PROFILE_A, b"same", None, [9; 32]);
        assert_ne!(a.lineage_digest(), b.lineage_digest());
    }

    #[test]
    fn config_bytes_are_committed_exactly() {
        let a = deterministic(SOURCE_A, PROFILE_A, b"{\"a\":1}", None, [9; 32]);
        let b = deterministic(SOURCE_A, PROFILE_A, b"{\"a\":2}", None, [9; 32]);
        assert_ne!(a.config_digest(), b.config_digest());
        assert_ne!(a.lineage_digest(), b.lineage_digest());
    }

    #[test]
    fn config_projection_semantics_are_bound_separately_from_config_bytes() {
        let a = deterministic(SOURCE_A, PROFILE_A, b"same", None, [9; 32]);
        let b = deterministic(SOURCE_A, PROFILE_B, b"same", None, [9; 32]);
        assert_ne!(a.config_profile_digest(), b.config_profile_digest());
        assert_ne!(a.lineage_digest(), b.lineage_digest());
    }

    #[test]
    fn absent_and_explicitly_empty_genesis_are_distinct() {
        let absent = deterministic(SOURCE_A, PROFILE_A, b"same", None, [9; 32]);
        let empty = deterministic(SOURCE_A, PROFILE_A, b"same", Some(b""), [9; 32]);
        assert_ne!(absent.genesis_commitment(), empty.genesis_commitment());
        assert_ne!(absent.lineage_digest(), empty.lineage_digest());
    }

    #[test]
    fn malformed_source_and_profile_commitments_fail_closed() {
        assert_eq!(
            CognitiveExecutionLineageV1::issue_with_nonce(
                "git:abc123",
                "profile",
                PROFILE_A,
                b"config",
                None,
                [1; 32],
            ),
            Err(ExecutionLineageError::MalformedDigest {
                field: "source_generation_digest"
            })
        );
        assert_eq!(
            CognitiveExecutionLineageV1::issue_with_nonce(
                SOURCE_A,
                "profile",
                "decorative",
                b"config",
                None,
                [1; 32],
            ),
            Err(ExecutionLineageError::MalformedDigest {
                field: "config_profile_digest"
            })
        );
    }

    #[test]
    fn empty_config_profile_fails_closed() {
        assert_eq!(
            CognitiveExecutionLineageV1::issue_with_nonce(
                SOURCE_A,
                "   ",
                PROFILE_A,
                b"config",
                None,
                [1; 32],
            ),
            Err(ExecutionLineageError::MissingConfigProfile)
        );
    }

    #[test]
    fn persistence_revalidates_lineage_commitment() {
        let valid = deterministic(SOURCE_A, PROFILE_A, b"config", None, [4; 32]);
        let encoded = serde_json::to_string(&valid).unwrap();
        let decoded: CognitiveExecutionLineageV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, valid);

        let mut value = serde_json::to_value(&valid).unwrap();
        value["body"]["source_generation_digest"] = serde_json::json!(SOURCE_B);
        assert!(serde_json::from_value::<CognitiveExecutionLineageV1>(value).is_err());
    }

    #[test]
    fn persistence_rejects_lineage_digest_tampering() {
        let valid = deterministic(SOURCE_A, PROFILE_A, b"config", None, [4; 32]);
        let mut value = serde_json::to_value(&valid).unwrap();
        value["lineage_digest"] = serde_json::json!(SOURCE_B);
        assert!(serde_json::from_value::<CognitiveExecutionLineageV1>(value).is_err());
    }
}
