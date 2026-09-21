// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared semantic identity transcript for evidence-like artifacts.
//!
//! This crate freezes canonical preimage bytes and the identifier shape. It
//! deliberately does not own a cryptographic implementation or grant source
//! authority. Domain-native adapters validate/rederive their evidence, hash the
//! exact preimage with reviewed SHA-256 machinery, and then construct the ID.

use std::fmt;

use serde::{Deserialize, Serialize};

pub const SEMANTIC_EVIDENCE_DOMAIN_V1: &[u8] = b"melothaea:semantic-evidence:v1";
pub const SEMANTIC_TRANSCRIPT_VERSION_V1: &str = "melothaea-semantic-transcript-v1";
pub const SEMANTIC_DIGEST_ALGORITHM_SHA256: &str = "sha256";
pub const SEMANTIC_NO_PROFILE_V1: &str = "none";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum SemanticFieldKind {
    Utf8 = 0x01,
    Bytes = 0x02,
    UnsignedInteger = 0x03,
    SignedInteger = 0x04,
    FiniteF64 = 0x05,
    Sequence = 0x06,
    Boolean = 0x07,
    DigestReference = 0x08,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticTranscriptError {
    TagNotStrictlyIncreasing { previous: u16, next: u16 },
    NonFiniteFloat,
    EmptyIdentityField(&'static str),
    UnsupportedDigestAlgorithm(String),
    UnsupportedTranscriptVersion(String),
    InvalidDigestHex,
}

impl fmt::Display for SemanticTranscriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TagNotStrictlyIncreasing { previous, next } => write!(
                f,
                "semantic transcript tags must be strictly increasing: previous={previous}, next={next}"
            ),
            Self::NonFiniteFloat => write!(f, "semantic transcript rejects NaN and infinity"),
            Self::EmptyIdentityField(field) => {
                write!(f, "semantic evidence identity field {field} must not be empty")
            }
            Self::UnsupportedDigestAlgorithm(value) => write!(
                f,
                "unsupported semantic evidence digest algorithm: {value}"
            ),
            Self::UnsupportedTranscriptVersion(value) => write!(
                f,
                "unsupported semantic transcript version: {value}"
            ),
            Self::InvalidDigestHex => write!(
                f,
                "semantic evidence digest must be exactly 64 lower-case hexadecimal characters"
            ),
        }
    }
}

impl std::error::Error for SemanticTranscriptError {}

/// Canonical typed transcript builder for one semantic payload.
///
/// Tags must be strictly increasing, which makes field order explicit and
/// prevents duplicate tags. Maps are intentionally not a primitive: source
/// adapters must project unordered structures into schema-defined canonical
/// sequences before constructing this transcript.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SemanticTranscriptV1 {
    bytes: Vec<u8>,
    last_tag: Option<u16>,
}

impl SemanticTranscriptV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn push_utf8(
        &mut self,
        tag: u16,
        value: &str,
    ) -> Result<(), SemanticTranscriptError> {
        self.push_field(tag, SemanticFieldKind::Utf8, value.as_bytes())
    }

    pub fn push_bytes(
        &mut self,
        tag: u16,
        value: &[u8],
    ) -> Result<(), SemanticTranscriptError> {
        self.push_field(tag, SemanticFieldKind::Bytes, value)
    }

    pub fn push_u64(
        &mut self,
        tag: u16,
        value: u64,
    ) -> Result<(), SemanticTranscriptError> {
        let payload = minimal_unsigned_be(value);
        self.push_field(tag, SemanticFieldKind::UnsignedInteger, &payload)
    }

    pub fn push_i64(
        &mut self,
        tag: u16,
        value: i64,
    ) -> Result<(), SemanticTranscriptError> {
        let mut payload = Vec::with_capacity(9);
        payload.push(u8::from(value.is_negative()));
        payload.extend_from_slice(&minimal_unsigned_be(value.unsigned_abs()));
        self.push_field(tag, SemanticFieldKind::SignedInteger, &payload)
    }

    pub fn push_f64(
        &mut self,
        tag: u16,
        value: f64,
    ) -> Result<(), SemanticTranscriptError> {
        if !value.is_finite() {
            return Err(SemanticTranscriptError::NonFiniteFloat);
        }
        let canonical = if value == 0.0 { 0.0 } else { value };
        self.push_field(
            tag,
            SemanticFieldKind::FiniteF64,
            &canonical.to_bits().to_be_bytes(),
        )
    }

    pub fn push_bool(
        &mut self,
        tag: u16,
        value: bool,
    ) -> Result<(), SemanticTranscriptError> {
        self.push_field(tag, SemanticFieldKind::Boolean, &[u8::from(value)])
    }

    /// Encode an ordered sequence of already-canonical item byte strings.
    ///
    /// Payload = count:u64-be || (len:u64-be || item)*. Item boundaries are
    /// therefore unambiguous and item order remains semantic.
    pub fn push_sequence(
        &mut self,
        tag: u16,
        items: &[Vec<u8>],
    ) -> Result<(), SemanticTranscriptError> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(items.len() as u64).to_be_bytes());
        for item in items {
            payload.extend_from_slice(&(item.len() as u64).to_be_bytes());
            payload.extend_from_slice(item);
        }
        self.push_field(tag, SemanticFieldKind::Sequence, &payload)
    }

    pub fn push_digest_reference(
        &mut self,
        tag: u16,
        digest: [u8; 32],
    ) -> Result<(), SemanticTranscriptError> {
        self.push_field(tag, SemanticFieldKind::DigestReference, &digest)
    }

    fn push_field(
        &mut self,
        tag: u16,
        kind: SemanticFieldKind,
        payload: &[u8],
    ) -> Result<(), SemanticTranscriptError> {
        if let Some(previous) = self.last_tag {
            if tag <= previous {
                return Err(SemanticTranscriptError::TagNotStrictlyIncreasing {
                    previous,
                    next: tag,
                });
            }
        }

        self.bytes.extend_from_slice(&tag.to_be_bytes());
        self.bytes.push(kind as u8);
        self.bytes
            .extend_from_slice(&(payload.len() as u64).to_be_bytes());
        self.bytes.extend_from_slice(payload);
        self.last_tag = Some(tag);
        Ok(())
    }
}

/// Stable semantic reference for one exact evidence-like statement.
///
/// This is ordinary serializable data. Construction validates identifier shape
/// but does not prove that `digest_hex` was computed correctly. A source-specific
/// adapter must independently hash [`semantic_sha256_preimage`] with reviewed
/// SHA-256 machinery before granting source authority.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EvidenceSemanticIdV1 {
    pub digest_algorithm: String,
    pub transcript_version: String,
    pub namespace: String,
    pub schema_version: String,
    pub profile_id: String,
    pub record_id: String,
    pub digest_hex: String,
}

impl EvidenceSemanticIdV1 {
    pub fn from_sha256_digest(
        namespace: impl Into<String>,
        schema_version: impl Into<String>,
        profile_id: impl Into<String>,
        record_id: impl Into<String>,
        digest: [u8; 32],
    ) -> Result<Self, SemanticTranscriptError> {
        let value = Self {
            digest_algorithm: SEMANTIC_DIGEST_ALGORITHM_SHA256.to_owned(),
            transcript_version: SEMANTIC_TRANSCRIPT_VERSION_V1.to_owned(),
            namespace: namespace.into(),
            schema_version: schema_version.into(),
            profile_id: profile_id.into(),
            record_id: record_id.into(),
            digest_hex: encode_lower_hex(&digest),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), SemanticTranscriptError> {
        if self.digest_algorithm != SEMANTIC_DIGEST_ALGORITHM_SHA256 {
            return Err(SemanticTranscriptError::UnsupportedDigestAlgorithm(
                self.digest_algorithm.clone(),
            ));
        }
        if self.transcript_version != SEMANTIC_TRANSCRIPT_VERSION_V1 {
            return Err(SemanticTranscriptError::UnsupportedTranscriptVersion(
                self.transcript_version.clone(),
            ));
        }
        validate_identity_text("namespace", &self.namespace)?;
        validate_identity_text("schema_version", &self.schema_version)?;
        validate_identity_text("profile_id", &self.profile_id)?;
        validate_identity_text("record_id", &self.record_id)?;
        decode_lower_hex_32(&self.digest_hex)?;
        Ok(())
    }

    pub fn digest_bytes(&self) -> Result<[u8; 32], SemanticTranscriptError> {
        self.validate()?;
        decode_lower_hex_32(&self.digest_hex)
    }

    pub fn matches_sha256_digest(
        &self,
        digest: [u8; 32],
    ) -> Result<bool, SemanticTranscriptError> {
        Ok(self.digest_bytes()? == digest)
    }
}

/// Exact bytes that must be hashed with SHA-256 to produce a V1 semantic ID.
///
/// Freezing the preimage in the shared types crate avoids coupling this crate to
/// one crypto provider while still giving every source adapter exactly the same
/// domain-separated byte contract.
pub fn semantic_sha256_preimage(
    namespace: &str,
    schema_version: &str,
    profile_id: &str,
    record_id: &str,
    semantic_payload: &SemanticTranscriptV1,
) -> Result<Vec<u8>, SemanticTranscriptError> {
    validate_identity_text("namespace", namespace)?;
    validate_identity_text("schema_version", schema_version)?;
    validate_identity_text("profile_id", profile_id)?;
    validate_identity_text("record_id", record_id)?;

    let mut identity = SemanticTranscriptV1::new();
    identity.push_utf8(1, namespace)?;
    identity.push_utf8(2, schema_version)?;
    identity.push_utf8(3, profile_id)?;
    identity.push_utf8(4, record_id)?;
    identity.push_bytes(5, semantic_payload.as_bytes())?;

    let mut preimage = Vec::with_capacity(SEMANTIC_EVIDENCE_DOMAIN_V1.len() + identity.as_bytes().len());
    preimage.extend_from_slice(SEMANTIC_EVIDENCE_DOMAIN_V1);
    preimage.extend_from_slice(identity.as_bytes());
    Ok(preimage)
}

fn validate_identity_text(
    field: &'static str,
    value: &str,
) -> Result<(), SemanticTranscriptError> {
    if value.is_empty() {
        return Err(SemanticTranscriptError::EmptyIdentityField(field));
    }
    Ok(())
}

fn minimal_unsigned_be(value: u64) -> Vec<u8> {
    let bytes = value.to_be_bytes();
    let first = bytes
        .iter()
        .position(|byte| *byte != 0)
        .unwrap_or(bytes.len() - 1);
    bytes[first..].to_vec()
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn decode_lower_hex_32(value: &str) -> Result<[u8; 32], SemanticTranscriptError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SemanticTranscriptError::InvalidDigestHex);
    }

    let bytes = value.as_bytes();
    let mut out = [0u8; 32];
    for (index, slot) in out.iter_mut().enumerate() {
        let high = hex_nibble(bytes[index * 2]).ok_or(SemanticTranscriptError::InvalidDigestHex)?;
        let low =
            hex_nibble(bytes[index * 2 + 1]).ok_or(SemanticTranscriptError::InvalidDigestHex)?;
        *slot = (high << 4) | low;
    }
    Ok(out)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_payload() -> SemanticTranscriptV1 {
        let mut payload = SemanticTranscriptV1::new();
        payload.push_utf8(1, "trial-01").unwrap();
        payload.push_u64(2, 7).unwrap();
        payload.push_bool(3, true).unwrap();
        payload
    }

    #[test]
    fn same_semantics_produce_same_preimage() {
        let left = semantic_sha256_preimage(
            "melothaea.test",
            "example-v1",
            "source-native",
            "record-01",
            &example_payload(),
        )
        .unwrap();
        let right = semantic_sha256_preimage(
            "melothaea.test",
            "example-v1",
            "source-native",
            "record-01",
            &example_payload(),
        )
        .unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn identity_metadata_is_in_the_preimage() {
        let payload = example_payload();
        let base = semantic_sha256_preimage(
            "melothaea.test",
            "example-v1",
            "source-native",
            "record-01",
            &payload,
        )
        .unwrap();
        let changed_namespace = semantic_sha256_preimage(
            "melothaea.other",
            "example-v1",
            "source-native",
            "record-01",
            &payload,
        )
        .unwrap();
        let changed_record = semantic_sha256_preimage(
            "melothaea.test",
            "example-v1",
            "source-native",
            "record-02",
            &payload,
        )
        .unwrap();
        assert_ne!(base, changed_namespace);
        assert_ne!(base, changed_record);
    }

    #[test]
    fn payload_change_changes_preimage() {
        let original = example_payload();
        let mut changed = SemanticTranscriptV1::new();
        changed.push_utf8(1, "trial-01").unwrap();
        changed.push_u64(2, 8).unwrap();
        changed.push_bool(3, true).unwrap();

        let left = semantic_sha256_preimage("n", "v1", "p", "r", &original).unwrap();
        let right = semantic_sha256_preimage("n", "v1", "p", "r", &changed).unwrap();
        assert_ne!(left, right);
    }

    #[test]
    fn tags_must_be_strictly_increasing() {
        let mut transcript = SemanticTranscriptV1::new();
        transcript.push_utf8(2, "second").unwrap();
        assert_eq!(
            transcript.push_utf8(1, "first"),
            Err(SemanticTranscriptError::TagNotStrictlyIncreasing {
                previous: 2,
                next: 1,
            })
        );
    }

    #[test]
    fn non_finite_float_is_rejected() {
        let mut transcript = SemanticTranscriptV1::new();
        assert_eq!(
            transcript.push_f64(1, f64::NAN),
            Err(SemanticTranscriptError::NonFiniteFloat)
        );
        assert_eq!(
            transcript.push_f64(1, f64::INFINITY),
            Err(SemanticTranscriptError::NonFiniteFloat)
        );
    }

    #[test]
    fn negative_zero_normalizes_to_positive_zero() {
        let mut negative = SemanticTranscriptV1::new();
        negative.push_f64(1, -0.0).unwrap();
        let mut positive = SemanticTranscriptV1::new();
        positive.push_f64(1, 0.0).unwrap();
        assert_eq!(negative, positive);
    }

    #[test]
    fn semantic_id_round_trip_requires_explicit_validation() {
        let identity = EvidenceSemanticIdV1::from_sha256_digest(
            "melothaea.test",
            "example-v1",
            "source-native",
            "record-01",
            [0xAB; 32],
        )
        .unwrap();
        let json = serde_json::to_string(&identity).unwrap();
        let decoded: EvidenceSemanticIdV1 = serde_json::from_str(&json).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded, identity);
        assert!(decoded.matches_sha256_digest([0xAB; 32]).unwrap());
        assert!(!decoded.matches_sha256_digest([0xAC; 32]).unwrap());
    }

    #[test]
    fn malformed_or_uppercase_digest_is_rejected() {
        let mut identity = EvidenceSemanticIdV1::from_sha256_digest(
            "melothaea.test",
            "example-v1",
            "source-native",
            "record-01",
            [0; 32],
        )
        .unwrap();
        identity.digest_hex = "A".repeat(64);
        assert_eq!(
            identity.validate(),
            Err(SemanticTranscriptError::InvalidDigestHex)
        );
        identity.digest_hex = "0".repeat(63);
        assert_eq!(
            identity.validate(),
            Err(SemanticTranscriptError::InvalidDigestHex)
        );
    }

    #[test]
    fn unknown_transcript_version_is_rejected() {
        let mut identity = EvidenceSemanticIdV1::from_sha256_digest(
            "melothaea.test",
            "example-v1",
            "source-native",
            "record-01",
            [0; 32],
        )
        .unwrap();
        identity.transcript_version = "future-v2".to_owned();
        assert_eq!(
            identity.validate(),
            Err(SemanticTranscriptError::UnsupportedTranscriptVersion(
                "future-v2".to_owned()
            ))
        );
    }

    #[test]
    fn exact_integer_encoding_does_not_require_float_conversion() {
        let mut rational = SemanticTranscriptV1::new();
        rational.push_i64(1, -1).unwrap();
        rational.push_u64(2, 3).unwrap();

        let mut approximate = SemanticTranscriptV1::new();
        approximate.push_f64(1, -1.0 / 3.0).unwrap();

        assert_ne!(rational.as_bytes(), approximate.as_bytes());
    }
}
