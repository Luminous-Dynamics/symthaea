// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared semantic identity for evidence-like artifacts.
//!
//! This module deliberately does not grant source authority. A semantic ID
//! identifies one exact typed semantic statement under one exact namespace,
//! schema, profile, and record identity. Domain-native validators remain the
//! authority for deciding whether a statement is admissible evidence.

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
/// Tags must be strictly increasing, making field order explicit and preventing
/// duplicate tags. Maps are intentionally not a primitive: source adapters must
/// project unordered structures into schema-defined canonical sequences first.
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
    /// The sequence payload is count || (len || item)* using u64 big-endian
    /// lengths, so concatenation cannot create ambiguity between item splits.
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
/// Construction does not authenticate a producer or grant scientific/source
/// authority. Those remain separate boundaries.
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
    pub fn from_payload(
        namespace: impl Into<String>,
        schema_version: impl Into<String>,
        profile_id: impl Into<String>,
        record_id: impl Into<String>,
        semantic_payload: &SemanticTranscriptV1,
    ) -> Result<Self, SemanticTranscriptError> {
        let namespace = namespace.into();
        let schema_version = schema_version.into();
        let profile_id = profile_id.into();
        let record_id = record_id.into();

        validate_identity_text("namespace", &namespace)?;
        validate_identity_text("schema_version", &schema_version)?;
        validate_identity_text("profile_id", &profile_id)?;
        validate_identity_text("record_id", &record_id)?;

        let transcript = identity_transcript(
            &namespace,
            &schema_version,
            &profile_id,
            &record_id,
            semantic_payload,
        )?;

        let mut hasher = Sha256::new();
        hasher.update(SEMANTIC_EVIDENCE_DOMAIN_V1);
        hasher.update(transcript.as_bytes());
        let digest: [u8; 32] = hasher.finalize().into();

        Ok(Self {
            digest_algorithm: SEMANTIC_DIGEST_ALGORITHM_SHA256.to_owned(),
            transcript_version: SEMANTIC_TRANSCRIPT_VERSION_V1.to_owned(),
            namespace,
            schema_version,
            profile_id,
            record_id,
            digest_hex: encode_lower_hex(&digest),
        })
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

    pub fn verify_payload(
        &self,
        semantic_payload: &SemanticTranscriptV1,
    ) -> Result<bool, SemanticTranscriptError> {
        self.validate()?;
        let expected = Self::from_payload(
            self.namespace.clone(),
            self.schema_version.clone(),
            self.profile_id.clone(),
            self.record_id.clone(),
            semantic_payload,
        )?;
        Ok(expected == *self)
    }
}

fn identity_transcript(
    namespace: &str,
    schema_version: &str,
    profile_id: &str,
    record_id: &str,
    semantic_payload: &SemanticTranscriptV1,
) -> Result<SemanticTranscriptV1, SemanticTranscriptError> {
    let mut transcript = SemanticTranscriptV1::new();
    transcript.push_utf8(1, namespace)?;
    transcript.push_utf8(2, schema_version)?;
    transcript.push_utf8(3, profile_id)?;
    transcript.push_utf8(4, record_id)?;
    transcript.push_bytes(5, semantic_payload.as_bytes())?;
    Ok(transcript)
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

    fn example_id(payload: &SemanticTranscriptV1) -> EvidenceSemanticIdV1 {
        EvidenceSemanticIdV1::from_payload(
            "melothaea.test",
            "example-v1",
            "source-native",
            "record-01",
            payload,
        )
        .unwrap()
    }

    #[test]
    fn same_semantics_produce_same_identity() {
        let left = example_id(&example_payload());
        let right = example_id(&example_payload());
        assert_eq!(left, right);
        assert!(left.verify_payload(&example_payload()).unwrap());
    }

    #[test]
    fn identity_fields_are_domain_separated() {
        let payload = example_payload();
        let base = example_id(&payload);
        let changed = EvidenceSemanticIdV1::from_payload(
            "melothaea.other",
            "example-v1",
            "source-native",
            "record-01",
            &payload,
        )
        .unwrap();
        assert_ne!(base.digest_hex, changed.digest_hex);
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
    fn payload_change_invalidates_identity() {
        let original = example_payload();
        let identity = example_id(&original);

        let mut changed = SemanticTranscriptV1::new();
        changed.push_utf8(1, "trial-01").unwrap();
        changed.push_u64(2, 8).unwrap();
        changed.push_bool(3, true).unwrap();

        assert!(!identity.verify_payload(&changed).unwrap());
    }

    #[test]
    fn serde_round_trip_requires_post_parse_validation() {
        let identity = example_id(&example_payload());
        let json = serde_json::to_string(&identity).unwrap();
        let decoded: EvidenceSemanticIdV1 = serde_json::from_str(&json).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded, identity);
    }

    #[test]
    fn malformed_or_uppercase_digest_is_rejected() {
        let mut identity = example_id(&example_payload());
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
        let mut identity = example_id(&example_payload());
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
