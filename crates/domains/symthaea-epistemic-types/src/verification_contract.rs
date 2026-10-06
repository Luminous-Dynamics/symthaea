//! Substrate-neutral boundary for adapter-produced claim verification evidence.
//!
//! This module does not resolve controller documents, perform cryptography, or make
//! network calls. It defines the exact request an adapter must satisfy and the
//! evidence envelope it may return after doing those operations.
//!
//! The separation is intentional:
//!
//! `ClaimAuthorship` expresses declarations.
//! `VerificationRequest` states what an adapter is being asked to establish.
//! `VerificationEvidence` records which external resolution/cryptographic artifacts
//! were used to establish it.
//!
//! A `Verified` outcome therefore means "the adapter attests that it performed the
//! required checks", not that the substrate-neutral core itself performed them.

use crate::{
    ClaimAuthorIdentity, ClaimControllerIdentity, ClaimProofPurpose, ClaimVerificationMethod,
    FederatedClaim, FederationDependency,
};
use chrono::{DateTime, FixedOffset};
use serde::{Deserialize, Serialize};
use url::Url;

pub const VERIFICATION_REQUEST_SCHEMA_VERSION: u16 = 1;
pub const VERIFICATION_EVIDENCE_SCHEMA_VERSION: u16 = 9;
pub const VERIFICATION_EVIDENCE_DIGEST_VERSION: u16 = 11;
pub const CRYPTOGRAPHIC_VERIFICATION_RECEIPT_SCHEMA_VERSION: u16 = 4;

/// Typed identifier for the verification relationship under which a verification
/// method is permitted to validate a proof.
///
/// This is deliberately distinct from proof purpose: adapters must establish both
/// the requested purpose and the controller-document relationship rather than
/// assuming that one string semantically subsumes the other.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClaimVerificationRelationship(String);

impl ClaimVerificationRelationship {
    pub fn new(value: impl Into<String>) -> Result<Self, VerificationFailure> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(VerificationFailure::Structural(
                "claim verification relationship must be non-empty".into(),
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        if self.0.trim().is_empty() {
            Err(VerificationFailure::Structural(
                "claim verification relationship must be non-empty".into(),
            ))
        } else {
            Ok(())
        }
    }
}

/// Compare URL values using the URL parser/serializer model required by CID.
///
/// URL equivalence is intentionally semantic rather than raw-string equality:
/// parse both values, serialize them, then compare the serialized URLs. Raw input
/// strings remain stored separately, so provenance/digest domains do not silently
/// collapse merely because two inputs are URL-equivalent.
pub fn url_values_equivalent(left: &str, right: &str) -> Result<bool, url::ParseError> {
    let left = Url::parse(left)?.to_string();
    let right = Url::parse(right)?.to_string();
    Ok(left == right)
}

fn is_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn decode_sha256_multibase(value: &str) -> Result<[u8; 32], VerificationFailure> {
    let encoded = value.strip_prefix('z').ok_or_else(|| {
        VerificationFailure::Structural(
            "controller document digestMultibase must use the base58-btc multibase prefix"
                .into(),
        )
    })?;
    let bytes = bs58::decode(encoded).into_vec().map_err(|_| {
        VerificationFailure::Structural(
            "controller document digestMultibase must contain valid base58-btc data".into(),
        )
    })?;
    if bytes.len() != 34 || bytes[0] != 0x12 || bytes[1] != 0x20 {
        return Err(VerificationFailure::Structural(
            "controller document digestMultibase must encode a SHA-256 multihash".into(),
        ));
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&bytes[2..]);
    Ok(digest)
}

fn validate_sha256_multibase(
    value: &str,
    expected_hex_digest: Option<&str>,
) -> Result<(), VerificationFailure> {
    let digest = decode_sha256_multibase(value)?;
    if let Some(expected) = expected_hex_digest {
        let actual = hex::encode(digest);
        if !is_hex_digest(expected) || !actual.eq_ignore_ascii_case(expected) {
            return Err(VerificationFailure::ControllerDocumentIntegrityMismatch {
                expected: expected.to_owned(),
                actual,
            });
        }
    }
    Ok(())
}

/// Policy describing whether a controller-document snapshot must match a known
/// cryptographic content pin. Unpinned is explicitly weaker: it records the
/// observed document digest without claiming that the verifier had a trusted
/// expected value for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControllerDocumentIntegrityPolicy {
    Unpinned,
    Sha256Digest(String),
}

impl ControllerDocumentIntegrityPolicy {
    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        match self {
            Self::Unpinned => Ok(()),
            Self::Sha256Digest(expected) if is_hex_digest(expected) => Ok(()),
            Self::Sha256Digest(_) => Err(VerificationFailure::Structural(
                "expected controller document digest must be a 64-character hexadecimal digest"
                    .into(),
            )),
        }
    }
}

/// Adapter-produced statement about whether the dereferenced controller document
/// matched the requested content-integrity policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControllerDocumentIntegrityAttestation {
    Unpinned { actual_digest: String },
    Sha256Digest {
        expected_digest: String,
        actual_digest: String,
    },
}

impl ControllerDocumentIntegrityAttestation {
    pub fn from_policy(
        policy: &ControllerDocumentIntegrityPolicy,
        actual_digest: impl Into<String>,
    ) -> Result<Self, VerificationFailure> {
        policy.validate_structure()?;
        let actual_digest = actual_digest.into();
        if !is_hex_digest(&actual_digest) {
            return Err(VerificationFailure::Structural(
                "controller document digest must be a 64-character hexadecimal digest".into(),
            ));
        }

        match policy {
            ControllerDocumentIntegrityPolicy::Unpinned => {
                Ok(Self::Unpinned { actual_digest })
            }
            ControllerDocumentIntegrityPolicy::Sha256Digest(expected_digest) => {
                if expected_digest != &actual_digest {
                    return Err(VerificationFailure::ControllerDocumentIntegrityMismatch {
                        expected: expected_digest.clone(),
                        actual: actual_digest,
                    });
                }
                Ok(Self::Sha256Digest {
                    expected_digest: expected_digest.clone(),
                    actual_digest,
                })
            }
        }
    }

    pub fn actual_digest(&self) -> &str {
        match self {
            Self::Unpinned { actual_digest }
            | Self::Sha256Digest {
                actual_digest, ..
            } => actual_digest,
        }
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        match self {
            Self::Unpinned { actual_digest } if is_hex_digest(actual_digest) => Ok(()),
            Self::Sha256Digest {
                expected_digest,
                actual_digest,
            } if is_hex_digest(expected_digest) && actual_digest == expected_digest => Ok(()),
            Self::Unpinned { .. } | Self::Sha256Digest { .. } => {
                Err(VerificationFailure::Structural(
                    "controller document integrity attestation is malformed".into(),
                ))
            }
        }
    }

    pub fn matches_policy(&self, policy: &ControllerDocumentIntegrityPolicy) -> bool {
        match (policy, self) {
            (
                ControllerDocumentIntegrityPolicy::Unpinned,
                Self::Unpinned { .. },
            ) => true,
            (
                ControllerDocumentIntegrityPolicy::Sha256Digest(expected),
                Self::Sha256Digest {
                    expected_digest,
                    actual_digest,
                },
            ) => expected == expected_digest && expected == actual_digest,
            _ => false,
        }
    }

    pub fn identity_digest(&self) -> String {
        let encoded = (
            "symthaea:controller-document-integrity:v1",
            self,
        );
        let bytes =
            serde_json::to_vec(&encoded).expect("controller document integrity is serializable");
        crate::sha256_hex(&bytes)
    }
}

/// Lifecycle metadata read from the resolved verification-method definition.
///
/// The W3C Controlled Identifiers model treats expires and revoked as method-level
/// timestamps. The core does not infer lifecycle state from absence of either property;
/// it only evaluates the explicit timestamps supplied by the adapter for the snapshot
/// it resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationMethodLifecycle {
    pub expires: Option<String>,
    pub revoked: Option<String>,
}

impl VerificationMethodLifecycle {
    pub fn new(
        expires: Option<&str>,
        revoked: Option<&str>,
    ) -> Result<Self, VerificationFailure> {
        let value = Self {
            expires: expires.map(str::to_owned),
            revoked: revoked.map(str::to_owned),
        };
        value.validate_structure()?;
        Ok(value)
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        for (field, value) in [
            ("verification method expires", self.expires.as_deref()),
            ("verification method revoked", self.revoked.as_deref()),
        ] {
            if let Some(value) = value {
                parse_timestamp(field, value)?;
            }
        }
        Ok(())
    }

    /// Reject the verification method at or after an explicit expiry/revocation.
    ///
    /// This evaluates only the supplied snapshot facts. Historical acceptance after
    /// a later revocation therefore requires a historical snapshot whose lifecycle
    /// state can be established by the adapter.
    pub fn validate_for_use_at(&self, reference_time: &str) -> Result<(), VerificationFailure> {
        self.validate_structure()?;
        let reference_time =
            parse_timestamp("verification method evaluation time", reference_time)?;

        if let Some(revoked) = self.revoked.as_deref() {
            let revoked_at = parse_timestamp("verification method revoked", revoked)?;
            if reference_time >= revoked_at {
                return Err(VerificationFailure::VerificationMethodRevoked {
                    at: revoked.to_owned(),
                });
            }
        }

        if let Some(expires) = self.expires.as_deref() {
            let expires_at = parse_timestamp("verification method expires", expires)?;
            if reference_time >= expires_at {
                return Err(VerificationFailure::VerificationMethodExpired {
                    at: expires.to_owned(),
                });
            }
        }

        Ok(())
    }
}

/// Network/dereferencing controls bound to a verification request.
///
/// These are application security policy, not W3C protocol requirements. The default
/// derived from the controller-document URL permits only that URL scheme, forbids
/// redirects, and caps the response size so a future network adapter has a fail-closed
/// baseline instead of inheriting an ambient HTTP client policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControllerDocumentNetworkPolicy {
    pub allowed_schemes: Vec<String>,
    pub max_response_bytes: u64,
    pub max_redirects: u16,
    pub require_effective_url_match: bool,
}

impl ControllerDocumentNetworkPolicy {
    pub fn strict_for_url(url: &str) -> Result<Self, VerificationFailure> {
        let parsed =
            Url::parse(url).map_err(|_| VerificationFailure::InvalidControllerDocumentUrl)?;
        Ok(Self {
            allowed_schemes: vec![parsed.scheme().to_ascii_lowercase()],
            max_response_bytes: 4 * 1024 * 1024,
            max_redirects: 0,
            require_effective_url_match: true,
        })
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        if self.allowed_schemes.is_empty() {
            return Err(VerificationFailure::Structural(
                "controller document network policy must allow at least one URL scheme".into(),
            ));
        }
        let mut schemes = self.allowed_schemes.clone();
        for scheme in &mut schemes {
            if scheme.trim().is_empty() {
                return Err(VerificationFailure::Structural(
                    "controller document network policy scheme must be non-empty".into(),
                ));
            }
            *scheme = scheme.to_ascii_lowercase();
        }
        schemes.sort();
        schemes.dedup();
        if schemes != self.allowed_schemes.iter().map(|s| s.to_ascii_lowercase()).collect::<Vec<_>>() {
            return Err(VerificationFailure::Structural(
                "controller document network policy schemes must be canonical sorted unique values"
                    .into(),
            ));
        }
        if self.max_response_bytes == 0 {
            return Err(VerificationFailure::Structural(
                "controller document network policy max response bytes must be non-zero".into(),
            ));
        }
        Ok(())
    }

    fn allows_scheme(&self, scheme: &str) -> bool {
        self.allowed_schemes
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(scheme))
    }
}

/// Source from which an adapter obtained the controller-document bytes.
///
/// This is provenance about resolution, not proof of document integrity by itself;
/// the content digest and controller-document identity checks remain mandatory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControllerDocumentResolutionSource {
    Network,
    Cache,
    HistoricalRegistry,
    ApplicationSnapshot,
}

impl ControllerDocumentResolutionSource {
    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        Ok(())
    }
}

/// Adapter attestation for the concrete controller-document dereference performed
/// for a verification-method resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControllerDocumentDereferenceAttestation {
    pub requested_url: String,
    pub effective_url: String,
    pub response_media_type: String,
    pub response_size_bytes: u64,
    pub redirect_count: u16,
    pub resolved_at: String,
    pub source: ControllerDocumentResolutionSource,
    pub document_digest: String,
    /// Optional Multibase-encoded SHA-256 multihash for the dereferenced bytes.
    #[serde(default, rename = "digestMultibase", alias = "digest_multibase")]
    pub digest_multibase: Option<String>,
}

impl ControllerDocumentDereferenceAttestation {
    pub fn from_adapter(
        request: &VerificationRequest,
        effective_url: impl Into<String>,
        response_media_type: impl Into<String>,
        response_size_bytes: u64,
        redirect_count: u16,
        resolved_at: impl Into<String>,
        source: ControllerDocumentResolutionSource,
        document_digest: impl Into<String>,
        digest_multibase: Option<String>,
    ) -> Result<Self, VerificationFailure> {
        request.validate_structure()?;
        let requested_url = request.controller_document_ref()?;
        let value = Self {
            requested_url,
            effective_url: effective_url.into(),
            response_media_type: response_media_type.into(),
            response_size_bytes,
            redirect_count,
            resolved_at: resolved_at.into(),
            source,
            document_digest: document_digest.into(),
            digest_multibase,
        };
        value.validate_against_request(request)?;
        Ok(value)
    }

    pub fn validate_against_request(
        &self,
        request: &VerificationRequest,
    ) -> Result<(), VerificationFailure> {
        request.controller_document_network_policy.validate_structure()?;
        request.controller_document_integrity_policy.validate_structure()?;
        let expected_url = request.controller_document_ref()?;
        if self.requested_url != expected_url {
            return Err(VerificationFailure::ControllerDocumentMismatch {
                expected: expected_url,
                actual: self.requested_url.clone(),
            });
        }

        self.validate_against_policy(
            &request.controller_document_network_policy,
            &expected_url,
            &self.document_digest,
        )?;
        match &request.controller_document_integrity_policy {
            ControllerDocumentIntegrityPolicy::Unpinned => {}
            ControllerDocumentIntegrityPolicy::Sha256Digest(expected)
                if expected == &self.document_digest => {}
            ControllerDocumentIntegrityPolicy::Sha256Digest(expected) => {
                return Err(VerificationFailure::ControllerDocumentIntegrityMismatch {
                    expected: expected.clone(),
                    actual: self.document_digest.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn validate_against_policy(
        &self,
        policy: &ControllerDocumentNetworkPolicy,
        expected_url: &str,
        expected_digest: &str,
    ) -> Result<(), VerificationFailure> {
        policy.validate_structure()?;
        let requested = Url::parse(&self.requested_url)
            .map_err(|_| VerificationFailure::InvalidControllerDocumentUrl)?;
        let effective = Url::parse(&self.effective_url)
            .map_err(|_| VerificationFailure::InvalidControllerDocumentUrl)?;

        if !url_values_equivalent(&self.requested_url, expected_url).unwrap_or(false)
            || self.document_digest != expected_digest
        {
            return Err(VerificationFailure::ControllerDocumentDereferenceMismatch);
        }
        if self.response_media_type.trim().is_empty() {
            return Err(VerificationFailure::Structural(
                "controller document dereference response media type must be non-empty".into(),
            ));
        }
        if !policy.allows_scheme(requested.scheme())
            || !policy.allows_scheme(effective.scheme())
        {
            return Err(VerificationFailure::ControllerDocumentNetworkPolicyViolation);
        }
        if policy.require_effective_url_match
            && !url_values_equivalent(effective.as_str(), requested.as_str()).unwrap_or(false)
        {
            return Err(VerificationFailure::ControllerDocumentEffectiveUrlMismatch);
        }
        if self.redirect_count > policy.max_redirects {
            return Err(VerificationFailure::ControllerDocumentRedirectLimitExceeded);
        }
        if self.response_size_bytes > policy.max_response_bytes {
            return Err(VerificationFailure::ControllerDocumentResponseTooLarge);
        }
        self.source.validate_structure()?;
        parse_timestamp("controller document resolved at", &self.resolved_at)?;
        if !is_hex_digest(&self.document_digest) {
            return Err(VerificationFailure::Structural(
                "controller document dereference digest must be a 64-character hexadecimal digest"
                    .into(),
            ));
        }
        if let Some(digest_multibase) = &self.digest_multibase {
            validate_sha256_multibase(digest_multibase, Some(&self.document_digest))?;
        }
        Ok(())
    }

    pub fn matches_document(&self, document_url: &str, document_digest: &str) -> bool {
        url_values_equivalent(&self.requested_url, document_url).unwrap_or(false)
            && self.document_digest == document_digest
    }
}

/// States whether the resolved controller-document facts describe only the current
/// document state or a historical state tied to a specific evaluation instant.
///
/// A current mutable document is never silently treated as evidence about an earlier
/// proof. Historical evaluation requires an explicit historical snapshot reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControllerDocumentSnapshotScope {
    Current {
        resolved_at: String,
    },
    HistoricalAt {
        state_at: String,
        snapshot_reference: String,
        resolved_at: String,
    },
}

impl ControllerDocumentSnapshotScope {
    pub fn current(resolved_at: impl Into<String>) -> Result<Self, VerificationFailure> {
        let scope = Self::Current {
            resolved_at: resolved_at.into(),
        };
        scope.validate_structure()?;
        Ok(scope)
    }

    pub fn historical_at(
        state_at: impl Into<String>,
        snapshot_reference: impl Into<String>,
        resolved_at: impl Into<String>,
    ) -> Result<Self, VerificationFailure> {
        let scope = Self::HistoricalAt {
            state_at: state_at.into(),
            snapshot_reference: snapshot_reference.into(),
            resolved_at: resolved_at.into(),
        };
        scope.validate_structure()?;
        Ok(scope)
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        match self {
            Self::Current { resolved_at } => {
                parse_timestamp("controller document resolved at", resolved_at)?;
            }
            Self::HistoricalAt {
                state_at,
                snapshot_reference,
                resolved_at,
            } => {
                parse_timestamp("controller document state at", state_at)?;
                parse_timestamp("controller document resolved at", resolved_at)?;
                if snapshot_reference.trim().is_empty() {
                    return Err(VerificationFailure::Structural(
                        "historical controller document snapshot reference must be non-empty"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Require the snapshot to describe the exact state relevant to lifecycle evaluation.
    ///
    /// A current snapshot is usable only for its own resolution instant. Historical
    /// evaluation requires a historical snapshot explicitly tied to the reference time.
    pub fn resolved_at(&self) -> &str {
        match self {
            Self::Current { resolved_at } | Self::HistoricalAt { resolved_at, .. } => resolved_at,
        }
    }

    pub fn validate_for_reference_time(
        &self,
        reference_time: &str,
    ) -> Result<(), VerificationFailure> {
        self.validate_structure()?;
        let reference_time =
            parse_timestamp("verification method evaluation time", reference_time)?;

        match self {
            Self::Current { resolved_at } => {
                let resolved_at =
                    parse_timestamp("controller document resolved at", resolved_at)?;
                if resolved_at == reference_time {
                    Ok(())
                } else {
                    Err(VerificationFailure::HistoricalStateRequired)
                }
            }
            Self::HistoricalAt {
                state_at,
                resolved_at,
                ..
            } => {
                let state_at =
                    parse_timestamp("controller document state at", state_at)?;
                if state_at != reference_time {
                    return Err(VerificationFailure::HistoricalStateMismatch);
                }
                let resolved_at =
                    parse_timestamp("controller document resolved at", resolved_at)?;
                if resolved_at < state_at {
                    return Err(VerificationFailure::InvalidSnapshotOrdering);
                }
                Ok(())
            }
        }
    }
}

/// Temporal and anti-replay inputs supplied by the proof and the verifier.
///
/// The core validates syntax, temporal ordering, and exact domain/challenge matching.
/// It does not maintain a challenge-consumption store; one-time challenge tracking
/// remains an adapter/application responsibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationFreshnessContext {
    pub proof_created: Option<String>,
    pub proof_expires: Option<String>,
    pub proof_domain: Option<String>,
    pub proof_challenge: Option<String>,
    pub verification_time: String,
    pub expected_domain: Option<String>,
    pub expected_challenge: Option<String>,
}

impl VerificationFreshnessContext {
    pub fn validate(&self) -> Result<(), VerificationFailure> {
        let verification_time = parse_timestamp("verification time", &self.verification_time)?;

        let created = self
            .proof_created
            .as_deref()
            .map(|value| parse_timestamp("proof created", value))
            .transpose()?;

        let expires = self
            .proof_expires
            .as_deref()
            .map(|value| parse_timestamp("proof expires", value))
            .transpose()?;

        if let (Some(created), Some(expires)) = (created, expires) {
            if expires < created {
                return Err(VerificationFailure::InvalidValidityWindow);
            }
        }

        if created.is_some_and(|created| created > verification_time) {
            return Err(VerificationFailure::ProofCreatedInFuture);
        }

        if expires.is_some_and(|expires| verification_time >= expires) {
            return Err(VerificationFailure::ProofExpired);
        }

        for (name, value) in [
            ("proof domain", self.proof_domain.as_deref()),
            ("proof challenge", self.proof_challenge.as_deref()),
            ("expected domain", self.expected_domain.as_deref()),
            ("expected challenge", self.expected_challenge.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(VerificationFailure::Structural(format!(
                    "{name} must be non-empty when present"
                )));
            }
        }

        if let Some(expected_domain) = &self.expected_domain {
            if self.proof_domain.as_deref() != Some(expected_domain.as_str()) {
                return Err(VerificationFailure::DomainMismatch {
                    expected: expected_domain.clone(),
                    actual: self.proof_domain.clone(),
                });
            }
        }

        if let Some(expected_challenge) = &self.expected_challenge {
            if self.proof_challenge.as_deref() != Some(expected_challenge.as_str()) {
                return Err(VerificationFailure::ChallengeMismatch {
                    expected: expected_challenge.clone(),
                    actual: self.proof_challenge.clone(),
                });
            }
        }

        Ok(())
    }

    pub fn lifecycle_reference_time(&self) -> &str {
        self.proof_created
            .as_deref()
            .unwrap_or(self.verification_time.as_str())
    }

    pub fn replay_context_digest(&self) -> String {
        let encoded = (
            "symthaea:verification-freshness:v1",
            &self.proof_domain,
            &self.proof_challenge,
            &self.expected_domain,
            &self.expected_challenge,
        );
        let bytes = serde_json::to_vec(&encoded)
            .expect("verification freshness context is serializable");
        crate::sha256_hex(&bytes)
    }
}

/// Validate the XML Schema 1.1 `dateTimeStamp` lexical and temporal boundary
/// used by Controlled Identifiers lifecycle fields and verification freshness.
pub fn validate_xsd11_date_time_stamp(value: &str) -> Result<(), VerificationFailure> {
    parse_timestamp("timestamp", value).map(|_| ())
}

fn parse_timestamp(
    field: &'static str,
    value: &str,
) -> Result<DateTime<FixedOffset>, VerificationFailure> {
    let end_of_day = validate_xsd11_date_time_stamp_lexical(value).map_err(|_| {
        VerificationFailure::InvalidTimestamp {
            field,
            value: value.to_owned(),
        }
    })?;

    // XSD 1.1 dateTimeStamp permits the lexical end-of-day form
    // YYYY-MM-DDT24:00:00(.0+)?timezone. Chrono deliberately rejects 24:00:00,
    // so normalize only that standards-defined spelling to the equivalent next-day
    // instant while retaining the caller's original lexical representation.
    let parse_value = if end_of_day {
        let t = value.find('T').ok_or_else(|| VerificationFailure::InvalidTimestamp {
            field,
            value: value.to_owned(),
        })?;
        let mut normalized = value.to_owned();
        normalized.replace_range(t + 1..t + 3, "00");
        normalized
    } else {
        value.to_owned()
    };

    let parsed = DateTime::parse_from_rfc3339(&parse_value).map_err(|_| {
        VerificationFailure::InvalidTimestamp {
            field,
            value: value.to_owned(),
        }
    })?;

    if end_of_day {
        parsed
            .checked_add_signed(chrono::Duration::days(1))
            .ok_or_else(|| VerificationFailure::InvalidTimestamp {
                field,
                value: value.to_owned(),
            })
    } else {
        Ok(parsed)
    }
}

fn validate_xsd11_date_time_stamp_lexical(value: &str) -> Result<bool, ()> {
    let bytes = value.as_bytes();
    let mut index = 0;

    if bytes.first() == Some(&b'-') {
        index += 1;
    }

    let year_start = index;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    let year_len = index - year_start;
    if year_len < 4 || (year_len > 4 && bytes[year_start] == b'0') {
        return Err(());
    }
    if bytes.get(index) != Some(&b'-') {
        return Err(());
    }
    index += 1;

    let month = parse_two_digits(bytes, &mut index)?;
    if bytes.get(index) != Some(&b'-') {
        return Err(());
    }
    index += 1;

    let day = parse_two_digits(bytes, &mut index)?;
    if !(1..=12).contains(&month) || day == 0 || day > 31 {
        return Err(());
    }

    if bytes.get(index) != Some(&b'T') {
        return Err(());
    }
    index += 1;

    let hour = parse_two_digits(bytes, &mut index)?;
    if bytes.get(index) != Some(&b':') {
        return Err(());
    }
    index += 1;

    let minute = parse_two_digits(bytes, &mut index)?;
    if bytes.get(index) != Some(&b':') {
        return Err(());
    }
    index += 1;

    let second = parse_two_digits(bytes, &mut index)?;
    if minute > 59 || second > 59 {
        return Err(());
    }

    let end_of_day = if hour == 24 {
        minute == 0 && second == 0
    } else {
        hour <= 23
    };
    if !end_of_day && hour > 23 {
        return Err(());
    }

    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let fraction_start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if fraction_start == index {
            return Err(());
        }
        if end_of_day && bytes[fraction_start..index].iter().any(|byte| *byte != b'0') {
            return Err(());
        }
    }

    match bytes.get(index) {
        Some(b'Z') if index + 1 == bytes.len() => Ok(end_of_day),
        Some(b'+') | Some(b'-') => {
            if index + 6 != bytes.len()
                || bytes.get(index + 3) != Some(&b':')
                || !bytes[index + 1].is_ascii_digit()
                || !bytes[index + 2].is_ascii_digit()
                || !bytes[index + 4].is_ascii_digit()
                || !bytes[index + 5].is_ascii_digit()
            {
                return Err(());
            }
            let offset_hour = (bytes[index + 1] - b'0') as u8 * 10
                + (bytes[index + 2] - b'0') as u8;
            let offset_minute = (bytes[index + 4] - b'0') as u8 * 10
                + (bytes[index + 5] - b'0') as u8;
            if offset_hour > 14 || offset_minute > 59 || (offset_hour == 14 && offset_minute != 0) {
                return Err(());
            }
            Ok(end_of_day)
        }
        _ => Err(()),
    }
}

fn parse_two_digits(bytes: &[u8], index: &mut usize) -> Result<u8, ()> {
    if *index + 2 > bytes.len()
        || !bytes[*index].is_ascii_digit()
        || !bytes[*index + 1].is_ascii_digit()
    {
        return Err(());
    }
    let value = (bytes[*index] - b'0') * 10 + (bytes[*index + 1] - b'0');
    *index += 2;
    Ok(value)
}

/// Typed identity of the concrete controlled-identifier document resource
/// dereferenced for a verification method. This is intentionally distinct from the
/// controller identity expressed by a verification method.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClaimControllerDocumentIdentity(String);

impl ClaimControllerDocumentIdentity {
    pub fn new(value: impl Into<String>) -> Result<Self, VerificationFailure> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(VerificationFailure::Structural(
                "controller document identity must be non-empty".into(),
            ));
        }
        Url::parse(&value)
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        Self::new(self.0.clone()).map(|_| ())
    }
}

/// Adapter attestation for the exact Controlled Identifiers verification-method
/// retrieval boundary.
///
/// This captures the security-critical result of the W3C retrieval algorithm without
/// performing dereferencing itself. The constructor receives the already-dereferenced
/// controller document facts and checks the invariants that can be evaluated purely
/// from those facts:
///
/// * the method identifier's primary resource is the controller document URL;
/// * the controller document's `id` is that URL;
/// * the resolved method identifier is exactly the requested method;
/// * the method's declared controller is exactly that controller-document URL; and
/// * the exact method is a member of the requested verification relationship.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationMethodResolution {
    pub schema_version: u16,
    pub verification_method: ClaimVerificationMethod,
    pub verification_method_type: String,
    pub verification_method_material_digest: String,
    pub controller_document_ref: String,
    pub controller_document_id: ClaimControllerDocumentIdentity,
    pub resolved_verification_method_controller: ClaimControllerIdentity,
    pub verification_relationship: ClaimVerificationRelationship,
    pub relationship_methods: Vec<ClaimVerificationMethod>,
    pub relationship_methods_digest: String,
    pub controller_document_digest: String,
    pub controller_document_integrity: ControllerDocumentIntegrityAttestation,
    pub verification_method_lifecycle: VerificationMethodLifecycle,
    pub controller_document_snapshot_scope: ControllerDocumentSnapshotScope,
    pub controller_document_network_policy: ControllerDocumentNetworkPolicy,
    pub controller_document_dereference: Option<ControllerDocumentDereferenceAttestation>,
}

impl VerificationMethodResolution {
    pub const SCHEMA_VERSION: u16 = 5;

    pub fn with_controller_document_dereference(
        mut self,
        dereference: ControllerDocumentDereferenceAttestation,
        request: &VerificationRequest,
    ) -> Result<Self, VerificationFailure> {
        dereference.validate_against_request(request)?;
        if !dereference.matches_document(
            &self.controller_document_ref,
            &self.controller_document_digest,
        ) {
            return Err(VerificationFailure::ControllerDocumentDereferenceMismatch);
        }
        self.controller_document_dereference = Some(dereference);
        Ok(self)
    }

    pub fn from_controller_document(
        request: &VerificationRequest,
        controller_document_ref: impl Into<String>,
        controller_document_id: ClaimControllerDocumentIdentity,
        resolved_verification_method: ClaimVerificationMethod,
        verification_method_type: impl Into<String>,
        verification_method_material_digest: impl Into<String>,
        resolved_verification_method_controller: ClaimControllerIdentity,
        relationship_methods: &[ClaimVerificationMethod],
        controller_document_digest: impl Into<String>,
        verification_method_lifecycle: VerificationMethodLifecycle,
        controller_document_snapshot_scope: ControllerDocumentSnapshotScope,
    ) -> Result<Self, VerificationFailure> {
        request.validate_structure()?;
        let controller_document_ref = controller_document_ref.into();
        let verification_method_type = verification_method_type.into();
        let verification_method_material_digest = verification_method_material_digest.into();
        let controller_document_digest = controller_document_digest.into();

        let method_url = Url::parse(request.verification_method.as_str())
            .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        if method_url.fragment().is_none() {
            return Err(VerificationFailure::InvalidVerificationMethodUrl);
        }
        let mut expected_document_url = method_url.clone();
        expected_document_url.set_fragment(None);
        let expected_document_url = expected_document_url.to_string();

        if !url_values_equivalent(&controller_document_ref, &expected_document_url)
            .unwrap_or(false)
        {
            return Err(VerificationFailure::ControllerDocumentMismatch {
                expected: expected_document_url,
                actual: controller_document_ref,
            });
        }

        if !url_values_equivalent(
            resolved_verification_method.as_str(),
            request.verification_method.as_str(),
        )
        .unwrap_or(false)
        {
            return Err(VerificationFailure::VerificationMethodMismatch {
                expected: request.verification_method.clone(),
                actual: resolved_verification_method,
            });
        }

        controller_document_id.validate_structure()
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        let document_id = Url::parse(controller_document_id.as_str())
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        if !url_values_equivalent(document_id.as_str(), &controller_document_ref)
            .unwrap_or(false)
        {
            return Err(VerificationFailure::ControllerDocumentMismatch {
                expected: controller_document_ref.clone(),
                actual: controller_document_id.as_str().to_owned(),
            });
        }

        resolved_verification_method_controller
            .validate_structure()
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        let resolved_controller = Url::parse(resolved_verification_method_controller.as_str())
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        if !url_values_equivalent(resolved_controller.as_str(), &controller_document_ref)
            .unwrap_or(false)
        {
            return Err(VerificationFailure::ControllerMismatch {
                expected: ClaimControllerIdentity::new(controller_document_ref.clone())
                    .expect("validated controller document URL"),
                actual: resolved_verification_method_controller,
            });
        }

        if !url_values_equivalent(
            request.expected_controller.as_str(),
            &controller_document_ref,
        )
        .unwrap_or(false)
        {
            return Err(VerificationFailure::ControllerMismatch {
                expected: request.expected_controller.clone(),
                actual: ClaimControllerIdentity::new(controller_document_ref.clone())
                    .expect("validated controller document URL"),
            });
        }

        let mut methods = relationship_methods.to_vec();
        for method in &methods {
            method.validate_structure()
                .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
            Url::parse(method.as_str())
                .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        }
        methods.sort();
        methods.dedup();

        if !methods.iter().any(|method| {
            url_values_equivalent(method.as_str(), request.verification_method.as_str())
                .unwrap_or(false)
        }) {
            return Err(VerificationFailure::VerificationMethodNotInRelationship);
        }

        let encoded_members =
            ("symthaea:verification-relationship-members:v1",
             request.expected_verification_relationship.as_str(),
             methods.iter().map(ClaimVerificationMethod::as_str).collect::<Vec<_>>());
        let bytes = serde_json::to_vec(&encoded_members)
            .expect("verification relationship member set is serializable");
        let relationship_methods_digest = crate::sha256_hex(&bytes);

        if !is_hex_digest(&controller_document_digest) {
            return Err(VerificationFailure::Structural(
                "controller document digest must be a 64-character hexadecimal digest".into(),
            ));
        }
        if verification_method_type.trim().is_empty() {
            return Err(VerificationFailure::Structural(
                "verification method type must be non-empty".into(),
            ));
        }
        if !is_hex_digest(&verification_method_material_digest) {
            return Err(VerificationFailure::Structural(
                "verification method material digest must be a 64-character hexadecimal digest"
                    .into(),
            ));
        }

        let controller_document_integrity =
            ControllerDocumentIntegrityAttestation::from_policy(
                &request.controller_document_integrity_policy,
                &controller_document_digest,
            )?;
        verification_method_lifecycle.validate_structure()?;
        controller_document_snapshot_scope.validate_for_reference_time(
            request.freshness.lifecycle_reference_time(),
        )?;

        let resolution = Self {
            schema_version: Self::SCHEMA_VERSION,
            verification_method: request.verification_method.clone(),
            verification_method_type,
            verification_method_material_digest,
            controller_document_ref,
            controller_document_id,
            resolved_verification_method_controller,
            verification_relationship: request.expected_verification_relationship.clone(),
            relationship_methods: methods,
            relationship_methods_digest,
            controller_document_digest,
            controller_document_integrity,
            verification_method_lifecycle,
            controller_document_snapshot_scope,
            controller_document_network_policy:
                request.controller_document_network_policy.clone(),
            controller_document_dereference: None,
        };
        resolution.verification_method_lifecycle.validate_for_use_at(
            request.freshness.lifecycle_reference_time(),
        )?;
        Ok(resolution)
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        if self.schema_version != Self::SCHEMA_VERSION {
            return Err(VerificationFailure::Structural(
                "unsupported verification method resolution schema version".into(),
            ));
        }
        let verification_method_url = Url::parse(self.verification_method.as_str())
            .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        if verification_method_url.fragment().is_none() {
            return Err(VerificationFailure::InvalidVerificationMethodUrl);
        }

        if self.verification_method_type.trim().is_empty() {
            return Err(VerificationFailure::Structural(
                "verification method type must be non-empty".into(),
            ));
        }
        if !is_hex_digest(&self.verification_method_material_digest) {
            return Err(VerificationFailure::Structural(
                "verification method material digest must be a 64-character hexadecimal digest"
                    .into(),
            ));
        }

        let mut expected_document_url =
            Url::parse(self.verification_method.as_str())
                .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        expected_document_url.set_fragment(None);
        if !url_values_equivalent(expected_document_url.as_str(), &self.controller_document_ref)
            .unwrap_or(false)
        {
            return Err(VerificationFailure::ControllerDocumentMismatch {
                expected: expected_document_url.to_string(),
                actual: self.controller_document_ref.clone(),
            });
        }

        let document_id = Url::parse(self.controller_document_id.as_str())
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        if !url_values_equivalent(document_id.as_str(), &self.controller_document_ref)
            .unwrap_or(false)
        {
            return Err(VerificationFailure::ControllerDocumentMismatch {
                expected: self.controller_document_ref.clone(),
                actual: self.controller_document_id.as_str().to_owned(),
            });
        }

        let resolved_controller =
            Url::parse(self.resolved_verification_method_controller.as_str())
                .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        if !url_values_equivalent(
            resolved_controller.as_str(),
            &self.controller_document_ref,
        )
        .unwrap_or(false)
        {
            return Err(VerificationFailure::ControllerMismatch {
                expected: ClaimControllerIdentity::new(self.controller_document_ref.clone())
                    .expect("validated controller document URL"),
                actual: self.resolved_verification_method_controller.clone(),
            });
        }

        self.verification_relationship.validate_structure()?;
        let mut canonical_methods = self.relationship_methods.clone();
        for method in &canonical_methods {
            method
                .validate_structure()
                .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
            Url::parse(method.as_str())
                .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        }
        let canonical_method_ids = canonical_methods
            .iter()
            .map(|method| {
                Url::parse(method.as_str())
                    .map(|url| url.to_string())
                    .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut unique_canonical_method_ids = canonical_method_ids.clone();
        unique_canonical_method_ids.sort();
        unique_canonical_method_ids.dedup();
        if unique_canonical_method_ids.len() != canonical_method_ids.len() {
            return Err(VerificationFailure::Structural(
                "relationship method set contains semantically duplicate URL values".into(),
            ));
        }

        canonical_methods.sort();
        canonical_methods.dedup();
        if canonical_methods != self.relationship_methods {
            return Err(VerificationFailure::Structural(
                "relationship method set must be sorted and unique".into(),
            ));
        }
        if !self
            .relationship_methods
            .iter()
            .any(|method| method == &self.verification_method)
        {
            return Err(VerificationFailure::VerificationMethodNotInRelationship);
        }
        let encoded_members = (
            "symthaea:verification-relationship-members:v1",
            self.verification_relationship.as_str(),
            self.relationship_methods
                .iter()
                .map(ClaimVerificationMethod::as_str)
                .collect::<Vec<_>>(),
        );
        let bytes = serde_json::to_vec(&encoded_members)
            .expect("verification relationship member set is serializable");
        if crate::sha256_hex(&bytes) != self.relationship_methods_digest {
            return Err(VerificationFailure::Structural(
                "relationship methods digest does not match member set".into(),
            ));
        }
        if !is_hex_digest(&self.relationship_methods_digest) {
            return Err(VerificationFailure::Structural(
                "relationship methods digest must be a 64-character hexadecimal digest".into(),
            ));
        }
        if !is_hex_digest(&self.controller_document_digest) {
            return Err(VerificationFailure::Structural(
                "controller document digest must be a 64-character hexadecimal digest".into(),
            ));
        }
        self.controller_document_integrity.validate_structure()?;
        if self.controller_document_integrity.actual_digest()
            != self.controller_document_digest
        {
            return Err(VerificationFailure::ControllerDocumentIntegrityMismatch {
                expected: self.controller_document_digest.clone(),
                actual: self.controller_document_integrity.actual_digest().to_owned(),
            });
        }
        self.verification_method_lifecycle.validate_structure()?;
        self.controller_document_snapshot_scope.validate_structure()?;
        self.controller_document_network_policy.validate_structure()?;
        let dereference = self
            .controller_document_dereference
            .as_ref()
            .ok_or(VerificationFailure::MissingControllerDocumentDereference)?;
        dereference.validate_against_policy(
            &self.controller_document_network_policy,
            &self.controller_document_ref,
            &self.controller_document_digest,
        )?;
        let snapshot_resolved_at = parse_timestamp(
            "controller document resolved at",
            self.controller_document_snapshot_scope.resolved_at(),
        )?;
        let dereference_resolved_at =
            parse_timestamp("controller document resolved at", &dereference.resolved_at)?;
        if snapshot_resolved_at != dereference_resolved_at {
            return Err(VerificationFailure::ControllerDocumentDereferenceTimeMismatch);
        }
        Ok(())
    }

    pub fn resolution_digest(&self) -> String {
        let encoded = (
            "symthaea:verification-method-resolution:v1",
            self.schema_version,
            self.verification_method.as_str(),
            self.verification_method_type.as_str(),
            self.verification_method_material_digest.as_str(),
            self.controller_document_ref.as_str(),
            self.controller_document_id.as_str(),
            self.resolved_verification_method_controller.as_str(),
            self.verification_relationship.as_str(),
            self.relationship_methods_digest.as_str(),
            self.controller_document_digest.as_str(),
            &self.controller_document_integrity,
            &self.verification_method_lifecycle,
            &self.controller_document_snapshot_scope,
            &self.controller_document_network_policy,
            &self.controller_document_dereference,
        );
        let bytes = serde_json::to_vec(&encoded)
            .expect("verification method resolution is serializable");
        crate::sha256_hex(&bytes)
    }

    pub fn matches_request(&self, request: &VerificationRequest) -> bool {
        self.validate_structure().is_ok()
            && url_values_equivalent(
                self.verification_method.as_str(),
                request.verification_method.as_str(),
            )
            .unwrap_or(false)
            && url_values_equivalent(
                self.resolved_verification_method_controller.as_str(),
                request.expected_controller.as_str(),
            )
            .unwrap_or(false)
            && self.verification_relationship == request.expected_verification_relationship
            && self.controller_document_network_policy
                == request.controller_document_network_policy
            && self
                .controller_document_integrity
                .matches_policy(&request.controller_document_integrity_policy)
            && self
                .verification_method_lifecycle
                .validate_for_use_at(request.freshness.lifecycle_reference_time())
                .is_ok()
            && self
                .controller_document_snapshot_scope
                .validate_for_reference_time(request.freshness.lifecycle_reference_time())
                .is_ok()
            && self
                .controller_document_dereference
                .as_ref()
                .is_some_and(|value| value.validate_against_request(request).is_ok())
    }
}

/// The exact structural and identity inputs an external verification adapter must
/// operate over before it can return cryptographic/controller evidence.
///
/// The request is derived from the claim itself, so adapters cannot accidentally
/// verify one claim while attaching evidence to another representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationRequest {
    pub schema_version: u16,
    pub claim_representation_digest: String,
    pub statement_digest: String,
    pub author: ClaimAuthorIdentity,
    pub proof_purpose: ClaimProofPurpose,
    /// Exact JCS-transformed representation digest that the request bound before
    /// cryptographic verification. This makes the representation binding durable
    /// alongside the cryptographic receipt.
    #[serde(default)]
    pub expected_transformed_document_digest: Option<String>,
    pub verification_method: ClaimVerificationMethod,
    pub expected_controller: ClaimControllerIdentity,
    pub expected_verification_relationship: ClaimVerificationRelationship,
    pub controller_document_integrity_policy: ControllerDocumentIntegrityPolicy,
    pub controller_document_network_policy: ControllerDocumentNetworkPolicy,
    pub freshness: VerificationFreshnessContext,
}

impl VerificationRequest {
    /// Create a request only when the claim has a complete local authorship binding
    /// for the exact expected purpose and controller.
    ///
    /// This is still structural. The adapter must separately resolve the verification
    /// method, bind it to the expected controller document, validate the permitted
    /// relationship, and verify the cryptographic proof.
    pub fn from_claim(
        claim: &FederatedClaim,
        expected_purpose: ClaimProofPurpose,
        expected_controller: ClaimControllerIdentity,
        expected_verification_relationship: ClaimVerificationRelationship,
        freshness: VerificationFreshnessContext,
    ) -> Result<Self, VerificationFailure> {
        claim
            .validate_structure()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?;
        freshness.validate()?;

        let authorship = claim
            .authorship_binding()
            .ok_or(VerificationFailure::MissingAuthorship)?;

        if authorship.author() != &claim.author {
            return Err(VerificationFailure::Structural(
                "claim authorship author must match declared claim author".into(),
            ));
        }
        if !authorship.proof_purpose_matches(&expected_purpose) {
            return Err(VerificationFailure::ProofPurposeMismatch {
                expected: expected_purpose,
                actual: authorship.proof_purpose().clone(),
            });
        }

        let verification_method = authorship
            .verification_method()
            .cloned()
            .ok_or(VerificationFailure::MissingVerificationMethod)?;
        let controller = authorship
            .verification_controller()
            .cloned()
            .ok_or(VerificationFailure::MissingVerificationController)?;

        if controller != expected_controller {
            return Err(VerificationFailure::ControllerMismatch {
                expected: expected_controller,
                actual: controller,
            });
        }

        let statement_digest = claim
            .statement_identity()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?
            .digest();

        Ok(Self {
            schema_version: VERIFICATION_REQUEST_SCHEMA_VERSION,
            claim_representation_digest: claim.canonical_digest(),
            statement_digest,
            author: claim.author.clone(),
            proof_purpose: expected_purpose,
            verification_method,
            expected_controller,
            expected_verification_relationship,
            expected_transformed_document_digest: None,
            controller_document_integrity_policy: ControllerDocumentIntegrityPolicy::Unpinned,
            controller_document_network_policy:
                ControllerDocumentNetworkPolicy::strict_for_url(
                    &format!("{verification_method}"),
                )?,
            freshness,
        })
    }

    pub fn with_expected_transformed_document_digest(
        mut self,
        digest: impl Into<String>,
    ) -> Result<Self, VerificationFailure> {
        let digest = digest.into();
        if !is_hex_digest(&digest) {
            return Err(VerificationFailure::Structural(
                "expected transformed document digest must be a 64-character hexadecimal digest"
                    .into(),
            ));
        }
        self.expected_transformed_document_digest = Some(digest);
        Ok(self)
    }

    pub fn with_controller_document_network_policy(
        mut self,
        policy: ControllerDocumentNetworkPolicy,
    ) -> Result<Self, VerificationFailure> {
        policy.validate_structure()?;
        self.controller_document_network_policy = policy;
        Ok(self)
    }

    pub fn with_controller_document_integrity(
        mut self,
        policy: ControllerDocumentIntegrityPolicy,
    ) -> Result<Self, VerificationFailure> {
        policy.validate_structure()?;
        self.controller_document_integrity_policy = policy;
        Ok(self)
    }

    pub fn controller_document_ref(&self) -> Result<String, VerificationFailure> {
        let method_url = Url::parse(self.verification_method.as_str())
            .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        if method_url.fragment().is_none() {
            return Err(VerificationFailure::InvalidVerificationMethodUrl);
        }
        let mut document_url = method_url;
        document_url.set_fragment(None);
        Ok(document_url.to_string())
    }

    /// Typed external dependencies required for the adapter-side resolution step.
    ///
    /// The core deliberately does not turn these into fetches; an adapter may resolve
    /// them through a DID/controller-document system, Holochain, a local cache, or any
    /// other substrate.
    pub fn dependencies(&self) -> Result<Vec<FederationDependency>, VerificationFailure> {
        Ok(vec![
            FederationDependency::VerificationMethod(self.verification_method.as_str().to_owned()),
            FederationDependency::ControllerDocument(self.controller_document_ref()?),
        ])
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        if self.schema_version != VERIFICATION_REQUEST_SCHEMA_VERSION {
            return Err(VerificationFailure::Structural(
                "unsupported verification request schema version".into(),
            ));
        }
        if !is_hex_digest(&self.claim_representation_digest) {
            return Err(VerificationFailure::Structural(
                "claim representation digest must be a 64-character hexadecimal digest".into(),
            ));
        }
        if !is_hex_digest(&self.statement_digest) {
            return Err(VerificationFailure::Structural(
                "statement digest must be a 64-character hexadecimal digest".into(),
            ));
        }
        self.author
            .validate_structure()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?;
        self.proof_purpose
            .validate_structure()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?;
        self.verification_method
            .validate_structure()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?;
        let verification_method_url = Url::parse(self.verification_method.as_str())
            .map_err(|_| VerificationFailure::InvalidVerificationMethodUrl)?;
        if verification_method_url.fragment().is_none() {
            return Err(VerificationFailure::InvalidVerificationMethodUrl);
        }
        self.expected_controller
            .validate_structure()
            .map_err(|reason| VerificationFailure::Structural(reason.to_owned()))?;
        Url::parse(self.expected_controller.as_str())
            .map_err(|_| VerificationFailure::InvalidControllerDocumentId)?;
        self.expected_verification_relationship.validate_structure()?;
        if let Some(digest) = &self.expected_transformed_document_digest {
            if !is_hex_digest(digest) {
                return Err(VerificationFailure::Structural(
                    "expected transformed document digest must be a 64-character hexadecimal digest"
                        .into(),
                ));
            }
        }
        self.controller_document_integrity_policy.validate_structure()?;
        self.controller_document_network_policy.validate_structure()?;
        self.freshness.validate()?;
        Ok(())
    }
}

/// Typed cryptographic proof result produced by a concrete adapter.
///
/// This receipt binds the exact cryptographic suite, proof representation,
/// proof-purpose/method identity, resolved public-key identity, suite-stage
/// hashes, cryptographic input, proof identity, and detached proof value.
///
/// It is an adapter attestation; the substrate-neutral core does not itself
/// perform cryptographic verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CryptographicVerificationReceipt {
    pub schema_version: u16,
    /// Identity of the exact claim representation this receipt is attached to.
    pub claim_representation_digest: String,
    /// Identity of the exact proposition/statement this receipt is attached to.
    pub statement_digest: String,
    pub cryptosuite: String,
    pub proof_type: String,
    pub proof_purpose: ClaimProofPurpose,
    /// Exact JCS-transformed representation digest that the request bound before
    /// cryptographic verification. This makes the representation binding durable
    /// alongside the cryptographic receipt.
    #[serde(default)]
    pub expected_transformed_document_digest: Option<String>,
    pub verification_method: ClaimVerificationMethod,
    pub verification_method_type: String,
    pub verification_method_material_digest: String,
    pub transformed_document_digest: String,
    pub proof_configuration_digest: String,
    pub cryptographic_input_digest: String,
    pub proof_digest: String,
    pub proof_value_multibase: String,
    pub freshness: VerificationFreshnessContext,
}

impl CryptographicVerificationReceipt {
    pub fn from_adapter_verification(
        request: &VerificationRequest,
        resolution: &VerificationMethodResolution,
        proof_type: impl Into<String>,
        cryptosuite: impl Into<String>,
        transformed_document_digest: impl Into<String>,
        proof_configuration_digest: impl Into<String>,
        cryptographic_input_digest: impl Into<String>,
        proof_digest: impl Into<String>,
        proof_value_multibase: impl Into<String>,
    ) -> Result<Self, VerificationFailure> {
        request.validate_structure()?;
        resolution.validate_structure()?;
        if !resolution.matches_request(request) {
            return Err(VerificationFailure::ResolutionRequestMismatch);
        }

        let receipt = Self {
            schema_version: CRYPTOGRAPHIC_VERIFICATION_RECEIPT_SCHEMA_VERSION,
            claim_representation_digest: request.claim_representation_digest.clone(),
            statement_digest: request.statement_digest.clone(),
            cryptosuite: cryptosuite.into(),
            proof_type: proof_type.into(),
            proof_purpose: request.proof_purpose.clone(),
            expected_transformed_document_digest: request
                .expected_transformed_document_digest
                .clone(),
            verification_method: resolution.verification_method.clone(),
            verification_method_type: resolution.verification_method_type.clone(),
            verification_method_material_digest:
                resolution.verification_method_material_digest.clone(),
            transformed_document_digest: transformed_document_digest.into(),
            proof_configuration_digest: proof_configuration_digest.into(),
            cryptographic_input_digest: cryptographic_input_digest.into(),
            proof_digest: proof_digest.into(),
            proof_value_multibase: proof_value_multibase.into(),
            freshness: request.freshness.clone(),
        };
        receipt.validate_against(request, resolution)?;
        Ok(receipt)
    }

    pub fn receipt_digest(&self) -> String {
        let encoded = (
            "symthaea:cryptographic-verification-receipt:v2",
            self.schema_version,
            &self.claim_representation_digest,
            &self.statement_digest,
            &self.cryptosuite,
            &self.proof_type,
            &self.proof_purpose,
            &self.expected_transformed_document_digest,
            &self.verification_method,
            &self.verification_method_type,
            &self.verification_method_material_digest,
            &self.transformed_document_digest,
            &self.proof_configuration_digest,
            &self.cryptographic_input_digest,
            &self.proof_digest,
            &self.proof_value_multibase,
            &self.freshness.replay_context_digest(),
            &self.freshness.verification_time,
            &self.freshness.proof_created,
            &self.freshness.proof_expires,
        );
        let bytes = serde_json::to_vec(&encoded)
            .expect("cryptographic verification receipt is serializable");
        crate::sha256_hex(&bytes)
    }

    pub fn validate_against(
        &self,
        request: &VerificationRequest,
        resolution: &VerificationMethodResolution,
    ) -> Result<(), VerificationFailure> {
        if self.schema_version != CRYPTOGRAPHIC_VERIFICATION_RECEIPT_SCHEMA_VERSION {
            return Err(VerificationFailure::Structural(
                "unsupported cryptographic verification receipt schema version".into(),
            ));
        }
        if !resolution.matches_request(request) {
            return Err(VerificationFailure::ResolutionRequestMismatch);
        }
        if !is_hex_digest(&self.claim_representation_digest)
            || self.claim_representation_digest != request.claim_representation_digest
        {
            return Err(VerificationFailure::Structural(
                "cryptographic receipt claim representation identity does not match request"
                    .into(),
            ));
        }
        if !is_hex_digest(&self.statement_digest) || self.statement_digest != request.statement_digest
        {
            return Err(VerificationFailure::Structural(
                "cryptographic receipt statement identity does not match request".into(),
            ));
        }
        if self.proof_type.trim().is_empty() || self.cryptosuite.trim().is_empty() {
            return Err(VerificationFailure::Structural(
                "cryptographic proof type and cryptosuite must be non-empty".into(),
            ));
        }
        if self.proof_purpose != request.proof_purpose {
            return Err(VerificationFailure::ProofPurposeMismatch {
                expected: request.proof_purpose.clone(),
                actual: self.proof_purpose.clone(),
            });
        }
        if self.expected_transformed_document_digest
            != request.expected_transformed_document_digest
        {
            return Err(VerificationFailure::TransformedDocumentDigestMismatch {
                expected: request.expected_transformed_document_digest.clone(),
                actual: self.expected_transformed_document_digest.clone(),
            });
        }
        if let Some(digest) = &self.expected_transformed_document_digest {
            if !is_hex_digest(digest) {
                return Err(VerificationFailure::Structural(
                    "cryptographic receipt expected transformed document digest must be a 64-character hexadecimal digest"
                        .into(),
                ));
            }
        }
        if let Some(expected) = &self.expected_transformed_document_digest {
            if expected != &self.transformed_document_digest {
                return Err(VerificationFailure::TransformedDocumentDigestMismatch {
                    expected: Some(expected.clone()),
                    actual: Some(self.transformed_document_digest.clone()),
                });
            }
        }
        if self.verification_method != resolution.verification_method {
            return Err(VerificationFailure::VerificationMethodMismatch {
                expected: resolution.verification_method.clone(),
                actual: self.verification_method.clone(),
            });
        }
        if self.verification_method_type != resolution.verification_method_type {
            return Err(VerificationFailure::Structural(
                "cryptographic receipt verification method type does not match resolution".into(),
            ));
        }
        if self.verification_method_material_digest
            != resolution.verification_method_material_digest
        {
            return Err(VerificationFailure::Structural(
                "cryptographic receipt verification material does not match resolution".into(),
            ));
        }
        for (name, value) in [
            ("transformed document digest", self.transformed_document_digest.as_str()),
            ("proof configuration digest", self.proof_configuration_digest.as_str()),
            ("cryptographic input digest", self.cryptographic_input_digest.as_str()),
            ("proof digest", self.proof_digest.as_str()),
        ] {
            if !is_hex_digest(value) {
                return Err(VerificationFailure::Structural(format!(
                    "{name} must be a 64-character hexadecimal digest"
                )));
            }
        }
        if !self.proof_value_multibase.starts_with('z') {
            return Err(VerificationFailure::Structural(
                "cryptographic proofValue must use the base58-btc multibase prefix".into(),
            ));
        }
        let proof_bytes = bs58::decode(&self.proof_value_multibase[1..])
            .into_vec()
            .map_err(|_| {
                VerificationFailure::Structural(
                    "cryptographic proofValue must contain valid base58-btc data".into(),
                )
            })?;
        if proof_bytes.is_empty() {
            return Err(VerificationFailure::Structural(
                "cryptographic proofValue must contain non-empty proof bytes".into(),
            ));
        }

        self.freshness.validate()?;
        if self.freshness != request.freshness {
            return Err(VerificationFailure::FreshnessMismatch);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationEvidence {
    pub schema_version: u16,
    pub claim_representation_digest: String,
    pub statement_digest: String,
    pub author: ClaimAuthorIdentity,
    pub proof_purpose: ClaimProofPurpose,
    /// Exact JCS-transformed representation binding carried through the evidence envelope.
    #[serde(default)]
    pub expected_transformed_document_digest: Option<String>,
    pub verification_method: ClaimVerificationMethod,
    pub controller: ClaimControllerIdentity,
    /// Controller identity actually read from the resolved verification-method definition.
    ///
    /// This is intentionally distinct from the identity of the controller document
    /// itself: a verification method's controller MUST be checked explicitly.
    pub resolved_verification_method_controller: ClaimControllerIdentity,
    /// Verification-method identity actually read from the resolved controller document.
    pub controller_document_verification_method: ClaimVerificationMethod,
    pub verification_relationship: ClaimVerificationRelationship,
    pub controller_document_ref: String,
    pub controller_document_digest: String,
    pub resolution: VerificationMethodResolution,
    pub cryptographic_verification: CryptographicVerificationReceipt,
    pub freshness: VerificationFreshnessContext,
}

impl VerificationEvidence {
    /// Construct adapter evidence after the adapter has completed its external
    /// resolution and cryptographic checks.
    pub fn from_adapter_attestation(
        request: &VerificationRequest,
        resolution: VerificationMethodResolution,
        cryptographic_verification: CryptographicVerificationReceipt,
    ) -> Result<Self, VerificationFailure> {
        request.validate_structure()?;
        resolution.validate_structure()?;
        if !resolution.matches_request(request) {
            return Err(VerificationFailure::ResolutionRequestMismatch);
        }
        cryptographic_verification.validate_against(request, &resolution)?;

        Ok(Self {
            schema_version: VERIFICATION_EVIDENCE_SCHEMA_VERSION,
            claim_representation_digest: request.claim_representation_digest.clone(),
            statement_digest: request.statement_digest.clone(),
            author: request.author.clone(),
            proof_purpose: request.proof_purpose.clone(),
            expected_transformed_document_digest: request
                .expected_transformed_document_digest
                .clone(),
            verification_method: resolution.verification_method.clone(),
            controller: request.expected_controller.clone(),
            resolved_verification_method_controller:
                resolution.resolved_verification_method_controller.clone(),
            controller_document_verification_method: resolution.verification_method.clone(),
            verification_relationship: resolution.verification_relationship.clone(),
            controller_document_ref: resolution.controller_document_ref.clone(),
            controller_document_digest: resolution.controller_document_digest.clone(),
            resolution,
            cryptographic_verification,
            freshness: request.freshness.clone(),
        })
    }

    /// A deterministic representation of the adapter's attestation record.
    ///
    /// This is an evidence-record identity, not a substitute for the cryptographic
    /// verification it describes.
    pub fn evidence_digest(&self) -> String {
        let encoded = (
            "symthaea:verification-evidence:v4",
            self.schema_version,
            VERIFICATION_EVIDENCE_DIGEST_VERSION,
            &self.claim_representation_digest,
            &self.statement_digest,
            self.author.as_str(),
            self.proof_purpose.as_str(),
            &self.expected_transformed_document_digest,
            self.verification_method.as_str(),
            self.controller.as_str(),
            self.resolved_verification_method_controller.as_str(),
            self.controller_document_verification_method.as_str(),
            &self.controller_document_ref,
            &self.controller_document_digest,
            &self.controller_document_integrity_digest(),
            &self.resolution.resolution_digest(),
            &self.verification_relationship,
            &self.cryptographic_verification.receipt_digest(),
            &self.freshness.replay_context_digest(),
            &self.freshness.verification_time,
            &self.freshness.proof_created,
            &self.freshness.proof_expires,
        );
        let bytes = serde_json::to_vec(&encoded)
            .expect("verification evidence is serializable");
        crate::sha256_hex(&bytes)
    }

    fn controller_document_integrity_digest(&self) -> String {
        self.resolution.controller_document_integrity.identity_digest()
    }

    pub fn validate_structure(&self) -> Result<(), VerificationFailure> {
        if self.schema_version != VERIFICATION_EVIDENCE_SCHEMA_VERSION {
            return Err(VerificationFailure::Structural(
                "unsupported verification evidence schema version".into(),
            ));
        }
        let request = VerificationRequest {
            schema_version: VERIFICATION_REQUEST_SCHEMA_VERSION,
            claim_representation_digest: self.claim_representation_digest.clone(),
            statement_digest: self.statement_digest.clone(),
            author: self.author.clone(),
            proof_purpose: self.proof_purpose.clone(),
            verification_method: self.verification_method.clone(),
            expected_controller: self.controller.clone(),
            expected_verification_relationship: self.verification_relationship.clone(),
            expected_transformed_document_digest: self
                .cryptographic_verification
                .expected_transformed_document_digest
                .clone(),
            controller_document_integrity_policy:
                match &self.resolution.controller_document_integrity {
                    ControllerDocumentIntegrityAttestation::Unpinned { .. } => {
                        ControllerDocumentIntegrityPolicy::Unpinned
                    }
                    ControllerDocumentIntegrityAttestation::Sha256Digest {
                        expected_digest,
                        ..
                    } => ControllerDocumentIntegrityPolicy::Sha256Digest(expected_digest.clone()),
                },
            controller_document_network_policy: self
                .resolution
                .controller_document_network_policy
                .clone(),
            freshness: self.freshness.clone(),
        };
        request.validate_structure()?;
        self.resolution.validate_structure()?;
        if !self.resolution.matches_request(&request) {
            return Err(VerificationFailure::ResolutionRequestMismatch);
        }
        if self.expected_transformed_document_digest
            != request.expected_transformed_document_digest
        {
            return Err(VerificationFailure::TransformedDocumentDigestMismatch {
                expected: request.expected_transformed_document_digest.clone(),
                actual: self.expected_transformed_document_digest.clone(),
            });
        }
        if self.resolution.controller_document_ref != self.controller_document_ref
            || self.resolution.controller_document_digest != self.controller_document_digest
            || self.resolution.verification_method != self.controller_document_verification_method
            || self.resolution.resolved_verification_method_controller
                != self.resolved_verification_method_controller
            || self.resolution.verification_relationship != self.verification_relationship
        {
            return Err(VerificationFailure::ResolutionEvidenceMismatch);
        }
        self.freshness.validate()?;
        if self.freshness != request.freshness {
            return Err(VerificationFailure::FreshnessMismatch);
        }
        self.resolved_verification_method_controller.validate_structure()?;
        self.controller_document_verification_method.validate_structure()?;
        if self.resolved_verification_method_controller != self.controller {
            return Err(VerificationFailure::ControllerMismatch {
                expected: self.controller.clone(),
                actual: self.resolved_verification_method_controller.clone(),
            });
        }
        if self.controller_document_verification_method != self.verification_method {
            return Err(VerificationFailure::VerificationMethodMismatch {
                expected: self.verification_method.clone(),
                actual: self.controller_document_verification_method.clone(),
            });
        }

        for (name, value) in [
            ("controller document reference", self.controller_document_ref.as_str()),
            ("verification relationship", self.verification_relationship.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(VerificationFailure::Structural(format!(
                    "{name} must be non-empty"
                )));
            }
        }
        self.cryptographic_verification
            .validate_against(&request, &self.resolution)?;

        if !is_hex_digest(&self.controller_document_digest) {
            return Err(VerificationFailure::Structural(
                "controller document digest must be a 64-character hexadecimal digest".into(),
            ));
        }
        Ok(())
    }

    pub fn matches_request(&self, request: &VerificationRequest) -> bool {
        self.validate_structure().is_ok()
            && self.claim_representation_digest == request.claim_representation_digest
            && self.statement_digest == request.statement_digest
            && self.author == request.author
            && self.proof_purpose == request.proof_purpose
            && self.expected_transformed_document_digest
                == request.expected_transformed_document_digest
            && self.verification_method == request.verification_method
            && self.controller == request.expected_controller
            && self.resolved_verification_method_controller == request.expected_controller
            && self.controller_document_verification_method == request.verification_method
            && self.verification_relationship == request.expected_verification_relationship
            && self.freshness == request.freshness
            && self.resolution.matches_request(request)
    }
}

/// Outcome vocabulary for an adapter.
///
/// `Unresolved` is intentionally separate from `Invalid`: a missing controller
/// document or verification method may be retryable, while a cryptographic or
/// structural failure is definitive for the current artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationOutcome {
    Verified(VerificationEvidence),
    Invalid(VerificationFailure),
    Unresolved(Vec<FederationDependency>),
}

impl VerificationOutcome {
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationFailure {
    Structural(String),
    MissingAuthorship,
    MissingVerificationMethod,
    MissingVerificationController,
    ProofPurposeMismatch {
        expected: ClaimProofPurpose,
        actual: ClaimProofPurpose,
    },
    ControllerMismatch {
        expected: ClaimControllerIdentity,
        actual: ClaimControllerIdentity,
    },
    VerificationMethodMismatch {
        expected: ClaimVerificationMethod,
        actual: ClaimVerificationMethod,
    },
    VerificationMethodNotFound,
    InvalidVerificationMethodUrl,
    InvalidControllerDocumentId,
    ControllerDocumentMismatch {
        expected: String,
        actual: String,
    },
    VerificationRelationshipMismatch {
        expected: ClaimVerificationRelationship,
        actual: ClaimVerificationRelationship,
    },
    VerificationMethodNotInRelationship,
    InvalidTimestamp {
        field: &'static str,
        value: String,
    },
    InvalidValidityWindow,
    ProofCreatedInFuture,
    ProofExpired,
    DomainMismatch {
        expected: String,
        actual: Option<String>,
    },
    ChallengeMismatch {
        expected: String,
        actual: Option<String>,
    },
    FreshnessMismatch,
    ResolutionRequestMismatch,
    ResolutionEvidenceMismatch,
    ControllerDocumentIntegrityMismatch {
        expected: String,
        actual: String,
    },
    VerificationMethodExpired {
        at: String,
    },
    VerificationMethodRevoked {
        at: String,
    },
    HistoricalStateRequired,
    HistoricalStateMismatch,
    InvalidSnapshotOrdering,
    InvalidControllerDocumentUrl,
    ControllerDocumentNetworkPolicyViolation,
    ControllerDocumentEffectiveUrlMismatch,
    ControllerDocumentRedirectLimitExceeded,
    ControllerDocumentResponseTooLarge,
    MissingControllerDocumentDereference,
    ControllerDocumentDereferenceMismatch,
    ControllerDocumentDereferenceTimeMismatch,
    MissingExpectedTransformedDocumentDigest,
    TransformedDocumentDigestMismatch {
        expected: Option<String>,
        actual: Option<String>,
    },
    CryptographicVerificationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CanonicalAdmissionReceipt, ProvenanceRelation, ProvenanceRelationKind,
        ProvenanceValidationReport, ProvenanceView,
    };

    fn freshness(
        proof_created: Option<&str>,
        proof_expires: Option<&str>,
        proof_domain: Option<&str>,
        proof_challenge: Option<&str>,
        verification_time: &str,
        expected_domain: Option<&str>,
        expected_challenge: Option<&str>,
    ) -> VerificationFreshnessContext {
        VerificationFreshnessContext {
            proof_created: proof_created.map(str::to_owned),
            proof_expires: proof_expires.map(str::to_owned),
            proof_domain: proof_domain.map(str::to_owned),
            proof_challenge: proof_challenge.map(str::to_owned),
            verification_time: verification_time.to_owned(),
            expected_domain: expected_domain.map(str::to_owned),
            expected_challenge: expected_challenge.map(str::to_owned),
        }
    }

    fn default_freshness() -> VerificationFreshnessContext {
        freshness(
            Some("2026-10-05T00:00:00Z"),
            Some("2026-10-05T03:00:00Z"),
            Some("example.test"),
            Some("challenge-1"),
            "2026-10-05T02:00:00Z",
            Some("example.test"),
            Some("challenge-1"),
        )
    }

    fn fixture_claim() -> FederatedClaim {
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let validation = ProvenanceValidationReport::from_relations(std::slice::from_ref(&relation));
        let view = ProvenanceView::from_relations(std::slice::from_ref(&relation), validation.clone())
            .unwrap();
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-verification",
            Some("frontier:verification".into()),
            "canonical:verification",
            Some("family:verification".into()),
            view.snapshot_digest.clone(),
            view.validation.validator_version.clone(),
            view.validation.snapshot_schema_version,
        )
        .unwrap();

        FederatedClaim::new(
            "claim:verification",
            "canonical:verification",
            "family:verification",
            "author:verification",
            "statement:verification",
            view,
            receipt,
        )
        .unwrap()
        .with_authorship(
            crate::ClaimAuthorship::new(
                crate::ClaimAuthorIdentity::new("author:verification").unwrap(),
                ClaimProofPurpose::new("assertionMethod").unwrap(),
                Some(ClaimVerificationMethod::new(
                    "https://example.test/controller#key-1",
                ).unwrap()),
            )
            .unwrap()
            .with_verification_controller(
                ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn resolved_method(request: &VerificationRequest) -> VerificationMethodResolution {
        let resolution = VerificationMethodResolution::from_controller_document(
            request,
            "https://example.test/controller",
            ClaimControllerDocumentIdentity::new("https://example.test/controller").unwrap(),
            request.verification_method.clone(),
            "Multikey",
            &"44".repeat(32),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            &[request.verification_method.clone()],
            &"11".repeat(32),
            VerificationMethodLifecycle::new(None, None).unwrap(),
            ControllerDocumentSnapshotScope::historical_at(
                request.freshness.lifecycle_reference_time(),
                "snapshot:verification-history",
                request.freshness.verification_time.as_str(),
            )
            .unwrap(),
        )
        .unwrap();
        let dereference = ControllerDocumentDereferenceAttestation::from_adapter(
            request,
            "https://example.test/controller",
            "application/cid",
            1024,
            0,
            request.freshness.verification_time.clone(),
            ControllerDocumentResolutionSource::HistoricalRegistry,
            &"11".repeat(32),
            None,
        )
        .unwrap();
        resolution
            .with_controller_document_dereference(dereference, request)
            .unwrap()
    }

    #[test]
    fn snapshot_scope_rejects_current_state_for_historical_evaluation() {
        let current =
            ControllerDocumentSnapshotScope::current("2026-10-05T02:00:00Z").unwrap();
        assert!(matches!(
            current.validate_for_reference_time("2026-10-05T01:59:59Z"),
            Err(VerificationFailure::HistoricalStateRequired)
        ));
        assert!(current.validate_for_reference_time("2026-10-05T02:00:00Z").is_ok());

        let historical = ControllerDocumentSnapshotScope::historical_at(
            "2026-10-05T01:00:00Z",
            "snapshot:2026-10-05T01:00:00Z",
            "2026-10-05T02:00:00Z",
        )
        .unwrap();
        assert!(historical
            .validate_for_reference_time("2026-10-05T01:00:00+00:00")
            .is_ok());
        assert!(matches!(
            historical.validate_for_reference_time("2026-10-05T01:00:01Z"),
            Err(VerificationFailure::HistoricalStateMismatch)
        ));

        let backwards = ControllerDocumentSnapshotScope::historical_at(
            "2026-10-05T02:00:00Z",
            "snapshot:future",
            "2026-10-05T01:59:59Z",
        )
        .unwrap();
        assert!(matches!(
            backwards.validate_for_reference_time("2026-10-05T02:00:00Z"),
            Err(VerificationFailure::InvalidSnapshotOrdering)
        ));
    }

    #[test]
    fn controller_document_identity_is_a_distinct_namespace() {
        assert!(ClaimControllerDocumentIdentity::new(
            "https://example.test/controller"
        ).unwrap().validate_structure().is_ok());
        assert!(matches!(
            ClaimControllerDocumentIdentity::new("   "),
            Err(VerificationFailure::Structural(_))
        ));
        assert!(matches!(
            ClaimControllerDocumentIdentity::new("not-a-url"),
            Err(VerificationFailure::InvalidControllerDocumentId)
        ));
    }

    #[test]
    fn resolution_rejects_method_absent_from_requested_relationship() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let result = VerificationMethodResolution::from_controller_document(
            &request,
            "https://example.test/controller",
            ClaimControllerDocumentIdentity::new("https://example.test/controller").unwrap(),
            request.verification_method.clone(),
            "Multikey",
            &"44".repeat(32),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            &[],
            &"11".repeat(32),
            VerificationMethodLifecycle::new(None, None).unwrap(),
            ControllerDocumentSnapshotScope::historical_at(
                request.freshness.lifecycle_reference_time(),
                "snapshot:verification-history",
                request.freshness.verification_time.as_str(),
            )
            .unwrap(),
        );

        assert!(matches!(
            result,
            Err(VerificationFailure::VerificationMethodNotInRelationship)
        ));
    }

    #[test]
    fn resolution_receipt_enforces_exact_retrieval_invariants() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let resolution = resolved_method(&request);
        assert!(resolution.validate_structure().is_ok());
        assert!(resolution.matches_request(&request));
        assert!(!resolution.resolution_digest().is_empty());

        let mut wrong_ref = resolution.clone();
        wrong_ref.controller_document_ref = "https://other.example".into();
        assert!(matches!(
            wrong_ref.validate_structure(),
            Err(VerificationFailure::ControllerDocumentMismatch { .. })
        ));

        let mut wrong_controller = resolution.clone();
        wrong_controller.resolved_verification_method_controller =
            ClaimControllerIdentity::new("https://other.example").unwrap();
        assert!(matches!(
            wrong_controller.validate_structure(),
            Err(VerificationFailure::ControllerMismatch { .. })
        ));

        let mut missing_method = resolution.clone();
        missing_method.relationship_methods.clear();
        assert!(matches!(
            missing_method.validate_structure(),
            Err(VerificationFailure::VerificationMethodNotInRelationship)
        ));
    }

    #[test]
    fn deserialized_resolution_rechecks_relationship_membership_and_digest() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let resolution = resolved_method(&request);
        let mut value = serde_json::to_value(&resolution).unwrap();

        value["relationship_methods"] = serde_json::json!([]);
        let decoded: VerificationMethodResolution = serde_json::from_value(value).unwrap();
        assert!(matches!(
            decoded.validate_structure(),
            Err(VerificationFailure::VerificationMethodNotInRelationship)
        ));

        let mut value = serde_json::to_value(&resolution).unwrap();
        value["relationship_methods_digest"] = serde_json::json!("22".repeat(32));
        let decoded: VerificationMethodResolution = serde_json::from_value(value).unwrap();
        assert!(matches!(
            decoded.validate_structure(),
            Err(VerificationFailure::Structural(_))
        ));
    }

    #[test]
    fn dereference_multibase_digest_must_match_the_document_digest() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let valid = ControllerDocumentDereferenceAttestation::from_adapter(
            &request,
            "https://example.test/controller",
            "application/cid",
            1024,
            0,
            request.freshness.verification_time.clone(),
            ControllerDocumentResolutionSource::Network,
            &"11".repeat(32),
            Some("zQmPVGjYFugq4XUyBfoTHG6c3qxfBS26jEdaFM1gdAVuMZ2".into()),
        )
        .unwrap();
        assert_eq!(
            valid.digest_multibase.as_deref(),
            Some("zQmPVGjYFugq4XUyBfoTHG6c3qxfBS26jEdaFM1gdAVuMZ2")
        );

        let mismatch = ControllerDocumentDereferenceAttestation::from_adapter(
            &request,
            "https://example.test/controller",
            "application/cid",
            1024,
            0,
            request.freshness.verification_time.clone(),
            ControllerDocumentResolutionSource::Network,
            &"11".repeat(32),
            Some("zQmPVGjYFugq4XUyBfoTHG6c3qxfBS26jEdaFM1gdAVuMAA".into()),
        );
        assert!(matches!(
            mismatch,
            Err(VerificationFailure::ControllerDocumentIntegrityMismatch { .. })
                | Err(VerificationFailure::Structural(_))
        ));
    }

    #[test]
    fn digest_multibase_uses_standards_property_name_on_the_wire() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let dereference = ControllerDocumentDereferenceAttestation::from_adapter(
            &request,
            "https://example.test/controller",
            "application/cid",
            1024,
            0,
            request.freshness.verification_time.clone(),
            ControllerDocumentResolutionSource::Network,
            &"11".repeat(32),
            Some("zQmPVGjYFugq4XUyBfoTHG6c3qxfBS26jEdaFM1gdAVuMZ2".into()),
        )
        .unwrap();

        let encoded = serde_json::to_value(&dereference).unwrap();
        assert!(encoded.get("digestMultibase").is_some());
        assert!(encoded.get("digest_multibase").is_none());

        let mut legacy = encoded.clone();
        let digest = legacy["digestMultibase"].take();
        legacy["digest_multibase"] = digest;
        let decoded: ControllerDocumentDereferenceAttestation =
            serde_json::from_value(legacy).unwrap();
        assert_eq!(decoded.digest_multibase, dereference.digest_multibase);
    }

    #[test]
    fn deserialized_dereference_receipt_rechecks_policy_and_effective_url() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let resolution = resolved_method(&request);
        let mut value = serde_json::to_value(&resolution).unwrap();
        value["controller_document_dereference"]["effective_url"] =
            serde_json::json!("https://other.example/controller");
        let decoded: VerificationMethodResolution = serde_json::from_value(value).unwrap();
        assert!(matches!(
            decoded.validate_structure(),
            Err(VerificationFailure::ControllerDocumentEffectiveUrlMismatch)
        ));

        let mut value = serde_json::to_value(&resolution).unwrap();
        value["controller_document_network_policy"]["allowed_schemes"] =
            serde_json::json!(["http"]);
        let decoded: VerificationMethodResolution = serde_json::from_value(value).unwrap();
        assert!(matches!(
            decoded.validate_structure(),
            Err(VerificationFailure::ControllerDocumentNetworkPolicyViolation)
        ));
    }

    #[test]
    fn deserialized_dereference_receipt_rejects_empty_media_type() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let resolution = resolved_method(&request);
        let mut value = serde_json::to_value(&resolution).unwrap();
        value["controller_document_dereference"]["response_media_type"] = serde_json::json!("   ");
        let decoded: VerificationMethodResolution = serde_json::from_value(value).unwrap();

        assert!(matches!(
            decoded.validate_structure(),
            Err(VerificationFailure::Structural(message))
                if message.contains("response media type must be non-empty")
        ));
    }

    #[test]
    fn controller_document_integrity_policy_binds_the_resolved_snapshot() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap()
        .with_controller_document_integrity(
            ControllerDocumentIntegrityPolicy::Sha256Digest("11".repeat(32)),
        )
        .unwrap();

        let resolution = {
            let resolution = VerificationMethodResolution::from_controller_document(
                &request,
                "https://example.test/controller",
                ClaimControllerDocumentIdentity::new("https://example.test/controller").unwrap(),
                request.verification_method.clone(),
                "Multikey",
                &"44".repeat(32),
                ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
                &[request.verification_method.clone()],
                &"11".repeat(32),
                VerificationMethodLifecycle::new(None, None).unwrap(),
                ControllerDocumentSnapshotScope::historical_at(
                    request.freshness.lifecycle_reference_time(),
                    "snapshot:verification-history",
                    request.freshness.verification_time.as_str(),
                )
                .unwrap(),
            )
            .unwrap();
            let dereference = ControllerDocumentDereferenceAttestation::from_adapter(
                &request,
                "https://example.test/controller",
                "application/cid",
                1024,
                0,
                request.freshness.verification_time.clone(),
                ControllerDocumentResolutionSource::HistoricalRegistry,
                &"11".repeat(32),
                Some("zQmPVGjYFugq4XUyBfoTHG6c3qxfBS26jEdaFM1gdAVuMZ2".into()),
            )
            .unwrap();
            resolution
                .with_controller_document_dereference(dereference, &request)
                .unwrap()
        };
        assert!(resolution
            .controller_document_integrity
            .matches_policy(&ControllerDocumentIntegrityPolicy::Sha256Digest(
                "11".repeat(32)
            )));

        assert!(matches!(
            VerificationMethodResolution::from_controller_document(
                &request,
                "https://example.test/controller",
                ClaimControllerDocumentIdentity::new("https://example.test/controller").unwrap(),
                request.verification_method.clone(),
                "Multikey",
                &"44".repeat(32),
                ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
                &[request.verification_method.clone()],
                &"22".repeat(32),
                VerificationMethodLifecycle::new(None, None).unwrap(),
                ControllerDocumentSnapshotScope::historical_at(
                    request.freshness.lifecycle_reference_time(),
                    "snapshot:verification-history",
                    request.freshness.verification_time.as_str(),
                )
                .unwrap(),
            ),
            Err(VerificationFailure::ControllerDocumentIntegrityMismatch { .. })
        ));
    }

    #[test]
    fn resolution_cannot_downgrade_request_network_policy() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let weaker = ControllerDocumentNetworkPolicy {
            allowed_schemes: vec!["https".into()],
            max_response_bytes: 16 * 1024 * 1024,
            max_redirects: 4,
            require_effective_url_match: false,
        };
        let weaker_request = request
            .clone()
            .with_controller_document_network_policy(weaker)
            .unwrap();

        let resolution = resolved_method(&weaker_request);
        assert!(resolution.matches_request(&weaker_request));
        assert!(!resolution.matches_request(&request));
    }

    #[test]
    fn timestamp_parser_matches_xsd11_date_time_stamp_boundary() {
        let end_of_day = parse_timestamp("timestamp", "2026-10-05T24:00:00Z").unwrap();
        let next_day = parse_timestamp("timestamp", "2026-10-06T00:00:00Z").unwrap();
        assert_eq!(end_of_day, next_day);

        let end_of_day_fraction =
            parse_timestamp("timestamp", "2026-10-05T24:00:00.000Z").unwrap();
        assert_eq!(end_of_day_fraction, next_day);

        for invalid in [
            "2026-10-05T00:00:60Z",
            "2026-10-05T24:00:00.1Z",
            "2026-10-05T23:59:59",
            "2026-10-05T23:59:59+14:01",
        ] {
            assert!(matches!(
                parse_timestamp("timestamp", invalid),
                Err(VerificationFailure::InvalidTimestamp { .. })
            ));
        }
    }

    #[test]
    fn url_identity_uses_parsed_serialized_equivalence() {
        assert!(url_values_equivalent(
            "https://EXAMPLE.TEST:443/controller/./#key-1",
            "https://example.test/controller/#key-1",
        )
        .unwrap());

        assert!(!url_values_equivalent(
            "https://example.test/controller#key-1",
            "https://example.test/controller#key-2",
        )
        .unwrap());

        assert!(url_values_equivalent(
            "https://example.test/controller",
            "https://example.test/controller",
        )
        .unwrap());
    }

    #[test]
    fn resolution_accepts_equivalent_controller_document_urls_without_rewriting_provenance() {
        let claim = fixture_claim();
        let mut request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://EXAMPLE.TEST/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();
        request.verification_method =
            ClaimVerificationMethod::new("https://EXAMPLE.TEST:443/controller/../controller#key-1")
                .unwrap();

        let resolution = VerificationMethodResolution::from_controller_document(
            &request,
            "https://example.test/controller",
            ClaimControllerDocumentIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationMethod::new("https://example.test/controller#key-1").unwrap(),
            "Multikey",
            &"44".repeat(32),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            &[ClaimVerificationMethod::new(
                "https://example.test/controller#key-1",
            )
            .unwrap()],
            &"11".repeat(32),
            VerificationMethodLifecycle::new(None, None).unwrap(),
            ControllerDocumentSnapshotScope::historical_at(
                request.freshness.lifecycle_reference_time(),
                "snapshot:verification-url-equivalence",
                request.freshness.verification_time.as_str(),
            )
            .unwrap(),
        );

        assert!(resolution.is_ok(), "{resolution:?}");
        let resolution = resolution.unwrap();
        assert_eq!(
            resolution.controller_document_ref,
            "https://example.test/controller/"
        );
        assert_eq!(
            request.expected_controller.as_str(),
            "https://EXAMPLE.TEST/controller"
        );
    }

    #[test]
    fn deserialized_resolution_rejects_semantically_duplicate_relationship_methods() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let mut resolution = resolved_method(&request);
        resolution.relationship_methods.push(
            ClaimVerificationMethod::new(
                "https://EXAMPLE.TEST:443/controller/../controller#key-1",
            )
            .unwrap(),
        );
        resolution.relationship_methods.sort();

        assert!(matches!(
            resolution.validate_structure(),
            Err(VerificationFailure::Structural(message))
                if message.contains("semantically duplicate URL values")
        ));
    }

    #[test]
    fn deserialized_verification_request_and_resolution_require_fragment_identifiers() {
        let claim = fixture_claim();
        let mut request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        request.verification_method =
            ClaimVerificationMethod::new("https://example.test/controller").unwrap();
        assert!(matches!(
            request.validate_structure(),
            Err(VerificationFailure::InvalidVerificationMethodUrl)
        ));

        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let mut resolution = resolved_method(&request);
        resolution.verification_method =
            ClaimVerificationMethod::new("https://example.test/controller").unwrap();
        assert!(matches!(
            resolution.validate_structure(),
            Err(VerificationFailure::InvalidVerificationMethodUrl)
        ));
    }

    #[test]
    fn lifecycle_respects_proof_time_and_requires_historical_state_for_post_revocation_use() {
        let revoked_later =
            VerificationMethodLifecycle::new(None, Some("2026-10-05T02:30:00Z")).unwrap();
        assert!(revoked_later
            .validate_for_use_at("2026-10-05T02:29:59Z")
            .is_ok());
        assert!(matches!(
            revoked_later.validate_for_use_at("2026-10-05T02:30:00Z"),
            Err(VerificationFailure::VerificationMethodRevoked { .. })
        ));

        let expired =
            VerificationMethodLifecycle::new(Some("2026-10-05T02:00:00Z"), None).unwrap();
        assert!(matches!(
            expired.validate_for_use_at("2026-10-05T02:00:00Z"),
            Err(VerificationFailure::VerificationMethodExpired { .. })
        ));

        let freshness_before_revocation = freshness(
            Some("2026-10-05T02:20:00Z"),
            None,
            Some("example.test"),
            Some("challenge-1"),
            "2026-10-05T03:00:00Z",
            Some("example.test"),
            Some("challenge-1"),
        );
        assert!(revoked_later
            .validate_for_use_at(freshness_before_revocation.lifecycle_reference_time())
            .is_ok());
    }

    #[test]
    fn resolution_rejects_current_snapshot_for_historical_proof() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let result = VerificationMethodResolution::from_controller_document(
            &request,
            "https://example.test/controller",
            ClaimControllerDocumentIdentity::new("https://example.test/controller").unwrap(),
            request.verification_method.clone(),
            "Multikey",
            &"44".repeat(32),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            &[request.verification_method.clone()],
            &"11".repeat(32),
            VerificationMethodLifecycle::new(None, None).unwrap(),
            ControllerDocumentSnapshotScope::current(
                request.freshness.verification_time.as_str(),
            )
            .unwrap(),
        );

        assert!(matches!(
            result,
            Err(VerificationFailure::HistoricalStateRequired)
        ));
    }

    #[test]
    fn resolution_rejects_method_lifecycle_failure_at_proof_time() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let result = VerificationMethodResolution::from_controller_document(
            &request,
            "https://example.test/controller",
            ClaimControllerDocumentIdentity::new("https://example.test/controller").unwrap(),
            request.verification_method.clone(),
            "Multikey",
            &"44".repeat(32),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            &[request.verification_method.clone()],
            &"11".repeat(32),
            VerificationMethodLifecycle::new(
                Some("2026-10-05T00:30:00Z"),
                None,
            )
            .unwrap(),
            ControllerDocumentSnapshotScope::historical_at(
                request.freshness.lifecycle_reference_time(),
                "snapshot:verification-history",
                request.freshness.verification_time.as_str(),
            )
            .unwrap(),
        );

        assert!(matches!(
            result,
            Err(VerificationFailure::VerificationMethodExpired { .. })
        ));
    }

    #[test]
    fn evidence_cannot_turn_an_unpinned_request_into_a_pinned_claim() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let mut evidence = make_evidence(
            &request,
            &"11".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();
        evidence.resolution.controller_document_integrity =
            ControllerDocumentIntegrityAttestation::Sha256Digest {
                expected_digest: "11".repeat(32),
                actual_digest: "11".repeat(32),
            };
        assert!(evidence.validate_structure().is_ok());
        assert!(!evidence.matches_request(&request));
    }

    #[test]
    fn dereference_attestation_rejects_redirects_oversize_and_scheme_downgrade() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let redirected = ControllerDocumentDereferenceAttestation::from_adapter(
            &request,
            "https://other.example/controller",
            "application/cid",
            1024,
            1,
            request.freshness.verification_time.clone(),
            ControllerDocumentResolutionSource::Network,
            &"11".repeat(32),
            None,
        );
        assert!(matches!(
            redirected,
            Err(VerificationFailure::ControllerDocumentEffectiveUrlMismatch)
        ));

        let oversized = ControllerDocumentDereferenceAttestation::from_adapter(
            &request,
            "https://example.test/controller",
            "application/cid",
            5 * 1024 * 1024,
            0,
            request.freshness.verification_time.clone(),
            ControllerDocumentResolutionSource::Network,
            &"11".repeat(32),
            None,
        );
        assert!(matches!(
            oversized,
            Err(VerificationFailure::ControllerDocumentResponseTooLarge)
        ));

        let mut downgrade_policy = request.clone();
        downgrade_policy.controller_document_network_policy.allowed_schemes =
            vec!["http".into()];
        let scheme = ControllerDocumentDereferenceAttestation::from_adapter(
            &downgrade_policy,
            "https://example.test/controller",
            "application/cid",
            1024,
            0,
            downgrade_policy.freshness.verification_time.clone(),
            ControllerDocumentResolutionSource::Network,
            &"11".repeat(32),
            None,
        );
        assert!(matches!(
            scheme,
            Err(VerificationFailure::ControllerDocumentNetworkPolicyViolation)
        ));

        let pinned_request = request
            .clone()
            .with_controller_document_integrity(
                ControllerDocumentIntegrityPolicy::Sha256Digest("11".repeat(32)),
            )
            .unwrap();
        let pinned_mismatch = ControllerDocumentDereferenceAttestation::from_adapter(
            &pinned_request,
            "https://example.test/controller",
            "application/cid",
            1024,
            0,
            pinned_request.freshness.verification_time.clone(),
            ControllerDocumentResolutionSource::Cache,
            &"22".repeat(32),
            None,
        );
        assert!(matches!(
            pinned_mismatch,
            Err(VerificationFailure::ControllerDocumentIntegrityMismatch { .. })
        ));
    }

    #[test]
    fn resolution_rejects_dereference_snapshot_time_mismatch() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let mut resolution = resolved_method(&request);
        let dereference = ControllerDocumentDereferenceAttestation::from_adapter(
            &request,
            "https://example.test/controller",
            "application/cid",
            1024,
            0,
            "2026-10-05T02:10:00Z",
            ControllerDocumentResolutionSource::HistoricalRegistry,
            &"11".repeat(32),
            None,
        )
        .unwrap();
        resolution.controller_document_dereference = Some(dereference);

        assert!(matches!(
            resolution.validate_structure(),
            Err(VerificationFailure::ControllerDocumentDereferenceTimeMismatch)
        ));
    }

    #[test]
    fn resolution_digest_changes_when_verification_material_identity_changes() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let base = resolved_method(&request);
        let mut changed_type = base.clone();
        changed_type.verification_method_type = "JsonWebKey".into();
        assert_ne!(base.resolution_digest(), changed_type.resolution_digest());

        let mut changed_material = base;
        changed_material.verification_method_material_digest = "55".repeat(32);
        assert_ne!(changed_type.resolution_digest(), changed_material.resolution_digest());
    }

    #[test]
    fn resolution_digest_changes_when_integrity_policy_or_lifecycle_changes() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let base = resolved_method(&request);
        let mut pinned = base.clone();
        pinned.controller_document_integrity =
            ControllerDocumentIntegrityAttestation::Sha256Digest {
                expected_digest: "11".repeat(32),
                actual_digest: "11".repeat(32),
            };
        assert_ne!(base.resolution_digest(), pinned.resolution_digest());

        let mut lifecycle = base;
        lifecycle.verification_method_lifecycle =
            VerificationMethodLifecycle::new(None, Some("2026-10-05T02:30:00Z")).unwrap();
        assert_ne!(pinned.resolution_digest(), lifecycle.resolution_digest());
    }

    fn test_cryptographic_receipt(
        request: &VerificationRequest,
        resolution: &VerificationMethodResolution,
    ) -> CryptographicVerificationReceipt {
        CryptographicVerificationReceipt {
            schema_version: CRYPTOGRAPHIC_VERIFICATION_RECEIPT_SCHEMA_VERSION,
            claim_representation_digest: request.claim_representation_digest.clone(),
            statement_digest: request.statement_digest.clone(),
            cryptosuite: "ed25519-test".into(),
            proof_type: "DataIntegrityProof".into(),

            proof_purpose: request.proof_purpose.clone(),
            expected_transformed_document_digest: request
                .expected_transformed_document_digest
                .clone(),
            verification_method: resolution.verification_method.clone(),
            verification_method_type: resolution.verification_method_type.clone(),
            verification_method_material_digest:
                resolution.verification_method_material_digest.clone(),
            transformed_document_digest: "22".repeat(32),
            proof_configuration_digest: "33".repeat(32),
            cryptographic_input_digest: "44".repeat(32),
            proof_digest: "55".repeat(32),
            proof_value_multibase: format!("z{}", bs58::encode([0u8; 64]).into_string()),
            freshness: request.freshness.clone(),
        }
    }

    fn make_evidence(
        request: &VerificationRequest,
        controller_document_digest: &str,
        _cryptosuite: &str,
        _signed_payload_digest: &str,
        _proof_digest: &str,
    ) -> Result<VerificationEvidence, VerificationFailure> {
        let mut resolution = resolved_method(request);
        resolution.controller_document_digest = controller_document_digest.to_owned();
        resolution.validate_structure()?;
        VerificationEvidence::from_adapter_attestation(
            request,
            resolution.clone(),
            test_cryptographic_receipt(request, &resolution),
        )
    }




    #[test]
    fn cryptographic_receipt_preserves_exact_transformed_representation_binding() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap()
        .with_expected_transformed_document_digest("22".repeat(32))
        .unwrap();

        let resolution = resolved_method(&request);
        let receipt = test_cryptographic_receipt(&request, &resolution);
        assert_eq!(
            receipt.expected_transformed_document_digest.as_deref(),
            Some("22".repeat(32).as_str())
        );
        assert!(receipt.validate_against(&request, &resolution).is_ok());

        let evidence =
            VerificationEvidence::from_adapter_attestation(&request, resolution, receipt).unwrap();
        let encoded = serde_json::to_vec(&evidence).unwrap();
        let decoded: VerificationEvidence = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            decoded
                .cryptographic_verification
                .expected_transformed_document_digest
                .as_deref(),
            Some("22".repeat(32).as_str())
        );
        assert!(decoded.validate_structure().is_ok());

        let mut tampered = decoded;
        tampered
            .cryptographic_verification
            .expected_transformed_document_digest = Some("33".repeat(32));
        assert!(matches!(
            tampered.validate_structure(),
            Err(VerificationFailure::TransformedDocumentDigestMismatch { .. })
        ));
    }

    #[test]
    fn freshness_context_rejects_future_expiry_and_mismatched_replay_inputs() {
        let valid = default_freshness();
        assert!(valid.validate().is_ok());

        let mut future_created = valid.clone();
        future_created.proof_created = Some("2026-10-05T04:00:00Z".into());
        assert!(matches!(
            future_created.validate(),
            Err(VerificationFailure::ProofCreatedInFuture)
        ));

        let mut expired = valid.clone();
        expired.proof_expires = Some("2026-10-05T02:00:00Z".into());
        assert!(matches!(
            expired.validate(),
            Err(VerificationFailure::ProofExpired)
        ));

        let mut invalid_window = valid.clone();
        invalid_window.proof_created = Some("2026-10-05T02:30:00Z".into());
        invalid_window.proof_expires = Some("2026-10-05T02:15:00Z".into());
        assert!(matches!(
            invalid_window.validate(),
            Err(VerificationFailure::InvalidValidityWindow)
        ));

        let mut wrong_domain = valid.clone();
        wrong_domain.proof_domain = Some("other.example".into());
        assert!(matches!(
            wrong_domain.validate(),
            Err(VerificationFailure::DomainMismatch { .. })
        ));

        let mut wrong_challenge = valid;
        wrong_challenge.proof_challenge = Some("challenge-2".into());
        assert!(matches!(
            wrong_challenge.validate(),
            Err(VerificationFailure::ChallengeMismatch { .. })
        ));
    }

    #[test]
    fn freshness_rejects_malformed_timestamps_and_blank_security_context() {
        let mut malformed = default_freshness();
        malformed.verification_time = "2026-10-05T02:00:00".into();
        assert!(matches!(
            malformed.validate(),
            Err(VerificationFailure::InvalidTimestamp { field: "verification time", .. })
        ));

        let mut blank_domain = default_freshness();
        blank_domain.expected_domain = Some("   ".into());
        assert!(matches!(
            blank_domain.validate(),
            Err(VerificationFailure::Structural(_))
        ));
    }

    #[test]
    fn replay_context_digest_changes_when_domain_or_challenge_changes() {
        let base = default_freshness();
        let mut domain = base.clone();
        domain.proof_domain = Some("other.example".into());
        let mut challenge = base.clone();
        challenge.proof_challenge = Some("challenge-2".into());
        assert_ne!(base.replay_context_digest(), domain.replay_context_digest());
        assert_ne!(base.replay_context_digest(), challenge.replay_context_digest());
    }


    #[test]
    fn request_binds_exact_claim_purpose_and_controller() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        assert_eq!(request.schema_version, VERIFICATION_REQUEST_SCHEMA_VERSION);
        assert_eq!(request.author.as_str(), "author:verification");
        assert_eq!(
            request.verification_method.as_str(),
            "https://example.test/controller#key-1"
        );
        assert_eq!(request.dependencies().unwrap().len(), 2);
        assert_eq!(
            request.expected_verification_relationship.as_str(),
            "assertionMethod"
        );
    }

    #[test]
    fn request_rejects_wrong_purpose_and_controller() {
        let claim = fixture_claim();
        let wrong_purpose = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("authentication").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap_err();
        assert!(matches!(
            wrong_purpose,
            VerificationFailure::ProofPurposeMismatch { .. }
        ));

        let wrong_controller = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/other").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap_err();
        assert!(matches!(
            wrong_controller,
            VerificationFailure::ControllerMismatch { .. }
        ));
    }

    #[test]
    fn evidence_binds_adapter_artifacts_to_exact_request() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let evidence = make_evidence(
            &request,
            &"11".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        assert!(evidence.validate_structure().is_ok());
        assert!(evidence.matches_request(&request));
        assert!(!evidence.evidence_digest().is_empty());
    }

    #[test]
    fn cryptographic_receipt_rejects_empty_proof_value() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();
        let resolution = resolved_method(&request);
        let mut receipt = test_cryptographic_receipt(&request, &resolution);
        receipt.proof_value_multibase = "z".into();

        assert!(matches!(
            receipt.validate_against(&request, &resolution),
            Err(VerificationFailure::Structural(message))
                if message.contains("non-empty proof bytes")
        ));
    }

    #[test]
    fn cryptographic_receipt_rejects_freshness_substitution() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let mut resolution = resolved_method(&request);
        resolution.validate_structure().unwrap();
        let mut receipt = test_cryptographic_receipt(&request, &resolution);
        receipt.freshness.verification_time = "2026-10-05T02:00:01Z".into();

        assert!(matches!(
            receipt.validate_against(&request, &resolution),
            Err(VerificationFailure::FreshnessMismatch)
        ));
        assert_ne!(
            receipt.receipt_digest(),
            test_cryptographic_receipt(&request, &resolution).receipt_digest()
        );
    }

    #[test]
    fn cryptographic_receipt_is_request_identity_bound() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();
        let evidence = make_evidence(
            &request,
            &"11".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        let mut tampered = evidence.clone();
        tampered
            .cryptographic_verification
            .claim_representation_digest = "77".repeat(32);
        assert!(matches!(
            tampered.validate_structure(),
            Err(VerificationFailure::Structural(message))
                if message.contains("claim representation identity")
        ));

        let mut statement_tampered = evidence.clone();
        statement_tampered.cryptographic_verification.statement_digest = "88".repeat(32);
        assert!(matches!(
            statement_tampered.validate_structure(),
            Err(VerificationFailure::Structural(message))
                if message.contains("statement identity")
        ));
    }

    #[test]
    fn cryptographic_receipt_is_part_of_typed_evidence_identity() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let evidence = make_evidence(
            &request,
            &"11".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        let mut tampered = evidence.clone();
        tampered
            .cryptographic_verification
            .verification_method_material_digest = "66".repeat(32);
        assert!(matches!(
            tampered.validate_structure(),
            Err(VerificationFailure::Structural(message))
                if message.contains("verification material")
        ));
        assert_ne!(evidence.evidence_digest(), tampered.evidence_digest());
    }

    #[test]
    fn evidence_rejects_malformed_external_artifacts() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let mut resolution = resolved_method(&request);
        resolution.controller_document_digest = "not-a-digest".into();
        let result = VerificationEvidence::from_adapter_attestation(
            &request,
            resolution.clone(),
            test_cryptographic_receipt(&request, &resolution),
        );
        assert!(matches!(
            result,
            Err(VerificationFailure::Structural(_))
        ));
    }

    #[test]
    fn evidence_rejects_a_controller_relationship_different_from_the_request() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        assert!(matches!(
            {
            let mut resolution = resolved_method(&request);
            resolution.verification_relationship =
                ClaimVerificationRelationship::new("authentication").unwrap();
            VerificationEvidence::from_adapter_attestation(
                &request,
                resolution.clone(),
                test_cryptographic_receipt(&request, &resolution),
                )
        },
            Err(VerificationFailure::VerificationRelationshipMismatch { .. })
        ));
    }

    #[test]
    fn evidence_rejects_resolved_verification_method_controller_substitution() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let mut resolution = resolved_method(&request);
        resolution.resolved_verification_method_controller =
            ClaimControllerIdentity::new("https://example.test/other").unwrap();
        let result = VerificationEvidence::from_adapter_attestation(
            &request,
            resolution.clone(),
            test_cryptographic_receipt(&request, &resolution),
            );

        assert!(matches!(
            result,
            Err(VerificationFailure::ControllerMismatch { .. })
        ));
    }

    #[test]
    fn evidence_rejects_controller_document_method_substitution() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let mut resolution = resolved_method(&request);
        resolution.verification_method =
            ClaimVerificationMethod::new("https://example.test/controller#key-2").unwrap();
        resolution.relationship_methods =
            vec![resolution.verification_method.clone()];
        let encoded_members = (
            "symthaea:verification-relationship-members:v1",
            request.expected_verification_relationship.as_str(),
            resolution
                .relationship_methods
                .iter()
                .map(ClaimVerificationMethod::as_str)
                .collect::<Vec<_>>(),
        );
        resolution.relationship_methods_digest =
            crate::sha256_hex(&serde_json::to_vec(&encoded_members).unwrap());
        let result = VerificationEvidence::from_adapter_attestation(
            &request,
            resolution.clone(),
            test_cryptographic_receipt(&request, &resolution),
            );

        assert!(matches!(
            result,
            Err(VerificationFailure::VerificationMethodMismatch { .. })
        ));
    }

    #[test]
    fn evidence_cannot_be_replayed_under_a_different_freshness_context() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let evidence = make_evidence(
            &request,
            &"11".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        let mut replayed = evidence.clone();
        replayed.freshness.proof_challenge = Some("challenge-2".into());

        assert!(!replayed.matches_request(&request));
        assert_ne!(evidence.evidence_digest(), replayed.evidence_digest());
    }

    #[test]
    fn evidence_digest_covers_resolved_method_and_controller_assertions() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();

        let base = make_evidence(
            &request,
            &"11".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        let mut method = base.clone();
        method.controller_document_verification_method =
            ClaimVerificationMethod::new("https://example.test/controller#key-2").unwrap();
        assert_ne!(base.evidence_digest(), method.evidence_digest());

        let mut controller = base;
        controller.resolved_verification_method_controller =
            ClaimControllerIdentity::new("https://example.test/other").unwrap();
        assert_ne!(controller.evidence_digest(), method.evidence_digest());
    }

    #[test]
    fn cryptographic_receipt_rejects_transformed_document_binding_substitution() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap()
        .with_expected_transformed_document_digest("22".repeat(32))
        .unwrap();

        let resolution = resolved_method(&request);
        let mut receipt = test_cryptographic_receipt(&request, &resolution);
        receipt.transformed_document_digest = "33".repeat(32);

        assert!(matches!(
            receipt.validate_against(&request, &resolution),
            Err(VerificationFailure::TransformedDocumentDigestMismatch {
                expected: Some(expected),
                actual: Some(actual),
            }) if expected == "22".repeat(32) && actual == "33".repeat(32)
        ));
    }


    #[test]
    fn evidence_rejects_top_level_transformed_document_binding_substitution() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap()
        .with_expected_transformed_document_digest("22".repeat(32))
        .unwrap();

        let resolution = resolved_method(&request);
        let receipt = test_cryptographic_receipt(&request, &resolution);
        let mut evidence =
            VerificationEvidence::from_adapter_attestation(&request, resolution, receipt).unwrap();

        evidence.expected_transformed_document_digest = Some("33".repeat(32));

        assert!(matches!(
            evidence.validate_structure(),
            Err(VerificationFailure::TransformedDocumentDigestMismatch {
                expected: Some(expected),
                actual: Some(actual),
            }) if expected == "22".repeat(32) && actual == "33".repeat(32)
        ));
    }

    #[test]
    fn evidence_digest_covers_expected_transformed_document_binding() {
        let claim = fixture_claim();
        let mut request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap()
        .with_expected_transformed_document_digest("22".repeat(32))
        .unwrap();

        let resolution = resolved_method(&request);
        let receipt = test_cryptographic_receipt(&request, &resolution);
        let base =
            VerificationEvidence::from_adapter_attestation(&request, resolution, receipt).unwrap();

        request.expected_transformed_document_digest = Some("33".repeat(32));
        let mut rebound = base.clone();
        rebound.expected_transformed_document_digest = request
            .expected_transformed_document_digest
            .clone();
        rebound.cryptographic_verification.expected_transformed_document_digest = request
            .expected_transformed_document_digest
            .clone();

        assert_ne!(base.evidence_digest(), rebound.evidence_digest());
    }

    #[test]
    fn evidence_changes_when_controller_snapshot_or_claim_changes() {
        let claim = fixture_claim();
        let request = VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();
        let a = make_evidence(
            &request,
            &"11".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        let b = make_evidence(
            &request,
            &"44".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        assert_ne!(a.evidence_digest(), b.evidence_digest());

        let mut changed_claim = claim.clone();
        changed_claim.source_event = Some("event:changed".into());
        let changed_request = VerificationRequest::from_claim(
            &changed_claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            ClaimVerificationRelationship::new("assertionMethod").unwrap(),
            default_freshness(),
        )
        .unwrap();
        let c = make_evidence(
            &changed_request,
            &"11".repeat(32),
            "ed25519",
            &"22".repeat(32),
            &"33".repeat(32),
        )
        .unwrap();

        assert_ne!(a.evidence_digest(), c.evidence_digest());
    }
}
