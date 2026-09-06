// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Canonical evidence-object identity for RCA epistemic governance.
//!
//! `CognitiveEvidenceRefV1::claim_digest` identifies the exact claim/observation
//! payload represented by a record. It is **not** the identity of the complete
//! evidence object: two evidence records may carry the same claim while having
//! different source, model, observation, verifier, version, or freshness
//! provenance.
//!
//! Lineage/currentness/relation policy is keyed by evidence-object identity, so
//! this module derives that join key from the complete validated evidence record
//! instead of accepting a producer-authored `evidence_id`.

use serde::{Deserialize, Deserializer, Serialize};
use symthaea_types::cognitive_evidence::{
    CognitiveEvidenceAuthorityV1, ValidatedCognitiveEvidenceRefV1,
};

pub const COGNITIVE_EVIDENCE_IDENTITY_SCHEMA_VERSION: u16 = 1;
pub const COGNITIVE_EVIDENCE_IDENTITY_PROFILE_V1: &str = "rca-cognitive-evidence-identity-v1";

/// Normative evidence-object identity semantics.
///
/// Identity is not delegated to `serde_json`, Rust `Hash`, debug formatting, or
/// enum discriminant layout. Every field is explicitly labelled and encoded.
/// Optional fields have an explicit presence byte so `None` cannot collapse into
/// an empty value. Any semantic change requires a new profile.
pub const COGNITIVE_EVIDENCE_IDENTITY_CONTRACT_V1: &str = concat!(
    "rca-cognitive-evidence-identity-v1\n",
    "commitment=blake3_domain_separated_labelled_fields_v1\n",
    "field=label_len_u64le|label_utf8|value_len_u64le|value_bytes\n",
    "option_text=label_len_u64le|label_utf8|presence_u8|[value_len_u64le|value_utf8]\n",
    "option_u64=label_len_u64le|label_utf8|presence_u8|[u64le]\n",
    "authority=stable_ascii_variant_name_not_rust_discriminant\n",
    "field_order=evidence_schema_version,authority,source,source_version,claim_digest,source_digest,model_id,model_version,observation_id,formal_verifier,freshness_epoch\n",
    "claim_digest_is_payload_identity_not_evidence_object_identity\n",
    "evidence_id_is_derived_never_producer_asserted\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-evidence-identity-contract:v1\0";
const EVIDENCE_ID_DOMAIN: &[u8] = b"symthaea:rca-evidence-object:v1\0";

/// Persistable binding between one validated evidence record and its derived
/// governance join key.
///
/// Deserialization recomputes both the profile digest and evidence-object id.
/// A persisted object cannot change either value without invalidating the
/// binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoundCognitiveEvidenceV1 {
    schema_version: u16,
    identity_profile: String,
    identity_profile_digest: String,
    evidence_id: String,
    evidence: ValidatedCognitiveEvidenceRefV1,
}

impl BoundCognitiveEvidenceV1 {
    pub fn new(evidence: ValidatedCognitiveEvidenceRefV1) -> Self {
        let identity_profile_digest = evidence_identity_profile_digest_v1();
        let evidence_id = cognitive_evidence_id_v1(&evidence);
        Self {
            schema_version: COGNITIVE_EVIDENCE_IDENTITY_SCHEMA_VERSION,
            identity_profile: COGNITIVE_EVIDENCE_IDENTITY_PROFILE_V1.to_string(),
            identity_profile_digest,
            evidence_id,
            evidence,
        }
    }

    pub fn evidence_id(&self) -> &str {
        &self.evidence_id
    }

    pub fn identity_profile(&self) -> &str {
        &self.identity_profile
    }

    pub fn identity_profile_digest(&self) -> &str {
        &self.identity_profile_digest
    }

    pub fn evidence(&self) -> &ValidatedCognitiveEvidenceRefV1 {
        &self.evidence
    }

    /// Fail closed when a governance artifact references a different evidence
    /// object, even if both records happen to carry the same claim digest.
    pub fn require_governance_id(&self, governance_evidence_id: &str) -> Result<(), EvidenceIdentityError> {
        validate_digest(governance_evidence_id)?;
        if governance_evidence_id == self.evidence_id {
            Ok(())
        } else {
            Err(EvidenceIdentityError::GovernanceIdentityMismatch)
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundCognitiveEvidenceWireV1 {
    schema_version: u16,
    identity_profile: String,
    identity_profile_digest: String,
    evidence_id: String,
    evidence: ValidatedCognitiveEvidenceRefV1,
}

impl<'de> Deserialize<'de> for BoundCognitiveEvidenceV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = BoundCognitiveEvidenceWireV1::deserialize(deserializer)?;
        if wire.schema_version != COGNITIVE_EVIDENCE_IDENTITY_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(
                EvidenceIdentityError::UnsupportedSchemaVersion {
                    found: wire.schema_version,
                },
            ));
        }
        if wire.identity_profile != COGNITIVE_EVIDENCE_IDENTITY_PROFILE_V1 {
            return Err(serde::de::Error::custom(
                EvidenceIdentityError::UnexpectedIdentityProfile,
            ));
        }
        validate_digest(&wire.identity_profile_digest).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.evidence_id).map_err(serde::de::Error::custom)?;

        let expected_profile_digest = evidence_identity_profile_digest_v1();
        if wire.identity_profile_digest != expected_profile_digest {
            return Err(serde::de::Error::custom(
                EvidenceIdentityError::ProfileDigestMismatch,
            ));
        }

        let expected_evidence_id = cognitive_evidence_id_v1(&wire.evidence);
        if wire.evidence_id != expected_evidence_id {
            return Err(serde::de::Error::custom(
                EvidenceIdentityError::EvidenceIdentityMismatch,
            ));
        }

        Ok(Self {
            schema_version: wire.schema_version,
            identity_profile: wire.identity_profile,
            identity_profile_digest: wire.identity_profile_digest,
            evidence_id: wire.evidence_id,
            evidence: wire.evidence,
        })
    }
}

pub fn evidence_identity_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        COGNITIVE_EVIDENCE_IDENTITY_CONTRACT_V1.as_bytes(),
    )
}

/// Derive the canonical governance join key for a complete validated evidence
/// record.
///
/// Equal complete records produce equal ids. Equal claims with different
/// provenance produce different ids.
pub fn cognitive_evidence_id_v1(evidence: &ValidatedCognitiveEvidenceRefV1) -> String {
    let raw = evidence.as_raw();
    let mut hasher = blake3::Hasher::new();
    hasher.update(EVIDENCE_ID_DOMAIN);

    hash_text_field(
        &mut hasher,
        b"identity_profile_digest",
        &evidence_identity_profile_digest_v1(),
    );
    hash_field(
        &mut hasher,
        b"evidence_schema_version",
        &raw.schema_version.to_le_bytes(),
    );
    hash_text_field(
        &mut hasher,
        b"authority",
        authority_tag(raw.authority),
    );
    hash_text_field(&mut hasher, b"source", &raw.source);
    hash_option_text(&mut hasher, b"source_version", raw.source_version.as_deref());
    hash_text_field(&mut hasher, b"claim_digest", &raw.claim_digest);
    hash_option_text(&mut hasher, b"source_digest", raw.source_digest.as_deref());
    hash_option_text(&mut hasher, b"model_id", raw.model_id.as_deref());
    hash_option_text(&mut hasher, b"model_version", raw.model_version.as_deref());
    hash_option_text(&mut hasher, b"observation_id", raw.observation_id.as_deref());
    hash_option_text(
        &mut hasher,
        b"formal_verifier",
        raw.formal_verifier.as_deref(),
    );
    hash_option_u64(&mut hasher, b"freshness_epoch", raw.freshness_epoch);

    format!("blake3:{}", hasher.finalize().to_hex())
}

fn authority_tag(authority: CognitiveEvidenceAuthorityV1) -> &'static str {
    match authority {
        CognitiveEvidenceAuthorityV1::SyntheticFixture => "synthetic_fixture",
        CognitiveEvidenceAuthorityV1::InternalInference => "internal_inference",
        CognitiveEvidenceAuthorityV1::InternalSimulation => "internal_simulation",
        CognitiveEvidenceAuthorityV1::RetrievedExternalClaim => "retrieved_external_claim",
        CognitiveEvidenceAuthorityV1::EmpiricalObservation => "empirical_observation",
        CognitiveEvidenceAuthorityV1::FormalDerivation => "formal_derivation",
    }
}

fn hash_field(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn hash_text_field(hasher: &mut blake3::Hasher, label: &[u8], value: &str) {
    hash_field(hasher, label, value.as_bytes());
}

fn hash_option_text(hasher: &mut blake3::Hasher, label: &[u8], value: Option<&str>) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    match value {
        None => {
            hasher.update(&[0]);
        }
        Some(text) => {
            hasher.update(&[1]);
            hasher.update(&(text.len() as u64).to_le_bytes());
            hasher.update(text.as_bytes());
        }
    }
}

fn hash_option_u64(hasher: &mut blake3::Hasher, label: &[u8], value: Option<u64>) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    match value {
        None => {
            hasher.update(&[0]);
        }
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_le_bytes());
        }
    }
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn validate_digest(digest: &str) -> Result<(), EvidenceIdentityError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(EvidenceIdentityError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(EvidenceIdentityError::MalformedDigest);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceIdentityError {
    UnsupportedSchemaVersion { found: u16 },
    UnexpectedIdentityProfile,
    MalformedDigest,
    ProfileDigestMismatch,
    EvidenceIdentityMismatch,
    GovernanceIdentityMismatch,
}

impl std::fmt::Display for EvidenceIdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported evidence-identity schema version {found}; expected {COGNITIVE_EVIDENCE_IDENTITY_SCHEMA_VERSION}"
            ),
            Self::UnexpectedIdentityProfile => write!(
                f,
                "unexpected evidence-identity profile; expected {COGNITIVE_EVIDENCE_IDENTITY_PROFILE_V1:?}"
            ),
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::ProfileDigestMismatch => {
                f.write_str("evidence-identity profile digest does not match the v1 contract")
            }
            Self::EvidenceIdentityMismatch => {
                f.write_str("evidence-object id does not match the complete validated evidence record")
            }
            Self::GovernanceIdentityMismatch => {
                f.write_str("governance artifact references a different evidence object")
            }
        }
    }
}

impl std::error::Error for EvidenceIdentityError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_types::cognitive_evidence::{
        CognitiveEvidenceRefV1, COGNITIVE_EVIDENCE_SCHEMA_VERSION,
    };

    const CLAIM: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SOURCE_A: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SOURCE_B: &str =
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn external_claim(source: &str, version: Option<&str>) -> ValidatedCognitiveEvidenceRefV1 {
        CognitiveEvidenceRefV1 {
            schema_version: COGNITIVE_EVIDENCE_SCHEMA_VERSION,
            authority: CognitiveEvidenceAuthorityV1::RetrievedExternalClaim,
            source: source.to_string(),
            source_version: version.map(str::to_string),
            claim_digest: CLAIM.to_string(),
            source_digest: Some(SOURCE_A.to_string()),
            model_id: None,
            model_version: None,
            observation_id: None,
            formal_verifier: None,
            freshness_epoch: Some(7),
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn evidence_identity_is_deterministic() {
        let evidence = external_claim("source-a", Some("v1"));
        assert_eq!(
            cognitive_evidence_id_v1(&evidence),
            cognitive_evidence_id_v1(&evidence)
        );
    }

    #[test]
    fn claim_digest_is_not_evidence_object_identity() {
        let evidence = external_claim("source-a", Some("v1"));
        assert_ne!(cognitive_evidence_id_v1(&evidence), evidence.claim_digest());
    }

    #[test]
    fn same_claim_from_different_provenance_stays_distinct() {
        let a = external_claim("source-a", Some("v1"));
        let b = external_claim("source-b", Some("v1"));
        assert_eq!(a.claim_digest(), b.claim_digest());
        assert_ne!(cognitive_evidence_id_v1(&a), cognitive_evidence_id_v1(&b));
    }

    #[test]
    fn source_version_is_identity_bearing() {
        let a = external_claim("source-a", Some("v1"));
        let b = external_claim("source-a", Some("v2"));
        assert_ne!(cognitive_evidence_id_v1(&a), cognitive_evidence_id_v1(&b));
    }

    #[test]
    fn freshness_epoch_is_identity_bearing_metadata() {
        let a = external_claim("source-a", Some("v1"));
        let mut raw = a.as_raw().clone();
        raw.freshness_epoch = Some(8);
        let b = raw.validate().unwrap();
        assert_ne!(cognitive_evidence_id_v1(&a), cognitive_evidence_id_v1(&b));
    }

    #[test]
    fn model_revision_changes_internal_simulation_identity() {
        let make = |version: &str| {
            CognitiveEvidenceRefV1 {
                schema_version: COGNITIVE_EVIDENCE_SCHEMA_VERSION,
                authority: CognitiveEvidenceAuthorityV1::InternalSimulation,
                source: "simulator".into(),
                source_version: None,
                claim_digest: CLAIM.into(),
                source_digest: None,
                model_id: Some("world-model".into()),
                model_version: Some(version.into()),
                observation_id: None,
                formal_verifier: None,
                freshness_epoch: None,
            }
            .validate()
            .unwrap()
        };
        assert_ne!(
            cognitive_evidence_id_v1(&make("v1")),
            cognitive_evidence_id_v1(&make("v2"))
        );
    }

    #[test]
    fn persisted_binding_revalidates_identity() {
        let binding = BoundCognitiveEvidenceV1::new(external_claim("source-a", Some("v1")));
        let encoded = serde_json::to_string(&binding).unwrap();
        let decoded: BoundCognitiveEvidenceV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, binding);
    }

    #[test]
    fn tampered_persisted_identity_fails_closed() {
        let binding = BoundCognitiveEvidenceV1::new(external_claim("source-a", Some("v1")));
        let mut value = serde_json::to_value(&binding).unwrap();
        value["evidence_id"] = serde_json::Value::String(SOURCE_B.to_string());
        assert!(serde_json::from_value::<BoundCognitiveEvidenceV1>(value).is_err());
    }

    #[test]
    fn tampered_profile_digest_fails_closed() {
        let binding = BoundCognitiveEvidenceV1::new(external_claim("source-a", Some("v1")));
        let mut value = serde_json::to_value(&binding).unwrap();
        value["identity_profile_digest"] = serde_json::Value::String(SOURCE_B.to_string());
        assert!(serde_json::from_value::<BoundCognitiveEvidenceV1>(value).is_err());
    }

    #[test]
    fn governance_join_requires_exact_evidence_object_id() {
        let binding = BoundCognitiveEvidenceV1::new(external_claim("source-a", Some("v1")));
        assert!(binding.require_governance_id(binding.evidence_id()).is_ok());
        assert_eq!(
            binding.require_governance_id(SOURCE_B),
            Err(EvidenceIdentityError::GovernanceIdentityMismatch)
        );
    }

    #[test]
    fn identity_profile_has_strict_commitment() {
        let digest = evidence_identity_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
        assert_eq!(digest, evidence_identity_profile_digest_v1());
    }
}
