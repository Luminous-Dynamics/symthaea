// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Typed content identities for replayable Futures provenance.
//!
//! The raw v1/v2 ledger schemas intentionally remain wire-compatible and may
//! contain human-readable hash labels. New evidence boundaries should prefer
//! these types when a field semantically claims content identity.
//!
//! [`ContentDigest`] validates a canonical algorithm-tagged digest string. It
//! proves representation validity only; it does **not** prove that some external
//! bytes were actually hashed to that value. An ingestion/verifier layer must
//! still compute the digest over the admitted bytes before granting evidence
//! authority.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceError {
    EmptyText { field: &'static str },
    UnsupportedDigestAlgorithm { algorithm: String },
    NonCanonicalDigest,
    EmptyList { field: &'static str },
    DuplicateReference { namespace: String, id: String },
}

impl fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyText { field } => write!(f, "{field} must not be empty"),
            Self::UnsupportedDigestAlgorithm { algorithm } => {
                write!(f, "unsupported content-digest algorithm: {algorithm}")
            }
            Self::NonCanonicalDigest => write!(
                f,
                "content digest must be canonical sha256:<64 lowercase hex characters>"
            ),
            Self::EmptyList { field } => write!(f, "{field} must not be empty"),
            Self::DuplicateReference { namespace, id } => {
                write!(f, "duplicate content-addressed reference {namespace}/{id}")
            }
        }
    }
}

impl std::error::Error for ProvenanceError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DigestAlgorithm {
    Sha256,
}

impl DigestAlgorithm {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
        }
    }
}

/// Canonical content digest representation.
///
/// V1 supports SHA-256 only. The algorithm remains explicit so adding another
/// qualified digest family later does not require sentinel prefixes or an
/// ambiguous untagged byte string.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ContentDigest {
    algorithm: DigestAlgorithm,
    hex: String,
}

impl ContentDigest {
    pub fn parse(value: impl AsRef<str>) -> Result<Self, ProvenanceError> {
        let value = value.as_ref();
        let Some((algorithm, hex)) = value.split_once(':') else {
            return Err(ProvenanceError::NonCanonicalDigest);
        };

        if algorithm != "sha256" {
            if algorithm.is_empty() || algorithm.trim().eq_ignore_ascii_case("sha256") {
                return Err(ProvenanceError::NonCanonicalDigest);
            }
            return Err(ProvenanceError::UnsupportedDigestAlgorithm {
                algorithm: algorithm.to_string(),
            });
        }

        if hex.len() != 64
            || !hex
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(ProvenanceError::NonCanonicalDigest);
        }

        Ok(Self {
            algorithm: DigestAlgorithm::Sha256,
            hex: hex.to_string(),
        })
    }

    pub const fn algorithm(&self) -> DigestAlgorithm {
        self.algorithm
    }

    pub fn hex(&self) -> &str {
        &self.hex
    }

    pub fn canonical(&self) -> String {
        format!("{}:{}", self.algorithm.name(), self.hex)
    }
}

impl fmt::Display for ContentDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.algorithm.name(), self.hex)
    }
}

impl TryFrom<String> for ContentDigest {
    type Error = ProvenanceError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl From<ContentDigest> for String {
    fn from(value: ContentDigest) -> Self {
        value.canonical()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ProvenanceLabel(String);

impl ProvenanceLabel {
    pub fn new(field: &'static str, value: impl Into<String>) -> Result<Self, ProvenanceError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ProvenanceError::EmptyText { field });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ProvenanceLabel {
    type Error = ProvenanceError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new("provenance label", value)
    }
}

impl From<ProvenanceLabel> for String {
    fn from(value: ProvenanceLabel) -> Self {
        value.0
    }
}

/// One named artifact or external snapshot with a canonical declared digest.
///
/// This is content-*addressed metadata*, not evidence that the referenced bytes
/// were independently rehashed. A verifier must still establish that equality.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContentAddressedRef {
    namespace: ProvenanceLabel,
    id: ProvenanceLabel,
    digest: ContentDigest,
}

impl ContentAddressedRef {
    pub fn new(
        namespace: impl Into<String>,
        id: impl Into<String>,
        digest: ContentDigest,
    ) -> Result<Self, ProvenanceError> {
        Ok(Self {
            namespace: ProvenanceLabel::new("content reference namespace", namespace)?,
            id: ProvenanceLabel::new("content reference id", id)?,
            digest,
        })
    }

    pub fn namespace(&self) -> &str {
        self.namespace.as_str()
    }

    pub fn id(&self) -> &str {
        self.id.as_str()
    }

    pub fn digest(&self) -> &ContentDigest {
        &self.digest
    }
}

#[derive(Deserialize)]
struct TypedForecastProvenanceRepr {
    observations: Vec<ContentAddressedRef>,
    input_snapshot: ContentAddressedRef,
    transformations: Vec<ContentAddressedRef>,
    observation_policy: ContentAddressedRef,
    models: Vec<ContentAddressedRef>,
    generators: Vec<ContentAddressedRef>,
}

/// Typed provenance required by future qualified forecast/economic bridges.
///
/// The structure distinguishes observation inputs, transformation artifacts,
/// observation policy, model artifacts, and trajectory generators. It does not
/// collapse them into a bag of strings, and it requires at least one model and
/// generator artifact for a forecast-producing lineage. Deserialization is
/// routed back through [`Self::new`] so replay cannot bypass these invariants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TypedForecastProvenance {
    observations: Vec<ContentAddressedRef>,
    input_snapshot: ContentAddressedRef,
    transformations: Vec<ContentAddressedRef>,
    observation_policy: ContentAddressedRef,
    models: Vec<ContentAddressedRef>,
    generators: Vec<ContentAddressedRef>,
}

impl TypedForecastProvenance {
    pub fn new(
        observations: Vec<ContentAddressedRef>,
        input_snapshot: ContentAddressedRef,
        transformations: Vec<ContentAddressedRef>,
        observation_policy: ContentAddressedRef,
        models: Vec<ContentAddressedRef>,
        generators: Vec<ContentAddressedRef>,
    ) -> Result<Self, ProvenanceError> {
        if observations.is_empty() {
            return Err(ProvenanceError::EmptyList {
                field: "observation references",
            });
        }
        if models.is_empty() {
            return Err(ProvenanceError::EmptyList {
                field: "model references",
            });
        }
        if generators.is_empty() {
            return Err(ProvenanceError::EmptyList {
                field: "generator references",
            });
        }

        Self::ensure_unique(&observations)?;
        Self::ensure_unique(&transformations)?;
        Self::ensure_unique(&models)?;
        Self::ensure_unique(&generators)?;

        Ok(Self {
            observations,
            input_snapshot,
            transformations,
            observation_policy,
            models,
            generators,
        })
    }

    fn ensure_unique(values: &[ContentAddressedRef]) -> Result<(), ProvenanceError> {
        let mut seen = BTreeSet::new();
        for value in values {
            let key = (value.namespace().to_string(), value.id().to_string());
            if !seen.insert(key) {
                return Err(ProvenanceError::DuplicateReference {
                    namespace: value.namespace().to_string(),
                    id: value.id().to_string(),
                });
            }
        }
        Ok(())
    }

    pub fn observations(&self) -> &[ContentAddressedRef] {
        &self.observations
    }

    pub fn input_snapshot(&self) -> &ContentAddressedRef {
        &self.input_snapshot
    }

    pub fn transformations(&self) -> &[ContentAddressedRef] {
        &self.transformations
    }

    pub fn observation_policy(&self) -> &ContentAddressedRef {
        &self.observation_policy
    }

    pub fn models(&self) -> &[ContentAddressedRef] {
        &self.models
    }

    pub fn generators(&self) -> &[ContentAddressedRef] {
        &self.generators
    }
}

impl TryFrom<TypedForecastProvenanceRepr> for TypedForecastProvenance {
    type Error = ProvenanceError;

    fn try_from(repr: TypedForecastProvenanceRepr) -> Result<Self, Self::Error> {
        Self::new(
            repr.observations,
            repr.input_snapshot,
            repr.transformations,
            repr.observation_policy,
            repr.models,
            repr.generators,
        )
    }
}

impl<'de> Deserialize<'de> for TypedForecastProvenance {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = TypedForecastProvenanceRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO_SHA256: &str =
        "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    const ONE_SHA256: &str =
        "sha256:1111111111111111111111111111111111111111111111111111111111111111";

    fn digest(value: &str) -> ContentDigest {
        ContentDigest::parse(value).unwrap()
    }

    fn addressed(namespace: &str, id: &str, value: &str) -> ContentAddressedRef {
        ContentAddressedRef::new(namespace, id, digest(value)).unwrap()
    }

    #[test]
    fn canonical_sha256_round_trips() {
        let parsed = digest(ZERO_SHA256);
        assert_eq!(parsed.algorithm(), DigestAlgorithm::Sha256);
        assert_eq!(parsed.hex().len(), 64);
        assert_eq!(parsed.to_string(), ZERO_SHA256);
        assert_eq!(parsed.canonical(), ZERO_SHA256);
    }

    #[test]
    fn malformed_or_noncanonical_digests_fail_closed() {
        for value in [
            "",
            "sha256:",
            "sha256:abc123",
            "SHA256:0000000000000000000000000000000000000000000000000000000000000000",
            "sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "sha256:gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg",
            " sha256:0000000000000000000000000000000000000000000000000000000000000000",
        ] {
            assert_eq!(
                ContentDigest::parse(value),
                Err(ProvenanceError::NonCanonicalDigest),
                "unexpectedly accepted {value:?}"
            );
        }
    }

    #[test]
    fn unknown_algorithm_is_not_reinterpreted_as_sha256() {
        let value = "sha512:0000000000000000000000000000000000000000000000000000000000000000";
        assert_eq!(
            ContentDigest::parse(value),
            Err(ProvenanceError::UnsupportedDigestAlgorithm {
                algorithm: "sha512".into()
            })
        );
    }

    #[test]
    fn addressed_reference_requires_real_identity_labels() {
        assert!(matches!(
            ContentAddressedRef::new("", "dataset-v1", digest(ZERO_SHA256)),
            Err(ProvenanceError::EmptyText { .. })
        ));
        assert!(matches!(
            ContentAddressedRef::new("dataset", "   ", digest(ZERO_SHA256)),
            Err(ProvenanceError::EmptyText { .. })
        ));
    }

    #[test]
    fn typed_forecast_provenance_keeps_artifact_roles_separate() {
        let provenance = TypedForecastProvenance::new(
            vec![addressed("dataset", "public-vintage", ZERO_SHA256)],
            addressed("input-snapshot", "model-inputs", ONE_SHA256),
            vec![addressed("transform", "seasonal-adjustment-v1", ZERO_SHA256)],
            addressed("observation-policy", "econ-observe-v1", ONE_SHA256),
            vec![addressed("model", "agent-model-v1", ZERO_SHA256)],
            vec![addressed("generator", "econ-generator-v1", ONE_SHA256)],
        )
        .unwrap();

        assert_eq!(provenance.observations().len(), 1);
        assert_eq!(provenance.transformations().len(), 1);
        assert_eq!(provenance.models()[0].namespace(), "model");
        assert_eq!(provenance.generators()[0].namespace(), "generator");
    }

    #[test]
    fn typed_forecast_provenance_rejects_missing_models_or_generators() {
        let observation = addressed("dataset", "public-vintage", ZERO_SHA256);
        let input = addressed("input-snapshot", "model-inputs", ONE_SHA256);
        let policy = addressed("observation-policy", "econ-observe-v1", ONE_SHA256);
        let model = addressed("model", "agent-model-v1", ZERO_SHA256);
        let generator = addressed("generator", "econ-generator-v1", ONE_SHA256);

        assert!(matches!(
            TypedForecastProvenance::new(
                vec![observation.clone()],
                input.clone(),
                vec![],
                policy.clone(),
                vec![],
                vec![generator.clone()],
            ),
            Err(ProvenanceError::EmptyList {
                field: "model references"
            })
        ));
        assert!(matches!(
            TypedForecastProvenance::new(
                vec![observation],
                input,
                vec![],
                policy,
                vec![model],
                vec![],
            ),
            Err(ProvenanceError::EmptyList {
                field: "generator references"
            })
        ));
    }

    #[test]
    fn duplicate_content_identity_is_rejected_within_each_role() {
        let duplicate = addressed("model", "same-model", ZERO_SHA256);
        assert!(matches!(
            TypedForecastProvenance::new(
                vec![addressed("dataset", "public-vintage", ZERO_SHA256)],
                addressed("input-snapshot", "model-inputs", ONE_SHA256),
                vec![],
                addressed("observation-policy", "econ-observe-v1", ONE_SHA256),
                vec![duplicate.clone(), duplicate],
                vec![addressed("generator", "econ-generator-v1", ONE_SHA256)],
            ),
            Err(ProvenanceError::DuplicateReference { .. })
        ));
    }
}
