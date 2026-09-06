// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evidence currentness, supersession, and typed evidential relations for RCA v1.
//!
//! Authentic historical evidence is not automatically current evidence. Producers
//! do not declare `current = true`; currentness is evaluated against an explicit
//! time/generation context at the point of use.

use serde::{Deserialize, Deserializer, Serialize};

pub const COGNITIVE_CURRENTNESS_SCHEMA_VERSION: u16 = 1;
pub const EVIDENCE_RELATION_STRENGTH_SCALE: u32 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceCurrentnessModeV1 {
    /// Evidence whose proposition does not become stale merely because time or
    /// a runtime generation advances. Supersession can still invalidate it.
    Immutable,
    /// Evidence about mutable state. At least one time or generation boundary
    /// is mandatory.
    Dynamic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceCurrentnessV1 {
    pub schema_version: u16,
    pub evidence_id: String,
    pub mode: EvidenceCurrentnessModeV1,
    pub observed_at_unix_ms: Option<u64>,
    pub valid_until_unix_ms: Option<u64>,
    pub source_generation: Option<u64>,
    pub model_generation: Option<u64>,
    pub environment_generation: Option<u64>,
    /// Content-addressed evidence that explicitly supersedes this record.
    pub superseded_by: Option<String>,
}

impl EvidenceCurrentnessV1 {
    pub fn validate(self) -> Result<ValidatedEvidenceCurrentnessV1, CurrentnessError> {
        ValidatedEvidenceCurrentnessV1::try_from(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedEvidenceCurrentnessV1(EvidenceCurrentnessV1);

impl ValidatedEvidenceCurrentnessV1 {
    pub fn evidence_id(&self) -> &str {
        &self.0.evidence_id
    }

    pub fn as_raw(&self) -> &EvidenceCurrentnessV1 {
        &self.0
    }

    pub fn assess(&self, context: &CurrentnessContextV1) -> CurrentnessAssessmentV1 {
        if self.0.superseded_by.is_some() {
            return CurrentnessAssessmentV1::Superseded;
        }

        if let Some(valid_until) = self.0.valid_until_unix_ms {
            if context.now_unix_ms > valid_until {
                return CurrentnessAssessmentV1::Expired;
            }
        }

        let mut missing = Vec::new();
        if let Some(expected) = self.0.source_generation {
            match context.source_generation {
                Some(current) if current != expected => {
                    return CurrentnessAssessmentV1::GenerationMismatch {
                        dimension: CurrentnessDimensionV1::Source,
                        expected,
                        current,
                    };
                }
                Some(_) => {}
                None => missing.push(CurrentnessDimensionV1::Source),
            }
        }
        if let Some(expected) = self.0.model_generation {
            match context.model_generation {
                Some(current) if current != expected => {
                    return CurrentnessAssessmentV1::GenerationMismatch {
                        dimension: CurrentnessDimensionV1::Model,
                        expected,
                        current,
                    };
                }
                Some(_) => {}
                None => missing.push(CurrentnessDimensionV1::Model),
            }
        }
        if let Some(expected) = self.0.environment_generation {
            match context.environment_generation {
                Some(current) if current != expected => {
                    return CurrentnessAssessmentV1::GenerationMismatch {
                        dimension: CurrentnessDimensionV1::Environment,
                        expected,
                        current,
                    };
                }
                Some(_) => {}
                None => missing.push(CurrentnessDimensionV1::Environment),
            }
        }

        if missing.is_empty() {
            CurrentnessAssessmentV1::Current
        } else {
            CurrentnessAssessmentV1::Underdetermined { missing }
        }
    }
}

impl TryFrom<EvidenceCurrentnessV1> for ValidatedEvidenceCurrentnessV1 {
    type Error = CurrentnessError;

    fn try_from(value: EvidenceCurrentnessV1) -> Result<Self, Self::Error> {
        if value.schema_version != COGNITIVE_CURRENTNESS_SCHEMA_VERSION {
            return Err(CurrentnessError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }
        validate_digest(&value.evidence_id)?;

        if let Some(superseded_by) = value.superseded_by.as_deref() {
            validate_digest(superseded_by)?;
            if superseded_by == value.evidence_id {
                return Err(CurrentnessError::SelfSupersession);
            }
        }

        if let Some(valid_until) = value.valid_until_unix_ms {
            let observed_at = value
                .observed_at_unix_ms
                .ok_or(CurrentnessError::ExpiryWithoutObservationTime)?;
            if valid_until < observed_at {
                return Err(CurrentnessError::ExpiryBeforeObservation {
                    observed_at_unix_ms: observed_at,
                    valid_until_unix_ms: valid_until,
                });
            }
        }

        let has_dynamic_boundary = value.valid_until_unix_ms.is_some()
            || value.source_generation.is_some()
            || value.model_generation.is_some()
            || value.environment_generation.is_some();

        match value.mode {
            EvidenceCurrentnessModeV1::Immutable if has_dynamic_boundary => {
                return Err(CurrentnessError::ImmutableHasDynamicBoundary);
            }
            EvidenceCurrentnessModeV1::Dynamic if !has_dynamic_boundary => {
                return Err(CurrentnessError::DynamicWithoutBoundary);
            }
            _ => {}
        }

        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedEvidenceCurrentnessV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        EvidenceCurrentnessV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentnessContextV1 {
    pub now_unix_ms: u64,
    pub source_generation: Option<u64>,
    pub model_generation: Option<u64>,
    pub environment_generation: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurrentnessDimensionV1 {
    Source,
    Model,
    Environment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurrentnessAssessmentV1 {
    Current,
    Expired,
    Superseded,
    GenerationMismatch {
        dimension: CurrentnessDimensionV1,
        expected: u64,
        current: u64,
    },
    Underdetermined {
        missing: Vec<CurrentnessDimensionV1>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceRelationKindV1 {
    Supports,
    Contradicts,
    Weakens,
    Defeats,
    Supersedes,
    /// This label does not imply independent corroboration. Independence is
    /// established only through the lineage module.
    Corroborates,
    Irrelevant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "target_kind", rename_all = "snake_case")]
pub enum EvidenceRelationTargetV1 {
    Proposition { proposition_id: String },
    Evidence { evidence_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRelationV1 {
    pub schema_version: u16,
    pub relation_id: String,
    pub evidence_id: String,
    pub relation: EvidenceRelationKindV1,
    pub target: EvidenceRelationTargetV1,
    /// Fixed-point relation strength in [0, 1_000_000]. This is not a posterior
    /// probability and may not be summed as independent evidence without lineage.
    pub strength_ppm: u32,
}

impl EvidenceRelationV1 {
    pub fn validate(self) -> Result<ValidatedEvidenceRelationV1, EvidenceRelationError> {
        ValidatedEvidenceRelationV1::try_from(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedEvidenceRelationV1(EvidenceRelationV1);

impl ValidatedEvidenceRelationV1 {
    pub fn as_raw(&self) -> &EvidenceRelationV1 {
        &self.0
    }
}

impl TryFrom<EvidenceRelationV1> for ValidatedEvidenceRelationV1 {
    type Error = EvidenceRelationError;

    fn try_from(value: EvidenceRelationV1) -> Result<Self, Self::Error> {
        if value.schema_version != COGNITIVE_CURRENTNESS_SCHEMA_VERSION {
            return Err(EvidenceRelationError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }
        validate_digest(&value.relation_id).map_err(EvidenceRelationError::Currentness)?;
        validate_digest(&value.evidence_id).map_err(EvidenceRelationError::Currentness)?;
        if value.strength_ppm > EVIDENCE_RELATION_STRENGTH_SCALE {
            return Err(EvidenceRelationError::StrengthOutOfRange {
                found: value.strength_ppm,
            });
        }

        match (&value.relation, &value.target) {
            (
                EvidenceRelationKindV1::Supersedes,
                EvidenceRelationTargetV1::Evidence { evidence_id },
            ) => {
                validate_digest(evidence_id).map_err(EvidenceRelationError::Currentness)?;
                if evidence_id == &value.evidence_id {
                    return Err(EvidenceRelationError::SelfSupersession);
                }
            }
            (EvidenceRelationKindV1::Supersedes, _) => {
                return Err(EvidenceRelationError::SupersedesRequiresEvidenceTarget);
            }
            (_, EvidenceRelationTargetV1::Proposition { proposition_id }) => {
                validate_digest(proposition_id).map_err(EvidenceRelationError::Currentness)?;
            }
            (_, EvidenceRelationTargetV1::Evidence { .. }) => {
                return Err(EvidenceRelationError::NonSupersessionRequiresPropositionTarget);
            }
        }

        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedEvidenceRelationV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        EvidenceRelationV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurrentnessError {
    UnsupportedSchemaVersion { found: u16 },
    MalformedDigest,
    SelfSupersession,
    ExpiryWithoutObservationTime,
    ExpiryBeforeObservation {
        observed_at_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
    ImmutableHasDynamicBoundary,
    DynamicWithoutBoundary,
}

impl std::fmt::Display for CurrentnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported currentness schema version {found}; expected {COGNITIVE_CURRENTNESS_SCHEMA_VERSION}"
            ),
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::SelfSupersession => f.write_str("evidence cannot supersede itself"),
            Self::ExpiryWithoutObservationTime => {
                f.write_str("time-bounded evidence requires observed_at_unix_ms")
            }
            Self::ExpiryBeforeObservation {
                observed_at_unix_ms,
                valid_until_unix_ms,
            } => write!(
                f,
                "valid_until_unix_ms {valid_until_unix_ms} precedes observed_at_unix_ms {observed_at_unix_ms}"
            ),
            Self::ImmutableHasDynamicBoundary => {
                f.write_str("immutable evidence cannot declare time/generation currentness boundaries")
            }
            Self::DynamicWithoutBoundary => f.write_str(
                "dynamic evidence requires at least one expiry or generation boundary",
            ),
        }
    }
}

impl std::error::Error for CurrentnessError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceRelationError {
    UnsupportedSchemaVersion { found: u16 },
    Currentness(CurrentnessError),
    StrengthOutOfRange { found: u32 },
    SelfSupersession,
    SupersedesRequiresEvidenceTarget,
    NonSupersessionRequiresPropositionTarget,
}

impl std::fmt::Display for EvidenceRelationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported evidence relation schema version {found}; expected {COGNITIVE_CURRENTNESS_SCHEMA_VERSION}"
            ),
            Self::Currentness(error) => error.fmt(f),
            Self::StrengthOutOfRange { found } => write!(
                f,
                "evidence relation strength {found} exceeds scale {EVIDENCE_RELATION_STRENGTH_SCALE}"
            ),
            Self::SelfSupersession => f.write_str("evidence cannot supersede itself"),
            Self::SupersedesRequiresEvidenceTarget => {
                f.write_str("supersedes relations require an evidence target")
            }
            Self::NonSupersessionRequiresPropositionTarget => {
                f.write_str("non-supersession relations require a proposition target")
            }
        }
    }
}

impl std::error::Error for EvidenceRelationError {}

fn validate_digest(digest: &str) -> Result<(), CurrentnessError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(CurrentnessError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(CurrentnessError::MalformedDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn dynamic() -> EvidenceCurrentnessV1 {
        EvidenceCurrentnessV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            evidence_id: A.into(),
            mode: EvidenceCurrentnessModeV1::Dynamic,
            observed_at_unix_ms: Some(100),
            valid_until_unix_ms: Some(200),
            source_generation: Some(7),
            model_generation: None,
            environment_generation: Some(11),
            superseded_by: None,
        }
    }

    fn context() -> CurrentnessContextV1 {
        CurrentnessContextV1 {
            now_unix_ms: 150,
            source_generation: Some(7),
            model_generation: None,
            environment_generation: Some(11),
        }
    }

    #[test]
    fn dynamic_evidence_without_boundary_fails_closed() {
        let mut raw = dynamic();
        raw.valid_until_unix_ms = None;
        raw.source_generation = None;
        raw.environment_generation = None;
        assert_eq!(raw.validate(), Err(CurrentnessError::DynamicWithoutBoundary));
    }

    #[test]
    fn historical_validity_does_not_imply_currentness() {
        let valid = dynamic().validate().unwrap();
        let mut ctx = context();
        ctx.now_unix_ms = 201;
        assert_eq!(valid.assess(&ctx), CurrentnessAssessmentV1::Expired);
    }

    #[test]
    fn generation_drift_invalidates_dynamic_evidence() {
        let valid = dynamic().validate().unwrap();
        let mut ctx = context();
        ctx.environment_generation = Some(12);
        assert_eq!(
            valid.assess(&ctx),
            CurrentnessAssessmentV1::GenerationMismatch {
                dimension: CurrentnessDimensionV1::Environment,
                expected: 11,
                current: 12,
            }
        );
    }

    #[test]
    fn missing_generation_context_is_underdetermined_not_current() {
        let valid = dynamic().validate().unwrap();
        let mut ctx = context();
        ctx.source_generation = None;
        assert_eq!(
            valid.assess(&ctx),
            CurrentnessAssessmentV1::Underdetermined {
                missing: vec![CurrentnessDimensionV1::Source]
            }
        );
    }

    #[test]
    fn immutable_evidence_can_remain_current_until_superseded() {
        let valid = EvidenceCurrentnessV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            evidence_id: A.into(),
            mode: EvidenceCurrentnessModeV1::Immutable,
            observed_at_unix_ms: Some(1),
            valid_until_unix_ms: None,
            source_generation: None,
            model_generation: None,
            environment_generation: None,
            superseded_by: None,
        }
        .validate()
        .unwrap();
        assert_eq!(valid.assess(&context()), CurrentnessAssessmentV1::Current);
    }

    #[test]
    fn supersession_dominates_other_currentness_checks() {
        let mut raw = dynamic();
        raw.superseded_by = Some(B.into());
        let valid = raw.validate().unwrap();
        assert_eq!(valid.assess(&context()), CurrentnessAssessmentV1::Superseded);
    }

    #[test]
    fn typed_defeaters_are_preserved_not_collapsed_into_support() {
        let relation = EvidenceRelationV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            relation_id: C.into(),
            evidence_id: A.into(),
            relation: EvidenceRelationKindV1::Defeats,
            target: EvidenceRelationTargetV1::Proposition {
                proposition_id: B.into(),
            },
            strength_ppm: 900_000,
        }
        .validate()
        .unwrap();
        assert_eq!(relation.as_raw().relation, EvidenceRelationKindV1::Defeats);
    }

    #[test]
    fn corroboration_label_does_not_encode_independence() {
        let relation = EvidenceRelationV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            relation_id: C.into(),
            evidence_id: A.into(),
            relation: EvidenceRelationKindV1::Corroborates,
            target: EvidenceRelationTargetV1::Proposition {
                proposition_id: B.into(),
            },
            strength_ppm: 500_000,
        };
        assert!(relation.validate().is_ok());
        // Independence remains a separate lineage assessment; there is no
        // independence field in EvidenceRelationV1.
    }

    #[test]
    fn supersession_requires_evidence_target_and_rejects_self_target() {
        let wrong_target = EvidenceRelationV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            relation_id: C.into(),
            evidence_id: A.into(),
            relation: EvidenceRelationKindV1::Supersedes,
            target: EvidenceRelationTargetV1::Proposition {
                proposition_id: B.into(),
            },
            strength_ppm: 1_000_000,
        };
        assert_eq!(
            wrong_target.validate(),
            Err(EvidenceRelationError::SupersedesRequiresEvidenceTarget)
        );

        let self_target = EvidenceRelationV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            relation_id: C.into(),
            evidence_id: A.into(),
            relation: EvidenceRelationKindV1::Supersedes,
            target: EvidenceRelationTargetV1::Evidence {
                evidence_id: A.into(),
            },
            strength_ppm: 1_000_000,
        };
        assert_eq!(self_target.validate(), Err(EvidenceRelationError::SelfSupersession));
    }

    #[test]
    fn validated_currentness_revalidates_after_persistence() {
        let valid = dynamic().validate().unwrap();
        let json = serde_json::to_string(&valid).unwrap();
        let decoded: ValidatedEvidenceCurrentnessV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, valid);
    }
}