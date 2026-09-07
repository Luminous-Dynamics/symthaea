// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical references to external scientific evidence.
//!
//! This module validates local *reference shape* only. A content digest is not
//! recomputed here, a claimed date is not independently timestamped here, and a
//! role such as `ExperimentalObservation` is not proof that an experiment was
//! valid. Stronger authority belongs behind a separate verifier boundary.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeSet;
use std::fmt;

const MAX_ID_LEN: usize = 256;
const MAX_LOCATOR_LEN: usize = 2048;
const MAX_SUBJECT_LEN: usize = 1024;
const MAX_ISSUER_LEN: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalEvidenceError {
    EmptyField { field: &'static str },
    FieldTooLong { field: &'static str, max_len: usize },
    ControlCharacter { field: &'static str },
    InvalidSha256,
    InvalidClaimedUtcDate(u32),
    EmptyBundle,
    DuplicateEvidenceId(String),
    UnknownEvidenceId(String),
    RoleMismatch {
        evidence_id: String,
        expected: EvidenceRole,
        actual: EvidenceRole,
    },
    SameEvidenceReference,
    SameContentIdentity,
    MissingClaimedUtcDate(String),
    NonIncreasingDeclaredChronology {
        earlier_yyyymmdd: u32,
        later_yyyymmdd: u32,
    },
}

impl fmt::Display for ExternalEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField { field } => write!(f, "external evidence field `{field}` is empty"),
            Self::FieldTooLong { field, max_len } => {
                write!(f, "external evidence field `{field}` exceeds {max_len} bytes")
            }
            Self::ControlCharacter { field } => {
                write!(f, "external evidence field `{field}` contains a control character")
            }
            Self::InvalidSha256 => write!(
                f,
                "external evidence SHA-256 must be exactly 64 hexadecimal characters"
            ),
            Self::InvalidClaimedUtcDate(value) => {
                write!(f, "claimed UTC date `{value}` is not a valid YYYYMMDD date")
            }
            Self::EmptyBundle => write!(f, "external evidence bundle is empty"),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate external evidence id `{id}`"),
            Self::UnknownEvidenceId(id) => write!(f, "unknown external evidence id `{id}`"),
            Self::RoleMismatch {
                evidence_id,
                expected,
                actual,
            } => write!(
                f,
                "external evidence `{evidence_id}` has role {actual:?}, expected {expected:?}"
            ),
            Self::SameEvidenceReference => {
                write!(f, "temporal relation requires distinct evidence references")
            }
            Self::SameContentIdentity => {
                write!(f, "distinct references resolve to the same content identity")
            }
            Self::MissingClaimedUtcDate(id) => {
                write!(f, "external evidence `{id}` has no claimed UTC date")
            }
            Self::NonIncreasingDeclaredChronology {
                earlier_yyyymmdd,
                later_yyyymmdd,
            } => write!(
                f,
                "declared chronology is not increasing: {earlier_yyyymmdd} !< {later_yyyymmdd}"
            ),
        }
    }
}

impl std::error::Error for ExternalEvidenceError {}

fn validate_text(
    value: String,
    field: &'static str,
    max_len: usize,
) -> Result<String, ExternalEvidenceError> {
    if value.trim().is_empty() {
        return Err(ExternalEvidenceError::EmptyField { field });
    }
    if value.len() > max_len {
        return Err(ExternalEvidenceError::FieldTooLong { field, max_len });
    }
    if value.chars().any(char::is_control) {
        return Err(ExternalEvidenceError::ControlCharacter { field });
    }
    Ok(value)
}

macro_rules! validated_string_type {
    ($name:ident, $field:literal, $max_len:expr) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, ExternalEvidenceError> {
                Ok(Self(validate_text(value.into(), $field, $max_len)?))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let raw = String::deserialize(deserializer)?;
                Self::new(raw).map_err(serde::de::Error::custom)
            }
        }
    };
}

validated_string_type!(ExternalEvidenceId, "evidence_id", MAX_ID_LEN);
validated_string_type!(EvidenceLocator, "locator", MAX_LOCATOR_LEN);
validated_string_type!(EvidenceSubject, "subject", MAX_SUBJECT_LEN);
validated_string_type!(EvidenceIssuer, "issuer", MAX_ISSUER_LEN);

/// SHA-256-shaped content identity supplied by an external source.
///
/// Shape is validated; artifact bytes are not fetched or re-hashed here.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub fn new(value: impl AsRef<str>) -> Result<Self, ExternalEvidenceError> {
        let value = value.as_ref();
        if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(ExternalEvidenceError::InvalidSha256);
        }
        Ok(Self(value.to_ascii_lowercase()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

/// Caller-declared UTC calendar date. Calendar validity is checked; chronology
/// is not independently attested by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ClaimedUtcDate(u32);

impl ClaimedUtcDate {
    pub fn new(yyyymmdd: u32) -> Result<Self, ExternalEvidenceError> {
        let year = yyyymmdd / 10_000;
        let month = (yyyymmdd / 100) % 100;
        let day = yyyymmdd % 100;
        if year == 0 || month == 0 || month > 12 || day == 0 || day > days_in_month(year, month) {
            return Err(ExternalEvidenceError::InvalidClaimedUtcDate(yyyymmdd));
        }
        Ok(Self(yyyymmdd))
    }

    pub fn yyyymmdd(self) -> u32 {
        self.0
    }
}

impl<'de> Deserialize<'de> for ClaimedUtcDate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = u32::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Declared purpose of a reference, not proof that the referenced claim is true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EvidenceRole {
    ArtifactContent,
    ChronologyAttestation,
    Preregistration,
    ImplementationSnapshot,
    DependencySnapshot,
    SolverExecution,
    ExperimentalObservation,
    IndependentReplication,
    ExperimentalFeasibility,
    ExternalValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceReferenceInterpretation {
    ReferenceOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeclaredChronologyInterpretation {
    DeclaredChronologyOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalEvidenceReference {
    pub id: ExternalEvidenceId,
    pub role: EvidenceRole,
    pub content_sha256: Sha256Digest,
    pub locator: EvidenceLocator,
    pub subject: EvidenceSubject,
    pub claimed_utc_date: Option<ClaimedUtcDate>,
    pub issuer: Option<EvidenceIssuer>,
}

impl ExternalEvidenceReference {
    pub fn try_new(
        id: impl Into<String>,
        role: EvidenceRole,
        sha256_hex: impl AsRef<str>,
        locator: impl Into<String>,
        subject: impl Into<String>,
        claimed_utc_yyyymmdd: Option<u32>,
        issuer: Option<String>,
    ) -> Result<Self, ExternalEvidenceError> {
        Ok(Self {
            id: ExternalEvidenceId::new(id)?,
            role,
            content_sha256: Sha256Digest::new(sha256_hex)?,
            locator: EvidenceLocator::new(locator)?,
            subject: EvidenceSubject::new(subject)?,
            claimed_utc_date: claimed_utc_yyyymmdd.map(ClaimedUtcDate::new).transpose()?,
            issuer: issuer.map(EvidenceIssuer::new).transpose()?,
        })
    }

    pub fn interpretation(&self) -> EvidenceReferenceInterpretation {
        EvidenceReferenceInterpretation::ReferenceOnly
    }
}

/// Validated collection with unique caller-facing evidence IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExternalEvidenceBundle {
    references: Vec<ExternalEvidenceReference>,
}

impl ExternalEvidenceBundle {
    pub fn new(references: Vec<ExternalEvidenceReference>) -> Result<Self, ExternalEvidenceError> {
        if references.is_empty() {
            return Err(ExternalEvidenceError::EmptyBundle);
        }
        let mut seen = BTreeSet::new();
        for reference in &references {
            let id = reference.id.as_str().to_owned();
            if !seen.insert(id.clone()) {
                return Err(ExternalEvidenceError::DuplicateEvidenceId(id));
            }
        }
        Ok(Self { references })
    }

    pub fn references(&self) -> &[ExternalEvidenceReference] {
        &self.references
    }

    pub fn get(&self, id: &str) -> Result<&ExternalEvidenceReference, ExternalEvidenceError> {
        self.references
            .iter()
            .find(|reference| reference.id.as_str() == id)
            .ok_or_else(|| ExternalEvidenceError::UnknownEvidenceId(id.to_owned()))
    }

    pub fn require_role(
        &self,
        id: &str,
        expected: EvidenceRole,
    ) -> Result<&ExternalEvidenceReference, ExternalEvidenceError> {
        let reference = self.get(id)?;
        if reference.role != expected {
            return Err(ExternalEvidenceError::RoleMismatch {
                evidence_id: id.to_owned(),
                expected,
                actual: reference.role,
            });
        }
        Ok(reference)
    }

    pub fn require_distinct_content(
        &self,
        left_id: &str,
        right_id: &str,
    ) -> Result<(), ExternalEvidenceError> {
        if left_id == right_id {
            return Err(ExternalEvidenceError::SameEvidenceReference);
        }
        let left = self.get(left_id)?;
        let right = self.get(right_id)?;
        if left.content_sha256 == right.content_sha256 {
            return Err(ExternalEvidenceError::SameContentIdentity);
        }
        Ok(())
    }

    /// Produce a locally consistent, declared-only temporal relation.
    pub fn declared_temporal_order(
        &self,
        earlier_id: &str,
        later_id: &str,
    ) -> Result<DeclaredTemporalRelation, ExternalEvidenceError> {
        self.require_distinct_content(earlier_id, later_id)?;
        let earlier = self.get(earlier_id)?;
        let later = self.get(later_id)?;
        let earlier_date = earlier
            .claimed_utc_date
            .ok_or_else(|| ExternalEvidenceError::MissingClaimedUtcDate(earlier_id.to_owned()))?;
        let later_date = later
            .claimed_utc_date
            .ok_or_else(|| ExternalEvidenceError::MissingClaimedUtcDate(later_id.to_owned()))?;
        if earlier_date >= later_date {
            return Err(ExternalEvidenceError::NonIncreasingDeclaredChronology {
                earlier_yyyymmdd: earlier_date.yyyymmdd(),
                later_yyyymmdd: later_date.yyyymmdd(),
            });
        }
        Ok(DeclaredTemporalRelation {
            earlier_id: earlier.id.clone(),
            later_id: later.id.clone(),
            earlier_date,
            later_date,
        })
    }

    pub fn require_preregistration_before(
        &self,
        preregistration_id: &str,
        target_id: &str,
    ) -> Result<DeclaredTemporalRelation, ExternalEvidenceError> {
        self.require_role(preregistration_id, EvidenceRole::Preregistration)?;
        self.declared_temporal_order(preregistration_id, target_id)
    }
}

impl<'de> Deserialize<'de> for ExternalEvidenceBundle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawBundle {
            references: Vec<ExternalEvidenceReference>,
        }
        let raw = RawBundle::deserialize(deserializer)?;
        Self::new(raw.references).map_err(serde::de::Error::custom)
    }
}

/// Output-only temporal relation. It intentionally does **not** implement
/// `Deserialize`: callers cannot mint one from JSON without the bundle checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeclaredTemporalRelation {
    earlier_id: ExternalEvidenceId,
    later_id: ExternalEvidenceId,
    earlier_date: ClaimedUtcDate,
    later_date: ClaimedUtcDate,
}

impl DeclaredTemporalRelation {
    pub fn earlier_id(&self) -> &ExternalEvidenceId {
        &self.earlier_id
    }

    pub fn later_id(&self) -> &ExternalEvidenceId {
        &self.later_id
    }

    pub fn earlier_date(&self) -> ClaimedUtcDate {
        self.earlier_date
    }

    pub fn later_date(&self) -> ClaimedUtcDate {
        self.later_date
    }

    pub fn interpretation(&self) -> DeclaredChronologyInterpretation {
        DeclaredChronologyInterpretation::DeclaredChronologyOnly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn reference(
        id: &str,
        role: EvidenceRole,
        digest_char: char,
        date: Option<u32>,
    ) -> ExternalEvidenceReference {
        ExternalEvidenceReference::try_new(
            id,
            role,
            digest(digest_char),
            format!("artifact://{id}"),
            format!("subject:{id}"),
            date,
            Some("fixture issuer".into()),
        )
        .unwrap()
    }

    #[test]
    fn digest_and_calendar_validation_fail_closed() {
        assert!(Sha256Digest::new("bad").is_err());
        assert!(ClaimedUtcDate::new(20240229).is_ok());
        assert!(ClaimedUtcDate::new(20230229).is_err());
    }

    #[test]
    fn malformed_reference_serde_cannot_bypass_validated_fields() {
        let json = r#"{
            "id":"artifact-a",
            "role":"ArtifactContent",
            "content_sha256":"not-a-digest",
            "locator":"artifact://a",
            "subject":"A",
            "claimed_utc_date":20230229,
            "issuer":null
        }"#;
        assert!(serde_json::from_str::<ExternalEvidenceReference>(json).is_err());
    }

    #[test]
    fn bundle_deserialization_rechecks_unique_ids() {
        let a = reference("same", EvidenceRole::ArtifactContent, 'a', Some(20200101));
        let b = reference("same", EvidenceRole::ArtifactContent, 'b', Some(20210101));
        let json = serde_json::json!({ "references": [a, b] }).to_string();
        assert!(serde_json::from_str::<ExternalEvidenceBundle>(&json).is_err());
    }

    #[test]
    fn role_and_content_distinctness_fail_closed() {
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("a", EvidenceRole::SolverExecution, 'a', Some(20200101)),
            reference("b", EvidenceRole::ArtifactContent, 'a', Some(20210101)),
        ])
        .unwrap();
        assert!(matches!(
            bundle.require_role("a", EvidenceRole::ExperimentalObservation),
            Err(ExternalEvidenceError::RoleMismatch { .. })
        ));
        assert!(matches!(
            bundle.declared_temporal_order("a", "b"),
            Err(ExternalEvidenceError::SameContentIdentity)
        ));
    }

    #[test]
    fn declared_chronology_is_increasing_and_never_verified_locally() {
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("early", EvidenceRole::Preregistration, 'a', Some(20200101)),
            reference("late", EvidenceRole::SolverExecution, 'b', Some(20210101)),
        ])
        .unwrap();
        let relation = bundle
            .require_preregistration_before("early", "late")
            .expect("declared order should be locally consistent");
        assert_eq!(relation.earlier_id().as_str(), "early");
        assert_eq!(
            relation.interpretation(),
            DeclaredChronologyInterpretation::DeclaredChronologyOnly
        );
        assert_eq!(
            bundle.get("late").unwrap().interpretation(),
            EvidenceReferenceInterpretation::ReferenceOnly
        );
    }

    #[test]
    fn reversed_declared_chronology_is_rejected() {
        let bundle = ExternalEvidenceBundle::new(vec![
            reference("early", EvidenceRole::Preregistration, 'a', Some(20220101)),
            reference("late", EvidenceRole::SolverExecution, 'b', Some(20210101)),
        ])
        .unwrap();
        assert!(matches!(
            bundle.require_preregistration_before("early", "late"),
            Err(ExternalEvidenceError::NonIncreasingDeclaredChronology { .. })
        ));
    }

    #[test]
    fn reference_round_trip_retains_reference_only_authority() {
        let reference = reference(
            "experiment",
            EvidenceRole::ExperimentalObservation,
            'a',
            Some(20260101),
        );
        let json = serde_json::to_string(&reference).unwrap();
        let parsed: ExternalEvidenceReference = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, reference);
        assert_eq!(parsed.interpretation(), EvidenceReferenceInterpretation::ReferenceOnly);
    }
}
