// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance-first prospective prediction commitments.
//!
//! This module creates immutable commitments for predictions made before
//! outcome exposure. It intentionally cannot emit an observation, replication,
//! or official-criterion event. Supersession is represented by a new
//! commitment with the prior commitment as a parent.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProspectiveProvenance {
    pub exact_input_digest: String,
    pub artifact_digest: String,
    pub model_lineage: String,
    pub knowledge_cutoff: String,
    pub exposure_cutoff: String,
}

impl ProspectiveProvenance {
    pub fn new(
        input_digest: impl Into<String>,
        artifact_digest: impl Into<String>,
        model_lineage: impl Into<String>,
        knowledge_cutoff: impl Into<String>,
        exposure_cutoff: impl Into<String>,
    ) -> Result<Self, CommitmentError> {
        let value = Self {
            exact_input_digest: input_digest.into(),
            artifact_digest: artifact_digest.into(),
            model_lineage: model_lineage.into(),
            knowledge_cutoff: knowledge_cutoff.into(),
            exposure_cutoff: exposure_cutoff.into(),
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), CommitmentError> {
        for (name, value) in [
            ("exact_input_digest", &self.exact_input_digest),
            ("artifact_digest", &self.artifact_digest),
            ("model_lineage", &self.model_lineage),
            ("knowledge_cutoff", &self.knowledge_cutoff),
            ("exposure_cutoff", &self.exposure_cutoff),
        ] {
            if value.trim().is_empty() {
                return Err(CommitmentError::MissingField(name));
            }
        }
        validate_utc_cutoff(&self.knowledge_cutoff)?;
        validate_utc_cutoff(&self.exposure_cutoff)?;
        if self.exposure_cutoff < self.knowledge_cutoff {
            return Err(CommitmentError::ExposureBeforeKnowledgeCutoff);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProspectivePredictionCommitment {
    event_id: String,
    challenge_id: String,
    criteria_generation: String,
    mapping_generation: String,
    actor_id: String,
    created_at: String,
    provenance: ProspectiveProvenance,
    payload_digest: String,
    parent_event_ids: Vec<String>,
}

impl ProspectivePredictionCommitment {
    pub fn commit(
        challenge_id: impl Into<String>,
        criteria_generation: impl Into<String>,
        mapping_generation: impl Into<String>,
        actor_id: impl Into<String>,
        created_at: impl Into<String>,
        provenance: ProspectiveProvenance,
        prediction_payload: &[u8],
    ) -> Result<Self, CommitmentError> {
        let challenge_id = challenge_id.into();
        let criteria_generation = criteria_generation.into();
        let mapping_generation = mapping_generation.into();
        let actor_id = actor_id.into();
        let created_at = created_at.into();

        require_nonempty("challenge_id", &challenge_id)?;
        require_nonempty("criteria_generation", &criteria_generation)?;
        require_nonempty("mapping_generation", &mapping_generation)?;
        require_nonempty("actor_id", &actor_id)?;
        require_nonempty("created_at", &created_at)?;
        if prediction_payload.is_empty() {
            return Err(CommitmentError::EmptyPredictionPayload);
        }
        provenance.validate()?;

        let payload_digest = sha256_hex(prediction_payload);
        let event_id = commitment_digest(
            &challenge_id,
            &criteria_generation,
            &mapping_generation,
            &actor_id,
            &created_at,
            &provenance,
            &payload_digest,
            &[],
        );

        Ok(Self {
            event_id,
            challenge_id,
            criteria_generation,
            mapping_generation,
            actor_id,
            created_at,
            provenance,
            payload_digest,
            parent_event_ids: Vec::new(),
        })
    }

    pub fn event_id(&self) -> &str { &self.event_id }
    pub fn challenge_id(&self) -> &str { &self.challenge_id }
    pub fn criteria_generation(&self) -> &str { &self.criteria_generation }
    pub fn mapping_generation(&self) -> &str { &self.mapping_generation }
    pub fn actor_id(&self) -> &str { &self.actor_id }
    pub fn created_at(&self) -> &str { &self.created_at }
    pub fn provenance(&self) -> &ProspectiveProvenance { &self.provenance }
    pub fn payload_digest(&self) -> &str { &self.payload_digest }
    pub fn parent_event_ids(&self) -> &[String] { &self.parent_event_ids }

    /// Verify that the immutable event identifier still matches every
    /// identity-bearing field. This is intended for deserialized records and
    /// downstream envelope verification; it emits no evidence.
    pub fn verify_integrity(&self) -> bool {
        self.event_id == commitment_digest(
            &self.challenge_id,
            &self.criteria_generation,
            &self.mapping_generation,
            &self.actor_id,
            &self.created_at,
            &self.provenance,
            &self.payload_digest,
            &self.parent_event_ids,
        )
    }

    /// Create a superseding commitment. The original remains unchanged.
    pub fn supersede(
        &self,
        actor_id: impl Into<String>,
        created_at: impl Into<String>,
        prediction_payload: &[u8],
        provenance: ProspectiveProvenance,
    ) -> Result<Self, CommitmentError> {
        let mut next = Self::commit(
            self.challenge_id.clone(),
            self.criteria_generation.clone(),
            self.mapping_generation.clone(),
            actor_id,
            created_at,
            provenance,
            prediction_payload,
        )?;
        next.parent_event_ids.push(self.event_id.clone());
        next.event_id = commitment_digest(
            &next.challenge_id,
            &next.criteria_generation,
            &next.mapping_generation,
            &next.actor_id,
            &next.created_at,
            &next.provenance,
            &next.payload_digest,
            &next.parent_event_ids,
        );
        Ok(next)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitmentError {
    MissingField(&'static str),
    EmptyPredictionPayload,
    ExposureBeforeKnowledgeCutoff,
    InvalidCutoff,
}

impl fmt::Display for CommitmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(field) => write!(f, "missing required field: {field}"),
            Self::EmptyPredictionPayload => write!(f, "prediction payload must not be empty"),
            Self::ExposureBeforeKnowledgeCutoff => {
                write!(f, "exposure cutoff must not precede knowledge cutoff")
            }
            Self::InvalidCutoff => write!(f, "cutoff must be canonical UTC YYYY-MM-DDTHH:MM:SSZ"),
        }
    }
}

impl std::error::Error for CommitmentError {}

fn require_nonempty(name: &'static str, value: &str) -> Result<(), CommitmentError> {
    if value.trim().is_empty() {
        Err(CommitmentError::MissingField(name))
    } else {
        Ok(())
    }
}

fn validate_utc_cutoff(value: &str) -> Result<(), CommitmentError> {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
        || ![0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]
            .iter()
            .all(|&i| bytes[i].is_ascii_digit())
    {
        return Err(CommitmentError::InvalidCutoff);
    }
    let year = (bytes[0] - b'0') as u32 * 1000
        + (bytes[1] - b'0') as u32 * 100
        + (bytes[2] - b'0') as u32 * 10
        + (bytes[3] - b'0') as u32;
    let month = (bytes[5] - b'0') * 10 + bytes[6] - b'0';
    let day = (bytes[8] - b'0') * 10 + bytes[9] - b'0';
    let hour = (bytes[11] - b'0') * 10 + bytes[12] - b'0';
    let minute = (bytes[14] - b'0') * 10 + bytes[15] - b'0';
    let second = (bytes[17] - b'0') * 10 + bytes[18] - b'0';
    if !(1..=12).contains(&month)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(CommitmentError::InvalidCutoff);
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => unreachable!(),
    };
    if day == 0 || day > days_in_month {
        return Err(CommitmentError::InvalidCutoff);
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn commitment_digest(
    challenge_id: &str,
    criteria_generation: &str,
    mapping_generation: &str,
    actor_id: &str,
    created_at: &str,
    provenance: &ProspectiveProvenance,
    payload_digest: &str,
    parents: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea:prospective-prediction-commitment:v1\0");
    for value in [
        challenge_id,
        criteria_generation,
        mapping_generation,
        actor_id,
        created_at,
        &provenance.exact_input_digest,
        &provenance.artifact_digest,
        &provenance.model_lineage,
        &provenance.knowledge_cutoff,
        &provenance.exposure_cutoff,
        payload_digest,
    ] {
        hasher.update((value.len() as u64).to_be_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update((parents.len() as u64).to_be_bytes());
    for parent in parents {
        hasher.update((parent.len() as u64).to_be_bytes());
        hasher.update(parent.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> ProspectiveProvenance {
        ProspectiveProvenance::new(
            "sha256:input-v1",
            "sha256:artifact-v1",
            "model:lineage-v1",
            "2026-09-28T08:00:00Z",
            "2026-09-28T09:00:00Z",
        ).unwrap()
    }

    fn commit(payload: &[u8]) -> ProspectivePredictionCommitment {
        ProspectivePredictionCommitment::commit(
            "MPB-2026-09-23-01",
            "DESCI-BIO-001B-2026-09-28-01",
            "BIO-MILLENNIUM-001B-2026-09-28-01",
            "actor:test",
            "2026-09-28T09:00:00Z",
            provenance(),
            payload,
        ).unwrap()
    }

    #[test]
    fn deterministic_commitment_identity() {
        assert_eq!(commit(b"prediction").event_id(), commit(b"prediction").event_id());
    }

    #[test]
    fn changing_prediction_changes_identity_and_payload_digest() {
        let a = commit(b"prediction-a");
        let b = commit(b"prediction-b");
        assert_ne!(a.event_id(), b.event_id());
        assert_ne!(a.payload_digest(), b.payload_digest());
    }

    #[test]
    fn missing_exposure_cutoff_is_rejected() {
        let result = ProspectiveProvenance::new(
            "sha256:input", "sha256:artifact", "model:v1",
            "2026-09-28T08:00:00Z", "",
        );
        assert_eq!(result, Err(CommitmentError::MissingField("exposure_cutoff")));
    }

    #[test]
    fn retrospective_cutoff_order_is_rejected() {
        let result = ProspectiveProvenance::new(
            "sha256:input", "sha256:artifact", "model:v1",
            "2026-09-28T10:00:00Z", "2026-09-28T09:00:00Z",
        );
        assert_eq!(result, Err(CommitmentError::ExposureBeforeKnowledgeCutoff));
    }

    #[test]
    fn supersession_creates_new_event_and_preserves_parent() {
        let original = commit(b"prediction-v1");
        let replacement = original.supersede(
            "actor:test",
            "2026-09-28T10:00:00Z",
            b"prediction-v2",
            provenance(),
        ).unwrap();

        assert_ne!(original.event_id(), replacement.event_id());
        assert_eq!(replacement.parent_event_ids(), &[original.event_id().to_string()]);
        assert_eq!(original.payload_digest(), commit(b"prediction-v1").payload_digest());
    }

    #[test]
    fn tampered_event_id_fails_integrity_verification() {
        let mut value = commit(b"prediction");
        value.event_id = "sha256:tampered".into();
        assert!(!value.verify_integrity());
    }

    #[test]
    fn malformed_cutoff_is_rejected_before_lexical_comparison() {
        let result = ProspectiveProvenance::new(
            "sha256:input", "sha256:artifact", "model:v1",
            "2026-09-28T08:00:00Z", "2026-09-28T09:00Z",
        );
        assert_eq!(result, Err(CommitmentError::InvalidCutoff));
    }

    #[test]
    fn invalid_gregorian_dates_are_rejected() {
        for cutoff in [
            "2026-02-29T09:00:00Z",
            "2026-04-31T09:00:00Z",
            "2026-00-10T09:00:00Z",
        ] {
            assert_eq!(
                ProspectiveProvenance::new(
                    "sha256:input", "sha256:artifact", "model:v1", cutoff, "2026-05-01T09:00:00Z",
                ),
                Err(CommitmentError::InvalidCutoff)
            );
        }
    }

    #[test]
    fn leap_day_is_accepted_only_in_leap_years() {
        assert!(ProspectiveProvenance::new(
            "sha256:input", "sha256:artifact", "model:v1",
            "2028-02-29T09:00:00Z", "2028-03-01T09:00:00Z",
        ).is_ok());
        assert_eq!(
            ProspectiveProvenance::new(
                "sha256:input", "sha256:artifact", "model:v1",
                "2100-02-29T09:00:00Z", "2100-03-01T09:00:00Z",
            ),
            Err(CommitmentError::InvalidCutoff)
        );
    }

    #[test]
    fn empty_prediction_is_rejected() {
        let result = ProspectivePredictionCommitment::commit(
            "MPB-2026-09-23-01",
            "DESCI-BIO-001B-2026-09-28-01",
            "BIO-MILLENNIUM-001B-2026-09-28-01",
            "actor:test",
            "2026-09-28T09:00:00Z",
            provenance(),
            b"",
        );
        assert_eq!(result, Err(CommitmentError::EmptyPredictionPayload));
    }
}
