// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance-preserving records of distinct replication attempts.
//! Replication is an attempt record, not proof of truth or criterion completion.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

use crate::external_observation::ExternalExperimentalObservation;
use crate::independent_assessment::IndependentAssessment;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicationOutcome {
    ReplicatedSupportive,
    NonReplicatingContradictory,
    NullResult,
    Inconclusive,
    ProceduralInvalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationInput {
    pub replication_id: String,
    pub execution_id: String,
    pub observer_id: String,
    pub institution_id: String,
    /// Explicit attestation that this replication is independent of the original observer.
    /// The crate does not authenticate this claim.
    pub independent_from_original_observer: bool,
    pub independence_basis: String,
    pub replicated_at: String,
    pub outcome: ReplicationOutcome,
    /// Opaque caller-supplied replication record; no scientific interpretation is assigned.
    pub replication_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicationRecord {
    pub replication_id: String,
    pub execution_id: String,
    pub observer_id: String,
    pub institution_id: String,
    pub independent_from_original_observer: bool,
    pub independence_basis: String,
    pub replicated_at: String,
    pub outcome: ReplicationOutcome,
    pub observation_id: String,
    pub observation_record_digest: String,
    pub assessment_id: String,
    pub assessment_record_digest: String,
    pub commitment_event_id: String,
    pub candidate_id: String,
    pub binding_digest: String,
    pub replication_digest: String,
    pub record_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicationError {
    MissingField(&'static str),
    EmptyPayload,
    NotIndependent,
    InvalidTimestamp,
    InvalidObservation,
    InvalidAssessment,
    LinkMismatch(&'static str),
}

impl fmt::Display for ReplicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(s) => write!(f, "missing required field: {s}"),
            Self::EmptyPayload => write!(f, "replication payload must not be empty"),
            Self::NotIndependent => write!(f, "replication independence must be explicitly attested"),
            Self::InvalidTimestamp => write!(f, "replicated_at must be canonical UTC YYYY-MM-DDTHH:MM:SSZ"),
            Self::InvalidObservation => write!(f, "observation envelope integrity check failed"),
            Self::InvalidAssessment => write!(f, "assessment envelope integrity check failed"),
            Self::LinkMismatch(s) => write!(f, "replication linkage mismatch: {s}"),
        }
    }
}

impl std::error::Error for ReplicationError {}

impl ReplicationRecord {
    /// Record a distinct replication attempt against one exact observation and assessment.
    /// This does not establish scientific truth or promote official criterion evidence.
    pub fn record(
        observation: &ExternalExperimentalObservation,
        assessment: &IndependentAssessment,
        input: ReplicationInput,
    ) -> Result<Self, ReplicationError> {
        if !observation.verify_integrity() {
            return Err(ReplicationError::InvalidObservation);
        }
        if !assessment.verify_integrity() {
            return Err(ReplicationError::InvalidAssessment);
        }

        if assessment.observation_id != observation.observation_id {
            return Err(ReplicationError::LinkMismatch("assessment observation_id"));
        }
        if assessment.observation_record_digest != observation.record_digest {
            return Err(ReplicationError::LinkMismatch("assessment observation_record_digest"));
        }
        if assessment.commitment_event_id != observation.commitment_event_id {
            return Err(ReplicationError::LinkMismatch("commitment_event_id"));
        }
        if assessment.candidate_id != observation.candidate_id {
            return Err(ReplicationError::LinkMismatch("candidate_id"));
        }
        if assessment.binding_digest != observation.binding_digest {
            return Err(ReplicationError::LinkMismatch("binding_digest"));
        }

        for (name, value) in [
            ("replication_id", &input.replication_id),
            ("execution_id", &input.execution_id),
            ("observer_id", &input.observer_id),
            ("institution_id", &input.institution_id),
            ("independence_basis", &input.independence_basis),
        ] {
            if value.trim().is_empty() {
                return Err(ReplicationError::MissingField(name));
            }
        }

        if input.execution_id == observation.execution_id || input.observer_id == observation.observer_id {
            return Err(ReplicationError::NotIndependent);
        }
        if !input.independent_from_original_observer {
            return Err(ReplicationError::NotIndependent);
        }
        if !valid_utc(&input.replicated_at) {
            return Err(ReplicationError::InvalidTimestamp);
        }
        if input.replication_payload.is_empty() {
            return Err(ReplicationError::EmptyPayload);
        }

        let replication_digest =
            hash(b"symthaea:replication-payload:v1\0", &input.replication_payload);
        let mut r = Self {
            replication_id: input.replication_id,
            execution_id: input.execution_id,
            observer_id: input.observer_id,
            institution_id: input.institution_id,
            independent_from_original_observer: input.independent_from_original_observer,
            independence_basis: input.independence_basis,
            replicated_at: input.replicated_at,
            outcome: input.outcome,
            observation_id: observation.observation_id.clone(),
            observation_record_digest: observation.record_digest.clone(),
            assessment_id: assessment.assessment_id.clone(),
            assessment_record_digest: assessment.record_digest.clone(),
            commitment_event_id: observation.commitment_event_id.clone(),
            candidate_id: observation.candidate_id.clone(),
            binding_digest: observation.binding_digest.clone(),
            replication_digest,
            record_digest: String::new(),
        };
        r.record_digest = r.digest();
        Ok(r)
    }

    /// Checks envelope consistency only; it cannot authenticate independence or prove replication.
    pub fn verify_integrity(&self) -> bool {
        [
            self.replication_id.as_str(),
            self.execution_id.as_str(),
            self.observer_id.as_str(),
            self.institution_id.as_str(),
            self.independence_basis.as_str(),
            self.observation_id.as_str(),
            self.observation_record_digest.as_str(),
            self.assessment_id.as_str(),
            self.assessment_record_digest.as_str(),
            self.commitment_event_id.as_str(),
            self.candidate_id.as_str(),
            self.binding_digest.as_str(),
            self.replication_digest.as_str(),
        ]
        .iter()
        .all(|s| !s.trim().is_empty())
            && self.independent_from_original_observer
            && valid_utc(&self.replicated_at)
            && self.record_digest == self.digest()
    }

    fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"symthaea:replication-record:v1\0");
        for s in [
            self.replication_id.as_str(),
            self.execution_id.as_str(),
            self.observer_id.as_str(),
            self.institution_id.as_str(),
            if self.independent_from_original_observer { "true" } else { "false" },
            self.independence_basis.as_str(),
            self.replicated_at.as_str(),
            outcome_code(self.outcome),
            self.observation_id.as_str(),
            self.observation_record_digest.as_str(),
            self.assessment_id.as_str(),
            self.assessment_record_digest.as_str(),
            self.commitment_event_id.as_str(),
            self.candidate_id.as_str(),
            self.binding_digest.as_str(),
            self.replication_digest.as_str(),
        ] {
            h.update((s.len() as u64).to_be_bytes());
            h.update(s.as_bytes());
        }
        format!("sha256:{:x}", h.finalize())
    }
}

fn outcome_code(o: ReplicationOutcome) -> &'static str {
    match o {
        ReplicationOutcome::ReplicatedSupportive => "replicated_supportive",
        ReplicationOutcome::NonReplicatingContradictory => "non_replicating_contradictory",
        ReplicationOutcome::NullResult => "null_result",
        ReplicationOutcome::Inconclusive => "inconclusive",
        ReplicationOutcome::ProceduralInvalid => "procedural_invalid",
    }
}

fn hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(domain);
    h.update((bytes.len() as u64).to_be_bytes());
    h.update(bytes);
    format!("sha256:{:x}", h.finalize())
}

fn valid_utc(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'Z'
        || ![0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]
            .iter()
            .all(|&i| b[i].is_ascii_digit())
    {
        return false;
    }
    let pair = |i: usize| (b[i] - b'0') as u32 * 10 + (b[i + 1] - b'0') as u32;
    let y = (b[0] - b'0') as u32 * 1000
        + (b[1] - b'0') as u32 * 100
        + (b[2] - b'0') as u32 * 10
        + (b[3] - b'0') as u32;
    let m = pair(5);
    let d = pair(8);
    if !(1..=12).contains(&m) || pair(11) > 23 || pair(14) > 59 || pair(17) > 59 {
        return false;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let max = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    d >= 1 && d <= max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_commitment::{
        commit_candidate_envelope, CandidatePredictionBinding, CandidatePredictionSource,
    };
    use crate::external_observation::{
        ExternalExperimentalObservation, ExternalObservationInput, ObservationDisposition,
    };
    use crate::independent_assessment::{AssessmentInput, AssessmentOutcome, IndependentAssessment};
    use crate::prospective::ProspectiveProvenance;

    fn source() -> CandidatePredictionSource {
        CandidatePredictionSource {
            candidate_id: "candidate:1".into(),
            source_candidate_id: "source:1".into(),
            test_specification_id: "test:1".into(),
            measurement_specification_id: "measure:1".into(),
            left_lineage: "left".into(),
            right_lineage: "right".into(),
        }
    }

    fn setup() -> (ExternalExperimentalObservation, IndependentAssessment) {
        let s = source();
        let b = CandidatePredictionBinding::from_source(&s, b"prediction").unwrap();
        let p = ProspectiveProvenance::new(
            "input",
            "artifact",
            b.lineage_digest().unwrap(),
            "2026-09-28T08:00:00Z",
            "2026-09-28T09:00:00Z",
        )
        .unwrap();
        let c = commit_candidate_envelope(
            &s,
            "challenge",
            "criteria",
            "mapping",
            "predictor",
            "2026-09-28T09:00:00Z",
            p,
            b"prediction",
        )
        .unwrap();
        let o = ExternalExperimentalObservation::ingest(
            &c,
            &b,
            ExternalObservationInput {
                observation_id: "obs:1".into(),
                execution_id: "exec:original".into(),
                observer_id: "observer:original".into(),
                institution_id: "institution:original".into(),
                observed_at: "2026-09-29T10:00:00Z".into(),
                disposition: ObservationDisposition::Reported,
                observation_payload: b"opaque observation".to_vec(),
            },
        )
        .unwrap();
        let a = IndependentAssessment::assess(
            &o,
            AssessmentInput {
                assessment_id: "assessment:1".into(),
                assessor_id: "assessor:1".into(),
                assessor_institution_id: "institution:review".into(),
                independent_from_observer: true,
                independence_basis: "separate review".into(),
                assessed_at: "2026-09-29T11:00:00Z".into(),
                outcome: AssessmentOutcome::Supports,
                assessment_payload: b"assessment".to_vec(),
            },
        )
        .unwrap();
        (o, a)
    }

    fn input(outcome: ReplicationOutcome) -> ReplicationInput {
        ReplicationInput {
            replication_id: "replication:1".into(),
            execution_id: "exec:replica".into(),
            observer_id: "observer:replica".into(),
            institution_id: "institution:replica".into(),
            independent_from_original_observer: true,
            independence_basis: "distinct execution and observer".into(),
            replicated_at: "2026-09-29T12:00:00Z".into(),
            outcome,
            replication_payload: b"opaque replication record".to_vec(),
        }
    }

    #[test]
    fn links_exact_observation_and_assessment() {
        let (o, a) = setup();
        let r = ReplicationRecord::record(&o, &a, input(ReplicationOutcome::ReplicatedSupportive)).unwrap();
        assert!(r.verify_integrity());
        assert_eq!(r.observation_id, o.observation_id);
        assert_eq!(r.observation_record_digest, o.record_digest);
        assert_eq!(r.assessment_id, a.assessment_id);
        assert_eq!(r.assessment_record_digest, a.record_digest);
        assert_eq!(r.commitment_event_id, o.commitment_event_id);
        assert_eq!(r.candidate_id, o.candidate_id);
        assert_eq!(r.binding_digest, o.binding_digest);
    }

    #[test]
    fn preserves_all_replication_outcomes() {
        let (o, a) = setup();
        for outcome in [
            ReplicationOutcome::ReplicatedSupportive,
            ReplicationOutcome::NonReplicatingContradictory,
            ReplicationOutcome::NullResult,
            ReplicationOutcome::Inconclusive,
            ReplicationOutcome::ProceduralInvalid,
        ] {
            let r = ReplicationRecord::record(&o, &a, input(outcome)).unwrap();
            assert_eq!(r.outcome, outcome);
        }
    }

    #[test]
    fn rejects_same_execution_observer_and_missing_attestation() {
        let (o, a) = setup();
        let mut i = input(ReplicationOutcome::Inconclusive);
        i.execution_id = o.execution_id.clone();
        assert_eq!(ReplicationRecord::record(&o, &a, i), Err(ReplicationError::NotIndependent));

        let mut i = input(ReplicationOutcome::Inconclusive);
        i.observer_id = o.observer_id.clone();
        assert_eq!(ReplicationRecord::record(&o, &a, i), Err(ReplicationError::NotIndependent));

        let mut i = input(ReplicationOutcome::Inconclusive);
        i.independent_from_original_observer = false;
        assert_eq!(ReplicationRecord::record(&o, &a, i), Err(ReplicationError::NotIndependent));
    }

    #[test]
    fn rejects_link_mismatch_and_invalid_inputs() {
        let (o, mut a) = setup();
        a.observation_record_digest = "sha256:changed".into();
        a.record_digest = "sha256:stale".into();
        assert_eq!(
            ReplicationRecord::record(&o, &a, input(ReplicationOutcome::Supports)),
            Err(ReplicationError::InvalidAssessment)
        );

        let (o, a) = setup();
        let mut i = input(ReplicationOutcome::Inconclusive);
        i.replicated_at = "2026-02-29T12:00:00Z".into();
        assert_eq!(ReplicationRecord::record(&o, &a, i), Err(ReplicationError::InvalidTimestamp));

        let mut i = input(ReplicationOutcome::Inconclusive);
        i.replication_payload.clear();
        assert_eq!(ReplicationRecord::record(&o, &a, i), Err(ReplicationError::EmptyPayload));
    }

    #[test]
    fn detects_record_tampering_and_round_trips() {
        let (o, a) = setup();
        let r = ReplicationRecord::record(&o, &a, input(ReplicationOutcome::NullResult)).unwrap();
        assert!(r.verify_integrity());
        let json = serde_json::to_vec(&r).unwrap();
        let decoded: ReplicationRecord = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded, r);
        let mut tampered = r;
        tampered.outcome = ReplicationOutcome::Contradictory;
        assert!(!tampered.verify_integrity());
    }
}
