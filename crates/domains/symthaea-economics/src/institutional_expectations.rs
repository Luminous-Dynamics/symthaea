// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, append-only institutional expectation state.
//!
//! Expectations are observations about an agent at a point in time. They are
//! deliberately non-authoritative: recording or updating a belief never changes
//! the authoritative institution state.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectationRepresentation {
    Probability,
    Interval,
    Ordinal,
    Categorical,
    DeterministicForecast,
    BoundedHeuristic,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InformationSetIdentity {
    pub hash: String,
    pub as_of: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectationRecord {
    pub expectation_id: String,
    pub subject_id: String,
    pub observation_time: u64,
    pub referenced_institution_hash: String,
    pub expectation_rule_hash: String,
    pub information_set: InformationSetIdentity,
    pub representation: ExpectationRepresentation,
    pub belief_value: String,
    pub confidence: Option<u8>,
    pub signal_identity: Option<String>,
    pub current: bool,
    pub superseded_by: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActualInstitutionObservation {
    pub observation_id: String,
    pub observation_time: u64,
    pub institution_hash: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectationUpdate {
    pub expectation_id: String,
    pub update_id: String,
    pub update_time: u64,
    pub superseded_expectation_id: String,
    pub new_belief_value: String,
    pub information_set: InformationSetIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectationError {
    DuplicateExpectation,
    DuplicateUpdate,
    EmptyIdentity,
    EmptyProfileHash,
    InformationSetAfterObservation,
    ConfidenceOutOfRange,
    HistoricalRewrite,
    UnknownExpectation,
    UpdateBeforeObservation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectationBook {
    records: BTreeMap<String, ExpectationRecord>,
    updates: BTreeMap<String, ExpectationUpdate>,
    actual_observations: BTreeMap<String, ActualInstitutionObservation>,
}

impl ExpectationBook {
    pub fn new() -> Self {
        Self {
            records: BTreeMap::new(),
            updates: BTreeMap::new(),
            actual_observations: BTreeMap::new(),
        }
    }

    pub fn records(&self) -> &BTreeMap<String, ExpectationRecord> {
        &self.records
    }

    pub fn updates(&self) -> &BTreeMap<String, ExpectationUpdate> {
        &self.updates
    }

    pub fn actual_observations(&self) -> &BTreeMap<String, ActualInstitutionObservation> {
        &self.actual_observations
    }

    pub fn record(&mut self, record: ExpectationRecord) -> Result<(), ExpectationError> {
        validate_record(&record)?;
        if self.records.contains_key(&record.expectation_id) {
            return Err(ExpectationError::DuplicateExpectation);
        }
        self.records.insert(record.expectation_id.clone(), record);
        Ok(())
    }

    pub fn record_actual_observation(
        &mut self,
        observation: ActualInstitutionObservation,
    ) -> Result<(), ExpectationError> {
        if observation.observation_id.is_empty() || observation.institution_hash.is_empty() {
            return Err(ExpectationError::EmptyIdentity);
        }
        if self.actual_observations.contains_key(&observation.observation_id) {
            return Err(ExpectationError::DuplicateExpectation);
        }
        self.actual_observations.insert(observation.observation_id.clone(), observation);
        Ok(())
    }

    pub fn update(&mut self, update: ExpectationUpdate) -> Result<(), ExpectationError> {
        if update.update_id.is_empty() || update.expectation_id.is_empty() {
            return Err(ExpectationError::EmptyIdentity);
        }
        if self.updates.contains_key(&update.update_id) {
            return Err(ExpectationError::DuplicateUpdate);
        }
        let existing = self
            .records
            .get(&update.superseded_expectation_id)
            .ok_or(ExpectationError::UnknownExpectation)?;
        if update.update_time < existing.observation_time {
            return Err(ExpectationError::UpdateBeforeObservation);
        }
        if update.information_set.as_of > update.update_time {
            return Err(ExpectationError::InformationSetAfterObservation);
        }

        let updated = ExpectationRecord {
            expectation_id: update.expectation_id.clone(),
            subject_id: existing.subject_id.clone(),
            observation_time: update.update_time,
            referenced_institution_hash: existing.referenced_institution_hash.clone(),
            expectation_rule_hash: existing.expectation_rule_hash.clone(),
            information_set: update.information_set.clone(),
            representation: existing.representation,
            belief_value: update.new_belief_value.clone(),
            confidence: existing.confidence,
            signal_identity: existing.signal_identity.clone(),
            current: true,
            superseded_by: None,
        };
        validate_record(&updated)?;

        if let Some(previous) = self.records.get_mut(&update.superseded_expectation_id) {
            previous.current = false;
            previous.superseded_by = Some(update.expectation_id.clone());
        }
        self.records.insert(update.expectation_id.clone(), updated);
        self.updates.insert(update.update_id.clone(), update);
        Ok(())
    }
}

impl Default for ExpectationBook {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_record(record: &ExpectationRecord) -> Result<(), ExpectationError> {
    if record.expectation_id.is_empty() || record.subject_id.is_empty() {
        return Err(ExpectationError::EmptyIdentity);
    }
    if record.referenced_institution_hash.is_empty() || record.expectation_rule_hash.is_empty() {
        return Err(ExpectationError::EmptyProfileHash);
    }
    if record.information_set.as_of > record.observation_time {
        return Err(ExpectationError::InformationSetAfterObservation);
    }
    if record.confidence.is_some_and(|value| value > 100) {
        return Err(ExpectationError::ConfidenceOutOfRange);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, time: u64, belief: &str) -> ExpectationRecord {
        ExpectationRecord {
            expectation_id: id.into(),
            subject_id: "agent-1".into(),
            observation_time: time,
            referenced_institution_hash: "profile-v0".into(),
            expectation_rule_hash: "expect-rule-v0".into(),
            information_set: InformationSetIdentity { hash: "info-v0".into(), as_of: time },
            representation: ExpectationRepresentation::Categorical,
            belief_value: belief.into(),
            confidence: Some(80),
            signal_identity: None,
            current: true,
            superseded_by: None,
        }
    }

    #[test]
    fn recording_expectation_does_not_change_an_external_institution() {
        let before = "profile-v0";
        let mut book = ExpectationBook::new();
        book.record(record("e1", 10, "persist")).unwrap();
        assert_eq!(before, "profile-v0");
        assert_eq!(book.records()["e1"].referenced_institution_hash, before);
    }

    #[test]
    fn future_information_is_rejected() {
        let mut r = record("e1", 10, "persist");
        r.information_set.as_of = 11;
        assert_eq!(ExpectationBook::new().record(r), Err(ExpectationError::InformationSetAfterObservation));
    }

    #[test]
    fn duplicate_expectation_is_rejected() {
        let mut book = ExpectationBook::new();
        book.record(record("e1", 10, "persist")).unwrap();
        assert_eq!(book.record(record("e1", 10, "other")), Err(ExpectationError::DuplicateExpectation));
    }

    #[test]
    fn later_belief_supersedes_without_rewriting_history() {
        let mut book = ExpectationBook::new();
        book.record(record("e1", 10, "persist")).unwrap();
        book.update(ExpectationUpdate {
            expectation_id: "e2".into(),
            update_id: "u1".into(),
            update_time: 20,
            superseded_expectation_id: "e1".into(),
            new_belief_value: "fail".into(),
            information_set: InformationSetIdentity { hash: "info-v1".into(), as_of: 20 },
        }).unwrap();
        assert!(!book.records()["e1"].current);
        assert_eq!(book.records()["e1"].belief_value, "persist");
        assert_eq!(book.records()["e1"].superseded_by.as_deref(), Some("e2"));
        assert!(book.records()["e2"].current);
    }

    #[test]
    fn actual_observation_is_separate_from_expectation() {
        let mut book = ExpectationBook::new();
        book.record(record("e1", 10, "persist")).unwrap();
        book.record_actual_observation(ActualInstitutionObservation {
            observation_id: "o1".into(),
            observation_time: 30,
            institution_hash: "profile-v1".into(),
            source: "authority-observation".into(),
        }).unwrap();
        assert_eq!(book.records()["e1"].referenced_institution_hash, "profile-v0");
        assert_eq!(book.actual_observations()["o1"].institution_hash, "profile-v1");
    }

    #[test]
    fn later_outcome_cannot_rewrite_historical_expectation() {
        let mut book = ExpectationBook::new();
        book.record(record("e1", 10, "persist")).unwrap();
        book.record_actual_observation(ActualInstitutionObservation {
            observation_id: "o1".into(),
            observation_time: 30,
            institution_hash: "profile-v2".into(),
            source: "holdout-observation".into(),
        }).unwrap();
        assert_eq!(book.records()["e1"].belief_value, "persist");
    }
}