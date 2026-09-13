// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Substantive, scope-bound consent evidence for Wisdom & Care deliberation.
//!
//! Consent is not inferred from preferences, cultural defaults, empathy, or benefit.
//! This module represents only explicit direct consent or an explicitly authorized
//! proxy path. Emergency authority, if separately modeled, is not consent.

use std::collections::{BTreeSet, HashMap};

use crate::evidence_ledger::{DeliberationEvidenceLedger, FactClaimId};
use crate::perspective::StakeholderId;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConsentScopeId(String);

impl ConsentScopeId {
    pub fn new(value: impl Into<String>) -> Result<Self, ConsentError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ConsentError::EmptyScope);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentState {
    Unknown,
    Affirmed,
    Refused,
    Withdrawn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentProvenance {
    /// Direct statement by the affected person for this exact scope.
    DirectStatement,
    /// Statement by a separately authorized proxy for this exact scope.
    AuthorizedProxy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConsentRecord {
    pub stakeholder: StakeholderId,
    pub scope: ConsentScopeId,
    pub state: ConsentState,
    pub provenance: ConsentProvenance,
    /// Evidence for the consent/refusal/withdrawal statement itself.
    pub fact_evidence: BTreeSet<FactClaimId>,
    /// Optional evidence relevant to decision-making capacity/context.
    pub capacity_evidence: BTreeSet<FactClaimId>,
    /// Evidence relevant to coercion/pressure assessment.
    pub coercion_evidence: BTreeSet<FactClaimId>,
    pub coercion_risk: f32,
    /// Logical evidence revision; this deliberately does not pretend to be secure wall-clock time.
    pub recorded_revision: u64,
    /// Optional logical expiry revision, inclusive.
    pub expires_after_revision: Option<u64>,
}

impl ConsentRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stakeholder: StakeholderId,
        scope: ConsentScopeId,
        state: ConsentState,
        provenance: ConsentProvenance,
        fact_evidence: impl IntoIterator<Item = FactClaimId>,
        capacity_evidence: impl IntoIterator<Item = FactClaimId>,
        coercion_evidence: impl IntoIterator<Item = FactClaimId>,
        coercion_risk: f32,
        recorded_revision: u64,
        expires_after_revision: Option<u64>,
    ) -> Result<Self, ConsentError> {
        validate_metric(coercion_risk)?;
        if expires_after_revision.is_some_and(|expiry| expiry < recorded_revision) {
            return Err(ConsentError::ExpiryBeforeRecord);
        }
        Ok(Self {
            stakeholder,
            scope,
            state,
            provenance,
            fact_evidence: fact_evidence.into_iter().collect(),
            capacity_evidence: capacity_evidence.into_iter().collect(),
            coercion_evidence: coercion_evidence.into_iter().collect(),
            coercion_risk,
            recorded_revision,
            expires_after_revision,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsentPolicy {
    /// Maximum tolerated coercion risk for an affirmation to count as effective.
    pub max_coercion_risk: f32,
    /// Whether positive consent requires explicit capacity/context evidence.
    pub require_capacity_evidence: bool,
}

impl ConsentPolicy {
    pub fn new(
        max_coercion_risk: f32,
        require_capacity_evidence: bool,
    ) -> Result<Self, ConsentError> {
        validate_metric(max_coercion_risk)?;
        Ok(Self {
            max_coercion_risk,
            require_capacity_evidence,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentValidity {
    Effective,
    NotEffective,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConsentInvalidityReason {
    NoRecordForScope,
    UnknownState,
    Refused,
    Withdrawn,
    RecordFromFuture,
    Expired,
    CapacityEvidenceMissing,
    CoercionRiskTooHigh,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsentEvaluation {
    pub state: ConsentState,
    pub validity: ConsentValidity,
    pub reasons: BTreeSet<ConsentInvalidityReason>,
}

impl ConsentEvaluation {
    pub fn is_effective(&self) -> bool {
        self.validity == ConsentValidity::Effective
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConsentLedger {
    records: HashMap<(StakeholderId, ConsentScopeId), Vec<ConsentRecord>>,
}

impl ConsentLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(
        &mut self,
        record: ConsentRecord,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), ConsentError> {
        validate_fact_refs(&record.fact_evidence, evidence)?;
        validate_fact_refs(&record.capacity_evidence, evidence)?;
        validate_fact_refs(&record.coercion_evidence, evidence)?;

        if record.state != ConsentState::Unknown && record.fact_evidence.is_empty() {
            return Err(ConsentError::StatementEvidenceRequired);
        }

        let key = (record.stakeholder.clone(), record.scope.clone());
        let history = self.records.entry(key).or_default();
        if history
            .last()
            .is_some_and(|prior| record.recorded_revision <= prior.recorded_revision)
        {
            return Err(ConsentError::NonMonotonicRevision);
        }
        history.push(record);
        Ok(())
    }

    pub fn latest(
        &self,
        stakeholder: &StakeholderId,
        scope: &ConsentScopeId,
    ) -> Option<&ConsentRecord> {
        self.records
            .get(&(stakeholder.clone(), scope.clone()))
            .and_then(|history| history.last())
    }

    pub fn evaluate(
        &self,
        stakeholder: &StakeholderId,
        scope: &ConsentScopeId,
        current_revision: u64,
        policy: ConsentPolicy,
    ) -> ConsentEvaluation {
        let Some(record) = self.latest(stakeholder, scope) else {
            return ConsentEvaluation {
                state: ConsentState::Unknown,
                validity: ConsentValidity::NotEffective,
                reasons: [ConsentInvalidityReason::NoRecordForScope]
                    .into_iter()
                    .collect(),
            };
        };

        let mut reasons = BTreeSet::new();
        match record.state {
            ConsentState::Unknown => {
                reasons.insert(ConsentInvalidityReason::UnknownState);
            }
            ConsentState::Refused => {
                reasons.insert(ConsentInvalidityReason::Refused);
            }
            ConsentState::Withdrawn => {
                reasons.insert(ConsentInvalidityReason::Withdrawn);
            }
            ConsentState::Affirmed => {}
        }

        if record.recorded_revision > current_revision {
            reasons.insert(ConsentInvalidityReason::RecordFromFuture);
        }
        if record
            .expires_after_revision
            .is_some_and(|expiry| current_revision > expiry)
        {
            reasons.insert(ConsentInvalidityReason::Expired);
        }
        if policy.require_capacity_evidence
            && record.state == ConsentState::Affirmed
            && record.capacity_evidence.is_empty()
        {
            reasons.insert(ConsentInvalidityReason::CapacityEvidenceMissing);
        }
        if record.state == ConsentState::Affirmed
            && record.coercion_risk > policy.max_coercion_risk
        {
            reasons.insert(ConsentInvalidityReason::CoercionRiskTooHigh);
        }

        ConsentEvaluation {
            state: record.state,
            validity: if record.state == ConsentState::Affirmed && reasons.is_empty() {
                ConsentValidity::Effective
            } else {
                ConsentValidity::NotEffective
            },
            reasons,
        }
    }
}

fn validate_fact_refs(
    ids: &BTreeSet<FactClaimId>,
    evidence: &DeliberationEvidenceLedger,
) -> Result<(), ConsentError> {
    for id in ids {
        if evidence.facts().get(id).is_none() {
            return Err(ConsentError::MissingFact(id.clone()));
        }
    }
    Ok(())
}

fn validate_metric(value: f32) -> Result<(), ConsentError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ConsentError::InvalidMetric(value))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConsentError {
    EmptyScope,
    InvalidMetric(f32),
    ExpiryBeforeRecord,
    MissingFact(FactClaimId),
    StatementEvidenceRequired,
    NonMonotonicRevision,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stakeholder() -> StakeholderId {
        StakeholderId::new("person-a").unwrap()
    }

    fn scope(value: &str) -> ConsentScopeId {
        ConsentScopeId::new(value).unwrap()
    }

    fn fact(value: &str) -> FactClaimId {
        FactClaimId::new(value).unwrap()
    }

    fn evidence() -> DeliberationEvidenceLedger {
        let mut evidence = DeliberationEvidenceLedger::new();
        evidence
            .add_fact(fact("statement"), "person explicitly responded", 0.99)
            .unwrap();
        evidence
            .add_fact(fact("capacity"), "capacity/context assessed", 0.8)
            .unwrap();
        evidence
            .add_fact(fact("coercion"), "coercion assessment performed", 0.8)
            .unwrap();
        evidence
    }

    fn record(
        state: ConsentState,
        target_scope: ConsentScopeId,
        revision: u64,
    ) -> ConsentRecord {
        ConsentRecord::new(
            stakeholder(),
            target_scope,
            state,
            ConsentProvenance::DirectStatement,
            [fact("statement")],
            [fact("capacity")],
            [fact("coercion")],
            0.05,
            revision,
            None,
        )
        .unwrap()
    }

    fn policy() -> ConsentPolicy {
        ConsentPolicy::new(0.2, true).unwrap()
    }

    #[test]
    fn affirmed_scope_bound_consent_can_be_effective() {
        let evidence = evidence();
        let mut ledger = ConsentLedger::new();
        let target = scope("option-a");
        ledger
            .record(record(ConsentState::Affirmed, target.clone(), 10), &evidence)
            .unwrap();
        assert!(ledger
            .evaluate(&stakeholder(), &target, 10, policy())
            .is_effective());
    }

    #[test]
    fn consent_for_one_scope_is_not_reused_for_another() {
        let evidence = evidence();
        let mut ledger = ConsentLedger::new();
        ledger
            .record(record(ConsentState::Affirmed, scope("option-a"), 10), &evidence)
            .unwrap();
        let evaluation = ledger.evaluate(&stakeholder(), &scope("option-b"), 10, policy());
        assert!(!evaluation.is_effective());
        assert!(evaluation
            .reasons
            .contains(&ConsentInvalidityReason::NoRecordForScope));
    }

    #[test]
    fn refusal_is_never_converted_into_benefit_based_consent() {
        let evidence = evidence();
        let mut ledger = ConsentLedger::new();
        let target = scope("option-a");
        ledger
            .record(record(ConsentState::Refused, target.clone(), 10), &evidence)
            .unwrap();
        let evaluation = ledger.evaluate(&stakeholder(), &target, 10, policy());
        assert!(!evaluation.is_effective());
        assert!(evaluation.reasons.contains(&ConsentInvalidityReason::Refused));
    }

    #[test]
    fn withdrawal_supersedes_prior_affirmation() {
        let evidence = evidence();
        let mut ledger = ConsentLedger::new();
        let target = scope("option-a");
        ledger
            .record(record(ConsentState::Affirmed, target.clone(), 10), &evidence)
            .unwrap();
        ledger
            .record(record(ConsentState::Withdrawn, target.clone(), 11), &evidence)
            .unwrap();
        let evaluation = ledger.evaluate(&stakeholder(), &target, 11, policy());
        assert_eq!(evaluation.state, ConsentState::Withdrawn);
        assert!(!evaluation.is_effective());
        assert!(evaluation
            .reasons
            .contains(&ConsentInvalidityReason::Withdrawn));
    }

    #[test]
    fn coercion_risk_can_invalidate_affirmation() {
        let evidence = evidence();
        let mut ledger = ConsentLedger::new();
        let target = scope("option-a");
        let coerced = ConsentRecord::new(
            stakeholder(),
            target.clone(),
            ConsentState::Affirmed,
            ConsentProvenance::DirectStatement,
            [fact("statement")],
            [fact("capacity")],
            [fact("coercion")],
            0.8,
            10,
            None,
        )
        .unwrap();
        ledger.record(coerced, &evidence).unwrap();
        let evaluation = ledger.evaluate(&stakeholder(), &target, 10, policy());
        assert!(!evaluation.is_effective());
        assert!(evaluation
            .reasons
            .contains(&ConsentInvalidityReason::CoercionRiskTooHigh));
    }

    #[test]
    fn capacity_requirement_is_explicit_policy_not_assumption() {
        let evidence = evidence();
        let mut ledger = ConsentLedger::new();
        let target = scope("option-a");
        let no_capacity = ConsentRecord::new(
            stakeholder(),
            target.clone(),
            ConsentState::Affirmed,
            ConsentProvenance::DirectStatement,
            [fact("statement")],
            [],
            [fact("coercion")],
            0.05,
            10,
            None,
        )
        .unwrap();
        ledger.record(no_capacity, &evidence).unwrap();
        assert!(!ledger
            .evaluate(&stakeholder(), &target, 10, policy())
            .is_effective());
        let permissive = ConsentPolicy::new(0.2, false).unwrap();
        assert!(ledger
            .evaluate(&stakeholder(), &target, 10, permissive)
            .is_effective());
    }

    #[test]
    fn expired_consent_is_not_effective() {
        let evidence = evidence();
        let mut ledger = ConsentLedger::new();
        let target = scope("option-a");
        let expiring = ConsentRecord::new(
            stakeholder(),
            target.clone(),
            ConsentState::Affirmed,
            ConsentProvenance::DirectStatement,
            [fact("statement")],
            [fact("capacity")],
            [fact("coercion")],
            0.05,
            10,
            Some(12),
        )
        .unwrap();
        ledger.record(expiring, &evidence).unwrap();
        let evaluation = ledger.evaluate(&stakeholder(), &target, 13, policy());
        assert!(!evaluation.is_effective());
        assert!(evaluation.reasons.contains(&ConsentInvalidityReason::Expired));
    }

    #[test]
    fn consent_record_must_bind_real_evidence() {
        let evidence = evidence();
        let mut ledger = ConsentLedger::new();
        let bad = ConsentRecord::new(
            stakeholder(),
            scope("option-a"),
            ConsentState::Affirmed,
            ConsentProvenance::DirectStatement,
            [fact("missing")],
            [fact("capacity")],
            [fact("coercion")],
            0.05,
            10,
            None,
        )
        .unwrap();
        assert!(matches!(
            ledger.record(bad, &evidence),
            Err(ConsentError::MissingFact(_))
        ));
    }
}
