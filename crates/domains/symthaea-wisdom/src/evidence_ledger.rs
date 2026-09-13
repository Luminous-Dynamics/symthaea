// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bearing factual and normative ledgers for Wisdom & Care deliberation.
//!
//! WCARE-03 makes a type-level distinction between factual premises and normative
//! premises. A factual claim cannot be passed where a normative claim is expected,
//! and confidence in one domain is never used as confidence in the other.
//!
//! The ledger also preserves contradictions, supersession, provenance, and the
//! dependency from consequential decisions back to the premises they relied on.
//! Contradicted or superseded premises mark dependent decisions for review; they
//! are never silently rewritten to a replacement premise.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FactClaimId(String);

impl FactClaimId {
    pub fn new(value: impl Into<String>) -> Result<Self, EvidenceLedgerError> {
        let value = value.into();
        validate_identifier(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NormativeClaimId(String);

impl NormativeClaimId {
    pub fn new(value: impl Into<String>) -> Result<Self, EvidenceLedgerError> {
        let value = value.into();
        validate_identifier(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DecisionId(String);

impl DecisionId {
    pub fn new(value: impl Into<String>) -> Result<Self, EvidenceLedgerError> {
        let value = value.into();
        validate_identifier(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceRelation {
    Supports,
    Contradicts,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceObservation {
    pub source_ref: String,
    pub relation: EvidenceRelation,
    pub confidence: f32,
}

impl EvidenceObservation {
    pub fn new(
        source_ref: impl Into<String>,
        relation: EvidenceRelation,
        confidence: f32,
    ) -> Result<Self, EvidenceLedgerError> {
        let source_ref = source_ref.into();
        validate_source_ref(&source_ref)?;
        validate_confidence(confidence)?;
        Ok(Self {
            source_ref,
            relation,
            confidence,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactClaim {
    pub id: FactClaimId,
    pub proposition: String,
    pub confidence: f32,
    pub evidence: Vec<EvidenceObservation>,
    pub superseded_by: Option<FactClaimId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormativeClaim {
    pub id: NormativeClaimId,
    pub proposition: String,
    pub confidence: f32,
    pub evidence: Vec<EvidenceObservation>,
    pub superseded_by: Option<NormativeClaimId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactConflict {
    pub left: FactClaimId,
    pub right: FactClaimId,
    pub source_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormativeConflict {
    pub left: NormativeClaimId,
    pub right: NormativeClaimId,
    pub source_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DecisionReviewReason {
    SupersededFact(FactClaimId),
    SupersededNormativeClaim(NormativeClaimId),
    ContradictedFact(FactClaimId),
    ContradictedNormativeClaim(NormativeClaimId),
    UnresolvedFactConflict(FactClaimId, FactClaimId),
    UnresolvedNormativeConflict(NormativeClaimId, NormativeClaimId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionRecord {
    pub id: DecisionId,
    pub fact_premises: BTreeSet<FactClaimId>,
    pub normative_premises: BTreeSet<NormativeClaimId>,
    review_reasons: BTreeSet<DecisionReviewReason>,
}

impl DecisionRecord {
    pub fn requires_review(&self) -> bool {
        !self.review_reasons.is_empty()
    }

    pub fn review_reasons(&self) -> &BTreeSet<DecisionReviewReason> {
        &self.review_reasons
    }
}

#[derive(Debug, Clone, Default)]
pub struct FactLedger {
    claims: BTreeMap<FactClaimId, FactClaim>,
    conflicts: BTreeMap<(FactClaimId, FactClaimId), BTreeSet<String>>,
}

impl FactLedger {
    pub fn get(&self, id: &FactClaimId) -> Option<&FactClaim> {
        self.claims.get(id)
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    pub fn conflicts(&self) -> Vec<FactConflict> {
        self.conflicts
            .iter()
            .map(|((left, right), sources)| FactConflict {
                left: left.clone(),
                right: right.clone(),
                source_refs: sources.iter().cloned().collect(),
            })
            .collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct NormativeLedger {
    claims: BTreeMap<NormativeClaimId, NormativeClaim>,
    conflicts: BTreeMap<(NormativeClaimId, NormativeClaimId), BTreeSet<String>>,
}

impl NormativeLedger {
    pub fn get(&self, id: &NormativeClaimId) -> Option<&NormativeClaim> {
        self.claims.get(id)
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    pub fn conflicts(&self) -> Vec<NormativeConflict> {
        self.conflicts
            .iter()
            .map(|((left, right), sources)| NormativeConflict {
                left: left.clone(),
                right: right.clone(),
                source_refs: sources.iter().cloned().collect(),
            })
            .collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct DecisionDependencyLedger {
    decisions: BTreeMap<DecisionId, DecisionRecord>,
}

impl DecisionDependencyLedger {
    pub fn get(&self, id: &DecisionId) -> Option<&DecisionRecord> {
        self.decisions.get(id)
    }

    pub fn decisions_requiring_review(&self) -> Vec<&DecisionRecord> {
        self.decisions
            .values()
            .filter(|decision| decision.requires_review())
            .collect()
    }
}

/// Combined deliberation ledger with typed factual and normative stores.
///
/// Mutable access to the sub-ledgers is intentionally not exposed: mutations
/// must pass through this type so dependent decisions can be invalidated.
#[derive(Debug, Clone, Default)]
pub struct DeliberationEvidenceLedger {
    facts: FactLedger,
    normative: NormativeLedger,
    decisions: DecisionDependencyLedger,
}

impl DeliberationEvidenceLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn facts(&self) -> &FactLedger {
        &self.facts
    }

    pub fn normative(&self) -> &NormativeLedger {
        &self.normative
    }

    pub fn decisions(&self) -> &DecisionDependencyLedger {
        &self.decisions
    }

    pub fn add_fact(
        &mut self,
        id: FactClaimId,
        proposition: impl Into<String>,
        confidence: f32,
    ) -> Result<(), EvidenceLedgerError> {
        validate_confidence(confidence)?;
        let proposition = proposition.into();
        validate_proposition(&proposition)?;
        if self.facts.claims.contains_key(&id) {
            return Err(EvidenceLedgerError::DuplicateFact(id));
        }
        self.facts.claims.insert(
            id.clone(),
            FactClaim {
                id,
                proposition,
                confidence,
                evidence: Vec::new(),
                superseded_by: None,
            },
        );
        Ok(())
    }

    pub fn add_normative_claim(
        &mut self,
        id: NormativeClaimId,
        proposition: impl Into<String>,
        confidence: f32,
    ) -> Result<(), EvidenceLedgerError> {
        validate_confidence(confidence)?;
        let proposition = proposition.into();
        validate_proposition(&proposition)?;
        if self.normative.claims.contains_key(&id) {
            return Err(EvidenceLedgerError::DuplicateNormativeClaim(id));
        }
        self.normative.claims.insert(
            id.clone(),
            NormativeClaim {
                id,
                proposition,
                confidence,
                evidence: Vec::new(),
                superseded_by: None,
            },
        );
        Ok(())
    }

    pub fn add_fact_evidence(
        &mut self,
        id: &FactClaimId,
        observation: EvidenceObservation,
    ) -> Result<(), EvidenceLedgerError> {
        let relation = observation.relation;
        let claim = self
            .facts
            .claims
            .get_mut(id)
            .ok_or_else(|| EvidenceLedgerError::MissingFact(id.clone()))?;
        claim.evidence.push(observation);
        if relation == EvidenceRelation::Contradicts {
            self.mark_decisions_for_fact(
                id,
                DecisionReviewReason::ContradictedFact(id.clone()),
            );
        }
        Ok(())
    }

    pub fn add_normative_evidence(
        &mut self,
        id: &NormativeClaimId,
        observation: EvidenceObservation,
    ) -> Result<(), EvidenceLedgerError> {
        let relation = observation.relation;
        let claim = self
            .normative
            .claims
            .get_mut(id)
            .ok_or_else(|| EvidenceLedgerError::MissingNormativeClaim(id.clone()))?;
        claim.evidence.push(observation);
        if relation == EvidenceRelation::Contradicts {
            self.mark_decisions_for_normative(
                id,
                DecisionReviewReason::ContradictedNormativeClaim(id.clone()),
            );
        }
        Ok(())
    }

    pub fn supersede_fact(
        &mut self,
        old: &FactClaimId,
        replacement: &FactClaimId,
    ) -> Result<(), EvidenceLedgerError> {
        if old == replacement {
            return Err(EvidenceLedgerError::SelfSupersession);
        }
        if !self.facts.claims.contains_key(replacement) {
            return Err(EvidenceLedgerError::MissingFact(replacement.clone()));
        }
        let old_claim = self
            .facts
            .claims
            .get_mut(old)
            .ok_or_else(|| EvidenceLedgerError::MissingFact(old.clone()))?;
        if old_claim.superseded_by.is_some() {
            return Err(EvidenceLedgerError::AlreadySupersededFact(old.clone()));
        }
        old_claim.superseded_by = Some(replacement.clone());
        self.mark_decisions_for_fact(old, DecisionReviewReason::SupersededFact(old.clone()));
        Ok(())
    }

    pub fn supersede_normative_claim(
        &mut self,
        old: &NormativeClaimId,
        replacement: &NormativeClaimId,
    ) -> Result<(), EvidenceLedgerError> {
        if old == replacement {
            return Err(EvidenceLedgerError::SelfSupersession);
        }
        if !self.normative.claims.contains_key(replacement) {
            return Err(EvidenceLedgerError::MissingNormativeClaim(
                replacement.clone(),
            ));
        }
        let old_claim = self
            .normative
            .claims
            .get_mut(old)
            .ok_or_else(|| EvidenceLedgerError::MissingNormativeClaim(old.clone()))?;
        if old_claim.superseded_by.is_some() {
            return Err(EvidenceLedgerError::AlreadySupersededNormativeClaim(
                old.clone(),
            ));
        }
        old_claim.superseded_by = Some(replacement.clone());
        self.mark_decisions_for_normative(
            old,
            DecisionReviewReason::SupersededNormativeClaim(old.clone()),
        );
        Ok(())
    }

    pub fn record_fact_conflict(
        &mut self,
        a: &FactClaimId,
        b: &FactClaimId,
        source_ref: impl Into<String>,
    ) -> Result<(), EvidenceLedgerError> {
        if a == b {
            return Err(EvidenceLedgerError::SelfConflict);
        }
        if !self.facts.claims.contains_key(a) {
            return Err(EvidenceLedgerError::MissingFact(a.clone()));
        }
        if !self.facts.claims.contains_key(b) {
            return Err(EvidenceLedgerError::MissingFact(b.clone()));
        }
        let source_ref = source_ref.into();
        validate_source_ref(&source_ref)?;
        let (left, right) = ordered_fact_pair(a.clone(), b.clone());
        self.facts
            .conflicts
            .entry((left.clone(), right.clone()))
            .or_default()
            .insert(source_ref);
        let reason = DecisionReviewReason::UnresolvedFactConflict(left, right);
        self.mark_decisions_for_fact(a, reason.clone());
        self.mark_decisions_for_fact(b, reason);
        Ok(())
    }

    pub fn record_normative_conflict(
        &mut self,
        a: &NormativeClaimId,
        b: &NormativeClaimId,
        source_ref: impl Into<String>,
    ) -> Result<(), EvidenceLedgerError> {
        if a == b {
            return Err(EvidenceLedgerError::SelfConflict);
        }
        if !self.normative.claims.contains_key(a) {
            return Err(EvidenceLedgerError::MissingNormativeClaim(a.clone()));
        }
        if !self.normative.claims.contains_key(b) {
            return Err(EvidenceLedgerError::MissingNormativeClaim(b.clone()));
        }
        let source_ref = source_ref.into();
        validate_source_ref(&source_ref)?;
        let (left, right) = ordered_normative_pair(a.clone(), b.clone());
        self.normative
            .conflicts
            .entry((left.clone(), right.clone()))
            .or_default()
            .insert(source_ref);
        let reason = DecisionReviewReason::UnresolvedNormativeConflict(left, right);
        self.mark_decisions_for_normative(a, reason.clone());
        self.mark_decisions_for_normative(b, reason);
        Ok(())
    }

    pub fn register_decision(
        &mut self,
        id: DecisionId,
        fact_premises: impl IntoIterator<Item = FactClaimId>,
        normative_premises: impl IntoIterator<Item = NormativeClaimId>,
    ) -> Result<(), EvidenceLedgerError> {
        if self.decisions.decisions.contains_key(&id) {
            return Err(EvidenceLedgerError::DuplicateDecision(id));
        }
        let fact_premises: BTreeSet<_> = fact_premises.into_iter().collect();
        let normative_premises: BTreeSet<_> = normative_premises.into_iter().collect();
        for fact in &fact_premises {
            if !self.facts.claims.contains_key(fact) {
                return Err(EvidenceLedgerError::MissingFact(fact.clone()));
            }
        }
        for claim in &normative_premises {
            if !self.normative.claims.contains_key(claim) {
                return Err(EvidenceLedgerError::MissingNormativeClaim(claim.clone()));
            }
        }

        let mut review_reasons = BTreeSet::new();
        for fact in &fact_premises {
            let claim = &self.facts.claims[fact];
            if claim.superseded_by.is_some() {
                review_reasons.insert(DecisionReviewReason::SupersededFact(fact.clone()));
            }
            if claim
                .evidence
                .iter()
                .any(|evidence| evidence.relation == EvidenceRelation::Contradicts)
            {
                review_reasons.insert(DecisionReviewReason::ContradictedFact(fact.clone()));
            }
        }
        for claim_id in &normative_premises {
            let claim = &self.normative.claims[claim_id];
            if claim.superseded_by.is_some() {
                review_reasons.insert(DecisionReviewReason::SupersededNormativeClaim(
                    claim_id.clone(),
                ));
            }
            if claim
                .evidence
                .iter()
                .any(|evidence| evidence.relation == EvidenceRelation::Contradicts)
            {
                review_reasons.insert(DecisionReviewReason::ContradictedNormativeClaim(
                    claim_id.clone(),
                ));
            }
        }
        for ((left, right), _) in &self.facts.conflicts {
            if fact_premises.contains(left) || fact_premises.contains(right) {
                review_reasons.insert(DecisionReviewReason::UnresolvedFactConflict(
                    left.clone(),
                    right.clone(),
                ));
            }
        }
        for ((left, right), _) in &self.normative.conflicts {
            if normative_premises.contains(left) || normative_premises.contains(right) {
                review_reasons.insert(DecisionReviewReason::UnresolvedNormativeConflict(
                    left.clone(),
                    right.clone(),
                ));
            }
        }

        self.decisions.decisions.insert(
            id.clone(),
            DecisionRecord {
                id,
                fact_premises,
                normative_premises,
                review_reasons,
            },
        );
        Ok(())
    }

    fn mark_decisions_for_fact(&mut self, id: &FactClaimId, reason: DecisionReviewReason) {
        for decision in self.decisions.decisions.values_mut() {
            if decision.fact_premises.contains(id) {
                decision.review_reasons.insert(reason.clone());
            }
        }
    }

    fn mark_decisions_for_normative(
        &mut self,
        id: &NormativeClaimId,
        reason: DecisionReviewReason,
    ) {
        for decision in self.decisions.decisions.values_mut() {
            if decision.normative_premises.contains(id) {
                decision.review_reasons.insert(reason.clone());
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceLedgerError {
    EmptyIdentifier,
    EmptyProposition,
    EmptySourceRef,
    InvalidConfidence(f32),
    DuplicateFact(FactClaimId),
    DuplicateNormativeClaim(NormativeClaimId),
    DuplicateDecision(DecisionId),
    MissingFact(FactClaimId),
    MissingNormativeClaim(NormativeClaimId),
    AlreadySupersededFact(FactClaimId),
    AlreadySupersededNormativeClaim(NormativeClaimId),
    SelfSupersession,
    SelfConflict,
}

fn validate_identifier(value: &str) -> Result<(), EvidenceLedgerError> {
    if value.trim().is_empty() {
        Err(EvidenceLedgerError::EmptyIdentifier)
    } else {
        Ok(())
    }
}

fn validate_proposition(value: &str) -> Result<(), EvidenceLedgerError> {
    if value.trim().is_empty() {
        Err(EvidenceLedgerError::EmptyProposition)
    } else {
        Ok(())
    }
}

fn validate_source_ref(value: &str) -> Result<(), EvidenceLedgerError> {
    if value.trim().is_empty() {
        Err(EvidenceLedgerError::EmptySourceRef)
    } else {
        Ok(())
    }
}

fn validate_confidence(value: f32) -> Result<(), EvidenceLedgerError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(EvidenceLedgerError::InvalidConfidence(value))
    }
}

fn ordered_fact_pair(a: FactClaimId, b: FactClaimId) -> (FactClaimId, FactClaimId) {
    if a <= b { (a, b) } else { (b, a) }
}

fn ordered_normative_pair(
    a: NormativeClaimId,
    b: NormativeClaimId,
) -> (NormativeClaimId, NormativeClaimId) {
    if a <= b { (a, b) } else { (b, a) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(id: &str) -> FactClaimId {
        FactClaimId::new(id).unwrap()
    }

    fn norm(id: &str) -> NormativeClaimId {
        NormativeClaimId::new(id).unwrap()
    }

    fn decision(id: &str) -> DecisionId {
        DecisionId::new(id).unwrap()
    }

    #[test]
    fn rejects_invalid_confidence_instead_of_hiding_it_by_clamping() {
        let mut ledger = DeliberationEvidenceLedger::new();
        assert!(matches!(
            ledger.add_fact(fact("f1"), "the bridge is open", f32::NAN),
            Err(EvidenceLedgerError::InvalidConfidence(_))
        ));
        assert!(matches!(
            ledger.add_normative_claim(norm("n1"), "crossing is justified", 1.1),
            Err(EvidenceLedgerError::InvalidConfidence(_))
        ));
    }

    #[test]
    fn fact_and_normative_confidence_remain_independent() {
        let mut ledger = DeliberationEvidenceLedger::new();
        let f1 = fact("f1");
        let n1 = norm("n1");
        ledger.add_fact(f1.clone(), "the medication is available", 0.99).unwrap();
        ledger
            .add_normative_claim(n1.clone(), "we ought to recommend it", 0.35)
            .unwrap();

        assert_eq!(ledger.facts().get(&f1).unwrap().confidence, 0.99);
        assert_eq!(ledger.normative().get(&n1).unwrap().confidence, 0.35);
    }

    #[test]
    fn contradictory_fact_evidence_marks_dependent_decision_for_review() {
        let mut ledger = DeliberationEvidenceLedger::new();
        let f1 = fact("f1");
        let d1 = decision("d1");
        ledger.add_fact(f1.clone(), "route A is safe", 0.8).unwrap();
        ledger
            .register_decision(d1.clone(), [f1.clone()], [])
            .unwrap();
        assert!(!ledger.decisions().get(&d1).unwrap().requires_review());

        ledger
            .add_fact_evidence(
                &f1,
                EvidenceObservation::new("sensor-2", EvidenceRelation::Contradicts, 0.9).unwrap(),
            )
            .unwrap();

        let record = ledger.decisions().get(&d1).unwrap();
        assert!(record.requires_review());
        assert!(record
            .review_reasons()
            .contains(&DecisionReviewReason::ContradictedFact(f1)));
    }

    #[test]
    fn superseded_premise_does_not_silently_rewrite_decision() {
        let mut ledger = DeliberationEvidenceLedger::new();
        let old = fact("old");
        let replacement = fact("new");
        let d1 = decision("d1");
        ledger.add_fact(old.clone(), "old estimate", 0.7).unwrap();
        ledger
            .add_fact(replacement.clone(), "new estimate", 0.9)
            .unwrap();
        ledger
            .register_decision(d1.clone(), [old.clone()], [])
            .unwrap();

        ledger.supersede_fact(&old, &replacement).unwrap();

        let record = ledger.decisions().get(&d1).unwrap();
        assert!(record.fact_premises.contains(&old));
        assert!(!record.fact_premises.contains(&replacement));
        assert!(record
            .review_reasons()
            .contains(&DecisionReviewReason::SupersededFact(old)));
    }

    #[test]
    fn normative_disagreement_is_preserved_not_averaged_away() {
        let mut ledger = DeliberationEvidenceLedger::new();
        let autonomy = norm("autonomy");
        let beneficence = norm("beneficence");
        ledger
            .add_normative_claim(autonomy.clone(), "respect the refusal", 0.8)
            .unwrap();
        ledger
            .add_normative_claim(beneficence.clone(), "intervene for welfare", 0.8)
            .unwrap();
        ledger
            .record_normative_conflict(&autonomy, &beneficence, "ethics-review-1")
            .unwrap();

        let conflicts = ledger.normative().conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].source_refs, vec!["ethics-review-1".to_string()]);
    }

    #[test]
    fn decision_registered_after_existing_conflict_starts_review_required() {
        let mut ledger = DeliberationEvidenceLedger::new();
        let n1 = norm("n1");
        let n2 = norm("n2");
        let d1 = decision("d1");
        ledger.add_normative_claim(n1.clone(), "value A", 0.7).unwrap();
        ledger.add_normative_claim(n2.clone(), "value B", 0.7).unwrap();
        ledger
            .record_normative_conflict(&n1, &n2, "review-board")
            .unwrap();
        ledger
            .register_decision(d1.clone(), [], [n1.clone()])
            .unwrap();

        let record = ledger.decisions().get(&d1).unwrap();
        assert!(record.requires_review());
        assert!(record.review_reasons().contains(
            &DecisionReviewReason::UnresolvedNormativeConflict(n1.min(n2.clone()), n2.max(n1))
        ));
    }

    #[test]
    fn conflict_sources_are_deduplicated_without_erasing_conflict() {
        let mut ledger = DeliberationEvidenceLedger::new();
        let f1 = fact("a");
        let f2 = fact("b");
        ledger.add_fact(f1.clone(), "claim A", 0.6).unwrap();
        ledger.add_fact(f2.clone(), "claim B", 0.6).unwrap();
        ledger
            .record_fact_conflict(&f1, &f2, "source-1")
            .unwrap();
        ledger
            .record_fact_conflict(&f2, &f1, "source-1")
            .unwrap();

        let conflicts = ledger.facts().conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].source_refs.len(), 1);
    }
}
