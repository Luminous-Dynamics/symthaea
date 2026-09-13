// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed factual and normative evidence ledgers for Wisdom & Care deliberation.
//!
//! Facts and normative judgments deliberately use different ID types and stores.
//! Confidence in a factual premise therefore cannot silently stand in for moral
//! confidence. Contradictions and supersession remain visible, and decisions are
//! bound to the exact premises they used rather than being silently rewritten.

use std::collections::{BTreeMap, BTreeSet};

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, EvidenceLedgerError> {
                let value = value.into();
                if value.trim().is_empty() {
                    return Err(EvidenceLedgerError::EmptyIdentifier);
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}

id_type!(FactClaimId);
id_type!(NormativeClaimId);
id_type!(DecisionId);

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
pub struct Claim<I> {
    pub id: I,
    pub proposition: String,
    pub confidence: f32,
    pub evidence: Vec<EvidenceObservation>,
    pub superseded_by: Option<I>,
}

pub type FactClaim = Claim<FactClaimId>;
pub type NormativeClaim = Claim<NormativeClaimId>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conflict<I> {
    pub left: I,
    pub right: I,
    pub source_refs: Vec<String>,
}

pub type FactConflict = Conflict<FactClaimId>;
pub type NormativeConflict = Conflict<NormativeClaimId>;

#[derive(Debug, Clone)]
struct TypedLedger<I: Ord> {
    claims: BTreeMap<I, Claim<I>>,
    conflicts: BTreeMap<(I, I), BTreeSet<String>>,
}

impl<I: Ord> Default for TypedLedger<I> {
    fn default() -> Self {
        Self {
            claims: BTreeMap::new(),
            conflicts: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FactLedger {
    inner: TypedLedger<FactClaimId>,
}

impl FactLedger {
    pub fn get(&self, id: &FactClaimId) -> Option<&FactClaim> {
        self.inner.claims.get(id)
    }

    pub fn len(&self) -> usize {
        self.inner.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.claims.is_empty()
    }

    pub fn conflicts(&self) -> Vec<FactConflict> {
        conflicts_from(&self.inner)
    }
}

#[derive(Debug, Clone, Default)]
pub struct NormativeLedger {
    inner: TypedLedger<NormativeClaimId>,
}

impl NormativeLedger {
    pub fn get(&self, id: &NormativeClaimId) -> Option<&NormativeClaim> {
        self.inner.claims.get(id)
    }

    pub fn len(&self) -> usize {
        self.inner.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.claims.is_empty()
    }

    pub fn conflicts(&self) -> Vec<NormativeConflict> {
        conflicts_from(&self.inner)
    }
}

fn conflicts_from<I: Ord + Clone>(ledger: &TypedLedger<I>) -> Vec<Conflict<I>> {
    ledger
        .conflicts
        .iter()
        .map(|((left, right), sources)| Conflict {
            left: left.clone(),
            right: right.clone(),
            source_refs: sources.iter().cloned().collect(),
        })
        .collect()
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

/// Mutation must pass through this aggregate so premise changes invalidate the
/// exact decisions that depended on them. The sub-ledgers are exposed read-only.
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
        let proposition = proposition.into();
        validate_claim(&proposition, confidence)?;
        if self.facts.inner.claims.contains_key(&id) {
            return Err(EvidenceLedgerError::DuplicateFact(id));
        }
        self.facts.inner.claims.insert(
            id.clone(),
            Claim {
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
        let proposition = proposition.into();
        validate_claim(&proposition, confidence)?;
        if self.normative.inner.claims.contains_key(&id) {
            return Err(EvidenceLedgerError::DuplicateNormativeClaim(id));
        }
        self.normative.inner.claims.insert(
            id.clone(),
            Claim {
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
        self.facts
            .inner
            .claims
            .get_mut(id)
            .ok_or_else(|| EvidenceLedgerError::MissingFact(id.clone()))?
            .evidence
            .push(observation);
        if relation == EvidenceRelation::Contradicts {
            self.mark_fact(id, DecisionReviewReason::ContradictedFact(id.clone()));
        }
        Ok(())
    }

    pub fn add_normative_evidence(
        &mut self,
        id: &NormativeClaimId,
        observation: EvidenceObservation,
    ) -> Result<(), EvidenceLedgerError> {
        let relation = observation.relation;
        self.normative
            .inner
            .claims
            .get_mut(id)
            .ok_or_else(|| EvidenceLedgerError::MissingNormativeClaim(id.clone()))?
            .evidence
            .push(observation);
        if relation == EvidenceRelation::Contradicts {
            self.mark_normative(
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
        if !self.facts.inner.claims.contains_key(replacement) {
            return Err(EvidenceLedgerError::MissingFact(replacement.clone()));
        }
        let old_claim = self
            .facts
            .inner
            .claims
            .get_mut(old)
            .ok_or_else(|| EvidenceLedgerError::MissingFact(old.clone()))?;
        if old_claim.superseded_by.is_some() {
            return Err(EvidenceLedgerError::AlreadySupersededFact(old.clone()));
        }
        old_claim.superseded_by = Some(replacement.clone());
        self.mark_fact(old, DecisionReviewReason::SupersededFact(old.clone()));
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
        if !self.normative.inner.claims.contains_key(replacement) {
            return Err(EvidenceLedgerError::MissingNormativeClaim(replacement.clone()));
        }
        let old_claim = self
            .normative
            .inner
            .claims
            .get_mut(old)
            .ok_or_else(|| EvidenceLedgerError::MissingNormativeClaim(old.clone()))?;
        if old_claim.superseded_by.is_some() {
            return Err(EvidenceLedgerError::AlreadySupersededNormativeClaim(
                old.clone(),
            ));
        }
        old_claim.superseded_by = Some(replacement.clone());
        self.mark_normative(
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
        require_two_claims(&self.facts.inner, a, b, |id| {
            EvidenceLedgerError::MissingFact(id)
        })?;
        let source_ref = source_ref.into();
        validate_source_ref(&source_ref)?;
        let (left, right) = ordered_pair(a.clone(), b.clone())?;
        self.facts
            .inner
            .conflicts
            .entry((left.clone(), right.clone()))
            .or_default()
            .insert(source_ref);
        let reason = DecisionReviewReason::UnresolvedFactConflict(left, right);
        self.mark_fact(a, reason.clone());
        self.mark_fact(b, reason);
        Ok(())
    }

    pub fn record_normative_conflict(
        &mut self,
        a: &NormativeClaimId,
        b: &NormativeClaimId,
        source_ref: impl Into<String>,
    ) -> Result<(), EvidenceLedgerError> {
        require_two_claims(&self.normative.inner, a, b, |id| {
            EvidenceLedgerError::MissingNormativeClaim(id)
        })?;
        let source_ref = source_ref.into();
        validate_source_ref(&source_ref)?;
        let (left, right) = ordered_pair(a.clone(), b.clone())?;
        self.normative
            .inner
            .conflicts
            .entry((left.clone(), right.clone()))
            .or_default()
            .insert(source_ref);
        let reason = DecisionReviewReason::UnresolvedNormativeConflict(left, right);
        self.mark_normative(a, reason.clone());
        self.mark_normative(b, reason);
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

        for premise in &fact_premises {
            if !self.facts.inner.claims.contains_key(premise) {
                return Err(EvidenceLedgerError::MissingFact(premise.clone()));
            }
        }
        for premise in &normative_premises {
            if !self.normative.inner.claims.contains_key(premise) {
                return Err(EvidenceLedgerError::MissingNormativeClaim(premise.clone()));
            }
        }

        let mut review_reasons = BTreeSet::new();
        collect_fact_reasons(&self.facts.inner, &fact_premises, &mut review_reasons);
        collect_normative_reasons(
            &self.normative.inner,
            &normative_premises,
            &mut review_reasons,
        );

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

    fn mark_fact(&mut self, id: &FactClaimId, reason: DecisionReviewReason) {
        for decision in self.decisions.decisions.values_mut() {
            if decision.fact_premises.contains(id) {
                decision.review_reasons.insert(reason.clone());
            }
        }
    }

    fn mark_normative(&mut self, id: &NormativeClaimId, reason: DecisionReviewReason) {
        for decision in self.decisions.decisions.values_mut() {
            if decision.normative_premises.contains(id) {
                decision.review_reasons.insert(reason.clone());
            }
        }
    }
}

fn collect_fact_reasons(
    ledger: &TypedLedger<FactClaimId>,
    premises: &BTreeSet<FactClaimId>,
    reasons: &mut BTreeSet<DecisionReviewReason>,
) {
    for premise in premises {
        let claim = ledger.claims.get(premise).expect("premise validated");
        if claim.superseded_by.is_some() {
            reasons.insert(DecisionReviewReason::SupersededFact(premise.clone()));
        }
        if has_contradicting_evidence(claim) {
            reasons.insert(DecisionReviewReason::ContradictedFact(premise.clone()));
        }
    }
    for (left, right) in ledger.conflicts.keys() {
        if premises.contains(left) || premises.contains(right) {
            reasons.insert(DecisionReviewReason::UnresolvedFactConflict(
                left.clone(),
                right.clone(),
            ));
        }
    }
}

fn collect_normative_reasons(
    ledger: &TypedLedger<NormativeClaimId>,
    premises: &BTreeSet<NormativeClaimId>,
    reasons: &mut BTreeSet<DecisionReviewReason>,
) {
    for premise in premises {
        let claim = ledger.claims.get(premise).expect("premise validated");
        if claim.superseded_by.is_some() {
            reasons.insert(DecisionReviewReason::SupersededNormativeClaim(
                premise.clone(),
            ));
        }
        if has_contradicting_evidence(claim) {
            reasons.insert(DecisionReviewReason::ContradictedNormativeClaim(
                premise.clone(),
            ));
        }
    }
    for (left, right) in ledger.conflicts.keys() {
        if premises.contains(left) || premises.contains(right) {
            reasons.insert(DecisionReviewReason::UnresolvedNormativeConflict(
                left.clone(),
                right.clone(),
            ));
        }
    }
}

fn has_contradicting_evidence<I>(claim: &Claim<I>) -> bool {
    claim
        .evidence
        .iter()
        .any(|evidence| evidence.relation == EvidenceRelation::Contradicts)
}

fn require_two_claims<I: Ord + Clone>(
    ledger: &TypedLedger<I>,
    a: &I,
    b: &I,
    missing: impl Fn(I) -> EvidenceLedgerError,
) -> Result<(), EvidenceLedgerError> {
    if a == b {
        return Err(EvidenceLedgerError::SelfConflict);
    }
    if !ledger.claims.contains_key(a) {
        return Err(missing(a.clone()));
    }
    if !ledger.claims.contains_key(b) {
        return Err(missing(b.clone()));
    }
    Ok(())
}

fn ordered_pair<I: Ord>(a: I, b: I) -> Result<(I, I), EvidenceLedgerError> {
    if a == b {
        Err(EvidenceLedgerError::SelfConflict)
    } else if a < b {
        Ok((a, b))
    } else {
        Ok((b, a))
    }
}

fn validate_claim(proposition: &str, confidence: f32) -> Result<(), EvidenceLedgerError> {
    if proposition.trim().is_empty() {
        return Err(EvidenceLedgerError::EmptyProposition);
    }
    validate_confidence(confidence)
}

fn validate_source_ref(source_ref: &str) -> Result<(), EvidenceLedgerError> {
    if source_ref.trim().is_empty() {
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
    fn invalid_confidence_is_rejected_not_clamped() {
        let mut ledger = DeliberationEvidenceLedger::new();
        assert!(matches!(
            ledger.add_fact(fact("f1"), "bridge is open", f32::NAN),
            Err(EvidenceLedgerError::InvalidConfidence(_))
        ));
        assert!(matches!(
            ledger.add_normative_claim(norm("n1"), "crossing is justified", 1.1),
            Err(EvidenceLedgerError::InvalidConfidence(_))
        ));
    }

    #[test]
    fn factual_and_normative_confidence_are_independent() {
        let mut ledger = DeliberationEvidenceLedger::new();
        let f1 = fact("f1");
        let n1 = norm("n1");
        ledger
            .add_fact(f1.clone(), "medicine is available", 0.99)
            .unwrap();
        ledger
            .add_normative_claim(n1.clone(), "we ought to recommend it", 0.35)
            .unwrap();
        assert_eq!(ledger.facts().get(&f1).unwrap().confidence, 0.99);
        assert_eq!(ledger.normative().get(&n1).unwrap().confidence, 0.35);
    }

    #[test]
    fn contradiction_invalidates_dependent_decision() {
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
        assert!(ledger
            .decisions()
            .get(&d1)
            .unwrap()
            .review_reasons()
            .contains(&DecisionReviewReason::ContradictedFact(f1)));
    }

    #[test]
    fn supersession_marks_review_without_rewriting_premise() {
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
    fn normative_disagreement_remains_explicit() {
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
    fn decision_registered_after_conflict_starts_review_required() {
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
        assert!(ledger.decisions().get(&d1).unwrap().requires_review());
    }

    #[test]
    fn conflict_provenance_is_deduplicated_without_erasing_conflict() {
        let mut ledger = DeliberationEvidenceLedger::new();
        let f1 = fact("a");
        let f2 = fact("b");
        ledger.add_fact(f1.clone(), "claim A", 0.6).unwrap();
        ledger.add_fact(f2.clone(), "claim B", 0.6).unwrap();
        ledger.record_fact_conflict(&f1, &f2, "source-1").unwrap();
        ledger.record_fact_conflict(&f2, &f1, "source-1").unwrap();
        let conflicts = ledger.facts().conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].source_refs.len(), 1);
    }
}
