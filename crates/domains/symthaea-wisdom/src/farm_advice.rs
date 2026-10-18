// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Farm-advice profile for the existing WCARE-08 consequence/accountability ledger.
//!
//! This module stores farm-specific context and references only. Consequence
//! predictions, observed outcomes, premise dependencies, repairs and retractions
//! remain owned by WCARE-08 and the canonical deliberation evidence ledger.
//! Raw farm coordinates and private identifiers do not belong in these records.

use std::collections::BTreeSet;

use crate::accountability::AccountabilityCaseId;
use crate::evidence_ledger::{DecisionId, DeliberationEvidenceLedger, FactClaimId};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AdviceEpisodeId(String);

impl AdviceEpisodeId {
    pub fn new(value: impl Into<String>) -> Result<Self, FarmAdviceError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(FarmAdviceError::EmptyIdentifier);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FarmOperation {
    CropProduction,
    Irrigation,
    Livestock,
    SoilWater,
    PestRisk,
    Harvest,
    Mechanization,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsefulWindow {
    starts_at_epoch_seconds: i64,
    ends_at_epoch_seconds: i64,
}

impl UsefulWindow {
    pub fn new(starts_at_epoch_seconds: i64, ends_at_epoch_seconds: i64) -> Result<Self, FarmAdviceError> {
        if starts_at_epoch_seconds < 0 || ends_at_epoch_seconds <= starts_at_epoch_seconds {
            return Err(FarmAdviceError::InvalidUsefulWindow);
        }
        Ok(Self {
            starts_at_epoch_seconds,
            ends_at_epoch_seconds,
        })
    }

    pub fn starts_at_epoch_seconds(&self) -> i64 {
        self.starts_at_epoch_seconds
    }

    pub fn ends_at_epoch_seconds(&self) -> i64 {
        self.ends_at_epoch_seconds
    }
}

/// Opaque, purpose-bound site/asset reference. Never store coordinates here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FarmAdviceContext {
    site_ref: String,
    operation: FarmOperation,
    season_ref: Option<String>,
    useful_window: Option<UsefulWindow>,
}

impl FarmAdviceContext {
    pub fn new(
        site_ref: impl Into<String>,
        operation: FarmOperation,
        season_ref: Option<String>,
        useful_window: Option<UsefulWindow>,
    ) -> Result<Self, FarmAdviceError> {
        let site_ref = site_ref.into();
        if site_ref.trim().is_empty() {
            return Err(FarmAdviceError::EmptySiteReference);
        }
        if season_ref.as_ref().is_some_and(|value| value.trim().is_empty()) {
            return Err(FarmAdviceError::EmptySeasonReference);
        }
        Ok(Self {
            site_ref,
            operation,
            season_ref,
            useful_window,
        })
    }

    pub fn site_ref(&self) -> &str {
        &self.site_ref
    }

    pub fn operation(&self) -> FarmOperation {
        self.operation
    }

    pub fn season_ref(&self) -> Option<&str> {
        self.season_ref.as_deref()
    }

    pub fn useful_window(&self) -> Option<UsefulWindow> {
        self.useful_window
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveSource {
    FarmerDeclared,
    AdviserDeclared,
    NotSupplied,
}

/// The stated objective is metadata, not a value inferred from farm size,
/// demographic traits, engagement history, or model-generated text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FarmerObjective {
    source: ObjectiveSource,
    summary: Option<String>,
    unit: Option<String>,
}

impl FarmerObjective {
    pub fn new(
        source: ObjectiveSource,
        summary: Option<String>,
        unit: Option<String>,
    ) -> Result<Self, FarmAdviceError> {
        match source {
            ObjectiveSource::NotSupplied if summary.is_some() || unit.is_some() => {
                return Err(FarmAdviceError::ObjectiveSourceMismatch);
            }
            ObjectiveSource::NotSupplied => {}
            ObjectiveSource::FarmerDeclared | ObjectiveSource::AdviserDeclared => {
                if summary.as_deref().is_none_or(|value| value.trim().is_empty()) {
                    return Err(FarmAdviceError::ObjectiveSummaryRequired);
                }
                if unit.as_ref().is_some_and(|value| value.trim().is_empty()) {
                    return Err(FarmAdviceError::EmptyObjectiveUnit);
                }
            }
        }
        Ok(Self { source, summary, unit })
    }

    pub fn source(&self) -> ObjectiveSource {
        self.source
    }

    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    pub fn unit(&self) -> Option<&str> {
        self.unit.as_deref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdviceEvidenceKind {
    Measured,
    Reported,
    Forecast,
    Modelled,
    Inferred,
    Stale,
    Conflicting,
    Unknown,
}

/// A premise projection references existing factual claims; it does not copy
/// their content into a second farm-advice ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvicePremise {
    kind: AdviceEvidenceKind,
    fact_claims: BTreeSet<FactClaimId>,
    as_of_epoch_seconds: Option<i64>,
}

impl AdvicePremise {
    pub fn new(
        kind: AdviceEvidenceKind,
        fact_claims: impl IntoIterator<Item = FactClaimId>,
        as_of_epoch_seconds: Option<i64>,
    ) -> Result<Self, FarmAdviceError> {
        let fact_claims: BTreeSet<_> = fact_claims.into_iter().collect();
        match kind {
            AdviceEvidenceKind::Unknown => {
                if !fact_claims.is_empty() || as_of_epoch_seconds.is_some() {
                    return Err(FarmAdviceError::UnknownMustRemainUnknown);
                }
            }
            AdviceEvidenceKind::Conflicting => {
                if fact_claims.len() < 2 {
                    return Err(FarmAdviceError::ConflictingPremisesNeedMultipleClaims);
                }
                validate_as_of(as_of_epoch_seconds)?;
            }
            _ => {
                if fact_claims.is_empty() {
                    return Err(FarmAdviceError::PremiseClaimRequired);
                }
                validate_as_of(as_of_epoch_seconds)?;
            }
        }
        Ok(Self {
            kind,
            fact_claims,
            as_of_epoch_seconds,
        })
    }

    pub fn kind(&self) -> AdviceEvidenceKind {
        self.kind
    }

    pub fn fact_claims(&self) -> &BTreeSet<FactClaimId> {
        &self.fact_claims
    }

    pub fn as_of_epoch_seconds(&self) -> Option<i64> {
        self.as_of_epoch_seconds
    }
}

fn validate_as_of(value: Option<i64>) -> Result<(), FarmAdviceError> {
    match value {
        Some(seconds) if seconds >= 0 => Ok(()),
        _ => Err(FarmAdviceError::PremiseAsOfRequired),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FarmerAdviceDisposition {
    AcceptedAsSuggested,
    ModifiedByFarmer,
    Deferred,
    Declined,
    ReferredToAdviser,
    UnableToAct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdviceReviewState {
    Active,
    RequiresReview,
    Superseded,
    Retracted,
}

/// Farm-specific context bound to the existing WCARE-08 accountability case.
///
/// Predicted and observed quantitative claims are stored as separate references
/// to the evidence ledger. Their values, source provenance, consequences and
/// repair obligations are not duplicated by this profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FarmAdviceProfile {
    episode_id: AdviceEpisodeId,
    accountability_case_id: AccountabilityCaseId,
    decision_id: DecisionId,
    context: FarmAdviceContext,
    objective: FarmerObjective,
    premises: Vec<AdvicePremise>,
    predicted_metric_claims: BTreeSet<FactClaimId>,
    observed_metric_claims: BTreeSet<FactClaimId>,
    disposition: Option<FarmerAdviceDisposition>,
    review_state: AdviceReviewState,
}

impl FarmAdviceProfile {
    pub fn new(
        episode_id: AdviceEpisodeId,
        accountability_case_id: AccountabilityCaseId,
        decision_id: DecisionId,
        context: FarmAdviceContext,
        objective: FarmerObjective,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<Self, FarmAdviceError> {
        if evidence.decisions().get(&decision_id).is_none() {
            return Err(FarmAdviceError::DecisionReferenceNotFound);
        }
        Ok(Self {
            episode_id,
            accountability_case_id,
            decision_id,
            context,
            objective,
            premises: Vec::new(),
            predicted_metric_claims: BTreeSet::new(),
            observed_metric_claims: BTreeSet::new(),
            disposition: None,
            review_state: AdviceReviewState::Active,
        })
    }

    pub fn episode_id(&self) -> &AdviceEpisodeId {
        &self.episode_id
    }

    pub fn accountability_case_id(&self) -> &AccountabilityCaseId {
        &self.accountability_case_id
    }

    pub fn decision_id(&self) -> &DecisionId {
        &self.decision_id
    }

    pub fn context(&self) -> &FarmAdviceContext {
        &self.context
    }

    pub fn objective(&self) -> &FarmerObjective {
        &self.objective
    }

    pub fn premises(&self) -> &[AdvicePremise] {
        &self.premises
    }

    pub fn predicted_metric_claims(&self) -> &BTreeSet<FactClaimId> {
        &self.predicted_metric_claims
    }

    pub fn observed_metric_claims(&self) -> &BTreeSet<FactClaimId> {
        &self.observed_metric_claims
    }

    pub fn disposition(&self) -> Option<FarmerAdviceDisposition> {
        self.disposition
    }

    pub fn review_state(&self) -> AdviceReviewState {
        self.review_state
    }

    pub fn add_premise(
        &mut self,
        premise: AdvicePremise,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), FarmAdviceError> {
        validate_fact_references(&premise.fact_claims, evidence)?;
        for existing in &self.premises {
            if existing.fact_claims.iter().any(|id| premise.fact_claims.contains(id)) {
                return Err(FarmAdviceError::DuplicatePremiseReference);
            }
        }
        self.premises.push(premise);
        Ok(())
    }

    pub fn record_predicted_metric(
        &mut self,
        claim: FactClaimId,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), FarmAdviceError> {
        require_fact(&claim, evidence)?;
        if self.observed_metric_claims.contains(&claim) {
            return Err(FarmAdviceError::PredictionObservationMustBeSeparate);
        }
        if !self.predicted_metric_claims.insert(claim.clone()) {
            return Err(FarmAdviceError::DuplicateMetricReference(claim));
        }
        Ok(())
    }

    pub fn record_observed_metric(
        &mut self,
        claim: FactClaimId,
        evidence: &DeliberationEvidenceLedger,
    ) -> Result<(), FarmAdviceError> {
        require_fact(&claim, evidence)?;
        if self.predicted_metric_claims.contains(&claim) {
            return Err(FarmAdviceError::PredictionObservationMustBeSeparate);
        }
        if !self.observed_metric_claims.insert(claim.clone()) {
            return Err(FarmAdviceError::DuplicateMetricReference(claim));
        }
        Ok(())
    }

    /// Declining or deferring advice is a valid farmer decision, not a penalty.
    pub fn record_disposition(&mut self, disposition: FarmerAdviceDisposition) {
        self.disposition = Some(disposition);
    }

    /// This projection only marks that advice needs review. WCARE-08 remains
    /// responsible for durable correction, retraction, supersession and repair.
    pub fn mark_requires_review(&mut self) {
        self.review_state = AdviceReviewState::RequiresReview;
    }

    pub fn mark_superseded(&mut self) {
        self.review_state = AdviceReviewState::Superseded;
    }

    pub fn mark_retracted(&mut self) {
        self.review_state = AdviceReviewState::Retracted;
    }
}

fn require_fact(
    claim: &FactClaimId,
    evidence: &DeliberationEvidenceLedger,
) -> Result<(), FarmAdviceError> {
    if evidence.facts().get(claim).is_none() {
        return Err(FarmAdviceError::FactReferenceNotFound(claim.clone()));
    }
    Ok(())
}

fn validate_fact_references(
    claims: &BTreeSet<FactClaimId>,
    evidence: &DeliberationEvidenceLedger,
) -> Result<(), FarmAdviceError> {
    for claim in claims {
        require_fact(claim, evidence)?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FarmAdviceError {
    EmptyIdentifier,
    EmptySiteReference,
    EmptySeasonReference,
    InvalidUsefulWindow,
    ObjectiveSourceMismatch,
    ObjectiveSummaryRequired,
    EmptyObjectiveUnit,
    UnknownMustRemainUnknown,
    ConflictingPremisesNeedMultipleClaims,
    PremiseClaimRequired,
    PremiseAsOfRequired,
    DuplicatePremiseReference,
    FactReferenceNotFound(FactClaimId),
    DecisionReferenceNotFound,
    PredictionObservationMustBeSeparate,
    DuplicateMetricReference(FactClaimId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_ledger::{EvidenceObservation, EvidenceRelation, NormativeClaimId};

    fn fact(value: &str) -> FactClaimId {
        FactClaimId::new(value).unwrap()
    }

    fn decision(value: &str) -> DecisionId {
        DecisionId::new(value).unwrap()
    }

    fn profile(evidence: &DeliberationEvidenceLedger) -> FarmAdviceProfile {
        FarmAdviceProfile::new(
            AdviceEpisodeId::new("episode-1").unwrap(),
            AccountabilityCaseId::new("wcare-case-1").unwrap(),
            decision("decision-1"),
            FarmAdviceContext::new(
                "opaque-site-a",
                FarmOperation::Irrigation,
                Some("season-2026-a".to_owned()),
                Some(UsefulWindow::new(1_800_000_000, 1_800_086_400).unwrap()),
            ).unwrap(),
            FarmerObjective::new(
                ObjectiveSource::FarmerDeclared,
                Some("Restore sufficient irrigation flow".to_owned()),
                Some("flow-rate".to_owned()),
            ).unwrap(),
            evidence,
        ).unwrap()
    }

    fn evidence() -> DeliberationEvidenceLedger {
        let mut evidence = DeliberationEvidenceLedger::new();
        evidence.add_fact(fact("premise-measured"), "reported flow measurement", 0.9).unwrap();
        evidence.add_fact(fact("forecast-flow"), "modelled flow forecast", 0.7).unwrap();
        evidence.add_fact(fact("observed-flow"), "subsequent measured flow", 0.95).unwrap();
        evidence.add_fact(fact("conflict-a"), "sensor reports low flow", 0.8).unwrap();
        evidence.add_fact(fact("conflict-b"), "farmer observation reports normal flow", 0.75).unwrap();
        evidence.add_normative_claim(NormativeClaimId::new("goal").unwrap(), "farmers choose their own goals", 0.9).unwrap();
        evidence.register_decision(decision("decision-1"), [fact("premise-measured")], [NormativeClaimId::new("goal").unwrap()]).unwrap();
        evidence
    }

    #[test]
    fn unknown_premise_is_not_fabricated_into_a_fact() {
        let mut p = profile(&evidence());
        let premise = AdvicePremise::new(AdviceEvidenceKind::Unknown, [], None).unwrap();
        p.add_premise(premise, &evidence()).unwrap();
        assert_eq!(p.premises()[0].kind(), AdviceEvidenceKind::Unknown);
        assert!(p.premises()[0].fact_claims().is_empty());
    }

    #[test]
    fn measured_premise_requires_an_existing_claim_and_as_of_time() {
        assert_eq!(
            AdvicePremise::new(AdviceEvidenceKind::Measured, [fact("premise-measured")], None),
            Err(FarmAdviceError::PremiseAsOfRequired)
        );
        let mut p = profile(&evidence());
        let premise = AdvicePremise::new(AdviceEvidenceKind::Measured, [fact("premise-measured")], Some(1_800_000_000)).unwrap();
        p.add_premise(premise, &evidence()).unwrap();
        assert!(p.add_premise(
            AdvicePremise::new(AdviceEvidenceKind::Reported, [fact("missing")], Some(1_800_000_000)).unwrap(),
            &evidence(),
        ).is_err());
    }

    #[test]
    fn contradictory_sources_remain_explicit_not_averaged() {
        let premise = AdvicePremise::new(
            AdviceEvidenceKind::Conflicting,
            [fact("conflict-a"), fact("conflict-b")],
            Some(1_800_000_000),
        ).unwrap();
        assert_eq!(premise.kind(), AdviceEvidenceKind::Conflicting);
        assert_eq!(premise.fact_claims().len(), 2);
        assert_eq!(
            AdvicePremise::new(AdviceEvidenceKind::Conflicting, [fact("conflict-a")], Some(1_800_000_000)),
            Err(FarmAdviceError::ConflictingPremisesNeedMultipleClaims)
        );
    }

    #[test]
    fn predicted_and_observed_metric_refs_cannot_be_the_same_claim() {
        let e = evidence();
        let mut p = profile(&e);
        p.record_predicted_metric(fact("forecast-flow"), &e).unwrap();
        assert_eq!(
            p.record_observed_metric(fact("forecast-flow"), &e),
            Err(FarmAdviceError::PredictionObservationMustBeSeparate)
        );
        p.record_observed_metric(fact("observed-flow"), &e).unwrap();
        assert_eq!(p.predicted_metric_claims().len(), 1);
        assert_eq!(p.observed_metric_claims().len(), 1);
    }

    #[test]
    fn farmer_declining_advice_is_a_valid_disposition() {
        let e = evidence();
        let mut p = profile(&e);
        p.record_disposition(FarmerAdviceDisposition::Declined);
        assert_eq!(p.disposition(), Some(FarmerAdviceDisposition::Declined));
        assert_eq!(p.review_state(), AdviceReviewState::Active);
    }

    #[test]
    fn premise_review_does_not_silently_rewrite_the_episode() {
        let e = evidence();
        let mut p = profile(&e);
        p.add_premise(
            AdvicePremise::new(AdviceEvidenceKind::Measured, [fact("premise-measured")], Some(1_800_000_000)).unwrap(),
            &e,
        ).unwrap();
        p.mark_requires_review();
        assert_eq!(p.review_state(), AdviceReviewState::RequiresReview);
        assert_eq!(p.premises()[0].fact_claims().iter().next(), Some(&fact("premise-measured")));
    }

    #[test]
    fn invalid_useful_window_is_rejected() {
        assert_eq!(UsefulWindow::new(10, 10), Err(FarmAdviceError::InvalidUsefulWindow));
    }
}
