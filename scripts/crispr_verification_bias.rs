//! Probability-sampled verification audit for off-target triage benchmarks.
//!
//! Motivation: measured labels can be selectively observed because existing tools nominate
//! some candidates more often than others. A complete-case score on that measured subset is
//! descriptive of the subset, not automatically the full frozen candidate universe.
//!
//! This module computes inverse-inclusion-probability (Horvitz-Thompson) estimates of candidate
//! and positive counts, plus ratio (Hájek-style) precision/recall estimates from a declared
//! probability audit sample. It deliberately does not compute confidence intervals, prove that
//! the declared sample design was followed, or recover candidates absent from the frozen
//! candidate universe. An independent audit must verify the ranking census, sample frame,
//! inclusion probabilities, source permissions, and outcome-blind selection procedure.
//!
//! It is a report-only retrospective metric kernel, not a CRISPR design, wet-lab, clinical,
//! or safety tool.
//!
//! Compile without Cargo/network:
//! rustc --edition=2024 --test scripts/crispr_verification_bias.rs -o /tmp/crispr-verification-bias-tests
//! /tmp/crispr-verification-bias-tests

use std::collections::HashSet;
use std::fmt;

const TOLERANCE: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct VerificationDesign {
    pub design_id: String,
    pub population_size: usize,
    pub top_k: usize,
    /// Attestation from the frozen design manifest; this module cannot prove it from rows.
    pub probability_sample: bool,
    /// Inclusion probabilities refer to resolved, analyzable outcomes—not merely invitations.
    pub resolved_outcome_inclusion_probabilities_verified: bool,
    /// All population members had nonzero chance of audit selection.
    pub all_candidates_have_nonzero_inclusion_probability: bool,
    /// Candidate order and K were frozen before audit outcomes were revealed.
    pub ranking_frozen_before_outcome_reveal: bool,
    /// Selection was outcome-blind conditional on the declared sampling strata/design.
    pub selection_outcome_blind: bool,
    /// A separate verifier confirmed one rank for every candidate in the declared universe.
    pub full_ranking_census_verified: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuditOutcome {
    Positive,
    AssayDefinedNegative,
    UnknownUnassayed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuditedCandidate {
    pub candidate_id: String,
    /// One-based position in the independently frozen complete ranking.
    pub rank: usize,
    /// An unresolved outcome is a hard error; it must never be coerced to an assay-negative.
    pub outcome: AuditOutcome,
    /// Known probability that this candidate appears in the final resolved-label audit sample.
    pub inclusion_probability: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerificationBiasReport {
    pub design_id: String,
    pub population_size: usize,
    pub top_k: usize,
    pub audited_count: usize,
    pub audited_positive_count: usize,
    pub audited_top_k_count: usize,
    pub audited_top_k_positive_count: usize,
    /// Horvitz-Thompson count estimate; finite-sample estimates may differ from population_size.
    pub estimated_population_candidate_count: f64,
    pub estimated_population_positive_count: f64,
    pub estimated_top_k_candidate_count: f64,
    pub estimated_top_k_positive_count: f64,
    /// Ratio of weighted positive estimates. Not an unbiased finite-sample ratio estimator.
    pub design_weighted_recall_at_k: Option<f64>,
    /// Ratio of weighted top-K positives to weighted top-K membership; avoids clipping.
    pub design_weighted_precision_at_k: Option<f64>,
    pub design_weighted_positive_prevalence: Option<f64>,
    /// Subset-only complete-case figures are shown as diagnostics, not population estimates.
    pub naive_audit_sample_recall_at_k: Option<f64>,
    pub naive_audit_sample_precision_at_k: Option<f64>,
    pub overall_kish_effective_sample_size: f64,
    pub top_k_kish_effective_sample_size: f64,
    pub interpretation: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VerificationBiasError {
    EmptyDesignId,
    EmptySample,
    InvalidPopulationSize,
    InvalidTopK { top_k: usize, population_size: usize },
    NonProbabilitySample,
    ResolvedOutcomeInclusionProbabilitiesNotVerified,
    UnresolvedOutcomeIncludedInAudit(String),
    ZeroInclusionProbabilityInFrame,
    RankingNotFrozen,
    OutcomeDependentSelection,
    RankingCensusNotVerified,
    SampleLargerThanPopulation { sample_count: usize, population_size: usize },
    EmptyCandidateId,
    DuplicateCandidateId(String),
    InvalidRank { candidate_id: String, rank: usize, population_size: usize },
    DuplicateSampledRank(usize),
    InvalidInclusionProbability { candidate_id: String, probability_bits: u64 },
}

impl fmt::Display for VerificationBiasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDesignId => write!(f, "verification design ID must not be blank"),
            Self::EmptySample => write!(f, "verification audit sample must not be empty"),
            Self::InvalidPopulationSize => write!(f, "population size must be positive"),
            Self::InvalidTopK { top_k, population_size } => write!(f, "top_k {top_k} must be in 1..={population_size}"),
            Self::NonProbabilitySample => write!(f, "non-probability verification samples cannot identify population metrics"),
            Self::ResolvedOutcomeInclusionProbabilitiesNotVerified => write!(f, "inclusion probabilities for resolved, analyzable outcomes are not independently verified"),
            Self::UnresolvedOutcomeIncludedInAudit(id) => write!(f, "candidate {id} has an unknown/unresolved outcome; it cannot be treated as negative or silently dropped from the audit"),
            Self::ZeroInclusionProbabilityInFrame => write!(f, "every candidate in the target frame must have nonzero audit inclusion probability"),
            Self::RankingNotFrozen => write!(f, "ranking and K must be frozen before audit outcomes are revealed"),
            Self::OutcomeDependentSelection => write!(f, "audit selection must be outcome-blind conditional on the declared design"),
            Self::RankingCensusNotVerified => write!(f, "the complete candidate ranking census must be independently verified"),
            Self::SampleLargerThanPopulation { sample_count, population_size } => write!(f, "sample count {sample_count} exceeds declared population size {population_size}"),
            Self::EmptyCandidateId => write!(f, "candidate ID must not be blank"),
            Self::DuplicateCandidateId(id) => write!(f, "duplicate candidate ID: {id}"),
            Self::InvalidRank { candidate_id, rank, population_size } => write!(f, "candidate {candidate_id} has rank {rank}, expected 1..={population_size}"),
            Self::DuplicateSampledRank(rank) => write!(f, "sample contains duplicate rank {rank}"),
            Self::InvalidInclusionProbability { candidate_id, probability_bits } => {
                let probability = f64::from_bits(*probability_bits);
                write!(f, "candidate {candidate_id} has invalid inclusion probability {probability}")
            }
        }
    }
}

impl std::error::Error for VerificationBiasError {}

fn validate(
    design: &VerificationDesign,
    sample: &[AuditedCandidate],
) -> Result<(), VerificationBiasError> {
    if design.design_id.trim().is_empty() {
        return Err(VerificationBiasError::EmptyDesignId);
    }
    if design.population_size == 0 {
        return Err(VerificationBiasError::InvalidPopulationSize);
    }
    if design.top_k == 0 || design.top_k > design.population_size {
        return Err(VerificationBiasError::InvalidTopK {
            top_k: design.top_k,
            population_size: design.population_size,
        });
    }
    if !design.probability_sample {
        return Err(VerificationBiasError::NonProbabilitySample);
    }
    if !design.resolved_outcome_inclusion_probabilities_verified {
        return Err(VerificationBiasError::ResolvedOutcomeInclusionProbabilitiesNotVerified);
    }
    if !design.all_candidates_have_nonzero_inclusion_probability {
        return Err(VerificationBiasError::ZeroInclusionProbabilityInFrame);
    }
    if !design.ranking_frozen_before_outcome_reveal {
        return Err(VerificationBiasError::RankingNotFrozen);
    }
    if !design.selection_outcome_blind {
        return Err(VerificationBiasError::OutcomeDependentSelection);
    }
    if !design.full_ranking_census_verified {
        return Err(VerificationBiasError::RankingCensusNotVerified);
    }
    if sample.is_empty() {
        return Err(VerificationBiasError::EmptySample);
    }
    if sample.len() > design.population_size {
        return Err(VerificationBiasError::SampleLargerThanPopulation {
            sample_count: sample.len(),
            population_size: design.population_size,
        });
    }

    let mut ids = HashSet::with_capacity(sample.len());
    let mut ranks = HashSet::with_capacity(sample.len());
    for row in sample {
        if row.candidate_id.trim().is_empty() {
            return Err(VerificationBiasError::EmptyCandidateId);
        }
        if !ids.insert(row.candidate_id.as_str()) {
            return Err(VerificationBiasError::DuplicateCandidateId(row.candidate_id.clone()));
        }
        if row.rank == 0 || row.rank > design.population_size {
            return Err(VerificationBiasError::InvalidRank {
                candidate_id: row.candidate_id.clone(),
                rank: row.rank,
                population_size: design.population_size,
            });
        }
        if !ranks.insert(row.rank) {
            return Err(VerificationBiasError::DuplicateSampledRank(row.rank));
        }
        if row.outcome == AuditOutcome::UnknownUnassayed {
            return Err(VerificationBiasError::UnresolvedOutcomeIncludedInAudit(row.candidate_id.clone()));
        }
        let p = row.inclusion_probability;
        if !p.is_finite() || p <= 0.0 || p > 1.0 {
            return Err(VerificationBiasError::InvalidInclusionProbability {
                candidate_id: row.candidate_id.clone(),
                probability_bits: p.to_bits(),
            });
        }
    }
    Ok(())
}

fn kish_effective_sample_size(weights: &[f64]) -> f64 {
    let sum = weights.iter().sum::<f64>();
    let squares = weights.iter().map(|w| w * w).sum::<f64>();
    if squares <= TOLERANCE { 0.0 } else { sum * sum / squares }
}

pub fn verification_bias_report(
    design: &VerificationDesign,
    sample: &[AuditedCandidate],
) -> Result<VerificationBiasReport, VerificationBiasError> {
    validate(design, sample)?;

    let mut population_weights = Vec::with_capacity(sample.len());
    let mut top_k_weights = Vec::new();
    let mut total_positive_hat = 0.0;
    let mut top_k_positive_hat = 0.0;
    let mut total_positive_observed = 0usize;
    let mut top_k_positive_observed = 0usize;
    let mut top_k_observed = 0usize;

    for row in sample {
        let weight = 1.0 / row.inclusion_probability;
        population_weights.push(weight);
        if row.outcome == AuditOutcome::Positive {
            total_positive_hat += weight;
            total_positive_observed += 1;
        }
        if row.rank <= design.top_k {
            top_k_observed += 1;
            top_k_weights.push(weight);
            if row.outcome == AuditOutcome::Positive {
                top_k_positive_hat += weight;
                top_k_positive_observed += 1;
            }
        }
    }

    let estimated_population_count = population_weights.iter().sum::<f64>();
    let estimated_top_k_count = top_k_weights.iter().sum::<f64>();
    let recall = (total_positive_hat > TOLERANCE)
        .then_some(top_k_positive_hat / total_positive_hat);
    let precision = (estimated_top_k_count > TOLERANCE)
        .then_some(top_k_positive_hat / estimated_top_k_count);
    let prevalence = (estimated_population_count > TOLERANCE)
        .then_some(total_positive_hat / estimated_population_count);
    let naive_recall = (total_positive_observed > 0)
        .then_some(top_k_positive_observed as f64 / total_positive_observed as f64);
    let naive_precision = (top_k_observed > 0)
        .then_some(top_k_positive_observed as f64 / top_k_observed as f64);

    Ok(VerificationBiasReport {
        design_id: design.design_id.clone(),
        population_size: design.population_size,
        top_k: design.top_k,
        audited_count: sample.len(),
        audited_positive_count: total_positive_observed,
        audited_top_k_count: top_k_observed,
        audited_top_k_positive_count: top_k_positive_observed,
        estimated_population_candidate_count: estimated_population_count,
        estimated_population_positive_count: total_positive_hat,
        estimated_top_k_candidate_count: estimated_top_k_count,
        estimated_top_k_positive_count: top_k_positive_hat,
        design_weighted_recall_at_k: recall,
        design_weighted_precision_at_k: precision,
        design_weighted_positive_prevalence: prevalence,
        naive_audit_sample_recall_at_k: naive_recall,
        naive_audit_sample_precision_at_k: naive_precision,
        overall_kish_effective_sample_size: kish_effective_sample_size(&population_weights),
        top_k_kish_effective_sample_size: kish_effective_sample_size(&top_k_weights),
        interpretation: "Probability-audit estimates are conditional on a correct frozen frame and inclusion probabilities; ratio estimates are not finite-sample unbiased and no confidence interval is computed. Non-probability samples, zero-inclusion strata, unverified rankings, and outcome-informed sampling are blocked.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(n: usize, k: usize) -> VerificationDesign {
        VerificationDesign {
            design_id: "audit-v1".to_owned(),
            population_size: n,
            top_k: k,
            probability_sample: true,
            all_candidates_have_nonzero_inclusion_probability: true,
            resolved_outcome_inclusion_probabilities_verified: true,
            ranking_frozen_before_outcome_reveal: true,
            selection_outcome_blind: true,
            full_ranking_census_verified: true,
        }
    }

    fn row(id: &str, rank: usize, positive: bool, pi: f64) -> AuditedCandidate {
        AuditedCandidate {
            candidate_id: id.to_owned(),
            rank,
            outcome: if positive { AuditOutcome::Positive } else { AuditOutcome::AssayDefinedNegative },
            inclusion_probability: pi,
        }
    }

    fn f14_sample() -> Vec<AuditedCandidate> {
        vec![
            row("r1", 1, true, 0.5),
            row("r2", 2, false, 0.5),
            row("r5", 5, true, 0.25),
            row("r7", 7, false, 0.5),
        ]
    }

    #[test]
    fn full_census_recovers_exact_top_k_metrics() {
        let s = vec![row("a", 1, true, 1.0), row("b", 2, false, 1.0),
                     row("c", 3, true, 1.0), row("d", 4, false, 1.0)];
        let r = verification_bias_report(&design(4, 2), &s).unwrap();
        assert!((r.estimated_population_candidate_count - 4.0).abs() < TOLERANCE);
        assert!((r.estimated_population_positive_count - 2.0).abs() < TOLERANCE);
        assert!((r.estimated_top_k_candidate_count - 2.0).abs() < TOLERANCE);
        assert!((r.estimated_top_k_positive_count - 1.0).abs() < TOLERANCE);
        assert!((r.design_weighted_recall_at_k.unwrap() - 0.5).abs() < TOLERANCE);
        assert!((r.design_weighted_precision_at_k.unwrap() - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn f14_weights_nonuniform_verification_and_exposes_complete_case_bias() {
        let r = verification_bias_report(&design(10, 4), &f14_sample()).unwrap();
        assert_eq!(r.audited_count, 4);
        assert_eq!(r.audited_positive_count, 2);
        assert_eq!(r.audited_top_k_count, 2);
        assert_eq!(r.audited_top_k_positive_count, 1);
        assert!((r.estimated_population_candidate_count - 10.0).abs() < TOLERANCE);
        assert!((r.estimated_population_positive_count - 6.0).abs() < TOLERANCE);
        assert!((r.estimated_top_k_candidate_count - 4.0).abs() < TOLERANCE);
        assert!((r.estimated_top_k_positive_count - 2.0).abs() < TOLERANCE);
        assert!((r.design_weighted_recall_at_k.unwrap() - 1.0 / 3.0).abs() < TOLERANCE);
        assert!((r.design_weighted_precision_at_k.unwrap() - 0.5).abs() < TOLERANCE);
        assert!((r.naive_audit_sample_recall_at_k.unwrap() - 0.5).abs() < TOLERANCE);
        assert!((r.naive_audit_sample_precision_at_k.unwrap() - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn rank_membership_is_derived_from_frozen_rank_and_k() {
        let r = verification_bias_report(&design(10, 4), &f14_sample()).unwrap();
        assert_eq!(r.audited_top_k_count, 2);
        assert_eq!(r.audited_top_k_positive_count, 1);
    }

    #[test]
    fn empty_sample_is_rejected() {
        assert_eq!(verification_bias_report(&design(4, 2), &[]), Err(VerificationBiasError::EmptySample));
    }

    #[test]
    fn invalid_population_size_is_rejected() {
        let mut d = design(0, 0);
        d.population_size = 0;
        assert_eq!(verification_bias_report(&d, &[row("a", 1, true, 1.0)]), Err(VerificationBiasError::InvalidPopulationSize));
    }

    #[test]
    fn invalid_top_k_is_rejected() {
        assert!(matches!(verification_bias_report(&design(4, 5), &[row("a", 1, true, 1.0)]), Err(VerificationBiasError::InvalidTopK { .. })));
        assert!(matches!(verification_bias_report(&design(4, 0), &[row("a", 1, true, 1.0)]), Err(VerificationBiasError::InvalidTopK { .. })));
    }

    #[test]
    fn non_probability_sample_is_rejected() {
        let mut d = design(4, 2); d.probability_sample = false;
        assert_eq!(verification_bias_report(&d, &[row("a", 1, true, 1.0)]), Err(VerificationBiasError::NonProbabilitySample));
    }

    #[test]
    fn zero_inclusion_probability_frame_is_rejected() {
        let mut d = design(4, 2); d.all_candidates_have_nonzero_inclusion_probability = false;
        assert_eq!(verification_bias_report(&d, &[row("a", 1, true, 1.0)]), Err(VerificationBiasError::ZeroInclusionProbabilityInFrame));
    }

    #[test]
    fn post_outcome_ranking_is_rejected() {
        let mut d = design(4, 2); d.ranking_frozen_before_outcome_reveal = false;
        assert_eq!(verification_bias_report(&d, &[row("a", 1, true, 1.0)]), Err(VerificationBiasError::RankingNotFrozen));
    }

    #[test]
    fn outcome_dependent_sampling_is_rejected() {
        let mut d = design(4, 2); d.selection_outcome_blind = false;
        assert_eq!(verification_bias_report(&d, &[row("a", 1, true, 1.0)]), Err(VerificationBiasError::OutcomeDependentSelection));
    }

    #[test]
    fn resolved_outcome_inclusion_probabilities_must_be_verified() {
        let mut d = design(4, 2); d.resolved_outcome_inclusion_probabilities_verified = false;
        assert_eq!(verification_bias_report(&d, &[row("a", 1, true, 1.0)]), Err(VerificationBiasError::ResolvedOutcomeInclusionProbabilitiesNotVerified));
    }

    #[test]
    fn unknown_outcome_in_probability_audit_fails_closed() {
        let mut unknown = row("u", 1, false, 1.0);
        unknown.outcome = AuditOutcome::UnknownUnassayed;
        assert_eq!(verification_bias_report(&design(4, 2), &[unknown]), Err(VerificationBiasError::UnresolvedOutcomeIncludedInAudit("u".to_owned())));
    }

    #[test]
    fn unverified_ranking_census_is_rejected() {
        let mut d = design(4, 2); d.full_ranking_census_verified = false;
        assert_eq!(verification_bias_report(&d, &[row("a", 1, true, 1.0)]), Err(VerificationBiasError::RankingCensusNotVerified));
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let s = vec![row("a", 1, true, 1.0), row("a", 2, false, 1.0)];
        assert_eq!(verification_bias_report(&design(4, 2), &s), Err(VerificationBiasError::DuplicateCandidateId("a".to_owned())));
    }

    #[test]
    fn duplicate_sampled_ranks_are_rejected() {
        let s = vec![row("a", 1, true, 1.0), row("b", 1, false, 1.0)];
        assert_eq!(verification_bias_report(&design(4, 2), &s), Err(VerificationBiasError::DuplicateSampledRank(1)));
    }

    #[test]
    fn invalid_inclusion_probabilities_are_rejected() {
        for pi in [0.0, -0.1, 1.01, f64::NAN, f64::INFINITY] {
            assert!(matches!(verification_bias_report(&design(4, 2), &[row("a", 1, true, pi)]), Err(VerificationBiasError::InvalidInclusionProbability { .. })));
        }
    }

    #[test]
    fn ranks_outside_population_are_rejected() {
        assert!(matches!(verification_bias_report(&design(4, 2), &[row("a", 5, true, 1.0)]), Err(VerificationBiasError::InvalidRank { .. })));
        assert!(matches!(verification_bias_report(&design(4, 2), &[row("a", 0, true, 1.0)]), Err(VerificationBiasError::InvalidRank { .. })));
    }

    #[test]
    fn blank_design_and_candidate_ids_are_rejected() {
        let mut d = design(4, 2); d.design_id = " ".to_owned();
        assert_eq!(verification_bias_report(&d, &[row("a", 1, true, 1.0)]), Err(VerificationBiasError::EmptyDesignId));
        let d = design(4, 2);
        assert_eq!(verification_bias_report(&d, &[row(" ", 1, true, 1.0)]), Err(VerificationBiasError::EmptyCandidateId));
    }

    #[test]
    fn empty_and_full_positive_cases_have_explicit_estimability() {
        let all_negative = vec![row("a", 1, false, 1.0), row("b", 3, false, 1.0)];
        let r = verification_bias_report(&design(4, 2), &all_negative).unwrap();
        assert_eq!(r.design_weighted_recall_at_k, None);
        assert_eq!(r.design_weighted_positive_prevalence, Some(0.0));
        let no_top_k = vec![row("c", 3, true, 1.0), row("d", 4, false, 1.0)];
        let r = verification_bias_report(&design(4, 2), &no_top_k).unwrap();
        assert_eq!(r.design_weighted_precision_at_k, None);
    }

    #[test]
    fn sample_cannot_exceed_population_size() {
        let s = vec![row("a", 1, true, 1.0), row("b", 2, false, 1.0), row("c", 3, true, 1.0)];
        assert!(matches!(verification_bias_report(&design(2, 1), &s), Err(VerificationBiasError::SampleLargerThanPopulation { .. })));
    }

    #[test]
    fn probability_one_sample_has_unit_weights() {
        let s = vec![row("a", 1, true, 1.0), row("b", 2, false, 1.0)];
        let r = verification_bias_report(&design(2, 1), &s).unwrap();
        assert_eq!(r.overall_kish_effective_sample_size, 2.0);
        assert_eq!(r.top_k_kish_effective_sample_size, 1.0);
    }
}
