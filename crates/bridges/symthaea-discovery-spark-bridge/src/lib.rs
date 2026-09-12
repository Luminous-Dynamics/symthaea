// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-shot adapter from Spark Bayesian expected-information-gain planning
//! into domain-neutral Symthaea discovery experiment proposals.
//!
//! The bridge deliberately uses Spark's **current-belief ranking only**. It
//! never calls Spark's multi-step `greedy_sequence`, because that planner
//! advances belief with a simulated MAP-world outcome. A simulated outcome is
//! useful planning state, not observed evidence.

#![forbid(unsafe_code)]

use spark_engine::{
    ExperimentDesign, ExperimentInfoGain, HypothesisBelief, rank_experiments,
    standard_candidate_experiments,
};
use std::collections::BTreeSet;
use symthaea_discovery::{
    Candidate, CandidateId, CandidateOrigin, DiscoveryError, Evaluation, EvidenceKind,
    ExperimentProposal, ExperimentSelector, ResourceEstimate,
};

/// Candidate kind accepted by this Spark-specific selector.
pub const SPARK_LCF_CANDIDATE_KIND: &str = "spark_lcf_anomaly_program";
/// Explicit planning convention used when mapping Spark's month estimate into
/// the discovery contract's seconds field.
pub const PLANNING_MONTH_SECONDS: f64 = 30.0 * 24.0 * 60.0 * 60.0;
/// Descriptive method label repeated on translated proposals.
pub const SPARK_EIG_METHOD: &str =
    "spark-engine signature-level Bayesian expected-information-gain ranking";

/// Construct the descriptive discovery candidate that this adapter accepts.
pub fn spark_lcf_candidate(id: impl Into<String>) -> Result<Candidate, DiscoveryError> {
    let candidate = Candidate::new(
        CandidateId::new(id.into())?,
        SPARK_LCF_CANDIDATE_KIND,
        CandidateOrigin::Imported {
            source: "spark-engine::ExperimentDesigner".into(),
        },
    )?
    .with_spec("planner", "spark-engine::optimal_experiment")
    .with_spec("belief_semantics", "signature-level decision-support");
    Ok(candidate)
}

/// Current-belief EIG selector backed by Spark's existing ranking logic.
///
/// This type contains no execution authority and never mutates its belief.
#[derive(Debug, Clone)]
pub struct SparkEigSelector {
    belief: HypothesisBelief,
    designs: Vec<ExperimentDesign>,
    minimum_eig_bits: f64,
}

impl SparkEigSelector {
    /// Construct from an explicit Spark belief and candidate-design set.
    pub fn new(
        belief: HypothesisBelief,
        designs: Vec<ExperimentDesign>,
        minimum_eig_bits: f64,
    ) -> Result<Self, DiscoveryError> {
        if !minimum_eig_bits.is_finite() || minimum_eig_bits < 0.0 {
            return Err(DiscoveryError::Invalid(
                "minimum EIG must be finite and non-negative".into(),
            ));
        }
        validate_designs(&designs)?;
        Ok(Self {
            belief,
            designs,
            minimum_eig_bits,
        })
    }

    /// Standard Spark anomaly-program selector using Spark's skeptical priors
    /// and standard candidate experiments.
    pub fn standard() -> Result<Self, DiscoveryError> {
        Self::new(
            HypothesisBelief::default_priors(),
            standard_candidate_experiments(),
            spark_engine::optimal_experiment::MIN_USEFUL_EIG_BITS,
        )
    }

    pub fn belief(&self) -> &HypothesisBelief {
        &self.belief
    }

    pub fn designs(&self) -> &[ExperimentDesign] {
        &self.designs
    }

    pub fn minimum_eig_bits(&self) -> f64 {
        self.minimum_eig_bits
    }

    /// Translate the complete current-belief Spark ranking into generic
    /// discovery proposals. The order is exactly Spark's `rank_experiments`
    /// order after adapter validation.
    pub fn rank_proposals(
        &self,
        candidate: &Candidate,
    ) -> Result<Vec<ExperimentProposal>, DiscoveryError> {
        validate_candidate(candidate)?;
        validate_designs(&self.designs)?;

        let ranked = rank_experiments(&self.designs, &self.belief);
        let mut proposals = Vec::with_capacity(ranked.len());
        for info in &ranked {
            validate_ranked_info(info)?;
            let design = self
                .designs
                .iter()
                .find(|design| design.name == info.name)
                .ok_or_else(|| {
                    DiscoveryError::Invalid(format!(
                        "Spark ranking referenced unknown design {:?}",
                        info.name
                    ))
                })?;
            proposals.push(proposal_from_spark(candidate, design, info)?);
        }
        Ok(proposals)
    }

    fn select_current_best(
        &self,
        candidate: &Candidate,
        evaluations: &[Evaluation],
    ) -> Result<Option<ExperimentProposal>, DiscoveryError> {
        validate_candidate(candidate)?;
        validate_evaluations(candidate, evaluations)?;
        let proposals = self.rank_proposals(candidate)?;
        Ok(proposals.into_iter().find(|proposal| {
            proposal
                .expected_information_gain_bits
                .is_some_and(|bits| bits >= self.minimum_eig_bits)
        }))
    }
}

impl ExperimentSelector for SparkEigSelector {
    fn name(&self) -> &'static str {
        "spark-current-belief-eig"
    }

    fn select_next(
        &self,
        candidate: &Candidate,
        evaluations: &[Evaluation],
    ) -> Result<Option<ExperimentProposal>, DiscoveryError> {
        self.select_current_best(candidate, evaluations)
    }
}

fn validate_candidate(candidate: &Candidate) -> Result<(), DiscoveryError> {
    // CandidateId is a public tuple type in discovery v0. Revalidate at the
    // bridge boundary so direct construction/deserialization cannot bypass the
    // constructor check here.
    CandidateId::new(candidate.id.0.clone())?;
    if candidate.kind != SPARK_LCF_CANDIDATE_KIND {
        return Err(DiscoveryError::Invalid(format!(
            "Spark EIG adapter requires candidate kind {SPARK_LCF_CANDIDATE_KIND:?}, got {:?}",
            candidate.kind
        )));
    }
    Ok(())
}

fn validate_evaluations(
    candidate: &Candidate,
    evaluations: &[Evaluation],
) -> Result<(), DiscoveryError> {
    for evaluation in evaluations {
        if evaluation.candidate_id != candidate.id {
            return Err(DiscoveryError::CandidateMismatch {
                expected: candidate.id.0.clone(),
                found: evaluation.candidate_id.0.clone(),
            });
        }
        evaluation.validate()?;
    }
    Ok(())
}

fn validate_designs(designs: &[ExperimentDesign]) -> Result<(), DiscoveryError> {
    let mut names = BTreeSet::new();
    for design in designs {
        if design.name.trim().is_empty() || design.research_question.trim().is_empty() {
            return Err(DiscoveryError::Invalid(
                "Spark experiment designs require non-empty name and research question".into(),
            ));
        }
        if !names.insert(design.name.as_str()) {
            return Err(DiscoveryError::Invalid(format!(
                "duplicate Spark experiment name {:?}",
                design.name
            )));
        }
        if !design.estimated_cost_usd.is_finite() || design.estimated_cost_usd < 0.0 {
            return Err(DiscoveryError::Invalid(format!(
                "Spark experiment {:?} has invalid estimated cost",
                design.name
            )));
        }
        if !design.duration_months.is_finite() || design.duration_months < 0.0 {
            return Err(DiscoveryError::Invalid(format!(
                "Spark experiment {:?} has invalid duration",
                design.name
            )));
        }
    }
    Ok(())
}

fn validate_ranked_info(info: &ExperimentInfoGain) -> Result<(), DiscoveryError> {
    for (name, value) in [
        ("EIG", info.eig_bits),
        ("cost", info.cost_usd),
        ("duration", info.duration_months),
        ("EIG per $100K", info.eig_per_100k_usd),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(DiscoveryError::Invalid(format!(
                "Spark ranking {:?} has invalid {name}",
                info.name
            )));
        }
    }
    Ok(())
}

fn proposal_from_spark(
    candidate: &Candidate,
    design: &ExperimentDesign,
    info: &ExperimentInfoGain,
) -> Result<ExperimentProposal, DiscoveryError> {
    if design.name != info.name {
        return Err(DiscoveryError::Invalid(format!(
            "Spark design/ranking name mismatch: {:?} vs {:?}",
            design.name, info.name
        )));
    }

    let duration_seconds = design.duration_months * PLANNING_MONTH_SECONDS;
    if !duration_seconds.is_finite() || duration_seconds < 0.0 {
        return Err(DiscoveryError::Invalid(format!(
            "Spark experiment {:?} duration cannot be represented in seconds",
            design.name
        )));
    }

    let proposal = ExperimentProposal {
        id: format!("spark-eig::{}", design.name.trim()),
        candidate_id: candidate.id.clone(),
        question: design.research_question.trim().to_owned(),
        method: SPARK_EIG_METHOD.to_owned(),
        expected_information_gain_bits: Some(info.eig_bits),
        estimated_resource: Some(ResourceEstimate {
            amount: design.estimated_cost_usd,
            unit: "USD".into(),
        }),
        estimated_duration_seconds: Some(duration_seconds),
        intended_evidence: vec![EvidenceKind::Experiment],
    };
    proposal.validate()?;
    Ok(proposal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_engine::rank_experiments;
    use std::collections::BTreeMap;

    fn candidate() -> Candidate {
        spark_lcf_candidate("spark-program-001").unwrap()
    }

    #[test]
    fn standard_selector_returns_current_belief_experiment_proposal() {
        let selector = SparkEigSelector::standard().unwrap();
        let proposal = selector.select_next(&candidate(), &[]).unwrap().unwrap();

        assert_eq!(proposal.candidate_id.0, "spark-program-001");
        assert_eq!(proposal.method, SPARK_EIG_METHOD);
        assert_eq!(proposal.intended_evidence, vec![EvidenceKind::Experiment]);
        assert!(proposal.expected_information_gain_bits.unwrap() > 0.0);
        assert_eq!(proposal.estimated_resource.as_ref().unwrap().unit, "USD");
    }

    #[test]
    fn translated_ranking_preserves_spark_order_and_eig() {
        let selector = SparkEigSelector::standard().unwrap();
        let spark_ranked = rank_experiments(selector.designs(), selector.belief());
        let translated = selector.rank_proposals(&candidate()).unwrap();

        assert_eq!(spark_ranked.len(), translated.len());
        for (spark, proposal) in spark_ranked.iter().zip(&translated) {
            assert_eq!(proposal.id, format!("spark-eig::{}", spark.name));
            assert_eq!(proposal.expected_information_gain_bits, Some(spark.eig_bits));
            assert_eq!(
                proposal.estimated_resource.as_ref().unwrap().amount,
                spark.cost_usd
            );
            assert_eq!(
                proposal.estimated_duration_seconds,
                Some(spark.duration_months * PLANNING_MONTH_SECONDS)
            );
        }
    }

    #[test]
    fn selection_does_not_mutate_or_simulate_belief_updates() {
        let selector = SparkEigSelector::standard().unwrap();
        let before: Vec<_> = selector.belief().iter().collect();
        let _ = selector.select_next(&candidate(), &[]).unwrap();
        let after: Vec<_> = selector.belief().iter().collect();
        assert_eq!(before, after);
    }

    #[test]
    fn wrong_candidate_kind_fails_closed() {
        let wrong = Candidate::new(
            CandidateId::new("other").unwrap(),
            "photovoltaic_material",
            CandidateOrigin::UserProposed,
        )
        .unwrap();
        assert!(SparkEigSelector::standard()
            .unwrap()
            .select_next(&wrong, &[])
            .is_err());
    }

    #[test]
    fn public_tuple_candidate_id_is_revalidated() {
        let invalid = Candidate {
            id: CandidateId("   ".into()),
            kind: SPARK_LCF_CANDIDATE_KIND.into(),
            specification: BTreeMap::new(),
            origin: CandidateOrigin::UserProposed,
        };
        assert!(SparkEigSelector::standard()
            .unwrap()
            .select_next(&invalid, &[])
            .is_err());
    }

    #[test]
    fn mismatched_evaluation_candidate_is_rejected() {
        let evaluation = Evaluation {
            candidate_id: CandidateId::new("different").unwrap(),
            objectives: vec![],
            constraints: vec![],
            predictions: vec![],
            pareto_rank: None,
        };
        assert!(matches!(
            SparkEigSelector::standard()
                .unwrap()
                .select_next(&candidate(), &[evaluation]),
            Err(DiscoveryError::CandidateMismatch { .. })
        ));
    }

    #[test]
    fn duplicate_design_names_are_rejected() {
        let mut designs = standard_candidate_experiments();
        let duplicate = designs[0].clone();
        designs.push(duplicate);
        assert!(SparkEigSelector::new(HypothesisBelief::uniform(), designs, 0.0).is_err());
    }

    #[test]
    fn invalid_cost_or_duration_is_rejected() {
        let mut bad_cost = standard_candidate_experiments();
        bad_cost[0].estimated_cost_usd = -1.0;
        assert!(SparkEigSelector::new(HypothesisBelief::uniform(), bad_cost, 0.0).is_err());

        let mut bad_duration = standard_candidate_experiments();
        bad_duration[0].duration_months = f64::NAN;
        assert!(SparkEigSelector::new(HypothesisBelief::uniform(), bad_duration, 0.0).is_err());
    }

    #[test]
    fn no_encoded_information_can_return_no_next_experiment() {
        let mut design = standard_candidate_experiments().remove(0);
        design.expected_outcomes.clear();
        let selector = SparkEigSelector::new(HypothesisBelief::uniform(), vec![design], 0.05)
            .unwrap();
        assert!(selector.select_next(&candidate(), &[]).unwrap().is_none());
    }

    #[test]
    fn physical_setup_is_not_copied_into_generic_method_field() {
        let selector = SparkEigSelector::standard().unwrap();
        let proposal = selector.select_next(&candidate(), &[]).unwrap().unwrap();
        assert_eq!(proposal.method, SPARK_EIG_METHOD);
        assert!(!proposal.method.contains("loading_ratio"));
        assert!(!proposal.method.contains("trigger"));
    }
}
