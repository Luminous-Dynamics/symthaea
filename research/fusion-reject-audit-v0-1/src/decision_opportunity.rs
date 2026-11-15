//! Conditional decision-opportunity audits for screened-out fusion candidates.
//!
//! This module reuses the parent crate's frozen candidate partition, sampled
//! identities, hypergeometric inversion and Bonferroni familywise correction.
//! It measures opportunities relative to a frozen incumbent frontier; it does
//! not claim global Pareto recall or quantify unbounded decision regret.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    AuditError, AuditOutcome, RejectAuditPlan, audit_rejected_candidates,
    upper_hypergeometric_population_successes,
};

pub const DECISION_RULE: &str = "epsilon-frontier-extension-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectiveDefinition {
    pub objective_id: String,
    pub unit: String,
    pub direction: ObjectiveDirection,
    /// Materiality tolerance in this objective's declared unit.
    pub epsilon: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectiveValue {
    pub objective_id: String,
    pub unit: String,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IncumbentFrontierPoint {
    pub candidate_id: String,
    pub objectives: Vec<ObjectiveValue>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionEvaluation {
    pub candidate_id: String,
    pub reference_outcome: AuditOutcome,
    /// Declared applicability result for the exact reference evaluation.
    /// False does not mean infeasible; it means this comparison cannot classify
    /// the candidate as a confirmed decision opportunity or non-opportunity.
    pub within_declared_domain: bool,
    pub objectives: Vec<ObjectiveValue>,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StratumDecisionEvaluations {
    pub stratum_id: String,
    /// Must match the already-frozen RejectStratum sample identities in draw order.
    pub evaluations: Vec<DecisionEvaluation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionOpportunityPlan {
    pub campaign_id: String,
    pub objective_manifest_sha256: String,
    pub opportunity_rule_sha256: String,
    pub incumbent_frontier_sha256: String,
    pub target_opportunity_recall: f64,
    /// The canonical feasibility audit plan owns the universe, partition,
    /// seed, strata, sample identities and exact statistical bound procedure.
    pub base_plan: RejectAuditPlan,
    pub objective_definitions: Vec<ObjectiveDefinition>,
    pub incumbent_frontier: Vec<IncumbentFrontierPoint>,
    /// Exact high-fidelity reference results for every accepted candidate.
    pub accepted_evaluations: Vec<DecisionEvaluation>,
    /// Exact high-fidelity reference results for every sampled rejected candidate.
    pub rejected_evaluations: Vec<StratumDecisionEvaluations>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DecisionOpportunityError {
    BaseAuditRefused(Vec<AuditError>),
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    InvalidOpportunityRecallTarget,
    EmptyObjectiveDefinitions,
    DuplicateObjectiveId(String),
    InvalidObjectiveDefinition(String),
    EmptyIncumbentFrontier,
    DuplicateIncumbentId(String),
    InvalidObjectiveVector(String),
    DuplicateAcceptedEvaluation(String),
    MissingAcceptedEvaluation(String),
    UnexpectedAcceptedEvaluation(String),
    AcceptedOutcomeMismatch(String),
    DuplicateStratumEvaluations(String),
    MissingStratumEvaluations(String),
    UnexpectedStratumEvaluations(String),
    StratumEvaluationCountMismatch(String),
    DuplicateRejectedEvaluation(String),
    RejectedSampleIdentityMismatch {
        stratum_id: String,
        sample_index: usize,
    },
    RejectedOutcomeMismatch {
        stratum_id: String,
        sample_index: usize,
    },
    EmptyEvidenceRef(String),
    IncumbentFrontierNotEpsilonNondominated {
        first_candidate_id: String,
        second_candidate_id: String,
    },
    IntegerOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionOpportunityDisposition {
    ConfirmedOpportunity,
    ConfirmedNoOpportunity,
    Unresolved,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionOpportunityStratumReport {
    pub stratum_id: String,
    pub population_size: u64,
    pub sample_size: u64,
    pub inclusion_probability: f64,
    pub confirmed_opportunities_in_sample: u64,
    pub confirmed_no_opportunities_in_sample: u64,
    pub unresolved_in_sample: u64,
    pub potential_opportunities_in_sample: u64,
    pub upper_missed_opportunity_count: u64,
    pub per_stratum_error_probability: f64,
    pub is_full_census: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionOpportunityReport {
    pub decision_rule: &'static str,
    pub campaign_id: String,
    pub familywise_confidence_level: f64,
    pub target_opportunity_recall: f64,
    pub accepted_confirmed_opportunities: u64,
    pub accepted_confirmed_no_opportunity: u64,
    pub accepted_unresolved: u64,
    pub rejected_population: u64,
    pub audited_rejected_candidates: u64,
    pub rejected_opportunities_observed: u64,
    pub rejected_unresolved_observed: u64,
    pub missed_decision_opportunities_upper_bound: u64,
    /// Conditional on the frozen objective vector and incumbent frontier.
    /// This is not global Pareto recall.
    pub conditional_opportunity_recall_lower_bound: Option<f64>,
    pub opportunity_target_met: bool,
    /// Feasibility and decision-opportunity recall remain separate metrics.
    pub feasible_recall_lower_bound: Option<f64>,
    pub feasibility_target_met: bool,
    pub strata: Vec<DecisionOpportunityStratumReport>,
}

/// Evaluate decision opportunities relative to a frozen incumbent frontier.
///
/// A confirmed opportunity must be feasible and within the declared model
/// domain, must not be epsilon-dominated by any incumbent, must not be
/// epsilon-equivalent to an incumbent, and must improve at least one objective
/// by more than its declared epsilon against at least one incumbent. This
/// recognizes new material trade-offs without collapsing objectives into a
/// single scalar.
///
/// Failed, non-converged, unsupported, inconclusive or out-of-domain evaluations
/// remain unresolved and therefore count as potential missed opportunities in
/// the conservative rejected-population bound.
pub fn audit_decision_opportunities(
    plan: &DecisionOpportunityPlan,
) -> Result<DecisionOpportunityReport, Vec<DecisionOpportunityError>> {
    let feasibility_report = audit_rejected_candidates(&plan.base_plan)
        .map_err(|errors| vec![DecisionOpportunityError::BaseAuditRefused(errors)])?;

    let mut errors = Vec::new();
    if plan.campaign_id.trim().is_empty() {
        errors.push(DecisionOpportunityError::EmptyIdentity("campaign_id"));
    }
    for (value, label) in [
        (&plan.objective_manifest_sha256, "objective_manifest_sha256"),
        (&plan.opportunity_rule_sha256, "opportunity_rule_sha256"),
        (&plan.incumbent_frontier_sha256, "incumbent_frontier_sha256"),
    ] {
        if !is_sha256(value) {
            errors.push(DecisionOpportunityError::InvalidSha256(label));
        }
    }
    if !plan.target_opportunity_recall.is_finite()
        || plan.target_opportunity_recall <= 0.0
        || plan.target_opportunity_recall > 1.0
    {
        errors.push(DecisionOpportunityError::InvalidOpportunityRecallTarget);
    }
    if plan.objective_definitions.is_empty() {
        errors.push(DecisionOpportunityError::EmptyObjectiveDefinitions);
    }

    let mut objective_ids = BTreeSet::new();
    for definition in &plan.objective_definitions {
        if definition.objective_id.trim().is_empty()
            || definition.unit.trim().is_empty()
            || !definition.epsilon.is_finite()
            || definition.epsilon < 0.0
        {
            errors.push(DecisionOpportunityError::InvalidObjectiveDefinition(
                definition.objective_id.clone(),
            ));
        }
        if !objective_ids.insert(definition.objective_id.as_str()) {
            errors.push(DecisionOpportunityError::DuplicateObjectiveId(
                definition.objective_id.clone(),
            ));
        }
    }

    if plan.incumbent_frontier.is_empty() {
        errors.push(DecisionOpportunityError::EmptyIncumbentFrontier);
    }
    let mut incumbent_ids = BTreeSet::new();
    for point in &plan.incumbent_frontier {
        if point.candidate_id.trim().is_empty() {
            errors.push(DecisionOpportunityError::EmptyIdentity(
                "incumbent_frontier.candidate_id",
            ));
        }
        if !incumbent_ids.insert(point.candidate_id.as_str()) {
            errors.push(DecisionOpportunityError::DuplicateIncumbentId(
                point.candidate_id.clone(),
            ));
        }
        if !valid_objective_vector(&point.objectives, &plan.objective_definitions) {
            errors.push(DecisionOpportunityError::InvalidObjectiveVector(
                point.candidate_id.clone(),
            ));
        }
    }

    for first_index in 0..plan.incumbent_frontier.len() {
        for second_index in first_index + 1..plan.incumbent_frontier.len() {
            let first = &plan.incumbent_frontier[first_index];
            let second = &plan.incumbent_frontier[second_index];
            let first_dominates =
                epsilon_dominates(&first.objectives, &second.objectives, &plan.objective_definitions);
            let second_dominates =
                epsilon_dominates(&second.objectives, &first.objectives, &plan.objective_definitions);
            if first_dominates || second_dominates {
                errors.push(
                    DecisionOpportunityError::IncumbentFrontierNotEpsilonNondominated {
                        first_candidate_id: first.candidate_id.clone(),
                        second_candidate_id: second.candidate_id.clone(),
                    },
                );
            }
        }
    }

    let accepted_expected: BTreeMap<&str, AuditOutcome> = plan
        .base_plan
        .accepted
        .candidates
        .iter()
        .map(|candidate| (candidate.candidate_id.as_str(), candidate.outcome))
        .collect();
    let mut accepted_results = BTreeMap::new();
    for evaluation in &plan.accepted_evaluations {
        if evaluation.evidence_ref.trim().is_empty() {
            errors.push(DecisionOpportunityError::EmptyEvidenceRef(
                evaluation.candidate_id.clone(),
            ));
        }
        if !accepted_expected.contains_key(evaluation.candidate_id.as_str()) {
            errors.push(DecisionOpportunityError::UnexpectedAcceptedEvaluation(
                evaluation.candidate_id.clone(),
            ));
        }
        if let Some(expected) = accepted_expected.get(evaluation.candidate_id.as_str())
            && *expected != evaluation.reference_outcome
        {
            errors.push(DecisionOpportunityError::AcceptedOutcomeMismatch(
                evaluation.candidate_id.clone(),
            ));
        }
        let needs_objectives = evaluation.reference_outcome == AuditOutcome::ConfirmedFeasible
            && evaluation.within_declared_domain;
        if !evaluation.objectives.is_empty()
            && !valid_objective_vector(&evaluation.objectives, &plan.objective_definitions)
        {
            errors.push(DecisionOpportunityError::InvalidObjectiveVector(
                evaluation.candidate_id.clone(),
            ));
        }
        if needs_objectives && evaluation.objectives.is_empty() {
            errors.push(DecisionOpportunityError::InvalidObjectiveVector(
                evaluation.candidate_id.clone(),
            ));
        }
        if accepted_results
            .insert(evaluation.candidate_id.as_str(), evaluation)
            .is_some()
        {
            errors.push(DecisionOpportunityError::DuplicateAcceptedEvaluation(
                evaluation.candidate_id.clone(),
            ));
        }
    }
    for candidate_id in accepted_expected.keys() {
        if !accepted_results.contains_key(candidate_id) {
            errors.push(DecisionOpportunityError::MissingAcceptedEvaluation(
                (*candidate_id).to_string(),
            ));
        }
    }

    let expected_strata: BTreeMap<&str, &super::RejectStratum> = plan
        .base_plan
        .strata
        .iter()
        .map(|stratum| (stratum.stratum_id.as_str(), stratum))
        .collect();
    let mut supplied_strata = BTreeMap::new();
    for stratum in &plan.rejected_evaluations {
        if !expected_strata.contains_key(stratum.stratum_id.as_str()) {
            errors.push(DecisionOpportunityError::UnexpectedStratumEvaluations(
                stratum.stratum_id.clone(),
            ));
        }
        if supplied_strata
            .insert(stratum.stratum_id.as_str(), stratum)
            .is_some()
        {
            errors.push(DecisionOpportunityError::DuplicateStratumEvaluations(
                stratum.stratum_id.clone(),
            ));
        }
    }

    for (stratum_id, expected) in &expected_strata {
        let Some(supplied) = supplied_strata.get(stratum_id) else {
            errors.push(DecisionOpportunityError::MissingStratumEvaluations(
                (*stratum_id).to_string(),
            ));
            continue;
        };
        if supplied.evaluations.len() != expected.samples.len() {
            errors.push(DecisionOpportunityError::StratumEvaluationCountMismatch(
                (*stratum_id).to_string(),
            ));
            continue;
        }
        let mut seen = BTreeSet::new();
        for (sample_index, (base_sample, evaluation)) in expected
            .samples
            .iter()
            .zip(&supplied.evaluations)
            .enumerate()
        {
            if evaluation.evidence_ref.trim().is_empty() {
                errors.push(DecisionOpportunityError::EmptyEvidenceRef(
                    evaluation.candidate_id.clone(),
                ));
            }
            if base_sample.candidate_id != evaluation.candidate_id
                || !seen.insert(evaluation.candidate_id.as_str())
            {
                errors.push(DecisionOpportunityError::RejectedSampleIdentityMismatch {
                    stratum_id: (*stratum_id).to_string(),
                    sample_index,
                });
            }
            if base_sample.outcome != evaluation.reference_outcome {
                errors.push(DecisionOpportunityError::RejectedOutcomeMismatch {
                    stratum_id: (*stratum_id).to_string(),
                    sample_index,
                });
            }
            let needs_objectives =
                evaluation.reference_outcome == AuditOutcome::ConfirmedFeasible
                    && evaluation.within_declared_domain;
            if !evaluation.objectives.is_empty()
                && !valid_objective_vector(&evaluation.objectives, &plan.objective_definitions)
            {
                errors.push(DecisionOpportunityError::InvalidObjectiveVector(
                    evaluation.candidate_id.clone(),
                ));
            }
            if needs_objectives && evaluation.objectives.is_empty() {
                errors.push(DecisionOpportunityError::InvalidObjectiveVector(
                    evaluation.candidate_id.clone(),
                ));
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    let mut accepted_opportunities = 0_u64;
    let mut accepted_no_opportunity = 0_u64;
    let mut accepted_unresolved = 0_u64;
    for evaluation in &plan.accepted_evaluations {
        match classify_evaluation(evaluation, &plan.incumbent_frontier, &plan.objective_definitions) {
            DecisionOpportunityDisposition::ConfirmedOpportunity => accepted_opportunities += 1,
            DecisionOpportunityDisposition::ConfirmedNoOpportunity => {
                accepted_no_opportunity += 1;
            }
            DecisionOpportunityDisposition::Unresolved => accepted_unresolved += 1,
        }
    }

    let strata_count = plan.base_plan.strata.len();
    let familywise_alpha = 1.0 - plan.base_plan.familywise_confidence_level;
    let per_stratum_alpha = if strata_count == 0 {
        familywise_alpha
    } else {
        familywise_alpha / strata_count as f64
    };

    let mut strata_reports = Vec::with_capacity(strata_count);
    let mut rejected_population = 0_u64;
    let mut audited_rejected = 0_u64;
    let mut rejected_opportunities_observed = 0_u64;
    let mut rejected_unresolved_observed = 0_u64;
    let mut missed_opportunities_upper = 0_u64;

    for base_stratum in &plan.base_plan.strata {
        let supplied = supplied_strata
            .get(base_stratum.stratum_id.as_str())
            .expect("validated stratum result must exist");

        let mut opportunities = 0_u64;
        let mut no_opportunities = 0_u64;
        let mut unresolved = 0_u64;
        for evaluation in &supplied.evaluations {
            match classify_evaluation(evaluation, &plan.incumbent_frontier, &plan.objective_definitions) {
                DecisionOpportunityDisposition::ConfirmedOpportunity => opportunities += 1,
                DecisionOpportunityDisposition::ConfirmedNoOpportunity => no_opportunities += 1,
                DecisionOpportunityDisposition::Unresolved => unresolved += 1,
            }
        }
        let potential = opportunities
            .checked_add(unresolved)
            .ok_or_else(|| vec![DecisionOpportunityError::IntegerOverflow])?;
        let upper = upper_hypergeometric_population_successes(
            base_stratum.rejected_candidate_ids.len(),
            base_stratum.sample_size,
            usize::try_from(potential)
                .map_err(|_| vec![DecisionOpportunityError::IntegerOverflow])?,
            per_stratum_alpha,
        )
        .map_err(|_| vec![DecisionOpportunityError::IntegerOverflow])?;

        let population_u64 = u64::try_from(base_stratum.rejected_candidate_ids.len())
            .map_err(|_| vec![DecisionOpportunityError::IntegerOverflow])?;
        let sample_u64 = u64::try_from(base_stratum.sample_size)
            .map_err(|_| vec![DecisionOpportunityError::IntegerOverflow])?;
        let upper_u64 =
            u64::try_from(upper).map_err(|_| vec![DecisionOpportunityError::IntegerOverflow])?;

        rejected_population = rejected_population
            .checked_add(population_u64)
            .ok_or_else(|| vec![DecisionOpportunityError::IntegerOverflow])?;
        audited_rejected = audited_rejected
            .checked_add(sample_u64)
            .ok_or_else(|| vec![DecisionOpportunityError::IntegerOverflow])?;
        rejected_opportunities_observed = rejected_opportunities_observed
            .checked_add(opportunities)
            .ok_or_else(|| vec![DecisionOpportunityError::IntegerOverflow])?;
        rejected_unresolved_observed = rejected_unresolved_observed
            .checked_add(unresolved)
            .ok_or_else(|| vec![DecisionOpportunityError::IntegerOverflow])?;
        missed_opportunities_upper = missed_opportunities_upper
            .checked_add(upper_u64)
            .ok_or_else(|| vec![DecisionOpportunityError::IntegerOverflow])?;

        strata_reports.push(DecisionOpportunityStratumReport {
            stratum_id: base_stratum.stratum_id.clone(),
            population_size: population_u64,
            sample_size: sample_u64,
            inclusion_probability: if population_u64 == 0 {
                0.0
            } else {
                sample_u64 as f64 / population_u64 as f64
            },
            confirmed_opportunities_in_sample: opportunities,
            confirmed_no_opportunities_in_sample: no_opportunities,
            unresolved_in_sample: unresolved,
            potential_opportunities_in_sample: potential,
            upper_missed_opportunity_count: upper_u64,
            per_stratum_error_probability: per_stratum_alpha,
            is_full_census: base_stratum.sample_size == base_stratum.rejected_candidate_ids.len(),
        });
    }

    let opportunity_denominator = accepted_opportunities
        .checked_add(accepted_unresolved)
        .and_then(|value| value.checked_add(missed_opportunities_upper))
        .ok_or_else(|| vec![DecisionOpportunityError::IntegerOverflow])?;
    let conditional_recall = if opportunity_denominator == 0 {
        None
    } else {
        Some(accepted_opportunities as f64 / opportunity_denominator as f64)
    };
    let opportunity_target_met = conditional_recall
        .map(|value| value >= plan.target_opportunity_recall)
        .unwrap_or(false);

    Ok(DecisionOpportunityReport {
        decision_rule: DECISION_RULE,
        campaign_id: plan.campaign_id.clone(),
        familywise_confidence_level: plan.base_plan.familywise_confidence_level,
        target_opportunity_recall: plan.target_opportunity_recall,
        accepted_confirmed_opportunities: accepted_opportunities,
        accepted_confirmed_no_opportunity: accepted_no_opportunity,
        accepted_unresolved,
        rejected_population,
        audited_rejected_candidates: audited_rejected,
        rejected_opportunities_observed,
        rejected_unresolved_observed,
        missed_decision_opportunities_upper_bound: missed_opportunities_upper,
        conditional_opportunity_recall_lower_bound: conditional_recall,
        opportunity_target_met,
        feasible_recall_lower_bound: feasibility_report.feasible_recall_lower_bound,
        feasibility_target_met: feasibility_report.target_met,
        strata: strata_reports,
    })
}

fn classify_evaluation(
    evaluation: &DecisionEvaluation,
    incumbent_frontier: &[IncumbentFrontierPoint],
    definitions: &[ObjectiveDefinition],
) -> DecisionOpportunityDisposition {
    if !evaluation.within_declared_domain {
        return DecisionOpportunityDisposition::Unresolved;
    }
    match evaluation.reference_outcome {
        AuditOutcome::ConfirmedInfeasible => {
            DecisionOpportunityDisposition::ConfirmedNoOpportunity
        }
        AuditOutcome::ConfirmedFeasible => {
            if is_epsilon_frontier_extension(
                &evaluation.candidate_id,
                &evaluation.objectives,
                incumbent_frontier,
                definitions,
            ) {
                DecisionOpportunityDisposition::ConfirmedOpportunity
            } else {
                DecisionOpportunityDisposition::ConfirmedNoOpportunity
            }
        }
        AuditOutcome::Failed
        | AuditOutcome::NonConverged
        | AuditOutcome::Unsupported
        | AuditOutcome::Inconclusive => DecisionOpportunityDisposition::Unresolved,
    }
}

fn is_epsilon_frontier_extension(
    candidate_id: &str,
    candidate: &[ObjectiveValue],
    frontier: &[IncumbentFrontierPoint],
    definitions: &[ObjectiveDefinition],
) -> bool {
    if frontier.iter().any(|point| point.candidate_id == candidate_id) {
        return false;
    }
    if frontier
        .iter()
        .any(|point| epsilon_equivalent(candidate, &point.objectives, definitions))
    {
        return false;
    }
    if frontier
        .iter()
        .any(|point| epsilon_dominates(&point.objectives, candidate, definitions))
    {
        return false;
    }

    frontier.iter().any(|point| {
        candidate
            .iter()
            .zip(&point.objectives)
            .zip(definitions)
            .any(|((candidate_value, incumbent_value), definition)| {
                oriented_value(candidate_value, definition)
                    > oriented_value(incumbent_value, definition) + definition.epsilon
            })
    })
}

fn epsilon_equivalent(
    first: &[ObjectiveValue],
    second: &[ObjectiveValue],
    definitions: &[ObjectiveDefinition],
) -> bool {
    first
        .iter()
        .zip(second)
        .zip(definitions)
        .all(|((first_value, second_value), definition)| {
            (oriented_value(first_value, definition) - oriented_value(second_value, definition))
                .abs()
                <= definition.epsilon
        })
}

fn epsilon_dominates(
    first: &[ObjectiveValue],
    second: &[ObjectiveValue],
    definitions: &[ObjectiveDefinition],
) -> bool {
    let no_worse = first
        .iter()
        .zip(second)
        .zip(definitions)
        .all(|((first_value, second_value), definition)| {
            oriented_value(first_value, definition)
                >= oriented_value(second_value, definition) - definition.epsilon
        });
    let materially_better = first
        .iter()
        .zip(second)
        .zip(definitions)
        .any(|((first_value, second_value), definition)| {
            oriented_value(first_value, definition)
                > oriented_value(second_value, definition) + definition.epsilon
        });
    no_worse && materially_better
}

fn oriented_value(value: &ObjectiveValue, definition: &ObjectiveDefinition) -> f64 {
    match definition.direction {
        ObjectiveDirection::Maximize => value.value,
        ObjectiveDirection::Minimize => -value.value,
    }
}

fn valid_objective_vector(
    values: &[ObjectiveValue],
    definitions: &[ObjectiveDefinition],
) -> bool {
    if values.len() != definitions.len() {
        return false;
    }
    values.iter().zip(definitions).all(|(value, definition)| {
        value.objective_id == definition.objective_id
            && value.unit == definition.unit
            && value.value.is_finite()
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AcceptedCandidate, AcceptedCandidateManifest, AuditSample, FrozenCandidateUniverse,
        RejectStratum,
    };

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn objective_values(cost: f64, availability: f64) -> Vec<ObjectiveValue> {
        vec![
            ObjectiveValue {
                objective_id: "lifetime-cost".to_string(),
                unit: "MUSD_2026".to_string(),
                value: cost,
            },
            ObjectiveValue {
                objective_id: "availability".to_string(),
                unit: "fraction".to_string(),
                value: availability,
            },
        ]
    }

    fn feasible_eval(
        candidate_id: &str,
        cost: f64,
        availability: f64,
    ) -> DecisionEvaluation {
        DecisionEvaluation {
            candidate_id: candidate_id.to_string(),
            reference_outcome: AuditOutcome::ConfirmedFeasible,
            within_declared_domain: true,
            objectives: objective_values(cost, availability),
            evidence_ref: format!("reference-run-{candidate_id}"),
        }
    }

    fn non_objective_eval(
        candidate_id: &str,
        outcome: AuditOutcome,
        within_domain: bool,
    ) -> DecisionEvaluation {
        DecisionEvaluation {
            candidate_id: candidate_id.to_string(),
            reference_outcome: outcome,
            within_declared_domain: within_domain,
            objectives: Vec::new(),
            evidence_ref: format!("reference-run-{candidate_id}"),
        }
    }

    fn sample_ids(stratum_id: &str, population: &[String], sample_size: usize, seed: u64) -> Vec<String> {
        super::super::deterministic_audit_sample_ids(stratum_id, population, sample_size, seed).unwrap()
    }

    fn base_plan() -> RejectAuditPlan {
        let seed = 0x1234_5678;
        let accepted = vec![
            AcceptedCandidate {
                candidate_id: "accepted-0".to_string(),
                outcome: AuditOutcome::ConfirmedFeasible,
                evidence_ref: "accepted-evidence-0".to_string(),
            },
            AcceptedCandidate {
                candidate_id: "accepted-1".to_string(),
                outcome: AuditOutcome::Failed,
                evidence_ref: "accepted-evidence-1".to_string(),
            },
        ];
        let rejected = vec![
            "rejected-0".to_string(),
            "rejected-1".to_string(),
            "rejected-2".to_string(),
            "rejected-3".to_string(),
        ];
        let selected = sample_ids("reject-all", &rejected, 4, seed);
        let samples = selected
            .into_iter()
            .map(|candidate_id| AuditSample {
                candidate_id,
                outcome: AuditOutcome::ConfirmedInfeasible,
                evidence_ref: "sample-evidence".to_string(),
            })
            .collect();
        let mut candidate_ids = vec!["accepted-0".to_string(), "accepted-1".to_string()];
        candidate_ids.extend(rejected.clone());
        candidate_ids.sort();

        RejectAuditPlan {
            frozen_universe: FrozenCandidateUniverse {
                candidate_ids,
                manifest_sha256: digest('a'),
            },
            sampling_seed: seed,
            seed_commitment_ref: "seed-commitment".to_string(),
            seed_committed_before_outcomes: true,
            familywise_confidence_level: 0.95,
            target_feasible_recall: 0.99,
            accepted: AcceptedCandidateManifest {
                candidates: accepted,
                manifest_sha256: digest('b'),
                evidence_ref: "accepted-manifest".to_string(),
            },
            strata: vec![RejectStratum {
                stratum_id: "reject-all".to_string(),
                stratum_definition_sha256: digest('c'),
                population_manifest_sha256: digest('d'),
                selection_receipt_ref: "selection-receipt".to_string(),
                rejected_candidate_ids: rejected,
                sample_size: 4,
                samples,
            }],
        }
    }

    fn definitions() -> Vec<ObjectiveDefinition> {
        vec![
            ObjectiveDefinition {
                objective_id: "lifetime-cost".to_string(),
                unit: "MUSD_2026".to_string(),
                direction: ObjectiveDirection::Minimize,
                epsilon: 1.0,
            },
            ObjectiveDefinition {
                objective_id: "availability".to_string(),
                unit: "fraction".to_string(),
                direction: ObjectiveDirection::Maximize,
                epsilon: 0.01,
            },
        ]
    }

    fn frontier() -> Vec<IncumbentFrontierPoint> {
        vec![
            IncumbentFrontierPoint {
                candidate_id: "incumbent-a".to_string(),
                objectives: objective_values(100.0, 0.90),
            },
            IncumbentFrontierPoint {
                candidate_id: "incumbent-b".to_string(),
                objectives: objective_values(120.0, 0.95),
            },
        ]
    }

    fn plan() -> DecisionOpportunityPlan {
        let mut base = base_plan();
        for sample in &mut base.strata[0].samples {
            sample.outcome = if sample.candidate_id == "rejected-0" {
                AuditOutcome::ConfirmedFeasible
            } else {
                AuditOutcome::ConfirmedInfeasible
            };
        }
        let rejected_evaluations = base.strata[0]
            .samples
            .iter()
            .map(|sample| {
                if sample.candidate_id == "rejected-0" {
                    feasible_eval(&sample.candidate_id, 95.0, 0.94)
                } else {
                    non_objective_eval(
                        &sample.candidate_id,
                        AuditOutcome::ConfirmedInfeasible,
                        true,
                    )
                }
            })
            .collect();

        DecisionOpportunityPlan {
            campaign_id: "campaign-1".to_string(),
            objective_manifest_sha256: digest('e'),
            opportunity_rule_sha256: digest('f'),
            incumbent_frontier_sha256: digest('1'),
            target_opportunity_recall: 0.95,
            base_plan: base,
            objective_definitions: definitions(),
            incumbent_frontier: frontier(),
            accepted_evaluations: vec![
                feasible_eval("accepted-0", 90.0, 0.93),
                non_objective_eval("accepted-1", AuditOutcome::Failed, false),
            ],
            rejected_evaluations: vec![StratumDecisionEvaluations {
                stratum_id: "reject-all".to_string(),
                evaluations: rejected_evaluations,
            }],
        }
    }

    #[test]
    fn detects_a_material_frontier_extension() {
        let report = audit_decision_opportunities(&plan()).unwrap();
        assert_eq!(report.accepted_confirmed_opportunities, 1);
        assert_eq!(report.accepted_unresolved, 1);
        assert_eq!(report.rejected_opportunities_observed, 1);
        assert_eq!(report.rejected_unresolved_observed, 0);
        assert_eq!(report.feasible_recall_lower_bound, Some(1.0 / 3.0));
        assert!(report.strata[0].is_full_census);
    }

    #[test]
    fn high_feasibility_recall_does_not_imply_frontier_opportunity_recall() {
        let mut test_plan = plan();
        test_plan.accepted_evaluations[0] = feasible_eval("accepted-0", 130.0, 0.70);
        let report = audit_decision_opportunities(&test_plan).unwrap();
        assert_eq!(report.accepted_confirmed_opportunities, 0);
        assert_eq!(report.accepted_unresolved, 1);
        assert_eq!(report.rejected_opportunities_observed, 1);
        assert_eq!(report.conditional_opportunity_recall_lower_bound, Some(0.0));
    }

    #[test]
    fn unresolved_and_out_of_domain_outcomes_remain_potential_opportunities() {
        let mut test_plan = plan();
        for evaluation in &mut test_plan.rejected_evaluations[0].evaluations {
            if evaluation.candidate_id == "rejected-1" {
                *evaluation = non_objective_eval(
                    "rejected-1",
                    AuditOutcome::NonConverged,
                    true,
                );
            }
            if evaluation.candidate_id == "rejected-2" {
                *evaluation = non_objective_eval(
                    "rejected-2",
                    AuditOutcome::ConfirmedInfeasible,
                    false,
                );
            }
        }
        let sample_1 = test_plan.base_plan.strata[0]
            .samples
            .iter()
            .position(|sample| sample.candidate_id == "rejected-1")
            .unwrap();
        let sample_2 = test_plan.base_plan.strata[0]
            .samples
            .iter()
            .position(|sample| sample.candidate_id == "rejected-2")
            .unwrap();
        test_plan.base_plan.strata[0].samples[sample_1].outcome = AuditOutcome::NonConverged;
        test_plan.base_plan.strata[0].samples[sample_2].outcome =
            AuditOutcome::ConfirmedInfeasible;

        let report = audit_decision_opportunities(&test_plan).unwrap();
        assert_eq!(report.rejected_unresolved_observed, 2);
        assert_eq!(report.strata[0].potential_opportunities_in_sample, 3);
    }

    #[test]
    fn a_candidate_epsilon_dominated_by_an_incumbent_is_not_an_opportunity() {
        let mut test_plan = plan();
        test_plan.rejected_evaluations[0].evaluations[0] =
            feasible_eval("rejected-0", 150.0, 0.70);
        test_plan.base_plan.strata[0].samples[0].outcome = AuditOutcome::ConfirmedFeasible;
        let report = audit_decision_opportunities(&test_plan).unwrap();
        assert_eq!(report.rejected_opportunities_observed, 0);
    }

    #[test]
    fn an_epsilon_equivalent_candidate_is_not_a_new_opportunity() {
        let mut test_plan = plan();
        test_plan.rejected_evaluations[0].evaluations[0] =
            feasible_eval("rejected-0", 100.5, 0.905);
        test_plan.base_plan.strata[0].samples[0].outcome = AuditOutcome::ConfirmedFeasible;
        let report = audit_decision_opportunities(&test_plan).unwrap();
        assert_eq!(report.rejected_opportunities_observed, 0);
    }

    #[test]
    fn objective_units_and_order_are_mandatory() {
        let mut test_plan = plan();
        test_plan.accepted_evaluations[0].objectives.swap(0, 1);
        let errors = audit_decision_opportunities(&test_plan).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            DecisionOpportunityError::InvalidObjectiveVector(_)
        )));
    }

    #[test]
    fn missing_rejected_sample_evaluation_is_rejected() {
        let mut test_plan = plan();
        test_plan.rejected_evaluations[0].evaluations.pop();
        let errors = audit_decision_opportunities(&test_plan).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            DecisionOpportunityError::StratumEvaluationCountMismatch(_)
        )));
    }

    #[test]
    fn rejected_sample_identity_order_must_match_frozen_draw() {
        let mut test_plan = plan();
        test_plan.rejected_evaluations[0].evaluations.swap(0, 1);
        let errors = audit_decision_opportunities(&test_plan).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            DecisionOpportunityError::RejectedSampleIdentityMismatch { .. }
        )));
    }

    #[test]
    fn reference_frontier_must_be_nondominated() {
        let mut test_plan = plan();
        test_plan.incumbent_frontier.push(IncumbentFrontierPoint {
            candidate_id: "dominated-incumbent".to_string(),
            objectives: objective_values(140.0, 0.80),
        });
        let errors = audit_decision_opportunities(&test_plan).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            DecisionOpportunityError::IncumbentFrontierNotEpsilonNondominated { .. }
        )));
    }

    #[test]
    fn candidate_identity_already_in_frontier_is_not_recounted() {
        let mut test_plan = plan();
        test_plan.rejected_evaluations[0].evaluations[0] =
            feasible_eval("incumbent-a", 90.0, 0.90);
        let errors = audit_decision_opportunities(&test_plan).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            DecisionOpportunityError::RejectedSampleIdentityMismatch { .. }
        )));
    }

    #[test]
    fn decision_recall_is_reported_separately_from_feasibility_recall() {
        let report = audit_decision_opportunities(&plan()).unwrap();
        assert!(report.feasible_recall_lower_bound.is_some());
        assert!(report.conditional_opportunity_recall_lower_bound.is_some());
        assert_eq!(report.decision_rule, DECISION_RULE);
        assert_eq!(report.campaign_id, "campaign-1");
    }

    #[test]
    fn zero_rejected_audit_sample_keeps_the_full_population_as_possible() {
        let mut test_plan = plan();
        test_plan.base_plan.strata[0].sample_size = 0;
        test_plan.base_plan.strata[0].samples.clear();
        test_plan.rejected_evaluations[0].evaluations.clear();
        let report = audit_decision_opportunities(&test_plan).unwrap();
        assert_eq!(
            report.missed_decision_opportunities_upper_bound,
            4
        );
    }
}
