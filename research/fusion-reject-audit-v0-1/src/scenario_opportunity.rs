//! Scenario-wise conditional decision-opportunity audit.
//!
//! This module reuses the existing frozen-candidate audit, finite-population
//! hypergeometric inversion and the canonical epsilon-frontier predicate. It
//! reports one bound per frozen scenario plus a conservative union bound for
//! candidates that could change the decision in any required scenario.
//!
//! Scenario outcomes are not averaged into one synthetic "robustness score".
//! An unresolved scenario remains unresolved, while an opportunity in any
//! scenario remains visible in the union metric.

use std::collections::{BTreeMap, BTreeSet};

use super::decision_opportunity::{
    IncumbentFrontierPoint, ObjectiveDefinition, ObjectiveValue, epsilon_dominates,
    epsilon_equivalent, is_epsilon_frontier_extension, valid_objective_vector,
};
use super::{
    AuditError, AuditOutcome, BOUND_METHOD, RejectAuditPlan, SAMPLING_ALGORITHM,
    audit_rejected_candidates, upper_hypergeometric_population_successes,
};

pub const SCENARIO_OPPORTUNITY_VERSION: &str = "fusion-scenario-opportunity/0.1";
pub const SCENARIO_OPPORTUNITY_RULE: &str = "epsilon-frontier-any-required-scenario-v1";

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioDefinition {
    pub scenario_id: String,
    /// Identity of all scenario inputs, conditions, model profile and scope.
    pub scenario_sha256: String,
    /// Identity of the high-fidelity incumbent frontier for this scenario.
    pub incumbent_frontier_sha256: String,
    pub incumbent_frontier: Vec<IncumbentFrontierPoint>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioEvaluation {
    pub scenario_id: String,
    pub run_id: String,
    pub reference_outcome: AuditOutcome,
    pub within_declared_domain: bool,
    pub objectives: Vec<ObjectiveValue>,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateScenarioBundle {
    pub candidate_id: String,
    /// Must follow the exact frozen scenario manifest order.
    pub evaluations: Vec<ScenarioEvaluation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StratumScenarioBundles {
    pub stratum_id: String,
    /// Must follow the exact frozen base-plan sample draw order.
    pub bundles: Vec<CandidateScenarioBundle>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioOpportunityPlan {
    pub campaign_id: String,
    pub scenario_manifest_sha256: String,
    pub objective_manifest_sha256: String,
    pub opportunity_rule_sha256: String,
    pub primary_scenario_id: String,
    pub target_opportunity_recall: f64,
    /// Owns the exact candidate universe, population partition, sample draw,
    /// base feasibility outcomes and primary-scenario evidence references.
    pub base_plan: RejectAuditPlan,
    /// Shared objective identities, units, directions and epsilons across cases.
    pub objective_definitions: Vec<ObjectiveDefinition>,
    /// Strictly ordered by scenario ID; the order is part of the contract.
    pub scenarios: Vec<ScenarioDefinition>,
    /// Complete evaluation bundles in accepted-manifest order.
    pub accepted_bundles: Vec<CandidateScenarioBundle>,
    /// Complete evaluation bundles in frozen stratum/sample order.
    pub rejected_strata: Vec<StratumScenarioBundles>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScenarioOpportunityError {
    BaseAuditRefused(Vec<AuditError>),
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    InvalidRecallTarget,
    EmptyObjectiveDefinitions,
    InvalidObjectiveDefinition(String),
    DuplicateObjectiveId(String),
    EmptyScenarioManifest,
    DuplicateScenarioId(String),
    ScenarioManifestNotSorted,
    UnknownPrimaryScenario(String),
    EmptyScenarioFrontier(String),
    DuplicateFrontierCandidate {
        scenario_id: String,
        candidate_id: String,
    },
    InvalidScenarioFrontier(String),
    InvalidObjectiveVector {
        scenario_id: String,
        candidate_id: String,
    },
    AcceptedBundleCountMismatch,
    RejectedStratumCountMismatch,
    DuplicateRejectedStratum(String),
    MissingRejectedStratum(String),
    UnexpectedRejectedStratum(String),
    RejectedBundleCountMismatch(String),
    AcceptedBundleIdentityMismatch {
        bundle_index: usize,
    },
    RejectedBundleIdentityMismatch {
        stratum_id: String,
        sample_index: usize,
    },
    BundleScenarioCountMismatch(String),
    BundleScenarioOrderMismatch {
        candidate_id: String,
        scenario_index: usize,
    },
    EmptyRunId(String),
    EmptyEvidenceRef(String),
    PrimaryOutcomeMismatch(String),
    PrimaryEvidenceMismatch(String),
    OutOfDomainOutcomeMustBeUnresolved {
        candidate_id: String,
        scenario_id: String,
    },
    DuplicateBundleCandidate(String),
    EmptyStratumBundles(String),
    ScenarioCountOverflow,
    IntegerOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScenarioDisposition {
    ConfirmedOpportunity,
    ConfirmedNoOpportunity,
    Unresolved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnionOpportunityDisposition {
    OpportunityInAtLeastOneScenario,
    NoOpportunityInAnyScenario,
    UnresolvedAcrossScenarioFamily,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioStratumAuditReport {
    pub stratum_id: String,
    pub scenario_id: String,
    pub stratum_definition_sha256: String,
    pub population_manifest_sha256: String,
    pub selection_receipt_ref: String,
    pub population_size: u64,
    pub sample_size: u64,
    pub inclusion_probability: f64,
    pub confirmed_opportunities_in_sample: u64,
    pub confirmed_no_opportunities_in_sample: u64,
    pub unresolved_in_sample: u64,
    pub potential_opportunities_in_sample: u64,
    pub upper_missed_opportunity_count: u64,
    pub per_hypothesis_error_probability: f64,
    pub is_full_census: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioRecallReport {
    pub scenario_id: String,
    pub scenario_sha256: String,
    pub incumbent_frontier_sha256: String,
    pub accepted_confirmed_opportunities: u64,
    pub accepted_confirmed_no_opportunity: u64,
    pub accepted_unresolved: u64,
    pub rejected_population: u64,
    pub audited_rejected_candidates: u64,
    pub rejected_opportunities_observed: u64,
    pub rejected_unresolved_observed: u64,
    pub missed_opportunities_upper_bound: u64,
    pub conditional_opportunity_recall_lower_bound: Option<f64>,
    pub target_met: bool,
    pub per_hypothesis_error_probability: f64,
    pub strata: Vec<ScenarioStratumAuditReport>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnionStratumAuditReport {
    pub stratum_id: String,
    pub stratum_definition_sha256: String,
    pub population_manifest_sha256: String,
    pub selection_receipt_ref: String,
    pub population_size: u64,
    pub sample_size: u64,
    pub inclusion_probability: f64,
    pub confirmed_union_opportunities_in_sample: u64,
    pub confirmed_union_no_opportunity_in_sample: u64,
    pub unresolved_union_in_sample: u64,
    pub potential_union_opportunities_in_sample: u64,
    pub upper_missed_union_opportunity_count: u64,
    pub per_hypothesis_error_probability: f64,
    pub is_full_census: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnionOpportunityRecallReport {
    pub accepted_confirmed_opportunities: u64,
    pub accepted_confirmed_no_opportunity: u64,
    pub accepted_unresolved: u64,
    pub rejected_population: u64,
    pub audited_rejected_candidates: u64,
    pub rejected_opportunities_observed: u64,
    pub rejected_unresolved_observed: u64,
    pub missed_union_opportunities_upper_bound: u64,
    pub conditional_any_scenario_recall_lower_bound: Option<f64>,
    pub target_met: bool,
    pub per_hypothesis_error_probability: f64,
    pub strata: Vec<UnionStratumAuditReport>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioEvaluationReceiptRef {
    pub candidate_id: String,
    pub scenario_id: String,
    pub run_id: String,
    pub evidence_ref: String,
    pub reference_outcome: AuditOutcome,
    pub within_declared_domain: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioOpportunityReport {
    pub contract_version: &'static str,
    pub bound_method: &'static str,
    pub sampling_algorithm: &'static str,
    pub opportunity_rule: &'static str,
    pub campaign_id: String,
    pub scenario_manifest_sha256: String,
    pub objective_manifest_sha256: String,
    pub opportunity_rule_sha256: String,
    pub frozen_universe_manifest_sha256: String,
    pub accepted_manifest_sha256: String,
    pub primary_scenario_id: String,
    pub familywise_confidence_level: f64,
    pub target_opportunity_recall: f64,
    /// Number of simultaneous bounds: every stratum × (feasibility, union,
    /// plus every scenario-specific claim).
    pub simultaneous_bound_count: u64,
    pub familywise_error_probability: f64,
    pub feasible_recall_lower_bound: Option<f64>,
    pub feasibility_target_met: bool,
    pub union: UnionOpportunityRecallReport,
    pub scenarios: Vec<ScenarioRecallReport>,
    /// Exact ordered run/evidence references consumed by this report.
    pub evaluation_receipts: Vec<ScenarioEvaluationReceiptRef>,
}

/// Audit candidate opportunities across a frozen family of required scenarios.
///
/// A union opportunity is confirmed if any scenario confirms one. A union
/// non-opportunity is confirmed only if every scenario confirms no opportunity.
/// All other unions are unresolved. The union and every scenario-specific
/// stratum bound share one Bonferroni familywise allocation.
pub fn audit_scenario_opportunities(
    plan: &ScenarioOpportunityPlan,
) -> Result<ScenarioOpportunityReport, Vec<ScenarioOpportunityError>> {
    // First run the base audit at its declared confidence to validate its
    // frozen partition, sampling receipts and numeric inputs.
    let base_validation_report = audit_rejected_candidates(&plan.base_plan)
        .map_err(|errors| vec![ScenarioOpportunityError::BaseAuditRefused(errors)])?;

    let mut errors = validate_plan(plan);
    if !errors.is_empty() {
        return Err(std::mem::take(&mut errors));
    }

    let scenario_count = plan.scenarios.len();
    let stratum_count = plan.base_plan.strata.len();
    // One simultaneous hypothesis per stratum for feasibility, one for the
    // any-scenario union, and one for every scenario-specific opportunity bound.
    let hypothesis_count = stratum_count
        .checked_mul(
            scenario_count
                .checked_add(2)
                .ok_or_else(|| vec![ScenarioOpportunityError::ScenarioCountOverflow])?,
        )
        .ok_or_else(|| vec![ScenarioOpportunityError::ScenarioCountOverflow])?;
    let familywise_alpha = 1.0 - plan.base_plan.familywise_confidence_level;
    let per_hypothesis_alpha = if hypothesis_count == 0 {
        familywise_alpha
    } else {
        familywise_alpha / hypothesis_count as f64
    };

    // The inherited feasibility metric is part of this combined report, so
    // tighten its per-stratum allocation to the same alpha used by every
    // scenario and union hypothesis. This makes all reported bounds jointly
    // covered by the original familywise confidence level.
    let feasibility_report = if stratum_count == 0 {
        base_validation_report
    } else {
        let mut adjusted_base_plan = plan.base_plan.clone();
        adjusted_base_plan.familywise_confidence_level =
            1.0 - per_hypothesis_alpha * stratum_count as f64;
        audit_rejected_candidates(&adjusted_base_plan)
            .map_err(|errors| vec![ScenarioOpportunityError::BaseAuditRefused(errors)])?
    };

    let accepted_map: BTreeMap<&str, &CandidateScenarioBundle> = plan
        .accepted_bundles
        .iter()
        .map(|bundle| (bundle.candidate_id.as_str(), bundle))
        .collect();
    let rejected_map: BTreeMap<&str, &StratumScenarioBundles> = plan
        .rejected_strata
        .iter()
        .map(|stratum| (stratum.stratum_id.as_str(), stratum))
        .collect();

    let mut accepted_scenario_counts = vec![(0_u64, 0_u64, 0_u64); scenario_count];
    let mut accepted_union_counts = (0_u64, 0_u64, 0_u64);
    for candidate in &plan.base_plan.accepted.candidates {
        let bundle = accepted_map
            .get(candidate.candidate_id.as_str())
            .expect("validated accepted bundle must exist");
        let mut scenario_dispositions = Vec::with_capacity(scenario_count);
        for (index, (evaluation, scenario)) in
            bundle.evaluations.iter().zip(&plan.scenarios).enumerate()
        {
            let disposition = classify_scenario_evaluation(
                &bundle.candidate_id,
                evaluation,
                scenario,
                &plan.objective_definitions,
            );
            increment_counts(&mut accepted_scenario_counts[index], disposition);
            scenario_dispositions.push(disposition);
        }
        increment_union_counts(
            &mut accepted_union_counts,
            classify_union(&scenario_dispositions),
        );
    }

    let mut scenario_reports = Vec::with_capacity(scenario_count);
    let mut scenario_upper_totals = vec![0_u64; scenario_count];
    let mut scenario_population_totals = vec![0_u64; scenario_count];
    let mut scenario_sample_totals = vec![0_u64; scenario_count];
    let mut scenario_observed_opportunity_totals = vec![0_u64; scenario_count];
    let mut scenario_observed_unresolved_totals = vec![0_u64; scenario_count];
    let mut scenario_stratum_reports = vec![Vec::with_capacity(stratum_count); scenario_count];

    let mut union_upper_total = 0_u64;
    let mut union_population_total = 0_u64;
    let mut union_sample_total = 0_u64;
    let mut union_observed_opportunities = 0_u64;
    let mut union_observed_unresolved = 0_u64;
    let mut union_stratum_reports = Vec::with_capacity(stratum_count);

    for base_stratum in &plan.base_plan.strata {
        let bundle_group = rejected_map
            .get(base_stratum.stratum_id.as_str())
            .expect("validated rejected stratum bundle must exist");
        let population_size = base_stratum.rejected_candidate_ids.len();
        let sample_size = base_stratum.sample_size;
        let population_u64 = u64::try_from(population_size)
            .map_err(|_| vec![ScenarioOpportunityError::IntegerOverflow])?;
        let sample_u64 = u64::try_from(sample_size)
            .map_err(|_| vec![ScenarioOpportunityError::IntegerOverflow])?;

        let mut per_scenario_dispositions: Vec<Vec<ScenarioDisposition>> =
            vec![Vec::with_capacity(sample_size); scenario_count];
        let mut union_dispositions = Vec::with_capacity(sample_size);

        for bundle in &bundle_group.bundles {
            let mut candidate_dispositions = Vec::with_capacity(scenario_count);
            for (index, (evaluation, scenario)) in
                bundle.evaluations.iter().zip(&plan.scenarios).enumerate()
            {
                let disposition = classify_scenario_evaluation(
                    &bundle.candidate_id,
                    evaluation,
                    scenario,
                    &plan.objective_definitions,
                );
                per_scenario_dispositions[index].push(disposition);
                candidate_dispositions.push(disposition);
            }
            union_dispositions.push(classify_union(&candidate_dispositions));
        }

        for (index, scenario) in plan.scenarios.iter().enumerate() {
            let mut counts = (0_u64, 0_u64, 0_u64);
            for disposition in &per_scenario_dispositions[index] {
                increment_counts(&mut counts, *disposition);
            }
            let potential = counts
                .0
                .checked_add(counts.2)
                .ok_or_else(|| vec![ScenarioOpportunityError::IntegerOverflow])?;
            let upper = upper_hypergeometric_population_successes(
                population_size,
                sample_size,
                usize::try_from(potential)
                    .map_err(|_| vec![ScenarioOpportunityError::IntegerOverflow])?,
                per_hypothesis_alpha,
            )
            .map_err(|_| vec![ScenarioOpportunityError::IntegerOverflow])?;
            let upper_u64 = u64::try_from(upper)
                .map_err(|_| vec![ScenarioOpportunityError::IntegerOverflow])?;

            scenario_upper_totals[index] = checked_add(scenario_upper_totals[index], upper_u64)?;
            scenario_population_totals[index] =
                checked_add(scenario_population_totals[index], population_u64)?;
            scenario_sample_totals[index] = checked_add(scenario_sample_totals[index], sample_u64)?;
            scenario_observed_opportunity_totals[index] =
                checked_add(scenario_observed_opportunity_totals[index], counts.0)?;
            scenario_observed_unresolved_totals[index] =
                checked_add(scenario_observed_unresolved_totals[index], counts.2)?;

            scenario_stratum_reports[index].push(ScenarioStratumAuditReport {
                stratum_id: base_stratum.stratum_id.clone(),
                scenario_id: scenario.scenario_id.clone(),
                stratum_definition_sha256: base_stratum.stratum_definition_sha256.clone(),
                population_manifest_sha256: base_stratum.population_manifest_sha256.clone(),
                selection_receipt_ref: base_stratum.selection_receipt_ref.clone(),
                population_size: population_u64,
                sample_size: sample_u64,
                inclusion_probability: ratio(sample_size, population_size),
                confirmed_opportunities_in_sample: counts.0,
                confirmed_no_opportunities_in_sample: counts.1,
                unresolved_in_sample: counts.2,
                potential_opportunities_in_sample: potential,
                upper_missed_opportunity_count: upper_u64,
                per_hypothesis_error_probability: per_hypothesis_alpha,
                is_full_census: sample_size == population_size,
            });
        }

        let mut union_counts = (0_u64, 0_u64, 0_u64);
        for disposition in &union_dispositions {
            increment_union_counts(&mut union_counts, *disposition);
        }
        let union_potential = union_counts
            .0
            .checked_add(union_counts.2)
            .ok_or_else(|| vec![ScenarioOpportunityError::IntegerOverflow])?;
        let union_upper = upper_hypergeometric_population_successes(
            population_size,
            sample_size,
            usize::try_from(union_potential)
                .map_err(|_| vec![ScenarioOpportunityError::IntegerOverflow])?,
            per_hypothesis_alpha,
        )
        .map_err(|_| vec![ScenarioOpportunityError::IntegerOverflow])?;
        let union_upper_u64 = u64::try_from(union_upper)
            .map_err(|_| vec![ScenarioOpportunityError::IntegerOverflow])?;

        union_upper_total = checked_add(union_upper_total, union_upper_u64)?;
        union_population_total = checked_add(union_population_total, population_u64)?;
        union_sample_total = checked_add(union_sample_total, sample_u64)?;
        union_observed_opportunities = checked_add(union_observed_opportunities, union_counts.0)?;
        union_observed_unresolved = checked_add(union_observed_unresolved, union_counts.2)?;

        union_stratum_reports.push(UnionStratumAuditReport {
            stratum_id: base_stratum.stratum_id.clone(),
            stratum_definition_sha256: base_stratum.stratum_definition_sha256.clone(),
            population_manifest_sha256: base_stratum.population_manifest_sha256.clone(),
            selection_receipt_ref: base_stratum.selection_receipt_ref.clone(),
            population_size: population_u64,
            sample_size: sample_u64,
            inclusion_probability: ratio(sample_size, population_size),
            confirmed_union_opportunities_in_sample: union_counts.0,
            confirmed_union_no_opportunity_in_sample: union_counts.1,
            unresolved_union_in_sample: union_counts.2,
            potential_union_opportunities_in_sample: union_potential,
            upper_missed_union_opportunity_count: union_upper_u64,
            per_hypothesis_error_probability: per_hypothesis_alpha,
            is_full_census: sample_size == population_size,
        });
    }

    for (index, scenario) in plan.scenarios.iter().enumerate() {
        let accepted = accepted_scenario_counts[index];
        let denominator = accepted
            .0
            .checked_add(accepted.2)
            .and_then(|value| value.checked_add(scenario_upper_totals[index]))
            .ok_or_else(|| vec![ScenarioOpportunityError::IntegerOverflow])?;
        let recall = lower_recall(accepted.0, denominator);
        scenario_reports.push(ScenarioRecallReport {
            scenario_id: scenario.scenario_id.clone(),
            scenario_sha256: scenario.scenario_sha256.clone(),
            incumbent_frontier_sha256: scenario.incumbent_frontier_sha256.clone(),
            accepted_confirmed_opportunities: accepted.0,
            accepted_confirmed_no_opportunity: accepted.1,
            accepted_unresolved: accepted.2,
            rejected_population: scenario_population_totals[index],
            audited_rejected_candidates: scenario_sample_totals[index],
            rejected_opportunities_observed: scenario_observed_opportunity_totals[index],
            rejected_unresolved_observed: scenario_observed_unresolved_totals[index],
            missed_opportunities_upper_bound: scenario_upper_totals[index],
            conditional_opportunity_recall_lower_bound: recall,
            target_met: recall
                .map(|value| value >= plan.target_opportunity_recall)
                .unwrap_or(false),
            per_hypothesis_error_probability: per_hypothesis_alpha,
            strata: std::mem::take(&mut scenario_stratum_reports[index]),
        });
    }

    let union_denominator = accepted_union_counts
        .0
        .checked_add(accepted_union_counts.2)
        .and_then(|value| value.checked_add(union_upper_total))
        .ok_or_else(|| vec![ScenarioOpportunityError::IntegerOverflow])?;
    let union_recall = lower_recall(accepted_union_counts.0, union_denominator);

    let union_report = UnionOpportunityRecallReport {
        accepted_confirmed_opportunities: accepted_union_counts.0,
        accepted_confirmed_no_opportunity: accepted_union_counts.1,
        accepted_unresolved: accepted_union_counts.2,
        rejected_population: union_population_total,
        audited_rejected_candidates: union_sample_total,
        rejected_opportunities_observed: union_observed_opportunities,
        rejected_unresolved_observed: union_observed_unresolved,
        missed_union_opportunities_upper_bound: union_upper_total,
        conditional_any_scenario_recall_lower_bound: union_recall,
        target_met: union_recall
            .map(|value| value >= plan.target_opportunity_recall)
            .unwrap_or(false),
        per_hypothesis_error_probability: per_hypothesis_alpha,
        strata: union_stratum_reports,
    };

    let mut evaluation_receipts = Vec::new();
    for bundle in &plan.accepted_bundles {
        for evaluation in &bundle.evaluations {
            evaluation_receipts.push(ScenarioEvaluationReceiptRef {
                candidate_id: bundle.candidate_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
                run_id: evaluation.run_id.clone(),
                evidence_ref: evaluation.evidence_ref.clone(),
                reference_outcome: evaluation.reference_outcome,
                within_declared_domain: evaluation.within_declared_domain,
            });
        }
    }
    for base_stratum in &plan.base_plan.strata {
        let bundle_group = rejected_map
            .get(base_stratum.stratum_id.as_str())
            .expect("validated rejected stratum bundle must exist");
        for bundle in &bundle_group.bundles {
            for evaluation in &bundle.evaluations {
                evaluation_receipts.push(ScenarioEvaluationReceiptRef {
                    candidate_id: bundle.candidate_id.clone(),
                    scenario_id: evaluation.scenario_id.clone(),
                    run_id: evaluation.run_id.clone(),
                    evidence_ref: evaluation.evidence_ref.clone(),
                    reference_outcome: evaluation.reference_outcome,
                    within_declared_domain: evaluation.within_declared_domain,
                });
            }
        }
    }

    Ok(ScenarioOpportunityReport {
        contract_version: SCENARIO_OPPORTUNITY_VERSION,
        bound_method: BOUND_METHOD,
        sampling_algorithm: SAMPLING_ALGORITHM,
        opportunity_rule: SCENARIO_OPPORTUNITY_RULE,
        campaign_id: plan.campaign_id.clone(),
        scenario_manifest_sha256: plan.scenario_manifest_sha256.clone(),
        objective_manifest_sha256: plan.objective_manifest_sha256.clone(),
        opportunity_rule_sha256: plan.opportunity_rule_sha256.clone(),
        frozen_universe_manifest_sha256: plan.base_plan.frozen_universe.manifest_sha256.clone(),
        accepted_manifest_sha256: plan.base_plan.accepted.manifest_sha256.clone(),
        primary_scenario_id: plan.primary_scenario_id.clone(),
        familywise_confidence_level: plan.base_plan.familywise_confidence_level,
        target_opportunity_recall: plan.target_opportunity_recall,
        simultaneous_bound_count: u64::try_from(hypothesis_count)
            .map_err(|_| vec![ScenarioOpportunityError::IntegerOverflow])?,
        familywise_error_probability: familywise_alpha,
        feasible_recall_lower_bound: feasibility_report.feasible_recall_lower_bound,
        feasibility_target_met: feasibility_report.target_met,
        union: union_report,
        scenarios: scenario_reports,
        evaluation_receipts,
    })
}

fn validate_plan(plan: &ScenarioOpportunityPlan) -> Vec<ScenarioOpportunityError> {
    let mut errors = Vec::new();
    for (value, field) in [
        (plan.campaign_id.as_str(), "campaign_id"),
        (plan.primary_scenario_id.as_str(), "primary_scenario_id"),
    ] {
        if value.trim().is_empty() {
            errors.push(ScenarioOpportunityError::EmptyIdentity(field));
        }
    }
    for (digest, field) in [
        (&plan.scenario_manifest_sha256, "scenario_manifest_sha256"),
        (&plan.objective_manifest_sha256, "objective_manifest_sha256"),
        (&plan.opportunity_rule_sha256, "opportunity_rule_sha256"),
    ] {
        if !is_sha256(digest) {
            errors.push(ScenarioOpportunityError::InvalidSha256(field));
        }
    }
    if !plan.target_opportunity_recall.is_finite()
        || plan.target_opportunity_recall <= 0.0
        || plan.target_opportunity_recall > 1.0
    {
        errors.push(ScenarioOpportunityError::InvalidRecallTarget);
    }
    if plan.objective_definitions.is_empty() {
        errors.push(ScenarioOpportunityError::EmptyObjectiveDefinitions);
    }
    let mut objective_ids = BTreeSet::new();
    for definition in &plan.objective_definitions {
        if definition.objective_id.trim().is_empty()
            || definition.unit.trim().is_empty()
            || !definition.epsilon.is_finite()
            || definition.epsilon < 0.0
        {
            errors.push(ScenarioOpportunityError::InvalidObjectiveDefinition(
                definition.objective_id.clone(),
            ));
        }
        if !objective_ids.insert(definition.objective_id.as_str()) {
            errors.push(ScenarioOpportunityError::DuplicateObjectiveId(
                definition.objective_id.clone(),
            ));
        }
    }
    if plan.scenarios.is_empty() {
        errors.push(ScenarioOpportunityError::EmptyScenarioManifest);
    }
    let mut scenario_ids = BTreeSet::new();
    let mut previous_id: Option<&str> = None;
    for scenario in &plan.scenarios {
        if scenario.scenario_id.trim().is_empty() {
            errors.push(ScenarioOpportunityError::EmptyIdentity("scenario_id"));
        }
        if !scenario_ids.insert(scenario.scenario_id.as_str()) {
            errors.push(ScenarioOpportunityError::DuplicateScenarioId(
                scenario.scenario_id.clone(),
            ));
        }
        if let Some(previous) = previous_id
            && scenario.scenario_id.as_str() <= previous
        {
            errors.push(ScenarioOpportunityError::ScenarioManifestNotSorted);
        }
        previous_id = Some(&scenario.scenario_id);

        if !is_sha256(&scenario.scenario_sha256) {
            errors.push(ScenarioOpportunityError::InvalidSha256(
                "scenario.scenario_sha256",
            ));
        }
        if !is_sha256(&scenario.incumbent_frontier_sha256) {
            errors.push(ScenarioOpportunityError::InvalidSha256(
                "scenario.incumbent_frontier_sha256",
            ));
        }
        if scenario.incumbent_frontier.is_empty() {
            errors.push(ScenarioOpportunityError::EmptyScenarioFrontier(
                scenario.scenario_id.clone(),
            ));
        }

        let mut frontier_ids = BTreeSet::new();
        for point in &scenario.incumbent_frontier {
            if point.candidate_id.trim().is_empty() {
                errors.push(ScenarioOpportunityError::EmptyIdentity(
                    "incumbent_frontier.candidate_id",
                ));
            }
            if !frontier_ids.insert(point.candidate_id.as_str()) {
                errors.push(ScenarioOpportunityError::DuplicateFrontierCandidate {
                    scenario_id: scenario.scenario_id.clone(),
                    candidate_id: point.candidate_id.clone(),
                });
            }
            if !valid_objective_vector(&point.objectives, &plan.objective_definitions) {
                errors.push(ScenarioOpportunityError::InvalidObjectiveVector {
                    scenario_id: scenario.scenario_id.clone(),
                    candidate_id: point.candidate_id.clone(),
                });
            }
        }
        for left in 0..scenario.incumbent_frontier.len() {
            for right in left + 1..scenario.incumbent_frontier.len() {
                let first = &scenario.incumbent_frontier[left];
                let second = &scenario.incumbent_frontier[right];
                if epsilon_dominates(
                    &first.objectives,
                    &second.objectives,
                    &plan.objective_definitions,
                ) || epsilon_dominates(
                    &second.objectives,
                    &first.objectives,
                    &plan.objective_definitions,
                ) || epsilon_equivalent(
                    &first.objectives,
                    &second.objectives,
                    &plan.objective_definitions,
                ) {
                    errors.push(ScenarioOpportunityError::InvalidScenarioFrontier(
                        scenario.scenario_id.clone(),
                    ));
                }
            }
        }
    }
    if !scenario_ids.contains(plan.primary_scenario_id.as_str()) {
        errors.push(ScenarioOpportunityError::UnknownPrimaryScenario(
            plan.primary_scenario_id.clone(),
        ));
    }

    let accepted_expected: BTreeMap<&str, &super::AcceptedCandidate> = plan
        .base_plan
        .accepted
        .candidates
        .iter()
        .map(|candidate| (candidate.candidate_id.as_str(), candidate))
        .collect();
    if plan.accepted_bundles.len() != accepted_expected.len() {
        errors.push(ScenarioOpportunityError::AcceptedBundleCountMismatch);
    }
    let mut accepted_seen = BTreeSet::new();
    for (index, bundle) in plan.accepted_bundles.iter().enumerate() {
        if plan
            .base_plan
            .accepted
            .candidates
            .get(index)
            .is_some_and(|expected| expected.candidate_id != bundle.candidate_id)
        {
            errors.push(ScenarioOpportunityError::AcceptedBundleIdentityMismatch {
                bundle_index: index,
            });
        }
        if !accepted_expected.contains_key(bundle.candidate_id.as_str()) {
            errors.push(ScenarioOpportunityError::AcceptedBundleIdentityMismatch {
                bundle_index: index,
            });
        }
        if !accepted_seen.insert(bundle.candidate_id.as_str()) {
            errors.push(ScenarioOpportunityError::DuplicateBundleCandidate(
                bundle.candidate_id.clone(),
            ));
        }
        if let Some(expected) = accepted_expected.get(bundle.candidate_id.as_str()) {
            validate_bundle(
                bundle,
                &plan.scenarios,
                &plan.objective_definitions,
                &plan.primary_scenario_id,
                Some((expected.outcome, expected.evidence_ref.as_str())),
                &mut errors,
            );
        } else {
            validate_bundle(
                bundle,
                &plan.scenarios,
                &plan.objective_definitions,
                &plan.primary_scenario_id,
                None,
                &mut errors,
            );
        }
    }
    for candidate_id in accepted_expected.keys() {
        if !accepted_seen.contains(candidate_id) {
            errors.push(ScenarioOpportunityError::AcceptedBundleIdentityMismatch {
                bundle_index: plan.accepted_bundles.len(),
            });
        }
    }

    let expected_strata: BTreeMap<&str, &super::RejectStratum> = plan
        .base_plan
        .strata
        .iter()
        .map(|stratum| (stratum.stratum_id.as_str(), stratum))
        .collect();
    if plan.rejected_strata.len() != expected_strata.len() {
        errors.push(ScenarioOpportunityError::RejectedStratumCountMismatch);
    }
    let mut supplied_strata = BTreeSet::new();
    for bundle_group in &plan.rejected_strata {
        if !expected_strata.contains_key(bundle_group.stratum_id.as_str()) {
            errors.push(ScenarioOpportunityError::UnexpectedRejectedStratum(
                bundle_group.stratum_id.clone(),
            ));
            continue;
        }
        if !supplied_strata.insert(bundle_group.stratum_id.as_str()) {
            errors.push(ScenarioOpportunityError::DuplicateRejectedStratum(
                bundle_group.stratum_id.clone(),
            ));
        }
        let expected = expected_strata[bundle_group.stratum_id.as_str()];
        if bundle_group.bundles.len() != expected.samples.len() {
            errors.push(ScenarioOpportunityError::RejectedBundleCountMismatch(
                bundle_group.stratum_id.clone(),
            ));
            continue;
        }
        let mut candidate_seen = BTreeSet::new();
        for (sample_index, (sample, bundle)) in expected
            .samples
            .iter()
            .zip(&bundle_group.bundles)
            .enumerate()
        {
            if sample.candidate_id != bundle.candidate_id {
                errors.push(ScenarioOpportunityError::RejectedBundleIdentityMismatch {
                    stratum_id: bundle_group.stratum_id.clone(),
                    sample_index,
                });
            }
            if !candidate_seen.insert(bundle.candidate_id.as_str()) {
                errors.push(ScenarioOpportunityError::DuplicateBundleCandidate(
                    bundle.candidate_id.clone(),
                ));
            }
            validate_bundle(
                bundle,
                &plan.scenarios,
                &plan.objective_definitions,
                &plan.primary_scenario_id,
                Some((sample.outcome, sample.evidence_ref.as_str())),
                &mut errors,
            );
        }
    }
    for stratum_id in expected_strata.keys() {
        if !supplied_strata.contains(stratum_id) {
            errors.push(ScenarioOpportunityError::MissingRejectedStratum(
                (*stratum_id).to_string(),
            ));
        }
    }
    errors
}

fn validate_bundle(
    bundle: &CandidateScenarioBundle,
    scenarios: &[ScenarioDefinition],
    definitions: &[ObjectiveDefinition],
    primary_scenario_id: &str,
    primary_expected: Option<(AuditOutcome, &str)>,
    errors: &mut Vec<ScenarioOpportunityError>,
) {
    if bundle.evaluations.len() != scenarios.len() {
        errors.push(ScenarioOpportunityError::BundleScenarioCountMismatch(
            bundle.candidate_id.clone(),
        ));
        return;
    }
    let mut primary_found = false;
    for (index, (evaluation, scenario)) in bundle.evaluations.iter().zip(scenarios).enumerate() {
        if evaluation.scenario_id != scenario.scenario_id {
            errors.push(ScenarioOpportunityError::BundleScenarioOrderMismatch {
                candidate_id: bundle.candidate_id.clone(),
                scenario_index: index,
            });
        }
        if evaluation.run_id.trim().is_empty() {
            errors.push(ScenarioOpportunityError::EmptyRunId(
                bundle.candidate_id.clone(),
            ));
        }
        if evaluation.evidence_ref.trim().is_empty() {
            errors.push(ScenarioOpportunityError::EmptyEvidenceRef(
                bundle.candidate_id.clone(),
            ));
        }
        if !evaluation.within_declared_domain
            && matches!(
                evaluation.reference_outcome,
                AuditOutcome::ConfirmedFeasible | AuditOutcome::ConfirmedInfeasible
            )
        {
            errors.push(
                ScenarioOpportunityError::OutOfDomainOutcomeMustBeUnresolved {
                    candidate_id: bundle.candidate_id.clone(),
                    scenario_id: evaluation.scenario_id.clone(),
                },
            );
        }
        let needs_objectives = evaluation.reference_outcome == AuditOutcome::ConfirmedFeasible
            && evaluation.within_declared_domain;
        if !evaluation.objectives.is_empty()
            && !valid_objective_vector(&evaluation.objectives, definitions)
        {
            errors.push(ScenarioOpportunityError::InvalidObjectiveVector {
                scenario_id: evaluation.scenario_id.clone(),
                candidate_id: bundle.candidate_id.clone(),
            });
        }
        if needs_objectives && evaluation.objectives.is_empty() {
            errors.push(ScenarioOpportunityError::InvalidObjectiveVector {
                scenario_id: evaluation.scenario_id.clone(),
                candidate_id: bundle.candidate_id.clone(),
            });
        }
        if evaluation.scenario_id == primary_scenario_id {
            primary_found = true;
            if let Some((expected_outcome, expected_evidence)) = primary_expected {
                if evaluation.reference_outcome != expected_outcome {
                    errors.push(ScenarioOpportunityError::PrimaryOutcomeMismatch(
                        bundle.candidate_id.clone(),
                    ));
                }
                if evaluation.evidence_ref != expected_evidence {
                    errors.push(ScenarioOpportunityError::PrimaryEvidenceMismatch(
                        bundle.candidate_id.clone(),
                    ));
                }
            }
        }
    }
    if !primary_found {
        errors.push(ScenarioOpportunityError::UnknownPrimaryScenario(
            primary_scenario_id.to_string(),
        ));
    }
}

fn classify_scenario_evaluation(
    candidate_id: &str,
    evaluation: &ScenarioEvaluation,
    scenario: &ScenarioDefinition,
    definitions: &[ObjectiveDefinition],
) -> ScenarioDisposition {
    if !evaluation.within_declared_domain {
        return ScenarioDisposition::Unresolved;
    }
    match evaluation.reference_outcome {
        AuditOutcome::ConfirmedInfeasible => ScenarioDisposition::ConfirmedNoOpportunity,
        AuditOutcome::ConfirmedFeasible => {
            if is_epsilon_frontier_extension(
                candidate_id,
                &evaluation.objectives,
                &scenario.incumbent_frontier,
                definitions,
            ) {
                ScenarioDisposition::ConfirmedOpportunity
            } else {
                ScenarioDisposition::ConfirmedNoOpportunity
            }
        }
        AuditOutcome::Failed
        | AuditOutcome::NonConverged
        | AuditOutcome::Unsupported
        | AuditOutcome::Inconclusive => ScenarioDisposition::Unresolved,
    }
}

fn classify_union(dispositions: &[ScenarioDisposition]) -> UnionOpportunityDisposition {
    if dispositions.contains(&ScenarioDisposition::ConfirmedOpportunity) {
        UnionOpportunityDisposition::OpportunityInAtLeastOneScenario
    } else if dispositions
        .iter()
        .all(|value| *value == ScenarioDisposition::ConfirmedNoOpportunity)
    {
        UnionOpportunityDisposition::NoOpportunityInAnyScenario
    } else {
        UnionOpportunityDisposition::UnresolvedAcrossScenarioFamily
    }
}

fn increment_counts(counts: &mut (u64, u64, u64), disposition: ScenarioDisposition) {
    match disposition {
        ScenarioDisposition::ConfirmedOpportunity => counts.0 += 1,
        ScenarioDisposition::ConfirmedNoOpportunity => counts.1 += 1,
        ScenarioDisposition::Unresolved => counts.2 += 1,
    }
}

fn increment_union_counts(counts: &mut (u64, u64, u64), disposition: UnionOpportunityDisposition) {
    match disposition {
        UnionOpportunityDisposition::OpportunityInAtLeastOneScenario => counts.0 += 1,
        UnionOpportunityDisposition::NoOpportunityInAnyScenario => counts.1 += 1,
        UnionOpportunityDisposition::UnresolvedAcrossScenarioFamily => counts.2 += 1,
    }
}

fn lower_recall(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn checked_add(left: u64, right: u64) -> Result<u64, Vec<ScenarioOpportunityError>> {
    left.checked_add(right)
        .ok_or_else(|| vec![ScenarioOpportunityError::IntegerOverflow])
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_opportunity::ObjectiveDirection;
    use crate::{
        AcceptedCandidate, AcceptedCandidateManifest, AuditSample, FrozenCandidateUniverse,
        RejectStratum, deterministic_audit_sample_ids,
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

    fn objective_definitions() -> Vec<ObjectiveDefinition> {
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

    fn scenario(
        scenario_id: &str,
        frontier: Vec<IncumbentFrontierPoint>,
        digest_char: char,
    ) -> ScenarioDefinition {
        ScenarioDefinition {
            scenario_id: scenario_id.to_string(),
            scenario_sha256: digest(digest_char),
            incumbent_frontier_sha256: digest(digest_char.to_ascii_uppercase()),
            incumbent_frontier: frontier,
        }
    }

    fn frontier_nominal() -> Vec<IncumbentFrontierPoint> {
        vec![
            IncumbentFrontierPoint {
                candidate_id: "nominal-incumbent-a".to_string(),
                objectives: objective_values(100.0, 0.90),
            },
            IncumbentFrontierPoint {
                candidate_id: "nominal-incumbent-b".to_string(),
                objectives: objective_values(120.0, 0.95),
            },
        ]
    }

    fn frontier_stress() -> Vec<IncumbentFrontierPoint> {
        vec![
            IncumbentFrontierPoint {
                candidate_id: "stress-incumbent-a".to_string(),
                objectives: objective_values(110.0, 0.88),
            },
            IncumbentFrontierPoint {
                candidate_id: "stress-incumbent-b".to_string(),
                objectives: objective_values(125.0, 0.93),
            },
        ]
    }

    fn eval(
        _candidate_id: &str,
        scenario_id: &str,
        run_id: &str,
        outcome: AuditOutcome,
        within_declared_domain: bool,
        objectives: Vec<ObjectiveValue>,
        evidence_ref: String,
    ) -> ScenarioEvaluation {
        ScenarioEvaluation {
            scenario_id: scenario_id.to_string(),
            run_id: run_id.to_string(),
            reference_outcome: outcome,
            within_declared_domain,
            objectives,
            evidence_ref,
        }
    }

    fn accepted_bundle(
        candidate_id: &str,
        primary_outcome: AuditOutcome,
        primary_values: Vec<ObjectiveValue>,
        stress_outcome: AuditOutcome,
        stress_values: Vec<ObjectiveValue>,
    ) -> CandidateScenarioBundle {
        CandidateScenarioBundle {
            candidate_id: candidate_id.to_string(),
            evaluations: vec![
                eval(
                    candidate_id,
                    "nominal",
                    &format!("nominal-run-{candidate_id}"),
                    primary_outcome,
                    true,
                    primary_values,
                    format!("primary-run-{candidate_id}"),
                ),
                eval(
                    candidate_id,
                    "stress",
                    &format!("stress-run-{candidate_id}"),
                    stress_outcome,
                    true,
                    stress_values,
                    format!("stress-run-receipt-{candidate_id}"),
                ),
            ],
        }
    }

    fn rejected_bundle(
        candidate_id: &str,
        nominal: ScenarioEvaluation,
        stress: ScenarioEvaluation,
    ) -> CandidateScenarioBundle {
        CandidateScenarioBundle {
            candidate_id: candidate_id.to_string(),
            evaluations: vec![nominal, stress],
        }
    }

    fn plan() -> ScenarioOpportunityPlan {
        let seed = 0x1234_5678_u64;
        let accepted = vec![
            AcceptedCandidate {
                candidate_id: "accepted-0".to_string(),
                outcome: AuditOutcome::ConfirmedFeasible,
                evidence_ref: "primary-run-accepted-0".to_string(),
            },
            AcceptedCandidate {
                candidate_id: "accepted-1".to_string(),
                outcome: AuditOutcome::Failed,
                evidence_ref: "primary-run-accepted-1".to_string(),
            },
        ];
        let rejected_ids = vec![
            "rejected-0".to_string(),
            "rejected-1".to_string(),
            "rejected-2".to_string(),
            "rejected-3".to_string(),
        ];
        let selected =
            deterministic_audit_sample_ids("reject-all", &rejected_ids, 4, seed).unwrap();
        let primary_outcomes = BTreeMap::from([
            ("rejected-0", AuditOutcome::ConfirmedFeasible),
            ("rejected-1", AuditOutcome::ConfirmedInfeasible),
            ("rejected-2", AuditOutcome::NonConverged),
            ("rejected-3", AuditOutcome::ConfirmedInfeasible),
        ]);
        let samples = selected
            .iter()
            .map(|candidate_id| AuditSample {
                candidate_id: candidate_id.clone(),
                outcome: primary_outcomes[candidate_id.as_str()],
                evidence_ref: format!("primary-run-{candidate_id}"),
            })
            .collect();
        let mut universe = vec!["accepted-0".to_string(), "accepted-1".to_string()];
        universe.extend(rejected_ids.clone());
        universe.sort();

        let base_plan = RejectAuditPlan {
            frozen_universe: FrozenCandidateUniverse {
                candidate_ids: universe,
                manifest_sha256: digest('a'),
            },
            sampling_seed: seed,
            seed_commitment_ref: "primary-seed-commitment".to_string(),
            seed_committed_before_outcomes: true,
            familywise_confidence_level: 0.95,
            target_feasible_recall: 0.99,
            accepted: AcceptedCandidateManifest {
                candidates: accepted,
                manifest_sha256: digest('b'),
                evidence_ref: "primary-accepted-manifest".to_string(),
            },
            strata: vec![RejectStratum {
                stratum_id: "reject-all".to_string(),
                stratum_definition_sha256: digest('c'),
                population_manifest_sha256: digest('d'),
                selection_receipt_ref: "primary-selection-receipt".to_string(),
                rejected_candidate_ids: rejected_ids,
                sample_size: 4,
                samples,
            }],
        };

        let rejected_bundles = base_plan.strata[0]
            .samples
            .iter()
            .map(|sample| {
                let candidate_id = sample.candidate_id.as_str();
                let (nominal_outcome, nominal_values) = match candidate_id {
                    "rejected-0" => (
                        AuditOutcome::ConfirmedFeasible,
                        objective_values(95.0, 0.94),
                    ),
                    "rejected-1" => (AuditOutcome::ConfirmedInfeasible, Vec::new()),
                    "rejected-2" => (AuditOutcome::NonConverged, Vec::new()),
                    _ => (AuditOutcome::ConfirmedInfeasible, Vec::new()),
                };
                let (stress_outcome, stress_in_domain, stress_values) = match candidate_id {
                    "rejected-0" => (
                        AuditOutcome::ConfirmedFeasible,
                        true,
                        objective_values(100.0, 0.96),
                    ),
                    "rejected-1" => (
                        AuditOutcome::ConfirmedFeasible,
                        true,
                        objective_values(105.0, 0.97),
                    ),
                    "rejected-2" => (AuditOutcome::Inconclusive, false, Vec::new()),
                    _ => (AuditOutcome::ConfirmedInfeasible, true, Vec::new()),
                };
                rejected_bundle(
                    candidate_id,
                    eval(
                        candidate_id,
                        "nominal",
                        &format!("nominal-run-{candidate_id}"),
                        nominal_outcome,
                        true,
                        nominal_values,
                        format!("primary-run-{candidate_id}"),
                    ),
                    eval(
                        candidate_id,
                        "stress",
                        &format!("stress-run-{candidate_id}"),
                        stress_outcome,
                        stress_in_domain,
                        stress_values,
                        format!("stress-run-receipt-{candidate_id}"),
                    ),
                )
            })
            .collect();

        ScenarioOpportunityPlan {
            campaign_id: "two-scenario-campaign-v1".to_string(),
            scenario_manifest_sha256: digest('e'),
            objective_manifest_sha256: digest('f'),
            opportunity_rule_sha256: digest('1'),
            primary_scenario_id: "nominal".to_string(),
            target_opportunity_recall: 0.99,
            base_plan,
            objective_definitions: objective_definitions(),
            scenarios: vec![
                scenario("nominal", frontier_nominal(), '2'),
                scenario("stress", frontier_stress(), '3'),
            ],
            accepted_bundles: vec![
                accepted_bundle(
                    "accepted-0",
                    AuditOutcome::ConfirmedFeasible,
                    objective_values(95.0, 0.94),
                    AuditOutcome::ConfirmedFeasible,
                    objective_values(150.0, 0.70),
                ),
                accepted_bundle(
                    "accepted-1",
                    AuditOutcome::Failed,
                    Vec::new(),
                    AuditOutcome::Unsupported,
                    Vec::new(),
                ),
            ],
            rejected_strata: vec![StratumScenarioBundles {
                stratum_id: "reject-all".to_string(),
                bundles: rejected_bundles,
            }],
        }
    }

    #[test]
    fn reports_scenario_specific_and_union_recall_separately() {
        let report = audit_scenario_opportunities(&plan()).unwrap();
        assert_eq!(report.scenarios.len(), 2);
        assert_eq!(report.simultaneous_bound_count, 4);
        assert_eq!(report.evaluation_receipts.len(), 12);
        assert_eq!(report.bound_method, BOUND_METHOD);
        assert_eq!(report.sampling_algorithm, SAMPLING_ALGORITHM);
        assert_eq!(report.union.accepted_confirmed_opportunities, 1);
        assert_eq!(report.union.accepted_unresolved, 1);
        assert_eq!(report.union.rejected_opportunities_observed, 2);
        assert_eq!(report.union.rejected_unresolved_observed, 1);
        assert_eq!(report.union.missed_union_opportunities_upper_bound, 3);
        assert_eq!(
            report.union.conditional_any_scenario_recall_lower_bound,
            Some(0.2)
        );
        assert_eq!(report.scenarios[0].scenario_id, "nominal");
        assert_eq!(
            report.scenarios[0].conditional_opportunity_recall_lower_bound,
            Some(0.25)
        );
        assert_eq!(report.scenarios[1].scenario_id, "stress");
        assert_eq!(
            report.scenarios[1].conditional_opportunity_recall_lower_bound,
            Some(0.0)
        );
    }

    #[test]
    fn non_nominal_only_opportunity_is_counted_in_the_union() {
        let report = audit_scenario_opportunities(&plan()).unwrap();
        let stress = &report.scenarios[1];
        assert_eq!(stress.rejected_opportunities_observed, 2);
        assert_eq!(report.union.rejected_opportunities_observed, 2);
    }

    #[test]
    fn opportunity_in_one_scenario_keeps_union_positive_when_another_is_unresolved() {
        let mut test_plan = plan();
        let bundle = test_plan.rejected_strata[0]
            .bundles
            .iter_mut()
            .find(|bundle| bundle.candidate_id == "rejected-0")
            .unwrap();
        bundle.evaluations[1].reference_outcome = AuditOutcome::Inconclusive;
        bundle.evaluations[1].within_declared_domain = false;
        let report = audit_scenario_opportunities(&test_plan).unwrap();
        assert_eq!(report.union.rejected_opportunities_observed, 2);
        assert_eq!(report.union.rejected_unresolved_observed, 1);
        assert_eq!(report.scenarios[1].rejected_unresolved_observed, 2);
    }

    #[test]
    fn unresolved_scenario_prevents_union_no_opportunity() {
        let mut test_plan = plan();
        let bundle = test_plan.rejected_strata[0]
            .bundles
            .iter_mut()
            .find(|bundle| bundle.candidate_id == "rejected-3")
            .unwrap();
        bundle.evaluations[1].reference_outcome = AuditOutcome::Unsupported;
        let report = audit_scenario_opportunities(&test_plan).unwrap();
        assert_eq!(report.union.rejected_unresolved_observed, 2);
        assert_eq!(report.union.rejected_opportunities_observed, 2);
    }

    #[test]
    fn missing_scenario_result_is_rejected() {
        let mut test_plan = plan();
        test_plan.rejected_strata[0].bundles[0].evaluations.pop();
        let errors = audit_scenario_opportunities(&test_plan).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ScenarioOpportunityError::BundleScenarioCountMismatch(_)
        )));
    }

    #[test]
    fn scenario_order_is_part_of_the_bundle_identity() {
        let mut test_plan = plan();
        test_plan.rejected_strata[0].bundles[0]
            .evaluations
            .swap(0, 1);
        let errors = audit_scenario_opportunities(&test_plan).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ScenarioOpportunityError::BundleScenarioOrderMismatch { .. }
        )));
    }

    #[test]
    fn primary_scenario_outcome_must_match_base_audit() {
        let mut test_plan = plan();
        test_plan.rejected_strata[0].bundles[0].evaluations[0].reference_outcome =
            AuditOutcome::ConfirmedInfeasible;
        let errors = audit_scenario_opportunities(&test_plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, ScenarioOpportunityError::PrimaryOutcomeMismatch(_)))
        );
    }

    #[test]
    fn primary_scenario_evidence_must_match_base_audit() {
        let mut test_plan = plan();
        test_plan.accepted_bundles[0].evaluations[0].evidence_ref =
            "a-different-primary-run".to_string();
        let errors = audit_scenario_opportunities(&test_plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, ScenarioOpportunityError::PrimaryEvidenceMismatch(_)))
        );
    }

    #[test]
    fn out_of_domain_cannot_be_a_confirmed_feasibility_label() {
        let mut test_plan = plan();
        let bundle = test_plan.rejected_strata[0]
            .bundles
            .iter_mut()
            .find(|bundle| bundle.candidate_id == "rejected-2")
            .unwrap();
        bundle.evaluations[0].reference_outcome = AuditOutcome::ConfirmedInfeasible;
        bundle.evaluations[0].within_declared_domain = false;
        let errors = audit_scenario_opportunities(&test_plan).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ScenarioOpportunityError::OutOfDomainOutcomeMustBeUnresolved { .. }
        )));
    }

    #[test]
    fn scenario_frontiers_must_be_nondominated_and_non_equivalent() {
        let mut test_plan = plan();
        test_plan.scenarios[0]
            .incumbent_frontier
            .push(IncumbentFrontierPoint {
                candidate_id: "duplicate-tradeoff".to_string(),
                objectives: objective_values(100.5, 0.905),
            });
        let errors = audit_scenario_opportunities(&test_plan).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, ScenarioOpportunityError::InvalidScenarioFrontier(_)))
        );
    }

    #[test]
    fn wrong_objective_units_or_order_are_rejected() {
        let mut test_plan = plan();
        test_plan.accepted_bundles[0].evaluations[0]
            .objectives
            .swap(0, 1);
        let errors = audit_scenario_opportunities(&test_plan).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ScenarioOpportunityError::InvalidObjectiveVector { .. }
        )));
    }

    #[test]
    fn sample_identity_order_is_preserved() {
        let mut test_plan = plan();
        test_plan.rejected_strata[0].bundles.swap(0, 1);
        let errors = audit_scenario_opportunities(&test_plan).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ScenarioOpportunityError::RejectedBundleIdentityMismatch { .. }
        )));
    }

    #[test]
    fn familywise_error_is_allocated_across_union_and_scenario_strata() {
        let report = audit_scenario_opportunities(&plan()).unwrap();
        assert_eq!(report.simultaneous_bound_count, 4);
        assert!((report.union.per_hypothesis_error_probability - (0.05 / 4.0)).abs() < 1e-12);
        assert!(
            (report.scenarios[0].per_hypothesis_error_probability - (0.05 / 4.0)).abs() < 1e-12
        );
        assert!(
            (report.scenarios[1].per_hypothesis_error_probability - (0.05 / 4.0)).abs() < 1e-12
        );
        let mut adjusted_base = plan().base_plan;
        adjusted_base.familywise_confidence_level = 1.0 - (0.05 / 4.0);
        let expected_feasibility = audit_rejected_candidates(&adjusted_base).unwrap();
        assert_eq!(
            report.feasible_recall_lower_bound,
            expected_feasibility.feasible_recall_lower_bound
        );
    }

    #[test]
    fn zero_sample_leaves_every_candidate_as_a_possible_opportunity() {
        let mut test_plan = plan();
        test_plan.base_plan.strata[0].sample_size = 0;
        test_plan.base_plan.strata[0].samples.clear();
        test_plan.rejected_strata[0].bundles.clear();
        let report = audit_scenario_opportunities(&test_plan).unwrap();
        assert_eq!(report.union.missed_union_opportunities_upper_bound, 4);
        assert_eq!(report.scenarios[0].missed_opportunities_upper_bound, 4);
        assert_eq!(report.scenarios[1].missed_opportunities_upper_bound, 4);
    }

    #[test]
    fn changed_scenario_manifest_identity_changes_report_identity() {
        let first = plan();
        let mut second = plan();
        second.scenario_manifest_sha256 = digest('9');
        let first_report = audit_scenario_opportunities(&first).unwrap();
        let second_report = audit_scenario_opportunities(&second).unwrap();
        assert_ne!(
            first.scenario_manifest_sha256,
            second.scenario_manifest_sha256
        );
        assert_eq!(first_report.campaign_id, second_report.campaign_id);
        assert_ne!(
            first_report.scenario_manifest_sha256,
            second_report.scenario_manifest_sha256
        );
        assert_eq!(first_report.scenarios.len(), second_report.scenarios.len());
    }
}
