// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic end-to-end scientific planning trace.
//!
//! This module composes the already-separated 002H evidence-gap plan, 002I
//! Pareto frontier, and 002J FEP admissibility policy into one auditable
//! artifact. It does not create observations, replications, criterion
//! evidence, truth claims, or completion status.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use symthaea_evidence_plane::scientific_decision_frontier::{
    frontier, ParetoAssessment, ScientificDecisionCandidate, ScientificDecisionFrontier,
};
use symthaea_evidence_plane::gap_aware_outcome_planner::GapAwareOutcomePlan;

use crate::scientific_frontier_admissibility::{
    admit, ScientificAdmissibilityError, ScientificAdmissibilityPolicy,
    ScientificAdmissibilityResult,
};

/// Explicit metrics required to bridge a gap-aware candidate into the
/// multi-objective decision frontier.
///
/// These values are planner-supplied inputs. This module does not derive them
/// from experimental observations or reinterpret them as scientific truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificPlanningMetrics {
    pub candidate_id: String,
    pub eig_per_cost: f64,
    pub pragmatic_risk: f64,
}

/// Complete deterministic planning trace across 002H → 002I → 002J.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificPlanningTrace {
    pub schema_version: String,
    pub gap_plan: GapAwareOutcomePlan,
    pub metrics: Vec<ScientificPlanningMetrics>,
    pub gap_candidate_ids: Vec<String>,
    pub frontier: ScientificDecisionFrontier,
    pub admissibility: ScientificAdmissibilityResult,
    pub policy: ScientificAdmissibilityPolicy,
    pub artifact_digest: String,
}

/// Inputs to the deterministic trace builder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificPlanningTraceInput {
    pub gap_plan: GapAwareOutcomePlan,
    pub metrics: Vec<ScientificPlanningMetrics>,
    pub policy: ScientificAdmissibilityPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScientificPlanningTraceError {
    EmptyCandidateId,
    DuplicateMetricId(String),
    MissingMetrics(String),
    ExtraMetrics(String),
    NonFiniteMetric(String),
    Frontier(String),
    Admissibility(String),
    Serialization(String),
}

impl fmt::Display for ScientificPlanningTraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ScientificPlanningTraceError {}

const SCHEMA_VERSION: &str = "mycelix.symthaea.scientific-planning-trace.1.0.0";

fn candidate_id(
    left_model_id: &str,
    right_model_id: &str,
    outcome_id: &str,
) -> String {
    // Length-prefix each component so arbitrary model/outcome IDs cannot
    // collide merely because they contain the separator.
    format!(
        "{}:{}{}:{}{}:{}{}",
        left_model_id.len(),
        left_model_id,
        right_model_id.len(),
        right_model_id,
        outcome_id.len(),
        outcome_id
    )
}

fn expected_candidate(
    assessment: &symthaea_evidence_plane::gap_aware_outcome_planner::GapAwareOutcomeAssessment,
) -> ScientificDecisionCandidate {
    ScientificDecisionCandidate {
        candidate_id: candidate_id(
            &assessment.candidate.left_model_id,
            &assessment.candidate.right_model_id,
            &assessment.candidate.outcome_id,
        ),
        gap_coverage: assessment.gap_coverage,
        likelihood_separation: assessment.candidate.likelihood_separation,
        trajectory_divergence: assessment.candidate.trajectory_divergence,
        eig_per_cost: 0.0,
        pragmatic_risk: 0.0,
        left_lineage: assessment.candidate.left_lineage.clone(),
        right_lineage: assessment.candidate.right_lineage.clone(),
    }
}

/// Build one deterministic trace without introducing a scientific authority
/// transition.
pub fn build(
    input: &ScientificPlanningTraceInput,
) -> Result<ScientificPlanningTrace, ScientificPlanningTraceError> {
    let mut gap_assessments = input.gap_plan.assessments.clone();
    gap_assessments.sort_by(|a, b| {
        candidate_id(
            &a.candidate.left_model_id,
            &a.candidate.right_model_id,
            &a.candidate.outcome_id,
        )
        .cmp(&candidate_id(
            &b.candidate.left_model_id,
            &b.candidate.right_model_id,
            &b.candidate.outcome_id,
        ))
    });

    let mut metrics_by_id = BTreeMap::new();
    for metric in &input.metrics {
        if metric.candidate_id.trim().is_empty() {
            return Err(ScientificPlanningTraceError::EmptyCandidateId);
        }
        if !metric.eig_per_cost.is_finite() || !metric.pragmatic_risk.is_finite() {
            return Err(ScientificPlanningTraceError::NonFiniteMetric(
                metric.candidate_id.clone(),
            ));
        }
        if metrics_by_id
            .insert(metric.candidate_id.clone(), metric)
            .is_some()
        {
            return Err(ScientificPlanningTraceError::DuplicateMetricId(
                metric.candidate_id.clone(),
            ));
        }
    }

    let expected_ids: BTreeSet<String> = gap_assessments
        .iter()
        .map(|a| {
            candidate_id(
                &a.candidate.left_model_id,
                &a.candidate.right_model_id,
                &a.candidate.outcome_id,
            )
        })
        .collect();

    for expected in &expected_ids {
        if !metrics_by_id.contains_key(expected) {
            return Err(ScientificPlanningTraceError::MissingMetrics(expected.clone()));
        }
    }
    for supplied in metrics_by_id.keys() {
        if !expected_ids.contains(supplied) {
            return Err(ScientificPlanningTraceError::ExtraMetrics(supplied.clone()));
        }
    }

    let candidates = gap_assessments
        .iter()
        .map(|assessment| {
            let id = candidate_id(
                &assessment.candidate.left_model_id,
                &assessment.candidate.right_model_id,
                &assessment.candidate.outcome_id,
            );
            let metric = metrics_by_id
                .get(&id)
                .expect("validated exact metric coverage");
            let mut candidate = expected_candidate(assessment);
            candidate.eig_per_cost = metric.eig_per_cost;
            candidate.pragmatic_risk = metric.pragmatic_risk;
            candidate
        })
        .collect::<Vec<_>>();

    let frontier_result = frontier(&candidates)
        .map_err(|e| ScientificPlanningTraceError::Frontier(e.to_string()))?;
    let admissibility = admit(&frontier_result, &input.policy)
        .map_err(|e: ScientificAdmissibilityError| {
            ScientificPlanningTraceError::Admissibility(e.to_string())
        })?;

    let mut canonical_gap_plan = input.gap_plan.clone();
    canonical_gap_plan.assessments = gap_assessments;
    let mut canonical_metrics = input.metrics.clone();
    canonical_metrics.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));

    #[derive(Serialize)]
    struct DigestMaterial<'a> {
        schema_version: &'static str,
        gap_plan: &'a GapAwareOutcomePlan,
        metrics: &'a [ScientificPlanningMetrics],
        frontier: &'a ScientificDecisionFrontier,
        policy: &'a ScientificAdmissibilityPolicy,
        admissibility: &'a ScientificAdmissibilityResult,
    }

    let material = DigestMaterial {
        schema_version: SCHEMA_VERSION,
        gap_plan: &canonical_gap_plan,
        metrics: &canonical_metrics,
        frontier: &frontier_result,
        policy: &input.policy,
        admissibility: &admissibility,
    };
    let bytes = serde_json::to_vec(&material)
        .map_err(|e| ScientificPlanningTraceError::Serialization(e.to_string()))?;
    let digest = format!("{:x}", Sha256::digest(bytes));

    Ok(ScientificPlanningTrace {
        schema_version: SCHEMA_VERSION.into(),
        gap_plan: canonical_gap_plan,
        metrics: canonical_metrics,
        gap_candidate_ids: expected_ids.into_iter().collect(),
        frontier: frontier_result,
        admissibility,
        policy: input.policy,
        artifact_digest: digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
    use symthaea_core::scientific_active_inference::ScientificTrajectoryForecast;
    use symthaea_core::scientific_outcome_likelihood::{
        OutcomeLikelihoodModel, ScientificOutcomePrototype,
    };
    use symthaea_core::scientific_trajectory_outcome_discrimination::discover;
    use symthaea_evidence_plane::gap_aware_outcome_planner::{
        plan, OutcomeEvidenceMapping,
    };

    fn hv(seed: u64) -> ContinuousHV {
        ContinuousHV::random(HDC_DIMENSION, seed)
    }

    fn input() -> ScientificPlanningTraceInput {
        let forecasts = vec![
            ScientificTrajectoryForecast {
                model_id: "a".into(),
                lineage: "la".into(),
                prior: 0.5,
                horizon_seconds: 10.0,
                predicted_state: hv(1),
                outcome_probabilities: vec![],
            },
            ScientificTrajectoryForecast {
                model_id: "b".into(),
                lineage: "lb".into(),
                prior: 0.5,
                horizon_seconds: 10.0,
                predicted_state: hv(2),
                outcome_probabilities: vec![],
            },
        ];
        let outcomes = vec![
            ScientificOutcomePrototype {
                outcome_id: "o1".into(),
                prototype: hv(1),
            },
            ScientificOutcomePrototype {
                outcome_id: "o2".into(),
                prototype: hv(2),
            },
        ];
        let discrimination =
            discover(&forecasts, &outcomes, &OutcomeLikelihoodModel::default()).unwrap();
        let gaps = ["replication", "provenance"]
            .into_iter()
            .map(String::from)
            .collect::<BTreeSet<_>>();
        let mappings = vec![
            OutcomeEvidenceMapping {
                outcome_id: "o1".into(),
                advances_predicates: ["replication"].into_iter().map(String::from).collect(),
            },
            OutcomeEvidenceMapping {
                outcome_id: "o2".into(),
                advances_predicates: ["replication", "provenance"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            },
        ];
        let gap_plan = plan(&discrimination, &mappings, &gaps).unwrap();
        let metrics = gap_plan
            .assessments
            .iter()
            .enumerate()
            .map(|(i, a)| ScientificPlanningMetrics {
                candidate_id: candidate_id(
                    &a.candidate.left_model_id,
                    &a.candidate.right_model_id,
                    &a.candidate.outcome_id,
                ),
                eig_per_cost: 0.5 + i as f64,
                pragmatic_risk: 0.25 + i as f64 * 0.1,
            })
            .collect();
        ScientificPlanningTraceInput {
            gap_plan,
            metrics,
            policy: ScientificAdmissibilityPolicy::default(),
        }
    }

    #[test]
    fn builds_full_trace_and_preserves_lineage() {
        let trace = build(&input()).unwrap();
        assert_eq!(trace.schema_version, SCHEMA_VERSION);
        assert!(!trace.artifact_digest.is_empty());
        assert_eq!(trace.frontier.assessments.len(), trace.gap_candidate_ids.len());
        for assessment in &trace.frontier.assessments {
            assert!(!assessment.candidate.left_lineage.is_empty());
            assert!(!assessment.candidate.right_lineage.is_empty());
        }
    }

    #[test]
    fn missing_metrics_are_rejected() {
        let mut i = input();
        i.metrics.pop();
        assert!(matches!(
            build(&i),
            Err(ScientificPlanningTraceError::MissingMetrics(_))
        ));
    }

    #[test]
    fn extra_metrics_are_rejected() {
        let mut i = input();
        i.metrics.push(ScientificPlanningMetrics {
            candidate_id: "unknown".into(),
            eig_per_cost: 1.0,
            pragmatic_risk: 1.0,
        });
        assert!(matches!(
            build(&i),
            Err(ScientificPlanningTraceError::ExtraMetrics(_))
        ));
    }

    #[test]
    fn duplicate_metrics_are_rejected() {
        let mut i = input();
        i.metrics.push(i.metrics[0].clone());
        assert!(matches!(
            build(&i),
            Err(ScientificPlanningTraceError::DuplicateMetricId(_))
        ));
    }

    #[test]
    fn input_permutation_does_not_change_trace_or_digest() {
        let mut a = input();
        let mut b = input();
        b.gap_plan.assessments.reverse();
        b.metrics.reverse();
        let ta = build(&a).unwrap();
        let tb = build(&b).unwrap();
        assert_eq!(ta, tb);
        assert_eq!(ta.artifact_digest, tb.artifact_digest);
        a.metrics[0].eig_per_cost += 0.125;
        assert_ne!(build(&a).unwrap().artifact_digest, ta.artifact_digest);
    }

    #[test]
    fn policy_can_only_remove_frontier_members() {
        let mut i = input();
        i.policy.min_eig_per_cost = Some(1.5);
        let trace = build(&i).unwrap();
        let frontier_ids: BTreeSet<_> = trace
            .frontier
            .assessments
            .iter()
            .filter(|a| a.is_frontier)
            .map(|a| a.candidate.candidate_id.clone())
            .collect();
        let admissible_ids: BTreeSet<_> = trace
            .admissibility
            .admissible
            .iter()
            .map(|a| a.candidate.candidate_id.clone())
            .collect();
        assert!(admissible_ids.is_subset(&frontier_ids));
    }

    #[test]
    fn digest_changes_when_policy_changes() {
        let a = build(&input()).unwrap();
        let mut i = input();
        i.policy.min_gap_coverage = Some(1.0);
        let b = build(&i).unwrap();
        assert_ne!(a.artifact_digest, b.artifact_digest);
    }
}
