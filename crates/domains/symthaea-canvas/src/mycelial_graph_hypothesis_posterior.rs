// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact, bounded posterior over feasible combinations of candidate mycelial fusions.
//!
//! This v1 posterior makes one explicit assumption: candidate classifier outputs
//! are conditionally independent Bernoulli probabilities *before* conditioning
//! on the pairwise incompatibility constraints already defined by the fusion
//! counterfactual contract. It is not a learned joint CRF and it does not infer
//! dependencies that are absent from the supplied evidence.
//!
//! Small candidate sets are exactly enumerated. Infeasible combinations receive
//! no posterior mass. If all feasible combinations have zero prior probability,
//! evaluation fails closed rather than inventing a posterior.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_fusion_counterfactual::{
    MycelialFusionCandidate, MycelialFusionCounterfactualError,
    evaluate_mycelial_fusion_counterfactuals,
};
use crate::mycelial_reconstruction_metrics::AnnotatedMycelialSkeleton;

pub const MYCELIAL_GRAPH_HYPOTHESIS_POSTERIOR_SCHEMA_VERSION: u16 = 1;
/// Exact enumeration is bounded at 2^16 = 65,536 graph hypotheses.
pub const MAX_MYCELIAL_POSTERIOR_CANDIDATES: usize = 16;
pub const MAX_MYCELIAL_POSTERIOR_HYPOTHESES: usize = 1 << MAX_MYCELIAL_POSTERIOR_CANDIDATES;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialGraphHypothesisPrior {
    /// Product Bernoulli prior conditioned on explicit pairwise incompatibilities.
    ConditionalIndependentBernoulli,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialFusionConflictPair {
    pub first_candidate_id: String,
    pub second_candidate_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFusionCandidateMarginal {
    pub candidate_id: String,
    /// Original calibrated classifier output before feasibility conditioning.
    pub prior_probability: f64,
    /// Marginal inclusion probability after conditioning on feasible combinations.
    pub posterior_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialGraphHypothesis {
    /// Sorted candidate IDs accepted in this exact graph hypothesis.
    pub accepted_candidate_ids: Vec<String>,
    /// Product-Bernoulli log prior before conditioning; None means exact zero mass.
    pub log_prior_probability: Option<f64>,
    /// Normalized posterior probability; zero for feasible but zero-prior hypotheses.
    pub posterior_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialGraphHypothesisPosterior {
    pub schema_version: u16,
    pub prior: MycelialGraphHypothesisPrior,
    pub source_graph_digest: String,
    pub candidate_inventory_digest: String,
    pub candidate_ids: Vec<String>,
    pub conflicts: Vec<MycelialFusionConflictPair>,
    /// Total Boolean assignments considered (2^candidate_ids.len()).
    pub total_hypothesis_count: usize,
    /// Feasible Boolean assignments; includes feasible assignments with zero prior mass.
    pub feasible_hypothesis_count: usize,
    /// Feasible assignments with strictly positive prior mass.
    pub nonzero_prior_feasible_hypothesis_count: usize,
    pub infeasible_hypothesis_count: usize,
    /// Log of total prior mass retained by feasibility conditioning.
    pub log_feasible_prior_mass: f64,
    pub posterior_entropy_nats: f64,
    pub candidate_marginals: Vec<MycelialFusionCandidateMarginal>,
    /// Deterministically ordered by ascending bit mask over sorted candidate IDs.
    pub hypotheses: Vec<MycelialGraphHypothesis>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialGraphHypothesisPosteriorError {
    TooManyCandidates(usize),
    InvalidFusionCandidates(String),
    ZeroFeasiblePosteriorMass,
    ReportMismatch,
}

impl fmt::Display for MycelialGraphHypothesisPosteriorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyCandidates(count) => write!(
                f,
                "exact graph-hypothesis candidate count {count} exceeds {MAX_MYCELIAL_POSTERIOR_CANDIDATES}"
            ),
            Self::InvalidFusionCandidates(reason) => {
                write!(f, "invalid fusion candidates for graph posterior: {reason}")
            }
            Self::ZeroFeasiblePosteriorMass => write!(
                f,
                "all feasible graph hypotheses have zero prior probability"
            ),
            Self::ReportMismatch => write!(
                f,
                "graph-hypothesis posterior does not match independent recomputation"
            ),
        }
    }
}

impl Error for MycelialGraphHypothesisPosteriorError {}

fn log_bernoulli_probability(probability: f64, accepted: bool) -> f64 {
    if accepted {
        if probability == 0.0 {
            f64::NEG_INFINITY
        } else {
            probability.ln()
        }
    } else if probability == 1.0 {
        f64::NEG_INFINITY
    } else {
        (-probability).ln_1p()
    }
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let maximum = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return maximum;
    }
    let scaled_sum: f64 = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| (value - maximum).exp())
        .sum();
    maximum + scaled_sum.ln()
}

fn conflict_pairs_from_impacts(
    impacts: &[crate::mycelial_fusion_counterfactual::MycelialFusionCandidateImpact],
) -> Vec<MycelialFusionConflictPair> {
    let mut conflicts = Vec::new();
    for impact in impacts {
        for other_id in &impact.conflicting_candidate_ids {
            if impact.candidate_id < *other_id {
                conflicts.push(MycelialFusionConflictPair {
                    first_candidate_id: impact.candidate_id.clone(),
                    second_candidate_id: other_id.clone(),
                });
            }
        }
    }
    conflicts.sort_by(|left, right| {
        left.first_candidate_id
            .cmp(&right.first_candidate_id)
            .then_with(|| left.second_candidate_id.cmp(&right.second_candidate_id))
    });
    conflicts.dedup();
    conflicts
}

pub fn evaluate_mycelial_graph_hypothesis_posterior(
    skeleton: &AnnotatedMycelialSkeleton,
    candidates: &[MycelialFusionCandidate],
) -> Result<
    MycelialGraphHypothesisPosterior,
    MycelialGraphHypothesisPosteriorError,
> {
    if candidates.len() > MAX_MYCELIAL_POSTERIOR_CANDIDATES {
        return Err(MycelialGraphHypothesisPosteriorError::TooManyCandidates(
            candidates.len(),
        ));
    }

    // Reuse the existing counterfactual validator and conflict detector. This
    // keeps candidate semantics and incompatibility rules in one authoritative place.
    let counterfactual = evaluate_mycelial_fusion_counterfactuals(
        skeleton,
        candidates,
        &[],
        &[],
    )
    .map_err(|error: MycelialFusionCounterfactualError| {
        MycelialGraphHypothesisPosteriorError::InvalidFusionCandidates(error.to_string())
    })?;

    let mut candidate_probabilities: BTreeMap<String, f64> = BTreeMap::new();
    for impact in &counterfactual.candidates {
        candidate_probabilities.insert(impact.candidate_id.clone(), impact.fusion_probability);
    }
    let candidate_ids: Vec<_> = candidate_probabilities.keys().cloned().collect();
    let probabilities: Vec<_> = candidate_ids
        .iter()
        .map(|candidate_id| candidate_probabilities[candidate_id])
        .collect();
    let candidate_indices: BTreeMap<_, _> = candidate_ids
        .iter()
        .enumerate()
        .map(|(index, candidate_id)| (candidate_id.as_str(), index))
        .collect();
    let conflicts = conflict_pairs_from_impacts(&counterfactual.candidates);
    let conflict_indices: Vec<_> = conflicts
        .iter()
        .map(|pair| {
            (
                candidate_indices[pair.first_candidate_id.as_str()],
                candidate_indices[pair.second_candidate_id.as_str()],
            )
        })
        .collect();

    let total_hypothesis_count = 1usize << candidate_ids.len();
    let mut hypotheses = Vec::new();
    let mut feasible_log_priors = Vec::new();
    let mut feasible_masks = Vec::new();
    let mut infeasible_hypothesis_count = 0usize;

    for mask in 0..total_hypothesis_count {
        let feasible = conflict_indices
            .iter()
            .all(|(left, right)| mask & (1usize << left) == 0 || mask & (1usize << right) == 0);
        if !feasible {
            infeasible_hypothesis_count += 1;
            continue;
        }

        let mut log_prior = 0.0_f64;
        let mut accepted_candidate_ids = Vec::new();
        for (index, candidate_id) in candidate_ids.iter().enumerate() {
            let accepted = mask & (1usize << index) != 0;
            let term = log_bernoulli_probability(probabilities[index], accepted);
            if !term.is_finite() {
                log_prior = f64::NEG_INFINITY;
                break;
            }
            log_prior += term;
            if accepted {
                accepted_candidate_ids.push(candidate_id.clone());
            }
        }
        let log_prior_probability = log_prior.is_finite().then_some(log_prior);
        hypotheses.push(MycelialGraphHypothesis {
            accepted_candidate_ids,
            log_prior_probability,
            posterior_probability: 0.0,
        });
        feasible_masks.push(mask);
        feasible_log_priors.push(log_prior);
    }

    let log_feasible_prior_mass = log_sum_exp(&feasible_log_priors);
    if !log_feasible_prior_mass.is_finite() {
        return Err(MycelialGraphHypothesisPosteriorError::ZeroFeasiblePosteriorMass);
    }

    let mut posterior_entropy_nats = 0.0;
    let mut candidate_marginal_mass = vec![0.0_f64; candidate_ids.len()];
    let mut nonzero_prior_feasible_hypothesis_count = 0usize;
    for ((hypothesis, mask), log_prior) in hypotheses
        .iter_mut()
        .zip(feasible_masks.iter())
        .zip(feasible_log_priors.iter())
    {
        if log_prior.is_finite() {
            let probability = (*log_prior - log_feasible_prior_mass).exp();
            hypothesis.posterior_probability = probability;
            nonzero_prior_feasible_hypothesis_count += 1;
            if probability > 0.0 {
                posterior_entropy_nats -= probability * probability.ln();
            }
            for (index, marginal_mass) in candidate_marginal_mass.iter_mut().enumerate() {
                if mask & (1usize << index) != 0 {
                    *marginal_mass += probability;
                }
            }
        }
    }

    let candidate_marginals = candidate_ids
        .iter()
        .zip(probabilities.iter())
        .zip(candidate_marginal_mass.iter())
        .map(|((candidate_id, prior_probability), posterior_probability)| {
            MycelialFusionCandidateMarginal {
                candidate_id: candidate_id.clone(),
                prior_probability: *prior_probability,
                posterior_probability: *posterior_probability,
            }
        })
        .collect();

    Ok(MycelialGraphHypothesisPosterior {
        schema_version: MYCELIAL_GRAPH_HYPOTHESIS_POSTERIOR_SCHEMA_VERSION,
        prior: MycelialGraphHypothesisPrior::ConditionalIndependentBernoulli,
        source_graph_digest: counterfactual.source_graph_digest,
        candidate_inventory_digest: counterfactual.candidate_inventory_digest,
        candidate_ids,
        conflicts,
        total_hypothesis_count,
        feasible_hypothesis_count: hypotheses.len(),
        nonzero_prior_feasible_hypothesis_count,
        infeasible_hypothesis_count,
        log_feasible_prior_mass,
        posterior_entropy_nats,
        candidate_marginals,
        hypotheses,
    })
}

impl MycelialGraphHypothesisPosterior {
    pub fn validate_against(
        &self,
        skeleton: &AnnotatedMycelialSkeleton,
        candidates: &[MycelialFusionCandidate],
    ) -> Result<(), MycelialGraphHypothesisPosteriorError> {
        let expected =
            evaluate_mycelial_graph_hypothesis_posterior(skeleton, candidates)?;
        if self != &expected {
            return Err(MycelialGraphHypothesisPosteriorError::ReportMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_fusion_counterfactual::{
        MycelialFusionBridgeLengthBasis, MycelialFusionTarget,
    };
    use crate::mycelial_reconstruction_metrics::{
        AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind,
        MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION, ReconstructionDimensionality,
        ReconstructionLengthUnit, ReconstructionPoint3,
    };

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn skeleton() -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
            schema_version: MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "frame-posterior".to_string(),
            coordinate_frame_id: "microscope-stage-posterior".to_string(),
            calibration_digest: digest('b'),
            source_observation_digest: digest('a'),
            dimensionality: ReconstructionDimensionality::TwoD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: "human-annotation".to_string(),
            producer_version: "v1".to_string(),
            junctions: vec![
                AnnotatedJunction {
                    id: "A".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 0.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "B".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 1.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "C".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 10.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "D".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 11.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "E".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 20.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "F".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 21.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "G".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 30.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "H".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 31.0, y: 0.0, z: 0.0 },
                },
            ],
            segments: vec![
                AnnotatedHyphaSegment {
                    id: "ab".to_string(),
                    start_junction_id: "A".to_string(),
                    end_junction_id: "B".to_string(),
                    length: 1.0,
                },
                AnnotatedHyphaSegment {
                    id: "cd".to_string(),
                    start_junction_id: "C".to_string(),
                    end_junction_id: "D".to_string(),
                    length: 1.0,
                },
                AnnotatedHyphaSegment {
                    id: "ef".to_string(),
                    start_junction_id: "E".to_string(),
                    end_junction_id: "F".to_string(),
                    length: 1.0,
                },
                AnnotatedHyphaSegment {
                    id: "gh".to_string(),
                    start_junction_id: "G".to_string(),
                    end_junction_id: "H".to_string(),
                    length: 1.0,
                },
            ],
        }
    }

    fn candidate(id: &str, source: &str, target: &str, probability: f64, group: Option<&str>) -> MycelialFusionCandidate {
        MycelialFusionCandidate {
            candidate_id: id.to_string(),
            source_tip_junction_id: source.to_string(),
            target: MycelialFusionTarget::TipJunction {
                junction_id: target.to_string(),
            },
            bridge_length: 1.0,
            bridge_length_unit: ReconstructionLengthUnit::Micrometer,
            bridge_length_basis: MycelialFusionBridgeLengthBasis::Measured,
            bridge_length_evidence_digest: digest('e'),
            fusion_probability: probability,
            calibration_policy_id: "temperature-scaled-v1".to_string(),
            probability_evidence_digest: digest('f'),
            conflict_group_id: group.map(str::to_string),
        }
    }

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() < 1.0e-12, "{left} != {right}");
    }

    #[test]
    fn independent_candidates_match_product_bernoulli_probabilities() {
        let source = skeleton();
        let candidates = vec![
            candidate("one", "B", "D", 0.2, None),
            candidate("two", "F", "H", 0.7, None),
        ];
        let report =
            evaluate_mycelial_graph_hypothesis_posterior(&source, &candidates).unwrap();
        assert_eq!(report.total_hypothesis_count, 4);
        assert_eq!(report.feasible_hypothesis_count, 4);
        assert_eq!(report.infeasible_hypothesis_count, 0);
        assert_eq!(report.candidate_ids, vec!["one".to_string(), "two".to_string()]);
        close(report.candidate_marginals[0].posterior_probability, 0.2);
        close(report.candidate_marginals[1].posterior_probability, 0.7);
        close(report.hypotheses[0].posterior_probability, 0.24);
        close(report.hypotheses[1].posterior_probability, 0.06);
        close(report.hypotheses[2].posterior_probability, 0.56);
        close(report.hypotheses[3].posterior_probability, 0.14);
        report.validate_against(&source, &candidates).unwrap();
    }

    #[test]
    fn conflict_conditioning_removes_impossible_joint_hypothesis_and_renormalizes() {
        let source = skeleton();
        let candidates = vec![
            candidate("one", "B", "D", 0.8, Some("same-contact")),
            candidate("two", "D", "B", 0.6, Some("same-contact")),
        ];
        let report =
            evaluate_mycelial_graph_hypothesis_posterior(&source, &candidates).unwrap();
        assert_eq!(report.total_hypothesis_count, 4);
        assert_eq!(report.feasible_hypothesis_count, 3);
        assert_eq!(report.infeasible_hypothesis_count, 1);
        assert_eq!(report.conflicts.len(), 1);
        close(report.candidate_marginals[0].posterior_probability, 0.6153846153846154);
        close(report.candidate_marginals[1].posterior_probability, 0.23076923076923078);
        close(report.hypotheses.iter().map(|hypothesis| hypothesis.posterior_probability).sum(), 1.0);
    }

    #[test]
    fn hard_conflicting_probabilities_with_no_feasible_prior_mass_fail_closed() {
        let source = skeleton();
        let candidates = vec![
            candidate("one", "B", "D", 1.0, Some("same-contact")),
            candidate("two", "D", "B", 1.0, Some("same-contact")),
        ];
        assert_eq!(
            evaluate_mycelial_graph_hypothesis_posterior(&source, &candidates).unwrap_err(),
            MycelialGraphHypothesisPosteriorError::ZeroFeasiblePosteriorMass
        );
    }

    #[test]
    fn empty_candidate_inventory_has_one_empty_graph_hypothesis() {
        let report =
            evaluate_mycelial_graph_hypothesis_posterior(&skeleton(), &[]).unwrap();
        assert_eq!(report.total_hypothesis_count, 1);
        assert_eq!(report.feasible_hypothesis_count, 1);
        assert!(report.hypotheses[0].accepted_candidate_ids.is_empty());
        close(report.hypotheses[0].posterior_probability, 1.0);
    }

    #[test]
    fn report_tampering_is_detected_by_independent_recomputation() {
        let source = skeleton();
        let candidates = vec![candidate("one", "B", "D", 0.2, None)];
        let mut report =
            evaluate_mycelial_graph_hypothesis_posterior(&source, &candidates).unwrap();
        report.hypotheses[0].posterior_probability = 0.9;
        assert_eq!(
            report.validate_against(&source, &candidates).unwrap_err(),
            MycelialGraphHypothesisPosteriorError::ReportMismatch
        );
    }
}
