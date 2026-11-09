// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Bounded two-stage static-model robust diagnostic policy oracle.
//!
//! This HLB/FEP adapter compares a globally fixed latent measurement model
//! against a rectangular relaxation that may change the stage-two model at
//! each observed first-stage history. The policy tree is shared across model
//! vertices and branches only on public observations.
//!
//! This is a bounded finite two-stage oracle, not a general robust-MDP solver.
//! It reuses the static-versus-rectangular semantics of the connectomics
//! research programme without duplicating its domain-specific data contract.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

const PROB_TOL: f64 = 1e-9;
const RISK_TOL: f64 = 1e-10;
pub const STOP_ACTION_ID: &str = "STOP";

/// Terminal decision for the binary task query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StaticDiagnosticDecision {
    Support,
    Refute,
}

/// Declared feasibility state for an externally supplied diagnostic action.
/// STOP is an internal policy option and does not require an external receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StaticDiagnosticFeasibilityStatus {
    Feasible,
    Disallowed,
    Inaccessible,
    NotAssessed,
}

/// The risk/certificate outputs are conditional on this finite supplied model set.
/// The enum is intentionally explicit so they cannot be mistaken for an open-world
/// guarantee or evidence that the actual data-generating process is represented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StaticDiagnosticClaimScope {
    FiniteDeclaredModelSetOnly,
}

/// Caller-supplied action feasibility evidence. The digest is required to bind
/// the action protocol and eligibility decision. This pure evaluator checks
/// completeness and SHA-256 shape; it does not authenticate the receipt issuer
/// or independently establish real-world feasibility. Those checks belong
/// upstream before this input is constructed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticDiagnosticActionEligibility {
    pub action_id: String,
    pub status: StaticDiagnosticFeasibilityStatus,
    pub feasibility_receipt_sha256: String,
}

/// One fixed model vertex and its complete two-stage categorical kernels.
///
/// Tensor dimensions:
/// - stage one: world × first-stage history;
/// - stage two: history × local action × world × second-stage outcome;
/// - one-shot: action × world × one-shot outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticDiagnosticModelVertex {
    pub model_vertex_id: String,
    pub model_prior_probability: f64,
    pub world_prior_probabilities: Vec<f64>,
    pub stage_one_outcome_probabilities_by_world: Vec<Vec<f64>>,
    pub stage_two_outcome_probabilities: Vec<Vec<Vec<Vec<f64>>>>,
    pub one_shot_outcome_probabilities_by_action: Vec<Vec<Vec<f64>>>,
}

/// Bounded finite two-stage problem. STOP is an explicit zero-cost option at
/// every second-stage history; the no-probe baseline stops before stage one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticModelDiagnosticProblem {
    pub scenario_id: String,
    pub world_ids: Vec<String>,
    /// True means Support is the correct terminal decision in this world.
    pub world_is_query_true: Vec<bool>,
    pub model_vertices: Vec<StaticDiagnosticModelVertex>,
    pub root_action_id: String,
    pub root_action_cost: f64,
    /// Every externally selectable action must have an explicit feasibility
    /// state and a digest-bound feasibility receipt before policy search.
    pub action_eligibility: Vec<StaticDiagnosticActionEligibility>,
    pub first_stage_history_ids: Vec<String>,
    pub second_stage_action_ids_by_history: Vec<Vec<String>>,
    pub second_stage_action_costs_by_history: Vec<Vec<f64>>,
    pub second_stage_outcome_ids: Vec<String>,
    pub one_shot_action_ids: Vec<String>,
    pub one_shot_action_costs: Vec<f64>,
    pub one_shot_outcome_ids: Vec<String>,
    pub maximum_worst_branch_cost: f64,
    pub certificate_maximum_robust_risk: f64,
}

/// Exact search budget. The required number of policy evaluations is computed
/// before search; exhaustion returns an error, never a partial policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticModelDiagnosticConfig {
    pub maximum_policy_evaluations: u64,
}

impl Default for StaticModelDiagnosticConfig {
    fn default() -> Self {
        Self {
            maximum_policy_evaluations: 100_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticDiagnosticHistoryAction {
    pub first_stage_history_id: String,
    pub action_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticDiagnosticTerminalDecision {
    pub first_stage_history_id: String,
    pub second_stage_outcome_id: String,
    pub decision: StaticDiagnosticDecision,
}

/// Public policy tree: it contains observations and actions, never hidden
/// model-vertex or ground-truth world identifiers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticDiagnosticPolicyTree {
    pub root_action_id: String,
    pub second_stage_action_by_history: Vec<StaticDiagnosticHistoryAction>,
    pub terminal_decision_by_history_and_outcome: Vec<StaticDiagnosticTerminalDecision>,
    pub worst_branch_cost: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticDiagnosticModelRisk {
    pub model_vertex_id: String,
    pub risk: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticDiagnosticOneShotAssessment {
    pub action_id: String,
    pub action_cost: f64,
    pub worst_case_risk: f64,
    pub decision_by_outcome: Vec<(String, StaticDiagnosticDecision)>,
    pub per_model_risks: Vec<StaticDiagnosticModelRisk>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticDiagnosticOneShotResult {
    pub best_action_id: String,
    pub best_worst_case_risk: f64,
    pub action_assessments: Vec<StaticDiagnosticOneShotAssessment>,
}

/// Bayesian posterior after public observations. Model/world labels identify
/// entries in the declared manifest; no hidden identity is exposed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticDiagnosticPublicPosterior {
    pub world_probabilities: Vec<(String, f64)>,
    pub model_probabilities: Vec<(String, f64)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticModelDiagnosticReport {
    pub scenario_id: String,
    pub claim_scope: StaticDiagnosticClaimScope,
    pub policy_evaluations: u64,
    pub static_robust_risk: f64,
    pub static_policy: StaticDiagnosticPolicyTree,
    pub static_per_model_risks: Vec<StaticDiagnosticModelRisk>,
    pub rectangular_minimax_risk: f64,
    pub rectangular_policy: StaticDiagnosticPolicyTree,
    pub rectangular_per_root_model_risks: Vec<StaticDiagnosticModelRisk>,
    pub rectangular_risk_of_static_policy: f64,
    pub clairvoyant_lower_bound_risk: f64,
    pub bayesian_model_average_optimal_risk: f64,
    pub no_probe_robust_risk: f64,
    pub one_shot: StaticDiagnosticOneShotResult,
    pub static_rectangular_conservatism_gap: f64,
    pub static_gain_over_best_one_shot: f64,
    /// Stress result when stage one and stage two use different fixed vertices.
    /// This is not a static-model guarantee or the rectangular relaxation.
    pub cross_stage_switching_stress_risk: f64,
    pub static_robust_certificate: bool,
    pub rectangular_robust_certificate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticModelDiagnosticReceipt {
    pub schema_version: u32,
    pub source_git_sha: String,
    pub input_sha256: String,
    pub task_specification_sha256: String,
    pub model_kernel_sha256: String,
    pub static_policy_sha256: String,
    pub rectangular_policy_sha256: String,
    pub result_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticModelDiagnosticEvaluation {
    pub report: StaticModelDiagnosticReport,
    pub receipt: StaticModelDiagnosticReceipt,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum StaticModelDiagnosticError {
    #[error("scenario identifier must not be empty")]
    EmptyScenarioId,
    #[error("identity set is empty, contains blanks, or contains duplicates: {0}")]
    InvalidIdentitySet(&'static str),
    #[error("world count {actual} is outside 2..={maximum}")]
    InvalidWorldCount { actual: usize, maximum: usize },
    #[error("model count {actual} is outside 2..={maximum}")]
    InvalidModelCount { actual: usize, maximum: usize },
    #[error("history count {actual} is outside 1..={maximum}")]
    InvalidHistoryCount { actual: usize, maximum: usize },
    #[error("stage-two outcome count {actual} is outside 1..={maximum}")]
    InvalidStageTwoOutcomeCount { actual: usize, maximum: usize },
    #[error("one-shot outcome count {actual} is outside 1..={maximum}")]
    InvalidOneShotOutcomeCount { actual: usize, maximum: usize },
    #[error("invalid model prior for vertex {0}")]
    InvalidModelPrior(String),
    #[error("invalid world prior for vertex {0}")]
    InvalidWorldPrior(String),
    #[error("invalid stage-one tensor for vertex {0}")]
    InvalidStageOneTensor(String),
    #[error("invalid stage-two tensor for vertex {0}")]
    InvalidStageTwoTensor(String),
    #[error("invalid one-shot tensor for vertex {0}")]
    InvalidOneShotTensor(String),
    #[error("probability in {location} at {index} is invalid")]
    InvalidProbability { location: String, index: usize },
    #[error("probability mass in {location} sums to {sum}, expected one")]
    ProbabilityMassNotNormalized { location: String, sum: f64 },
    #[error("invalid action menu/costs for history {0}")]
    InvalidActionMenu(String),
    #[error("action is missing a valid feasible receipt: {0}")]
    InvalidActionEligibility(String),
    #[error("STOP must have zero cost and deterministically emit the STOP outcome")]
    InvalidStopContract,
    #[error("root/branch costs are invalid")]
    InvalidCost,
    #[error("certificate risk threshold is invalid")]
    InvalidCertificateThreshold,
    #[error("search budget must be positive")]
    EmptySearchBudget,
    #[error("search requires {required} policy evaluations, budget is {budget}")]
    SearchBudgetExceeded { required: u64, budget: u64 },
    #[error("no policy satisfies the branch-cost bound")]
    NoFeasiblePolicy,
    #[error("rectangular risk {rectangular_risk} is below static risk {static_risk}")]
    RectangularityInvariant { static_risk: f64, rectangular_risk: f64 },
    #[error("rectangular evaluation of the static policy {rectangular_risk} is below its static risk {static_risk}")]
    StaticPolicyRectangularityInvariant {
        static_risk: f64,
        rectangular_risk: f64,
    },
    #[error("clairvoyant lower bound {clairvoyant_risk} exceeds static robust risk {static_risk}")]
    ClairvoyanceInvariant { clairvoyant_risk: f64, static_risk: f64 },
    #[error("public observation history is unknown or has zero predictive probability")]
    ImpossiblePublicHistory,
    #[error("source Git SHA must be 40 lowercase hexadecimal characters")]
    InvalidSourceGitSha,
    #[error("receipt serialization failed")]
    SerializationFailure,
}

#[derive(Debug, Clone)]
struct CandidatePolicy {
    action_indices: Vec<usize>,
    decisions: Vec<StaticDiagnosticDecision>,
    static_per_model_risks: Vec<f64>,
    static_robust_risk: f64,
    rectangular_per_root_model_risks: Vec<f64>,
    rectangular_robust_risk: f64,
    worst_branch_cost: f64,
}

#[derive(Debug, Clone)]
struct OneShotCandidate {
    decisions: Vec<StaticDiagnosticDecision>,
    per_model_risks: Vec<f64>,
    robust_risk: f64,
}

/// Exhaustive finite policy-tree comparison. A single model vertex controls
/// each full static trajectory. The rectangular comparator may change the
/// second-stage vertex separately at each observed history.
pub fn evaluate_static_model_diagnostic_problem(
    problem: &StaticModelDiagnosticProblem,
    config: StaticModelDiagnosticConfig,
    source_git_sha: &str,
) -> Result<StaticModelDiagnosticEvaluation, StaticModelDiagnosticError> {
    validate_problem(problem)?;
    if config.maximum_policy_evaluations == 0 {
        return Err(StaticModelDiagnosticError::EmptySearchBudget);
    }
    if !is_lower_hex(source_git_sha, 40) {
        return Err(StaticModelDiagnosticError::InvalidSourceGitSha);
    }
    let required = required_policy_evaluations(problem)?;
    if required > config.maximum_policy_evaluations {
        return Err(StaticModelDiagnosticError::SearchBudgetExceeded {
            required,
            budget: config.maximum_policy_evaluations,
        });
    }

    let mut best_static: Option<CandidatePolicy> = None;
    let mut best_rectangular: Option<CandidatePolicy> = None;
    let mut best_bayesian_average = f64::INFINITY;
    let mut best_per_model = vec![f64::INFINITY; problem.model_vertices.len()];
    let mut policy_evaluations = 0_u64;
    let action_combinations = action_combination_count(problem)?;
    let decision_masks =
        1_u64 << (problem.first_stage_history_ids.len() * problem.second_stage_outcome_ids.len());

    for action_code in 0..action_combinations {
        let action_indices = decode_action_choices(problem, action_code);
        let worst_branch_cost = policy_worst_branch_cost(problem, &action_indices);
        if worst_branch_cost > problem.maximum_worst_branch_cost + RISK_TOL {
            continue;
        }
        for decision_mask in 0..decision_masks {
            policy_evaluations += 1;
            let decisions = decode_decisions(problem, decision_mask);
            let static_per_model_risks = problem
                .model_vertices
                .iter()
                .map(|vertex| static_policy_risk(problem, vertex, &action_indices, &decisions))
                .collect::<Vec<_>>();
            let static_robust_risk = static_per_model_risks.iter().copied().fold(0.0_f64, f64::max);
            let rectangular_per_root_model_risks = problem
                .model_vertices
                .iter()
                .map(|root_vertex| {
                    rectangular_policy_risk(problem, root_vertex, &action_indices, &decisions)
                })
                .collect::<Vec<_>>();
            let rectangular_robust_risk = rectangular_per_root_model_risks
                .iter()
                .copied()
                .fold(0.0_f64, f64::max);
            let bayesian_average_risk = static_per_model_risks
                .iter()
                .zip(&problem.model_vertices)
                .map(|(risk, vertex)| risk * vertex.model_prior_probability)
                .sum::<f64>();

            for (index, risk) in static_per_model_risks.iter().copied().enumerate() {
                best_per_model[index] = best_per_model[index].min(risk);
            }
            best_bayesian_average = best_bayesian_average.min(bayesian_average_risk);

            let candidate = CandidatePolicy {
                action_indices,
                decisions,
                static_per_model_risks,
                static_robust_risk,
                rectangular_per_root_model_risks,
                rectangular_robust_risk,
                worst_branch_cost,
            };
            if best_static.as_ref().is_none_or(|best| candidate_precedes(&candidate, best, true)) {
                best_static = Some(candidate.clone());
            }
            if best_rectangular
                .as_ref()
                .is_none_or(|best| candidate_precedes(&candidate, best, false))
            {
                best_rectangular = Some(candidate);
            }
        }
    }

    let best_static = best_static.ok_or(StaticModelDiagnosticError::NoFeasiblePolicy)?;
    let best_rectangular = best_rectangular.ok_or(StaticModelDiagnosticError::NoFeasiblePolicy)?;
    let clairvoyant_lower_bound_risk = best_per_model.iter().copied().fold(0.0_f64, f64::max);
    if best_rectangular.rectangular_robust_risk + RISK_TOL < best_static.static_robust_risk {
        return Err(StaticModelDiagnosticError::RectangularityInvariant {
            static_risk: best_static.static_robust_risk,
            rectangular_risk: best_rectangular.rectangular_robust_risk,
        });
    }
    if clairvoyant_lower_bound_risk > best_static.static_robust_risk + RISK_TOL {
        return Err(StaticModelDiagnosticError::ClairvoyanceInvariant {
            clairvoyant_risk: clairvoyant_lower_bound_risk,
            static_risk: best_static.static_robust_risk,
        });
    }

    let one_shot = evaluate_one_shot(problem)?;
    let one_shot_decision_masks = 1_u64 << problem.one_shot_outcome_ids.len();
    let one_shot_policy_evaluations =
        (problem.one_shot_action_ids.len() as u64) * one_shot_decision_masks;
    policy_evaluations += one_shot_policy_evaluations;
    let no_probe_robust_risk = no_probe_risk(problem);
    let rectangular_risk_of_static_policy = problem
        .model_vertices
        .iter()
        .map(|root_vertex| {
            rectangular_policy_risk(
                problem,
                root_vertex,
                &best_static.action_indices,
                &best_static.decisions,
            )
        })
        .fold(0.0_f64, f64::max);
    if rectangular_risk_of_static_policy + RISK_TOL < best_static.static_robust_risk {
        return Err(StaticModelDiagnosticError::StaticPolicyRectangularityInvariant {
            static_risk: best_static.static_robust_risk,
            rectangular_risk: rectangular_risk_of_static_policy,
        });
    }
    let cross_stage_switching_stress_risk = cross_stage_switching_risk(
        problem,
        &best_static.action_indices,
        &best_static.decisions,
    );

    let report = StaticModelDiagnosticReport {
        scenario_id: problem.scenario_id.clone(),
        claim_scope: StaticDiagnosticClaimScope::FiniteDeclaredModelSetOnly,
        policy_evaluations,
        static_robust_risk: best_static.static_robust_risk,
        static_policy: materialize_policy(
            problem,
            &best_static.action_indices,
            &best_static.decisions,
            best_static.worst_branch_cost,
        ),
        static_per_model_risks: materialize_risks(problem, &best_static.static_per_model_risks),
        rectangular_minimax_risk: best_rectangular.rectangular_robust_risk,
        rectangular_policy: materialize_policy(
            problem,
            &best_rectangular.action_indices,
            &best_rectangular.decisions,
            best_rectangular.worst_branch_cost,
        ),
        rectangular_per_root_model_risks: materialize_risks(
            problem,
            &best_rectangular.rectangular_per_root_model_risks,
        ),
        rectangular_risk_of_static_policy,
        clairvoyant_lower_bound_risk,
        bayesian_model_average_optimal_risk: best_bayesian_average,
        no_probe_robust_risk,
        static_rectangular_conservatism_gap: (best_rectangular.rectangular_robust_risk
            - best_static.static_robust_risk)
            .max(0.0),
        static_gain_over_best_one_shot: (one_shot.best_worst_case_risk
            - best_static.static_robust_risk)
            .max(0.0),
        one_shot,
        cross_stage_switching_stress_risk,
        static_robust_certificate: best_static.static_robust_risk
            <= problem.certificate_maximum_robust_risk,
        rectangular_robust_certificate: best_rectangular.rectangular_robust_risk
            <= problem.certificate_maximum_robust_risk,
    };
    let receipt = build_receipt(problem, config, source_git_sha, &report)?;
    Ok(StaticModelDiagnosticEvaluation { report, receipt })
}

/// Bayesian posterior from public observations only. The posterior is over
/// model and world hypotheses; no hidden identifier is read by the caller.
pub fn posterior_after_public_history(
    problem: &StaticModelDiagnosticProblem,
    first_stage_history_id: &str,
    second_stage_observation: Option<(&str, &str)>,
) -> Result<StaticDiagnosticPublicPosterior, StaticModelDiagnosticError> {
    validate_problem(problem)?;
    let history_index = problem
        .first_stage_history_ids
        .iter()
        .position(|id| id == first_stage_history_id)
        .ok_or(StaticModelDiagnosticError::ImpossiblePublicHistory)?;
    let mut joint_mass = vec![vec![0.0; problem.world_ids.len()]; problem.model_vertices.len()];
    let mut evidence_mass = 0.0;

    for (model_index, vertex) in problem.model_vertices.iter().enumerate() {
        for world_index in 0..problem.world_ids.len() {
            let mut mass = vertex.model_prior_probability
                * vertex.world_prior_probabilities[world_index]
                * vertex.stage_one_outcome_probabilities_by_world[world_index][history_index];
            if let Some((action_id, outcome_id)) = second_stage_observation {
                let action_index = problem.second_stage_action_ids_by_history[history_index]
                    .iter()
                    .position(|id| id == action_id)
                    .ok_or(StaticModelDiagnosticError::ImpossiblePublicHistory)?;
                let outcome_index = problem
                    .second_stage_outcome_ids
                    .iter()
                    .position(|id| id == outcome_id)
                    .ok_or(StaticModelDiagnosticError::ImpossiblePublicHistory)?;
                mass *= vertex.stage_two_outcome_probabilities[history_index][action_index]
                    [world_index][outcome_index];
            }
            joint_mass[model_index][world_index] = mass;
            evidence_mass += mass;
        }
    }
    if !evidence_mass.is_finite() || evidence_mass <= 0.0 {
        return Err(StaticModelDiagnosticError::ImpossiblePublicHistory);
    }

    let model_probabilities = problem
        .model_vertices
        .iter()
        .enumerate()
        .map(|(model_index, vertex)| {
            (
                vertex.model_vertex_id.clone(),
                joint_mass[model_index].iter().sum::<f64>() / evidence_mass,
            )
        })
        .collect();
    let world_probabilities = problem
        .world_ids
        .iter()
        .enumerate()
        .map(|(world_index, world_id)| {
            (
                world_id.clone(),
                joint_mass
                    .iter()
                    .map(|model_mass| model_mass[world_index])
                    .sum::<f64>()
                    / evidence_mass,
            )
        })
        .collect();
    Ok(StaticDiagnosticPublicPosterior {
        world_probabilities,
        model_probabilities,
    })
}

/// Exact two-world/two-model fixture. The calibration outcome identifies which
/// fixed measurement model is operating, while the world remains unknown.
pub fn symmetric_calibration_fixture() -> StaticModelDiagnosticProblem {
    let world_ids = vec!["QUERY_TRUE".to_owned(), "QUERY_FALSE".to_owned()];
    let histories = vec!["CAL-A".to_owned(), "CAL-B".to_owned()];
    let actions_by_history = vec![
        vec![STOP_ACTION_ID.to_owned(), "ACTION-A".to_owned(), "ACTION-B".to_owned()],
        vec![STOP_ACTION_ID.to_owned(), "ACTION-A".to_owned(), "ACTION-B".to_owned()],
    ];
    let outcomes = vec![
        STOP_ACTION_ID.to_owned(),
        "NEGATIVE".to_owned(),
        "POSITIVE".to_owned(),
    ];

    let vertices = ["MODEL-A", "MODEL-B"]
        .into_iter()
        .enumerate()
        .map(|(model_index, model_id)| {
            let calibration = if model_index == 0 {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            };
            let stage_one_outcome_probabilities_by_world =
                vec![calibration.clone(), calibration];
            let stage_two_outcome_probabilities = (0..histories.len())
                .map(|_| {
                    actions_by_history[0]
                        .iter()
                        .map(|action_id| {
                            (0..world_ids.len())
                                .map(|world_index| {
                                    if action_id == STOP_ACTION_ID {
                                        vec![1.0, 0.0, 0.0]
                                    } else {
                                        let truthful = (*action_id == "ACTION-A") == (model_index == 0);
                                        let world_true = world_index == 0;
                                        if truthful == world_true {
                                            vec![0.0, 0.0, 1.0]
                                        } else {
                                            vec![0.0, 1.0, 0.0]
                                        }
                                    }
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let one_shot_outcome_probabilities_by_action = ["ACTION-A", "ACTION-B"]
                .iter()
                .map(|action_id| {
                    (0..world_ids.len())
                        .map(|world_index| {
                            let truthful = (*action_id == "ACTION-A") == (model_index == 0);
                            if truthful == (world_index == 0) {
                                vec![0.0, 1.0]
                            } else {
                                vec![1.0, 0.0]
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            StaticDiagnosticModelVertex {
                model_vertex_id: model_id.to_owned(),
                model_prior_probability: 0.5,
                world_prior_probabilities: vec![0.5, 0.5],
                stage_one_outcome_probabilities_by_world,
                stage_two_outcome_probabilities,
                one_shot_outcome_probabilities_by_action,
            }
        })
        .collect();

    StaticModelDiagnosticProblem {
        scenario_id: "hlb-static-model-two-stage-v1".to_owned(),
        world_ids,
        world_is_query_true: vec![true, false],
        model_vertices: vertices,
        root_action_id: "CALIBRATE".to_owned(),
        root_action_cost: 1.0,
        action_eligibility: vec![
            StaticDiagnosticActionEligibility {
                action_id: "CALIBRATE".to_owned(),
                status: StaticDiagnosticFeasibilityStatus::Feasible,
                feasibility_receipt_sha256: "c".repeat(64),
            },
            StaticDiagnosticActionEligibility {
                action_id: "ACTION-A".to_owned(),
                status: StaticDiagnosticFeasibilityStatus::Feasible,
                feasibility_receipt_sha256: "d".repeat(64),
            },
            StaticDiagnosticActionEligibility {
                action_id: "ACTION-B".to_owned(),
                status: StaticDiagnosticFeasibilityStatus::Feasible,
                feasibility_receipt_sha256: "e".repeat(64),
            },
        ],
        first_stage_history_ids: histories,
        second_stage_action_ids_by_history: actions_by_history,
        second_stage_action_costs_by_history: vec![vec![0.0, 1.0, 3.0], vec![0.0, 3.0, 1.0]],
        second_stage_outcome_ids: outcomes,
        one_shot_action_ids: vec!["ACTION-A".to_owned(), "ACTION-B".to_owned()],
        one_shot_action_costs: vec![1.0, 1.0],
        one_shot_outcome_ids: vec!["NEGATIVE".to_owned(), "POSITIVE".to_owned()],
        maximum_worst_branch_cost: 2.0,
        certificate_maximum_robust_risk: 0.01,
    }
}

/// Variant where both model vertices produce the same calibration observation.
pub fn uninformative_calibration_fixture() -> StaticModelDiagnosticProblem {
    let mut problem = symmetric_calibration_fixture();
    problem.first_stage_history_ids = vec!["CAL-A".to_owned()];
    problem.second_stage_action_ids_by_history = vec![vec![
        STOP_ACTION_ID.to_owned(),
        "ACTION-A".to_owned(),
        "ACTION-B".to_owned(),
    ]];
    problem.second_stage_action_costs_by_history = vec![vec![0.0, 1.0, 1.0]];
    for vertex in &mut problem.model_vertices {
        vertex.stage_one_outcome_probabilities_by_world = vec![vec![1.0], vec![1.0]];
        vertex.stage_two_outcome_probabilities =
            vec![vertex.stage_two_outcome_probabilities[0].clone()];
    }
    problem.scenario_id = "hlb-static-model-uninformative-calibration-v1".to_owned();
    problem
}

fn validate_problem(
    problem: &StaticModelDiagnosticProblem,
) -> Result<(), StaticModelDiagnosticError> {
    if problem.scenario_id.trim().is_empty() {
        return Err(StaticModelDiagnosticError::EmptyScenarioId);
    }
    validate_ids(&problem.world_ids, "world")?;
    validate_ids(
        &problem
            .model_vertices
            .iter()
            .map(|vertex| vertex.model_vertex_id.clone())
            .collect::<Vec<_>>(),
        "model vertex",
    )?;
    validate_ids(&problem.first_stage_history_ids, "first-stage history")?;
    validate_ids(&problem.second_stage_outcome_ids, "second-stage outcome")?;
    validate_ids(&problem.one_shot_action_ids, "one-shot action")?;
    validate_ids(&problem.one_shot_outcome_ids, "one-shot outcome")?;

    if !(2..=4).contains(&problem.world_ids.len()) {
        return Err(StaticModelDiagnosticError::InvalidWorldCount {
            actual: problem.world_ids.len(),
            maximum: 4,
        });
    }
    if !(2..=4).contains(&problem.model_vertices.len()) {
        return Err(StaticModelDiagnosticError::InvalidModelCount {
            actual: problem.model_vertices.len(),
            maximum: 4,
        });
    }
    if problem.first_stage_history_ids.is_empty() || problem.first_stage_history_ids.len() > 4 {
        return Err(StaticModelDiagnosticError::InvalidHistoryCount {
            actual: problem.first_stage_history_ids.len(),
            maximum: 4,
        });
    }
    if problem.second_stage_outcome_ids.is_empty() || problem.second_stage_outcome_ids.len() > 4 {
        return Err(StaticModelDiagnosticError::InvalidStageTwoOutcomeCount {
            actual: problem.second_stage_outcome_ids.len(),
            maximum: 4,
        });
    }
    if problem.one_shot_outcome_ids.is_empty()
        || problem.one_shot_outcome_ids.len() > 4
        || problem.one_shot_action_ids.is_empty()
        || problem.one_shot_action_ids.len() > 4
        || problem.world_is_query_true.len() != problem.world_ids.len()
        || problem.second_stage_action_ids_by_history.len() != problem.first_stage_history_ids.len()
        || problem.second_stage_action_costs_by_history.len() != problem.first_stage_history_ids.len()
        || problem.one_shot_action_costs.len() != problem.one_shot_action_ids.len()
    {
        return Err(StaticModelDiagnosticError::InvalidActionMenu(
            "problem-level dimensions".to_owned(),
        ));
    }
    if !problem.root_action_cost.is_finite()
        || problem.root_action_cost < 0.0
        || !problem.maximum_worst_branch_cost.is_finite()
        || problem.maximum_worst_branch_cost < problem.root_action_cost
    {
        return Err(StaticModelDiagnosticError::InvalidCost);
    }
    if !problem.certificate_maximum_robust_risk.is_finite()
        || !(0.0..=1.0).contains(&problem.certificate_maximum_robust_risk)
    {
        return Err(StaticModelDiagnosticError::InvalidCertificateThreshold);
    }

    validate_ids(
        &problem
            .action_eligibility
            .iter()
            .map(|action| action.action_id.clone())
            .collect::<Vec<_>>(),
        "action eligibility",
    )?;
    let eligibility = problem
        .action_eligibility
        .iter()
        .map(|action| (action.action_id.as_str(), action))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut referenced_actions = std::collections::BTreeSet::new();
    referenced_actions.insert(problem.root_action_id.as_str());
    for menu in &problem.second_stage_action_ids_by_history {
        for action_id in menu.iter().filter(|id| id.as_str() != STOP_ACTION_ID) {
            referenced_actions.insert(action_id.as_str());
        }
    }
    for action_id in &problem.one_shot_action_ids {
        referenced_actions.insert(action_id.as_str());
    }
    for action_id in referenced_actions {
        let Some(action) = eligibility.get(action_id) else {
            return Err(StaticModelDiagnosticError::InvalidActionEligibility(
                action_id.to_owned(),
            ));
        };
        if action.status != StaticDiagnosticFeasibilityStatus::Feasible
            || !is_lower_hex(&action.feasibility_receipt_sha256, 64)
        {
            return Err(StaticModelDiagnosticError::InvalidActionEligibility(
                action_id.to_owned(),
            ));
        }
    }

    let stop_outcome_index = problem
        .second_stage_outcome_ids
        .iter()
        .position(|id| id == STOP_ACTION_ID)
        .ok_or(StaticModelDiagnosticError::InvalidStopContract)?;
    for history_index in 0..problem.first_stage_history_ids.len() {
        let actions = &problem.second_stage_action_ids_by_history[history_index];
        let costs = &problem.second_stage_action_costs_by_history[history_index];
        if actions.is_empty() || actions.len() > 4 || actions.len() != costs.len() {
            return Err(StaticModelDiagnosticError::InvalidActionMenu(
                problem.first_stage_history_ids[history_index].clone(),
            ));
        }
        validate_ids(actions, "second-stage action")?;
        if !actions.iter().any(|id| id == STOP_ACTION_ID) {
            return Err(StaticModelDiagnosticError::InvalidStopContract);
        }
        for (action_id, cost) in actions.iter().zip(costs) {
            if !cost.is_finite()
                || *cost < 0.0
                || (action_id == STOP_ACTION_ID && *cost != 0.0)
                || (action_id != STOP_ACTION_ID && *cost == 0.0)
            {
                return Err(StaticModelDiagnosticError::InvalidActionMenu(
                    problem.first_stage_history_ids[history_index].clone(),
                ));
            }
        }
    }
    for cost in &problem.one_shot_action_costs {
        if !cost.is_finite() || *cost < 0.0 {
            return Err(StaticModelDiagnosticError::InvalidCost);
        }
        if *cost > problem.maximum_worst_branch_cost + RISK_TOL {
            return Err(StaticModelDiagnosticError::InvalidActionMenu(
                "one-shot action exceeds the shared cost ceiling".to_owned(),
            ));
        }
    }

    let model_prior_sum = problem
        .model_vertices
        .iter()
        .map(|vertex| vertex.model_prior_probability)
        .sum::<f64>();
    if !model_prior_sum.is_finite() || (model_prior_sum - 1.0).abs() > PROB_TOL {
        return Err(StaticModelDiagnosticError::InvalidModelPrior(
            "model prior mass".to_owned(),
        ));
    }

    let mut covered_histories = std::collections::BTreeSet::new();
    for vertex in &problem.model_vertices {
        if !vertex.model_prior_probability.is_finite()
            || !(0.0..=1.0).contains(&vertex.model_prior_probability)
        {
            return Err(StaticModelDiagnosticError::InvalidModelPrior(
                vertex.model_vertex_id.clone(),
            ));
        }
        if vertex.world_prior_probabilities.len() != problem.world_ids.len() {
            return Err(StaticModelDiagnosticError::InvalidWorldPrior(
                vertex.model_vertex_id.clone(),
            ));
        }
        validate_distribution(
            &vertex.world_prior_probabilities,
            format!("{} world prior", vertex.model_vertex_id),
        )
        .map_err(|_| StaticModelDiagnosticError::InvalidWorldPrior(vertex.model_vertex_id.clone()))?;

        if vertex.stage_one_outcome_probabilities_by_world.len() != problem.world_ids.len()
            || vertex
                .stage_one_outcome_probabilities_by_world
                .iter()
                .any(|values| values.len() != problem.first_stage_history_ids.len())
        {
            return Err(StaticModelDiagnosticError::InvalidStageOneTensor(
                vertex.model_vertex_id.clone(),
            ));
        }
        for (world_index, distribution) in
            vertex.stage_one_outcome_probabilities_by_world.iter().enumerate()
        {
            validate_distribution(
                distribution,
                format!("{} stage-one world {world_index}", vertex.model_vertex_id),
            )?;
            for (history_index, probability) in distribution.iter().enumerate() {
                if *probability > 0.0 {
                    covered_histories.insert(history_index);
                }
            }
        }

        if vertex.stage_two_outcome_probabilities.len() != problem.first_stage_history_ids.len() {
            return Err(StaticModelDiagnosticError::InvalidStageTwoTensor(
                vertex.model_vertex_id.clone(),
            ));
        }
        for history_index in 0..problem.first_stage_history_ids.len() {
            let actions = &vertex.stage_two_outcome_probabilities[history_index];
            if actions.len() != problem.second_stage_action_ids_by_history[history_index].len() {
                return Err(StaticModelDiagnosticError::InvalidStageTwoTensor(
                    vertex.model_vertex_id.clone(),
                ));
            }
            let stop_action_index = problem.second_stage_action_ids_by_history[history_index]
                .iter()
                .position(|id| id == STOP_ACTION_ID)
                .expect("STOP was validated");
            for (action_index, worlds) in actions.iter().enumerate() {
                if worlds.len() != problem.world_ids.len() {
                    return Err(StaticModelDiagnosticError::InvalidStageTwoTensor(
                        vertex.model_vertex_id.clone(),
                    ));
                }
                for (world_index, distribution) in worlds.iter().enumerate() {
                    if distribution.len() != problem.second_stage_outcome_ids.len() {
                        return Err(StaticModelDiagnosticError::InvalidStageTwoTensor(
                            vertex.model_vertex_id.clone(),
                        ));
                    }
                    validate_distribution(
                        distribution,
                        format!(
                            "{} stage-two h={history_index} a={action_index} w={world_index}",
                            vertex.model_vertex_id
                        ),
                    )?;
                    if action_index == stop_action_index
                        && (distribution[stop_outcome_index] != 1.0
                            || distribution.iter().enumerate().any(|(index, value)| {
                                index != stop_outcome_index && *value != 0.0
                            }))
                    {
                        return Err(StaticModelDiagnosticError::InvalidStopContract);
                    }
                }
            }
        }

        if vertex.one_shot_outcome_probabilities_by_action.len() != problem.one_shot_action_ids.len()
            || vertex
                .one_shot_outcome_probabilities_by_action
                .iter()
                .any(|worlds| {
                    worlds.len() != problem.world_ids.len()
                        || worlds
                            .iter()
                            .any(|outcomes| outcomes.len() != problem.one_shot_outcome_ids.len())
                })
        {
            return Err(StaticModelDiagnosticError::InvalidOneShotTensor(
                vertex.model_vertex_id.clone(),
            ));
        }
        for (action_index, worlds) in
            vertex.one_shot_outcome_probabilities_by_action.iter().enumerate()
        {
            for (world_index, distribution) in worlds.iter().enumerate() {
                validate_distribution(
                    distribution,
                    format!(
                        "{} one-shot a={action_index} w={world_index}",
                        vertex.model_vertex_id
                    ),
                )?;
            }
        }
    }
    if covered_histories.len() != problem.first_stage_history_ids.len() {
        return Err(StaticModelDiagnosticError::InvalidStageOneTensor(
            "model vertices do not cover the public history alphabet".to_owned(),
        ));
    }
    Ok(())
}

fn validate_distribution(
    values: &[f64],
    location: String,
) -> Result<(), StaticModelDiagnosticError> {
    let mut sum = 0.0;
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(StaticModelDiagnosticError::InvalidProbability {
                location: location.clone(),
                index,
            });
        }
        sum += value;
    }
    if !sum.is_finite() || (sum - 1.0).abs() > PROB_TOL {
        return Err(StaticModelDiagnosticError::ProbabilityMassNotNormalized { location, sum });
    }
    Ok(())
}

fn validate_ids(ids: &[String], label: &'static str) -> Result<(), StaticModelDiagnosticError> {
    if ids.is_empty()
        || ids.iter().any(|id| id.trim().is_empty())
        || ids.iter().collect::<std::collections::BTreeSet<_>>().len() != ids.len()
    {
        return Err(StaticModelDiagnosticError::InvalidIdentitySet(label));
    }
    Ok(())
}

fn required_policy_evaluations(
    problem: &StaticModelDiagnosticProblem,
) -> Result<u64, StaticModelDiagnosticError> {
    let actions = action_combination_count(problem)?;
    let decisions = 1_u64 << (problem.first_stage_history_ids.len() * problem.second_stage_outcome_ids.len());
    let one_shot_masks = 1_u64 << problem.one_shot_outcome_ids.len();
    actions
        .checked_mul(decisions)
        .and_then(|count| {
            count.checked_add(problem.one_shot_action_ids.len() as u64 * one_shot_masks)
        })
        .ok_or(StaticModelDiagnosticError::SearchBudgetExceeded {
            required: u64::MAX,
            budget: 0,
        })
}

fn action_combination_count(
    problem: &StaticModelDiagnosticProblem,
) -> Result<u64, StaticModelDiagnosticError> {
    problem
        .second_stage_action_ids_by_history
        .iter()
        .try_fold(1_u64, |count, actions| {
            count
                .checked_mul(actions.len() as u64)
                .ok_or(StaticModelDiagnosticError::SearchBudgetExceeded {
                    required: u64::MAX,
                    budget: 0,
                })
        })
}

fn decode_action_choices(problem: &StaticModelDiagnosticProblem, mut code: u64) -> Vec<usize> {
    problem
        .second_stage_action_ids_by_history
        .iter()
        .map(|actions| {
            let choice = (code % actions.len() as u64) as usize;
            code /= actions.len() as u64;
            choice
        })
        .collect()
}

fn decode_decisions(
    problem: &StaticModelDiagnosticProblem,
    mask: u64,
) -> Vec<StaticDiagnosticDecision> {
    (0..problem.first_stage_history_ids.len() * problem.second_stage_outcome_ids.len())
        .map(|index| {
            if mask & (1_u64 << index) == 0 {
                StaticDiagnosticDecision::Support
            } else {
                StaticDiagnosticDecision::Refute
            }
        })
        .collect()
}

fn policy_worst_branch_cost(
    problem: &StaticModelDiagnosticProblem,
    action_indices: &[usize],
) -> f64 {
    problem.root_action_cost
        + action_indices
            .iter()
            .enumerate()
            .map(|(history_index, action_index)| {
                problem.second_stage_action_costs_by_history[history_index][*action_index]
            })
            .fold(0.0_f64, f64::max)
}

fn candidate_precedes(candidate: &CandidatePolicy, best: &CandidatePolicy, static_mode: bool) -> bool {
    let candidate_risk = if static_mode {
        candidate.static_robust_risk
    } else {
        candidate.rectangular_robust_risk
    };
    let best_risk = if static_mode {
        best.static_robust_risk
    } else {
        best.rectangular_robust_risk
    };
    if candidate_risk + RISK_TOL < best_risk {
        return true;
    }
    if best_risk + RISK_TOL < candidate_risk {
        return false;
    }
    if candidate.worst_branch_cost + RISK_TOL < best.worst_branch_cost {
        return true;
    }
    if best.worst_branch_cost + RISK_TOL < candidate.worst_branch_cost {
        return false;
    }
    candidate.action_indices < best.action_indices
        || (candidate.action_indices == best.action_indices && candidate.decisions < best.decisions)
}

fn is_wrong(decision: StaticDiagnosticDecision, world_is_true: bool) -> f64 {
    if (decision == StaticDiagnosticDecision::Support) == world_is_true {
        0.0
    } else {
        1.0
    }
}

fn static_policy_risk(
    problem: &StaticModelDiagnosticProblem,
    vertex: &StaticDiagnosticModelVertex,
    action_indices: &[usize],
    decisions: &[StaticDiagnosticDecision],
) -> f64 {
    let mut risk = 0.0;
    let outcome_count = problem.second_stage_outcome_ids.len();
    for world_index in 0..problem.world_ids.len() {
        for history_index in 0..problem.first_stage_history_ids.len() {
            let p_history =
                vertex.stage_one_outcome_probabilities_by_world[world_index][history_index];
            let action_index = action_indices[history_index];
            for outcome_index in 0..outcome_count {
                let p_outcome = vertex.stage_two_outcome_probabilities[history_index]
                    [action_index][world_index][outcome_index];
                let decision = decisions[history_index * outcome_count + outcome_index];
                risk += vertex.world_prior_probabilities[world_index]
                    * p_history
                    * p_outcome
                    * is_wrong(decision, problem.world_is_query_true[world_index]);
            }
        }
    }
    risk
}

/// Rectangular relaxation: the root vertex supplies the world prior and
/// stage-one kernel, but the adversary may choose a distinct stage-two vertex
/// independently for each observed first-stage history.
fn rectangular_policy_risk(
    problem: &StaticModelDiagnosticProblem,
    root_vertex: &StaticDiagnosticModelVertex,
    action_indices: &[usize],
    decisions: &[StaticDiagnosticDecision],
) -> f64 {
    let outcome_count = problem.second_stage_outcome_ids.len();
    let mut total = 0.0;
    for history_index in 0..problem.first_stage_history_ids.len() {
        let joint_world_mass = (0..problem.world_ids.len())
            .map(|world_index| {
                root_vertex.world_prior_probabilities[world_index]
                    * root_vertex.stage_one_outcome_probabilities_by_world[world_index][history_index]
            })
            .collect::<Vec<_>>();
        let p_history = joint_world_mass.iter().sum::<f64>();
        if p_history == 0.0 {
            continue;
        }
        let posterior_world = joint_world_mass
            .iter()
            .map(|mass| mass / p_history)
            .collect::<Vec<_>>();
        let action_index = action_indices[history_index];
        let worst_conditional = problem
            .model_vertices
            .iter()
            .map(|stage_two_vertex| {
                let mut conditional = 0.0;
                for world_index in 0..problem.world_ids.len() {
                    for outcome_index in 0..outcome_count {
                        let decision = decisions[history_index * outcome_count + outcome_index];
                        conditional += posterior_world[world_index]
                            * stage_two_vertex.stage_two_outcome_probabilities[history_index]
                                [action_index][world_index][outcome_index]
                            * is_wrong(decision, problem.world_is_query_true[world_index]);
                    }
                }
                conditional
            })
            .fold(0.0_f64, f64::max);
        total += p_history * worst_conditional;
    }
    total
}

/// Risk when the stage-one and stage-two kernels are generated by different
/// fixed model vertices. This explicitly violates the static-model assumption.
fn cross_stage_switching_risk(
    problem: &StaticModelDiagnosticProblem,
    action_indices: &[usize],
    decisions: &[StaticDiagnosticDecision],
) -> f64 {
    let outcome_count = problem.second_stage_outcome_ids.len();
    let mut worst = 0.0_f64;
    for root_vertex in &problem.model_vertices {
        for stage_two_vertex in &problem.model_vertices {
            if root_vertex.model_vertex_id == stage_two_vertex.model_vertex_id {
                continue;
            }
            let mut risk = 0.0;
            for world_index in 0..problem.world_ids.len() {
                for history_index in 0..problem.first_stage_history_ids.len() {
                    let p_history =
                        root_vertex.stage_one_outcome_probabilities_by_world[world_index][history_index];
                    let action_index = action_indices[history_index];
                    for outcome_index in 0..outcome_count {
                        let decision = decisions[history_index * outcome_count + outcome_index];
                        risk += root_vertex.world_prior_probabilities[world_index]
                            * p_history
                            * stage_two_vertex.stage_two_outcome_probabilities[history_index]
                                [action_index][world_index][outcome_index]
                            * is_wrong(decision, problem.world_is_query_true[world_index]);
                    }
                }
            }
            worst = worst.max(risk);
        }
    }
    worst
}

fn materialize_policy(
    problem: &StaticModelDiagnosticProblem,
    action_indices: &[usize],
    decisions: &[StaticDiagnosticDecision],
    worst_branch_cost: f64,
) -> StaticDiagnosticPolicyTree {
    let mut actions = Vec::with_capacity(action_indices.len());
    let mut terminal = Vec::with_capacity(decisions.len());
    for (history_index, history_id) in problem.first_stage_history_ids.iter().enumerate() {
        actions.push(StaticDiagnosticHistoryAction {
            first_stage_history_id: history_id.clone(),
            action_id: problem.second_stage_action_ids_by_history[history_index]
                [action_indices[history_index]]
                .clone(),
        });
        for (outcome_index, outcome_id) in problem.second_stage_outcome_ids.iter().enumerate() {
            terminal.push(StaticDiagnosticTerminalDecision {
                first_stage_history_id: history_id.clone(),
                second_stage_outcome_id: outcome_id.clone(),
                decision: decisions[history_index * problem.second_stage_outcome_ids.len()
                    + outcome_index],
            });
        }
    }
    StaticDiagnosticPolicyTree {
        root_action_id: problem.root_action_id.clone(),
        second_stage_action_by_history: actions,
        terminal_decision_by_history_and_outcome: terminal,
        worst_branch_cost,
    }
}

fn materialize_risks(
    problem: &StaticModelDiagnosticProblem,
    risks: &[f64],
) -> Vec<StaticDiagnosticModelRisk> {
    problem
        .model_vertices
        .iter()
        .zip(risks)
        .map(|(vertex, risk)| StaticDiagnosticModelRisk {
            model_vertex_id: vertex.model_vertex_id.clone(),
            risk: *risk,
        })
        .collect()
}

fn no_probe_risk(problem: &StaticModelDiagnosticProblem) -> f64 {
    [StaticDiagnosticDecision::Support, StaticDiagnosticDecision::Refute]
        .into_iter()
        .map(|decision| {
            problem
                .model_vertices
                .iter()
                .map(|vertex| {
                    vertex
                        .world_prior_probabilities
                        .iter()
                        .zip(&problem.world_is_query_true)
                        .map(|(probability, is_true)| {
                            probability * is_wrong(decision, *is_true)
                        })
                        .sum::<f64>()
                })
                .fold(0.0_f64, f64::max)
        })
        .fold(f64::INFINITY, f64::min)
}

fn evaluate_one_shot(
    problem: &StaticModelDiagnosticProblem,
) -> Result<StaticDiagnosticOneShotResult, StaticModelDiagnosticError> {
    let masks = 1_u64 << problem.one_shot_outcome_ids.len();
    let mut assessments = Vec::with_capacity(problem.one_shot_action_ids.len());
    for action_index in 0..problem.one_shot_action_ids.len() {
        let mut best: Option<OneShotCandidate> = None;
        for mask in 0..masks {
            let decisions = (0..problem.one_shot_outcome_ids.len())
                .map(|index| {
                    if mask & (1_u64 << index) == 0 {
                        StaticDiagnosticDecision::Support
                    } else {
                        StaticDiagnosticDecision::Refute
                    }
                })
                .collect::<Vec<_>>();
            let per_model_risks = problem
                .model_vertices
                .iter()
                .map(|vertex| {
                    let mut risk = 0.0;
                    for world_index in 0..problem.world_ids.len() {
                        for outcome_index in 0..problem.one_shot_outcome_ids.len() {
                            risk += vertex.world_prior_probabilities[world_index]
                                * vertex.one_shot_outcome_probabilities_by_action[action_index]
                                    [world_index][outcome_index]
                                * is_wrong(decisions[outcome_index], problem.world_is_query_true[world_index]);
                        }
                    }
                    risk
                })
                .collect::<Vec<_>>();
            let robust_risk = per_model_risks.iter().copied().fold(0.0_f64, f64::max);
            let candidate = OneShotCandidate {
                decisions,
                per_model_risks,
                robust_risk,
            };
            if best.as_ref().is_none_or(|current| {
                candidate.robust_risk + RISK_TOL < current.robust_risk
                    || ((candidate.robust_risk - current.robust_risk).abs() <= RISK_TOL
                        && candidate.decisions < current.decisions)
            }) {
                best = Some(candidate);
            }
        }
        let best = best.expect("one-shot outcome set is non-empty");
        assessments.push(StaticDiagnosticOneShotAssessment {
            action_id: problem.one_shot_action_ids[action_index].clone(),
            action_cost: problem.one_shot_action_costs[action_index],
            worst_case_risk: best.robust_risk,
            decision_by_outcome: problem
                .one_shot_outcome_ids
                .iter()
                .zip(&best.decisions)
                .map(|(outcome, decision)| (outcome.clone(), *decision))
                .collect(),
            per_model_risks: materialize_risks(problem, &best.per_model_risks),
        });
    }
    let best = assessments
        .iter()
        .min_by(|left, right| {
            left.worst_case_risk
                .total_cmp(&right.worst_case_risk)
                .then_with(|| left.action_cost.total_cmp(&right.action_cost))
                .then_with(|| left.action_id.cmp(&right.action_id))
        })
        .ok_or(StaticModelDiagnosticError::NoFeasiblePolicy)?;
    Ok(StaticDiagnosticOneShotResult {
        best_action_id: best.action_id.clone(),
        best_worst_case_risk: best.worst_case_risk,
        action_assessments: assessments,
    })
}

fn build_receipt(
    problem: &StaticModelDiagnosticProblem,
    config: StaticModelDiagnosticConfig,
    source_git_sha: &str,
    report: &StaticModelDiagnosticReport,
) -> Result<StaticModelDiagnosticReceipt, StaticModelDiagnosticError> {
    let input_bytes = serde_json::to_vec(&(problem, config))
        .map_err(|_| StaticModelDiagnosticError::SerializationFailure)?;
    let task_bytes = serde_json::to_vec(&(
        &problem.world_ids,
        &problem.world_is_query_true,
        &problem.root_action_id,
        problem.root_action_cost,
        problem.maximum_worst_branch_cost,
        problem.certificate_maximum_robust_risk,
    ))
    .map_err(|_| StaticModelDiagnosticError::SerializationFailure)?;
    let kernel_bytes = serde_json::to_vec(&problem.model_vertices)
        .map_err(|_| StaticModelDiagnosticError::SerializationFailure)?;
    let static_policy_bytes = serde_json::to_vec(&report.static_policy)
        .map_err(|_| StaticModelDiagnosticError::SerializationFailure)?;
    let rectangular_policy_bytes = serde_json::to_vec(&report.rectangular_policy)
        .map_err(|_| StaticModelDiagnosticError::SerializationFailure)?;
    let result_bytes =
        serde_json::to_vec(report).map_err(|_| StaticModelDiagnosticError::SerializationFailure)?;

    Ok(StaticModelDiagnosticReceipt {
        schema_version: 1,
        source_git_sha: source_git_sha.to_owned(),
        input_sha256: sha256_hex(&input_bytes),
        task_specification_sha256: sha256_hex(&task_bytes),
        model_kernel_sha256: sha256_hex(&kernel_bytes),
        static_policy_sha256: sha256_hex(&static_policy_bytes),
        rectangular_policy_sha256: sha256_hex(&rectangular_policy_bytes),
        result_sha256: sha256_hex(&result_bytes),
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_separates_static_and_rectangular_risk() {
        let problem = symmetric_calibration_fixture();
        let result = evaluate_static_model_diagnostic_problem(
            &problem,
            StaticModelDiagnosticConfig::default(),
            &"a".repeat(40),
        )
        .expect("known-answer policy oracle");
        assert_eq!(
            result.report.claim_scope,
            StaticDiagnosticClaimScope::FiniteDeclaredModelSetOnly
        );
        assert_eq!(result.report.static_robust_risk, 0.0);
        assert_eq!(result.report.clairvoyant_lower_bound_risk, 0.0);
        assert!((result.report.rectangular_minimax_risk - 0.5).abs() < 1e-12);
        assert!((result.report.rectangular_risk_of_static_policy - 1.0).abs() < 1e-12);
        assert_eq!(result.report.no_probe_robust_risk, 0.5);
        assert!((result.report.one_shot.best_worst_case_risk - 0.5).abs() < 1e-12);
        assert_eq!(result.report.static_policy.worst_branch_cost, 2.0);
        assert_eq!(result.report.policy_evaluations, 264);
        assert_eq!(result.report.static_policy.root_action_id, "CALIBRATE");
        assert!(result.report.static_policy.second_stage_action_by_history.iter().any(|x| {
            x.first_stage_history_id == "CAL-A" && x.action_id == "ACTION-A"
        }));
        assert!(result.report.static_policy.second_stage_action_by_history.iter().any(|x| {
            x.first_stage_history_id == "CAL-B" && x.action_id == "ACTION-B"
        }));
        assert!(result.report.static_policy.terminal_decision_by_history_and_outcome.iter().any(|x| {
            x.first_stage_history_id == "CAL-A"
                && x.second_stage_outcome_id == "POSITIVE"
                && x.decision == StaticDiagnosticDecision::Support
        }));
        assert!(result.report.static_policy.terminal_decision_by_history_and_outcome.iter().any(|x| {
            x.first_stage_history_id == "CAL-B"
                && x.second_stage_outcome_id == "NEGATIVE"
                && x.decision == StaticDiagnosticDecision::Refute
        }));
        assert_eq!(result.report.cross_stage_switching_stress_risk, 1.0);
        assert!(is_lower_hex(&result.receipt.input_sha256, 64));
    }

    #[test]
    fn uninformative_calibration_removes_static_rectangularity_gap() {
        let result = evaluate_static_model_diagnostic_problem(
            &uninformative_calibration_fixture(),
            StaticModelDiagnosticConfig::default(),
            &"b".repeat(40),
        )
        .expect("uninformative calibration fixture");
        assert!(result.report.static_robust_risk >= 0.5 - 1e-12);
        assert!(result.report.static_rectangular_conservatism_gap.abs() < 1e-12);
    }

    #[test]
    fn identical_model_vertices_have_zero_rectangularity_gap() {
        let mut problem = symmetric_calibration_fixture();
        let source = problem.model_vertices[0].clone();
        let shared_stage_one = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        problem.model_vertices[0].stage_one_outcome_probabilities_by_world =
            shared_stage_one.clone();
        problem.model_vertices[1].stage_one_outcome_probabilities_by_world = shared_stage_one;
        problem.model_vertices[1].stage_two_outcome_probabilities =
            source.stage_two_outcome_probabilities;
        problem.model_vertices[1].one_shot_outcome_probabilities_by_action =
            source.one_shot_outcome_probabilities_by_action;
        let result = evaluate_static_model_diagnostic_problem(
            &problem,
            StaticModelDiagnosticConfig::default(),
            &"c".repeat(40),
        )
        .expect("identical model vertices");
        assert!(result.report.static_rectangular_conservatism_gap.abs() < 1e-12);
    }

    #[test]
    fn posterior_updates_from_public_observations_only() {
        let problem = symmetric_calibration_fixture();
        let calibrated = posterior_after_public_history(&problem, "CAL-A", None)
            .expect("calibration outcome");
        assert_eq!(
            calibrated.world_probabilities,
            vec![
                ("QUERY_TRUE".to_owned(), 0.5),
                ("QUERY_FALSE".to_owned(), 0.5)
            ]
        );
        assert_eq!(
            calibrated.model_probabilities,
            vec![
                ("MODEL-A".to_owned(), 1.0),
                ("MODEL-B".to_owned(), 0.0)
            ]
        );
        let classified = posterior_after_public_history(
            &problem,
            "CAL-A",
            Some(("ACTION-A", "POSITIVE")),
        )
        .expect("terminal observation");
        assert_eq!(
            classified.world_probabilities,
            vec![
                ("QUERY_TRUE".to_owned(), 1.0),
                ("QUERY_FALSE".to_owned(), 0.0)
            ]
        );
    }

    #[test]
    fn infeasible_or_unreceipted_actions_never_enter_policy_search() {
        let mut problem = symmetric_calibration_fixture();
        problem.action_eligibility[2].status = StaticDiagnosticFeasibilityStatus::Disallowed;
        assert!(matches!(
            evaluate_static_model_diagnostic_problem(
                &problem,
                StaticModelDiagnosticConfig::default(),
                &"d".repeat(40),
            ),
            Err(StaticModelDiagnosticError::InvalidActionEligibility(id)) if id == "ACTION-B"
        ));

        let mut missing_receipt = symmetric_calibration_fixture();
        missing_receipt.action_eligibility[1].feasibility_receipt_sha256 = "bad".to_owned();
        assert!(matches!(
            evaluate_static_model_diagnostic_problem(
                &missing_receipt,
                StaticModelDiagnosticConfig::default(),
                &"e".repeat(40),
            ),
            Err(StaticModelDiagnosticError::InvalidActionEligibility(id)) if id == "ACTION-A"
        ));
    }

    #[test]
    fn search_budget_exhaustion_fails_closed() {
        assert!(matches!(
            evaluate_static_model_diagnostic_problem(
                &symmetric_calibration_fixture(),
                StaticModelDiagnosticConfig {
                    maximum_policy_evaluations: 100,
                },
                &"d".repeat(40),
            ),
            Err(StaticModelDiagnosticError::SearchBudgetExceeded { .. })
        ));
    }

    #[test]
    fn impossible_observations_and_invalid_source_sha_fail_closed() {
        let problem = symmetric_calibration_fixture();
        assert!(matches!(
            posterior_after_public_history(&problem, "CAL-A", Some(("ACTION-A", "UNKNOWN"))),
            Err(StaticModelDiagnosticError::ImpossiblePublicHistory)
        ));
        assert!(matches!(
            evaluate_static_model_diagnostic_problem(
                &problem,
                StaticModelDiagnosticConfig::default(),
                "not-a-commit",
            ),
            Err(StaticModelDiagnosticError::InvalidSourceGitSha)
        ));
    }
}
