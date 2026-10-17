// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Outcome-generating active causal-graph benchmark using small binary SCMs.
//!
//! This tranche deliberately uses all DAGs on three observed variables so the
//! graph-hypothesis catalogue is enumerable and exact. Hidden worlds vary graph
//! families and structural-equation parameters. It is a benchmark substrate,
//! not a claim that three-variable simulations establish general causal discovery.

use serde::{Deserialize, Serialize};

use crate::experiment_design::{
    ExperimentCandidate, ExperimentDesignError, HypothesisPrior, ModelLikelihoodScenario,
    rank_experiments, rank_experiments_robust, update_posterior,
};

const NODE_COUNT: usize = 3;
const OUTCOME_COUNT: usize = 1 << NODE_COUNT;
const EDGE_PAIRS: [(usize, usize); 6] = [
    (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1),
];
const WORLD_VARIANTS_PER_FAMILY: usize = 4;
const MAX_TRIALS_PER_WORLD: usize = 10_000;
const MAX_MEASUREMENT_BUDGET: usize = 64;
const GENERATOR_VERSION: &str = "sensebench-scm-graphs-0.1.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScmGraphFamily {
    ChainDevelopment,
    ForkDevelopment,
    ColliderHeldOut,
    SparseRandomHeldOut,
    DenseRandomHeldOut,
    InhibitoryMechanismHeldOut,
    NullGraphNegativeControl,
}

impl ScmGraphFamily {
    fn all() -> [Self; 7] {
        [
            Self::ChainDevelopment,
            Self::ForkDevelopment,
            Self::ColliderHeldOut,
            Self::SparseRandomHeldOut,
            Self::DenseRandomHeldOut,
            Self::InhibitoryMechanismHeldOut,
            Self::NullGraphNegativeControl,
        ]
    }

    fn held_out(self) -> bool {
        !matches!(self, Self::ChainDevelopment | Self::ForkDevelopment)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ChainDevelopment => "chain_development",
            Self::ForkDevelopment => "fork_development",
            Self::ColliderHeldOut => "collider_held_out",
            Self::SparseRandomHeldOut => "sparse_random_held_out",
            Self::DenseRandomHeldOut => "dense_random_held_out",
            Self::InhibitoryMechanismHeldOut => "inhibitory_mechanism_held_out",
            Self::NullGraphNegativeControl => "null_graph_negative_control",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScmGraphPolicy {
    NominalExpectedInformationGain,
    PredictiveOutcomeEntropy,
    ModelWeightedExpectedInformationGain,
    WorstCaseRobust,
    DevelopmentFixedDesign,
    UniformRandomAction,
    PassiveOnly,
    KnownMechanicsOracle,
    StructureBlindRandomGraph,
}

impl ScmGraphPolicy {
    fn all() -> [Self; 9] {
        [
            Self::NominalExpectedInformationGain,
            Self::PredictiveOutcomeEntropy,
            Self::ModelWeightedExpectedInformationGain,
            Self::WorstCaseRobust,
            Self::DevelopmentFixedDesign,
            Self::UniformRandomAction,
            Self::PassiveOnly,
            Self::KnownMechanicsOracle,
            Self::StructureBlindRandomGraph,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScmGraphPolicySummary {
    pub world_id: String,
    /// Evaluator-side truth is emitted only in the final report, not supplied to policies.
    pub truth_graph_id: String,
    pub truth_edge_count: usize,
    pub family: ScmGraphFamily,
    pub held_out_family: bool,
    pub policy: ScmGraphPolicy,
    pub trials: usize,
    pub measurement_budget: usize,
    pub mean_skeleton_f1: f64,
    pub skeleton_f1_ci95: [f64; 2],
    pub mean_directed_edge_f1: f64,
    pub directed_edge_f1_ci95: [f64; 2],
    /// Skeleton of the predicted Markov equivalence class; unresolved directions are allowed.
    pub mean_cpdag_skeleton_f1: f64,
    pub cpdag_skeleton_f1_ci95: [f64; 2],
    /// F1 for arrows compelled by the true Markov equivalence class.
    pub mean_compelled_direction_f1: f64,
    pub compelled_direction_f1_ci95: [f64; 2],
    /// Number of pairwise absent/undirected/forward/reverse mark disagreements.
    pub mean_cpdag_mark_distance: f64,
    pub cpdag_mark_distance_ci95: [f64; 2],
    pub markov_equivalence_class_recovery_rate: f64,
    pub markov_equivalence_class_recovery_wilson_ci95: [f64; 2],
    pub mean_posterior_mass_on_true_markov_class: f64,
    pub posterior_mass_on_true_markov_class_ci95: [f64; 2],
    /// One unit for a missing/extra adjacency or a reversed edge.
    pub mean_structural_hamming_distance: f64,
    pub structural_hamming_distance_ci95: [f64; 2],
    pub exact_graph_recovery_rate: f64,
    pub exact_graph_recovery_wilson_ci95: [f64; 2],
    pub mean_posterior_mass_on_true_graph: f64,
    pub posterior_mass_on_true_graph_ci95: [f64; 2],
    pub mean_graph_posterior_brier_score: f64,
    pub graph_posterior_brier_ci95: [f64; 2],
    /// Fraction of trials with a posterior-max probability at or above threshold.
    pub graph_claim_coverage: f64,
    pub graph_claim_coverage_wilson_ci95: [f64; 2],
    /// Wrong MAP graph among accepted graph claims; None if no claim was accepted.
    pub false_graph_claim_rate_among_accepted: Option<f64>,
    pub false_graph_claim_rate_among_accepted_wilson_ci95: Option<[f64; 2]>,
    pub false_graph_claim_rate_all_trials: f64,
    pub mean_measurements_used: f64,
    pub mean_interventions_used: f64,
    pub mean_measurements_to_threshold: Option<f64>,
    pub mean_interventions_to_threshold: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScmGraphPolicyAggregate {
    pub policy: ScmGraphPolicy,
    pub held_out_worlds: usize,
    /// Each world has equal weight, regardless of trials run in that world.
    pub mean_held_out_skeleton_f1: f64,
    pub held_out_skeleton_f1_world_ci95: [f64; 2],
    pub mean_held_out_directed_edge_f1: f64,
    pub held_out_directed_edge_f1_world_ci95: [f64; 2],
    pub mean_held_out_cpdag_skeleton_f1: f64,
    pub held_out_cpdag_skeleton_f1_world_ci95: [f64; 2],
    pub mean_held_out_compelled_direction_f1: f64,
    pub held_out_compelled_direction_f1_world_ci95: [f64; 2],
    pub mean_held_out_cpdag_mark_distance: f64,
    pub held_out_cpdag_mark_distance_world_ci95: [f64; 2],
    pub held_out_markov_equivalence_class_recovery_rate: f64,
    pub mean_held_out_shd: f64,
    pub held_out_shd_world_ci95: [f64; 2],
    pub worst_world_directed_edge_f1: f64,
    pub worst_world_shd: f64,
    pub mean_held_out_exact_graph_recovery_rate: f64,
    pub mean_held_out_claim_coverage: f64,
    pub mean_held_out_false_graph_claim_rate_all_trials: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScmGraphBenchmarkReport {
    pub schema_version: u16,
    pub generator_version: String,
    pub seed: u64,
    pub trials_per_world: usize,
    pub measurement_budget: usize,
    pub posterior_threshold: f64,
    pub observed_node_count: usize,
    pub candidate_dag_count: usize,
    pub world_variants_per_family: usize,
    pub world_count: usize,
    pub held_out_world_count: usize,
    pub declared_model_scenarios: Vec<String>,
    pub fixed_design_action_id: String,
    pub worlds: Vec<ScmGraphPolicySummary>,
    pub held_out_policy_aggregates: Vec<ScmGraphPolicyAggregate>,
    pub interpretation_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScmGraphBenchmarkError {
    #[error("trials per world must be between 3 and MAX_TRIALS_PER_WORLD")]
    InvalidTrialCount,
    #[error("measurement budget must be between 1 and MAX_MEASUREMENT_BUDGET")]
    InvalidMeasurementBudget,
    #[error("posterior threshold must be finite and in (1 / candidate DAG count, 1]")]
    InvalidPosteriorThreshold,
    #[error("a selected action or candidate graph is missing")]
    MissingCandidate,
    #[error("graph generator did not produce the expected DAG catalogue")]
    InvalidDagCatalogue,
    #[error(transparent)]
    InvalidExperimentDesign(#[from] ExperimentDesignError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CausalDag {
    /// Six possible directed edges, in EDGE_PAIRS order. Cyclic masks are rejected.
    edge_mask: u8,
}

impl CausalDag {
    fn has_edge(self, source: usize, target: usize) -> bool {
        EDGE_PAIRS.iter()
            .position(|pair| *pair == (source, target))
            .is_some_and(|index| (self.edge_mask & (1 << index)) != 0)
    }

    fn edge_count(self) -> usize {
        self.edge_mask.count_ones() as usize
    }

    fn id(self) -> String {
        format!("dag_{:02x}", self.edge_mask)
    }
}

fn adjacent(graph: CausalDag, first: usize, second: usize) -> bool {
    graph.has_edge(first, second) || graph.has_edge(second, first)
}

fn same_skeleton(left: CausalDag, right: CausalDag) -> bool {
    (0..NODE_COUNT).all(|first| {
        ((first + 1)..NODE_COUNT).all(|second| {
            adjacent(left, first, second) == adjacent(right, first, second)
        })
    })
}

fn has_unshielded_collider(graph: CausalDag, left: usize, center: usize, right: usize) -> bool {
    left != center
        && center != right
        && left != right
        && graph.has_edge(left, center)
        && graph.has_edge(right, center)
        && !adjacent(graph, left, right)
}

fn same_unshielded_colliders(left: CausalDag, right: CausalDag) -> bool {
    (0..NODE_COUNT).all(|left_node| {
        (0..NODE_COUNT).all(|center| {
            (0..NODE_COUNT).all(|right_node| {
                has_unshielded_collider(left, left_node, center, right_node)
                    == has_unshielded_collider(right, left_node, center, right_node)
            })
        })
    })
}

/// Two DAGs are Markov equivalent exactly when they have the same skeleton and
/// the same unshielded colliders (for the Markov/faithfulness setting modelled here).
fn markov_equivalent(left: CausalDag, right: CausalDag) -> bool {
    same_skeleton(left, right) && same_unshielded_colliders(left, right)
}

fn cpdag_marks(graph: CausalDag, catalogue: &[CausalDag]) -> Vec<EdgeMark> {
    let equivalent: Vec<CausalDag> = catalogue.iter().copied()
        .filter(|candidate| markov_equivalent(graph, *candidate))
        .collect();
    UNORDERED_PAIRS.iter().map(|(first, second)| {
        if !adjacent(graph, *first, *second) {
            return EdgeMark::Absent;
        }
        let always_forward = equivalent.iter()
            .all(|candidate| candidate.has_edge(*first, *second));
        let always_reverse = equivalent.iter()
            .all(|candidate| candidate.has_edge(*second, *first));
        if always_forward {
            EdgeMark::Forward
        } else if always_reverse {
            EdgeMark::Reverse
        } else {
            EdgeMark::Undirected
        }
    }).collect()
}

/// Choose the Markov equivalence class with the greatest summed posterior mass.
/// This keeps the emitted partial graph aligned to a valid CPDAG class rather than
/// independently thresholding pairwise arrow probabilities into a possibly cyclic
/// or otherwise invalid partially directed graph.
fn posterior_cpdag(
    posterior: &[HypothesisPrior],
    catalogue: &[CausalDag],
) -> (Vec<EdgeMark>, f64) {
    let mut best_graph_index = 0usize;
    let mut best_mass = -1.0_f64;
    for (candidate_index, candidate) in catalogue.iter().enumerate() {
        let class_mass = catalogue.iter().enumerate()
            .filter(|(_, other)| markov_equivalent(*candidate, **other))
            .map(|(index, _)| posterior[index].probability)
            .sum::<f64>();
        if class_mass > best_mass {
            best_mass = class_mass;
            best_graph_index = candidate_index;
        }
    }
    (
        cpdag_marks(catalogue[best_graph_index], catalogue),
        best_mass.max(0.0),
    )
}

fn posterior_mass_on_class(
    posterior: &[HypothesisPrior],
    catalogue: &[CausalDag],
    target: CausalDag,
) -> f64 {
    catalogue.iter().enumerate()
        .filter(|(_, candidate)| markov_equivalent(target, **candidate))
        .map(|(index, _)| posterior[index].probability)
        .sum()
}

fn cpdag_metrics(predicted: &[EdgeMark], truth: &[EdgeMark]) -> (f64, f64, f64, bool) {
    let mut skeleton_tp = 0usize;
    let mut skeleton_fp = 0usize;
    let mut skeleton_fn = 0usize;
    let mut compelled_tp = 0usize;
    let mut compelled_fp = 0usize;
    let mut compelled_fn = 0usize;
    let mut mark_distance = 0.0;

    for (prediction, target) in predicted.iter().zip(truth) {
        let predicted_adjacent = *prediction != EdgeMark::Absent;
        let true_adjacent = *target != EdgeMark::Absent;
        match (predicted_adjacent, true_adjacent) {
            (true, true) => skeleton_tp += 1,
            (true, false) => skeleton_fp += 1,
            (false, true) => skeleton_fn += 1,
            (false, false) => {}
        }

        let predicted_directed = matches!(*prediction, EdgeMark::Forward | EdgeMark::Reverse);
        let true_directed = matches!(*target, EdgeMark::Forward | EdgeMark::Reverse);
        if predicted_directed && prediction == target && true_directed {
            compelled_tp += 1;
        } else if predicted_directed {
            // A direction on a truly undirected edge is an overclaim.
            compelled_fp += 1;
        }
        if true_directed && prediction != target {
            compelled_fn += 1;
        }
        if prediction != target {
            mark_distance += 1.0;
        }
    }

    (
        f1_from_counts(skeleton_tp, skeleton_fp, skeleton_fn),
        f1_from_counts(compelled_tp, compelled_fp, compelled_fn),
        mark_distance,
        predicted == truth,
    )
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeMark {
    Absent,
    Undirected,
    Forward,
    Reverse,
}

const UNORDERED_PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

#[derive(Debug, Clone, Copy, PartialEq)]
struct ScmParameters {
    intercept: f64,
    /// Weight per possible directed edge, in EDGE_PAIRS order.
    edge_weights: [f64; 6],
}

#[derive(Debug, Clone)]
struct ScmWorld {
    id: String,
    family: ScmGraphFamily,
    truth: CausalDag,
    parameters: ScmParameters,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionKind {
    Observe,
    Intervene { node: usize, value: usize },
}

#[derive(Debug, Clone)]
struct ScmAction {
    id: String,
    kind: ActionKind,
    cost: f64,
}

#[derive(Debug, Clone)]
struct TrialResult {
    skeleton_f1: f64,
    directed_edge_f1: f64,
    shd: f64,
    exact_graph: bool,
    cpdag_skeleton_f1: f64,
    compelled_direction_f1: f64,
    cpdag_mark_distance: f64,
    markov_equivalence_class_correct: bool,
    posterior_true_graph: f64,
    posterior_true_markov_class: f64,
    graph_posterior_brier: f64,
    claim_accepted: bool,
    false_graph_claim: bool,
    measurements_used: usize,
    interventions_used: usize,
    measurements_to_threshold: Option<usize>,
    interventions_to_threshold: Option<usize>,
}

/// Enumerate all labeled DAGs on three observed variables. The exact count is 25.
/// The small catalogue permits exact outcome likelihoods and known-answer graph metrics.
fn all_dags() -> Vec<CausalDag> {
    (0_u8..(1 << EDGE_PAIRS.len()))
        .filter(|mask| is_acyclic(*mask))
        .map(|edge_mask| CausalDag { edge_mask })
        .collect()
}

fn is_acyclic(edge_mask: u8) -> bool {
    let mut reachability = [[false; NODE_COUNT]; NODE_COUNT];
    for (index, (source, target)) in EDGE_PAIRS.iter().copied().enumerate() {
        if edge_mask & (1 << index) != 0 {
            reachability[source][target] = true;
        }
    }
    for pivot in 0..NODE_COUNT {
        for source in 0..NODE_COUNT {
            for target in 0..NODE_COUNT {
                let reachable_via_pivot =
                    reachability[source][pivot] && reachability[pivot][target];
                if reachable_via_pivot {
                    reachability[source][target] = true;
                }
            }
        }
    }
    (0..NODE_COUNT).all(|node| !reachability[node][node])
}

fn edge_mask_with(edges: &[(usize, usize)]) -> u8 {
    let mut mask = 0_u8;
    for &(source, target) in edges {
        if let Some(index) = EDGE_PAIRS.iter().position(|pair| *pair == (source, target)) {
            mask |= 1 << index;
        }
    }
    mask
}

fn shuffled_nodes(rng: &mut BenchmarkRng) -> [usize; NODE_COUNT] {
    let mut nodes = [0, 1, 2];
    for index in (1..NODE_COUNT).rev() {
        let other = rng.index(index + 1);
        nodes.swap(index, other);
    }
    nodes
}

fn random_dag_mask(rng: &mut BenchmarkRng, edge_probability: f64) -> u8 {
    let order = shuffled_nodes(rng);
    let mut edges = Vec::new();
    for earlier in 0..NODE_COUNT {
        for later in (earlier + 1)..NODE_COUNT {
            if rng.unit_f64() < edge_probability {
                edges.push((order[earlier], order[later]));
            }
        }
    }
    edge_mask_with(&edges)
}

fn graph_for_family(family: ScmGraphFamily, rng: &mut BenchmarkRng) -> CausalDag {
    let mask = match family {
        ScmGraphFamily::ChainDevelopment => {
            let order = shuffled_nodes(rng);
            edge_mask_with(&[(order[0], order[1]), (order[1], order[2])])
        }
        ScmGraphFamily::ForkDevelopment => {
            let center = rng.index(NODE_COUNT);
            let children: Vec<usize> = (0..NODE_COUNT).filter(|node| *node != center).collect();
            edge_mask_with(&[(center, children[0]), (center, children[1])])
        }
        ScmGraphFamily::ColliderHeldOut => {
            let center = rng.index(NODE_COUNT);
            let parents: Vec<usize> = (0..NODE_COUNT).filter(|node| *node != center).collect();
            edge_mask_with(&[(parents[0], center), (parents[1], center)])
        }
        ScmGraphFamily::SparseRandomHeldOut => random_dag_mask(rng, 0.28),
        ScmGraphFamily::DenseRandomHeldOut => random_dag_mask(rng, 0.82),
        ScmGraphFamily::InhibitoryMechanismHeldOut => random_dag_mask(rng, 0.52),
        ScmGraphFamily::NullGraphNegativeControl => 0,
    };
    CausalDag { edge_mask: mask }
}

fn parameters_for_family(family: ScmGraphFamily, rng: &mut BenchmarkRng) -> ScmParameters {
    let (intercept, magnitude, jitter, signed) = match family {
        ScmGraphFamily::ChainDevelopment => (0.12, 0.60, 0.035, false),
        ScmGraphFamily::ForkDevelopment => (0.16, 0.54, 0.04, false),
        ScmGraphFamily::ColliderHeldOut => (0.24, 0.47, 0.06, false),
        ScmGraphFamily::SparseRandomHeldOut => (0.20, 0.42, 0.12, false),
        ScmGraphFamily::DenseRandomHeldOut => (0.12, 0.28, 0.025, false),
        ScmGraphFamily::InhibitoryMechanismHeldOut => (0.22, 0.43, 0.08, true),
        ScmGraphFamily::NullGraphNegativeControl => (0.24, 0.55, 0.10, true),
    };
    let mut edge_weights = [0.0; 6];
    for weight in &mut edge_weights {
        let amplitude = (magnitude + (rng.unit_f64() - 0.5) * 2.0 * jitter).max(0.02);
        let sign = if signed && rng.unit_f64() < 0.5 { -1.0 } else { 1.0 };
        *weight = sign * amplitude;
    }
    ScmParameters { intercept, edge_weights }
}

fn generate_worlds(seed: u64) -> Vec<ScmWorld> {
    let families = ScmGraphFamily::all();
    let mut worlds = Vec::with_capacity(families.len() * WORLD_VARIANTS_PER_FAMILY);
    for (family_index, family) in families.iter().copied().enumerate() {
        for variant in 0..WORLD_VARIANTS_PER_FAMILY {
            let world_seed = seed
                ^ (family_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
                ^ (variant as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
            let mut rng = BenchmarkRng::new(world_seed);
            let truth = graph_for_family(family, &mut rng);
            let parameters = parameters_for_family(family, &mut rng);
            worlds.push(ScmWorld {
                id: format!("{}-variant-{variant:02}", family.as_str()),
                family,
                truth,
                parameters,
            });
        }
    }
    worlds
}

fn actions() -> Vec<ScmAction> {
    let mut result = vec![ScmAction {
        id: "observe".into(),
        kind: ActionKind::Observe,
        cost: 1.0,
    }];
    for node in 0..NODE_COUNT {
        for value in 0..=1 {
            result.push(ScmAction {
                id: format!("do_{node}_{value}"),
                kind: ActionKind::Intervene { node, value },
                cost: 1.0,
            });
        }
    }
    result
}

fn params_nominal() -> ScmParameters {
    ScmParameters { intercept: 0.15, edge_weights: [0.62; 6] }
}

fn params_alternative() -> ScmParameters {
    ScmParameters {
        intercept: 0.22,
        edge_weights: [0.42, -0.34, 0.38, 0.31, -0.27, 0.46],
    }
}

fn outcome_distribution(
    graph: CausalDag,
    action: &ScmAction,
    parameters: ScmParameters,
) -> Vec<f64> {
    let mut distribution = vec![0.0; OUTCOME_COUNT];
    for state in 0..OUTCOME_COUNT {
        let mut probability = 1.0;
        for node in 0..NODE_COUNT {
            if let ActionKind::Intervene { node: intervention_node, value } = action.kind {
                if node == intervention_node {
                    if ((state >> node) & 1) != value {
                        probability = 0.0;
                        break;
                    }
                    continue;
                }
            }

            let mut p_one = parameters.intercept;
            for (edge_index, (source, target)) in EDGE_PAIRS.iter().copied().enumerate() {
                if target == node
                    && graph.has_edge(source, target)
                    && ((state >> source) & 1) == 1
                {
                    p_one += parameters.edge_weights[edge_index];
                }
            }
            p_one = p_one.clamp(0.01, 0.99);
            probability *= if ((state >> node) & 1) == 1 {
                p_one
            } else {
                1.0 - p_one
            };
            if probability == 0.0 {
                break;
            }
        }
        distribution[state] = probability;
    }
    distribution
}

fn candidate_set(
    graphs: &[CausalDag],
    actions: &[ScmAction],
    parameters: ScmParameters,
) -> Vec<ExperimentCandidate> {
    actions.iter().map(|action| ExperimentCandidate {
        id: action.id.clone(),
        cost: action.cost,
        likelihoods: graphs.iter()
            .map(|graph| outcome_distribution(*graph, action, parameters))
            .collect(),
    }).collect()
}

fn declared_model_scenarios(
    graphs: &[CausalDag],
    actions: &[ScmAction],
) -> Vec<ModelLikelihoodScenario> {
    vec![
        ModelLikelihoodScenario {
            scenario_id: "nominal_positive_effects".into(),
            weight: 0.5,
            candidates: candidate_set(graphs, actions, params_nominal()),
        },
        ModelLikelihoodScenario {
            scenario_id: "alternative_mixed_effects".into(),
            weight: 0.5,
            candidates: candidate_set(graphs, actions, params_alternative()),
        },
    ]
}

fn uniform_priors(graphs: &[CausalDag]) -> Vec<HypothesisPrior> {
    graphs.iter().map(|graph| HypothesisPrior {
        id: graph.id(),
        probability: 1.0 / graphs.len() as f64,
    }).collect()
}

fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities.iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.log2())
        .sum()
}

fn predictive_outcome_entropy(candidate: &ExperimentCandidate, priors: &[HypothesisPrior]) -> f64 {
    let outcomes = candidate.likelihoods.first().map(Vec::len).unwrap_or(0);
    let mut marginal = vec![0.0; outcomes];
    for (prior, likelihoods) in priors.iter().zip(&candidate.likelihoods) {
        for (index, probability) in likelihoods.iter().enumerate() {
            marginal[index] += prior.probability * probability;
        }
    }
    entropy_bits(&marginal)
}

fn fixed_design_action(
    graphs: &[CausalDag],
    actions: &[ScmAction],
    priors: &[HypothesisPrior],
    development_worlds: &[ScmWorld],
) -> Result<String, ScmGraphBenchmarkError> {
    let development_count = development_worlds.iter()
        .filter(|world| !world.family.held_out())
        .count();
    if development_count == 0 {
        return Err(ScmGraphBenchmarkError::MissingCandidate);
    }

    // Tune a single action on development-family mechanism parameters only.
    // Truth graph IDs are not used to choose the action. Held-out worlds are not
    // passed to this function and therefore cannot affect this baseline.
    let mut totals: Vec<(String, f64)> = actions.iter()
        .map(|action| (action.id.clone(), 0.0))
        .collect();
    for world in development_worlds.iter().filter(|world| !world.family.held_out()) {
        let candidates = candidate_set(graphs, actions, world.parameters);
        let scores = rank_experiments(priors, &candidates)?;
        for (action_id, total) in &mut totals {
            let score = scores.iter().find(|score| score.candidate_id == *action_id)
                .ok_or(ScmGraphBenchmarkError::MissingCandidate)?;
            *total += score.information_gain_per_cost / development_count as f64;
        }
    }
    totals.into_iter()
        .max_by(|(left_id, left_score), (right_id, right_score)| {
            left_score.total_cmp(right_score).then_with(|| right_id.cmp(left_id))
        })
        .map(|(action_id, _)| action_id)
        .ok_or(ScmGraphBenchmarkError::MissingCandidate)
}

fn choose_action(
    policy: ScmGraphPolicy,
    posterior: &[HypothesisPrior],
    scenarios: &[ModelLikelihoodScenario],
    oracle_candidates: &[ExperimentCandidate],
    fixed_action_id: &str,
    rng: &mut BenchmarkRng,
) -> Result<Option<String>, ScmGraphBenchmarkError> {
    let nominal = scenarios.first().ok_or(ScmGraphBenchmarkError::MissingCandidate)?;
    let selected = match policy {
        ScmGraphPolicy::NominalExpectedInformationGain => {
            rank_experiments(posterior, &nominal.candidates)?
                .first().map(|score| score.candidate_id.clone())
        }
        ScmGraphPolicy::PredictiveOutcomeEntropy => nominal.candidates.iter()
            .max_by(|left, right| {
                predictive_outcome_entropy(left, posterior)
                    .total_cmp(&predictive_outcome_entropy(right, posterior))
                    .then_with(|| right.id.cmp(&left.id))
            })
            .map(|candidate| candidate.id.clone()),
        ScmGraphPolicy::ModelWeightedExpectedInformationGain => {
            let scores = rank_experiments_robust(posterior, scenarios)?;
            scores.into_iter()
                .max_by(|left, right| {
                    left.expected_information_gain_per_cost
                        .total_cmp(&right.expected_information_gain_per_cost)
                        .then_with(|| right.candidate_id.cmp(&left.candidate_id))
                })
                .map(|score| score.candidate_id)
        }
        ScmGraphPolicy::WorstCaseRobust => rank_experiments_robust(posterior, scenarios)?
            .first().map(|score| score.candidate_id.clone()),
        ScmGraphPolicy::DevelopmentFixedDesign => Some(fixed_action_id.to_owned()),
        ScmGraphPolicy::UniformRandomAction => {
            Some(nominal.candidates[rng.index(nominal.candidates.len())].id.clone())
        }
        ScmGraphPolicy::PassiveOnly => Some("observe".into()),
        ScmGraphPolicy::KnownMechanicsOracle => rank_experiments(posterior, oracle_candidates)?
            .first().map(|score| score.candidate_id.clone()),
        ScmGraphPolicy::StructureBlindRandomGraph => None,
    };
    Ok(selected)
}

fn sample_outcome(distribution: &[f64], rng: &mut BenchmarkRng) -> usize {
    let draw = rng.unit_f64();
    let mut cumulative = 0.0;
    for (outcome, probability) in distribution.iter().enumerate() {
        cumulative += probability;
        if draw < cumulative {
            return outcome;
        }
    }
    distribution.len().saturating_sub(1)
}

fn posterior_top_index(posterior: &[HypothesisPrior]) -> usize {
    let mut best = 0;
    for index in 1..posterior.len() {
        if posterior[index].probability > posterior[best].probability {
            best = index;
        }
    }
    best
}

fn structure_metrics(predicted: CausalDag, truth: CausalDag) -> (f64, f64, f64, bool) {
    let mut skeleton_tp = 0usize;
    let mut skeleton_fp = 0usize;
    let mut skeleton_fn = 0usize;
    let mut directed_tp = 0usize;
    let mut directed_fp = 0usize;
    let mut directed_fn = 0usize;
    let mut shd = 0.0;

    for first in 0..NODE_COUNT {
        for second in (first + 1)..NODE_COUNT {
            let true_forward = truth.has_edge(first, second);
            let true_reverse = truth.has_edge(second, first);
            let pred_forward = predicted.has_edge(first, second);
            let pred_reverse = predicted.has_edge(second, first);
            let true_adjacent = true_forward || true_reverse;
            let pred_adjacent = pred_forward || pred_reverse;

            match (true_adjacent, pred_adjacent) {
                (true, true) => skeleton_tp += 1,
                (false, true) => skeleton_fp += 1,
                (true, false) => skeleton_fn += 1,
                (false, false) => {}
            }
            directed_tp += if pred_forward && true_forward { 1 } else { 0 };
            directed_tp += if pred_reverse && true_reverse { 1 } else { 0 };
            directed_fp += if pred_forward && !true_forward { 1 } else { 0 };
            directed_fp += if pred_reverse && !true_reverse { 1 } else { 0 };
            directed_fn += if true_forward && !pred_forward { 1 } else { 0 };
            directed_fn += if true_reverse && !pred_reverse { 1 } else { 0 };

            let exact_pair = true_forward == pred_forward && true_reverse == pred_reverse;
            if !exact_pair {
                // Standard small-graph edit accounting: reversal is one edit,
                // while adding/removing an adjacency is one edit.
                shd += 1.0;
            }
        }
    }

    let skeleton_f1 = f1_from_counts(skeleton_tp, skeleton_fp, skeleton_fn);
    let directed_f1 = f1_from_counts(directed_tp, directed_fp, directed_fn);
    (skeleton_f1, directed_f1, shd, predicted.edge_mask == truth.edge_mask)
}

fn f1_from_counts(tp: usize, fp: usize, fn_: usize) -> f64 {
    if tp + fp == 0 && tp + fn_ == 0 {
        return 1.0;
    }
    if tp == 0 {
        return 0.0;
    }
    (2.0 * tp as f64) / (2 * tp + fp + fn_) as f64
}

fn simulate_trial(
    policy: ScmGraphPolicy,
    graphs: &[CausalDag],
    actions: &[ScmAction],
    priors: &[HypothesisPrior],
    scenarios: &[ModelLikelihoodScenario],
    oracle_candidates: &[ExperimentCandidate],
    world: &ScmWorld,
    fixed_action_id: &str,
    measurement_budget: usize,
    posterior_threshold: f64,
    seed: u64,
) -> Result<TrialResult, ScmGraphBenchmarkError> {
    let mut posterior = priors.to_vec();
    let mut outcome_rng = BenchmarkRng::new(seed ^ 0x0B5E_7A71);
    let mut selection_rng = BenchmarkRng::new(seed ^ 0x51EC_7100);
    let truth_index = graphs.iter().position(|graph| *graph == world.truth)
        .ok_or(ScmGraphBenchmarkError::InvalidDagCatalogue)?;
    let mut measurements_used = 0usize;
    let mut interventions_used = 0usize;
    let mut measurements_to_threshold = None;
    let mut interventions_to_threshold = None;

    for _step in 1..=measurement_budget {
        if policy == ScmGraphPolicy::StructureBlindRandomGraph {
            break;
        }
        let selected_id = choose_action(
            policy,
            &posterior,
            scenarios,
            oracle_candidates,
            fixed_action_id,
            &mut selection_rng,
        )?.ok_or(ScmGraphBenchmarkError::MissingCandidate)?;
        let action = actions.iter().find(|action| action.id == selected_id)
            .ok_or(ScmGraphBenchmarkError::MissingCandidate)?;
        let hidden_candidate = oracle_candidates.iter()
            .find(|candidate| candidate.id == selected_id)
            .ok_or(ScmGraphBenchmarkError::MissingCandidate)?;
        let true_distribution = hidden_candidate.likelihoods.get(truth_index)
            .ok_or(ScmGraphBenchmarkError::MissingCandidate)?;
        let observed_outcome = sample_outcome(true_distribution, &mut outcome_rng);
        let model_candidate = scenarios[0].candidates.iter()
            .find(|candidate| candidate.id == selected_id)
            .ok_or(ScmGraphBenchmarkError::MissingCandidate)?;
        posterior = update_posterior(&posterior, model_candidate, observed_outcome)?.posterior;
        measurements_used += 1;
        if matches!(action.kind, ActionKind::Intervene { .. }) {
            interventions_used += 1;
        }
        if posterior[posterior_top_index(&posterior)].probability >= posterior_threshold {
            measurements_to_threshold = Some(measurements_used);
            interventions_to_threshold = Some(interventions_used);
            break;
        }
    }

    let map_index = if policy == ScmGraphPolicy::StructureBlindRandomGraph {
        selection_rng.index(graphs.len())
    } else {
        posterior_top_index(&posterior)
    };
    let predicted = graphs[map_index];
    let truth = graphs[truth_index];
    let (skeleton_f1, directed_edge_f1, shd, exact_graph) = structure_metrics(predicted, truth);
    let posterior_class_prediction = posterior_cpdag(&posterior, graphs).0;
    let predicted_cpdag = if policy == ScmGraphPolicy::StructureBlindRandomGraph {
        cpdag_marks(predicted, graphs)
    } else {
        posterior_class_prediction
    };
    let truth_cpdag = cpdag_marks(truth, graphs);
    let (cpdag_skeleton_f1, compelled_direction_f1, cpdag_mark_distance, markov_class_correct) =
        cpdag_metrics(&predicted_cpdag, &truth_cpdag);
    let posterior_true_markov_class = posterior_mass_on_class(&posterior, graphs, truth);
    let max_probability = posterior[posterior_top_index(&posterior)].probability;
    let claim_accepted = policy != ScmGraphPolicy::StructureBlindRandomGraph
        && max_probability >= posterior_threshold;
    let false_graph_claim = claim_accepted && !exact_graph;
    let graph_posterior_brier_score = posterior.iter().enumerate()
        .map(|(index, hypothesis)| {
            let target = if index == truth_index { 1.0 } else { 0.0 };
            (hypothesis.probability - target).powi(2)
        })
        .sum();

    Ok(TrialResult {
        skeleton_f1,
        directed_edge_f1,
        shd,
        exact_graph,
        cpdag_skeleton_f1,
        compelled_direction_f1,
        cpdag_mark_distance,
        markov_equivalence_class_correct: markov_class_correct,
        posterior_true_graph: posterior[truth_index].probability,
        posterior_true_markov_class,
        graph_posterior_brier_score,
        claim_accepted,
        false_graph_claim,
        measurements_used,
        interventions_used,
        measurements_to_threshold,
        interventions_to_threshold,
    })
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn mean_ci95(values: &[f64], minimum: f64, maximum: f64) -> [f64; 2] {
    if values.len() < 2 {
        return [minimum, maximum];
    }
    let count = values.len() as f64;
    let average = mean(values);
    let variance = values.iter().map(|value| (value - average).powi(2)).sum::<f64>()
        / (count - 1.0);
    let half_width = 1.96 * (variance / count).sqrt();
    [
        (average - half_width).clamp(minimum, maximum),
        (average + half_width).clamp(minimum, maximum),
    ]
}

fn wilson_ci95(successes: usize, count: usize) -> [f64; 2] {
    if count == 0 {
        return [0.0, 1.0];
    }
    let n = count as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt()
        / denominator;
    [(center - half_width).clamp(0.0, 1.0), (center + half_width).clamp(0.0, 1.0)]
}

fn summarize_world(
    world: &ScmWorld,
    policy: ScmGraphPolicy,
    measurement_budget: usize,
    trials: Vec<TrialResult>,
) -> ScmGraphPolicySummary {
    let count = trials.len();
    let skeleton: Vec<f64> = trials.iter().map(|trial| trial.skeleton_f1).collect();
    let directed: Vec<f64> = trials.iter().map(|trial| trial.directed_edge_f1).collect();
    let shd: Vec<f64> = trials.iter().map(|trial| trial.shd).collect();
    let cpdag_skeleton: Vec<f64> = trials.iter().map(|trial| trial.cpdag_skeleton_f1).collect();
    let compelled_direction: Vec<f64> = trials.iter()
        .map(|trial| trial.compelled_direction_f1).collect();
    let cpdag_distance: Vec<f64> = trials.iter().map(|trial| trial.cpdag_mark_distance).collect();
    let cpdag_correct = trials.iter()
        .filter(|trial| trial.markov_equivalence_class_correct).count();
    let true_mass: Vec<f64> = trials.iter().map(|trial| trial.posterior_true_graph).collect();
    let true_class_mass: Vec<f64> = trials.iter()
        .map(|trial| trial.posterior_true_markov_class).collect();
    let brier: Vec<f64> = trials.iter().map(|trial| trial.graph_posterior_brier_score).collect();
    let exact = trials.iter().filter(|trial| trial.exact_graph).count();
    let accepted = trials.iter().filter(|trial| trial.claim_accepted).count();
    let false_claims = trials.iter().filter(|trial| trial.false_graph_claim).count();
    let accepted_false_claims = trials.iter()
        .filter(|trial| trial.claim_accepted && trial.false_graph_claim).count();
    let measurements: Vec<f64> = trials.iter()
        .map(|trial| trial.measurements_used as f64).collect();
    let interventions: Vec<f64> = trials.iter()
        .map(|trial| trial.interventions_used as f64).collect();
    let measurements_to_threshold: Vec<f64> = trials.iter()
        .filter_map(|trial| trial.measurements_to_threshold.map(|value| value as f64)).collect();
    let interventions_to_threshold: Vec<f64> = trials.iter()
        .filter_map(|trial| trial.interventions_to_threshold.map(|value| value as f64)).collect();

    ScmGraphPolicySummary {
        world_id: world.id.clone(),
        truth_graph_id: world.truth.id(),
        truth_edge_count: world.truth.edge_count(),
        family: world.family,
        held_out_family: world.family.held_out(),
        policy,
        trials: count,
        measurement_budget,
        mean_skeleton_f1: mean(&skeleton),
        skeleton_f1_ci95: mean_ci95(&skeleton, 0.0, 1.0),
        mean_directed_edge_f1: mean(&directed),
        directed_edge_f1_ci95: mean_ci95(&directed, 0.0, 1.0),
        mean_cpdag_skeleton_f1: mean(&cpdag_skeleton),
        cpdag_skeleton_f1_ci95: mean_ci95(&cpdag_skeleton, 0.0, 1.0),
        mean_compelled_direction_f1: mean(&compelled_direction),
        compelled_direction_f1_ci95: mean_ci95(&compelled_direction, 0.0, 1.0),
        mean_cpdag_mark_distance: mean(&cpdag_distance),
        cpdag_mark_distance_ci95: mean_ci95(&cpdag_distance, 0.0, (NODE_COUNT * (NODE_COUNT - 1) / 2) as f64),
        markov_equivalence_class_recovery_rate: cpdag_correct as f64 / count as f64,
        markov_equivalence_class_recovery_wilson_ci95: wilson_ci95(cpdag_correct, count),
        mean_posterior_mass_on_true_markov_class: mean(&true_class_mass),
        posterior_mass_on_true_markov_class_ci95: mean_ci95(&true_class_mass, 0.0, 1.0),
        mean_structural_hamming_distance: mean(&shd),
        structural_hamming_distance_ci95: mean_ci95(&shd, 0.0, (NODE_COUNT * (NODE_COUNT - 1) / 2) as f64),
        exact_graph_recovery_rate: exact as f64 / count as f64,
        exact_graph_recovery_wilson_ci95: wilson_ci95(exact, count),
        mean_posterior_mass_on_true_graph: mean(&true_mass),
        posterior_mass_on_true_graph_ci95: mean_ci95(&true_mass, 0.0, 1.0),
        mean_graph_posterior_brier_score: mean(&brier),
        graph_posterior_brier_ci95: mean_ci95(&brier, 0.0, 2.0),
        graph_claim_coverage: accepted as f64 / count as f64,
        graph_claim_coverage_wilson_ci95: wilson_ci95(accepted, count),
        false_graph_claim_rate_among_accepted: (accepted > 0)
            .then_some(accepted_false_claims as f64 / accepted as f64),
        false_graph_claim_rate_among_accepted_wilson_ci95: (accepted > 0)
            .then_some(wilson_ci95(accepted_false_claims, accepted)),
        false_graph_claim_rate_all_trials: false_claims as f64 / count as f64,
        mean_measurements_used: mean(&measurements),
        mean_interventions_used: mean(&interventions),
        mean_measurements_to_threshold: (!measurements_to_threshold.is_empty())
            .then(|| mean(&measurements_to_threshold)),
        mean_interventions_to_threshold: (!interventions_to_threshold.is_empty())
            .then(|| mean(&interventions_to_threshold)),
    }
}

fn aggregate_held_out(
    summaries: &[ScmGraphPolicySummary],
    policies: &[ScmGraphPolicy],
) -> Vec<ScmGraphPolicyAggregate> {
    policies.iter().copied().map(|policy| {
        let rows: Vec<&ScmGraphPolicySummary> = summaries.iter()
            .filter(|row| row.policy == policy && row.held_out_family)
            .collect();
        let skeleton: Vec<f64> = rows.iter().map(|row| row.mean_skeleton_f1).collect();
        let directed: Vec<f64> = rows.iter().map(|row| row.mean_directed_edge_f1).collect();
        let cpdag_skeleton: Vec<f64> = rows.iter().map(|row| row.mean_cpdag_skeleton_f1).collect();
        let compelled: Vec<f64> = rows.iter().map(|row| row.mean_compelled_direction_f1).collect();
        let cpdag_distance: Vec<f64> = rows.iter().map(|row| row.mean_cpdag_mark_distance).collect();
        let cpdag_accuracy: Vec<f64> = rows.iter()
            .map(|row| row.markov_equivalence_class_recovery_rate).collect();
        let shd: Vec<f64> = rows.iter().map(|row| row.mean_structural_hamming_distance).collect();
        let exact: Vec<f64> = rows.iter().map(|row| row.exact_graph_recovery_rate).collect();
        let coverage: Vec<f64> = rows.iter().map(|row| row.graph_claim_coverage).collect();
        let false_claims: Vec<f64> = rows.iter()
            .map(|row| row.false_graph_claim_rate_all_trials).collect();
        ScmGraphPolicyAggregate {
            policy,
            held_out_worlds: rows.len(),
            mean_held_out_skeleton_f1: mean(&skeleton),
            held_out_skeleton_f1_world_ci95: mean_ci95(&skeleton, 0.0, 1.0),
            mean_held_out_directed_edge_f1: mean(&directed),
            held_out_directed_edge_f1_world_ci95: mean_ci95(&directed, 0.0, 1.0),
            mean_held_out_cpdag_skeleton_f1: mean(&cpdag_skeleton),
            held_out_cpdag_skeleton_f1_world_ci95: mean_ci95(&cpdag_skeleton, 0.0, 1.0),
            mean_held_out_compelled_direction_f1: mean(&compelled),
            held_out_compelled_direction_f1_world_ci95: mean_ci95(&compelled, 0.0, 1.0),
            mean_held_out_cpdag_mark_distance: mean(&cpdag_distance),
            held_out_cpdag_mark_distance_world_ci95: mean_ci95(&cpdag_distance, 0.0, (NODE_COUNT * (NODE_COUNT - 1) / 2) as f64),
            held_out_markov_equivalence_class_recovery_rate: mean(&cpdag_accuracy),
            mean_held_out_shd: mean(&shd),
            held_out_shd_world_ci95: mean_ci95(&shd, 0.0, (NODE_COUNT * (NODE_COUNT - 1) / 2) as f64),
            worst_world_directed_edge_f1: directed.iter().copied().fold(1.0, f64::min),
            worst_world_shd: shd.iter().copied().fold(0.0, f64::max),
            mean_held_out_exact_graph_recovery_rate: mean(&exact),
            mean_held_out_claim_coverage: mean(&coverage),
            mean_held_out_false_graph_claim_rate_all_trials: mean(&false_claims),
        }
    }).collect()
}

/// Run a deterministic graph-recovery experiment under equal measurement budgets.
///
/// Candidate hypotheses are all 25 labeled DAGs over three observed variables.
/// There are seven seed-generated world families (two development families and
/// five held-out families), each with four parameterized worlds. The policy never
/// receives the hidden truth graph or hidden structural-equation parameters.
/// Outcomes are sampled from the hidden graph's exact binary SCM distribution.
pub fn run_scm_graph_benchmark(
    trials_per_world: usize,
    measurement_budget: usize,
    seed: u64,
    posterior_threshold: f64,
) -> Result<ScmGraphBenchmarkReport, ScmGraphBenchmarkError> {
    if !(3..=MAX_TRIALS_PER_WORLD).contains(&trials_per_world) {
        return Err(ScmGraphBenchmarkError::InvalidTrialCount);
    }
    if !(1..=MAX_MEASUREMENT_BUDGET).contains(&measurement_budget) {
        return Err(ScmGraphBenchmarkError::InvalidMeasurementBudget);
    }

    let graphs = all_dags();
    if graphs.len() != 25 {
        return Err(ScmGraphBenchmarkError::InvalidDagCatalogue);
    }
    if !posterior_threshold.is_finite()
        || posterior_threshold <= 1.0 / graphs.len() as f64
        || posterior_threshold > 1.0
    {
        return Err(ScmGraphBenchmarkError::InvalidPosteriorThreshold);
    }

    let actions = actions();
    let priors = uniform_priors(&graphs);
    let scenarios = declared_model_scenarios(&graphs, &actions);
    let worlds = generate_worlds(seed);
    let fixed_action_id = fixed_design_action(&graphs, &actions, &priors, &worlds)?;
    let policies = ScmGraphPolicy::all();
    let mut summaries = Vec::with_capacity(worlds.len() * policies.len());

    for (world_index, world) in worlds.iter().enumerate() {
        let oracle_candidates = candidate_set(&graphs, &actions, world.parameters);
        for policy in policies {
            let mut trials = Vec::with_capacity(trials_per_world);
            for trial_index in 0..trials_per_world {
                let trial_seed = seed
                    ^ (world_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
                    ^ (trial_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                trials.push(simulate_trial(
                    policy,
                    &graphs,
                    &actions,
                    &priors,
                    &scenarios,
                    &oracle_candidates,
                    world,
                    &fixed_action_id,
                    measurement_budget,
                    posterior_threshold,
                    trial_seed,
                )?);
            }
            summaries.push(summarize_world(world, policy, measurement_budget, trials));
        }
    }

    Ok(ScmGraphBenchmarkReport {
        schema_version: 1,
        generator_version: GENERATOR_VERSION.into(),
        seed,
        trials_per_world,
        measurement_budget,
        posterior_threshold,
        observed_node_count: NODE_COUNT,
        candidate_dag_count: graphs.len(),
        world_variants_per_family: WORLD_VARIANTS_PER_FAMILY,
        world_count: worlds.len(),
        held_out_world_count: worlds.iter().filter(|world| world.family.held_out()).count(),
        declared_model_scenarios: scenarios.iter()
            .map(|scenario| scenario.scenario_id.clone()).collect(),
        fixed_design_action_id: fixed_action_id,
        held_out_policy_aggregates: aggregate_held_out(&summaries, &policies),
        worlds: summaries,
        interpretation_boundary: concat!(
            "Synthetic software benchmark only. The candidate set enumerates all labeled DAGs ",
            "on three observed variables, but outcomes use hidden structural-equation parameters. ",
            "Held-out regimes are source-visible and are not cryptographically sealed. The ",
            "held-out aggregate confidence intervals treat parameterized worlds as units, but ",
            "four variants per family are not a population sample of all causal systems. The ",
            "known-mechanics oracle is an optimistic diagnostic, not an available deployment policy."
        ).into(),
    })
}

struct BenchmarkRng {
    state: u64,
}

impl BenchmarkRng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    fn index(&mut self, upper_bound: usize) -> usize {
        assert!(upper_bound > 0, "index bound must be positive");
        let bound = upper_bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_catalogue_contains_all_twenty_five_three_node_dags() {
        let graphs = all_dags();
        assert_eq!(graphs.len(), 25);
        assert!(graphs.iter().all(|graph| is_acyclic(graph.edge_mask)));
        assert_eq!(
            graphs.iter().map(|graph| graph.edge_mask).collect::<std::collections::BTreeSet<_>>().len(),
            25
        );
    }

    #[test]
    fn exact_scm_distributions_are_normalized_and_interventions_respect_do_value() {
        let graphs = all_dags();
        let parameters = params_nominal();
        for graph in &graphs {
            let observe = ScmAction { id: "observe".into(), kind: ActionKind::Observe, cost: 1.0 };
            let distribution = outcome_distribution(*graph, &observe, parameters);
            assert!((distribution.iter().sum::<f64>() - 1.0).abs() < 1.0e-9);
            assert!(distribution.iter().all(|value| value.is_finite() && *value >= 0.0));

            for node in 0..NODE_COUNT {
                for value in 0..=1 {
                    let action = ScmAction {
                        id: format!("do_{node}_{value}"),
                        kind: ActionKind::Intervene { node, value },
                        cost: 1.0,
                    };
                    let distribution = outcome_distribution(*graph, &action, parameters);
                    assert!((distribution.iter().sum::<f64>() - 1.0).abs() < 1.0e-9);
                    assert!(distribution.iter().enumerate().all(|(state, probability)| {
                        ((state >> node) & 1) == value || *probability == 0.0
                    }));
                }
            }
        }
    }

    #[test]
    fn graph_metrics_have_known_answers_for_exact_reversal_and_null_graphs() {
        let empty = CausalDag { edge_mask: 0 };
        let forward = CausalDag { edge_mask: edge_mask_with(&[(0, 1)]) };
        let reverse = CausalDag { edge_mask: edge_mask_with(&[(1, 0)]) };
        assert_eq!(structure_metrics(empty, empty), (1.0, 1.0, 0.0, true));
        assert_eq!(structure_metrics(forward, forward), (1.0, 1.0, 0.0, true));
        let (skeleton, directed, shd, exact) = structure_metrics(reverse, forward);
        assert_eq!(skeleton, 1.0);
        assert_eq!(directed, 0.0);
        assert_eq!(shd, 1.0);
        assert!(!exact);
    }

    #[test]
    fn worlds_replay_for_a_seed_and_cover_held_out_families() {
        let first = generate_worlds(44);
        let replay = generate_worlds(44);
        let changed = generate_worlds(45);
        assert_eq!(first.len(), 28);
        assert_eq!(first.len(), replay.len());
        assert_eq!(first.iter().map(|world| world.truth).collect::<Vec<_>>(),
            replay.iter().map(|world| world.truth).collect::<Vec<_>>());
        assert_eq!(first.iter().map(|world| world.parameters).collect::<Vec<_>>(),
            replay.iter().map(|world| world.parameters).collect::<Vec<_>>());
        assert_ne!(first.iter().map(|world| world.truth).collect::<Vec<_>>(),
            changed.iter().map(|world| world.truth).collect::<Vec<_>>());
        assert_eq!(first.iter().filter(|world| world.family.held_out()).count(), 20);
        assert!(first.iter().filter(|world| {
            world.family == ScmGraphFamily::NullGraphNegativeControl
        }).all(|world| world.truth.edge_count() == 0));
    }

    #[test]
    fn development_fixed_action_ignores_held_out_worlds() {
        let graphs = all_dags();
        let actions = actions();
        let priors = uniform_priors(&graphs);
        let worlds = generate_worlds(42);
        let chosen_all = fixed_design_action(&graphs, &actions, &priors, &worlds).unwrap();
        let dev_only: Vec<ScmWorld> = worlds.iter()
            .filter(|world| !world.family.held_out())
            .cloned()
            .collect();
        let chosen_dev = fixed_design_action(&graphs, &actions, &priors, &dev_only).unwrap();
        assert_eq!(chosen_all, chosen_dev);
        assert!(!chosen_all.is_empty());
    }

    #[test]
    fn run_is_deterministic_and_reports_world_level_aggregates() {
        let a = run_scm_graph_benchmark(3, 4, 123, 0.95).unwrap();
        let b = run_scm_graph_benchmark(3, 4, 123, 0.95).unwrap();
        assert_eq!(a, b);
        assert_eq!(serde_json::to_vec(&a).unwrap(), serde_json::to_vec(&b).unwrap());
        assert_eq!(a.world_count, 28);
        assert_eq!(a.held_out_world_count, 20);
        assert_eq!(a.candidate_dag_count, 25);
        assert_eq!(a.worlds.len(), 28 * ScmGraphPolicy::all().len());
        assert_eq!(a.held_out_policy_aggregates.len(), ScmGraphPolicy::all().len());
    }

    #[test]
    fn held_out_claim_metrics_are_bounded_and_null_false_claims_are_visible() {
        let report = run_scm_graph_benchmark(6, 5, 0x5EED, 0.90).unwrap();
        for row in &report.worlds {
            assert!((0.0..=1.0).contains(&row.mean_skeleton_f1));
            assert!((0.0..=1.0).contains(&row.mean_directed_edge_f1));
            assert!((0.0..=3.0).contains(&row.mean_structural_hamming_distance));
            assert!((0.0..=1.0).contains(&row.graph_claim_coverage));
            assert!((0.0..=1.0).contains(&row.false_graph_claim_rate_all_trials));
            if row.family == ScmGraphFamily::NullGraphNegativeControl {
                assert!(row.mean_structural_hamming_distance >= 0.0);
            }
        }
        for aggregate in &report.held_out_policy_aggregates {
            assert_eq!(aggregate.held_out_worlds, 20);
            assert!((0.0..=1.0).contains(&aggregate.mean_held_out_directed_edge_f1));
            assert!((0.0..=3.0).contains(&aggregate.mean_held_out_shd));
            assert!((0.0..=1.0).contains(&aggregate.mean_held_out_claim_coverage));
        }
    }

    #[test]
    fn invalid_configuration_fails_closed() {
        assert_eq!(
            run_scm_graph_benchmark(2, 8, 1, 0.95),
            Err(ScmGraphBenchmarkError::InvalidTrialCount)
        );
        assert_eq!(
            run_scm_graph_benchmark(3, 0, 1, 0.95),
            Err(ScmGraphBenchmarkError::InvalidMeasurementBudget)
        );
        assert_eq!(
            run_scm_graph_benchmark(3, 8, 1, 1.0 / 25.0),
            Err(ScmGraphBenchmarkError::InvalidPosteriorThreshold)
        );
    }
}
