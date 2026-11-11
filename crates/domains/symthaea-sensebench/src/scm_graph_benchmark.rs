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
const MECHANISM_REGIME_COUNT: usize = 4;
const MAX_TRIALS_PER_WORLD: usize = 10_000;
const MAX_MEASUREMENT_BUDGET: usize = 64;
const GENERATOR_VERSION: &str = "sensebench-scm-graphs-0.6.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScmGraphFamily {
    ChainDevelopment,
    ForkDevelopment,
    ColliderHeldOut,
    SparseRandomHeldOut,
    DenseRandomHeldOut,
    RandomMotifHeldOut,
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
            Self::RandomMotifHeldOut,
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
            Self::RandomMotifHeldOut => "random_motif_held_out",
            Self::NullGraphNegativeControl => "null_graph_negative_control",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScmMechanismRegime {
    WeakPositiveDevelopment,
    StrongPositiveDevelopment,
    MixedSignHeldOut,
    NearNullHeldOut,
}

impl ScmMechanismRegime {
    fn all() -> [Self; MECHANISM_REGIME_COUNT] {
        [
            Self::WeakPositiveDevelopment,
            Self::StrongPositiveDevelopment,
            Self::MixedSignHeldOut,
            Self::NearNullHeldOut,
        ]
    }

    fn held_out(self) -> bool {
        matches!(self, Self::MixedSignHeldOut | Self::NearNullHeldOut)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::WeakPositiveDevelopment => "weak_positive_development",
            Self::StrongPositiveDevelopment => "strong_positive_development",
            Self::MixedSignHeldOut => "mixed_sign_held_out",
            Self::NearNullHeldOut => "near_null_held_out",
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
    PcObservationalBaseline,
    PcExactConditionalBaseline,
}

impl ScmGraphPolicy {
    fn all() -> [Self; 11] {
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
            Self::PcObservationalBaseline,
            Self::PcExactConditionalBaseline,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScmGraphPolicySummary {
    pub world_id: String,
    pub graph_variant_index: usize,
    /// Evaluator-side truth is emitted only in the final report, not supplied to policies.
    pub truth_graph_id: String,
    pub truth_edge_count: usize,
    pub family: ScmGraphFamily,
    pub mechanism_regime: ScmMechanismRegime,
    pub held_out_family: bool,
    pub held_out_world: bool,
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
    /// PC baseline only: mean number of edge marks changed by valid-catalogue projection.
    pub pc_raw_to_projected_mark_changes_mean: Option<f64>,
    /// PC baseline only: fraction of trials where projection changes at least one mark.
    pub pc_projection_changed_trial_rate: Option<f64>,
    pub markov_equivalence_class_recovery_rate: f64,
    pub markov_equivalence_class_recovery_wilson_ci95: [f64; 2],
    /// Claim coverage using posterior mass summed over a Markov equivalence class.
    pub cpdag_claim_coverage: f64,
    pub cpdag_claim_coverage_wilson_ci95: [f64; 2],
    pub false_cpdag_claim_rate_among_accepted: Option<f64>,
    pub false_cpdag_claim_rate_among_accepted_wilson_ci95: Option<[f64; 2]>,
    pub false_cpdag_claim_rate_all_trials: f64,
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
    /// Distinct graph topologies represented among the held-out worlds.
    pub held_out_topology_instances: usize,
    /// Confidence intervals resample topology instances after averaging paired
    /// held-out mechanism regimes; they are conditional on the seven fixed family templates.
    pub mean_held_out_skeleton_f1: f64,
    pub held_out_skeleton_f1_topology_instance_ci95: [f64; 2],
    pub mean_held_out_directed_edge_f1: f64,
    pub held_out_directed_edge_f1_topology_instance_ci95: [f64; 2],
    pub mean_held_out_cpdag_skeleton_f1: f64,
    pub held_out_cpdag_skeleton_f1_topology_instance_ci95: [f64; 2],
    pub mean_held_out_compelled_direction_f1: f64,
    pub held_out_compelled_direction_f1_topology_instance_ci95: [f64; 2],
    pub mean_held_out_cpdag_mark_distance: f64,
    pub held_out_cpdag_mark_distance_topology_instance_ci95: [f64; 2],
    pub held_out_markov_equivalence_class_recovery_rate: f64,
    pub mean_held_out_cpdag_claim_coverage: f64,
    pub mean_held_out_false_cpdag_claim_rate_all_trials: f64,
    pub mean_held_out_shd: f64,
    pub held_out_shd_topology_instance_ci95: [f64; 2],
    pub worst_world_directed_edge_f1: f64,
    pub worst_world_shd: f64,
    pub mean_held_out_exact_graph_recovery_rate: f64,
    pub mean_held_out_claim_coverage: f64,
    pub mean_held_out_false_graph_claim_rate_all_trials: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScmGraphShiftSplit {
    AnyHeldOut,
    TopologyShiftOnly,
    MechanismShiftOnly,
    JointShift,
}

impl ScmGraphShiftSplit {
    fn all() -> [Self; 4] {
        [
            Self::AnyHeldOut,
            Self::TopologyShiftOnly,
            Self::MechanismShiftOnly,
            Self::JointShift,
        ]
    }

    fn includes(self, row: &ScmGraphPolicySummary) -> bool {
        match self {
            Self::AnyHeldOut => row.held_out_world,
            Self::TopologyShiftOnly => row.held_out_family && !row.mechanism_regime.held_out(),
            Self::MechanismShiftOnly => !row.held_out_family && row.mechanism_regime.held_out(),
            Self::JointShift => row.held_out_family && row.mechanism_regime.held_out(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScmGraphFactorizedShiftAggregate {
    pub policy: ScmGraphPolicy,
    pub split: ScmGraphShiftSplit,
    pub world_rows: usize,
    pub topology_instances: usize,
    pub mean_skeleton_f1: f64,
    pub skeleton_f1_topology_instance_ci95: [f64; 2],
    pub mean_directed_edge_f1: f64,
    pub directed_edge_f1_topology_instance_ci95: [f64; 2],
    pub mean_cpdag_skeleton_f1: f64,
    pub cpdag_skeleton_f1_topology_instance_ci95: [f64; 2],
    pub mean_structural_hamming_distance: f64,
    pub shd_topology_instance_ci95: [f64; 2],
    pub mean_exact_graph_recovery_rate: f64,
    pub mean_markov_equivalence_class_recovery_rate: f64,
    pub mean_full_graph_claim_coverage: f64,
    pub mean_false_full_graph_claim_rate: f64,
    pub mean_cpdag_claim_coverage: f64,
    pub mean_false_cpdag_claim_rate: f64,
    pub worst_topology_directed_edge_f1: f64,
    pub worst_topology_shd: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScmGraphPairedPolicyDelta {
    pub policy: ScmGraphPolicy,
    pub comparator: ScmGraphPolicy,
    pub split: ScmGraphShiftSplit,
    pub paired_world_rows: usize,
    pub paired_topology_instances: usize,
    /// Policy minus comparator; positive values mean better directed-edge F1.
    pub delta_directed_edge_f1: f64,
    pub delta_directed_edge_f1_topology_instance_ci95: [f64; 2],
    /// Policy minus comparator; positive values mean better CPDAG skeleton F1.
    pub delta_cpdag_skeleton_f1: f64,
    pub delta_cpdag_skeleton_f1_topology_instance_ci95: [f64; 2],
    /// Policy minus comparator; negative values mean fewer graph edits.
    pub delta_structural_hamming_distance: f64,
    pub delta_shd_topology_instance_ci95: [f64; 2],
    /// Policy minus comparator; negative values mean fewer measurements/interventions.
    pub delta_measurements_used: f64,
    pub delta_measurements_topology_instance_ci95: [f64; 2],
    pub delta_interventions_used: f64,
    pub delta_interventions_topology_instance_ci95: [f64; 2],
    /// Policy minus comparator; negative values mean fewer false graph claims.
    pub delta_false_graph_claim_rate: f64,
    pub delta_false_graph_claim_rate_topology_instance_ci95: [f64; 2],
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
    pub markov_equivalence_class_count: usize,
    pub world_variants_per_family: usize,
    pub mechanism_regime_count: usize,
    pub world_count: usize,
    pub development_world_count: usize,
    pub held_out_world_count: usize,
    pub declared_model_scenarios: Vec<String>,
    pub fixed_design_action_id: String,
    pub worlds: Vec<ScmGraphPolicySummary>,
    pub held_out_policy_aggregates: Vec<ScmGraphPolicyAggregate>,
    pub factorized_shift_aggregates: Vec<ScmGraphFactorizedShiftAggregate>,
    pub paired_policy_deltas: Vec<ScmGraphPairedPolicyDelta>,
    pub observational_pc_baseline: ScmGraphPolicy,
    pub exact_observational_pc_baseline: ScmGraphPolicy,
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

#[derive(Debug, Clone)]
struct MarkovClassCatalog {
    class_for_graph: Vec<usize>,
    members: Vec<Vec<usize>>,
    cpdag_marks: Vec<Vec<EdgeMark>>,
}

fn build_markov_class_catalog(catalogue: &[CausalDag]) -> MarkovClassCatalog {
    let mut class_for_graph = vec![usize::MAX; catalogue.len()];
    let mut members: Vec<Vec<usize>> = Vec::new();
    let mut marks: Vec<Vec<EdgeMark>> = Vec::new();
    for (candidate_index, candidate) in catalogue.iter().enumerate() {
        if class_for_graph[candidate_index] != usize::MAX {
            continue;
        }
        let class_index = members.len();
        let class_members: Vec<usize> = catalogue.iter().enumerate()
            .filter(|(_, other)| markov_equivalent(*candidate, **other))
            .map(|(index, _)| index)
            .collect();
        for member in &class_members {
            class_for_graph[*member] = class_index;
        }
        members.push(class_members);
        marks.push(cpdag_marks(*candidate, catalogue));
    }
    MarkovClassCatalog {
        class_for_graph,
        members,
        cpdag_marks: marks,
    }
}

/// Select the Markov equivalence class with the greatest summed posterior mass.
/// Its CPDAG comes from the precomputed valid class catalogue, avoiding pairwise
/// threshold combinations that can create invalid partially directed graphs.
fn posterior_cpdag(
    posterior: &[HypothesisPrior],
    classes: &MarkovClassCatalog,
) -> (Vec<EdgeMark>, f64, usize) {
    let mut best_class_index = 0usize;
    let mut best_mass = -1.0_f64;
    for (class_index, members) in classes.members.iter().enumerate() {
        let class_mass = members.iter()
            .map(|member| posterior[*member].probability)
            .sum::<f64>();
        if class_mass > best_mass {
            best_mass = class_mass;
            best_class_index = class_index;
        }
    }
    (
        classes.cpdag_marks[best_class_index].clone(),
        best_mass.max(0.0),
        best_class_index,
    )
}

fn posterior_mass_on_class(
    posterior: &[HypothesisPrior],
    classes: &MarkovClassCatalog,
    target_graph_index: usize,
) -> f64 {
    classes.members[classes.class_for_graph[target_graph_index]].iter()
        .map(|member| posterior[*member].probability)
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


const PC_G2_CRITICAL_DF1_95: f64 = 3.841_458_820_694_124;
const PC_G2_CRITICAL_DF2_95: f64 = 5.991_464_547_107_979;

/// Conditional G-squared statistic for binary variables X and Y, optionally
/// conditioning on the third binary variable. This transparent three-node PC
/// baseline uses asymptotic 5% thresholds; at low sample counts these tests are
/// approximate and must be judged on the held-out benchmark, not presumed reliable.
fn g_squared_for_table(table: &[[f64; 2]; 2]) -> f64 {
    let n = table.iter().flatten().sum::<f64>();
    if n == 0.0 {
        return 0.0;
    }
    let rows = [
        table[0][0] + table[0][1],
        table[1][0] + table[1][1],
    ];
    let columns = [
        table[0][0] + table[1][0],
        table[0][1] + table[1][1],
    ];
    let mut g_squared = 0.0;
    for x_value in 0..2 {
        for y_value in 0..2 {
            let observed = table[x_value][y_value];
            if observed == 0.0 {
                continue;
            }
            let expected = rows[x_value] * columns[y_value] / n;
            if expected > 0.0 {
                g_squared += 2.0 * observed * (observed / expected).ln();
            }
        }
    }
    g_squared.max(0.0)
}

fn contingency_tables(
    observations: &[usize],
    x: usize,
    y: usize,
    conditioning: Option<usize>,
) -> Vec<[[f64; 2]; 2]> {
    let strata = if conditioning.is_some() { 2 } else { 1 };
    let mut counts = vec![[[0.0_f64; 2]; 2]; strata];
    for state in observations {
        let x_value = (state >> x) & 1;
        let y_value = (state >> y) & 1;
        let stratum = conditioning.map_or(0, |node| (state >> node) & 1);
        counts[stratum][x_value][y_value] += 1.0;
    }
    counts
}

/// Asymptotic G-squared statistic for binary variables X and Y, optionally
/// conditioned on the third binary variable.
fn g_squared_ci_statistic(
    observations: &[usize],
    x: usize,
    y: usize,
    conditioning: Option<usize>,
) -> f64 {
    contingency_tables(observations, x, y, conditioning).iter()
        .map(g_squared_for_table)
        .sum::<f64>()
        .max(0.0)
}

fn log_binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    (1..=k).map(|i| ((n - k + i) as f64).ln() - (i as f64).ln()).sum()
}

/// Enumerate the exact conditional null distribution of G-squared for one 2x2
/// table, conditioning on both margins. The cell count follows a hypergeometric
/// distribution under independence.
fn exact_g_squared_stratum_distribution(table: &[[f64; 2]; 2]) -> Vec<(f64, f64)> {
    let n = table.iter().flatten().map(|count| *count as usize).sum::<usize>();
    let row_one = (table[1][0] + table[1][1]) as usize;
    let column_one = (table[0][1] + table[1][1]) as usize;
    let low = row_one.saturating_add(column_one).saturating_sub(n);
    let high = row_one.min(column_one);
    if low > high {
        return Vec::new();
    }

    let denominator = log_binomial_coefficient(n, row_one);
    (low..=high).map(|both_one| {
        let row_zero = n - row_one;
        let column_zero_in_row_one = row_one - both_one;
        let column_one_in_row_zero = column_one - both_one;
        let table_zero_zero = row_zero - column_one_in_row_zero;
        let expected_table = [
            [table_zero_zero as f64, column_one_in_row_zero as f64],
            [column_zero_in_row_one as f64, both_one as f64],
        ];
        let log_probability = log_binomial_coefficient(column_one, both_one)
            + log_binomial_coefficient(n - column_one, row_one - both_one)
            - denominator;
        (g_squared_for_table(&expected_table), log_probability.exp())
    }).collect()
}

/// Exact conditional G-squared p-value for unconditional or one-variable-
/// conditioned independence between binary variables. The test conditions on
/// each stratum's observed margins and enumerates the finite null distribution.
fn exact_conditional_g_squared_p_value(
    observations: &[usize],
    x: usize,
    y: usize,
    conditioning: Option<usize>,
) -> f64 {
    if observations.is_empty() {
        return 1.0;
    }
    let tables = contingency_tables(observations, x, y, conditioning);
    let observed = tables.iter().map(g_squared_for_table).sum::<f64>();
    let mut null_distribution = vec![(0.0_f64, 1.0_f64)];
    for table in &tables {
        let stratum_distribution = exact_g_squared_stratum_distribution(table);
        let mut expanded = Vec::with_capacity(
            null_distribution.len().saturating_mul(stratum_distribution.len()),
        );
        for (prior_statistic, prior_probability) in &null_distribution {
            for (stratum_statistic, stratum_probability) in &stratum_distribution {
                expanded.push((
                    prior_statistic + stratum_statistic,
                    prior_probability * stratum_probability,
                ));
            }
        }
        null_distribution = expanded;
    }
    null_distribution.iter()
        .filter(|(statistic, _)| *statistic + 1.0e-12 >= observed)
        .map(|(_, probability)| *probability)
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

fn pc_ci_test_independent(
    observations: &[usize],
    x: usize,
    y: usize,
    conditioning: Option<usize>,
    exact_conditional: bool,
) -> bool {
    if exact_conditional {
        exact_conditional_g_squared_p_value(observations, x, y, conditioning) >= 0.05
    } else {
        let statistic = g_squared_ci_statistic(observations, x, y, conditioning);
        let critical = if conditioning.is_some() {
            PC_G2_CRITICAL_DF2_95
        } else {
            PC_G2_CRITICAL_DF1_95
        };
        statistic <= critical
    }
}

fn pc_observational_cpdag(
    observations: &[usize],
    classes: &MarkovClassCatalog,
) -> (Vec<EdgeMark>, usize, usize) {
    pc_observational_cpdag_with_ci_test(observations, classes, false)
}

fn pc_exact_conditional_cpdag(
    observations: &[usize],
    classes: &MarkovClassCatalog,
) -> (Vec<EdgeMark>, usize, usize) {
    pc_observational_cpdag_with_ci_test(observations, classes, true)
}

fn pc_observational_cpdag_with_ci_test(
    observations: &[usize],
    classes: &MarkovClassCatalog,
    exact_conditional: bool,
) -> (Vec<EdgeMark>, usize, usize) {
    if observations.is_empty() {
        let marks = vec![EdgeMark::Undirected; UNORDERED_PAIRS.len()];
        return project_to_valid_cpdag(&marks, classes);
    }

    let mut adjacent_matrix = [[true; NODE_COUNT]; NODE_COUNT];
    for node in 0..NODE_COUNT {
        adjacent_matrix[node][node] = false;
    }
    let mut separating_sets: [[Option<usize>; NODE_COUNT]; NODE_COUNT] =
        [[None; NODE_COUNT]; NODE_COUNT];

    // PC level 0: remove unconditionally independent pairs.
    for (first, second) in UNORDERED_PAIRS {
        if pc_ci_test_independent(observations, first, second, None, exact_conditional) {
            adjacent_matrix[first][second] = false;
            adjacent_matrix[second][first] = false;
            separating_sets[first][second] = None;
            separating_sets[second][first] = None;
        }
    }

    // PC level 1: take one immutable adjacency snapshot for the entire level.
    // Removing an edge for one pair must not change the conditioning-set search
    // for another pair at the same level (which would make results order-dependent).
    let adjacency_at_level = adjacent_matrix;
    for (first, second) in UNORDERED_PAIRS {
        if !adjacency_at_level[first][second] {
            continue;
        }
        let third = (0..NODE_COUNT)
            .find(|node| *node != first && *node != second)
            .expect("three-node graph has one third variable");
        if adjacency_at_level[first][third] || adjacency_at_level[second][third] {
            if pc_ci_test_independent(observations, first, second, Some(third), exact_conditional) {
                adjacent_matrix[first][second] = false;
                adjacent_matrix[second][first] = false;
                separating_sets[first][second] = Some(third);
                separating_sets[second][first] = Some(third);
            }
        }
    }

    let mut raw_marks = vec![EdgeMark::Undirected; UNORDERED_PAIRS.len()];
    for (index, (first, second)) in UNORDERED_PAIRS.iter().copied().enumerate() {
        if !adjacent_matrix[first][second] {
            raw_marks[index] = EdgeMark::Absent;
        }
    }

    // For three observed variables, the PC orientation phase's only compelled
    // arrows are those entering an unshielded collider.
    for center in 0..NODE_COUNT {
        for left in 0..NODE_COUNT {
            for right in (left + 1)..NODE_COUNT {
                if left == center || right == center
                    || adjacent_matrix[left][right]
                    || !adjacent_matrix[left][center]
                    || !adjacent_matrix[right][center]
                {
                    continue;
                }
                if separating_sets[left][right] == Some(center) {
                    continue;
                }
                for parent in [left, right] {
                    let pair = (parent.min(center), parent.max(center));
                    if let Some(index) = UNORDERED_PAIRS.iter().position(|candidate| *candidate == pair) {
                        raw_marks[index] = if parent < center {
                            EdgeMark::Forward
                        } else {
                            EdgeMark::Reverse
                        };
                    }
                }
            }
        }
    }

    project_to_valid_cpdag(&raw_marks, classes)
}

/// Project finite-sample PC output onto the nearest valid CPDAG in the enumerated
/// catalogue. This tie-break is deterministic and prevents inconsistent edge marks.
fn project_to_valid_cpdag(
    raw_marks: &[EdgeMark],
    classes: &MarkovClassCatalog,
) -> (Vec<EdgeMark>, usize, usize) {
    let mut best_class_index = 0usize;
    let mut best_distance = f64::INFINITY;
    for (class_index, candidate_marks) in classes.cpdag_marks.iter().enumerate() {
        let distance = cpdag_metrics(raw_marks, candidate_marks).2;
        if distance < best_distance {
            best_distance = distance;
            best_class_index = class_index;
        }
    }
    let projected = classes.cpdag_marks[best_class_index].clone();
    let changed_marks = raw_marks.iter().zip(&projected)
        .filter(|(raw, projected)| raw != projected)
        .count();
    (projected, best_class_index, changed_marks)
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
    graph_variant_index: usize,
    family: ScmGraphFamily,
    mechanism_regime: ScmMechanismRegime,
    held_out: bool,
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
    pc_projection_mark_changes: Option<usize>,
    markov_equivalence_class_correct: bool,
    cpdag_claim_accepted: bool,
    false_cpdag_claim: bool,
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
        ScmGraphFamily::RandomMotifHeldOut => random_dag_mask(rng, 0.52),
        ScmGraphFamily::NullGraphNegativeControl => 0,
    };
    CausalDag { edge_mask: mask }
}

fn parameters_for_regime(regime: ScmMechanismRegime, rng: &mut BenchmarkRng) -> ScmParameters {
    // Mechanism parameters are sampled independently of graph-family identity.
    // This crosses topology and mechanism so the generator does not hard-code a
    // family-specific coefficient signature a policy could exploit.
    let (intercept, magnitude, jitter, signed) = match regime {
        ScmMechanismRegime::WeakPositiveDevelopment => (0.18, 0.34, 0.12, false),
        ScmMechanismRegime::StrongPositiveDevelopment => (0.12, 0.60, 0.035, false),
        ScmMechanismRegime::MixedSignHeldOut => (0.22, 0.43, 0.08, true),
        ScmMechanismRegime::NearNullHeldOut => (0.24, 0.075, 0.02, false),
    };
    let flip_signs = signed && rng.unit_f64() < 0.5;
    let mut edge_weights = [0.0; 6];
    for (index, weight) in edge_weights.iter_mut().enumerate() {
        let amplitude = (magnitude + (rng.unit_f64() - 0.5) * 2.0 * jitter).max(0.02);
        // Mixed-sign worlds always contain both positive and negative coefficients;
        // a seeded polarity flip varies which edge indices carry each sign.
        let sign = if signed && ((index % 2 == 0) ^ flip_signs) { -1.0 } else { 1.0 };
        *weight = sign * amplitude;
    }
    ScmParameters { intercept, edge_weights }
}

fn generate_worlds(seed: u64) -> Vec<ScmWorld> {
    let families = ScmGraphFamily::all();
    let mechanism_regimes = ScmMechanismRegime::all();
    let mut worlds = Vec::with_capacity(
        families.len() * WORLD_VARIANTS_PER_FAMILY * mechanism_regimes.len(),
    );
    for (family_index, family) in families.iter().copied().enumerate() {
        for graph_variant_index in 0..WORLD_VARIANTS_PER_FAMILY {
            // Draw each graph once; hold this exact topology fixed across all
            // mechanism regimes to create paired factorial comparisons.
            let topology_seed = seed
                ^ (family_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
                ^ (graph_variant_index as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
                ^ 0x4752_4150_485F_5345;
            let mut graph_rng = BenchmarkRng::new(topology_seed);
            let truth = graph_for_family(family, &mut graph_rng);

            for (regime_index, mechanism_regime) in mechanism_regimes.iter().copied().enumerate() {
                // Parameters are drawn from a disjoint stream, independently of
                // the number/pattern of random draws used to choose the topology.
                let parameter_seed = seed
                    ^ (graph_variant_index as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
                    ^ (regime_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ 0x4D45_4348_414E_4953;
                let mut parameter_rng = BenchmarkRng::new(parameter_seed);
                let parameters = parameters_for_regime(mechanism_regime, &mut parameter_rng);
                let held_out = family.held_out() || mechanism_regime.held_out();
                worlds.push(ScmWorld {
                    id: format!(
                        "{}-graph-{graph_variant_index:02}-{}",
                        family.as_str(),
                        mechanism_regime.as_str()
                    ),
                    graph_variant_index,
                    family,
                    mechanism_regime,
                    held_out,
                    truth,
                    parameters,
                });
            }
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
        .filter(|world| !world.held_out)
        .count();
    if development_count == 0 {
        return Err(ScmGraphBenchmarkError::MissingCandidate);
    }

    // Tune a single action on development graph/mechanism combinations only.
    // Truth graph IDs are not used to choose the action. Held-out worlds are not
    // passed to this function and therefore cannot affect this baseline.
    let mut totals: Vec<(String, f64)> = actions.iter()
        .map(|action| (action.id.clone(), 0.0))
        .collect();
    for world in development_worlds.iter().filter(|world| !world.held_out) {
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
        ScmGraphPolicy::PassiveOnly
        | ScmGraphPolicy::PcObservationalBaseline
        | ScmGraphPolicy::PcExactConditionalBaseline => {
            Some("observe".into())
        }
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
    // Floating-point cumulative mass may end a few ulps below one. Falling back
    // to the final *positive-mass* outcome prevents a hard intervention from
    // ever sampling an impossible state just because of rounding.
    distribution.iter().rposition(|probability| *probability > 0.0).unwrap_or(0)
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
    classes: &MarkovClassCatalog,
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
    let mut observed_states = Vec::with_capacity(measurement_budget);
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
        if matches!(policy, ScmGraphPolicy::PcObservationalBaseline | ScmGraphPolicy::PcExactConditionalBaseline) {
            observed_states.push(observed_outcome);
            measurements_used += 1;
            continue;
        }
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

    let (pc_class_index, pc_projection_mark_changes) =
        if matches!(policy, ScmGraphPolicy::PcObservationalBaseline | ScmGraphPolicy::PcExactConditionalBaseline) {
            let (_, class_index, projection_mark_changes) = if policy == ScmGraphPolicy::PcExactConditionalBaseline {
                pc_exact_conditional_cpdag(&observed_states, classes)
            } else {
                pc_observational_cpdag(&observed_states, classes)
            };
            // The PC baseline submits a CPDAG rather than claiming one fully oriented
            // DAG. Represent its class as a uniform distribution over member DAGs so
            // posterior-scored fields do not falsely retain the original uniform prior.
            posterior.iter_mut().for_each(|hypothesis| hypothesis.probability = 0.0);
            let members = &classes.members[class_index];
            let member_probability = 1.0 / members.len() as f64;
            for member in members {
                posterior[*member].probability = member_probability;
            }
            (Some(class_index), Some(projection_mark_changes))
        } else {
            (None, None)
        };

    let map_index = if policy == ScmGraphPolicy::StructureBlindRandomGraph {
        selection_rng.index(graphs.len())
    } else if let Some(class_index) = pc_class_index {
        classes.members[class_index][0]
    } else {
        posterior_top_index(&posterior)
    };
    let predicted = graphs[map_index];
    let truth = graphs[truth_index];
    let (skeleton_f1, directed_edge_f1, shd, exact_graph) = structure_metrics(predicted, truth);
    let (posterior_class_prediction, posterior_top_class_mass, _) =
        posterior_cpdag(&posterior, classes);
    let predicted_cpdag = if policy == ScmGraphPolicy::StructureBlindRandomGraph {
        classes.cpdag_marks[classes.class_for_graph[map_index]].clone()
    } else if let Some(class_index) = pc_class_index {
        classes.cpdag_marks[class_index].clone()
    } else {
        posterior_class_prediction
    };
    let truth_cpdag = classes.cpdag_marks[classes.class_for_graph[truth_index]].clone();
    let (cpdag_skeleton_f1, compelled_direction_f1, cpdag_mark_distance, markov_class_correct) =
        cpdag_metrics(&predicted_cpdag, &truth_cpdag);
    let posterior_true_markov_class = posterior_mass_on_class(&posterior, classes, truth_index);
    let max_probability = posterior[posterior_top_index(&posterior)].probability;
    let claim_accepted = !matches!(
        policy,
        ScmGraphPolicy::StructureBlindRandomGraph
            | ScmGraphPolicy::PcObservationalBaseline
            | ScmGraphPolicy::PcExactConditionalBaseline
    ) && max_probability >= posterior_threshold;
    let false_graph_claim = claim_accepted && !exact_graph;
    let cpdag_claim_accepted = matches!(policy, ScmGraphPolicy::PcObservationalBaseline | ScmGraphPolicy::PcExactConditionalBaseline)
        || (policy != ScmGraphPolicy::StructureBlindRandomGraph
            && posterior_top_class_mass >= posterior_threshold);
    let false_cpdag_claim = cpdag_claim_accepted && !markov_class_correct;
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
        pc_projection_mark_changes,
        markov_equivalence_class_correct: markov_class_correct,
        cpdag_claim_accepted,
        false_cpdag_claim,
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
    let pc_projection_changes: Vec<f64> = trials.iter()
        .filter_map(|trial| trial.pc_projection_mark_changes.map(|count| count as f64))
        .collect();
    let pc_projection_changed_trials = trials.iter()
        .filter(|trial| trial.pc_projection_mark_changes.is_some_and(|count| count > 0))
        .count();
    let cpdag_correct = trials.iter()
        .filter(|trial| trial.markov_equivalence_class_correct).count();
    let cpdag_claims = trials.iter().filter(|trial| trial.cpdag_claim_accepted).count();
    let false_cpdag_claims = trials.iter().filter(|trial| trial.false_cpdag_claim).count();
    let accepted_false_cpdag_claims = trials.iter()
        .filter(|trial| trial.cpdag_claim_accepted && trial.false_cpdag_claim).count();
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
        graph_variant_index: world.graph_variant_index,
        truth_graph_id: world.truth.id(),
        truth_edge_count: world.truth.edge_count(),
        family: world.family,
        mechanism_regime: world.mechanism_regime,
        held_out_family: world.family.held_out(),
        held_out_world: world.held_out,
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
        pc_raw_to_projected_mark_changes_mean: (!pc_projection_changes.is_empty())
            .then(|| mean(&pc_projection_changes)),
        pc_projection_changed_trial_rate: (!pc_projection_changes.is_empty())
            .then_some(pc_projection_changed_trials as f64 / pc_projection_changes.len() as f64),
        markov_equivalence_class_recovery_rate: cpdag_correct as f64 / count as f64,
        markov_equivalence_class_recovery_wilson_ci95: wilson_ci95(cpdag_correct, count),
        cpdag_claim_coverage: cpdag_claims as f64 / count as f64,
        cpdag_claim_coverage_wilson_ci95: wilson_ci95(cpdag_claims, count),
        false_cpdag_claim_rate_among_accepted: (cpdag_claims > 0)
            .then_some(accepted_false_cpdag_claims as f64 / cpdag_claims as f64),
        false_cpdag_claim_rate_among_accepted_wilson_ci95: (cpdag_claims > 0)
            .then_some(wilson_ci95(accepted_false_cpdag_claims, cpdag_claims)),
        false_cpdag_claim_rate_all_trials: false_cpdag_claims as f64 / count as f64,
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

fn topology_instance_means(
    rows: &[&ScmGraphPolicySummary],
    metric: fn(&ScmGraphPolicySummary) -> f64,
) -> Vec<f64> {
    let mut keys: Vec<(ScmGraphFamily, usize)> = Vec::new();
    for row in rows {
        let key = (row.family, row.graph_variant_index);
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys.iter().map(|(family, variant)| {
        let paired_regime_scores: Vec<f64> = rows.iter()
            .filter(|row| row.family == *family && row.graph_variant_index == *variant)
            .map(|row| metric(row))
            .collect();
        mean(&paired_regime_scores)
    }).collect()
}

fn aggregate_factorized_shifts(
    summaries: &[ScmGraphPolicySummary],
    policies: &[ScmGraphPolicy],
) -> Vec<ScmGraphFactorizedShiftAggregate> {
    let mut aggregates = Vec::with_capacity(policies.len() * ScmGraphShiftSplit::all().len());
    for policy in policies.iter().copied() {
        let policy_rows: Vec<&ScmGraphPolicySummary> = summaries.iter()
            .filter(|row| row.policy == policy)
            .collect();
        for split in ScmGraphShiftSplit::all() {
            let rows: Vec<&ScmGraphPolicySummary> = policy_rows.iter()
                .copied()
                .filter(|row| split.includes(row))
                .collect();
            let skeleton = topology_instance_means(&rows, |row| row.mean_skeleton_f1);
            let directed = topology_instance_means(&rows, |row| row.mean_directed_edge_f1);
            let cpdag_skeleton = topology_instance_means(&rows, |row| row.mean_cpdag_skeleton_f1);
            let shd = topology_instance_means(&rows, |row| row.mean_structural_hamming_distance);
            let exact = topology_instance_means(&rows, |row| row.exact_graph_recovery_rate);
            let class_recovery = topology_instance_means(
                &rows, |row| row.markov_equivalence_class_recovery_rate,
            );
            let graph_coverage = topology_instance_means(&rows, |row| row.graph_claim_coverage);
            let graph_false_claims = topology_instance_means(
                &rows, |row| row.false_graph_claim_rate_all_trials,
            );
            let cpdag_coverage = topology_instance_means(&rows, |row| row.cpdag_claim_coverage);
            let cpdag_false_claims = topology_instance_means(
                &rows, |row| row.false_cpdag_claim_rate_all_trials,
            );
            aggregates.push(ScmGraphFactorizedShiftAggregate {
                policy,
                split,
                world_rows: rows.len(),
                topology_instances: skeleton.len(),
                mean_skeleton_f1: mean(&skeleton),
                skeleton_f1_topology_instance_ci95: mean_ci95(&skeleton, 0.0, 1.0),
                mean_directed_edge_f1: mean(&directed),
                directed_edge_f1_topology_instance_ci95: mean_ci95(&directed, 0.0, 1.0),
                mean_cpdag_skeleton_f1: mean(&cpdag_skeleton),
                cpdag_skeleton_f1_topology_instance_ci95: mean_ci95(&cpdag_skeleton, 0.0, 1.0),
                mean_structural_hamming_distance: mean(&shd),
                shd_topology_instance_ci95: mean_ci95(
                    &shd, 0.0, (NODE_COUNT * (NODE_COUNT - 1) / 2) as f64,
                ),
                mean_exact_graph_recovery_rate: mean(&exact),
                mean_markov_equivalence_class_recovery_rate: mean(&class_recovery),
                mean_full_graph_claim_coverage: mean(&graph_coverage),
                mean_false_full_graph_claim_rate: mean(&graph_false_claims),
                mean_cpdag_claim_coverage: mean(&cpdag_coverage),
                mean_false_cpdag_claim_rate: mean(&cpdag_false_claims),
                worst_topology_directed_edge_f1: directed.iter().copied().fold(1.0, f64::min),
                worst_topology_shd: shd.iter().copied().fold(0.0, f64::max),
            });
        }
    }
    aggregates
}

fn paired_topology_delta_means(
    summaries: &[ScmGraphPolicySummary],
    policy: ScmGraphPolicy,
    comparator: ScmGraphPolicy,
    split: ScmGraphShiftSplit,
    metric: fn(&ScmGraphPolicySummary) -> f64,
) -> (usize, Vec<f64>) {
    let target_rows: Vec<&ScmGraphPolicySummary> = summaries.iter()
        .filter(|row| row.policy == policy && split.includes(row))
        .collect();
    let mut topology_keys: Vec<(ScmGraphFamily, usize)> = Vec::new();
    for row in &target_rows {
        let key = (row.family, row.graph_variant_index);
        if !topology_keys.contains(&key) {
            topology_keys.push(key);
        }
    }

    let per_topology_deltas = topology_keys.iter().filter_map(|(family, variant)| {
        let paired_rows: Vec<f64> = target_rows.iter()
            .filter(|target| target.family == *family && target.graph_variant_index == *variant)
            .filter_map(|target| {
                let baseline = summaries.iter().find(|row| {
                    row.world_id == target.world_id && row.policy == comparator
                })?;
                Some(metric(target) - metric(baseline))
            })
            .collect();
        (!paired_rows.is_empty()).then(|| mean(&paired_rows))
    }).collect();
    (target_rows.len(), per_topology_deltas)
}

fn aggregate_paired_policy_deltas(
    summaries: &[ScmGraphPolicySummary],
    policies: &[ScmGraphPolicy],
) -> Vec<ScmGraphPairedPolicyDelta> {
    let comparators = [
        ScmGraphPolicy::NominalExpectedInformationGain,
        ScmGraphPolicy::DevelopmentFixedDesign,
        ScmGraphPolicy::UniformRandomAction,
        ScmGraphPolicy::PcObservationalBaseline,
        ScmGraphPolicy::PcExactConditionalBaseline,
    ];
    let mut output = Vec::new();
    for policy in policies.iter().copied() {
        for comparator in comparators {
            if policy == comparator {
                continue;
            }
            for split in ScmGraphShiftSplit::all() {
                let (paired_world_rows, directed) = paired_topology_delta_means(
                    summaries, policy, comparator, split, |row| row.mean_directed_edge_f1,
                );
                let (_, cpdag_skeleton) = paired_topology_delta_means(
                    summaries, policy, comparator, split, |row| row.mean_cpdag_skeleton_f1,
                );
                let (_, shd) = paired_topology_delta_means(
                    summaries, policy, comparator, split,
                    |row| row.mean_structural_hamming_distance,
                );
                let (_, measurements) = paired_topology_delta_means(
                    summaries, policy, comparator, split, |row| row.mean_measurements_used,
                );
                let (_, interventions) = paired_topology_delta_means(
                    summaries, policy, comparator, split, |row| row.mean_interventions_used,
                );
                let (_, false_claims) = paired_topology_delta_means(
                    summaries, policy, comparator, split,
                    |row| row.false_graph_claim_rate_all_trials,
                );
                output.push(ScmGraphPairedPolicyDelta {
                    policy,
                    comparator,
                    split,
                    paired_world_rows,
                    paired_topology_instances: directed.len(),
                    delta_directed_edge_f1: mean(&directed),
                    delta_directed_edge_f1_topology_instance_ci95: mean_ci95(&directed, -1.0, 1.0),
                    delta_cpdag_skeleton_f1: mean(&cpdag_skeleton),
                    delta_cpdag_skeleton_f1_topology_instance_ci95: mean_ci95(&cpdag_skeleton, -1.0, 1.0),
                    delta_structural_hamming_distance: mean(&shd),
                    delta_shd_topology_instance_ci95: mean_ci95(
                        &shd, -(NODE_COUNT * (NODE_COUNT - 1) / 2) as f64,
                        (NODE_COUNT * (NODE_COUNT - 1) / 2) as f64,
                    ),
                    delta_measurements_used: mean(&measurements),
                    delta_measurements_topology_instance_ci95: mean_ci95(&measurements, -64.0, 64.0),
                    delta_interventions_used: mean(&interventions),
                    delta_interventions_topology_instance_ci95: mean_ci95(&interventions, -64.0, 64.0),
                    delta_false_graph_claim_rate: mean(&false_claims),
                    delta_false_graph_claim_rate_topology_instance_ci95: mean_ci95(&false_claims, -1.0, 1.0),
                });
            }
        }
    }
    output
}

fn aggregate_held_out(
    summaries: &[ScmGraphPolicySummary],
    policies: &[ScmGraphPolicy],
) -> Vec<ScmGraphPolicyAggregate> {
    policies.iter().copied().map(|policy| {
        let rows: Vec<&ScmGraphPolicySummary> = summaries.iter()
            .filter(|row| row.policy == policy && row.held_out_world)
            .collect();
        // First average over held-out mechanism regimes within each exact topology.
        // The resulting uncertainty unit is one topology instance, not one regime row.
        let skeleton = topology_instance_means(&rows, |row| row.mean_skeleton_f1);
        let directed = topology_instance_means(&rows, |row| row.mean_directed_edge_f1);
        let cpdag_skeleton = topology_instance_means(&rows, |row| row.mean_cpdag_skeleton_f1);
        let compelled = topology_instance_means(&rows, |row| row.mean_compelled_direction_f1);
        let cpdag_distance = topology_instance_means(&rows, |row| row.mean_cpdag_mark_distance);
        let cpdag_accuracy = topology_instance_means(&rows, |row| row.markov_equivalence_class_recovery_rate);
        let cpdag_coverage = topology_instance_means(&rows, |row| row.cpdag_claim_coverage);
        let cpdag_false_claims = topology_instance_means(&rows, |row| row.false_cpdag_claim_rate_all_trials);
        let shd = topology_instance_means(&rows, |row| row.mean_structural_hamming_distance);
        let exact = topology_instance_means(&rows, |row| row.exact_graph_recovery_rate);
        let coverage = topology_instance_means(&rows, |row| row.graph_claim_coverage);
        let false_claims = topology_instance_means(&rows, |row| row.false_graph_claim_rate_all_trials);
        ScmGraphPolicyAggregate {
            policy,
            held_out_worlds: rows.len(),
            held_out_topology_instances: skeleton.len(),
            mean_held_out_skeleton_f1: mean(&skeleton),
            held_out_skeleton_f1_topology_instance_ci95: mean_ci95(&skeleton, 0.0, 1.0),
            mean_held_out_directed_edge_f1: mean(&directed),
            held_out_directed_edge_f1_topology_instance_ci95: mean_ci95(&directed, 0.0, 1.0),
            mean_held_out_cpdag_skeleton_f1: mean(&cpdag_skeleton),
            held_out_cpdag_skeleton_f1_topology_instance_ci95: mean_ci95(&cpdag_skeleton, 0.0, 1.0),
            mean_held_out_compelled_direction_f1: mean(&compelled),
            held_out_compelled_direction_f1_topology_instance_ci95: mean_ci95(&compelled, 0.0, 1.0),
            mean_held_out_cpdag_mark_distance: mean(&cpdag_distance),
            held_out_cpdag_mark_distance_topology_instance_ci95: mean_ci95(&cpdag_distance, 0.0, (NODE_COUNT * (NODE_COUNT - 1) / 2) as f64),
            held_out_markov_equivalence_class_recovery_rate: mean(&cpdag_accuracy),
            mean_held_out_cpdag_claim_coverage: mean(&cpdag_coverage),
            mean_held_out_false_cpdag_claim_rate_all_trials: mean(&cpdag_false_claims),
            mean_held_out_shd: mean(&shd),
            held_out_shd_topology_instance_ci95: mean_ci95(&shd, 0.0, (NODE_COUNT * (NODE_COUNT - 1) / 2) as f64),
            worst_world_directed_edge_f1: rows.iter()
                .map(|row| row.mean_directed_edge_f1).fold(1.0, f64::min),
            worst_world_shd: rows.iter()
                .map(|row| row.mean_structural_hamming_distance).fold(0.0, f64::max),
            mean_held_out_exact_graph_recovery_rate: mean(&exact),
            mean_held_out_claim_coverage: mean(&coverage),
            mean_held_out_false_graph_claim_rate_all_trials: mean(&false_claims),
        }
    }).collect()
}

/// Run a deterministic graph-recovery experiment under equal measurement budgets.
///
/// Candidate hypotheses are all 25 labeled DAGs over three observed variables.
/// Seven graph-family templates each generate four graph variants. Every exact
/// topology is crossed with four distinct mechanism regimes, giving 112 worlds.
/// Two graph families and two mechanism regimes are development-only; a world is
/// held out if either factor is held out (16 development / 96 held-out worlds).
/// Topology and mechanism use separate RNG streams. The policy never receives the
/// hidden truth graph or hidden structural-equation parameters.
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
    let classes = build_markov_class_catalog(&graphs);
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
                    &classes,
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
        schema_version: 6,
        generator_version: GENERATOR_VERSION.into(),
        seed,
        trials_per_world,
        measurement_budget,
        posterior_threshold,
        observed_node_count: NODE_COUNT,
        candidate_dag_count: graphs.len(),
        markov_equivalence_class_count: classes.members.len(),
        world_variants_per_family: WORLD_VARIANTS_PER_FAMILY,
        mechanism_regime_count: ScmMechanismRegime::all().len(),
        world_count: worlds.len(),
        development_world_count: worlds.iter().filter(|world| !world.held_out).count(),
        held_out_world_count: worlds.iter().filter(|world| world.held_out).count(),
        declared_model_scenarios: scenarios.iter()
            .map(|scenario| scenario.scenario_id.clone()).collect(),
        fixed_design_action_id: fixed_action_id,
        held_out_policy_aggregates: aggregate_held_out(&summaries, &policies),
        factorized_shift_aggregates: aggregate_factorized_shifts(&summaries, &policies),
        paired_policy_deltas: aggregate_paired_policy_deltas(&summaries, &policies),
        observational_pc_baseline: ScmGraphPolicy::PcObservationalBaseline,
        exact_observational_pc_baseline: ScmGraphPolicy::PcExactConditionalBaseline,
        worlds: summaries,
        interpretation_boundary: concat!(
            "Synthetic software benchmark only. The candidate set enumerates all labeled DAGs ",
            "on three observed variables, but outcomes use hidden structural-equation parameters. ",
            "Held-out regimes are source-visible and are not cryptographically sealed. The ",
            "held-out intervals average paired mechanism regimes within topology, but are conditional ",
            "on seven hand-authored graph-family templates; they do not support population-wide SCM generalization. The ",
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
        assert_eq!(build_markov_class_catalog(&graphs).members.len(), 11);
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
    fn sampled_intervention_outcomes_respect_hard_do_assignments() {
        for node in 0..NODE_COUNT {
            for value in 0..=1 {
                let action = ScmAction {
                    id: format!("do_{node}_{value}"),
                    kind: ActionKind::Intervene { node, value },
                    cost: 1.0,
                };
                let distribution = outcome_distribution(
                    CausalDag { edge_mask: edge_mask_with(&[(0, 1), (2, 1)]) },
                    &action,
                    params_alternative(),
                );
                for seed in 0..256_u64 {
                    let mut rng = BenchmarkRng::new(seed);
                    let state = sample_outcome(&distribution, &mut rng);
                    assert_eq!((state >> node) & 1, value);
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
    fn pc_g_squared_baseline_recovers_independence_and_an_unshielded_collider() {
        let catalogue = all_dags();
        let classes = build_markov_class_catalog(&catalogue);

        // Exactly balanced observations make all three variables independent.
        let mut independent = Vec::new();
        for state in 0..OUTCOME_COUNT {
            independent.extend(std::iter::repeat_n(state, 40));
        }
        for (first, second) in UNORDERED_PAIRS {
            assert!(g_squared_ci_statistic(&independent, first, second, None) < 1.0e-10);
        }
        let (empty_marks, empty_class, empty_projection_changes) =
            pc_observational_cpdag(&independent, &classes);
        assert_eq!(empty_projection_changes, 0);
        assert_eq!(empty_marks, vec![EdgeMark::Absent; UNORDERED_PAIRS.len()]);
        assert_eq!(
            empty_marks,
            classes.cpdag_marks[empty_class],
            "PC output must be projected to a valid catalogue CPDAG"
        );

        // Build a balanced collider distribution: X and Y are independent roots,
        // while Z has P(Z=1)=0.2 when X=Y=0 and 0.8 otherwise.
        // The exact finite sample uses counts that realize those probabilities.
        let mut collider_observations = Vec::new();
        for x in 0..=1_usize {
            for y in 0..=1_usize {
                let z_ones = if x == 0 && y == 0 { 20 } else { 80 };
                let z_zeros = 100 - z_ones;
                let state_zero = x | (y << 2);
                let state_one = state_zero | (1 << 1);
                collider_observations.extend(std::iter::repeat_n(state_zero, z_zeros));
                collider_observations.extend(std::iter::repeat_n(state_one, z_ones));
            }
        }
        assert!(
            g_squared_ci_statistic(&collider_observations, 0, 2, None)
                < PC_G2_CRITICAL_DF1_95,
            "collider parents should be marginally independent"
        );
        assert!(
            g_squared_ci_statistic(&collider_observations, 0, 1, None)
                > PC_G2_CRITICAL_DF1_95,
            "parent and collider child should be dependent"
        );
        let (collider_marks, collider_class, collider_projection_changes) =
            pc_observational_cpdag(&collider_observations, &classes);
        assert_eq!(collider_projection_changes, 0);
        assert_eq!(
            collider_marks,
            vec![EdgeMark::Forward, EdgeMark::Absent, EdgeMark::Reverse]
        );
        assert_eq!(collider_marks, classes.cpdag_marks[collider_class]);
    }

    #[test]
    fn cpdag_projection_audit_counts_raw_mark_changes() {
        let catalogue = all_dags();
        let classes = build_markov_class_catalog(&catalogue);

        // A complete triangle has no unshielded collider, so the valid CPDAG
        // leaves its three edges undirected. Three directed marks are therefore
        // a useful known-answer fixture for the corrective projection step.
        let raw_cycle_marks = vec![EdgeMark::Forward; UNORDERED_PAIRS.len()];
        let (projected, class_index, changed_marks) =
            project_to_valid_cpdag(&raw_cycle_marks, &classes);
        assert_eq!(projected, classes.cpdag_marks[class_index]);
        assert_eq!(changed_marks, raw_cycle_marks.iter().zip(&projected)
            .filter(|(raw, projected)| raw != projected).count());
        assert!(changed_marks > 0);
    }

    #[test]
    fn exact_conditional_g2_is_conservative_on_balanced_null_and_detects_large_dependence() {
        let independent: Vec<usize> = (0..OUTCOME_COUNT)
            .flat_map(|state| std::iter::repeat_n(state, 20))
            .collect();
        assert!(
            exact_conditional_g_squared_p_value(&independent, 0, 1, None) > 0.99,
            "balanced independent variables should have a null p-value near one"
        );
        assert!(
            exact_conditional_g_squared_p_value(&independent, 0, 1, Some(2)) > 0.99,
            "balanced strata should retain conditional independence"
        );

        let mut dependent = Vec::new();
        for x in 0..=1_usize {
            for y in 0..=1_usize {
                let copies = if x == y { 100 } else { 5 };
                let state = x | (y << 1);
                dependent.extend(std::iter::repeat_n(state, copies));
            }
        }
        let p_value = exact_conditional_g_squared_p_value(&dependent, 0, 1, None);
        assert!((0.0..=1.0).contains(&p_value));
        assert!(p_value < 0.05, "large binary dependence should be rejected");
    }

    #[test]
    fn exact_conditional_pc_recovers_known_unshielded_collider() {
        let catalogue = all_dags();
        let classes = build_markov_class_catalog(&catalogue);
        let mut collider_observations = Vec::new();
        for x in 0..=1_usize {
            for y in 0..=1_usize {
                let z_ones = if x == 0 && y == 0 { 20 } else { 80 };
                let z_zeros = 100 - z_ones;
                let state_zero = x | (y << 2);
                let state_one = state_zero | (1 << 1);
                collider_observations.extend(std::iter::repeat_n(state_zero, z_zeros));
                collider_observations.extend(std::iter::repeat_n(state_one, z_ones));
            }
        }
        assert!(
            exact_conditional_g_squared_p_value(
                &collider_observations, 0, 2, Some(1),
            ) < 0.05,
            "conditioning on the collider should induce dependence between its parents"
        );
        let (marks, class_index, changed_marks) =
            pc_exact_conditional_cpdag(&collider_observations, &classes);
        assert_eq!(marks, vec![EdgeMark::Forward, EdgeMark::Absent, EdgeMark::Reverse]);
        assert_eq!(marks, classes.cpdag_marks[class_index]);
        assert_eq!(changed_marks, 0);
    }

    #[test]
    fn cpdag_metrics_do_not_penalize_unidentifiable_chain_directions() {
        let catalogue = all_dags();
        let chain_forward = CausalDag {
            edge_mask: edge_mask_with(&[(0, 1), (1, 2)]),
        };
        let chain_reverse = CausalDag {
            edge_mask: edge_mask_with(&[(1, 0), (2, 1)]),
        };
        let collider = CausalDag {
            edge_mask: edge_mask_with(&[(0, 1), (2, 1)]),
        };
        assert!(markov_equivalent(chain_forward, chain_reverse));
        assert!(!markov_equivalent(chain_forward, collider));

        let chain_marks = cpdag_marks(chain_forward, &catalogue);
        assert_eq!(
            chain_marks,
            vec![EdgeMark::Undirected, EdgeMark::Absent, EdgeMark::Undirected]
        );
        let collider_marks = cpdag_marks(collider, &catalogue);
        assert_eq!(
            collider_marks,
            vec![EdgeMark::Forward, EdgeMark::Absent, EdgeMark::Reverse]
        );

        let (skeleton, compelled, distance, same_class) =
            cpdag_metrics(&chain_marks, &cpdag_marks(chain_reverse, &catalogue));
        assert_eq!(skeleton, 1.0);
        assert_eq!(compelled, 1.0);
        assert_eq!(distance, 0.0);
        assert!(same_class);

        let (skeleton, compelled, distance, same_class) =
            cpdag_metrics(&chain_marks, &collider_marks);
        assert_eq!(skeleton, 1.0);
        assert_eq!(compelled, 0.0);
        assert_eq!(distance, 2.0);
        assert!(!same_class);
    }

    #[test]
    fn posterior_cpdag_sums_probability_over_markov_equivalent_dags() {
        let catalogue = all_dags();
        let chain_forward = CausalDag {
            edge_mask: edge_mask_with(&[(0, 1), (1, 2)]),
        };
        let chain_reverse = CausalDag {
            edge_mask: edge_mask_with(&[(1, 0), (2, 1)]),
        };
        let mut posterior = uniform_priors(&catalogue);
        let forward_index = catalogue.iter().position(|graph| *graph == chain_forward).unwrap();
        let reverse_index = catalogue.iter().position(|graph| *graph == chain_reverse).unwrap();
        posterior[forward_index].probability = 0.3;
        posterior[reverse_index].probability = 0.3;
        let remainder = (1.0 - 0.6) / (catalogue.len() - 2) as f64;
        for (index, hypothesis) in posterior.iter_mut().enumerate() {
            if index != forward_index && index != reverse_index {
                hypothesis.probability = remainder;
            }
        }
        let classes = build_markov_class_catalog(&catalogue);
        let (marks, class_mass, class_index) = posterior_cpdag(&posterior, &classes);
        assert_eq!(marks, cpdag_marks(chain_forward, &catalogue));
        assert!((class_mass - (0.6 + remainder)).abs() < 1.0e-12);
        assert_eq!(classes.members[class_index].len(), 3);
    }

    #[test]
    fn worlds_replay_for_a_seed_and_cover_held_out_families() {
        let first = generate_worlds(44);
        let replay = generate_worlds(44);
        let changed = generate_worlds(45);
        assert_eq!(first.len(), 112);
        assert_eq!(first.len(), replay.len());
        assert_eq!(first.iter().map(|world| world.truth).collect::<Vec<_>>(),
            replay.iter().map(|world| world.truth).collect::<Vec<_>>());
        assert_eq!(first.iter().map(|world| world.parameters).collect::<Vec<_>>(),
            replay.iter().map(|world| world.parameters).collect::<Vec<_>>());
        assert_ne!(first.iter().map(|world| world.truth).collect::<Vec<_>>(),
            changed.iter().map(|world| world.truth).collect::<Vec<_>>());
        assert_eq!(first.iter().filter(|world| world.family.held_out()).count(), 80);
        assert_eq!(first.iter().filter(|world| world.held_out).count(), 96);
        assert_eq!(first.iter().filter(|world| !world.held_out).count(), 16);
        assert!(first.iter().filter(|world| {
            world.family == ScmGraphFamily::NullGraphNegativeControl
        }).all(|world| world.truth.edge_count() == 0));
    }

    #[test]
    fn graph_families_are_crossed_with_independent_mechanism_regimes() {
        let worlds = generate_worlds(90210);
        for family in ScmGraphFamily::all() {
            let family_worlds: Vec<&ScmWorld> = worlds.iter()
                .filter(|world| world.family == family)
                .collect();
            assert_eq!(
                family_worlds.len(),
                WORLD_VARIANTS_PER_FAMILY * ScmMechanismRegime::all().len()
            );
            for graph_variant_index in 0..WORLD_VARIANTS_PER_FAMILY {
                let paired: Vec<&ScmWorld> = family_worlds.iter()
                    .copied()
                    .filter(|world| world.graph_variant_index == graph_variant_index)
                    .collect();
                assert_eq!(paired.len(), ScmMechanismRegime::all().len());
                for regime in ScmMechanismRegime::all() {
                    assert_eq!(
                        paired.iter().filter(|world| world.mechanism_regime == regime).count(),
                        1,
                        "each exact topology must appear once with every mechanism regime"
                    );
                }
                assert!(paired.iter().all(|world| world.truth == paired[0].truth));
            }
        }

        for graph_variant_index in 0..WORLD_VARIANTS_PER_FAMILY {
            for regime in ScmMechanismRegime::all() {
                let paired: Vec<&ScmWorld> = worlds.iter()
                    .filter(|world| {
                        world.graph_variant_index == graph_variant_index
                            && world.mechanism_regime == regime
                    })
                    .collect();
                assert_eq!(paired.len(), ScmGraphFamily::all().len());
                assert!(paired.iter()
                    .all(|world| world.parameters == paired[0].parameters));
            }
        }

        for world in &worlds {
            assert_eq!(
                world.held_out,
                world.family.held_out() || world.mechanism_regime.held_out()
            );
            match world.mechanism_regime {
                ScmMechanismRegime::WeakPositiveDevelopment
                | ScmMechanismRegime::StrongPositiveDevelopment => {
                    assert!(world.parameters.edge_weights.iter().all(|weight| *weight > 0.0));
                }
                ScmMechanismRegime::MixedSignHeldOut => {
                    assert!(world.parameters.edge_weights.iter().any(|weight| *weight < 0.0));
                    assert!(world.parameters.edge_weights.iter().any(|weight| *weight > 0.0));
                }
                ScmMechanismRegime::NearNullHeldOut => {
                    assert!(world.parameters.edge_weights.iter()
                        .all(|weight| weight.abs() <= 0.095 + 1.0e-12));
                }
            }
        }
    }

    #[test]
    fn development_fixed_action_ignores_held_out_worlds() {
        let graphs = all_dags();
        let actions = actions();
        let priors = uniform_priors(&graphs);
        let worlds = generate_worlds(42);
        let chosen_all = fixed_design_action(&graphs, &actions, &priors, &worlds).unwrap();
        let dev_only: Vec<ScmWorld> = worlds.iter()
            .filter(|world| !world.held_out)
            .cloned()
            .collect();
        let chosen_dev = fixed_design_action(&graphs, &actions, &priors, &dev_only).unwrap();
        assert_eq!(chosen_all, chosen_dev);
        assert!(!chosen_all.is_empty());
    }

    #[test]
    fn run_is_deterministic_and_reports_world_level_aggregates() {
        let a = run_scm_graph_benchmark(3, 2, 123, 0.95).unwrap();
        let b = run_scm_graph_benchmark(3, 2, 123, 0.95).unwrap();
        assert_eq!(a, b);
        assert_eq!(serde_json::to_vec(&a).unwrap(), serde_json::to_vec(&b).unwrap());
        assert_eq!(a.schema_version, 6);
        assert_eq!(a.generator_version, "sensebench-scm-graphs-0.6.0");
        assert_eq!(ScmGraphPolicy::all().len(), 11);
        assert_eq!(a.observational_pc_baseline, ScmGraphPolicy::PcObservationalBaseline);
        assert_eq!(a.exact_observational_pc_baseline, ScmGraphPolicy::PcExactConditionalBaseline);
        for row in &a.worlds {
            let is_pc = matches!(
                row.policy,
                ScmGraphPolicy::PcObservationalBaseline | ScmGraphPolicy::PcExactConditionalBaseline
            );
            assert_eq!(row.pc_raw_to_projected_mark_changes_mean.is_some(), is_pc);
            assert_eq!(row.pc_projection_changed_trial_rate.is_some(), is_pc);
        }
        assert_eq!(a.world_count, 112);
        assert_eq!(a.development_world_count, 16);
        assert_eq!(a.held_out_world_count, 96);
        assert_eq!(a.mechanism_regime_count, 4);
        assert_eq!(a.candidate_dag_count, 25);
        assert_eq!(a.markov_equivalence_class_count, 11);
        assert_eq!(a.worlds.len(), 112 * ScmGraphPolicy::all().len());
        assert_eq!(a.held_out_policy_aggregates.len(), ScmGraphPolicy::all().len());
        assert_eq!(
            a.factorized_shift_aggregates.len(),
            ScmGraphPolicy::all().len() * ScmGraphShiftSplit::all().len()
        );
    }

    #[test]
    fn factorized_shift_splits_have_expected_world_and_topology_counts() {
        let report = run_scm_graph_benchmark(3, 1, 2026, 0.90).unwrap();
        for policy in ScmGraphPolicy::all() {
            let find = |split| report.factorized_shift_aggregates.iter()
                .find(|row| row.policy == policy && row.split == split).unwrap();
            let any = find(ScmGraphShiftSplit::AnyHeldOut);
            let topology = find(ScmGraphShiftSplit::TopologyShiftOnly);
            let mechanism = find(ScmGraphShiftSplit::MechanismShiftOnly);
            let joint = find(ScmGraphShiftSplit::JointShift);

            assert_eq!((any.world_rows, any.topology_instances), (96, 28));
            assert_eq!((topology.world_rows, topology.topology_instances), (40, 20));
            assert_eq!((mechanism.world_rows, mechanism.topology_instances), (16, 8));
            assert_eq!((joint.world_rows, joint.topology_instances), (40, 20));
            assert_eq!(
                topology.world_rows + mechanism.world_rows + joint.world_rows,
                any.world_rows,
                "the three factorized held-out partitions should be disjoint and exhaustive"
            );
            for split in [any, topology, mechanism, joint] {
                assert!((0.0..=1.0).contains(&split.mean_directed_edge_f1));
                assert!((0.0..=3.0).contains(&split.mean_structural_hamming_distance));
            }
        }
    }

    #[test]
    fn paired_deltas_use_the_same_worlds_and_topology_clustered_intervals() {
        let report = run_scm_graph_benchmark(3, 1, 5050, 0.90).unwrap();
        assert_eq!(
            report.paired_policy_deltas.len(),
            (ScmGraphPolicy::all().len() * 5 - 5)
                * ScmGraphShiftSplit::all().len()
        );
        for delta in &report.paired_policy_deltas {
            assert_ne!(delta.policy, delta.comparator);
            let expected_rows = match delta.split {
                ScmGraphShiftSplit::AnyHeldOut => 96,
                ScmGraphShiftSplit::TopologyShiftOnly => 40,
                ScmGraphShiftSplit::MechanismShiftOnly => 16,
                ScmGraphShiftSplit::JointShift => 40,
            };
            let expected_topologies = match delta.split {
                ScmGraphShiftSplit::AnyHeldOut => 28,
                ScmGraphShiftSplit::TopologyShiftOnly => 20,
                ScmGraphShiftSplit::MechanismShiftOnly => 8,
                ScmGraphShiftSplit::JointShift => 20,
            };
            assert_eq!(delta.paired_world_rows, expected_rows);
            assert_eq!(delta.paired_topology_instances, expected_topologies);
            assert!((-1.0..=1.0).contains(&delta.delta_directed_edge_f1));
            assert!((-1.0..=1.0).contains(&delta.delta_cpdag_skeleton_f1));
            assert!((-3.0..=3.0).contains(&delta.delta_structural_hamming_distance));
            assert!((-64.0..=64.0).contains(&delta.delta_measurements_used));
            assert!((-64.0..=64.0).contains(&delta.delta_interventions_used));
            assert!((-1.0..=1.0).contains(&delta.delta_false_graph_claim_rate));
        }
    }

    #[test]
    fn held_out_claim_metrics_are_bounded_and_null_false_claims_are_visible() {
        let report = run_scm_graph_benchmark(6, 2, 0x5EED, 0.90).unwrap();
        for row in report.worlds.iter()
            .filter(|row| matches!(row.policy, ScmGraphPolicy::PcObservationalBaseline | ScmGraphPolicy::PcExactConditionalBaseline))
        {
            assert_eq!(row.mean_measurements_used, 2.0);
            assert_eq!(row.mean_interventions_used, 0.0);
            assert_eq!(row.graph_claim_coverage, 0.0);
            assert_eq!(row.cpdag_claim_coverage, 1.0);
            assert!(row.pc_raw_to_projected_mark_changes_mean.is_some());
            assert!(row.pc_projection_changed_trial_rate.is_some());
        }
        for row in &report.worlds {
            assert!((0.0..=1.0).contains(&row.mean_skeleton_f1));
            assert!((0.0..=1.0).contains(&row.mean_directed_edge_f1));
            assert!((0.0..=1.0).contains(&row.mean_cpdag_skeleton_f1));
            assert!((0.0..=1.0).contains(&row.mean_compelled_direction_f1));
            assert!((0.0..=3.0).contains(&row.mean_cpdag_mark_distance));
            assert!((0.0..=1.0).contains(&row.markov_equivalence_class_recovery_rate));
            assert!((0.0..=1.0).contains(&row.mean_posterior_mass_on_true_markov_class));
            assert!((0.0..=3.0).contains(&row.mean_structural_hamming_distance));
            assert!((0.0..=1.0).contains(&row.graph_claim_coverage));
            assert!((0.0..=1.0).contains(&row.false_graph_claim_rate_all_trials));
            assert!((0.0..=1.0).contains(&row.cpdag_claim_coverage));
            assert!((0.0..=1.0).contains(&row.false_cpdag_claim_rate_all_trials));
            assert!(
                row.cpdag_claim_coverage + 1.0e-12 >= row.graph_claim_coverage,
                "a qualified full-DAG claim also qualifies its Markov-equivalence class"
            );
            if row.family == ScmGraphFamily::NullGraphNegativeControl {
                assert!(row.mean_structural_hamming_distance >= 0.0);
            }
        }
        for aggregate in &report.held_out_policy_aggregates {
            assert_eq!(aggregate.held_out_worlds, 96);
            assert_eq!(aggregate.held_out_topology_instances, 28);
            assert!((0.0..=1.0).contains(&aggregate.mean_held_out_directed_edge_f1));
            assert!((0.0..=1.0).contains(&aggregate.mean_held_out_cpdag_skeleton_f1));
            assert!((0.0..=1.0).contains(&aggregate.mean_held_out_compelled_direction_f1));
            assert!((0.0..=3.0).contains(&aggregate.mean_held_out_cpdag_mark_distance));
            assert!((0.0..=1.0).contains(&aggregate.held_out_markov_equivalence_class_recovery_rate));
            assert!((0.0..=1.0).contains(&aggregate.mean_held_out_cpdag_claim_coverage));
            assert!((0.0..=1.0).contains(&aggregate.mean_held_out_false_cpdag_claim_rate_all_trials));
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
