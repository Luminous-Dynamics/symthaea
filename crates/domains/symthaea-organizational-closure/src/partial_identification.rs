//! Decision-focused partial identification and max-potency experiments.
//!
//! This deterministic finite-world evaluator measures the guaranteed
//! worst-case contraction of an effect identified set before observing
//! outcomes, then validates the post-intervention bounds and robust action.
//! It does not implement arbitrary ADMG identification or stochastic coverage.

use super::experiment_design::{ExperimentBudget, ExperimentEvidenceStatus, ExperimentTrial, ProbeId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectInterval {
    pub lower: i64,
    pub upper: i64,
}

impl EffectInterval {
    pub fn new(lower: i64, upper: i64) -> Self {
        Self { lower, upper }
    }

    pub fn is_valid(&self) -> bool {
        self.lower <= self.upper
    }

    pub fn contains(&self, value: i64) -> bool {
        self.lower <= value && value <= self.upper
    }

    pub fn width(&self) -> i128 {
        i128::from(self.upper) - i128::from(self.lower)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectedCausalEdge {
    pub cause: String,
    pub effect: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BidirectedCausalEdge {
    pub left: String,
    pub right: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalGraphDescriptor {
    pub variables: BTreeSet<String>,
    pub directed_edges: BTreeSet<DirectedCausalEdge>,
    pub bidirected_edges: BTreeSet<BidirectedCausalEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEffectQuery {
    pub treatment_variable: String,
    pub outcome_variable: String,
    pub target_population: String,
    pub decision_threshold: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialIdentificationExperiment {
    pub id: ProbeId,
    pub intervened_variables: BTreeSet<String>,
    /// Variables measured by this experiment (the W set in the max-potency
    /// formalism). It must be non-empty and disjoint from interventions.
    pub measured_variables: BTreeSet<String>,
    pub cost_units: u64,
    pub risk_units: u64,
}

impl PartialIdentificationExperiment {
    pub fn new(
        id: impl Into<ProbeId>,
        intervened_variable: impl Into<String>,
        cost_units: u64,
        risk_units: u64,
    ) -> Self {
        Self {
            id: id.into(),
            intervened_variables: [intervened_variable.into()].into(),
            measured_variables: BTreeSet::new(),
            cost_units,
            risk_units,
        }
    }

    pub fn with_interventions(
        mut self,
        variables: impl IntoIterator<Item = String>,
    ) -> Self {
        self.intervened_variables = variables.into_iter().collect();
        self
    }

    pub fn measuring(
        mut self,
        variables: impl IntoIterator<Item = String>,
    ) -> Self {
        self.measured_variables = variables.into_iter().collect();
        self
    }
}

/// Agent-facing information; no admissible-world outcomes or true effects are
/// exposed here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialIdentificationScenario {
    pub graph: CausalGraphDescriptor,
    pub query: CausalEffectQuery,
    pub initial_bounds: EffectInterval,
    pub experiments: Vec<PartialIdentificationExperiment>,
    pub budget: ExperimentBudget,
}

/// One evaluator-only admissible world.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissibleCausalWorld {
    pub world_id: String,
    pub target_effect: i64,
    pub outcomes: BTreeMap<ProbeId, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialIdentificationOracle {
    pub actual_world_id: String,
    pub admissible_worlds: Vec<AdmissibleCausalWorld>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RobustDecision {
    Act,
    DoNotAct,
    Defer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialIdentificationTrace {
    /// Selected as a batch before any of these outcomes were observed.
    pub trials: Vec<ExperimentTrial>,
    /// The agent's claimed post-outcome identified set.
    pub declared_bounds: EffectInterval,
    pub decision: RobustDecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialIdentificationQualification {
    Qualified,
    NotQualified,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionQuality {
    RobustlyCorrect,
    SafeButConservative,
    Unsafe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialIdentificationFindingCode {
    EmptyGraphVariables,
    EmptyVariableId,
    UnknownDirectedEdgeVariable,
    SelfDirectedEdge,
    DirectedCycle,
    UnknownBidirectedEdgeVariable,
    SelfBidirectedEdge,
    EmptyTreatmentVariable,
    EmptyOutcomeVariable,
    SameTreatmentAndOutcome,
    UnknownQueryVariable,
    EmptyTargetPopulation,
    InvalidInitialBounds,
    EmptyExperimentSet,
    EmptyExperimentId,
    DuplicateExperimentId,
    UnknownIntervenedVariable,
    EmptyExperimentInterventions,
    EmptyMeasuredVariableSet,
    UnknownMeasuredVariable,
    ExperimentIntervenesOnMeasuredVariable,
    EmptyWorldSet,
    DuplicateWorldId,
    UnknownActualWorld,
    WorldEffectOutsideInitialBounds,
    InitialBoundsDoNotMatchFiniteWorldEnvelope,
    MissingWorldExperimentOutcome,
    EmptyWorldExperimentOutcome,
    UnknownWorldExperimentOutcome,
    UnknownSelectedExperiment,
    DuplicateSelectedExperiment,
    TooManySelectedExperiments,
    CostBudgetExceeded,
    RiskBudgetExceeded,
    DeliveryRejected,
    IsolationRejected,
    ObservationRejected,
    EvidenceUnavailable,
    ObservationContradictsOracle,
    ObservedWorldSetEmpty,
    DeclaredBoundsInvalid,
    DeclaredBoundsExcludeAdmissibleWorld,
    UnsafeActDecision,
    UnsafeDoNotActDecision,
    EnumerationBudgetExceeded,
    GraphPruningCertificateDisagreement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialIdentificationFinding {
    pub code: PartialIdentificationFindingCode,
    pub trial_index: Option<usize>,
    pub experiment_id: Option<ProbeId>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialIdentificationPolicy {
    /// Maximum number of experiment subsets for exact oracle enumeration.
    pub max_subset_evaluations: usize,
}

impl Default for PartialIdentificationPolicy {
    fn default() -> Self {
        Self { max_subset_evaluations: 65_536 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialIdentificationReport {
    pub qualification: PartialIdentificationQualification,
    pub decision_quality: Option<DecisionQuality>,
    pub exact_identified_bounds: Option<EffectInterval>,
    pub declared_bounds: EffectInterval,
    pub selected_experiment_count: usize,
    pub total_cost_units: u64,
    pub total_risk_units: u64,
    pub possible_worlds_before: usize,
    pub possible_worlds_after: usize,
    pub guaranteed_potency: Option<i128>,
    pub optimal_guaranteed_potency: Option<i128>,
    pub guaranteed_potency_regret: Option<i128>,
    pub normalized_potency_regret: Option<f64>,
    pub selected_experiment_ids: Vec<ProbeId>,
    pub oracle_optimal_experiment_ids: Vec<ProbeId>,
    pub zero_potency_experiment_count: usize,
    pub selected_zero_potency_count: usize,
    /// Number of candidates certified by the published AllPathsBlocked
    /// graph condition before finite-world consistency checking.
    pub path_interception_candidate_count: usize,
    /// Candidates whose graph certificate agrees with exact finite-world
    /// zero singleton potency. They are not yet removed from subset search.
    pub path_interception_confirmed_count: usize,
    /// Graph certificates contradicted by the finite outcome oracle.
    pub path_interception_disagreement_count: usize,
    pub findings: Vec<PartialIdentificationFinding>,
}

impl PartialIdentificationReport {
    pub fn valid_bounds(&self) -> bool {
        self.exact_identified_bounds.is_some()
            && self.findings.iter().all(|finding| {
                finding.code != PartialIdentificationFindingCode::DeclaredBoundsExcludeAdmissibleWorld
                    && finding.code != PartialIdentificationFindingCode::DeclaredBoundsInvalid
            })
    }

    pub fn robust_decision(&self) -> bool {
        self.qualification == PartialIdentificationQualification::Qualified
            && self.decision_quality == Some(DecisionQuality::RobustlyCorrect)
    }
}

fn push_finding(
    findings: &mut Vec<PartialIdentificationFinding>,
    code: PartialIdentificationFindingCode,
    trial_index: Option<usize>,
    experiment_id: Option<ProbeId>,
    detail: impl Into<String>,
) {
    findings.push(PartialIdentificationFinding {
        code,
        trial_index,
        experiment_id,
        detail: detail.into(),
    });
}

fn validate_graph(scenario: &PartialIdentificationScenario,
    findings: &mut Vec<PartialIdentificationFinding>) {
    let graph = &scenario.graph;
    if graph.variables.is_empty() {
        push_finding(findings, PartialIdentificationFindingCode::EmptyGraphVariables,
            None, None, "causal graph has no variables");
    }
    for variable in &graph.variables {
        if variable.trim().is_empty() {
            push_finding(findings, PartialIdentificationFindingCode::EmptyVariableId,
                None, None, "causal graph contains an empty variable ID");
        }
    }

    let mut indegree: BTreeMap<String, usize> =
        graph.variables.iter().map(|variable| (variable.clone(), 0)).collect();
    let mut outgoing: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for edge in &graph.directed_edges {
        if !graph.variables.contains(&edge.cause) || !graph.variables.contains(&edge.effect) {
            push_finding(findings, PartialIdentificationFindingCode::UnknownDirectedEdgeVariable,
                None, None, "directed edge references an undeclared variable");
            continue;
        }
        if edge.cause == edge.effect {
            push_finding(findings, PartialIdentificationFindingCode::SelfDirectedEdge,
                None, None, "directed self-edge is invalid");
            continue;
        }
        outgoing.entry(edge.cause.clone()).or_default().insert(edge.effect.clone());
        *indegree.entry(edge.effect.clone()).or_default() += 1;
    }

    let mut ready: BTreeSet<String> = indegree.iter()
        .filter_map(|(variable, degree)| (*degree == 0).then_some(variable.clone()))
        .collect();
    let mut visited = 0usize;
    while let Some(variable) = ready.pop_first() {
        visited += 1;
        if let Some(children) = outgoing.get(&variable) {
            for child in children {
                if let Some(degree) = indegree.get_mut(child) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        ready.insert(child.clone());
                    }
                }
            }
        }
    }
    if visited != graph.variables.len() {
        push_finding(findings, PartialIdentificationFindingCode::DirectedCycle,
            None, None, "directed causal edges contain a cycle");
    }

    for edge in &graph.bidirected_edges {
        if !graph.variables.contains(&edge.left) || !graph.variables.contains(&edge.right) {
            push_finding(findings, PartialIdentificationFindingCode::UnknownBidirectedEdgeVariable,
                None, None, "bidirected edge references an undeclared variable");
        } else if edge.left == edge.right {
            push_finding(findings, PartialIdentificationFindingCode::SelfBidirectedEdge,
                None, None, "bidirected self-edge is invalid");
        }
    }

    if scenario.query.treatment_variable.trim().is_empty() {
        push_finding(findings, PartialIdentificationFindingCode::EmptyTreatmentVariable,
            None, None, "treatment variable is empty");
    }
    if scenario.query.outcome_variable.trim().is_empty() {
        push_finding(findings, PartialIdentificationFindingCode::EmptyOutcomeVariable,
            None, None, "outcome variable is empty");
    }
    if scenario.query.treatment_variable == scenario.query.outcome_variable {
        push_finding(findings, PartialIdentificationFindingCode::SameTreatmentAndOutcome,
            None, None, "treatment and outcome variables must differ");
    }
    if !graph.variables.contains(&scenario.query.treatment_variable)
        || !graph.variables.contains(&scenario.query.outcome_variable) {
        push_finding(findings, PartialIdentificationFindingCode::UnknownQueryVariable,
            None, None, "effect query references a variable absent from the graph");
    }
    if scenario.query.target_population.trim().is_empty() {
        push_finding(findings, PartialIdentificationFindingCode::EmptyTargetPopulation,
            None, None, "target population must be declared");
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PathNode {
    Observed(String),
    Latent(usize),
}

fn path_adjacency(graph: &CausalGraphDescriptor) -> BTreeMap<PathNode, BTreeSet<PathNode>> {
    let mut adjacency: BTreeMap<PathNode, BTreeSet<PathNode>> = graph.variables.iter()
        .map(|variable| (PathNode::Observed(variable.clone()), BTreeSet::new()))
        .collect();

    for edge in &graph.directed_edges {
        adjacency.entry(PathNode::Observed(edge.cause.clone())).or_default()
            .insert(PathNode::Observed(edge.effect.clone()));
    }
    for (index, edge) in graph.bidirected_edges.iter().enumerate() {
        let latent = PathNode::Latent(index);
        adjacency.entry(latent.clone()).or_default()
            .insert(PathNode::Observed(edge.left.clone()));
        adjacency.entry(latent).or_default()
            .insert(PathNode::Observed(edge.right.clone()));
    }
    adjacency
}

fn reaches_any_target(
    adjacency: &BTreeMap<PathNode, BTreeSet<PathNode>>,
    start: PathNode,
    targets: &BTreeSet<PathNode>,
    removed: &BTreeSet<PathNode>,
) -> bool {
    if removed.contains(&start) || !adjacency.contains_key(&start) {
        return false;
    }
    let mut visited = BTreeSet::new();
    let mut queue = std::collections::VecDeque::new();
    visited.insert(start.clone());
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        if targets.contains(&current) {
            return true;
        }
        if let Some(children) = adjacency.get(&current) {
            for child in children {
                if !removed.contains(child) && visited.insert(child.clone()) {
                    queue.push_back(child.clone());
                }
            }
        }
    }
    false
}

/// Compute R*_theta: hidden response types in every bidirected district that
/// contains a response type with a path to the target outcome after removing
/// the query treatment. This is the district-hull step in the published
/// AllPathsBlocked criterion.
fn relevant_latent_district_hull(
    graph: &CausalGraphDescriptor,
    query: &CausalEffectQuery,
) -> BTreeSet<usize> {
    let adjacency = path_adjacency(graph);
    let removed = [PathNode::Observed(query.treatment_variable.clone())].into();
    let target = [PathNode::Observed(query.outcome_variable.clone())].into();
    let mut relevant_latents = BTreeSet::new();

    for (index, _) in graph.bidirected_edges.iter().enumerate() {
        if reaches_any_target(&adjacency, PathNode::Latent(index), &target, &removed) {
            relevant_latents.insert(index);
        }
    }
    if relevant_latents.is_empty() {
        return BTreeSet::new();
    }

    let mut district_adjacency: BTreeMap<String, BTreeSet<String>> =
        graph.variables.iter().map(|v| (v.clone(), BTreeSet::new())).collect();
    for edge in &graph.bidirected_edges {
        district_adjacency.entry(edge.left.clone()).or_default().insert(edge.right.clone());
        district_adjacency.entry(edge.right.clone()).or_default().insert(edge.left.clone());
    }

    let mut component_of = BTreeMap::new();
    let mut component = 0usize;
    for variable in &graph.variables {
        if component_of.contains_key(variable) {
            continue;
        }
        let mut queue = std::collections::VecDeque::from([variable.clone()]);
        component_of.insert(variable.clone(), component);
        while let Some(current) = queue.pop_front() {
            if let Some(neighbours) = district_adjacency.get(&current) {
                for neighbour in neighbours {
                    if !component_of.contains_key(neighbour) {
                        component_of.insert(neighbour.clone(), component);
                        queue.push_back(neighbour.clone());
                    }
                }
            }
        }
        component += 1;
    }

    let mut relevant_components = BTreeSet::new();
    for index in &relevant_latents {
        if let Some(edge) = graph.bidirected_edges.iter().nth(*index) {
            if let Some(component_id) = component_of.get(&edge.left) {
                relevant_components.insert(*component_id);
            }
        }
    }

    graph.bidirected_edges.iter().enumerate()
        .filter_map(|(index, edge)| {
            component_of.get(&edge.left)
                .filter(|component_id| relevant_components.contains(component_id))
                .map(|_| index)
        })
        .collect()
}

/// Published AllPathsBlocked test: remove the experiment's intervention
/// variables and search from every latent in R*_theta to its measured W set.
/// A reachable measured variable means the experiment is not graph-pruned.
fn all_paths_blocked_for_experiment(
    graph: &CausalGraphDescriptor,
    query: &CausalEffectQuery,
    experiment: &PartialIdentificationExperiment,
) -> bool {
    let relevant_latents = relevant_latent_district_hull(graph, query);
    if relevant_latents.is_empty() {
        return true;
    }
    let adjacency = path_adjacency(graph);
    let removed: BTreeSet<PathNode> = experiment.intervened_variables.iter()
        .cloned().map(PathNode::Observed).collect();
    let targets: BTreeSet<PathNode> = experiment.measured_variables.iter()
        .cloned().map(PathNode::Observed).collect();

    relevant_latents.into_iter().all(|latent| {
        !reaches_any_target(&adjacency, PathNode::Latent(latent), &targets, &removed)
    })
}

fn width_for_worlds(worlds: &[&AdmissibleCausalWorld]) -> i128 {
    if worlds.is_empty() {
        return 0;
    }
    let lower = worlds.iter().map(|world| world.target_effect).min().unwrap_or(0);
    let upper = worlds.iter().map(|world| world.target_effect).max().unwrap_or(0);
    i128::from(upper) - i128::from(lower)
}

/// Computes the largest remaining identified-set width over every joint
/// outcome cell induced by a selected experiment set.
fn worst_case_remaining_width(
    worlds: &[AdmissibleCausalWorld],
    selected: &[ProbeId],
) -> i128 {
    let mut cells: BTreeMap<Vec<String>, Vec<&AdmissibleCausalWorld>> = BTreeMap::new();
    for world in worlds {
        let signature = selected.iter().map(|id| {
            world.outcomes.get(id).cloned().unwrap_or_default()
        }).collect::<Vec<_>>();
        cells.entry(signature).or_default().push(world);
    }
    cells.values().map(|cell| width_for_worlds(cell)).max().unwrap_or(0)
}

fn guaranteed_potency(worlds: &[AdmissibleCausalWorld],
    initial_bounds: EffectInterval, selected: &[ProbeId]) -> i128 {
    initial_bounds.width() - worst_case_remaining_width(worlds, selected)
}

fn exact_bounds_after_observations(worlds: &[AdmissibleCausalWorld],
    observations: &BTreeMap<ProbeId, String>) -> Option<EffectInterval> {
    let consistent: Vec<&AdmissibleCausalWorld> = worlds.iter()
        .filter(|world| observations.iter().all(|(probe, outcome)| {
            world.outcomes.get(probe).is_some_and(|candidate| candidate == outcome)
        }))
        .collect();
    if consistent.is_empty() {
        return None;
    }
    Some(EffectInterval {
        lower: consistent.iter().map(|world| world.target_effect).min()?,
        upper: consistent.iter().map(|world| world.target_effect).max()?,
    })
}

fn expected_decision(bounds: EffectInterval, threshold: i64) -> RobustDecision {
    if bounds.lower > threshold {
        RobustDecision::Act
    } else if bounds.upper <= threshold {
        RobustDecision::DoNotAct
    } else {
        RobustDecision::Defer
    }
}

fn enumeration_optimum(
    scenario: &PartialIdentificationScenario,
    oracle: &PartialIdentificationOracle,
    policy: &PartialIdentificationPolicy,
) -> Result<(i128, Vec<ProbeId>), PartialIdentificationFindingCode> {
    let count = scenario.experiments.len();
    let subset_count = 1_u128.checked_shl(count as u32)
        .ok_or(PartialIdentificationFindingCode::EnumerationBudgetExceeded)?;
    if subset_count > policy.max_subset_evaluations as u128 {
        return Err(PartialIdentificationFindingCode::EnumerationBudgetExceeded);
    }

    let mut best_potency = i128::MIN;
    let mut best_cost = u64::MAX;
    let mut best_ids = Vec::new();

    for mask in 0..subset_count {
        let selected = scenario.experiments.iter().enumerate()
            .filter_map(|(index, experiment)| {
                ((mask & (1_u128 << index)) != 0).then_some(experiment)
            })
            .collect::<Vec<_>>();
        if selected.len() > scenario.budget.max_trials {
            continue;
        }

        let cost = selected.iter().try_fold(0_u64, |sum, experiment| {
            sum.checked_add(experiment.cost_units)
        }).unwrap_or(u64::MAX);
        let risk = selected.iter().try_fold(0_u64, |sum, experiment| {
            sum.checked_add(experiment.risk_units)
        }).unwrap_or(u64::MAX);
        if cost > scenario.budget.max_cost_units || risk > scenario.budget.max_risk_units {
            continue;
        }
        let ids = selected.iter().map(|experiment| experiment.id.clone()).collect::<Vec<_>>();
        let potency = guaranteed_potency(&oracle.admissible_worlds, scenario.initial_bounds, &ids);
        if potency > best_potency || (potency == best_potency && cost < best_cost) {
            best_potency = potency;
            best_cost = cost;
            best_ids = ids;
        }
    }

    if best_potency == i128::MIN {
        return Ok((0, Vec::new()));
    }
    Ok((best_potency, best_ids))
}

/// Evaluate a batch experiment set and a post-outcome robust decision.
///
/// The selected probes are chosen before outcomes arrive. The evaluator
/// computes guaranteed potency against all admissible worlds, obtains the
/// actual world outcomes from a hidden oracle, validates the agent's declared
/// post-outcome bound, and checks whether the decision is robust over it.
pub fn evaluate_partial_identification(
    scenario: &PartialIdentificationScenario,
    oracle: &PartialIdentificationOracle,
    trace: &PartialIdentificationTrace,
    policy: &PartialIdentificationPolicy,
) -> PartialIdentificationReport {
    let mut findings = Vec::new();
    validate_graph(scenario, &mut findings);

    if !scenario.initial_bounds.is_valid() {
        push_finding(&mut findings, PartialIdentificationFindingCode::InvalidInitialBounds,
            None, None, "initial effect bounds have lower greater than upper");
    }
    if scenario.experiments.is_empty() {
        push_finding(&mut findings, PartialIdentificationFindingCode::EmptyExperimentSet,
            None, None, "no candidate experiments were declared");
    }

    let mut experiments: BTreeMap<ProbeId, &PartialIdentificationExperiment> = BTreeMap::new();
    for experiment in &scenario.experiments {
        if experiment.id.0.trim().is_empty() {
            push_finding(&mut findings, PartialIdentificationFindingCode::EmptyExperimentId,
                None, Some(experiment.id.clone()), "experiment ID is empty");
        }
        if !scenario.graph.variables.contains(&experiment.intervened_variable) {
            push_finding(&mut findings, PartialIdentificationFindingCode::UnknownIntervenedVariable,
                None, Some(experiment.id.clone()), "experiment intervenes on an undeclared variable");
        }
        if experiments.insert(experiment.id.clone(), experiment).is_some() {
            push_finding(&mut findings, PartialIdentificationFindingCode::DuplicateExperimentId,
                None, Some(experiment.id.clone()), "experiment ID is duplicated");
        }
    }

    if oracle.admissible_worlds.is_empty() {
        push_finding(&mut findings, PartialIdentificationFindingCode::EmptyWorldSet,
            None, None, "oracle contains no admissible causal worlds");
    }
    let mut worlds: BTreeMap<String, &AdmissibleCausalWorld> = BTreeMap::new();
    for world in &oracle.admissible_worlds {
        if world.world_id.trim().is_empty() {
            push_finding(&mut findings, PartialIdentificationFindingCode::DuplicateWorldId,
                None, None, "admissible world ID is empty");
        }
        if worlds.insert(world.world_id.clone(), world).is_some() {
            push_finding(&mut findings, PartialIdentificationFindingCode::DuplicateWorldId,
                None, None, "admissible world ID is duplicated");
        }
        if !scenario.initial_bounds.contains(world.target_effect) {
            push_finding(&mut findings, PartialIdentificationFindingCode::WorldEffectOutsideInitialBounds,
                None, None, format!("world {} lies outside the initial effect bounds", world.world_id));
        }
        for experiment_id in experiments.keys() {
            match world.outcomes.get(experiment_id) {
                None => push_finding(&mut findings, PartialIdentificationFindingCode::MissingWorldExperimentOutcome,
                    None, Some(experiment_id.clone()), format!("world {} omits an experiment outcome", world.world_id)),
                Some(outcome) if outcome.trim().is_empty() => push_finding(&mut findings,
                    PartialIdentificationFindingCode::EmptyWorldExperimentOutcome, None, Some(experiment_id.clone()),
                    format!("world {} has an empty experiment outcome", world.world_id)),
                Some(_) => {}
            }
        }
        for experiment_id in world.outcomes.keys() {
            if !experiments.contains_key(experiment_id) {
                push_finding(&mut findings, PartialIdentificationFindingCode::UnknownWorldExperimentOutcome,
                    None, Some(experiment_id.clone()), format!("world {} references an unknown experiment", world.world_id));
            }
        }
    }
    if !worlds.contains_key(&oracle.actual_world_id) {
        push_finding(&mut findings, PartialIdentificationFindingCode::UnknownActualWorld,
            None, None, "actual world ID is absent from the admissible world set");
    }
    if !oracle.admissible_worlds.is_empty() {
        let lower = oracle.admissible_worlds.iter().map(|world| world.target_effect).min().unwrap_or(0);
        let upper = oracle.admissible_worlds.iter().map(|world| world.target_effect).max().unwrap_or(0);
        if scenario.initial_bounds != EffectInterval::new(lower, upper) {
            push_finding(&mut findings, PartialIdentificationFindingCode::InitialBoundsDoNotMatchFiniteWorldEnvelope,
                None, None, "initial interval must equal the finite admissible-world effect envelope");
        }
    }

    if !findings.is_empty() {
        return PartialIdentificationReport {
            qualification: PartialIdentificationQualification::Inconclusive,
            decision_quality: None,
            exact_identified_bounds: None,
            declared_bounds: trace.declared_bounds,
            selected_experiment_count: trace.trials.len(),
            total_cost_units: 0,
            total_risk_units: 0,
            possible_worlds_before: oracle.admissible_worlds.len(),
            possible_worlds_after: 0,
            guaranteed_potency: None,
            optimal_guaranteed_potency: None,
            guaranteed_potency_regret: None,
            normalized_potency_regret: None,
            selected_experiment_ids: Vec::new(),
            oracle_optimal_experiment_ids: Vec::new(),
            zero_potency_experiment_count: 0,
            selected_zero_potency_count: 0,
            findings,
        };
    }

    let actual_world = worlds[&oracle.actual_world_id];
    let mut selected_ids = Vec::new();
    let mut selected_id_set = BTreeSet::new();
    let mut observed = BTreeMap::new();
    let mut total_cost = 0_u64;
    let mut total_risk = 0_u64;
    let mut unresolved = false;

    for (index, trial) in trace.trials.iter().enumerate() {
        let trial_index = index + 1;
        let Some(experiment) = experiments.get(&trial.probe_id) else {
            push_finding(&mut findings, PartialIdentificationFindingCode::UnknownSelectedExperiment,
                Some(trial_index), Some(trial.probe_id.clone()), "selected experiment is undeclared");
            continue;
        };
        if !selected_id_set.insert(trial.probe_id.clone()) {
            push_finding(&mut findings, PartialIdentificationFindingCode::DuplicateSelectedExperiment,
                Some(trial_index), Some(trial.probe_id.clone()), "an experiment was selected more than once");
        }
        selected_ids.push(trial.probe_id.clone());
        if trial.observed_outcome.trim().is_empty() {
            push_finding(&mut findings, PartialIdentificationFindingCode::ObservationRejected,
                Some(trial_index), Some(trial.probe_id.clone()), "observed experiment outcome is empty");
            unresolved = true;
        }
        let next_cost = total_cost.checked_add(experiment.cost_units);
        let next_risk = total_risk.checked_add(experiment.risk_units);
        if next_cost.is_none_or(|value| value > scenario.budget.max_cost_units) {
            push_finding(&mut findings, PartialIdentificationFindingCode::CostBudgetExceeded,
                Some(trial_index), Some(trial.probe_id.clone()), "selected experiment set exceeds cost budget");
        }
        if next_risk.is_none_or(|value| value > scenario.budget.max_risk_units) {
            push_finding(&mut findings, PartialIdentificationFindingCode::RiskBudgetExceeded,
                Some(trial_index), Some(trial.probe_id.clone()), "selected experiment set exceeds risk budget");
        }
        total_cost = next_cost.unwrap_or(u64::MAX);
        total_risk = next_risk.unwrap_or(u64::MAX);
        if trial_index > scenario.budget.max_trials {
            push_finding(&mut findings, PartialIdentificationFindingCode::TooManySelectedExperiments,
                Some(trial_index), Some(trial.probe_id.clone()), "selected experiment set exceeds trial budget");
        }

        for (label, status) in [
            ("delivery", trial.delivery),
            ("isolation", trial.isolation),
            ("observation", trial.observation),
        ] {
            match status {
                ExperimentEvidenceStatus::Verified => {}
                ExperimentEvidenceStatus::Rejected => {
                    unresolved = true;
                    let code = match label {
                        "delivery" => PartialIdentificationFindingCode::DeliveryRejected,
                        "isolation" => PartialIdentificationFindingCode::IsolationRejected,
                        _ => PartialIdentificationFindingCode::ObservationRejected,
                    };
                    push_finding(&mut findings, code, Some(trial_index), Some(trial.probe_id.clone()),
                        format!("{label} evidence was rejected"));
                }
                ExperimentEvidenceStatus::Unavailable => {
                    unresolved = true;
                    push_finding(&mut findings, PartialIdentificationFindingCode::EvidenceUnavailable,
                        Some(trial_index), Some(trial.probe_id.clone()),
                        format!("{label} evidence is unavailable"));
                }
            }
        }

        if trial.observed_outcome != actual_world.outcomes[&trial.probe_id] {
            unresolved = true;
            push_finding(&mut findings, PartialIdentificationFindingCode::ObservationContradictsOracle,
                Some(trial_index), Some(trial.probe_id.clone()),
                "observed outcome differs from the independent actual-world oracle");
        }
        observed.insert(trial.probe_id.clone(), trial.observed_outcome.clone());
    }

    if selected_ids.len() > scenario.budget.max_trials {
        push_finding(&mut findings, PartialIdentificationFindingCode::TooManySelectedExperiments,
            None, None, "selected experiment set exceeds trial budget");
    }
    if unresolved {
        return PartialIdentificationReport {
            qualification: PartialIdentificationQualification::Inconclusive,
            decision_quality: None,
            exact_identified_bounds: None,
            declared_bounds: trace.declared_bounds,
            selected_experiment_count: selected_ids.len(),
            total_cost_units: total_cost,
            total_risk_units: total_risk,
            possible_worlds_before: oracle.admissible_worlds.len(),
            possible_worlds_after: 0,
            guaranteed_potency: None,
            optimal_guaranteed_potency: None,
            guaranteed_potency_regret: None,
            normalized_potency_regret: None,
            selected_experiment_ids: selected_ids.into_iter().collect(),
            oracle_optimal_experiment_ids: Vec::new(),
            zero_potency_experiment_count: 0,
            selected_zero_potency_count: 0,
            findings,
        };
    }

    if findings.iter().any(|finding| matches!(
        finding.code,
        PartialIdentificationFindingCode::UnknownSelectedExperiment
            | PartialIdentificationFindingCode::DuplicateSelectedExperiment
            | PartialIdentificationFindingCode::CostBudgetExceeded
            | PartialIdentificationFindingCode::RiskBudgetExceeded
            | PartialIdentificationFindingCode::TooManySelectedExperiments
    )) {
        return PartialIdentificationReport {
            qualification: PartialIdentificationQualification::NotQualified,
            decision_quality: None,
            exact_identified_bounds: None,
            declared_bounds: trace.declared_bounds,
            selected_experiment_count: selected_ids.len(),
            total_cost_units: total_cost,
            total_risk_units: total_risk,
            possible_worlds_before: oracle.admissible_worlds.len(),
            possible_worlds_after: 0,
            guaranteed_potency: None,
            optimal_guaranteed_potency: None,
            guaranteed_potency_regret: None,
            normalized_potency_regret: None,
            selected_experiment_ids: selected_ids.into_iter().collect(),
            oracle_optimal_experiment_ids: Vec::new(),
            zero_potency_experiment_count: 0,
            selected_zero_potency_count: 0,
            findings,
        };
    }

    let selected_vec = selected_ids.iter().cloned().collect::<Vec<_>>();
    let selection_potency = guaranteed_potency(
        &oracle.admissible_worlds,
        scenario.initial_bounds,
        &selected_vec,
    );
    let selected_zero_potency_count = selected_vec.iter()
        .filter(|id| guaranteed_potency(&oracle.admissible_worlds, scenario.initial_bounds,
            std::slice::from_ref(*id)) == 0)
        .count();
    let zero_potency_experiment_count = scenario.experiments.iter()
        .filter(|experiment| guaranteed_potency(&oracle.admissible_worlds,
            scenario.initial_bounds, std::slice::from_ref(&experiment.id)) == 0)
        .count();

    let (optimal_potency, optimal_ids) = match enumeration_optimum(scenario, oracle, policy) {
        Ok(value) => value,
        Err(code) => {
            push_finding(&mut findings, code, None, None, "exact subset-oracle enumeration budget was exceeded");
            return PartialIdentificationReport {
                qualification: PartialIdentificationQualification::Inconclusive,
                decision_quality: None,
                exact_identified_bounds: None,
                declared_bounds: trace.declared_bounds,
                selected_experiment_count: selected_vec.len(),
                total_cost_units: total_cost,
                total_risk_units: total_risk,
                possible_worlds_before: oracle.admissible_worlds.len(),
                possible_worlds_after: 0,
                guaranteed_potency: Some(selection_potency),
                optimal_guaranteed_potency: None,
                guaranteed_potency_regret: None,
                normalized_potency_regret: None,
                selected_experiment_ids: selected_vec,
                oracle_optimal_experiment_ids: Vec::new(),
                zero_potency_experiment_count,
                selected_zero_potency_count,
                findings,
            };
        }
    };

    let exact_bounds = exact_bounds_after_observations(&oracle.admissible_worlds, &observed);
    let Some(exact_bounds) = exact_bounds else {
        push_finding(&mut findings, PartialIdentificationFindingCode::ObservedWorldSetEmpty,
            None, None, "observed outcome combination is inconsistent with all admissible worlds");
        return PartialIdentificationReport {
            qualification: PartialIdentificationQualification::Inconclusive,
            decision_quality: None,
            exact_identified_bounds: None,
            declared_bounds: trace.declared_bounds,
            selected_experiment_count: selected_vec.len(),
            total_cost_units: total_cost,
            total_risk_units: total_risk,
            possible_worlds_before: oracle.admissible_worlds.len(),
            possible_worlds_after: 0,
            guaranteed_potency: Some(selection_potency),
            optimal_guaranteed_potency: Some(optimal_potency),
            guaranteed_potency_regret: Some((optimal_potency - selection_potency).max(0)),
            normalized_potency_regret: if optimal_potency > 0 {
                Some((optimal_potency - selection_potency).max(0) as f64 / optimal_potency as f64)
            } else { Some(0.0) },
            selected_experiment_ids: selected_vec,
            oracle_optimal_experiment_ids: optimal_ids,
            zero_potency_experiment_count,
            selected_zero_potency_count,
            findings,
        };
    };

    if !trace.declared_bounds.is_valid() {
        push_finding(&mut findings, PartialIdentificationFindingCode::DeclaredBoundsInvalid,
            None, None, "declared post-experiment bounds have lower greater than upper");
    }
    let compatible_worlds = oracle.admissible_worlds.iter().filter(|world| {
        observed.iter().all(|(id, outcome)| world.outcomes.get(id).is_some_and(|actual| actual == outcome))
    }).collect::<Vec<_>>();
    if compatible_worlds.iter().any(|world| !trace.declared_bounds.contains(world.target_effect)) {
        push_finding(&mut findings, PartialIdentificationFindingCode::DeclaredBoundsExcludeAdmissibleWorld,
            None, None, "declared effect interval excludes at least one world consistent with the observed outcomes");
    }

    let expected = expected_decision(exact_bounds, scenario.query.decision_threshold);
    let declared_robust = match trace.decision {
        RobustDecision::Act => trace.declared_bounds.lower > scenario.query.decision_threshold,
        RobustDecision::DoNotAct => trace.declared_bounds.upper <= scenario.query.decision_threshold,
        RobustDecision::Defer => true,
    };
    let decision_quality = if !declared_robust {
        match trace.decision {
            RobustDecision::Act => {
                push_finding(&mut findings, PartialIdentificationFindingCode::UnsafeActDecision,
                    None, None, "agent acted although its own declared interval crosses the decision threshold");
            }
            RobustDecision::DoNotAct => {
                push_finding(&mut findings, PartialIdentificationFindingCode::UnsafeDoNotActDecision,
                    None, None, "agent rejected although its own declared interval crosses the decision threshold");
            }
            RobustDecision::Defer => {}
        }
        DecisionQuality::Unsafe
    } else if trace.decision == expected {
        DecisionQuality::RobustlyCorrect
    } else {
        DecisionQuality::SafeButConservative
    };

    let regret = (optimal_potency - selection_potency).max(0);
    let normalized_regret = if optimal_potency > 0 {
        Some(regret as f64 / optimal_potency as f64)
    } else {
        Some(0.0)
    };
    let qualification = if decision_quality == DecisionQuality::Unsafe
        || findings.iter().any(|finding| matches!(
            finding.code,
            PartialIdentificationFindingCode::DeclaredBoundsInvalid
                | PartialIdentificationFindingCode::DeclaredBoundsExcludeAdmissibleWorld
        )) {
        PartialIdentificationQualification::NotQualified
    } else {
        PartialIdentificationQualification::Qualified
    };

    PartialIdentificationReport {
        qualification,
        decision_quality: Some(decision_quality),
        exact_identified_bounds: Some(exact_bounds),
        declared_bounds: trace.declared_bounds,
        selected_experiment_count: selected_vec.len(),
        total_cost_units: total_cost,
        total_risk_units: total_risk,
        possible_worlds_before: oracle.admissible_worlds.len(),
        possible_worlds_after: compatible_worlds.len(),
        guaranteed_potency: Some(selection_potency),
        optimal_guaranteed_potency: Some(optimal_potency),
        guaranteed_potency_regret: Some(regret),
        normalized_potency_regret: normalized_regret,
        selected_experiment_ids: selected_vec,
        oracle_optimal_experiment_ids: optimal_ids,
        zero_potency_experiment_count,
        selected_zero_potency_count,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario() -> (PartialIdentificationScenario, PartialIdentificationOracle) {
        let graph = CausalGraphDescriptor {
            variables: ["treatment", "outcome", "confounder"].into_iter().map(str::to_owned).collect(),
            directed_edges: [
                DirectedCausalEdge { cause: "treatment".into(), effect: "outcome".into() },
                DirectedCausalEdge { cause: "confounder".into(), effect: "treatment".into() },
                DirectedCausalEdge { cause: "confounder".into(), effect: "outcome".into() },
            ].into(),
            bidirected_edges: BTreeSet::new(),
        };
        let scenario = PartialIdentificationScenario {
            graph,
            query: CausalEffectQuery {
                treatment_variable: "treatment".into(),
                outcome_variable: "outcome".into(),
                target_population: "population-a".into(),
                decision_threshold: 0,
            },
            initial_bounds: EffectInterval::new(-2, 3),
            experiments: vec![
                PartialIdentificationExperiment::new("probe-a", "treatment", 1, 1),
                PartialIdentificationExperiment::new("probe-b", "confounder", 1, 1),
            ],
            budget: ExperimentBudget { max_cost_units: 2, max_risk_units: 2, max_trials: 2 },
        };
        let oracle = PartialIdentificationOracle {
            actual_world_id: "world-2".into(),
            admissible_worlds: vec![
                AdmissibleCausalWorld { world_id: "world-1".into(), target_effect: -2,
                    outcomes: [("probe-a".into(), "a".into()), ("probe-b".into(), "x".into())].into() },
                AdmissibleCausalWorld { world_id: "world-2".into(), target_effect: 0,
                    outcomes: [("probe-a".into(), "a".into()), ("probe-b".into(), "y".into())].into() },
                AdmissibleCausalWorld { world_id: "world-3".into(), target_effect: 3,
                    outcomes: [("probe-a".into(), "b".into()), ("probe-b".into(), "y".into())].into() },
            ],
        };
        (scenario, oracle)
    }

    fn trial(id: &str, outcome: &str) -> ExperimentTrial {
        ExperimentTrial::qualified(id, outcome)
    }

    #[test]
    fn exact_bounds_and_max_potency_are_recomputed_from_admissible_worlds() {
        let (scenario, oracle) = scenario();
        let trace = PartialIdentificationTrace {
            trials: vec![trial("probe-a", "a"), trial("probe-b", "y")],
            declared_bounds: EffectInterval::new(0, 0),
            decision: RobustDecision::DoNotAct,
        };
        let report = evaluate_partial_identification(
            &scenario, &oracle, &trace, &PartialIdentificationPolicy::default(),
        );
        assert_eq!(report.qualification, PartialIdentificationQualification::Qualified, "{:?}", report.findings);
        assert_eq!(report.exact_identified_bounds, Some(EffectInterval::new(0, 0)));
        assert_eq!(report.guaranteed_potency, Some(5));
        assert_eq!(report.optimal_guaranteed_potency, Some(5));
        assert_eq!(report.decision_quality, Some(DecisionQuality::RobustlyCorrect));
        assert!(report.robust_decision());
    }

    #[test]
    fn batch_with_lower_worst_case_potency_reports_oracle_regret() {
        let (scenario, oracle) = scenario();
        let trace = PartialIdentificationTrace {
            trials: vec![trial("probe-a", "a")],
            declared_bounds: EffectInterval::new(-2, 0),
            decision: RobustDecision::DoNotAct,
        };
        let report = evaluate_partial_identification(
            &scenario, &oracle, &trace, &PartialIdentificationPolicy::default(),
        );
        assert_eq!(report.guaranteed_potency, Some(3));
        assert_eq!(report.optimal_guaranteed_potency, Some(5));
        assert_eq!(report.guaranteed_potency_regret, Some(2));
    }

    #[test]
    fn false_narrowing_of_effect_bounds_is_rejected() {
        let (scenario, oracle) = scenario();
        let trace = PartialIdentificationTrace {
            trials: vec![trial("probe-a", "a")],
            declared_bounds: EffectInterval::new(-1, -1),
            decision: RobustDecision::DoNotAct,
        };
        let report = evaluate_partial_identification(
            &scenario, &oracle, &trace, &PartialIdentificationPolicy::default(),
        );
        assert_eq!(report.qualification, PartialIdentificationQualification::NotQualified);
        assert!(report.findings.iter().any(|f| f.code == PartialIdentificationFindingCode::DeclaredBoundsExcludeAdmissibleWorld));
    }

    #[test]
    fn conservative_defer_is_safe_but_scored_separately() {
        let (scenario, oracle) = scenario();
        let trace = PartialIdentificationTrace {
            trials: vec![trial("probe-a", "a")],
            declared_bounds: EffectInterval::new(-2, 1),
            decision: RobustDecision::Defer,
        };
        let report = evaluate_partial_identification(
            &scenario, &oracle, &trace, &PartialIdentificationPolicy::default(),
        );
        assert_eq!(report.qualification, PartialIdentificationQualification::Qualified);
        assert_eq!(report.decision_quality, Some(DecisionQuality::SafeButConservative));
    }

    #[test]
    fn unavailable_intervention_evidence_does_not_tighten_bounds() {
        let (scenario, oracle) = scenario();
        let mut observation = trial("probe-a", "a");
        observation.delivery = ExperimentEvidenceStatus::Unavailable;
        let trace = PartialIdentificationTrace {
            trials: vec![observation],
            declared_bounds: EffectInterval::new(-2, 0),
            decision: RobustDecision::DoNotAct,
        };
        let report = evaluate_partial_identification(
            &scenario, &oracle, &trace, &PartialIdentificationPolicy::default(),
        );
        assert_eq!(report.qualification, PartialIdentificationQualification::Inconclusive);
        assert_eq!(report.exact_identified_bounds, None);
    }

    #[test]
    fn actual_world_observation_must_match_the_independent_oracle() {
        let (scenario, oracle) = scenario();
        let trace = PartialIdentificationTrace {
            trials: vec![trial("probe-a", "observer-corruption")],
            declared_bounds: EffectInterval::new(-2, 0),
            decision: RobustDecision::DoNotAct,
        };
        let report = evaluate_partial_identification(
            &scenario, &oracle, &trace, &PartialIdentificationPolicy::default(),
        );
        assert_eq!(report.qualification, PartialIdentificationQualification::Inconclusive);
        assert!(report.findings.iter().any(|f| f.code == PartialIdentificationFindingCode::ObservationContradictsOracle));
    }

    #[test]
    fn false_decisive_action_is_rejected_when_the_interval_crosses_threshold() {
        let (scenario, oracle) = scenario();
        let trace = PartialIdentificationTrace {
            trials: vec![],
            declared_bounds: EffectInterval::new(-2, 3),
            decision: RobustDecision::Act,
        };
        let report = evaluate_partial_identification(
            &scenario, &oracle, &trace, &PartialIdentificationPolicy::default(),
        );
        assert_eq!(report.qualification, PartialIdentificationQualification::NotQualified);
        assert!(report.findings.iter().any(|f| f.code == PartialIdentificationFindingCode::UnsafeActDecision));
    }

    #[test]
    fn exact_subset_oracle_budget_exhaustion_is_inconclusive() {
        let (scenario, oracle) = scenario();
        let trace = PartialIdentificationTrace {
            trials: vec![],
            declared_bounds: EffectInterval::new(-2, 3),
            decision: RobustDecision::Defer,
        };
        let report = evaluate_partial_identification(
            &scenario, &oracle, &trace,
            &PartialIdentificationPolicy { max_subset_evaluations: 1 },
        );
        assert_eq!(report.qualification, PartialIdentificationQualification::Inconclusive);
        assert!(report.findings.iter().any(|f| f.code == PartialIdentificationFindingCode::EnumerationBudgetExceeded));
    }

    #[test]
    fn directed_cycles_are_rejected_as_malformed_graph_input() {
        let (mut scenario, oracle) = scenario();
        scenario.graph.directed_edges.insert(DirectedCausalEdge {
            cause: "outcome".into(), effect: "treatment".into(),
        });
        let trace = PartialIdentificationTrace {
            trials: vec![],
            declared_bounds: EffectInterval::new(-2, 3),
            decision: RobustDecision::Defer,
        };
        let report = evaluate_partial_identification(
            &scenario, &oracle, &trace, &PartialIdentificationPolicy::default(),
        );
        assert_eq!(report.qualification, PartialIdentificationQualification::Inconclusive);
        assert!(report.findings.iter().any(|f| f.code == PartialIdentificationFindingCode::DirectedCycle));
    }
}
