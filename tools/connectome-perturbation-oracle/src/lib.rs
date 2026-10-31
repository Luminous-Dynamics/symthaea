//! Deterministic synthetic oracle for perturbation-grounded connectome dynamics.
//!
//! This is a mathematical test harness, not a biological simulator. It uses a
//! discrete-time linear delayed network so that direct, indirect, delayed,
//! recurrent, and common-input-confounded effects have exact known answers.

use std::fmt;

/// A directed structural connection. The delay is measured in discrete epochs
/// and must be at least one; instantaneous algebraic loops are unsupported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    pub source: usize,
    pub target: usize,
    pub weight: f64,
    pub delay: usize,
}

/// An externally applied input to one node at one epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Injection {
    pub node: usize,
    pub epoch: usize,
    pub amplitude: f64,
}

/// A versionless synthetic network used only by this known-answer fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    pub node_count: usize,
    pub edges: Vec<Edge>,
}


/// Context used by an explicitly state-conditioned synthetic circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Resting,
    Active,
}

/// Declares when a contextual edge is available. This is a model condition,
/// not a claim that an anatomical synapse appears or disappears biologically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeScope {
    Both,
    RestingOnly,
    ActiveOnly,
}

/// Node output nonlinearity for an exact synthetic fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeTransfer {
    Identity,
    Threshold { threshold: f64 },
    Clamp { minimum: f64, maximum: f64 },
}

/// A delay-bearing edge with an explicit state applicability rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextualEdge {
    pub edge: Edge,
    pub scope: EdgeScope,
}

/// Contextual nonlinear network, kept separate from the linear reference
/// network so that comparisons cannot silently change model semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextualNetwork {
    pub node_count: usize,
    pub edges: Vec<ContextualEdge>,
    pub transfers: Vec<NodeTransfer>,
}

/// A declared extrasynaptic/latent coupling kept separate from anatomical edges.
///
/// This is a phenomenological model term. Its presence does not identify a
/// peptide, receptor, or any specific biological mechanism.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtrasynapticCoupling {
    pub source: usize,
    pub target: usize,
    pub weight: f64,
    pub delay: usize,
}

/// A model that keeps direct anatomical routes and extrasynaptic terms typed
/// separately. Structural graph queries must use the anatomy field only.
#[derive(Debug, Clone, PartialEq)]
pub struct CouplingModel {
    pub anatomy: Network,
    pub extrasynaptic: Vec<ExtrasynapticCoupling>,
}

/// Node-major state values indexed as values[epoch][node].
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    pub values: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OracleError {
    ZeroNodes,
    ZeroHorizon,
    NodeOutOfRange { node: usize, node_count: usize },
    EpochOutOfRange { epoch: usize, horizon: usize },
    ZeroDelay,
    EdgeIndexOutOfRange { edge_index: usize, edge_count: usize },
    DuplicateEdgeIndex,
    BaselineQueryUnreachable,
    NonFiniteParameter,
    ShapeMismatch,
    TargetOutOfRange { node: usize, node_count: usize },
    InvalidThreshold,
    TransferCountMismatch { expected: usize, actual: usize },
    InvalidTransferParameters,
}

impl fmt::Display for OracleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroNodes => write!(formatter, "network must contain at least one node"),
            Self::ZeroHorizon => write!(formatter, "horizon must contain at least one epoch"),
            Self::NodeOutOfRange { node, node_count } => {
                write!(formatter, "node {node} is outside 0..{node_count}")
            }
            Self::EpochOutOfRange { epoch, horizon } => {
                write!(formatter, "epoch {epoch} is outside 0..{horizon}")
            }
            Self::ZeroDelay => write!(formatter, "edge delay must be at least one epoch"),
            Self::EdgeIndexOutOfRange { edge_index, edge_count } => {
                write!(formatter, "edge index {edge_index} is outside 0..{edge_count}")
            }
            Self::DuplicateEdgeIndex => write!(formatter, "edge indices must be distinct"),
            Self::BaselineQueryUnreachable => {
                write!(formatter, "baseline reachability query must be true for an interaction")
            }
            Self::NonFiniteParameter => write!(formatter, "weights and amplitudes must be finite"),
            Self::ShapeMismatch => write!(formatter, "trajectory shapes do not match"),
            Self::TargetOutOfRange { node, node_count } => {
                write!(formatter, "target node {node} is outside 0..{node_count}")
            }
            Self::InvalidThreshold => {
                write!(formatter, "detection threshold must be finite and non-negative")
            }
            Self::TransferCountMismatch { expected, actual } => {
                write!(formatter, "expected {expected} node transfers, received {actual}")
            }
            Self::InvalidTransferParameters => {
                write!(formatter, "node transfer parameters must be finite and ordered")
            }
        }
    }
}

impl std::error::Error for OracleError {}

impl Trajectory {
    pub fn node_trace(&self, node: usize) -> Result<Vec<f64>, OracleError> {
        let node_count = self.values.first().map_or(0, Vec::len);
        if node >= node_count {
            return Err(OracleError::TargetOutOfRange { node, node_count });
        }
        Ok(self.values.iter().map(|epoch| epoch[node]).collect())
    }
}

fn validate(network: &Network, horizon: usize, inputs: &[Injection]) -> Result<(), OracleError> {
    if network.node_count == 0 {
        return Err(OracleError::ZeroNodes);
    }
    if horizon == 0 {
        return Err(OracleError::ZeroHorizon);
    }

    for edge in &network.edges {
        if edge.source >= network.node_count {
            return Err(OracleError::NodeOutOfRange {
                node: edge.source,
                node_count: network.node_count,
            });
        }
        if edge.target >= network.node_count {
            return Err(OracleError::NodeOutOfRange {
                node: edge.target,
                node_count: network.node_count,
            });
        }
        if edge.delay == 0 {
            return Err(OracleError::ZeroDelay);
        }
        if !edge.weight.is_finite() {
            return Err(OracleError::NonFiniteParameter);
        }
    }

    for input in inputs {
        if input.node >= network.node_count {
            return Err(OracleError::NodeOutOfRange {
                node: input.node,
                node_count: network.node_count,
            });
        }
        if input.epoch >= horizon {
            return Err(OracleError::EpochOutOfRange {
                epoch: input.epoch,
                horizon,
            });
        }
        if !input.amplitude.is_finite() {
            return Err(OracleError::NonFiniteParameter);
        }
    }
    Ok(())
}

/// Simulate x[t, target] = input[t, target] + Σ(weight * x[t-delay, source]).
///
/// The epoch update is order-independent because all edges read only earlier
/// epochs. Non-finite accumulated states fail closed rather than entering a
/// benchmark report.
pub fn simulate(
    network: &Network,
    horizon: usize,
    inputs: &[Injection],
) -> Result<Trajectory, OracleError> {
    validate(network, horizon, inputs)?;
    let mut values = vec![vec![0.0; network.node_count]; horizon];

    for epoch in 0..horizon {
        for input in inputs.iter().filter(|input| input.epoch == epoch) {
            values[epoch][input.node] += input.amplitude;
        }

        for edge in &network.edges {
            if epoch >= edge.delay {
                let source_value = values[epoch - edge.delay][edge.source];
                values[epoch][edge.target] += edge.weight * source_value;
            }
        }

        if values[epoch].iter().any(|value| !value.is_finite()) {
            return Err(OracleError::NonFiniteParameter);
        }
    }

    Ok(Trajectory { values })
}


fn apply_transfer(transfer: NodeTransfer, value: f64) -> Result<f64, OracleError> {
    if !value.is_finite() {
        return Err(OracleError::NonFiniteParameter);
    }
    let output = match transfer {
        NodeTransfer::Identity => value,
        NodeTransfer::Threshold { threshold } => {
            if !threshold.is_finite() {
                return Err(OracleError::InvalidTransferParameters);
            }
            if value >= threshold { 1.0 } else { 0.0 }
        }
        NodeTransfer::Clamp { minimum, maximum } => {
            if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
                return Err(OracleError::InvalidTransferParameters);
            }
            value.clamp(minimum, maximum)
        }
    };
    if !output.is_finite() {
        return Err(OracleError::NonFiniteParameter);
    }
    Ok(output)
}

/// Simulate a nonlinear network under one frozen context/state.
///
/// All edges read only earlier epochs. The transfer function is applied to
/// each node after its external input and incoming delayed signals are summed.
pub fn simulate_contextual(
    network: &ContextualNetwork,
    state: CircuitState,
    horizon: usize,
    inputs: &[Injection],
) -> Result<Trajectory, OracleError> {
    if network.transfers.len() != network.node_count {
        return Err(OracleError::TransferCountMismatch {
            expected: network.node_count,
            actual: network.transfers.len(),
        });
    }

    let base = Network {
        node_count: network.node_count,
        edges: network.edges.iter().map(|item| item.edge).collect(),
    };
    validate(&base, horizon, inputs)?;

    for transfer in &network.transfers {
        match *transfer {
            NodeTransfer::Identity => {}
            NodeTransfer::Threshold { threshold } => {
                if !threshold.is_finite() {
                    return Err(OracleError::InvalidTransferParameters);
                }
            }
            NodeTransfer::Clamp { minimum, maximum } => {
                if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
                    return Err(OracleError::InvalidTransferParameters);
                }
            }
        }
    }

    let mut values = vec![vec![0.0; network.node_count]; horizon];
    for epoch in 0..horizon {
        let mut summed = vec![0.0; network.node_count];
        for input in inputs.iter().filter(|input| input.epoch == epoch) {
            summed[input.node] += input.amplitude;
        }

        for contextual_edge in &network.edges {
            let enabled = match contextual_edge.scope {
                EdgeScope::Both => true,
                EdgeScope::RestingOnly => state == CircuitState::Resting,
                EdgeScope::ActiveOnly => state == CircuitState::Active,
            };
            let edge = contextual_edge.edge;
            if enabled && epoch >= edge.delay {
                summed[edge.target] += edge.weight * values[epoch - edge.delay][edge.source];
            }
        }

        for node in 0..network.node_count {
            values[epoch][node] = apply_transfer(network.transfers[node], summed[node])?;
        }
    }

    Ok(Trajectory { values })
}

fn with_interventions(
    baseline: &[Injection],
    interventions: &[Injection],
) -> Vec<Injection> {
    let mut combined = baseline.to_vec();
    combined.extend_from_slice(interventions);
    combined
}

/// Time-resolved two-factor interaction: Y(AB) - Y(A) - Y(B) + Y(0).
///
/// A nonzero value identifies non-additivity in this synthetic model under the
/// declared state and observation horizon. It does not identify a biological
/// molecular mechanism.
pub fn factorial_interaction_trace(
    network: &ContextualNetwork,
    state: CircuitState,
    horizon: usize,
    baseline_inputs: &[Injection],
    intervention_a: Injection,
    intervention_b: Injection,
    target: usize,
) -> Result<Vec<f64>, OracleError> {
    if target >= network.node_count {
        return Err(OracleError::TargetOutOfRange {
            node: target,
            node_count: network.node_count,
        });
    }
    let baseline = simulate_contextual(network, state, horizon, baseline_inputs)?;
    let only_a = simulate_contextual(
        network,
        state,
        horizon,
        &with_interventions(baseline_inputs, &[intervention_a]),
    )?;
    let only_b = simulate_contextual(
        network,
        state,
        horizon,
        &with_interventions(baseline_inputs, &[intervention_b]),
    )?;
    let both = simulate_contextual(
        network,
        state,
        horizon,
        &with_interventions(baseline_inputs, &[intervention_a, intervention_b]),
    )?;

    let mut interaction = Vec::with_capacity(horizon);
    for epoch in 0..horizon {
        let value = both.values[epoch][target]
            - only_a.values[epoch][target]
            - only_b.values[epoch][target]
            + baseline.values[epoch][target];
        if !value.is_finite() {
            return Err(OracleError::NonFiniteParameter);
        }
        interaction.push(value);
    }
    Ok(interaction)
}

/// Time-resolved third-order finite difference over three simultaneous
/// interventions, requiring all eight factorial arms.
pub fn third_order_interaction_trace(
    network: &ContextualNetwork,
    state: CircuitState,
    horizon: usize,
    baseline_inputs: &[Injection],
    intervention_a: Injection,
    intervention_b: Injection,
    intervention_c: Injection,
    target: usize,
) -> Result<Vec<f64>, OracleError> {
    if target >= network.node_count {
        return Err(OracleError::TargetOutOfRange {
            node: target,
            node_count: network.node_count,
        });
    }
    let run = |items: &[Injection]| -> Result<Trajectory, OracleError> {
        simulate_contextual(
            network,
            state,
            horizon,
            &with_interventions(baseline_inputs, items),
        )
    };

    let baseline = run(&[])?;
    let a = run(&[intervention_a])?;
    let b = run(&[intervention_b])?;
    let c = run(&[intervention_c])?;
    let ab = run(&[intervention_a, intervention_b])?;
    let ac = run(&[intervention_a, intervention_c])?;
    let bc = run(&[intervention_b, intervention_c])?;
    let abc = run(&[intervention_a, intervention_b, intervention_c])?;

    let mut interaction = Vec::with_capacity(horizon);
    for epoch in 0..horizon {
        let value = abc.values[epoch][target]
            - ab.values[epoch][target]
            - ac.values[epoch][target]
            - bc.values[epoch][target]
            + a.values[epoch][target]
            + b.values[epoch][target]
            + c.values[epoch][target]
            - baseline.values[epoch][target];
        if !value.is_finite() {
            return Err(OracleError::NonFiniteParameter);
        }
        interaction.push(value);
    }
    Ok(interaction)
}

fn combined_model_network(model: &CouplingModel) -> Result<Network, OracleError> {
    let mut edges = model.anatomy.edges.clone();
    for coupling in &model.extrasynaptic {
        if coupling.source >= model.anatomy.node_count {
            return Err(OracleError::NodeOutOfRange {
                node: coupling.source,
                node_count: model.anatomy.node_count,
            });
        }
        if coupling.target >= model.anatomy.node_count {
            return Err(OracleError::NodeOutOfRange {
                node: coupling.target,
                node_count: model.anatomy.node_count,
            });
        }
        if coupling.delay == 0 {
            return Err(OracleError::ZeroDelay);
        }
        if !coupling.weight.is_finite() {
            return Err(OracleError::NonFiniteParameter);
        }
        edges.push(Edge {
            source: coupling.source,
            target: coupling.target,
            weight: coupling.weight,
            delay: coupling.delay,
        });
    }
    Ok(Network {
        node_count: model.anatomy.node_count,
        edges,
    })
}

/// Simulate a declared phenomenological model with extrasynaptic terms.
///
/// The anatomical graph remains separately accessible on the model. Callers
/// must not write extrasynaptic terms back as structural edges.
pub fn simulate_with_extrasynaptic(
    model: &CouplingModel,
    horizon: usize,
    inputs: &[Injection],
) -> Result<Trajectory, OracleError> {
    let combined = combined_model_network(model)?;
    simulate(&combined, horizon, inputs)
}

/// Paired intervention replay for a model with distinct anatomical and
/// extrasynaptic channels. Only the intervention differs between the arms.
pub fn intervention_effect_with_extrasynaptic(
    model: &CouplingModel,
    horizon: usize,
    baseline_inputs: &[Injection],
    intervention: Injection,
) -> Result<Trajectory, OracleError> {
    let combined = combined_model_network(model)?;
    intervention_effect(&combined, horizon, baseline_inputs, intervention)
}

/// Estimate a paired finite-horizon intervention response by subtracting
/// baseline replay from replay with the additional intervention.
///
/// Baseline inputs are identical in both arms. This is a deterministic
/// difference-in-differences contract for the synthetic fixture, not a
/// biological causal estimator.
pub fn intervention_effect(
    network: &Network,
    horizon: usize,
    baseline_inputs: &[Injection],
    intervention: Injection,
) -> Result<Trajectory, OracleError> {
    let baseline = simulate(network, horizon, baseline_inputs)?;
    let mut intervention_inputs = baseline_inputs.to_vec();
    intervention_inputs.push(intervention);
    let treated = simulate(network, horizon, &intervention_inputs)?;

    let mut delta = Vec::with_capacity(horizon);
    for epoch in 0..horizon {
        let mut row = Vec::with_capacity(network.node_count);
        for node in 0..network.node_count {
            let value = treated.values[epoch][node] - baseline.values[epoch][node];
            if !value.is_finite() {
                return Err(OracleError::NonFiniteParameter);
            }
            row.push(value);
        }
        delta.push(row);
    }
    Ok(Trajectory { values: delta })
}


/// Exact synthetic factorial-interaction case.
#[derive(Debug, Clone, PartialEq)]
pub struct InteractionCaseResult {
    pub id: &'static str,
    pub context: &'static str,
    pub interaction: Vec<f64>,
    pub expected: Vec<f64>,
    pub passed: bool,
}

/// Six known-answer nonlinear cases: threshold synergy, additive null,
/// saturation, third-order synergy, and a state-gated edge in resting/active
/// contexts. These cases are mathematical fixtures, not biological claims.
pub fn interaction_suite() -> Result<Vec<InteractionCaseResult>, OracleError> {
    let both = EdgeScope::Both;
    let identity = NodeTransfer::Identity;
    let input_a = Injection { node: 0, epoch: 0, amplitude: 1.0 };
    let input_b = Injection { node: 1, epoch: 0, amplitude: 1.0 };
    let input_c = Injection { node: 2, epoch: 0, amplitude: 1.0 };

    let threshold_pair = ContextualNetwork {
        node_count: 3,
        edges: vec![
            ContextualEdge { edge: Edge { source: 0, target: 2, weight: 0.5, delay: 1 }, scope: both },
            ContextualEdge { edge: Edge { source: 1, target: 2, weight: 0.5, delay: 1 }, scope: both },
        ],
        transfers: vec![identity, identity, NodeTransfer::Threshold { threshold: 1.0 }],
    };
    let threshold_interaction = factorial_interaction_trace(
        &threshold_pair, CircuitState::Resting, 3, &[], input_a, input_b, 2,
    )?;

    let additive_pair = ContextualNetwork {
        node_count: 3,
        edges: threshold_pair.edges.clone(),
        transfers: vec![identity; 3],
    };
    let additive_interaction = factorial_interaction_trace(
        &additive_pair, CircuitState::Resting, 3, &[], input_a, input_b, 2,
    )?;

    let saturation = ContextualNetwork {
        node_count: 3,
        edges: vec![
            ContextualEdge { edge: Edge { source: 0, target: 2, weight: 1.0, delay: 1 }, scope: both },
            ContextualEdge { edge: Edge { source: 1, target: 2, weight: 1.0, delay: 1 }, scope: both },
        ],
        transfers: vec![
            identity,
            identity,
            NodeTransfer::Clamp { minimum: 0.0, maximum: 1.0 },
        ],
    };
    let saturation_interaction = factorial_interaction_trace(
        &saturation, CircuitState::Resting, 3, &[], input_a, input_b, 2,
    )?;

    let threshold_triple = ContextualNetwork {
        node_count: 4,
        edges: vec![
            ContextualEdge { edge: Edge { source: 0, target: 3, weight: 0.5, delay: 1 }, scope: both },
            ContextualEdge { edge: Edge { source: 1, target: 3, weight: 0.5, delay: 1 }, scope: both },
            ContextualEdge { edge: Edge { source: 2, target: 3, weight: 0.5, delay: 1 }, scope: both },
        ],
        transfers: vec![
            identity, identity, identity,
            NodeTransfer::Threshold { threshold: 1.5 },
        ],
    };
    let third_order = third_order_interaction_trace(
        &threshold_triple,
        CircuitState::Active,
        3,
        &[],
        input_a,
        input_b,
        input_c,
        3,
    )?;

    let state_gated = ContextualNetwork {
        node_count: 2,
        edges: vec![ContextualEdge {
            edge: Edge { source: 0, target: 1, weight: 0.75, delay: 1 },
            scope: EdgeScope::ActiveOnly,
        }],
        transfers: vec![identity; 2],
    };
    let state_input = [Injection { node: 0, epoch: 0, amplitude: 1.0 }];
    let resting = simulate_contextual(
        &state_gated, CircuitState::Resting, 3, &state_input,
    )?.node_trace(1)?;
    let active = simulate_contextual(
        &state_gated, CircuitState::Active, 3, &state_input,
    )?.node_trace(1)?;

    let mut cases = vec![
        ("pairwise_threshold_synergy", "resting", threshold_interaction, vec![0.0, 1.0, 0.0]),
        ("pairwise_additive_null", "resting", additive_interaction, vec![0.0, 0.0, 0.0]),
        ("saturation_subadditivity", "resting", saturation_interaction, vec![0.0, -1.0, 0.0]),
        ("third_order_threshold_synergy", "active", third_order, vec![0.0, 1.0, 0.0]),
        ("state_gated_edge_resting", "resting", resting, vec![0.0, 0.0, 0.0]),
        ("state_gated_edge_active", "active", active, vec![0.0, 0.75, 0.0]),
    ];

    Ok(cases
        .drain(..)
        .map(|(id, context, interaction, expected)| {
            let passed = interaction.len() == expected.len()
                && interaction.iter().zip(&expected)
                    .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-12);
            InteractionCaseResult { id, context, interaction, expected, passed }
        })
        .collect())
}

/// Deterministic JSON for the nonlinear/state-conditioned factorial suite.
pub fn interaction_suite_json() -> Result<String, OracleError> {
    let cases = interaction_suite()?;
    let mut json = String::from(
        "{\n  \"schema_version\": 1,\n  \"evidence_class\": \"synthetic_known_answer\",\n  \"claim_ceiling\": \"SYNTHETIC_ONLY\",\n  \"cases\": [\n",
    );
    for (index, case) in cases.iter().enumerate() {
        if index > 0 { json.push_str(",\n"); }
        let status = if case.passed { "PASS" } else { "FAIL" };
        json.push_str(&format!(
            "    {{\"id\":\"{}\",\"context\":\"{}\",\"interaction\":{},\"expected\":{},\"status\":\"{}\"}}",
            case.id, case.context, vector_json(&case.interaction),
            vector_json(&case.expected), status
        ));
    }
    json.push_str("\n  ]\n}\n");
    Ok(json)
}

/// Root-mean-square error for one node's complete finite-horizon response.
pub fn node_rmse(
    reference: &Trajectory,
    candidate: &Trajectory,
    node: usize,
) -> Result<f64, OracleError> {
    if reference.values.len() != candidate.values.len()
        || reference.values.is_empty()
        || reference
            .values
            .iter()
            .zip(&candidate.values)
            .any(|(left, right)| left.len() != right.len())
    {
        return Err(OracleError::ShapeMismatch);
    }

    let node_count = reference.values[0].len();
    if node >= node_count {
        return Err(OracleError::TargetOutOfRange { node, node_count });
    }

    let squared_error = reference
        .values
        .iter()
        .zip(&candidate.values)
        .map(|(left, right)| {
            let difference = left[node] - right[node];
            difference * difference
        })
        .sum::<f64>();
    let mse = squared_error / reference.values.len() as f64;
    if !mse.is_finite() {
        return Err(OracleError::NonFiniteParameter);
    }
    Ok(mse.sqrt())
}

/// First epoch whose absolute response exceeds a preregistered threshold.
pub fn first_detectable_epoch(
    trace: &[f64],
    threshold: f64,
) -> Result<Option<usize>, OracleError> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(OracleError::InvalidThreshold);
    }
    if trace.iter().any(|value| !value.is_finite()) {
        return Err(OracleError::NonFiniteParameter);
    }
    Ok(trace.iter().position(|value| value.abs() > threshold))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReachabilityInteraction {
    pub baseline_reachable: bool,
    pub without_first: bool,
    pub without_second: bool,
    pub without_both: bool,
    /// Binary-loss interaction: L(both) - L(first) - L(second) + L(none).
    /// +1 means the pair is jointly critical although neither edge is;
    /// -1 means both single deletions already destroy the query.
    pub interaction: i8,
}

/// Directed reachability after removing the listed edge indices.
///
/// This is a graph-query oracle, separate from the neural-state simulator.
/// It supports exact small-graph tests for redundant paths and bridge edges.
pub fn directed_path_exists(
    network: &Network,
    source: usize,
    target: usize,
    removed_edge_indices: &[usize],
) -> Result<bool, OracleError> {
    validate(network, 1, &[])?;
    if source >= network.node_count {
        return Err(OracleError::NodeOutOfRange {
            node: source,
            node_count: network.node_count,
        });
    }
    if target >= network.node_count {
        return Err(OracleError::NodeOutOfRange {
            node: target,
            node_count: network.node_count,
        });
    }

    let mut removed = vec![false; network.edges.len()];
    for &edge_index in removed_edge_indices {
        if edge_index >= network.edges.len() {
            return Err(OracleError::EdgeIndexOutOfRange {
                edge_index,
                edge_count: network.edges.len(),
            });
        }
        if removed[edge_index] {
            return Err(OracleError::DuplicateEdgeIndex);
        }
        removed[edge_index] = true;
    }

    if source == target {
        return Ok(true);
    }

    let mut visited = vec![false; network.node_count];
    let mut pending = std::collections::VecDeque::new();
    visited[source] = true;
    pending.push_back(source);

    while let Some(node) = pending.pop_front() {
        for (edge_index, edge) in network.edges.iter().enumerate() {
            if !removed[edge_index] && edge.source == node && !visited[edge.target] {
                if edge.target == target {
                    return Ok(true);
                }
                visited[edge.target] = true;
                pending.push_back(edge.target);
            }
        }
    }
    Ok(false)
}

/// Exact pairwise interaction for a binary directed-reachability query.
///
/// A true baseline query is required. This avoids reporting an interaction
/// around an already-failed baseline. The metric is a query-level topology
/// diagnostic, not a physiological causal effect.
pub fn pairwise_edge_reachability_interaction(
    network: &Network,
    source: usize,
    target: usize,
    first_edge: usize,
    second_edge: usize,
) -> Result<ReachabilityInteraction, OracleError> {
    if first_edge == second_edge {
        return Err(OracleError::DuplicateEdgeIndex);
    }

    let baseline_reachable = directed_path_exists(network, source, target, &[])?;
    if !baseline_reachable {
        return Err(OracleError::BaselineQueryUnreachable);
    }
    let without_first =
        directed_path_exists(network, source, target, &[first_edge])?;
    let without_second =
        directed_path_exists(network, source, target, &[second_edge])?;
    let without_both =
        directed_path_exists(network, source, target, &[first_edge, second_edge])?;

    let loss = |reachable: bool| if reachable { 0_i8 } else { 1_i8 };
    let interaction = loss(without_both) - loss(without_first) - loss(without_second)
        + loss(baseline_reachable);

    Ok(ReachabilityInteraction {
        baseline_reachable,
        without_first,
        without_second,
        without_both,
        interaction,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaseResult {
    pub id: &'static str,
    pub target_node: usize,
    pub reference_effect: Vec<f64>,
    pub candidate_effect: Vec<f64>,
    pub target_rmse: f64,
    pub negative_control: bool,
}

impl CaseResult {
    /// Positive cases require exact replay agreement; the rewired-graph
    /// negative control must be measurably distinguished from reference.
    pub fn passed(&self) -> bool {
        if self.negative_control {
            self.target_rmse > 1.0e-12
        } else {
            self.target_rmse <= 1.0e-12
        }
    }
}

fn make_network(node_count: usize, edges: &[(usize, usize, f64, usize)]) -> Network {
    Network {
        node_count,
        edges: edges
            .iter()
            .map(|&(source, target, weight, delay)| Edge {
                source,
                target,
                weight,
                delay,
            })
            .collect(),
    }
}

fn evaluate_case(
    id: &'static str,
    reference_network: &Network,
    candidate_network: &Network,
    horizon: usize,
    baseline_inputs: &[Injection],
    intervention: Injection,
    target_node: usize,
    negative_control: bool,
) -> Result<CaseResult, OracleError> {
    let reference = intervention_effect(
        reference_network,
        horizon,
        baseline_inputs,
        intervention,
    )?;
    let candidate = intervention_effect(
        candidate_network,
        horizon,
        baseline_inputs,
        intervention,
    )?;
    let reference_effect = reference.node_trace(target_node)?;
    let candidate_effect = candidate.node_trace(target_node)?;
    let target_rmse = node_rmse(&reference, &candidate, target_node)?;

    Ok(CaseResult {
        id,
        target_node,
        reference_effect,
        candidate_effect,
        target_rmse,
        negative_control,
    })
}

/// Six deterministic cases with exact expected behaviors:
/// direct, indirect, delayed, recurrent, shared-input confounding, and
/// a rewired-topology negative control.
pub fn reference_suite() -> Result<Vec<CaseResult>, OracleError> {
    let empty = &[];

    let direct = make_network(2, &[(0, 1, 0.5, 1)]);
    let indirect = make_network(3, &[(0, 1, 0.5, 1), (1, 2, -0.25, 1)]);
    let delayed = make_network(2, &[(0, 1, 0.8, 2)]);
    let recurrent = make_network(2, &[(0, 1, 1.0, 1), (1, 0, 0.5, 1)]);
    let no_edges = make_network(2, &[]);
    let common_inputs = [
        Injection {
            node: 0,
            epoch: 0,
            amplitude: 1.0,
        },
        Injection {
            node: 1,
            epoch: 0,
            amplitude: 1.0,
        },
    ];
    let rewired_truth = make_network(3, &[(0, 1, 0.9, 1)]);
    let rewired_candidate = make_network(3, &[(0, 2, 0.9, 1)]);

    Ok(vec![
        evaluate_case(
            "direct_one_step",
            &direct,
            &direct,
            4,
            empty,
            Injection {
                node: 0,
                epoch: 0,
                amplitude: 2.0,
            },
            1,
            false,
        )?,
        evaluate_case(
            "indirect_two_hop_sign",
            &indirect,
            &indirect,
            4,
            empty,
            Injection {
                node: 0,
                epoch: 0,
                amplitude: 2.0,
            },
            2,
            false,
        )?,
        evaluate_case(
            "explicit_two_epoch_delay",
            &delayed,
            &delayed,
            5,
            empty,
            Injection {
                node: 0,
                epoch: 0,
                amplitude: 2.0,
            },
            1,
            false,
        )?,
        evaluate_case(
            "recurrent_feedback",
            &recurrent,
            &recurrent,
            5,
            empty,
            Injection {
                node: 0,
                epoch: 0,
                amplitude: 2.0,
            },
            0,
            false,
        )?,
        evaluate_case(
            "shared_input_not_direct_causation",
            &no_edges,
            &no_edges,
            4,
            &common_inputs,
            Injection {
                node: 0,
                epoch: 0,
                amplitude: 1.0,
            },
            1,
            false,
        )?,
        evaluate_case(
            "rewired_topology_negative_control",
            &rewired_truth,
            &rewired_candidate,
            4,
            empty,
            Injection {
                node: 0,
                epoch: 0,
                amplitude: 1.0,
            },
            1,
            true,
        )?,
    ])
}

/// Render the reference suite as deterministic JSON without third-party
/// serialization dependencies. The format is declared in the companion schema.
pub fn reference_suite_json() -> Result<String, OracleError> {
    let cases = reference_suite()?;
    let mut json = String::from(
        "{\n  \"schema_version\": 1,\n  \"evidence_class\": \"synthetic_known_answer\",\n  \"claim_ceiling\": \"SYNTHETIC_ONLY\",\n  \"cases\": [\n",
    );

    for (index, case) in cases.iter().enumerate() {
        if index > 0 {
            json.push_str(",\n");
        }
        let status = if case.passed() { "PASS" } else { "FAIL" };
        let control = if case.negative_control {
            "rewired_topology_negative_control"
        } else {
            "exact_replay"
        };
        json.push_str(&format!(
            "    {{\"id\":\"{}\",\"target_node\":{},\"reference_effect\":{},\"candidate_effect\":{},\"target_rmse\":{:.12},\"control\":\"{}\",\"status\":\"{}\"}}",
            case.id,
            case.target_node,
            vector_json(&case.reference_effect),
            vector_json(&case.candidate_effect),
            case.target_rmse,
            control,
            status
        ));
    }
    json.push_str("\n  ]\n}\n");
    Ok(json)
}

fn vector_json(values: &[f64]) -> String {
    let mut json = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        json.push_str(&format!("{value:.12}"));
    }
    json.push(']');
    json
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(values: &[f64]) -> Vec<f64> {
        values.to_vec()
    }

    #[test]
    fn direct_one_step_response_has_exact_amplitude_and_latency() {
        let cases = reference_suite().expect("suite is valid");
        let case = cases.iter().find(|case| case.id == "direct_one_step").unwrap();
        assert_eq!(case.reference_effect, trace(&[0.0, 1.0, 0.0, 0.0]));
        assert_eq!(
            first_detectable_epoch(&case.reference_effect, 1.0e-12).unwrap(),
            Some(1)
        );
        assert!(case.passed());
    }

    #[test]
    fn indirect_path_is_delayed_and_preserves_signed_product() {
        let cases = reference_suite().expect("suite is valid");
        let case = cases
            .iter()
            .find(|case| case.id == "indirect_two_hop_sign")
            .unwrap();
        assert_eq!(case.reference_effect, trace(&[0.0, 0.0, -0.25, 0.0]));
        assert_eq!(
            first_detectable_epoch(&case.reference_effect, 1.0e-12).unwrap(),
            Some(2)
        );
        assert!(case.passed());
    }

    #[test]
    fn explicit_edge_delay_is_not_mistaken_for_path_length() {
        let cases = reference_suite().expect("suite is valid");
        let case = cases
            .iter()
            .find(|case| case.id == "explicit_two_epoch_delay")
            .unwrap();
        assert_eq!(case.reference_effect, trace(&[0.0, 0.0, 1.6, 0.0, 0.0]));
        assert!(case.passed());
    }

    #[test]
    fn recurrent_feedback_produces_a_finite_impulse_response() {
        let cases = reference_suite().expect("suite is valid");
        let case = cases
            .iter()
            .find(|case| case.id == "recurrent_feedback")
            .unwrap();
        assert_eq!(case.reference_effect, trace(&[2.0, 0.0, 1.0, 0.0, 0.5]));
        assert!(case.passed());
    }

    #[test]
    fn shared_input_correlation_does_not_create_interventional_effect() {
        let network = make_network(2, &[]);
        let common_inputs = [
            Injection {
                node: 0,
                epoch: 0,
                amplitude: 1.0,
            },
            Injection {
                node: 1,
                epoch: 0,
                amplitude: 1.0,
            },
        ];
        let baseline = simulate(&network, 3, &common_inputs).unwrap();
        assert_eq!(baseline.values[0][0], baseline.values[0][1]);

        let effect = intervention_effect(
            &network,
            3,
            &common_inputs,
            Injection {
                node: 0,
                epoch: 0,
                amplitude: 1.0,
            },
        )
        .unwrap();
        assert_eq!(effect.node_trace(1).unwrap(), trace(&[0.0, 0.0, 0.0]));
        assert_eq!(effect.node_trace(0).unwrap(), trace(&[1.0, 0.0, 0.0]));
    }

    #[test]
    fn rewired_topology_is_detected_by_the_negative_control() {
        let cases = reference_suite().expect("suite is valid");
        let case = cases
            .iter()
            .find(|case| case.id == "rewired_topology_negative_control")
            .unwrap();
        assert_eq!(case.reference_effect, trace(&[0.0, 0.9, 0.0, 0.0]));
        assert_eq!(case.candidate_effect, trace(&[0.0, 0.0, 0.0, 0.0]));
        assert!((case.target_rmse - 0.45).abs() < 1.0e-12);
        assert!(case.passed());
    }

    #[test]
    fn suite_replay_is_byte_deterministic() {
        assert_eq!(reference_suite_json().unwrap(), reference_suite_json().unwrap());
    }

    #[test]
    fn invalid_zero_delay_fails_closed() {
        let network = make_network(2, &[(0, 1, 1.0, 0)]);
        let result = simulate(&network, 2, &[]);
        assert_eq!(result, Err(OracleError::ZeroDelay));
    }

    #[test]
    fn invalid_non_finite_weight_fails_closed() {
        let network = make_network(2, &[(0, 1, f64::NAN, 1)]);
        let result = simulate(&network, 2, &[]);
        assert_eq!(result, Err(OracleError::NonFiniteParameter));
    }

    #[test]
    fn out_of_range_injection_fails_closed() {
        let network = make_network(2, &[]);
        let result = simulate(
            &network,
            2,
            &[Injection {
                node: 2,
                epoch: 0,
                amplitude: 1.0,
            }],
        );
        assert!(matches!(
            result,
            Err(OracleError::NodeOutOfRange { node: 2, node_count: 2 })
        ));
    }

    #[test]
    fn invalid_horizon_and_threshold_fail_closed() {
        let network = make_network(1, &[]);
        assert_eq!(simulate(&network, 0, &[]), Err(OracleError::ZeroHorizon));
        assert_eq!(
            first_detectable_epoch(&[0.0, 1.0], f64::NAN),
            Err(OracleError::InvalidThreshold)
        );
    }

    #[test]
    fn rmse_rejects_incompatible_shapes() {
        let left = Trajectory {
            values: vec![vec![0.0, 1.0]],
        };
        let right = Trajectory {
            values: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
        };
        assert_eq!(node_rmse(&left, &right, 0), Err(OracleError::ShapeMismatch));
    }


    #[test]
    fn redundant_parallel_paths_have_positive_pairwise_interaction() {
        let network = make_network(
            4,
            &[(0, 1, 1.0, 1), (1, 3, 1.0, 1), (0, 2, 1.0, 1), (2, 3, 1.0, 1)],
        );
        let result = pairwise_edge_reachability_interaction(&network, 0, 3, 0, 2).unwrap();
        assert!(result.baseline_reachable);
        assert!(result.without_first);
        assert!(result.without_second);
        assert!(!result.without_both);
        assert_eq!(result.interaction, 1);
    }

    #[test]
    fn serial_bottleneck_edges_have_negative_pairwise_interaction() {
        let network = make_network(3, &[(0, 1, 1.0, 1), (1, 2, 1.0, 1)]);
        let result = pairwise_edge_reachability_interaction(&network, 0, 2, 0, 1).unwrap();
        assert!(result.baseline_reachable);
        assert!(!result.without_first);
        assert!(!result.without_second);
        assert!(!result.without_both);
        assert_eq!(result.interaction, -1);
    }

    #[test]
    fn graph_query_rejects_duplicate_or_out_of_range_edge_indices() {
        let network = make_network(2, &[(0, 1, 1.0, 1)]);
        assert_eq!(
            directed_path_exists(&network, 0, 1, &[0, 0]),
            Err(OracleError::DuplicateEdgeIndex)
        );
        assert_eq!(
            directed_path_exists(&network, 0, 1, &[1]),
            Err(OracleError::EdgeIndexOutOfRange {
                edge_index: 1,
                edge_count: 1
            })
        );
    }

    #[test]
    fn pairwise_interaction_requires_reachable_baseline_and_distinct_edges() {
        let network = make_network(3, &[(0, 1, 1.0, 1), (1, 2, 1.0, 1)]);
        assert_eq!(
            pairwise_edge_reachability_interaction(&network, 0, 2, 0, 0),
            Err(OracleError::DuplicateEdgeIndex)
        );
        assert_eq!(
            pairwise_edge_reachability_interaction(&network, 2, 0, 0, 1),
            Err(OracleError::BaselineQueryUnreachable)
        );
    }


    #[test]
    fn extrasynaptic_effect_can_exist_without_an_anatomical_direct_edge() {
        let model = CouplingModel {
            anatomy: make_network(2, &[]),
            extrasynaptic: vec![ExtrasynapticCoupling {
                source: 0,
                target: 1,
                weight: 0.4,
                delay: 1,
            }],
        };
        assert!(!directed_path_exists(&model.anatomy, 0, 1, &[]).unwrap());

        let effect = intervention_effect_with_extrasynaptic(
            &model,
            4,
            &[],
            Injection {
                node: 0,
                epoch: 0,
                amplitude: 2.0,
            },
        )
        .unwrap();
        assert_eq!(effect.node_trace(1).unwrap(), trace(&[0.0, 0.8, 0.0, 0.0]));
    }

    #[test]
    fn anatomy_only_simulation_does_not_silently_include_extrasynaptic_terms() {
        let model = CouplingModel {
            anatomy: make_network(2, &[]),
            extrasynaptic: vec![ExtrasynapticCoupling {
                source: 0,
                target: 1,
                weight: 1.0,
                delay: 1,
            }],
        };
        let input = [Injection {
            node: 0,
            epoch: 0,
            amplitude: 1.0,
        }];
        let anatomical = simulate(&model.anatomy, 3, &input).unwrap();
        let extended = simulate_with_extrasynaptic(&model, 3, &input).unwrap();
        assert_eq!(anatomical.node_trace(1).unwrap(), trace(&[0.0, 0.0, 0.0]));
        assert_eq!(extended.node_trace(1).unwrap(), trace(&[0.0, 1.0, 0.0]));
    }

    #[test]
    fn invalid_extrasynaptic_coupling_fails_closed() {
        let model = CouplingModel {
            anatomy: make_network(2, &[]),
            extrasynaptic: vec![ExtrasynapticCoupling {
                source: 0,
                target: 2,
                weight: 1.0,
                delay: 1,
            }],
        };
        assert_eq!(
            simulate_with_extrasynaptic(&model, 3, &[]),
            Err(OracleError::NodeOutOfRange {
                node: 2,
                node_count: 2
            })
        );
    }


    #[test]
    fn json_report_matches_the_pinned_golden_fixture_byte_for_byte() {
        let expected = include_str!("../fixtures/reference-suite.expected.json");
        assert_eq!(reference_suite_json().unwrap(), expected);
    }


    #[test]
    fn threshold_gate_exposes_pairwise_synergy_missed_by_single_target_trials() {
        let network = ContextualNetwork {
            node_count: 3,
            edges: vec![
                ContextualEdge {
                    edge: Edge { source: 0, target: 2, weight: 0.5, delay: 1 },
                    scope: EdgeScope::Both,
                },
                ContextualEdge {
                    edge: Edge { source: 1, target: 2, weight: 0.5, delay: 1 },
                    scope: EdgeScope::Both,
                },
            ],
            transfers: vec![
                NodeTransfer::Identity,
                NodeTransfer::Identity,
                NodeTransfer::Threshold { threshold: 1.0 },
            ],
        };
        let a = Injection { node: 0, epoch: 0, amplitude: 1.0 };
        let b = Injection { node: 1, epoch: 0, amplitude: 1.0 };
        let interaction = factorial_interaction_trace(
            &network, CircuitState::Resting, 3, &[], a, b, 2,
        ).unwrap();
        assert_eq!(interaction, trace(&[0.0, 1.0, 0.0]));
    }

    #[test]
    fn additive_linear_network_has_zero_factorial_interaction() {
        let network = ContextualNetwork {
            node_count: 3,
            edges: vec![
                ContextualEdge {
                    edge: Edge { source: 0, target: 2, weight: 0.5, delay: 1 },
                    scope: EdgeScope::Both,
                },
                ContextualEdge {
                    edge: Edge { source: 1, target: 2, weight: 0.5, delay: 1 },
                    scope: EdgeScope::Both,
                },
            ],
            transfers: vec![NodeTransfer::Identity; 3],
        };
        let a = Injection { node: 0, epoch: 0, amplitude: 1.0 };
        let b = Injection { node: 1, epoch: 0, amplitude: 1.0 };
        let interaction = factorial_interaction_trace(
            &network, CircuitState::Resting, 3, &[], a, b, 2,
        ).unwrap();
        assert_eq!(interaction, trace(&[0.0, 0.0, 0.0]));
    }

    #[test]
    fn saturation_produces_negative_pairwise_interaction() {
        let network = ContextualNetwork {
            node_count: 3,
            edges: vec![
                ContextualEdge {
                    edge: Edge { source: 0, target: 2, weight: 1.0, delay: 1 },
                    scope: EdgeScope::Both,
                },
                ContextualEdge {
                    edge: Edge { source: 1, target: 2, weight: 1.0, delay: 1 },
                    scope: EdgeScope::Both,
                },
            ],
            transfers: vec![
                NodeTransfer::Identity,
                NodeTransfer::Identity,
                NodeTransfer::Clamp { minimum: 0.0, maximum: 1.0 },
            ],
        };
        let a = Injection { node: 0, epoch: 0, amplitude: 1.0 };
        let b = Injection { node: 1, epoch: 0, amplitude: 1.0 };
        let interaction = factorial_interaction_trace(
            &network, CircuitState::Resting, 3, &[], a, b, 2,
        ).unwrap();
        assert_eq!(interaction, trace(&[0.0, -1.0, 0.0]));
    }

    #[test]
    fn three_way_threshold_gate_has_nonzero_third_order_interaction() {
        let network = ContextualNetwork {
            node_count: 4,
            edges: vec![
                ContextualEdge {
                    edge: Edge { source: 0, target: 3, weight: 0.5, delay: 1 },
                    scope: EdgeScope::Both,
                },
                ContextualEdge {
                    edge: Edge { source: 1, target: 3, weight: 0.5, delay: 1 },
                    scope: EdgeScope::Both,
                },
                ContextualEdge {
                    edge: Edge { source: 2, target: 3, weight: 0.5, delay: 1 },
                    scope: EdgeScope::Both,
                },
            ],
            transfers: vec![
                NodeTransfer::Identity,
                NodeTransfer::Identity,
                NodeTransfer::Identity,
                NodeTransfer::Threshold { threshold: 1.5 },
            ],
        };
        let a = Injection { node: 0, epoch: 0, amplitude: 1.0 };
        let b = Injection { node: 1, epoch: 0, amplitude: 1.0 };
        let c = Injection { node: 2, epoch: 0, amplitude: 1.0 };
        let interaction = third_order_interaction_trace(
            &network, CircuitState::Active, 3, &[], a, b, c, 3,
        ).unwrap();
        assert_eq!(interaction, trace(&[0.0, 1.0, 0.0]));
    }

    #[test]
    fn state_gated_edge_changes_response_without_changing_topology() {
        let network = ContextualNetwork {
            node_count: 2,
            edges: vec![ContextualEdge {
                edge: Edge { source: 0, target: 1, weight: 0.75, delay: 1 },
                scope: EdgeScope::ActiveOnly,
            }],
            transfers: vec![NodeTransfer::Identity; 2],
        };
        let input = [Injection { node: 0, epoch: 0, amplitude: 1.0 }];
        let resting = simulate_contextual(&network, CircuitState::Resting, 3, &input).unwrap();
        let active = simulate_contextual(&network, CircuitState::Active, 3, &input).unwrap();
        assert_eq!(resting.node_trace(1).unwrap(), trace(&[0.0, 0.0, 0.0]));
        assert_eq!(active.node_trace(1).unwrap(), trace(&[0.0, 0.75, 0.0]));
        assert_eq!(network.edges.len(), 1);
    }

    #[test]
    fn contextual_network_rejects_transfer_count_and_bad_clamp() {
        let mismatch = ContextualNetwork {
            node_count: 2,
            edges: vec![],
            transfers: vec![NodeTransfer::Identity],
        };
        assert_eq!(
            simulate_contextual(&mismatch, CircuitState::Resting, 2, &[]),
            Err(OracleError::TransferCountMismatch { expected: 2, actual: 1 })
        );

        let bad_clamp = ContextualNetwork {
            node_count: 1,
            edges: vec![],
            transfers: vec![NodeTransfer::Clamp { minimum: 2.0, maximum: 1.0 }],
        };
        assert_eq!(
            simulate_contextual(&bad_clamp, CircuitState::Resting, 2, &[]),
            Err(OracleError::InvalidTransferParameters)
        );
    }


    #[test]
    fn nonlinear_interaction_suite_matches_all_known_answers() {
        let cases = interaction_suite().expect("interaction suite is valid");
        assert_eq!(cases.len(), 6);
        assert!(cases.iter().all(|case| case.passed));
    }

    #[test]
    fn nonlinear_interaction_report_is_byte_deterministic() {
        assert_eq!(interaction_suite_json().unwrap(), interaction_suite_json().unwrap());
    }

    #[test]
    fn all_reference_cases_meet_their_declared_control() {
        let cases = reference_suite().expect("suite is valid");
        assert_eq!(cases.len(), 6);
        assert!(cases.iter().all(CaseResult::passed));
    }
}
