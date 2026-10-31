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
    fn all_reference_cases_meet_their_declared_control() {
        let cases = reference_suite().expect("suite is valid");
        assert_eq!(cases.len(), 6);
        assert!(cases.iter().all(CaseResult::passed));
    }
}
