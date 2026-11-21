//! Exact small-N balanced W1 reference for nonnegative phase-space measures.
//!
//! The declared domain is a periodic cylinder S^1_L x R. Ground cost is the
//! Euclidean product of shortest-arc position distance and scaled velocity
//! distance. Signed delta-f measures are rejected and remain on a separate
//! characteristic-MMD lane.
//!
//! A successive-shortest-augmenting-path min-cost-flow solver evaluates the
//! discrete Kantorovich linear program. This is a correctness reference for
//! small supports, not a production-scale Sinkhorn implementation.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

/// One nonnegative macro-particle in periodic-x / unbounded-v phase space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseSpaceParticle1d {
    pub position: f64,
    pub velocity: f64,
    pub weight: f64,
}

/// Frozen ground metric and mass-match contract.
///
/// Both measures are normalized to probability mass after their original
/// total masses are checked against maximum_relative_mass_mismatch.
/// Original masses and their relative mismatch are always returned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransportConfig {
    pub domain_length: f64,
    pub position_length_scale: f64,
    pub velocity_length_scale: f64,
    pub maximum_relative_mass_mismatch: f64,
}

/// Result of exact-small-N balanced 1-Wasserstein transport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransportResult {
    /// W1 between the two probability-normalized empirical measures.
    pub normalized_w1: f64,
    pub normalized_transport_cost: f64,
    pub candidate_total_mass: f64,
    pub reference_total_mass: f64,
    pub relative_mass_mismatch: f64,
    /// Probability mass successfully transported (one for nonempty measures).
    pub transported_fraction: f64,
    pub candidate_particle_count: usize,
    pub reference_particle_count: usize,
    pub candidate_atom_count: usize,
    pub reference_atom_count: usize,
    pub ground_cost_evaluations: u128,
    pub augmenting_path_count: usize,
    pub shortest_path_relaxations: u128,
}

/// Invalid metric, unsupported measure or failed numerical optimization.
#[derive(Debug, Clone, PartialEq)]
pub enum TransportError {
    InvalidDomainLength,
    InvalidPositionScale,
    InvalidVelocityScale,
    InvalidMassTolerance,
    NegativeWeight {
        measure: &'static str,
        particle: usize,
    },
    NonFinitePosition {
        measure: &'static str,
        particle: usize,
    },
    NonFiniteVelocity {
        measure: &'static str,
        particle: usize,
    },
    NonFiniteWeight {
        measure: &'static str,
        particle: usize,
    },
    TotalMassOverflow {
        measure: &'static str,
    },
    RelativeMassMismatch {
        value: f64,
        maximum: f64,
    },
    NonFiniteGroundCost,
    NoAugmentingPath,
    NoProgress,
    IterationLimitExceeded,
    NonFiniteOutput,
    MateriallyNegativeTransportCost {
        value: f64,
        tolerance: f64,
    },
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDomainLength => write!(f, "domain length must be finite and positive"),
            Self::InvalidPositionScale => {
                write!(f, "position length scale must be finite and positive")
            }
            Self::InvalidVelocityScale => {
                write!(f, "velocity length scale must be finite and positive")
            }
            Self::InvalidMassTolerance => write!(
                f,
                "relative mass mismatch tolerance must be finite and in [0, 1e-6]"
            ),
            Self::NegativeWeight { measure, particle } => {
                write!(
                    f,
                    "{measure} particle {particle} has a negative weight; signed OT is unsupported"
                )
            }
            Self::NonFinitePosition { measure, particle } => {
                write!(f, "{measure} particle {particle} has a non-finite position")
            }
            Self::NonFiniteVelocity { measure, particle } => {
                write!(f, "{measure} particle {particle} has a non-finite velocity")
            }
            Self::NonFiniteWeight { measure, particle } => {
                write!(f, "{measure} particle {particle} has a non-finite weight")
            }
            Self::TotalMassOverflow { measure } => write!(f, "{measure} total mass overflowed"),
            Self::RelativeMassMismatch { value, maximum } => {
                write!(
                    f,
                    "relative total-mass mismatch {value:e} exceeds maximum {maximum:e}"
                )
            }
            Self::NonFiniteGroundCost => write!(f, "ground metric produced a non-finite cost"),
            Self::NoAugmentingPath => write!(f, "transport residual graph has no augmenting path"),
            Self::NoProgress => write!(f, "transport solver made no positive progress"),
            Self::IterationLimitExceeded => {
                write!(f, "transport solver exceeded its augmenting-path limit")
            }
            Self::NonFiniteOutput => write!(f, "transport solver produced a non-finite output"),
            Self::MateriallyNegativeTransportCost { value, tolerance } => {
                write!(
                    f,
                    "transport objective is materially negative ({value:e}; tolerance {tolerance:e})"
                )
            }
        }
    }
}

impl Error for TransportError {}

#[derive(Debug, Default, Clone, Copy)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

#[derive(Debug, Clone, Copy)]
struct WeightedAtom {
    position: f64,
    velocity: f64,
    weight: f64,
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn validate_config(config: TransportConfig) -> Result<(), TransportError> {
    if !config.domain_length.is_finite() || config.domain_length <= 0.0 {
        return Err(TransportError::InvalidDomainLength);
    }
    if !config.position_length_scale.is_finite() || config.position_length_scale <= 0.0 {
        return Err(TransportError::InvalidPositionScale);
    }
    if !config.velocity_length_scale.is_finite() || config.velocity_length_scale <= 0.0 {
        return Err(TransportError::InvalidVelocityScale);
    }
    if !config.maximum_relative_mass_mismatch.is_finite()
        || !(0.0..=1.0e-6).contains(&config.maximum_relative_mass_mismatch)
    {
        return Err(TransportError::InvalidMassTolerance);
    }
    Ok(())
}

/// Canonicalize periodic coordinates and aggregate co-located nonnegative
/// weights before measuring total mass.
fn aggregate_positive_measure(
    particles: &[PhaseSpaceParticle1d],
    measure: &'static str,
    domain_length: f64,
) -> Result<(Vec<WeightedAtom>, f64), TransportError> {
    let mut atoms = Vec::with_capacity(particles.len());
    for (index, particle) in particles.iter().enumerate() {
        if !particle.position.is_finite() {
            return Err(TransportError::NonFinitePosition {
                measure,
                particle: index,
            });
        }
        if !particle.velocity.is_finite() {
            return Err(TransportError::NonFiniteVelocity {
                measure,
                particle: index,
            });
        }
        if !particle.weight.is_finite() {
            return Err(TransportError::NonFiniteWeight {
                measure,
                particle: index,
            });
        }
        if particle.weight < 0.0 {
            return Err(TransportError::NegativeWeight {
                measure,
                particle: index,
            });
        }
        if particle.weight > 0.0 {
            atoms.push(WeightedAtom {
                position: canonical_zero(particle.position.rem_euclid(domain_length)),
                velocity: canonical_zero(particle.velocity),
                weight: particle.weight,
            });
        }
    }

    atoms.sort_by(|left, right| {
        left.position
            .total_cmp(&right.position)
            .then_with(|| left.velocity.total_cmp(&right.velocity))
            .then_with(|| left.weight.total_cmp(&right.weight))
    });

    let mut aggregate = Vec::with_capacity(atoms.len());
    let mut total_mass = CompensatedSum::default();
    let mut index = 0;
    while index < atoms.len() {
        let position = atoms[index].position;
        let velocity = atoms[index].velocity;
        let mut weight = CompensatedSum::default();
        while index < atoms.len()
            && atoms[index].position == position
            && atoms[index].velocity == velocity
        {
            weight.add(atoms[index].weight);
            index += 1;
        }
        let weight = weight.total();
        if !weight.is_finite() {
            return Err(TransportError::TotalMassOverflow { measure });
        }
        if weight > 0.0 {
            total_mass.add(weight);
            aggregate.push(WeightedAtom {
                position,
                velocity,
                weight,
            });
        }
    }

    let total_mass = total_mass.total();
    if !total_mass.is_finite() {
        return Err(TransportError::TotalMassOverflow { measure });
    }
    Ok((aggregate, total_mass))
}

fn ground_cost(
    left: WeightedAtom,
    right: WeightedAtom,
    config: TransportConfig,
) -> Result<f64, TransportError> {
    let raw_position_delta = (left.position - right.position).abs();
    let periodic_delta =
        raw_position_delta.min((config.domain_length - raw_position_delta).max(0.0));
    let velocity_delta = (left.velocity - right.velocity).abs();
    let scaled_position = periodic_delta / config.position_length_scale;
    let scaled_velocity = velocity_delta / config.velocity_length_scale;
    let distance = scaled_position.hypot(scaled_velocity);
    if !distance.is_finite() {
        return Err(TransportError::NonFiniteGroundCost);
    }
    Ok(distance)
}

#[derive(Debug, Clone)]
struct FlowEdge {
    to: usize,
    reverse_index: usize,
    residual_capacity: f64,
    unit_cost: f64,
}

fn add_edge(graph: &mut [Vec<FlowEdge>], from: usize, to: usize, capacity: f64, unit_cost: f64) {
    let forward_reverse_index = graph[to].len();
    let reverse_reverse_index = graph[from].len();
    graph[from].push(FlowEdge {
        to,
        reverse_index: forward_reverse_index,
        residual_capacity: capacity,
        unit_cost,
    });
    graph[to].push(FlowEdge {
        to: from,
        reverse_index: reverse_reverse_index,
        residual_capacity: 0.0,
        unit_cost: -unit_cost,
    });
}

#[derive(Debug)]
struct ResidualShortestPath {
    distances: Vec<f64>,
    predecessors: Vec<Option<(usize, usize)>>,
    relaxations: u128,
}

/// Bellman-Ford shortest path over the residual network. Reverse arcs have
/// negative costs, so Dijkstra without valid potentials is not sufficient.
fn shortest_residual_path(
    graph: &[Vec<FlowEdge>],
    source: usize,
    sink: usize,
) -> Result<ResidualShortestPath, TransportError> {
    let node_count = graph.len();
    let mut distance = vec![f64::INFINITY; node_count];
    let mut predecessor = vec![None; node_count];
    let mut relaxations = 0_u128;
    distance[source] = 0.0;

    for _ in 0..node_count.saturating_sub(1) {
        let mut changed = false;
        for (from, edges) in graph.iter().enumerate() {
            if !distance[from].is_finite() {
                continue;
            }
            for (edge_index, edge) in edges.iter().enumerate() {
                if edge.residual_capacity <= 0.0 {
                    continue;
                }
                let candidate = distance[from] + edge.unit_cost;
                relaxations += 1;
                if candidate < distance[edge.to] {
                    distance[edge.to] = candidate;
                    predecessor[edge.to] = Some((from, edge_index));
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    if !distance[sink].is_finite() {
        return Err(TransportError::NoAugmentingPath);
    }
    Ok(ResidualShortestPath {
        distances: distance,
        predecessors: predecessor,
        relaxations,
    })
}

fn min_cost_transport(
    supplies: &[f64],
    demands: &[f64],
    costs: &[Vec<f64>],
) -> Result<(f64, usize, u128), TransportError> {
    let supply_count = supplies.len();
    let demand_count = demands.len();
    if supply_count == 0 && demand_count == 0 {
        return Ok((0.0, 0, 0));
    }
    if supply_count == 0 || demand_count == 0 {
        return Err(TransportError::NoAugmentingPath);
    }

    let source = 0;
    let left_start = 1;
    let right_start = left_start + supply_count;
    let sink = right_start + demand_count;
    let mut graph = vec![Vec::<FlowEdge>::new(); sink + 1];

    for (index, supply) in supplies.iter().enumerate() {
        add_edge(&mut graph, source, left_start + index, *supply, 0.0);
    }
    for (i, cost_row) in costs.iter().enumerate() {
        for (j, cost) in cost_row.iter().enumerate() {
            add_edge(
                &mut graph,
                left_start + i,
                right_start + j,
                f64::INFINITY,
                *cost,
            );
        }
    }
    for (index, demand) in demands.iter().enumerate() {
        add_edge(&mut graph, right_start + index, sink, *demand, 0.0);
    }

    let mut total_flow = 0.0;
    let mut total_cost = CompensatedSum::default();
    let mut augmenting_paths = 0_usize;
    let mut shortest_path_relaxations = 0_u128;
    const TARGET_FLOW: f64 = 1.0;
    const FLOW_TOLERANCE: f64 = 1.0e-12;
    const MAX_AUGMENTATIONS: usize = 1_000_000;

    while TARGET_FLOW - total_flow > FLOW_TOLERANCE {
        if augmenting_paths >= MAX_AUGMENTATIONS {
            return Err(TransportError::IterationLimitExceeded);
        }
        let shortest_path = shortest_residual_path(&graph, source, sink)?;
        shortest_path_relaxations = shortest_path_relaxations
            .checked_add(shortest_path.relaxations)
            .ok_or(TransportError::NonFiniteOutput)?;
        let distance = shortest_path.distances;
        let predecessor = shortest_path.predecessors;

        let mut amount = TARGET_FLOW - total_flow;
        let mut node = sink;
        while node != source {
            let (previous, edge_index) =
                predecessor[node].ok_or(TransportError::NoAugmentingPath)?;
            amount = amount.min(graph[previous][edge_index].residual_capacity);
            node = previous;
        }
        if !amount.is_finite() || amount <= 0.0 {
            return Err(TransportError::NoProgress);
        }

        node = sink;
        while node != source {
            let (previous, edge_index) =
                predecessor[node].ok_or(TransportError::NoAugmentingPath)?;
            let (to, reverse_index, unit_cost) = {
                let edge = &graph[previous][edge_index];
                (edge.to, edge.reverse_index, edge.unit_cost)
            };
            graph[previous][edge_index].residual_capacity =
                (graph[previous][edge_index].residual_capacity - amount).max(0.0);
            graph[to][reverse_index].residual_capacity += amount;
            if !unit_cost.is_finite() {
                return Err(TransportError::NonFiniteOutput);
            }
            node = previous;
        }

        total_flow += amount;
        total_cost.add(amount * distance[sink]);
        augmenting_paths += 1;
    }

    if !total_flow.is_finite()
        || (TARGET_FLOW - total_flow).abs() > 1.0e-10
        || !total_cost.total().is_finite()
    {
        return Err(TransportError::NonFiniteOutput);
    }

    let raw_cost = total_cost.total();
    let roundoff_tolerance = 256.0 * f64::EPSILON * (raw_cost.abs() + 1.0);
    if raw_cost < -roundoff_tolerance {
        return Err(TransportError::MateriallyNegativeTransportCost {
            value: raw_cost,
            tolerance: roundoff_tolerance,
        });
    }
    Ok((
        raw_cost.max(0.0),
        augmenting_paths,
        shortest_path_relaxations,
    ))
}

/// Compute balanced W1 for finite nonnegative phase-space measures.
///
/// The input total masses must agree within the configured relative tolerance.
/// Both are then normalized to probability mass for the W1 calculation;
/// original mass and mismatch are returned explicitly. Signed delta-f inputs
/// are rejected instead of silently taking absolute values.
///
/// This small-N reference uses successive shortest paths and Bellman-Ford on
/// a dense bipartite residual graph; it is not a production Sinkhorn solver.
pub fn compare_phase_space_measures_w1(
    candidate: &[PhaseSpaceParticle1d],
    reference: &[PhaseSpaceParticle1d],
    config: TransportConfig,
) -> Result<TransportResult, TransportError> {
    validate_config(config)?;
    let (candidate_atoms, candidate_total_mass) =
        aggregate_positive_measure(candidate, "candidate", config.domain_length)?;
    let (reference_atoms, reference_total_mass) =
        aggregate_positive_measure(reference, "reference", config.domain_length)?;

    let denominator = candidate_total_mass.max(reference_total_mass);
    let relative_mass_mismatch = if denominator == 0.0 {
        0.0
    } else {
        (candidate_total_mass - reference_total_mass).abs() / denominator
    };
    if relative_mass_mismatch > config.maximum_relative_mass_mismatch {
        return Err(TransportError::RelativeMassMismatch {
            value: relative_mass_mismatch,
            maximum: config.maximum_relative_mass_mismatch,
        });
    }

    if candidate_total_mass == 0.0 && reference_total_mass == 0.0 {
        return Ok(TransportResult {
            normalized_w1: 0.0,
            normalized_transport_cost: 0.0,
            candidate_total_mass,
            reference_total_mass,
            relative_mass_mismatch,
            transported_fraction: 0.0,
            candidate_particle_count: candidate.len(),
            reference_particle_count: reference.len(),
            candidate_atom_count: 0,
            reference_atom_count: 0,
            ground_cost_evaluations: 0,
            augmenting_path_count: 0,
            shortest_path_relaxations: 0,
        });
    }

    if candidate_total_mass == 0.0 || reference_total_mass == 0.0 {
        return Err(TransportError::RelativeMassMismatch {
            value: relative_mass_mismatch,
            maximum: config.maximum_relative_mass_mismatch,
        });
    }

    let supplies = candidate_atoms
        .iter()
        .map(|atom| atom.weight / candidate_total_mass)
        .collect::<Vec<_>>();
    let demands = reference_atoms
        .iter()
        .map(|atom| atom.weight / reference_total_mass)
        .collect::<Vec<_>>();

    let mut costs = vec![vec![0.0; reference_atoms.len()]; candidate_atoms.len()];
    for (i, left) in candidate_atoms.iter().enumerate() {
        for (j, right) in reference_atoms.iter().enumerate() {
            costs[i][j] = ground_cost(*left, *right, config)?;
        }
    }

    let ground_cost_evaluations = (candidate_atoms.len() as u128)
        .checked_mul(reference_atoms.len() as u128)
        .ok_or(TransportError::NonFiniteOutput)?;
    let (normalized_transport_cost, augmenting_path_count, shortest_path_relaxations) =
        min_cost_transport(&supplies, &demands, &costs)?;

    if !normalized_transport_cost.is_finite() {
        return Err(TransportError::NonFiniteOutput);
    }

    Ok(TransportResult {
        normalized_w1: normalized_transport_cost,
        normalized_transport_cost,
        candidate_total_mass,
        reference_total_mass,
        relative_mass_mismatch,
        transported_fraction: 1.0,
        candidate_particle_count: candidate.len(),
        reference_particle_count: reference.len(),
        candidate_atom_count: candidate_atoms.len(),
        reference_atom_count: reference_atoms.len(),
        ground_cost_evaluations,
        augmenting_path_count,
        shortest_path_relaxations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TransportConfig {
        TransportConfig {
            domain_length: 8.0,
            position_length_scale: 0.5,
            velocity_length_scale: 1.0,
            maximum_relative_mass_mismatch: 1.0e-12,
        }
    }

    fn particle(position: f64, velocity: f64, weight: f64) -> PhaseSpaceParticle1d {
        PhaseSpaceParticle1d {
            position,
            velocity,
            weight,
        }
    }

    fn assert_near(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "expected {left} ~= {right} within {tolerance}"
        );
    }

    #[test]
    fn identical_measure_has_zero_w1() {
        let measure = [particle(0.5, -0.25, 0.4), particle(5.0, 1.5, 0.6)];
        let result = compare_phase_space_measures_w1(&measure, &measure, config()).unwrap();
        assert_near(result.normalized_w1, 0.0, 1.0e-12);
        assert_eq!(result.transported_fraction, 1.0);
    }

    #[test]
    fn permutation_and_colocated_split_merge_are_invariant() {
        let candidate = [particle(1.0, 0.25, 0.5), particle(6.0, -0.5, 0.5)];
        let permutation = [candidate[1], candidate[0]];
        let positive_split = [
            particle(1.0, 0.25, 0.2),
            particle(1.0, 0.25, 0.3),
            particle(6.0, -0.5, 0.2),
            particle(6.0, -0.5, 0.3),
        ];
        let permuted = compare_phase_space_measures_w1(&candidate, &permutation, config()).unwrap();
        let split = compare_phase_space_measures_w1(&candidate, &positive_split, config()).unwrap();
        assert_near(permuted.normalized_w1, 0.0, 1.0e-12);
        assert_near(split.normalized_w1, 0.0, 1.0e-12);
        assert_eq!(split.reference_atom_count, 2);
    }

    #[test]
    fn periodic_seam_uses_shortest_arc_distance() {
        let candidate = [particle(0.1, 0.0, 1.0)];
        let reference = [particle(7.9, 0.0, 1.0)];
        let result = compare_phase_space_measures_w1(&candidate, &reference, config()).unwrap();
        assert_near(result.normalized_w1, 0.4, 1.0e-12);
    }

    #[test]
    fn velocity_translation_uses_declared_scale() {
        let candidate = [particle(2.0, 0.0, 1.0)];
        let reference = [particle(2.0, 2.0, 1.0)];
        let result = compare_phase_space_measures_w1(&candidate, &reference, config()).unwrap();
        assert_near(result.normalized_w1, 2.0, 1.0e-12);
    }

    #[test]
    fn joint_correlation_change_is_detected_despite_equal_marginals() {
        let candidate = [particle(0.0, 0.0, 0.5), particle(4.0, 1.0, 0.5)];
        let reference = [particle(0.0, 1.0, 0.5), particle(4.0, 0.0, 0.5)];
        let result = compare_phase_space_measures_w1(&candidate, &reference, config()).unwrap();
        assert_near(result.normalized_w1, 1.0, 1.0e-12);
    }

    #[test]
    fn unequal_total_mass_is_not_silently_normalized() {
        let candidate = [particle(0.0, 0.0, 1.0)];
        let reference = [particle(0.0, 0.0, 0.9)];
        assert!(matches!(
            compare_phase_space_measures_w1(&candidate, &reference, config()),
            Err(TransportError::RelativeMassMismatch { .. })
        ));
    }

    #[test]
    fn signed_delta_f_weights_are_rejected() {
        let candidate = [particle(0.0, 0.0, 0.5), particle(1.0, 0.0, -0.5)];
        let reference = [particle(0.0, 0.0, 1.0)];
        assert!(matches!(
            compare_phase_space_measures_w1(&candidate, &reference, config()),
            Err(TransportError::NegativeWeight {
                measure: "candidate",
                particle: 1
            })
        ));
    }

    #[test]
    fn two_empty_measures_have_zero_cost_but_one_empty_is_rejected() {
        let empty = compare_phase_space_measures_w1(&[], &[], config()).unwrap();
        assert_eq!(empty.normalized_w1, 0.0);
        assert_eq!(empty.transported_fraction, 0.0);
        let nonempty = [particle(0.0, 0.0, 1.0)];
        assert!(matches!(
            compare_phase_space_measures_w1(&[], &nonempty, config()),
            Err(TransportError::RelativeMassMismatch { .. })
        ));
    }

    #[test]
    fn ground_cost_and_flow_accounting_are_explicit() {
        let candidate = [particle(0.0, 0.0, 0.5), particle(4.0, 0.0, 0.5)];
        let reference = [particle(1.0, 0.0, 0.5), particle(5.0, 0.0, 0.5)];
        let result = compare_phase_space_measures_w1(&candidate, &reference, config()).unwrap();
        assert_near(result.normalized_w1, 2.0, 1.0e-12);
        assert_eq!(result.ground_cost_evaluations, 4);
        assert!(result.augmenting_path_count >= 1);
        assert!(result.shortest_path_relaxations > 0);
    }

    #[test]
    fn invalid_scales_and_non_finite_data_fail_closed() {
        let measure = [particle(0.0, 0.0, 1.0)];
        let mut invalid = config();
        invalid.domain_length = 0.0;
        assert_eq!(
            compare_phase_space_measures_w1(&measure, &measure, invalid),
            Err(TransportError::InvalidDomainLength)
        );
        invalid = config();
        invalid.velocity_length_scale = f64::NAN;
        assert_eq!(
            compare_phase_space_measures_w1(&measure, &measure, invalid),
            Err(TransportError::InvalidVelocityScale)
        );
        let non_finite = [particle(f64::NAN, 0.0, 1.0)];
        assert!(matches!(
            compare_phase_space_measures_w1(&non_finite, &measure, config()),
            Err(TransportError::NonFinitePosition { .. })
        ));
    }

    #[test]
    fn min_cost_flow_can_reroute_previous_transport() {
        let candidate = [particle(0.0, 0.0, 0.6), particle(4.0, 0.0, 0.4)];
        let reference = [particle(1.0, 0.0, 0.4), particle(7.0, 0.0, 0.6)];
        let result = compare_phase_space_measures_w1(&candidate, &reference, config()).unwrap();
        assert!(result.normalized_w1.is_finite());
        assert!(result.normalized_w1 >= 0.0);
    }
}
