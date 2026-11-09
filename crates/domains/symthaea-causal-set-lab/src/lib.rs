// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal, independently auditable causal-set research primitives.
//!
//! This first milestone implements a fixed-cardinality (Poisson process
//! conditioned on N) sprinkling into a 1+1D Minkowski Alexandrov interval.
//! Null coordinates are u = t + x and v = t - x, so causal precedence is
//! component-wise order: u_a <= u_b and v_a <= v_b, with at least one strict.
//!
//! This is a kinematics/estimator testbed, NOT a quantum-gravity dynamics
//! implementation. It does not derive Einstein dynamics, a quantum measure,
//! or a continuum limit. See the crate README and the programme specification.
#![forbid(unsafe_code)]

pub mod bayesian_dimension;
pub mod causal_set_nd;
pub mod curved_frw_1p1;
pub mod curved_order_statistics;
pub mod interval_profile;
pub mod joint_likelihood;
pub mod joint_signature;
pub mod link_matrix;
pub mod link_sampling_bias;

use std::collections::HashSet;
use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

pub use bayesian_dimension::{
    BayesianDimensionError, DimensionCalibrationSamples, DimensionInferenceStatus,
    DimensionPosteriorEntry, DimensionPrior, FiniteSizeDimensionCalibration,
    FiniteSizeDimensionPosterior, MAX_BAYESIAN_DIMENSION, MAX_CALIBRATION_SUPPORT_DISTANCE,
    MAX_DIMENSION_CALIBRATION_EVENTS, MAX_DIMENSION_CALIBRATION_PAIR_EVALUATIONS,
    MAX_DIMENSION_CALIBRATION_REPLICATES, MIN_BAYESIAN_DIMENSION,
    MIN_DIMENSION_CALIBRATION_REPLICATES, calibrate_minkowski_ordering_fraction,
    estimate_finite_size_dimension_posterior,
};
pub use causal_set_nd::{CausalSetNd, CausalSetNdError, EventNd};
pub use curved_frw_1p1::{ConformalFrw1p1, CurvedFrwError};
pub use curved_order_statistics::{
    CurvedOrderFractionReference, CurvedOrderStatisticsError, DimensionlessCurvatureEstimate,
    MIN_CURVED_ORDER_QMC_POINTS, estimate_dimensionless_curvature,
    qmc_expected_comparable_pair_fraction,
};
pub use interval_profile::{
    IntervalAbundanceProfile, IntervalProfileError, MAX_INTERVAL_PROFILE_BINS,
};
pub use joint_likelihood::{
    DimensionProbability, JointDimensionModel, JointDimensionPosterior, JointGaussianCalibration,
    JointLikelihoodError, infer_joint_dimension,
};
pub use joint_signature::{
    JointCausalSetSignature, JointSignatureError, MAX_SIGNATURE_SPACETIME_DIMENSION,
    MIN_SIGNATURE_SPACETIME_DIMENSION,
};
pub use link_matrix::{
    LinkMatrix, LinkMatrixError, MAX_LINK_MATRIX_BYTES, MAX_LINK_MATRIX_EVENTS,
    MAX_LINK_ORACLE_EVENTS,
};
pub use link_sampling_bias::{
    DoubleScalingProductWindow, FiniteConeLinkProfile, LinkSamplingBiasError,
    MAX_LINK_SPATIAL_DIMENSION, MIN_LINK_SPATIAL_DIMENSION, MIN_ROOT_ADMISSIBILITY_SCALE,
    ProductWindowLinkProfile, alexandrov_volume_coefficient, double_scaling_product_window,
    hyperbolic_ball_volume, link_intensity_profile, log_hyperbolic_ball_volume,
};

/// Count the number of order intervals containing k interior events.
///
/// The returned vector is indexed by k: entry k counts unordered related
/// endpoint pairs (a,b) for which exactly k events z satisfy a ≺ z ≺ b.
/// A pair is considered once, in its causal orientation. The implementation
/// builds successor/predecessor bitsets, so it is a reference primitive with
/// O(N^2) bits of storage and O(R ceil(N/64)) interval-intersection work,
/// where R is the number of related pairs. This is intended for finite-size
/// action/reference tests, not yet a sparse large-N production implementation.
fn interval_size_histogram_by<F>(event_count: usize, mut precedes: F) -> Vec<u64>
where
    F: FnMut(usize, usize) -> bool,
{
    let mut histogram = vec![0_u64; event_count.saturating_sub(1)];
    if event_count < 2 {
        return histogram;
    }

    let word_count = event_count.div_ceil(64);
    let mut successors = vec![vec![0_u64; word_count]; event_count];
    let mut predecessors = vec![vec![0_u64; word_count]; event_count];

    for first in 0..event_count {
        for second in 0..event_count {
            if first != second && precedes(first, second) {
                successors[first][second / 64] |= 1_u64 << (second % 64);
                predecessors[second][first / 64] |= 1_u64 << (first % 64);
            }
        }
    }

    for first in 0..event_count {
        for second_word in 0..word_count {
            let mut related = successors[first][second_word];
            while related != 0 {
                let bit = related.trailing_zeros() as usize;
                let second = second_word * 64 + bit;
                related &= related - 1;
                if second >= event_count {
                    continue;
                }

                // A strict interior event is both a successor of first and
                // a predecessor of second. Endpoints are absent by irreflexivity.
                let interior_count: usize = successors[first]
                    .iter()
                    .zip(&predecessors[second])
                    .map(|(after_first, before_second)| {
                        (after_first & before_second).count_ones() as usize
                    })
                    .sum();
                histogram[interior_count] += 1;
            }
        }
    }

    histogram
}

/// Persistent segment-tree node for one prefix of the u-sorted event set.
#[derive(Clone, Copy, Debug, Default)]
struct PersistentCountNode {
    left_child: usize,
    right_child: usize,
    count: usize,
}

fn persistent_count_insert(
    previous_root: usize,
    left: usize,
    right: usize,
    position: usize,
    nodes: &mut Vec<PersistentCountNode>,
) -> usize {
    let mut updated = nodes[previous_root];
    updated.count += 1;
    let new_root = nodes.len();
    nodes.push(updated);

    if right - left > 1 {
        let middle = left + (right - left) / 2;
        if position < middle {
            let previous_child = nodes[new_root].left_child;
            let updated_child =
                persistent_count_insert(previous_child, left, middle, position, nodes);
            nodes[new_root].left_child = updated_child;
        } else {
            let previous_child = nodes[new_root].right_child;
            let updated_child =
                persistent_count_insert(previous_child, middle, right, position, nodes);
            nodes[new_root].right_child = updated_child;
        }
    }
    new_root
}

fn persistent_count_range(
    root: usize,
    left: usize,
    right: usize,
    query_left: usize,
    query_right: usize,
    nodes: &[PersistentCountNode],
) -> usize {
    if root == 0 || query_right <= left || right <= query_left {
        return 0;
    }
    if query_left <= left && right <= query_right {
        return nodes[root].count;
    }

    let middle = left + (right - left) / 2;
    persistent_count_range(
        nodes[root].left_child,
        left,
        middle,
        query_left,
        query_right,
        nodes,
    ) + persistent_count_range(
        nodes[root].right_child,
        middle,
        right,
        query_left,
        query_right,
        nodes,
    )
}

/// Exact 1+1D open-interval histogram via persistent 2D orthogonal range count.
///
/// Events are sorted/grouped by u and each prefix gets a persistent segment
/// tree over compressed v. A closed rectangle query is the difference of two
/// prefix versions and takes O(log N), then the endpoint-coordinate groups
/// are excluded because identical-coordinate events are incomparable.
/// Every unordered related endpoint pair is visited once. Complexity is
/// O(N^2 log N) time and O(N log N) memory.
fn interval_size_histogram_1p1(events: &[Event2D]) -> Vec<u64> {
    use std::collections::HashMap;

    let event_count = events.len();
    let mut histogram = vec![0_u64; event_count.saturating_sub(1)];
    if event_count < 2 {
        return histogram;
    }

    let mut u_coordinates: Vec<f64> = events.iter().map(|event| event.u).collect();
    u_coordinates.sort_by(f64::total_cmp);
    u_coordinates.dedup_by(|left, right| *left == *right);

    let mut v_coordinates: Vec<f64> = events.iter().map(|event| event.v).collect();
    v_coordinates.sort_by(f64::total_cmp);
    v_coordinates.dedup_by(|left, right| *left == *right);

    let mut order: Vec<usize> = (0..event_count).collect();
    order.sort_by(|&left, &right| {
        events[left]
            .u
            .total_cmp(&events[right].u)
            .then_with(|| events[left].v.total_cmp(&events[right].v))
            .then_with(|| events[left].id.cmp(&events[right].id))
    });

    // Node zero is the immutable all-zero tree shared by all empty prefixes.
    let mut nodes = vec![PersistentCountNode::default()];
    let mut roots = Vec::with_capacity(u_coordinates.len() + 1);
    roots.push(0_usize);
    let mut root = 0_usize;
    let mut order_index = 0_usize;

    for u in &u_coordinates {
        while order_index < order.len() && events[order[order_index]].u == *u {
            let event = events[order[order_index]];
            let v_position = v_coordinates
                .binary_search_by(|candidate| candidate.total_cmp(&event.v))
                .expect("event v coordinate is present in compression");
            root = persistent_count_insert(root, 0, v_coordinates.len(), v_position, &mut nodes);
            order_index += 1;
        }
        roots.push(root);
    }

    let mut coordinate_multiplicity: HashMap<(u64, u64), usize> =
        HashMap::with_capacity(event_count);
    for event in events {
        *coordinate_multiplicity
            .entry((event.u.to_bits(), event.v.to_bits()))
            .or_insert(0) += 1;
    }

    for first_index in 0..event_count {
        for second_index in (first_index + 1)..event_count {
            let first = events[first_index];
            let second = events[second_index];
            let (past, future) = if first.causally_precedes(second) {
                (first, second)
            } else if second.causally_precedes(first) {
                (second, first)
            } else {
                continue;
            };

            let u_before = u_coordinates.partition_point(|u| *u < past.u);
            let u_through = u_coordinates.partition_point(|u| *u <= future.u);
            let v_from = v_coordinates.partition_point(|v| *v < past.v);
            let v_through = v_coordinates.partition_point(|v| *v <= future.v);

            let upper_count = persistent_count_range(
                roots[u_through],
                0,
                v_coordinates.len(),
                v_from,
                v_through,
                &nodes,
            );
            let lower_count = persistent_count_range(
                roots[u_before],
                0,
                v_coordinates.len(),
                v_from,
                v_through,
                &nodes,
            );
            let closed_rectangle_count = upper_count - lower_count;

            let endpoint_duplicates = coordinate_multiplicity
                .get(&(past.u.to_bits(), past.v.to_bits()))
                .copied()
                .unwrap_or(0)
                + coordinate_multiplicity
                    .get(&(future.u.to_bits(), future.v.to_bits()))
                    .copied()
                    .unwrap_or(0);
            let interior_count = closed_rectangle_count
                .checked_sub(endpoint_duplicates)
                .expect("both endpoint-coordinate groups lie in the closed interval");
            histogram[interior_count] += 1;
        }
    }

    histogram
}

/// One sprinkled event in 1+1D Minkowski spacetime, represented in null
/// coordinates u = t + x and v = t - x.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Event2D {
    pub id: usize,
    pub u: f64,
    pub v: f64,
}

impl Event2D {
    pub fn coordinate_time(self) -> f64 {
        0.5 * (self.u + self.v)
    }

    pub fn coordinate_space(self) -> f64 {
        0.5 * (self.u - self.v)
    }

    /// Strict causal precedence. Null-separated events are included.
    pub fn causally_precedes(self, other: Self) -> bool {
        self.u <= other.u && self.v <= other.v && (self.u < other.u || self.v < other.v)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CausalSetError {
    InvalidExtent,
    InvalidRapidity,
    NonFiniteEvent { id: usize },
    NegativeCoordinate { id: usize },
    EventOutsideBounds { id: usize },
    DuplicateEventId { id: usize },
}

impl fmt::Display for CausalSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExtent => write!(f, "null-coordinate extents must be finite and positive"),
            Self::InvalidRapidity => write!(
                f,
                "rapidity must be finite and its boost scales representable"
            ),
            Self::NonFiniteEvent { id } => write!(f, "event {id} has a non-finite coordinate"),
            Self::NegativeCoordinate { id } => {
                write!(f, "event {id} has a negative null coordinate")
            }
            Self::EventOutsideBounds { id } => {
                write!(f, "event {id} lies outside the declared interval bounds")
            }
            Self::DuplicateEventId { id } => write!(f, "event id {id} is duplicated"),
        }
    }
}

impl Error for CausalSetError {}

/// Finite causal set embedded in a bounded 1+1D null-coordinate rectangle.
///
/// For an unboosted Alexandrov interval with endpoints separated by proper
/// time T, the region is u,v in [0,T]. Under a Lorentz boost it becomes a
/// rectangle with extents T exp(-rapidity), T exp(+rapidity); its causal
/// relations are unchanged.
#[derive(Clone, Debug)]
pub struct CausalSet2D {
    events: Vec<Event2D>,
    u_extent: f64,
    v_extent: f64,
}

impl CausalSet2D {
    /// Validate and construct an event set in [0,u_extent] x [0,v_extent].
    pub fn new(
        mut events: Vec<Event2D>,
        u_extent: f64,
        v_extent: f64,
    ) -> Result<Self, CausalSetError> {
        if !u_extent.is_finite() || !v_extent.is_finite() || u_extent <= 0.0 || v_extent <= 0.0 {
            return Err(CausalSetError::InvalidExtent);
        }

        let mut ids = HashSet::with_capacity(events.len());
        for event in &events {
            if !event.u.is_finite() || !event.v.is_finite() {
                return Err(CausalSetError::NonFiniteEvent { id: event.id });
            }
            if event.u < 0.0 || event.v < 0.0 {
                return Err(CausalSetError::NegativeCoordinate { id: event.id });
            }
            if event.u > u_extent || event.v > v_extent {
                return Err(CausalSetError::EventOutsideBounds { id: event.id });
            }
            if !ids.insert(event.id) {
                return Err(CausalSetError::DuplicateEventId { id: event.id });
            }
        }

        // IEEE -0.0 and +0.0 represent the same spacetime coordinate. Store
        // one canonical zero so total-order sorting and coordinate grouping
        // preserve the mathematical partial order.
        for event in &mut events {
            if event.u == 0.0 {
                event.u = 0.0;
            }
            if event.v == 0.0 {
                event.v = 0.0;
            }
        }

        Ok(Self {
            events,
            u_extent,
            v_extent,
        })
    }

    /// Fixed-N sprinkling into a 1+1D Minkowski Alexandrov interval.
    ///
    /// This is the conditional distribution of a Poisson sprinkling given
    /// its cardinality N: event coordinates are i.i.d. uniform in the causal
    /// diamond. The seed makes the sample reproducible.
    pub fn sprinkle_fixed_cardinality(
        seed: u64,
        event_count: usize,
        proper_time: f64,
    ) -> Result<Self, CausalSetError> {
        if !proper_time.is_finite() || proper_time <= 0.0 {
            return Err(CausalSetError::InvalidExtent);
        }

        let mut rng = SplitMix64::new(seed);
        let events = (0..event_count)
            .map(|id| Event2D {
                id,
                u: rng.unit_f64() * proper_time,
                v: rng.unit_f64() * proper_time,
            })
            .collect();

        Self::new(events, proper_time, proper_time)
    }

    pub fn events(&self) -> &[Event2D] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn null_coordinate_extents(&self) -> (f64, f64) {
        (self.u_extent, self.v_extent)
    }

    /// 1+1D Minkowski Alexandrov-interval volume for proper-time separation T.
    ///
    /// With u=t+x and v=t-x, the volume element is (1/2) du dv.
    pub fn minkowski_interval_volume_1p1(proper_time: f64) -> Option<f64> {
        if !proper_time.is_finite() || proper_time <= 0.0 {
            return None;
        }
        Some(0.5 * proper_time * proper_time)
    }

    /// Apply a passive Lorentz boost of rapidity eta in null coordinates:
    /// u' = exp(-eta) u, v' = exp(+eta) v.
    ///
    /// This transforms coordinates and the bounding rectangle, not the
    /// physical sprinkling. All causal relations must remain invariant.
    pub fn lorentz_boosted(&self, rapidity: f64) -> Result<Self, CausalSetError> {
        if !rapidity.is_finite() {
            return Err(CausalSetError::InvalidRapidity);
        }
        let u_scale = (-rapidity).exp();
        let v_scale = rapidity.exp();
        let u_extent = self.u_extent * u_scale;
        let v_extent = self.v_extent * v_scale;
        if !u_scale.is_finite()
            || !v_scale.is_finite()
            || u_scale == 0.0
            || v_scale == 0.0
            || !u_extent.is_finite()
            || !v_extent.is_finite()
            || u_extent <= 0.0
            || v_extent <= 0.0
        {
            return Err(CausalSetError::InvalidRapidity);
        }

        let events = self
            .events
            .iter()
            .map(|event| Event2D {
                id: event.id,
                u: event.u * u_scale,
                v: event.v * v_scale,
            })
            .collect();

        Self::new(events, u_extent, v_extent)
    }

    /// Count unordered pairs related by the partial order.
    ///
    /// In 1+1D null coordinates, a ≺ b iff u_a <= u_b and v_a <= v_b,
    /// with at least one strict inequality. Sorting by (u, v) and querying
    /// a Fenwick tree for prior v <= v_b counts every related pair once.
    /// Identical-coordinate events are excluded because neither strictly
    /// precedes the other. Complexity: O(N log N) time and O(N) memory.
    pub fn related_pair_count(&self) -> u64 {
        let n = self.events.len();
        if n < 2 {
            return 0;
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            self.events[a]
                .u
                .total_cmp(&self.events[b].u)
                .then_with(|| self.events[a].v.total_cmp(&self.events[b].v))
                .then_with(|| self.events[a].id.cmp(&self.events[b].id))
        });

        let mut v_coordinates: Vec<f64> = self.events.iter().map(|event| event.v).collect();
        v_coordinates.sort_by(f64::total_cmp);
        v_coordinates.dedup_by(|a, b| *a == *b);

        let mut fenwick = FenwickTree::new(v_coordinates.len());
        let mut related = 0_u64;
        let mut previous_coordinate: Option<(f64, f64)> = None;
        let mut previous_duplicates = 0_u64;

        for event_index in order {
            let event = self.events[event_index];
            match previous_coordinate {
                Some((u, v)) if u == event.u && v == event.v => {
                    previous_duplicates += 1;
                }
                _ => {
                    previous_coordinate = Some((event.u, event.v));
                    previous_duplicates = 0;
                }
            }

            let v_index = v_coordinates
                .binary_search_by(|candidate| candidate.total_cmp(&event.v))
                .expect("event v coordinate was included in compression");
            let previous_in_lower_orthant = fenwick.prefix_sum(v_index + 1);

            // All earlier sorted events have u <= event.u. The Fenwick
            // prefix restricts to v <= event.v. Remove exact-coordinate
            // duplicates, which are not causally related.
            related += previous_in_lower_orthant - previous_duplicates;
            fenwick.add(v_index, 1);
        }

        related
    }

    /// Histogram of cardinalities of open order intervals.
    ///
    /// Bin k counts causally related endpoint pairs with exactly k events
    /// strictly between the endpoints. The histogram is invariant under
    /// order-preserving relabelling and Lorentz boosts of the embedding.
    pub fn interval_size_histogram(&self) -> Vec<u64> {
        interval_size_histogram_1p1(&self.events)
    }

    /// Retained generic bitset reference for 1+1D interval-histogram comparison.
    ///
    /// This deliberately uses the dimension-generic successor/predecessor
    /// intersection path rather than the specialized persistent range tree.
    /// It is useful as an independent implementation and a benchmark baseline.
    pub fn interval_size_histogram_bitset_reference(&self) -> Vec<u64> {
        interval_size_histogram_by(self.events.len(), |first, second| {
            self.events[first].causally_precedes(self.events[second])
        })
    }

    /// Dimensionless two-dimensional Benincasa-Dowker action bracket.
    ///
    /// For a 1+1D causal set:
    ///     S / hbar = 2 N - 4 N_0 + 8 N_1 - 4 N_2,
    /// where N_k counts causally related endpoint pairs with exactly k
    /// elements strictly inside the open order interval. This returns only
    /// the integer combinatorial bracket, not a physical action with units.
    /// The continuum Einstein-Hilbert interpretation requires manifold-like
    /// Poisson-sprinkling assumptions and separate continuum-limit checks.
    pub fn bdg_action_2d_dimensionless_bracket(&self) -> i128 {
        let abundance = self.interval_size_histogram();
        let n = self.events.len() as i128;
        let n0 = i128::from(abundance.first().copied().unwrap_or(0));
        let n1 = i128::from(abundance.get(1).copied().unwrap_or(0));
        let n2 = i128::from(abundance.get(2).copied().unwrap_or(0));
        2 * n - 4 * n0 + 8 * n1 - 4 * n2
    }

    /// Myrheim ordering fraction r = R / binom(N,2), where R counts
    /// unordered related pairs. Returns None for fewer than two events.
    pub fn ordering_fraction(&self) -> Option<f64> {
        let n = self.events.len() as f64;
        if n < 2.0 {
            return None;
        }
        Some(self.related_pair_count() as f64 / (n * (n - 1.0) / 2.0))
    }

    /// Longest-chain event IDs, in causal order.
    ///
    /// In 1+1D null coordinates this is a longest nondecreasing subsequence
    /// of v after sorting by (u, v, id). Equal-coordinate events are grouped
    /// and represented once because identical events are incomparable.
    /// The reconstructed chain is therefore boost-invariant under positive
    /// rescalings of u and v. Complexity: O(N log N) time and O(N) memory.
    pub fn longest_chain_ids(&self) -> Vec<usize> {
        let n = self.events.len();
        if n == 0 {
            return Vec::new();
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            self.events[a]
                .u
                .total_cmp(&self.events[b].u)
                .then_with(|| self.events[a].v.total_cmp(&self.events[b].v))
                .then_with(|| self.events[a].id.cmp(&self.events[b].id))
        });

        let mut tails_values: Vec<f64> = Vec::new();
        let mut tails_events: Vec<usize> = Vec::new();
        let mut parent: Vec<Option<usize>> = vec![None; n];

        let mut cursor = 0;
        while cursor < n {
            let representative = order[cursor];
            let event = self.events[representative];

            // Process one representative per identical coordinate. Sorting
            // by id makes this the deterministic lowest-id representative.
            let mut next = cursor + 1;
            while next < n {
                let candidate = self.events[order[next]];
                if candidate.u == event.u && candidate.v == event.v {
                    next += 1;
                } else {
                    break;
                }
            }

            // Longest nondecreasing subsequence: equal v values may chain
            // when u strictly increases. Identical (u, v) points were
            // grouped above and cannot chain with each other.
            let position = tails_values.partition_point(|&tail| tail <= event.v);
            if position > 0 {
                parent[representative] = Some(tails_events[position - 1]);
            }

            if position == tails_values.len() {
                tails_values.push(event.v);
                tails_events.push(representative);
            } else {
                tails_values[position] = event.v;
                tails_events[position] = representative;
            }

            cursor = next;
        }

        let mut chain = Vec::with_capacity(tails_events.len());
        let mut current = tails_events.last().copied();
        while let Some(index) = current {
            chain.push(self.events[index].id);
            current = parent[index];
        }
        chain.reverse();
        chain
    }

    pub fn longest_chain_length(&self) -> usize {
        self.longest_chain_ids().len()
    }

    /// Estimate continuum spacetime dimension by inverting the Myrheim-Meyer
    /// ordering-fraction relation. This is an estimator, not an exact
    /// dimension label for a finite causal set.
    pub fn estimate_myrheim_meyer_dimension(&self, max_dimension: f64) -> Option<f64> {
        estimate_dimension_from_ordering_fraction(self.ordering_fraction()?, max_dimension)
    }
}

/// Expected ordering fraction in a flat d-dimensional Minkowski Alexandrov
/// interval for a sufficiently dense Poisson sprinkling:
///
/// r(d) = Gamma(d+1) Gamma(d/2) / (2 Gamma(3d/2)).
pub fn expected_ordering_fraction(dimension: f64) -> Option<f64> {
    if !dimension.is_finite() || !(1.0..=32.0).contains(&dimension) {
        return None;
    }
    let log_r = ln_gamma(dimension + 1.0) + ln_gamma(dimension / 2.0)
        - 2.0_f64.ln()
        - ln_gamma(1.5 * dimension);
    let value = log_r.exp();
    if value.is_finite() { Some(value) } else { None }
}

/// Numerically invert the monotone Myrheim-Meyer relation on [1,max_dimension].
pub fn estimate_dimension_from_ordering_fraction(
    observed_fraction: f64,
    max_dimension: f64,
) -> Option<f64> {
    if !observed_fraction.is_finite()
        || !(0.0..=1.0).contains(&observed_fraction)
        || !max_dimension.is_finite()
        || !(1.0..=32.0).contains(&max_dimension)
    {
        return None;
    }

    let at_one = expected_ordering_fraction(1.0)?;
    let at_max = expected_ordering_fraction(max_dimension)?;
    if observed_fraction > at_one || observed_fraction < at_max {
        return None;
    }

    let mut low = 1.0;
    let mut high = max_dimension;
    for _ in 0..96 {
        let mid = 0.5 * (low + high);
        let expected = expected_ordering_fraction(mid)?;
        if expected > observed_fraction {
            low = mid;
        } else {
            high = mid;
        }
    }
    Some(0.5 * (low + high))
}

/// Lanczos log-gamma approximation (g=7), sufficient for the estimator's
/// low-dimensional numerical inversion. No external numerical dependency.
fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if z < 0.5 {
        return PI.ln() - (PI * z).sin().ln() - ln_gamma(1.0 - z);
    }

    let shifted = z - 1.0;
    let mut x = COEFFS[0];
    for (index, coefficient) in COEFFS.iter().enumerate().skip(1) {
        x += coefficient / (shifted + index as f64);
    }
    let t = shifted + 7.5;
    0.5 * (2.0 * PI).ln() + (shifted + 0.5) * t.ln() - t + x.ln()
}

/// Fenwick tree over nonnegative integer counts.
struct FenwickTree {
    tree: Vec<u64>,
}

impl FenwickTree {
    fn new(size: usize) -> Self {
        Self {
            tree: vec![0; size + 1],
        }
    }

    fn add(&mut self, index: usize, value: u64) {
        let mut cursor = index + 1;
        while cursor < self.tree.len() {
            self.tree[cursor] += value;
            let low_bit = cursor & (!cursor + 1);
            cursor += low_bit;
        }
    }

    /// Sum of the first prefix_len compressed coordinates.
    fn prefix_sum(&self, prefix_len: usize) -> u64 {
        let mut cursor = prefix_len;
        let mut total = 0_u64;
        while cursor > 0 {
            total += self.tree[cursor];
            cursor &= cursor - 1;
        }
        total
    }
}

/// Small, deterministic SplitMix64 generator. This is for reproducible
/// sampling, not cryptography.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        (mantissa as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_interval_size_histogram(causet: &CausalSet2D) -> Vec<u64> {
        let n = causet.events.len();
        let mut histogram = vec![0_u64; n.saturating_sub(1)];
        for first in 0..n {
            for second in (first + 1)..n {
                let (past, future) =
                    if causet.events[first].causally_precedes(causet.events[second]) {
                        (first, second)
                    } else if causet.events[second].causally_precedes(causet.events[first]) {
                        (second, first)
                    } else {
                        continue;
                    };
                let interior = (0..n)
                    .filter(|&event| {
                        causet.events[past].causally_precedes(causet.events[event])
                            && causet.events[event].causally_precedes(causet.events[future])
                    })
                    .count();
                histogram[interior] += 1;
            }
        }
        histogram
    }

    fn brute_related_pair_count(causet: &CausalSet2D) -> u64 {
        let mut count = 0_u64;
        for i in 0..causet.events.len() {
            for j in (i + 1)..causet.events.len() {
                let a = causet.events[i];
                let b = causet.events[j];
                if a.causally_precedes(b) || b.causally_precedes(a) {
                    count += 1;
                }
            }
        }
        count
    }

    fn brute_longest_chain_length(causet: &CausalSet2D) -> usize {
        let n = causet.events.len();
        if n == 0 {
            return 0;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            causet.events[a]
                .u
                .total_cmp(&causet.events[b].u)
                .then_with(|| causet.events[a].v.total_cmp(&causet.events[b].v))
                .then_with(|| causet.events[a].id.cmp(&causet.events[b].id))
        });
        let mut best = vec![1_usize; n];
        for j in 0..n {
            for i in 0..j {
                if causet.events[order[i]].causally_precedes(causet.events[order[j]]) {
                    best[j] = best[j].max(best[i] + 1);
                }
            }
        }
        best.into_iter().max().unwrap_or(0)
    }

    #[test]
    fn expected_myrheim_fractions_match_reference_values() {
        assert!((expected_ordering_fraction(2.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((expected_ordering_fraction(4.0).unwrap() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn myrheim_dimension_inversion_recovers_integer_dimensions() {
        for dimension in [2.0, 3.0, 4.0, 5.0] {
            let fraction = expected_ordering_fraction(dimension).unwrap();
            let recovered = estimate_dimension_from_ordering_fraction(fraction, 8.0).unwrap();
            assert!(
                (recovered - dimension).abs() < 1e-8,
                "d={dimension}, recovered={recovered}"
            );
        }
    }

    #[test]
    fn fixed_seed_sprinkling_is_reproducible() {
        let a = CausalSet2D::sprinkle_fixed_cardinality(42, 256, 10.0).unwrap();
        let b = CausalSet2D::sprinkle_fixed_cardinality(42, 256, 10.0).unwrap();
        assert_eq!(a.events(), b.events());
    }

    #[test]
    fn one_plus_one_minkowski_recovers_half_ordering_fraction() {
        let causet = CausalSet2D::sprinkle_fixed_cardinality(0xC0FFEE, 1_500, 1.0).unwrap();
        let fraction = causet.ordering_fraction().unwrap();
        assert!(
            (fraction - 0.5).abs() < 0.06,
            "ordering fraction={fraction}"
        );
        let dimension = causet.estimate_myrheim_meyer_dimension(8.0).unwrap();
        assert!(
            (dimension - 2.0).abs() < 0.25,
            "estimated dimension={dimension}"
        );
    }

    #[test]
    fn lorentz_boost_preserves_order_invariants_and_chain_tie_breaks() {
        let causet = CausalSet2D::sprinkle_fixed_cardinality(7, 512, 3.0).unwrap();
        for rapidity in [-1.7, -0.4, 0.0, 0.7, 1.7] {
            let boosted = causet.lorentz_boosted(rapidity).unwrap();
            assert_eq!(causet.related_pair_count(), boosted.related_pair_count());
            assert_eq!(
                causet.interval_size_histogram(),
                boosted.interval_size_histogram()
            );
            assert_eq!(
                causet.bdg_action_2d_dimensionless_bracket(),
                boosted.bdg_action_2d_dimensionless_bracket(),
                "2D BD action bracket changed under rapidity={rapidity}"
            );
            assert_eq!(
                causet.longest_chain_ids(),
                boosted.longest_chain_ids(),
                "selected maximum chain changed under boost eta={rapidity}"
            );
            assert!(
                (causet.ordering_fraction().unwrap() - boosted.ordering_fraction().unwrap()).abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn interval_size_histogram_matches_a_four_event_total_order() {
        let events = (0..4)
            .map(|id| Event2D {
                id,
                u: id as f64,
                v: id as f64,
            })
            .collect();
        let causet = CausalSet2D::new(events, 4.0, 4.0).unwrap();
        assert_eq!(causet.interval_size_histogram(), vec![3, 2, 1]);
    }

    #[test]
    fn two_dimensional_bdg_action_matches_hand_counted_chain_and_antichain() {
        let chain_events = (0..5)
            .map(|id| Event2D {
                id,
                u: id as f64,
                v: id as f64,
            })
            .collect();
        let chain = CausalSet2D::new(chain_events, 5.0, 5.0).unwrap();
        assert_eq!(chain.interval_size_histogram(), vec![4, 3, 2, 1]);
        assert_eq!(chain.bdg_action_2d_dimensionless_bracket(), 10);

        let antichain_events = (0..5)
            .map(|id| Event2D {
                id,
                u: id as f64,
                v: (4 - id) as f64,
            })
            .collect();
        let antichain = CausalSet2D::new(antichain_events, 5.0, 5.0).unwrap();
        assert_eq!(antichain.interval_size_histogram(), vec![0, 0, 0, 0]);
        assert_eq!(antichain.bdg_action_2d_dimensionless_bracket(), 10);
    }

    #[test]
    fn interval_histogram_handles_empty_and_antichain_controls() {
        let empty = CausalSet2D::new(Vec::new(), 1.0, 1.0).unwrap();
        assert!(empty.interval_size_histogram().is_empty());

        let antichain = CausalSet2D::new(
            vec![
                Event2D {
                    id: 0,
                    u: 0.1,
                    v: 0.9,
                },
                Event2D {
                    id: 1,
                    u: 0.3,
                    v: 0.7,
                },
                Event2D {
                    id: 2,
                    u: 0.5,
                    v: 0.5,
                },
                Event2D {
                    id: 3,
                    u: 0.7,
                    v: 0.3,
                },
            ],
            1.0,
            1.0,
        )
        .unwrap();
        assert_eq!(antichain.interval_size_histogram(), vec![0, 0, 0]);
    }

    #[test]
    fn longest_chain_has_expected_square_root_order_of_growth() {
        let causet = CausalSet2D::sprinkle_fixed_cardinality(2026, 1_024, 1.0).unwrap();
        let scaled = causet.longest_chain_length() as f64 / (causet.len() as f64).sqrt();
        assert!((1.25..=2.5).contains(&scaled), "L/sqrt(N)={scaled}");
    }

    #[test]
    fn identical_coordinate_events_are_not_related_or_chained() {
        let causet = CausalSet2D::new(
            vec![
                Event2D {
                    id: 10,
                    u: 0.1,
                    v: 0.1,
                },
                Event2D {
                    id: 11,
                    u: 0.1,
                    v: 0.1,
                },
                Event2D {
                    id: 12,
                    u: 0.2,
                    v: 0.2,
                },
            ],
            1.0,
            1.0,
        )
        .unwrap();

        assert_eq!(causet.related_pair_count(), 2);
        assert_eq!(causet.longest_chain_length(), 2);
        assert_eq!(
            causet.related_pair_count(),
            brute_related_pair_count(&causet)
        );
        assert_eq!(
            causet.longest_chain_length(),
            brute_longest_chain_length(&causet)
        );
    }

    #[test]
    fn n_log_n_observables_match_bruteforce_on_quantized_orders() {
        // Quantization deliberately creates repeated u values, repeated v
        // values, equal-coordinate events and incomparable pairs.
        for seed in 0..24 {
            let mut rng = SplitMix64::new(seed);
            let events = (0..48)
                .map(|id| Event2D {
                    id,
                    u: (rng.unit_f64() * 6.0).floor(),
                    v: (rng.unit_f64() * 6.0).floor(),
                })
                .collect();

            let causet = CausalSet2D::new(events, 6.0, 6.0).unwrap();
            assert_eq!(
                causet.related_pair_count(),
                brute_related_pair_count(&causet),
                "related-pair mismatch for seed={seed}"
            );
            assert_eq!(
                causet.interval_size_histogram(),
                brute_interval_size_histogram(&causet),
                "interval-size histogram mismatch for seed={seed}"
            );

            let chain = causet.longest_chain_ids();
            assert_eq!(
                chain.len(),
                brute_longest_chain_length(&causet),
                "longest-chain mismatch for seed={seed}"
            );
            for pair in chain.windows(2) {
                let a = *causet
                    .events()
                    .iter()
                    .find(|event| event.id == pair[0])
                    .unwrap();
                let b = *causet
                    .events()
                    .iter()
                    .find(|event| event.id == pair[1])
                    .unwrap();
                assert!(
                    a.causally_precedes(b),
                    "returned chain contains a non-causal edge for seed={seed}"
                );
            }
        }
    }

    #[test]
    fn invalid_coordinates_are_rejected() {
        assert!(CausalSet2D::sprinkle_fixed_cardinality(1, 10, 0.0).is_err());
        assert!(
            CausalSet2D::new(
                vec![Event2D {
                    id: 0,
                    u: f64::NAN,
                    v: 0.5
                }],
                1.0,
                1.0
            )
            .is_err()
        );
        assert!(
            CausalSet2D::new(
                vec![
                    Event2D {
                        id: 0,
                        u: 0.2,
                        v: 0.3
                    },
                    Event2D {
                        id: 0,
                        u: 0.4,
                        v: 0.5
                    }
                ],
                1.0,
                1.0
            )
            .is_err()
        );
    }
}
