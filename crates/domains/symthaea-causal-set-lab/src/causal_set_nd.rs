// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dimension-generic causal-set kinematics for flat Minkowski Alexandrov intervals.
//!
//! Coordinates are (t, x_1, ..., x_(D-1)) with c = 1. A fixed-cardinality
//! sprinkling is a homogeneous Poisson sprinkling conditioned on N: sample
//! time with density proportional to (T/2 - |t|)^(D-1), then sample a point
//! uniformly in the spatial ball available at that time.
//!
//! This is finite-set kinematics and estimator code, not causal-set quantum
//! dynamics, a quantum measure, or an Einstein-equation solver.

#![forbid(unsafe_code)]

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

/// One event in D-dimensional Minkowski spacetime, with time at coordinate 0.
#[derive(Clone, Debug, PartialEq)]
pub struct EventNd {
    pub id: usize,
    pub coordinates: Vec<f64>,
}

impl EventNd {
    pub fn spacetime_dimension(&self) -> usize {
        self.coordinates.len()
    }

    /// Strict causal precedence, including null-separated events.
    pub fn causally_precedes(&self, other: &Self) -> bool {
        if self.coordinates.len() != other.coordinates.len() {
            return false;
        }
        let dt = other.coordinates[0] - self.coordinates[0];
        if dt <= 0.0 {
            return false;
        }
        let spatial_distance = self.coordinates[1..]
            .iter()
            .zip(&other.coordinates[1..])
            .fold(0.0_f64, |distance, (a, b)| distance.hypot(b - a));
        dt >= spatial_distance
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CausalSetNdError {
    UnsupportedDimension {
        dimension: usize,
    },
    InvalidProperTime,
    NonFiniteEvent {
        id: usize,
    },
    DuplicateEventId {
        id: usize,
    },
    CoordinateCount {
        id: usize,
        expected: usize,
        actual: usize,
    },
    InvalidRapidity,
    InvalidBoostAxis {
        axis: usize,
        spatial_dimensions: usize,
    },
}

impl fmt::Display for CausalSetNdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDimension { dimension } => {
                write!(f, "spacetime dimension must be 2, 3 or 4, got {dimension}")
            }
            Self::InvalidProperTime => {
                write!(f, "proper-time separation must be finite and positive")
            }
            Self::NonFiniteEvent { id } => write!(f, "event {id} has a non-finite coordinate"),
            Self::DuplicateEventId { id } => write!(f, "event id {id} is duplicated"),
            Self::CoordinateCount {
                id,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "event {id} has {actual} coordinates; expected {expected}"
                )
            }
            Self::InvalidRapidity => write!(
                f,
                "rapidity must be finite with representable boost factors"
            ),
            Self::InvalidBoostAxis {
                axis,
                spatial_dimensions,
            } => {
                write!(
                    f,
                    "spatial boost axis {axis} is outside 0..{spatial_dimensions}"
                )
            }
        }
    }
}

impl Error for CausalSetNdError {}

/// A finite event set embedded in D-dimensional flat Minkowski spacetime.
#[derive(Clone, Debug)]
pub struct CausalSetNd {
    spacetime_dimension: usize,
    events: Vec<EventNd>,
}

impl CausalSetNd {
    pub fn new(spacetime_dimension: usize, events: Vec<EventNd>) -> Result<Self, CausalSetNdError> {
        validate_dimension(spacetime_dimension)?;
        let mut ids = std::collections::HashSet::with_capacity(events.len());
        for event in &events {
            if event.coordinates.len() != spacetime_dimension {
                return Err(CausalSetNdError::CoordinateCount {
                    id: event.id,
                    expected: spacetime_dimension,
                    actual: event.coordinates.len(),
                });
            }
            if event
                .coordinates
                .iter()
                .any(|coordinate| !coordinate.is_finite())
            {
                return Err(CausalSetNdError::NonFiniteEvent { id: event.id });
            }
            if !ids.insert(event.id) {
                return Err(CausalSetNdError::DuplicateEventId { id: event.id });
            }
        }
        Ok(Self {
            spacetime_dimension,
            events,
        })
    }

    /// Fixed-cardinality Poisson sprinkling into a flat Alexandrov interval.
    ///
    /// The two endpoints have proper-time separation T. For D spacetime
    /// dimensions, |t|/(T/2) has a Beta(1,D) distribution. Conditional on t,
    /// the spatial position is uniform in the (D-1)-ball of radius T/2-|t|.
    /// This gives a spatially and temporally uniform sprinkling per spacetime
    /// volume, conditioned on the requested event count.
    pub fn sprinkle_fixed_cardinality(
        seed: u64,
        event_count: usize,
        spacetime_dimension: usize,
        proper_time: f64,
    ) -> Result<Self, CausalSetNdError> {
        validate_dimension(spacetime_dimension)?;
        if !proper_time.is_finite() || proper_time <= 0.0 {
            return Err(CausalSetNdError::InvalidProperTime);
        }

        let spatial_dimensions = spacetime_dimension - 1;
        let half_time = 0.5 * proper_time;
        let mut rng = SplitMix64::new(seed);
        let mut events = Vec::with_capacity(event_count);

        for id in 0..event_count {
            // Inverse CDF for Beta(1,D): F(s)=1-(1-s)^D.
            let s = 1.0 - (1.0 - rng.unit_f64()).powf(1.0 / spacetime_dimension as f64);
            let time_sign = if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 };
            let time = time_sign * half_time * s;
            let spatial_radius = (half_time - time.abs()).max(0.0);
            let radius = spatial_radius * rng.unit_f64().powf(1.0 / spatial_dimensions as f64);
            let direction = random_unit_direction(spatial_dimensions, &mut rng);
            let mut coordinates = Vec::with_capacity(spacetime_dimension);
            coordinates.push(time);
            coordinates.extend(direction.into_iter().map(|component| radius * component));
            events.push(EventNd { id, coordinates });
        }

        Self::new(spacetime_dimension, events)
    }

    pub fn spacetime_dimension(&self) -> usize {
        self.spacetime_dimension
    }

    pub fn spatial_dimensions(&self) -> usize {
        self.spacetime_dimension - 1
    }

    pub fn events(&self) -> &[EventNd] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Alexandrov-interval spacetime volume for endpoint proper-time T.
    ///
    /// V_D = V_(D-1) T^D / (2^(D-1) D), where V_n is the unit n-ball volume.
    pub fn minkowski_interval_volume(spacetime_dimension: usize, proper_time: f64) -> Option<f64> {
        if !(2..=4).contains(&spacetime_dimension) || !proper_time.is_finite() || proper_time <= 0.0
        {
            return None;
        }
        let spatial_dimension = spacetime_dimension - 1;
        let log_unit_ball = (spatial_dimension as f64 / 2.0) * PI.ln()
            - ln_gamma(spatial_dimension as f64 / 2.0 + 1.0);
        let log_volume = log_unit_ball + spacetime_dimension as f64 * proper_time.ln()
            - spatial_dimension as f64 * 2.0_f64.ln()
            - (spacetime_dimension as f64).ln();
        Some(log_volume.exp())
    }

    /// Count unordered causally related pairs. Reference O(N^2) implementation.
    pub fn related_pair_count(&self) -> u64 {
        let mut count = 0_u64;
        for i in 0..self.events.len() {
            for j in (i + 1)..self.events.len() {
                if self.events[i].causally_precedes(&self.events[j])
                    || self.events[j].causally_precedes(&self.events[i])
                {
                    count += 1;
                }
            }
        }
        count
    }

    /// Histogram of cardinalities of open order intervals.
    ///
    /// Bin k counts causally related endpoint pairs with exactly k events
    /// strictly between the endpoints. The bitset reference is dimension
    /// generic and invariant under Lorentz boosts of the embedding.
    pub fn interval_size_histogram(&self) -> Vec<u64> {
        super::interval_size_histogram_by(self.events.len(), |first, second| {
            self.events[first].causally_precedes(&self.events[second])
        })
    }

    /// Myrheim ordering fraction r = R / choose(N,2).
    pub fn ordering_fraction(&self) -> Option<f64> {
        let n = self.events.len() as f64;
        if n < 2.0 {
            return None;
        }
        Some(self.related_pair_count() as f64 / (n * (n - 1.0) / 2.0))
    }

    /// O(N^2) reference longest-chain length in arbitrary D.
    pub fn longest_chain_length(&self) -> usize {
        let n = self.events.len();
        if n == 0 {
            return 0;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            self.events[a].coordinates[0]
                .total_cmp(&self.events[b].coordinates[0])
                .then_with(|| self.events[a].id.cmp(&self.events[b].id))
        });

        let mut best = vec![1_usize; n];
        let mut maximum = 1_usize;
        for j in 0..n {
            for i in 0..j {
                if self.events[order[i]].causally_precedes(&self.events[order[j]]) {
                    best[j] = best[j].max(best[i] + 1);
                }
            }
            maximum = maximum.max(best[j]);
        }
        maximum
    }

    pub fn estimate_myrheim_meyer_dimension(&self, max_dimension: f64) -> Option<f64> {
        estimate_dimension_from_ordering_fraction(self.ordering_fraction()?, max_dimension)
    }

    /// Passive Lorentz boost along one selected spatial axis.
    ///
    /// axis is zero-based among spatial coordinates, not including time.
    pub fn lorentz_boosted(&self, rapidity: f64, axis: usize) -> Result<Self, CausalSetNdError> {
        let spatial_dimensions = self.spatial_dimensions();
        if axis >= spatial_dimensions {
            return Err(CausalSetNdError::InvalidBoostAxis {
                axis,
                spatial_dimensions,
            });
        }
        if !rapidity.is_finite() {
            return Err(CausalSetNdError::InvalidRapidity);
        }
        let cosh = rapidity.cosh();
        let sinh = rapidity.sinh();
        if !cosh.is_finite() || !sinh.is_finite() {
            return Err(CausalSetNdError::InvalidRapidity);
        }

        let coordinate_axis = axis + 1;
        let events = self
            .events
            .iter()
            .map(|event| {
                let mut coordinates = event.coordinates.clone();
                let time = event.coordinates[0];
                let position = event.coordinates[coordinate_axis];
                coordinates[0] = cosh * time - sinh * position;
                coordinates[coordinate_axis] = cosh * position - sinh * time;
                EventNd {
                    id: event.id,
                    coordinates,
                }
            })
            .collect();

        // A boost changes the coordinate shape of the Alexandrov interval,
        // so construction here validates finite coordinates, not original
        // unboosted bounds.
        Self::new(self.spacetime_dimension, events)
    }
}

fn validate_dimension(dimension: usize) -> Result<(), CausalSetNdError> {
    if !(2..=4).contains(&dimension) {
        return Err(CausalSetNdError::UnsupportedDimension { dimension });
    }
    Ok(())
}

fn random_unit_direction(spatial_dimensions: usize, rng: &mut SplitMix64) -> Vec<f64> {
    match spatial_dimensions {
        1 => vec![if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 }],
        2 => {
            let theta = 2.0 * PI * rng.unit_f64();
            vec![theta.cos(), theta.sin()]
        }
        3 => {
            let z = 2.0 * rng.unit_f64() - 1.0;
            let theta = 2.0 * PI * rng.unit_f64();
            let radial = (1.0 - z * z).max(0.0).sqrt();
            vec![radial * theta.cos(), radial * theta.sin(), z]
        }
        _ => unreachable!("spacetime dimensions are validated to 2..=4"),
    }
}

/// Expected Myrheim–Meyer ordering fraction for a flat D-dimensional Minkowski
/// Alexandrov interval in the dense Poisson limit.
pub fn expected_ordering_fraction(spacetime_dimension: f64) -> Option<f64> {
    if !spacetime_dimension.is_finite() || !(1.0..=32.0).contains(&spacetime_dimension) {
        return None;
    }
    let log_r = ln_gamma(spacetime_dimension + 1.0) + ln_gamma(spacetime_dimension / 2.0)
        - 2.0_f64.ln()
        - ln_gamma(1.5 * spacetime_dimension);
    let value = log_r.exp();
    if value.is_finite() { Some(value) } else { None }
}

/// Numerically invert the monotone Myrheim–Meyer ordering-fraction relation.
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
        if expected_ordering_fraction(mid)? > observed_fraction {
            low = mid;
        } else {
            high = mid;
        }
    }
    Some(0.5 * (low + high))
}

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

/// Deterministic SplitMix64. Reproducibility only; not cryptography.
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
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_interval_size_histogram(causet: &CausalSetNd) -> Vec<u64> {
        let n = causet.events.len();
        let mut histogram = vec![0_u64; n.saturating_sub(1)];
        for first in 0..n {
            for second in (first + 1)..n {
                let (past, future) = if causet.events[first].causally_precedes(&causet.events[second]) {
                    (first, second)
                } else if causet.events[second].causally_precedes(&causet.events[first]) {
                    (second, first)
                } else {
                    continue;
                };
                let interior = (0..n)
                    .filter(|&event| {
                        causet.events[past].causally_precedes(&causet.events[event])
                            && causet.events[event].causally_precedes(&causet.events[future])
                    })
                    .count();
                histogram[interior] += 1;
            }
        }
        histogram
    }

    #[test]
    fn interval_volumes_match_closed_form_references() {
        assert!((CausalSetNd::minkowski_interval_volume(2, 1.0).unwrap() - 0.5).abs() < 1e-12);
        assert!(
            (CausalSetNd::minkowski_interval_volume(3, 1.0).unwrap() - PI / 12.0).abs() < 1e-12
        );
        assert!(
            (CausalSetNd::minkowski_interval_volume(4, 1.0).unwrap() - PI / 24.0).abs() < 1e-12
        );
    }

    #[test]
    fn ordering_fraction_reference_values_match() {
        assert!((expected_ordering_fraction(2.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((expected_ordering_fraction(3.0).unwrap() - 8.0 / 35.0).abs() < 1e-12);
        assert!((expected_ordering_fraction(4.0).unwrap() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn fixed_seed_sprinkling_is_reproducible_in_all_supported_dimensions() {
        for dimension in 2..=4 {
            let first =
                CausalSetNd::sprinkle_fixed_cardinality(0xC0FFEE, 256, dimension, 2.0).unwrap();
            let second =
                CausalSetNd::sprinkle_fixed_cardinality(0xC0FFEE, 256, dimension, 2.0).unwrap();
            assert_eq!(first.events(), second.events());
        }
    }

    #[test]
    fn lorentz_boost_preserves_pair_counts_order_fraction_and_chain_length() {
        for dimension in 2..=4 {
            let causet =
                CausalSetNd::sprinkle_fixed_cardinality(0x51A7, 384, dimension, 1.0).unwrap();
            for rapidity in [-1.2, -0.35, 0.0, 0.6, 1.2] {
                let boosted = causet.lorentz_boosted(rapidity, 0).unwrap();
                assert_eq!(causet.related_pair_count(), boosted.related_pair_count());
                assert_eq!(
                    causet.interval_size_histogram(),
                    boosted.interval_size_histogram()
                );
                assert_eq!(
                    causet.longest_chain_length(),
                    boosted.longest_chain_length()
                );
                assert!(
                    (causet.ordering_fraction().unwrap() - boosted.ordering_fraction().unwrap())
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn interval_size_histogram_is_dimension_generic_and_boost_invariant() {
        for dimension in 2..=4 {
            let events = (0..4)
                .map(|id| {
                    let mut coordinates = vec![id as f64];
                    coordinates.resize(dimension, 0.0);
                    EventNd { id, coordinates }
                })
                .collect();
            let causet = CausalSetNd::new(dimension, events).unwrap();
            let expected = vec![3, 2, 1];
            assert_eq!(causet.interval_size_histogram(), expected);

            let boosted = causet.lorentz_boosted(0.7, 0).unwrap();
            assert_eq!(boosted.interval_size_histogram(), expected);
        }
    }

    #[test]
    fn interval_histograms_match_brute_force_on_quantized_orders() {
        for dimension in 2..=4 {
            for seed in 0..12_u64 {
                let mut rng = super::SplitMix64::new(seed + 0xD1A0);
                let events = (0..36)
                    .map(|id| {
                        let coordinates = (0..dimension)
                            .map(|_| (rng.unit_f64() * 6.0).floor())
                            .collect();
                        EventNd { id, coordinates }
                    })
                    .collect();
                let causet = CausalSetNd::new(dimension, events).unwrap();
                assert_eq!(
                    causet.interval_size_histogram(),
                    brute_interval_size_histogram(&causet),
                    "D={dimension}, seed={seed}"
                );
            }
        }
    }

    #[test]
    fn deterministic_samples_recover_myrheim_meyer_dimension() {
        // Statistical smoke tests; exact acceptance bounds are seeded and
        // deliberately wider than the expected Monte Carlo standard error.
        for (dimension, tolerance) in [(2, 0.08), (3, 0.055), (4, 0.045)] {
            let causet = CausalSetNd::sprinkle_fixed_cardinality(
                0xD1A0 + dimension as u64,
                1_500,
                dimension,
                1.0,
            )
            .unwrap();
            let observed = causet.ordering_fraction().unwrap();
            let expected = expected_ordering_fraction(dimension as f64).unwrap();
            assert!(
                (observed - expected).abs() < tolerance,
                "D={dimension}, observed={observed}, expected={expected}"
            );
            let estimated = causet.estimate_myrheim_meyer_dimension(8.0).unwrap();
            assert!(
                (estimated - dimension as f64).abs() < 0.45,
                "D={dimension}, estimated={estimated}"
            );
        }
    }

    #[test]
    fn malformed_events_and_boosts_are_rejected() {
        assert!(matches!(
            CausalSetNd::sprinkle_fixed_cardinality(1, 10, 5, 1.0),
            Err(CausalSetNdError::UnsupportedDimension { .. })
        ));
        assert!(CausalSetNd::sprinkle_fixed_cardinality(1, 10, 3, 0.0).is_err());
        assert!(
            CausalSetNd::new(
                2,
                vec![EventNd {
                    id: 0,
                    coordinates: vec![f64::NAN, 0.0]
                }]
            )
            .is_err()
        );
        let valid = CausalSetNd::sprinkle_fixed_cardinality(1, 8, 3, 1.0).unwrap();
        assert!(valid.lorentz_boosted(f64::INFINITY, 0).is_err());
        assert!(valid.lorentz_boosted(0.5, 3).is_err());
    }
}
