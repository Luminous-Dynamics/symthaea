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

use std::collections::HashSet;
use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

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
        self.u <= other.u
            && self.v <= other.v
            && (self.u < other.u || self.v < other.v)
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
            Self::InvalidRapidity => write!(f, "rapidity must be finite and its boost scales representable"),
            Self::NonFiniteEvent { id } => write!(f, "event {id} has a non-finite coordinate"),
            Self::NegativeCoordinate { id } => write!(f, "event {id} has a negative null coordinate"),
            Self::EventOutsideBounds { id } => write!(f, "event {id} lies outside the declared interval bounds"),
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
        events: Vec<Event2D>,
        u_extent: f64,
        v_extent: f64,
    ) -> Result<Self, CausalSetError> {
        if !u_extent.is_finite()
            || !v_extent.is_finite()
            || u_extent <= 0.0
            || v_extent <= 0.0
        {
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

        Ok(Self { events, u_extent, v_extent })
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

        let events = self.events.iter().map(|event| Event2D {
            id: event.id,
            u: event.u * u_scale,
            v: event.v * v_scale,
        }).collect();

        Self::new(events, u_extent, v_extent)
    }

    /// Count unordered pairs related by the partial order.
    pub fn related_pair_count(&self) -> u64 {
        let mut count = 0_u64;
        for i in 0..self.events.len() {
            for j in (i + 1)..self.events.len() {
                let a = self.events[i];
                let b = self.events[j];
                if a.causally_precedes(b) || b.causally_precedes(a) {
                    count += 1;
                }
            }
        }
        count
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

    /// Longest-chain event IDs, in causal order. Dynamic programming over
    /// the transitive order takes O(N^2) time and O(N) auxiliary memory.
    pub fn longest_chain_ids(&self) -> Vec<usize> {
        let n = self.events.len();
        if n == 0 {
            return Vec::new();
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            self.events[a]
                .coordinate_time()
                .total_cmp(&self.events[b].coordinate_time())
                .then_with(|| self.events[a].id.cmp(&self.events[b].id))
        });

        let mut best_ending_here = vec![1_usize; n];
        let mut parent: Vec<Option<usize>> = vec![None; n];

        for j in 0..n {
            let event_j = self.events[order[j]];
            for i in 0..j {
                let event_i = self.events[order[i]];
                if event_i.causally_precedes(event_j)
                    && best_ending_here[i] + 1 > best_ending_here[j]
                {
                    best_ending_here[j] = best_ending_here[i] + 1;
                    parent[j] = Some(i);
                }
            }
        }

        let mut end = 0;
        for i in 1..n {
            if best_ending_here[i] > best_ending_here[end] {
                end = i;
            }
        }

        let mut chain = Vec::with_capacity(best_ending_here[end]);
        let mut cursor = Some(end);
        while let Some(index) = cursor {
            chain.push(self.events[order[index]].id);
            cursor = parent[index];
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
    let log_r = ln_gamma(dimension + 1.0)
        + ln_gamma(dimension / 2.0)
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
            assert!((recovered - dimension).abs() < 1e-8, "d={dimension}, recovered={recovered}");
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
        assert!((fraction - 0.5).abs() < 0.06, "ordering fraction={fraction}");
        let dimension = causet.estimate_myrheim_meyer_dimension(8.0).unwrap();
        assert!((dimension - 2.0).abs() < 0.25, "estimated dimension={dimension}");
    }

    #[test]
    fn lorentz_boost_preserves_order_invariants() {
        let causet = CausalSet2D::sprinkle_fixed_cardinality(7, 512, 3.0).unwrap();
        let boosted = causet.lorentz_boosted(1.7).unwrap();
        assert_eq!(causet.related_pair_count(), boosted.related_pair_count());
        assert_eq!(causet.longest_chain_ids(), boosted.longest_chain_ids());
        assert!((causet.ordering_fraction().unwrap() - boosted.ordering_fraction().unwrap()).abs() < 1e-12);
    }

    #[test]
    fn longest_chain_has_expected_square_root_order_of_growth() {
        let causet = CausalSet2D::sprinkle_fixed_cardinality(2026, 1_024, 1.0).unwrap();
        let scaled = causet.longest_chain_length() as f64 / (causet.len() as f64).sqrt();
        assert!((1.25..=2.5).contains(&scaled), "L/sqrt(N)={scaled}");
    }

    #[test]
    fn invalid_coordinates_are_rejected() {
        assert!(CausalSet2D::sprinkle_fixed_cardinality(1, 10, 0.0).is_err());
        assert!(CausalSet2D::new(
            vec![Event2D { id: 0, u: f64::NAN, v: 0.5 }],
            1.0,
            1.0
        ).is_err());
        assert!(CausalSet2D::new(
            vec![Event2D { id: 0, u: 0.2, v: 0.3 }, Event2D { id: 0, u: 0.4, v: 0.5 }],
            1.0,
            1.0
        ).is_err());
    }
}
