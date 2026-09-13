// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Bounded path-class witnesses for explicitly supported topological spaces.
//!
//! V1 supports only `S^1` and the abstract product torus `T^2 = S^1 x S^1`.
//! A witness classifies the exact **piecewise-shortest-arc interpolation** of
//! supplied samples under an identity-bound ambiguity policy. It does not infer
//! how an undersampled physical trajectory moved between observations and is not
//! a generic homotopy oracle.

use std::f64::consts::{PI, TAU};

use blake3::Hasher;
use thiserror::Error;

/// Maximum sample count accepted by the V1 bounded witness implementation.
pub const MAX_WINDING_SAMPLES: usize = 1_000_000;

/// Exact interpretation policy for sampled winding witnesses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindingPolicy {
    closure_tolerance: f64,
    antipodal_ambiguity_margin: f64,
    integrality_tolerance: f64,
    max_samples: usize,
}

impl WindingPolicy {
    /// Construct an exact winding-witness policy.
    ///
    /// `antipodal_ambiguity_margin` excludes segments whose shortest arc is too
    /// close to `pi`, where clockwise/counter-clockwise interpolation ceases to
    /// be uniquely justified by the endpoint samples alone.
    pub fn new(
        closure_tolerance: f64,
        antipodal_ambiguity_margin: f64,
        integrality_tolerance: f64,
        max_samples: usize,
    ) -> Result<Self, TopologyWitnessError> {
        if !closure_tolerance.is_finite()
            || closure_tolerance <= 0.0
            || closure_tolerance >= PI
        {
            return Err(TopologyWitnessError::InvalidPolicy {
                reason: format!(
                    "closure tolerance must be finite and in (0, pi), got {closure_tolerance}"
                ),
            });
        }
        if !antipodal_ambiguity_margin.is_finite()
            || antipodal_ambiguity_margin <= 0.0
            || antipodal_ambiguity_margin >= PI
        {
            return Err(TopologyWitnessError::InvalidPolicy {
                reason: format!(
                    "antipodal ambiguity margin must be finite and in (0, pi), got {antipodal_ambiguity_margin}"
                ),
            });
        }
        if !integrality_tolerance.is_finite()
            || integrality_tolerance <= 0.0
            || integrality_tolerance >= 0.5
        {
            return Err(TopologyWitnessError::InvalidPolicy {
                reason: format!(
                    "integrality tolerance must be finite and in (0, 0.5), got {integrality_tolerance}"
                ),
            });
        }
        if !(2..=MAX_WINDING_SAMPLES).contains(&max_samples) {
            return Err(TopologyWitnessError::InvalidPolicy {
                reason: format!(
                    "max_samples must be in 2..={MAX_WINDING_SAMPLES}, got {max_samples}"
                ),
            });
        }
        Ok(Self {
            closure_tolerance,
            antipodal_ambiguity_margin,
            integrality_tolerance,
            max_samples,
        })
    }

    /// Named deterministic software-reference policy.
    pub fn reference_v1() -> Self {
        Self::new(1e-9, 1e-9, 1e-9, 65_536).expect("reference winding policy is valid")
    }

    /// Endpoint closure tolerance in radians.
    pub fn closure_tolerance(&self) -> f64 {
        self.closure_tolerance
    }

    /// Exclusion margin around an antipodal `pi` segment.
    pub fn antipodal_ambiguity_margin(&self) -> f64 {
        self.antipodal_ambiguity_margin
    }

    /// Allowed distance from the nearest integer winding after lifted summation.
    pub fn integrality_tolerance(&self) -> f64 {
        self.integrality_tolerance
    }

    /// Maximum accepted sample count.
    pub fn max_samples(&self) -> usize {
        self.max_samples
    }

    /// Deterministic identity of this exact interpretation policy.
    pub fn identity(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-winding-policy-v1\0");
        hasher.update(&self.closure_tolerance.to_bits().to_le_bytes());
        hasher.update(&self.antipodal_ambiguity_margin.to_bits().to_le_bytes());
        hasher.update(&self.integrality_tolerance.to_bits().to_le_bytes());
        hasher.update(&(self.max_samples as u64).to_le_bytes());
        *hasher.finalize().as_bytes()
    }
}

impl Default for WindingPolicy {
    fn default() -> Self {
        Self::reference_v1()
    }
}

/// Fail-closed errors for bounded topology witnesses.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum TopologyWitnessError {
    /// Interpretation policy is malformed.
    #[error("invalid winding policy: {reason}")]
    InvalidPolicy { reason: String },
    /// A path must contain at least two samples.
    #[error("path requires at least two samples, got {actual}")]
    TooFewSamples { actual: usize },
    /// The explicit sample budget was exceeded.
    #[error("path has {actual} samples, exceeding policy maximum {maximum}")]
    TooManySamples { actual: usize, maximum: usize },
    /// A supplied angular coordinate is NaN or infinite.
    #[error("non-finite angular sample at sample {sample}, coordinate {coordinate}")]
    NonFiniteSample { sample: usize, coordinate: usize },
    /// The explicit endpoint does not close under the selected angular metric.
    #[error(
        "path is not closed on coordinate {coordinate}: endpoint distance {distance} exceeds tolerance {tolerance}"
    )]
    NotClosed {
        coordinate: usize,
        distance: f64,
        tolerance: f64,
    },
    /// A sampled segment is too close to antipodal to select one shortest arc.
    #[error(
        "segment {segment}, coordinate {coordinate} is shortest-arc ambiguous: |delta|={absolute_delta}, allowed < {maximum_unique_delta}"
    )]
    AmbiguousShortestArc {
        segment: usize,
        coordinate: usize,
        absolute_delta: f64,
        maximum_unique_delta: f64,
    },
    /// Lifted turn count is not sufficiently close to an integer under policy.
    #[error(
        "lifted path on coordinate {coordinate} gives {turns} turns, not within {tolerance} of integer {nearest_integer}"
    )]
    NonIntegralWinding {
        coordinate: usize,
        turns: f64,
        nearest_integer: i64,
        tolerance: f64,
    },
    /// A derived arithmetic quantity became non-finite.
    #[error("non-finite winding arithmetic at {stage}")]
    NonFiniteComputation { stage: &'static str },
}

/// Exact winding-class witness for one sampled `S^1` loop.
#[derive(Clone, Debug, PartialEq)]
pub struct CircleWindingWitness {
    winding: i64,
    lifted_angle: f64,
    path_identity: [u8; 32],
    policy_identity: [u8; 32],
    witness_identity: [u8; 32],
}

impl CircleWindingWitness {
    /// Integer winding number of the declared piecewise-shortest loop.
    pub fn winding(&self) -> i64 {
        self.winding
    }

    /// Accumulated lifted angular change, including the explicit closing segment.
    pub fn lifted_angle(&self) -> f64 {
        self.lifted_angle
    }

    /// Identity of canonicalized samples under the selected path interpretation.
    pub fn path_identity(&self) -> [u8; 32] {
        self.path_identity
    }

    /// Exact winding-policy identity.
    pub fn policy_identity(&self) -> [u8; 32] {
        self.policy_identity
    }

    /// Identity binding path, policy and resulting winding class.
    pub fn identity(&self) -> [u8; 32] {
        self.witness_identity
    }
}

/// Exact pair of fundamental winding numbers for an abstract `T^2` sampled loop.
#[derive(Clone, Debug, PartialEq)]
pub struct TorusWindingWitness {
    winding: [i64; 2],
    lifted_angles: [f64; 2],
    path_identity: [u8; 32],
    policy_identity: [u8; 32],
    witness_identity: [u8; 32],
}

impl TorusWindingWitness {
    /// Winding pair `(m, n)` in the declared ordered angular coordinates.
    pub fn winding(&self) -> [i64; 2] {
        self.winding
    }

    /// Accumulated lifted angular changes for both ordered circle factors.
    pub fn lifted_angles(&self) -> [f64; 2] {
        self.lifted_angles
    }

    /// Identity of canonicalized angular samples.
    pub fn path_identity(&self) -> [u8; 32] {
        self.path_identity
    }

    /// Exact winding-policy identity.
    pub fn policy_identity(&self) -> [u8; 32] {
        self.policy_identity
    }

    /// Identity binding path, policy and winding pair.
    pub fn identity(&self) -> [u8; 32] {
        self.witness_identity
    }
}

/// Classify a sampled `S^1` loop under an exact piecewise-shortest policy.
pub fn circle_winding_witness(
    samples: &[f64],
    policy: WindingPolicy,
) -> Result<CircleWindingWitness, TopologyWitnessError> {
    validate_sample_count(samples.len(), policy)?;
    let canonical = samples
        .iter()
        .enumerate()
        .map(|(sample, &angle)| canonical_angle(angle, sample, 0))
        .collect::<Result<Vec<_>, _>>()?;

    require_closed(&canonical, 0, policy)?;
    let (winding, lifted_angle) = winding_for_coordinate(&canonical, 0, policy)?;
    let policy_identity = policy.identity();
    let path_identity = hash_circle_path(&canonical);
    let witness_identity = hash_circle_witness(path_identity, policy_identity, winding);
    Ok(CircleWindingWitness {
        winding,
        lifted_angle,
        path_identity,
        policy_identity,
        witness_identity,
    })
}

/// Classify a sampled loop in the abstract product torus `T^2 = S^1 x S^1`.
///
/// Samples are ordered angular coordinates `[theta, phi]`; this function does
/// not infer those coordinates from an arbitrary `R^3` torus embedding.
pub fn torus_winding_witness(
    samples: &[[f64; 2]],
    policy: WindingPolicy,
) -> Result<TorusWindingWitness, TopologyWitnessError> {
    validate_sample_count(samples.len(), policy)?;
    let mut canonical = Vec::with_capacity(samples.len());
    for (sample_index, sample) in samples.iter().enumerate() {
        canonical.push([
            canonical_angle(sample[0], sample_index, 0)?,
            canonical_angle(sample[1], sample_index, 1)?,
        ]);
    }

    for coordinate in 0..2 {
        let projected: Vec<f64> = canonical.iter().map(|sample| sample[coordinate]).collect();
        require_closed(&projected, coordinate, policy)?;
    }

    let first: Vec<f64> = canonical.iter().map(|sample| sample[0]).collect();
    let second: Vec<f64> = canonical.iter().map(|sample| sample[1]).collect();
    let (w0, a0) = winding_for_coordinate(&first, 0, policy)?;
    let (w1, a1) = winding_for_coordinate(&second, 1, policy)?;
    let winding = [w0, w1];
    let lifted_angles = [a0, a1];
    let policy_identity = policy.identity();
    let path_identity = hash_torus_path(&canonical);
    let witness_identity = hash_torus_witness(path_identity, policy_identity, winding);
    Ok(TorusWindingWitness {
        winding,
        lifted_angles,
        path_identity,
        policy_identity,
        witness_identity,
    })
}

fn validate_sample_count(
    actual: usize,
    policy: WindingPolicy,
) -> Result<(), TopologyWitnessError> {
    if actual < 2 {
        return Err(TopologyWitnessError::TooFewSamples { actual });
    }
    if actual > policy.max_samples {
        return Err(TopologyWitnessError::TooManySamples {
            actual,
            maximum: policy.max_samples,
        });
    }
    Ok(())
}

fn canonical_angle(
    angle: f64,
    sample: usize,
    coordinate: usize,
) -> Result<f64, TopologyWitnessError> {
    if !angle.is_finite() {
        return Err(TopologyWitnessError::NonFiniteSample { sample, coordinate });
    }
    // +0.0 canonicalizes IEEE-754 negative zero.
    Ok(((angle + PI).rem_euclid(TAU) - PI) + 0.0)
}

fn shortest_delta(from: f64, to: f64) -> f64 {
    ((to - from + PI).rem_euclid(TAU) - PI) + 0.0
}

fn require_closed(
    samples: &[f64],
    coordinate: usize,
    policy: WindingPolicy,
) -> Result<(), TopologyWitnessError> {
    let distance = shortest_delta(samples[0], samples[samples.len() - 1]).abs();
    if distance > policy.closure_tolerance {
        Err(TopologyWitnessError::NotClosed {
            coordinate,
            distance,
            tolerance: policy.closure_tolerance,
        })
    } else {
        Ok(())
    }
}

fn winding_for_coordinate(
    samples: &[f64],
    coordinate: usize,
    policy: WindingPolicy,
) -> Result<(i64, f64), TopologyWitnessError> {
    let mut lifted = 0.0_f64;
    for segment in 0..samples.len() {
        let from = samples[segment];
        let to = if segment + 1 < samples.len() {
            samples[segment + 1]
        } else {
            samples[0]
        };
        let delta = shortest_delta(from, to);
        let absolute_delta = delta.abs();
        let maximum_unique_delta = PI - policy.antipodal_ambiguity_margin;
        if absolute_delta >= maximum_unique_delta {
            return Err(TopologyWitnessError::AmbiguousShortestArc {
                segment,
                coordinate,
                absolute_delta,
                maximum_unique_delta,
            });
        }
        lifted += delta;
        if !lifted.is_finite() {
            return Err(TopologyWitnessError::NonFiniteComputation {
                stage: "lifted angular accumulation",
            });
        }
    }

    let turns = lifted / TAU;
    if !turns.is_finite() {
        return Err(TopologyWitnessError::NonFiniteComputation {
            stage: "turn normalization",
        });
    }
    let rounded = turns.round();
    if rounded.abs() > i64::MAX as f64 {
        return Err(TopologyWitnessError::NonFiniteComputation {
            stage: "integer winding conversion",
        });
    }
    let nearest_integer = rounded as i64;
    if (turns - rounded).abs() > policy.integrality_tolerance {
        return Err(TopologyWitnessError::NonIntegralWinding {
            coordinate,
            turns,
            nearest_integer,
            tolerance: policy.integrality_tolerance,
        });
    }
    Ok((nearest_integer, lifted))
}

fn hash_circle_path(samples: &[f64]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-circle-sampled-path-v1\0");
    hasher.update(&(samples.len() as u64).to_le_bytes());
    for sample in samples {
        hasher.update(&sample.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn hash_torus_path(samples: &[[f64; 2]]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-torus2-sampled-path-v1\0");
    hasher.update(&(samples.len() as u64).to_le_bytes());
    for sample in samples {
        hasher.update(&sample[0].to_bits().to_le_bytes());
        hasher.update(&sample[1].to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn hash_circle_witness(path: [u8; 32], policy: [u8; 32], winding: i64) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-circle-winding-witness-v1\0");
    hasher.update(&path);
    hasher.update(&policy);
    hasher.update(&winding.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn hash_torus_witness(path: [u8; 32], policy: [u8; 32], winding: [i64; 2]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-torus2-winding-witness-v1\0");
    hasher.update(&path);
    hasher.update(&policy);
    hasher.update(&winding[0].to_le_bytes());
    hasher.update(&winding[1].to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_positive_and_contractible_classes_are_distinct() {
        let policy = WindingPolicy::reference_v1();
        let once = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0, TAU];
        let contractible = [0.0, 0.4, -0.2, 0.0];
        let once = circle_winding_witness(&once, policy).unwrap();
        let contractible = circle_winding_witness(&contractible, policy).unwrap();
        assert_eq!(once.winding(), 1);
        assert_eq!(contractible.winding(), 0);
        assert_ne!(once.identity(), contractible.identity());
    }

    #[test]
    fn equivalent_angle_representatives_have_same_path_identity() {
        let policy = WindingPolicy::reference_v1();
        let a = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0, TAU];
        let b = [0.0, PI / 2.0, -PI, -PI / 2.0, 0.0];
        let a = circle_winding_witness(&a, policy).unwrap();
        let b = circle_winding_witness(&b, policy).unwrap();
        assert_eq!(a.winding(), b.winding());
        assert_eq!(a.path_identity(), b.path_identity());
        assert_eq!(a.identity(), b.identity());
    }

    #[test]
    fn antipodal_segment_is_rejected_as_sampling_ambiguity() {
        let policy = WindingPolicy::reference_v1();
        let error = circle_winding_witness(&[0.0, PI, 0.0], policy).unwrap_err();
        assert!(matches!(
            error,
            TopologyWitnessError::AmbiguousShortestArc { segment: 0, .. }
        ));
    }

    #[test]
    fn open_path_is_not_promoted_to_loop_class() {
        let policy = WindingPolicy::reference_v1();
        let error = circle_winding_witness(&[0.0, 0.5, 1.0], policy).unwrap_err();
        assert!(matches!(error, TopologyWitnessError::NotClosed { .. }));
    }

    #[test]
    fn torus_witness_tracks_two_ordered_windings() {
        let policy = WindingPolicy::reference_v1();
        let samples: Vec<[f64; 2]> = (0..=8)
            .map(|index| {
                let t = index as f64 / 8.0;
                [TAU * t, -2.0 * TAU * t]
            })
            .collect();
        let witness = torus_winding_witness(&samples, policy).unwrap();
        assert_eq!(witness.winding(), [1, -2]);
    }

    #[test]
    fn resampling_can_preserve_class_without_claiming_same_path_identity() {
        let policy = WindingPolicy::reference_v1();
        let coarse: Vec<f64> = (0..=4).map(|i| TAU * i as f64 / 4.0).collect();
        let fine: Vec<f64> = (0..=16).map(|i| TAU * i as f64 / 16.0).collect();
        let coarse = circle_winding_witness(&coarse, policy).unwrap();
        let fine = circle_winding_witness(&fine, policy).unwrap();
        assert_eq!(coarse.winding(), 1);
        assert_eq!(fine.winding(), 1);
        assert_ne!(coarse.path_identity(), fine.path_identity());
    }

    #[test]
    fn policy_changes_witness_identity() {
        let a_policy = WindingPolicy::reference_v1();
        let b_policy = WindingPolicy::new(1e-8, 1e-9, 1e-9, 65_536).unwrap();
        let samples = [0.0, PI / 2.0, PI, -PI / 2.0, 0.0];
        let a = circle_winding_witness(&samples, a_policy).unwrap();
        let b = circle_winding_witness(&samples, b_policy).unwrap();
        assert_eq!(a.winding(), b.winding());
        assert_ne!(a.policy_identity(), b.policy_identity());
        assert_ne!(a.identity(), b.identity());
    }

    #[test]
    fn non_finite_samples_fail_closed() {
        let error = circle_winding_witness(&[0.0, f64::NAN, 0.0], WindingPolicy::reference_v1())
            .unwrap_err();
        assert_eq!(
            error,
            TopologyWitnessError::NonFiniteSample {
                sample: 1,
                coordinate: 0
            }
        );
    }
}