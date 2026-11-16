//! Reproducible two-factor design baselines for the optical-ISRU v0.2 tournament.
//!
//! This file is intentionally dependency-free and can be checked independently of
//! the Symthaea workspace:
//!
//!   rustc --edition=2024 --test research/asteroid-optical-isru-v0.2/design_generators.rs -o /tmp/isru-design-tests
//!   /tmp/isru-design-tests
//!   rustc --edition=2024 research/asteroid-optical-isru-v0.2/design_generators.rs -o /tmp/isru-designs
//!   /tmp/isru-designs > designs.csv
//!
//! Coordinates are normalized to [0, 1). The axes represent two design variables
//! only (e.g. normalized irradiance and morphology); this is not a physical model.
//!
//! LHS uses independently shuffled strata and within-stratum jitter. Its PRNG and
//! bounded-integer sampler are local and deterministic so no dependency version
//! or platform RNG state can silently change the design.
//!
//! Sobol is the standard, unscrambled two-dimensional base-2 sequence. Dimension
//! one uses powers-of-two direction numbers. Dimension two uses primitive
//! polynomial x + 1 with initial direction number m_1 = 1 and the s = 1,
//! a = 0 recurrence v_j = v_(j-1) XOR (v_(j-1) >> 1). Index zero is included.
//! No digital shift or scramble is applied; callers must state any boundary
//! exclusion or feasibility filtering they apply.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::io::{self, Write};

const SOBOL_MAX_INDEX_COUNT: u128 = (u32::MAX as u128) + 1;
const SOBOL_SCALE: f64 = 4_294_967_296.0;
const NONZERO_FALLBACK_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesignError {
    ZeroSampleCount,
    ZeroSeed,
    SobolIndexRangeExceeded,
}

impl fmt::Display for DesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSampleCount => write!(f, "sample count must be greater than zero"),
            Self::ZeroSeed => write!(f, "LHS seed must be non-zero"),
            Self::SobolIndexRangeExceeded => {
                write!(f, "Sobol sample count exceeds the supported 32-bit index range")
            }
        }
    }
}

impl Error for DesignError {}

/// Deterministic xorshift64 generator. A zero state is forbidden because it is
/// absorbing for xorshift; public entry points reject a zero seed.
fn next_u64(state: &mut u64) -> u64 {
    let mut value = *state;
    if value == 0 {
        value = NONZERO_FALLBACK_SEED;
    }
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value
}

/// Unbiased integer draw in 0..upper_exclusive using rejection sampling.
fn sample_below(state: &mut u64, upper_exclusive: usize) -> usize {
    debug_assert!(upper_exclusive > 0);
    let bound = upper_exclusive as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = next_u64(state);
        if value >= threshold {
            return (value % bound) as usize;
        }
    }
}

/// Uniform f64 in [0, 1), using the upper 53 random bits.
fn unit_interval(state: &mut u64) -> f64 {
    let mantissa = next_u64(state) >> 11;
    (mantissa as f64) * (1.0 / 9_007_199_254_740_992.0)
}

fn shuffle<T>(values: &mut [T], state: &mut u64) {
    for upper in (1..values.len()).rev() {
        let selected = sample_below(state, upper + 1);
        values.swap(upper, selected);
    }
}

/// Generate a genuine two-dimensional Latin hypercube design.
///
/// Each axis independently receives a permutation of 0..n. Exactly one
/// coordinate is sampled uniformly inside each stratum on each axis.
/// A non-zero seed fully determines the output.
pub fn latin_hypercube_2d(
    sample_count: usize,
    seed: u64,
) -> Result<Vec<Point2>, DesignError> {
    if sample_count == 0 {
        return Err(DesignError::ZeroSampleCount);
    }
    if seed == 0 {
        return Err(DesignError::ZeroSeed);
    }

    let mut state = seed;
    let mut x_strata: Vec<usize> = (0..sample_count).collect();
    let mut y_strata: Vec<usize> = (0..sample_count).collect();
    shuffle(&mut x_strata, &mut state);
    shuffle(&mut y_strata, &mut state);

    let denominator = sample_count as f64;
    let mut points = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let x = (x_strata[index] as f64 + unit_interval(&mut state)) / denominator;
        let y = (y_strata[index] as f64 + unit_interval(&mut state)) / denominator;
        points.push(Point2 { x, y });
    }
    Ok(points)
}

fn sobol_direction_numbers_2d() -> ([u32; 32], [u32; 32]) {
    let mut first = [0_u32; 32];
    let mut second = [0_u32; 32];

    for bit in 0..32 {
        first[bit] = 1_u32 << (31 - bit);
    }

    second[0] = 1_u32 << 31;
    for bit in 1..32 {
        second[bit] = second[bit - 1] ^ (second[bit - 1] >> 1);
    }

    (first, second)
}

/// Generate the standard unscrambled two-dimensional Sobol sequence.
///
/// The first point is (0, 0). The sequence is deterministic and has no seed.
/// At most 2^32 points are supported by this 32-bit direction-number
/// implementation. It is not a scrambled Sobol design.
pub fn sobol_2d(sample_count: usize) -> Result<Vec<Point2>, DesignError> {
    if sample_count == 0 {
        return Err(DesignError::ZeroSampleCount);
    }
    if sample_count as u128 > SOBOL_MAX_INDEX_COUNT {
        return Err(DesignError::SobolIndexRangeExceeded);
    }

    let (first_directions, second_directions) = sobol_direction_numbers_2d();
    let mut points = Vec::with_capacity(sample_count);

    for index in 0..sample_count {
        let index = index as u32;
        let gray = index ^ (index >> 1);
        let mut x_bits = 0_u32;
        let mut y_bits = 0_u32;

        for bit in 0..32 {
            if (gray & (1_u32 << bit)) != 0 {
                x_bits ^= first_directions[bit];
                y_bits ^= second_directions[bit];
            }
        }

        points.push(Point2 {
            x: f64::from(x_bits) / SOBOL_SCALE,
            y: f64::from(y_bits) / SOBOL_SCALE,
        });
    }

    Ok(points)
}

fn emit_csv<W: Write>(
    writer: &mut W,
    sample_count: usize,
    lhs_seed: u64,
) -> Result<(), Box<dyn Error>> {
    writeln!(writer, "algorithm,index,x,y,seed")?;

    for (index, point) in latin_hypercube_2d(sample_count, lhs_seed)?
        .into_iter()
        .enumerate()
    {
        writeln!(
            writer,
            "latin_hypercube_2d,{index},{:.17},{:.17},{lhs_seed}",
            point.x, point.y
        )?;
    }

    for (index, point) in sobol_2d(sample_count)?.into_iter().enumerate() {
        writeln!(
            writer,
            "sobol_2d,{index},{:.17},{:.17},none",
            point.x, point.y
        )?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    emit_csv(&mut output, 16, 0x4953_5255_444f_4531)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stratum(value: f64, count: usize) -> usize {
        ((value * count as f64).floor() as usize).min(count - 1)
    }

    #[test]
    fn lhs_is_reproducible_for_the_same_seed() {
        let first = latin_hypercube_2d(64, 0x1234_5678_9abc_def0).unwrap();
        let second = latin_hypercube_2d(64, 0x1234_5678_9abc_def0).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn lhs_changes_for_a_different_seed() {
        let first = latin_hypercube_2d(64, 0x1234_5678_9abc_def0).unwrap();
        let second = latin_hypercube_2d(64, 0x0fed_cba9_8765_4321).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn lhs_has_exactly_one_point_in_every_stratum_on_both_axes() {
        const COUNT: usize = 97;
        let points = latin_hypercube_2d(COUNT, 0x4953_5255).unwrap();
        let x: BTreeSet<_> = points.iter().map(|point| stratum(point.x, COUNT)).collect();
        let y: BTreeSet<_> = points.iter().map(|point| stratum(point.y, COUNT)).collect();
        assert_eq!(x, (0..COUNT).collect());
        assert_eq!(y, (0..COUNT).collect());
    }

    #[test]
    fn lhs_points_are_finite_and_inside_the_unit_half_open_square() {
        let points = latin_hypercube_2d(128, 0x4953_5255).unwrap();
        assert!(points.iter().all(|point| {
            point.x.is_finite()
                && point.y.is_finite()
                && (0.0..1.0).contains(&point.x)
                && (0.0..1.0).contains(&point.y)
        }));
    }

    #[test]
    fn sobol_first_eight_points_match_the_canonical_unscrambled_sequence() {
        let expected = [
            (0.0, 0.0),
            (0.5, 0.5),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.375, 0.375),
            (0.875, 0.875),
            (0.625, 0.125),
            (0.125, 0.625),
        ];
        let actual = sobol_2d(expected.len()).unwrap();

        for (point, (expected_x, expected_y)) in actual.iter().zip(expected) {
            assert!((point.x - expected_x).abs() < f64::EPSILON);
            assert!((point.y - expected_y).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn sobol_points_are_unique_and_inside_the_unit_half_open_square() {
        let points = sobol_2d(1024).unwrap();
        let identities: BTreeSet<_> = points
            .iter()
            .map(|point| (point.x.to_bits(), point.y.to_bits()))
            .collect();
        assert_eq!(identities.len(), points.len());
        assert!(points.iter().all(|point| {
            point.x.is_finite()
                && point.y.is_finite()
                && (0.0..1.0).contains(&point.x)
                && (0.0..1.0).contains(&point.y)
        }));
    }

    #[test]
    fn zero_sample_count_is_rejected() {
        assert_eq!(
            latin_hypercube_2d(0, 1).unwrap_err(),
            DesignError::ZeroSampleCount
        );
        assert_eq!(sobol_2d(0).unwrap_err(), DesignError::ZeroSampleCount);
    }

    #[test]
    fn zero_lhs_seed_is_rejected() {
        assert_eq!(
            latin_hypercube_2d(8, 0).unwrap_err(),
            DesignError::ZeroSeed
        );
    }

    #[test]
    fn csv_is_stable_and_keeps_sobol_seed_absent() {
        let mut first = Vec::new();
        let mut second = Vec::new();
        emit_csv(&mut first, 8, 42).unwrap();
        emit_csv(&mut second, 8, 42).unwrap();
        assert_eq!(first, second);

        let output = String::from_utf8(first).unwrap();
        assert!(output.starts_with("algorithm,index,x,y,seed\n"));
        assert!(output.contains("sobol_2d,0,0.00000000000000000,0.00000000000000000,none"));
    }
}
