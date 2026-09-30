//! Dimension-scaling observatory contracts for ContinuousHV research.
//!
//! This module separates deterministic representation cost from analytic
//! concentration expectations and from empirical task-quality measurements.
//! It provides a stable machine-readable skeleton for 1K..256K sweeps without
//! implying that larger dimensionality is automatically better.

use super::resolution_space::{HdcResolution, ResolutionClass};

pub const DIMENSION_OBSERVATORY_SCHEMA_VERSION: u32 = 1;

pub const OBSERVATORY_DIMENSIONS: &[usize] =
    &[1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DimensionProfile {
    pub schema_version: u32,
    pub resolution: usize,
    pub resolution_class: ResolutionClass,
    pub continuous_bytes: usize,
    pub binary_bytes: usize,
    /// Standard deviation of the raw dot product for independent
    /// Uniform(-1, 1) vectors: sqrt(D) / 3.
    pub independent_dot_stddev: f64,
    /// Leading-order standard deviation of cosine similarity for independent
    /// isotropic vectors: 1 / sqrt(D).
    pub independent_cosine_stddev: f64,
}

impl DimensionProfile {
    pub fn for_dimension(
        dimension: usize,
    ) -> Result<Self, super::resolution_space::ResolutionError> {
        let resolution = HdcResolution::new(dimension)?;
        let d = dimension as f64;

        Ok(Self {
            schema_version: DIMENSION_OBSERVATORY_SCHEMA_VERSION,
            resolution: dimension,
            resolution_class: resolution.class(),
            continuous_bytes: resolution.f32_bytes()?,
            binary_bytes: resolution.binary_bytes()?,
            independent_dot_stddev: d.sqrt() / 3.0,
            independent_cosine_stddev: 1.0 / d.sqrt(),
        })
    }

    pub fn profiles() -> Vec<Self> {
        OBSERVATORY_DIMENSIONS
            .iter()
            .map(|&dimension| {
                Self::for_dimension(dimension).expect("observatory dimensions are valid")
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_ladder_is_complete_and_ordered() {
        let profiles = DimensionProfile::profiles();
        assert_eq!(profiles.len(), OBSERVATORY_DIMENSIONS.len());

        for (profile, &dimension) in profiles.iter().zip(OBSERVATORY_DIMENSIONS) {
            assert_eq!(profile.resolution, dimension);
            assert_eq!(profile.schema_version, DIMENSION_OBSERVATORY_SCHEMA_VERSION);
        }

        assert!(profiles
            .windows(2)
            .all(|pair| pair[0].resolution < pair[1].resolution));
    }

    #[test]
    fn representation_cost_doubles_with_dimension() {
        let profiles = DimensionProfile::profiles();
        for pair in profiles.windows(2) {
            assert_eq!(pair[1].continuous_bytes, pair[0].continuous_bytes * 2);
            assert_eq!(pair[1].binary_bytes, pair[0].binary_bytes * 2);
        }
    }

    #[test]
    fn cosine_concentration_improves_with_dimension() {
        let profiles = DimensionProfile::profiles();
        for pair in profiles.windows(2) {
            assert!(pair[1].independent_cosine_stddev < pair[0].independent_cosine_stddev);
        }
    }

    #[test]
    fn raw_dot_noise_grows_but_relative_cosine_noise_shrinks() {
        let profiles = DimensionProfile::profiles();
        for pair in profiles.windows(2) {
            assert!(pair[1].independent_dot_stddev > pair[0].independent_dot_stddev);
            assert!(
                pair[1].independent_cosine_stddev < pair[0].independent_cosine_stddev
            );
        }
    }

    #[test]
    fn canonical_and_exploratory_boundary_is_explicit() {
        let profiles = DimensionProfile::profiles();
        assert_eq!(profiles[6].resolution_class, ResolutionClass::Canonical);
        assert_eq!(profiles[7].resolution_class, ResolutionClass::Exploratory);
        assert_eq!(profiles[8].resolution_class, ResolutionClass::Exploratory);
    }
}
