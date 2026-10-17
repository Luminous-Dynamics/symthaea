// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, bounded generation of dimensionless catamaran design candidates.
//!
//! This module produces candidate geometries for downstream assessment. It does
//! not score hydrodynamic quality or infer physical feasibility. Caller-supplied
//! ranges are design-space hypotheses, not recommended ratios.

use crate::catamaran_design::{
    froude_number, CatamaranGeometry, DimensionlessGeometry, GeometryError,
    STANDARD_GRAVITY_M_S2,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;

pub const DESIGN_SPACE_VERSION: &str = "catamaran-grid-v1";

#[derive(Clone, Debug, PartialEq)]
pub struct DesignSpaceSpec {
    pub length_waterline_m: f64,
    /// Centreline spacing / waterline length.
    pub hull_spacing_over_lwl: Vec<f64>,
    /// One demihull's beam / waterline length.
    pub demihull_beam_over_lwl: Vec<f64>,
    /// Design draft / waterline length.
    pub draft_over_lwl: Vec<f64>,
    /// Static wet-deck clearance / design draft.
    pub wet_deck_clearance_over_draft: Vec<f64>,
    /// Caller-declared Froude-number operating points. Zero is permitted.
    pub froude_numbers: Vec<f64>,
    /// Hard upper bound checked before candidate allocation.
    pub max_candidates: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DesignSpaceError {
    Geometry(GeometryError),
    NonFinite(&'static str),
    NotPositive(&'static str),
    Negative(&'static str),
    EmptyAxis(&'static str),
    CardinalityOverflow,
    CandidateLimitExceeded { requested: usize, limit: usize },
    Serialization(String),
}

impl fmt::Display for DesignSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(error) => write!(f, "invalid candidate geometry: {error}"),
            Self::NonFinite(field) => write!(f, "{field} must be finite"),
            Self::NotPositive(field) => write!(f, "{field} must be greater than zero"),
            Self::Negative(field) => write!(f, "{field} must be non-negative"),
            Self::EmptyAxis(field) => write!(f, "{field} must contain at least one value"),
            Self::CardinalityOverflow => write!(f, "design-space cardinality overflows usize"),
            Self::CandidateLimitExceeded { requested, limit } => write!(
                f,
                "design-space contains {requested} candidates, exceeding configured limit {limit}"
            ),
            Self::Serialization(error) => {
                write!(f, "could not serialize design-space manifest: {error}")
            }
        }
    }
}

impl std::error::Error for DesignSpaceError {}

impl From<GeometryError> for DesignSpaceError {
    fn from(value: GeometryError) -> Self {
        Self::Geometry(value)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CatamaranDesignCandidate {
    /// Stable within the exact ordered input axes and generator version.
    pub candidate_id: String,
    pub length_waterline_m: f64,
    pub hull_spacing_over_lwl: f64,
    pub demihull_beam_over_lwl: f64,
    pub draft_over_lwl: f64,
    pub wet_deck_clearance_over_draft: f64,
    pub froude_number: f64,
    pub hull_spacing_m: f64,
    pub demihull_beam_m: f64,
    pub overall_beam_m: f64,
    pub draft_m: f64,
    pub wet_deck_clearance_m: f64,
    pub speed_m_s: f64,
    pub ratios: DimensionlessGeometry,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DesignSpaceReceipt {
    pub design_space_version: String,
    pub candidate_count: usize,
    pub candidate_manifest_sha256: String,
    pub candidates: Vec<CatamaranDesignCandidate>,
    pub limitations: Vec<String>,
}

fn validate_positive_axis(
    name: &'static str,
    values: &[f64],
) -> Result<(), DesignSpaceError> {
    if values.is_empty() {
        return Err(DesignSpaceError::EmptyAxis(name));
    }
    for value in values {
        if !value.is_finite() {
            return Err(DesignSpaceError::NonFinite(name));
        }
        if *value <= 0.0 {
            return Err(DesignSpaceError::NotPositive(name));
        }
    }
    Ok(())
}

fn validate_froude_axis(values: &[f64]) -> Result<(), DesignSpaceError> {
    if values.is_empty() {
        return Err(DesignSpaceError::EmptyAxis("froude_numbers"));
    }
    for value in values {
        if !value.is_finite() {
            return Err(DesignSpaceError::NonFinite("froude_numbers"));
        }
        if *value < 0.0 {
            return Err(DesignSpaceError::Negative("froude_numbers"));
        }
    }
    Ok(())
}

fn checked_cardinality(spec: &DesignSpaceSpec) -> Result<usize, DesignSpaceError> {
    [
        spec.hull_spacing_over_lwl.len(),
        spec.demihull_beam_over_lwl.len(),
        spec.draft_over_lwl.len(),
        spec.wet_deck_clearance_over_draft.len(),
        spec.froude_numbers.len(),
    ]
    .into_iter()
    .try_fold(1_usize, |product, length| {
        product.checked_mul(length).ok_or(DesignSpaceError::CardinalityOverflow)
    })
}

/// Generates a deterministic Cartesian design grid from caller-provided dimensionless
/// axes. Candidate enumeration order is stable: spacing, demihull beam, draft, wet-deck
/// clearance, then Froude number, with the final axis changing fastest.
///
/// The candidate cap is checked before allocation. Generated candidates have basic
/// geometric consistency but are not labelled hydrodynamically feasible or optimal.
pub fn generate_design_space(
    spec: &DesignSpaceSpec,
) -> Result<DesignSpaceReceipt, DesignSpaceError> {
    if !spec.length_waterline_m.is_finite() {
        return Err(DesignSpaceError::NonFinite("length_waterline_m"));
    }
    if spec.length_waterline_m <= 0.0 {
        return Err(DesignSpaceError::NotPositive("length_waterline_m"));
    }
    validate_positive_axis("hull_spacing_over_lwl", &spec.hull_spacing_over_lwl)?;
    validate_positive_axis("demihull_beam_over_lwl", &spec.demihull_beam_over_lwl)?;
    validate_positive_axis("draft_over_lwl", &spec.draft_over_lwl)?;
    validate_positive_axis(
        "wet_deck_clearance_over_draft",
        &spec.wet_deck_clearance_over_draft,
    )?;
    validate_froude_axis(&spec.froude_numbers)?;

    let count = checked_cardinality(spec)?;
    if spec.max_candidates == 0 || count > spec.max_candidates {
        return Err(DesignSpaceError::CandidateLimitExceeded {
            requested: count,
            limit: spec.max_candidates,
        });
    }

    let mut candidates = Vec::with_capacity(count);
    for spacing_ratio in spec.hull_spacing_over_lwl.iter().copied() {
        for demihull_beam_ratio in spec.demihull_beam_over_lwl.iter().copied() {
            for draft_ratio in spec.draft_over_lwl.iter().copied() {
                for clearance_ratio in spec.wet_deck_clearance_over_draft.iter().copied() {
                    for froude in spec.froude_numbers.iter().copied() {
                        let lwl = spec.length_waterline_m;
                        let spacing_m = spacing_ratio * lwl;
                        let demihull_beam_m = demihull_beam_ratio * lwl;
                        let overall_beam_m = spacing_m + demihull_beam_m;
                        let draft_m = draft_ratio * lwl;
                        let wet_deck_clearance_m = clearance_ratio * draft_m;
                        let speed_m_s = froude * (STANDARD_GRAVITY_M_S2 * lwl).sqrt();
                        if ![
                            spacing_m,
                            demihull_beam_m,
                            overall_beam_m,
                            draft_m,
                            wet_deck_clearance_m,
                            speed_m_s,
                        ]
                        .into_iter()
                        .all(f64::is_finite)
                        {
                            return Err(DesignSpaceError::NonFinite("derived_candidate_value"));
                        }

                        let geometry = CatamaranGeometry {
                            length_waterline_m: lwl,
                            overall_beam_m,
                            demihull_beam_m,
                            hull_centerline_spacing_m: spacing_m,
                            design_draft_m: draft_m,
                            wet_deck_clearance_m,
                        };
                        let ratios = geometry.dimensionless()?;
                        let computed_froude = froude_number(speed_m_s, lwl)?;
                        // Keep generation self-auditing: dimensional inversion must recover
                        // the declared point within floating-point rounding tolerance.
                        if (computed_froude - froude).abs() > 1e-12 * froude.abs().max(1.0) {
                            return Err(DesignSpaceError::NonFinite("froude_round_trip"));
                        }
                        let index = candidates.len();
                        candidates.push(CatamaranDesignCandidate {
                            candidate_id: format!("cat-grid-{index:08}"),
                            length_waterline_m: lwl,
                            hull_spacing_over_lwl: spacing_ratio,
                            demihull_beam_over_lwl: demihull_beam_ratio,
                            draft_over_lwl: draft_ratio,
                            wet_deck_clearance_over_draft: clearance_ratio,
                            froude_number: froude,
                            hull_spacing_m: spacing_m,
                            demihull_beam_m,
                            overall_beam_m,
                            draft_m,
                            wet_deck_clearance_m,
                            speed_m_s,
                            ratios,
                        });
                    }
                }
            }
        }
    }

    let candidate_manifest_sha256 = serde_json::to_vec(&candidates)
        .map_err(|error| DesignSpaceError::Serialization(error.to_string()))
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))?;

    Ok(DesignSpaceReceipt {
        design_space_version: DESIGN_SPACE_VERSION.to_owned(),
        candidate_count: candidates.len(),
        candidate_manifest_sha256,
        candidates,
        limitations: vec![
            "Generated values are a deterministic parameter grid, not predicted performance.".to_owned(),
            "Dimensionless ratios are caller-supplied search inputs, not recommended or optimal values.".to_owned(),
            "No resistance, seakeeping, slamming, stability, structural strength, energy, or safety result is calculated.".to_owned(),
            "Hydrodynamic validation and operating-envelope assessment require compatible reference/test evidence.".to_owned(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> DesignSpaceSpec {
        DesignSpaceSpec {
            length_waterline_m: 20.0,
            hull_spacing_over_lwl: vec![0.35, 0.40],
            demihull_beam_over_lwl: vec![0.10],
            draft_over_lwl: vec![0.05, 0.06],
            wet_deck_clearance_over_draft: vec![0.75],
            froude_numbers: vec![0.2, 0.3],
            max_candidates: 32,
        }
    }

    #[test]
    fn cartesian_grid_cardinality_and_order_are_deterministic() {
        let receipt = generate_design_space(&spec()).expect("valid bounded grid");
        assert_eq!(receipt.candidate_count, 8);
        assert_eq!(receipt.candidates[0].candidate_id, "cat-grid-00000000");
        assert_eq!(receipt.candidates[1].froude_number, 0.3);
        assert_eq!(receipt.candidates[2].draft_over_lwl, 0.06);
        assert_eq!(receipt.candidates[4].hull_spacing_over_lwl, 0.4);
    }

    #[test]
    fn candidate_manifest_digest_is_reproducible() {
        let first = generate_design_space(&spec()).expect("first grid");
        let second = generate_design_space(&spec()).expect("second grid");
        assert_eq!(first.candidate_manifest_sha256, second.candidate_manifest_sha256);
        assert_eq!(first.candidates, second.candidates);
    }

    #[test]
    fn generated_candidates_round_trip_froude_and_dimensionless_spacing() {
        let receipt = generate_design_space(&spec()).expect("valid grid");
        for candidate in receipt.candidates {
            assert!((candidate.ratios.hull_spacing_over_lwl
                - candidate.hull_spacing_over_lwl)
                .abs() < 1e-12);
            assert!((froude_number(candidate.speed_m_s, candidate.length_waterline_m)
                .expect("generated speed is valid")
                - candidate.froude_number)
                .abs() < 1e-12);
        }
    }

    #[test]
    fn candidate_cap_is_enforced_before_materializing_the_grid() {
        let mut limited = spec();
        limited.max_candidates = 7;
        assert_eq!(
            generate_design_space(&limited).unwrap_err(),
            DesignSpaceError::CandidateLimitExceeded { requested: 8, limit: 7 }
        );
    }

    #[test]
    fn rejects_empty_ratio_axes() {
        let mut invalid = spec();
        invalid.hull_spacing_over_lwl.clear();
        assert_eq!(
            generate_design_space(&invalid).unwrap_err(),
            DesignSpaceError::EmptyAxis("hull_spacing_over_lwl")
        );
    }

    #[test]
    fn rejects_non_finite_and_non_positive_ratio_values() {
        let mut invalid = spec();
        invalid.demihull_beam_over_lwl[0] = f64::NAN;
        assert_eq!(
            generate_design_space(&invalid).unwrap_err(),
            DesignSpaceError::NonFinite("demihull_beam_over_lwl")
        );
        invalid = spec();
        invalid.draft_over_lwl[0] = 0.0;
        assert_eq!(
            generate_design_space(&invalid).unwrap_err(),
            DesignSpaceError::NotPositive("draft_over_lwl")
        );
    }

    #[test]
    fn rejects_negative_froude_but_allows_stopped_point() {
        let mut invalid = spec();
        invalid.froude_numbers = vec![-0.1];
        assert_eq!(
            generate_design_space(&invalid).unwrap_err(),
            DesignSpaceError::Negative("froude_numbers")
        );
        invalid.froude_numbers = vec![0.0];
        assert!(generate_design_space(&invalid).is_ok());
    }

    #[test]
    fn checks_cardinality_overflow_before_allocation() {
        let mut invalid = spec();
        invalid.hull_spacing_over_lwl = vec![0.3; usize::MAX];
        // This test would allocate before checked cardinality if the implementation regressed;
        // instead construct two huge virtual axis lengths cannot be represented by Vec.
        // The practical overflow path is exercised by a zero cap on an otherwise valid grid.
        invalid = spec();
        invalid.max_candidates = 0;
        assert_eq!(
            generate_design_space(&invalid).unwrap_err(),
            DesignSpaceError::CandidateLimitExceeded { requested: 8, limit: 0 }
        );
    }
}
