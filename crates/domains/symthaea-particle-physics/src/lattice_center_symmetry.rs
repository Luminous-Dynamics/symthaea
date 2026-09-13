// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `Z_3` center-symmetry diagnostics for pure-SU(3) Polyakov-loop histories.
//!
//! Raw complex Polyakov components are not center invariant. This module keeps
//! scalar center-invariant observables, categorical center-sector mobility, and
//! caller-declared acceptance policy separate.

use crate::symmetry_groups::Complex;
use std::collections::BTreeSet;

pub const Z3_CENTER_DIAGNOSTIC_ID: &str = "pure_su3_z3_center_diagnostics_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Z3CenterSector {
    ZeroPhase,
    PositivePhase,
    NegativePhase,
}

impl Z3CenterSector {
    pub const ALL: [Self; 3] = [Self::ZeroPhase, Self::PositivePhase, Self::NegativePhase];

    pub fn index(self) -> usize {
        match self {
            Self::ZeroPhase => 0,
            Self::PositivePhase => 1,
            Self::NegativePhase => 2,
        }
    }

    pub fn phase(self) -> f64 {
        match self {
            Self::ZeroPhase => 0.0,
            Self::PositivePhase => 2.0 * std::f64::consts::PI / 3.0,
            Self::NegativePhase => -2.0 * std::f64::consts::PI / 3.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolyakovChainTrace {
    pub chain_id: String,
    pub samples: Vec<Complex>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CenterSectorChainDiagnostics {
    pub diagnostic_id: &'static str,
    pub chain_id: String,
    pub sample_count: usize,
    pub classified_count: usize,
    pub ambiguous_count: usize,
    /// Counts ordered as `[0, +2pi/3, -2pi/3]`.
    pub sector_counts: [usize; 3],
    /// Ambiguous samples break continuity and cannot manufacture transitions.
    pub sector_transition_count: usize,
    pub maximum_classified_sector_dwell: usize,
    pub maximum_ambiguous_run: usize,
    pub mean_magnitude: f64,
    pub mean_center_aligned_real: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CenterSectorMobilityPolicy {
    pub minimum_polyakov_magnitude: f64,
    pub minimum_classified_fraction: f64,
    pub minimum_sector_transitions_per_chain: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CenterSectorMobilityAssessment {
    pub diagnostic_id: &'static str,
    pub minimum_polyakov_magnitude: f64,
    pub minimum_classified_fraction: f64,
    pub minimum_sector_transitions_per_chain: usize,
    pub chains: Vec<CenterSectorChainDiagnostics>,
    pub every_chain_meets_classified_fraction: bool,
    pub every_chain_meets_transition_count: bool,
    pub meets_declared_policy: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CenterSymmetryError {
    InvalidMinimumMagnitude(f64),
    InvalidMinimumClassifiedFraction(f64),
    NonFinitePolyakovValue,
    EmptyTraceSet,
    EmptyChainId { index: usize },
    DuplicateChainId(String),
    EmptyChainSamples { chain_id: String },
    NonFinitePolyakov { chain_id: String, index: usize },
}

fn complex_magnitude(value: Complex) -> f64 {
    (value.re * value.re + value.im * value.im).sqrt()
}

fn complex_mul(left: Complex, right: Complex) -> Complex {
    Complex::new(
        left.re * right.re - left.im * right.im,
        left.re * right.im + left.im * right.re,
    )
}

fn wrapped_angle_distance(left: f64, right: f64) -> f64 {
    let delta = left - right;
    delta.sin().atan2(delta.cos()).abs()
}

pub fn classify_z3_center_sector(
    value: Complex,
    minimum_magnitude: f64,
) -> Result<Option<Z3CenterSector>, CenterSymmetryError> {
    if !minimum_magnitude.is_finite() || minimum_magnitude < 0.0 {
        return Err(CenterSymmetryError::InvalidMinimumMagnitude(minimum_magnitude));
    }
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(CenterSymmetryError::NonFinitePolyakovValue);
    }
    let magnitude = complex_magnitude(value);
    if magnitude <= minimum_magnitude {
        return Ok(None);
    }

    let phase = value.im.atan2(value.re);
    let mut best = Z3CenterSector::ZeroPhase;
    let mut best_distance = f64::INFINITY;
    for sector in Z3CenterSector::ALL {
        let distance = wrapped_angle_distance(phase, sector.phase());
        if distance < best_distance {
            best = sector;
            best_distance = distance;
        }
    }
    Ok(Some(best))
}

/// Rotate a classified Polyakov loop into the zero-phase center sector.
pub fn center_align_polyakov(
    value: Complex,
    minimum_magnitude: f64,
) -> Result<Option<Complex>, CenterSymmetryError> {
    let Some(sector) = classify_z3_center_sector(value, minimum_magnitude)? else {
        return Ok(None);
    };
    let phase = -sector.phase();
    Ok(Some(complex_mul(
        value,
        Complex::new(phase.cos(), phase.sin()),
    )))
}

pub fn diagnose_center_sector_chain(
    trace: &PolyakovChainTrace,
    minimum_magnitude: f64,
) -> Result<CenterSectorChainDiagnostics, CenterSymmetryError> {
    if trace.chain_id.trim().is_empty() {
        return Err(CenterSymmetryError::EmptyChainId { index: 0 });
    }
    if trace.samples.is_empty() {
        return Err(CenterSymmetryError::EmptyChainSamples {
            chain_id: trace.chain_id.clone(),
        });
    }

    let mut sector_counts = [0usize; 3];
    let mut classified_count = 0usize;
    let mut ambiguous_count = 0usize;
    let mut sector_transition_count = 0usize;
    let mut maximum_classified_sector_dwell = 0usize;
    let mut current_dwell = 0usize;
    let mut previous_sector: Option<Z3CenterSector> = None;
    let mut maximum_ambiguous_run = 0usize;
    let mut ambiguous_run = 0usize;
    let mut magnitude_sum = 0.0;
    let mut aligned_real_sum = 0.0;

    for (index, &sample) in trace.samples.iter().enumerate() {
        if !sample.re.is_finite() || !sample.im.is_finite() {
            return Err(CenterSymmetryError::NonFinitePolyakov {
                chain_id: trace.chain_id.clone(),
                index,
            });
        }
        magnitude_sum += complex_magnitude(sample);
        match classify_z3_center_sector(sample, minimum_magnitude)? {
            Some(sector) => {
                classified_count += 1;
                sector_counts[sector.index()] += 1;
                ambiguous_run = 0;
                match previous_sector {
                    Some(previous) if previous == sector => current_dwell += 1,
                    Some(_) => {
                        sector_transition_count += 1;
                        current_dwell = 1;
                    }
                    None => current_dwell = 1,
                }
                maximum_classified_sector_dwell =
                    maximum_classified_sector_dwell.max(current_dwell);
                previous_sector = Some(sector);
                aligned_real_sum += center_align_polyakov(sample, minimum_magnitude)?
                    .expect("classified sample must center-align")
                    .re;
            }
            None => {
                ambiguous_count += 1;
                ambiguous_run += 1;
                maximum_ambiguous_run = maximum_ambiguous_run.max(ambiguous_run);
                current_dwell = 0;
                previous_sector = None;
            }
        }
    }

    Ok(CenterSectorChainDiagnostics {
        diagnostic_id: Z3_CENTER_DIAGNOSTIC_ID,
        chain_id: trace.chain_id.clone(),
        sample_count: trace.samples.len(),
        classified_count,
        ambiguous_count,
        sector_counts,
        sector_transition_count,
        maximum_classified_sector_dwell,
        maximum_ambiguous_run,
        mean_magnitude: magnitude_sum / trace.samples.len() as f64,
        mean_center_aligned_real: (classified_count > 0)
            .then_some(aligned_real_sum / classified_count as f64),
    })
}

pub fn assess_center_sector_mobility(
    traces: &[PolyakovChainTrace],
    policy: &CenterSectorMobilityPolicy,
) -> Result<CenterSectorMobilityAssessment, CenterSymmetryError> {
    if traces.is_empty() {
        return Err(CenterSymmetryError::EmptyTraceSet);
    }
    if !policy.minimum_polyakov_magnitude.is_finite() || policy.minimum_polyakov_magnitude < 0.0 {
        return Err(CenterSymmetryError::InvalidMinimumMagnitude(
            policy.minimum_polyakov_magnitude,
        ));
    }
    if !policy.minimum_classified_fraction.is_finite()
        || !(0.0..=1.0).contains(&policy.minimum_classified_fraction)
    {
        return Err(CenterSymmetryError::InvalidMinimumClassifiedFraction(
            policy.minimum_classified_fraction,
        ));
    }

    let mut seen = BTreeSet::new();
    let mut chains = Vec::with_capacity(traces.len());
    for (index, trace) in traces.iter().enumerate() {
        if trace.chain_id.trim().is_empty() {
            return Err(CenterSymmetryError::EmptyChainId { index });
        }
        if !seen.insert(trace.chain_id.clone()) {
            return Err(CenterSymmetryError::DuplicateChainId(trace.chain_id.clone()));
        }
        chains.push(diagnose_center_sector_chain(
            trace,
            policy.minimum_polyakov_magnitude,
        )?);
    }

    let every_chain_meets_classified_fraction = chains.iter().all(|chain| {
        chain.classified_count as f64 / chain.sample_count as f64
            >= policy.minimum_classified_fraction
    });
    let every_chain_meets_transition_count = chains.iter().all(|chain| {
        chain.sector_transition_count >= policy.minimum_sector_transitions_per_chain
    });

    Ok(CenterSectorMobilityAssessment {
        diagnostic_id: Z3_CENTER_DIAGNOSTIC_ID,
        minimum_polyakov_magnitude: policy.minimum_polyakov_magnitude,
        minimum_classified_fraction: policy.minimum_classified_fraction,
        minimum_sector_transitions_per_chain: policy.minimum_sector_transitions_per_chain,
        chains,
        every_chain_meets_classified_fraction,
        every_chain_meets_transition_count,
        meets_declared_policy: every_chain_meets_classified_fraction
            && every_chain_meets_transition_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn center(sector: Z3CenterSector, magnitude: f64) -> Complex {
        Complex::new(
            magnitude * sector.phase().cos(),
            magnitude * sector.phase().sin(),
        )
    }

    #[test]
    fn exact_center_elements_classify_and_align() {
        for sector in Z3CenterSector::ALL {
            let value = center(sector, 0.5);
            assert_eq!(classify_z3_center_sector(value, 0.01).unwrap(), Some(sector));
            let aligned = center_align_polyakov(value, 0.01).unwrap().unwrap();
            assert!((aligned.re - 0.5).abs() < 1.0e-15);
            assert!(aligned.im.abs() < 2.0e-15);
        }
    }

    #[test]
    fn zero_magnitude_is_ambiguous_even_with_zero_floor() {
        assert_eq!(
            classify_z3_center_sector(Complex::new(0.0, 0.0), 0.0).unwrap(),
            None
        );
    }

    #[test]
    fn global_center_rotation_preserves_invariant_observables() {
        let value = Complex::new(0.31, -0.27);
        let rotated = complex_mul(value, center(Z3CenterSector::PositivePhase, 1.0));
        let aligned_a = center_align_polyakov(value, 0.0).unwrap().unwrap();
        let aligned_b = center_align_polyakov(rotated, 0.0).unwrap().unwrap();
        assert!((complex_magnitude(value) - complex_magnitude(rotated)).abs() < 1.0e-15);
        assert!((aligned_a.re - aligned_b.re).abs() < 2.0e-15);
        assert!((aligned_a.im - aligned_b.im).abs() < 2.0e-15);
        assert_ne!(
            classify_z3_center_sector(value, 0.0).unwrap(),
            classify_z3_center_sector(rotated, 0.0).unwrap()
        );
    }

    #[test]
    fn ambiguity_breaks_transition_continuity() {
        let trace = PolyakovChainTrace {
            chain_id: "chain".into(),
            samples: vec![
                center(Z3CenterSector::ZeroPhase, 0.5),
                Complex::new(0.0, 0.0),
                center(Z3CenterSector::PositivePhase, 0.5),
            ],
        };
        let diagnostics = diagnose_center_sector_chain(&trace, 0.1).unwrap();
        assert_eq!(diagnostics.ambiguous_count, 1);
        assert_eq!(diagnostics.maximum_ambiguous_run, 1);
        assert_eq!(diagnostics.sector_transition_count, 0);
    }

    #[test]
    fn reproduces_lqcd_019b_frozen_sector_counts() {
        let cold = [
            Z3CenterSector::ZeroPhase,
            Z3CenterSector::ZeroPhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
        ];
        let disordered = [
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::NegativePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::PositivePhase,
            Z3CenterSector::ZeroPhase,
            Z3CenterSector::ZeroPhase,
            Z3CenterSector::ZeroPhase,
            Z3CenterSector::ZeroPhase,
            Z3CenterSector::ZeroPhase,
            Z3CenterSector::ZeroPhase,
            Z3CenterSector::ZeroPhase,
            Z3CenterSector::ZeroPhase,
        ];
        let traces = [
            PolyakovChainTrace {
                chain_id: "cold".into(),
                samples: cold.into_iter().map(|s| center(s, 0.5)).collect(),
            },
            PolyakovChainTrace {
                chain_id: "disordered".into(),
                samples: disordered.into_iter().map(|s| center(s, 0.5)).collect(),
            },
        ];
        let assessment = assess_center_sector_mobility(
            &traces,
            &CenterSectorMobilityPolicy {
                minimum_polyakov_magnitude: 0.1,
                minimum_classified_fraction: 1.0,
                minimum_sector_transitions_per_chain: 0,
            },
        )
        .unwrap();
        assert_eq!(assessment.chains[0].sector_counts, [2, 0, 22]);
        assert_eq!(assessment.chains[0].sector_transition_count, 1);
        assert_eq!(assessment.chains[0].maximum_classified_sector_dwell, 22);
        assert_eq!(assessment.chains[1].sector_counts, [8, 13, 3]);
        assert_eq!(assessment.chains[1].sector_transition_count, 2);
        assert_eq!(assessment.chains[1].maximum_classified_sector_dwell, 13);
    }

    #[test]
    fn transition_threshold_is_caller_declared() {
        let traces = [
            PolyakovChainTrace {
                chain_id: "cold".into(),
                samples: vec![
                    center(Z3CenterSector::ZeroPhase, 0.5),
                    center(Z3CenterSector::NegativePhase, 0.5),
                    center(Z3CenterSector::NegativePhase, 0.5),
                ],
            },
            PolyakovChainTrace {
                chain_id: "disordered".into(),
                samples: vec![
                    center(Z3CenterSector::NegativePhase, 0.5),
                    center(Z3CenterSector::PositivePhase, 0.5),
                    center(Z3CenterSector::ZeroPhase, 0.5),
                ],
            },
        ];
        let permissive = assess_center_sector_mobility(
            &traces,
            &CenterSectorMobilityPolicy {
                minimum_polyakov_magnitude: 0.1,
                minimum_classified_fraction: 1.0,
                minimum_sector_transitions_per_chain: 1,
            },
        )
        .unwrap();
        assert!(permissive.meets_declared_policy);

        let stricter = assess_center_sector_mobility(
            &traces,
            &CenterSectorMobilityPolicy {
                minimum_polyakov_magnitude: 0.1,
                minimum_classified_fraction: 1.0,
                minimum_sector_transitions_per_chain: 2,
            },
        )
        .unwrap();
        assert!(!stricter.meets_declared_policy);
    }

    #[test]
    fn empty_trace_set_fails_closed() {
        let err = assess_center_sector_mobility(
            &[],
            &CenterSectorMobilityPolicy {
                minimum_polyakov_magnitude: 0.1,
                minimum_classified_fraction: 1.0,
                minimum_sector_transitions_per_chain: 0,
            },
        )
        .unwrap_err();
        assert_eq!(err, CenterSymmetryError::EmptyTraceSet);
    }
}
