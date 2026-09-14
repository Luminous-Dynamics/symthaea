// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Theory-neutral bridge between geometric and causal measurement families.
//!
//! GEOM-002C intentionally does not define a combined score. Fisher-Rao
//! trajectory geometry and multiscale causal measurements remain separate
//! observables so later intervention experiments can test whether they covary
//! rather than assuming that relationship in advance.

use std::fmt;

use symthaea_causal_reasoning::multiscale_causal::MultiscaleCausalSweep;
use symthaea_causal_reasoning::multiscale_sensitivity::CausalSensitivityEnvelope;
use symthaea_frontier_physics::geometric_emergence::TrajectoryMetrics;

/// Bridge-level validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometricCausalBridgeError {
    EmptyCausalProfile,
    InvalidPeakScaleIndex {
        peak_scale_index: usize,
        scale_count: usize,
    },
    SensitivityScaleCountMismatch {
        causal_scales: usize,
        sensitivity_scales: usize,
    },
    SensitivityStateCountMismatch {
        scale_index: usize,
        causal_states: usize,
        sensitivity_states: usize,
    },
    GeometryDimensionMismatch {
        reference: usize,
        condition: usize,
    },
    CausalStateProfileMismatch,
    SensitivityPresenceMismatch,
}

impl fmt::Display for GeometricCausalBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCausalProfile => write!(f, "causal profile must contain at least one scale"),
            Self::InvalidPeakScaleIndex {
                peak_scale_index,
                scale_count,
            } => write!(
                f,
                "causal peak scale index {peak_scale_index} is outside {scale_count} scales"
            ),
            Self::SensitivityScaleCountMismatch {
                causal_scales,
                sensitivity_scales,
            } => write!(
                f,
                "causal profile has {causal_scales} scales but sensitivity envelope has {sensitivity_scales}"
            ),
            Self::SensitivityStateCountMismatch {
                scale_index,
                causal_states,
                sensitivity_states,
            } => write!(
                f,
                "scale {scale_index} state-count mismatch: causal={causal_states}, sensitivity={sensitivity_states}"
            ),
            Self::GeometryDimensionMismatch {
                reference,
                condition,
            } => write!(
                f,
                "geometry dimensions differ across conditions: reference={reference}, condition={condition}"
            ),
            Self::CausalStateProfileMismatch => {
                write!(f, "causal state-count profiles differ across conditions")
            }
            Self::SensitivityPresenceMismatch => write!(
                f,
                "both compared profiles must either include sensitivity summaries or omit them"
            ),
        }
    }
}

impl std::error::Error for GeometricCausalBridgeError {}

/// Compact summary of the multiscale causal measurement family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalProfileSummary {
    pub finest_effective_information_bits: f64,
    pub peak_effective_information_bits: f64,
    pub peak_scale_index: usize,
    pub peak_advantage_bits: f64,
    pub criterion_met: bool,
}

/// Compact summary of coarse-graining sensitivity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalSensitivitySummary {
    pub variant_count: usize,
    pub peak_scale_stability: f64,
    pub criterion_met_fraction: f64,
    pub max_advantage_range_width_bits: f64,
}

/// Joint report that preserves geometry and causality as separate fields.
///
/// There is deliberately no combined scalar, rank, consciousness label, or
/// gravity/consciousness similarity score.
#[derive(Debug, Clone, PartialEq)]
pub struct JointGeometricCausalProfile {
    pub geometry: TrajectoryMetrics,
    pub causal: CausalProfileSummary,
    pub causal_state_counts: Vec<usize>,
    pub sensitivity: Option<CausalSensitivitySummary>,
}

/// Geometry-only deltas between a reference and intervention condition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometryContrast {
    pub path_length_delta: f64,
    pub endpoint_displacement_delta: f64,
    pub geodesic_efficiency_delta: f64,
    pub excess_path_length_delta: f64,
    pub mean_step_length_delta: f64,
    pub step_length_variance_delta: f64,
    pub max_step_length_delta: f64,
}

/// Causal-only deltas between a reference and intervention condition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalContrast {
    pub finest_effective_information_delta_bits: f64,
    pub peak_effective_information_delta_bits: f64,
    pub peak_advantage_delta_bits: f64,
    pub peak_scale_changed: bool,
    pub criterion_changed: bool,
}

/// Sensitivity-only deltas between comparable conditions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensitivityContrast {
    pub peak_scale_stability_delta: f64,
    pub criterion_met_fraction_delta: f64,
    pub max_advantage_range_width_delta_bits: f64,
}

/// Side-by-side intervention contrast. No cross-family score is computed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointProfileContrast {
    pub geometry: GeometryContrast,
    pub causal: CausalContrast,
    pub sensitivity: Option<SensitivityContrast>,
}

/// Build a joint report from independently computed measurement families.
pub fn build_joint_profile(
    geometry: TrajectoryMetrics,
    causal: &MultiscaleCausalSweep,
    sensitivity: Option<&CausalSensitivityEnvelope>,
) -> Result<JointGeometricCausalProfile, GeometricCausalBridgeError> {
    if causal.scales.is_empty() {
        return Err(GeometricCausalBridgeError::EmptyCausalProfile);
    }

    let peak_scale_index = causal.decision.peak_scale_index;
    if peak_scale_index >= causal.scales.len() {
        return Err(GeometricCausalBridgeError::InvalidPeakScaleIndex {
            peak_scale_index,
            scale_count: causal.scales.len(),
        });
    }

    let causal_state_counts: Vec<usize> = causal
        .scales
        .iter()
        .map(|scale| scale.state_count)
        .collect();

    let sensitivity = if let Some(envelope) = sensitivity {
        if envelope.scales.len() != causal.scales.len() {
            return Err(GeometricCausalBridgeError::SensitivityScaleCountMismatch {
                causal_scales: causal.scales.len(),
                sensitivity_scales: envelope.scales.len(),
            });
        }

        for (scale_index, (causal_scale, sensitivity_scale)) in causal
            .scales
            .iter()
            .zip(envelope.scales.iter())
            .enumerate()
        {
            if causal_scale.state_count != sensitivity_scale.state_count {
                return Err(GeometricCausalBridgeError::SensitivityStateCountMismatch {
                    scale_index,
                    causal_states: causal_scale.state_count,
                    sensitivity_states: sensitivity_scale.state_count,
                });
            }
        }

        let max_advantage_range_width_bits = envelope
            .scales
            .iter()
            .map(|scale| scale.advantage_vs_finest_bits.width())
            .fold(0.0_f64, f64::max);

        Some(CausalSensitivitySummary {
            variant_count: envelope.variant_count,
            peak_scale_stability: envelope.peak_scale_stability,
            criterion_met_fraction: envelope.criterion_met_fraction,
            max_advantage_range_width_bits,
        })
    } else {
        None
    };

    Ok(JointGeometricCausalProfile {
        geometry,
        causal: CausalProfileSummary {
            finest_effective_information_bits: causal.scales[0].effective_information_bits,
            peak_effective_information_bits: causal.scales[peak_scale_index]
                .effective_information_bits,
            peak_scale_index,
            peak_advantage_bits: causal.decision.peak_advantage_bits,
            criterion_met: causal.decision.criterion_met,
        },
        causal_state_counts,
        sensitivity,
    })
}

/// Compare two already-built joint profiles without mixing measurement families.
pub fn contrast_joint_profiles(
    reference: &JointGeometricCausalProfile,
    condition: &JointGeometricCausalProfile,
) -> Result<JointProfileContrast, GeometricCausalBridgeError> {
    if reference.geometry.dimensions != condition.geometry.dimensions {
        return Err(GeometricCausalBridgeError::GeometryDimensionMismatch {
            reference: reference.geometry.dimensions,
            condition: condition.geometry.dimensions,
        });
    }
    if reference.causal_state_counts != condition.causal_state_counts {
        return Err(GeometricCausalBridgeError::CausalStateProfileMismatch);
    }

    let sensitivity = match (reference.sensitivity, condition.sensitivity) {
        (Some(reference_sensitivity), Some(condition_sensitivity)) => Some(SensitivityContrast {
            peak_scale_stability_delta: condition_sensitivity.peak_scale_stability
                - reference_sensitivity.peak_scale_stability,
            criterion_met_fraction_delta: condition_sensitivity.criterion_met_fraction
                - reference_sensitivity.criterion_met_fraction,
            max_advantage_range_width_delta_bits: condition_sensitivity
                .max_advantage_range_width_bits
                - reference_sensitivity.max_advantage_range_width_bits,
        }),
        (None, None) => None,
        _ => return Err(GeometricCausalBridgeError::SensitivityPresenceMismatch),
    };

    Ok(JointProfileContrast {
        geometry: GeometryContrast {
            path_length_delta: condition.geometry.path_length - reference.geometry.path_length,
            endpoint_displacement_delta: condition.geometry.endpoint_displacement
                - reference.geometry.endpoint_displacement,
            geodesic_efficiency_delta: condition.geometry.geodesic_efficiency
                - reference.geometry.geodesic_efficiency,
            excess_path_length_delta: condition.geometry.excess_path_length
                - reference.geometry.excess_path_length,
            mean_step_length_delta: condition.geometry.mean_step_length
                - reference.geometry.mean_step_length,
            step_length_variance_delta: condition.geometry.step_length_variance
                - reference.geometry.step_length_variance,
            max_step_length_delta: condition.geometry.max_step_length
                - reference.geometry.max_step_length,
        },
        causal: CausalContrast {
            finest_effective_information_delta_bits: condition
                .causal
                .finest_effective_information_bits
                - reference.causal.finest_effective_information_bits,
            peak_effective_information_delta_bits: condition.causal.peak_effective_information_bits
                - reference.causal.peak_effective_information_bits,
            peak_advantage_delta_bits: condition.causal.peak_advantage_bits
                - reference.causal.peak_advantage_bits,
            peak_scale_changed: condition.causal.peak_scale_index
                != reference.causal.peak_scale_index,
            criterion_changed: condition.causal.criterion_met != reference.causal.criterion_met,
        },
        sensitivity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_causal_reasoning::multiscale_causal::{
        CausalAdvantageRule, CoarseGrainingSpec, analyze_multiscale_causality,
    };
    use symthaea_causal_reasoning::multiscale_sensitivity::{
        SweepVariant, analyze_coarse_graining_sensitivity,
    };
    use symthaea_frontier_physics::geometric_emergence::GeometricEmergenceObservatory;

    fn asymmetric_fine_tpm() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
        ]
    }

    fn direct_geometry() -> TrajectoryMetrics {
        let mut observatory = GeometricEmergenceObservatory::new();
        for state in [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.1, 0.9],
        ] {
            observatory
                .push_distribution(&state)
                .expect("valid probability state");
        }
        observatory.metrics().expect("valid trajectory")
    }

    fn detour_geometry() -> TrajectoryMetrics {
        let mut observatory = GeometricEmergenceObservatory::new();
        for state in [[0.9, 0.1], [0.5, 0.5], [0.9, 0.1], [0.1, 0.9]] {
            observatory
                .push_distribution(&state)
                .expect("valid probability state");
        }
        observatory.metrics().expect("valid trajectory")
    }

    fn causal_profile(assignment: Vec<usize>) -> MultiscaleCausalSweep {
        let rule = CausalAdvantageRule::new(0.1).expect("valid rule");
        analyze_multiscale_causality(
            "fine",
            &asymmetric_fine_tpm(),
            &[CoarseGrainingSpec::new("macro", assignment)],
            rule,
        )
        .expect("valid causal sweep")
    }

    fn sensitivity_envelope() -> CausalSensitivityEnvelope {
        let variants = [
            SweepVariant::new(
                "aligned",
                vec![CoarseGrainingSpec::new("macro", vec![0, 0, 0, 1])],
            ),
            SweepVariant::new(
                "alternative",
                vec![CoarseGrainingSpec::new("macro", vec![0, 0, 1, 1])],
            ),
        ];
        analyze_coarse_graining_sensitivity(
            "fine",
            &asymmetric_fine_tpm(),
            &variants,
            CausalAdvantageRule::new(0.1).expect("valid rule"),
        )
        .expect("valid sensitivity envelope")
    }

    #[test]
    fn joint_profile_preserves_separate_measurement_families() {
        let causal = causal_profile(vec![0, 0, 0, 1]);
        let sensitivity = sensitivity_envelope();
        let profile = build_joint_profile(direct_geometry(), &causal, Some(&sensitivity))
            .expect("compatible measurements");

        assert_eq!(profile.geometry.dimensions, 2);
        assert_eq!(profile.causal_state_counts, vec![4, 2]);
        assert!(profile.causal.peak_advantage_bits > 0.1);
        let sensitivity = profile.sensitivity.expect("sensitivity supplied");
        assert_eq!(sensitivity.variant_count, 2);
        assert!(sensitivity.max_advantage_range_width_bits > 0.5);
    }

    #[test]
    fn intervention_contrast_reports_independent_deltas() {
        let sensitivity = sensitivity_envelope();
        let reference_causal = causal_profile(vec![0, 0, 0, 1]);
        let condition_causal = causal_profile(vec![0, 0, 1, 1]);
        let reference = build_joint_profile(
            direct_geometry(),
            &reference_causal,
            Some(&sensitivity),
        )
        .expect("reference profile");
        let condition = build_joint_profile(
            detour_geometry(),
            &condition_causal,
            Some(&sensitivity),
        )
        .expect("condition profile");

        let contrast = contrast_joint_profiles(&reference, &condition).expect("comparable profiles");
        assert!(contrast.geometry.path_length_delta > 0.0);
        assert!(contrast.geometry.geodesic_efficiency_delta < 0.0);
        assert!(contrast.causal.peak_advantage_delta_bits < 0.0);
        assert!(contrast.causal.criterion_changed);
        assert_eq!(
            contrast
                .sensitivity
                .expect("both profiles include sensitivity")
                .peak_scale_stability_delta,
            0.0
        );
    }

    #[test]
    fn mismatched_geometry_dimensions_fail_closed() {
        let causal = causal_profile(vec![0, 0, 0, 1]);
        let reference = build_joint_profile(direct_geometry(), &causal, None)
            .expect("reference profile");
        let mut incompatible = reference.clone();
        incompatible.geometry.dimensions = 3;

        assert!(matches!(
            contrast_joint_profiles(&reference, &incompatible),
            Err(GeometricCausalBridgeError::GeometryDimensionMismatch { .. })
        ));
    }

    #[test]
    fn sensitivity_presence_mismatch_fails_closed() {
        let causal = causal_profile(vec![0, 0, 0, 1]);
        let sensitivity = sensitivity_envelope();
        let reference = build_joint_profile(direct_geometry(), &causal, Some(&sensitivity))
            .expect("reference profile");
        let condition = build_joint_profile(detour_geometry(), &causal, None)
            .expect("condition profile");

        assert_eq!(
            contrast_joint_profiles(&reference, &condition),
            Err(GeometricCausalBridgeError::SensitivityPresenceMismatch)
        );
    }
}
