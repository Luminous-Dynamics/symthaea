// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Theory-neutral intact/lesion/sham/rescue experiment contract.
//!
//! GEOM-003A standardizes intervention comparisons before architecture-specific
//! lesions are introduced. It reports geometric, causal, and sensitivity
//! observables separately and never emits a global consciousness score.

use std::collections::HashSet;
use std::fmt;

use symthaea_geometric_causal_bridge::{
    GeometricCausalBridgeError, JointGeometricCausalProfile, JointProfileContrast,
    contrast_joint_profiles,
};

const EFFECT_EPSILON: f64 = 1.0e-12;

/// Execution metadata that must be matched across all four conditions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionMetadata {
    pub protocol_id: String,
    pub seed: u64,
    pub observation_steps: u64,
    pub allocated_work_units: u64,
}

impl ConditionMetadata {
    pub fn new(
        protocol_id: impl Into<String>,
        seed: u64,
        observation_steps: u64,
        allocated_work_units: u64,
    ) -> Self {
        Self {
            protocol_id: protocol_id.into(),
            seed,
            observation_steps,
            allocated_work_units,
        }
    }
}

/// One named condition and its already-qualified measurement profile.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionRecord {
    pub label: String,
    pub metadata: ConditionMetadata,
    pub profile: JointGeometricCausalProfile,
}

impl ConditionRecord {
    pub fn new(
        label: impl Into<String>,
        metadata: ConditionMetadata,
        profile: JointGeometricCausalProfile,
    ) -> Self {
        Self {
            label: label.into(),
            metadata,
            profile,
        }
    }
}

/// Contract failures that invalidate an intervention quartet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterventionContractError {
    EmptyConditionLabel,
    DuplicateConditionLabel { label: String },
    EmptyProtocolId,
    MetadataMismatch { role: &'static str, field: &'static str },
    Bridge {
        comparison: &'static str,
        source: GeometricCausalBridgeError,
    },
}

impl fmt::Display for InterventionContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyConditionLabel => write!(f, "condition labels must not be empty"),
            Self::DuplicateConditionLabel { label } => {
                write!(f, "condition label must be unique: {label}")
            }
            Self::EmptyProtocolId => write!(f, "measurement protocol id must not be empty"),
            Self::MetadataMismatch { role, field } => {
                write!(f, "{role} metadata differs from intact condition for field {field}")
            }
            Self::Bridge { comparison, source } => {
                write!(f, "{comparison} profiles are not comparable: {source}")
            }
        }
    }
}

impl std::error::Error for InterventionContractError {}

/// Per-observable recovery for Fisher-Rao trajectory measurements.
///
/// Each field is `None` when the lesion produced effectively zero change in
/// that observable, because there is then no lesion effect to rescue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometryRecovery {
    pub path_length: Option<f64>,
    pub endpoint_displacement: Option<f64>,
    pub geodesic_efficiency: Option<f64>,
    pub excess_path_length: Option<f64>,
    pub mean_step_length: Option<f64>,
    pub step_length_variance: Option<f64>,
    pub max_step_length: Option<f64>,
}

/// Per-observable recovery for causal measurements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalRecovery {
    pub finest_effective_information_bits: Option<f64>,
    pub peak_effective_information_bits: Option<f64>,
    pub peak_advantage_bits: Option<f64>,
}

/// Per-observable recovery for coarse-graining sensitivity measurements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensitivityRecovery {
    pub peak_scale_stability: Option<f64>,
    pub criterion_met_fraction: Option<f64>,
    pub max_advantage_range_width_bits: Option<f64>,
}

/// Complete theory-neutral intervention report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterventionQuartetReport {
    pub lesion_vs_intact: JointProfileContrast,
    pub sham_vs_intact: JointProfileContrast,
    pub lesion_vs_sham: JointProfileContrast,
    pub rescue_vs_lesion: JointProfileContrast,
    pub rescue_vs_intact: JointProfileContrast,
    pub geometry_recovery: GeometryRecovery,
    pub causal_recovery: CausalRecovery,
    pub sensitivity_recovery: Option<SensitivityRecovery>,
}

fn validate_labels(records: [&ConditionRecord; 4]) -> Result<(), InterventionContractError> {
    let mut labels = HashSet::with_capacity(records.len());
    for record in records {
        let label = record.label.trim();
        if label.is_empty() {
            return Err(InterventionContractError::EmptyConditionLabel);
        }
        if !labels.insert(label.to_owned()) {
            return Err(InterventionContractError::DuplicateConditionLabel {
                label: label.to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_metadata(
    intact: &ConditionRecord,
    candidate: &ConditionRecord,
    role: &'static str,
) -> Result<(), InterventionContractError> {
    if intact.metadata.protocol_id.trim().is_empty() || candidate.metadata.protocol_id.trim().is_empty()
    {
        return Err(InterventionContractError::EmptyProtocolId);
    }
    if intact.metadata.protocol_id != candidate.metadata.protocol_id {
        return Err(InterventionContractError::MetadataMismatch {
            role,
            field: "protocol_id",
        });
    }
    if intact.metadata.seed != candidate.metadata.seed {
        return Err(InterventionContractError::MetadataMismatch {
            role,
            field: "seed",
        });
    }
    if intact.metadata.observation_steps != candidate.metadata.observation_steps {
        return Err(InterventionContractError::MetadataMismatch {
            role,
            field: "observation_steps",
        });
    }
    if intact.metadata.allocated_work_units != candidate.metadata.allocated_work_units {
        return Err(InterventionContractError::MetadataMismatch {
            role,
            field: "allocated_work_units",
        });
    }
    Ok(())
}

fn contrast(
    comparison: &'static str,
    reference: &ConditionRecord,
    condition: &ConditionRecord,
) -> Result<JointProfileContrast, InterventionContractError> {
    contrast_joint_profiles(&reference.profile, &condition.profile).map_err(|source| {
        InterventionContractError::Bridge {
            comparison,
            source,
        }
    })
}

/// Fractional rescue of one scalar observable.
///
/// - `1.0`: rescue returned exactly to intact value.
/// - `0.0`: rescue is no closer to intact than the lesion value.
/// - `< 0.0`: rescue moved farther from intact than the lesion.
/// - `None`: lesion had no measurable effect to rescue.
///
/// Values are deliberately not clamped, so failed or overshooting rescues remain
/// visible instead of being converted into a success-looking bounded score.
pub fn recovery_fraction(intact: f64, lesion: f64, rescue: f64) -> Option<f64> {
    let lesion_effect = (lesion - intact).abs();
    if lesion_effect <= EFFECT_EPSILON {
        None
    } else {
        Some(1.0 - (rescue - intact).abs() / lesion_effect)
    }
}

fn geometry_recovery(
    intact: &JointGeometricCausalProfile,
    lesion: &JointGeometricCausalProfile,
    rescue: &JointGeometricCausalProfile,
) -> GeometryRecovery {
    GeometryRecovery {
        path_length: recovery_fraction(
            intact.geometry.path_length,
            lesion.geometry.path_length,
            rescue.geometry.path_length,
        ),
        endpoint_displacement: recovery_fraction(
            intact.geometry.endpoint_displacement,
            lesion.geometry.endpoint_displacement,
            rescue.geometry.endpoint_displacement,
        ),
        geodesic_efficiency: recovery_fraction(
            intact.geometry.geodesic_efficiency,
            lesion.geometry.geodesic_efficiency,
            rescue.geometry.geodesic_efficiency,
        ),
        excess_path_length: recovery_fraction(
            intact.geometry.excess_path_length,
            lesion.geometry.excess_path_length,
            rescue.geometry.excess_path_length,
        ),
        mean_step_length: recovery_fraction(
            intact.geometry.mean_step_length,
            lesion.geometry.mean_step_length,
            rescue.geometry.mean_step_length,
        ),
        step_length_variance: recovery_fraction(
            intact.geometry.step_length_variance,
            lesion.geometry.step_length_variance,
            rescue.geometry.step_length_variance,
        ),
        max_step_length: recovery_fraction(
            intact.geometry.max_step_length,
            lesion.geometry.max_step_length,
            rescue.geometry.max_step_length,
        ),
    }
}

fn causal_recovery(
    intact: &JointGeometricCausalProfile,
    lesion: &JointGeometricCausalProfile,
    rescue: &JointGeometricCausalProfile,
) -> CausalRecovery {
    CausalRecovery {
        finest_effective_information_bits: recovery_fraction(
            intact.causal.finest_effective_information_bits,
            lesion.causal.finest_effective_information_bits,
            rescue.causal.finest_effective_information_bits,
        ),
        peak_effective_information_bits: recovery_fraction(
            intact.causal.peak_effective_information_bits,
            lesion.causal.peak_effective_information_bits,
            rescue.causal.peak_effective_information_bits,
        ),
        peak_advantage_bits: recovery_fraction(
            intact.causal.peak_advantage_bits,
            lesion.causal.peak_advantage_bits,
            rescue.causal.peak_advantage_bits,
        ),
    }
}

fn sensitivity_recovery(
    intact: &JointGeometricCausalProfile,
    lesion: &JointGeometricCausalProfile,
    rescue: &JointGeometricCausalProfile,
) -> Option<SensitivityRecovery> {
    match (intact.sensitivity, lesion.sensitivity, rescue.sensitivity) {
        (Some(intact), Some(lesion), Some(rescue)) => Some(SensitivityRecovery {
            peak_scale_stability: recovery_fraction(
                intact.peak_scale_stability,
                lesion.peak_scale_stability,
                rescue.peak_scale_stability,
            ),
            criterion_met_fraction: recovery_fraction(
                intact.criterion_met_fraction,
                lesion.criterion_met_fraction,
                rescue.criterion_met_fraction,
            ),
            max_advantage_range_width_bits: recovery_fraction(
                intact.max_advantage_range_width_bits,
                lesion.max_advantage_range_width_bits,
                rescue.max_advantage_range_width_bits,
            ),
        }),
        _ => None,
    }
}

/// Analyze one matched intact/lesion/sham/rescue quartet.
///
/// The role of each argument is fixed by the function signature, preventing a
/// caller from relabeling the best-looking condition after measurement.
pub fn analyze_intervention_quartet(
    intact: &ConditionRecord,
    lesion: &ConditionRecord,
    sham: &ConditionRecord,
    rescue: &ConditionRecord,
) -> Result<InterventionQuartetReport, InterventionContractError> {
    validate_labels([intact, lesion, sham, rescue])?;
    if intact.metadata.protocol_id.trim().is_empty() {
        return Err(InterventionContractError::EmptyProtocolId);
    }
    validate_metadata(intact, lesion, "lesion")?;
    validate_metadata(intact, sham, "sham")?;
    validate_metadata(intact, rescue, "rescue")?;

    let lesion_vs_intact = contrast("lesion-vs-intact", intact, lesion)?;
    let sham_vs_intact = contrast("sham-vs-intact", intact, sham)?;
    let lesion_vs_sham = contrast("lesion-vs-sham", sham, lesion)?;
    let rescue_vs_lesion = contrast("rescue-vs-lesion", lesion, rescue)?;
    let rescue_vs_intact = contrast("rescue-vs-intact", intact, rescue)?;

    Ok(InterventionQuartetReport {
        lesion_vs_intact,
        sham_vs_intact,
        lesion_vs_sham,
        rescue_vs_lesion,
        rescue_vs_intact,
        geometry_recovery: geometry_recovery(&intact.profile, &lesion.profile, &rescue.profile),
        causal_recovery: causal_recovery(&intact.profile, &lesion.profile, &rescue.profile),
        sensitivity_recovery: sensitivity_recovery(
            &intact.profile,
            &lesion.profile,
            &rescue.profile,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_frontier_physics::geometric_emergence::TrajectoryMetrics;
    use symthaea_geometric_causal_bridge::{
        CausalProfileSummary, CausalSensitivitySummary,
    };

    fn metadata() -> ConditionMetadata {
        ConditionMetadata::new("geom-003a-v1", 42, 1_000, 10_000)
    }

    fn profile(
        path_length: f64,
        efficiency: f64,
        peak_advantage: f64,
        sensitivity_width: f64,
    ) -> JointGeometricCausalProfile {
        JointGeometricCausalProfile {
            geometry: TrajectoryMetrics {
                samples: 5,
                dimensions: 2,
                path_length,
                endpoint_displacement: 1.0,
                geodesic_efficiency: efficiency,
                excess_path_length: path_length - 1.0,
                mean_step_length: path_length / 4.0,
                step_length_variance: (path_length - 1.0).abs(),
                max_step_length: path_length / 2.0,
            },
            causal: CausalProfileSummary {
                finest_effective_information_bits: 0.8,
                peak_effective_information_bits: 0.8 + peak_advantage.max(0.0),
                peak_scale_index: if peak_advantage > 0.0 { 1 } else { 0 },
                peak_advantage_bits: peak_advantage,
                criterion_met: peak_advantage >= 0.1,
            },
            causal_state_counts: vec![4, 2],
            sensitivity: Some(CausalSensitivitySummary {
                variant_count: 4,
                peak_scale_stability: 0.75,
                criterion_met_fraction: 0.5,
                max_advantage_range_width_bits: sensitivity_width,
            }),
        }
    }

    fn condition(label: &str, profile: JointGeometricCausalProfile) -> ConditionRecord {
        ConditionRecord::new(label, metadata(), profile)
    }

    #[test]
    fn matched_quartet_reports_separate_effects_and_recovery() {
        let intact = condition("intact", profile(1.0, 0.9, 0.2, 0.2));
        let lesion = condition("lesion", profile(3.0, 0.4, 0.0, 0.8));
        let sham = condition("sham", profile(1.2, 0.85, 0.18, 0.25));
        let rescue = condition("rescue", profile(1.5, 0.8, 0.15, 0.3));

        let report = analyze_intervention_quartet(&intact, &lesion, &sham, &rescue)
            .expect("matched quartet");

        assert!(report.lesion_vs_intact.geometry.path_length_delta > 0.0);
        assert!(report.sham_vs_intact.geometry.path_length_delta < 0.5);
        assert!(report.lesion_vs_sham.geometry.path_length_delta > 1.0);
        assert!(report.rescue_vs_lesion.geometry.path_length_delta < 0.0);
        assert!(report.rescue_vs_intact.geometry.path_length_delta > 0.0);

        let path_recovery = report.geometry_recovery.path_length.expect("lesion changed path");
        assert!((path_recovery - 0.75).abs() < 1.0e-12);
        let efficiency_recovery = report
            .geometry_recovery
            .geodesic_efficiency
            .expect("lesion changed efficiency");
        assert!((efficiency_recovery - 0.8).abs() < 1.0e-12);
        let causal_recovery = report
            .causal_recovery
            .peak_advantage_bits
            .expect("lesion changed causal advantage");
        assert!((causal_recovery - 0.75).abs() < 1.0e-12);
    }

    #[test]
    fn zero_lesion_effect_has_no_recovery_fraction() {
        assert_eq!(recovery_fraction(1.0, 1.0, 1.0), None);
    }

    #[test]
    fn worsening_rescue_is_negative_not_clamped() {
        let recovery = recovery_fraction(1.0, 2.0, 3.0).expect("lesion effect exists");
        assert!((recovery + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn mismatched_metadata_fails_closed() {
        let intact = condition("intact", profile(1.0, 0.9, 0.2, 0.2));
        let mut lesion = condition("lesion", profile(3.0, 0.4, 0.0, 0.8));
        lesion.metadata.seed = 43;
        let sham = condition("sham", profile(1.2, 0.85, 0.18, 0.25));
        let rescue = condition("rescue", profile(1.5, 0.8, 0.15, 0.3));

        assert_eq!(
            analyze_intervention_quartet(&intact, &lesion, &sham, &rescue),
            Err(InterventionContractError::MetadataMismatch {
                role: "lesion",
                field: "seed",
            })
        );
    }

    #[test]
    fn duplicate_condition_labels_fail_closed() {
        let intact = condition("same", profile(1.0, 0.9, 0.2, 0.2));
        let lesion = condition("same", profile(3.0, 0.4, 0.0, 0.8));
        let sham = condition("sham", profile(1.2, 0.85, 0.18, 0.25));
        let rescue = condition("rescue", profile(1.5, 0.8, 0.15, 0.3));

        assert!(matches!(
            analyze_intervention_quartet(&intact, &lesion, &sham, &rescue),
            Err(InterventionContractError::DuplicateConditionLabel { .. })
        ));
    }
}
