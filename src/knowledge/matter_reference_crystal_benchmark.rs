// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pre-execution protocol for the first live Matter reference-crystal capsule.
//!
//! This boundary exists to prevent an integration/demo benchmark from silently
//! acquiring the authority of a full thermodynamic materials study. It freezes
//! the target and numerical criteria before execution, while deliberately
//! withholding thermodynamic phase admission in v1. A restricted phase set may
//! be exercised diagnostically, but it can never satisfy the crystal phase gate.
//!
//! Likewise, producing DFPT dynamical matrices is not treated as a complete
//! phonon-dispersion workflow. Final phonon admission requires explicit
//! post-processing execution evidence rather than an implicit parser step.

use std::collections::BTreeSet;
use std::fmt;

use symthaea_evidence_plane::external_receipt::{
    EvidenceReferenceInterpretation, EvidenceRole, ExternalEvidenceBundle, ExternalEvidenceError,
    Sha256Digest,
};

pub use super::matter_crystal_identity::{AcousticSumRuleTreatment, RelaxationCellMode};

const MIN_CONVERGENCE_SAMPLES: u32 = 3;
const MAX_PHASE_LABELS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceBenchmarkAuthorityV1 {
    ExecutionReferenceOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermodynamicPhaseEligibilityV1 {
    WithheldByProtocol,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhononFinalizationPolicyV1 {
    ExplicitPostprocessingExecutionRequired,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferencePhaseCoverageV1 {
    Withheld,
    RestrictedDiagnostic { phase_labels: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceCrystalNumericalCriteriaV1 {
    pub energy_tolerance_ev_per_atom_bits: u64,
    pub force_tolerance_ev_per_angstrom_bits: u64,
    pub max_abs_stress_tolerance_gpa_bits: Option<u64>,
    pub max_relative_cell_change_tolerance_bits: Option<u64>,
    pub minimum_cutoff_samples: u32,
    pub minimum_kpoint_samples: u32,
    pub phonon_q_grid: [u32; 3],
    pub phonon_q_shift_fractional_bits: [u64; 3],
    pub q_coordinate_tolerance_bits: u64,
    pub imaginary_frequency_tolerance_bits: u64,
    pub acoustic_sum_rule: AcousticSumRuleTreatment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceCrystalBenchmarkProtocolV1 {
    pub benchmark_id: String,
    pub target_label: String,
    /// Strictly increasing atomic-number order gives one encoding per element set.
    pub target_atomic_numbers: Vec<u16>,
    pub target_structure_artifact_id: String,
    pub protocol_preregistration_evidence_id: String,
    pub solver_policy_snapshot_id: String,
    pub pseudopotential_policy_snapshot_id: String,
    pub energy_criterion_evidence_id: String,
    pub force_criterion_evidence_id: String,
    pub stress_criterion_evidence_id: Option<String>,
    pub cell_change_criterion_evidence_id: Option<String>,
    pub q_coordinate_criterion_evidence_id: String,
    pub imaginary_frequency_criterion_evidence_id: String,
    pub relaxation_mode: RelaxationCellMode,
    pub numerical: ReferenceCrystalNumericalCriteriaV1,
    pub phase_coverage: ReferencePhaseCoverageV1,
    pub phonon_finalization: PhononFinalizationPolicyV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundReferenceCrystalBenchmarkProtocolV1 {
    pub protocol: ReferenceCrystalBenchmarkProtocolV1,
    pub target_structure_sha256: Sha256Digest,
    pub protocol_preregistration_sha256: Sha256Digest,
    pub solver_policy_sha256: Sha256Digest,
    pub pseudopotential_policy_sha256: Sha256Digest,
    pub energy_criterion_sha256: Sha256Digest,
    pub force_criterion_sha256: Sha256Digest,
    pub stress_criterion_sha256: Option<Sha256Digest>,
    pub cell_change_criterion_sha256: Option<Sha256Digest>,
    pub q_coordinate_criterion_sha256: Sha256Digest,
    pub imaginary_frequency_criterion_sha256: Sha256Digest,
    pub authority: ReferenceBenchmarkAuthorityV1,
    pub thermodynamic_phase_eligibility: ThermodynamicPhaseEligibilityV1,
    pub phonon_finalization: PhononFinalizationPolicyV1,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub protocol_interpretation: ReferenceBenchmarkProtocolInterpretationV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceBenchmarkProtocolInterpretationV1 {
    LocallyValidatedDeclaredPreregistrationV1,
}

pub fn bind_reference_crystal_benchmark_protocol_v1(
    protocol: ReferenceCrystalBenchmarkProtocolV1,
    bundle: &ExternalEvidenceBundle,
) -> Result<BoundReferenceCrystalBenchmarkProtocolV1, ReferenceBenchmarkError> {
    validate_protocol_shape(&protocol)?;

    let target = bundle.require_role(
        &protocol.target_structure_artifact_id,
        EvidenceRole::ArtifactContent,
    )?;
    let preregistration = bundle.require_role(
        &protocol.protocol_preregistration_evidence_id,
        EvidenceRole::Preregistration,
    )?;
    let solver = bundle.require_role(
        &protocol.solver_policy_snapshot_id,
        EvidenceRole::ImplementationSnapshot,
    )?;
    let pseudo = bundle.require_role(
        &protocol.pseudopotential_policy_snapshot_id,
        EvidenceRole::DependencySnapshot,
    )?;
    let energy = require_preregistration(bundle, &protocol.energy_criterion_evidence_id)?;
    let force = require_preregistration(bundle, &protocol.force_criterion_evidence_id)?;
    let q_coordinate =
        require_preregistration(bundle, &protocol.q_coordinate_criterion_evidence_id)?;
    let imaginary = require_preregistration(
        bundle,
        &protocol.imaginary_frequency_criterion_evidence_id,
    )?;

    let (stress_sha, cell_sha) = match protocol.relaxation_mode {
        RelaxationCellMode::FixedCell => {
            if protocol.stress_criterion_evidence_id.is_some()
                || protocol.cell_change_criterion_evidence_id.is_some()
            {
                return Err(ReferenceBenchmarkError::VariableCellCriteriaForbiddenForFixedCell);
            }
            (None, None)
        }
        RelaxationCellMode::VariableCell => {
            let stress_id = protocol
                .stress_criterion_evidence_id
                .as_deref()
                .ok_or(ReferenceBenchmarkError::MissingVariableCellStressCriterion)?;
            let cell_id = protocol
                .cell_change_criterion_evidence_id
                .as_deref()
                .ok_or(ReferenceBenchmarkError::MissingVariableCellChangeCriterion)?;
            let stress = require_preregistration(bundle, stress_id)?;
            let cell = require_preregistration(bundle, cell_id)?;
            (
                Some(stress.content_sha256.clone()),
                Some(cell.content_sha256.clone()),
            )
        }
    };

    let semantic_roots = [
        target.content_sha256.as_str(),
        preregistration.content_sha256.as_str(),
        solver.content_sha256.as_str(),
        pseudo.content_sha256.as_str(),
    ];
    if semantic_roots.iter().collect::<BTreeSet<_>>().len() != semantic_roots.len() {
        return Err(ReferenceBenchmarkError::CoreSemanticContentAlias);
    }

    Ok(BoundReferenceCrystalBenchmarkProtocolV1 {
        protocol,
        target_structure_sha256: target.content_sha256.clone(),
        protocol_preregistration_sha256: preregistration.content_sha256.clone(),
        solver_policy_sha256: solver.content_sha256.clone(),
        pseudopotential_policy_sha256: pseudo.content_sha256.clone(),
        energy_criterion_sha256: energy.content_sha256.clone(),
        force_criterion_sha256: force.content_sha256.clone(),
        stress_criterion_sha256: stress_sha,
        cell_change_criterion_sha256: cell_sha,
        q_coordinate_criterion_sha256: q_coordinate.content_sha256.clone(),
        imaginary_frequency_criterion_sha256: imaginary.content_sha256.clone(),
        authority: ReferenceBenchmarkAuthorityV1::ExecutionReferenceOnly,
        thermodynamic_phase_eligibility:
            ThermodynamicPhaseEligibilityV1::WithheldByProtocol,
        phonon_finalization: PhononFinalizationPolicyV1::ExplicitPostprocessingExecutionRequired,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        protocol_interpretation:
            ReferenceBenchmarkProtocolInterpretationV1::LocallyValidatedDeclaredPreregistrationV1,
    })
}

fn require_preregistration<'a>(
    bundle: &'a ExternalEvidenceBundle,
    id: &str,
) -> Result<&'a symthaea_evidence_plane::external_receipt::ExternalEvidenceReference, ReferenceBenchmarkError>
{
    Ok(bundle.require_role(id, EvidenceRole::Preregistration)?)
}

fn validate_protocol_shape(
    protocol: &ReferenceCrystalBenchmarkProtocolV1,
) -> Result<(), ReferenceBenchmarkError> {
    for (field, value) in [
        ("benchmark_id", protocol.benchmark_id.as_str()),
        ("target_label", protocol.target_label.as_str()),
        (
            "target_structure_artifact_id",
            protocol.target_structure_artifact_id.as_str(),
        ),
        (
            "protocol_preregistration_evidence_id",
            protocol.protocol_preregistration_evidence_id.as_str(),
        ),
        (
            "solver_policy_snapshot_id",
            protocol.solver_policy_snapshot_id.as_str(),
        ),
        (
            "pseudopotential_policy_snapshot_id",
            protocol.pseudopotential_policy_snapshot_id.as_str(),
        ),
        (
            "energy_criterion_evidence_id",
            protocol.energy_criterion_evidence_id.as_str(),
        ),
        (
            "force_criterion_evidence_id",
            protocol.force_criterion_evidence_id.as_str(),
        ),
        (
            "q_coordinate_criterion_evidence_id",
            protocol.q_coordinate_criterion_evidence_id.as_str(),
        ),
        (
            "imaginary_frequency_criterion_evidence_id",
            protocol.imaginary_frequency_criterion_evidence_id.as_str(),
        ),
    ] {
        validate_text(field, value)?;
    }

    if protocol.target_atomic_numbers.is_empty() {
        return Err(ReferenceBenchmarkError::EmptyTargetElementSet);
    }
    let mut previous = 0_u16;
    for &z in &protocol.target_atomic_numbers {
        if !(1..=118).contains(&z) {
            return Err(ReferenceBenchmarkError::InvalidAtomicNumber(z));
        }
        if z <= previous {
            return Err(ReferenceBenchmarkError::TargetAtomicNumbersNotStrictlyIncreasing);
        }
        previous = z;
    }

    let numerical = &protocol.numerical;
    positive_finite(
        numerical.energy_tolerance_ev_per_atom_bits,
        "energy tolerance",
    )?;
    positive_finite(
        numerical.force_tolerance_ev_per_angstrom_bits,
        "force tolerance",
    )?;
    finite_nonnegative(
        numerical.q_coordinate_tolerance_bits,
        "q-coordinate tolerance",
    )?;
    finite_nonnegative(
        numerical.imaginary_frequency_tolerance_bits,
        "imaginary-frequency tolerance",
    )?;
    if numerical.minimum_cutoff_samples < MIN_CONVERGENCE_SAMPLES {
        return Err(ReferenceBenchmarkError::InsufficientCutoffSamples(
            numerical.minimum_cutoff_samples,
        ));
    }
    if numerical.minimum_kpoint_samples < MIN_CONVERGENCE_SAMPLES {
        return Err(ReferenceBenchmarkError::InsufficientKPointSamples(
            numerical.minimum_kpoint_samples,
        ));
    }
    if numerical.phonon_q_grid.iter().any(|&value| value == 0) {
        return Err(ReferenceBenchmarkError::ZeroPhononGridDimension);
    }
    for bits in numerical.phonon_q_shift_fractional_bits {
        let value = f64::from_bits(bits);
        if !value.is_finite() || !(0.0..1.0).contains(&value) {
            return Err(ReferenceBenchmarkError::InvalidPhononGridShift);
        }
    }

    match protocol.relaxation_mode {
        RelaxationCellMode::FixedCell => {
            if numerical.max_abs_stress_tolerance_gpa_bits.is_some()
                || numerical.max_relative_cell_change_tolerance_bits.is_some()
            {
                return Err(ReferenceBenchmarkError::VariableCellMetricsForbiddenForFixedCell);
            }
        }
        RelaxationCellMode::VariableCell => {
            let stress = numerical
                .max_abs_stress_tolerance_gpa_bits
                .ok_or(ReferenceBenchmarkError::MissingVariableCellStressTolerance)?;
            let cell = numerical
                .max_relative_cell_change_tolerance_bits
                .ok_or(ReferenceBenchmarkError::MissingVariableCellChangeTolerance)?;
            positive_finite(stress, "stress tolerance")?;
            positive_finite(cell, "cell-change tolerance")?;
        }
    }

    match &protocol.phase_coverage {
        ReferencePhaseCoverageV1::Withheld => {}
        ReferencePhaseCoverageV1::RestrictedDiagnostic { phase_labels } => {
            if phase_labels.is_empty() {
                return Err(ReferenceBenchmarkError::EmptyRestrictedPhaseSet);
            }
            if phase_labels.len() > MAX_PHASE_LABELS {
                return Err(ReferenceBenchmarkError::TooManyRestrictedPhases(
                    phase_labels.len(),
                ));
            }
            let mut seen = BTreeSet::new();
            for label in phase_labels {
                validate_text("phase_label", label)?;
                if !seen.insert(label.as_str()) {
                    return Err(ReferenceBenchmarkError::DuplicateRestrictedPhaseLabel(
                        label.clone(),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_text(field: &'static str, value: &str) -> Result<(), ReferenceBenchmarkError> {
    if value.trim().is_empty() {
        return Err(ReferenceBenchmarkError::EmptyField(field));
    }
    if value.chars().any(char::is_control) {
        return Err(ReferenceBenchmarkError::ControlCharacter(field));
    }
    Ok(())
}

fn positive_finite(bits: u64, field: &'static str) -> Result<f64, ReferenceBenchmarkError> {
    let value = f64::from_bits(bits);
    if !value.is_finite() || value <= 0.0 {
        return Err(ReferenceBenchmarkError::InvalidPositiveFinite(field));
    }
    Ok(value)
}

fn finite_nonnegative(bits: u64, field: &'static str) -> Result<f64, ReferenceBenchmarkError> {
    let value = f64::from_bits(bits);
    if !value.is_finite() || value < 0.0 {
        return Err(ReferenceBenchmarkError::InvalidNonnegativeFinite(field));
    }
    Ok(value)
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReferenceBenchmarkError {
    ExternalEvidence(ExternalEvidenceError),
    EmptyField(&'static str),
    ControlCharacter(&'static str),
    EmptyTargetElementSet,
    InvalidAtomicNumber(u16),
    TargetAtomicNumbersNotStrictlyIncreasing,
    InvalidPositiveFinite(&'static str),
    InvalidNonnegativeFinite(&'static str),
    InsufficientCutoffSamples(u32),
    InsufficientKPointSamples(u32),
    ZeroPhononGridDimension,
    InvalidPhononGridShift,
    MissingVariableCellStressTolerance,
    MissingVariableCellChangeTolerance,
    VariableCellMetricsForbiddenForFixedCell,
    MissingVariableCellStressCriterion,
    MissingVariableCellChangeCriterion,
    VariableCellCriteriaForbiddenForFixedCell,
    EmptyRestrictedPhaseSet,
    TooManyRestrictedPhases(usize),
    DuplicateRestrictedPhaseLabel(String),
    CoreSemanticContentAlias,
}

impl From<ExternalEvidenceError> for ReferenceBenchmarkError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for ReferenceBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "reference-crystal benchmark protocol rejected: {self:?}")
    }
}

impl std::error::Error for ReferenceBenchmarkError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_evidence_plane::external_receipt::ExternalEvidenceReference;

    fn numerical(mode: RelaxationCellMode) -> ReferenceCrystalNumericalCriteriaV1 {
        ReferenceCrystalNumericalCriteriaV1 {
            energy_tolerance_ev_per_atom_bits: 1e-3_f64.to_bits(),
            force_tolerance_ev_per_angstrom_bits: 1e-2_f64.to_bits(),
            max_abs_stress_tolerance_gpa_bits: matches!(mode, RelaxationCellMode::VariableCell)
                .then(|| 0.1_f64.to_bits()),
            max_relative_cell_change_tolerance_bits:
                matches!(mode, RelaxationCellMode::VariableCell).then(|| 1e-4_f64.to_bits()),
            minimum_cutoff_samples: 3,
            minimum_kpoint_samples: 3,
            phonon_q_grid: [4, 4, 4],
            phonon_q_shift_fractional_bits: [0.0_f64.to_bits(); 3],
            q_coordinate_tolerance_bits: 1e-10_f64.to_bits(),
            imaginary_frequency_tolerance_bits: 0.0_f64.to_bits(),
            acoustic_sum_rule: AcousticSumRuleTreatment::Translational,
        }
    }

    fn protocol(mode: RelaxationCellMode) -> ReferenceCrystalBenchmarkProtocolV1 {
        ReferenceCrystalBenchmarkProtocolV1 {
            benchmark_id: "matter-reference-si-v1".into(),
            target_label: "silicon-diamond-reference".into(),
            target_atomic_numbers: vec![14],
            target_structure_artifact_id: "structure:si".into(),
            protocol_preregistration_evidence_id: "prereg:protocol".into(),
            solver_policy_snapshot_id: "impl:solver-policy".into(),
            pseudopotential_policy_snapshot_id: "dep:pseudo-policy".into(),
            energy_criterion_evidence_id: "prereg:numerics".into(),
            force_criterion_evidence_id: "prereg:numerics".into(),
            stress_criterion_evidence_id: matches!(mode, RelaxationCellMode::VariableCell)
                .then(|| "prereg:numerics".into()),
            cell_change_criterion_evidence_id: matches!(mode, RelaxationCellMode::VariableCell)
                .then(|| "prereg:numerics".into()),
            q_coordinate_criterion_evidence_id: "prereg:numerics".into(),
            imaginary_frequency_criterion_evidence_id: "prereg:numerics".into(),
            relaxation_mode: mode,
            numerical: numerical(mode),
            phase_coverage: ReferencePhaseCoverageV1::Withheld,
            phonon_finalization:
                PhononFinalizationPolicyV1::ExplicitPostprocessingExecutionRequired,
        }
    }

    fn reference(id: &str, role: EvidenceRole, digest_byte: u8) -> ExternalEvidenceReference {
        ExternalEvidenceReference::try_new(
            id,
            role,
            format!("{digest_byte:02x}").repeat(32),
            format!("test://{id}"),
            format!("fixture {id}"),
            Some(20260908),
            Some("matter-reference-test".into()),
        )
        .unwrap()
    }

    fn bundle() -> ExternalEvidenceBundle {
        ExternalEvidenceBundle::new(vec![
            reference("structure:si", EvidenceRole::ArtifactContent, 1),
            reference("prereg:protocol", EvidenceRole::Preregistration, 2),
            reference("impl:solver-policy", EvidenceRole::ImplementationSnapshot, 3),
            reference("dep:pseudo-policy", EvidenceRole::DependencySnapshot, 4),
            reference("prereg:numerics", EvidenceRole::Preregistration, 5),
        ])
        .unwrap()
    }

    #[test]
    fn silicon_execution_reference_shape_is_valid() {
        validate_protocol_shape(&protocol(RelaxationCellMode::FixedCell)).unwrap();
    }

    #[test]
    fn restricted_phase_set_cannot_upgrade_benchmark_authority() {
        let mut value = protocol(RelaxationCellMode::FixedCell);
        value.phase_coverage = ReferencePhaseCoverageV1::RestrictedDiagnostic {
            phase_labels: vec!["diamond".into(), "beta-tin".into()],
        };
        let bound = bind_reference_crystal_benchmark_protocol_v1(value, &bundle()).unwrap();
        assert_eq!(bound.authority, ReferenceBenchmarkAuthorityV1::ExecutionReferenceOnly);
        assert_eq!(
            bound.thermodynamic_phase_eligibility,
            ThermodynamicPhaseEligibilityV1::WithheldByProtocol
        );
        assert_eq!(
            bound.phonon_finalization,
            PhononFinalizationPolicyV1::ExplicitPostprocessingExecutionRequired
        );
    }

    #[test]
    fn target_element_order_is_canonical() {
        let mut value = protocol(RelaxationCellMode::FixedCell);
        value.target_atomic_numbers = vec![14, 8];
        assert_eq!(
            validate_protocol_shape(&value).unwrap_err(),
            ReferenceBenchmarkError::TargetAtomicNumbersNotStrictlyIncreasing
        );
    }

    #[test]
    fn variable_cell_requires_stress_and_cell_metrics() {
        let mut value = protocol(RelaxationCellMode::VariableCell);
        value.numerical.max_abs_stress_tolerance_gpa_bits = None;
        assert_eq!(
            validate_protocol_shape(&value).unwrap_err(),
            ReferenceBenchmarkError::MissingVariableCellStressTolerance
        );
    }

    #[test]
    fn fewer_than_three_convergence_samples_are_rejected() {
        let mut value = protocol(RelaxationCellMode::FixedCell);
        value.numerical.minimum_cutoff_samples = 2;
        assert_eq!(
            validate_protocol_shape(&value).unwrap_err(),
            ReferenceBenchmarkError::InsufficientCutoffSamples(2)
        );
    }

    #[test]
    fn zero_q_grid_dimension_is_rejected() {
        let mut value = protocol(RelaxationCellMode::FixedCell);
        value.numerical.phonon_q_grid = [4, 0, 4];
        assert_eq!(
            validate_protocol_shape(&value).unwrap_err(),
            ReferenceBenchmarkError::ZeroPhononGridDimension
        );
    }
}
