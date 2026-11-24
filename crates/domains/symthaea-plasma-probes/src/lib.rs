// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic, fail-closed primitives for adversarial kinetic probe design.
//!
//! This crate validates the v0.1 pulse-family wire contract, enumerates a
//! bounded deterministic grid, and computes a domain-separated canonical
//! digest for each intervention. It does not execute a Vlasov solver or
//! establish physical admissibility / numerical convergence.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const CONTRACT_VERSION: &str = "0.1";
const EQUATION_ID: &str = "periodic_1d1v_collisionless_vlasov_poisson";
const OPERATOR_ID: &str = "sinusoidal_velocity_kick_v1";
const OPERATOR_DEFINITION: &str = "v_after(x) = v_before(x) + A*sin(2*pi*m*(x-x_min)/L + phase)";
const AMPLITUDE_SEMANTICS: &str = "maximum_absolute_delta_v";
const APPLICATION_SEMANTICS: &str = "instantaneous_canonical_velocity_kick";
const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;
const MAX_PERIODIC_LIFT_HARMONICS: usize = 4_096;
const DIGEST_DOMAIN: &[u8] = b"symthaea.akpd.pulse-candidate.v0.1\0";

/// Errors returned while parsing, validating or enumerating probe contracts.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ContractError {
    #[error("invalid JSON contract: {0}")]
    JsonParse(String),
    #[error("unsupported contract version: {0}")]
    UnsupportedVersion(String),
    #[error("invalid contract identifier")]
    InvalidContractId,
    #[error("unsupported equation cohort: {0}")]
    UnsupportedEquation(String),
    #[error("unsupported intervention operator")]
    UnsupportedOperator,
    #[error("invalid field {field}: {reason}")]
    InvalidField {
        field: &'static str,
        reason: &'static str,
    },
    #[error("sealed-test contracts must prohibit acquisition")]
    SealedTestAcquisition,
    #[error("synthetic fixtures cannot authorize physical execution")]
    SyntheticExecutionAuthorized,
    #[error("physical candidate provenance is incomplete")]
    PhysicalProvenanceIncomplete,
    #[error("candidate is outside the admissible range for {0}")]
    CandidateOutOfRange(&'static str),
    #[error("grid step counts must all be positive")]
    InvalidGrid,
    #[error("grid contains {requested} raw points, exceeding proposal budget {budget}")]
    ProposalBudgetExceeded { requested: usize, budget: usize },
    #[error("grid size overflow")]
    GridSizeOverflow,
    #[error("Gramian state dimension must be positive")]
    InvalidGramianDimension,
    #[error("Gramian vector dimension mismatch: expected {expected}, got {actual}")]
    GramianDimensionMismatch { expected: usize, actual: usize },
    #[error("Gramian weight must be finite and non-negative")]
    InvalidGramianWeight,
    #[error("Gramian direction or arithmetic contains non-finite values")]
    InvalidGramianArithmetic,
    #[error("projection basis must be orthonormal within the declared tolerance")]
    InvalidProjectionBasis,
    #[error("response matrix requires non-empty input and output mode registries")]
    EmptyResponseModeRegistry,
    #[error("response mode registry contains duplicate mode IDs")]
    DuplicateResponseMode,
    #[error("unknown response mode ID: {0}")]
    UnknownResponseMode(u32),
    #[error("response matrix dimensions overflow")]
    ResponseMatrixSizeOverflow,
    #[error("response matrix entry must be finite")]
    NonFiniteResponseValue,
    #[error("response matrix axes do not match")]
    ResponseMatrixAxisMismatch,
    #[error("response matrix contains an unmeasured entry")]
    MissingResponseEntry,
    #[error("response reference matrix has zero norm")]
    ZeroResponseReferenceNorm,
    #[error("finite-amplitude response estimator requires positive finite pulse amplitudes")]
    InvalidResponseAmplitude,
    #[error("storage byte-count arithmetic overflowed")]
    StorageByteCountOverflow,
    #[error("asset amortization requires at least one rollout")]
    ZeroAmortizationRollouts,
    #[error("min-max normalization sample is empty")]
    EmptyNormalizationSample,
    #[error("min-max normalization support is degenerate or non-finite")]
    InvalidNormalizationSupport,
    #[error("min-max normalization output interval is invalid")]
    InvalidNormalizationOutputInterval,
    #[error("normalization coordinate is non-finite")]
    NonFiniteNormalizationCoordinate,
    #[error("periodic Fourier lift harmonic budget must be positive")]
    InvalidHarmonicBudget,
    #[error("open-boundary coordinate must lie in [0, 1]")]
    InvalidOpenBoundaryCoordinate,
    #[error("canonical symplectic basis contains non-finite values")]
    NonFiniteSymplecticBasis,
    #[error("canonical symplectic pair defect exceeds the declared tolerance")]
    InvalidSymplecticBasis,
    #[error("invalid sensor/actuator timing field {field}: {reason}")]
    InvalidControlTimingField {
        field: &'static str,
        reason: &'static str,
    },
    #[error("invalid KIRD record field {field}: {reason}")]
    InvalidKirdField {
        field: &'static str,
        reason: &'static str,
    },
}

/// A parsed and semantically checked v0.1 pulse-family contract.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PulseFamilyContract {
    schema_version: String,
    contract_id: String,
    contract_status: String,
    physics_cohort: PhysicsCohort,
    intervention_operator: InterventionOperator,
    admissible_controls: AdmissibleControls,
    search_budget: SearchBudget,
    evaluation_policy: EvaluationPolicy,
    split_role: String,
    acquisition_allowed: bool,
    provenance: Provenance,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PhysicsCohort {
    equation: String,
    normalization_id: String,
    spatial_domain: SpatialDomain,
    velocity_domain: VelocityDomain,
    time_unit: String,
    maximum_time: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpatialDomain {
    minimum: f64,
    maximum: f64,
    periodic: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VelocityDomain {
    minimum: f64,
    maximum: f64,
    velocity_unit: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InterventionOperator {
    operator_id: String,
    definition: String,
    amplitude_semantics: String,
    application_semantics: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdmissibleControls {
    spatial_mode: IntegerRange,
    phase_rad: NumericRange,
    kick_amplitude: UnitRange,
    application_time: UnitRange,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegerRange {
    minimum: u32,
    maximum: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericRange {
    minimum: f64,
    maximum: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UnitRange {
    minimum: f64,
    maximum: f64,
    unit: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchBudget {
    maximum_proposals: usize,
    maximum_high_fidelity_confirmations: usize,
    maximum_reference_steps: usize,
    maximum_wall_time_seconds: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluationPolicy {
    independent_confirmation_required: bool,
    minimum_spatial_refinement_levels: usize,
    minimum_time_refinement_levels: usize,
    minimum_post_pulse_observation_duration: f64,
    required_controls: Vec<String>,
    response_metrics: Vec<String>,
    #[serde(default)]
    confirmation_reference_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    fixture_class: String,
    initial_state_sha256: String,
    reference_solver_revision: String,
    candidate_representation_revision: String,
    generator_seed: u64,
    physical_execution_authorized: bool,
    #[serde(default)]
    source_record_refs: Vec<String>,
}

impl PulseFamilyContract {
    /// Parse JSON and apply semantic checks that JSON Schema cannot express.
    pub fn parse_json(input: &str) -> Result<Self, ContractError> {
        let contract: Self = serde_json::from_str(input)
            .map_err(|error| ContractError::JsonParse(error.to_string()))?;
        contract.validate()?;
        Ok(contract)
    }

    /// Validate cross-field constraints and fail-closed protocol requirements.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != CONTRACT_VERSION {
            return Err(ContractError::UnsupportedVersion(
                self.schema_version.clone(),
            ));
        }
        if !valid_contract_id(&self.contract_id) {
            return Err(ContractError::InvalidContractId);
        }
        if self.physics_cohort.equation != EQUATION_ID {
            return Err(ContractError::UnsupportedEquation(
                self.physics_cohort.equation.clone(),
            ));
        }
        if self.physics_cohort.normalization_id.trim().is_empty()
            || self.physics_cohort.time_unit.trim().is_empty()
            || self
                .physics_cohort
                .velocity_domain
                .velocity_unit
                .trim()
                .is_empty()
        {
            return Err(ContractError::InvalidField {
                field: "physics_cohort",
                reason: "normalization and units must be non-empty",
            });
        }
        if !finite_increasing(
            self.physics_cohort.spatial_domain.minimum,
            self.physics_cohort.spatial_domain.maximum,
        ) || !self.physics_cohort.spatial_domain.periodic
        {
            return Err(ContractError::InvalidField {
                field: "physics_cohort.spatial_domain",
                reason: "domain bounds must be finite and increasing, with periodic=true",
            });
        }
        if !finite_increasing(
            self.physics_cohort.velocity_domain.minimum,
            self.physics_cohort.velocity_domain.maximum,
        ) {
            return Err(ContractError::InvalidField {
                field: "physics_cohort.velocity_domain",
                reason: "velocity bounds must be finite and increasing",
            });
        }
        if !self.physics_cohort.maximum_time.is_finite() || self.physics_cohort.maximum_time <= 0.0
        {
            return Err(ContractError::InvalidField {
                field: "physics_cohort.maximum_time",
                reason: "physical horizon must be finite and positive",
            });
        }

        let operator = &self.intervention_operator;
        if operator.operator_id != OPERATOR_ID
            || operator.definition != OPERATOR_DEFINITION
            || operator.amplitude_semantics != AMPLITUDE_SEMANTICS
            || operator.application_semantics != APPLICATION_SEMANTICS
        {
            return Err(ContractError::UnsupportedOperator);
        }

        let controls = &self.admissible_controls;
        if controls.spatial_mode.minimum == 0
            || controls.spatial_mode.minimum > controls.spatial_mode.maximum
        {
            return Err(ContractError::InvalidField {
                field: "admissible_controls.spatial_mode",
                reason: "mode bounds must be positive and ordered",
            });
        }
        if !finite_ordered(controls.phase_rad.minimum, controls.phase_rad.maximum)
            || controls.phase_rad.minimum < -PI
            || controls.phase_rad.maximum > PI
        {
            return Err(ContractError::InvalidField {
                field: "admissible_controls.phase_rad",
                reason: "phase bounds must be ordered within [-pi, pi]",
            });
        }
        if !finite_ordered(
            controls.kick_amplitude.minimum,
            controls.kick_amplitude.maximum,
        ) || controls.kick_amplitude.minimum < 0.0
            || controls.kick_amplitude.maximum <= 0.0
            || controls.kick_amplitude.unit != self.physics_cohort.velocity_domain.velocity_unit
        {
            return Err(ContractError::InvalidField {
                field: "admissible_controls.kick_amplitude",
                reason: "amplitude bounds must be finite, non-negative, ordered, and use the velocity unit",
            });
        }
        if !finite_ordered(
            controls.application_time.minimum,
            controls.application_time.maximum,
        ) || controls.application_time.minimum < 0.0
            || controls.application_time.maximum > self.physics_cohort.maximum_time
            || controls.application_time.unit != self.physics_cohort.time_unit
        {
            return Err(ContractError::InvalidField {
                field: "admissible_controls.application_time",
                reason: "time bounds must lie inside the physical horizon and use its declared unit",
            });
        }

        let budget = &self.search_budget;
        if budget.maximum_proposals == 0
            || budget.maximum_high_fidelity_confirmations == 0
            || budget.maximum_high_fidelity_confirmations > budget.maximum_proposals
            || budget.maximum_reference_steps == 0
            || !budget.maximum_wall_time_seconds.is_finite()
            || budget.maximum_wall_time_seconds <= 0.0
        {
            return Err(ContractError::InvalidField {
                field: "search_budget",
                reason: "proposal, confirmation, step and wall-time budgets must be positive and internally consistent",
            });
        }

        let policy = &self.evaluation_policy;
        let post_pulse_end =
            controls.application_time.maximum + policy.minimum_post_pulse_observation_duration;
        if !policy.independent_confirmation_required
            || policy.minimum_spatial_refinement_levels == 0
            || policy.minimum_time_refinement_levels == 0
            || !policy.minimum_post_pulse_observation_duration.is_finite()
            || policy.minimum_post_pulse_observation_duration <= 0.0
            || !post_pulse_end.is_finite()
            || post_pulse_end > self.physics_cohort.maximum_time
        {
            return Err(ContractError::InvalidField {
                field: "evaluation_policy",
                reason: "independent confirmation, spatial/time refinement, and a positive post-pulse observation window are mandatory",
            });
        }
        for required in ["no_pulse", "sign_reversed", "phase_mismatch"] {
            if !policy.required_controls.iter().any(|item| item == required) {
                return Err(ContractError::InvalidField {
                    field: "evaluation_policy.required_controls",
                    reason: "no_pulse, sign_reversed and phase_mismatch controls are mandatory",
                });
            }
        }
        for required in ["complex_amplitude", "wrapped_phase", "waveform_error"] {
            if !policy.response_metrics.iter().any(|item| item == required) {
                return Err(ContractError::InvalidField {
                    field: "evaluation_policy.response_metrics",
                    reason: "complex amplitude, wrapped phase and waveform error are mandatory",
                });
            }
        }

        if self.split_role == "sealed_test" && self.acquisition_allowed {
            return Err(ContractError::SealedTestAcquisition);
        }
        if !matches!(
            self.split_role.as_str(),
            "discovery" | "development" | "confirmation" | "sealed_test"
        ) {
            return Err(ContractError::InvalidField {
                field: "split_role",
                reason: "unknown split role",
            });
        }
        if matches!(self.split_role.as_str(), "confirmation" | "sealed_test")
            && self
                .evaluation_policy
                .confirmation_reference_id
                .as_ref()
                .is_none_or(|value| value.trim().is_empty())
        {
            return Err(ContractError::InvalidField {
                field: "evaluation_policy.confirmation_reference_id",
                reason: "confirmation and sealed-test roles require a pinned independent reference",
            });
        }

        let provenance = &self.provenance;
        if !valid_sha256(&provenance.initial_state_sha256)
            || provenance.reference_solver_revision.trim().is_empty()
            || provenance
                .candidate_representation_revision
                .trim()
                .is_empty()
        {
            return Err(ContractError::InvalidField {
                field: "provenance",
                reason: "state digest and source revisions must be present and well formed",
            });
        }
        if !matches!(
            provenance.fixture_class.as_str(),
            "synthetic_schema_test_only" | "physical_candidate"
        ) {
            return Err(ContractError::InvalidField {
                field: "provenance.fixture_class",
                reason: "unknown fixture class",
            });
        }
        if self.contract_status == "synthetic_fixture_only" {
            if provenance.fixture_class != "synthetic_schema_test_only"
                || provenance.physical_execution_authorized
            {
                return Err(ContractError::SyntheticExecutionAuthorized);
            }
        } else if self.contract_status == "physical_candidate" {
            if provenance.fixture_class != "physical_candidate"
                || !provenance.physical_execution_authorized
                || provenance
                    .initial_state_sha256
                    .bytes()
                    .all(|byte| byte == b'0')
            {
                return Err(ContractError::PhysicalProvenanceIncomplete);
            }
        } else if self.contract_status != "development_contract" {
            return Err(ContractError::InvalidField {
                field: "contract_status",
                reason: "unknown contract status",
            });
        }
        if provenance.fixture_class == "synthetic_schema_test_only"
            && provenance.physical_execution_authorized
        {
            return Err(ContractError::SyntheticExecutionAuthorized);
        }

        Ok(())
    }

    /// Identifier used to scope candidate digests to this frozen family.
    pub fn contract_id(&self) -> &str {
        &self.contract_id
    }

    /// Stable fingerprint of the physics cohort, intervention family and initial-state identity.
    ///
    /// Candidate-representation revisions are intentionally excluded so the
    /// same physical intervention has the same identity across competitors.
    pub fn family_fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.akpd.pulse-family.v0.1\0");
        update_length_prefixed(&mut hasher, self.contract_id.as_bytes());
        update_length_prefixed(&mut hasher, self.physics_cohort.equation.as_bytes());
        update_length_prefixed(&mut hasher, self.physics_cohort.normalization_id.as_bytes());
        update_f64(&mut hasher, self.physics_cohort.spatial_domain.minimum);
        update_f64(&mut hasher, self.physics_cohort.spatial_domain.maximum);
        update_f64(&mut hasher, self.physics_cohort.velocity_domain.minimum);
        update_f64(&mut hasher, self.physics_cohort.velocity_domain.maximum);
        update_length_prefixed(
            &mut hasher,
            self.physics_cohort.velocity_domain.velocity_unit.as_bytes(),
        );
        update_length_prefixed(&mut hasher, self.physics_cohort.time_unit.as_bytes());
        update_f64(&mut hasher, self.physics_cohort.maximum_time);
        update_length_prefixed(
            &mut hasher,
            self.intervention_operator.operator_id.as_bytes(),
        );
        update_length_prefixed(
            &mut hasher,
            self.intervention_operator.definition.as_bytes(),
        );
        update_length_prefixed(
            &mut hasher,
            self.intervention_operator.amplitude_semantics.as_bytes(),
        );
        update_length_prefixed(
            &mut hasher,
            self.intervention_operator.application_semantics.as_bytes(),
        );
        hasher.update(&[u8::from(self.physics_cohort.spatial_domain.periodic)]);
        update_f64(
            &mut hasher,
            self.evaluation_policy
                .minimum_post_pulse_observation_duration,
        );
        hasher.update(&self.admissible_controls.spatial_mode.minimum.to_le_bytes());
        hasher.update(&self.admissible_controls.spatial_mode.maximum.to_le_bytes());
        update_f64(&mut hasher, self.admissible_controls.phase_rad.minimum);
        update_f64(&mut hasher, self.admissible_controls.phase_rad.maximum);
        update_f64(&mut hasher, self.admissible_controls.kick_amplitude.minimum);
        update_f64(&mut hasher, self.admissible_controls.kick_amplitude.maximum);
        update_length_prefixed(
            &mut hasher,
            self.admissible_controls.kick_amplitude.unit.as_bytes(),
        );
        update_f64(
            &mut hasher,
            self.admissible_controls.application_time.minimum,
        );
        update_f64(
            &mut hasher,
            self.admissible_controls.application_time.maximum,
        );
        update_length_prefixed(
            &mut hasher,
            self.admissible_controls.application_time.unit.as_bytes(),
        );
        update_length_prefixed(&mut hasher, self.provenance.initial_state_sha256.as_bytes());
        update_length_prefixed(
            &mut hasher,
            self.provenance.reference_solver_revision.as_bytes(),
        );
        hasher.update(&self.provenance.generator_seed.to_le_bytes());
        for source_ref in self
            .provenance
            .source_record_refs
            .iter()
            .collect::<BTreeSet<_>>()
        {
            update_length_prefixed(&mut hasher, source_ref.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    /// Initial-state generator seed recorded in the family provenance.
    pub fn generator_seed(&self) -> u64 {
        self.provenance.generator_seed
    }

    /// Immutable source records attached to the family.
    pub fn source_record_refs(&self) -> &[String] {
        &self.provenance.source_record_refs
    }

    /// Maximum raw grid points permitted by this contract.
    pub fn maximum_proposals(&self) -> usize {
        self.search_budget.maximum_proposals
    }

    /// Whether the family may be used to acquire / search intervention candidates.
    pub fn can_acquire(&self) -> bool {
        self.acquisition_allowed && self.split_role != "sealed_test"
    }

    /// True only for a provenance-marked physical candidate.
    ///
    /// This flag is not proof that the referenced files exist or that rights,
    /// numerical convergence, or independent confirmation have been verified.
    pub fn physical_execution_authorized(&self) -> bool {
        self.contract_status == "physical_candidate"
            && self.provenance.physical_execution_authorized
    }
}

/// Matrix-free positive-semidefinite empirical Gramian.
///
/// The represented matrix is `W = sum_i weight_i * direction_i * direction_i^T`,
/// but W is never materialized. Each stored factor is
/// sqrt(weight_i) * direction_i, so storage is O(rank * state_dimension).
///
/// For a reachability Gramian, directions may be propagated intervention
/// increments. For an observability Gramian, they may be pullbacks of
/// output-sensitivity directions to the checkpoint state. The caller is
/// responsible for deriving those directions for the actual discrete solver.
#[derive(Clone, Debug, PartialEq)]
pub struct EmpiricalGramian {
    dimension: usize,
    weighted_factors: Vec<Vec<f64>>,
}

impl EmpiricalGramian {
    /// Construct an empty Gramian for a positive state dimension.
    pub fn new(dimension: usize) -> Result<Self, ContractError> {
        if dimension == 0 {
            return Err(ContractError::InvalidGramianDimension);
        }
        Ok(Self {
            dimension,
            weighted_factors: Vec::new(),
        })
    }

    /// Add one non-negative weighted sensitivity / reachability direction.
    ///
    /// A zero weight is a no-op. Negative and non-finite weights are rejected.
    /// The direction and its weighted factor must remain finite.
    pub fn add_direction(&mut self, direction: &[f64], weight: f64) -> Result<(), ContractError> {
        if direction.len() != self.dimension {
            return Err(ContractError::GramianDimensionMismatch {
                expected: self.dimension,
                actual: direction.len(),
            });
        }
        if !weight.is_finite() || weight < 0.0 {
            return Err(ContractError::InvalidGramianWeight);
        }
        if direction.iter().any(|value| !value.is_finite()) {
            return Err(ContractError::InvalidGramianArithmetic);
        }
        if weight == 0.0 {
            return Ok(());
        }

        let scale = weight.sqrt();
        let factor = direction
            .iter()
            .map(|value| value * scale)
            .collect::<Vec<_>>();
        if factor.iter().any(|value| !value.is_finite()) {
            return Err(ContractError::InvalidGramianArithmetic);
        }
        self.weighted_factors.push(factor);
        Ok(())
    }

    /// Apply W to a vector without constructing the dense state-by-state matrix.
    pub fn apply(&self, vector: &[f64]) -> Result<Vec<f64>, ContractError> {
        self.validate_vector(vector)?;
        let mut output = vec![0.0; self.dimension];

        for factor in &self.weighted_factors {
            let projection = dot(factor, vector)?;
            for (slot, value) in output.iter_mut().zip(factor) {
                *slot += projection * value;
                if !slot.is_finite() {
                    return Err(ContractError::InvalidGramianArithmetic);
                }
            }
        }
        Ok(output)
    }

    /// Evaluate x^T W x as a sum of squared factor projections.
    pub fn quadratic_form(&self, vector: &[f64]) -> Result<f64, ContractError> {
        self.validate_vector(vector)?;
        let mut value = 0.0;
        for factor in &self.weighted_factors {
            let projection = dot(factor, vector)?;
            value += projection * projection;
            if !value.is_finite() {
                return Err(ContractError::InvalidGramianArithmetic);
            }
        }
        Ok(value)
    }

    /// Return trace(W), computed from the stored low-rank factors.
    pub fn trace(&self) -> Result<f64, ContractError> {
        let mut trace = 0.0;
        for factor in &self.weighted_factors {
            for value in factor {
                trace += value * value;
                if !trace.is_finite() {
                    return Err(ContractError::InvalidGramianArithmetic);
                }
            }
        }
        Ok(trace)
    }

    /// Fraction of Gramian trace captured by an orthonormal retained basis.
    ///
    /// This is a subspace diagnostic, not a balanced-truncation error bound or
    /// a physical-fidelity certificate. The basis must be orthonormal.
    pub fn captured_trace_fraction(
        &self,
        orthonormal_basis: &[Vec<f64>],
    ) -> Result<f64, ContractError> {
        const ORTHOGONALITY_TOLERANCE: f64 = 1.0e-8;

        let total = self.trace()?;
        if total <= 0.0 {
            return Err(ContractError::InvalidGramianArithmetic);
        }

        for (index, basis_vector) in orthonormal_basis.iter().enumerate() {
            self.validate_vector(basis_vector)?;
            for (other_index, other) in orthonormal_basis.iter().enumerate().take(index + 1) {
                let overlap = dot(basis_vector, other)?;
                let expected = if index == other_index { 1.0 } else { 0.0 };
                if (overlap - expected).abs() > ORTHOGONALITY_TOLERANCE {
                    return Err(ContractError::InvalidProjectionBasis);
                }
            }
        }

        let mut captured = 0.0;
        for basis_vector in orthonormal_basis {
            captured += self.quadratic_form(basis_vector)?;
            if !captured.is_finite() {
                return Err(ContractError::InvalidGramianArithmetic);
            }
        }
        let fraction = captured / total;
        if !fraction.is_finite() || !(0.0..=(1.0 + ORTHOGONALITY_TOLERANCE)).contains(&fraction) {
            return Err(ContractError::InvalidGramianArithmetic);
        }
        Ok(fraction.min(1.0))
    }

    /// State dimension of this Gramian.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of stored rank-one factors.
    pub fn factor_count(&self) -> usize {
        self.weighted_factors.len()
    }

    /// Number of stored scalar values (excluding vector/container overhead).
    pub fn stored_scalar_count(&self) -> usize {
        self.weighted_factors.len() * self.dimension
    }

    fn validate_vector(&self, vector: &[f64]) -> Result<(), ContractError> {
        if vector.len() != self.dimension {
            return Err(ContractError::GramianDimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(ContractError::InvalidGramianArithmetic);
        }
        Ok(())
    }
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, ContractError> {
    if left.len() != right.len() {
        return Err(ContractError::GramianDimensionMismatch {
            expected: left.len(),
            actual: right.len(),
        });
    }
    let mut result = 0.0;
    for (a, b) in left.iter().zip(right) {
        result += a * b;
        if !result.is_finite() {
            return Err(ContractError::InvalidGramianArithmetic);
        }
    }
    Ok(result)
}

/// One finite complex response coefficient with an explicit phase convention.
///
/// The convention is y = real + i * imag for the registered output-mode
/// quadrature. Missing measurements are represented by None in the matrix,
/// never by an implicit zero.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ComplexResponse {
    pub real: f64,
    pub imag: f64,
}

impl ComplexResponse {
    pub fn new(real: f64, imag: f64) -> Result<Self, ContractError> {
        if !real.is_finite() || !imag.is_finite() {
            return Err(ContractError::NonFiniteResponseValue);
        }
        Ok(Self { real, imag })
    }

    pub fn magnitude(self) -> f64 {
        self.real.hypot(self.imag)
    }

    pub fn phase_rad(self) -> Option<f64> {
        (self.magnitude() > 0.0).then(|| self.imag.atan2(self.real))
    }

    fn add(self, other: Self) -> Result<Self, ContractError> {
        Self::new(self.real + other.real, self.imag + other.imag)
    }

    fn sub(self, other: Self) -> Result<Self, ContractError> {
        Self::new(self.real - other.real, self.imag - other.imag)
    }

    fn scale(self, factor: f64) -> Result<Self, ContractError> {
        if !factor.is_finite() {
            return Err(ContractError::InvalidResponseAmplitude);
        }
        Self::new(self.real * factor, self.imag * factor)
    }

    fn norm_squared(self) -> f64 {
        self.real.mul_add(self.real, self.imag * self.imag)
    }
}

/// Phase-resolved output-mode by input-mode response matrix.
///
/// Axes retain caller-declared ordering and mode identity. An unmeasured
/// coefficient is None, which keeps missing / below-floor / unresolved data
/// distinct from a measured zero response.
#[derive(Clone, Debug, PartialEq)]
pub struct ModeResponseMatrix {
    input_modes: Vec<u32>,
    output_modes: Vec<u32>,
    entries: Vec<Option<ComplexResponse>>,
}

impl ModeResponseMatrix {
    pub fn new(input_modes: Vec<u32>, output_modes: Vec<u32>) -> Result<Self, ContractError> {
        if input_modes.is_empty() || output_modes.is_empty() {
            return Err(ContractError::EmptyResponseModeRegistry);
        }
        if has_duplicates(&input_modes) || has_duplicates(&output_modes) {
            return Err(ContractError::DuplicateResponseMode);
        }
        let entry_count = input_modes
            .len()
            .checked_mul(output_modes.len())
            .ok_or(ContractError::ResponseMatrixSizeOverflow)?;
        Ok(Self {
            input_modes,
            output_modes,
            entries: vec![None; entry_count],
        })
    }

    pub fn set(
        &mut self,
        input_mode: u32,
        output_mode: u32,
        response: ComplexResponse,
    ) -> Result<(), ContractError> {
        if !response.real.is_finite() || !response.imag.is_finite() {
            return Err(ContractError::NonFiniteResponseValue);
        }
        let index = self.entry_index(input_mode, output_mode)?;
        self.entries[index] = Some(response);
        Ok(())
    }

    pub fn get(
        &self,
        input_mode: u32,
        output_mode: u32,
    ) -> Result<Option<ComplexResponse>, ContractError> {
        let index = self.entry_index(input_mode, output_mode)?;
        Ok(self.entries[index])
    }

    pub fn is_complete(&self) -> bool {
        self.entries.iter().all(Option::is_some)
    }

    pub fn measured_entry_count(&self) -> usize {
        self.entries.iter().filter(|entry| entry.is_some()).count()
    }

    pub fn input_modes(&self) -> &[u32] {
        &self.input_modes
    }

    pub fn output_modes(&self) -> &[u32] {
        &self.output_modes
    }

    /// Relative Frobenius error against a complete reference matrix.
    ///
    /// Both matrices must declare exactly matching ordered mode registries.
    /// A zero reference norm is not assigned an arbitrary finite relative error.
    pub fn relative_frobenius_error(&self, reference: &Self) -> Result<f64, ContractError> {
        if self.input_modes != reference.input_modes || self.output_modes != reference.output_modes
        {
            return Err(ContractError::ResponseMatrixAxisMismatch);
        }
        if !self.is_complete() || !reference.is_complete() {
            return Err(ContractError::MissingResponseEntry);
        }

        let mut difference = 0.0;
        let mut reference_norm = 0.0;
        for (candidate, target) in self.entries.iter().zip(&reference.entries) {
            let candidate = (*candidate).ok_or(ContractError::MissingResponseEntry)?;
            let target = (*target).ok_or(ContractError::MissingResponseEntry)?;
            difference += candidate.sub(target)?.norm_squared();
            reference_norm += target.norm_squared();
            if !difference.is_finite() || !reference_norm.is_finite() {
                return Err(ContractError::InvalidGramianArithmetic);
            }
        }
        if reference_norm <= 0.0 {
            return Err(ContractError::ZeroResponseReferenceNorm);
        }
        let error = (difference / reference_norm).sqrt();
        if !error.is_finite() {
            return Err(ContractError::InvalidGramianArithmetic);
        }
        Ok(error)
    }

    fn entry_index(&self, input_mode: u32, output_mode: u32) -> Result<usize, ContractError> {
        let input_index = self
            .input_modes
            .iter()
            .position(|mode| *mode == input_mode)
            .ok_or(ContractError::UnknownResponseMode(input_mode))?;
        let output_index = self
            .output_modes
            .iter()
            .position(|mode| *mode == output_mode)
            .ok_or(ContractError::UnknownResponseMode(output_mode))?;
        Ok(output_index * self.input_modes.len() + input_index)
    }
}

fn has_duplicates(values: &[u32]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().any(|value| !seen.insert(*value))
}

/// Finite-amplitude first- and second-order coefficients from matched signed runs.
///
/// C1 = (R(+A) - R(-A)) / (2A)
/// C2 = (R(+A) + R(-A) - 2R(0)) / (2A^2)
///
/// C2 is the Taylor coefficient multiplying A^2, not the unscaled second
/// derivative. These are asymptotic coefficients only after amplitude
/// convergence has been established independently.
pub fn signed_response_coefficients(
    positive: ComplexResponse,
    negative: ComplexResponse,
    zero: ComplexResponse,
    amplitude: f64,
) -> Result<(ComplexResponse, ComplexResponse), ContractError> {
    if !amplitude.is_finite() || amplitude <= 0.0 {
        return Err(ContractError::InvalidResponseAmplitude);
    }
    let first_denominator = 2.0 * amplitude;
    let second_denominator = 2.0 * amplitude * amplitude;
    if !first_denominator.is_finite()
        || first_denominator <= 0.0
        || !second_denominator.is_finite()
        || second_denominator <= 0.0
    {
        return Err(ContractError::InvalidResponseAmplitude);
    }
    let first = positive.sub(negative)?.scale(1.0 / first_denominator)?;
    let second = positive
        .add(negative)?
        .sub(zero.scale(2.0)?)?
        .scale(1.0 / second_denominator)?;
    Ok((first, second))
}

/// Mixed second-order finite-amplitude coefficient for two distinct controls.
///
/// C_ab = [R(+a,+b)-R(+a,-b)-R(-a,+b)+R(-a,-b)] / (4ab)
///
/// This identifies one registered mixed direction. It does not reconstruct
/// a general nonlinear Volterra kernel from a finite intervention set.
pub fn mixed_response_coefficient(
    positive_positive: ComplexResponse,
    positive_negative: ComplexResponse,
    negative_positive: ComplexResponse,
    negative_negative: ComplexResponse,
    amplitude_a: f64,
    amplitude_b: f64,
) -> Result<ComplexResponse, ContractError> {
    if !amplitude_a.is_finite()
        || !amplitude_b.is_finite()
        || amplitude_a <= 0.0
        || amplitude_b <= 0.0
    {
        return Err(ContractError::InvalidResponseAmplitude);
    }
    let denominator = 4.0 * amplitude_a * amplitude_b;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(ContractError::InvalidResponseAmplitude);
    }
    let numerator = positive_positive
        .sub(positive_negative)?
        .sub(negative_positive)?
        .add(negative_negative)?;
    numerator.scale(1.0 / denominator)
}

/// Wrapped signed phase difference candidate minus reference in [-pi, pi).
///
/// Returns None if either complex response has zero magnitude; phase is then
/// undefined and must not be reported as zero.
pub fn wrapped_phase_difference(
    candidate: ComplexResponse,
    reference: ComplexResponse,
) -> Option<f64> {
    let candidate_phase = candidate.phase_rad()?;
    let reference_phase = reference.phase_rad()?;
    let difference = candidate_phase - reference_phase;
    Some((difference + PI).rem_euclid(TAU) - PI)
}

/// Serialized bytes required to store one checkpoint and its interpretation metadata.
///
/// These counters are caller-supplied byte lengths from the actual serialization.
/// The ledger intentionally separates representation payload from the metadata
/// required to decode, transfer, validate and replay the state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateStorageBreakdown {
    pub state_payload_bytes: u64,
    pub basis_or_codec_metadata_bytes: u64,
    pub coordinate_and_normalization_bytes: u64,
    pub component_and_lineage_bytes: u64,
    pub residual_memory_bytes: u64,
    pub uncertainty_and_validity_bytes: u64,
    pub provenance_and_contract_bytes: u64,
}

impl StateStorageBreakdown {
    /// Sum all serialized checkpoint bytes, including required interpretation metadata.
    pub fn total_state_bytes(self) -> Result<u64, ContractError> {
        [
            self.state_payload_bytes,
            self.basis_or_codec_metadata_bytes,
            self.coordinate_and_normalization_bytes,
            self.component_and_lineage_bytes,
            self.residual_memory_bytes,
            self.uncertainty_and_validity_bytes,
            self.provenance_and_contract_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(ContractError::StorageByteCountOverflow)
        })
    }
}

/// Separately accounted persistent model assets and peak execution memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KineticStorageAccounting {
    pub checkpoint: StateStorageBreakdown,
    pub model_asset_bytes: u64,
    pub decoder_asset_bytes: u64,
    pub peak_working_set_bytes: u64,
}

impl KineticStorageAccounting {
    /// Checkpoint bytes only; excludes model / decoder assets and working memory.
    pub fn checkpoint_bytes(self) -> Result<u64, ContractError> {
        self.checkpoint.total_state_bytes()
    }

    /// Persistent model / decoder assets, not repeated for each checkpoint.
    pub fn persistent_asset_bytes(self) -> Result<u64, ContractError> {
        self.model_asset_bytes
            .checked_add(self.decoder_asset_bytes)
            .ok_or(ContractError::StorageByteCountOverflow)
    }

    /// Average model / decoder asset bytes allocated to one rollout.
    pub fn amortized_asset_bytes_per_rollout(
        self,
        rollout_count: u64,
    ) -> Result<f64, ContractError> {
        if rollout_count == 0 {
            return Err(ContractError::ZeroAmortizationRollouts);
        }
        Ok(self.persistent_asset_bytes()? as f64 / rollout_count as f64)
    }

    /// Checkpoint bytes plus amortized model / decoder bytes per rollout.
    ///
    /// Peak working-set memory is reported separately and is not added to
    /// persistent storage, because it is a runtime resource rather than a
    /// checkpoint or model artifact.
    pub fn amortized_persistent_bytes_per_rollout(
        self,
        rollout_count: u64,
    ) -> Result<f64, ContractError> {
        Ok(self.checkpoint_bytes()? as f64
            + self.amortized_asset_bytes_per_rollout(rollout_count)?)
    }
}

/// Affine min-max coordinate map fitted to one sample's observed support.
///
/// The input/output intervals are explicit so an audit can distinguish the
/// sample-dependent support from the coordinate convention. This primitive
/// is a normalization mechanism, not an endorsement of per-sample min-max
/// normalization for open or periodic physical domains.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMaxCoordinateMap {
    input_minimum: f64,
    input_maximum: f64,
    output_minimum: f64,
    output_maximum: f64,
}

impl MinMaxCoordinateMap {
    /// Fit an affine min-max map from finite sample coordinates.
    pub fn fit(
        positions: &[f64],
        output_minimum: f64,
        output_maximum: f64,
    ) -> Result<Self, ContractError> {
        if positions.is_empty() {
            return Err(ContractError::EmptyNormalizationSample);
        }
        if positions.iter().any(|value| !value.is_finite()) {
            return Err(ContractError::NonFiniteNormalizationCoordinate);
        }
        let input_minimum = positions.iter().copied().fold(f64::INFINITY, f64::min);
        let input_maximum = positions.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::new(input_minimum, input_maximum, output_minimum, output_maximum)
    }

    /// Construct a map from explicit finite, strictly increasing intervals.
    pub fn new(
        input_minimum: f64,
        input_maximum: f64,
        output_minimum: f64,
        output_maximum: f64,
    ) -> Result<Self, ContractError> {
        let input_width = input_maximum - input_minimum;
        let output_width = output_maximum - output_minimum;
        if !input_minimum.is_finite()
            || !input_maximum.is_finite()
            || !input_width.is_finite()
            || input_width <= 0.0
        {
            return Err(ContractError::InvalidNormalizationSupport);
        }
        if !output_minimum.is_finite()
            || !output_maximum.is_finite()
            || !output_width.is_finite()
            || output_width <= 0.0
        {
            return Err(ContractError::InvalidNormalizationOutputInterval);
        }
        Ok(Self {
            input_minimum,
            input_maximum,
            output_minimum,
            output_maximum,
        })
    }

    /// Apply the affine map; values outside the fitted support extrapolate.
    pub fn apply(self, position: f64) -> Result<f64, ContractError> {
        if !position.is_finite() {
            return Err(ContractError::NonFiniteNormalizationCoordinate);
        }
        let input_width = self.input_maximum - self.input_minimum;
        let output_width = self.output_maximum - self.output_minimum;
        let normalized =
            self.output_minimum + (position - self.input_minimum) / input_width * output_width;
        if !normalized.is_finite() {
            return Err(ContractError::InvalidNormalizationSupport);
        }
        Ok(normalized)
    }

    pub fn input_support(self) -> (f64, f64) {
        (self.input_minimum, self.input_maximum)
    }

    pub fn output_interval(self) -> (f64, f64) {
        (self.output_minimum, self.output_maximum)
    }
}

/// One intervention in a frozen pulse family.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PulseCandidate {
    pub spatial_mode: u32,
    pub phase_rad: f64,
    pub kick_velocity_amplitude: f64,
    pub application_time: f64,
}

impl PulseCandidate {
    /// Check that the candidate belongs to this family and uses finite, admissible values.
    pub fn validate_for(&self, contract: &PulseFamilyContract) -> Result<(), ContractError> {
        contract.validate()?;
        let bounds = &contract.admissible_controls;

        if self.spatial_mode < bounds.spatial_mode.minimum
            || self.spatial_mode > bounds.spatial_mode.maximum
        {
            return Err(ContractError::CandidateOutOfRange("spatial_mode"));
        }
        if !self.phase_rad.is_finite()
            || self.phase_rad < bounds.phase_rad.minimum
            || self.phase_rad > bounds.phase_rad.maximum
        {
            return Err(ContractError::CandidateOutOfRange("phase_rad"));
        }
        if !self.kick_velocity_amplitude.is_finite()
            || self.kick_velocity_amplitude < bounds.kick_amplitude.minimum
            || self.kick_velocity_amplitude > bounds.kick_amplitude.maximum
        {
            return Err(ContractError::CandidateOutOfRange(
                "kick_velocity_amplitude",
            ));
        }
        if !self.application_time.is_finite()
            || self.application_time < bounds.application_time.minimum
            || self.application_time > bounds.application_time.maximum
            || self.application_time > contract.physics_cohort.maximum_time
        {
            return Err(ContractError::CandidateOutOfRange("application_time"));
        }
        Ok(())
    }

    /// Canonical BLAKE3 identity for this intervention under the frozen family.
    ///
    /// Phase is wrapped to [-pi, pi), and signed zero is canonicalized. A zero
    /// kick is represented as one no-op regardless of mode, phase or time.
    pub fn canonical_digest(
        &self,
        contract: &PulseFamilyContract,
    ) -> Result<String, ContractError> {
        self.validate_for(contract)?;

        let no_op = self.kick_velocity_amplitude == 0.0;
        let mode = if no_op { 0 } else { self.spatial_mode };
        let phase = if no_op {
            0.0
        } else {
            normalize_phase(self.phase_rad)
        };
        let amplitude = canonical_zero(self.kick_velocity_amplitude);
        let application_time = if no_op {
            0.0
        } else {
            canonical_zero(self.application_time)
        };

        let mut hasher = blake3::Hasher::new();
        hasher.update(DIGEST_DOMAIN);
        let family_fingerprint = contract.family_fingerprint();
        update_length_prefixed(&mut hasher, family_fingerprint.as_bytes());
        update_length_prefixed(
            &mut hasher,
            contract.intervention_operator.operator_id.as_bytes(),
        );
        hasher.update(&mode.to_le_bytes());
        hasher.update(&phase.to_bits().to_le_bytes());
        hasher.update(&amplitude.to_bits().to_le_bytes());
        hasher.update(&application_time.to_bits().to_le_bytes());
        Ok(hasher.finalize().to_hex().to_string())
    }

    /// Return the equivalent non-negative-amplitude pulse with reversed sign.
    ///
    /// A sinusoidal kick changes sign under a pi phase shift. The returned
    /// value must still be checked against the family's phase range.
    pub fn sign_reversed(self) -> Self {
        Self {
            phase_rad: normalize_phase(self.phase_rad + PI),
            ..self
        }
    }
}

/// Deterministic tensor-grid resolution for a bounded pulse family.
///
/// Each numeric axis includes both endpoints when there are at least two
/// steps. A one-point axis uses its midpoint. Modes are enumerated as every
/// integer in the range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridSpec {
    pub phase_steps: usize,
    pub amplitude_steps: usize,
    pub application_time_steps: usize,
}

impl GridSpec {
    pub fn validate(self) -> Result<(), ContractError> {
        if self.phase_steps == 0 || self.amplitude_steps == 0 || self.application_time_steps == 0 {
            return Err(ContractError::InvalidGrid);
        }
        Ok(())
    }
}

/// Enumerate a deterministic, de-duplicated grid within the declared proposal budget.
///
/// The raw Cartesian grid size is bounded before enumeration. Physically
/// equivalent endpoint phases and zero-amplitude no-ops are de-duplicated by
/// canonical digest without relying on hash-map iteration order.
pub fn enumerate_grid(
    contract: &PulseFamilyContract,
    grid: GridSpec,
) -> Result<Vec<PulseCandidate>, ContractError> {
    contract.validate()?;
    grid.validate()?;

    let modes = usize::try_from(
        u64::from(contract.admissible_controls.spatial_mode.maximum)
            - u64::from(contract.admissible_controls.spatial_mode.minimum)
            + 1,
    )
    .map_err(|_| ContractError::GridSizeOverflow)?;
    let raw_count = modes
        .checked_mul(grid.phase_steps)
        .and_then(|value| value.checked_mul(grid.amplitude_steps))
        .and_then(|value| value.checked_mul(grid.application_time_steps))
        .ok_or(ContractError::GridSizeOverflow)?;

    if raw_count > contract.maximum_proposals() {
        return Err(ContractError::ProposalBudgetExceeded {
            requested: raw_count,
            budget: contract.maximum_proposals(),
        });
    }

    let phase_values = linspace(
        contract.admissible_controls.phase_rad.minimum,
        contract.admissible_controls.phase_rad.maximum,
        grid.phase_steps,
    )?;
    let amplitude_values = linspace(
        contract.admissible_controls.kick_amplitude.minimum,
        contract.admissible_controls.kick_amplitude.maximum,
        grid.amplitude_steps,
    )?;
    let time_values = linspace(
        contract.admissible_controls.application_time.minimum,
        contract.admissible_controls.application_time.maximum,
        grid.application_time_steps,
    )?;

    let mut seen = BTreeSet::new();
    let mut candidates = Vec::new();

    for mode in contract.admissible_controls.spatial_mode.minimum
        ..=contract.admissible_controls.spatial_mode.maximum
    {
        for &phase_rad in &phase_values {
            for &kick_velocity_amplitude in &amplitude_values {
                for &application_time in &time_values {
                    let candidate = PulseCandidate {
                        spatial_mode: mode,
                        phase_rad,
                        kick_velocity_amplitude,
                        application_time,
                    };
                    let digest = candidate.canonical_digest(contract)?;
                    if seen.insert(digest) {
                        candidates.push(candidate);
                    }
                }
            }
        }
    }

    Ok(candidates)
}

fn finite_ordered(minimum: f64, maximum: f64) -> bool {
    minimum.is_finite() && maximum.is_finite() && minimum <= maximum
}

fn finite_increasing(minimum: f64, maximum: f64) -> bool {
    minimum.is_finite() && maximum.is_finite() && minimum < maximum
}

fn valid_contract_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() >= 6
        && bytes.len() <= 64
        && bytes.starts_with(b"AKPD-")
        && (bytes[5].is_ascii_uppercase() || bytes[5].is_ascii_digit())
        && bytes[6..].iter().all(|byte| {
            byte.is_ascii_uppercase()
                || byte.is_ascii_digit()
                || matches!(*byte, b'-' | b'_' | b'.')
        })
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn normalize_phase(phase: f64) -> f64 {
    let wrapped = (phase + PI).rem_euclid(TAU) - PI;
    canonical_zero(wrapped)
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn update_length_prefixed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    let length = bytes.len() as u64;
    hasher.update(&length.to_le_bytes());
    hasher.update(bytes);
}

fn update_f64(hasher: &mut blake3::Hasher, value: f64) {
    hasher.update(&canonical_zero(value).to_bits().to_le_bytes());
}

fn linspace(minimum: f64, maximum: f64, steps: usize) -> Result<Vec<f64>, ContractError> {
    if steps == 0 || !finite_ordered(minimum, maximum) {
        return Err(ContractError::InvalidGrid);
    }
    if steps == 1 {
        let midpoint = minimum * 0.5 + maximum * 0.5;
        if !midpoint.is_finite() {
            return Err(ContractError::InvalidGrid);
        }
        return Ok(vec![canonical_zero(midpoint)]);
    }

    let span = maximum - minimum;
    if !span.is_finite() {
        return Err(ContractError::InvalidGrid);
    }

    let denominator = (steps - 1) as f64;
    let mut values = Vec::with_capacity(steps);
    for index in 0..steps {
        let value = if index == 0 {
            minimum
        } else if index + 1 == steps {
            maximum
        } else {
            minimum + span * (index as f64 / denominator)
        };
        if !value.is_finite() {
            return Err(ContractError::InvalidGrid);
        }
        values.push(canonical_zero(value));
    }
    Ok(values)
}

/// Periodic Fourier-character feature lift for a normalized torus coordinate.
///
/// The input is interpreted modulo one. For harmonic k the emitted pair is
/// [cos(2*pi*k*xi), sin(2*pi*k*xi)]. No raw non-periodic coordinate is included,
/// so the feature map and all of its derivatives are periodic at the seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeriodicTorusLift {
    max_harmonic: usize,
}

impl PeriodicTorusLift {
    /// Construct a periodic lift with harmonics 1 through max_harmonic.
    pub fn new(max_harmonic: usize) -> Result<Self, ContractError> {
        if max_harmonic == 0 || max_harmonic > MAX_PERIODIC_LIFT_HARMONICS {
            return Err(ContractError::InvalidHarmonicBudget);
        }
        Ok(Self { max_harmonic })
    }

    /// Return interleaved cosine/sine features, wrapping xi modulo one.
    pub fn features(&self, xi: f64) -> Result<Vec<f64>, ContractError> {
        let xi = self.wrapped_coordinate(xi)?;
        let mut features = Vec::with_capacity(self.max_harmonic * 2);
        for harmonic in 1..=self.max_harmonic {
            let phase = TAU * harmonic as f64 * xi;
            features.push(phase.cos());
            features.push(phase.sin());
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err(ContractError::InvalidGramianArithmetic);
        }
        Ok(features)
    }

    /// Derivatives of the interleaved features with respect to normalized xi.
    pub fn derivative(&self, xi: f64) -> Result<Vec<f64>, ContractError> {
        let xi = self.wrapped_coordinate(xi)?;
        let mut derivative = Vec::with_capacity(self.max_harmonic * 2);
        for harmonic in 1..=self.max_harmonic {
            let angular_frequency = TAU * harmonic as f64;
            let phase = angular_frequency * xi;
            derivative.push(-angular_frequency * phase.sin());
            derivative.push(angular_frequency * phase.cos());
        }
        if derivative.iter().any(|value| !value.is_finite()) {
            return Err(ContractError::InvalidGramianArithmetic);
        }
        Ok(derivative)
    }

    pub fn max_harmonic(&self) -> usize {
        self.max_harmonic
    }

    fn wrapped_coordinate(&self, xi: f64) -> Result<f64, ContractError> {
        if !xi.is_finite() {
            return Err(ContractError::NonFiniteNormalizationCoordinate);
        }
        Ok(xi.rem_euclid(1.0))
    }
}

/// Non-periodic normalized coordinate features for a known open interval.
///
/// Unlike PeriodicTorusLift, this map deliberately preserves the distinction
/// between xi=0 and xi=1 and reports distance to each physical boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpenBoundaryCoordinateFeatures {
    pub coordinate: f64,
    pub distance_to_lower_boundary: f64,
    pub distance_to_upper_boundary: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenBoundaryCoordinateLift;

impl OpenBoundaryCoordinateLift {
    pub fn features(xi: f64) -> Result<OpenBoundaryCoordinateFeatures, ContractError> {
        if !xi.is_finite() {
            return Err(ContractError::NonFiniteNormalizationCoordinate);
        }
        if !(0.0..=1.0).contains(&xi) {
            return Err(ContractError::InvalidOpenBoundaryCoordinate);
        }
        Ok(OpenBoundaryCoordinateFeatures {
            coordinate: xi,
            distance_to_lower_boundary: xi,
            distance_to_upper_boundary: 1.0 - xi,
        })
    }
}

/// Audited timing envelope for a sparse, asynchronous kinetic-control interface.
#[derive(Clone, Debug, PartialEq)]
pub struct SensorActuatorTimingSummary {
    pub contract_id: String,
    pub diagnostic_channel_count: usize,
    pub actuator_channel_count: usize,
    pub maximum_observation_age_seconds: f64,
    pub maximum_actuator_delivery_delay_seconds: f64,
    pub controller_update_period_seconds: f64,
    pub controller_deadline_seconds: f64,
    pub controller_worst_case_execution_seconds: f64,
}

/// Parse and semantically audit a sensor/actuator timing contract.
pub fn validate_sensor_actuator_timing_contract_json(
    input: &str,
) -> Result<SensorActuatorTimingSummary, ContractError> {
    let value: serde_json::Value =
        serde_json::from_str(input).map_err(|error| ContractError::JsonParse(error.to_string()))?;
    validate_sensor_actuator_timing_contract_value(&value)
}

/// Audit timing consistency after the caller has validated the JSON Schema.
///
/// This checks that the declared observation age covers the sample period,
/// transport latency, jitter and clock skew; that every channel fits within the
/// controller's maximum observation age; that WCET fits the deadline and update
/// period; and that hidden full-state access is prohibited.
pub fn validate_sensor_actuator_timing_contract_value(
    value: &serde_json::Value,
) -> Result<SensorActuatorTimingSummary, ContractError> {
    if timing_string(value, "schema_version")? != "0.1" {
        return Err(control_timing_error(
            "schema_version",
            "unsupported timing contract version",
        ));
    }
    let contract_id = timing_string(value, "contract_id")?.to_owned();
    if contract_id.trim().is_empty() {
        return Err(control_timing_error(
            "contract_id",
            "contract identity must be non-empty",
        ));
    }

    let clock_skew =
        timing_number(value, "clock.maximum_clock_skew_seconds")?;
    if clock_skew < 0.0 {
        return Err(control_timing_error(
            "clock.maximum_clock_skew_seconds",
            "clock skew must be non-negative",
        ));
    }

    let diagnostics = timing_array(value, "diagnostics")?;
    let actuators = timing_array(value, "actuators")?;
    if diagnostics.is_empty() || actuators.is_empty() {
        return Err(control_timing_error(
            "diagnostics",
            "at least one diagnostic and actuator channel are required",
        ));
    }

    let mut diagnostic_ids = BTreeSet::new();
    let mut maximum_observation_age_seconds = 0.0_f64;
    for diagnostic in diagnostics {
        let channel_id = timing_string(diagnostic, "channel_id")?;
        if !diagnostic_ids.insert(channel_id.to_owned()) {
            return Err(control_timing_error(
                "diagnostics.channel_id",
                "diagnostic channel IDs must be unique",
            ));
        }
        let sampling_period =
            timing_number(diagnostic, "sampling_period_seconds")?;
        let transport_latency =
            timing_number(diagnostic, "transport_latency_seconds")?;
        let timestamp_age_bound =
            timing_number(diagnostic, "timestamp_age_bound_seconds")?;
        let jitter = timing_number(diagnostic, "jitter_bound_seconds")?;
        let noise_std = timing_number(diagnostic, "noise_std")?;
        if sampling_period <= 0.0
            || transport_latency < 0.0
            || timestamp_age_bound < 0.0
            || jitter < 0.0
            || noise_std < 0.0
        {
            return Err(control_timing_error(
                "diagnostics",
                "sampling period must be positive and latency/age/jitter/noise non-negative",
            ));
        }
        let minimum_age_bound = sampling_period + transport_latency + jitter + clock_skew;
        let allowance = 1.0e-12_f64.max(minimum_age_bound.abs() * 1.0e-10);
        if !minimum_age_bound.is_finite()
            || timestamp_age_bound + allowance < minimum_age_bound
        {
            return Err(control_timing_error(
                "diagnostics.timestamp_age_bound_seconds",
                "age bound must cover sampling period + transport latency + jitter + clock skew",
            ));
        }
        maximum_observation_age_seconds =
            maximum_observation_age_seconds.max(timestamp_age_bound);
    }

    let mut actuator_ids = BTreeSet::new();
    let mut maximum_actuator_delivery_delay_seconds = 0.0_f64;
    for actuator in actuators {
        let channel_id = timing_string(actuator, "channel_id")?;
        if !actuator_ids.insert(channel_id.to_owned()) {
            return Err(control_timing_error(
                "actuators.channel_id",
                "actuator channel IDs must be unique",
            ));
        }
        let update_period = timing_number(actuator, "update_period_seconds")?;
        let delivery_delay = timing_number(actuator, "delivery_delay_seconds")?;
        let bandwidth = timing_number(actuator, "bandwidth_hz")?;
        let amplitude = timing_number(actuator, "maximum_amplitude")?;
        let slew = timing_number(actuator, "maximum_slew_per_second")?;
        if update_period <= 0.0
            || delivery_delay < 0.0
            || bandwidth <= 0.0
            || amplitude <= 0.0
            || slew <= 0.0
        {
            return Err(control_timing_error(
                "actuators",
                "actuator update period/bandwidth/amplitude/slew must be positive and delay non-negative",
            ));
        }
        maximum_actuator_delivery_delay_seconds =
            maximum_actuator_delivery_delay_seconds.max(delivery_delay);
    }

    let controller_update_period =
        timing_number(value, "controller.update_period_seconds")?;
    let controller_wcet =
        timing_number(value, "controller.worst_case_execution_seconds")?;
    let controller_deadline = timing_number(value, "controller.deadline_seconds")?;
    let maximum_allowed_observation_age =
        timing_number(value, "controller.maximum_observation_age_seconds")?;
    if controller_update_period <= 0.0
        || controller_wcet < 0.0
        || controller_deadline <= 0.0
        || maximum_allowed_observation_age <= 0.0
    {
        return Err(control_timing_error(
            "controller",
            "controller update/deadline/age must be positive and WCET non-negative",
        ));
    }
    if controller_wcet + clock_skew > controller_deadline
        || controller_deadline > controller_update_period
    {
        return Err(control_timing_error(
            "controller.deadline_seconds",
            "WCET plus clock skew must fit the deadline, which must fit the update period",
        ));
    }
    if maximum_observation_age_seconds > maximum_allowed_observation_age {
        return Err(control_timing_error(
            "controller.maximum_observation_age_seconds",
            "controller age allowance is smaller than a diagnostic channel's declared age bound",
        ));
    }
    if timing_bool(value, "controller.full_state_access")? {
        return Err(control_timing_error(
            "controller.full_state_access",
            "deployment interface cannot expose hidden full state",
        ));
    }
    if timing_bool(value, "claim_boundary.closed_loop_experiment_performed")?
        || timing_bool(value, "claim_boundary.stability_claimed")?
    {
        return Err(control_timing_error(
            "claim_boundary",
            "synthetic timing fixtures cannot claim closed-loop or stability results",
        ));
    }

    Ok(SensorActuatorTimingSummary {
        contract_id,
        diagnostic_channel_count: diagnostics.len(),
        actuator_channel_count: actuators.len(),
        maximum_observation_age_seconds,
        maximum_actuator_delivery_delay_seconds,
        controller_update_period_seconds: controller_update_period,
        controller_deadline_seconds: controller_deadline,
        controller_worst_case_execution_seconds: controller_wcet,
    })
}

fn control_timing_error(field: &'static str, reason: &'static str) -> ContractError {
    ContractError::InvalidControlTimingField { field, reason }
}

fn timing_value<'a>(
    root: &'a serde_json::Value,
    field: &'static str,
) -> Result<&'a serde_json::Value, ContractError> {
    let mut current = root;
    for segment in field.split('.') {
        current = current
            .get(segment)
            .ok_or_else(|| control_timing_error(field, "required field is missing"))?;
    }
    Ok(current)
}

fn timing_string<'a>(
    root: &'a serde_json::Value,
    field: &'static str,
) -> Result<&'a str, ContractError> {
    timing_value(root, field)?
        .as_str()
        .ok_or_else(|| control_timing_error(field, "must be a string"))
}

fn timing_bool(root: &serde_json::Value, field: &'static str) -> Result<bool, ContractError> {
    timing_value(root, field)?
        .as_bool()
        .ok_or_else(|| control_timing_error(field, "must be a boolean"))
}

fn timing_number(root: &serde_json::Value, field: &'static str) -> Result<f64, ContractError> {
    let number = timing_value(root, field)?
        .as_f64()
        .ok_or_else(|| control_timing_error(field, "must be a finite number"))?;
    if !number.is_finite() {
        return Err(control_timing_error(field, "must be finite"));
    }
    Ok(number)
}

fn timing_array<'a>(
    root: &'a serde_json::Value,
    field: &'static str,
) -> Result<&'a [serde_json::Value], ContractError> {
    timing_value(root, field)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| control_timing_error(field, "must be an array"))
}

/// Result of projecting a kinetic fluctuation into a weighted discrete micro-space.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedMicroProjection {
    /// Projected fluctuation. Every supplied constraint moment is zero within tolerance.
    pub micro_fluctuation: Vec<f64>,
    /// Constraint moments before projection.
    pub pre_projection_moments: Vec<f64>,
    /// Constraint moments after projection.
    pub post_projection_moments: Vec<f64>,
    /// Weighted norm of the correction, using the supplied diagonal metric.
    pub weighted_correction_norm: f64,
}

/// Project values into the nullspace of caller-supplied discrete moment constraints.
///
/// Solves the weighted minimum-correction problem
///
///   minimize 0.5 * (g_micro - g)^T W (g_micro - g)
///   subject to C * g_micro = 0,
///
/// where W is a positive diagonal metric and each row of C is one discrete
/// collision-invariant / macro-moment constraint. The caller must derive C
/// from the exact discrete collision operator and declared micro-macro split;
/// this routine does not assume which moments are invariant.
///
/// The KKT system is assembled as C W^-1 C^T and solved with pivoted Gaussian
/// elimination. Empty, non-finite, non-positive, dimensionally inconsistent,
/// or linearly dependent constraints fail closed.
pub fn project_weighted_micro_fluctuation(
    values: &[f64],
    constraints: &[Vec<f64>],
    diagonal_weights: &[f64],
    tolerance: f64,
) -> Result<WeightedMicroProjection, ContractError> {
    let dimension = values.len();
    if dimension == 0
        || diagonal_weights.len() != dimension
        || constraints.is_empty()
        || !tolerance.is_finite()
        || tolerance <= 0.0
    {
        return Err(ContractError::InvalidProjectionBasis);
    }
    if values.iter().any(|value| !value.is_finite())
        || diagonal_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
        || constraints
            .iter()
            .any(|row| row.len() != dimension || row.iter().any(|value| !value.is_finite()))
    {
        return Err(ContractError::InvalidGramianArithmetic);
    }

    let constraint_count = constraints.len();
    let mut gram = vec![vec![0.0; constraint_count]; constraint_count];
    let mut rhs = vec![0.0; constraint_count];
    let mut pre_projection_moments = vec![0.0; constraint_count];

    for (row_index, row) in constraints.iter().enumerate() {
        for (value, coefficient) in values.iter().zip(row) {
            pre_projection_moments[row_index] =
                coefficient.mul_add(*value, pre_projection_moments[row_index]);
        }
        rhs[row_index] = pre_projection_moments[row_index];
        for (column_index, other_row) in constraints.iter().enumerate() {
            let mut entry = 0.0;
            for ((left, right), weight) in row.iter().zip(other_row).zip(diagonal_weights) {
                entry += left * right / weight;
            }
            if !entry.is_finite() {
                return Err(ContractError::InvalidGramianArithmetic);
            }
            gram[row_index][column_index] = entry;
        }
    }

    let multipliers = solve_dense_system(gram, rhs)?;
    let mut micro_fluctuation = values.to_vec();
    for (coefficient_index, value) in micro_fluctuation.iter_mut().enumerate() {
        let mut correction = 0.0;
        for (row, multiplier) in constraints.iter().zip(&multipliers) {
            correction += row[coefficient_index] * multiplier;
        }
        *value -= correction / diagonal_weights[coefficient_index];
        if !value.is_finite() {
            return Err(ContractError::InvalidGramianArithmetic);
        }
    }

    let mut post_projection_moments = vec![0.0; constraint_count];
    for (row_index, row) in constraints.iter().enumerate() {
        for (value, coefficient) in micro_fluctuation.iter().zip(row) {
            post_projection_moments[row_index] =
                coefficient.mul_add(*value, post_projection_moments[row_index]);
        }
    }
    let maximum_moment_scale = pre_projection_moments
        .iter()
        .map(|moment| moment.abs())
        .fold(1.0_f64, f64::max);
    if post_projection_moments
        .iter()
        .any(|moment| moment.abs() > tolerance * maximum_moment_scale)
    {
        return Err(ContractError::InvalidProjectionBasis);
    }

    let mut correction_norm_squared = 0.0;
    for ((projected, original), weight) in micro_fluctuation.iter().zip(values).zip(diagonal_weights) {
        let correction = projected - original;
        correction_norm_squared += weight * correction * correction;
    }
    if !correction_norm_squared.is_finite() || correction_norm_squared < 0.0 {
        return Err(ContractError::InvalidGramianArithmetic);
    }

    Ok(WeightedMicroProjection {
        micro_fluctuation,
        pre_projection_moments,
        post_projection_moments,
        weighted_correction_norm: correction_norm_squared.sqrt(),
    })
}

fn solve_dense_system(
    mut matrix: Vec<Vec<f64>>,
    mut rhs: Vec<f64>,
) -> Result<Vec<f64>, ContractError> {
    let dimension = rhs.len();
    if dimension == 0
        || matrix.len() != dimension
        || matrix.iter().any(|row| row.len() != dimension)
        || matrix
            .iter()
            .flatten()
            .chain(rhs.iter())
            .any(|value| !value.is_finite())
    {
        return Err(ContractError::InvalidProjectionBasis);
    }

    let scale = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(ContractError::InvalidProjectionBasis);
    }
    let pivot_floor = f64::EPSILON * scale * dimension as f64 * 128.0;

    for pivot_index in 0..dimension {
        let pivot_row = (pivot_index..dimension)
            .max_by(|left, right| {
                matrix[*left][pivot_index]
                    .abs()
                    .total_cmp(&matrix[*right][pivot_index].abs())
            })
            .ok_or(ContractError::InvalidProjectionBasis)?;
        if matrix[pivot_row][pivot_index].abs() <= pivot_floor {
            return Err(ContractError::InvalidProjectionBasis);
        }
        matrix.swap(pivot_index, pivot_row);
        rhs.swap(pivot_index, pivot_row);

        let pivot = matrix[pivot_index][pivot_index];
        for column in pivot_index..dimension {
            matrix[pivot_index][column] /= pivot;
        }
        rhs[pivot_index] /= pivot;

        for row_index in 0..dimension {
            if row_index == pivot_index {
                continue;
            }
            let factor = matrix[row_index][pivot_index];
            for column in pivot_index..dimension {
                matrix[row_index][column] -= factor * matrix[pivot_index][column];
            }
            rhs[row_index] -= factor * rhs[pivot_index];
        }
    }

    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(ContractError::InvalidGramianArithmetic);
    }
    Ok(rhs)
}

/// Canonical 2D symplectic-pairing defect for a manufactured basis fixture.
///
/// For J = [[0, 1], [-1, 0]] and basis columns q, p:
/// q^T J p = q[0] * p[1] - q[1] * p[0].
///
/// In two dimensions, the pair preserves the canonical symplectic form exactly
/// when this oriented area equals one. This is a small independent oracle, not
/// a general high-dimensional symplectic reduced-basis constructor.
pub fn canonical_symplectic_pair_defect_2d(q: [f64; 2], p: [f64; 2]) -> Result<f64, ContractError> {
    if q.iter().chain(p.iter()).any(|value| !value.is_finite()) {
        return Err(ContractError::NonFiniteSymplecticBasis);
    }
    let pairing = q[0].mul_add(p[1], -(q[1] * p[0]));
    let defect = (pairing - 1.0).abs();
    if !pairing.is_finite() || !defect.is_finite() {
        return Err(ContractError::NonFiniteSymplecticBasis);
    }
    Ok(defect)
}

/// Require a manufactured 2D canonical basis to preserve the symplectic form.
pub fn validate_canonical_symplectic_pair_2d(
    q: [f64; 2],
    p: [f64; 2],
    tolerance: f64,
) -> Result<f64, ContractError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ContractError::InvalidProjectionBasis);
    }
    let defect = canonical_symplectic_pair_defect_2d(q, p)?;
    if defect > tolerance {
        return Err(ContractError::InvalidSymplecticBasis);
    }
    Ok(defect)
}

/// Summary of a semantically checked KIRD checkpoint record.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KirdRecordSummary {
    pub record_id: String,
    pub record_status: String,
    pub checkpoint_bytes: u64,
    pub persistent_system_bytes: u64,
    pub model_asset_bytes: u64,
    pub decoder_asset_bytes: u64,
    pub amortization_rollouts: u64,
    pub amortized_asset_bytes_per_rollout: f64,
    pub amortized_persistent_bytes_per_rollout: f64,
    pub peak_working_set_bytes: u64,
    pub registered_probe_count: u64,
    pub confirmed_probe_count: u64,
    pub response_coverage_kind: String,
    pub globally_bounded: bool,
    pub covering_radius: Option<f64>,
    pub verified_lipschitz_constant: Option<f64>,
    pub uncertainty_budget: Option<f64>,
    pub global_normalized_error_bound: Option<f64>,
    pub physical_gate_status: String,
}

/// Validate KIRD cross-field semantics after JSON Schema validation.
///
/// This deliberately does not claim to verify external SHA-256 artifacts or
/// physical convergence. It checks digest syntax / non-placeholder identity,
/// accounting arithmetic, split policy, response-coverage consistency and
/// metric/status consistency. The caller must separately run the JSON Schema
/// validator and resolve each digest against immutable evidence.
pub fn validate_kird_record_json(input: &str) -> Result<KirdRecordSummary, ContractError> {
    let value: serde_json::Value =
        serde_json::from_str(input).map_err(|error| ContractError::JsonParse(error.to_string()))?;
    validate_kird_record_value(&value)
}

/// Validate an already-parsed KIRD JSON value.
pub fn validate_kird_record_value(
    value: &serde_json::Value,
) -> Result<KirdRecordSummary, ContractError> {
    let record_id = kird_string(value, "record_id")?;
    if !record_id.starts_with("KIRD-") {
        return Err(kird_error("record_id", "must use the KIRD- prefix"));
    }
    let record_status = kird_string(value, "record_status")?;
    if !matches!(
        record_status,
        "synthetic_fixture_only" | "physical_measurement"
    ) {
        return Err(kird_error("record_status", "unsupported record status"));
    }

    let split_role = kird_string(value, "task_contract.split_role")?;
    let acquisition_allowed = kird_bool(value, "task_contract.acquisition_allowed")?;
    if split_role == "sealed_test" && acquisition_allowed {
        return Err(kird_error(
            "task_contract.acquisition_allowed",
            "sealed-test records must prohibit new acquisition",
        ));
    }

    let physical_authorized = kird_bool(value, "provenance.physical_execution_authorized")?;
    let is_physical = record_status == "physical_measurement";
    if physical_authorized != is_physical {
        return Err(kird_error(
            "provenance.physical_execution_authorized",
            "must agree with record_status",
        ));
    }

    let digest_paths = [
        "cohort.initial_state_sha256",
        "task_contract.intervention_family_sha256",
        "task_contract.qoi_registry_sha256",
        "candidate.configuration_sha256",
        "provenance.candidate_output_sha256",
        "provenance.run_receipt_sha256",
    ];
    for digest_path in digest_paths {
        let digest = kird_string(value, digest_path)?;
        if !is_sha256_hex(digest) {
            return Err(kird_error(
                digest_path,
                "must be a lowercase SHA-256 hex digest",
            ));
        }
        if is_physical && digest.bytes().all(|byte| byte == b'0') {
            return Err(kird_error(
                digest_path,
                "physical measurements cannot use the all-zero placeholder digest",
            ));
        }
    }
    if is_physical {
        let reference_digest = kird_string(value, "provenance.reference_bundle_sha256")?;
        if !is_sha256_hex(reference_digest) || reference_digest.bytes().all(|byte| byte == b'0') {
            return Err(kird_error(
                "provenance.reference_bundle_sha256",
                "physical measurements require a non-placeholder reference digest",
            ));
        }
    }

    let byte_fields = [
        "storage.state_payload_bytes",
        "storage.basis_codec_metadata_bytes",
        "storage.coordinate_normalization_bytes",
        "storage.component_lineage_bytes",
        "storage.residual_memory_bytes",
        "storage.uncertainty_validity_bytes",
        "storage.provenance_contract_bytes",
    ];
    let mut checkpoint_bytes = 0_u64;
    for field in byte_fields {
        checkpoint_bytes = checkpoint_bytes
            .checked_add(kird_u64(value, field)?)
            .ok_or(ContractError::StorageByteCountOverflow)?;
    }
    let model_asset_bytes = kird_u64(value, "storage.model_asset_bytes")?;
    let decoder_asset_bytes = kird_u64(value, "storage.decoder_asset_bytes")?;
    let persistent_assets = model_asset_bytes
        .checked_add(decoder_asset_bytes)
        .ok_or(ContractError::StorageByteCountOverflow)?;
    let amortization_rollouts = kird_u64(value, "storage.amortization_rollouts")?;
    if amortization_rollouts == 0 {
        return Err(ContractError::ZeroAmortizationRollouts);
    }
    let measured_rollouts = kird_u64(value, "cost.measured_rollouts")?;
    if measured_rollouts == 0 {
        return Err(kird_error("cost.measured_rollouts", "must be positive"));
    }
    let _peak_working_set_bytes = kird_u64(value, "storage.peak_working_set_bytes")?;

    for field in [
        "cost.encode_seconds",
        "cost.decode_seconds",
        "cost.advance_seconds",
        "cost.observe_seconds",
        "cost.transition_seconds",
        "cost.fallback_seconds",
        "cost.training_seconds",
        "cost.energy_joules",
    ] {
        if kird_optional_number(value, field)?.is_some_and(|number| number < 0.0) {
            return Err(kird_error(field, "must be non-negative"));
        }
    }

    let registered_probe_count = kird_u64(value, "response_scope.registered_probe_count")?;
    let confirmed_probe_count = kird_u64(value, "response_scope.confirmed_probe_count")?;
    if confirmed_probe_count > registered_probe_count {
        return Err(kird_error(
            "response_scope.confirmed_probe_count",
            "cannot exceed registered probe count",
        ));
    }
    let response_coverage_kind = kird_string(value, "response_scope.coverage_kind")?;
    let globally_bounded = kird_bool(value, "response_scope.globally_bounded")?;
    match response_coverage_kind {
        "none" => {
            if registered_probe_count != 0 || confirmed_probe_count != 0 || globally_bounded {
                return Err(kird_error(
                    "response_scope",
                    "none coverage requires zero probes and no global-bound claim",
                ));
            }
        }
        "finite_empirical" => {
            if registered_probe_count == 0 || confirmed_probe_count == 0 || globally_bounded {
                return Err(kird_error(
                    "response_scope",
                    "finite empirical coverage requires confirmed probes and cannot claim a global bound",
                ));
            }
        }
        "certified_global" => {
            if !globally_bounded {
                return Err(kird_error(
                    "response_scope.globally_bounded",
                    "certified_global coverage requires a true global-bound flag",
                ));
            }
            validate_kird_global_coverage(value, registered_probe_count, confirmed_probe_count)?;
        }
        _ => {
            return Err(kird_error(
                "response_scope.coverage_kind",
                "unsupported response coverage kind",
            ));
        }
    }
    if globally_bounded && response_coverage_kind != "certified_global" {
        return Err(kird_error(
            "response_scope.coverage_kind",
            "a global bound cannot be inferred from finite empirical coverage",
        ));
    }

    for tier in [
        "distortion.microscopic",
        "distortion.weak_observable",
        "distortion.macroscopic_qoi",
        "distortion.sensitivity",
    ] {
        validate_kird_distortion_tier(value, tier, is_physical)?;
    }
    validate_kird_response_tier(value)?;

    let physical_gate_status = kird_string(value, "physical_gates.status")?;
    let gate_results = kird_array(value, "physical_gates.gate_results")?;
    let gate_statuses = gate_results
        .iter()
        .map(|gate| kird_string(gate, "status"))
        .collect::<Result<Vec<_>, _>>()?;
    match physical_gate_status {
        "PASS" => {
            if gate_statuses.is_empty() || gate_statuses.iter().any(|status| *status != "PASS") {
                return Err(kird_error(
                    "physical_gates.status",
                    "PASS requires at least one gate and every gate must PASS",
                ));
            }
        }
        "FAIL" => {
            if !gate_statuses.contains(&"FAIL") {
                return Err(kird_error(
                    "physical_gates.status",
                    "FAIL requires at least one failing physical gate",
                ));
            }
        }
        "UNKNOWN" | "UNSUPPORTED" => {}
        _ => {
            return Err(kird_error(
                "physical_gates.status",
                "unsupported gate disposition",
            ));
        }
    }
    if record_status == "synthetic_fixture_only" && physical_gate_status == "PASS" {
        return Err(kird_error(
            "physical_gates.status",
            "synthetic fixtures cannot pass physical qualification",
        ));
    }

    let persistent_system_bytes = checkpoint_bytes
        .checked_add(persistent_assets)
        .ok_or(ContractError::StorageByteCountOverflow)?;
    let amortized_asset_bytes_per_rollout = persistent_assets as f64 / amortization_rollouts as f64;
    let amortized_persistent_bytes_per_rollout =
        checkpoint_bytes as f64 + amortized_asset_bytes_per_rollout;
    if !amortized_asset_bytes_per_rollout.is_finite()
        || !amortized_persistent_bytes_per_rollout.is_finite()
    {
        return Err(ContractError::StorageByteCountOverflow);
    }

    let (
        covering_radius,
        verified_lipschitz_constant,
        uncertainty_budget,
        global_normalized_error_bound,
    ) = if response_coverage_kind == "certified_global" {
        let uncertainty_budget = kird_number(value, "response_scope.reference_uncertainty")?
            + kird_number(value, "response_scope.candidate_uncertainty")?
            + kird_number(value, "response_scope.measurement_uncertainty")?;
        (
            Some(kird_number(value, "response_scope.covering_radius")?),
            Some(kird_number(
                value,
                "response_scope.verified_lipschitz_constant",
            )?),
            Some(uncertainty_budget),
            Some(kird_number(
                value,
                "response_scope.global_normalized_error_bound",
            )?),
        )
    } else {
        (None, None, None, None)
    };

    Ok(KirdRecordSummary {
        record_id: record_id.to_owned(),
        record_status: record_status.to_owned(),
        checkpoint_bytes,
        persistent_system_bytes,
        model_asset_bytes,
        decoder_asset_bytes,
        amortization_rollouts,
        amortized_asset_bytes_per_rollout,
        amortized_persistent_bytes_per_rollout,
        peak_working_set_bytes: _peak_working_set_bytes,
        registered_probe_count,
        confirmed_probe_count,
        response_coverage_kind: response_coverage_kind.to_owned(),
        globally_bounded,
        covering_radius,
        verified_lipschitz_constant,
        uncertainty_budget,
        global_normalized_error_bound,
        physical_gate_status: physical_gate_status.to_owned(),
    })
}

fn validate_kird_global_coverage(
    root: &serde_json::Value,
    registered_probe_count: u64,
    confirmed_probe_count: u64,
) -> Result<(), ContractError> {
    if registered_probe_count == 0 || confirmed_probe_count == 0 {
        return Err(kird_error(
            "response_scope",
            "certified_global coverage requires registered and confirmed probes",
        ));
    }
    if confirmed_probe_count != registered_probe_count {
        return Err(kird_error(
            "response_scope.confirmed_probe_count",
            "a global cover cannot contain unconfirmed or unresolved registered probes",
        ));
    }

    let metric_id = kird_string(root, "response_scope.coverage_metric_id")?;
    let method = kird_string(root, "response_scope.bound_method")?;
    let checker = kird_string(root, "response_scope.independent_checker_revision")?;
    if metric_id.trim().is_empty() || method.trim().is_empty() || checker.trim().is_empty() {
        return Err(kird_error(
            "response_scope",
            "global certification requires metric, method and checker identities",
        ));
    }

    let covering_radius = kird_number(root, "response_scope.covering_radius")?;
    let lipschitz_constant = kird_number(root, "response_scope.verified_lipschitz_constant")?;
    let reference_uncertainty = kird_number(root, "response_scope.reference_uncertainty")?;
    let candidate_uncertainty = kird_number(root, "response_scope.candidate_uncertainty")?;
    let measurement_uncertainty = kird_number(root, "response_scope.measurement_uncertainty")?;
    let maximum_sampled_error =
        kird_number(root, "response_scope.maximum_sampled_normalized_error")?;
    let global_bound = kird_number(root, "response_scope.global_normalized_error_bound")?;

    if [
        covering_radius,
        lipschitz_constant,
        reference_uncertainty,
        candidate_uncertainty,
        measurement_uncertainty,
        maximum_sampled_error,
        global_bound,
    ]
    .iter()
    .any(|value| *value < 0.0)
    {
        return Err(kird_error(
            "response_scope",
            "cover radius, Lipschitz constant, uncertainty and error bounds must be non-negative",
        ));
    }

    if !kird_bool(root, "response_scope.branch_association_stable")? {
        return Err(kird_error(
            "response_scope.branch_association_stable",
            "global certification cannot cross an unresolved response-branch change",
        ));
    }

    for digest_path in [
        "response_scope.intervention_domain_sha256",
        "response_scope.metric_definition_sha256",
        "response_scope.compactness_evidence_sha256",
        "response_scope.covering_set_sha256",
        "response_scope.cover_radius_evidence_sha256",
        "response_scope.bound_evidence_sha256",
        "response_scope.branch_stability_evidence_sha256",
    ] {
        let digest = kird_string(root, digest_path)?;
        if !is_sha256_hex(digest) || digest.bytes().all(|byte| byte == b'0') {
            return Err(kird_error(
                digest_path,
                "global certification requires non-placeholder proof digests",
            ));
        }
    }

    let uncertainty = reference_uncertainty + candidate_uncertainty + measurement_uncertainty;
    let lipschitz_cover_term = lipschitz_constant * covering_radius;
    let required_bound = maximum_sampled_error + lipschitz_cover_term + uncertainty;
    if !uncertainty.is_finite()
        || !lipschitz_cover_term.is_finite()
        || !required_bound.is_finite()
        || !global_bound.is_finite()
    {
        return Err(kird_error(
            "response_scope.global_normalized_error_bound",
            "certificate arithmetic must remain finite",
        ));
    }
    let allowance = 1.0e-12_f64.max(required_bound.abs() * 1.0e-10);
    if global_bound + allowance < required_bound {
        return Err(kird_error(
            "response_scope.global_normalized_error_bound",
            "must conservatively cover sampled error + Lipschitz cover term + uncertainty",
        ));
    }

    if kird_optional_number(
        root,
        "distortion.intervention_response.worst_confirmed_normalized_error",
    )?
    .is_some_and(|confirmed_error| maximum_sampled_error + allowance < confirmed_error)
    {
        return Err(kird_error(
            "response_scope.maximum_sampled_normalized_error",
            "cannot be smaller than the worst confirmed response error",
        ));
    }

    Ok(())
}

fn validate_kird_distortion_tier(
    root: &serde_json::Value,
    field: &'static str,
    is_physical: bool,
) -> Result<(), ContractError> {
    let status = kird_string(root, kird_join_field(field, "status"))?;
    if !matches!(status, "PASS" | "FAIL" | "UNKNOWN" | "UNSUPPORTED") {
        return Err(kird_error(field, "unsupported distortion status"));
    }
    if status == "PASS" || status == "FAIL" {
        let metric_id = kird_string(root, kird_join_field(field, "metric_id"))?;
        if metric_id.trim().is_empty() {
            return Err(kird_error(field, "measured distortion requires metric_id"));
        }
        let observed = kird_number(root, kird_join_field(field, "observed_error"))?;
        let tolerance = kird_number(root, kird_join_field(field, "tolerance"))?;
        let normalized = kird_number(root, kird_join_field(field, "normalized_error"))?;
        if observed < 0.0 || tolerance <= 0.0 || normalized < 0.0 {
            return Err(kird_error(field, "invalid error or tolerance range"));
        }
        let expected = observed / tolerance;
        let allowance = 1.0e-9_f64.max(expected.abs() * 1.0e-8);
        if (normalized - expected).abs() > allowance {
            return Err(kird_error(
                field,
                "normalized_error must equal observed_error / tolerance",
            ));
        }
        if status == "PASS" && normalized > 1.0 {
            return Err(kird_error(field, "PASS requires normalized_error <= 1"));
        }
        let digest_path = kird_join_field(field, "evidence_sha256");
        let digest = kird_string(root, digest_path)?;
        if !is_sha256_hex(digest) || (is_physical && digest.bytes().all(|byte| byte == b'0')) {
            return Err(kird_error(
                digest_path,
                "requires a valid, non-placeholder SHA-256 digest for physical evidence",
            ));
        }
    }
    Ok(())
}

fn validate_kird_response_tier(root: &serde_json::Value) -> Result<(), ContractError> {
    let status = kird_string(root, "distortion.intervention_response.status")?;
    if !matches!(status, "PASS" | "FAIL" | "UNKNOWN" | "UNSUPPORTED") {
        return Err(kird_error(
            "distortion.intervention_response.status",
            "unsupported response status",
        ));
    }
    let missing = kird_u64(
        root,
        "distortion.intervention_response.missing_branch_count",
    )?;
    let misassociated = kird_u64(
        root,
        "distortion.intervention_response.misassociated_branch_count",
    )?;
    if status == "PASS" || status == "FAIL" {
        let registered = kird_u64(root, "response_scope.registered_probe_count")?;
        let confirmed = kird_u64(root, "response_scope.confirmed_probe_count")?;
        if registered == 0 || confirmed == 0 {
            return Err(kird_error(
                "distortion.intervention_response",
                "measured response dispositions require confirmed intervention probes",
            ));
        }
    }
    if status == "PASS" {
        let normalized = kird_number(
            root,
            "distortion.intervention_response.worst_confirmed_normalized_error",
        )?;
        let evidence = kird_string(root, "distortion.intervention_response.evidence_sha256")?;
        let coverage_kind = kird_string(root, "response_scope.coverage_kind")?;
        let global_bound = if coverage_kind == "certified_global" {
            Some(kird_number(
                root,
                "response_scope.global_normalized_error_bound",
            )?)
        } else {
            None
        };
        if normalized > 1.0
            || global_bound.is_some_and(|bound| bound > 1.0)
            || missing != 0
            || misassociated != 0
            || !is_sha256_hex(evidence)
            || (kird_string(root, "record_status")? == "physical_measurement"
                && evidence.bytes().all(|byte| byte == b'0'))
        {
            return Err(kird_error(
                "distortion.intervention_response",
                "PASS requires empirical/global error <= 1, zero missing/misassociated branches and non-placeholder evidence",
            ));
        }
    }
    Ok(())
}

fn kird_error(field: &'static str, reason: &'static str) -> ContractError {
    ContractError::InvalidKirdField { field, reason }
}

fn kird_join_field(parent: &'static str, child: &'static str) -> &'static str {
    match (parent, child) {
        ("distortion.microscopic", "status") => "distortion.microscopic.status",
        ("distortion.weak_observable", "status") => "distortion.weak_observable.status",
        ("distortion.macroscopic_qoi", "status") => "distortion.macroscopic_qoi.status",
        ("distortion.sensitivity", "status") => "distortion.sensitivity.status",
        ("distortion.microscopic", "metric_id") => "distortion.microscopic.metric_id",
        ("distortion.microscopic", "observed_error") => "distortion.microscopic.observed_error",
        ("distortion.microscopic", "tolerance") => "distortion.microscopic.tolerance",
        ("distortion.microscopic", "normalized_error") => "distortion.microscopic.normalized_error",
        ("distortion.microscopic", "evidence_sha256") => "distortion.microscopic.evidence_sha256",
        ("distortion.weak_observable", "metric_id") => "distortion.weak_observable.metric_id",
        ("distortion.weak_observable", "observed_error") => {
            "distortion.weak_observable.observed_error"
        }
        ("distortion.weak_observable", "tolerance") => "distortion.weak_observable.tolerance",
        ("distortion.weak_observable", "normalized_error") => {
            "distortion.weak_observable.normalized_error"
        }
        ("distortion.weak_observable", "evidence_sha256") => {
            "distortion.weak_observable.evidence_sha256"
        }
        ("distortion.macroscopic_qoi", "metric_id") => "distortion.macroscopic_qoi.metric_id",
        ("distortion.macroscopic_qoi", "observed_error") => {
            "distortion.macroscopic_qoi.observed_error"
        }
        ("distortion.macroscopic_qoi", "tolerance") => "distortion.macroscopic_qoi.tolerance",
        ("distortion.macroscopic_qoi", "normalized_error") => {
            "distortion.macroscopic_qoi.normalized_error"
        }
        ("distortion.macroscopic_qoi", "evidence_sha256") => {
            "distortion.macroscopic_qoi.evidence_sha256"
        }
        ("distortion.sensitivity", "metric_id") => "distortion.sensitivity.metric_id",
        ("distortion.sensitivity", "observed_error") => "distortion.sensitivity.observed_error",
        ("distortion.sensitivity", "tolerance") => "distortion.sensitivity.tolerance",
        ("distortion.sensitivity", "normalized_error") => "distortion.sensitivity.normalized_error",
        ("distortion.sensitivity", "evidence_sha256") => "distortion.sensitivity.evidence_sha256",
        _ => "distortion",
    }
}

fn kird_value<'a>(
    root: &'a serde_json::Value,
    field: &'static str,
) -> Result<&'a serde_json::Value, ContractError> {
    let mut current = root;
    for segment in field.split('.') {
        current = current
            .get(segment)
            .ok_or_else(|| kird_error(field, "required field is missing"))?;
    }
    Ok(current)
}

fn kird_string<'a>(
    root: &'a serde_json::Value,
    field: &'static str,
) -> Result<&'a str, ContractError> {
    kird_value(root, field)?
        .as_str()
        .ok_or_else(|| kird_error(field, "must be a string"))
}

fn kird_bool(root: &serde_json::Value, field: &'static str) -> Result<bool, ContractError> {
    kird_value(root, field)?
        .as_bool()
        .ok_or_else(|| kird_error(field, "must be a boolean"))
}

fn kird_u64(root: &serde_json::Value, field: &'static str) -> Result<u64, ContractError> {
    kird_value(root, field)?
        .as_u64()
        .ok_or_else(|| kird_error(field, "must be a non-negative integer"))
}

fn kird_number(root: &serde_json::Value, field: &'static str) -> Result<f64, ContractError> {
    let number = kird_value(root, field)?
        .as_f64()
        .ok_or_else(|| kird_error(field, "must be a finite number"))?;
    if !number.is_finite() {
        return Err(kird_error(field, "must be finite"));
    }
    Ok(number)
}

fn kird_optional_number(
    root: &serde_json::Value,
    field: &'static str,
) -> Result<Option<f64>, ContractError> {
    let mut current = root;
    for segment in field.split('.') {
        match current.get(segment) {
            Some(value) => current = value,
            None => return Ok(None),
        }
    }
    let number = current
        .as_f64()
        .ok_or_else(|| kird_error(field, "must be a finite number"))?;
    if !number.is_finite() {
        return Err(kird_error(field, "must be finite"));
    }
    Ok(Some(number))
}

fn kird_array<'a>(
    root: &'a serde_json::Value,
    field: &'static str,
) -> Result<&'a [serde_json::Value], ContractError> {
    kird_value(root, field)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| kird_error(field, "must be an array"))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Checked-in synthetic KIRD fixture used by the semantic auditor tests.
pub const KIRD_SYNTHETIC_FIXTURE: &str = include_str!(
    "../../../../benchmarks/plasma/rate-distortion/fixtures/synthetic-kird-point-v0.1.json"
);
pub const KIRD_SYNTHETIC_CERTIFIED_FIXTURE: &str = include_str!(
    "../../../../benchmarks/plasma/rate-distortion/fixtures/synthetic-kird-certified-cover-v0.1.json"
);

/// Checked-in KIRD JSON Schema. Consumers should still run a Draft 2020-12
/// validator; this constant makes the exact source available to integration tests.
pub const KIRD_POINT_SCHEMA: &str =
    include_str!("../../../../benchmarks/plasma/rate-distortion/kird-point-v0.1.schema.json");

#[cfg(test)]
mod tests {
    use super::*;

    const SYNTHETIC_FIXTURE: &str = include_str!(
        "../../../../benchmarks/plasma/adversarial-probes/fixtures/synthetic-pulse-family-v0.1.json"
    );

    fn fixture_value() -> serde_json::Value {
        serde_json::from_str(SYNTHETIC_FIXTURE).expect("checked-in fixture must be valid JSON")
    }

    fn certified_kird_value() -> serde_json::Value {
        serde_json::from_str(KIRD_SYNTHETIC_CERTIFIED_FIXTURE)
            .expect("synthetic certified cover fixture must be valid JSON")
    }

    fn parse_value(value: &serde_json::Value) -> Result<PulseFamilyContract, ContractError> {
        let input = serde_json::to_string(value).expect("JSON value must serialize");
        PulseFamilyContract::parse_json(&input)
    }

    fn contract() -> PulseFamilyContract {
        PulseFamilyContract::parse_json(SYNTHETIC_FIXTURE).expect("synthetic fixture must parse")
    }

    fn candidate(
        spatial_mode: u32,
        phase_rad: f64,
        kick_velocity_amplitude: f64,
        application_time: f64,
    ) -> PulseCandidate {
        PulseCandidate {
            spatial_mode,
            phase_rad,
            kick_velocity_amplitude,
            application_time,
        }
    }

    #[test]
    fn factorized_gramian_matches_explicit_small_matrix() {
        let mut gramian = EmpiricalGramian::new(2).unwrap();
        gramian.add_direction(&[1.0, 2.0], 2.0).unwrap();
        gramian.add_direction(&[0.0, 1.0], 0.5).unwrap();

        // W = 2*[1,2][1,2]^T + 0.5*[0,1][0,1]^T
        //   = [[2,4],[4,8.5]]
        let applied = gramian.apply(&[3.0, 4.0]).unwrap();
        assert!((applied[0] - 22.0).abs() < 1.0e-12);
        assert!((applied[1] - 46.0).abs() < 1.0e-12);
        assert!((gramian.quadratic_form(&[3.0, 4.0]).unwrap() - 250.0).abs() < 1.0e-12);
        assert!((gramian.trace().unwrap() - 10.5).abs() < 1.0e-12);
        assert_eq!(gramian.factor_count(), 2);
        assert_eq!(gramian.stored_scalar_count(), 4);
    }

    #[test]
    fn factorized_gramian_rejects_invalid_weights_and_dimensions() {
        let mut gramian = EmpiricalGramian::new(2).unwrap();
        assert_eq!(
            gramian.add_direction(&[1.0, 0.0], -1.0),
            Err(ContractError::InvalidGramianWeight)
        );
        assert_eq!(
            gramian.add_direction(&[1.0], 1.0),
            Err(ContractError::GramianDimensionMismatch {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            gramian.add_direction(&[f64::NAN, 0.0], 1.0),
            Err(ContractError::InvalidGramianArithmetic)
        );
        assert_eq!(
            EmpiricalGramian::new(0).unwrap_err(),
            ContractError::InvalidGramianDimension
        );
    }

    #[test]
    fn zero_weight_is_a_no_op() {
        let mut gramian = EmpiricalGramian::new(2).unwrap();
        gramian.add_direction(&[1.0, 2.0], 0.0).unwrap();
        assert_eq!(gramian.factor_count(), 0);
        assert_eq!(gramian.trace().unwrap(), 0.0);
    }

    #[test]
    fn orthonormal_basis_captures_expected_gramian_trace() {
        let mut gramian = EmpiricalGramian::new(2).unwrap();
        gramian.add_direction(&[1.0, 2.0], 2.0).unwrap();
        gramian.add_direction(&[0.0, 1.0], 0.5).unwrap();

        let first_axis = vec![1.0, 0.0];
        let second_axis = vec![0.0, 1.0];
        let captured = gramian
            .captured_trace_fraction(&[first_axis, second_axis])
            .unwrap();
        assert!((captured - 1.0).abs() < 1.0e-12);

        let first_only = gramian.captured_trace_fraction(&[vec![1.0, 0.0]]).unwrap();
        assert!((first_only - (2.0 / 10.5)).abs() < 1.0e-12);
    }

    #[test]
    fn gramian_rejects_non_orthonormal_projection_basis() {
        let mut gramian = EmpiricalGramian::new(2).unwrap();
        gramian.add_direction(&[1.0, 0.0], 1.0).unwrap();
        assert_eq!(
            gramian.captured_trace_fraction(&[vec![1.0, 0.0], vec![1.0, 0.0]]),
            Err(ContractError::InvalidProjectionBasis)
        );
    }

    #[test]
    fn response_matrix_preserves_mode_identity_and_missing_entries() {
        let mut matrix = ModeResponseMatrix::new(vec![1, 3], vec![1, 2]).unwrap();
        assert_eq!(matrix.measured_entry_count(), 0);
        assert!(!matrix.is_complete());
        matrix
            .set(3, 2, ComplexResponse::new(0.5, -0.25).unwrap())
            .unwrap();
        assert_eq!(
            matrix.get(3, 2).unwrap(),
            Some(ComplexResponse {
                real: 0.5,
                imag: -0.25
            })
        );
        assert_eq!(matrix.get(1, 2).unwrap(), None);
        assert_eq!(matrix.get(7, 2), Err(ContractError::UnknownResponseMode(7)));
    }

    #[test]
    fn response_matrix_rejects_empty_and_duplicate_mode_registries() {
        assert_eq!(
            ModeResponseMatrix::new(Vec::new(), vec![1]),
            Err(ContractError::EmptyResponseModeRegistry)
        );
        assert_eq!(
            ModeResponseMatrix::new(vec![1, 1], vec![1]),
            Err(ContractError::DuplicateResponseMode)
        );
        assert_eq!(
            ModeResponseMatrix::new(vec![1], vec![2, 2]),
            Err(ContractError::DuplicateResponseMode)
        );
    }

    #[test]
    fn response_matrix_requires_finite_measurements() {
        let mut matrix = ModeResponseMatrix::new(vec![1], vec![2]).unwrap();
        assert_eq!(
            matrix.set(
                1,
                2,
                ComplexResponse {
                    real: f64::NAN,
                    imag: 0.0
                }
            ),
            Err(ContractError::NonFiniteResponseValue)
        );
    }

    #[test]
    fn response_matrix_relative_error_is_complex_and_axis_qualified() {
        let mut reference = ModeResponseMatrix::new(vec![1, 2], vec![1, 2]).unwrap();
        let mut candidate = ModeResponseMatrix::new(vec![1, 2], vec![1, 2]).unwrap();
        for output in [1, 2] {
            for input in [1, 2] {
                let value = ComplexResponse::new((input + output) as f64, 0.5).unwrap();
                reference.set(input, output, value).unwrap();
                candidate.set(input, output, value).unwrap();
            }
        }
        assert_eq!(candidate.relative_frobenius_error(&reference).unwrap(), 0.0);
        let mut changed = candidate.clone();
        changed
            .set(1, 1, ComplexResponse::new(2.0, 1.0).unwrap())
            .unwrap();
        assert!(changed.relative_frobenius_error(&reference).unwrap() > 0.0);
        let wrong_axes = ModeResponseMatrix::new(vec![2, 1], vec![1, 2]).unwrap();
        assert_eq!(
            candidate.relative_frobenius_error(&wrong_axes),
            Err(ContractError::ResponseMatrixAxisMismatch)
        );
    }

    #[test]
    fn response_matrix_relative_error_rejects_missing_entries_and_zero_reference() {
        let mut reference = ModeResponseMatrix::new(vec![1], vec![1]).unwrap();
        let candidate = ModeResponseMatrix::new(vec![1], vec![1]).unwrap();
        reference
            .set(1, 1, ComplexResponse::new(1.0, 0.0).unwrap())
            .unwrap();
        assert_eq!(
            candidate.relative_frobenius_error(&reference),
            Err(ContractError::MissingResponseEntry)
        );

        let mut zero_reference = ModeResponseMatrix::new(vec![1], vec![1]).unwrap();
        zero_reference
            .set(1, 1, ComplexResponse::default())
            .unwrap();
        assert_eq!(
            zero_reference.relative_frobenius_error(&zero_reference),
            Err(ContractError::ZeroResponseReferenceNorm)
        );
    }

    #[test]
    fn signed_response_coefficients_recover_linear_and_quadratic_complex_terms() {
        let amplitude = 0.1;
        let c0 = ComplexResponse::new(2.0, -1.0).unwrap();
        let c1 = ComplexResponse::new(3.0, 0.5).unwrap();
        let c2 = ComplexResponse::new(-2.0, 1.5).unwrap();
        let positive = c0
            .add(c1.scale(amplitude).unwrap())
            .unwrap()
            .add(c2.scale(amplitude * amplitude).unwrap())
            .unwrap();
        let negative = c0
            .sub(c1.scale(amplitude).unwrap())
            .unwrap()
            .add(c2.scale(amplitude * amplitude).unwrap())
            .unwrap();
        let (estimated_c1, estimated_c2) =
            signed_response_coefficients(positive, negative, c0, amplitude).unwrap();
        assert!((estimated_c1.real - c1.real).abs() < 1.0e-12);
        assert!((estimated_c1.imag - c1.imag).abs() < 1.0e-12);
        assert!((estimated_c2.real - c2.real).abs() < 1.0e-12);
        assert!((estimated_c2.imag - c2.imag).abs() < 1.0e-12);
    }

    #[test]
    fn mixed_response_coefficient_recovers_complex_cross_term() {
        let base = ComplexResponse::new(0.2, -0.1).unwrap();
        let amplitude_a = 0.1;
        let amplitude_b = 0.2;
        let cross = ComplexResponse::new(2.0, -3.0).unwrap();
        let response = |sign_a: f64, sign_b: f64| {
            base.add(
                cross
                    .scale(sign_a * sign_b * amplitude_a * amplitude_b)
                    .unwrap(),
            )
            .unwrap()
        };
        let estimate = mixed_response_coefficient(
            response(1.0, 1.0),
            response(1.0, -1.0),
            response(-1.0, 1.0),
            response(-1.0, -1.0),
            amplitude_a,
            amplitude_b,
        )
        .unwrap();
        assert!((estimate.real - cross.real).abs() < 1.0e-12);
        assert!((estimate.imag - cross.imag).abs() < 1.0e-12);
    }

    #[test]
    fn response_estimators_reject_zero_non_finite_and_overflowed_amplitudes() {
        let value = ComplexResponse::new(1.0, 0.0).unwrap();
        assert_eq!(
            signed_response_coefficients(value, value, value, 0.0),
            Err(ContractError::InvalidResponseAmplitude)
        );
        assert_eq!(
            mixed_response_coefficient(value, value, value, value, 0.1, f64::INFINITY),
            Err(ContractError::InvalidResponseAmplitude)
        );
        assert_eq!(
            mixed_response_coefficient(value, value, value, value, f64::MAX, f64::MAX),
            Err(ContractError::InvalidResponseAmplitude)
        );
    }

    #[test]
    fn wrapped_phase_difference_handles_branch_cut_and_zero_magnitude() {
        let candidate = ComplexResponse::new(-1.0, -1.0e-9).unwrap();
        let reference = ComplexResponse::new(-1.0, 1.0e-9).unwrap();
        let error = wrapped_phase_difference(candidate, reference).unwrap();
        assert!(error.abs() < 1.0e-6);
        assert_eq!(
            wrapped_phase_difference(ComplexResponse::default(), reference),
            None
        );
    }

    #[test]
    fn storage_ledger_counts_payload_and_required_metadata() {
        let state = StateStorageBreakdown {
            state_payload_bytes: 1_000,
            basis_or_codec_metadata_bytes: 100,
            coordinate_and_normalization_bytes: 40,
            component_and_lineage_bytes: 20,
            residual_memory_bytes: 200,
            uncertainty_and_validity_bytes: 30,
            provenance_and_contract_bytes: 50,
        };
        assert_eq!(state.total_state_bytes().unwrap(), 1_440);
    }

    #[test]
    fn storage_ledger_rejects_byte_count_overflow() {
        let state = StateStorageBreakdown {
            state_payload_bytes: u64::MAX,
            basis_or_codec_metadata_bytes: 1,
            ..StateStorageBreakdown::default()
        };
        assert_eq!(
            state.total_state_bytes(),
            Err(ContractError::StorageByteCountOverflow)
        );
    }

    #[test]
    fn storage_accounting_separates_checkpoint_assets_and_working_set() {
        let accounting = KineticStorageAccounting {
            checkpoint: StateStorageBreakdown {
                state_payload_bytes: 1_000,
                basis_or_codec_metadata_bytes: 100,
                ..StateStorageBreakdown::default()
            },
            model_asset_bytes: 8_000,
            decoder_asset_bytes: 2_000,
            peak_working_set_bytes: 50_000,
        };
        assert_eq!(accounting.checkpoint_bytes().unwrap(), 1_100);
        assert_eq!(accounting.persistent_asset_bytes().unwrap(), 10_000);
        assert_eq!(
            accounting.amortized_asset_bytes_per_rollout(10).unwrap(),
            1_000.0
        );
        assert_eq!(
            accounting
                .amortized_persistent_bytes_per_rollout(10)
                .unwrap(),
            2_100.0
        );
        assert_eq!(accounting.peak_working_set_bytes, 50_000);
    }

    #[test]
    fn storage_accounting_rejects_zero_rollout_amortization() {
        let accounting = KineticStorageAccounting::default();
        assert_eq!(
            accounting.amortized_asset_bytes_per_rollout(0),
            Err(ContractError::ZeroAmortizationRollouts)
        );
    }

    #[test]
    fn storage_asset_sum_rejects_overflow() {
        let accounting = KineticStorageAccounting {
            model_asset_bytes: u64::MAX,
            decoder_asset_bytes: 1,
            ..KineticStorageAccounting::default()
        };
        assert_eq!(
            accounting.persistent_asset_bytes(),
            Err(ContractError::StorageByteCountOverflow)
        );
    }

    #[test]
    fn minmax_normalization_exposes_global_inlier_drift_when_sample_support_expands() {
        let inliers = [0.1, 0.3, 0.5, 0.7, 0.9];
        let augmented = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9];
        let before = MinMaxCoordinateMap::fit(&inliers, -PI, PI).expect("base sample has support");
        let after =
            MinMaxCoordinateMap::fit(&augmented, -PI, PI).expect("augmented sample has support");

        let before_coordinates = inliers
            .iter()
            .map(|position| before.apply(*position).unwrap())
            .collect::<Vec<_>>();
        let after_coordinates = inliers
            .iter()
            .map(|position| after.apply(*position).unwrap())
            .collect::<Vec<_>>();
        let displacements = before_coordinates
            .iter()
            .zip(&after_coordinates)
            .map(|(left, right)| (right - left).abs())
            .collect::<Vec<_>>();
        let maximum = displacements.iter().copied().fold(0.0_f64, f64::max);
        let mean = displacements.iter().sum::<f64>() / displacements.len() as f64;

        assert!((maximum - 0.698_131_700_797_731_9).abs() < 1.0e-12);
        assert!((mean - 0.349_065_850_398_865_84).abs() < 1.0e-12);
        assert_eq!(before.input_support(), (0.1, 0.9));
        assert_eq!(after.input_support(), (0.0, 0.9));
    }

    #[test]
    fn minmax_normalization_rejects_empty_degenerate_and_non_finite_samples() {
        assert_eq!(
            MinMaxCoordinateMap::fit(&[], -1.0, 1.0),
            Err(ContractError::EmptyNormalizationSample)
        );
        assert_eq!(
            MinMaxCoordinateMap::fit(&[0.5, 0.5], -1.0, 1.0),
            Err(ContractError::InvalidNormalizationSupport)
        );
        assert_eq!(
            MinMaxCoordinateMap::fit(&[0.0, f64::NAN], -1.0, 1.0),
            Err(ContractError::NonFiniteNormalizationCoordinate)
        );
        assert_eq!(
            MinMaxCoordinateMap::fit(&[0.0, 1.0], 1.0, 1.0),
            Err(ContractError::InvalidNormalizationOutputInterval)
        );
    }

    #[test]
    fn minmax_normalization_reports_non_finite_application() {
        let map = MinMaxCoordinateMap::new(0.0, 1.0, -1.0, 1.0).unwrap();
        assert_eq!(
            map.apply(f64::INFINITY),
            Err(ContractError::NonFiniteNormalizationCoordinate)
        );
    }

    #[test]
    fn kird_synthetic_fixture_passes_semantic_audit_and_reports_all_bytes() {
        let report = validate_kird_record_json(KIRD_SYNTHETIC_FIXTURE).unwrap();
        assert_eq!(report.record_id, "KIRD-SYNTHETIC-WIRE-FIXTURE-001");
        assert_eq!(report.checkpoint_bytes, 1_504);
        assert_eq!(report.persistent_system_bytes, 6_112);
        assert_eq!(report.model_asset_bytes, 4_096);
        assert_eq!(report.decoder_asset_bytes, 512);
        assert_eq!(report.amortization_rollouts, 100);
        assert_eq!(report.amortized_asset_bytes_per_rollout, 46.08);
        assert_eq!(report.amortized_persistent_bytes_per_rollout, 1_550.08);
        assert_eq!(report.peak_working_set_bytes, 16_384);
        assert_eq!(report.response_coverage_kind, "none");
        assert!(!report.globally_bounded);
        assert_eq!(report.physical_gate_status, "UNKNOWN");
    }

    #[test]
    fn kird_schema_and_fixture_are_valid_json_documents() {
        let schema: serde_json::Value = serde_json::from_str(KIRD_POINT_SCHEMA).unwrap();
        let fixture: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert_eq!(fixture["record_status"], "synthetic_fixture_only");
    }

    #[test]
    fn kird_audit_rejects_sealed_test_acquisition() {
        let mut value: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        value["task_contract"]["split_role"] = serde_json::json!("sealed_test");
        value["task_contract"]["acquisition_allowed"] = serde_json::json!(true);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "task_contract.acquisition_allowed",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_synthetic_physical_authorization() {
        let mut value: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        value["provenance"]["physical_execution_authorized"] = serde_json::json!(true);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "provenance.physical_execution_authorized",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_checkpoint_byte_overflow() {
        let mut value: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        value["storage"]["state_payload_bytes"] = serde_json::json!(u64::MAX);
        value["storage"]["basis_codec_metadata_bytes"] = serde_json::json!(1);
        assert_eq!(
            validate_kird_record_value(&value),
            Err(ContractError::StorageByteCountOverflow)
        );
    }

    #[test]
    fn kird_audit_accepts_manufactured_lipschitz_cover_certificate() {
        let value = certified_kird_value();
        let summary = validate_kird_record_value(&value).unwrap();
        assert_eq!(summary.response_coverage_kind, "certified_global");
        assert!(summary.globally_bounded);
        assert_eq!(summary.covering_radius, Some(0.1));
        assert_eq!(summary.verified_lipschitz_constant, Some(2.0));
        assert_eq!(summary.uncertainty_budget, Some(0.05));
        assert_eq!(summary.global_normalized_error_bound, Some(0.75));
    }

    #[test]
    fn kird_audit_rejects_global_cover_with_unconfirmed_registered_probes() {
        let mut value = certified_kird_value();
        value["response_scope"]["registered_probe_count"] = serde_json::json!(2);
        value["response_scope"]["confirmed_probe_count"] = serde_json::json!(1);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.confirmed_probe_count",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_global_cover_without_domain_or_fill_distance_proof() {
        let mut value = certified_kird_value();
        value["response_scope"]["cover_radius_evidence_sha256"] =
            serde_json::json!("0000000000000000000000000000000000000000000000000000000000000000");
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.cover_radius_evidence_sha256",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_understated_lipschitz_cover_bound() {
        let mut value = certified_kird_value();
        value["response_scope"]["global_normalized_error_bound"] = serde_json::json!(0.74);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.global_normalized_error_bound",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_unstable_branch_association_in_global_certificate() {
        let mut value = certified_kird_value();
        value["response_scope"]["branch_association_stable"] = serde_json::json!(false);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.branch_association_stable",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_global_cover_that_understates_sampled_error() {
        let mut value = certified_kird_value();
        value["response_scope"]["maximum_sampled_normalized_error"] = serde_json::json!(0.49);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.maximum_sampled_normalized_error",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_global_certificate_without_branch_proof_digest() {
        let mut value = certified_kird_value();
        value["response_scope"]["branch_stability_evidence_sha256"] =
            serde_json::json!("0000000000000000000000000000000000000000000000000000000000000000");
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.branch_stability_evidence_sha256",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_response_pass_when_global_bound_exceeds_tolerance() {
        let mut value = certified_kird_value();
        value["response_scope"]["global_normalized_error_bound"] = serde_json::json!(1.01);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "distortion.intervention_response",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_finite_probe_set_claimed_as_global_bound() {
        let mut value: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        value["response_scope"]["coverage_kind"] = serde_json::json!("finite_empirical");
        value["response_scope"]["registered_probe_count"] = serde_json::json!(20);
        value["response_scope"]["confirmed_probe_count"] = serde_json::json!(20);
        value["response_scope"]["globally_bounded"] = serde_json::json!(true);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_inconsistent_probe_counts() {
        let mut value: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        value["response_scope"]["registered_probe_count"] = serde_json::json!(2);
        value["response_scope"]["confirmed_probe_count"] = serde_json::json!(3);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.confirmed_probe_count",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_false_distortion_pass() {
        let mut value: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        value["distortion"]["macroscopic_qoi"] = serde_json::json!({
            "status":"PASS",
            "metric_id":"qoi-relative-l2",
            "observed_error":0.2,
            "tolerance":0.1,
            "normalized_error":2.0,
            "evidence_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        });
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "distortion.macroscopic_qoi",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_synthetic_physical_gate_pass() {
        let mut value: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        value["physical_gates"] = serde_json::json!({
            "status":"PASS",
            "gate_results":[{"gate_id":"energy","status":"PASS"}]
        });
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "physical_gates.status",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_placeholder_digest_for_physical_record() {
        let mut value: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        value["record_status"] = serde_json::json!("physical_measurement");
        value["provenance"]["physical_execution_authorized"] = serde_json::json!(true);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "cohort.initial_state_sha256",
                ..
            })
        ));
    }

    #[test]
    fn periodic_torus_lift_is_continuous_at_the_coordinate_seam() {
        let lift = PeriodicTorusLift::new(4).unwrap();
        let at_zero = lift.features(0.0).unwrap();
        let at_one = lift.features(1.0).unwrap();
        assert_eq!(at_zero.len(), 8);
        assert_eq!(at_zero.len(), at_one.len());
        for (left, right) in at_zero.iter().zip(&at_one) {
            assert!((left - right).abs() < 1.0e-12);
        }
    }

    #[test]
    fn periodic_torus_lift_is_continuous_across_near_seam_samples() {
        let lift = PeriodicTorusLift::new(4).unwrap();
        let epsilon = 1.0e-9;
        let left = lift.features(1.0 - epsilon).unwrap();
        let right = lift.features(epsilon).unwrap();
        let left_derivative = lift.derivative(1.0 - epsilon).unwrap();
        let right_derivative = lift.derivative(epsilon).unwrap();
        let max_feature_gap = left
            .iter()
            .zip(&right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_derivative_gap = left_derivative
            .iter()
            .zip(&right_derivative)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_feature_gap < 1.0e-6);
        assert!(max_derivative_gap < 1.0e-5);
    }

    #[test]
    fn periodic_torus_lift_derivatives_match_at_the_seam() {
        let lift = PeriodicTorusLift::new(4).unwrap();
        let at_zero = lift.derivative(0.0).unwrap();
        let at_one = lift.derivative(1.0).unwrap();
        for (left, right) in at_zero.iter().zip(&at_one) {
            assert!((left - right).abs() < 1.0e-10);
        }
    }

    #[test]
    fn periodic_torus_lift_wraps_integer_translations() {
        let lift = PeriodicTorusLift::new(5).unwrap();
        let reference = lift.features(0.375).unwrap();
        for translated in [1.375, -0.625, 3.375] {
            let candidate = lift.features(translated).unwrap();
            for (left, right) in reference.iter().zip(&candidate) {
                assert!((left - right).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn periodic_torus_lift_rejects_unbounded_harmonic_budgets() {
        assert_eq!(
            PeriodicTorusLift::new(MAX_PERIODIC_LIFT_HARMONICS + 1),
            Err(ContractError::InvalidHarmonicBudget)
        );
    }

    #[test]
    fn periodic_torus_lift_rejects_zero_harmonics_and_non_finite_coordinates() {
        assert_eq!(
            PeriodicTorusLift::new(0),
            Err(ContractError::InvalidHarmonicBudget)
        );
        let lift = PeriodicTorusLift::new(1).unwrap();
        assert_eq!(
            lift.features(f64::NAN),
            Err(ContractError::NonFiniteNormalizationCoordinate)
        );
        assert_eq!(
            lift.derivative(f64::INFINITY),
            Err(ContractError::NonFiniteNormalizationCoordinate)
        );
    }

    #[test]
    fn open_boundary_lift_keeps_opposite_faces_distinct() {
        let lower = OpenBoundaryCoordinateLift::features(0.0).unwrap();
        let upper = OpenBoundaryCoordinateLift::features(1.0).unwrap();
        assert_eq!(lower.coordinate, 0.0);
        assert_eq!(lower.distance_to_lower_boundary, 0.0);
        assert_eq!(lower.distance_to_upper_boundary, 1.0);
        assert_eq!(upper.coordinate, 1.0);
        assert_eq!(upper.distance_to_lower_boundary, 1.0);
        assert_eq!(upper.distance_to_upper_boundary, 0.0);
        assert_ne!(lower, upper);
    }

    #[test]
    fn open_boundary_lift_rejects_periodic_wrapping_inputs() {
        assert_eq!(
            OpenBoundaryCoordinateLift::features(-0.01),
            Err(ContractError::InvalidOpenBoundaryCoordinate)
        );
        assert_eq!(
            OpenBoundaryCoordinateLift::features(1.01),
            Err(ContractError::InvalidOpenBoundaryCoordinate)
        );
    }

    #[test]
    fn kird_audit_requires_reference_bundle_for_physical_measurements() {
        let mut value: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        value["record_status"] = serde_json::json!("physical_measurement");
        value["provenance"]["physical_execution_authorized"] = serde_json::json!(true);
        for pointer in [
            "/cohort/initial_state_sha256",
            "/task_contract/intervention_family_sha256",
            "/task_contract/qoi_registry_sha256",
            "/candidate/configuration_sha256",
            "/provenance/candidate_output_sha256",
            "/provenance/run_receipt_sha256",
        ] {
            let field = pointer.trim_start_matches('/').replace('/', ".");
            let mut current = &mut value;
            let mut parts = field.split('.').peekable();
            while let Some(part) = parts.next() {
                if parts.peek().is_none() {
                    current[part] = serde_json::json!(
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    );
                } else {
                    current = &mut current[part];
                }
            }
        }
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "provenance.reference_bundle_sha256",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_placeholder_physical_reference_bundle_digest() {
        let mut value: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        value["record_status"] = serde_json::json!("physical_measurement");
        value["provenance"]["physical_execution_authorized"] = serde_json::json!(true);
        for pointer in [
            "/cohort/initial_state_sha256",
            "/task_contract/intervention_family_sha256",
            "/task_contract/qoi_registry_sha256",
            "/candidate/configuration_sha256",
            "/provenance/candidate_output_sha256",
            "/provenance/run_receipt_sha256",
        ] {
            let field = pointer.trim_start_matches('/').replace('/', ".");
            let mut current = &mut value;
            let mut parts = field.split('.').peekable();
            while let Some(part) = parts.next() {
                if parts.peek().is_none() {
                    current[part] = serde_json::json!(
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    );
                } else {
                    current = &mut current[part];
                }
            }
        }
        value["provenance"]["reference_bundle_sha256"] =
            serde_json::json!("0000000000000000000000000000000000000000000000000000000000000000");
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "provenance.reference_bundle_sha256",
                ..
            })
        ));
    }

    #[test]
    fn valid_physical_kird_record_requires_and_accepts_reference_identity() {
        let mut value: serde_json::Value = serde_json::from_str(KIRD_SYNTHETIC_FIXTURE).unwrap();
        value["record_status"] = serde_json::json!("physical_measurement");
        value["provenance"]["physical_execution_authorized"] = serde_json::json!(true);
        for pointer in [
            "/cohort/initial_state_sha256",
            "/task_contract/intervention_family_sha256",
            "/task_contract/qoi_registry_sha256",
            "/candidate/configuration_sha256",
            "/provenance/candidate_output_sha256",
            "/provenance/run_receipt_sha256",
        ] {
            let field = pointer.trim_start_matches('/').replace('/', ".");
            let mut current = &mut value;
            let mut parts = field.split('.').peekable();
            while let Some(part) = parts.next() {
                if parts.peek().is_none() {
                    current[part] = serde_json::json!(
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    );
                } else {
                    current = &mut current[part];
                }
            }
        }
        value["provenance"]["reference_bundle_sha256"] =
            serde_json::json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        let report = validate_kird_record_value(&value).unwrap();
        assert_eq!(report.record_status, "physical_measurement");
        assert_eq!(report.physical_gate_status, "UNKNOWN");
    }

    const SENSOR_ACTUATOR_FIXTURE: &str = include_str!(
        "../../../../benchmarks/plasma/control-interface/fixtures/synthetic-multirate-interface-v0.1.json"
    );

    fn sensor_actuator_fixture_value() -> serde_json::Value {
        serde_json::from_str(SENSOR_ACTUATOR_FIXTURE).unwrap()
    }

    #[test]
    fn sensor_actuator_timing_auditor_accepts_consistent_asynchronous_fixture() {
        let report =
            validate_sensor_actuator_timing_contract_json(SENSOR_ACTUATOR_FIXTURE).unwrap();
        assert_eq!(report.contract_id, "KINETIC-MULTIRATE-INTERFACE-SYNTHETIC-001");
        assert_eq!(report.diagnostic_channel_count, 2);
        assert_eq!(report.actuator_channel_count, 1);
        assert_eq!(report.maximum_observation_age_seconds, 0.04);
        assert_eq!(report.maximum_actuator_delivery_delay_seconds, 0.006);
        assert_eq!(report.controller_update_period_seconds, 0.02);
    }

    #[test]
    fn sensor_actuator_timing_auditor_rejects_understated_timestamp_age() {
        let mut value = sensor_actuator_fixture_value();
        value["diagnostics"][0]["timestamp_age_bound_seconds"] = serde_json::json!(0.01);
        assert!(matches!(
            validate_sensor_actuator_timing_contract_value(&value),
            Err(ContractError::InvalidControlTimingField {
                field: "diagnostics.timestamp_age_bound_seconds",
                ..
            })
        ));
    }

    #[test]
    fn sensor_actuator_timing_auditor_rejects_missed_controller_deadline() {
        let mut value = sensor_actuator_fixture_value();
        value["controller"]["deadline_seconds"] = serde_json::json!(0.001);
        assert!(matches!(
            validate_sensor_actuator_timing_contract_value(&value),
            Err(ContractError::InvalidControlTimingField {
                field: "controller.deadline_seconds",
                ..
            })
        ));
    }

    #[test]
    fn sensor_actuator_timing_auditor_rejects_hidden_full_state_access() {
        let mut value = sensor_actuator_fixture_value();
        value["controller"]["full_state_access"] = serde_json::json!(true);
        assert!(matches!(
            validate_sensor_actuator_timing_contract_value(&value),
            Err(ContractError::InvalidControlTimingField {
                field: "controller.full_state_access",
                ..
            })
        ));
    }

    #[test]
    fn sensor_actuator_timing_auditor_rejects_duplicate_channel_ids() {
        let mut value = sensor_actuator_fixture_value();
        value["diagnostics"][1]["channel_id"] = value["diagnostics"][0]["channel_id"].clone();
        assert!(matches!(
            validate_sensor_actuator_timing_contract_value(&value),
            Err(ContractError::InvalidControlTimingField {
                field: "diagnostics.channel_id",
                ..
            })
        ));
    }

    #[test]
    fn weighted_micro_projector_annihilates_all_declared_moments() {
        let values = [1.0, 2.0, -1.0, 4.0];
        let weights = [1.0, 2.0, 1.5, 3.0];
        let constraints = vec![vec![1.0, 1.0, 1.0, 1.0], vec![-1.0, -0.25, 0.5, 1.0]];
        let result = project_weighted_micro_fluctuation(
            &values,
            &constraints,
            &weights,
            1.0e-11,
        )
        .unwrap();
        assert!(result
            .post_projection_moments
            .iter()
            .all(|moment| moment.abs() < 1.0e-10));
        assert!(result.weighted_correction_norm.is_finite());
        assert!(result.weighted_correction_norm > 0.0);
        assert_ne!(result.micro_fluctuation, values);
    }

    #[test]
    fn weighted_micro_projector_is_idempotent() {
        let values = [2.0, -1.0, 3.0, 0.5];
        let weights = [0.5, 1.0, 2.0, 3.0];
        let constraints = vec![vec![1.0, 1.0, 1.0, 1.0], vec![-1.0, 0.0, 0.0, 1.0]];
        let first = project_weighted_micro_fluctuation(
            &values,
            &constraints,
            &weights,
            1.0e-11,
        )
        .unwrap();
        let second = project_weighted_micro_fluctuation(
            &first.micro_fluctuation,
            &constraints,
            &weights,
            1.0e-11,
        )
        .unwrap();
        for (left, right) in first
            .micro_fluctuation
            .iter()
            .zip(&second.micro_fluctuation)
        {
            assert!((left - right).abs() < 1.0e-10);
        }
        assert!(second.weighted_correction_norm < 1.0e-10);
    }

    #[test]
    fn weighted_micro_projector_rejects_dependent_constraints() {
        let values = [1.0, 2.0, 3.0];
        let weights = [1.0, 1.0, 1.0];
        let constraints = vec![vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0]];
        assert_eq!(
            project_weighted_micro_fluctuation(&values, &constraints, &weights, 1.0e-10),
            Err(ContractError::InvalidProjectionBasis)
        );
    }

    #[test]
    fn weighted_micro_projector_rejects_invalid_weights_dimensions_and_tolerance() {
        let values = [1.0, -1.0, 2.0];
        let constraints = vec![vec![1.0, 1.0, 1.0]];
        assert_eq!(
            project_weighted_micro_fluctuation(&values, &constraints, &[1.0, 0.0, 1.0], 1.0e-10),
            Err(ContractError::InvalidGramianArithmetic)
        );
        assert_eq!(
            project_weighted_micro_fluctuation(&values, &constraints, &[1.0, 1.0], 1.0e-10),
            Err(ContractError::InvalidProjectionBasis)
        );
        assert_eq!(
            project_weighted_micro_fluctuation(&values, &constraints, &[1.0, 1.0, 1.0], 0.0),
            Err(ContractError::InvalidProjectionBasis)
        );
    }

    #[test]
    fn canonical_symplectic_pair_oracle_accepts_area_preserving_basis() {
        let scale = 3.5;
        let q = [scale, 0.0];
        let p = [0.0, 1.0 / scale];
        let defect = canonical_symplectic_pair_defect_2d(q, p).unwrap();
        assert!(defect < 1.0e-12);
        assert!(validate_canonical_symplectic_pair_2d(q, p, 1.0e-12).is_ok());
    }

    #[test]
    fn canonical_symplectic_pair_oracle_rejects_structure_defect() {
        let q = [1.0, 0.0];
        let p = [0.0, 2.0];
        assert_eq!(canonical_symplectic_pair_defect_2d(q, p), Ok(1.0));
        assert_eq!(
            validate_canonical_symplectic_pair_2d(q, p, 1.0e-12),
            Err(ContractError::InvalidSymplecticBasis)
        );
    }

    #[test]
    fn canonical_symplectic_pair_oracle_rejects_non_finite_values_and_tolerance() {
        assert_eq!(
            canonical_symplectic_pair_defect_2d([f64::NAN, 0.0], [0.0, 1.0]),
            Err(ContractError::NonFiniteSymplecticBasis)
        );
        assert_eq!(
            validate_canonical_symplectic_pair_2d([1.0, 0.0], [0.0, 1.0], f64::NAN),
            Err(ContractError::InvalidProjectionBasis)
        );
    }

    #[test]
    fn accepts_synthetic_wire_fixture_but_never_authorizes_physical_execution() {
        let contract = contract();
        assert!(!contract.physical_execution_authorized());
        assert!(contract.can_acquire());
        assert_eq!(contract.contract_id(), "AKPD-SYNTHETIC-PULSE-FAMILY-001");
        assert_eq!(contract.generator_seed(), 7);
        assert!(contract.source_record_refs().is_empty());
    }

    #[test]
    fn rejects_inverted_spatial_mode_range() {
        let mut value = fixture_value();
        value["admissible_controls"]["spatial_mode"]["minimum"] = serde_json::json!(4);
        value["admissible_controls"]["spatial_mode"]["maximum"] = serde_json::json!(1);
        assert!(matches!(
            parse_value(&value),
            Err(ContractError::InvalidField {
                field: "admissible_controls.spatial_mode",
                ..
            })
        ));
    }

    #[test]
    fn rejects_time_range_beyond_declared_horizon() {
        let mut value = fixture_value();
        value["admissible_controls"]["application_time"]["maximum"] = serde_json::json!(31.0);
        assert!(matches!(
            parse_value(&value),
            Err(ContractError::InvalidField {
                field: "admissible_controls.application_time",
                ..
            })
        ));
    }

    #[test]
    fn rejects_insufficient_post_pulse_observation_window() {
        let mut value = fixture_value();
        value["admissible_controls"]["application_time"]["maximum"] = serde_json::json!(29.0);
        assert!(matches!(
            parse_value(&value),
            Err(ContractError::InvalidField {
                field: "evaluation_policy",
                ..
            })
        ));
    }

    #[test]
    fn rejects_sealed_test_acquisition() {
        let mut value = fixture_value();
        value["split_role"] = serde_json::json!("sealed_test");
        assert_eq!(
            parse_value(&value).unwrap_err(),
            ContractError::SealedTestAcquisition
        );
    }

    #[test]
    fn rejects_synthetic_fixture_physical_execution() {
        let mut value = fixture_value();
        value["provenance"]["physical_execution_authorized"] = serde_json::json!(true);
        assert_eq!(
            parse_value(&value).unwrap_err(),
            ContractError::SyntheticExecutionAuthorized
        );
    }

    #[test]
    fn requires_complex_phase_and_waveform_metrics() {
        let mut value = fixture_value();
        value["evaluation_policy"]["response_metrics"] = serde_json::json!(["peak_time"]);
        assert!(matches!(
            parse_value(&value),
            Err(ContractError::InvalidField {
                field: "evaluation_policy.response_metrics",
                ..
            })
        ));
    }

    #[test]
    fn candidate_must_belong_to_declared_family() {
        let family = contract();
        assert!(candidate(2, 0.2, 0.01, 3.0).validate_for(&family).is_ok());
        assert_eq!(
            candidate(5, 0.2, 0.01, 3.0).validate_for(&family),
            Err(ContractError::CandidateOutOfRange("spatial_mode"))
        );
        assert_eq!(
            candidate(2, 0.2, 0.03, 3.0).validate_for(&family),
            Err(ContractError::CandidateOutOfRange(
                "kick_velocity_amplitude"
            ))
        );
    }

    #[test]
    fn rejects_non_finite_candidate_parameters() {
        let family = contract();
        assert_eq!(
            candidate(2, f64::NAN, 0.01, 3.0).validate_for(&family),
            Err(ContractError::CandidateOutOfRange("phase_rad"))
        );
        assert_eq!(
            candidate(2, 0.2, 0.01, f64::INFINITY).validate_for(&family),
            Err(ContractError::CandidateOutOfRange("application_time"))
        );
    }

    #[test]
    fn equivalent_phase_endpoints_have_the_same_digest() {
        let family = contract();
        let negative_pi = candidate(2, -PI, 0.01, 3.0);
        let positive_pi = candidate(2, PI, 0.01, 3.0);
        assert_eq!(
            negative_pi.canonical_digest(&family).unwrap(),
            positive_pi.canonical_digest(&family).unwrap()
        );
    }

    #[test]
    fn no_op_kicks_collapse_to_one_canonical_identity() {
        let family = contract();
        let first = candidate(1, -PI, 0.0, 0.0);
        let second = candidate(4, PI, -0.0, 20.0);
        assert_eq!(
            first.canonical_digest(&family).unwrap(),
            second.canonical_digest(&family).unwrap()
        );
    }

    #[test]
    fn sign_reversal_is_a_pi_phase_shift() {
        let family = contract();
        let pulse = candidate(2, 0.1, 0.01, 3.0);
        let reversed = pulse.sign_reversed();
        assert!(reversed.validate_for(&family).is_ok());
        assert_ne!(
            pulse.canonical_digest(&family).unwrap(),
            reversed.canonical_digest(&family).unwrap()
        );
    }

    #[test]
    fn grid_enumeration_is_deterministic_and_deduplicates_physical_equivalents() {
        let family = contract();
        let spec = GridSpec {
            phase_steps: 3,
            amplitude_steps: 2,
            application_time_steps: 2,
        };
        let first = enumerate_grid(&family, spec).unwrap();
        let second = enumerate_grid(&family, spec).unwrap();
        assert_eq!(first, second);
        let digests = first
            .iter()
            .map(|item| item.canonical_digest(&family).unwrap())
            .collect::<BTreeSet<_>>();
        assert_eq!(digests.len(), first.len());
        assert!(first.len() < 4 * 3 * 2 * 2);
    }

    #[test]
    fn grid_enumeration_fails_closed_when_budget_is_exceeded() {
        let family = contract();
        let error = enumerate_grid(
            &family,
            GridSpec {
                phase_steps: 9,
                amplitude_steps: 3,
                application_time_steps: 2,
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ContractError::ProposalBudgetExceeded {
                requested: 216,
                budget: 128
            }
        ));
    }

    #[test]
    fn grid_requires_positive_axis_resolution() {
        let family = contract();
        assert_eq!(
            enumerate_grid(
                &family,
                GridSpec {
                    phase_steps: 0,
                    amplitude_steps: 1,
                    application_time_steps: 1
                }
            ),
            Err(ContractError::InvalidGrid)
        );
    }

    #[test]
    fn digest_is_scoped_to_contract_and_physics_identity() {
        let first_family = contract();
        let mut second_input = fixture_value();
        second_input["contract_id"] = serde_json::json!("AKPD-SYNTHETIC-PULSE-FAMILY-002");
        let second_family = parse_value(&second_input).unwrap();
        let pulse = candidate(2, 0.2, 0.01, 3.0);
        assert_ne!(
            pulse.canonical_digest(&first_family).unwrap(),
            pulse.canonical_digest(&second_family).unwrap()
        );

        let mut changed_physics = contract();
        changed_physics.physics_cohort.normalization_id = "different-normalization".to_owned();
        assert_ne!(
            pulse.canonical_digest(&first_family).unwrap(),
            pulse.canonical_digest(&changed_physics).unwrap()
        );
    }

    #[test]
    fn confirmation_role_requires_an_independent_reference_identity() {
        let mut value = fixture_value();
        value["split_role"] = serde_json::json!("confirmation");
        assert!(matches!(
            parse_value(&value),
            Err(ContractError::InvalidField {
                field: "evaluation_policy.confirmation_reference_id",
                ..
            })
        ));
    }

    #[test]
    fn rejects_unknown_fields_in_wire_contract() {
        let mut value = fixture_value();
        value
            .as_object_mut()
            .expect("fixture root must be an object")
            .insert("unreviewed_override".to_owned(), serde_json::json!(true));
        assert!(matches!(
            parse_value(&value),
            Err(ContractError::JsonParse(_))
        ));
    }
}
