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

use serde::Deserialize;
use thiserror::Error;

const CONTRACT_VERSION: &str = "0.1";
const EQUATION_ID: &str = "periodic_1d1v_collisionless_vlasov_poisson";
const OPERATOR_ID: &str = "sinusoidal_velocity_kick_v1";
const OPERATOR_DEFINITION: &str = "v_after(x) = v_before(x) + A*sin(2*pi*m*(x-x_min)/L + phase)";
const AMPLITUDE_SEMANTICS: &str = "maximum_absolute_delta_v";
const APPLICATION_SEMANTICS: &str = "instantaneous_canonical_velocity_kick";
const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;
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

#[cfg(test)]
mod tests {
    use super::*;

    const SYNTHETIC_FIXTURE: &str = include_str!(
        "../../../../benchmarks/plasma/adversarial-probes/fixtures/synthetic-pulse-family-v0.1.json"
    );

    fn fixture_value() -> serde_json::Value {
        serde_json::from_str(SYNTHETIC_FIXTURE).expect("checked-in fixture must be valid JSON")
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
    fn minmax_normalization_exposes_global_inlier_drift_from_low_weight_outlier_witness() {
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
