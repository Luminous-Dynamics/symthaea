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
const OPERATOR_DEFINITION: &str =
    "v_after(x) = v_before(x) + A*sin(2*pi*m*(x-x_min)/L + phase)";
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
        let contract: Self =
            serde_json::from_str(input).map_err(|error| ContractError::JsonParse(error.to_string()))?;
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
            || self.physics_cohort.velocity_domain.velocity_unit.trim().is_empty()
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
        if !self.physics_cohort.maximum_time.is_finite()
            || self.physics_cohort.maximum_time <= 0.0
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
        if !policy.independent_confirmation_required
            || policy.minimum_spatial_refinement_levels == 0
            || policy.minimum_time_refinement_levels == 0
        {
            return Err(ContractError::InvalidField {
                field: "evaluation_policy",
                reason: "independent confirmation and spatial/time refinement are mandatory",
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
            || provenance.candidate_representation_revision.trim().is_empty()
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
                || provenance.initial_state_sha256.bytes().all(|byte| byte == b'0')
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
        update_length_prefixed(&mut hasher, self.physics_cohort.velocity_domain.velocity_unit.as_bytes());
        update_length_prefixed(&mut hasher, self.physics_cohort.time_unit.as_bytes());
        update_f64(&mut hasher, self.physics_cohort.maximum_time);
        update_length_prefixed(&mut hasher, self.intervention_operator.operator_id.as_bytes());
        update_length_prefixed(&mut hasher, self.intervention_operator.definition.as_bytes());
        hasher.update(&self.admissible_controls.spatial_mode.minimum.to_le_bytes());
        hasher.update(&self.admissible_controls.spatial_mode.maximum.to_le_bytes());
        update_f64(&mut hasher, self.admissible_controls.phase_rad.minimum);
        update_f64(&mut hasher, self.admissible_controls.phase_rad.maximum);
        update_f64(&mut hasher, self.admissible_controls.kick_amplitude.minimum);
        update_f64(&mut hasher, self.admissible_controls.kick_amplitude.maximum);
        update_length_prefixed(&mut hasher, self.admissible_controls.kick_amplitude.unit.as_bytes());
        update_f64(&mut hasher, self.admissible_controls.application_time.minimum);
        update_f64(&mut hasher, self.admissible_controls.application_time.maximum);
        update_length_prefixed(&mut hasher, self.admissible_controls.application_time.unit.as_bytes());
        update_length_prefixed(&mut hasher, self.provenance.initial_state_sha256.as_bytes());
        update_length_prefixed(&mut hasher, self.provenance.reference_solver_revision.as_bytes());
        hasher.update(&self.provenance.generator_seed.to_le_bytes());
        for source_ref in self.provenance.source_record_refs.iter().collect::<BTreeSet<_>>() {
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
    pub fn validate_for(
        &self,
        contract: &PulseFamilyContract,
    ) -> Result<(), ContractError> {
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
            return Err(ContractError::CandidateOutOfRange("kick_velocity_amplitude"));
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
        if self.phase_steps == 0
            || self.amplitude_steps == 0
            || self.application_time_steps == 0
        {
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
        && bytes[5..].iter().all(|byte| {
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

    const SYNTHETIC_FIXTURE: &str = r#"{
      "schema_version": "0.1",
      "contract_id": "AKPD-SYNTHETIC-PULSE-FAMILY-001",
      "contract_status": "synthetic_fixture_only",
      "physics_cohort": {
        "equation": "periodic_1d1v_collisionless_vlasov_poisson",
        "normalization_id": "synthetic_schema_fixture_not_a_physical_run",
        "spatial_domain": {"minimum": -3.141592653589793, "maximum": 3.141592653589793, "periodic": true},
        "velocity_domain": {"minimum": -6.0, "maximum": 6.0, "velocity_unit": "v_th"},
        "time_unit": "omega_pe^-1",
        "maximum_time": 30.0
      },
      "intervention_operator": {
        "operator_id": "sinusoidal_velocity_kick_v1",
        "definition": "v_after(x) = v_before(x) + A*sin(2*pi*m*(x-x_min)/L + phase)",
        "amplitude_semantics": "maximum_absolute_delta_v",
        "application_semantics": "instantaneous_canonical_velocity_kick"
      },
      "admissible_controls": {
        "spatial_mode": {"minimum": 1, "maximum": 4},
        "phase_rad": {"minimum": -3.141592653589793, "maximum": 3.141592653589793},
        "kick_amplitude": {"minimum": 0.0, "maximum": 0.02, "unit": "v_th"},
        "application_time": {"minimum": 0.0, "maximum": 20.0, "unit": "omega_pe^-1"}
      },
      "search_budget": {
        "maximum_proposals": 128,
        "maximum_high_fidelity_confirmations": 8,
        "maximum_reference_steps": 1000000,
        "maximum_wall_time_seconds": 3600.0
      },
      "evaluation_policy": {
        "independent_confirmation_required": true,
        "minimum_spatial_refinement_levels": 2,
        "minimum_time_refinement_levels": 2,
        "required_controls": ["no_pulse", "sign_reversed", "phase_mismatch", "off_mode_sham"],
        "response_metrics": ["complex_amplitude", "wrapped_phase", "waveform_error", "peak_time", "branch_association"]
      },
      "split_role": "development",
      "acquisition_allowed": true,
      "provenance": {
        "fixture_class": "synthetic_schema_test_only",
        "initial_state_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "reference_solver_revision": "synthetic-fixture-no-solver",
        "candidate_representation_revision": "synthetic-fixture-no-representation",
        "generator_seed": 7,
        "physical_execution_authorized": false,
        "source_record_refs": []
      }
    }"#;

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
        let input = SYNTHETIC_FIXTURE.replace(
            "\"minimum\": 1, \"maximum\": 4",
            "\"minimum\": 4, \"maximum\": 1",
        );
        assert!(matches!(
            PulseFamilyContract::parse_json(&input),
            Err(ContractError::InvalidField {
                field: "admissible_controls.spatial_mode",
                ..
            })
        ));
    }

    #[test]
    fn rejects_time_range_beyond_declared_horizon() {
        let input = SYNTHETIC_FIXTURE.replace(
            "\"maximum\": 20.0, \"unit\": \"omega_pe^-1\"",
            "\"maximum\": 31.0, \"unit\": \"omega_pe^-1\"",
        );
        assert!(matches!(
            PulseFamilyContract::parse_json(&input),
            Err(ContractError::InvalidField {
                field: "admissible_controls.application_time",
                ..
            })
        ));
    }

    #[test]
    fn rejects_sealed_test_acquisition() {
        let input = SYNTHETIC_FIXTURE
            .replace("\"split_role\": \"development\"", "\"split_role\": \"sealed_test\"");
        assert_eq!(
            PulseFamilyContract::parse_json(&input).unwrap_err(),
            ContractError::SealedTestAcquisition
        );
    }

    #[test]
    fn rejects_synthetic_fixture_physical_execution() {
        let input = SYNTHETIC_FIXTURE.replace(
            "\"physical_execution_authorized\": false",
            "\"physical_execution_authorized\": true",
        );
        assert_eq!(
            PulseFamilyContract::parse_json(&input).unwrap_err(),
            ContractError::SyntheticExecutionAuthorized
        );
    }

    #[test]
    fn requires_complex_phase_and_waveform_metrics() {
        let input = SYNTHETIC_FIXTURE.replace(
            "\"complex_amplitude\", \"wrapped_phase\", \"waveform_error\", ",
            "",
        );
        assert!(matches!(
            PulseFamilyContract::parse_json(&input),
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
            Err(ContractError::CandidateOutOfRange("kick_velocity_amplitude"))
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
        let second_input = SYNTHETIC_FIXTURE.replace(
            "AKPD-SYNTHETIC-PULSE-FAMILY-001",
            "AKPD-SYNTHETIC-PULSE-FAMILY-002",
        );
        let second_family = PulseFamilyContract::parse_json(&second_input).unwrap();
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
        let input = SYNTHETIC_FIXTURE.replace(
            "\\"split_role\\": \\"development\\"",
            "\\"split_role\\": \\"confirmation\\"",
        );
        assert!(matches!(
            PulseFamilyContract::parse_json(&input),
            Err(ContractError::InvalidField {
                field: "evaluation_policy.confirmation_reference_id",
                ..
            })
        ));
    }

    #[test]
    fn rejects_unknown_fields_in_wire_contract() {
        let input = SYNTHETIC_FIXTURE.replace(
            "\"contract_status\": \"synthetic_fixture_only\"",
            "\"contract_status\": \"synthetic_fixture_only\", \"unreviewed_override\": true",
        );
        assert!(matches!(
            PulseFamilyContract::parse_json(&input),
            Err(ContractError::JsonParse(_))
        ));
    }
}
