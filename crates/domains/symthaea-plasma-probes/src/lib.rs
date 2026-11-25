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
const MAX_MEASURE_PARTICLES: usize = 1_000_000;
const MAX_FOURIER_MEASURE_WORK: usize = 16_777_216;
const MAX_SCALAR_MODE_HORIZON_STEPS: u32 = 1_000_000;
const MAX_DELAY_CANCELLATION_SAMPLES: usize = 1_000_000;
const MAX_DELAY_HISTORY_WORK: usize = 4_194_304;
const MAX_PERIODIC_DERIVATIVE_SAMPLES: usize = 65_536;
const MAX_PERIODIC_DERIVATIVE_WORK: usize = 8_388_608;
const MAX_RESPONSE_SIGNATURE_CLUSTER_MEMBERS: usize = 128;
const MAX_RESPONSE_SIGNATURE_CLUSTER_WORK: usize = 4_194_304;
const MAX_KINETIC_DISCRIMINATION_HYPOTHESES: usize = 16;
const MAX_KINETIC_DISCRIMINATION_WORK: usize = 32_768;
const MAX_KINETIC_DISCRIMINATION_DESIGNS: usize = 128;
const MAX_KINETIC_DISCRIMINATION_PORTFOLIO_WORK: usize = 4_194_304;
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
    #[error(
        "response signature requires non-empty contract, structure, intervention and QoI identities"
    )]
    InvalidResponseSignatureIdentity,
    #[error("response signature contains duplicate intervention or QoI identities")]
    DuplicateResponseSignatureAxis,
    #[error("response signature dimensions do not match its declared axes")]
    ResponseSignatureShapeMismatch,
    #[error("response signatures have different task-contract identities")]
    ResponseSignatureContractMismatch,
    #[error("response signatures have different intervention, QoI or time axes")]
    ResponseSignatureAxisMismatch,
    #[error("response signatures have different per-QoI normalization scales")]
    ResponseSignatureScaleMismatch,
    #[error("response signature time samples must be finite and strictly increasing")]
    InvalidResponseSignatureTimes,
    #[error("response signature normalization scale must be finite and positive")]
    InvalidResponseSignatureScale,
    #[error("response signature tolerance must be finite and non-negative")]
    InvalidResponseSignatureTolerance,
    #[error("response signature cluster must contain at least one member")]
    EmptyResponseSignatureCluster,
    #[error("response signature cluster exceeds its member or pairwise work budget")]
    ResponseSignatureClusterWorkBudgetExceeded,
    #[error("kinetic discrimination requires at least two competing hypotheses")]
    InsufficientDiscriminationHypotheses,
    #[error("kinetic discrimination hypothesis IDs must be unique and non-empty")]
    InvalidDiscriminationHypothesisIdentity,
    #[error("kinetic discrimination priors must be finite and strictly positive")]
    InvalidDiscriminationPrior,
    #[error("kinetic discrimination noise variance must be finite and positive")]
    InvalidDiscriminationNoiseVariance,
    #[error("kinetic discrimination cost must be finite and positive")]
    InvalidDiscriminationCost,
    #[error("kinetic discrimination quadrature exceeded its bounded work budget")]
    DiscriminationWorkBudgetExceeded,
    #[error("kinetic discrimination portfolio must contain at least one design")]
    EmptyDiscriminationPortfolio,
    #[error("kinetic discrimination portfolio exceeds the bounded design/work budget")]
    DiscriminationPortfolioWorkBudgetExceeded,
    #[error("candidate designs do not share the same ordered hypotheses, priors and noise model")]
    IncomparableDiscriminationDesigns,
    #[error("kinetic discrimination portfolio contains duplicate intervention IDs")]
    DuplicateDiscriminationIntervention,
    #[error("kinetic discrimination requires a stable, identified response branch for every hypothesis")]
    InvalidDiscriminationResponseBranch,
    #[error("kinetic discrimination noise envelope must be finite, positive and ordered")]
    InvalidDiscriminationNoiseEnvelope,
    #[error("kinetic discrimination arithmetic produced an invalid information gain")]
    InvalidDiscriminationArithmetic,
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
    #[error("response-weighted refinement budget or input registry is invalid")]
    InvalidResponseRefinementInput,
    #[error("response-weighted refinement byte accounting overflow")]
    ResponseRefinementByteOverflow,
    #[error("open-boundary coordinate must lie in [0, 1]")]
    InvalidOpenBoundaryCoordinate,
    #[error("particle position and weight arrays must be non-empty and have equal lengths")]
    InvalidParticleMeasureShape,
    #[error("particle measure weights must be finite, non-negative and have positive total mass")]
    InvalidParticleMeasureWeights,
    #[error("particle measure sample exceeds the bounded work budget")]
    ParticleMeasureWorkBudgetExceeded,
    #[error("spectral refinement signal or threshold is invalid")]
    InvalidRefinementSignal,
    #[error("memory budget is smaller than the reserved headroom")]
    InvalidRefinementMemoryBudget,
    #[error("projected refinement memory omits required temporary buffers")]
    IncompleteRefinementMemoryEstimate,
    #[error("canonical symplectic basis contains non-finite values")]
    NonFiniteSymplecticBasis,
    #[error("canonical symplectic pair defect exceeds the declared tolerance")]
    InvalidSymplecticBasis,
    #[error("invalid coupled-mode authority contract: {0}")]
    InvalidCoupledModeAuthority(&'static str),
    #[error("invalid periodic derivative configuration: {0}")]
    InvalidPeriodicDerivativeConfig(&'static str),
    #[error("periodic derivative work exceeds its bounded budget")]
    PeriodicDerivativeWorkBudgetExceeded,
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
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

/// Action assigned to a spectral mode by the response-weighted coarsening planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModeRefinementDisposition {
    DiscardedWithinBudget,
    RetainedBudgetExceeded,
    RetainedUnknownResponse,
}

/// Per-mode evidence produced by response-weighted spectral refinement.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModeRefinementDecision {
    pub input_mode: u32,
    pub coefficient_magnitude: f64,
    pub spectral_energy: f64,
    pub weighted_response_gain: Option<f64>,
    pub response_loss_bound: Option<f64>,
    pub encoded_bytes: u64,
    pub disposition: ModeRefinementDisposition,
}

/// Conservative response-loss budget plan for a proposed mode truncation.
///
/// For each known mode k, the loss bound is
/// |a_k| * sum_q w_q * |G[q,k]|. Missing response entries remain UNKNOWN and
/// are always retained. Known modes are considered in ascending loss-per-byte
/// order; this is a deterministic greedy policy, not an optimal knapsack
/// solution and not a global nonlinear error bound.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResponseWeightedRefinementPlan {
    pub response_loss_budget: f64,
    pub accumulated_discarded_response_bound: f64,
    pub discarded_state_bytes: u64,
    pub decisions: Vec<ModeRefinementDecision>,
}

/// Plan a conservative mode coarsening decision from a phase-resolved response matrix.
///
/// All input modes must have exactly one complex coefficient and one positive
/// encoded-byte count. Every output mode has a finite, non-negative QoI weight;
/// at least one weight must be positive. Any unmeasured response entry prevents
/// that input mode from being discarded.
pub fn plan_response_weighted_refinement(
    response: &ModeResponseMatrix,
    coefficients: &[(u32, ComplexResponse)],
    encoded_bytes: &[(u32, u64)],
    output_weights: &[f64],
    response_loss_budget: f64,
) -> Result<ResponseWeightedRefinementPlan, ContractError> {
    if !response_loss_budget.is_finite()
        || response_loss_budget < 0.0
        || output_weights.len() != response.output_modes().len()
        || output_weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
        || !output_weights.iter().any(|weight| *weight > 0.0)
        || coefficients.len() != response.input_modes().len()
        || encoded_bytes.len() != response.input_modes().len()
    {
        return Err(ContractError::InvalidResponseRefinementInput);
    }

    let mut coefficient_ids = BTreeSet::new();
    for (mode, coefficient) in coefficients {
        if !coefficient_ids.insert(*mode)
            || !response.input_modes().contains(mode)
            || !coefficient.real.is_finite()
            || !coefficient.imag.is_finite()
        {
            return Err(ContractError::InvalidResponseRefinementInput);
        }
    }
    let mut byte_ids = BTreeSet::new();
    for (mode, bytes) in encoded_bytes {
        if !byte_ids.insert(*mode) || *bytes == 0 || !response.input_modes().contains(mode) {
            return Err(ContractError::InvalidResponseRefinementInput);
        }
    }
    let expected_ids = response.input_modes().iter().copied().collect::<BTreeSet<_>>();
    if coefficient_ids != expected_ids || byte_ids != expected_ids {
        return Err(ContractError::InvalidResponseRefinementInput);
    }

    let mut decisions = Vec::with_capacity(response.input_modes().len());
    for mode in response.input_modes() {
        let coefficient = coefficients
            .iter()
            .find(|(candidate, _)| candidate == mode)
            .map(|(_, value)| *value)
            .ok_or(ContractError::InvalidResponseRefinementInput)?;
        let bytes = encoded_bytes
            .iter()
            .find(|(candidate, _)| candidate == mode)
            .map(|(_, value)| *value)
            .ok_or(ContractError::InvalidResponseRefinementInput)?;
        let mut weighted_gain = 0.0_f64;
        let mut complete = true;
        for (output_index, output_mode) in response.output_modes().iter().enumerate() {
            match response.get(*mode, *output_mode)? {
                Some(value) => {
                    weighted_gain += output_weights[output_index] * value.magnitude();
                    if !weighted_gain.is_finite() {
                        return Err(ContractError::InvalidGramianArithmetic);
                    }
                }
                None => complete = false,
            }
        }
        let coefficient_magnitude = coefficient.magnitude();
        let spectral_energy = coefficient.norm_squared();
        if !coefficient_magnitude.is_finite() || !spectral_energy.is_finite() {
            return Err(ContractError::InvalidGramianArithmetic);
        }
        let response_loss_bound = if complete {
            let bound = coefficient_magnitude * weighted_gain;
            if !bound.is_finite() {
                return Err(ContractError::InvalidGramianArithmetic);
            }
            Some(bound)
        } else {
            None
        };
        decisions.push(ModeRefinementDecision {
            input_mode: *mode,
            coefficient_magnitude,
            spectral_energy,
            weighted_response_gain: complete.then_some(weighted_gain),
            response_loss_bound,
            encoded_bytes: bytes,
            disposition: if complete {
                ModeRefinementDisposition::RetainedBudgetExceeded
            } else {
                ModeRefinementDisposition::RetainedUnknownResponse
            },
        });
    }

    let mut candidates = decisions
        .iter()
        .enumerate()
        .filter_map(|(index, decision)| {
            decision.response_loss_bound.map(|bound| {
                (
                    index,
                    bound / decision.encoded_bytes as f64,
                    decision.input_mode,
                    bound,
                    decision.encoded_bytes,
                )
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.2.cmp(&right.2))
    });

    let mut accumulated_discarded_response_bound = 0.0_f64;
    let mut discarded_state_bytes = 0_u64;
    for (index, _, _, bound, bytes) in candidates {
        let proposed_bound = accumulated_discarded_response_bound + bound;
        if proposed_bound.is_finite() && proposed_bound <= response_loss_budget {
            accumulated_discarded_response_bound = proposed_bound;
            discarded_state_bytes = discarded_state_bytes
                .checked_add(bytes)
                .ok_or(ContractError::ResponseRefinementByteOverflow)?;
            decisions[index].disposition = ModeRefinementDisposition::DiscardedWithinBudget;
        }
    }

    Ok(ResponseWeightedRefinementPlan {
        response_loss_budget,
        accumulated_discarded_response_bound,
        discarded_state_bytes,
        decisions,
    })
}

/// A frozen task-conditioned signature of a kinetic state's future responses.
///
/// Flattening order is intervention-major, then QoI-major, then time-minor.
/// The signature is only defined relative to its contract_id, intervention
/// registry, QoI registry, time grid and physical structure class. Missing
/// response values and unresolved branch labels remain explicit.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KineticResponseSignature {
    contract_id: String,
    structure_class_id: String,
    intervention_ids: Vec<String>,
    qoi_ids: Vec<String>,
    qoi_scales: Vec<f64>,
    time_samples: Vec<f64>,
    responses: Vec<Option<ComplexResponse>>,
    branch_ids: Vec<Option<String>>,
}

/// Owned input for constructing a task-conditioned kinetic response signature.
#[derive(Clone, Debug, PartialEq)]
pub struct KineticResponseSignatureInput {
    pub contract_id: String,
    pub structure_class_id: String,
    pub intervention_ids: Vec<String>,
    pub qoi_ids: Vec<String>,
    pub qoi_scales: Vec<f64>,
    pub time_samples: Vec<f64>,
    pub responses: Vec<Option<ComplexResponse>>,
    pub branch_ids: Vec<Option<String>>,
}

/// Pairwise discrepancy and admissibility diagnostics for two response signatures.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ResponseSignatureComparison {
    /// RMS complex-response discrepancy normalized by a task-fixed scale.
    pub normalized_response_distance: Option<f64>,
    /// Worst normalized complex-response error across all measured signature entries.
    pub maximum_normalized_entry_error: Option<f64>,
    pub absolute_frobenius_error: f64,
    pub measured_entry_count: usize,
    pub missing_response_count: usize,
    pub branch_mismatch_count: usize,
    pub unresolved_branch_count: usize,
    pub structure_class_compatible: bool,
}

/// One competing kinetic-state hypothesis for a candidate intervention.
///
/// The predicted response is one registered complex output coefficient. The
/// first implementation uses a finite categorical hypothesis set and a
/// circularly phase-aware complex observation likelihood with independent,
/// equal-variance real/imaginary noise.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KineticDiscriminationHypothesis {
    pub hypothesis_id: String,
    pub prior_probability: f64,
    pub predicted_response: ComplexResponse,
    pub response_branch_id: String,
    pub branch_association_stable: bool,
}

/// One candidate intervention evaluated for expected information gain.
///
/// \`noise_variance_per_quadrature\` is the independently calibrated variance
/// of each real/imaginary observation quadrature. It is not inferred from the
/// same candidate predictions. \`estimated_evaluation_cost_seconds\` is an
/// external cost estimate and is reported separately from information gain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KineticDiscriminationDesign {
    pub intervention_id: String,
    pub hypotheses: Vec<KineticDiscriminationHypothesis>,
    pub noise_variance_per_quadrature: f64,
    pub estimated_evaluation_cost_seconds: f64,
}

/// Expected-information-gain report for one finite complex-response design.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct KineticDiscriminationReport {
    pub hypothesis_count: usize,
    pub prior_entropy_bits: f64,
    pub expected_posterior_entropy_bits: f64,
    pub expected_information_gain_bits: f64,
    pub estimated_evaluation_cost_seconds: f64,
    pub information_gain_bits_per_cost_second: f64,
    pub noise_variance_per_quadrature: f64,
    pub quadrature_order_per_axis: usize,
}

/// Information-gain sensitivity over a declared interval of Gaussian
/// observation-noise variance per real/imaginary quadrature.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct KineticDiscriminationNoiseEnvelopeReport {
    pub minimum_noise_variance: f64,
    pub maximum_noise_variance: f64,
    /// EIG at the maximum noise endpoint (worst information under this model).
    pub worst_case_endpoint_information_gain_bits: f64,
    /// EIG at the minimum noise endpoint (best information under this model).
    pub best_case_endpoint_information_gain_bits: f64,
    pub worst_case_endpoint_information_gain_bits_per_cost_second: f64,
    pub best_case_endpoint_information_gain_bits_per_cost_second: f64,
    pub monotone_noise_degradation_check: bool,
}

/// One evaluated candidate intervention in a matched design portfolio.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KineticDiscriminationCandidateReport {
    pub intervention_id: String,
    pub report: KineticDiscriminationReport,
}

/// Pareto analysis across candidate interventions evaluated against the same
/// ordered state hypotheses, prior distribution and observation-noise model.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KineticDiscriminationPortfolioReport {
    pub candidate_reports: Vec<KineticDiscriminationCandidateReport>,
    pub pareto_front_indices: Vec<usize>,
    pub best_information_gain_per_cost_index: usize,
}

/// Evaluate a candidate design at both endpoints of a declared Gaussian
/// observation-noise variance interval.
///
/// For fixed hypothesis predictions and priors, independent additive Gaussian
/// noise with larger variance is a degraded observation channel: the larger-noise
/// observation can be generated by adding independent Gaussian noise to the
/// smaller-noise observation. By the data-processing inequality, EIG cannot
/// increase with variance. This helper checks that endpoint ordering numerically.
/// It does not certify that a real plasma instrument/reference noise lies inside
/// the caller-declared interval.
pub fn evaluate_kinetic_discrimination_noise_envelope(
    design: &KineticDiscriminationDesign,
    minimum_noise_variance: f64,
    maximum_noise_variance: f64,
) -> Result<KineticDiscriminationNoiseEnvelopeReport, ContractError> {
    if !minimum_noise_variance.is_finite()
        || !maximum_noise_variance.is_finite()
        || minimum_noise_variance <= 0.0
        || maximum_noise_variance < minimum_noise_variance
    {
        return Err(ContractError::InvalidDiscriminationNoiseEnvelope);
    }
    let mut lower_noise_design = design.clone();
    lower_noise_design.noise_variance_per_quadrature = minimum_noise_variance;
    let mut upper_noise_design = design.clone();
    upper_noise_design.noise_variance_per_quadrature = maximum_noise_variance;
    let best_case = evaluate_kinetic_discrimination_design(&lower_noise_design)?;
    let worst_case = evaluate_kinetic_discrimination_design(&upper_noise_design)?;
    let allowance = 1.0e-9_f64.max(best_case.expected_information_gain_bits * 1.0e-8);
    if worst_case.expected_information_gain_bits
        > best_case.expected_information_gain_bits + allowance
    {
        return Err(ContractError::InvalidDiscriminationArithmetic);
    }
    Ok(KineticDiscriminationNoiseEnvelopeReport {
        minimum_noise_variance,
        maximum_noise_variance,
        worst_case_endpoint_information_gain_bits: worst_case.expected_information_gain_bits,
        best_case_endpoint_information_gain_bits: best_case.expected_information_gain_bits,
        worst_case_endpoint_information_gain_bits_per_cost_second: worst_case
            .information_gain_bits_per_cost_second,
        best_case_endpoint_information_gain_bits_per_cost_second: best_case
            .information_gain_bits_per_cost_second,
        monotone_noise_degradation_check: true,
    })
}

/// Evaluate and compare a bounded set of candidate interventions.
///
/// Candidates must use exactly the same ordered hypothesis IDs and the same
/// normalized prior distribution / observation-noise variance. This prevents
/// design scores from being compared across changed hypothesis populations or
/// incompatible measurement channels. The Pareto frontier maximizes EIG while
/// minimizing declared evaluation cost; the best EIG-per-cost index is
/// reported separately as one scalar policy option.
pub fn evaluate_kinetic_discrimination_portfolio(
    designs: &[KineticDiscriminationDesign],
) -> Result<KineticDiscriminationPortfolioReport, ContractError> {
    if designs.is_empty() {
        return Err(ContractError::EmptyDiscriminationPortfolio);
    }
    if designs.len() > MAX_KINETIC_DISCRIMINATION_DESIGNS {
        return Err(ContractError::DiscriminationPortfolioWorkBudgetExceeded);
    }

    let reference = &designs[0];
    let reference_probabilities = normalized_discrimination_priors(&reference.hypotheses)?;
    let reference_ids = reference
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.hypothesis_id.as_str())
        .collect::<Vec<_>>();
    let mut intervention_ids = BTreeSet::new();
    let mut total_work = 0_usize;
    let mut candidate_reports = Vec::with_capacity(designs.len());

    for design in designs {
        if !intervention_ids.insert(design.intervention_id.as_str()) {
            return Err(ContractError::DuplicateDiscriminationIntervention);
        }
        if design.hypotheses.len() != reference.hypotheses.len()
            || design.noise_variance_per_quadrature != reference.noise_variance_per_quadrature
            || design
                .hypotheses
                .iter()
                .map(|hypothesis| hypothesis.hypothesis_id.as_str())
                .collect::<Vec<_>>()
                != reference_ids
        {
            return Err(ContractError::IncomparableDiscriminationDesigns);
        }
        let probabilities = normalized_discrimination_priors(&design.hypotheses)?;
        if probabilities
            .iter()
            .zip(&reference_probabilities)
            .any(|(left, right)| (left - right).abs() > 1.0e-12)
        {
            return Err(ContractError::IncomparableDiscriminationDesigns);
        }
        let count = design.hypotheses.len();
        let design_work = count
            .checked_mul(count)
            .and_then(|value| value.checked_mul(9 * 9))
            .ok_or(ContractError::DiscriminationPortfolioWorkBudgetExceeded)?;
        total_work = total_work
            .checked_add(design_work)
            .ok_or(ContractError::DiscriminationPortfolioWorkBudgetExceeded)?;
        if total_work > MAX_KINETIC_DISCRIMINATION_PORTFOLIO_WORK {
            return Err(ContractError::DiscriminationPortfolioWorkBudgetExceeded);
        }

        candidate_reports.push(KineticDiscriminationCandidateReport {
            intervention_id: design.intervention_id.clone(),
            report: evaluate_kinetic_discrimination_design(design)?,
        });
    }

    let mut pareto_front_indices = Vec::new();
    for candidate_index in 0..candidate_reports.len() {
        let candidate = &candidate_reports[candidate_index].report;
        let dominated = candidate_reports.iter().enumerate().any(|(other_index, other)| {
            if other_index == candidate_index {
                return false;
            }
            let no_worse_information =
                other.report.expected_information_gain_bits
                    >= candidate.expected_information_gain_bits;
            let no_worse_cost = other.report.estimated_evaluation_cost_seconds
                <= candidate.estimated_evaluation_cost_seconds;
            let strictly_better = other.report.expected_information_gain_bits
                > candidate.expected_information_gain_bits
                || other.report.estimated_evaluation_cost_seconds
                    < candidate.estimated_evaluation_cost_seconds;
            no_worse_information && no_worse_cost && strictly_better
        });
        if !dominated {
            pareto_front_indices.push(candidate_index);
        }
    }

    let best_information_gain_per_cost_index = (0..candidate_reports.len())
        .max_by(|left_index, right_index| {
            let left = &candidate_reports[*left_index].report;
            let right = &candidate_reports[*right_index].report;
            left.information_gain_bits_per_cost_second
                .total_cmp(&right.information_gain_bits_per_cost_second)
                .then_with(|| {
                    left.expected_information_gain_bits
                        .total_cmp(&right.expected_information_gain_bits)
                })
                .then_with(|| {
                    right
                        .estimated_evaluation_cost_seconds
                        .total_cmp(&left.estimated_evaluation_cost_seconds)
                })
        })
        .ok_or(ContractError::EmptyDiscriminationPortfolio)?;

    Ok(KineticDiscriminationPortfolioReport {
        candidate_reports,
        pareto_front_indices,
        best_information_gain_per_cost_index,
    })
}

fn normalized_discrimination_priors(
    hypotheses: &[KineticDiscriminationHypothesis],
) -> Result<Vec<f64>, ContractError> {
    if hypotheses.len() < 2 || hypotheses.len() > MAX_KINETIC_DISCRIMINATION_HYPOTHESES {
        return Err(ContractError::InsufficientDiscriminationHypotheses);
    }
    let mut seen_ids = BTreeSet::new();
    let mut maximum_prior = 0.0_f64;
    for hypothesis in hypotheses {
        if hypothesis.hypothesis_id.trim().is_empty()
            || !seen_ids.insert(hypothesis.hypothesis_id.as_str())
        {
            return Err(ContractError::InvalidDiscriminationHypothesisIdentity);
        }
        if !hypothesis.prior_probability.is_finite() || hypothesis.prior_probability <= 0.0 {
            return Err(ContractError::InvalidDiscriminationPrior);
        }
        maximum_prior = maximum_prior.max(hypothesis.prior_probability);
    }
    let scaled_sum = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability / maximum_prior)
        .sum::<f64>();
    if !scaled_sum.is_finite() || scaled_sum <= 0.0 {
        return Err(ContractError::InvalidDiscriminationPrior);
    }
    Ok(hypotheses
        .iter()
        .map(|hypothesis| (hypothesis.prior_probability / maximum_prior) / scaled_sum)
        .collect())
}

/// Nine-point Gauss-Hermite rule for an expectation under a standard normal.
/// Nodes and weights are the physicists' exp(-x^2) rule; the evaluator applies
/// z=sqrt(2)*x and divides weights by sqrt(pi).
const GAUSS_HERMITE_9_NODES: [f64; 9] = [
    -3.190_993_201_781_528,
    -2.266_580_584_531_843,
    -1.468_553_289_216_668,
    -0.723_551_018_752_838,
    0.0,
    0.723_551_018_752_838,
    1.468_553_289_216_668,
    2.266_580_584_531_843,
    3.190_993_201_781_528,
];

const GAUSS_HERMITE_9_WEIGHTS: [f64; 9] = [
    0.000_039_606_977_263_264_4,
    0.004_943_624_275_536_947,
    0.088_474_527_394_376_57,
    0.432_651_559_002_555_75,
    0.720_235_215_606_050_9,
    0.432_651_559_002_555_75,
    0.088_474_527_394_376_57,
    0.004_943_624_275_536_947,
    0.000_039_606_977_263_264_4,
];

/// Estimate mutual information I(H;Y|u) in bits for one candidate intervention.
///
/// The expectation is integrated by deterministic 9x9 Gauss-Hermite
/// quadrature over independent Gaussian real/imaginary observation noise.
/// This is a finite-hypothesis, homoscedastic complex-Gaussian likelihood
/// primitive; it is not a universal likelihood for kinetic solver residuals.
pub fn evaluate_kinetic_discrimination_design(
    design: &KineticDiscriminationDesign,
) -> Result<KineticDiscriminationReport, ContractError> {
    if design.intervention_id.trim().is_empty() {
        return Err(ContractError::InvalidDiscriminationHypothesisIdentity);
    }
    let hypothesis_count = design.hypotheses.len();
    if hypothesis_count < 2 {
        return Err(ContractError::InsufficientDiscriminationHypotheses);
    }
    if hypothesis_count > MAX_KINETIC_DISCRIMINATION_HYPOTHESES {
        return Err(ContractError::DiscriminationWorkBudgetExceeded);
    }
    if !design.noise_variance_per_quadrature.is_finite()
        || design.noise_variance_per_quadrature <= 0.0
    {
        return Err(ContractError::InvalidDiscriminationNoiseVariance);
    }
    if !design.estimated_evaluation_cost_seconds.is_finite()
        || design.estimated_evaluation_cost_seconds <= 0.0
    {
        return Err(ContractError::InvalidDiscriminationCost);
    }
    let work = hypothesis_count
        .checked_mul(hypothesis_count)
        .and_then(|count| count.checked_mul(9 * 9))
        .ok_or(ContractError::DiscriminationWorkBudgetExceeded)?;
    if work > MAX_KINETIC_DISCRIMINATION_WORK {
        return Err(ContractError::DiscriminationWorkBudgetExceeded);
    }

    let mut seen_ids = BTreeSet::new();
    let mut maximum_prior = 0.0_f64;
    for hypothesis in &design.hypotheses {
        if hypothesis.hypothesis_id.trim().is_empty()
            || !seen_ids.insert(hypothesis.hypothesis_id.as_str())
        {
            return Err(ContractError::InvalidDiscriminationHypothesisIdentity);
        }
        if !hypothesis.prior_probability.is_finite() || hypothesis.prior_probability <= 0.0 {
            return Err(ContractError::InvalidDiscriminationPrior);
        }
        if !hypothesis.predicted_response.real.is_finite()
            || !hypothesis.predicted_response.imag.is_finite()
        {
            return Err(ContractError::NonFiniteResponseValue);
        }
        if hypothesis.response_branch_id.trim().is_empty()
            || !hypothesis.branch_association_stable
        {
            return Err(ContractError::InvalidDiscriminationResponseBranch);
        }
        maximum_prior = maximum_prior.max(hypothesis.prior_probability);
    }

    // Max scaling avoids overflow when perfectly valid unnormalized priors are large.
    let scaled_prior_sum = design
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability / maximum_prior)
        .sum::<f64>();
    if !scaled_prior_sum.is_finite() || scaled_prior_sum <= 0.0 {
        return Err(ContractError::InvalidDiscriminationPrior);
    }
    let probabilities = design
        .hypotheses
        .iter()
        .map(|hypothesis| (hypothesis.prior_probability / maximum_prior) / scaled_prior_sum)
        .collect::<Vec<_>>();
    let prior_entropy_bits = probabilities
        .iter()
        .map(|probability| -probability * probability.log2())
        .sum::<f64>();
    if !prior_entropy_bits.is_finite() {
        return Err(ContractError::InvalidDiscriminationArithmetic);
    }

    let variance = design.noise_variance_per_quadrature;
    let standard_deviation = variance.sqrt();
    let log_two = std::f64::consts::LN_2;
    let weight_normalizer = PI.sqrt();
    let mut expected_information_nats = 0.0_f64;

    for (true_index, true_hypothesis) in design.hypotheses.iter().enumerate() {
        let true_prior = probabilities[true_index];
        for real_index in 0..GAUSS_HERMITE_9_NODES.len() {
            let real_noise = std::f64::consts::SQRT_2
                * GAUSS_HERMITE_9_NODES[real_index]
                * standard_deviation;
            let real_weight = GAUSS_HERMITE_9_WEIGHTS[real_index] / weight_normalizer;
            for imag_index in 0..GAUSS_HERMITE_9_NODES.len() {
                let imag_noise = std::f64::consts::SQRT_2
                    * GAUSS_HERMITE_9_NODES[imag_index]
                    * standard_deviation;
                let quadrature_weight =
                    real_weight * GAUSS_HERMITE_9_WEIGHTS[imag_index] / weight_normalizer;
                let observation_real = true_hypothesis.predicted_response.real + real_noise;
                let observation_imag = true_hypothesis.predicted_response.imag + imag_noise;
                if !observation_real.is_finite() || !observation_imag.is_finite() {
                    return Err(ContractError::InvalidDiscriminationArithmetic);
                }

                let mut log_joint = Vec::with_capacity(hypothesis_count);
                for (hypothesis_index, hypothesis) in design.hypotheses.iter().enumerate() {
                    let delta_real = observation_real - hypothesis.predicted_response.real;
                    let delta_imag = observation_imag - hypothesis.predicted_response.imag;
                    let normalized_distance = delta_real.hypot(delta_imag) / standard_deviation;
                    // Very remote hypotheses can underflow to effectively zero likelihood;
                    // clamp the log likelihood to retain stable log-sum-exp arithmetic.
                    let log_likelihood = if normalized_distance.is_finite()
                        && normalized_distance < 1.0e150
                    {
                        -0.5 * normalized_distance * normalized_distance
                    } else {
                        -1.0e300
                    };
                    log_joint.push(probabilities[hypothesis_index].ln() + log_likelihood);
                }

                let maximum_log_joint = log_joint.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let log_mixture = maximum_log_joint
                    + log_joint
                        .iter()
                        .map(|value| (value - maximum_log_joint).exp())
                        .sum::<f64>()
                        .ln();
                let true_delta_real = observation_real - true_hypothesis.predicted_response.real;
                let true_delta_imag = observation_imag - true_hypothesis.predicted_response.imag;
                let true_normalized_distance =
                    true_delta_real.hypot(true_delta_imag) / standard_deviation;
                let true_log_likelihood = -0.5 * true_normalized_distance * true_normalized_distance;
                let information_nats = true_log_likelihood - log_mixture;
                if !information_nats.is_finite() || !quadrature_weight.is_finite() {
                    return Err(ContractError::InvalidDiscriminationArithmetic);
                }
                expected_information_nats +=
                    true_prior * quadrature_weight * information_nats;
            }
        }
    }

    if !expected_information_nats.is_finite() {
        return Err(ContractError::InvalidDiscriminationArithmetic);
    }
    let information_tolerance = 1.0e-10_f64.max(prior_entropy_bits * 1.0e-9);
    if expected_information_nats < -information_tolerance * log_two
        || expected_information_nats / log_two > prior_entropy_bits + information_tolerance
    {
        return Err(ContractError::InvalidDiscriminationArithmetic);
    }
    let expected_information_gain_bits =
        (expected_information_nats / log_two).clamp(0.0, prior_entropy_bits);
    let expected_posterior_entropy_bits =
        (prior_entropy_bits - expected_information_gain_bits).max(0.0);
    let information_gain_bits_per_cost_second =
        expected_information_gain_bits / design.estimated_evaluation_cost_seconds;
    if !information_gain_bits_per_cost_second.is_finite() {
        return Err(ContractError::InvalidDiscriminationArithmetic);
    }

    Ok(KineticDiscriminationReport {
        hypothesis_count,
        prior_entropy_bits,
        expected_posterior_entropy_bits,
        expected_information_gain_bits,
        estimated_evaluation_cost_seconds: design.estimated_evaluation_cost_seconds,
        information_gain_bits_per_cost_second,
        noise_variance_per_quadrature: variance,
        quadrature_order_per_axis: GAUSS_HERMITE_9_NODES.len(),
    })
}

/// Finite all-pairs diameter audit for a proposed response-equivalence class.
///
/// This report establishes only a diameter bound on the supplied finite set of
/// response signatures. It does not establish equivalence for untested states
/// or interventions, and it is not a global quotient certificate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ResponseSignatureClusterAudit {
    pub member_count: usize,
    pub compared_pair_count: usize,
    pub compatible_pair_count: usize,
    pub incompatible_pair_count: usize,
    pub maximum_pairwise_normalized_rms_distance: Option<f64>,
    pub maximum_pairwise_normalized_entry_error: Option<f64>,
    pub complete_response_coverage: bool,
    pub stable_branch_association: bool,
    pub compatible_structure_class: bool,
    pub within_finite_class_diameter: bool,
}

/// Audit every pair in a proposed finite state class.
///
/// Unlike connected-component clustering of pairwise-compatible edges, this
/// requires every pair in the proposed class to satisfy the frozen tolerance.
/// Admission uses the maximum normalized complex-response error at any
/// intervention/QoI/time entry, while retaining RMS distance as a diagnostic.
pub fn audit_response_signature_cluster(
    signatures: &[KineticResponseSignature],
    tolerance: f64,
) -> Result<ResponseSignatureClusterAudit, ContractError> {
    if signatures.is_empty() {
        return Err(ContractError::EmptyResponseSignatureCluster);
    }
    if signatures.len() > MAX_RESPONSE_SIGNATURE_CLUSTER_MEMBERS {
        return Err(ContractError::ResponseSignatureClusterWorkBudgetExceeded);
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ContractError::InvalidResponseSignatureTolerance);
    }

    let member_count = signatures.len();
    let pair_count = member_count
        .checked_mul(member_count.saturating_sub(1))
        .and_then(|count| count.checked_div(2))
        .ok_or(ContractError::ResponseSignatureClusterWorkBudgetExceeded)?;
    let entries_per_signature = signatures[0].responses.len();
    let comparison_work = pair_count
        .checked_mul(entries_per_signature)
        .ok_or(ContractError::ResponseSignatureClusterWorkBudgetExceeded)?;
    if comparison_work > MAX_RESPONSE_SIGNATURE_CLUSTER_WORK {
        return Err(ContractError::ResponseSignatureClusterWorkBudgetExceeded);
    }

    let mut compatible_pair_count = 0_usize;
    let mut maximum_pairwise_normalized_rms_distance = None::<f64>;
    let mut maximum_pairwise_normalized_entry_error = None::<f64>;
    let mut complete_response_coverage = true;
    let mut stable_branch_association = true;
    let mut compatible_structure_class = true;

    for left_index in 0..member_count {
        for right_index in (left_index + 1)..member_count {
            let comparison = signatures[left_index].compare(&signatures[right_index])?;
            complete_response_coverage &= comparison.missing_response_count == 0;
            stable_branch_association &=
                comparison.branch_mismatch_count == 0 && comparison.unresolved_branch_count == 0;
            compatible_structure_class &= comparison.structure_class_compatible;

            if let Some(distance) = comparison.normalized_response_distance {
                maximum_pairwise_normalized_rms_distance = Some(
                    maximum_pairwise_normalized_rms_distance
                        .map_or(distance, |current| current.max(distance)),
                );
            }
            if let Some(error) = comparison.maximum_normalized_entry_error {
                maximum_pairwise_normalized_entry_error = Some(
                    maximum_pairwise_normalized_entry_error
                        .map_or(error, |current| current.max(error)),
                );
            }
            if KineticResponseSignature::pairwise_compatible(comparison, tolerance)? {
                compatible_pair_count += 1;
            }
        }
    }

    let incompatible_pair_count = pair_count - compatible_pair_count;
    Ok(ResponseSignatureClusterAudit {
        member_count,
        compared_pair_count: pair_count,
        compatible_pair_count,
        incompatible_pair_count,
        maximum_pairwise_normalized_rms_distance,
        maximum_pairwise_normalized_entry_error,
        complete_response_coverage,
        stable_branch_association,
        compatible_structure_class,
        within_finite_class_diameter: incompatible_pair_count == 0,
    })
}

impl KineticResponseSignature {
    /// Construct a signature with exact task axes and per-QoI normalization scales.
    pub fn new(input: KineticResponseSignatureInput) -> Result<Self, ContractError> {
        let KineticResponseSignatureInput {
            contract_id,
            structure_class_id,
            intervention_ids,
            qoi_ids,
            qoi_scales,
            time_samples,
            responses,
            branch_ids,
        } = input;
        if contract_id.trim().is_empty()
            || structure_class_id.trim().is_empty()
            || intervention_ids.is_empty()
            || qoi_ids.is_empty()
            || time_samples.is_empty()
        {
            return Err(ContractError::InvalidResponseSignatureIdentity);
        }
        if has_duplicate_strings(&intervention_ids) || has_duplicate_strings(&qoi_ids) {
            return Err(ContractError::DuplicateResponseSignatureAxis);
        }
        if qoi_scales.len() != qoi_ids.len() {
            return Err(ContractError::ResponseSignatureShapeMismatch);
        }
        if qoi_scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(ContractError::InvalidResponseSignatureScale);
        }
        if time_samples.iter().any(|time| !time.is_finite())
            || time_samples.windows(2).any(|window| window[1] <= window[0])
        {
            return Err(ContractError::InvalidResponseSignatureTimes);
        }
        let expected_entries = intervention_ids
            .len()
            .checked_mul(qoi_ids.len())
            .and_then(|count| count.checked_mul(time_samples.len()))
            .ok_or(ContractError::ResponseMatrixSizeOverflow)?;
        if responses.len() != expected_entries || branch_ids.len() != expected_entries {
            return Err(ContractError::ResponseSignatureShapeMismatch);
        }
        if responses
            .iter()
            .flatten()
            .any(|response| !response.real.is_finite() || !response.imag.is_finite())
        {
            return Err(ContractError::NonFiniteResponseValue);
        }
        if branch_ids
            .iter()
            .flatten()
            .any(|branch| branch.trim().is_empty())
        {
            return Err(ContractError::InvalidResponseSignatureIdentity);
        }
        Ok(Self {
            contract_id,
            structure_class_id,
            intervention_ids,
            qoi_ids,
            qoi_scales,
            time_samples,
            responses,
            branch_ids,
        })
    }

    /// Compare signatures using frozen per-QoI scales and identical task axes.
    ///
    /// Each complex response residual is divided by the registered scale for
    /// its QoI before the RMS is taken across the shared response grid. Scales
    /// are part of the signature, so callers cannot change normalization
    /// between pairwise comparisons.
    pub fn compare(&self, other: &Self) -> Result<ResponseSignatureComparison, ContractError> {
        if self.contract_id != other.contract_id {
            return Err(ContractError::ResponseSignatureContractMismatch);
        }
        if self.intervention_ids != other.intervention_ids
            || self.qoi_ids != other.qoi_ids
            || self.time_samples != other.time_samples
        {
            return Err(ContractError::ResponseSignatureAxisMismatch);
        }
        if self.qoi_scales != other.qoi_scales {
            return Err(ContractError::ResponseSignatureScaleMismatch);
        }

        let mut squared_error = 0.0_f64;
        let mut squared_normalized_error = 0.0_f64;
        let mut maximum_normalized_entry_error = None::<f64>;
        let mut measured_entry_count = 0_usize;
        let mut missing_response_count = 0_usize;
        let mut branch_mismatch_count = 0_usize;
        let mut unresolved_branch_count = 0_usize;
        let time_count = self.time_samples.len();
        let qoi_count = self.qoi_ids.len();

        for (index, (((left_response, right_response), left_branch), right_branch)) in self
            .responses
            .iter()
            .zip(&other.responses)
            .zip(&self.branch_ids)
            .zip(&other.branch_ids)
            .enumerate()
        {
            match (left_response, right_response) {
                (Some(left), Some(right)) => {
                    let real_delta = left.real - right.real;
                    let imag_delta = left.imag - right.imag;
                    let squared_delta = real_delta.mul_add(real_delta, imag_delta * imag_delta);
                    let qoi_index = (index / time_count) % qoi_count;
                    let scale = self.qoi_scales[qoi_index];
                    let normalized_entry_error = squared_delta.sqrt() / scale;
                    if !normalized_entry_error.is_finite() {
                        return Err(ContractError::InvalidGramianArithmetic);
                    }
                    maximum_normalized_entry_error = Some(
                        maximum_normalized_entry_error
                            .map_or(normalized_entry_error, |current| {
                                current.max(normalized_entry_error)
                            }),
                    );
                    squared_error += squared_delta;
                    squared_normalized_error += squared_delta / (scale * scale);
                    measured_entry_count += 1;
                }
                _ => missing_response_count += 1,
            }
            match (left_branch, right_branch) {
                (Some(left), Some(right)) if left != right => branch_mismatch_count += 1,
                (Some(_), Some(_)) => {}
                _ => unresolved_branch_count += 1,
            }
        }
        if !squared_error.is_finite() || !squared_normalized_error.is_finite() {
            return Err(ContractError::InvalidGramianArithmetic);
        }
        let absolute_frobenius_error = squared_error.sqrt();
        let normalized_response_distance = if measured_entry_count > 0 {
            Some((squared_normalized_error / measured_entry_count as f64).sqrt())
        } else {
            None
        };
        if normalized_response_distance.is_some_and(|distance| !distance.is_finite()) {
            return Err(ContractError::InvalidGramianArithmetic);
        }

        Ok(ResponseSignatureComparison {
            normalized_response_distance,
            maximum_normalized_entry_error,
            absolute_frobenius_error,
            measured_entry_count,
            missing_response_count,
            branch_mismatch_count,
            unresolved_branch_count,
            structure_class_compatible: self.structure_class_id == other.structure_class_id,
        })
    }

    /// Pairwise tolerance test only; this relation is not generally transitive.
    ///
    /// Do not build equivalence classes by blindly unioning pairwise-compatible
    /// edges. A class/cluster must separately satisfy a whole-class diameter
    /// bound or a certified quotient construction.
    pub fn pairwise_compatible(
        comparison: ResponseSignatureComparison,
        tolerance: f64,
    ) -> Result<bool, ContractError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(ContractError::InvalidResponseSignatureTolerance);
        }
        Ok(comparison.missing_response_count == 0
            && comparison.branch_mismatch_count == 0
            && comparison.unresolved_branch_count == 0
            && comparison.structure_class_compatible
            && comparison
                .maximum_normalized_entry_error
                .is_some_and(|error| error <= tolerance))
    }

    pub fn contract_id(&self) -> &str {
        &self.contract_id
    }

    pub fn intervention_ids(&self) -> &[String] {
        &self.intervention_ids
    }

    pub fn qoi_ids(&self) -> &[String] {
        &self.qoi_ids
    }

    pub fn qoi_scales(&self) -> &[f64] {
        &self.qoi_scales
    }

    pub fn time_samples(&self) -> &[f64] {
        &self.time_samples
    }
}

fn has_duplicate_strings(values: &[String]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().any(|value| !seen.insert(value))
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

/// Low-order Fourier moments of a weighted empirical particle measure on a torus.
///
/// For modes k=1..K this computes
/// \`m_k = sum_i (w_i / sum_j w_j) exp(i 2*pi*k*x_i)\`.
/// It is invariant to particle ordering and exact co-located particle split/merge
/// in the represented empirical measure. A finite K is only a band-limited
/// observable embedding; it is not a unique reconstruction of an arbitrary
/// continuous or discrete measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightedFourierMeasureEmbedding {
    max_harmonic: usize,
}

impl WeightedFourierMeasureEmbedding {
    /// Construct a bounded Fourier-moment embedding.
    pub fn new(max_harmonic: usize) -> Result<Self, ContractError> {
        if max_harmonic == 0 || max_harmonic > MAX_PERIODIC_LIFT_HARMONICS {
            return Err(ContractError::InvalidHarmonicBudget);
        }
        Ok(Self { max_harmonic })
    }

    /// Compute normalized complex Fourier moments for a weighted particle sample.
    ///
    /// Coordinates are wrapped modulo one. Weights are normalized using a
    /// max-scaled sum to avoid overflowing when all finite weights are large.
    /// The operation budget is bounded before allocating the output vector.
    pub fn moments(
        &self,
        positions: &[f64],
        weights: &[f64],
    ) -> Result<Vec<ComplexResponse>, ContractError> {
        if positions.is_empty()
            || positions.len() != weights.len()
            || positions.len() > MAX_MEASURE_PARTICLES
        {
            return Err(ContractError::InvalidParticleMeasureShape);
        }
        if positions.iter().any(|position| !position.is_finite()) {
            return Err(ContractError::NonFiniteNormalizationCoordinate);
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(ContractError::InvalidParticleMeasureWeights);
        }
        let work = positions
            .len()
            .checked_mul(self.max_harmonic)
            .ok_or(ContractError::ParticleMeasureWorkBudgetExceeded)?;
        if work > MAX_FOURIER_MEASURE_WORK {
            return Err(ContractError::ParticleMeasureWorkBudgetExceeded);
        }

        let maximum_weight = weights.iter().copied().fold(0.0_f64, f64::max);
        if maximum_weight <= 0.0 {
            return Err(ContractError::InvalidParticleMeasureWeights);
        }
        let scaled_total = weights
            .iter()
            .map(|weight| weight / maximum_weight)
            .sum::<f64>();
        if !scaled_total.is_finite() || scaled_total <= 0.0 {
            return Err(ContractError::InvalidParticleMeasureWeights);
        }

        let normalized_weights = weights
            .iter()
            .map(|weight| (weight / maximum_weight) / scaled_total)
            .collect::<Vec<_>>();
        let wrapped_positions = positions
            .iter()
            .map(|position| position.rem_euclid(1.0))
            .collect::<Vec<_>>();

        let mut moments = Vec::with_capacity(self.max_harmonic);
        for harmonic in 1..=self.max_harmonic {
            let mut real_sum = 0.0;
            let mut real_compensation = 0.0;
            let mut imag_sum = 0.0;
            let mut imag_compensation = 0.0;
            for (position, weight) in wrapped_positions.iter().zip(&normalized_weights) {
                let phase = TAU * harmonic as f64 * position;
                kahan_add(weight * phase.cos(), &mut real_sum, &mut real_compensation);
                kahan_add(weight * phase.sin(), &mut imag_sum, &mut imag_compensation);
            }
            moments.push(ComplexResponse::new(real_sum, imag_sum)?);
        }
        Ok(moments)
    }

    pub fn max_harmonic(&self) -> usize {
        self.max_harmonic
    }
}

/// Compensated summation for bounded empirical Fourier moments.
fn kahan_add(value: f64, sum: &mut f64, compensation: &mut f64) {
    let adjusted = value - *compensation;
    let next = *sum + adjusted;
    *compensation = (next - *sum) - adjusted;
    *sum = next;
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

/// Inputs for a memory-bounded spectral refinement decision.
///
/// The response risk is an upper bound in the normalized future-QoI metric.
/// None means the response is unresolved; it is never interpreted as zero risk.
/// projected_peak_bytes must include the padded state, FFT work buffers, split
/// stages and other temporary allocations required by the proposed refinement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralRefinementInput {
    pub spectral_tail_measure: f64,
    pub spectral_tail_threshold: f64,
    pub response_risk_upper_bound: Option<f64>,
    pub response_risk_threshold: f64,
    pub response_assessment_fresh: bool,
    pub response_branch_association_stable: bool,
    pub response_estimator_latency_seconds: f64,
    pub refinement_deadline_seconds: f64,
    pub projected_peak_bytes: u64,
    pub projected_peak_includes_temporary_buffers: bool,
    pub memory_budget_bytes: u64,
    pub reserved_headroom_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SpectralRefinementDecision {
    Hold,
    Refine,
    BlockedByMemory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SpectralRefinementReason {
    NoTrigger,
    SpectralTail,
    FutureResponse,
    Both,
    ResponseUnknown,
    SpectralTailAndResponseUnknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SpectralRefinementQualification {
    NoTrigger,
    Triggered,
    UnknownResponse,
    UnknownBlockedByMemory,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SpectralRefinementReport {
    pub decision: SpectralRefinementDecision,
    pub reason: SpectralRefinementReason,
    pub qualification: SpectralRefinementQualification,
    pub spectral_tail_ratio: f64,
    pub response_risk_upper_bound: Option<f64>,
    pub response_assessment_valid: bool,
    pub effective_memory_budget_bytes: u64,
    pub projected_peak_bytes: u64,
    pub projected_peak_includes_temporary_buffers: bool,
    pub remaining_headroom_bytes: u64,
}

/// Decide whether to refine using spectral-tail and future-response signals.
///
/// This is a policy primitive, not a physical error estimator. It assumes the
/// caller supplies a valid response upper bound where available and a
/// conservative projected peak-memory estimate. An unresolved response signal
/// triggers refinement if budget permits; if it does not, the result is
/// BLOCKED_BY_MEMORY / UNKNOWN, never a qualified hold.
pub fn decide_spectral_refinement(
    input: SpectralRefinementInput,
) -> Result<SpectralRefinementReport, ContractError> {
    if !input.spectral_tail_measure.is_finite()
        || input.spectral_tail_measure < 0.0
        || !input.spectral_tail_threshold.is_finite()
        || input.spectral_tail_threshold <= 0.0
        || !input.response_risk_threshold.is_finite()
        || input.response_risk_threshold <= 0.0
        || !input.response_estimator_latency_seconds.is_finite()
        || input.response_estimator_latency_seconds < 0.0
        || !input.refinement_deadline_seconds.is_finite()
        || input.refinement_deadline_seconds <= 0.0
        || input
            .response_risk_upper_bound
            .is_some_and(|risk| !risk.is_finite() || risk < 0.0)
    {
        return Err(ContractError::InvalidRefinementSignal);
    }
    if !input.projected_peak_includes_temporary_buffers {
        return Err(ContractError::IncompleteRefinementMemoryEstimate);
    }
    if input.reserved_headroom_bytes > input.memory_budget_bytes {
        return Err(ContractError::InvalidRefinementMemoryBudget);
    }

    let effective_memory_budget_bytes = input.memory_budget_bytes - input.reserved_headroom_bytes;
    let spectral_tail_ratio = input.spectral_tail_measure / input.spectral_tail_threshold;
    if !spectral_tail_ratio.is_finite() {
        return Err(ContractError::InvalidRefinementSignal);
    }
    let tail_triggered = spectral_tail_ratio >= 1.0;
    let response_unknown = input.response_risk_upper_bound.is_none()
        || !input.response_assessment_fresh
        || !input.response_branch_association_stable
        || input.response_estimator_latency_seconds > input.refinement_deadline_seconds;
    let response_triggered = !response_unknown
        && input
            .response_risk_upper_bound
            .is_some_and(|risk| risk >= input.response_risk_threshold);

    let reason = match (tail_triggered, response_triggered, response_unknown) {
        (true, true, _) => SpectralRefinementReason::Both,
        (true, false, true) => SpectralRefinementReason::SpectralTailAndResponseUnknown,
        (true, false, false) => SpectralRefinementReason::SpectralTail,
        (false, true, _) => SpectralRefinementReason::FutureResponse,
        (false, false, true) => SpectralRefinementReason::ResponseUnknown,
        (false, false, false) => SpectralRefinementReason::NoTrigger,
    };
    let must_refine = tail_triggered || response_triggered || response_unknown;

    if !must_refine {
        return Ok(SpectralRefinementReport {
            decision: SpectralRefinementDecision::Hold,
            reason,
            qualification: SpectralRefinementQualification::NoTrigger,
            spectral_tail_ratio,
            response_risk_upper_bound: input.response_risk_upper_bound,
            response_assessment_valid: !response_unknown,
            effective_memory_budget_bytes,
            projected_peak_bytes: input.projected_peak_bytes,
            projected_peak_includes_temporary_buffers: input
                .projected_peak_includes_temporary_buffers,
            remaining_headroom_bytes: effective_memory_budget_bytes
                .saturating_sub(input.projected_peak_bytes),
        });
    }

    if input.projected_peak_bytes > effective_memory_budget_bytes {
        return Ok(SpectralRefinementReport {
            decision: SpectralRefinementDecision::BlockedByMemory,
            reason,
            qualification: SpectralRefinementQualification::UnknownBlockedByMemory,
            spectral_tail_ratio,
            response_risk_upper_bound: input.response_risk_upper_bound,
            response_assessment_valid: !response_unknown,
            effective_memory_budget_bytes,
            projected_peak_bytes: input.projected_peak_bytes,
            projected_peak_includes_temporary_buffers: input
                .projected_peak_includes_temporary_buffers,
            remaining_headroom_bytes: 0,
        });
    }

    Ok(SpectralRefinementReport {
        decision: SpectralRefinementDecision::Refine,
        reason,
        qualification: if response_unknown {
            SpectralRefinementQualification::UnknownResponse
        } else {
            SpectralRefinementQualification::Triggered
        },
        spectral_tail_ratio,
        response_risk_upper_bound: input.response_risk_upper_bound,
        response_assessment_valid: !response_unknown,
        effective_memory_budget_bytes,
        projected_peak_bytes: input.projected_peak_bytes,
        projected_peak_includes_temporary_buffers: input.projected_peak_includes_temporary_buffers,
        remaining_headroom_bytes: effective_memory_budget_bytes - input.projected_peak_bytes,
    })
}

/// Audited timing envelope for a sparse, asynchronous kinetic-control interface.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SensorActuatorTimingSummary {
    pub contract_id: String,
    pub diagnostic_channel_count: usize,
    pub actuator_channel_count: usize,
    pub maximum_observation_age_seconds: f64,
    pub maximum_actuator_delivery_delay_seconds: f64,
    pub maximum_actuator_command_latency_seconds: f64,
    pub minimum_effective_actuator_bandwidth_hz: f64,
    pub conservative_closed_loop_latency_seconds: f64,
    pub minimum_effective_diagnostic_bandwidth_hz: f64,
    pub target_instability_frequency_hz: f64,
    pub diagnostic_bandwidth_multiple: f64,
    pub actuator_bandwidth_multiple: f64,
    pub target_instability_efold_time_seconds: f64,
    pub loop_latency_efold_fraction: f64,
    pub minimum_required_bandwidth_multiple: f64,
    pub maximum_allowed_loop_latency_efold_fraction: f64,
    pub interface_timescale_qualified: bool,
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

    let clock_skew = timing_number(value, "clock.maximum_clock_skew_seconds")?;
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
    let mut minimum_effective_diagnostic_bandwidth_hz = f64::INFINITY;
    for diagnostic in diagnostics {
        let channel_id = timing_string(diagnostic, "channel_id")?;
        if !diagnostic_ids.insert(channel_id.to_owned()) {
            return Err(control_timing_error(
                "diagnostics.channel_id",
                "diagnostic channel IDs must be unique",
            ));
        }
        let sampling_period = timing_number(diagnostic, "sampling_period_seconds")?;
        let measurement_bandwidth = timing_number(diagnostic, "bandwidth_hz")?;
        let transport_latency = timing_number(diagnostic, "transport_latency_seconds")?;
        let timestamp_age_bound = timing_number(diagnostic, "timestamp_age_bound_seconds")?;
        let jitter = timing_number(diagnostic, "jitter_bound_seconds")?;
        let noise_std = timing_number(diagnostic, "noise_std")?;
        if sampling_period <= 0.0
            || measurement_bandwidth <= 0.0
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
        if !minimum_age_bound.is_finite() || timestamp_age_bound + allowance < minimum_age_bound {
            return Err(control_timing_error(
                "diagnostics.timestamp_age_bound_seconds",
                "age bound must cover sampling period + transport latency + jitter + clock skew",
            ));
        }
        maximum_observation_age_seconds = maximum_observation_age_seconds.max(timestamp_age_bound);
        let sampling_nyquist_hz = 0.5 / sampling_period;
        minimum_effective_diagnostic_bandwidth_hz = minimum_effective_diagnostic_bandwidth_hz
            .min(measurement_bandwidth.min(sampling_nyquist_hz));
    }

    let mut actuator_ids = BTreeSet::new();
    let mut maximum_actuator_delivery_delay_seconds = 0.0_f64;
    let mut maximum_actuator_command_latency_seconds = 0.0_f64;
    let mut minimum_effective_actuator_bandwidth_hz = f64::INFINITY;
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
        maximum_actuator_command_latency_seconds = maximum_actuator_command_latency_seconds
            .max(update_period + delivery_delay + clock_skew);
        let command_nyquist_hz = 0.5 / update_period;
        minimum_effective_actuator_bandwidth_hz =
            minimum_effective_actuator_bandwidth_hz.min(bandwidth.min(command_nyquist_hz));
    }

    let controller_update_period = timing_number(value, "controller.update_period_seconds")?;
    let controller_wcet = timing_number(value, "controller.worst_case_execution_seconds")?;
    let controller_deadline = timing_number(value, "controller.deadline_seconds")?;
    let maximum_allowed_observation_age =
        timing_number(value, "controller.maximum_observation_age_seconds")?;
    let target_instability_frequency_hz =
        timing_number(value, "controller.target_instability_frequency_hz")?;
    let target_instability_efold_time_seconds =
        timing_number(value, "controller.target_instability_efold_time_seconds")?;
    let minimum_bandwidth_multiple = timing_number(value, "controller.minimum_bandwidth_multiple")?;
    let maximum_loop_latency_efold_fraction =
        timing_number(value, "controller.maximum_loop_latency_efold_fraction")?;
    if controller_update_period <= 0.0
        || controller_wcet < 0.0
        || controller_deadline <= 0.0
        || maximum_allowed_observation_age <= 0.0
        || target_instability_frequency_hz <= 0.0
        || target_instability_efold_time_seconds <= 0.0
        || minimum_bandwidth_multiple <= 0.0
        || maximum_loop_latency_efold_fraction <= 0.0
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

    let conservative_closed_loop_latency_seconds = maximum_observation_age_seconds
        + controller_wcet
        + maximum_actuator_command_latency_seconds;
    let diagnostic_bandwidth_multiple =
        minimum_effective_diagnostic_bandwidth_hz / target_instability_frequency_hz;
    let actuator_bandwidth_multiple =
        minimum_effective_actuator_bandwidth_hz / target_instability_frequency_hz;
    let loop_latency_efold_fraction =
        conservative_closed_loop_latency_seconds / target_instability_efold_time_seconds;
    let interface_timescale_qualified = diagnostic_bandwidth_multiple >= minimum_bandwidth_multiple
        && actuator_bandwidth_multiple >= minimum_bandwidth_multiple
        && loop_latency_efold_fraction <= maximum_loop_latency_efold_fraction;
    if !maximum_actuator_command_latency_seconds.is_finite()
        || !minimum_effective_actuator_bandwidth_hz.is_finite()
        || !minimum_effective_diagnostic_bandwidth_hz.is_finite()
        || !conservative_closed_loop_latency_seconds.is_finite()
        || !diagnostic_bandwidth_multiple.is_finite()
        || !actuator_bandwidth_multiple.is_finite()
        || !loop_latency_efold_fraction.is_finite()
    {
        return Err(control_timing_error(
            "actuators",
            "derived actuator and loop timing metrics must be finite",
        ));
    }

    Ok(SensorActuatorTimingSummary {
        contract_id,
        diagnostic_channel_count: diagnostics.len(),
        actuator_channel_count: actuators.len(),
        maximum_observation_age_seconds,
        maximum_actuator_delivery_delay_seconds,
        maximum_actuator_command_latency_seconds,
        minimum_effective_actuator_bandwidth_hz,
        conservative_closed_loop_latency_seconds,
        minimum_effective_diagnostic_bandwidth_hz,
        target_instability_frequency_hz,
        diagnostic_bandwidth_multiple,
        actuator_bandwidth_multiple,
        target_instability_efold_time_seconds,
        loop_latency_efold_fraction,
        minimum_required_bandwidth_multiple: minimum_bandwidth_multiple,
        maximum_allowed_loop_latency_efold_fraction: maximum_loop_latency_efold_fraction,
        interface_timescale_qualified,
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

/// One unstable scalar mode with asynchronous measurements and delayed actuation.
///
/// The reference discrete dynamics are
/// \`x[k+1] = exp(gamma*dt) * x[k] + b*u[k]\`.
/// Diagnostic sample j observes \`c*x[k_j] + noise\` and becomes available
/// after the declared integer-step delay. Actuator commands are applied on
/// their declared update schedule after their integer-step delivery delay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarUnstableModeContract {
    pub growth_rate_per_second: f64,
    pub oscillation_frequency_hz: f64,
    pub time_step_seconds: f64,
    pub horizon_steps: u32,
    pub diagnostic_gain: f64,
    pub diagnostic_noise_std: f64,
    pub diagnostic_bandwidth_hz: f64,
    pub diagnostic_sampling_stride_steps: u32,
    pub diagnostic_delivery_delay_steps: u32,
    pub actuator_gain: f64,
    pub actuator_bandwidth_hz: f64,
    pub actuator_update_stride_steps: u32,
    pub actuator_delivery_delay_steps: u32,
    pub controller_execution_latency_seconds: f64,
    pub minimum_bandwidth_multiple: f64,
    pub maximum_loop_latency_efold_fraction: f64,
    pub minimum_observability_fisher_information: f64,
    pub minimum_controllability_gramian: f64,
}

/// Exact scalar-mode observability / controllability and timing summary.
///
/// The observability value is the noise-whitened Fisher information for the
/// initial scalar state across samples that arrive within the horizon.
/// The controllability value is the finite-horizon discrete controllability
/// Gramian for unit-energy actuator impulses on the declared update schedule.
/// Neither scalar quantity is a stability proof for a coupled kinetic system.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ScalarUnstableModeReport {
    pub discrete_growth_factor_per_step: f64,
    pub observation_sample_count: u32,
    pub actuator_command_count: u32,
    pub observability_fisher_information: f64,
    pub controllability_gramian: f64,
    pub effective_diagnostic_bandwidth_hz: f64,
    pub effective_actuator_bandwidth_hz: f64,
    pub diagnostic_bandwidth_multiple: f64,
    pub actuator_bandwidth_multiple: f64,
    pub conservative_loop_latency_seconds: f64,
    pub loop_latency_efold_fraction: f64,
    pub diagnostic_delay_efold_fraction: f64,
    pub actuator_delay_efold_fraction: f64,
    pub minimum_required_bandwidth_multiple: f64,
    pub maximum_allowed_loop_latency_efold_fraction: f64,
    pub minimum_required_observability_fisher_information: f64,
    pub minimum_required_controllability_gramian: f64,
    pub mode_interface_qualified: bool,
}

/// Compute the exact finite-horizon scalar observability and controllability
/// Gramians for the declared asynchronous sensor and delayed actuator schedule.
///
/// This is a manufactured linear-mode oracle. It makes no claim about nonlinear
/// Vlasov-Poisson stabilizability or about a multi-mode controller.
pub fn evaluate_scalar_unstable_mode_interface(
    contract: ScalarUnstableModeContract,
) -> Result<ScalarUnstableModeReport, ContractError> {
    let positive_finite = [
        contract.growth_rate_per_second,
        contract.oscillation_frequency_hz,
        contract.time_step_seconds,
        contract.diagnostic_noise_std,
        contract.diagnostic_bandwidth_hz,
        contract.actuator_bandwidth_hz,
        contract.minimum_bandwidth_multiple,
        contract.maximum_loop_latency_efold_fraction,
        contract.minimum_observability_fisher_information,
        contract.minimum_controllability_gramian,
    ];
    if positive_finite
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || !contract.diagnostic_gain.is_finite()
        || contract.diagnostic_gain < 0.0
        || !contract.actuator_gain.is_finite()
        || contract.actuator_gain < 0.0
        || !contract.controller_execution_latency_seconds.is_finite()
        || contract.controller_execution_latency_seconds < 0.0
        || contract.diagnostic_sampling_stride_steps == 0
        || contract.actuator_update_stride_steps == 0
        || contract.horizon_steps == 0
    {
        return Err(control_timing_error(
            "scalar_mode",
            "growth, frequency, time step, noise, bandwidths and margins must be positive; gains/latency non-negative; strides and horizon positive",
        ));
    }
    if contract.horizon_steps > MAX_SCALAR_MODE_HORIZON_STEPS {
        return Err(control_timing_error(
            "scalar_mode.horizon_steps",
            "horizon exceeds bounded scalar-mode work budget",
        ));
    }

    let growth_exponent = contract.growth_rate_per_second * contract.time_step_seconds;
    if !growth_exponent.is_finite() {
        return Err(control_timing_error(
            "scalar_mode.growth_rate_per_second",
            "growth exponent must be finite",
        ));
    }
    let growth_factor = growth_exponent.exp();
    if !growth_factor.is_finite() || growth_factor <= 0.0 {
        return Err(control_timing_error(
            "scalar_mode.growth_rate_per_second",
            "discrete growth factor must be finite and positive",
        ));
    }

    let gain_to_noise = contract.diagnostic_gain / contract.diagnostic_noise_std;
    let information_scale = gain_to_noise * gain_to_noise;
    let actuator_scale = contract.actuator_gain * contract.actuator_gain;
    if !information_scale.is_finite() || !actuator_scale.is_finite() {
        return Err(control_timing_error(
            "scalar_mode",
            "squared gain/noise scales must be finite",
        ));
    }

    let mut observation_sample_count = 0_u32;
    let mut observability_fisher_information = 0.0_f64;
    let mut observation_compensation = 0.0_f64;
    let last_sample_step = contract
        .horizon_steps
        .saturating_sub(contract.diagnostic_delivery_delay_steps);
    let mut sample_step = 0_u32;
    while sample_step <= last_sample_step {
        let exponent = 2.0 * growth_exponent * sample_step as f64;
        let contribution = information_scale * exponent.exp();
        if !contribution.is_finite() {
            return Err(control_timing_error(
                "scalar_mode.observability_fisher_information",
                "finite-horizon observation information overflowed",
            ));
        }
        kahan_add(
            contribution,
            &mut observability_fisher_information,
            &mut observation_compensation,
        );
        observation_sample_count += 1;
        match sample_step.checked_add(contract.diagnostic_sampling_stride_steps) {
            Some(next) => sample_step = next,
            None => break,
        }
    }

    let mut actuator_command_count = 0_u32;
    let mut controllability_gramian = 0.0_f64;
    let mut actuator_compensation = 0.0_f64;
    let mut command_step = 0_u32;
    while let Some(applied_step) = command_step.checked_add(contract.actuator_delivery_delay_steps)
    {
        if applied_step >= contract.horizon_steps {
            break;
        }
        let remaining_steps = contract.horizon_steps - 1 - applied_step;
        let exponent = 2.0 * growth_exponent * remaining_steps as f64;
        let contribution = actuator_scale * exponent.exp();
        if !contribution.is_finite() {
            return Err(control_timing_error(
                "scalar_mode.controllability_gramian",
                "finite-horizon actuator Gramian overflowed",
            ));
        }
        kahan_add(
            contribution,
            &mut controllability_gramian,
            &mut actuator_compensation,
        );
        actuator_command_count += 1;
        match command_step.checked_add(contract.actuator_update_stride_steps) {
            Some(next) => command_step = next,
            None => break,
        }
    }

    let diagnostic_interval_seconds =
        contract.diagnostic_sampling_stride_steps as f64 * contract.time_step_seconds;
    let actuator_interval_seconds =
        contract.actuator_update_stride_steps as f64 * contract.time_step_seconds;
    let effective_diagnostic_bandwidth_hz = contract
        .diagnostic_bandwidth_hz
        .min(0.5 / diagnostic_interval_seconds);
    let effective_actuator_bandwidth_hz = contract
        .actuator_bandwidth_hz
        .min(0.5 / actuator_interval_seconds);
    let diagnostic_bandwidth_multiple =
        effective_diagnostic_bandwidth_hz / contract.oscillation_frequency_hz;
    let actuator_bandwidth_multiple =
        effective_actuator_bandwidth_hz / contract.oscillation_frequency_hz;
    let diagnostic_delay_seconds =
        contract.diagnostic_delivery_delay_steps as f64 * contract.time_step_seconds;
    let actuator_delay_seconds =
        contract.actuator_delivery_delay_steps as f64 * contract.time_step_seconds;
    let conservative_loop_latency_seconds = diagnostic_interval_seconds
        + diagnostic_delay_seconds
        + contract.controller_execution_latency_seconds
        + actuator_interval_seconds
        + actuator_delay_seconds;
    let loop_latency_efold_fraction =
        conservative_loop_latency_seconds * contract.growth_rate_per_second;
    let diagnostic_delay_efold_fraction =
        diagnostic_delay_seconds * contract.growth_rate_per_second;
    let actuator_delay_efold_fraction = actuator_delay_seconds * contract.growth_rate_per_second;
    let derived = [
        effective_diagnostic_bandwidth_hz,
        effective_actuator_bandwidth_hz,
        diagnostic_bandwidth_multiple,
        actuator_bandwidth_multiple,
        conservative_loop_latency_seconds,
        loop_latency_efold_fraction,
        diagnostic_delay_efold_fraction,
        actuator_delay_efold_fraction,
        observability_fisher_information,
        controllability_gramian,
    ];
    if derived.iter().any(|value| !value.is_finite()) {
        return Err(control_timing_error(
            "scalar_mode",
            "derived Gramian and timing metrics must be finite",
        ));
    }

    let mode_interface_qualified = observation_sample_count > 0
        && actuator_command_count > 0
        && observability_fisher_information >= contract.minimum_observability_fisher_information
        && controllability_gramian >= contract.minimum_controllability_gramian
        && diagnostic_bandwidth_multiple >= contract.minimum_bandwidth_multiple
        && actuator_bandwidth_multiple >= contract.minimum_bandwidth_multiple
        && loop_latency_efold_fraction <= contract.maximum_loop_latency_efold_fraction;

    Ok(ScalarUnstableModeReport {
        discrete_growth_factor_per_step: growth_factor,
        observation_sample_count,
        actuator_command_count,
        observability_fisher_information,
        controllability_gramian,
        effective_diagnostic_bandwidth_hz,
        effective_actuator_bandwidth_hz,
        diagnostic_bandwidth_multiple,
        actuator_bandwidth_multiple,
        conservative_loop_latency_seconds,
        loop_latency_efold_fraction,
        diagnostic_delay_efold_fraction,
        actuator_delay_efold_fraction,
        minimum_required_bandwidth_multiple: contract.minimum_bandwidth_multiple,
        maximum_allowed_loop_latency_efold_fraction: contract.maximum_loop_latency_efold_fraction,
        minimum_required_observability_fisher_information: contract
            .minimum_observability_fisher_information,
        minimum_required_controllability_gramian: contract.minimum_controllability_gramian,
        mode_interface_qualified,
    })
}

/// Delayed field and interval-mean current/gradient observations for the
/// one-dimensional transverse Maxwell subsystem.
///
/// Every vector must have the same non-zero spatial length. The current and
/// gradient vectors are interval means over the exact preceding delay window,
/// not arbitrary point samples.
#[derive(Clone, Debug, PartialEq)]
pub struct MaxwellDelayCancellationObservation {
    pub delayed_electric_longitudinal: Vec<f64>,
    pub delayed_electric_transverse: Vec<f64>,
    pub delayed_magnetic_transverse: Vec<f64>,
    pub interval_mean_current_longitudinal: Vec<f64>,
    pub interval_mean_current_transverse: Vec<f64>,
    pub interval_mean_gradient_electric_transverse: Vec<f64>,
    pub interval_mean_gradient_magnetic_transverse: Vec<f64>,
}

/// Configuration for a periodic, uniform-grid Fourier derivative estimator.
///
/// The regularizer is W_k = 1 / (1 + lambda * k^4), where k is the integer
/// Fourier mode. measurement_noise_variance is an explicit white, independent
/// sample-noise model; it is used only to predict the linear estimator output
/// variance and is not inferred from the field samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicDerivativeConfig {
    pub domain_length: f64,
    pub maximum_harmonic: usize,
    pub tikhonov_strength: f64,
    pub measurement_noise_variance: f64,
}

/// Periodic spectral derivative and its linear white-noise amplification report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PeriodicDerivativeEstimate {
    pub derivative: Vec<f64>,
    pub filter_weights: Vec<f64>,
    /// Variance gain G such that white sample noise variance sigma^2 maps to
    /// predicted derivative variance sigma^2 * G under the documented DFT convention.
    pub white_noise_variance_gain: f64,
    pub predicted_white_noise_variance: f64,
    pub sample_count: usize,
    pub maximum_harmonic: usize,
}

/// Estimate a periodic spatial derivative from uniformly sampled real data.
///
/// The input grid is x_j = j*L/N on a declared periodic interval of length L.
/// The mean/DC component is omitted from differentiation. A truncated real
/// Fourier series is reconstructed and differentiated analytically. The
/// regularizer W_k = 1/(1+lambda*k^4) is applied symmetrically to positive and
/// negative modes. The Nyquist mode is excluded because its real collocation
/// derivative is not represented by a unique sine/cosine pair.
///
/// For independent, zero-mean white sample noise with variance sigma^2, this
/// estimator has output variance sigma^2 * (2/N) * sum_k (2*pi*k/L)^2 W_k^2.
/// Correlated noise, nonuniform grids, open boundaries and unresolved physical
/// bandwidth are outside this contract.
pub fn estimate_periodic_derivative(
    samples: &[f64],
    config: PeriodicDerivativeConfig,
) -> Result<PeriodicDerivativeEstimate, ContractError> {
    let sample_count = samples.len();
    if !(3..=MAX_PERIODIC_DERIVATIVE_SAMPLES).contains(&sample_count) {
        return Err(ContractError::InvalidPeriodicDerivativeConfig(
            "sample count must be between 3 and the bounded maximum",
        ));
    }
    if samples.iter().any(|value| !value.is_finite()) {
        return Err(ContractError::InvalidPeriodicDerivativeConfig(
            "all periodic field samples must be finite",
        ));
    }
    if !config.domain_length.is_finite() || config.domain_length <= 0.0 {
        return Err(ContractError::InvalidPeriodicDerivativeConfig(
            "domain length must be finite and positive",
        ));
    }
    if config.maximum_harmonic == 0 || config.maximum_harmonic > (sample_count - 1) / 2 {
        return Err(ContractError::InvalidPeriodicDerivativeConfig(
            "maximum harmonic must be positive and strictly below the Nyquist mode",
        ));
    }
    if !config.tikhonov_strength.is_finite() || config.tikhonov_strength < 0.0 {
        return Err(ContractError::InvalidPeriodicDerivativeConfig(
            "Tikhonov strength must be finite and non-negative",
        ));
    }
    if !config.measurement_noise_variance.is_finite() || config.measurement_noise_variance < 0.0 {
        return Err(ContractError::InvalidPeriodicDerivativeConfig(
            "white measurement-noise variance must be finite and non-negative",
        ));
    }
    let transform_work = sample_count
        .checked_mul(config.maximum_harmonic)
        .and_then(|work| work.checked_mul(2))
        .ok_or(ContractError::PeriodicDerivativeWorkBudgetExceeded)?;
    if transform_work > MAX_PERIODIC_DERIVATIVE_WORK {
        return Err(ContractError::PeriodicDerivativeWorkBudgetExceeded);
    }

    let mut cosine_coefficients = Vec::with_capacity(config.maximum_harmonic);
    let mut sine_coefficients = Vec::with_capacity(config.maximum_harmonic);
    let mut filter_weights = Vec::with_capacity(config.maximum_harmonic);
    let mut white_noise_variance_gain = 0.0_f64;
    let mut gain_compensation = 0.0_f64;

    for harmonic in 1..=config.maximum_harmonic {
        let k_squared = (harmonic as f64) * (harmonic as f64);
        let penalty = config.tikhonov_strength * k_squared * k_squared;
        let denominator = 1.0 + penalty;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(ContractError::InvalidPeriodicDerivativeConfig(
                "regularization filter arithmetic overflowed",
            ));
        }
        let weight = 1.0 / denominator;
        let mut cosine_sum = 0.0_f64;
        let mut cosine_compensation = 0.0_f64;
        let mut sine_sum = 0.0_f64;
        let mut sine_compensation = 0.0_f64;

        for (sample_index, sample) in samples.iter().enumerate() {
            let phase = TAU * harmonic as f64 * sample_index as f64 / sample_count as f64;
            kahan_add(
                *sample * phase.cos(),
                &mut cosine_sum,
                &mut cosine_compensation,
            );
            kahan_add(*sample * phase.sin(), &mut sine_sum, &mut sine_compensation);
        }

        let cosine_coefficient = (2.0 / sample_count as f64) * cosine_sum * weight;
        let sine_coefficient = (2.0 / sample_count as f64) * sine_sum * weight;
        if !cosine_coefficient.is_finite() || !sine_coefficient.is_finite() {
            return Err(ContractError::InvalidPeriodicDerivativeConfig(
                "Fourier coefficient arithmetic produced a non-finite value",
            ));
        }
        cosine_coefficients.push(cosine_coefficient);
        sine_coefficients.push(sine_coefficient);
        filter_weights.push(weight);

        let angular_frequency = TAU * harmonic as f64 / config.domain_length;
        let gain_term =
            (2.0 / sample_count as f64) * angular_frequency * angular_frequency * weight * weight;
        if !gain_term.is_finite() {
            return Err(ContractError::InvalidPeriodicDerivativeConfig(
                "white-noise gain arithmetic produced a non-finite value",
            ));
        }
        kahan_add(
            gain_term,
            &mut white_noise_variance_gain,
            &mut gain_compensation,
        );
    }

    let mut derivative = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        let mut value = 0.0_f64;
        let mut compensation = 0.0_f64;
        for harmonic_index in 0..config.maximum_harmonic {
            let harmonic = (harmonic_index + 1) as f64;
            let phase = TAU * harmonic * sample_index as f64 / sample_count as f64;
            let angular_frequency = TAU * harmonic / config.domain_length;
            let mode_derivative = angular_frequency
                * (-cosine_coefficients[harmonic_index] * phase.sin()
                    + sine_coefficients[harmonic_index] * phase.cos());
            kahan_add(mode_derivative, &mut value, &mut compensation);
        }
        if !value.is_finite() {
            return Err(ContractError::InvalidPeriodicDerivativeConfig(
                "reconstructed derivative is non-finite",
            ));
        }
        derivative.push(value);
    }

    let predicted_white_noise_variance =
        config.measurement_noise_variance * white_noise_variance_gain;
    if !white_noise_variance_gain.is_finite() || !predicted_white_noise_variance.is_finite() {
        return Err(ContractError::InvalidPeriodicDerivativeConfig(
            "predicted white-noise variance is non-finite",
        ));
    }

    Ok(PeriodicDerivativeEstimate {
        derivative,
        filter_weights,
        white_noise_variance_gain,
        predicted_white_noise_variance,
        sample_count,
        maximum_harmonic: config.maximum_harmonic,
    })
}

/// Time-stamped current and transverse field-gradient history over a delay window.
///
/// Each measurement is stored as [time_index][spatial_index]. The first and
/// last timestamps must exactly match the requested integration interval; the
/// integrator will not extrapolate missing history or silently change the
/// delay window.
#[derive(Clone, Debug, PartialEq)]
pub struct MaxwellDelayHistorySamples {
    pub sample_times_seconds: Vec<f64>,
    pub current_longitudinal: Vec<Vec<f64>>,
    pub current_transverse: Vec<Vec<f64>>,
    pub gradient_electric_transverse: Vec<Vec<f64>>,
    pub gradient_magnetic_transverse: Vec<Vec<f64>>,
}

/// Time-averaged current and field-gradient observations over a delay window.
#[derive(Clone, Debug, PartialEq)]
pub struct MaxwellDelayIntervalAverages {
    pub current_longitudinal: Vec<f64>,
    pub current_transverse: Vec<f64>,
    pub gradient_electric_transverse: Vec<f64>,
    pub gradient_magnetic_transverse: Vec<f64>,
}

/// Integrate sampled currents and field gradients over an exact delay window.
///
/// Nonuniform sample times are supported through piecewise-linear trapezoidal
/// integration. Every history must share the same time and spatial axes.
/// Incomplete windows, repeated/decreasing timestamps, ragged arrays, non-finite
/// data and excessive time-space work are rejected rather than extrapolated.
pub fn integrate_maxwell_delay_history(
    history: &MaxwellDelayHistorySamples,
    window_start_seconds: f64,
    window_end_seconds: f64,
) -> Result<MaxwellDelayIntervalAverages, ContractError> {
    if !window_start_seconds.is_finite()
        || !window_end_seconds.is_finite()
        || window_end_seconds <= window_start_seconds
    {
        return Err(control_timing_error(
            "delay_window",
            "window endpoints must be finite and strictly increasing",
        ));
    }

    let time_count = history.sample_times_seconds.len();
    if !(2..=MAX_DELAY_CANCELLATION_SAMPLES).contains(&time_count) {
        return Err(control_timing_error(
            "delay_history",
            "history must contain at least two and no more than the bounded number of samples",
        ));
    }
    if history.sample_times_seconds[0] != window_start_seconds
        || history.sample_times_seconds[time_count - 1] != window_end_seconds
    {
        return Err(control_timing_error(
            "delay_history",
            "history must include exact samples at both delay-window endpoints",
        ));
    }
    if history
        .sample_times_seconds
        .iter()
        .any(|time| !time.is_finite())
        || history
            .sample_times_seconds
            .windows(2)
            .any(|pair| pair[1] <= pair[0])
    {
        return Err(control_timing_error(
            "delay_history.sample_times_seconds",
            "timestamps must be finite and strictly increasing",
        ));
    }

    let spatial_count = history.current_longitudinal.first().map_or(0, Vec::len);
    let time_space_work = time_count
        .checked_mul(spatial_count)
        .ok_or_else(|| control_timing_error("delay_history", "time-space work overflowed"))?;
    let total_field_work = time_space_work
        .checked_mul(4)
        .ok_or_else(|| control_timing_error("delay_history", "four-field work overflowed"))?;
    if spatial_count == 0 || spatial_count > MAX_DELAY_CANCELLATION_SAMPLES {
        return Err(control_timing_error(
            "delay_history",
            "spatial axis must be non-empty and within the sample budget",
        ));
    }
    if total_field_work > MAX_DELAY_HISTORY_WORK {
        return Err(control_timing_error(
            "delay_history",
            "four-field time-space history exceeds the bounded integration budget",
        ));
    }

    for (field, values) in [
        ("current_longitudinal", &history.current_longitudinal),
        ("current_transverse", &history.current_transverse),
        (
            "gradient_electric_transverse",
            &history.gradient_electric_transverse,
        ),
        (
            "gradient_magnetic_transverse",
            &history.gradient_magnetic_transverse,
        ),
    ] {
        if values.len() != time_count
            || values
                .iter()
                .any(|row| row.len() != spatial_count || row.iter().any(|value| !value.is_finite()))
        {
            return Err(control_timing_error(
                field,
                "every time row must match the common spatial axis and contain finite values",
            ));
        }
    }

    let interval_duration = window_end_seconds - window_start_seconds;
    if !interval_duration.is_finite() || interval_duration <= 0.0 {
        return Err(control_timing_error(
            "delay_window",
            "window duration must be finite and positive",
        ));
    }

    let current_longitudinal = integrate_history_field(
        &history.sample_times_seconds,
        &history.current_longitudinal,
        interval_duration,
    )?;
    let current_transverse = integrate_history_field(
        &history.sample_times_seconds,
        &history.current_transverse,
        interval_duration,
    )?;
    let gradient_electric_transverse = integrate_history_field(
        &history.sample_times_seconds,
        &history.gradient_electric_transverse,
        interval_duration,
    )?;
    let gradient_magnetic_transverse = integrate_history_field(
        &history.sample_times_seconds,
        &history.gradient_magnetic_transverse,
        interval_duration,
    )?;

    Ok(MaxwellDelayIntervalAverages {
        current_longitudinal,
        current_transverse,
        gradient_electric_transverse,
        gradient_magnetic_transverse,
    })
}

fn integrate_history_field(
    sample_times: &[f64],
    values: &[Vec<f64>],
    interval_duration: f64,
) -> Result<Vec<f64>, ContractError> {
    let spatial_count = values.first().map_or(0, Vec::len);
    let mut integrals = vec![0.0_f64; spatial_count];
    let mut compensations = vec![0.0_f64; spatial_count];
    for time_index in 0..sample_times.len() - 1 {
        let delta_time = sample_times[time_index + 1] - sample_times[time_index];
        for (spatial_index, (left, right)) in values[time_index]
            .iter()
            .zip(&values[time_index + 1])
            .enumerate()
        {
            let segment_integral = 0.5 * (*left + *right) * delta_time;
            if !segment_integral.is_finite() {
                return Err(control_timing_error(
                    "delay_history",
                    "trapezoidal integration produced a non-finite segment",
                ));
            }
            kahan_add(
                segment_integral,
                &mut integrals[spatial_index],
                &mut compensations[spatial_index],
            );
        }
    }
    let mut averages = Vec::with_capacity(spatial_count);
    for integral in integrals {
        let average = integral / interval_duration;
        if !average.is_finite() {
            return Err(control_timing_error(
                "delay_history",
                "interval mean is non-finite",
            ));
        }
        averages.push(average);
    }
    Ok(averages)
}

/// Deterministic cancellation commands predicted from the Maxwell delay identities.
///
/// The fields are the baseline command only. Any learned or optimized residual
/// must be added as a separate, independently bounded component.
#[derive(Clone, Debug, PartialEq)]
pub struct MaxwellDelayCancellationCommand {
    pub external_electric_longitudinal: Vec<f64>,
    pub external_electric_transverse: Vec<f64>,
    pub external_magnetic_transverse: Vec<f64>,
}

/// First-order / interval-mean delay compensation for the 1D2V Maxwell field
/// equations. With exact interval means, these are the exact delayed-to-current
/// field-cancellation identities:
///
/// - H1 = -E1(t-tau) + tau * mean(J1)
/// - H2 = -E2(t-tau) + tau * mean(J2 + d_x B3)
/// - Bext = -B3(t-tau) + tau * mean(d_x E2)
///
/// The deterministic predictor is deliberately independent of any learned
/// residual operator. It does not validate sensor noise, derivative bias,
/// actuation feasibility, or the kinetic model that supplied the observations.
pub fn predict_maxwell_delay_cancellation(
    observation: &MaxwellDelayCancellationObservation,
    delay_seconds: f64,
) -> Result<MaxwellDelayCancellationCommand, ContractError> {
    if !delay_seconds.is_finite() || delay_seconds < 0.0 {
        return Err(control_timing_error(
            "delay_seconds",
            "must be finite and non-negative",
        ));
    }

    let lengths = [
        observation.delayed_electric_longitudinal.len(),
        observation.delayed_electric_transverse.len(),
        observation.delayed_magnetic_transverse.len(),
        observation.interval_mean_current_longitudinal.len(),
        observation.interval_mean_current_transverse.len(),
        observation.interval_mean_gradient_electric_transverse.len(),
        observation.interval_mean_gradient_magnetic_transverse.len(),
    ];
    let sample_count = lengths[0];
    if sample_count == 0 || lengths.iter().any(|length| *length != sample_count) {
        return Err(control_timing_error(
            "field_observations",
            "all seven spatial arrays must have equal, non-zero lengths",
        ));
    }
    if sample_count > MAX_DELAY_CANCELLATION_SAMPLES {
        return Err(control_timing_error(
            "field_observations",
            "spatial sample count exceeds the bounded work budget",
        ));
    }

    let all_values = observation
        .delayed_electric_longitudinal
        .iter()
        .chain(observation.delayed_electric_transverse.iter())
        .chain(observation.delayed_magnetic_transverse.iter())
        .chain(observation.interval_mean_current_longitudinal.iter())
        .chain(observation.interval_mean_current_transverse.iter())
        .chain(
            observation
                .interval_mean_gradient_electric_transverse
                .iter(),
        )
        .chain(
            observation
                .interval_mean_gradient_magnetic_transverse
                .iter(),
        );
    if all_values.clone().any(|value| !value.is_finite()) {
        return Err(control_timing_error(
            "field_observations",
            "all field, current and gradient observations must be finite",
        ));
    }

    let mut external_electric_longitudinal = Vec::with_capacity(sample_count);
    let mut external_electric_transverse = Vec::with_capacity(sample_count);
    let mut external_magnetic_transverse = Vec::with_capacity(sample_count);

    for index in 0..sample_count {
        let h1 = -observation.delayed_electric_longitudinal[index]
            + delay_seconds * observation.interval_mean_current_longitudinal[index];
        let h2 = -observation.delayed_electric_transverse[index]
            + delay_seconds
                * (observation.interval_mean_current_transverse[index]
                    + observation.interval_mean_gradient_magnetic_transverse[index]);
        let b_ext = -observation.delayed_magnetic_transverse[index]
            + delay_seconds * observation.interval_mean_gradient_electric_transverse[index];

        if !h1.is_finite() || !h2.is_finite() || !b_ext.is_finite() {
            return Err(control_timing_error(
                "delay_compensation",
                "delay-integral arithmetic produced a non-finite command",
            ));
        }
        external_electric_longitudinal.push(h1);
        external_electric_transverse.push(h2);
        external_magnetic_transverse.push(b_ext);
    }

    Ok(MaxwellDelayCancellationCommand {
        external_electric_longitudinal,
        external_electric_transverse,
        external_magnetic_transverse,
    })
}

/// Two-real-mode authority contract with two sampled diagnostics and one actuator direction.
///
/// The full two-state space is treated as the declared unstable subspace. Both
/// eigenvalues of the discrete transition matrix must lie outside the unit
/// circle. This is an analytic interface oracle, not a kinetic linearization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoupledTwoModeAuthorityContract {
    /// Row-major discrete transition matrix [a00, a01, a10, a11].
    pub transition_matrix: [f64; 4],
    pub horizon_steps: u32,
    /// Two diagnostic rows, each observing both state coordinates.
    pub diagnostic_gain: [[f64; 2]; 2],
    /// Independent positive measurement-noise variances for the two channels.
    pub diagnostic_noise_variance: [f64; 2],
    pub diagnostic_sampling_stride_steps: u32,
    pub diagnostic_delivery_delay_steps: u32,
    pub actuator_gain: [f64; 2],
    pub actuator_update_stride_steps: u32,
    pub actuator_delivery_delay_steps: u32,
    pub minimum_observability_eigenvalue: f64,
    pub minimum_controllability_eigenvalue: f64,
}

/// Finite-horizon authority report on the complete two-dimensional unstable subspace.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CoupledTwoModeAuthorityReport {
    pub minimum_unstable_eigenvalue_modulus: f64,
    pub spectral_radius: f64,
    pub observation_sample_count: u32,
    pub actuator_command_count: u32,
    /// Symmetric 2x2 matrices stored as [m00, m01, m11].
    pub observability_gramian: [f64; 3],
    pub observability_min_eigenvalue: f64,
    pub observability_max_eigenvalue: f64,
    pub observability_condition_number: Option<f64>,
    pub controllability_gramian: [f64; 3],
    pub controllability_min_eigenvalue: f64,
    pub controllability_max_eigenvalue: f64,
    pub controllability_condition_number: Option<f64>,
    pub observable: bool,
    pub controllable: bool,
    pub authority_qualified: bool,
}

/// Compute exact finite-horizon two-state observability and controllability
/// Gramians under the declared asynchronous sample / command schedules.
///
/// Samples are admitted only if their delivery completes within the horizon.
/// Commands contribute only after their delivery time and are propagated to
/// the terminal step. Noise variance whitens the scalar observation Gramian.
/// The report is restricted to the full two-state unstable subspace; no stable
/// directions can inflate its authority score.
pub fn evaluate_coupled_two_mode_authority(
    contract: CoupledTwoModeAuthorityContract,
) -> Result<CoupledTwoModeAuthorityReport, ContractError> {
    let a = contract.transition_matrix;
    if a.iter().any(|value| !value.is_finite())
        || contract
            .diagnostic_gain
            .iter()
            .flatten()
            .chain(contract.actuator_gain.iter())
            .any(|value| !value.is_finite())
        || contract
            .diagnostic_noise_variance
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || !contract.minimum_observability_eigenvalue.is_finite()
        || contract.minimum_observability_eigenvalue <= 0.0
        || !contract.minimum_controllability_eigenvalue.is_finite()
        || contract.minimum_controllability_eigenvalue <= 0.0
        || contract.horizon_steps < 2
        || contract.horizon_steps > MAX_SCALAR_MODE_HORIZON_STEPS
        || contract.diagnostic_sampling_stride_steps == 0
        || contract.actuator_update_stride_steps == 0
    {
        return Err(ContractError::InvalidCoupledModeAuthority(
            "finite matrices, positive noise / margins, bounded horizon and positive strides are required",
        ));
    }

    let trace = a[0] + a[3];
    let determinant = a[0].mul_add(a[3], -(a[1] * a[2]));
    let discriminant = trace.mul_add(trace, -4.0 * determinant);
    if !trace.is_finite() || !determinant.is_finite() || !discriminant.is_finite() {
        return Err(ContractError::InvalidCoupledModeAuthority(
            "transition eigenvalue arithmetic is non-finite",
        ));
    }
    let (minimum_modulus, spectral_radius) = if discriminant >= 0.0 {
        let root = discriminant.sqrt();
        let lambda_a = 0.5 * (trace + root);
        let lambda_b = 0.5 * (trace - root);
        (
            lambda_a.abs().min(lambda_b.abs()),
            lambda_a.abs().max(lambda_b.abs()),
        )
    } else {
        let modulus = determinant.sqrt();
        (modulus, modulus)
    };
    if !minimum_modulus.is_finite() || !spectral_radius.is_finite() || minimum_modulus <= 1.0 {
        return Err(ContractError::InvalidCoupledModeAuthority(
            "both declared modes must be unstable (eigenvalue modulus > 1)",
        ));
    }

    let mut observability = [0.0_f64; 3];
    let mut observability_compensation = [0.0_f64; 3];
    let mut observation_sample_count = 0_u32;
    let mut power = [1.0, 0.0, 0.0, 1.0];
    let mut sample_step = 0_u32;
    loop {
        let delivered_step = sample_step
            .checked_add(contract.diagnostic_delivery_delay_steps)
            .ok_or(ContractError::InvalidCoupledModeAuthority(
                "diagnostic delivery step overflowed",
            ))?;
        if delivered_step >= contract.horizon_steps {
            break;
        }
        for channel in 0..2 {
            let gain = contract.diagnostic_gain[channel];
            let c0 = gain[0] * power[0] + gain[1] * power[2];
            let c1 = gain[0] * power[1] + gain[1] * power[3];
            let inverse_variance = 1.0 / contract.diagnostic_noise_variance[channel];
            let contribution = [
                c0 * c0 * inverse_variance,
                c0 * c1 * inverse_variance,
                c1 * c1 * inverse_variance,
            ];
            if contribution.iter().any(|value| !value.is_finite()) {
                return Err(ContractError::InvalidCoupledModeAuthority(
                    "observability Gramian overflowed",
                ));
            }
            for index in 0..3 {
                kahan_add(
                    contribution[index],
                    &mut observability[index],
                    &mut observability_compensation[index],
                );
            }
        }
        observation_sample_count = observation_sample_count.saturating_add(1);
        let next_sample_step =
            match sample_step.checked_add(contract.diagnostic_sampling_stride_steps) {
                Some(next) => next,
                None => break,
            };
        let next_delivery_step = next_sample_step
            .checked_add(contract.diagnostic_delivery_delay_steps)
            .ok_or(ContractError::InvalidCoupledModeAuthority(
                "diagnostic delivery step overflowed",
            ))?;
        if next_delivery_step >= contract.horizon_steps {
            break;
        }
        sample_step = next_sample_step;
        let sample_transition = matrix_power_2x2(a, contract.diagnostic_sampling_stride_steps)?;
        power = multiply_mat2(power, sample_transition)?;
    }

    let mut controllability = [0.0_f64; 3];
    let mut controllability_compensation = [0.0_f64; 3];
    let mut actuator_command_count = 0_u32;
    let mut command_step = 0_u32;
    while command_step < contract.horizon_steps {
        let applied_step = command_step
            .checked_add(contract.actuator_delivery_delay_steps)
            .ok_or(ContractError::InvalidCoupledModeAuthority(
                "actuator delivery step overflowed",
            ))?;
        if applied_step >= contract.horizon_steps {
            break;
        }
        let remaining_steps = contract.horizon_steps - 1 - applied_step;
        let propagation = matrix_power_2x2(a, remaining_steps)?;
        let propagated_b = multiply_mat2_vec(propagation, contract.actuator_gain);
        let contribution = [
            propagated_b[0] * propagated_b[0],
            propagated_b[0] * propagated_b[1],
            propagated_b[1] * propagated_b[1],
        ];
        if contribution.iter().any(|value| !value.is_finite()) {
            return Err(ContractError::InvalidCoupledModeAuthority(
                "controllability Gramian overflowed",
            ));
        }
        for index in 0..3 {
            kahan_add(
                contribution[index],
                &mut controllability[index],
                &mut controllability_compensation[index],
            );
        }
        actuator_command_count = actuator_command_count.saturating_add(1);
        command_step = match command_step.checked_add(contract.actuator_update_stride_steps) {
            Some(next) => next,
            None => break,
        };
    }

    if observation_sample_count == 0 || actuator_command_count == 0 {
        return Err(ContractError::InvalidCoupledModeAuthority(
            "no diagnostic sample or actuator command arrives within the horizon",
        ));
    }

    let (observability_min, observability_max, observability_condition) =
        symmetric_2x2_spectrum(observability)?;
    let (controllability_min, controllability_max, controllability_condition) =
        symmetric_2x2_spectrum(controllability)?;
    let observable = observability_min >= contract.minimum_observability_eigenvalue;
    let controllable = controllability_min >= contract.minimum_controllability_eigenvalue;

    Ok(CoupledTwoModeAuthorityReport {
        minimum_unstable_eigenvalue_modulus: minimum_modulus,
        spectral_radius,
        observation_sample_count,
        actuator_command_count,
        observability_gramian: observability,
        observability_min_eigenvalue: observability_min,
        observability_max_eigenvalue: observability_max,
        observability_condition_number: observability_condition,
        controllability_gramian: controllability,
        controllability_min_eigenvalue: controllability_min,
        controllability_max_eigenvalue: controllability_max,
        controllability_condition_number: controllability_condition,
        observable,
        controllable,
        authority_qualified: observable && controllable,
    })
}

fn multiply_mat2(left: [f64; 4], right: [f64; 4]) -> Result<[f64; 4], ContractError> {
    let product = [
        left[0].mul_add(right[0], left[1] * right[2]),
        left[0].mul_add(right[1], left[1] * right[3]),
        left[2].mul_add(right[0], left[3] * right[2]),
        left[2].mul_add(right[1], left[3] * right[3]),
    ];
    if product.iter().any(|value| !value.is_finite()) {
        return Err(ContractError::InvalidCoupledModeAuthority(
            "matrix power overflowed",
        ));
    }
    Ok(product)
}

fn multiply_mat2_vec(matrix: [f64; 4], vector: [f64; 2]) -> [f64; 2] {
    [
        matrix[0].mul_add(vector[0], matrix[1] * vector[1]),
        matrix[2].mul_add(vector[0], matrix[3] * vector[1]),
    ]
}

fn matrix_power_2x2(matrix: [f64; 4], exponent: u32) -> Result<[f64; 4], ContractError> {
    let mut result = [1.0, 0.0, 0.0, 1.0];
    let mut factor = matrix;
    let mut power = exponent;
    while power > 0 {
        if power & 1 == 1 {
            result = multiply_mat2(result, factor)?;
        }
        power >>= 1;
        if power > 0 {
            factor = multiply_mat2(factor, factor)?;
        }
    }
    Ok(result)
}

fn symmetric_2x2_spectrum(matrix: [f64; 3]) -> Result<(f64, f64, Option<f64>), ContractError> {
    let [a, b, d] = matrix;
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(ContractError::InvalidCoupledModeAuthority(
            "Gramian matrix is non-finite",
        ));
    }
    let discriminant = (a - d).mul_add(a - d, 4.0 * b * b);
    let root = discriminant.sqrt();
    if !discriminant.is_finite() || !root.is_finite() {
        return Err(ContractError::InvalidCoupledModeAuthority(
            "Gramian eigenvalue arithmetic overflowed",
        ));
    }
    let minimum = (0.5 * (a + d - root)).max(0.0);
    let maximum = (0.5 * (a + d + root)).max(0.0);
    if !minimum.is_finite() || !maximum.is_finite() {
        return Err(ContractError::InvalidCoupledModeAuthority(
            "Gramian eigenvalues are non-finite",
        ));
    }
    let condition = if minimum > 0.0 {
        let ratio = maximum / minimum;
        ratio.is_finite().then_some(ratio)
    } else {
        None
    };
    Ok((minimum, maximum, condition))
}

/// Two-quadrature real representation of one complex growing kinetic mode.
///
/// The unforced dynamics are
/// \`x[k+1] = exp(gamma*dt) * R(omega*dt) * x[k]\`,
/// where \`x=[Re(a), Im(a)]\` and R is a planar rotation. A diagnostic observes
/// a fixed real row vector; an actuator injects a fixed two-component vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OscillatoryUnstableModeContract {
    pub growth_rate_per_second: f64,
    pub oscillation_frequency_hz: f64,
    pub time_step_seconds: f64,
    pub horizon_steps: u32,
    pub diagnostic_gain: [f64; 2],
    pub diagnostic_noise_std: f64,
    pub diagnostic_bandwidth_hz: f64,
    pub diagnostic_sampling_stride_steps: u32,
    pub diagnostic_delivery_delay_steps: u32,
    pub actuator_gain: [f64; 2],
    pub actuator_bandwidth_hz: f64,
    pub actuator_update_stride_steps: u32,
    pub actuator_delivery_delay_steps: u32,
    pub controller_execution_latency_seconds: f64,
    pub minimum_bandwidth_multiple: f64,
    pub maximum_loop_latency_efold_fraction: f64,
    pub minimum_observability_min_eigenvalue: f64,
    pub minimum_controllability_min_eigenvalue: f64,
}

/// Matrix-valued finite-horizon sensing and actuation scores for one complex mode.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct OscillatoryUnstableModeReport {
    pub discrete_growth_factor_per_step: f64,
    pub phase_advance_radians_per_step: f64,
    pub observation_sample_count: u32,
    pub actuator_command_count: u32,
    pub observability_gramian: [f64; 3],
    pub observability_min_eigenvalue: f64,
    pub observability_max_eigenvalue: f64,
    pub controllability_gramian: [f64; 3],
    pub controllability_min_eigenvalue: f64,
    pub controllability_max_eigenvalue: f64,
    pub effective_diagnostic_bandwidth_hz: f64,
    pub effective_actuator_bandwidth_hz: f64,
    pub diagnostic_bandwidth_multiple: f64,
    pub actuator_bandwidth_multiple: f64,
    pub conservative_loop_latency_seconds: f64,
    pub loop_latency_efold_fraction: f64,
    pub diagnostic_delay_efold_fraction: f64,
    pub actuator_delay_efold_fraction: f64,
    pub minimum_required_bandwidth_multiple: f64,
    pub maximum_allowed_loop_latency_efold_fraction: f64,
    pub minimum_required_observability_min_eigenvalue: f64,
    pub minimum_required_controllability_min_eigenvalue: f64,
    pub mode_interface_qualified: bool,
}

/// Compute exact discrete finite-horizon 2x2 observability and controllability
/// Gramians for a manufactured complex growing mode with asynchronous sensing
/// and delayed, sparse actuation.
///
/// This model captures both quadratures and therefore detects sampling/actuation
/// schedules that alias the oscillation phase. It remains a single-mode linear
/// oracle, not a multi-mode Vlasov controller or nonlinear stability proof.
pub fn evaluate_oscillatory_unstable_mode_interface(
    contract: OscillatoryUnstableModeContract,
) -> Result<OscillatoryUnstableModeReport, ContractError> {
    let positive_finite = [
        contract.growth_rate_per_second,
        contract.oscillation_frequency_hz,
        contract.time_step_seconds,
        contract.diagnostic_noise_std,
        contract.diagnostic_bandwidth_hz,
        contract.actuator_bandwidth_hz,
        contract.minimum_bandwidth_multiple,
        contract.maximum_loop_latency_efold_fraction,
        contract.minimum_observability_min_eigenvalue,
        contract.minimum_controllability_min_eigenvalue,
    ];
    if positive_finite
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || contract
            .diagnostic_gain
            .iter()
            .any(|value| !value.is_finite())
        || contract
            .actuator_gain
            .iter()
            .any(|value| !value.is_finite())
        || !contract.controller_execution_latency_seconds.is_finite()
        || contract.controller_execution_latency_seconds < 0.0
        || contract.diagnostic_sampling_stride_steps == 0
        || contract.actuator_update_stride_steps == 0
        || contract.horizon_steps == 0
    {
        return Err(control_timing_error(
            "oscillatory_mode",
            "growth, frequency, time step, noise, bandwidths and margins must be positive; gains/latency finite; strides and horizon positive",
        ));
    }
    if contract.horizon_steps > MAX_SCALAR_MODE_HORIZON_STEPS {
        return Err(control_timing_error(
            "oscillatory_mode.horizon_steps",
            "horizon exceeds bounded mode-oracle work budget",
        ));
    }

    let growth_exponent = contract.growth_rate_per_second * contract.time_step_seconds;
    let phase_step = TAU * contract.oscillation_frequency_hz * contract.time_step_seconds;
    if !growth_exponent.is_finite() || !phase_step.is_finite() {
        return Err(control_timing_error(
            "oscillatory_mode",
            "per-step growth and phase must be finite",
        ));
    }
    let growth_factor = growth_exponent.exp();
    if !growth_factor.is_finite() || growth_factor <= 0.0 {
        return Err(control_timing_error(
            "oscillatory_mode.growth_rate_per_second",
            "discrete growth factor must be finite and positive",
        ));
    }

    let inverse_noise_variance =
        1.0 / (contract.diagnostic_noise_std * contract.diagnostic_noise_std);
    if !inverse_noise_variance.is_finite() {
        return Err(control_timing_error(
            "oscillatory_mode.diagnostic_noise_std",
            "inverse noise variance must be finite",
        ));
    }

    // Symmetric matrices are stored as [m00, m01, m11].
    let mut observability = [0.0_f64; 3];
    let mut observability_compensation = [0.0_f64; 3];
    let mut observation_sample_count = 0_u32;
    let last_sample_step = contract
        .horizon_steps
        .saturating_sub(contract.diagnostic_delivery_delay_steps);
    let mut sample_step = 0_u32;
    while sample_step <= last_sample_step {
        let exponent = growth_exponent * sample_step as f64;
        let radius = exponent.exp();
        let angle = (phase_step * sample_step as f64).rem_euclid(TAU);
        let (sin_phase, cos_phase) = angle.sin_cos();
        let c0 = radius
            * (contract.diagnostic_gain[0] * cos_phase + contract.diagnostic_gain[1] * sin_phase);
        let c1 = radius
            * (-contract.diagnostic_gain[0] * sin_phase + contract.diagnostic_gain[1] * cos_phase);
        let contribution = [
            c0 * c0 * inverse_noise_variance,
            c0 * c1 * inverse_noise_variance,
            c1 * c1 * inverse_noise_variance,
        ];
        if contribution.iter().any(|value| !value.is_finite()) {
            return Err(control_timing_error(
                "oscillatory_mode.observability_gramian",
                "finite-horizon observation Gramian overflowed",
            ));
        }
        for index in 0..3 {
            kahan_add(
                contribution[index],
                &mut observability[index],
                &mut observability_compensation[index],
            );
        }
        observation_sample_count += 1;
        match sample_step.checked_add(contract.diagnostic_sampling_stride_steps) {
            Some(next) => sample_step = next,
            None => break,
        }
    }

    let mut controllability = [0.0_f64; 3];
    let mut controllability_compensation = [0.0_f64; 3];
    let mut actuator_command_count = 0_u32;
    let mut command_step = 0_u32;
    while let Some(applied_step) = command_step.checked_add(contract.actuator_delivery_delay_steps)
    {
        if applied_step >= contract.horizon_steps {
            break;
        }
        let remaining_steps = contract.horizon_steps - 1 - applied_step;
        let exponent = growth_exponent * remaining_steps as f64;
        let radius = exponent.exp();
        let angle = (phase_step * remaining_steps as f64).rem_euclid(TAU);
        let (sin_phase, cos_phase) = angle.sin_cos();
        let b0 = radius
            * (cos_phase * contract.actuator_gain[0] - sin_phase * contract.actuator_gain[1]);
        let b1 = radius
            * (sin_phase * contract.actuator_gain[0] + cos_phase * contract.actuator_gain[1]);
        let contribution = [b0 * b0, b0 * b1, b1 * b1];
        if contribution.iter().any(|value| !value.is_finite()) {
            return Err(control_timing_error(
                "oscillatory_mode.controllability_gramian",
                "finite-horizon actuator Gramian overflowed",
            ));
        }
        for index in 0..3 {
            kahan_add(
                contribution[index],
                &mut controllability[index],
                &mut controllability_compensation[index],
            );
        }
        actuator_command_count += 1;
        match command_step.checked_add(contract.actuator_update_stride_steps) {
            Some(next) => command_step = next,
            None => break,
        }
    }

    let observability_eigenvalues = symmetric_2x2_eigenvalues(observability);
    let controllability_eigenvalues = symmetric_2x2_eigenvalues(controllability);
    let diagnostic_interval_seconds =
        contract.diagnostic_sampling_stride_steps as f64 * contract.time_step_seconds;
    let actuator_interval_seconds =
        contract.actuator_update_stride_steps as f64 * contract.time_step_seconds;
    let effective_diagnostic_bandwidth_hz = contract
        .diagnostic_bandwidth_hz
        .min(0.5 / diagnostic_interval_seconds);
    let effective_actuator_bandwidth_hz = contract
        .actuator_bandwidth_hz
        .min(0.5 / actuator_interval_seconds);
    let diagnostic_bandwidth_multiple =
        effective_diagnostic_bandwidth_hz / contract.oscillation_frequency_hz;
    let actuator_bandwidth_multiple =
        effective_actuator_bandwidth_hz / contract.oscillation_frequency_hz;
    let diagnostic_delay_seconds =
        contract.diagnostic_delivery_delay_steps as f64 * contract.time_step_seconds;
    let actuator_delay_seconds =
        contract.actuator_delivery_delay_steps as f64 * contract.time_step_seconds;
    let conservative_loop_latency_seconds = diagnostic_interval_seconds
        + diagnostic_delay_seconds
        + contract.controller_execution_latency_seconds
        + actuator_interval_seconds
        + actuator_delay_seconds;
    let loop_latency_efold_fraction =
        conservative_loop_latency_seconds * contract.growth_rate_per_second;
    let diagnostic_delay_efold_fraction =
        diagnostic_delay_seconds * contract.growth_rate_per_second;
    let actuator_delay_efold_fraction = actuator_delay_seconds * contract.growth_rate_per_second;
    let derived = [
        observability_eigenvalues[0],
        observability_eigenvalues[1],
        controllability_eigenvalues[0],
        controllability_eigenvalues[1],
        effective_diagnostic_bandwidth_hz,
        effective_actuator_bandwidth_hz,
        diagnostic_bandwidth_multiple,
        actuator_bandwidth_multiple,
        conservative_loop_latency_seconds,
        loop_latency_efold_fraction,
        diagnostic_delay_efold_fraction,
        actuator_delay_efold_fraction,
    ];
    if derived.iter().any(|value| !value.is_finite()) {
        return Err(control_timing_error(
            "oscillatory_mode",
            "derived Gramian eigenvalues and timing metrics must be finite",
        ));
    }

    let mode_interface_qualified = observation_sample_count > 0
        && actuator_command_count > 0
        && observability_eigenvalues[0] >= contract.minimum_observability_min_eigenvalue
        && controllability_eigenvalues[0] >= contract.minimum_controllability_min_eigenvalue
        && diagnostic_bandwidth_multiple >= contract.minimum_bandwidth_multiple
        && actuator_bandwidth_multiple >= contract.minimum_bandwidth_multiple
        && loop_latency_efold_fraction <= contract.maximum_loop_latency_efold_fraction;

    Ok(OscillatoryUnstableModeReport {
        discrete_growth_factor_per_step: growth_factor,
        phase_advance_radians_per_step: phase_step.rem_euclid(TAU),
        observation_sample_count,
        actuator_command_count,
        observability_gramian: observability,
        observability_min_eigenvalue: observability_eigenvalues[0],
        observability_max_eigenvalue: observability_eigenvalues[1],
        controllability_gramian: controllability,
        controllability_min_eigenvalue: controllability_eigenvalues[0],
        controllability_max_eigenvalue: controllability_eigenvalues[1],
        effective_diagnostic_bandwidth_hz,
        effective_actuator_bandwidth_hz,
        diagnostic_bandwidth_multiple,
        actuator_bandwidth_multiple,
        conservative_loop_latency_seconds,
        loop_latency_efold_fraction,
        diagnostic_delay_efold_fraction,
        actuator_delay_efold_fraction,
        minimum_required_bandwidth_multiple: contract.minimum_bandwidth_multiple,
        maximum_allowed_loop_latency_efold_fraction: contract.maximum_loop_latency_efold_fraction,
        minimum_required_observability_min_eigenvalue: contract
            .minimum_observability_min_eigenvalue,
        minimum_required_controllability_min_eigenvalue: contract
            .minimum_controllability_min_eigenvalue,
        mode_interface_qualified,
    })
}

fn symmetric_2x2_eigenvalues(matrix: [f64; 3]) -> [f64; 2] {
    let difference = matrix[0] - matrix[2];
    let discriminant = difference
        .mul_add(difference, 4.0 * matrix[1] * matrix[1])
        .sqrt();
    let trace = matrix[0] + matrix[2];
    let minimum = ((trace - discriminant) * 0.5).max(0.0);
    let maximum = (trace + discriminant) * 0.5;
    [minimum, maximum]
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
    for ((projected, original), weight) in
        micro_fluctuation.iter().zip(values).zip(diagonal_weights)
    {
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
        for value in matrix[pivot_index].iter_mut().skip(pivot_index) {
            *value /= pivot;
        }
        rhs[pivot_index] /= pivot;
        let normalized_pivot_row = matrix[pivot_index].clone();

        for (row_index, row) in matrix.iter_mut().enumerate() {
            if row_index == pivot_index {
                continue;
            }
            let factor = row[pivot_index];
            for (value, pivot_value) in row
                .iter_mut()
                .skip(pivot_index)
                .zip(normalized_pivot_row.iter().skip(pivot_index))
            {
                *value -= factor * pivot_value;
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
    validate_kird_interval_cover(root, covering_radius, registered_probe_count)?;
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

/// Verify a one-dimensional compact interval r-net using exact integer ticks.
///
/// This verifies the cover geometry in the declared normalized interval metric.
/// It does not prove the response Lipschitz constant; that remains a separate
/// proof artifact checked by an independent certificate process.
fn validate_kird_interval_cover(
    root: &serde_json::Value,
    declared_radius: f64,
    registered_probe_count: u64,
) -> Result<(), ContractError> {
    let metric_type = kird_string(root, "response_scope.coverage_domain.metric_type")?;
    if metric_type != "normalized_interval_abs" {
        return Err(kird_error(
            "response_scope.coverage_domain.metric_type",
            "this verifier currently supports only one-dimensional normalized interval absolute distance",
        ));
    }
    let coordinate_id = kird_string(root, "response_scope.coverage_domain.coordinate_id")?;
    if coordinate_id.trim().is_empty() {
        return Err(kird_error(
            "response_scope.coverage_domain.coordinate_id",
            "must identify the normalized intervention coordinate",
        ));
    }

    let minimum_tick = kird_u64(root, "response_scope.coverage_domain.minimum_tick")?;
    let maximum_tick = kird_u64(root, "response_scope.coverage_domain.maximum_tick")?;
    let tick_scale = kird_u64(root, "response_scope.coverage_domain.tick_scale")?;
    let radius_ticks = kird_u64(root, "response_scope.coverage_domain.radius_ticks")?;
    if tick_scale == 0 || minimum_tick >= maximum_tick || maximum_tick > tick_scale {
        return Err(kird_error(
            "response_scope.coverage_domain",
            "normalized interval ticks require 0 <= minimum < maximum <= tick_scale",
        ));
    }

    let centers = kird_array(root, "response_scope.coverage_domain.probe_centers_ticks")?;
    if centers.len() as u64 != registered_probe_count {
        return Err(kird_error(
            "response_scope.coverage_domain.probe_centers_ticks",
            "the geometric cover must contain exactly one center per registered intervention probe",
        ));
    }
    let mut intervals = Vec::with_capacity(centers.len());
    let mut sorted_centers = Vec::with_capacity(centers.len());
    for center_value in centers {
        let center = center_value.as_u64().ok_or_else(|| {
            kird_error(
                "response_scope.coverage_domain.probe_centers_ticks",
                "each center must be a non-negative integer tick",
            )
        })?;
        if center < minimum_tick || center > maximum_tick {
            return Err(kird_error(
                "response_scope.coverage_domain.probe_centers_ticks",
                "probe center lies outside the declared compact intervention interval",
            ));
        }
        sorted_centers.push(center);
        let left = center.saturating_sub(radius_ticks).max(minimum_tick);
        let right = center.saturating_add(radius_ticks).min(maximum_tick);
        intervals.push((left, right));
    }
    sorted_centers.sort_unstable();
    if sorted_centers.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(kird_error(
            "response_scope.coverage_domain.probe_centers_ticks",
            "duplicate probe centers do not increase cover coverage",
        ));
    }

    intervals.sort_unstable_by_key(|interval| interval.0);
    let mut covered_until = minimum_tick;
    for (left, right) in intervals {
        if left > covered_until {
            return Err(kird_error(
                "response_scope.coverage_domain",
                "probe balls leave a gap in the declared compact interval",
            ));
        }
        covered_until = covered_until.max(right);
    }
    if covered_until < maximum_tick {
        return Err(kird_error(
            "response_scope.coverage_domain",
            "probe balls do not cover the upper end of the declared compact interval",
        ));
    }

    let geometric_radius = radius_ticks as f64 / tick_scale as f64;
    let allowance = 1.0e-12_f64.max(geometric_radius.abs() * 1.0e-10);
    if !declared_radius.is_finite() || (declared_radius - geometric_radius).abs() > allowance {
        return Err(kird_error(
            "response_scope.covering_radius",
            "declared covering radius must equal radius_ticks / tick_scale",
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
pub const KINETIC_DISCRIMINATION_FIXTURE: &str = include_str!(
    "../../../../benchmarks/plasma/kinetic-discrimination/fixtures/complex-response-eig-known-answers-v0.1.json"
);
pub const KINETIC_DISCRIMINATION_SCHEMA: &str = include_str!(
    "../../../../benchmarks/plasma/kinetic-discrimination/complex-response-eig-design-v0.1.schema.json"
);
pub const WEIGHTED_FOURIER_MEASURE_FIXTURE: &str = include_str!(
    "../../../../benchmarks/plasma/topology-lifting/fixtures/weighted-fourier-measure-known-answer-v0.1.json"
);
pub const WEIGHTED_MEASURE_TAIL_STABILITY_FIXTURE: &str = include_str!(
    "../../../../benchmarks/plasma/topology-lifting/fixtures/weighted-measure-tail-stability-v0.1.json"
);

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
    fn kird_audit_verifies_interval_probe_balls_cover_the_entire_domain() {
        let value = certified_kird_value();
        let report = validate_kird_record_value(&value).unwrap();
        assert_eq!(report.response_coverage_kind, "certified_global");
        assert_eq!(report.covering_radius, Some(0.1));
    }

    #[test]
    fn kird_audit_rejects_a_gap_in_the_certified_interval_cover() {
        let mut value = certified_kird_value();
        value["response_scope"]["coverage_domain"]["radius_ticks"] = serde_json::json!(4);
        value["response_scope"]["covering_radius"] = serde_json::json!(0.04);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.coverage_domain",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_interval_cover_probe_count_mismatch() {
        let mut value = certified_kird_value();
        value["response_scope"]["coverage_domain"]["probe_centers_ticks"] =
            serde_json::json!([0, 10]);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.coverage_domain.probe_centers_ticks",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_probe_centers_outside_the_compact_domain() {
        let mut value = certified_kird_value();
        value["response_scope"]["coverage_domain"]["probe_centers_ticks"] =
            serde_json::json!([11]);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.coverage_domain.probe_centers_ticks",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_covering_radius_inconsistent_with_integer_geometry() {
        let mut value = certified_kird_value();
        value["response_scope"]["covering_radius"] = serde_json::json!(0.09);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.covering_radius",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_duplicate_centers_in_interval_cover() {
        let mut value = certified_kird_value();
        value["response_scope"]["coverage_domain"]["probe_centers_ticks"] =
            serde_json::json!([0, 0]);
        value["response_scope"]["registered_probe_count"] = serde_json::json!(2);
        value["response_scope"]["confirmed_probe_count"] = serde_json::json!(2);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.coverage_domain.probe_centers_ticks",
                ..
            })
        ));
    }

    #[test]
    fn kird_audit_rejects_non_compact_or_degenerate_tick_domain() {
        let mut value = certified_kird_value();
        value["response_scope"]["coverage_domain"]["maximum_tick"] = serde_json::json!(0);
        assert!(matches!(
            validate_kird_record_value(&value),
            Err(ContractError::InvalidKirdField {
                field: "response_scope.coverage_domain",
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
    fn weighted_fourier_measure_embedding_matches_checked_in_known_answer() {
        let fixture: serde_json::Value =
            serde_json::from_str(WEIGHTED_FOURIER_MEASURE_FIXTURE).unwrap();
        let positions = fixture["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>();
        let weights = fixture["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>();
        let harmonics = fixture["max_harmonic"].as_u64().unwrap() as usize;
        let expected = fixture["expected_normalized_complex_moments"]
            .as_array()
            .unwrap();
        let embedding = WeightedFourierMeasureEmbedding::new(harmonics).unwrap();
        let actual = embedding.moments(&positions, &weights).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, moment) in actual.iter().enumerate() {
            let expected_real = expected[index]["real"].as_f64().unwrap();
            let expected_imag = expected[index]["imag"].as_f64().unwrap();
            assert!((moment.real - expected_real).abs() < 1.0e-12);
            assert!((moment.imag - expected_imag).abs() < 1.0e-12);
        }
        assert!(
            fixture["claim_boundary"]["finite_fourier_band_is_not_injective"]
                .as_bool()
                .unwrap()
        );
        assert!(
            fixture["claim_boundary"]["no_physical_plasma_run"]
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn weighted_measure_tail_outlier_stability_and_minmax_drift() {
        let fixture: serde_json::Value =
            serde_json::from_str(WEIGHTED_MEASURE_TAIL_STABILITY_FIXTURE).unwrap();
        let parse_numbers = |field: &str| {
            fixture[field]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let base_positions = parse_numbers("base_positions");
        let base_weights = parse_numbers("base_weights");
        let augmented_positions = parse_numbers("augmented_positions");
        let augmented_weights = parse_numbers("augmented_weights");
        let epsilon = fixture["outlier_mass"].as_f64().unwrap();
        let harmonics = fixture["max_harmonic"].as_u64().unwrap() as usize;

        let base_map = MinMaxCoordinateMap::fit(&base_positions, 0.0, 1.0).unwrap();
        let augmented_map = MinMaxCoordinateMap::fit(&augmented_positions, 0.0, 1.0).unwrap();
        let maximum_coordinate_shift = base_positions
            .iter()
            .map(|position| {
                (base_map.apply(*position).unwrap() - augmented_map.apply(*position).unwrap()).abs()
            })
            .fold(0.0_f64, f64::max);
        let expected_coordinate_shift =
            fixture["sample_extrema_normalization_witness"]
                ["max_inlier_normalized_coordinate_displacement"]
                .as_f64()
                .unwrap();
        assert!((maximum_coordinate_shift - expected_coordinate_shift).abs() < 1.0e-12);

        let embedding = WeightedFourierMeasureEmbedding::new(harmonics).unwrap();
        let base_moments = embedding.moments(&base_positions, &base_weights).unwrap();
        let augmented_moments = embedding
            .moments(&augmented_positions, &augmented_weights)
            .unwrap();
        let maximum_moment_change = base_moments
            .iter()
            .zip(&augmented_moments)
            .map(|(base, augmented)| (augmented.real - base.real).hypot(augmented.imag - base.imag))
            .fold(0.0_f64, f64::max);
        let theoretical_bound = 2.0 * epsilon / (1.0 + epsilon);
        let declared_bound = fixture["mathematical_bound"]["expected_upper_bound"]
            .as_f64()
            .unwrap();
        assert!((theoretical_bound - declared_bound).abs() < 1.0e-24);
        assert!(maximum_moment_change <= theoretical_bound + 1.0e-15);
        assert!(
            fixture["claim_boundary"]["low_mass_does_not_imply_dynamically_irrelevant"]
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn finite_fourier_band_has_an_explicit_noninjectivity_witness() {
        let first_four = WeightedFourierMeasureEmbedding::new(4).unwrap();
        let five_point_measure = (0..5).map(|index| index as f64 / 5.0).collect::<Vec<_>>();
        let ten_point_measure = (0..10).map(|index| index as f64 / 10.0).collect::<Vec<_>>();
        let five_weights = vec![0.2; 5];
        let ten_weights = vec![0.1; 10];

        assert_ne!(five_point_measure, ten_point_measure);
        let five_moments = first_four
            .moments(&five_point_measure, &five_weights)
            .unwrap();
        let ten_moments = first_four
            .moments(&ten_point_measure, &ten_weights)
            .unwrap();
        for (left, right) in five_moments.iter().zip(&ten_moments) {
            assert!(left.magnitude() < 1.0e-12);
            assert!(right.magnitude() < 1.0e-12);
            assert!((left.real - right.real).abs() < 1.0e-12);
            assert!((left.imag - right.imag).abs() < 1.0e-12);
        }

        let fifth_moment = WeightedFourierMeasureEmbedding::new(5)
            .unwrap()
            .moments(&five_point_measure, &five_weights)
            .unwrap()[4];
        assert!((fifth_moment.real - 1.0).abs() < 1.0e-12);
        assert!(fifth_moment.imag.abs() < 1.0e-12);
    }

    #[test]
    fn weighted_fourier_measure_embedding_is_permutation_invariant() {
        let embedding = WeightedFourierMeasureEmbedding::new(8).unwrap();
        let positions = [0.05, 0.23, 0.61, 0.94];
        let weights = [0.1, 0.2, 0.3, 0.4];
        let permutation = [0.61, 0.05, 0.94, 0.23];
        let permuted_weights = [0.3, 0.1, 0.4, 0.2];
        let original = embedding.moments(&positions, &weights).unwrap();
        let permuted = embedding.moments(&permutation, &permuted_weights).unwrap();
        for (left, right) in original.iter().zip(&permuted) {
            assert!((left.real - right.real).abs() < 1.0e-12);
            assert!((left.imag - right.imag).abs() < 1.0e-12);
        }
    }

    #[test]
    fn weighted_fourier_measure_embedding_is_invariant_to_exact_coincident_split_merge() {
        let embedding = WeightedFourierMeasureEmbedding::new(12).unwrap();
        let original = embedding.moments(&[0.2, 0.7], &[0.4, 0.6]).unwrap();
        let split = embedding
            .moments(&[0.2, 0.2, 0.7], &[0.1, 0.3, 0.6])
            .unwrap();
        let merged = embedding.moments(&[0.7, 0.2], &[0.6, 0.4]).unwrap();
        for expected in [&split, &merged] {
            for (left, right) in original.iter().zip(expected.iter()) {
                assert!((left.real - right.real).abs() < 1.0e-12);
                assert!((left.imag - right.imag).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn weighted_fourier_measure_embedding_is_invariant_to_global_weight_scale() {
        let embedding = WeightedFourierMeasureEmbedding::new(6).unwrap();
        let base = embedding
            .moments(&[0.13, 0.42, 0.81], &[1.0, 2.0, 7.0])
            .unwrap();
        let scaled = embedding
            .moments(&[0.13, 0.42, 0.81], &[1.0e200, 2.0e200, 7.0e200])
            .unwrap();
        for (left, right) in base.iter().zip(&scaled) {
            assert!((left.real - right.real).abs() < 1.0e-12);
            assert!((left.imag - right.imag).abs() < 1.0e-12);
        }
    }

    #[test]
    fn weighted_fourier_measure_embedding_transforms_equivariantly_under_torus_translation() {
        let embedding = WeightedFourierMeasureEmbedding::new(5).unwrap();
        let positions = [0.11, 0.34, 0.77];
        let weights = [0.2, 0.5, 0.3];
        let shift = 0.173;
        let original = embedding.moments(&positions, &weights).unwrap();
        let translated_positions = positions.map(|position| position + shift);
        let translated = embedding.moments(&translated_positions, &weights).unwrap();
        for (index, (before, after)) in original.iter().zip(&translated).enumerate() {
            let phase = TAU * (index + 1) as f64 * shift;
            let expected_real = before.real * phase.cos() - before.imag * phase.sin();
            let expected_imag = before.real * phase.sin() + before.imag * phase.cos();
            assert!((after.real - expected_real).abs() < 1.0e-12);
            assert!((after.imag - expected_imag).abs() < 1.0e-12);
        }
    }

    #[test]
    fn weighted_fourier_measure_embedding_rejects_invalid_shapes_weights_and_budgets() {
        let embedding = WeightedFourierMeasureEmbedding::new(3).unwrap();
        assert_eq!(
            embedding.moments(&[], &[]),
            Err(ContractError::InvalidParticleMeasureShape)
        );
        assert_eq!(
            embedding.moments(&[0.2, 0.4], &[1.0]),
            Err(ContractError::InvalidParticleMeasureShape)
        );
        assert_eq!(
            embedding.moments(&[0.2, 0.4], &[0.0, 0.0]),
            Err(ContractError::InvalidParticleMeasureWeights)
        );
        assert_eq!(
            embedding.moments(&[0.2, 0.4], &[1.0, -1.0]),
            Err(ContractError::InvalidParticleMeasureWeights)
        );
        assert_eq!(
            embedding.moments(&[0.2, 0.4], &[1.0, f64::NAN]),
            Err(ContractError::InvalidParticleMeasureWeights)
        );
        assert_eq!(
            embedding.moments(&[0.2, f64::INFINITY], &[1.0, 1.0]),
            Err(ContractError::NonFiniteNormalizationCoordinate)
        );
        assert_eq!(
            WeightedFourierMeasureEmbedding::new(0),
            Err(ContractError::InvalidHarmonicBudget)
        );
    }

    #[test]
    fn weighted_fourier_measure_embedding_rejects_excessive_work_before_allocation() {
        let embedding = WeightedFourierMeasureEmbedding::new(MAX_PERIODIC_LIFT_HARMONICS).unwrap();
        let positions = vec![0.5; MAX_MEASURE_PARTICLES];
        let weights = vec![1.0; MAX_MEASURE_PARTICLES];
        assert_eq!(
            embedding.moments(&positions, &weights),
            Err(ContractError::ParticleMeasureWorkBudgetExceeded)
        );
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

    const SCALAR_UNSTABLE_MODE_FIXTURE: &str = include_str!(
        "../../../../benchmarks/plasma/control-interface/fixtures/scalar-unstable-mode-known-answer-v0.1.json"
    );

    fn scalar_unstable_mode_contract() -> ScalarUnstableModeContract {
        let value: serde_json::Value = serde_json::from_str(SCALAR_UNSTABLE_MODE_FIXTURE).unwrap();
        let model = &value["model"];
        let diagnostic = &value["diagnostic"];
        let actuator = &value["actuator"];
        let controller = &value["controller"];
        ScalarUnstableModeContract {
            growth_rate_per_second: model["growth_rate_per_second"].as_f64().unwrap(),
            oscillation_frequency_hz: model["oscillation_frequency_hz"].as_f64().unwrap(),
            time_step_seconds: model["time_step_seconds"].as_f64().unwrap(),
            horizon_steps: model["horizon_steps"].as_u64().unwrap() as u32,
            diagnostic_gain: diagnostic["gain"].as_f64().unwrap(),
            diagnostic_noise_std: diagnostic["noise_std"].as_f64().unwrap(),
            diagnostic_bandwidth_hz: diagnostic["bandwidth_hz"].as_f64().unwrap(),
            diagnostic_sampling_stride_steps: diagnostic["sampling_stride_steps"].as_u64().unwrap()
                as u32,
            diagnostic_delivery_delay_steps: diagnostic["delivery_delay_steps"].as_u64().unwrap()
                as u32,
            actuator_gain: actuator["gain"].as_f64().unwrap(),
            actuator_bandwidth_hz: actuator["bandwidth_hz"].as_f64().unwrap(),
            actuator_update_stride_steps: actuator["update_stride_steps"].as_u64().unwrap() as u32,
            actuator_delivery_delay_steps: actuator["delivery_delay_steps"].as_u64().unwrap()
                as u32,
            controller_execution_latency_seconds: controller["execution_latency_seconds"]
                .as_f64()
                .unwrap(),
            minimum_bandwidth_multiple: controller["minimum_bandwidth_multiple"].as_f64().unwrap(),
            maximum_loop_latency_efold_fraction: controller["maximum_loop_latency_efold_fraction"]
                .as_f64()
                .unwrap(),
            minimum_observability_fisher_information:
                controller["minimum_observability_fisher_information"]
                    .as_f64()
                    .unwrap(),
            minimum_controllability_gramian: controller["minimum_controllability_gramian"]
                .as_f64()
                .unwrap(),
        }
    }

    const OSCILLATORY_UNSTABLE_MODE_FIXTURE: &str = include_str!(
        "../../../../benchmarks/plasma/control-interface/fixtures/oscillatory-unstable-mode-known-answer-v0.1.json"
    );

    fn oscillatory_unstable_mode_contract() -> OscillatoryUnstableModeContract {
        let value: serde_json::Value =
            serde_json::from_str(OSCILLATORY_UNSTABLE_MODE_FIXTURE).unwrap();
        let model = &value["model"];
        let diagnostic = &value["diagnostic"];
        let actuator = &value["actuator"];
        let controller = &value["controller"];
        OscillatoryUnstableModeContract {
            growth_rate_per_second: model["growth_rate_per_second"].as_f64().unwrap(),
            oscillation_frequency_hz: model["oscillation_frequency_hz"].as_f64().unwrap(),
            time_step_seconds: model["time_step_seconds"].as_f64().unwrap(),
            horizon_steps: model["horizon_steps"].as_u64().unwrap() as u32,
            diagnostic_gain: [
                diagnostic["gain_real"].as_f64().unwrap(),
                diagnostic["gain_imag"].as_f64().unwrap(),
            ],
            diagnostic_noise_std: diagnostic["noise_std"].as_f64().unwrap(),
            diagnostic_bandwidth_hz: diagnostic["bandwidth_hz"].as_f64().unwrap(),
            diagnostic_sampling_stride_steps: diagnostic["sampling_stride_steps"].as_u64().unwrap()
                as u32,
            diagnostic_delivery_delay_steps: diagnostic["delivery_delay_steps"].as_u64().unwrap()
                as u32,
            actuator_gain: [
                actuator["gain_real"].as_f64().unwrap(),
                actuator["gain_imag"].as_f64().unwrap(),
            ],
            actuator_bandwidth_hz: actuator["bandwidth_hz"].as_f64().unwrap(),
            actuator_update_stride_steps: actuator["update_stride_steps"].as_u64().unwrap() as u32,
            actuator_delivery_delay_steps: actuator["delivery_delay_steps"].as_u64().unwrap()
                as u32,
            controller_execution_latency_seconds: controller["execution_latency_seconds"]
                .as_f64()
                .unwrap(),
            minimum_bandwidth_multiple: controller["minimum_bandwidth_multiple"].as_f64().unwrap(),
            maximum_loop_latency_efold_fraction: controller["maximum_loop_latency_efold_fraction"]
                .as_f64()
                .unwrap(),
            minimum_observability_min_eigenvalue:
                controller["minimum_observability_min_eigenvalue"]
                    .as_f64()
                    .unwrap(),
            minimum_controllability_min_eigenvalue:
                controller["minimum_controllability_min_eigenvalue"]
                    .as_f64()
                    .unwrap(),
        }
    }

    const REFINEMENT_POLICY_FIXTURE: &str = include_str!(
        "../../../../benchmarks/plasma/adaptive-refinement/fixtures/synthetic-refinement-cases-v0.1.json"
    );

    fn refinement_input(
        spectral_tail_measure: f64,
        response_risk_upper_bound: Option<f64>,
        projected_peak_bytes: u64,
    ) -> SpectralRefinementInput {
        SpectralRefinementInput {
            spectral_tail_measure,
            spectral_tail_threshold: 10.0,
            response_risk_upper_bound,
            response_risk_threshold: 1.0,
            response_assessment_fresh: true,
            response_branch_association_stable: true,
            response_estimator_latency_seconds: 0.005,
            refinement_deadline_seconds: 0.01,
            projected_peak_bytes,
            projected_peak_includes_temporary_buffers: true,
            memory_budget_bytes: 1_000,
            reserved_headroom_bytes: 100,
        }
    }

    #[test]
    fn spectral_refinement_fixture_matches_all_six_known_answers() {
        let fixture: serde_json::Value = serde_json::from_str(REFINEMENT_POLICY_FIXTURE).unwrap();
        let thresholds = &fixture["thresholds"];
        let memory = &fixture["memory_policy"];
        let tail_threshold = thresholds["spectral_tail_threshold"].as_f64().unwrap();
        let response_threshold = thresholds["response_risk_threshold"].as_f64().unwrap();
        let budget = memory["memory_budget_bytes"].as_u64().unwrap();
        let reserve = memory["reserved_headroom_bytes"].as_u64().unwrap();

        for case in fixture["cases"].as_array().unwrap() {
            let response_risk = case["response_risk_upper_bound"].as_f64();
            let input = SpectralRefinementInput {
                spectral_tail_measure: case["spectral_tail_measure"].as_f64().unwrap(),
                spectral_tail_threshold: tail_threshold,
                response_risk_upper_bound: response_risk,
                response_risk_threshold: response_threshold,
                response_assessment_fresh: case["response_assessment_fresh"].as_bool().unwrap(),
                response_branch_association_stable: case["response_branch_association_stable"]
                    .as_bool()
                    .unwrap(),
                response_estimator_latency_seconds: case["response_estimator_latency_seconds"]
                    .as_f64()
                    .unwrap(),
                refinement_deadline_seconds: case["refinement_deadline_seconds"].as_f64().unwrap(),
                projected_peak_bytes: case["projected_peak_bytes"].as_u64().unwrap(),
                projected_peak_includes_temporary_buffers:
                    memory["projected_peak_includes_temporary_buffers"]
                        .as_bool()
                        .unwrap(),
                memory_budget_bytes: budget,
                reserved_headroom_bytes: reserve,
            };
            let report = decide_spectral_refinement(input).unwrap();
            assert_eq!(
                format!("{:?}", report.decision),
                case["expected_decision"].as_str().unwrap(),
                "decision mismatch for {}",
                case["case_id"].as_str().unwrap()
            );
            assert_eq!(
                format!("{:?}", report.reason),
                case["expected_reason"].as_str().unwrap(),
                "reason mismatch for {}",
                case["case_id"].as_str().unwrap()
            );
            assert_eq!(
                format!("{:?}", report.qualification),
                case["expected_qualification"].as_str().unwrap(),
                "qualification mismatch for {}",
                case["case_id"].as_str().unwrap()
            );
        }
    }

    #[test]
    fn spectral_refinement_distinguishes_matched_tail_different_response_witnesses() {
        let low_response =
            decide_spectral_refinement(refinement_input(2.0, Some(0.5), 700)).unwrap();
        let high_response =
            decide_spectral_refinement(refinement_input(2.0, Some(1.2), 700)).unwrap();
        assert_eq!(
            low_response.spectral_tail_ratio,
            high_response.spectral_tail_ratio
        );
        assert_eq!(low_response.decision, SpectralRefinementDecision::Hold);
        assert_eq!(high_response.decision, SpectralRefinementDecision::Refine);
        assert_eq!(
            high_response.reason,
            SpectralRefinementReason::FutureResponse
        );
    }

    #[test]
    fn spectral_refinement_holds_when_both_signals_are_below_threshold() {
        let report = decide_spectral_refinement(refinement_input(5.0, Some(0.5), 700)).unwrap();
        assert_eq!(report.decision, SpectralRefinementDecision::Hold);
        assert_eq!(report.reason, SpectralRefinementReason::NoTrigger);
        assert_eq!(
            report.qualification,
            SpectralRefinementQualification::NoTrigger
        );
        assert_eq!(report.remaining_headroom_bytes, 200);
    }

    #[test]
    fn spectral_refinement_responds_to_future_qoi_even_when_tail_is_quiet() {
        let report = decide_spectral_refinement(refinement_input(2.0, Some(1.2), 700)).unwrap();
        assert_eq!(report.decision, SpectralRefinementDecision::Refine);
        assert_eq!(report.reason, SpectralRefinementReason::FutureResponse);
        assert_eq!(
            report.qualification,
            SpectralRefinementQualification::Triggered
        );
    }

    #[test]
    fn spectral_refinement_responds_to_spectral_tail_trigger() {
        let report = decide_spectral_refinement(refinement_input(10.0, Some(0.2), 700)).unwrap();
        assert_eq!(report.decision, SpectralRefinementDecision::Refine);
        assert_eq!(report.reason, SpectralRefinementReason::SpectralTail);
    }

    #[test]
    fn spectral_refinement_treats_missing_response_as_unknown_not_zero() {
        let report = decide_spectral_refinement(refinement_input(1.0, None, 700)).unwrap();
        assert_eq!(report.decision, SpectralRefinementDecision::Refine);
        assert_eq!(report.reason, SpectralRefinementReason::ResponseUnknown);
        assert_eq!(
            report.qualification,
            SpectralRefinementQualification::UnknownResponse
        );
    }

    #[test]
    fn spectral_refinement_reports_unknown_when_memory_blocks_required_growth() {
        let report = decide_spectral_refinement(refinement_input(1.0, Some(2.0), 901)).unwrap();
        assert_eq!(report.decision, SpectralRefinementDecision::BlockedByMemory);
        assert_eq!(
            report.qualification,
            SpectralRefinementQualification::UnknownBlockedByMemory
        );
        assert_eq!(report.remaining_headroom_bytes, 0);
    }

    #[test]
    fn spectral_refinement_treats_stale_response_assessment_as_unknown() {
        let mut input = refinement_input(1.0, Some(0.1), 700);
        input.response_assessment_fresh = false;
        let report = decide_spectral_refinement(input).unwrap();
        assert_eq!(report.decision, SpectralRefinementDecision::Refine);
        assert_eq!(report.reason, SpectralRefinementReason::ResponseUnknown);
        assert_eq!(
            report.qualification,
            SpectralRefinementQualification::UnknownResponse
        );
        assert!(!report.response_assessment_valid);
    }

    #[test]
    fn spectral_refinement_treats_unstable_branch_association_as_unknown() {
        let mut input = refinement_input(1.0, Some(0.1), 700);
        input.response_branch_association_stable = false;
        let report = decide_spectral_refinement(input).unwrap();
        assert_eq!(report.decision, SpectralRefinementDecision::Refine);
        assert_eq!(report.reason, SpectralRefinementReason::ResponseUnknown);
        assert!(!report.response_assessment_valid);
    }

    #[test]
    fn spectral_refinement_treats_late_response_estimate_as_unknown() {
        let mut input = refinement_input(1.0, Some(0.1), 700);
        input.response_estimator_latency_seconds = 0.02;
        let report = decide_spectral_refinement(input).unwrap();
        assert_eq!(report.decision, SpectralRefinementDecision::Refine);
        assert_eq!(report.reason, SpectralRefinementReason::ResponseUnknown);
        assert!(!report.response_assessment_valid);
    }

    #[test]
    fn spectral_refinement_rejects_projected_memory_without_temporary_buffers() {
        let mut input = refinement_input(1.0, Some(0.1), 700);
        input.projected_peak_includes_temporary_buffers = false;
        assert_eq!(
            decide_spectral_refinement(input),
            Err(ContractError::IncompleteRefinementMemoryEstimate)
        );
    }

    #[test]
    fn spectral_refinement_labels_tail_trigger_with_unknown_response_explicitly() {
        let mut input = refinement_input(12.0, Some(0.1), 700);
        input.response_branch_association_stable = false;
        let report = decide_spectral_refinement(input).unwrap();
        assert_eq!(report.decision, SpectralRefinementDecision::Refine);
        assert_eq!(
            report.reason,
            SpectralRefinementReason::SpectralTailAndResponseUnknown
        );
        assert_eq!(
            report.qualification,
            SpectralRefinementQualification::UnknownResponse
        );
    }

    #[test]
    fn spectral_refinement_rejects_invalid_signal_and_memory_contracts() {
        assert_eq!(
            decide_spectral_refinement(refinement_input(f64::NAN, Some(0.1), 100)),
            Err(ContractError::InvalidRefinementSignal)
        );
        let mut invalid_budget = refinement_input(1.0, Some(0.1), 100);
        invalid_budget.reserved_headroom_bytes = 1_001;
        assert_eq!(
            decide_spectral_refinement(invalid_budget),
            Err(ContractError::InvalidRefinementMemoryBudget)
        );
    }

    #[test]
    fn scalar_unstable_mode_oracle_matches_exact_observability_and_controllability_sums() {
        let report =
            evaluate_scalar_unstable_mode_interface(scalar_unstable_mode_contract()).unwrap();
        assert!((report.discrete_growth_factor_per_step - 1.005012520859401).abs() < 1.0e-13);
        assert_eq!(report.observation_sample_count, 5);
        assert_eq!(report.actuator_command_count, 5);
        assert!((report.observability_fisher_information - 2082.4542913757846).abs() < 1.0e-9);
        assert!((report.controllability_gramian - 1.3015339321098656).abs() < 1.0e-12);
        assert!((report.effective_diagnostic_bandwidth_hz - 25.0).abs() < 1.0e-12);
        assert!((report.effective_actuator_bandwidth_hz - 20.0).abs() < 1.0e-12);
        assert!((report.diagnostic_bandwidth_multiple - 5.0).abs() < 1.0e-12);
        assert!((report.actuator_bandwidth_multiple - 4.0).abs() < 1.0e-12);
        assert!((report.conservative_loop_latency_seconds - 0.065).abs() < 1.0e-12);
        assert!((report.loop_latency_efold_fraction - 0.0325).abs() < 1.0e-12);
        assert!((report.diagnostic_delay_efold_fraction - 0.005).abs() < 1.0e-12);
        assert!((report.actuator_delay_efold_fraction - 0.005).abs() < 1.0e-12);
        assert!(
            report.observability_fisher_information
                >= report.minimum_required_observability_fisher_information
        );
        assert!(report.controllability_gramian >= report.minimum_required_controllability_gramian);
        assert!(report.mode_interface_qualified);
    }

    #[test]
    fn scalar_unstable_mode_oracle_marks_bandwidth_limited_observation_unqualified() {
        let mut contract = scalar_unstable_mode_contract();
        contract.diagnostic_sampling_stride_steps = 10;
        let report = evaluate_scalar_unstable_mode_interface(contract).unwrap();
        assert!(report.diagnostic_bandwidth_multiple < report.minimum_required_bandwidth_multiple);
        assert!(!report.mode_interface_qualified);
    }

    #[test]
    fn scalar_unstable_mode_oracle_marks_no_delivered_actuation_unqualified() {
        let mut contract = scalar_unstable_mode_contract();
        contract.actuator_delivery_delay_steps = contract.horizon_steps;
        let report = evaluate_scalar_unstable_mode_interface(contract).unwrap();
        assert_eq!(report.actuator_command_count, 0);
        assert_eq!(report.controllability_gramian, 0.0);
        assert!(!report.mode_interface_qualified);
    }

    #[test]
    fn scalar_unstable_mode_oracle_rejects_insufficient_observation_information() {
        let mut contract = scalar_unstable_mode_contract();
        contract.diagnostic_noise_std = 10.0;
        let report = evaluate_scalar_unstable_mode_interface(contract).unwrap();
        assert!(
            report.observability_fisher_information
                < report.minimum_required_observability_fisher_information
        );
        assert!(!report.mode_interface_qualified);
    }

    #[test]
    fn scalar_unstable_mode_oracle_rejects_insufficient_actuator_authority() {
        let mut contract = scalar_unstable_mode_contract();
        contract.actuator_gain = 0.01;
        let report = evaluate_scalar_unstable_mode_interface(contract).unwrap();
        assert!(report.controllability_gramian < report.minimum_required_controllability_gramian);
        assert!(!report.mode_interface_qualified);
    }

    #[test]
    fn scalar_unstable_mode_oracle_rejects_invalid_noise_gain_and_schedule() {
        let mut zero_noise = scalar_unstable_mode_contract();
        zero_noise.diagnostic_noise_std = 0.0;
        assert!(matches!(
            evaluate_scalar_unstable_mode_interface(zero_noise),
            Err(ContractError::InvalidControlTimingField {
                field: "scalar_mode",
                ..
            })
        ));

        let mut invalid_gain = scalar_unstable_mode_contract();
        invalid_gain.actuator_gain = f64::NAN;
        assert!(matches!(
            evaluate_scalar_unstable_mode_interface(invalid_gain),
            Err(ContractError::InvalidControlTimingField {
                field: "scalar_mode",
                ..
            })
        ));

        let mut invalid_stride = scalar_unstable_mode_contract();
        invalid_stride.diagnostic_sampling_stride_steps = 0;
        assert!(matches!(
            evaluate_scalar_unstable_mode_interface(invalid_stride),
            Err(ContractError::InvalidControlTimingField {
                field: "scalar_mode",
                ..
            })
        ));
    }

    #[test]
    fn scalar_unstable_mode_oracle_fails_closed_on_growth_overflow_and_excessive_horizon() {
        let mut overflow = scalar_unstable_mode_contract();
        overflow.growth_rate_per_second = 1_000.0;
        overflow.time_step_seconds = 1.0;
        assert!(matches!(
            evaluate_scalar_unstable_mode_interface(overflow),
            Err(ContractError::InvalidControlTimingField {
                field: "scalar_mode.growth_rate_per_second",
                ..
            })
        ));

        let mut excessive = scalar_unstable_mode_contract();
        excessive.horizon_steps = MAX_SCALAR_MODE_HORIZON_STEPS + 1;
        assert!(matches!(
            evaluate_scalar_unstable_mode_interface(excessive),
            Err(ContractError::InvalidControlTimingField {
                field: "scalar_mode.horizon_steps",
                ..
            })
        ));
    }

    #[test]
    fn oscillatory_unstable_mode_oracle_matches_two_by_two_gramian_known_answer() {
        let report =
            evaluate_oscillatory_unstable_mode_interface(oscillatory_unstable_mode_contract())
                .unwrap();
        let expected =
            serde_json::from_str::<serde_json::Value>(OSCILLATORY_UNSTABLE_MODE_FIXTURE).unwrap();
        let known = &expected["known_answers"];
        assert_eq!(report.observation_sample_count, 5);
        assert_eq!(report.actuator_command_count, 5);
        for (actual, expected) in report
            .observability_gramian
            .iter()
            .zip(known["observability_gramian"].as_array().unwrap())
        {
            assert!((actual - expected.as_f64().unwrap()).abs() < 1.0e-9);
        }
        for (actual, expected) in report
            .controllability_gramian
            .iter()
            .zip(known["controllability_gramian"].as_array().unwrap())
        {
            assert!((actual - expected.as_f64().unwrap()).abs() < 1.0e-12);
        }
        assert!(
            (report.observability_min_eigenvalue
                - known["observability_min_eigenvalue"].as_f64().unwrap())
            .abs()
                < 1.0e-8
        );
        assert!(
            (report.controllability_min_eigenvalue
                - known["controllability_min_eigenvalue"].as_f64().unwrap())
            .abs()
                < 1.0e-12
        );
        assert!((report.diagnostic_bandwidth_multiple - 5.0).abs() < 1.0e-12);
        assert!((report.actuator_bandwidth_multiple - 4.0).abs() < 1.0e-12);
        assert!((report.conservative_loop_latency_seconds - 0.065).abs() < 1.0e-12);
        assert!((report.loop_latency_efold_fraction - 0.013).abs() < 1.0e-12);
        assert!(report.mode_interface_qualified);
    }

    #[test]
    fn oscillatory_unstable_mode_oracle_detects_sampling_and_actuator_phase_aliasing() {
        let mut contract = oscillatory_unstable_mode_contract();
        contract.oscillation_frequency_hz = 50.0;
        let report = evaluate_oscillatory_unstable_mode_interface(contract).unwrap();
        assert_eq!(report.observation_sample_count, 5);
        assert_eq!(report.actuator_command_count, 5);
        assert!(report.observability_min_eigenvalue < 1.0e-7);
        assert!(report.controllability_min_eigenvalue < 1.0e-10);
        assert!(!report.mode_interface_qualified);
    }

    #[test]
    fn oscillatory_unstable_mode_oracle_rejects_insufficient_quadrature_information_and_authority()
    {
        let mut poor_sensor = oscillatory_unstable_mode_contract();
        poor_sensor.diagnostic_noise_std = 10.0;
        let sensor_report = evaluate_oscillatory_unstable_mode_interface(poor_sensor).unwrap();
        assert!(
            sensor_report.observability_min_eigenvalue
                < sensor_report.minimum_required_observability_min_eigenvalue
        );
        assert!(!sensor_report.mode_interface_qualified);

        let mut weak_actuator = oscillatory_unstable_mode_contract();
        weak_actuator.actuator_gain = [0.005, 0.002];
        let actuator_report = evaluate_oscillatory_unstable_mode_interface(weak_actuator).unwrap();
        assert!(
            actuator_report.controllability_min_eigenvalue
                < actuator_report.minimum_required_controllability_min_eigenvalue
        );
        assert!(!actuator_report.mode_interface_qualified);
    }

    #[test]
    fn oscillatory_unstable_mode_oracle_rejects_invalid_noise_schedules_and_growth() {
        let mut zero_noise = oscillatory_unstable_mode_contract();
        zero_noise.diagnostic_noise_std = 0.0;
        assert!(matches!(
            evaluate_oscillatory_unstable_mode_interface(zero_noise),
            Err(ContractError::InvalidControlTimingField {
                field: "oscillatory_mode",
                ..
            })
        ));

        let mut invalid_stride = oscillatory_unstable_mode_contract();
        invalid_stride.diagnostic_sampling_stride_steps = 0;
        assert!(matches!(
            evaluate_oscillatory_unstable_mode_interface(invalid_stride),
            Err(ContractError::InvalidControlTimingField {
                field: "oscillatory_mode",
                ..
            })
        ));

        let mut overflow = oscillatory_unstable_mode_contract();
        overflow.growth_rate_per_second = 1_000.0;
        overflow.time_step_seconds = 1.0;
        assert!(matches!(
            evaluate_oscillatory_unstable_mode_interface(overflow),
            Err(ContractError::InvalidControlTimingField {
                field: "oscillatory_mode.growth_rate_per_second",
                ..
            })
        ));
    }

    #[test]
    fn sensor_actuator_timing_auditor_accepts_consistent_asynchronous_fixture() {
        let report =
            validate_sensor_actuator_timing_contract_json(SENSOR_ACTUATOR_FIXTURE).unwrap();
        assert_eq!(
            report.contract_id,
            "KINETIC-MULTIRATE-INTERFACE-SYNTHETIC-001"
        );
        assert_eq!(report.diagnostic_channel_count, 2);
        assert_eq!(report.actuator_channel_count, 1);
        assert!((report.maximum_observation_age_seconds - 0.04).abs() < 1.0e-12);
        assert!((report.maximum_actuator_delivery_delay_seconds - 0.006).abs() < 1.0e-12);
        assert!((report.maximum_actuator_command_latency_seconds - 0.0265).abs() < 1.0e-12);
        assert!((report.minimum_effective_actuator_bandwidth_hz - 20.0).abs() < 1.0e-12);
        assert!((report.conservative_closed_loop_latency_seconds - 0.0715).abs() < 1.0e-12);
        assert!((report.minimum_effective_diagnostic_bandwidth_hz - 20.0).abs() < 1.0e-12);
        assert!((report.target_instability_frequency_hz - 5.0).abs() < 1.0e-12);
        assert!((report.diagnostic_bandwidth_multiple - 4.0).abs() < 1.0e-12);
        assert!((report.actuator_bandwidth_multiple - 4.0).abs() < 1.0e-12);
        assert!((report.target_instability_efold_time_seconds - 0.1).abs() < 1.0e-12);
        assert!((report.loop_latency_efold_fraction - 0.715).abs() < 1.0e-12);
        assert!(report.interface_timescale_qualified);
        assert!((report.controller_update_period_seconds - 0.02).abs() < 1.0e-12);
    }

    #[test]
    fn sensor_actuator_timescale_audit_rejects_underresolved_observation_rate_as_unqualified() {
        let mut value = sensor_actuator_fixture_value();
        value["diagnostics"][1]["sampling_period_seconds"] = serde_json::json!(0.2);
        value["diagnostics"][1]["timestamp_age_bound_seconds"] = serde_json::json!(0.22);
        value["controller"]["maximum_observation_age_seconds"] = serde_json::json!(0.25);
        let report = validate_sensor_actuator_timing_contract_value(&value).unwrap();
        assert!(report.diagnostic_bandwidth_multiple < report.minimum_required_bandwidth_multiple);
        assert!(!report.interface_timescale_qualified);
    }

    #[test]
    fn sensor_actuator_timescale_audit_rejects_loop_latency_too_slow_for_growth_time() {
        let mut value = sensor_actuator_fixture_value();
        value["controller"]["target_instability_efold_time_seconds"] = serde_json::json!(0.05);
        let report = validate_sensor_actuator_timing_contract_value(&value).unwrap();
        assert!(
            report.loop_latency_efold_fraction > report.maximum_allowed_loop_latency_efold_fraction
        );
        assert!(!report.interface_timescale_qualified);
    }

    #[test]
    fn sensor_actuator_timescale_audit_rejects_non_positive_target_frequency_and_margins() {
        let mut zero_frequency = sensor_actuator_fixture_value();
        zero_frequency["controller"]["target_instability_frequency_hz"] = serde_json::json!(0.0);
        assert!(matches!(
            validate_sensor_actuator_timing_contract_value(&zero_frequency),
            Err(ContractError::InvalidControlTimingField {
                field: "controller",
                ..
            })
        ));

        let mut zero_margin = sensor_actuator_fixture_value();
        zero_margin["controller"]["minimum_bandwidth_multiple"] = serde_json::json!(0.0);
        assert!(matches!(
            validate_sensor_actuator_timing_contract_value(&zero_margin),
            Err(ContractError::InvalidControlTimingField {
                field: "controller",
                ..
            })
        ));
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
        let first_channel_id = value["diagnostics"][0]["channel_id"].clone();
        value["diagnostics"][1]["channel_id"] = first_channel_id;
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
        let result =
            project_weighted_micro_fluctuation(&values, &constraints, &weights, 1.0e-11).unwrap();
        assert!(
            result
                .post_projection_moments
                .iter()
                .all(|moment| moment.abs() < 1.0e-10)
        );
        assert!(result.weighted_correction_norm.is_finite());
        assert!(result.weighted_correction_norm > 0.0);
        assert_ne!(result.micro_fluctuation, values);
    }

    #[test]
    fn weighted_micro_projector_is_idempotent() {
        let values = [2.0, -1.0, 3.0, 0.5];
        let weights = [0.5, 1.0, 2.0, 3.0];
        let constraints = vec![vec![1.0, 1.0, 1.0, 1.0], vec![-1.0, 0.0, 0.0, 1.0]];
        let first =
            project_weighted_micro_fluctuation(&values, &constraints, &weights, 1.0e-11).unwrap();
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
    fn weighted_micro_projector_is_invariant_to_constraint_row_scaling() {
        let values = [1.25, -2.0, 0.75, 3.0, -0.5];
        let weights = [0.5, 1.0, 2.0, 1.5, 3.0];
        let base_constraints = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        ];
        let scaled_constraints = vec![
            base_constraints[0]
                .iter()
                .map(|value| value * 7.0)
                .collect(),
            base_constraints[1]
                .iter()
                .map(|value| value * -0.25)
                .collect(),
        ];
        let base =
            project_weighted_micro_fluctuation(&values, &base_constraints, &weights, 1.0e-10)
                .unwrap();
        let scaled =
            project_weighted_micro_fluctuation(&values, &scaled_constraints, &weights, 1.0e-10)
                .unwrap();
        for (left, right) in base.micro_fluctuation.iter().zip(&scaled.micro_fluctuation) {
            assert!((left - right).abs() < 1.0e-9);
        }
    }

    #[test]
    fn weighted_micro_projector_correction_is_orthogonal_to_micro_nullspace() {
        let values = [1.0, -2.0, 0.5, 3.0, -1.5];
        let weights = [1.0, 2.0, 0.5, 3.0, 1.5];
        let constraints = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
        ];
        let result =
            project_weighted_micro_fluctuation(&values, &constraints, &weights, 1.0e-10).unwrap();
        let correction: Vec<f64> = result
            .micro_fluctuation
            .iter()
            .zip(values)
            .map(|(projected, original)| projected - original)
            .collect();
        let micro_tangent = [0.0, 1.0, -2.0, 1.0, 0.0];
        let weighted_inner_product = correction
            .iter()
            .zip(micro_tangent)
            .zip(weights)
            .map(|((delta, tangent), weight)| delta * tangent * weight)
            .sum::<f64>();
        assert!(weighted_inner_product.abs() < 1.0e-9);
    }

    #[test]
    fn weighted_micro_projector_rejects_nearly_dependent_constraints() {
        let values = [1.0, -1.0, 2.0, 0.5];
        let weights = [1.0, 1.0, 1.0, 1.0];
        let constraints = vec![vec![1.0, 1.0, 1.0, 1.0], vec![1.0, 1.0 + 1.0e-14, 1.0, 1.0]];
        assert_eq!(
            project_weighted_micro_fluctuation(&values, &constraints, &weights, 1.0e-10),
            Err(ContractError::InvalidProjectionBasis)
        );
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

    fn make_kinetic_response_signature(
        value: f64,
        contract_id: &str,
        structure_class_id: &str,
        branch_id: Option<&str>,
    ) -> KineticResponseSignature {
        KineticResponseSignature::new(KineticResponseSignatureInput {
            contract_id: contract_id.to_owned(),
            structure_class_id: structure_class_id.to_owned(),
            intervention_ids: vec!["pulse-mode-2-phase-0.3".to_owned()],
            qoi_ids: vec!["electric-field-mode-2".to_owned()],
            qoi_scales: vec![1.0],
            time_samples: vec![0.0, 0.5],
            responses: vec![
                Some(ComplexResponse::new(value, 0.0).unwrap()),
                Some(ComplexResponse::new(value, 0.0).unwrap()),
            ],
            branch_ids: vec![branch_id.map(str::to_owned), branch_id.map(str::to_owned)],
        })
        .unwrap()
    }

    #[test]
    fn kinetic_response_signature_comparison_is_symmetric_with_fixed_contract_scale() {
        let left =
            make_kinetic_response_signature(1.0, "contract-a", "hamiltonian-class-a", Some("echo"));
        let right =
            make_kinetic_response_signature(1.5, "contract-a", "hamiltonian-class-a", Some("echo"));
        let forward = left.compare(&right).unwrap();
        let reverse = right.compare(&left).unwrap();
        assert_eq!(forward.normalized_response_distance, Some(0.5));
        assert_eq!(
            forward.normalized_response_distance,
            reverse.normalized_response_distance
        );
        assert!(KineticResponseSignature::pairwise_compatible(forward, 0.5).unwrap());
    }

    #[test]
    fn kinetic_response_signature_has_a_reproducible_json_receipt() {
        let signature = make_kinetic_response_signature(
            0.25,
            "contract-v1",
            "structure-class-v1",
            Some("echo"),
        );
        let value = serde_json::to_value(&signature).unwrap();
        assert_eq!(value["contract_id"], "contract-v1");
        assert_eq!(value["structure_class_id"], "structure-class-v1");
        assert_eq!(value["intervention_ids"][0], "pulse-mode-2-phase-0.3");
        assert_eq!(value["qoi_ids"][0], "electric-field-mode-2");
        assert_eq!(value["responses"][0]["real"], 0.25);
        assert_eq!(value["branch_ids"][0], "echo");
    }

    #[test]
    fn kinetic_response_signature_rejects_different_task_contract_or_axes() {
        let base = make_kinetic_response_signature(1.0, "contract-a", "class-a", Some("echo"));
        let other_contract =
            make_kinetic_response_signature(1.0, "contract-b", "class-a", Some("echo"));
        assert_eq!(
            base.compare(&other_contract),
            Err(ContractError::ResponseSignatureContractMismatch)
        );
        let other_axes = KineticResponseSignature::new(KineticResponseSignatureInput {
            contract_id: "contract-a".to_owned(),
            structure_class_id: "class-a".to_owned(),
            intervention_ids: vec!["different-pulse".to_owned()],
            qoi_ids: vec!["electric-field-mode-2".to_owned()],
            qoi_scales: vec![1.0],
            time_samples: vec![0.0, 0.5],
            responses: vec![
                Some(ComplexResponse::new(1.0, 0.0).unwrap()),
                Some(ComplexResponse::new(1.0, 0.0).unwrap()),
            ],
            branch_ids: vec![Some("echo".to_owned()), Some("echo".to_owned())],
        })
        .unwrap();
        assert_eq!(
            base.compare(&other_axes),
            Err(ContractError::ResponseSignatureAxisMismatch)
        );
    }

    #[test]
    fn kinetic_response_signature_rejects_missing_responses_and_unresolved_branches_for_merge() {
        let complete = make_kinetic_response_signature(1.0, "contract-a", "class-a", Some("echo"));
        let incomplete = KineticResponseSignature::new(KineticResponseSignatureInput {
            contract_id: "contract-a".to_owned(),
            structure_class_id: "class-a".to_owned(),
            intervention_ids: vec!["pulse-mode-2-phase-0.3".to_owned()],
            qoi_ids: vec!["electric-field-mode-2".to_owned()],
            qoi_scales: vec![1.0],
            time_samples: vec![0.0, 0.5],
            responses: vec![None, Some(ComplexResponse::new(1.0, 0.0).unwrap())],
            branch_ids: vec![None, Some("echo".to_owned())],
        })
        .unwrap();
        let comparison = complete.compare(&incomplete).unwrap();
        assert_eq!(comparison.missing_response_count, 1);
        assert_eq!(comparison.unresolved_branch_count, 1);
        assert!(!KineticResponseSignature::pairwise_compatible(comparison, 1.0).unwrap());
    }

    #[test]
    fn kinetic_response_signature_keeps_structure_and_branch_identity_as_separate_gates() {
        let base =
            make_kinetic_response_signature(1.0, "contract-a", "hamiltonian-class-a", Some("echo"));
        let other_structure = make_kinetic_response_signature(
            1.0,
            "contract-a",
            "different-structure-class",
            Some("echo"),
        );
        let other_branch = make_kinetic_response_signature(
            1.0,
            "contract-a",
            "hamiltonian-class-a",
            Some("trapped"),
        );
        let structure_comparison = base.compare(&other_structure).unwrap();
        let branch_comparison = base.compare(&other_branch).unwrap();
        assert!(!structure_comparison.structure_class_compatible);
        assert!(!KineticResponseSignature::pairwise_compatible(structure_comparison, 1.0).unwrap());
        assert_eq!(branch_comparison.branch_mismatch_count, 2);
        assert!(!KineticResponseSignature::pairwise_compatible(branch_comparison, 1.0).unwrap());
    }

    #[test]
    fn approximate_pairwise_response_compatibility_is_not_transitive() {
        let state_a = make_kinetic_response_signature(0.0, "contract-a", "class-a", Some("branch"));
        let state_b =
            make_kinetic_response_signature(0.75, "contract-a", "class-a", Some("branch"));
        let state_c = make_kinetic_response_signature(1.5, "contract-a", "class-a", Some("branch"));
        let ab = state_a.compare(&state_b).unwrap();
        let bc = state_b.compare(&state_c).unwrap();
        let ac = state_a.compare(&state_c).unwrap();
        assert!(KineticResponseSignature::pairwise_compatible(ab, 1.0).unwrap());
        assert!(KineticResponseSignature::pairwise_compatible(bc, 1.0).unwrap());
        assert!(!KineticResponseSignature::pairwise_compatible(ac, 1.0).unwrap());
    }

    #[test]
    fn kinetic_response_signature_rejects_mismatched_qoi_scales() {
        let left = KineticResponseSignature::new(KineticResponseSignatureInput {
            contract_id: "contract".to_owned(),
            structure_class_id: "class".to_owned(),
            intervention_ids: vec!["pulse".to_owned()],
            qoi_ids: vec!["electric-field".to_owned()],
            qoi_scales: vec![1.0],
            time_samples: vec![0.0],
            responses: vec![Some(ComplexResponse::new(1.0, 0.0).unwrap())],
            branch_ids: vec![Some("branch".to_owned())],
        })
        .unwrap();
        let right = KineticResponseSignature::new(KineticResponseSignatureInput {
            contract_id: "contract".to_owned(),
            structure_class_id: "class".to_owned(),
            intervention_ids: vec!["pulse".to_owned()],
            qoi_ids: vec!["electric-field".to_owned()],
            qoi_scales: vec![2.0],
            time_samples: vec![0.0],
            responses: vec![Some(ComplexResponse::new(1.0, 0.0).unwrap())],
            branch_ids: vec![Some("branch".to_owned())],
        })
        .unwrap();
        assert_eq!(
            left.compare(&right),
            Err(ContractError::ResponseSignatureScaleMismatch)
        );
    }

    #[test]
    fn kinetic_response_signature_uses_each_qoi_scale_before_rms() {
        let left = KineticResponseSignature::new(KineticResponseSignatureInput {
            contract_id: "contract".to_owned(),
            structure_class_id: "class".to_owned(),
            intervention_ids: vec!["pulse".to_owned()],
            qoi_ids: vec!["electric-field".to_owned(), "energy".to_owned()],
            qoi_scales: vec![1.0, 10.0],
            time_samples: vec![0.0],
            responses: vec![
                Some(ComplexResponse::new(1.0, 0.0).unwrap()),
                Some(ComplexResponse::new(1.0, 0.0).unwrap()),
            ],
            branch_ids: vec![Some("branch".to_owned()), Some("branch".to_owned())],
        })
        .unwrap();
        let right = KineticResponseSignature::new(KineticResponseSignatureInput {
            contract_id: "contract".to_owned(),
            structure_class_id: "class".to_owned(),
            intervention_ids: vec!["pulse".to_owned()],
            qoi_ids: vec!["electric-field".to_owned(), "energy".to_owned()],
            qoi_scales: vec![1.0, 10.0],
            time_samples: vec![0.0],
            responses: vec![
                Some(ComplexResponse::new(0.0, 0.0).unwrap()),
                Some(ComplexResponse::new(0.0, 0.0).unwrap()),
            ],
            branch_ids: vec![Some("branch".to_owned()), Some("branch".to_owned())],
        })
        .unwrap();
        let compared = left.compare(&right).unwrap();
        let expected = ((1.0_f64.powi(2) + 0.1_f64.powi(2)) / 2.0).sqrt();
        assert!((compared.normalized_response_distance.unwrap() - expected).abs() < 1.0e-12);
    }

    #[test]
    fn response_signature_admission_uses_worst_entry_not_rms_average() {
        let left = KineticResponseSignature::new(KineticResponseSignatureInput {
            contract_id: "contract".to_owned(),
            structure_class_id: "class".to_owned(),
            intervention_ids: vec!["pulse".to_owned()],
            qoi_ids: vec!["field".to_owned()],
            qoi_scales: vec![1.0],
            time_samples: (0..10).map(f64::from).collect(),
            responses: (0..10)
                .map(|_| Some(ComplexResponse::new(0.0, 0.0).unwrap()))
                .collect(),
            branch_ids: (0..10).map(|_| Some("branch".to_owned())).collect(),
        })
        .unwrap();
        let mut right_input = KineticResponseSignatureInput {
            contract_id: "contract".to_owned(),
            structure_class_id: "class".to_owned(),
            intervention_ids: vec!["pulse".to_owned()],
            qoi_ids: vec!["field".to_owned()],
            qoi_scales: vec![1.0],
            time_samples: (0..10).map(f64::from).collect(),
            responses: (0..10)
                .map(|_| Some(ComplexResponse::new(0.0, 0.0).unwrap()))
                .collect(),
            branch_ids: (0..10).map(|_| Some("branch".to_owned())).collect(),
        };
        right_input.responses[9] = Some(ComplexResponse::new(2.0, 0.0).unwrap());
        let right = KineticResponseSignature::new(right_input).unwrap();
        let comparison = left.compare(&right).unwrap();
        assert!(comparison.normalized_response_distance.unwrap() < 1.0);
        assert_eq!(comparison.maximum_normalized_entry_error, Some(2.0));
        assert!(!KineticResponseSignature::pairwise_compatible(comparison, 1.0).unwrap());
    }

    #[test]
    fn response_signature_cluster_audit_rejects_nontransitive_chain_merge() {
        let state_a = make_kinetic_response_signature(0.0, "contract-a", "class-a", Some("branch"));
        let state_b =
            make_kinetic_response_signature(0.75, "contract-a", "class-a", Some("branch"));
        let state_c = make_kinetic_response_signature(1.5, "contract-a", "class-a", Some("branch"));
        assert!(KineticResponseSignature::pairwise_compatible(
            state_a.compare(&state_b).unwrap(),
            1.0
        )
        .unwrap());
        assert!(KineticResponseSignature::pairwise_compatible(
            state_b.compare(&state_c).unwrap(),
            1.0
        )
        .unwrap());
        let audit = audit_response_signature_cluster(&[state_a, state_b, state_c], 1.0).unwrap();
        assert_eq!(audit.member_count, 3);
        assert_eq!(audit.compared_pair_count, 3);
        assert_eq!(audit.compatible_pair_count, 2);
        assert_eq!(audit.incompatible_pair_count, 1);
        assert_eq!(audit.maximum_pairwise_normalized_entry_error, Some(1.5));
        assert!(!audit.within_finite_class_diameter);
    }

    #[test]
    fn response_signature_cluster_audit_accepts_only_complete_bounded_classes() {
        let state_a = make_kinetic_response_signature(0.0, "contract-a", "class-a", Some("branch"));
        let state_b =
            make_kinetic_response_signature(0.25, "contract-a", "class-a", Some("branch"));
        let audit = audit_response_signature_cluster(&[state_a, state_b], 0.25).unwrap();
        assert_eq!(audit.compared_pair_count, 1);
        assert_eq!(audit.compatible_pair_count, 1);
        assert!(audit.complete_response_coverage);
        assert!(audit.stable_branch_association);
        assert!(audit.compatible_structure_class);
        assert!(audit.within_finite_class_diameter);
    }

    #[test]
    fn response_signature_cluster_audit_rejects_missing_or_unresolved_response() {
        let incomplete = KineticResponseSignatureInput {
            contract_id: "contract".to_owned(),
            structure_class_id: "class".to_owned(),
            intervention_ids: vec!["pulse".to_owned()],
            qoi_ids: vec!["field".to_owned()],
            qoi_scales: vec![1.0],
            time_samples: vec![0.0],
            responses: vec![None],
            branch_ids: vec![None],
        };
        let left = make_kinetic_response_signature(0.0, "contract", "class", Some("branch"));
        let right = KineticResponseSignature::new(incomplete.clone()).unwrap();
        let audit = audit_response_signature_cluster(&[left, right], 1.0).unwrap();
        assert_eq!(audit.incompatible_pair_count, 1);
        assert!(!audit.complete_response_coverage);
        assert!(!audit.stable_branch_association);
        assert!(!audit.within_finite_class_diameter);
    }

    #[test]
    fn kinetic_discrimination_known_answer_fixture_matches_eig_bounds() {
        let fixture: serde_json::Value =
            serde_json::from_str(KINETIC_DISCRIMINATION_FIXTURE).unwrap();
        assert_eq!(fixture["status"], "synthetic_analytic_only");
        assert_eq!(fixture["quadrature_order_per_axis"], 9);
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 4);

        for case in cases {
            let hypotheses = case["hypotheses"]
                .as_array()
                .unwrap()
                .iter()
                .map(|hypothesis| KineticDiscriminationHypothesis {
                    hypothesis_id: hypothesis["hypothesis_id"].as_str().unwrap().to_owned(),
                    prior_probability: hypothesis["prior_probability"].as_f64().unwrap(),
                    predicted_response: ComplexResponse::new(
                        hypothesis["predicted_response"]["real"].as_f64().unwrap(),
                        hypothesis["predicted_response"]["imag"].as_f64().unwrap(),
                    )
                    .unwrap(),
                    response_branch_id: hypothesis["response_branch_id"]
                        .as_str()
                        .unwrap()
                        .to_owned(),
                    branch_association_stable: hypothesis["branch_association_stable"]
                        .as_bool()
                        .unwrap(),
                })
                .collect();
            let design = KineticDiscriminationDesign {
                intervention_id: case["intervention_id"].as_str().unwrap().to_owned(),
                hypotheses,
                noise_variance_per_quadrature: case["noise_variance_per_quadrature"]
                    .as_f64()
                    .unwrap(),
                estimated_evaluation_cost_seconds: case["estimated_evaluation_cost_seconds"]
                    .as_f64()
                    .unwrap(),
            };
            let report = evaluate_kinetic_discrimination_design(&design).unwrap();
            let expected_prior_entropy =
                case["expected_prior_entropy_bits"].as_f64().unwrap();
            let minimum_eig = case["expected_information_gain_min_bits"].as_f64().unwrap();
            let maximum_eig = case["expected_information_gain_max_bits"].as_f64().unwrap();
            assert!(
                (report.prior_entropy_bits - expected_prior_entropy).abs() < 1.0e-10,
                "prior entropy mismatch for {}",
                case["case_id"].as_str().unwrap()
            );
            assert!(
                report.expected_information_gain_bits + 1.0e-10 >= minimum_eig
                    && report.expected_information_gain_bits <= maximum_eig + 1.0e-10,
                "EIG outside known-answer bounds for {}: {}",
                case["case_id"].as_str().unwrap(),
                report.expected_information_gain_bits
            );
            assert!(
                (report.prior_entropy_bits
                    - report.expected_information_gain_bits
                    - report.expected_posterior_entropy_bits)
                    .abs()
                    < 1.0e-12
            );
            assert!(
                (report.information_gain_bits_per_cost_second
                    - report.expected_information_gain_bits
                        / report.estimated_evaluation_cost_seconds)
                    .abs()
                    < 1.0e-12
            );
        }
    }

    #[test]
    fn kinetic_discrimination_portfolio_fixture_matches_expected_pareto_front() {
        let fixture: serde_json::Value =
            serde_json::from_str(KINETIC_DISCRIMINATION_FIXTURE).unwrap();
        let schema: serde_json::Value =
            serde_json::from_str(KINETIC_DISCRIMINATION_SCHEMA).unwrap();
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );

        let portfolios = fixture["portfolios"].as_array().unwrap();
        assert_eq!(portfolios.len(), 1);
        for portfolio in portfolios {
            let designs = portfolio["candidate_designs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|candidate| {
                    let hypotheses = candidate["hypotheses"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|hypothesis| KineticDiscriminationHypothesis {
                            hypothesis_id: hypothesis["hypothesis_id"]
                                .as_str()
                                .unwrap()
                                .to_owned(),
                            prior_probability: hypothesis["prior_probability"]
                                .as_f64()
                                .unwrap(),
                            predicted_response: ComplexResponse::new(
                                hypothesis["predicted_response"]["real"]
                                    .as_f64()
                                    .unwrap(),
                                hypothesis["predicted_response"]["imag"]
                                    .as_f64()
                                    .unwrap(),
                            )
                            .unwrap(),
                            response_branch_id: hypothesis["response_branch_id"]
                                .as_str()
                                .unwrap()
                                .to_owned(),
                            branch_association_stable: hypothesis["branch_association_stable"]
                                .as_bool()
                                .unwrap(),
                        })
                        .collect();
                    KineticDiscriminationDesign {
                        intervention_id: candidate["intervention_id"]
                            .as_str()
                            .unwrap()
                            .to_owned(),
                        hypotheses,
                        noise_variance_per_quadrature: candidate
                            ["noise_variance_per_quadrature"]
                            .as_f64()
                            .unwrap(),
                        estimated_evaluation_cost_seconds: candidate
                            ["estimated_evaluation_cost_seconds"]
                            .as_f64()
                            .unwrap(),
                    }
                })
                .collect::<Vec<_>>();
            let report = evaluate_kinetic_discrimination_portfolio(&designs).unwrap();
            let actual_front = report
                .pareto_front_indices
                .iter()
                .map(|index| report.candidate_reports[*index].intervention_id.as_str())
                .collect::<Vec<_>>();
            let expected_front = portfolio["expected_pareto_front_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(actual_front, expected_front);
            assert_eq!(
                report.candidate_reports[report.best_information_gain_per_cost_index]
                    .intervention_id,
                portfolio["expected_best_ratio_id"].as_str().unwrap()
            );
        }
    }

    fn gaussian_discrimination_design(
        responses: &[(f64, f64)],
        priors: &[f64],
        variance: f64,
        cost_seconds: f64,
    ) -> KineticDiscriminationDesign {
        assert_eq!(responses.len(), priors.len());
        KineticDiscriminationDesign {
            intervention_id: "manufactured-pulse-probe-v1".to_owned(),
            hypotheses: responses
                .iter()
                .zip(priors)
                .enumerate()
                .map(|(index, ((real, imag), prior))| KineticDiscriminationHypothesis {
                    hypothesis_id: format!("state-{index}"),
                    prior_probability: *prior,
                    predicted_response: ComplexResponse::new(*real, *imag).unwrap(),
                    response_branch_id: format!("branch-{index}"),
                    branch_association_stable: true,
                })
                .collect(),
            noise_variance_per_quadrature: variance,
            estimated_evaluation_cost_seconds: cost_seconds,
        }
    }

    #[test]
    fn kinetic_discrimination_noise_envelope_reports_worst_and_best_endpoints() {
        let design = gaussian_discrimination_design(
            &[(-1.0, 0.25), (1.0, -0.25)],
            &[0.5, 0.5],
            0.1,
            2.0,
        );
        let report =
            evaluate_kinetic_discrimination_noise_envelope(&design, 0.02, 4.0).unwrap();
        assert!(report.best_case_endpoint_information_gain_bits
            > report.worst_case_endpoint_information_gain_bits);
        assert!((report.best_case_endpoint_information_gain_bits_per_cost_second
            - report.best_case_endpoint_information_gain_bits / 2.0)
            .abs()
            < 1.0e-12);
        assert!((report.worst_case_endpoint_information_gain_bits_per_cost_second
            - report.worst_case_endpoint_information_gain_bits / 2.0)
            .abs()
            < 1.0e-12);
        assert!(report.monotone_noise_degradation_check);
    }

    #[test]
    fn kinetic_discrimination_noise_envelope_rejects_invalid_ranges() {
        let design = gaussian_discrimination_design(
            &[(-1.0, 0.0), (1.0, 0.0)],
            &[0.5, 0.5],
            0.1,
            1.0,
        );
        assert_eq!(
            evaluate_kinetic_discrimination_noise_envelope(&design, 0.0, 1.0),
            Err(ContractError::InvalidDiscriminationNoiseEnvelope)
        );
        assert_eq!(
            evaluate_kinetic_discrimination_noise_envelope(&design, 2.0, 1.0),
            Err(ContractError::InvalidDiscriminationNoiseEnvelope)
        );
        assert_eq!(
            evaluate_kinetic_discrimination_noise_envelope(&design, f64::NAN, 2.0),
            Err(ContractError::InvalidDiscriminationNoiseEnvelope)
        );
    }

    #[test]
    fn kinetic_discrimination_portfolio_returns_pareto_front_and_best_ratio() {
        let priors = [0.5, 0.5];
        let weak = gaussian_discrimination_design(
            &[(0.0, 0.0), (0.2, 0.0)],
            &priors,
            0.5,
            1.0,
        );
        let strong_expensive = gaussian_discrimination_design(
            &[(-2.0, 0.0), (2.0, 0.0)],
            &priors,
            0.5,
            10.0,
        );
        let dominated = gaussian_discrimination_design(
            &[(0.0, 0.0), (0.1, 0.0)],
            &priors,
            0.5,
            2.0,
        );
        let strong_cheap = gaussian_discrimination_design(
            &[(-2.0, 0.0), (2.0, 0.0)],
            &priors,
            0.5,
            2.0,
        );
        let mut designs = vec![weak, strong_expensive, dominated, strong_cheap];
        for (index, design) in designs.iter_mut().enumerate() {
            design.intervention_id = format!("pulse-{index}");
        }
        let report = evaluate_kinetic_discrimination_portfolio(&designs).unwrap();
        assert!(report.pareto_front_indices.contains(&0));
        assert!(!report.pareto_front_indices.contains(&2));
        assert!(report.pareto_front_indices.contains(&3));
        assert_eq!(report.best_information_gain_per_cost_index, 3);
    }

    #[test]
    fn kinetic_discrimination_portfolio_rejects_changed_hypothesis_priors_or_noise() {
        let priors = [0.5, 0.5];
        let first =
            gaussian_discrimination_design(&[(-1.0, 0.0), (1.0, 0.0)], &priors, 0.1, 1.0);
        let mut changed_prior =
            gaussian_discrimination_design(&[(-0.5, 0.0), (0.5, 0.0)], &[0.9, 0.1], 0.1, 1.0);
        changed_prior.intervention_id = "different-prior".to_owned();
        assert_eq!(
            evaluate_kinetic_discrimination_portfolio(&[first.clone(), changed_prior]),
            Err(ContractError::IncomparableDiscriminationDesigns)
        );
        let mut changed_noise =
            gaussian_discrimination_design(&[(-0.5, 0.0), (0.5, 0.0)], &priors, 0.2, 1.0);
        changed_noise.intervention_id = "different-noise".to_owned();
        assert_eq!(
            evaluate_kinetic_discrimination_portfolio(&[first, changed_noise]),
            Err(ContractError::IncomparableDiscriminationDesigns)
        );
    }

    #[test]
    fn kinetic_discrimination_portfolio_rejects_duplicates_and_empty_sets() {
        let design = gaussian_discrimination_design(
            &[(-1.0, 0.0), (1.0, 0.0)],
            &[0.5, 0.5],
            0.1,
            1.0,
        );
        assert_eq!(
            evaluate_kinetic_discrimination_portfolio(&[]),
            Err(ContractError::EmptyDiscriminationPortfolio)
        );
        assert_eq!(
            evaluate_kinetic_discrimination_portfolio(&[design.clone(), design]),
            Err(ContractError::DuplicateDiscriminationIntervention)
        );
    }

    #[test]
    fn kinetic_discrimination_eig_is_zero_for_identical_predictions() {
        let design = gaussian_discrimination_design(
            &[(1.0, -0.25), (1.0, -0.25), (1.0, -0.25)],
            &[0.2, 0.3, 0.5],
            0.1,
            2.0,
        );
        let report = evaluate_kinetic_discrimination_design(&design).unwrap();
        assert!(report.expected_information_gain_bits.abs() < 1.0e-12);
        assert!((report.expected_posterior_entropy_bits - report.prior_entropy_bits).abs() < 1.0e-12);
    }

    #[test]
    fn kinetic_discrimination_eig_detects_phase_only_response_differences() {
        let design = gaussian_discrimination_design(
            &[(1.0, 0.0), (0.0, 1.0)],
            &[0.5, 0.5],
            0.02,
            1.0,
        );
        let report = evaluate_kinetic_discrimination_design(&design).unwrap();
        assert!(report.expected_information_gain_bits > 0.95);
        assert!(report.expected_information_gain_bits <= report.prior_entropy_bits + 1.0e-12);
        assert!(report.expected_posterior_entropy_bits < 0.05);
    }

    #[test]
    fn kinetic_discrimination_eig_decreases_as_observation_noise_increases() {
        let responses = [(-1.0, 0.25), (1.0, -0.25)];
        let priors = [0.5, 0.5];
        let low_noise = evaluate_kinetic_discrimination_design(
            &gaussian_discrimination_design(&responses, &priors, 0.02, 1.0),
        )
        .unwrap();
        let high_noise = evaluate_kinetic_discrimination_design(
            &gaussian_discrimination_design(&responses, &priors, 4.0, 1.0),
        )
        .unwrap();
        assert!(low_noise.expected_information_gain_bits > high_noise.expected_information_gain_bits);
        assert!(high_noise.expected_information_gain_bits < 0.2);
    }

    #[test]
    fn kinetic_discrimination_eig_respects_imbalanced_priors() {
        let report = evaluate_kinetic_discrimination_design(&gaussian_discrimination_design(
            &[(-2.0, 0.0), (2.0, 0.0)],
            &[0.9, 0.1],
            0.001,
            1.0,
        ))
        .unwrap();
        let expected_entropy = -(0.9_f64 * 0.9_f64.log2() + 0.1_f64 * 0.1_f64.log2());
        assert!((report.prior_entropy_bits - expected_entropy).abs() < 1.0e-12);
        assert!(report.expected_information_gain_bits <= expected_entropy + 1.0e-12);
        assert!(report.expected_information_gain_bits > 0.3);
    }

    #[test]
    fn kinetic_discrimination_cost_changes_only_cost_normalized_utility() {
        let responses = [(-1.0, 0.0), (1.0, 0.0)];
        let priors = [0.5, 0.5];
        let cheap = evaluate_kinetic_discrimination_design(&gaussian_discrimination_design(
            &responses, &priors, 0.1, 2.0,
        ))
        .unwrap();
        let expensive = evaluate_kinetic_discrimination_design(&gaussian_discrimination_design(
            &responses, &priors, 0.1, 8.0,
        ))
        .unwrap();
        assert!((cheap.expected_information_gain_bits - expensive.expected_information_gain_bits).abs() < 1.0e-12);
        assert!((cheap.information_gain_bits_per_cost_second
            - 4.0 * expensive.information_gain_bits_per_cost_second)
            .abs()
            < 1.0e-12);
    }

    #[test]
    fn kinetic_discrimination_rejects_unstable_or_unidentified_response_branches() {
        let mut design = gaussian_discrimination_design(
            &[(-1.0, 0.0), (1.0, 0.0)],
            &[0.5, 0.5],
            0.1,
            1.0,
        );
        design.hypotheses[0].branch_association_stable = false;
        assert_eq!(
            evaluate_kinetic_discrimination_design(&design),
            Err(ContractError::InvalidDiscriminationResponseBranch)
        );
        design.hypotheses[0].branch_association_stable = true;
        design.hypotheses[0].response_branch_id.clear();
        assert_eq!(
            evaluate_kinetic_discrimination_design(&design),
            Err(ContractError::InvalidDiscriminationResponseBranch)
        );
    }

    #[test]
    fn kinetic_discrimination_normalizes_extreme_prior_magnitudes_safely() {
        let design = gaussian_discrimination_design(
            &[(-1.0, 0.0), (1.0, 0.0)],
            &[1.0e308, 1.0e308],
            0.1,
            1.0,
        );
        let report = evaluate_kinetic_discrimination_design(&design).unwrap();
        assert!((report.prior_entropy_bits - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn kinetic_discrimination_rejects_invalid_designs_and_bounded_work_overflow() {
        let mut design = gaussian_discrimination_design(
            &[(-1.0, 0.0), (1.0, 0.0)],
            &[0.5, 0.5],
            0.1,
            1.0,
        );
        design.intervention_id.clear();
        assert_eq!(
            evaluate_kinetic_discrimination_design(&design),
            Err(ContractError::InvalidDiscriminationHypothesisIdentity)
        );
        design.intervention_id = "probe".to_owned();
        design.noise_variance_per_quadrature = 0.0;
        assert_eq!(
            evaluate_kinetic_discrimination_design(&design),
            Err(ContractError::InvalidDiscriminationNoiseVariance)
        );
        design.noise_variance_per_quadrature = 0.1;
        design.estimated_evaluation_cost_seconds = f64::NAN;
        assert_eq!(
            evaluate_kinetic_discrimination_design(&design),
            Err(ContractError::InvalidDiscriminationCost)
        );
        design.estimated_evaluation_cost_seconds = 1.0;
        design.hypotheses[0].prior_probability = 0.0;
        assert_eq!(
            evaluate_kinetic_discrimination_design(&design),
            Err(ContractError::InvalidDiscriminationPrior)
        );
        design.hypotheses[0].prior_probability = 0.5;
        design.hypotheses[1].hypothesis_id = design.hypotheses[0].hypothesis_id.clone();
        assert_eq!(
            evaluate_kinetic_discrimination_design(&design),
            Err(ContractError::InvalidDiscriminationHypothesisIdentity)
        );
        design.hypotheses[1].hypothesis_id = "second".to_owned();
        design.hypotheses = (0..=MAX_KINETIC_DISCRIMINATION_HYPOTHESES)
            .map(|index| KineticDiscriminationHypothesis {
                hypothesis_id: format!("state-{index}"),
                prior_probability: 1.0,
                predicted_response: ComplexResponse::new(index as f64, 0.0).unwrap(),
                response_branch_id: format!("branch-{index}"),
                branch_association_stable: true,
            })
            .collect();
        assert_eq!(
            evaluate_kinetic_discrimination_design(&design),
            Err(ContractError::DiscriminationWorkBudgetExceeded)
        );
    }

    #[test]
    fn response_signature_cluster_audit_is_bounded_and_fail_closed() {
        assert_eq!(
            audit_response_signature_cluster(&[], 1.0),
            Err(ContractError::EmptyResponseSignatureCluster)
        );
        let signatures = (0..=MAX_RESPONSE_SIGNATURE_CLUSTER_MEMBERS)
            .map(|_| make_kinetic_response_signature(0.0, "contract", "class", Some("branch")))
            .collect::<Vec<_>>();
        assert_eq!(
            audit_response_signature_cluster(&signatures, 1.0),
            Err(ContractError::ResponseSignatureClusterWorkBudgetExceeded)
        );
        let one = [make_kinetic_response_signature(0.0, "contract", "class", Some("branch"))];
        assert!(audit_response_signature_cluster(&one, f64::NAN).is_err());
    }

    #[test]
    fn kinetic_response_signature_requires_valid_axes_times_and_scale() {
        assert_eq!(
            KineticResponseSignature::new(KineticResponseSignatureInput {
                contract_id: "contract".to_owned(),
                structure_class_id: "class".to_owned(),
                intervention_ids: vec!["pulse".to_owned()],
                qoi_ids: vec!["qoi".to_owned()],
                qoi_scales: vec![1.0],
                time_samples: vec![0.0, 0.0],
                responses: vec![
                    Some(ComplexResponse::new(1.0, 0.0).unwrap()),
                    Some(ComplexResponse::new(1.0, 0.0).unwrap())
                ],
                branch_ids: vec![Some("branch".to_owned()), Some("branch".to_owned())],
            })
            .unwrap_err(),
            ContractError::InvalidResponseSignatureTimes
        );
        let invalid_scale = KineticResponseSignature::new(KineticResponseSignatureInput {
            contract_id: "contract".to_owned(),
            structure_class_id: "class".to_owned(),
            intervention_ids: vec!["pulse".to_owned()],
            qoi_ids: vec!["qoi".to_owned()],
            qoi_scales: vec![0.0],
            time_samples: vec![0.0],
            responses: vec![Some(ComplexResponse::new(1.0, 0.0).unwrap())],
            branch_ids: vec![Some("branch".to_owned())],
        });
        assert_eq!(
            invalid_scale.unwrap_err(),
            ContractError::InvalidResponseSignatureScale
        );
    }

    const COUPLED_TWO_MODE_AUTHORITY_FIXTURE: &str = include_str!(
        "../../../../benchmarks/plasma/control-interface/fixtures/coupled-unstable-mode-authority-known-answer-v0.1.json"
    );

    fn coupled_two_mode_contract(
        transition_matrix: [f64; 4],
        diagnostic_gain: [[f64; 2]; 2],
        actuator_gain: [f64; 2],
    ) -> CoupledTwoModeAuthorityContract {
        CoupledTwoModeAuthorityContract {
            transition_matrix,
            horizon_steps: 12,
            diagnostic_gain,
            diagnostic_noise_variance: [1.0, 1.0],
            diagnostic_sampling_stride_steps: 1,
            diagnostic_delivery_delay_steps: 0,
            actuator_gain,
            actuator_update_stride_steps: 1,
            actuator_delivery_delay_steps: 0,
            minimum_observability_eigenvalue: 1.0e-6,
            minimum_controllability_eigenvalue: 1.0e-6,
        }
    }

    fn maxwell_delay_observation_fixture() -> MaxwellDelayCancellationObservation {
        MaxwellDelayCancellationObservation {
            delayed_electric_longitudinal: vec![0.6, -0.2],
            delayed_electric_transverse: vec![1.2, 0.4],
            delayed_magnetic_transverse: vec![-0.4, 0.8],
            interval_mean_current_longitudinal: vec![0.8, -0.4],
            interval_mean_current_transverse: vec![-0.2, 0.6],
            interval_mean_gradient_electric_transverse: vec![-1.0, 0.2],
            interval_mean_gradient_magnetic_transverse: vec![0.6, -0.8],
        }
    }

    fn maxwell_delay_history_fixture() -> MaxwellDelayHistorySamples {
        MaxwellDelayHistorySamples {
            sample_times_seconds: vec![2.0, 2.25, 3.0],
            current_longitudinal: vec![vec![1.0, 2.0], vec![1.25, 2.5], vec![2.0, 4.0]],
            current_transverse: vec![vec![-1.0, 0.0], vec![-0.75, 0.5], vec![0.0, 2.0]],
            gradient_electric_transverse: vec![vec![2.0, -2.0], vec![1.5, -1.5], vec![0.0, 0.0]],
            gradient_magnetic_transverse: vec![vec![0.0, 1.0], vec![0.5, 1.25], vec![2.0, 2.0]],
        }
    }

    fn periodic_derivative_config(
        sample_count: usize,
        maximum_harmonic: usize,
        tikhonov_strength: f64,
        measurement_noise_variance: f64,
    ) -> PeriodicDerivativeConfig {
        assert!(sample_count >= 3);
        PeriodicDerivativeConfig {
            domain_length: 2.5,
            maximum_harmonic,
            tikhonov_strength,
            measurement_noise_variance,
        }
    }

    #[test]
    fn periodic_derivative_recovers_manufactured_fourier_modes() {
        let sample_count = 33;
        let length = 2.5;
        let samples = (0..sample_count)
            .map(|index| {
                let phase = TAU * index as f64 / sample_count as f64;
                1.7 * (2.0 * phase).cos() - 0.35 * (3.0 * phase).sin() + 4.2
            })
            .collect::<Vec<_>>();
        let estimate = estimate_periodic_derivative(
            &samples,
            periodic_derivative_config(sample_count, 4, 0.0, 0.0),
        )
        .unwrap();
        for (index, derivative) in estimate.derivative.iter().enumerate() {
            let phase = TAU * index as f64 / sample_count as f64;
            let expected = -1.7 * (2.0 * TAU / length) * (2.0 * phase).sin()
                - 0.35 * (3.0 * TAU / length) * (3.0 * phase).cos();
            assert!((derivative - expected).abs() < 1.0e-11);
        }
        assert!(estimate.white_noise_variance_gain.is_finite());
        assert_eq!(estimate.filter_weights, vec![1.0; 4]);
    }

    #[test]
    fn periodic_derivative_reports_the_analytic_white_noise_gain() {
        let sample_count = 9;
        let config = periodic_derivative_config(sample_count, 3, 0.25, 0.04);
        let samples = vec![0.0; sample_count];
        let estimate = estimate_periodic_derivative(&samples, config).unwrap();
        let expected_gain = (1..=3)
            .map(|harmonic| {
                let k = harmonic as f64;
                let weight = 1.0 / (1.0 + 0.25 * k.powi(4));
                (2.0 / sample_count as f64)
                    * (TAU * k / config.domain_length).powi(2)
                    * weight.powi(2)
            })
            .sum::<f64>();
        assert!((estimate.white_noise_variance_gain - expected_gain).abs() < 1.0e-12);
        assert!(
            (estimate.predicted_white_noise_variance
                - config.measurement_noise_variance * expected_gain)
                .abs()
                < 1.0e-12
        );
        assert!(
            estimate
                .derivative
                .iter()
                .all(|value| value.abs() < 1.0e-12)
        );
    }

    #[test]
    fn periodic_derivative_regularization_reduces_white_noise_gain() {
        let samples = (0..33)
            .map(|index| {
                let phase = TAU * index as f64 / 33.0;
                (9.0 * phase).sin() + 0.1 * (2.0 * phase).cos()
            })
            .collect::<Vec<_>>();
        let raw =
            estimate_periodic_derivative(&samples, periodic_derivative_config(33, 12, 0.0, 0.2))
                .unwrap();
        let regularized =
            estimate_periodic_derivative(&samples, periodic_derivative_config(33, 12, 0.5, 0.2))
                .unwrap();
        assert!(regularized.white_noise_variance_gain < raw.white_noise_variance_gain);
        assert!(regularized.predicted_white_noise_variance < raw.predicted_white_noise_variance);
        assert!(
            regularized
                .filter_weights
                .iter()
                .zip(&raw.filter_weights)
                .all(|(filtered, unfiltered)| filtered <= unfiltered)
        );
    }

    #[test]
    fn periodic_derivative_is_equivariant_to_cyclic_grid_translations() {
        let sample_count = 31;
        let samples = (0..sample_count)
            .map(|index| {
                let phase = TAU * index as f64 / sample_count as f64;
                0.8 * phase.cos() + 0.4 * (3.0 * phase).sin()
            })
            .collect::<Vec<_>>();
        let shifted = samples
            .iter()
            .cycle()
            .skip(7)
            .take(sample_count)
            .copied()
            .collect::<Vec<_>>();
        let config = periodic_derivative_config(sample_count, 8, 0.03, 0.1);
        let base = estimate_periodic_derivative(&samples, config).unwrap();
        let translated = estimate_periodic_derivative(&shifted, config).unwrap();
        for index in 0..sample_count {
            let translated_index = (index + sample_count - 7) % sample_count;
            assert!(
                (base.derivative[index] - translated.derivative[translated_index]).abs() < 1.0e-10
            );
        }
    }

    #[test]
    fn periodic_derivative_rejects_invalid_grid_noise_and_work_budgets() {
        let samples = vec![0.0; 8];
        assert!(matches!(
            estimate_periodic_derivative(
                &samples,
                PeriodicDerivativeConfig {
                    domain_length: 0.0,
                    maximum_harmonic: 1,
                    tikhonov_strength: 0.0,
                    measurement_noise_variance: 0.0,
                }
            ),
            Err(ContractError::InvalidPeriodicDerivativeConfig(_))
        ));
        assert!(matches!(
            estimate_periodic_derivative(&samples, periodic_derivative_config(8, 4, 0.0, 0.0)),
            Err(ContractError::InvalidPeriodicDerivativeConfig(_))
        ));
        assert!(matches!(
            estimate_periodic_derivative(&samples, periodic_derivative_config(8, 1, -0.1, 0.0)),
            Err(ContractError::InvalidPeriodicDerivativeConfig(_))
        ));
        assert!(matches!(
            estimate_periodic_derivative(&samples, periodic_derivative_config(8, 1, 0.0, f64::NAN)),
            Err(ContractError::InvalidPeriodicDerivativeConfig(_))
        ));
        let mut non_finite = samples.clone();
        non_finite[2] = f64::INFINITY;
        assert!(matches!(
            estimate_periodic_derivative(&non_finite, periodic_derivative_config(8, 1, 0.0, 0.0)),
            Err(ContractError::InvalidPeriodicDerivativeConfig(_))
        ));
        let maximum_samples = MAX_PERIODIC_DERIVATIVE_SAMPLES;
        let excessive = vec![0.0; maximum_samples];
        assert_eq!(
            estimate_periodic_derivative(
                &excessive,
                periodic_derivative_config(maximum_samples, (maximum_samples - 1) / 2, 0.0, 0.0)
            ),
            Err(ContractError::PeriodicDerivativeWorkBudgetExceeded)
        );
    }

    #[test]
    fn maxwell_delay_history_integrates_nonuniform_time_samples() {
        let averages =
            integrate_maxwell_delay_history(&maxwell_delay_history_fixture(), 2.0, 3.0).unwrap();
        assert_eq!(averages.current_longitudinal, vec![1.5, 3.0]);
        assert_eq!(averages.current_transverse, vec![-0.5, 1.0]);
        assert_eq!(averages.gradient_electric_transverse, vec![1.0, -1.0]);
        assert_eq!(averages.gradient_magnetic_transverse, vec![1.0, 1.5]);
    }

    #[test]
    fn maxwell_delay_history_rejects_missing_endpoints_and_nonmonotone_time() {
        let history = maxwell_delay_history_fixture();
        assert!(matches!(
            integrate_maxwell_delay_history(&history, 2.1, 3.0),
            Err(ContractError::InvalidControlTimingField {
                field: "delay_history",
                ..
            })
        ));
        let mut repeated_time = history;
        repeated_time.sample_times_seconds[1] = 2.0;
        assert!(matches!(
            integrate_maxwell_delay_history(&repeated_time, 2.0, 3.0),
            Err(ContractError::InvalidControlTimingField {
                field: "delay_history.sample_times_seconds",
                ..
            })
        ));
    }

    #[test]
    fn maxwell_delay_history_rejects_ragged_nonfinite_and_excessive_work() {
        let mut ragged = maxwell_delay_history_fixture();
        ragged.current_transverse[1].pop();
        assert!(matches!(
            integrate_maxwell_delay_history(&ragged, 2.0, 3.0),
            Err(ContractError::InvalidControlTimingField {
                field: "current_transverse",
                ..
            })
        ));
        let mut nonfinite = maxwell_delay_history_fixture();
        nonfinite.gradient_electric_transverse[1][0] = f64::NAN;
        assert!(matches!(
            integrate_maxwell_delay_history(&nonfinite, 2.0, 3.0),
            Err(ContractError::InvalidControlTimingField {
                field: "gradient_electric_transverse",
                ..
            })
        ));
        let wide = vec![0.0; MAX_DELAY_HISTORY_WORK / 8 + 1];
        let excessive = MaxwellDelayHistorySamples {
            sample_times_seconds: vec![0.0, 1.0],
            current_longitudinal: vec![wide.clone(), wide],
            current_transverse: Vec::new(),
            gradient_electric_transverse: Vec::new(),
            gradient_magnetic_transverse: Vec::new(),
        };
        assert!(matches!(
            integrate_maxwell_delay_history(&excessive, 0.0, 1.0),
            Err(ContractError::InvalidControlTimingField {
                field: "delay_history",
                ..
            })
        ));
    }

    #[test]
    fn maxwell_delay_history_means_feed_the_cancellation_predictor() {
        let averages =
            integrate_maxwell_delay_history(&maxwell_delay_history_fixture(), 2.0, 3.0).unwrap();
        let observation = MaxwellDelayCancellationObservation {
            delayed_electric_longitudinal: vec![0.6, -0.2],
            delayed_electric_transverse: vec![1.2, 0.4],
            delayed_magnetic_transverse: vec![-0.4, 0.8],
            interval_mean_current_longitudinal: averages.current_longitudinal,
            interval_mean_current_transverse: averages.current_transverse,
            interval_mean_gradient_electric_transverse: averages.gradient_electric_transverse,
            interval_mean_gradient_magnetic_transverse: averages.gradient_magnetic_transverse,
        };
        let command = predict_maxwell_delay_cancellation(&observation, 1.0).unwrap();
        assert!((command.external_electric_longitudinal[0] - 0.9).abs() < 1.0e-12);
        assert!((command.external_electric_longitudinal[1] - 3.2).abs() < 1.0e-12);
        assert!((command.external_electric_transverse[0] - -0.7).abs() < 1.0e-12);
        assert!((command.external_electric_transverse[1] - 2.1).abs() < 1.0e-12);
        assert!((command.external_magnetic_transverse[0] - 1.4).abs() < 1.0e-12);
        assert!((command.external_magnetic_transverse[1] - -1.8).abs() < 1.0e-12);
    }

    #[test]
    fn maxwell_delay_predictor_matches_manufactured_interval_integral_identity() {
        let command =
            predict_maxwell_delay_cancellation(&maxwell_delay_observation_fixture(), 0.5).unwrap();
        assert!((command.external_electric_longitudinal[0] - -0.2).abs() < 1.0e-12);
        assert!((command.external_electric_transverse[0] - -1.0).abs() < 1.0e-12);
        assert!((command.external_magnetic_transverse[0] - -0.1).abs() < 1.0e-12);
        assert!((command.external_electric_longitudinal[1] - 0.0).abs() < 1.0e-12);
        assert!((command.external_electric_transverse[1] - -0.5).abs() < 1.0e-12);
        assert!((command.external_magnetic_transverse[1] - -0.7).abs() < 1.0e-12);
    }

    #[test]
    fn maxwell_delay_predictor_reduces_to_instantaneous_field_cancellation_at_zero_delay() {
        let observation = maxwell_delay_observation_fixture();
        let command = predict_maxwell_delay_cancellation(&observation, 0.0).unwrap();
        assert_eq!(
            command.external_electric_longitudinal,
            observation
                .delayed_electric_longitudinal
                .iter()
                .map(|value| -*value)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            command.external_electric_transverse,
            observation
                .delayed_electric_transverse
                .iter()
                .map(|value| -*value)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            command.external_magnetic_transverse,
            observation
                .delayed_magnetic_transverse
                .iter()
                .map(|value| -*value)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn maxwell_delay_predictor_rejects_invalid_delay_and_spatial_shapes() {
        let observation = maxwell_delay_observation_fixture();
        assert!(matches!(
            predict_maxwell_delay_cancellation(&observation, f64::NAN),
            Err(ContractError::InvalidControlTimingField {
                field: "delay_seconds",
                ..
            })
        ));
        assert!(matches!(
            predict_maxwell_delay_cancellation(&observation, -0.1),
            Err(ContractError::InvalidControlTimingField {
                field: "delay_seconds",
                ..
            })
        ));
        let mut mismatched = observation.clone();
        mismatched.delayed_electric_transverse.pop();
        assert!(matches!(
            predict_maxwell_delay_cancellation(&mismatched, 0.5),
            Err(ContractError::InvalidControlTimingField {
                field: "field_observations",
                ..
            })
        ));
        let mut empty = observation;
        empty.delayed_electric_longitudinal.clear();
        empty.delayed_electric_transverse.clear();
        empty.delayed_magnetic_transverse.clear();
        empty.interval_mean_current_longitudinal.clear();
        empty.interval_mean_current_transverse.clear();
        empty.interval_mean_gradient_electric_transverse.clear();
        empty.interval_mean_gradient_magnetic_transverse.clear();
        assert!(matches!(
            predict_maxwell_delay_cancellation(&empty, 0.5),
            Err(ContractError::InvalidControlTimingField {
                field: "field_observations",
                ..
            })
        ));
    }

    #[test]
    fn maxwell_delay_predictor_rejects_non_finite_measurements_and_overflow() {
        let mut observation = maxwell_delay_observation_fixture();
        observation.interval_mean_current_longitudinal[0] = f64::INFINITY;
        assert!(matches!(
            predict_maxwell_delay_cancellation(&observation, 0.5),
            Err(ContractError::InvalidControlTimingField {
                field: "field_observations",
                ..
            })
        ));
        let mut observation = maxwell_delay_observation_fixture();
        observation.interval_mean_current_longitudinal[0] = f64::MAX;
        assert!(matches!(
            predict_maxwell_delay_cancellation(&observation, f64::MAX),
            Err(ContractError::InvalidControlTimingField {
                field: "delay_compensation",
                ..
            })
        ));
    }

    #[test]
    fn coupled_two_mode_fixture_matches_all_three_known_answers() {
        let fixture: serde_json::Value =
            serde_json::from_str(COUPLED_TWO_MODE_AUTHORITY_FIXTURE).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let matrix = case["transition_matrix"].as_array().unwrap();
            let diagnostics = case["diagnostic_gain"].as_array().unwrap();
            let contract = CoupledTwoModeAuthorityContract {
                transition_matrix: [
                    matrix[0].as_f64().unwrap(),
                    matrix[1].as_f64().unwrap(),
                    matrix[2].as_f64().unwrap(),
                    matrix[3].as_f64().unwrap(),
                ],
                horizon_steps: case["horizon_steps"].as_u64().unwrap() as u32,
                diagnostic_gain: [
                    [
                        diagnostics[0][0].as_f64().unwrap(),
                        diagnostics[0][1].as_f64().unwrap(),
                    ],
                    [
                        diagnostics[1][0].as_f64().unwrap(),
                        diagnostics[1][1].as_f64().unwrap(),
                    ],
                ],
                diagnostic_noise_variance: [
                    case["diagnostic_noise_variance"][0].as_f64().unwrap(),
                    case["diagnostic_noise_variance"][1].as_f64().unwrap(),
                ],
                diagnostic_sampling_stride_steps: case["diagnostic_sampling_stride_steps"]
                    .as_u64()
                    .unwrap() as u32,
                diagnostic_delivery_delay_steps: case["diagnostic_delivery_delay_steps"]
                    .as_u64()
                    .unwrap() as u32,
                actuator_gain: [
                    case["actuator_gain"][0].as_f64().unwrap(),
                    case["actuator_gain"][1].as_f64().unwrap(),
                ],
                actuator_update_stride_steps: case["actuator_update_stride_steps"].as_u64().unwrap()
                    as u32,
                actuator_delivery_delay_steps: case["actuator_delivery_delay_steps"]
                    .as_u64()
                    .unwrap() as u32,
                minimum_observability_eigenvalue: case["minimum_observability_eigenvalue"]
                    .as_f64()
                    .unwrap(),
                minimum_controllability_eigenvalue: case["minimum_controllability_eigenvalue"]
                    .as_f64()
                    .unwrap(),
            };
            let report = evaluate_coupled_two_mode_authority(contract).unwrap();
            let expected = &case["expected"];
            assert_eq!(
                report.observation_sample_count,
                expected["observation_sample_count"].as_u64().unwrap() as u32
            );
            assert_eq!(
                report.actuator_command_count,
                expected["actuator_command_count"].as_u64().unwrap() as u32
            );
            assert_eq!(report.observable, expected["observable"].as_bool().unwrap());
            assert_eq!(
                report.controllable,
                expected["controllable"].as_bool().unwrap()
            );
            assert_eq!(
                report.authority_qualified,
                expected["authority_qualified"].as_bool().unwrap()
            );
        }
    }

    #[test]
    fn coupled_two_mode_authority_accepts_distinct_unstable_modes_with_independent_sensors() {
        let report = evaluate_coupled_two_mode_authority(coupled_two_mode_contract(
            [1.2, 0.0, 0.0, 1.1],
            [[1.0, 0.0], [0.0, 1.0]],
            [1.0, 1.0],
        ))
        .unwrap();
        assert!((report.minimum_unstable_eigenvalue_modulus - 1.1).abs() < 1.0e-12);
        assert!((report.spectral_radius - 1.2).abs() < 1.0e-12);
        assert_eq!(report.observation_sample_count, 12);
        assert_eq!(report.actuator_command_count, 12);
        assert!(report.observable);
        assert!(report.controllable);
        assert!(report.authority_qualified);
        assert!(report.observability_min_eigenvalue > 1.0e-6);
        assert!(report.controllability_min_eigenvalue > 1.0e-6);
        assert!(report.observability_condition_number.is_some());
        assert!(report.controllability_condition_number.is_some());
    }

    #[test]
    fn coupled_two_mode_authority_rejects_shared_sensor_on_repeated_unstable_eigenspace() {
        let report = evaluate_coupled_two_mode_authority(coupled_two_mode_contract(
            [1.2, 0.0, 0.0, 1.2],
            [[1.0, 1.0], [2.0, 2.0]],
            [1.0, 1.0],
        ))
        .unwrap();
        assert!(!report.observable);
        assert!(!report.authority_qualified);
        assert_eq!(report.observability_min_eigenvalue, 0.0);
        assert_eq!(report.observability_condition_number, None);
    }

    #[test]
    fn coupled_two_mode_authority_rejects_rank_one_actuator_on_repeated_unstable_eigenspace() {
        let report = evaluate_coupled_two_mode_authority(coupled_two_mode_contract(
            [1.2, 0.0, 0.0, 1.2],
            [[1.0, 0.0], [0.0, 1.0]],
            [1.0, 1.0],
        ))
        .unwrap();
        assert!(report.observable);
        assert!(!report.controllable);
        assert!(!report.authority_qualified);
        assert_eq!(report.controllability_min_eigenvalue, 0.0);
        assert_eq!(report.controllability_condition_number, None);
    }

    #[test]
    fn coupled_two_mode_authority_advances_by_the_declared_sensor_stride() {
        let mut contract =
            coupled_two_mode_contract([1.2, 0.0, 0.0, 1.1], [[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0]);
        contract.horizon_steps = 5;
        contract.diagnostic_sampling_stride_steps = 2;
        contract.actuator_update_stride_steps = 2;
        let report = evaluate_coupled_two_mode_authority(contract).unwrap();
        assert_eq!(report.observation_sample_count, 3);
        assert_eq!(report.actuator_command_count, 3);
        let expected_x = 1.0 + 1.2_f64.powi(4) + 1.2_f64.powi(8);
        let expected_y = 1.0 + 1.1_f64.powi(4) + 1.1_f64.powi(8);
        assert!((report.observability_gramian[0] - expected_x).abs() < 1.0e-10);
        assert!(report.observability_gramian[1].abs() < 1.0e-12);
        assert!((report.observability_gramian[2] - expected_y).abs() < 1.0e-10);
    }

    #[test]
    fn coupled_two_mode_authority_applies_sensor_and_actuator_delivery_delays() {
        let mut contract =
            coupled_two_mode_contract([1.2, 0.0, 0.0, 1.1], [[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0]);
        contract.diagnostic_delivery_delay_steps = 4;
        contract.actuator_delivery_delay_steps = 5;
        let report = evaluate_coupled_two_mode_authority(contract).unwrap();
        assert_eq!(report.observation_sample_count, 8);
        assert_eq!(report.actuator_command_count, 7);
    }

    #[test]
    fn coupled_two_mode_authority_rejects_late_or_absent_channel_updates() {
        let mut contract =
            coupled_two_mode_contract([1.2, 0.0, 0.0, 1.1], [[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0]);
        contract.diagnostic_delivery_delay_steps = contract.horizon_steps;
        assert!(matches!(
            evaluate_coupled_two_mode_authority(contract),
            Err(ContractError::InvalidCoupledModeAuthority(_))
        ));
        contract.diagnostic_delivery_delay_steps = 0;
        contract.actuator_delivery_delay_steps = contract.horizon_steps;
        assert!(matches!(
            evaluate_coupled_two_mode_authority(contract),
            Err(ContractError::InvalidCoupledModeAuthority(_))
        ));
    }

    #[test]
    fn coupled_two_mode_authority_rejects_invalid_and_non_unstable_contracts() {
        let mut contract =
            coupled_two_mode_contract([1.2, 0.0, 0.0, 1.1], [[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0]);
        contract.diagnostic_noise_variance = [0.0, 1.0];
        assert!(matches!(
            evaluate_coupled_two_mode_authority(contract),
            Err(ContractError::InvalidCoupledModeAuthority(_))
        ));
        contract.diagnostic_noise_variance = [1.0, 1.0];
        contract.transition_matrix = [1.2, 0.0, 0.0, 0.95];
        assert!(matches!(
            evaluate_coupled_two_mode_authority(contract),
            Err(ContractError::InvalidCoupledModeAuthority(_))
        ));
    }

    #[test]
    fn response_weighted_refinement_retains_low_energy_future_critical_mode() {
        let mut matrix = ModeResponseMatrix::new(vec![1, 2], vec![10]).unwrap();
        matrix.set(1, 10, ComplexResponse::new(0.01, 0.0).unwrap()).unwrap();
        matrix.set(2, 10, ComplexResponse::new(100.0, 0.0).unwrap()).unwrap();
        let plan = plan_response_weighted_refinement(
            &matrix,
            &[
                (1, ComplexResponse::new(1.0, 0.0).unwrap()),
                (2, ComplexResponse::new(1.0e-4, 0.0).unwrap()),
            ],
            &[(1, 100), (2, 100)],
            &[1.0],
            0.02,
        )
        .unwrap();
        let mode_one = plan.decisions.iter().find(|item| item.input_mode == 1).unwrap();
        let mode_two = plan.decisions.iter().find(|item| item.input_mode == 2).unwrap();
        assert!(mode_two.spectral_energy < mode_one.spectral_energy);
        assert_eq!(mode_one.disposition, ModeRefinementDisposition::DiscardedWithinBudget);
        assert_eq!(mode_two.disposition, ModeRefinementDisposition::RetainedBudgetExceeded);
        assert!((plan.accumulated_discarded_response_bound - 0.01).abs() < 1.0e-12);
        assert_eq!(plan.discarded_state_bytes, 100);
    }

    #[test]
    fn response_weighted_refinement_retains_modes_with_missing_response_entries() {
        let mut matrix = ModeResponseMatrix::new(vec![1, 2], vec![10]).unwrap();
        matrix.set(1, 10, ComplexResponse::new(0.1, 0.0).unwrap()).unwrap();
        let plan = plan_response_weighted_refinement(
            &matrix,
            &[
                (1, ComplexResponse::new(1.0, 0.0).unwrap()),
                (2, ComplexResponse::new(1.0e-8, 0.0).unwrap()),
            ],
            &[(1, 64), (2, 64)],
            &[1.0],
            1.0,
        )
        .unwrap();
        let missing = plan.decisions.iter().find(|item| item.input_mode == 2).unwrap();
        assert_eq!(missing.weighted_response_gain, None);
        assert_eq!(missing.response_loss_bound, None);
        assert_eq!(missing.disposition, ModeRefinementDisposition::RetainedUnknownResponse);
    }

    #[test]
    fn response_weighted_refinement_rejects_duplicate_or_missing_mode_metadata() {
        let matrix = ModeResponseMatrix::new(vec![1, 2], vec![10]).unwrap();
        let result = plan_response_weighted_refinement(
            &matrix,
            &[
                (1, ComplexResponse::new(1.0, 0.0).unwrap()),
                (1, ComplexResponse::new(1.0, 0.0).unwrap()),
            ],
            &[(1, 8), (2, 8)],
            &[1.0],
            1.0,
        );
        assert_eq!(result, Err(ContractError::InvalidResponseRefinementInput));
    }

    #[test]
    fn response_weighted_refinement_rejects_invalid_weights_and_budget() {
        let matrix = ModeResponseMatrix::new(vec![1], vec![10]).unwrap();
        let coefficients = [(1, ComplexResponse::new(1.0, 0.0).unwrap())];
        let bytes = [(1, 8)];
        assert_eq!(
            plan_response_weighted_refinement(&matrix, &coefficients, &bytes, &[-1.0], 1.0),
            Err(ContractError::InvalidResponseRefinementInput)
        );
        assert_eq!(
            plan_response_weighted_refinement(&matrix, &coefficients, &bytes, &[1.0], f64::NAN),
            Err(ContractError::InvalidResponseRefinementInput)
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
