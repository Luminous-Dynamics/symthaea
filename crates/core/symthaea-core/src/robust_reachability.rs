// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Analytic robust terminal-set reachability for MANIFOLD-006C.
//!
//! This module deliberately uses the smallest complete disturbed control fixture:
//!
//! `x_dot = u + d`, `u in [u_min,u_max]`, `d in [d_min,d_max]`.
//!
//! The V1 information pattern is constant open-loop control chosen before the
//! disturbance realization. Robust reachability means one admissible control keeps
//! **every** declared disturbance realization inside a closed terminal target set.
//! It is intentionally distinct from nominal or existential reachability.

use blake3::Hasher;
use thiserror::Error;

use crate::kinodynamic_reachability::{
    BoundedSingleIntegrator1D, DynamicsTimeDomain, PlantTimeProfile,
};
use crate::reachability::UnknownReason;

/// Maximum finite-arithmetic control-boundary guard accepted by the V1 fixture.
pub const MAX_ROBUST_CONTROL_TOLERANCE: f64 = 1.0e-6;
/// Reference finite-arithmetic guard for robust-control interval classification.
pub const DEFAULT_ROBUST_CONTROL_TOLERANCE: f64 = 1.0e-12;

/// Explicit quantifier semantics for the disturbance set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisturbanceQuantifier1D {
    /// Robust claims quantify over every measurable disturbance in the interval.
    UniversalAdversarial,
}

/// Exact bounded scalar disturbance profile in plant/model time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundedDisturbance1D {
    minimum: f64,
    maximum: f64,
    identity: [u8; 32],
}

impl BoundedDisturbance1D {
    /// Construct a finite closed disturbance interval `[minimum, maximum]`.
    pub fn new(minimum: f64, maximum: f64) -> Result<Self, RobustReachabilityError> {
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return Err(RobustReachabilityError::InvalidDisturbance {
                reason: format!(
                    "disturbance interval must be finite with minimum <= maximum, got [{minimum}, {maximum}]"
                ),
            });
        }
        let minimum = canonical_zero(minimum);
        let maximum = canonical_zero(maximum);
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-bounded-disturbance-1d-v1\0");
        hasher.update(b"universal-adversarial\0plant-model-seconds\0");
        hasher.update(&minimum.to_bits().to_le_bytes());
        hasher.update(&maximum.to_bits().to_le_bytes());
        Ok(Self {
            minimum,
            maximum,
            identity: *hasher.finalize().as_bytes(),
        })
    }

    /// Inclusive minimum disturbance.
    pub fn minimum(&self) -> f64 {
        self.minimum
    }

    /// Inclusive maximum disturbance.
    pub fn maximum(&self) -> f64 {
        self.maximum
    }

    /// Universal/adversarial quantifier used by robust checks.
    pub fn quantifier(&self) -> DisturbanceQuantifier1D {
        DisturbanceQuantifier1D::UniversalAdversarial
    }

    /// Disturbance evolves in the same explicit plant/model time domain.
    pub fn time_domain(&self) -> DynamicsTimeDomain {
        DynamicsTimeDomain::PlantModelSeconds
    }

    /// Exact profile identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Closed scalar terminal target or reachable interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerminalInterval1D {
    lower: f64,
    upper: f64,
    identity: [u8; 32],
}

impl TerminalInterval1D {
    /// Construct a finite closed interval `[lower, upper]`.
    pub fn new(lower: f64, upper: f64) -> Result<Self, RobustReachabilityError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(RobustReachabilityError::InvalidTarget {
                reason: format!(
                    "terminal interval must be finite with lower <= upper, got [{lower}, {upper}]"
                ),
            });
        }
        let lower = canonical_zero(lower);
        let upper = canonical_zero(upper);
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-terminal-interval-1d-v1\0");
        hasher.update(&lower.to_bits().to_le_bytes());
        hasher.update(&upper.to_bits().to_le_bytes());
        Ok(Self {
            lower,
            upper,
            identity: *hasher.finalize().as_bytes(),
        })
    }

    /// Inclusive lower endpoint.
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// Inclusive upper endpoint.
    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// Whether a finite value is inside the interval.
    pub fn contains(&self, value: f64) -> bool {
        value.is_finite() && value >= self.lower && value <= self.upper
    }

    /// Whether another closed interval is fully contained in this interval.
    pub fn contains_interval(&self, other: TerminalInterval1D) -> bool {
        other.lower >= self.lower && other.upper <= self.upper
    }

    /// Closed intersection when non-empty.
    pub fn intersection(&self, other: TerminalInterval1D) -> Option<TerminalInterval1D> {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);
        if lower <= upper {
            TerminalInterval1D::new(lower, upper).ok()
        } else {
            None
        }
    }

    /// Exact interval identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Explicit finite-arithmetic policy for robust-control classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustReachabilityPolicy1D {
    absolute_control_tolerance: f64,
    identity: [u8; 32],
}

impl RobustReachabilityPolicy1D {
    /// Construct a bounded finite absolute control-space ambiguity guard.
    pub fn new(absolute_control_tolerance: f64) -> Result<Self, RobustReachabilityError> {
        if !absolute_control_tolerance.is_finite()
            || absolute_control_tolerance < 0.0
            || absolute_control_tolerance > MAX_ROBUST_CONTROL_TOLERANCE
        {
            return Err(RobustReachabilityError::InvalidPolicy {
                reason: format!(
                    "robust control tolerance must be finite and in [0, {MAX_ROBUST_CONTROL_TOLERANCE}], got {absolute_control_tolerance}"
                ),
            });
        }
        let absolute_control_tolerance = canonical_zero(absolute_control_tolerance);
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-robust-reachability-policy-1d-v1\0");
        hasher.update(&absolute_control_tolerance.to_bits().to_le_bytes());
        Ok(Self {
            absolute_control_tolerance,
            identity: *hasher.finalize().as_bytes(),
        })
    }

    /// Named reference policy for the analytic V1 fixture.
    pub fn reference_v1() -> Self {
        Self::new(DEFAULT_ROBUST_CONTROL_TOLERANCE)
            .expect("reference robust-reachability policy is valid")
    }

    /// Absolute numerical ambiguity guard in control units.
    pub fn absolute_control_tolerance(&self) -> f64 {
        self.absolute_control_tolerance
    }

    /// Exact policy identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Exact robust terminal-set query under one disturbed single-integrator model.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustSingleIntegratorQuery1D {
    model_identity: [u8; 32],
    initial_state: f64,
    target: TerminalInterval1D,
    disturbance: BoundedDisturbance1D,
    plant_time: PlantTimeProfile,
    policy: RobustReachabilityPolicy1D,
    identity: [u8; 32],
}

impl RobustSingleIntegratorQuery1D {
    /// Construct one exact robust terminal-set query.
    pub fn new(
        model: &BoundedSingleIntegrator1D,
        initial_state: f64,
        target: TerminalInterval1D,
        disturbance: BoundedDisturbance1D,
        plant_time: PlantTimeProfile,
        policy: RobustReachabilityPolicy1D,
    ) -> Result<Self, RobustReachabilityError> {
        if !initial_state.is_finite() {
            return Err(RobustReachabilityError::InvalidQuery {
                reason: "robust initial state must be finite".to_string(),
            });
        }
        let initial_state = canonical_zero(initial_state);
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-robust-single-integrator-query-1d-v1\0");
        hasher.update(&model.identity());
        hasher.update(&initial_state.to_bits().to_le_bytes());
        hasher.update(&target.identity());
        hasher.update(&disturbance.identity());
        hasher.update(&plant_time.identity());
        hasher.update(&policy.identity());
        Ok(Self {
            model_identity: model.identity(),
            initial_state,
            target,
            disturbance,
            plant_time,
            policy,
            identity: *hasher.finalize().as_bytes(),
        })
    }

    /// Exact model identity.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }

    /// Initial scalar state.
    pub fn initial_state(&self) -> f64 {
        self.initial_state
    }

    /// Closed terminal target interval.
    pub fn target(&self) -> TerminalInterval1D {
        self.target
    }

    /// Exact universal/adversarial disturbance profile.
    pub fn disturbance(&self) -> BoundedDisturbance1D {
        self.disturbance
    }

    /// Exact plant/model-time horizon.
    pub fn plant_time(&self) -> PlantTimeProfile {
        self.plant_time
    }

    /// Numerical classification policy.
    pub fn policy(&self) -> RobustReachabilityPolicy1D {
        self.policy
    }

    /// Exact query identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Raw and actuator-clipped robust-control bounds for one query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustControlRequirement1D {
    required_lower: f64,
    required_upper: f64,
    admissible_lower: f64,
    admissible_upper: f64,
}

impl RobustControlRequirement1D {
    /// Minimum constant control required by terminal lower containment.
    pub fn required_lower(&self) -> f64 {
        self.required_lower
    }

    /// Maximum constant control allowed by terminal upper containment.
    pub fn required_upper(&self) -> f64 {
        self.required_upper
    }

    /// Required lower bound clipped against actuator minimum.
    pub fn admissible_lower(&self) -> f64 {
        self.admissible_lower
    }

    /// Required upper bound clipped against actuator maximum.
    pub fn admissible_upper(&self) -> f64 {
        self.admissible_upper
    }

    /// Whether the unconstrained robust-control requirement is non-empty.
    pub fn raw_nonempty(&self) -> bool {
        self.required_lower <= self.required_upper
    }

    /// Whether actuator bounds leave at least one robust constant control.
    pub fn admissible_nonempty(&self) -> bool {
        self.admissible_lower <= self.admissible_upper
    }
}

/// Constructive robust witness: one open-loop control and its full disturbance envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustControlWitness1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    control: f64,
    terminal_envelope: TerminalInterval1D,
    identity: [u8; 32],
}

impl RobustControlWitness1D {
    /// Constant open-loop control selected before disturbance realization.
    pub fn control(&self) -> f64 {
        self.control
    }

    /// Exact terminal envelope under every declared disturbance realization.
    pub fn terminal_envelope(&self) -> TerminalInterval1D {
        self.terminal_envelope
    }

    /// Exact witness identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Independent verification receipt for a robust containment witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobustWitnessVerificationReceipt1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    witness_identity: [u8; 32],
    verifier_identity: [u8; 32],
}

impl RobustWitnessVerificationReceipt1D {
    /// Exact model identity.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }

    /// Exact query identity.
    pub fn query_identity(&self) -> [u8; 32] {
        self.query_identity
    }

    /// Verified witness identity.
    pub fn witness_identity(&self) -> [u8; 32] {
        self.witness_identity
    }

    /// Exact independent verifier identity.
    pub fn verifier_identity(&self) -> [u8; 32] {
        self.verifier_identity
    }
}

/// Analytic reason no V1 constant open-loop control can robustly contain the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotRobustReason1D {
    /// Target interval is strictly narrower than the universal disturbance envelope.
    TargetTooNarrow,
    /// Every robustly sufficient control lies below the actuator minimum.
    RequiresControlBelowMinimum,
    /// Every robustly sufficient control lies above the actuator maximum.
    RequiresControlAboveMaximum,
}

/// Independently verifiable analytic certificate that V1 open-loop robust containment fails.
#[derive(Clone, Debug, PartialEq)]
pub struct NotRobustCertificate1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    requirement: RobustControlRequirement1D,
    reason: NotRobustReason1D,
    identity: [u8; 32],
}

impl NotRobustCertificate1D {
    /// Exact robust-control requirement used by the separator.
    pub fn requirement(&self) -> RobustControlRequirement1D {
        self.requirement
    }

    /// Specific analytic separation reason.
    pub fn reason(&self) -> NotRobustReason1D {
        self.reason
    }

    /// Exact certificate identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Independent receipt for a not-robust certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotRobustVerificationReceipt1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    certificate_identity: [u8; 32],
    verifier_identity: [u8; 32],
}

impl NotRobustVerificationReceipt1D {
    /// Verified certificate identity.
    pub fn certificate_identity(&self) -> [u8; 32] {
        self.certificate_identity
    }

    /// Exact verifier identity.
    pub fn verifier_identity(&self) -> [u8; 32] {
        self.verifier_identity
    }
}

/// Proof-strength result for exact V1 open-loop robust terminal-set reachability.
#[derive(Clone, Debug, PartialEq)]
pub enum RobustReachabilityResult1D {
    /// One admissible constant control contains every disturbance realization in target.
    RobustFeasible {
        /// Constructive robust witness.
        witness: RobustControlWitness1D,
        /// Independent exact containment verification.
        verification: RobustWitnessVerificationReceipt1D,
    },
    /// Exact analytic separation proves no V1 constant open-loop control is robust.
    CertifiedNotRobust {
        /// Analytic negative certificate.
        certificate: NotRobustCertificate1D,
        /// Independent certificate verification.
        verification: NotRobustVerificationReceipt1D,
    },
    /// Finite arithmetic is too close to a robust-control boundary for a strong claim.
    Unknown {
        /// Shared proof-strength unknown reason.
        reason: UnknownReason,
        /// Bounded diagnostic detail.
        detail: String,
    },
}

/// Fail-closed errors in the robust reference fixture.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RobustReachabilityError {
    /// Disturbance profile is malformed.
    #[error("invalid disturbance profile: {reason}")]
    InvalidDisturbance { reason: String },
    /// Terminal target interval is malformed.
    #[error("invalid terminal target: {reason}")]
    InvalidTarget { reason: String },
    /// Robust numerical policy is malformed.
    #[error("invalid robust reachability policy: {reason}")]
    InvalidPolicy { reason: String },
    /// Robust query is malformed.
    #[error("invalid robust reachability query: {reason}")]
    InvalidQuery { reason: String },
    /// Exact model/query identities disagree.
    #[error("robust reachability identity mismatch: {reason}")]
    IdentityMismatch { reason: String },
    /// Finite arithmetic escaped the declared numerical model.
    #[error("robust reachability numerical failure: {reason}")]
    Numerical { reason: String },
    /// Positive witness failed independent containment verification.
    #[error("invalid robust witness: {reason}")]
    InvalidWitness { reason: String },
    /// Negative certificate failed independent verification.
    #[error("invalid not-robust certificate: {reason}")]
    InvalidCertificate { reason: String },
}

/// Exact terminal interval under one fixed admissible control and every declared disturbance.
pub fn fixed_control_terminal_envelope(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
    control: f64,
) -> Result<TerminalInterval1D, RobustReachabilityError> {
    require_model_query_identity(model, query)?;
    if !model.controls().contains(control) {
        return Err(RobustReachabilityError::InvalidWitness {
            reason: format!(
                "fixed control {control} violates exact bounds [{}, {}]",
                model.controls().minimum(),
                model.controls().maximum()
            ),
        });
    }
    let horizon = query.plant_time().horizon_seconds();
    let lower_rate = checked_add(control, query.disturbance().minimum())?;
    let upper_rate = checked_add(control, query.disturbance().maximum())?;
    let lower = checked_affine_terminal(query.initial_state(), lower_rate, horizon)?;
    let upper = checked_affine_terminal(query.initial_state(), upper_rate, horizon)?;
    TerminalInterval1D::new(lower, upper)
}

/// Exact terminal interval reachable for some admissible constant control and some disturbance.
pub fn existential_terminal_interval(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
) -> Result<TerminalInterval1D, RobustReachabilityError> {
    require_model_query_identity(model, query)?;
    let horizon = query.plant_time().horizon_seconds();
    let lower_rate = checked_add(model.controls().minimum(), query.disturbance().minimum())?;
    let upper_rate = checked_add(model.controls().maximum(), query.disturbance().maximum())?;
    let lower = checked_affine_terminal(query.initial_state(), lower_rate, horizon)?;
    let upper = checked_affine_terminal(query.initial_state(), upper_rate, horizon)?;
    TerminalInterval1D::new(lower, upper)
}

/// Exact control inequalities required for universal terminal containment.
pub fn robust_control_requirement(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
) -> Result<RobustControlRequirement1D, RobustReachabilityError> {
    require_model_query_identity(model, query)?;
    let horizon = query.plant_time().horizon_seconds();
    let target = query.target();
    let lower_delta = checked_sub(target.lower(), query.initial_state())?;
    let upper_delta = checked_sub(target.upper(), query.initial_state())?;
    let required_lower = checked_sub(
        checked_div(lower_delta, horizon)?,
        query.disturbance().minimum(),
    )?;
    let required_upper = checked_sub(
        checked_div(upper_delta, horizon)?,
        query.disturbance().maximum(),
    )?;
    Ok(RobustControlRequirement1D {
        required_lower,
        required_upper,
        admissible_lower: required_lower.max(model.controls().minimum()),
        admissible_upper: required_upper.min(model.controls().maximum()),
    })
}

/// Complete analytic V1 robust reachability solver for constant open-loop control.
pub fn solve_robust_single_integrator_analytic(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
) -> Result<RobustReachabilityResult1D, RobustReachabilityError> {
    let requirement = robust_control_requirement(model, query)?;
    let tolerance = query.policy().absolute_control_tolerance();

    if requirement.required_lower() > requirement.required_upper() {
        let gap = checked_sub(requirement.required_lower(), requirement.required_upper())?;
        if gap <= tolerance {
            return Ok(RobustReachabilityResult1D::Unknown {
                reason: UnknownReason::NumericalFailure,
                detail: format!(
                    "robust control requirement is empty only inside numerical guard: gap={gap}, tolerance={tolerance}"
                ),
            });
        }
        return certified_not_robust(
            model,
            query,
            requirement,
            NotRobustReason1D::TargetTooNarrow,
        );
    }

    if requirement.admissible_nonempty() {
        let control = checked_add(
            checked_mul(0.5, requirement.admissible_lower())?,
            checked_mul(0.5, requirement.admissible_upper())?,
        )?;
        let envelope = fixed_control_terminal_envelope(model, query, control)?;
        if !query.target().contains_interval(envelope) {
            return Ok(RobustReachabilityResult1D::Unknown {
                reason: UnknownReason::NumericalFailure,
                detail: "robust-control interval is non-empty but finite arithmetic reconstructed an envelope outside the exact target"
                    .to_string(),
            });
        }
        let witness = build_robust_witness(model, query, control, envelope);
        let verification = verify_robust_witness(model, query, &witness)?;
        return Ok(RobustReachabilityResult1D::RobustFeasible {
            witness,
            verification,
        });
    }

    if requirement.required_lower() > model.controls().maximum() {
        let gap = checked_sub(requirement.required_lower(), model.controls().maximum())?;
        if gap <= tolerance {
            return Ok(RobustReachabilityResult1D::Unknown {
                reason: UnknownReason::NumericalFailure,
                detail: format!(
                    "robust control lies above actuator maximum only inside numerical guard: gap={gap}, tolerance={tolerance}"
                ),
            });
        }
        return certified_not_robust(
            model,
            query,
            requirement,
            NotRobustReason1D::RequiresControlAboveMaximum,
        );
    }

    if requirement.required_upper() < model.controls().minimum() {
        let gap = checked_sub(model.controls().minimum(), requirement.required_upper())?;
        if gap <= tolerance {
            return Ok(RobustReachabilityResult1D::Unknown {
                reason: UnknownReason::NumericalFailure,
                detail: format!(
                    "robust control lies below actuator minimum only inside numerical guard: gap={gap}, tolerance={tolerance}"
                ),
            });
        }
        return certified_not_robust(
            model,
            query,
            requirement,
            NotRobustReason1D::RequiresControlBelowMinimum,
        );
    }

    Ok(RobustReachabilityResult1D::Unknown {
        reason: UnknownReason::NumericalFailure,
        detail: "robust-control intersection is numerically indeterminate".to_string(),
    })
}

/// Independently verify a constructive robust containment witness.
pub fn verify_robust_witness(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
    witness: &RobustControlWitness1D,
) -> Result<RobustWitnessVerificationReceipt1D, RobustReachabilityError> {
    require_model_query_identity(model, query)?;
    if witness.model_identity != model.identity() || witness.query_identity != query.identity() {
        return Err(RobustReachabilityError::InvalidWitness {
            reason: "robust witness subject identities do not match".to_string(),
        });
    }
    if !model.controls().contains(witness.control()) {
        return Err(RobustReachabilityError::InvalidWitness {
            reason: "robust witness control is outside exact actuator bounds".to_string(),
        });
    }
    let requirement = robust_control_requirement(model, query)?;
    if !requirement.raw_nonempty()
        || witness.control() < requirement.admissible_lower()
        || witness.control() > requirement.admissible_upper()
    {
        return Err(RobustReachabilityError::InvalidWitness {
            reason: "robust witness control does not satisfy independently recomputed containment inequalities"
                .to_string(),
        });
    }
    let envelope = fixed_control_terminal_envelope(model, query, witness.control())?;
    if witness.terminal_envelope != envelope {
        return Err(RobustReachabilityError::InvalidWitness {
            reason: "robust witness terminal envelope does not match independent recomputation"
                .to_string(),
        });
    }
    if !query.target().contains_interval(envelope) {
        return Err(RobustReachabilityError::InvalidWitness {
            reason: "recomputed universal disturbance envelope is not contained in target"
                .to_string(),
        });
    }
    let expected = hash_robust_witness(model.identity(), query.identity(), witness.control(), envelope);
    if witness.identity != expected {
        return Err(RobustReachabilityError::InvalidWitness {
            reason: "robust witness identity does not match independent theorem inputs".to_string(),
        });
    }
    Ok(RobustWitnessVerificationReceipt1D {
        model_identity: model.identity(),
        query_identity: query.identity(),
        witness_identity: witness.identity(),
        verifier_identity: robust_witness_verifier_identity(),
    })
}

/// Independently verify a certificate that V1 constant open-loop robust containment fails.
pub fn verify_not_robust_certificate(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
    certificate: &NotRobustCertificate1D,
) -> Result<NotRobustVerificationReceipt1D, RobustReachabilityError> {
    require_model_query_identity(model, query)?;
    if certificate.model_identity != model.identity() || certificate.query_identity != query.identity() {
        return Err(RobustReachabilityError::InvalidCertificate {
            reason: "not-robust certificate subject identities do not match".to_string(),
        });
    }
    let requirement = robust_control_requirement(model, query)?;
    if certificate.requirement != requirement {
        return Err(RobustReachabilityError::InvalidCertificate {
            reason: "not-robust certificate control requirement does not replay exactly".to_string(),
        });
    }
    let tolerance = query.policy().absolute_control_tolerance();
    let reason_holds = match certificate.reason {
        NotRobustReason1D::TargetTooNarrow => {
            requirement.required_lower()
                > checked_add(requirement.required_upper(), tolerance)?
        }
        NotRobustReason1D::RequiresControlBelowMinimum => {
            requirement.raw_nonempty()
                && requirement.required_upper()
                    < checked_sub(model.controls().minimum(), tolerance)?
        }
        NotRobustReason1D::RequiresControlAboveMaximum => {
            requirement.raw_nonempty()
                && requirement.required_lower()
                    > checked_add(model.controls().maximum(), tolerance)?
        }
    };
    if !reason_holds {
        return Err(RobustReachabilityError::InvalidCertificate {
            reason: "declared not-robust separator does not independently hold beyond the numerical guard"
                .to_string(),
        });
    }
    let expected = hash_not_robust_certificate(
        model.identity(),
        query.identity(),
        requirement,
        certificate.reason,
    );
    if certificate.identity != expected {
        return Err(RobustReachabilityError::InvalidCertificate {
            reason: "not-robust certificate identity does not match independent theorem inputs"
                .to_string(),
        });
    }
    Ok(NotRobustVerificationReceipt1D {
        model_identity: model.identity(),
        query_identity: query.identity(),
        certificate_identity: certificate.identity(),
        verifier_identity: not_robust_verifier_identity(),
    })
}

fn certified_not_robust(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
    requirement: RobustControlRequirement1D,
    reason: NotRobustReason1D,
) -> Result<RobustReachabilityResult1D, RobustReachabilityError> {
    let certificate = build_not_robust_certificate(model, query, requirement, reason);
    let verification = verify_not_robust_certificate(model, query, &certificate)?;
    Ok(RobustReachabilityResult1D::CertifiedNotRobust {
        certificate,
        verification,
    })
}

fn build_robust_witness(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
    control: f64,
    terminal_envelope: TerminalInterval1D,
) -> RobustControlWitness1D {
    RobustControlWitness1D {
        model_identity: model.identity(),
        query_identity: query.identity(),
        control,
        terminal_envelope,
        identity: hash_robust_witness(
            model.identity(),
            query.identity(),
            control,
            terminal_envelope,
        ),
    }
}

fn build_not_robust_certificate(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
    requirement: RobustControlRequirement1D,
    reason: NotRobustReason1D,
) -> NotRobustCertificate1D {
    NotRobustCertificate1D {
        model_identity: model.identity(),
        query_identity: query.identity(),
        requirement,
        reason,
        identity: hash_not_robust_certificate(
            model.identity(),
            query.identity(),
            requirement,
            reason,
        ),
    }
}

fn require_model_query_identity(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
) -> Result<(), RobustReachabilityError> {
    if query.model_identity() != model.identity() {
        return Err(RobustReachabilityError::IdentityMismatch {
            reason: "robust query was constructed for a different single-integrator model"
                .to_string(),
        });
    }
    Ok(())
}

fn hash_robust_witness(
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    control: f64,
    envelope: TerminalInterval1D,
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-robust-control-witness-1d-v1\0");
    hasher.update(&model_identity);
    hasher.update(&query_identity);
    hasher.update(&control.to_bits().to_le_bytes());
    hasher.update(&envelope.identity());
    *hasher.finalize().as_bytes()
}

fn hash_not_robust_certificate(
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    requirement: RobustControlRequirement1D,
    reason: NotRobustReason1D,
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-not-robust-certificate-1d-v1\0");
    hasher.update(&model_identity);
    hasher.update(&query_identity);
    for value in [
        requirement.required_lower(),
        requirement.required_upper(),
        requirement.admissible_lower(),
        requirement.admissible_upper(),
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.update(&[match reason {
        NotRobustReason1D::TargetTooNarrow => 0,
        NotRobustReason1D::RequiresControlBelowMinimum => 1,
        NotRobustReason1D::RequiresControlAboveMaximum => 2,
    }]);
    *hasher.finalize().as_bytes()
}

fn robust_witness_verifier_identity() -> [u8; 32] {
    *blake3::hash(b"symthaea-robust-witness-verifier-1d-v1\0").as_bytes()
}

fn not_robust_verifier_identity() -> [u8; 32] {
    *blake3::hash(b"symthaea-not-robust-verifier-1d-v1\0").as_bytes()
}

fn checked_affine_terminal(
    start: f64,
    rate: f64,
    duration: f64,
) -> Result<f64, RobustReachabilityError> {
    let value = rate.mul_add(duration, start);
    finite("affine terminal transition", value)
}

fn checked_add(left: f64, right: f64) -> Result<f64, RobustReachabilityError> {
    finite("addition", left + right)
}

fn checked_sub(left: f64, right: f64) -> Result<f64, RobustReachabilityError> {
    finite("subtraction", left - right)
}

fn checked_mul(left: f64, right: f64) -> Result<f64, RobustReachabilityError> {
    finite("multiplication", left * right)
}

fn checked_div(left: f64, right: f64) -> Result<f64, RobustReachabilityError> {
    if right == 0.0 {
        return Err(RobustReachabilityError::Numerical {
            reason: "division by zero in robust reachability theorem".to_string(),
        });
    }
    finite("division", left / right)
}

fn finite(operation: &str, value: f64) -> Result<f64, RobustReachabilityError> {
    if !value.is_finite() {
        return Err(RobustReachabilityError::Numerical {
            reason: format!("robust reachability {operation} became non-finite"),
        });
    }
    Ok(canonical_zero(value))
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}
