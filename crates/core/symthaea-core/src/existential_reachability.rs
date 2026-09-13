// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Success-only constructive existential terminal reachability for MANIFOLD-006C1.
//!
//! This module deliberately does **not** infer existence from overlap with a conservative
//! outer reachable enclosure. Existence is established only by one subject-bound admissible
//! `(control, disturbance)` realization whose terminal value is independently replayed through
//! MANIFOLD-006C's outward-conservative arithmetic and fully enclosed by the target.
//!
//! Failure to certify a candidate has no infeasibility authority.

use blake3::Hasher;
use thiserror::Error;

use crate::kinodynamic_reachability::BoundedSingleIntegrator1D;
use crate::robust_reachability::{
    BoundedDisturbance1D, RobustSingleIntegratorQuery1D, TerminalInterval1D,
    fixed_control_terminal_envelope,
};

/// Constructive existential terminal witness for one fixed admissible realization.
#[derive(Clone, Debug, PartialEq)]
pub struct ExistentialTerminalWitness1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    control: f64,
    disturbance: f64,
    terminal_enclosure: TerminalInterval1D,
    identity: [u8; 32],
}

impl ExistentialTerminalWitness1D {
    /// Exact model identity bound by the witness.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }

    /// Exact original robust-query identity bound by the witness.
    pub fn query_identity(&self) -> [u8; 32] {
        self.query_identity
    }

    /// Fixed admissible open-loop control used by this realization.
    pub fn control(&self) -> f64 {
        self.control
    }

    /// Fixed admissible disturbance realization used by this witness.
    pub fn disturbance(&self) -> f64 {
        self.disturbance
    }

    /// Conservative outward terminal enclosure for the fixed realization.
    pub fn terminal_enclosure(&self) -> TerminalInterval1D {
        self.terminal_enclosure
    }

    /// Deterministic witness identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Independent verification receipt for a constructive existential witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExistentialWitnessVerificationReceipt1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    witness_identity: [u8; 32],
    verifier_identity: [u8; 32],
}

impl ExistentialWitnessVerificationReceipt1D {
    /// Exact model identity.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }

    /// Exact original query identity.
    pub fn query_identity(&self) -> [u8; 32] {
        self.query_identity
    }

    /// Verified witness identity.
    pub fn witness_identity(&self) -> [u8; 32] {
        self.witness_identity
    }

    /// Distinct verifier-domain identity for constructive existential authority.
    pub fn verifier_identity(&self) -> [u8; 32] {
        self.verifier_identity
    }
}

/// One positively established existential terminal theorem.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedExistentialTerminal1D {
    witness: ExistentialTerminalWitness1D,
    verification: ExistentialWitnessVerificationReceipt1D,
}

impl CertifiedExistentialTerminal1D {
    /// Constructive realization that establishes existence.
    pub fn witness(&self) -> &ExistentialTerminalWitness1D {
        &self.witness
    }

    /// Independent verification receipt for the witness.
    pub fn verification(&self) -> &ExistentialWitnessVerificationReceipt1D {
        &self.verification
    }
}

/// Fail-closed errors while attempting or replaying constructive existential evidence.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ExistentialReachabilityError {
    /// The original query belongs to a different model.
    #[error("existential witness model/query mismatch")]
    ModelQueryMismatch,
    /// Proposed fixed control is not admissible.
    #[error("existential control {control} is outside [{minimum}, {maximum}]")]
    ControlOutsideBounds {
        /// Proposed control.
        control: f64,
        /// Inclusive model minimum.
        minimum: f64,
        /// Inclusive model maximum.
        maximum: f64,
    },
    /// Proposed fixed disturbance is not a member of the original disturbance set.
    #[error("existential disturbance {disturbance} is outside [{minimum}, {maximum}]")]
    DisturbanceOutsideBounds {
        /// Proposed disturbance.
        disturbance: f64,
        /// Inclusive disturbance minimum.
        minimum: f64,
        /// Inclusive disturbance maximum.
        maximum: f64,
    },
    /// The candidate realization does not establish terminal target membership.
    #[error("candidate realization does not establish existential target membership")]
    CandidateNotEstablished,
    /// Existing witness is not bound to this model/query.
    #[error("existential witness subject identity mismatch")]
    WitnessSubjectMismatch,
    /// Existing witness payload disagrees with independent replay.
    #[error("existential witness replay mismatch: {reason}")]
    WitnessReplayMismatch {
        /// Bounded diagnostic reason.
        reason: String,
    },
    /// Underlying qualified 006C arithmetic rejected the synthetic replay.
    #[error("existential witness arithmetic replay failed: {reason}")]
    ArithmeticReplay {
        /// Bounded diagnostic reason.
        reason: String,
    },
}

/// Certify one explicit admissible realization as existential terminal evidence.
///
/// This is success-only authority. An error means only that this candidate did not establish
/// existence; it is never a global not-reachable theorem.
pub fn certify_existential_terminal_witness(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
    control: f64,
    disturbance: f64,
) -> Result<CertifiedExistentialTerminal1D, ExistentialReachabilityError> {
    validate_subject_and_candidate(model, query, control, disturbance)?;
    let control = canonical_zero(control);
    let disturbance = canonical_zero(disturbance);
    let terminal_enclosure = replay_fixed_realization(model, query, control, disturbance)?;
    if !query.target().contains_interval(terminal_enclosure) {
        return Err(ExistentialReachabilityError::CandidateNotEstablished);
    }

    let witness = ExistentialTerminalWitness1D {
        model_identity: model.identity(),
        query_identity: query.identity(),
        control,
        disturbance,
        terminal_enclosure,
        identity: hash_witness(
            model.identity(),
            query.identity(),
            control,
            disturbance,
            terminal_enclosure,
        ),
    };
    let verification = verify_existential_terminal_witness(model, query, &witness)?;
    Ok(CertifiedExistentialTerminal1D {
        witness,
        verification,
    })
}

/// Independently replay one constructive existential witness.
pub fn verify_existential_terminal_witness(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
    witness: &ExistentialTerminalWitness1D,
) -> Result<ExistentialWitnessVerificationReceipt1D, ExistentialReachabilityError> {
    if query.model_identity() != model.identity() {
        return Err(ExistentialReachabilityError::ModelQueryMismatch);
    }
    if witness.model_identity != model.identity() || witness.query_identity != query.identity() {
        return Err(ExistentialReachabilityError::WitnessSubjectMismatch);
    }
    validate_subject_and_candidate(model, query, witness.control, witness.disturbance)?;

    let replayed = replay_fixed_realization(model, query, witness.control, witness.disturbance)?;
    if replayed != witness.terminal_enclosure {
        return Err(ExistentialReachabilityError::WitnessReplayMismatch {
            reason: "terminal enclosure does not match independent outward replay".to_string(),
        });
    }
    if !query.target().contains_interval(replayed) {
        return Err(ExistentialReachabilityError::WitnessReplayMismatch {
            reason: "replayed fixed realization is not fully enclosed by target".to_string(),
        });
    }

    let expected = hash_witness(
        model.identity(),
        query.identity(),
        witness.control,
        witness.disturbance,
        replayed,
    );
    if witness.identity != expected {
        return Err(ExistentialReachabilityError::WitnessReplayMismatch {
            reason: "witness identity does not match theorem inputs".to_string(),
        });
    }

    Ok(ExistentialWitnessVerificationReceipt1D {
        model_identity: model.identity(),
        query_identity: query.identity(),
        witness_identity: witness.identity(),
        verifier_identity: existential_verifier_identity(),
    })
}

fn validate_subject_and_candidate(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
    control: f64,
    disturbance: f64,
) -> Result<(), ExistentialReachabilityError> {
    if query.model_identity() != model.identity() {
        return Err(ExistentialReachabilityError::ModelQueryMismatch);
    }
    let controls = model.controls();
    if !controls.contains(control) {
        return Err(ExistentialReachabilityError::ControlOutsideBounds {
            control,
            minimum: controls.minimum(),
            maximum: controls.maximum(),
        });
    }
    let disturbances = query.disturbance();
    if !disturbance.is_finite()
        || disturbance < disturbances.minimum()
        || disturbance > disturbances.maximum()
    {
        return Err(ExistentialReachabilityError::DisturbanceOutsideBounds {
            disturbance,
            minimum: disturbances.minimum(),
            maximum: disturbances.maximum(),
        });
    }
    Ok(())
}

fn replay_fixed_realization(
    model: &BoundedSingleIntegrator1D,
    query: &RobustSingleIntegratorQuery1D,
    control: f64,
    disturbance: f64,
) -> Result<TerminalInterval1D, ExistentialReachabilityError> {
    let point_disturbance = BoundedDisturbance1D::new(disturbance, disturbance).map_err(|error| {
        ExistentialReachabilityError::ArithmeticReplay {
            reason: error.to_string(),
        }
    })?;
    let point_query = RobustSingleIntegratorQuery1D::new(
        model,
        query.initial_state(),
        query.target(),
        point_disturbance,
        query.plant_time(),
        query.policy(),
    )
    .map_err(|error| ExistentialReachabilityError::ArithmeticReplay {
        reason: error.to_string(),
    })?;
    fixed_control_terminal_envelope(model, &point_query, control).map_err(|error| {
        ExistentialReachabilityError::ArithmeticReplay {
            reason: error.to_string(),
        }
    })
}

fn hash_witness(
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    control: f64,
    disturbance: f64,
    terminal_enclosure: TerminalInterval1D,
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-existential-terminal-witness-1d-v1\0");
    hasher.update(b"success-only-fixed-realization-outward-replay\0");
    hasher.update(&model_identity);
    hasher.update(&query_identity);
    hasher.update(&canonical_zero(control).to_bits().to_le_bytes());
    hasher.update(&canonical_zero(disturbance).to_bits().to_le_bytes());
    hasher.update(&terminal_enclosure.identity());
    *hasher.finalize().as_bytes()
}

fn existential_verifier_identity() -> [u8; 32] {
    *blake3::hash(
        b"symthaea-existential-terminal-verifier-1d-v1\0success-only-fixed-realization-outward-replay\0",
    )
    .as_bytes()
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}
