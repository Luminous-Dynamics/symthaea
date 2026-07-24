// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic staged rollout gates for authorized releases.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::release_promotion::AuthorizedReleasePromotion;
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};

pub const ROLLOUT_PLAN_SCHEMA: &str = "symthaea.fabrication.rollout-plan.v1";
pub const ROLLOUT_OBSERVATION_SCHEMA: &str = "symthaea.fabrication.rollout-observation.v1";
pub const ROLLOUT_ADVANCE_SCHEMA: &str = "symthaea.fabrication.rollout-advance.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RolloutPhase {
    Canary,
    Limited,
    General,
}

impl RolloutPhase {
    pub fn next(self) -> Option<Self> {
        match self {
            Self::Canary => Some(Self::Limited),
            Self::Limited => Some(Self::General),
            Self::General => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutPlan {
    pub schema_version: String,
    pub promotion_digest: Sha256Digest,
    pub canary_machine_limit: u32,
    pub limited_machine_limit: u32,
    pub minimum_observation_s: u64,
    pub minimum_attempts_per_phase: u32,
    pub maximum_failure_basis_points: u16,
    pub created_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutObservation {
    pub schema_version: String,
    pub promotion_digest: Sha256Digest,
    pub phase: RolloutPhase,
    pub started_at_unix_s: u64,
    pub ended_at_unix_s: u64,
    pub attempted_jobs: u32,
    pub successful_jobs: u32,
    pub failed_jobs: u32,
    pub uncertain_jobs: u32,
    pub emergency_stops: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutAdvance {
    pub schema_version: String,
    pub promotion_digest: Sha256Digest,
    pub plan_digest: Sha256Digest,
    pub observation_digest: Sha256Digest,
    pub from_phase: RolloutPhase,
    pub to_phase: RolloutPhase,
    pub advance_sequence: u64,
    pub authorized_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RolloutError {
    UnsupportedSchema,
    InvalidPlan,
    InvalidObservation,
    PromotionMismatch,
    PhaseMismatch,
    TerminalPhase,
    ObservationTooShort {
        actual_s: u64,
        required_s: u64,
    },
    InsufficientAttempts {
        actual: u32,
        required: u32,
    },
    CountMismatch,
    FailureBudgetExceeded {
        actual_basis_points: u16,
        maximum_basis_points: u16,
    },
    EmergencyStopObserved,
    InvalidWindow,
    SequenceZero,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedRolloutAdvance {
    advance: RolloutAdvance,
    advance_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedRolloutAdvance {
    pub fn advance(&self) -> &RolloutAdvance {
        &self.advance
    }
    pub fn advance_digest(&self) -> Sha256Digest {
        self.advance_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutTracker {
    promotion_digest: Option<Sha256Digest>,
    current_phase: Option<RolloutPhase>,
    latest_advance_sequence: Option<u64>,
    latest_advance_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RolloutTrackingError {
    PromotionSubstitution,
    PhaseSkip,
    SequenceRollback,
    SameSequenceSubstitution,
}

impl RolloutPlan {
    pub fn new(
        promotion: &AuthorizedReleasePromotion,
        canary_machine_limit: u32,
        limited_machine_limit: u32,
        minimum_observation_s: u64,
        minimum_attempts_per_phase: u32,
        maximum_failure_basis_points: u16,
        created_at_unix_s: u64,
        expires_at_unix_s: u64,
    ) -> Result<Self, RolloutError> {
        let plan = Self {
            schema_version: ROLLOUT_PLAN_SCHEMA.into(),
            promotion_digest: promotion.promotion_digest(),
            canary_machine_limit,
            limited_machine_limit,
            minimum_observation_s,
            minimum_attempts_per_phase,
            maximum_failure_basis_points,
            created_at_unix_s,
            expires_at_unix_s,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), RolloutError> {
        if self.schema_version != ROLLOUT_PLAN_SCHEMA {
            return Err(RolloutError::UnsupportedSchema);
        }
        if self.canary_machine_limit == 0
            || self.limited_machine_limit < self.canary_machine_limit
            || self.minimum_observation_s == 0
            || self.minimum_attempts_per_phase == 0
            || self.maximum_failure_basis_points > 10_000
        {
            return Err(RolloutError::InvalidPlan);
        }
        if self.created_at_unix_s >= self.expires_at_unix_s {
            return Err(RolloutError::InvalidWindow);
        }
        Ok(())
    }
}

impl RolloutObservation {
    pub fn validate(&self) -> Result<(), RolloutError> {
        if self.schema_version != ROLLOUT_OBSERVATION_SCHEMA {
            return Err(RolloutError::UnsupportedSchema);
        }
        if self.started_at_unix_s >= self.ended_at_unix_s || self.attempted_jobs == 0 {
            return Err(RolloutError::InvalidObservation);
        }
        if self
            .successful_jobs
            .saturating_add(self.failed_jobs)
            .saturating_add(self.uncertain_jobs)
            != self.attempted_jobs
        {
            return Err(RolloutError::CountMismatch);
        }
        Ok(())
    }
}

pub fn digest_rollout_plan(plan: &RolloutPlan) -> Result<Sha256Digest, RolloutError> {
    plan.validate()?;
    digest_serialized(b"symthaea.fabrication.rollout-plan-digest.v1\0", plan)
}

pub fn digest_rollout_observation(
    observation: &RolloutObservation,
) -> Result<Sha256Digest, RolloutError> {
    observation.validate()?;
    digest_serialized(
        b"symthaea.fabrication.rollout-observation-digest.v1\0",
        observation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn authorize_rollout_advance(
    plan: &RolloutPlan,
    observation: &RolloutObservation,
    advance_sequence: u64,
    authorized_at_unix_s: u64,
    expires_at_unix_s: u64,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedRolloutAdvance, RolloutError> {
    plan.validate()?;
    observation.validate()?;
    if advance_sequence == 0 {
        return Err(RolloutError::SequenceZero);
    }
    if observation.promotion_digest != plan.promotion_digest {
        return Err(RolloutError::PromotionMismatch);
    }
    let Some(next_phase) = observation.phase.next() else {
        return Err(RolloutError::TerminalPhase);
    };
    let elapsed = observation.ended_at_unix_s - observation.started_at_unix_s;
    if elapsed < plan.minimum_observation_s {
        return Err(RolloutError::ObservationTooShort {
            actual_s: elapsed,
            required_s: plan.minimum_observation_s,
        });
    }
    if observation.attempted_jobs < plan.minimum_attempts_per_phase {
        return Err(RolloutError::InsufficientAttempts {
            actual: observation.attempted_jobs,
            required: plan.minimum_attempts_per_phase,
        });
    }
    if observation.emergency_stops != 0 {
        return Err(RolloutError::EmergencyStopObserved);
    }
    let adverse = observation
        .failed_jobs
        .saturating_add(observation.uncertain_jobs);
    let failure_basis_points =
        ((u64::from(adverse) * 10_000) / u64::from(observation.attempted_jobs)) as u16;
    if failure_basis_points > plan.maximum_failure_basis_points {
        return Err(RolloutError::FailureBudgetExceeded {
            actual_basis_points: failure_basis_points,
            maximum_basis_points: plan.maximum_failure_basis_points,
        });
    }
    if authorized_at_unix_s >= expires_at_unix_s
        || authorized_at_unix_s < observation.ended_at_unix_s
        || expires_at_unix_s > plan.expires_at_unix_s
    {
        return Err(RolloutError::InvalidWindow);
    }
    let advance = RolloutAdvance {
        schema_version: ROLLOUT_ADVANCE_SCHEMA.into(),
        promotion_digest: plan.promotion_digest,
        plan_digest: digest_rollout_plan(plan)?,
        observation_digest: digest_rollout_observation(observation)?,
        from_phase: observation.phase,
        to_phase: next_phase,
        advance_sequence,
        authorized_at_unix_s,
        expires_at_unix_s,
    };
    let advance_digest = digest_rollout_advance(&advance)?;
    if ceremony.purpose() != "release-rollout-advance" {
        return Err(RolloutError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != advance_digest {
        return Err(RolloutError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedRolloutAdvance {
        advance,
        advance_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

pub fn digest_rollout_advance(advance: &RolloutAdvance) -> Result<Sha256Digest, RolloutError> {
    if advance.schema_version != ROLLOUT_ADVANCE_SCHEMA {
        return Err(RolloutError::UnsupportedSchema);
    }
    if advance.advance_sequence == 0 || advance.from_phase.next() != Some(advance.to_phase) {
        return Err(RolloutError::PhaseMismatch);
    }
    if advance.authorized_at_unix_s >= advance.expires_at_unix_s {
        return Err(RolloutError::InvalidWindow);
    }
    digest_serialized(b"symthaea.fabrication.rollout-advance-digest.v1\0", advance)
}

impl RolloutTracker {
    pub fn begin(&mut self, promotion_digest: Sha256Digest) -> Result<(), RolloutTrackingError> {
        if self
            .promotion_digest
            .is_some_and(|digest| digest != promotion_digest)
        {
            return Err(RolloutTrackingError::PromotionSubstitution);
        }
        self.promotion_digest = Some(promotion_digest);
        self.current_phase.get_or_insert(RolloutPhase::Canary);
        Ok(())
    }

    pub fn accept(
        &mut self,
        authorized: &AuthorizedRolloutAdvance,
    ) -> Result<(), RolloutTrackingError> {
        let advance = authorized.advance();
        self.begin(advance.promotion_digest)?;
        if self.current_phase != Some(advance.from_phase) {
            return Err(RolloutTrackingError::PhaseSkip);
        }
        if let Some(latest) = self.latest_advance_sequence {
            if advance.advance_sequence < latest {
                return Err(RolloutTrackingError::SequenceRollback);
            }
            if advance.advance_sequence == latest {
                if self.latest_advance_digest == Some(authorized.advance_digest()) {
                    return Ok(());
                }
                return Err(RolloutTrackingError::SameSequenceSubstitution);
            }
            if advance.advance_sequence != latest.saturating_add(1) {
                return Err(RolloutTrackingError::PhaseSkip);
            }
        }
        self.current_phase = Some(advance.to_phase);
        self.latest_advance_sequence = Some(advance.advance_sequence);
        self.latest_advance_digest = Some(authorized.advance_digest());
        Ok(())
    }

    pub fn current_phase(&self) -> Option<RolloutPhase> {
        self.current_phase
    }
}

fn digest_serialized<T: Serialize>(domain: &[u8], value: &T) -> Result<Sha256Digest, RolloutError> {
    let bytes =
        serde_json::to_vec(value).map_err(|error| RolloutError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_count_mismatch_fails() {
        let observation = RolloutObservation {
            schema_version: ROLLOUT_OBSERVATION_SCHEMA.into(),
            promotion_digest: Sha256Digest([1; 32]),
            phase: RolloutPhase::Canary,
            started_at_unix_s: 1,
            ended_at_unix_s: 2,
            attempted_jobs: 10,
            successful_jobs: 9,
            failed_jobs: 0,
            uncertain_jobs: 0,
            emergency_stops: 0,
        };
        assert_eq!(observation.validate(), Err(RolloutError::CountMismatch));
    }

    #[test]
    fn phase_order_is_total() {
        assert_eq!(RolloutPhase::Canary.next(), Some(RolloutPhase::Limited));
        assert_eq!(RolloutPhase::Limited.next(), Some(RolloutPhase::General));
        assert_eq!(RolloutPhase::General.next(), None);
    }
}
