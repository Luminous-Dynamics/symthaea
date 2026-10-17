// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Narrow appliance adapter over the shared finite-hypothesis acquisition kernel.
//!
//! This module does not recompute EIG or expected decision value. It adds only
//! appliance observation/protection gates and explicit abstention outcomes.
//! All scores, likelihood math, feasibility filtering, and model digests remain
//! owned by `acquisition.rs`.

use serde::{Deserialize, Serialize};

use crate::{
    AcquisitionObjective, AcquisitionRanking, FiniteHypothesisModel, rank_feasible_actions,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplianceProtectionState {
    Clear,
    Tripped,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplianceObservationQuality {
    Complete,
    Missing,
    Stale,
    Conflicting,
    InvalidUnits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplianceAbstentionReason {
    ProtectionTripped,
    ProtectionUnknown,
    ObservationsMissing,
    ObservationsStale,
    ObservationsConflicting,
    ObservationUnitsInvalid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ApplianceAcquisitionDecision {
    Ranked {
        ranking: AcquisitionRanking,
    },
    NoDecisionValue {
        ranking: AcquisitionRanking,
    },
    NoDiscriminatingMeasurement {
        ranking: AcquisitionRanking,
    },
    NoEligibleMeasurement {
        ranking: AcquisitionRanking,
    },
    Abstain {
        reason: ApplianceAbstentionReason,
    },
}

/// Apply appliance-specific evidence gates and delegate all acquisition math to
/// the shared finite-hypothesis kernel.
///
/// This function is advisory only. It returns a ranking/abstention state and
/// never commands hardware or marks an appliance safe to operate.
pub fn rank_appliance_measurements(
    model: &FiniteHypothesisModel,
    protection: ApplianceProtectionState,
    observations: ApplianceObservationQuality,
    objective: AcquisitionObjective,
) -> Result<ApplianceAcquisitionDecision, String> {
    // Validate the declared model before using it, but never use a score to
    // override a protection event or unreliable physical observations.
    model.validate()?;

    let protection_reason = match protection {
        ApplianceProtectionState::Clear => None,
        ApplianceProtectionState::Tripped => {
            Some(ApplianceAbstentionReason::ProtectionTripped)
        }
        ApplianceProtectionState::Unknown => {
            Some(ApplianceAbstentionReason::ProtectionUnknown)
        }
    };
    if let Some(reason) = protection_reason {
        return Ok(ApplianceAcquisitionDecision::Abstain { reason });
    }

    let observation_reason = match observations {
        ApplianceObservationQuality::Complete => None,
        ApplianceObservationQuality::Missing => {
            Some(ApplianceAbstentionReason::ObservationsMissing)
        }
        ApplianceObservationQuality::Stale => {
            Some(ApplianceAbstentionReason::ObservationsStale)
        }
        ApplianceObservationQuality::Conflicting => {
            Some(ApplianceAbstentionReason::ObservationsConflicting)
        }
        ApplianceObservationQuality::InvalidUnits => {
            Some(ApplianceAbstentionReason::ObservationUnitsInvalid)
        }
    };
    if let Some(reason) = observation_reason {
        return Ok(ApplianceAcquisitionDecision::Abstain { reason });
    }

    let ranking = rank_feasible_actions(model, objective)?;
    if ranking.ranked_actions.is_empty() {
        return Ok(ApplianceAcquisitionDecision::NoEligibleMeasurement { ranking });
    }

    let best = &ranking.ranked_actions[0].metrics;
    match objective {
        AcquisitionObjective::ExpectedDecisionValue
        | AcquisitionObjective::DecisionValuePerCost
            if best.expected_sample_information_value_milli <= 1e-9 =>
        {
            Ok(ApplianceAcquisitionDecision::NoDecisionValue { ranking })
        }
        AcquisitionObjective::ExpectedInformationGain
        | AcquisitionObjective::InformationGainPerCost
            if best.expected_information_gain_nats <= 1e-12 =>
        {
            Ok(ApplianceAcquisitionDecision::NoDiscriminatingMeasurement { ranking })
        }
        _ => Ok(ApplianceAcquisitionDecision::Ranked { ranking }),
    }
}
