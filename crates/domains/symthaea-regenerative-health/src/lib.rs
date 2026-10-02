pub mod crucible;
pub mod sensor_health;
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic regenerative health contract for cyber-physical vehicles.
//!
//! Diagnosis, intervention, and recovery are deliberately separate.
//! A repair action never implies recovery; recovery requires fresh,
//! provenance-bearing post-action evidence.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegenerativeHealthState {
    Nominal, Suspect, Restricted, RepairPending, Healing, VerificationPending, Recovered, Quarantined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegenerativeAction {
    ReduceLoad, IsolateComponent, ThermalHeal, SealLeak, QualifiedRepair, ReplaceModule, Inspect,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthObservation {
    pub observation_id: String,
    pub component_id: String,
    pub timestamp_ms: u64,
    pub normalized_residual: f64,
    pub uncertainty: f64,
    pub evidence_ids: Vec<String>,
    pub configuration_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveryEvidence {
    pub evidence_id: String,
    pub component_id: String,
    pub timestamp_ms: u64,
    pub normalized_residual: f64,
    pub uncertainty: f64,
    pub configuration_digest: String,
    pub intervention_id: String,
    pub independent_verification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegenerativePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub warning_sigma_milli: u32,
    pub restricted_sigma_milli: u32,
    pub recovery_sigma_milli: u32,
    pub maximum_observation_age_ms: u64,
    pub maximum_recovery_age_ms: u64,
    pub allowed_actions: BTreeSet<RegenerativeAction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegenerativeIssue {
    EmptyIdentity, InvalidUncertainty, InvalidResidual, MissingEvidence, StaleObservation,
    FutureObservation,
    UnqualifiedAction(RegenerativeAction), MissingRecoveryEvidence, StaleRecoveryEvidence,
    FutureRecoveryEvidence,
    RecoveryResidualTooHigh, RecoveryNotIndependent, ConfigurationMismatch,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegenerativeDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub component_id: String,
    pub state: RegenerativeHealthState,
    pub actions: Vec<RegenerativeAction>,
    pub issues: Vec<RegenerativeIssue>,
}

impl RegenerativeDecision {
    pub fn canonical_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }
}

#[derive(Debug, Clone)]
pub struct RegenerativeHealthGate {
    policy: RegenerativePolicy,
}

impl RegenerativeHealthGate {
    pub fn new(policy: RegenerativePolicy) -> Result<Self, &'static str> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.maximum_observation_age_ms == 0
            || policy.maximum_recovery_age_ms == 0
            || policy.warning_sigma_milli == 0
            || policy.restricted_sigma_milli <= policy.warning_sigma_milli
            || policy.recovery_sigma_milli >= policy.restricted_sigma_milli
            || policy.allowed_actions.is_empty()
        { return Err("invalid regenerative policy"); }
        Ok(Self { policy })
    }

    pub fn assess(&self, observation: &HealthObservation, recovery: Option<&RecoveryEvidence>, requested_action: Option<RegenerativeAction>, now_ms: u64) -> RegenerativeDecision {
        let mut issues = Vec::new();
        if observation.observation_id.trim().is_empty() || observation.component_id.trim().is_empty() || observation.configuration_digest.trim().is_empty() {
            issues.push(RegenerativeIssue::EmptyIdentity);
        }
        if !observation.normalized_residual.is_finite() || observation.normalized_residual < 0.0 {
            issues.push(RegenerativeIssue::InvalidResidual);
        }
        if !observation.uncertainty.is_finite() || observation.uncertainty <= 0.0 {
            issues.push(RegenerativeIssue::InvalidUncertainty);
        }
        if observation.evidence_ids.is_empty() || observation.evidence_ids.iter().any(|id| id.trim().is_empty()) {
            issues.push(RegenerativeIssue::MissingEvidence);
        }
        if observation.timestamp_ms > now_ms {
            issues.push(RegenerativeIssue::FutureObservation);
        } else if now_ms.saturating_sub(observation.timestamp_ms) > self.policy.maximum_observation_age_ms {
            issues.push(RegenerativeIssue::StaleObservation);
        }
        if let Some(action) = requested_action {
            if !self.policy.allowed_actions.contains(&action) {
                issues.push(RegenerativeIssue::UnqualifiedAction(action));
            }
        }

        let residual_milli = if observation.normalized_residual.is_finite() && observation.normalized_residual >= 0.0 {
            (observation.normalized_residual * 1000.0).round() as u32
        } else { 0 };
        let mut state = if !issues.is_empty() {
            RegenerativeHealthState::Quarantined
        } else if residual_milli >= self.policy.restricted_sigma_milli {
            RegenerativeHealthState::Restricted
        } else if residual_milli >= self.policy.warning_sigma_milli {
            RegenerativeHealthState::Suspect
        } else if let Some(action) = requested_action {
            match action {
                RegenerativeAction::ReduceLoad | RegenerativeAction::IsolateComponent | RegenerativeAction::Inspect => RegenerativeHealthState::Restricted,
                _ => if recovery.is_some() { RegenerativeHealthState::VerificationPending } else { RegenerativeHealthState::RepairPending },
            }
        } else { RegenerativeHealthState::Nominal };

        let mut actions = Vec::new();
        if state == RegenerativeHealthState::Restricted { actions.push(RegenerativeAction::ReduceLoad); }

        if let (Some(recovery), Some(action)) = (recovery, requested_action) {
            if !matches!(action, RegenerativeAction::ReduceLoad | RegenerativeAction::IsolateComponent | RegenerativeAction::Inspect) {
                if recovery.component_id != observation.component_id || recovery.configuration_digest != observation.configuration_digest {
                    issues.push(RegenerativeIssue::ConfigurationMismatch);
                }
                if recovery.evidence_id.trim().is_empty() || recovery.intervention_id.trim().is_empty() {
                    issues.push(RegenerativeIssue::MissingRecoveryEvidence);
                }
                if !recovery.normalized_residual.is_finite() || recovery.normalized_residual < 0.0 || !recovery.uncertainty.is_finite() || recovery.uncertainty <= 0.0 {
                    issues.push(RegenerativeIssue::InvalidResidual);
                }
                if recovery.timestamp_ms > now_ms {
                    issues.push(RegenerativeIssue::FutureRecoveryEvidence);
                } else if now_ms.saturating_sub(recovery.timestamp_ms) > self.policy.maximum_recovery_age_ms {
                    issues.push(RegenerativeIssue::StaleRecoveryEvidence);
                }
                let recovery_milli = if recovery.normalized_residual.is_finite() && recovery.normalized_residual >= 0.0 { (recovery.normalized_residual * 1000.0).round() as u32 } else { u32::MAX };
                if recovery_milli > self.policy.recovery_sigma_milli { issues.push(RegenerativeIssue::RecoveryResidualTooHigh); }
                if !recovery.independent_verification { issues.push(RegenerativeIssue::RecoveryNotIndependent); }
                if issues.is_empty() {
                    state = RegenerativeHealthState::Recovered;
                    actions.push(action);
                }
            }
        }

        RegenerativeDecision { schema_version: self.policy.schema_version.clone(), policy_id: self.policy.policy_id.clone(), component_id: observation.component_id.clone(), state, actions, issues }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn gate() -> RegenerativeHealthGate {
        RegenerativeHealthGate::new(RegenerativePolicy {
            schema_version: "0.1".into(), policy_id: "road-v1".into(),
            warning_sigma_milli: 2000, restricted_sigma_milli: 4000, recovery_sigma_milli: 1500,
            maximum_observation_age_ms: 1000, maximum_recovery_age_ms: 1000,
            allowed_actions: [RegenerativeAction::ThermalHeal, RegenerativeAction::QualifiedRepair].into_iter().collect(),
        }).unwrap()
    }
    fn observation(residual: f64) -> HealthObservation {
        HealthObservation { observation_id: "obs-1".into(), component_id: "battery-1".into(), timestamp_ms: 1000, normalized_residual: residual, uncertainty: 1.0, evidence_ids: vec!["e-1".into()], configuration_digest: "cfg-1".into() }
    }
    #[test] fn repair_never_implies_recovery() {
        let d = gate().assess(&observation(0.5), None, Some(RegenerativeAction::ThermalHeal), 1000);
        assert_eq!(d.state, RegenerativeHealthState::RepairPending);
    }
    #[test] fn missing_independent_verification_cannot_recover() {
        let r = RecoveryEvidence { evidence_id: "r-1".into(), component_id: "battery-1".into(), timestamp_ms: 1000, normalized_residual: 0.1, uncertainty: 1.0, configuration_digest: "cfg-1".into(), intervention_id: "heal-1".into(), independent_verification: false };
        let d = gate().assess(&observation(0.1), Some(&r), Some(RegenerativeAction::ThermalHeal), 1000);
        assert_ne!(d.state, RegenerativeHealthState::Recovered);
        assert!(d.issues.contains(&RegenerativeIssue::RecoveryNotIndependent));
    }
    #[test] fn qualified_repair_requires_fresh_independent_evidence() {
        let r = RecoveryEvidence { evidence_id: "r-1".into(), component_id: "battery-1".into(), timestamp_ms: 1000, normalized_residual: 0.5, uncertainty: 1.0, configuration_digest: "cfg-1".into(), intervention_id: "heal-1".into(), independent_verification: true };
        let d = gate().assess(&observation(0.5), Some(&r), Some(RegenerativeAction::ThermalHeal), 1000);
        assert_eq!(d.state, RegenerativeHealthState::Recovered);
    }
    #[test] fn stale_observation_quarantines() {
        let d = gate().assess(&observation(0.1), None, None, 3001);
        assert_eq!(d.state, RegenerativeHealthState::Quarantined);
    }
    #[test] fn unqualified_action_cannot_enter_healing_path() {
        let d = gate().assess(&observation(0.5), None, Some(RegenerativeAction::ReplaceModule), 1000);
        assert_eq!(d.state, RegenerativeHealthState::Quarantined);
        assert!(d.issues.contains(&RegenerativeIssue::UnqualifiedAction(RegenerativeAction::ReplaceModule)));
    }

    #[test] fn future_observation_quarantines() {
        let d = gate().assess(&observation(0.1), None, None, 999);
        assert_eq!(d.state, RegenerativeHealthState::Quarantined);
        assert!(d.issues.contains(&RegenerativeIssue::FutureObservation));
    }
}
