// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Graduated defense response types for the immune system architecture.
//!
//! Maps SafetyLevel to proportional defensive actions. Every action
//! is evaluated by the moral algebra before execution.

use serde::{Deserialize, Serialize};

use crate::safety::SafetyLevel;

/// A proposed defensive action with moral evaluation metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefenseAction {
    /// What kind of defensive action.
    pub kind: DefenseActionKind,
    /// Safety level that triggered this action.
    pub trigger_level: SafetyLevel,
    /// Moral severity score (0.0 = harmless, 1.0 = extreme).
    /// Actions above DEFENSE_MAX_MORAL_SEVERITY are blocked.
    pub moral_severity: f32,
    /// Whether this action was approved by moral algebra.
    pub morally_approved: bool,
    /// Cycle at which this action was proposed.
    pub proposed_at: usize,
    /// Cycle at which this action expires (auto-release).
    pub expires_at: usize,
    /// Human-readable reason for this action.
    pub reason: String,
}

/// Enumeration of graduated defensive actions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DefenseActionKind {
    // -- Yellow (elevated monitoring) --
    /// Increase peer behavior scoring frequency.
    IncreasePeerScoring { interval_reduction: f32 },
    /// Rate-limit governance proposals for a specific agent.
    RateLimitProposals {
        agent_id: String,
        max_per_minute: u32,
    },
    /// Boost neuromodulator vigilance (NE).
    BoostVigilance { ne_delta: f32 },

    // -- Orange (active intervention) --
    /// Temporarily quarantine a peer (reduce trust, limit interaction).
    QuarantinePeer {
        peer_id: String,
        duration_cycles: u32,
    },
    /// Suspend non-Guardian governance proposals.
    SuspendNonGuardianProposals,
    /// Reduce motor output to read-only.
    RestrictMotorToReadOnly,
    /// Boost cortisol stress response.
    StressResponse { cortisol_delta: f32 },

    // -- Red (emergency halt) --
    /// Emergency governance freeze (Guardians only).
    EmergencyGovernanceFreeze,
    /// Disconnect from all untrusted peers.
    DisconnectUntrusted,
    /// Physical safe position (HAL e-stop).
    PhysicalSafePosition,
    /// Emergency radio beacon.
    EmergencyBeacon,
    /// Halt all motor output.
    HaltMotor,
}

impl std::fmt::Display for DefenseActionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncreasePeerScoring { .. } => f.write_str("IncreasePeerScoring"),
            Self::RateLimitProposals { .. } => f.write_str("RateLimitProposals"),
            Self::BoostVigilance { .. } => f.write_str("BoostVigilance"),
            Self::QuarantinePeer { .. } => f.write_str("QuarantinePeer"),
            Self::SuspendNonGuardianProposals => f.write_str("SuspendNonGuardianProposals"),
            Self::RestrictMotorToReadOnly => f.write_str("RestrictMotorToReadOnly"),
            Self::StressResponse { .. } => f.write_str("StressResponse"),
            Self::EmergencyGovernanceFreeze => f.write_str("EmergencyGovernanceFreeze"),
            Self::DisconnectUntrusted => f.write_str("DisconnectUntrusted"),
            Self::PhysicalSafePosition => f.write_str("PhysicalSafePosition"),
            Self::EmergencyBeacon => f.write_str("EmergencyBeacon"),
            Self::HaltMotor => f.write_str("HaltMotor"),
        }
    }
}

impl DefenseActionKind {
    /// Moral severity of this action (used by moral algebra evaluation).
    ///
    /// Lower = less harmful. Quarantine and disconnect affect other agents
    /// and thus have higher severity.
    pub fn base_severity(&self) -> f32 {
        match self {
            Self::IncreasePeerScoring { .. } => 0.05,
            Self::RateLimitProposals { .. } => 0.15,
            Self::BoostVigilance { .. } => 0.05,
            Self::QuarantinePeer { .. } => 0.45,
            Self::SuspendNonGuardianProposals => 0.35,
            Self::RestrictMotorToReadOnly => 0.10,
            Self::StressResponse { .. } => 0.10,
            Self::EmergencyGovernanceFreeze => 0.55,
            Self::DisconnectUntrusted => 0.50,
            Self::PhysicalSafePosition => 0.15,
            Self::EmergencyBeacon => 0.10,
            Self::HaltMotor => 0.20,
        }
    }

    /// Whether this action affects other agents (requires higher moral scrutiny).
    pub fn affects_others(&self) -> bool {
        matches!(
            self,
            Self::QuarantinePeer { .. }
                | Self::RateLimitProposals { .. }
                | Self::SuspendNonGuardianProposals
                | Self::DisconnectUntrusted
                | Self::EmergencyGovernanceFreeze
        )
    }
}

/// Propose graduated defense actions based on safety level.
///
/// Returns a list of actions appropriate for the given level.
/// Each action still needs moral algebra approval before execution.
pub fn propose_defense_actions(safety_level: SafetyLevel, cycle: usize) -> Vec<DefenseAction> {
    let mut actions = Vec::new();

    let make = |kind: DefenseActionKind, reason: &str, duration: usize| -> DefenseAction {
        DefenseAction {
            moral_severity: kind.base_severity(),
            kind,
            trigger_level: safety_level,
            morally_approved: false, // pending moral algebra review
            proposed_at: cycle,
            expires_at: cycle + duration,
            reason: reason.to_string(),
        }
    };

    match safety_level {
        SafetyLevel::Green => {} // no defensive actions needed
        SafetyLevel::Yellow => {
            actions.push(make(
                DefenseActionKind::IncreasePeerScoring {
                    interval_reduction: 0.5,
                },
                "Elevated monitoring: increase peer scoring frequency",
                500,
            ));
            actions.push(make(
                DefenseActionKind::BoostVigilance {
                    ne_delta: super::thresholds::SAFETY_YELLOW_NE_BOOST,
                },
                "Elevated monitoring: boost NE vigilance",
                200,
            ));
        }
        SafetyLevel::Orange => {
            actions.push(make(
                DefenseActionKind::SuspendNonGuardianProposals,
                "Active intervention: only Guardian-tier proposals accepted",
                super::thresholds::DEFENSE_QUARANTINE_MAX_CYCLES as usize,
            ));
            actions.push(make(
                DefenseActionKind::RestrictMotorToReadOnly,
                "Active intervention: motor output restricted to read-only",
                super::thresholds::DEFENSE_QUARANTINE_MAX_CYCLES as usize,
            ));
            actions.push(make(
                DefenseActionKind::StressResponse {
                    cortisol_delta: super::thresholds::SAFETY_ORANGE_CORTISOL_BOOST,
                },
                "Active intervention: cortisol stress response for threat focus",
                300,
            ));
        }
        SafetyLevel::Red => {
            actions.push(make(
                DefenseActionKind::EmergencyGovernanceFreeze,
                "Emergency: governance frozen, Guardians only",
                super::thresholds::DEFENSE_QUARANTINE_MAX_CYCLES as usize,
            ));
            actions.push(make(
                DefenseActionKind::DisconnectUntrusted,
                "Emergency: disconnecting untrusted peers",
                super::thresholds::DEFENSE_QUARANTINE_MAX_CYCLES as usize,
            ));
            actions.push(make(
                DefenseActionKind::HaltMotor,
                "Emergency: all motor output halted",
                super::thresholds::DEFENSE_QUARANTINE_MAX_CYCLES as usize,
            ));
            actions.push(make(
                DefenseActionKind::EmergencyBeacon,
                "Emergency: broadcasting distress beacon on all radio tiers",
                100,
            ));
            actions.push(make(
                DefenseActionKind::PhysicalSafePosition,
                "Emergency: commanding safe physical posture",
                super::thresholds::DEFENSE_QUARANTINE_MAX_CYCLES as usize,
            ));
        }
    }

    actions
}

/// Filter defense actions through moral algebra approval.
///
/// Actions affecting others require severity < DEFENSE_MAX_MORAL_SEVERITY.
/// Pure self-protective actions are always approved.
pub fn moral_filter(actions: &mut [DefenseAction]) {
    for action in actions.iter_mut() {
        if action.kind.affects_others() {
            action.morally_approved =
                action.moral_severity < super::thresholds::DEFENSE_MAX_MORAL_SEVERITY;
        } else {
            // Self-protective actions are always morally permissible
            action.morally_approved = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_green_no_actions() {
        let actions = propose_defense_actions(SafetyLevel::Green, 0);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_yellow_increases_monitoring() {
        let actions = propose_defense_actions(SafetyLevel::Yellow, 100);
        assert!(!actions.is_empty());
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.kind, DefenseActionKind::IncreasePeerScoring { .. }))
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.kind, DefenseActionKind::BoostVigilance { .. }))
        );
    }

    #[test]
    fn test_orange_restricts_governance_and_motor() {
        let actions = propose_defense_actions(SafetyLevel::Orange, 200);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.kind, DefenseActionKind::SuspendNonGuardianProposals))
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.kind, DefenseActionKind::RestrictMotorToReadOnly))
        );
    }

    #[test]
    fn test_red_emergency_halt() {
        let actions = propose_defense_actions(SafetyLevel::Red, 300);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.kind, DefenseActionKind::EmergencyGovernanceFreeze))
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.kind, DefenseActionKind::HaltMotor))
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.kind, DefenseActionKind::DisconnectUntrusted))
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.kind, DefenseActionKind::EmergencyBeacon))
        );
    }

    #[test]
    fn test_moral_filter_approves_self_protective() {
        let mut actions = propose_defense_actions(SafetyLevel::Red, 0);
        moral_filter(&mut actions);
        // HaltMotor and PhysicalSafePosition are self-protective
        let halt = actions
            .iter()
            .find(|a| matches!(a.kind, DefenseActionKind::HaltMotor))
            .unwrap();
        assert!(halt.morally_approved);
    }

    #[test]
    fn test_moral_filter_checks_severity_for_others() {
        let mut actions = propose_defense_actions(SafetyLevel::Red, 0);
        moral_filter(&mut actions);
        // DisconnectUntrusted and EmergencyGovernanceFreeze affect others
        for action in &actions {
            if action.kind.affects_others() {
                assert!(
                    action.moral_severity < super::super::thresholds::DEFENSE_MAX_MORAL_SEVERITY,
                    "Action {:?} has severity {} >= threshold",
                    action.kind,
                    action.moral_severity
                );
            }
        }
    }

    #[test]
    fn test_all_actions_have_expiry() {
        for level in [SafetyLevel::Yellow, SafetyLevel::Orange, SafetyLevel::Red] {
            let actions = propose_defense_actions(level, 100);
            for action in &actions {
                assert!(
                    action.expires_at > action.proposed_at,
                    "Action {:?} at level {:?} has no expiry",
                    action.kind,
                    level
                );
            }
        }
    }

    #[test]
    fn test_severity_monotonicity() {
        // Higher safety levels should produce higher average severity
        let yellow = propose_defense_actions(SafetyLevel::Yellow, 0);
        let orange = propose_defense_actions(SafetyLevel::Orange, 0);
        let red = propose_defense_actions(SafetyLevel::Red, 0);

        let avg = |actions: &[DefenseAction]| -> f32 {
            if actions.is_empty() {
                return 0.0;
            }
            actions.iter().map(|a| a.moral_severity).sum::<f32>() / actions.len() as f32
        };

        assert!(
            avg(&yellow) <= avg(&orange),
            "Yellow avg severity should be <= Orange"
        );
        // Red may include self-protective low-severity actions, so we check max instead
        let max_orange = orange
            .iter()
            .map(|a| a.moral_severity)
            .fold(0.0f32, f32::max);
        let max_red = red.iter().map(|a| a.moral_severity).fold(0.0f32, f32::max);
        assert!(
            max_orange <= max_red,
            "Orange max severity should be <= Red"
        );
    }

    #[test]
    fn test_defense_action_kind_base_severity_bounded() {
        let kinds = vec![
            DefenseActionKind::IncreasePeerScoring {
                interval_reduction: 0.5,
            },
            DefenseActionKind::RateLimitProposals {
                agent_id: "test".into(),
                max_per_minute: 5,
            },
            DefenseActionKind::BoostVigilance { ne_delta: 0.03 },
            DefenseActionKind::QuarantinePeer {
                peer_id: "test".into(),
                duration_cycles: 100,
            },
            DefenseActionKind::SuspendNonGuardianProposals,
            DefenseActionKind::RestrictMotorToReadOnly,
            DefenseActionKind::StressResponse {
                cortisol_delta: 0.05,
            },
            DefenseActionKind::EmergencyGovernanceFreeze,
            DefenseActionKind::DisconnectUntrusted,
            DefenseActionKind::PhysicalSafePosition,
            DefenseActionKind::EmergencyBeacon,
            DefenseActionKind::HaltMotor,
        ];
        for kind in &kinds {
            let sev = kind.base_severity();
            assert!(
                sev >= 0.0 && sev <= 1.0,
                "{:?} severity {} out of bounds",
                kind,
                sev
            );
        }
    }
}
