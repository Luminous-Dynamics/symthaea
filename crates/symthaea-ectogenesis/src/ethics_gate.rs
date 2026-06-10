// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consent proxy escalation and ethical intervention gating.
//!
//! The ethics gate enforces increasingly stringent requirements for
//! interventions as fetal sentience develops. This implements a
//! graduated consent proxy model:
//!
//! - **PreSentient** (weeks 0-4): Low bar (benefit_score >= 0.3).
//! - **EmergingSentience** (weeks 5-23): Guardian consent + moderate benefit (>= 0.5).
//! - **Sentient** (weeks 24+): Guardian consent + ethics board + strong benefit (>= 0.7).

use crate::types::{ConsentProxy, GestationalWeek};

/// Ethical intervention gate with consent proxy escalation.
///
/// All interventions on the developing fetus must pass through this gate,
/// which checks that appropriate consent and benefit thresholds are met
/// based on the current developmental stage.
#[derive(Debug, Clone)]
pub struct EctogenesisEthicsGate {
    /// Whether a legal guardian has given informed consent.
    pub guardian_consent: bool,
    /// Whether the institutional ethics board has approved.
    pub ethics_board_approved: bool,
}

impl EctogenesisEthicsGate {
    /// Create a new ethics gate with the given approvals.
    pub fn new(guardian_consent: bool, ethics_board_approved: bool) -> Self {
        Self {
            guardian_consent,
            ethics_board_approved,
        }
    }

    /// Create a fully approved gate (for testing).
    pub fn fully_approved() -> Self {
        Self::new(true, true)
    }

    /// Create a gate with no approvals.
    pub fn unapproved() -> Self {
        Self::new(false, false)
    }

    /// Determine the consent proxy tier for the current week.
    pub fn consent_proxy(&self, week: GestationalWeek) -> ConsentProxy {
        ConsentProxy::from_week(week)
    }

    /// Check whether a proposed intervention is ethically permissible.
    ///
    /// # Arguments
    /// * `week` - Current gestational week.
    /// * `intervention` - Description of the proposed intervention.
    /// * `benefit_score` - Expected benefit (0-1, where 1 = life-saving).
    ///
    /// # Returns
    /// `Ok(())` if the intervention is permitted, `Err(reason)` otherwise.
    pub fn check_intervention(
        &self,
        week: GestationalWeek,
        _intervention: &str,
        benefit_score: f64,
    ) -> Result<(), String> {
        let proxy = self.consent_proxy(week);

        match proxy {
            ConsentProxy::PreSentient => {
                if benefit_score < 0.3 {
                    return Err("Insufficient benefit for pre-sentient intervention".into());
                }
                Ok(())
            }
            ConsentProxy::EmergingSentience => {
                if !self.guardian_consent {
                    return Err("Guardian consent required for emerging sentience".into());
                }
                if benefit_score < 0.5 {
                    return Err("Clear benefit required for emerging sentience".into());
                }
                Ok(())
            }
            ConsentProxy::Sentient => {
                if !self.guardian_consent {
                    return Err("Guardian consent required for sentient being".into());
                }
                if !self.ethics_board_approved {
                    return Err("Ethics board approval required for sentient being".into());
                }
                if benefit_score < 0.7 {
                    return Err("Strong benefit required for sentient being intervention".into());
                }
                Ok(())
            }
        }
    }

    /// Compute the minimum benefit score required for the current week.
    pub fn minimum_benefit(&self, week: GestationalWeek) -> f64 {
        match ConsentProxy::from_week(week) {
            ConsentProxy::PreSentient => 0.3,
            ConsentProxy::EmergingSentience => 0.5,
            ConsentProxy::Sentient => 0.7,
        }
    }

    /// Whether this gate has all necessary approvals for the given week.
    pub fn has_required_approvals(&self, week: GestationalWeek) -> bool {
        match ConsentProxy::from_week(week) {
            ConsentProxy::PreSentient => true,
            ConsentProxy::EmergingSentience => self.guardian_consent,
            ConsentProxy::Sentient => self.guardian_consent && self.ethics_board_approved,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_sentient_allows_low_bar() {
        let gate = EctogenesisEthicsGate::unapproved();
        let result = gate.check_intervention(GestationalWeek::new(3), "nutrient adjustment", 0.4);
        assert!(result.is_ok(), "Pre-sentient should allow benefit >= 0.3");
    }

    #[test]
    fn test_pre_sentient_rejects_insufficient_benefit() {
        let gate = EctogenesisEthicsGate::unapproved();
        let result =
            gate.check_intervention(GestationalWeek::new(3), "experimental procedure", 0.1);
        assert!(result.is_err(), "Pre-sentient should reject benefit < 0.3");
    }

    #[test]
    fn test_emerging_sentience_requires_guardian() {
        let gate = EctogenesisEthicsGate::unapproved();
        let result = gate.check_intervention(GestationalWeek::new(15), "hormone adjustment", 0.8);
        assert!(
            result.is_err(),
            "Emerging sentience should require guardian consent"
        );
        assert!(result.unwrap_err().contains("Guardian consent"));
    }

    #[test]
    fn test_emerging_sentience_with_guardian_and_benefit() {
        let gate = EctogenesisEthicsGate::new(true, false);
        let result = gate.check_intervention(GestationalWeek::new(15), "hormone adjustment", 0.6);
        assert!(
            result.is_ok(),
            "Should allow with guardian + adequate benefit"
        );
    }

    #[test]
    fn test_emerging_sentience_rejects_low_benefit() {
        let gate = EctogenesisEthicsGate::new(true, false);
        let result = gate.check_intervention(GestationalWeek::new(15), "cosmetic gene edit", 0.3);
        assert!(
            result.is_err(),
            "Should reject low benefit in emerging sentience"
        );
    }

    #[test]
    fn test_sentient_requires_all_approvals() {
        // No approvals
        let gate = EctogenesisEthicsGate::unapproved();
        let result = gate.check_intervention(GestationalWeek::new(30), "steroid injection", 0.9);
        assert!(result.is_err());

        // Guardian only
        let gate = EctogenesisEthicsGate::new(true, false);
        let result = gate.check_intervention(GestationalWeek::new(30), "steroid injection", 0.9);
        assert!(result.is_err(), "Should require ethics board too");
        assert!(result.unwrap_err().contains("Ethics board"));

        // Both
        let gate = EctogenesisEthicsGate::fully_approved();
        let result = gate.check_intervention(GestationalWeek::new(30), "steroid injection", 0.9);
        assert!(
            result.is_ok(),
            "Should allow with all approvals + high benefit"
        );
    }

    #[test]
    fn test_sentient_rejects_low_benefit() {
        let gate = EctogenesisEthicsGate::fully_approved();
        let result = gate.check_intervention(GestationalWeek::new(30), "elective procedure", 0.5);
        assert!(result.is_err(), "Sentient should require benefit >= 0.7");
    }

    #[test]
    fn test_consent_proxy_escalation_at_boundaries() {
        let gate = EctogenesisEthicsGate::unapproved();
        assert_eq!(
            gate.consent_proxy(GestationalWeek::new(4)),
            ConsentProxy::PreSentient
        );
        assert_eq!(
            gate.consent_proxy(GestationalWeek::new(5)),
            ConsentProxy::EmergingSentience
        );
        assert_eq!(
            gate.consent_proxy(GestationalWeek::new(23)),
            ConsentProxy::EmergingSentience
        );
        assert_eq!(
            gate.consent_proxy(GestationalWeek::new(24)),
            ConsentProxy::Sentient
        );
    }

    #[test]
    fn test_minimum_benefit_increases() {
        let gate = EctogenesisEthicsGate::unapproved();
        let min_pre = gate.minimum_benefit(GestationalWeek::new(2));
        let min_em = gate.minimum_benefit(GestationalWeek::new(15));
        let min_sent = gate.minimum_benefit(GestationalWeek::new(30));
        assert!(min_pre < min_em);
        assert!(min_em < min_sent);
    }

    #[test]
    fn test_has_required_approvals() {
        let gate = EctogenesisEthicsGate::new(true, false);
        assert!(gate.has_required_approvals(GestationalWeek::new(2)));
        assert!(gate.has_required_approvals(GestationalWeek::new(15)));
        assert!(!gate.has_required_approvals(GestationalWeek::new(30)));
    }

    #[test]
    fn test_fully_approved_gate() {
        let gate = EctogenesisEthicsGate::fully_approved();
        assert!(gate.has_required_approvals(GestationalWeek::new(40)));
    }
}
