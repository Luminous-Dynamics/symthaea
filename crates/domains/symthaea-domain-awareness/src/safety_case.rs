// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Formal safety obligations for evidence-first domain awareness.
//!
//! The canonical obligation set lives in `symthaea-formal-safety` so engineering,
//! deployment, and domain-awareness consumers share one reviewed source of truth.

use symthaea_formal_safety::{SafetyCase, SafetyCaseTemplate};

/// Construct the canonical fail-closed safety case for a domain-awareness deployment.
///
/// This helper intentionally adds no local obligations. Any change to the baseline
/// epistemic/safety contract must be made in the shared `DomainAwareness` template,
/// preventing duplicate safety policies from drifting apart.
pub fn domain_awareness_safety_case(subject: impl Into<String>) -> SafetyCase {
    SafetyCase::from_template(subject, SafetyCaseTemplate::DomainAwareness)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_case_is_substantial_and_fail_closed_until_evidence_exists() {
        let case = domain_awareness_safety_case("harbor-awareness-node");
        assert!(case.obligations.len() >= 18);
        assert!(!case.is_discharged());
    }

    #[test]
    fn authority_separation_is_an_explicit_shared_obligation() {
        let case = domain_awareness_safety_case("airspace-awareness-node");
        assert!(case.obligations.iter().any(|obligation| {
            obligation
                .claim
                .contains("independent fail-closed safety and authorization boundary")
        }));
    }

    #[test]
    fn rf_silence_and_abstention_are_explicit_shared_obligations() {
        let case = domain_awareness_safety_case("passive-awareness-node");
        assert!(case
            .obligations
            .iter()
            .any(|obligation| obligation.claim.contains("passive radio silence")));
        assert!(case
            .obligations
            .iter()
            .any(|obligation| obligation.claim.contains("explicit abstention path")));
    }
}
