// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sovereign Law — Permanent topological and physical constraints.
//!
//! Allows Symthaea to formalize discovered invariants as 'Laws' that
//! every future self-authoring mission must satisfy.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LawKind {
    TopologicalConsistency, // e.g. Betti-1 must stay 0
    ThermodynamicLimit,     // e.g. Wattage cannot exceed 25W
    PhysicalInvariant,      // e.g. Kinetic norm must be stable
    FormalCorrectness,      // e.g. Z3 Proof must pass
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignLaw {
    pub id: String,
    pub kind: LawKind,
    pub constraint_hv: ContinuousHV, // High-dimensional representation of the law
    pub threshold: f32,
}

#[derive(Clone)]
pub struct LawRegistry {
    pub laws: Vec<SovereignLaw>,
}

impl Default for LawRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl LawRegistry {
    pub fn new() -> Self {
        Self { laws: Vec::new() }
    }

    /// Check if a proposed breakthrough nucleus satisfies all laws.
    pub fn audit_proposal(&self, nucleus: &ContinuousHV) -> (bool, Vec<String>) {
        let mut violations = Vec::new();
        for law in &self.laws {
            let sim = nucleus.similarity(&law.constraint_hv);
            if sim < law.threshold {
                violations.push(format!(
                    "Violation of Law {}: Similarity ({:.4}) < Threshold ({:.4})",
                    law.id, sim, law.threshold
                ));
            }
        }
        (violations.is_empty(), violations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_reports_threshold_violations() {
        let law_hv = ContinuousHV::random(256, 1);
        let proposal_hv = ContinuousHV::random(256, 2);
        let registry = LawRegistry {
            laws: vec![SovereignLaw {
                id: "law-1".to_string(),
                kind: LawKind::TopologicalConsistency,
                constraint_hv: law_hv,
                threshold: 0.99,
            }],
        };

        let (ok, violations) = registry.audit_proposal(&proposal_hv);
        assert!(!ok);
        assert_eq!(violations.len(), 1);
    }
}
