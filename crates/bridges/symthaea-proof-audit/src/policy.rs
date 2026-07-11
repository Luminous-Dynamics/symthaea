// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Axiom policy and the provenance audit.
//!
//! A [`AxiomPolicy`] declares the "constitutional base" — the axioms a proof is
//! permitted to reduce to. [`audit`] checks an [`AxiomReport`] against it and
//! returns every violation: an unproved proof (`sorry`), a classical axiom where
//! only constructive reasoning is allowed, or any axiom outside the declared
//! base (the "undeclared structural assumption" a self-authoring generator must
//! not silently introduce).

use crate::axioms::{AxiomReport, is_classical, is_sorry};
use std::collections::BTreeSet;

/// The set of axioms a proof is allowed to depend on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxiomPolicy {
    allowed: BTreeSet<String>,
    /// Whether classical (choice-based) axioms are permitted.
    allow_classical: bool,
}

impl AxiomPolicy {
    /// Constructive/constitutional base: only Lean's `propext` and `Quot.sound`
    /// (used even by constructive proofs); classical choice is rejected. This is
    /// the strictest sensible base — a proof passing it is fully constructive
    /// modulo propositional extensionality and quotient soundness.
    pub fn constitutional() -> AxiomPolicy {
        AxiomPolicy {
            allowed: ["propext", "Quot.sound"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            allow_classical: false,
        }
    }

    /// Classical base: the constitutional base plus `Classical.choice`, matching
    /// standard Mathlib proofs.
    pub fn classical() -> AxiomPolicy {
        let mut p = AxiomPolicy::constitutional();
        p.allowed.insert("Classical.choice".to_string());
        p.allow_classical = true;
        p
    }

    /// Add an explicitly permitted axiom to the base.
    pub fn allow(mut self, axiom: &str) -> AxiomPolicy {
        self.allowed.insert(axiom.to_string());
        self
    }

    /// The permitted axioms.
    pub fn allowed(&self) -> &BTreeSet<String> {
        &self.allowed
    }
}

/// A single provenance violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    /// The proof depends on `sorryAx` — it is not actually proved.
    Sorry,
    /// A classical axiom was used where only constructive reasoning is allowed.
    ClassicalAxiom(String),
    /// An axiom outside the declared base — an undeclared structural assumption.
    UndeclaredAxiom(String),
}

/// The outcome of a provenance audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditVerdict {
    pub theorem: String,
    pub violations: Vec<Violation>,
}

impl AuditVerdict {
    /// True iff the proof reduces cleanly to the declared base.
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Audit a proof's axiom dependencies against a policy.
pub fn audit(report: &AxiomReport, policy: &AxiomPolicy) -> AuditVerdict {
    let mut violations = Vec::new();
    for axiom in &report.axioms {
        if is_sorry(axiom) {
            violations.push(Violation::Sorry);
            continue;
        }
        if is_classical(axiom) && !policy.allow_classical {
            violations.push(Violation::ClassicalAxiom(axiom.clone()));
            continue;
        }
        if !policy.allowed().contains(axiom) {
            violations.push(Violation::UndeclaredAxiom(axiom.clone()));
        }
    }
    AuditVerdict {
        theorem: report.theorem.clone(),
        violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(axioms: &[&str]) -> AxiomReport {
        AxiomReport {
            theorem: "t".to_string(),
            axioms: axioms.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn clean_constructive_proof_passes_constitutional() {
        let v = audit(
            &report(&["propext", "Quot.sound"]),
            &AxiomPolicy::constitutional(),
        );
        assert!(v.is_clean());
    }

    #[test]
    fn axiom_free_proof_passes() {
        assert!(audit(&report(&[]), &AxiomPolicy::constitutional()).is_clean());
    }

    #[test]
    fn sorry_is_rejected() {
        let v = audit(&report(&["sorryAx"]), &AxiomPolicy::constitutional());
        assert!(!v.is_clean());
        assert_eq!(v.violations, vec![Violation::Sorry]);
    }

    #[test]
    fn classical_rejected_by_constitutional_but_allowed_by_classical() {
        let r = report(&["propext", "Classical.choice"]);
        assert_eq!(
            audit(&r, &AxiomPolicy::constitutional()).violations,
            vec![Violation::ClassicalAxiom("Classical.choice".to_string())]
        );
        assert!(audit(&r, &AxiomPolicy::classical()).is_clean());
    }

    #[test]
    fn undeclared_axiom_is_rejected() {
        let v = audit(
            &report(&["propext", "MyAdHocAxiom"]),
            &AxiomPolicy::classical(),
        );
        assert_eq!(
            v.violations,
            vec![Violation::UndeclaredAxiom("MyAdHocAxiom".to_string())]
        );
        // ...unless explicitly allowed.
        let relaxed = AxiomPolicy::classical().allow("MyAdHocAxiom");
        assert!(audit(&report(&["propext", "MyAdHocAxiom"]), &relaxed).is_clean());
    }

    #[test]
    fn multiple_violations_reported() {
        let v = audit(
            &report(&["sorryAx", "Classical.choice", "Weird"]),
            &AxiomPolicy::constitutional(),
        );
        assert_eq!(v.violations.len(), 3);
    }
}
