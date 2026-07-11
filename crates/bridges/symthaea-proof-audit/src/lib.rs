// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-proof-audit
//!
//! An **axiom-provenance and spec-conformance gate** for machine-generated
//! proofs. It enforces that a proof reduces to a *declared axiom base* — no
//! `sorry`, no classical axioms where only constructive reasoning is allowed,
//! and no undeclared structural assumptions — and that the theorem actually
//! proved is the one that was pinned.
//!
//! ## Why this exists (and what it is *not*)
//!
//! A self-authoring system that proves its own code correct will, left
//! unchecked, drift: its generator invents ad-hoc assumptions to force a proof
//! checker green, and the codebase's implicit foundations quietly change. This
//! gate is the antidote.
//!
//! This is the honest, automatable core of the "reverse mathematics" intuition —
//! *what does this proof actually assume?* — but it is **not** reverse
//! mathematics. Reverse math is a human research programme that calibrates the
//! logical strength of infinitary theorems against a fixed hierarchy of
//! subsystems of second-order arithmetic; there is no algorithm that returns a
//! theorem's minimal axioms. What *is* mechanical is reading the axioms a proof
//! depends on — Lean's `#print axioms` — and enforcing a policy on them. That is
//! what this crate does.
//!
//! ## Two checks
//!
//! 1. **Axiom provenance** ([`audit`]): parse `#print axioms` output and reject
//!    `sorryAx`, classical axioms (when constructive-only), and any axiom outside
//!    the declared base.
//! 2. **Spec conformance** ([`spec::statement_matches`]): the check axiom
//!    auditing misses — a clean proof of the *wrong* theorem (e.g. a vacuously
//!    true goal) is still a failure.
//!
//! ## Example
//!
//! ```
//! use symthaea_proof_audit::{gate, GateInput, AxiomPolicy};
//!
//! let report = gate(&GateInput {
//!     print_axioms_output: "'thm' depends on axioms: [propext, Quot.sound]",
//!     proved_statement: "a + b = b + a",
//!     expected_statement: "a + b = b + a",
//!     policy: &AxiomPolicy::constitutional(),
//! });
//! assert!(report.accepted());
//! ```

pub mod axioms;
pub mod policy;
pub mod spec;

pub use axioms::{AxiomReport, is_classical, is_sorry};
pub use policy::{AuditVerdict, AxiomPolicy, Violation, audit};
pub use spec::{normalize_statement, source_contains_sorry, statement_matches};

/// Everything the gate needs to decide accept/reject for one proof.
pub struct GateInput<'a> {
    /// Raw output of `#print axioms <name>` from Lean.
    pub print_axioms_output: &'a str,
    /// The statement the proof actually established.
    pub proved_statement: &'a str,
    /// The pinned/intended statement (the spec).
    pub expected_statement: &'a str,
    /// The axiom policy to enforce.
    pub policy: &'a AxiomPolicy,
}

/// The combined gate outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateReport {
    pub audit: AuditVerdict,
    pub spec_conforms: bool,
}

impl GateReport {
    /// Accepted iff the axiom audit is clean *and* the proved statement matches
    /// the pinned spec.
    pub fn accepted(&self) -> bool {
        self.audit.is_clean() && self.spec_conforms
    }
}

/// Run the full provenance + spec-conformance gate on one proof.
pub fn gate(input: &GateInput) -> GateReport {
    let report = AxiomReport::parse(input.print_axioms_output);
    let audit = audit(&report, input.policy);
    let spec_conforms = statement_matches(input.proved_statement, input.expected_statement);
    GateReport {
        audit,
        spec_conforms,
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn accepts_clean_conforming_proof() {
        let r = gate(&GateInput {
            print_axioms_output: "'gate_add_comm' depends on axioms: [propext, Quot.sound]",
            proved_statement: "a + b = b + a",
            expected_statement: "a + b = b + a",
            policy: &AxiomPolicy::constitutional(),
        });
        assert!(r.accepted());
    }

    #[test]
    fn rejects_undeclared_axiom() {
        // Generator smuggled in an ad-hoc axiom to force the checker green.
        let r = gate(&GateInput {
            print_axioms_output: "'thm' depends on axioms: [propext, MyConvenientAxiom]",
            proved_statement: "P x",
            expected_statement: "P x",
            policy: &AxiomPolicy::classical(),
        });
        assert!(!r.accepted());
        assert!(matches!(
            r.audit.violations.as_slice(),
            [Violation::UndeclaredAxiom(a)] if a == "MyConvenientAxiom"
        ));
    }

    #[test]
    fn rejects_sorry_even_with_matching_spec() {
        let r = gate(&GateInput {
            print_axioms_output: "'thm' depends on axioms: [sorryAx]",
            proved_statement: "hard_theorem",
            expected_statement: "hard_theorem",
            policy: &AxiomPolicy::classical(),
        });
        assert!(!r.accepted());
        assert_eq!(r.audit.violations, vec![Violation::Sorry]);
    }

    #[test]
    fn rejects_clean_proof_of_wrong_theorem() {
        // Axiom-clean, but it proved `True` instead of the pinned spec — the
        // vacuous-truth trap that axiom auditing alone cannot catch.
        let r = gate(&GateInput {
            print_axioms_output: "'thm' does not depend on any axioms",
            proved_statement: "True",
            expected_statement: "forall n, n + 0 = n",
            policy: &AxiomPolicy::constitutional(),
        });
        assert!(r.audit.is_clean());
        assert!(!r.spec_conforms);
        assert!(!r.accepted());
    }
}
