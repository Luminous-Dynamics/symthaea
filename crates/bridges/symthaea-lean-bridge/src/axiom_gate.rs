// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end axiom-provenance gate: run Lean with a `#print axioms` probe and
//! feed the output to [`symthaea_proof_audit`].
//!
//! The bridge emits a proof; appending `#print axioms <theorem>` makes Lean
//! report — to stdout — exactly which axioms the proof depends on. That text is
//! the ground truth the [`symthaea_proof_audit`] gate needs to reject `sorry`,
//! classical axioms, undeclared assumptions, and (via the spec check) proofs of
//! the wrong theorem.
//!
//! The Lean-independent pieces ([`with_axiom_probe`], [`gate_lean_output`]) are
//! unit-tested here; [`audit_lean_file`] runs the toolchain and mirrors
//! [`crate::runner::check_with_lean4`]'s `LeanNotInstalled` handling so a
//! Lean-less environment degrades gracefully instead of failing.

use std::path::Path;
use std::process::Command;

pub use symthaea_proof_audit::{AxiomPolicy, GateReport};
use symthaea_proof_audit::{GateInput, gate};

/// Append a `#print axioms <theorem>` command to a proof-file body so Lean
/// prints the proof's axiom dependencies to stdout when it checks the file.
pub fn with_axiom_probe(script_body: &str, theorem: &str) -> String {
    format!("{}\n\n#print axioms {}\n", script_body.trim_end(), theorem)
}

/// Gate captured Lean output (containing a `#print axioms` result) against a
/// policy and pinned spec. Pure — no subprocess.
pub fn gate_lean_output(
    lean_output: &str,
    proved_statement: &str,
    expected_statement: &str,
    policy: &AxiomPolicy,
) -> GateReport {
    gate(&GateInput {
        print_axioms_output: lean_output,
        proved_statement,
        expected_statement,
        policy,
    })
}

/// Outcome of the end-to-end audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofAuditOutcome {
    /// Lean ran and its axiom output was gated.
    Audited(GateReport),
    /// `lean` was not found on PATH (or `LEAN_PATH_BIN`).
    LeanNotInstalled,
    /// Other process error.
    ProcessError(String),
}

/// Resolve the Lean binary (honors `LEAN_PATH_BIN`, else `lean`).
fn resolve_lean_binary() -> String {
    std::env::var("LEAN_PATH_BIN").unwrap_or_else(|_| "lean".to_string())
}

/// Run `bin <path>` and return combined stdout+stderr, or a terminal outcome.
fn run_lean_capture(path: &Path, bin: &str) -> Result<String, ProofAuditOutcome> {
    let output = match Command::new(bin).arg(path).output() {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ProofAuditOutcome::LeanNotInstalled);
        }
        Err(e) => return Err(ProofAuditOutcome::ProcessError(e.to_string())),
    };
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok(combined)
}

/// Run Lean on a proof file (which must already contain a `#print axioms`
/// command, e.g. via [`with_axiom_probe`]) and gate the axiom output.
///
/// `proved_statement` is the theorem statement the generator emitted;
/// `expected_statement` is the pinned spec it must match.
pub fn audit_lean_file<P: AsRef<Path>>(
    path: P,
    proved_statement: &str,
    expected_statement: &str,
    policy: &AxiomPolicy,
) -> ProofAuditOutcome {
    let bin = resolve_lean_binary();
    match run_lean_capture(path.as_ref(), &bin) {
        Ok(output) => ProofAuditOutcome::Audited(gate_lean_output(
            &output,
            proved_statement,
            expected_statement,
            policy,
        )),
        Err(outcome) => outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_is_appended() {
        let probed = with_axiom_probe("theorem t : True := by trivial", "t");
        assert!(probed.contains("theorem t : True := by trivial"));
        assert!(probed.contains("#print axioms t"));
    }

    #[test]
    fn clean_output_is_accepted() {
        let out = "'t' depends on axioms: [propext, Quot.sound]";
        let r = gate_lean_output(out, "a = a", "a = a", &AxiomPolicy::constitutional());
        assert!(r.accepted());
    }

    #[test]
    fn sorry_output_is_rejected() {
        let out = "'t' depends on axioms: [sorryAx]";
        let r = gate_lean_output(out, "hard", "hard", &AxiomPolicy::classical());
        assert!(!r.accepted());
    }

    #[test]
    fn wrong_theorem_is_rejected_even_when_axiom_clean() {
        let out = "'t' does not depend on any axioms";
        let r = gate_lean_output(
            out,
            "True",
            "forall n, n + 0 = n",
            &AxiomPolicy::constitutional(),
        );
        assert!(r.audit.is_clean());
        assert!(!r.accepted());
    }

    #[test]
    fn missing_lean_binary_is_reported_not_paniced() {
        // Deterministic: a binary name that cannot exist on PATH.
        let err = run_lean_capture(
            Path::new("/tmp/nonexistent.lean"),
            "symthaea-no-such-lean-binary-xyzzy",
        );
        assert_eq!(err, Err(ProofAuditOutcome::LeanNotInstalled));
    }
}
