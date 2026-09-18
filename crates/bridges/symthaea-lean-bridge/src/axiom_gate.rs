// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end Lean axiom/provenance gates.
//!
//! The high-assurance path is [`audit_authenticated_lean_source`] or
//! [`audit_authenticated_lean_file`]. Those functions do not trust a caller to
//! describe what theorem Lean proved. Instead they append a fresh conformance
//! theorem whose type is the pinned expected statement and whose proof term is
//! the generated theorem itself. Lean must typecheck that bridge before an axiom
//! receipt can be accepted.
//!
//! The older [`audit_lean_file_legacy`] entry point remains for compatibility,
//! but its `proved_statement` argument is caller-supplied and therefore cannot
//! authenticate statement conformance across a hostile boundary.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub use symthaea_proof_audit::{AxiomPolicy, GateReport};
use symthaea_proof_audit::{GateInput, gate};

use crate::subprocess::{RunError, run_bounded_on_file};

/// The generated theorem whose successful typecheck authenticates statement
/// conformance against the pinned Lean expression.
pub const SPEC_CONFORMANCE_THEOREM: &str = "SymthaeaProofAudit.__spec_conformance";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn safe_theorem_name(theorem: &str) -> bool {
    !theorem.is_empty()
        && theorem.split('.').all(|segment| {
            !segment.is_empty()
                && segment
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '\'')
        })
}

/// Append a `#print axioms <theorem>` command to a proof body.
///
/// This authenticates the axiom report for that theorem name, but by itself it
/// does not authenticate a separately supplied textual description of the
/// theorem's type. Prefer [`with_authenticated_spec_probe`] for claim checking.
pub fn with_axiom_probe(script_body: &str, theorem: &str) -> Result<String, String> {
    if !safe_theorem_name(theorem) {
        return Err("theorem name is not a safe dotted Lean identifier".into());
    }
    Ok(format!(
        "{}\n\n#print axioms {}\n",
        script_body.trim_end(),
        theorem
    ))
}

/// Append a Lean-checked bridge from `theorem` to the pinned expected statement.
///
/// The generated declaration is equivalent in spirit to:
///
/// ```text
/// theorem SymthaeaProofAudit.__spec_conformance : (EXPECTED) := _root_.THEOREM
/// ```
///
/// If the original theorem has a different type, Lean rejects the file before
/// any proof receipt can be emitted. The expected statement is trusted
/// specification input and should be hash-bound by the caller to a frozen
/// challenge before invoking this function.
pub fn with_authenticated_spec_probe(
    script_body: &str,
    theorem: &str,
    expected_statement: &str,
) -> Result<String, String> {
    if !safe_theorem_name(theorem) {
        return Err("theorem name is not a safe dotted Lean identifier".into());
    }
    if expected_statement.trim().is_empty() {
        return Err("expected statement must not be empty".into());
    }

    Ok(format!(
        "{}\n\nnamespace SymthaeaProofAudit\n\
         theorem __spec_conformance : (\n{}\n) := (_root_.{})\n\
         end SymthaeaProofAudit\n\n\
         #print axioms {}\n",
        script_body.trim_end(),
        expected_statement.trim(),
        theorem,
        SPEC_CONFORMANCE_THEOREM,
    ))
}

/// Gate captured Lean output against a policy and caller-supplied statement
/// description. Pure; no subprocess.
///
/// This helper is suitable for policy unit tests but should not be treated as a
/// high-assurance statement-conformance boundary because `proved_statement` is
/// not authenticated by Lean here.
pub fn gate_lean_output(
    lean_output: &str,
    expected_theorem: &str,
    proved_statement: &str,
    expected_statement: &str,
    policy: &AxiomPolicy,
) -> GateReport {
    gate(&GateInput {
        print_axioms_output: lean_output,
        expected_theorem,
        proved_statement,
        expected_statement,
        policy,
    })
}

/// Gate output from [`with_authenticated_spec_probe`].
///
/// Statement conformance has already been established by Lean typechecking the
/// generated bridge theorem, so the same pinned statement is supplied on both
/// sides of the pure policy gate. The remaining output check authenticates the
/// conformance theorem name and its actual axiom dependencies.
pub fn gate_authenticated_lean_output(
    lean_output: &str,
    expected_statement: &str,
    policy: &AxiomPolicy,
) -> GateReport {
    gate_lean_output(
        lean_output,
        SPEC_CONFORMANCE_THEOREM,
        expected_statement,
        expected_statement,
        policy,
    )
}

/// Outcome of an end-to-end Lean audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofAuditOutcome {
    /// Lean ran and its captured axiom output was gated.
    Audited(GateReport),
    /// `lean` was not found on PATH (or `LEAN_PATH_BIN`).
    LeanNotInstalled,
    /// Other process/source-generation error.
    ProcessError(String),
}

fn resolve_lean_binary() -> String {
    std::env::var("LEAN_PATH_BIN").unwrap_or_else(|_| "lean".to_string())
}

fn run_lean_capture(path: &Path, bin: &str) -> Result<String, ProofAuditOutcome> {
    let output = match run_bounded_on_file(bin, path) {
        Ok(output) => output,
        Err(RunError::NotFound) => return Err(ProofAuditOutcome::LeanNotInstalled),
        Err(RunError::Io(error)) => return Err(ProofAuditOutcome::ProcessError(error)),
    };
    if output.timed_out {
        return Err(ProofAuditOutcome::ProcessError(
            "lean did not finish within the configured timeout and was killed".to_string(),
        ));
    }

    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    if output.truncated {
        combined.push_str("\n[... output truncated at the capture limit ...]");
    }
    if !output.success() {
        return Err(ProofAuditOutcome::ProcessError(format!(
            "Lean exited with failure: {combined}"
        )));
    }
    Ok(combined)
}

/// Compatibility path for callers that already generated a file containing a
/// `#print axioms` probe.
///
/// # Security limitation
///
/// `proved_statement` is caller-supplied text. Although the axiom output really
/// comes from Lean, this function cannot establish that the supplied string is
/// the theorem type Lean actually checked. New evidence-bearing code must use
/// [`audit_authenticated_lean_source`] or [`audit_authenticated_lean_file`].
#[deprecated(
    note = "caller-supplied proved_statement is not authenticated; use audit_authenticated_lean_source/file"
)]
pub fn audit_lean_file_legacy<P: AsRef<Path>>(
    path: P,
    expected_theorem: &str,
    proved_statement: &str,
    expected_statement: &str,
    policy: &AxiomPolicy,
) -> ProofAuditOutcome {
    let bin = resolve_lean_binary();
    match run_lean_capture(path.as_ref(), &bin) {
        Ok(output) => ProofAuditOutcome::Audited(gate_lean_output(
            &output,
            expected_theorem,
            proved_statement,
            expected_statement,
            policy,
        )),
        Err(outcome) => outcome,
    }
}

/// Backward-compatible alias for the historical low-level API.
///
/// This intentionally carries the same deprecation warning as
/// [`audit_lean_file_legacy`].
#[deprecated(
    note = "caller-supplied proved_statement is not authenticated; use audit_authenticated_lean_source/file"
)]
pub fn audit_lean_file<P: AsRef<Path>>(
    path: P,
    expected_theorem: &str,
    proved_statement: &str,
    expected_statement: &str,
    policy: &AxiomPolicy,
) -> ProofAuditOutcome {
    #[allow(deprecated)]
    audit_lean_file_legacy(
        path,
        expected_theorem,
        proved_statement,
        expected_statement,
        policy,
    )
}

struct TempLeanFile {
    path: PathBuf,
}

impl TempLeanFile {
    fn create(source: &str) -> Result<Self, ProofAuditOutcome> {
        let directory = std::env::temp_dir();
        let pid = std::process::id();

        for _ in 0..64 {
            let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!(
                "symthaea-proof-audit-{pid}-{sequence}.lean"
            ));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    if let Err(error) = file.write_all(source.as_bytes()) {
                        let _ = fs::remove_file(&path);
                        return Err(ProofAuditOutcome::ProcessError(format!(
                            "failed to write authenticated Lean source: {error}"
                        )));
                    }
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(ProofAuditOutcome::ProcessError(format!(
                        "failed to create authenticated Lean source: {error}"
                    )));
                }
            }
        }

        Err(ProofAuditOutcome::ProcessError(
            "failed to allocate unique temporary Lean source".into(),
        ))
    }
}

impl Drop for TempLeanFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// High-assurance verification path for an in-memory generated Lean proof.
///
/// The function itself constructs the conformance theorem, writes the resulting
/// source to a create-new temporary file, runs the real Lean subprocess, and
/// accepts only the axiom report for that generated conformance theorem.
pub fn audit_authenticated_lean_source(
    script_body: &str,
    theorem: &str,
    expected_statement: &str,
    policy: &AxiomPolicy,
) -> ProofAuditOutcome {
    let source = match with_authenticated_spec_probe(script_body, theorem, expected_statement) {
        Ok(source) => source,
        Err(error) => return ProofAuditOutcome::ProcessError(error),
    };
    let temp = match TempLeanFile::create(&source) {
        Ok(temp) => temp,
        Err(outcome) => return outcome,
    };
    let bin = resolve_lean_binary();
    match run_lean_capture(&temp.path, &bin) {
        Ok(output) => ProofAuditOutcome::Audited(gate_authenticated_lean_output(
            &output,
            expected_statement,
            policy,
        )),
        Err(outcome) => outcome,
    }
}

/// High-assurance verification path for a generated Lean file.
///
/// The original source file is read as data; the verifier itself appends the
/// conformance theorem and probe in a separate temporary file. Callers therefore
/// cannot obtain statement authority merely by supplying a favorable
/// `proved_statement` string.
pub fn audit_authenticated_lean_file<P: AsRef<Path>>(
    path: P,
    theorem: &str,
    expected_statement: &str,
    policy: &AxiomPolicy,
) -> ProofAuditOutcome {
    let source = match fs::read_to_string(path.as_ref()) {
        Ok(source) => source,
        Err(error) => {
            return ProofAuditOutcome::ProcessError(format!(
                "failed to read Lean proof source: {error}"
            ));
        }
    };
    audit_authenticated_lean_source(&source, theorem, expected_statement, policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_axiom_probe_is_appended() {
        let probed = with_axiom_probe("theorem t : True := by trivial", "t").unwrap();
        assert!(probed.contains("theorem t : True := by trivial"));
        assert!(probed.contains("#print axioms t"));
    }

    #[test]
    fn authenticated_probe_binds_theorem_to_expected_type() {
        let probed = with_authenticated_spec_probe(
            "theorem t : True := by trivial",
            "t",
            "True",
        )
        .unwrap();
        assert!(probed.contains("theorem __spec_conformance"));
        assert!(probed.contains(": (\nTrue\n) := (_root_.t)"));
        assert!(probed.contains(&format!("#print axioms {SPEC_CONFORMANCE_THEOREM}")));
    }

    #[test]
    fn authenticated_probe_rejects_empty_spec_and_unsafe_name() {
        assert!(with_authenticated_spec_probe("theorem t : True := by trivial", "t", " ").is_err());
        assert!(
            with_authenticated_spec_probe(
                "theorem t : True := by trivial",
                "t\naxiom injected : False",
                "True",
            )
            .is_err()
        );
    }

    #[test]
    fn authenticated_output_accepts_only_conformance_theorem_report() {
        let output = format!(
            "'{}' depends on axioms: [propext, Quot.sound]",
            SPEC_CONFORMANCE_THEOREM
        );
        let report = gate_authenticated_lean_output(
            &output,
            "forall a, a = a",
            &AxiomPolicy::constitutional(),
        );
        assert!(report.accepted());

        let wrong = "'t' depends on axioms: [propext, Quot.sound]";
        let report = gate_authenticated_lean_output(
            wrong,
            "forall a, a = a",
            &AxiomPolicy::constitutional(),
        );
        assert!(!report.accepted());
        assert!(report.evidence_error.is_some());
    }

    #[test]
    fn sorry_output_is_rejected() {
        let output = format!("'{}' depends on axioms: [sorryAx]", SPEC_CONFORMANCE_THEOREM);
        let report = gate_authenticated_lean_output(
            &output,
            "hard_theorem",
            &AxiomPolicy::classical(),
        );
        assert!(!report.accepted());
    }

    #[test]
    fn legacy_pure_gate_still_rejects_wrong_declared_statement() {
        let output = "'t' does not depend on any axioms";
        let report = gate_lean_output(
            output,
            "t",
            "True",
            "forall n, n + 0 = n",
            &AxiomPolicy::constitutional(),
        );
        assert!(report.audit.is_clean());
        assert!(!report.accepted());
    }

    #[test]
    fn missing_lean_binary_is_reported_not_panicked() {
        let error = run_lean_capture(
            Path::new("/tmp/nonexistent.lean"),
            "symthaea-no-such-lean-binary-xyzzy",
        );
        assert_eq!(error, Err(ProofAuditOutcome::LeanNotInstalled));
    }

    #[test]
    fn malformed_output_fails_closed() {
        let report = gate_authenticated_lean_output(
            "Lean compilation failed",
            "P",
            &AxiomPolicy::constitutional(),
        );
        assert!(!report.accepted());
        assert!(report.evidence_error.is_some());
    }

    #[test]
    fn temp_source_is_removed_on_drop() {
        let path = {
            let temp = TempLeanFile::create("theorem t : True := by trivial").unwrap();
            let path = temp.path.clone();
            assert!(path.exists());
            path
        };
        assert!(!path.exists());
    }
}
