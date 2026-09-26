// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed auditing for a set of named Lean theorems in one source file.
//!
//! Existing `#print axioms` lines are stripped from the exact source bytes;
//! each requested theorem then receives one safe probe in a create-new temporary
//! file and is sent through `axiom_gate::audit_lean_file`, which invokes real
//! Lean and applies `symthaea-proof-audit` policy/spec checks.
//!
//! The declaration extractor intentionally supports only simple
//! `theorem name ... := ...` subjects. Unsupported or ambiguous syntax fails
//! closed; this module is not a general Lean parser.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::axiom_gate::{AxiomPolicy, ProofAuditOutcome, audit_lean_file, with_axiom_probe};

static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TheoremAuditCase<'a> {
    pub theorem: &'a str,
    pub expected_statement: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TheoremAuditResult {
    pub theorem: String,
    pub observed_statement: Option<String>,
    pub outcome: ProofAuditOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TheoremSetAuditReport {
    pub results: Vec<TheoremAuditResult>,
    pub setup_error: Option<String>,
    pub cleanup_errors: Vec<String>,
}

impl TheoremSetAuditReport {
    pub fn accepted(&self) -> bool {
        self.setup_error.is_none()
            && self.cleanup_errors.is_empty()
            && !self.results.is_empty()
            && self.results.iter().all(|result| {
                matches!(&result.outcome, ProofAuditOutcome::Audited(report) if report.accepted())
            })
    }
}

pub fn strip_axiom_probe_lines(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        if line.trim_start().starts_with("#print axioms ") {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn lean_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '\''
}

/// Extract the declaration tail for exactly one simple theorem declaration.
pub fn extract_simple_theorem_statement(source: &str, theorem: &str) -> Result<String, String> {
    let short_name = theorem.rsplit('.').next().unwrap_or(theorem);
    if short_name.is_empty() || !short_name.chars().all(lean_ident_char) {
        return Err("theorem short name is not a safe Lean identifier".into());
    }

    let needle = format!("theorem {short_name}");
    let starts: Vec<usize> = source
        .match_indices(&needle)
        .filter_map(|(idx, _)| {
            let line_start = source[..idx].rfind('\n').map_or(0, |p| p + 1);
            if !source[line_start..idx].trim().is_empty() {
                return None;
            }
            let after = idx + needle.len();
            if source[after..].chars().next().is_some_and(lean_ident_char) {
                return None;
            }
            Some(idx)
        })
        .collect();

    if starts.len() != 1 {
        return Err(format!(
            "expected exactly one simple theorem declaration for {theorem}, found {}",
            starts.len()
        ));
    }

    let after_name = &source[starts[0] + needle.len()..];
    let Some(end) = after_name.find(":=") else {
        return Err(format!("theorem {theorem} has no supported ':=' terminator"));
    };
    let statement = after_name[..end].trim();
    if statement.is_empty() || !statement.contains(':') {
        return Err(format!("theorem {theorem} has an empty/malformed statement"));
    }
    Ok(statement.to_string())
}

fn temp_component(theorem: &str) -> String {
    theorem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

pub fn audit_lean_theorem_set<P: AsRef<Path>>(
    source_path: P,
    cases: &[TheoremAuditCase<'_>],
    policy: &AxiomPolicy,
) -> TheoremSetAuditReport {
    let source_path = source_path.as_ref();
    let source = match fs::read_to_string(source_path) {
        Ok(source) => source,
        Err(error) => {
            return TheoremSetAuditReport {
                results: Vec::new(),
                setup_error: Some(format!("read {}: {error}", source_path.display())),
                cleanup_errors: Vec::new(),
            };
        }
    };

    let base = strip_axiom_probe_lines(&source);
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "symthaea_lean_theorem_audit_{}_{}",
        std::process::id(), nonce
    ));
    if let Err(error) = fs::create_dir(&dir) {
        return TheoremSetAuditReport {
            results: Vec::new(),
            setup_error: Some(format!("create audit temp dir {}: {error}", dir.display())),
            cleanup_errors: Vec::new(),
        };
    }

    let mut results = Vec::with_capacity(cases.len());
    let mut cleanup_errors = Vec::new();

    for case in cases {
        let observed_statement = match extract_simple_theorem_statement(&source, case.theorem) {
            Ok(statement) => statement,
            Err(error) => {
                results.push(TheoremAuditResult {
                    theorem: case.theorem.to_string(),
                    observed_statement: None,
                    outcome: ProofAuditOutcome::ProcessError(format!(
                        "source statement extraction failed: {error}"
                    )),
                });
                continue;
            }
        };

        let probed = match with_axiom_probe(&base, case.theorem) {
            Ok(script) => script,
            Err(error) => {
                results.push(TheoremAuditResult {
                    theorem: case.theorem.to_string(),
                    observed_statement: Some(observed_statement),
                    outcome: ProofAuditOutcome::ProcessError(format!(
                        "axiom probe construction failed: {error}"
                    )),
                });
                continue;
            }
        };

        let path = dir.join(format!("{}.lean", temp_component(case.theorem)));
        let write_result = (|| -> Result<(), String> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|error| format!("create {}: {error}", path.display()))?;
            file.write_all(probed.as_bytes())
                .map_err(|error| format!("write {}: {error}", path.display()))?;
            file.flush()
                .map_err(|error| format!("flush {}: {error}", path.display()))?;
            Ok(())
        })();

        let outcome = match write_result {
            Ok(()) => audit_lean_file(
                &path,
                case.theorem,
                &observed_statement,
                case.expected_statement,
                policy,
            ),
            Err(error) => ProofAuditOutcome::ProcessError(error),
        };

        results.push(TheoremAuditResult {
            theorem: case.theorem.to_string(),
            observed_statement: Some(observed_statement),
            outcome,
        });

        if path.exists() {
            if let Err(error) = fs::remove_file(&path) {
                cleanup_errors.push(format!("remove {}: {error}", path.display()));
            }
        }
    }

    if let Err(error) = fs::remove_dir(&dir) {
        cleanup_errors.push(format!("remove temp dir {}: {error}", dir.display()));
    }

    TheoremSetAuditReport {
        results,
        setup_error: None,
        cleanup_errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_only_axiom_probe_lines() {
        let src = "theorem t : True := by trivial\n#print axioms t\n-- #print axioms comment\ntheorem u : True := by trivial\n";
        let stripped = strip_axiom_probe_lines(src);
        assert!(stripped.contains("theorem t"));
        assert!(stripped.contains("theorem u"));
        assert!(stripped.contains("-- #print axioms comment"));
        assert!(!stripped.lines().any(|line| line.trim_start() == "#print axioms t"));
    }

    #[test]
    fn extracts_multiline_simple_statement() {
        let src = "namespace N\ntheorem t (a : Bool) :\n  Bool.xor a false = a := by\n  cases a <;> rfl\nend N\n";
        let statement = extract_simple_theorem_statement(src, "N.t").unwrap();
        assert_eq!(
            symthaea_proof_audit::normalize_statement(&statement),
            "(a : Bool) : Bool.xor a false = a"
        );
    }

    #[test]
    fn longer_identifier_does_not_match_short_theorem_name() {
        let src = "theorem tExtra : True := by trivial\n";
        assert!(extract_simple_theorem_statement(src, "t").is_err());
    }

    #[test]
    fn duplicate_declaration_fails_closed() {
        let src = "theorem t : True := by trivial\ntheorem t : True := by trivial\n";
        assert!(extract_simple_theorem_statement(src, "t").is_err());
    }

    #[test]
    fn empty_report_is_not_accepted() {
        let report = TheoremSetAuditReport {
            results: Vec::new(),
            setup_error: None,
            cleanup_errors: Vec::new(),
        };
        assert!(!report.accepted());
    }
}
