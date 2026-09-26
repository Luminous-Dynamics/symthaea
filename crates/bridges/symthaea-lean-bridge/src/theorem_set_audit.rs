// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed auditing for named Lean theorem sets.
//!
//! Existing `#print axioms` lines are stripped from the exact source bytes;
//! each requested theorem then receives one safe probe in a create-new temporary
//! file and is sent through `axiom_gate::audit_lean_file`, which invokes real
//! Lean and applies `symthaea-proof-audit` policy/spec checks.
//!
//! A theorem subject may also depend on exact prelude files. In that case the
//! adapter composes the stripped preludes in caller-supplied order before the
//! theorem source, while extracting the theorem statement only from the theorem
//! source itself. This supports exact parent -> child formal lineages without
//! teaching workflows to concatenate proof files or parse Lean output.
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

fn setup_error(message: String) -> TheoremSetAuditReport {
    TheoremSetAuditReport {
        results: Vec::new(),
        setup_error: Some(message),
        cleanup_errors: Vec::new(),
    }
}

fn read_source(path: &Path) -> Result<String, TheoremSetAuditReport> {
    fs::read_to_string(path).map_err(|error| setup_error(format!("read {}: {error}", path.display())))
}

fn compose_stripped_sources(preludes: &[String], theorem_source: &str) -> String {
    let mut combined = String::new();
    for prelude in preludes {
        combined.push_str(&strip_axiom_probe_lines(prelude));
        if !combined.ends_with('\n') {
            combined.push('\n');
        }
    }
    combined.push_str(&strip_axiom_probe_lines(theorem_source));
    combined
}

fn audit_source_text(
    theorem_source: &str,
    base: &str,
    cases: &[TheoremAuditCase<'_>],
    policy: &AxiomPolicy,
) -> TheoremSetAuditReport {
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "symthaea_lean_theorem_audit_{}_{}",
        std::process::id(), nonce
    ));
    if let Err(error) = fs::create_dir(&dir) {
        return setup_error(format!("create audit temp dir {}: {error}", dir.display()));
    }

    let mut results = Vec::with_capacity(cases.len());
    let mut cleanup_errors = Vec::new();

    for case in cases {
        let observed_statement = match extract_simple_theorem_statement(theorem_source, case.theorem) {
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

        let probed = match with_axiom_probe(base, case.theorem) {
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

/// Audit a self-contained Lean theorem source file.
pub fn audit_lean_theorem_set<P: AsRef<Path>>(
    source_path: P,
    cases: &[TheoremAuditCase<'_>],
    policy: &AxiomPolicy,
) -> TheoremSetAuditReport {
    audit_composed_lean_theorem_set(&[], source_path.as_ref(), cases, policy)
}

/// Audit a theorem source after exact prelude files have been prepended.
///
/// Prelude order is semantic and is preserved exactly. Existing axiom probes
/// are stripped from every fragment before composition. The observed theorem
/// statement is extracted only from `theorem_source_path`, so a theorem with
/// the same short name in a prelude cannot satisfy or confuse the target spec.
pub fn audit_composed_lean_theorem_set(
    prelude_paths: &[&Path],
    theorem_source_path: &Path,
    cases: &[TheoremAuditCase<'_>],
    policy: &AxiomPolicy,
) -> TheoremSetAuditReport {
    let theorem_source = match read_source(theorem_source_path) {
        Ok(source) => source,
        Err(report) => return report,
    };

    let mut preludes = Vec::with_capacity(prelude_paths.len());
    for path in prelude_paths {
        match read_source(path) {
            Ok(source) => preludes.push(source),
            Err(report) => return report,
        }
    }

    let base = compose_stripped_sources(&preludes, &theorem_source);
    audit_source_text(&theorem_source, &base, cases, policy)
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
    fn composed_sources_preserve_order_and_strip_all_probes() {
        let preludes = vec![
            "def parentA : Bool := true\n#print axioms parentA\n".to_string(),
            "def parentB : Bool := false\n#print axioms parentB\n".to_string(),
        ];
        let theorem_source =
            "theorem child : parentA = true := by rfl\n#print axioms child\n";
        let composed = compose_stripped_sources(&preludes, theorem_source);
        let a = composed.find("def parentA").unwrap();
        let b = composed.find("def parentB").unwrap();
        let child = composed.find("theorem child").unwrap();
        assert!(a < b && b < child);
        assert!(!composed.contains("#print axioms"));
    }

    #[test]
    fn target_statement_extraction_does_not_search_preludes() {
        let prelude = "theorem target : False := by trivial\n";
        let theorem_source = "theorem target : True := by trivial\n";
        let composed = compose_stripped_sources(&[prelude.to_string()], theorem_source);
        assert!(composed.matches("theorem target").count() == 2);
        assert_eq!(
            extract_simple_theorem_statement(theorem_source, "target").unwrap(),
            ": True"
        );
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
