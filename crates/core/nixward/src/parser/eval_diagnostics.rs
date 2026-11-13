// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic diagnostics for Nix evaluation and rebuild stderr.
//!
//! This module is deliberately non-executing: it does not invoke Nix, read host
//! files, edit a flake, or authorize a repair. It converts captured diagnostic
//! text into typed findings with source-line evidence so higher layers can
//! reason over facts without treating model-generated explanations as proof.
//!
//! The first rules target high-value, easy-to-confuse cases:
//! - removed/renamed nixpkgs attributes (fatal evaluation findings);
//! - module option, assertion, type, and fixed-output hash failures;
//! - binary-cache authentication/availability warnings (non-fatal by default).
//!
//! Unknown errors are retained as unknown findings rather than guessed.

use serde::{Deserialize, Serialize};

/// Stable machine-readable diagnostic classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationIssueKind {
    RemovedPackage,
    RenamedPackage,
    UnknownOption,
    ModuleAssertion,
    ModuleTypeConflict,
    HashMismatch,
    CacheAuthorization,
    CacheUnavailable,
    UnknownEvaluationError,
}

/// Diagnostic severity. Cache transport warnings must not mask a fatal evaluator
/// error and must never be upgraded to a fatal rebuild cause by this parser alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSeverity {
    Error,
    Warning,
}

/// A source location parsed from a Nix stack frame (one-based line/column).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationSourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
}

/// One-based line evidence from the captured stderr stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationEvidence {
    pub line_number: usize,
    pub text: String,
}

/// A single deterministic finding. Recommendations are review prompts, not
/// executable patches and not evidence that a suggested repair is correct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationDiagnostic {
    pub kind: EvaluationIssueKind,
    pub severity: EvaluationSeverity,
    pub title: String,
    pub subject: Option<String>,
    pub explanation: String,
    pub source: Option<EvaluationSourceLocation>,
    pub evidence: Vec<EvaluationEvidence>,
    pub recommendations: Vec<String>,
}

/// Structured interpretation of one captured Nix stderr stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct NixEvaluationReport {
    pub diagnostics: Vec<EvaluationDiagnostic>,
    /// Index into diagnostics, if a fatal evaluator finding was recognized.
    /// Warnings such as cache HTTP 401 are deliberately excluded.
    pub primary_failure: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StackFrame {
    file: String,
    line: usize,
    column: usize,
    evidence_line: usize,
}

impl NixEvaluationReport {
    /// Analyze captured stderr without executing commands or accessing the host.
    ///
    /// The parser is conservative: a known pattern produces a typed finding;
    /// otherwise the first error line is retained as an unknown finding.
    pub fn analyze(stderr: &str) -> Self {
        let lines: Vec<&str> = stderr.lines().collect();
        let frames: Vec<StackFrame> = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| parse_stack_frame(line, index + 1))
            .collect();

        let mut diagnostics = Vec::new();

        // Prefer a specific fatal cause over a generic error fallback.
        if let Some((index, line)) = find_line(&lines, |lower| {
            lower.contains("has been removed from nixpkgs")
        }) {
            let subject = subject_before(line, " has been removed from nixpkgs");
            let source = relevant_source(&frames);
            let mut evidence = vec![evidence(index, line)];
            append_source_evidence(&mut evidence, &frames, source.as_ref());
            diagnostics.push(EvaluationDiagnostic {
                kind: EvaluationIssueKind::RemovedPackage,
                severity: EvaluationSeverity::Error,
                title: format!(
                    "Package attribute {} was removed from the pinned nixpkgs revision",
                    subject.as_deref().unwrap_or("<unknown>")
                ),
                subject,
                explanation: "The evaluator reached an attribute that exists as a deliberate throwing alias in the selected nixpkgs revision. The first relevant local Nix source frame is retained when present.".into(),
                source,
                evidence,
                recommendations: vec![
                    "Inspect the exact locked nixpkgs revision and upstream package history; prefer a narrowly reviewed nixpkgs pin over updating every flake input.".into(),
                    "Inspect overlays that dereference prev.<attribute>; preserve any runtime wrapper only after checking whether it is still needed by the restored package.".into(),
                    "Re-evaluate and build the candidate system closure before attempting activation.".into(),
                ],
            });
        } else if let Some((index, line)) = find_line(&lines, |lower| {
            lower.contains("has been renamed to/replaced by")
                || lower.contains("has been renamed to")
                || lower.contains("has been replaced by")
        }) {
            let lower = line.to_lowercase();
            let marker = if lower.contains("has been renamed to/replaced by") {
                " has been renamed to/replaced by"
            } else if lower.contains("has been renamed to") {
                " has been renamed to"
            } else {
                " has been replaced by"
            };
            let subject = subject_before(line, marker);
            let source = relevant_source(&frames);
            let mut evidence = vec![evidence(index, line)];
            append_source_evidence(&mut evidence, &frames, source.as_ref());
            diagnostics.push(EvaluationDiagnostic {
                kind: EvaluationIssueKind::RenamedPackage,
                severity: EvaluationSeverity::Error,
                title: format!(
                    "Package attribute {} was renamed or replaced",
                    subject.as_deref().unwrap_or("<unknown>")
                ),
                subject,
                explanation: "The evaluator encountered a nixpkgs alias that explicitly reports a replacement name.".into(),
                source,
                evidence,
                recommendations: vec![
                    "Use the replacement named by the error only after verifying it exists in the exact locked nixpkgs revision.".into(),
                    "Search overlays and package lists for every reference to the old attribute; evaluate the complete host configuration after the change.".into(),
                ],
            });
        } else if let Some((index, line)) = find_line(&lines, |lower| {
            lower.contains("the option")
                && (lower.contains("does not exist") || lower.contains("is not defined"))
        }) {
            let source = relevant_source(&frames);
            let mut evidence = vec![evidence(index, line)];
            append_source_evidence(&mut evidence, &frames, source.as_ref());
            diagnostics.push(EvaluationDiagnostic {
                kind: EvaluationIssueKind::UnknownOption,
                severity: EvaluationSeverity::Error,
                title: "NixOS module option is unknown".into(),
                subject: extract_backticked_subject(line),
                explanation: "The module evaluator reports that a referenced option is not declared in the loaded module set.".into(),
                source,
                evidence,
                recommendations: vec![
                    "Check the option name against the module set selected by the locked inputs; confirm the module that defines it is imported.".into(),
                    "Do not suppress the error by deleting the setting until its intended behavior and replacement are understood.".into(),
                ],
            });
        } else if let Some((index, line)) = find_line(&lines, |lower| {
            lower.contains("assertion")
                && (lower.contains("failed") || lower.contains("violated"))
        }) {
            let source = relevant_source(&frames);
            let mut evidence = vec![evidence(index, line)];
            append_source_evidence(&mut evidence, &frames, source.as_ref());
            diagnostics.push(EvaluationDiagnostic {
                kind: EvaluationIssueKind::ModuleAssertion,
                severity: EvaluationSeverity::Error,
                title: "NixOS module assertion failed".into(),
                subject: None,
                explanation: "A module assertion rejected the merged configuration. The assertion text and nearest relevant source frame should guide diagnosis.".into(),
                source,
                evidence,
                recommendations: vec![
                    "Read the complete assertion message and inspect the option definitions that contribute to it.".into(),
                    "Prefer resolving the conflicting configuration intent over overriding or disabling the assertion.".into(),
                ],
            });
        } else if let Some((index, line)) = find_line(&lines, |lower| {
            lower.contains("conflicting definition")
                || lower.contains("definitions of")
                || lower.contains("not of type")
                || lower.contains("expected a value of type")
        }) {
            let source = relevant_source(&frames);
            let mut evidence = vec![evidence(index, line)];
            append_source_evidence(&mut evidence, &frames, source.as_ref());
            diagnostics.push(EvaluationDiagnostic {
                kind: EvaluationIssueKind::ModuleTypeConflict,
                severity: EvaluationSeverity::Error,
                title: "NixOS option definitions conflict or have the wrong type".into(),
                subject: extract_backticked_subject(line),
                explanation: "The module system could not merge the supplied option definitions into the declared option type.".into(),
                source,
                evidence,
                recommendations: vec![
                    "Inspect all definitions of the affected option, including imports, mkDefault/mkForce priorities, and type declarations.".into(),
                    "Keep the declared option type and module invariants intact; make the smallest definition-level correction.".into(),
                ],
            });
        } else if let Some((index, line)) = find_line(&lines, |lower| {
            lower.contains("hash mismatch")
                || (lower.contains("got: sha256-") && lower.contains("expected"))
        }) {
            let source = relevant_source(&frames);
            let mut evidence = vec![evidence(index, line)];
            append_source_evidence(&mut evidence, &frames, source.as_ref());
            diagnostics.push(EvaluationDiagnostic {
                kind: EvaluationIssueKind::HashMismatch,
                severity: EvaluationSeverity::Error,
                title: "Fixed-output derivation hash does not match".into(),
                subject: None,
                explanation: "The fetched or generated output differs from the declared content hash.".into(),
                source,
                evidence,
                recommendations: vec![
                    "Verify the source URL, version, and build inputs before updating a hash; do not blindly accept a newly printed hash.".into(),
                    "Retain the old hash and compare the fetched source and derivation inputs when investigating a reproducibility or supply-chain concern.".into(),
                ],
            });
        }

        // Cache transport/auth failures are separate warnings. They are not
        // promoted to the primary failure merely because they occur near an error.
        if let Some((index, line)) = find_line(&lines, |lower| {
            (lower.contains("401")
                || lower.contains("403")
                || lower.contains("unauthorized"))
                && (lower.contains("cache")
                    || lower.contains("substituter")
                    || lower.contains("cachix"))
        }) {
            diagnostics.push(EvaluationDiagnostic {
                kind: EvaluationIssueKind::CacheAuthorization,
                severity: EvaluationSeverity::Warning,
                title: "Binary cache returned an authorization error".into(),
                subject: extract_cache_host(line),
                explanation: "The cache endpoint denied access or requires credentials. This generally affects substitution availability, not Nix expression evaluation correctness.".into(),
                source: None,
                evidence: vec![evidence(index, line)],
                recommendations: vec![
                    "Verify whether the cache is intentionally private and whether the configured client has the required credentials.".into(),
                    "Keep substituter URLs and trusted public keys consistent; do not add an unverified key or disable signature checks to silence the warning.".into(),
                    "Treat a cache warning separately from the first fatal evaluator error.".into(),
                ],
            });
        } else if let Some((index, line)) = find_line(&lines, |lower| {
            (lower.contains("cache does not exist")
                || (lower.contains("substituter") && lower.contains("not found"))
                || (lower.contains("unable to download") && lower.contains("cache-info")))
                && !lower.contains("401")
                && !lower.contains("403")
        }) {
            diagnostics.push(EvaluationDiagnostic {
                kind: EvaluationIssueKind::CacheUnavailable,
                severity: EvaluationSeverity::Warning,
                title: "Binary cache is unavailable".into(),
                subject: extract_cache_host(line),
                explanation: "Nix could not obtain cache metadata or reach the configured substituter. This is a cache availability issue and should not be confused with a configuration evaluation failure.".into(),
                source: None,
                evidence: vec![evidence(index, line)],
                recommendations: vec![
                    "Check the substituter hostname and service availability; allow Nix to use the other configured caches or build locally.".into(),
                    "Do not change package selections or flake pins solely to address a cache transport warning.".into(),
                ],
            });
        }

        if diagnostics
            .iter()
            .all(|diagnostic| diagnostic.severity != EvaluationSeverity::Error)
        {
            if let Some((index, line)) =
                find_line(&lines, |lower| lower.starts_with("error:"))
            {
                let source = relevant_source(&frames);
                let mut evidence = vec![evidence(index, line)];
                append_source_evidence(&mut evidence, &frames, source.as_ref());
                diagnostics.push(EvaluationDiagnostic {
                    kind: EvaluationIssueKind::UnknownEvaluationError,
                    severity: EvaluationSeverity::Error,
                    title: "Unclassified Nix evaluation error".into(),
                    subject: None,
                    explanation: "The captured error did not match a known deterministic rule. Preserve the trace and request further diagnosis rather than generating an ungrounded repair.".into(),
                    source,
                    evidence,
                    recommendations: vec![
                        "Capture the complete stderr stream with --show-trace and preserve the exact flake reference and lockfile.".into(),
                        "Do not apply an automated repair until the error is classified and a candidate passes read-only evaluation.".into(),
                    ],
                });
            }
        }

        let primary_failure = diagnostics
            .iter()
            .position(|diagnostic| diagnostic.severity == EvaluationSeverity::Error);

        Self {
            diagnostics,
            primary_failure,
        }
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == EvaluationSeverity::Error)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == EvaluationSeverity::Warning)
            .count()
    }

    pub fn primary(&self) -> Option<&EvaluationDiagnostic> {
        self.primary_failure
            .and_then(|index| self.diagnostics.get(index))
    }
}

fn find_line<'a>(
    lines: &'a [&str],
    predicate: impl Fn(&str) -> bool,
) -> Option<(usize, &'a str)> {
    lines
        .iter()
        .enumerate()
        .find(|(_, line)| predicate(&line.to_lowercase()))
        .map(|(index, line)| (index, *line))
}

fn evidence(index: usize, line: &str) -> EvaluationEvidence {
    EvaluationEvidence {
        line_number: index + 1,
        text: line.to_string(),
    }
}

fn subject_before(line: &str, marker: &str) -> Option<String> {
    let marker_index = line.to_lowercase().find(&marker.to_lowercase())?;
    let before_marker = &line[..marker_index];
    let without_prefix = before_marker.strip_prefix("error:").unwrap_or(before_marker);
    let subject = without_prefix
        .trim()
        .trim_matches(|character: char| {
            character == '\'' || character == '"' || character == char::from(96)
        });
    if subject.is_empty() {
        None
    } else {
        Some(subject.to_string())
    }
}

fn extract_backticked_subject(line: &str) -> Option<String> {
    // Nix diagnostics traditionally delimit option names as `option':
    // (opening backtick, closing apostrophe). Accept a symmetric backtick as
    // well for fixtures and other tooling that emits `option`.
    let opening = char::from(96);
    let start = line.find(opening)? + opening.len_utf8();
    let remainder = &line[start..];
    let end = remainder
        .find('\'')
        .or_else(|| remainder.find(opening))?;
    let subject = remainder[..end].trim();
    (!subject.is_empty()).then(|| subject.to_string())
}

fn extract_cache_host(line: &str) -> Option<String> {
    line.split_whitespace()
        .find(|part| part.contains(".cachix.org") || part.contains("cache."))
        .map(|part| part.trim_matches(|character: char| {
            character == '\'' || character == '"' || character == ',' || character == ':'
                || character == '(' || character == ')'
        }).to_string())
}

fn parse_stack_frame(line: &str, evidence_line: usize) -> Option<StackFrame> {
    let rest = line.trim().strip_prefix("at ")?;
    let frame = rest.trim_end_matches(':');
    let (before_column, column) = frame.rsplit_once(':')?;
    let (file, line_number) = before_column.rsplit_once(':')?;
    Some(StackFrame {
        file: file.to_string(),
        line: line_number.parse().ok()?,
        column: column.parse().ok()?,
        evidence_line,
    })
}

fn relevant_source(frames: &[StackFrame]) -> Option<EvaluationSourceLocation> {
    let frame = frames
        .iter()
        .find(|frame| {
            let path = frame.file.to_lowercase();
            path.contains("/etc/nixos/")
                || path.contains("/overlays/")
                || path.contains("/hosts/")
                || path.contains("/modules/")
        })
        .or_else(|| {
            frames.iter().find(|frame| {
                let path = frame.file.to_lowercase();
                path.ends_with(".nix")
                    && !path.contains("/lib/modules.nix")
                    && !path.contains("/lib/attrsets.nix")
                    && !path.contains("/lib/trivial.nix")
            })
        })?;
    Some(EvaluationSourceLocation {
        file: frame.file.clone(),
        line: frame.line,
        column: frame.column,
    })
}

fn append_source_evidence(
    evidence: &mut Vec<EvaluationEvidence>,
    frames: &[StackFrame],
    source: Option<&EvaluationSourceLocation>,
) {
    let Some(source) = source else {
        return;
    };
    if let Some(frame) = frames.iter().find(|frame| {
        frame.file == source.file
            && frame.line == source.line
            && frame.column == source.column
    }) {
        if !evidence
            .iter()
            .any(|item| item.line_number == frame.evidence_line)
        {
            // This is a stack-frame string, not a claim about source-file contents.
            evidence.push(EvaluationEvidence {
                line_number: frame.evidence_line,
                text: format!("at {}:{}:{}:", frame.file, frame.line, frame.column),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BRAVE_TRACE: &str = r#"
warning: unable to download 'https://cuda-maintainers.cachix.org/nix-cache-info': HTTP error 401

building the system configuration...
error:
       … while evaluating the attribute 'value'
         at /nix/store/example-nixpkgs/lib/modules.nix:1304:5:
       … while evaluating definitions from '/etc/nixos/overlays/default.nix':
         at /etc/nixos/overlays/default.nix:41:11:
          40|   youtube-dl = prev.yt-dlp;
          41|   brave = prev.brave.overrideAttrs (old: {
          42|     postFixup = (old.postFixup or "") + ''
       error: brave has been removed from nixpkgs, as there were no committers among its maintainers to unblock security issues
"#;

    #[test]
    fn removed_package_is_primary_and_cache_401_is_only_a_warning() {
        let report = NixEvaluationReport::analyze(BRAVE_TRACE);
        assert_eq!(report.error_count(), 1);
        assert_eq!(report.warning_count(), 1);

        let primary = report.primary().expect("fatal error should be identified");
        assert_eq!(primary.kind, EvaluationIssueKind::RemovedPackage);
        assert_eq!(primary.subject.as_deref(), Some("brave"));
        let source = primary.source.as_ref().expect("local overlay frame expected");
        assert_eq!(source.file, "/etc/nixos/overlays/default.nix");
        assert_eq!(source.line, 41);
        assert_eq!(source.column, 11);

        assert_eq!(
            report.diagnostics[1].kind,
            EvaluationIssueKind::CacheAuthorization
        );
        assert_eq!(report.diagnostics[1].severity, EvaluationSeverity::Warning);
    }

    #[test]
    fn cache_warning_alone_does_not_become_primary_failure() {
        let report = NixEvaluationReport::analyze(
            "warning: unable to download 'https://example.cachix.org/nix-cache-info': HTTP error 401",
        );
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.warning_count(), 1);
        assert!(report.primary().is_none());
    }

    #[test]
    fn unknown_option_parses_nix_backtick_apostrophe_quoting() {
        let report = NixEvaluationReport::analyze(
            "error: The option `services.example.enable' does not exist. Definition values:",
        );
        let primary = report.primary().unwrap();
        assert_eq!(primary.kind, EvaluationIssueKind::UnknownOption);
        assert_eq!(
            primary.subject.as_deref(),
            Some("services.example.enable")
        );
        assert!(primary.recommendations.iter().all(|item| !item.contains("replace")));
    }

    #[test]
    fn unknown_option_also_accepts_symmetric_backticks() {
        let delimiter = char::from(96);
        let message = format!(
            "error: The option {delimiter}services.example.enable{delimiter} does not exist."
        );
        let report = NixEvaluationReport::analyze(&message);
        assert_eq!(
            report.primary().and_then(|diagnostic| diagnostic.subject.as_deref()),
            Some("services.example.enable")
        );
    }

    #[test]
    fn unknown_errors_are_preserved_as_unknown() {
        let report = NixEvaluationReport::analyze(
            "error: an unfamiliar evaluator failure happened here",
        );
        let primary = report.primary().unwrap();
        assert_eq!(primary.kind, EvaluationIssueKind::UnknownEvaluationError);
        assert_eq!(primary.evidence[0].line_number, 1);
    }

    #[test]
    fn cache_403_is_a_warning_not_a_fatal_failure() {
        let report = NixEvaluationReport::analyze(
            "warning: https://private-cache.example/cache-info returned HTTP 403 Forbidden",
        );
        assert_eq!(report.warning_count(), 1);
        assert!(report.primary().is_none());
    }

    #[test]
    fn removed_attribute_subject_handles_quotes() {
        let report =
            NixEvaluationReport::analyze("error: 'brave-origin' has been removed from nixpkgs");
        assert_eq!(
            report.primary().unwrap().subject.as_deref(),
            Some("brave-origin")
        );
    }
}
