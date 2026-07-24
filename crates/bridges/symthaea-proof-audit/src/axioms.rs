// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Parsing and classification of Lean `#print axioms` output.
//!
//! Lean 4's `#print axioms foo` reports the axioms a proof transitively depends
//! on — the ground truth of what a proof actually assumes. Two output shapes:
//!
//! ```text
//! 'foo' does not depend on any axioms
//! 'foo' depends on axioms: [propext, Classical.choice, Quot.sound]
//! ```
//!
//! A proof that used `sorry` reports `sorryAx` in the list — which is how the
//! provenance gate catches an unproved proof even if the source string was
//! scrubbed.

use std::fmt;

/// A `#print axioms` transcript that cannot be authenticated as a Lean report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxiomParseError {
    /// No supported `#print axioms` report was present in the captured output.
    MissingReport,
    /// A report marker was present, but its theorem name or axiom list was malformed.
    MalformedReport(String),
}

impl fmt::Display for AxiomParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingReport => write!(f, "missing Lean #print axioms report"),
            Self::MalformedReport(reason) => {
                write!(f, "malformed Lean #print axioms report: {reason}")
            }
        }
    }
}

impl std::error::Error for AxiomParseError {}

/// The axiom dependencies of a single proved theorem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxiomReport {
    /// The theorem name, if recoverable from the output.
    pub theorem: String,
    /// Axioms the proof transitively depends on (order as reported).
    pub axioms: Vec<String>,
}

impl AxiomReport {
    /// Parse the textual output of `#print axioms <name>`.
    ///
    /// Empty output, compiler diagnostics, and truncated lists are errors,
    /// never evidence of an axiom-free proof.
    pub fn parse(output: &str) -> Result<AxiomReport, AxiomParseError> {
        let flat = output.replace('\n', " ");
        let theorem = first_quoted(&flat).ok_or_else(|| {
            if flat.contains("depend on any axioms") || flat.contains("depends on axioms:") {
                AxiomParseError::MalformedReport("missing quoted theorem name".into())
            } else {
                AxiomParseError::MissingReport
            }
        })?;

        if flat.contains("does not depend on any axioms") {
            return Ok(AxiomReport {
                theorem,
                axioms: Vec::new(),
            });
        }

        let idx = flat
            .find("depends on axioms:")
            .ok_or(AxiomParseError::MissingReport)?;
        let after = &flat[idx + "depends on axioms:".len()..];
        let open = after
            .find('[')
            .ok_or_else(|| AxiomParseError::MalformedReport("missing opening '['".into()))?;
        let close = after[open + 1..]
            .find(']')
            .map(|relative| open + 1 + relative)
            .ok_or_else(|| AxiomParseError::MalformedReport("missing closing ']'".into()))?;
        let axioms = after[open + 1..close]
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect();

        Ok(AxiomReport { theorem, axioms })
    }

    /// True if the proof depends on `sorryAx` — i.e. it is not actually proved.
    pub fn has_sorry(&self) -> bool {
        self.axioms.iter().any(|a| is_sorry(a))
    }

    /// Classical (choice-based) axioms the proof depends on.
    pub fn classical_axioms(&self) -> Vec<&str> {
        self.axioms
            .iter()
            .filter(|a| is_classical(a))
            .map(|s| s.as_str())
            .collect()
    }
}

/// Whether an axiom name denotes `sorry`.
pub fn is_sorry(axiom: &str) -> bool {
    axiom == "sorryAx" || axiom.ends_with(".sorryAx")
}

/// Whether an axiom name denotes classical reasoning (choice / excluded middle).
/// In Lean, `Classical.choice` yields excluded middle via Diaconescu's theorem,
/// so it is the marker that a proof left the constructive fragment.
pub fn is_classical(axiom: &str) -> bool {
    axiom.starts_with("Classical.") || axiom == "Classical.choice"
}

/// The first single-quoted token (`'foo'`) in a string.
fn first_quoted(s: &str) -> Option<String> {
    let start = s.find('\'')?;
    let rest = &s[start + 1..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_no_axioms() {
        let r = AxiomReport::parse("'foo' does not depend on any axioms").unwrap();
        assert_eq!(r.theorem, "foo");
        assert!(r.axioms.is_empty());
        assert!(!r.has_sorry());
    }

    #[test]
    fn parses_axiom_list() {
        let r =
            AxiomReport::parse("'thm' depends on axioms: [propext, Classical.choice, Quot.sound]")
                .unwrap();
        assert_eq!(r.theorem, "thm");
        assert_eq!(r.axioms, vec!["propext", "Classical.choice", "Quot.sound"]);
        assert_eq!(r.classical_axioms(), vec!["Classical.choice"]);
    }

    #[test]
    fn detects_sorry() {
        let r = AxiomReport::parse("'bad' depends on axioms: [sorryAx]").unwrap();
        assert!(r.has_sorry());
    }

    #[test]
    fn tolerates_multiline_output() {
        let out = "'gate_mono' depends on axioms: [propext,\n  Quot.sound]";
        let r = AxiomReport::parse(out).unwrap();
        assert_eq!(r.axioms, vec!["propext", "Quot.sound"]);
    }

    #[test]
    fn classification_helpers() {
        assert!(is_sorry("sorryAx"));
        assert!(!is_sorry("propext"));
        assert!(is_classical("Classical.choice"));
        assert!(is_classical("Classical.em"));
        assert!(!is_classical("propext"));
    }

    #[test]
    fn rejects_empty_or_unrelated_output() {
        assert_eq!(AxiomReport::parse(""), Err(AxiomParseError::MissingReport));
        assert_eq!(
            AxiomReport::parse("error: declaration uses 'sorry'"),
            Err(AxiomParseError::MissingReport)
        );
    }

    #[test]
    fn rejects_truncated_axiom_list() {
        assert!(matches!(
            AxiomReport::parse("'t' depends on axioms: [propext"),
            Err(AxiomParseError::MalformedReport(_))
        ));
    }
}
