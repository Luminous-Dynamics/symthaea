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
        const NO_AXIOMS: &str = " does not depend on any axioms";
        const WITH_AXIOMS: &str = " depends on axioms:";
        const TRUNCATION: &str = "[... output truncated at the capture limit ...]";

        if output.contains(TRUNCATION) {
            return Err(AxiomParseError::MalformedReport(
                "captured Lean output was truncated".into(),
            ));
        }

        // A Lean proof file can itself emit stdout (for example with #eval).
        // Never trust the first report-looking string: it could be caller-
        // generated text preceding the real #print axioms output. Exactly one
        // report marker is required; two reports are ambiguous and fail closed.
        let no_axioms: Vec<usize> = output.match_indices(NO_AXIOMS).map(|(i, _)| i).collect();
        let with_axioms: Vec<usize> = output.match_indices(WITH_AXIOMS).map(|(i, _)| i).collect();
        if no_axioms.len() + with_axioms.len() != 1 {
            return Err(if no_axioms.is_empty() && with_axioms.is_empty() {
                AxiomParseError::MissingReport
            } else {
                AxiomParseError::MalformedReport(
                    "multiple or ambiguous axiom-report markers".into(),
                )
            });
        }

        let (marker, marker_len, is_axiom_free) = if let Some(index) = no_axioms.first() {
            (*index, NO_AXIOMS.len(), true)
        } else {
            (*with_axioms.first().expect("exactly one marker was checked"), WITH_AXIOMS.len(), false)
        };

        // Bind the theorem name to the quoted token immediately preceding the
        // unique marker, not to the first quoted text in arbitrary compiler or
        // user-generated output.
        let before = &output[..marker];
        let closing_quote = before.rfind('\'').ok_or_else(|| {
            AxiomParseError::MalformedReport("missing quoted theorem name".into())
        })?;
        let before_closing = &before[..closing_quote];
        let opening_quote = before_closing.rfind('\'').ok_or_else(|| {
            AxiomParseError::MalformedReport("missing quoted theorem name".into())
        })?;
        let theorem = before_closing[opening_quote + 1..closing_quote].trim();
        if theorem.is_empty() || theorem.chars().any(char::is_control) {
            return Err(AxiomParseError::MalformedReport(
                "invalid quoted theorem name".into(),
            ));
        }

        if is_axiom_free {
            return Ok(AxiomReport {
                theorem: theorem.to_owned(),
                axioms: Vec::new(),
            });
        }

        let after = &output[marker + marker_len..];
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

        Ok(AxiomReport {
            theorem: theorem.to_owned(),
            axioms,
        })
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
    fn rejects_ambiguous_multiple_reports_including_output_spoofing() {
        let output = concat!(
            "'thm' does not depend on any axioms\n",
            "'thm' depends on axioms: [sorryAx]"
        );
        assert!(matches!(
            AxiomReport::parse(output),
            Err(AxiomParseError::MalformedReport(_))
        ));
    }

    #[test]
    fn binds_name_to_report_not_unrelated_quoted_diagnostic() {
        let output = "warning: unrelated theorem 'other' was skipped\n'foo' does not depend on any axioms";
        let report = AxiomReport::parse(output).unwrap();
        assert_eq!(report.theorem, "foo");
        assert!(report.axioms.is_empty());
    }

    #[test]
    fn rejects_truncated_output_even_when_a_complete_report_precedes_it() {
        let output = concat!(
            "'thm' does not depend on any axioms\n",
            "[... output truncated at the capture limit ...]"
        );
        assert!(matches!(
            AxiomReport::parse(output),
            Err(AxiomParseError::MalformedReport(_))
        ));
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
