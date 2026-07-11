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
    pub fn parse(output: &str) -> AxiomReport {
        let flat = output.replace('\n', " ");
        let theorem = first_quoted(&flat).unwrap_or_default();

        if flat.contains("does not depend on any axioms") {
            return AxiomReport {
                theorem,
                axioms: Vec::new(),
            };
        }

        let axioms = match flat.find("depends on axioms:") {
            Some(idx) => {
                let after = &flat[idx + "depends on axioms:".len()..];
                let inner = match (after.find('['), after.find(']')) {
                    (Some(a), Some(b)) if b > a => &after[a + 1..b],
                    _ => after.trim(),
                };
                inner
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            None => Vec::new(),
        };

        AxiomReport { theorem, axioms }
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
        let r = AxiomReport::parse("'foo' does not depend on any axioms");
        assert_eq!(r.theorem, "foo");
        assert!(r.axioms.is_empty());
        assert!(!r.has_sorry());
    }

    #[test]
    fn parses_axiom_list() {
        let r =
            AxiomReport::parse("'thm' depends on axioms: [propext, Classical.choice, Quot.sound]");
        assert_eq!(r.theorem, "thm");
        assert_eq!(r.axioms, vec!["propext", "Classical.choice", "Quot.sound"]);
        assert_eq!(r.classical_axioms(), vec!["Classical.choice"]);
    }

    #[test]
    fn detects_sorry() {
        let r = AxiomReport::parse("'bad' depends on axioms: [sorryAx]");
        assert!(r.has_sorry());
    }

    #[test]
    fn tolerates_multiline_output() {
        let out = "'gate_mono' depends on axioms: [propext,\n  Quot.sound]";
        let r = AxiomReport::parse(out);
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
}
