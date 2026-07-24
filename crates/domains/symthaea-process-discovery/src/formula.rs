// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Molecular formula string comparison, refined (Phase A.2 follow-up) to
//! separate a genuine compositional disagreement from a mere presentation
//! convention difference.
//!
//! Motivated by a real finding from the first live RDKit run: this crate's
//! `molecular_formula()` returns `"ClH"` for hydrogen chloride, RDKit
//! returns `"HCl"` -- a Hill-notation element-ordering convention for
//! carbon-free compounds, not a disagreement about what the molecule
//! actually is (both agree: 1 H, 1 Cl). Comparing formula strings for exact
//! equality conflated this with a real compositional mismatch, which would
//! overstate what "N disagreements" actually means to a reviewer. Parsing
//! both sides into element-count maps and comparing THOSE is what makes the
//! distinction honest.

use std::collections::BTreeMap;

/// Parses a Hill-notation-shaped formula string (`"C6H5NO2"`, `"ClH"`, ...)
/// into an element -> count map. Returns `None` for anything that doesn't
/// match the expected shape (uppercase-led element symbol, optional
/// lowercase continuation, optional digit count) -- fails closed rather
/// than guessing at a malformed or unexpected format.
pub fn parse_formula(formula: &str) -> Option<BTreeMap<String, u32>> {
    if formula.is_empty() {
        return None;
    }
    let chars: Vec<char> = formula.chars().collect();
    let mut map = BTreeMap::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_uppercase() {
            return None;
        }
        let mut symbol = String::new();
        symbol.push(chars[i]);
        i += 1;
        while i < chars.len() && chars[i].is_ascii_lowercase() {
            symbol.push(chars[i]);
            i += 1;
        }
        let mut digits = String::new();
        while i < chars.len() && chars[i].is_ascii_digit() {
            digits.push(chars[i]);
            i += 1;
        }
        let count: u32 = if digits.is_empty() {
            1
        } else {
            digits.parse().ok()?
        };
        *map.entry(symbol).or_insert(0) += count;
    }
    Some(map)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormulaComparison {
    /// The two formula strings are byte-identical.
    ExactMatch,
    /// The strings differ, but parse to the same element-count map -- a
    /// presentation/convention difference (e.g. element ordering), not a
    /// disagreement about what the molecule is.
    RepresentationOnlyDifference,
    /// The parsed element-count maps genuinely differ, OR either string
    /// failed to parse at all -- a real compositional disagreement worth a
    /// human look. Failing to parse is deliberately bucketed here, not
    /// treated as "unknown, skip it": an un-parseable formula from an
    /// external source is itself worth surfacing, not silently dropped.
    CompositionDisagreement,
}

pub fn compare_formulas(a: &str, b: &str) -> FormulaComparison {
    if a == b {
        return FormulaComparison::ExactMatch;
    }
    match (parse_formula(a), parse_formula(b)) {
        (Some(ma), Some(mb)) if ma == mb => FormulaComparison::RepresentationOnlyDifference,
        _ => FormulaComparison::CompositionDisagreement,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_simple_formula() {
        let m = parse_formula("C6H5NO2").unwrap();
        assert_eq!(m.get("C"), Some(&6));
        assert_eq!(m.get("H"), Some(&5));
        assert_eq!(m.get("N"), Some(&1));
        assert_eq!(m.get("O"), Some(&2));
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn multi_letter_element_symbol_parses_correctly() {
        let m = parse_formula("ClH").unwrap();
        assert_eq!(m.get("Cl"), Some(&1));
        assert_eq!(m.get("H"), Some(&1));
    }

    #[test]
    fn hcl_and_clh_parse_to_the_identical_map() {
        // The exact real-world case this module exists for.
        assert_eq!(parse_formula("HCl"), parse_formula("ClH"));
    }

    #[test]
    fn implicit_count_of_one_is_correctly_inferred() {
        let m = parse_formula("H2O").unwrap();
        assert_eq!(m.get("H"), Some(&2));
        assert_eq!(m.get("O"), Some(&1)); // no trailing digit -- implicit 1
    }

    #[test]
    fn empty_string_fails_to_parse() {
        assert!(parse_formula("").is_none());
    }

    #[test]
    fn lowercase_leading_character_fails_to_parse() {
        assert!(parse_formula("cH4").is_none());
    }

    #[test]
    fn exact_string_match_is_exact_match() {
        assert_eq!(
            compare_formulas("C2H6O", "C2H6O"),
            FormulaComparison::ExactMatch
        );
    }

    #[test]
    fn ordering_difference_is_representation_only_not_a_disagreement() {
        assert_eq!(
            compare_formulas("HCl", "ClH"),
            FormulaComparison::RepresentationOnlyDifference
        );
    }

    #[test]
    fn genuinely_different_composition_is_a_disagreement() {
        assert_eq!(
            compare_formulas("C2H6O", "C3H8O"),
            FormulaComparison::CompositionDisagreement
        );
    }

    #[test]
    fn unparseable_formula_fails_closed_as_a_disagreement() {
        assert_eq!(
            compare_formulas("not a formula!!", "C2H6O"),
            FormulaComparison::CompositionDisagreement
        );
    }
}
