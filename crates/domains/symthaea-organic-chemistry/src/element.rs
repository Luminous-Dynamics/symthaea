// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Element data for the organic-relevant subset of the periodic table.
//!
//! Standard atomic weights are IUPAC conventional values (2021). `normal_valence`
//! is the lowest common neutral valence, used to derive implicit hydrogen counts
//! for the SMILES "organic subset" atoms.

/// Static data for one chemical element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Element {
    /// Element symbol, e.g. `"C"`.
    pub symbol: &'static str,
    /// Atomic number (proton count).
    pub atomic_number: u8,
    /// Standard atomic weight (g/mol), IUPAC conventional value.
    pub weight: f64,
    /// Lowest common neutral valence, used for implicit-H derivation.
    pub normal_valence: u8,
}

/// Look up an element by symbol (case-sensitive, proper element casing).
///
/// Covers the organic-chemistry working set (CHNOPS + halogens + B, Si).
pub fn lookup(symbol: &str) -> Option<Element> {
    let e = |symbol: &'static str, atomic_number: u8, weight: f64, normal_valence: u8| Element {
        symbol,
        atomic_number,
        weight,
        normal_valence,
    };
    Some(match symbol {
        "H" => e("H", 1, 1.008, 1),
        "B" => e("B", 5, 10.81, 3),
        "C" => e("C", 6, 12.011, 4),
        "N" => e("N", 7, 14.007, 3),
        "O" => e("O", 8, 15.999, 2),
        "F" => e("F", 9, 18.998, 1),
        "Si" => e("Si", 14, 28.085, 4),
        "P" => e("P", 15, 30.974, 3),
        "S" => e("S", 16, 32.06, 2),
        "Cl" => e("Cl", 17, 35.45, 1),
        "Br" => e("Br", 35, 79.904, 1),
        "I" => e("I", 53, 126.904, 1),
        _ => return None,
    })
}

/// Whether a symbol is in the SMILES "organic subset" that may be written
/// without brackets and receives implicit hydrogens automatically.
pub fn is_organic_subset(symbol: &str) -> bool {
    matches!(
        symbol,
        "B" | "C" | "N" | "O" | "P" | "S" | "F" | "Cl" | "Br" | "I"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carbon_weight_and_valence() {
        let c = lookup("C").unwrap();
        assert_eq!(c.atomic_number, 6);
        assert_eq!(c.normal_valence, 4);
        assert!((c.weight - 12.011).abs() < 1e-9);
    }

    #[test]
    fn two_letter_symbol_resolves() {
        assert_eq!(lookup("Cl").unwrap().atomic_number, 17);
        assert_eq!(lookup("Br").unwrap().atomic_number, 35);
    }

    #[test]
    fn unknown_symbol_is_none() {
        assert!(lookup("Xx").is_none());
    }

    #[test]
    fn organic_subset_membership() {
        assert!(is_organic_subset("C"));
        assert!(is_organic_subset("Cl"));
        assert!(!is_organic_subset("Si"));
        assert!(!is_organic_subset("H"));
    }
}
