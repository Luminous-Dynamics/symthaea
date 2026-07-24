// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Periodic-table metadata for all 118 confirmed real elements (Z=1-118).
//!
//! Pure descriptive data (symbol, standard atomic mass, period, block) --
//! NOT basis-set or ECP availability, which is a separate, correctness-
//! sensitive question tracked per basis provider (see
//! `basis::sto3g::Sto3g::supports_element` /
//! `basis::basis_631g::Basis631G::supports_element`).
//!
//! Data fetched directly via `curl` from `Bowserinator/Periodic-Table-JSON`
//! (a well-known, widely-used open dataset), vendored verbatim at
//! `element_data/reference/periodic_table.json`, generated into this table
//! via `scripts/generate_element_data.py` -- deliberately not via
//! `WebFetch`, whose AI-summarization step was found to corrupt dense
//! numeric tables during Phase A.7 (see the project memory note
//! `feedback_webfetch_summarization_corrupts_dense_tables.md`). The source
//! dataset has 119 entries; Z=119 (Ununennium) is a hypothetical,
//! undiscovered element with a placeholder/predicted mass and is excluded.
//!
//! Phase Q1, 2026-07-16.

/// Periodic-table block (s/p/d/f), per the standard IUPAC-style convention
/// (La/Ac classified as f-block, Lu/Lr as d-block, matching the source
/// dataset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Block {
    S,
    P,
    D,
    F,
}

/// Periodic-table metadata for one element.
#[derive(Debug, Clone, Copy)]
pub struct ElementMetadata {
    pub atomic_number: u8,
    pub symbol: &'static str,
    /// Standard atomic weight (IUPAC), amu.
    pub atomic_mass: f64,
    pub period: u8,
    pub block: Block,
}

#[rustfmt::skip]
pub(crate) static ELEMENT_METADATA: [ElementMetadata; 118] = [
    ElementMetadata { atomic_number: 1, symbol: "H", atomic_mass: 1.008, period: 1, block: Block::S },
    ElementMetadata { atomic_number: 2, symbol: "He", atomic_mass: 4.0026022, period: 1, block: Block::S },
    ElementMetadata { atomic_number: 3, symbol: "Li", atomic_mass: 6.94, period: 2, block: Block::S },
    ElementMetadata { atomic_number: 4, symbol: "Be", atomic_mass: 9.01218315, period: 2, block: Block::S },
    ElementMetadata { atomic_number: 5, symbol: "B", atomic_mass: 10.81, period: 2, block: Block::P },
    ElementMetadata { atomic_number: 6, symbol: "C", atomic_mass: 12.011, period: 2, block: Block::P },
    ElementMetadata { atomic_number: 7, symbol: "N", atomic_mass: 14.007, period: 2, block: Block::P },
    ElementMetadata { atomic_number: 8, symbol: "O", atomic_mass: 15.999, period: 2, block: Block::P },
    ElementMetadata { atomic_number: 9, symbol: "F", atomic_mass: 18.9984031636, period: 2, block: Block::P },
    ElementMetadata { atomic_number: 10, symbol: "Ne", atomic_mass: 20.17976, period: 2, block: Block::P },
    ElementMetadata { atomic_number: 11, symbol: "Na", atomic_mass: 22.989769282, period: 3, block: Block::S },
    ElementMetadata { atomic_number: 12, symbol: "Mg", atomic_mass: 24.305, period: 3, block: Block::S },
    ElementMetadata { atomic_number: 13, symbol: "Al", atomic_mass: 26.98153857, period: 3, block: Block::P },
    ElementMetadata { atomic_number: 14, symbol: "Si", atomic_mass: 28.085, period: 3, block: Block::P },
    ElementMetadata { atomic_number: 15, symbol: "P", atomic_mass: 30.9737619985, period: 3, block: Block::P },
    ElementMetadata { atomic_number: 16, symbol: "S", atomic_mass: 32.06, period: 3, block: Block::P },
    ElementMetadata { atomic_number: 17, symbol: "Cl", atomic_mass: 35.45, period: 3, block: Block::P },
    ElementMetadata { atomic_number: 18, symbol: "Ar", atomic_mass: 39.9481, period: 3, block: Block::P },
    ElementMetadata { atomic_number: 19, symbol: "K", atomic_mass: 39.09831, period: 4, block: Block::S },
    ElementMetadata { atomic_number: 20, symbol: "Ca", atomic_mass: 40.0784, period: 4, block: Block::S },
    ElementMetadata { atomic_number: 21, symbol: "Sc", atomic_mass: 44.9559085, period: 4, block: Block::D },
    ElementMetadata { atomic_number: 22, symbol: "Ti", atomic_mass: 47.8671, period: 4, block: Block::D },
    ElementMetadata { atomic_number: 23, symbol: "V", atomic_mass: 50.94151, period: 4, block: Block::D },
    ElementMetadata { atomic_number: 24, symbol: "Cr", atomic_mass: 51.99616, period: 4, block: Block::D },
    ElementMetadata { atomic_number: 25, symbol: "Mn", atomic_mass: 54.9380443, period: 4, block: Block::D },
    ElementMetadata { atomic_number: 26, symbol: "Fe", atomic_mass: 55.8452, period: 4, block: Block::D },
    ElementMetadata { atomic_number: 27, symbol: "Co", atomic_mass: 58.9331944, period: 4, block: Block::D },
    ElementMetadata { atomic_number: 28, symbol: "Ni", atomic_mass: 58.69344, period: 4, block: Block::D },
    ElementMetadata { atomic_number: 29, symbol: "Cu", atomic_mass: 63.5463, period: 4, block: Block::D },
    ElementMetadata { atomic_number: 30, symbol: "Zn", atomic_mass: 65.382, period: 4, block: Block::D },
    ElementMetadata { atomic_number: 31, symbol: "Ga", atomic_mass: 69.7231, period: 4, block: Block::P },
    ElementMetadata { atomic_number: 32, symbol: "Ge", atomic_mass: 72.6308, period: 4, block: Block::P },
    ElementMetadata { atomic_number: 33, symbol: "As", atomic_mass: 74.9215956, period: 4, block: Block::P },
    ElementMetadata { atomic_number: 34, symbol: "Se", atomic_mass: 78.9718, period: 4, block: Block::P },
    ElementMetadata { atomic_number: 35, symbol: "Br", atomic_mass: 79.904, period: 4, block: Block::P },
    ElementMetadata { atomic_number: 36, symbol: "Kr", atomic_mass: 83.7982, period: 4, block: Block::P },
    ElementMetadata { atomic_number: 37, symbol: "Rb", atomic_mass: 85.46783, period: 5, block: Block::S },
    ElementMetadata { atomic_number: 38, symbol: "Sr", atomic_mass: 87.621, period: 5, block: Block::S },
    ElementMetadata { atomic_number: 39, symbol: "Y", atomic_mass: 88.905842, period: 5, block: Block::D },
    ElementMetadata { atomic_number: 40, symbol: "Zr", atomic_mass: 91.2242, period: 5, block: Block::D },
    ElementMetadata { atomic_number: 41, symbol: "Nb", atomic_mass: 92.906372, period: 5, block: Block::D },
    ElementMetadata { atomic_number: 42, symbol: "Mo", atomic_mass: 95.951, period: 5, block: Block::D },
    ElementMetadata { atomic_number: 43, symbol: "Tc", atomic_mass: 98.0, period: 5, block: Block::D },
    ElementMetadata { atomic_number: 44, symbol: "Ru", atomic_mass: 101.072, period: 5, block: Block::D },
    ElementMetadata { atomic_number: 45, symbol: "Rh", atomic_mass: 102.905502, period: 5, block: Block::D },
    ElementMetadata { atomic_number: 46, symbol: "Pd", atomic_mass: 106.421, period: 5, block: Block::D },
    ElementMetadata { atomic_number: 47, symbol: "Ag", atomic_mass: 107.86822, period: 5, block: Block::D },
    ElementMetadata { atomic_number: 48, symbol: "Cd", atomic_mass: 112.4144, period: 5, block: Block::D },
    ElementMetadata { atomic_number: 49, symbol: "In", atomic_mass: 114.8181, period: 5, block: Block::P },
    ElementMetadata { atomic_number: 50, symbol: "Sn", atomic_mass: 118.7107, period: 5, block: Block::P },
    ElementMetadata { atomic_number: 51, symbol: "Sb", atomic_mass: 121.7601, period: 5, block: Block::P },
    ElementMetadata { atomic_number: 52, symbol: "Te", atomic_mass: 127.603, period: 5, block: Block::P },
    ElementMetadata { atomic_number: 53, symbol: "I", atomic_mass: 126.904473, period: 5, block: Block::P },
    ElementMetadata { atomic_number: 54, symbol: "Xe", atomic_mass: 131.2936, period: 5, block: Block::P },
    ElementMetadata { atomic_number: 55, symbol: "Cs", atomic_mass: 132.905451966, period: 6, block: Block::S },
    ElementMetadata { atomic_number: 56, symbol: "Ba", atomic_mass: 137.3277, period: 6, block: Block::S },
    ElementMetadata { atomic_number: 57, symbol: "La", atomic_mass: 138.905477, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 58, symbol: "Ce", atomic_mass: 140.1161, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 59, symbol: "Pr", atomic_mass: 140.907662, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 60, symbol: "Nd", atomic_mass: 144.2423, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 61, symbol: "Pm", atomic_mass: 145.0, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 62, symbol: "Sm", atomic_mass: 150.362, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 63, symbol: "Eu", atomic_mass: 151.9641, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 64, symbol: "Gd", atomic_mass: 157.253, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 65, symbol: "Tb", atomic_mass: 158.925352, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 66, symbol: "Dy", atomic_mass: 162.5001, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 67, symbol: "Ho", atomic_mass: 164.930332, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 68, symbol: "Er", atomic_mass: 167.2593, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 69, symbol: "Tm", atomic_mass: 168.934222, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 70, symbol: "Yb", atomic_mass: 173.0451, period: 6, block: Block::F },
    ElementMetadata { atomic_number: 71, symbol: "Lu", atomic_mass: 174.96681, period: 6, block: Block::D },
    ElementMetadata { atomic_number: 72, symbol: "Hf", atomic_mass: 178.492, period: 6, block: Block::D },
    ElementMetadata { atomic_number: 73, symbol: "Ta", atomic_mass: 180.947882, period: 6, block: Block::D },
    ElementMetadata { atomic_number: 74, symbol: "W", atomic_mass: 183.841, period: 6, block: Block::D },
    ElementMetadata { atomic_number: 75, symbol: "Re", atomic_mass: 186.2071, period: 6, block: Block::D },
    ElementMetadata { atomic_number: 76, symbol: "Os", atomic_mass: 190.233, period: 6, block: Block::D },
    ElementMetadata { atomic_number: 77, symbol: "Ir", atomic_mass: 192.2173, period: 6, block: Block::D },
    ElementMetadata { atomic_number: 78, symbol: "Pt", atomic_mass: 195.0849, period: 6, block: Block::D },
    ElementMetadata { atomic_number: 79, symbol: "Au", atomic_mass: 196.9665695, period: 6, block: Block::D },
    ElementMetadata { atomic_number: 80, symbol: "Hg", atomic_mass: 200.5923, period: 6, block: Block::D },
    ElementMetadata { atomic_number: 81, symbol: "Tl", atomic_mass: 204.38, period: 6, block: Block::P },
    ElementMetadata { atomic_number: 82, symbol: "Pb", atomic_mass: 207.21, period: 6, block: Block::P },
    ElementMetadata { atomic_number: 83, symbol: "Bi", atomic_mass: 208.980401, period: 6, block: Block::P },
    ElementMetadata { atomic_number: 84, symbol: "Po", atomic_mass: 209.0, period: 6, block: Block::P },
    ElementMetadata { atomic_number: 85, symbol: "At", atomic_mass: 210.0, period: 6, block: Block::P },
    ElementMetadata { atomic_number: 86, symbol: "Rn", atomic_mass: 222.0, period: 6, block: Block::P },
    ElementMetadata { atomic_number: 87, symbol: "Fr", atomic_mass: 223.0, period: 7, block: Block::S },
    ElementMetadata { atomic_number: 88, symbol: "Ra", atomic_mass: 226.0, period: 7, block: Block::S },
    ElementMetadata { atomic_number: 89, symbol: "Ac", atomic_mass: 227.0, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 90, symbol: "Th", atomic_mass: 232.03774, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 91, symbol: "Pa", atomic_mass: 231.035882, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 92, symbol: "U", atomic_mass: 238.028913, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 93, symbol: "Np", atomic_mass: 237.0, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 94, symbol: "Pu", atomic_mass: 244.0, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 95, symbol: "Am", atomic_mass: 243.0, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 96, symbol: "Cm", atomic_mass: 247.0, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 97, symbol: "Bk", atomic_mass: 247.0, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 98, symbol: "Cf", atomic_mass: 251.0, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 99, symbol: "Es", atomic_mass: 252.0, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 100, symbol: "Fm", atomic_mass: 257.0, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 101, symbol: "Md", atomic_mass: 258.0, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 102, symbol: "No", atomic_mass: 259.0, period: 7, block: Block::F },
    ElementMetadata { atomic_number: 103, symbol: "Lr", atomic_mass: 266.0, period: 7, block: Block::D },
    ElementMetadata { atomic_number: 104, symbol: "Rf", atomic_mass: 267.0, period: 7, block: Block::D },
    ElementMetadata { atomic_number: 105, symbol: "Db", atomic_mass: 268.0, period: 7, block: Block::D },
    ElementMetadata { atomic_number: 106, symbol: "Sg", atomic_mass: 269.0, period: 7, block: Block::D },
    ElementMetadata { atomic_number: 107, symbol: "Bh", atomic_mass: 270.0, period: 7, block: Block::D },
    ElementMetadata { atomic_number: 108, symbol: "Hs", atomic_mass: 269.0, period: 7, block: Block::D },
    ElementMetadata { atomic_number: 109, symbol: "Mt", atomic_mass: 278.0, period: 7, block: Block::D },
    ElementMetadata { atomic_number: 110, symbol: "Ds", atomic_mass: 281.0, period: 7, block: Block::D },
    ElementMetadata { atomic_number: 111, symbol: "Rg", atomic_mass: 282.0, period: 7, block: Block::D },
    ElementMetadata { atomic_number: 112, symbol: "Cn", atomic_mass: 285.0, period: 7, block: Block::D },
    ElementMetadata { atomic_number: 113, symbol: "Nh", atomic_mass: 286.0, period: 7, block: Block::P },
    ElementMetadata { atomic_number: 114, symbol: "Fl", atomic_mass: 289.0, period: 7, block: Block::P },
    ElementMetadata { atomic_number: 115, symbol: "Mc", atomic_mass: 289.0, period: 7, block: Block::P },
    ElementMetadata { atomic_number: 116, symbol: "Lv", atomic_mass: 293.0, period: 7, block: Block::P },
    ElementMetadata { atomic_number: 117, symbol: "Ts", atomic_mass: 294.0, period: 7, block: Block::P },
    ElementMetadata { atomic_number: 118, symbol: "Og", atomic_mass: 294.0, period: 7, block: Block::P },
];

/// Look up periodic-table metadata by atomic number. Returns `None` for
/// `z == 0` or `z > 118` (there is no element beyond the 118 confirmed
/// real ones -- see the module doc for why Z=119 is excluded).
pub fn element_metadata(z: u8) -> Option<&'static ElementMetadata> {
    if z == 0 {
        return None;
    }
    ELEMENT_METADATA.get((z - 1) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_118_elements_present_and_contiguous() {
        for z in 1u8..=118 {
            let m = element_metadata(z).unwrap_or_else(|| panic!("missing metadata for Z={z}"));
            assert_eq!(m.atomic_number, z);
        }
    }

    #[test]
    fn test_z_0_and_z_119_return_none() {
        assert!(element_metadata(0).is_none());
        assert!(element_metadata(119).is_none());
    }

    #[test]
    fn test_known_element_values() {
        let h = element_metadata(1).unwrap();
        assert_eq!(h.symbol, "H");
        assert!((h.atomic_mass - 1.008).abs() < 1e-6);
        assert_eq!(h.period, 1);
        assert_eq!(h.block, Block::S);

        let fe = element_metadata(26).unwrap();
        assert_eq!(fe.symbol, "Fe");
        assert_eq!(fe.period, 4);
        assert_eq!(fe.block, Block::D);

        let u = element_metadata(92).unwrap();
        assert_eq!(u.symbol, "U");
        assert_eq!(u.period, 7);
        assert_eq!(u.block, Block::F);
    }

    #[test]
    fn test_atomic_masses_increase_monotonically_within_a_period() {
        // Not a strict law across the whole table (isotope-abundance effects
        // cause a few well-known local inversions, e.g. Ar>K, Co>Ni, Te>I --
        // real chemistry, not a data error), but within Period 2 (Li-Ne) and
        // Period 3 (Na-Ar) it should hold as a basic sanity check.
        let period2: Vec<f64> = (3u8..=10)
            .map(|z| element_metadata(z).unwrap().atomic_mass)
            .collect();
        for w in period2.windows(2) {
            assert!(w[1] > w[0], "Period 2 mass not increasing: {:?}", period2);
        }
    }
}
