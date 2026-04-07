// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Training data: experimental bandgaps for ~60 semiconductors.
//!
//! Sources:
//! - Zhuo et al. (2018), J. Phys. Chem. Lett. 9, 1668 (PBE + HSE06 benchmarks)
//! - Borlido et al. (2019), J. Chem. Theory Comput. 15, 5069
//! - NIST / Madelung, Semiconductors: Data Handbook (3rd ed.)

use serde::{Deserialize, Serialize};

/// Crystal system classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrystalSystem {
    Cubic,
    Hexagonal,
    Tetragonal,
    Orthorhombic,
    Monoclinic,
    Rhombohedral,
    Triclinic,
}

/// A training data point: composition + crystal system + experimental bandgap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandgapDataPoint {
    /// Composition as (atomic_number, stoichiometric_fraction) pairs.
    /// Fractions sum to 1.0.
    pub composition: Vec<(u8, f64)>,
    /// Crystal system.
    pub crystal: CrystalSystem,
    /// Experimental bandgap in eV.
    pub bandgap_exp: f64,
    /// Material name/formula for display.
    pub name: String,
}

/// Load the curated training dataset (~60 well-characterized semiconductors).
///
/// All experimental bandgaps from room-temperature measurements unless noted.
/// Compositions normalized to sum to 1.0.
pub fn load_training_data() -> Vec<BandgapDataPoint> {
    vec![
        // --- Elemental semiconductors ---
        dp("C (diamond)", &[(6, 1.0)], CrystalSystem::Cubic, 5.47),
        dp("Si", &[(14, 1.0)], CrystalSystem::Cubic, 1.12),
        dp("Ge", &[(32, 1.0)], CrystalSystem::Cubic, 0.66),
        dp("alpha-Sn", &[(50, 1.0)], CrystalSystem::Cubic, 0.08),

        // --- III-V semiconductors ---
        dp("BN (cubic)", &[(5, 0.5), (7, 0.5)], CrystalSystem::Cubic, 6.25),
        dp("BN (hex)", &[(5, 0.5), (7, 0.5)], CrystalSystem::Hexagonal, 5.97),
        dp("BP", &[(5, 0.5), (15, 0.5)], CrystalSystem::Cubic, 2.0),
        dp("BAs", &[(5, 0.5), (33, 0.5)], CrystalSystem::Cubic, 1.46),
        dp("AlN", &[(13, 0.5), (7, 0.5)], CrystalSystem::Hexagonal, 6.13),
        dp("AlP", &[(13, 0.5), (15, 0.5)], CrystalSystem::Cubic, 2.45),
        dp("AlAs", &[(13, 0.5), (33, 0.5)], CrystalSystem::Cubic, 2.16),
        dp("AlSb", &[(13, 0.5), (51, 0.5)], CrystalSystem::Cubic, 1.62),
        dp("GaN", &[(31, 0.5), (7, 0.5)], CrystalSystem::Hexagonal, 3.39),
        dp("GaP", &[(31, 0.5), (15, 0.5)], CrystalSystem::Cubic, 2.26),
        dp("GaAs", &[(31, 0.5), (33, 0.5)], CrystalSystem::Cubic, 1.42),
        dp("GaSb", &[(31, 0.5), (51, 0.5)], CrystalSystem::Cubic, 0.73),
        dp("InN", &[(49, 0.5), (7, 0.5)], CrystalSystem::Hexagonal, 0.69),
        dp("InP", &[(49, 0.5), (15, 0.5)], CrystalSystem::Cubic, 1.35),
        dp("InAs", &[(49, 0.5), (33, 0.5)], CrystalSystem::Cubic, 0.36),
        dp("InSb", &[(49, 0.5), (51, 0.5)], CrystalSystem::Cubic, 0.17),

        // --- II-VI semiconductors ---
        dp("ZnO", &[(30, 0.5), (8, 0.5)], CrystalSystem::Hexagonal, 3.37),
        dp("ZnS", &[(30, 0.5), (16, 0.5)], CrystalSystem::Cubic, 3.68),
        dp("ZnSe", &[(30, 0.5), (34, 0.5)], CrystalSystem::Cubic, 2.70),
        dp("ZnTe", &[(30, 0.5), (52, 0.5)], CrystalSystem::Cubic, 2.26),
        dp("CdO", &[(48, 0.5), (8, 0.5)], CrystalSystem::Cubic, 2.28),
        dp("CdS", &[(48, 0.5), (16, 0.5)], CrystalSystem::Hexagonal, 2.42),
        dp("CdSe", &[(48, 0.5), (34, 0.5)], CrystalSystem::Hexagonal, 1.73),
        dp("CdTe", &[(48, 0.5), (52, 0.5)], CrystalSystem::Cubic, 1.49),
        dp("MgO", &[(12, 0.5), (8, 0.5)], CrystalSystem::Cubic, 7.83),
        dp("MgS", &[(12, 0.5), (16, 0.5)], CrystalSystem::Cubic, 4.45),
        dp("CaO", &[(20, 0.5), (8, 0.5)], CrystalSystem::Cubic, 7.03),
        dp("BaO", &[(56, 0.5), (8, 0.5)], CrystalSystem::Cubic, 3.98),
        dp("SrO", &[(38, 0.5), (8, 0.5)], CrystalSystem::Cubic, 5.90),

        // --- IV-IV and IV-VI ---
        dp("SiC (3C)", &[(14, 0.5), (6, 0.5)], CrystalSystem::Cubic, 2.36),
        dp("SiC (4H)", &[(14, 0.5), (6, 0.5)], CrystalSystem::Hexagonal, 3.26),
        dp("PbS", &[(82, 0.5), (16, 0.5)], CrystalSystem::Cubic, 0.37),
        dp("PbSe", &[(82, 0.5), (34, 0.5)], CrystalSystem::Cubic, 0.27),
        dp("PbTe", &[(82, 0.5), (52, 0.5)], CrystalSystem::Cubic, 0.31),
        dp("SnO2", &[(50, 1.0/3.0), (8, 2.0/3.0)], CrystalSystem::Tetragonal, 3.60),
        dp("SnS", &[(50, 0.5), (16, 0.5)], CrystalSystem::Orthorhombic, 1.07),

        // --- Transition metal oxides ---
        dp("TiO2 (rutile)", &[(22, 1.0/3.0), (8, 2.0/3.0)], CrystalSystem::Tetragonal, 3.03),
        dp("TiO2 (anatase)", &[(22, 1.0/3.0), (8, 2.0/3.0)], CrystalSystem::Tetragonal, 3.20),
        dp("Cu2O", &[(29, 2.0/3.0), (8, 1.0/3.0)], CrystalSystem::Cubic, 2.17),
        dp("Fe2O3", &[(26, 0.4), (8, 0.6)], CrystalSystem::Rhombohedral, 2.20),
        dp("NiO", &[(28, 0.5), (8, 0.5)], CrystalSystem::Cubic, 3.70),
        dp("WO3", &[(74, 0.25), (8, 0.75)], CrystalSystem::Monoclinic, 2.62),
        dp("V2O5", &[(23, 2.0/7.0), (8, 5.0/7.0)], CrystalSystem::Orthorhombic, 2.30),

        // --- Halides and other ---
        dp("NaCl", &[(11, 0.5), (17, 0.5)], CrystalSystem::Cubic, 8.50),
        dp("KCl", &[(19, 0.5), (17, 0.5)], CrystalSystem::Cubic, 8.40),
        dp("LiF", &[(3, 0.5), (9, 0.5)], CrystalSystem::Cubic, 13.60),
        dp("LiCl", &[(3, 0.5), (17, 0.5)], CrystalSystem::Cubic, 9.40),
        dp("CaF2", &[(20, 1.0/3.0), (9, 2.0/3.0)], CrystalSystem::Cubic, 11.80),
        dp("BaF2", &[(56, 1.0/3.0), (9, 2.0/3.0)], CrystalSystem::Cubic, 9.10),

        // --- Chalcopyrites ---
        dp("CuInSe2", &[(29, 0.25), (49, 0.25), (34, 0.5)], CrystalSystem::Tetragonal, 1.01),
        dp("CuGaSe2", &[(29, 0.25), (31, 0.25), (34, 0.5)], CrystalSystem::Tetragonal, 1.68),
        dp("CuInS2", &[(29, 0.25), (49, 0.25), (16, 0.5)], CrystalSystem::Tetragonal, 1.53),
        dp("AgGaSe2", &[(47, 0.25), (31, 0.25), (34, 0.5)], CrystalSystem::Tetragonal, 1.83),

        // --- Perovskites ---
        dp("SrTiO3", &[(38, 0.2), (22, 0.2), (8, 0.6)], CrystalSystem::Cubic, 3.25),
        dp("BaTiO3", &[(56, 0.2), (22, 0.2), (8, 0.6)], CrystalSystem::Tetragonal, 3.20),
        dp("KNbO3", &[(19, 0.2), (41, 0.2), (8, 0.6)], CrystalSystem::Orthorhombic, 3.10),
    ]
}

/// Helper to build a data point.
fn dp(name: &str, comp: &[(u8, f64)], crystal: CrystalSystem, bandgap: f64) -> BandgapDataPoint {
    BandgapDataPoint {
        composition: comp.to_vec(),
        crystal,
        bandgap_exp: bandgap,
        name: name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_data_loads() {
        let data = load_training_data();
        assert!(data.len() >= 55, "Expected >= 55 materials, got {}", data.len());
    }

    #[test]
    fn test_compositions_valid() {
        let data = load_training_data();
        for d in &data {
            let sum: f64 = d.composition.iter().map(|(_, f)| f).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "{}: composition fractions sum to {}, expected 1.0",
                d.name, sum
            );
            assert!(d.bandgap_exp > 0.0, "{}: bandgap must be positive", d.name);
        }
    }
}
