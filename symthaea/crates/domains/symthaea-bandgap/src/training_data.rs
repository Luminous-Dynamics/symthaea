// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Semiconductor Bandgap Training Dataset
//!
//! Hardcoded experimental bandgap values for ~200 well-known binary, ternary,
//! and select quaternary semiconductors. Analogous to AME2020 in the nuclear
//! crate — a curated reference table that anchors ML residual learning.
//!
//! Sources:
//! - Madelung, O. (2004). *Semiconductors: Data Handbook*. Springer.
//! - Zhuo, Y. et al. (2018). J. Phys. Chem. Lett. 9, 1668.
//! - Borlido, P. et al. (2019). J. Chem. Theory Comput. 15, 5069.
//! - Haynes, W.M. (2016). *CRC Handbook of Chemistry and Physics*, 97th ed.

use serde::{Deserialize, Serialize};

// ─── Types ───────────────────────────────────────────────────────────────────

/// Crystal system classification (7 systems + Unknown).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrystalSystem {
    Cubic,
    Hexagonal,
    Tetragonal,
    Orthorhombic,
    Monoclinic,
    Trigonal,
    Triclinic,
    Unknown,
}

impl CrystalSystem {
    /// Ordinal encoding for ML features (0-based).
    pub fn ordinal(self) -> u8 {
        match self {
            Self::Cubic => 0,
            Self::Hexagonal => 1,
            Self::Tetragonal => 2,
            Self::Orthorhombic => 3,
            Self::Monoclinic => 4,
            Self::Trigonal => 5,
            Self::Triclinic => 6,
            Self::Unknown => 7,
        }
    }
}

/// A single material entry in the training dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialEntry {
    /// Chemical formula (e.g. "GaAs", "SrTiO3").
    pub formula: &'static str,
    /// Composition as (atomic_number, molar_fraction) pairs. Fractions sum to 1.0.
    pub composition: Vec<(u8, f64)>,
    /// Experimental bandgap in eV (room temperature unless noted).
    pub experimental_gap: f64,
    /// Crystal system.
    pub crystal_system: CrystalSystem,
}

// Backward-compatible aliases for ml_bandgap.rs
/// Type alias for backward compatibility.
pub type BandgapDataPoint = MaterialEntry;

impl MaterialEntry {
    /// Backward-compatible accessor: returns `experimental_gap`.
    #[inline]
    pub fn bandgap_exp(&self) -> f64 {
        self.experimental_gap
    }
    /// Backward-compatible accessor: returns `crystal_system`.
    #[inline]
    pub fn crystal(&self) -> &CrystalSystem {
        &self.crystal_system
    }
    /// Backward-compatible accessor: returns `formula` as String.
    #[inline]
    pub fn name(&self) -> String {
        self.formula.to_string()
    }
}

// ─── Helper ──────────────────────────────────────────────────────────────────

/// Shorthand constructor for inline use.
fn me(
    formula: &'static str,
    composition: &[(u8, f64)],
    experimental_gap: f64,
    crystal_system: CrystalSystem,
) -> MaterialEntry {
    MaterialEntry {
        formula,
        composition: composition.to_vec(),
        experimental_gap,
        crystal_system,
    }
}

use CrystalSystem::*;

// Atomic numbers (for readability)
const H: u8 = 1;
const LI: u8 = 3;
const BE: u8 = 4;
const B: u8 = 5;
const C: u8 = 6;
const N: u8 = 7;
const O: u8 = 8;
const F: u8 = 9;
const NA: u8 = 11;
const MG: u8 = 12;
const AL: u8 = 13;
const SI: u8 = 14;
const P: u8 = 15;
const S: u8 = 16;
const CL: u8 = 17;
const K: u8 = 19;
const CA: u8 = 20;
const SC: u8 = 21;
const TI: u8 = 22;
const V: u8 = 23;
const CR: u8 = 24;
const MN: u8 = 25;
const FE: u8 = 26;
const CO: u8 = 27;
const NI: u8 = 28;
const CU: u8 = 29;
const ZN: u8 = 30;
const GA: u8 = 31;
const GE: u8 = 32;
const AS: u8 = 33;
const SE: u8 = 34;
const BR: u8 = 35;
const RB: u8 = 37;
const SR: u8 = 38;
const Y: u8 = 39;
const ZR: u8 = 40;
const NB: u8 = 41;
const MO: u8 = 42;
const AG: u8 = 47;
const CD: u8 = 48;
const IN: u8 = 49;
const SN: u8 = 50;
const SB: u8 = 51;
const TE: u8 = 52;
const I: u8 = 53;
const CS: u8 = 55;
const BA: u8 = 56;
const LA: u8 = 57;
const CE: u8 = 58;
const PR: u8 = 59;
const ND: u8 = 60;
const GD: u8 = 64;
const HF: u8 = 72;
const TA: u8 = 73;
const W: u8 = 74;
const RE: u8 = 75;
const HG: u8 = 80;
const TL: u8 = 81;
const PB: u8 = 82;
const BI: u8 = 83;

// ─── Dataset ─────────────────────────────────────────────────────────────────

/// Load the complete semiconductor bandgap training dataset (~200 materials).
///
/// Each entry contains experimental bandgap values (eV, room temperature)
/// from standard references. Composition vectors are normalised to sum = 1.0.
pub fn load_training_data() -> Vec<MaterialEntry> {
    vec![
        // ══════════════════════════════════════════════════════════════════
        // Group IV Elemental & Binary
        // ══════════════════════════════════════════════════════════════════
        me("Si", &[(SI, 1.0)], 1.17, Cubic),
        me("Ge", &[(GE, 1.0)], 0.66, Cubic),
        me("C-dia", &[(C, 1.0)], 5.47, Cubic),
        me("alpha-Sn", &[(SN, 1.0)], 0.08, Cubic),
        me("SiC-4H", &[(SI, 0.5), (C, 0.5)], 3.26, Hexagonal),
        me("SiC-6H", &[(SI, 0.5), (C, 0.5)], 3.02, Hexagonal),
        me("SiC-3C", &[(SI, 0.5), (C, 0.5)], 2.36, Cubic),
        // ══════════════════════════════════════════════════════════════════
        // III-V Semiconductors
        // ══════════════════════════════════════════════════════════════════
        me("GaAs", &[(GA, 0.5), (AS, 0.5)], 1.42, Cubic),
        me("GaN", &[(GA, 0.5), (N, 0.5)], 3.39, Hexagonal),
        me("InP", &[(IN, 0.5), (P, 0.5)], 1.34, Cubic),
        me("AlN", &[(AL, 0.5), (N, 0.5)], 6.20, Hexagonal),
        me("InN", &[(IN, 0.5), (N, 0.5)], 0.69, Hexagonal),
        me("GaP", &[(GA, 0.5), (P, 0.5)], 2.26, Cubic),
        me("InAs", &[(IN, 0.5), (AS, 0.5)], 0.35, Cubic),
        me("AlAs", &[(AL, 0.5), (AS, 0.5)], 2.15, Cubic),
        me("GaSb", &[(GA, 0.5), (SB, 0.5)], 0.73, Cubic),
        me("InSb", &[(IN, 0.5), (SB, 0.5)], 0.17, Cubic),
        me("BN-c", &[(B, 0.5), (N, 0.5)], 6.00, Cubic),
        me("AlP", &[(AL, 0.5), (P, 0.5)], 2.45, Cubic),
        me("AlSb", &[(AL, 0.5), (SB, 0.5)], 1.62, Cubic),
        me("BP", &[(B, 0.5), (P, 0.5)], 2.00, Cubic),
        me("BAs", &[(B, 0.5), (AS, 0.5)], 1.46, Cubic),
        // ══════════════════════════════════════════════════════════════════
        // II-VI Semiconductors
        // ══════════════════════════════════════════════════════════════════
        me("ZnO", &[(ZN, 0.5), (O, 0.5)], 3.37, Hexagonal),
        me("ZnS", &[(ZN, 0.5), (S, 0.5)], 3.68, Cubic),
        me("ZnSe", &[(ZN, 0.5), (SE, 0.5)], 2.70, Cubic),
        me("ZnTe", &[(ZN, 0.5), (TE, 0.5)], 2.26, Cubic),
        me("CdS", &[(CD, 0.5), (S, 0.5)], 2.42, Hexagonal),
        me("CdSe", &[(CD, 0.5), (SE, 0.5)], 1.73, Hexagonal),
        me("CdTe", &[(CD, 0.5), (TE, 0.5)], 1.44, Cubic),
        me("CdO", &[(CD, 0.5), (O, 0.5)], 2.16, Cubic),
        me("MgO", &[(MG, 0.5), (O, 0.5)], 7.83, Cubic),
        me("BaO", &[(BA, 0.5), (O, 0.5)], 3.90, Cubic),
        me("MgS", &[(MG, 0.5), (S, 0.5)], 5.40, Cubic),
        me("MgSe", &[(MG, 0.5), (SE, 0.5)], 4.05, Cubic),
        me("MgTe", &[(MG, 0.5), (TE, 0.5)], 3.60, Cubic),
        me("CaO", &[(CA, 0.5), (O, 0.5)], 7.03, Cubic),
        me("SrO", &[(SR, 0.5), (O, 0.5)], 5.90, Cubic),
        me("BaS", &[(BA, 0.5), (S, 0.5)], 3.88, Cubic),
        me("BaSe", &[(BA, 0.5), (SE, 0.5)], 3.58, Cubic),
        me("BaTe", &[(BA, 0.5), (TE, 0.5)], 3.08, Cubic),
        me("BeO", &[(BE, 0.5), (O, 0.5)], 10.6, Hexagonal),
        me("BeS", &[(BE, 0.5), (S, 0.5)], 5.50, Cubic),
        me("BeSe", &[(BE, 0.5), (SE, 0.5)], 4.50, Cubic),
        me("BeTe", &[(BE, 0.5), (TE, 0.5)], 2.80, Cubic),
        me("HgS", &[(HG, 0.5), (S, 0.5)], 2.10, Trigonal),
        me("HgSe", &[(HG, 0.5), (SE, 0.5)], 0.08, Cubic),
        me("HgTe", &[(HG, 0.5), (TE, 0.5)], 0.00, Cubic), // semimetal
        // ══════════════════════════════════════════════════════════════════
        // Chalcogenides (Cu, Sn, Ti, Fe, W, V, Mo, Cr, Bi, Sb, In)
        // ══════════════════════════════════════════════════════════════════
        me("Cu2O", &[(CU, 2.0 / 3.0), (O, 1.0 / 3.0)], 2.17, Cubic),
        me("CuO", &[(CU, 0.5), (O, 0.5)], 1.20, Monoclinic),
        me("SnO2", &[(SN, 1.0 / 3.0), (O, 2.0 / 3.0)], 3.60, Tetragonal),
        me(
            "TiO2-rut",
            &[(TI, 1.0 / 3.0), (O, 2.0 / 3.0)],
            3.03,
            Tetragonal,
        ),
        me(
            "TiO2-ana",
            &[(TI, 1.0 / 3.0), (O, 2.0 / 3.0)],
            3.20,
            Tetragonal,
        ),
        me("Fe2O3", &[(FE, 2.0 / 5.0), (O, 3.0 / 5.0)], 2.10, Trigonal),
        me("WO3", &[(W, 0.25), (O, 0.75)], 2.60, Monoclinic),
        me(
            "V2O5",
            &[(V, 2.0 / 7.0), (O, 5.0 / 7.0)],
            2.30,
            Orthorhombic,
        ),
        me("MoO3", &[(MO, 0.25), (O, 0.75)], 3.00, Orthorhombic),
        me("Cr2O3", &[(CR, 2.0 / 5.0), (O, 3.0 / 5.0)], 3.40, Trigonal),
        me("CuS", &[(CU, 0.5), (S, 0.5)], 1.20, Hexagonal),
        me("Cu2S", &[(CU, 2.0 / 3.0), (S, 1.0 / 3.0)], 1.21, Monoclinic),
        me("SnS", &[(SN, 0.5), (S, 0.5)], 1.07, Orthorhombic),
        me("SnS2", &[(SN, 1.0 / 3.0), (S, 2.0 / 3.0)], 2.18, Hexagonal),
        me("FeS2", &[(FE, 1.0 / 3.0), (S, 2.0 / 3.0)], 0.95, Cubic),
        me("MoS2", &[(MO, 1.0 / 3.0), (S, 2.0 / 3.0)], 1.23, Hexagonal),
        me("WS2", &[(W, 1.0 / 3.0), (S, 2.0 / 3.0)], 1.35, Hexagonal),
        me(
            "Bi2S3",
            &[(BI, 2.0 / 5.0), (S, 3.0 / 5.0)],
            1.30,
            Orthorhombic,
        ),
        me(
            "Sb2S3",
            &[(SB, 2.0 / 5.0), (S, 3.0 / 5.0)],
            1.72,
            Orthorhombic,
        ),
        me(
            "In2S3",
            &[(IN, 2.0 / 5.0), (S, 3.0 / 5.0)],
            2.10,
            Tetragonal,
        ),
        // ══════════════════════════════════════════════════════════════════
        // Halides (Alkali, Silver, Copper)
        // ══════════════════════════════════════════════════════════════════
        me("NaCl", &[(NA, 0.5), (CL, 0.5)], 8.50, Cubic),
        me("KCl", &[(K, 0.5), (CL, 0.5)], 8.40, Cubic),
        me("LiF", &[(LI, 0.5), (F, 0.5)], 13.6, Cubic),
        me("NaF", &[(NA, 0.5), (F, 0.5)], 11.5, Cubic),
        me("CaF2", &[(CA, 1.0 / 3.0), (F, 2.0 / 3.0)], 11.8, Cubic),
        me("BaF2", &[(BA, 1.0 / 3.0), (F, 2.0 / 3.0)], 9.10, Cubic),
        me("AgCl", &[(AG, 0.5), (CL, 0.5)], 3.25, Cubic),
        me("AgBr", &[(AG, 0.5), (BR, 0.5)], 2.70, Cubic),
        me("LiCl", &[(LI, 0.5), (CL, 0.5)], 9.40, Cubic),
        me("KBr", &[(K, 0.5), (BR, 0.5)], 7.40, Cubic),
        me("KI", &[(K, 0.5), (I, 0.5)], 6.20, Cubic),
        me("NaBr", &[(NA, 0.5), (BR, 0.5)], 7.10, Cubic),
        me("NaI", &[(NA, 0.5), (I, 0.5)], 5.90, Cubic),
        me("RbCl", &[(RB, 0.5), (CL, 0.5)], 8.20, Cubic),
        me("RbBr", &[(RB, 0.5), (BR, 0.5)], 7.20, Cubic),
        me("CsCl", &[(CS, 0.5), (CL, 0.5)], 8.30, Cubic),
        me("CsBr", &[(CS, 0.5), (BR, 0.5)], 7.30, Cubic),
        me("CsI", &[(CS, 0.5), (I, 0.5)], 6.20, Cubic),
        me("LiBr", &[(LI, 0.5), (BR, 0.5)], 7.60, Cubic),
        me("LiI", &[(LI, 0.5), (I, 0.5)], 6.00, Cubic),
        // ══════════════════════════════════════════════════════════════════
        // Perovskites (Oxide & Halide)
        // ══════════════════════════════════════════════════════════════════
        me("SrTiO3", &[(SR, 0.2), (TI, 0.2), (O, 0.6)], 3.25, Cubic),
        me(
            "BaTiO3",
            &[(BA, 0.2), (TI, 0.2), (O, 0.6)],
            3.20,
            Tetragonal,
        ),
        me(
            "PbTiO3",
            &[(PB, 0.2), (TI, 0.2), (O, 0.6)],
            3.40,
            Tetragonal,
        ),
        me(
            "CsPbBr3",
            &[(CS, 0.2), (PB, 0.2), (BR, 0.6)],
            2.30,
            Orthorhombic,
        ),
        me(
            "MAPbI3",
            &[
                (C, 1.0 / 12.0),
                (N, 1.0 / 12.0),
                (H, 6.0 / 12.0),
                (PB, 1.0 / 12.0),
                (I, 3.0 / 12.0),
            ],
            1.55,
            Tetragonal,
        ),
        me(
            "KNbO3",
            &[(K, 0.2), (NB, 0.2), (O, 0.6)],
            3.30,
            Orthorhombic,
        ),
        me(
            "NaNbO3",
            &[(NA, 0.2), (NB, 0.2), (O, 0.6)],
            3.40,
            Orthorhombic,
        ),
        me("LaAlO3", &[(LA, 0.2), (AL, 0.2), (O, 0.6)], 5.60, Trigonal),
        me("KTaO3", &[(K, 0.2), (TA, 0.2), (O, 0.6)], 3.60, Cubic),
        me("CsPbCl3", &[(CS, 0.2), (PB, 0.2), (CL, 0.6)], 3.00, Cubic),
        me(
            "CsSnI3",
            &[(CS, 0.2), (SN, 0.2), (I, 0.6)],
            1.30,
            Orthorhombic,
        ),
        // ══════════════════════════════════════════════════════════════════
        // Nitrides
        // ══════════════════════════════════════════════════════════════════
        me("Si3N4", &[(SI, 3.0 / 7.0), (N, 4.0 / 7.0)], 5.30, Hexagonal),
        me("BN-hex", &[(B, 0.5), (N, 0.5)], 5.97, Hexagonal),
        me("ScN", &[(SC, 0.5), (N, 0.5)], 0.90, Cubic),
        me("Ge3N4", &[(GE, 3.0 / 7.0), (N, 4.0 / 7.0)], 3.20, Hexagonal),
        me(
            "Ta3N5",
            &[(TA, 3.0 / 8.0), (N, 5.0 / 8.0)],
            2.10,
            Orthorhombic,
        ),
        // ══════════════════════════════════════════════════════════════════
        // Phosphides & Arsenides (beyond III-V)
        // ══════════════════════════════════════════════════════════════════
        me("ZnP2", &[(ZN, 1.0 / 3.0), (P, 2.0 / 3.0)], 2.18, Tetragonal),
        me(
            "Zn3P2",
            &[(ZN, 3.0 / 5.0), (P, 2.0 / 5.0)],
            1.50,
            Tetragonal,
        ),
        me("CdP2", &[(CD, 1.0 / 3.0), (P, 2.0 / 3.0)], 1.92, Tetragonal),
        me(
            "Cd3P2",
            &[(CD, 3.0 / 5.0), (P, 2.0 / 5.0)],
            0.50,
            Tetragonal,
        ),
        me(
            "GeAs2",
            &[(GE, 1.0 / 3.0), (AS, 2.0 / 3.0)],
            0.57,
            Monoclinic,
        ),
        me(
            "ZnAs2",
            &[(ZN, 1.0 / 3.0), (AS, 2.0 / 3.0)],
            0.93,
            Monoclinic,
        ),
        // ══════════════════════════════════════════════════════════════════
        // Tellurides (Thermoelectric & Topological)
        // ══════════════════════════════════════════════════════════════════
        me("PbTe", &[(PB, 0.5), (TE, 0.5)], 0.31, Cubic),
        me(
            "Bi2Te3",
            &[(BI, 2.0 / 5.0), (TE, 3.0 / 5.0)],
            0.15,
            Trigonal,
        ),
        me("SnTe", &[(SN, 0.5), (TE, 0.5)], 0.18, Cubic),
        me("PbSe", &[(PB, 0.5), (SE, 0.5)], 0.26, Cubic),
        me("PbS", &[(PB, 0.5), (S, 0.5)], 0.37, Cubic),
        me(
            "Sb2Te3",
            &[(SB, 2.0 / 5.0), (TE, 3.0 / 5.0)],
            0.21,
            Trigonal,
        ),
        me("GeTe", &[(GE, 0.5), (TE, 0.5)], 0.36, Trigonal),
        me(
            "Bi2Se3",
            &[(BI, 2.0 / 5.0), (SE, 3.0 / 5.0)],
            0.30,
            Trigonal,
        ),
        me(
            "Ag2Te",
            &[(AG, 2.0 / 3.0), (TE, 1.0 / 3.0)],
            0.67,
            Monoclinic,
        ),
        me(
            "Cu2Te",
            &[(CU, 2.0 / 3.0), (TE, 1.0 / 3.0)],
            1.04,
            Hexagonal,
        ),
        // ══════════════════════════════════════════════════════════════════
        // Transition Metal Oxides
        // ══════════════════════════════════════════════════════════════════
        me("NiO", &[(NI, 0.5), (O, 0.5)], 3.70, Cubic),
        me("CoO", &[(CO, 0.5), (O, 0.5)], 2.40, Cubic),
        me("MnO", &[(MN, 0.5), (O, 0.5)], 3.60, Cubic),
        me("FeO", &[(FE, 0.5), (O, 0.5)], 2.40, Cubic),
        me(
            "CuAlO2",
            &[(CU, 0.25), (AL, 0.25), (O, 0.5)],
            3.50,
            Trigonal,
        ),
        me(
            "ZnFe2O4",
            &[(ZN, 1.0 / 7.0), (FE, 2.0 / 7.0), (O, 4.0 / 7.0)],
            1.90,
            Cubic,
        ),
        me(
            "CoFe2O4",
            &[(CO, 1.0 / 7.0), (FE, 2.0 / 7.0), (O, 4.0 / 7.0)],
            1.20,
            Cubic,
        ),
        me(
            "NiFe2O4",
            &[(NI, 1.0 / 7.0), (FE, 2.0 / 7.0), (O, 4.0 / 7.0)],
            1.60,
            Cubic,
        ),
        me(
            "MnFe2O4",
            &[(MN, 1.0 / 7.0), (FE, 2.0 / 7.0), (O, 4.0 / 7.0)],
            1.00,
            Cubic,
        ),
        // ══════════════════════════════════════════════════════════════════
        // Wide-gap Oxides & Insulators
        // ══════════════════════════════════════════════════════════════════
        me("Al2O3", &[(AL, 2.0 / 5.0), (O, 3.0 / 5.0)], 8.80, Trigonal),
        me("SiO2", &[(SI, 1.0 / 3.0), (O, 2.0 / 3.0)], 8.90, Trigonal),
        me("HfO2", &[(HF, 1.0 / 3.0), (O, 2.0 / 3.0)], 5.80, Monoclinic),
        me("ZrO2", &[(ZR, 1.0 / 3.0), (O, 2.0 / 3.0)], 5.00, Monoclinic),
        me("La2O3", &[(LA, 2.0 / 5.0), (O, 3.0 / 5.0)], 5.50, Hexagonal),
        me("CeO2", &[(CE, 1.0 / 3.0), (O, 2.0 / 3.0)], 3.19, Cubic),
        me(
            "Ga2O3",
            &[(GA, 2.0 / 5.0), (O, 3.0 / 5.0)],
            4.80,
            Monoclinic,
        ),
        me("Y2O3", &[(Y, 2.0 / 5.0), (O, 3.0 / 5.0)], 5.60, Cubic),
        me("Sc2O3", &[(SC, 2.0 / 5.0), (O, 3.0 / 5.0)], 6.00, Cubic),
        me(
            "Nb2O5",
            &[(NB, 2.0 / 7.0), (O, 5.0 / 7.0)],
            3.40,
            Monoclinic,
        ),
        me(
            "Ta2O5",
            &[(TA, 2.0 / 7.0), (O, 5.0 / 7.0)],
            4.00,
            Orthorhombic,
        ),
        me("GeO2", &[(GE, 1.0 / 3.0), (O, 2.0 / 3.0)], 5.35, Tetragonal),
        me("SnO", &[(SN, 0.5), (O, 0.5)], 2.70, Tetragonal),
        me("PbO", &[(PB, 0.5), (O, 0.5)], 1.90, Tetragonal),
        me(
            "Bi2O3",
            &[(BI, 2.0 / 5.0), (O, 3.0 / 5.0)],
            2.80,
            Monoclinic,
        ),
        me("In2O3", &[(IN, 2.0 / 5.0), (O, 3.0 / 5.0)], 2.90, Cubic),
        me("Pr2O3", &[(PR, 2.0 / 5.0), (O, 3.0 / 5.0)], 3.90, Hexagonal),
        me("Nd2O3", &[(ND, 2.0 / 5.0), (O, 3.0 / 5.0)], 4.70, Hexagonal),
        me("Gd2O3", &[(GD, 2.0 / 5.0), (O, 3.0 / 5.0)], 5.30, Cubic),
        // ══════════════════════════════════════════════════════════════════
        // Selenides
        // ══════════════════════════════════════════════════════════════════
        me(
            "In2Se3",
            &[(IN, 2.0 / 5.0), (SE, 3.0 / 5.0)],
            1.30,
            Hexagonal,
        ),
        me("Ga2Se3", &[(GA, 2.0 / 5.0), (SE, 3.0 / 5.0)], 2.00, Cubic),
        me(
            "Sb2Se3",
            &[(SB, 2.0 / 5.0), (SE, 3.0 / 5.0)],
            1.12,
            Orthorhombic,
        ),
        me("CuSe", &[(CU, 0.5), (SE, 0.5)], 1.10, Hexagonal),
        me(
            "Ag2Se",
            &[(AG, 2.0 / 3.0), (SE, 1.0 / 3.0)],
            0.15,
            Orthorhombic,
        ),
        me("GeSe", &[(GE, 0.5), (SE, 0.5)], 1.10, Orthorhombic),
        me(
            "GeSe2",
            &[(GE, 1.0 / 3.0), (SE, 2.0 / 3.0)],
            2.70,
            Monoclinic,
        ),
        me("SnSe", &[(SN, 0.5), (SE, 0.5)], 0.90, Orthorhombic),
        me(
            "SnSe2",
            &[(SN, 1.0 / 3.0), (SE, 2.0 / 3.0)],
            1.00,
            Hexagonal,
        ),
        me(
            "FeSe2",
            &[(FE, 1.0 / 3.0), (SE, 2.0 / 3.0)],
            1.00,
            Orthorhombic,
        ),
        me(
            "MoSe2",
            &[(MO, 1.0 / 3.0), (SE, 2.0 / 3.0)],
            1.10,
            Hexagonal,
        ),
        me("WSe2", &[(W, 1.0 / 3.0), (SE, 2.0 / 3.0)], 1.20, Hexagonal),
        // ══════════════════════════════════════════════════════════════════
        // Sulfides (beyond II-VI)
        // ══════════════════════════════════════════════════════════════════
        me(
            "Ga2S3",
            &[(GA, 2.0 / 5.0), (S, 3.0 / 5.0)],
            2.50,
            Monoclinic,
        ),
        me("GeS", &[(GE, 0.5), (S, 0.5)], 1.55, Orthorhombic),
        me("GeS2", &[(GE, 1.0 / 3.0), (S, 2.0 / 3.0)], 3.20, Monoclinic),
        me(
            "As2S3",
            &[(AS, 2.0 / 5.0), (S, 3.0 / 5.0)],
            2.40,
            Monoclinic,
        ),
        me("Tl2S", &[(TL, 2.0 / 3.0), (S, 1.0 / 3.0)], 1.00, Tetragonal),
        me("Ag2S", &[(AG, 2.0 / 3.0), (S, 1.0 / 3.0)], 0.92, Monoclinic),
        me("La2S3", &[(LA, 2.0 / 5.0), (S, 3.0 / 5.0)], 2.90, Cubic),
        me("Ce2S3", &[(CE, 2.0 / 5.0), (S, 3.0 / 5.0)], 2.10, Cubic),
        // ══════════════════════════════════════════════════════════════════
        // Ternary Chalcopyrites (I-III-VI2)
        // ══════════════════════════════════════════════════════════════════
        me(
            "CuGaS2",
            &[(CU, 0.25), (GA, 0.25), (S, 0.5)],
            2.43,
            Tetragonal,
        ),
        me(
            "CuGaSe2",
            &[(CU, 0.25), (GA, 0.25), (SE, 0.5)],
            1.68,
            Tetragonal,
        ),
        me(
            "CuInS2",
            &[(CU, 0.25), (IN, 0.25), (S, 0.5)],
            1.53,
            Tetragonal,
        ),
        me(
            "CuInSe2",
            &[(CU, 0.25), (IN, 0.25), (SE, 0.5)],
            1.04,
            Tetragonal,
        ),
        me(
            "CuInTe2",
            &[(CU, 0.25), (IN, 0.25), (TE, 0.5)],
            0.96,
            Tetragonal,
        ),
        me(
            "CuAlS2",
            &[(CU, 0.25), (AL, 0.25), (S, 0.5)],
            3.49,
            Tetragonal,
        ),
        me(
            "CuAlSe2",
            &[(CU, 0.25), (AL, 0.25), (SE, 0.5)],
            2.67,
            Tetragonal,
        ),
        me(
            "AgGaS2",
            &[(AG, 0.25), (GA, 0.25), (S, 0.5)],
            2.73,
            Tetragonal,
        ),
        me(
            "AgGaSe2",
            &[(AG, 0.25), (GA, 0.25), (SE, 0.5)],
            1.83,
            Tetragonal,
        ),
        me(
            "AgInS2",
            &[(AG, 0.25), (IN, 0.25), (S, 0.5)],
            1.87,
            Tetragonal,
        ),
        me(
            "AgInSe2",
            &[(AG, 0.25), (IN, 0.25), (SE, 0.5)],
            1.24,
            Tetragonal,
        ),
        me(
            "ZnSiP2",
            &[(ZN, 0.25), (SI, 0.25), (P, 0.5)],
            2.07,
            Tetragonal,
        ),
        me(
            "ZnGeP2",
            &[(ZN, 0.25), (GE, 0.25), (P, 0.5)],
            1.99,
            Tetragonal,
        ),
        me(
            "CdGeP2",
            &[(CD, 0.25), (GE, 0.25), (P, 0.5)],
            1.72,
            Tetragonal,
        ),
        // ══════════════════════════════════════════════════════════════════
        // Fluorides & Halides (additional)
        // ══════════════════════════════════════════════════════════════════
        me(
            "MgF2",
            &[(MG, 1.0 / 3.0), (F, 2.0 / 3.0)],
            10.80,
            Tetragonal,
        ),
        me("SrF2", &[(SR, 1.0 / 3.0), (F, 2.0 / 3.0)], 11.25, Cubic),
        me("LaF3", &[(LA, 0.25), (F, 0.75)], 10.40, Trigonal),
        me("AlF3", &[(AL, 0.25), (F, 0.75)], 10.80, Trigonal),
        me("CuBr", &[(CU, 0.5), (BR, 0.5)], 2.91, Cubic),
        me("CuCl", &[(CU, 0.5), (CL, 0.5)], 3.40, Cubic),
        me("CuI", &[(CU, 0.5), (I, 0.5)], 3.10, Cubic),
        me("AgI", &[(AG, 0.5), (I, 0.5)], 2.82, Hexagonal),
        // ══════════════════════════════════════════════════════════════════
        // Layered Materials & TMDCs
        // ══════════════════════════════════════════════════════════════════
        me("TiS2", &[(TI, 1.0 / 3.0), (S, 2.0 / 3.0)], 0.50, Hexagonal),
        me("HfS2", &[(HF, 1.0 / 3.0), (S, 2.0 / 3.0)], 1.96, Hexagonal),
        me(
            "HfSe2",
            &[(HF, 1.0 / 3.0), (SE, 2.0 / 3.0)],
            1.13,
            Hexagonal,
        ),
        me("ZrS2", &[(ZR, 1.0 / 3.0), (S, 2.0 / 3.0)], 1.68, Hexagonal),
        me(
            "ZrSe2",
            &[(ZR, 1.0 / 3.0), (SE, 2.0 / 3.0)],
            1.20,
            Hexagonal,
        ),
        me("ReS2", &[(RE, 1.0 / 3.0), (S, 2.0 / 3.0)], 1.35, Triclinic),
        me(
            "ReSe2",
            &[(RE, 1.0 / 3.0), (SE, 2.0 / 3.0)],
            1.15,
            Triclinic,
        ),
        // ══════════════════════════════════════════════════════════════════
        // Oxyhalides & Oxychalcogenides
        // ══════════════════════════════════════════════════════════════════
        me(
            "Bi2O2Se",
            &[(BI, 2.0 / 5.0), (O, 2.0 / 5.0), (SE, 1.0 / 5.0)],
            0.80,
            Tetragonal,
        ),
        me(
            "BiOCl",
            &[(BI, 1.0 / 3.0), (O, 1.0 / 3.0), (CL, 1.0 / 3.0)],
            3.40,
            Tetragonal,
        ),
        // ══════════════════════════════════════════════════════════════════
        // II-IV-V2 Pnictides
        // ══════════════════════════════════════════════════════════════════
        me(
            "ZnSnP2",
            &[(ZN, 0.25), (SN, 0.25), (P, 0.5)],
            1.66,
            Tetragonal,
        ),
        me(
            "CdSnP2",
            &[(CD, 0.25), (SN, 0.25), (P, 0.5)],
            1.17,
            Tetragonal,
        ),
        me(
            "ZnSnAs2",
            &[(ZN, 0.25), (SN, 0.25), (AS, 0.5)],
            0.73,
            Tetragonal,
        ),
        me(
            "CdSnAs2",
            &[(CD, 0.25), (SN, 0.25), (AS, 0.5)],
            0.26,
            Tetragonal,
        ),
        me(
            "CdGeAs2",
            &[(CD, 0.25), (GE, 0.25), (AS, 0.5)],
            0.57,
            Tetragonal,
        ),
        me(
            "ZnGeAs2",
            &[(ZN, 0.25), (GE, 0.25), (AS, 0.5)],
            1.15,
            Tetragonal,
        ),
        me(
            "ZnSiAs2",
            &[(ZN, 0.25), (SI, 0.25), (AS, 0.5)],
            1.74,
            Tetragonal,
        ),
        me(
            "CdSiAs2",
            &[(CD, 0.25), (SI, 0.25), (AS, 0.5)],
            1.55,
            Tetragonal,
        ),
    ]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_has_at_least_150_entries() {
        let data = load_training_data();
        assert!(
            data.len() >= 150,
            "Expected >=150 entries, got {}",
            data.len()
        );
    }

    #[test]
    fn compositions_sum_to_one() {
        let data = load_training_data();
        for entry in &data {
            let sum: f64 = entry.composition.iter().map(|(_, f)| f).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "{}: composition fractions sum to {:.6}, expected 1.0",
                entry.formula,
                sum
            );
        }
    }

    #[test]
    fn no_negative_bandgaps() {
        let data = load_training_data();
        for entry in &data {
            assert!(
                entry.experimental_gap >= 0.0,
                "{}: negative bandgap {}",
                entry.formula,
                entry.experimental_gap
            );
        }
    }

    #[test]
    fn no_duplicate_formulas() {
        let data = load_training_data();
        let mut seen = std::collections::HashSet::new();
        for entry in &data {
            assert!(
                seen.insert(entry.formula),
                "Duplicate formula: {}",
                entry.formula
            );
        }
    }

    #[test]
    fn known_bandgaps_correct() {
        let data = load_training_data();
        let find = |name: &str| data.iter().find(|e| e.formula == name).unwrap();

        assert!((find("Si").experimental_gap - 1.17).abs() < 0.01);
        assert!((find("GaAs").experimental_gap - 1.42).abs() < 0.01);
        assert!((find("GaN").experimental_gap - 3.39).abs() < 0.01);
        assert!((find("LiF").experimental_gap - 13.6).abs() < 0.1);
        assert!((find("Ge").experimental_gap - 0.66).abs() < 0.01);
        assert!((find("SrTiO3").experimental_gap - 3.25).abs() < 0.01);
    }

    #[test]
    fn all_atomic_numbers_valid() {
        let data = load_training_data();
        for entry in &data {
            for &(z, frac) in &entry.composition {
                assert!(z >= 1 && z <= 118, "{}: invalid Z={}", entry.formula, z);
                assert!(
                    frac > 0.0 && frac <= 1.0,
                    "{}: invalid fraction {} for Z={}",
                    entry.formula,
                    frac,
                    z
                );
            }
        }
    }

    #[test]
    fn crystal_system_ordinals_unique() {
        use CrystalSystem::*;
        let systems = [
            Cubic,
            Hexagonal,
            Tetragonal,
            Orthorhombic,
            Monoclinic,
            Trigonal,
            Triclinic,
            Unknown,
        ];
        let ordinals: Vec<u8> = systems.iter().map(|s| s.ordinal()).collect();
        let unique: std::collections::HashSet<u8> = ordinals.iter().copied().collect();
        assert_eq!(ordinals.len(), unique.len());
    }

    #[test]
    fn backward_compat_accessors() {
        let data = load_training_data();
        let si = data.iter().find(|e| e.formula == "Si").unwrap();
        assert!((si.bandgap_exp() - 1.17).abs() < 0.01);
        assert_eq!(*si.crystal(), CrystalSystem::Cubic);
        assert_eq!(si.name(), "Si");
    }
}
