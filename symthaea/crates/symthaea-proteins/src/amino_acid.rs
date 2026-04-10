// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Amino acid definitions and physicochemical property tables.
//!
//! All property values are from published literature:
//! - Hydrophobicity: Kyte & Doolittle (1982) *J. Mol. Biol.* 157, 105-132
//! - Van der Waals volumes: Creighton (1993) *Proteins*, 2nd ed.
//! - Molecular weights: standard monoisotopic residue masses
//! - Chou-Fasman propensities: Chou & Fasman (1978) *Adv. Enzymol.* 47, 45-148

use serde::{Deserialize, Serialize};

/// The 20 standard amino acids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AminoAcid {
    Ala,
    Arg,
    Asn,
    Asp,
    Cys,
    Gln,
    Glu,
    Gly,
    His,
    Ile,
    Leu,
    Lys,
    Met,
    Phe,
    Pro,
    Ser,
    Thr,
    Trp,
    Tyr,
    Val,
}

impl AminoAcid {
    /// All 20 standard amino acids in canonical order.
    pub const ALL: [AminoAcid; 20] = [
        AminoAcid::Ala,
        AminoAcid::Arg,
        AminoAcid::Asn,
        AminoAcid::Asp,
        AminoAcid::Cys,
        AminoAcid::Gln,
        AminoAcid::Glu,
        AminoAcid::Gly,
        AminoAcid::His,
        AminoAcid::Ile,
        AminoAcid::Leu,
        AminoAcid::Lys,
        AminoAcid::Met,
        AminoAcid::Phe,
        AminoAcid::Pro,
        AminoAcid::Ser,
        AminoAcid::Thr,
        AminoAcid::Trp,
        AminoAcid::Tyr,
        AminoAcid::Val,
    ];

    /// Parse a single-letter IUPAC amino acid code.
    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_uppercase() {
            'A' => Some(AminoAcid::Ala),
            'R' => Some(AminoAcid::Arg),
            'N' => Some(AminoAcid::Asn),
            'D' => Some(AminoAcid::Asp),
            'C' => Some(AminoAcid::Cys),
            'Q' => Some(AminoAcid::Gln),
            'E' => Some(AminoAcid::Glu),
            'G' => Some(AminoAcid::Gly),
            'H' => Some(AminoAcid::His),
            'I' => Some(AminoAcid::Ile),
            'L' => Some(AminoAcid::Leu),
            'K' => Some(AminoAcid::Lys),
            'M' => Some(AminoAcid::Met),
            'F' => Some(AminoAcid::Phe),
            'P' => Some(AminoAcid::Pro),
            'S' => Some(AminoAcid::Ser),
            'T' => Some(AminoAcid::Thr),
            'W' => Some(AminoAcid::Trp),
            'Y' => Some(AminoAcid::Tyr),
            'V' => Some(AminoAcid::Val),
            _ => None,
        }
    }

    /// Single-letter IUPAC code for this amino acid.
    pub fn to_char(self) -> char {
        match self {
            AminoAcid::Ala => 'A',
            AminoAcid::Arg => 'R',
            AminoAcid::Asn => 'N',
            AminoAcid::Asp => 'D',
            AminoAcid::Cys => 'C',
            AminoAcid::Gln => 'Q',
            AminoAcid::Glu => 'E',
            AminoAcid::Gly => 'G',
            AminoAcid::His => 'H',
            AminoAcid::Ile => 'I',
            AminoAcid::Leu => 'L',
            AminoAcid::Lys => 'K',
            AminoAcid::Met => 'M',
            AminoAcid::Phe => 'F',
            AminoAcid::Pro => 'P',
            AminoAcid::Ser => 'S',
            AminoAcid::Thr => 'T',
            AminoAcid::Trp => 'W',
            AminoAcid::Tyr => 'Y',
            AminoAcid::Val => 'V',
        }
    }

    /// Zero-based index (0..19) matching the order of [`ALL`].
    pub fn index(self) -> usize {
        match self {
            AminoAcid::Ala => 0,
            AminoAcid::Arg => 1,
            AminoAcid::Asn => 2,
            AminoAcid::Asp => 3,
            AminoAcid::Cys => 4,
            AminoAcid::Gln => 5,
            AminoAcid::Glu => 6,
            AminoAcid::Gly => 7,
            AminoAcid::His => 8,
            AminoAcid::Ile => 9,
            AminoAcid::Leu => 10,
            AminoAcid::Lys => 11,
            AminoAcid::Met => 12,
            AminoAcid::Phe => 13,
            AminoAcid::Pro => 14,
            AminoAcid::Ser => 15,
            AminoAcid::Thr => 16,
            AminoAcid::Trp => 17,
            AminoAcid::Tyr => 18,
            AminoAcid::Val => 19,
        }
    }
}

/// Three-state secondary structure classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecondaryStructure {
    Helix,
    Sheet,
    Coil,
}

impl SecondaryStructure {
    /// Parse DSSP-style single character (H=Helix, E=Sheet, everything else=Coil).
    pub fn from_char(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'H' => SecondaryStructure::Helix,
            'E' => SecondaryStructure::Sheet,
            _ => SecondaryStructure::Coil,
        }
    }

    /// DSSP-style single character.
    pub fn to_char(self) -> char {
        match self {
            SecondaryStructure::Helix => 'H',
            SecondaryStructure::Sheet => 'E',
            SecondaryStructure::Coil => 'C',
        }
    }

    /// Numeric class label for ML (0=H, 1=E, 2=C).
    pub fn class_id(self) -> u8 {
        match self {
            SecondaryStructure::Helix => 0,
            SecondaryStructure::Sheet => 1,
            SecondaryStructure::Coil => 2,
        }
    }

    /// Reconstruct from numeric class label.
    pub fn from_class_id(id: u8) -> Self {
        match id {
            0 => SecondaryStructure::Helix,
            1 => SecondaryStructure::Sheet,
            _ => SecondaryStructure::Coil,
        }
    }
}

// ---------------------------------------------------------------------------
// Property tables — indexed by AminoAcid::index()
// ---------------------------------------------------------------------------

/// Kyte-Doolittle hydrophobicity scale (1982).
const HYDROPHOBICITY: [f64; 20] = [
    1.8,  // Ala
    -4.5, // Arg
    -3.5, // Asn
    -3.5, // Asp
    2.5,  // Cys
    -3.5, // Gln
    -3.5, // Glu
    -0.4, // Gly
    -3.2, // His
    4.5,  // Ile
    3.8,  // Leu
    -3.9, // Lys
    1.9,  // Met
    2.8,  // Phe
    -1.6, // Pro
    -0.8, // Ser
    -0.7, // Thr
    -0.9, // Trp
    -1.3, // Tyr
    4.2,  // Val
];

/// Net charge at physiological pH (~7.0).
const CHARGE: [f64; 20] = [
    0.0,  // Ala
    1.0,  // Arg
    0.0,  // Asn
    -1.0, // Asp
    0.0,  // Cys
    0.0,  // Gln
    -1.0, // Glu
    0.0,  // Gly
    0.1,  // His (partially protonated, pKa ~6.0)
    0.0,  // Ile
    0.0,  // Leu
    1.0,  // Lys
    0.0,  // Met
    0.0,  // Phe
    0.0,  // Pro
    0.0,  // Ser
    0.0,  // Thr
    0.0,  // Trp
    0.0,  // Tyr
    0.0,  // Val
];

/// Van der Waals volume in cubic angstroms.
const VOLUME: [f64; 20] = [
    88.6,  // Ala
    173.4, // Arg
    114.1, // Asn
    111.1, // Asp
    108.5, // Cys
    143.8, // Gln
    138.4, // Glu
    60.1,  // Gly
    153.2, // His
    166.7, // Ile
    166.7, // Leu
    168.6, // Lys
    162.9, // Met
    189.9, // Phe
    112.7, // Pro
    89.0,  // Ser
    116.1, // Thr
    227.8, // Trp
    193.6, // Tyr
    140.0, // Val
];

/// Molecular weight in daltons (residue mass).
const WEIGHT: [f64; 20] = [
    89.1,  // Ala
    174.2, // Arg
    132.1, // Asn
    133.1, // Asp
    121.2, // Cys
    146.1, // Gln
    147.1, // Glu
    75.0,  // Gly
    155.2, // His
    131.2, // Ile
    131.2, // Leu
    146.2, // Lys
    149.2, // Met
    165.2, // Phe
    115.1, // Pro
    105.1, // Ser
    119.1, // Thr
    204.2, // Trp
    181.2, // Tyr
    117.1, // Val
];

/// Chou-Fasman helix propensity (P_alpha). Chou & Fasman (1978).
const HELIX_PROPENSITY: [f64; 20] = [
    1.42, // Ala
    0.98, // Arg
    0.67, // Asn
    1.01, // Asp
    0.70, // Cys
    1.11, // Gln
    1.51, // Glu
    0.57, // Gly
    1.00, // His
    1.08, // Ile
    1.21, // Leu
    1.16, // Lys
    1.45, // Met
    1.13, // Phe
    0.57, // Pro
    0.77, // Ser
    0.83, // Thr
    1.08, // Trp
    0.69, // Tyr
    1.06, // Val
];

/// Chou-Fasman sheet propensity (P_beta). Chou & Fasman (1978).
const SHEET_PROPENSITY: [f64; 20] = [
    0.83, // Ala
    0.93, // Arg
    0.89, // Asn
    0.54, // Asp
    1.19, // Cys
    1.10, // Gln
    0.37, // Glu
    0.75, // Gly
    0.87, // His
    1.60, // Ile
    1.30, // Leu
    0.74, // Lys
    1.05, // Met
    1.38, // Phe
    0.55, // Pro
    0.75, // Ser
    1.19, // Thr
    1.37, // Trp
    1.47, // Tyr
    1.70, // Val
];

/// Chou-Fasman coil propensity (P_coil). Chou & Fasman (1978).
const COIL_PROPENSITY: [f64; 20] = [
    0.66, // Ala
    1.08, // Arg
    1.56, // Asn
    1.46, // Asp
    1.19, // Cys
    0.98, // Gln
    0.74, // Glu
    1.56, // Gly
    0.95, // His
    0.47, // Ile
    0.59, // Leu
    1.01, // Lys
    0.60, // Met
    0.60, // Phe
    1.52, // Pro
    1.43, // Ser
    0.96, // Thr
    0.96, // Trp
    1.14, // Tyr
    0.50, // Val
];

// ---------------------------------------------------------------------------
// Property accessor functions
// ---------------------------------------------------------------------------

/// Kyte-Doolittle hydrophobicity (1982). Range roughly -4.5 to +4.5.
pub fn hydrophobicity(aa: AminoAcid) -> f64 {
    HYDROPHOBICITY[aa.index()]
}

/// Net charge at physiological pH (~7.0).
pub fn charge(aa: AminoAcid) -> f64 {
    CHARGE[aa.index()]
}

/// Van der Waals volume in cubic angstroms.
pub fn volume(aa: AminoAcid) -> f64 {
    VOLUME[aa.index()]
}

/// Molecular weight in daltons (residue mass).
pub fn weight(aa: AminoAcid) -> f64 {
    WEIGHT[aa.index()]
}

/// Chou-Fasman helix propensity (P_alpha). Values > 1.0 favour helix.
pub fn helix_propensity(aa: AminoAcid) -> f64 {
    HELIX_PROPENSITY[aa.index()]
}

/// Chou-Fasman sheet propensity (P_beta). Values > 1.0 favour sheet.
pub fn sheet_propensity(aa: AminoAcid) -> f64 {
    SHEET_PROPENSITY[aa.index()]
}

/// Chou-Fasman coil propensity. Values > 1.0 favour coil/turn.
pub fn coil_propensity(aa: AminoAcid) -> f64 {
    COIL_PROPENSITY[aa.index()]
}

/// True for classical helix breakers (Pro, Gly).
pub fn is_helix_breaker(aa: AminoAcid) -> bool {
    matches!(aa, AminoAcid::Pro | AminoAcid::Gly)
}

/// True if Kyte-Doolittle hydrophobicity > 0.
pub fn is_hydrophobic(aa: AminoAcid) -> bool {
    hydrophobicity(aa) > 0.0
}

/// True if charged at physiological pH (|charge| > 0.5).
pub fn is_charged(aa: AminoAcid) -> bool {
    charge(aa).abs() > 0.5
}

/// True if aromatic side chain (Phe, Trp, Tyr, His).
pub fn is_aromatic(aa: AminoAcid) -> bool {
    matches!(
        aa,
        AminoAcid::Phe | AminoAcid::Trp | AminoAcid::Tyr | AminoAcid::His
    )
}

/// True if small side chain (Gly, Ala, Ser).
pub fn is_small(aa: AminoAcid) -> bool {
    matches!(aa, AminoAcid::Gly | AminoAcid::Ala | AminoAcid::Ser)
}

/// Van der Waals volume normalized to [0, 1] (min=Gly 60.1, max=Trp 227.8).
pub fn volume_normalized(aa: AminoAcid) -> f64 {
    const MIN_VOL: f64 = 60.1; // Gly
    const MAX_VOL: f64 = 227.8; // Trp
    (volume(aa) - MIN_VOL) / (MAX_VOL - MIN_VOL)
}

/// Parse a string of one-letter amino acid codes, skipping unknown characters.
pub fn parse_sequence(s: &str) -> Vec<AminoAcid> {
    s.chars().filter_map(AminoAcid::from_char).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_constant_length() {
        assert_eq!(AminoAcid::ALL.len(), 20);
    }

    #[test]
    fn test_index_roundtrip() {
        for (i, &aa) in AminoAcid::ALL.iter().enumerate() {
            assert_eq!(aa.index(), i);
        }
    }

    #[test]
    fn test_char_roundtrip() {
        for &aa in &AminoAcid::ALL {
            let c = aa.to_char();
            assert_eq!(AminoAcid::from_char(c), Some(aa));
        }
    }

    #[test]
    fn test_from_char_lowercase() {
        assert_eq!(AminoAcid::from_char('a'), Some(AminoAcid::Ala));
        assert_eq!(AminoAcid::from_char('r'), Some(AminoAcid::Arg));
    }

    #[test]
    fn test_from_char_unknown() {
        assert_eq!(AminoAcid::from_char('X'), None);
        assert_eq!(AminoAcid::from_char('B'), None);
    }

    #[test]
    fn test_parse_sequence() {
        // A=Ala, C=Cys, G=Gly, T=Thr — all 4 are valid amino acids
        let seq = parse_sequence("ACGT");
        assert_eq!(seq.len(), 4);
        assert_eq!(seq[0], AminoAcid::Ala);
        assert_eq!(seq[1], AminoAcid::Cys);
        assert_eq!(seq[2], AminoAcid::Gly);
        assert_eq!(seq[3], AminoAcid::Thr);
    }

    #[test]
    fn test_parse_sequence_skips_invalid() {
        let seq = parse_sequence("AX1C");
        assert_eq!(seq.len(), 2);
        assert_eq!(seq[0], AminoAcid::Ala);
        assert_eq!(seq[1], AminoAcid::Cys);
    }

    #[test]
    fn test_property_table_sizes() {
        assert_eq!(HYDROPHOBICITY.len(), 20);
        assert_eq!(CHARGE.len(), 20);
        assert_eq!(VOLUME.len(), 20);
        assert_eq!(WEIGHT.len(), 20);
        assert_eq!(HELIX_PROPENSITY.len(), 20);
        assert_eq!(SHEET_PROPENSITY.len(), 20);
        assert_eq!(COIL_PROPENSITY.len(), 20);
    }

    #[test]
    fn test_known_hydrophobicity_values() {
        assert!((hydrophobicity(AminoAcid::Ile) - 4.5).abs() < 1e-10);
        assert!((hydrophobicity(AminoAcid::Arg) - (-4.5)).abs() < 1e-10);
        assert!((hydrophobicity(AminoAcid::Gly) - (-0.4)).abs() < 1e-10);
    }

    #[test]
    fn test_known_charge_values() {
        assert!((charge(AminoAcid::Arg) - 1.0).abs() < 1e-10);
        assert!((charge(AminoAcid::Asp) - (-1.0)).abs() < 1e-10);
        assert!((charge(AminoAcid::Ala) - 0.0).abs() < 1e-10);
        assert!((charge(AminoAcid::His) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_known_weight_values() {
        assert!((weight(AminoAcid::Gly) - 75.0).abs() < 1e-10);
        assert!((weight(AminoAcid::Trp) - 204.2).abs() < 1e-10);
    }

    #[test]
    fn test_helix_breakers() {
        assert!(is_helix_breaker(AminoAcid::Pro));
        assert!(is_helix_breaker(AminoAcid::Gly));
        assert!(!is_helix_breaker(AminoAcid::Ala));
        assert!(!is_helix_breaker(AminoAcid::Leu));
    }

    #[test]
    fn test_is_hydrophobic() {
        assert!(is_hydrophobic(AminoAcid::Ile));
        assert!(is_hydrophobic(AminoAcid::Val));
        assert!(is_hydrophobic(AminoAcid::Leu));
        assert!(!is_hydrophobic(AminoAcid::Arg));
        assert!(!is_hydrophobic(AminoAcid::Asp));
    }

    #[test]
    fn test_secondary_structure_char_roundtrip() {
        assert_eq!(
            SecondaryStructure::from_char('H'),
            SecondaryStructure::Helix
        );
        assert_eq!(
            SecondaryStructure::from_char('E'),
            SecondaryStructure::Sheet
        );
        assert_eq!(SecondaryStructure::from_char('C'), SecondaryStructure::Coil);
        assert_eq!(SecondaryStructure::from_char('X'), SecondaryStructure::Coil);

        assert_eq!(SecondaryStructure::Helix.to_char(), 'H');
        assert_eq!(SecondaryStructure::Sheet.to_char(), 'E');
        assert_eq!(SecondaryStructure::Coil.to_char(), 'C');
    }

    #[test]
    fn test_chou_fasman_propensity_known_values() {
        // Ala is a strong helix former
        assert!(helix_propensity(AminoAcid::Ala) > 1.0);
        // Val is a strong sheet former
        assert!(sheet_propensity(AminoAcid::Val) > 1.0);
        // Gly is a strong coil former
        assert!(coil_propensity(AminoAcid::Gly) > 1.0);
    }
}
