// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complete periodic table with electronic properties for bandgap prediction.
//!
//! All 103 elements (H through Lr) with Pauling electronegativity, empirical
//! atomic radius, first ionization energy, electron affinity, atomic mass,
//! group/period, and typical valence electrons.
//!
//! Data sourced from WebElements / NIST / CRC Handbook of Chemistry and Physics.

use serde::{Deserialize, Serialize};

/// Electronic and structural properties of a chemical element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    /// Atomic number (proton count).
    pub z: u8,
    /// IUPAC symbol (1-2 characters).
    pub symbol: &'static str,
    /// Full English name.
    pub name: &'static str,
    /// Pauling electronegativity (0.0 if unknown/not defined).
    pub electronegativity: f64,
    /// Empirical atomic radius in pm (0.0 if unknown).
    pub atomic_radius: f64,
    /// First ionization energy in eV.
    pub ionization_energy: f64,
    /// Electron affinity in eV (0.0 if unknown or endothermic).
    pub electron_affinity: f64,
    /// Standard atomic mass in amu.
    pub atomic_mass: f64,
    /// IUPAC group 1-18 (0 for lanthanides/actinides).
    pub group: u8,
    /// Period 1-7.
    pub period: u8,
    /// Typical number of valence electrons.
    pub valence_electrons: u8,
}

const fn el(
    z: u8,
    symbol: &'static str,
    name: &'static str,
    electronegativity: f64,
    atomic_radius: f64,
    ionization_energy: f64,
    electron_affinity: f64,
    atomic_mass: f64,
    group: u8,
    period: u8,
    valence_electrons: u8,
) -> Element {
    Element {
        z,
        symbol,
        name,
        electronegativity,
        atomic_radius,
        ionization_energy,
        electron_affinity,
        atomic_mass,
        group,
        period,
        valence_electrons,
    }
}

/// All 103 elements indexed by Z-1 (so ELEMENTS[0] = Hydrogen, Z=1).
///
/// Values from IUPAC, WebElements, and NIST databases.
/// Electronegativity: Pauling scale (IUPAC 2016). 0.0 = not defined.
/// Atomic radius: empirical values in pm.
/// Ionization energy: first IE in eV (NIST).
/// Electron affinity: in eV (0.0 if endothermic or unknown).
/// Atomic mass: standard atomic weight in amu.
static ELEMENTS: [Element; 103] = [
    // ---- Period 1 ----
    el(
        1, "H", "Hydrogen", 2.20, 25.0, 13.598, 0.754, 1.008, 1, 1, 1,
    ),
    el(
        2, "He", "Helium", 0.00, 31.0, 24.587, 0.000, 4.003, 18, 1, 2,
    ),
    // ---- Period 2 ----
    el(
        3, "Li", "Lithium", 0.98, 145.0, 5.392, 0.618, 6.941, 1, 2, 1,
    ),
    el(
        4,
        "Be",
        "Beryllium",
        1.57,
        105.0,
        9.323,
        0.000,
        9.012,
        2,
        2,
        2,
    ),
    el(5, "B", "Boron", 2.04, 85.0, 8.298, 0.277, 10.81, 13, 2, 3),
    el(
        6, "C", "Carbon", 2.55, 70.0, 11.260, 1.263, 12.011, 14, 2, 4,
    ),
    el(
        7, "N", "Nitrogen", 3.04, 65.0, 14.534, 0.000, 14.007, 15, 2, 5,
    ),
    el(
        8, "O", "Oxygen", 3.44, 60.0, 13.618, 1.461, 15.999, 16, 2, 6,
    ),
    el(
        9, "F", "Fluorine", 3.98, 50.0, 17.423, 3.401, 18.998, 17, 2, 7,
    ),
    el(
        10, "Ne", "Neon", 0.00, 38.0, 21.565, 0.000, 20.180, 18, 2, 8,
    ),
    // ---- Period 3 ----
    el(
        11, "Na", "Sodium", 0.93, 180.0, 5.139, 0.548, 22.990, 1, 3, 1,
    ),
    el(
        12,
        "Mg",
        "Magnesium",
        1.31,
        150.0,
        7.646,
        0.000,
        24.305,
        2,
        3,
        2,
    ),
    el(
        13,
        "Al",
        "Aluminium",
        1.61,
        125.0,
        5.986,
        0.441,
        26.982,
        13,
        3,
        3,
    ),
    el(
        14, "Si", "Silicon", 1.90, 110.0, 8.152, 1.390, 28.086, 14, 3, 4,
    ),
    el(
        15,
        "P",
        "Phosphorus",
        2.19,
        100.0,
        10.487,
        0.747,
        30.974,
        15,
        3,
        5,
    ),
    el(
        16, "S", "Sulfur", 2.58, 100.0, 10.360, 2.077, 32.06, 16, 3, 6,
    ),
    el(
        17, "Cl", "Chlorine", 3.16, 100.0, 12.968, 3.617, 35.45, 17, 3, 7,
    ),
    el(
        18, "Ar", "Argon", 0.00, 71.0, 15.760, 0.000, 39.948, 18, 3, 8,
    ),
    // ---- Period 4 ----
    el(
        19,
        "K",
        "Potassium",
        0.82,
        220.0,
        4.341,
        0.501,
        39.098,
        1,
        4,
        1,
    ),
    el(
        20, "Ca", "Calcium", 1.00, 180.0, 6.113, 0.025, 40.078, 2, 4, 2,
    ),
    el(
        21, "Sc", "Scandium", 1.36, 160.0, 6.562, 0.188, 44.956, 3, 4, 2,
    ),
    el(
        22, "Ti", "Titanium", 1.54, 140.0, 6.828, 0.079, 47.867, 4, 4, 2,
    ),
    el(
        23, "V", "Vanadium", 1.63, 135.0, 6.746, 0.525, 50.942, 5, 4, 2,
    ),
    el(
        24, "Cr", "Chromium", 1.66, 140.0, 6.767, 0.666, 51.996, 6, 4, 1,
    ),
    el(
        25,
        "Mn",
        "Manganese",
        1.55,
        140.0,
        7.434,
        0.000,
        54.938,
        7,
        4,
        2,
    ),
    el(26, "Fe", "Iron", 1.83, 140.0, 7.902, 0.151, 55.845, 8, 4, 2),
    el(
        27, "Co", "Cobalt", 1.88, 135.0, 7.881, 0.662, 58.933, 9, 4, 2,
    ),
    el(
        28, "Ni", "Nickel", 1.91, 135.0, 7.640, 1.156, 58.693, 10, 4, 2,
    ),
    el(
        29, "Cu", "Copper", 1.90, 135.0, 7.726, 1.236, 63.546, 11, 4, 1,
    ),
    el(30, "Zn", "Zinc", 1.65, 135.0, 9.394, 0.000, 65.38, 12, 4, 2),
    el(
        31, "Ga", "Gallium", 1.81, 130.0, 5.999, 0.300, 69.723, 13, 4, 3,
    ),
    el(
        32,
        "Ge",
        "Germanium",
        2.01,
        125.0,
        7.900,
        1.233,
        72.630,
        14,
        4,
        4,
    ),
    el(
        33, "As", "Arsenic", 2.18, 115.0, 9.789, 0.810, 74.922, 15, 4, 5,
    ),
    el(
        34, "Se", "Selenium", 2.55, 115.0, 9.752, 2.021, 78.971, 16, 4, 6,
    ),
    el(
        35, "Br", "Bromine", 2.96, 115.0, 11.814, 3.364, 79.904, 17, 4, 7,
    ),
    el(
        36, "Kr", "Krypton", 3.00, 88.0, 14.000, 0.000, 83.798, 18, 4, 8,
    ),
    // ---- Period 5 ----
    el(
        37, "Rb", "Rubidium", 0.82, 235.0, 4.177, 0.486, 85.468, 1, 5, 1,
    ),
    el(
        38,
        "Sr",
        "Strontium",
        0.95,
        200.0,
        5.695,
        0.052,
        87.62,
        2,
        5,
        2,
    ),
    el(
        39, "Y", "Yttrium", 1.22, 180.0, 6.217, 0.307, 88.906, 3, 5, 2,
    ),
    el(
        40,
        "Zr",
        "Zirconium",
        1.33,
        155.0,
        6.634,
        0.426,
        91.224,
        4,
        5,
        2,
    ),
    el(
        41, "Nb", "Niobium", 1.60, 145.0, 6.759, 0.893, 92.906, 5, 5, 1,
    ),
    el(
        42,
        "Mo",
        "Molybdenum",
        2.16,
        145.0,
        7.092,
        0.746,
        95.95,
        6,
        5,
        1,
    ),
    el(
        43,
        "Tc",
        "Technetium",
        1.90,
        135.0,
        7.280,
        0.550,
        98.0,
        7,
        5,
        2,
    ),
    el(
        44,
        "Ru",
        "Ruthenium",
        2.20,
        130.0,
        7.361,
        1.050,
        101.07,
        8,
        5,
        1,
    ),
    el(
        45, "Rh", "Rhodium", 2.28, 135.0, 7.459, 1.137, 102.906, 9, 5, 1,
    ),
    el(
        46,
        "Pd",
        "Palladium",
        2.20,
        140.0,
        8.337,
        0.557,
        106.42,
        10,
        5,
        0,
    ),
    el(
        47, "Ag", "Silver", 1.93, 160.0, 7.576, 1.302, 107.868, 11, 5, 1,
    ),
    el(
        48, "Cd", "Cadmium", 1.69, 155.0, 8.994, 0.000, 112.414, 12, 5, 2,
    ),
    el(
        49, "In", "Indium", 1.78, 155.0, 5.786, 0.300, 114.818, 13, 5, 3,
    ),
    el(
        50, "Sn", "Tin", 1.96, 145.0, 7.344, 1.112, 118.710, 14, 5, 4,
    ),
    el(
        51, "Sb", "Antimony", 2.05, 145.0, 8.640, 1.047, 121.760, 15, 5, 5,
    ),
    el(
        52,
        "Te",
        "Tellurium",
        2.10,
        140.0,
        9.010,
        1.971,
        127.60,
        16,
        5,
        6,
    ),
    el(
        53, "I", "Iodine", 2.66, 140.0, 10.451, 3.059, 126.904, 17, 5, 7,
    ),
    el(
        54, "Xe", "Xenon", 2.60, 108.0, 12.130, 0.000, 131.293, 18, 5, 8,
    ),
    // ---- Period 6 ----
    el(
        55, "Cs", "Caesium", 0.79, 260.0, 3.894, 0.472, 132.905, 1, 6, 1,
    ),
    el(
        56, "Ba", "Barium", 0.89, 215.0, 5.212, 0.145, 137.327, 2, 6, 2,
    ),
    // Lanthanides (group = 0)
    el(
        57,
        "La",
        "Lanthanum",
        1.10,
        195.0,
        5.577,
        0.500,
        138.905,
        0,
        6,
        2,
    ),
    el(
        58, "Ce", "Cerium", 1.12, 185.0, 5.539, 0.500, 140.116, 0, 6, 2,
    ),
    el(
        59,
        "Pr",
        "Praseodymium",
        1.13,
        185.0,
        5.473,
        0.500,
        140.908,
        0,
        6,
        2,
    ),
    el(
        60,
        "Nd",
        "Neodymium",
        1.14,
        185.0,
        5.525,
        0.500,
        144.242,
        0,
        6,
        2,
    ),
    el(
        61,
        "Pm",
        "Promethium",
        1.13,
        185.0,
        5.582,
        0.500,
        145.0,
        0,
        6,
        2,
    ),
    el(
        62, "Sm", "Samarium", 1.17, 185.0, 5.644, 0.500, 150.36, 0, 6, 2,
    ),
    el(
        63, "Eu", "Europium", 1.20, 185.0, 5.670, 0.500, 151.964, 0, 6, 2,
    ),
    el(
        64,
        "Gd",
        "Gadolinium",
        1.20,
        180.0,
        6.150,
        0.500,
        157.25,
        0,
        6,
        2,
    ),
    el(
        65, "Tb", "Terbium", 1.10, 175.0, 5.864, 0.500, 158.925, 0, 6, 2,
    ),
    el(
        66,
        "Dy",
        "Dysprosium",
        1.22,
        175.0,
        5.939,
        0.500,
        162.500,
        0,
        6,
        2,
    ),
    el(
        67, "Ho", "Holmium", 1.23, 175.0, 6.022, 0.500, 164.930, 0, 6, 2,
    ),
    el(
        68, "Er", "Erbium", 1.24, 175.0, 6.108, 0.500, 167.259, 0, 6, 2,
    ),
    el(
        69, "Tm", "Thulium", 1.25, 175.0, 6.184, 0.500, 168.934, 0, 6, 2,
    ),
    el(
        70,
        "Yb",
        "Ytterbium",
        1.10,
        175.0,
        6.254,
        0.500,
        173.045,
        0,
        6,
        2,
    ),
    el(
        71, "Lu", "Lutetium", 1.27, 175.0, 5.426, 0.500, 174.967, 3, 6, 2,
    ),
    // Post-lanthanide d-block
    el(
        72, "Hf", "Hafnium", 1.30, 155.0, 6.825, 0.000, 178.49, 4, 6, 2,
    ),
    el(
        73, "Ta", "Tantalum", 1.50, 145.0, 7.550, 0.322, 180.948, 5, 6, 2,
    ),
    el(
        74, "W", "Tungsten", 2.36, 135.0, 7.864, 0.816, 183.84, 6, 6, 2,
    ),
    el(
        75, "Re", "Rhenium", 1.90, 135.0, 7.833, 0.150, 186.207, 7, 6, 2,
    ),
    el(
        76, "Os", "Osmium", 2.20, 130.0, 8.438, 1.100, 190.23, 8, 6, 2,
    ),
    el(
        77, "Ir", "Iridium", 2.20, 135.0, 8.967, 1.565, 192.217, 9, 6, 2,
    ),
    el(
        78, "Pt", "Platinum", 2.28, 135.0, 8.959, 2.128, 195.084, 10, 6, 1,
    ),
    el(
        79, "Au", "Gold", 2.54, 135.0, 9.226, 2.309, 196.967, 11, 6, 1,
    ),
    el(
        80, "Hg", "Mercury", 2.00, 150.0, 10.437, 0.000, 200.592, 12, 6, 2,
    ),
    // Post-lanthanide p-block
    el(
        81, "Tl", "Thallium", 1.62, 190.0, 6.108, 0.200, 204.38, 13, 6, 3,
    ),
    el(82, "Pb", "Lead", 2.33, 180.0, 7.417, 0.364, 207.2, 14, 6, 4),
    el(
        83, "Bi", "Bismuth", 2.02, 160.0, 7.286, 0.946, 208.980, 15, 6, 5,
    ),
    el(
        84, "Po", "Polonium", 2.00, 190.0, 8.414, 1.900, 209.0, 16, 6, 6,
    ),
    el(
        85, "At", "Astatine", 2.20, 127.0, 9.318, 2.800, 210.0, 17, 6, 7,
    ),
    el(
        86, "Rn", "Radon", 2.20, 120.0, 10.749, 0.000, 222.0, 18, 6, 8,
    ),
    // ---- Period 7 ----
    el(
        87, "Fr", "Francium", 0.70, 260.0, 4.073, 0.000, 223.0, 1, 7, 1,
    ),
    el(
        88, "Ra", "Radium", 0.90, 215.0, 5.278, 0.000, 226.0, 2, 7, 2,
    ),
    // Actinides (group = 0)
    el(
        89, "Ac", "Actinium", 1.10, 195.0, 5.170, 0.000, 227.0, 0, 7, 2,
    ),
    el(
        90, "Th", "Thorium", 1.30, 180.0, 6.307, 0.000, 232.038, 0, 7, 2,
    ),
    el(
        91,
        "Pa",
        "Protactinium",
        1.50,
        180.0,
        5.890,
        0.000,
        231.036,
        0,
        7,
        2,
    ),
    el(
        92, "U", "Uranium", 1.38, 175.0, 6.194, 0.000, 238.029, 0, 7, 2,
    ),
    el(
        93,
        "Np",
        "Neptunium",
        1.36,
        175.0,
        6.266,
        0.000,
        237.0,
        0,
        7,
        2,
    ),
    el(
        94,
        "Pu",
        "Plutonium",
        1.28,
        175.0,
        6.026,
        0.000,
        244.0,
        0,
        7,
        2,
    ),
    el(
        95,
        "Am",
        "Americium",
        1.13,
        175.0,
        5.974,
        0.000,
        243.0,
        0,
        7,
        2,
    ),
    el(
        96, "Cm", "Curium", 1.28, 176.0, 5.991, 0.000, 247.0, 0, 7, 2,
    ),
    el(
        97,
        "Bk",
        "Berkelium",
        1.30,
        176.0,
        6.198,
        0.000,
        247.0,
        0,
        7,
        2,
    ),
    el(
        98,
        "Cf",
        "Californium",
        1.30,
        176.0,
        6.282,
        0.000,
        251.0,
        0,
        7,
        2,
    ),
    el(
        99,
        "Es",
        "Einsteinium",
        1.30,
        176.0,
        6.420,
        0.000,
        252.0,
        0,
        7,
        2,
    ),
    el(
        100, "Fm", "Fermium", 1.30, 176.0, 6.500, 0.000, 257.0, 0, 7, 2,
    ),
    el(
        101,
        "Md",
        "Mendelevium",
        1.30,
        176.0,
        6.580,
        0.000,
        258.0,
        0,
        7,
        2,
    ),
    el(
        102, "No", "Nobelium", 1.30, 176.0, 6.650, 0.000, 259.0, 0, 7, 2,
    ),
    el(
        103,
        "Lr",
        "Lawrencium",
        1.30,
        176.0,
        4.900,
        0.000,
        266.0,
        0,
        7,
        3,
    ),
];

// ---------------------------------------------------------------------------
// Legacy flat-array interface (kept for backward compatibility)
// ---------------------------------------------------------------------------

/// Pauling electronegativity for elements 1-103. Index 0 is unused (dummy).
/// Values from IUPAC 2016 recommendations. 0.0 = unknown/not applicable.
pub const PAULING_EN: [f64; 104] = [
    0.00, // 0: placeholder
    2.20, // 1: H
    0.00, // 2: He (noble gas)
    0.98, // 3: Li
    1.57, // 4: Be
    2.04, // 5: B
    2.55, // 6: C
    3.04, // 7: N
    3.44, // 8: O
    3.98, // 9: F
    0.00, // 10: Ne
    0.93, // 11: Na
    1.31, // 12: Mg
    1.61, // 13: Al
    1.90, // 14: Si
    2.19, // 15: P
    2.58, // 16: S
    3.16, // 17: Cl
    0.00, // 18: Ar
    0.82, // 19: K
    1.00, // 20: Ca
    1.36, // 21: Sc
    1.54, // 22: Ti
    1.63, // 23: V
    1.66, // 24: Cr
    1.55, // 25: Mn
    1.83, // 26: Fe
    1.88, // 27: Co
    1.91, // 28: Ni
    1.90, // 29: Cu
    1.65, // 30: Zn
    1.81, // 31: Ga
    2.01, // 32: Ge
    2.18, // 33: As
    2.55, // 34: Se
    2.96, // 35: Br
    3.00, // 36: Kr
    0.82, // 37: Rb
    0.95, // 38: Sr
    1.22, // 39: Y
    1.33, // 40: Zr
    1.60, // 41: Nb
    2.16, // 42: Mo
    1.90, // 43: Tc
    2.20, // 44: Ru
    2.28, // 45: Rh
    2.20, // 46: Pd
    1.93, // 47: Ag
    1.69, // 48: Cd
    1.78, // 49: In
    1.96, // 50: Sn
    2.05, // 51: Sb
    2.10, // 52: Te
    2.66, // 53: I
    2.60, // 54: Xe
    0.79, // 55: Cs
    0.89, // 56: Ba
    1.10, // 57: La
    1.12, // 58: Ce
    1.13, // 59: Pr
    1.14, // 60: Nd
    1.13, // 61: Pm
    1.17, // 62: Sm
    1.20, // 63: Eu
    1.20, // 64: Gd
    1.10, // 65: Tb
    1.22, // 66: Dy
    1.23, // 67: Ho
    1.24, // 68: Er
    1.25, // 69: Tm
    1.10, // 70: Yb
    1.27, // 71: Lu
    1.30, // 72: Hf
    1.50, // 73: Ta
    2.36, // 74: W
    1.90, // 75: Re
    2.20, // 76: Os
    2.20, // 77: Ir
    2.28, // 78: Pt
    2.54, // 79: Au
    2.00, // 80: Hg
    1.62, // 81: Tl
    2.33, // 82: Pb
    2.02, // 83: Bi
    2.00, // 84: Po
    2.20, // 85: At
    2.20, // 86: Rn
    0.70, // 87: Fr
    0.90, // 88: Ra
    1.10, // 89: Ac
    1.30, // 90: Th
    1.50, // 91: Pa
    1.38, // 92: U
    1.36, // 93: Np
    1.28, // 94: Pu
    1.13, // 95: Am
    1.28, // 96: Cm
    1.30, // 97: Bk
    1.30, // 98: Cf
    1.30, // 99: Es
    1.30, // 100: Fm
    1.30, // 101: Md
    1.30, // 102: No
    1.30, // 103: Lr
];

/// Element symbol lookup (Z=1..103).
pub const ELEMENT_SYMBOLS: [&str; 104] = [
    "??", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
];

/// Covalent radius in Angstroms (approximate, Cordero et al. 2008).
/// Index 0 unused. 0.0 = unknown.
pub const COVALENT_RADIUS: [f64; 104] = [
    0.00, // 0
    0.31, // H
    0.28, // He
    1.28, // Li
    0.96, // Be
    0.84, // B
    0.76, // C
    0.71, // N
    0.66, // O
    0.57, // F
    0.58, // Ne
    1.66, // Na
    1.41, // Mg
    1.21, // Al
    1.11, // Si
    1.07, // P
    1.05, // S
    1.02, // Cl
    1.06, // Ar
    2.03, // K
    1.76, // Ca
    1.70, // Sc
    1.60, // Ti
    1.53, // V
    1.39, // Cr
    1.39, // Mn (low spin)
    1.32, // Fe (low spin)
    1.26, // Co (low spin)
    1.24, // Ni
    1.32, // Cu
    1.22, // Zn
    1.22, // Ga
    1.20, // Ge
    1.19, // As
    1.20, // Se
    1.20, // Br
    1.16, // Kr
    2.20, // Rb
    1.95, // Sr
    1.90, // Y
    1.75, // Zr
    1.64, // Nb
    1.54, // Mo
    1.47, // Tc
    1.46, // Ru
    1.42, // Rh
    1.39, // Pd
    1.45, // Ag
    1.44, // Cd
    1.42, // In
    1.39, // Sn
    1.39, // Sb
    1.38, // Te
    1.39, // I
    1.40, // Xe
    2.44, // Cs
    2.15, // Ba
    2.07, // La
    2.04, // Ce
    2.03, // Pr
    2.01, // Nd
    1.99, // Pm
    1.98, // Sm
    1.98, // Eu
    1.96, // Gd
    1.94, // Tb
    1.92, // Dy
    1.92, // Ho
    1.89, // Er
    1.90, // Tm
    1.87, // Yb
    1.87, // Lu
    1.75, // Hf
    1.70, // Ta
    1.62, // W
    1.51, // Re
    1.44, // Os
    1.41, // Ir
    1.36, // Pt
    1.36, // Au
    1.32, // Hg
    1.45, // Tl
    1.46, // Pb
    1.48, // Bi
    1.40, // Po
    1.50, // At
    1.50, // Rn
    2.60, // Fr
    2.21, // Ra
    2.15, // Ac
    2.06, // Th
    2.00, // Pa
    1.96, // U
    1.90, // Np
    1.87, // Pu
    1.80, // Am
    1.69, // Cm
    1.60, // Bk  (estimate)
    1.60, // Cf  (estimate)
    1.60, // Es  (estimate)
    1.60, // Fm  (estimate)
    1.60, // Md  (estimate)
    1.60, // No  (estimate)
    1.60, // Lr  (estimate)
];

// ---------------------------------------------------------------------------
// Struct-based API
// ---------------------------------------------------------------------------

/// Look up an element by atomic number (Z = 1..=103).
///
/// # Panics
///
/// Panics if `z` is 0 or greater than 103.
pub fn element(z: u8) -> &'static Element {
    assert!(z >= 1 && z <= 103, "atomic number must be 1..=103, got {z}");
    &ELEMENTS[(z - 1) as usize]
}

/// Look up an element by its chemical symbol (case-sensitive).
///
/// Returns `None` if no match is found.
pub fn by_symbol(s: &str) -> Option<&'static Element> {
    ELEMENTS.iter().find(|e| e.symbol == s)
}

/// Compute the absolute Pauling electronegativity difference between two elements.
///
/// Useful for estimating bond ionicity and bandgap trends in binary compounds.
pub fn electronegativity_difference(z1: u8, z2: u8) -> f64 {
    let e1 = element(z1);
    let e2 = element(z2);
    (e1.electronegativity - e2.electronegativity).abs()
}

// ---------------------------------------------------------------------------
// Legacy flat-array helpers (kept for backward compatibility with features.rs
// and bandgap_baseline.rs)
// ---------------------------------------------------------------------------

/// Atomic number for common elements (convenience).
pub fn atomic_number(symbol: &str) -> Option<u8> {
    ELEMENT_SYMBOLS
        .iter()
        .position(|&s| s == symbol)
        .map(|i| i as u8)
}

/// Get Pauling electronegativity for an element by Z (flat-array lookup).
pub fn electronegativity(z: u8) -> f64 {
    if (z as usize) < PAULING_EN.len() {
        PAULING_EN[z as usize]
    } else {
        1.3 // default estimate for superheavy
    }
}

/// s/p/d/f block classification based on electron configuration.
/// Returns 0=s, 1=p, 2=d, 3=f.
pub fn element_block(z: u8) -> u8 {
    match z {
        1 | 2 => 0,                                                   // s-block
        3 | 4 | 11 | 12 | 19 | 20 | 37 | 38 | 55 | 56 | 87 | 88 => 0, // s-block
        57..=71 | 89..=103 => 3,          // f-block (lanthanides, actinides)
        21..=30 | 39..=48 | 72..=80 => 2, // d-block
        _ => 1,                           // p-block
    }
}

/// Period (row) of the periodic table.
pub fn element_period(z: u8) -> u8 {
    match z {
        1..=2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        55..=86 => 6,
        87..=103 => 7,
        _ => 8,
    }
}

/// Group (column, 1-18) of the periodic table. Returns 0 for f-block.
pub fn element_group(z: u8) -> u8 {
    match z {
        1 => 1,
        2 => 18,
        3 | 11 | 19 | 37 | 55 | 87 => 1,
        4 | 12 | 20 | 38 | 56 | 88 => 2,
        57..=71 | 89..=103 => 0, // f-block, no group
        5 | 13 | 31 | 49 | 81 => 13,
        6 | 14 | 32 | 50 | 82 => 14,
        7 | 15 | 33 | 51 | 83 => 15,
        8 | 16 | 34 | 52 | 84 => 16,
        9 | 17 | 35 | 53 | 85 => 17,
        10 | 18 | 36 | 54 | 86 => 18,
        21 | 39 | 72 => 3,
        22 | 40 | 73 => 4,
        23 | 41 | 74 => 5,
        24 | 42 | 75 => 6,
        25 | 43 | 76 => 7,
        26 | 44 | 77 => 8,
        27 | 45 | 78 => 9,
        28 | 46 | 79 => 10,
        29 | 47 | 80 => 11,
        30 | 48 => 12,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Struct-based API tests ----

    #[test]
    fn test_hydrogen() {
        let h = element(1);
        assert_eq!(h.symbol, "H");
        assert_eq!(h.name, "Hydrogen");
        assert!((h.electronegativity - 2.20).abs() < 0.01);
        assert!((h.ionization_energy - 13.598).abs() < 0.01);
        assert_eq!(h.period, 1);
        assert_eq!(h.group, 1);
    }

    #[test]
    fn test_carbon() {
        let c = element(6);
        assert_eq!(c.symbol, "C");
        assert!((c.electronegativity - 2.55).abs() < 0.01);
        assert!((c.ionization_energy - 11.26).abs() < 0.01);
        assert_eq!(c.valence_electrons, 4);
    }

    #[test]
    fn test_silicon() {
        let si = element(14);
        assert_eq!(si.symbol, "Si");
        assert!((si.electronegativity - 1.90).abs() < 0.01);
        assert_eq!(si.group, 14);
        assert_eq!(si.period, 3);
        assert_eq!(si.valence_electrons, 4);
    }

    #[test]
    fn test_gallium() {
        let ga = element(31);
        assert_eq!(ga.symbol, "Ga");
        assert!((ga.electronegativity - 1.81).abs() < 0.01);
    }

    #[test]
    fn test_germanium() {
        let ge = element(32);
        assert_eq!(ge.symbol, "Ge");
        assert!((ge.electronegativity - 2.01).abs() < 0.01);
    }

    #[test]
    fn test_arsenic() {
        let arsenic = element(33);
        assert_eq!(arsenic.symbol, "As");
        assert!((arsenic.electronegativity - 2.18).abs() < 0.01);
    }

    #[test]
    fn test_by_symbol() {
        let o = by_symbol("O").unwrap();
        assert_eq!(o.z, 8);
        assert!((o.electronegativity - 3.44).abs() < 0.01);

        let none = by_symbol("Xx");
        assert!(none.is_none());
    }

    #[test]
    fn test_electronegativity_difference() {
        // Ga(1.81) - As(2.18) = 0.37
        let diff = electronegativity_difference(31, 33);
        assert!((diff - 0.37).abs() < 0.01);

        // Na(0.93) - Cl(3.16) = 2.23
        let diff_nacl = electronegativity_difference(11, 17);
        assert!((diff_nacl - 2.23).abs() < 0.01);
    }

    #[test]
    fn test_all_103_elements() {
        assert_eq!(ELEMENTS.len(), 103);
        for (i, elem) in ELEMENTS.iter().enumerate() {
            assert_eq!(elem.z as usize, i + 1, "element at index {i} has wrong Z");
            assert!(!elem.symbol.is_empty());
            assert!(!elem.name.is_empty());
            assert!(elem.atomic_mass > 0.0);
            assert!(elem.ionization_energy > 0.0);
            assert!(elem.period >= 1 && elem.period <= 7);
        }
    }

    #[test]
    fn test_lawrencium() {
        let lr = element(103);
        assert_eq!(lr.symbol, "Lr");
        assert_eq!(lr.name, "Lawrencium");
        assert_eq!(lr.period, 7);
    }

    #[test]
    fn test_lanthanides_group_zero() {
        for z in 57..=70 {
            let e = element(z);
            assert_eq!(e.group, 0, "{} (Z={}) should be group 0", e.name, z);
        }
    }

    #[test]
    fn test_actinides_group_zero() {
        for z in 89..=102 {
            let e = element(z);
            assert_eq!(e.group, 0, "{} (Z={}) should be group 0", e.name, z);
        }
    }

    #[test]
    fn test_noble_gases_zero_electronegativity() {
        for z in [2u8, 10, 18] {
            let e = element(z);
            assert!(e.electronegativity == 0.0, "{} should have EN=0.0", e.name);
        }
    }

    #[test]
    fn test_key_electronegativities() {
        let cases: &[(u8, f64)] = &[
            (1, 2.20),  // H
            (3, 0.98),  // Li
            (4, 1.57),  // Be
            (5, 2.04),  // B
            (6, 2.55),  // C
            (7, 3.04),  // N
            (8, 3.44),  // O
            (9, 3.98),  // F
            (14, 1.90), // Si
            (32, 2.01), // Ge
            (31, 1.81), // Ga
            (33, 2.18), // As
            (49, 1.78), // In
            (15, 2.19), // P
            (16, 2.58), // S
            (34, 2.55), // Se
            (52, 2.10), // Te
            (30, 1.65), // Zn
            (48, 1.69), // Cd
            (50, 1.96), // Sn
            (82, 2.33), // Pb
        ];
        for &(z, expected) in cases {
            let e = element(z);
            assert!(
                (e.electronegativity - expected).abs() < 0.02,
                "{} (Z={}): EN={}, expected {}",
                e.symbol,
                z,
                e.electronegativity,
                expected
            );
        }
    }

    #[test]
    fn test_struct_and_array_consistency() {
        // Verify ELEMENTS struct data matches the legacy PAULING_EN array
        for z in 1u8..=103 {
            let e = element(z);
            let en_array = PAULING_EN[z as usize];
            assert!(
                (e.electronegativity - en_array).abs() < 0.02,
                "{} (Z={}): struct EN={}, array EN={}",
                e.symbol,
                z,
                e.electronegativity,
                en_array
            );
        }
    }

    #[test]
    fn test_ionization_energy_trends() {
        // IE should generally increase across a period (with d-block exceptions)
        // Simple check: Li < Be, Na < Mg, etc.
        assert!(element(3).ionization_energy < element(4).ionization_energy); // Li < Be
        assert!(element(11).ionization_energy < element(12).ionization_energy); // Na < Mg
        // Noble gases have the highest IE in their period
        assert!(element(10).ionization_energy > element(9).ionization_energy); // Ne > F
    }

    #[test]
    #[should_panic(expected = "atomic number must be 1..=103")]
    fn test_element_z_zero_panics() {
        element(0);
    }

    #[test]
    #[should_panic(expected = "atomic number must be 1..=103")]
    fn test_element_z_104_panics() {
        element(104);
    }

    // ---- Legacy flat-array API tests ----

    #[test]
    fn test_electronegativity_known() {
        assert!((electronegativity(8) - 3.44).abs() < 0.01); // O
        assert!((electronegativity(14) - 1.90).abs() < 0.01); // Si
        assert!((electronegativity(31) - 1.81).abs() < 0.01); // Ga
    }

    #[test]
    fn test_atomic_number() {
        assert_eq!(atomic_number("Si"), Some(14));
        assert_eq!(atomic_number("Ga"), Some(31));
        assert_eq!(atomic_number("Xx"), None);
    }

    #[test]
    fn test_element_block() {
        assert_eq!(element_block(14), 1); // Si = p-block
        assert_eq!(element_block(26), 2); // Fe = d-block
        assert_eq!(element_block(63), 3); // Eu = f-block
    }
}
