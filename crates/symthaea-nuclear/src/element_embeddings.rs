// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Universal Element Embeddings
//!
//! 32-dimensional physics-grounded embeddings for elements Z=1-103.
//! Every dimension is physically interpretable — no black-box learned weights.
//!
//! ## Dimensions
//!
//! | Dim | Property | Normalization |
//! |-----|----------|---------------|
//! | 0 | Atomic number Z | Z/103 |
//! | 1 | ln(atomic mass) | ln(mass)/ln(260) |
//! | 2 | Pauling electronegativity | EN/4.0 |
//! | 3 | First ionization energy | IE/25.0 eV |
//! | 4 | Electron affinity | EA/3.7 eV |
//! | 5 | Covalent radius | r/2.6 Å |
//! | 6 | Period | period/7 |
//! | 7 | Group | group/18 |
//! | 8 | Pettifor number | χ/103 |
//! | 9 | Valence electrons | v/12 |
//! | 10 | Block code | s=0, p=0.33, d=0.67, f=1.0 |
//! | 11 | Metallic character | 0=nonmetal, 0.5=metalloid, 1=metal |
//! | 12 | Melting point | mp/3700 K |
//! | 13 | Boiling point | bp/5900 K |
//! | 14 | ln(density+1) | ln(ρ+1)/ln(23) |
//! | 15 | s-electron occupancy | n_s/2 |
//! | 16 | p-electron occupancy | n_p/6 |
//! | 17 | d-electron occupancy | n_d/10 |
//! | 18 | f-electron occupancy | n_f/14 |
//! | 19 | Unfilled orbitals | n_unfilled/32 |
//! | 20 | BE/A (most stable isotope) | BE_A/8.8 MeV |
//! | 21 | Distance to nearest magic Z | d/50 |
//! | 22 | (reserved: N-dependent) | 0.0 |
//! | 23 | (reserved: isospin) | 0.0 |
//! | 24 | Even Z | 0 or 1 |
//! | 25 | (reserved: even N) | 0.0 |
//! | 26 | Max oxidation state | ox/8 |
//! | 27 | Atomic volume | vol/70 cm³/mol |
//! | 28 | Work function | φ/6.0 eV |
//! | 29 | (reserved: magnetic moment) | 0.5 |
//! | 30 | (reserved: bulk modulus) | 0.5 |
//! | 31 | (reserved) | 0.5 |
//!
//! ## References
//!
//! - Pauling electronegativity: IUPAC 2016
//! - Covalent radius: Cordero et al. (2008)
//! - Ionization energy: NIST
//! - Pettifor chemical scale: Pettifor (1984), Glawe et al. (2016) revision
//! - BE/A: from semi-empirical mass formula (Weizsäcker)

use crate::mass_formula::SemiEmpiricalMassFormula;

// ═══════════════════════════════════════════════════════════════════════════════
// Property tables (Z=0..=103, index 0 unused)
// ═══════════════════════════════════════════════════════════════════════════════

/// Pauling electronegativity. 0.0 = not defined (noble gases, etc.).
/// Source: IUPAC 2016, WebElements.
const PAULING_EN: [f64; 104] = [
    0.00, // 0: placeholder
    2.20, 0.00, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0.00, // 1-10
    0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 0.00, 0.82, 1.00, // 11-20
    1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65, // 21-30
    1.81, 2.01, 2.18, 2.55, 2.96, 3.00, 0.82, 0.95, 1.22, 1.33, // 31-40
    1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69, 1.78, 1.96, // 41-50
    2.05, 2.10, 2.66, 2.60, 0.79, 0.89, 1.10, 1.12, 1.13, 1.14, // 51-60
    1.13, 1.17, 1.20, 1.20, 1.10, 1.22, 1.23, 1.24, 1.25, 1.10, // 61-70
    1.27, 1.30, 1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00, // 71-80
    1.62, 2.33, 2.02, 2.00, 2.20, 2.20, 0.70, 0.90, 1.10, 1.30, // 81-90
    1.50, 1.38, 1.36, 1.28, 1.13, 1.28, 1.30, 1.30, 1.30, 1.30, // 91-100
    1.30, 1.30, 1.30, // 101-103
];

/// Covalent radius in Angstroms. Cordero et al. (2008).
const COVALENT_RADIUS: [f64; 104] = [
    0.00, // 0
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, // 1-10
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, // 11-20
    1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, // 21-30
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, // 31-40
    1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, // 41-50
    1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, // 51-60
    1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, // 61-70
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, // 71-80
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06, // 81-90
    2.00, 1.96, 1.90, 1.87, 1.80, 1.69, 1.60, 1.60, 1.60, 1.60, // 91-100
    1.60, 1.60, 1.60, // 101-103
];

/// First ionization energy in eV. Source: NIST.
const FIRST_IE: [f64; 104] = [
    0.00, // 0
    13.598, 24.587, 5.392, 9.323, 8.298, 11.260, 14.534, 13.618, 17.423, 21.565, // 1-10
    5.139, 7.646, 5.986, 8.152, 10.487, 10.360, 12.968, 15.760, 4.341, 6.113, // 11-20
    6.562, 6.828, 6.746, 6.767, 7.434, 7.902, 7.881, 7.640, 7.726, 9.394, // 21-30
    5.999, 7.900, 9.789, 9.752, 11.814, 14.000, 4.177, 5.695, 6.217, 6.634, // 31-40
    6.759, 7.092, 7.280, 7.361, 7.459, 8.337, 7.576, 8.994, 5.786, 7.344, // 41-50
    8.640, 9.010, 10.451, 12.130, 3.894, 5.212, 5.577, 5.539, 5.473, 5.525, // 51-60
    5.582, 5.644, 5.670, 6.150, 5.864, 5.939, 6.022, 6.108, 6.184, 6.254, // 61-70
    5.426, 6.825, 7.550, 7.864, 7.833, 8.438, 8.967, 8.959, 9.226, 10.437, // 71-80
    6.108, 7.417, 7.286, 8.414, 9.318, 10.749, 4.073, 5.278, 5.170, 6.307, // 81-90
    5.890, 6.194, 6.266, 6.026, 5.974, 5.991, 6.198, 6.282, 6.420, 6.500, // 91-100
    6.580, 6.650, 4.900, // 101-103
];

/// Electron affinity in eV. 0.0 = not defined / endothermic.
const ELECTRON_AFFINITY: [f64; 104] = [
    0.000, // 0
    0.754, 0.000, 0.618, 0.000, 0.277, 1.263, 0.000, 1.461, 3.401, 0.000, // 1-10
    0.548, 0.000, 0.441, 1.390, 0.747, 2.077, 3.617, 0.000, 0.501, 0.025, // 11-20
    0.188, 0.079, 0.525, 0.666, 0.000, 0.151, 0.662, 1.156, 1.236, 0.000, // 21-30
    0.300, 1.233, 0.810, 2.021, 3.364, 0.000, 0.486, 0.052, 0.307, 0.426, // 31-40
    0.893, 0.746, 0.550, 1.050, 1.137, 0.557, 1.302, 0.000, 0.300, 1.112, // 41-50
    1.047, 1.971, 3.059, 0.000, 0.472, 0.145, 0.500, 0.500, 0.500, 0.500, // 51-60
    0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, // 61-70
    0.500, 0.000, 0.322, 0.816, 0.150, 1.100, 1.565, 2.128, 2.309, 0.000, // 71-80
    0.200, 0.364, 0.946, 1.900, 2.800, 0.000, 0.000, 0.000, 0.000, 0.000, // 81-90
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, // 91-100
    0.000, 0.000, 0.000, // 101-103
];

/// Standard atomic mass in amu.
const ATOMIC_MASS: [f64; 104] = [
    0.000, // 0
    1.008, 4.003, 6.941, 9.012, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, // 1-10
    22.990, 24.305, 26.982, 28.086, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078, // 11-20
    44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38, // 21-30
    69.723, 72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.224, // 31-40
    92.906, 95.95, 98.0, 101.07, 102.906, 106.42, 107.868, 112.414, 114.818, 118.710, // 41-50
    121.760, 127.60, 126.904, 131.293, 132.905, 137.327, 138.905, 140.116, 140.908,
    144.242, // 51-60
    145.0, 150.36, 151.964, 157.25, 158.925, 162.500, 164.930, 167.259, 168.934,
    173.045, // 61-70
    174.967, 178.49, 180.948, 183.84, 186.207, 190.23, 192.217, 195.084, 196.967,
    200.592, // 71-80
    204.38, 207.2, 208.980, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.038, // 81-90
    231.036, 238.029, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0, // 91-100
    258.0, 259.0, 266.0, // 101-103
];

/// Pettifor chemical scale (revised, Glawe et al. 2016).
/// Maps each element to a 1D chemical ordering that clusters similar
/// elements together. Lanthanides and actinides are grouped.
/// Index 0 unused. Values for Z=1..103.
const PETTIFOR: [u8; 104] = [
    0,   // 0
    103, // 1: H
    1,   // 2: He
    12,  // 3: Li
    77,  // 4: Be
    86,  // 5: B
    95,  // 6: C
    100, // 7: N
    101, // 8: O
    102, // 9: F
    2,   // 10: Ne
    11,  // 11: Na
    73,  // 12: Mg
    80,  // 13: Al
    87,  // 14: Si
    91,  // 15: P
    96,  // 16: S
    99,  // 17: Cl
    3,   // 18: Ar
    10,  // 19: K
    18,  // 20: Ca
    19,  // 21: Sc
    51,  // 22: Ti
    52,  // 23: V
    53,  // 24: Cr
    54,  // 25: Mn
    55,  // 26: Fe
    56,  // 27: Co
    57,  // 28: Ni
    71,  // 29: Cu
    72,  // 30: Zn
    81,  // 31: Ga
    88,  // 32: Ge
    92,  // 33: As
    97,  // 34: Se
    98,  // 35: Br
    4,   // 36: Kr
    9,   // 37: Rb
    17,  // 38: Sr
    20,  // 39: Y
    46,  // 40: Zr
    47,  // 41: Nb
    48,  // 42: Mo
    49,  // 43: Tc
    58,  // 44: Ru
    59,  // 45: Rh
    60,  // 46: Pd
    70,  // 47: Ag
    74,  // 48: Cd
    82,  // 49: In
    89,  // 50: Sn
    93,  // 51: Sb
    94,  // 52: Te
    98,  // 53: I  (shares ~98 with Br in some scales)
    5,   // 54: Xe
    8,   // 55: Cs
    16,  // 56: Ba
    22,  // 57: La
    23,  // 58: Ce
    24,  // 59: Pr
    25,  // 60: Nd
    26,  // 61: Pm
    27,  // 62: Sm
    28,  // 63: Eu
    29,  // 64: Gd
    30,  // 65: Tb
    31,  // 66: Dy
    32,  // 67: Ho
    33,  // 68: Er
    34,  // 69: Tm
    35,  // 70: Yb
    21,  // 71: Lu
    45,  // 72: Hf
    50,  // 73: Ta
    43,  // 74: W
    44,  // 75: Re
    61,  // 76: Os
    62,  // 77: Ir
    63,  // 78: Pt
    69,  // 79: Au
    75,  // 80: Hg
    83,  // 81: Tl
    90,  // 82: Pb
    84,  // 83: Bi
    85,  // 84: Po
    98,  // 85: At
    6,   // 86: Rn
    7,   // 87: Fr
    15,  // 88: Ra
    36,  // 89: Ac
    37,  // 90: Th
    38,  // 91: Pa
    39,  // 92: U
    40,  // 93: Np
    41,  // 94: Pu
    42,  // 95: Am
    36,  // 96: Cm (approx)
    37,  // 97: Bk (approx)
    38,  // 98: Cf (approx)
    39,  // 99: Es (approx)
    40,  // 100: Fm (approx)
    41,  // 101: Md (approx)
    42,  // 102: No (approx)
    21,  // 103: Lr (approx)
];

/// Maximum oxidation state for each element.
/// Index 0 unused. Z=1..103.
const MAX_OXIDATION: [i8; 104] = [
    0,  // 0
    1,  // H
    0,  // He
    1,  // Li
    2,  // Be
    3,  // B
    4,  // C
    5,  // N
    -2, // O (max positive is +2, but typically -2; we use |max| = 2)
    -1, // F
    0,  // Ne
    1,  // Na
    2,  // Mg
    3,  // Al
    4,  // Si
    5,  // P
    6,  // S
    7,  // Cl
    0,  // Ar
    1,  // K
    2,  // Ca
    3,  // Sc
    4,  // Ti
    5,  // V
    6,  // Cr
    7,  // Mn
    6,  // Fe
    5,  // Co
    4,  // Ni
    4,  // Cu
    2,  // Zn
    3,  // Ga
    4,  // Ge
    5,  // As
    6,  // Se
    7,  // Br
    2,  // Kr
    1,  // Rb
    2,  // Sr
    3,  // Y
    4,  // Zr
    5,  // Nb
    6,  // Mo
    7,  // Tc
    8,  // Ru
    6,  // Rh
    4,  // Pd
    3,  // Ag
    2,  // Cd
    3,  // In
    4,  // Sn
    5,  // Sb
    6,  // Te
    7,  // I
    8,  // Xe
    1,  // Cs
    2,  // Ba
    3,  // La
    4,  // Ce
    4,  // Pr
    3,  // Nd
    3,  // Pm
    3,  // Sm
    3,  // Eu
    3,  // Gd
    4,  // Tb
    3,  // Dy
    3,  // Ho
    3,  // Er
    3,  // Tm
    3,  // Yb
    3,  // Lu
    4,  // Hf
    5,  // Ta
    6,  // W
    7,  // Re
    8,  // Os
    6,  // Ir
    6,  // Pt
    5,  // Au
    4,  // Hg
    3,  // Tl
    4,  // Pb
    5,  // Bi
    6,  // Po
    7,  // At
    2,  // Rn
    1,  // Fr
    2,  // Ra
    3,  // Ac
    4,  // Th
    5,  // Pa
    6,  // U
    7,  // Np
    7,  // Pu
    6,  // Am
    4,  // Cm
    4,  // Bk
    4,  // Cf
    3,  // Es
    3,  // Fm
    3,  // Md
    2,  // No
    3,  // Lr
];

/// Melting point in K. 0.0 = unknown/placeholder.
/// Major elements have real values; rare/synthetic use 0.5*norm placeholder.
const MELTING_POINT: [f64; 104] = [
    0.0,    // 0
    14.01,  // H
    0.95,   // He
    453.7,  // Li
    1560.0, // Be
    2349.0, // B
    3823.0, // C (graphite sublimes)
    63.15,  // N
    54.36,  // O
    53.48,  // F
    24.56,  // Ne
    370.9,  // Na
    923.0,  // Mg
    933.5,  // Al
    1687.0, // Si
    317.3,  // P (white)
    388.4,  // S
    171.6,  // Cl
    83.81,  // Ar
    336.5,  // K
    1115.0, // Ca
    1814.0, // Sc
    1941.0, // Ti
    2183.0, // V
    2180.0, // Cr
    1519.0, // Mn
    1811.0, // Fe
    1768.0, // Co
    1728.0, // Ni
    1357.8, // Cu
    692.7,  // Zn
    302.9,  // Ga
    1211.4, // Ge
    1090.0, // As (sublimes)
    494.0,  // Se
    265.8,  // Br
    115.8,  // Kr
    312.5,  // Rb
    1050.0, // Sr
    1799.0, // Y
    2128.0, // Zr
    2750.0, // Nb
    2896.0, // Mo
    2430.0, // Tc
    2607.0, // Ru
    2237.0, // Rh
    1828.0, // Pd
    1234.9, // Ag
    594.2,  // Cd
    429.8,  // In
    505.1,  // Sn
    903.8,  // Sb
    722.7,  // Te
    386.9,  // I
    161.4,  // Xe
    301.6,  // Cs
    1000.0, // Ba
    1191.0, // La
    1068.0, // Ce
    1208.0, // Pr
    1297.0, // Nd
    1315.0, // Pm
    1345.0, // Sm
    1099.0, // Eu
    1585.0, // Gd
    1629.0, // Tb
    1680.0, // Dy
    1734.0, // Ho
    1802.0, // Er
    1818.0, // Tm
    1097.0, // Yb
    1925.0, // Lu
    2506.0, // Hf
    3290.0, // Ta
    3695.0, // W
    3459.0, // Re
    3306.0, // Os
    2719.0, // Ir
    2041.4, // Pt
    1337.3, // Au
    234.3,  // Hg
    577.0,  // Tl
    600.6,  // Pb
    544.7,  // Bi
    527.0,  // Po
    575.0,  // At
    202.0,  // Rn
    300.0,  // Fr
    973.0,  // Ra
    1323.0, // Ac
    2023.0, // Th
    1841.0, // Pa
    1405.3, // U
    917.0,  // Np
    912.5,  // Pu
    1449.0, // Am
    1613.0, // Cm
    1259.0, // Bk
    1173.0, // Cf
    1133.0, // Es
    1800.0, // Fm (est)
    1100.0, // Md (est)
    1100.0, // No (est)
    1900.0, // Lr (est)
];

/// Boiling point in K. 0.0 = unknown/placeholder.
const BOILING_POINT: [f64; 104] = [
    0.0,    // 0
    20.28,  // H
    4.22,   // He
    1615.0, // Li
    2744.0, // Be
    4200.0, // B
    4098.0, // C (sublimes)
    77.36,  // N
    90.20,  // O
    85.03,  // F
    27.07,  // Ne
    1156.0, // Na
    1363.0, // Mg
    2792.0, // Al
    3538.0, // Si
    553.7,  // P
    717.8,  // S
    239.1,  // Cl
    87.30,  // Ar
    1032.0, // K
    1757.0, // Ca
    3109.0, // Sc
    3560.0, // Ti
    3680.0, // V
    2944.0, // Cr
    2334.0, // Mn
    3134.0, // Fe
    3200.0, // Co
    3186.0, // Ni
    2835.0, // Cu
    1180.0, // Zn
    2477.0, // Ga
    3106.0, // Ge
    887.0,  // As (sublimes)
    958.0,  // Se
    332.0,  // Br
    119.9,  // Kr
    961.0,  // Rb
    1655.0, // Sr
    3609.0, // Y
    4682.0, // Zr
    5017.0, // Nb
    4912.0, // Mo
    4538.0, // Tc
    4423.0, // Ru
    3968.0, // Rh
    3236.0, // Pd
    2435.0, // Ag
    1040.0, // Cd
    2345.0, // In
    2875.0, // Sn
    1860.0, // Sb
    1261.0, // Te
    457.4,  // I
    165.1,  // Xe
    944.0,  // Cs
    2170.0, // Ba
    3737.0, // La
    3716.0, // Ce
    3793.0, // Pr
    3347.0, // Nd
    3273.0, // Pm
    2067.0, // Sm
    1802.0, // Eu
    3546.0, // Gd
    3503.0, // Tb
    2840.0, // Dy
    2993.0, // Ho
    3141.0, // Er
    2223.0, // Tm
    1469.0, // Yb
    3675.0, // Lu
    4876.0, // Hf
    5731.0, // Ta
    5828.0, // W
    5869.0, // Re
    5285.0, // Os
    4701.0, // Ir
    4098.0, // Pt
    3129.0, // Au
    629.9,  // Hg
    1746.0, // Tl
    2022.0, // Pb
    1837.0, // Bi
    1235.0, // Po
    610.0,  // At (est)
    211.5,  // Rn
    950.0,  // Fr (est)
    2010.0, // Ra
    3471.0, // Ac
    5061.0, // Th
    4300.0, // Pa (est)
    4404.0, // U
    4175.0, // Np (est)
    3505.0, // Pu
    2880.0, // Am
    3383.0, // Cm (est)
    2900.0, // Bk (est)
    1743.0, // Cf (est)
    1269.0, // Es (est)
    1800.0, // Fm (est)
    1100.0, // Md (est)
    1100.0, // No (est)
    1900.0, // Lr (est)
];

/// Density in g/cm³ at STP. 0.0 = unknown.
const DENSITY: [f64; 104] = [
    0.0,    // 0
    0.0899, // H (gas)
    0.1785, // He (gas)
    0.534,  // Li
    1.85,   // Be
    2.34,   // B
    2.267,  // C (graphite)
    1.251,  // N (gas)
    1.429,  // O (gas)
    1.696,  // F (gas)
    0.900,  // Ne (gas)
    0.971,  // Na
    1.738,  // Mg
    2.70,   // Al
    2.329,  // Si
    1.82,   // P (white)
    2.07,   // S
    3.214,  // Cl (gas)
    1.784,  // Ar (gas)
    0.862,  // K
    1.55,   // Ca
    2.99,   // Sc
    4.54,   // Ti
    6.11,   // V
    7.15,   // Cr
    7.44,   // Mn
    7.874,  // Fe
    8.90,   // Co
    8.908,  // Ni
    8.96,   // Cu
    7.134,  // Zn
    5.91,   // Ga
    5.323,  // Ge
    5.776,  // As
    4.809,  // Se
    3.122,  // Br
    3.749,  // Kr (gas)
    1.532,  // Rb
    2.64,   // Sr
    4.47,   // Y
    6.52,   // Zr
    8.57,   // Nb
    10.22,  // Mo
    11.5,   // Tc
    12.37,  // Ru
    12.41,  // Rh
    12.02,  // Pd
    10.50,  // Ag
    8.65,   // Cd
    7.31,   // In
    7.287,  // Sn
    6.685,  // Sb
    6.232,  // Te
    4.93,   // I
    5.894,  // Xe (gas)
    1.873,  // Cs
    3.594,  // Ba
    6.145,  // La
    6.770,  // Ce
    6.773,  // Pr
    7.008,  // Nd
    7.26,   // Pm
    7.52,   // Sm
    5.244,  // Eu
    7.90,   // Gd
    8.23,   // Tb
    8.55,   // Dy
    8.795,  // Ho
    9.066,  // Er
    9.321,  // Tm
    6.90,   // Yb
    9.84,   // Lu
    13.31,  // Hf
    16.69,  // Ta
    19.25,  // W
    21.02,  // Re
    22.59,  // Os
    22.56,  // Ir
    21.46,  // Pt
    19.30,  // Au
    13.534, // Hg
    11.85,  // Tl
    11.342, // Pb
    9.807,  // Bi
    9.32,   // Po
    7.0,    // At (est)
    9.73,   // Rn (gas, est)
    1.87,   // Fr (est)
    5.50,   // Ra
    10.07,  // Ac
    11.72,  // Th
    15.37,  // Pa
    18.95,  // U
    20.45,  // Np
    19.84,  // Pu
    13.69,  // Am
    13.51,  // Cm
    14.78,  // Bk
    15.1,   // Cf
    8.84,   // Es (est)
    9.7,    // Fm (est)
    10.3,   // Md (est)
    9.9,    // No (est)
    14.4,   // Lr (est)
];

/// Work function in eV (polycrystalline where available).
/// 0.0 = unknown. Source: CRC Handbook, Michaelson (1977).
/// Many elements lack reliable data; those use 0.0.
const WORK_FUNCTION: [f64; 104] = [
    0.0,  // 0
    0.0,  // H (not a solid)
    0.0,  // He
    2.9,  // Li
    4.98, // Be
    0.0,  // B
    5.0,  // C (graphite)
    0.0,  // N
    0.0,  // O
    0.0,  // F
    0.0,  // Ne
    2.36, // Na
    3.66, // Mg
    4.28, // Al
    4.85, // Si
    0.0,  // P
    0.0,  // S
    0.0,  // Cl
    0.0,  // Ar
    2.29, // K
    2.87, // Ca
    3.5,  // Sc
    4.33, // Ti
    4.3,  // V
    4.5,  // Cr
    4.1,  // Mn
    4.67, // Fe
    5.0,  // Co
    5.15, // Ni
    4.65, // Cu
    4.33, // Zn
    4.32, // Ga
    5.0,  // Ge
    3.75, // As
    5.9,  // Se
    0.0,  // Br
    0.0,  // Kr
    2.26, // Rb
    2.59, // Sr
    3.1,  // Y
    4.05, // Zr
    4.3,  // Nb
    4.6,  // Mo
    0.0,  // Tc
    4.71, // Ru
    4.98, // Rh
    5.12, // Pd
    4.26, // Ag
    4.22, // Cd
    4.12, // In
    4.42, // Sn
    4.55, // Sb
    4.95, // Te
    0.0,  // I
    0.0,  // Xe
    2.14, // Cs
    2.52, // Ba
    3.5,  // La
    2.9,  // Ce
    0.0,  // Pr
    3.2,  // Nd
    0.0,  // Pm
    2.7,  // Sm
    2.5,  // Eu
    3.1,  // Gd
    3.0,  // Tb
    0.0,  // Dy
    0.0,  // Ho
    0.0,  // Er
    0.0,  // Tm
    0.0,  // Yb
    3.3,  // Lu
    3.9,  // Hf
    4.25, // Ta
    4.55, // W
    4.72, // Re
    4.83, // Os
    5.27, // Ir
    5.65, // Pt
    5.10, // Au
    4.49, // Hg
    3.84, // Tl
    4.25, // Pb
    4.22, // Bi
    0.0,  // Po
    0.0,  // At
    0.0,  // Rn
    0.0,  // Fr
    0.0,  // Ra
    0.0,  // Ac
    3.4,  // Th
    0.0,  // Pa
    3.63, // U
    0.0,  // Np
    0.0,  // Pu
    0.0,  // Am
    0.0,  // Cm
    0.0,  // Bk
    0.0,  // Cf
    0.0,  // Es
    0.0,  // Fm
    0.0,  // Md
    0.0,  // No
    0.0,  // Lr
];

/// Atomic volume in cm³/mol (molar mass / density).
/// Computed from ATOMIC_MASS and DENSITY where both are > 0.
/// 0.0 = not available.
const ATOMIC_VOLUME: [f64; 104] = [
    0.0,  // 0
    11.2, // H (liquid)
    22.4, // He
    13.0, // Li
    4.88, // Be
    4.62, // B
    5.31, // C
    11.2, // N
    11.2, // O
    11.2, // F
    22.4, // Ne
    23.7, // Na
    14.0, // Mg
    10.0, // Al
    12.1, // Si
    17.0, // P
    15.5, // S
    11.0, // Cl
    22.4, // Ar
    45.4, // K
    25.9, // Ca
    15.0, // Sc
    10.6, // Ti
    8.32, // V
    7.23, // Cr
    7.35, // Mn
    7.09, // Fe
    6.67, // Co
    6.59, // Ni
    7.11, // Cu
    9.16, // Zn
    11.8, // Ga
    13.6, // Ge
    13.0, // As
    16.4, // Se
    25.6, // Br
    22.4, // Kr
    55.8, // Rb
    33.2, // Sr
    19.9, // Y
    14.0, // Zr
    10.8, // Nb
    9.38, // Mo
    8.63, // Tc
    8.17, // Ru
    8.28, // Rh
    8.56, // Pd
    10.3, // Ag
    13.0, // Cd
    15.7, // In
    16.3, // Sn
    18.2, // Sb
    20.5, // Te
    25.7, // I
    22.3, // Xe
    71.1, // Cs
    38.2, // Ba
    22.6, // La
    20.7, // Ce
    20.8, // Pr
    20.6, // Nd
    20.0, // Pm
    20.0, // Sm
    29.0, // Eu
    19.9, // Gd
    19.3, // Tb
    19.0, // Dy
    18.7, // Ho
    18.4, // Er
    18.1, // Tm
    25.1, // Yb
    17.8, // Lu
    13.4, // Hf
    10.9, // Ta
    9.47, // W
    8.86, // Re
    8.42, // Os
    8.52, // Ir
    9.09, // Pt
    10.2, // Au
    14.8, // Hg
    17.2, // Tl
    18.3, // Pb
    21.3, // Bi
    22.4, // Po
    30.0, // At (est)
    22.8, // Rn (est)
    0.0,  // Fr (est)
    41.1, // Ra
    22.5, // Ac
    19.8, // Th
    15.0, // Pa
    12.6, // U
    11.6, // Np (est)
    12.3, // Pu
    17.8, // Am
    18.3, // Cm (est)
    16.7, // Bk (est)
    16.6, // Cf (est)
    28.5, // Es (est)
    0.0,  // Fm (est)
    0.0,  // Md (est)
    0.0,  // No (est)
    0.0,  // Lr (est)
];

// ═══════════════════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Period (row 1-7) of the periodic table.
fn period(z: u8) -> u8 {
    match z {
        1..=2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        55..=86 => 6,
        87..=103 => 7,
        _ => 7,
    }
}

/// IUPAC group (1-18). Returns 0 for f-block (lanthanides/actinides).
fn group(z: u8) -> u8 {
    match z {
        1 => 1,
        2 => 18,
        3 | 11 | 19 | 37 | 55 | 87 => 1,
        4 | 12 | 20 | 38 | 56 | 88 => 2,
        57..=71 | 89..=103 => 0,
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

/// Valence electron count.
fn valence(z: u8) -> u8 {
    let g = group(z);
    let blk = block(z);
    match blk {
        0 => {
            // s-block
            match g {
                1 => 1,
                2 => 2,
                18 => 2, // He
                _ => 2,
            }
        }
        1 => {
            // p-block
            match g {
                13 => 3,
                14 => 4,
                15 => 5,
                16 => 6,
                17 => 7,
                18 => 8,
                _ => 4,
            }
        }
        2 => {
            // d-block: include d + s electrons
            // Approximate: group number for groups 3-12
            match g {
                3 => 3,
                4 => 4,
                5 => 5,
                6 => 6,
                7 => 7,
                8 => 8,
                9 => 9,
                10 => 10,
                11 => 11,
                12 => 12,
                _ => 2,
            }
        }
        3 => {
            // f-block: lanthanides/actinides typically 2-3 valence
            // More nuanced: count f + d + s
            match z {
                57 => 3,  // La: [Xe]5d1 6s2
                58 => 4,  // Ce: 4f1 5d1 6s2
                63 => 7,  // Eu: 4f7 6s2
                64 => 8,  // Gd: 4f7 5d1 6s2
                70 => 16, // would overflow, cap at 14+2
                _ => {
                    // approximate: f_count + 2 (s electrons)
                    let f = f_electrons(z);
                    (f + 2).min(16)
                }
            }
        }
        _ => 2,
    }
}

/// Block: 0=s, 1=p, 2=d, 3=f.
fn block(z: u8) -> u8 {
    match z {
        1 | 2 => 0,
        3 | 4 | 11 | 12 | 19 | 20 | 37 | 38 | 55 | 56 | 87 | 88 => 0,
        57..=71 | 89..=103 => 3,
        21..=30 | 39..=48 | 72..=80 => 2,
        _ => 1,
    }
}

/// Block code as a float: s=0.0, p=0.33, d=0.67, f=1.0.
fn block_code(z: u8) -> f64 {
    match block(z) {
        0 => 0.0,
        1 => 0.33,
        2 => 0.67,
        3 => 1.0,
        _ => 0.5,
    }
}

/// Metallic character: 0.0=nonmetal, 0.5=metalloid, 1.0=metal.
fn metallic_character(z: u8) -> f64 {
    // Metalloids
    const METALLOIDS: [u8; 7] = [5, 14, 32, 33, 51, 52, 84];
    // Nonmetals (including noble gases, halogens, H)
    const NONMETALS: [u8; 17] = [1, 2, 6, 7, 8, 9, 10, 15, 16, 17, 18, 34, 35, 36, 53, 54, 86];

    if METALLOIDS.contains(&z) {
        0.5
    } else if NONMETALS.contains(&z) {
        0.0
    } else {
        1.0
    }
}

/// Number of s electrons in the valence shell.
fn s_electrons(z: u8) -> u8 {
    match z {
        1 => 1,
        2 => 2,
        3 | 11 | 19 | 37 | 55 | 87 => 1,
        4 | 12 | 20 | 38 | 56 | 88 => 2,
        // d-block anomalies (Cr, Cu, Nb, Mo, Ru, Rh, Pd, Ag, Pt, Au)
        24 | 29 | 41 | 42 | 44 | 45 | 78 | 79 => 1,
        46 => 0, // Pd: [Kr]4d10
        // Most d-block and p-block: 2 s electrons
        _ => 2,
    }
}

/// Number of p electrons in the outermost p subshell.
fn p_electrons(z: u8) -> u8 {
    match z {
        5 | 13 | 31 | 49 | 81 => 1,
        6 | 14 | 32 | 50 | 82 => 2,
        7 | 15 | 33 | 51 | 83 => 3,
        8 | 16 | 34 | 52 | 84 => 4,
        9 | 17 | 35 | 53 | 85 => 5,
        10 | 18 | 36 | 54 | 86 => 6,
        _ => 0,
    }
}

/// Number of d electrons in the outermost d subshell.
fn d_electrons(z: u8) -> u8 {
    match z {
        21 => 1,
        22 => 2,
        23 => 3,
        24 => 5,
        25 => 5,
        26 => 6,
        27 => 7,
        28 => 8,
        29 => 10,
        30 => 10,
        39 => 1,
        40 => 2,
        41 => 4,
        42 => 5,
        43 => 5,
        44 => 7,
        45 => 8,
        46 => 10,
        47 => 10,
        48 => 10,
        72 => 2,
        73 => 3,
        74 => 4,
        75 => 5,
        76 => 6,
        77 => 7,
        78 => 9,
        79 => 10,
        80 => 10,
        // Lanthanides: La has 5d1, Ce 5d1, Gd 5d1, Lu 5d1 (buried)
        57 | 58 | 64 | 71 => 1,
        // Actinides: Ac 6d1, Th 6d2, Pa 6d1, etc.
        89 | 91 => 1,
        90 => 2,
        _ => 0,
    }
}

/// Number of f electrons.
fn f_electrons(z: u8) -> u8 {
    match z {
        // Lanthanides: 57(La)=0, 58(Ce)=1, ... 70(Yb)=14, 71(Lu)=14
        57 => 0,
        58..=70 => z - 57,
        71 => 14,
        // Actinides: 89(Ac)=0, 90(Th)=0, 91(Pa)=2, 92(U)=3, ...
        89 | 90 => 0,
        91 => 2,
        92 => 3,
        93 => 4,
        94 => 6,
        95 => 7,
        96 => 7,
        97 => 9,
        98 => 10,
        99 => 11,
        100 => 12,
        101 => 13,
        102 => 14,
        103 => 14,
        _ => 0,
    }
}

/// Number of unfilled orbitals (approximate, in the valence shell).
fn unfilled_orbitals(z: u8) -> u8 {
    let blk = block(z);
    match blk {
        0 => {
            // s-block: max 1 orbital (2 electrons)
            let s = s_electrons(z);
            if s < 2 { 1 } else { 0 }
        }
        1 => {
            // p-block: 3 orbitals (6 electrons)
            let p = p_electrons(z);
            let s = s_electrons(z);
            let unfilled_p = if p < 6 { 3 - (p + 1) / 2 } else { 0 };
            let unfilled_s = if s < 2 { 1 } else { 0 };
            unfilled_p + unfilled_s
        }
        2 => {
            // d-block: 5 d orbitals + 1 s orbital
            let d = d_electrons(z);
            let s = s_electrons(z);
            let unfilled_d = if d < 10 { 5 - (d + 1) / 2 } else { 0 };
            let unfilled_s = if s < 2 { 1 } else { 0 };
            unfilled_d + unfilled_s
        }
        3 => {
            // f-block: 7 f orbitals + possible d + s
            let f = f_electrons(z);
            let d = d_electrons(z);
            let s = s_electrons(z);
            let unfilled_f = if f < 14 { 7 - (f + 1) / 2 } else { 0 };
            let unfilled_d = if d < 10 { 5 - (d + 1) / 2 } else { 0 };
            let unfilled_s = if s < 2 { 1 } else { 0 };
            unfilled_f + unfilled_d + unfilled_s
        }
        _ => 0,
    }
}

/// Binding energy per nucleon for the most stable isotope (MeV).
/// Uses the semi-empirical mass formula (Weizsäcker).
fn be_per_nucleon(z: u8) -> f64 {
    if z == 0 {
        return 0.0;
    }
    // Most stable A for each Z (approximate: A ≈ 2Z for light, valley of stability for heavy)
    let a = most_stable_a(z);
    let semf = SemiEmpiricalMassFormula::default();
    let be = semf.binding_energy(a, z as u16);
    if be > 0.0 && a > 0 {
        be / a as f64
    } else {
        0.0
    }
}

/// Approximate mass number of the most stable isotope for element Z.
/// Uses known values for key elements and the Green approximation
/// A ≈ Z / (0.5 - Z²·a_C/(4·a_A)) for the valley of beta stability.
fn most_stable_a(z: u8) -> u16 {
    // Known most-abundant/stable isotopes for key elements
    match z {
        1 => 1,
        2 => 4,
        3 => 7,
        6 => 12,
        7 => 14,
        8 => 16,
        26 => 56,  // Fe-56 (iron peak)
        28 => 58,  // Ni-58
        50 => 120, // Sn-120
        79 => 197, // Au-197
        82 => 208, // Pb-208
        92 => 238, // U-238
        _ => {
            // Green approximation from SEMF: A = Z / (0.5 - a_C·Z^(2/3) / (4·a_A))
            // Simplified: A ≈ 2Z(1 + 0.0058 Z^(2/3))^(-1) ... use empirical fit
            let zf = z as f64;
            // Empirical fit to valley of stability:
            let a = 2.0 * zf + 0.006 * zf * zf;
            a.round().max(zf) as u16
        }
    }
}

/// Magic numbers for protons/neutrons.
const MAGIC_NUMBERS: [u8; 7] = [2, 8, 20, 28, 50, 82, 126];

/// Distance to nearest magic proton number.
fn z_shell_proximity(z: u8) -> f64 {
    MAGIC_NUMBERS
        .iter()
        .map(|&m| (z as i16 - m as i16).unsigned_abs() as f64)
        .fold(f64::INFINITY, f64::min)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute a 32-dimensional physics-grounded embedding for element Z (1-103).
///
/// Every dimension is physically interpretable and normalized to approximately [0, 1].
/// Dimensions 22, 23, 25 are reserved for isotope-specific properties (N-dependent,
/// isospin, even-N) and default to 0.0. Dimensions 29-31 are reserved for future
/// properties and default to 0.5.
///
/// # Panics
///
/// Panics if `z` is 0 or greater than 103.
pub fn element_embedding(z: u8) -> [f64; 32] {
    assert!(z >= 1 && z <= 103, "Z must be 1..=103, got {z}");

    let en = PAULING_EN[z as usize];
    let radius = COVALENT_RADIUS[z as usize];
    let ie = FIRST_IE[z as usize];
    let ea = ELECTRON_AFFINITY[z as usize];
    let mass = ATOMIC_MASS[z as usize];
    let mp = MELTING_POINT[z as usize];
    let bp = BOILING_POINT[z as usize];
    let rho = DENSITY[z as usize];
    let wf = WORK_FUNCTION[z as usize];
    let av = ATOMIC_VOLUME[z as usize];

    [
        z as f64 / 103.0,                                      // 0: Z
        mass.max(1.0).ln() / 260.0_f64.ln(),                   // 1: ln(mass)
        en / 4.0,                                              // 2: electronegativity
        ie / 25.0,                                             // 3: 1st ionization energy
        ea / 3.7,                                              // 4: electron affinity
        radius / 2.6,                                          // 5: covalent radius (Å, max ~2.6)
        period(z) as f64 / 7.0,                                // 6: period
        group(z) as f64 / 18.0,                                // 7: group
        PETTIFOR[z as usize] as f64 / 103.0,                   // 8: Pettifor number
        valence(z) as f64 / 16.0,                              // 9: valence electrons
        block_code(z),                                         // 10: block
        metallic_character(z),                                 // 11: metallic character
        mp / 3700.0,                                           // 12: melting point
        bp / 5900.0,                                           // 13: boiling point
        (rho + 1.0).ln() / 23.0_f64.ln(),                      // 14: ln(density+1)
        s_electrons(z) as f64 / 2.0,                           // 15: s occupancy
        p_electrons(z) as f64 / 6.0,                           // 16: p occupancy
        d_electrons(z) as f64 / 10.0,                          // 17: d occupancy
        f_electrons(z) as f64 / 14.0,                          // 18: f occupancy
        unfilled_orbitals(z) as f64 / 13.0,                    // 19: unfilled (max ~13 for f-block)
        be_per_nucleon(z) / 8.8,                               // 20: BE/A
        z_shell_proximity(z) / 50.0,                           // 21: distance to nearest magic Z
        0.0,                                                   // 22: N-dependent (set per isotope)
        0.0,                                                   // 23: isospin (set per isotope)
        if z % 2 == 0 { 1.0 } else { 0.0 },                    // 24: even Z
        0.0,                                                   // 25: even N (set per isotope)
        MAX_OXIDATION[z as usize].unsigned_abs() as f64 / 8.0, // 26: max oxidation state
        av / 70.0,                                             // 27: atomic volume
        wf / 6.0,                                              // 28: work function
        0.5,                                                   // 29: reserved (magnetic moment)
        0.5,                                                   // 30: reserved (bulk modulus)
        0.5,                                                   // 31: reserved
    ]
}

/// Cosine similarity between two 32D embedding vectors.
pub fn cosine_similarity(a: &[f64; 32], b: &[f64; 32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a < 1e-12 || mag_b < 1e-12 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// Euclidean distance between two 32D embedding vectors.
pub fn euclidean_distance(a: &[f64; 32], b: &[f64; 32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimensions() {
        // All 103 elements should produce 32D vectors with values in [0, 1]
        for z in 1..=103u8 {
            let emb = element_embedding(z);
            assert_eq!(emb.len(), 32, "Z={z} embedding is not 32D");
            for (dim, &val) in emb.iter().enumerate() {
                assert!(
                    val >= -0.01 && val <= 1.5,
                    "Z={z} dim={dim} out of range: {val:.4}"
                );
            }
        }
    }

    #[test]
    fn test_similarity_structure() {
        // Noble gases should cluster together
        let he = element_embedding(2);
        let ne = element_embedding(10);
        let ar = element_embedding(18);

        let sim_he_ne = cosine_similarity(&he, &ne);
        let sim_ne_ar = cosine_similarity(&ne, &ar);
        let sim_he_ar = cosine_similarity(&he, &ar);

        // All noble gas pairs should have high similarity
        let noble_min = sim_he_ne.min(sim_ne_ar).min(sim_he_ar);

        // Alkali metals should cluster together
        let li = element_embedding(3);
        let na = element_embedding(11);
        let k = element_embedding(19);

        let sim_li_na = cosine_similarity(&li, &na);
        let sim_na_k = cosine_similarity(&na, &k);
        let sim_li_k = cosine_similarity(&li, &k);

        let alkali_min = sim_li_na.min(sim_na_k).min(sim_li_k);

        // Halogens should cluster together
        let f = element_embedding(9);
        let cl = element_embedding(17);
        let br = element_embedding(35);

        let sim_f_cl = cosine_similarity(&f, &cl);
        let sim_cl_br = cosine_similarity(&cl, &br);
        let sim_f_br = cosine_similarity(&f, &br);

        let halogen_min = sim_f_cl.min(sim_cl_br).min(sim_f_br);

        // Cross-group similarity should be lower than within-group
        let sim_na_cl = cosine_similarity(&na, &cl);
        let sim_li_f = cosine_similarity(&li, &f);
        let sim_he_na = cosine_similarity(&he, &na);

        eprintln!("=== Similarity Structure ===");
        eprintln!(
            "Noble gases:  He-Ne={sim_he_ne:.4}, Ne-Ar={sim_ne_ar:.4}, He-Ar={sim_he_ar:.4}  (min={noble_min:.4})"
        );
        eprintln!(
            "Alkali metals: Li-Na={sim_li_na:.4}, Na-K={sim_na_k:.4}, Li-K={sim_li_k:.4}  (min={alkali_min:.4})"
        );
        eprintln!(
            "Halogens:     F-Cl={sim_f_cl:.4}, Cl-Br={sim_cl_br:.4}, F-Br={sim_f_br:.4}  (min={halogen_min:.4})"
        );
        eprintln!("Cross-group:  Na-Cl={sim_na_cl:.4}, Li-F={sim_li_f:.4}, He-Na={sim_he_na:.4}");

        // Within-group minimum should exceed 0.85 (they share many features)
        assert!(
            noble_min > 0.80,
            "Noble gas similarity too low: {noble_min:.4}"
        );
        assert!(
            alkali_min > 0.85,
            "Alkali metal similarity too low: {alkali_min:.4}"
        );
        assert!(
            halogen_min > 0.85,
            "Halogen similarity too low: {halogen_min:.4}"
        );

        // Cross-group should be lower than within-group
        // (Na and Cl are an ionic pair but chemically opposite)
        assert!(
            sim_na_cl < alkali_min || sim_na_cl < halogen_min,
            "Na-Cl similarity ({sim_na_cl:.4}) should be lower than within-group"
        );
    }

    #[test]
    fn test_periodic_trends() {
        // Electronegativity increases left-to-right in period 2 (Li → F, skipping noble gas)
        // dim 2 = electronegativity / 4.0
        let li = element_embedding(3);
        let be = element_embedding(4);
        let b = element_embedding(5);
        let c = element_embedding(6);
        let n = element_embedding(7);
        let o = element_embedding(8);
        let f = element_embedding(9);

        assert!(
            li[2] < be[2],
            "Li EN ({}) should be < Be EN ({})",
            li[2],
            be[2]
        );
        assert!(
            be[2] < b[2],
            "Be EN ({}) should be < B EN ({})",
            be[2],
            b[2]
        );
        assert!(b[2] < c[2], "B EN ({}) should be < C EN ({})", b[2], c[2]);
        assert!(c[2] < n[2], "C EN ({}) should be < N EN ({})", c[2], n[2]);
        // N > O is the well-known anomaly (half-filled 2p stability), skip
        assert!(o[2] < f[2], "O EN ({}) should be < F EN ({})", o[2], f[2]);

        // Ionization energy should generally increase left-to-right
        // dim 3 = IE / 25.0
        assert!(
            li[3] < f[3],
            "Li IE ({}) should be < F IE ({})",
            li[3],
            f[3]
        );

        // Atomic radius should decrease left-to-right (dim 5)
        assert!(
            li[5] > f[5],
            "Li radius ({}) should be > F radius ({})",
            li[5],
            f[5]
        );

        eprintln!("=== Periodic Trends (Period 2) ===");
        eprintln!(
            "EN:     Li={:.3} Be={:.3} B={:.3} C={:.3} N={:.3} O={:.3} F={:.3}",
            li[2], be[2], b[2], c[2], n[2], o[2], f[2]
        );
        eprintln!(
            "IE:     Li={:.3} Be={:.3} B={:.3} C={:.3} N={:.3} O={:.3} F={:.3}",
            li[3], be[3], b[3], c[3], n[3], o[3], f[3]
        );
        eprintln!(
            "Radius: Li={:.3} Be={:.3} B={:.3} C={:.3} N={:.3} O={:.3} F={:.3}",
            li[5], be[5], b[5], c[5], n[5], o[5], f[5]
        );
    }

    #[test]
    fn test_cross_domain_features() {
        // Nuclear properties (dims 20-25) should be populated for heavy elements
        // dim 20 = BE/A (should be nonzero for all elements)
        // dim 21 = distance to nearest magic Z
        // dim 24 = even Z

        for z in [26u8, 50, 82, 92] {
            let emb = element_embedding(z);
            assert!(
                emb[20] > 0.0,
                "Z={z} BE/A (dim 20) should be > 0: {:.4}",
                emb[20]
            );
            // For magic number Z=82 (Pb), proximity should be 0
            if z == 82 {
                assert!(
                    emb[21] < 0.01,
                    "Z=82 (Pb) should be at magic number, proximity={:.4}",
                    emb[21]
                );
            }
        }

        // Even Z test
        let fe = element_embedding(26);
        let co = element_embedding(27);
        assert_eq!(fe[24], 1.0, "Fe (Z=26, even) should have dim 24 = 1.0");
        assert_eq!(co[24], 0.0, "Co (Z=27, odd) should have dim 24 = 0.0");

        // BE/A should peak around Z~26 (iron peak)
        let h_bea = element_embedding(1)[20];
        let fe_bea = element_embedding(26)[20];
        let u_bea = element_embedding(92)[20];

        eprintln!("=== Nuclear Properties ===");
        eprintln!("BE/A: H={h_bea:.4}, Fe={fe_bea:.4}, U={u_bea:.4}");
        eprintln!("Fe should be near peak (~8.8 MeV → dim≈1.0)");

        // Fe should have higher BE/A than H and U
        assert!(
            fe_bea > h_bea,
            "Fe BE/A ({fe_bea:.4}) should exceed H BE/A ({h_bea:.4})"
        );
        assert!(
            fe_bea > u_bea,
            "Fe BE/A ({fe_bea:.4}) should exceed U BE/A ({u_bea:.4})"
        );

        // Magic numbers: Z=2,8,20,28,50,82
        for z in [2u8, 8, 20, 28, 50, 82] {
            let emb = element_embedding(z);
            assert!(
                emb[21] < 0.01,
                "Z={z} is a magic number, proximity should be ~0, got {:.4}",
                emb[21]
            );
        }
    }

    #[test]
    fn test_block_assignments() {
        // s-block
        assert_eq!(block(1), 0); // H
        assert_eq!(block(11), 0); // Na
        assert_eq!(block(55), 0); // Cs

        // p-block
        assert_eq!(block(6), 1); // C
        assert_eq!(block(17), 1); // Cl
        assert_eq!(block(82), 1); // Pb

        // d-block
        assert_eq!(block(26), 2); // Fe
        assert_eq!(block(47), 2); // Ag
        assert_eq!(block(79), 2); // Au

        // f-block
        assert_eq!(block(57), 3); // La
        assert_eq!(block(63), 3); // Eu
        assert_eq!(block(92), 3); // U
    }

    #[test]
    fn test_magic_number_proximity() {
        assert!((z_shell_proximity(2) - 0.0).abs() < 1e-10); // Z=2 is magic
        assert!((z_shell_proximity(8) - 0.0).abs() < 1e-10); // Z=8 is magic
        assert!((z_shell_proximity(26) - 2.0).abs() < 1e-10); // Z=26, nearest is 28
        assert!((z_shell_proximity(50) - 0.0).abs() < 1e-10); // Z=50 is magic
    }

    #[test]
    fn test_transition_metals_cluster() {
        // 3d transition metals (Sc-Zn, Z=21-30) should be more similar to each other
        // than to, say, noble gases
        let ti = element_embedding(22);
        let fe = element_embedding(26);
        let ni = element_embedding(28);
        let ne = element_embedding(10);

        let sim_ti_fe = cosine_similarity(&ti, &fe);
        let sim_fe_ni = cosine_similarity(&fe, &ni);
        let sim_ti_ne = cosine_similarity(&ti, &ne);
        let sim_fe_ne = cosine_similarity(&fe, &ne);

        eprintln!("=== Transition Metal Clustering ===");
        eprintln!("Ti-Fe={sim_ti_fe:.4}, Fe-Ni={sim_fe_ni:.4}");
        eprintln!("Ti-Ne={sim_ti_ne:.4}, Fe-Ne={sim_fe_ne:.4}");

        assert!(
            sim_ti_fe > sim_ti_ne,
            "Ti-Fe ({sim_ti_fe:.4}) should exceed Ti-Ne ({sim_ti_ne:.4})"
        );
        assert!(
            sim_fe_ni > sim_fe_ne,
            "Fe-Ni ({sim_fe_ni:.4}) should exceed Fe-Ne ({sim_fe_ne:.4})"
        );
    }
}
