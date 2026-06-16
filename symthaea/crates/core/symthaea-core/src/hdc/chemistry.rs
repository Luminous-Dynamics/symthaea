// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Chemistry Module
//!
//! Pure numerical/symbolic chemistry computations.
//! All 118 elements, stoichiometry, thermochemistry, equilibrium, and kinetics.
//!
//! **IMPORTANT**: Molecules are NOT encoded as HDC bundles of elements.
//! Chemistry is non-linear: H₂O ≠ bundle(H₂, O). This module is purely
//! numerical — element data, formula parsing, and physics calculations.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. Periodic Table
// ---------------------------------------------------------------------------

/// Block in the periodic table (s, p, d, or f).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Block {
    S,
    P,
    D,
    F,
}

/// A single element with standard physical/chemical properties.
#[derive(Debug, Clone, PartialEq)]
pub struct Element {
    pub atomic_number: u32,
    pub symbol: &'static str,
    pub name: &'static str,
    /// Atomic mass in g/mol (IUPAC 2021)
    pub atomic_mass: f64,
    /// Pauling electronegativity (0.0 if unknown or noble gas without stable compounds)
    pub electronegativity: f64,
    pub period: u32,
    /// Group 1-18; 0 for lanthanides and actinides
    pub group: u32,
    pub block: Block,
    /// First ionization energy in kJ/mol
    pub ionization_energy: f64,
    /// Electron affinity in kJ/mol (0.0 if anion is unstable)
    pub electron_affinity: f64,
    /// Density in g/cm³ at STP; 0.0 for gases where convention differs
    pub density: f64,
    /// Melting point in K
    pub melting_point: f64,
    /// Boiling point in K
    pub boiling_point: f64,
}

impl Element {
    #[allow(clippy::too_many_arguments)]
    const fn new(
        atomic_number: u32,
        symbol: &'static str,
        name: &'static str,
        atomic_mass: f64,
        electronegativity: f64,
        period: u32,
        group: u32,
        block: Block,
        ionization_energy: f64,
        electron_affinity: f64,
        density: f64,
        melting_point: f64,
        boiling_point: f64,
    ) -> Self {
        Self {
            atomic_number,
            symbol,
            name,
            atomic_mass,
            electronegativity,
            period,
            group,
            block,
            ionization_energy,
            electron_affinity,
            density,
            melting_point,
            boiling_point,
        }
    }
}

/// Returns all 118 elements with accurate NIST/IUPAC data.
pub fn all_elements() -> Vec<Element> {
    vec![
        // Period 1
        Element::new(
            1,
            "H",
            "Hydrogen",
            1.008,
            2.20,
            1,
            1,
            Block::S,
            1312.0,
            72.8,
            0.0,
            14.01,
            20.28,
        ),
        Element::new(
            2,
            "He",
            "Helium",
            4.0026,
            0.00,
            1,
            18,
            Block::S,
            2372.3,
            0.0,
            0.0,
            0.95,
            4.22,
        ),
        // Period 2
        Element::new(
            3,
            "Li",
            "Lithium",
            6.941,
            0.98,
            2,
            1,
            Block::S,
            520.2,
            59.6,
            0.534,
            453.65,
            1615.0,
        ),
        Element::new(
            4,
            "Be",
            "Beryllium",
            9.0122,
            1.57,
            2,
            2,
            Block::S,
            899.5,
            0.0,
            1.848,
            1560.0,
            2742.0,
        ),
        Element::new(
            5,
            "B",
            "Boron",
            10.811,
            2.04,
            2,
            13,
            Block::P,
            800.6,
            26.7,
            2.370,
            2349.0,
            4200.0,
        ),
        Element::new(
            6,
            "C",
            "Carbon",
            12.011,
            2.55,
            2,
            14,
            Block::P,
            1086.5,
            153.9,
            2.267,
            3800.0,
            5100.0,
        ),
        Element::new(
            7,
            "N",
            "Nitrogen",
            14.007,
            3.04,
            2,
            15,
            Block::P,
            1402.3,
            7.0,
            0.0,
            63.15,
            77.36,
        ),
        Element::new(
            8,
            "O",
            "Oxygen",
            15.999,
            3.44,
            2,
            16,
            Block::P,
            1313.9,
            141.0,
            0.0,
            54.36,
            90.20,
        ),
        Element::new(
            9,
            "F",
            "Fluorine",
            18.998,
            3.98,
            2,
            17,
            Block::P,
            1681.0,
            328.0,
            0.0,
            53.53,
            85.03,
        ),
        Element::new(
            10,
            "Ne",
            "Neon",
            20.180,
            0.00,
            2,
            18,
            Block::P,
            2080.7,
            0.0,
            0.0,
            24.56,
            27.07,
        ),
        // Period 3
        Element::new(
            11,
            "Na",
            "Sodium",
            22.990,
            0.93,
            3,
            1,
            Block::S,
            495.8,
            52.8,
            0.971,
            370.87,
            1156.0,
        ),
        Element::new(
            12,
            "Mg",
            "Magnesium",
            24.305,
            1.31,
            3,
            2,
            Block::S,
            737.7,
            0.0,
            1.738,
            923.0,
            1363.0,
        ),
        Element::new(
            13,
            "Al",
            "Aluminium",
            26.982,
            1.61,
            3,
            13,
            Block::P,
            577.5,
            42.5,
            2.700,
            933.47,
            2792.0,
        ),
        Element::new(
            14,
            "Si",
            "Silicon",
            28.086,
            1.90,
            3,
            14,
            Block::P,
            786.5,
            133.6,
            2.329,
            1687.0,
            3538.0,
        ),
        Element::new(
            15,
            "P",
            "Phosphorus",
            30.974,
            2.19,
            3,
            15,
            Block::P,
            1011.8,
            72.0,
            1.823,
            317.30,
            550.0,
        ),
        Element::new(
            16,
            "S",
            "Sulfur",
            32.06,
            2.58,
            3,
            16,
            Block::P,
            1000.0,
            200.4,
            2.067,
            388.36,
            717.87,
        ),
        Element::new(
            17,
            "Cl",
            "Chlorine",
            35.45,
            3.16,
            3,
            17,
            Block::P,
            1251.2,
            348.6,
            0.0,
            171.65,
            239.11,
        ),
        Element::new(
            18,
            "Ar",
            "Argon",
            39.948,
            0.00,
            3,
            18,
            Block::P,
            1520.6,
            0.0,
            0.0,
            83.80,
            87.30,
        ),
        // Period 4
        Element::new(
            19,
            "K",
            "Potassium",
            39.098,
            0.82,
            4,
            1,
            Block::S,
            418.8,
            48.4,
            0.862,
            336.53,
            1032.0,
        ),
        Element::new(
            20,
            "Ca",
            "Calcium",
            40.078,
            1.00,
            4,
            2,
            Block::S,
            589.8,
            2.4,
            1.550,
            1115.0,
            1757.0,
        ),
        Element::new(
            21,
            "Sc",
            "Scandium",
            44.956,
            1.36,
            4,
            3,
            Block::D,
            633.1,
            18.1,
            2.985,
            1814.0,
            3109.0,
        ),
        Element::new(
            22,
            "Ti",
            "Titanium",
            47.867,
            1.54,
            4,
            4,
            Block::D,
            658.8,
            7.6,
            4.507,
            1941.0,
            3560.0,
        ),
        Element::new(
            23,
            "V",
            "Vanadium",
            50.942,
            1.63,
            4,
            5,
            Block::D,
            650.9,
            50.6,
            6.110,
            2183.0,
            3680.0,
        ),
        Element::new(
            24,
            "Cr",
            "Chromium",
            51.996,
            1.66,
            4,
            6,
            Block::D,
            652.9,
            64.3,
            7.150,
            2180.0,
            2944.0,
        ),
        Element::new(
            25,
            "Mn",
            "Manganese",
            54.938,
            1.55,
            4,
            7,
            Block::D,
            717.3,
            0.0,
            7.440,
            1519.0,
            2334.0,
        ),
        Element::new(
            26,
            "Fe",
            "Iron",
            55.845,
            1.83,
            4,
            8,
            Block::D,
            762.5,
            15.7,
            7.874,
            1811.0,
            3134.0,
        ),
        Element::new(
            27,
            "Co",
            "Cobalt",
            58.933,
            1.88,
            4,
            9,
            Block::D,
            760.4,
            63.7,
            8.900,
            1768.0,
            3200.0,
        ),
        Element::new(
            28,
            "Ni",
            "Nickel",
            58.693,
            1.91,
            4,
            10,
            Block::D,
            737.1,
            112.0,
            8.908,
            1728.0,
            3186.0,
        ),
        Element::new(
            29,
            "Cu",
            "Copper",
            63.546,
            1.90,
            4,
            11,
            Block::D,
            745.5,
            118.4,
            8.960,
            1357.77,
            2835.0,
        ),
        Element::new(
            30,
            "Zn",
            "Zinc",
            65.38,
            1.65,
            4,
            12,
            Block::D,
            906.4,
            0.0,
            7.133,
            692.68,
            1180.0,
        ),
        Element::new(
            31,
            "Ga",
            "Gallium",
            69.723,
            1.81,
            4,
            13,
            Block::P,
            578.8,
            28.9,
            5.904,
            302.91,
            2477.0,
        ),
        Element::new(
            32,
            "Ge",
            "Germanium",
            72.630,
            2.01,
            4,
            14,
            Block::P,
            762.2,
            119.0,
            5.323,
            1211.40,
            3106.0,
        ),
        Element::new(
            33,
            "As",
            "Arsenic",
            74.922,
            2.18,
            4,
            15,
            Block::P,
            947.0,
            78.0,
            5.727,
            1090.0,
            887.0,
        ),
        Element::new(
            34,
            "Se",
            "Selenium",
            78.971,
            2.55,
            4,
            16,
            Block::P,
            941.0,
            195.0,
            4.809,
            494.0,
            958.0,
        ),
        Element::new(
            35,
            "Br",
            "Bromine",
            79.904,
            2.96,
            4,
            17,
            Block::P,
            1139.9,
            324.6,
            3.103,
            265.85,
            332.0,
        ),
        Element::new(
            36,
            "Kr",
            "Krypton",
            83.798,
            3.00,
            4,
            18,
            Block::P,
            1350.8,
            0.0,
            0.0,
            115.79,
            119.93,
        ),
        // Period 5
        Element::new(
            37,
            "Rb",
            "Rubidium",
            85.468,
            0.82,
            5,
            1,
            Block::S,
            403.0,
            46.9,
            1.532,
            312.46,
            961.0,
        ),
        Element::new(
            38,
            "Sr",
            "Strontium",
            87.62,
            0.95,
            5,
            2,
            Block::S,
            549.5,
            5.0,
            2.640,
            1050.0,
            1655.0,
        ),
        Element::new(
            39,
            "Y",
            "Yttrium",
            88.906,
            1.22,
            5,
            3,
            Block::D,
            600.0,
            29.6,
            4.472,
            1799.0,
            3609.0,
        ),
        Element::new(
            40,
            "Zr",
            "Zirconium",
            91.224,
            1.33,
            5,
            4,
            Block::D,
            640.1,
            41.1,
            6.511,
            2128.0,
            4682.0,
        ),
        Element::new(
            41,
            "Nb",
            "Niobium",
            92.906,
            1.60,
            5,
            5,
            Block::D,
            652.1,
            86.1,
            8.570,
            2750.0,
            5017.0,
        ),
        Element::new(
            42,
            "Mo",
            "Molybdenum",
            95.96,
            2.16,
            5,
            6,
            Block::D,
            684.3,
            71.9,
            10.220,
            2896.0,
            4912.0,
        ),
        Element::new(
            43,
            "Tc",
            "Technetium",
            98.0,
            1.90,
            5,
            7,
            Block::D,
            702.0,
            53.0,
            11.000,
            2430.0,
            4538.0,
        ),
        Element::new(
            44,
            "Ru",
            "Ruthenium",
            101.07,
            2.20,
            5,
            8,
            Block::D,
            710.2,
            101.3,
            12.370,
            2607.0,
            4423.0,
        ),
        Element::new(
            45,
            "Rh",
            "Rhodium",
            102.906,
            2.28,
            5,
            9,
            Block::D,
            719.7,
            109.7,
            12.450,
            2237.0,
            3968.0,
        ),
        Element::new(
            46,
            "Pd",
            "Palladium",
            106.42,
            2.20,
            5,
            10,
            Block::D,
            804.4,
            53.7,
            12.023,
            1828.05,
            3236.0,
        ),
        Element::new(
            47,
            "Ag",
            "Silver",
            107.868,
            1.93,
            5,
            11,
            Block::D,
            731.0,
            125.6,
            10.490,
            1234.93,
            2435.0,
        ),
        Element::new(
            48,
            "Cd",
            "Cadmium",
            112.411,
            1.69,
            5,
            12,
            Block::D,
            867.8,
            0.0,
            8.650,
            594.22,
            1040.0,
        ),
        Element::new(
            49,
            "In",
            "Indium",
            114.818,
            1.78,
            5,
            13,
            Block::P,
            558.3,
            28.9,
            7.310,
            429.75,
            2345.0,
        ),
        Element::new(
            50,
            "Sn",
            "Tin",
            118.710,
            1.96,
            5,
            14,
            Block::P,
            708.6,
            107.3,
            7.287,
            505.08,
            2875.0,
        ),
        Element::new(
            51,
            "Sb",
            "Antimony",
            121.760,
            2.05,
            5,
            15,
            Block::P,
            834.0,
            103.2,
            6.685,
            903.78,
            1860.0,
        ),
        Element::new(
            52,
            "Te",
            "Tellurium",
            127.60,
            2.10,
            5,
            16,
            Block::P,
            869.3,
            190.2,
            6.232,
            722.66,
            1261.0,
        ),
        Element::new(
            53,
            "I",
            "Iodine",
            126.904,
            2.66,
            5,
            17,
            Block::P,
            1008.4,
            295.2,
            4.933,
            386.85,
            457.4,
        ),
        Element::new(
            54,
            "Xe",
            "Xenon",
            131.293,
            2.60,
            5,
            18,
            Block::P,
            1170.4,
            0.0,
            0.0,
            161.40,
            165.03,
        ),
        // Period 6
        Element::new(
            55,
            "Cs",
            "Caesium",
            132.905,
            0.79,
            6,
            1,
            Block::S,
            375.7,
            45.5,
            1.873,
            301.59,
            944.0,
        ),
        Element::new(
            56,
            "Ba",
            "Barium",
            137.327,
            0.89,
            6,
            2,
            Block::S,
            502.9,
            13.9,
            3.510,
            1000.0,
            2143.0,
        ),
        // Lanthanides (period 6, group 0 = f-block)
        Element::new(
            57,
            "La",
            "Lanthanum",
            138.905,
            1.10,
            6,
            0,
            Block::F,
            538.1,
            48.3,
            6.162,
            1193.0,
            3737.0,
        ),
        Element::new(
            58,
            "Ce",
            "Cerium",
            140.116,
            1.12,
            6,
            0,
            Block::F,
            534.4,
            50.0,
            6.770,
            1068.0,
            3716.0,
        ),
        Element::new(
            59,
            "Pr",
            "Praseodymium",
            140.908,
            1.13,
            6,
            0,
            Block::F,
            527.0,
            50.0,
            6.773,
            1208.0,
            3793.0,
        ),
        Element::new(
            60,
            "Nd",
            "Neodymium",
            144.242,
            1.14,
            6,
            0,
            Block::F,
            533.1,
            50.0,
            7.007,
            1297.0,
            3347.0,
        ),
        Element::new(
            61,
            "Pm",
            "Promethium",
            145.0,
            1.13,
            6,
            0,
            Block::F,
            540.0,
            50.0,
            7.260,
            1315.0,
            3273.0,
        ),
        Element::new(
            62,
            "Sm",
            "Samarium",
            150.36,
            1.17,
            6,
            0,
            Block::F,
            544.5,
            50.0,
            7.520,
            1345.0,
            2067.0,
        ),
        Element::new(
            63,
            "Eu",
            "Europium",
            151.964,
            1.20,
            6,
            0,
            Block::F,
            547.1,
            50.0,
            5.244,
            1099.0,
            1802.0,
        ),
        Element::new(
            64,
            "Gd",
            "Gadolinium",
            157.25,
            1.20,
            6,
            0,
            Block::F,
            593.4,
            50.0,
            7.895,
            1585.0,
            3546.0,
        ),
        Element::new(
            65,
            "Tb",
            "Terbium",
            158.925,
            1.10,
            6,
            0,
            Block::F,
            565.8,
            50.0,
            8.229,
            1629.0,
            3503.0,
        ),
        Element::new(
            66,
            "Dy",
            "Dysprosium",
            162.500,
            1.22,
            6,
            0,
            Block::F,
            573.0,
            50.0,
            8.550,
            1680.0,
            2840.0,
        ),
        Element::new(
            67,
            "Ho",
            "Holmium",
            164.930,
            1.23,
            6,
            0,
            Block::F,
            581.0,
            50.0,
            8.795,
            1734.0,
            2993.0,
        ),
        Element::new(
            68,
            "Er",
            "Erbium",
            167.259,
            1.24,
            6,
            0,
            Block::F,
            589.3,
            50.0,
            9.066,
            1802.0,
            3141.0,
        ),
        Element::new(
            69,
            "Tm",
            "Thulium",
            168.934,
            1.25,
            6,
            0,
            Block::F,
            596.7,
            50.0,
            9.321,
            1818.0,
            2223.0,
        ),
        Element::new(
            70,
            "Yb",
            "Ytterbium",
            173.045,
            1.10,
            6,
            0,
            Block::F,
            603.4,
            50.0,
            6.965,
            1092.0,
            1469.0,
        ),
        Element::new(
            71,
            "Lu",
            "Lutetium",
            174.967,
            1.27,
            6,
            3,
            Block::D,
            523.5,
            50.0,
            9.841,
            1925.0,
            3675.0,
        ),
        // d-block period 6
        Element::new(
            72,
            "Hf",
            "Hafnium",
            178.49,
            1.30,
            6,
            4,
            Block::D,
            658.5,
            0.0,
            13.310,
            2506.0,
            4876.0,
        ),
        Element::new(
            73,
            "Ta",
            "Tantalum",
            180.948,
            1.50,
            6,
            5,
            Block::D,
            728.4,
            31.0,
            16.650,
            3290.0,
            5731.0,
        ),
        Element::new(
            74,
            "W",
            "Tungsten",
            183.84,
            2.36,
            6,
            6,
            Block::D,
            758.8,
            78.6,
            19.250,
            3695.0,
            5828.0,
        ),
        Element::new(
            75,
            "Re",
            "Rhenium",
            186.207,
            1.90,
            6,
            7,
            Block::D,
            755.8,
            5.8,
            21.020,
            3459.0,
            5869.0,
        ),
        Element::new(
            76,
            "Os",
            "Osmium",
            190.23,
            2.20,
            6,
            8,
            Block::D,
            840.0,
            106.1,
            22.590,
            3306.0,
            5285.0,
        ),
        Element::new(
            77,
            "Ir",
            "Iridium",
            192.217,
            2.20,
            6,
            9,
            Block::D,
            880.0,
            151.0,
            22.560,
            2719.0,
            4701.0,
        ),
        Element::new(
            78,
            "Pt",
            "Platinum",
            195.084,
            2.28,
            6,
            10,
            Block::D,
            870.4,
            205.3,
            21.450,
            2041.4,
            4098.0,
        ),
        Element::new(
            79,
            "Au",
            "Gold",
            196.967,
            2.54,
            6,
            11,
            Block::D,
            890.1,
            222.8,
            19.300,
            1337.33,
            3129.0,
        ),
        Element::new(
            80,
            "Hg",
            "Mercury",
            200.592,
            2.00,
            6,
            12,
            Block::D,
            1007.1,
            0.0,
            13.534,
            234.43,
            629.88,
        ),
        Element::new(
            81,
            "Tl",
            "Thallium",
            204.38,
            1.62,
            6,
            13,
            Block::P,
            589.4,
            19.2,
            11.850,
            577.0,
            1746.0,
        ),
        Element::new(
            82,
            "Pb",
            "Lead",
            207.2,
            2.33,
            6,
            14,
            Block::P,
            715.6,
            35.1,
            11.340,
            600.61,
            2022.0,
        ),
        Element::new(
            83,
            "Bi",
            "Bismuth",
            208.980,
            2.02,
            6,
            15,
            Block::P,
            703.2,
            91.2,
            9.780,
            544.55,
            1837.0,
        ),
        Element::new(
            84,
            "Po",
            "Polonium",
            209.0,
            2.00,
            6,
            16,
            Block::P,
            812.1,
            136.0,
            9.196,
            527.0,
            1235.0,
        ),
        Element::new(
            85,
            "At",
            "Astatine",
            210.0,
            2.20,
            6,
            17,
            Block::P,
            899.0,
            270.0,
            0.0,
            575.0,
            610.0,
        ),
        Element::new(
            86,
            "Rn",
            "Radon",
            222.0,
            0.00,
            6,
            18,
            Block::P,
            1037.0,
            0.0,
            0.0,
            202.0,
            211.3,
        ),
        // Period 7
        Element::new(
            87,
            "Fr",
            "Francium",
            223.0,
            0.70,
            7,
            1,
            Block::S,
            380.0,
            46.9,
            1.870,
            300.0,
            950.0,
        ),
        Element::new(
            88,
            "Ra",
            "Radium",
            226.0,
            0.90,
            7,
            2,
            Block::S,
            509.3,
            9.6,
            5.500,
            973.0,
            2010.0,
        ),
        // Actinides (period 7, group 0)
        Element::new(
            89,
            "Ac",
            "Actinium",
            227.0,
            1.10,
            7,
            0,
            Block::F,
            499.0,
            33.8,
            10.070,
            1323.0,
            3471.0,
        ),
        Element::new(
            90,
            "Th",
            "Thorium",
            232.038,
            1.30,
            7,
            0,
            Block::F,
            587.0,
            112.0,
            11.720,
            2023.0,
            5061.0,
        ),
        Element::new(
            91,
            "Pa",
            "Protactinium",
            231.036,
            1.50,
            7,
            0,
            Block::F,
            568.0,
            53.0,
            15.370,
            1845.0,
            4300.0,
        ),
        Element::new(
            92,
            "U",
            "Uranium",
            238.029,
            1.38,
            7,
            0,
            Block::F,
            597.6,
            51.0,
            18.950,
            1408.0,
            4404.0,
        ),
        Element::new(
            93,
            "Np",
            "Neptunium",
            237.0,
            1.36,
            7,
            0,
            Block::F,
            604.5,
            45.9,
            20.450,
            912.0,
            4175.0,
        ),
        Element::new(
            94,
            "Pu",
            "Plutonium",
            244.0,
            1.28,
            7,
            0,
            Block::F,
            584.7,
            -48.0,
            19.816,
            912.5,
            3505.0,
        ),
        Element::new(
            95,
            "Am",
            "Americium",
            243.0,
            1.30,
            7,
            0,
            Block::F,
            578.0,
            9.0,
            13.670,
            1449.0,
            2880.0,
        ),
        Element::new(
            96,
            "Cm",
            "Curium",
            247.0,
            1.30,
            7,
            0,
            Block::F,
            581.0,
            27.7,
            13.510,
            1613.0,
            3383.0,
        ),
        Element::new(
            97,
            "Bk",
            "Berkelium",
            247.0,
            1.30,
            7,
            0,
            Block::F,
            601.0,
            27.7,
            14.780,
            1259.0,
            2900.0,
        ),
        Element::new(
            98,
            "Cf",
            "Californium",
            251.0,
            1.30,
            7,
            0,
            Block::F,
            608.0,
            27.7,
            15.100,
            1173.0,
            1743.0,
        ),
        Element::new(
            99,
            "Es",
            "Einsteinium",
            252.0,
            1.30,
            7,
            0,
            Block::F,
            619.0,
            27.7,
            8.840,
            1133.0,
            1269.0,
        ),
        Element::new(
            100,
            "Fm",
            "Fermium",
            257.0,
            1.30,
            7,
            0,
            Block::F,
            627.0,
            27.7,
            0.0,
            1800.0,
            0.0,
        ),
        Element::new(
            101,
            "Md",
            "Mendelevium",
            258.0,
            1.30,
            7,
            0,
            Block::F,
            635.0,
            27.7,
            0.0,
            1100.0,
            0.0,
        ),
        Element::new(
            102,
            "No",
            "Nobelium",
            259.0,
            1.30,
            7,
            0,
            Block::F,
            642.0,
            27.7,
            0.0,
            1100.0,
            0.0,
        ),
        Element::new(
            103,
            "Lr",
            "Lawrencium",
            266.0,
            1.30,
            7,
            3,
            Block::D,
            470.0,
            27.7,
            0.0,
            1900.0,
            0.0,
        ),
        // d-block period 7
        Element::new(
            104,
            "Rf",
            "Rutherfordium",
            267.0,
            0.00,
            7,
            4,
            Block::D,
            580.0,
            0.0,
            23.20,
            2400.0,
            5800.0,
        ),
        Element::new(
            105,
            "Db",
            "Dubnium",
            268.0,
            0.00,
            7,
            5,
            Block::D,
            665.0,
            0.0,
            29.30,
            0.0,
            0.0,
        ),
        Element::new(
            106,
            "Sg",
            "Seaborgium",
            271.0,
            0.00,
            7,
            6,
            Block::D,
            757.0,
            0.0,
            35.00,
            0.0,
            0.0,
        ),
        Element::new(
            107,
            "Bh",
            "Bohrium",
            272.0,
            0.00,
            7,
            7,
            Block::D,
            742.0,
            0.0,
            37.10,
            0.0,
            0.0,
        ),
        Element::new(
            108,
            "Hs",
            "Hassium",
            277.0,
            0.00,
            7,
            8,
            Block::D,
            730.0,
            0.0,
            40.70,
            0.0,
            0.0,
        ),
        Element::new(
            109,
            "Mt",
            "Meitnerium",
            278.0,
            0.00,
            7,
            9,
            Block::D,
            800.0,
            0.0,
            37.40,
            0.0,
            0.0,
        ),
        Element::new(
            110,
            "Ds",
            "Darmstadtium",
            281.0,
            0.00,
            7,
            10,
            Block::D,
            955.0,
            0.0,
            34.80,
            0.0,
            0.0,
        ),
        Element::new(
            111,
            "Rg",
            "Roentgenium",
            282.0,
            0.00,
            7,
            11,
            Block::D,
            1022.7,
            0.0,
            28.70,
            0.0,
            0.0,
        ),
        Element::new(
            112,
            "Cn",
            "Copernicium",
            285.0,
            0.00,
            7,
            12,
            Block::D,
            1155.0,
            0.0,
            23.70,
            283.0,
            340.0,
        ),
        Element::new(
            113,
            "Nh",
            "Nihonium",
            286.0,
            0.00,
            7,
            13,
            Block::P,
            704.0,
            0.0,
            16.00,
            700.0,
            1430.0,
        ),
        Element::new(
            114,
            "Fl",
            "Flerovium",
            289.0,
            0.00,
            7,
            14,
            Block::P,
            832.2,
            0.0,
            14.00,
            340.0,
            420.0,
        ),
        Element::new(
            115,
            "Mc",
            "Moscovium",
            290.0,
            0.00,
            7,
            15,
            Block::P,
            538.3,
            0.0,
            13.50,
            670.0,
            1400.0,
        ),
        Element::new(
            116,
            "Lv",
            "Livermorium",
            293.0,
            0.00,
            7,
            16,
            Block::P,
            723.6,
            0.0,
            12.90,
            637.0,
            1035.0,
        ),
        Element::new(
            117,
            "Ts",
            "Tennessine",
            294.0,
            0.00,
            7,
            17,
            Block::P,
            736.0,
            0.0,
            7.17,
            723.0,
            883.0,
        ),
        Element::new(
            118,
            "Og",
            "Oganesson",
            294.0,
            0.00,
            7,
            18,
            Block::P,
            860.1,
            0.0,
            5.00,
            258.0,
            263.0,
        ),
    ]
}

/// Build a lookup map from symbol → Element for fast access.
pub fn element_map(elements: &[Element]) -> HashMap<&str, &Element> {
    elements.iter().map(|e| (e.symbol, e)).collect()
}

// ---------------------------------------------------------------------------
// 2. Stoichiometry
// ---------------------------------------------------------------------------

/// Parse a chemical formula into a map of element symbol → atom count.
///
/// Handles:
/// - Simple formulas: "H2O", "NaCl", "CO2"
/// - Two-letter symbols: "Ca", "Fe", "Mg"
/// - Parenthesized groups with multipliers: "Ca(OH)2", "Fe2(SO4)3"
///
/// Does NOT handle nested parentheses.
pub fn parse_formula(formula: &str) -> HashMap<String, u32> {
    let chars: Vec<char> = formula.chars().collect();
    let mut counts: HashMap<String, u32> = HashMap::new();
    parse_formula_segment(&chars, 0, chars.len(), 1, &mut counts);
    counts
}

fn parse_formula_segment(
    chars: &[char],
    start: usize,
    end: usize,
    multiplier: u32,
    counts: &mut HashMap<String, u32>,
) {
    let mut i = start;
    while i < end {
        if chars[i] == '(' {
            // find matching ')'
            let mut depth = 1usize;
            let mut j = i + 1;
            while j < end && depth > 0 {
                if chars[j] == '(' {
                    depth += 1;
                }
                if chars[j] == ')' {
                    depth -= 1;
                }
                j += 1;
            }
            // j now points one past ')'
            let close = j - 1;
            // read number after ')'
            let (group_mult, advance) = read_number(chars, j, end);
            let group_mult = if group_mult == 0 { 1 } else { group_mult };
            parse_formula_segment(chars, i + 1, close, multiplier * group_mult, counts);
            i = j + advance;
        } else if chars[i].is_ascii_uppercase() {
            // element symbol
            let mut sym = String::new();
            sym.push(chars[i]);
            i += 1;
            while i < end && chars[i].is_ascii_lowercase() {
                sym.push(chars[i]);
                i += 1;
            }
            let (count, advance) = read_number(chars, i, end);
            let count = if count == 0 { 1 } else { count };
            *counts.entry(sym).or_insert(0) += count * multiplier;
            i += advance;
        } else {
            i += 1; // skip unexpected chars
        }
    }
}

/// Read an integer starting at position `start` in `chars`.
/// Returns (value, chars_consumed). If no digits found, returns (0, 0).
fn read_number(chars: &[char], start: usize, end: usize) -> (u32, usize) {
    let mut n = 0u32;
    let mut consumed = 0usize;
    let mut idx = start;
    while idx < end && chars[idx].is_ascii_digit() {
        n = n * 10 + (chars[idx] as u32 - '0' as u32);
        consumed += 1;
        idx += 1;
    }
    (n, consumed)
}

/// Calculate molar mass of a formula in g/mol.
pub fn molar_mass(formula: &str, elements: &[Element]) -> Result<f64, String> {
    let map = element_map(elements);
    let parsed = parse_formula(formula);
    let mut mass = 0.0f64;
    for (sym, count) in &parsed {
        match map.get(sym.as_str()) {
            Some(el) => mass += el.atomic_mass * (*count as f64),
            None => return Err(format!("Unknown element symbol: {sym}")),
        }
    }
    Ok(mass)
}

// ---------------------------------------------------------------------------
// Equation balancing via Gaussian elimination (null-space of atom matrix)
// ---------------------------------------------------------------------------

/// Balance a chemical equation.
///
/// Given reactant and product formula strings, returns integer stoichiometric
/// coefficients [r1, r2, ..., p1, p2, ...] (all positive). Uses Gaussian
/// elimination over rationals on the atom matrix.
///
/// # Errors
/// Returns `Err` if the system cannot be balanced (no valid integer solution found).
pub fn balance_equation(reactants: &[&str], products: &[&str]) -> Result<Vec<i32>, String> {
    // Collect all unique elements
    let mut element_set: Vec<String> = Vec::new();
    let all_formulas: Vec<&str> = reactants.iter().chain(products.iter()).copied().collect();
    let mut parsed: Vec<HashMap<String, u32>> = Vec::with_capacity(all_formulas.len());
    for f in &all_formulas {
        let p = parse_formula(f);
        for sym in p.keys() {
            if !element_set.contains(sym) {
                element_set.push(sym.clone());
            }
        }
        parsed.push(p);
    }
    element_set.sort();

    let n_elem = element_set.len();
    let n_species = all_formulas.len();

    if n_species < 2 {
        return Err("Need at least 2 species to balance".to_string());
    }

    // Build atom matrix: rows = elements, cols = species
    // Reactants have positive sign, products have negative sign (to enforce Ax=0)
    let mut mat: Vec<Vec<f64>> = vec![vec![0.0; n_species]; n_elem];
    for (col, (formula_map, is_product)) in parsed
        .iter()
        .zip(
            (0..reactants.len())
                .map(|_| false)
                .chain((0..products.len()).map(|_| true)),
        )
        .enumerate()
    {
        for (row, elem) in element_set.iter().enumerate() {
            let count = *formula_map.get(elem).unwrap_or(&0) as f64;
            mat[row][col] = if is_product { -count } else { count };
        }
    }

    // Gaussian elimination to find null space
    // We look for a non-trivial solution to mat * coeffs = 0
    // Strategy: augment with identity, reduce, back-substitute
    // Since this is null-space, we fix the last variable = 1 and solve

    // Work with rational arithmetic (numerator/denominator pairs) for exactness
    // Represent as (num: i64, den: i64)
    type Rat = (i64, i64);

    fn rat(n: i64) -> Rat {
        (n, 1)
    }
    fn rat_from_f64(f: f64) -> Rat {
        // Convert to fraction with denominator up to 10000
        let sign = if f < 0.0 { -1i64 } else { 1i64 };
        let f = f.abs();
        let n = (f * 10000.0).round() as i64;
        let g = gcd(n, 10000);
        (sign * n / g, 10000 / g)
    }
    fn gcd(a: i64, b: i64) -> i64 {
        let (mut a, mut b) = (a.abs(), b.abs());
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a.max(1)
    }
    fn rat_add(a: Rat, b: Rat) -> Rat {
        let n = a.0 * b.1 + b.0 * a.1;
        let d = a.1 * b.1;
        let g = gcd(n.abs(), d.abs());
        (n / g, d / g)
    }
    fn rat_sub(a: Rat, b: Rat) -> Rat {
        rat_add(a, (-b.0, b.1))
    }
    fn rat_mul(a: Rat, b: Rat) -> Rat {
        let n = a.0 * b.0;
        let d = a.1 * b.1;
        let g = gcd(n.abs(), d.abs());
        (n / g, d / g)
    }
    fn rat_div(a: Rat, b: Rat) -> Rat {
        rat_mul(a, (b.1, b.0))
    }
    fn rat_is_zero(a: Rat) -> bool {
        a.0 == 0
    }

    // Build rational matrix: n_elem rows, n_species cols
    let mut rmat: Vec<Vec<Rat>> = mat
        .iter()
        .map(|row| row.iter().map(|&v| rat_from_f64(v)).collect())
        .collect();

    // Row reduce
    let rows = n_elem;
    let cols = n_species;
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut row = 0usize;
    for col in 0..cols {
        // find pivot
        let pivot_row = (row..rows).find(|&r| !rat_is_zero(rmat[r][col]));
        if let Some(pr) = pivot_row {
            rmat.swap(row, pr);
            let pv = rmat[row][col];
            for c in 0..cols {
                rmat[row][c] = rat_div(rmat[row][c], pv);
            }
            for r in 0..rows {
                if r != row && !rat_is_zero(rmat[r][col]) {
                    let factor = rmat[r][col];
                    for c in 0..cols {
                        let sub = rat_mul(factor, rmat[row][c]);
                        rmat[r][c] = rat_sub(rmat[r][c], sub);
                    }
                }
            }
            pivot_cols.push(col);
            row += 1;
        }
    }

    // Free variables: columns not in pivot_cols
    let free_vars: Vec<usize> = (0..cols).filter(|c| !pivot_cols.contains(c)).collect();
    if free_vars.is_empty() {
        return Err("System is over-determined; no null space found".to_string());
    }

    // Set free variable(s) to 1, solve for pivot vars
    let free_col = free_vars[0];
    let mut solution: Vec<Rat> = vec![rat(0); cols];
    solution[free_col] = rat(1);

    for (i, &pc) in pivot_cols.iter().enumerate() {
        if i < rows {
            let mut val = rat(0);
            for &fc in &free_vars {
                val = rat_sub(val, rat_mul(rmat[i][fc], solution[fc]));
            }
            solution[pc] = val;
        }
    }

    // Convert to integers by finding LCM of denominators
    let lcm_den = solution.iter().fold(1i64, |acc, r| {
        let g = gcd(acc, r.1.abs());
        acc / g * r.1.abs()
    });

    let int_solution: Vec<i64> = solution.iter().map(|r| r.0 * (lcm_den / r.1)).collect();

    // Check all non-zero and find if we need to flip signs
    // Reactants (first n_react) should be positive
    let n_react = reactants.len();
    let sign = if int_solution[0] < 0 { -1i64 } else { 1i64 };
    let int_solution: Vec<i64> = int_solution.iter().map(|&x| x * sign).collect();

    if int_solution.iter().any(|&x| x == 0) {
        return Err("Degenerate solution (zero coefficient found)".to_string());
    }

    // Products should be negative in our convention → flip to positive
    let result: Vec<i32> = int_solution
        .iter()
        .enumerate()
        .map(|(i, &x)| if i < n_react { x as i32 } else { (-x) as i32 })
        .collect();

    Ok(result)
}

/// Calculate theoretical yield in grams.
///
/// # Arguments
/// * `limiting_reagent_mol` — moles of limiting reagent available
/// * `limiting_coeff` — stoichiometric coefficient of limiting reagent
/// * `product_coeff` — stoichiometric coefficient of desired product
/// * `product_molar_mass` — molar mass of product (g/mol)
pub fn theoretical_yield(
    limiting_reagent_mol: f64,
    limiting_coeff: i32,
    product_coeff: i32,
    product_molar_mass: f64,
) -> f64 {
    let moles_product = limiting_reagent_mol * (product_coeff as f64) / (limiting_coeff as f64);
    moles_product * product_molar_mass
}

// ---------------------------------------------------------------------------
// 3. Thermochemistry
// ---------------------------------------------------------------------------

/// Standard thermochemical data at 298 K from NIST-JANAF tables.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermochemicalData {
    pub formula: &'static str,
    /// ΔH°f in kJ/mol
    pub delta_hf_kj_mol: f64,
    /// S° in J/(mol·K)
    pub delta_sf_j_mol_k: f64,
    /// ΔG°f in kJ/mol
    pub delta_gf_kj_mol: f64,
}

impl ThermochemicalData {
    const fn new(formula: &'static str, dhf: f64, sf: f64, dgf: f64) -> Self {
        Self {
            formula,
            delta_hf_kj_mol: dhf,
            delta_sf_j_mol_k: sf,
            delta_gf_kj_mol: dgf,
        }
    }
}

/// Returns standard thermochemical data for 60+ common compounds (NIST-JANAF).
pub fn thermochemical_database() -> Vec<ThermochemicalData> {
    vec![
        // Elements (reference state = 0)
        ThermochemicalData::new("H2(g)", 0.0, 130.68, 0.0),
        ThermochemicalData::new("O2(g)", 0.0, 205.15, 0.0),
        ThermochemicalData::new("N2(g)", 0.0, 191.61, 0.0),
        ThermochemicalData::new("C(graphite)", 0.0, 5.74, 0.0),
        ThermochemicalData::new("S(s)", 0.0, 31.80, 0.0),
        ThermochemicalData::new("Na(s)", 0.0, 51.30, 0.0),
        ThermochemicalData::new("Fe(s)", 0.0, 27.28, 0.0),
        ThermochemicalData::new("Ca(s)", 0.0, 41.59, 0.0),
        ThermochemicalData::new("Mg(s)", 0.0, 32.67, 0.0),
        ThermochemicalData::new("Al(s)", 0.0, 28.33, 0.0),
        ThermochemicalData::new("Cl2(g)", 0.0, 223.08, 0.0),
        ThermochemicalData::new("F2(g)", 0.0, 202.78, 0.0),
        // Water
        ThermochemicalData::new("H2O(l)", -285.830, 69.950, -237.140),
        ThermochemicalData::new("H2O(g)", -241.826, 188.835, -228.572),
        // Carbon compounds
        ThermochemicalData::new("CO2(g)", -393.509, 213.785, -394.359),
        ThermochemicalData::new("CO(g)", -110.525, 197.660, -137.168),
        ThermochemicalData::new("CH4(g)", -74.870, 186.250, -50.750),
        ThermochemicalData::new("C2H6(g)", -84.680, 229.490, -32.820),
        ThermochemicalData::new("C3H8(g)", -103.850, 269.910, -23.490),
        ThermochemicalData::new("C2H4(g)", 52.400, 219.330, 68.150),
        ThermochemicalData::new("C2H2(g)", 227.400, 200.940, 209.200),
        ThermochemicalData::new("C6H6(l)", 49.100, 173.260, 124.500),
        ThermochemicalData::new("C6H6(g)", 82.900, 269.200, 129.700),
        // Nitrogen compounds
        ThermochemicalData::new("NH3(g)", -46.110, 192.770, -16.450),
        ThermochemicalData::new("NO(g)", 90.250, 210.760, 86.550),
        ThermochemicalData::new("NO2(g)", 33.200, 240.040, 51.310),
        ThermochemicalData::new("N2O4(g)", 9.160, 304.290, 97.890),
        ThermochemicalData::new("HNO3(l)", -174.100, 155.600, -80.710),
        // Sulfur compounds
        ThermochemicalData::new("SO2(g)", -296.830, 248.200, -300.190),
        ThermochemicalData::new("SO3(g)", -395.720, 256.770, -371.060),
        ThermochemicalData::new("H2S(g)", -20.630, 205.810, -33.560),
        ThermochemicalData::new("H2SO4(l)", -813.989, 156.904, -690.003),
        // Halides
        ThermochemicalData::new("HCl(g)", -92.307, 186.908, -95.299),
        ThermochemicalData::new("HF(g)", -271.100, 173.780, -273.200),
        ThermochemicalData::new("NaCl(s)", -411.153, 72.113, -384.138),
        ThermochemicalData::new("KCl(s)", -436.747, 82.590, -408.531),
        // Oxides and hydroxides
        ThermochemicalData::new("CaO(s)", -635.090, 38.200, -604.030),
        ThermochemicalData::new("Ca(OH)2(s)", -986.090, 83.390, -898.490),
        ThermochemicalData::new("CaCO3(s)", -1207.600, 91.700, -1129.100),
        ThermochemicalData::new("MgO(s)", -601.600, 26.950, -569.300),
        ThermochemicalData::new("Al2O3(s)", -1675.700, 50.920, -1582.300),
        ThermochemicalData::new("Fe2O3(s)", -824.200, 87.400, -742.200),
        ThermochemicalData::new("Fe3O4(s)", -1118.400, 146.400, -1015.400),
        ThermochemicalData::new("FeS2(s)", -178.200, 52.930, -166.900),
        ThermochemicalData::new("SiO2(s)", -910.700, 41.840, -856.300),
        ThermochemicalData::new("TiO2(s)", -944.000, 50.290, -888.800),
        ThermochemicalData::new("NaOH(s)", -425.930, 64.455, -379.530),
        ThermochemicalData::new("KOH(s)", -424.580, 78.870, -378.840),
        ThermochemicalData::new("H3PO4(l)", -1271.660, 150.820, -1119.100),
        // Organics
        ThermochemicalData::new("C6H12O6(s)", -1273.300, 212.100, -910.400), // glucose
        ThermochemicalData::new("C12H22O11(s)", -2222.100, 360.200, -1543.100), // sucrose
        ThermochemicalData::new("C2H5OH(l)", -277.690, 160.700, -174.780),   // ethanol
        ThermochemicalData::new("CH3OH(l)", -238.660, 126.800, -166.270),    // methanol
        ThermochemicalData::new("C3H6O(l)", -248.100, 200.400, -155.400),    // acetone
        ThermochemicalData::new("CH3COOH(l)", -484.500, 159.800, -389.900),  // acetic acid
        ThermochemicalData::new("HCOOH(l)", -424.720, 128.950, -361.400),    // formic acid
        ThermochemicalData::new("C2H2O4(s)", -829.910, 115.600, -697.900),   // oxalic acid
        ThermochemicalData::new("C7H8(l)", 12.000, 220.000, 113.600),        // toluene
        // Biological approximations
        ThermochemicalData::new("ATP(aq)", -2981.000, 0.0, 0.0), // approximate
        ThermochemicalData::new("ADP(aq)", -2000.000, 0.0, 0.0), // approximate
    ]
}

/// Look up thermochemical data by formula string.
pub fn find_thermo<'a>(
    formula: &str,
    data: &'a [ThermochemicalData],
) -> Option<&'a ThermochemicalData> {
    data.iter().find(|d| d.formula == formula)
}

/// Hess's Law: ΔH°rxn = Σ ν_i * ΔH°f(products) - Σ ν_j * ΔH°f(reactants).
///
/// Pass each species as `(formula_str, stoich_coeff)` where:
/// - negative coefficient = reactant
/// - positive coefficient = product
///
/// Returns ΔH°rxn in kJ/mol.
pub fn hess_law(reaction_formulas: &[(&str, f64)], data: &[ThermochemicalData]) -> f64 {
    reaction_formulas.iter().fold(0.0, |acc, (formula, coeff)| {
        let dhf = find_thermo(formula, data)
            .map(|d| d.delta_hf_kj_mol)
            .unwrap_or(0.0);
        acc + coeff * dhf
    })
}

/// Gibbs Free Energy: ΔG = ΔH - T·ΔS (returns kJ/mol).
///
/// # Arguments
/// * `delta_h_kj` — enthalpy change in kJ/mol
/// * `temp_k` — temperature in K
/// * `delta_s_j_k` — entropy change in J/(mol·K)
pub fn gibbs_free_energy(delta_h_kj: f64, temp_k: f64, delta_s_j_k: f64) -> f64 {
    delta_h_kj - temp_k * (delta_s_j_k / 1000.0)
}

/// Temperature at which ΔG = 0: T = ΔH / ΔS (equilibrium crossing point).
///
/// Returns ∞ if ΔS ≈ 0.
pub fn equilibrium_temp(delta_h_kj: f64, delta_s_j_k: f64) -> f64 {
    if delta_s_j_k.abs() < 1e-12 {
        f64::INFINITY
    } else {
        (delta_h_kj * 1000.0) / delta_s_j_k
    }
}

/// Returns `true` if the reaction is spontaneous (ΔG < 0).
pub fn is_spontaneous(delta_g_kj: f64) -> bool {
    delta_g_kj < 0.0
}

// ---------------------------------------------------------------------------
// 4. Chemical Equilibrium
// ---------------------------------------------------------------------------

/// Gas constant in kJ/(mol·K).
pub const R_KJ: f64 = 8.314e-3;
/// Gas constant in J/(mol·K).
pub const R_J: f64 = 8.314;

/// Keq from standard Gibbs free energy: Keq = exp(−ΔG°/(RT)).
pub fn keq_from_gibbs(delta_g_kj: f64, temp_k: f64) -> f64 {
    (-delta_g_kj / (R_KJ * temp_k)).exp()
}

/// ΔG° from equilibrium constant: ΔG° = −RT ln(Keq) in kJ/mol.
pub fn gibbs_from_keq(keq: f64, temp_k: f64) -> f64 {
    -R_KJ * temp_k * keq.ln()
}

/// Result of an ICE (Initial, Change, Equilibrium) table calculation.
#[derive(Debug, Clone)]
pub struct IceTable {
    /// Initial concentrations [M]
    pub initial: Vec<f64>,
    /// Change in concentration (±x * |coefficient|)
    pub change: Vec<f64>,
    /// Equilibrium concentrations = initial + change
    pub equilibrium: Vec<f64>,
}

/// Solve an ICE table for a single equilibrium reaction.
///
/// Finds x (the extent of reaction) such that the equilibrium expression equals Keq.
/// The equilibrium expression is:
/// ```text
///   Keq = Π [product_i + ν_i * x]^ν_i  /  Π [reactant_j - |ν_j| * x]^|ν_j|
/// ```
///
/// Uses bisection over a physically valid range for x.
///
/// # Arguments
/// * `initial_concentrations` — initial [M] for each species (reactants first, then products)
/// * `stoich_coeffs` — negative for reactants, positive for products (same order as concentrations)
/// * `keq` — equilibrium constant
pub fn solve_ice(
    initial_concentrations: &[f64],
    stoich_coeffs: &[i32],
    keq: f64,
) -> Result<IceTable, String> {
    if initial_concentrations.len() != stoich_coeffs.len() {
        return Err("initial_concentrations and stoich_coeffs must have same length".to_string());
    }
    let n = initial_concentrations.len();

    // The equilibrium expression K(x) = products/reactants evaluated at extent x
    let keq_expr = |x: f64| -> f64 {
        let mut numerator = 1.0f64;
        let mut denominator = 1.0f64;
        for i in 0..n {
            let c = initial_concentrations[i] + (stoich_coeffs[i] as f64) * x;
            let nu = stoich_coeffs[i].abs() as i32;
            if stoich_coeffs[i] > 0 {
                // product
                numerator *= c.max(0.0).powi(nu);
            } else {
                // reactant
                denominator *= c.max(0.0).powi(nu);
            }
        }
        if denominator < 1e-300 {
            return f64::INFINITY;
        }
        numerator / denominator
    };

    // Determine valid range for x
    // x must keep all concentrations non-negative
    // Reactants: initial + coeff*x >= 0 → x <= -initial/coeff (coeff < 0)
    // Products: initial + coeff*x >= 0 → x >= -initial/coeff (coeff > 0, and initial could be 0)
    let x_min_raw = stoich_coeffs
        .iter()
        .zip(initial_concentrations.iter())
        .filter(|&(&c, _)| c < 0)
        .map(|(&c, &conc)| -conc / (c as f64)) // c < 0, so this is positive upper bound
        .fold(f64::INFINITY, f64::min);
    let x_max = if x_min_raw.is_infinite() || x_min_raw < 1e-12 {
        1e6
    } else {
        x_min_raw * 0.9999
    };

    // For reverse direction
    let x_min = stoich_coeffs
        .iter()
        .zip(initial_concentrations.iter())
        .filter(|&(&c, _)| c > 0)
        .map(|(&c, &conc)| -conc / (c as f64)) // c > 0, conc >= 0, so this is <= 0
        .fold(-f64::INFINITY, f64::max);
    let x_lo = x_min * 0.9999;
    let x_hi = x_max;

    // f(x) = K(x) - Keq; find root by bisection
    let f = |x: f64| keq_expr(x) - keq;

    let f_lo = f(x_lo);
    let f_hi = f(x_hi);

    // Try to find a bracket
    let (mut lo, mut hi) = if f_lo * f_hi <= 0.0 {
        (x_lo, x_hi)
    } else if f_lo.abs() < f_hi.abs() {
        // closer at lo side, try narrow range
        (x_lo, 0.0_f64.max(x_hi * 0.01))
    } else {
        (0.0, x_hi)
    };

    // Ensure bracket
    if f(lo) * f(hi) > 0.0 {
        // Try harder: scan for sign change
        let steps = 1000;
        let dx = (hi - lo) / steps as f64;
        let mut found = false;
        for s in 0..steps {
            let a = lo + dx * s as f64;
            let b = a + dx;
            if f(a) * f(b) <= 0.0 {
                lo = a;
                hi = b;
                found = true;
                break;
            }
        }
        if !found {
            return Err(
                "Could not bracket equilibrium solution; check Keq and initial conditions"
                    .to_string(),
            );
        }
    }

    // Bisection
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if (hi - lo).abs() < 1e-14 {
            break;
        }
        if f(lo) * f(mid) <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let x_sol = (lo + hi) / 2.0;

    let change: Vec<f64> = stoich_coeffs.iter().map(|&c| (c as f64) * x_sol).collect();
    let equilibrium: Vec<f64> = initial_concentrations
        .iter()
        .zip(change.iter())
        .map(|(&c0, &dc)| (c0 + dc).max(0.0))
        .collect();

    Ok(IceTable {
        initial: initial_concentrations.to_vec(),
        change,
        equilibrium,
    })
}

/// A perturbation to an equilibrium system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Perturbation {
    AddReactant,
    AddProduct,
    IncreaseTemp,
    DecreasePressure,
}

/// Predict the direction of equilibrium shift according to Le Chatelier's principle.
///
/// # Arguments
/// * `rxn_type` — `"exothermic"` or `"endothermic"` (only relevant for `IncreaseTemp`)
/// * `perturbation` — type of perturbation applied
///
/// Returns `"shifts right"`, `"shifts left"`, or `"no change"`.
pub fn le_chatelier(rxn_type: &str, perturbation: Perturbation) -> &'static str {
    match perturbation {
        Perturbation::AddReactant => "shifts right",
        Perturbation::AddProduct => "shifts left",
        Perturbation::IncreaseTemp => {
            if rxn_type.eq_ignore_ascii_case("exothermic") {
                "shifts left"
            } else {
                "shifts right"
            }
        }
        Perturbation::DecreasePressure => "shifts left",
    }
}

// ---------------------------------------------------------------------------
// 5. Reaction Kinetics
// ---------------------------------------------------------------------------

/// Rate of reaction: r = k * Π[A_i]^m_i
///
/// # Arguments
/// * `rate_constant` — k (units depend on reaction order)
/// * `concentrations` — [M] of each species
/// * `orders` — partial orders (can be fractional)
pub fn reaction_rate(rate_constant: f64, concentrations: &[f64], orders: &[f64]) -> f64 {
    let product = concentrations
        .iter()
        .zip(orders.iter())
        .fold(1.0f64, |acc, (&c, &m)| acc * c.max(0.0).powf(m));
    rate_constant * product
}

/// Half-life for a first-order reaction: t₁/₂ = ln(2)/k
pub fn half_life_first_order(k: f64) -> f64 {
    std::f64::consts::LN_2 / k
}

/// Half-life for a second-order reaction: t₁/₂ = 1/(k·c₀)
pub fn half_life_second_order(k: f64, c0: f64) -> f64 {
    1.0 / (k * c0)
}

/// Arrhenius rate constant: k(T) = A·exp(−Ea/(RT))
///
/// # Arguments
/// * `a` — pre-exponential factor (same units as k)
/// * `ea_j_mol` — activation energy in J/mol
/// * `temp_k` — temperature in K
pub fn arrhenius_k(a: f64, ea_j_mol: f64, temp_k: f64) -> f64 {
    a * (-ea_j_mol / (R_J * temp_k)).exp()
}

/// Activation energy from two rate constants at two temperatures.
///
/// Ea = R * ln(k2/k1) / (1/T1 - 1/T2)
pub fn activation_energy(k1: f64, k2: f64, t1: f64, t2: f64) -> f64 {
    R_J * (k2 / k1).ln() / (1.0 / t1 - 1.0 / t2)
}

/// Pre-exponential Arrhenius factor: A = k / exp(−Ea/(RT))
pub fn arrhenius_factor_a(k: f64, ea_j_mol: f64, temp_k: f64) -> f64 {
    k / (-ea_j_mol / (R_J * temp_k)).exp()
}

/// Concentration via zero-order integrated rate law: [A]t = [A]₀ − k·t
pub fn concentration_zero_order(c0: f64, k: f64, t: f64) -> f64 {
    (c0 - k * t).max(0.0)
}

/// Concentration via first-order integrated rate law: [A]t = [A]₀·exp(−k·t)
pub fn concentration_first_order(c0: f64, k: f64, t: f64) -> f64 {
    c0 * (-k * t).exp()
}

/// Concentration via second-order integrated rate law: 1/[A]t = 1/[A]₀ + k·t
pub fn concentration_second_order(c0: f64, k: f64, t: f64) -> f64 {
    1.0 / (1.0 / c0 + k * t)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-2;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // --- Element table ---

    #[test]
    fn test_all_118_elements_present() {
        let elems = all_elements();
        assert_eq!(elems.len(), 118, "Should have exactly 118 elements");
        assert_eq!(elems[0].symbol, "H");
        assert_eq!(elems[117].symbol, "Og");
    }

    #[test]
    fn test_element_data_hydrogen() {
        let elems = all_elements();
        let h = &elems[0];
        assert_eq!(h.atomic_number, 1);
        assert_eq!(h.name, "Hydrogen");
        assert!(approx_eq(h.atomic_mass, 1.008, 0.01));
        assert!(approx_eq(h.electronegativity, 2.20, 0.01));
        assert_eq!(h.period, 1);
        assert_eq!(h.group, 1);
        assert_eq!(h.block, Block::S);
    }

    #[test]
    fn test_element_carbon() {
        let elems = all_elements();
        let c = &elems[5]; // index 5 = C (atomic number 6)
        assert_eq!(c.atomic_number, 6);
        assert!(approx_eq(c.atomic_mass, 12.011, 0.01));
        assert_eq!(c.block, Block::P);
    }

    // --- Formula parsing ---

    #[test]
    fn test_parse_water() {
        let result = parse_formula("H2O");
        assert_eq!(result.get("H"), Some(&2));
        assert_eq!(result.get("O"), Some(&1));
    }

    #[test]
    fn test_parse_calcium_hydroxide() {
        let result = parse_formula("Ca(OH)2");
        assert_eq!(result.get("Ca"), Some(&1));
        assert_eq!(result.get("O"), Some(&2));
        assert_eq!(result.get("H"), Some(&2));
    }

    #[test]
    fn test_parse_sulfuric_acid() {
        let result = parse_formula("H2SO4");
        assert_eq!(result.get("H"), Some(&2));
        assert_eq!(result.get("S"), Some(&1));
        assert_eq!(result.get("O"), Some(&4));
    }

    #[test]
    fn test_parse_iron_sulfate() {
        // Fe2(SO4)3
        let result = parse_formula("Fe2(SO4)3");
        assert_eq!(result.get("Fe"), Some(&2));
        assert_eq!(result.get("S"), Some(&3));
        assert_eq!(result.get("O"), Some(&12));
    }

    #[test]
    fn test_parse_glucose() {
        let result = parse_formula("C6H12O6");
        assert_eq!(result.get("C"), Some(&6));
        assert_eq!(result.get("H"), Some(&12));
        assert_eq!(result.get("O"), Some(&6));
    }

    // --- Molar mass ---

    #[test]
    fn test_molar_mass_water() {
        let elems = all_elements();
        let mm = molar_mass("H2O", &elems).unwrap();
        assert!(approx_eq(mm, 18.015, 0.05), "H2O molar mass = {mm}");
    }

    #[test]
    fn test_molar_mass_co2() {
        let elems = all_elements();
        let mm = molar_mass("CO2", &elems).unwrap();
        assert!(approx_eq(mm, 44.01, 0.05), "CO2 molar mass = {mm}");
    }

    #[test]
    fn test_molar_mass_nacl() {
        let elems = all_elements();
        let mm = molar_mass("NaCl", &elems).unwrap();
        // Na=22.990, Cl=35.45 → 58.44
        assert!(approx_eq(mm, 58.44, 0.1), "NaCl molar mass = {mm}");
    }

    // --- Thermochemistry ---

    #[test]
    fn test_hess_law_methane_combustion() {
        // CH4(g) + 2O2(g) → CO2(g) + 2H2O(l)
        // ΔH = ΔHf(CO2) + 2*ΔHf(H2O(l)) - ΔHf(CH4) - 2*ΔHf(O2)
        //    = -393.509 + 2*(-285.830) - (-74.870) - 0
        //    ≈ -890.3 kJ/mol
        let db = thermochemical_database();
        let dh = hess_law(
            &[
                ("CO2(g)", 1.0),
                ("H2O(l)", 2.0),
                ("CH4(g)", -1.0),
                ("O2(g)", -2.0),
            ],
            &db,
        );
        assert!(
            approx_eq(dh, -890.3, 5.0),
            "CH4 combustion ΔH = {dh} kJ/mol"
        );
    }

    #[test]
    fn test_gibbs_free_energy_water_formation() {
        // H2O(l): ΔH = -285.830 kJ/mol, ΔS = 69.95 - 130.68 - 0.5*205.15 J/(mol·K)
        //            = 69.95 - 130.68 - 102.575 = -163.305 J/(mol·K) (for formation)
        // ΔG = ΔH - T*ΔS = -285.830 - 298*(-163.305/1000) = -285.830 + 48.664 = -237.166 kJ/mol
        let delta_h = -285.830;
        let delta_s = -163.305; // J/(mol·K)
        let dg = gibbs_free_energy(delta_h, 298.0, delta_s);
        assert!(approx_eq(dg, -237.14, 1.0), "ΔG water formation = {dg}");
    }

    #[test]
    fn test_gibbs_spontaneity() {
        assert!(is_spontaneous(-50.0));
        assert!(!is_spontaneous(50.0));
        assert!(!is_spontaneous(0.0));
    }

    #[test]
    fn test_equilibrium_temp() {
        // T = ΔH/ΔS; for ΔH = -100 kJ/mol, ΔS = -200 J/(mol·K)
        // T = (-100*1000)/(-200) = 500 K
        let t = equilibrium_temp(-100.0, -200.0);
        assert!(approx_eq(t, 500.0, 0.1), "Equilibrium T = {t} K");
    }

    // --- Equilibrium ---

    #[test]
    fn test_keq_gibbs_roundtrip() {
        let dg = -20.0; // kJ/mol
        let temp = 298.0;
        let keq = keq_from_gibbs(dg, temp);
        let dg2 = gibbs_from_keq(keq, temp);
        assert!(approx_eq(dg, dg2, 1e-6), "Roundtrip: {dg} → {keq} → {dg2}");
    }

    #[test]
    fn test_keq_from_gibbs_known() {
        // ΔG = 0 → Keq = 1
        assert!(approx_eq(keq_from_gibbs(0.0, 298.0), 1.0, 1e-10));
        // ΔG < 0 → Keq > 1 (products favoured)
        assert!(keq_from_gibbs(-10.0, 298.0) > 1.0);
        // ΔG > 0 → Keq < 1 (reactants favoured)
        assert!(keq_from_gibbs(10.0, 298.0) < 1.0);
    }

    #[test]
    fn test_le_chatelier_add_reactant() {
        assert_eq!(
            le_chatelier("exothermic", Perturbation::AddReactant),
            "shifts right"
        );
    }

    #[test]
    fn test_le_chatelier_increase_temp_exothermic() {
        assert_eq!(
            le_chatelier("exothermic", Perturbation::IncreaseTemp),
            "shifts left"
        );
    }

    #[test]
    fn test_le_chatelier_increase_temp_endothermic() {
        assert_eq!(
            le_chatelier("endothermic", Perturbation::IncreaseTemp),
            "shifts right"
        );
    }

    // --- Kinetics ---

    #[test]
    fn test_arrhenius_k_values() {
        // k at T1 and T2 should differ
        let a = 1e13f64;
        let ea = 50000.0; // J/mol
        let k1 = arrhenius_k(a, ea, 300.0);
        let k2 = arrhenius_k(a, ea, 400.0);
        assert!(k2 > k1, "Higher T should give higher k");
    }

    #[test]
    fn test_activation_energy_roundtrip() {
        let a = 1e13f64;
        let ea_true = 80000.0; // J/mol
        let k1 = arrhenius_k(a, ea_true, 300.0);
        let k2 = arrhenius_k(a, ea_true, 400.0);
        let ea_calc = activation_energy(k1, k2, 300.0, 400.0);
        assert!(
            approx_eq(ea_calc, ea_true, 1.0),
            "Ea roundtrip: {ea_calc} vs {ea_true}"
        );
    }

    #[test]
    fn test_half_life_first_order() {
        let k = 0.1; // s⁻¹
        let t_half = half_life_first_order(k);
        let c0 = 1.0;
        let c_at_half = concentration_first_order(c0, k, t_half);
        assert!(
            approx_eq(c_at_half, 0.5, 1e-6),
            "c at half-life should be 0.5, got {c_at_half}"
        );
    }

    #[test]
    fn test_half_life_first_order_formula() {
        // t₁/₂ = ln(2)/k
        let k = 0.693147; // ≈ ln(2), so t₁/₂ ≈ 1
        assert!(approx_eq(half_life_first_order(k), 1.0, 1e-5));
    }

    #[test]
    fn test_concentration_integrated_laws() {
        let c0 = 2.0;
        let k = 0.5;
        let t = 1.0;
        let c_zero = concentration_zero_order(c0, k, t);
        let c_first = concentration_first_order(c0, k, t);
        let c_second = concentration_second_order(c0, k, t);
        assert!(approx_eq(c_zero, 1.5, 1e-10));
        assert!(approx_eq(c_first, 2.0 * (-0.5f64).exp(), 1e-10));
        assert!(approx_eq(c_second, 1.0 / (0.5 + 0.5), 1e-10));
    }

    #[test]
    fn test_reaction_rate_law() {
        // r = k * [A]^2 * [B]^1, k=0.1, [A]=2.0, [B]=3.0 → r = 0.1*4*3 = 1.2
        let r = reaction_rate(0.1, &[2.0, 3.0], &[2.0, 1.0]);
        assert!(approx_eq(r, 1.2, 1e-10));
    }

    #[test]
    fn test_theoretical_yield() {
        // 2H2 + O2 → 2H2O; limiting: O2 (1 mol), coeff=1 product coeff=2, MM(H2O)=18.015
        let y = theoretical_yield(1.0, 1, 2, 18.015);
        assert!(approx_eq(y, 36.030, 0.01));
    }

    #[test]
    fn test_ice_table_simple() {
        // A ⇌ B, Ka = 1.0, [A]0 = 1.0, [B]0 = 0.0
        // x = 0.5 (since (0+x)/(1-x) = 1 → x = 0.5)
        let result = solve_ice(&[1.0, 0.0], &[-1, 1], 1.0).unwrap();
        assert!(approx_eq(result.equilibrium[0], 0.5, 0.01));
        assert!(approx_eq(result.equilibrium[1], 0.5, 0.01));
    }
}
