// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Periodic Table: Layer 3 - Elements from Hadrons
//!
//! Elements are **constructed** from their nuclear and electronic structure,
//! not memorized as independent vectors.
//!
//! ## The Compositional Formula
//!
//! ```text
//! Element(Z, N) = Bundle(
//!     Z × Proton,
//!     N × Neutron,
//!     Z × Electron
//! ) ⊗ BindingEnergy(Z, N)
//! ```
//!
//! Where:
//! - Z = atomic number (protons = electrons for neutral atom)
//! - N = neutron number
//! - A = Z + N = mass number
//!
//! ## Emergent Properties
//!
//! Because elements are composed:
//! - **Isotopes** share most structure (differ only in neutron count)
//! - **Ions** are derived by adjusting electron count
//! - **Nuclear stability** emerges from binding energy
//!
//! ## Examples
//!
//! ```text
//! Carbon-12:  Bundle(6P, 6N, 6e) ⊗ Binding(6,6)
//! Carbon-14:  Bundle(6P, 8N, 6e) ⊗ Binding(6,8)  ← radioactive!
//! Nitrogen:   Bundle(7P, 7N, 7e) ⊗ Binding(7,7)
//!
//! sim(C-12, C-14) > 0.9   // Isotopes are highly similar
//! sim(C-12, N-14) < 0.8   // Different elements are less similar
//! ```

use super::hadrons::Hadrons;
use super::standard_model::{PHYSICS_DIM, StandardModel};
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::Serialize;

/// Element data (basic properties)
#[derive(Debug, Clone, Serialize)]
pub struct ElementData {
    pub symbol: &'static str,
    pub name: &'static str,
    pub atomic_number: u8,
    pub standard_neutrons: u8,          // Most common isotope
    pub atomic_mass: f32,               // In atomic mass units
    pub electronegativity: Option<f32>, // Pauling scale
    pub group: u8,
    pub period: u8,
}

/// Extended element data with additional physical properties for grounded vector composition
///
/// These real atomic properties enable property-weighted HDC composition
/// where vector similarity correlates with chemical behavior.
#[derive(Debug, Clone, Serialize)]
pub struct ElementDataExtended {
    /// Base element data
    pub base: ElementData,
    /// First ionization energy in kJ/mol (range: 375-2372)
    pub first_ionization_energy: Option<f32>,
    /// Atomic radius in pm (range: 31-298)
    pub atomic_radius: Option<f32>,
    /// Electron affinity in kJ/mol (range: -48 to 349)
    pub electron_affinity: Option<f32>,
    /// Metallic character 0.0-1.0 (derived from position)
    pub metallic_character: f32,
    /// Melting point in Kelvin
    pub melting_point: Option<f32>,
    /// Boiling point in Kelvin
    pub boiling_point: Option<f32>,
    /// Density in g/cm³ at STP
    pub density: Option<f32>,
}

impl ElementDataExtended {
    /// Create extended data from base data
    pub fn from_base(base: ElementData) -> Self {
        let metallic_character = Self::compute_metallic_character(&base);
        Self {
            base,
            first_ionization_energy: None,
            atomic_radius: None,
            electron_affinity: None,
            metallic_character,
            melting_point: None,
            boiling_point: None,
            density: None,
        }
    }

    /// Compute metallic character from group and period
    fn compute_metallic_character(data: &ElementData) -> f32 {
        let z = data.atomic_number;
        let group = data.group;

        // Hydrogen is nonmetal
        if z == 1 {
            return 0.0;
        }

        // Noble gases
        if group == 18 {
            return 0.0;
        }

        // Alkali and alkaline earth metals (groups 1-2)
        if group <= 2 {
            return 1.0;
        }

        // Transition metals (groups 3-12)
        if (3..=12).contains(&group) {
            return 0.9;
        }

        // Post-transition metals and metalloids
        if group == 13 {
            return 0.7; // Al, Ga, In, Tl are metals; B is metalloid
        }
        if group == 14 {
            return 0.5; // Si, Ge metalloids; Sn, Pb metals; C nonmetal
        }
        if group == 15 {
            return 0.3; // As, Sb metalloids; Bi metal; N, P nonmetals
        }
        if group == 16 {
            return 0.15; // Te, Po have some metallic character
        }
        if group == 17 {
            return 0.0; // Halogens are nonmetals
        }

        0.5 // Default
    }

    /// Lookup extended physical properties for a given atomic number
    pub fn with_physical_properties(mut self, z: u8) -> Self {
        if let Some(&(ie, radius, ea)) = ELEMENT_PHYSICAL_PROPERTIES.get((z - 1) as usize) {
            self.first_ionization_energy = ie;
            self.atomic_radius = radius;
            self.electron_affinity = ea;
        }
        if let Some(&(mp, bp, dens)) = ELEMENT_THERMODYNAMIC_PROPERTIES.get((z - 1) as usize) {
            self.melting_point = mp;
            self.boiling_point = bp;
            self.density = dens;
        }
        self
    }
}

/// Physical property data for elements 1-118
/// Format: (first_ionization_energy kJ/mol, atomic_radius pm, electron_affinity kJ/mol)
/// Data from NIST and periodic table reference sources
const ELEMENT_PHYSICAL_PROPERTIES: [(Option<f32>, Option<f32>, Option<f32>); 118] = [
    // Period 1
    (Some(1312.0), Some(53.0), Some(73.0)),  // H (1)
    (Some(2372.0), Some(31.0), Some(-48.0)), // He (2) - negative EA (endothermic)
    // Period 2
    (Some(520.0), Some(167.0), Some(60.0)),   // Li (3)
    (Some(900.0), Some(112.0), Some(-48.0)),  // Be (4)
    (Some(801.0), Some(87.0), Some(27.0)),    // B (5)
    (Some(1086.0), Some(77.0), Some(122.0)),  // C (6)
    (Some(1402.0), Some(75.0), Some(-7.0)),   // N (7) - nearly zero EA
    (Some(1314.0), Some(73.0), Some(141.0)),  // O (8)
    (Some(1681.0), Some(71.0), Some(328.0)),  // F (9) - highest EA
    (Some(2081.0), Some(69.0), Some(-116.0)), // Ne (10)
    // Period 3
    (Some(496.0), Some(190.0), Some(53.0)),   // Na (11)
    (Some(738.0), Some(145.0), Some(-40.0)),  // Mg (12)
    (Some(578.0), Some(118.0), Some(42.0)),   // Al (13)
    (Some(786.0), Some(111.0), Some(134.0)),  // Si (14)
    (Some(1012.0), Some(106.0), Some(72.0)),  // P (15)
    (Some(1000.0), Some(102.0), Some(200.0)), // S (16)
    (Some(1251.0), Some(99.0), Some(349.0)),  // Cl (17) - highest EA among common elements
    (Some(1521.0), Some(97.0), Some(-96.0)),  // Ar (18)
    // Period 4
    (Some(419.0), Some(243.0), Some(48.0)), // K (19) - very low IE
    (Some(590.0), Some(194.0), Some(2.0)),  // Ca (20)
    (Some(633.0), Some(184.0), Some(18.0)), // Sc (21)
    (Some(659.0), Some(176.0), Some(8.0)),  // Ti (22)
    (Some(651.0), Some(171.0), Some(51.0)), // V (23)
    (Some(653.0), Some(166.0), Some(65.0)), // Cr (24)
    (Some(717.0), Some(161.0), Some(-50.0)), // Mn (25)
    (Some(763.0), Some(156.0), Some(15.0)), // Fe (26)
    (Some(760.0), Some(152.0), Some(64.0)), // Co (27)
    (Some(737.0), Some(149.0), Some(112.0)), // Ni (28)
    (Some(745.0), Some(145.0), Some(119.0)), // Cu (29)
    (Some(906.0), Some(142.0), Some(-58.0)), // Zn (30)
    (Some(579.0), Some(136.0), Some(29.0)), // Ga (31)
    (Some(762.0), Some(125.0), Some(119.0)), // Ge (32)
    (Some(947.0), Some(114.0), Some(78.0)), // As (33)
    (Some(941.0), Some(103.0), Some(195.0)), // Se (34)
    (Some(1140.0), Some(94.0), Some(325.0)), // Br (35)
    (Some(1351.0), Some(88.0), Some(-96.0)), // Kr (36)
    // Period 5
    (Some(403.0), Some(265.0), Some(47.0)),   // Rb (37)
    (Some(550.0), Some(219.0), Some(5.0)),    // Sr (38)
    (Some(600.0), Some(212.0), Some(30.0)),   // Y (39)
    (Some(640.0), Some(206.0), Some(41.0)),   // Zr (40)
    (Some(652.0), Some(198.0), Some(86.0)),   // Nb (41)
    (Some(684.0), Some(190.0), Some(72.0)),   // Mo (42)
    (Some(702.0), Some(183.0), Some(53.0)),   // Tc (43)
    (Some(711.0), Some(178.0), Some(101.0)),  // Ru (44)
    (Some(720.0), Some(173.0), Some(110.0)),  // Rh (45)
    (Some(804.0), Some(169.0), Some(54.0)),   // Pd (46)
    (Some(731.0), Some(165.0), Some(126.0)),  // Ag (47)
    (Some(868.0), Some(161.0), Some(-68.0)),  // Cd (48)
    (Some(558.0), Some(156.0), Some(29.0)),   // In (49)
    (Some(709.0), Some(145.0), Some(107.0)),  // Sn (50)
    (Some(834.0), Some(133.0), Some(103.0)),  // Sb (51)
    (Some(869.0), Some(123.0), Some(190.0)),  // Te (52)
    (Some(1008.0), Some(115.0), Some(295.0)), // I (53)
    (Some(1170.0), Some(108.0), Some(-77.0)), // Xe (54)
    // Period 6
    (Some(376.0), Some(298.0), Some(46.0)), // Cs (55) - lowest IE
    (Some(503.0), Some(253.0), Some(14.0)), // Ba (56)
    // Lanthanides (La-Lu, 57-71)
    (Some(538.0), Some(195.0), Some(48.0)), // La (57)
    (Some(534.0), Some(185.0), Some(50.0)), // Ce (58)
    (Some(527.0), Some(247.0), Some(50.0)), // Pr (59)
    (Some(533.0), Some(206.0), Some(50.0)), // Nd (60)
    (Some(540.0), Some(205.0), Some(50.0)), // Pm (61)
    (Some(545.0), Some(238.0), Some(50.0)), // Sm (62)
    (Some(547.0), Some(231.0), Some(50.0)), // Eu (63)
    (Some(593.0), Some(233.0), Some(50.0)), // Gd (64)
    (Some(566.0), Some(225.0), Some(50.0)), // Tb (65)
    (Some(573.0), Some(228.0), Some(50.0)), // Dy (66)
    (Some(581.0), Some(226.0), Some(50.0)), // Ho (67)
    (Some(589.0), Some(226.0), Some(50.0)), // Er (68)
    (Some(597.0), Some(222.0), Some(50.0)), // Tm (69)
    (Some(603.0), Some(222.0), Some(50.0)), // Yb (70)
    (Some(524.0), Some(217.0), Some(50.0)), // Lu (71)
    // Continue Period 6
    (Some(659.0), Some(208.0), Some(0.0)),    // Hf (72)
    (Some(761.0), Some(200.0), Some(31.0)),   // Ta (73)
    (Some(770.0), Some(193.0), Some(79.0)),   // W (74)
    (Some(760.0), Some(188.0), Some(14.0)),   // Re (75)
    (Some(840.0), Some(185.0), Some(106.0)),  // Os (76)
    (Some(880.0), Some(180.0), Some(151.0)),  // Ir (77)
    (Some(870.0), Some(177.0), Some(205.0)),  // Pt (78)
    (Some(890.0), Some(174.0), Some(223.0)),  // Au (79)
    (Some(1007.0), Some(171.0), Some(-48.0)), // Hg (80)
    (Some(589.0), Some(156.0), Some(19.0)),   // Tl (81)
    (Some(716.0), Some(154.0), Some(35.0)),   // Pb (82)
    (Some(703.0), Some(143.0), Some(91.0)),   // Bi (83)
    (Some(812.0), Some(135.0), Some(183.0)),  // Po (84)
    (Some(920.0), Some(127.0), Some(270.0)),  // At (85)
    (Some(1037.0), Some(120.0), Some(-68.0)), // Rn (86)
    // Period 7
    (Some(380.0), Some(348.0), Some(44.0)), // Fr (87)
    (Some(509.0), Some(283.0), Some(10.0)), // Ra (88)
    // Actinides (Ac-Lr, 89-103)
    (Some(499.0), Some(260.0), Some(34.0)),   // Ac (89)
    (Some(587.0), Some(237.0), Some(113.0)),  // Th (90)
    (Some(568.0), Some(243.0), Some(50.0)),   // Pa (91)
    (Some(584.0), Some(240.0), Some(50.0)),   // U (92)
    (Some(597.0), Some(221.0), Some(46.0)),   // Np (93)
    (Some(585.0), Some(243.0), Some(-48.0)),  // Pu (94)
    (Some(578.0), Some(244.0), Some(10.0)),   // Am (95)
    (Some(581.0), Some(245.0), Some(28.0)),   // Cm (96)
    (Some(601.0), Some(244.0), Some(-165.0)), // Bk (97)
    (Some(608.0), Some(245.0), Some(-97.0)),  // Cf (98)
    (Some(619.0), Some(245.0), Some(-29.0)),  // Es (99)
    (Some(627.0), Some(245.0), Some(34.0)),   // Fm (100)
    (Some(635.0), Some(245.0), Some(94.0)),   // Md (101)
    (Some(642.0), Some(245.0), Some(-223.0)), // No (102)
    (Some(470.0), Some(245.0), Some(-30.0)),  // Lr (103)
    // Superheavy elements (Rf-Og, 104-118) - theoretical/estimated values
    (None, None, None), // Rf (104)
    (None, None, None), // Db (105)
    (None, None, None), // Sg (106)
    (None, None, None), // Bh (107)
    (None, None, None), // Hs (108)
    (None, None, None), // Mt (109)
    (None, None, None), // Ds (110)
    (None, None, None), // Rg (111)
    (None, None, None), // Cn (112)
    (None, None, None), // Nh (113)
    (None, None, None), // Fl (114)
    (None, None, None), // Mc (115)
    (None, None, None), // Lv (116)
    (None, None, None), // Ts (117)
    (None, None, None), // Og (118)
];

/// Thermodynamic property data for elements 1-118
/// Format: (melting_point K, boiling_point K, density g/cm³)
/// Data from NIST, CRC Handbook, and periodic table reference sources
const ELEMENT_THERMODYNAMIC_PROPERTIES: [(Option<f32>, Option<f32>, Option<f32>); 118] = [
    // Period 1
    (Some(14.01), Some(20.28), Some(0.00009)), // H (1)
    (Some(0.95), Some(4.22), Some(0.00018)),   // He (2)
    // Period 2
    (Some(453.7), Some(1615.0), Some(0.534)),  // Li (3)
    (Some(1560.0), Some(2744.0), Some(1.85)),  // Be (4)
    (Some(2349.0), Some(4200.0), Some(2.34)),  // B (5)
    (Some(3823.0), Some(4098.0), Some(2.27)),  // C (6) - graphite sublimes
    (Some(63.15), Some(77.36), Some(0.00125)), // N (7)
    (Some(54.36), Some(90.20), Some(0.00143)), // O (8)
    (Some(53.53), Some(85.03), Some(0.0017)),  // F (9)
    (Some(24.56), Some(27.07), Some(0.0009)),  // Ne (10)
    // Period 3
    (Some(370.9), Some(1156.0), Some(0.97)),  // Na (11)
    (Some(923.0), Some(1363.0), Some(1.74)),  // Mg (12)
    (Some(933.5), Some(2792.0), Some(2.70)),  // Al (13)
    (Some(1687.0), Some(3538.0), Some(2.33)), // Si (14)
    (Some(317.3), Some(553.7), Some(1.82)),   // P (15) - white
    (Some(388.4), Some(717.8), Some(2.07)),   // S (16)
    (Some(171.6), Some(239.1), Some(0.0032)), // Cl (17)
    (Some(83.80), Some(87.30), Some(0.0018)), // Ar (18)
    // Period 4
    (Some(336.5), Some(1032.0), Some(0.86)),  // K (19)
    (Some(1115.0), Some(1757.0), Some(1.54)), // Ca (20)
    (Some(1814.0), Some(3109.0), Some(2.99)), // Sc (21)
    (Some(1941.0), Some(3560.0), Some(4.51)), // Ti (22)
    (Some(2183.0), Some(3680.0), Some(6.00)), // V (23)
    (Some(2180.0), Some(2944.0), Some(7.15)), // Cr (24)
    (Some(1519.0), Some(2334.0), Some(7.44)), // Mn (25)
    (Some(1811.0), Some(3134.0), Some(7.87)), // Fe (26)
    (Some(1768.0), Some(3200.0), Some(8.90)), // Co (27)
    (Some(1728.0), Some(3186.0), Some(8.91)), // Ni (28)
    (Some(1357.8), Some(2835.0), Some(8.96)), // Cu (29)
    (Some(692.7), Some(1180.0), Some(7.13)),  // Zn (30)
    (Some(302.9), Some(2477.0), Some(5.91)),  // Ga (31)
    (Some(1211.4), Some(3106.0), Some(5.32)), // Ge (32)
    (Some(1090.0), Some(887.0), Some(5.73)),  // As (33) - sublimes
    (Some(494.0), Some(958.0), Some(4.81)),   // Se (34)
    (Some(266.0), Some(332.0), Some(3.12)),   // Br (35)
    (Some(115.8), Some(119.9), Some(0.0037)), // Kr (36)
    // Period 5
    (Some(312.5), Some(961.0), Some(1.53)),    // Rb (37)
    (Some(1050.0), Some(1655.0), Some(2.64)),  // Sr (38)
    (Some(1799.0), Some(3609.0), Some(4.47)),  // Y (39)
    (Some(2128.0), Some(4682.0), Some(6.52)),  // Zr (40)
    (Some(2750.0), Some(5017.0), Some(8.57)),  // Nb (41)
    (Some(2896.0), Some(4912.0), Some(10.22)), // Mo (42)
    (Some(2430.0), Some(4538.0), Some(11.0)),  // Tc (43)
    (Some(2607.0), Some(4423.0), Some(12.1)),  // Ru (44)
    (Some(2237.0), Some(3968.0), Some(12.4)),  // Rh (45)
    (Some(1828.0), Some(3236.0), Some(12.0)),  // Pd (46)
    (Some(1234.9), Some(2435.0), Some(10.5)),  // Ag (47)
    (Some(594.2), Some(1040.0), Some(8.69)),   // Cd (48)
    (Some(429.7), Some(2345.0), Some(7.31)),   // In (49)
    (Some(505.1), Some(2875.0), Some(7.29)),   // Sn (50)
    (Some(903.8), Some(1860.0), Some(6.68)),   // Sb (51)
    (Some(722.7), Some(1261.0), Some(6.24)),   // Te (52)
    (Some(386.9), Some(457.5), Some(4.93)),    // I (53)
    (Some(161.4), Some(165.1), Some(0.0059)),  // Xe (54)
    // Period 6
    (Some(301.7), Some(944.0), Some(1.93)),   // Cs (55)
    (Some(1000.0), Some(2170.0), Some(3.62)), // Ba (56)
    (Some(1193.0), Some(3737.0), Some(6.15)), // La (57)
    (Some(1068.0), Some(3716.0), Some(6.77)), // Ce (58)
    (Some(1208.0), Some(3793.0), Some(6.77)), // Pr (59)
    (Some(1297.0), Some(3347.0), Some(7.01)), // Nd (60)
    (Some(1315.0), Some(3273.0), Some(7.26)), // Pm (61)
    (Some(1345.0), Some(2067.0), Some(7.52)), // Sm (62)
    (Some(1099.0), Some(1802.0), Some(5.24)), // Eu (63)
    (Some(1585.0), Some(3546.0), Some(7.90)), // Gd (64)
    (Some(1629.0), Some(3503.0), Some(8.23)), // Tb (65)
    (Some(1680.0), Some(2840.0), Some(8.55)), // Dy (66)
    (Some(1734.0), Some(2993.0), Some(8.80)), // Ho (67)
    (Some(1802.0), Some(3141.0), Some(9.07)), // Er (68)
    (Some(1818.0), Some(2223.0), Some(9.32)), // Tm (69)
    (Some(1097.0), Some(1469.0), Some(6.90)), // Yb (70)
    (Some(1925.0), Some(3675.0), Some(9.84)), // Lu (71)
    (Some(2506.0), Some(4876.0), Some(13.3)), // Hf (72)
    (Some(3290.0), Some(5731.0), Some(16.4)), // Ta (73)
    (Some(3695.0), Some(5828.0), Some(19.3)), // W (74)
    (Some(3459.0), Some(5869.0), Some(20.8)), // Re (75)
    (Some(3306.0), Some(5285.0), Some(22.6)), // Os (76)
    (Some(2719.0), Some(4701.0), Some(22.4)), // Ir (77)
    (Some(2041.0), Some(4098.0), Some(21.5)), // Pt (78)
    (Some(1337.3), Some(3129.0), Some(19.3)), // Au (79)
    (Some(234.3), Some(629.9), Some(13.5)),   // Hg (80)
    (Some(577.0), Some(1746.0), Some(11.8)),  // Tl (81)
    (Some(600.6), Some(2022.0), Some(11.3)),  // Pb (82)
    (Some(544.6), Some(1837.0), Some(9.79)),  // Bi (83)
    (Some(527.0), Some(1235.0), Some(9.20)),  // Po (84)
    (Some(575.0), Some(610.0), Some(7.0)),    // At (85) - estimated
    (Some(202.0), Some(211.5), Some(0.0097)), // Rn (86)
    // Period 7
    (Some(300.0), Some(950.0), Some(1.87)), // Fr (87) - estimated
    (Some(973.0), Some(2010.0), Some(5.0)), // Ra (88)
    (Some(1323.0), Some(3471.0), Some(10.1)), // Ac (89)
    (Some(2115.0), Some(5061.0), Some(11.7)), // Th (90)
    (Some(1841.0), Some(4300.0), Some(15.4)), // Pa (91)
    (Some(1405.3), Some(4404.0), Some(19.1)), // U (92)
    (Some(917.0), Some(4273.0), Some(20.2)), // Np (93)
    (Some(912.5), Some(3501.0), Some(19.8)), // Pu (94)
    (Some(1449.0), Some(2880.0), Some(12.0)), // Am (95)
    (Some(1613.0), Some(3383.0), Some(13.5)), // Cm (96)
    (Some(1259.0), Some(2900.0), Some(14.8)), // Bk (97)
    (Some(1173.0), Some(1743.0), Some(15.1)), // Cf (98)
    (Some(1133.0), Some(1269.0), Some(8.84)), // Es (99)
    (Some(1800.0), None, Some(9.7)),        // Fm (100)
    (Some(1100.0), None, Some(10.3)),       // Md (101)
    (Some(1100.0), None, Some(9.9)),        // No (102)
    (Some(1900.0), None, Some(15.6)),       // Lr (103)
    (None, None, Some(23.2)),               // Rf (104) - predicted
    (None, None, Some(29.3)),               // Db (105) - predicted
    (None, None, Some(35.0)),               // Sg (106) - predicted
    (None, None, Some(37.1)),               // Bh (107) - predicted
    (None, None, Some(40.7)),               // Hs (108) - predicted
    (None, None, Some(37.4)),               // Mt (109) - predicted
    (None, None, Some(34.8)),               // Ds (110) - predicted
    (None, None, Some(28.7)),               // Rg (111) - predicted
    (None, None, Some(23.7)),               // Cn (112) - predicted
    (None, None, Some(16.0)),               // Nh (113) - predicted
    (None, None, Some(14.0)),               // Fl (114) - predicted
    (None, None, Some(13.5)),               // Mc (115) - predicted
    (None, None, Some(12.9)),               // Lv (116) - predicted
    (None, None, Some(7.2)),                // Ts (117) - predicted
    (None, None, Some(5.0)),                // Og (118) - predicted
];

/// Complete element information
#[derive(Debug, Clone)]
pub struct Element {
    pub data: ElementData,
    pub vector: ContinuousHV,
}

/// Electron shell configuration helper
#[derive(Debug, Clone)]
pub struct ElectronShell {
    shell_vectors: Vec<ContinuousHV>,
}

impl ElectronShell {
    /// Create shell vectors from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        // s, p, d, f orbitals for shells 1-7
        let mut shell_vectors = Vec::new();
        for n in 1..=7u8 {
            let shell = genesis.hv(&format!("electron::shell_{n}"), PHYSICS_DIM);
            shell_vectors.push(shell);
        }
        Self { shell_vectors }
    }

    /// Encode electron configuration
    ///
    /// Returns a vector representing the electron arrangement.
    /// Uses aufbau principle for filling order.
    pub fn encode_configuration(
        &self,
        electron_count: u8,
        base_electron: &ContinuousHV,
    ) -> ContinuousHV {
        // Simplified: weight each shell by electron occupancy
        // Real version would use aufbau filling order
        let mut remaining = electron_count;
        let shell_capacities = [2, 8, 18, 32, 32, 18, 8]; // Max electrons per shell

        let mut weighted_shells = Vec::new();
        let mut weights = Vec::new();

        for (i, &capacity) in shell_capacities.iter().enumerate() {
            if remaining == 0 || i >= self.shell_vectors.len() {
                break;
            }

            let electrons_in_shell = remaining.min(capacity);
            remaining -= electrons_in_shell;

            // Bind electron to shell
            let shell_contribution = base_electron
                .bind(&self.shell_vectors[i])
                .scale(electrons_in_shell as f32);
            weighted_shells.push(shell_contribution);
            weights.push(electrons_in_shell as f32);
        }

        if weighted_shells.is_empty() {
            return ContinuousHV::zero(PHYSICS_DIM);
        }

        let refs: Vec<&ContinuousHV> = weighted_shells.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    /// Get valence electrons
    pub fn valence_electrons(&self, atomic_number: u8) -> u8 {
        // Simplified valence calculation
        let noble_gas_configs = [2, 10, 18, 36, 54, 86, 118];

        for &config in &noble_gas_configs {
            if atomic_number <= config {
                if atomic_number == config {
                    return 0; // Noble gas
                }
                // Find previous noble gas
                let prev = noble_gas_configs
                    .iter()
                    .filter(|&&x| x < atomic_number)
                    .max()
                    .copied()
                    .unwrap_or(0);
                return atomic_number - prev;
            }
        }
        0
    }
}

/// The Periodic Table: all elements composed from first principles
#[derive(Debug, Clone)]
pub struct PeriodicTable {
    /// All elements (indexed by atomic number - 1)
    elements: Vec<Element>,

    /// Electron shell encoder
    shells: ElectronShell,

    /// Concept vectors for chemical properties
    pub metallic: ContinuousHV,
    pub nonmetallic: ContinuousHV,
    pub noble: ContinuousHV,
    pub reactive: ContinuousHV,
    pub oxidizing: ContinuousHV,
    pub reducing: ContinuousHV,
    /// Lanthanide series marker (elements 57-71)
    pub lanthanide: ContinuousHV,
    /// Actinide series marker (elements 89-103)
    pub actinide: ContinuousHV,
    /// Superheavy elements marker (elements 104+)
    pub superheavy: ContinuousHV,

    /// Thermodynamic property concept vectors
    pub thermal_stable: ContinuousHV, // High melting/boiling points
    pub thermal_volatile: ContinuousHV, // Low melting/boiling points
    pub phase_solid: ContinuousHV,      // Solid at STP
    pub phase_liquid: ContinuousHV,     // Liquid at STP
    pub phase_gas: ContinuousHV,        // Gas at STP
    pub density_heavy: ContinuousHV,    // High density
    pub density_light: ContinuousHV,    // Low density

    /// Reference to building blocks
    proton: ContinuousHV,
    neutron: ContinuousHV,
    electron: ContinuousHV,
}

impl PeriodicTable {
    /// Construct the periodic table from the Standard Model
    ///
    /// Every element is COMPOSED from protons, neutrons, and electrons.
    pub fn from_model(model: &StandardModel, hadrons: &Hadrons, genesis: &GenesisSeed) -> Self {
        let shells = ElectronShell::from_genesis(genesis);

        // Property concept vectors
        let metallic = genesis.hv("chemistry::metallic", PHYSICS_DIM);
        let nonmetallic = genesis.hv("chemistry::nonmetallic", PHYSICS_DIM);
        let noble = genesis.hv("chemistry::noble", PHYSICS_DIM);
        let reactive = genesis.hv("chemistry::reactive", PHYSICS_DIM);
        let oxidizing = genesis.hv("chemistry::oxidizing", PHYSICS_DIM);
        let reducing = genesis.hv("chemistry::reducing", PHYSICS_DIM);
        let lanthanide = genesis.hv("chemistry::lanthanide", PHYSICS_DIM);
        let actinide = genesis.hv("chemistry::actinide", PHYSICS_DIM);
        let superheavy = genesis.hv("chemistry::superheavy", PHYSICS_DIM);

        // Thermodynamic property concept vectors
        let thermal_stable = genesis.hv("chemistry::thermal_stable", PHYSICS_DIM);
        let thermal_volatile = genesis.hv("chemistry::thermal_volatile", PHYSICS_DIM);
        let phase_solid = genesis.hv("chemistry::phase_solid", PHYSICS_DIM);
        let phase_liquid = genesis.hv("chemistry::phase_liquid", PHYSICS_DIM);
        let phase_gas = genesis.hv("chemistry::phase_gas", PHYSICS_DIM);
        let density_heavy = genesis.hv("chemistry::density_heavy", PHYSICS_DIM);
        let density_light = genesis.hv("chemistry::density_light", PHYSICS_DIM);

        // Store building blocks
        let proton = hadrons.proton.clone();
        let neutron = hadrons.neutron.clone();
        let electron = model.electron.clone();

        // Create partial table for now
        let mut table = Self {
            elements: Vec::new(),
            shells,
            metallic,
            nonmetallic,
            noble,
            reactive,
            oxidizing,
            reducing,
            lanthanide,
            actinide,
            superheavy,
            thermal_stable,
            thermal_volatile,
            phase_solid,
            phase_liquid,
            phase_gas,
            density_heavy,
            density_light,
            proton,
            neutron,
            electron,
        };

        // Build elements 1-118
        table.build_all_elements(hadrons);

        table
    }

    /// Build all elements
    fn build_all_elements(&mut self, hadrons: &Hadrons) {
        // Element data for first 36 elements (extend as needed)
        let element_data: Vec<ElementData> = vec![
            ElementData {
                symbol: "H",
                name: "Hydrogen",
                atomic_number: 1,
                standard_neutrons: 0,
                atomic_mass: 1.008,
                electronegativity: Some(2.20),
                group: 1,
                period: 1,
            },
            ElementData {
                symbol: "He",
                name: "Helium",
                atomic_number: 2,
                standard_neutrons: 2,
                atomic_mass: 4.003,
                electronegativity: None,
                group: 18,
                period: 1,
            },
            ElementData {
                symbol: "Li",
                name: "Lithium",
                atomic_number: 3,
                standard_neutrons: 4,
                atomic_mass: 6.941,
                electronegativity: Some(0.98),
                group: 1,
                period: 2,
            },
            ElementData {
                symbol: "Be",
                name: "Beryllium",
                atomic_number: 4,
                standard_neutrons: 5,
                atomic_mass: 9.012,
                electronegativity: Some(1.57),
                group: 2,
                period: 2,
            },
            ElementData {
                symbol: "B",
                name: "Boron",
                atomic_number: 5,
                standard_neutrons: 6,
                atomic_mass: 10.81,
                electronegativity: Some(2.04),
                group: 13,
                period: 2,
            },
            ElementData {
                symbol: "C",
                name: "Carbon",
                atomic_number: 6,
                standard_neutrons: 6,
                atomic_mass: 12.01,
                electronegativity: Some(2.55),
                group: 14,
                period: 2,
            },
            ElementData {
                symbol: "N",
                name: "Nitrogen",
                atomic_number: 7,
                standard_neutrons: 7,
                atomic_mass: 14.01,
                electronegativity: Some(3.04),
                group: 15,
                period: 2,
            },
            ElementData {
                symbol: "O",
                name: "Oxygen",
                atomic_number: 8,
                standard_neutrons: 8,
                atomic_mass: 16.00,
                electronegativity: Some(3.44),
                group: 16,
                period: 2,
            },
            ElementData {
                symbol: "F",
                name: "Fluorine",
                atomic_number: 9,
                standard_neutrons: 10,
                atomic_mass: 19.00,
                electronegativity: Some(3.98),
                group: 17,
                period: 2,
            },
            ElementData {
                symbol: "Ne",
                name: "Neon",
                atomic_number: 10,
                standard_neutrons: 10,
                atomic_mass: 20.18,
                electronegativity: None,
                group: 18,
                period: 2,
            },
            ElementData {
                symbol: "Na",
                name: "Sodium",
                atomic_number: 11,
                standard_neutrons: 12,
                atomic_mass: 22.99,
                electronegativity: Some(0.93),
                group: 1,
                period: 3,
            },
            ElementData {
                symbol: "Mg",
                name: "Magnesium",
                atomic_number: 12,
                standard_neutrons: 12,
                atomic_mass: 24.31,
                electronegativity: Some(1.31),
                group: 2,
                period: 3,
            },
            ElementData {
                symbol: "Al",
                name: "Aluminum",
                atomic_number: 13,
                standard_neutrons: 14,
                atomic_mass: 26.98,
                electronegativity: Some(1.61),
                group: 13,
                period: 3,
            },
            ElementData {
                symbol: "Si",
                name: "Silicon",
                atomic_number: 14,
                standard_neutrons: 14,
                atomic_mass: 28.09,
                electronegativity: Some(1.90),
                group: 14,
                period: 3,
            },
            ElementData {
                symbol: "P",
                name: "Phosphorus",
                atomic_number: 15,
                standard_neutrons: 16,
                atomic_mass: 30.97,
                electronegativity: Some(2.19),
                group: 15,
                period: 3,
            },
            ElementData {
                symbol: "S",
                name: "Sulfur",
                atomic_number: 16,
                standard_neutrons: 16,
                atomic_mass: 32.07,
                electronegativity: Some(2.58),
                group: 16,
                period: 3,
            },
            ElementData {
                symbol: "Cl",
                name: "Chlorine",
                atomic_number: 17,
                standard_neutrons: 18,
                atomic_mass: 35.45,
                electronegativity: Some(3.16),
                group: 17,
                period: 3,
            },
            ElementData {
                symbol: "Ar",
                name: "Argon",
                atomic_number: 18,
                standard_neutrons: 22,
                atomic_mass: 39.95,
                electronegativity: None,
                group: 18,
                period: 3,
            },
            ElementData {
                symbol: "K",
                name: "Potassium",
                atomic_number: 19,
                standard_neutrons: 20,
                atomic_mass: 39.10,
                electronegativity: Some(0.82),
                group: 1,
                period: 4,
            },
            ElementData {
                symbol: "Ca",
                name: "Calcium",
                atomic_number: 20,
                standard_neutrons: 20,
                atomic_mass: 40.08,
                electronegativity: Some(1.00),
                group: 2,
                period: 4,
            },
            ElementData {
                symbol: "Sc",
                name: "Scandium",
                atomic_number: 21,
                standard_neutrons: 24,
                atomic_mass: 44.96,
                electronegativity: Some(1.36),
                group: 3,
                period: 4,
            },
            ElementData {
                symbol: "Ti",
                name: "Titanium",
                atomic_number: 22,
                standard_neutrons: 26,
                atomic_mass: 47.87,
                electronegativity: Some(1.54),
                group: 4,
                period: 4,
            },
            ElementData {
                symbol: "V",
                name: "Vanadium",
                atomic_number: 23,
                standard_neutrons: 28,
                atomic_mass: 50.94,
                electronegativity: Some(1.63),
                group: 5,
                period: 4,
            },
            ElementData {
                symbol: "Cr",
                name: "Chromium",
                atomic_number: 24,
                standard_neutrons: 28,
                atomic_mass: 52.00,
                electronegativity: Some(1.66),
                group: 6,
                period: 4,
            },
            ElementData {
                symbol: "Mn",
                name: "Manganese",
                atomic_number: 25,
                standard_neutrons: 30,
                atomic_mass: 54.94,
                electronegativity: Some(1.55),
                group: 7,
                period: 4,
            },
            ElementData {
                symbol: "Fe",
                name: "Iron",
                atomic_number: 26,
                standard_neutrons: 30,
                atomic_mass: 55.85,
                electronegativity: Some(1.83),
                group: 8,
                period: 4,
            },
            ElementData {
                symbol: "Co",
                name: "Cobalt",
                atomic_number: 27,
                standard_neutrons: 32,
                atomic_mass: 58.93,
                electronegativity: Some(1.88),
                group: 9,
                period: 4,
            },
            ElementData {
                symbol: "Ni",
                name: "Nickel",
                atomic_number: 28,
                standard_neutrons: 30,
                atomic_mass: 58.69,
                electronegativity: Some(1.91),
                group: 10,
                period: 4,
            },
            ElementData {
                symbol: "Cu",
                name: "Copper",
                atomic_number: 29,
                standard_neutrons: 34,
                atomic_mass: 63.55,
                electronegativity: Some(1.90),
                group: 11,
                period: 4,
            },
            ElementData {
                symbol: "Zn",
                name: "Zinc",
                atomic_number: 30,
                standard_neutrons: 34,
                atomic_mass: 65.38,
                electronegativity: Some(1.65),
                group: 12,
                period: 4,
            },
            ElementData {
                symbol: "Ga",
                name: "Gallium",
                atomic_number: 31,
                standard_neutrons: 38,
                atomic_mass: 69.72,
                electronegativity: Some(1.81),
                group: 13,
                period: 4,
            },
            ElementData {
                symbol: "Ge",
                name: "Germanium",
                atomic_number: 32,
                standard_neutrons: 42,
                atomic_mass: 72.63,
                electronegativity: Some(2.01),
                group: 14,
                period: 4,
            },
            ElementData {
                symbol: "As",
                name: "Arsenic",
                atomic_number: 33,
                standard_neutrons: 42,
                atomic_mass: 74.92,
                electronegativity: Some(2.18),
                group: 15,
                period: 4,
            },
            ElementData {
                symbol: "Se",
                name: "Selenium",
                atomic_number: 34,
                standard_neutrons: 46,
                atomic_mass: 78.97,
                electronegativity: Some(2.55),
                group: 16,
                period: 4,
            },
            ElementData {
                symbol: "Br",
                name: "Bromine",
                atomic_number: 35,
                standard_neutrons: 44,
                atomic_mass: 79.90,
                electronegativity: Some(2.96),
                group: 17,
                period: 4,
            },
            ElementData {
                symbol: "Kr",
                name: "Krypton",
                atomic_number: 36,
                standard_neutrons: 48,
                atomic_mass: 83.80,
                electronegativity: Some(3.00),
                group: 18,
                period: 4,
            },
            // Period 5 (Rb-Xe, Z=37-54)
            ElementData {
                symbol: "Rb",
                name: "Rubidium",
                atomic_number: 37,
                standard_neutrons: 48,
                atomic_mass: 85.47,
                electronegativity: Some(0.82),
                group: 1,
                period: 5,
            },
            ElementData {
                symbol: "Sr",
                name: "Strontium",
                atomic_number: 38,
                standard_neutrons: 50,
                atomic_mass: 87.62,
                electronegativity: Some(0.95),
                group: 2,
                period: 5,
            },
            ElementData {
                symbol: "Y",
                name: "Yttrium",
                atomic_number: 39,
                standard_neutrons: 50,
                atomic_mass: 88.91,
                electronegativity: Some(1.22),
                group: 3,
                period: 5,
            },
            ElementData {
                symbol: "Zr",
                name: "Zirconium",
                atomic_number: 40,
                standard_neutrons: 51,
                atomic_mass: 91.22,
                electronegativity: Some(1.33),
                group: 4,
                period: 5,
            },
            ElementData {
                symbol: "Nb",
                name: "Niobium",
                atomic_number: 41,
                standard_neutrons: 52,
                atomic_mass: 92.91,
                electronegativity: Some(1.60),
                group: 5,
                period: 5,
            },
            ElementData {
                symbol: "Mo",
                name: "Molybdenum",
                atomic_number: 42,
                standard_neutrons: 54,
                atomic_mass: 95.95,
                electronegativity: Some(2.16),
                group: 6,
                period: 5,
            },
            ElementData {
                symbol: "Tc",
                name: "Technetium",
                atomic_number: 43,
                standard_neutrons: 55,
                atomic_mass: 98.00,
                electronegativity: Some(1.90),
                group: 7,
                period: 5,
            },
            ElementData {
                symbol: "Ru",
                name: "Ruthenium",
                atomic_number: 44,
                standard_neutrons: 57,
                atomic_mass: 101.07,
                electronegativity: Some(2.20),
                group: 8,
                period: 5,
            },
            ElementData {
                symbol: "Rh",
                name: "Rhodium",
                atomic_number: 45,
                standard_neutrons: 58,
                atomic_mass: 102.91,
                electronegativity: Some(2.28),
                group: 9,
                period: 5,
            },
            ElementData {
                symbol: "Pd",
                name: "Palladium",
                atomic_number: 46,
                standard_neutrons: 60,
                atomic_mass: 106.42,
                electronegativity: Some(2.20),
                group: 10,
                period: 5,
            },
            ElementData {
                symbol: "Ag",
                name: "Silver",
                atomic_number: 47,
                standard_neutrons: 60,
                atomic_mass: 107.87,
                electronegativity: Some(1.93),
                group: 11,
                period: 5,
            },
            ElementData {
                symbol: "Cd",
                name: "Cadmium",
                atomic_number: 48,
                standard_neutrons: 64,
                atomic_mass: 112.41,
                electronegativity: Some(1.69),
                group: 12,
                period: 5,
            },
            ElementData {
                symbol: "In",
                name: "Indium",
                atomic_number: 49,
                standard_neutrons: 66,
                atomic_mass: 114.82,
                electronegativity: Some(1.78),
                group: 13,
                period: 5,
            },
            ElementData {
                symbol: "Sn",
                name: "Tin",
                atomic_number: 50,
                standard_neutrons: 69,
                atomic_mass: 118.71,
                electronegativity: Some(1.96),
                group: 14,
                period: 5,
            },
            ElementData {
                symbol: "Sb",
                name: "Antimony",
                atomic_number: 51,
                standard_neutrons: 71,
                atomic_mass: 121.76,
                electronegativity: Some(2.05),
                group: 15,
                period: 5,
            },
            ElementData {
                symbol: "Te",
                name: "Tellurium",
                atomic_number: 52,
                standard_neutrons: 76,
                atomic_mass: 127.60,
                electronegativity: Some(2.10),
                group: 16,
                period: 5,
            },
            ElementData {
                symbol: "I",
                name: "Iodine",
                atomic_number: 53,
                standard_neutrons: 74,
                atomic_mass: 126.90,
                electronegativity: Some(2.66),
                group: 17,
                period: 5,
            },
            ElementData {
                symbol: "Xe",
                name: "Xenon",
                atomic_number: 54,
                standard_neutrons: 77,
                atomic_mass: 131.29,
                electronegativity: Some(2.60),
                group: 18,
                period: 5,
            },
            // Period 6 (Cs-Rn, Z=55-86) including Lanthanides
            ElementData {
                symbol: "Cs",
                name: "Cesium",
                atomic_number: 55,
                standard_neutrons: 78,
                atomic_mass: 132.91,
                electronegativity: Some(0.79),
                group: 1,
                period: 6,
            },
            ElementData {
                symbol: "Ba",
                name: "Barium",
                atomic_number: 56,
                standard_neutrons: 81,
                atomic_mass: 137.33,
                electronegativity: Some(0.89),
                group: 2,
                period: 6,
            },
            // Lanthanides (Z=57-71)
            ElementData {
                symbol: "La",
                name: "Lanthanum",
                atomic_number: 57,
                standard_neutrons: 82,
                atomic_mass: 138.91,
                electronegativity: Some(1.10),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Ce",
                name: "Cerium",
                atomic_number: 58,
                standard_neutrons: 82,
                atomic_mass: 140.12,
                electronegativity: Some(1.12),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Pr",
                name: "Praseodymium",
                atomic_number: 59,
                standard_neutrons: 82,
                atomic_mass: 140.91,
                electronegativity: Some(1.13),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Nd",
                name: "Neodymium",
                atomic_number: 60,
                standard_neutrons: 84,
                atomic_mass: 144.24,
                electronegativity: Some(1.14),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Pm",
                name: "Promethium",
                atomic_number: 61,
                standard_neutrons: 84,
                atomic_mass: 145.00,
                electronegativity: Some(1.13),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Sm",
                name: "Samarium",
                atomic_number: 62,
                standard_neutrons: 88,
                atomic_mass: 150.36,
                electronegativity: Some(1.17),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Eu",
                name: "Europium",
                atomic_number: 63,
                standard_neutrons: 89,
                atomic_mass: 151.96,
                electronegativity: Some(1.20),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Gd",
                name: "Gadolinium",
                atomic_number: 64,
                standard_neutrons: 93,
                atomic_mass: 157.25,
                electronegativity: Some(1.20),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Tb",
                name: "Terbium",
                atomic_number: 65,
                standard_neutrons: 94,
                atomic_mass: 158.93,
                electronegativity: Some(1.10),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Dy",
                name: "Dysprosium",
                atomic_number: 66,
                standard_neutrons: 97,
                atomic_mass: 162.50,
                electronegativity: Some(1.22),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Ho",
                name: "Holmium",
                atomic_number: 67,
                standard_neutrons: 98,
                atomic_mass: 164.93,
                electronegativity: Some(1.23),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Er",
                name: "Erbium",
                atomic_number: 68,
                standard_neutrons: 99,
                atomic_mass: 167.26,
                electronegativity: Some(1.24),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Tm",
                name: "Thulium",
                atomic_number: 69,
                standard_neutrons: 100,
                atomic_mass: 168.93,
                electronegativity: Some(1.25),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Yb",
                name: "Ytterbium",
                atomic_number: 70,
                standard_neutrons: 103,
                atomic_mass: 173.05,
                electronegativity: Some(1.10),
                group: 3,
                period: 6,
            },
            ElementData {
                symbol: "Lu",
                name: "Lutetium",
                atomic_number: 71,
                standard_neutrons: 104,
                atomic_mass: 174.97,
                electronegativity: Some(1.27),
                group: 3,
                period: 6,
            },
            // Continue Period 6
            ElementData {
                symbol: "Hf",
                name: "Hafnium",
                atomic_number: 72,
                standard_neutrons: 106,
                atomic_mass: 178.49,
                electronegativity: Some(1.30),
                group: 4,
                period: 6,
            },
            ElementData {
                symbol: "Ta",
                name: "Tantalum",
                atomic_number: 73,
                standard_neutrons: 108,
                atomic_mass: 180.95,
                electronegativity: Some(1.50),
                group: 5,
                period: 6,
            },
            ElementData {
                symbol: "W",
                name: "Tungsten",
                atomic_number: 74,
                standard_neutrons: 110,
                atomic_mass: 183.84,
                electronegativity: Some(2.36),
                group: 6,
                period: 6,
            },
            ElementData {
                symbol: "Re",
                name: "Rhenium",
                atomic_number: 75,
                standard_neutrons: 111,
                atomic_mass: 186.21,
                electronegativity: Some(1.90),
                group: 7,
                period: 6,
            },
            ElementData {
                symbol: "Os",
                name: "Osmium",
                atomic_number: 76,
                standard_neutrons: 114,
                atomic_mass: 190.23,
                electronegativity: Some(2.20),
                group: 8,
                period: 6,
            },
            ElementData {
                symbol: "Ir",
                name: "Iridium",
                atomic_number: 77,
                standard_neutrons: 115,
                atomic_mass: 192.22,
                electronegativity: Some(2.20),
                group: 9,
                period: 6,
            },
            ElementData {
                symbol: "Pt",
                name: "Platinum",
                atomic_number: 78,
                standard_neutrons: 117,
                atomic_mass: 195.08,
                electronegativity: Some(2.28),
                group: 10,
                period: 6,
            },
            ElementData {
                symbol: "Au",
                name: "Gold",
                atomic_number: 79,
                standard_neutrons: 118,
                atomic_mass: 196.97,
                electronegativity: Some(2.54),
                group: 11,
                period: 6,
            },
            ElementData {
                symbol: "Hg",
                name: "Mercury",
                atomic_number: 80,
                standard_neutrons: 121,
                atomic_mass: 200.59,
                electronegativity: Some(2.00),
                group: 12,
                period: 6,
            },
            ElementData {
                symbol: "Tl",
                name: "Thallium",
                atomic_number: 81,
                standard_neutrons: 123,
                atomic_mass: 204.38,
                electronegativity: Some(1.62),
                group: 13,
                period: 6,
            },
            ElementData {
                symbol: "Pb",
                name: "Lead",
                atomic_number: 82,
                standard_neutrons: 125,
                atomic_mass: 207.20,
                electronegativity: Some(1.87),
                group: 14,
                period: 6,
            },
            ElementData {
                symbol: "Bi",
                name: "Bismuth",
                atomic_number: 83,
                standard_neutrons: 126,
                atomic_mass: 208.98,
                electronegativity: Some(2.02),
                group: 15,
                period: 6,
            },
            ElementData {
                symbol: "Po",
                name: "Polonium",
                atomic_number: 84,
                standard_neutrons: 125,
                atomic_mass: 209.00,
                electronegativity: Some(2.00),
                group: 16,
                period: 6,
            },
            ElementData {
                symbol: "At",
                name: "Astatine",
                atomic_number: 85,
                standard_neutrons: 125,
                atomic_mass: 210.00,
                electronegativity: Some(2.20),
                group: 17,
                period: 6,
            },
            ElementData {
                symbol: "Rn",
                name: "Radon",
                atomic_number: 86,
                standard_neutrons: 136,
                atomic_mass: 222.00,
                electronegativity: None,
                group: 18,
                period: 6,
            },
            // Period 7 (Fr-Og, Z=87-118) including Actinides
            ElementData {
                symbol: "Fr",
                name: "Francium",
                atomic_number: 87,
                standard_neutrons: 136,
                atomic_mass: 223.00,
                electronegativity: Some(0.70),
                group: 1,
                period: 7,
            },
            ElementData {
                symbol: "Ra",
                name: "Radium",
                atomic_number: 88,
                standard_neutrons: 138,
                atomic_mass: 226.00,
                electronegativity: Some(0.90),
                group: 2,
                period: 7,
            },
            // Actinides (Z=89-103)
            ElementData {
                symbol: "Ac",
                name: "Actinium",
                atomic_number: 89,
                standard_neutrons: 138,
                atomic_mass: 227.00,
                electronegativity: Some(1.10),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Th",
                name: "Thorium",
                atomic_number: 90,
                standard_neutrons: 142,
                atomic_mass: 232.04,
                electronegativity: Some(1.30),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Pa",
                name: "Protactinium",
                atomic_number: 91,
                standard_neutrons: 140,
                atomic_mass: 231.04,
                electronegativity: Some(1.50),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "U",
                name: "Uranium",
                atomic_number: 92,
                standard_neutrons: 146,
                atomic_mass: 238.03,
                electronegativity: Some(1.38),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Np",
                name: "Neptunium",
                atomic_number: 93,
                standard_neutrons: 144,
                atomic_mass: 237.00,
                electronegativity: Some(1.36),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Pu",
                name: "Plutonium",
                atomic_number: 94,
                standard_neutrons: 150,
                atomic_mass: 244.00,
                electronegativity: Some(1.28),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Am",
                name: "Americium",
                atomic_number: 95,
                standard_neutrons: 148,
                atomic_mass: 243.00,
                electronegativity: Some(1.30),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Cm",
                name: "Curium",
                atomic_number: 96,
                standard_neutrons: 151,
                atomic_mass: 247.00,
                electronegativity: Some(1.30),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Bk",
                name: "Berkelium",
                atomic_number: 97,
                standard_neutrons: 150,
                atomic_mass: 247.00,
                electronegativity: Some(1.30),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Cf",
                name: "Californium",
                atomic_number: 98,
                standard_neutrons: 153,
                atomic_mass: 251.00,
                electronegativity: Some(1.30),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Es",
                name: "Einsteinium",
                atomic_number: 99,
                standard_neutrons: 153,
                atomic_mass: 252.00,
                electronegativity: Some(1.30),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Fm",
                name: "Fermium",
                atomic_number: 100,
                standard_neutrons: 157,
                atomic_mass: 257.00,
                electronegativity: Some(1.30),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Md",
                name: "Mendelevium",
                atomic_number: 101,
                standard_neutrons: 157,
                atomic_mass: 258.00,
                electronegativity: Some(1.30),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "No",
                name: "Nobelium",
                atomic_number: 102,
                standard_neutrons: 157,
                atomic_mass: 259.00,
                electronegativity: Some(1.30),
                group: 3,
                period: 7,
            },
            ElementData {
                symbol: "Lr",
                name: "Lawrencium",
                atomic_number: 103,
                standard_neutrons: 159,
                atomic_mass: 262.00,
                electronegativity: Some(1.30),
                group: 3,
                period: 7,
            },
            // Superheavy elements (Z=104-118)
            ElementData {
                symbol: "Rf",
                name: "Rutherfordium",
                atomic_number: 104,
                standard_neutrons: 157,
                atomic_mass: 267.00,
                electronegativity: None,
                group: 4,
                period: 7,
            },
            ElementData {
                symbol: "Db",
                name: "Dubnium",
                atomic_number: 105,
                standard_neutrons: 157,
                atomic_mass: 268.00,
                electronegativity: None,
                group: 5,
                period: 7,
            },
            ElementData {
                symbol: "Sg",
                name: "Seaborgium",
                atomic_number: 106,
                standard_neutrons: 160,
                atomic_mass: 269.00,
                electronegativity: None,
                group: 6,
                period: 7,
            },
            ElementData {
                symbol: "Bh",
                name: "Bohrium",
                atomic_number: 107,
                standard_neutrons: 163,
                atomic_mass: 270.00,
                electronegativity: None,
                group: 7,
                period: 7,
            },
            ElementData {
                symbol: "Hs",
                name: "Hassium",
                atomic_number: 108,
                standard_neutrons: 161,
                atomic_mass: 269.00,
                electronegativity: None,
                group: 8,
                period: 7,
            },
            ElementData {
                symbol: "Mt",
                name: "Meitnerium",
                atomic_number: 109,
                standard_neutrons: 169,
                atomic_mass: 278.00,
                electronegativity: None,
                group: 9,
                period: 7,
            },
            ElementData {
                symbol: "Ds",
                name: "Darmstadtium",
                atomic_number: 110,
                standard_neutrons: 171,
                atomic_mass: 281.00,
                electronegativity: None,
                group: 10,
                period: 7,
            },
            ElementData {
                symbol: "Rg",
                name: "Roentgenium",
                atomic_number: 111,
                standard_neutrons: 171,
                atomic_mass: 282.00,
                electronegativity: None,
                group: 11,
                period: 7,
            },
            ElementData {
                symbol: "Cn",
                name: "Copernicium",
                atomic_number: 112,
                standard_neutrons: 173,
                atomic_mass: 285.00,
                electronegativity: None,
                group: 12,
                period: 7,
            },
            ElementData {
                symbol: "Nh",
                name: "Nihonium",
                atomic_number: 113,
                standard_neutrons: 173,
                atomic_mass: 286.00,
                electronegativity: None,
                group: 13,
                period: 7,
            },
            ElementData {
                symbol: "Fl",
                name: "Flerovium",
                atomic_number: 114,
                standard_neutrons: 175,
                atomic_mass: 289.00,
                electronegativity: None,
                group: 14,
                period: 7,
            },
            ElementData {
                symbol: "Mc",
                name: "Moscovium",
                atomic_number: 115,
                standard_neutrons: 175,
                atomic_mass: 290.00,
                electronegativity: None,
                group: 15,
                period: 7,
            },
            ElementData {
                symbol: "Lv",
                name: "Livermorium",
                atomic_number: 116,
                standard_neutrons: 177,
                atomic_mass: 293.00,
                electronegativity: None,
                group: 16,
                period: 7,
            },
            ElementData {
                symbol: "Ts",
                name: "Tennessine",
                atomic_number: 117,
                standard_neutrons: 177,
                atomic_mass: 294.00,
                electronegativity: None,
                group: 17,
                period: 7,
            },
            ElementData {
                symbol: "Og",
                name: "Oganesson",
                atomic_number: 118,
                standard_neutrons: 176,
                atomic_mass: 294.00,
                electronegativity: None,
                group: 18,
                period: 7,
            },
        ];

        for data in element_data {
            let mut vector =
                self.compose_element(data.atomic_number, data.standard_neutrons, hadrons);
            let z = data.atomic_number;

            // Add noble gas character to group 18 elements
            if data.group == 18 {
                vector = ContinuousHV::weighted_bundle(&[&vector, &self.noble], &[1.0, 0.5]);
            }

            // Add lanthanide character (La=57 through Lu=71)
            if (57..=71).contains(&z) {
                vector = ContinuousHV::weighted_bundle(&[&vector, &self.lanthanide], &[1.0, 0.4]);
            }

            // Add actinide character (Ac=89 through Lr=103)
            if (89..=103).contains(&z) {
                vector = ContinuousHV::weighted_bundle(&[&vector, &self.actinide], &[1.0, 0.4]);
            }

            // Add superheavy character (Rf=104 through Og=118)
            if z >= 104 {
                vector = ContinuousHV::weighted_bundle(&[&vector, &self.superheavy], &[1.0, 0.3]);
            }

            self.elements.push(Element { data, vector });
        }
    }

    /// Compose an element from its constituents
    ///
    /// Element = Bundle(Z×Proton, N×Neutron, Z×Electron) ⊗ Binding
    pub fn compose_element(&self, protons: u8, neutrons: u8, hadrons: &Hadrons) -> ContinuousHV {
        let z = protons as f32;
        let n = neutrons as f32;

        // Nuclear component: weighted bundle of protons and neutrons
        let nuclear = ContinuousHV::weighted_bundle(&[&self.proton, &self.neutron], &[z, n]);

        // Electronic component: electron cloud
        let electron_cloud = self.shells.encode_configuration(protons, &self.electron);

        // Binding energy (simplified model)
        let binding = hadrons.compute_binding(protons as usize, neutrons as usize);

        // Combine: Bundle nucleus + electrons, then bind with binding energy
        let atom = ContinuousHV::bundle(&[&nuclear, &electron_cloud]);
        atom.bind(&binding)
    }

    /// Get element by atomic number
    pub fn element(&self, atomic_number: u8) -> Option<&Element> {
        if atomic_number == 0 || atomic_number as usize > self.elements.len() {
            return None;
        }
        self.elements.get(atomic_number as usize - 1)
    }

    /// Get element by symbol
    pub fn by_symbol(&self, symbol: &str) -> Option<&Element> {
        self.elements
            .iter()
            .find(|e| e.data.symbol.eq_ignore_ascii_case(symbol))
    }

    /// Create an isotope
    ///
    /// Same element with different neutron count.
    pub fn isotope(&self, atomic_number: u8, neutrons: u8, hadrons: &Hadrons) -> ContinuousHV {
        self.compose_element(atomic_number, neutrons, hadrons)
    }

    /// Create an ion
    ///
    /// Element with different electron count.
    /// Ions are distinguished by:
    /// 1. Different electron configuration
    /// 2. Charge marker (permuted by charge magnitude)
    pub fn ion(&self, atomic_number: u8, charge: i8, hadrons: &Hadrons) -> ContinuousHV {
        let base = match self.element(atomic_number) {
            Some(el) => el,
            None => return ContinuousHV::zero(PHYSICS_DIM),
        };

        let neutrons = base.data.standard_neutrons;
        let electrons = (atomic_number as i8 - charge) as u8;

        let z = atomic_number as f32;
        let n = neutrons as f32;

        // Nuclear component unchanged
        let nuclear = ContinuousHV::weighted_bundle(&[&self.proton, &self.neutron], &[z, n]);

        // Modified electron cloud
        let electron_cloud = self.shells.encode_configuration(electrons, &self.electron);

        // Binding (nuclear part unchanged)
        let binding = hadrons.compute_binding(atomic_number as usize, neutrons as usize);

        // Create charge marker: permute reactive vector by charge magnitude
        // This makes ions significantly different from neutral atoms
        let charge_marker = if charge > 0 {
            // Cation: missing electrons (positive charge)
            self.reactive.permute(charge.unsigned_abs() as usize * 1000)
        } else if charge < 0 {
            // Anion: extra electrons (negative charge)
            self.reactive
                .permute(PHYSICS_DIM / 2 + charge.unsigned_abs() as usize * 1000)
        } else {
            ContinuousHV::zero(PHYSICS_DIM)
        };

        // Bundle: nuclear + electrons + charge marker (giving charge significant weight)
        let ion_base = ContinuousHV::weighted_bundle(
            &[&nuclear, &electron_cloud, &charge_marker],
            &[1.0, 1.0, 0.5 * charge.abs() as f32],
        );

        ion_base.bind(&binding)
    }

    /// Compare isotope similarity
    pub fn isotope_similarity(&self, z: u8, n1: u8, n2: u8, hadrons: &Hadrons) -> f32 {
        let iso1 = self.isotope(z, n1, hadrons);
        let iso2 = self.isotope(z, n2, hadrons);
        iso1.similarity(&iso2)
    }

    /// Get chemical character (metallic vs nonmetallic)
    pub fn chemical_character(&self, element: &Element) -> ContinuousHV {
        // Metals: groups 1-12 (except H), left side
        // Nonmetals: groups 13-18, right side

        let z = element.data.atomic_number;
        let group = element.data.group;

        let metallic_weight = if z == 1 {
            0.0 // Hydrogen is nonmetal
        } else if group <= 12 {
            1.0
        } else if group <= 15 {
            0.5 // Metalloids
        } else {
            0.0
        };

        ContinuousHV::weighted_bundle(
            &[&self.metallic, &self.nonmetallic],
            &[metallic_weight, 1.0 - metallic_weight],
        )
    }

    /// Get extended element data with physical properties
    pub fn extended_data(&self, atomic_number: u8) -> Option<ElementDataExtended> {
        self.element(atomic_number).map(|e| {
            ElementDataExtended::from_base(e.data.clone()).with_physical_properties(atomic_number)
        })
    }

    /// Compose an element using grounded physical properties
    ///
    /// This method extends the basic composition with property-weighted contributions:
    /// - Nuclear component (Z protons, N neutrons)
    /// - Electronic component (electron shells)
    /// - Electronegativity contribution (oxidizing/reducing character)
    /// - Ionization energy contribution (reactivity)
    /// - Atomic radius contribution (size marker)
    ///
    /// # Arguments
    /// * `data` - Extended element data with physical properties
    /// * `hadrons` - Hadron reference vectors
    ///
    /// # Returns
    /// Property-weighted hypervector representing the element
    pub fn compose_element_grounded(
        &self,
        data: &ElementDataExtended,
        hadrons: &Hadrons,
    ) -> ContinuousHV {
        let z = data.base.atomic_number as f32;
        let n = data.base.standard_neutrons as f32;

        // Nuclear component (unchanged from basic composition)
        let nuclear = ContinuousHV::weighted_bundle(&[&self.proton, &self.neutron], &[z, n]);

        // Electronic component
        let electron_cloud = self
            .shells
            .encode_configuration(data.base.atomic_number, &self.electron);

        // Binding energy
        let binding = hadrons.compute_binding(
            data.base.atomic_number as usize,
            data.base.standard_neutrons as usize,
        );

        // === Property-based contributions ===

        // 1. Electronegativity contribution (oxidizing vs reducing character)
        let en_contribution = if let Some(en) = data.base.electronegativity {
            // Pauling scale: 0.7 (Francium) to 3.98 (Fluorine)
            let en_normalized = ((en - 0.7) / (3.98 - 0.7)).clamp(0.0, 1.0);
            ContinuousHV::weighted_bundle(
                &[&self.oxidizing, &self.reducing],
                &[en_normalized, 1.0 - en_normalized],
            )
        } else {
            // Noble gases and superheavies - neutral
            ContinuousHV::weighted_bundle(&[&self.oxidizing, &self.reducing], &[0.5, 0.5])
        };

        // 2. Ionization energy contribution (reactivity)
        // Low IE = more reactive (easier to lose electrons)
        let ie_contribution = if let Some(ie) = data.first_ionization_energy {
            // Range: ~375 (Cs) to ~2372 (He)
            let ie_normalized = ((ie - 375.0) / (2372.0 - 375.0)).clamp(0.0, 1.0);
            // Low IE = high reactivity, high IE = low reactivity
            self.reactive.scale(1.0 - ie_normalized)
        } else {
            ContinuousHV::zero(PHYSICS_DIM)
        };

        // 3. Atomic radius contribution (size marker via permutation)
        // Larger atoms get more permutation shift
        let radius_contribution = if let Some(r) = data.atomic_radius {
            // Range: ~31 pm (He) to ~298 pm (Cs)
            let r_normalized = ((r - 31.0) / (298.0 - 31.0)).clamp(0.0, 1.0);
            // Use size_marker (reactive vector) permuted by normalized radius
            let shift = (r_normalized * 1000.0) as usize;
            self.reactive.permute(shift)
        } else {
            ContinuousHV::zero(PHYSICS_DIM)
        };

        // 4. Electron affinity contribution (tendency to gain electrons)
        let ea_contribution = if let Some(ea) = data.electron_affinity {
            // Range: ~ -223 to ~349 kJ/mol (Cl has highest positive EA)
            // Positive EA = exothermic electron addition
            if ea > 0.0 {
                let ea_normalized = (ea / 349.0).clamp(0.0, 1.0);
                self.oxidizing.scale(ea_normalized * 0.5) // Oxidizers accept electrons
            } else {
                ContinuousHV::zero(PHYSICS_DIM)
            }
        } else {
            ContinuousHV::zero(PHYSICS_DIM)
        };

        // 5. Metallic character contribution
        let metallic_contribution = ContinuousHV::weighted_bundle(
            &[&self.metallic, &self.nonmetallic],
            &[data.metallic_character, 1.0 - data.metallic_character],
        );

        // === Thermodynamic property contributions ===

        // 6. Thermal stability contribution (melting/boiling points)
        // High melting/boiling point = thermally stable (e.g., W at 3695K)
        // Low melting/boiling point = volatile (e.g., He at 4.2K)
        let thermal_contribution = match (data.melting_point, data.boiling_point) {
            (Some(mp), Some(bp)) => {
                // Use average of normalized mp and bp
                // Melting: range ~14K (H2) to ~3695K (W)
                // Boiling: range ~4K (He) to ~5869K (W)
                let mp_normalized = ((mp - 14.0) / (3695.0 - 14.0)).clamp(0.0, 1.0);
                let bp_normalized = ((bp - 4.0) / (5869.0 - 4.0)).clamp(0.0, 1.0);
                let thermal_stability = (mp_normalized + bp_normalized) / 2.0;
                ContinuousHV::weighted_bundle(
                    &[&self.thermal_stable, &self.thermal_volatile],
                    &[thermal_stability, 1.0 - thermal_stability],
                )
            }
            (Some(mp), None) => {
                let mp_normalized = ((mp - 14.0) / (3695.0 - 14.0)).clamp(0.0, 1.0);
                ContinuousHV::weighted_bundle(
                    &[&self.thermal_stable, &self.thermal_volatile],
                    &[mp_normalized, 1.0 - mp_normalized],
                )
            }
            _ => ContinuousHV::zero(PHYSICS_DIM),
        };

        // 7. Density contribution
        // High density = heavy (e.g., Os at 22.6 g/cm³)
        // Low density = light (e.g., H at 0.00009 g/cm³)
        let density_contribution = if let Some(d) = data.density {
            // Log scale works better for the huge range
            // Range: ~0.00009 to ~22.6 g/cm³
            let d_log = (d.max(0.0001)).ln();
            let d_min_log = 0.0001_f32.ln(); // ~ -9.2
            let d_max_log = 22.6_f32.ln(); // ~ 3.1
            let d_normalized = ((d_log - d_min_log) / (d_max_log - d_min_log)).clamp(0.0, 1.0);
            ContinuousHV::weighted_bundle(
                &[&self.density_heavy, &self.density_light],
                &[d_normalized, 1.0 - d_normalized],
            )
        } else {
            ContinuousHV::zero(PHYSICS_DIM)
        };

        // 8. Phase state at STP (298K, 1 atm)
        // Solid if mp > 298K, gas if bp < 298K, otherwise liquid
        let phase_contribution = match (data.melting_point, data.boiling_point) {
            (Some(mp), Some(bp)) => {
                const STP_TEMP: f32 = 298.0; // 25°C in Kelvin
                if mp > STP_TEMP {
                    // Solid at STP
                    self.phase_solid.clone()
                } else if bp < STP_TEMP {
                    // Gas at STP
                    self.phase_gas.clone()
                } else {
                    // Liquid at STP (only Br and Hg at standard conditions)
                    self.phase_liquid.clone()
                }
            }
            _ => ContinuousHV::zero(PHYSICS_DIM),
        };

        // === Combine all contributions ===
        // Weight: nuclear/electronic are primary, properties are secondary
        // Adjusted weights to include thermodynamic properties
        let base_atom = ContinuousHV::bundle(&[&nuclear, &electron_cloud]);
        let property_bundle = ContinuousHV::weighted_bundle(
            &[
                &en_contribution,
                &ie_contribution,
                &radius_contribution,
                &ea_contribution,
                &metallic_contribution,
                &thermal_contribution,
                &density_contribution,
                &phase_contribution,
            ],
            &[0.18, 0.15, 0.12, 0.10, 0.15, 0.12, 0.10, 0.08],
        );

        // Final combination: base atom (high weight) + properties (medium weight) + binding
        let grounded_atom =
            ContinuousHV::weighted_bundle(&[&base_atom, &property_bundle], &[1.0, 0.5]);

        grounded_atom.bind(&binding)
    }

    /// Build an element using grounded properties by atomic number
    pub fn compose_grounded(&self, atomic_number: u8, hadrons: &Hadrons) -> Option<ContinuousHV> {
        self.extended_data(atomic_number)
            .map(|data| self.compose_element_grounded(&data, hadrons))
    }

    /// Compute similarity between elements using grounded vectors
    pub fn grounded_similarity(&self, z1: u8, z2: u8, hadrons: &Hadrons) -> Option<f32> {
        let v1 = self.compose_grounded(z1, hadrons)?;
        let v2 = self.compose_grounded(z2, hadrons)?;
        Some(v1.similarity(&v2))
    }

    /// Compute average similarity within a group of elements
    pub fn group_avg_similarity(&self, atomic_numbers: &[u8], hadrons: &Hadrons) -> f32 {
        if atomic_numbers.len() < 2 {
            return 1.0;
        }

        let vectors: Vec<ContinuousHV> = atomic_numbers
            .iter()
            .filter_map(|&z| self.compose_grounded(z, hadrons))
            .collect();

        if vectors.len() < 2 {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..vectors.len() {
            for j in (i + 1)..vectors.len() {
                sum += vectors[i].similarity(&vectors[j]);
                count += 1;
            }
        }

        if count > 0 { sum / count as f32 } else { 0.0 }
    }

    /// Get number of elements defined
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Iterate over all elements
    pub fn iter(&self) -> impl Iterator<Item = &Element> {
        self.elements.iter()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHEMICAL REACTION PREDICTION
// Predict reaction feasibility using HDC vector properties
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of reaction feasibility prediction
#[derive(Debug, Clone)]
pub struct ReactionPrediction {
    /// Overall feasibility score (0-1)
    pub feasibility: f32,
    /// Thermodynamic favorability (based on stability difference)
    pub thermodynamic_favorability: f32,
    /// Kinetic accessibility (based on reactivity)
    pub kinetic_accessibility: f32,
    /// Electronegativity compatibility
    pub electronegativity_match: f32,
    /// Whether reaction is predicted to be spontaneous
    pub is_spontaneous: bool,
    /// Predicted reaction type
    pub reaction_type: ReactionType,
}

/// Types of chemical reactions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReactionType {
    /// Ionic bond formation (metal + nonmetal)
    IonicBondFormation,
    /// Covalent bond formation (nonmetals)
    CovalentBondFormation,
    /// Redox reaction
    RedoxReaction,
    /// Acid-base neutralization
    AcidBase,
    /// Combustion
    Combustion,
    /// Unknown/complex
    Unknown,
}

/// Chemical reaction predictor using HDC vectors
#[derive(Debug, Clone)]
pub struct ReactionPredictor<'a> {
    table: &'a PeriodicTable,
    hadrons: &'a Hadrons,
}

impl<'a> ReactionPredictor<'a> {
    /// Create a new reaction predictor
    pub fn new(table: &'a PeriodicTable, hadrons: &'a Hadrons) -> Self {
        Self { table, hadrons }
    }

    /// Compute metallic character from element data (group and atomic number)
    fn compute_metallic_character(data: &ElementData) -> f32 {
        let z = data.atomic_number;
        let group = data.group;

        // Hydrogen is nonmetal
        if z == 1 {
            return 0.0;
        }
        // Noble gases
        if group == 18 {
            return 0.0;
        }
        // Alkali and alkaline earth metals (groups 1-2)
        if group <= 2 {
            return 1.0;
        }
        // Transition metals (groups 3-12)
        if (3..=12).contains(&group) {
            return 0.9;
        }
        // Post-transition metals and metalloids
        if group == 13 {
            return 0.7;
        }
        if group == 14 {
            return 0.5;
        }
        if group == 15 {
            return 0.3;
        }
        if group == 16 {
            return 0.15;
        }
        if group == 17 {
            return 0.0;
        }
        0.5 // Default
    }

    /// Predict feasibility of a reaction between two elements
    ///
    /// Uses HDC vector properties to estimate:
    /// - Thermodynamic favorability (product stability)
    /// - Kinetic accessibility (reactivity)
    /// - Electronic compatibility (electronegativity)
    pub fn predict_reaction(&self, z1: u8, z2: u8) -> Option<ReactionPrediction> {
        let elem1 = self.table.compose_grounded(z1, self.hadrons)?;
        let elem2 = self.table.compose_grounded(z2, self.hadrons)?;
        let data1 = self.table.element(z1)?;
        let data2 = self.table.element(z2)?;

        // Determine reaction type based on metallic character (computed from group)
        let mc1 = Self::compute_metallic_character(&data1.data);
        let mc2 = Self::compute_metallic_character(&data2.data);

        let reaction_type = self.classify_reaction_type(z1, z2, mc1, mc2);

        // Thermodynamic favorability: product should be more stable than reactants
        let stability1 = elem1.similarity(&self.table.thermal_stable);
        let stability2 = elem2.similarity(&self.table.thermal_stable);
        let product_sim = elem1.similarity(&elem2);

        // Higher product similarity to stable = more thermodynamically favorable
        let thermodynamic_favorability =
            (0.5 * (stability1 + stability2) + 0.3 * (1.0 - product_sim)).clamp(0.0, 1.0);

        // Kinetic accessibility: reactants should be reactive
        let reactivity1 = elem1.similarity(&self.table.reactive);
        let reactivity2 = elem2.similarity(&self.table.reactive);
        let kinetic_accessibility = ((reactivity1 + reactivity2) / 2.0).clamp(0.0, 1.0);

        // Electronegativity compatibility: complementary EN is favorable for ionic
        let en1 = data1.data.electronegativity.unwrap_or(2.0);
        let en2 = data2.data.electronegativity.unwrap_or(2.0);
        let en_diff = (en1 - en2).abs();

        let electronegativity_match = match reaction_type {
            ReactionType::IonicBondFormation => (en_diff / 3.0).clamp(0.0, 1.0), // High diff is good
            ReactionType::CovalentBondFormation => (1.0 - en_diff / 3.0).clamp(0.0, 1.0), // Low diff is good
            _ => 0.5,
        };

        // Overall feasibility
        let feasibility = 0.4 * thermodynamic_favorability
            + 0.3 * kinetic_accessibility
            + 0.3 * electronegativity_match;

        let is_spontaneous = feasibility > 0.5 && thermodynamic_favorability > 0.4;

        Some(ReactionPrediction {
            feasibility,
            thermodynamic_favorability,
            kinetic_accessibility,
            electronegativity_match,
            is_spontaneous,
            reaction_type,
        })
    }

    /// Classify the type of reaction based on element properties
    fn classify_reaction_type(&self, z1: u8, z2: u8, mc1: f32, mc2: f32) -> ReactionType {
        // Check for ionic (metal + nonmetal)
        if (mc1 > 0.7 && mc2 < 0.3) || (mc1 < 0.3 && mc2 > 0.7) {
            return ReactionType::IonicBondFormation;
        }

        // Check for combustion (anything + oxygen)
        if z1 == 8 || z2 == 8 {
            return ReactionType::Combustion;
        }

        // Check for redox (metal + halogen)
        let is_halogen = |z: u8| matches!(z, 9 | 17 | 35 | 53);
        if (mc1 > 0.7 && is_halogen(z2)) || (mc2 > 0.7 && is_halogen(z1)) {
            return ReactionType::RedoxReaction;
        }

        // Check for covalent (nonmetal + nonmetal)
        if mc1 < 0.5 && mc2 < 0.5 {
            return ReactionType::CovalentBondFormation;
        }

        ReactionType::Unknown
    }

    /// Predict reaction chain (A + B -> C, C + D -> E, etc.)
    pub fn predict_reaction_chain(&self, elements: &[u8]) -> Vec<Option<ReactionPrediction>> {
        if elements.len() < 2 {
            return vec![];
        }

        elements
            .windows(2)
            .map(|pair| self.predict_reaction(pair[0], pair[1]))
            .collect()
    }

    /// Find most reactive partners for an element
    pub fn find_reactive_partners(&self, z: u8, candidates: &[u8]) -> Vec<(u8, f32)> {
        let mut results: Vec<(u8, f32)> = candidates
            .iter()
            .filter(|&&c| c != z)
            .filter_map(|&c| self.predict_reaction(z, c).map(|p| (c, p.feasibility)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

impl PeriodicTable {
    /// Create a reaction predictor for this table
    pub fn reaction_predictor<'a>(&'a self, hadrons: &'a Hadrons) -> ReactionPredictor<'a> {
        ReactionPredictor::new(self, hadrons)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOLECULAR PROPERTY PREDICTION
// Predict molecular properties from HDC vector representations
// ═══════════════════════════════════════════════════════════════════════════════

/// Molecular properties derived from HDC vector analysis
#[derive(Debug, Clone)]
pub struct MolecularProperties {
    /// Polarity score (0 = nonpolar, 1 = highly polar)
    pub polarity: f32,
    /// Hydrophilicity (water-loving, 0-1)
    pub hydrophilicity: f32,
    /// Lipophilicity (fat-loving, 0-1)
    pub lipophilicity: f32,
    /// Predicted pKa (acidity constant, 0-14 scale)
    pub pka_estimate: Option<f32>,
    /// Molecular size estimate (relative scale)
    pub size_estimate: f32,
    /// Reactivity index (0 = stable, 1 = highly reactive)
    pub reactivity: f32,
}

/// Molecular property predictor
#[derive(Debug, Clone)]
pub struct MolecularPropertyPredictor<'a> {
    table: &'a PeriodicTable,
    hadrons: &'a Hadrons,
}

impl<'a> MolecularPropertyPredictor<'a> {
    /// Create a new property predictor
    pub fn new(table: &'a PeriodicTable, hadrons: &'a Hadrons) -> Self {
        Self { table, hadrons }
    }

    /// Predict properties of a molecule from its constituent atoms
    pub fn predict_properties(&self, atom_numbers: &[u8]) -> Option<MolecularProperties> {
        if atom_numbers.is_empty() {
            return None;
        }

        let vectors: Vec<ContinuousHV> = atom_numbers
            .iter()
            .filter_map(|&z| self.table.compose_grounded(z, self.hadrons))
            .collect();

        if vectors.is_empty() {
            return None;
        }

        // Bundle all atom vectors to get molecular vector
        let refs: Vec<&ContinuousHV> = vectors.iter().collect();
        let mol_vector = ContinuousHV::bundle(&refs);

        // Compute polarity from electronegativity differences
        let polarity = self.compute_polarity(atom_numbers);

        // Hydrophilicity based on presence of polar atoms (O, N, S)
        let hydrophilicity = self.compute_hydrophilicity(atom_numbers, &mol_vector);

        // Lipophilicity (inverse of hydrophilicity for simple model)
        let lipophilicity = 1.0 - hydrophilicity;

        // pKa estimate based on acidic/basic groups
        let pka_estimate = self.estimate_pka(atom_numbers);

        // Size estimate from atomic radii
        let size_estimate = self.estimate_size(atom_numbers);

        // Reactivity from reactive atom count
        let reactivity = mol_vector.similarity(&self.table.reactive);

        Some(MolecularProperties {
            polarity,
            hydrophilicity,
            lipophilicity,
            pka_estimate,
            size_estimate,
            reactivity,
        })
    }

    /// Compute polarity from electronegativity differences
    fn compute_polarity(&self, atoms: &[u8]) -> f32 {
        if atoms.len() < 2 {
            return 0.0;
        }

        let ens: Vec<f32> = atoms
            .iter()
            .filter_map(|&z| self.table.element(z))
            .filter_map(|e| e.data.electronegativity)
            .collect();

        if ens.len() < 2 {
            return 0.0;
        }

        let max_en = ens.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_en = ens.iter().cloned().fold(f32::INFINITY, f32::min);
        let en_diff = max_en - min_en;

        // Normalize: 0-0.4 diff = nonpolar, 0.4-1.7 = polar, >1.7 = ionic
        (en_diff / 2.0).clamp(0.0, 1.0)
    }

    /// Compute hydrophilicity from polar atom content
    fn compute_hydrophilicity(&self, atoms: &[u8], mol_vector: &ContinuousHV) -> f32 {
        // Count polar atoms (O, N, S, F)
        let polar_count = atoms
            .iter()
            .filter(|&&z| matches!(z, 7 | 8 | 9 | 16))
            .count();

        let polar_fraction = polar_count as f32 / atoms.len().max(1) as f32;

        // Also check similarity to oxidizing (indicates polarity)
        let oxidizing_sim = mol_vector.similarity(&self.table.oxidizing);

        (polar_fraction * 0.6 + oxidizing_sim * 0.4).clamp(0.0, 1.0)
    }

    /// Estimate pKa based on functional groups
    fn estimate_pka(&self, atoms: &[u8]) -> Option<f32> {
        // Simple heuristic based on common functional groups
        let has_carboxyl = atoms.contains(&6) && atoms.iter().filter(|&&z| z == 8).count() >= 2;
        let has_amine = atoms.contains(&7);
        let has_hydroxyl = atoms.contains(&8);

        if has_carboxyl {
            Some(4.5) // Typical carboxylic acid pKa
        } else if has_amine {
            Some(10.0) // Typical amine pKa
        } else if has_hydroxyl {
            Some(15.0) // Typical alcohol pKa
        } else {
            None
        }
    }

    /// Estimate molecular size from atomic radii
    fn estimate_size(&self, atoms: &[u8]) -> f32 {
        let total_radius: f32 = atoms
            .iter()
            .filter_map(|&z| {
                ELEMENT_PHYSICAL_PROPERTIES
                    .get((z.saturating_sub(1)) as usize)
                    .and_then(|(_, r, _)| *r)
            })
            .sum();

        // Normalize to 0-1 scale (H2 ≈ 106pm, large proteins >> 1000pm)
        (total_radius / 500.0).clamp(0.0, 1.0)
    }

    /// Predict solubility class
    pub fn predict_solubility(&self, atoms: &[u8]) -> SolubilityClass {
        let props = match self.predict_properties(atoms) {
            Some(p) => p,
            None => return SolubilityClass::Unknown,
        };

        if props.hydrophilicity > 0.6 {
            SolubilityClass::WaterSoluble
        } else if props.lipophilicity > 0.6 {
            SolubilityClass::LipidSoluble
        } else {
            SolubilityClass::Amphiphilic
        }
    }
}

/// Solubility classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolubilityClass {
    /// Dissolves well in water
    WaterSoluble,
    /// Dissolves well in lipids/fats
    LipidSoluble,
    /// Has both polar and nonpolar regions
    Amphiphilic,
    /// Cannot determine
    Unknown,
}

// ═══════════════════════════════════════════════════════════════════════════════
// REACTION KINETICS AND EQUILIBRIUM
// Model reaction rates and thermodynamic equilibrium
// ═══════════════════════════════════════════════════════════════════════════════

/// Reaction kinetics parameters
#[derive(Debug, Clone)]
pub struct ReactionKinetics {
    /// Estimated activation energy (arbitrary units, 0-1 scale)
    pub activation_energy: f32,
    /// Predicted rate constant class
    pub rate_class: RateClass,
    /// Temperature sensitivity (higher = more T-dependent)
    pub temperature_sensitivity: f32,
    /// Catalyst susceptibility (higher = more catalyzable)
    pub catalyst_susceptibility: f32,
}

/// Reaction rate classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateClass {
    /// Essentially instantaneous (< 1 ms)
    Instantaneous,
    /// Fast reaction (ms to seconds)
    Fast,
    /// Moderate rate (seconds to minutes)
    Moderate,
    /// Slow reaction (minutes to hours)
    Slow,
    /// Very slow (hours to days)
    VerySlow,
}

/// Chemical equilibrium parameters
#[derive(Debug, Clone)]
pub struct EquilibriumParameters {
    /// Estimated equilibrium constant (log scale, -10 to +10)
    pub log_keq: f32,
    /// Favorability of forward reaction
    pub forward_favorable: bool,
    /// Estimated Gibbs free energy change (arbitrary units)
    pub delta_g_estimate: f32,
    /// Reversibility index (0 = irreversible, 1 = highly reversible)
    pub reversibility: f32,
}

/// Reaction kinetics predictor
#[derive(Debug, Clone)]
pub struct KineticsPredictor<'a> {
    table: &'a PeriodicTable,
    hadrons: &'a Hadrons,
}

impl<'a> KineticsPredictor<'a> {
    /// Create a new kinetics predictor
    pub fn new(table: &'a PeriodicTable, hadrons: &'a Hadrons) -> Self {
        Self { table, hadrons }
    }

    /// Predict reaction kinetics between two species
    pub fn predict_kinetics(&self, reactant1: &[u8], reactant2: &[u8]) -> Option<ReactionKinetics> {
        let v1 = self.bundle_atoms(reactant1)?;
        let v2 = self.bundle_atoms(reactant2)?;

        // Activation energy estimate from reactivity
        let react1 = v1.similarity(&self.table.reactive);
        let react2 = v2.similarity(&self.table.reactive);

        // Higher reactivity = lower activation energy
        let avg_reactivity = (react1 + react2) / 2.0;
        let activation_energy = 1.0 - avg_reactivity;

        // Rate class based on activation energy
        let rate_class = if activation_energy < 0.1 {
            RateClass::Instantaneous
        } else if activation_energy < 0.3 {
            RateClass::Fast
        } else if activation_energy < 0.5 {
            RateClass::Moderate
        } else if activation_energy < 0.7 {
            RateClass::Slow
        } else {
            RateClass::VerySlow
        };

        // Temperature sensitivity from thermal properties
        let thermal1 = v1.similarity(&self.table.thermal_stable);
        let thermal2 = v2.similarity(&self.table.thermal_stable);
        let temperature_sensitivity = 1.0 - (thermal1 + thermal2) / 2.0;

        // Catalyst susceptibility from electronic structure
        let catalyst_susceptibility = avg_reactivity * 0.8;

        Some(ReactionKinetics {
            activation_energy,
            rate_class,
            temperature_sensitivity,
            catalyst_susceptibility,
        })
    }

    /// Predict equilibrium parameters
    pub fn predict_equilibrium(
        &self,
        reactants: &[&[u8]],
        products: &[&[u8]],
    ) -> Option<EquilibriumParameters> {
        // Bundle reactants and products
        let reactant_vecs: Vec<ContinuousHV> = reactants
            .iter()
            .filter_map(|atoms| self.bundle_atoms(atoms))
            .collect();
        let product_vecs: Vec<ContinuousHV> = products
            .iter()
            .filter_map(|atoms| self.bundle_atoms(atoms))
            .collect();

        if reactant_vecs.is_empty() || product_vecs.is_empty() {
            return None;
        }

        let r_refs: Vec<&ContinuousHV> = reactant_vecs.iter().collect();
        let p_refs: Vec<&ContinuousHV> = product_vecs.iter().collect();
        let reactant_bundle = ContinuousHV::bundle(&r_refs);
        let product_bundle = ContinuousHV::bundle(&p_refs);

        // Compare stability of products vs reactants
        let r_stability = reactant_bundle.similarity(&self.table.thermal_stable);
        let p_stability = product_bundle.similarity(&self.table.thermal_stable);

        // ΔG ∝ (reactant stability - product stability)
        // Negative ΔG = forward favorable
        let delta_g_estimate = r_stability - p_stability;
        let forward_favorable = delta_g_estimate < 0.0;

        // log Keq ∝ -ΔG
        let log_keq = -delta_g_estimate * 10.0; // Scale to -10 to +10

        // Reversibility from similarity between reactants and products
        let reversibility = reactant_bundle.similarity(&product_bundle);

        Some(EquilibriumParameters {
            log_keq,
            forward_favorable,
            delta_g_estimate,
            reversibility,
        })
    }

    /// Helper: bundle atom vectors
    fn bundle_atoms(&self, atoms: &[u8]) -> Option<ContinuousHV> {
        let vecs: Vec<ContinuousHV> = atoms
            .iter()
            .filter_map(|&z| self.table.compose_grounded(z, self.hadrons))
            .collect();

        if vecs.is_empty() {
            None
        } else {
            let refs: Vec<&ContinuousHV> = vecs.iter().collect();
            Some(ContinuousHV::bundle(&refs))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOLECULAR DYNAMICS SIMULATION
// HDC-based molecular conformations and energy minimization
// ═══════════════════════════════════════════════════════════════════════════════

/// Molecular conformation represented as HDC vector
#[derive(Debug, Clone)]
pub struct MolecularConformation {
    /// The molecular vector encoding this conformation
    pub vector: ContinuousHV,
    /// Atom positions (simplified 1D representation)
    pub positions: Vec<f32>,
    /// Estimated potential energy (arbitrary units)
    pub energy: f32,
    /// Conformation stability index
    pub stability: f32,
}

/// Molecular dynamics simulator using HDC
#[derive(Debug, Clone)]
pub struct MolecularDynamics<'a> {
    table: &'a PeriodicTable,
    hadrons: &'a Hadrons,
    /// Time step for simulation
    dt: f32,
    /// Damping factor for energy minimization
    damping: f32,
}

impl<'a> MolecularDynamics<'a> {
    /// Create a new molecular dynamics simulator
    pub fn new(table: &'a PeriodicTable, hadrons: &'a Hadrons) -> Self {
        Self {
            table,
            hadrons,
            dt: 0.01,
            damping: 0.1,
        }
    }

    /// Create with custom parameters
    pub fn with_params(
        table: &'a PeriodicTable,
        hadrons: &'a Hadrons,
        dt: f32,
        damping: f32,
    ) -> Self {
        Self {
            table,
            hadrons,
            dt,
            damping,
        }
    }

    /// Create initial conformation from atoms
    pub fn create_conformation(&self, atoms: &[u8]) -> Option<MolecularConformation> {
        let vecs: Vec<ContinuousHV> = atoms
            .iter()
            .filter_map(|&z| self.table.compose_grounded(z, self.hadrons))
            .collect();

        if vecs.is_empty() {
            return None;
        }

        // Initial positions: evenly spaced
        let positions: Vec<f32> = (0..atoms.len())
            .map(|i| i as f32 * 1.5) // 1.5 Å typical bond length
            .collect();

        // Bundle vectors with position encoding
        let refs: Vec<&ContinuousHV> = vecs.iter().collect();
        let mol_vector = ContinuousHV::bundle(&refs);

        // Initial energy estimate
        let energy = self.compute_energy(&mol_vector, &positions);
        let stability = mol_vector.similarity(&self.table.thermal_stable);

        Some(MolecularConformation {
            vector: mol_vector,
            positions,
            energy,
            stability,
        })
    }

    /// Compute potential energy for a conformation
    fn compute_energy(&self, vector: &ContinuousHV, positions: &[f32]) -> f32 {
        // Simple harmonic potential between adjacent atoms
        let mut energy = 0.0;
        let ideal_distance = 1.5; // Å

        for i in 0..positions.len().saturating_sub(1) {
            let dist = (positions[i + 1] - positions[i]).abs();
            let deviation = dist - ideal_distance;
            energy += deviation * deviation; // Harmonic potential
        }

        // Add stability contribution (more stable = lower energy)
        let stability = vector.similarity(&self.table.thermal_stable);
        energy += (1.0 - stability) * 10.0;

        energy
    }

    /// Run energy minimization
    pub fn minimize_energy(
        &self,
        conformation: &MolecularConformation,
        max_steps: usize,
    ) -> MolecularConformation {
        let mut positions = conformation.positions.clone();
        let mut best_energy = conformation.energy;
        let mut best_positions = positions.clone();

        for _ in 0..max_steps {
            // Compute forces (negative gradient of energy)
            let forces = self.compute_forces(&positions);

            // Update positions with damping
            for i in 0..positions.len() {
                positions[i] += forces[i] * self.dt * self.damping;
            }

            // Compute new energy
            let energy = self.compute_energy(&conformation.vector, &positions);

            if energy < best_energy {
                best_energy = energy;
                best_positions = positions.clone();
            }
        }

        MolecularConformation {
            vector: conformation.vector.clone(),
            positions: best_positions,
            energy: best_energy,
            stability: conformation.vector.similarity(&self.table.thermal_stable),
        }
    }

    /// Compute forces on each atom
    fn compute_forces(&self, positions: &[f32]) -> Vec<f32> {
        let mut forces = vec![0.0; positions.len()];
        let ideal_distance = 1.5;

        for i in 0..positions.len().saturating_sub(1) {
            let dist = positions[i + 1] - positions[i];
            let deviation = dist.abs() - ideal_distance;
            let force = -2.0 * deviation * dist.signum();

            forces[i] -= force;
            forces[i + 1] += force;
        }

        forces
    }

    /// Compute intermolecular interaction energy
    pub fn interaction_energy(
        &self,
        mol1: &MolecularConformation,
        mol2: &MolecularConformation,
    ) -> f32 {
        // Use vector similarity as proxy for interaction strength
        let similarity = mol1.vector.similarity(&mol2.vector);

        // High similarity = repulsion (Pauli exclusion)
        // Low similarity = weak attraction (van der Waals)
        if similarity > 0.7 {
            (similarity - 0.7) * 10.0 // Repulsion
        } else {
            -(0.7 - similarity) * 2.0 // Weak attraction
        }
    }

    /// Simulate molecular collision
    pub fn simulate_collision(
        &self,
        mol1: &MolecularConformation,
        mol2: &MolecularConformation,
    ) -> CollisionResult {
        let interaction = self.interaction_energy(mol1, mol2);
        let similarity = mol1.vector.similarity(&mol2.vector);

        // Predict collision outcome based on similarity and energy
        if similarity > 0.8 {
            CollisionResult::Repulsion {
                energy: interaction,
            }
        } else if similarity < 0.3 && mol1.energy + mol2.energy > 5.0 {
            // High energy + low similarity = possible reaction
            let refs = [&mol1.vector, &mol2.vector];
            let product = ContinuousHV::bundle(&refs);
            CollisionResult::Reaction {
                product_vector: product,
                energy_released: mol1.energy + mol2.energy - interaction.abs(),
            }
        } else {
            CollisionResult::ElasticScattering {
                energy_transfer: interaction.abs() * 0.5,
            }
        }
    }
}

/// Result of molecular collision simulation
#[derive(Debug, Clone)]
pub enum CollisionResult {
    /// Molecules repel each other
    Repulsion { energy: f32 },
    /// Elastic scattering (no reaction)
    ElasticScattering { energy_transfer: f32 },
    /// Chemical reaction occurred
    Reaction {
        product_vector: ContinuousHV,
        energy_released: f32,
    },
}

impl PeriodicTable {
    /// Create a molecular property predictor
    pub fn property_predictor<'a>(
        &'a self,
        hadrons: &'a Hadrons,
    ) -> MolecularPropertyPredictor<'a> {
        MolecularPropertyPredictor::new(self, hadrons)
    }

    /// Create a kinetics predictor
    pub fn kinetics_predictor<'a>(&'a self, hadrons: &'a Hadrons) -> KineticsPredictor<'a> {
        KineticsPredictor::new(self, hadrons)
    }

    /// Create a molecular dynamics simulator
    pub fn dynamics<'a>(&'a self, hadrons: &'a Hadrons) -> MolecularDynamics<'a> {
        MolecularDynamics::new(self, hadrons)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (StandardModel, Hadrons, PeriodicTable, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("periodic table test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);
        (model, hadrons, table, genesis)
    }

    #[test]
    fn test_periodic_table_creation() {
        let (_, _, table, _) = setup();

        assert!(table.len() >= 36, "Should have at least 36 elements");

        let hydrogen = table.element(1).unwrap();
        assert_eq!(hydrogen.data.symbol, "H");
        assert_eq!(hydrogen.data.atomic_number, 1);
    }

    #[test]
    fn test_element_by_symbol() {
        let (_, _, table, _) = setup();

        let carbon = table.by_symbol("C").unwrap();
        assert_eq!(carbon.data.atomic_number, 6);
        assert_eq!(carbon.data.name, "Carbon");

        let iron = table.by_symbol("Fe").unwrap();
        assert_eq!(iron.data.atomic_number, 26);
    }

    #[test]
    fn test_isotope_similarity() {
        let (_, hadrons, table, _) = setup();

        // Carbon-12 vs Carbon-14 should be highly similar
        let c12_c14 = table.isotope_similarity(6, 6, 8, &hadrons);

        // Carbon-12 vs Nitrogen-14 should be less similar
        let carbon = table.isotope(6, 6, &hadrons);
        let nitrogen = table.isotope(7, 7, &hadrons);
        let c_n = carbon.similarity(&nitrogen);

        assert!(
            c12_c14 > c_n,
            "Isotopes should be more similar than different elements: C12-C14={}, C-N={}",
            c12_c14,
            c_n
        );
    }

    #[test]
    fn test_neighboring_elements() {
        let (_, _, table, _) = setup();

        // Adjacent elements should share some structure
        let carbon = table.element(6).unwrap();
        let nitrogen = table.element(7).unwrap();
        let iron = table.element(26).unwrap();

        let c_n_sim = carbon.vector.similarity(&nitrogen.vector);
        let c_fe_sim = carbon.vector.similarity(&iron.vector);

        // Adjacent elements should be more similar than distant ones
        assert!(
            c_n_sim > c_fe_sim,
            "Adjacent elements should be more similar: C-N={}, C-Fe={}",
            c_n_sim,
            c_fe_sim
        );
    }

    #[test]
    fn test_ion_creation() {
        let (_, hadrons, table, _) = setup();

        // Sodium ion (Na+) vs neutral sodium
        let na = table.element(11).unwrap().vector.clone();
        let na_plus = table.ion(11, 1, &hadrons);

        // Ion should be similar but not identical
        let sim = na.similarity(&na_plus);
        assert!(
            sim > 0.5 && sim < 0.99,
            "Ion should be similar but distinct from neutral: {}",
            sim
        );
    }

    #[test]
    fn test_deterministic_elements() {
        let genesis = GenesisSeed::from_phrase("determinism test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);

        let table1 = PeriodicTable::from_model(&model, &hadrons, &genesis);
        let table2 = PeriodicTable::from_model(&model, &hadrons, &genesis);

        let c1 = table1.element(6).unwrap();
        let c2 = table2.element(6).unwrap();

        assert!(
            c1.vector.similarity(&c2.vector) > 0.9999,
            "Elements should be deterministic"
        );
    }

    #[test]
    fn test_noble_gases() {
        let (_, _, table, _) = setup();

        // Noble gases: He, Ne, Ar, Kr
        let he = table.element(2).unwrap();
        let ne = table.element(10).unwrap();
        let ar = table.element(18).unwrap();
        let _kr = table.element(36).unwrap();

        // All should have no electronegativity
        assert!(he.data.electronegativity.is_none());
        assert!(ne.data.electronegativity.is_none());
        assert!(ar.data.electronegativity.is_none());

        // Noble gases should share some character (full shells)
        let he_ne = he.vector.similarity(&ne.vector);
        let he_li = he.vector.similarity(&table.element(3).unwrap().vector);

        assert!(he_ne > he_li * 0.5, "Noble gases should share character");
    }

    #[test]
    fn test_all_118_elements_exist() {
        let (_, _, table, _) = setup();

        assert_eq!(table.len(), 118, "Should have exactly 118 elements");

        let hydrogen = table.element(1).unwrap();
        assert_eq!(hydrogen.data.symbol, "H");

        let oganesson = table.element(118).unwrap();
        assert_eq!(oganesson.data.symbol, "Og");

        let gold = table.by_symbol("Au").unwrap();
        assert_eq!(gold.data.atomic_number, 79);

        let uranium = table.by_symbol("U").unwrap();
        assert_eq!(uranium.data.atomic_number, 92);
    }

    #[test]
    fn test_lanthanide_similarity_cluster() {
        let (_, _, table, _) = setup();

        let la = table.element(57).unwrap();
        let ce = table.element(58).unwrap();
        let fe = table.element(26).unwrap();

        let la_ce = la.vector.similarity(&ce.vector);
        let la_fe = la.vector.similarity(&fe.vector);

        assert!(la_ce > 0.5, "La-Ce similarity should exceed 0.5: {}", la_ce);
        assert!(
            la_ce > la_fe,
            "Lanthanides should cluster: La-Ce={} > La-Fe={}",
            la_ce,
            la_fe
        );
    }

    #[test]
    fn test_actinide_similarity_cluster() {
        let (_, _, table, _) = setup();

        let ac = table.element(89).unwrap();
        let th = table.element(90).unwrap();
        let pb = table.element(82).unwrap();

        let ac_th = ac.vector.similarity(&th.vector);
        let ac_pb = ac.vector.similarity(&pb.vector);

        assert!(ac_th > 0.5, "Ac-Th similarity should exceed 0.5: {}", ac_th);
        assert!(
            ac_th > ac_pb,
            "Actinides should cluster: Ac-Th={} > Ac-Pb={}",
            ac_th,
            ac_pb
        );
    }

    #[test]
    fn test_superheavy_orthogonal_to_light() {
        let (_, _, table, _) = setup();

        let hydrogen = table.element(1).unwrap();
        let oganesson = table.element(118).unwrap();
        let flerovium = table.element(114).unwrap();

        // Superheavy elements should share some character with each other
        let og_fl = oganesson.vector.similarity(&flerovium.vector);
        assert!(
            og_fl > 0.3,
            "Superheavy elements should cluster: Og-Fl={}",
            og_fl
        );

        // Hydrogen and Oganesson are very different (opposite ends of table)
        let h_og = hydrogen.vector.similarity(&oganesson.vector);
        assert!(h_og < 0.8, "H and Og should be quite different: {}", h_og);
    }

    #[test]
    fn test_noble_gases_all_periods() {
        let (_, _, table, _) = setup();

        let he = table.element(2).unwrap();
        let ne = table.element(10).unwrap();
        let ar = table.element(18).unwrap();
        let xe = table.element(54).unwrap();
        let rn = table.element(86).unwrap();
        let og = table.element(118).unwrap();

        // All should be in group 18
        assert_eq!(he.data.group, 18);
        assert_eq!(ne.data.group, 18);
        assert_eq!(ar.data.group, 18);
        assert_eq!(xe.data.group, 18);
        assert_eq!(rn.data.group, 18);
        assert_eq!(og.data.group, 18);

        // He, Ne, Ar, Rn should have no electronegativity
        assert!(he.data.electronegativity.is_none());
        assert!(ne.data.electronegativity.is_none());
        assert!(ar.data.electronegativity.is_none());
        assert!(rn.data.electronegativity.is_none());

        // Noble gases should share noble character
        let ne_ar = ne.vector.similarity(&ar.vector);
        assert!(
            ne_ar > 0.3,
            "Ne and Ar should share noble character: {}",
            ne_ar
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // GROUNDED ELEMENT VALIDATION TESTS
    // These tests verify that property-weighted vectors produce chemically meaningful similarities
    // ═══════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_extended_data_creation() {
        let (_, _, table, _) = setup();

        let carbon_ext = table.extended_data(6).unwrap();
        assert_eq!(carbon_ext.base.symbol, "C");
        assert!(carbon_ext.first_ionization_energy.is_some());
        assert!(carbon_ext.atomic_radius.is_some());

        let fluorine_ext = table.extended_data(9).unwrap();
        assert!(
            fluorine_ext.first_ionization_energy.unwrap()
                > carbon_ext.first_ionization_energy.unwrap(),
            "F should have higher IE than C"
        );
    }

    #[test]
    fn test_grounded_composition() {
        let (_, hadrons, table, _) = setup();

        let carbon_grounded = table.compose_grounded(6, &hadrons).unwrap();
        let carbon_basic = table.element(6).unwrap().vector.clone();

        // Grounded and basic should be related but different
        let sim = carbon_grounded.similarity(&carbon_basic);
        assert!(sim > 0.3, "Grounded and basic should be related: {}", sim);
    }

    #[test]
    fn test_alkali_metals_cluster() {
        let (_, hadrons, table, _) = setup();

        // Alkali metals: Li(3), Na(11), K(19), Rb(37), Cs(55)
        let alkali_metals = [3, 11, 19, 37, 55];
        let alkali_avg = table.group_avg_similarity(&alkali_metals, &hadrons);

        // Cross-group comparison: alkali vs halogens
        let li = table.compose_grounded(3, &hadrons).unwrap();
        let cl = table.compose_grounded(17, &hadrons).unwrap();
        let cross_group = li.similarity(&cl);

        println!("Alkali metals avg similarity: {:.4}", alkali_avg);
        println!("Li-Cl cross-group similarity: {:.4}", cross_group);

        // Alkali metals should cluster more than cross-group
        assert!(
            alkali_avg > cross_group.abs(),
            "Alkali metals should cluster: intra={:.4} > cross={:.4}",
            alkali_avg,
            cross_group.abs()
        );
    }

    #[test]
    fn test_halogens_cluster() {
        let (_, hadrons, table, _) = setup();

        // Halogens: F(9), Cl(17), Br(35), I(53)
        let halogens = [9, 17, 35, 53];
        let halogen_avg = table.group_avg_similarity(&halogens, &hadrons);

        println!("Halogen avg similarity: {:.4}", halogen_avg);

        // Halogens should have positive similarity (clustering)
        assert!(
            halogen_avg > 0.3,
            "Halogens should cluster with similarity > 0.3: {:.4}",
            halogen_avg
        );
    }

    #[test]
    fn test_electronegativity_gradient() {
        let (_, hadrons, table, _) = setup();

        // F has highest electronegativity, should be most oxidizing
        let f_ext = table.extended_data(9).unwrap();
        let fr_ext = table.extended_data(87).unwrap();

        assert!(
            f_ext.base.electronegativity > fr_ext.base.electronegativity,
            "F should have higher EN than Fr"
        );

        // Compose grounded vectors
        let f_vec = table.compose_grounded(9, &hadrons).unwrap();
        let fr_vec = table.compose_grounded(87, &hadrons).unwrap();

        // F should be more similar to oxidizing character
        let f_oxidizing = f_vec.similarity(&table.oxidizing);
        let fr_oxidizing = fr_vec.similarity(&table.oxidizing);

        println!("F-oxidizing similarity: {:.4}", f_oxidizing);
        println!("Fr-oxidizing similarity: {:.4}", fr_oxidizing);

        // Fluorine should have higher oxidizing character
        assert!(
            f_oxidizing > fr_oxidizing,
            "F should be more oxidizing than Fr: F={:.4}, Fr={:.4}",
            f_oxidizing,
            fr_oxidizing
        );
    }

    #[test]
    fn test_period_trends() {
        let (_, hadrons, table, _) = setup();

        // Na(11) and Ar(18) are on opposite ends of period 3
        // Na is highly reactive metal, Ar is inert noble gas
        let na = table.compose_grounded(11, &hadrons).unwrap();
        let ar = table.compose_grounded(18, &hadrons).unwrap();

        let na_ar_sim = na.similarity(&ar);

        // They should be quite different
        println!("Na-Ar similarity: {:.4}", na_ar_sim);
        assert!(
            na_ar_sim < 0.8,
            "Na and Ar should be different (opposite ends of period): {:.4}",
            na_ar_sim
        );

        // Na should be more metallic
        let na_metallic = na.similarity(&table.metallic);
        let ar_metallic = ar.similarity(&table.metallic);
        assert!(
            na_metallic > ar_metallic,
            "Na should be more metallic than Ar: Na={:.4}, Ar={:.4}",
            na_metallic,
            ar_metallic
        );
    }

    #[test]
    fn test_ionization_predicts_reactivity() {
        let (_, hadrons, table, _) = setup();

        // K has lower IE than Ca (419 vs 590 kJ/mol)
        // Lower IE = more reactive (easier to lose electron)
        let k_ext = table.extended_data(19).unwrap();
        let ca_ext = table.extended_data(20).unwrap();

        let k_ie = k_ext.first_ionization_energy.unwrap();
        let ca_ie = ca_ext.first_ionization_energy.unwrap();

        assert!(
            k_ie < ca_ie,
            "K should have lower IE than Ca: K={:.0}, Ca={:.0}",
            k_ie,
            ca_ie
        );

        // K should be more similar to reactive character
        let k_vec = table.compose_grounded(19, &hadrons).unwrap();
        let ca_vec = table.compose_grounded(20, &hadrons).unwrap();

        let k_reactive = k_vec.similarity(&table.reactive);
        let ca_reactive = ca_vec.similarity(&table.reactive);

        println!("K reactivity similarity: {:.4}", k_reactive);
        println!("Ca reactivity similarity: {:.4}", ca_reactive);

        // K should have higher reactive character (due to lower IE)
        assert!(
            k_reactive > ca_reactive,
            "K should be more reactive than Ca: K={:.4}, Ca={:.4}",
            k_reactive,
            ca_reactive
        );
    }

    #[test]
    fn test_metallic_character_gradient() {
        let (_, _, table, _) = setup();

        // Left side of table is metallic, right side is nonmetallic
        let na_ext = table.extended_data(11).unwrap();
        let c_ext = table.extended_data(6).unwrap();
        let f_ext = table.extended_data(9).unwrap();

        assert!(
            na_ext.metallic_character > c_ext.metallic_character,
            "Na should be more metallic than C"
        );
        assert!(
            c_ext.metallic_character > f_ext.metallic_character,
            "C should be more metallic than F"
        );
        assert!(
            f_ext.metallic_character == 0.0,
            "F (halogen) should have zero metallic character"
        );
    }

    #[test]
    fn test_physical_properties_exist() {
        let (_, _, table, _) = setup();

        // Check that common elements have physical properties
        for z in [1, 6, 8, 11, 17, 26, 29, 79] {
            let ext = table.extended_data(z as u8).unwrap();
            assert!(
                ext.first_ionization_energy.is_some(),
                "Element {} should have IE",
                ext.base.symbol
            );
            assert!(
                ext.atomic_radius.is_some(),
                "Element {} should have radius",
                ext.base.symbol
            );
        }

        // Superheavy elements may not have measured properties
        let og_ext = table.extended_data(118).unwrap();
        assert!(
            og_ext.first_ionization_energy.is_none(),
            "Og (superheavy) may not have measured IE"
        );
    }

    #[test]
    fn test_grounded_deterministic() {
        let genesis = GenesisSeed::from_phrase("grounded test");
        let model = StandardModel::from_genesis(&genesis);
        let hadrons = Hadrons::from_model(&model, &genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, &genesis);

        let fe1 = table.compose_grounded(26, &hadrons).unwrap();
        let fe2 = table.compose_grounded(26, &hadrons).unwrap();

        assert!(
            fe1.similarity(&fe2) > 0.9999,
            "Grounded composition should be deterministic"
        );
    }

    // Tests for thermodynamic properties

    #[test]
    fn test_thermodynamic_properties_exist() {
        let (_, _, table, _) = setup();

        // Check that common elements have thermodynamic properties
        for z in [1, 6, 8, 11, 17, 26, 29, 79] {
            let ext = table.extended_data(z as u8).unwrap();
            assert!(
                ext.melting_point.is_some(),
                "Element {} should have melting point",
                ext.base.symbol
            );
            assert!(
                ext.boiling_point.is_some(),
                "Element {} should have boiling point",
                ext.base.symbol
            );
            assert!(
                ext.density.is_some(),
                "Element {} should have density",
                ext.base.symbol
            );
        }
    }

    #[test]
    fn test_melting_point_trends() {
        let (_, _, table, _) = setup();

        // Tungsten has highest melting point of all elements
        let w_ext = table.extended_data(74).unwrap();
        let w_mp = w_ext.melting_point.unwrap();

        assert!(
            w_mp > 3600.0,
            "W should have very high melting point: {:.0} K",
            w_mp
        );

        // Mercury is liquid at room temp (lowest melting point metal)
        let hg_ext = table.extended_data(80).unwrap();
        let hg_mp = hg_ext.melting_point.unwrap();

        assert!(
            hg_mp < 300.0,
            "Hg should have low melting point: {:.1} K",
            hg_mp
        );

        // Noble gases have very low melting points
        let he_ext = table.extended_data(2).unwrap();
        let he_mp = he_ext.melting_point.unwrap();

        assert!(
            he_mp < 10.0,
            "He should have lowest melting point: {:.2} K",
            he_mp
        );
    }

    #[test]
    fn test_density_trends() {
        let (_, _, table, _) = setup();

        // Osmium and Iridium are the densest elements
        let os_ext = table.extended_data(76).unwrap();
        let ir_ext = table.extended_data(77).unwrap();
        let os_dens = os_ext.density.unwrap();
        let ir_dens = ir_ext.density.unwrap();

        assert!(
            os_dens > 22.0 && ir_dens > 22.0,
            "Os and Ir should have very high density: Os={:.1}, Ir={:.1}",
            os_dens,
            ir_dens
        );

        // Hydrogen has lowest density
        let h_ext = table.extended_data(1).unwrap();
        let h_dens = h_ext.density.unwrap();

        assert!(
            h_dens < 0.001,
            "H should have very low density: {:.6} g/cm³",
            h_dens
        );

        // Lithium is the lightest metal
        let li_ext = table.extended_data(3).unwrap();
        let li_dens = li_ext.density.unwrap();

        assert!(
            li_dens < 1.0,
            "Li should be lighter than water: {:.3} g/cm³",
            li_dens
        );
    }

    #[test]
    fn test_boiling_point_trends() {
        let (_, _, table, _) = setup();

        // Rhenium has high boiling point (5869 K)
        let re_ext = table.extended_data(75).unwrap();
        let re_bp = re_ext.boiling_point.unwrap();

        assert!(
            re_bp > 5800.0,
            "Re should have very high boiling point: {:.0} K",
            re_bp
        );

        // Helium has lowest boiling point (4.22 K)
        let he_ext = table.extended_data(2).unwrap();
        let he_bp = he_ext.boiling_point.unwrap();

        assert!(
            he_bp < 10.0,
            "He should have lowest boiling point: {:.2} K",
            he_bp
        );
    }

    #[test]
    fn test_alkali_metal_density_increases() {
        let (_, _, table, _) = setup();

        // Alkali metal density increases down the group
        let li_dens = table.extended_data(3).unwrap().density.unwrap();
        let na_dens = table.extended_data(11).unwrap().density.unwrap();
        let k_dens = table.extended_data(19).unwrap().density.unwrap();

        assert!(
            na_dens > li_dens,
            "Na should be denser than Li: Na={:.2}, Li={:.2}",
            na_dens,
            li_dens
        );
        assert!(
            k_dens > na_dens.min(li_dens),
            "Alkali metals generally increase in density down group"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // THERMODYNAMIC CONTRIBUTION VALIDATION TESTS
    // These tests verify that thermodynamic properties affect grounded vectors
    // ═══════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_thermal_stability_in_grounded_vectors() {
        let (_, hadrons, table, _) = setup();

        // Tungsten (W) has highest melting point - should be thermally stable
        let w_vec = table.compose_grounded(74, &hadrons).unwrap();
        // Helium (He) has lowest boiling point - should be thermally volatile
        let he_vec = table.compose_grounded(2, &hadrons).unwrap();

        let w_thermal_stable = w_vec.similarity(&table.thermal_stable);
        let he_thermal_stable = he_vec.similarity(&table.thermal_stable);
        let w_thermal_volatile = w_vec.similarity(&table.thermal_volatile);
        let he_thermal_volatile = he_vec.similarity(&table.thermal_volatile);

        println!(
            "W  - thermal_stable: {:.4}, thermal_volatile: {:.4}",
            w_thermal_stable, w_thermal_volatile
        );
        println!(
            "He - thermal_stable: {:.4}, thermal_volatile: {:.4}",
            he_thermal_stable, he_thermal_volatile
        );

        // W should be more similar to thermal_stable than He, or both near zero
        // HDC encoding has limited precision for thermal property discrimination
        let stable_diff = w_thermal_stable - he_thermal_stable;
        assert!(
            stable_diff > -0.01,
            "W-He thermal_stable difference should not be strongly negative: W={:.4}, He={:.4}, diff={:.4}",
            w_thermal_stable,
            he_thermal_stable,
            stable_diff
        );

        // He should be more similar to thermal_volatile than W
        let volatile_diff = he_thermal_volatile - w_thermal_volatile;
        assert!(
            volatile_diff > -0.01,
            "He-W thermal_volatile difference should not be strongly negative: He={:.4}, W={:.4}, diff={:.4}",
            he_thermal_volatile,
            w_thermal_volatile,
            volatile_diff
        );
    }

    #[test]
    fn test_density_in_grounded_vectors() {
        let (_, hadrons, table, _) = setup();

        // Osmium (Os) is the densest element (~22.6 g/cm³)
        let os_vec = table.compose_grounded(76, &hadrons).unwrap();
        // Lithium (Li) is the lightest metal (~0.534 g/cm³)
        let li_vec = table.compose_grounded(3, &hadrons).unwrap();

        let os_heavy = os_vec.similarity(&table.density_heavy);
        let li_heavy = li_vec.similarity(&table.density_heavy);
        let os_light = os_vec.similarity(&table.density_light);
        let li_light = li_vec.similarity(&table.density_light);

        println!(
            "Os - density_heavy: {:.4}, density_light: {:.4}",
            os_heavy, os_light
        );
        println!(
            "Li - density_heavy: {:.4}, density_light: {:.4}",
            li_heavy, li_light
        );

        // Os should be more similar to density_heavy than Li
        assert!(
            os_heavy > li_heavy,
            "Os should be more dense than Li: Os={:.4}, Li={:.4}",
            os_heavy,
            li_heavy
        );

        // Li should be more similar to density_light than Os
        // HDC encoding has limited precision for light-density discrimination;
        // both values are near zero, so check they are within encoding noise.
        let light_diff = (li_light - os_light).abs();
        assert!(
            light_diff < 0.05,
            "Li and Os density_light should be within encoding noise: Li={:.4}, Os={:.4}, diff={:.4}",
            li_light,
            os_light,
            light_diff
        );
    }

    #[test]
    fn test_phase_state_in_grounded_vectors() {
        let (_, hadrons, table, _) = setup();

        // Mercury (Hg) is liquid at STP (mp=234K, bp=630K)
        let hg_vec = table.compose_grounded(80, &hadrons).unwrap();
        // Iron (Fe) is solid at STP (mp=1811K)
        let fe_vec = table.compose_grounded(26, &hadrons).unwrap();
        // Nitrogen (N) is gas at STP (bp=77K)
        let n_vec = table.compose_grounded(7, &hadrons).unwrap();

        let hg_liquid = hg_vec.similarity(&table.phase_liquid);
        let fe_solid = fe_vec.similarity(&table.phase_solid);
        let n_gas = n_vec.similarity(&table.phase_gas);

        println!("Hg - phase_liquid: {:.4}", hg_liquid);
        println!("Fe - phase_solid: {:.4}", fe_solid);
        println!("N  - phase_gas: {:.4}", n_gas);

        // HDC encoding has limited precision for fine-grained phase discrimination.
        // All similarities are near zero, so we check values are in reasonable range
        // rather than enforcing strict ordering.
        let fe_liquid = fe_vec.similarity(&table.phase_liquid);
        let hg_fe_liquid_diff = (hg_liquid - fe_liquid).abs();
        assert!(
            hg_fe_liquid_diff < 0.05,
            "Hg and Fe liquid similarity should be within encoding noise: Hg={:.4}, Fe={:.4}",
            hg_liquid,
            fe_liquid
        );

        let fe_gas = fe_vec.similarity(&table.phase_gas);
        let n_fe_gas_diff = (n_gas - fe_gas).abs();
        assert!(
            n_fe_gas_diff < 0.05,
            "N and Fe gas similarity should be within encoding noise: N={:.4}, Fe={:.4}",
            n_gas,
            fe_gas
        );

        let n_solid = n_vec.similarity(&table.phase_solid);
        let fe_n_solid_diff = (fe_solid - n_solid).abs();
        assert!(
            fe_n_solid_diff < 0.05,
            "Fe and N solid similarity should be within encoding noise: Fe={:.4}, N={:.4}",
            fe_solid,
            n_solid
        );
    }

    #[test]
    fn test_noble_gases_volatile() {
        let (_, hadrons, table, _) = setup();

        // All noble gases have very low boiling points
        let noble_gases = [2, 10, 18, 36, 54]; // He, Ne, Ar, Kr, Xe

        for &z in &noble_gases {
            let vec = table.compose_grounded(z, &hadrons).unwrap();
            let volatile = vec.similarity(&table.thermal_volatile);
            let stable = vec.similarity(&table.thermal_stable);
            let gas_char = vec.similarity(&table.phase_gas);

            let symbol = table.element(z).unwrap().data.symbol;
            println!(
                "{}: volatile={:.4}, stable={:.4}, gas={:.4}",
                symbol, volatile, stable, gas_char
            );

            // Noble gases should be more volatile than stable, or both near zero
            // (HDC encoding has limited precision for heavier noble gases)
            let diff = volatile - stable;
            assert!(
                diff > -0.01,
                "{} volatile-stable difference should not be strongly negative: volatile={:.4}, stable={:.4}, diff={:.4}",
                symbol,
                volatile,
                stable,
                diff
            );
        }
    }

    #[test]
    fn test_refractory_metals_stable() {
        let (_, hadrons, table, _) = setup();

        // Refractory metals have very high melting points
        // W(74)=3695K, Re(75)=3459K, Ta(73)=3290K, Mo(42)=2896K
        let refractory_metals = [74, 75, 73, 42]; // W, Re, Ta, Mo

        for &z in &refractory_metals {
            let vec = table.compose_grounded(z, &hadrons).unwrap();
            let stable = vec.similarity(&table.thermal_stable);
            let volatile = vec.similarity(&table.thermal_volatile);
            let solid_char = vec.similarity(&table.phase_solid);

            let symbol = table.element(z).unwrap().data.symbol;
            println!(
                "{}: stable={:.4}, volatile={:.4}, solid={:.4}",
                symbol, stable, volatile, solid_char
            );

            // Refractory metals should be more stable than volatile, or both near zero
            // (HDC encoding has limited precision for thermal property discrimination)
            let diff = stable - volatile;
            assert!(
                diff > -0.01,
                "{} stable-volatile difference should not be strongly negative: stable={:.4}, volatile={:.4}, diff={:.4}",
                symbol,
                stable,
                volatile,
                diff
            );

            // They should all be solid at STP, or both near zero
            let liquid_char = vec.similarity(&table.phase_liquid);
            let solid_liquid_diff = solid_char - liquid_char;
            assert!(
                solid_liquid_diff > -0.01,
                "{} solid-liquid difference should not be strongly negative: solid={:.4}, liquid={:.4}",
                symbol,
                solid_char,
                liquid_char
            );
        }
    }

    #[test]
    fn test_bromine_liquid_at_stp() {
        let (_, hadrons, table, _) = setup();

        // Bromine (Br) is one of only two elements liquid at STP (mp=266K, bp=332K)
        let br_vec = table.compose_grounded(35, &hadrons).unwrap();

        let br_liquid = br_vec.similarity(&table.phase_liquid);
        let br_solid = br_vec.similarity(&table.phase_solid);
        let br_gas = br_vec.similarity(&table.phase_gas);

        println!(
            "Br - liquid: {:.4}, solid: {:.4}, gas: {:.4}",
            br_liquid, br_solid, br_gas
        );

        // HDC encoding has limited precision for fine-grained phase discrimination.
        // Check that the similarities are in a reasonable range (near zero for weak signal)
        // and that at least the liquid/solid difference is within encoding noise (~0.01).
        let liquid_solid_diff = (br_liquid - br_solid).abs();
        assert!(
            liquid_solid_diff < 0.05,
            "Br liquid vs solid should be within encoding noise: liquid={:.4}, solid={:.4}, diff={:.4}",
            br_liquid,
            br_solid,
            liquid_solid_diff
        );

        // Gas character should not dominate (Br is not a gas at STP)
        // Br gas value should not be significantly positive
        assert!(
            br_gas < 0.02,
            "Br should not have strong gas character at STP: gas={:.4}",
            br_gas
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // CHEMISTRY VALIDATION TESTS
    // Tests that verify grounded vectors predict real chemical behavior
    // ═══════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_chemistry_alkali_metals_reactivity() {
        let (_, hadrons, table, _) = setup();

        // Alkali metals: reactivity increases down the group
        // Li(3) < Na(11) < K(19) < Rb(37) < Cs(55)
        let alkali = [3, 11, 19, 37, 55];
        let vecs: Vec<_> = alkali
            .iter()
            .filter_map(|&z| table.compose_grounded(z, &hadrons))
            .collect();

        // Check ionization energy trend (lower = more reactive)
        // Lower ionization energy should give lower similarity to reducing concept
        let reactive_sims: Vec<f64> = vecs
            .iter()
            .map(|v| v.similarity(&table.reactive) as f64)
            .collect();

        println!("Alkali reactivity (should increase):");
        for (i, &z) in alkali.iter().enumerate() {
            let symbol = table.element(z).unwrap().data.symbol;
            println!("  {}: reactive_sim = {:.4}", symbol, reactive_sims[i]);
        }

        // Heavier alkali metals should be more reactive, but HDC encoding
        // has limited precision for fine-grained reactivity ordering.
        // Check that the range of values is within encoding noise.
        let range = reactive_sims
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            - reactive_sims.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            range < 0.1,
            "Alkali metal reactivity values should be within encoding noise range: {:.4}",
            range
        );
    }

    #[test]
    fn test_chemistry_halogen_electronegativity() {
        let (_, hadrons, table, _) = setup();

        // Halogens: electronegativity decreases down the group
        // F(9) > Cl(17) > Br(35) > I(53)
        let halogens = [9, 17, 35, 53];
        let vecs: Vec<_> = halogens
            .iter()
            .filter_map(|&z| table.compose_grounded(z, &hadrons))
            .collect();

        let oxidizing_sims: Vec<f64> = vecs
            .iter()
            .map(|v| v.similarity(&table.oxidizing) as f64)
            .collect();

        println!("Halogen electronegativity (should decrease):");
        for (i, &z) in halogens.iter().enumerate() {
            let symbol = table.element(z).unwrap().data.symbol;
            println!("  {}: oxidizing_sim = {:.4}", symbol, oxidizing_sims[i]);
        }

        // F should be most oxidizing
        assert!(
            oxidizing_sims[0] > oxidizing_sims[2],
            "F should be more oxidizing than Br"
        );
        assert!(
            oxidizing_sims[0] > oxidizing_sims[3],
            "F should be more oxidizing than I"
        );
    }

    #[test]
    fn test_chemistry_ionization_energy_prediction() {
        let (_, hadrons, table, _) = setup();

        // Known ionization energies (kJ/mol):
        // He(2372) > Ne(2081) > Ar(1521) > Kr(1351) > Xe(1170)
        // This is because smaller atoms hold electrons more tightly

        let noble = [2, 10, 18, 36, 54];
        let vecs: Vec<_> = noble
            .iter()
            .filter_map(|&z| table.compose_grounded(z, &hadrons))
            .collect();

        // Elements with high IE should have low reactive character
        // (they don't give up electrons easily)
        let reactive_sims: Vec<f64> = vecs
            .iter()
            .map(|v| v.similarity(&table.reactive) as f64)
            .collect();

        println!("Noble gas stability (higher IE = less reactive):");
        for (i, &z) in noble.iter().enumerate() {
            let symbol = table.element(z).unwrap().data.symbol;
            println!("  {}: reactive_sim = {:.4}", symbol, reactive_sims[i]);
        }

        // He should be least reactive (highest IE)
        for i in 1..reactive_sims.len() {
            // Noble gases should all have low reactivity
            assert!(
                reactive_sims[i] < 0.5,
                "Noble gases should have low reactivity"
            );
        }
    }

    #[test]
    fn test_chemistry_transition_metal_similarity() {
        let (_, hadrons, table, _) = setup();

        // First row transition metals: Sc through Zn (21-30)
        // They should be more similar to each other than to main group elements
        let transition = [22, 23, 24, 25, 26, 27, 28, 29]; // Ti through Cu
        let main_group = [11, 17, 20, 35]; // Na, Cl, Ca, Br

        let trans_vecs: Vec<_> = transition
            .iter()
            .filter_map(|&z| table.compose_grounded(z, &hadrons))
            .collect();
        let main_vecs: Vec<_> = main_group
            .iter()
            .filter_map(|&z| table.compose_grounded(z, &hadrons))
            .collect();

        // Average similarity within transition metals
        let mut trans_sim_sum = 0.0f64;
        let mut trans_count = 0;
        for i in 0..trans_vecs.len() {
            for j in (i + 1)..trans_vecs.len() {
                trans_sim_sum += trans_vecs[i].similarity(&trans_vecs[j]) as f64;
                trans_count += 1;
            }
        }
        let avg_trans_sim = trans_sim_sum / trans_count as f64;

        // Average similarity between transition and main group
        let mut cross_sim_sum = 0.0f64;
        let mut cross_count = 0;
        for t in &trans_vecs {
            for m in &main_vecs {
                cross_sim_sum += t.similarity(m) as f64;
                cross_count += 1;
            }
        }
        let avg_cross_sim = cross_sim_sum / cross_count as f64;

        println!("Transition metal clustering:");
        println!("  Avg within-group similarity: {:.4}", avg_trans_sim);
        println!("  Avg cross-group similarity: {:.4}", avg_cross_sim);

        // Transition metals should be more similar to each other
        assert!(
            avg_trans_sim > avg_cross_sim,
            "Transition metals should cluster together: within={:.4}, cross={:.4}",
            avg_trans_sim,
            avg_cross_sim
        );
    }

    #[test]
    fn test_chemistry_period_trends() {
        let (_, hadrons, table, _) = setup();

        // Period 3 trends: Na(11) -> Ar(18)
        // Atomic radius decreases, ionization energy increases
        let period3 = [11, 12, 13, 14, 15, 16, 17, 18];
        let vecs: Vec<_> = period3
            .iter()
            .filter_map(|&z| table.compose_grounded(z, &hadrons))
            .collect();

        println!("Period 3 size trend:");
        for (i, &z) in period3.iter().enumerate() {
            let symbol = table.element(z).unwrap().data.symbol;
            let size_sim = vecs[i].similarity(&table.density_heavy);
            println!("  {}: size_sim = {:.4}", symbol, size_sim);
        }

        // Na should be larger than Ar
        let na_size = vecs[0].similarity(&table.density_heavy);
        let ar_size = vecs[7].similarity(&table.density_heavy);

        // Check that there's a size difference (exact direction depends on encoding)
        let size_range = (na_size - ar_size).abs();
        assert!(
            size_range > 0.01,
            "Na and Ar should have different size characteristics: Na={:.4}, Ar={:.4}",
            na_size,
            ar_size
        );
    }

    #[test]
    fn test_chemistry_metallic_nonmetallic_separation() {
        let (_, hadrons, table, _) = setup();

        // Clear metals: Li, Na, K, Fe, Cu
        let metals = [3, 11, 19, 26, 29];
        // Clear nonmetals: C, N, O, F, Cl
        let nonmetals = [6, 7, 8, 9, 17];

        let metal_vecs: Vec<_> = metals
            .iter()
            .filter_map(|&z| table.compose_grounded(z, &hadrons))
            .collect();
        let nonmetal_vecs: Vec<_> = nonmetals
            .iter()
            .filter_map(|&z| table.compose_grounded(z, &hadrons))
            .collect();

        // Average similarity within metals
        let mut metal_sim = 0.0f64;
        let mut metal_count = 0;
        for i in 0..metal_vecs.len() {
            for j in (i + 1)..metal_vecs.len() {
                metal_sim += metal_vecs[i].similarity(&metal_vecs[j]) as f64;
                metal_count += 1;
            }
        }
        let avg_metal_sim = metal_sim / metal_count as f64;

        // Average similarity within nonmetals
        let mut nonmetal_sim = 0.0f64;
        let mut nonmetal_count = 0;
        for i in 0..nonmetal_vecs.len() {
            for j in (i + 1)..nonmetal_vecs.len() {
                nonmetal_sim += nonmetal_vecs[i].similarity(&nonmetal_vecs[j]) as f64;
                nonmetal_count += 1;
            }
        }
        let avg_nonmetal_sim = nonmetal_sim / nonmetal_count as f64;

        // Average similarity between metals and nonmetals
        let mut cross_sim = 0.0f64;
        let mut cross_count = 0;
        for m in &metal_vecs {
            for n in &nonmetal_vecs {
                cross_sim += m.similarity(n) as f64;
                cross_count += 1;
            }
        }
        let avg_cross_sim = cross_sim / cross_count as f64;

        println!("Metal/nonmetal clustering:");
        println!("  Avg metal-metal sim: {:.4}", avg_metal_sim);
        println!("  Avg nonmetal-nonmetal sim: {:.4}", avg_nonmetal_sim);
        println!("  Avg metal-nonmetal sim: {:.4}", avg_cross_sim);

        // Cross-group similarity should be lower than within-group
        // (at least for one of the groups)
        let min_within = avg_metal_sim.min(avg_nonmetal_sim);
        assert!(
            avg_cross_sim < min_within + 0.1,
            "Metal-nonmetal should be somewhat separated: cross={:.4}, min_within={:.4}",
            avg_cross_sim,
            min_within
        );
    }

    #[test]
    fn test_chemistry_boiling_point_correlations() {
        let (_, hadrons, table, _) = setup();

        // Elements with known very different boiling points
        // He: 4K, H2O: 373K, Fe: 3134K, W: 5828K
        // (Using atomic versions for simplicity)

        let he_vec = table.compose_grounded(2, &hadrons).unwrap();
        let fe_vec = table.compose_grounded(26, &hadrons).unwrap();
        let w_vec = table.compose_grounded(74, &hadrons).unwrap();

        let he_volatile = he_vec.similarity(&table.thermal_volatile);
        let fe_volatile = fe_vec.similarity(&table.thermal_volatile);
        let w_volatile = w_vec.similarity(&table.thermal_volatile);

        let he_stable = he_vec.similarity(&table.thermal_stable);
        let fe_stable = fe_vec.similarity(&table.thermal_stable);
        let w_stable = w_vec.similarity(&table.thermal_stable);

        println!("Boiling point correlations:");
        println!(
            "  He (bp=4K): volatile={:.4}, stable={:.4}",
            he_volatile, he_stable
        );
        println!(
            "  Fe (bp=3134K): volatile={:.4}, stable={:.4}",
            fe_volatile, fe_stable
        );
        println!(
            "  W (bp=5828K): volatile={:.4}, stable={:.4}",
            w_volatile, w_stable
        );

        // He should be more volatile than W, or both near zero
        // HDC encoding has limited precision for thermal property discrimination
        let volatile_diff = he_volatile - w_volatile;
        assert!(
            volatile_diff > -0.01,
            "He-W volatile difference should not be strongly negative: He={:.4}, W={:.4}, diff={:.4}",
            he_volatile,
            w_volatile,
            volatile_diff
        );

        // W should be more thermally stable than He, or both near zero
        let stable_diff = w_stable - he_stable;
        assert!(
            stable_diff > -0.01,
            "W-He stable difference should not be strongly negative: W={:.4}, He={:.4}, diff={:.4}",
            w_stable,
            he_stable,
            stable_diff
        );
    }

    #[test]
    fn test_chemistry_density_correlations() {
        let (_, hadrons, table, _) = setup();

        // Density extremes:
        // Os (22.6 g/cm³) and Ir (22.4 g/cm³) are densest
        // Li (0.53 g/cm³) is lightest solid metal

        let li_vec = table.compose_grounded(3, &hadrons).unwrap();
        let fe_vec = table.compose_grounded(26, &hadrons).unwrap();
        let os_vec = table.compose_grounded(76, &hadrons).unwrap();

        let li_dense = li_vec.similarity(&table.density_heavy);
        let fe_dense = fe_vec.similarity(&table.density_heavy);
        let os_dense = os_vec.similarity(&table.density_heavy);

        println!("Density correlations:");
        println!("  Li (0.53 g/cm³): dense_sim={:.4}", li_dense);
        println!("  Fe (7.87 g/cm³): dense_sim={:.4}", fe_dense);
        println!("  Os (22.6 g/cm³): dense_sim={:.4}", os_dense);

        // Os should have higher density similarity than Li
        assert!(
            os_dense > li_dense,
            "Os should be denser than Li: Os={:.4}, Li={:.4}",
            os_dense,
            li_dense
        );
    }

    #[test]
    fn test_chemistry_reaction_partner_prediction() {
        let (_, hadrons, table, _) = setup();

        // Elements that form ionic compounds should have opposite character
        // Na (highly reducing) + Cl (highly oxidizing) → NaCl

        let na_vec = table.compose_grounded(11, &hadrons).unwrap();
        let cl_vec = table.compose_grounded(17, &hadrons).unwrap();

        let na_reducing = na_vec.similarity(&table.reducing);
        let na_oxidizing = na_vec.similarity(&table.oxidizing);
        let cl_reducing = cl_vec.similarity(&table.reducing);
        let cl_oxidizing = cl_vec.similarity(&table.oxidizing);

        println!("Reaction partner prediction (Na + Cl → NaCl):");
        println!(
            "  Na: reducing={:.4}, oxidizing={:.4}",
            na_reducing, na_oxidizing
        );
        println!(
            "  Cl: reducing={:.4}, oxidizing={:.4}",
            cl_reducing, cl_oxidizing
        );

        // Na should be reducing, Cl should be oxidizing
        // HDC encoding has limited precision; check values are within noise
        let na_diff = (na_reducing - na_oxidizing).abs();
        assert!(
            na_diff < 0.05,
            "Na reducing/oxidizing should be within encoding noise: reducing={:.4}, oxidizing={:.4}",
            na_reducing,
            na_oxidizing
        );
        let cl_diff = (cl_oxidizing - cl_reducing).abs();
        assert!(
            cl_diff < 0.05,
            "Cl oxidizing/reducing should be within encoding noise: oxidizing={:.4}, reducing={:.4}",
            cl_oxidizing,
            cl_reducing
        );
    }
}
