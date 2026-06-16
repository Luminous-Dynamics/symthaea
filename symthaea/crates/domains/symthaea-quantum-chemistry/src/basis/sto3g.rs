// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! STO-3G minimal basis set.
//!
//! Each Slater-type orbital is expanded in 3 primitive Gaussians.
//! Data from Hehre, Stewart & Pople (1969), J. Chem. Phys. 51, 2657.
//! Verified against the Basis Set Exchange (BSE).

use super::{BasisSet, BasisSetProvider, ContractedGaussian, PrimitiveGaussian, ShellType};
use crate::molecule::Molecule;

/// STO-3G basis set provider.
pub struct Sto3g;

/// STO-3G shell data: (exponents, coefficients) for each contraction.
struct ShellData {
    shell_type: ShellType,
    exponents: [f64; 3],
    coefficients: [f64; 3],
}

/// Get STO-3G shell data for an element by atomic number.
/// Returns a list of shells (1s for H/He, 1s+2s+2p for Li-Ne, etc.)
fn shells_for_element(z: u8) -> Vec<ShellData> {
    match z {
        // ── Hydrogen (Z=1) ──────────────────────────────────────────────
        1 => vec![ShellData {
            shell_type: ShellType::S,
            exponents: [3.425_250_914, 0.623_913_730, 0.168_855_404],
            coefficients: [0.154_328_967, 0.535_328_142, 0.444_634_542],
        }],

        // ── Helium (Z=2) ───────────────────────────────────────────────
        2 => vec![ShellData {
            shell_type: ShellType::S,
            exponents: [6.362_421_394, 1.158_922_999, 0.313_654_629],
            coefficients: [0.154_328_967, 0.535_328_142, 0.444_634_542],
        }],

        // ── Lithium (Z=3) ──────────────────────────────────────────────
        3 => vec![
            // 1s
            ShellData {
                shell_type: ShellType::S,
                exponents: [16.119_575_0, 2.936_200_86, 0.794_650_487],
                coefficients: [0.154_328_967, 0.535_328_142, 0.444_634_542],
            },
            // 2s
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.636_289_746_0, 0.147_860_016_3, 0.048_088_602_55],
                coefficients: [-0.099_967_230, 0.399_512_826, 0.700_115_469],
            },
            // 2p
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.636_289_746_0, 0.147_860_016_3, 0.048_088_602_55],
                coefficients: [0.155_916_275, 0.607_683_719, 0.391_957_393],
            },
        ],

        // ── Beryllium (Z=4) ────────────────────────────────────────────
        4 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [30.167_871_0, 5.495_115_31, 1.487_192_69],
                coefficients: [0.154_328_967, 0.535_328_142, 0.444_634_542],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [1.314_833_11, 0.305_308_946, 0.099_370_700_2],
                coefficients: [-0.099_967_230, 0.399_512_826, 0.700_115_469],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [1.314_833_11, 0.305_308_946, 0.099_370_700_2],
                coefficients: [0.155_916_275, 0.607_683_719, 0.391_957_393],
            },
        ],

        // ── Boron (Z=5) ────────────────────────────────────────────────
        5 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [48.791_113_0, 8.887_362_45, 2.405_267_04],
                coefficients: [0.154_328_967, 0.535_328_142, 0.444_634_542],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [2.236_956_14, 0.519_520_437, 0.169_061_850],
                coefficients: [-0.099_967_230, 0.399_512_826, 0.700_115_469],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [2.236_956_14, 0.519_520_437, 0.169_061_850],
                coefficients: [0.155_916_275, 0.607_683_719, 0.391_957_393],
            },
        ],

        // ── Carbon (Z=6) ───────────────────────────────────────────────
        6 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [71.616_837_0, 13.045_096_0, 3.530_512_16],
                coefficients: [0.154_328_967, 0.535_328_142, 0.444_634_542],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [2.941_249_36, 0.683_483_096, 0.222_289_916],
                coefficients: [-0.099_967_230, 0.399_512_826, 0.700_115_469],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [2.941_249_36, 0.683_483_096, 0.222_289_916],
                coefficients: [0.155_916_275, 0.607_683_719, 0.391_957_393],
            },
        ],

        // ── Nitrogen (Z=7) ─────────────────────────────────────────────
        7 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [99.106_169_0, 18.052_312_0, 4.885_660_24],
                coefficients: [0.154_328_967, 0.535_328_142, 0.444_634_542],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [3.780_455_88, 0.878_496_823, 0.285_714_310],
                coefficients: [-0.099_967_230, 0.399_512_826, 0.700_115_469],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [3.780_455_88, 0.878_496_823, 0.285_714_310],
                coefficients: [0.155_916_275, 0.607_683_719, 0.391_957_393],
            },
        ],

        // ── Oxygen (Z=8) ───────────────────────────────────────────────
        8 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [130.709_320_0, 23.808_866_0, 6.443_608_31],
                coefficients: [0.154_328_967, 0.535_328_142, 0.444_634_542],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [5.033_151_32, 1.169_596_13, 0.380_389_000],
                coefficients: [-0.099_967_230, 0.399_512_826, 0.700_115_469],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [5.033_151_32, 1.169_596_13, 0.380_389_000],
                coefficients: [0.155_916_275, 0.607_683_719, 0.391_957_393],
            },
        ],

        // ── Fluorine (Z=9) ─────────────────────────────────────────────
        9 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [166.679_134_0, 30.360_816_0, 8.216_820_07],
                coefficients: [0.154_328_967, 0.535_328_142, 0.444_634_542],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [6.464_803_25, 1.502_281_25, 0.488_588_486],
                coefficients: [-0.099_967_230, 0.399_512_826, 0.700_115_469],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [6.464_803_25, 1.502_281_25, 0.488_588_486],
                coefficients: [0.155_916_275, 0.607_683_719, 0.391_957_393],
            },
        ],

        // ── Neon (Z=10) ────────────────────────────────────────────────
        10 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [207.015_600_0, 37.708_151_0, 10.205_297_0],
                coefficients: [0.154_328_967, 0.535_328_142, 0.444_634_542],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [8.246_315_12, 1.916_266_29, 0.623_214_190],
                coefficients: [-0.099_967_230, 0.399_512_826, 0.700_115_469],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [8.246_315_12, 1.916_266_29, 0.623_214_190],
                coefficients: [0.155_916_275, 0.607_683_719, 0.391_957_393],
            },
        ],

        _ => panic!("STO-3G not implemented for Z={}", z),
    }
}

impl BasisSetProvider for Sto3g {
    fn name() -> &'static str {
        "STO-3G"
    }

    fn build(molecule: &Molecule) -> BasisSet {
        let mut functions = Vec::new();

        for atom in &molecule.atoms {
            let shells = shells_for_element(atom.atomic_number);

            for shell in shells {
                // For each Cartesian component of this shell
                for (l, m, n) in shell.shell_type.cartesian_components() {
                    let primitives: Vec<PrimitiveGaussian> = shell
                        .exponents
                        .iter()
                        .zip(shell.coefficients.iter())
                        .map(|(&alpha, &coeff)| PrimitiveGaussian {
                            alpha,
                            coeff,
                            center: atom.position,
                            l,
                            m,
                            n,
                        })
                        .collect();

                    functions.push(ContractedGaussian {
                        primitives,
                        shell_type: shell.shell_type,
                    });
                }
            }
        }

        BasisSet {
            name: "STO-3G".to_string(),
            functions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Molecule;

    #[test]
    fn test_h2_basis_count() {
        let h2 = Molecule::h2();
        let basis = Sto3g::build(&h2);
        // H2: each H has 1 s-function → 2 total
        assert_eq!(basis.n_basis(), 2);
    }

    #[test]
    fn test_water_basis_count() {
        let water = Molecule::water();
        let basis = Sto3g::build(&water);
        // O: 1s + 2s + 2px + 2py + 2pz = 5
        // H: 1s each = 2
        // Total = 7
        assert_eq!(basis.n_basis(), 7);
    }

    #[test]
    fn test_heh_plus_basis_count() {
        let heh = Molecule::heh_plus();
        let basis = Sto3g::build(&heh);
        // He: 1s = 1, H: 1s = 1 → 2
        assert_eq!(basis.n_basis(), 2);
    }

    #[test]
    fn test_primitives_per_contraction() {
        let h2 = Molecule::h2();
        let basis = Sto3g::build(&h2);
        // STO-3G = 3 primitives per contracted function
        for func in &basis.functions {
            assert_eq!(func.primitives.len(), 3);
        }
    }

    #[test]
    fn test_h_exponents_match_bse() {
        // BSE reference for H STO-3G:
        // α = 3.42525091, 0.62391373, 0.16885540
        let h = Molecule::new(vec![crate::molecule::Atom::new(1, 0.0, 0.0, 0.0)]);
        let basis = Sto3g::build(&h);
        let alphas: Vec<f64> = basis.functions[0]
            .primitives
            .iter()
            .map(|p| p.alpha)
            .collect();
        assert!((alphas[0] - 3.425_250_914).abs() < 1e-6);
        assert!((alphas[1] - 0.623_913_730).abs() < 1e-6);
        assert!((alphas[2] - 0.168_855_404).abs() < 1e-6);
    }
}
