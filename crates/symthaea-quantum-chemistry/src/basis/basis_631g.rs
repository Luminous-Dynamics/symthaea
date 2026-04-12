// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! 6-31G split-valence basis set.
//!
//! Core orbitals: 6 primitive Gaussians contracted to 1 function.
//! Valence: 3 primitives (inner) + 1 primitive (outer) = 2 functions.
//! This "split-valence" design lets valence orbitals adjust their radial extent.
//!
//! Significantly more accurate than STO-3G for molecular energies, geometries,
//! and properties. Roughly doubles the basis set size.
//!
//! Data from Basis Set Exchange (BSE), Hehre, Ditchfield & Pople (1972),
//! J. Chem. Phys. 56, 2257.

use super::{BasisSet, BasisSetProvider, ContractedGaussian, PrimitiveGaussian, ShellType};
use crate::molecule::Molecule;

pub struct Basis631G;

struct ShellData631G {
    shell_type: ShellType,
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
}

fn shells_for_element(z: u8) -> Vec<ShellData631G> {
    match z {
        // ── Hydrogen (Z=1): 2 s-functions (3+1 split) ────────────────────
        1 => vec![
            // Inner valence: 3 primitives
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![18.731_137_0, 2.825_394_0, 0.640_121_7],
                coefficients: vec![0.033_495_0, 0.234_727_0, 0.813_757_0],
            },
            // Outer valence: 1 primitive
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![0.161_278_0],
                coefficients: vec![1.0],
            },
        ],

        // ── Carbon (Z=6): 1s core + 2sp inner + 2sp outer ────────────────
        6 => vec![
            // 1s core: 6 primitives
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![
                    3047.524_9, 457.369_51, 103.948_69, 29.210_155, 9.286_663, 3.163_927,
                ],
                coefficients: vec![
                    0.001_835, 0.014_037, 0.068_843, 0.232_184, 0.467_941, 0.362_312,
                ],
            },
            // 2s inner: 3 primitives (Hehre 1972 segmented contraction)
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![7.868_272_4, 1.881_288_5, 0.544_249_3],
                coefficients: vec![-0.119_332, 0.155_916_1, 1.058_119_9],
            },
            // 2p inner: 3 primitives (same exponents as 2s inner)
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![7.868_272_4, 1.881_288_5, 0.544_249_3],
                coefficients: vec![0.068_999, 0.316_424, 0.744_308],
            },
            // 2s outer: 1 primitive
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![0.168_714_4],
                coefficients: vec![1.0],
            },
            // 2p outer: 1 primitive
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![0.168_714_4],
                coefficients: vec![1.0],
            },
        ],

        // ── Nitrogen (Z=7) — Hehre (1972) segmented contraction ──────────
        // NOTE: BSE provides general-contraction coefficients with different
        // normalization. These are the segmented contraction values that work
        // with our per-primitive normalization.
        7 => vec![
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![
                    4173.511_5, 627.457_91, 142.902_09, 40.234_330, 12.820_213, 4.390_437,
                ],
                coefficients: vec![
                    0.001_835, 0.013_995, 0.068_587, 0.232_241, 0.469_070, 0.360_455,
                ],
            },
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![11.626_362, 2.716_280, 0.772_218],
                coefficients: vec![-0.114_961, 0.169_112, 1.037_480],
            },
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![11.626_362, 2.716_280, 0.772_218],
                coefficients: vec![0.067_580, 0.323_907, 0.740_895],
            },
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![0.212_031_3],
                coefficients: vec![1.0],
            },
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![0.212_031_3],
                coefficients: vec![1.0],
            },
        ],

        // ── Oxygen (Z=8) ──────────────────────────────────────────────────
        8 => vec![
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![
                    5484.671_7,
                    825.234_95,
                    188.046_96,
                    52.964_500,
                    16.897_571,
                    5.799_635_3,
                ],
                coefficients: vec![
                    0.001_831, 0.013_950, 0.068_445, 0.232_714, 0.470_193, 0.358_521,
                ],
            },
            // 2s inner: 3 primitives (Hehre 1972 segmented contraction)
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![15.539_616, 3.599_934, 1.013_762],
                coefficients: vec![-0.110_778, 0.148_026, 1.050_969],
            },
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![15.539_616, 3.599_934, 1.013_762],
                coefficients: vec![0.070_874, 0.339_753, 0.727_159],
            },
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![0.270_006],
                coefficients: vec![1.0],
            },
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![0.270_006],
                coefficients: vec![1.0],
            },
        ],

        _ => panic!("6-31G not implemented for Z={}. Supported: H, C, N, O.", z),
    }
}

impl BasisSetProvider for Basis631G {
    fn name() -> &'static str {
        "6-31G"
    }

    fn build(molecule: &Molecule) -> BasisSet {
        let mut functions = Vec::new();

        for atom in &molecule.atoms {
            let shells = shells_for_element(atom.atomic_number);

            for shell in shells {
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

                    let mut cg = ContractedGaussian {
                        primitives,
                        shell_type: shell.shell_type,
                    };
                    // Normalize contracted function so ⟨χ|χ⟩ = 1
                    cg.normalize();
                    functions.push(cg);
                }
            }
        }

        BasisSet {
            name: "6-31G".to_string(),
            functions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Molecule;

    #[test]
    fn test_h2_631g_basis_count() {
        let h2 = Molecule::h2();
        let basis = Basis631G::build(&h2);
        // H: 2 s-functions each → 4 total
        assert_eq!(basis.n_basis(), 4, "H2 6-31G should have 4 basis functions");
    }

    #[test]
    fn test_water_631g_basis_count() {
        let water = Molecule::water();
        let basis = Basis631G::build(&water);
        // O: 1s(core) + 2s(inner) + 2px,2py,2pz(inner) + 2s(outer) + 2px,2py,2pz(outer)
        //  = 1 + 1 + 3 + 1 + 3 = 9
        // H: 2 s-functions each × 2 = 4
        // Total = 13
        assert_eq!(
            basis.n_basis(),
            13,
            "H2O 6-31G should have 13 basis functions"
        );
    }

    #[test]
    fn test_631g_larger_than_sto3g() {
        let water = Molecule::water();
        let sto3g = crate::basis::sto3g::Sto3g::build(&water);
        let b631g = Basis631G::build(&water);
        assert!(
            b631g.n_basis() > sto3g.n_basis(),
            "6-31G ({}) should be larger than STO-3G ({})",
            b631g.n_basis(),
            sto3g.n_basis()
        );
    }

    #[test]
    fn test_631g_hf_water_more_accurate() {
        use crate::scf::rhf::{restricted_hartree_fock, RhfConfig};

        let water = Molecule::water();

        let sto3g = crate::basis::sto3g::Sto3g::build(&water);
        let e_sto3g = restricted_hartree_fock(&water, &sto3g, &RhfConfig::default()).total_energy;

        let b631g = Basis631G::build(&water);
        let e_631g = restricted_hartree_fock(&water, &b631g, &RhfConfig::default()).total_energy;

        // 6-31G should give a lower (more negative) energy than STO-3G
        // because it has more variational freedom
        assert!(
            e_631g < e_sto3g,
            "6-31G ({:.4}) should be lower than STO-3G ({:.4})",
            e_631g,
            e_sto3g
        );

        // Reference: HF/6-31G water ≈ -75.585 Ha (vs STO-3G ≈ -74.96)
        assert!(
            e_631g < -75.0,
            "H2O HF/6-31G = {:.4}, expected < -75.0",
            e_631g
        );
    }
}
