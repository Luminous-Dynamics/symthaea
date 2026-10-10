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
//! Data from Basis Set Exchange (BSE) 6-31G version 1 (Gaussian 09/GAMESS),
//! whose underlying basis was published by Hehre, Ditchfield & Pople, J. Chem.
//! Phys. 56, 2257-2261 (1972), DOI: 10.1063/1.1677527.
//!
//! Pinned source provenance (BSE repository):
//! - dataset repository commit: 4adaf1372c7101620ca1a9f3130be9ae97fb8f30
//! - 6-31G component JSON blob: 200641c79d517e0bab441be3f29f49ba34d735e9
//! - 6-31G table JSON blob: 603f82017c3300e2cbdf6e83aa0383c1c4316ad9
//! - hydrogen 31G component JSON blob: 997a8ef2577ea40259198884b7d2e0330b757311
//!
//! Coefficients use the source Gaussian/GAMESS convention. Integral routines
//! multiply each coefficient by the analytic primitive normalization and
//! normalize each contracted AO by one common factor. That common normalization
//! cannot change coefficient signs or relative magnitudes.

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
                exponents: vec![18.731_136_96, 2.825_394_365, 0.640_121_6923],
                coefficients: vec![0.033_494_604_34, 0.234_726_9535, 0.813_757_3261],
            },
            // Outer valence: 1 primitive
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![0.161_277_7588],
                coefficients: vec![1.0],
            },
        ],

        // ── Carbon (Z=6): 1s core + 2sp inner + 2sp outer ────────────────
        6 => vec![
            // 1s core: 6 primitives
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![
                    3047.524_880,
                    457.369_518,
                    103.948_6850,
                    29.210_15530,
                    9.286_662960,
                    3.163_926960,
                ],
                coefficients: vec![
                    0.001_834_737_132,
                    0.014_037_32281,
                    0.068_842_62226,
                    0.232_184_4432,
                    0.467_941_3484,
                    0.362_311_9853,
                ],
            },
            // 2s inner: 3 primitives, pinned BSE 6-31G v1 contraction
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![7.868_272_350, 1.881_288_540, 0.544_249_2580],
                coefficients: vec![-0.119_332_4198, -0.160_854_1517, 1.143_456_438],
            },
            // 2p inner: 3 primitives (same exponents as 2s inner)
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![7.868_272_350, 1.881_288_540, 0.544_249_2580],
                coefficients: vec![0.068_999_06659, 0.316_423_9610, 0.744_308_2909],
            },
            // 2s outer: 1 primitive
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![0.168_714_4782],
                coefficients: vec![1.0],
            },
            // 2p outer: 1 primitive
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![0.168_714_4782],
                coefficients: vec![1.0],
            },
        ],

        // ── Nitrogen (Z=7), pinned BSE 6-31G v1 contraction ────────────────
        7 => vec![
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![
                    4173.511_460,
                    627.457_9110,
                    142.902_0930,
                    40.234_32930,
                    12.820_21290,
                    4.390_437010,
                ],
                coefficients: vec![
                    0.001_834_772160,
                    0.013_994_62700,
                    0.068_586_55181,
                    0.232_240_8730,
                    0.469_069_9481,
                    0.360_455_1991,
                ],
            },
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![11.626_36186, 2.716_279_807, 0.772_218_3966],
                coefficients: vec![-0.114_961_1817, -0.169_117_4786, 1.145_851_947],
            },
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![11.626_36186, 2.716_279_807, 0.772_218_3966],
                coefficients: vec![0.067_579_74388, 0.323_907_2959, 0.740_895_1398],
            },
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![0.212_031_4975],
                coefficients: vec![1.0],
            },
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![0.212_031_4975],
                coefficients: vec![1.0],
            },
        ],

        // ── Oxygen (Z=8) ──────────────────────────────────────────────────
        8 => vec![
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![
                    5484.671_660,
                    825.234_9460,
                    188.046_9580,
                    52.964_50000,
                    16.897_57040,
                    5.799_635340,
                ],
                coefficients: vec![
                    0.001_831_074430,
                    0.013_950_17220,
                    0.068_445_07810,
                    0.232_714_3360,
                    0.470_192_8980,
                    0.358_520_8530,
                ],
            },
            // 2s inner: 3 primitives, pinned BSE 6-31G v1 contraction
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![15.539_61625, 3.599_933_586, 1.013_761_750],
                coefficients: vec![-0.110_777_5495, -0.148_026_2627, 1.130_767_015],
            },
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![15.539_61625, 3.599_933_586, 1.013_761_750],
                coefficients: vec![0.070_874_26823, 0.339_752_8391, 0.727_158_5773],
            },
            ShellData631G {
                shell_type: ShellType::S,
                exponents: vec![0.270_005_8226],
                coefficients: vec![1.0],
            },
            ShellData631G {
                shell_type: ShellType::P,
                exponents: vec![0.270_005_8226],
                coefficients: vec![1.0],
            },
        ],

        _ => panic!("6-31G not implemented for Z={}. Supported: H, C, N, O.", z),
    }
}

impl Basis631G {
    /// Non-panicking capability query: does this provider have real 6-31G
    /// data for element `z`? (Phase Q1, 2026-07-16.) Only H/C/N/O -- this
    /// crate's 6-31G coverage was deliberately NOT widened alongside
    /// STO-3G's Phase A.8 extension, since the H2O/CH4 6-31G energy
    /// discrepancy (Phase Q0) is real, unresolved, and specific to this
    /// basis's data/contraction scheme; extending coverage before that's
    /// understood risks propagating an unknown bug into more elements. See
    /// `QUANTUM_CHEMISTRY_COMPLETENESS_ROADMAP_2026-07-16.md`'s Q5 entry.
    pub fn supports_element(z: u8) -> bool {
        matches!(z, 1 | 6 | 7 | 8)
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
    use crate::integrals::overlap::overlap_matrix;
    use crate::molecule::{Atom, Molecule};

    #[test]
    fn test_inner_valence_s_contractions_match_pinned_bse_v1() {
        let cases = [
            (
                6u8,
                vec![7.868_272_350, 1.881_288_540, 0.544_249_2580],
                vec![-0.119_332_4198, -0.160_854_1517, 1.143_456_438],
            ),
            (
                7u8,
                vec![11.626_36186, 2.716_279_807, 0.772_218_3966],
                vec![-0.114_961_1817, -0.169_117_4786, 1.145_851_947],
            ),
            (
                8u8,
                vec![15.539_61625, 3.599_933_586, 1.013_761_750],
                vec![-0.110_777_5495, -0.148_026_2627, 1.130_767_015],
            ),
        ];
        for (z, expected_exponents, expected_coefficients) in cases {
            let shells = shells_for_element(z);
            assert_eq!(shells[1].shell_type, ShellType::S, "Z={z}");
            assert_eq!(shells[1].exponents, expected_exponents, "Z={z} BSE exponents");
            assert_eq!(
                shells[1].coefficients,
                expected_coefficients,
                "Z={z} BSE raw contraction coefficients"
            );
        }
    }

    #[test]
    fn test_outer_sp_shell_exponents_match_pinned_bse_v1() {
        let cases = [(6u8, 0.168_714_4782), (7u8, 0.212_031_4975), (8u8, 0.270_005_8226)];
        for (z, expected_exponent) in cases {
            let shells = shells_for_element(z);
            assert_eq!(shells[3].exponents, vec![expected_exponent], "Z={z} outer s");
            assert_eq!(shells[4].exponents, vec![expected_exponent], "Z={z} outer p");
        }
    }

    fn shell(
        shell_type: ShellType,
        exponents: &[f64],
        coefficients: &[f64],
    ) -> (ShellType, Vec<f64>, Vec<f64>) {
        (shell_type, exponents.to_vec(), coefficients.to_vec())
    }

    fn assert_shell_data(
        atomic_number: u8,
        expected: &[(ShellType, Vec<f64>, Vec<f64>)],
    ) {
        let actual = shells_for_element(atomic_number);
        assert_eq!(actual.len(), expected.len(), "Z={atomic_number} shell count");
        for (index, (actual, (expected_type, expected_exponents, expected_coefficients))) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(actual.shell_type, *expected_type, "Z={atomic_number} shell={index}");
            assert_eq!(
                actual.exponents, *expected_exponents,
                "Z={atomic_number} shell={index} exponents"
            );
            assert_eq!(
                actual.coefficients,
                *expected_coefficients,
                "Z={atomic_number} shell={index} BSE raw coefficients"
            );
        }
    }

    #[test]
    fn test_all_supported_shell_data_match_pinned_bse_v1() {
        assert_shell_data(
            1,
            &[
                shell(
                    ShellType::S,
                    &[18.731_136_96, 2.825_394_365, 0.640_121_6923],
                    &[0.033_494_604_34, 0.234_726_9535, 0.813_757_3261],
                ),
                shell(ShellType::S, &[0.161_277_7588], &[1.0]),
            ],
        );
        assert_shell_data(
            6,
            &[
                shell(
                    ShellType::S,
                    &[
                        3047.524_880,
                        457.369_518,
                        103.948_6850,
                        29.210_15530,
                        9.286_662960,
                        3.163_926960,
                    ],
                    &[
                        0.001_834_737132,
                        0.014_037_32281,
                        0.068_842_62226,
                        0.232_184_4432,
                        0.467_941_3484,
                        0.362_311_9853,
                    ],
                ),
                shell(
                    ShellType::S,
                    &[7.868_272_350, 1.881_288_540, 0.544_249_2580],
                    &[-0.119_332_4198, -0.160_854_1517, 1.143_456_438],
                ),
                shell(
                    ShellType::P,
                    &[7.868_272_350, 1.881_288_540, 0.544_249_2580],
                    &[0.068_999_06659, 0.316_423_9610, 0.744_308_2909],
                ),
                shell(ShellType::S, &[0.168_714_4782], &[1.0]),
                shell(ShellType::P, &[0.168_714_4782], &[1.0]),
            ],
        );
        assert_shell_data(
            7,
            &[
                shell(
                    ShellType::S,
                    &[
                        4173.511_460,
                        627.457_9110,
                        142.902_0930,
                        40.234_32930,
                        12.820_21290,
                        4.390_437010,
                    ],
                    &[
                        0.001_834_772160,
                        0.013_994_62700,
                        0.068_586_55181,
                        0.232_240_8730,
                        0.469_069_9481,
                        0.360_455_1991,
                    ],
                ),
                shell(
                    ShellType::S,
                    &[11.626_36186, 2.716_279_807, 0.772_218_3966],
                    &[-0.114_961_1817, -0.169_117_4786, 1.145_851_947],
                ),
                shell(
                    ShellType::P,
                    &[11.626_36186, 2.716_279_807, 0.772_218_3966],
                    &[0.067_579_74388, 0.323_907_2959, 0.740_895_1398],
                ),
                shell(ShellType::S, &[0.212_031_4975], &[1.0]),
                shell(ShellType::P, &[0.212_031_4975], &[1.0]),
            ],
        );
        assert_shell_data(
            8,
            &[
                shell(
                    ShellType::S,
                    &[
                        5484.671_660,
                        825.234_9460,
                        188.046_9580,
                        52.964_50000,
                        16.897_57040,
                        5.799_635340,
                    ],
                    &[
                        0.001_831_074430,
                        0.013_950_17220,
                        0.068_445_07810,
                        0.232_714_3360,
                        0.470_192_8980,
                        0.358_520_8530,
                    ],
                ),
                shell(
                    ShellType::S,
                    &[15.539_61625, 3.599_933_586, 1.013_761_750],
                    &[-0.110_777_5495, -0.148_026_2627, 1.130_767_015],
                ),
                shell(
                    ShellType::P,
                    &[15.539_61625, 3.599_933_586, 1.013_761_750],
                    &[0.070_874_26823, 0.339_752_8391, 0.727_158_5773],
                ),
                shell(ShellType::S, &[0.270_005_8226], &[1.0]),
                shell(ShellType::P, &[0.270_005_8226], &[1.0]),
            ],
        );
    }

    #[test]
    fn test_core_valence_s_overlap_matches_pinned_bse_v1() {
        // Analytic BSE v1 fingerprints for the same-center normalized 1s-core
        // versus inner-valence-s overlap. This exercises the real Rust primitive
        // and contracted-AO normalization path, not just the source literals.
        let cases = [
            (6u8, 0.219_058_848_267_941_4),
            (7u8, 0.222_215_815_911_390_1),
            (8u8, 0.233_689_857_197_009_0),
        ];
        for (atomic_number, expected) in cases {
            let molecule = Molecule::new(vec![Atom::new(atomic_number, 0.0, 0.0, 0.0)]);
            let basis = Basis631G::build(&molecule);
            let overlap = overlap_matrix(&basis.functions);
            let core_inner_s = overlap[1];
            assert!(
                (core_inner_s - expected).abs() < 1e-10,
                "Z={atomic_number} core/inner-s overlap {core_inner_s:.15} != BSE v1 {expected:.15}"
            );
        }
    }

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
        use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

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

    #[test]
    fn test_supports_element_agrees_with_shells_for_element_boundary() {
        // Phase Q1 (2026-07-16): H/C/N/O supported, everything else isn't --
        // must agree with what shells_for_element actually does.
        for z in [1u8, 6, 7, 8] {
            assert!(Basis631G::supports_element(z), "Z={z} should be supported");
            let _ = shells_for_element(z); // must not panic
        }
        for z in [2u8, 9, 17, 54] {
            assert!(
                !Basis631G::supports_element(z),
                "Z={z} should not be supported"
            );
        }
    }
}
