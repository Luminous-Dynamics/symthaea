// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semi-empirical descriptors for high-entropy-alloy screening.
//!
//! These descriptors are inexpensive hypothesis filters, not thermodynamic
//! stability proofs. In particular, classic thresholds for atomic-size mismatch,
//! Ω, and VEC are composition-class heuristics with known exceptions. This module
//! intentionally exposes no `is_stable` or phase-certification API.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const R_GAS_J_MOL_K: f64 = 8.314_462_618;
const FRACTION_TOLERANCE: f64 = 1.0e-6;

/// Element-level inputs required for inexpensive HEA descriptors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeaElementDescriptor {
    /// Atomic number, used only as a stable element identifier here.
    pub atomic_number: u16,
    /// Atomic fraction in the candidate alloy.
    pub atomic_fraction: f64,
    /// Atomic/metallic radius in pm from a consistently chosen source table.
    pub atomic_radius_pm: f64,
    /// Element melting point in kelvin.
    pub melting_point_k: f64,
    /// Valence electron concentration used by the selected VEC convention.
    ///
    /// `None` is preferred over inventing a value for elements whose convention
    /// is ambiguous in a given study.
    pub vec: Option<f64>,
}

/// Equiatomic binary mixing-enthalpy input used by the common pair model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairMixingEnthalpy {
    /// First element atomic number.
    pub atomic_number_a: u16,
    /// Second element atomic number.
    pub atomic_number_b: u16,
    /// Equiatomic binary mixing enthalpy in kJ/mol.
    pub delta_h_equiatomic_kj_mol: f64,
}

/// Classic VEC-derived crystal-structure tendency.
///
/// This is deliberately named a *tendency*: VEC rules have documented exceptions
/// and must not be promoted to an experimentally established phase assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VecStructureTendency {
    /// VEC < 6.87: classic rule tends toward BCC.
    BccLeaning,
    /// 6.87 <= VEC < 8.0: classic rule tends toward mixed BCC/FCC.
    MixedBccFcc,
    /// VEC >= 8.0: classic rule tends toward FCC.
    FccLeaning,
}

/// Named classic screening criteria, retained as telemetry only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassicHeaScreeningFlags {
    /// Ideal configurational entropy is at least 1.5R.
    pub entropy_ge_1_5_r: bool,
    /// Atomic-size mismatch is at most 6.6%.
    pub atomic_size_delta_le_6_6_pct: bool,
    /// Ω is at least 1.1, if Ω could be calculated.
    pub omega_ge_1_1: Option<bool>,
    /// VEC-derived structure tendency, if VEC could be calculated.
    pub vec_tendency: Option<VecStructureTendency>,
}

/// Computed HEA screening descriptors.
///
/// No field in this structure establishes thermodynamic or experimental
/// stability. It is suitable for ranking, ablation design, and deciding which
/// expensive calculations to run next.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeaScreeningDescriptors {
    /// Ideal configurational entropy in J/(mol K).
    pub configurational_entropy_j_mol_k: f64,
    /// Atomic-size mismatch δ in percent.
    pub atomic_size_mismatch_pct: f64,
    /// Composition-weighted average atomic radius in pm.
    pub mean_atomic_radius_pm: f64,
    /// Composition-weighted average melting point in K.
    pub mean_melting_point_k: f64,
    /// Composition-weighted VEC, absent if any element lacks a VEC input.
    pub vec: Option<f64>,
    /// Pair-model mixing enthalpy in kJ/mol, absent if pair data are incomplete.
    pub mixing_enthalpy_kj_mol: Option<f64>,
    /// Ω = Tm * ΔSconf / |ΔHmix|, absent if mixing enthalpy is unavailable or ~0.
    pub omega: Option<f64>,
    /// Named classic heuristic thresholds for convenient comparison.
    pub classic_flags: ClassicHeaScreeningFlags,
}

/// Calculate inexpensive semi-empirical HEA descriptors.
pub fn calculate_hea_descriptors(
    elements: &[HeaElementDescriptor],
    pair_enthalpies: &[PairMixingEnthalpy],
) -> Result<HeaScreeningDescriptors, HeaScreeningError> {
    validate_elements(elements)?;

    let entropy = -R_GAS_J_MOL_K
        * elements
            .iter()
            .map(|e| e.atomic_fraction * e.atomic_fraction.ln())
            .sum::<f64>();

    let mean_radius = elements
        .iter()
        .map(|e| e.atomic_fraction * e.atomic_radius_pm)
        .sum::<f64>();
    let delta = 100.0
        * elements
            .iter()
            .map(|e| {
                e.atomic_fraction * (1.0 - e.atomic_radius_pm / mean_radius).powi(2)
            })
            .sum::<f64>()
            .sqrt();
    let mean_melting = elements
        .iter()
        .map(|e| e.atomic_fraction * e.melting_point_k)
        .sum::<f64>();

    let vec = if elements.iter().all(|e| e.vec.is_some()) {
        Some(
            elements
                .iter()
                .map(|e| e.atomic_fraction * e.vec.expect("checked above"))
                .sum(),
        )
    } else {
        None
    };

    let mixing_enthalpy = calculate_pair_mixing_enthalpy(elements, pair_enthalpies)?;
    let omega = mixing_enthalpy.and_then(|delta_h| {
        let magnitude_j_mol = delta_h.abs() * 1000.0;
        if magnitude_j_mol <= 1.0e-12 {
            None
        } else {
            Some(mean_melting * entropy / magnitude_j_mol)
        }
    });

    let vec_tendency = vec.map(|v| {
        if v < 6.87 {
            VecStructureTendency::BccLeaning
        } else if v < 8.0 {
            VecStructureTendency::MixedBccFcc
        } else {
            VecStructureTendency::FccLeaning
        }
    });

    Ok(HeaScreeningDescriptors {
        configurational_entropy_j_mol_k: entropy,
        atomic_size_mismatch_pct: delta,
        mean_atomic_radius_pm: mean_radius,
        mean_melting_point_k: mean_melting,
        vec,
        mixing_enthalpy_kj_mol: mixing_enthalpy,
        omega,
        classic_flags: ClassicHeaScreeningFlags {
            entropy_ge_1_5_r: entropy >= 1.5 * R_GAS_J_MOL_K,
            atomic_size_delta_le_6_6_pct: delta <= 6.6,
            omega_ge_1_1: omega.map(|value| value >= 1.1),
            vec_tendency,
        },
    })
}

fn validate_elements(elements: &[HeaElementDescriptor]) -> Result<(), HeaScreeningError> {
    if elements.len() < 2 {
        return Err(HeaScreeningError::NeedMultipleElements);
    }
    let mut seen = HashSet::new();
    let mut total_fraction = 0.0;
    for element in elements {
        if !seen.insert(element.atomic_number) {
            return Err(HeaScreeningError::DuplicateElement {
                atomic_number: element.atomic_number,
            });
        }
        for (field, value) in [
            ("atomic_fraction", element.atomic_fraction),
            ("atomic_radius_pm", element.atomic_radius_pm),
            ("melting_point_k", element.melting_point_k),
        ] {
            if !value.is_finite() {
                return Err(HeaScreeningError::NonFiniteValue {
                    field,
                    value,
                });
            }
        }
        if element.atomic_fraction <= 0.0 || element.atomic_fraction > 1.0 {
            return Err(HeaScreeningError::InvalidAtomicFraction {
                atomic_number: element.atomic_number,
                value: element.atomic_fraction,
            });
        }
        if element.atomic_radius_pm <= 0.0 {
            return Err(HeaScreeningError::NonPositiveRadius {
                atomic_number: element.atomic_number,
            });
        }
        if element.melting_point_k <= 0.0 {
            return Err(HeaScreeningError::NonPositiveMeltingPoint {
                atomic_number: element.atomic_number,
            });
        }
        if let Some(vec) = element.vec {
            if !vec.is_finite() || vec < 0.0 {
                return Err(HeaScreeningError::InvalidVec {
                    atomic_number: element.atomic_number,
                    value: vec,
                });
            }
        }
        total_fraction += element.atomic_fraction;
    }
    if (total_fraction - 1.0).abs() > FRACTION_TOLERANCE {
        return Err(HeaScreeningError::FractionsDoNotSumToOne {
            sum: total_fraction,
        });
    }
    Ok(())
}

/// Common pair-table estimate ΔHmix = 4 Σ_{i<j} x_i x_j H_ij.
///
/// Returns `None` if no pair table is supplied. If a table is supplied, every
/// unique element pair is required; partial pair data are rejected rather than
/// silently interpreted as zero interaction.
fn calculate_pair_mixing_enthalpy(
    elements: &[HeaElementDescriptor],
    pair_enthalpies: &[PairMixingEnthalpy],
) -> Result<Option<f64>, HeaScreeningError> {
    if pair_enthalpies.is_empty() {
        return Ok(None);
    }

    let fractions: HashMap<u16, f64> = elements
        .iter()
        .map(|e| (e.atomic_number, e.atomic_fraction))
        .collect();
    let mut pairs = HashMap::new();
    for pair in pair_enthalpies {
        if pair.atomic_number_a == pair.atomic_number_b {
            return Err(HeaScreeningError::SelfPair {
                atomic_number: pair.atomic_number_a,
            });
        }
        if !pair.delta_h_equiatomic_kj_mol.is_finite() {
            return Err(HeaScreeningError::NonFiniteValue {
                field: "delta_h_equiatomic_kj_mol",
                value: pair.delta_h_equiatomic_kj_mol,
            });
        }
        if !fractions.contains_key(&pair.atomic_number_a)
            || !fractions.contains_key(&pair.atomic_number_b)
        {
            return Err(HeaScreeningError::PairElementNotInComposition);
        }
        let key = ordered_pair(pair.atomic_number_a, pair.atomic_number_b);
        if pairs
            .insert(key, pair.delta_h_equiatomic_kj_mol)
            .is_some()
        {
            return Err(HeaScreeningError::DuplicatePair {
                atomic_number_a: key.0,
                atomic_number_b: key.1,
            });
        }
    }

    let mut total = 0.0;
    for i in 0..elements.len() {
        for j in (i + 1)..elements.len() {
            let a = elements[i].atomic_number;
            let b = elements[j].atomic_number;
            let key = ordered_pair(a, b);
            let h = pairs
                .get(&key)
                .ok_or(HeaScreeningError::MissingPair {
                    atomic_number_a: key.0,
                    atomic_number_b: key.1,
                })?;
            total += 4.0 * elements[i].atomic_fraction * elements[j].atomic_fraction * h;
        }
    }
    Ok(Some(total))
}

fn ordered_pair(a: u16, b: u16) -> (u16, u16) {
    if a < b { (a, b) } else { (b, a) }
}

/// HEA descriptor validation failures.
#[derive(Debug, Clone, PartialEq)]
pub enum HeaScreeningError {
    /// Descriptor calculation requires a multicomponent alloy.
    NeedMultipleElements,
    /// The same element was supplied more than once.
    DuplicateElement {
        /// Atomic number of the duplicate.
        atomic_number: u16,
    },
    /// A numeric input was NaN or infinite.
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Atomic fraction must be in (0, 1].
    InvalidAtomicFraction {
        /// Element atomic number.
        atomic_number: u16,
        /// Invalid fraction.
        value: f64,
    },
    /// Atomic fractions must sum to one.
    FractionsDoNotSumToOne {
        /// Supplied sum.
        sum: f64,
    },
    /// Atomic radius must be positive.
    NonPositiveRadius {
        /// Element atomic number.
        atomic_number: u16,
    },
    /// Melting point must be positive.
    NonPositiveMeltingPoint {
        /// Element atomic number.
        atomic_number: u16,
    },
    /// VEC input was negative or non-finite.
    InvalidVec {
        /// Element atomic number.
        atomic_number: u16,
        /// Invalid VEC.
        value: f64,
    },
    /// A pair referenced an element outside the candidate composition.
    PairElementNotInComposition,
    /// Pair data cannot describe self-interaction.
    SelfPair {
        /// Element atomic number.
        atomic_number: u16,
    },
    /// The same unordered binary pair was supplied more than once.
    DuplicatePair {
        /// First atomic number.
        atomic_number_a: u16,
        /// Second atomic number.
        atomic_number_b: u16,
    },
    /// A required pair interaction was absent from a supplied pair table.
    MissingPair {
        /// First atomic number.
        atomic_number_a: u16,
        /// Second atomic number.
        atomic_number_b: u16,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary() -> Vec<HeaElementDescriptor> {
        vec![
            HeaElementDescriptor {
                atomic_number: 22,
                atomic_fraction: 0.5,
                atomic_radius_pm: 147.0,
                melting_point_k: 1941.0,
                vec: Some(4.0),
            },
            HeaElementDescriptor {
                atomic_number: 41,
                atomic_fraction: 0.5,
                atomic_radius_pm: 146.0,
                melting_point_k: 2750.0,
                vec: Some(5.0),
            },
        ]
    }

    #[test]
    fn equiatomic_binary_entropy_is_r_ln_2() {
        let d = calculate_hea_descriptors(&binary(), &[]).unwrap();
        let expected = R_GAS_J_MOL_K * 2.0_f64.ln();
        assert!((d.configurational_entropy_j_mol_k - expected).abs() < 1.0e-10);
    }

    #[test]
    fn pair_enthalpy_convention_reproduces_equiatomic_binary_input() {
        let pairs = [PairMixingEnthalpy {
            atomic_number_a: 22,
            atomic_number_b: 41,
            delta_h_equiatomic_kj_mol: -3.5,
        }];
        let d = calculate_hea_descriptors(&binary(), &pairs).unwrap();
        assert!((d.mixing_enthalpy_kj_mol.unwrap() + 3.5).abs() < 1.0e-12);
    }

    #[test]
    fn missing_vec_remains_unknown_instead_of_being_invented() {
        let mut elements = binary();
        elements[1].vec = None;
        let d = calculate_hea_descriptors(&elements, &[]).unwrap();
        assert_eq!(d.vec, None);
        assert_eq!(d.classic_flags.vec_tendency, None);
    }

    #[test]
    fn partial_pair_table_is_rejected() {
        let elements = vec![
            HeaElementDescriptor {
                atomic_number: 22,
                atomic_fraction: 1.0 / 3.0,
                atomic_radius_pm: 147.0,
                melting_point_k: 1941.0,
                vec: Some(4.0),
            },
            HeaElementDescriptor {
                atomic_number: 40,
                atomic_fraction: 1.0 / 3.0,
                atomic_radius_pm: 160.0,
                melting_point_k: 2128.0,
                vec: Some(4.0),
            },
            HeaElementDescriptor {
                atomic_number: 41,
                atomic_fraction: 1.0 / 3.0,
                atomic_radius_pm: 146.0,
                melting_point_k: 2750.0,
                vec: Some(5.0),
            },
        ];
        let pairs = [PairMixingEnthalpy {
            atomic_number_a: 22,
            atomic_number_b: 40,
            delta_h_equiatomic_kj_mol: 1.0,
        }];
        assert!(matches!(
            calculate_hea_descriptors(&elements, &pairs),
            Err(HeaScreeningError::MissingPair { .. })
        ));
    }

    #[test]
    fn vec_thresholds_are_reported_as_tendencies_only() {
        let mut elements = binary();
        elements[0].vec = Some(5.0);
        elements[1].vec = Some(5.0);
        let d = calculate_hea_descriptors(&elements, &[]).unwrap();
        assert_eq!(
            d.classic_flags.vec_tendency,
            Some(VecStructureTendency::BccLeaning)
        );
    }

    #[test]
    fn wolverine_radius_table_flags_high_size_mismatch_without_stability_claim() {
        // Uses the same radii/fractions currently encoded in high_entropy_alloys.rs.
        // Er VEC is deliberately left unknown here rather than choosing a convention.
        let elements = vec![
            HeaElementDescriptor {
                atomic_number: 22,
                atomic_fraction: 0.25,
                atomic_radius_pm: 147.0,
                melting_point_k: 1941.0,
                vec: Some(4.0),
            },
            HeaElementDescriptor {
                atomic_number: 40,
                atomic_fraction: 0.25,
                atomic_radius_pm: 160.0,
                melting_point_k: 2128.0,
                vec: Some(4.0),
            },
            HeaElementDescriptor {
                atomic_number: 41,
                atomic_fraction: 0.20,
                atomic_radius_pm: 146.0,
                melting_point_k: 2750.0,
                vec: Some(5.0),
            },
            HeaElementDescriptor {
                atomic_number: 73,
                atomic_fraction: 0.15,
                atomic_radius_pm: 146.0,
                melting_point_k: 3290.0,
                vec: Some(5.0),
            },
            HeaElementDescriptor {
                atomic_number: 68,
                atomic_fraction: 0.15,
                atomic_radius_pm: 176.0,
                melting_point_k: 1802.0,
                vec: None,
            },
        ];
        let d = calculate_hea_descriptors(&elements, &[]).unwrap();
        assert!((d.atomic_size_mismatch_pct - 6.9878).abs() < 0.01);
        assert!((d.configurational_entropy_j_mol_k - 13.166).abs() < 0.02);
        assert!(!d.classic_flags.atomic_size_delta_le_6_6_pct);
        assert!(d.classic_flags.entropy_ge_1_5_r);
        assert_eq!(d.vec, None);
        assert_eq!(d.omega, None);
    }
}