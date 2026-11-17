// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Formula-level SymFP reference: spectra, rank templates, leakage and pLS.
//!
//! This crate deliberately returns spectral compatibility evidence only.
//! It does not prove that an input structure or dynamical system possesses
//! the target symmetry.

use std::fmt;

pub const MAX_TEMPLATE_DEGREE: usize = 11;
pub const CRYSTALLOGRAPHIC_POINT_GROUP_COUNT: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointGroupTemplateV1 {
    pub schoenflies: &'static str,
    pub hermann_mauguin: &'static str,
    /// Natural-parity fixed-subspace dimensions for degrees l = 0..11.
    pub fixed_dimensions: [u8; MAX_TEMPLATE_DEGREE + 1],
}

pub const CRYSTALLOGRAPHIC_TEMPLATES_V1: [PointGroupTemplateV1;
    CRYSTALLOGRAPHIC_POINT_GROUP_COUNT] = [
    PointGroupTemplateV1 {
        schoenflies: "C1",
        hermann_mauguin: "1",
        fixed_dimensions: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
    },
    PointGroupTemplateV1 {
        schoenflies: "Ci",
        hermann_mauguin: "-1",
        fixed_dimensions: [1, 0, 5, 0, 9, 0, 13, 0, 17, 0, 21, 0],
    },
    PointGroupTemplateV1 {
        schoenflies: "C2",
        hermann_mauguin: "2",
        fixed_dimensions: [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11],
    },
    PointGroupTemplateV1 {
        schoenflies: "Cs",
        hermann_mauguin: "m",
        fixed_dimensions: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    PointGroupTemplateV1 {
        schoenflies: "C2h",
        hermann_mauguin: "2/m",
        fixed_dimensions: [1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0],
    },
    PointGroupTemplateV1 {
        schoenflies: "D2",
        hermann_mauguin: "222",
        fixed_dimensions: [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5],
    },
    PointGroupTemplateV1 {
        schoenflies: "C2v",
        hermann_mauguin: "mm2",
        fixed_dimensions: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    },
    PointGroupTemplateV1 {
        schoenflies: "D2h",
        hermann_mauguin: "mmm",
        fixed_dimensions: [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0],
    },
    PointGroupTemplateV1 {
        schoenflies: "C4",
        hermann_mauguin: "4",
        fixed_dimensions: [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5],
    },
    PointGroupTemplateV1 {
        schoenflies: "S4",
        hermann_mauguin: "-4",
        fixed_dimensions: [1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6],
    },
    PointGroupTemplateV1 {
        schoenflies: "C4h",
        hermann_mauguin: "4/m",
        fixed_dimensions: [1, 0, 1, 0, 3, 0, 3, 0, 5, 0, 5, 0],
    },
    PointGroupTemplateV1 {
        schoenflies: "D4",
        hermann_mauguin: "422",
        fixed_dimensions: [1, 0, 1, 0, 2, 1, 2, 1, 3, 2, 3, 2],
    },
    PointGroupTemplateV1 {
        schoenflies: "C4v",
        hermann_mauguin: "4mm",
        fixed_dimensions: [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
    },
    PointGroupTemplateV1 {
        schoenflies: "D2d",
        hermann_mauguin: "-42m",
        fixed_dimensions: [1, 0, 1, 1, 2, 1, 2, 2, 3, 2, 3, 3],
    },
    PointGroupTemplateV1 {
        schoenflies: "D4h",
        hermann_mauguin: "4/mmm",
        fixed_dimensions: [1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3, 0],
    },
    PointGroupTemplateV1 {
        schoenflies: "C3",
        hermann_mauguin: "3",
        fixed_dimensions: [1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7],
    },
    PointGroupTemplateV1 {
        schoenflies: "S6",
        hermann_mauguin: "-3",
        fixed_dimensions: [1, 0, 1, 0, 3, 0, 5, 0, 5, 0, 7, 0],
    },
    PointGroupTemplateV1 {
        schoenflies: "D3",
        hermann_mauguin: "32",
        fixed_dimensions: [1, 0, 1, 1, 2, 1, 3, 2, 3, 3, 4, 3],
    },
    PointGroupTemplateV1 {
        schoenflies: "C3v",
        hermann_mauguin: "3m",
        fixed_dimensions: [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    },
    PointGroupTemplateV1 {
        schoenflies: "D3d",
        hermann_mauguin: "-3m",
        fixed_dimensions: [1, 0, 1, 0, 2, 0, 3, 0, 3, 0, 4, 0],
    },
    PointGroupTemplateV1 {
        schoenflies: "C6",
        hermann_mauguin: "6",
        fixed_dimensions: [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3],
    },
    PointGroupTemplateV1 {
        schoenflies: "C3h",
        hermann_mauguin: "-6",
        fixed_dimensions: [1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4],
    },
    PointGroupTemplateV1 {
        schoenflies: "C6h",
        hermann_mauguin: "6/m",
        fixed_dimensions: [1, 0, 1, 0, 1, 0, 3, 0, 3, 0, 3, 0],
    },
    PointGroupTemplateV1 {
        schoenflies: "D6",
        hermann_mauguin: "622",
        fixed_dimensions: [1, 0, 1, 0, 1, 0, 2, 1, 2, 1, 2, 1],
    },
    PointGroupTemplateV1 {
        schoenflies: "C6v",
        hermann_mauguin: "6mm",
        fixed_dimensions: [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    },
    PointGroupTemplateV1 {
        schoenflies: "D3h",
        hermann_mauguin: "-6m2",
        fixed_dimensions: [1, 0, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2],
    },
    PointGroupTemplateV1 {
        schoenflies: "D6h",
        hermann_mauguin: "6/mmm",
        fixed_dimensions: [1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0],
    },
    PointGroupTemplateV1 {
        schoenflies: "T",
        hermann_mauguin: "23",
        fixed_dimensions: [1, 0, 0, 1, 1, 0, 2, 1, 1, 2, 2, 1],
    },
    PointGroupTemplateV1 {
        schoenflies: "Th",
        hermann_mauguin: "m-3",
        fixed_dimensions: [1, 0, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0],
    },
    PointGroupTemplateV1 {
        schoenflies: "O",
        hermann_mauguin: "432",
        fixed_dimensions: [1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    },
    PointGroupTemplateV1 {
        schoenflies: "Td",
        hermann_mauguin: "-43m",
        fixed_dimensions: [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    },
    PointGroupTemplateV1 {
        schoenflies: "Oh",
        hermann_mauguin: "m-3m",
        fixed_dimensions: [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    },
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymFpProfileV1 {
    pub tau_abs: f64,
    pub tau_rel: f64,
    pub epsilon_norm: f64,
    pub epsilon_log: f64,
    pub jacobi_relative_tolerance: f64,
    pub jacobi_max_sweeps: usize,
}

impl Default for SymFpProfileV1 {
    fn default() -> Self {
        Self {
            tau_abs: 0.0,
            tau_rel: 1.0e-6,
            epsilon_norm: 1.0e-10,
            epsilon_log: 1.0e-10,
            jacobi_relative_tolerance: 1.0e-12,
            jacobi_max_sweeps: 80,
        }
    }
}

impl SymFpProfileV1 {
    /// Paper's validation-selected setting for degrees 0..6.
    pub fn paper_degree_6() -> Self {
        Self {
            tau_rel: 3.0e-5,
            ..Self::default()
        }
    }

    /// Paper's validation-selected setting for degrees 0..9.
    pub fn paper_degree_9() -> Self {
        Self {
            tau_rel: 1.0e-6,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymFpErrorV1 {
    InvalidProfile,
    EmptyDegreeSet,
    MissingDegreeZero,
    DegreesNotStrictlyIncreasing,
    ZeroDimension,
    MatrixLengthMismatch,
    NonFiniteMatrix,
    NonFiniteSpectrum,
    NegativeSingularValue,
    SpectrumNotDescending,
    SingularSpectrumLengthMismatch,
    SvdDidNotConverge { sweeps: usize, residual_bits: u64 },
    NonFiniteDerivedValue,
    UnknownDegree { degree: usize },
    EmptyCandidateFamily,
    DuplicateCandidateName,
}

impl fmt::Display for SymFpErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SymFpErrorV1 {}

#[derive(Debug, Clone, PartialEq)]
pub struct SvdReceiptV1 {
    pub singular_values: Vec<f64>,
    pub sweeps: usize,
    pub max_normalized_off_diagonal: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpectrumBlockV1 {
    pub degree: usize,
    pub singular_values: Vec<f64>,
    pub svd_sweeps: Option<usize>,
    pub svd_residual: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SymFpReceiptV1 {
    pub degrees: Vec<usize>,
    pub tau: f64,
    pub rank_sequence: Vec<usize>,
    pub spectra: Vec<SpectrumBlockV1>,
    pub profile: SymFpProfileV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointGroupMatchV1 {
    pub minimum_l1_distance: usize,
    pub matched_schoenflies: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DegreeLeakageV1 {
    pub degree: usize,
    pub fixed_subspace_dimension: usize,
    pub spectral_tail_energy: f64,
    pub total_energy: f64,
    pub leakage_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TargetLeakageV1 {
    pub target_schoenflies: &'static str,
    pub degrees: Vec<DegreeLeakageV1>,
    pub pooled_leakage_score: f64,
    pub p_ls: f64,
    /// This is compatibility with rank constraints, not proof of symmetry.
    pub authority: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RepresentationBlockV1 {
    pub degree: usize,
    pub rows: usize,
    pub columns: usize,
    /// Row-major representation matrix with shape rows × columns.
    pub values: Vec<f64>,
}

fn validate_profile(profile: SymFpProfileV1) -> Result<(), SymFpErrorV1> {
    if !profile.tau_abs.is_finite()
        || profile.tau_abs < 0.0
        || !profile.tau_rel.is_finite()
        || profile.tau_rel < 0.0
        || !profile.epsilon_norm.is_finite()
        || profile.epsilon_norm <= 0.0
        || !profile.epsilon_log.is_finite()
        || profile.epsilon_log <= 0.0
        || !profile.jacobi_relative_tolerance.is_finite()
        || profile.jacobi_relative_tolerance <= 0.0
        || profile.jacobi_relative_tolerance >= 1.0
        || profile.jacobi_max_sweeps == 0
    {
        return Err(SymFpErrorV1::InvalidProfile);
    }
    Ok(())
}

fn validate_degrees(degrees: &[usize]) -> Result<(), SymFpErrorV1> {
    if degrees.is_empty() {
        return Err(SymFpErrorV1::EmptyDegreeSet);
    }
    if degrees[0] != 0 {
        return Err(SymFpErrorV1::MissingDegreeZero);
    }
    if degrees.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(SymFpErrorV1::DegreesNotStrictlyIncreasing);
    }
    if degrees.iter().any(|degree| *degree > MAX_TEMPLATE_DEGREE) {
        let degree = *degrees
            .iter()
            .find(|degree| **degree > MAX_TEMPLATE_DEGREE)
            .unwrap();
        return Err(SymFpErrorV1::UnknownDegree { degree });
    }
    Ok(())
}

/// Compute singular values with a scaled one-sided Jacobi algorithm.
///
/// The input is transposed when it is wide, so the iterative matrix is always
/// tall or square. No normal-equation matrix A^T A is formed.
pub fn one_sided_jacobi_svd(
    rows: usize,
    columns: usize,
    values: &[f64],
    profile: SymFpProfileV1,
) -> Result<SvdReceiptV1, SymFpErrorV1> {
    validate_profile(profile)?;
    if rows == 0 || columns == 0 {
        return Err(SymFpErrorV1::ZeroDimension);
    }
    let expected = rows
        .checked_mul(columns)
        .ok_or(SymFpErrorV1::NonFiniteDerivedValue)?;
    if values.len() != expected {
        return Err(SymFpErrorV1::MatrixLengthMismatch);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SymFpErrorV1::NonFiniteMatrix);
    }

    let scale = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok(SvdReceiptV1 {
            singular_values: vec![0.0; rows.min(columns)],
            sweeps: 0,
            max_normalized_off_diagonal: 0.0,
        });
    }

    let tall_rows = rows.max(columns);
    let tall_columns = rows.min(columns);
    let mut work = vec![vec![0.0; tall_columns]; tall_rows];

    if rows >= columns {
        for row in 0..rows {
            for column in 0..columns {
                work[row][column] = values[row * columns + column] / scale;
            }
        }
    } else {
        for row in 0..rows {
            for column in 0..columns {
                work[column][row] = values[row * columns + column] / scale;
            }
        }
    }

    let mut completed_sweeps = 0usize;
    let mut converged = tall_columns <= 1;
    for sweep in 0..profile.jacobi_max_sweeps {
        let mut changed = false;

        for left in 0..tall_columns {
            for right in (left + 1)..tall_columns {
                let mut alpha = 0.0_f64;
                let mut beta = 0.0_f64;
                let mut gamma = 0.0_f64;
                for row in work.iter().take(tall_rows) {
                    let a = row[left];
                    let b = row[right];
                    alpha += a * a;
                    beta += b * b;
                    gamma += a * b;
                }

                let product = alpha * beta;
                if product == 0.0 {
                    continue;
                }
                let denominator = product.sqrt();
                let correlation = gamma.abs() / denominator;
                if correlation <= profile.jacobi_relative_tolerance {
                    continue;
                }

                let tau = (beta - alpha) / (2.0 * gamma);
                let root = tau.hypot(1.0);
                let tangent = if tau >= 0.0 {
                    1.0 / (tau + root)
                } else {
                    -1.0 / (-tau + root)
                };
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = cosine * tangent;

                for row in work.iter_mut().take(tall_rows) {
                    let a = row[left];
                    let b = row[right];
                    row[left] = cosine * a - sine * b;
                    row[right] = sine * a + cosine * b;
                }
                changed = true;
            }
        }

        completed_sweeps = sweep + 1;
        if !changed {
            converged = true;
            break;
        }
    }

    // Recompute the final orthogonality defect after the last rotation sweep.
    let mut maximum_correlation = 0.0_f64;
    for left in 0..tall_columns {
        for right in (left + 1)..tall_columns {
            let mut alpha = 0.0_f64;
            let mut beta = 0.0_f64;
            let mut gamma = 0.0_f64;
            for row in work.iter().take(tall_rows) {
                let a = row[left];
                let b = row[right];
                alpha += a * a;
                beta += b * b;
                gamma += a * b;
            }
            if alpha > 0.0 && beta > 0.0 {
                maximum_correlation = maximum_correlation.max(gamma.abs() / (alpha * beta).sqrt());
            }
        }
    }
    if maximum_correlation <= profile.jacobi_relative_tolerance {
        converged = true;
    }
    if !converged {
        return Err(SymFpErrorV1::SvdDidNotConverge {
            sweeps: completed_sweeps,
            residual_bits: maximum_correlation.to_bits(),
        });
    }

    let mut singular_values = Vec::with_capacity(tall_columns);
    for column in 0..tall_columns {
        let norm = work
            .iter()
            .fold(0.0_f64, |accumulator, row| accumulator.hypot(row[column]));
        let singular = norm * scale;
        if !singular.is_finite() {
            return Err(SymFpErrorV1::NonFiniteDerivedValue);
        }
        singular_values.push(singular);
    }
    singular_values.sort_by(|left, right| right.total_cmp(left));

    Ok(SvdReceiptV1 {
        singular_values,
        sweeps: completed_sweeps,
        max_normalized_off_diagonal: maximum_correlation,
    })
}

/// Compute an SVD for each degree block and then apply the paper's SymFP rules.
pub fn symfp_from_representation_blocks(
    blocks: &[RepresentationBlockV1],
    profile: SymFpProfileV1,
) -> Result<SymFpReceiptV1, SymFpErrorV1> {
    validate_profile(profile)?;
    let degrees: Vec<usize> = blocks.iter().map(|block| block.degree).collect();
    validate_degrees(&degrees)?;

    let mut spectra = Vec::with_capacity(blocks.len());
    for block in blocks {
        if block.rows != 2 * block.degree + 1 {
            return Err(SymFpErrorV1::MatrixLengthMismatch);
        }
        let svd = one_sided_jacobi_svd(block.rows, block.columns, &block.values, profile)?;
        spectra.push(SpectrumBlockV1 {
            degree: block.degree,
            singular_values: svd.singular_values,
            svd_sweeps: Some(svd.sweeps),
            svd_residual: Some(svd.max_normalized_off_diagonal),
        });
    }
    analyze_spectra(&spectra, profile)
}

/// Apply the paper's numerical-rank threshold to already computed spectra.
pub fn analyze_spectra(
    spectra: &[SpectrumBlockV1],
    profile: SymFpProfileV1,
) -> Result<SymFpReceiptV1, SymFpErrorV1> {
    validate_profile(profile)?;
    let degrees: Vec<usize> = spectra.iter().map(|block| block.degree).collect();
    validate_degrees(&degrees)?;

    for block in spectra {
        if block.singular_values.is_empty() || block.singular_values.len() > 2 * block.degree + 1 {
            return Err(SymFpErrorV1::SingularSpectrumLengthMismatch);
        }
        if block.singular_values.iter().any(|value| !value.is_finite()) {
            return Err(SymFpErrorV1::NonFiniteSpectrum);
        }
        if block.singular_values.iter().any(|value| *value < 0.0) {
            return Err(SymFpErrorV1::NegativeSingularValue);
        }
        if block
            .singular_values
            .windows(2)
            .any(|pair| pair[0] < pair[1])
        {
            return Err(SymFpErrorV1::SpectrumNotDescending);
        }
    }

    let leading_degree_zero = spectra[0].singular_values[0];
    let tau = profile.tau_abs.max(profile.tau_rel * leading_degree_zero);
    if !tau.is_finite() {
        return Err(SymFpErrorV1::NonFiniteDerivedValue);
    }

    let rank_sequence = spectra
        .iter()
        .map(|block| {
            block
                .singular_values
                .iter()
                .filter(|value| **value > tau)
                .count()
        })
        .collect();

    Ok(SymFpReceiptV1 {
        degrees,
        tau,
        rank_sequence,
        spectra: spectra.to_vec(),
        profile,
    })
}

pub fn classify_rank_template(
    receipt: &SymFpReceiptV1,
    candidates: &[PointGroupTemplateV1],
) -> Result<PointGroupMatchV1, SymFpErrorV1> {
    if candidates.is_empty() {
        return Err(SymFpErrorV1::EmptyCandidateFamily);
    }
    let mut names = std::collections::BTreeSet::new();
    for candidate in candidates {
        if !names.insert(candidate.schoenflies) {
            return Err(SymFpErrorV1::DuplicateCandidateName);
        }
    }

    let mut minimum = usize::MAX;
    let mut matches = Vec::new();
    for candidate in candidates {
        let mut distance = 0usize;
        for (index, degree) in receipt.degrees.iter().enumerate() {
            let expected = candidate.fixed_dimensions[*degree] as usize;
            distance += expected.abs_diff(receipt.rank_sequence[index]);
        }
        if distance < minimum {
            minimum = distance;
            matches.clear();
            matches.push(candidate.schoenflies);
        } else if distance == minimum {
            matches.push(candidate.schoenflies);
        }
    }

    Ok(PointGroupMatchV1 {
        minimum_l1_distance: minimum,
        matched_schoenflies: matches,
    })
}

pub fn template_by_name(name: &str) -> Option<&'static PointGroupTemplateV1> {
    CRYSTALLOGRAPHIC_TEMPLATES_V1
        .iter()
        .find(|candidate| candidate.schoenflies == name)
}

pub fn target_leakage(
    receipt: &SymFpReceiptV1,
    target: &PointGroupTemplateV1,
) -> Result<TargetLeakageV1, SymFpErrorV1> {
    let mut degree_results = Vec::with_capacity(receipt.spectra.len());
    let mut pooled = 0.0_f64;

    for block in &receipt.spectra {
        let fixed_dimension = target.fixed_dimensions[block.degree] as usize;
        let mut total_energy = 0.0_f64;
        for singular in &block.singular_values {
            total_energy += singular * singular;
        }
        let mut tail_energy = 0.0_f64;
        for singular in block.singular_values.iter().skip(fixed_dimension) {
            tail_energy += singular * singular;
        }
        let denominator = total_energy + receipt.profile.epsilon_norm;
        if !total_energy.is_finite() || !tail_energy.is_finite() || !denominator.is_finite() {
            return Err(SymFpErrorV1::NonFiniteDerivedValue);
        }
        let leakage = tail_energy / denominator;
        if !leakage.is_finite() {
            return Err(SymFpErrorV1::NonFiniteDerivedValue);
        }
        pooled += leakage;
        degree_results.push(DegreeLeakageV1 {
            degree: block.degree,
            fixed_subspace_dimension: fixed_dimension,
            spectral_tail_energy: tail_energy,
            total_energy,
            leakage_score: leakage,
        });
    }

    let pooled_leakage_score = pooled / degree_results.len() as f64;
    let p_ls = -(pooled_leakage_score + receipt.profile.epsilon_log).log10();
    if !pooled_leakage_score.is_finite() || !p_ls.is_finite() {
        return Err(SymFpErrorV1::NonFiniteDerivedValue);
    }

    Ok(TargetLeakageV1 {
        target_schoenflies: target.schoenflies,
        degrees: degree_results,
        pooled_leakage_score,
        p_ls,
        authority: "SPECTRAL_COMPATIBILITY_ONLY_NOT_A_SYMMETRY_CERTIFICATE",
    })
}

pub fn templates_are_ordered_for_subgroup(
    subgroup: &PointGroupTemplateV1,
    supergroup: &PointGroupTemplateV1,
    degrees: &[usize],
) -> bool {
    degrees
        .iter()
        .all(|degree| subgroup.fixed_dimensions[*degree] >= supergroup.fixed_dimensions[*degree])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "left={left:.16e}, right={right:.16e}, tolerance={tolerance:.3e}"
        );
    }

    fn spectra_for_rank_template(name: &str, maximum_degree: usize) -> Vec<SpectrumBlockV1> {
        let template = template_by_name(name).unwrap();
        (0..=maximum_degree)
            .map(|degree| SpectrumBlockV1 {
                degree,
                singular_values: (0..=2 * degree)
                    .map(|index| {
                        if index < template.fixed_dimensions[degree] as usize {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
                svd_sweeps: None,
                svd_residual: None,
            })
            .collect()
    }

    #[test]
    fn all_32_published_templates_are_valid_and_l0_to_l9_is_injective() {
        assert_eq!(CRYSTALLOGRAPHIC_TEMPLATES_V1.len(), 32);
        for template in &CRYSTALLOGRAPHIC_TEMPLATES_V1 {
            for degree in 0..=MAX_TEMPLATE_DEGREE {
                assert!(template.fixed_dimensions[degree] as usize <= 2 * degree + 1);
            }
        }

        let degrees: Vec<usize> = (0..=9).collect();
        let mut rows = std::collections::BTreeSet::new();
        for template in &CRYSTALLOGRAPHIC_TEMPLATES_V1 {
            let row: Vec<u8> = degrees
                .iter()
                .map(|degree| template.fixed_dimensions[*degree])
                .collect();
            assert!(
                rows.insert(row),
                "duplicate template: {}",
                template.schoenflies
            );
        }
    }

    #[test]
    fn l0_to_l6_has_exactly_the_two_published_template_collisions() {
        let degrees: Vec<usize> = (0..=6).collect();
        let mut counts = std::collections::BTreeMap::<Vec<u8>, Vec<&str>>::new();
        for template in &CRYSTALLOGRAPHIC_TEMPLATES_V1 {
            let row: Vec<u8> = degrees
                .iter()
                .map(|degree| template.fixed_dimensions[*degree])
                .collect();
            counts.entry(row).or_default().push(template.schoenflies);
        }
        let ties: Vec<Vec<&str>> = counts
            .into_values()
            .filter(|names| names.len() > 1)
            .collect();
        assert_eq!(ties.len(), 2);
        assert!(ties.iter().any(|names| names == &vec!["D6", "D6h"]));
        assert!(ties.iter().any(|names| names == &vec!["O", "Oh"]));
    }

    #[test]
    fn paper_threshold_profiles_are_reproduced() {
        close(SymFpProfileV1::paper_degree_6().tau_rel, 3.0e-5, 0.0);
        close(SymFpProfileV1::paper_degree_9().tau_rel, 1.0e-6, 0.0);
        close(SymFpProfileV1::default().epsilon_norm, 1.0e-10, 0.0);
        close(SymFpProfileV1::default().epsilon_log, 1.0e-10, 0.0);
    }

    #[test]
    fn rank_template_matching_recovers_each_exact_crystallographic_template_at_l9() {
        let profile = SymFpProfileV1::paper_degree_9();
        for template in &CRYSTALLOGRAPHIC_TEMPLATES_V1 {
            let receipt =
                analyze_spectra(&spectra_for_rank_template(template.schoenflies, 9), profile)
                    .unwrap();
            let matched = classify_rank_template(&receipt, &CRYSTALLOGRAPHIC_TEMPLATES_V1).unwrap();
            assert_eq!(matched.minimum_l1_distance, 0);
            assert_eq!(matched.matched_schoenflies, vec![template.schoenflies]);
        }
    }

    #[test]
    fn l6_classifier_returns_both_minimizers_for_published_collisions() {
        for target in ["D6", "D6h", "O", "Oh"] {
            let receipt = analyze_spectra(
                &spectra_for_rank_template(target, 6),
                SymFpProfileV1::paper_degree_6(),
            )
            .unwrap();
            let matched = classify_rank_template(&receipt, &CRYSTALLOGRAPHIC_TEMPLATES_V1).unwrap();
            assert_eq!(matched.minimum_l1_distance, 0);
            assert_eq!(matched.matched_schoenflies.len(), 2);
        }
    }

    #[test]
    fn c2_spectrum_can_have_zero_leakage_against_cs_without_proving_cs() {
        let receipt = analyze_spectra(
            &spectra_for_rank_template("C2", 9),
            SymFpProfileV1::paper_degree_9(),
        )
        .unwrap();
        let cs = template_by_name("Cs").unwrap();
        let leakage = target_leakage(&receipt, cs).unwrap();

        assert_eq!(leakage.pooled_leakage_score, 0.0);
        close(leakage.p_ls, 10.0, 1.0e-12);
        assert_eq!(
            leakage.authority,
            "SPECTRAL_COMPATIBILITY_ONLY_NOT_A_SYMMETRY_CERTIFICATE"
        );
        let classification =
            classify_rank_template(&receipt, &CRYSTALLOGRAPHIC_TEMPLATES_V1).unwrap();
        assert_eq!(classification.matched_schoenflies, vec!["C2"]);
    }

    #[test]
    fn leakage_and_p_ls_are_monotone_for_a_declared_subgroup_pair() {
        let receipt = analyze_spectra(
            &spectra_for_rank_template("C1", 9),
            SymFpProfileV1::paper_degree_9(),
        )
        .unwrap();
        let c2 = target_leakage(&receipt, template_by_name("C2").unwrap()).unwrap();
        let d4 = target_leakage(&receipt, template_by_name("D4").unwrap()).unwrap();
        let degrees: Vec<usize> = (0..=9).collect();

        assert!(templates_are_ordered_for_subgroup(
            template_by_name("C2").unwrap(),
            template_by_name("D4").unwrap(),
            &degrees
        ));
        assert!(c2.pooled_leakage_score <= d4.pooled_leakage_score);
        assert!(c2.p_ls >= d4.p_ls);
    }

    #[test]
    fn rank_threshold_uses_degree_zero_scale_for_every_degree() {
        let blocks = vec![
            SpectrumBlockV1 {
                degree: 0,
                singular_values: vec![10.0],
                svd_sweeps: None,
                svd_residual: None,
            },
            SpectrumBlockV1 {
                degree: 1,
                singular_values: vec![1.0, 0.0002, 0.00005],
                svd_sweeps: None,
                svd_residual: None,
            },
        ];
        let receipt = analyze_spectra(
            &blocks,
            SymFpProfileV1 {
                tau_abs: 0.0,
                tau_rel: 1.0e-5,
                ..SymFpProfileV1::default()
            },
        )
        .unwrap();
        close(receipt.tau, 0.0001, 1.0e-16);
        assert_eq!(receipt.rank_sequence, vec![1, 2]);
    }

    #[test]
    fn l1_template_matching_returns_all_exact_ties() {
        let receipt = analyze_spectra(
            &spectra_for_rank_template("D6", 6),
            SymFpProfileV1::paper_degree_6(),
        )
        .unwrap();
        let match_result =
            classify_rank_template(&receipt, &CRYSTALLOGRAPHIC_TEMPLATES_V1).unwrap();
        assert_eq!(match_result.matched_schoenflies, vec!["D6", "D6h"]);
    }

    #[test]
    fn zero_block_energy_has_zero_leakage_and_maximum_regularized_pls() {
        let blocks = (0..=2)
            .map(|degree| SpectrumBlockV1 {
                degree,
                singular_values: vec![0.0; 2 * degree + 1],
                svd_sweeps: None,
                svd_residual: None,
            })
            .collect::<Vec<_>>();
        let receipt = analyze_spectra(&blocks, SymFpProfileV1::default()).unwrap();
        let score = target_leakage(&receipt, template_by_name("D4").unwrap()).unwrap();
        assert_eq!(score.pooled_leakage_score, 0.0);
        close(score.p_ls, 10.0, 1.0e-12);
    }

    #[test]
    fn one_sided_jacobi_svd_recovers_known_rectangular_spectrum() {
        let receipt = one_sided_jacobi_svd(
            2,
            3,
            &[3.0, 0.0, 0.0, 0.0, 4.0, 0.0],
            SymFpProfileV1::default(),
        )
        .unwrap();
        assert_eq!(receipt.singular_values.len(), 2);
        close(receipt.singular_values[0], 4.0, 1.0e-12);
        close(receipt.singular_values[1], 3.0, 1.0e-12);
    }

    #[test]
    fn singular_spectrum_is_invariant_under_orthogonal_left_action() {
        let matrix = [1.0, 2.0, 3.0, -2.0, 1.0, 0.5, 0.25, -1.0, 2.0];
        let angle: f64 = 0.731;
        let cosine = angle.cos();
        let sine = angle.sin();
        let mut rotated = matrix;
        for column in 0..3 {
            let x = matrix[column];
            let y = matrix[3 + column];
            rotated[column] = cosine * x - sine * y;
            rotated[3 + column] = sine * x + cosine * y;
        }
        let profile = SymFpProfileV1::default();
        let first = one_sided_jacobi_svd(3, 3, &matrix, profile).unwrap();
        let second = one_sided_jacobi_svd(3, 3, &rotated, profile).unwrap();
        for (left, right) in first
            .singular_values
            .iter()
            .zip(second.singular_values.iter())
        {
            close(*left, *right, 2.0e-11);
        }
    }

    #[test]
    fn rank_deficient_and_zero_matrices_return_finite_sorted_spectra() {
        let rank_one = one_sided_jacobi_svd(
            3,
            2,
            &[1.0, 2.0, 2.0, 4.0, -1.0, -2.0],
            SymFpProfileV1::default(),
        )
        .unwrap();
        assert!(rank_one.singular_values[0] > 0.0);
        assert!(rank_one.singular_values[1] < 1.0e-12);

        let zero = one_sided_jacobi_svd(3, 2, &[0.0; 6], SymFpProfileV1::default()).unwrap();
        assert_eq!(zero.singular_values, vec![0.0, 0.0]);
    }

    #[test]
    fn malformed_and_nonfinite_spectra_fail_closed() {
        let profile = SymFpProfileV1::default();
        assert_eq!(
            analyze_spectra(&[], profile),
            Err(SymFpErrorV1::EmptyDegreeSet)
        );
        assert_eq!(
            analyze_spectra(
                &[SpectrumBlockV1 {
                    degree: 1,
                    singular_values: vec![1.0],
                    svd_sweeps: None,
                    svd_residual: None,
                }],
                profile
            ),
            Err(SymFpErrorV1::MissingDegreeZero)
        );
        assert_eq!(
            analyze_spectra(
                &[
                    SpectrumBlockV1 {
                        degree: 0,
                        singular_values: vec![1.0],
                        svd_sweeps: None,
                        svd_residual: None,
                    },
                    SpectrumBlockV1 {
                        degree: 1,
                        singular_values: vec![0.1, 0.2],
                        svd_sweeps: None,
                        svd_residual: None,
                    }
                ],
                profile
            ),
            Err(SymFpErrorV1::SpectrumNotDescending)
        );
        assert_eq!(
            one_sided_jacobi_svd(1, 2, &[f64::NAN, 0.0], profile),
            Err(SymFpErrorV1::NonFiniteMatrix)
        );
    }

    #[test]
    fn spectrum_scales_with_the_input_matrix() {
        let matrix = [1.0, 2.0, 3.0, -2.0, 1.0, 0.5];
        let scaled: Vec<f64> = matrix.iter().map(|value| value * 7.0).collect();
        let profile = SymFpProfileV1::default();
        let first = one_sided_jacobi_svd(2, 3, &matrix, profile).unwrap();
        let second = one_sided_jacobi_svd(2, 3, &scaled, profile).unwrap();
        for (left, right) in first
            .singular_values
            .iter()
            .zip(second.singular_values.iter())
        {
            close(*right, *left * 7.0, 1.0e-11);
        }
    }
}
