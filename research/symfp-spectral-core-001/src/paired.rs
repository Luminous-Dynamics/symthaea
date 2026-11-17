// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Reproducible spectral perturbation bounds derived from paired representation
//! matrices. The declared reference matrices are bound to the SymFP receipt;
//! observed-minus-reference matrices produce outward-rounded Frobenius upper
//! bounds, which dominate the spectral/operator norm.
//!
//! This certifies only the numerical difference between the two supplied
//! representation artifacts. It does not certify that the reference is the
//! physically true representation or that a sample-level bound generalizes.

use std::collections::BTreeSet;
use std::fmt;

use super::robust::{
    RobustPointGroupMatchV1, RobustSymFpErrorV1, classify_rank_templates_with_perturbation_bounds,
};
use super::{
    PointGroupTemplateV1, SymFpErrorV1, SymFpProfileV1, SymFpReceiptV1,
    one_sided_jacobi_svd,
};

#[derive(Debug, Clone, PartialEq)]
pub struct PairedRepresentationBlockV1 {
    pub degree: usize,
    pub rows: usize,
    pub columns: usize,
    pub reference_representation_id: String,
    pub observed_representation_id: String,
    pub reference_values: Vec<f64>,
    pub observed_values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpectralOperatorBoundV1 {
    pub degree: usize,
    pub rows: usize,
    pub columns: usize,
    pub reference_representation_id: String,
    pub observed_representation_id: String,
    pub frobenius_upper_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedRobustSymFpReceiptV1 {
    pub schema_version: u32,
    pub reference_receipt: SymFpReceiptV1,
    pub candidates: Vec<PointGroupTemplateV1>,
    pub paired_blocks: Vec<PairedRepresentationBlockV1>,
    pub derived_operator_bounds: Vec<SpectralOperatorBoundV1>,
    pub classification: RobustPointGroupMatchV1,
    pub authority: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairedSymFpErrorV1 {
    InvalidIdentifier,
    EmptyBlocks,
    BlockCountMismatch,
    DegreeMismatch,
    InvalidDimension,
    MatrixLengthMismatch,
    NonFiniteMatrix,
    PerturbationOverflow,
    ReferenceSvdProvenanceMissing,
    ReferenceSpectrumMismatch,
    ReferenceProfileMismatch,
    DuplicateDegree,
    DuplicateCandidate,
    BaseSvd(SymFpErrorV1),
    RobustClassifier(RobustSymFpErrorV1),
}

impl fmt::Display for PairedSymFpErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PairedSymFpErrorV1 {}

fn outward_add_nonnegative(left: f64, right: f64) -> Result<f64, PairedSymFpErrorV1> {
    let sum = left + right;
    if !sum.is_finite() {
        return Err(PairedSymFpErrorV1::PerturbationOverflow);
    }
    let upper = sum.next_up();
    if upper.is_finite() {
        Ok(upper)
    } else {
        Err(PairedSymFpErrorV1::PerturbationOverflow)
    }
}

fn outward_square(value: f64) -> Result<f64, PairedSymFpErrorV1> {
    let square = value * value;
    if !square.is_finite() {
        return Err(PairedSymFpErrorV1::PerturbationOverflow);
    }
    let upper = square.next_up();
    if upper.is_finite() {
        Ok(upper)
    } else {
        Err(PairedSymFpErrorV1::PerturbationOverflow)
    }
}

/// Derive an outward-rounded Frobenius upper bound from nonnegative
/// entrywise upper bounds on an observed-minus-reference matrix. Since
/// ||A||₂ <= ||A||F, this is a valid operator-norm envelope. Every square,
/// accumulation and final square root is rounded upward by one representable
/// f64 value. The paired wrapper also rounds each subtraction outward before
/// calling this function.
pub fn frobenius_operator_norm_upper_bound(
    delta_values: &[f64],
) -> Result<f64, PairedSymFpErrorV1> {
    if delta_values.iter().any(|value| !value.is_finite()) {
        return Err(PairedSymFpErrorV1::NonFiniteMatrix);
    }
    if delta_values.iter().all(|value| *value == 0.0) {
        return Ok(0.0);
    }
    let mut squared_sum_upper = 0.0_f64;
    for value in delta_values {
        let square_upper = outward_square(value.abs())?;
        squared_sum_upper = outward_add_nonnegative(squared_sum_upper, square_upper)?;
    }
    let root = squared_sum_upper.sqrt();
    if !root.is_finite() {
        return Err(PairedSymFpErrorV1::PerturbationOverflow);
    }
    let upper = root.next_up();
    if upper.is_finite() {
        Ok(upper)
    } else {
        Err(PairedSymFpErrorV1::PerturbationOverflow)
    }
}

/// Return an outward-rounded bound on the exact real difference of two
/// supplied finite f64 values. Subtraction itself is rounded to nearest, so
/// advance its absolute result by one representable value before norm
/// accumulation. Identical input values retain an exact zero difference.
fn outward_abs_difference(left: f64, right: f64) -> Result<f64, PairedSymFpErrorV1> {
    let difference = left - right;
    if !difference.is_finite() {
        return Err(PairedSymFpErrorV1::PerturbationOverflow);
    }
    if left == right {
        return Ok(0.0);
    }
    let upper = difference.abs().next_up();
    if upper.is_finite() {
        Ok(upper)
    } else {
        Err(PairedSymFpErrorV1::PerturbationOverflow)
    }
}

fn validate_paired_blocks(
    reference_receipt: &SymFpReceiptV1,
    paired_blocks: &[PairedRepresentationBlockV1],
) -> Result<Vec<SpectralOperatorBoundV1>, PairedSymFpErrorV1> {
    if paired_blocks.is_empty() {
        return Err(PairedSymFpErrorV1::EmptyBlocks);
    }
    if paired_blocks.len() != reference_receipt.spectra.len() {
        return Err(PairedSymFpErrorV1::BlockCountMismatch);
    }

    let mut degrees = BTreeSet::new();
    let mut expected_reference_id: Option<&str> = None;
    let mut expected_observed_id: Option<&str> = None;
    let mut bounds = Vec::with_capacity(paired_blocks.len());

    for (receipt_block, paired) in reference_receipt.spectra.iter().zip(paired_blocks) {
        if !degrees.insert(paired.degree) {
            return Err(PairedSymFpErrorV1::DuplicateDegree);
        }
        if paired.degree != receipt_block.degree {
            return Err(PairedSymFpErrorV1::DegreeMismatch);
        }
        if paired.reference_representation_id.trim().is_empty()
            || paired.observed_representation_id.trim().is_empty()
        {
            return Err(PairedSymFpErrorV1::InvalidIdentifier);
        }
        if let Some(reference_id) = expected_reference_id {
            if reference_id != paired.reference_representation_id {
                return Err(PairedSymFpErrorV1::InvalidIdentifier);
            }
        } else {
            expected_reference_id = Some(&paired.reference_representation_id);
        }
        if let Some(observed_id) = expected_observed_id {
            if observed_id != paired.observed_representation_id {
                return Err(PairedSymFpErrorV1::InvalidIdentifier);
            }
        } else {
            expected_observed_id = Some(&paired.observed_representation_id);
        }
        if paired.rows == 0 || paired.columns == 0 {
            return Err(PairedSymFpErrorV1::InvalidDimension);
        }
        let expected_length = paired
            .rows
            .checked_mul(paired.columns)
            .ok_or(PairedSymFpErrorV1::MatrixLengthMismatch)?;
        if paired.reference_values.len() != expected_length
            || paired.observed_values.len() != expected_length
        {
            return Err(PairedSymFpErrorV1::MatrixLengthMismatch);
        }
        if paired
            .reference_values
            .iter()
            .chain(paired.observed_values.iter())
            .any(|value| !value.is_finite())
        {
            return Err(PairedSymFpErrorV1::NonFiniteMatrix);
        }
        if receipt_block.svd_sweeps.is_none() || receipt_block.svd_residual.is_none() {
            return Err(PairedSymFpErrorV1::ReferenceSvdProvenanceMissing);
        }

        let svd = one_sided_jacobi_svd(
            paired.rows,
            paired.columns,
            &paired.reference_values,
            reference_receipt.profile,
        )
        .map_err(PairedSymFpErrorV1::BaseSvd)?;
        if receipt_block.singular_values.len() != svd.singular_values.len()
            || receipt_block
                .singular_values
                .iter()
                .zip(svd.singular_values.iter())
                .any(|(left, right)| left.to_bits() != right.to_bits())
            || receipt_block.svd_sweeps != Some(svd.sweeps)
            || receipt_block.svd_residual != Some(svd.max_normalized_off_diagonal)
        {
            return Err(PairedSymFpErrorV1::ReferenceSpectrumMismatch);
        }

        let mut delta_values = Vec::with_capacity(expected_length);
        for (observed, reference) in paired
            .observed_values
            .iter()
            .zip(paired.reference_values.iter())
        {
            delta_values.push(outward_abs_difference(*observed, *reference)?);
        }
        let frobenius_upper_bound = frobenius_operator_norm_upper_bound(&delta_values)?;
        bounds.push(SpectralOperatorBoundV1 {
            degree: paired.degree,
            rows: paired.rows,
            columns: paired.columns,
            reference_representation_id: paired.reference_representation_id.clone(),
            observed_representation_id: paired.observed_representation_id.clone(),
            frobenius_upper_bound,
        });
    }
    Ok(bounds)
}

/// Bind the reference spectral receipt to its source matrices, derive one
/// perturbation bound per degree, and pass only those derived bounds to the
/// robust SymFP classifier.
pub fn classify_paired_representations(
    reference_receipt: &SymFpReceiptV1,
    candidates: &[PointGroupTemplateV1],
    paired_blocks: &[PairedRepresentationBlockV1],
) -> Result<PairedRobustSymFpReceiptV1, PairedSymFpErrorV1> {
    let derived_operator_bounds = validate_paired_blocks(reference_receipt, paired_blocks)?;
    let bounds: Vec<f64> = derived_operator_bounds
        .iter()
        .map(|bound| bound.frobenius_upper_bound)
        .collect();
    let classification = classify_rank_templates_with_perturbation_bounds(
        reference_receipt,
        candidates,
        &bounds,
    )
    .map_err(PairedSymFpErrorV1::RobustClassifier)?;
    let mut candidate_names = BTreeSet::new();
    for candidate in candidates {
        if !candidate_names.insert(candidate.schoenflies) {
            return Err(PairedSymFpErrorV1::DuplicateCandidate);
        }
    }
    Ok(PairedRobustSymFpReceiptV1 {
        schema_version: 1,
        reference_receipt: reference_receipt.clone(),
        candidates: candidates.to_vec(),
        paired_blocks: paired_blocks.to_vec(),
        derived_operator_bounds,
        classification,
        authority: "PAIRED_SPECTRAL_COMPATIBILITY_ONLY_NOT_A_SYMMETRY_CERTIFICATE",
    })
}

/// Independent deterministic replay of the paired matrices, reference SVD,
/// outward-rounded perturbation envelopes and robust template decision.
pub fn verify_paired_robust_symfp_receipt(
    receipt: &PairedRobustSymFpReceiptV1,
) -> Result<(), PairedSymFpErrorV1> {
    let replayed = classify_paired_representations(
        &receipt.reference_receipt,
        &receipt.candidates,
        &receipt.paired_blocks,
    )?;
    if *receipt != replayed {
        return Err(PairedSymFpErrorV1::ReferenceSpectrumMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{
        RepresentationBlockV1, SymFpProfileV1, symfp_from_representation_blocks, template_by_name,
    };
    use super::*;

    fn diagonal_values(dimension: usize, rank: usize, perturbation: f64) -> Vec<f64> {
        let mut values = vec![0.0; dimension * dimension];
        for index in 0..dimension {
            values[index * dimension + index] = if index < rank { 1.0 + perturbation } else { perturbation };
        }
        values
    }

    fn paired_fixture(
        maximum_degree: usize,
        perturbation: f64,
    ) -> (SymFpReceiptV1, Vec<PairedRepresentationBlockV1>) {
        let template = template_by_name("C2").unwrap();
        let mut reference = Vec::new();
        let mut paired = Vec::new();
        for degree in 0..=maximum_degree {
            let dimension = 2 * degree + 1;
            let reference_values =
                diagonal_values(dimension, template.fixed_dimensions[degree] as usize, 0.0);
            let observed_values =
                diagonal_values(dimension, template.fixed_dimensions[degree] as usize, perturbation);
            reference.push(RepresentationBlockV1 {
                degree,
                rows: dimension,
                columns: dimension,
                values: reference_values.clone(),
            });
            paired.push(PairedRepresentationBlockV1 {
                degree,
                rows: dimension,
                columns: dimension,
                reference_representation_id: "reference-c2-fixture-v1".to_owned(),
                observed_representation_id: "observed-c2-fixture-v1".to_owned(),
                reference_values,
                observed_values,
            });
        }
        (
            symfp_from_representation_blocks(&reference, SymFpProfileV1::paper_degree_9())
                .unwrap(),
            paired,
        )
    }

    #[test]
    fn zero_delta_pairs_derive_zero_bound_and_replay_exact_unique_label() {
        let (reference, paired) = paired_fixture(9, 0.0);
        let result = classify_paired_representations(
            &reference,
            super::super::CRYSTALLOGRAPHIC_TEMPLATES_V1.as_slice(),
            &paired,
        )
        .unwrap();
        assert_eq!(result.derived_operator_bounds.iter().map(|bound| bound.frobenius_upper_bound).collect::<Vec<_>>(), vec![0.0; 10]);
        assert_eq!(result.classification.guaranteed_nearest_template, Some("C2"));
        verify_paired_robust_symfp_receipt(&result).unwrap();
    }

    #[test]
    fn subtraction_roundoff_is_expanded_before_norm_accumulation() {
        let upper = outward_abs_difference(1.0, 2.0_f64.powi(-54)).unwrap();
        assert_eq!(upper.to_bits(), 1.0_f64.next_up().to_bits());
    }

    #[test]
    fn paired_delta_generates_a_reproducible_operator_norm_upper_bound() {
        let delta = [3.0, 0.0, 0.0, 4.0];
        let upper = frobenius_operator_norm_upper_bound(&delta).unwrap();
        assert!(upper >= 5.0);
        assert!(upper < 5.0 + 1.0e-12);
    }

    #[test]
    fn observed_reference_pair_is_bound_to_the_receipt_source_matrix() {
        let (reference, mut paired) = paired_fixture(3, 0.0);
        paired[1].reference_values[0] = 2.0;
        assert_eq!(
            classify_paired_representations(
                &reference,
                super::super::CRYSTALLOGRAPHIC_TEMPLATES_V1.as_slice(),
                &paired,
            ),
            Err(PairedSymFpErrorV1::ReferenceSpectrumMismatch)
        );
    }

    #[test]
    fn caller_cannot_forge_a_smaller_bound_without_changing_the_matrix_pair() {
        let (reference, paired) = paired_fixture(3, 0.01);
        let mut receipt = classify_paired_representations(
            &reference,
            super::super::CRYSTALLOGRAPHIC_TEMPLATES_V1.as_slice(),
            &paired,
        )
        .unwrap();
        receipt.derived_operator_bounds[1].frobenius_upper_bound = 0.0;
        assert_eq!(
            verify_paired_robust_symfp_receipt(&receipt),
            Err(PairedSymFpErrorV1::ReferenceSpectrumMismatch)
        );
    }

    #[test]
    fn overflow_and_invalid_matrix_shapes_fail_closed() {
        assert_eq!(
            frobenius_operator_norm_upper_bound(&[f64::MAX]),
            Err(PairedSymFpErrorV1::PerturbationOverflow)
        );
        let (reference, mut paired) = paired_fixture(2, 0.0);
        paired[1].observed_values.pop();
        assert_eq!(
            classify_paired_representations(
                &reference,
                &[*template_by_name("C2").unwrap()],
                &paired,
            ),
            Err(PairedSymFpErrorV1::MatrixLengthMismatch)
        );
    }
}
