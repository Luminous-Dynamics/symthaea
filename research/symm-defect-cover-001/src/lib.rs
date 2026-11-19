// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact interval-cover verification for uniform one-step symmetry-defect bounds.
//!
//! For each verified one-dimensional cell C_i centred at x_i with radius r_i,
//! and caller-supplied certified defect/Lipschitz bounds, derive
//!
//!   sup_{x in D} d(x) <= max_i (d_upper(x_i) + ceil(K * r_i)).
//!
//! This crate verifies that the closed interval cells cover the declared
//! one-dimensional domain. It does not prove that sample residual or Lipschitz
//! certificates are truthful, nor does it verify what the domain means
//! physically. Higher-dimensional geometric covers are deliberately out of
//! scope for v0.1.

use std::collections::BTreeSet;
use std::fmt;

use symm_error_envelope_reference::{RationalBoundV1, StepPremiseV1};

const MAX_SAMPLES: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefectSampleV1 {
    pub sample_id: String,
    /// Sample location in the declared coordinate fixed-point unit.
    pub center: i128,
    /// The certified cell is the closed interval [center-radius, center+radius].
    pub covering_radius: u128,
    /// Upper bound on the observed one-step defect at the sample.
    pub defect_upper: u128,
    pub coordinate_unit_id: String,
    pub defect_unit_id: String,
    /// External evidence identifier for the sample residual upper bound.
    pub sample_certificate_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefectCoverProblemV1 {
    pub case_id: String,
    pub domain_id: String,
    pub coordinate_unit_id: String,
    pub defect_unit_id: String,
    pub domain_lower: i128,
    pub domain_upper: i128,
    /// Certified Lipschitz upper bound K for the scalar defect function
    /// d(x)=||F(Tx)-T(F(x))|| on the declared domain.
    pub defect_lipschitz: RationalBoundV1,
    pub lipschitz_certificate_id: String,
    /// External evidence identifier defining the declared domain interval.
    pub domain_certificate_id: String,
    pub samples: Vec<DefectSampleV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefectCoverErrorV1 {
    EmptyCaseId,
    EmptyDomainId,
    EmptyCoordinateUnitId,
    EmptyDefectUnitId,
    InvalidDomainInterval,
    ZeroLipschitzDenominator,
    MissingLipschitzCertificate,
    MissingDomainCertificate,
    EmptySampleSet,
    TooManySamples,
    EmptySampleId,
    DuplicateSampleId,
    SampleOutsideDomain { sample_id: String },
    SampleUnitMismatch { sample_id: String },
    MissingSampleCertificate { sample_id: String },
    CoordinateOverflow { sample_id: String },
    CoverageGap { from: i128, to: i128 },
    ArithmeticOverflow,
    InvalidMapLipschitzDenominator,
    MissingMapLipschitzCertificate,
    ReceiptMismatch,
}

impl fmt::Display for DefectCoverErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DefectCoverErrorV1 {}

fn append_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u64).to_le_bytes());
    target.extend_from_slice(value);
}

fn append_string(target: &mut Vec<u8>, value: &str) {
    append_bytes(target, value.as_bytes());
}

fn append_u128(target: &mut Vec<u8>, value: u128) {
    target.extend_from_slice(&value.to_le_bytes());
}

fn append_i128(target: &mut Vec<u8>, value: i128) {
    target.extend_from_slice(&value.to_le_bytes());
}

impl DefectCoverProblemV1 {
    pub fn validate(&self) -> Result<(), DefectCoverErrorV1> {
        if self.case_id.trim().is_empty() {
            return Err(DefectCoverErrorV1::EmptyCaseId);
        }
        if self.domain_id.trim().is_empty() {
            return Err(DefectCoverErrorV1::EmptyDomainId);
        }
        if self.coordinate_unit_id.trim().is_empty() {
            return Err(DefectCoverErrorV1::EmptyCoordinateUnitId);
        }
        if self.defect_unit_id.trim().is_empty() {
            return Err(DefectCoverErrorV1::EmptyDefectUnitId);
        }
        if self.domain_lower > self.domain_upper {
            return Err(DefectCoverErrorV1::InvalidDomainInterval);
        }
        if self.defect_lipschitz.denominator == 0 {
            return Err(DefectCoverErrorV1::ZeroLipschitzDenominator);
        }
        if self.lipschitz_certificate_id.trim().is_empty() {
            return Err(DefectCoverErrorV1::MissingLipschitzCertificate);
        }
        if self.domain_certificate_id.trim().is_empty() {
            return Err(DefectCoverErrorV1::MissingDomainCertificate);
        }
        if self.samples.is_empty() {
            return Err(DefectCoverErrorV1::EmptySampleSet);
        }
        if self.samples.len() > MAX_SAMPLES {
            return Err(DefectCoverErrorV1::TooManySamples);
        }

        let mut ids = BTreeSet::new();
        for sample in &self.samples {
            if sample.sample_id.trim().is_empty() {
                return Err(DefectCoverErrorV1::EmptySampleId);
            }
            if !ids.insert(&sample.sample_id) {
                return Err(DefectCoverErrorV1::DuplicateSampleId);
            }
            if sample.center < self.domain_lower || sample.center > self.domain_upper {
                return Err(DefectCoverErrorV1::SampleOutsideDomain {
                    sample_id: sample.sample_id.clone(),
                });
            }
            if sample.coordinate_unit_id != self.coordinate_unit_id
                || sample.defect_unit_id != self.defect_unit_id
            {
                return Err(DefectCoverErrorV1::SampleUnitMismatch {
                    sample_id: sample.sample_id.clone(),
                });
            }
            if sample.sample_certificate_id.trim().is_empty() {
                return Err(DefectCoverErrorV1::MissingSampleCertificate {
                    sample_id: sample.sample_id.clone(),
                });
            }
            let radius = i128::try_from(sample.covering_radius).map_err(|_| {
                DefectCoverErrorV1::CoordinateOverflow {
                    sample_id: sample.sample_id.clone(),
                }
            })?;
            sample.center.checked_sub(radius).ok_or_else(|| {
                DefectCoverErrorV1::CoordinateOverflow {
                    sample_id: sample.sample_id.clone(),
                }
            })?;
            sample.center.checked_add(radius).ok_or_else(|| {
                DefectCoverErrorV1::CoordinateOverflow {
                    sample_id: sample.sample_id.clone(),
                }
            })?;
        }
        Ok(())
    }

    /// Canonical versioned bytes. These bytes are not a cryptographic digest.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"luminous.symm_defect_cover.interval.v1\0".to_vec();
        append_string(&mut bytes, &self.case_id);
        append_string(&mut bytes, &self.domain_id);
        append_string(&mut bytes, &self.coordinate_unit_id);
        append_string(&mut bytes, &self.defect_unit_id);
        append_i128(&mut bytes, self.domain_lower);
        append_i128(&mut bytes, self.domain_upper);
        append_u128(&mut bytes, self.defect_lipschitz.numerator);
        append_u128(&mut bytes, self.defect_lipschitz.denominator);
        append_string(&mut bytes, &self.lipschitz_certificate_id);
        append_string(&mut bytes, &self.domain_certificate_id);

        let mut samples: Vec<&DefectSampleV1> = self.samples.iter().collect();
        samples.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
        bytes.extend_from_slice(&(samples.len() as u64).to_le_bytes());
        for sample in samples {
            append_string(&mut bytes, &sample.sample_id);
            append_i128(&mut bytes, sample.center);
            append_u128(&mut bytes, sample.covering_radius);
            append_u128(&mut bytes, sample.defect_upper);
            append_string(&mut bytes, &sample.coordinate_unit_id);
            append_string(&mut bytes, &sample.defect_unit_id);
            append_string(&mut bytes, &sample.sample_certificate_id);
        }
        bytes
    }
}

fn checked_ceil_mul_div(
    value: u128,
    numerator: u128,
    denominator: u128,
) -> Result<u128, DefectCoverErrorV1> {
    if denominator == 0 {
        return Err(DefectCoverErrorV1::ZeroLipschitzDenominator);
    }
    let product = value
        .checked_mul(numerator)
        .ok_or(DefectCoverErrorV1::ArithmeticOverflow)?;
    let quotient = product / denominator;
    if product % denominator == 0 {
        Ok(quotient)
    } else {
        quotient
            .checked_add(1)
            .ok_or(DefectCoverErrorV1::ArithmeticOverflow)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefectCellReceiptV1 {
    pub sample_id: String,
    pub cell_lower: i128,
    pub cell_upper: i128,
    pub observed_defect_upper: u128,
    pub coverage_inflation_upper: u128,
    pub uniform_cell_defect_upper: u128,
    pub sample_certificate_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefectCoverReceiptV1 {
    pub schema_version: u32,
    pub case_id: String,
    pub problem_contract_bytes: Vec<u8>,
    pub domain_lower: i128,
    pub domain_upper: i128,
    pub verified_interval_cover: bool,
    pub cells: Vec<DefectCellReceiptV1>,
    pub uniform_defect_upper: u128,
    pub witness_sample_id: String,
    pub maximum_sampled_defect_upper: u128,
    pub maximum_coverage_inflation_upper: u128,
    pub maximum_covering_radius: u128,
    pub sample_count: usize,
}

fn build_cell_receipts(
    problem: &DefectCoverProblemV1,
) -> Result<Vec<DefectCellReceiptV1>, DefectCoverErrorV1> {
    let mut cells = Vec::with_capacity(problem.samples.len());
    for sample in &problem.samples {
        let radius =
            i128::try_from(sample.covering_radius).map_err(|_| {
                DefectCoverErrorV1::CoordinateOverflow {
                    sample_id: sample.sample_id.clone(),
                }
            })?;
        let raw_lower = sample.center.checked_sub(radius).ok_or_else(|| {
            DefectCoverErrorV1::CoordinateOverflow {
                sample_id: sample.sample_id.clone(),
            }
        })?;
        let raw_upper = sample.center.checked_add(radius).ok_or_else(|| {
            DefectCoverErrorV1::CoordinateOverflow {
                sample_id: sample.sample_id.clone(),
            }
        })?;
        let cell_lower = raw_lower.max(problem.domain_lower);
        let cell_upper = raw_upper.min(problem.domain_upper);
        let coverage_inflation_upper = checked_ceil_mul_div(
            sample.covering_radius,
            problem.defect_lipschitz.numerator,
            problem.defect_lipschitz.denominator,
        )?;
        let uniform_cell_defect_upper = sample
            .defect_upper
            .checked_add(coverage_inflation_upper)
            .ok_or(DefectCoverErrorV1::ArithmeticOverflow)?;
        cells.push(DefectCellReceiptV1 {
            sample_id: sample.sample_id.clone(),
            cell_lower,
            cell_upper,
            observed_defect_upper: sample.defect_upper,
            coverage_inflation_upper,
            uniform_cell_defect_upper,
            sample_certificate_id: sample.sample_certificate_id.clone(),
        });
    }
    cells.sort_by(|left, right| {
        (left.cell_lower, left.cell_upper, left.sample_id.as_str()).cmp(&(
            right.cell_lower,
            right.cell_upper,
            right.sample_id.as_str(),
        ))
    });
    Ok(cells)
}

fn verify_interval_cover(
    domain_lower: i128,
    domain_upper: i128,
    cells: &[DefectCellReceiptV1],
) -> Result<(), DefectCoverErrorV1> {
    let mut covered_until = domain_lower;
    let mut started = false;
    for cell in cells {
        if cell.cell_upper < domain_lower || cell.cell_lower > domain_upper {
            continue;
        }
        if !started {
            if cell.cell_lower > domain_lower {
                return Err(DefectCoverErrorV1::CoverageGap {
                    from: domain_lower,
                    to: cell.cell_lower,
                });
            }
            started = true;
            covered_until = cell.cell_upper;
        } else if cell.cell_lower > covered_until {
            return Err(DefectCoverErrorV1::CoverageGap {
                from: covered_until,
                to: cell.cell_lower,
            });
        } else {
            covered_until = covered_until.max(cell.cell_upper);
        }
        if covered_until >= domain_upper {
            return Ok(());
        }
    }
    if !started || covered_until < domain_upper {
        let from = if started { covered_until } else { domain_lower };
        return Err(DefectCoverErrorV1::CoverageGap {
            from,
            to: domain_upper,
        });
    }
    Ok(())
}

/// Verify exact interval coverage, then derive a conservative uniform defect
/// bound from the sample residual bounds and the certified Lipschitz constant.
pub fn certify_uniform_defect_cover(
    problem: &DefectCoverProblemV1,
) -> Result<DefectCoverReceiptV1, DefectCoverErrorV1> {
    problem.validate()?;
    let cells = build_cell_receipts(problem)?;
    verify_interval_cover(problem.domain_lower, problem.domain_upper, &cells)?;

    let witness = cells
        .iter()
        .max_by_key(|cell| cell.uniform_cell_defect_upper)
        .ok_or(DefectCoverErrorV1::EmptySampleSet)?;
    let uniform_defect_upper = witness.uniform_cell_defect_upper;
    let witness_sample_id = witness.sample_id.clone();
    let maximum_sampled_defect_upper = cells
        .iter()
        .map(|cell| cell.observed_defect_upper)
        .max()
        .unwrap_or(0);
    let maximum_coverage_inflation_upper = cells
        .iter()
        .map(|cell| cell.coverage_inflation_upper)
        .max()
        .unwrap_or(0);
    let maximum_covering_radius = problem
        .samples
        .iter()
        .map(|sample| sample.covering_radius)
        .max()
        .unwrap_or(0);

    Ok(DefectCoverReceiptV1 {
        schema_version: 1,
        case_id: problem.case_id.clone(),
        problem_contract_bytes: problem.canonical_bytes(),
        domain_lower: problem.domain_lower,
        domain_upper: problem.domain_upper,
        verified_interval_cover: true,
        cells,
        uniform_defect_upper,
        witness_sample_id,
        maximum_sampled_defect_upper,
        maximum_coverage_inflation_upper,
        maximum_covering_radius,
        sample_count: problem.samples.len(),
    })
}

/// Recompute the cover, all cell bounds, and the final uniform defect bound.
pub fn verify_defect_cover_receipt(
    problem: &DefectCoverProblemV1,
    receipt: &DefectCoverReceiptV1,
) -> Result<(), DefectCoverErrorV1> {
    if receipt.schema_version != 1
        || receipt.case_id != problem.case_id
        || receipt.problem_contract_bytes != problem.canonical_bytes()
    {
        return Err(DefectCoverErrorV1::ReceiptMismatch);
    }
    let replayed = certify_uniform_defect_cover(problem)?;
    if *receipt != replayed {
        return Err(DefectCoverErrorV1::ReceiptMismatch);
    }
    Ok(())
}

/// Convert a verified whole-domain cover bound into the error-envelope
/// recurrence. Because the bound already covers the complete invariant
/// domain, the additional state-dependent defect term is exactly zero. The
/// caller must separately establish that both compared trajectories stay in
/// that same domain at every step.
pub fn to_uniform_domain_step_premise(
    problem: &DefectCoverProblemV1,
    receipt: &DefectCoverReceiptV1,
    map_lipschitz: RationalBoundV1,
    map_lipschitz_certificate_id: &str,
) -> Result<StepPremiseV1, DefectCoverErrorV1> {
    if map_lipschitz.denominator == 0 {
        return Err(DefectCoverErrorV1::InvalidMapLipschitzDenominator);
    }
    if map_lipschitz_certificate_id.trim().is_empty() {
        return Err(DefectCoverErrorV1::MissingMapLipschitzCertificate);
    }
    verify_defect_cover_receipt(problem, receipt)?;
    Ok(StepPremiseV1 {
        map_lipschitz,
        map_lipschitz_certificate_id: map_lipschitz_certificate_id.to_owned(),
        defect_variation_lipschitz: RationalBoundV1 {
            numerator: 0,
            denominator: 1,
        },
        defect_variation_certificate_id: format!(
            "uniform-domain-cover-absorbs-defect-growth:{}",
            receipt.case_id
        ),
        equivariance_defect_upper: receipt.uniform_defect_upper,
        defect_certificate_id: format!("verified-interval-cover:{}", receipt.case_id),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(
        sample_id: &str,
        center: i128,
        radius: u128,
        defect_upper: u128,
    ) -> DefectSampleV1 {
        DefectSampleV1 {
            sample_id: sample_id.to_owned(),
            center,
            covering_radius: radius,
            defect_upper,
            coordinate_unit_id: "micrometre-int-v1".to_owned(),
            defect_unit_id: "micrometre-int-v1".to_owned(),
            sample_certificate_id: format!("sample-residual-{sample_id}"),
        }
    }

    fn fixture() -> DefectCoverProblemV1 {
        DefectCoverProblemV1 {
            case_id: "one-dimensional-defect-cover-v1".to_owned(),
            domain_id: "closed-domain-0-100-v1".to_owned(),
            coordinate_unit_id: "micrometre-int-v1".to_owned(),
            defect_unit_id: "micrometre-int-v1".to_owned(),
            domain_lower: 0,
            domain_upper: 100,
            defect_lipschitz: RationalBoundV1 {
                numerator: 1,
                denominator: 10,
            },
            lipschitz_certificate_id: "toy-defect-lipschitz-1-over-10".to_owned(),
            domain_certificate_id: "toy-domain-definition-0-100".to_owned(),
            samples: vec![
                sample("left", 0, 25, 2),
                sample("middle", 50, 25, 3),
                sample("right", 100, 25, 1),
            ],
        }
    }

    #[test]
    fn touching_closed_cells_verify_the_complete_domain_and_produce_uniform_bound() {
        let problem = fixture();
        let receipt = certify_uniform_defect_cover(&problem).unwrap();
        assert!(receipt.verified_interval_cover);
        assert_eq!(receipt.uniform_defect_upper, 6);
        assert_eq!(receipt.witness_sample_id, "middle");
        assert_eq!(receipt.maximum_sampled_defect_upper, 3);
        assert_eq!(receipt.maximum_coverage_inflation_upper, 3);
        assert_eq!(receipt.maximum_covering_radius, 25);
        assert_eq!(receipt.sample_count, 3);
        assert_eq!(
            receipt.cells.iter().map(|cell| (cell.cell_lower, cell.cell_upper)).collect::<Vec<_>>(),
            vec![(0_i128, 25), (25, 75), (75, 100)]
        );
    }

    #[test]
    fn nonuniform_cover_uses_each_cells_own_radius() {
        let mut problem = fixture();
        problem.samples = vec![
            sample("left", 0, 40, 2),
            sample("middle", 50, 10, 3),
            sample("right", 100, 40, 1),
        ];
        let receipt = certify_uniform_defect_cover(&problem).unwrap();
        assert_eq!(receipt.uniform_defect_upper, 6);
        let inflation = receipt
            .cells
            .iter()
            .map(|cell| cell.coverage_inflation_upper)
            .collect::<Vec<_>>();
        assert_eq!(inflation, vec![4, 1, 4]);
    }

    #[test]
    fn a_real_gap_abstains_instead_of_treating_samples_as_a_cover() {
        let mut problem = fixture();
        problem.samples = vec![
            sample("left", 0, 20, 0),
            sample("middle", 50, 20, 0),
            sample("right", 100, 20, 0),
        ];
        assert_eq!(
            certify_uniform_defect_cover(&problem),
            Err(DefectCoverErrorV1::CoverageGap { from: 20, to: 30 })
        );
    }

    #[test]
    fn exact_single_point_domain_is_covered_by_a_zero_radius_sample() {
        let mut problem = fixture();
        problem.domain_lower = 7;
        problem.domain_upper = 7;
        problem.samples = vec![sample("point", 7, 0, 4)];
        let receipt = certify_uniform_defect_cover(&problem).unwrap();
        assert_eq!(receipt.uniform_defect_upper, 4);
        assert_eq!(receipt.cells[0].cell_lower, 7);
        assert_eq!(receipt.cells[0].cell_upper, 7);
    }

    #[test]
    fn unit_mismatch_and_missing_sample_evidence_fail_closed() {
        let mut problem = fixture();
        problem.samples[1].defect_unit_id = "millimetre-int-v1".to_owned();
        assert_eq!(
            problem.validate(),
            Err(DefectCoverErrorV1::SampleUnitMismatch {
                sample_id: "middle".to_owned()
            })
        );

        let mut problem = fixture();
        problem.samples[0].sample_certificate_id.clear();
        assert_eq!(
            problem.validate(),
            Err(DefectCoverErrorV1::MissingSampleCertificate {
                sample_id: "left".to_owned()
            })
        );
    }

    #[test]
    fn arithmetic_overflow_in_coverage_inflation_or_cell_endpoint_fails_closed() {
        let mut problem = fixture();
        problem.defect_lipschitz.numerator = u128::MAX;
        problem.defect_lipschitz.denominator = 1;
        assert_eq!(
            certify_uniform_defect_cover(&problem),
            Err(DefectCoverErrorV1::ArithmeticOverflow)
        );

        let mut problem = fixture();
        problem.samples[0].covering_radius = u128::MAX;
        assert_eq!(
            problem.validate(),
            Err(DefectCoverErrorV1::CoordinateOverflow {
                sample_id: "left".to_owned()
            })
        );
    }

    #[test]
    fn duplicate_ids_and_nonpositive_or_reversed_domain_contracts_are_rejected() {
        let mut problem = fixture();
        problem.samples[1].sample_id = problem.samples[0].sample_id.clone();
        assert_eq!(
            problem.validate(),
            Err(DefectCoverErrorV1::DuplicateSampleId)
        );

        let mut problem = fixture();
        problem.domain_lower = 101;
        assert_eq!(
            problem.validate(),
            Err(DefectCoverErrorV1::InvalidDomainInterval)
        );
    }

    #[test]
    fn receipt_replay_rejects_tampering_and_mutated_premises() {
        let problem = fixture();
        let receipt = certify_uniform_defect_cover(&problem).unwrap();
        verify_defect_cover_receipt(&problem, &receipt).unwrap();

        let mut forged = receipt.clone();
        forged.uniform_defect_upper = 0;
        assert_eq!(
            verify_defect_cover_receipt(&problem, &forged),
            Err(DefectCoverErrorV1::ReceiptMismatch)
        );

        let mut changed = problem.clone();
        changed.samples[1].defect_upper = 4;
        assert_eq!(
            verify_defect_cover_receipt(&changed, &receipt),
            Err(DefectCoverErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn verified_cover_converts_to_a_global_defect_step_premise() {
        let problem = fixture();
        let receipt = certify_uniform_defect_cover(&problem).unwrap();
        let step = to_uniform_domain_step_premise(
            &problem,
            &receipt,
            RationalBoundV1 {
                numerator: 3,
                denominator: 2,
            },
            "step-map-lipschitz-cert",
        )
        .unwrap();
        assert_eq!(step.equivariance_defect_upper, 6);
        assert_eq!(step.defect_variation_lipschitz.numerator, 0);
        assert_eq!(step.defect_variation_lipschitz.denominator, 1);
        assert_eq!(step.map_lipschitz.numerator, 3);
        assert_eq!(step.map_lipschitz.denominator, 2);
        assert!(step.defect_certificate_id.contains("verified-interval-cover"));

        let mut forged = receipt;
        forged.verified_interval_cover = false;
        assert_eq!(
            to_uniform_domain_step_premise(
                &problem,
                &forged,
                RationalBoundV1 {
                    numerator: 3,
                    denominator: 2,
                },
                "step-map-lipschitz-cert",
            ),
            Err(DefectCoverErrorV1::ReceiptMismatch)
        );
    }

    #[test]
    fn malformed_map_lipschitz_and_missing_map_certificate_are_rejected() {
        let problem = fixture();
        let receipt = certify_uniform_defect_cover(&problem).unwrap();
        assert_eq!(
            to_uniform_domain_step_premise(
                &problem,
                &receipt,
                RationalBoundV1 {
                    numerator: 1,
                    denominator: 0,
                },
                "map-cert"
            ),
            Err(DefectCoverErrorV1::InvalidMapLipschitzDenominator)
        );
        assert_eq!(
            to_uniform_domain_step_premise(
                &problem,
                &receipt,
                RationalBoundV1 {
                    numerator: 1,
                    denominator: 1,
                },
                " "
            ),
            Err(DefectCoverErrorV1::MissingMapLipschitzCertificate)
        );
    }
}
