// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Joint nonlinear Sommer-type scales from static-potential fit replicates.
//!
//! Scale uncertainties are derived by transforming every shared jackknife
//! `(sigma,e)` replicate before forming covariance. This module does not infer
//! fit parameters, choose targets, or declare a physical lattice spacing.

use std::collections::BTreeSet;

pub const JOINT_SOMMER_SCALE_ID: &str = "joint_jackknife_sommer_scale_v1";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticPotentialParameterSample {
    pub sigma: f64,
    pub e: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SommerScaleTarget {
    pub scale_id: String,
    pub c: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointSommerScaleEstimate {
    pub analysis_id: &'static str,
    pub fit_lineage: String,
    pub target_ids: Vec<String>,
    pub c_values: Vec<f64>,
    pub central_values: Vec<f64>,
    pub jackknife_mean: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub standard_errors: Vec<f64>,
    pub replicate_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SommerScaleError {
    MissingFitLineage,
    TooFewReplicates(usize),
    EmptyTargets,
    EmptyScaleId { index: usize },
    DuplicateScaleId(String),
    NonFiniteParameter { sample_index: usize },
    NonPositiveSigma { sample_index: usize, sigma: f64 },
    InvalidTarget { target_index: usize, c: f64 },
    TargetNotAboveCoulomb {
        sample_index: usize,
        target_index: usize,
        c: f64,
        e: f64,
    },
    NonFiniteDerivedScale {
        sample_index: usize,
        target_index: usize,
    },
}

fn validate_sample(
    sample: StaticPotentialParameterSample,
    sample_index: usize,
) -> Result<(), SommerScaleError> {
    if !sample.sigma.is_finite() || !sample.e.is_finite() {
        return Err(SommerScaleError::NonFiniteParameter { sample_index });
    }
    if sample.sigma <= 0.0 {
        return Err(SommerScaleError::NonPositiveSigma {
            sample_index,
            sigma: sample.sigma,
        });
    }
    Ok(())
}

fn derived_scale(
    sample: StaticPotentialParameterSample,
    sample_index: usize,
    target: &SommerScaleTarget,
    target_index: usize,
) -> Result<f64, SommerScaleError> {
    if target.c <= sample.e {
        return Err(SommerScaleError::TargetNotAboveCoulomb {
            sample_index,
            target_index,
            c: target.c,
            e: sample.e,
        });
    }
    let value = ((target.c - sample.e) / sample.sigma).sqrt();
    if !value.is_finite() {
        return Err(SommerScaleError::NonFiniteDerivedScale {
            sample_index,
            target_index,
        });
    }
    Ok(value)
}

/// Transform one central fit and all shared jackknife parameter replicates into
/// Sommer-type scales `r_c = sqrt((c-e)/sigma)` and their joint covariance.
pub fn joint_jackknife_sommer_scales(
    central: StaticPotentialParameterSample,
    replicates: &[StaticPotentialParameterSample],
    targets: &[SommerScaleTarget],
    fit_lineage: &str,
) -> Result<JointSommerScaleEstimate, SommerScaleError> {
    if fit_lineage.trim().is_empty() {
        return Err(SommerScaleError::MissingFitLineage);
    }
    if replicates.len() < 2 {
        return Err(SommerScaleError::TooFewReplicates(replicates.len()));
    }
    if targets.is_empty() {
        return Err(SommerScaleError::EmptyTargets);
    }

    validate_sample(central, 0)?;
    for (index, replicate) in replicates.iter().copied().enumerate() {
        validate_sample(replicate, index + 1)?;
    }

    let mut seen_ids = BTreeSet::new();
    for (index, target) in targets.iter().enumerate() {
        if target.scale_id.trim().is_empty() {
            return Err(SommerScaleError::EmptyScaleId { index });
        }
        if !seen_ids.insert(target.scale_id.clone()) {
            return Err(SommerScaleError::DuplicateScaleId(target.scale_id.clone()));
        }
        if !target.c.is_finite() {
            return Err(SommerScaleError::InvalidTarget {
                target_index: index,
                c: target.c,
            });
        }
    }

    let central_values = targets
        .iter()
        .enumerate()
        .map(|(target_index, target)| derived_scale(central, 0, target, target_index))
        .collect::<Result<Vec<_>, _>>()?;

    let replicate_values = replicates
        .iter()
        .copied()
        .enumerate()
        .map(|(replicate_index, sample)| {
            targets
                .iter()
                .enumerate()
                .map(|(target_index, target)| {
                    derived_scale(sample, replicate_index + 1, target, target_index)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let replicate_count = replicate_values.len();
    let width = targets.len();
    let mut jackknife_mean = vec![0.0; width];
    for row in &replicate_values {
        for (mean, value) in jackknife_mean.iter_mut().zip(row) {
            *mean += *value;
        }
    }
    for mean in &mut jackknife_mean {
        *mean /= replicate_count as f64;
    }

    let factor = (replicate_count - 1) as f64 / replicate_count as f64;
    let mut covariance = vec![vec![0.0; width]; width];
    for row in &replicate_values {
        for i in 0..width {
            let di = row[i] - jackknife_mean[i];
            for j in 0..width {
                covariance[i][j] += factor * di * (row[j] - jackknife_mean[j]);
            }
        }
    }
    let standard_errors = (0..width)
        .map(|index| covariance[index][index].sqrt())
        .collect::<Vec<_>>();

    Ok(JointSommerScaleEstimate {
        analysis_id: JOINT_SOMMER_SCALE_ID,
        fit_lineage: fit_lineage.to_owned(),
        target_ids: targets.iter().map(|target| target.scale_id.clone()).collect(),
        c_values: targets.iter().map(|target| target.c).collect(),
        central_values,
        jackknife_mean,
        covariance,
        standard_errors,
        replicate_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_replicates() -> Vec<StaticPotentialParameterSample> {
        (0..12)
            .map(|k| {
                let theta = 2.0 * std::f64::consts::PI * (k + 1) as f64 / 12.0;
                StaticPotentialParameterSample {
                    sigma: 0.18 + 0.0025 * theta.sin() + 0.001 * (2.0 * theta).cos(),
                    e: 0.25 + 0.008 * theta.cos() - 0.003 * (2.0 * theta).sin(),
                }
            })
            .collect()
    }

    fn targets() -> Vec<SommerScaleTarget> {
        vec![
            SommerScaleTarget { scale_id: "r0".into(), c: 1.65 },
            SommerScaleTarget { scale_id: "r4".into(), c: 4.0 },
            SommerScaleTarget { scale_id: "r6".into(), c: 6.0 },
        ]
    }

    #[test]
    fn reproduces_independent_lqcd_020s_joint_scale_oracle() {
        let estimate = joint_jackknife_sommer_scales(
            StaticPotentialParameterSample { sigma: 0.18, e: 0.25 },
            &fixture_replicates(),
            &targets(),
            "synthetic-fit-lineage",
        )
        .unwrap();

        let central = [2.788_866_755_113_585, 4.564_354_645_876_384, 5.651_941_652_604_39];
        let errors = [0.053_088_282_404_862_31, 0.081_424_416_957_583_01, 0.100_173_454_142_290_01];
        let covariance = [
            [0.002_818_365_728_698_413, 0.004_203_509_237_759_222, 0.005_100_095_468_994_386],
            [0.004_203_509_237_759_222, 0.006_629_935_676_882_331, 0.008_145_573_852_299_368],
            [0.005_100_095_468_994_386, 0.008_145_573_852_299_368, 0.010_034_720_914_797_48],
        ];
        for (actual, expected) in estimate.central_values.iter().zip(central) {
            assert!((actual - expected).abs() < 2.0e-14);
        }
        for (actual, expected) in estimate.standard_errors.iter().zip(errors) {
            assert!((actual - expected).abs() < 2.0e-14);
        }
        for (actual_row, expected_row) in estimate.covariance.iter().zip(covariance) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 3.0e-14);
            }
        }
        assert_eq!(estimate.replicate_count, 12);
        assert_eq!(estimate.target_ids, ["r0", "r4", "r6"]);
    }

    #[test]
    fn invalid_scale_domain_fails_closed() {
        let error = joint_jackknife_sommer_scales(
            StaticPotentialParameterSample { sigma: 0.18, e: 1.65 },
            &fixture_replicates(),
            &[SommerScaleTarget { scale_id: "r0".into(), c: 1.65 }],
            "fixture",
        )
        .unwrap_err();
        assert!(matches!(error, SommerScaleError::TargetNotAboveCoulomb { .. }));
    }
}
