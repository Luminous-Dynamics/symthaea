// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Covariance-aware, prior-free experiment discrimination.
//!
//! The caller supplies Cov(Y_first - Y_second) for every hypothesis pair.
//! This matrix must already include the shared model, calibration,
//! measurement, and numerical-error contributions intended by the declared
//! uncertainty model. The selector does not assume hypotheses are independent.
//!
//! The Mahalanobis score is a covariance-geometry screening statistic, not
//! expected information gain, a Bayes factor, a posterior, or a truth oracle.
//! Singular support mismatches are quarantined for review. No jitter or silent
//! regularization is applied.

use crate::ParameterSetting;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Display, Formatter};

pub const COVARIANCE_SCHEMA_VERSION: &str = "physics-experiment-design-covariance-v0.1";
pub const COVARIANCE_ALGORITHM_VERSION: &str = "correlation-whitened-jacobi-pseudoinverse-v0.1";
pub const RELATIVE_EIGENVALUE_TOLERANCE: f64 = 1.0e-12;
pub const RELATIVE_SUPPORT_TOLERANCE: f64 = 1.0e-10;
const NEGATIVE_EIGENVALUE_TOLERANCE: f64 = 1.0e-10;
const JACOBI_CONVERGENCE_TOLERANCE: f64 = 1.0e-14;

/// Complete design request. Each experiment must provide one difference
/// covariance matrix for every unordered pair of competing hypotheses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CovarianceAwareDesignInput {
    pub schema_version: String,
    pub target: String,
    pub hypothesis_ids: Vec<String>,
    pub experiments: Vec<CovarianceAwareCandidateExperiment>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CovarianceAwareCandidateExperiment {
    pub experiment_id: String,
    pub description: String,
    pub settings: Vec<ParameterSetting>,
    pub cost_units: f64,
    pub duration_hours: f64,
    /// Matrix order for means and covariance rows/columns.
    pub observables: Vec<CovarianceObservable>,
    pub predictions: Vec<VectorHypothesisPrediction>,
    pub pairwise_difference_covariances: Vec<PairwiseDifferenceCovariance>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CovarianceObservable {
    pub observable_id: String,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorHypothesisPrediction {
    pub hypothesis_id: String,
    pub model_descriptor: String,
    pub model_digest_sha256: String,
    /// Means in exactly the experiment's observable order.
    pub means: Vec<f64>,
}

/// IDs must be in canonical lexical order. Matrix rows and columns follow the
/// experiment's observable order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairwiseDifferenceCovariance {
    pub first_hypothesis_id: String,
    pub second_hypothesis_id: String,
    /// Covariance of Y_first - Y_second, not two independent marginal covariances.
    pub covariance: Vec<Vec<f64>>,
    pub covariance_descriptor: String,
    pub covariance_descriptor_digest_sha256: String,
    /// Hash of canonical row-major IEEE-754 f64 bit patterns and dimensions.
    pub covariance_matrix_digest_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CovarianceDesignReport {
    pub schema_version: String,
    pub algorithm_version: String,
    pub target: String,
    pub criterion: String,
    pub relative_eigenvalue_tolerance: f64,
    pub relative_support_tolerance: f64,
    pub ranked_experiments: Vec<RankedCovarianceExperiment>,
    /// A candidate with a mean difference outside a singular covariance's
    /// support is not ranked until omitted-noise assumptions are reviewed.
    pub support_review_required: Vec<SupportReviewRequired>,
    pub limitations: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankedCovarianceExperiment {
    pub rank: usize,
    pub experiment_id: String,
    pub description: String,
    pub settings: Vec<ParameterSetting>,
    pub cost_units: f64,
    pub duration_hours: f64,
    pub worst_case_separation: f64,
    pub worst_case_separation_squared: f64,
    pub separation_per_cost_unit: f64,
    pub closest_hypothesis_pair: [String; 2],
    pub observable_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SupportReviewRequired {
    pub experiment_id: String,
    pub first_hypothesis_id: String,
    pub second_hypothesis_id: String,
    pub reason: String,
    pub null_space_residual: f64,
    pub standardized_mean_norm: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CovarianceDesignError {
    InvalidInput(String),
    InvalidCovariance(String),
    DigestMismatch(String),
    NonPsdCovariance(String),
    NumericalFailure(String),
}

impl Display for CovarianceDesignError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid covariance design: {message}"),
            Self::InvalidCovariance(message) => write!(f, "invalid covariance: {message}"),
            Self::DigestMismatch(message) => write!(f, "covariance digest mismatch: {message}"),
            Self::NonPsdCovariance(message) => write!(f, "covariance is not positive semidefinite: {message}"),
            Self::NumericalFailure(message) => write!(f, "covariance numerical failure: {message}"),
        }
    }
}
impl Error for CovarianceDesignError {}

/// Rank by the minimum pairwise Mahalanobis separation, then divide by
/// caller-supplied relative cost. Covariance is supplied for the difference
/// vector, preserving cross-hypothesis covariance and cross-observable terms.
///
/// The matrix is converted to a correlation matrix before eigendecomposition.
/// This makes eigenvalue cutoff and support decisions invariant to positive
/// diagonal unit rescaling (for example, metres to millimetres).
///
/// If a mean difference has a material component in a zero-variance direction,
/// the candidate is emitted in support_review_required and is excluded from
/// automatic ranking. The pseudoinverse alone would miss this condition.
pub fn rank_covariance_aware_experiments(
    input: &CovarianceAwareDesignInput,
) -> Result<CovarianceDesignReport, CovarianceDesignError> {
    validate_input(input)?;

    let mut hypotheses = input.hypothesis_ids.clone();
    hypotheses.sort();
    let mut ranked = Vec::new();
    let mut review = Vec::new();

    for experiment in &input.experiments {
        let predictions: BTreeMap<&str, &VectorHypothesisPrediction> = experiment
            .predictions.iter().map(|p| (p.hypothesis_id.as_str(), p)).collect();
        let covariances: BTreeMap<(&str, &str), &PairwiseDifferenceCovariance> = experiment
            .pairwise_difference_covariances.iter()
            .map(|c| ((c.first_hypothesis_id.as_str(), c.second_hypothesis_id.as_str()), c))
            .collect();

        let mut worst_squared = f64::INFINITY;
        let mut closest_pair: Option<[String; 2]> = None;
        let mut requires_review = false;

        'pairs: for i in 0..hypotheses.len() {
            for j in (i + 1)..hypotheses.len() {
                let first_id = &hypotheses[i];
                let second_id = &hypotheses[j];
                let first = predictions[first_id.as_str()].means.as_slice();
                let second = predictions[second_id.as_str()].means.as_slice();
                let delta: Vec<f64> = first.iter().zip(second).map(|(a, b)| a - b).collect();
                if delta.iter().any(|value| !value.is_finite()) {
                    return Err(CovarianceDesignError::NumericalFailure(format!(
                        "non-finite mean difference for {} vs {} in {}",
                        first_id, second_id, experiment.experiment_id
                    )));
                }
                let covariance = covariances[&(first_id.as_str(), second_id.as_str())];
                match mahalanobis_squared(&delta, &covariance.covariance)? {
                    PairScore::Finite(score) => {
                        if score < worst_squared {
                            worst_squared = score;
                            closest_pair = Some([first_id.clone(), second_id.clone()]);
                        }
                    }
                    PairScore::OutsideSupport { residual, norm } => {
                        review.push(SupportReviewRequired {
                            experiment_id: experiment.experiment_id.clone(),
                            first_hypothesis_id: first_id.clone(),
                            second_hypothesis_id: second_id.clone(),
                            reason: "mean difference has a material component in a zero-variance covariance direction; inspect omitted noise and support assumptions before automatic selection".to_owned(),
                            null_space_residual: residual,
                            standardized_mean_norm: norm,
                        });
                        requires_review = true;
                        break 'pairs;
                    }
                }
            }
        }

        if requires_review {
            continue;
        }
        let pair = closest_pair.expect("validation requires at least two hypotheses");
        let separation = worst_squared.sqrt();
        let score = separation / experiment.cost_units;
        if !separation.is_finite() || !score.is_finite() {
            return Err(CovarianceDesignError::NumericalFailure(format!(
                "non-finite score for experiment {}", experiment.experiment_id
            )));
        }
        ranked.push(RankedCovarianceExperiment {
            rank: 0,
            experiment_id: experiment.experiment_id.clone(),
            description: experiment.description.clone(),
            settings: canonical_settings(&experiment.settings),
            cost_units: experiment.cost_units,
            duration_hours: experiment.duration_hours,
            worst_case_separation: separation,
            worst_case_separation_squared: worst_squared,
            separation_per_cost_unit: score,
            closest_hypothesis_pair: pair,
            observable_count: experiment.observables.len(),
        });
    }

    ranked.sort_by(|a, b| {
        b.separation_per_cost_unit.total_cmp(&a.separation_per_cost_unit)
            .then_with(|| a.experiment_id.cmp(&b.experiment_id))
    });
    for (index, candidate) in ranked.iter_mut().enumerate() {
        candidate.rank = index + 1;
    }
    review.sort_by(|a, b| a.experiment_id.cmp(&b.experiment_id)
        .then_with(|| a.first_hypothesis_id.cmp(&b.first_hypothesis_id))
        .then_with(|| a.second_hypothesis_id.cmp(&b.second_hypothesis_id)));

    Ok(CovarianceDesignReport {
        schema_version: COVARIANCE_SCHEMA_VERSION.to_owned(),
        algorithm_version: COVARIANCE_ALGORITHM_VERSION.to_owned(),
        target: input.target.clone(),
        criterion: "prior-free maximin Mahalanobis separation of rival mean vectors under the supplied covariance of their difference, divided by declared relative cost".to_owned(),
        relative_eigenvalue_tolerance: RELATIVE_EIGENVALUE_TOLERANCE,
        relative_support_tolerance: RELATIVE_SUPPORT_TOLERANCE,
        ranked_experiments: ranked,
        support_review_required: review,
        limitations: vec![
            "This is a covariance-geometry screening statistic, not expected information gain, a Bayes factor, a posterior, or a truth oracle.".to_owned(),
            "The supplied covariance must include the cross-hypothesis and cross-observable covariance contributions intended by the uncertainty model.".to_owned(),
            "The descriptor digest binds provenance text; the matrix digest binds numeric matrix values. Neither proves that an executable generated the supplied values.".to_owned(),
            "A mean outside a singular covariance support is quarantined for review, not assigned a finite pseudoinverse score or an automatic infinite score.".to_owned(),
            "Covariance uncertainty sets, nuisance profiling, non-Gaussian predictive distributions, and calibrated Monte Carlo EIG are not implemented in this v0.1 selector.".to_owned(),
            "Cost is caller-defined relative units. Solver calls, CPU/GPU, memory, wall time, tokens, and human review must also be reported separately.".to_owned(),
            "Predictions are not observations. Symtropy execution and independent qualification remain separate authorities.".to_owned(),
        ],
    })
}

enum PairScore {
    Finite(f64),
    OutsideSupport { residual: f64, norm: f64 },
}

fn mahalanobis_squared(
    delta: &[f64],
    covariance: &[Vec<f64>],
) -> Result<PairScore, CovarianceDesignError> {
    let n = delta.len();
    let mut scales = vec![0.0; n];
    let mut active = Vec::new();
    for i in 0..n {
        if covariance[i][i] == 0.0 {
            if delta[i] != 0.0 {
                let norm = delta.iter().fold(0.0_f64, |acc, value| acc.hypot(*value));
                return Ok(PairScore::OutsideSupport { residual: delta[i].abs(), norm });
            }
        } else {
            scales[i] = covariance[i][i].sqrt();
            active.push(i);
        }
    }
    if active.is_empty() {
        return Ok(PairScore::Finite(0.0));
    }

    let m = active.len();
    let mut correlation = vec![vec![0.0; m]; m];
    let mut standardized_delta = vec![0.0; m];
    for (row, &i) in active.iter().enumerate() {
        standardized_delta[row] = delta[i] / scales[i];
        for (column, &j) in active.iter().enumerate() {
            correlation[row][column] = covariance[i][j] / scales[i] / scales[j];
        }
    }
    if standardized_delta.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceDesignError::NumericalFailure(
            "standardized mean difference overflowed".to_owned(),
        ));
    }
    let norm = standardized_delta.iter().fold(0.0_f64, |acc, value| acc.hypot(*value));
    let (eigenvalues, eigenvectors) = jacobi_symmetric_eigen(&correlation)
        .ok_or_else(|| CovarianceDesignError::NumericalFailure("Jacobi eigensolver did not converge".to_owned()))?;
    let max_eigenvalue = eigenvalues.iter().copied().fold(0.0_f64, f64::max).max(1.0);
    let cutoff = RELATIVE_EIGENVALUE_TOLERANCE * max_eigenvalue;
    let negative_cutoff = NEGATIVE_EIGENVALUE_TOLERANCE * max_eigenvalue;
    if let Some(minimum) = eigenvalues.iter().copied().reduce(f64::min) {
        if minimum < -negative_cutoff {
            return Err(CovarianceDesignError::NonPsdCovariance(format!(
                "minimum standardized eigenvalue {minimum}"
            )));
        }
    }

    let mut score = 0.0;
    let mut null_projection_squared = 0.0;
    for (mode, eigenvalue) in eigenvalues.iter().enumerate() {
        let projection: f64 = (0..m)
            .map(|row| eigenvectors[row][mode] * standardized_delta[row])
            .sum();
        if *eigenvalue > cutoff {
            score += projection * projection / eigenvalue;
        } else {
            null_projection_squared += projection * projection;
        }
    }
    let residual = null_projection_squared.sqrt();
    if residual > RELATIVE_SUPPORT_TOLERANCE * norm.max(1.0) {
        return Ok(PairScore::OutsideSupport { residual, norm });
    }
    if !score.is_finite() || score < 0.0 {
        return Err(CovarianceDesignError::NumericalFailure("non-finite pseudoinverse quadratic form".to_owned()));
    }
    Ok(PairScore::Finite(score))
}

/// Jacobi eigensolver for a finite real symmetric matrix. Eigenvectors are
/// returned as columns. Matrices are correlation-scaled before this call.
fn jacobi_symmetric_eigen(matrix: &[Vec<f64>]) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    if n == 1 {
        return Some((vec![matrix[0][0]], vec![vec![1.0]]));
    }
    let mut a = matrix.to_vec();
    let mut vectors = vec![vec![0.0; n]; n];
    for (i, row) in vectors.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let max_iterations = 100usize.saturating_mul(n).saturating_mul(n);
    let mut converged = false;
    for _ in 0..max_iterations {
        let mut p = 0;
        let mut q = 1;
        let mut largest = 0.0_f64;
        let mut diagonal_scale = 1.0_f64;
        for i in 0..n {
            diagonal_scale = diagonal_scale.max(a[i][i].abs());
            for j in (i + 1)..n {
                if a[i][j].abs() > largest {
                    largest = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest <= JACOBI_CONVERGENCE_TOLERANCE * diagonal_scale {
            converged = true;
            break;
        }
        let apq = a[p][q];
        if apq == 0.0 {
            continue;
        }
        let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let app = a[p][p];
        let aqq = a[q][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for k in 0..n {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
            let vkp = vectors[k][p];
            let vkq = vectors[k][q];
            vectors[k][p] = c * vkp - s * vkq;
            vectors[k][q] = s * vkp + c * vkq;
        }
    }
    if !converged {
        return None;
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    if eigenvalues.iter().any(|v| !v.is_finite())
        || vectors.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    Some((eigenvalues, vectors))
}

fn validate_input(input: &CovarianceAwareDesignInput) -> Result<(), CovarianceDesignError> {
    if input.schema_version != COVARIANCE_SCHEMA_VERSION {
        return Err(CovarianceDesignError::InvalidInput("unsupported schema version".to_owned()));
    }
    if input.target.trim().is_empty() || input.hypothesis_ids.len() < 2 || input.experiments.is_empty() {
        return Err(CovarianceDesignError::InvalidInput("target, at least two hypotheses, and at least one experiment are required".to_owned()));
    }
    let hypotheses: BTreeSet<&str> = input.hypothesis_ids.iter().map(String::as_str).collect();
    if hypotheses.len() != input.hypothesis_ids.len() || hypotheses.iter().any(|id| id.trim().is_empty()) {
        return Err(CovarianceDesignError::InvalidInput("hypothesis IDs must be non-empty and unique".to_owned()));
    }
    let expected_pairs = input.hypothesis_ids.len() * (input.hypothesis_ids.len() - 1) / 2;
    let mut experiment_ids = BTreeSet::new();

    for experiment in &input.experiments {
        let eid = &experiment.experiment_id;
        if eid.trim().is_empty() || !experiment_ids.insert(eid.as_str()) {
            return Err(CovarianceDesignError::InvalidInput(format!("empty or duplicate experiment ID: {eid}")));
        }
        if experiment.description.trim().is_empty()
            || experiment.settings.is_empty()
            || !experiment.cost_units.is_finite()
            || experiment.cost_units <= 0.0
            || !experiment.duration_hours.is_finite()
            || experiment.duration_hours < 0.0
            || experiment.observables.is_empty() {
            return Err(CovarianceDesignError::InvalidInput(format!("invalid description, settings, cost, duration, or observables for {eid}")));
        }
        let mut setting_ids = BTreeSet::new();
        if experiment.settings.iter().any(|s| s.parameter_id.trim().is_empty()
            || s.unit.trim().is_empty() || !s.value.is_finite()
            || !setting_ids.insert(s.parameter_id.as_str())) {
            return Err(CovarianceDesignError::InvalidInput(format!("invalid or duplicate parameter setting in {eid}")));
        }
        let mut observable_ids = BTreeSet::new();
        if experiment.observables.iter().any(|o| o.observable_id.trim().is_empty()
            || o.unit.trim().is_empty() || !observable_ids.insert(o.observable_id.as_str())) {
            return Err(CovarianceDesignError::InvalidInput(format!("invalid or duplicate observable in {eid}")));
        }

        if experiment.predictions.len() != input.hypothesis_ids.len() {
            return Err(CovarianceDesignError::InvalidInput(format!("wrong prediction count in {eid}")));
        }
        let mut prediction_ids = BTreeSet::new();
        for prediction in &experiment.predictions {
            if !hypotheses.contains(prediction.hypothesis_id.as_str())
                || !prediction_ids.insert(prediction.hypothesis_id.as_str())
                || prediction.means.len() != experiment.observables.len()
                || prediction.means.iter().any(|v| !v.is_finite())
                || prediction.model_descriptor.trim().is_empty() {
                return Err(CovarianceDesignError::InvalidInput(format!("invalid vector prediction for {} in {eid}", prediction.hypothesis_id)));
            }
            if !descriptor_digest_matches(&prediction.model_descriptor, &prediction.model_digest_sha256) {
                return Err(CovarianceDesignError::DigestMismatch(format!("model descriptor for {} in {eid}", prediction.hypothesis_id)));
            }
        }

        if experiment.pairwise_difference_covariances.len() != expected_pairs {
            return Err(CovarianceDesignError::InvalidInput(format!("wrong covariance pair count in {eid}")));
        }
        let mut supplied_pairs = BTreeSet::new();
        for pair in &experiment.pairwise_difference_covariances {
            let first = pair.first_hypothesis_id.as_str();
            let second = pair.second_hypothesis_id.as_str();
            if first >= second || !hypotheses.contains(first) || !hypotheses.contains(second) {
                return Err(CovarianceDesignError::InvalidInput(format!("unknown or non-canonical covariance pair ({first}, {second}) in {eid}")));
            }
            if !supplied_pairs.insert((first.to_owned(), second.to_owned())) {
                return Err(CovarianceDesignError::InvalidInput(format!("duplicate covariance pair ({first}, {second}) in {eid}")));
            }
            let n = experiment.observables.len();
            if pair.covariance.len() != n || pair.covariance.iter().any(|row| row.len() != n) {
                return Err(CovarianceDesignError::InvalidCovariance(format!("wrong matrix shape for ({first}, {second}) in {eid}")));
            }
            if pair.covariance.iter().flatten().any(|v| !v.is_finite()) {
                return Err(CovarianceDesignError::InvalidCovariance(format!("non-finite entry for ({first}, {second}) in {eid}")));
            }
            for i in 0..n {
                for j in 0..n {
                    let a = pair.covariance[i][j];
                    let b = pair.covariance[j][i];
                    if (a - b).abs() > 1.0e-12 * a.abs().max(b.abs()) {
                        return Err(CovarianceDesignError::InvalidCovariance(format!("asymmetric matrix for ({first}, {second}) in {eid}")));
                    }
                }
                if pair.covariance[i][i] < 0.0 {
                    return Err(CovarianceDesignError::InvalidCovariance(format!("negative diagonal for ({first}, {second}) in {eid}")));
                }
            }
            if pair.covariance_descriptor.trim().is_empty()
                || !descriptor_digest_matches(&pair.covariance_descriptor, &pair.covariance_descriptor_digest_sha256) {
                return Err(CovarianceDesignError::DigestMismatch(format!("covariance provenance descriptor for ({first}, {second}) in {eid}")));
            }
            if matrix_digest_sha256(&pair.covariance) != pair.covariance_matrix_digest_sha256 {
                return Err(CovarianceDesignError::DigestMismatch(format!("numeric covariance matrix for ({first}, {second}) in {eid}")));
            }
            let correlation = correlation_matrix(&pair.covariance)
                .ok_or_else(|| CovarianceDesignError::InvalidCovariance(format!("nonzero covariance with a zero-variance observable for ({first}, {second}) in {eid}")))?;
            let (eigenvalues, _) = jacobi_symmetric_eigen(&correlation)
                .ok_or_else(|| CovarianceDesignError::NumericalFailure(format!("eigendecomposition for ({first}, {second}) in {eid}")))?;
            let max_eigenvalue = eigenvalues.iter().copied().fold(0.0_f64, f64::max).max(1.0);
            if let Some(minimum) = eigenvalues.iter().copied().reduce(f64::min) {
                if minimum < -NEGATIVE_EIGENVALUE_TOLERANCE * max_eigenvalue {
                    return Err(CovarianceDesignError::NonPsdCovariance(format!("minimum eigenvalue {minimum} for ({first}, {second}) in {eid}")));
                }
            }
        }
        for i in 0..input.hypothesis_ids.len() {
            for j in (i + 1)..input.hypothesis_ids.len() {
                let (first, second) = if input.hypothesis_ids[i] < input.hypothesis_ids[j] {
                    (input.hypothesis_ids[i].clone(), input.hypothesis_ids[j].clone())
                } else {
                    (input.hypothesis_ids[j].clone(), input.hypothesis_ids[i].clone())
                };
                if !supplied_pairs.contains(&(first.clone(), second.clone())) {
                    return Err(CovarianceDesignError::InvalidInput(format!("missing covariance pair ({first}, {second}) in {eid}")));
                }
            }
        }
    }
    Ok(())
}

fn correlation_matrix(covariance: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = covariance.len();
    let scales: Vec<f64> = (0..n).map(|i| covariance[i][i].sqrt()).collect();
    let mut correlation = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if scales[i] == 0.0 || scales[j] == 0.0 {
                if covariance[i][j] != 0.0 {
                    return None;
                }
            } else {
                correlation[i][j] = covariance[i][j] / scales[i] / scales[j];
                if !correlation[i][j].is_finite() {
                    return None;
                }
            }
        }
    }
    Some(correlation)
}

fn descriptor_digest_matches(descriptor: &str, expected: &str) -> bool {
    let digest = Sha256::digest(descriptor.as_bytes());
    format!("{digest:x}") == expected
}

/// SHA-256 over a version tag, dimensions, then row-major f64 bit patterns.
pub fn matrix_digest_sha256(matrix: &[Vec<f64>]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-covariance-matrix-f64-be-v1\0");
    hasher.update((matrix.len() as u64).to_be_bytes());
    for row in matrix {
        hasher.update((row.len() as u64).to_be_bytes());
        for value in row {
            hasher.update(value.to_bits().to_be_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn canonical_settings(settings: &[ParameterSetting]) -> Vec<ParameterSetting> {
    let mut output = settings.to_vec();
    output.sort_by(|a, b| a.parameter_id.cmp(&b.parameter_id));
    output
}

/// Synthetic exact-known-answer fixture; not an EGB calculation.
pub fn covariance_redundancy_fixture() -> CovarianceAwareDesignInput {
    serde_json::from_str(include_str!("../fixtures/covariance-redundancy-v0.1.json"))
        .expect("covariance regression fixture must remain valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_aware_score_is_four_before_and_after_redundancy() {
        let report = rank_covariance_aware_experiments(&covariance_redundancy_fixture()).unwrap();
        assert_eq!(report.ranked_experiments.len(), 2);
        for candidate in &report.ranked_experiments {
            assert!((candidate.worst_case_separation_squared - 4.0).abs() < 1.0e-10);
            assert!((candidate.worst_case_separation - 2.0).abs() < 1.0e-10);
        }
        assert!(report.support_review_required.is_empty());
    }

    #[test]
    fn duplicated_observable_does_not_increase_the_score() {
        let report = rank_covariance_aware_experiments(&covariance_redundancy_fixture()).unwrap();
        let base = report.ranked_experiments.iter().find(|x| x.experiment_id == "base_two_observables").unwrap();
        let redundant = report.ranked_experiments.iter().find(|x| x.experiment_id == "with_exact_redundant_sum").unwrap();
        assert!((base.worst_case_separation_squared - redundant.worst_case_separation_squared).abs() < 1.0e-10);
        assert_eq!(1.0_f64.powi(2) / 1.0 + 2.0_f64.powi(2) / 1.0, 5.0);
        assert_eq!(1.0_f64.powi(2) / 1.0 + 2.0_f64.powi(2) / 1.0 + 3.0_f64.powi(2) / 3.0, 8.0);
    }

    #[test]
    fn diagonal_unit_rescaling_preserves_the_score() {
        let mut input = covariance_redundancy_fixture();
        let experiment = input.experiments.iter_mut().find(|x| x.experiment_id == "base_two_observables").unwrap();
        experiment.observables[0].unit = "millibase-unit".to_owned();
        for prediction in &mut experiment.predictions {
            prediction.means[0] *= 1000.0;
        }
        for pair in &mut experiment.pairwise_difference_covariances {
            for row in &mut pair.covariance {
                row[0] *= 1000.0;
            }
            for value in &mut pair.covariance[0] {
                *value *= 1000.0;
            }
            pair.covariance_matrix_digest_sha256 = matrix_digest_sha256(&pair.covariance);
        }
        let report = rank_covariance_aware_experiments(&input).unwrap();
        let score = report.ranked_experiments.iter().find(|x| x.experiment_id == "base_two_observables").unwrap().worst_case_separation_squared;
        assert!((score - 4.0).abs() < 1.0e-10);
    }

    #[test]
    fn invertible_observable_basis_change_preserves_the_score() {
        let mut input = covariance_redundancy_fixture();
        let experiment = input.experiments.iter_mut()
            .find(|x| x.experiment_id == "base_two_observables")
            .unwrap();

        // A = [[2, 1], [-1, 1]], det(A) = 3, so this is invertible.
        // Transport every hypothesis mean by A and the covariance by A S A^T.
        for prediction in &mut experiment.predictions {
            let x = prediction.means[0];
            let y = prediction.means[1];
            prediction.means = vec![2.0 * x + y, -x + y];
        }
        experiment.observables[0].unit = "basis-row-1".to_owned();
        experiment.observables[1].unit = "basis-row-2".to_owned();

        let pair = &mut experiment.pairwise_difference_covariances[0];
        pair.covariance = vec![vec![7.0, -0.5], vec![-0.5, 1.0]];
        pair.covariance_matrix_digest_sha256 = matrix_digest_sha256(&pair.covariance);

        let report = rank_covariance_aware_experiments(&input).unwrap();
        let score = report.ranked_experiments.iter()
            .find(|x| x.experiment_id == "base_two_observables")
            .unwrap()
            .worst_case_separation_squared;
        assert!((score - 4.0).abs() < 1.0e-10);
    }

    #[test]
    fn singular_covariance_outside_support_is_quarantined() {
        let mut input = covariance_redundancy_fixture();
        let experiment = &mut input.experiments[0];
        experiment.predictions.iter_mut().find(|p| p.hypothesis_id == "H1").unwrap().means = vec![1.0, 0.0];
        let pair = &mut experiment.pairwise_difference_covariances[0];
        pair.covariance = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        pair.covariance_matrix_digest_sha256 = matrix_digest_sha256(&pair.covariance);
        let target = experiment.experiment_id.clone();
        let report = rank_covariance_aware_experiments(&input).unwrap();
        assert!(report.ranked_experiments.iter().all(|x| x.experiment_id != target));
        assert!(report.support_review_required.iter().any(|x| x.experiment_id == target));
    }

    #[test]
    fn singular_covariance_inside_support_is_scored() {
        let mut input = covariance_redundancy_fixture();
        let experiment = &mut input.experiments[0];
        experiment.predictions.iter_mut().find(|p| p.hypothesis_id == "H1").unwrap().means = vec![1.0, 1.0];
        let pair = &mut experiment.pairwise_difference_covariances[0];
        pair.covariance = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        pair.covariance_matrix_digest_sha256 = matrix_digest_sha256(&pair.covariance);
        let target = experiment.experiment_id.clone();
        let report = rank_covariance_aware_experiments(&input).unwrap();
        let score = report.ranked_experiments.iter().find(|x| x.experiment_id == target).unwrap().worst_case_separation_squared;
        assert!((score - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn indefinite_covariance_and_tampered_matrix_are_rejected() {
        let mut input = covariance_redundancy_fixture();
        let pair = &mut input.experiments[0].pairwise_difference_covariances[0];
        pair.covariance = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        pair.covariance_matrix_digest_sha256 = matrix_digest_sha256(&pair.covariance);
        assert!(matches!(rank_covariance_aware_experiments(&input), Err(CovarianceDesignError::NonPsdCovariance(_))));

        let mut input = covariance_redundancy_fixture();
        input.experiments[0].pairwise_difference_covariances[0].covariance[0][0] = 1.25;
        assert!(matches!(rank_covariance_aware_experiments(&input), Err(CovarianceDesignError::DigestMismatch(_))));
    }

    #[test]
    fn missing_covariance_pair_and_non_finite_values_are_rejected() {
        let mut input = covariance_redundancy_fixture();
        input.experiments[0].pairwise_difference_covariances.clear();
        assert!(matches!(rank_covariance_aware_experiments(&input), Err(CovarianceDesignError::InvalidInput(_))));

        let mut input = covariance_redundancy_fixture();
        input.experiments[0].pairwise_difference_covariances[0].covariance[0][0] = f64::NAN;
        input.experiments[0].pairwise_difference_covariances[0].covariance_matrix_digest_sha256 =
            matrix_digest_sha256(&input.experiments[0].pairwise_difference_covariances[0].covariance);
        assert!(matches!(rank_covariance_aware_experiments(&input), Err(CovarianceDesignError::InvalidCovariance(_))));
    }
}
