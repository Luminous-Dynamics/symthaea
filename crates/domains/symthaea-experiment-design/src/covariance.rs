// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Covariance-aware, prior-free experiment discrimination.
//!
//! The caller supplies one PSD covariance for the *stacked output vector*
//! across all hypotheses. Pairwise Cov(Y_i - Y_j) matrices are derived from
//! its blocks, so they are coherent by construction and retain common-error
//! and cross-hypothesis terms.
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

pub const COVARIANCE_SCHEMA_VERSION: &str = "physics-experiment-design-joint-covariance-v0.2";
pub const COVARIANCE_ALGORITHM_VERSION: &str = "joint-output-correlation-whitened-jacobi-v0.2";
pub const RELATIVE_EIGENVALUE_TOLERANCE: f64 = 1.0e-12;
pub const RELATIVE_SUPPORT_TOLERANCE: f64 = 1.0e-10;
const NEGATIVE_EIGENVALUE_TOLERANCE: f64 = 1.0e-10;
const JACOBI_CONVERGENCE_TOLERANCE: f64 = 1.0e-14;
const MAX_HYPOTHESES: usize = 8;
const MAX_OBSERVABLES: usize = 8;
const MAX_JOINT_COVARIANCE_DIMENSION: usize = 16;
const MAX_EXPERIMENTS: usize = 256;

/// Complete design request. Each experiment provides a single joint covariance
/// over all hypothesis × observable outputs, in the explicit canonical order.
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
    pub joint_output_covariance: JointOutputCovariance,
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

/// One joint covariance over all competing hypotheses' output vectors.
///
/// Rows and columns use hypothesis-major order:
/// sorted hypothesis IDs, then the experiment's declared observable order.
/// The pairwise difference covariance is derived as
/// Gamma_ii + Gamma_jj - Gamma_ij - Gamma_ji. This avoids accepting
/// individually valid but mutually impossible pairwise covariance inputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointOutputCovariance {
    pub hypothesis_order: Vec<String>,
    pub observable_order: Vec<String>,
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
            Self::NonPsdCovariance(message) => {
                write!(f, "covariance is not positive semidefinite: {message}")
            }
            Self::NumericalFailure(message) => write!(f, "covariance numerical failure: {message}"),
        }
    }
}
impl Error for CovarianceDesignError {}

/// Rank by the minimum pairwise Mahalanobis separation, then divide
/// by caller-supplied relative cost. The caller supplies one PSD covariance
/// for the complete stacked vector (Y_1, ..., Y_H), with each Y_i ordered by
/// the experiment's declared observables. Pairwise difference covariance is
/// derived from the joint blocks:
///
/// Cov(Y_i - Y_j) = Gamma_ii + Gamma_jj - Gamma_ij - Gamma_ji.
///
/// Supplying the full joint covariance makes cross-hypothesis and
/// cross-observable dependence explicit and guarantees that all derived
/// pairwise difference covariances come from one common uncertainty model.
/// The caller must justify the coupling/shared-noise semantics in the
/// covariance descriptor. If no defensible joint coupling exists, use
/// interval/set-valued design rather than inventing one.
///
/// The matrix is correlation-standardized before eigendecomposition. This
/// makes eigenvalue cutoff and support decisions invariant to positive
/// diagonal unit rescaling (for example, metres to millimetres).
///
/// If a mean difference has a material component in a zero-variance direction,
/// the candidate is emitted in support_review_required and excluded from
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
            .predictions
            .iter()
            .map(|prediction| (prediction.hypothesis_id.as_str(), prediction))
            .collect();
        let joint = &experiment.joint_output_covariance.covariance;
        let observable_count = experiment.observables.len();

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
                let difference_covariance =
                    derive_difference_covariance(joint, i, j, observable_count);
                match mahalanobis_squared(&delta, &difference_covariance)? {
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
                "non-finite score for experiment {}",
                experiment.experiment_id
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
            observable_count,
        });
    }

    ranked.sort_by(|a, b| {
        b.separation_per_cost_unit
            .total_cmp(&a.separation_per_cost_unit)
            .then_with(|| a.experiment_id.cmp(&b.experiment_id))
    });
    for (index, candidate) in ranked.iter_mut().enumerate() {
        candidate.rank = index + 1;
    }
    review.sort_by(|a, b| {
        a.experiment_id
            .cmp(&b.experiment_id)
            .then_with(|| a.first_hypothesis_id.cmp(&b.first_hypothesis_id))
            .then_with(|| a.second_hypothesis_id.cmp(&b.second_hypothesis_id))
    });

    Ok(CovarianceDesignReport {
        schema_version: COVARIANCE_SCHEMA_VERSION.to_owned(),
        algorithm_version: COVARIANCE_ALGORITHM_VERSION.to_owned(),
        target: input.target.clone(),
        criterion: "prior-free maximin Mahalanobis separation of rival mean vectors derived from one caller-supplied joint output covariance, divided by declared relative cost".to_owned(),
        relative_eigenvalue_tolerance: RELATIVE_EIGENVALUE_TOLERANCE,
        relative_support_tolerance: RELATIVE_SUPPORT_TOLERANCE,
        ranked_experiments: ranked,
        support_review_required: review,
        limitations: vec![
            "This is a covariance-geometry screening statistic, not expected information gain, a Bayes factor, a posterior, or a truth oracle.".to_owned(),
            "The full joint covariance is supplied over the stacked hypothesis-output vector; all pairwise difference covariances are derived from its blocks and are jointly coherent by construction.".to_owned(),
            "The covariance descriptor must justify how rival hypotheses are coupled and which shared model, calibration, measurement, and numerical-error sources are included.".to_owned(),
            "The descriptor digest binds provenance text; the matrix digest binds numeric matrix values. Neither proves that an executable generated the supplied values.".to_owned(),
            "A mean outside a singular covariance support is quarantined for review, not assigned a finite pseudoinverse score or an automatic infinite score.".to_owned(),
            "Covariance uncertainty sets, nuisance profiling, non-Gaussian predictive distributions, and calibrated Monte Carlo EIG are not implemented in this v0.2 selector.".to_owned(),
            "Cost is caller-defined relative units. Solver calls, CPU/GPU, memory, wall time, tokens, and human review must also be reported separately.".to_owned(),
            "Predictions are not observations. Symtropy execution and independent qualification remain separate authorities.".to_owned(),
        ],
    })
}

/// Derive Cov(Y_i - Y_j) from one joint covariance over the stacked
/// hypothesis-major vector (Y_0, ..., Y_H-1).
fn derive_difference_covariance(
    joint: &[Vec<f64>],
    first_hypothesis: usize,
    second_hypothesis: usize,
    observable_count: usize,
) -> Vec<Vec<f64>> {
    let mut difference = vec![vec![0.0; observable_count]; observable_count];
    for (a, row) in difference.iter_mut().enumerate() {
        for (b, cell) in row.iter_mut().enumerate() {
            let first_a = first_hypothesis * observable_count + a;
            let first_b = first_hypothesis * observable_count + b;
            let second_a = second_hypothesis * observable_count + a;
            let second_b = second_hypothesis * observable_count + b;
            let forward = joint[first_a][first_b] + joint[second_a][second_b]
                - joint[first_a][second_b]
                - joint[second_a][first_b];

            let reverse = joint[first_b][first_a] + joint[second_b][second_a]
                - joint[first_b][second_a]
                - joint[second_b][first_a];

            // The exact block formula is symmetric. Average both floating-point
            // evaluation orders. The transposed iteration evaluates the same two
            // terms in reverse order, so both matrix triangles are bit-identical.
            *cell = 0.5 * forward + 0.5 * reverse;
        }
    }
    difference
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
        let diagonal = covariance[i][i];
        if !diagonal.is_finite() {
            return Err(CovarianceDesignError::InvalidCovariance(
                "derived difference covariance has a non-finite diagonal".to_owned(),
            ));
        }
        if diagonal < 0.0 {
            return Err(CovarianceDesignError::NonPsdCovariance(format!(
                "derived difference covariance has negative variance {diagonal} at observable {i}"
            )));
        }
        if diagonal == 0.0 {
            for j in 0..n {
                if covariance[i][j] != 0.0 || covariance[j][i] != 0.0 {
                    return Err(CovarianceDesignError::InvalidCovariance(format!(
                        "derived difference covariance has nonzero cross-covariance with zero-variance observable {i}"
                    )));
                }
            }
            if delta[i] != 0.0 {
                let norm = delta.iter().fold(0.0_f64, |acc, value| acc.hypot(*value));
                return Ok(PairScore::OutsideSupport {
                    residual: delta[i].abs(),
                    norm,
                });
            }
        } else {
            scales[i] = diagonal.sqrt();
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
    let norm = standardized_delta
        .iter()
        .fold(0.0_f64, |acc, value| acc.hypot(*value));
    let (eigenvalues, eigenvectors) = jacobi_symmetric_eigen(&correlation).ok_or_else(|| {
        CovarianceDesignError::NumericalFailure("Jacobi eigensolver did not converge".to_owned())
    })?;
    let max_eigenvalue = eigenvalues.iter().copied().fold(0.0_f64, f64::max).max(1.0);
    let cutoff = RELATIVE_EIGENVALUE_TOLERANCE * max_eigenvalue;
    let negative_cutoff = NEGATIVE_EIGENVALUE_TOLERANCE * max_eigenvalue;
    if let Some(minimum) = eigenvalues.iter().copied().reduce(f64::min)
        && minimum < -negative_cutoff
    {
        return Err(CovarianceDesignError::NonPsdCovariance(format!(
            "minimum standardized eigenvalue {minimum}"
        )));
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
        return Err(CovarianceDesignError::NumericalFailure(
            "non-finite pseudoinverse quadratic form".to_owned(),
        ));
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
        for (i, row) in a.iter().enumerate() {
            diagonal_scale = diagonal_scale.max(row[i].abs());
            for (j, value) in row.iter().enumerate().skip(i + 1) {
                if value.abs() > largest {
                    largest = value.abs();
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
        || vectors.iter().flatten().any(|v| !v.is_finite())
    {
        return None;
    }
    Some((eigenvalues, vectors))
}

fn validate_input(input: &CovarianceAwareDesignInput) -> Result<(), CovarianceDesignError> {
    if input.schema_version != COVARIANCE_SCHEMA_VERSION {
        return Err(CovarianceDesignError::InvalidInput(
            "unsupported schema version".to_owned(),
        ));
    }
    if input.target.trim().is_empty()
        || input.hypothesis_ids.len() < 2
        || input.experiments.is_empty()
    {
        return Err(CovarianceDesignError::InvalidInput(
            "target, at least two hypotheses, and at least one experiment are required".to_owned(),
        ));
    }
    if input.hypothesis_ids.len() > MAX_HYPOTHESES || input.experiments.len() > MAX_EXPERIMENTS {
        return Err(CovarianceDesignError::InvalidInput(format!(
            "input exceeds resource caps: at most {MAX_HYPOTHESES} hypotheses and {MAX_EXPERIMENTS} experiments"
        )));
    }
    let hypotheses: BTreeSet<&str> = input.hypothesis_ids.iter().map(String::as_str).collect();
    if hypotheses.len() != input.hypothesis_ids.len()
        || hypotheses.iter().any(|id| id.trim().is_empty())
    {
        return Err(CovarianceDesignError::InvalidInput(
            "hypothesis IDs must be non-empty and unique".to_owned(),
        ));
    }
    let mut canonical_hypotheses = input.hypothesis_ids.clone();
    canonical_hypotheses.sort();

    let mut experiment_ids = BTreeSet::new();
    for experiment in &input.experiments {
        let eid = &experiment.experiment_id;
        if eid.trim().is_empty() || !experiment_ids.insert(eid.as_str()) {
            return Err(CovarianceDesignError::InvalidInput(format!(
                "empty or duplicate experiment ID: {eid}"
            )));
        }
        if experiment.observables.len() > MAX_OBSERVABLES
            || experiment
                .observables
                .len()
                .saturating_mul(input.hypothesis_ids.len())
                > MAX_JOINT_COVARIANCE_DIMENSION
        {
            return Err(CovarianceDesignError::InvalidInput(format!(
                "stacked joint covariance dimension exceeds {MAX_JOINT_COVARIANCE_DIMENSION} in {eid}"
            )));
        }
        if experiment.description.trim().is_empty()
            || experiment.settings.is_empty()
            || !experiment.cost_units.is_finite()
            || experiment.cost_units <= 0.0
            || !experiment.duration_hours.is_finite()
            || experiment.duration_hours < 0.0
            || experiment.observables.is_empty()
        {
            return Err(CovarianceDesignError::InvalidInput(format!(
                "invalid description, settings, cost, duration, or observables for {eid}"
            )));
        }
        let mut setting_ids = BTreeSet::new();
        if experiment.settings.iter().any(|setting| {
            setting.parameter_id.trim().is_empty()
                || setting.unit.trim().is_empty()
                || !setting.value.is_finite()
                || !setting_ids.insert(setting.parameter_id.as_str())
        }) {
            return Err(CovarianceDesignError::InvalidInput(format!(
                "invalid or duplicate parameter setting in {eid}"
            )));
        }
        let mut observable_ids = BTreeSet::new();
        if experiment.observables.iter().any(|observable| {
            observable.observable_id.trim().is_empty()
                || observable.unit.trim().is_empty()
                || !observable_ids.insert(observable.observable_id.as_str())
        }) {
            return Err(CovarianceDesignError::InvalidInput(format!(
                "invalid or duplicate observable in {eid}"
            )));
        }

        if experiment.predictions.len() != input.hypothesis_ids.len() {
            return Err(CovarianceDesignError::InvalidInput(format!(
                "wrong prediction count in {eid}"
            )));
        }
        let mut prediction_ids = BTreeSet::new();
        for prediction in &experiment.predictions {
            if !hypotheses.contains(prediction.hypothesis_id.as_str())
                || !prediction_ids.insert(prediction.hypothesis_id.as_str())
                || prediction.means.len() != experiment.observables.len()
                || prediction.means.iter().any(|value| !value.is_finite())
                || prediction.model_descriptor.trim().is_empty()
            {
                return Err(CovarianceDesignError::InvalidInput(format!(
                    "invalid vector prediction for {} in {eid}",
                    prediction.hypothesis_id
                )));
            }
            if !descriptor_digest_matches(
                &prediction.model_descriptor,
                &prediction.model_digest_sha256,
            ) {
                return Err(CovarianceDesignError::DigestMismatch(format!(
                    "model descriptor for {} in {eid}",
                    prediction.hypothesis_id
                )));
            }
        }

        let covariance = &experiment.joint_output_covariance;
        let canonical_observables: Vec<String> = experiment
            .observables
            .iter()
            .map(|observable| observable.observable_id.clone())
            .collect();
        if covariance.hypothesis_order != canonical_hypotheses
            || covariance.observable_order != canonical_observables
        {
            return Err(CovarianceDesignError::InvalidInput(format!(
                "joint covariance order must be sorted hypothesis IDs followed by the declared observable order in {eid}"
            )));
        }
        let dimension = input.hypothesis_ids.len() * experiment.observables.len();
        if covariance.covariance.len() != dimension
            || covariance
                .covariance
                .iter()
                .any(|row| row.len() != dimension)
        {
            return Err(CovarianceDesignError::InvalidCovariance(format!(
                "joint covariance must be {dimension}×{dimension} in {eid}"
            )));
        }
        if covariance
            .covariance
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(CovarianceDesignError::InvalidCovariance(format!(
                "joint covariance contains a non-finite entry in {eid}"
            )));
        }
        for i in 0..dimension {
            for j in 0..dimension {
                let a = covariance.covariance[i][j];
                let b = covariance.covariance[j][i];
                if (a - b).abs() > 1.0e-12 * a.abs().max(b.abs()) {
                    return Err(CovarianceDesignError::InvalidCovariance(format!(
                        "joint covariance is asymmetric in {eid}"
                    )));
                }
            }
            if covariance.covariance[i][i] < 0.0 {
                return Err(CovarianceDesignError::InvalidCovariance(format!(
                    "joint covariance has a negative diagonal in {eid}"
                )));
            }
        }
        if covariance.covariance_descriptor.trim().is_empty()
            || !descriptor_digest_matches(
                &covariance.covariance_descriptor,
                &covariance.covariance_descriptor_digest_sha256,
            )
        {
            return Err(CovarianceDesignError::DigestMismatch(format!(
                "joint covariance provenance descriptor in {eid}"
            )));
        }
        if matrix_digest_sha256(&covariance.covariance)
            != covariance.covariance_matrix_digest_sha256
        {
            return Err(CovarianceDesignError::DigestMismatch(format!(
                "joint numeric covariance matrix in {eid}"
            )));
        }

        let correlation = correlation_matrix(&covariance.covariance).ok_or_else(|| {
            CovarianceDesignError::InvalidCovariance(format!(
                "joint covariance has invalid zero-variance cross-covariance in {eid}"
            ))
        })?;
        let (eigenvalues, _) = jacobi_symmetric_eigen(&correlation).ok_or_else(|| {
            CovarianceDesignError::NumericalFailure(format!(
                "joint covariance eigendecomposition did not converge in {eid}"
            ))
        })?;
        let max_eigenvalue = eigenvalues.iter().copied().fold(0.0_f64, f64::max).max(1.0);
        let negative_cutoff = NEGATIVE_EIGENVALUE_TOLERANCE * max_eigenvalue;
        if let Some(minimum) = eigenvalues.iter().copied().reduce(f64::min)
            && minimum < -negative_cutoff
        {
            return Err(CovarianceDesignError::NonPsdCovariance(format!(
                "joint covariance minimum standardized eigenvalue {minimum} in {eid}"
            )));
        }
    }
    Ok(())
}

fn correlation_matrix(covariance: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = covariance.len();
    if (0..n).any(|index| covariance[index][index] < 0.0) {
        return None;
    }
    let scales: Vec<f64> = (0..n).map(|i| covariance[i][i].sqrt()).collect();
    let mut correlation = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let value = if scales[i] == 0.0 || scales[j] == 0.0 {
                if covariance[i][j] != 0.0 || covariance[j][i] != 0.0 {
                    return None;
                }
                0.0
            } else {
                let forward = covariance[i][j] / scales[i] / scales[j];
                let reverse = covariance[j][i] / scales[j] / scales[i];
                if !forward.is_finite() || !reverse.is_finite() {
                    return None;
                }
                0.5 * forward + 0.5 * reverse
            };
            if !value.is_finite() {
                return None;
            }
            correlation[i][j] = value;
            correlation[j][i] = value;
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

    fn rehash_joint(experiment: &mut CovarianceAwareCandidateExperiment) {
        let joint = &mut experiment.joint_output_covariance;
        joint.covariance_matrix_digest_sha256 = matrix_digest_sha256(&joint.covariance);
    }

    fn scalar_three_hypothesis_fixture(joint: Vec<Vec<f64>>) -> CovarianceAwareDesignInput {
        let mut input = covariance_redundancy_fixture();
        input.hypothesis_ids = vec!["H0".to_owned(), "H1".to_owned(), "H2".to_owned()];
        input.experiments.truncate(1);
        let experiment = &mut input.experiments[0];
        experiment.experiment_id = "three_scalar_hypotheses".to_owned();
        experiment.observables = vec![CovarianceObservable {
            observable_id: "scalar".to_owned(),
            unit: "u".to_owned(),
        }];
        experiment.predictions = [("H0", 0.0), ("H1", 0.1), ("H2", 2.0)]
            .iter()
            .map(|(id, mean)| {
                let descriptor = format!("synthetic-control:model:{id}:v1");
                VectorHypothesisPrediction {
                    hypothesis_id: (*id).to_owned(),
                    model_digest_sha256: format!("{:x}", Sha256::digest(descriptor.as_bytes())),
                    model_descriptor: descriptor,
                    means: vec![*mean],
                }
            })
            .collect();

        let descriptor = "synthetic-joint-output-covariance:three-hypotheses:v1".to_owned();
        experiment.joint_output_covariance = JointOutputCovariance {
            hypothesis_order: vec!["H0".to_owned(), "H1".to_owned(), "H2".to_owned()],
            observable_order: vec!["scalar".to_owned()],
            covariance_matrix_digest_sha256: matrix_digest_sha256(&joint),
            covariance_descriptor_digest_sha256: format!(
                "{:x}",
                Sha256::digest(descriptor.as_bytes())
            ),
            covariance_descriptor: descriptor,
            covariance: joint,
        };
        input
    }

    #[test]
    fn jacobi_solver_reconstructs_a_deterministic_16_dimensional_psd_matrix() {
        let n = 16;
        let rank = 7;
        let factor: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..rank)
                    .map(|j| {
                        let value = ((i + 1) * (j + 3) % 19) as f64;
                        value / 19.0 + if i == j { 0.25 } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        let mut covariance = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let value: f64 = factor[i]
                    .iter()
                    .zip(&factor[j])
                    .map(|(left, right)| left * right)
                    .sum();
                covariance[i][j] = value;
                covariance[j][i] = value;
            }
        }
        let correlation = correlation_matrix(&covariance).unwrap();
        let (eigenvalues, eigenvectors) = jacobi_symmetric_eigen(&correlation).unwrap();

        let mut max_eigenpair_residual = 0.0_f64;
        let mut max_orthogonality_error = 0.0_f64;
        for i in 0..n {
            for mode in 0..n {
                let product: f64 = (0..n)
                    .map(|j| correlation[i][j] * eigenvectors[j][mode])
                    .sum();
                max_eigenpair_residual = max_eigenpair_residual
                    .max((product - eigenvalues[mode] * eigenvectors[i][mode]).abs());
            }
            for j in 0..n {
                let dot: f64 = (0..n)
                    .map(|mode| eigenvectors[mode][i] * eigenvectors[mode][j])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                max_orthogonality_error = max_orthogonality_error.max((dot - expected).abs());
            }
        }
        assert!(max_eigenpair_residual < 1.0e-10);
        assert!(max_orthogonality_error < 1.0e-10);
    }

    #[test]
    fn derived_difference_covariance_is_exactly_symmetric_after_roundoff() {
        let joint = vec![
            vec![2.0, 0.3, 1.5, 0.2 + 1.0e-14],
            vec![0.3, 1.0, 0.1, 0.75],
            vec![1.5, 0.2, 2.0, 0.4],
            vec![0.2, 0.75, 0.4, 1.5],
        ];
        let difference = derive_difference_covariance(&joint, 0, 1, 2);
        assert_eq!(difference[0][1], difference[1][0]);

        let almost_symmetric = vec![vec![1.0, 0.2 + 1.0e-14], vec![0.2, 1.0]];
        let correlation = correlation_matrix(&almost_symmetric).unwrap();
        assert_eq!(correlation[0][1], correlation[1][0]);
    }

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
        let base = report
            .ranked_experiments
            .iter()
            .find(|x| x.experiment_id == "base_two_observables")
            .unwrap();
        let redundant = report
            .ranked_experiments
            .iter()
            .find(|x| x.experiment_id == "with_exact_redundant_sum")
            .unwrap();
        assert!(
            (base.worst_case_separation_squared - redundant.worst_case_separation_squared).abs()
                < 1.0e-10
        );
        assert_eq!(1.0_f64.powi(2) / 1.0 + 2.0_f64.powi(2) / 1.0, 5.0);
        assert_eq!(
            1.0_f64.powi(2) / 1.0 + 2.0_f64.powi(2) / 1.0 + 3.0_f64.powi(2) / 3.0,
            8.0
        );
    }

    #[test]
    fn cross_hypothesis_common_error_is_subtracted_from_the_difference_covariance() {
        let mut input = covariance_redundancy_fixture();
        input.hypothesis_ids = vec!["H0".to_owned(), "H1".to_owned()];
        input.experiments.truncate(1);
        let experiment = &mut input.experiments[0];
        experiment.experiment_id = "shared_common_error".to_owned();
        experiment.observables = vec![CovarianceObservable {
            observable_id: "response".to_owned(),
            unit: "u".to_owned(),
        }];
        experiment.predictions = [("H0", 0.0), ("H1", 1.0)]
            .iter()
            .map(|(id, mean)| {
                let descriptor = format!("synthetic-control:model:{id}:shared-error-v1");
                VectorHypothesisPrediction {
                    hypothesis_id: (*id).to_owned(),
                    model_digest_sha256: format!("{:x}", Sha256::digest(descriptor.as_bytes())),
                    model_descriptor: descriptor,
                    means: vec![*mean],
                }
            })
            .collect();

        // Both models share covariance 1.5 from a common error source.
        // Marginal variances are 2 each, so Var(Y0-Y1)=2+2-1.5-1.5=1.
        let descriptor = "shared-common-error:two-hypothesis-joint-output:v1".to_owned();
        let gamma = vec![vec![2.0, 1.5], vec![1.5, 2.0]];
        experiment.joint_output_covariance = JointOutputCovariance {
            hypothesis_order: vec!["H0".to_owned(), "H1".to_owned()],
            observable_order: vec!["response".to_owned()],
            covariance_matrix_digest_sha256: matrix_digest_sha256(&gamma),
            covariance_descriptor_digest_sha256: format!(
                "{:x}",
                Sha256::digest(descriptor.as_bytes())
            ),
            covariance_descriptor: descriptor,
            covariance: gamma,
        };

        let report = rank_covariance_aware_experiments(&input).unwrap();
        let score = report.ranked_experiments[0].worst_case_separation_squared;
        assert!((score - 1.0).abs() < 1.0e-10);
        // An incorrect independence assumption would use variance 2 + 2 = 4,
        // yielding D²=0.25 rather than the correctly derived D²=1.
        assert!((1.0_f64 / (2.0 + 2.0) - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn three_hypotheses_derive_all_difference_covariances_from_one_joint_model() {
        // Y0=0, Y1=Z, Y2=2Z for Var(Z)=1 gives
        // Var(Y0-Y1)=1, Var(Y0-Y2)=4 and Var(Y1-Y2)=1.
        let joint = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 4.0],
        ];
        let input = scalar_three_hypothesis_fixture(joint);
        let report = rank_covariance_aware_experiments(&input).unwrap();
        let candidate = &report.ranked_experiments[0];
        assert!((candidate.worst_case_separation_squared - 0.01).abs() < 1.0e-10);
        assert_eq!(
            candidate.closest_hypothesis_pair,
            ["H0".to_owned(), "H1".to_owned()]
        );
    }

    #[test]
    fn invalid_joint_covariance_is_rejected_even_when_each_derived_pair_variance_is_positive() {
        // All derived pairwise scalar difference variances are positive:
        // 0.2, 0.2, 3.8. But this proposed joint covariance has eigenvalue -0.8.
        // A pairwise-only contract could miss this; the joint matrix cannot.
        let joint = vec![
            vec![1.0, 0.9, 0.9],
            vec![0.9, 1.0, -0.9],
            vec![0.9, -0.9, 1.0],
        ];
        let input = scalar_three_hypothesis_fixture(joint);
        assert!(matches!(
            rank_covariance_aware_experiments(&input),
            Err(CovarianceDesignError::NonPsdCovariance(_))
        ));
    }

    #[test]
    fn diagonal_unit_rescaling_preserves_the_score() {
        let mut input = covariance_redundancy_fixture();
        let experiment = input
            .experiments
            .iter_mut()
            .find(|x| x.experiment_id == "base_two_observables")
            .unwrap();
        experiment.observables[0].unit = "millibase-unit".to_owned();
        for prediction in &mut experiment.predictions {
            prediction.means[0] *= 1000.0;
        }
        let h = experiment.joint_output_covariance.hypothesis_order.len();
        let p = experiment.joint_output_covariance.observable_order.len();
        let mut transform = vec![1.0; h * p];
        for hypothesis in 0..h {
            transform[hypothesis * p] = 1000.0;
        }
        for i in 0..h * p {
            for j in 0..h * p {
                experiment.joint_output_covariance.covariance[i][j] *= transform[i] * transform[j];
            }
        }
        rehash_joint(experiment);
        let report = rank_covariance_aware_experiments(&input).unwrap();
        let score = report
            .ranked_experiments
            .iter()
            .find(|x| x.experiment_id == "base_two_observables")
            .unwrap()
            .worst_case_separation_squared;
        assert!((score - 4.0).abs() < 1.0e-10);
    }

    #[test]
    fn joint_covariance_unit_rescaling_is_invariant_for_three_hypotheses() {
        let joint = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0, 4.0],
        ];
        let mut input = scalar_three_hypothesis_fixture(joint);
        let baseline = rank_covariance_aware_experiments(&input).unwrap();
        let baseline_score = baseline.ranked_experiments[0].worst_case_separation_squared;
        let experiment = &mut input.experiments[0];
        experiment.observables[0].unit = "milliu".to_owned();
        for prediction in &mut experiment.predictions {
            prediction.means[0] *= 1000.0;
        }
        for row in &mut experiment.joint_output_covariance.covariance {
            for value in row {
                *value *= 1_000_000.0;
            }
        }
        rehash_joint(experiment);
        let rescaled = rank_covariance_aware_experiments(&input).unwrap();
        let rescaled_score = rescaled.ranked_experiments[0].worst_case_separation_squared;
        assert!((rescaled_score - baseline_score).abs() < 1.0e-10);
    }

    #[test]
    fn invertible_observable_basis_change_preserves_the_score() {
        let mut input = covariance_redundancy_fixture();
        let experiment = input
            .experiments
            .iter_mut()
            .find(|x| x.experiment_id == "base_two_observables")
            .unwrap();
        // A = [[2, 1], [-1, 1]], det(A) = 3.
        let a = [[2.0, 1.0], [-1.0, 1.0]];
        for prediction in &mut experiment.predictions {
            let x = prediction.means[0];
            let y = prediction.means[1];
            prediction.means = vec![a[0][0] * x + a[0][1] * y, a[1][0] * x + a[1][1] * y];
        }
        experiment.observables[0].unit = "basis-row-1".to_owned();
        experiment.observables[1].unit = "basis-row-2".to_owned();

        let h = experiment.joint_output_covariance.hypothesis_order.len();
        let mut transform = vec![vec![0.0; 2 * h]; 2 * h];
        for hypothesis in 0..h {
            for row in 0..2 {
                for column in 0..2 {
                    transform[hypothesis * 2 + row][hypothesis * 2 + column] = a[row][column];
                }
            }
        }
        let gamma = experiment.joint_output_covariance.covariance.clone();
        let mut transformed = vec![vec![0.0; 2 * h]; 2 * h];
        for i in 0..2 * h {
            for j in 0..2 * h {
                for k in 0..2 * h {
                    for l in 0..2 * h {
                        transformed[i][j] += transform[i][k] * gamma[k][l] * transform[j][l];
                    }
                }
            }
        }
        experiment.joint_output_covariance.covariance = transformed;
        rehash_joint(experiment);

        let report = rank_covariance_aware_experiments(&input).unwrap();
        let score = report
            .ranked_experiments
            .iter()
            .find(|x| x.experiment_id == "base_two_observables")
            .unwrap()
            .worst_case_separation_squared;
        assert!((score - 4.0).abs() < 1.0e-10);
    }

    #[test]
    fn negative_derived_difference_variance_is_rejected() {
        let covariance = vec![vec![-1.0e-12]];
        assert!(matches!(
            mahalanobis_squared(&[1.0], &covariance),
            Err(CovarianceDesignError::NonPsdCovariance(_))
        ));
    }

    #[test]
    fn nonzero_cross_covariance_with_zero_variance_is_rejected() {
        let covariance = vec![vec![0.0, 1.0e-14], vec![1.0e-14, 1.0]];
        assert!(matches!(
            mahalanobis_squared(&[0.0, 1.0], &covariance),
            Err(CovarianceDesignError::InvalidCovariance(_))
        ));
    }

    #[test]
    fn singular_joint_covariance_outside_difference_support_is_quarantined() {
        let mut input = covariance_redundancy_fixture();
        let experiment = &mut input.experiments[0];
        experiment
            .predictions
            .iter_mut()
            .find(|prediction| prediction.hypothesis_id == "H1")
            .unwrap()
            .means = vec![1.0, 0.0];
        experiment.joint_output_covariance.covariance = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
        ];
        rehash_joint(experiment);
        let target = experiment.experiment_id.clone();
        let report = rank_covariance_aware_experiments(&input).unwrap();
        assert!(
            report
                .ranked_experiments
                .iter()
                .all(|x| x.experiment_id != target)
        );
        assert!(
            report
                .support_review_required
                .iter()
                .any(|x| x.experiment_id == target)
        );
    }

    #[test]
    fn singular_joint_covariance_inside_difference_support_is_scored() {
        let mut input = covariance_redundancy_fixture();
        let experiment = &mut input.experiments[0];
        experiment
            .predictions
            .iter_mut()
            .find(|prediction| prediction.hypothesis_id == "H1")
            .unwrap()
            .means = vec![1.0, 1.0];
        experiment.joint_output_covariance.covariance = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
        ];
        rehash_joint(experiment);
        let target = experiment.experiment_id.clone();
        let report = rank_covariance_aware_experiments(&input).unwrap();
        let score = report
            .ranked_experiments
            .iter()
            .find(|x| x.experiment_id == target)
            .unwrap()
            .worst_case_separation_squared;
        assert!((score - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn indefinite_joint_covariance_and_tampered_matrix_are_rejected() {
        let mut input = covariance_redundancy_fixture();
        let experiment = &mut input.experiments[0];
        experiment.joint_output_covariance.covariance = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 2.0],
            vec![0.0, 0.0, 2.0, 1.0],
        ];
        rehash_joint(experiment);
        assert!(matches!(
            rank_covariance_aware_experiments(&input),
            Err(CovarianceDesignError::NonPsdCovariance(_))
        ));

        let mut tampered = covariance_redundancy_fixture();
        tampered.experiments[0].joint_output_covariance.covariance[2][2] = 1.25;
        assert!(matches!(
            rank_covariance_aware_experiments(&tampered),
            Err(CovarianceDesignError::DigestMismatch(_))
        ));
    }

    #[test]
    fn joint_covariance_order_dimensions_and_finiteness_are_validated() {
        let mut wrong_order = covariance_redundancy_fixture();
        wrong_order.experiments[0]
            .joint_output_covariance
            .hypothesis_order
            .reverse();
        assert!(matches!(
            rank_covariance_aware_experiments(&wrong_order),
            Err(CovarianceDesignError::InvalidInput(_))
        ));

        let mut wrong_shape = covariance_redundancy_fixture();
        wrong_shape.experiments[0]
            .joint_output_covariance
            .covariance
            .pop();
        rehash_joint(&mut wrong_shape.experiments[0]);
        assert!(matches!(
            rank_covariance_aware_experiments(&wrong_shape),
            Err(CovarianceDesignError::InvalidCovariance(_))
        ));

        let mut non_finite = covariance_redundancy_fixture();
        non_finite.experiments[0].joint_output_covariance.covariance[0][0] = f64::NAN;
        rehash_joint(&mut non_finite.experiments[0]);
        assert!(matches!(
            rank_covariance_aware_experiments(&non_finite),
            Err(CovarianceDesignError::InvalidCovariance(_))
        ));
    }
}
