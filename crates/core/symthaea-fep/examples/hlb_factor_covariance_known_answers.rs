// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact-head known-answer fixture for the low-rank factor covariance kernel.
//!
//! Output is synthetic numerical-reference evidence only. It is not a runtime
//! benchmark, covariance-estimation result, physical calibration or SOTA claim.

use serde_json::json;
use symthaea_fep::factor_covariance::FactorCovarianceModel;

fn fixture(dimension: usize, rank: usize) -> FactorCovarianceModel {
    FactorCovarianceModel {
        model_version: "factor-covariance-v1".to_owned(),
        feature_ids: (0..dimension)
            .map(|index| format!("sensor-{index:04}"))
            .collect(),
        feature_units: (0..dimension)
            .map(|_| "normalized".to_owned())
            .collect(),
        diagonal_variances: (0..dimension)
            .map(|index| ((index % 5) as f64 * 0.35 - 0.7).exp())
            .collect(),
        factor_ids: (0..rank)
            .map(|index| format!("shared-mode-{index:03}"))
            .collect(),
        loadings: (0..dimension)
            .map(|row| {
                (0..rank)
                    .map(|column| (((row + 1) * (column + 2)) as f64).sin() * 0.04)
                    .collect()
            })
            .collect(),
        calibration_provenance_digest: "a".repeat(64),
    }
}

/// Independent dense Cholesky reference. This implementation does not call
/// the production Woodbury or covariance-materialization helpers.
fn dense_reference(model: &FactorCovarianceModel, residual: &[f64]) -> f64 {
    let dimension = model.dimension();
    let rank = model.rank();
    let mut covariance = vec![vec![0.0; dimension]; dimension];

    for row in 0..dimension {
        covariance[row][row] = model.diagonal_variances[row];
        for column in 0..dimension {
            for factor in 0..rank {
                covariance[row][column] +=
                    model.loadings[row][factor] * model.loadings[column][factor];
            }
        }
    }

    let mut lower = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = covariance[row][column];
            for inner in 0..column {
                value -= lower[row][inner] * lower[column][inner];
            }
            if row == column {
                lower[row][column] = value.sqrt();
            } else {
                lower[row][column] = value / lower[column][column];
            }
        }
    }

    let mut forward = vec![0.0; dimension];
    for row in 0..dimension {
        let mut value = residual[row];
        for column in 0..row {
            value -= lower[row][column] * forward[column];
        }
        forward[row] = value / lower[row][row];
    }

    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut value = forward[row];
        for column in (row + 1)..dimension {
            value -= lower[column][row] * solution[column];
        }
        solution[row] = value / lower[row][row];
    }

    residual.iter().zip(solution).map(|(left, right)| left * right).sum()
}

fn run_case(
    dimension: usize,
    rank: usize,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let model = fixture(dimension, rank);
    let residual = (0..dimension)
        .map(|index| ((index + 3) as f64 * 0.17).cos())
        .collect::<Vec<_>>();
    let woodbury = model.mahalanobis_squared(&residual)?;
    let dense = dense_reference(&model, &residual);
    let absolute_error = (woodbury - dense).abs();

    Ok(json!({
        "dimension": dimension,
        "rank": rank,
        "woodbury_mahalanobis_squared": woodbury,
        "dense_reference_mahalanobis_squared": dense,
        "absolute_error": absolute_error,
        "model_digest": model.model_digest()?
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dimensions_and_ranks = [(2, 1), (3, 2), (8, 4), (32, 4), (128, 8), (512, 8)];
    let cases = dimensions_and_ranks
        .into_iter()
        .map(|(dimension, rank)| run_case(dimension, rank))
        .collect::<Result<Vec<_>, _>>()?;
    let maximum_absolute_error = cases
        .iter()
        .filter_map(|case| case.get("absolute_error").and_then(serde_json::Value::as_f64))
        .fold(0.0_f64, f64::max);
    let output = json!({
        "schema_version": "hlb-factor-covariance-known-answers-v1",
        "status": "synthetic_known_answer_fixture_only",
        "source_sha": std::env::var("SYMTHAEA_SOURCE_SHA").ok(),
        "maximum_absolute_error": maximum_absolute_error,
        "cases": cases
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
