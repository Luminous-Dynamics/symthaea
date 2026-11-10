//! Reproducible known-answer and negative-control run for correlated Gaussian diagnostics.
//!
//! This is a synthetic mathematical fixture, not a sensor-data or deployment result.

use serde_json::json;
use symthaea_fep::multivariate_gaussian_decision_diagnostic::{
    MultivariateGaussianDecisionDiagnosticConfig, MultivariateGaussianDecisionHypothesis,
    MultivariateGaussianDiagnosticAction, evaluate_multivariate_gaussian_decision_diagnostics,
};

fn run_case(
    name: &str,
    means: [Vec<f64>; 2],
    covariance: Vec<Vec<f64>>,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let dimension = means[0].len();
    let hypotheses = vec![
        MultivariateGaussianDecisionHypothesis {
            id: "H0".to_owned(),
            prior_probability: 0.5,
            mean_by_action: vec![means[0].clone()],
            loss_by_final_decision: vec![0.0, 1.0],
        },
        MultivariateGaussianDecisionHypothesis {
            id: "H1".to_owned(),
            prior_probability: 0.5,
            mean_by_action: vec![means[1].clone()],
            loss_by_final_decision: vec![1.0, 0.0],
        },
    ];
    let actions = vec![MultivariateGaussianDiagnosticAction {
        id: name.to_owned(),
        feature_ids: (0..dimension)
            .map(|index| format!("sensor-{index}"))
            .collect(),
        covariance,
        observation_cost: 0.0,
        risk_score: 0.0,
    }];
    let result = evaluate_multivariate_gaussian_decision_diagnostics(
        &hypotheses,
        &actions,
        MultivariateGaussianDecisionDiagnosticConfig::default(),
        &"a".repeat(64),
    )?;
    Ok(json!({
        "case": name,
        "dimension": dimension,
        "prior_bayes_risk": result.prior_bayes_risk,
        "mahalanobis_squared": result.actions[0].mahalanobis_squared,
        "expected_posterior_bayes_risk": result.actions[0].expected_posterior_bayes_risk,
        "evsi": result.actions[0].expected_value_of_sample_information,
        "problem_digest": result.problem_digest,
        "evaluation_digest": result.evaluation_digest,
        "normal_cdf_approximation_error_envelope":
            result.actions[0].normal_cdf_approximation_error_envelope
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root_three = 3.0_f64.sqrt();
    let two_dimensional = run_case(
        "correlated_2d",
        [vec![0.0, 0.0], vec![root_three, root_three]],
        vec![vec![1.0, 0.5], vec![0.5, 1.0]],
    )?;
    let independent_negative_control = run_case(
        "independent_marginal_negative_control",
        [vec![0.0, 0.0], vec![root_three, root_three]],
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    )?;
    let component = (8.0_f64 / 3.0).sqrt();
    let three_dimensional = run_case(
        "correlated_3d",
        [vec![0.0; 3], vec![component; 3]],
        vec![
            vec![1.0, 0.5, 0.5],
            vec![0.5, 1.0, 0.5],
            vec![0.5, 0.5, 1.0],
        ],
    )?;
    let source_sha = std::env::var("SYMTHAEA_SOURCE_SHA").ok();
    let output = json!({
        "schema_version": "hlb-multivariate-gaussian-known-answers-v1",
        "status": "synthetic_known_answer_fixture_only",
        "source_sha": source_sha,
        "cases": [
            two_dimensional,
            independent_negative_control,
            three_dimensional
        ]
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
