//! Dependency-free proxy-training reference for the R-IDeA de-amplification term.
//!
//! Build from the repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_ridea_proxy.rs -o /tmp/sdcb-ridea-proxy
//!
//! This implements the released MSE + hinge-margin + L2 objective for a
//! caller-supplied linear feature map. It is a transparent proxy-training
//! reference, NOT the paper's full Pyro/model-specific proxy training pipeline.

const EPS: f64 = 1e-12;
const MAX_ROWS: usize = 4096;
const MAX_FEATURES: usize = 256;
const MAX_STEPS: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyError {
    EmptyTrainingSet,
    InvalidDimensions,
    NonFiniteInput,
    InvalidConfiguration,
    SearchSpaceTooLarge,
    NumericalFailure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProxyTrainConfig {
    /// Hinge threshold tau: encourage |f_hat(x) - g(x)| >= tau.
    pub margin_tau: f64,
    /// Weight alpha on the mean hinge-margin penalty.
    pub margin_alpha: f64,
    /// Weight on mean squared proxy coefficients.
    pub l2: f64,
    pub learning_rate: f64,
    pub steps: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProxyFitResult {
    pub weights: Vec<f64>,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub mse: f64,
    pub margin_penalty: f64,
    pub l2_penalty: f64,
    pub mean_disagreement: f64,
    pub minimum_disagreement: f64,
    pub maximum_disagreement: f64,
    pub steps_run: usize,
}

fn validate_features(features: &[Vec<f64>]) -> Result<usize, ProxyError> {
    if features.is_empty() {
        return Err(ProxyError::EmptyTrainingSet);
    }
    if features.len() > MAX_ROWS || features[0].is_empty() || features[0].len() > MAX_FEATURES {
        return Err(ProxyError::SearchSpaceTooLarge);
    }
    let dimension = features[0].len();
    if features.iter().any(|row| row.len() != dimension) {
        return Err(ProxyError::InvalidDimensions);
    }
    if features.iter().flatten().any(|x| !x.is_finite()) {
        return Err(ProxyError::NonFiniteInput);
    }
    Ok(dimension)
}

fn validate_inputs(
    features: &[Vec<f64>],
    targets: &[f64],
    current_predictions: &[f64],
    initial_weights: &[f64],
    config: &ProxyTrainConfig,
) -> Result<(), ProxyError> {
    let dimension = validate_features(features)?;
    if targets.len() != features.len()
        || current_predictions.len() != features.len()
        || initial_weights.len() != dimension
    {
        return Err(ProxyError::InvalidDimensions);
    }
    if targets.iter().chain(current_predictions).chain(initial_weights)
        .any(|x| !x.is_finite())
    {
        return Err(ProxyError::NonFiniteInput);
    }
    if !config.margin_tau.is_finite()
        || config.margin_tau < 0.0
        || !config.margin_alpha.is_finite()
        || config.margin_alpha < 0.0
        || !config.l2.is_finite()
        || config.l2 < 0.0
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
    {
        return Err(ProxyError::InvalidConfiguration);
    }
    if config.steps > MAX_STEPS {
        return Err(ProxyError::SearchSpaceTooLarge);
    }
    Ok(())
}

pub fn linear_predict(weights: &[f64], features: &[f64]) -> Result<f64, ProxyError> {
    if weights.len() != features.len() || weights.is_empty() {
        return Err(ProxyError::InvalidDimensions);
    }
    if weights.iter().chain(features).any(|x| !x.is_finite()) {
        return Err(ProxyError::NonFiniteInput);
    }
    let prediction: f64 = weights.iter().zip(features).map(|(w, x)| w * x).sum();
    if !prediction.is_finite() {
        return Err(ProxyError::NumericalFailure);
    }
    Ok(prediction)
}

#[derive(Debug, Clone, PartialEq)]
struct LossBreakdown {
    total: f64,
    mse: f64,
    margin: f64,
    regularization: f64,
    mean_gap: f64,
    min_gap: f64,
    max_gap: f64,
}

fn loss_breakdown(
    features: &[Vec<f64>],
    targets: &[f64],
    current_predictions: &[f64],
    weights: &[f64],
    config: &ProxyTrainConfig,
) -> Result<LossBreakdown, ProxyError> {
    let n = features.len() as f64;
    let d = weights.len() as f64;
    let mut mse = 0.0;
    let mut margin = 0.0;
    let mut mean_gap = 0.0;
    let mut min_gap = f64::INFINITY;
    let mut max_gap = f64::NEG_INFINITY;

    for (i, row) in features.iter().enumerate() {
        let prediction = linear_predict(weights, row)?;
        let residual = prediction - targets[i];
        let gap = (current_predictions[i] - prediction).abs();
        mse += residual * residual / n;
        margin += (config.margin_tau - gap).max(0.0) / n;
        mean_gap += gap / n;
        min_gap = min_gap.min(gap);
        max_gap = max_gap.max(gap);
    }

    let regularization = config.l2 * weights.iter().map(|w| w * w).sum::<f64>() / d;
    let total = mse + config.margin_alpha * margin + regularization;
    if [mse, margin, regularization, total, mean_gap, min_gap, max_gap]
        .iter()
        .any(|x| !x.is_finite())
    {
        return Err(ProxyError::NumericalFailure);
    }

    Ok(LossBreakdown {
        total,
        mse,
        margin,
        regularization,
        mean_gap,
        min_gap,
        max_gap,
    })
}

/// Evaluate the released objective:
///
/// L(g) = mean((g(x)-y)^2)
///      + alpha * mean(relu(tau - |f_hat(x)-g(x)|))
///      + l2 * mean(weight^2).
///
/// current_predictions are frozen values from f_hat; they are not optimized.
pub fn proxy_training_loss(
    features: &[Vec<f64>],
    targets: &[f64],
    current_predictions: &[f64],
    weights: &[f64],
    config: &ProxyTrainConfig,
) -> Result<f64, ProxyError> {
    validate_inputs(features, targets, current_predictions, weights, config)?;
    Ok(loss_breakdown(features, targets, current_predictions, weights, config)?.total)
}

fn objective_gradient(
    features: &[Vec<f64>],
    targets: &[f64],
    current_predictions: &[f64],
    weights: &[f64],
    config: &ProxyTrainConfig,
) -> Result<Vec<f64>, ProxyError> {
    let n = features.len() as f64;
    let d = weights.len() as f64;
    let mut gradient = vec![0.0; weights.len()];

    for (i, row) in features.iter().enumerate() {
        let prediction = linear_predict(weights, row)?;
        let residual = prediction - targets[i];
        let signed_gap = current_predictions[i] - prediction;

        for j in 0..weights.len() {
            gradient[j] += 2.0 * residual * row[j] / n;

            if signed_gap.abs() < config.margin_tau {
                // d/dg relu(tau - |f_hat - g|) = sign(f_hat - g)
                // away from the nondifferentiable equality point.
                gradient[j] += config.margin_alpha * signed_gap.signum() * row[j] / n;
            }
        }
    }

    for j in 0..weights.len() {
        gradient[j] += 2.0 * config.l2 * weights[j] / d;
    }

    if gradient.iter().any(|x| !x.is_finite()) {
        return Err(ProxyError::NumericalFailure);
    }
    Ok(gradient)
}

/// Fit a linear proxy with full-batch gradient descent.
///
/// The initial coefficients should normally be the current model's coefficients,
/// mirroring the upstream R-IDeA helper's initialization from theta_loc. A
/// caller-supplied richer feature map can represent nonlinear basis functions.
pub fn fit_linear_proxy(
    features: &[Vec<f64>],
    targets: &[f64],
    current_predictions: &[f64],
    initial_weights: &[f64],
    config: &ProxyTrainConfig,
) -> Result<ProxyFitResult, ProxyError> {
    validate_inputs(features, targets, current_predictions, initial_weights, config)?;
    let initial = loss_breakdown(
        features,
        targets,
        current_predictions,
        initial_weights,
        config,
    )?;

    let mut weights = initial_weights.to_vec();
    let mut steps_run = 0usize;

    for _ in 0..config.steps {
        let gradient = objective_gradient(
            features,
            targets,
            current_predictions,
            &weights,
            config,
        )?;
        let next: Vec<f64> = weights
            .iter()
            .zip(gradient.iter())
            .map(|(weight, grad)| weight - config.learning_rate * grad)
            .collect();

        if next.iter().any(|x| !x.is_finite()) {
            return Err(ProxyError::NumericalFailure);
        }
        weights = next;
        steps_run += 1;
    }

    let final_loss = loss_breakdown(
        features,
        targets,
        current_predictions,
        &weights,
        config,
    )?;

    Ok(ProxyFitResult {
        weights,
        initial_loss: initial.total,
        final_loss: final_loss.total,
        mse: final_loss.mse,
        margin_penalty: final_loss.margin,
        l2_penalty: final_loss.regularization,
        mean_disagreement: final_loss.mean_gap,
        minimum_disagreement: final_loss.min_gap,
        maximum_disagreement: final_loss.max_gap,
        steps_run,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ProxyTrainConfig {
        ProxyTrainConfig {
            margin_tau: 0.5,
            margin_alpha: 0.0,
            l2: 0.0,
            learning_rate: 0.05,
            steps: 100,
        }
    }

    #[test]
    fn linear_prediction_matches_hand_calculation() {
        assert!((linear_predict(&[2.0, -1.0], &[3.0, 4.0]).unwrap() - 2.0).abs() < EPS);
    }

    #[test]
    fn mse_only_training_reduces_prediction_error() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let f = vec![0.0, 0.0, 0.0];
        let result = fit_linear_proxy(&x, &y, &f, &[0.0], &config()).unwrap();
        assert!(result.final_loss < result.initial_loss);
        assert!((result.weights[0] - 2.0).abs() < 0.02);
    }

    #[test]
    fn margin_penalty_pushes_proxy_away_from_current_predictor() {
        let x = vec![vec![1.0]];
        let y = vec![0.1];
        let f = vec![0.0];
        let initial = [0.05];
        let mut cfg = config();
        cfg.margin_tau = 0.5;
        cfg.margin_alpha = 2.0;
        cfg.learning_rate = 0.02;
        cfg.steps = 100;
        let result = fit_linear_proxy(&x, &y, &f, &initial, &cfg).unwrap();
        assert!(result.weights[0] > initial[0]);
        assert!(result.mean_disagreement > 0.5);
    }

    #[test]
    fn margin_zero_recovers_mse_only_objective() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![1.0, 2.0];
        let f = vec![0.0, 0.0];
        let mut a = config();
        a.margin_alpha = 0.0;
        let mut b = a.clone();
        b.margin_tau = 5.0;
        assert_eq!(
            fit_linear_proxy(&x, &y, &f, &[0.0], &a).unwrap().weights,
            fit_linear_proxy(&x, &y, &f, &[0.0], &b).unwrap().weights
        );
    }

    #[test]
    fn l2_regularization_reduces_weight_norm() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];
        let f = vec![0.0, 0.0, 0.0];
        let mut no_reg = config();
        no_reg.learning_rate = 0.05;
        let mut reg = no_reg.clone();
        reg.l2 = 1.0;
        let a = fit_linear_proxy(&x, &y, &f, &[0.0], &no_reg).unwrap();
        let b = fit_linear_proxy(&x, &y, &f, &[0.0], &reg).unwrap();
        assert!(b.weights[0].abs() < a.weights[0].abs());
    }

    #[test]
    fn zero_steps_preserves_initial_weights_and_reports_loss() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![1.0, 2.0];
        let f = vec![0.0, 0.0];
        let mut cfg = config();
        cfg.steps = 0;
        let initial = vec![0.3];
        let result = fit_linear_proxy(&x, &y, &f, &initial, &cfg).unwrap();
        assert_eq!(result.weights, initial);
        assert_eq!(result.steps_run, 0);
    }

    #[test]
    fn gradient_matches_finite_difference_away_from_hinge_kink() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![0.2, 0.4];
        let f = vec![0.15, 0.4];
        let w = vec![0.1];
        let mut cfg = config();
        cfg.margin_tau = 0.5;
        cfg.margin_alpha = 0.7;
        cfg.l2 = 0.2;
        let analytic = objective_gradient(&x, &y, &f, &w, &cfg).unwrap()[0];
        let eps = 1e-6;
        let mut plus = w.clone();
        plus[0] += eps;
        let mut minus = w.clone();
        minus[0] -= eps;
        let l_plus = proxy_training_loss(&x, &y, &f, &plus, &cfg).unwrap();
        let l_minus = proxy_training_loss(&x, &y, &f, &minus, &cfg).unwrap();
        let numeric = (l_plus - l_minus) / (2.0 * eps);
        assert!((analytic - numeric).abs() < 1e-6);
    }

    #[test]
    fn empty_training_data_is_rejected() {
        let cfg = config();
        assert_eq!(
            fit_linear_proxy(&[], &[], &[], &[], &cfg),
            Err(ProxyError::EmptyTrainingSet)
        );
    }

    #[test]
    fn inconsistent_dimensions_are_rejected() {
        let cfg = config();
        assert_eq!(
            fit_linear_proxy(
                &[vec![1.0], vec![1.0, 2.0]],
                &[1.0, 2.0],
                &[0.0, 0.0],
                &[0.0],
                &cfg
            ),
            Err(ProxyError::InvalidDimensions)
        );
    }

    #[test]
    fn non_finite_data_is_rejected() {
        let cfg = config();
        assert_eq!(
            fit_linear_proxy(
                &[vec![f64::NAN]],
                &[0.0],
                &[0.0],
                &[0.0],
                &cfg
            ),
            Err(ProxyError::NonFiniteInput)
        );
    }

    #[test]
    fn invalid_margin_or_learning_rate_is_rejected() {
        let x = vec![vec![1.0]];
        let y = vec![0.0];
        let f = vec![0.0];
        let mut cfg = config();
        cfg.margin_tau = -1.0;
        assert_eq!(
            fit_linear_proxy(&x, &y, &f, &[0.0], &cfg),
            Err(ProxyError::InvalidConfiguration)
        );
        cfg.margin_tau = 0.5;
        cfg.learning_rate = 0.0;
        assert_eq!(
            fit_linear_proxy(&x, &y, &f, &[0.0], &cfg),
            Err(ProxyError::InvalidConfiguration)
        );
    }

    #[test]
    fn non_finite_gradient_state_fails_closed() {
        let x = vec![vec![1.0]];
        let y = vec![1e308];
        let f = vec![0.0];
        let mut cfg = config();
        cfg.learning_rate = 1e308;
        cfg.steps = 2;
        assert_eq!(
            fit_linear_proxy(&x, &y, &f, &[0.0], &cfg),
            Err(ProxyError::NumericalFailure)
        );
    }

    #[test]
    fn iteration_cap_is_enforced() {
        let x = vec![vec![1.0]];
        let y = vec![1.0];
        let f = vec![0.0];
        let mut cfg = config();
        cfg.steps = MAX_STEPS + 1;
        assert_eq!(
            fit_linear_proxy(&x, &y, &f, &[0.0], &cfg),
            Err(ProxyError::SearchSpaceTooLarge)
        );
    }
}
