//! Exact finite Bayesian model averaging for Gaussian predictive models.
//!
//! Each model supplies the predictive distribution of a new observation and
//! its cross-covariance with one scalar scientific quantity theta. The
//! observation updates model weights by its prequential predictive density;
//! within-model theta moments use exact Gaussian conditioning. The mixture
//! variance includes both within-model and between-model terms.
//!
//! The robust support set is independent of Bayesian prior/posterior weights.
//! A zero-weight model may therefore remain in the worst-case evaluation set.

const TOL: f64 = 1.0e-9;
const PSD_TOL: f64 = 1.0e-10;
const PIVOT_FLOOR: f64 = 1.0e-300;

pub type Matrix = Vec<Vec<f64>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    EmptyModelSet,
    InvalidModelIdentity,
    DuplicateModelIdentity,
    InvalidPriorWeights,
    InvalidObservation,
    InvalidPredictiveDimensions,
    InvalidPredictiveCovariance,
    NonPositiveDefinitePredictiveCovariance,
    InvalidScientificMoments,
    InvalidJointCovariance,
    InvalidRobustSupportSet,
    NumericFailure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussianModel {
    pub model_id: String,
    /// Prior probability for Bayesian model averaging. Zero is allowed so
    /// the model may remain in the independent robust support set.
    pub prior_weight: f64,
    pub theta_mean: f64,
    pub theta_variance: f64,
    pub predictive_mean: Vec<f64>,
    /// Predictive covariance S for the next observation under this model.
    pub innovation_covariance: Matrix,
    /// Cov(theta, y) under this model, one element per observation dimension.
    pub theta_observation_covariance: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelPosterior {
    pub model_id: String,
    pub prior_weight: f64,
    pub log_predictive_density: f64,
    pub posterior_weight: f64,
    pub theta_posterior_mean: f64,
    pub theta_posterior_variance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelAveragingReport {
    pub components: Vec<ModelPosterior>,
    pub mixture_theta_mean: f64,
    pub within_model_variance: f64,
    pub between_model_variance: f64,
    pub total_mixture_variance: f64,
    pub robust_support_ids: Vec<String>,
    /// Worst component squared-error risk for the common Bayesian action
    /// theta_hat = mixture_theta_mean; not a Bayesian probability average.
    pub robust_worst_case_squared_error_at_mixture_mean: f64,
}

fn square_dimension(matrix: &[Vec<f64>]) -> Result<usize, Error> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return Err(Error::InvalidPredictiveDimensions);
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Error::InvalidPredictiveCovariance);
    }
    Ok(n)
}

fn cholesky(matrix: &[Vec<f64>]) -> Result<Matrix, Error> {
    let n = square_dimension(matrix)?;
    let mut lower = vec![vec![0.0; n]; n];
    let scale = matrix.iter().flatten().map(|v| v.abs()).fold(0.0_f64, f64::max);
    for i in 0..n {
        for j in 0..i {
            if (matrix[i][j] - matrix[j][i]).abs() > TOL * scale.max(f64::MIN_POSITIVE) {
                return Err(Error::InvalidPredictiveCovariance);
            }
        }
    }
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i][j];
            for k in 0..j {
                value -= lower[i][k] * lower[j][k];
            }
            if i == j {
                if !value.is_finite() || value <= PIVOT_FLOOR {
                    return Err(Error::NonPositiveDefinitePredictiveCovariance);
                }
                lower[i][j] = value.sqrt();
            } else {
                lower[i][j] = value / lower[j][j];
                if !lower[i][j].is_finite() {
                    return Err(Error::NumericFailure);
                }
            }
        }
    }
    Ok(lower)
}

fn solve_cholesky(lower: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, Error> {
    let n = lower.len();
    if rhs.len() != n {
        return Err(Error::InvalidPredictiveDimensions);
    }
    let mut forward = vec![0.0; n];
    for i in 0..n {
        let mut value = rhs[i];
        for j in 0..i {
            value -= lower[i][j] * forward[j];
        }
        forward[i] = value / lower[i][i];
    }
    let mut result = vec![0.0; n];
    for i in (0..n).rev() {
        let mut value = forward[i];
        for j in (i + 1)..n {
            value -= lower[j][i] * result[j];
        }
        result[i] = value / lower[i][i];
    }
    if result.iter().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(Error::NumericFailure)
    }
}

fn log_predictive_density(
    observation: &[f64],
    model: &GaussianModel,
    lower: &[Vec<f64>],
) -> Result<(f64, Vec<f64>), Error> {
    let n = observation.len();
    if model.predictive_mean.len() != n
        || model.theta_observation_covariance.len() != n
    {
        return Err(Error::InvalidPredictiveDimensions);
    }
    let innovation: Vec<f64> = observation.iter()
        .zip(&model.predictive_mean)
        .map(|(y, mean)| y - mean)
        .collect();
    let solved = solve_cholesky(lower, &innovation)?;
    let quadratic: f64 = innovation.iter().zip(&solved).map(|(a, b)| a * b).sum();
    let log_determinant = 2.0 * (0..n).map(|i| lower[i][i].ln()).sum::<f64>();
    let log_density = -0.5 * (
        (n as f64) * (2.0 * std::f64::consts::PI).ln()
        + log_determinant
        + quadratic
    );
    if !log_density.is_finite() || !quadratic.is_finite() {
        return Err(Error::NumericFailure);
    }
    Ok((log_density, solved))
}

fn validate_models(
    models: &[GaussianModel],
    observation: &[f64],
    robust_support_ids: &[String],
) -> Result<Vec<Matrix>, Error> {
    if models.is_empty() {
        return Err(Error::EmptyModelSet);
    }
    if observation.is_empty() || observation.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidObservation);
    }

    let mut prior_sum = 0.0;
    let mut factors = Vec::with_capacity(models.len());
    for (index, model) in models.iter().enumerate() {
        if model.model_id.trim().is_empty() {
            return Err(Error::InvalidModelIdentity);
        }
        if models[..index].iter().any(|earlier| earlier.model_id == model.model_id) {
            return Err(Error::DuplicateModelIdentity);
        }
        if !model.prior_weight.is_finite()
            || model.prior_weight < 0.0
            || model.prior_weight > 1.0
        {
            return Err(Error::InvalidPriorWeights);
        }
        prior_sum += model.prior_weight;
        if !model.theta_mean.is_finite()
            || !model.theta_variance.is_finite()
            || model.theta_variance < 0.0
        {
            return Err(Error::InvalidScientificMoments);
        }
        if model.predictive_mean.len() != observation.len()
            || model.theta_observation_covariance.len() != observation.len()
        {
            return Err(Error::InvalidPredictiveDimensions);
        }
        if model.predictive_mean.iter().any(|value| !value.is_finite())
            || model.theta_observation_covariance.iter().any(|value| !value.is_finite())
        {
            return Err(Error::InvalidScientificMoments);
        }
        let lower = cholesky(&model.innovation_covariance)?;
        if lower.len() != observation.len() {
            return Err(Error::InvalidPredictiveDimensions);
        }
        // Check that the supplied joint Gaussian covariance is valid:
        // Var(theta | y) = Var(theta) - c^T S^-1 c >= 0.
        let solved_cross = solve_cholesky(&lower, &model.theta_observation_covariance)?;
        let explained: f64 = model.theta_observation_covariance.iter()
            .zip(&solved_cross).map(|(a, b)| a * b).sum();
        let tolerance = PSD_TOL * model.theta_variance.abs().max(explained.abs()).max(1.0);
        if !explained.is_finite() || explained < -tolerance
            || model.theta_variance - explained < -tolerance
        {
            return Err(Error::InvalidJointCovariance);
        }
        factors.push(lower);
    }
    if !prior_sum.is_finite() || (prior_sum - 1.0).abs() > TOL {
        return Err(Error::InvalidPriorWeights);
    }

    if robust_support_ids.is_empty() {
        return Err(Error::InvalidRobustSupportSet);
    }
    for (index, support_id) in robust_support_ids.iter().enumerate() {
        if support_id.trim().is_empty()
            || robust_support_ids[..index].contains(support_id)
            || !models.iter().any(|model| model.model_id == *support_id)
        {
            return Err(Error::InvalidRobustSupportSet);
        }
    }
    Ok(factors)
}

fn posterior_weighted(log_weights: &[f64]) -> Result<Vec<f64>, Error> {
    let maximum = log_weights.iter().copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return Err(Error::NumericFailure);
    }
    let unnormalized: Vec<f64> = log_weights.iter()
        .map(|value| if value.is_finite() { (value - maximum).exp() } else { 0.0 })
        .collect();
    let total: f64 = unnormalized.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::NumericFailure);
    }
    Ok(unnormalized.iter().map(|value| value / total).collect())
}

/// Update model probabilities using prequential predictive densities and
/// produce the exact scalar-theta Gaussian-mixture moments.
pub fn update_models(
    models: &[GaussianModel],
    observation: &[f64],
    robust_support_ids: &[String],
) -> Result<ModelAveragingReport, Error> {
    let factors = validate_models(models, observation, robust_support_ids)?;
    let mut log_densities = Vec::with_capacity(models.len());
    let mut log_weights = Vec::with_capacity(models.len());
    let mut posterior_means = Vec::with_capacity(models.len());
    let mut posterior_variances = Vec::with_capacity(models.len());

    for (model, lower) in models.iter().zip(&factors) {
        let (log_likelihood, solved_innovation) =
            log_predictive_density(observation, model, lower)?;
        let solved_cross = solve_cholesky(lower, &model.theta_observation_covariance)?;
        let mean_update: f64 = model.theta_observation_covariance.iter()
            .zip(&solved_innovation).map(|(a, b)| a * b).sum();
        let explained_variance: f64 = model.theta_observation_covariance.iter()
            .zip(&solved_cross).map(|(a, b)| a * b).sum();
        let posterior_variance = model.theta_variance - explained_variance;
        let tolerance = PSD_TOL * model.theta_variance.abs().max(explained_variance.abs()).max(1.0);
        if posterior_variance < -tolerance || !posterior_variance.is_finite() {
            return Err(Error::InvalidJointCovariance);
        }
        let posterior_mean = model.theta_mean + mean_update;
        if !posterior_mean.is_finite() {
            return Err(Error::NumericFailure);
        }

        log_densities.push(log_likelihood);
        log_weights.push(if model.prior_weight > 0.0 {
            model.prior_weight.ln() + log_likelihood
        } else {
            f64::NEG_INFINITY
        });
        posterior_means.push(posterior_mean);
        posterior_variances.push(posterior_variance.max(0.0));
    }

    let posterior_weights = posterior_weighted(&log_weights)?;
    let mixture_mean: f64 = posterior_weights.iter()
        .zip(&posterior_means).map(|(w, mean)| w * mean).sum();
    let within_variance: f64 = posterior_weights.iter()
        .zip(&posterior_variances).map(|(w, variance)| w * variance).sum();
    let between_variance: f64 = posterior_weights.iter()
        .zip(&posterior_means).map(|(w, mean)| w * (mean - mixture_mean).powi(2)).sum();
    let total_variance = within_variance + between_variance;
    if !mixture_mean.is_finite()
        || !within_variance.is_finite()
        || !between_variance.is_finite()
        || !total_variance.is_finite()
    {
        return Err(Error::NumericFailure);
    }

    let components: Vec<ModelPosterior> = models.iter().enumerate().map(|(i, model)| {
        ModelPosterior {
            model_id: model.model_id.clone(),
            prior_weight: model.prior_weight,
            log_predictive_density: log_densities[i],
            posterior_weight: posterior_weights[i],
            theta_posterior_mean: posterior_means[i],
            theta_posterior_variance: posterior_variances[i],
        }
    }).collect();

    let mut robust_worst_case = f64::NEG_INFINITY;
    for support_id in robust_support_ids {
        let index = models.iter().position(|model| model.model_id == *support_id)
            .ok_or(Error::InvalidRobustSupportSet)?;
        let squared_error_risk = posterior_variances[index]
            + (posterior_means[index] - mixture_mean).powi(2);
        robust_worst_case = robust_worst_case.max(squared_error_risk);
    }

    Ok(ModelAveragingReport {
        components,
        mixture_theta_mean: mixture_mean,
        within_model_variance: within_variance,
        between_model_variance: between_variance,
        total_mixture_variance: total_variance,
        robust_support_ids: robust_support_ids.to_vec(),
        robust_worst_case_squared_error_at_mixture_mean: robust_worst_case,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(
        id: &str, weight: f64, theta_mean: f64, theta_variance: f64,
        predictive_mean: Vec<f64>, innovation_covariance: Matrix, cross: Vec<f64>,
    ) -> GaussianModel {
        GaussianModel {
            model_id: id.to_string(),
            prior_weight: weight,
            theta_mean,
            theta_variance,
            predictive_mean,
            innovation_covariance,
            theta_observation_covariance: cross,
        }
    }

    fn fixture_models() -> Vec<GaussianModel> {
        vec![
            model("fast-drift",0.7,0.0,2.0,vec![0.0],vec![vec![1.0]],vec![0.8]),
            model("slow-drift",0.3,2.0,3.0,vec![1.0],vec![vec![2.0]],vec![1.0]),
        ]
    }

    fn close(a:f64,b:f64,tol:f64) {
        assert!((a-b).abs()<=tol,"actual={a:.15}, expected={b:.15}");
    }

    #[test]
    fn predictive_density_updates_model_weights() {
        let r=update_models(&fixture_models(),&[0.5],&["fast-drift".into(),"slow-drift".into()]).unwrap();
        close(r.components[0].posterior_weight,0.7560919066448245,1.0e-13);
        close(r.components[1].posterior_weight,0.2439080933551755,1.0e-13);
        close(r.components.iter().map(|c|c.posterior_weight).sum(),1.0,1.0e-14);
    }

    #[test]
    fn total_variance_includes_between_model_component() {
        let r=update_models(&fixture_models(),&[0.5],&["fast-drift".into(),"slow-drift".into()]).unwrap();
        close(r.mixture_theta_mean,0.729275926029487,1.0e-13);
        close(r.within_model_variance,1.6380552264249,1.0e-13);
        close(r.between_model_variance,0.3360998646772312,1.0e-13);
        close(r.total_mixture_variance,1.9741550911021313,1.0e-13);
        assert!(r.total_mixture_variance>r.within_model_variance);
    }

    #[test]
    fn robust_support_is_not_pruned_by_small_posterior_weight() {
        let mut models=fixture_models();
        models.push(model("rare-but-plausible",0.0,5.0,4.0,vec![0.5],vec![vec![1.0]],vec![0.0]));
        let support=vec!["fast-drift".into(),"rare-but-plausible".into()];
        let r=update_models(&models,&[0.5],&support).unwrap();
        assert_eq!(r.components[2].posterior_weight,0.0);
        assert_eq!(r.robust_support_ids,support);
        assert!(r.robust_worst_case_squared_error_at_mixture_mean>4.0);
    }

    #[test]
    fn multivariate_innovation_uses_full_covariance() {
        let m=model(
            "coupled",1.0,1.2,3.5,vec![0.0,1.0],
            vec![vec![2.0,0.5],vec![0.5,1.0]],vec![0.4,0.2]
        );
        let r=update_models(&[m],&[1.0,0.0],&["coupled".into()]).unwrap();
        close(r.components[0].log_predictive_density,-3.2605421032341995,1.0e-13);
        close(r.components[0].theta_posterior_mean,1.2571428571428571,1.0e-13);
        close(r.components[0].theta_posterior_variance,3.4085714285714284,1.0e-13);
    }

    #[test]
    fn zero_observation_cross_covariance_does_not_move_theta_mean() {
        let m=model("independent",1.0,3.0,2.0,vec![0.0],vec![vec![1.0]],vec![0.0]);
        let r=update_models(&[m],&[100.0],&["independent".into()]).unwrap();
        close(r.components[0].theta_posterior_mean,3.0,0.0);
        close(r.components[0].theta_posterior_variance,2.0,0.0);
    }

    #[test]
    fn severe_likelihood_difference_is_normalized_in_log_space() {
        let models=vec![
            model("near",0.5,0.0,1.0,vec![0.0],vec![vec![1.0]],vec![0.2]),
            model("far",0.5,0.0,1.0,vec![0.0],vec![vec![0.1]],vec![0.1]),
        ];
        let r=update_models(&models,&[10.0],&["near".into(),"far".into()]).unwrap();
        close(r.components.iter().map(|c|c.posterior_weight).sum(),1.0,1.0e-14);
        assert!(r.components.iter().all(|c|c.posterior_weight.is_finite()));
    }

    #[test]
    fn invalid_predictive_covariance_and_joint_moments_fail_closed() {
        let asymmetric=model("bad-a",1.0,0.0,1.0,vec![0.0,0.0],
            vec![vec![1.0,0.2],vec![0.0,1.0]],vec![0.0,0.0]);
        assert_eq!(
            update_models(&[asymmetric],&[0.0,0.0],&["bad-a".into()]),
            Err(Error::InvalidPredictiveCovariance)
        );
        let impossible=model("bad-joint",1.0,0.0,0.1,vec![0.0],
            vec![vec![1.0]],vec![1.0]);
        assert_eq!(
            update_models(&[impossible],&[0.0],&["bad-joint".into()]),
            Err(Error::InvalidJointCovariance)
        );
    }

    #[test]
    fn zero_prior_weight_component_can_remain_in_robust_set() {
        let models=vec![
            model("bayesian",1.0,0.0,1.0,vec![0.0],vec![vec![1.0]],vec![0.2]),
            model("support-only",0.0,8.0,2.0,vec![0.0],vec![vec![1.0]],vec![0.0]),
        ];
        let r=update_models(&models,&[0.3],&["bayesian".into(),"support-only".into()]).unwrap();
        assert_eq!(r.components[1].posterior_weight,0.0);
        assert!(r.robust_worst_case_squared_error_at_mixture_mean>2.0);
    }

    #[test]
    fn invalid_prior_sum_and_support_ids_are_rejected() {
        let bad=vec![
            model("a",0.4,0.0,1.0,vec![0.0],vec![vec![1.0]],vec![0.1]),
            model("b",0.4,0.0,1.0,vec![0.0],vec![vec![1.0]],vec![0.1]),
        ];
        assert_eq!(
            update_models(&bad,&[0.0],&["a".into()]),
            Err(Error::InvalidPriorWeights)
        );
        assert_eq!(
            update_models(&fixture_models(),&[0.0],&["missing".into()]),
            Err(Error::InvalidRobustSupportSet)
        );
    }

    #[test]
    fn mixture_with_one_model_reduces_to_that_model() {
        let m=model("single",1.0,2.0,1.4,vec![0.0],vec![vec![1.0]],vec![0.3]);
        let r=update_models(&[m],&[0.5],&["single".into()]).unwrap();
        close(r.mixture_theta_mean,2.15,1.0e-14);
        close(r.total_mixture_variance,1.31,1.0e-14);
        close(r.between_model_variance,0.0,0.0);
    }
}
