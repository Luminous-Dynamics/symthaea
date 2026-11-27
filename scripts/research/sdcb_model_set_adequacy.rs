//! Open-world adequacy gate for finite Gaussian covariance-model sets.
//!
//! A normalized Bayesian model-weight vector does not prove that any
//! registered model is adequate. This module reuses the exact Gaussian
//! mixture evaluator and flags observations whose best positive-prior model
//! fails a threshold frozen on independent chronological validation data.
//!
//! The threshold is supplied by the caller; this module does not learn it
//! from the observation being assessed.

#[path = "sdcb_covariance_model_mixture.rs"]
mod mixture;

use mixture::{GaussianModel, ModelAveragingReport};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSetStatus {
    WithinCalibratedEnvelope,
    FlaggedInadequate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelSetAdequacyReport {
    pub mixture: ModelAveragingReport,
    pub best_supported_model_id: String,
    pub best_log_predictive_density: f64,
    /// Log predictive density under the Bayesian mixture, using prior weights.
    pub mixture_log_predictive_density: f64,
    pub minimum_acceptable_best_model_log_density: f64,
    pub status: ModelSetStatus,
    /// False means downstream scientific-decision updates must abstain or
    /// invoke an explicitly authorized model-expansion/review path.
    pub decision_eligible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidAdequacyThreshold,
    NoPositivePriorModel,
    NumericFailure,
    Mixture(mixture::Error),
}

impl From<mixture::Error> for Error {
    fn from(value: mixture::Error) -> Self {
        Error::Mixture(value)
    }
}

fn log_sum_exp(values: &[f64]) -> Result<f64, Error> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(Error::NumericFailure);
    }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return Err(Error::NumericFailure);
    }
    let sum: f64 = values.iter().map(|value| (value - maximum).exp()).sum();
    let result = maximum + sum.ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(Error::NumericFailure)
    }
}

/// Assess whether at least one model with nonzero declared Bayesian prior
/// probability passes a preregistered held-out predictive-score threshold.
///
/// This is a model-set inadequacy alarm, not proof that the passing model is
/// physically true or that the model family is complete. The threshold must
/// be calibrated on independent chronological validation data for the same
/// observation units, dimensions and reporting protocol.
pub fn assess_model_set(
    models: &[GaussianModel],
    observation: &[f64],
    robust_support_ids: &[String],
    minimum_acceptable_best_model_log_density: f64,
) -> Result<ModelSetAdequacyReport, Error> {
    if !minimum_acceptable_best_model_log_density.is_finite() {
        return Err(Error::InvalidAdequacyThreshold);
    }
    let mixture_report = mixture::update_models(models, observation, robust_support_ids)?;

    let mut best: Option<(String, f64)> = None;
    let mut weighted_log_densities = Vec::new();
    for component in &mixture_report.components {
        if component.prior_weight <= 0.0 {
            continue;
        }
        weighted_log_densities.push(
            component.prior_weight.ln() + component.log_predictive_density
        );
        let replace = match &best {
            None => true,
            Some((best_id, best_score)) => {
                component.log_predictive_density > *best_score
                    || (component.log_predictive_density == *best_score
                        && component.model_id < *best_id)
            }
        };
        if replace {
            best = Some((
                component.model_id.clone(),
                component.log_predictive_density,
            ));
        }
    }
    let (best_supported_model_id, best_log_predictive_density) =
        best.ok_or(Error::NoPositivePriorModel)?;
    let mixture_log_predictive_density = log_sum_exp(&weighted_log_densities)?;
    let decision_eligible =
        best_log_predictive_density >= minimum_acceptable_best_model_log_density;

    Ok(ModelSetAdequacyReport {
        mixture: mixture_report,
        best_supported_model_id,
        best_log_predictive_density,
        mixture_log_predictive_density,
        minimum_acceptable_best_model_log_density,
        status: if decision_eligible {
            ModelSetStatus::WithinCalibratedEnvelope
        } else {
            ModelSetStatus::FlaggedInadequate
        },
        decision_eligible,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str, weight: f64, predictive_mean: f64, variance: f64) -> GaussianModel {
        GaussianModel {
            model_id: id.into(),
            prior_weight: weight,
            theta_mean: 0.0,
            theta_variance: 1.0,
            predictive_mean: vec![predictive_mean],
            innovation_covariance: vec![vec![variance]],
            theta_observation_covariance: vec![0.0],
        }
    }

    #[test]
    fn normalized_weights_do_not_hide_all_models_being_wrong() {
        let models = vec![
            model("narrow-a", 0.5, 0.0, 0.1),
            model("narrow-b", 0.5, 0.0, 0.2),
        ];
        let report = assess_model_set(
            &models,
            &[20.0],
            &["narrow-a".into(), "narrow-b".into()],
            -20.0,
        ).unwrap();

        let posterior_sum: f64 = report.mixture.components.iter()
            .map(|component| component.posterior_weight).sum();
        assert!((posterior_sum - 1.0).abs() < 1.0e-12);
        assert_eq!(report.status, ModelSetStatus::FlaggedInadequate);
        assert!(!report.decision_eligible);
        assert!(report.best_log_predictive_density < -20.0);
    }

    #[test]
    fn a_model_that_predicts_the_observation_can_pass_the_gate() {
        let models = vec![
            model("misspecified", 0.5, 0.0, 0.1),
            model("plausible", 0.5, 20.0, 1.0),
        ];
        let report = assess_model_set(
            &models,
            &[20.0],
            &["misspecified".into(), "plausible".into()],
            -20.0,
        ).unwrap();
        assert_eq!(report.best_supported_model_id, "plausible");
        assert_eq!(report.status, ModelSetStatus::WithinCalibratedEnvelope);
        assert!(report.decision_eligible);
    }

    #[test]
    fn zero_prior_weight_support_model_cannot_rescue_bayesian_model_set() {
        let models = vec![
            model("bad-but-supported", 1.0, 0.0, 0.1),
            model("zero-prior-fit", 0.0, 20.0, 1.0),
        ];
        let report = assess_model_set(
            &models,
            &[20.0],
            &["bad-but-supported".into(), "zero-prior-fit".into()],
            -20.0,
        ).unwrap();
        assert_eq!(report.mixture.components[1].posterior_weight, 0.0);
        assert_eq!(report.best_supported_model_id, "bad-but-supported");
        assert_eq!(report.status, ModelSetStatus::FlaggedInadequate);
        assert!(report.mixture.robust_support_ids.contains(&"zero-prior-fit".into()));
    }

    #[test]
    fn adequacy_threshold_is_external_and_must_be_finite() {
        let models = vec![model("one", 1.0, 0.0, 1.0)];
        assert_eq!(
            assess_model_set(&models, &[0.0], &["one".into()], f64::NAN),
            Err(Error::InvalidAdequacyThreshold)
        );
    }
}
