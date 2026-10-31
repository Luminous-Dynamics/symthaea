//! Exact small-model credit-path oracle for spatiotemporal learning comparisons.
//!
//! This scalar two-module recurrence provides an analytical BPTT reference,
//! centered finite-difference checks, temporal-detachment controls, and a
//! local auxiliary-head gradient. It is a dependency-graph unit test, not a
//! model benchmark and not an implementation of STDL or BrainTrace.

use std::fmt;

const MAX_STEPS: usize = 100_000;
const FINITE_DIFFERENCE_EPSILON: f64 = 1.0e-6;

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarRecurrentConfig {
    /// Hidden-layer recurrent coefficient.
    pub alpha: f64,
    /// Downstream-module recurrent coefficient.
    pub beta: f64,
    /// Input-to-hidden trainable weight.
    pub input_weight: f64,
    /// Hidden-to-downstream trainable weight.
    pub interlayer_weight: f64,
    /// Downstream readout weight.
    pub readout_weight: f64,
    /// Scalar target at the final time.
    pub target: f64,
    /// Causal input stream.
    pub inputs: Vec<f64>,
}

impl ScalarRecurrentConfig {
    pub fn reference() -> Self {
        Self {
            alpha: 0.7,
            beta: 0.8,
            input_weight: 0.4,
            interlayer_weight: 0.6,
            readout_weight: 1.2,
            target: 0.25,
            inputs: vec![0.2, -0.1, 0.4, 0.3, -0.2, 0.15],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarForwardTrace {
    pub hidden: Vec<f64>,
    pub downstream: Vec<f64>,
    pub prediction: f64,
    pub loss: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarGradient {
    pub input_weight: f64,
    pub interlayer_weight: f64,
    pub readout_weight: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl ScalarGradient {
    pub fn as_array(self) -> [f64; 5] {
        [
            self.input_weight,
            self.interlayer_weight,
            self.readout_weight,
            self.alpha,
            self.beta,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetachmentComparison {
    pub full_bptt_input_weight_gradient: f64,
    pub detached_hidden_input_weight_gradient: f64,
    pub detached_hidden_and_downstream_input_weight_gradient: f64,
    pub local_auxiliary_input_weight_gradient: f64,
    /// Fraction of the full input-weight gradient retained by hidden detachment.
    pub hidden_detach_gradient_ratio: Option<f64>,
    /// Fraction retained when both recurrent paths are detached.
    pub double_detach_gradient_ratio: Option<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScalarOracleError {
    EmptyInput,
    TooManySteps { requested: usize, maximum: usize },
    NonFiniteParameter { name: &'static str },
    NonFiniteInput { index: usize },
    InvalidAuxiliaryReadout,
}

impl fmt::Display for ScalarOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "scalar recurrent input stream must not be empty"),
            Self::TooManySteps { requested, maximum } => {
                write!(f, "scalar recurrent stream length {requested} exceeds {maximum}")
            }
            Self::NonFiniteParameter { name } => {
                write!(f, "scalar recurrent parameter {name} must be finite")
            }
            Self::NonFiniteInput { index } => {
                write!(f, "scalar recurrent input at index {index} must be finite")
            }
            Self::InvalidAuxiliaryReadout => {
                write!(f, "auxiliary readout weight must be finite")
            }
        }
    }
}

impl std::error::Error for ScalarOracleError {}

pub fn validate(config: &ScalarRecurrentConfig) -> Result<(), ScalarOracleError> {
    if config.inputs.is_empty() {
        return Err(ScalarOracleError::EmptyInput);
    }
    if config.inputs.len() > MAX_STEPS {
        return Err(ScalarOracleError::TooManySteps {
            requested: config.inputs.len(),
            maximum: MAX_STEPS,
        });
    }
    for (name, value) in [
        ("alpha", config.alpha),
        ("beta", config.beta),
        ("input_weight", config.input_weight),
        ("interlayer_weight", config.interlayer_weight),
        ("readout_weight", config.readout_weight),
        ("target", config.target),
    ] {
        if !value.is_finite() {
            return Err(ScalarOracleError::NonFiniteParameter { name });
        }
    }
    for (index, value) in config.inputs.iter().enumerate() {
        if !value.is_finite() {
            return Err(ScalarOracleError::NonFiniteInput { index });
        }
    }
    Ok(())
}

/// Run the full forward recurrence:
/// h_t = alpha*h_(t-1) + w*x_t
/// g_t = beta*g_(t-1) + v*h_t
/// y = c*g_T; L = 0.5*(y-target)^2.
pub fn forward(
    config: &ScalarRecurrentConfig,
) -> Result<ScalarForwardTrace, ScalarOracleError> {
    validate(config)?;
    let mut hidden = Vec::with_capacity(config.inputs.len());
    let mut downstream = Vec::with_capacity(config.inputs.len());
    let mut previous_hidden = 0.0;
    let mut previous_downstream = 0.0;

    for input in &config.inputs {
        let current_hidden =
            config.alpha * previous_hidden + config.input_weight * input;
        let current_downstream =
            config.beta * previous_downstream + config.interlayer_weight * current_hidden;
        hidden.push(current_hidden);
        downstream.push(current_downstream);
        previous_hidden = current_hidden;
        previous_downstream = current_downstream;
    }

    let prediction = config.readout_weight * previous_downstream;
    let residual = prediction - config.target;
    Ok(ScalarForwardTrace {
        hidden,
        downstream,
        prediction,
        loss: 0.5 * residual * residual,
    })
}

/// Reverse-mode BPTT gradient for all five scalar parameters.
pub fn full_bptt_gradient(
    config: &ScalarRecurrentConfig,
) -> Result<ScalarGradient, ScalarOracleError> {
    let trace = forward(config)?;
    let final_index = config.inputs.len() - 1;
    let residual = trace.prediction - config.target;

    let mut gradient = ScalarGradient {
        input_weight: 0.0,
        interlayer_weight: 0.0,
        readout_weight: residual * trace.downstream[final_index],
        alpha: 0.0,
        beta: 0.0,
    };

    let mut adjoint_downstream = residual * config.readout_weight;
    let mut adjoint_hidden = 0.0;

    for index in (0..config.inputs.len()).rev() {
        let previous_hidden = if index == 0 { 0.0 } else { trace.hidden[index - 1] };
        let previous_downstream =
            if index == 0 { 0.0 } else { trace.downstream[index - 1] };

        gradient.interlayer_weight += adjoint_downstream * trace.hidden[index];
        gradient.beta += adjoint_downstream * previous_downstream;

        let current_hidden_adjoint =
            adjoint_downstream * config.interlayer_weight + adjoint_hidden;
        gradient.input_weight += current_hidden_adjoint * config.inputs[index];
        gradient.alpha += current_hidden_adjoint * previous_hidden;

        adjoint_hidden = current_hidden_adjoint * config.alpha;
        adjoint_downstream *= config.beta;
    }

    Ok(gradient)
}

/// Compute dL/d(input_weight) after specified temporal stop-gradient boundaries.
///
/// Hidden detachment prevents credit from propagating from h_t to h_(t-1).
/// Downstream detachment prevents credit from propagating from g_t to g_(t-1).
/// Forward values are unchanged; only the gradient dependency graph changes.
pub fn input_weight_gradient_with_detachment(
    config: &ScalarRecurrentConfig,
    detach_hidden_temporal: bool,
    detach_downstream_temporal: bool,
) -> Result<f64, ScalarOracleError> {
    let trace = forward(config)?;
    let residual = trace.prediction - config.target;
    let mut hidden_sensitivity = 0.0;
    let mut downstream_sensitivity = 0.0;

    for input in &config.inputs {
        hidden_sensitivity = if detach_hidden_temporal {
            *input
        } else {
            config.alpha * hidden_sensitivity + input
        };
        downstream_sensitivity = if detach_downstream_temporal {
            config.interlayer_weight * hidden_sensitivity
        } else {
            config.beta * downstream_sensitivity
                + config.interlayer_weight * hidden_sensitivity
        };
    }

    Ok(residual * config.readout_weight * downstream_sensitivity)
}

/// Local auxiliary objective applied at the hidden boundary at the final step.
///
/// The auxiliary head has its own readout. With hidden temporal detachment,
/// only the current final-step input contributes to d h_T / d w. This gradient
/// is intentionally not claimed to equal the global BPTT gradient.
pub fn local_auxiliary_input_weight_gradient(
    config: &ScalarRecurrentConfig,
    auxiliary_readout_weight: f64,
) -> Result<f64, ScalarOracleError> {
    validate(config)?;
    if !auxiliary_readout_weight.is_finite() {
        return Err(ScalarOracleError::InvalidAuxiliaryReadout);
    }
    let trace = forward(config)?;
    let final_hidden = *trace.hidden.last().ok_or(ScalarOracleError::EmptyInput)?;
    let auxiliary_prediction = auxiliary_readout_weight * final_hidden;
    let auxiliary_residual = auxiliary_prediction - config.target;
    let final_input = *config.inputs.last().ok_or(ScalarOracleError::EmptyInput)?;
    Ok(auxiliary_residual * auxiliary_readout_weight * final_input)
}

pub fn compare_detachment_and_auxiliary(
    config: &ScalarRecurrentConfig,
    auxiliary_readout_weight: f64,
) -> Result<DetachmentComparison, ScalarOracleError> {
    let full = full_bptt_gradient(config)?.input_weight;
    let hidden_detached =
        input_weight_gradient_with_detachment(config, true, false)?;
    let both_detached =
        input_weight_gradient_with_detachment(config, true, true)?;
    let local_auxiliary =
        local_auxiliary_input_weight_gradient(config, auxiliary_readout_weight)?;

    let ratio = |value: f64| {
        (full.abs() > f64::EPSILON).then_some(value / full)
    };

    Ok(DetachmentComparison {
        full_bptt_input_weight_gradient: full,
        detached_hidden_input_weight_gradient: hidden_detached,
        detached_hidden_and_downstream_input_weight_gradient: both_detached,
        local_auxiliary_input_weight_gradient: local_auxiliary,
        hidden_detach_gradient_ratio: ratio(hidden_detached),
        double_detach_gradient_ratio: ratio(both_detached),
    })
}

/// Centered finite-difference estimate of the full loss gradient.
pub fn centered_finite_difference(
    config: &ScalarRecurrentConfig,
) -> Result<ScalarGradient, ScalarOracleError> {
    validate(config)?;
    let differentiate = |parameter: usize| {
        let mut plus = config.clone();
        let mut minus = config.clone();
        match parameter {
            0 => {
                plus.input_weight += FINITE_DIFFERENCE_EPSILON;
                minus.input_weight -= FINITE_DIFFERENCE_EPSILON;
            }
            1 => {
                plus.interlayer_weight += FINITE_DIFFERENCE_EPSILON;
                minus.interlayer_weight -= FINITE_DIFFERENCE_EPSILON;
            }
            2 => {
                plus.readout_weight += FINITE_DIFFERENCE_EPSILON;
                minus.readout_weight -= FINITE_DIFFERENCE_EPSILON;
            }
            3 => {
                plus.alpha += FINITE_DIFFERENCE_EPSILON;
                minus.alpha -= FINITE_DIFFERENCE_EPSILON;
            }
            4 => {
                plus.beta += FINITE_DIFFERENCE_EPSILON;
                minus.beta -= FINITE_DIFFERENCE_EPSILON;
            }
            _ => unreachable!("parameter index is restricted to 0..5"),
        }
        let plus_loss = forward(&plus)?.loss;
        let minus_loss = forward(&minus)?.loss;
        Ok((plus_loss - minus_loss) / (2.0 * FINITE_DIFFERENCE_EPSILON))
    };

    Ok(ScalarGradient {
        input_weight: differentiate(0)?,
        interlayer_weight: differentiate(1)?,
        readout_weight: differentiate(2)?,
        alpha: differentiate(3)?,
        beta: differentiate(4)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.12e}, expected={expected:.12e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn reverse_mode_bptt_matches_independent_known_answer_vector() {
        let config = ScalarRecurrentConfig::reference();
        let gradient = full_bptt_gradient(&config).unwrap();
        close(gradient.input_weight, 0.07154630345318397, 1.0e-12);
        close(gradient.interlayer_weight, 0.04769753563545599, 1.0e-12);
        close(gradient.readout_weight, 0.02384876781772800, 1.0e-12);
        close(gradient.alpha, 0.05479641412485119, 1.0e-12);
        close(gradient.beta, 0.06125805832273921, 1.0e-12);
    }

    #[test]
    fn reverse_mode_bptt_matches_centered_finite_difference() {
        let config = ScalarRecurrentConfig::reference();
        let reverse = full_bptt_gradient(&config).unwrap().as_array();
        let finite = centered_finite_difference(&config).unwrap().as_array();
        for (actual, expected) in reverse.into_iter().zip(finite) {
            close(actual, expected, 1.0e-7);
        }
    }

    #[test]
    fn no_detachment_matches_full_bptt_input_weight_gradient() {
        let config = ScalarRecurrentConfig::reference();
        let exact = full_bptt_gradient(&config).unwrap().input_weight;
        let no_detach =
            input_weight_gradient_with_detachment(&config, false, false).unwrap();
        close(no_detach, exact, 1.0e-12);
    }

    #[test]
    fn hidden_detachment_changes_long_horizon_credit() {
        let config = ScalarRecurrentConfig::reference();
        let exact = full_bptt_gradient(&config).unwrap().input_weight;
        let detached =
            input_weight_gradient_with_detachment(&config, true, false).unwrap();
        assert!((exact - detached).abs() > 1.0e-8);
    }

    #[test]
    fn detaching_both_recurrent_modules_keeps_only_last_step_input_credit() {
        let config = ScalarRecurrentConfig::reference();
        let trace = forward(&config).unwrap();
        let residual = trace.prediction - config.target;
        let expected = residual
            * config.readout_weight
            * config.interlayer_weight
            * *config.inputs.last().unwrap();
        let actual =
            input_weight_gradient_with_detachment(&config, true, true).unwrap();
        close(actual, expected, 1.0e-12);
    }

    #[test]
    fn local_auxiliary_gradient_is_a_distinct_objective_from_global_bptt() {
        let config = ScalarRecurrentConfig::reference();
        let global = full_bptt_gradient(&config).unwrap().input_weight;
        let local = local_auxiliary_input_weight_gradient(&config, 0.9).unwrap();
        assert!((global - local).abs() > 1.0e-8);
    }

    #[test]
    fn detachment_comparison_reports_gradient_coverage_ratios() {
        let config = ScalarRecurrentConfig::reference();
        let comparison = compare_detachment_and_auxiliary(&config, 0.9).unwrap();
        assert!(comparison.hidden_detach_gradient_ratio.is_some());
        assert!(comparison.double_detach_gradient_ratio.is_some());
    }

    #[test]
    fn non_finite_input_is_rejected() {
        let mut config = ScalarRecurrentConfig::reference();
        config.inputs[1] = f64::NAN;
        assert!(matches!(
            forward(&config),
            Err(ScalarOracleError::NonFiniteInput { index: 1 })
        ));
    }

    #[test]
    fn empty_input_is_rejected() {
        let mut config = ScalarRecurrentConfig::reference();
        config.inputs.clear();
        assert_eq!(
            forward(&config),
            Err(ScalarOracleError::EmptyInput)
        );
    }
}
