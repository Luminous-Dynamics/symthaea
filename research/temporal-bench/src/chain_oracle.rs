//! 2-4 module recurrent chain oracle for spatial and temporal credit cuts.
//!
//! The forward model is a scalar recurrent chain. The oracle computes full
//! reverse-mode BPTT, centered finite differences, temporal stop-gradient
//! masks and an STDL-style spatial cut with a separate local auxiliary
//! objective. It does not implement STDL or a production SNN.

use std::fmt;

const MIN_MODULES: usize = 2;
const MAX_MODULES: usize = 4;
const MAX_STEPS: usize = 100_000;
const FD_EPSILON: f64 = 1.0e-6;

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarChainConfig {
    pub recurrent_weights: Vec<f64>,
    pub feedforward_weights: Vec<f64>,
    pub readout_weight: f64,
    pub target: f64,
    pub inputs: Vec<f64>,
}

impl ScalarChainConfig {
    pub fn two_module_reference() -> Self {
        Self {
            recurrent_weights: vec![0.7, 0.8],
            feedforward_weights: vec![0.4, 0.6],
            readout_weight: 1.2,
            target: 0.25,
            inputs: vec![0.2, -0.1, 0.4, 0.3, -0.2, 0.15],
        }
    }

    pub fn four_module_reference() -> Self {
        Self {
            recurrent_weights: vec![0.65, 0.7, 0.75, 0.8],
            feedforward_weights: vec![0.35, 0.45, 0.55, 0.65],
            readout_weight: 0.9,
            target: -0.1,
            inputs: vec![0.3, -0.2, 0.1, 0.4, -0.15, 0.25, 0.05],
        }
    }

    pub fn module_count(&self) -> usize {
        self.recurrent_weights.len()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarChainTrace {
    pub states: Vec<Vec<f64>>,
    pub prediction: f64,
    pub loss: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarChainGradient {
    pub recurrent_weights: Vec<f64>,
    pub feedforward_weights: Vec<f64>,
    pub readout_weight: f64,
    pub auxiliary_readout_weight: Option<f64>,
}

impl ScalarChainGradient {
    pub fn flatten_main(&self) -> Vec<f64> {
        let mut values = Vec::with_capacity(
            self.recurrent_weights.len() + self.feedforward_weights.len() + 1,
        );
        values.push(self.readout_weight);
        values.extend(self.recurrent_weights.iter().copied());
        values.extend(self.feedforward_weights.iter().copied());
        values
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpatialCutResult {
    pub gradient: ScalarChainGradient,
    pub global_loss: f64,
    pub auxiliary_loss: f64,
    pub boundary_module: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChainOracleError {
    InvalidModuleCount { requested: usize, minimum: usize, maximum: usize },
    ParameterLengthMismatch,
    EmptyInput,
    TooManySteps { requested: usize, maximum: usize },
    NonFiniteParameter { name: &'static str, index: Option<usize> },
    NonFiniteInput { index: usize },
    InvalidSpatialBoundary { requested: usize, maximum: usize },
    InvalidAuxiliaryScale,
}

impl fmt::Display for ChainOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModuleCount { requested, minimum, maximum } => write!(
                f,
                "module count {requested} is outside {minimum}..={maximum}"
            ),
            Self::ParameterLengthMismatch => {
                write!(f, "recurrent and feedforward weight arrays must have equal length")
            }
            Self::EmptyInput => write!(f, "chain input stream must not be empty"),
            Self::TooManySteps { requested, maximum } => {
                write!(f, "chain stream length {requested} exceeds {maximum}")
            }
            Self::NonFiniteParameter { name, index } => {
                write!(f, "chain parameter {name} at {index:?} must be finite")
            }
            Self::NonFiniteInput { index } => {
                write!(f, "chain input at index {index} must be finite")
            }
            Self::InvalidSpatialBoundary { requested, maximum } => write!(
                f,
                "spatial cut boundary {requested} must be less than last module {maximum}"
            ),
            Self::InvalidAuxiliaryScale => {
                write!(f, "auxiliary loss scale and readout weight must be finite")
            }
        }
    }
}

impl std::error::Error for ChainOracleError {}

pub fn validate(config: &ScalarChainConfig) -> Result<(), ChainOracleError> {
    let modules = config.recurrent_weights.len();
    if !(MIN_MODULES..=MAX_MODULES).contains(&modules) {
        return Err(ChainOracleError::InvalidModuleCount {
            requested: modules,
            minimum: MIN_MODULES,
            maximum: MAX_MODULES,
        });
    }
    if config.feedforward_weights.len() != modules {
        return Err(ChainOracleError::ParameterLengthMismatch);
    }
    if config.inputs.is_empty() {
        return Err(ChainOracleError::EmptyInput);
    }
    if config.inputs.len() > MAX_STEPS {
        return Err(ChainOracleError::TooManySteps {
            requested: config.inputs.len(),
            maximum: MAX_STEPS,
        });
    }
    if !config.readout_weight.is_finite() {
        return Err(ChainOracleError::NonFiniteParameter {
            name: "readout_weight",
            index: None,
        });
    }
    if !config.target.is_finite() {
        return Err(ChainOracleError::NonFiniteParameter {
            name: "target",
            index: None,
        });
    }
    for (index, value) in config.recurrent_weights.iter().enumerate() {
        if !value.is_finite() {
            return Err(ChainOracleError::NonFiniteParameter {
                name: "recurrent_weight",
                index: Some(index),
            });
        }
    }
    for (index, value) in config.feedforward_weights.iter().enumerate() {
        if !value.is_finite() {
            return Err(ChainOracleError::NonFiniteParameter {
                name: "feedforward_weight",
                index: Some(index),
            });
        }
    }
    for (index, value) in config.inputs.iter().enumerate() {
        if !value.is_finite() {
            return Err(ChainOracleError::NonFiniteInput { index });
        }
    }
    Ok(())
}

/// Forward recurrence:
/// state[t][module] = recurrent[module] * state[t-1][module]
///                   + feedforward[module] * previous spatial activation.
pub fn forward(
    config: &ScalarChainConfig,
) -> Result<ScalarChainTrace, ChainOracleError> {
    validate(config)?;
    let modules = config.module_count();
    let mut states: Vec<Vec<f64>> = Vec::with_capacity(config.inputs.len());

    for (time, input) in config.inputs.iter().enumerate() {
        let mut current = vec![0.0; modules];
        for module in 0..modules {
            let previous_temporal = if time == 0 {
                0.0
            } else {
                states[time - 1][module]
            };
            let previous_spatial = if module == 0 {
                *input
            } else {
                current[module - 1]
            };
            current[module] = config.recurrent_weights[module] * previous_temporal
                + config.feedforward_weights[module] * previous_spatial;
        }
        states.push(current);
    }

    let final_state = states[config.inputs.len() - 1][modules - 1];
    let prediction = config.readout_weight * final_state;
    let residual = prediction - config.target;
    Ok(ScalarChainTrace {
        states,
        prediction,
        loss: 0.5 * residual * residual,
    })
}

/// Full BPTT reverse-mode gradient over every recurrent and spatial weight.
pub fn full_bptt_gradient(
    config: &ScalarChainConfig,
) -> Result<ScalarChainGradient, ChainOracleError> {
    let trace = forward(config)?;
    let time_steps = config.inputs.len();
    let modules = config.module_count();
    let mut adjoints = vec![vec![0.0; modules]; time_steps];
    let residual = trace.prediction - config.target;
    adjoints[time_steps - 1][modules - 1] = residual * config.readout_weight;

    let mut gradient = ScalarChainGradient {
        recurrent_weights: vec![0.0; modules],
        feedforward_weights: vec![0.0; modules],
        readout_weight: residual * trace.states[time_steps - 1][modules - 1],
        auxiliary_readout_weight: None,
    };

    for time in (0..time_steps).rev() {
        for module in (0..modules).rev() {
            let adjoint = adjoints[time][module];
            let previous_temporal = if time == 0 {
                0.0
            } else {
                trace.states[time - 1][module]
            };
            let previous_spatial = if module == 0 {
                config.inputs[time]
            } else {
                trace.states[time][module - 1]
            };

            gradient.recurrent_weights[module] += adjoint * previous_temporal;
            gradient.feedforward_weights[module] += adjoint * previous_spatial;

            if time > 0 {
                adjoints[time - 1][module] +=
                    adjoint * config.recurrent_weights[module];
            }
            if module > 0 {
                adjoints[time][module - 1] +=
                    adjoint * config.feedforward_weights[module];
            }
        }
    }
    Ok(gradient)
}

/// BPTT with temporal stop-gradient applied selectively to module states.
///
/// A true entry in detach_temporal[module] prevents loss credit from flowing
/// from time t into that module's state at time t-1. Forward states are
/// unchanged. Spatial cross-module credit is otherwise preserved.
pub fn gradient_with_temporal_detachment(
    config: &ScalarChainConfig,
    detach_temporal: &[bool],
) -> Result<ScalarChainGradient, ChainOracleError> {
    validate(config)?;
    let modules = config.module_count();
    if detach_temporal.len() != modules {
        return Err(ChainOracleError::ParameterLengthMismatch);
    }
    let trace = forward(config)?;
    let time_steps = config.inputs.len();
    let mut adjoints = vec![vec![0.0; modules]; time_steps];
    let residual = trace.prediction - config.target;
    adjoints[time_steps - 1][modules - 1] = residual * config.readout_weight;

    let mut gradient = ScalarChainGradient {
        recurrent_weights: vec![0.0; modules],
        feedforward_weights: vec![0.0; modules],
        readout_weight: residual * trace.states[time_steps - 1][modules - 1],
        auxiliary_readout_weight: None,
    };

    for time in (0..time_steps).rev() {
        for module in (0..modules).rev() {
            let adjoint = adjoints[time][module];
            let previous_temporal = if time == 0 {
                0.0
            } else {
                trace.states[time - 1][module]
            };
            let previous_spatial = if module == 0 {
                config.inputs[time]
            } else {
                trace.states[time][module - 1]
            };
            gradient.recurrent_weights[module] += adjoint * previous_temporal;
            gradient.feedforward_weights[module] += adjoint * previous_spatial;

            if time > 0 && !detach_temporal[module] {
                adjoints[time - 1][module] +=
                    adjoint * config.recurrent_weights[module];
            }
            if module > 0 {
                adjoints[time][module - 1] +=
                    adjoint * config.feedforward_weights[module];
            }
        }
    }
    Ok(gradient)
}

/// STDL-style spatial cut with a separate local auxiliary objective.
///
/// The global loss gradient stops at the selected boundary when entering the
/// upstream module. A local auxiliary head is attached at that boundary and
/// trains the upstream subnetwork on its own objective. auxiliary_loss_scale
/// weights that objective relative to the global downstream loss.
pub fn spatial_cut_with_auxiliary(
    config: &ScalarChainConfig,
    boundary_module: usize,
    auxiliary_readout_weight: f64,
    auxiliary_loss_scale: f64,
) -> Result<SpatialCutResult, ChainOracleError> {
    validate(config)?;
    let modules = config.module_count();
    if boundary_module >= modules - 1 {
        return Err(ChainOracleError::InvalidSpatialBoundary {
            requested: boundary_module,
            maximum: modules - 2,
        });
    }
    if !auxiliary_readout_weight.is_finite()
        || !auxiliary_loss_scale.is_finite()
        || auxiliary_loss_scale < 0.0
    {
        return Err(ChainOracleError::InvalidAuxiliaryScale);
    }

    let trace = forward(config)?;
    let time_steps = config.inputs.len();
    let mut global_adjoints = vec![vec![0.0; modules]; time_steps];
    let mut local_adjoints = vec![vec![0.0; modules]; time_steps];
    let residual = trace.prediction - config.target;
    global_adjoints[time_steps - 1][modules - 1] = residual * config.readout_weight;

    let local_state = trace.states[time_steps - 1][boundary_module];
    let auxiliary_prediction = auxiliary_readout_weight * local_state;
    let auxiliary_residual = auxiliary_prediction - config.target;
    local_adjoints[time_steps - 1][boundary_module] =
        auxiliary_loss_scale * auxiliary_residual * auxiliary_readout_weight;
    let auxiliary_loss = 0.5 * auxiliary_residual * auxiliary_residual;

    let mut gradient = ScalarChainGradient {
        recurrent_weights: vec![0.0; modules],
        feedforward_weights: vec![0.0; modules],
        readout_weight: residual * trace.states[time_steps - 1][modules - 1],
        auxiliary_readout_weight: Some(
            auxiliary_loss_scale * auxiliary_residual * local_state,
        ),
    };

    for time in (0..time_steps).rev() {
        for module in (0..modules).rev() {
            let global_adjoint = global_adjoints[time][module];
            let local_adjoint = local_adjoints[time][module];
            let total_adjoint = global_adjoint + local_adjoint;
            let previous_temporal = if time == 0 {
                0.0
            } else {
                trace.states[time - 1][module]
            };
            let previous_spatial = if module == 0 {
                config.inputs[time]
            } else {
                trace.states[time][module - 1]
            };

            gradient.recurrent_weights[module] += total_adjoint * previous_temporal;
            gradient.feedforward_weights[module] += total_adjoint * previous_spatial;

            if time > 0 {
                if global_adjoint != 0.0 {
                    global_adjoints[time - 1][module] +=
                        global_adjoint * config.recurrent_weights[module];
                }
                if local_adjoint != 0.0 {
                    local_adjoints[time - 1][module] +=
                        local_adjoint * config.recurrent_weights[module];
                }
            }
            if module > 0 {
                if module != boundary_module + 1 {
                    global_adjoints[time][module - 1] +=
                        global_adjoint * config.feedforward_weights[module];
                }
                local_adjoints[time][module - 1] +=
                    local_adjoint * config.feedforward_weights[module];
            }
        }
    }

    Ok(SpatialCutResult {
        gradient,
        global_loss: trace.loss,
        auxiliary_loss,
        boundary_module,
    })
}

/// Centered finite-difference BPTT reference for every chain parameter.
pub fn centered_finite_difference(
    config: &ScalarChainConfig,
) -> Result<ScalarChainGradient, ChainOracleError> {
    validate(config)?;
    let modules = config.module_count();
    let parameter_count = 1 + 2 * modules;
    let mut finite = Vec::with_capacity(parameter_count);

    for parameter_index in 0..parameter_count {
        let mut plus = config.clone();
        let mut minus = config.clone();
        perturb(&mut plus, parameter_index, FD_EPSILON);
        perturb(&mut minus, parameter_index, -FD_EPSILON);
        finite.push(
            (forward(&plus)?.loss - forward(&minus)?.loss) / (2.0 * FD_EPSILON),
        );
    }

    Ok(ScalarChainGradient {
        readout_weight: finite[0],
        recurrent_weights: finite[1..1 + modules].to_vec(),
        feedforward_weights: finite[1 + modules..].to_vec(),
        auxiliary_readout_weight: None,
    })
}

fn perturb(config: &mut ScalarChainConfig, index: usize, delta: f64) {
    let modules = config.module_count();
    if index == 0 {
        config.readout_weight += delta;
    } else if index <= modules {
        config.recurrent_weights[index - 1] += delta;
    } else {
        let feedforward_index = index - 1 - modules;
        config.feedforward_weights[feedforward_index] += delta;
    }
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

    fn check_gradients(config: &ScalarChainConfig) {
        let exact = full_bptt_gradient(config).unwrap();
        let finite = centered_finite_difference(config).unwrap();
        for (actual, expected) in exact.flatten_main().into_iter().zip(finite.flatten_main()) {
            close(actual, expected, 2.0e-7);
        }
    }

    #[test]
    fn two_module_chain_bptt_matches_finite_difference() {
        check_gradients(&ScalarChainConfig::two_module_reference());
    }

    #[test]
    fn four_module_chain_bptt_matches_finite_difference() {
        check_gradients(&ScalarChainConfig::four_module_reference());
    }

    #[test]
    fn all_temporal_modules_attached_matches_full_bptt() {
        let config = ScalarChainConfig::four_module_reference();
        let exact = full_bptt_gradient(&config).unwrap().flatten_main();
        let attached =
            gradient_with_temporal_detachment(&config, &[false, false, false, false])
                .unwrap()
                .flatten_main();
        for (actual, expected) in attached.into_iter().zip(exact) {
            close(actual, expected, 1.0e-12);
        }
    }

    #[test]
    fn detaching_all_temporal_modules_changes_credit_coverage() {
        let config = ScalarChainConfig::four_module_reference();
        let exact = full_bptt_gradient(&config).unwrap().flatten_main();
        let detached =
            gradient_with_temporal_detachment(&config, &[true, true, true, true])
                .unwrap()
                .flatten_main();
        assert!(exact
            .iter()
            .zip(detached)
            .any(|(full, cut)| (full - cut).abs() > 1.0e-8));
    }

    #[test]
    fn spatial_cut_blocks_global_gradient_from_downstream_into_early_modules() {
        let config = ScalarChainConfig::four_module_reference();
        let cut = spatial_cut_with_auxiliary(&config, 1, 0.9, 0.0).unwrap();
        close(cut.gradient.feedforward_weights[0], 0.0, 1.0e-12);
        close(cut.gradient.recurrent_weights[0], 0.0, 1.0e-12);
        assert!(cut.gradient.feedforward_weights[2].abs() > 1.0e-8);
    }

    #[test]
    fn local_auxiliary_loss_restores_an_upstream_training_signal() {
        let config = ScalarChainConfig::four_module_reference();
        let cut_only = spatial_cut_with_auxiliary(&config, 1, 0.9, 0.0).unwrap();
        let cut_with_aux = spatial_cut_with_auxiliary(&config, 1, 0.9, 1.0).unwrap();
        close(cut_only.gradient.feedforward_weights[0], 0.0, 1.0e-12);
        assert!(cut_with_aux.gradient.feedforward_weights[0].abs() > 1.0e-8);
    }

    #[test]
    fn local_auxiliary_gradient_is_reported_separately() {
        let config = ScalarChainConfig::four_module_reference();
        let cut = spatial_cut_with_auxiliary(&config, 1, 0.9, 1.0).unwrap();
        assert!(cut.gradient.auxiliary_readout_weight.unwrap().abs() > 1.0e-8);
        assert!(cut.auxiliary_loss > 0.0);
    }

    #[test]
    fn invalid_spatial_boundary_is_rejected() {
        let config = ScalarChainConfig::four_module_reference();
        assert!(matches!(
            spatial_cut_with_auxiliary(&config, 3, 0.9, 1.0),
            Err(ChainOracleError::InvalidSpatialBoundary { .. })
        ));
    }

    #[test]
    fn temporal_detachment_mask_must_match_module_count() {
        let config = ScalarChainConfig::four_module_reference();
        assert!(matches!(
            gradient_with_temporal_detachment(&config, &[true, false]),
            Err(ChainOracleError::ParameterLengthMismatch)
        ));
    }
}
