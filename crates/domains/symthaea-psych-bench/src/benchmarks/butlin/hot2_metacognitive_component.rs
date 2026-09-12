//! HOT-2 component-level test of Symthaea's production metacognitive self-model.
//!
//! This isolates one prerequisite for HOT-2: can the real `MetaCognitiveLayer`
//! prospectively estimate its own prediction-error tendency and adapt that
//! estimate when the error regime changes?
//!
//! The experiment deliberately does **not** inject confidence into the full
//! cognitive loop and cannot promote a Butlin support tier. The system-level
//! corruption/intervention theorem remains separate.

use symthaea::wisdom::MetaCognitiveLayer;

const ACQUISITION_TRIALS: usize = 32;
const REVERSAL_TRIALS: usize = 32;
const EARLY_WINDOW: usize = 8;
const LATE_WINDOW: usize = 8;
const LOW_ERROR: f32 = 0.20;
const HIGH_ERROR: f32 = 0.80;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetacognitiveTrial {
    /// Prospective estimate made before the actual error is revealed.
    pub predicted_error: f32,
    /// Independently supplied actual prediction error.
    pub actual_error: f32,
}

impl MetacognitiveTrial {
    pub fn absolute_meta_error(self) -> f32 {
        (self.predicted_error - self.actual_error).abs()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hot2MetacognitiveComponentRun {
    pub acquisition: Vec<MetacognitiveTrial>,
    pub reversal: Vec<MetacognitiveTrial>,
    pub acquisition_mean_meta_error: f32,
    pub reversal_early_mean_meta_error: f32,
    pub reversal_late_mean_meta_error: f32,
    pub reversal_adaptation_gain: f32,
    /// Error of a negative-control predictor frozen at the acquisition regime.
    pub frozen_control_reversal_error: f32,
    /// Final production self-model accuracy reported by `MetaCognitiveLayer`.
    pub final_internal_accuracy: f32,
}

impl Hot2MetacognitiveComponentRun {
    pub fn claim_scope_note(&self) -> &'static str {
        "Component-level metacognitive prerequisite only: the production MetaCognitiveLayer \
         prospectively estimates its own error tendency and is tested for adaptation after a \
         controlled error-regime shift. This does not test confidence corruption inside the full \
         CognitiveLoopService, downstream behavioral use, or consciousness, and cannot promote a \
         HOT-2 Butlin support tier."
    }
}

pub fn run_component_experiment() -> Hot2MetacognitiveComponentRun {
    let mut meta = MetaCognitiveLayer::new();

    let acquisition = run_regime(&mut meta, ACQUISITION_TRIALS, LOW_ERROR);
    let reversal = run_regime(&mut meta, REVERSAL_TRIALS, HIGH_ERROR);

    let acquisition_mean_meta_error = mean_meta_error(&acquisition);
    let reversal_early_mean_meta_error = mean_meta_error(&reversal[..EARLY_WINDOW]);
    let reversal_late_mean_meta_error =
        mean_meta_error(&reversal[reversal.len() - LATE_WINDOW..]);
    let reversal_adaptation_gain =
        reversal_early_mean_meta_error - reversal_late_mean_meta_error;
    let frozen_control_reversal_error = (LOW_ERROR - HIGH_ERROR).abs();

    Hot2MetacognitiveComponentRun {
        acquisition,
        reversal,
        acquisition_mean_meta_error,
        reversal_early_mean_meta_error,
        reversal_late_mean_meta_error,
        reversal_adaptation_gain,
        frozen_control_reversal_error,
        final_internal_accuracy: meta.accuracy(),
    }
}

fn run_regime(
    meta: &mut MetaCognitiveLayer,
    trials: usize,
    actual_error: f32,
) -> Vec<MetacognitiveTrial> {
    let mut out = Vec::with_capacity(trials);
    for _ in 0..trials {
        // Critically, this is read BEFORE actual_error is supplied to
        // update_self_model below. It is therefore a prospective estimate,
        // not a post-hoc reconstruction.
        let predicted_error = meta.predict_own_error(actual_error);
        out.push(MetacognitiveTrial {
            predicted_error,
            actual_error,
        });
        meta.update_self_model(actual_error);
    }
    out
}

fn mean_meta_error(trials: &[MetacognitiveTrial]) -> f32 {
    if trials.is_empty() {
        return 0.0;
    }
    trials
        .iter()
        .copied()
        .map(MetacognitiveTrial::absolute_meta_error)
        .sum::<f32>()
        / trials.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_self_model_is_well_calibrated_in_stable_regime() {
        let run = run_component_experiment();
        assert!(
            run.acquisition_mean_meta_error < 0.02,
            "stable-regime meta error too high: {}",
            run.acquisition_mean_meta_error
        );
    }

    #[test]
    fn production_self_model_detects_and_adapts_to_regime_shift() {
        let run = run_component_experiment();
        assert!(run.reversal_early_mean_meta_error > 0.4);
        assert!(
            run.reversal_late_mean_meta_error < run.reversal_early_mean_meta_error,
            "late error {} did not improve from early error {}",
            run.reversal_late_mean_meta_error,
            run.reversal_early_mean_meta_error
        );
        assert!(run.reversal_adaptation_gain > 0.15);
    }

    #[test]
    fn adaptive_self_model_beats_frozen_acquisition_prediction_late_in_reversal() {
        let run = run_component_experiment();
        assert!(
            run.reversal_late_mean_meta_error < run.frozen_control_reversal_error,
            "adaptive={} frozen={}",
            run.reversal_late_mean_meta_error,
            run.frozen_control_reversal_error
        );
    }

    #[test]
    fn result_refuses_hot2_promotion() {
        let run = run_component_experiment();
        assert!(run.claim_scope_note().contains("cannot promote"));
        assert!(run.claim_scope_note().contains("CognitiveLoopService"));
    }
}
