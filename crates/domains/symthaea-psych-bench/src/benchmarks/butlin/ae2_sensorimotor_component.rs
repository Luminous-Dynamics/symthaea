//! AE-2 component-level contingency reversal using Symthaea's production
//! sensorimotor contingency engine.
//!
//! This is intentionally *not* the final AE-2 Butlin runner. It proves a
//! narrower prerequisite: the real embodiment contingency learner can make
//! prospective action→sensation predictions and adapt when the external world
//! reverses those contingencies, while the same capability is absent when the
//! embodiment runtime is disabled.
//!
//! Actions are externally prescribed on a balanced schedule. Therefore this
//! result says nothing yet about autonomous action selection, global cognitive
//! integration, or consciousness. It cannot mint `FunctionallySupported`.
//!
//! The current production `SensorimotorEngine::learn` adapts visual, tactile,
//! and proprioceptive outcome channels but not `auditory_expected`. This task
//! therefore holds the auditory channel constant and decodes consequences only
//! from the three channels the production learner actually updates. That keeps
//! the experiment about demonstrated learning rather than an unrelated frozen
//! feature.

use super::ae2_contingency_task::{
    score_adaptation, Action, AdaptationMetrics, Consequence, ContingencyWorld, TrialObservation,
};
use symthaea::consciousness::embodiment::external_contingency::ExternalContingencyRuntime;
use symthaea::consciousness::embodied_cognition::{
    ActionPattern, BodyPart, MovementType, SensoryPrediction,
};
use symthaea::hdc::binary_hv::BinaryHV;

const ACQUISITION_TRIALS: usize = 64;
const REVERSAL_TRIALS: usize = 64;
const RECOVERY_WINDOW: usize = 6;
const NEUTRAL_AUDITORY_CHANNEL: f64 = 0.50;

#[derive(Debug, Clone, PartialEq)]
pub struct ComponentArmResult {
    pub enabled: bool,
    pub metrics: AdaptationMetrics,
    pub observation_updates: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ae2SensorimotorComponentRun {
    pub baseline: ComponentArmResult,
    pub embodiment_disabled: ComponentArmResult,
}

impl Ae2SensorimotorComponentRun {
    pub fn claim_scope_note(&self) -> &'static str {
        "Component-level functional prerequisite only: Symthaea's production sensorimotor \
         contingency engine is being tested against an externally controlled reversal task. \
         Actions are prescribed, not autonomously selected; the disabled arm removes the \
         contingency runtime by construction; no unrelated sham or full CognitiveLoopService \
         integration is established here. This result cannot promote an AE-2 Butlin support tier."
    }
}

pub fn run_component_experiment() -> Ae2SensorimotorComponentRun {
    Ae2SensorimotorComponentRun {
        baseline: run_arm(true),
        embodiment_disabled: run_arm(false),
    }
}

fn run_arm(enabled: bool) -> ComponentArmResult {
    let mut runtime = ExternalContingencyRuntime::new(enabled, 32);
    let mut world = ContingencyWorld::new(ACQUISITION_TRIALS)
        .expect("fixed AE-2 acquisition schedule is valid");
    let mut observations = Vec::with_capacity(ACQUISITION_TRIALS + REVERSAL_TRIALS);
    let mut observation_updates = 0_u64;

    for trial in 0..(ACQUISITION_TRIALS + REVERSAL_TRIALS) {
        let action = if trial % 2 == 0 { Action::A } else { Action::B };
        let action_pattern = action_pattern(action);

        let predicted = runtime
            .predict(&action_pattern)
            .ok()
            .flatten()
            .map(|p| decode_consequence(&p));

        let observation = world.step(action, predicted);
        let actual = consequence_sensation(observation.observed);

        if runtime.observe(action_pattern, actual).is_ok() {
            observation_updates += 1;
        }
        observations.push(observation);
    }

    ComponentArmResult {
        enabled,
        metrics: score_adaptation(&observations, RECOVERY_WINDOW),
        observation_updates,
    }
}

fn action_pattern(action: Action) -> ActionPattern {
    match action {
        Action::A => ActionPattern {
            involved_parts: vec![BodyPart::RightHand],
            movement_type: MovementType::Reach,
            intensity: 0.5,
            duration: 1,
            encoding: BinaryHV::random(0xAE20_A),
        },
        Action::B => ActionPattern {
            involved_parts: vec![BodyPart::LeftHand],
            movement_type: MovementType::Push,
            intensity: 0.5,
            duration: 1,
            encoding: BinaryHV::random(0xAE20_B),
        },
    }
}

fn consequence_sensation(consequence: Consequence) -> SensoryPrediction {
    match consequence {
        Consequence::X => SensoryPrediction {
            visual_change: 0.15,
            tactile_expected: 0.85,
            proprioceptive_change: 0.20,
            auditory_expected: NEUTRAL_AUDITORY_CHANNEL,
            encoding: BinaryHV::random(0xAE20_0),
        },
        Consequence::Y => SensoryPrediction {
            visual_change: 0.85,
            tactile_expected: 0.15,
            proprioceptive_change: 0.80,
            auditory_expected: NEUTRAL_AUDITORY_CHANNEL,
            encoding: BinaryHV::random(0xAE20_1),
        },
    }
}

fn decode_consequence(prediction: &SensoryPrediction) -> Consequence {
    let dx = learned_channel_distance(prediction, &consequence_sensation(Consequence::X));
    let dy = learned_channel_distance(prediction, &consequence_sensation(Consequence::Y));
    if dx <= dy {
        Consequence::X
    } else {
        Consequence::Y
    }
}

/// Distance over exactly the channels `SensorimotorEngine::learn` currently
/// updates. Auditory output is intentionally excluded until production
/// learning adapts that channel too.
fn learned_channel_distance(a: &SensoryPrediction, b: &SensoryPrediction) -> f64 {
    ((a.visual_change - b.visual_change).powi(2)
        + (a.tactile_expected - b.tactile_expected).powi(2)
        + (a.proprioceptive_change - b.proprioceptive_change).powi(2))
    .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_sensorimotor_engine_learns_and_reverses() {
        let run = run_component_experiment();
        assert_eq!(run.baseline.observation_updates, 128);
        assert!(run.baseline.metrics.prediction_coverage > 0.95);
        assert!(run.baseline.metrics.acquisition_prediction_accuracy > 0.95);
        assert!(run.baseline.metrics.reversal_prediction_accuracy > 0.70);
        assert!(run.baseline.metrics.reversal_recovery_trial.is_some());
    }

    #[test]
    fn disabled_embodiment_cannot_learn_through_side_channel() {
        let run = run_component_experiment();
        assert_eq!(run.embodiment_disabled.observation_updates, 0);
        assert_eq!(run.embodiment_disabled.metrics.prediction_coverage, 0.0);
        assert_eq!(run.embodiment_disabled.metrics.acquisition_prediction_accuracy, 0.0);
        assert_eq!(run.embodiment_disabled.metrics.reversal_prediction_accuracy, 0.0);
        assert_eq!(run.embodiment_disabled.metrics.reversal_recovery_trial, None);
    }

    #[test]
    fn task_does_not_discriminate_on_unlearned_auditory_channel() {
        let x = consequence_sensation(Consequence::X);
        let y = consequence_sensation(Consequence::Y);
        assert_eq!(x.auditory_expected, y.auditory_expected);
        assert!(learned_channel_distance(&x, &y) > 0.0);
    }

    #[test]
    fn component_result_explicitly_refuses_butlin_promotion() {
        let run = run_component_experiment();
        assert!(run.claim_scope_note().contains("cannot promote"));
        assert!(run.claim_scope_note().contains("prescribed"));
    }
}
