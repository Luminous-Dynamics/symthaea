//! Deterministic two-action/two-consequence contingency-reversal task for AE-2.
//!
//! The task itself is architecture-agnostic: a runner supplies an action and
//! a predicted consequence. This module supplies the environment transition
//! and computes adaptation metrics. Passing these unit tests proves only the
//! task semantics, not that Symthaea has passed the task.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    A,
    B,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Consequence {
    X,
    Y,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Acquisition,
    Reversal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrialObservation {
    pub phase: Phase,
    pub action: Action,
    pub predicted: Consequence,
    pub observed: Consequence,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptationMetrics {
    pub acquisition_prediction_accuracy: f64,
    pub reversal_prediction_accuracy: f64,
    /// First reversal-trial index after which `window` consecutive
    /// predictions are correct. None means criterion was never reached.
    pub reversal_recovery_trial: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ContingencyWorld {
    acquisition_trials: usize,
    trial: usize,
}

impl ContingencyWorld {
    pub fn new(acquisition_trials: usize) -> Result<Self, String> {
        if acquisition_trials == 0 {
            return Err("acquisition_trials must be nonzero".into());
        }
        Ok(Self {
            acquisition_trials,
            trial: 0,
        })
    }

    pub fn phase(&self) -> Phase {
        if self.trial < self.acquisition_trials {
            Phase::Acquisition
        } else {
            Phase::Reversal
        }
    }

    pub fn consequence_for(&self, action: Action) -> Consequence {
        match (self.phase(), action) {
            (Phase::Acquisition, Action::A) => Consequence::X,
            (Phase::Acquisition, Action::B) => Consequence::Y,
            (Phase::Reversal, Action::A) => Consequence::Y,
            (Phase::Reversal, Action::B) => Consequence::X,
        }
    }

    pub fn step(&mut self, action: Action, predicted: Consequence) -> TrialObservation {
        let phase = self.phase();
        let observed = self.consequence_for(action);
        self.trial += 1;
        TrialObservation {
            phase,
            action,
            predicted,
            observed,
        }
    }
}

pub fn score_adaptation(observations: &[TrialObservation], window: usize) -> AdaptationMetrics {
    let accuracy = |phase: Phase| {
        let samples: Vec<_> = observations.iter().filter(|o| o.phase == phase).collect();
        if samples.is_empty() {
            return 0.0;
        }
        let correct = samples.iter().filter(|o| o.predicted == o.observed).count();
        correct as f64 / samples.len() as f64
    };

    let reversal: Vec<_> = observations
        .iter()
        .filter(|o| o.phase == Phase::Reversal)
        .collect();
    let recovery = if window == 0 {
        None
    } else {
        reversal
            .windows(window)
            .position(|w| w.iter().all(|o| o.predicted == o.observed))
            .map(|idx| idx + window)
    };

    AdaptationMetrics {
        acquisition_prediction_accuracy: accuracy(Phase::Acquisition),
        reversal_prediction_accuracy: accuracy(Phase::Reversal),
        reversal_recovery_trial: recovery,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reversal_swaps_both_action_consequences() {
        let mut world = ContingencyWorld::new(2).unwrap();
        assert_eq!(world.consequence_for(Action::A), Consequence::X);
        world.step(Action::A, Consequence::X);
        world.step(Action::B, Consequence::Y);
        assert_eq!(world.phase(), Phase::Reversal);
        assert_eq!(world.consequence_for(Action::A), Consequence::Y);
        assert_eq!(world.consequence_for(Action::B), Consequence::X);
    }

    #[test]
    fn scorer_detects_reversal_recovery() {
        let observations = vec![
            TrialObservation { phase: Phase::Acquisition, action: Action::A, predicted: Consequence::X, observed: Consequence::X },
            TrialObservation { phase: Phase::Acquisition, action: Action::B, predicted: Consequence::Y, observed: Consequence::Y },
            TrialObservation { phase: Phase::Reversal, action: Action::A, predicted: Consequence::X, observed: Consequence::Y },
            TrialObservation { phase: Phase::Reversal, action: Action::B, predicted: Consequence::X, observed: Consequence::X },
            TrialObservation { phase: Phase::Reversal, action: Action::A, predicted: Consequence::Y, observed: Consequence::Y },
        ];
        let m = score_adaptation(&observations, 2);
        assert_eq!(m.acquisition_prediction_accuracy, 1.0);
        assert!((m.reversal_prediction_accuracy - (2.0 / 3.0)).abs() < 1e-12);
        assert_eq!(m.reversal_recovery_trial, Some(3));
    }
}
