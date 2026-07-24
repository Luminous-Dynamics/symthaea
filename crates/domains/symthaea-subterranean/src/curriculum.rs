// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic emergency curriculum for subterranean policy training.
//!
//! The baseline trainer always started from the same benign home state. This
//! module supplies reproducible held-out-style scenarios and timed disturbances
//! so learning is exposed to the hazards the runtime supervisor must handle.

use crate::mission::SubterraneanMissionIntent;
use crate::types::{
    AQUIFER_RISK, BATTERY_RATIO, COMM_SIGNAL, CUTTER_TEMP_C, DEPTH_M, GAS_RISK, HULL_STRESS,
    LOCALIZATION_CONFIDENCE, RELAY_LINK_QUALITY, ROOF_STABILITY, SEAL_INTEGRITY, SLIP_RATIO,
    SLURRY_LOAD, SPOIL_BUFFER_FILL, SubterraneanState, WATER_INGRESS_RATIO,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubterraneanScenarioKind {
    NominalTransit,
    AquiferBreach,
    GasPocket,
    RoofFailure,
    CommunicationsBlackout,
    SpoilJam,
    ThermalRunaway,
    BatteryEmergency,
    SensorFault,
    FloodBlackout,
    GasLocalizationLoss,
    LowBatteryWithdrawal,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ScenarioEventKind {
    AquiferBreach,
    GasPocket,
    RoofFailure,
    CommunicationsBlackout,
    SpoilJam,
    ThermalRunaway,
    BatteryDrop,
    GasSensorDropout,
    LocalizationLoss,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScenarioEvent {
    pub at_step: usize,
    pub kind: ScenarioEventKind,
    pub magnitude: f64,
}

impl ScenarioEvent {
    pub fn apply(self, state: &mut SubterraneanState) {
        let magnitude = self.magnitude.clamp(0.0, 1.0);
        match self.kind {
            ScenarioEventKind::AquiferBreach => {
                state.channels[AQUIFER_RISK] =
                    state.channels[AQUIFER_RISK].max(0.7 + 0.3 * magnitude);
                state.channels[WATER_INGRESS_RATIO] =
                    state.channels[WATER_INGRESS_RATIO].max(0.25 + 0.65 * magnitude);
                state.channels[SEAL_INTEGRITY] =
                    state.channels[SEAL_INTEGRITY].min(0.75 - 0.55 * magnitude);
            }
            ScenarioEventKind::GasPocket => {
                state.channels[GAS_RISK] = state.channels[GAS_RISK].max(0.35 + 0.65 * magnitude);
            }
            ScenarioEventKind::RoofFailure => {
                state.channels[ROOF_STABILITY] =
                    state.channels[ROOF_STABILITY].min(0.6 - 0.5 * magnitude);
                state.channels[HULL_STRESS] =
                    state.channels[HULL_STRESS].max(0.4 + 0.55 * magnitude);
            }
            ScenarioEventKind::CommunicationsBlackout => {
                state.channels[COMM_SIGNAL] = (0.25 * (1.0 - magnitude)).clamp(0.0, 1.0);
                state.channels[RELAY_LINK_QUALITY] = (0.2 * (1.0 - magnitude)).clamp(0.0, 1.0);
                state.channels[LOCALIZATION_CONFIDENCE] =
                    state.channels[LOCALIZATION_CONFIDENCE].min(0.55 - 0.35 * magnitude);
            }
            ScenarioEventKind::SpoilJam => {
                state.channels[SPOIL_BUFFER_FILL] =
                    state.channels[SPOIL_BUFFER_FILL].max(0.72 + 0.27 * magnitude);
                state.channels[SLURRY_LOAD] =
                    state.channels[SLURRY_LOAD].max(0.6 + 0.35 * magnitude);
                state.channels[SLIP_RATIO] =
                    state.channels[SLIP_RATIO].max(0.45 + 0.45 * magnitude);
            }
            ScenarioEventKind::ThermalRunaway => {
                state.channels[CUTTER_TEMP_C] =
                    state.channels[CUTTER_TEMP_C].max(105.0 + 70.0 * magnitude);
            }
            ScenarioEventKind::BatteryDrop => {
                state.channels[BATTERY_RATIO] =
                    state.channels[BATTERY_RATIO].min(0.28 - 0.24 * magnitude);
            }
            ScenarioEventKind::GasSensorDropout => {
                state.channels[GAS_RISK] = f64::NAN;
            }
            ScenarioEventKind::LocalizationLoss => {
                state.channels[LOCALIZATION_CONFIDENCE] =
                    state.channels[LOCALIZATION_CONFIDENCE].min(0.5 - 0.45 * magnitude);
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubterraneanScenario {
    pub kind: SubterraneanScenarioKind,
    pub initial_depth_m: f64,
    pub events: Vec<ScenarioEvent>,
}

impl SubterraneanScenario {
    pub fn mission_intent(&self) -> SubterraneanMissionIntent {
        match self.kind {
            SubterraneanScenarioKind::NominalTransit => SubterraneanMissionIntent::FollowVein,
            SubterraneanScenarioKind::CommunicationsBlackout
            | SubterraneanScenarioKind::SensorFault => SubterraneanMissionIntent::HoldPosition,
            SubterraneanScenarioKind::AquiferBreach
            | SubterraneanScenarioKind::GasPocket
            | SubterraneanScenarioKind::RoofFailure
            | SubterraneanScenarioKind::SpoilJam
            | SubterraneanScenarioKind::ThermalRunaway
            | SubterraneanScenarioKind::BatteryEmergency
            | SubterraneanScenarioKind::FloodBlackout
            | SubterraneanScenarioKind::GasLocalizationLoss
            | SubterraneanScenarioKind::LowBatteryWithdrawal => {
                SubterraneanMissionIntent::ReturnHome
            }
        }
    }

    pub fn nominal() -> Self {
        Self {
            kind: SubterraneanScenarioKind::NominalTransit,
            initial_depth_m: 8.0,
            events: Vec::new(),
        }
    }

    pub fn initialize(&self, state: &mut SubterraneanState) {
        *state = SubterraneanState::home();
        state.channels[DEPTH_M] = self.initial_depth_m.clamp(0.0, 200.0);
    }

    pub fn apply_events(&self, step: usize, state: &mut SubterraneanState) -> usize {
        let mut applied = 0;
        for event in &self.events {
            if event.at_step == step {
                event.apply(state);
                applied += 1;
            }
        }
        applied
    }
}

#[derive(Debug, Clone)]
pub struct ScenarioCurriculum {
    scenarios: Vec<SubterraneanScenario>,
}

impl ScenarioCurriculum {
    pub fn standard(steps_per_episode: usize) -> Self {
        let inject = (steps_per_episode / 5).max(1);
        let scenario = |kind, event_kind, magnitude, depth| SubterraneanScenario {
            kind,
            initial_depth_m: depth,
            events: vec![ScenarioEvent {
                at_step: inject,
                kind: event_kind,
                magnitude,
            }],
        };
        Self {
            scenarios: vec![
                SubterraneanScenario::nominal(),
                scenario(
                    SubterraneanScenarioKind::AquiferBreach,
                    ScenarioEventKind::AquiferBreach,
                    0.9,
                    70.0,
                ),
                scenario(
                    SubterraneanScenarioKind::GasPocket,
                    ScenarioEventKind::GasPocket,
                    0.9,
                    55.0,
                ),
                scenario(
                    SubterraneanScenarioKind::RoofFailure,
                    ScenarioEventKind::RoofFailure,
                    0.9,
                    45.0,
                ),
                scenario(
                    SubterraneanScenarioKind::CommunicationsBlackout,
                    ScenarioEventKind::CommunicationsBlackout,
                    1.0,
                    90.0,
                ),
                scenario(
                    SubterraneanScenarioKind::SpoilJam,
                    ScenarioEventKind::SpoilJam,
                    0.9,
                    35.0,
                ),
                scenario(
                    SubterraneanScenarioKind::ThermalRunaway,
                    ScenarioEventKind::ThermalRunaway,
                    0.9,
                    30.0,
                ),
                scenario(
                    SubterraneanScenarioKind::BatteryEmergency,
                    ScenarioEventKind::BatteryDrop,
                    0.9,
                    18.0,
                ),
                scenario(
                    SubterraneanScenarioKind::SensorFault,
                    ScenarioEventKind::GasSensorDropout,
                    1.0,
                    40.0,
                ),
            ],
        }
    }

    /// Held-out compound hazards are intentionally excluded from the training
    /// curriculum. They test composition and recovery sequencing rather than
    /// memorization of one disturbance family.
    pub fn compound_holdout(steps_per_episode: usize) -> Self {
        let inject = (steps_per_episode / 5).max(1);
        let delayed = (inject + steps_per_episode / 10).min(steps_per_episode.saturating_sub(1));
        Self {
            scenarios: vec![
                SubterraneanScenario {
                    kind: SubterraneanScenarioKind::FloodBlackout,
                    initial_depth_m: 85.0,
                    events: vec![
                        ScenarioEvent {
                            at_step: inject,
                            kind: ScenarioEventKind::AquiferBreach,
                            magnitude: 0.9,
                        },
                        ScenarioEvent {
                            at_step: delayed,
                            kind: ScenarioEventKind::CommunicationsBlackout,
                            magnitude: 1.0,
                        },
                    ],
                },
                SubterraneanScenario {
                    kind: SubterraneanScenarioKind::GasLocalizationLoss,
                    initial_depth_m: 65.0,
                    events: vec![
                        ScenarioEvent {
                            at_step: inject,
                            kind: ScenarioEventKind::GasPocket,
                            magnitude: 0.9,
                        },
                        ScenarioEvent {
                            at_step: delayed,
                            kind: ScenarioEventKind::LocalizationLoss,
                            magnitude: 1.0,
                        },
                    ],
                },
                SubterraneanScenario {
                    kind: SubterraneanScenarioKind::LowBatteryWithdrawal,
                    initial_depth_m: 18.0,
                    events: vec![
                        ScenarioEvent {
                            at_step: inject,
                            kind: ScenarioEventKind::GasPocket,
                            magnitude: 0.85,
                        },
                        ScenarioEvent {
                            at_step: delayed,
                            kind: ScenarioEventKind::BatteryDrop,
                            magnitude: 0.85,
                        },
                    ],
                },
            ],
        }
    }

    pub fn comprehensive(steps_per_episode: usize) -> Self {
        let mut scenarios = Self::standard(steps_per_episode).scenarios;
        scenarios.extend(Self::compound_holdout(steps_per_episode).scenarios);
        Self { scenarios }
    }

    pub fn len(&self) -> usize {
        self.scenarios.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }

    pub fn get(&self, index: usize) -> &SubterraneanScenario {
        &self.scenarios[index % self.scenarios.len()]
    }

    pub fn scenarios(&self) -> &[SubterraneanScenario] {
        &self.scenarios
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_curriculum_covers_each_major_emergency_family() {
        let curriculum = ScenarioCurriculum::standard(100);
        assert_eq!(curriculum.len(), 9);
        assert!(
            curriculum
                .scenarios()
                .iter()
                .any(|scenario| scenario.kind == SubterraneanScenarioKind::AquiferBreach)
        );
        assert!(
            curriculum
                .scenarios()
                .iter()
                .any(|scenario| scenario.kind == SubterraneanScenarioKind::BatteryEmergency)
        );
        assert!(
            curriculum
                .scenarios()
                .iter()
                .any(|scenario| scenario.kind == SubterraneanScenarioKind::SensorFault)
        );
    }

    #[test]
    fn scenario_events_are_deterministic_and_step_bounded() {
        let curriculum = ScenarioCurriculum::standard(100);
        let scenario = curriculum.get(1);
        let mut a = SubterraneanState::home();
        let mut b = SubterraneanState::home();
        scenario.initialize(&mut a);
        scenario.initialize(&mut b);
        for step in 0..100 {
            scenario.apply_events(step, &mut a);
            scenario.apply_events(step, &mut b);
        }
        assert_eq!(a.channels, b.channels);
        assert!(a.channels[WATER_INGRESS_RATIO] > 0.5);
    }

    #[test]
    fn compound_holdout_contains_preregistered_multi_event_cases() {
        let curriculum = ScenarioCurriculum::compound_holdout(100);
        assert_eq!(curriculum.len(), 3);
        assert!(
            curriculum
                .scenarios()
                .iter()
                .all(|scenario| scenario.events.len() == 2)
        );
        assert!(
            curriculum
                .scenarios()
                .iter()
                .any(|scenario| { scenario.kind == SubterraneanScenarioKind::FloodBlackout })
        );
    }
}
