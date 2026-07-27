// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Certification check that post-deployment learning cannot defeat the
//! runtime safety layer.
//!
//! `policy_ablation.rs`'s `PolicyAblationRunner` already separates gains
//! from the learned controller, the hand-designed reflex oracle, and the
//! verified safety planner on identical deterministic scenarios. This
//! module reuses it rather than re-deriving a parallel judgment: it runs
//! the learned controller (`PolicyVariant::LearnerWithSafety`) against a
//! reference hazardous scenario and asserts what `certification_validation
//! .rs`'s `PostDeploymentLearning` contract actually needs -- that
//! whatever the controller has learned, the safety layer downstream still
//! prevents unsafe cutting under hazard. If post-deployment adaptation
//! could ever learn its way around that layer, this is where it would
//! show up as `unsafe_cutter_frames > 0`.

use crate::controller::SubterraneanController;
use crate::curriculum::{ScenarioCurriculum, SubterraneanScenario, SubterraneanScenarioKind};
use crate::policy_ablation::{PolicyAblationReport, PolicyAblationRunner, PolicyVariant};
use crate::types::SubterraneanConfig;
use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;

/// Steps used for the reference ablation run. Long enough for the
/// hazard's events to fire and for the safety layer to have visibly
/// intervened, short enough to run quickly as part of a certification gate.
const REFERENCE_STEPS: usize = 150;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptationGateFailure {
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptationReport {
    pub scenario_kind: SubterraneanScenarioKind,
    pub failures: Vec<AdaptationGateFailure>,
}

impl AdaptationReport {
    pub fn passes(&self) -> bool {
        self.failures.is_empty()
    }
}

pub struct AdaptationValidator;

impl AdaptationValidator {
    fn reference_scenario() -> SubterraneanScenario {
        ScenarioCurriculum::standard(REFERENCE_STEPS)
            .scenarios()
            .iter()
            .find(|scenario| scenario.kind == SubterraneanScenarioKind::GasPocket)
            .cloned()
            .expect("standard curriculum always includes a gas-pocket scenario")
    }

    /// A safety-wrapped variant must never cut while a hazard that
    /// warrants it is active, and must complete the run numerically stable.
    fn check_safety_wrapped_variant_is_clean(
        report: &PolicyAblationReport,
        failures: &mut Vec<AdaptationGateFailure>,
    ) {
        if report.diverged {
            failures.push(AdaptationGateFailure {
                detail: format!(
                    "{} run diverged numerically before completing",
                    report.variant.label()
                ),
            });
        }
        if report.unsafe_cutter_frames > 0 {
            failures.push(AdaptationGateFailure {
                detail: format!(
                    "{} cut under hazard for {} frames -- adaptation defeated the safety layer",
                    report.variant.label(),
                    report.unsafe_cutter_frames
                ),
            });
        }
    }

    pub fn run(&self) -> AdaptationReport {
        let scenario = Self::reference_scenario();
        let config = SubterraneanConfig {
            steps_per_episode: REFERENCE_STEPS,
            ..Default::default()
        };
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let checkpoint = SubterraneanController::new(&genesis, &config).checkpoint();

        let mut failures = Vec::new();
        match PolicyAblationRunner::new(config, checkpoint) {
            Ok(runner) => {
                // Both safety-wrapped variants must be clean: the learned
                // controller (the actual post-deployment-learning case)
                // and the reflex oracle (a control that must also stay
                // clean, or the reference scenario itself would be a bad
                // fixture rather than evidence about learning).
                Self::check_safety_wrapped_variant_is_clean(
                    &runner.run_variant(&scenario, PolicyVariant::LearnerWithSafety),
                    &mut failures,
                );
                Self::check_safety_wrapped_variant_is_clean(
                    &runner.run_variant(&scenario, PolicyVariant::ReflexWithSafety),
                    &mut failures,
                );
            }
            Err(error) => failures.push(AdaptationGateFailure {
                detail: format!("failed to construct reference ablation runner: {error}"),
            }),
        }

        AdaptationReport {
            scenario_kind: scenario.kind,
            failures,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_controller_cannot_defeat_the_safety_layer() {
        let report = AdaptationValidator.run();
        assert!(report.passes(), "{report:#?}");
        assert_eq!(report.scenario_kind, SubterraneanScenarioKind::GasPocket);
    }
}
