// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical ProgSuite development-stage context over long-range work architecture.
//!
//! This adapter retains the complete native-derived work architecture and
//! ordered development program, then derives the generic cross-axis stage
//! context from those exact artifacts. It makes no new composition choices.

use crate::development_context::{
    WorkDevelopmentContextErrorV1, WorkDevelopmentContextV1,
    derive_work_development_context,
};
use crate::prog_suite::ProgSuitePlanV1;
use crate::prog_suite_development_program::{
    ProgSuiteDevelopmentProgramErrorV1, ProgSuiteDevelopmentProgramV1,
    derive_prog_suite_development_program,
};
use crate::prog_suite_work_architecture::{
    ProgSuiteWorkArchitectureErrorV1, ProgSuiteWorkArchitectureV1,
    derive_prog_suite_work_architecture,
};
use serde::{Deserialize, Serialize};

pub const PROG_SUITE_DEVELOPMENT_CONTEXT_VERSION: &str =
    "melothaea-prog-suite-development-context-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteDevelopmentContextV1 {
    pub version: String,
    /// Retains native plan, FORM declaration, tonal trajectory, and metric
    /// architecture as one canonical source-derived artifact.
    pub work_architecture: ProgSuiteWorkArchitectureV1,
    /// Retains ordered thematic development semantics and their FORM-002/003
    /// bindings from the same exact native declaration.
    pub development_program: ProgSuiteDevelopmentProgramV1,
    /// Canonical stage -> tonal/metric region coverage.
    pub context: WorkDevelopmentContextV1,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteDevelopmentContextErrorV1 {
    WorkArchitecture(ProgSuiteWorkArchitectureErrorV1),
    DevelopmentProgram(ProgSuiteDevelopmentProgramErrorV1),
    DevelopmentContext(WorkDevelopmentContextErrorV1),
    SourceBindingMismatch,
    WrongVersion { found: String },
    CanonicalContextMismatch,
}

pub fn derive_prog_suite_development_context(
    plan: &ProgSuitePlanV1,
) -> Result<ProgSuiteDevelopmentContextV1, ProgSuiteDevelopmentContextErrorV1> {
    let work_architecture = derive_prog_suite_work_architecture(plan)
        .map_err(ProgSuiteDevelopmentContextErrorV1::WorkArchitecture)?;
    let development_program = derive_prog_suite_development_program(&work_architecture.binding)
        .map_err(ProgSuiteDevelopmentContextErrorV1::DevelopmentProgram)?;

    // Both adapters must retain the exact same canonical FORM source binding.
    if development_program.source_work_bridge_version != work_architecture.binding.version {
        return Err(ProgSuiteDevelopmentContextErrorV1::SourceBindingMismatch);
    }

    let context = derive_work_development_context(
        &development_program.program,
        &work_architecture.binding.work_plan,
        &work_architecture.tonal_trajectory,
        &work_architecture.metric_architecture,
    )
    .map_err(ProgSuiteDevelopmentContextErrorV1::DevelopmentContext)?;

    Ok(ProgSuiteDevelopmentContextV1 {
        version: PROG_SUITE_DEVELOPMENT_CONTEXT_VERSION.into(),
        work_architecture,
        development_program,
        context,
    })
}

impl ProgSuiteDevelopmentContextV1 {
    /// Rebuild from the retained native plan and require exact equality across
    /// all derived planning axes. Hand-editing any stage context or retained
    /// architecture cannot create ProgSuite authority.
    pub fn validate(&self) -> Result<(), ProgSuiteDevelopmentContextErrorV1> {
        if self.version != PROG_SUITE_DEVELOPMENT_CONTEXT_VERSION {
            return Err(ProgSuiteDevelopmentContextErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        self.work_architecture
            .validate()
            .map_err(ProgSuiteDevelopmentContextErrorV1::WorkArchitecture)?;
        self.context
            .validate(
                &self.development_program.program,
                &self.work_architecture.binding.work_plan,
                &self.work_architecture.tonal_trajectory,
                &self.work_architecture.metric_architecture,
            )
            .map_err(ProgSuiteDevelopmentContextErrorV1::DevelopmentContext)?;

        let canonical = derive_prog_suite_development_context(
            &self.work_architecture.binding.native_plan,
        )?;
        if &canonical != self {
            return Err(ProgSuiteDevelopmentContextErrorV1::CanonicalContextMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Key, PitchClass, Style, plan_prog_suite};

    fn plan(seed: u64) -> ProgSuitePlanV1 {
        plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            seed,
            &Style::ProgFolk.spec(),
        )
        .unwrap()
    }

    #[test]
    fn each_native_development_stage_has_exact_section_context() {
        let artifact = derive_prog_suite_development_context(&plan(5)).unwrap();
        artifact.validate().unwrap();

        for (stage_name, region_id) in [
            ("prog-suite:development:B", "prog-suite:B"),
            ("prog-suite:development:C", "prog-suite:C"),
            ("prog-suite:development:ReturnA", "prog-suite:ReturnA"),
        ] {
            let stage = &artifact.context.stages[stage_name];
            assert_eq!(stage.tonal_region_ids, vec![region_id.to_string()]);
            assert_eq!(stage.metric_region_ids, vec![region_id.to_string()]);
        }
    }

    #[test]
    fn c_stage_exposes_relative_key_and_five_four_context_without_copying_values() {
        let artifact = derive_prog_suite_development_context(&plan(5)).unwrap();
        let stage = &artifact.context.stages["prog-suite:development:C"];
        let tonal_id = &stage.tonal_region_ids[0];
        let metric_id = &stage.metric_region_ids[0];
        assert_eq!(
            artifact.work_architecture.tonal_trajectory.regions[tonal_id].key,
            artifact.work_architecture.binding.native_plan.home_key.relative()
        );
        assert_eq!(
            artifact.work_architecture.metric_architecture.regions[metric_id]
                .meter
                .numerator(),
            5
        );
    }

    #[test]
    fn ordered_b_composite_keeps_same_home_key_seven_four_context() {
        let artifact = derive_prog_suite_development_context(&plan(5)).unwrap();
        let b_program = &artifact.development_program.program.stages[0];
        assert_eq!(b_program.operations.len(), 2);
        let b_context = &artifact.context.stages[&b_program.stage_id];
        let tonal = &artifact.work_architecture.tonal_trajectory.regions
            [&b_context.tonal_region_ids[0]];
        let metric = &artifact.work_architecture.metric_architecture.regions
            [&b_context.metric_region_ids[0]];
        assert_eq!(tonal.key, artifact.work_architecture.binding.native_plan.home_key);
        assert_eq!(metric.meter.numerator(), 7);
    }

    #[test]
    fn hand_edited_context_cannot_become_prog_suite_authority() {
        let mut artifact = derive_prog_suite_development_context(&plan(5)).unwrap();
        artifact
            .context
            .stages
            .get_mut("prog-suite:development:C")
            .unwrap()
            .metric_region_ids = vec!["prog-suite:B".into()];
        assert!(matches!(
            artifact.validate(),
            Err(ProgSuiteDevelopmentContextErrorV1::DevelopmentContext(_))
                | Err(ProgSuiteDevelopmentContextErrorV1::CanonicalContextMismatch)
        ));
    }
}
