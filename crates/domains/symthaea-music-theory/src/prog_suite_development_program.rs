// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native ProgSuite -> ordered [`crate::DevelopmentProgramV1`] projection.
//!
//! ProgSuite already freezes its architectural choices in `ProgSuitePlanV1`
//! and translates them into FORM-000/001/002/003 through
//! `ProgSuiteWorkBindingV1`. This adapter does not re-plan from `source_seed`
//! and does not inspect a completed score. It converts the already-frozen B,
//! C, and ReturnA declarations into an ordered development program.
//!
//! The important semantic gain is the native Retrograde-Inversion case.
//! FORM-002 V1 deliberately stores a canonical set and therefore preserves that
//! ordered composite as one `Other(label)` projection. `DevelopmentProgramV1`
//! can now retain the actual ordered operations `Inversion -> Retrograde` while
//! binding them to that exact FORM-002 projection and to the existing FORM-003
//! promise that makes the stage due.

use crate::development_program::{
    DEVELOPMENT_PROGRAM_VERSION, DevelopmentEvidenceRequirementV1,
    DevelopmentGraphProjectionV1, DevelopmentOperationV1, DevelopmentProgramErrorV1,
    DevelopmentProgramV1, DevelopmentStageV1,
};
use crate::prog_suite::ProgSuiteTransformV1;
use crate::prog_suite_work_bridge::{
    PROG_SUITE_WORK_BRIDGE_VERSION, ProgSuiteWorkBindingV1, ProgSuiteWorkBridgeErrorV1,
    bridge_prog_suite_plan,
};
use crate::thematic_identity::ThematicTransformationClassV1;
use crate::work_obligation::WorkObligationKindV2;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PROG_SUITE_DEVELOPMENT_PROGRAM_VERSION: &str =
    "melothaea-prog-suite-development-program-v1";

const PROGRAM_ID: &str = "prog-suite:development-program";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteDevelopmentProgramV1 {
    pub version: String,
    /// Exact generic contract version consumed by this adapter.
    pub development_program_version: String,
    /// Exact FORM bridge version from which the program was projected.
    pub source_work_bridge_version: String,
    /// Exact native planner contract retained by the source binding.
    pub source_plan_version: String,
    pub program: DevelopmentProgramV1,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteDevelopmentProgramErrorV1 {
    SourceBridge(ProgSuiteWorkBridgeErrorV1),
    /// The supplied serialized binding differs from the canonical translation
    /// of its retained native plan. Do not grant program authority to stale or
    /// hand-edited FORM artifacts.
    SourceBindingMismatch,
    MissingSectionBinding { section_index: usize },
    MissingNativeSection { section_index: usize },
    MissingDerivation { derivation_id: String },
    MissingDerivationObligation { derivation_id: String },
    DuplicateDerivationObligation { derivation_id: String },
    CompositeProjectionMissingCustomLabel { derivation_id: String },
    DevelopmentProgram(DevelopmentProgramErrorV1),
}

#[derive(Debug, Clone)]
struct StageSeed {
    section_index: usize,
    stage_id: String,
    work_node_id: String,
    input_identity_id: String,
    output_identity_id: String,
    derivation_id: String,
    obligation_id: String,
    operations: Vec<DevelopmentOperationV1>,
    graph_projection: DevelopmentGraphProjectionV1,
    evidence_requirement: DevelopmentEvidenceRequirementV1,
}

/// Derive the ordered development program from one exact native work binding.
///
/// This function deliberately reconstructs the canonical work binding from the
/// retained frozen native plan and requires full equality before proceeding.
/// That is validation, not re-planning: no choice is re-derived from
/// `source_seed`, and the frozen native plan remains authoritative.
pub fn derive_prog_suite_development_program(
    binding: &ProgSuiteWorkBindingV1,
) -> Result<ProgSuiteDevelopmentProgramV1, ProgSuiteDevelopmentProgramErrorV1> {
    let canonical = bridge_prog_suite_plan(&binding.native_plan)
        .map_err(ProgSuiteDevelopmentProgramErrorV1::SourceBridge)?;
    if &canonical != binding {
        return Err(ProgSuiteDevelopmentProgramErrorV1::SourceBindingMismatch);
    }

    let source_identity_id = section_binding(binding, 0)?.thematic_identity_id.clone();

    let mut seeds = Vec::with_capacity(3);
    for section_index in [1usize, 2, 3] {
        seeds.push(stage_seed(binding, section_index)?);
    }

    // Program prerequisites are not a second hand-maintained dependency graph.
    // They are the subset of each stage's existing FORM-003 obligation
    // prerequisites that correspond to another stage in this same program.
    let obligation_to_stage: BTreeMap<_, _> = seeds
        .iter()
        .map(|seed| (seed.obligation_id.clone(), seed.stage_id.clone()))
        .collect();

    let mut stages = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let section = binding.native_plan.sections.get(seed.section_index).ok_or(
            ProgSuiteDevelopmentProgramErrorV1::MissingNativeSection {
                section_index: seed.section_index,
            },
        )?;
        let obligation = &binding.obligation_plan.obligations[&seed.obligation_id];
        let prerequisite_stage_ids = obligation
            .prerequisites
            .iter()
            .filter_map(|obligation_id| obligation_to_stage.get(obligation_id).cloned())
            .collect();

        stages.push(DevelopmentStageV1 {
            stage_id: seed.stage_id,
            work_node_id: seed.work_node_id,
            start: section.start,
            end: section.end,
            input_identity_id: seed.input_identity_id,
            output_identity_id: seed.output_identity_id,
            derivation_id: seed.derivation_id,
            obligation_id: seed.obligation_id,
            prerequisite_stage_ids,
            operations: seed.operations,
            graph_projection: seed.graph_projection,
            evidence_requirement: seed.evidence_requirement,
        });
    }

    let program = DevelopmentProgramV1 {
        version: DEVELOPMENT_PROGRAM_VERSION.into(),
        program_id: PROGRAM_ID.into(),
        source_identity_id,
        stages,
    };
    program
        .validate(
            &binding.work_plan,
            &binding.thematic_graph,
            &binding.obligation_plan,
        )
        .map_err(ProgSuiteDevelopmentProgramErrorV1::DevelopmentProgram)?;

    Ok(ProgSuiteDevelopmentProgramV1 {
        version: PROG_SUITE_DEVELOPMENT_PROGRAM_VERSION.into(),
        development_program_version: DEVELOPMENT_PROGRAM_VERSION.into(),
        source_work_bridge_version: PROG_SUITE_WORK_BRIDGE_VERSION.into(),
        source_plan_version: binding.native_plan.version.clone(),
        program,
    })
}

fn stage_seed(
    binding: &ProgSuiteWorkBindingV1,
    section_index: usize,
) -> Result<StageSeed, ProgSuiteDevelopmentProgramErrorV1> {
    let section_binding = section_binding(binding, section_index)?;
    let section = binding.native_plan.sections.get(section_index).ok_or(
        ProgSuiteDevelopmentProgramErrorV1::MissingNativeSection { section_index },
    )?;
    let derivation_id = section_binding
        .derivation_id
        .clone()
        .ok_or_else(|| ProgSuiteDevelopmentProgramErrorV1::MissingDerivation {
            derivation_id: format!("<section-{section_index}-has-no-derivation>"),
        })?;
    let derivation = binding
        .thematic_graph
        .derivations
        .get(&derivation_id)
        .ok_or_else(|| ProgSuiteDevelopmentProgramErrorV1::MissingDerivation {
            derivation_id: derivation_id.clone(),
        })?;
    let obligation_id = obligation_for_derivation(binding, &derivation_id)?;

    let stage_name = match section_index {
        1 => "B",
        2 => "C",
        3 => "ReturnA",
        _ => "unknown",
    };
    let stage_id = format!("prog-suite:development:{stage_name}");
    let operations = operations_for_transform(stage_name, section.transformation);

    let (graph_projection, evidence_requirement) = if operations.len() == 1 {
        (
            DevelopmentGraphProjectionV1::ExactSingleClass,
            DevelopmentEvidenceRequirementV1::SingleOperationRelationMeasured,
        )
    } else {
        let label = match derivation.transformations.as_slice() {
            [ThematicTransformationClassV1::Other(label)] if !label.trim().is_empty() => {
                label.clone()
            }
            _ => {
                return Err(
                    ProgSuiteDevelopmentProgramErrorV1::CompositeProjectionMissingCustomLabel {
                        derivation_id,
                    },
                );
            }
        };
        (
            DevelopmentGraphProjectionV1::OrderedComposite {
                label,
                operation_ids: operations
                    .iter()
                    .map(|operation| operation.operation_id.clone())
                    .collect(),
            },
            DevelopmentEvidenceRequirementV1::OrderedCompositeRelationMeasured,
        )
    };

    Ok(StageSeed {
        section_index,
        stage_id,
        work_node_id: section_binding.work_node_id.clone(),
        input_identity_id: derivation.source_id.clone(),
        output_identity_id: section_binding.thematic_identity_id.clone(),
        derivation_id,
        obligation_id,
        operations,
        graph_projection,
        evidence_requirement,
    })
}

fn section_binding(
    binding: &ProgSuiteWorkBindingV1,
    section_index: usize,
) -> Result<&crate::prog_suite_work_bridge::ProgSuiteSectionWorkBindingV1, ProgSuiteDevelopmentProgramErrorV1>
{
    binding
        .section_bindings
        .iter()
        .find(|section| section.section_index == section_index)
        .ok_or(ProgSuiteDevelopmentProgramErrorV1::MissingSectionBinding { section_index })
}

fn obligation_for_derivation(
    binding: &ProgSuiteWorkBindingV1,
    derivation_id: &str,
) -> Result<String, ProgSuiteDevelopmentProgramErrorV1> {
    let mut matches = binding
        .obligation_plan
        .obligations
        .iter()
        .filter(|(_, obligation)| {
            matches!(
                &obligation.kind,
                WorkObligationKindV2::RealizeThematicDerivation {
                    derivation_id: candidate
                } if candidate == derivation_id
            )
        })
        .map(|(id, _)| id.clone());

    let Some(first) = matches.next() else {
        return Err(
            ProgSuiteDevelopmentProgramErrorV1::MissingDerivationObligation {
                derivation_id: derivation_id.into(),
            },
        );
    };
    if matches.next().is_some() {
        return Err(
            ProgSuiteDevelopmentProgramErrorV1::DuplicateDerivationObligation {
                derivation_id: derivation_id.into(),
            },
        );
    }
    Ok(first)
}

fn operations_for_transform(
    stage_name: &str,
    transform: ProgSuiteTransformV1,
) -> Vec<DevelopmentOperationV1> {
    let operation = |ordinal: usize, name: &str, class| DevelopmentOperationV1 {
        operation_id: format!(
            "prog-suite:development:{stage_name}:op-{ordinal:02}-{name}"
        ),
        class,
    };

    match transform {
        ProgSuiteTransformV1::Original => vec![operation(
            1,
            "literal-return",
            ThematicTransformationClassV1::LiteralReturn,
        )],
        ProgSuiteTransformV1::Inversion => vec![operation(
            1,
            "inversion",
            ThematicTransformationClassV1::Inversion,
        )],
        ProgSuiteTransformV1::Retrograde => vec![operation(
            1,
            "retrograde",
            ThematicTransformationClassV1::Retrograde,
        )],
        ProgSuiteTransformV1::RetrogradeInversion => vec![
            operation(1, "inversion", ThematicTransformationClassV1::Inversion),
            operation(2, "retrograde", ThematicTransformationClassV1::Retrograde),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Key, PitchClass, Style, bridge_prog_suite_plan, plan_prog_suite};

    fn binding() -> ProgSuiteWorkBindingV1 {
        let plan = plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            5,
            &Style::ProgFolk.spec(),
        )
        .unwrap();
        bridge_prog_suite_plan(&plan).unwrap()
    }

    #[test]
    fn default_prog_suite_becomes_three_ordered_work_stages() {
        let binding = binding();
        let projected = derive_prog_suite_development_program(&binding).unwrap();
        let program = &projected.program;
        assert_eq!(program.stages.len(), 3);
        assert_eq!(program.source_identity_id, "prog-suite:P");
        assert_eq!(
            program
                .stages
                .iter()
                .map(|stage| stage.stage_id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "prog-suite:development:B",
                "prog-suite:development:C",
                "prog-suite:development:ReturnA",
            ]
        );
        program
            .validate(
                &binding.work_plan,
                &binding.thematic_graph,
                &binding.obligation_plan,
            )
            .unwrap();
    }

    #[test]
    fn native_retrograde_inversion_preserves_exact_operation_order() {
        let projected = derive_prog_suite_development_program(&binding()).unwrap();
        let b = &projected.program.stages[0];
        assert_eq!(
            b.operations
                .iter()
                .map(|operation| operation.class.clone())
                .collect::<Vec<_>>(),
            vec![
                ThematicTransformationClassV1::Inversion,
                ThematicTransformationClassV1::Retrograde,
            ]
        );
        assert_eq!(
            b.operations
                .iter()
                .map(|operation| operation.operation_id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "prog-suite:development:B:op-01-inversion",
                "prog-suite:development:B:op-02-retrograde",
            ]
        );
        assert!(matches!(
            &b.graph_projection,
            DevelopmentGraphProjectionV1::OrderedComposite {
                label,
                operation_ids,
            } if label == "prog-suite:retrograde-inversion-v1"
                && operation_ids == &vec![
                    "prog-suite:development:B:op-01-inversion".to_string(),
                    "prog-suite:development:B:op-02-retrograde".to_string(),
                ]
        ));
        assert_eq!(
            b.evidence_requirement,
            DevelopmentEvidenceRequirementV1::OrderedCompositeRelationMeasured
        );
    }

    #[test]
    fn c_and_return_keep_lossless_single_operation_projections() {
        let projected = derive_prog_suite_development_program(&binding()).unwrap();
        let c = &projected.program.stages[1];
        assert_eq!(c.operations.len(), 1);
        assert_eq!(c.operations[0].class, ThematicTransformationClassV1::Inversion);
        assert_eq!(c.graph_projection, DevelopmentGraphProjectionV1::ExactSingleClass);

        let return_a = &projected.program.stages[2];
        assert_eq!(return_a.operations.len(), 1);
        assert_eq!(
            return_a.operations[0].class,
            ThematicTransformationClassV1::LiteralReturn
        );
        assert_eq!(
            return_a.graph_projection,
            DevelopmentGraphProjectionV1::ExactSingleClass
        );
    }

    #[test]
    fn return_stage_dependencies_are_derived_from_form003_promises() {
        let projected = derive_prog_suite_development_program(&binding()).unwrap();
        assert_eq!(
            projected.program.stages[0].prerequisite_stage_ids,
            Vec::<String>::new()
        );
        assert_eq!(
            projected.program.stages[1].prerequisite_stage_ids,
            Vec::<String>::new()
        );
        assert_eq!(
            projected.program.stages[2].prerequisite_stage_ids,
            vec![
                "prog-suite:development:B".to_string(),
                "prog-suite:development:C".to_string(),
            ]
        );
    }

    #[test]
    fn edited_frozen_plan_not_seed_drives_program_operations() {
        let mut original = binding();
        let source_seed = original.native_plan.source_seed;
        original.native_plan.sections[1].transformation = ProgSuiteTransformV1::Retrograde;
        original.native_plan.sections[2].transformation = ProgSuiteTransformV1::RetrogradeInversion;
        original.native_plan.validate().unwrap();
        let edited = bridge_prog_suite_plan(&original.native_plan).unwrap();
        assert_eq!(edited.native_plan.source_seed, source_seed);

        let projected = derive_prog_suite_development_program(&edited).unwrap();
        assert_eq!(
            projected.program.stages[0].operations[0].class,
            ThematicTransformationClassV1::Retrograde
        );
        assert_eq!(projected.program.stages[0].operations.len(), 1);
        assert_eq!(
            projected.program.stages[1]
                .operations
                .iter()
                .map(|operation| operation.class.clone())
                .collect::<Vec<_>>(),
            vec![
                ThematicTransformationClassV1::Inversion,
                ThematicTransformationClassV1::Retrograde,
            ]
        );
    }

    #[test]
    fn standalone_retrograde_is_a_lossless_single_class_stage() {
        let mut original = binding();
        original.native_plan.sections[1].transformation = ProgSuiteTransformV1::Retrograde;
        original.native_plan.sections[2].transformation = ProgSuiteTransformV1::Inversion;
        original.native_plan.validate().unwrap();
        let edited = bridge_prog_suite_plan(&original.native_plan).unwrap();
        let projected = derive_prog_suite_development_program(&edited).unwrap();
        let b = &projected.program.stages[0];
        assert_eq!(b.operations.len(), 1);
        assert_eq!(b.operations[0].class, ThematicTransformationClassV1::Retrograde);
        assert_eq!(b.graph_projection, DevelopmentGraphProjectionV1::ExactSingleClass);
        assert_eq!(
            b.evidence_requirement,
            DevelopmentEvidenceRequirementV1::SingleOperationRelationMeasured
        );
    }

    #[test]
    fn stale_or_forged_work_binding_is_rejected_before_program_authority() {
        let mut forged = binding();
        forged.thematic_graph.derivations.get_mut("prog-suite:derive-B").unwrap()
            .transformations = vec![ThematicTransformationClassV1::Retrograde];
        assert_eq!(
            derive_prog_suite_development_program(&forged),
            Err(ProgSuiteDevelopmentProgramErrorV1::SourceBindingMismatch)
        );
    }

    #[test]
    fn wrapper_binds_all_contract_versions() {
        let binding = binding();
        let projected = derive_prog_suite_development_program(&binding).unwrap();
        assert_eq!(
            projected.version,
            PROG_SUITE_DEVELOPMENT_PROGRAM_VERSION
        );
        assert_eq!(projected.development_program_version, DEVELOPMENT_PROGRAM_VERSION);
        assert_eq!(projected.source_work_bridge_version, PROG_SUITE_WORK_BRIDGE_VERSION);
        assert_eq!(projected.source_plan_version, binding.native_plan.version);
    }
}
