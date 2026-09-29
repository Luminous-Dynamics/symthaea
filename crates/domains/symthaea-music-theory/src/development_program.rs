// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ordered work-scale thematic development programs.
//!
//! FORM-002 deliberately stores a canonical *set* of transformation classes.
//! That is appropriate for genealogy, but not sufficient to represent an
//! ordered compositional process such as inversion -> retrograde. This module
//! adds a planner-level declaration without changing FORM-002: stages occur in
//! exact work spans, contain ordered operation identities/classes, reference the
//! FORM-002 derivation they project into, reference the FORM-003 promise that
//! makes the stage due, and state the symbolic evidence theorem required later.
//!
//! V1 is declaration only. It does not execute transformations and it does not
//! convert a valid plan into evidence authority.

use crate::rhythm::Duration;
use crate::thematic_identity::{
    ThematicGraphErrorV1, ThematicIdentityGraphV1, ThematicTransformationClassV1,
};
use crate::work_obligation::{
    WorkObligationErrorV2, WorkObligationKindV2, WorkObligationPlanV2,
};
use crate::work_plan::{HierarchicalWorkPlanV1, WorkNodeKindV1, WorkPlanErrorV1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const DEVELOPMENT_PROGRAM_VERSION: &str = "melothaea-development-program-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DevelopmentOperationV1 {
    /// Stable identity of this exact operation occurrence. Repeated classes are
    /// allowed; repeated operation IDs are not.
    pub operation_id: String,
    pub class: ThematicTransformationClassV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevelopmentGraphProjectionV1 {
    /// One ordered operation maps losslessly to the same single FORM-002 class.
    ExactSingleClass,
    /// FORM-002 V1 cannot retain operation order/multiplicity. The derivation
    /// must therefore contain one explicit custom label, while this receipt
    /// binds that label to the exact operation-ID sequence preserved here.
    OrderedComposite {
        label: String,
        operation_ids: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevelopmentEvidenceRequirementV1 {
    /// A completed-score measurement must establish the declared single
    /// transformation relation, not merely observe the target identity.
    SingleOperationRelationMeasured,
    /// A completed-score measurement must establish the exact ordered composite
    /// represented by this stage. Independent measurements of its component
    /// classes are insufficient.
    OrderedCompositeRelationMeasured,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DevelopmentStageV1 {
    pub stage_id: String,
    /// Exact work context containing this development episode.
    pub work_node_id: String,
    pub start: Duration,
    pub end: Duration,
    pub input_identity_id: String,
    pub output_identity_id: String,
    pub derivation_id: String,
    /// FORM-003 promise whose resolution is the evidence-bearing completion of
    /// this stage.
    pub obligation_id: String,
    /// Program-order dependencies. They are separate from thematic ancestry:
    /// two later episodes may both derive independently from the same P theme.
    pub prerequisite_stage_ids: Vec<String>,
    /// Exact execution order within this stage.
    pub operations: Vec<DevelopmentOperationV1>,
    pub graph_projection: DevelopmentGraphProjectionV1,
    pub evidence_requirement: DevelopmentEvidenceRequirementV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DevelopmentProgramV1 {
    pub version: String,
    pub program_id: String,
    /// Root thematic lineage this program is allowed to develop.
    pub source_identity_id: String,
    /// Canonical chronological stage order. V1 forbids overlap inside one
    /// program; parallel development belongs in separate programs.
    pub stages: Vec<DevelopmentStageV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DevelopmentProgramErrorV1 {
    WrongVersion { found: String },
    InvalidWorkPlan(WorkPlanErrorV1),
    InvalidThematicGraph(ThematicGraphErrorV1),
    InvalidObligationPlan(WorkObligationErrorV2),
    EmptyProgramId,
    MissingProgramSourceIdentity { identity_id: String },
    EmptyStages,
    EmptyStageId { stage_index: usize },
    DuplicateStageId { stage_id: String },
    EmptyOperationId { stage_id: String, operation_index: usize },
    DuplicateOperationId { operation_id: String },
    EmptyOperations { stage_id: String },
    EmptyCustomOperationClass { stage_id: String, operation_id: String },
    MissingWorkNode { stage_id: String, work_node_id: String },
    StageTargetsWorkRoot { stage_id: String },
    InvalidStageSpan { stage_id: String },
    StageOutsideWorkNode { stage_id: String, work_node_id: String },
    StageOrderOverlap {
        earlier_stage_id: String,
        later_stage_id: String,
    },
    MissingInputIdentity { stage_id: String, identity_id: String },
    MissingOutputIdentity { stage_id: String, identity_id: String },
    IdentityOutsideProgramLineage { stage_id: String, identity_id: String },
    OutputIdentityIntroducedElsewhere {
        stage_id: String,
        identity_id: String,
        introduced_in: String,
    },
    MissingDerivation { stage_id: String, derivation_id: String },
    DerivationEndpointsMismatch { stage_id: String, derivation_id: String },
    MissingObligation { stage_id: String, obligation_id: String },
    ObligationKindMismatch { stage_id: String, obligation_id: String },
    ObligationDueContextMismatch { stage_id: String, obligation_id: String },
    ObligationResolutionOutsideDueWindow { stage_id: String, obligation_id: String },
    ObligationCreatedAfterStageStart { stage_id: String, obligation_id: String },
    UnknownPrerequisiteStage { stage_id: String, prerequisite_stage_id: String },
    DuplicatePrerequisiteStage { stage_id: String, prerequisite_stage_id: String },
    MissingObligationPrerequisite {
        stage_id: String,
        prerequisite_stage_id: String,
        prerequisite_obligation_id: String,
    },
    MissingThematicPrerequisite {
        stage_id: String,
        producer_stage_id: String,
    },
    ExactProjectionRequiresSingleOperation { stage_id: String, found: usize },
    CompositeProjectionRequiresMultipleOperations { stage_id: String, found: usize },
    EmptyCompositeLabel { stage_id: String },
    CompositeOperationOrderMismatch { stage_id: String },
    GraphProjectionMismatch { stage_id: String, derivation_id: String },
    EvidenceRequirementMismatch { stage_id: String },
}

impl DevelopmentProgramV1 {
    pub fn validate(
        &self,
        work_plan: &HierarchicalWorkPlanV1,
        thematic_graph: &ThematicIdentityGraphV1,
        obligation_plan: &WorkObligationPlanV2,
    ) -> Result<(), DevelopmentProgramErrorV1> {
        if self.version != DEVELOPMENT_PROGRAM_VERSION {
            return Err(DevelopmentProgramErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.program_id.trim().is_empty() {
            return Err(DevelopmentProgramErrorV1::EmptyProgramId);
        }

        // Preserve the original FORM-layer failure rather than disguising it as
        // a DevelopmentProgram-local missing reference.
        work_plan
            .validate()
            .map_err(DevelopmentProgramErrorV1::InvalidWorkPlan)?;
        thematic_graph
            .validate(work_plan)
            .map_err(DevelopmentProgramErrorV1::InvalidThematicGraph)?;
        obligation_plan
            .validate(work_plan, thematic_graph)
            .map_err(DevelopmentProgramErrorV1::InvalidObligationPlan)?;

        if !thematic_graph.identities.contains_key(&self.source_identity_id) {
            return Err(DevelopmentProgramErrorV1::MissingProgramSourceIdentity {
                identity_id: self.source_identity_id.clone(),
            });
        }
        if self.stages.is_empty() {
            return Err(DevelopmentProgramErrorV1::EmptyStages);
        }

        let mut stage_ids = BTreeSet::new();
        let mut operation_ids = BTreeSet::new();
        let mut seen_stages: BTreeMap<String, (&DevelopmentStageV1, String)> = BTreeMap::new();
        let mut output_producers: BTreeMap<String, String> = BTreeMap::new();
        let mut previous: Option<&DevelopmentStageV1> = None;

        for (stage_index, stage) in self.stages.iter().enumerate() {
            if stage.stage_id.trim().is_empty() {
                return Err(DevelopmentProgramErrorV1::EmptyStageId { stage_index });
            }
            if !stage_ids.insert(stage.stage_id.clone()) {
                return Err(DevelopmentProgramErrorV1::DuplicateStageId {
                    stage_id: stage.stage_id.clone(),
                });
            }
            if stage.operations.is_empty() {
                return Err(DevelopmentProgramErrorV1::EmptyOperations {
                    stage_id: stage.stage_id.clone(),
                });
            }
            for (operation_index, operation) in stage.operations.iter().enumerate() {
                if operation.operation_id.trim().is_empty() {
                    return Err(DevelopmentProgramErrorV1::EmptyOperationId {
                        stage_id: stage.stage_id.clone(),
                        operation_index,
                    });
                }
                if !operation_ids.insert(operation.operation_id.clone()) {
                    return Err(DevelopmentProgramErrorV1::DuplicateOperationId {
                        operation_id: operation.operation_id.clone(),
                    });
                }
                if let ThematicTransformationClassV1::Other(label) = &operation.class
                    && label.trim().is_empty()
                {
                    return Err(DevelopmentProgramErrorV1::EmptyCustomOperationClass {
                        stage_id: stage.stage_id.clone(),
                        operation_id: operation.operation_id.clone(),
                    });
                }
            }

            let node = work_plan.nodes.get(&stage.work_node_id).ok_or_else(|| {
                DevelopmentProgramErrorV1::MissingWorkNode {
                    stage_id: stage.stage_id.clone(),
                    work_node_id: stage.work_node_id.clone(),
                }
            })?;
            if node.kind == WorkNodeKindV1::Work {
                return Err(DevelopmentProgramErrorV1::StageTargetsWorkRoot {
                    stage_id: stage.stage_id.clone(),
                });
            }
            if compare_duration(stage.start, stage.end) != Ordering::Less {
                return Err(DevelopmentProgramErrorV1::InvalidStageSpan {
                    stage_id: stage.stage_id.clone(),
                });
            }
            if compare_duration(stage.start, node.start) == Ordering::Less
                || compare_duration(stage.end, node.end) == Ordering::Greater
            {
                return Err(DevelopmentProgramErrorV1::StageOutsideWorkNode {
                    stage_id: stage.stage_id.clone(),
                    work_node_id: stage.work_node_id.clone(),
                });
            }
            if let Some(previous) = previous
                && compare_duration(stage.start, previous.end) == Ordering::Less
            {
                return Err(DevelopmentProgramErrorV1::StageOrderOverlap {
                    earlier_stage_id: previous.stage_id.clone(),
                    later_stage_id: stage.stage_id.clone(),
                });
            }

            for (identity_id, input) in [
                (&stage.input_identity_id, true),
                (&stage.output_identity_id, false),
            ] {
                if !thematic_graph.identities.contains_key(identity_id) {
                    return Err(if input {
                        DevelopmentProgramErrorV1::MissingInputIdentity {
                            stage_id: stage.stage_id.clone(),
                            identity_id: identity_id.clone(),
                        }
                    } else {
                        DevelopmentProgramErrorV1::MissingOutputIdentity {
                            stage_id: stage.stage_id.clone(),
                            identity_id: identity_id.clone(),
                        }
                    });
                }
                if !identity_descends_from(thematic_graph, &self.source_identity_id, identity_id) {
                    return Err(DevelopmentProgramErrorV1::IdentityOutsideProgramLineage {
                        stage_id: stage.stage_id.clone(),
                        identity_id: identity_id.clone(),
                    });
                }
            }

            let output = &thematic_graph.identities[&stage.output_identity_id];
            if output.introduced_in != stage.work_node_id {
                return Err(DevelopmentProgramErrorV1::OutputIdentityIntroducedElsewhere {
                    stage_id: stage.stage_id.clone(),
                    identity_id: stage.output_identity_id.clone(),
                    introduced_in: output.introduced_in.clone(),
                });
            }

            let derivation = thematic_graph
                .derivations
                .get(&stage.derivation_id)
                .ok_or_else(|| DevelopmentProgramErrorV1::MissingDerivation {
                    stage_id: stage.stage_id.clone(),
                    derivation_id: stage.derivation_id.clone(),
                })?;
            if derivation.source_id != stage.input_identity_id
                || derivation.target_id != stage.output_identity_id
            {
                return Err(DevelopmentProgramErrorV1::DerivationEndpointsMismatch {
                    stage_id: stage.stage_id.clone(),
                    derivation_id: stage.derivation_id.clone(),
                });
            }

            validate_projection(stage, &derivation.transformations)?;

            let obligation = obligation_plan
                .obligations
                .get(&stage.obligation_id)
                .ok_or_else(|| DevelopmentProgramErrorV1::MissingObligation {
                    stage_id: stage.stage_id.clone(),
                    obligation_id: stage.obligation_id.clone(),
                })?;
            if !matches!(
                &obligation.kind,
                WorkObligationKindV2::RealizeThematicDerivation { derivation_id }
                    if derivation_id == &stage.derivation_id
            ) {
                return Err(DevelopmentProgramErrorV1::ObligationKindMismatch {
                    stage_id: stage.stage_id.clone(),
                    obligation_id: stage.obligation_id.clone(),
                });
            }
            if obligation.due_context != stage.work_node_id {
                return Err(DevelopmentProgramErrorV1::ObligationDueContextMismatch {
                    stage_id: stage.stage_id.clone(),
                    obligation_id: stage.obligation_id.clone(),
                });
            }
            if compare_duration(stage.end, obligation.due.earliest) == Ordering::Less
                || compare_duration(stage.end, obligation.due.latest) == Ordering::Greater
            {
                return Err(
                    DevelopmentProgramErrorV1::ObligationResolutionOutsideDueWindow {
                        stage_id: stage.stage_id.clone(),
                        obligation_id: stage.obligation_id.clone(),
                    },
                );
            }
            if compare_duration(obligation.created_at, stage.start) == Ordering::Greater {
                return Err(DevelopmentProgramErrorV1::ObligationCreatedAfterStageStart {
                    stage_id: stage.stage_id.clone(),
                    obligation_id: stage.obligation_id.clone(),
                });
            }

            let mut prerequisites = BTreeSet::new();
            for prerequisite_stage_id in &stage.prerequisite_stage_ids {
                if !prerequisites.insert(prerequisite_stage_id.clone()) {
                    return Err(DevelopmentProgramErrorV1::DuplicatePrerequisiteStage {
                        stage_id: stage.stage_id.clone(),
                        prerequisite_stage_id: prerequisite_stage_id.clone(),
                    });
                }
                let Some((_, prerequisite_obligation_id)) =
                    seen_stages.get(prerequisite_stage_id)
                else {
                    return Err(DevelopmentProgramErrorV1::UnknownPrerequisiteStage {
                        stage_id: stage.stage_id.clone(),
                        prerequisite_stage_id: prerequisite_stage_id.clone(),
                    });
                };
                if !obligation.prerequisites.contains(prerequisite_obligation_id) {
                    return Err(DevelopmentProgramErrorV1::MissingObligationPrerequisite {
                        stage_id: stage.stage_id.clone(),
                        prerequisite_stage_id: prerequisite_stage_id.clone(),
                        prerequisite_obligation_id: prerequisite_obligation_id.clone(),
                    });
                }
            }

            // If a stage consumes an identity produced by an earlier stage in
            // this same program, temporal/program dependency must be explicit.
            if let Some(producer_stage_id) = output_producers.get(&stage.input_identity_id)
                && !prerequisites.contains(producer_stage_id)
            {
                return Err(DevelopmentProgramErrorV1::MissingThematicPrerequisite {
                    stage_id: stage.stage_id.clone(),
                    producer_stage_id: producer_stage_id.clone(),
                });
            }

            seen_stages.insert(
                stage.stage_id.clone(),
                (stage, stage.obligation_id.clone()),
            );
            output_producers.insert(stage.output_identity_id.clone(), stage.stage_id.clone());
            previous = Some(stage);
        }

        Ok(())
    }
}

fn validate_projection(
    stage: &DevelopmentStageV1,
    graph_transformations: &[ThematicTransformationClassV1],
) -> Result<(), DevelopmentProgramErrorV1> {
    match &stage.graph_projection {
        DevelopmentGraphProjectionV1::ExactSingleClass => {
            if stage.operations.len() != 1 {
                return Err(
                    DevelopmentProgramErrorV1::ExactProjectionRequiresSingleOperation {
                        stage_id: stage.stage_id.clone(),
                        found: stage.operations.len(),
                    },
                );
            }
            if graph_transformations.len() != 1
                || graph_transformations.first() != Some(&stage.operations[0].class)
            {
                return Err(DevelopmentProgramErrorV1::GraphProjectionMismatch {
                    stage_id: stage.stage_id.clone(),
                    derivation_id: stage.derivation_id.clone(),
                });
            }
            if stage.evidence_requirement
                != DevelopmentEvidenceRequirementV1::SingleOperationRelationMeasured
            {
                return Err(DevelopmentProgramErrorV1::EvidenceRequirementMismatch {
                    stage_id: stage.stage_id.clone(),
                });
            }
        }
        DevelopmentGraphProjectionV1::OrderedComposite {
            label,
            operation_ids,
        } => {
            if stage.operations.len() < 2 {
                return Err(
                    DevelopmentProgramErrorV1::CompositeProjectionRequiresMultipleOperations {
                        stage_id: stage.stage_id.clone(),
                        found: stage.operations.len(),
                    },
                );
            }
            if label.trim().is_empty() {
                return Err(DevelopmentProgramErrorV1::EmptyCompositeLabel {
                    stage_id: stage.stage_id.clone(),
                });
            }
            let actual_ids: Vec<_> = stage
                .operations
                .iter()
                .map(|operation| operation.operation_id.clone())
                .collect();
            if operation_ids != &actual_ids {
                return Err(DevelopmentProgramErrorV1::CompositeOperationOrderMismatch {
                    stage_id: stage.stage_id.clone(),
                });
            }
            let expected = ThematicTransformationClassV1::Other(label.clone());
            if graph_transformations.len() != 1
                || graph_transformations.first() != Some(&expected)
            {
                return Err(DevelopmentProgramErrorV1::GraphProjectionMismatch {
                    stage_id: stage.stage_id.clone(),
                    derivation_id: stage.derivation_id.clone(),
                });
            }
            if stage.evidence_requirement
                != DevelopmentEvidenceRequirementV1::OrderedCompositeRelationMeasured
            {
                return Err(DevelopmentProgramErrorV1::EvidenceRequirementMismatch {
                    stage_id: stage.stage_id.clone(),
                });
            }
        }
    }
    Ok(())
}

fn identity_descends_from(
    graph: &ThematicIdentityGraphV1,
    root_identity_id: &str,
    candidate_identity_id: &str,
) -> bool {
    if candidate_identity_id == root_identity_id {
        return true;
    }
    let mut current = candidate_identity_id;
    let mut visited = BTreeSet::new();
    while visited.insert(current.to_string()) {
        let Some(parent) = graph
            .derivations
            .values()
            .find(|derivation| derivation.target_id == current)
            .map(|derivation| derivation.source_id.as_str())
        else {
            return false;
        };
        if parent == root_identity_id {
            return true;
        }
        current = parent;
    }
    false
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FormalFunctionV1, ObligationDueWindowV2, ThematicDerivationV1, ThematicIdentityV1,
        ThematicOriginV1, WorkNodeV1, WorkObligationV2,
    };

    fn work_plan() -> HierarchicalWorkPlanV1 {
        let mut work = HierarchicalWorkPlanV1::new("work", Duration::new(16, 1)).unwrap();
        for (id, start, end, function) in [
            ("A", 0, 4, FormalFunctionV1::Establish),
            ("B", 4, 10, FormalFunctionV1::Develop),
            ("C", 10, 16, FormalFunctionV1::Develop),
        ] {
            work.insert_node(
                id,
                WorkNodeV1 {
                    parent_id: Some("work".into()),
                    label: None,
                    kind: WorkNodeKindV1::Section,
                    start: Duration::new(start, 1),
                    end: Duration::new(end, 1),
                    functions: vec![function],
                },
            )
            .unwrap();
        }
        work
    }

    fn graph() -> ThematicIdentityGraphV1 {
        let mut graph = ThematicIdentityGraphV1::default();
        for (id, origin, node) in [
            ("P", ThematicOriginV1::Independent, "A"),
            ("P-frag", ThematicOriginV1::Derived, "B"),
            ("P-inv", ThematicOriginV1::Derived, "C"),
        ] {
            graph
                .insert_identity(
                    id,
                    ThematicIdentityV1 {
                        label: None,
                        origin,
                        introduced_in: node.into(),
                    },
                )
                .unwrap();
        }
        graph
            .insert_derivation(
                "derive-frag",
                ThematicDerivationV1 {
                    source_id: "P".into(),
                    target_id: "P-frag".into(),
                    transformations: vec![ThematicTransformationClassV1::Fragmentation],
                },
            )
            .unwrap();
        graph
            .insert_derivation(
                "derive-inv",
                ThematicDerivationV1 {
                    source_id: "P-frag".into(),
                    target_id: "P-inv".into(),
                    transformations: vec![ThematicTransformationClassV1::Inversion],
                },
            )
            .unwrap();
        graph
    }

    fn obligation(
        due_context: &str,
        earliest: i64,
        latest: i64,
        derivation_id: &str,
        prerequisites: &[&str],
    ) -> WorkObligationV2 {
        WorkObligationV2 {
            declared_in: "A".into(),
            created_at: Duration::zero(),
            due_context: due_context.into(),
            due: ObligationDueWindowV2 {
                earliest: Duration::new(earliest, 1),
                latest: Duration::new(latest, 1),
            },
            priority_per_mille: 1000,
            prerequisites: prerequisites.iter().map(|id| (*id).into()).collect(),
            conflicts_with: Vec::new(),
            kind: WorkObligationKindV2::RealizeThematicDerivation {
                derivation_id: derivation_id.into(),
            },
        }
    }

    fn obligations() -> WorkObligationPlanV2 {
        let mut plan = WorkObligationPlanV2::default();
        plan.insert("ob-frag", obligation("B", 4, 10, "derive-frag", &[]))
            .unwrap();
        plan.insert(
            "ob-inv",
            obligation("C", 10, 16, "derive-inv", &["ob-frag"]),
        )
        .unwrap();
        plan
    }

    fn program() -> DevelopmentProgramV1 {
        DevelopmentProgramV1 {
            version: DEVELOPMENT_PROGRAM_VERSION.into(),
            program_id: "development:P".into(),
            source_identity_id: "P".into(),
            stages: vec![
                DevelopmentStageV1 {
                    stage_id: "stage-frag".into(),
                    work_node_id: "B".into(),
                    start: Duration::new(4, 1),
                    end: Duration::new(10, 1),
                    input_identity_id: "P".into(),
                    output_identity_id: "P-frag".into(),
                    derivation_id: "derive-frag".into(),
                    obligation_id: "ob-frag".into(),
                    prerequisite_stage_ids: Vec::new(),
                    operations: vec![DevelopmentOperationV1 {
                        operation_id: "op-frag".into(),
                        class: ThematicTransformationClassV1::Fragmentation,
                    }],
                    graph_projection: DevelopmentGraphProjectionV1::ExactSingleClass,
                    evidence_requirement:
                        DevelopmentEvidenceRequirementV1::SingleOperationRelationMeasured,
                },
                DevelopmentStageV1 {
                    stage_id: "stage-inv".into(),
                    work_node_id: "C".into(),
                    start: Duration::new(10, 1),
                    end: Duration::new(16, 1),
                    input_identity_id: "P-frag".into(),
                    output_identity_id: "P-inv".into(),
                    derivation_id: "derive-inv".into(),
                    obligation_id: "ob-inv".into(),
                    prerequisite_stage_ids: vec!["stage-frag".into()],
                    operations: vec![DevelopmentOperationV1 {
                        operation_id: "op-inv".into(),
                        class: ThematicTransformationClassV1::Inversion,
                    }],
                    graph_projection: DevelopmentGraphProjectionV1::ExactSingleClass,
                    evidence_requirement:
                        DevelopmentEvidenceRequirementV1::SingleOperationRelationMeasured,
                },
            ],
        }
    }

    #[test]
    fn ordered_linear_development_program_validates_against_all_form_contracts() {
        program()
            .validate(&work_plan(), &graph(), &obligations())
            .unwrap();
    }

    #[test]
    fn malformed_form_contract_error_is_preserved_exactly() {
        let mut work = work_plan();
        work.version = "wrong-form-version".into();
        assert!(matches!(
            program().validate(&work, &graph(), &obligations()),
            Err(DevelopmentProgramErrorV1::InvalidWorkPlan(
                WorkPlanErrorV1::WrongVersion { .. }
            ))
        ));
    }

    #[test]
    fn consuming_an_intermediate_identity_requires_explicit_stage_and_obligation_dependency() {
        let mut program = program();
        program.stages[1].prerequisite_stage_ids.clear();
        assert!(matches!(
            program.validate(&work_plan(), &graph(), &obligations()),
            Err(DevelopmentProgramErrorV1::MissingThematicPrerequisite { .. })
        ));
    }

    #[test]
    fn stage_span_must_live_inside_its_declared_work_context() {
        let mut program = program();
        program.stages[0].start = Duration::new(3, 1);
        assert!(matches!(
            program.validate(&work_plan(), &graph(), &obligations()),
            Err(DevelopmentProgramErrorV1::StageOutsideWorkNode { .. })
        ));
    }

    #[test]
    fn overlapping_stages_are_rejected_in_one_v1_program() {
        let mut program = program();
        program.stages[1].start = Duration::new(9, 1);
        assert!(matches!(
            program.validate(&work_plan(), &graph(), &obligations()),
            Err(DevelopmentProgramErrorV1::StageOrderOverlap { .. })
        ));
    }

    #[test]
    fn composite_projection_binds_exact_operation_order_instead_of_a_class_set() {
        let work = work_plan();
        let mut graph = graph();
        graph
            .insert_identity(
                "P-ri",
                ThematicIdentityV1 {
                    label: None,
                    origin: ThematicOriginV1::Derived,
                    introduced_in: "B".into(),
                },
            )
            .unwrap();
        graph
            .insert_derivation(
                "derive-ri",
                ThematicDerivationV1 {
                    source_id: "P".into(),
                    target_id: "P-ri".into(),
                    transformations: vec![ThematicTransformationClassV1::Other(
                        "invert-then-retrograde".into(),
                    )],
                },
            )
            .unwrap();

        let mut obligations = WorkObligationPlanV2::default();
        obligations
            .insert("ob-ri", obligation("B", 4, 10, "derive-ri", &[]))
            .unwrap();

        let composite = DevelopmentProgramV1 {
            version: DEVELOPMENT_PROGRAM_VERSION.into(),
            program_id: "development:ri".into(),
            source_identity_id: "P".into(),
            stages: vec![DevelopmentStageV1 {
                stage_id: "stage-ri".into(),
                work_node_id: "B".into(),
                start: Duration::new(4, 1),
                end: Duration::new(10, 1),
                input_identity_id: "P".into(),
                output_identity_id: "P-ri".into(),
                derivation_id: "derive-ri".into(),
                obligation_id: "ob-ri".into(),
                prerequisite_stage_ids: Vec::new(),
                operations: vec![
                    DevelopmentOperationV1 {
                        operation_id: "invert".into(),
                        class: ThematicTransformationClassV1::Inversion,
                    },
                    DevelopmentOperationV1 {
                        operation_id: "retrograde".into(),
                        class: ThematicTransformationClassV1::Retrograde,
                    },
                ],
                graph_projection: DevelopmentGraphProjectionV1::OrderedComposite {
                    label: "invert-then-retrograde".into(),
                    operation_ids: vec!["invert".into(), "retrograde".into()],
                },
                evidence_requirement:
                    DevelopmentEvidenceRequirementV1::OrderedCompositeRelationMeasured,
            }],
        };
        composite.validate(&work, &graph, &obligations).unwrap();

        let mut wrong_receipt = composite.clone();
        if let DevelopmentGraphProjectionV1::OrderedComposite { operation_ids, .. } =
            &mut wrong_receipt.stages[0].graph_projection
        {
            operation_ids.reverse();
        }
        assert!(matches!(
            wrong_receipt.validate(&work, &graph, &obligations),
            Err(DevelopmentProgramErrorV1::CompositeOperationOrderMismatch { .. })
        ));
    }

    #[test]
    fn unordered_form_class_set_cannot_impersonate_an_ordered_composite() {
        let work = work_plan();
        let mut graph = graph();
        graph
            .insert_identity(
                "P-ri",
                ThematicIdentityV1 {
                    label: None,
                    origin: ThematicOriginV1::Derived,
                    introduced_in: "B".into(),
                },
            )
            .unwrap();
        graph
            .insert_derivation(
                "derive-ri",
                ThematicDerivationV1 {
                    source_id: "P".into(),
                    target_id: "P-ri".into(),
                    transformations: vec![
                        ThematicTransformationClassV1::Inversion,
                        ThematicTransformationClassV1::Retrograde,
                    ],
                },
            )
            .unwrap();
        let mut obligations = WorkObligationPlanV2::default();
        obligations
            .insert("ob-ri", obligation("B", 4, 10, "derive-ri", &[]))
            .unwrap();
        let mut composite = program();
        composite.program_id = "development:bad-composite".into();
        composite.stages.truncate(1);
        let stage = &mut composite.stages[0];
        stage.stage_id = "stage-ri".into();
        stage.output_identity_id = "P-ri".into();
        stage.derivation_id = "derive-ri".into();
        stage.obligation_id = "ob-ri".into();
        stage.operations = vec![
            DevelopmentOperationV1 {
                operation_id: "invert".into(),
                class: ThematicTransformationClassV1::Inversion,
            },
            DevelopmentOperationV1 {
                operation_id: "retrograde".into(),
                class: ThematicTransformationClassV1::Retrograde,
            },
        ];
        stage.graph_projection = DevelopmentGraphProjectionV1::OrderedComposite {
            label: "invert-then-retrograde".into(),
            operation_ids: vec!["invert".into(), "retrograde".into()],
        };
        stage.evidence_requirement =
            DevelopmentEvidenceRequirementV1::OrderedCompositeRelationMeasured;
        assert!(matches!(
            composite.validate(&work, &graph, &obligations),
            Err(DevelopmentProgramErrorV1::GraphProjectionMismatch { .. })
        ));
    }

    #[test]
    fn evidence_requirement_must_match_projection_authority() {
        let mut program = program();
        program.stages[0].evidence_requirement =
            DevelopmentEvidenceRequirementV1::OrderedCompositeRelationMeasured;
        assert!(matches!(
            program.validate(&work_plan(), &graph(), &obligations()),
            Err(DevelopmentProgramErrorV1::EvidenceRequirementMismatch { .. })
        ));
    }
}
