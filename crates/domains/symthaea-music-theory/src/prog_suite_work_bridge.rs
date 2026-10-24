// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native ProgSuite -> work-scale FORM authority bridge.
//!
//! ProgSuite has its own first-class plan/realization boundary. This adapter
//! preserves that native plan while translating its exact mixed-meter timeline,
//! section hierarchy, thematic genealogy, and prospective promises into the
//! generic FORM-000/001/002/003 contracts. It does not inspect a completed score
//! to manufacture declaration state.
//!
//! One projection loss is explicit: native ProgSuite stores only quarter-note
//! beats per bar, while FORM-000 `TimeSignature` also requires grouping. The
//! bridge therefore uses the compatibility quarter-note grouping and records
//! that the grouping itself was not source-authored.
//!
//! FORM-002 V1 likewise stores an unordered canonical set of transformation
//! classes, so it cannot faithfully encode the ordered native composition
//! `invert().retrograde()`. That composite remains one explicit `Other` class
//! rather than being misrepresented as two independent source->target claims.

use crate::prog_suite::{
    ProgSuitePlanErrorV1, ProgSuitePlanV1, ProgSuiteRealizationV1, ProgSuiteTransformV1,
};
use crate::temporal_map::{TempoV1, TemporalMapErrorV1, TemporalMapV1};
use crate::temporal_score::{TemporalScoreErrorV1, TemporalScoreV1};
use crate::thematic_identity::{
    ThematicDerivationV1, ThematicGraphErrorV1, ThematicIdentityGraphV1,
    ThematicIdentityV1, ThematicOriginV1, ThematicTransformationClassV1,
};
use crate::work_obligation::{
    ObligationDueWindowV2, WorkObligationErrorV2, WorkObligationKindV2,
    WorkObligationPlanV2, WorkObligationV2,
};
use crate::work_plan::{
    FormalFunctionV1, HierarchicalWorkPlanV1, WorkNodeKindV1, WorkNodeV1,
    WorkPlanErrorV1,
};
use serde::{Deserialize, Serialize};

pub const PROG_SUITE_WORK_BRIDGE_VERSION: &str = "melothaea-prog-suite-work-bridge-v1";

const ROOT_ID: &str = "prog-suite:work";
const NODE_A: &str = "prog-suite:A";
const NODE_B: &str = "prog-suite:B";
const NODE_C: &str = "prog-suite:C";
const NODE_RETURN_A: &str = "prog-suite:ReturnA";

const IDENTITY_P: &str = "prog-suite:P";
const IDENTITY_B: &str = "prog-suite:B-material";
const IDENTITY_C: &str = "prog-suite:C-material";
const IDENTITY_RETURN: &str = "prog-suite:P-return";

const DERIVE_B: &str = "prog-suite:derive-B";
const DERIVE_C: &str = "prog-suite:derive-C";
const RETURN_A: &str = "prog-suite:return-A";
const RETROGRADE_INVERSION_LABEL: &str = "prog-suite:retrograde-inversion-v1";

const OBLIGATION_ESTABLISH: &str = "prog-suite:establish-primary";
const OBLIGATION_TRANSFORM_B: &str = "prog-suite:transform-b";
const OBLIGATION_TRANSFORM_C: &str = "prog-suite:transform-c";
const OBLIGATION_REACH_RELATIVE: &str = "prog-suite:reach-relative";
const OBLIGATION_RETURN_PRIMARY: &str = "prog-suite:return-primary";
const OBLIGATION_RETURN_HOME: &str = "prog-suite:return-home";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteMeterProjectionV1 {
    /// Native source knows only N quarter-note beats per bar. FORM-000 uses
    /// `TimeSignature::quarter_note_meter(N)` for compatibility; no claim is
    /// made that the resulting grouping was explicitly composed upstream.
    LegacyQuarterNoteBeatCount { beats_per_bar: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteSectionWorkBindingV1 {
    pub section_index: usize,
    pub work_node_id: String,
    pub thematic_identity_id: String,
    pub derivation_id: Option<String>,
    pub meter_projection: ProgSuiteMeterProjectionV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteWorkBindingV1 {
    pub version: String,
    /// Retained native declaration. FORM artifacts are translations of this
    /// exact plan, not replacements for it.
    pub native_plan: ProgSuitePlanV1,
    pub temporal_map: TemporalMapV1,
    pub work_plan: HierarchicalWorkPlanV1,
    pub thematic_graph: ThematicIdentityGraphV1,
    pub obligation_plan: WorkObligationPlanV2,
    pub section_bindings: Vec<ProgSuiteSectionWorkBindingV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteWorkRealizationBindingV1 {
    pub declaration: ProgSuiteWorkBindingV1,
    pub temporal_score: TemporalScoreV1,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteWorkBridgeErrorV1 {
    NativePlan(ProgSuitePlanErrorV1),
    TemporalMap(TemporalMapErrorV1),
    WorkPlan(WorkPlanErrorV1),
    ThematicGraph(ThematicGraphErrorV1),
    ObligationPlan(WorkObligationErrorV2),
    TemporalScore(TemporalScoreErrorV1),
}

/// Translate one already-frozen native ProgSuite plan into generic work-scale
/// authority. No score evidence is consumed here.
pub fn bridge_prog_suite_plan(
    plan: &ProgSuitePlanV1,
) -> Result<ProgSuiteWorkBindingV1, ProgSuiteWorkBridgeErrorV1> {
    plan.validate()
        .map_err(ProgSuiteWorkBridgeErrorV1::NativePlan)?;

    let temporal_map = build_temporal_map(plan)?;
    let work_plan = build_work_plan(plan)?;
    let thematic_graph = build_thematic_graph(plan, &work_plan)?;
    let obligation_plan = build_obligation_plan(plan, &work_plan, &thematic_graph)?;

    let section_bindings = vec![
        section_binding(0, NODE_A, IDENTITY_P, None, plan.sections[0].meter),
        section_binding(
            1,
            NODE_B,
            IDENTITY_B,
            Some(DERIVE_B),
            plan.sections[1].meter,
        ),
        section_binding(
            2,
            NODE_C,
            IDENTITY_C,
            Some(DERIVE_C),
            plan.sections[2].meter,
        ),
        section_binding(
            3,
            NODE_RETURN_A,
            IDENTITY_RETURN,
            Some(RETURN_A),
            plan.sections[3].meter,
        ),
    ];

    Ok(ProgSuiteWorkBindingV1 {
        version: PROG_SUITE_WORK_BRIDGE_VERSION.into(),
        native_plan: plan.clone(),
        temporal_map,
        work_plan,
        thematic_graph,
        obligation_plan,
        section_bindings,
    })
}

/// Bind a completed native realization to the translated FORM-000 timeline and
/// FORM-001 work span. This is still structural binding, not thematic evidence.
pub fn bind_prog_suite_realization(
    realization: &ProgSuiteRealizationV1,
) -> Result<ProgSuiteWorkRealizationBindingV1, ProgSuiteWorkBridgeErrorV1> {
    let declaration = bridge_prog_suite_plan(&realization.plan)?;
    let temporal_score = TemporalScoreV1::bind(
        realization.score.clone(),
        declaration.temporal_map.clone(),
    )
    .map_err(ProgSuiteWorkBridgeErrorV1::TemporalScore)?;
    declaration
        .work_plan
        .validate_for_temporal_score(&temporal_score)
        .map_err(ProgSuiteWorkBridgeErrorV1::WorkPlan)?;

    Ok(ProgSuiteWorkRealizationBindingV1 {
        declaration,
        temporal_score,
    })
}

fn build_temporal_map(
    plan: &ProgSuitePlanV1,
) -> Result<TemporalMapV1, ProgSuiteWorkBridgeErrorV1> {
    let tempo = TempoV1::from_f32_exact(plan.tempo_bpm)
        .map_err(ProgSuiteWorkBridgeErrorV1::TemporalMap)?;
    let opening_meter = crate::meter::TimeSignature::quarter_note_meter(plan.sections[0].meter);
    let mut map = TemporalMapV1::new(tempo, opening_meter);

    for section in plan.sections.iter().skip(1) {
        map.push_change(
            section.start,
            Some(crate::meter::TimeSignature::quarter_note_meter(section.meter)),
            None,
        )
        .map_err(ProgSuiteWorkBridgeErrorV1::TemporalMap)?;
    }
    map.validate()
        .map_err(ProgSuiteWorkBridgeErrorV1::TemporalMap)?;
    Ok(map)
}

fn build_work_plan(
    plan: &ProgSuitePlanV1,
) -> Result<HierarchicalWorkPlanV1, ProgSuiteWorkBridgeErrorV1> {
    let mut work = HierarchicalWorkPlanV1::new(ROOT_ID, plan.total_beats)
        .map_err(ProgSuiteWorkBridgeErrorV1::WorkPlan)?;

    let specs = [
        (
            NODE_A,
            "A — primary statement",
            vec![FormalFunctionV1::Establish],
        ),
        (
            NODE_B,
            "B — first transformed contrast",
            vec![FormalFunctionV1::Contrast, FormalFunctionV1::Destabilize],
        ),
        (
            NODE_C,
            "C — second transformation / relative-key departure",
            vec![FormalFunctionV1::Develop, FormalFunctionV1::Destabilize],
        ),
        (
            NODE_RETURN_A,
            "Return A — primary restoration",
            vec![
                FormalFunctionV1::Return,
                FormalFunctionV1::Resolve,
                FormalFunctionV1::Close,
            ],
        ),
    ];

    for (index, (node_id, label, functions)) in specs.into_iter().enumerate() {
        let section = &plan.sections[index];
        work.insert_node(
            node_id,
            WorkNodeV1 {
                parent_id: Some(ROOT_ID.into()),
                label: Some(label.into()),
                kind: WorkNodeKindV1::Section,
                start: section.start,
                end: section.end,
                functions,
            },
        )
        .map_err(ProgSuiteWorkBridgeErrorV1::WorkPlan)?;
    }
    work.validate()
        .map_err(ProgSuiteWorkBridgeErrorV1::WorkPlan)?;
    Ok(work)
}

fn build_thematic_graph(
    plan: &ProgSuitePlanV1,
    work_plan: &HierarchicalWorkPlanV1,
) -> Result<ThematicIdentityGraphV1, ProgSuiteWorkBridgeErrorV1> {
    let mut graph = ThematicIdentityGraphV1::default();
    for (id, label, origin, node) in [
        (IDENTITY_P, "P", ThematicOriginV1::Independent, NODE_A),
        (IDENTITY_B, "P/B", ThematicOriginV1::Derived, NODE_B),
        (IDENTITY_C, "P/C", ThematicOriginV1::Derived, NODE_C),
        (
            IDENTITY_RETURN,
            "P return",
            ThematicOriginV1::Derived,
            NODE_RETURN_A,
        ),
    ] {
        graph
            .insert_identity(
                id,
                ThematicIdentityV1 {
                    label: Some(label.into()),
                    origin,
                    introduced_in: node.into(),
                },
            )
            .map_err(ProgSuiteWorkBridgeErrorV1::ThematicGraph)?;
    }

    for (id, target, transforms) in [
        (
            DERIVE_B,
            IDENTITY_B,
            transformation_classes(plan.sections[1].transformation),
        ),
        (
            DERIVE_C,
            IDENTITY_C,
            transformation_classes(plan.sections[2].transformation),
        ),
        (
            RETURN_A,
            IDENTITY_RETURN,
            vec![ThematicTransformationClassV1::LiteralReturn],
        ),
    ] {
        graph
            .insert_derivation(
                id,
                ThematicDerivationV1 {
                    source_id: IDENTITY_P.into(),
                    target_id: target.into(),
                    transformations: transforms,
                },
            )
            .map_err(ProgSuiteWorkBridgeErrorV1::ThematicGraph)?;
    }
    graph
        .validate(work_plan)
        .map_err(ProgSuiteWorkBridgeErrorV1::ThematicGraph)?;
    Ok(graph)
}

fn transformation_classes(
    transformation: ProgSuiteTransformV1,
) -> Vec<ThematicTransformationClassV1> {
    match transformation {
        ProgSuiteTransformV1::Original => vec![ThematicTransformationClassV1::LiteralReturn],
        ProgSuiteTransformV1::Inversion => vec![ThematicTransformationClassV1::Inversion],
        ProgSuiteTransformV1::Retrograde => vec![ThematicTransformationClassV1::Retrograde],
        ProgSuiteTransformV1::RetrogradeInversion => vec![
            ThematicTransformationClassV1::Other(RETROGRADE_INVERSION_LABEL.into()),
        ],
    }
}

fn build_obligation_plan(
    plan: &ProgSuitePlanV1,
    work_plan: &HierarchicalWorkPlanV1,
    thematic_graph: &ThematicIdentityGraphV1,
) -> Result<WorkObligationPlanV2, ProgSuiteWorkBridgeErrorV1> {
    let a = &plan.sections[0];
    let b = &plan.sections[1];
    let c = &plan.sections[2];
    let return_a = &plan.sections[3];
    let mut obligations = WorkObligationPlanV2::default();

    let entries = [
        (
            OBLIGATION_ESTABLISH,
            obligation(
                NODE_A,
                a.start,
                NODE_A,
                a.start,
                a.end,
                &[],
                WorkObligationKindV2::PresentThematicIdentity {
                    identity_id: IDENTITY_P.into(),
                },
            ),
        ),
        (
            OBLIGATION_TRANSFORM_B,
            obligation(
                NODE_A,
                a.start,
                NODE_B,
                b.start,
                b.end,
                &[OBLIGATION_ESTABLISH],
                WorkObligationKindV2::RealizeThematicDerivation {
                    derivation_id: DERIVE_B.into(),
                },
            ),
        ),
        (
            OBLIGATION_TRANSFORM_C,
            obligation(
                NODE_A,
                a.start,
                NODE_C,
                c.start,
                c.end,
                &[OBLIGATION_ESTABLISH],
                WorkObligationKindV2::RealizeThematicDerivation {
                    derivation_id: DERIVE_C.into(),
                },
            ),
        ),
        (
            OBLIGATION_REACH_RELATIVE,
            obligation(
                NODE_A,
                a.start,
                NODE_C,
                c.start,
                c.end,
                &[OBLIGATION_ESTABLISH],
                WorkObligationKindV2::ReachTonalCenter { key: c.key },
            ),
        ),
        (
            OBLIGATION_RETURN_PRIMARY,
            obligation(
                NODE_A,
                a.start,
                NODE_RETURN_A,
                return_a.start,
                return_a.end,
                &[
                    OBLIGATION_REACH_RELATIVE,
                    OBLIGATION_TRANSFORM_B,
                    OBLIGATION_TRANSFORM_C,
                ],
                WorkObligationKindV2::RealizeThematicDerivation {
                    derivation_id: RETURN_A.into(),
                },
            ),
        ),
        (
            OBLIGATION_RETURN_HOME,
            obligation(
                NODE_A,
                a.start,
                NODE_RETURN_A,
                return_a.start,
                return_a.end,
                &[OBLIGATION_RETURN_PRIMARY],
                WorkObligationKindV2::ReachTonalCenter { key: plan.home_key },
            ),
        ),
    ];

    for (id, entry) in entries {
        obligations
            .insert(id, entry)
            .map_err(ProgSuiteWorkBridgeErrorV1::ObligationPlan)?;
    }
    obligations
        .validate(work_plan, thematic_graph)
        .map_err(ProgSuiteWorkBridgeErrorV1::ObligationPlan)?;
    Ok(obligations)
}

fn obligation(
    declared_in: &str,
    created_at: crate::rhythm::Duration,
    due_context: &str,
    earliest: crate::rhythm::Duration,
    latest: crate::rhythm::Duration,
    prerequisites: &[&str],
    kind: WorkObligationKindV2,
) -> WorkObligationV2 {
    WorkObligationV2 {
        declared_in: declared_in.into(),
        created_at,
        due_context: due_context.into(),
        due: ObligationDueWindowV2 { earliest, latest },
        priority_per_mille: 1000,
        prerequisites: prerequisites.iter().map(|id| (*id).into()).collect(),
        conflicts_with: Vec::new(),
        kind,
    }
}

fn section_binding(
    section_index: usize,
    work_node_id: &str,
    thematic_identity_id: &str,
    derivation_id: Option<&str>,
    beats_per_bar: u8,
) -> ProgSuiteSectionWorkBindingV1 {
    ProgSuiteSectionWorkBindingV1 {
        section_index,
        work_node_id: work_node_id.into(),
        thematic_identity_id: thematic_identity_id.into(),
        derivation_id: derivation_id.map(str::to_string),
        meter_projection: ProgSuiteMeterProjectionV1::LegacyQuarterNoteBeatCount {
            beats_per_bar,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, Motif, MusicalIntent, PitchClass, Style, TimeSignature,
        plan_prog_suite, realize_prog_suite_with_plan,
    };

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn plan() -> ProgSuitePlanV1 {
        plan_prog_suite(
            Key::major(PitchClass::C),
            93.7,
            5,
            &Style::ProgFolk.spec(),
        )
        .unwrap()
    }

    #[test]
    fn mixed_meter_timeline_is_authoritative_at_exact_native_boundaries() {
        let binding = bridge_prog_suite_plan(&plan()).unwrap();
        assert_eq!(binding.temporal_map.points.len(), 4);
        assert_eq!(
            binding.temporal_map.meter_at(Duration::new(0, 1)).unwrap(),
            TimeSignature::new(4, 4).unwrap()
        );
        assert_eq!(
            binding.temporal_map.meter_at(Duration::new(32, 1)).unwrap(),
            TimeSignature::new(7, 4).unwrap()
        );
        assert_eq!(
            binding.temporal_map.meter_at(Duration::new(88, 1)).unwrap(),
            TimeSignature::new(5, 4).unwrap()
        );
        assert_eq!(
            binding.temporal_map.meter_at(Duration::new(128, 1)).unwrap(),
            TimeSignature::new(4, 4).unwrap()
        );
        let tempo = binding.temporal_map.tempo_at(Duration::zero()).unwrap();
        assert_eq!((tempo.bpm() as f32).to_bits(), 93.7_f32.to_bits());
    }

    #[test]
    fn compatibility_meter_grouping_projection_is_explicit() {
        let binding = bridge_prog_suite_plan(&plan()).unwrap();
        let counts: Vec<_> = binding
            .section_bindings
            .iter()
            .map(|section| match section.meter_projection {
                ProgSuiteMeterProjectionV1::LegacyQuarterNoteBeatCount { beats_per_bar } => {
                    beats_per_bar
                }
            })
            .collect();
        assert_eq!(counts, vec![4, 7, 5, 4]);
        assert_eq!(
            binding.temporal_map.meter_at(Duration::new(32, 1)).unwrap().grouping(),
            &[1, 1, 1, 1, 1, 1, 1]
        );
    }

    #[test]
    fn work_tree_preserves_all_four_exact_native_spans() {
        let binding = bridge_prog_suite_plan(&plan()).unwrap();
        let children = binding.work_plan.children_of(ROOT_ID).unwrap();
        let spans: Vec<_> = children
            .iter()
            .map(|(id, node)| ((*id).to_string(), node.start, node.end))
            .collect();
        assert_eq!(
            spans,
            vec![
                (NODE_A.into(), Duration::new(0, 1), Duration::new(32, 1)),
                (NODE_B.into(), Duration::new(32, 1), Duration::new(88, 1)),
                (NODE_C.into(), Duration::new(88, 1), Duration::new(128, 1)),
                (
                    NODE_RETURN_A.into(),
                    Duration::new(128, 1),
                    Duration::new(160, 1),
                ),
            ]
        );
    }

    #[test]
    fn native_transformations_become_explicit_thematic_genealogy() {
        let binding = bridge_prog_suite_plan(&plan()).unwrap();
        assert_eq!(
            binding.thematic_graph.derivations[DERIVE_B].transformations,
            vec![ThematicTransformationClassV1::Other(
                RETROGRADE_INVERSION_LABEL.into()
            )]
        );
        assert_eq!(
            binding.thematic_graph.derivations[DERIVE_C].transformations,
            vec![ThematicTransformationClassV1::Inversion]
        );
        assert_eq!(
            binding.thematic_graph.derivations[RETURN_A].transformations,
            vec![ThematicTransformationClassV1::LiteralReturn]
        );
        assert!(matches!(
            binding.obligation_plan.obligations[OBLIGATION_RETURN_PRIMARY].kind,
            WorkObligationKindV2::RealizeThematicDerivation { ref derivation_id }
                if derivation_id == RETURN_A
        ));
    }

    #[test]
    fn ordered_composite_is_not_misrepresented_as_two_independent_classes() {
        let mut edited = plan();
        edited.sections[1].transformation = ProgSuiteTransformV1::RetrogradeInversion;
        edited.sections[2].transformation = ProgSuiteTransformV1::Retrograde;
        edited.validate().unwrap();
        let binding = bridge_prog_suite_plan(&edited).unwrap();
        assert_eq!(
            binding.thematic_graph.derivations[DERIVE_B].transformations,
            vec![ThematicTransformationClassV1::Other(
                RETROGRADE_INVERSION_LABEL.into()
            )]
        );
        assert_ne!(
            binding.thematic_graph.derivations[DERIVE_B].transformations,
            vec![
                ThematicTransformationClassV1::Inversion,
                ThematicTransformationClassV1::Retrograde,
            ]
        );
    }

    #[test]
    fn edited_frozen_transforms_drive_genealogy_without_reconsulting_seed() {
        let mut edited = plan();
        let original_seed = edited.source_seed;
        edited.sections[1].transformation = ProgSuiteTransformV1::Retrograde;
        edited.sections[2].transformation = ProgSuiteTransformV1::RetrogradeInversion;
        edited.validate().unwrap();
        let binding = bridge_prog_suite_plan(&edited).unwrap();
        assert_eq!(binding.native_plan.source_seed, original_seed);
        assert_eq!(
            binding.thematic_graph.derivations[DERIVE_B].transformations,
            vec![ThematicTransformationClassV1::Retrograde]
        );
        assert_eq!(
            binding.thematic_graph.derivations[DERIVE_C].transformations,
            vec![ThematicTransformationClassV1::Other(
                RETROGRADE_INVERSION_LABEL.into()
            )]
        );
    }

    #[test]
    fn prospective_promises_validate_as_one_dependency_graph() {
        let binding = bridge_prog_suite_plan(&plan()).unwrap();
        assert_eq!(binding.obligation_plan.obligations.len(), 6);
        assert_eq!(
            binding
                .obligation_plan
                .dependency_order(&binding.work_plan, &binding.thematic_graph)
                .unwrap(),
            vec![
                OBLIGATION_ESTABLISH.into(),
                OBLIGATION_REACH_RELATIVE.into(),
                OBLIGATION_TRANSFORM_B.into(),
                OBLIGATION_TRANSFORM_C.into(),
                OBLIGATION_RETURN_PRIMARY.into(),
                OBLIGATION_RETURN_HOME.into(),
            ]
        );
    }

    #[test]
    fn completed_realization_binds_to_temporal_and_work_authority() {
        let plan = plan();
        let realization = realize_prog_suite_with_plan(&plan, &motif(), &MusicalIntent::default())
            .unwrap();
        let bound = bind_prog_suite_realization(&realization).unwrap();
        assert_eq!(bound.temporal_score.score.total_beats, Duration::new(160, 1));
        assert_eq!(
            bound.temporal_score.meter_at(Duration::new(88, 1)).unwrap(),
            TimeSignature::new(5, 4).unwrap()
        );
        assert_eq!(
            bound.declaration.work_plan.nodes[ROOT_ID].end,
            bound.temporal_score.score.total_beats
        );
    }

    #[test]
    fn stale_or_truncated_score_span_cannot_bind_to_the_work_plan() {
        let plan = plan();
        let mut realization =
            realize_prog_suite_with_plan(&plan, &motif(), &MusicalIntent::default()).unwrap();
        realization.score.total_beats = Duration::new(159, 1);
        assert_eq!(
            bind_prog_suite_realization(&realization),
            Err(ProgSuiteWorkBridgeErrorV1::WorkPlan(
                WorkPlanErrorV1::RootEndMismatch
            ))
        );
    }

    #[test]
    fn legacy_opening_tempo_must_match_the_authoritative_map() {
        let plan = plan();
        let mut realization =
            realize_prog_suite_with_plan(&plan, &motif(), &MusicalIntent::default()).unwrap();
        realization.score.tempo_bpm = 94.0;
        assert_eq!(
            bind_prog_suite_realization(&realization),
            Err(ProgSuiteWorkBridgeErrorV1::TemporalScore(
                TemporalScoreErrorV1::OpeningTempoMismatch
            ))
        );
    }
}
