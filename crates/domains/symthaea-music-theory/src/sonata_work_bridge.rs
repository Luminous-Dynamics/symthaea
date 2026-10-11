// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native Sonata → work-scale planning bridge.
//!
//! This adapter lets the existing, already-useful Sonata engine inhabit the
//! FORM-001/002/003 work-scale contracts without rewriting Sonata itself. It is
//! intentionally declaration-only: it translates the prospective `SonataPlan`
//! into hierarchy, thematic genealogy, and immutable work obligations. The
//! completed score and Sonata verifier remain separate evidence sources.

use crate::obligation::{ObligationKind, ObligationStatus, ReturnTransformation};
use crate::rhythm::Duration;
use crate::sonata::{PlannedSonataSection, SonataPlan, SonataSectionKind};
use crate::thematic_identity::{
    ThematicDerivationV1, ThematicGraphErrorV1, ThematicIdentityGraphV1, ThematicIdentityV1,
    ThematicOriginV1, ThematicTransformationClassV1,
};
use crate::work_obligation::{
    ObligationDueWindowV2, WorkObligationErrorV2, WorkObligationKindV2, WorkObligationPlanV2,
    WorkObligationV2,
};
use crate::work_plan::{
    FormalFunctionV1, HierarchicalWorkPlanV1, WorkNodeKindV1, WorkNodeV1, WorkPlanErrorV1,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const SONATA_WORK_BRIDGE_VERSION: &str = "melothaea-sonata-work-bridge-v1";

const ROOT_ID: &str = "sonata:work";
const PRIMARY_ID: &str = "sonata:P";
const SECONDARY_ID: &str = "sonata:S";
const DEVELOPMENT_ID: &str = "sonata:P-development";
const PRIMARY_RETURN_ID: &str = "sonata:P-return";
const SECONDARY_RETURN_ID: &str = "sonata:S-return";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SonataSectionWorkBindingV1 {
    pub section: SonataSectionKind,
    pub work_node_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SonataWorkBindingV1 {
    pub version: String,
    pub sections: Vec<SonataSectionWorkBindingV1>,
    pub work_plan: HierarchicalWorkPlanV1,
    pub thematic_graph: ThematicIdentityGraphV1,
    pub obligation_plan: WorkObligationPlanV2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SonataWorkBridgeErrorV1 {
    WrongSectionCount { found: usize },
    WrongSectionOrder { index: usize },
    FirstSectionMustStartAtZero,
    NonPositiveSectionSpan { index: usize },
    NonContiguousSections { left_index: usize, right_index: usize },
    WrongSectionKey { section: SonataSectionKind },
    WrongLegacyObligationCount { found: usize },
    LegacyObligationNotProspective { obligation_id: u64 },
    LegacyObligationCreatedAfterZero { obligation_id: u64 },
    MissingLegacyContrastKey,
    MissingLegacyDevelopmentClimax,
    MissingLegacyPrimaryReturn,
    MissingLegacyHomeReturn,
    MissingLegacySecondaryReturn,
    MissingLegacyHomeCadence,
    DuplicateOrUnsupportedLegacyObligation { obligation_id: u64 },
    LegacyDuePointMismatch { obligation_id: u64 },
    WorkPlan(WorkPlanErrorV1),
    ThematicGraph(ThematicGraphErrorV1),
    ObligationPlan(WorkObligationErrorV2),
}

pub fn bridge_native_sonata_plan(
    plan: &SonataPlan,
) -> Result<SonataWorkBindingV1, SonataWorkBridgeErrorV1> {
    validate_native_sonata_plan(plan)?;

    let last = plan.sections.last().expect("validated five-section sonata");
    let mut work_plan = HierarchicalWorkPlanV1::new(ROOT_ID, last.end)
        .map_err(SonataWorkBridgeErrorV1::WorkPlan)?;
    let mut section_bindings = Vec::with_capacity(plan.sections.len());

    for section in &plan.sections {
        let (node_id, label, functions) = section_descriptor(section.kind);
        work_plan
            .insert_node(
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
            .map_err(SonataWorkBridgeErrorV1::WorkPlan)?;
        section_bindings.push(SonataSectionWorkBindingV1 {
            section: section.kind,
            work_node_id: node_id.into(),
        });
    }

    let thematic_graph = build_thematic_graph();
    thematic_graph
        .validate(&work_plan)
        .map_err(SonataWorkBridgeErrorV1::ThematicGraph)?;

    let obligation_plan = build_obligation_plan(plan);
    obligation_plan
        .validate(&work_plan, &thematic_graph)
        .map_err(SonataWorkBridgeErrorV1::ObligationPlan)?;

    Ok(SonataWorkBindingV1 {
        version: SONATA_WORK_BRIDGE_VERSION.into(),
        sections: section_bindings,
        work_plan,
        thematic_graph,
        obligation_plan,
    })
}

pub fn validate_native_sonata_plan(plan: &SonataPlan) -> Result<(), SonataWorkBridgeErrorV1> {
    const EXPECTED: [SonataSectionKind; 5] = [
        SonataSectionKind::ExpositionPrimary,
        SonataSectionKind::ExpositionSecondary,
        SonataSectionKind::Development,
        SonataSectionKind::RecapitulationPrimary,
        SonataSectionKind::RecapitulationSecondary,
    ];
    if plan.sections.len() != EXPECTED.len() {
        return Err(SonataWorkBridgeErrorV1::WrongSectionCount {
            found: plan.sections.len(),
        });
    }
    if plan.sections[0].start != Duration::zero() {
        return Err(SonataWorkBridgeErrorV1::FirstSectionMustStartAtZero);
    }
    for (index, (section, expected)) in plan.sections.iter().zip(EXPECTED).enumerate() {
        if section.kind != expected {
            return Err(SonataWorkBridgeErrorV1::WrongSectionOrder { index });
        }
        if compare_duration(section.end, section.start) != Ordering::Greater {
            return Err(SonataWorkBridgeErrorV1::NonPositiveSectionSpan { index });
        }
        if index > 0 && plan.sections[index - 1].end != section.start {
            return Err(SonataWorkBridgeErrorV1::NonContiguousSections {
                left_index: index - 1,
                right_index: index,
            });
        }
    }

    for section in &plan.sections {
        let expected_key = match section.kind {
            SonataSectionKind::ExpositionPrimary | SonataSectionKind::RecapitulationPrimary => {
                plan.home_key
            }
            SonataSectionKind::ExpositionSecondary => plan.contrast_key,
            SonataSectionKind::Development => plan.development_key,
            SonataSectionKind::RecapitulationSecondary => plan.home_key,
        };
        if section.key != expected_key {
            return Err(SonataWorkBridgeErrorV1::WrongSectionKey {
                section: section.kind,
            });
        }
    }

    validate_legacy_obligations(plan)
}

fn validate_legacy_obligations(plan: &SonataPlan) -> Result<(), SonataWorkBridgeErrorV1> {
    let items = plan.obligations.items();
    if items.len() != 6 {
        return Err(SonataWorkBridgeErrorV1::WrongLegacyObligationCount { found: items.len() });
    }
    let exposition_secondary = section(plan, SonataSectionKind::ExpositionSecondary);
    let development = section(plan, SonataSectionKind::Development);
    let recap_primary = section(plan, SonataSectionKind::RecapitulationPrimary);
    let recap_secondary = section(plan, SonataSectionKind::RecapitulationSecondary);

    let mut contrast = false;
    let mut climax = false;
    let mut primary_return = false;
    let mut home_return = false;
    let mut secondary_return = false;
    let mut cadence = false;

    for obligation in items {
        if obligation.status != ObligationStatus::Pending || obligation.resolution_note.is_some() {
            return Err(SonataWorkBridgeErrorV1::LegacyObligationNotProspective {
                obligation_id: obligation.id,
            });
        }
        if obligation.created_at != Duration::zero() {
            return Err(SonataWorkBridgeErrorV1::LegacyObligationCreatedAfterZero {
                obligation_id: obligation.id,
            });
        }
        let expected_due = match &obligation.kind {
            ObligationKind::ReachKey { key } if *key == plan.contrast_key && !contrast => {
                contrast = true;
                exposition_secondary.start
            }
            ObligationKind::ReachClimax if !climax => {
                climax = true;
                development.end
            }
            ObligationKind::ReturnMotif {
                motif_id,
                transformation: ReturnTransformation::Literal,
            } if motif_id == "sonata.primary" && !primary_return => {
                primary_return = true;
                recap_primary.end
            }
            ObligationKind::ReachKey { key } if *key == plan.home_key && !home_return => {
                home_return = true;
                recap_primary.start
            }
            ObligationKind::ReturnMotif {
                motif_id,
                transformation: ReturnTransformation::Transposed,
            } if motif_id == "sonata.secondary" && !secondary_return => {
                secondary_return = true;
                recap_secondary.end
            }
            ObligationKind::Cadence { .. } if !cadence => {
                cadence = true;
                recap_secondary.end
            }
            _ => {
                return Err(SonataWorkBridgeErrorV1::DuplicateOrUnsupportedLegacyObligation {
                    obligation_id: obligation.id,
                });
            }
        };
        if obligation.due_by != expected_due {
            return Err(SonataWorkBridgeErrorV1::LegacyDuePointMismatch {
                obligation_id: obligation.id,
            });
        }
    }

    if !contrast {
        return Err(SonataWorkBridgeErrorV1::MissingLegacyContrastKey);
    }
    if !climax {
        return Err(SonataWorkBridgeErrorV1::MissingLegacyDevelopmentClimax);
    }
    if !primary_return {
        return Err(SonataWorkBridgeErrorV1::MissingLegacyPrimaryReturn);
    }
    if !home_return {
        return Err(SonataWorkBridgeErrorV1::MissingLegacyHomeReturn);
    }
    if !secondary_return {
        return Err(SonataWorkBridgeErrorV1::MissingLegacySecondaryReturn);
    }
    if !cadence {
        return Err(SonataWorkBridgeErrorV1::MissingLegacyHomeCadence);
    }
    Ok(())
}

fn build_thematic_graph() -> ThematicIdentityGraphV1 {
    let mut graph = ThematicIdentityGraphV1::default();
    for (id, origin, node) in [
        (
            PRIMARY_ID,
            ThematicOriginV1::Independent,
            node_id(SonataSectionKind::ExpositionPrimary),
        ),
        (
            SECONDARY_ID,
            ThematicOriginV1::Derived,
            node_id(SonataSectionKind::ExpositionSecondary),
        ),
        (
            DEVELOPMENT_ID,
            ThematicOriginV1::Derived,
            node_id(SonataSectionKind::Development),
        ),
        (
            PRIMARY_RETURN_ID,
            ThematicOriginV1::Derived,
            node_id(SonataSectionKind::RecapitulationPrimary),
        ),
        (
            SECONDARY_RETURN_ID,
            ThematicOriginV1::Derived,
            node_id(SonataSectionKind::RecapitulationSecondary),
        ),
    ] {
        graph
            .insert_identity(
                id,
                ThematicIdentityV1 {
                    label: Some(id.into()),
                    origin,
                    introduced_in: node.into(),
                },
            )
            .expect("static nonempty thematic identity");
    }

    for (id, source, target, transformations) in [
        (
            "sonata:derive-secondary",
            PRIMARY_ID,
            SECONDARY_ID,
            vec![ThematicTransformationClassV1::Other(
                "sonata-secondary-contrast".into(),
            )],
        ),
        (
            "sonata:develop-primary",
            PRIMARY_ID,
            DEVELOPMENT_ID,
            vec![ThematicTransformationClassV1::Fragmentation],
        ),
        (
            "sonata:return-primary",
            PRIMARY_ID,
            PRIMARY_RETURN_ID,
            vec![ThematicTransformationClassV1::LiteralReturn],
        ),
        (
            "sonata:return-secondary",
            SECONDARY_ID,
            SECONDARY_RETURN_ID,
            vec![ThematicTransformationClassV1::Transposition],
        ),
    ] {
        graph
            .insert_derivation(
                id,
                ThematicDerivationV1 {
                    source_id: source.into(),
                    target_id: target.into(),
                    transformations,
                },
            )
            .expect("static nonempty thematic derivation");
    }
    graph
}

fn build_obligation_plan(plan: &SonataPlan) -> WorkObligationPlanV2 {
    let opening = section(plan, SonataSectionKind::ExpositionPrimary);
    let secondary = section(plan, SonataSectionKind::ExpositionSecondary);
    let development = section(plan, SonataSectionKind::Development);
    let recap_primary = section(plan, SonataSectionKind::RecapitulationPrimary);
    let recap_secondary = section(plan, SonataSectionKind::RecapitulationSecondary);
    let declared_in = node_id(SonataSectionKind::ExpositionPrimary);

    let mut obligations = WorkObligationPlanV2::default();
    let entries = [
        (
            "sonata:establish-primary",
            obligation(
                declared_in,
                opening.start,
                node_id(SonataSectionKind::ExpositionPrimary),
                opening.start,
                opening.end,
                1000,
                &[],
                WorkObligationKindV2::PresentThematicIdentity {
                    identity_id: PRIMARY_ID.into(),
                },
            ),
        ),
        (
            "sonata:establish-secondary",
            obligation(
                declared_in,
                opening.start,
                node_id(SonataSectionKind::ExpositionSecondary),
                secondary.start,
                secondary.end,
                900,
                &["sonata:establish-primary"],
                WorkObligationKindV2::PresentThematicIdentity {
                    identity_id: SECONDARY_ID.into(),
                },
            ),
        ),
        (
            "sonata:reach-contrast-key",
            obligation(
                declared_in,
                opening.start,
                node_id(SonataSectionKind::ExpositionSecondary),
                secondary.start,
                secondary.start,
                750,
                &["sonata:establish-primary"],
                WorkObligationKindV2::ReachTonalCenter {
                    key: plan.contrast_key,
                },
            ),
        ),
        (
            "sonata:develop-primary",
            obligation(
                declared_in,
                opening.start,
                node_id(SonataSectionKind::Development),
                development.start,
                development.end,
                850,
                &["sonata:establish-primary"],
                WorkObligationKindV2::RealizeThematicDerivation {
                    derivation_id: "sonata:develop-primary".into(),
                },
            ),
        ),
        (
            "sonata:development-climax",
            obligation(
                declared_in,
                opening.start,
                node_id(SonataSectionKind::Development),
                development.end,
                development.end,
                800,
                &["sonata:establish-secondary"],
                WorkObligationKindV2::ReachClimax,
            ),
        ),
        (
            "sonata:return-primary",
            obligation(
                declared_in,
                opening.start,
                node_id(SonataSectionKind::RecapitulationPrimary),
                recap_primary.end,
                recap_primary.end,
                1000,
                &["sonata:develop-primary"],
                WorkObligationKindV2::PresentThematicIdentity {
                    identity_id: PRIMARY_RETURN_ID.into(),
                },
            ),
        ),
        (
            "sonata:return-home",
            obligation(
                declared_in,
                opening.start,
                node_id(SonataSectionKind::RecapitulationPrimary),
                recap_primary.start,
                recap_primary.start,
                1000,
                &["sonata:reach-contrast-key"],
                WorkObligationKindV2::ReachTonalCenter { key: plan.home_key },
            ),
        ),
        (
            "sonata:return-secondary",
            obligation(
                declared_in,
                opening.start,
                node_id(SonataSectionKind::RecapitulationSecondary),
                recap_secondary.end,
                recap_secondary.end,
                1000,
                &["sonata:establish-secondary"],
                WorkObligationKindV2::PresentThematicIdentity {
                    identity_id: SECONDARY_RETURN_ID.into(),
                },
            ),
        ),
        (
            "sonata:close-home",
            obligation(
                declared_in,
                opening.start,
                node_id(SonataSectionKind::RecapitulationSecondary),
                recap_secondary.end,
                recap_secondary.end,
                1000,
                &["sonata:return-home", "sonata:return-secondary"],
                WorkObligationKindV2::ReachTonalCenter { key: plan.home_key },
            ),
        ),
    ];
    for (id, entry) in entries {
        obligations
            .insert(id, entry)
            .expect("static nonempty obligation identity");
    }
    obligations
}

#[allow(clippy::too_many_arguments)]
fn obligation(
    declared_in: &str,
    created_at: Duration,
    due_context: &str,
    earliest: Duration,
    latest: Duration,
    priority_per_mille: u16,
    prerequisites: &[&str],
    kind: WorkObligationKindV2,
) -> WorkObligationV2 {
    WorkObligationV2 {
        declared_in: declared_in.into(),
        created_at,
        due_context: due_context.into(),
        due: ObligationDueWindowV2 { earliest, latest },
        priority_per_mille,
        prerequisites: prerequisites.iter().map(|id| (*id).into()).collect(),
        conflicts_with: Vec::new(),
        kind,
    }
}

fn section(plan: &SonataPlan, kind: SonataSectionKind) -> &PlannedSonataSection {
    plan.sections
        .iter()
        .find(|section| section.kind == kind)
        .expect("canonical sonata section validated")
}

fn node_id(kind: SonataSectionKind) -> &'static str {
    match kind {
        SonataSectionKind::ExpositionPrimary => "sonata:exposition-primary",
        SonataSectionKind::ExpositionSecondary => "sonata:exposition-secondary",
        SonataSectionKind::Development => "sonata:development",
        SonataSectionKind::RecapitulationPrimary => "sonata:recapitulation-primary",
        SonataSectionKind::RecapitulationSecondary => "sonata:recapitulation-secondary",
    }
}

fn section_descriptor(kind: SonataSectionKind) -> (&'static str, &'static str, Vec<FormalFunctionV1>) {
    match kind {
        SonataSectionKind::ExpositionPrimary => (
            node_id(kind),
            "Exposition — Primary",
            vec![FormalFunctionV1::Establish],
        ),
        SonataSectionKind::ExpositionSecondary => (
            node_id(kind),
            "Exposition — Secondary",
            vec![FormalFunctionV1::Contrast],
        ),
        SonataSectionKind::Development => (
            node_id(kind),
            "Development",
            vec![FormalFunctionV1::Develop, FormalFunctionV1::Destabilize],
        ),
        SonataSectionKind::RecapitulationPrimary => (
            node_id(kind),
            "Recapitulation — Primary",
            vec![FormalFunctionV1::Return, FormalFunctionV1::Resolve],
        ),
        SonataSectionKind::RecapitulationSecondary => (
            node_id(kind),
            "Recapitulation — Secondary",
            vec![FormalFunctionV1::Return, FormalFunctionV1::Resolve],
        ),
    }
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Key, Motif, PitchClass};

    fn plan() -> SonataPlan {
        let motif = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
            (2, Duration::quarter()),
        ]);
        crate::plan_sonata(Key::major(PitchClass::C), 4.0, &motif, 11)
    }

    #[test]
    fn canonical_native_sonata_bridges_without_rewriting_sonata() {
        let source = plan();
        let binding = bridge_native_sonata_plan(&source).unwrap();
        assert_eq!(binding.version, SONATA_WORK_BRIDGE_VERSION);
        assert_eq!(binding.sections.len(), 5);
        assert_eq!(binding.work_plan.nodes.len(), 6);
        assert!(binding.thematic_graph.validate(&binding.work_plan).is_ok());
        assert!(binding
            .obligation_plan
            .validate(&binding.work_plan, &binding.thematic_graph)
            .is_ok());
        assert_eq!(
            binding.work_plan.nodes[ROOT_ID].end,
            source.sections.last().unwrap().end
        );
    }

    #[test]
    fn thematic_bridge_retains_primary_and_secondary_genealogy() {
        let binding = bridge_native_sonata_plan(&plan()).unwrap();
        let primary_ancestors = binding
            .thematic_graph
            .ancestors_of(&binding.work_plan, PRIMARY_RETURN_ID)
            .unwrap();
        assert_eq!(primary_ancestors, vec![PRIMARY_ID]);
        let secondary_ancestors = binding
            .thematic_graph
            .ancestors_of(&binding.work_plan, SECONDARY_RETURN_ID)
            .unwrap();
        assert_eq!(secondary_ancestors, vec![PRIMARY_ID, SECONDARY_ID]);
    }

    #[test]
    fn return_promises_are_created_at_opening_but_due_much_later() {
        let binding = bridge_native_sonata_plan(&plan()).unwrap();
        let promise = &binding.obligation_plan.obligations["sonata:return-secondary"];
        assert_eq!(promise.created_at, Duration::zero());
        let recap = section(&plan(), SonataSectionKind::RecapitulationSecondary).clone();
        assert_eq!(promise.due.latest, recap.end);
        assert!(compare_duration(promise.due.latest, promise.created_at) == Ordering::Greater);
    }

    #[test]
    fn forged_section_key_fails_native_bridge_admission() {
        let mut source = plan();
        source.sections[1].key = source.home_key;
        assert!(matches!(
            bridge_native_sonata_plan(&source),
            Err(SonataWorkBridgeErrorV1::WrongSectionKey {
                section: SonataSectionKind::ExpositionSecondary
            })
        ));
    }

    #[test]
    fn forged_section_order_fails_native_bridge_admission() {
        let mut source = plan();
        source.sections.swap(1, 2);
        assert!(matches!(
            bridge_native_sonata_plan(&source),
            Err(SonataWorkBridgeErrorV1::WrongSectionOrder { index: 1 })
        ));
    }

    #[test]
    fn fulfilled_legacy_plan_cannot_masquerade_as_prospective_source() {
        let mut source = plan();
        let id = source.obligations.items()[0].id;
        source
            .obligations
            .get_mut(id)
            .unwrap()
            .fulfil("retrospective evidence");
        assert!(matches!(
            bridge_native_sonata_plan(&source),
            Err(SonataWorkBridgeErrorV1::LegacyObligationNotProspective { .. })
        ));
    }

    #[test]
    fn forged_primary_return_transformation_is_rejected() {
        let mut source = plan();
        let id = source
            .obligations
            .items()
            .iter()
            .find(|obligation| matches!(
                &obligation.kind,
                ObligationKind::ReturnMotif { motif_id, .. } if motif_id == "sonata.primary"
            ))
            .unwrap()
            .id;
        source.obligations.get_mut(id).unwrap().kind = ObligationKind::ReturnMotif {
            motif_id: "sonata.primary".into(),
            transformation: ReturnTransformation::Inverted,
        };
        assert!(matches!(
            bridge_native_sonata_plan(&source),
            Err(SonataWorkBridgeErrorV1::DuplicateOrUnsupportedLegacyObligation { .. })
        ));
    }
}
