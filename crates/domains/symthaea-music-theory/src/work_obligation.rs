// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable work-scale prospective musical memory.
//!
//! The legacy [`crate::obligation::ObligationLedger`] is useful inside local
//! composition engines, but it combines a promise with mutable fulfillment
//! state. Long-form planning needs a stronger separation: this module freezes
//! what a work promises in advance, including exact due windows, dependencies,
//! conflicts, and references to FORM-002 thematic identities. Resolution
//! evidence is intentionally not stored here.

use crate::harmony::Key;
use crate::rhythm::Duration;
use crate::thematic_identity::{ThematicGraphErrorV1, ThematicIdentityGraphV1};
use crate::work_plan::{HierarchicalWorkPlanV1, WorkPlanErrorV1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const WORK_OBLIGATION_PLAN_VERSION: &str = "melothaea-work-obligation-plan-v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObligationDueWindowV2 {
    /// Earliest acceptable resolution beat.
    pub earliest: Duration,
    /// Latest acceptable resolution beat. The interval is closed so a promise
    /// may resolve exactly at a work/section boundary.
    pub latest: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkObligationKindV2 {
    /// Make an already-declared FORM-002 thematic identity present by the due window.
    PresentThematicIdentity { identity_id: String },
    /// Realize the declared relationship represented by one FORM-002 derivation edge.
    RealizeThematicDerivation { derivation_id: String },
    ReachTonalCenter { key: Key },
    ReachClimax,
    EnterVoice { voice_id: String },
    Custom { label: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkObligationV2 {
    /// Work-plan node in which the promise becomes narratively active.
    pub declared_in: String,
    pub created_at: Duration,
    /// Work-plan node whose span contains the complete due window.
    pub due_context: String,
    pub due: ObligationDueWindowV2,
    /// Exact deterministic priority in 0..=1000 rather than a floating score.
    pub priority_per_mille: u16,
    /// Stable obligation IDs, sorted and unique.
    pub prerequisites: Vec<String>,
    /// Stable obligation IDs, sorted and unique. Conflict declarations must be symmetric.
    pub conflicts_with: Vec<String>,
    pub kind: WorkObligationKindV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkObligationPlanV2 {
    pub version: String,
    pub obligations: BTreeMap<String, WorkObligationV2>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkObligationErrorV2 {
    WrongVersion { found: String },
    InvalidWorkPlan(WorkPlanErrorV1),
    InvalidThematicGraph(ThematicGraphErrorV1),
    EmptyObligationId,
    EmptyDeclaredNode { obligation_id: String },
    EmptyDueContext { obligation_id: String },
    MissingDeclaredNode { obligation_id: String, node_id: String },
    MissingDueContext { obligation_id: String, node_id: String },
    CreatedOutsideDeclaredNode { obligation_id: String },
    NegativeDueWindow { obligation_id: String },
    InvalidDueWindow { obligation_id: String },
    DueBeforeCreation { obligation_id: String },
    DueOutsideContext { obligation_id: String },
    PriorityOutOfRange { obligation_id: String, found: u16 },
    NonCanonicalPrerequisites { obligation_id: String },
    NonCanonicalConflicts { obligation_id: String },
    SelfPrerequisite { obligation_id: String },
    SelfConflict { obligation_id: String },
    MissingPrerequisite { obligation_id: String, prerequisite_id: String },
    MissingConflict { obligation_id: String, conflict_id: String },
    AsymmetricConflict { obligation_id: String, conflict_id: String },
    PrerequisiteConflict { obligation_id: String, other_id: String },
    DependencyCycle { obligation_id: String },
    PrerequisiteDueTooLate {
        obligation_id: String,
        prerequisite_id: String,
    },
    EmptyKindReference { obligation_id: String },
    UnknownThematicIdentity { obligation_id: String, identity_id: String },
    UnknownThematicDerivation { obligation_id: String, derivation_id: String },
    ThematicIdentityNotYetIntroduced { obligation_id: String, identity_id: String },
    ThematicDerivationNotYetIntroduced { obligation_id: String, derivation_id: String },
    EmptyVoiceId { obligation_id: String },
    EmptyCustomLabel { obligation_id: String },
}

impl Default for WorkObligationPlanV2 {
    fn default() -> Self {
        Self {
            version: WORK_OBLIGATION_PLAN_VERSION.into(),
            obligations: BTreeMap::new(),
        }
    }
}

impl WorkObligationPlanV2 {
    pub fn insert(
        &mut self,
        obligation_id: impl Into<String>,
        obligation: WorkObligationV2,
    ) -> Result<(), WorkObligationErrorV2> {
        let obligation_id = obligation_id.into();
        if obligation_id.trim().is_empty() {
            return Err(WorkObligationErrorV2::EmptyObligationId);
        }
        self.obligations.insert(obligation_id, obligation);
        Ok(())
    }

    pub fn validate(
        &self,
        work_plan: &HierarchicalWorkPlanV1,
        thematic_graph: &ThematicIdentityGraphV1,
    ) -> Result<(), WorkObligationErrorV2> {
        if self.version != WORK_OBLIGATION_PLAN_VERSION {
            return Err(WorkObligationErrorV2::WrongVersion {
                found: self.version.clone(),
            });
        }
        work_plan
            .validate()
            .map_err(WorkObligationErrorV2::InvalidWorkPlan)?;
        thematic_graph
            .validate(work_plan)
            .map_err(WorkObligationErrorV2::InvalidThematicGraph)?;

        for (obligation_id, obligation) in &self.obligations {
            validate_one(obligation_id, obligation, work_plan, thematic_graph)?;
        }

        for (obligation_id, obligation) in &self.obligations {
            for prerequisite_id in &obligation.prerequisites {
                let prerequisite = self.obligations.get(prerequisite_id).ok_or_else(|| {
                    WorkObligationErrorV2::MissingPrerequisite {
                        obligation_id: obligation_id.clone(),
                        prerequisite_id: prerequisite_id.clone(),
                    }
                })?;
                if obligation.conflicts_with.binary_search(prerequisite_id).is_ok() {
                    return Err(WorkObligationErrorV2::PrerequisiteConflict {
                        obligation_id: obligation_id.clone(),
                        other_id: prerequisite_id.clone(),
                    });
                }
                if compare_duration(prerequisite.due.latest, obligation.due.latest)
                    == Ordering::Greater
                {
                    return Err(WorkObligationErrorV2::PrerequisiteDueTooLate {
                        obligation_id: obligation_id.clone(),
                        prerequisite_id: prerequisite_id.clone(),
                    });
                }
            }

            for conflict_id in &obligation.conflicts_with {
                let conflict = self.obligations.get(conflict_id).ok_or_else(|| {
                    WorkObligationErrorV2::MissingConflict {
                        obligation_id: obligation_id.clone(),
                        conflict_id: conflict_id.clone(),
                    }
                })?;
                if conflict.conflicts_with.binary_search(obligation_id).is_err() {
                    return Err(WorkObligationErrorV2::AsymmetricConflict {
                        obligation_id: obligation_id.clone(),
                        conflict_id: conflict_id.clone(),
                    });
                }
                if conflict.prerequisites.binary_search(obligation_id).is_ok() {
                    return Err(WorkObligationErrorV2::PrerequisiteConflict {
                        obligation_id: conflict_id.clone(),
                        other_id: obligation_id.clone(),
                    });
                }
            }
        }

        // A dependency cycle would make prospective resolution impossible.
        for obligation_id in self.obligations.keys() {
            let mut visiting = BTreeSet::new();
            let mut visited = BTreeSet::new();
            if dependency_cycle(obligation_id, &self.obligations, &mut visiting, &mut visited) {
                return Err(WorkObligationErrorV2::DependencyCycle {
                    obligation_id: obligation_id.clone(),
                });
            }
        }
        Ok(())
    }

    /// Stable prerequisite-first order suitable for a planner or visualization.
    pub fn dependency_order(
        &self,
        work_plan: &HierarchicalWorkPlanV1,
        thematic_graph: &ThematicIdentityGraphV1,
    ) -> Result<Vec<String>, WorkObligationErrorV2> {
        self.validate(work_plan, thematic_graph)?;
        let mut indegree: BTreeMap<String, usize> = self
            .obligations
            .keys()
            .map(|id| (id.clone(), 0usize))
            .collect();
        let mut dependents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (id, obligation) in &self.obligations {
            indegree.insert(id.clone(), obligation.prerequisites.len());
            for prerequisite in &obligation.prerequisites {
                dependents
                    .entry(prerequisite.as_str())
                    .or_default()
                    .push(id.as_str());
            }
        }
        let mut ready: BTreeSet<String> = indegree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(id.clone()))
            .collect();
        let mut ordered = Vec::with_capacity(self.obligations.len());
        while let Some(id) = ready.pop_first() {
            ordered.push(id.clone());
            if let Some(children) = dependents.get(id.as_str()) {
                for child in children {
                    let degree = indegree.get_mut(*child).expect("known dependent");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert((*child).to_string());
                    }
                }
            }
        }
        debug_assert_eq!(ordered.len(), self.obligations.len());
        Ok(ordered)
    }
}

fn validate_one(
    obligation_id: &str,
    obligation: &WorkObligationV2,
    work_plan: &HierarchicalWorkPlanV1,
    thematic_graph: &ThematicIdentityGraphV1,
) -> Result<(), WorkObligationErrorV2> {
    if obligation_id.trim().is_empty() {
        return Err(WorkObligationErrorV2::EmptyObligationId);
    }
    if obligation.declared_in.trim().is_empty() {
        return Err(WorkObligationErrorV2::EmptyDeclaredNode {
            obligation_id: obligation_id.into(),
        });
    }
    if obligation.due_context.trim().is_empty() {
        return Err(WorkObligationErrorV2::EmptyDueContext {
            obligation_id: obligation_id.into(),
        });
    }
    let declared_node = work_plan
        .nodes
        .get(&obligation.declared_in)
        .ok_or_else(|| WorkObligationErrorV2::MissingDeclaredNode {
            obligation_id: obligation_id.into(),
            node_id: obligation.declared_in.clone(),
        })?;
    let due_context = work_plan
        .nodes
        .get(&obligation.due_context)
        .ok_or_else(|| WorkObligationErrorV2::MissingDueContext {
            obligation_id: obligation_id.into(),
            node_id: obligation.due_context.clone(),
        })?;

    if compare_duration(obligation.created_at, declared_node.start) == Ordering::Less
        || compare_duration(obligation.created_at, declared_node.end) != Ordering::Less
    {
        return Err(WorkObligationErrorV2::CreatedOutsideDeclaredNode {
            obligation_id: obligation_id.into(),
        });
    }
    if obligation.due.earliest.num() < 0 || obligation.due.latest.num() < 0 {
        return Err(WorkObligationErrorV2::NegativeDueWindow {
            obligation_id: obligation_id.into(),
        });
    }
    if compare_duration(obligation.due.latest, obligation.due.earliest) == Ordering::Less {
        return Err(WorkObligationErrorV2::InvalidDueWindow {
            obligation_id: obligation_id.into(),
        });
    }
    if compare_duration(obligation.due.earliest, obligation.created_at) == Ordering::Less {
        return Err(WorkObligationErrorV2::DueBeforeCreation {
            obligation_id: obligation_id.into(),
        });
    }
    if compare_duration(obligation.due.earliest, due_context.start) == Ordering::Less
        || compare_duration(obligation.due.latest, due_context.end) == Ordering::Greater
    {
        return Err(WorkObligationErrorV2::DueOutsideContext {
            obligation_id: obligation_id.into(),
        });
    }
    if obligation.priority_per_mille > 1000 {
        return Err(WorkObligationErrorV2::PriorityOutOfRange {
            obligation_id: obligation_id.into(),
            found: obligation.priority_per_mille,
        });
    }
    validate_id_list(obligation_id, &obligation.prerequisites, true)?;
    validate_id_list(obligation_id, &obligation.conflicts_with, false)?;

    match &obligation.kind {
        WorkObligationKindV2::PresentThematicIdentity { identity_id } => {
            if identity_id.trim().is_empty() {
                return Err(WorkObligationErrorV2::EmptyKindReference {
                    obligation_id: obligation_id.into(),
                });
            }
            let identity = thematic_graph.identities.get(identity_id).ok_or_else(|| {
                WorkObligationErrorV2::UnknownThematicIdentity {
                    obligation_id: obligation_id.into(),
                    identity_id: identity_id.clone(),
                }
            })?;
            let intro = work_plan
                .nodes
                .get(&identity.introduced_in)
                .expect("thematic graph already validated");
            if compare_duration(obligation.due.earliest, intro.start) == Ordering::Less {
                return Err(WorkObligationErrorV2::ThematicIdentityNotYetIntroduced {
                    obligation_id: obligation_id.into(),
                    identity_id: identity_id.clone(),
                });
            }
        }
        WorkObligationKindV2::RealizeThematicDerivation { derivation_id } => {
            if derivation_id.trim().is_empty() {
                return Err(WorkObligationErrorV2::EmptyKindReference {
                    obligation_id: obligation_id.into(),
                });
            }
            let derivation = thematic_graph.derivations.get(derivation_id).ok_or_else(|| {
                WorkObligationErrorV2::UnknownThematicDerivation {
                    obligation_id: obligation_id.into(),
                    derivation_id: derivation_id.clone(),
                }
            })?;
            let target = thematic_graph
                .identities
                .get(&derivation.target_id)
                .expect("thematic graph already validated");
            let intro = work_plan
                .nodes
                .get(&target.introduced_in)
                .expect("thematic graph already validated");
            if compare_duration(obligation.due.earliest, intro.start) == Ordering::Less {
                return Err(WorkObligationErrorV2::ThematicDerivationNotYetIntroduced {
                    obligation_id: obligation_id.into(),
                    derivation_id: derivation_id.clone(),
                });
            }
        }
        WorkObligationKindV2::EnterVoice { voice_id } if voice_id.trim().is_empty() => {
            return Err(WorkObligationErrorV2::EmptyVoiceId {
                obligation_id: obligation_id.into(),
            });
        }
        WorkObligationKindV2::Custom { label } if label.trim().is_empty() => {
            return Err(WorkObligationErrorV2::EmptyCustomLabel {
                obligation_id: obligation_id.into(),
            });
        }
        _ => {}
    }
    Ok(())
}

fn validate_id_list(
    obligation_id: &str,
    values: &[String],
    prerequisite: bool,
) -> Result<(), WorkObligationErrorV2> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(if prerequisite {
            WorkObligationErrorV2::NonCanonicalPrerequisites {
                obligation_id: obligation_id.into(),
            }
        } else {
            WorkObligationErrorV2::NonCanonicalConflicts {
                obligation_id: obligation_id.into(),
            }
        });
    }
    if values.iter().any(|value| value == obligation_id) {
        return Err(if prerequisite {
            WorkObligationErrorV2::SelfPrerequisite {
                obligation_id: obligation_id.into(),
            }
        } else {
            WorkObligationErrorV2::SelfConflict {
                obligation_id: obligation_id.into(),
            }
        });
    }
    Ok(())
}

fn dependency_cycle<'a>(
    current: &'a str,
    obligations: &'a BTreeMap<String, WorkObligationV2>,
    visiting: &mut BTreeSet<&'a str>,
    visited: &mut BTreeSet<&'a str>,
) -> bool {
    if visited.contains(current) {
        return false;
    }
    if !visiting.insert(current) {
        return true;
    }
    if let Some(obligation) = obligations.get(current) {
        for prerequisite in &obligation.prerequisites {
            if dependency_cycle(prerequisite, obligations, visiting, visited) {
                return true;
            }
        }
    }
    visiting.remove(current);
    visited.insert(current);
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
        FormalFunctionV1, ThematicDerivationV1, ThematicIdentityV1, ThematicOriginV1,
        ThematicTransformationClassV1, WorkNodeKindV1, WorkNodeV1,
    };

    fn work_plan() -> HierarchicalWorkPlanV1 {
        let mut plan = HierarchicalWorkPlanV1::new("work", Duration::new(160, 1)).unwrap();
        for (id, start, end, functions) in [
            ("opening", 0, 32, vec![FormalFunctionV1::Establish]),
            (
                "development",
                32,
                80,
                vec![FormalFunctionV1::Develop, FormalFunctionV1::Destabilize],
            ),
            (
                "return",
                80,
                144,
                vec![FormalFunctionV1::Return, FormalFunctionV1::Synthesize],
            ),
            (
                "coda",
                144,
                160,
                vec![FormalFunctionV1::Resolve, FormalFunctionV1::Close],
            ),
        ] {
            plan.insert_node(
                id,
                WorkNodeV1 {
                    parent_id: Some("work".into()),
                    label: None,
                    kind: if id == "coda" {
                        WorkNodeKindV1::Coda
                    } else {
                        WorkNodeKindV1::Section
                    },
                    start: Duration::new(start, 1),
                    end: Duration::new(end, 1),
                    functions,
                },
            )
            .unwrap();
        }
        plan
    }

    fn thematic_graph() -> ThematicIdentityGraphV1 {
        let mut graph = ThematicIdentityGraphV1::default();
        for (id, origin, node) in [
            ("A", ThematicOriginV1::Independent, "opening"),
            ("B", ThematicOriginV1::Independent, "development"),
            ("A1", ThematicOriginV1::Derived, "development"),
            ("AB", ThematicOriginV1::Synthesis, "return"),
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
        for (id, source, target, class) in [
            (
                "a-a1",
                "A",
                "A1",
                ThematicTransformationClassV1::Fragmentation,
            ),
            (
                "a1-ab",
                "A1",
                "AB",
                ThematicTransformationClassV1::HarmonicReinterpretation,
            ),
            (
                "b-ab",
                "B",
                "AB",
                ThematicTransformationClassV1::Reorchestration,
            ),
        ] {
            graph
                .insert_derivation(
                    id,
                    ThematicDerivationV1 {
                        source_id: source.into(),
                        target_id: target.into(),
                        transformations: vec![class],
                    },
                )
                .unwrap();
        }
        graph
    }

    fn obligation(
        declared_in: &str,
        created: i64,
        due_context: &str,
        earliest: i64,
        latest: i64,
        prerequisites: &[&str],
        kind: WorkObligationKindV2,
    ) -> WorkObligationV2 {
        WorkObligationV2 {
            declared_in: declared_in.into(),
            created_at: Duration::new(created, 1),
            due_context: due_context.into(),
            due: ObligationDueWindowV2 {
                earliest: Duration::new(earliest, 1),
                latest: Duration::new(latest, 1),
            },
            priority_per_mille: 900,
            prerequisites: prerequisites.iter().map(|id| (*id).into()).collect(),
            conflicts_with: Vec::new(),
            kind,
        }
    }

    fn valid_plan() -> WorkObligationPlanV2 {
        let mut plan = WorkObligationPlanV2::default();
        plan.insert(
            "establish-a",
            obligation(
                "opening",
                0,
                "opening",
                0,
                32,
                &[],
                WorkObligationKindV2::PresentThematicIdentity {
                    identity_id: "A".into(),
                },
            ),
        )
        .unwrap();
        plan.insert(
            "develop-a",
            obligation(
                "opening",
                8,
                "development",
                32,
                80,
                &["establish-a"],
                WorkObligationKindV2::RealizeThematicDerivation {
                    derivation_id: "a-a1".into(),
                },
            ),
        )
        .unwrap();
        plan.insert(
            "introduce-b",
            obligation(
                "opening",
                8,
                "development",
                32,
                80,
                &[],
                WorkObligationKindV2::PresentThematicIdentity {
                    identity_id: "B".into(),
                },
            ),
        )
        .unwrap();
        plan.insert(
            "synthesize-ab",
            obligation(
                "opening",
                16,
                "return",
                80,
                144,
                &["develop-a", "introduce-b"],
                WorkObligationKindV2::PresentThematicIdentity {
                    identity_id: "AB".into(),
                },
            ),
        )
        .unwrap();
        plan.insert(
            "close-home",
            obligation(
                "opening",
                24,
                "coda",
                144,
                160,
                &["synthesize-ab"],
                WorkObligationKindV2::ReachTonalCenter {
                    key: Key::major(crate::PitchClass::C),
                },
            ),
        )
        .unwrap();
        plan
    }

    #[test]
    fn long_range_promises_can_span_most_of_the_work() {
        let plan = valid_plan();
        assert!(plan.validate(&work_plan(), &thematic_graph()).is_ok());
        let final_promise = &plan.obligations["close-home"];
        assert_eq!(final_promise.created_at, Duration::new(24, 1));
        assert_eq!(final_promise.due.latest, Duration::new(160, 1));
    }

    #[test]
    fn dependency_order_is_stable_and_prerequisite_first() {
        let order = valid_plan()
            .dependency_order(&work_plan(), &thematic_graph())
            .unwrap();
        let pos = |id: &str| order.iter().position(|candidate| candidate == id).unwrap();
        assert!(pos("establish-a") < pos("develop-a"));
        assert!(pos("develop-a") < pos("synthesize-ab"));
        assert!(pos("introduce-b") < pos("synthesize-ab"));
        assert!(pos("synthesize-ab") < pos("close-home"));
    }

    #[test]
    fn due_window_must_fit_its_declared_context() {
        let mut plan = valid_plan();
        plan.obligations.get_mut("close-home").unwrap().due.earliest = Duration::new(140, 1);
        assert!(matches!(
            plan.validate(&work_plan(), &thematic_graph()),
            Err(WorkObligationErrorV2::DueOutsideContext { .. })
        ));
    }

    #[test]
    fn thematic_promise_cannot_be_due_before_identity_exists() {
        let mut plan = valid_plan();
        let obligation = plan.obligations.get_mut("synthesize-ab").unwrap();
        obligation.due_context = "development".into();
        obligation.due = ObligationDueWindowV2 {
            earliest: Duration::new(64, 1),
            latest: Duration::new(80, 1),
        };
        assert!(matches!(
            plan.validate(&work_plan(), &thematic_graph()),
            Err(WorkObligationErrorV2::ThematicIdentityNotYetIntroduced { .. })
        ));
    }

    #[test]
    fn prerequisite_latest_due_cannot_follow_dependent_deadline() {
        let mut plan = valid_plan();
        plan.obligations.get_mut("develop-a").unwrap().due.latest = Duration::new(150, 1);
        plan.obligations.get_mut("develop-a").unwrap().due_context = "work".into();
        assert!(matches!(
            plan.validate(&work_plan(), &thematic_graph()),
            Err(WorkObligationErrorV2::PrerequisiteDueTooLate { .. })
        ));
    }

    #[test]
    fn conflicts_must_be_symmetric() {
        let mut plan = valid_plan();
        plan.obligations
            .get_mut("develop-a")
            .unwrap()
            .conflicts_with = vec!["introduce-b".into()];
        assert!(matches!(
            plan.validate(&work_plan(), &thematic_graph()),
            Err(WorkObligationErrorV2::AsymmetricConflict { .. })
        ));

        plan.obligations
            .get_mut("introduce-b")
            .unwrap()
            .conflicts_with = vec!["develop-a".into()];
        assert!(plan.validate(&work_plan(), &thematic_graph()).is_ok());
    }

    #[test]
    fn dependency_and_conflict_cannot_describe_the_same_relationship() {
        let mut plan = valid_plan();
        plan.obligations
            .get_mut("develop-a")
            .unwrap()
            .conflicts_with = vec!["establish-a".into()];
        plan.obligations
            .get_mut("establish-a")
            .unwrap()
            .conflicts_with = vec!["develop-a".into()];
        assert!(matches!(
            plan.validate(&work_plan(), &thematic_graph()),
            Err(WorkObligationErrorV2::PrerequisiteConflict { .. })
        ));
    }

    #[test]
    fn dependency_cycles_fail_closed() {
        let mut plan = valid_plan();
        plan.obligations
            .get_mut("establish-a")
            .unwrap()
            .prerequisites = vec!["close-home".into()];
        // Equalize the cycle participants' latest deadlines so the deadline
        // theorem cannot mask the dependency-cycle theorem.
        for id in ["establish-a", "develop-a", "synthesize-ab"] {
            let obligation = plan.obligations.get_mut(id).unwrap();
            obligation.due_context = "work".into();
            obligation.due.latest = Duration::new(160, 1);
        }
        assert!(matches!(
            plan.validate(&work_plan(), &thematic_graph()),
            Err(WorkObligationErrorV2::DependencyCycle { .. })
        ));
    }

    #[test]
    fn priority_is_exact_and_bounded() {
        let mut plan = valid_plan();
        plan.obligations.get_mut("close-home").unwrap().priority_per_mille = 1001;
        assert!(matches!(
            plan.validate(&work_plan(), &thematic_graph()),
            Err(WorkObligationErrorV2::PriorityOutOfRange { found: 1001, .. })
        ));
    }
}
