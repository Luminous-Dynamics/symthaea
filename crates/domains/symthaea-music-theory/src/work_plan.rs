// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Work-scale formal hierarchy above existing local form engines.
//!
//! `Form`, `SonataPlan`, fugue, passacaglia, and other existing engines remain
//! useful local structures. `HierarchicalWorkPlanV1` does not replace them; it
//! defines the larger temporal/formal tree that can contain movements, sections,
//! transitions, interludes, and codas across an entire work.
//!
//! V1 is intentionally structural. Thematic ancestry, compositional obligation
//! dependencies, orchestration arcs, and artistic-quality claims belong to later
//! contracts so this hierarchy can be reused across genres without importing one
//! style's assumptions.

use crate::rhythm::Duration;
use crate::temporal_score::{TemporalScoreErrorV1, TemporalScoreV1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const HIERARCHICAL_WORK_PLAN_VERSION: &str = "melothaea-hierarchical-work-plan-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WorkNodeKindV1 {
    Work,
    Movement,
    Section,
    Transition,
    Interlude,
    Coda,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FormalFunctionV1 {
    Establish,
    Contrast,
    Develop,
    Destabilize,
    Transition,
    Return,
    Synthesize,
    Resolve,
    Close,
    /// Extensibility without pretending an unfamiliar formal function belongs
    /// to one of the predefined Western labels above.
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkNodeV1 {
    pub parent_id: Option<String>,
    /// Human-facing name only; stable identity is the map key containing this node.
    pub label: Option<String>,
    pub kind: WorkNodeKindV1,
    /// Exact half-open span `[start, end)` in quarter-note beats from work start.
    pub start: Duration,
    pub end: Duration,
    /// Ordered canonically by enum/string ordering. These describe intended
    /// formal function; they are not evidence that listeners perceive it.
    pub functions: Vec<FormalFunctionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HierarchicalWorkPlanV1 {
    pub version: String,
    pub root_id: String,
    /// Stable node identities are map keys. BTreeMap gives deterministic key order.
    pub nodes: BTreeMap<String, WorkNodeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkPlanErrorV1 {
    WrongVersion { found: String },
    EmptyRootId,
    MissingRoot,
    RootMustBeWork,
    RootMustHaveNoParent,
    AdditionalRoot { node_id: String },
    EmptyNodeId,
    EmptyLabel { node_id: String },
    InvalidOtherFunction { node_id: String },
    NonCanonicalFunctions { node_id: String },
    NegativeStart { node_id: String },
    NonPositiveSpan { node_id: String },
    MissingParent { node_id: String, parent_id: String },
    SelfParent { node_id: String },
    Cycle { node_id: String },
    ChildOutsideParent { node_id: String, parent_id: String },
    LeafHasChildren { node_id: String },
    MovementParentMustBeWork { node_id: String },
    InvalidLeafParent { node_id: String },
    SiblingOverlap {
        parent_id: String,
        left_id: String,
        right_id: String,
    },
    RootMustStartAtZero,
    RootEndMismatch,
    InvalidTemporalScore(TemporalScoreErrorV1),
}

impl HierarchicalWorkPlanV1 {
    pub fn new(root_id: impl Into<String>, end: Duration) -> Result<Self, WorkPlanErrorV1> {
        let root_id = root_id.into();
        if root_id.trim().is_empty() {
            return Err(WorkPlanErrorV1::EmptyRootId);
        }
        let root = WorkNodeV1 {
            parent_id: None,
            label: None,
            kind: WorkNodeKindV1::Work,
            start: Duration::zero(),
            end,
            functions: Vec::new(),
        };
        let mut nodes = BTreeMap::new();
        nodes.insert(root_id.clone(), root);
        let plan = Self {
            version: HIERARCHICAL_WORK_PLAN_VERSION.into(),
            root_id,
            nodes,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn insert_node(
        &mut self,
        node_id: impl Into<String>,
        node: WorkNodeV1,
    ) -> Result<(), WorkPlanErrorV1> {
        let node_id = node_id.into();
        if node_id.trim().is_empty() {
            return Err(WorkPlanErrorV1::EmptyNodeId);
        }
        let previous = self.nodes.insert(node_id.clone(), node);
        if let Err(error) = self.validate() {
            match previous {
                Some(previous) => {
                    self.nodes.insert(node_id, previous);
                }
                None => {
                    self.nodes.remove(&node_id);
                }
            }
            return Err(error);
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), WorkPlanErrorV1> {
        if self.version != HIERARCHICAL_WORK_PLAN_VERSION {
            return Err(WorkPlanErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.root_id.trim().is_empty() {
            return Err(WorkPlanErrorV1::EmptyRootId);
        }
        let root = self
            .nodes
            .get(&self.root_id)
            .ok_or(WorkPlanErrorV1::MissingRoot)?;
        if root.kind != WorkNodeKindV1::Work {
            return Err(WorkPlanErrorV1::RootMustBeWork);
        }
        if root.parent_id.is_some() {
            return Err(WorkPlanErrorV1::RootMustHaveNoParent);
        }
        if root.start != Duration::zero() {
            return Err(WorkPlanErrorV1::RootMustStartAtZero);
        }

        let mut children: BTreeMap<&str, Vec<(&str, &WorkNodeV1)>> = BTreeMap::new();
        for (node_id, node) in &self.nodes {
            if node_id.trim().is_empty() {
                return Err(WorkPlanErrorV1::EmptyNodeId);
            }
            if node_id != &self.root_id && node.parent_id.is_none() {
                return Err(WorkPlanErrorV1::AdditionalRoot {
                    node_id: node_id.clone(),
                });
            }
            if node.label.as_ref().is_some_and(|label| label.trim().is_empty()) {
                return Err(WorkPlanErrorV1::EmptyLabel {
                    node_id: node_id.clone(),
                });
            }
            validate_functions(node_id, &node.functions)?;
            if node.start.num() < 0 {
                return Err(WorkPlanErrorV1::NegativeStart {
                    node_id: node_id.clone(),
                });
            }
            if compare_duration(node.end, node.start) != Ordering::Greater {
                return Err(WorkPlanErrorV1::NonPositiveSpan {
                    node_id: node_id.clone(),
                });
            }

            if let Some(parent_id) = node.parent_id.as_deref() {
                if parent_id == node_id {
                    return Err(WorkPlanErrorV1::SelfParent {
                        node_id: node_id.clone(),
                    });
                }
                let parent = self.nodes.get(parent_id).ok_or_else(|| {
                    WorkPlanErrorV1::MissingParent {
                        node_id: node_id.clone(),
                        parent_id: parent_id.into(),
                    }
                })?;
                if compare_duration(node.start, parent.start) == Ordering::Less
                    || compare_duration(node.end, parent.end) == Ordering::Greater
                {
                    return Err(WorkPlanErrorV1::ChildOutsideParent {
                        node_id: node_id.clone(),
                        parent_id: parent_id.into(),
                    });
                }
                match node.kind {
                    WorkNodeKindV1::Work => {
                        return Err(WorkPlanErrorV1::AdditionalRoot {
                            node_id: node_id.clone(),
                        });
                    }
                    WorkNodeKindV1::Movement if parent.kind != WorkNodeKindV1::Work => {
                        return Err(WorkPlanErrorV1::MovementParentMustBeWork {
                            node_id: node_id.clone(),
                        });
                    }
                    WorkNodeKindV1::Section
                    | WorkNodeKindV1::Transition
                    | WorkNodeKindV1::Interlude
                    | WorkNodeKindV1::Coda
                        if !matches!(
                            parent.kind,
                            WorkNodeKindV1::Work | WorkNodeKindV1::Movement
                        ) =>
                    {
                        return Err(WorkPlanErrorV1::InvalidLeafParent {
                            node_id: node_id.clone(),
                        });
                    }
                    _ => {}
                }
                children
                    .entry(parent_id)
                    .or_default()
                    .push((node_id.as_str(), node));
            }
        }

        // Parent chains must terminate at the declared root. Earlier structural
        // checks may reject some malformed graphs before this defensive cycle
        // walk; either path is a valid fail-closed outcome.
        for node_id in self.nodes.keys() {
            let mut seen = BTreeSet::new();
            let mut cursor = node_id.as_str();
            loop {
                if !seen.insert(cursor.to_string()) {
                    return Err(WorkPlanErrorV1::Cycle {
                        node_id: node_id.clone(),
                    });
                }
                let node = self.nodes.get(cursor).expect("validated map key");
                match node.parent_id.as_deref() {
                    Some(parent_id) => cursor = parent_id,
                    None if cursor == self.root_id => break,
                    None => {
                        return Err(WorkPlanErrorV1::AdditionalRoot {
                            node_id: cursor.into(),
                        });
                    }
                }
            }
        }

        // In V1 only Work and Movement may contain children. Section-level
        // subphrase hierarchy can be added later without making it implicit now.
        for (parent_id, siblings) in &mut children {
            let parent = self.nodes.get(*parent_id).expect("known parent");
            if !matches!(parent.kind, WorkNodeKindV1::Work | WorkNodeKindV1::Movement) {
                return Err(WorkPlanErrorV1::LeafHasChildren {
                    node_id: (*parent_id).into(),
                });
            }
            siblings.sort_by(|left, right| {
                compare_duration(left.1.start, right.1.start)
                    .then_with(|| compare_duration(left.1.end, right.1.end))
                    .then_with(|| left.0.cmp(right.0))
            });
            for pair in siblings.windows(2) {
                let (left_id, left) = pair[0];
                let (right_id, right) = pair[1];
                if compare_duration(left.end, right.start) == Ordering::Greater {
                    return Err(WorkPlanErrorV1::SiblingOverlap {
                        parent_id: (*parent_id).into(),
                        left_id: left_id.into(),
                        right_id: right_id.into(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Stronger validation when the plan is bound to a concrete work score.
    pub fn validate_for_temporal_score(
        &self,
        temporal_score: &TemporalScoreV1,
    ) -> Result<(), WorkPlanErrorV1> {
        self.validate()?;
        temporal_score
            .validate()
            .map_err(WorkPlanErrorV1::InvalidTemporalScore)?;
        let root = self.nodes.get(&self.root_id).expect("validated root");
        if root.end != temporal_score.score.total_beats {
            return Err(WorkPlanErrorV1::RootEndMismatch);
        }
        Ok(())
    }

    /// Direct children in exact chronological order, independent of map key order.
    pub fn children_of(&self, parent_id: &str) -> Result<Vec<(&str, &WorkNodeV1)>, WorkPlanErrorV1> {
        self.validate()?;
        if !self.nodes.contains_key(parent_id) {
            return Err(WorkPlanErrorV1::MissingParent {
                node_id: parent_id.into(),
                parent_id: parent_id.into(),
            });
        }
        let mut children: Vec<_> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.parent_id.as_deref() == Some(parent_id))
            .map(|(id, node)| (id.as_str(), node))
            .collect();
        children.sort_by(|left, right| {
            compare_duration(left.1.start, right.1.start)
                .then_with(|| compare_duration(left.1.end, right.1.end))
                .then_with(|| left.0.cmp(right.0))
        });
        Ok(children)
    }
}

fn validate_functions(node_id: &str, functions: &[FormalFunctionV1]) -> Result<(), WorkPlanErrorV1> {
    for function in functions {
        if let FormalFunctionV1::Other(label) = function
            && label.trim().is_empty()
        {
            return Err(WorkPlanErrorV1::InvalidOtherFunction {
                node_id: node_id.into(),
            });
        }
    }
    if functions.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(WorkPlanErrorV1::NonCanonicalFunctions {
            node_id: node_id.into(),
        });
    }
    Ok(())
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Emphasis, Key, PartId, Pitch, PitchClass, Score, ScoreNote, TempoV1,
        TemporalMapV1, TimeSignature, VoiceRole,
    };

    fn node(
        parent: &str,
        kind: WorkNodeKindV1,
        start: i64,
        end: i64,
        functions: Vec<FormalFunctionV1>,
    ) -> WorkNodeV1 {
        WorkNodeV1 {
            parent_id: Some(parent.into()),
            label: None,
            kind,
            start: Duration::new(start, 1),
            end: Duration::new(end, 1),
            functions,
        }
    }

    fn multi_movement_plan() -> HierarchicalWorkPlanV1 {
        let mut plan = HierarchicalWorkPlanV1::new("work", Duration::new(160, 1)).unwrap();
        plan.insert_node(
            "m1",
            node(
                "work",
                WorkNodeKindV1::Movement,
                0,
                80,
                vec![FormalFunctionV1::Establish, FormalFunctionV1::Develop],
            ),
        )
        .unwrap();
        plan.insert_node(
            "m2",
            node(
                "work",
                WorkNodeKindV1::Movement,
                80,
                160,
                vec![FormalFunctionV1::Return, FormalFunctionV1::Resolve],
            ),
        )
        .unwrap();
        plan.insert_node(
            "m1-a",
            node(
                "m1",
                WorkNodeKindV1::Section,
                0,
                32,
                vec![FormalFunctionV1::Establish],
            ),
        )
        .unwrap();
        plan.insert_node(
            "m1-dev",
            node(
                "m1",
                WorkNodeKindV1::Section,
                32,
                80,
                vec![FormalFunctionV1::Develop, FormalFunctionV1::Destabilize],
            ),
        )
        .unwrap();
        plan.insert_node(
            "m2-return",
            node(
                "m2",
                WorkNodeKindV1::Section,
                80,
                144,
                vec![FormalFunctionV1::Return, FormalFunctionV1::Synthesize],
            ),
        )
        .unwrap();
        plan.insert_node(
            "m2-coda",
            node(
                "m2",
                WorkNodeKindV1::Coda,
                144,
                160,
                vec![FormalFunctionV1::Resolve, FormalFunctionV1::Close],
            ),
        )
        .unwrap();
        plan
    }

    #[test]
    fn supports_nested_work_movement_section_hierarchy() {
        let plan = multi_movement_plan();
        assert!(plan.validate().is_ok());
        let root_children = plan.children_of("work").unwrap();
        assert_eq!(
            root_children.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec!["m1", "m2"]
        );
        let movement_children = plan.children_of("m2").unwrap();
        assert_eq!(
            movement_children
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            vec!["m2-return", "m2-coda"]
        );
    }

    #[test]
    fn functions_are_explicit_intent_not_a_quality_score() {
        let plan = multi_movement_plan();
        let node = plan.nodes.get("m1-dev").unwrap();
        assert_eq!(
            node.functions,
            vec![FormalFunctionV1::Develop, FormalFunctionV1::Destabilize]
        );
    }

    #[test]
    fn child_must_stay_inside_parent_span() {
        let mut plan = HierarchicalWorkPlanV1::new("work", Duration::new(100, 1)).unwrap();
        plan.insert_node(
            "movement",
            node(
                "work",
                WorkNodeKindV1::Movement,
                10,
                80,
                vec![FormalFunctionV1::Develop],
            ),
        )
        .unwrap();
        assert!(matches!(
            plan.insert_node(
                "outside",
                node(
                    "movement",
                    WorkNodeKindV1::Section,
                    0,
                    20,
                    vec![FormalFunctionV1::Establish],
                ),
            ),
            Err(WorkPlanErrorV1::ChildOutsideParent { .. })
        ));
    }

    #[test]
    fn overlapping_siblings_fail_closed_but_gaps_are_allowed() {
        let mut plan = HierarchicalWorkPlanV1::new("work", Duration::new(100, 1)).unwrap();
        plan.insert_node(
            "a",
            node(
                "work",
                WorkNodeKindV1::Section,
                0,
                20,
                vec![FormalFunctionV1::Establish],
            ),
        )
        .unwrap();
        plan.insert_node(
            "b",
            node(
                "work",
                WorkNodeKindV1::Transition,
                30,
                40,
                vec![FormalFunctionV1::Transition],
            ),
        )
        .unwrap();
        assert!(matches!(
            plan.insert_node(
                "overlap",
                node(
                    "work",
                    WorkNodeKindV1::Section,
                    15,
                    35,
                    vec![FormalFunctionV1::Contrast],
                ),
            ),
            Err(WorkPlanErrorV1::SiblingOverlap { .. })
        ));
    }

    #[test]
    fn leaf_nodes_cannot_hide_an_unmodeled_subhierarchy() {
        let mut plan = HierarchicalWorkPlanV1::new("work", Duration::new(100, 1)).unwrap();
        plan.insert_node(
            "section",
            node(
                "work",
                WorkNodeKindV1::Section,
                0,
                50,
                vec![FormalFunctionV1::Establish],
            ),
        )
        .unwrap();
        assert!(matches!(
            plan.insert_node(
                "child",
                node(
                    "section",
                    WorkNodeKindV1::Section,
                    0,
                    10,
                    vec![FormalFunctionV1::Develop],
                ),
            ),
            Err(WorkPlanErrorV1::InvalidLeafParent { .. })
                | Err(WorkPlanErrorV1::LeafHasChildren { .. })
        ));
    }

    #[test]
    fn formal_functions_must_be_unique_and_canonical() {
        let mut plan = HierarchicalWorkPlanV1::new("work", Duration::new(20, 1)).unwrap();
        assert!(matches!(
            plan.insert_node(
                "bad",
                node(
                    "work",
                    WorkNodeKindV1::Section,
                    0,
                    10,
                    vec![FormalFunctionV1::Return, FormalFunctionV1::Establish],
                ),
            ),
            Err(WorkPlanErrorV1::NonCanonicalFunctions { .. })
        ));
        assert!(matches!(
            plan.insert_node(
                "bad-other",
                node(
                    "work",
                    WorkNodeKindV1::Section,
                    0,
                    10,
                    vec![FormalFunctionV1::Other("  ".into())],
                ),
            ),
            Err(WorkPlanErrorV1::InvalidOtherFunction { .. })
        ));
    }

    #[test]
    fn root_span_must_match_bound_temporal_score() {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: Pitch::from_midi(60),
            onset: Duration::zero(),
            duration: Duration::new(160, 1),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        let map = TemporalMapV1::new(
            TempoV1::integer(120).unwrap(),
            TimeSignature::new(4, 4).unwrap(),
        );
        let temporal = TemporalScoreV1::bind(score, map).unwrap();
        assert!(multi_movement_plan()
            .validate_for_temporal_score(&temporal)
            .is_ok());

        let short = HierarchicalWorkPlanV1::new("work", Duration::new(159, 1)).unwrap();
        assert_eq!(
            short.validate_for_temporal_score(&temporal),
            Err(WorkPlanErrorV1::RootEndMismatch)
        );
    }

    #[test]
    fn malformed_parent_cycle_fails_closed() {
        let mut plan = multi_movement_plan();
        plan.nodes.get_mut("m1").unwrap().parent_id = Some("m2".into());
        plan.nodes.get_mut("m2").unwrap().parent_id = Some("m1".into());
        assert!(plan.validate().is_err());
    }
}
