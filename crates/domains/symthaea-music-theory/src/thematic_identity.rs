// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Work-scale thematic genealogy.
//!
//! A long work needs to remember not only that material recurs, but how later
//! identities descend from earlier ones. This graph records declared ancestry
//! independently from score-side verification: independent themes may be
//! introduced, a derived identity has exactly one parent, and a synthesis has
//! multiple parents. Transformation classes describe intended relationships;
//! they do not by themselves prove the completed score realizes them or that a
//! listener perceives the relationship.

use crate::rhythm::Duration;
use crate::work_plan::{HierarchicalWorkPlanV1, WorkNodeKindV1, WorkPlanErrorV1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const THEMATIC_IDENTITY_GRAPH_VERSION: &str = "melothaea-thematic-identity-graph-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThematicOriginV1 {
    Independent,
    Derived,
    Synthesis,
}

/// Declared relationship class only. Concrete pitch/rhythm/harmony/orchestration
/// measurements belong to score-side evidence adapters in later tranches.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThematicTransformationClassV1 {
    LiteralReturn,
    Transposition,
    Inversion,
    Retrograde,
    Augmentation,
    Diminution,
    Sequence,
    Fragmentation,
    RhythmicDisplacement,
    MetricRecontextualization,
    HarmonicReinterpretation,
    RegistralMigration,
    Reorchestration,
    Erosion,
    Restoration,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThematicIdentityV1 {
    pub label: Option<String>,
    pub origin: ThematicOriginV1,
    /// Work-plan leaf in which this identity is first declared to exist.
    pub introduced_in: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThematicDerivationV1 {
    pub source_id: String,
    pub target_id: String,
    /// Canonical sorted unique set of declared transformation classes.
    pub transformations: Vec<ThematicTransformationClassV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThematicIdentityGraphV1 {
    pub version: String,
    pub identities: BTreeMap<String, ThematicIdentityV1>,
    /// Stable derivation IDs are map keys so multiple distinct declared edges
    /// between the same pair cannot be confused accidentally.
    pub derivations: BTreeMap<String, ThematicDerivationV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThematicGraphErrorV1 {
    WrongVersion { found: String },
    InvalidWorkPlan(WorkPlanErrorV1),
    EmptyIdentityId,
    EmptyDerivationId,
    EmptyLabel { identity_id: String },
    EmptyIntroductionNode { identity_id: String },
    MissingIntroductionNode { identity_id: String, node_id: String },
    IntroductionMustBeLeaf { identity_id: String, node_id: String },
    EmptySourceId { derivation_id: String },
    EmptyTargetId { derivation_id: String },
    MissingSource { derivation_id: String, source_id: String },
    MissingTarget { derivation_id: String, target_id: String },
    SelfDerivation { derivation_id: String },
    EmptyTransformations { derivation_id: String },
    InvalidOtherTransformation { derivation_id: String },
    NonCanonicalTransformations { derivation_id: String },
    DuplicateSourceTarget { left_id: String, right_id: String },
    IndependentHasIncoming { identity_id: String },
    DerivedNeedsExactlyOneParent { identity_id: String, incoming: usize },
    SynthesisNeedsMultipleParents { identity_id: String, incoming_sources: usize },
    DerivationMovesBackwardInWork {
        derivation_id: String,
        source_node_id: String,
        target_node_id: String,
    },
    Cycle { identity_id: String },
    UnknownIdentity { identity_id: String },
}

impl Default for ThematicIdentityGraphV1 {
    fn default() -> Self {
        Self {
            version: THEMATIC_IDENTITY_GRAPH_VERSION.into(),
            identities: BTreeMap::new(),
            derivations: BTreeMap::new(),
        }
    }
}

impl ThematicIdentityGraphV1 {
    pub fn insert_identity(
        &mut self,
        identity_id: impl Into<String>,
        identity: ThematicIdentityV1,
    ) -> Result<(), ThematicGraphErrorV1> {
        let identity_id = identity_id.into();
        if identity_id.trim().is_empty() {
            return Err(ThematicGraphErrorV1::EmptyIdentityId);
        }
        self.identities.insert(identity_id, identity);
        Ok(())
    }

    pub fn insert_derivation(
        &mut self,
        derivation_id: impl Into<String>,
        derivation: ThematicDerivationV1,
    ) -> Result<(), ThematicGraphErrorV1> {
        let derivation_id = derivation_id.into();
        if derivation_id.trim().is_empty() {
            return Err(ThematicGraphErrorV1::EmptyDerivationId);
        }
        self.derivations.insert(derivation_id, derivation);
        Ok(())
    }

    pub fn validate(
        &self,
        work_plan: &HierarchicalWorkPlanV1,
    ) -> Result<(), ThematicGraphErrorV1> {
        if self.version != THEMATIC_IDENTITY_GRAPH_VERSION {
            return Err(ThematicGraphErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        work_plan
            .validate()
            .map_err(ThematicGraphErrorV1::InvalidWorkPlan)?;

        for (identity_id, identity) in &self.identities {
            if identity_id.trim().is_empty() {
                return Err(ThematicGraphErrorV1::EmptyIdentityId);
            }
            if identity.label.as_ref().is_some_and(|label| label.trim().is_empty()) {
                return Err(ThematicGraphErrorV1::EmptyLabel {
                    identity_id: identity_id.clone(),
                });
            }
            if identity.introduced_in.trim().is_empty() {
                return Err(ThematicGraphErrorV1::EmptyIntroductionNode {
                    identity_id: identity_id.clone(),
                });
            }
            let node = work_plan.nodes.get(&identity.introduced_in).ok_or_else(|| {
                ThematicGraphErrorV1::MissingIntroductionNode {
                    identity_id: identity_id.clone(),
                    node_id: identity.introduced_in.clone(),
                }
            })?;
            if matches!(node.kind, WorkNodeKindV1::Work | WorkNodeKindV1::Movement) {
                return Err(ThematicGraphErrorV1::IntroductionMustBeLeaf {
                    identity_id: identity_id.clone(),
                    node_id: identity.introduced_in.clone(),
                });
            }
        }

        let mut pair_owner: BTreeMap<(&str, &str), &str> = BTreeMap::new();
        let mut incoming: BTreeMap<&str, Vec<(&str, &ThematicDerivationV1)>> = BTreeMap::new();
        let mut outgoing: BTreeMap<&str, Vec<&str>> = BTreeMap::new();

        for (derivation_id, derivation) in &self.derivations {
            if derivation_id.trim().is_empty() {
                return Err(ThematicGraphErrorV1::EmptyDerivationId);
            }
            if derivation.source_id.trim().is_empty() {
                return Err(ThematicGraphErrorV1::EmptySourceId {
                    derivation_id: derivation_id.clone(),
                });
            }
            if derivation.target_id.trim().is_empty() {
                return Err(ThematicGraphErrorV1::EmptyTargetId {
                    derivation_id: derivation_id.clone(),
                });
            }
            let source = self.identities.get(&derivation.source_id).ok_or_else(|| {
                ThematicGraphErrorV1::MissingSource {
                    derivation_id: derivation_id.clone(),
                    source_id: derivation.source_id.clone(),
                }
            })?;
            let target = self.identities.get(&derivation.target_id).ok_or_else(|| {
                ThematicGraphErrorV1::MissingTarget {
                    derivation_id: derivation_id.clone(),
                    target_id: derivation.target_id.clone(),
                }
            })?;
            if derivation.source_id == derivation.target_id {
                return Err(ThematicGraphErrorV1::SelfDerivation {
                    derivation_id: derivation_id.clone(),
                });
            }
            validate_transformations(derivation_id, &derivation.transformations)?;

            let pair = (derivation.source_id.as_str(), derivation.target_id.as_str());
            if let Some(previous) = pair_owner.insert(pair, derivation_id.as_str()) {
                return Err(ThematicGraphErrorV1::DuplicateSourceTarget {
                    left_id: previous.into(),
                    right_id: derivation_id.clone(),
                });
            }

            let source_node = work_plan
                .nodes
                .get(&source.introduced_in)
                .expect("identity introduction validated");
            let target_node = work_plan
                .nodes
                .get(&target.introduced_in)
                .expect("identity introduction validated");
            if compare_duration(target_node.start, source_node.start) == Ordering::Less {
                return Err(ThematicGraphErrorV1::DerivationMovesBackwardInWork {
                    derivation_id: derivation_id.clone(),
                    source_node_id: source.introduced_in.clone(),
                    target_node_id: target.introduced_in.clone(),
                });
            }

            incoming
                .entry(derivation.target_id.as_str())
                .or_default()
                .push((derivation_id.as_str(), derivation));
            outgoing
                .entry(derivation.source_id.as_str())
                .or_default()
                .push(derivation.target_id.as_str());
        }

        for (identity_id, identity) in &self.identities {
            let incoming_edges = incoming
                .get(identity_id.as_str())
                .map_or(&[][..], |edges| edges.as_slice());
            let distinct_sources: BTreeSet<_> = incoming_edges
                .iter()
                .map(|(_, edge)| edge.source_id.as_str())
                .collect();
            match identity.origin {
                ThematicOriginV1::Independent if !incoming_edges.is_empty() => {
                    return Err(ThematicGraphErrorV1::IndependentHasIncoming {
                        identity_id: identity_id.clone(),
                    });
                }
                ThematicOriginV1::Derived if distinct_sources.len() != 1 => {
                    return Err(ThematicGraphErrorV1::DerivedNeedsExactlyOneParent {
                        identity_id: identity_id.clone(),
                        incoming: distinct_sources.len(),
                    });
                }
                ThematicOriginV1::Synthesis if distinct_sources.len() < 2 => {
                    return Err(ThematicGraphErrorV1::SynthesisNeedsMultipleParents {
                        identity_id: identity_id.clone(),
                        incoming_sources: distinct_sources.len(),
                    });
                }
                _ => {}
            }
        }

        // Genealogy must be acyclic even when several identities are introduced
        // inside the same work node.
        for identity_id in self.identities.keys() {
            let mut visiting = BTreeSet::new();
            let mut visited = BTreeSet::new();
            if has_cycle(
                identity_id,
                &outgoing,
                &mut visiting,
                &mut visited,
            ) {
                return Err(ThematicGraphErrorV1::Cycle {
                    identity_id: identity_id.clone(),
                });
            }
        }
        Ok(())
    }

    /// All transitive ancestors, returned in stable identity order.
    pub fn ancestors_of(
        &self,
        work_plan: &HierarchicalWorkPlanV1,
        identity_id: &str,
    ) -> Result<Vec<String>, ThematicGraphErrorV1> {
        self.validate(work_plan)?;
        if !self.identities.contains_key(identity_id) {
            return Err(ThematicGraphErrorV1::UnknownIdentity {
                identity_id: identity_id.into(),
            });
        }
        let mut reverse: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for derivation in self.derivations.values() {
            reverse
                .entry(derivation.target_id.as_str())
                .or_default()
                .push(derivation.source_id.as_str());
        }
        let mut ancestors = BTreeSet::new();
        let mut stack = reverse.get(identity_id).cloned().unwrap_or_default();
        while let Some(current) = stack.pop() {
            if ancestors.insert(current.to_string()) {
                if let Some(parents) = reverse.get(current) {
                    stack.extend(parents.iter().copied());
                }
            }
        }
        Ok(ancestors.into_iter().collect())
    }
}

fn validate_transformations(
    derivation_id: &str,
    transformations: &[ThematicTransformationClassV1],
) -> Result<(), ThematicGraphErrorV1> {
    if transformations.is_empty() {
        return Err(ThematicGraphErrorV1::EmptyTransformations {
            derivation_id: derivation_id.into(),
        });
    }
    for transformation in transformations {
        if let ThematicTransformationClassV1::Other(label) = transformation
            && label.trim().is_empty()
        {
            return Err(ThematicGraphErrorV1::InvalidOtherTransformation {
                derivation_id: derivation_id.into(),
            });
        }
    }
    if transformations.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(ThematicGraphErrorV1::NonCanonicalTransformations {
            derivation_id: derivation_id.into(),
        });
    }
    Ok(())
}

fn has_cycle<'a>(
    current: &'a str,
    outgoing: &BTreeMap<&'a str, Vec<&'a str>>,
    visiting: &mut BTreeSet<&'a str>,
    visited: &mut BTreeSet<&'a str>,
) -> bool {
    if visited.contains(current) {
        return false;
    }
    if !visiting.insert(current) {
        return true;
    }
    if let Some(children) = outgoing.get(current) {
        for child in children {
            if has_cycle(child, outgoing, visiting, visited) {
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
    use crate::{FormalFunctionV1, WorkNodeV1};

    fn work_plan() -> HierarchicalWorkPlanV1 {
        let mut plan = HierarchicalWorkPlanV1::new("work", Duration::new(160, 1)).unwrap();
        for (id, start, end, functions) in [
            ("a", 0, 32, vec![FormalFunctionV1::Establish]),
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
            let kind = if id == "coda" {
                WorkNodeKindV1::Coda
            } else {
                WorkNodeKindV1::Section
            };
            plan.insert_node(
                id,
                WorkNodeV1 {
                    parent_id: Some("work".into()),
                    label: None,
                    kind,
                    start: Duration::new(start, 1),
                    end: Duration::new(end, 1),
                    functions,
                },
            )
            .unwrap();
        }
        plan
    }

    fn identity(origin: ThematicOriginV1, introduced_in: &str) -> ThematicIdentityV1 {
        ThematicIdentityV1 {
            label: None,
            origin,
            introduced_in: introduced_in.into(),
        }
    }

    fn edge(source: &str, target: &str, class: ThematicTransformationClassV1) -> ThematicDerivationV1 {
        ThematicDerivationV1 {
            source_id: source.into(),
            target_id: target.into(),
            transformations: vec![class],
        }
    }

    fn valid_graph() -> ThematicIdentityGraphV1 {
        let mut graph = ThematicIdentityGraphV1::default();
        graph
            .insert_identity("A", identity(ThematicOriginV1::Independent, "a"))
            .unwrap();
        graph
            .insert_identity("B", identity(ThematicOriginV1::Independent, "development"))
            .unwrap();
        graph
            .insert_identity("A1", identity(ThematicOriginV1::Derived, "development"))
            .unwrap();
        graph
            .insert_identity("AB", identity(ThematicOriginV1::Synthesis, "return"))
            .unwrap();
        graph
            .insert_identity("AB-final", identity(ThematicOriginV1::Derived, "coda"))
            .unwrap();
        graph
            .insert_derivation(
                "a-to-a1",
                edge("A", "A1", ThematicTransformationClassV1::Fragmentation),
            )
            .unwrap();
        graph
            .insert_derivation(
                "a1-to-ab",
                edge(
                    "A1",
                    "AB",
                    ThematicTransformationClassV1::HarmonicReinterpretation,
                ),
            )
            .unwrap();
        graph
            .insert_derivation(
                "b-to-ab",
                edge("B", "AB", ThematicTransformationClassV1::Reorchestration),
            )
            .unwrap();
        graph
            .insert_derivation(
                "ab-to-final",
                edge(
                    "AB",
                    "AB-final",
                    ThematicTransformationClassV1::Restoration,
                ),
            )
            .unwrap();
        graph
    }

    #[test]
    fn independent_derived_and_synthesis_genealogy_is_distinct() {
        let graph = valid_graph();
        let plan = work_plan();
        assert!(graph.validate(&plan).is_ok());
        assert_eq!(graph.identities["A"].origin, ThematicOriginV1::Independent);
        assert_eq!(graph.identities["A1"].origin, ThematicOriginV1::Derived);
        assert_eq!(graph.identities["AB"].origin, ThematicOriginV1::Synthesis);
    }

    #[test]
    fn final_identity_retains_transitive_multi_parent_ancestry() {
        let graph = valid_graph();
        let ancestors = graph.ancestors_of(&work_plan(), "AB-final").unwrap();
        assert_eq!(ancestors, vec!["A", "A1", "AB", "B"]);
    }

    #[test]
    fn derived_identity_requires_exactly_one_distinct_parent() {
        let mut graph = valid_graph();
        graph
            .insert_derivation(
                "b-to-a1",
                edge("B", "A1", ThematicTransformationClassV1::Sequence),
            )
            .unwrap();
        assert!(matches!(
            graph.validate(&work_plan()),
            Err(ThematicGraphErrorV1::DerivedNeedsExactlyOneParent {
                identity_id,
                incoming: 2
            }) if identity_id == "A1"
        ));
    }

    #[test]
    fn synthesis_requires_multiple_distinct_parents() {
        let mut graph = valid_graph();
        graph.derivations.remove("b-to-ab");
        assert!(matches!(
            graph.validate(&work_plan()),
            Err(ThematicGraphErrorV1::SynthesisNeedsMultipleParents {
                identity_id,
                incoming_sources: 1
            }) if identity_id == "AB"
        ));
    }

    #[test]
    fn independent_identity_cannot_secretly_have_a_parent() {
        let mut graph = valid_graph();
        graph
            .insert_derivation(
                "a1-to-b",
                edge("A1", "B", ThematicTransformationClassV1::Restoration),
            )
            .unwrap();
        assert!(matches!(
            graph.validate(&work_plan()),
            Err(ThematicGraphErrorV1::IndependentHasIncoming { identity_id }) if identity_id == "B"
        ));
    }

    #[test]
    fn ancestry_cannot_move_backward_in_the_work() {
        let mut graph = valid_graph();
        graph.identities.get_mut("A1").unwrap().introduced_in = "a".into();
        graph.identities.get_mut("A").unwrap().introduced_in = "development".into();
        assert!(matches!(
            graph.validate(&work_plan()),
            Err(ThematicGraphErrorV1::DerivationMovesBackwardInWork { .. })
        ));
    }

    #[test]
    fn transformation_classes_are_nonempty_unique_and_canonical() {
        let mut graph = valid_graph();
        graph.derivations.get_mut("a-to-a1").unwrap().transformations = vec![];
        assert!(matches!(
            graph.validate(&work_plan()),
            Err(ThematicGraphErrorV1::EmptyTransformations { .. })
        ));

        let mut graph = valid_graph();
        graph.derivations.get_mut("a-to-a1").unwrap().transformations = vec![
            ThematicTransformationClassV1::Restoration,
            ThematicTransformationClassV1::Fragmentation,
        ];
        assert!(matches!(
            graph.validate(&work_plan()),
            Err(ThematicGraphErrorV1::NonCanonicalTransformations { .. })
        ));
    }

    #[test]
    fn introduction_is_bound_to_a_work_plan_leaf() {
        let mut graph = valid_graph();
        graph.identities.get_mut("A").unwrap().introduced_in = "work".into();
        assert!(matches!(
            graph.validate(&work_plan()),
            Err(ThematicGraphErrorV1::IntroductionMustBeLeaf { .. })
        ));
    }

    #[test]
    fn same_node_genealogy_still_cannot_cycle() {
        let mut graph = ThematicIdentityGraphV1::default();
        graph
            .insert_identity("X", identity(ThematicOriginV1::Derived, "development"))
            .unwrap();
        graph
            .insert_identity("Y", identity(ThematicOriginV1::Derived, "development"))
            .unwrap();
        graph
            .insert_derivation(
                "x-y",
                edge("X", "Y", ThematicTransformationClassV1::Inversion),
            )
            .unwrap();
        graph
            .insert_derivation(
                "y-x",
                edge("Y", "X", ThematicTransformationClassV1::Retrograde),
            )
            .unwrap();
        assert!(matches!(
            graph.validate(&work_plan()),
            Err(ThematicGraphErrorV1::Cycle { .. })
        ));
    }
}
