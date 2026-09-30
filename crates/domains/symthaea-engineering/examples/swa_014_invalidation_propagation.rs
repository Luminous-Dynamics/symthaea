//! SWA-014 — deterministic invalidation propagation.
//!
//! A dependency revision changes the *current applicability* of downstream
//! evidence. It does not rewrite the historical evidence that was produced
//! under the previous revision.
//!
//! SWA-014 turns the revalidation frontier into a propagation record that can
//! reach claims, replay witnesses, counterevidence branches, and decision
//! contexts while preserving their historical identities.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
enum DependencyKind {
    Model,
    Parameters,
    Solver,
    Scenario,
    Dataset,
    UncertaintyModel,
    ContextOfUse,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct DependencyChange {
    kind: DependencyKind,
    from: u64,
    to: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct DependencyFingerprint {
    revisions: Vec<(DependencyKind, u64)>,
}

impl DependencyFingerprint {
    fn revision(model: u64, parameters: u64, solver: u64, scenario: u64, dataset: u64, uncertainty: u64, context: u64) -> Self {
        Self {
            revisions: vec![
                (DependencyKind::Model, model),
                (DependencyKind::Parameters, parameters),
                (DependencyKind::Solver, solver),
                (DependencyKind::Scenario, scenario),
                (DependencyKind::Dataset, dataset),
                (DependencyKind::UncertaintyModel, uncertainty),
                (DependencyKind::ContextOfUse, context),
            ],
        }
    }

    fn get(&self, kind: &DependencyKind) -> Option<u64> {
        self.revisions.iter().find(|(k, _)| k == kind).map(|(_, revision)| *revision)
    }

    fn changes_to(&self, current: &Self) -> Option<Vec<DependencyChange>> {
        let mut changes = Vec::new();
        for (kind, historical) in &self.revisions {
            let current_revision = current.get(kind)?;
            if *historical != current_revision {
                changes.push(DependencyChange {
                    kind: kind.clone(),
                    from: *historical,
                    to: current_revision,
                });
            }
        }
        Some(changes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum NodeKind {
    Evidence,
    Claim,
    ReplayWitness,
    CounterevidenceBranch,
    DecisionContext,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct PropagationNode {
    id: String,
    kind: NodeKind,
    historical_dependencies: DependencyFingerprint,
    closure: Vec<DependencyKind>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum CurrentStatus {
    Current,
    RequiresRevalidation,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct PropagationEntry {
    node_id: String,
    node_kind: NodeKind,
    status: CurrentStatus,
    affected_dependencies: Vec<DependencyChange>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct InvalidationPropagation {
    source_changes: Vec<DependencyChange>,
    entries: Vec<PropagationEntry>,
}

impl InvalidationPropagation {
    fn compute(nodes: &[PropagationNode], current: &DependencyFingerprint) -> Self {
        let mut source_changes = BTreeSet::new();
        let mut entries = nodes
            .iter()
            .map(|node| {
                let changes = node.historical_dependencies.changes_to(current);
                let (status, affected) = match changes {
                    None => (CurrentStatus::Unknown, Vec::new()),
                    Some(all) => {
                        let relevant = all
                            .into_iter()
                            .filter(|change| node.closure.contains(&change.kind))
                            .collect::<Vec<_>>();
                        if relevant.is_empty() {
                            (CurrentStatus::Current, Vec::new())
                        } else {
                            for change in &relevant {
                                source_changes.insert(change.clone());
                            }
                            (CurrentStatus::RequiresRevalidation, relevant)
                        }
                    }
                };

                PropagationEntry {
                    node_id: node.id.clone(),
                    node_kind: node.kind.clone(),
                    status,
                    affected_dependencies: affected,
                }
            })
            .collect::<Vec<_>>();

        entries.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        Self {
            source_changes: source_changes.into_iter().collect(),
            entries,
        }
    }

    fn affected_ids(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|entry| entry.status == CurrentStatus::RequiresRevalidation)
            .map(|entry| entry.node_id.as_str())
            .collect()
    }
}

fn main() {
    let historical = DependencyFingerprint::revision(7, 3, 4, 2, 8, 5, 1);
    let current = DependencyFingerprint::revision(8, 3, 4, 2, 8, 5, 1);

    let nodes = vec![
        PropagationNode {
            id: "claim-thermal".into(),
            kind: NodeKind::Claim,
            historical_dependencies: historical.clone(),
            closure: vec![DependencyKind::Model, DependencyKind::Parameters, DependencyKind::Scenario],
        },
        PropagationNode {
            id: "counterevidence-thermal".into(),
            kind: NodeKind::CounterevidenceBranch,
            historical_dependencies: historical.clone(),
            closure: vec![DependencyKind::Model, DependencyKind::Parameters, DependencyKind::Scenario],
        },
        PropagationNode {
            id: "decision-context-thermal".into(),
            kind: NodeKind::DecisionContext,
            historical_dependencies: historical.clone(),
            closure: vec![DependencyKind::Model, DependencyKind::ContextOfUse],
        },
        PropagationNode {
            id: "witness-thermal".into(),
            kind: NodeKind::ReplayWitness,
            historical_dependencies: historical.clone(),
            closure: vec![DependencyKind::Model, DependencyKind::Parameters, DependencyKind::Solver],
        },
    ];

    let propagation = InvalidationPropagation::compute(&nodes, &current);
    assert_eq!(propagation.affected_ids(), vec![
        "claim-thermal",
        "counterevidence-thermal",
        "decision-context-thermal",
        "witness-thermal",
    ]);

    // The historical dependency records remain untouched.
    assert_eq!(nodes[0].historical_dependencies.get(&DependencyKind::Model), Some(7));

    println!("{}", serde_json::to_string_pretty(&propagation).expect("serialize propagation"));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, kind: NodeKind, closure: Vec<DependencyKind>) -> PropagationNode {
        PropagationNode {
            id: id.into(),
            kind,
            historical_dependencies: DependencyFingerprint::revision(7, 3, 4, 2, 8, 5, 1),
            closure,
        }
    }

    #[test]
    fn model_revision_propagates_only_to_dependent_nodes() {
        let nodes = vec![
            node(
                "claim",
                NodeKind::Claim,
                vec![DependencyKind::Model, DependencyKind::Parameters],
            ),
            node(
                "resilience",
                NodeKind::Evidence,
                vec![DependencyKind::Parameters],
            ),
        ];
        let current = DependencyFingerprint::revision(8, 3, 4, 2, 8, 5, 1);
        let result = InvalidationPropagation::compute(&nodes, &current);

        assert_eq!(result.entries[0].status, CurrentStatus::RequiresRevalidation);
        assert_eq!(result.entries[1].status, CurrentStatus::Current);
    }

    #[test]
    fn replay_witness_is_revalidation_sensitive_to_model_changes() {
        let nodes = vec![node(
            "witness",
            NodeKind::ReplayWitness,
            vec![DependencyKind::Model, DependencyKind::Solver],
        )];
        let current = DependencyFingerprint::revision(8, 3, 4, 2, 8, 5, 1);
        let result = InvalidationPropagation::compute(&nodes, &current);

        assert_eq!(result.entries[0].status, CurrentStatus::RequiresRevalidation);
        assert_eq!(result.entries[0].affected_dependencies[0].kind, DependencyKind::Model);
    }

    #[test]
    fn counterevidence_branch_is_not_deleted_or_resolved() {
        let nodes = vec![node(
            "counter",
            NodeKind::CounterevidenceBranch,
            vec![DependencyKind::Model],
        )];
        let current = DependencyFingerprint::revision(8, 3, 4, 2, 8, 5, 1);
        let result = InvalidationPropagation::compute(&nodes, &current);

        assert_eq!(result.entries[0].status, CurrentStatus::RequiresRevalidation);
        assert_eq!(result.entries[0].node_kind, NodeKind::CounterevidenceBranch);
    }

    #[test]
    fn context_only_change_does_not_stale_model_only_claim() {
        let nodes = vec![node(
            "claim",
            NodeKind::Claim,
            vec![DependencyKind::Model],
        )];
        let current = DependencyFingerprint::revision(7, 3, 4, 2, 8, 5, 2);
        let result = InvalidationPropagation::compute(&nodes, &current);

        assert_eq!(result.entries[0].status, CurrentStatus::Current);
    }

    #[test]
    fn context_change_revalidates_decision_context_when_declared() {
        let nodes = vec![node(
            "decision",
            NodeKind::DecisionContext,
            vec![DependencyKind::ContextOfUse],
        )];
        let current = DependencyFingerprint::revision(7, 3, 4, 2, 8, 5, 2);
        let result = InvalidationPropagation::compute(&nodes, &current);

        assert_eq!(result.entries[0].status, CurrentStatus::RequiresRevalidation);
        assert_eq!(
            result.entries[0].affected_dependencies[0].kind,
            DependencyKind::ContextOfUse
        );
    }

    #[test]
    fn incomplete_dependency_manifest_is_unknown() {
        let mut historical = DependencyFingerprint::revision(7, 3, 4, 2, 8, 5, 1);
        historical.revisions.retain(|(kind, _)| *kind != DependencyKind::Model);
        let node = PropagationNode {
            id: "incomplete".into(),
            kind: NodeKind::Claim,
            historical_dependencies: historical,
            closure: vec![DependencyKind::Model],
        };
        let current = DependencyFingerprint::revision(8, 3, 4, 2, 8, 5, 1);
        let result = InvalidationPropagation::compute(&[node], &current);

        assert_eq!(result.entries[0].status, CurrentStatus::Unknown);
    }

    #[test]
    fn historical_identity_remains_immutable() {
        let nodes = vec![node(
            "claim",
            NodeKind::Claim,
            vec![DependencyKind::Model],
        )];
        let current = DependencyFingerprint::revision(8, 3, 4, 2, 8, 5, 1);
        let _ = InvalidationPropagation::compute(&nodes, &current);

        assert_eq!(
            nodes[0].historical_dependencies.get(&DependencyKind::Model),
            Some(7)
        );
    }

    #[test]
    fn propagation_is_deterministic() {
        let mut nodes = vec![
            node("z", NodeKind::Claim, vec![DependencyKind::Model]),
            node("a", NodeKind::ReplayWitness, vec![DependencyKind::Model]),
        ];
        let current = DependencyFingerprint::revision(8, 3, 4, 2, 8, 5, 1);

        let first = InvalidationPropagation::compute(&nodes, &current);
        nodes.reverse();
        let second = InvalidationPropagation::compute(&nodes, &current);

        assert_eq!(first, second);
    }
}
