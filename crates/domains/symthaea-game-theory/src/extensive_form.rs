//! Explicit extensive-form transition contracts for Strategic IR.
//!
//! This module stops one layer before a solver: it describes the finite state
//! machine, terminal utilities, chance branches, and player decision points.
//! Information-set validation remains in strategic_context.

use crate::strategic::{ActionId, PlayerId};
use crate::strategic_context::{DecisionStateId, InformationStructure};

#[derive(Debug, Clone, PartialEq)]
pub enum ExtensiveNode {
    Decision { state: DecisionStateId, player: PlayerId, actions: Vec<Transition> },
    Chance { state: DecisionStateId, outcomes: Vec<ChanceTransition> },
    Terminal { state: DecisionStateId, payoffs: Vec<f64> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    pub action: ActionId,
    pub next: DecisionStateId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChanceTransition {
    pub probability: f64,
    pub next: DecisionStateId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtensiveGame {
    pub root: DecisionStateId,
    pub nodes: Vec<ExtensiveNode>,
    pub information: InformationStructure,
}

impl ExtensiveGame {
    pub fn validate(&self) -> Result<(), ExtensiveGameError> {
        self.information
            .validate()
            .map_err(ExtensiveGameError::InformationStructure)?;

        if self.node(self.root).is_none() {
            return Err(ExtensiveGameError::UnknownNode(self.root));
        }

        let players = self
            .nodes
            .iter()
            .find_map(|node| match node {
                ExtensiveNode::Terminal { payoffs, .. } => Some(payoffs.len()),
                _ => None,
            })
            .ok_or(ExtensiveGameError::NoTerminalNode)?;

        for (i, node) in self.nodes.iter().enumerate() {
            if self.nodes[..i].iter().any(|prior| prior.state() == node.state()) {
                return Err(ExtensiveGameError::DuplicateNodeId(node.state()));
            }
        }

        let mut reachable = Vec::new();
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            if reachable.contains(&id) {
                continue;
            }
            let Some(node) = self.node(id) else {
                return Err(ExtensiveGameError::UnknownNode(id));
            };
            reachable.push(id);

            match node {
                ExtensiveNode::Decision { state, player, actions } => {
                    let Some(decision_state) = self.information.state(*state) else {
                        return Err(ExtensiveGameError::DecisionStateMissingFromInformation(*state));
                    };
                    if decision_state.player != *player {
                        return Err(ExtensiveGameError::DecisionPlayerMismatch {
                            state: *state,
                            expected: decision_state.player,
                            actual: *player,
                        });
                    }
                    let action_ids: Vec<_> = actions.iter().map(|action| action.action).collect();
                    if action_ids != decision_state.legal_actions {
                        return Err(ExtensiveGameError::DecisionActionsMismatch(*state));
                    }
                    if actions.is_empty() {
                        return Err(ExtensiveGameError::NoActions(id));
                    }
                    for (i, action) in actions.iter().enumerate() {
                        if actions[..i].iter().any(|prior| prior.action == action.action) {
                            return Err(ExtensiveGameError::DuplicateAction {
                                node: id,
                                action: action.action,
                            });
                        }
                        stack.push(action.next);
                    }
                }
                ExtensiveNode::Chance { outcomes, .. } => {
                    if outcomes.is_empty() {
                        return Err(ExtensiveGameError::NoChanceOutcomes(id));
                    }
                    let mut total = 0.0;
                    for outcome in outcomes {
                        if !outcome.probability.is_finite()
                            || !(0.0..=1.0).contains(&outcome.probability)
                        {
                            return Err(ExtensiveGameError::InvalidChanceProbability(id));
                        }
                        total += outcome.probability;
                        stack.push(outcome.next);
                    }
                    if (total - 1.0).abs() > 1e-9 {
                        return Err(ExtensiveGameError::ChanceProbabilitiesDoNotSumToOne {
                            node: id,
                            total,
                        });
                    }
                }
                ExtensiveNode::Terminal { payoffs, .. } => {
                    if payoffs.len() != players || payoffs.iter().any(|p| !p.is_finite()) {
                        return Err(ExtensiveGameError::InvalidTerminalPayoffs(id));
                    }
                }
            }
        }

        for node in &self.nodes {
            let id = node.state();
            if !reachable.contains(&id) {
                return Err(ExtensiveGameError::UnreachableNode(id));
            }
        }
        Ok(())
    }

    pub fn node(&self, id: DecisionStateId) -> Option<&ExtensiveNode> {
        self.nodes.iter().find(|node| node.state() == id)
    }
}

impl ExtensiveNode {
    pub fn state(&self) -> DecisionStateId {
        match self {
            Self::Decision { state, .. }
            | Self::Chance { state, .. }
            | Self::Terminal { state, .. } => *state,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExtensiveGameError {
    InformationStructure(crate::strategic_context::InformationStructureError),
    UnknownNode(DecisionStateId),
    DuplicateNodeId(DecisionStateId),
    DecisionStateMissingFromInformation(DecisionStateId),
    DecisionPlayerMismatch { state: DecisionStateId, expected: PlayerId, actual: PlayerId },
    DecisionActionsMismatch(DecisionStateId),
    DuplicateAction { node: DecisionStateId, action: ActionId },
    NoActions(DecisionStateId),
    NoChanceOutcomes(DecisionStateId),
    InvalidChanceProbability(DecisionStateId),
    ChanceProbabilitiesDoNotSumToOne { node: DecisionStateId, total: f64 },
    InvalidTerminalPayoffs(DecisionStateId),
    NoTerminalNode,
    UnreachableNode(DecisionStateId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategic_context::{DecisionState, InformationSet, InformationSetId};

    fn info() -> InformationStructure {
        InformationStructure {
            decision_states: vec![DecisionState {
                state: DecisionStateId(0),
                player: PlayerId(0),
                information_set: InformationSetId(0),
                legal_actions: vec![ActionId(0), ActionId(1)],
            }],
            information_sets: vec![InformationSet {
                id: InformationSetId(0),
                player: PlayerId(0),
                members: vec![DecisionStateId(0)],
            }],
        }
    }

    #[test]
    fn validates_decision_to_terminal_transition() {
        let game = ExtensiveGame {
            root: DecisionStateId(0),
            nodes: vec![
                ExtensiveNode::Decision {
                    state: DecisionStateId(0),
                    player: PlayerId(0),
                    actions: vec![
                        Transition { action: ActionId(0), next: DecisionStateId(1) },
                        Transition { action: ActionId(1), next: DecisionStateId(2) },
                    ],
                },
                ExtensiveNode::Terminal { state: DecisionStateId(1), payoffs: vec![1.0] },
                ExtensiveNode::Terminal { state: DecisionStateId(2), payoffs: vec![0.0] },
            ],
            information: info(),
        };
        assert!(game.validate().is_ok());
    }

    #[test]
    fn rejects_non_normalized_chance() {
        let game = ExtensiveGame {
            root: DecisionStateId(0),
            nodes: vec![
                ExtensiveNode::Chance {
                    state: DecisionStateId(0),
                    outcomes: vec![
                        ChanceTransition { probability: 0.4, next: DecisionStateId(1) },
                        ChanceTransition { probability: 0.4, next: DecisionStateId(2) },
                    ],
                },
                ExtensiveNode::Terminal { state: DecisionStateId(1), payoffs: vec![1.0] },
                ExtensiveNode::Terminal { state: DecisionStateId(2), payoffs: vec![0.0] },
            ],
            information: info(),
        };
        assert!(matches!(
            game.validate(),
            Err(ExtensiveGameError::ChanceProbabilitiesDoNotSumToOne { .. })
        ));
    }
}
