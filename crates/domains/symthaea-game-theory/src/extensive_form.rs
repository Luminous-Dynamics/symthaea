//! Explicit extensive-form transition contracts for Strategic IR.
//!
//! This module stops one layer before a solver: it describes the finite state
//! machine, terminal utilities, chance branches, and player decision points.
//! Information-set validation remains in strategic_context.

use crate::strategic::{ActionId, PlayerId};
use crate::strategic_context::{DecisionStateId, InformationStructure, PerfectRecallEvidence};
use std::collections::{HashMap, HashSet};

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

        // Every information-bearing decision state must have exactly one
        // corresponding decision node, and every decision node must be
        // represented by the information structure. This prevents a solver
        // from silently receiving an incomplete or partially detached infoset.
        let decision_node_states: HashSet<_> = self
            .nodes
            .iter()
            .filter_map(|node| match node {
                ExtensiveNode::Decision { state, .. } => Some(*state),
                _ => None,
            })
            .collect();

        for state in &self.information.decision_states {
            if !matches!(self.node(state.state), Some(ExtensiveNode::Decision { .. })) {
                return Err(ExtensiveGameError::InformationStateMissingDecisionNode(state.state));
            }
        }
        for state in &decision_node_states {
            if self.information.state(*state).is_none() {
                return Err(ExtensiveGameError::DecisionNodeMissingInformation(*state));
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
                    let mut action_ids: Vec<_> =
                        actions.iter().map(|action| action.action).collect();
                    let mut legal_actions = decision_state.legal_actions.clone();
                    action_ids.sort_unstable();
                    legal_actions.sort_unstable();
                    if action_ids != legal_actions {
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

        // A finite extensive-form game is represented as an acyclic reachable
        // state graph here. Repeated-state/transposition semantics need an
        // explicit history model before they can safely share state IDs.
        if let Some((child, parent)) = self.find_multiple_parent() {
            return Err(ExtensiveGameError::MultipleParents { child, parent });
        }

        if let Some(cycle) = self.find_cycle() {
            return Err(ExtensiveGameError::CycleDetected(cycle));
        }

        // Information-set members are required to be reachable decision
        // states, not merely entries in the sidecar information structure.
        for info_set in &self.information.information_sets {
            for member in &info_set.members {
                if !reachable.contains(member)
                    || !matches!(self.node(*member), Some(ExtensiveNode::Decision { .. }))
                {
                    return Err(ExtensiveGameError::InformationMemberNotReachable(*member));
                }
            }
        }

        Ok(())
    }

    pub fn node(&self, id: DecisionStateId) -> Option<&ExtensiveNode> {
        self.nodes.iter().find(|node| node.state() == id)
    }

    /// Mechanically verify perfect recall from the reachable game histories.
    ///
    /// For every information set, all member states must induce the same
    /// sequence of that player's earlier information sets and chosen actions.
    /// Chance and opponent actions are intentionally omitted from the recalled
    /// sequence: they may differ while remaining indistinguishable to the
    /// acting player.
    /// Verify that every decision state has the same legal-action *set* as
    /// its information-set peers. The current IR represents availability as
    /// a common finite action vocabulary; ordering is not semantic.
    pub fn validate_action_availability(&self) -> Result<(), ExtensiveGameError> {
        self.information
            .validate()
            .map_err(ExtensiveGameError::InformationStructure)
    }

    pub fn verify_perfect_recall(&self) -> Result<PerfectRecallEvidence, ExtensiveGameError> {
        self.validate()?;

        // A caller cannot manufacture a Verified marker by constructing a
        // compatible-looking information structure: verification always runs
        // against the actual reachable game graph.
        let mut histories: HashMap<DecisionStateId, Vec<Vec<RecallStep>>> = HashMap::new();
        self.collect_histories(self.root, Vec::new(), &mut histories)?;

        for info_set in &self.information.information_sets {
            let mut expected: Option<Vec<RecallStep>> = None;
            for member in &info_set.members {
                let member_histories = histories
                    .get(member)
                    .ok_or(ExtensiveGameError::InformationMemberNotReachable(*member))?;
                let mut own_histories = member_histories
                    .iter()
                    .map(|history| {
                        history
                            .iter()
                            .filter(|step| step.player == info_set.player)
                            .cloned()
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                own_histories.sort();
                own_histories.dedup();

                if own_histories.len() != 1 {
                    return Err(ExtensiveGameError::PerfectRecallViolation {
                        information_set: info_set.id,
                        state: *member,
                    });
                }

                let actual = own_histories.pop().expect("one history after validation");
                if let Some(expected) = &expected {
                    if *expected != actual {
                        return Err(ExtensiveGameError::PerfectRecallViolation {
                            information_set: info_set.id,
                            state: *member,
                        });
                    }
                } else {
                    expected = Some(actual);
                }
            }
        }

        Ok(PerfectRecallEvidence::Verified)
    }

    fn collect_histories(
        &self,
        state: DecisionStateId,
        history: Vec<RecallStep>,
        histories: &mut HashMap<DecisionStateId, Vec<Vec<RecallStep>>>,
    ) -> Result<(), ExtensiveGameError> {
        histories.entry(state).or_default().push(history.clone());

        match self.node(state).ok_or(ExtensiveGameError::UnknownNode(state))? {
            ExtensiveNode::Decision { player, state, actions } => {
                let info_set = self
                    .information
                    .state(*state)
                    .expect("decision node has information state after validation")
                    .information_set;
                for action in actions {
                    let mut next_history = history.clone();
                    next_history.push(RecallStep {
                        player: *player,
                        information_set: info_set,
                        action: action.action,
                    });
                    self.collect_histories(action.next, next_history, histories)?;
                }
            }
            ExtensiveNode::Chance { outcomes, .. } => {
                for outcome in outcomes {
                    self.collect_histories(outcome.next, history.clone(), histories)?;
                }
            }
            ExtensiveNode::Terminal { .. } => {}
        }

        Ok(())
    }

    fn find_multiple_parent(&self) -> Option<(DecisionStateId, DecisionStateId)> {
        let mut parents = HashMap::<DecisionStateId, DecisionStateId>::new();
        let mut stack = vec![self.root];
        while let Some(state) = stack.pop() {
            let node = self.node(state)?;
            let children = match node {
                ExtensiveNode::Decision { actions, .. } => actions.iter().map(|a| a.next).collect::<Vec<_>>(),
                ExtensiveNode::Chance { outcomes, .. } => outcomes.iter().map(|o| o.next).collect::<Vec<_>>(),
                ExtensiveNode::Terminal { .. } => Vec::new(),
            };
            for child in children {
                if child == self.root { continue; }
                if let Some(previous) = parents.insert(child, state) {
                    if previous != state { return Some((child, previous)); }
                } else {
                    stack.push(child);
                }
            }
        }
        None
    }

    fn find_cycle(&self) -> Option<DecisionStateId> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Mark {
            Visiting,
            Done,
        }

        fn visit(
            game: &ExtensiveGame,
            state: DecisionStateId,
            marks: &mut HashMap<DecisionStateId, Mark>,
        ) -> Option<DecisionStateId> {
            if matches!(marks.get(&state), Some(Mark::Visiting)) {
                return Some(state);
            }
            if matches!(marks.get(&state), Some(Mark::Done)) {
                return None;
            }

            marks.insert(state, Mark::Visiting);
            let node = game.node(state)?;
            let children = match node {
                ExtensiveNode::Decision { actions, .. } =>
                    actions.iter().map(|a| a.next).collect::<Vec<_>>(),
                ExtensiveNode::Chance { outcomes, .. } =>
                    outcomes.iter().map(|o| o.next).collect::<Vec<_>>(),
                ExtensiveNode::Terminal { .. } => Vec::new(),
            };

            for child in children {
                if let Some(cycle) = visit(game, child, marks) {
                    return Some(cycle);
                }
            }
            marks.insert(state, Mark::Done);
            None
        }

        visit(self, self.root, &mut HashMap::new())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RecallStep {
    player: PlayerId,
    information_set: crate::strategic_context::InformationSetId,
    action: ActionId,
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
    InformationStateMissingDecisionNode(DecisionStateId),
    DecisionNodeMissingInformation(DecisionStateId),
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
    InformationMemberNotReachable(DecisionStateId),
    CycleDetected(DecisionStateId),
    MultipleParents { child: DecisionStateId, parent: DecisionStateId },
    PerfectRecallViolation {
        information_set: crate::strategic_context::InformationSetId,
        state: DecisionStateId,
    },
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
    fn rejects_decision_actions_that_disagree_with_information_set() {
        let mut game = ExtensiveGame {
            root: DecisionStateId(0),
            nodes: vec![
                ExtensiveNode::Decision {
                    state: DecisionStateId(0),
                    player: PlayerId(0),
                    actions: vec![
                        Transition { action: ActionId(0), next: DecisionStateId(1) },
                        Transition { action: ActionId(2), next: DecisionStateId(2) },
                    ],
                },
                ExtensiveNode::Terminal { state: DecisionStateId(1), payoffs: vec![1.0] },
                ExtensiveNode::Terminal { state: DecisionStateId(2), payoffs: vec![0.0] },
            ],
            information: info(),
        };
        assert_eq!(
            game.validate(),
            Err(ExtensiveGameError::DecisionActionsMismatch(DecisionStateId(0)))
        );

        game.nodes[0] = ExtensiveNode::Decision {
            state: DecisionStateId(0),
            player: PlayerId(1),
            actions: vec![
                Transition { action: ActionId(0), next: DecisionStateId(1) },
                Transition { action: ActionId(1), next: DecisionStateId(2) },
            ],
        };
        assert!(matches!(
            game.validate(),
            Err(ExtensiveGameError::DecisionPlayerMismatch { .. })
        ));
    }

    #[test]
    fn action_order_is_not_semantic() {
        let mut information = info();
        information.decision_states[0].legal_actions.reverse();

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
            information,
        };

        assert!(game.validate().is_ok());
        assert!(game.validate_action_availability().is_ok());
    }

    #[test]
    fn verifies_perfect_recall_for_single_decision() {
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

        assert_eq!(
            game.verify_perfect_recall(),
            Ok(PerfectRecallEvidence::Verified)
        );
    }

    #[test]
    fn rejects_forgetting_own_prior_action() {
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
                ExtensiveNode::Decision {
                    state: DecisionStateId(1),
                    player: PlayerId(0),
                    actions: vec![
                        Transition { action: ActionId(2), next: DecisionStateId(3) },
                    ],
                },
                ExtensiveNode::Decision {
                    state: DecisionStateId(2),
                    player: PlayerId(0),
                    actions: vec![
                        Transition { action: ActionId(2), next: DecisionStateId(4) },
                    ],
                },
                ExtensiveNode::Terminal { state: DecisionStateId(3), payoffs: vec![1.0] },
                ExtensiveNode::Terminal { state: DecisionStateId(4), payoffs: vec![0.0] },
            ],
            information: InformationStructure {
                decision_states: vec![
                    DecisionState {
                        state: DecisionStateId(0),
                        player: PlayerId(0),
                        information_set: InformationSetId(0),
                        legal_actions: vec![ActionId(0), ActionId(1)],
                    },
                    DecisionState {
                        state: DecisionStateId(1),
                        player: PlayerId(0),
                        information_set: InformationSetId(1),
                        legal_actions: vec![ActionId(2)],
                    },
                    DecisionState {
                        state: DecisionStateId(2),
                        player: PlayerId(0),
                        information_set: InformationSetId(1),
                        legal_actions: vec![ActionId(2)],
                    },
                ],
                information_sets: vec![
                    InformationSet {
                        id: InformationSetId(0),
                        player: PlayerId(0),
                        members: vec![DecisionStateId(0)],
                    },
                    InformationSet {
                        id: InformationSetId(1),
                        player: PlayerId(0),
                        members: vec![DecisionStateId(1), DecisionStateId(2)],
                    },
                ],
            },
        };

        assert!(matches!(
            game.verify_perfect_recall(),
            Err(ExtensiveGameError::PerfectRecallViolation {
                information_set: InformationSetId(1),
                ..
            })
        ));
    }

    #[test]
    fn rejects_information_member_without_decision_node() {
        let mut information = info();
        information.decision_states.push(DecisionState {
            state: DecisionStateId(3),
            player: PlayerId(0),
            information_set: InformationSetId(0),
            legal_actions: vec![ActionId(0), ActionId(1)],
        });
        information.information_sets[0].members.push(DecisionStateId(3));

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
            information,
        };

        assert_eq!(
            game.validate(),
            Err(ExtensiveGameError::InformationStateMissingDecisionNode(
                DecisionStateId(3)
            ))
        );
    }

    #[test]
    fn rejects_cycles() {
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
                ExtensiveNode::Decision {
                    state: DecisionStateId(1),
                    player: PlayerId(0),
                    actions: vec![
                        Transition { action: ActionId(0), next: DecisionStateId(0) },
                        Transition { action: ActionId(1), next: DecisionStateId(2) },
                    ],
                },
                ExtensiveNode::Terminal { state: DecisionStateId(2), payoffs: vec![0.0] },
            ],
            information: InformationStructure {
                decision_states: vec![
                    DecisionState {
                        state: DecisionStateId(0),
                        player: PlayerId(0),
                        information_set: InformationSetId(0),
                        legal_actions: vec![ActionId(0), ActionId(1)],
                    },
                    DecisionState {
                        state: DecisionStateId(1),
                        player: PlayerId(0),
                        information_set: InformationSetId(1),
                        legal_actions: vec![ActionId(0), ActionId(1)],
                    },
                ],
                information_sets: vec![
                    InformationSet {
                        id: InformationSetId(0),
                        player: PlayerId(0),
                        members: vec![DecisionStateId(0)],
                    },
                    InformationSet {
                        id: InformationSetId(1),
                        player: PlayerId(0),
                        members: vec![DecisionStateId(1)],
                    },
                ],
            },
        };

        assert_eq!(
            game.validate(),
            Err(ExtensiveGameError::CycleDetected(DecisionStateId(0)))
        );
    }

    #[test]
    fn rejects_duplicate_node_ids() {
        let game = ExtensiveGame {
            root: DecisionStateId(0),
            nodes: vec![
                ExtensiveNode::Terminal { state: DecisionStateId(0), payoffs: vec![1.0] },
                ExtensiveNode::Terminal { state: DecisionStateId(0), payoffs: vec![0.0] },
            ],
            information: info(),
        };
        assert_eq!(
            game.validate(),
            Err(ExtensiveGameError::DuplicateNodeId(DecisionStateId(0)))
        );
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
