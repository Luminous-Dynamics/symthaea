//! Explicit extensive-form transition contracts for Strategic IR.
//!
//! This module stops one layer before a solver: it describes the finite state
//! machine, terminal utilities, chance branches, and player decision points.
//! Information-set validation remains in strategic_context.

use crate::strategic::{ActionId, PlayerId};
use crate::strategic_context::{DecisionStateId, InformationStructure, PerfectRecallEvidence};
use std::collections::{HashMap, HashSet};

/// Stable identity for a chance outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChanceOutcomeId(pub usize);

/// Stable identity for a semantic observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObservationId(pub usize);

/// A semantic observation emitted at a concrete state for one player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Observation {
    pub observer: PlayerId,
    pub observation: ObservationId,
}

/// Lossless transition/observation history for semantic information encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryEvent {
    Decision { state: DecisionStateId, player: PlayerId, information_set: crate::strategic_context::InformationSetId, action: ActionId },
    Chance { state: DecisionStateId, outcome: ChanceOutcomeId, next: DecisionStateId },
    Observation { state: DecisionStateId, observer: PlayerId, observation: ObservationId },
}

/// Semantic mapping from a concrete history to a player's information set.
pub trait InformationEncoder {
    fn encode(
        &self,
        player: PlayerId,
        state: DecisionStateId,
        history: &[HistoryEvent],
    ) -> Result<crate::strategic_context::InformationSetId, InformationEncodingError>;
}

/// Failure returned by a semantic information encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InformationEncodingError {
    pub message: String,
}


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
    pub outcome: ChanceOutcomeId,
    pub probability: f64,
    pub next: DecisionStateId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtensiveGame {
    pub root: DecisionStateId,
    pub nodes: Vec<ExtensiveNode>,
    pub information: InformationStructure,
    /// Observations emitted when a state is entered. Multiple observers may
    /// receive different observations of the same world state.
    pub observations: HashMap<DecisionStateId, Vec<Observation>>,
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

        // Observation streams are semantic inputs to information encoding.
        // They must be unambiguous for each observer at a concrete state.
        for (state, observations) in &self.observations {
            if self.node(*state).is_none() {
                return Err(ExtensiveGameError::ObservationStateMissing(*state));
            }
            for (i, observation) in observations.iter().enumerate() {
                if observations[..i]
                    .iter()
                    .any(|prior| prior.observer == observation.observer)
                {
                    return Err(ExtensiveGameError::DuplicateObservationObserver {
                        state: *state,
                        observer: observation.observer,
                    });
                }
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
                    for (i, outcome) in outcomes.iter().enumerate() {
                        if outcomes[..i].iter().any(|prior| prior.outcome == outcome.outcome) {
                            return Err(ExtensiveGameError::DuplicateChanceOutcome {
                                node: id,
                                outcome: outcome.outcome,
                            });
                        }
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

    /// Verify that a semantic information encoder agrees with the declared partition.
    pub fn verify_information_encoder<E: InformationEncoder>(&self, encoder: &E) -> Result<(), ExtensiveGameError> {
        self.validate()?;
        let mut histories: HashMap<DecisionStateId, Vec<Vec<HistoryEvent>>> = HashMap::new();
        self.collect_information_histories(self.root, Vec::new(), &mut histories)?;
        for state in &self.information.decision_states {
            let paths = histories.get(&state.state).ok_or(ExtensiveGameError::InformationMemberNotReachable(state.state))?;
            for history in paths {
                let encoded = encoder.encode(state.player, state.state, history).map_err(|error| ExtensiveGameError::InformationEncodingFailed {
                        state: state.state,
                        message: error.message,
                    })?;
                if encoded != state.information_set {
                    return Err(ExtensiveGameError::InformationEncodingMismatch { state: state.state, expected: state.information_set, actual: encoded });
                }
            }
        }
        Ok(())
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

    fn collect_information_histories(&self, state: DecisionStateId, mut history: Vec<HistoryEvent>, histories: &mut HashMap<DecisionStateId, Vec<Vec<HistoryEvent>>>) -> Result<(), ExtensiveGameError> {
        if let Some(observations) = self.observations.get(&state) {
            for observation in observations {
                history.push(HistoryEvent::Observation {
                    state,
                    observer: observation.observer,
                    observation: observation.observation,
                });
            }
        }
        histories.entry(state).or_default().push(history.clone());
        match self.node(state).ok_or(ExtensiveGameError::UnknownNode(state))? {
            ExtensiveNode::Decision { player, state, actions } => {
                let info_set = self.information.state(*state).expect("validated decision state").information_set;
                for action in actions {
                    let mut next = history.clone();
                    next.push(HistoryEvent::Decision { state: *state, player: *player, information_set: info_set, action: action.action });
                    self.collect_information_histories(action.next, next, histories)?;
                }
            }
            ExtensiveNode::Chance { state, outcomes } => {
                for outcome in outcomes {
                    let mut next = history.clone();
                    next.push(HistoryEvent::Chance { state: *state, outcome: outcome.outcome, next: outcome.next });
                    self.collect_information_histories(outcome.next, next, histories)?;
                }
            }
            ExtensiveNode::Terminal { .. } => {}
        }
        Ok(())
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
    DuplicateChanceOutcome { node: DecisionStateId, outcome: ChanceOutcomeId },
    InvalidChanceProbability(DecisionStateId),
    ChanceProbabilitiesDoNotSumToOne { node: DecisionStateId, total: f64 },
    InvalidTerminalPayoffs(DecisionStateId),
    NoTerminalNode,
    UnreachableNode(DecisionStateId),
    InformationMemberNotReachable(DecisionStateId),
    CycleDetected(DecisionStateId),
    MultipleParents { child: DecisionStateId, parent: DecisionStateId },
    InformationEncodingFailed { state: DecisionStateId, message: String },
    ObservationStateMissing(DecisionStateId),
    DuplicateObservationObserver { state: DecisionStateId, observer: PlayerId },
    InformationEncodingMismatch { state: DecisionStateId, expected: crate::strategic_context::InformationSetId, actual: crate::strategic_context::InformationSetId },
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
            observations: HashMap::new(),
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
            observations: HashMap::new(),
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
            observations: HashMap::new(),
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
            observations: HashMap::new(),
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
            observations: HashMap::new(),
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
            observations: HashMap::new(),
};

        assert_eq!(
            game.validate(),
            Err(ExtensiveGameError::CycleDetected(DecisionStateId(0)))
        );
    }

    #[test]
    fn verifies_semantic_information_encoder() {
        struct Encoder;

        impl InformationEncoder for Encoder {
            fn encode(
                &self,
                _player: PlayerId,
                state: DecisionStateId,
                _history: &[HistoryEvent],
            ) -> Result<InformationSetId, InformationEncodingError> {
                if state == DecisionStateId(0) {
                    Ok(InformationSetId(0))
                } else {
                    Err(InformationEncodingError { message: "unexpected state".into() })
                }
            }
        }

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
            observations: HashMap::new(),
};

        assert_eq!(game.verify_information_encoder(&Encoder), Ok(()));
    }


    #[test]
    fn semantic_history_preserves_chance_identity_and_observation() {
        struct Encoder;

        impl InformationEncoder for Encoder {
            fn encode(
                &self,
                _player: PlayerId,
                _state: DecisionStateId,
                history: &[HistoryEvent],
            ) -> Result<InformationSetId, InformationEncodingError> {
                assert!(history.iter().any(|event| matches!(
                    event,
                    HistoryEvent::Chance {
                        outcome: ChanceOutcomeId(42),
                        ..
                    }
                )));
                assert!(history.iter().any(|event| matches!(
                    event,
                    HistoryEvent::Observation {
                        observer: PlayerId(0),
                        observation: ObservationId(7),
                        ..
                    }
                )));
                Ok(InformationSetId(0))
            }
        }

        let game = ExtensiveGame {
            root: DecisionStateId(0),
            nodes: vec![
                ExtensiveNode::Chance {
                    state: DecisionStateId(0),
                    outcomes: vec![
                        ChanceTransition {
                            outcome: ChanceOutcomeId(42),
                            probability: 1.0,
                            next: DecisionStateId(1),
                        },
                    ],
                },
                ExtensiveNode::Decision {
                    state: DecisionStateId(1),
                    player: PlayerId(0),
                    actions: vec![
                        Transition {
                            action: ActionId(0),
                            next: DecisionStateId(2),
                        },
                        Transition {
                            action: ActionId(1),
                            next: DecisionStateId(3),
                        },
                    ],
                },
                ExtensiveNode::Terminal {
                    state: DecisionStateId(2),
                    payoffs: vec![1.0],
                },
                ExtensiveNode::Terminal {
                    state: DecisionStateId(3),
                    payoffs: vec![0.0],
                },
            ],
            information: InformationStructure {
                decision_states: vec![DecisionState {
                    state: DecisionStateId(1),
                    player: PlayerId(0),
                    information_set: InformationSetId(0),
                    legal_actions: vec![ActionId(0), ActionId(1)],
                }],
                information_sets: vec![InformationSet {
                    id: InformationSetId(0),
                    player: PlayerId(0),
                    members: vec![DecisionStateId(1)],
                }],
            },
            observations: HashMap::from([(
                DecisionStateId(1),
                vec![Observation {
                    observer: PlayerId(0),
                    observation: ObservationId(7),
                }],
            )]),
        };

        assert_eq!(game.verify_information_encoder(&Encoder), Ok(()));
    }

    #[test]
    fn rejects_semantic_information_encoder_mismatch() {
        struct Encoder;

        impl InformationEncoder for Encoder {
            fn encode(
                &self,
                _player: PlayerId,
                _state: DecisionStateId,
                _history: &[HistoryEvent],
            ) -> Result<InformationSetId, InformationEncodingError> {
                Ok(InformationSetId(99))
            }
        }

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
            observations: HashMap::new(),
};

        assert!(matches!(
            game.verify_information_encoder(&Encoder),
            Err(ExtensiveGameError::InformationEncodingMismatch { .. })
        ));
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
            observations: HashMap::new(),
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
                        ChanceTransition { outcome: ChanceOutcomeId(0), probability: 0.4, next: DecisionStateId(1) },
                        ChanceTransition { outcome: ChanceOutcomeId(1), probability: 0.4, next: DecisionStateId(2) },
                    ],
                },
                ExtensiveNode::Terminal { state: DecisionStateId(1), payoffs: vec![1.0] },
                ExtensiveNode::Terminal { state: DecisionStateId(2), payoffs: vec![0.0] },
            ],
            information: info(),
            observations: HashMap::new(),
};
        assert!(matches!(
            game.validate(),
            Err(ExtensiveGameError::ChanceProbabilitiesDoNotSumToOne { .. })
        ));
    }
}
