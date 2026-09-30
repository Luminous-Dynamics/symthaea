//! Information-aware strategy contracts for the canonical Strategic IR.
//!
//! Agent policies receive only an explicit observation and legal-action set.
//! They do not receive the full world state by default. This module defines
//! representation and validation contracts; it does not implement a solver.

use crate::strategic::{ActionId, PlayerId, SolverCapabilities};

/// Stable identifier for a player's information set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InformationSetId(pub usize);

/// An action available at a decision point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Action {
    pub id: ActionId,
    pub label: String,
}

/// A probability distribution over a finite set of actions.
#[derive(Debug, Clone, PartialEq)]
pub struct MixedAction {
    pub action: ActionId,
    pub probability: f64,
}

/// An agent-local view at a decision point.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentContext {
    pub player: PlayerId,
    pub information_set: InformationSetId,
    /// Observation supplied to this agent, not an omniscient world state.
    pub observation: String,
    /// Actions currently legal for this player at this information set.
    pub legal_actions: Vec<ActionId>,
}

impl AgentContext {
    /// Validate that a context has a non-empty, duplicate-free legal action set.
    pub fn validate(&self) -> Result<(), ContextError> {
        if self.legal_actions.is_empty() {
            return Err(ContextError::NoLegalActions);
        }
        for (i, action) in self.legal_actions.iter().enumerate() {
            if self.legal_actions[..i].contains(action) {
                return Err(ContextError::DuplicateLegalAction(*action));
            }
        }
        Ok(())
    }

    pub fn permits(&self, action: ActionId) -> bool {
        self.legal_actions.contains(&action)
    }
}

/// A decision procedure operating only on the supplied agent context.
pub trait Policy {
    /// Return an action distribution over the context's legal actions.
    fn decide(&self, context: &AgentContext) -> Result<Vec<MixedAction>, ContextError>;
}


/// Stable identifier for a concrete decision state in an extensive-form model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DecisionStateId(pub usize);

/// One concrete decision state belonging to an information set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionState {
    pub state: DecisionStateId,
    pub player: PlayerId,
    pub information_set: InformationSetId,
    pub legal_actions: Vec<ActionId>,
}

/// A collection of decision states that are indistinguishable to a player.
///
/// Standard extensive-form solvers require member states to expose the same
/// action vocabulary. State-dependent availability should be modeled later as
/// an explicit action-availability semantics, rather than silently weakening
/// this invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InformationSet {
    pub id: InformationSetId,
    pub player: PlayerId,
    pub members: Vec<DecisionStateId>,
}

/// A finite information structure for extensive-form strategic models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InformationStructure {
    pub decision_states: Vec<DecisionState>,
    pub information_sets: Vec<InformationSet>,
}

impl InformationStructure {
    pub fn validate(&self) -> Result<(), InformationStructureError> {
        let mut membership = vec![None; self.decision_states.len()];

        for (i, state) in self.decision_states.iter().enumerate() {
            if self.decision_states[..i]
                .iter()
                .any(|prior| prior.state == state.state)
            {
                return Err(InformationStructureError::DuplicateDecisionStateId(state.state));
            }
        }
        for (i, info_set) in self.information_sets.iter().enumerate() {
            if self.information_sets[..i]
                .iter()
                .any(|prior| prior.id == info_set.id)
            {
                return Err(InformationStructureError::DuplicateInformationSetId(info_set.id));
            }
        }

        for info_set in &self.information_sets {
            if info_set.members.is_empty() {
                return Err(InformationStructureError::EmptyInformationSet(info_set.id));
            }
            for (i, state) in info_set.members.iter().enumerate() {
                if info_set.members[..i].contains(state) {
                    return Err(InformationStructureError::DuplicateMember {
                        information_set: info_set.id,
                        state: *state,
                    });
                }
            }

            let Some(first) = self.state(info_set.members[0]) else {
                return Err(InformationStructureError::UnknownDecisionState {
                    information_set: info_set.id,
                    state: info_set.members[0],
                });
            };

            if first.player != info_set.player {
                return Err(InformationStructureError::PlayerMismatch {
                    information_set: info_set.id,
                    expected: info_set.player,
                    actual: first.player,
                    state: first.state,
                });
            }
            validate_legal_actions(&first.legal_actions).map_err(|error| {
                InformationStructureError::InvalidLegalActions {
                    state: first.state,
                    error,
                }
            })?;

            for state_id in &info_set.members {
                let Some(state) = self.state(*state_id) else {
                    return Err(InformationStructureError::UnknownDecisionState {
                        information_set: info_set.id,
                        state: *state_id,
                    });
                };
                if state.information_set != info_set.id {
                    return Err(InformationStructureError::StateInformationSetMismatch {
                        state: state.state,
                        expected: info_set.id,
                        actual: state.information_set,
                    });
                }
                if state.player != info_set.player {
                    return Err(InformationStructureError::PlayerMismatch {
                        information_set: info_set.id,
                        expected: info_set.player,
                        actual: state.player,
                        state: state.state,
                    });
                }
                validate_legal_actions(&state.legal_actions).map_err(|error| {
                    InformationStructureError::InvalidLegalActions {
                        state: state.state,
                        error,
                    }
                })?;
                // Legal-action ordering is representational, not semantic.
                // Compare canonical sets so equivalent vocabularies cannot
                // diverge merely because their source order differs.
                let mut expected_actions = first.legal_actions.clone();
                let mut actual_actions = state.legal_actions.clone();
                expected_actions.sort_unstable();
                actual_actions.sort_unstable();
                if actual_actions != expected_actions {
                    return Err(InformationStructureError::InconsistentActionSet {
                        information_set: info_set.id,
                        expected: first.legal_actions.clone(),
                        actual: state.legal_actions.clone(),
                        state: state.state,
                    });
                }

                let state_index = self
                    .decision_states
                    .iter()
                    .position(|candidate| candidate.state == state.state)
                    .expect("validated decision-state id must exist");
                let slot = &mut membership[state_index];
                if let Some(previous) = *slot {
                    return Err(InformationStructureError::StateInMultipleInformationSets {
                        state: state.state,
                        first: previous,
                        second: info_set.id,
                    });
                }
                *slot = Some(info_set.id);
            }
        }

        for (index, state) in self.decision_states.iter().enumerate() {
            if membership[index].is_none() {
                return Err(InformationStructureError::UnassignedDecisionState(state.state));
            }
        }
        Ok(())
    }

    pub fn state(&self, id: DecisionStateId) -> Option<&DecisionState> {
        self.decision_states
            .iter()
            .find(|state| state.state == id)
    }

    /// Gate solver use on declared information-structure capabilities.
    ///
    /// Perfect recall is deliberately not a universal IR invariant. Solvers
    /// that require it must receive explicit evidence from the model layer.
    pub fn validate_for_solver(
        &self,
        capabilities: SolverCapabilities,
        perfect_recall: Option<PerfectRecallEvidence>,
    ) -> Result<(), SolverCompatibilityError> {
        self.validate()
            .map_err(SolverCompatibilityError::InvalidInformationStructure)?;

        if !capabilities.supports_imperfect_information
            && self.information_sets.iter().any(|set| set.members.len() > 1)
        {
            return Err(SolverCompatibilityError::ImperfectInformationUnsupported);
        }

        if capabilities.requires_perfect_recall {
            match perfect_recall {
                Some(PerfectRecallEvidence::Verified) => {}
                Some(PerfectRecallEvidence::Violated) => {
                    return Err(SolverCompatibilityError::PerfectRecallRequired)
                }
                None => return Err(SolverCompatibilityError::PerfectRecallUnverified),
            }
        }
        Ok(())
    }
}

/// Evidence supplied by a model validator for a solver requiring perfect recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfectRecallEvidence {
    Verified,
    Violated,
}

/// Validation failures for extensive-form information structures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InformationStructureError {
    EmptyInformationSet(InformationSetId),
    DuplicateDecisionStateId(DecisionStateId),
    DuplicateInformationSetId(InformationSetId),
    DuplicateMember {
        information_set: InformationSetId,
        state: DecisionStateId,
    },
    UnknownDecisionState {
        information_set: InformationSetId,
        state: DecisionStateId,
    },
    StateInformationSetMismatch {
        state: DecisionStateId,
        expected: InformationSetId,
        actual: InformationSetId,
    },
    PlayerMismatch {
        information_set: InformationSetId,
        expected: PlayerId,
        actual: PlayerId,
        state: DecisionStateId,
    },
    InvalidLegalActions {
        state: DecisionStateId,
        error: ContextError,
    },
    InconsistentActionSet {
        information_set: InformationSetId,
        expected: Vec<ActionId>,
        actual: Vec<ActionId>,
        state: DecisionStateId,
    },
    StateInMultipleInformationSets {
        state: DecisionStateId,
        first: InformationSetId,
        second: InformationSetId,
    },
    UnassignedDecisionState(DecisionStateId),
}

/// Solver/game information compatibility failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverCompatibilityError {
    InvalidInformationStructure(InformationStructureError),
    ImperfectInformationUnsupported,
    PerfectRecallRequired,
    PerfectRecallUnverified,
}

fn validate_legal_actions(actions: &[ActionId]) -> Result<(), ContextError> {
    if actions.is_empty() {
        return Err(ContextError::NoLegalActions);
    }
    for (i, action) in actions.iter().enumerate() {
        if actions[..i].contains(action) {
            return Err(ContextError::DuplicateLegalAction(*action));
        }
    }
    Ok(())
}

/// A complete contingent plan for a finite set of information sets.
/// Each information set may occur at most once in the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strategy {
    pub player: PlayerId,
    pub decisions: Vec<(InformationSetId, ActionId)>,
}

/// One information set and its legal actions, used to validate a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionPoint {
    pub player: PlayerId,
    pub information_set: InformationSetId,
    pub legal_actions: Vec<ActionId>,
}

impl Strategy {
    /// Validate uniqueness, coverage, and action legality for the supplied
    /// decision points. Coverage is exact: omitted or unknown information
    /// sets are rejected rather than assigned an implicit default.
    pub fn validate(&self, points: &[DecisionPoint]) -> Result<(), ContextError> {
        for (i, (set, _)) in self.decisions.iter().enumerate() {
            if self.decisions[..i].iter().any(|(prior, _)| prior == set) {
                return Err(ContextError::DuplicateInformationSet(*set));
            }
        }
        for (i, point) in points.iter().enumerate() {
            if point.player != self.player {
                return Err(ContextError::PlayerMismatch { information_set: point.information_set, expected: self.player, actual: point.player });
            }
            if points[..i].iter().any(|prior| prior.information_set == point.information_set) {
                return Err(ContextError::DuplicateInformationSet(point.information_set));
            }
            validate_legal_actions(&point.legal_actions)?;
            if !self.decisions.iter().any(|(set, _)| *set == point.information_set) {
                return Err(ContextError::MissingDecision(point.information_set));
            }
        }
        for (set, action) in &self.decisions {
            let Some(point) = points.iter().find(|point| point.information_set == *set) else {
                return Err(ContextError::UnknownInformationSet(*set));
            };
            if !point.legal_actions.contains(action) {
                return Err(ContextError::IllegalAction { information_set: *set, action: *action });
            }
        }
        Ok(())
    }
}

/// A profile of complete contingent plans, one per player.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyProfilePlan(pub Vec<Strategy>);

impl StrategyProfilePlan {
    /// Validate every player plan against its own decision points.
    pub fn validate(
        &self,
        points: &[(PlayerId, Vec<DecisionPoint>)],
    ) -> Result<(), ContextError> {
        if self.0.len() != points.len() {
            return Err(ContextError::ProfilePlayerCount {
                expected: points.len(),
                actual: self.0.len(),
            });
        }
        for (strategy, (player, decision_points)) in self.0.iter().zip(points) {
            if strategy.player != *player {
                return Err(ContextError::PlayerMismatch {
                    information_set: InformationSetId(usize::MAX),
                    expected: *player,
                    actual: strategy.player,
                });
            }
            strategy.validate(decision_points)?;
        }
        Ok(())
    }
}

/// Validation failures for information-aware decision contracts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    NoLegalActions,
    DuplicateLegalAction(ActionId),
    DuplicateInformationSet(InformationSetId),
    PlayerMismatch { information_set: InformationSetId, expected: PlayerId, actual: PlayerId },
    ProfilePlayerCount { expected: usize, actual: usize },
    MissingDecision(InformationSetId),
    UnknownInformationSet(InformationSetId),
    IllegalAction { information_set: InformationSetId, action: ActionId },
    EmptyDistribution,
    InvalidProbability(ActionId),
    DistributionDoesNotSumToOne,
    ProbabilityForIllegalAction(ActionId),
    DuplicateProbabilityAction(ActionId),
}

impl ContextError {
    /// Validate a policy distribution against the legal actions in a context.
    pub fn validate_distribution(
        context: &AgentContext,
        distribution: &[MixedAction],
    ) -> Result<(), Self> {
        context.validate()?;
        if distribution.is_empty() {
            return Err(Self::EmptyDistribution);
        }
        let mut total = 0.0;
        for (i, entry) in distribution.iter().enumerate() {
            if !entry.probability.is_finite() || !(0.0..=1.0).contains(&entry.probability) {
                return Err(Self::InvalidProbability(entry.action));
            }
            if !context.permits(entry.action) {
                return Err(Self::ProbabilityForIllegalAction(entry.action));
            }
            if distribution[..i].iter().any(|prior| prior.action == entry.action) {
                return Err(Self::DuplicateProbabilityAction(entry.action));
            }
            total += entry.probability;
        }
        if !total.is_finite() || (total - 1.0).abs() > 1e-9 {
            return Err(Self::DistributionDoesNotSumToOne);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> AgentContext {
        AgentContext {
            player: PlayerId(0),
            information_set: InformationSetId(7),
            observation: "public signal: red".into(),
            legal_actions: vec![ActionId(0), ActionId(1)],
        }
    }


    fn valid_information_structure() -> InformationStructure {
        InformationStructure {
            decision_states: vec![
                DecisionState {
                    state: DecisionStateId(0),
                    player: PlayerId(0),
                    information_set: InformationSetId(7),
                    legal_actions: vec![ActionId(0), ActionId(1)],
                },
                DecisionState {
                    state: DecisionStateId(1),
                    player: PlayerId(0),
                    information_set: InformationSetId(7),
                    legal_actions: vec![ActionId(0), ActionId(1)],
                },
            ],
            information_sets: vec![InformationSet {
                id: InformationSetId(7),
                player: PlayerId(0),
                members: vec![DecisionStateId(0), DecisionStateId(1)],
            }],
        }
    }

    #[test]
    fn information_sets_require_same_player_and_action_vocabulary() {
        let mut structure = valid_information_structure();
        structure.decision_states[1].legal_actions = vec![ActionId(0)];
        assert!(matches!(
            structure.validate(),
            Err(InformationStructureError::InconsistentActionSet { .. })
        ));

        structure.decision_states[1].legal_actions = vec![ActionId(0), ActionId(1)];
        structure.decision_states[1].player = PlayerId(1);
        assert!(matches!(
            structure.validate(),
            Err(InformationStructureError::PlayerMismatch { .. })
        ));
    }

    #[test]
    fn decision_states_cannot_be_shared_or_left_unassigned() {
        let mut structure = valid_information_structure();
        structure.information_sets.push(InformationSet {
            id: InformationSetId(8),
            player: PlayerId(0),
            members: vec![DecisionStateId(0)],
        });
        assert!(matches!(
            structure.validate(),
            Err(InformationStructureError::StateInMultipleInformationSets { .. })
        ));

        structure.information_sets[1].members = vec![DecisionStateId(1)];
        structure.decision_states.push(DecisionState {
            state: DecisionStateId(2),
            player: PlayerId(0),
            information_set: InformationSetId(8),
            legal_actions: vec![ActionId(0), ActionId(1)],
        });
        assert!(matches!(
            structure.validate(),
            Err(InformationStructureError::UnassignedDecisionState(DecisionStateId(2)))
        ));
    }


    #[test]
    fn decision_state_and_information_set_ids_must_be_unique() {
        let mut structure = valid_information_structure();
        structure.decision_states[1].state = DecisionStateId(0);
        assert_eq!(
            structure.validate(),
            Err(InformationStructureError::DuplicateDecisionStateId(DecisionStateId(0)))
        );

        structure.decision_states[1].state = DecisionStateId(1);
        structure.information_sets.push(InformationSet {
            id: InformationSetId(7),
            player: PlayerId(0),
            members: vec![DecisionStateId(0)],
        });
        assert_eq!(
            structure.validate(),
            Err(InformationStructureError::DuplicateInformationSetId(InformationSetId(7)))
        );
    }

    #[test]
    fn decision_state_must_name_its_containing_information_set() {
        let mut structure = valid_information_structure();
        structure.decision_states[1].information_set = InformationSetId(99);
        assert!(matches!(
            structure.validate(),
            Err(InformationStructureError::StateInformationSetMismatch { .. })
        ));
    }

    #[test]
    fn solver_capabilities_gate_imperfect_information_and_recall() {
        let structure = valid_information_structure();
        let no_imperfect = SolverCapabilities {
            supports_imperfect_information: false,
            requires_perfect_recall: false,
            supports_chance: false,
            supports_general_sum: true,
        };
        assert_eq!(
            structure.validate_for_solver(no_imperfect, None),
            Err(SolverCompatibilityError::ImperfectInformationUnsupported)
        );

        let cfr_like = SolverCapabilities {
            supports_imperfect_information: true,
            requires_perfect_recall: true,
            supports_chance: true,
            supports_general_sum: false,
        };
        assert_eq!(
            structure.validate_for_solver(cfr_like, None),
            Err(SolverCompatibilityError::PerfectRecallUnverified)
        );
        assert!(structure
            .validate_for_solver(cfr_like, Some(PerfectRecallEvidence::Verified))
            .is_ok());
    }


    #[test]
    fn information_set_action_order_is_not_semantic() {
        let mut structure = valid_information_structure();
        structure.decision_states[1].legal_actions = vec![ActionId(1), ActionId(0)];
        assert_eq!(structure.validate(), Ok(()));
    }

    #[test]
    fn contingent_strategy_rejects_duplicate_point_actions() {
        let points = vec![DecisionPoint {
            player: PlayerId(0),
            information_set: InformationSetId(7),
            legal_actions: vec![ActionId(0), ActionId(0)],
        }];
        let strategy = Strategy {
            player: PlayerId(0),
            decisions: vec![(InformationSetId(7), ActionId(0))],
        };
        assert_eq!(
            strategy.validate(&points),
            Err(ContextError::DuplicateLegalAction(ActionId(0)))
        );
    }

    #[test]
    fn policy_distribution_must_be_legal_and_normalized() {
        let context = context();
        assert!(ContextError::validate_distribution(&context, &[
            MixedAction { action: ActionId(0), probability: 0.25 },
            MixedAction { action: ActionId(1), probability: 0.75 },
        ]).is_ok());
        assert_eq!(
            ContextError::validate_distribution(&context, &[
                MixedAction { action: ActionId(0), probability: 1.0 },
                MixedAction { action: ActionId(9), probability: 0.0 },
            ]),
            Err(ContextError::ProbabilityForIllegalAction(ActionId(9)))
        );
    }

    #[test]
    fn rejects_duplicate_legal_actions() {
        let mut context = context();
        context.legal_actions.push(ActionId(1));
        assert_eq!(context.validate(), Err(ContextError::DuplicateLegalAction(ActionId(1))));
    }

    #[test]
    fn contingent_strategy_requires_exact_legal_coverage() {
        let points = vec![
            DecisionPoint { player: PlayerId(0), information_set: InformationSetId(7), legal_actions: vec![ActionId(0), ActionId(1)] },
            DecisionPoint { player: PlayerId(0), information_set: InformationSetId(8), legal_actions: vec![ActionId(2)] },
        ];
        let complete = Strategy {
            player: PlayerId(0),
            decisions: vec![(InformationSetId(7), ActionId(1)), (InformationSetId(8), ActionId(2))],
        };
        assert!(complete.validate(&points).is_ok());

        let incomplete = Strategy { player: PlayerId(0), decisions: vec![(InformationSetId(7), ActionId(1))] };
        assert_eq!(incomplete.validate(&points), Err(ContextError::MissingDecision(InformationSetId(8))));
    }

    #[test]
    fn rejects_illegal_contingent_action() {
        let points = vec![DecisionPoint {
            player: PlayerId(0),
            information_set: InformationSetId(7),
            legal_actions: vec![ActionId(0)],
        }];
        let strategy = Strategy {
            player: PlayerId(0),
            decisions: vec![(InformationSetId(7), ActionId(1))],
        };
        assert_eq!(
            strategy.validate(&points),
            Err(ContextError::IllegalAction { information_set: InformationSetId(7), action: ActionId(1) })
        );
    }
}
