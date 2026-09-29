//! Information-aware strategy contracts for the canonical Strategic IR.
//!
//! Agent policies receive only an explicit observation and legal-action set.
//! They do not receive the full world state by default. This module defines
//! representation and validation contracts; it does not implement a solver.

use crate::strategic::{ActionId, PlayerId};

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
            if point.legal_actions.is_empty() {
                return Err(ContextError::NoLegalActions);
            }
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

/// Validation failures for information-aware decision contracts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    NoLegalActions,
    DuplicateLegalAction(ActionId),
    DuplicateInformationSet(InformationSetId),
    PlayerMismatch { information_set: InformationSetId, expected: PlayerId, actual: PlayerId },
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
