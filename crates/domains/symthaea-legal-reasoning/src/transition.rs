// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hohfeldian legal-position state and power exercises.
//!
//! A power is represented as authority to apply an explicitly supplied atomic
//! mutation. This module verifies possession of that power, preserves
//! correlative closure, and rejects contradictory post-states. It does not infer
//! the legal effects of natural-language instruments.

use crate::hohfeld::{Jural, JuralRelation};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// A closed set of instantiated Hohfeldian relations.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LegalPositionState {
    relations: BTreeSet<JuralRelation>,
}

/// Failure to construct or mutate a legal-position state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionError {
    ContradictoryPosition {
        existing: JuralRelation,
        proposed: JuralRelation,
    },
    ExerciseRequiresPower {
        relation: JuralRelation,
    },
    PowerNotHeld {
        relation: JuralRelation,
    },
}

impl fmt::Display for TransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransitionError::ContradictoryPosition { existing, proposed } => write!(
                f,
                "proposed legal position {proposed:?} contradicts existing position {existing:?}"
            ),
            TransitionError::ExerciseRequiresPower { relation } => write!(
                f,
                "legal transition must be authorized by a Power relation, got {relation:?}"
            ),
            TransitionError::PowerNotHeld { relation } => write!(
                f,
                "authorizing legal power is not present in the state: {relation:?}"
            ),
        }
    }
}

impl Error for TransitionError {}

impl LegalPositionState {
    /// Construct a state and materialize every supplied relation's correlative.
    pub fn new(
        assertions: impl IntoIterator<Item = JuralRelation>,
    ) -> Result<Self, TransitionError> {
        let mut state = Self::default();
        for relation in assertions {
            state.assert_relation(relation)?;
        }
        Ok(state)
    }

    pub fn relations(&self) -> impl Iterator<Item = &JuralRelation> {
        self.relations.iter()
    }

    pub fn contains(&self, relation: &JuralRelation) -> bool {
        self.relations.contains(relation)
    }

    /// Add a relation and its correlative as one consistency-checked operation.
    pub fn assert_relation(&mut self, relation: JuralRelation) -> Result<(), TransitionError> {
        self.ensure_noncontradictory(&relation)?;
        let correlative = relation.correlative_relation();
        self.ensure_noncontradictory(&correlative)?;
        self.relations.insert(relation);
        self.relations.insert(correlative);
        Ok(())
    }

    /// Remove an exact assertion and its exact correlative.
    pub fn retract_relation(&mut self, relation: &JuralRelation) {
        self.relations.remove(relation);
        self.relations.remove(&relation.correlative_relation());
    }

    pub fn is_correlatively_closed(&self) -> bool {
        self.relations
            .iter()
            .all(|relation| self.relations.contains(&relation.correlative_relation()))
    }

    fn ensure_noncontradictory(&self, proposed: &JuralRelation) -> Result<(), TransitionError> {
        if let Some(existing) = self
            .relations
            .iter()
            .find(|existing| existing.is_opposite_of(proposed))
        {
            return Err(TransitionError::ContradictoryPosition {
                existing: existing.clone(),
                proposed: proposed.clone(),
            });
        }
        Ok(())
    }
}

/// An explicitly formalized legal mutation authorized by a held power.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerExercise {
    pub power: JuralRelation,
    pub retract: Vec<JuralRelation>,
    pub assert: Vec<JuralRelation>,
}

impl PowerExercise {
    pub fn new(power: JuralRelation) -> Self {
        Self {
            power,
            retract: Vec::new(),
            assert: Vec::new(),
        }
    }

    pub fn retract(mut self, relation: JuralRelation) -> Self {
        self.retract.push(relation);
        self
    }

    pub fn assert(mut self, relation: JuralRelation) -> Self {
        self.assert.push(relation);
        self
    }
}

/// Canonical record of an accepted atomic legal transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionRecord {
    pub power: JuralRelation,
    pub retracted: Vec<JuralRelation>,
    pub asserted: Vec<JuralRelation>,
}

/// Apply a power exercise transactionally.
///
/// On error, `state` is unchanged. The returned state remains correlatively
/// closed because every assertion and retraction operates on both sides.
pub fn exercise_power(
    state: &LegalPositionState,
    exercise: &PowerExercise,
) -> Result<(LegalPositionState, TransitionRecord), TransitionError> {
    if exercise.power.position != Jural::Power {
        return Err(TransitionError::ExerciseRequiresPower {
            relation: exercise.power.clone(),
        });
    }
    if !state.contains(&exercise.power) {
        return Err(TransitionError::PowerNotHeld {
            relation: exercise.power.clone(),
        });
    }

    let mut next = state.clone();
    let mut retracted = exercise.retract.clone();
    retracted.sort_unstable();
    retracted.dedup();
    for relation in &retracted {
        next.retract_relation(relation);
    }

    let mut asserted = exercise.assert.clone();
    asserted.sort_unstable();
    asserted.dedup();
    for relation in &asserted {
        next.assert_relation(relation.clone())?;
    }

    debug_assert!(next.is_correlatively_closed());
    Ok((
        next,
        TransitionRecord {
            power: exercise.power.clone(),
            retracted,
            asserted,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ActionId, PartyId};

    fn relation(
        holder: &str,
        counterparty: &str,
        position: Jural,
        action: &str,
    ) -> JuralRelation {
        JuralRelation::new(
            PartyId::new(holder).unwrap(),
            PartyId::new(counterparty).unwrap(),
            position,
            ActionId::new(action).unwrap(),
        )
    }

    #[test]
    fn state_materializes_correlatives() {
        let right = relation("creditor", "debtor", Jural::Right, "pay");
        let state = LegalPositionState::new([right.clone()]).unwrap();

        assert!(state.contains(&right));
        assert!(state.contains(&right.correlative_relation()));
        assert!(state.is_correlatively_closed());
    }

    #[test]
    fn contradictory_positions_are_rejected() {
        let right = relation("creditor", "debtor", Jural::Right, "pay");
        let result = LegalPositionState::new([right.clone(), right.opposite_relation()]);
        assert!(matches!(
            result,
            Err(TransitionError::ContradictoryPosition { .. })
        ));
    }

    #[test]
    fn held_power_can_apply_atomic_position_change() {
        let power = relation("court", "debtor", Jural::Power, "enter-judgment");
        let proposed_right = relation("creditor", "debtor", Jural::Right, "pay-judgment");
        let state = LegalPositionState::new([power.clone()]).unwrap();
        let exercise = PowerExercise::new(power).assert(proposed_right.clone());

        let (next, record) = exercise_power(&state, &exercise).unwrap();
        assert!(next.contains(&proposed_right));
        assert!(next.contains(&proposed_right.correlative_relation()));
        assert_eq!(record.asserted, vec![proposed_right]);
    }

    #[test]
    fn absent_or_non_power_authority_fails_closed() {
        let absent_power = relation("court", "debtor", Jural::Power, "enter-judgment");
        let state = LegalPositionState::new([]).unwrap();
        assert!(matches!(
            exercise_power(&state, &PowerExercise::new(absent_power)),
            Err(TransitionError::PowerNotHeld { .. })
        ));

        let right = relation("creditor", "debtor", Jural::Right, "pay");
        let state = LegalPositionState::new([right.clone()]).unwrap();
        assert!(matches!(
            exercise_power(&state, &PowerExercise::new(right)),
            Err(TransitionError::ExerciseRequiresPower { .. })
        ));
    }
}
