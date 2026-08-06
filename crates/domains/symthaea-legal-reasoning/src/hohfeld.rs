// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hohfeld's eight fundamental jural relations and their correlatives/opposites.
//!
//! Hohfeld (1913) analysed legal positions into four correlative pairs:
//! Right↔Duty, Privilege↔No-right, Power↔Liability, Immunity↔Disability. Each
//! also has a jural *opposite*.

use crate::model::{ActionId, PartyId, SourceRef};

/// One of Hohfeld's eight jural positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Jural {
    Right,
    Duty,
    Privilege,
    NoRight,
    Power,
    Liability,
    Immunity,
    Disability,
}

impl Jural {
    /// The jural **correlative** — the position the *other* party necessarily
    /// holds. If A has a Right, B has the correlative Duty.
    pub fn correlative(self) -> Jural {
        use Jural::*;
        match self {
            Right => Duty,
            Duty => Right,
            Privilege => NoRight,
            NoRight => Privilege,
            Power => Liability,
            Liability => Power,
            Immunity => Disability,
            Disability => Immunity,
        }
    }

    /// The jural **opposite** — the negation of the position for the same party.
    pub fn opposite(self) -> Jural {
        use Jural::*;
        match self {
            Right => NoRight,
            NoRight => Right,
            Privilege => Duty,
            Duty => Privilege,
            Power => Disability,
            Disability => Power,
            Immunity => Liability,
            Liability => Immunity,
        }
    }
}

/// A fully instantiated Hohfeldian position between two parties.
///
/// `holder` owns `position`; `counterparty` necessarily owns the correlative
/// position with respect to the same legally relevant action.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct JuralRelation {
    pub holder: PartyId,
    pub counterparty: PartyId,
    pub position: Jural,
    pub action: ActionId,
    pub source: Option<SourceRef>,
}

impl JuralRelation {
    pub fn new(holder: PartyId, counterparty: PartyId, position: Jural, action: ActionId) -> Self {
        Self {
            holder,
            counterparty,
            position,
            action,
            source: None,
        }
    }

    pub fn with_source(mut self, source: SourceRef) -> Self {
        self.source = Some(source);
        self
    }

    /// Materialize the necessarily corresponding position of the other party.
    pub fn correlative_relation(&self) -> Self {
        Self {
            holder: self.counterparty.clone(),
            counterparty: self.holder.clone(),
            position: self.position.correlative(),
            action: self.action.clone(),
            source: self.source.clone(),
        }
    }

    /// Materialize the logical opposite for the same holder and counterparty.
    pub fn opposite_relation(&self) -> Self {
        Self {
            holder: self.holder.clone(),
            counterparty: self.counterparty.clone(),
            position: self.position.opposite(),
            action: self.action.clone(),
            source: self.source.clone(),
        }
    }

    pub fn is_correlative_of(&self, other: &Self) -> bool {
        self.holder == other.counterparty
            && self.counterparty == other.holder
            && self.action == other.action
            && self.position.correlative() == other.position
    }

    pub fn is_opposite_of(&self, other: &Self) -> bool {
        self.holder == other.holder
            && self.counterparty == other.counterparty
            && self.action == other.action
            && self.position.opposite() == other.position
    }
}

/// Find contradictory positions asserted for the same party relation and action.
///
/// Each pair is returned once as indices into the supplied relation slice.
pub fn contradictory_relations(relations: &[JuralRelation]) -> Vec<(usize, usize)> {
    let mut conflicts = Vec::new();
    for left in 0..relations.len() {
        for right in (left + 1)..relations.len() {
            if relations[left].is_opposite_of(&relations[right]) {
                conflicts.push((left, right));
            }
        }
    }
    conflicts
}

#[cfg(test)]
mod tests {
    use super::{Jural::*, *};
    use crate::model::{ActionId, PartyId};

    #[test]
    fn correlatives() {
        assert_eq!(Right.correlative(), Duty);
        assert_eq!(Power.correlative(), Liability);
        assert_eq!(Privilege.correlative(), NoRight);
        assert_eq!(Immunity.correlative(), Disability);
    }

    #[test]
    fn opposites() {
        assert_eq!(Right.opposite(), NoRight);
        assert_eq!(Power.opposite(), Disability);
        assert_eq!(Privilege.opposite(), Duty);
        assert_eq!(Immunity.opposite(), Liability);
    }

    #[test]
    fn both_relations_are_involutions() {
        // Applying correlative or opposite twice returns the original.
        for j in [
            Right, Duty, Privilege, NoRight, Power, Liability, Immunity, Disability,
        ] {
            assert_eq!(j.correlative().correlative(), j);
            assert_eq!(j.opposite().opposite(), j);
        }
    }

    fn payment_right() -> JuralRelation {
        JuralRelation::new(
            PartyId::new("creditor").unwrap(),
            PartyId::new("debtor").unwrap(),
            Right,
            ActionId::new("pay_debt").unwrap(),
        )
    }

    #[test]
    fn relational_correlative_swaps_parties() {
        let right = payment_right();
        let duty = right.correlative_relation();

        assert_eq!(duty.holder.as_str(), "debtor");
        assert_eq!(duty.counterparty.as_str(), "creditor");
        assert_eq!(duty.position, Duty);
        assert!(right.is_correlative_of(&duty));
        assert_eq!(duty.correlative_relation(), right);
    }

    #[test]
    fn relational_opposites_preserve_parties_and_content() {
        let right = payment_right();
        let no_right = right.opposite_relation();

        assert_eq!(no_right.position, NoRight);
        assert!(right.is_opposite_of(&no_right));
        assert_eq!(no_right.opposite_relation(), right);
    }

    #[test]
    fn contradictory_relation_scan_is_pairwise_and_deterministic() {
        let right = payment_right();
        let no_right = right.opposite_relation();
        let duty = right.correlative_relation();
        let relations = vec![right, duty, no_right];

        assert_eq!(contradictory_relations(&relations), vec![(0, 2)]);
    }
}
