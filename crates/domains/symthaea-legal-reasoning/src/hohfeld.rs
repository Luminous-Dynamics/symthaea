// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hohfeld's eight fundamental jural relations and their correlatives/opposites.
//!
//! Hohfeld (1913) analysed legal positions into four correlative pairs:
//! Right↔Duty, Privilege↔No-right, Power↔Liability, Immunity↔Disability. Each
//! also has a jural *opposite*.

/// One of Hohfeld's eight jural positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[cfg(test)]
mod tests {
    use super::Jural::*;

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
}
