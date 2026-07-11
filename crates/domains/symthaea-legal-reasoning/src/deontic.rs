// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Standard deontic logic: obligation, permission, prohibition, and the
//! consistency of a set of norms.

/// A deontic status assigned to a named act.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Norm {
    /// O(a): the act is obligatory (must be done).
    Obligatory(String),
    /// P(a): the act is permitted (may be done).
    Permitted(String),
    /// F(a): the act is forbidden (must not be done); equivalently ¬P(a).
    Forbidden(String),
}

impl Norm {
    pub fn act(&self) -> &str {
        match self {
            Norm::Obligatory(a) | Norm::Permitted(a) | Norm::Forbidden(a) => a,
        }
    }
}

/// Acts that are simultaneously forbidden and (obligatory or permitted) — the
/// deontic conflicts. `O(a) → P(a)` and `F(a) → ¬P(a)`, so any such overlap is a
/// contradiction.
pub fn conflicting_acts(norms: &[Norm]) -> Vec<String> {
    let mut conflicts = Vec::new();
    for norm in norms {
        if let Norm::Forbidden(a) = norm {
            let clashes = norms
                .iter()
                .any(|other| matches!(other, Norm::Obligatory(b) | Norm::Permitted(b) if b == a));
            if clashes && !conflicts.contains(a) {
                conflicts.push(a.clone());
            }
        }
    }
    conflicts
}

/// Whether the norm set is deontically consistent (no act both required/allowed
/// and forbidden).
pub fn is_consistent(norms: &[Norm]) -> bool {
    conflicting_acts(norms).is_empty()
}

/// Whether `act` is permitted given the norms: explicitly permitted or
/// obligatory (`O→P`), and not forbidden.
pub fn is_permitted(norms: &[Norm], act: &str) -> bool {
    let forbidden = norms
        .iter()
        .any(|n| matches!(n, Norm::Forbidden(a) if a == act));
    if forbidden {
        return false;
    }
    norms
        .iter()
        .any(|n| matches!(n, Norm::Obligatory(a) | Norm::Permitted(a) if a == act))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consistent_norm_set() {
        let norms = vec![
            Norm::Obligatory("pay_tax".into()),
            Norm::Permitted("park_here".into()),
            Norm::Forbidden("steal".into()),
        ];
        assert!(is_consistent(&norms));
    }

    #[test]
    fn obligation_conflicts_with_prohibition() {
        // Must do and must not do the same act → inconsistent.
        let norms = vec![
            Norm::Obligatory("testify".into()),
            Norm::Forbidden("testify".into()),
        ];
        assert!(!is_consistent(&norms));
        assert_eq!(conflicting_acts(&norms), vec!["testify"]);
    }

    #[test]
    fn permission_conflicts_with_prohibition() {
        let norms = vec![
            Norm::Permitted("enter".into()),
            Norm::Forbidden("enter".into()),
        ];
        assert!(!is_consistent(&norms));
    }

    #[test]
    fn permission_derivation() {
        let norms = vec![
            Norm::Obligatory("vote".into()),
            Norm::Forbidden("bribe".into()),
        ];
        assert!(is_permitted(&norms, "vote")); // O(a) → P(a)
        assert!(!is_permitted(&norms, "bribe")); // F(a) → ¬P(a)
        assert!(!is_permitted(&norms, "unlisted")); // silent → not permitted here
    }
}
