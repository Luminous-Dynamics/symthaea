// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Presupposition-trigger detection — the parts of an utterance taken for
//! granted regardless of the main assertion (and its negation).

/// A kind of presupposition trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Trigger {
    /// "the X" — presupposes X exists (existence presupposition).
    DefiniteDescription,
    /// factive verb (know/realize/regret) — presupposes the complement is true.
    Factive,
    /// aspectual (stop/continue V-ing) — presupposes the prior state held.
    Aspectual,
    /// cleft ("it was X who…") — presupposes someone did it.
    Cleft,
}

const FACTIVES: &[&str] = &[
    "know",
    "knows",
    "knew",
    "realize",
    "realizes",
    "realized",
    "realise",
    "regret",
    "regrets",
    "regretted",
    "aware that",
];
const ASPECTUALS: &[&str] = &[
    "stop",
    "stopped",
    "stops",
    "continue",
    "continued",
    "resume",
];

/// Detect presupposition triggers present in an utterance (sorted, unique).
pub fn detect(utterance: &str) -> Vec<Trigger> {
    let u = format!(" {} ", utterance.trim().to_lowercase());
    let mut out = Vec::new();
    let push = |t: Trigger, out: &mut Vec<Trigger>| {
        if !out.contains(&t) {
            out.push(t);
        }
    };

    if u.contains(" the ") {
        push(Trigger::DefiniteDescription, &mut out);
    }
    if FACTIVES.iter().any(|k| u.contains(&format!(" {k} "))) {
        push(Trigger::Factive, &mut out);
    }
    if ASPECTUALS.iter().any(|k| u.contains(&format!(" {k} "))) {
        push(Trigger::Aspectual, &mut out);
    }
    if u.contains("it was ") && u.contains(" who ") {
        push(Trigger::Cleft, &mut out);
    }

    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definite_description() {
        // "The king of France is bald" presupposes a king of France exists.
        assert!(detect("The king of France is bald").contains(&Trigger::DefiniteDescription));
    }

    #[test]
    fn factive() {
        // "Mary realizes it's raining" presupposes it is raining.
        assert!(detect("Mary realizes it is raining").contains(&Trigger::Factive));
    }

    #[test]
    fn aspectual() {
        // "John stopped smoking" presupposes John used to smoke.
        assert!(detect("John stopped smoking").contains(&Trigger::Aspectual));
    }

    #[test]
    fn cleft() {
        assert!(detect("It was John who broke the window").contains(&Trigger::Cleft));
    }

    #[test]
    fn plain_assertion_has_no_triggers() {
        assert!(detect("Rain fell").is_empty());
    }
}
