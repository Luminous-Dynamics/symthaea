// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic finite automata — the recognizer for regular languages.

use std::collections::{HashMap, HashSet};

/// A deterministic finite automaton over `char` symbols. States are `usize`;
/// state `0` is not special — `start` names the initial state.
#[derive(Debug, Clone, Default)]
pub struct Dfa {
    transitions: HashMap<(usize, char), usize>,
    accept: HashSet<usize>,
    start: usize,
    n_states: usize,
}

impl Dfa {
    /// A DFA with `n_states` states (ids `0..n_states`) and the given start.
    pub fn new(n_states: usize, start: usize) -> Dfa {
        Dfa {
            transitions: HashMap::new(),
            accept: HashSet::new(),
            start,
            n_states,
        }
    }

    /// Add a transition δ(from, on) = to (builder-style).
    pub fn transition(mut self, from: usize, on: char, to: usize) -> Dfa {
        self.transitions.insert((from, on), to);
        self
    }

    /// Mark a state as accepting.
    pub fn accepting(mut self, state: usize) -> Dfa {
        self.accept.insert(state);
        self
    }

    /// Whether the DFA accepts `input`. A missing transition is an implicit
    /// dead state (reject).
    pub fn accepts(&self, input: &str) -> bool {
        let mut state = self.start;
        for ch in input.chars() {
            match self.transitions.get(&(state, ch)) {
                Some(&next) => state = next,
                None => return false,
            }
        }
        self.accept.contains(&state)
    }

    /// Number of states.
    pub fn state_count(&self) -> usize {
        self.n_states
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DFA accepting binary strings with an even number of 1s.
    fn even_ones() -> Dfa {
        // state 0 = even (accept), state 1 = odd.
        Dfa::new(2, 0)
            .transition(0, '0', 0)
            .transition(0, '1', 1)
            .transition(1, '0', 1)
            .transition(1, '1', 0)
            .accepting(0)
    }

    #[test]
    fn even_number_of_ones() {
        let d = even_ones();
        assert!(d.accepts("")); // zero 1s
        assert!(d.accepts("11"));
        assert!(d.accepts("0110"));
        assert!(!d.accepts("1"));
        assert!(!d.accepts("10110")); // three 1s
    }

    #[test]
    fn missing_transition_rejects() {
        let d = Dfa::new(1, 0).transition(0, 'a', 0).accepting(0);
        assert!(d.accepts("aaa"));
        assert!(!d.accepts("aba")); // 'b' has no transition
    }
}
