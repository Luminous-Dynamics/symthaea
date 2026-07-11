// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Context-free grammars in Chomsky Normal Form, recognized by the CYK
//! algorithm — the *context-free* rung of the Chomsky hierarchy, and the reason
//! natural language needs more than regular expressions (nested structure,
//! `aⁿbⁿ`, balanced brackets).
//!
//! A CNF grammar has only two production shapes: `A → B C` (two nonterminals)
//! and `A → a` (one terminal). The empty string is not in the language.

use std::collections::HashSet;

/// A context-free grammar in Chomsky Normal Form. Nonterminals are `String`
/// names; terminals are `char`s.
#[derive(Debug, Clone)]
pub struct Cfg {
    start: String,
    binary: Vec<(String, String, String)>,
    terminal: Vec<(String, char)>,
}

impl Cfg {
    /// A grammar with the given start symbol and no productions yet.
    pub fn new(start: &str) -> Cfg {
        Cfg {
            start: start.to_string(),
            binary: Vec::new(),
            terminal: Vec::new(),
        }
    }

    /// Add a binary production `a → b c` (builder-style).
    pub fn binary_rule(mut self, a: &str, b: &str, c: &str) -> Cfg {
        self.binary
            .push((a.to_string(), b.to_string(), c.to_string()));
        self
    }

    /// Add a terminal production `a → ch` (builder-style).
    pub fn terminal_rule(mut self, a: &str, ch: char) -> Cfg {
        self.terminal.push((a.to_string(), ch));
        self
    }

    /// Whether the grammar derives `input`, via the CYK dynamic program
    /// (O(n³·|productions|)).
    pub fn recognizes(&self, input: &str) -> bool {
        let chars: Vec<char> = input.chars().collect();
        let n = chars.len();
        if n == 0 {
            return false; // CNF cannot derive ε
        }
        // table[len - 1][i] = nonterminals deriving chars[i .. i + len].
        let mut table: Vec<Vec<HashSet<String>>> = vec![vec![HashSet::new(); n]; n];

        // Length-1 substrings via terminal rules.
        for (i, &ch) in chars.iter().enumerate() {
            for (a, t) in &self.terminal {
                if *t == ch {
                    table[0][i].insert(a.clone());
                }
            }
        }

        // Longer substrings via binary rules over every split point.
        for len in 2..=n {
            for i in 0..=n - len {
                for split in 1..len {
                    let left = table[split - 1][i].clone();
                    let right = table[len - split - 1][i + split].clone();
                    if left.is_empty() || right.is_empty() {
                        continue;
                    }
                    for (a, b, c) in &self.binary {
                        if left.contains(b) && right.contains(c) {
                            table[len - 1][i].insert(a.clone());
                        }
                    }
                }
            }
        }

        table[n - 1][0].contains(&self.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grammar for `aⁿbⁿ` (n ≥ 1): S→AB | AC, C→SB, A→a, B→b — the canonical
    /// non-regular context-free language.
    fn an_bn() -> Cfg {
        Cfg::new("S")
            .binary_rule("S", "A", "B")
            .binary_rule("S", "A", "C")
            .binary_rule("C", "S", "B")
            .terminal_rule("A", 'a')
            .terminal_rule("B", 'b')
    }

    #[test]
    fn recognizes_balanced_an_bn() {
        let g = an_bn();
        assert!(g.recognizes("ab"));
        assert!(g.recognizes("aabb"));
        assert!(g.recognizes("aaabbb"));
    }

    #[test]
    fn rejects_unbalanced() {
        let g = an_bn();
        assert!(!g.recognizes("aab")); // too few b
        assert!(!g.recognizes("abb")); // too many b
        assert!(!g.recognizes("ba")); // wrong order
        assert!(!g.recognizes("")); // ε not in language
        assert!(!g.recognizes("aabbb"));
    }

    #[test]
    fn single_terminal_grammar() {
        let g = Cfg::new("S").terminal_rule("S", 'a');
        assert!(g.recognizes("a"));
        assert!(!g.recognizes("aa"));
    }
}
