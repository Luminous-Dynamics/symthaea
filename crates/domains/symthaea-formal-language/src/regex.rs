// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! A small regular-expression engine: parse a pattern, compile it to an NFA by
//! Thompson's construction, and match by ε-closure simulation. This is the
//! *regular* rung of the Chomsky hierarchy, done honestly (real automaton, not
//! a `str::contains` shortcut).
//!
//! Supported syntax: literals, concatenation, alternation `|`, Kleene star `*`,
//! one-or-more `+`, optional `?`, grouping `(…)`, and `\c` to escape a
//! metacharacter. Matching is whole-string (anchored).

use std::collections::HashSet;

// --- AST ------------------------------------------------------------------

enum Re {
    Epsilon,
    Lit(char),
    Cat(Box<Re>, Box<Re>),
    Alt(Box<Re>, Box<Re>),
    Star(Box<Re>),
    Plus(Box<Re>),
    Opt(Box<Re>),
}

struct ReParser {
    chars: Vec<char>,
    pos: usize,
}

impl ReParser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        self.pos += 1;
        c
    }

    // alt := concat ('|' concat)*
    fn alt(&mut self) -> Result<Re, String> {
        let mut left = self.concat()?;
        while self.peek() == Some('|') {
            self.bump();
            let right = self.concat()?;
            left = Re::Alt(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // concat := postfix*  (empty ⇒ ε)
    fn concat(&mut self) -> Result<Re, String> {
        let mut node: Option<Re> = None;
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let next = self.postfix()?;
            node = Some(match node {
                None => next,
                Some(prev) => Re::Cat(Box::new(prev), Box::new(next)),
            });
        }
        Ok(node.unwrap_or(Re::Epsilon))
    }

    // postfix := atom ('*' | '+' | '?')*
    fn postfix(&mut self) -> Result<Re, String> {
        let mut node = self.atom()?;
        while let Some(c) = self.peek() {
            node = match c {
                '*' => Re::Star(Box::new(node)),
                '+' => Re::Plus(Box::new(node)),
                '?' => Re::Opt(Box::new(node)),
                _ => break,
            };
            self.bump();
        }
        Ok(node)
    }

    // atom := '(' alt ')' | '\' char | char
    fn atom(&mut self) -> Result<Re, String> {
        match self.bump() {
            Some('(') => {
                let inner = self.alt()?;
                match self.bump() {
                    Some(')') => Ok(inner),
                    _ => Err("expected ')'".to_string()),
                }
            }
            Some('\\') => match self.bump() {
                Some(c) => Ok(Re::Lit(c)),
                None => Err("trailing backslash".to_string()),
            },
            Some(c @ ('*' | '+' | '?')) => Err(format!("unexpected quantifier '{c}'")),
            Some(c) => Ok(Re::Lit(c)),
            None => Err("unexpected end of pattern".to_string()),
        }
    }
}

// --- NFA (Thompson) -------------------------------------------------------

#[derive(Default, Clone)]
struct NfaState {
    eps: Vec<usize>,
    trans: Vec<(char, usize)>,
}

/// A compiled regular expression, held as a Thompson NFA.
pub struct Regex {
    states: Vec<NfaState>,
    start: usize,
    accept: usize,
}

struct Builder {
    states: Vec<NfaState>,
}

impl Builder {
    fn state(&mut self) -> usize {
        self.states.push(NfaState::default());
        self.states.len() - 1
    }
    fn eps(&mut self, from: usize, to: usize) {
        self.states[from].eps.push(to);
    }
    fn sym(&mut self, from: usize, on: char, to: usize) {
        self.states[from].trans.push((on, to));
    }

    /// Compile a node to an NFA fragment, returning (start, accept).
    fn compile(&mut self, re: &Re) -> (usize, usize) {
        match re {
            Re::Epsilon => {
                let s = self.state();
                let a = self.state();
                self.eps(s, a);
                (s, a)
            }
            Re::Lit(c) => {
                let s = self.state();
                let a = self.state();
                self.sym(s, *c, a);
                (s, a)
            }
            Re::Cat(x, y) => {
                let (sx, ax) = self.compile(x);
                let (sy, ay) = self.compile(y);
                self.eps(ax, sy);
                (sx, ay)
            }
            Re::Alt(x, y) => {
                let s = self.state();
                let a = self.state();
                let (sx, ax) = self.compile(x);
                let (sy, ay) = self.compile(y);
                self.eps(s, sx);
                self.eps(s, sy);
                self.eps(ax, a);
                self.eps(ay, a);
                (s, a)
            }
            Re::Star(x) => {
                let s = self.state();
                let a = self.state();
                let (sx, ax) = self.compile(x);
                self.eps(s, sx);
                self.eps(s, a);
                self.eps(ax, sx);
                self.eps(ax, a);
                (s, a)
            }
            Re::Plus(x) => {
                let (sx, ax) = self.compile(x);
                let a = self.state();
                self.eps(ax, sx);
                self.eps(ax, a);
                (sx, a)
            }
            Re::Opt(x) => {
                let s = self.state();
                let a = self.state();
                let (sx, ax) = self.compile(x);
                self.eps(s, sx);
                self.eps(s, a);
                self.eps(ax, a);
                (s, a)
            }
        }
    }
}

impl Regex {
    /// Parse and compile a pattern. Returns `Err` on malformed syntax.
    pub fn new(pattern: &str) -> Result<Regex, String> {
        let mut parser = ReParser {
            chars: pattern.chars().collect(),
            pos: 0,
        };
        let ast = parser.alt()?;
        if parser.pos != parser.chars.len() {
            return Err(format!("unexpected '{}'", parser.chars[parser.pos]));
        }
        let mut b = Builder { states: Vec::new() };
        let (start, accept) = b.compile(&ast);
        Ok(Regex {
            states: b.states,
            start,
            accept,
        })
    }

    fn eps_closure(&self, set: HashSet<usize>) -> HashSet<usize> {
        let mut stack: Vec<usize> = set.iter().copied().collect();
        let mut closure = set;
        while let Some(s) = stack.pop() {
            for &t in &self.states[s].eps {
                if closure.insert(t) {
                    stack.push(t);
                }
            }
        }
        closure
    }

    /// Whether the pattern matches the whole `input` (anchored match).
    pub fn matches(&self, input: &str) -> bool {
        let mut current = self.eps_closure(HashSet::from([self.start]));
        for ch in input.chars() {
            let mut next = HashSet::new();
            for &s in &current {
                for &(c, t) in &self.states[s].trans {
                    if c == ch {
                        next.insert(t);
                    }
                }
            }
            if next.is_empty() {
                return false;
            }
            current = self.eps_closure(next);
        }
        current.contains(&self.accept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alternation_and_star() {
        let re = Regex::new("a(b|c)*d").unwrap();
        assert!(re.matches("ad"));
        assert!(re.matches("abd"));
        assert!(re.matches("abcbcd"));
        assert!(!re.matches("aed"));
        assert!(!re.matches("abc")); // no trailing d
    }

    #[test]
    fn plus_and_optional() {
        let re = Regex::new("ab+c?").unwrap();
        assert!(re.matches("ab"));
        assert!(re.matches("abbb"));
        assert!(re.matches("abbc"));
        assert!(!re.matches("ac")); // needs ≥1 b
        assert!(!re.matches("abcc")); // at most one c
    }

    #[test]
    fn anchored_whole_string() {
        let re = Regex::new("cat").unwrap();
        assert!(re.matches("cat"));
        assert!(!re.matches("cats")); // anchored: no partial match
        assert!(!re.matches("scat"));
    }

    #[test]
    fn escaped_metacharacter() {
        let re = Regex::new(r"a\*b").unwrap();
        assert!(re.matches("a*b"));
        assert!(!re.matches("ab"));
    }

    #[test]
    fn malformed_patterns_error() {
        assert!(Regex::new("(a").is_err());
        assert!(Regex::new("*a").is_err());
    }
}
