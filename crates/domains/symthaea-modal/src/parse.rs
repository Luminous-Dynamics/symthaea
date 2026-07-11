// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! A small recursive-descent parser for propositional modal formulas.
//!
//! Accepts both ASCII and Unicode operators, so the same formula can be typed
//! as `[]p -> p` or `□p → p`:
//!
//! | Connective | ASCII | Unicode | Word |
//! |------------|-------|---------|------|
//! | necessity  | `[]`  | `□`     | `box`, `nec` |
//! | possibility| `<>`  | `◇`     | `dia`, `pos` |
//! | negation   | `~` `!` | `¬`   | `not` |
//! | and        | `&`   | `∧`     | `and` |
//! | or         | `\|`  | `∨`     | `or`  |
//! | implies    | `->`  | `→`     | `implies` |
//!
//! Precedence (loosest→tightest): `->` (right-assoc) < `or` < `and` < unary
//! (`~`,`[]`,`<>`) < atom (variable or parenthesised formula).

use crate::kripke::{Formula, and, implies, necessarily, not, or, possibly, var};

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Impl,
    Or,
    And,
    Not,
    Box,
    Dia,
    LParen,
    RParen,
    Var(String),
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            '~' | '!' | '¬' => {
                out.push(Tok::Not);
                i += 1;
            }
            '&' | '∧' => {
                out.push(Tok::And);
                i += 1;
            }
            '|' | '∨' => {
                out.push(Tok::Or);
                i += 1;
            }
            '□' => {
                out.push(Tok::Box);
                i += 1;
            }
            '◇' => {
                out.push(Tok::Dia);
                i += 1;
            }
            '→' => {
                out.push(Tok::Impl);
                i += 1;
            }
            '-' if next == Some('>') => {
                out.push(Tok::Impl);
                i += 2;
            }
            '[' if next == Some(']') => {
                out.push(Tok::Box);
                i += 2;
            }
            '<' if next == Some('>') => {
                out.push(Tok::Dia);
                i += 2;
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                match word.to_lowercase().as_str() {
                    "and" => out.push(Tok::And),
                    "or" => out.push(Tok::Or),
                    "not" => out.push(Tok::Not),
                    "box" | "nec" | "necessarily" => out.push(Tok::Box),
                    "dia" | "diamond" | "pos" | "poss" | "possibly" => out.push(Tok::Dia),
                    "implies" => out.push(Tok::Impl),
                    _ => out.push(Tok::Var(word)),
                }
            }
            other => return Err(format!("unexpected character '{other}'")),
        }
    }
    Ok(out)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    // implication := disjunction ('->' implication)?   (right-associative)
    fn implication(&mut self) -> Result<Formula, String> {
        let lhs = self.disjunction()?;
        if self.peek() == Some(&Tok::Impl) {
            self.bump();
            let rhs = self.implication()?;
            Ok(implies(lhs, rhs))
        } else {
            Ok(lhs)
        }
    }

    // disjunction := conjunction ('or' conjunction)*
    fn disjunction(&mut self) -> Result<Formula, String> {
        let mut lhs = self.conjunction()?;
        while self.peek() == Some(&Tok::Or) {
            self.bump();
            let rhs = self.conjunction()?;
            lhs = or(lhs, rhs);
        }
        Ok(lhs)
    }

    // conjunction := unary ('and' unary)*
    fn conjunction(&mut self) -> Result<Formula, String> {
        let mut lhs = self.unary()?;
        while self.peek() == Some(&Tok::And) {
            self.bump();
            let rhs = self.unary()?;
            lhs = and(lhs, rhs);
        }
        Ok(lhs)
    }

    // unary := ('~' | '[]' | '<>') unary | atom
    fn unary(&mut self) -> Result<Formula, String> {
        match self.peek() {
            Some(Tok::Not) => {
                self.bump();
                Ok(not(self.unary()?))
            }
            Some(Tok::Box) => {
                self.bump();
                Ok(necessarily(self.unary()?))
            }
            Some(Tok::Dia) => {
                self.bump();
                Ok(possibly(self.unary()?))
            }
            _ => self.atom(),
        }
    }

    // atom := VAR | '(' implication ')'
    fn atom(&mut self) -> Result<Formula, String> {
        match self.bump() {
            Some(Tok::Var(name)) => Ok(var(&name)),
            Some(Tok::LParen) => {
                let inner = self.implication()?;
                match self.bump() {
                    Some(Tok::RParen) => Ok(inner),
                    _ => Err("expected ')'".to_string()),
                }
            }
            Some(t) => Err(format!("unexpected token {t:?}")),
            None => Err("unexpected end of formula".to_string()),
        }
    }
}

/// Parse a modal formula. Returns `Err` on malformed input rather than
/// panicking, so callers (e.g. the facade plugin) can fall back gracefully.
pub fn parse(s: &str) -> Result<Formula, String> {
    let toks = tokenize(s)?;
    if toks.is_empty() {
        return Err("empty formula".to_string());
    }
    let mut p = Parser { toks, pos: 0 };
    let f = p.implication()?;
    if p.pos != p.toks.len() {
        return Err(format!("trailing tokens after formula (at {})", p.pos));
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validity::{System, is_valid};

    #[test]
    fn parses_ascii_and_unicode_equivalently() {
        assert_eq!(parse("[]p -> p").unwrap(), parse("□p → p").unwrap());
        assert_eq!(parse("<>p").unwrap(), parse("◇p").unwrap());
        assert_eq!(parse("~p or q").unwrap(), parse("¬p ∨ q").unwrap());
    }

    #[test]
    fn precedence_and_right_assoc() {
        // -> is looser than or/and and right-associative.
        assert_eq!(
            parse("a and b -> c").unwrap(),
            parse("(a and b) -> c").unwrap()
        );
        assert_eq!(
            parse("a -> b -> c").unwrap(),
            parse("a -> (b -> c)").unwrap()
        );
    }

    #[test]
    fn parsed_t_axiom_separates_k_from_t() {
        // The whole point: a parsed formula feeds the real validity checker.
        let t = parse("[]p -> p").unwrap();
        assert!(!is_valid(&t, System::K));
        assert!(is_valid(&t, System::T));
    }

    #[test]
    fn parsed_4_axiom_separates_t_from_s4() {
        let four = parse("[]p -> [][]p").unwrap();
        assert!(!is_valid(&four, System::T));
        assert!(is_valid(&four, System::S4));
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse("p ->").is_err());
        assert!(parse("() p").is_err());
        assert!(parse("").is_err());
    }
}
