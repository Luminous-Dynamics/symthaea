// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-formal-language
//!
//! The computable core of the **Chomsky hierarchy** — the syntactic layer the
//! workspace was missing. Symthaea had semantics/pragmatics
//! (`symthaea-pragmatics`) and logic (`symthaea-modal`), but nothing for the
//! *structure* of strings: what makes a language regular vs. context-free, and
//! why natural language needs more than pattern matching.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Every recognizer is
//! a real automaton / algorithm, checked against textbook languages.
//!
//! ## Layers
//! - [`dfa`] — deterministic finite automata (regular languages)
//! - [`regex`] — a regex engine: parse → Thompson NFA → ε-closure simulation
//!   (also regular, but built from an automaton, not `str::contains`)
//! - [`grammar`] — context-free grammars in CNF, recognized by CYK
//!   (the non-regular rung: `aⁿbⁿ`, nesting)
//!
//! ## Example
//!
//! ```
//! use symthaea_formal_language::grammar::Cfg;
//! // aⁿbⁿ is context-free but NOT regular — no DFA/regex can recognize it.
//! let g = Cfg::new("S")
//!     .binary_rule("S", "A", "B").binary_rule("S", "A", "C")
//!     .binary_rule("C", "S", "B")
//!     .terminal_rule("A", 'a').terminal_rule("B", 'b');
//! assert!(g.recognizes("aaabbb"));
//! assert!(!g.recognizes("aabbb"));
//! ```

pub mod dfa;
pub mod grammar;
pub mod regex;

pub use dfa::Dfa;
pub use grammar::Cfg;
pub use regex::Regex;
