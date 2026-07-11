// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-pragmatics
//!
//! The formal, rule-based core of linguistic pragmatics: speech-act
//! classification, presupposition-trigger detection, and deixis resolution.
//! Fifth of the "hard" knowledge domains
//! (`symthaea/HARD_DOMAINS_PLAN_2026-07-07.md`).
//!
//! **Non-duplication / scope:** pragmatics is applied theory-of-mind. Rich
//! intent/implicature *inference* belongs to the main crate's ToM + NSM + Broca,
//! which this crate deliberately does NOT reimplement. What it provides is the
//! decidable, rule-based layer beneath that: classifying the illocutionary type,
//! spotting what an utterance presupposes, and resolving context-dependent terms.
//! Open-ended conversational implicature stays out of scope (LLM/ToM territory).
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link.
//!
//! ## Example
//!
//! ```
//! use symthaea_pragmatics::speech_act::{classify, SpeechAct};
//! use symthaea_pragmatics::presupposition::{detect, Trigger};
//! assert_eq!(classify("I promise to help."), SpeechAct::Commissive);
//! assert!(detect("John stopped smoking").contains(&Trigger::Aspectual));
//! ```

pub mod deixis;
pub mod presupposition;
pub mod speech_act;

pub use deixis::{Context, resolve};
pub use presupposition::{Trigger, detect};
pub use speech_act::{SpeechAct, classify};
