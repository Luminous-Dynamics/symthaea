// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-lambda
//!
//! The untyped lambda calculus — the foundational model of computation, and
//! directly relevant to Symthaea's own code/logic reasoning. Terms use de
//! Bruijn indices, so substitution is capture-free and α-equivalence is
//! structural equality.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked by running
//! Church arithmetic and the combinator calculus to normal form.
//!
//! ## Contents
//! - [`term`] — [`term::Term`], capture-free substitution, normal-order
//!   `step`/`normalize`
//! - [`church`] — Church numerals/booleans, `add`/`mul`/`succ`, and the S/K/I
//!   combinators
//!
//! ## Example
//!
//! ```
//! use symthaea_lambda::church::{add, numeral};
//! use symthaea_lambda::term::{app, normalize};
//! // add 2 3 reduces to the Church numeral 5.
//! let t = app(app(add(), numeral(2)), numeral(3));
//! assert_eq!(normalize(&t, 1000).unwrap(), numeral(5));
//! ```

pub mod church;
pub mod term;

pub use term::{Term, normalize, step};
