// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-coding-theory
//!
//! Error-correcting codes — the bridge between `symthaea-information-theory`
//! (Shannon's noisy-channel theorem) and `symthaea-finite-field` (the algebra of
//! Reed-Solomon). Where information theory says how many errors a channel can
//! survive, coding theory builds the codes that achieve it.
//!
//! Pure `std`, only depends on the sibling `symthaea-finite-field` crate.
//!
//! ## Contents
//! - [`hamming`] — Hamming distance/weight and the single-error-correcting
//!   Hamming(7,4) code
//! - [`repetition`] — repetition code with majority decoding
//! - [`reed_solomon`] — Reed-Solomon encoding + syndrome detection over GF(2⁸)
//!
//! ## Example
//!
//! ```
//! use symthaea_coding_theory::hamming::{hamming74_encode, hamming74_decode};
//! let data = [1, 0, 1, 1];
//! let mut codeword = hamming74_encode(data);
//! codeword[3] ^= 1; // flip one bit anywhere
//! assert_eq!(hamming74_decode(codeword), data); // still recovered
//! ```

pub mod hamming;
pub mod reed_solomon;
pub mod repetition;

pub use hamming::{distance, hamming74_decode, hamming74_encode, weight};
