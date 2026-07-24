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
//! - [`channel`] — deterministic bit, symbol, erasure, and burst channel models
//! - [`experiments`] — reproducible end-to-end decoder evidence and manifests
//! - [`parameters`] — shared minimum-distance and correction-capability metadata
//! - [`reliability`] — analytical independent-errata recovery guarantees
//! - [`interoperability`] — explicit profiles and independent golden vectors
//! - [`interleaving`] — checked rectangular burst-dispersing permutations
//! - [`hamming`] — Hamming distance/weight, Hamming(7,4), and Hamming(8,4) SECDED
//! - [`repetition`] — validated odd repetition codes with strict majority decoding
//! - [`reed_solomon`] — checked Reed-Solomon encoding, correction, mixed
//!   errata decoding, decode policies, shortened/fixed frames, and streaming parity over GF(2⁸)
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

pub mod channel;
pub mod experiments;
pub mod hamming;
pub mod interoperability;
pub mod interleaving;
pub mod parameters;
pub mod reed_solomon;
pub mod reliability;
pub mod repetition;

pub use hamming::{
    distance, hamming74_decode, hamming74_encode, hamming84_decode, hamming84_encode, weight,
};
