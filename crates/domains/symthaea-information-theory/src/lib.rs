// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-information-theory
//!
//! Shannon information theory — entropy, mutual information, KL divergence,
//! channel capacity, and source coding. The primitives existed scattered inline
//! (`alife`, `aesthetic`, `embeddings/channel.rs`, `frontier-physics`); this is
//! the clean, tested home, and the bridge from ML/HDC to coding theory
//! (`symthaea-coding-theory`) and cryptography.
//!
//! All measures are in **bits** (base-2). Pure `std`, zero dependencies, no
//! `symthaea-core` link.
//!
//! ## Contents
//! - [`entropy`] — entropy, joint/conditional entropy, mutual information, KL
//!   divergence, cross-entropy
//! - [`channel`] — binary symmetric / erasure channel capacity
//! - [`huffman`] — Huffman code lengths + the Shannon source-coding bound
//!
//! ## Example
//!
//! ```
//! use symthaea_information_theory::entropy::{entropy, mutual_information};
//! assert!((entropy(&[0.5, 0.5]) - 1.0).abs() < 1e-12); // fair coin = 1 bit
//! // Perfectly correlated variables share all their information.
//! let joint = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
//! assert!((mutual_information(&joint) - 1.0).abs() < 1e-12);
//! ```

pub mod channel;
pub mod entropy;
pub mod huffman;

pub use channel::{bec_capacity, bsc_capacity};
pub use entropy::{
    binary_entropy, conditional_entropy_y_given_x, cross_entropy, entropy, joint_entropy,
    kl_divergence, mutual_information,
};
