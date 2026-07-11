// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-group-theory
//!
//! Finite group theory — the core of abstract algebra, which the workspace had
//! only in physics-specific form (`particle-physics`'s gauge/symmetry groups)
//! and Lie-algebra representation theory (`symthaea-core`'s `lie_theory`). This
//! is the general finite-group layer.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked against known
//! structure (Zₙ orders, S₃ non-abelian, Lagrange's theorem).
//!
//! ## Contents
//! - [`permutation::Permutation`] — one-line permutations: compose, inverse,
//!   cycle decomposition, order, sign
//! - [`group::CayleyGroup`] — finite groups by Cayley table: identity, inverse,
//!   element order, generated subgroup, abelian test, Lagrange; `cyclic` and
//!   `symmetric` constructors
//!
//! ## Example
//!
//! ```
//! use symthaea_group_theory::CayleyGroup;
//! // S₃ is the smallest non-abelian group, and Lagrange holds.
//! let s3 = CayleyGroup::symmetric(3);
//! assert_eq!(s3.order(), 6);
//! assert!(!s3.is_abelian());
//! assert!(s3.lagrange_holds());
//! ```

pub mod group;
pub mod permutation;

pub use group::CayleyGroup;
pub use permutation::Permutation;
