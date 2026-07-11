// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-kinship
//!
//! The formal, testable core of anthropology: a **kinship algebra**. Given a
//! genealogy (people, parent→child edges, marriages), compute the kin term
//! between any two people — the English/Eskimo terminology system.
//!
//! This is the first of the "hard" knowledge domains
//! (`symthaea/HARD_DOMAINS_PLAN_2026-07-07.md`) built to test the thesis that
//! each hard domain contains a formal, deterministic core. Kinship is that core
//! for anthropology; it hooks into `mycelix-hearth`'s kinship model. The
//! *interpretive* parts of anthropology (culture, symbolism, ethnography) are
//! deliberately out of scope — this computes relations, it does not interpret
//! culture.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link.
//!
//! ## Scope (v0.1)
//!
//! - Consanguineal terms: self, parent…great-grandparent, child…great-grandchild,
//!   sibling, uncle/aunt (…granduncle), niece/nephew (…grandniece), cousins to
//!   arbitrary degree and removal.
//! - Affinal terms: spouse (husband/wife), parent/sibling/child-in-law.
//!
//! Not yet: descent-rule / exogamy checking, half- vs full-sibling distinction,
//! non-English terminology systems (Sudanese/Hawaiian/Iroquois).
//!
//! ## Example
//!
//! ```
//! use symthaea_kinship::{Genealogy, Sex};
//! let mut g = Genealogy::new();
//! g.person("grandpa", Sex::Male).person("dad", Sex::Male).person("ego", Sex::Male);
//! g.parent_of("grandpa", "dad").parent_of("dad", "ego");
//! assert_eq!(g.relation("ego", "grandpa").unwrap(), "grandfather");
//! ```

pub mod genealogy;
pub mod terms;

pub use genealogy::{Genealogy, Sex};
