// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-legal-reasoning
//!
//! The formal, testable core of law — deontic logic, defeasible rules, and
//! Hohfeldian jural relations. Third of the "hard" knowledge domains
//! (`symthaea/HARD_DOMAINS_PLAN_2026-07-07.md`), hooking into the governance/civic
//! clusters and the `EthicsEngine`.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link.
//!
//! **Scope note:** this engine *applies and checks* rules — norm consistency,
//! default-with-exception derivation, jural correlativity. It does NOT interpret
//! statutes, reason by analogy to precedent, or decide what the law *should* be;
//! those interpretive parts of law stay out of scope.
//!
//! ## Scope
//!
//! - [`deontic`]: obligation/permission/prohibition + norm-set consistency.
//! - [`defeasible`]: defaults with exceptions, forward-chained to a fixpoint.
//! - [`hohfeld`]: the eight jural positions with correlatives and opposites.
//!
//! ## Example
//!
//! ```
//! use symthaea_legal_reasoning::deontic::{Norm, is_consistent};
//! let norms = vec![Norm::Obligatory("testify".into()), Norm::Forbidden("testify".into())];
//! assert!(!is_consistent(&norms)); // can't be both required and forbidden
//! ```

pub mod defeasible;
pub mod deontic;
pub mod hohfeld;

pub use defeasible::{Rule, derive, entails};
pub use deontic::{Norm, is_consistent, is_permitted};
pub use hohfeld::Jural;
