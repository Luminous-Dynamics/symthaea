// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Epistemic-governance policy for Recursive Cognitive Architecture v1.
//!
//! `symthaea-types` owns canonical cognitive proposal/evidence wire types.
//! This crate owns policy that reasons over those artifacts: canonical
//! evidence-object identity, lineage, independence, currentness, defeaters,
//! relation-declaration provenance, use-specific declarer qualification, and
//! experiment qualification.
//! It does not own action authority or recursive-improvement promotion.

#![deny(unsafe_code)]

pub mod currentness;
pub mod experiment_contract;
pub mod identity;
pub mod lineage;
pub mod relation_provenance;
pub mod relation_qualification;
