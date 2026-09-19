// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain-neutral trust and detached-attestation primitives for Symthaea.
//!
//! This crate deliberately does not decide scientific, fabrication, deployment,
//! or mathematical policy. It provides a shared substrate for:
//! - validated SHA-256 identities;
//! - canonical purpose identifiers;
//! - lifecycle-aware, sequence-numbered trust snapshots;
//! - append-only temporal key lifecycle histories that distinguish retirement,
//!   prospective revocation, replacement, and retroactive compromise;
//! - content-addressed key-to-principal/organization/region/role bindings;
//! - frozen delegated trust-root policies and non-authorizing dual-threshold
//!   rotation contracts;
//! - detached signatures over already content-addressed subjects/payloads;
//! - structural reassessment of historical authority under newly learned
//!   temporal lifecycle facts;
//! - explicit attestation expectations that fail closed on substitution; and
//! - private-field verified capabilities that cannot be deserialized into
//!   authority.
//!
//! Cryptographic providers and private keys stay outside the crate behind narrow
//! signer/verifier traits.

mod attestation;
mod identity;
mod principal;
mod revalidation;
mod temporal_lifecycle;
mod trust;
mod trust_root;

pub use attestation::*;
pub use identity::*;
pub use principal::*;
pub use revalidation::*;
pub use temporal_lifecycle::*;
pub use trust::*;
pub use trust_root::*;
