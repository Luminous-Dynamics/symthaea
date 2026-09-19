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
//! - frozen delegated trust-root policies and dual-threshold rotation contracts;
//! - role-aware root authority capabilities that reuse cryptographic attestation
//!   verification and independently enforce principal/org/region quorum geometry;
//! - bounded interval-based time evidence with explicit source strength;
//! - authenticated trusted-time capabilities over root-authorized time sources;
//! - structural append-only Merkle transparency with inclusion and compact
//!   consistency proofs;
//! - authenticated, time-bound transparency checkpoints with monotonic tracking;
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
mod root_authority;
mod temporal_lifecycle;
mod time;
mod transparency;
mod transparency_checkpoint;
mod trust;
mod trust_root;
mod trusted_time;

#[cfg(test)]
mod root_authority_tests;
#[cfg(test)]
mod trusted_time_tests;

pub use attestation::*;
pub use identity::*;
pub use principal::*;
pub use revalidation::*;
pub use root_authority::*;
pub use temporal_lifecycle::*;
pub use time::*;
pub use transparency::*;
pub use transparency_checkpoint::*;
pub use trust::*;
pub use trust_root::*;
pub use trusted_time::*;
