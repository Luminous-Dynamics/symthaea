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
//! - detached signatures over already content-addressed subjects/payloads;
//! - explicit attestation expectations that fail closed on substitution; and
//! - private-field verified capabilities that cannot be deserialized into
//!   authority.
//!
//! Cryptographic providers and private keys stay outside the crate behind narrow
//! signer/verifier traits.

mod attestation;
mod identity;
mod temporal_lifecycle;
mod trust;

pub use attestation::*;
pub use identity::*;
pub use temporal_lifecycle::*;
pub use trust::*;
