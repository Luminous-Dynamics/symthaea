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
//! - structural quorum-feasibility proofs over the exact root-bound principal
//!   directory, with deterministic witnesses binding signatures, principals,
//!   organizations, regions, algorithms, and exact verification-key material;
//! - feasibility-gated root authority capabilities whose public identities carry
//!   the exact quorum-feasibility proof and concrete attestation-envelope capacity
//!   forward without rewriting historical frozen-root identities;
//! - role-aware root authority capabilities that reuse cryptographic attestation
//!   verification and independently enforce principal/org/region quorum geometry;
//! - bounded interval-based time evidence with explicit source strength;
//! - authenticated trusted-time capabilities over root-authorized time sources;
//! - structural append-only Merkle transparency with inclusion and compact
//!   consistency proofs;
//! - authenticated, time-bound transparency checkpoints with monotonic tracking;
//! - principal-bound transparency witness observations and configured diversity
//!   quorums whose identity metadata derives from the authorized trust root;
//! - event-independent witness-quorum constituency identities that distinguish
//!   committee plurality from repeated observations or key rotation;
//! - monitor receipts that detect witnessed equivocation and verify append-only
//!   growth within one exact root-authority lineage;
//! - stable transparency-log namespaces whose delegated authority can transfer
//!   across an exact dual-root transition only with append-only handoff proofs;
//! - namespaced witnessed-view monitors that can verify append-only continuity
//!   across authorized root rotations without claiming global consistency;
//! - fresh witnessed-head federation requiring multiple root-bound witness
//!   quorum events and multiple principal constituencies to converge on one
//!   maximal observed head under authenticated time;
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
mod root_authority_gate;
mod temporal_lifecycle;
mod time;
mod transparency;
mod transparency_checkpoint;
mod transparency_constituency;
mod transparency_head_federation;
mod transparency_head_federation_gate;
mod transparency_monitor;
mod transparency_namespace;
mod transparency_namespace_monitor;
mod transparency_witness;
mod trust;
mod trust_root;
mod trust_root_feasibility;
mod trusted_time;

#[cfg(test)]
mod root_authority_tests;
#[cfg(test)]
mod trusted_time_tests;

pub use attestation::*;
pub use identity::*;
pub use principal::*;
pub use revalidation::*;
pub use root_authority::{
    AuthorizedRoleError, AuthorizedTrustRoot, AuthorizedTrustSnapshot,
    GenesisRootAuthorizationError, GenesisTrustAnchorEvidence, GenesisTrustAnchorVerifier,
    RoleSignerIdentity, RootAuthorizationKind, RootRoleQuorumFinding, RootRoleQuorumProof,
    RootTransitionAuthorizationError, TrustSnapshotAuthorizationKind, prove_root_role_quorum,
};
pub use root_authority_gate::*;
pub use temporal_lifecycle::*;
pub use time::*;
pub use transparency::*;
pub use transparency_checkpoint::*;
pub use transparency_constituency::*;
pub use transparency_head_federation::{
    MAX_HEAD_FEDERATION_QUORUMS, MAX_HEAD_FEDERATION_VIEWS,
    TransparencyHeadFederationClosure, TransparencyHeadFederationFinding,
    TransparencyHeadFederationPolicy, TransparencyHeadFederationPolicyIssue,
    TransparencyHeadFederationReceipt,
};
pub use transparency_head_federation_gate::*;
pub use transparency_monitor::*;
pub use transparency_namespace::*;
pub use transparency_namespace_monitor::*;
pub use transparency_witness::*;
pub use trust::*;
pub use trust_root::*;
pub use trust_root_feasibility::*;
pub use trusted_time::*;
