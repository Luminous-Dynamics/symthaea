// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Project-neutral assurance primitives for autonomous systems.
//!
//! The crate is intentionally independent of Symthaea cognition. Models and
//! planners propose actions; trusted host code decides which authority values
//! they receive.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod action;
pub mod capability;

pub use action::{
    Action, ActionDescriptor, ActionError, ActionId, ActionRisk, Authorized, EvidenceReceipt,
    Executed, Observation, Observed, ObservedOutcome, Proposed, ResolutionDecision, Resolved,
    RiskAssessed,
};
pub use capability::{
    AuthorityRoot, BoundOneShotCapability, Capability, CapabilityKind, Deploy, Execute, GrantError,
    GrantId, GrantMetadata, Network, Observe, OneShotCapability, PrincipalId, Read, Scope,
    ScopeError, UpdateModel, Write,
};
