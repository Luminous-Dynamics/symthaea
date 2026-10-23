// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-browser
//!
//! Consciousness-driven browser agent for Symthaea.
//!
//! Bridges the cognitive loop to Chrome/Chromium through CDP while keeping
//! browser authority separate from cognitive confidence. Normal browser
//! mutation is mediated by [`BrowserExecutor`], which combines ordinary browser
//! policy with a bounded runtime: exact approvals for consequential actions,
//! action budgets, a failure circuit breaker, and privacy-minimized receipts.
//!
//! Raw page outputs are returned separately from durable receipt evidence.
//! `BrowserApproval` is currently a process-local exact approval object, not a
//! cryptographic credential; future Xenia/`symthaea-authority` integration can
//! authenticate the same request/decision semantics.

#![deny(unsafe_code)]

pub mod actions;
pub mod cdp;
pub mod config;
pub mod embodiment;
pub mod encoder;
pub mod executor;
pub mod hardening;
pub mod observation;
pub mod safety;
pub mod web_agent;

pub use actions::{BrowserAction, BrowserCapability, ElementSelector};
pub use cdp::CdpSession;
pub use config::BrowserAgentConfig;
pub use embodiment::BrowserBridge;
pub use encoder::BrowserHdcEncoder;
pub use executor::{
    ActionExecution, ActionOutcome, ActionOutput, ActionReceipt, BrowserExecutor,
};
pub use hardening::{
    BrowserApproval, BrowserApprovalRequest, BrowserConsequence, BrowserRuntimeDenial,
    BrowserRuntimeLimits, BrowserRuntimeSnapshot, action_digest, action_kind, consequence_of,
};
pub use observation::{
    AccessibleElement, MAX_OBSERVATION_TEXT_CHARS, PageObservation, UNTRUSTED_WEB_CONTENT_LABEL,
};
pub use safety::{BrowserSafetyPolicy, PolicyDecision, PolicyDenial};
pub use web_agent::{EvidenceStatus, WebAgent, WebAgentResult, WebClaim, WebResearchResult};
