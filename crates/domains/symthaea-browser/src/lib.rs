// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-browser
//!
//! Consciousness-driven browser agent for Symthaea.
//!
//! Bridges the cognitive loop to a headless Chrome/Chromium instance via the
//! Chrome DevTools Protocol (CDP). The browser becomes a sensory organ: the
//! accessibility tree is encoded as a 16,384D `ContinuousHV`, and actions
//! (navigate, click, type) are Phi-gated to ensure conscious intent.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │  Chrome / Chromium (headless)                       │
//! │  CDP WebSocket ← chromiumoxide                      │
//! └────────────────────┬────────────────────────────────┘
//!                      │ accessibility tree + DOM events
//! ┌────────────────────▼────────────────────────────────┐
//! │  CdpSession                                         │
//! │  navigate / click / type / screenshot               │
//! └────────────────────┬────────────────────────────────┘
//!                      │ PageObservation
//! ┌────────────────────▼────────────────────────────────┐
//! │  BrowserHdcEncoder                                  │
//! │  role codebook + text hashing → ContinuousHV(16384) │
//! └────────────────────┬────────────────────────────────┘
//!                      │
//! ┌────────────────────▼────────────────────────────────┐
//! │  BrowserBridge                                     │
//! │  bounded observation → perception + change telemetry│
//! └────────────────────┬────────────────────────────────┘
//!                      │ BrowserAction proposal
//! ┌────────────────────▼────────────────────────────────┐
//! │  BrowserExecutor                                   │
//! │  capability + Phi + URL policy → ActionReceipt     │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Safety
//!
//! Every `BrowserAction` requires both an explicit capability and a finite
//! `required_phi()` threshold. Phi can increase caution but cannot create
//! authority. `BrowserSafetyPolicy` also applies canonical origin checks and
//! denies local-network targets by default.

#![deny(unsafe_code)]

pub mod actions;
pub mod cdp;
pub mod config;
pub mod embodiment;
pub mod encoder;
pub mod executor;
pub mod observation;
pub mod safety;
pub mod web_agent;

pub use actions::{BrowserAction, BrowserCapability, ElementSelector};
pub use cdp::CdpSession;
pub use config::BrowserAgentConfig;
pub use embodiment::BrowserBridge;
pub use encoder::BrowserHdcEncoder;
pub use executor::{ActionOutcome, ActionOutput, ActionReceipt, BrowserExecutor};
pub use observation::{
    AccessibleElement, MAX_OBSERVATION_TEXT_CHARS, PageObservation, UNTRUSTED_WEB_CONTENT_LABEL,
};
pub use safety::{BrowserSafetyPolicy, PolicyDecision, PolicyDenial};
pub use web_agent::{EvidenceStatus, WebAgent, WebAgentResult, WebClaim, WebResearchResult};
