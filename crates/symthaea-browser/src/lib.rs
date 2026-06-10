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
//! │  BrowserBridge (EmbodimentBridge)                   │
//! │  thought_hv → BrowserAction → CdpSession            │
//! │  Phi-gated: low Phi → NoOp / read-only              │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Safety
//!
//! Every `BrowserAction` declares a `required_phi()` threshold. Actions that
//! mutate state (click, type, navigate) require higher consciousness levels.
//! A `BrowserSafetyPolicy` provides URL allowlist/blocklist filtering.

#![deny(unsafe_code)]

pub mod actions;
pub mod cdp;
pub mod config;
pub mod embodiment;
pub mod encoder;
pub mod observation;
pub mod safety;
pub mod web_agent;

pub use actions::{BrowserAction, ElementSelector};
pub use cdp::CdpSession;
pub use config::BrowserAgentConfig;
pub use embodiment::BrowserBridge;
pub use encoder::BrowserHdcEncoder;
pub use observation::{AccessibleElement, PageObservation};
pub use safety::BrowserSafetyPolicy;
pub use web_agent::{WebAgent, WebAgentResult, WebClaim};
