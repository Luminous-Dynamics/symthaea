// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Mycelix Pulse SMTP Gateway
//!
//! The sovereign SMTP exit node for Mycelix Pulse. Handles the bidirectional
//! bridge between legacy RFC 5322 mail and the Pulse DHT:
//!
//!   - **Inbound**: external MTA → this daemon on :25 → SPF/DKIM/DMARC
//!     verification → RFC 5322 parse → alias resolution → re-encrypt for
//!     recipient's current epoch → write to DHT via zome bridge.
//!
//!   - **Outbound**: Pulse user composes → zome writes OutboxEntry →
//!     gateway polls → DKIM-signs → MX lookup → mail-send TLS SMTP →
//!     external MTA. Bounces come back as DSN to our MX.
//!
//! Phase 5 of `PULSE_READINESS_PLAN.md`. This crate is a binary + library:
//!
//!   - The library exposes the full pipeline as testable modules.
//!   - The binary wires them to a TOML config and a TCP listener.
//!   - Phase 5A runs the binary inside a NixOS VM for end-to-end testing
//!     without a real VPS. Phase 5B deploys identically to a Hetzner host.
//!
//! See `mycelix-pulse/PULSE_READINESS_PLAN.md` §3.1 and §5 for the threat
//! model and architectural constraints.

pub mod auth;
pub mod bounce;
pub mod config;
pub mod error;
pub mod outbound;
pub mod parser;
pub mod pipeline;
pub mod rate_limit;
pub mod receiver;
pub mod signing_content;
pub mod verp;
pub mod zome;

pub use config::GatewayConfig;
pub use error::{GatewayError, GatewayResult};
