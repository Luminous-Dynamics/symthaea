// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Async FL coordinator for distributed federated learning.
//!
//! Replaces the Python `mycelix_fl.distributed.coordinator` with a native
//! Rust implementation using tokio for async operations.
//!
//! # Modules
//!
//! - [`config`] — Coordinator configuration
//! - [`node_manager`] — Node registration, authentication, and rate limiting
//! - [`round`] — Round state machine and gradient collection
//! - [`orchestrator`] — Top-level coordinator tying everything together

pub mod config;
pub mod node_manager;
pub mod orchestrator;
pub mod round;
