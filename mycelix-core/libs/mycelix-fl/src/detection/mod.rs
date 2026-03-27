// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-layer Byzantine detection stack.
//!
//! Combines multiple detection methods in a layered architecture:
//!
//! - [`shapley`] — Shapley-value based leave-one-out detection
//! - [`self_healing`] — Error correction for repairable gradients
//! - [`stack`] — Orchestrator that runs layers in sequence
//!
//! The stack can optionally include PoGQ (from [`crate::pogq`]) as the
//! first detection layer, followed by Shapley detection, and finally
//! self-healing for gradients that are close enough to repair.

pub mod self_healing;
pub mod shapley;
pub mod stack;
