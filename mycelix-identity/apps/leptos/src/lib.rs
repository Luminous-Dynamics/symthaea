// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mycelix Identity — Sovereign digital identity management.
//!
//! DID creation, MFA enrollment, credential issuance, social recovery,
//! trust scoring, and reputation aggregation. All data flows through
//! Holochain with graceful mock fallback.

pub mod app;
pub mod identity_context;
pub mod mock_data;
pub mod components;
pub mod pages;
