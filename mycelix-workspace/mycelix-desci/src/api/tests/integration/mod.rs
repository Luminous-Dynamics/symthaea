// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for Mycelix-DeSci API
//!
//! These tests verify the complete functionality of the REST API,
//! including claim lifecycle, query operations, trust management,
//! and error handling.

mod helpers;

mod test_api_claims;
mod test_api_query;
mod test_api_trust;
mod test_api_system;
mod test_claim_lifecycle;
mod test_concurrent_operations;
mod test_error_handling;
