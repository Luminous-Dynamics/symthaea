// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix-DeSci REST API Library
//!
//! This library provides the core components of the Mycelix-DeSci REST API,
//! exposed for integration testing and reusability.

pub mod error;
pub mod handlers;
pub mod metrics;
pub mod middleware;
pub mod models;
pub mod routes;
pub mod state;

// Re-export commonly used types
pub use error::{ApiError, Result};
pub use state::AppState;
