// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! API Middleware
//!
//! Custom middleware for the Mycelix-DeSci API

pub mod metrics_middleware;
pub mod rate_limit;

pub use metrics_middleware::MetricsMiddleware;
pub use rate_limit::{
    RateLimitConfig, create_query_rate_limit_config, create_rate_limit_config,
    create_strict_rate_limit_config,
};
