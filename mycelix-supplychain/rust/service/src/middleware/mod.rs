// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HTTP middleware modules
//!
//! Provides cross-cutting concerns for HTTP request handling

pub mod rate_limit;
pub mod security;
pub mod tracing;

pub use self::rate_limit::{create_rate_limiter, rate_limit_middleware, GlobalRateLimiter, RateLimitConfig};
pub use self::security::security_headers;
pub use self::tracing::trace_request;
