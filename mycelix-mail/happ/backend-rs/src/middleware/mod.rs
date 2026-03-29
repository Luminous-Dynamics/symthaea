// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Middleware modules

pub mod auth;
pub mod rate_limit;

pub use auth::{AuthenticatedUser, Claims, JwtSecret, create_token, verify_token};
pub use rate_limit::{RateLimitState, EndpointRateLimits, rate_limit_middleware};
