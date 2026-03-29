// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Authentication module for Mycelix ERP
//!
//! Provides JWT-based authentication with:
//! - User registration and login
//! - Password hashing with Argon2
//! - JWT access and refresh tokens
//! - API key management
//! - Multi-tenant support

mod models;
mod jwt;
mod password;
mod api;
mod middleware;

pub use models::*;
pub use jwt::*;
pub use password::*;
pub use api::*;
pub use middleware::*;
