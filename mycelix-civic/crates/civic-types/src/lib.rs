// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Civic Shared Types & Utilities
//!
//! Common functionality for all domain zomes in the Civic cluster:
//! - Evidence types shared across justice and media
//! - Status/phase traits for state machine validation
//! - Role-based authorization helpers
//! - Bridge types for cross-domain communication

pub mod bridge_types;
pub mod evidence;
pub mod roles;
pub mod status;

pub use bridge_types::*;
pub use evidence::*;
pub use roles::*;
pub use status::*;
