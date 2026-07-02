// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CRM (Customer Relationship Management) Module
//!
//! Provides comprehensive customer relationship management including:
//! - Contact management
//! - Lead tracking and scoring
//! - Sales pipeline (opportunities)
//! - Activity logging
//! - Email/call tracking integration

pub mod contacts;
pub mod leads;
pub mod opportunities;
pub mod activities;
pub mod api;

pub use contacts::*;
pub use leads::*;
pub use opportunities::*;
pub use activities::*;
pub use api::*;
