// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-Time Collaboration Module
//!
//! Provides WebSocket-based real-time updates for:
//! - Document collaboration (invoices, bills, POs)
//! - Live dashboard updates
//! - Notification delivery
//! - Presence tracking

pub mod api;
pub mod hub;
pub mod events;
pub mod session;

pub use api::*;
pub use hub::*;
pub use events::*;
pub use session::*;
