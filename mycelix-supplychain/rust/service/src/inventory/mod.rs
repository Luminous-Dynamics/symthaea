// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Inventory Management Module
//!
//! Provides comprehensive inventory management capabilities:
//! - Product catalog with SKUs and variants
//! - Warehouse and location management
//! - Stock levels and movements
//! - Inventory valuation (FIFO, LIFO, Average Cost)

pub mod products;
pub mod warehouses;
pub mod stock;
pub mod api;

pub use products::*;
pub use warehouses::*;
pub use stock::*;
pub use api::*;
