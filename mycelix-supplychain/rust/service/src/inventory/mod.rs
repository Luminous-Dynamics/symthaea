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
