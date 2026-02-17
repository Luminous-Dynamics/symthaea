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
