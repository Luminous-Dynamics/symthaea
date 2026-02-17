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
