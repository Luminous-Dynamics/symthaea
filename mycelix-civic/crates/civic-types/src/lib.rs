//! Mycelix Civic Shared Types & Utilities
//!
//! Common functionality for all domain zomes in the Civic cluster:
//! - Evidence types shared across justice and media
//! - Status/phase traits for state machine validation
//! - Role-based authorization helpers
//! - Bridge types for cross-domain communication

pub mod evidence;
pub mod status;
pub mod roles;
pub mod bridge_types;
pub mod cross_domain;

pub use evidence::*;
pub use status::*;
pub use roles::*;
pub use bridge_types::*;
pub use cross_domain::*;
