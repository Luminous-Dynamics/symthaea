//! HR (Human Resources) Module
//!
//! Provides comprehensive human resources management including:
//! - Employee records management
//! - Department/organization structure
//! - Time-off and leave management
//! - Basic payroll tracking
//! - Performance management

pub mod employees;
pub mod departments;
pub mod leave;
pub mod payroll;
pub mod api;

pub use employees::*;
pub use departments::*;
pub use leave::*;
pub use payroll::*;
pub use api::*;
