//! Mycelix-DeSci REST API Library
//!
//! This library provides the core components of the Mycelix-DeSci REST API,
//! exposed for integration testing and reusability.

pub mod error;
pub mod handlers;
pub mod metrics;
pub mod middleware;
pub mod models;
pub mod routes;
pub mod state;

// Re-export commonly used types
pub use error::{ApiError, Result};
pub use state::AppState;
