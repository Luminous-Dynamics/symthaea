//! Storage Backend Module
//!
//! Provides pluggable storage backends for federated learning data:
//! - Gradient records with integrity verification
//! - Credit system transactions
//! - Byzantine event logging
//! - Reputation tracking
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use fl_aggregator::storage::{StorageBackend, LocalFileBackend, BackendConfig};
//!
//! // Create backend
//! let backend = LocalFileBackend::new("/tmp/fl_data");
//! backend.connect().await?;
//!
//! // Store gradient
//! let record = GradientRecord::new("node_1", 1, &gradient, "hash123");
//! let id = backend.store_gradient(&record).await?;
//!
//! // Retrieve
//! let retrieved = backend.get_gradient(&id).await?;
//! ```

pub mod backend;
pub mod types;
pub mod composite;

#[cfg(feature = "storage-local")]
pub mod localfile;

#[cfg(feature = "storage-postgres")]
pub mod postgres;

pub use backend::{StorageBackend, StorageError, StorageResult, BackendConfig};
pub use types::*;
pub use composite::{CompositeBackend, RoutingStrategy};

#[cfg(feature = "storage-local")]
pub use localfile::LocalFileBackend;

#[cfg(feature = "storage-postgres")]
pub use postgres::PostgresBackend;
