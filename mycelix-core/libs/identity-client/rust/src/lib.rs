//! Mycelix Identity Client - Rust
//!
//! Shared identity library for Mycelix Rust applications.
//! Provides DID resolution, credential verification, and assurance level checks.
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_identity_client::{IdentityClient, IdentityClientConfig, AssuranceLevel};
//!
//! let config = IdentityClientConfig::default();
//! let client = IdentityClient::new(config);
//!
//! // Resolve a DID
//! let result = client.resolve_did("did:mycelix:abc123").await?;
//!
//! // Get assurance level
//! let level = client.get_assurance_level("did:mycelix:abc123").await?;
//! ```

mod types;
mod client;
mod error;

pub use types::*;
pub use client::*;
pub use error::*;
