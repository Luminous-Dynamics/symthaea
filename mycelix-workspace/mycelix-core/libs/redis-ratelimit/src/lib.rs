// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Redis Rate Limiter
//!
//! Distributed rate limiting using Redis for consistent rate limiting across
//! multiple service instances. Implements multiple algorithms:
//!
//! - **Sliding Window**: Smooth rate limiting with sub-window precision
//! - **Token Bucket**: Allows bursts while maintaining average rate
//! - **Fixed Window**: Simple counter-based limiting per time window
//! - **Leaky Bucket**: Constant output rate regardless of input
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_redis_ratelimit::{RateLimiter, SlidingWindowConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     let limiter = RateLimiter::new("redis://localhost:6379").await.unwrap();
//!
//!     let config = SlidingWindowConfig::new(100, Duration::from_secs(60));
//!
//!     match limiter.check("user:123", &config).await {
//!         Ok(result) if result.allowed => println!("Request allowed"),
//!         Ok(result) => println!("Rate limited, retry after {:?}", result.retry_after),
//!         Err(e) => eprintln!("Error: {}", e),
//!     }
//! }
//! ```

pub mod algorithms;
pub mod config;
pub mod limiter;
pub mod lua_scripts;

pub use config::{
    RateLimitConfig, SlidingWindowConfig, TokenBucketConfig,
    FixedWindowConfig, LeakyBucketConfig,
};
pub use limiter::{RateLimiter, RateLimitResult, RateLimiterBuilder};
pub use algorithms::Algorithm;

use thiserror::Error;

/// Errors that can occur in rate limiting operations
#[derive(Error, Debug)]
pub enum RateLimitError {
    #[error("Redis connection error: {0}")]
    ConnectionError(String),

    #[error("Redis command error: {0}")]
    RedisError(#[from] redis::RedisError),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("Lua script error: {0}")]
    ScriptError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Result type for rate limiting operations
pub type Result<T> = std::result::Result<T, RateLimitError>;

/// Version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Key prefix for all rate limit keys
pub const KEY_PREFIX: &str = "mycelix:ratelimit:";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }

    #[test]
    fn test_key_prefix() {
        assert!(KEY_PREFIX.starts_with("mycelix:"));
    }
}
