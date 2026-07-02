// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Structured logging configuration
//!
//! Provides JSON-structured logging with request tracing and performance metrics

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialize structured logging for the service
///
/// Configures JSON-formatted logs with:
/// - Environment-based log level filtering (RUST_LOG env var)
/// - Request correlation via span context
/// - Performance metrics
pub fn init_logging() {
    // Try to read log level from environment, default to "info"
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,provenance_service=debug"));

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .json()  // Use JSON format for structured logging
                .with_current_span(true)  // Include current span context
                .with_span_list(true)  // Include full span hierarchy
                .with_target(true)  // Include target (module path)
                .with_level(true)  // Include log level
                .with_thread_ids(false)  // Don't include thread IDs
                .with_thread_names(false)  // Don't include thread names
        )
        .init();

    tracing::info!("Structured logging initialized");
}
