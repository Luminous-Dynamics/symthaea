// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Logging Infrastructure
//!
//! Structured logging using tracing crate.

use crate::config::LoggingConfig;
use tracing::Level;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialize logging based on configuration
pub fn init(config: &LoggingConfig) {
    let level = parse_log_level(&config.level);

    let filter = EnvFilter::from_default_env()
        .add_directive(level.into())
        .add_directive(
            "mycelix_desci_core=debug"
                .parse()
                .expect("Static directive 'mycelix_desci_core=debug' must be valid")
        );

    let fmt_layer = match config.format.as_str() {
        "json" => fmt::layer().json().boxed(),
        _ => fmt::layer().boxed(),
    };

    let subscriber = tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer);

    if let Err(e) = tracing::subscriber::set_global_default(subscriber) {
        eprintln!("Failed to initialize logging: {}", e);
    }
}

/// Parse log level string to tracing::Level
fn parse_log_level(level: &str) -> Level {
    match level.to_lowercase().as_str() {
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => {
            eprintln!("Unknown log level '{}', defaulting to INFO", level);
            Level::INFO
        }
    }
}

/// Initialize with default settings (info level, text format)
pub fn init_default() {
    let config = LoggingConfig::default();
    init(&config);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_log_level() {
        assert_eq!(parse_log_level("debug"), Level::DEBUG);
        assert_eq!(parse_log_level("INFO"), Level::INFO);
        assert_eq!(parse_log_level("WARN"), Level::WARN);
        assert_eq!(parse_log_level("error"), Level::ERROR);
        assert_eq!(parse_log_level("unknown"), Level::INFO);
    }
}
