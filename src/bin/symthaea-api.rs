// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea API server entrypoint.

use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let addr = env::var("SYMTHAEA_API_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    symthaea::api::serve(&addr).await
}
