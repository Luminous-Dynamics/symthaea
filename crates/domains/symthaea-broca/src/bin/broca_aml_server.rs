// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#[tokio::main]
async fn main() {
    if let Err(error) = symthaea_broca::memory_challenge_http::serve_from_env().await {
        eprintln!("symthaea-broca-aml-server: {error}");
        std::process::exit(1);
    }
}
