// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Standalone gallery server binary (feature = `server`).
//!
//! Serves the flat-file symthaea-gallery store as a self-contained HTML
//! gallery with per-artwork rating controls that write through to the
//! persisted `AestheticMemory` (VISUAL_ART_IMPROVEMENT_PLAN_2026-07-10
//! Phase 4.1).
//!
//! Usage:
//! ```text
//! gallery-server [--root PATH] [--memory PATH] [--bind ADDR]
//!   --root    gallery store root       (default: .claude/gallery)
//!   --memory  aesthetic memory JSON    (default: .claude/aesthetic_memory.json)
//!   --bind    listen address           (default: 127.0.0.1:8402, dev/test range)
//! ```

use std::path::PathBuf;

use symthaea_gallery::server::{GalleryServerConfig, router};

const DEFAULT_BIND: &str = "127.0.0.1:8402";

fn parse_args() -> Result<(GalleryServerConfig, String), String> {
    let mut config = GalleryServerConfig::default();
    let mut bind = DEFAULT_BIND.to_string();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => {
                config.gallery_root = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--root requires a path".to_string())?,
                );
            }
            "--memory" => {
                config.aesthetic_memory_path = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--memory requires a path".to_string())?,
                );
            }
            "--bind" => {
                bind = args
                    .next()
                    .ok_or_else(|| "--bind requires an address".to_string())?;
            }
            "--help" | "-h" => {
                println!(
                    "gallery-server [--root PATH] [--memory PATH] [--bind ADDR]\n\
                     \x20 --root    gallery store root    (default: .claude/gallery)\n\
                     \x20 --memory  aesthetic memory JSON (default: .claude/aesthetic_memory.json)\n\
                     \x20 --bind    listen address        (default: {DEFAULT_BIND})"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok((config, bind))
}

#[tokio::main]
async fn main() {
    let (config, bind) = match parse_args() {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    println!(
        "Symthaea gallery server\n  store:  {}\n  memory: {}\n  listen: http://{}",
        config.gallery_root.display(),
        config.aesthetic_memory_path.display(),
        bind
    );

    let listener = match tokio::net::TcpListener::bind(&bind).await {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("error: failed to bind {bind}: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = axum::serve(listener, router(config)).await {
        eprintln!("error: server terminated: {e}");
        std::process::exit(1);
    }
}
