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
//! gallery-server [--root PATH] [--memory PATH] [--bind ADDR] [--allow-remote]
//!   --root          gallery store root       (default: .claude/gallery)
//!   --memory        aesthetic memory JSON    (default: .claude/aesthetic_memory.json)
//!   --bind          listen address           (default: 127.0.0.1:8402, dev/test range)
//!   --allow-remote  permit binding to a non-loopback address
//! ```
//!
//! No route requires authentication (including `POST /api/rate`, which
//! writes through to the persisted `AestheticMemory` EMA the cognitive
//! loop reads from). That's fine for the safe default (loopback-only) but
//! not for `--bind 0.0.0.0:...` or any other externally-reachable address
//! -- `--allow-remote` exists so that choice has to be explicit rather
//! than an operator accidentally exposing an unauthenticated write
//! endpoint by passing a wider `--bind` without realizing it.

use std::net::SocketAddr;
use std::path::PathBuf;

use symthaea_gallery::server::{GalleryServerConfig, router};

const DEFAULT_BIND: &str = "127.0.0.1:8402";

fn parse_args() -> Result<(GalleryServerConfig, String, bool), String> {
    let mut config = GalleryServerConfig::default();
    let mut bind = DEFAULT_BIND.to_string();
    let mut allow_remote = false;

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
            "--allow-remote" => {
                allow_remote = true;
            }
            "--help" | "-h" => {
                println!(
                    "gallery-server [--root PATH] [--memory PATH] [--bind ADDR] [--allow-remote]\n\
                     \x20 --root          gallery store root    (default: .claude/gallery)\n\
                     \x20 --memory        aesthetic memory JSON (default: .claude/aesthetic_memory.json)\n\
                     \x20 --bind          listen address        (default: {DEFAULT_BIND})\n\
                     \x20 --allow-remote  permit binding to a non-loopback address\n\
                     \x20                 (no route is authenticated; see module docs)"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok((config, bind, allow_remote))
}

/// `false` for anything that isn't unambiguously loopback-only -- an
/// unparseable address (including a hostname like `localhost`, which
/// would need DNS resolution to know for sure) is treated as non-loopback
/// (fail closed) rather than silently allowed through.
fn is_loopback_bind(bind: &str) -> bool {
    bind.parse::<SocketAddr>()
        .map(|addr| addr.ip().is_loopback())
        .unwrap_or(false)
}

#[tokio::main]
async fn main() {
    let (config, bind, allow_remote) = match parse_args() {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    if !allow_remote && !is_loopback_bind(&bind) {
        eprintln!(
            "error: refusing to bind {bind} -- no route on this server is authenticated \
             (including POST /api/rate, which writes through to persisted state), so \
             binding to a non-loopback address exposes it to any network peer. Pass \
             --allow-remote if you've verified that's acceptable for your deployment."
        );
        std::process::exit(2);
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the no-auth-on-remote-bind finding: the default
    /// bind and other loopback forms must be recognized as safe without
    /// requiring --allow-remote, while anything actually reachable from
    /// the network (or unparseable/ambiguous) must not be.
    #[test]
    fn loopback_addresses_are_recognized() {
        for addr in [DEFAULT_BIND, "127.0.0.1:8402", "127.0.0.1:0", "[::1]:8402"] {
            assert!(is_loopback_bind(addr), "{addr} should be loopback");
        }
    }

    #[test]
    fn non_loopback_addresses_are_rejected() {
        for addr in [
            "0.0.0.0:8402",
            "[::]:8402",
            "192.168.1.5:8402",
            "10.0.0.1:8402",
        ] {
            assert!(!is_loopback_bind(addr), "{addr} should not be loopback");
        }
    }

    #[test]
    fn unparseable_bind_fails_closed() {
        // Hostnames need DNS resolution to know for sure -- treated as
        // not-loopback rather than guessed at.
        for addr in ["localhost:8402", "not-an-address", ""] {
            assert!(!is_loopback_bind(addr), "{addr} should fail closed");
        }
    }
}
