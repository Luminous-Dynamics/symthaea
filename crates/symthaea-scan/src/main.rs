// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign Scan — detects installed apps and system info for NixOS migration.
//!
//! Three modes:
//!   sovereign-scan              — scan + print JSON to stdout (CLI / pipe)
//!   sovereign-scan --serve      — scan + serve results via WebSocket on localhost:7799
//!   sovereign-scan --pretty     — scan + print human-readable summary

mod scanner;

use std::net::SocketAddr;

use futures_util::{SinkExt, StreamExt};
use std::io::Read;
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;

const WS_PORT: u16 = 7799;
const BANNER: &str = r#"
  ╔══════════════════════════════════════════╗
  ║   Sovereign Scan — NixOS Migration Aid   ║
  ║   install.nixforhumanity.org             ║
  ╚══════════════════════════════════════════╝
"#;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.contains(&"--help".into()) || args.contains(&"-h".into()) {
        eprintln!("{BANNER}");
        eprintln!("Usage:");
        eprintln!("  sovereign-scan            Scan system, print JSON to stdout");
        eprintln!("  sovereign-scan --pretty   Scan system, print human-readable summary");
        eprintln!("  sovereign-scan --serve    Scan system, serve results via WebSocket");
        eprintln!("                            (requires a token; printed on startup)");
        eprintln!("                            (browser at install.nixforhumanity.org connects)");
        eprintln!("  sovereign-scan --serve --token <token>   Use a fixed token (advanced)");
        eprintln!();
        eprintln!("The scanner detects installed apps from winget, brew, dpkg, pacman,");
        eprintln!("flatpak, snap, rpm, and filesystem scans. Results are matched against");
        eprintln!("a database of 50+ apps with NixOS equivalents.");
        return;
    }

    eprintln!("{BANNER}");
    eprintln!("Scanning system...");

    let result = scanner::scan();

    if args.contains(&"--serve".into()) {
        let token = args
            .iter()
            .position(|a| a == "--token")
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(generate_auth_token);
        serve_websocket(result, token).await;
    } else if args.contains(&"--pretty".into()) {
        print_pretty(&result);
    } else {
        // Default: JSON to stdout
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    }
}

fn print_pretty(result: &scanner::ScanResult) {
    println!(
        "System: {} {} ({})",
        result.os.name, result.os.version, result.os.arch
    );
    println!("Kernel: {}", result.os.kernel);
    println!(
        "CPU:    {} ({} cores)",
        result.hardware.cpu_model, result.hardware.cpu_cores
    );
    println!("Memory: {:.1} GB", result.hardware.memory_gb);
    println!("GPU:    {}", result.hardware.gpu);
    println!(
        "Disk:   {:.0} GB total, {:.0} GB free",
        result.hardware.disk_total_gb, result.hardware.disk_free_gb
    );
    println!();
    println!("Apps detected: {}", result.migration.total_detected);
    println!("  Matched to NixOS:  {}", result.migration.matched);
    println!("  Native available:  {}", result.migration.native_available);
    println!("  Alternatives:      {}", result.migration.alternatives);
    println!("  No equivalent:     {}", result.migration.no_equivalent);
    println!();

    if result
        .installed_apps
        .iter()
        .any(|a| a.canonical_name.is_some())
    {
        println!("Matched apps:");
        for app in &result.installed_apps {
            if let Some(ref canonical) = app.canonical_name {
                let nix = app.nix_display.as_deref().unwrap_or("?");
                let quality = app.quality.as_deref().unwrap_or("?");
                let conf = app.confidence.unwrap_or(0);
                println!(
                    "  {:<25} -> {:<20} [{}, {}%]",
                    canonical, nix, quality, conf
                );
            }
        }
    }

    let unmatched: Vec<_> = result
        .installed_apps
        .iter()
        .filter(|a| a.canonical_name.is_none())
        .collect();
    if !unmatched.is_empty() && unmatched.len() <= 50 {
        println!();
        println!("Unmatched ({}):", unmatched.len());
        for app in &unmatched {
            println!("  {} ({})", app.detected_name, app.source);
        }
    } else if !unmatched.is_empty() {
        println!();
        println!(
            "Unmatched: {} apps (system packages, libraries, etc.)",
            unmatched.len()
        );
    }
}

fn generate_auth_token() -> String {
    let mut bytes = [0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(&mut bytes).is_ok() {
            return bytes.iter().map(|b| format!("{:02x}", b)).collect();
        }
    }

    // Extremely unlikely on Linux; fallback to a best-effort token.
    format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

async fn serve_websocket(result: scanner::ScanResult, token: String) {
    let addr = SocketAddr::from(([127, 0, 0, 1], WS_PORT));

    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to bind to port {WS_PORT}: {e}");
            eprintln!("Is another instance running?");
            std::process::exit(1);
        }
    };

    let json = serde_json::to_string(&result).unwrap();

    eprintln!();
    eprintln!(
        "Scan complete. {} apps detected, {} matched to NixOS.",
        result.migration.total_detected, result.migration.matched
    );
    eprintln!();
    eprintln!("WebSocket server listening on ws://127.0.0.1:{WS_PORT}");
    eprintln!("  Auth token: {token}");
    eprintln!("Open install.nixforhumanity.org in your browser — it will auto-connect.");
    eprintln!();
    eprintln!("Press Ctrl+C to stop.");

    let token = std::sync::Arc::new(token);

    // Accept connections until Ctrl+C
    loop {
        match listener.accept().await {
            Ok((stream, peer)) => {
                eprintln!("Browser connected from {peer}");
                let json = json.clone();
                let token = token.clone();
                tokio::spawn(async move {
                    match accept_async(stream).await {
                        Ok(mut ws) => {
                            // Auth gate: require an explicit token before sending any scan data.
                            let authed = match tokio::time::timeout(
                                std::time::Duration::from_secs(10),
                                ws.next(),
                            )
                            .await
                            {
                                Ok(Some(Ok(tokio_tungstenite::tungstenite::Message::Text(
                                    text,
                                )))) => serde_json::from_str::<serde_json::Value>(&text)
                                    .ok()
                                    .and_then(|v| {
                                        if v.get("action")?.as_str()? != "auth" {
                                            return Some(false);
                                        }
                                        let t = v.get("token")?.as_str()?;
                                        Some(t == token.as_str())
                                    })
                                    .unwrap_or(false),
                                _ => false,
                            };

                            if !authed {
                                let _ = ws
                                    .send(tokio_tungstenite::tungstenite::Message::Text(
                                        r#"{"type":"error","message":"Unauthorized: send {\"action\":\"auth\",\"token\":...} first"}"#.into(),
                                    ))
                                    .await;
                                let _ = ws.close(None).await;
                                eprintln!("Unauthorized WebSocket client rejected.");
                                return;
                            }

                            // Send scan results after auth
                            let msg = serde_json::json!({
                                "type": "scan_result",
                                "data": serde_json::from_str::<serde_json::Value>(&json).unwrap()
                            });
                            if let Err(e) = ws
                                .send(tokio_tungstenite::tungstenite::Message::Text(
                                    msg.to_string(),
                                ))
                                .await
                            {
                                eprintln!("Send error: {e}");
                                return;
                            }
                            eprintln!("Scan results sent to browser (authed).");

                            // Keep connection alive for potential follow-up queries
                            while let Some(Ok(msg)) = ws.next().await {
                                if msg.is_close() {
                                    break;
                                }
                                if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                                    if let Ok(cmd) =
                                        serde_json::from_str::<serde_json::Value>(&text)
                                    {
                                        match cmd.get("action").and_then(|v| v.as_str()) {
                                            Some("rescan") => {
                                                eprintln!("Rescanning...");
                                                let fresh = scanner::scan();
                                                let resp = serde_json::json!({
                                                    "type": "scan_result",
                                                    "data": fresh
                                                });
                                                let _ = ws.send(tokio_tungstenite::tungstenite::Message::Text(resp.to_string())).await;
                                            }
                                            Some("ping") => {
                                                let _ = ws.send(tokio_tungstenite::tungstenite::Message::Text(
                                                    r#"{"type":"pong"}"#.into()
                                                )).await;
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            eprintln!("Browser disconnected.");
                        }
                        Err(e) => eprintln!("WebSocket handshake error: {e}"),
                    }
                });
            }
            Err(e) => eprintln!("Accept error: {e}"),
        }
    }
}