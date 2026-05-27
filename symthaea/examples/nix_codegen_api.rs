// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nix Codegen HTTP API — standalone server exposing the pipeline.
//!
//! Endpoints:
//!   POST /generate-nix
//!     Body: {"prompt": "...", "max_iterations": 3}
//!     Returns: {"code": "...", "intent": "...", "parses": bool, "iterations": N}
//!
//!   GET /health
//!     Returns: {"status": "ok", "version": "1.0"}
//!
//! Built with std::net only — no external deps. Single-threaded for
//! simplicity (the pipeline is ~10-50ms per request, this server is
//! sufficient for the install.nixforhumanity.org use case).
//!
//! Run:
//!   cargo run --example nix_codegen_api --features code_generation -- 8128
//!
//! Test:
//!   curl -X POST http://localhost:8128/generate-nix \
//!     -H "Content-Type: application/json" \
//!     -d '{"prompt": "set up postgresql with pgvector"}'

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use symthaea::language::nix_codegen::{
    NixIntent, SelfImproveSource, generate_nix_with_rag_fast, shared_learned_cache,
};

/// Minimal JSON encoder for our response shape.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Extract a JSON string field value (very minimal — no nesting).
fn extract_string_field(body: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\"");
    let start = body.find(&key)? + key.len();
    let after = body[start..].trim_start_matches([' ', '\t', ':']);
    if !after.starts_with('"') {
        return None;
    }
    let inner = &after[1..];
    let mut result = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(result),
            '\\' => match chars.next()? {
                'n' => result.push('\n'),
                't' => result.push('\t'),
                'r' => result.push('\r'),
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                other => result.push(other),
            },
            other => result.push(other),
        }
    }
    None
}

fn extract_int_field(body: &str, field: &str) -> Option<u32> {
    let key = format!("\"{field}\"");
    let start = body.find(&key)? + key.len();
    let after = body[start..].trim_start_matches([' ', '\t', ':']);
    let end = after
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after.len());
    after[..end].parse().ok()
}

fn intent_to_str(intent: NixIntent) -> &'static str {
    match intent {
        NixIntent::DevShell => "DevShell",
        NixIntent::Service => "Service",
        NixIntent::Hardware => "Hardware",
        NixIntent::Desktop => "Desktop",
        NixIntent::User => "User",
        NixIntent::HomeManager => "HomeManager",
        NixIntent::Networking => "Networking",
        NixIntent::Secrets => "Secrets",
        NixIntent::FlakeTemplate => "FlakeTemplate",
        NixIntent::Generic => "Generic",
    }
}

fn handle_generate_nix(body: &str) -> String {
    let prompt = match extract_string_field(body, "prompt") {
        Some(p) if !p.is_empty() => p,
        _ => return json_response(400, r#"{"error":"missing or empty 'prompt' field"}"#),
    };
    let max_iter = extract_int_field(body, "max_iterations").unwrap_or(3) as usize;

    // Tier 3 + RAG fast-path: cache hit → idiom (parse-only) → RAG draft.
    // Records verified results into ~/.cache/symthaea/learned-idioms.json.
    // Fast mode skips `try_nix_eval` (~100s/call) for sub-second cold
    // latency. Callers needing deeper verification should call the
    // /generate-nix endpoint with deeper-verify support (future work) or
    // run the dry-build verifier server-side.
    let (result, src) = generate_nix_with_rag_fast(&prompt, max_iter);
    let base = result.base;
    let unknowns_count = result.unknown_options.len();
    let unknowns = result
        .unknown_options
        .iter()
        .map(|u| format!("\"{}\"", json_escape(&u.path)))
        .collect::<Vec<_>>()
        .join(",");
    let cache_status = match src {
        SelfImproveSource::LearnedCache => "\"cache_hit\"",
        SelfImproveSource::FreshlyGenerated { recorded: true } => "\"fresh_recorded\"",
        SelfImproveSource::FreshlyGenerated { recorded: false } => "\"fresh_not_recorded\"",
    };

    let body = format!(
        r#"{{"intent":"{intent}","code":"{code}","parses":{parses},"iterations":{iter},"source":"{source}","last_error":{last_err},"cache_status":{cache_status},"unknown_options_count":{uc},"unknown_options":[{u}]}}"#,
        intent = intent_to_str(base.intent),
        code = json_escape(&base.code),
        parses = base.parses,
        iter = base.iterations,
        source = format!("{:?}", base.source),
        last_err = match &base.last_error {
            Some(e) => format!("\"{}\"", json_escape(e)),
            None => "null".to_string(),
        },
        uc = unknowns_count,
        u = unknowns,
    );
    json_response(200, &body)
}

fn handle_health() -> String {
    let cache_size = shared_learned_cache().len();
    let body = format!(
        r#"{{"status":"ok","version":"1.2","service":"symthaea-nix-codegen","learned_idioms_cached":{cache_size},"pipeline":"cache+idiom+parse+index+rag (fast)","mode":"interactive"}}"#
    );
    json_response(200, &body)
}

fn json_response(status: u16, body: &str) -> String {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Internal Server Error",
    };
    format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{body}",
        len = body.len()
    )
}

fn handle_request(stream: &mut TcpStream) {
    let mut buf = vec![0u8; 16 * 1024];
    let n = match stream.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return,
    };
    let req = String::from_utf8_lossy(&buf[..n]).to_string();

    let first_line = req.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");

    let response = if method == "OPTIONS" {
        // CORS preflight
        format!(
            "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
    } else if method == "GET" && path == "/health" {
        handle_health()
    } else if method == "POST" && path == "/generate-nix" {
        // Find body after \r\n\r\n
        let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
        handle_generate_nix(body)
    } else if method == "GET" && path == "/" {
        json_response(
            200,
            r#"{"endpoints":["GET /health","POST /generate-nix"],"docs":"see examples/nix_codegen_api.rs"}"#,
        )
    } else {
        json_response(404, r#"{"error":"endpoint not found"}"#)
    };

    let _ = stream.write_all(response.as_bytes());
}

fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8128);

    let addr = format!("127.0.0.1:{port}");
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("bind failed on {addr}: {e}");
            std::process::exit(1);
        }
    };

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Symthaea Nix Codegen API");
    println!("│ Listening on http://{addr}");
    println!("└─────────────────────────────────────────────────────────");
    println!("\nEndpoints:");
    println!("  GET  /health");
    println!("  POST /generate-nix  body: {{\"prompt\":\"...\"}}");
    println!("\nExample:");
    println!("  curl -X POST http://{addr}/generate-nix -H 'Content-Type: application/json' \\");
    println!("    -d '{{\"prompt\":\"set up postgresql with pgvector\"}}'");
    println!("\n(Press Ctrl-C to stop)");

    for stream in listener.incoming() {
        match stream {
            Ok(mut s) => handle_request(&mut s),
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
}
