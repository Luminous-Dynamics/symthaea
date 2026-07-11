// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Minimal blocking Ollama `/api/embed` client.
//!
//! Raw `std::net::TcpStream` HTTP/1.1 — deliberately no reqwest/tokio, matching
//! this crate's no-async design (the caller is `Qwen3Embedder::embed()`, which
//! already runs on the `EmbeddingChannel` background thread, so blocking I/O is
//! correct here). Wire format cribbed from the existing raw-TcpStream Ollama
//! clients (`symthaea-probe-stream/src/backends.rs`, spinozist_geometry's
//! `jl_project` feeder) rather than inventing a fourth dialect.
//!
//! Default model is `embeddinggemma:300m` (768-D), the approved local
//! embedding model. Verified live 2026-07-09: sane semantic geometry
//! (cos("move arm left","move arm right")=0.92, both ~0.46 vs "grab the cup").

use anyhow::{Context, Result, anyhow, bail};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Default Ollama endpoint.
pub const DEFAULT_ENDPOINT: &str = "127.0.0.1:11434";

/// Default embedding model (768-D).
pub const DEFAULT_MODEL: &str = "embeddinggemma:300m";

/// Embedding dimension of `embeddinggemma:300m`.
pub const EMBEDDINGGEMMA_DIMENSION: usize = 768;

/// Embed one text via Ollama `/api/embed` (blocking).
///
/// `expected_dim` guards against silently projecting a wrong-sized vector into
/// HDC space (a mismatched model would corrupt every downstream similarity);
/// pass 0 to skip the check.
pub fn embed(endpoint: &str, model: &str, text: &str, expected_dim: usize) -> Result<Vec<f32>> {
    let body = serde_json::json!({ "model": model, "input": [text] }).to_string();
    let request = format!(
        "POST /api/embed HTTP/1.1\r\nHost: {endpoint}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );

    let mut stream =
        TcpStream::connect(endpoint).with_context(|| format!("connect to Ollama at {endpoint}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    stream.write_all(request.as_bytes())?;

    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    let response = String::from_utf8_lossy(&response);

    let status = response
        .lines()
        .next()
        .ok_or_else(|| anyhow!("empty response from Ollama"))?;
    if !status.contains("200") {
        bail!("Ollama /api/embed returned: {status}");
    }

    // Body follows the blank line; tolerate chunked transfer by scanning for
    // the JSON object start (single-object response).
    let body_start = response
        .find("\r\n\r\n")
        .map(|i| i + 4)
        .ok_or_else(|| anyhow!("malformed HTTP response"))?;
    let body = &response[body_start..];
    let json_start = body
        .find('{')
        .ok_or_else(|| anyhow!("no JSON body in Ollama response"))?;
    let json_end = body
        .rfind('}')
        .ok_or_else(|| anyhow!("truncated JSON body in Ollama response"))?;
    let parsed: serde_json::Value = serde_json::from_str(&body[json_start..=json_end])
        .context("parse Ollama embed response")?;

    let embedding: Vec<f32> = parsed["embeddings"][0]
        .as_array()
        .ok_or_else(|| anyhow!("no embeddings[0] array in Ollama response"))?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();

    if embedding.is_empty() {
        bail!("Ollama returned an empty embedding");
    }
    if expected_dim != 0 && embedding.len() != expected_dim {
        bail!(
            "Ollama model '{model}' returned {}-D embedding, expected {expected_dim}-D — \
             wrong model for this configuration",
            embedding.len()
        );
    }
    Ok(embedding)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live test — ignored by default; run with a local Ollama up:
    /// `cargo test -p symthaea-embeddings ollama -- --ignored`
    #[test]
    #[ignore = "requires local Ollama with embeddinggemma:300m"]
    fn live_embed_geometry() {
        let a = embed(
            DEFAULT_ENDPOINT,
            DEFAULT_MODEL,
            "move the arm to the left",
            768,
        )
        .unwrap();
        let b = embed(
            DEFAULT_ENDPOINT,
            DEFAULT_MODEL,
            "move the arm to the right",
            768,
        )
        .unwrap();
        let c = embed(
            DEFAULT_ENDPOINT,
            DEFAULT_MODEL,
            "the stock market fell today",
            768,
        )
        .unwrap();
        let cos = |x: &[f32], y: &[f32]| {
            let dot: f32 = x.iter().zip(y).map(|(p, q)| p * q).sum();
            let nx: f32 = x.iter().map(|p| p * p).sum::<f32>().sqrt();
            let ny: f32 = y.iter().map(|p| p * p).sum::<f32>().sqrt();
            dot / (nx * ny)
        };
        assert!(
            cos(&a, &b) > cos(&a, &c),
            "same-task phrases should be closer"
        );
    }
}
