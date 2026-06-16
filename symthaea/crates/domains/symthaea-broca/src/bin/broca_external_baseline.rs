// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! broca-external-baseline: measured HTTP adapter for external model baselines.
//!
//! Currently supports Ollama's `/api/generate` shape. The report is explicit
//! about whether a request was measured successfully.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone)]
struct Options {
    url: String,
    model: String,
    prompt: String,
    json_out: Option<String>,
}

#[derive(Debug, Serialize)]
struct ExternalBaselineReport {
    schema_version: u32,
    evidence_level: &'static str,
    measured: bool,
    provider: &'static str,
    url: String,
    model: String,
    prompt: String,
    output: Option<String>,
    latency_ms: u128,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaGenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    response: Option<String>,
    error: Option<String>,
}

fn main() -> Result<()> {
    let opts = parse_args()?;
    let start = Instant::now();
    let result = call_ollama(&opts);
    let latency_ms = start.elapsed().as_millis();

    let report = match result {
        Ok(output) => ExternalBaselineReport {
            schema_version: 1,
            evidence_level: "measured",
            measured: true,
            provider: "ollama",
            url: opts.url.clone(),
            model: opts.model.clone(),
            prompt: opts.prompt.clone(),
            output: Some(output),
            latency_ms,
            error: None,
        },
        Err(error) => ExternalBaselineReport {
            schema_version: 1,
            evidence_level: "failed-measurement",
            measured: false,
            provider: "ollama",
            url: opts.url.clone(),
            model: opts.model.clone(),
            prompt: opts.prompt.clone(),
            output: None,
            latency_ms,
            error: Some(format!("{error:#}")),
        },
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = opts.json_out {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn call_ollama(opts: &Options) -> Result<String> {
    let req = OllamaGenerateRequest {
        model: &opts.model,
        prompt: &opts.prompt,
        stream: false,
    };
    let body = serde_json::to_string(&req).context("encoding Ollama request JSON")?;
    let response_body = ureq::post(&opts.url)
        .set("Content-Type", "application/json")
        .send_string(&body)
        .with_context(|| format!("posting to {}", opts.url))?
        .into_string()
        .context("reading Ollama response body")?;
    let response: OllamaGenerateResponse =
        serde_json::from_str(&response_body).context("decoding Ollama response JSON")?;
    if let Some(error) = response.error {
        anyhow::bail!(error);
    }
    response
        .response
        .context("Ollama response did not include response text")
}

fn parse_args() -> Result<Options> {
    let mut opts = Options {
        url: "http://127.0.0.1:11434/api/generate".to_string(),
        model: "gemma3:latest".to_string(),
        prompt: "Say hello from a measured baseline.".to_string(),
        json_out: None,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--url" => {
                i += 1;
                opts.url = value(&args, i, "--url")?.to_string();
            }
            "--model" => {
                i += 1;
                opts.model = value(&args, i, "--model")?.to_string();
            }
            "--prompt" => {
                i += 1;
                opts.prompt = value(&args, i, "--prompt")?.to_string();
            }
            "--json-out" => {
                i += 1;
                opts.json_out = Some(value(&args, i, "--json-out")?.to_string());
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
        i += 1;
    }
    Ok(opts)
}

fn value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index)
        .map(String::as_str)
        .with_context(|| format!("{flag} requires a value"))
}

fn print_usage() {
    eprintln!(
        "Usage: broca-external-baseline [--url URL] [--model MODEL] [--prompt TEXT] [--json-out PATH]"
    );
}
