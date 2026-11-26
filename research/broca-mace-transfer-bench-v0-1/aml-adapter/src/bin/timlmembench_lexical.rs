// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline JSONL entry point for the TimlMemBench protocol smoke adapter.
//!
//! Run with:
//! cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/aml-adapter/Cargo.toml --bin timlmembench_lexical

use std::io::{self, BufRead, Write};

use broca_aml_adapter::timlmembench_protocol::TimlMemBenchLexicalAdapterV1;
use serde_json::{json, Value};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let mut adapter = TimlMemBenchLexicalAdapterV1::new();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Result<Value, _> = serde_json::from_str(&line);
        let is_close = request.as_ref()
            .ok()
            .and_then(|value| value.get("request_type"))
            .and_then(Value::as_str) == Some("close");
        let response = match request {
            Ok(value) => adapter.handle(&value),
            Err(error) => json!({
                "error": format!("invalid JSON request: {error}")
            }),
        };
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
        if is_close {
            break;
        }
    }
    Ok(())
}
