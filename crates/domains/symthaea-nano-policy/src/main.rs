// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Newline-delimited JSON adapter for research-only nanotechnology selection
//! and independent structural receipt verification.

use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};

/// Maximum input line size is shared with the library request contract.
const MAX_LINE_BYTES: usize = symthaea_nano_policy::MAX_LINE_BYTES;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptEnvelope {
    request_line: String,
    response_line: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptEnvelopeError {
    schema_version: String,
    status: String,
    error_code: String,
    message: String,
}

fn rejected_receipt_envelope(message: impl Into<String>) -> String {
    let response = ReceiptEnvelopeError {
        schema_version: symthaea_nano_policy::SCHEMA_VERSION.to_owned(),
        status: "rejected_input".to_owned(),
        error_code: "invalid_receipt_envelope".to_owned(),
        message: message.into().chars().take(300).collect(),
    };
    serde_json::to_string(&response).unwrap_or_else(|_| {
        format!(
            "{{\"schema_version\":\"{}\",\"status\":\"rejected_input\",\"error_code\":\"invalid_receipt_envelope\",\"message\":\"unable to serialize error\"}}",
            symthaea_nano_policy::SCHEMA_VERSION
        )
    })
}

fn handle_receipt_line(line: &str) -> String {
    if line.len() > MAX_LINE_BYTES {
        return rejected_receipt_envelope(format!(
            "receipt envelope exceeds {MAX_LINE_BYTES} bytes"
        ));
    }

    let envelope = match serde_json::from_str::<ReceiptEnvelope>(line) {
        Ok(envelope) => envelope,
        Err(error) => {
            return rejected_receipt_envelope(format!(
                "expected only request_line and response_line strings: {error}"
            ));
        }
    };
    let audit = symthaea_nano_policy::receipt::verify_receipt(
        &envelope.request_line,
        &envelope.response_line,
    );
    serde_json::to_string(&audit)
        .unwrap_or_else(|_| rejected_receipt_envelope("unable to serialize receipt audit"))
}

fn main() -> io::Result<()> {
    let mode = match std::env::args().nth(1).as_deref() {
        None | Some("--select") => false,
        Some("--verify-receipt") => true,
        Some(other) => {
            let error = io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown mode {other}; use --select or --verify-receipt"),
            );
            eprintln!("nano-policy: {error}");
            return Err(error);
        }
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("nano-policy: unable to read stdin line: {error}");
                return Err(error);
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let response = if mode {
            handle_receipt_line(&line)
        } else {
            symthaea_nano_policy::handle_line(&line)
        };
        writeln!(output, "{response}")?;
    }
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use symthaea_nano_policy::receipt::ReceiptAudit;

    #[test]
    fn cli_receipt_mode_accepts_an_exact_request_response_pair() {
        let request_line = include_str!("../fixtures/example-request.jsonl").trim_end();
        let response_line = symthaea_nano_policy::handle_line(request_line);
        let envelope = json!({
            "request_line": request_line,
            "response_line": response_line
        });
        let output = handle_receipt_line(&envelope.to_string());
        let audit: ReceiptAudit = serde_json::from_str(&output).unwrap();
        assert_eq!(audit.status, "passed", "{audit:?}");
    }

    #[test]
    fn cli_receipt_mode_rejects_unknown_envelope_fields() {
        let envelope = json!({
            "request_line": "{}",
            "response_line": "{}",
            "target_energy_eV": -1.0
        });
        let output = handle_receipt_line(&envelope.to_string());
        let error: ReceiptEnvelopeError = serde_json::from_str(&output).unwrap();
        assert_eq!(error.status, "rejected_input");
        assert_eq!(error.error_code, "invalid_receipt_envelope");
    }
}
