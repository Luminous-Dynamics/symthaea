// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Newline-delimited JSON adapter for research-only nanotechnology selection
//! and independent structural receipt verification.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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

enum BoundedLine {
    Accepted(Vec<u8>),
    TooLarge { sha256: String },
}

fn digest_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn rejected_request_line(sha256: String, error_code: &str, message: &str) -> String {
    let fallback_digest = sha256.clone();
    let response = symthaea_nano_policy::ErrorResponse {
        schema_version: symthaea_nano_policy::SCHEMA_VERSION.to_owned(),
        status: "rejected_input".to_owned(),
        request_sha256: sha256,
        error_code: error_code.to_owned(),
        message: message.chars().take(300).collect(),
    };
    serde_json::to_string(&response).unwrap_or_else(|_| {
        format!(
            "{{\"schema_version\":\"{}\",\"status\":\"internal_serialization_error\",\"request_sha256\":\"{}\"}}",
            symthaea_nano_policy::SCHEMA_VERSION,
            fallback_digest
        )
    })
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

/// Append a contiguous portion of line content. Retain at most `max_bytes`
/// bytes. Once the limit is crossed, hash existing retained bytes and subsequent
/// chunks without retaining the oversized request in memory.
fn append_line_content(
    chunk: &[u8],
    max_bytes: usize,
    content_len: &mut usize,
    too_large: &mut bool,
    kept: &mut Vec<u8>,
    hasher: &mut Option<Sha256>,
) {
    if chunk.is_empty() {
        return;
    }

    let next_len = content_len.saturating_add(chunk.len());
    if *too_large {
        if let Some(hash) = hasher.as_mut() {
            hash.update(chunk);
        }
        *content_len = next_len;
        return;
    }

    if next_len <= max_bytes {
        kept.extend_from_slice(chunk);
        *content_len = next_len;
        return;
    }

    let mut hash = Sha256::new();
    hash.update(kept.as_slice());
    hash.update(chunk);
    kept.clear();
    *too_large = true;
    *content_len = next_len;
    *hasher = Some(hash);
}

/// Read one line without allowing an unterminated or oversized line to grow
/// memory without bound. The digest excludes LF and the optional CR in CRLF.
fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    max_bytes: usize,
) -> io::Result<Option<BoundedLine>> {
    let mut kept = Vec::with_capacity(max_bytes.min(64 * 1024));
    let mut hasher: Option<Sha256> = None;
    let mut content_len = 0usize;
    let mut too_large = false;
    // Hold the final byte of each buffer until we know whether it is the CR in CRLF.
    let mut pending: Option<u8> = None;
    let mut saw_any_byte = false;

    loop {
        let (consumed, newline_found) = {
            let available = reader.fill_buf()?;
            if available.is_empty() {
                if !saw_any_byte {
                    return Ok(None);
                }
                break;
            }

            let newline = available.iter().position(|byte| *byte == b'\n');
            let consumed = newline.map_or(available.len(), |position| position + 1);
            let content_len_in_chunk = newline.unwrap_or(consumed);
            let content = &available[..content_len_in_chunk];

            if newline.is_some() {
                saw_any_byte = true;
                if !content.is_empty() {
                    if let Some(previous) = pending.take() {
                        append_line_content(
                            &[previous],
                            max_bytes,
                            &mut content_len,
                            &mut too_large,
                            &mut kept,
                            &mut hasher,
                        );
                    }
                    let logical_content = content.strip_suffix(&[b'\r']).unwrap_or(content);
                    append_line_content(
                        logical_content,
                        max_bytes,
                        &mut content_len,
                        &mut too_large,
                        &mut kept,
                        &mut hasher,
                    );
                } else if let Some(previous) = pending.take() {
                    if previous != b'\r' {
                        append_line_content(
                            &[previous],
                            max_bytes,
                            &mut content_len,
                            &mut too_large,
                            &mut kept,
                            &mut hasher,
                        );
                    }
                }
            } else if !content.is_empty() {
                saw_any_byte = true;
                if let Some(previous) = pending.take() {
                    append_line_content(
                        &[previous],
                        max_bytes,
                        &mut content_len,
                        &mut too_large,
                        &mut kept,
                        &mut hasher,
                    );
                }
                let split_at = content.len() - 1;
                append_line_content(
                    &content[..split_at],
                    max_bytes,
                    &mut content_len,
                    &mut too_large,
                    &mut kept,
                    &mut hasher,
                );
                pending = Some(content[split_at]);
            }
            (consumed, newline.is_some())
        };
        reader.consume(consumed);

        if newline_found {
            break;
        }
    }

    // EOF without LF: the final pending byte is content, even if it is CR.
    if let Some(last) = pending.take() {
        append_line_content(
            &[last],
            max_bytes,
            &mut content_len,
            &mut too_large,
            &mut kept,
            &mut hasher,
        );
    }

    if too_large {
        let hash = hasher.expect("oversized lines always initialize the streaming digest");
        Ok(Some(BoundedLine::TooLarge {
            sha256: format!("{:x}", hash.finalize()),
        }))
    } else {
        Ok(Some(BoundedLine::Accepted(kept)))
    }
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

fn handle_bounded_record(record: BoundedLine, verify_receipt: bool) -> Option<String> {
    match record {
        BoundedLine::TooLarge { sha256 } => Some(if verify_receipt {
            rejected_receipt_envelope(format!(
                "receipt envelope exceeds {MAX_LINE_BYTES} bytes"
            ))
        } else {
            rejected_request_line(
                sha256,
                "request_too_large",
                &format!("request line exceeds {MAX_LINE_BYTES} bytes"),
            )
        }),
        BoundedLine::Accepted(bytes) => {
            let line = match String::from_utf8(bytes) {
                Ok(line) => line,
                Err(error) => {
                    if verify_receipt {
                        return Some(rejected_receipt_envelope(
                            "input line is not valid UTF-8",
                        ));
                    }
                    let raw = error.into_bytes();
                    return Some(rejected_request_line(
                        digest_hex(&raw),
                        "invalid_utf8",
                        "request line is not valid UTF-8",
                    ));
                }
            };
            if line.trim().is_empty() {
                return None;
            }
            Some(if verify_receipt {
                handle_receipt_line(&line)
            } else {
                symthaea_nano_policy::handle_line(&line)
            })
        }
    }
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
    let mut reader = stdin.lock();
    let mut output = stdout.lock();

    while let Some(record) = read_bounded_line(&mut reader, MAX_LINE_BYTES)? {
        if let Some(response) = handle_bounded_record(record, mode) {
            writeln!(output, "{response}")?;
        }
    }
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Cursor;
    use symthaea_nano_policy::receipt::ReceiptAudit;

    #[test]
    fn bounded_reader_accepts_exact_limit_and_strips_crlf() {
        let mut reader = Cursor::new(b"abc\r\nnext".to_vec());
        match read_bounded_line(&mut reader, 3).unwrap().unwrap() {
            BoundedLine::Accepted(line) => assert_eq!(line, b"abc"),
            BoundedLine::TooLarge { .. } => panic!("exact limit should pass"),
        }
        match read_bounded_line(&mut reader, 4).unwrap().unwrap() {
            BoundedLine::Accepted(line) => assert_eq!(line, b"next"),
            BoundedLine::TooLarge { .. } => panic!("EOF line should pass"),
        }
        assert!(read_bounded_line(&mut reader, 4).unwrap().is_none());
    }

    #[test]
    fn bounded_reader_drains_oversized_line_and_preserves_next_record() {
        let raw = b"abcdef\nok".to_vec();
        let expected_digest = digest_hex(b"abcdef");
        let mut reader = Cursor::new(raw);
        match read_bounded_line(&mut reader, 3).unwrap().unwrap() {
            BoundedLine::TooLarge { sha256 } => assert_eq!(sha256, expected_digest),
            BoundedLine::Accepted(_) => panic!("oversized record must be rejected"),
        }
        match read_bounded_line(&mut reader, 3).unwrap().unwrap() {
            BoundedLine::Accepted(line) => assert_eq!(line, b"ok"),
            BoundedLine::TooLarge { .. } => panic!("next record should be independently read"),
        }
    }

    #[test]
    fn bounded_reader_handles_crlf_split_across_buffer_boundary() {
        let mut reader = io::BufReader::with_capacity(
            1,
            Cursor::new(b"oversized\r\nnext\n".to_vec()));
        );
        let expected_digest = digest_hex(b"oversized");
        match read_bounded_line(&mut reader, 3).unwrap().unwrap() {
            BoundedLine::TooLarge { sha256 } => assert_eq!(sha256, expected_digest),
            BoundedLine::Accepted(_) => panic!("oversized record must be rejected"),
        }
        match read_bounded_line(&mut reader, 4).unwrap().unwrap() {
            BoundedLine::Accepted(line) => assert_eq!(line, b"next"),
            BoundedLine::TooLarge { .. } => panic!("next record should be accepted"),
        }
    }

    #[test]
    fn bounded_reader_hashes_chunks_without_retaining_oversized_line() {
        let mut reader = io::BufReader::with_capacity(2, Cursor::new(b"abcdefgh\nok\n".to_vec()));
        match read_bounded_line(&mut reader, 3).unwrap().unwrap() {
            BoundedLine::TooLarge { sha256 } => assert_eq!(sha256, digest_hex(b"abcdefgh")),
            BoundedLine::Accepted(_) => panic!("oversized record must be rejected"),
        }
        match read_bounded_line(&mut reader, 2).unwrap().unwrap() {
            BoundedLine::Accepted(line) => assert_eq!(line, b"ok"),
            BoundedLine::TooLarge { .. } => panic!("next record should be accepted"),
        }
    }

    #[test]
    fn bounded_reader_hashes_crlf_without_terminator() {
        let mut reader = Cursor::new(b"oversized\r\n".to_vec());
        let expected_digest = digest_hex(b"oversized");
        match read_bounded_line(&mut reader, 3).unwrap().unwrap() {
            BoundedLine::TooLarge { sha256 } => assert_eq!(sha256, expected_digest),
            BoundedLine::Accepted(_) => panic!("oversized record must be rejected"),
        }
    }

    #[test]
    fn oversized_request_error_preserves_the_full_line_digest() {
        let digest = digest_hex(b"abcdef");
        let output = rejected_request_line(
            digest.clone(),
            "request_too_large",
            "request line exceeds test limit",
        );
        let response: symthaea_nano_policy::ErrorResponse =
            serde_json::from_str(&output).unwrap();
        assert_eq!(response.request_sha256, digest);
        assert_eq!(response.error_code, "request_too_large");
        assert_eq!(response.status, "rejected_input");
    }

    #[test]
    fn bounded_record_rejects_invalid_utf8_with_raw_line_digest() {
        let raw = vec![0xff, b'{', b'}'];
        let expected_digest = digest_hex(&raw);
        let output = handle_bounded_record(BoundedLine::Accepted(raw), false).unwrap();
        let response: symthaea_nano_policy::ErrorResponse =
            serde_json::from_str(&output).unwrap();
        assert_eq!(response.error_code, "invalid_utf8");
        assert_eq!(response.request_sha256, expected_digest);
    }

    #[test]
    fn bounded_record_skips_whitespace_and_rejects_oversized_select_request() {
        assert!(handle_bounded_record(
            BoundedLine::Accepted(b"   ".to_vec()),
            false,
        )
        .is_none());

        let digest = digest_hex(b"too-large");
        let output = handle_bounded_record(
            BoundedLine::TooLarge {
                sha256: digest.clone(),
            },
            false,
        )
        .unwrap();
        let response: symthaea_nano_policy::ErrorResponse =
            serde_json::from_str(&output).unwrap();
        assert_eq!(response.error_code, "request_too_large");
        assert_eq!(response.request_sha256, digest);
    }

    #[test]
    fn bounded_record_rejects_invalid_utf8_receipt_envelope() {
        let output = handle_bounded_record(
            BoundedLine::Accepted(vec![0xff]),
            true,
        )
        .unwrap();
        let response: ReceiptEnvelopeError = serde_json::from_str(&output).unwrap();
        assert_eq!(response.status, "rejected_input");
        assert_eq!(response.error_code, "invalid_receipt_envelope");
    }

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
