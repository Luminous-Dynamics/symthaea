// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical evidence encoding with RustCrypto SHA-256 commitments.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    for byte in digest {
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

pub fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, serde_json::Error> {
    let value = serde_json::to_value(value)?;
    let mut canonical = String::new();
    write_canonical_json(&value, &mut canonical);
    Ok(canonical.into_bytes())
}

pub fn canonical_json_sha256<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    Ok(sha256_hex(&canonical_json_bytes(value)?))
}

fn write_canonical_json(value: &serde_json::Value, output: &mut String) {
    match value {
        serde_json::Value::Null => output.push_str("null"),
        serde_json::Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        serde_json::Value::Number(value) => output.push_str(&value.to_string()),
        serde_json::Value::String(value) => {
            output.push_str(
                &serde_json::to_string(value).expect("serializing a JSON string cannot fail"),
            );
        }
        serde_json::Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical_json(value, output);
            }
            output.push(']');
        }
        serde_json::Value::Object(values) => {
            output.push('{');
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(key).expect("serializing a JSON key cannot fail"),
                );
                output.push(':');
                write_canonical_json(value, output);
            }
            output.push('}');
        }
    }
}

pub fn decode_hex_32(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let mut result = [0u8; 32];
    for (index, byte) in result.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_nist_sha256_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn rustcrypto_hashes_padding_boundary_lengths() {
        for length in [55usize, 56, 63, 64, 65, 1_000_000] {
            let payload = vec![b'a'; length];
            let digest = sha256_hex(&payload);
            assert_eq!(digest.len(), 64);
            assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
    }

    #[test]
    fn canonical_json_digest_ignores_pretty_printing() {
        let left: serde_json::Value = serde_json::from_str(r#"{"b":2,"a":1}"#).unwrap();
        let right: serde_json::Value = serde_json::from_str(r#"{"a":1,"b":2}"#).unwrap();
        assert_eq!(
            canonical_json_sha256(&left).unwrap(),
            canonical_json_sha256(&right).unwrap()
        );
    }

    #[test]
    fn parses_exact_256_bit_secret() {
        let value = "11".repeat(32);
        assert_eq!(decode_hex_32(&value), Some([0x11; 32]));
        assert!(decode_hex_32("11").is_none());
    }
}
