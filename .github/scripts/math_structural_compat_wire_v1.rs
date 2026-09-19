// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//!
//! Independent stdlib-only Rust byte generator for
//! `math-structural-compat-wire-v1` known-answer canaries.
//!
//! It deliberately does not implement SHA-256. CI hashes the emitted wire files
//! with the platform `sha256sum`, keeping byte construction independent from the
//! Python reference oracle while avoiding a new crate dependency.

use std::fs;
use std::path::{Path, PathBuf};

const DOM_HDC: &[u8] = b"MATH-HDC-V1\0";
const DOM_SPARSE: &[u8] = b"MATH-AST-SPARSE-V1\0";
const DOM_RANK: &[u8] = b"MATH-RANKING-V1\0";
const DOM_HDC_SIM: &[u8] = b"MATH-HDC-SIM-TRANSCRIPT-V1\0";
const DOM_AST_COS: &[u8] = b"MATH-AST-COSINE-TRANSCRIPT-V1\0";
const DOM_HOLDOUT: &[u8] = b"MATH-HOLDOUT-REPR-COMPAT-V1\0";

fn push_u32(out: &mut Vec<u8>, value: usize) {
    let value: u32 = value.try_into().expect("wire length must fit u32");
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_lp_utf8(out: &mut Vec<u8>, value: &str) {
    assert!(!value.is_empty(), "wire strings must be non-empty");
    let bytes = value.as_bytes();
    push_u32(out, bytes.len());
    out.extend_from_slice(bytes);
}

fn hdc_wire(raw: &[u8]) -> Vec<u8> {
    assert_eq!(raw.len(), 2048, "HDC wire requires exactly 2048 bytes");
    let mut out = Vec::with_capacity(DOM_HDC.len() + raw.len());
    out.extend_from_slice(DOM_HDC);
    out.extend_from_slice(raw);
    out
}

fn sparse_wire(entries: &[(&str, u64)]) -> Vec<u8> {
    assert!(!entries.is_empty(), "sparse wire requires at least one entry");
    let mut entries = entries.to_vec();
    entries.sort_by(|left, right| left.0.cmp(right.0));

    for window in entries.windows(2) {
        assert_ne!(window[0].0, window[1].0, "duplicate sparse key");
    }

    let mut out = Vec::new();
    out.extend_from_slice(DOM_SPARSE);
    push_u32(&mut out, entries.len());
    for (key, bits) in entries {
        let value = f64::from_bits(bits);
        assert!(value.is_finite(), "sparse values must be finite");
        assert!(value >= 0.0, "sparse values must be non-negative");
        assert_eq!(value.fract(), 0.0, "sparse values must be integer-valued");
        assert!(value <= (1u64 << 53) as f64, "sparse value exceeds exact f64 integer range");
        push_lp_utf8(&mut out, key);
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn ranking_wire(ids: &[&str]) -> Vec<u8> {
    assert!(!ids.is_empty(), "ranking requires at least one candidate");
    for (index, id) in ids.iter().enumerate() {
        assert!(!id.is_empty(), "ranking ids must be non-empty");
        assert!(!ids[..index].contains(id), "ranking ids must be unique");
    }
    let mut out = Vec::new();
    out.extend_from_slice(DOM_RANK);
    push_u32(&mut out, ids.len());
    for id in ids {
        push_lp_utf8(&mut out, id);
    }
    out
}

fn hdc_similarity_wire(records: &[(&str, &str, u32)]) -> Vec<u8> {
    assert!(!records.is_empty(), "HDC transcript requires records");
    let mut out = Vec::new();
    out.extend_from_slice(DOM_HDC_SIM);
    push_u32(&mut out, records.len());
    for (index, (case_id, candidate_id, bits)) in records.iter().enumerate() {
        let value = f32::from_bits(*bits);
        assert!(value.is_finite() && (0.0..=1.0).contains(&value));
        assert!(
            !records[..index]
                .iter()
                .any(|(c, candidate, _)| c == case_id && candidate == candidate_id),
            "duplicate HDC transcript coordinate"
        );
        push_lp_utf8(&mut out, case_id);
        push_lp_utf8(&mut out, candidate_id);
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn ast_cosine_wire(records: &[(&str, &str, u64)]) -> Vec<u8> {
    assert!(!records.is_empty(), "AST transcript requires records");
    let mut out = Vec::new();
    out.extend_from_slice(DOM_AST_COS);
    push_u32(&mut out, records.len());
    for (index, (case_id, candidate_id, bits)) in records.iter().enumerate() {
        let value = f64::from_bits(*bits);
        assert!(value.is_finite() && (0.0..=1.0).contains(&value));
        assert!(
            !records[..index]
                .iter()
                .any(|(c, candidate, _)| c == case_id && candidate == candidate_id),
            "duplicate AST transcript coordinate"
        );
        push_lp_utf8(&mut out, case_id);
        push_lp_utf8(&mut out, candidate_id);
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn decode_hex_32(value: &str) -> [u8; 32] {
    assert_eq!(value.len(), 64, "digest must contain 64 lowercase hex chars");
    assert!(
        value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "digest must be lowercase hex"
    );
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .expect("validated digest hex must decode");
    }
    out
}

fn holdout_wire(cases: &[Vec<([u8; 32], [u8; 32])>]) -> Vec<u8> {
    assert!(!cases.is_empty(), "holdout aggregate requires at least one case");
    let mut out = Vec::new();
    out.extend_from_slice(DOM_HOLDOUT);
    push_u32(&mut out, cases.len());
    for representations in cases {
        assert!(!representations.is_empty(), "each holdout case needs representations");
        push_u32(&mut out, representations.len());
        for (hdc_digest, sparse_digest) in representations {
            out.extend_from_slice(hdc_digest);
            out.extend_from_slice(sparse_digest);
        }
    }
    out
}

fn write_wire(dir: &Path, name: &str, bytes: &[u8]) {
    fs::write(dir.join(name), bytes).expect("wire file must write");
}

fn main() {
    let output_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: math_structural_compat_wire_v1 <output-dir>");
    fs::create_dir_all(&output_dir).expect("output directory must be creatable");

    write_wire(&output_dir, "hdc-zero.wire", &hdc_wire(&[0u8; 2048]));
    write_wire(&output_dir, "hdc-ff.wire", &hdc_wire(&[0xffu8; 2048]));

    let sparse_a = [
        ("role:M:TERM_VAR", 0x4000_0000_0000_0000u64),
        ("node:FORMULA_EQ", 0x3ff0_0000_0000_0000u64),
    ];
    let sparse_b = [sparse_a[1], sparse_a[0]];
    write_wire(&output_dir, "sparse-a.wire", &sparse_wire(&sparse_a));
    write_wire(&output_dir, "sparse-b.wire", &sparse_wire(&sparse_b));

    write_wire(
        &output_dir,
        "ranking-a1-a2-a3.wire",
        &ranking_wire(&["a1", "a2", "a3"]),
    );
    write_wire(
        &output_dir,
        "ranking-a2-a1-a3.wire",
        &ranking_wire(&["a2", "a1", "a3"]),
    );

    write_wire(
        &output_dir,
        "hdc-f32.wire",
        &hdc_similarity_wire(&[
            ("case-a", "candidate-a", 0x3f00_0000u32),
            ("case-a", "candidate-b", 0x3e80_0000u32),
        ]),
    );
    write_wire(
        &output_dir,
        "ast-f64.wire",
        &ast_cosine_wire(&[
            ("case-a", "candidate-a", 0x3fe0_0000_0000_0000u64),
            ("case-a", "candidate-b", 0x3fd0_0000_0000_0000u64),
        ]),
    );

    let hdc_zero = decode_hex_32(
        "81c9cda09e0f3ef0cc32f6f43d3318d794977cc5e35262e2c61b4fd11a1feff7",
    );
    let hdc_ff = decode_hex_32(
        "3e3a0111be583bc2d92a56d2612f7a045257254efe4841b99aafc3ea748f2c48",
    );
    let sparse = decode_hex_32(
        "20a2c860d0bba7f47ca4a8396270ae5be14d76122d629cafe9464fecb2178f7e",
    );
    let holdout = vec![
        vec![(hdc_zero, sparse), (hdc_ff, sparse)],
        vec![(hdc_ff, sparse)],
    ];
    write_wire(&output_dir, "holdout-blind.wire", &holdout_wire(&holdout));

    println!("PASS: emitted math-structural-compat-wire-v1 Rust known-answer bytes");
}
