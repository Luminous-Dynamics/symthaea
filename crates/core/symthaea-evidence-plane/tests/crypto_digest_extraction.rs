// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification harness for the neutral SHA-256 extraction candidate.
//!
//! The candidate is intentionally not exported by `symthaea-evidence-plane`
//! yet. This harness compiles the exact candidate source and the existing
//! fabrication implementation directly, without adding a crate dependency, so
//! migration cannot rely on source-copy assumptions alone.

#[path = "../src/crypto_digest.rs"]
mod candidate;
#[path = "../../../domains/symthaea-fabrication-kernel/src/crypto_digest.rs"]
mod fabrication;

use candidate::{Sha256, Sha256Digest, sha256};

#[test]
fn nist_style_known_answers_match() {
    let vectors: &[(&[u8], &str)] = &[
        (
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            b"The quick brown fox jumps over the lazy dog",
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
        ),
    ];

    for (input, expected) in vectors {
        assert_eq!(sha256(input).to_hex(), *expected);
    }
}

#[test]
fn every_chunking_of_representative_message_matches_one_shot() {
    let bytes: Vec<u8> = (0..=255).cycle().take(4097).collect();
    let expected = sha256(&bytes);

    for chunk_size in [1usize, 2, 3, 7, 31, 55, 56, 63, 64, 65, 127, 1024] {
        let mut incremental = Sha256::new();
        for chunk in bytes.chunks(chunk_size) {
            incremental.update(chunk);
        }
        assert_eq!(incremental.finalize(), expected, "chunk size {chunk_size}");
    }
}

#[test]
fn padding_boundaries_match_one_shot() {
    for len in [0usize, 1, 55, 56, 57, 63, 64, 65, 119, 120, 121, 127, 128, 129] {
        let bytes = vec![0xa5; len];
        let expected = sha256(&bytes);
        for chunk_size in [1usize, 7, 31, 63, 64] {
            let mut incremental = Sha256::new();
            for chunk in bytes.chunks(chunk_size) {
                incremental.update(chunk);
            }
            assert_eq!(
                incremental.finalize(),
                expected,
                "length {len}, chunk size {chunk_size}"
            );
        }
    }
}

#[test]
fn digest_hex_round_trip_is_exact() {
    for input in [b"symthaea".as_slice(), b"economic-science".as_slice()] {
        let digest = sha256(input);
        let text = digest.to_hex();
        assert_eq!(text.len(), 64);
        assert_eq!(Sha256Digest::from_hex(&text).unwrap(), digest);
    }
}

#[test]
fn neutral_candidate_matches_existing_fabrication_digest() {
    let mut corpus: Vec<Vec<u8>> = vec![
        Vec::new(),
        b"abc".to_vec(),
        b"economic-science".to_vec(),
        (0u8..=255).collect(),
    ];
    for len in [1usize, 55, 56, 57, 63, 64, 65, 127, 128, 129, 1024, 4097] {
        corpus.push((0..len).map(|index| (index.wrapping_mul(131) & 0xff) as u8).collect());
    }

    for bytes in corpus {
        assert_eq!(
            candidate::sha256(&bytes).to_hex(),
            fabrication::sha256(&bytes).to_hex(),
            "one-shot parity failed for {} bytes",
            bytes.len()
        );

        for chunk_size in [1usize, 7, 31, 64, 127] {
            let mut left = candidate::Sha256::new();
            let mut right = fabrication::Sha256::new();
            for chunk in bytes.chunks(chunk_size) {
                left.update(chunk);
                right.update(chunk);
            }
            assert_eq!(
                left.finalize().to_hex(),
                right.finalize().to_hex(),
                "incremental parity failed for {} bytes with chunk size {chunk_size}",
                bytes.len()
            );
        }
    }
}
