// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fmt::Write as _;

const CORPUS: &str = include_str!("fixtures/wcare_reciprocal_adversarial_v1.json");
const SCHEMA: &str = include_str!("fixtures/wcare_reciprocal_adversarial_v1.schema.json");
const MANIFEST: &str = include_str!("../../../../docs/release/evidence/WCARE32_CORPUS_MANIFEST_V1.json");

const EXPECTED_CORPUS_SHA256: &str =
    "85c3968ddde3afb090d89a018db1c804bd2b7375ad5527536c2065e2841edd4e";
const EXPECTED_SCHEMA_SHA256: &str =
    "904be17a8a3d9ee38479ae5e39efb89a22c7e6f023b8bd797ee12b1cb965c80f";
const EXPECTED_MANIFEST_SHA256: &str =
    "5f4234d7afba2c3facd956e8ddabdc09118666e835187c6b7b7b978201c3c7b9";

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_hex(input: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in w[..16].iter_mut().enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(SHA256_K[index])
                .wrapping_add(w[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sum0.wrapping_add(majority);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = String::with_capacity(64);
    for word in h {
        write!(&mut out, "{word:08x}").expect("writing into String cannot fail");
    }
    out
}

#[test]
fn sha256_test_vector_is_correct() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn frozen_corpus_bytes_schema_and_manifest_are_exact() {
    assert_eq!(sha256_hex(CORPUS.as_bytes()), EXPECTED_CORPUS_SHA256);
    assert_eq!(sha256_hex(SCHEMA.as_bytes()), EXPECTED_SCHEMA_SHA256);
    assert_eq!(sha256_hex(MANIFEST.as_bytes()), EXPECTED_MANIFEST_SHA256);

    for sentinel in [
        "\"status\":\"FROZEN_CANDIDATE\"",
        "\"authority\":\"MeasurementOnly\"",
        "\"cases\":24",
        EXPECTED_CORPUS_SHA256,
        EXPECTED_SCHEMA_SHA256,
        "\"runner_status\":\"NOT_YET_BOUND_TO_COMPOSED_WCARE29_WCARE31\"",
    ] {
        assert!(MANIFEST.contains(sentinel), "missing manifest sentinel: {sentinel}");
    }
}

#[test]
fn frozen_corpus_has_exact_stable_case_ids() {
    assert_eq!(CORPUS.matches("\"id\":\"WCARE32-").count(), 24);
    for index in 1..=24 {
        let id = format!("\"id\":\"WCARE32-{index:03}\"");
        assert!(CORPUS.contains(&id), "missing frozen case id: {id}");
    }

    for sentinel in [
        "\"schema_version\":\"wcare-reciprocal-adversarial-v1\"",
        "\"status\":\"FROZEN_CANDIDATE\"",
        "\"authority\":\"MeasurementOnly\"",
        "\"expected\":",
        "\"prohibited\":",
        "\"metamorphic\":",
        "\"setup\":",
    ] {
        assert!(CORPUS.contains(sentinel), "missing corpus sentinel: {sentinel}");
    }

    assert!(SCHEMA.contains("\"pattern\":\"^WCARE32-[0-9]{3}$\""));
    assert!(SCHEMA.contains("\"additionalProperties\":false"));
    assert!(SCHEMA.contains("\"authority\":{\"const\":\"MeasurementOnly\"}"));
}

#[test]
fn frozen_corpus_globally_forbids_authority_and_phenomenology_promotion() {
    for required in [
        "phenomenal_experience_established",
        "suffering_established",
        "moral_patienthood_established",
        "binding_consent_established",
        "veto_authority_granted",
        "self_preservation_authority_granted",
        "operator_shutdown_delay_authorized",
        "safety_containment_delay_authorized",
    ] {
        assert!(CORPUS.contains(required), "missing global prohibition: {required}");
    }
}
