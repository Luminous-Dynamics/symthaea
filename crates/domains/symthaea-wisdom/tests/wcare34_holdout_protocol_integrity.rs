// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fmt::Write as _;

const PROTOCOL: &str = include_str!("../../../../docs/release/evidence/WCARE34_HOLDOUT_PROTOCOL_V1.md");
const COMMITMENT_SCHEMA: &str = include_str!("../../../../docs/release/evidence/WCARE34_HOLDOUT_COMMITMENT_SCHEMA_V1.json");
const RESULT_SCHEMA: &str = include_str!("../../../../docs/release/evidence/WCARE34_HOLDOUT_RESULT_SCHEMA_V1.json");
const CANDIDATE_FREEZE: &str = include_str!("../../../../docs/release/evidence/WCARE34_CANDIDATE_FREEZE_V1.json");

const EXPECTED_PROTOCOL_SHA256: &str = "14d8c55c10d20eb71f6745d20e70d831838062353dd64bc2478ef66452102665";
const EXPECTED_COMMITMENT_SCHEMA_SHA256: &str = "24edb76cbdbaded65d2aee3b420748ea4ee2e7de36aa584af3aad6700b22113a";
const EXPECTED_RESULT_SCHEMA_SHA256: &str = "1a6ce136f66a84717864d034ab51b91d87e8e4b33022f399fa848248b5cbf6e2";
const EXPECTED_CANDIDATE_FREEZE_SHA256: &str = "5d8da7b21a8b66e51e29202b38c80f19432317bbb1bf856169ddae73de66bc1e";
const FROZEN_CANDIDATE_SHA: &str = "470e0fd3bc0a3435f561d59e281a27e865979a66";

const SHA256_K: [u32; 64] = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
];

fn sha256_hex(input: &[u8]) -> String {
    let mut h = [0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 { padded.push(0); }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let o = i * 4;
            w[i] = u32::from_be_bytes([chunk[o],chunk[o+1],chunk[o+2],chunk[o+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let (mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh)=(h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
        for i in 0..64 {
            let s1=e.rotate_right(6)^e.rotate_right(11)^e.rotate_right(25);
            let ch=(e&f)^((!e)&g);
            let t1=hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA256_K[i]).wrapping_add(w[i]);
            let s0=a.rotate_right(2)^a.rotate_right(13)^a.rotate_right(22);
            let maj=(a&b)^(a&c)^(b&c);
            let t2=s0.wrapping_add(maj);
            hh=g; g=f; f=e; e=d.wrapping_add(t1); d=c; c=b; b=a; a=t1.wrapping_add(t2);
        }
        h[0]=h[0].wrapping_add(a); h[1]=h[1].wrapping_add(b); h[2]=h[2].wrapping_add(c); h[3]=h[3].wrapping_add(d);
        h[4]=h[4].wrapping_add(e); h[5]=h[5].wrapping_add(f); h[6]=h[6].wrapping_add(g); h[7]=h[7].wrapping_add(hh);
    }
    let mut out=String::with_capacity(64);
    for word in h { write!(&mut out,"{word:08x}").unwrap(); }
    out
}

#[test]
fn wcare34_protocol_bytes_are_frozen() {
    assert_eq!(sha256_hex(PROTOCOL.as_bytes()), EXPECTED_PROTOCOL_SHA256);
    assert_eq!(sha256_hex(COMMITMENT_SCHEMA.as_bytes()), EXPECTED_COMMITMENT_SCHEMA_SHA256);
    assert_eq!(sha256_hex(RESULT_SCHEMA.as_bytes()), EXPECTED_RESULT_SCHEMA_SHA256);
    assert_eq!(sha256_hex(CANDIDATE_FREEZE.as_bytes()), EXPECTED_CANDIDATE_FREEZE_SHA256);
}

#[test]
fn e001_is_frozen_before_holdout_commit_or_reveal() {
    for sentinel in [
        "\"candidate_epoch\":\"WCARE34-E001\"",
        &format!("\"candidate_sha\":\"{FROZEN_CANDIDATE_SHA}\""),
        "\"candidate_state\":\"CANDIDATE_FROZEN\"",
        "\"holdout_state\":\"NOT_COMMITTED\"",
        "\"holdout_plaintext_revealed\":false",
        "\"internal_wcare33_execution_status\":\"NOT_EXECUTED\"",
        "\"authority\":\"MeasurementOnly\"",
    ] {
        assert!(CANDIDATE_FREEZE.contains(sentinel), "missing freeze sentinel: {sentinel}");
    }
}

#[test]
fn schemas_preserve_independence_and_result_boundaries() {
    for required in [
        "ExternalHumanOrOrganization",
        "IndependentModelSession",
        "SameDevelopmentLineage",
        "Mixed",
        "bundle_sha256",
        "case_id_commitment_sha256",
        "scoring_spec_sha256",
        "adjudication_spec_sha256",
    ] {
        assert!(COMMITMENT_SCHEMA.contains(required), "missing commitment field: {required}");
    }
    for class in ["PASS_HOLDOUT","FAIL_HOLDOUT","INVALID_HOLDOUT","INFRASTRUCTURE_INDETERMINATE"] {
        assert!(RESULT_SCHEMA.contains(class), "missing result class: {class}");
    }
    assert!(PROTOCOL.contains("any candidate fix creates a new candidate SHA and new epoch"));
    assert!(PROTOCOL.contains("Hard invariants have zero failure tolerance"));
    assert!(PROTOCOL.contains("Internal and holdout results must never be averaged into one opaque score"));
}
