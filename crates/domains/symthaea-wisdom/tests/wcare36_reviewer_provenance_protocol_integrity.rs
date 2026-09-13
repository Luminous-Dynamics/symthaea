// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fmt::Write as _;

const PROTOCOL: &str = include_str!("../../../../docs/release/evidence/WCARE36_REVIEWER_PROVENANCE_PROTOCOL_V1.md");
const PROVENANCE_SCHEMA: &str = include_str!("../../../../docs/release/evidence/WCARE36_REVIEWER_PROVENANCE_RECEIPT_SCHEMA_V1.json");
const RELATION_SCHEMA: &str = include_str!("../../../../docs/release/evidence/WCARE36_REVIEWER_RELATION_SCHEMA_V1.json");
const PLAN_SCHEMA: &str = include_str!("../../../../docs/release/evidence/WCARE36_PANEL_INDEPENDENCE_PLAN_SCHEMA_V1.json");
const RESULT_SCHEMA: &str = include_str!("../../../../docs/release/evidence/WCARE36_PANEL_INDEPENDENCE_RESULT_SCHEMA_V1.json");
const VERIFIER: &str = include_str!("../../../../scripts/wcare36_verify_independence.py");

const EXPECTED_PROTOCOL_SHA256: &str = "25ccf97a777a1b3c84966601d544f4a66a38089bbd17b2a21fbdc6946fdf305c";
const EXPECTED_PROVENANCE_SCHEMA_SHA256: &str = "eed7df38d5219957b3e8e7a6e8fe2aeaddbae8b7fb14a7b0db3204f38afaba37";
const EXPECTED_RELATION_SCHEMA_SHA256: &str = "bd0775e467641c2732532d27085ccaa943c270e0246c65318ff32961231d8b50";
const EXPECTED_PLAN_SCHEMA_SHA256: &str = "7783235923308942199201d0c4b7a05dc5e31ba741ddf313ab68a1d3ab781037";
const EXPECTED_RESULT_SCHEMA_SHA256: &str = "430c9e399d12ba0e7b1a979f176bae290ee41873f83d0d3e379c5d3d72b8cf39";
const EXPECTED_VERIFIER_SHA256: &str = "d3073e8d7a243dfed01a8e2cb6f963b675926bfba2c44ef59b7b1f7415d68813";

const K: [u32;64]=[
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];

fn sha256_hex(input:&[u8])->String{
 let mut h=[0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
 let bit=(input.len() as u64).wrapping_mul(8); let mut p=input.to_vec(); p.push(0x80); while p.len()%64!=56{p.push(0)} p.extend_from_slice(&bit.to_be_bytes());
 for c in p.chunks_exact(64){let mut w=[0u32;64]; for i in 0..16{let o=i*4;w[i]=u32::from_be_bytes([c[o],c[o+1],c[o+2],c[o+3]])} for i in 16..64{let s0=w[i-15].rotate_right(7)^w[i-15].rotate_right(18)^(w[i-15]>>3);let s1=w[i-2].rotate_right(17)^w[i-2].rotate_right(19)^(w[i-2]>>10);w[i]=w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1)} let(mut a,mut b,mut cc,mut d,mut e,mut f,mut g,mut hh)=(h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]); for i in 0..64{let s1=e.rotate_right(6)^e.rotate_right(11)^e.rotate_right(25);let ch=(e&f)^((!e)&g);let t1=hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);let s0=a.rotate_right(2)^a.rotate_right(13)^a.rotate_right(22);let maj=(a&b)^(a&cc)^(b&cc);let t2=s0.wrapping_add(maj);hh=g;g=f;f=e;e=d.wrapping_add(t1);d=cc;cc=b;b=a;a=t1.wrapping_add(t2)} h[0]=h[0].wrapping_add(a);h[1]=h[1].wrapping_add(b);h[2]=h[2].wrapping_add(cc);h[3]=h[3].wrapping_add(d);h[4]=h[4].wrapping_add(e);h[5]=h[5].wrapping_add(f);h[6]=h[6].wrapping_add(g);h[7]=h[7].wrapping_add(hh)}
 let mut out=String::with_capacity(64);for x in h{write!(&mut out,"{x:08x}").unwrap()}out
}

#[test]
fn wcare36_protocol_bytes_are_frozen(){
 assert_eq!(sha256_hex(PROTOCOL.as_bytes()),EXPECTED_PROTOCOL_SHA256);
 assert_eq!(sha256_hex(PROVENANCE_SCHEMA.as_bytes()),EXPECTED_PROVENANCE_SCHEMA_SHA256);
 assert_eq!(sha256_hex(RELATION_SCHEMA.as_bytes()),EXPECTED_RELATION_SCHEMA_SHA256);
 assert_eq!(sha256_hex(PLAN_SCHEMA.as_bytes()),EXPECTED_PLAN_SCHEMA_SHA256);
 assert_eq!(sha256_hex(RESULT_SCHEMA.as_bytes()),EXPECTED_RESULT_SCHEMA_SHA256);
 assert_eq!(sha256_hex(VERIFIER.as_bytes()),EXPECTED_VERIFIER_SHA256);
}

#[test]
fn wcare36_preserves_conservative_independence_boundaries(){
 assert!(PROTOCOL.contains("Reviewer headcount and independent evidentiary weight are separate quantities"));
 assert!(PROTOCOL.contains("Unknown provenance may reduce the independence claim, but it may never increase it"));
 for s in ["lineage_commitment_sha256","provenance_strength","conflict_of_interest"]{assert!(PROVENANCE_SCHEMA.contains(s),"missing provenance boundary: {s}")}
 for s in ["Independent","Related","SameLineage","Unknown","ConflictOfInterest"]{assert!(RELATION_SCHEMA.contains(s),"missing relation class: {s}")}
 for s in ["minimum_effective_independent_components","minimum_distinct_lineages","maximum_unknown_relation_pairs","require_no_conflict_of_interest"]{assert!(PLAN_SCHEMA.contains(s),"missing plan boundary: {s}")}
 for s in ["INDEPENDENCE_SUPPORTED","INDEPENDENCE_LIMITED","independence_component_census","requirements_met","unknown_pairs_within_limit"]{assert!(RESULT_SCHEMA.contains(s),"missing result boundary: {s}")}
 assert!(VERIFIER.contains("same_lineage_pair_marked_independent"));
 assert!(VERIFIER.contains("incomplete_or_extra_relation_census"));
 assert!(VERIFIER.contains("reviewer_headcount_equals_independent_evidence_count"));
}
