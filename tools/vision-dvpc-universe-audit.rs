//! VIS-DVPC-032: verify completeness of context-specific candidate-action universes.
//! A digest establishes byte identity, not manifest authenticity or statistical identification.
#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs;

const MANIFEST_HEADER: &str = "context_id\taction_id";
const TRACE_HEADER: &str = "context_id\taction_id\ttarget_probability\tlogging_probability\tselected\treward\tcluster_id";

#[derive(Clone, Debug)]
struct CandidateSet {
    context_id: String,
    action_count: usize,
    sha256: String,
}

#[derive(Clone, Debug)]
struct Report {
    verdict: &'static str,
    reason: &'static str,
    manifest_sha256: String,
    trace_sha256: String,
    manifest_pair_count: usize,
    trace_pair_count: usize,
    candidate_sets: Vec<CandidateSet>,
    detail: String,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32;64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
        0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
        0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
        0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
        0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
        0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
        0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut message = bytes.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 { message.push(0); }
    message.extend_from_slice(&bit_len.to_be_bytes());
    let mut h = [0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    for chunk in message.chunks_exact(64) {
        let mut w=[0u32;64];
        for (i,b) in chunk.chunks_exact(4).enumerate() { w[i]=u32::from_be_bytes([b[0],b[1],b[2],b[3]]); }
        for i in 16..64 {
            let x=w[i-15]; let y=w[i-2];
            let s0=x.rotate_right(7)^x.rotate_right(18)^(x>>3);
            let s1=y.rotate_right(17)^y.rotate_right(19)^(y>>10);
            w[i]=w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let [mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh]=h;
        for i in 0..64 {
            let s1=e.rotate_right(6)^e.rotate_right(11)^e.rotate_right(25);
            let ch=(e&f)^((!e)&g);
            let t1=hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0=a.rotate_right(2)^a.rotate_right(13)^a.rotate_right(22);
            let maj=(a&b)^(a&c)^(b&c);
            let t2=s0.wrapping_add(maj);
            hh=g; g=f; f=e; e=d.wrapping_add(t1); d=c; c=b; b=a; a=t1.wrapping_add(t2);
        }
        for (slot,value) in h.iter_mut().zip([a,b,c,d,e,f,g,hh]) { *slot=slot.wrapping_add(value); }
    }
    const HEX:&[u8;16]=b"0123456789abcdef";
    let mut out=String::with_capacity(64);
    for n in h { for b in n.to_be_bytes() { out.push(HEX[(b>>4) as usize] as char); out.push(HEX[(b&15) as usize] as char); } }
    out
}



fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn verify_digest(bytes: &[u8], expected: &str, name: &str) -> Result<String, String> {
    if !is_sha256_hex(expected) {
        return Err(invalid(format!("{name} expected SHA-256 must be 64 hexadecimal characters")));
    }
    let actual = sha256_hex(bytes);
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(invalid(format!("{name} SHA-256 mismatch: expected {expected}, actual {actual}")));
    }
    Ok(actual)
}

fn parse_manifest(input: &str) -> Result<BTreeSet<(String, String)>, String> {
    let mut lines = input.lines();
    if lines.next() != Some(MANIFEST_HEADER) {
        return Err(invalid("candidate manifest header is invalid"));
    }
    let mut pairs = BTreeSet::new();
    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 2 || fields[0].is_empty() || fields[1].is_empty() {
            return Err(invalid(format!("manifest line {} must contain non-empty context_id and action_id", index + 2)));
        }
        if !pairs.insert((fields[0].to_owned(), fields[1].to_owned())) {
            return Err(invalid(format!("manifest line {} duplicates a context/action pair", index + 2)));
        }
    }
    if pairs.is_empty() {
        return Err(invalid("candidate manifest contains no pairs"));
    }
    Ok(pairs)
}

fn parse_trace(input: &str) -> Result<BTreeSet<(String, String)>, String> {
    let mut lines = input.lines();
    if lines.next() != Some(TRACE_HEADER) {
        return Err(invalid("trace header is invalid"));
    }
    let mut pairs = BTreeSet::new();
    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 7 || fields[0].is_empty() || fields[1].is_empty() {
            return Err(invalid(format!("trace line {} must contain seven fields and non-empty IDs", index + 2)));
        }
        if !pairs.insert((fields[0].to_owned(), fields[1].to_owned())) {
            return Err(invalid(format!("trace line {} duplicates a context/action pair", index + 2)));
        }
    }
    if pairs.is_empty() {
        return Err(invalid("trace contains no context/action pairs"));
    }
    Ok(pairs)
}

fn contexts(pairs: &BTreeSet<(String, String)>) -> BTreeSet<String> {
    pairs.iter().map(|(context, _)| context.clone()).collect()
}

fn candidate_set_hash(actions: &BTreeSet<String>) -> String {
    let canonical = actions.iter().map(String::as_str).collect::<Vec<_>>().join("\n") + "\n";
    sha256_hex(canonical.as_bytes())
}

fn candidate_sets(manifest: &BTreeSet<(String, String)>) -> Vec<CandidateSet> {
    let mut by_context: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (context, action) in manifest {
        by_context.entry(context.clone()).or_default().insert(action.clone());
    }
    by_context
        .into_iter()
        .map(|(context_id, actions)| CandidateSet {
            context_id,
            action_count: actions.len(),
            sha256: candidate_set_hash(&actions),
        })
        .collect()
}

fn build_report(
    verdict: &'static str,
    reason: &'static str,
    manifest_sha256: String,
    trace_sha256: String,
    manifest_pairs: &BTreeSet<(String, String)>,
    trace_pairs: &BTreeSet<(String, String)>,
    detail: String,
) -> Report {
    Report {
        verdict,
        reason,
        manifest_sha256,
        trace_sha256,
        manifest_pair_count: manifest_pairs.len(),
        trace_pair_count: trace_pairs.len(),
        candidate_sets: candidate_sets(manifest_pairs),
        detail,
    }
}

fn compare_pairs(
    manifest_bytes: &[u8],
    trace_bytes: &[u8],
    expected_manifest_sha256: &str,
    expected_trace_sha256: &str,
) -> Result<Report, String> {
    // Digest validation occurs before parsing or comparing any semantic content.
    let manifest_sha256 = verify_digest(manifest_bytes, expected_manifest_sha256, "manifest")?;
    let trace_sha256 = verify_digest(trace_bytes, expected_trace_sha256, "trace")?;
    let manifest_text = std::str::from_utf8(manifest_bytes).map_err(|e| invalid(format!("manifest is not UTF-8: {e}")))?;
    let trace_text = std::str::from_utf8(trace_bytes).map_err(|e| invalid(format!("trace is not UTF-8: {e}")))?;
    let manifest_pairs = parse_manifest(manifest_text)?;
    let trace_pairs = parse_trace(trace_text)?;

    let manifest_contexts = contexts(&manifest_pairs);
    let trace_contexts = contexts(&trace_pairs);
    let missing_contexts: Vec<String> = manifest_contexts.difference(&trace_contexts).cloned().collect();
    if !missing_contexts.is_empty() {
        return Ok(build_report(
            "INCOMPLETE_ACTION_UNIVERSE", "MISSING_CONTEXT",
            manifest_sha256, trace_sha256, &manifest_pairs, &trace_pairs,
            format!("contexts absent from trace: {}", missing_contexts.join(",")),
        ));
    }
    let extra_contexts: Vec<String> = trace_contexts.difference(&manifest_contexts).cloned().collect();
    if !extra_contexts.is_empty() {
        return Ok(build_report(
            "INCOMPLETE_ACTION_UNIVERSE", "UNDECLARED_CONTEXT",
            manifest_sha256, trace_sha256, &manifest_pairs, &trace_pairs,
            format!("contexts absent from manifest: {}", extra_contexts.join(",")),
        ));
    }
    let missing_pairs: Vec<String> = manifest_pairs.difference(&trace_pairs)
        .map(|(context, action)| format!("{context}/{action}")).collect();
    if !missing_pairs.is_empty() {
        return Ok(build_report(
            "INCOMPLETE_ACTION_UNIVERSE", "MISSING_ACTION_PAIR",
            manifest_sha256, trace_sha256, &manifest_pairs, &trace_pairs,
            format!("manifest pairs absent from trace: {}", missing_pairs.join(",")),
        ));
    }
    let extra_pairs: Vec<String> = trace_pairs.difference(&manifest_pairs)
        .map(|(context, action)| format!("{context}/{action}")).collect();
    if !extra_pairs.is_empty() {
        return Ok(build_report(
            "INCOMPLETE_ACTION_UNIVERSE", "EXTRA_ACTION_PAIR",
            manifest_sha256, trace_sha256, &manifest_pairs, &trace_pairs,
            format!("trace pairs absent from manifest: {}", extra_pairs.join(",")),
        ));
    }

    Ok(build_report(
        "COMPLETE_ACTION_UNIVERSE", "EXACT_PAIR_MATCH",
        manifest_sha256, trace_sha256, &manifest_pairs, &trace_pairs,
        "manifest and trace contain the exact same unique context/action pairs".into(),
    ))
}

fn render(report: &Report) -> String {
    let mut out = String::from("record_type\tcontext_id\tfield\tvalue\n");
    let _ = writeln!(out, "SUMMARY\t-\tverdict\t{}", report.verdict);
    let _ = writeln!(out, "SUMMARY\t-\treason\t{}", report.reason);
    let _ = writeln!(out, "SUMMARY\t-\tmanifest_sha256\t{}", report.manifest_sha256);
    let _ = writeln!(out, "SUMMARY\t-\ttrace_sha256\t{}", report.trace_sha256);
    let _ = writeln!(out, "SUMMARY\t-\tmanifest_pair_count\t{}", report.manifest_pair_count);
    let _ = writeln!(out, "SUMMARY\t-\ttrace_pair_count\t{}", report.trace_pair_count);
    let _ = writeln!(out, "SUMMARY\t-\tdetail\t{}", report.detail);
    for candidate_set in &report.candidate_sets {
        let _ = writeln!(out, "CANDIDATE_SET\t{}\taction_count\t{}", candidate_set.context_id, candidate_set.action_count);
        let _ = writeln!(out, "CANDIDATE_SET\t{}\tcanonical_action_ids_sha256\t{}", candidate_set.context_id, candidate_set.sha256);
    }
    out.push_str("LIMIT\t-\tdigest_proves_authenticity\tfalse\n");
    out.push_str("LIMIT\t-\tcompleteness_proves_statistical_identification\tfalse\n");
    out
}

fn run(args: &[String]) -> Result<String, String> {
    let [_program, flag, manifest_path, trace_path, expected_manifest, expected_trace] = args else {
        return Err("usage: vision-dvpc-universe-audit --verify MANIFEST.tsv TRACE.tsv EXPECTED_MANIFEST_SHA256 EXPECTED_TRACE_SHA256".into());
    };
    if flag != "--verify" {
        return Err("usage: vision-dvpc-universe-audit --verify MANIFEST.tsv TRACE.tsv EXPECTED_MANIFEST_SHA256 EXPECTED_TRACE_SHA256".into());
    }
    let manifest = fs::read(manifest_path).map_err(|e| invalid(format!("cannot read manifest: {e}")))?;
    let trace = fs::read(trace_path).map_err(|e| invalid(format!("cannot read trace: {e}")))?;
    let report = compare_pairs(&manifest, &trace, expected_manifest, expected_trace)?;
    Ok(render(&report))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS-DVPC-032 error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = include_str!("../tests/fixtures/vision-dvpc-032-manifest.tsv");
    const TRACE: &str = include_str!("../tests/fixtures/vision-dvpc-032-complete-trace.tsv");

    fn expected_report(manifest: &str, trace: &str) -> Report {
        compare_pairs(
            manifest.as_bytes(), trace.as_bytes(),
            &sha256_hex(manifest.as_bytes()), &sha256_hex(trace.as_bytes())
        ).unwrap()
    }

    #[test]
    fn exact_unique_pair_match_passes() {
        let report = expected_report(MANIFEST, TRACE);
        assert_eq!(report.verdict, "COMPLETE_ACTION_UNIVERSE");
        assert_eq!(report.reason, "EXACT_PAIR_MATCH");
        assert_eq!(report.manifest_pair_count, 4);
        assert_eq!(report.trace_pair_count, 4);
        assert_eq!(report.candidate_sets.len(), 2);
        assert!(report.candidate_sets.iter().all(|set| set.action_count == 2));
    }

    #[test]
    fn row_reordering_does_not_change_pair_verdict_or_candidate_hashes() {
        let reversed_manifest = format!("{}\n{}\n{}\n{}\n{}\n",
            MANIFEST_HEADER, "c2\tb", "c2\ta", "c1\tb", "c1\ta");
        let reversed_trace = format!("{}\n{}\n{}\n{}\n{}\n",
            TRACE_HEADER,
            "c2\tb\t0.5\t0.5\tyes\t0.0\tg2",
            "c2\ta\t0.5\t0.5\tno\tNA\tg2",
            "c1\tb\t0.5\t0.5\tno\tNA\tg1",
            "c1\ta\t0.5\t0.5\tyes\t1.0\tg1");
        let base = expected_report(MANIFEST, TRACE);
        let permuted = expected_report(&reversed_manifest, &reversed_trace);
        assert_eq!(base.verdict, permuted.verdict);
        assert_eq!(
            base.candidate_sets.iter().map(|s| (&s.context_id, &s.sha256)).collect::<Vec<_>>(),
            permuted.candidate_sets.iter().map(|s| (&s.context_id, &s.sha256)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn missing_candidate_action_returns_incomplete_universe() {
        let trace = TRACE.replace("c1\tb\t0.5\t0.5\tno\tNA\tg1\n", "");
        let report = expected_report(MANIFEST, &trace);
        assert_eq!(report.verdict, "INCOMPLETE_ACTION_UNIVERSE");
        assert_eq!(report.reason, "MISSING_ACTION_PAIR");
    }

    #[test]
    fn extra_candidate_action_returns_incomplete_universe() {
        let trace = TRACE.replace("c2\tb\t0.5\t0.5\tyes\t0.0\tg2\n", "c2\tb\t0.5\t0.5\tyes\t0.0\tg2\nc2\tc\t0.0\t0.0\tno\tNA\tg2\n");
        let report = expected_report(MANIFEST, &trace);
        assert_eq!(report.verdict, "INCOMPLETE_ACTION_UNIVERSE");
        assert_eq!(report.reason, "EXTRA_ACTION_PAIR");
    }

    #[test]
    fn missing_context_is_a_distinct_verdict_reason() {
        let trace = TRACE.lines().filter(|line| !line.starts_with("c2\t")).collect::<Vec<_>>().join("\n") + "\n";
        let report = expected_report(MANIFEST, &trace);
        assert_eq!(report.verdict, "INCOMPLETE_ACTION_UNIVERSE");
        assert_eq!(report.reason, "MISSING_CONTEXT");
    }

    #[test]
    fn duplicate_pairs_fail_closed() {
        let manifest = format!("{MANIFEST}c1\ta\n");
        assert!(parse_manifest(&manifest).is_err());
        let trace = format!("{TRACE}c1\ta\t0.5\t0.5\tno\tNA\tg1\n");
        assert!(parse_trace(&trace).is_err());
    }

    #[test]
    fn invalid_or_wrong_expected_digests_fail_before_comparison() {
        assert!(compare_pairs(MANIFEST.as_bytes(), TRACE.as_bytes(), "bad", &sha256_hex(TRACE.as_bytes())).is_err());
        assert!(compare_pairs(MANIFEST.as_bytes(), TRACE.as_bytes(), &"0".repeat(64), &sha256_hex(TRACE.as_bytes())).is_err());
        assert!(compare_pairs(MANIFEST.as_bytes(), TRACE.as_bytes(), &sha256_hex(MANIFEST.as_bytes()), &"0".repeat(64)).is_err());
    }

    #[test]
    fn candidate_set_hash_is_canonical_and_order_independent() {
        let a: BTreeSet<String> = ["a".to_owned(), "b".to_owned()].into_iter().collect();
        let b: BTreeSet<String> = ["b".to_owned(), "a".to_owned()].into_iter().collect();
        assert_eq!(candidate_set_hash(&a), candidate_set_hash(&b));
    }

    #[test]
    fn malformed_headers_fail_closed() {
        assert!(parse_manifest("bad\nc1\ta\n").is_err());
        assert!(parse_trace("bad\nc1\ta\n").is_err());
    }

    #[test]
    fn sha256_known_vector() {
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
