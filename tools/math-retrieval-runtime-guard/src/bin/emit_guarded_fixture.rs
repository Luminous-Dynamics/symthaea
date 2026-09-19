// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MATH-RET-RUNTIME-001D guarded single-index fixture emitter.
//!
//! This is the #4324 deterministic S-arm emitter with one additional runtime
//! theorem: the representation backend is wrapped in `MembershipGuardBackend`
//! built from the exact frozen candidate-set artifact. An optional negative
//! mode injects an out-of-universe source and must fail before materialization
//! or evidence staging.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use symthaea_math_retrieval_membership_guard::{
    FrozenCandidateUniverse, MembershipGuardBackend,
};
use symthaea_math_retrieval_runtime_seam::{
    BackendExecution, BackendRetrieval, CanonicalSourceMaterializer, ControlBinding,
    ControlTransform, GraphIdentity, InMemoryEvidenceSink, PayloadAuditBatch, PayloadRole,
    QualifiedRetrievalBackend, QualifiedRetrievalRequest, RetrievalBudget, RetrievalError,
    RetrievalExecutor, RetrievalTraceReceipt, Sha256Digest, SingleIndexExecution, StopReason,
    TraceRetrieval,
};

fn required<'a>(m: &'a BTreeMap<String, String>, key: &str) -> &'a str {
    m.get(key)
        .unwrap_or_else(|| panic!("missing config key: {key}"))
}

fn parse_usize(m: &BTreeMap<String, String>, key: &str) -> usize {
    required(m, key)
        .parse()
        .unwrap_or_else(|_| panic!("{key}: invalid usize"))
}

fn parse_u64(m: &BTreeMap<String, String>, key: &str) -> u64 {
    required(m, key)
        .parse()
        .unwrap_or_else(|_| panic!("{key}: invalid u64"))
}

fn parse_u32(m: &BTreeMap<String, String>, key: &str) -> u32 {
    required(m, key)
        .parse()
        .unwrap_or_else(|_| panic!("{key}: invalid u32"))
}

fn digest(m: &BTreeMap<String, String>, key: &str) -> Sha256Digest {
    Sha256Digest::parse(required(m, key)).unwrap_or_else(|e| panic!("{key}: {e}"))
}

fn digest_csv(m: &BTreeMap<String, String>, key: &str) -> Vec<Sha256Digest> {
    required(m, key)
        .split(',')
        .map(|x| Sha256Digest::parse(x).unwrap_or_else(|e| panic!("{key}: {e}")))
        .collect()
}

fn load_config(path: &Path) -> BTreeMap<String, String> {
    let text = fs::read_to_string(path).expect("read guarded-adapter config");
    let mut out = BTreeMap::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .unwrap_or_else(|| panic!("config line {} must be key=value", line_no + 1));
        let key = key.trim().to_string();
        let value = value.trim().to_string();
        assert!(!key.is_empty(), "empty config key at line {}", line_no + 1);
        assert!(!value.is_empty(), "empty config value for {key}");
        assert!(
            out.insert(key.clone(), value).is_none(),
            "duplicate config key: {key}"
        );
    }
    out
}

fn j(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn digest_json(d: &Sha256Digest) -> String {
    j(d.as_str())
}

fn digest_list(xs: &[Sha256Digest]) -> String {
    format!(
        "[{}]",
        xs.iter().map(digest_json).collect::<Vec<_>>().join(",")
    )
}

fn compute_decimal(microunits: u64) -> String {
    let whole = microunits / 1_000_000;
    let frac = microunits % 1_000_000;
    if frac == 0 {
        return whole.to_string();
    }
    let mut frac_text = format!("{frac:06}");
    while frac_text.ends_with('0') {
        frac_text.pop();
    }
    format!("{whole}.{frac_text}")
}

fn serialize_trace(t: &RetrievalTraceReceipt) -> String {
    let retrieval = match &t.retrieval {
        TraceRetrieval::SingleIndex {
            index_manifest_sha256,
            index_artifact_sha256,
            requested_k,
            ranked_source_object_digests,
        } => format!(
            "{{\"mode\":\"SingleIndex\",\"single\":{{\"index_manifest_sha256\":{},\"index_artifact_sha256\":{},\"requested_k\":{},\"ranked_source_object_digests\":{}}}}}",
            digest_json(index_manifest_sha256),
            digest_json(index_artifact_sha256),
            requested_k,
            digest_list(ranked_source_object_digests),
        ),
        TraceRetrieval::Fusion { .. } => panic!("001D fixture intentionally targets SingleIndex arm S"),
    };

    let output = t
        .packing
        .output
        .iter()
        .map(|x| {
            format!(
                "{{\"rank\":{},\"source_object_sha256\":{},\"canonical_payload_bytes\":{}}}",
                x.rank,
                digest_json(&x.source_object_sha256),
                x.canonical_payload_bytes
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    let mut packing = format!(
        "{{\"context_packer_sha256\":{},\"input_ranked_source_object_digests\":{},\"output\":[{}],\"output_items_used\":{},\"output_bytes_used\":{},\"stop_reason\":{}",
        digest_json(&t.packing.context_packer_sha256),
        digest_list(&t.packing.input_ranked_source_object_digests),
        output,
        t.packing.output_items_used,
        t.packing.output_bytes_used,
        j(match t.packing.stop_reason {
            StopReason::Empty => "Empty",
            StopReason::RankedCandidatesExhausted => "RankedCandidatesExhausted",
            StopReason::ItemLimitReached => "ItemLimitReached",
            StopReason::FirstNonFittingItem => "FirstNonFittingItem",
        }),
    );
    if let Some(x) = &t.packing.first_nonfitting {
        packing.push_str(&format!(
            ",\"first_nonfitting\":{{\"rank\":{},\"source_object_sha256\":{},\"canonical_payload_bytes\":{}}}",
            x.rank,
            digest_json(&x.source_object_sha256),
            x.canonical_payload_bytes
        ));
    }
    packing.push('}');

    let controls = t
        .control_bindings
        .iter()
        .map(|x| {
            let transform = match x.control_transform {
                ControlTransform::None => "None",
                ControlTransform::RandomRetrieval => "RandomRetrieval",
                ControlTransform::ShuffledHdcVectors => "ShuffledHdcVectors",
                ControlTransform::PermutedChallengeAssociations => "PermutedChallengeAssociations",
            };
            let mut s = format!(
                "{{\"index_manifest_sha256\":{},\"control_transform\":{}",
                digest_json(&x.index_manifest_sha256),
                j(transform)
            );
            if let Some(seed) = x.control_seed {
                s.push_str(&format!(",\"control_seed\":{seed}"));
            }
            if let Some(d) = &x.control_artifact_sha256 {
                s.push_str(&format!(
                    ",\"control_artifact_sha256\":{}",
                    digest_json(d)
                ));
            }
            s.push('}');
            s
        })
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "{{\"version\":{},\"trace_id\":{},\"authority\":{},\"graph\":{{\"bundle_sha256\":{},\"graph_report_sha256\":{},\"experiment_sha256\":{},\"arm_id\":{},\"retrieval_binding_sha256\":{},\"candidate_set_sha256\":{},\"candidate_count\":{},\"context_packer_sha256\":{}}},\"experiment_seed\":{},\"query\":{{\"query_id\":{},\"query_source_object_sha256\":{}}},\"retrieval\":{},\"packing\":{},\"resources\":{{\"retrieval_queries_used\":{},\"wall_time_ms_used\":{},\"normalized_compute_units_used_decimal\":{}}},\"control_bindings\":[{}]}}\n",
        j(t.version),
        j(&t.trace_id),
        j(t.authority),
        digest_json(&t.graph.bundle_sha256),
        digest_json(&t.graph.graph_report_sha256),
        digest_json(&t.graph.experiment_sha256),
        j(&t.arm_id),
        digest_json(&t.graph.retrieval_binding_sha256),
        digest_json(&t.graph.candidate_set_sha256),
        t.graph.candidate_count,
        digest_json(&t.graph.context_packer_sha256),
        t.experiment_seed,
        j(&t.query_id),
        digest_json(&t.query_source_object_sha256),
        retrieval,
        packing,
        t.resources.retrieval_queries_used,
        t.resources.wall_time_ms_used,
        j(&compute_decimal(t.resources.normalized_compute_microunits_used)),
        controls,
    )
}

fn write_payload_plan(out_dir: &Path, audit: &PayloadAuditBatch) {
    let payload_dir = out_dir.join("payloads");
    fs::create_dir_all(&payload_dir).expect("create payload dir");
    let mut rows = String::from(
        "role\trank\tsource_object_sha256\tpayload_path\tcanonical_payload_bytes\n",
    );
    for entry in &audit.entries {
        let role = match entry.role {
            PayloadRole::Delivered => "Delivered",
            PayloadRole::FirstNonFitting => "FirstNonFitting",
        };
        let rel = format!("payloads/rank-{:02}.txt", entry.rank);
        fs::write(out_dir.join(&rel), &entry.canonical_payload_utf8).expect("write payload bytes");
        rows.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\n",
            role,
            entry.rank,
            entry.source_object_sha256,
            rel,
            entry.canonical_payload_utf8.len()
        ));
    }
    fs::write(out_dir.join("payload-plan.tsv"), rows).expect("write payload plan");
}

#[derive(Clone)]
struct FixtureBackend {
    index_manifest_sha256: Sha256Digest,
    index_artifact_sha256: Sha256Digest,
    ranked: Vec<Sha256Digest>,
}

impl QualifiedRetrievalBackend for FixtureBackend {
    fn execute(
        &mut self,
        _request: &QualifiedRetrievalRequest,
    ) -> Result<BackendExecution, RetrievalError> {
        Ok(BackendExecution {
            retrieval: BackendRetrieval::SingleIndex(SingleIndexExecution {
                index_manifest_sha256: self.index_manifest_sha256.clone(),
                index_artifact_sha256: self.index_artifact_sha256.clone(),
                requested_k: self.ranked.len() as u32,
                ranked_source_object_digests: self.ranked.clone(),
                control: ControlBinding {
                    index_manifest_sha256: self.index_manifest_sha256.clone(),
                    control_transform: ControlTransform::None,
                    control_seed: None,
                    control_artifact_sha256: None,
                },
            }),
            retrieval_queries_used: 1,
            wall_time_ms_used: 1,
            normalized_compute_microunits_used: 1_000_000,
        })
    }
}

struct FixtureMaterializer {
    payloads: BTreeMap<Sha256Digest, Vec<u8>>,
    calls: usize,
}

impl CanonicalSourceMaterializer for FixtureMaterializer {
    fn materialize(
        &mut self,
        source_object_sha256: &Sha256Digest,
    ) -> Result<Vec<u8>, RetrievalError> {
        self.calls += 1;
        self.payloads.get(source_object_sha256).cloned().ok_or_else(|| {
            RetrievalError::Materialization(format!(
                "no fixture payload for {source_object_sha256}"
            ))
        })
    }
}

fn fixed_digest(hex_byte: u8) -> Sha256Digest {
    Sha256Digest::parse(format!("sha256:{:064x}", hex_byte)).unwrap()
}

fn build_request(c: &BTreeMap<String, String>) -> QualifiedRetrievalRequest {
    QualifiedRetrievalRequest {
        trace_id: "math-ret-runtime-001d-fixture-trace".into(),
        audit_id: "math-ret-runtime-001d-fixture-audit".into(),
        experiment_id: required(c, "experiment_id").to_string(),
        arm_id: "S".into(),
        experiment_seed: parse_u64(c, "experiment_seed"),
        query_id: "fixture-query-001".into(),
        query_source_object_sha256: fixed_digest(0xf0),
        graph: GraphIdentity {
            bundle_sha256: digest(c, "bundle_sha256"),
            graph_report_sha256: digest(c, "graph_report_sha256"),
            experiment_sha256: digest(c, "experiment_sha256"),
            retrieval_binding_sha256: digest(c, "retrieval_binding_sha256"),
            candidate_set_sha256: digest(c, "candidate_set_sha256"),
            candidate_count: parse_usize(c, "candidate_count"),
            context_packer_sha256: digest(c, "context_packer_sha256"),
            source_object_contract_sha256: digest(c, "source_object_contract_sha256"),
            source_fetch_policy_sha256: digest(c, "source_fetch_policy_sha256"),
            payload_serialization_sha256: digest(c, "payload_serialization_sha256"),
        },
        budget: RetrievalBudget {
            max_output_items: parse_usize(c, "max_output_items"),
            max_output_bytes: parse_usize(c, "max_output_bytes"),
            max_output_item_bytes: parse_usize(c, "max_output_item_bytes"),
            max_retrieval_queries: parse_u32(c, "max_retrieval_queries"),
            max_normalized_compute_microunits: parse_u64(
                c,
                "max_normalized_compute_microunits",
            ),
            max_wall_time_ms: parse_u64(c, "max_wall_time_ms"),
        },
    }
}

fn main() {
    let mut args = env::args_os();
    let _program = args.next();
    let config_path = PathBuf::from(
        args.next()
            .expect("usage: emit_guarded_fixture <config> <out-dir> [--inject-illegal]"),
    );
    let out_dir = PathBuf::from(
        args.next()
            .expect("usage: emit_guarded_fixture <config> <out-dir> [--inject-illegal]"),
    );
    let inject_illegal = match args.next() {
        None => false,
        Some(flag) if flag == "--inject-illegal" => true,
        Some(flag) => panic!("unexpected argument: {:?}", flag),
    };
    assert!(args.next().is_none(), "unexpected extra arguments");

    let c = load_config(&config_path);
    let candidate_sources = digest_csv(&c, "candidate_source_sha256s");
    let universe = FrozenCandidateUniverse::new(
        digest(&c, "candidate_set_sha256"),
        candidate_sources,
    )
    .expect("frozen candidate universe must construct");
    assert_eq!(
        universe.candidate_count(),
        parse_usize(&c, "candidate_count"),
        "candidate artifact/cardinality mismatch"
    );

    let mut ranked = vec![fixed_digest(0xa1), fixed_digest(0xa2), fixed_digest(0xa3)];
    if inject_illegal {
        ranked[1] = fixed_digest(0xee);
    }

    let inner = FixtureBackend {
        index_manifest_sha256: digest(&c, "index_manifest_sha256"),
        index_artifact_sha256: digest(&c, "index_artifact_sha256"),
        ranked: ranked.clone(),
    };
    let mut backend = MembershipGuardBackend::new(inner, universe);
    let mut materializer = FixtureMaterializer {
        payloads: BTreeMap::from([
            (fixed_digest(0xa1), b"alpha theorem context\n".to_vec()),
            (fixed_digest(0xa2), b"beta lemma context\n".to_vec()),
            (fixed_digest(0xa3), b"gamma definition context\n".to_vec()),
        ]),
        calls: 0,
    };
    let mut sink = InMemoryEvidenceSink::default();
    let request = build_request(&c);

    if inject_illegal {
        let result = RetrievalExecutor::execute(&mut backend, &mut materializer, &mut sink, request);
        match result {
            Err(RetrievalError::Candidate(message)) => {
                assert!(
                    message.contains("outside frozen candidate universe"),
                    "negative fixture failed for wrong candidate reason: {message}"
                );
            }
            Err(other) => panic!("negative fixture failed for wrong reason: {other}"),
            Ok(_) => panic!("out-of-universe candidate unexpectedly qualified"),
        }
        assert_eq!(materializer.calls, 0, "illegal candidate reached materializer");
        assert!(sink.committed().is_empty(), "illegal candidate committed evidence");
        assert!(!sink.has_staged_state(), "illegal candidate left staged evidence");
        assert!(
            !out_dir.exists(),
            "negative execution must not create a runtime evidence directory"
        );
        println!("guarded illegal-candidate canary: PASS");
        return;
    }

    let outcome = RetrievalExecutor::execute(&mut backend, &mut materializer, &mut sink, request)
        .expect("guarded typed runtime fixture must execute");
    assert_eq!(sink.committed().len(), 1, "evidence transaction must commit once");
    assert!(outcome.evidence_committed);
    assert_eq!(materializer.calls, ranked.len());

    fs::create_dir_all(&out_dir).expect("create guarded-adapter output directory");
    fs::write(out_dir.join("trace.json"), serialize_trace(&outcome.trace))
        .expect("write frozen trace JSON");
    write_payload_plan(&out_dir, &outcome.payload_audit);
}
