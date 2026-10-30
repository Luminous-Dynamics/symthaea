//! HARDSEC-007 evidence-packet validator.
//!
//! Validates whether every supported hardware-security claim is backed by
//! evidence of the correct kind, scope and status. It does not validate the
//! truth of the evidence, authenticate its producer, verify signatures, or
//! certify a design.
//!
//! Example:
//!   cargo run --example hardsec_evidence_packet_audit --features hardsec-evidence-audit -- \
//!     --packet /path/to/evidence-packet.json \
//!     --receipt /path/outside/packet/evidence-audit.json
//!
//! See docs/research/HARDSEC_007_ENTANGLED_RARE_TRIGGER_BENCHMARK.md for the
//! packet contract and claim ceilings.

use sha2::{Digest, Sha256};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const CLAIM_SCOPE: &str = "packet_schema_and_claim_evidence_consistency_only";

#[derive(Debug, Clone)]
struct Args {
    packet: PathBuf,
    receipt: PathBuf,
}

#[derive(Debug, Clone)]
struct Evidence {
    id: String,
    kind: String,
    status: String,
    subject_sha256: String,
    evidence_sha256: String,
    uri: String,
    tool: String,
    tool_version: String,
    scope: String,
    proof_scope: Option<String>,
    assumptions_sha256: Option<String>,
    witness_sha256: Option<String>,
    stimuli_count: Option<u64>,
    activation_count: Option<u64>,
    bound_cycles: Option<u64>,
}

#[derive(Debug, Clone)]
struct ClaimSummary {
    claim_type: String,
    status: String,
    evidence_count: usize,
    scope: String,
}

#[derive(Debug, Clone)]
struct AuditSummary {
    packet_id: String,
    packet_sha256: String,
    design_id: String,
    netlist_sha256: String,
    model_sha256: String,
    evidence_count: usize,
    evidence: Vec<Evidence>,
    claims: Vec<ClaimSummary>,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut packet = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--packet" => packet = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(Some(Args {
        packet: packet.ok_or("--packet is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
    }))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn object<'a>(value: &'a Value, field: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{field} must be a JSON object"))
}

fn required_string(value: &Value, field: &str) -> Result<String, String> {
    let result = value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))?;
    if result.trim().is_empty() || result.trim() != result {
        return Err(format!("{field} must be non-empty and have no surrounding whitespace"));
    }
    Ok(result.to_owned())
}

fn optional_string(value: &Value, field: &str) -> Result<Option<String>, String> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(result)) if !result.trim().is_empty() && result.trim() == result => {
            Ok(Some(result.clone()))
        }
        _ => Err(format!("{field} must be a non-empty string when present")),
    }
}

fn optional_u64(value: &Value, field: &str) -> Result<Option<u64>, String> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(number) => number
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{field} must be a non-negative integer when present")),
    }
}

fn required_probability(value: &Value, field: &str) -> Result<f64, String> {
    let number = value
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{field} must be numeric"))?;
    if !number.is_finite() || !(0.0..=1.0).contains(&number) {
        return Err(format!("{field} must be finite and in [0, 1]"));
    }
    Ok(number)
}

fn parse_evidence(value: &Value, expected_subject: &str) -> Result<Evidence, String> {
    let id = required_string(value, "evidence_id")?;
    let kind = required_string(value, "kind")?;
    let status = required_string(value, "status")?;
    let subject_sha256 = required_string(value, "subject_sha256")?.to_ascii_lowercase();
    let evidence_sha256 = required_string(value, "evidence_sha256")?.to_ascii_lowercase();
    let uri = required_string(value, "uri")?;

    const KINDS: &[&str] = &[
        "stimulus_trace",
        "cover_witness",
        "formal_proof",
        "equivalence_report",
        "localization_map",
        "domain_measurement",
        "precedent",
        "artifact_manifest",
        "reviewer_decision",
    ];
    if !KINDS.contains(&kind.as_str()) {
        return Err(format!("evidence {id}: unsupported kind {kind}"));
    }
    if !["pass", "fail", "unknown", "not_applicable"].contains(&status.as_str()) {
        return Err(format!("evidence {id}: unsupported status {status}"));
    }
    if !valid_sha256(&subject_sha256) || !valid_sha256(&evidence_sha256) {
        return Err(format!("evidence {id}: subject/evidence SHA-256 must be 64 hex characters"));
    }
    if subject_sha256 != expected_subject {
        return Err(format!(
            "evidence {id}: subject SHA-256 does not match packet netlist"
        ));
    }

    let tool = required_string(value, "tool")?;
    let tool_version = required_string(value, "tool_version")?;
    let scope = required_string(value, "scope")?;
    let proof_scope = optional_string(value, "proof_scope")?;
    let assumptions_sha256 = optional_string(value, "assumptions_sha256")?;
    let witness_sha256 = optional_string(value, "witness_sha256")?;
    let stimuli_count = optional_u64(value, "stimuli_count")?;
    let activation_count = optional_u64(value, "activation_count")?;
    let bound_cycles = optional_u64(value, "bound_cycles")?;

    for (name, digest) in [
        ("assumptions_sha256", assumptions_sha256.as_deref()),
        ("witness_sha256", witness_sha256.as_deref()),
    ] {
        if let Some(digest) = digest {
            if !valid_sha256(digest) {
                return Err(format!("evidence {id}: {name} must be a SHA-256 digest"));
            }
        }
    }

    if let Some(count) = activation_count {
        if stimuli_count.is_none() {
            return Err(format!(
                "evidence {id}: activation_count requires stimuli_count"
            ));
        }
        if count > stimuli_count.unwrap_or(0) {
            return Err(format!(
                "evidence {id}: activation_count cannot exceed stimuli_count"
            ));
        }
    }

    Ok(Evidence {
        id,
        kind,
        status,
        subject_sha256,
        evidence_sha256,
        uri,
        tool,
        tool_version,
        scope,
        proof_scope,
        assumptions_sha256,
        witness_sha256,
        stimuli_count,
        activation_count,
        bound_cycles,
    })
}

fn claim_evidence<'a>(
    claim_type: &str,
    evidence_ids: &[String],
    evidence: &'a BTreeMap<String, Evidence>,
) -> Result<Vec<&'a Evidence>, String> {
    let mut selected = Vec::new();
    let mut unique = BTreeSet::new();
    for id in evidence_ids {
        if !unique.insert(id) {
            return Err(format!("claim {claim_type}: duplicate evidence reference {id}"));
        }
        let item = evidence
            .get(id)
            .ok_or_else(|| format!("claim {claim_type}: unknown evidence_id {id}"))?;
        selected.push(item);
    }
    Ok(selected)
}

fn require_kind_status(
    claim_type: &str,
    selected: &[&Evidence],
    kind: &str,
) -> Result<(), String> {
    if !selected
        .iter()
        .any(|item| item.kind == kind && item.status == "pass")
    {
        return Err(format!(
            "claim {claim_type}: requires passing {kind} evidence"
        ));
    }
    Ok(())
}

fn validate_supported_claim(
    claim_type: &str,
    selected: &[&Evidence],
    detector_decision: &str,
    detector_score: f64,
    detector_threshold: f64,
) -> Result<(), String> {
    if selected.iter().any(|item| item.status != "pass") {
        return Err(format!(
            "claim {claim_type}: supported claim cannot cite failed/unknown evidence"
        ));
    }

    match claim_type {
        "model_suspects_trojan" => {
            if detector_decision != "trojan_suspected"
                || detector_score < detector_threshold
            {
                return Err(
                    "model_suspects_trojan requires decision=trojan_suspected and score >= threshold"
                        .to_owned(),
                );
            }
        }
        "localization_provided" => {
            require_kind_status(claim_type, selected, "localization_map")?;
        }
        "trigger_activated" => {
            let witness = selected.iter().any(|item| {
                ["stimulus_trace", "cover_witness"].contains(&item.kind.as_str())
                    && item.status == "pass"
                    && item.activation_count.unwrap_or(0) > 0
                    && item.witness_sha256.is_some()
            });
            if !witness {
                return Err(format!(
                    "claim {claim_type}: requires passing stimulus/cover evidence with activation_count > 0 and witness_sha256"
                ));
            }
        }
        "trigger_not_observed_under_budget" => {
            let trace = selected.iter().any(|item| {
                item.kind == "stimulus_trace"
                    && item.status == "pass"
                    && item.stimuli_count.unwrap_or(0) > 0
                    && item.activation_count == Some(0)
            });
            if !trace {
                return Err(format!(
                    "claim {claim_type}: requires a passing stimulus trace with stimuli_count > 0 and activation_count = 0"
                ));
            }
        }
        "bounded_reachability_not_found" => {
            let bounded = selected.iter().any(|item| {
                item.kind == "formal_proof"
                    && item.status == "pass"
                    && item.proof_scope.as_deref() == Some("bounded")
                    && item.bound_cycles.unwrap_or(0) > 0
            });
            if !bounded {
                return Err(format!(
                    "claim {claim_type}: requires passing bounded formal evidence and a positive bound_cycles"
                ));
            }
        }
        "trigger_unreachable_under_assumptions" => {
            let proof = selected.iter().any(|item| {
                item.kind == "formal_proof"
                    && item.status == "pass"
                    && item.proof_scope.as_deref() == Some("unbounded")
                    && item.assumptions_sha256.is_some()
            });
            if !proof {
                return Err(format!(
                    "claim {claim_type}: requires passing unbounded formal proof and assumptions_sha256"
                ));
            }
        }
        "functional_equivalence" => {
            let equivalence = selected.iter().any(|item| {
                item.kind == "equivalence_report"
                    && item.status == "pass"
                    && matches!(
                        item.proof_scope.as_deref(),
                        Some("bounded" | "unbounded" | "assumption_scoped")
                    )
                    && (item.proof_scope.as_deref() != Some("bounded")
                        || item.bound_cycles.unwrap_or(0) > 0)
            });
            if !equivalence {
                return Err(format!(
                    "claim {claim_type}: requires passing equivalence_report with explicit bounded/unbounded/assumption_scoped semantics"
                ));
            }
        }
        "source_integrity_checked" => {
            require_kind_status(claim_type, selected, "artifact_manifest")?;
        }
        "reviewer_confirmed" => {
            require_kind_status(claim_type, selected, "reviewer_decision")?;
        }
        "trojan_absent" | "design_is_secure" => {
            return Err(format!(
                "claim {claim_type} is not supported by this packet validator"
            ));
        }
        _ => return Err(format!("unsupported claim_type: {claim_type}")),
    }
    Ok(())
}

fn validate_packet(value: &Value, packet_sha256: String) -> Result<AuditSummary, String> {
    if value.get("schema_version").and_then(Value::as_str)
        != Some("hardsec-evidence-packet-v1")
    {
        return Err("schema_version must be hardsec-evidence-packet-v1".to_owned());
    }
    let packet_id = required_string(value, "packet_id")?;
    let _dataset_id = required_string(value, "dataset_id")?;
    let subject = object(value, "subject")?;
    let design_id = required_string(&Value::Object(subject.clone()), "design_id")?;
    let netlist_sha256 =
        required_string(&Value::Object(subject.clone()), "netlist_sha256")?.to_ascii_lowercase();
    if !valid_sha256(&netlist_sha256) {
        return Err("subject.netlist_sha256 must be a valid SHA-256 digest".to_owned());
    }

    let detector = object(value, "detector")?;
    let detector_value = Value::Object(detector.clone());
    let model_sha256 =
        required_string(&detector_value, "model_sha256")?.to_ascii_lowercase();
    let config_sha256 =
        required_string(&detector_value, "config_sha256")?.to_ascii_lowercase();
    if !valid_sha256(&model_sha256) || !valid_sha256(&config_sha256) {
        return Err("detector model/config SHA-256 is invalid".to_owned());
    }
    let _method = required_string(&detector_value, "method")?;
    let detector_score = required_probability(&detector_value, "score")?;
    let detector_threshold = required_probability(&detector_value, "threshold")?;
    let detector_decision = required_string(&detector_value, "decision")?;
    if !["trojan_suspected", "no_alert", "abstain"].contains(&detector_decision.as_str()) {
        return Err("detector.decision must be trojan_suspected, no_alert, or abstain".to_owned());
    }

    let evidence_values = value
        .get("evidence")
        .and_then(Value::as_array)
        .ok_or("evidence must be a JSON array")?;
    if evidence_values.is_empty() {
        return Err("evidence must contain at least one item".to_owned());
    }

    let mut evidence = BTreeMap::new();
    for item in evidence_values {
        let parsed = parse_evidence(item, &netlist_sha256)?;
        let id = parsed.id.clone();
        if evidence.insert(id.clone(), parsed).is_some() {
            return Err(format!("duplicate evidence_id {id}"));
        }
    }

    let claims_value = value
        .get("claims")
        .and_then(Value::as_array)
        .ok_or("claims must be a JSON array")?;
    if claims_value.is_empty() {
        return Err("claims must contain at least one item".to_owned());
    }

    let mut claims = Vec::new();
    for item in claims_value {
        let claim_type = required_string(item, "claim_type")?;
        let status = required_string(item, "status")?;
        let scope = required_string(item, "scope")?;
        if !["supported", "contradicted", "unknown", "not_evaluated"].contains(&status.as_str()) {
            return Err(format!("claim {claim_type}: invalid status {status}"));
        }
        let ids = item
            .get("evidence_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("claim {claim_type}: evidence_ids must be an array"))?
            .iter()
            .map(|entry| {
                entry
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("claim {claim_type}: evidence_ids entries must be strings"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let selected = claim_evidence(&claim_type, &ids, &evidence)?;

        match status.as_str() {
            "supported" => {
                if claim_type != "model_suspects_trojan" && selected.is_empty() {
                    return Err(format!(
                        "claim {claim_type}: supported claim requires evidence references"
                    ));
                }
                validate_supported_claim(
                    &claim_type,
                    &selected,
                    &detector_decision,
                    detector_score,
                    detector_threshold,
                )?;
            }
            "contradicted" => {
                if selected.is_empty() || !selected.iter().any(|entry| entry.status == "fail") {
                    return Err(format!(
                        "claim {claim_type}: contradicted status requires at least one failed evidence item"
                    ));
                }
            }
            "unknown" | "not_evaluated" => {}
            _ => unreachable!(),
        }

        claims.push(ClaimSummary {
            claim_type,
            status,
            evidence_count: ids.len(),
            scope,
        });
    }

    claims.sort_by(|left, right| {
        left.claim_type
            .cmp(&right.claim_type)
            .then(left.status.cmp(&right.status))
            .then(left.scope.cmp(&right.scope))
    });

    let evidence_items = evidence.values().cloned().collect::<Vec<_>>();
    Ok(AuditSummary {
        packet_id,
        packet_sha256,
        design_id,
        netlist_sha256,
        model_sha256,
        evidence_count: evidence_items.len(),
        evidence: evidence_items,
        claims,
    })
}

fn receipt_target(packet: &Path, receipt: &Path) -> Result<PathBuf, String> {
    let packet = fs::canonicalize(packet)
        .map_err(|error| format!("canonicalize packet {}: {error}", packet.display()))?;
    let packet_parent = packet
        .parent()
        .ok_or("packet has no parent directory")?
        .to_path_buf();
    let parent = receipt
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let filename = receipt
        .file_name()
        .ok_or("receipt path must include a filename")?;
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize receipt parent {}: {error}", parent.display()))?;
    let target = parent.join(filename);
    if target.starts_with(&packet_parent) {
        return Err("receipt must be outside the packet directory".to_owned());
    }
    Ok(target)
}

fn run(args: Args) -> Result<(), String> {
    let packet_meta = fs::symlink_metadata(&args.packet)
        .map_err(|error| format!("stat packet {}: {error}", args.packet.display()))?;
    if packet_meta.file_type().is_symlink() || !packet_meta.is_file() {
        return Err("packet must be a regular non-symlink file".to_owned());
    }
    let packet_bytes = fs::read(&args.packet)
        .map_err(|error| format!("read packet {}: {error}", args.packet.display()))?;
    let packet_sha256 = sha256_bytes(&packet_bytes);
    let value: Value = serde_json::from_slice(&packet_bytes)
        .map_err(|error| format!("parse evidence packet JSON: {error}"))?;
    let summary = validate_packet(&value, packet_sha256)?;
    let receipt_path = receipt_target(&args.packet, &args.receipt)?;

    let claims = summary
        .claims
        .iter()
        .map(|claim| {
            json!({
                "claim_type": claim.claim_type,
                "status": claim.status,
                "evidence_count": claim.evidence_count,
                "scope": claim.scope,
            })
        })
        .collect::<Vec<_>>();
    let receipt = json!({
        "schema_version": "hardsec-evidence-audit-receipt-v1",
        "result": "pass",
        "claim_scope": CLAIM_SCOPE,
        "packet_id": summary.packet_id,
        "packet_sha256": summary.packet_sha256,
        "design_id": summary.design_id,
        "netlist_sha256": summary.netlist_sha256,
        "model_sha256": summary.model_sha256,
        "evidence_count": summary.evidence_count,
        "evidence": summary.evidence.iter().map(|item| json!({
            "evidence_id": item.id,
            "kind": item.kind,
            "status": item.status,
            "subject_sha256": item.subject_sha256,
            "evidence_sha256": item.evidence_sha256,
            "uri": item.uri,
            "tool": item.tool,
            "tool_version": item.tool_version,
            "scope": item.scope,
            "proof_scope": item.proof_scope,
            "assumptions_sha256": item.assumptions_sha256,
            "witness_sha256": item.witness_sha256,
            "stimuli_count": item.stimuli_count,
            "activation_count": item.activation_count,
            "bound_cycles": item.bound_cycles
        })).collect::<Vec<_>>(),
        "claims": claims,
        "signature": null,
        "security_verdict": null,
        "notes": [
            "This validates packet schema and internal claim/evidence consistency only.",
            "Evidence bytes, tool execution, signatures, provenance authenticity and claim truth were not independently verified.",
            "A supported model_suspects_trojan claim is a model alert, not proof that a Trojan exists.",
            "trigger_not_observed_under_budget is not evidence of unreachability.",
            "bounded_reachability_not_found is limited to its encoded bound and assumptions."
        ]
    });

    let serialized = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("serialize audit receipt: {error}"))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)
        .map_err(|error| format!("create new receipt {}: {error}", receipt_path.display()))?;
    let write_result = output
        .write_all(&serialized)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all());
    drop(output);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&receipt_path);
        return Err(format!("write receipt {}: {error}", receipt_path.display()));
    }

    println!("HARDSEC-007 evidence packet audit: PASS");
    println!("design: {}", summary.design_id);
    println!("evidence items: {}", summary.evidence_count);
    println!("claims: {}", summary.claims.len());
    println!("receipt: {}", receipt_path.display());
    println!("claim scope: {}", CLAIM_SCOPE);
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("HARDSEC-007 evidence packet audit failed: {error}");
                ExitCode::FAILURE
            }
        },
        Ok(None) => {
            println!(
                "usage: hardsec_evidence_packet_audit --packet FILE --receipt NEW_FILE"
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);
    const NETLIST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const MODEL: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const CONFIG: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const EVIDENCE: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const WITNESS: &str =
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const ASSUMPTIONS: &str =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn base_packet() -> Value {
        json!({
            "schema_version": "hardsec-evidence-packet-v1",
            "packet_id": "packet-001",
            "dataset_id": "fixture-dataset",
            "subject": {
                "design_id": "design-001",
                "netlist_sha256": NETLIST
            },
            "detector": {
                "method": "fixture-model",
                "model_sha256": MODEL,
                "config_sha256": CONFIG,
                "score": 0.91,
                "threshold": 0.80,
                "decision": "trojan_suspected"
            },
            "evidence": [{
                "evidence_id": "stim-1",
                "kind": "stimulus_trace",
                "subject_sha256": NETLIST,
                "evidence_sha256": EVIDENCE,
                "uri": "fixture/evidence/stimulus-trace.json",
                "status": "pass",
                "tool": "fixture-simulator",
                "tool_version": "1.0",
                "scope": "1000 independent vectors",
                "stimuli_count": 1000,
                "activation_count": 1,
                "witness_sha256": WITNESS
            }],
            "claims": [{
                "claim_type": "trigger_activated",
                "status": "supported",
                "scope": "observed under 1000-vector trace",
                "evidence_ids": ["stim-1"]
            }]
        })
    }

    struct Fixture {
        base: PathBuf,
        packet: PathBuf,
        receipt: PathBuf,
    }

    impl Fixture {
        fn new(value: &Value) -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let base = env::temp_dir().join(format!(
                "hardsec-evidence-packet-{}-{id}",
                std::process::id()
            ));
            let packet_dir = base.join("input");
            let receipt_dir = base.join("output");
            fs::create_dir_all(&packet_dir).unwrap();
            fs::create_dir_all(&receipt_dir).unwrap();
            let packet = packet_dir.join("packet.json");
            let receipt = receipt_dir.join("receipt.json");
            fs::write(&packet, serde_json::to_vec_pretty(value).unwrap()).unwrap();
            Self {
                base,
                packet,
                receipt,
            }
        }

        fn args(&self) -> Args {
            Args {
                packet: self.packet.clone(),
                receipt: self.receipt.clone(),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    fn add_evidence(packet: &mut Value, item: Value) {
        packet["evidence"].as_array_mut().unwrap().push(item);
    }

    fn evidence(kind: &str, status: &str) -> Value {
        json!({
            "evidence_id": format!("{kind}-1"),
            "kind": kind,
            "subject_sha256": NETLIST,
            "evidence_sha256": EVIDENCE,
            "uri": format!("fixture/evidence/{kind}.json"),
            "status": status,
            "tool": "fixture-tool",
            "tool_version": "1.0",
            "scope": "fixture scope"
        })
    }

    fn set_claim(packet: &mut Value, claim_type: &str, status: &str, ids: Vec<&str>) {
        packet["claims"] = json!([{
            "claim_type": claim_type,
            "status": status,
            "scope": "test scope",
            "evidence_ids": ids
        }]);
    }

    #[test]
    fn accepts_supported_trigger_activation_with_hashed_witness() {
        let packet = base_packet();
        let fixture = Fixture::new(&packet);
        run(fixture.args()).unwrap();
        let receipt: Value = serde_json::from_slice(&fs::read(&fixture.receipt).unwrap()).unwrap();
        assert_eq!(receipt["result"], "pass");
        assert_eq!(receipt["claim_scope"], CLAIM_SCOPE);
    }

    #[test]
    fn accepts_trigger_not_observed_under_finite_budget() {
        let mut packet = base_packet();
        packet["evidence"][0]["activation_count"] = json!(0);
        packet["evidence"][0]["witness_sha256"] = Value::Null;
        set_claim(
            &mut packet,
            "trigger_not_observed_under_budget",
            "supported",
            vec!["stim-1"],
        );
        let fixture = Fixture::new(&packet);
        run(fixture.args()).unwrap();
    }

    #[test]
    fn rejects_claim_that_promotes_bounded_search_to_unreachability() {
        let mut packet = base_packet();
        let mut formal = evidence("formal_proof", "pass");
        formal["proof_scope"] = json!("bounded");
        formal["bound_cycles"] = json!(20);
        add_evidence(&mut packet, formal);
        set_claim(
            &mut packet,
            "trigger_unreachable_under_assumptions",
            "supported",
            vec!["formal_proof-1"],
        );
        let error = validate_packet(&packet, "0".repeat(64)).unwrap_err();
        assert!(error.contains("unbounded formal proof"));
    }

    #[test]
    fn accepts_bounded_no_witness_only_as_bounded_claim() {
        let mut packet = base_packet();
        let mut formal = evidence("formal_proof", "pass");
        formal["proof_scope"] = json!("bounded");
        formal["bound_cycles"] = json!(20);
        add_evidence(&mut packet, formal);
        set_claim(
            &mut packet,
            "bounded_reachability_not_found",
            "supported",
            vec!["formal_proof-1"],
        );
        validate_packet(&packet, "0".repeat(64)).unwrap();
    }

    #[test]
    fn accepts_unbounded_unreachable_claim_only_with_assumption_digest() {
        let mut packet = base_packet();
        let mut formal = evidence("formal_proof", "pass");
        formal["proof_scope"] = json!("unbounded");
        formal["assumptions_sha256"] = json!(ASSUMPTIONS);
        add_evidence(&mut packet, formal);
        set_claim(
            &mut packet,
            "trigger_unreachable_under_assumptions",
            "supported",
            vec!["formal_proof-1"],
        );
        validate_packet(&packet, "0".repeat(64)).unwrap();
    }

    #[test]
    fn rejects_unbounded_claim_without_assumptions_digest() {
        let mut packet = base_packet();
        let mut formal = evidence("formal_proof", "pass");
        formal["proof_scope"] = json!("unbounded");
        add_evidence(&mut packet, formal);
        set_claim(
            &mut packet,
            "trigger_unreachable_under_assumptions",
            "supported",
            vec!["formal_proof-1"],
        );
        assert!(validate_packet(&packet, "0".repeat(64))
            .unwrap_err()
            .contains("assumptions_sha256"));
    }

    #[test]
    fn requires_passing_equivalence_report() {
        let mut packet = base_packet();
        let mut eq = evidence("equivalence_report", "unknown");
        eq["proof_scope"] = json!("unbounded");
        add_evidence(&mut packet, eq);
        set_claim(
            &mut packet,
            "functional_equivalence",
            "supported",
            vec!["equivalence_report-1"],
        );
        assert!(validate_packet(&packet, "0".repeat(64))
            .unwrap_err()
            .contains("supported claim cannot cite failed/unknown"));
    }

    #[test]
    fn rejects_a_generic_trojan_absent_claim() {
        let mut packet = base_packet();
        set_claim(&mut packet, "trojan_absent", "supported", vec!["stim-1"]);
        assert!(validate_packet(&packet, "0".repeat(64))
            .unwrap_err()
            .contains("not supported by this packet validator"));
    }

    #[test]
    fn requires_evidence_to_bind_the_exact_netlist() {
        let mut packet = base_packet();
        packet["evidence"][0]["subject_sha256"] = json!(MODEL);
        assert!(validate_packet(&packet, "0".repeat(64))
            .unwrap_err()
            .contains("does not match packet netlist"));
    }

    #[test]
    fn rejects_unknown_evidence_references() {
        let mut packet = base_packet();
        set_claim(
            &mut packet,
            "trigger_activated",
            "supported",
            vec!["missing-evidence"],
        );
        assert!(validate_packet(&packet, "0".repeat(64))
            .unwrap_err()
            .contains("unknown evidence_id"));
    }

    #[test]
    fn rejects_claim_without_activation_witness() {
        let mut packet = base_packet();
        packet["evidence"][0]["witness_sha256"] = Value::Null;
        set_claim(&mut packet, "trigger_activated", "supported", vec!["stim-1"]);
        assert!(validate_packet(&packet, "0".repeat(64))
            .unwrap_err()
            .contains("witness_sha256"));
    }

    #[test]
    fn requires_failed_evidence_for_contradicted_claim() {
        let mut packet = base_packet();
        set_claim(&mut packet, "trigger_activated", "contradicted", vec!["stim-1"]);
        assert!(validate_packet(&packet, "0".repeat(64))
            .unwrap_err()
            .contains("failed evidence"));
    }

    #[test]
    fn refuses_to_overwrite_receipt() {
        let packet = base_packet();
        let fixture = Fixture::new(&packet);
        fs::write(&fixture.receipt, b"preserve").unwrap();
        assert!(run(fixture.args()).unwrap_err().contains("File exists"));
    }

    #[test]
    fn receipt_must_be_outside_packet_directory() {
        let packet = base_packet();
        let mut fixture = Fixture::new(&packet);
        fixture.receipt = fixture.packet.parent().unwrap().join("receipt.json");
        assert!(run(fixture.args())
            .unwrap_err()
            .contains("outside the packet directory"));
    }
}
