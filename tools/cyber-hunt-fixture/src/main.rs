use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

mod evaluation;
mod query;

const GENERATOR_VERSION: &str = "cyber-hunt-fixture-v1";
const EVENT_SCHEMA_VERSION: &str = "cyber-hunt-event-v1";
const ORACLE_VERSION: &str = "cyber-hunt-oracle-v1";
const MANIFEST_VERSION: &str = "cyber-hunt-manifest-v1";
const SOLVER_MANIFEST_VERSION: &str = "cyber-hunt-solver-manifest-v1";
const DEFAULT_RECORDS: usize = 10_000;
const DEFAULT_SEED: u64 = 42;
const MIN_RECORDS: usize = 500;
const MAX_RECORDS: usize = 250_000;
const BASE_TIME_MS: i64 = 1_760_000_000_000;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Event {
    schema_version: String,
    record_id: String,
    source_event_id: String,
    delivery_id: String,
    event_time_ms: i64,
    ingest_time_ms: i64,
    host_id: String,
    principal_id: String,
    source_address: String,
    destination: String,
    bytes_sent: u64,
    device_posture: String,
    source: String,
    event_type: String,
    result: String,
    summary: String,
    change_ticket: Option<String>,
    integrity_state: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TruthLabel {
    record_id: String,
    security_label: String,
    data_quality: String,
    campaign_id: Option<String>,
    phase: Option<String>,
    rationale: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CampaignTruth {
    campaign_id: String,
    disposition: String,
    event_record_ids: Vec<String>,
    claim_ceiling: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Oracle {
    schema_version: String,
    generator_version: String,
    seed: u64,
    record_count: usize,
    labels: Vec<TruthLabel>,
    campaigns: Vec<CampaignTruth>,
    non_claims: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    sha256: String,
    utf8_bytes: usize,
    record_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    generator_version: String,
    source_commit: String,
    synthetic_only: bool,
    seed: u64,
    requested_record_count: usize,
    actual_record_count: usize,
    malicious_campaign_count: usize,
    benign_change_decoy_count: usize,
    duplicate_delivery_count: usize,
    source_identity_conflict_count: usize,
    solver_file: String,
    oracle_file: String,
    files: BTreeMap<String, FileIdentity>,
    qualification_state: String,
    claim_ceiling: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SolverManifest {
    schema_version: String,
    generator_version: String,
    source_commit: String,
    source_commit_pinned: bool,
    synthetic_only: bool,
    seed: u64,
    record_count: usize,
    event_file: String,
    event_sha256: String,
    event_utf8_bytes: usize,
    qualification_state: String,
    claim_ceiling: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Dataset {
    events: Vec<Event>,
    oracle: Oracle,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RenderedDataset {
    events_bytes: Vec<u8>,
    solver_manifest_bytes: Vec<u8>,
    oracle_bytes: Vec<u8>,
    manifest: Manifest,
    solver_manifest: SolverManifest,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RunReceipt {
    manifest: Manifest,
    solver_manifest: SolverManifest,
    verification: VerificationReport,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct VerificationReport {
    disposition: String,
    source_commit: String,
    source_commit_pinned: bool,
    seed: u64,
    record_count: usize,
    malicious_event_count: usize,
    benign_event_count: usize,
    duplicate_delivery_count: usize,
    source_identity_conflict_count: usize,
    event_sha256: String,
    oracle_sha256: String,
    non_claims: Vec<String>,
}

#[derive(Clone, Debug)]
struct Config {
    records: usize,
    seed: u64,
    out_dir: PathBuf,
    source_commit: String,
}

#[derive(Clone, Debug)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*: fixed, documented, non-cryptographic reproducibility PRNG.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, upper_exclusive: u64) -> u64 {
        debug_assert!(upper_exclusive > 0);
        self.next_u64() % upper_exclusive
    }

    fn signed_jitter(&mut self, magnitude_ms: i64) -> i64 {
        let width = (magnitude_ms as u64).saturating_mul(2).saturating_add(1);
        self.below(width) as i64 - magnitude_ms
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn unique_values<T: Ord>(values: &[T]) -> bool {
    values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

fn is_full_git_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn event_id(index: usize) -> String {
    format!("evt-{index:08}")
}

fn source_event_id(index: usize) -> String {
    format!("src-{index:08}")
}

fn delivery_id(index: usize) -> String {
    format!("delivery-{index:08}")
}

fn normal_event(index: usize, rng: &mut DeterministicRng, duration_ms: i64) -> Event {
    const SOURCES: [&str; 7] = [
        "identity",
        "endpoint",
        "dns",
        "proxy",
        "cloud-audit",
        "application",
        "file-audit",
    ];
    const KINDS: [(&str, &str, &str); 9] = [
        ("auth.success", "success", "Authentication succeeded for a synthetic account."),
        ("auth.failure", "denied", "Authentication attempt was denied by the identity service."),
        ("process.start", "observed", "A routine process start was recorded by endpoint telemetry."),
        ("dns.query", "observed", "A synthetic workload queried an internal service name."),
        ("network.connection", "allowed", "An allowed connection was recorded by the proxy."),
        ("file.read", "observed", "A workload read a file within its declared working set."),
        ("admin.change", "accepted", "A low-risk administrative change was recorded."),
        ("service.deploy", "success", "A routine service deployment completed."),
        ("backup.complete", "success", "A scheduled backup task completed successfully."),
    ];
    let kind = KINDS[rng.below(KINDS.len() as u64) as usize];
    let event_time_ms =
        BASE_TIME_MS + rng.below(duration_ms.max(1) as u64) as i64;
    let source_address = format!("192.0.2.{}", rng.below(254) + 1);
    let destination = if kind.0 == "network.transfer" {
        format!("backup-{:03}.storage.example.test", rng.below(64))
    } else {
        format!("service-{:03}.internal.example.test", rng.below(64))
    };
    let bytes_sent = match kind.0 {
        "network.transfer" => rng.below(500_000),
        "network.connection" => rng.below(50_000),
        _ => 0,
    };
    let device_posture = ["managed", "unmanaged", "unknown"]
        [rng.below(3) as usize]
        .to_string();
    Event {
        schema_version: EVENT_SCHEMA_VERSION.to_string(),
        record_id: event_id(index),
        source_event_id: source_event_id(index),
        delivery_id: delivery_id(index),
        event_time_ms,
        ingest_time_ms: event_time_ms + rng.signed_jitter(300_000),
        host_id: format!("host-{:03}", rng.below(128)),
        principal_id: format!("principal-{:03}", rng.below(256)),
        source_address,
        destination,
        bytes_sent,
        device_posture,
        source: SOURCES[rng.below(SOURCES.len() as u64) as usize].to_string(),
        event_type: kind.0.to_string(),
        result: kind.1.to_string(),
        summary: kind.2.to_string(),
        change_ticket: None,
        integrity_state: "observed".to_string(),
    }
}

fn attack_phase(phase: usize) -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
    // These are synthetic defensive telemetry observations, not executable attack steps.
    const PHASES: [(&str, &str, &str, &str, &str); 8] = [
        ("identity", "auth.failure", "denied", "Repeated authentication failures were recorded for a service identity.", "authentication-anomaly"),
        ("identity", "auth.failure", "denied", "A second authentication denial occurred in the same synthetic investigation window.", "authentication-anomaly"),
        ("identity", "auth.success", "success", "Authentication succeeded after repeated failures; device posture is unavailable.", "suspicious-authentication"),
        ("identity", "identity.role_grant", "accepted", "A role grant was recorded; compare its timing and identity context with the change record.", "privilege-change"),
        ("endpoint", "endpoint.remote_session", "observed", "A remote session was recorded; corroborating authorization context is not joined.", "lateral-session"),
        ("file-audit", "file.archive_create", "observed", "An archive was created following a burst of file reads.", "collection-anomaly"),
        ("proxy", "network.transfer", "allowed", "An outbound transfer exceeded the synthetic service baseline.", "egress-anomaly"),
        ("proxy", "network.connection", "closed", "The connection ended; destination inventory corroboration is missing.", "incomplete-corroboration"),
    ];
    PHASES[phase]
}

fn attack_event(
    index: usize,
    campaign: usize,
    phase: usize,
    rng: &mut DeterministicRng,
) -> Event {
    let (source, event_type, result, summary, _) = attack_phase(phase);
    let campaign_time = BASE_TIME_MS + (campaign as i64 + 1) * 30_000_000;
    let event_time_ms = campaign_time + phase as i64 * 60_000 + rng.signed_jitter(3_000);
    let host = if phase < 4 {
        format!("host-{:03}", campaign % 128)
    } else {
        format!("host-{:03}", (campaign * 7 + 41) % 128)
    };
    let principal = format!("principal-{:03}", (campaign * 13 + 17) % 256);
    let opaque_source_id = format!("src-{:016x}", rng.next_u64());
    let source_address = if phase < 4 {
        format!("203.0.113.{}", campaign % 254 + 1)
    } else {
        format!("198.51.100.{}", (campaign * 7) % 254 + 1)
    };
    let destination = if phase == 6 {
        "bulk-transfer-edge.example.test".to_string()
    } else if phase >= 4 {
        format!("service-{:03}.internal.example.test", (campaign * 11) % 64)
    } else {
        "identity.example.test".to_string()
    };
    let bytes_sent = if phase == 6 {
        10_000_000 + campaign as u64 * 1_000
    } else if phase == 5 {
        1_000 + campaign as u64
    } else {
        0
    };
    let device_posture = if phase == 2 { "unknown" } else { "not-applicable" }.to_string();
    Event {
        schema_version: EVENT_SCHEMA_VERSION.to_string(),
        record_id: event_id(index),
        source_event_id: opaque_source_id,
        delivery_id: delivery_id(index),
        event_time_ms,
        ingest_time_ms: event_time_ms + rng.signed_jitter(300_000),
        host_id: host,
        principal_id: principal,
        source_address,
        destination,
        bytes_sent,
        device_posture,
        source: source.to_string(),
        event_type: event_type.to_string(),
        result: result.to_string(),
        summary: summary.to_string(),
        change_ticket: None,
        integrity_state: "observed".to_string(),
    }
}

fn benign_change_event(index: usize, campaign: usize, phase: usize, rng: &mut DeterministicRng) -> Event {
    const PHASES: [(&str, &str, &str, &str); 4] = [
        ("change-system", "admin.change", "approved", "Approved synthetic change window opened."),
        ("endpoint", "service.deploy", "success", "The declared service deployment completed within the approved change window."),
        ("file-audit", "file.archive_create", "observed", "A scheduled maintenance archive was created under the approved change ticket."),
        ("proxy", "network.transfer", "allowed", "A documented backup transfer matched the approved change ticket and destination."),
    ];
    let (source, event_type, result, summary) = PHASES[phase];
    let campaign_time = BASE_TIME_MS + (campaign as i64 + 1) * 30_000_000 + 2_000_000;
    let event_time_ms = campaign_time + phase as i64 * 60_000 + rng.signed_jitter(3_000);
    let opaque_source_id = format!("src-{:016x}", rng.next_u64());
    let source_address = format!("192.0.2.{}", (campaign * 17 + 11) % 254 + 1);
    let destination = if phase == 3 {
        format!("backup-{:03}.storage.example.test", campaign % 64)
    } else {
        format!("service-{:03}.internal.example.test", (campaign * 11) % 64)
    };
    let bytes_sent = if phase == 3 {
        9_500_000 + campaign as u64 * 1_000
    } else {
        0
    };
    Event {
        schema_version: EVENT_SCHEMA_VERSION.to_string(),
        record_id: event_id(index),
        source_event_id: opaque_source_id,
        delivery_id: delivery_id(index),
        event_time_ms,
        ingest_time_ms: event_time_ms + rng.signed_jitter(300_000),
        host_id: format!("host-{:03}", 60 + campaign % 30),
        principal_id: format!("principal-{:03}", (campaign * 19 + 91) % 256),
        source_address,
        destination,
        bytes_sent,
        device_posture: "managed".to_string(),
        source: source.to_string(),
        event_type: event_type.to_string(),
        result: result.to_string(),
        summary: summary.to_string(),
        change_ticket: Some(format!("change-approved-{campaign:03}")),
        integrity_state: "observed".to_string(),
    }
}

fn label_for(
    index: usize,
    security_label: &str,
    data_quality: &str,
    campaign_id: Option<String>,
    phase: Option<String>,
    rationale: &str,
) -> TruthLabel {
    TruthLabel {
        record_id: event_id(index),
        security_label: security_label.to_string(),
        data_quality: data_quality.to_string(),
        campaign_id,
        phase,
        rationale: rationale.to_string(),
    }
}

fn valid_injection_slot(index: usize, protected: &BTreeSet<usize>, records: usize) -> bool {
    index > 0 && index < records && !protected.contains(&index) && !protected.contains(&(index - 1))
}

fn build_dataset(records: usize, seed: u64) -> Result<Dataset, String> {
    if !(MIN_RECORDS..=MAX_RECORDS).contains(&records) {
        return Err(format!(
            "records must be within {MIN_RECORDS}..={MAX_RECORDS}; received {records}"
        ));
    }

    let campaign_count = ((records / 5_000).max(2)).min(20);
    let duration_ms = (records as i64).saturating_mul(90_000).max(1);
    let mut rng = DeterministicRng::new(seed);
    let mut events: Vec<Event> = (0..records)
        .map(|i| normal_event(i, &mut rng, duration_ms))
        .collect();
    let mut labels: Vec<TruthLabel> = (0..records)
        .map(|i| label_for(
            i,
            "benign",
            "normal",
            None,
            None,
            "Synthetic background record; no planted malicious ground-truth label.",
        ))
        .collect();

    let mut protected = BTreeSet::<usize>::new();
    let mut campaigns = Vec::new();

    for campaign in 0..campaign_count {
        let mut start = ((campaign + 1) * records) / (campaign_count + 1);
        while (start..start + 8).any(|idx| idx >= records || protected.contains(&idx)) {
            start += 1;
            if start + 24 >= records {
                return Err("unable to place non-overlapping synthetic campaign".to_string());
            }
        }
        let campaign_id = format!("hunt-campaign-{campaign:03}");
        let mut attack_records = Vec::new();
        for phase in 0..8 {
            let index = start + phase;
            protected.insert(index);
            let (_, _, _, _, phase_name) = attack_phase(phase);
            events[index] = attack_event(index, campaign, phase, &mut rng);
            labels[index] = label_for(
                index,
                "malicious",
                "normal",
                Some(campaign_id.clone()),
                Some(phase_name),
                "Planted synthetic campaign phase; label is stored only in the evaluator oracle.",
            );
            attack_records.push(event_id(index));
        }
        campaigns.push(CampaignTruth {
            campaign_id: campaign_id.clone(),
            disposition: "malicious".to_string(),
            event_record_ids: attack_records,
            claim_ceiling: "Synthetic planted sequence only; does not prove a real intrusion or a particular actor.".to_string(),
        });

        // Four benign administration events mimic some event types in the suspicious chain.
        let decoy_start = start + 16;
        for phase in 0..4 {
            let index = decoy_start + phase;
            if index >= records || protected.contains(&index) {
                return Err("unable to place benign change decoy without overlap".to_string());
            }
            protected.insert(index);
            events[index] = benign_change_event(index, campaign, phase, &mut rng);
            labels[index] = label_for(
                index,
                "benign",
                "normal",
                Some(format!("benign-change-{campaign:03}")),
                Some(format!("approved-change-phase-{phase:02}")),
                "Planted benign administrative sequence with an explicit synthetic change-ticket reference.",
            );
        }
        campaigns.push(CampaignTruth {
            campaign_id: format!("benign-change-{campaign:03}"),
            disposition: "benign_decoy".to_string(),
            event_record_ids: (0..4).map(|phase| event_id(decoy_start + phase)).collect(),
            claim_ceiling: "Approved-change decoy in this synthetic fixture only.".to_string(),
        });
    }

    let mut duplicate_delivery_count = 0usize;
    let mut source_identity_conflict_count = 0usize;
    for index in 1..records {
        if !valid_injection_slot(index, &protected, records) {
            continue;
        }
        if index % 997 == 0 {
            let mut duplicate = events[index - 1].clone();
            duplicate.record_id = event_id(index);
            duplicate.delivery_id = delivery_id(index);
            duplicate.ingest_time_ms += 1_000;
            duplicate.integrity_state = "duplicate_delivery".to_string();
            events[index] = duplicate;
            let previous_security_label = labels[index - 1].security_label.clone();
            labels[index] = label_for(
                index,
                previous_security_label.as_str(),
                "duplicate_delivery",
                None,
                None,
                "Synthetic redelivery repeats the previous source event identity and payload; it is not independent corroboration.",
            );
            duplicate_delivery_count += 1;
        } else if index >= records / 2 && (index - records / 2) % 4_999 == 0 {
            let previous_source_id = events[index - 1].source_event_id.clone();
            let original_summary = events[index].summary.clone();
            events[index].source_event_id = previous_source_id;
            events[index].summary = format!(
                "Conflicting payload for a reused source event identity. Original synthetic summary: {}",
                original_summary
            );
            events[index].integrity_state = "source_id_conflict".to_string();
            labels[index] = label_for(
                index,
                "benign",
                "source_id_conflict",
                None,
                None,
                "Synthetic identity/payload conflict requiring source reconciliation; this quality label alone is not a malicious-activity label.",
            );
            source_identity_conflict_count += 1;
        }
    }

    // Stream order follows ingest time, not event time; preserve IDs so evidence can be reconciled.
    events.sort_by(|a, b| a.ingest_time_ms.cmp(&b.ingest_time_ms).then(a.record_id.cmp(&b.record_id)));

    // Sorting the oracle by record ID makes output stable and keeps ground truth out of the event records.
    labels.sort_by(|a, b| a.record_id.cmp(&b.record_id));
    Ok(Dataset {
        events,
        oracle: Oracle {
            schema_version: ORACLE_VERSION.to_string(),
            generator_version: GENERATOR_VERSION.to_string(),
            seed,
            record_count: records,
            labels,
            campaigns,
            non_claims: vec![
                "All records are synthetic. They are not logs from real organizations or devices.".to_string(),
                "A planted malicious label establishes only the simulator's assigned ground truth.".to_string(),
                "The fixture does not validate an operational detector, incident attribution, or production readiness.".to_string(),
                "The public oracle is unsuitable as a held-out capability benchmark.".to_string(),
            ],
        },
    })
}

fn render_jsonl<T: Serialize>(records: &[T]) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    for record in records {
        serde_json::to_writer(&mut output, record)
            .map_err(|e| format!("failed to serialize JSONL record: {e}"))?;
        output.push(b'\n');
    }
    Ok(output)
}

fn verify_payloads(
    events_bytes: &[u8],
    solver_manifest_bytes: &[u8],
    oracle_bytes: &[u8],
    manifest: &Manifest,
) -> Result<VerificationReport, String> {
    if manifest.schema_version != MANIFEST_VERSION
        || manifest.generator_version != GENERATOR_VERSION
        || !manifest.synthetic_only
        || manifest.qualification_state != "UNSCORED"
        || manifest.claim_ceiling.trim().is_empty()
        || manifest.source_commit.trim().is_empty()
        || manifest.solver_file != "solver/events.jsonl"
        || manifest.oracle_file != "evaluator/oracle.json"
    {
        return Err("manifest version, source identity, synthetic flag, or claim ceiling is invalid".to_string());
    }
    let expected_files: BTreeSet<String> = [
        "solver/events.jsonl".to_string(),
        "solver/manifest.json".to_string(),
        "evaluator/oracle.json".to_string(),
    ]
    .into_iter()
    .collect();
    let actual_files: BTreeSet<String> = manifest.files.keys().cloned().collect();
    if actual_files != expected_files {
        return Err("manifest file set must contain exactly solver events, solver manifest, and evaluator oracle artifacts".to_string());
    }

    let event_identity = &manifest.files["solver/events.jsonl"];
    if event_identity.sha256 != sha256(events_bytes)
        || event_identity.utf8_bytes != events_bytes.len()
    {
        return Err("solver event bytes do not match the manifest SHA-256/length".to_string());
    }
    let solver_manifest_identity = &manifest.files["solver/manifest.json"];
    if solver_manifest_identity.sha256 != sha256(solver_manifest_bytes)
        || solver_manifest_identity.utf8_bytes != solver_manifest_bytes.len()
        || solver_manifest_identity.record_count != 1
    {
        return Err("solver manifest bytes do not match the operator manifest SHA-256/length".to_string());
    }
    let solver_manifest: SolverManifest = serde_json::from_slice(solver_manifest_bytes)
        .map_err(|e| format!("cannot parse solver-only manifest: {e}"))?;
    if solver_manifest.schema_version != SOLVER_MANIFEST_VERSION
        || solver_manifest.generator_version != GENERATOR_VERSION
        || solver_manifest.source_commit != manifest.source_commit
        || solver_manifest.source_commit_pinned != is_full_git_sha(&solver_manifest.source_commit)
        || !solver_manifest.synthetic_only
        || solver_manifest.seed != manifest.seed
        || solver_manifest.record_count != manifest.actual_record_count
        || solver_manifest.event_file != "events.jsonl"
        || solver_manifest.event_sha256 != event_identity.sha256
        || solver_manifest.event_utf8_bytes != event_identity.utf8_bytes
        || solver_manifest.qualification_state != "UNSCORED"
        || solver_manifest.claim_ceiling.trim().is_empty()
    {
        return Err("solver manifest does not reconcile with the operator manifest and event stream".to_string());
    }
    let oracle_identity = &manifest.files["evaluator/oracle.json"];
    if oracle_identity.sha256 != sha256(oracle_bytes)
        || oracle_identity.utf8_bytes != oracle_bytes.len()
    {
        return Err("evaluator oracle bytes do not match the manifest SHA-256/length".to_string());
    }

    let event_text = std::str::from_utf8(events_bytes)
        .map_err(|e| format!("event stream is not UTF-8: {e}"))?;
    if !event_text.ends_with('\n') {
        return Err("event JSONL must end with LF".to_string());
    }
    let mut events = Vec::<Event>::new();
    for (index, line) in event_text.split_terminator('\n').enumerate() {
        if line.trim().is_empty() {
            return Err(format!("event JSONL contains blank row at line {}", index + 1));
        }
        let event: Event = serde_json::from_str(line)
            .map_err(|e| format!("event JSONL row {} is invalid: {e}", index + 1))?;
        events.push(event);
    }
    if events.is_empty()
        || events.len() != manifest.requested_record_count
        || events.len() != manifest.actual_record_count
        || events.len() != event_identity.record_count
    {
        return Err("event record count disagrees with the manifest".to_string());
    }

    let oracle: Oracle = serde_json::from_slice(oracle_bytes)
        .map_err(|e| format!("cannot parse evaluator oracle: {e}"))?;
    if oracle.schema_version != ORACLE_VERSION
        || oracle.generator_version != GENERATOR_VERSION
        || oracle.seed != manifest.seed
        || oracle.record_count != events.len()
        || oracle.labels.len() != events.len()
        || oracle.labels.len() != oracle_identity.record_count
        || oracle.non_claims.is_empty()
    {
        return Err("oracle versions, seed, record count, or non-claims are invalid".to_string());
    }

    let mut event_ids = BTreeSet::<String>::new();
    let mut delivery_ids = BTreeSet::<String>::new();
    for event in events.iter() {
        if event.schema_version != EVENT_SCHEMA_VERSION
            || event.record_id.trim().is_empty()
            || event.source_event_id.trim().is_empty()
            || event.delivery_id.trim().is_empty()
            || event.host_id.trim().is_empty()
            || event.principal_id.trim().is_empty()
            || event.source_address.trim().is_empty()
            || event.destination.trim().is_empty()
            || event.device_posture.trim().is_empty()
            || event.source.trim().is_empty()
            || event.event_type.trim().is_empty()
            || event.summary.trim().is_empty()
            || !matches!(
                event.integrity_state.as_str(),
                "observed" | "duplicate_delivery" | "source_id_conflict"
            )
        {
            return Err(format!("event {} has missing or unsupported fields", event.record_id));
        }
        if !event_ids.insert(event.record_id.clone()) {
            return Err(format!("duplicate solver record ID {}", event.record_id));
        }
        if !delivery_ids.insert(event.delivery_id.clone()) {
            return Err(format!("duplicate delivery ID {}", event.delivery_id));
        }
    }

    let mut labels_by_id = BTreeMap::<String, TruthLabel>::new();
    for label in &oracle.labels {
        if !matches!(label.security_label.as_str(), "benign" | "malicious")
            || !matches!(
                label.data_quality.as_str(),
                "normal" | "duplicate_delivery" | "source_id_conflict"
            )
            || label.rationale.trim().is_empty()
        {
            return Err(format!("oracle label {} has invalid fields", label.record_id));
        }
        if labels_by_id.insert(label.record_id.clone(), label.clone()).is_some() {
            return Err(format!("duplicate oracle record ID {}", label.record_id));
        }
    }
    if event_ids != labels_by_id.keys().cloned().collect::<BTreeSet<_>>()
    {
        return Err("solver record IDs and oracle record IDs do not match exactly".to_string());
    }

    let mut duplicate_delivery_count = 0usize;
    let mut source_identity_conflict_count = 0usize;
    for event in &events {
        let label = &labels_by_id[&event.record_id];
        let quality_agrees = matches!(
            (event.integrity_state.as_str(), label.data_quality.as_str()),
            ("observed", "normal")
                | ("duplicate_delivery", "duplicate_delivery")
                | ("source_id_conflict", "source_id_conflict")
        );
        if !quality_agrees {
            return Err(format!(
                "event/oracle data-quality mismatch for record {}",
                event.record_id
            ));
        }

        match event.integrity_state.as_str() {
            "duplicate_delivery" => {
                duplicate_delivery_count += 1;
                let matching_payload = events.iter().any(|other| {
                    other.record_id != event.record_id
                        && other.source_event_id == event.source_event_id
                        && other.event_time_ms == event.event_time_ms
                        && other.host_id == event.host_id
                        && other.principal_id == event.principal_id
                        && other.source_address == event.source_address
                        && other.destination == event.destination
                        && other.bytes_sent == event.bytes_sent
                        && other.device_posture == event.device_posture
                        && other.source == event.source
                        && other.event_type == event.event_type
                        && other.result == event.result
                        && other.summary == event.summary
                        && other.change_ticket == event.change_ticket
                });
                if !matching_payload {
                    return Err(format!(
                        "duplicate delivery {} has no matching source payload",
                        event.record_id
                    ));
                }
            }
            "source_id_conflict" => {
                source_identity_conflict_count += 1;
                let conflicting_payload = events.iter().any(|other| {
                    other.record_id != event.record_id
                        && other.source_event_id == event.source_event_id
                        && other.summary != event.summary
                });
                if !conflicting_payload {
                    return Err(format!(
                        "source-ID conflict {} has no differing payload",
                        event.record_id
                    ));
                }
            }
            _ => {}
        }
    }

    let malicious_event_count = labels_by_id
        .values()
        .filter(|label| label.security_label == "malicious")
        .count();
    let benign_event_count = labels_by_id
        .values()
        .filter(|label| label.security_label == "benign")
        .count();
    if malicious_event_count + benign_event_count != events.len() {
        return Err("oracle security label census does not cover every event".to_string());
    }

    let mut campaign_ids = BTreeSet::<String>::new();
    let mut covered_campaign_records = BTreeSet::<String>::new();
    let mut malicious_campaign_count = 0usize;
    let mut benign_change_decoy_count = 0usize;
    for campaign in &oracle.campaigns {
        if !campaign_ids.insert(campaign.campaign_id.clone())
            || campaign.campaign_id.trim().is_empty()
            || campaign.claim_ceiling.trim().is_empty()
            || campaign.event_record_ids.is_empty()
        {
            return Err(format!("campaign {} has malformed identity/claims", campaign.campaign_id));
        }
        if !unique_values(&campaign.event_record_ids) {
            return Err(format!("campaign {} repeats an event record ID", campaign.campaign_id));
        }
        let expected_label = match campaign.disposition.as_str() {
            "malicious" => {
                malicious_campaign_count += 1;
                "malicious"
            }
            "benign_decoy" => {
                benign_change_decoy_count += 1;
                "benign"
            }
            other => return Err(format!("unsupported campaign disposition {other}")),
        };
        for record_id in &campaign.event_record_ids {
            let Some(label) = labels_by_id.get(record_id) else {
                return Err(format!(
                    "campaign {} references unknown record {}",
                    campaign.campaign_id, record_id
                ));
            };
            if label.security_label != expected_label
                || label.campaign_id.as_deref() != Some(campaign.campaign_id.as_str())
            {
                return Err(format!(
                    "campaign {} does not agree with oracle label for {}",
                    campaign.campaign_id, record_id
                ));
            }
            if !covered_campaign_records.insert(record_id.clone()) {
                return Err(format!("record {} appears in multiple planted campaigns", record_id));
            }
        }
    }
    for label in labels_by_id.values() {
        if label.security_label == "malicious" && !covered_campaign_records.contains(&label.record_id) {
            return Err(format!("malicious label {} is not covered by a campaign", label.record_id));
        }
        if let Some(campaign_id) = label.campaign_id.as_ref() {
            if !campaign_ids.contains(campaign_id) {
                return Err(format!("label {} references unknown campaign {}", label.record_id, campaign_id));
            }
        }
    }

    if malicious_campaign_count != manifest.malicious_campaign_count
        || benign_change_decoy_count != manifest.benign_change_decoy_count
        || duplicate_delivery_count != manifest.duplicate_delivery_count
        || source_identity_conflict_count != manifest.source_identity_conflict_count
    {
        return Err("manifest campaign/data-quality counts disagree with the actual artifacts".to_string());
    }

    Ok(VerificationReport {
        disposition: "STRUCTURAL_PASS_NOT_CAPABILITY_PASS".to_string(),
        source_commit: manifest.source_commit.clone(),
        source_commit_pinned: is_full_git_sha(&manifest.source_commit),
        seed: manifest.seed,
        record_count: events.len(),
        malicious_event_count,
        benign_event_count,
        duplicate_delivery_count,
        source_identity_conflict_count,
        event_sha256: event_identity.sha256.clone(),
        oracle_sha256: oracle_identity.sha256.clone(),
        non_claims: vec![
            "This report establishes exact-byte identity and bounded internal structural consistency only.".to_string(),
            "The public oracle is not a blind detector-capability benchmark.".to_string(),
            "Planted synthetic ground truth does not establish a real incident, cause, attribution, or production security.".to_string(),
        ],
    })
}

fn render_dataset(records: usize, seed: u64, source_commit: &str) -> Result<RenderedDataset, String> {
    let dataset = build_dataset(records, seed)?;
    // The solver-visible data contains only Event fields; ground truth is a separate file.
    let events_bytes = render_jsonl(&dataset.events)?;
    let oracle_bytes = serde_json::to_vec_pretty(&dataset.oracle)
        .map_err(|e| format!("failed to serialize evaluator oracle: {e}"))?;
    let mut files = BTreeMap::new();
    files.insert(
        "solver/events.jsonl".to_string(),
        FileIdentity {
            sha256: sha256(&events_bytes),
            utf8_bytes: events_bytes.len(),
            record_count: dataset.events.len(),
        },
    );
    files.insert(
        "evaluator/oracle.json".to_string(),
        FileIdentity {
            sha256: sha256(&oracle_bytes),
            utf8_bytes: oracle_bytes.len(),
            record_count: dataset.oracle.labels.len(),
        },
    );
    let campaign_count = dataset.oracle.campaigns.iter().filter(|c| c.disposition == "malicious").count();
    let benign_change_decoy_count = dataset.oracle.campaigns.iter().filter(|c| c.disposition == "benign_decoy").count();
    let duplicate_delivery_count = dataset.events.iter().filter(|e| e.integrity_state == "duplicate_delivery").count();
    let source_identity_conflict_count = dataset.events.iter().filter(|e| e.integrity_state == "source_id_conflict").count();
    let solver_manifest = SolverManifest {
        schema_version: SOLVER_MANIFEST_VERSION.to_string(),
        generator_version: GENERATOR_VERSION.to_string(),
        source_commit: source_commit.to_string(),
        source_commit_pinned: is_full_git_sha(source_commit),
        synthetic_only: true,
        seed,
        record_count: files["solver/events.jsonl"].record_count,
        event_file: "events.jsonl".to_string(),
        event_sha256: files["solver/events.jsonl"].sha256.clone(),
        event_utf8_bytes: files["solver/events.jsonl"].utf8_bytes,
        qualification_state: "UNSCORED".to_string(),
        claim_ceiling: "Synthetic solver-visible event records only; no evaluator labels, campaign answers, or capability claims.".to_string(),
    };
    let solver_manifest_bytes = serde_json::to_vec_pretty(&solver_manifest)
        .map_err(|e| format!("failed to serialize solver manifest: {e}"))?;
    files.insert(
        "solver/manifest.json".to_string(),
        FileIdentity {
            sha256: sha256(&solver_manifest_bytes),
            utf8_bytes: solver_manifest_bytes.len(),
            record_count: 1,
        },
    );
    let manifest = Manifest {
        schema_version: MANIFEST_VERSION.to_string(),
        generator_version: GENERATOR_VERSION.to_string(),
        source_commit: source_commit.to_string(),
        synthetic_only: true,
        seed,
        requested_record_count: records,
        actual_record_count: files["solver/events.jsonl"].record_count,
        malicious_campaign_count: campaign_count,
        benign_change_decoy_count,
        duplicate_delivery_count,
        source_identity_conflict_count,
        solver_file: "solver/events.jsonl".to_string(),
        oracle_file: "evaluator/oracle.json".to_string(),
        files,
        qualification_state: "UNSCORED".to_string(),
        claim_ceiling: "Deterministic synthetic event-store construction and structural integrity only; no detector-quality, incident-truth, or operational security claim.".to_string(),
    };
    // Fail at fixture construction if corpus, solver manifest, oracle, and combined manifest drift apart.
    verify_payloads(&events_bytes, &solver_manifest_bytes, &oracle_bytes, &manifest)?;
    Ok(RenderedDataset { events_bytes, solver_manifest_bytes, oracle_bytes, manifest, solver_manifest })
}

fn verify_output_directory(root: &Path) -> Result<VerificationReport, String> {
    let manifest_path = root.join("manifest.json");
    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|e| format!("cannot read {}: {e}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("cannot parse manifest {}: {e}", manifest_path.display()))?;
    if manifest.solver_file != "solver/events.jsonl"
        || manifest.oracle_file != "evaluator/oracle.json"
    {
        return Err("manifest paths are not the fixed solver/evaluator contract paths".to_string());
    }
    let event_path = root.join("solver").join("events.jsonl");
    let oracle_path = root.join("evaluator").join("oracle.json");
    let events_bytes = fs::read(&event_path)
        .map_err(|e| format!("cannot read {}: {e}", event_path.display()))?;
    let solver_manifest_path = root.join("solver").join("manifest.json");
    let solver_manifest_bytes = fs::read(&solver_manifest_path)
        .map_err(|e| format!("cannot read {}: {e}", solver_manifest_path.display()))?;
    let oracle_bytes = fs::read(&oracle_path)
        .map_err(|e| format!("cannot read {}: {e}", oracle_path.display()))?;
    verify_payloads(&events_bytes, &solver_manifest_bytes, &oracle_bytes, &manifest)
}

fn write_dataset(config: &Config) -> Result<RunReceipt, String> {
    let rendered = render_dataset(config.records, config.seed, &config.source_commit)?;
    fs::create_dir_all(&config.out_dir)
        .map_err(|e| format!("cannot create output directory {}: {e}", config.out_dir.display()))?;
    if let Some(entry) = fs::read_dir(&config.out_dir)
        .map_err(|e| format!("cannot inspect output directory {}: {e}", config.out_dir.display()))?
        .next()
    {
        let entry = entry.map_err(|e| format!("cannot inspect existing output entry: {e}"))?;
        return Err(format!(
            "output directory is not empty: {} (found {})",
            config.out_dir.display(),
            entry.path().display()
        ));
    }
    let solver_dir = config.out_dir.join("solver");
    let evaluator_dir = config.out_dir.join("evaluator");
    let event_path = solver_dir.join("events.jsonl");
    let solver_manifest_path = solver_dir.join("manifest.json");
    let oracle_path = evaluator_dir.join("oracle.json");
    let manifest_path = config.out_dir.join("manifest.json");
    if event_path.exists() || solver_manifest_path.exists() || oracle_path.exists() || manifest_path.exists() {
        return Err(format!(
            "output files already exist in {}; choose a fresh --out-dir to avoid accidental overwrite",
            config.out_dir.display()
        ));
    }
    fs::create_dir_all(&solver_dir)
        .map_err(|e| format!("cannot create {}: {e}", solver_dir.display()))?;
    fs::create_dir_all(&evaluator_dir)
        .map_err(|e| format!("cannot create {}: {e}", evaluator_dir.display()))?;
    fs::write(&event_path, &rendered.events_bytes)
        .map_err(|e| format!("cannot write {}: {e}", event_path.display()))?;
    fs::write(&solver_manifest_path, &rendered.solver_manifest_bytes)
        .map_err(|e| format!("cannot write {}: {e}", solver_manifest_path.display()))?;
    fs::write(&oracle_path, &rendered.oracle_bytes)
        .map_err(|e| format!("cannot write {}: {e}", oracle_path.display()))?;
    let manifest_bytes = serde_json::to_vec_pretty(&rendered.manifest)
        .map_err(|e| format!("cannot serialize manifest: {e}"))?;
    fs::write(&manifest_path, &manifest_bytes)
        .map_err(|e| format!("cannot write {}: {e}", manifest_path.display()))?;
    let verification = verify_output_directory(&config.out_dir)?;
    if verification.source_commit != config.source_commit
        || verification.seed != config.seed
        || verification.record_count != config.records
    {
        return Err("read-back receipt does not match the requested source/seed/record count".to_string());
    }
    Ok(RunReceipt {
        manifest: rendered.manifest,
        solver_manifest: rendered.solver_manifest,
        verification,
    })
}

fn parse_args() -> Result<Option<Config>, String> {
    let mut records = DEFAULT_RECORDS;
    let mut seed = DEFAULT_SEED;
    let mut out_dir = None;
    let mut source_commit = env::var("SOURCE_COMMIT").unwrap_or_else(|_| "UNPINNED_LOCAL_SOURCE".to_string());
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--records" => {
                let value = args.next().ok_or_else(|| "--records requires a positive integer".to_string())?;
                records = value.parse().map_err(|_| format!("invalid --records value: {value}"))?;
            }
            "--seed" => {
                let value = args.next().ok_or_else(|| "--seed requires an unsigned integer".to_string())?;
                seed = value.parse().map_err(|_| format!("invalid --seed value: {value}"))?;
            }
            "--out-dir" => {
                let value = args.next().ok_or_else(|| "--out-dir requires a path".to_string())?;
                out_dir = Some(PathBuf::from(value));
            }
            "--source-commit" => {
                source_commit = args.next().ok_or_else(|| "--source-commit requires a commit identifier".to_string())?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    let out_dir = out_dir.unwrap_or_else(|| PathBuf::from(format!("target/cyber-hunt-{records}-{seed}")));
    Ok(Some(Config { records, seed, out_dir, source_commit }))
}

fn print_help() {
    println!(
        "Deterministic synthetic defensive threat-hunting fixture generator\n\n\
         Usage: cyber-hunt-fixture [--records N] [--seed N] [--out-dir PATH] [--source-commit SHA]\n\n\
         --records N       Total event rows (500..=250000; default 10000)\n\
         --seed N          Deterministic non-cryptographic seed (default 42)\n\
         --out-dir PATH    New or empty output directory (default target/cyber-hunt-N-SEED)\n\
         --source-commit   Source revision included as provenance metadata (default UNPINNED_LOCAL_SOURCE)\n\
         Query mode: cyber-hunt-fixture query --store-dir SOLVER_ONLY_DIR --requests-jsonl REQUESTS.jsonl\n\
         Evaluate mode: cyber-hunt-fixture evaluate --dataset-root DATASET_ROOT --predictions PREDICTIONS.json\n\n\
         Outputs: solver/events.jsonl + solver/manifest.json; evaluator/oracle.json; root manifest.json pins all three artifacts.\n\
         Stdout is a read-back receipt; local source metadata is not a trusted build attestation.\n\
         Every output is synthetic and UNSCORED. The public oracle is not held-out evaluation data."
    );
}

fn main() {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    if raw_args.first().map(String::as_str) == Some("query") {
        query::run_cli_or_exit(&raw_args[1..]);
        return;
    }
    if raw_args.first().map(String::as_str) == Some("evaluate") {
        evaluation::run_cli_or_exit(&raw_args[1..]);
        return;
    }
    match parse_args() {
        Ok(None) => print_help(),
        Err(error) => {
            eprintln!("cyber-hunt-fixture: FAIL: {error}");
            print_help();
            process::exit(2);
        }
        Ok(Some(config)) => match write_dataset(&config) {
            Ok(receipt) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&receipt).expect("receipt serialization is infallible")
                );
            }
            Err(error) => {
                eprintln!("cyber-hunt-fixture: FAIL: {error}");
                process::exit(1);
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_schema_matches_object_keys(value: &serde_json::Value, schema: &serde_json::Value) {
        let object = value.as_object().expect("schema sample must be an object");
        let actual: BTreeSet<String> = object.keys().cloned().collect();
        let required: BTreeSet<String> = schema["required"]
            .as_array()
            .expect("schema root must have a required array")
            .iter()
            .map(|key| key.as_str().expect("required field must be a string").to_string())
            .collect();
        assert_eq!(actual, required, "schema fields drifted from serialized type");
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn versioned_json_schemas_match_generated_record_shapes() {
        let event_schema: serde_json::Value = serde_json::from_str(include_str!("../schemas/event-v1.schema.json"))
            .expect("event schema must be valid JSON");
        let oracle_schema: serde_json::Value = serde_json::from_str(include_str!("../schemas/oracle-v1.schema.json"))
            .expect("oracle schema must be valid JSON");
        let manifest_schema: serde_json::Value = serde_json::from_str(include_str!("../schemas/manifest-v1.schema.json"))
            .expect("manifest schema must be valid JSON");
        let solver_manifest_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/solver-manifest-v1.schema.json"))
                .expect("solver manifest schema must be valid JSON");
        let query_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/query-v1.schema.json"))
                .expect("query schema must be valid JSON");
        let query_receipt_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/query-receipt-v1.schema.json"))
                .expect("query receipt schema must be valid JSON");
        let query_response_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/query-response-v1.schema.json"))
                .expect("query response schema must be valid JSON");
        let query_batch_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/query-batch-line-v1.schema.json"))
                .expect("query batch schema must be valid JSON");
        let predictions_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/predictions-v1.schema.json"))
                .expect("prediction schema must be valid JSON");
        let evaluation_receipt_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/evaluation-receipt-v1.schema.json"))
                .expect("evaluation receipt schema must be valid JSON");

        for schema in [
            &event_schema,
            &oracle_schema,
            &manifest_schema,
            &solver_manifest_schema,
            &query_schema,
            &query_receipt_schema,
            &query_response_schema,
            &predictions_schema,
            &evaluation_receipt_schema,
        ] {
            assert_eq!(schema["$schema"], "https://json-schema.org/draft/2020-12/schema");
            assert_eq!(schema["type"], "object");
            assert_eq!(schema["additionalProperties"], false);
        }

        let rendered = render_dataset(1_000, 42, "test-source-commit").expect("fixture should render");
        let event: serde_json::Value = serde_json::from_slice(
            rendered.events_bytes.split(|byte| *byte == b'\n').next().expect("first event line")
        ).expect("first event should parse");
        assert_eq!(event["schema_version"], EVENT_SCHEMA_VERSION);
        assert_eq!(event_schema["properties"]["schema_version"]["const"], EVENT_SCHEMA_VERSION);
        assert_schema_matches_object_keys(&event, &event_schema);

        let oracle: serde_json::Value = serde_json::from_slice(&rendered.oracle_bytes)
            .expect("oracle should parse");
        assert_eq!(oracle["schema_version"], ORACLE_VERSION);
        assert_schema_matches_object_keys(&oracle, &oracle_schema);
        let label_schema = &oracle_schema["properties"]["labels"]["items"];
        assert_schema_matches_object_keys(&oracle["labels"][0], label_schema);
        let campaign_schema = &oracle_schema["properties"]["campaigns"]["items"];
        assert_schema_matches_object_keys(&oracle["campaigns"][0], campaign_schema);

        let manifest_value = serde_json::to_value(&rendered.manifest).expect("manifest should serialize");
        assert_schema_matches_object_keys(&manifest_value, &manifest_schema);
        let file_schema = &manifest_schema["properties"]["files"]["properties"]["solver/events.jsonl"];
        assert_schema_matches_object_keys(&manifest_value["files"]["solver/events.jsonl"], file_schema);

        let solver_manifest_value =
            serde_json::to_value(&rendered.solver_manifest).expect("solver manifest should serialize");
        assert_schema_matches_object_keys(&solver_manifest_value, &solver_manifest_schema);
        let solver_file_schema = &manifest_schema["properties"]["files"]["properties"]["solver/manifest.json"];
        assert_schema_matches_object_keys(&manifest_value["files"]["solver/manifest.json"], solver_file_schema);
        assert_eq!(
            solver_manifest_schema["properties"]["schema_version"]["const"],
            SOLVER_MANIFEST_VERSION
        );
        assert_eq!(query_schema["properties"]["schema_version"]["const"], "cyber-hunt-query-v1");
        assert_eq!(query_receipt_schema["properties"]["schema_version"]["const"], "cyber-hunt-query-receipt-v1");
        assert_eq!(query_response_schema["required"].as_array().unwrap().len(), 2);
        assert_eq!(query_batch_schema["oneOf"].as_array().unwrap().len(), 2);
        assert_eq!(predictions_schema["properties"]["schema_version"]["const"], "cyber-hunt-predictions-v1");
        assert_eq!(evaluation_receipt_schema["properties"]["schema_version"]["const"], "cyber-hunt-evaluation-v1");
        assert!(query_schema["properties"]["filters"]["items"]["oneOf"].as_array().unwrap().len() == 3);
    }

    #[test]
    fn every_public_schema_passes_json_schema_draft_meta_validation() {
        let schemas = [
            ("event-v1", include_str!("../schemas/event-v1.schema.json")),
            ("oracle-v1", include_str!("../schemas/oracle-v1.schema.json")),
            ("manifest-v1", include_str!("../schemas/manifest-v1.schema.json")),
            ("solver-manifest-v1", include_str!("../schemas/solver-manifest-v1.schema.json")),
            ("query-v1", include_str!("../schemas/query-v1.schema.json")),
            ("query-receipt-v1", include_str!("../schemas/query-receipt-v1.schema.json")),
            ("query-response-v1", include_str!("../schemas/query-response-v1.schema.json")),
            ("query-batch-line-v1", include_str!("../schemas/query-batch-line-v1.schema.json")),
            ("predictions-v1", include_str!("../schemas/predictions-v1.schema.json")),
            ("evaluation-receipt-v1", include_str!("../schemas/evaluation-receipt-v1.schema.json")),
        ];
        for (name, text) in schemas {
            let schema: serde_json::Value =
                serde_json::from_str(text).unwrap_or_else(|error| panic!("{name}: invalid JSON: {error}"));
            jsonschema::meta::validate(&schema)
                .unwrap_or_else(|error| panic!("{name}: JSON Schema meta-validation failed: {error}"));
        }
    }

    #[test]
    fn sha256_known_answer() {
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn generated_dataset_has_one_truth_record_per_solver_record() {
        let dataset = build_dataset(1_000, 7).expect("valid fixture size should generate");
        assert_eq!(dataset.events.len(), 1_000);
        assert_eq!(dataset.oracle.labels.len(), 1_000);
        assert_eq!(dataset.oracle.record_count, 1_000);

        let event_ids: BTreeSet<_> = dataset.events.iter().map(|event| event.record_id.clone()).collect();
        let truth_ids: BTreeSet<_> = dataset.oracle.labels.iter().map(|label| label.record_id.clone()).collect();
        assert_eq!(event_ids.len(), 1_000);
        assert_eq!(truth_ids.len(), 1_000);
        assert_eq!(event_ids, truth_ids);
    }

    #[test]
    fn same_seed_yields_byte_identical_artifacts() {
        let a = render_dataset(1_000, 123, "test-source-commit").expect("fixture should render");
        let b = render_dataset(1_000, 123, "test-source-commit").expect("fixture should render");
        assert_eq!(a.events_bytes, b.events_bytes);
        assert_eq!(a.oracle_bytes, b.oracle_bytes);
        assert_eq!(a.manifest, b.manifest);
    }

    #[test]
    fn a_different_seed_changes_the_event_store() {
        let a = render_dataset(1_000, 123, "test-source-commit").expect("fixture should render");
        let b = render_dataset(1_000, 124, "test-source-commit").expect("fixture should render");
        assert_ne!(a.events_bytes, b.events_bytes);
    }

    #[test]
    fn event_view_does_not_contain_evaluator_labels() {
        let dataset = build_dataset(1_000, 7).expect("valid fixture size should generate");
        let row = serde_json::to_string(&dataset.events[0]).expect("event should serialize");
        assert!(!row.contains("security_label"));
        assert!(!row.contains("malicious"));
        assert!(!row.contains("benign_decoy"));
        assert!(!row.contains("hunt-campaign"));
        assert!(!row.contains("benign-change"));
        for event in &dataset.events {
            let visible = serde_json::to_string(event).expect("event should serialize");
            assert!(!visible.contains("security_label"));
            assert!(!visible.contains("hunt-campaign"));
            assert!(!visible.contains("benign-change"));
        }
    }

    #[test]
    fn duplicate_deliveries_and_conflicts_are_explicit_and_separate_from_security_labels() {
        let dataset = build_dataset(10_000, 42).expect("valid fixture size should generate");
        assert!(dataset.events.iter().any(|e| e.integrity_state == "duplicate_delivery"));
        assert!(dataset.events.iter().any(|e| e.integrity_state == "source_id_conflict"));
        assert!(dataset.oracle.labels.iter().any(|x| x.data_quality == "duplicate_delivery"));
        assert!(dataset.oracle.labels.iter().any(|x| x.data_quality == "source_id_conflict"));
        let malicious_transfer = dataset.events.iter().find(|event| {
            event.event_type == "network.transfer" && event.bytes_sent >= 10_000_000
        }).expect("planted high-volume transfer should exist");
        let benign_transfer = dataset.events.iter().find(|event| {
            event.event_type == "network.transfer"
                && event.bytes_sent >= 9_500_000
                && event.change_ticket.is_some()
        }).expect("similar high-volume approved transfer should exist");
        assert_ne!(malicious_transfer.destination, benign_transfer.destination);
        assert_eq!(malicious_transfer.device_posture, "not-applicable");
        assert_eq!(benign_transfer.device_posture, "managed");
    }

    #[test]
    fn attack_chain_and_benign_change_decoy_are_both_present() {
        let dataset = build_dataset(1_000, 42).expect("valid fixture size should generate");
        assert_eq!(dataset.oracle.campaigns.iter().filter(|c| c.disposition == "malicious").count(), 2);
        assert_eq!(dataset.oracle.campaigns.iter().filter(|c| c.disposition == "benign_decoy").count(), 2);
        assert!(dataset.oracle.labels.iter().any(|x| x.security_label == "malicious"));
        assert!(dataset.oracle.labels.iter().any(|x| x.campaign_id.as_deref().unwrap_or("").starts_with("benign-change-")));
    }

    #[test]
    fn unsupported_sizes_fail_closed() {
        assert!(build_dataset(MIN_RECORDS - 1, 42).is_err());
        assert!(build_dataset(MAX_RECORDS + 1, 42).is_err());
    }

    #[test]
    fn every_campaign_truth_id_references_an_existing_record() {
        let dataset = build_dataset(10_000, 42).expect("valid fixture size should generate");
        let event_ids: BTreeSet<_> = dataset.events.iter().map(|event| event.record_id.as_str()).collect();
        for campaign in &dataset.oracle.campaigns {
            for id in &campaign.event_record_ids {
                assert!(event_ids.contains(id.as_str()), "missing oracle record {id}");
            }
        }
    }

    #[test]
    fn write_readback_verifies_and_refuses_overwrite() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after UNIX epoch")
            .as_nanos();
        let out_dir = env::temp_dir().join(format!(
            "cyber-hunt-fixture-test-{}-{nonce}",
            process::id()
        ));
        let config = Config {
            records: 1_000,
            seed: 42,
            out_dir: out_dir.clone(),
            source_commit: "test-source-commit".to_string(),
        };
        let receipt = write_dataset(&config).expect("first write and disk read-back should pass");
        assert_eq!(receipt.verification.disposition, "STRUCTURAL_PASS_NOT_CAPABILITY_PASS");
        assert!(!receipt.verification.source_commit_pinned);
        assert_eq!(receipt.verification.record_count, 1_000);
        assert!(out_dir.join("solver/events.jsonl").is_file());
        assert!(out_dir.join("solver/manifest.json").is_file());
        assert!(out_dir.join("evaluator/oracle.json").is_file());
        assert!(out_dir.join("manifest.json").is_file());
        assert!(!out_dir.join("solver/oracle.json").exists());

        let error = write_dataset(&config)
            .expect_err("existing outputs must never be silently overwritten");
        assert!(error.contains("not empty"));
        fs::remove_dir_all(out_dir).expect("temporary output directory should be removed");
    }

    #[test]
    fn manifest_hashes_match_exact_rendered_bytes() {
        let rendered = render_dataset(1_000, 42, "test-source-commit").expect("fixture should render");
        assert_eq!(
            rendered.manifest.files["solver/events.jsonl"].sha256,
            sha256(&rendered.events_bytes)
        );
        assert_eq!(
            rendered.manifest.files["evaluator/oracle.json"].sha256,
            sha256(&rendered.oracle_bytes)
        );
        assert_eq!(rendered.manifest.files["solver/events.jsonl"].record_count, 1_000);
        assert_eq!(rendered.manifest.files["evaluator/oracle.json"].record_count, 1_000);
        assert_eq!(rendered.manifest.files["solver/manifest.json"].sha256, sha256(&rendered.solver_manifest_bytes));
        assert_eq!(rendered.manifest.files["solver/manifest.json"].record_count, 1);
        let report = verify_payloads(&rendered.events_bytes, &rendered.solver_manifest_bytes, &rendered.oracle_bytes, &rendered.manifest)
            .expect("rendered artifacts should verify");
        assert_eq!(report.disposition, "STRUCTURAL_PASS_NOT_CAPABILITY_PASS");
        assert_eq!(report.record_count, 1_000);
        assert!(report.malicious_event_count > 0);
        assert!(report.benign_event_count > report.malicious_event_count);
    }

    #[test]
    fn verifier_rejects_tampered_event_bytes_before_json_interpretation() {
        let rendered = render_dataset(1_000, 42, "test-source-commit").expect("fixture should render");
        let mut tampered = rendered.events_bytes.clone();
        tampered.push(b' ');
        let error = verify_payloads(&tampered, &rendered.solver_manifest_bytes, &rendered.oracle_bytes, &rendered.manifest)
            .expect_err("mutated event stream must fail integrity verification");
        assert!(error.contains("SHA-256/length"));
    }

    #[test]
    fn verifier_rejects_unknown_event_fields() {
        let rendered = render_dataset(1_000, 42, "test-source-commit").expect("fixture should render");
        let first_line_end = rendered.events_bytes.iter().position(|byte| *byte == b'\\n')
            .expect("event stream should have a first line");
        let mut first: serde_json::Value = serde_json::from_slice(&rendered.events_bytes[..first_line_end])
            .expect("first event should be JSON");
        first.as_object_mut().expect("event is a JSON object")
            .insert("security_label".to_string(), serde_json::Value::String("malicious".to_string()));
        let mut tampered_events = serde_json::to_vec(&first).expect("event should serialize");
        tampered_events.push(b'\\n');
        tampered_events.extend_from_slice(&rendered.events_bytes[first_line_end + 1..]);
        let mut changed_manifest = rendered.manifest.clone();
        let mut changed_solver_manifest: serde_json::Value =
            serde_json::from_slice(&rendered.solver_manifest_bytes)
                .expect("solver manifest should be JSON");
        changed_solver_manifest["event_sha256"] =
            serde_json::Value::String(sha256(&tampered_events));
        changed_solver_manifest["event_utf8_bytes"] =
            serde_json::Value::from(tampered_events.len());
        let changed_solver_manifest_bytes =
            serde_json::to_vec_pretty(&changed_solver_manifest).expect("solver manifest should serialize");
        let event_identity = changed_manifest.files.get_mut("solver/events.jsonl")
            .expect("event identity exists");
        event_identity.sha256 = sha256(&tampered_events);
        event_identity.utf8_bytes = tampered_events.len();
        let solver_manifest_identity = changed_manifest.files.get_mut("solver/manifest.json")
            .expect("solver manifest identity exists");
        solver_manifest_identity.sha256 = sha256(&changed_solver_manifest_bytes);
        solver_manifest_identity.utf8_bytes = changed_solver_manifest_bytes.len();
        let error = verify_payloads(
            &tampered_events,
            &changed_solver_manifest_bytes,
            &rendered.oracle_bytes,
            &changed_manifest,
        )
            .expect_err("unknown evaluator-only field must be rejected even with recomputed hashes");
        assert!(error.contains("unknown field") || error.contains("denied"));
    }

    #[test]
    fn verifier_rejects_missing_oracle_label_even_when_file_hash_is_updated() {
        let rendered = render_dataset(1_000, 42, "test-source-commit").expect("fixture should render");
        let mut oracle: serde_json::Value =
            serde_json::from_slice(&rendered.oracle_bytes).expect("oracle should be JSON");
        let labels = oracle.get_mut("labels").and_then(serde_json::Value::as_array_mut)
            .expect("oracle labels should be an array");
        labels.pop();
        let changed_oracle = serde_json::to_vec_pretty(&oracle).expect("oracle should serialize");
        let mut changed_manifest = rendered.manifest.clone();
        let identity = changed_manifest.files.get_mut("evaluator/oracle.json").expect("oracle identity exists");
        identity.sha256 = sha256(&changed_oracle);
        identity.utf8_bytes = changed_oracle.len();
        identity.record_count = 999;
        let error = verify_payloads(&rendered.events_bytes, &rendered.solver_manifest_bytes, &changed_oracle, &changed_manifest)
            .expect_err("partial oracle coverage must be rejected");
        assert!(error.contains("record count") || error.contains("IDs do not match"));
    }
}
