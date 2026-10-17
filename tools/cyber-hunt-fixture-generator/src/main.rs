use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const GENERATOR_VERSION: &str = "cyber-hunt-fixture-generator/0.5.0";
const EVENT_SCHEMA: &str = "cyber-hunt-event-v1";
const MANIFEST_SCHEMA: &str = "cyber-hunt-manifest-v1";
const TRUTH_SCHEMA: &str = "cyber-hunt-truth-v1";
const SCENARIO_ID: &str = "synthetic-identity-to-egress-v1";
const MIN_RECORDS: usize = 64;
const MAX_RECORDS: usize = 250_000;
const BASE_TIME_MS: i64 = 1_800_000_000_000;
const ATTACK_EVENT_COUNT: usize = 9;
const BENIGN_DECOY_COUNT: usize = 7;
const PLANTED_EVENT_COUNT: usize = ATTACK_EVENT_COUNT + BENIGN_DECOY_COUNT;
const SYNTHETIC_PRINCIPALS: [&str; 12] = [
    "a.chen", "j.smith", "m.patel", "it.admin", "svc-report", "svc-backup",
    "svc-build", "d.owens", "s.lee", "svc-inventory", "helpdesk.1", "r.garcia",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct EventRecord {
    schema_version: String,
    event_id: String,
    event_time_ms: i64,
    ingest_time_ms: i64,
    host_id: String,
    principal_id: String,
    source: String,
    event_type: String,
    outcome: String,
    details: BTreeMap<String, String>,
    synthetic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct EventLabel {
    event_id: String,
    label: String,
    rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ExpectedRelation {
    from_event_id: String,
    to_event_id: String,
    relation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct GroundTruth {
    schema_version: String,
    scenario_id: String,
    profile: String,
    seed: u64,
    event_count: usize,
    event_store_sha256: String,
    labels: Vec<EventLabel>,
    expected_relations: Vec<ExpectedRelation>,
    claim_ceiling: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct DatasetManifest {
    schema_version: String,
    generator_version: String,
    profile: String,
    scenario_id: String,
    seed: u64,
    record_count: usize,
    event_store_sha256: String,
    event_store_utf8_bytes: usize,
    ordering: String,
    qualification_state: String,
    generator_source_commit: String,
    source_commit_pinned: bool,
    claim_ceiling: String,
}

struct Dataset {
    event_store: Vec<u8>,
    manifest: DatasetManifest,
    truth: GroundTruth,
}

#[derive(Debug, Clone, Copy)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*: stable, tiny, deterministic generator for public synthetic fixtures.
        // This is not a cryptographic random-number generator.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, upper: u64) -> u64 {
        debug_assert!(upper > 0);
        self.next_u64() % upper
    }
}

fn details(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
    entries
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn encode_jsonl<T: Serialize>(records: &[T]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for record in records {
        serde_json::to_writer(&mut out, record)
            .map_err(|e| format!("cannot serialize JSONL record: {e}"))?;
        out.push(b'\n');
    }
    Ok(out)
}

fn next_event_id(rng: &mut Rng, used: &mut BTreeSet<String>) -> String {
    loop {
        let id = format!("evt-{:016x}", rng.next_u64());
        if used.insert(id.clone()) {
            return id;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_event(
    events: &mut Vec<EventRecord>,
    rng: &mut Rng,
    used_ids: &mut BTreeSet<String>,
    event_time_ms: i64,
    host_id: &str,
    principal_id: &str,
    source: &str,
    event_type: &str,
    outcome: &str,
    mut event_details: BTreeMap<String, String>,
) -> String {
    let event_id = next_event_id(rng, used_ids);
    // Independent ingestion jitter deliberately makes ingest order differ from event-time order.
    let jitter_ms = rng.below(120_000) as i64;
    event_details.insert("clock_profile".to_string(), "synthetic-v1".to_string());
    events.push(EventRecord {
        schema_version: EVENT_SCHEMA.to_string(),
        event_id: event_id.clone(),
        event_time_ms,
        ingest_time_ms: event_time_ms.saturating_add(jitter_ms),
        host_id: host_id.to_string(),
        principal_id: principal_id.to_string(),
        source: source.to_string(),
        event_type: event_type.to_string(),
        outcome: outcome.to_string(),
        details: event_details,
        synthetic: true,
    });
    event_id
}

fn background_event(
    index: usize,
    campaign_anchor_ms: i64,
    events: &mut Vec<EventRecord>,
    rng: &mut Rng,
    used_ids: &mut BTreeSet<String>,
) {
    let time_offset = rng.below(3_600_000) as i64 - 1_800_000;
    let event_time_ms = campaign_anchor_ms.saturating_add(time_offset);
    let host_number = 1 + rng.below(48);
    let host_id = format!("host-{host_number:03}");
    let principal = SYNTHETIC_PRINCIPALS[rng.below(SYNTHETIC_PRINCIPALS.len() as u64) as usize];
    match rng.below(7) {
        0 => {
            let address = format!("10.20.{}.{}", 1 + rng.below(20), 1 + rng.below(250));
            push_event(events, rng, used_ids, event_time_ms, &host_id, principal,
                "windows_security", "authentication", "success",
                details(&[("source_address", &address), ("logon_type", "interactive"), ("mfa", "satisfied")]));
        }
        1 => {
            let address = format!("10.20.{}.{}", 1 + rng.below(20), 1 + rng.below(250));
            push_event(events, rng, used_ids, event_time_ms, &host_id, principal,
                "windows_security", "authentication", "failure",
                details(&[("source_address", &address), ("status", "invalid_password"), ("logon_type", "network")]));
        }
        2 => {
            let images = ["explorer.exe", "chrome.exe", "powershell.exe", "svchost.exe", "backup-agent.exe"];
            let image = images[rng.below(images.len() as u64) as usize];
            let session = format!("sid-{:016x}", rng.next_u64());
            push_event(events, rng, used_ids, event_time_ms, &host_id, principal,
                "windows_sysmon", "process_start", "observed",
                details(&[("parent_image", "explorer.exe"), ("image", image), ("command_summary", "routine_user_or_service_activity"), ("session_id", &session)]));
        }
        3 => {
            let destinations = ["updates.example.invalid", "intranet.example.invalid", "telemetry.example.invalid", "docs.example.invalid"];
            let destination = destinations[rng.below(destinations.len() as u64) as usize];
            let port = if rng.below(5) == 0 { "80" } else { "443" };
            push_event(events, rng, used_ids, event_time_ms, &host_id, principal,
                "proxy_gateway", "network_connection", "allowed",
                details(&[("destination_domain", destination), ("destination_port", port), ("process_image", "chrome.exe"), ("bytes_out", "2048")]));
        }
        4 => {
            let paths = ["C:\\Users\\Public\\Documents\\report.csv", "C:\\Users\\Public\\Downloads\\installer.tmp", "D:\\Backups\\incremental.zip"];
            let path = paths[rng.below(paths.len() as u64) as usize];
            push_event(events, rng, used_ids, event_time_ms, &host_id, principal,
                "file_audit", "file_write", "completed",
                details(&[("file_path", path), ("file_bytes", "8192"), ("operation", "create")]));
        }
        5 => {
            let ticket_number = 100 + rng.below(900);
            let ticket = format!("CHG-{ticket_number}");
            push_event(events, rng, used_ids, event_time_ms, &host_id, principal,
                "change_audit", "administrative_change", "accepted",
                details(&[("change_ticket", &ticket), ("change_window", "approved"), ("scope", "routine_configuration")]));
        }
        _ => {
            let domains = ["portal.example.invalid", "repo.example.invalid", "status.example.invalid", "packages.example.invalid"];
            let domain = domains[rng.below(domains.len() as u64) as usize];
            push_event(events, rng, used_ids, event_time_ms, &host_id, principal,
                "dns_resolver", "dns_query", "answered",
                details(&[("query_name", domain), ("record_type", "A"), ("answer_profile", "synthetic_private_range")]));
        }
    }
    let _ = index; // Event IDs intentionally do not encode the generator-loop index.
}

fn generate_dataset(record_count: usize, seed: u64) -> Result<Dataset, String> {
    if !(MIN_RECORDS..=MAX_RECORDS).contains(&record_count) {
        return Err(format!(
            "record count must be between {MIN_RECORDS} and {MAX_RECORDS}"
        ));
    }
    let mut rng = Rng::new(seed);
    let mut events = Vec::with_capacity(record_count);
    let mut used_ids = BTreeSet::new();

    // Shift the whole environment first, so the planted incident does not become an isolated
    // time cluster when its clock changes. Background events share this shifted time anchor.
    let campaign_anchor_ms = BASE_TIME_MS.saturating_add(rng.below(7_200_000) as i64);

    for index in 0..(record_count - PLANTED_EVENT_COUNT) {
        background_event(index, campaign_anchor_ms, &mut events, &mut rng, &mut used_ids);
    }

    // Start the planted incident within the shifted background window.
    // Choose identities from the same naming pools as background activity; do not use a marker.
    let base = campaign_anchor_ms.saturating_add(600_000);
    let host = format!("host-{:03}", 1 + rng.below(48));
    let principal = SYNTHETIC_PRINCIPALS[rng.below(SYNTHETIC_PRINCIPALS.len() as u64) as usize].to_string();
    let session_id = format!("sid-{:016x}", rng.next_u64());
    let source_ip = format!("192.0.2.{}", 10 + rng.below(240)); // RFC 5737 documentation range.
    let external_ip = format!("198.51.100.{}", 10 + rng.below(240)); // RFC 5737 documentation range.

    let mut attack_ids = Vec::with_capacity(ATTACK_EVENT_COUNT);
    for offset in [0_i64, 10_000, 20_000] {
        let id = push_event(&mut events, &mut rng, &mut used_ids, base + offset, &host, &principal,
            "windows_security", "authentication", "failure",
            details(&[("source_address", source_ip.as_str()), ("status", "invalid_password"), ("logon_type", "network")]));
        attack_ids.push(id);
    }
    attack_ids.push(push_event(&mut events, &mut rng, &mut used_ids, base + 120_000, &host, &principal,
        "windows_security", "authentication", "success",
        details(&[("source_address", source_ip.as_str()), ("logon_type", "remote_interactive"), ("session_id", &session_id), ("device_posture", "not_reported")])));

    attack_ids.push(push_event(&mut events, &mut rng, &mut used_ids, base + 145_000, &host, &principal,
        "windows_sysmon", "process_start", "observed",
        details(&[("parent_image", "winword.exe"), ("image", "powershell.exe"), ("command_summary", "script_host_child_process"), ("session_id", &session_id), ("process_id", "4812")])));

    attack_ids.push(push_event(&mut events, &mut rng, &mut used_ids, base + 190_000, &host, &principal,
        "proxy_gateway", "network_connection", "allowed",
        details(&[("destination_address", external_ip.as_str()), ("destination_domain", "asset-review.example.invalid"), ("destination_port", "443"), ("process_image", "powershell.exe"), ("session_id", &session_id), ("bytes_out", "1280")])));

    attack_ids.push(push_event(&mut events, &mut rng, &mut used_ids, base + 240_000, &host, &principal,
        "file_audit", "file_write", "completed",
        details(&[("file_path", "C:\\Users\\Public\\cache\\report-bundle.tmp"), ("file_bytes", "1843200"), ("operation", "create"), ("session_id", &session_id)])));

    attack_ids.push(push_event(&mut events, &mut rng, &mut used_ids, base + 260_000, &host, &principal,
        "windows_sysmon", "process_start", "observed",
        details(&[("parent_image", "powershell.exe"), ("image", "tar.exe"), ("command_summary", "archive_creation"), ("session_id", &session_id), ("process_id", "4820")])));

    attack_ids.push(push_event(&mut events, &mut rng, &mut used_ids, base + 310_000, &host, &principal,
        "proxy_gateway", "network_transfer", "allowed",
        details(&[("destination_address", external_ip.as_str()), ("destination_domain", "asset-review.example.invalid"), ("destination_port", "443"), ("process_image", "tar.exe"), ("session_id", &session_id), ("bytes_out", "1761280")])));

    // Deliberately plausible benign alternatives: ordinary password typos, approved PowerShell
    // administration, and scheduled backup activity. They must not be scored as positives.
    let mut decoy_ids = Vec::with_capacity(BENIGN_DECOY_COUNT);
    // Put plausible benign activity inside the attack window so time clustering alone is insufficient.
    let decoy_base = base.saturating_add(30_000);
    for offset in [0_i64, 8_000, 16_000] {
        let id = push_event(&mut events, &mut rng, &mut used_ids, decoy_base + offset, "host-007", "j.smith",
            "windows_security", "authentication", "failure",
            details(&[("source_address", "10.99.4.18"), ("status", "invalid_password"), ("logon_type", "interactive")]));
        decoy_ids.push(id);
    }
    decoy_ids.push(push_event(&mut events, &mut rng, &mut used_ids, decoy_base + 60_000, "host-007", "j.smith",
        "windows_security", "authentication", "success",
        details(&[("source_address", "10.99.4.18"), ("logon_type", "interactive"), ("mfa", "satisfied"), ("device_posture", "compliant")])));

    decoy_ids.push(push_event(&mut events, &mut rng, &mut used_ids, decoy_base + 100_000, "host-012", "it.admin",
        "windows_sysmon", "process_start", "observed",
        details(&[("parent_image", "cmd.exe"), ("image", "powershell.exe"), ("command_summary", "Get-Service"), ("change_ticket", "CHG-1042"), ("change_window", "approved")])));

    decoy_ids.push(push_event(&mut events, &mut rng, &mut used_ids, decoy_base + 200_000, "host-025", "svc-backup",
        "file_audit", "file_write", "completed",
        details(&[("file_path", "D:\\Backups\\weekly-archive.zip"), ("file_bytes", "1761280"), ("operation", "create"), ("job_id", "BKP-2201")])));

    // Counterexample for the planted network_transfer: same event type, outcome,
    // process, and byte count, but an approved backup endpoint and benign job lineage.
    let benign_backup_session = format!("sid-{:016x}", rng.next_u64());
    decoy_ids.push(push_event(&mut events, &mut rng, &mut used_ids, decoy_base + 220_000, "host-025", "svc-backup",
        "proxy_gateway", "network_transfer", "allowed",
        details(&[
            ("destination_address", "192.0.2.40"),
            ("destination_domain", "backup.example.invalid"),
            ("destination_port", "443"),
            ("process_image", "tar.exe"),
            ("session_id", &benign_backup_session),
            ("bytes_out", "1761280"),
            ("transfer_profile", "approved_backup_endpoint"),
            ("job_id", "BKP-2201")
        ])));

    if events.len() != record_count || attack_ids.len() != ATTACK_EVENT_COUNT || decoy_ids.len() != BENIGN_DECOY_COUNT {
        return Err("internal fixture construction count mismatch".to_string());
    }
    events.sort_by(|a, b| (a.ingest_time_ms, &a.event_id).cmp(&(b.ingest_time_ms, &b.event_id)));
    let event_store = encode_jsonl(&events)?;
    let event_store_sha256 = digest(&event_store);

    let positives: BTreeSet<String> = attack_ids.iter().cloned().collect();
    let decoys: BTreeSet<String> = decoy_ids.iter().cloned().collect();
    let mut labels = Vec::with_capacity(events.len());
    for event in &events {
        let (label, rationale) = if positives.contains(&event.event_id) {
            ("positive", "member of the planted synthetic identity-to-egress causal sequence")
        } else if decoys.contains(&event.event_id) {
            ("negative", "plausible benign alternative with contextual evidence")
        } else {
            ("negative", "synthetic background event outside the planted causal sequence")
        };
        labels.push(EventLabel {
            event_id: event.event_id.clone(),
            label: label.to_string(),
            rationale: rationale.to_string(),
        });
    }

    let expected_relations = vec![
        ExpectedRelation { from_event_id: attack_ids[0].clone(), to_event_id: attack_ids[1].clone(), relation: "repeated_authentication_failure".to_string() },
        ExpectedRelation { from_event_id: attack_ids[1].clone(), to_event_id: attack_ids[2].clone(), relation: "repeated_authentication_failure".to_string() },
        ExpectedRelation { from_event_id: attack_ids[2].clone(), to_event_id: attack_ids[3].clone(), relation: "followed_by_successful_authentication".to_string() },
        ExpectedRelation { from_event_id: attack_ids[3].clone(), to_event_id: attack_ids[4].clone(), relation: "same_identity_and_session_process_start".to_string() },
        ExpectedRelation { from_event_id: attack_ids[4].clone(), to_event_id: attack_ids[5].clone(), relation: "process_to_external_connection".to_string() },
        ExpectedRelation { from_event_id: attack_ids[4].clone(), to_event_id: attack_ids[6].clone(), relation: "process_to_file_write".to_string() },
        ExpectedRelation { from_event_id: attack_ids[6].clone(), to_event_id: attack_ids[7].clone(), relation: "file_write_before_archive_process".to_string() },
        ExpectedRelation { from_event_id: attack_ids[7].clone(), to_event_id: attack_ids[8].clone(), relation: "archive_process_to_external_transfer".to_string() },
    ];

    let truth = GroundTruth {
        schema_version: TRUTH_SCHEMA.to_string(),
        scenario_id: SCENARIO_ID.to_string(),
        profile: "public-synthetic-smoke-v1".to_string(),
        seed,
        event_count: record_count,
        event_store_sha256: event_store_sha256.clone(),
        labels,
        expected_relations,
        claim_ceiling: "Synthetic event retrieval, correlation, and scoring conformance only; not real SOC readiness or real-world security.".to_string(),
    };

    let manifest = DatasetManifest {
        schema_version: MANIFEST_SCHEMA.to_string(),
        generator_version: GENERATOR_VERSION.to_string(),
        profile: "public-synthetic-smoke-v1".to_string(),
        scenario_id: SCENARIO_ID.to_string(),
        seed,
        record_count: record_count,
        event_store_sha256,
        event_store_utf8_bytes: event_store.len(),
        ordering: "ingest_time_ms_then_event_id; event_time_ms remains distinct".to_string(),
        qualification_state: "UNSCORED".to_string(),
        generator_source_commit: "UNPINNED_LOCAL_SOURCE".to_string(),
        source_commit_pinned: false,
        claim_ceiling: "Exact deterministic synthetic fixture identity only; not solver capability evidence.".to_string(),
    };
    Ok(Dataset { event_store, manifest, truth })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CampaignFamily {
    IdentityToEgress,
    ArtifactProvenance,
    CloudDataMovement,
    CrossDomainTrustChain,
    BenignOnlyControl,
}

impl CampaignFamily {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "identity-to-egress" => Ok(Self::IdentityToEgress),
            "artifact-provenance" => Ok(Self::ArtifactProvenance),
            "cloud-data-movement" => Ok(Self::CloudDataMovement),
            "cross-domain-trust-chain" => Ok(Self::CrossDomainTrustChain),
            "benign-only-control" => Ok(Self::BenignOnlyControl),
            _ => Err(format!(
                "unsupported --family {value}; expected identity-to-egress, artifact-provenance, cloud-data-movement, cross-domain-trust-chain, or benign-only-control"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::IdentityToEgress => "identity-to-egress",
            Self::ArtifactProvenance => "artifact-provenance",
            Self::CloudDataMovement => "cloud-data-movement",
            Self::CrossDomainTrustChain => "cross-domain-trust-chain",
            Self::BenignOnlyControl => "benign-only-control",
        }
    }
}

fn generate_artifact_provenance_dataset(record_count: usize, seed: u64) -> Result<Dataset, String> {
    const POSITIVE_COUNT: usize = 6;
    const BENIGN_LOOKALIKE_COUNT: usize = 6;
    const PLANTED_COUNT: usize = POSITIVE_COUNT + BENIGN_LOOKALIKE_COUNT;
    const SCENARIO: &str = "synthetic-artifact-lifecycle-v1";

    if !(MIN_RECORDS..=MAX_RECORDS).contains(&record_count) {
        return Err(format!(
            "record count must be between {MIN_RECORDS} and {MAX_RECORDS}"
        ));
    }

    let mut rng = Rng::new(seed);
    let mut events = Vec::with_capacity(record_count);
    let mut used_ids = BTreeSet::new();
    let campaign_anchor_ms = BASE_TIME_MS.saturating_add(rng.below(7_200_000) as i64);

    for index in 0..(record_count - PLANTED_COUNT) {
        background_event(index, campaign_anchor_ms, &mut events, &mut rng, &mut used_ids);
    }

    // A separate causal family: source revision, build, artifact bytes, signed
    // attestation, promotion, and observed deployment. The signature is valid but
    // binds a different digest from the artifact that is promoted.
    let base = campaign_anchor_ms.saturating_add(600_000);
    let build_host = format!("host-{:03}", 1 + rng.below(48));
    let build_principal = "svc-build";
    let source_commit = format!("commit-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);
    let build_id = format!("build-{:08x}", rng.next_u64() as u32);
    // Derive stable full-length SHA-256 identifiers from explicit synthetic byte strings;
    // do not left-pad 64-bit PRNG output and call it a digest.
    let artifact_digest = format!("sha256:{}", digest(format!("synthetic-artifact-bytes|{seed}|primary").as_bytes()));
    let mismatched_subject_digest = format!("sha256:{}", digest(format!("synthetic-artifact-bytes|{seed}|attestation-subject").as_bytes()));
    let benign_digest = format!("sha256:{}", digest(format!("synthetic-artifact-bytes|{seed}|benign").as_bytes()));
    let benign_commit = format!("commit-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);
    let benign_build = format!("build-{:08x}", rng.next_u64() as u32);

    let mut positive_ids = Vec::with_capacity(POSITIVE_COUNT);
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base, &build_host, build_principal,
        "source_control", "commit_checkout", "completed",
        details(&[
            ("repository", "service/catalog"),
            ("commit_id", &source_commit),
            ("branch", "release"),
            ("review_state", "pending"),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 30_000, &build_host, build_principal,
        "ci_audit", "build_job", "completed",
        details(&[
            ("build_id", &build_id),
            ("source_commit_id", &source_commit),
            ("workflow_revision", "build-workflow-17"),
            ("workflow_trust", "unverified_revision"),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 60_000, &build_host, build_principal,
        "artifact_registry", "artifact_publish", "published",
        details(&[
            ("build_id", &build_id),
            ("artifact_digest", &artifact_digest),
            ("source_commit_id", &source_commit),
            ("package", "service-catalog"),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 90_000, &build_host, "builder-prod",
        "attestation_verifier", "provenance_attestation", "signature_valid",
        details(&[
            ("subject_digest", &mismatched_subject_digest),
            ("source_commit_id", &source_commit),
            ("build_id", &build_id),
            ("predicate_type", "build-provenance"),
            ("signer", "builder-prod"),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 120_000, &build_host, "release-controller",
        "release_audit", "artifact_promotion", "promoted",
        details(&[
            ("artifact_digest", &artifact_digest),
            ("attestation_subject_digest", &mismatched_subject_digest),
            ("policy_decision", "override"),
            ("change_ticket", "CHG-2317"),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 150_000, &build_host, "deploy-controller",
        "deployment_audit", "deployment", "succeeded",
        details(&[
            ("artifact_digest", &artifact_digest),
            ("environment", "production"),
            ("release_channel", "stable"),
            ("change_ticket", "CHG-2317"),
        ]),
    ));

    // Benign lookalike: a valid signature whose subject digest matches the promoted
    // and deployed artifact. It overlaps the suspicious chain in event types and time.
    let benign_ids = vec![
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 5_000, "host-018", "svc-build",
            "source_control", "commit_checkout", "completed",
            details(&[
                ("repository", "service/catalog"),
                ("commit_id", &benign_commit),
                ("branch", "main"),
                ("review_state", "approved"),
            ]),
        ),
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 35_000, "host-018", "svc-build",
            "ci_audit", "build_job", "completed",
            details(&[
                ("build_id", &benign_build),
                ("source_commit_id", &benign_commit),
                ("workflow_revision", "build-workflow-12"),
                ("workflow_trust", "reviewed"),
            ]),
        ),
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 65_000, "host-018", "svc-build",
            "artifact_registry", "artifact_publish", "published",
            details(&[
                ("build_id", &benign_build),
                ("artifact_digest", &benign_digest),
                ("source_commit_id", &benign_commit),
                ("package", "service-catalog"),
            ]),
        ),
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 100_000, "host-018", "builder-prod",
            "attestation_verifier", "provenance_attestation", "signature_valid",
            details(&[
                ("subject_digest", &benign_digest),
                ("source_commit_id", &benign_commit),
                ("build_id", &benign_build),
                ("predicate_type", "build-provenance"),
                ("signer", "builder-prod"),
            ]),
        ),
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 125_000, "host-018", "release-controller",
            "release_audit", "artifact_promotion", "promoted",
            details(&[
                ("artifact_digest", &benign_digest),
                ("attestation_subject_digest", &benign_digest),
                ("policy_decision", "pass"),
                ("change_ticket", "CHG-1042"),
            ]),
        ),
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 155_000, "host-018", "deploy-controller",
            "deployment_audit", "deployment", "succeeded",
            details(&[
                ("artifact_digest", &benign_digest),
                ("environment", "staging"),
                ("release_channel", "stable"),
                ("change_ticket", "CHG-1042"),
            ]),
        ),
    ];

    if events.len() != record_count
        || positive_ids.len() != POSITIVE_COUNT
        || benign_ids.len() != BENIGN_LOOKALIKE_COUNT
    {
        return Err("internal artifact-provenance fixture construction count mismatch".to_string());
    }

    events.sort_by(|a, b| (a.ingest_time_ms, &a.event_id).cmp(&(b.ingest_time_ms, &b.event_id)));
    let event_store = encode_jsonl(&events)?;
    let event_store_sha256 = digest(&event_store);

    let positives: BTreeSet<String> = positive_ids.iter().cloned().collect();
    let benign: BTreeSet<String> = benign_ids.iter().cloned().collect();
    let mut labels = Vec::with_capacity(events.len());
    for event in &events {
        let (label, rationale) = if positives.contains(&event.event_id) {
            ("positive", "member of the synthetic artifact provenance mismatch causal sequence")
        } else if benign.contains(&event.event_id) {
            ("negative", "benign matched-digest source, build, artifact, attestation, promotion, and deployment lookalike")
        } else {
            ("negative", "synthetic background event outside the planted causal sequence")
        };
        labels.push(EventLabel {
            event_id: event.event_id.clone(),
            label: label.to_string(),
            rationale: rationale.to_string(),
        });
    }

    let expected_relations = vec![
        ExpectedRelation {
            from_event_id: positive_ids[0].clone(),
            to_event_id: positive_ids[1].clone(),
            relation: "source_revision_consumed_by_build".to_string(),
        },
        ExpectedRelation {
            from_event_id: positive_ids[1].clone(),
            to_event_id: positive_ids[2].clone(),
            relation: "build_produced_artifact_digest".to_string(),
        },
        ExpectedRelation {
            from_event_id: positive_ids[2].clone(),
            to_event_id: positive_ids[3].clone(),
            relation: "attestation_subject_digest_mismatch".to_string(),
        },
        ExpectedRelation {
            from_event_id: positive_ids[3].clone(),
            to_event_id: positive_ids[4].clone(),
            relation: "promotion_followed_mismatched_attestation".to_string(),
        },
        ExpectedRelation {
            from_event_id: positive_ids[4].clone(),
            to_event_id: positive_ids[5].clone(),
            relation: "promoted_artifact_observed_in_deployment".to_string(),
        },
    ];

    let truth = GroundTruth {
        schema_version: TRUTH_SCHEMA.to_string(),
        scenario_id: SCENARIO.to_string(),
        profile: "public-synthetic-smoke-v1".to_string(),
        seed,
        event_count: record_count,
        event_store_sha256: event_store_sha256.clone(),
        labels,
        expected_relations,
        claim_ceiling: "Synthetic build-to-artifact provenance consistency and causal correlation only; not real supply-chain assurance or real-world security.".to_string(),
    };

    let manifest = DatasetManifest {
        schema_version: MANIFEST_SCHEMA.to_string(),
        generator_version: GENERATOR_VERSION.to_string(),
        profile: "public-synthetic-smoke-v1".to_string(),
        scenario_id: SCENARIO.to_string(),
        seed,
        record_count,
        event_store_sha256,
        event_store_utf8_bytes: event_store.len(),
        ordering: "ingest_time_ms_then_event_id; event_time_ms remains distinct".to_string(),
        qualification_state: "UNSCORED".to_string(),
        generator_source_commit: "UNPINNED_LOCAL_SOURCE".to_string(),
        source_commit_pinned: false,
        claim_ceiling: "Exact deterministic synthetic artifact-provenance smoke fixture only; not solver capability evidence.".to_string(),
    };

    Ok(Dataset { event_store, manifest, truth })
}

fn generate_cloud_data_movement_dataset(record_count: usize, seed: u64) -> Result<Dataset, String> {
    const POSITIVE_COUNT: usize = 6;
    const BENIGN_LOOKALIKE_COUNT: usize = 6;
    const PLANTED_COUNT: usize = POSITIVE_COUNT + BENIGN_LOOKALIKE_COUNT;
    const SCENARIO: &str = "synthetic-cloud-data-movement-v1";

    if !(MIN_RECORDS..=MAX_RECORDS).contains(&record_count) {
        return Err(format!(
            "record count must be between {MIN_RECORDS} and {MAX_RECORDS}"
        ));
    }

    let mut rng = Rng::new(seed);
    let mut events = Vec::with_capacity(record_count);
    let mut used_ids = BTreeSet::new();
    let campaign_anchor_ms = BASE_TIME_MS.saturating_add(rng.below(7_200_000) as i64);

    for index in 0..(record_count - PLANTED_COUNT) {
        background_event(index, campaign_anchor_ms, &mut events, &mut rng, &mut used_ids);
    }

    // The granted role is scoped to one tenant path, but later storage reads target
    // another path; approval is absent. All operations are synthetic and offline.
    let base = campaign_anchor_ms.saturating_add(600_000);
    let principal_index = rng.below(SYNTHETIC_PRINCIPALS.len() as u64) as usize;
    let principal = SYNTHETIC_PRINCIPALS[principal_index];
    let host = format!("host-{:03}", 1 + rng.below(48));
    let object_id = format!("sp-{:016x}", rng.next_u64());
    let session_id = format!("cloud-session-{:016x}", rng.next_u64());
    let token_id = format!("token-{:016x}", rng.next_u64());
    let export_id = format!("export-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);
    let object_count = 28_104 + rng.below(5_000);
    let bytes_read = 48_000_000_000_u64 + rng.below(12_000_000_000);
    let source_scope = "tenant-beta/telemetry";
    let target_scope = "tenant-alpha/finance-prod";
    let destination_ip = format!("198.51.100.{}", 10 + rng.below(240));

    let mut positive_ids = Vec::with_capacity(POSITIVE_COUNT);
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base, &host, principal,
        "cloud_identity", "role_assignment", "granted",
        details(&[
            ("principal_object_id", &object_id),
            ("cloud_role", "StorageDataReader"),
            ("scope", source_scope),
            ("approval_state", "missing"),
            ("change_ticket", "none"),
            ("grant_source", "automation"),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 25_000, &host, principal,
        "cloud_identity", "token_issue", "issued",
        details(&[
            ("principal_object_id", &object_id),
            ("token_id", &token_id),
            ("session_id", &session_id),
            ("audience", "storage-api"),
            ("effective_scope", source_scope),
        ]),
    ));
    let positive_request_id = format!("req-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 50_000, &host, principal,
        "cloud_storage", "storage_list", "success",
        details(&[
            ("principal_object_id", &object_id),
            ("session_id", &session_id),
            ("resource_scope", target_scope),
            ("container", "finance-prod"),
            ("request_id", &positive_request_id),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 80_000, &host, principal,
        "cloud_storage", "object_read_batch", "success",
        details(&[
            ("principal_object_id", &object_id),
            ("session_id", &session_id),
            ("resource_scope", target_scope),
            ("object_count", &object_count.to_string()),
            ("bytes_read", &bytes_read.to_string()),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 110_000, &host, principal,
        "cloud_audit", "export_job", "created",
        details(&[
            ("principal_object_id", &object_id),
            ("session_id", &session_id),
            ("export_id", &export_id),
            ("resource_scope", target_scope),
            ("object_count", &object_count.to_string()),
            ("bytes_written", &bytes_read.to_string()),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 140_000, &host, principal,
        "cloud_egress", "data_transfer", "allowed",
        details(&[
            ("principal_object_id", &object_id),
            ("session_id", &session_id),
            ("export_id", &export_id),
            ("destination_address", &destination_ip),
            ("destination_profile", "unapproved_external"),
            ("bytes_out", &bytes_read.to_string()),
        ]),
    ));

    // Benign matched chain: same event types/outcomes, but the approved role scope,
    // read scope, export and destination are consistent with a routine backup.
    let benign_principal = SYNTHETIC_PRINCIPALS[(principal_index + 5) % SYNTHETIC_PRINCIPALS.len()];
    let benign_host = format!("host-{:03}", 1 + rng.below(48));
    let benign_object_id = format!("sp-{:016x}", rng.next_u64());
    let benign_session = format!("cloud-session-{:016x}", rng.next_u64());
    let benign_token = format!("token-{:016x}", rng.next_u64());
    let benign_export = format!("export-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);
    let benign_count = 4_000 + rng.below(2_000);
    let benign_bytes = 1_000_000_000_u64 + rng.below(2_000_000_000);
    let benign_scope = "tenant-alpha/backup";
    let benign_request_id = format!("req-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);

    let benign_ids = vec![
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 5_000, &benign_host, benign_principal,
            "cloud_identity", "role_assignment", "granted",
            details(&[
                ("principal_object_id", &benign_object_id),
                ("cloud_role", "StorageDataReader"),
                ("scope", benign_scope),
                ("approval_state", "approved"),
                ("change_ticket", "CHG-1042"),
                ("grant_source", "change_management"),
            ]),
        ),
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 30_000, &benign_host, benign_principal,
            "cloud_identity", "token_issue", "issued",
            details(&[
                ("principal_object_id", &benign_object_id),
                ("token_id", &benign_token),
                ("session_id", &benign_session),
                ("audience", "storage-api"),
                ("effective_scope", benign_scope),
            ]),
        ),
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 55_000, &benign_host, benign_principal,
            "cloud_storage", "storage_list", "success",
            details(&[
                ("principal_object_id", &benign_object_id),
                ("session_id", &benign_session),
                ("resource_scope", benign_scope),
                ("container", "backup"),
                ("request_id", &benign_request_id),
            ]),
        ),
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 85_000, &benign_host, benign_principal,
            "cloud_storage", "object_read_batch", "success",
            details(&[
                ("principal_object_id", &benign_object_id),
                ("session_id", &benign_session),
                ("resource_scope", benign_scope),
                ("object_count", &benign_count.to_string()),
                ("bytes_read", &benign_bytes.to_string()),
            ]),
        ),
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 115_000, &benign_host, benign_principal,
            "cloud_audit", "export_job", "created",
            details(&[
                ("principal_object_id", &benign_object_id),
                ("session_id", &benign_session),
                ("export_id", &benign_export),
                ("resource_scope", benign_scope),
                ("object_count", &benign_count.to_string()),
                ("bytes_written", &benign_bytes.to_string()),
            ]),
        ),
        push_event(
            &mut events, &mut rng, &mut used_ids, base + 145_000, &benign_host, benign_principal,
            "cloud_egress", "data_transfer", "allowed",
            details(&[
                ("principal_object_id", &benign_object_id),
                ("session_id", &benign_session),
                ("export_id", &benign_export),
                ("destination_address", "192.0.2.40"),
                ("destination_profile", "approved_backup_endpoint"),
                ("bytes_out", &benign_bytes.to_string()),
            ]),
        ),
    ];

    if events.len() != record_count
        || positive_ids.len() != POSITIVE_COUNT
        || benign_ids.len() != BENIGN_LOOKALIKE_COUNT
    {
        return Err("internal cloud data movement fixture construction count mismatch".to_string());
    }

    events.sort_by(|a, b| (a.ingest_time_ms, &a.event_id).cmp(&(b.ingest_time_ms, &b.event_id)));
    let event_store = encode_jsonl(&events)?;
    let event_store_sha256 = digest(&event_store);
    let positives: BTreeSet<String> = positive_ids.iter().cloned().collect();
    let benign: BTreeSet<String> = benign_ids.iter().cloned().collect();
    let mut labels = Vec::with_capacity(events.len());
    for event in &events {
        let (label, rationale) = if positives.contains(&event.event_id) {
            ("positive", "member of the synthetic cloud authorization-scope mismatch and data-movement chain")
        } else if benign.contains(&event.event_id) {
            ("negative", "benign approved, scope-consistent cloud backup lifecycle lookalike")
        } else {
            ("negative", "synthetic background event outside the planted causal sequence")
        };
        labels.push(EventLabel {
            event_id: event.event_id.clone(),
            label: label.to_string(),
            rationale: rationale.to_string(),
        });
    }

    let expected_relations = vec![
        ExpectedRelation {
            from_event_id: positive_ids[0].clone(),
            to_event_id: positive_ids[1].clone(),
            relation: "role_grant_issued_token".to_string(),
        },
        ExpectedRelation {
            from_event_id: positive_ids[1].clone(),
            to_event_id: positive_ids[2].clone(),
            relation: "token_used_for_out_of_scope_storage_list".to_string(),
        },
        ExpectedRelation {
            from_event_id: positive_ids[2].clone(),
            to_event_id: positive_ids[3].clone(),
            relation: "enumeration_followed_by_bulk_object_reads".to_string(),
        },
        ExpectedRelation {
            from_event_id: positive_ids[3].clone(),
            to_event_id: positive_ids[4].clone(),
            relation: "bulk_reads_collected_into_export".to_string(),
        },
        ExpectedRelation {
            from_event_id: positive_ids[4].clone(),
            to_event_id: positive_ids[5].clone(),
            relation: "export_transferred_to_unapproved_destination".to_string(),
        },
    ];

    let truth = GroundTruth {
        schema_version: TRUTH_SCHEMA.to_string(),
        scenario_id: SCENARIO.to_string(),
        profile: "public-synthetic-smoke-v1".to_string(),
        seed,
        event_count: record_count,
        event_store_sha256: event_store_sha256.clone(),
        labels,
        expected_relations,
        claim_ceiling: "Synthetic cloud identity authorization-scope and data-movement correlation only; not real cloud security assurance or real-world security.".to_string(),
    };
    let manifest = DatasetManifest {
        schema_version: MANIFEST_SCHEMA.to_string(),
        generator_version: GENERATOR_VERSION.to_string(),
        profile: "public-synthetic-smoke-v1".to_string(),
        scenario_id: SCENARIO.to_string(),
        seed,
        record_count,
        event_store_sha256,
        event_store_utf8_bytes: event_store.len(),
        ordering: "ingest_time_ms_then_event_id; event_time_ms remains distinct".to_string(),
        qualification_state: "UNSCORED".to_string(),
        generator_source_commit: "UNPINNED_LOCAL_SOURCE".to_string(),
        source_commit_pinned: false,
        claim_ceiling: "Exact deterministic cloud data-movement smoke fixture only; not solver capability evidence.".to_string(),
    };
    Ok(Dataset { event_store, manifest, truth })
}

fn generate_cross_domain_trust_chain_dataset(record_count: usize, seed: u64) -> Result<Dataset, String> {
    const POSITIVE_COUNT: usize = 12;
    const BENIGN_LOOKALIKE_COUNT: usize = 12;
    const PLANTED_COUNT: usize = POSITIVE_COUNT + BENIGN_LOOKALIKE_COUNT;
    const SCENARIO: &str = "synthetic-cross-domain-trust-chain-v1";

    if !(MIN_RECORDS..=MAX_RECORDS).contains(&record_count) {
        return Err(format!(
            "record count must be between {MIN_RECORDS} and {MAX_RECORDS}"
        ));
    }

    let mut rng = Rng::new(seed);
    let mut events = Vec::with_capacity(record_count);
    let mut used_ids = BTreeSet::new();
    let campaign_anchor_ms = BASE_TIME_MS.saturating_add(rng.below(7_200_000) as i64);

    for index in 0..(record_count - PLANTED_COUNT) {
        background_event(index, campaign_anchor_ms, &mut events, &mut rng, &mut used_ids);
    }

    // Cross-domain positive chain: CI identity, source revision, build, artifact,
    // attestation, promotion, deployment, cloud role/token, out-of-scope reads, export.
    // Every operation is synthetic; the intended signal is in joined evidence, not a
    // unique event type. The benign mirror uses the same types/outcomes with valid joins.
    let base = campaign_anchor_ms.saturating_add(600_000);
    let principal_index = rng.below(SYNTHETIC_PRINCIPALS.len() as u64) as usize;
    let principal = SYNTHETIC_PRINCIPALS[principal_index];
    let host = format!("host-{:03}", 1 + rng.below(48));
    let principal_object = format!("sp-{:016x}", rng.next_u64());
    let commit_id = format!("commit-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);
    let build_id = format!("build-{:08x}", rng.next_u64() as u32);
    let artifact_digest = format!("sha256:{}", digest(format!("cross-domain-artifact|{seed}|primary").as_bytes()));
    let attestation_digest = format!("sha256:{}", digest(format!("cross-domain-artifact|{seed}|attestation-subject").as_bytes()));
    let session_id = format!("cloud-session-{:016x}", rng.next_u64());
    let token_id = format!("token-{:016x}", rng.next_u64());
    let export_id = format!("export-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);
    let object_count = 18_000 + rng.below(8_000);
    let bytes_read = 32_000_000_000_u64 + rng.below(20_000_000_000);
    let assigned_scope = "tenant-beta/telemetry";
    let accessed_scope = "tenant-alpha/finance-prod";
    let outside_ip = format!("198.51.100.{}", 10 + rng.below(240));

    let mut positive_ids = Vec::with_capacity(POSITIVE_COUNT);
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 0, &host, principal,
        "source_control", "commit_checkout", "completed",
        details(&[
            ("repository", "service/catalog"),
            ("commit_id", &commit_id),
            ("branch", "release"),
            ("review_state", "pending"),
            ("principal_object_id", &principal_object),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 20_000, &host, principal,
        "ci_audit", "build_job", "completed",
        details(&[
            ("build_id", &build_id),
            ("source_commit_id", &commit_id),
            ("workflow_revision", "build-workflow-unverified"),
            ("workflow_trust", "unverified_revision"),
            ("principal_object_id", &principal_object),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 40_000, &host, principal,
        "artifact_registry", "artifact_publish", "published",
        details(&[
            ("build_id", &build_id),
            ("artifact_digest", &artifact_digest),
            ("source_commit_id", &commit_id),
            ("package", "service-catalog"),
            ("principal_object_id", &principal_object),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 60_000, &host, "builder-prod",
        "attestation_verifier", "provenance_attestation", "signature_valid",
        details(&[
            ("subject_digest", &attestation_digest),
            ("source_commit_id", &commit_id),
            ("build_id", &build_id),
            ("predicate_type", "build-provenance"),
            ("signer", "builder-prod"),
            ("principal_object_id", &principal_object),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 80_000, &host, "release-controller",
        "release_audit", "artifact_promotion", "promoted",
        details(&[
            ("artifact_digest", &artifact_digest),
            ("attestation_subject_digest", &attestation_digest),
            ("policy_decision", "override"),
            ("change_ticket", "none"),
            ("principal_object_id", &principal_object),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 100_000, &host, "deploy-controller",
        "deployment_audit", "deployment", "succeeded",
        details(&[
            ("artifact_digest", &artifact_digest),
            ("environment", "production"),
            ("release_channel", "stable"),
            ("principal_object_id", &principal_object),
            ("workload_identity", "enabled"),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 120_000, &host, principal,
        "cloud_identity", "role_assignment", "granted",
        details(&[
            ("principal_object_id", &principal_object),
            ("cloud_role", "StorageDataReader"),
            ("scope", assigned_scope),
            ("approval_state", "missing"),
            ("change_ticket", "none"),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 140_000, &host, principal,
        "cloud_identity", "token_issue", "issued",
        details(&[
            ("principal_object_id", &principal_object),
            ("token_id", &token_id),
            ("session_id", &session_id),
            ("audience", "storage-api"),
            ("effective_scope", assigned_scope),
        ]),
    ));
    let positive_request_id = format!("req-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 160_000, &host, principal,
        "cloud_storage", "storage_list", "success",
        details(&[
            ("principal_object_id", &principal_object),
            ("session_id", &session_id),
            ("resource_scope", accessed_scope),
            ("container", "finance-prod"),
            ("request_id", &positive_request_id),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 180_000, &host, principal,
        "cloud_storage", "object_read_batch", "success",
        details(&[
            ("principal_object_id", &principal_object),
            ("session_id", &session_id),
            ("resource_scope", accessed_scope),
            ("object_count", &object_count.to_string()),
            ("bytes_read", &bytes_read.to_string()),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 200_000, &host, principal,
        "cloud_audit", "export_job", "created",
        details(&[
            ("principal_object_id", &principal_object),
            ("session_id", &session_id),
            ("export_id", &export_id),
            ("resource_scope", accessed_scope),
            ("object_count", &object_count.to_string()),
            ("bytes_written", &bytes_read.to_string()),
        ]),
    ));
    positive_ids.push(push_event(
        &mut events, &mut rng, &mut used_ids, base + 220_000, &host, principal,
        "cloud_egress", "data_transfer", "allowed",
        details(&[
            ("principal_object_id", &principal_object),
            ("session_id", &session_id),
            ("export_id", &export_id),
            ("destination_address", &outside_ip),
            ("destination_profile", "unapproved_external"),
            ("bytes_out", &bytes_read.to_string()),
        ]),
    ));

    // Benign cross-domain mirror: reviewed build, matching attestation, normal
    // promotion/deployment, approved role, scope-consistent reads and backup egress.
    let benign_principal = SYNTHETIC_PRINCIPALS[(principal_index + 5) % SYNTHETIC_PRINCIPALS.len()];
    let benign_host = format!("host-{:03}", 1 + rng.below(48));
    let benign_object = format!("sp-{:016x}", rng.next_u64());
    let benign_commit = format!("commit-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);
    let benign_build = format!("build-{:08x}", rng.next_u64() as u32);
    let benign_digest = format!("sha256:{}", digest(format!("cross-domain-artifact|{seed}|benign").as_bytes()));
    let benign_session = format!("cloud-session-{:016x}", rng.next_u64());
    let benign_token = format!("token-{:016x}", rng.next_u64());
    let benign_export = format!("export-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);
    let benign_count = 3_000 + rng.below(2_000);
    let benign_bytes = 700_000_000_u64 + rng.below(1_500_000_000);
    let benign_scope = "tenant-alpha/backup";
    let benign_request_id = format!("req-{:012x}", rng.next_u64() & 0x0000_ffff_ffff_ffff);

    let benign_ids = vec![
        push_event(&mut events, &mut rng, &mut used_ids, base + 5_000, &benign_host, benign_principal,
            "source_control", "commit_checkout", "completed",
            details(&[
                ("repository", "service/catalog"),
                ("commit_id", &benign_commit),
                ("branch", "main"),
                ("review_state", "approved"),
                ("principal_object_id", &benign_object),
            ])),
        push_event(&mut events, &mut rng, &mut used_ids, base + 25_000, &benign_host, benign_principal,
            "ci_audit", "build_job", "completed",
            details(&[
                ("build_id", &benign_build),
                ("source_commit_id", &benign_commit),
                ("workflow_revision", "build-workflow-reviewed"),
                ("workflow_trust", "reviewed"),
                ("principal_object_id", &benign_object),
            ])),
        push_event(&mut events, &mut rng, &mut used_ids, base + 45_000, &benign_host, benign_principal,
            "artifact_registry", "artifact_publish", "published",
            details(&[
                ("build_id", &benign_build),
                ("artifact_digest", &benign_digest),
                ("source_commit_id", &benign_commit),
                ("package", "service-catalog"),
                ("principal_object_id", &benign_object),
            ])),
        push_event(&mut events, &mut rng, &mut used_ids, base + 65_000, &benign_host, "builder-prod",
            "attestation_verifier", "provenance_attestation", "signature_valid",
            details(&[
                ("subject_digest", &benign_digest),
                ("source_commit_id", &benign_commit),
                ("build_id", &benign_build),
                ("predicate_type", "build-provenance"),
                ("signer", "builder-prod"),
                ("principal_object_id", &benign_object),
            ])),
        push_event(&mut events, &mut rng, &mut used_ids, base + 85_000, &benign_host, "release-controller",
            "release_audit", "artifact_promotion", "promoted",
            details(&[
                ("artifact_digest", &benign_digest),
                ("attestation_subject_digest", &benign_digest),
                ("policy_decision", "pass"),
                ("change_ticket", "CHG-1042"),
                ("principal_object_id", &benign_object),
            ])),
        push_event(&mut events, &mut rng, &mut used_ids, base + 105_000, &benign_host, "deploy-controller",
            "deployment_audit", "deployment", "succeeded",
            details(&[
                ("artifact_digest", &benign_digest),
                ("environment", "staging"),
                ("release_channel", "stable"),
                ("principal_object_id", &benign_object),
                ("workload_identity", "enabled"),
            ])),
        push_event(&mut events, &mut rng, &mut used_ids, base + 125_000, &benign_host, benign_principal,
            "cloud_identity", "role_assignment", "granted",
            details(&[
                ("principal_object_id", &benign_object),
                ("cloud_role", "StorageDataReader"),
                ("scope", benign_scope),
                ("approval_state", "approved"),
                ("change_ticket", "CHG-1042"),
            ])),
        push_event(&mut events, &mut rng, &mut used_ids, base + 145_000, &benign_host, benign_principal,
            "cloud_identity", "token_issue", "issued",
            details(&[
                ("principal_object_id", &benign_object),
                ("token_id", &benign_token),
                ("session_id", &benign_session),
                ("audience", "storage-api"),
                ("effective_scope", benign_scope),
            ])),
        push_event(&mut events, &mut rng, &mut used_ids, base + 165_000, &benign_host, benign_principal,
            "cloud_storage", "storage_list", "success",
            details(&[
                ("principal_object_id", &benign_object),
                ("session_id", &benign_session),
                ("resource_scope", benign_scope),
                ("container", "backup"),
                ("request_id", &benign_request_id),
            ])),
        push_event(&mut events, &mut rng, &mut used_ids, base + 185_000, &benign_host, benign_principal,
            "cloud_storage", "object_read_batch", "success",
            details(&[
                ("principal_object_id", &benign_object),
                ("session_id", &benign_session),
                ("resource_scope", benign_scope),
                ("object_count", &benign_count.to_string()),
                ("bytes_read", &benign_bytes.to_string()),
            ])),
        push_event(&mut events, &mut rng, &mut used_ids, base + 205_000, &benign_host, benign_principal,
            "cloud_audit", "export_job", "created",
            details(&[
                ("principal_object_id", &benign_object),
                ("session_id", &benign_session),
                ("export_id", &benign_export),
                ("resource_scope", benign_scope),
                ("object_count", &benign_count.to_string()),
                ("bytes_written", &benign_bytes.to_string()),
            ])),
        push_event(&mut events, &mut rng, &mut used_ids, base + 225_000, &benign_host, benign_principal,
            "cloud_egress", "data_transfer", "allowed",
            details(&[
                ("principal_object_id", &benign_object),
                ("session_id", &benign_session),
                ("export_id", &benign_export),
                ("destination_address", "192.0.2.40"),
                ("destination_profile", "approved_backup_endpoint"),
                ("bytes_out", &benign_bytes.to_string()),
            ])),
    ];

    if events.len() != record_count
        || positive_ids.len() != POSITIVE_COUNT
        || benign_ids.len() != BENIGN_LOOKALIKE_COUNT
    {
        return Err("internal cross-domain trust-chain fixture construction count mismatch".to_string());
    }
    events.sort_by(|a, b| (a.ingest_time_ms, &a.event_id).cmp(&(b.ingest_time_ms, &b.event_id)));
    let event_store = encode_jsonl(&events)?;
    let event_store_sha256 = digest(&event_store);
    let positives: BTreeSet<String> = positive_ids.iter().cloned().collect();
    let benign: BTreeSet<String> = benign_ids.iter().cloned().collect();
    let mut labels = Vec::with_capacity(events.len());
    for event in &events {
        let (label, rationale) = if positives.contains(&event.event_id) {
            ("positive", "member of the synthetic identity/build/deployment/cloud authorization causal chain")
        } else if benign.contains(&event.event_id) {
            ("negative", "benign reviewed build with matching attestation and scope-consistent approved cloud access")
        } else {
            ("negative", "synthetic background event outside the planted cross-domain causal chain")
        };
        labels.push(EventLabel {
            event_id: event.event_id.clone(),
            label: label.to_string(),
            rationale: rationale.to_string(),
        });
    }

    let expected_relations = vec![
        ExpectedRelation { from_event_id: positive_ids[0].clone(), to_event_id: positive_ids[1].clone(), relation: "identity_used_for_source_build".to_string() },
        ExpectedRelation { from_event_id: positive_ids[1].clone(), to_event_id: positive_ids[2].clone(), relation: "build_published_artifact".to_string() },
        ExpectedRelation { from_event_id: positive_ids[2].clone(), to_event_id: positive_ids[3].clone(), relation: "attestation_subject_digest_mismatch".to_string() },
        ExpectedRelation { from_event_id: positive_ids[3].clone(), to_event_id: positive_ids[4].clone(), relation: "mismatched_artifact_promoted".to_string() },
        ExpectedRelation { from_event_id: positive_ids[4].clone(), to_event_id: positive_ids[5].clone(), relation: "artifact_deployed_with_workload_identity".to_string() },
        ExpectedRelation { from_event_id: positive_ids[5].clone(), to_event_id: positive_ids[6].clone(), relation: "deployed_identity_granted_cloud_role".to_string() },
        ExpectedRelation { from_event_id: positive_ids[6].clone(), to_event_id: positive_ids[7].clone(), relation: "role_grant_issued_token".to_string() },
        ExpectedRelation { from_event_id: positive_ids[7].clone(), to_event_id: positive_ids[8].clone(), relation: "token_used_outside_assigned_scope".to_string() },
        ExpectedRelation { from_event_id: positive_ids[8].clone(), to_event_id: positive_ids[9].clone(), relation: "storage_enumeration_preceded_bulk_reads".to_string() },
        ExpectedRelation { from_event_id: positive_ids[9].clone(), to_event_id: positive_ids[10].clone(), relation: "bulk_reads_collected_for_export".to_string() },
        ExpectedRelation { from_event_id: positive_ids[10].clone(), to_event_id: positive_ids[11].clone(), relation: "export_transferred_to_unapproved_destination".to_string() },
    ];

    let truth = GroundTruth {
        schema_version: TRUTH_SCHEMA.to_string(),
        scenario_id: SCENARIO.to_string(),
        profile: "public-synthetic-smoke-v1".to_string(),
        seed,
        event_count: record_count,
        event_store_sha256: event_store_sha256.clone(),
        labels,
        expected_relations,
        claim_ceiling: "Synthetic cross-domain identity, build provenance, deployment, cloud scope, and causal correlation only; not production assurance or real-world security.".to_string(),
    };
    let manifest = DatasetManifest {
        schema_version: MANIFEST_SCHEMA.to_string(),
        generator_version: GENERATOR_VERSION.to_string(),
        profile: "public-synthetic-smoke-v1".to_string(),
        scenario_id: SCENARIO.to_string(),
        seed,
        record_count,
        event_store_sha256,
        event_store_utf8_bytes: event_store.len(),
        ordering: "ingest_time_ms_then_event_id; event_time_ms remains distinct".to_string(),
        qualification_state: "UNSCORED".to_string(),
        generator_source_commit: "UNPINNED_LOCAL_SOURCE".to_string(),
        source_commit_pinned: false,
        claim_ceiling: "Synthetic cross-domain trust-chain smoke fixture only; not solver capability evidence.".to_string(),
    };
    Ok(Dataset { event_store, manifest, truth })
}

fn extract_benign_chain_events(dataset: Dataset) -> Result<Vec<EventRecord>, String> {
    let benign_ids: BTreeSet<String> = dataset.truth.labels.iter()
        .filter(|label| label.label == "negative" && label.rationale.to_lowercase().contains("benign"))
        .map(|label| label.event_id.clone())
        .collect();
    let text = std::str::from_utf8(&dataset.event_store)
        .map_err(|e| format!("benign fixture event store is not UTF-8: {e}"))?;
    let mut events = Vec::new();
    for line in text.lines() {
        let event: EventRecord = serde_json::from_str(line)
            .map_err(|e| format!("cannot decode benign event while composing control corpus: {e}"))?;
        if benign_ids.contains(&event.event_id) {
            events.push(event);
        }
    }
    if events.is_empty() {
        return Err("family did not provide any benign-only chain events".to_string());
    }
    Ok(events)
}

fn generate_benign_only_control_dataset(record_count: usize, seed: u64) -> Result<Dataset, String> {
    const SCENARIO: &str = "synthetic-benign-only-control-v1";
    if !(MIN_RECORDS..=MAX_RECORDS).contains(&record_count) {
        return Err(format!(
            "record count must be between {MIN_RECORDS} and {MAX_RECORDS}"
        ));
    }

    // Compose only oracle-designated benign chain records from each family. Relabeling
    // happens only after filtering by the hidden oracle, and original event IDs are
    // replaced with fresh deterministic IDs. No planted positive event is included.
    let source_datasets = [
        generate_dataset(64, seed ^ 0x11a2_b3c4_d5e6_f701)?,
        generate_artifact_provenance_dataset(64, seed ^ 0x22b3_c4d5_e6f7_0812)?,
        generate_cloud_data_movement_dataset(64, seed ^ 0x33c4_d5e6_f708_1923)?,
        generate_cross_domain_trust_chain_dataset(64, seed ^ 0x44d5_e6f7_0819_2a34)?,
    ];
    let mut benign_events = Vec::new();
    for dataset in source_datasets {
        benign_events.extend(extract_benign_chain_events(dataset)?);
    }
    if benign_events.len() >= record_count {
        return Err(format!(
            "benign control chain count {} exceeds requested record count {record_count}",
            benign_events.len()
        ));
    }

    let mut rng = Rng::new(seed ^ 0x5ae1_2026_1017_0c01);
    let mut events = Vec::with_capacity(record_count);
    let mut used_ids = BTreeSet::new();
    let background_count = record_count - benign_events.len();
    for index in 0..background_count {
        background_event(index, BASE_TIME_MS, &mut events, &mut rng, &mut used_ids);
    }
    for mut event in benign_events {
        event.event_id = next_event_id(&mut rng, &mut used_ids);
        events.push(event);
    }
    events.sort_by(|a, b| (a.ingest_time_ms, &a.event_id).cmp(&(b.ingest_time_ms, &b.event_id)));
    let event_store = encode_jsonl(&events)?;
    let event_store_sha256 = digest(&event_store);

    let labels = events.iter().map(|event| EventLabel {
        event_id: event.event_id.clone(),
        label: "negative".to_string(),
        rationale: "benign-only hard-negative control; no planted attack is asserted".to_string(),
    }).collect::<Vec<_>>();

    let truth = GroundTruth {
        schema_version: TRUTH_SCHEMA.to_string(),
        scenario_id: SCENARIO.to_string(),
        profile: "public-benign-only-control-v1".to_string(),
        seed,
        event_count: record_count,
        event_store_sha256: event_store_sha256.clone(),
        labels,
        expected_relations: Vec::new(),
        claim_ceiling: "Synthetic benign-only false-positive/control-rate evaluation only; not solver capability or production assurance.".to_string(),
    };
    let manifest = DatasetManifest {
        schema_version: MANIFEST_SCHEMA.to_string(),
        generator_version: GENERATOR_VERSION.to_string(),
        profile: "public-benign-only-control-v1".to_string(),
        scenario_id: SCENARIO.to_string(),
        seed,
        record_count,
        event_store_sha256,
        event_store_utf8_bytes: event_store.len(),
        ordering: "ingest_time_ms_then_event_id; event_time_ms remains distinct".to_string(),
        qualification_state: "UNSCORED".to_string(),
        generator_source_commit: "UNPINNED_LOCAL_SOURCE".to_string(),
        source_commit_pinned: false,
        claim_ceiling: "Exact deterministic benign-only hard-negative smoke fixture; not solver capability evidence.".to_string(),
    };
    Ok(Dataset { event_store, manifest, truth })
}

fn generate_dataset_for_family(
    record_count: usize,
    seed: u64,
    family: CampaignFamily,
) -> Result<Dataset, String> {
    match family {
        CampaignFamily::IdentityToEgress => generate_dataset(record_count, seed),
        CampaignFamily::ArtifactProvenance => generate_artifact_provenance_dataset(record_count, seed),
        CampaignFamily::CloudDataMovement => generate_cloud_data_movement_dataset(record_count, seed),
        CampaignFamily::CrossDomainTrustChain => generate_cross_domain_trust_chain_dataset(record_count, seed),
        CampaignFamily::BenignOnlyControl => generate_benign_only_control_dataset(record_count, seed),
    }
}

#[derive(Debug)]
struct Config {
    records: usize,
    seed: u64,
    family: CampaignFamily,
    events_out: PathBuf,
    manifest_out: PathBuf,
    truth_out: PathBuf,
    source_commit: String,
}

fn parse_args() -> Result<Option<Config>, String> {
    let mut args = env::args().skip(1);
    let mut records = 1_000usize;
    let mut seed = 20_261_017u64;
    let mut family = CampaignFamily::IdentityToEgress;
    let mut events_out = None;
    let mut manifest_out = None;
    let mut truth_out = None;
    let mut source_commit = "UNPINNED_LOCAL_SOURCE".to_string();

    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>, key: &str| {
            args.next().ok_or_else(|| format!("{key} requires a value"))
        };
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--records" => {
                records = value(&mut args, "--records")?.parse()
                    .map_err(|_| "--records must be an integer".to_string())?;
            }
            "--seed" => {
                seed = value(&mut args, "--seed")?.parse()
                    .map_err(|_| "--seed must be an unsigned integer".to_string())?;
            }
            "--family" => {
                family = CampaignFamily::parse(&value(&mut args, "--family")?)?;
            }
            "--events-out" => events_out = Some(PathBuf::from(value(&mut args, "--events-out")?)),
            "--manifest-out" => manifest_out = Some(PathBuf::from(value(&mut args, "--manifest-out")?)),
            "--truth-out" => truth_out = Some(PathBuf::from(value(&mut args, "--truth-out")?)),
            "--source-commit" => source_commit = value(&mut args, "--source-commit")?,
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    if !(MIN_RECORDS..=MAX_RECORDS).contains(&records) {
        return Err(format!("--records must be between {MIN_RECORDS} and {MAX_RECORDS}"));
    }
    let events_out = events_out.ok_or_else(|| "--events-out is required".to_string())?;
    let manifest_out = manifest_out.ok_or_else(|| "--manifest-out is required".to_string())?;
    let truth_out = truth_out.ok_or_else(|| "--truth-out is required".to_string())?;
    if events_out == manifest_out || events_out == truth_out || manifest_out == truth_out {
        return Err("events, manifest, and oracle outputs must use distinct paths".to_string());
    }
    if source_commit != "UNPINNED_LOCAL_SOURCE"
        && !(source_commit.len() == 40 && source_commit.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return Err("--source-commit must be a 40-character hexadecimal commit SHA".to_string());
    }

    Ok(Some(Config { records, seed, family, events_out, manifest_out, truth_out, source_commit }))
}

fn canonical_output_path(path: &Path) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("output path has no file name: {}", path.display()))?;
    let parent = fs::canonicalize(parent)
        .map_err(|e| format!("cannot canonicalize output directory {}: {e}", parent.display()))?;
    Ok(parent.join(file_name))
}

fn path_is_ancestor_or_equal(parent: &Path, child: &Path) -> bool {
    child.starts_with(parent)
}

fn write_dataset(config: &Config) -> Result<(), String> {
    let dataset = generate_dataset_for_family(config.records, config.seed, config.family)?;
    let mut manifest = dataset.manifest;
    // Source revision is recorded as provenance metadata; the caller must supply a clean,
    // exact source commit. Local mode remains explicitly unpinned and unqualified.
    let source_commit_is_pinned = config.source_commit.len() == 40
        && config.source_commit.bytes().all(|b| b.is_ascii_hexdigit());
    manifest.generator_source_commit = config.source_commit.clone();
    manifest.source_commit_pinned = source_commit_is_pinned;
    let mut manifest_json = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| format!("cannot serialize manifest: {e}"))?;
    manifest_json.push(b'\n');
    let mut truth = dataset.truth;
    truth.event_store_sha256 = digest(&dataset.event_store);
    let truth_json = serde_json::to_vec_pretty(&truth)
        .map_err(|e| format!("cannot serialize oracle: {e}"))?;

    for path in [&config.events_out, &config.manifest_out, &config.truth_out] {
        if path.as_os_str().is_empty() {
            return Err("output paths must not be empty".to_string());
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create output directory {}: {e}", parent.display()))?;
    }

    let events_path = canonical_output_path(&config.events_out)?;
    let manifest_path = canonical_output_path(&config.manifest_out)?;
    let truth_path = canonical_output_path(&config.truth_out)?;
    if events_path == manifest_path || events_path == truth_path || manifest_path == truth_path {
        return Err("output paths resolve to the same file".to_string());
    }
    let events_dir = events_path.parent().ok_or_else(|| "events output has no parent".to_string())?;
    let manifest_dir = manifest_path.parent().ok_or_else(|| "manifest output has no parent".to_string())?;
    let truth_dir = truth_path.parent().ok_or_else(|| "oracle output has no parent".to_string())?;
    if path_is_ancestor_or_equal(events_dir, truth_dir)
        || path_is_ancestor_or_equal(truth_dir, events_dir)
        || path_is_ancestor_or_equal(manifest_dir, truth_dir)
        || path_is_ancestor_or_equal(truth_dir, manifest_dir)
    {
        return Err(
            "evaluator oracle directory must be a disjoint sibling tree from solver event/manifest directories"
                .to_string(),
        );
    }

    fs::write(&config.events_out, &dataset.event_store)
        .map_err(|e| format!("cannot write event store {}: {e}", config.events_out.display()))?;
    fs::write(&config.manifest_out, &manifest_json)
        .map_err(|e| format!("cannot write manifest {}: {e}", config.manifest_out.display()))?;
    fs::write(&config.truth_out, &truth_json)
        .map_err(|e| format!("cannot write evaluator oracle {}: {e}", config.truth_out.display()))?;

    println!("generator_version={GENERATOR_VERSION}");
    println!("records={}", config.records);
    println!("seed={}", config.seed);
    println!("family={}", config.family.as_str());
    println!("event_store_sha256={}", digest(&dataset.event_store));
    println!("event_store_bytes={}", dataset.event_store.len());
    println!("manifest={}", config.manifest_out.display());
    println!("solver_events={}", config.events_out.display());
    println!("evaluator_oracle={}", config.truth_out.display());
    println!("oracle_visibility=EVALUATOR_ONLY_KEEP_OUTSIDE_SOLVER_FILESYSTEM");
    println!("source_commit_pinned={source_commit_is_pinned}");
    Ok(())
}

fn main() {
    match parse_args() {
        Ok(None) => {
            println!(
                "Usage: cyber-hunt-fixture-generator --events-out PATH --manifest-out PATH --truth-out PATH [--family identity-to-egress|artifact-provenance|cloud-data-movement|cross-domain-trust-chain|benign-only-control] [--records N] [--seed N] [--source-commit 40_HEX_SHA]\n\
                 Generates deterministic synthetic defensive telemetry only. Keep --truth-out outside the solver-visible filesystem."
            );
        }
        Ok(Some(config)) => {
            if let Err(error) = write_dataset(&config) {
                eprintln!("cyber-hunt-fixture-generator: FAIL: {error}");
                process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("cyber-hunt-fixture-generator: FAIL: {error}");
            eprintln!("Run --help for usage.");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_provenance_family_has_a_distinct_causal_graph_and_benign_matched_chain() {
        let dataset = generate_artifact_provenance_dataset(1_000, 2026)
            .expect("artifact-provenance dataset should generate");
        let events: Vec<EventRecord> = std::str::from_utf8(&dataset.event_store)
            .expect("JSONL should be UTF-8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("event should deserialize"))
            .collect();
        let positive_ids: BTreeSet<String> = dataset.truth.labels.iter()
            .filter(|label| label.label == "positive")
            .map(|label| label.event_id.clone())
            .collect();
        let positive_events: Vec<&EventRecord> = events.iter()
            .filter(|event| positive_ids.contains(&event.event_id))
            .collect();
        assert_eq!(dataset.manifest.scenario_id, "synthetic-artifact-lifecycle-v1");
        assert_eq!(dataset.truth.expected_relations.len(), 5);
        assert_eq!(positive_events.len(), 6);
        assert_eq!(
            dataset.truth.labels.iter().filter(|label| label.label == "negative").count(),
            994
        );
        for expected_type in [
            "commit_checkout",
            "build_job",
            "artifact_publish",
            "provenance_attestation",
            "artifact_promotion",
            "deployment",
        ] {
            assert!(
                positive_events.iter().any(|event| event.event_type == expected_type),
                "positive chain must contain {expected_type}"
            );
            assert!(
                events.iter().any(|event| event.event_type == expected_type
                    && !positive_ids.contains(&event.event_id)),
                "benign chain must contain a lookalike {expected_type}"
            );
        }
        let checkout = positive_events.iter().find(|event| event.event_type == "commit_checkout")
            .expect("planted source checkout");
        let build = positive_events.iter().find(|event| event.event_type == "build_job")
            .expect("planted build");
        let publish = positive_events.iter().find(|event| event.event_type == "artifact_publish")
            .expect("planted artifact publish");
        let attestation = positive_events.iter().find(|event| event.event_type == "provenance_attestation")
            .expect("planted attestation");
        let promotion = positive_events.iter().find(|event| event.event_type == "artifact_promotion")
            .expect("planted promotion");
        let deployment = positive_events.iter().find(|event| event.event_type == "deployment")
            .expect("planted deployment");
        assert_eq!(checkout.details.get("commit_id"), build.details.get("source_commit_id"));
        assert_eq!(build.details.get("source_commit_id"), publish.details.get("source_commit_id"));
        assert_ne!(
            publish.details.get("artifact_digest"),
            attestation.details.get("subject_digest"),
            "a valid signature for a different digest must not be treated as provenance for the published artifact"
        );
        assert_eq!(publish.details.get("artifact_digest"), promotion.details.get("artifact_digest"));
        assert_eq!(attestation.details.get("subject_digest"), promotion.details.get("attestation_subject_digest"));
        assert_eq!(promotion.details.get("artifact_digest"), deployment.details.get("artifact_digest"));

        let replay = generate_artifact_provenance_dataset(1_000, 2026)
            .expect("identical seed should replay");
        assert_eq!(dataset.event_store, replay.event_store);
        assert_eq!(dataset.manifest, replay.manifest);
        assert_eq!(dataset.truth, replay.truth);
        let different_seed = generate_artifact_provenance_dataset(1_000, 2027)
            .expect("different seed should generate");
        assert_ne!(dataset.manifest.event_store_sha256, different_seed.manifest.event_store_sha256);

        let benign_deployments: Vec<&EventRecord> = events.iter()
            .filter(|event| event.event_type == "deployment"
                && event.details.get("environment").map(String::as_str) == Some("staging")
                && !positive_ids.contains(&event.event_id))
            .collect();
        assert_eq!(benign_deployments.len(), 1);
        let benign_deployment = benign_deployments[0];
        let artifact_digest = benign_deployment.details.get("artifact_digest").expect("benign digest");
        let benign_promotion = events.iter()
            .find(|event| event.event_type == "artifact_promotion"
                && event.details.get("artifact_digest") == Some(artifact_digest))
            .expect("benign matching promotion should exist");
        let benign_attestation = events.iter()
            .find(|event| event.event_type == "provenance_attestation"
                && event.details.get("subject_digest") == Some(artifact_digest))
            .expect("benign matching attestation should exist");
        assert!(!positive_ids.contains(&benign_attestation.event_id));
        assert!(!positive_ids.contains(&benign_promotion.event_id));
    }

    #[test]
    fn benign_only_control_has_zero_positives_and_zero_attack_relations() {
        let dataset = generate_benign_only_control_dataset(1_000, 2026)
            .expect("benign-only control should generate");
        let events: Vec<EventRecord> = std::str::from_utf8(&dataset.event_store)
            .expect("event store should be UTF-8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("event should deserialize"))
            .collect();
        let event_ids: BTreeSet<String> = events.iter()
            .map(|event| event.event_id.clone())
            .collect();
        let label_ids: BTreeSet<String> = dataset.truth.labels.iter()
            .map(|label| label.event_id.clone())
            .collect();
        assert_eq!(dataset.manifest.scenario_id, "synthetic-benign-only-control-v1");
        assert_eq!(events.len(), 1_000);
        assert_eq!(event_ids.len(), 1_000);
        assert_eq!(label_ids, event_ids);
        assert!(dataset.truth.labels.iter().all(|label| label.label == "negative"));
        assert!(dataset.truth.expected_relations.is_empty());
        assert_eq!(dataset.truth.event_store_sha256, digest(&dataset.event_store));
        for expected_type in [
            "commit_checkout",
            "build_job",
            "artifact_publish",
            "provenance_attestation",
            "artifact_promotion",
            "deployment",
            "role_assignment",
            "token_issue",
            "storage_list",
            "object_read_batch",
            "export_job",
            "data_transfer",
            "network_transfer",
        ] {
            assert!(events.iter().any(|event| event.event_type == expected_type),
                "benign-only control should include the {expected_type} hard negative");
        }
        assert!(events.iter().all(|event| event.synthetic));
        let replay = generate_benign_only_control_dataset(1_000, 2026)
            .expect("same seed should replay");
        assert_eq!(dataset.event_store, replay.event_store);
        assert_eq!(dataset.truth, replay.truth);
        let other_seed = generate_benign_only_control_dataset(1_000, 2027)
            .expect("other seed should generate");
        assert_ne!(dataset.manifest.event_store_sha256, other_seed.manifest.event_store_sha256);
    }

    #[test]
    fn cross_domain_trust_chain_checks_identity_build_attestation_and_cloud_scope() {
        let dataset = generate_cross_domain_trust_chain_dataset(1_000, 2026)
            .expect("cross-domain trust-chain dataset should generate");
        let events: Vec<EventRecord> = std::str::from_utf8(&dataset.event_store)
            .expect("JSONL should be UTF-8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("event should deserialize"))
            .collect();
        let positive_ids: BTreeSet<String> = dataset.truth.labels.iter()
            .filter(|label| label.label == "positive")
            .map(|label| label.event_id.clone())
            .collect();
        let positive_events: Vec<&EventRecord> = events.iter()
            .filter(|event| positive_ids.contains(&event.event_id))
            .collect();

        assert_eq!(dataset.manifest.scenario_id, "synthetic-cross-domain-trust-chain-v1");
        assert_eq!(positive_events.len(), 12);
        assert_eq!(dataset.truth.expected_relations.len(), 11);
        assert_eq!(
            dataset.truth.labels.iter().filter(|label| label.label == "negative").count(),
            988
        );

        let find_positive = |event_type: &str| -> &EventRecord {
            positive_events.iter().copied()
                .find(|event| event.event_type == event_type)
                .unwrap_or_else(|| panic!("positive event {event_type} should exist"))
        };
        let checkout = find_positive("commit_checkout");
        let build = find_positive("build_job");
        let publish = find_positive("artifact_publish");
        let attestation = find_positive("provenance_attestation");
        let promotion = find_positive("artifact_promotion");
        let deployment = find_positive("deployment");
        let role = find_positive("role_assignment");
        let token = find_positive("token_issue");
        let storage = find_positive("storage_list");
        let transfer = find_positive("data_transfer");

        assert_eq!(checkout.details.get("commit_id"), build.details.get("source_commit_id"));
        assert_eq!(build.details.get("source_commit_id"), publish.details.get("source_commit_id"));
        assert_ne!(publish.details.get("artifact_digest"), attestation.details.get("subject_digest"));
        assert_eq!(publish.details.get("artifact_digest"), promotion.details.get("artifact_digest"));
        assert_eq!(attestation.details.get("subject_digest"), promotion.details.get("attestation_subject_digest"));
        assert_eq!(promotion.details.get("artifact_digest"), deployment.details.get("artifact_digest"));
        assert_eq!(role.details.get("approval_state").map(String::as_str), Some("missing"));
        assert_eq!(role.details.get("scope"), token.details.get("effective_scope"));
        assert_ne!(token.details.get("effective_scope"), storage.details.get("resource_scope"));
        assert_eq!(transfer.details.get("destination_profile").map(String::as_str), Some("unapproved_external"));

        let positive_principal = checkout.details.get("principal_object_id").expect("pipeline principal");
        for event in &positive_events {
            assert_eq!(
                event.details.get("principal_object_id"),
                Some(positive_principal),
                "the cross-domain chain must preserve principal lineage across {}",
                event.event_type
            );
        }

        let benign_checkout = events.iter().find(|event| event.event_type == "commit_checkout"
            && !positive_ids.contains(&event.event_id)
            && event.details.get("review_state").map(String::as_str) == Some("approved"))
            .expect("approved benign checkout should exist");
        let benign_principal = benign_checkout.details.get("principal_object_id")
            .expect("benign principal lineage");
        let benign_publish = events.iter().find(|event| event.event_type == "artifact_publish"
            && event.details.get("principal_object_id") == Some(benign_principal))
            .expect("benign artifact should exist");
        let benign_attestation = events.iter().find(|event| event.event_type == "provenance_attestation"
            && event.details.get("principal_object_id") == Some(benign_principal))
            .expect("benign attestation should exist");
        let benign_promotion = events.iter().find(|event| event.event_type == "artifact_promotion"
            && event.details.get("principal_object_id") == Some(benign_principal))
            .expect("benign promotion should exist");
        let benign_role = events.iter().find(|event| event.event_type == "role_assignment"
            && event.details.get("principal_object_id") == Some(benign_principal))
            .expect("benign role assignment should exist");
        let benign_token = events.iter().find(|event| event.event_type == "token_issue"
            && event.details.get("principal_object_id") == Some(benign_principal))
            .expect("benign token should exist");
        let benign_storage = events.iter().find(|event| event.event_type == "storage_list"
            && event.details.get("principal_object_id") == Some(benign_principal))
            .expect("benign storage access should exist");
        assert_eq!(benign_publish.details.get("artifact_digest"), benign_attestation.details.get("subject_digest"));
        assert_eq!(benign_publish.details.get("artifact_digest"), benign_promotion.details.get("artifact_digest"));
        assert_eq!(benign_role.details.get("approval_state").map(String::as_str), Some("approved"));
        assert_eq!(benign_role.details.get("scope"), benign_token.details.get("effective_scope"));
        assert_eq!(benign_token.details.get("effective_scope"), benign_storage.details.get("resource_scope"));

        let replay = generate_cross_domain_trust_chain_dataset(1_000, 2026)
            .expect("same seed should replay");
        assert_eq!(dataset.event_store, replay.event_store);
        assert_eq!(dataset.truth, replay.truth);
        let other_seed = generate_cross_domain_trust_chain_dataset(1_000, 2027)
            .expect("other seed should generate");
        assert_ne!(dataset.manifest.event_store_sha256, other_seed.manifest.event_store_sha256);
    }

    #[test]
    fn cloud_data_movement_family_masks_event_type_and_checks_scope_authority() {
        let dataset = generate_cloud_data_movement_dataset(1_000, 2026)
            .expect("cloud data movement dataset should generate");
        let events: Vec<EventRecord> = std::str::from_utf8(&dataset.event_store)
            .expect("JSONL should be UTF-8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("event should deserialize"))
            .collect();
        let positive_ids: BTreeSet<String> = dataset.truth.labels.iter()
            .filter(|label| label.label == "positive")
            .map(|label| label.event_id.clone())
            .collect();
        let positive_events: Vec<&EventRecord> = events.iter()
            .filter(|event| positive_ids.contains(&event.event_id))
            .collect();

        assert_eq!(dataset.manifest.scenario_id, "synthetic-cloud-data-movement-v1");
        assert_eq!(positive_events.len(), 6);
        assert_eq!(dataset.truth.expected_relations.len(), 5);
        assert_eq!(
            dataset.truth.labels.iter().filter(|label| label.label == "negative").count(),
            994
        );
        for expected_type in [
            "role_assignment",
            "token_issue",
            "storage_list",
            "object_read_batch",
            "export_job",
            "data_transfer",
        ] {
            assert!(positive_events.iter().any(|event| event.event_type == expected_type));
            assert!(events.iter().any(|event| event.event_type == expected_type
                && !positive_ids.contains(&event.event_id)),
                "benign chain must contain a lookalike {expected_type}");
        }

        let role = positive_events.iter().find(|event| event.event_type == "role_assignment")
            .expect("positive role assignment");
        let token = positive_events.iter().find(|event| event.event_type == "token_issue")
            .expect("positive token issue");
        let storage = positive_events.iter().find(|event| event.event_type == "storage_list")
            .expect("positive storage list");
        let transfer = positive_events.iter().find(|event| event.event_type == "data_transfer")
            .expect("positive transfer");
        assert_eq!(role.details.get("approval_state").map(String::as_str), Some("missing"));
        assert_eq!(role.details.get("scope"), token.details.get("effective_scope"));
        assert_ne!(token.details.get("effective_scope"), storage.details.get("resource_scope"));
        assert_eq!(transfer.details.get("destination_profile").map(String::as_str), Some("unapproved_external"));

        let benign_transfer = events.iter().find(|event| event.event_type == "data_transfer"
            && !positive_ids.contains(&event.event_id)
            && event.details.get("destination_profile").map(String::as_str) == Some("approved_backup_endpoint"))
            .expect("approved benign transfer should exist");
        let benign_object_id = benign_transfer.details.get("principal_object_id")
            .expect("benign principal object ID");
        let benign_role = events.iter().find(|event| event.event_type == "role_assignment"
            && event.details.get("principal_object_id") == Some(benign_object_id))
            .expect("benign role assignment should exist");
        let benign_token = events.iter().find(|event| event.event_type == "token_issue"
            && event.details.get("principal_object_id") == Some(benign_object_id))
            .expect("benign token issue should exist");
        let benign_storage = events.iter().find(|event| event.event_type == "storage_list"
            && event.details.get("principal_object_id") == Some(benign_object_id))
            .expect("benign storage list should exist");
        assert_eq!(benign_role.details.get("approval_state").map(String::as_str), Some("approved"));
        assert_eq!(benign_role.details.get("scope"), benign_token.details.get("effective_scope"));
        assert_eq!(benign_token.details.get("effective_scope"), benign_storage.details.get("resource_scope"));

        let replay = generate_cloud_data_movement_dataset(1_000, 2026)
            .expect("same seed should replay");
        assert_eq!(dataset.event_store, replay.event_store);
        assert_eq!(dataset.truth, replay.truth);
        let other_seed = generate_cloud_data_movement_dataset(1_000, 2027)
            .expect("other seed should generate");
        assert_ne!(dataset.manifest.event_store_sha256, other_seed.manifest.event_store_sha256);
    }

    #[test]
    fn schemas_declare_current_generator_version_and_all_scenario_ids() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let manifest_schema: serde_json::Value = serde_json::from_slice(
            &fs::read(root.join("docs/research/cyber-hunt-manifest-v1.schema.json"))
                .expect("manifest schema should be readable"),
        )
        .expect("manifest schema should be valid JSON");
        let truth_schema: serde_json::Value = serde_json::from_slice(
            &fs::read(root.join("docs/research/cyber-hunt-truth-v1.schema.json"))
                .expect("truth schema should be readable"),
        )
        .expect("truth schema should be valid JSON");

        let versions = manifest_schema["properties"]["generator_version"]["enum"]
            .as_array().expect("manifest schema should enumerate supported generator versions");
        assert!(versions.iter().any(|v| v.as_str() == Some(GENERATOR_VERSION)));

        let manifest_ids = manifest_schema["properties"]["scenario_id"]["enum"]
            .as_array().expect("manifest schema should enumerate scenario IDs");
        let truth_ids = truth_schema["properties"]["scenario_id"]["enum"]
            .as_array().expect("truth schema should enumerate scenario IDs");
        for scenario_id in [SCENARIO_ID, "synthetic-artifact-lifecycle-v1", "synthetic-cloud-data-movement-v1", "synthetic-cross-domain-trust-chain-v1", "synthetic-benign-only-control-v1"] {
            assert!(manifest_ids.iter().any(|v| v.as_str() == Some(scenario_id)));
            assert!(truth_ids.iter().any(|v| v.as_str() == Some(scenario_id)));
        }
    }

    #[test]
    fn campaign_family_selection_keeps_existing_family_as_default() {
        assert_eq!(
            CampaignFamily::parse("identity-to-egress").expect("identity family should parse"),
            CampaignFamily::IdentityToEgress
        );
        assert_eq!(
            CampaignFamily::parse("artifact-provenance").expect("artifact family should parse"),
            CampaignFamily::ArtifactProvenance
        );
        assert_eq!(
            CampaignFamily::parse("cloud-data-movement").expect("cloud family should parse"),
            CampaignFamily::CloudDataMovement
        );
        assert_eq!(
            CampaignFamily::parse("cross-domain-trust-chain").expect("cross-domain family should parse"),
            CampaignFamily::CrossDomainTrustChain
        );
        assert_eq!(
            CampaignFamily::parse("benign-only-control").expect("benign-only family should parse"),
            CampaignFamily::BenignOnlyControl
        );
        assert!(CampaignFamily::parse("invented-family").is_err());
        let cloud = generate_dataset_for_family(128, 2026, CampaignFamily::CloudDataMovement)
            .expect("cloud data movement family should generate");
        assert_eq!(cloud.manifest.scenario_id, "synthetic-cloud-data-movement-v1");
        let cross_domain = generate_dataset_for_family(128, 2026, CampaignFamily::CrossDomainTrustChain)
            .expect("cross-domain family should generate");
        assert_eq!(cross_domain.manifest.scenario_id, "synthetic-cross-domain-trust-chain-v1");
        let benign_only = generate_dataset_for_family(128, 2026, CampaignFamily::BenignOnlyControl)
            .expect("benign-only family should generate");
        assert_eq!(benign_only.manifest.scenario_id, "synthetic-benign-only-control-v1");
        let default = generate_dataset_for_family(128, 2026, CampaignFamily::IdentityToEgress)
            .expect("identity family should generate");
        let compatibility = generate_dataset(128, 2026)
            .expect("default generator should remain compatible");
        assert_eq!(default.event_store, compatibility.event_store);
        assert_eq!(default.truth, compatibility.truth);
    }

    #[test]
    fn every_family_preserves_cross_cutting_truth_and_digest_invariants() {
        let scenarios = [
            (CampaignFamily::IdentityToEgress, 9_usize),
            (CampaignFamily::ArtifactProvenance, 6_usize),
            (CampaignFamily::CloudDataMovement, 6_usize),
            (CampaignFamily::CrossDomainTrustChain, 12_usize),
            (CampaignFamily::BenignOnlyControl, 0_usize),
        ];
        for (family, expected_positive_count) in scenarios {
            for seed in [0_u64, 1_u64, 2026_u64] {
                let dataset = generate_dataset_for_family(128, seed, family)
                    .expect("each family should generate across fixed seed controls");
                assert_eq!(dataset.manifest.record_count, 128);
                assert_eq!(dataset.truth.event_count, 128);
                assert_eq!(dataset.truth.event_store_sha256, digest(&dataset.event_store));

                let events: Vec<EventRecord> = std::str::from_utf8(&dataset.event_store)
                    .expect("event store should be UTF-8")
                    .lines()
                    .map(|line| serde_json::from_str(line).expect("event should deserialize"))
                    .collect();
                let event_ids: BTreeSet<String> = events.iter()
                    .map(|event| event.event_id.clone())
                    .collect();
                let label_ids: BTreeSet<String> = dataset.truth.labels.iter()
                    .map(|label| label.event_id.clone())
                    .collect();
                let positive_ids: BTreeSet<String> = dataset.truth.labels.iter()
                    .filter(|label| label.label == "positive")
                    .map(|label| label.event_id.clone())
                    .collect();

                assert_eq!(events.len(), 128);
                assert_eq!(event_ids.len(), 128, "event IDs must be unique");
                assert_eq!(label_ids, event_ids, "truth must label every event exactly once");
                assert_eq!(positive_ids.len(), expected_positive_count);
                assert!(dataset.truth.labels.iter().all(|label|
                    label.label == "positive" || label.label == "negative"
                ));
                for relation in &dataset.truth.expected_relations {
                    assert!(positive_ids.contains(&relation.from_event_id),
                        "relation source must refer to a positive event");
                    assert!(positive_ids.contains(&relation.to_event_id),
                        "relation target must refer to a positive event");
                }
                assert!(events.iter().all(|event| event.synthetic));
                assert!(events.iter().all(|event|
                    !event.details.contains_key("ground_truth") &&
                    !event.details.contains_key("label")
                ));
            }
        }
    }

    #[test]
    fn sha256_known_answer() {
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn same_seed_produces_identical_event_bytes_and_truth() {
        let a = generate_dataset(1_000, 2026).expect("dataset should generate");
        let b = generate_dataset(1_000, 2026).expect("dataset should generate");
        assert_eq!(a.event_store, b.event_store);
        assert_eq!(a.manifest, b.manifest);
        assert_eq!(a.truth, b.truth);
    }

    #[test]
    fn distinct_seeds_change_the_dataset_digest() {
        let a = generate_dataset(1_000, 2026).expect("dataset should generate");
        let b = generate_dataset(1_000, 2027).expect("dataset should generate");
        assert_ne!(a.manifest.event_store_sha256, b.manifest.event_store_sha256);
    }

    #[test]
    fn planted_identity_and_timeline_vary_by_seed_without_special_marker_leaks() {
        let positive_signature = |dataset: &Dataset| {
            let events: Vec<EventRecord> = std::str::from_utf8(&dataset.event_store)
                .expect("JSONL should be UTF-8")
                .lines()
                .map(|line| serde_json::from_str(line).expect("event record should deserialize"))
                .collect();
            let positive_ids: BTreeSet<String> = dataset.truth.labels.iter()
                .filter(|label| label.label == "positive")
                .map(|label| label.event_id.clone())
                .collect();
            let positives: Vec<&EventRecord> = events.iter()
                .filter(|event| positive_ids.contains(&event.event_id))
                .collect();
            let hosts: BTreeSet<String> = positives.iter().map(|event| event.host_id.clone()).collect();
            let principals: BTreeSet<String> = positives.iter().map(|event| event.principal_id.clone()).collect();
            let source_ips: BTreeSet<String> = positives.iter()
                .filter_map(|event| event.details.get("source_address").cloned())
                .collect();
            let destination_ips: BTreeSet<String> = positives.iter()
                .filter_map(|event| event.details.get("destination_address").cloned())
                .collect();
            let start = positives.iter().map(|event| event.event_time_ms).min().expect("attack start");
            (hosts, principals, source_ips, destination_ips, start, positives.len())
        };

        let signatures: Vec<_> = (2026_u64..2042_u64)
            .map(|seed| {
                let dataset = generate_dataset(1_000, seed).expect("seeded dataset should generate");
                positive_signature(&dataset)
            })
            .collect();

        for signature in &signatures {
            assert_eq!(signature.5, ATTACK_EVENT_COUNT);
            assert_eq!(signature.0.len(), 1, "all planted events must share one ordinary host ID");
            assert_eq!(signature.1.len(), 1, "all planted events must share one ordinary principal ID");
            assert_eq!(signature.2.len(), 1, "authentication events must share one source IP");
            assert_eq!(signature.3.len(), 1, "egress events must share one destination IP");
            assert!(signature.0.iter().all(|host| host.starts_with("host-") && !host.starts_with("host-case-")));
            assert!(signature.1.iter().all(|principal| !principal.starts_with("svc-case-")));
        }

        let unique_hosts: BTreeSet<String> = signatures.iter()
            .map(|signature| signature.0.iter().next().expect("host").clone())
            .collect();
        let unique_principals: BTreeSet<String> = signatures.iter()
            .map(|signature| signature.1.iter().next().expect("principal").clone())
            .collect();
        let unique_source_ips: BTreeSet<String> = signatures.iter()
            .map(|signature| signature.2.iter().next().expect("source IP").clone())
            .collect();
        let unique_destination_ips: BTreeSet<String> = signatures.iter()
            .map(|signature| signature.3.iter().next().expect("destination IP").clone())
            .collect();
        let unique_starts: BTreeSet<i64> = signatures.iter().map(|signature| signature.4).collect();

        assert!(unique_hosts.len() > 1, "host identity should vary across seeds");
        assert!(unique_principals.len() > 1, "principal identity should vary across seeds");
        assert!(unique_source_ips.len() > 1, "source IP should vary across seeds");
        assert!(unique_destination_ips.len() > 1, "destination IP should vary across seeds");
        assert!(unique_starts.len() > 1, "campaign time should vary across seeds");
    }

    #[test]
    fn background_activity_overlaps_the_shifted_planted_incident_window() {
        let dataset = generate_dataset(1_000, 2026).expect("dataset should generate");
        let events: Vec<EventRecord> = std::str::from_utf8(&dataset.event_store)
            .expect("JSONL should be UTF-8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("event record should deserialize"))
            .collect();
        let positive_ids: BTreeSet<String> = dataset.truth.labels.iter()
            .filter(|label| label.label == "positive")
            .map(|label| label.event_id.clone())
            .collect();
        let attack_times: Vec<i64> = events.iter()
            .filter(|event| positive_ids.contains(&event.event_id))
            .map(|event| event.event_time_ms)
            .collect();
        let attack_start = *attack_times.iter().min().expect("planted attack should contain events");
        let attack_end = *attack_times.iter().max().expect("planted attack should contain events");
        let background_times: Vec<i64> = events.iter()
            .filter(|event| !positive_ids.contains(&event.event_id))
            .map(|event| event.event_time_ms)
            .collect();

        let overlapped_background = background_times.iter()
            .filter(|time| **time >= attack_start && **time <= attack_end)
            .count();
        assert!(background_times.iter().any(|time| *time < attack_start),
            "ordinary background events should exist before the incident");
        assert!(background_times.iter().any(|time| *time > attack_end),
            "ordinary background events should exist after the incident");
        assert!(overlapped_background >= 20,
            "incident window must contain substantial background noise, got {overlapped_background}");
    }

    #[test]
    fn event_store_sorts_by_ingestion_but_preserves_source_time_inversions() {
        let dataset = generate_dataset(1_000, 2026).expect("dataset should generate");
        let events: Vec<EventRecord> = std::str::from_utf8(&dataset.event_store)
            .expect("JSONL should be UTF-8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("event record should deserialize"))
            .collect();

        assert!(events.windows(2).all(|pair| {
            (pair[0].ingest_time_ms, &pair[0].event_id)
                <= (pair[1].ingest_time_ms, &pair[1].event_id)
        }), "event rows must be sorted by ingestion time with deterministic ID tie-breaking");
        assert!(events.iter().any(|event| event.event_time_ms != event.ingest_time_ms),
            "source time and ingestion time must remain separate fields");
        let source_time_inversions = events.windows(2)
            .filter(|pair| pair[0].event_time_ms > pair[1].event_time_ms)
            .count();
        assert!(source_time_inversions > 0,
            "the fixture must include out-of-order source times after ingestion-time sorting");
    }

    #[test]
    fn every_planted_event_type_has_a_benign_counterexample_in_its_family() {
        let families = [
            CampaignFamily::IdentityToEgress,
            CampaignFamily::ArtifactProvenance,
            CampaignFamily::CloudDataMovement,
        ];
        for family in families {
            let dataset = generate_dataset_for_family(1_000, 2026, family)
                .expect("family should generate");
            let events: Vec<EventRecord> = std::str::from_utf8(&dataset.event_store)
                .expect("event store should be UTF-8")
                .lines()
                .map(|line| serde_json::from_str(line).expect("event should deserialize"))
                .collect();
            let positive_ids: BTreeSet<String> = dataset.truth.labels.iter()
                .filter(|label| label.label == "positive")
                .map(|label| label.event_id.clone())
                .collect();
            let planted_types: BTreeSet<&str> = events.iter()
                .filter(|event| positive_ids.contains(&event.event_id))
                .map(|event| event.event_type.as_str())
                .collect();
            for event_type in planted_types {
                assert!(events.iter().any(|event|
                    event.event_type == event_type && !positive_ids.contains(&event.event_id)
                ), "family {:?} must include a benign lookalike for event type {event_type}", family);
            }
        }
    }

    #[test]
    fn all_truth_labels_reference_unique_existing_event_ids() {
        let dataset = generate_dataset(1_000, 2026).expect("dataset should generate");
        let events: Vec<EventRecord> = std::str::from_utf8(&dataset.event_store)
            .expect("JSONL should be UTF-8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("event record should deserialize"))
            .collect();
        let event_ids: BTreeSet<String> = events.iter().map(|e| e.event_id.clone()).collect();
        let label_ids: BTreeSet<String> = dataset.truth.labels.iter().map(|l| l.event_id.clone()).collect();
        assert_eq!(events.len(), 1_000);
        assert_eq!(event_ids.len(), 1_000);
        assert_eq!(label_ids, event_ids);
        assert_eq!(dataset.truth.labels.iter().filter(|l| l.label == "positive").count(), ATTACK_EVENT_COUNT);
        assert_eq!(dataset.truth.labels.iter().filter(|l| l.label == "negative").count(), 1_000 - ATTACK_EVENT_COUNT);
        assert!(events.iter().all(|e| e.synthetic));
        assert!(events.iter().all(|e| !e.details.contains_key("ground_truth") && !e.details.contains_key("label")));
    }

    #[test]
    fn planted_relations_reference_existing_positive_events() {
        let dataset = generate_dataset(1_000, 2026).expect("dataset should generate");
        let positive: BTreeSet<String> = dataset.truth.labels.iter()
            .filter(|label| label.label == "positive")
            .map(|label| label.event_id.clone())
            .collect();
        assert_eq!(dataset.truth.expected_relations.len(), 8);
        for relation in &dataset.truth.expected_relations {
            assert!(positive.contains(&relation.from_event_id));
            assert!(positive.contains(&relation.to_event_id));
        }
    }

    #[test]
    fn benign_lookalikes_overlap_the_planted_attack_timeline() {
        let dataset = generate_dataset(1_000, 2026).expect("dataset should generate");
        let events: Vec<EventRecord> = std::str::from_utf8(&dataset.event_store)
            .expect("JSONL should be UTF-8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("event record should deserialize"))
            .collect();
        let positive_ids: BTreeSet<String> = dataset.truth.labels.iter()
            .filter(|label| label.label == "positive")
            .map(|label| label.event_id.clone())
            .collect();
        let attack_start = events.iter()
            .filter(|event| positive_ids.contains(&event.event_id))
            .map(|event| event.event_time_ms)
            .min()
            .expect("planted attack should contain events");

        let benign_auth_failures: Vec<&EventRecord> = events.iter()
            .filter(|event| event.host_id == "host-007"
                && event.principal_id == "j.smith"
                && event.event_type == "authentication"
                && event.outcome == "failure"
                && event.details.get("source_address").map(String::as_str) == Some("10.99.4.18"))
            .collect();
        assert_eq!(benign_auth_failures.len(), 3);
        assert!(benign_auth_failures.iter().all(|event| !positive_ids.contains(&event.event_id)));
        let first_benign_auth = benign_auth_failures.iter()
            .map(|event| event.event_time_ms)
            .min()
            .expect("benign authentication decoy should exist");
        assert!(first_benign_auth >= attack_start);
        assert!(first_benign_auth < attack_start + 50_000);

        let benign_powershell = events.iter()
            .find(|event| event.host_id == "host-012"
                && event.principal_id == "it.admin"
                && event.event_type == "process_start"
                && event.details.get("command_summary").map(String::as_str) == Some("Get-Service"))
            .expect("approved PowerShell decoy should exist");
        assert!(!positive_ids.contains(&benign_powershell.event_id));
        assert!(benign_powershell.event_time_ms > attack_start + 120_000);
        assert!(benign_powershell.event_time_ms < attack_start + 145_000);

        let benign_backup = events.iter()
            .find(|event| event.host_id == "host-025"
                && event.principal_id == "svc-backup"
                && event.event_type == "file_write"
                && event.details.get("job_id").map(String::as_str) == Some("BKP-2201"))
            .expect("scheduled backup decoy should exist");
        assert!(!positive_ids.contains(&benign_backup.event_id));
        assert!(benign_backup.event_time_ms > attack_start + 190_000);
        assert!(benign_backup.event_time_ms < attack_start + 240_000);
    }

    #[test]
    fn canonical_paths_detect_aliases_and_separate_solver_from_oracle() {
        let root = env::temp_dir().join(format!("cyber-hunt-path-test-{}", process::id()));
        let solver = root.join("solver");
        let evaluator = root.join("evaluator");
        fs::create_dir_all(&solver).expect("solver directory should be created");
        fs::create_dir_all(&evaluator).expect("evaluator directory should be created");

        let events = canonical_output_path(&solver.join("events.jsonl")).expect("events path should resolve");
        let manifest = canonical_output_path(&solver.join("./manifest.json")).expect("manifest path should resolve");
        let truth = canonical_output_path(&evaluator.join("truth.json")).expect("truth path should resolve");
        assert_ne!(events, manifest);
        assert!(!path_is_ancestor_or_equal(events.parent().unwrap(), truth.parent().unwrap()));
        assert!(!path_is_ancestor_or_equal(truth.parent().unwrap(), events.parent().unwrap()));

        fs::remove_dir_all(root).expect("temporary directories should be removed");
    }

    #[test]
    fn undersized_or_oversized_corpora_fail_closed() {
        assert!(generate_dataset(MIN_RECORDS - 1, 1).is_err());
        assert!(generate_dataset(MAX_RECORDS + 1, 1).is_err());
    }

    #[test]
    fn zero_seed_is_supported_without_degenerate_rng() {
        let a = generate_dataset(128, 0).expect("zero seed should map to a fixed nonzero PRNG state");
        let b = generate_dataset(128, 0).expect("zero seed should be reproducible");
        assert_eq!(a.event_store, b.event_store);
    }
}
