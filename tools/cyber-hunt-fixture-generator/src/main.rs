use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const GENERATOR_VERSION: &str = "cyber-hunt-fixture-generator/0.1.0";
const EVENT_SCHEMA: &str = "cyber-hunt-event-v1";
const MANIFEST_SCHEMA: &str = "cyber-hunt-manifest-v1";
const TRUTH_SCHEMA: &str = "cyber-hunt-truth-v1";
const SCENARIO_ID: &str = "synthetic-identity-to-egress-v1";
const MIN_RECORDS: usize = 64;
const MAX_RECORDS: usize = 250_000;
const BASE_TIME_MS: i64 = 1_800_000_000_000;
const ATTACK_EVENT_COUNT: usize = 9;
const BENIGN_DECOY_COUNT: usize = 6;
const PLANTED_EVENT_COUNT: usize = ATTACK_EVENT_COUNT + BENIGN_DECOY_COUNT;

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
    events: &mut Vec<EventRecord>,
    rng: &mut Rng,
    used_ids: &mut BTreeSet<String>,
) {
    let time_offset = rng.below(3_600_000) as i64 - 1_800_000;
    let event_time_ms = BASE_TIME_MS.saturating_add(time_offset);
    let host_number = 1 + rng.below(48);
    let host_id = format!("host-{host_number:03}");
    let principals = [
        "a.chen", "j.smith", "m.patel", "it.admin", "svc-report", "svc-backup",
        "svc-build", "d.owens", "s.lee", "svc-inventory", "helpdesk.1", "r.garcia",
    ];
    let principal = principals[rng.below(principals.len() as u64) as usize];
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

    for index in 0..(record_count - PLANTED_EVENT_COUNT) {
        background_event(index, &mut events, &mut rng, &mut used_ids);
    }

    // Seed-dependent shift and case-local identities make trivial fixed-value matching less useful.
    // These identifiers are synthetic and do not refer to real hosts or principals.
    let base = BASE_TIME_MS.saturating_add(600_000 + rng.below(7_200_000) as i64);
    let host = format!("host-case-{:04x}", rng.next_u64() as u16);
    let principal = format!("svc-case-{:04x}", rng.next_u64() as u16);
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

#[derive(Debug)]
struct Config {
    records: usize,
    seed: u64,
    events_out: PathBuf,
    manifest_out: PathBuf,
    truth_out: PathBuf,
    source_commit: String,
}

fn parse_args() -> Result<Option<Config>, String> {
    let mut args = env::args().skip(1);
    let mut records = 1_000usize;
    let mut seed = 20_261_017u64;
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

    Ok(Some(Config { records, seed, events_out, manifest_out, truth_out, source_commit }))
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
    let dataset = generate_dataset(config.records, config.seed)?;
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
                "Usage: cyber-hunt-fixture-generator --events-out PATH --manifest-out PATH --truth-out PATH [--records N] [--seed N] [--source-commit 40_HEX_SHA]\n\
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
    fn planted_identity_and_timeline_vary_by_seed_while_the_chain_stays_coherent() {
        let a = generate_dataset(1_000, 2026).expect("first dataset should generate");
        let b = generate_dataset(1_000, 2027).expect("second dataset should generate");

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

        let (hosts_a, principals_a, source_ips_a, destination_ips_a, start_a, count_a) = positive_signature(&a);
        let (hosts_b, principals_b, source_ips_b, destination_ips_b, start_b, count_b) = positive_signature(&b);
        assert_eq!(count_a, ATTACK_EVENT_COUNT);
        assert_eq!(count_b, ATTACK_EVENT_COUNT);
        assert_eq!(hosts_a.len(), 1, "all planted attack events share one case-local host");
        assert_eq!(hosts_b.len(), 1, "all planted attack events share one case-local host");
        assert_eq!(principals_a.len(), 1, "all planted attack events share one case-local principal");
        assert_eq!(principals_b.len(), 1, "all planted attack events share one case-local principal");
        assert_ne!(hosts_a, hosts_b, "the planted host identity should be seed-dependent");
        assert_ne!(principals_a, principals_b, "the planted principal should be seed-dependent");
        assert_eq!(source_ips_a.len(), 1, "authentication evidence shares one synthetic source IP");
        assert_eq!(source_ips_b.len(), 1, "authentication evidence shares one synthetic source IP");
        assert_eq!(destination_ips_a.len(), 1, "egress evidence shares one synthetic destination IP");
        assert_eq!(destination_ips_b.len(), 1, "egress evidence shares one synthetic destination IP");
        assert_ne!(source_ips_a, source_ips_b, "the planted source IP should be seed-dependent");
        assert_ne!(destination_ips_a, destination_ips_b, "the planted destination IP should be seed-dependent");
        assert_ne!(start_a, start_b, "the planted campaign time should be seed-dependent");
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
