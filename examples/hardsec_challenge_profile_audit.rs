//! Audits a multi-axis HARDSEC challenge profile.
//!
//! This is a profile-integrity checker, not a detector, Trojan classifier or
//! unbounded security proof. It checks whether the declared profile status is
//! consistent with its evidence references, and can verify referenced local
//! bytes against SHA-256 digests.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug, Clone)]
struct Args {
    profile: PathBuf,
    receipt: PathBuf,
    evidence_root: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct EvidenceSummary {
    id: String,
    kind: String,
    status: String,
    sha256: String,
    local_path: Option<String>,
    assumptions_sha256: Option<String>,
    assumptions_local_path: Option<String>,
    subject_netlist_sha256: Option<String>,
    reference_netlist_sha256: Option<String>,
    proof_scope: Option<String>,
    formal_result: Option<String>,
    bound_cycles: Option<u64>,
    property_id: Option<String>,
}

#[derive(Debug, Clone)]
struct ProfileSummary {
    challenge_id: String,
    dataset_id: String,
    dataset_role: String,
    validity_status: String,
    netlist_sha256: String,
    subject_local_path: String,
    evidence: Vec<EvidenceSummary>,
    evidence_ids: BTreeSet<String>,
    active_trace_sha256: Option<String>,
    payload_effect_sha256: Option<String>,
    reference_netlist_sha256: Option<String>,
    reference_local_path: Option<String>,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut profile = None;
    let mut receipt = None;
    let mut evidence_root = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--profile" => profile = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            "--evidence-root" => evidence_root = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(Some(Args {
        profile: profile.ok_or("--profile is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
        evidence_root,
    }))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    let result = value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))?;
    if result.trim().is_empty() || result.trim() != result {
        return Err(format!("{field} must be non-empty and have no surrounding whitespace"));
    }
    Ok(result)
}

fn optional_string(value: &Value, field: &str) -> Result<Option<String>, String> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(item)) if !item.trim().is_empty() && item.trim() == item => {
            Ok(Some(item.clone()))
        }
        _ => Err(format!("{field} must be a non-empty string when present")),
    }
}

fn optional_u64(value: &Value, field: &str) -> Result<Option<u64>, String> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(item) => item
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{field} must be a non-negative integer")),
    }
}

fn safe_relative_path(value: &str, field: &str) -> Result<(), String> {
    if value.contains('\\') || value.contains('\0') {
        return Err(format!("{field} must use portable relative path separators"));
    }
    let path = Path::new(value);
    if path.is_absolute() || path.as_os_str().is_empty() {
        return Err(format!("{field} must be a non-empty relative path"));
    }
    let components = path.components().collect::<Vec<_>>();
    if components.is_empty()
        || components
            .iter()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("{field} must not contain '.', '..', root or prefix components"));
    }
    Ok(())
}

fn evidence_map(items: &[EvidenceSummary]) -> BTreeMap<String, EvidenceSummary> {
    items
        .iter()
        .cloned()
        .map(|item| (item.id.clone(), item))
        .collect()
}

fn evidence_by_id<'a>(
    evidence: &'a BTreeMap<String, EvidenceSummary>,
    id: &str,
    expected_kind: &str,
) -> Result<&'a EvidenceSummary, String> {
    let item = evidence
        .get(id)
        .ok_or_else(|| format!("unknown evidence_id {id}"))?;
    if item.kind != expected_kind {
        return Err(format!(
            "evidence {id} has kind {}, expected {expected_kind}",
            item.kind
        ));
    }
    if item.status != "pass" {
        return Err(format!("evidence {id} must have status=pass"));
    }
    Ok(item)
}

fn validate_profile(value: &Value, profile_sha256: String) -> Result<ProfileSummary, String> {
    if required_string(value, "schema_version")? != "hardsec-challenge-profile-v1" {
        return Err("unsupported schema_version".to_owned());
    }

    let challenge_id = required_string(value, "challenge_id")?.to_owned();
    let dataset_id = required_string(value, "dataset_id")?.to_owned();
    let dataset_role = required_string(value, "dataset_role")?.to_owned();
    let validity_status = required_string(value, "validity_status")?.to_owned();

    if ![
        "evaluation_corpus_member",
        "reference_fixture_only",
        "exploratory_only",
    ]
    .contains(&dataset_role.as_str())
    {
        return Err(format!("unsupported dataset_role {dataset_role}"));
    }
    if ![
        "witness_validated",
        "partially_evidenced",
        "benign_equivalent",
        "unknown_invalid",
    ]
    .contains(&validity_status.as_str())
    {
        return Err(format!("unsupported validity_status {validity_status}"));
    }

    let subject = value
        .get("subject")
        .ok_or("subject is required")?;
    let netlist_sha256 = required_string(subject, "netlist_sha256")?.to_ascii_lowercase();
    if !valid_sha256(&netlist_sha256) {
        return Err("subject.netlist_sha256 must be a SHA-256 digest".to_owned());
    }
    let subject_local_path = required_string(subject, "local_path")?.to_owned();
    safe_relative_path(&subject_local_path, "subject.local_path")?;

    let evidence_values = value
        .get("evidence")
        .and_then(Value::as_array)
        .ok_or("evidence must be an array")?;
    if evidence_values.is_empty() {
        return Err("evidence must not be empty".to_owned());
    }

    let mut evidence = Vec::with_capacity(evidence_values.len());
    let mut evidence_ids = BTreeSet::new();
    for item in evidence_values {
        let id = required_string(item, "evidence_id")?.to_owned();
        if !evidence_ids.insert(id.clone()) {
            return Err(format!("duplicate evidence_id {id}"));
        }

        let kind = required_string(item, "kind")?.to_owned();
        let status = required_string(item, "status")?.to_owned();
        let digest = required_string(item, "sha256")?.to_ascii_lowercase();
        if !valid_sha256(&digest) {
            return Err(format!("evidence {id}: sha256 must be a SHA-256 digest"));
        }
        let local_path = optional_string(item, "local_path")?;
        if let Some(path) = local_path.as_deref() {
            safe_relative_path(path, &format!("evidence {id}.local_path"))?;
        }

        let assumptions_sha256 = optional_string(item, "assumptions_sha256")?;
        let assumptions_local_path = optional_string(item, "assumptions_local_path")?;
        let subject_netlist_sha256 = optional_string(item, "subject_netlist_sha256")?;
        let reference_netlist_sha256 = optional_string(item, "reference_netlist_sha256")?;
        if let Some(assumptions_path) = assumptions_local_path.as_deref() {
            safe_relative_path(
                assumptions_path,
                &format!("evidence {id}.assumptions_local_path"),
            )?;
        }
        for (field, digest) in [
            ("assumptions_sha256", assumptions_sha256.as_deref()),
            ("subject_netlist_sha256", subject_netlist_sha256.as_deref()),
            ("reference_netlist_sha256", reference_netlist_sha256.as_deref()),
        ] {
            if let Some(digest) = digest {
                if !valid_sha256(digest) {
                    return Err(format!("evidence {id}: {field} must be a SHA-256 digest"));
                }
            }
        }
        if assumptions_sha256.is_some() != assumptions_local_path.is_some() {
            return Err(format!(
                "evidence {id}: assumptions_sha256 and assumptions_local_path must appear together"
            ));
        }

        evidence.push(EvidenceSummary {
            id,
            kind,
            status,
            sha256: digest,
            local_path,
            assumptions_sha256,
            assumptions_local_path,
            subject_netlist_sha256,
            reference_netlist_sha256,
            proof_scope: optional_string(item, "proof_scope")?,
            formal_result: optional_string(item, "formal_result")?,
            bound_cycles: optional_u64(item, "bound_cycles")?,
            property_id: optional_string(item, "property_id")?,
        });
    }

    let evidence_lookup = evidence_map(&evidence);

    let trigger = value.get("trigger").ok_or("trigger is required")?;
    let activation = trigger
        .get("activation")
        .ok_or("trigger.activation is required")?;
    let activation_status = required_string(activation, "status")?;
    let payload = value.get("payload").ok_or("payload is required")?;
    let payload_status = required_string(payload, "status")?;
    let paired = payload
        .get("paired_inactive_control")
        .ok_or("payload.paired_inactive_control is required")?;
    let paired_status = required_string(paired, "status")?;

    let mut active_trace_sha256 = None;
    let mut payload_effect_sha256 = None;

    if validity_status == "witness_validated" {
        if ![
            "evaluation_corpus_member",
            "reference_fixture_only",
        ]
        .contains(&dataset_role.as_str())
        {
            return Err(
                "witness_validated profile must be an evaluation corpus member or reference fixture"
                    .to_owned(),
            );
        }
        if activation_status != "observed" {
            return Err(
                "witness_validated requires trigger.activation.status=observed".to_owned(),
            );
        }
        if payload_status != "observed" {
            return Err("witness_validated requires payload.status=observed".to_owned());
        }
        if paired_status != "passed" {
            return Err(
                "witness_validated requires a passed paired inactive control".to_owned(),
            );
        }
        if paired.get("effect_observed").and_then(Value::as_bool) != Some(false) {
            return Err(
                "paired inactive control must explicitly record effect_observed=false".to_owned(),
            );
        }

        let activation_count = optional_u64(activation, "activation_count")?
            .ok_or("observed trigger requires activation_count")?;
        if activation_count == 0 {
            return Err("observed trigger requires activation_count > 0".to_owned());
        }
        let trace_digest = required_string(activation, "trace_sha256")?.to_ascii_lowercase();
        if !valid_sha256(&trace_digest) {
            return Err("trigger.activation.trace_sha256 is invalid".to_owned());
        }
        let active_id = required_string(activation, "witness_evidence_id")?;
        let active_item = evidence_by_id(&evidence_lookup, active_id, "active_trace")?;
        if active_item.sha256 != trace_digest {
            return Err(
                "active trace digest does not match its referenced evidence".to_owned(),
            );
        }
        active_trace_sha256 = Some(trace_digest);

        let payload_digest = required_string(payload, "effect_sha256")?.to_ascii_lowercase();
        if !valid_sha256(&payload_digest) {
            return Err("payload.effect_sha256 is invalid".to_owned());
        }
        let payload_id = required_string(payload, "witness_evidence_id")?;
        let payload_item = evidence_by_id(&evidence_lookup, payload_id, "payload_effect")?;
        if payload_item.sha256 != payload_digest {
            return Err(
                "payload effect digest does not match its referenced evidence".to_owned(),
            );
        }
        payload_effect_sha256 = Some(payload_digest);

        let inactive_id = required_string(paired, "trace_evidence_id")?;
        evidence_by_id(&evidence_lookup, inactive_id, "inactive_trace")?;

        let bounded_proofs = evidence
            .iter()
            .filter(|item| {
                item.kind == "formal_proof"
                    && item.status == "pass"
                    && item.formal_result.as_deref() == Some("bounded_no_counterexample")
            })
            .collect::<Vec<_>>();
        if bounded_proofs.is_empty() {
            return Err(
                "witness_validated requires an explicit bounded_no_counterexample evidence item"
                    .to_owned(),
            );
        }
        for proof in bounded_proofs {
            if proof.proof_scope.as_deref() != Some("bounded")
                || proof.bound_cycles.unwrap_or(0) == 0
                || proof.property_id.is_none()
                || proof.assumptions_sha256.is_none()
                || proof.assumptions_local_path.is_none()
            {
                return Err(format!(
                    "bounded proof {} requires bounded scope, positive bound, property and hashed assumptions bytes",
                    proof.id
                ));
            }
        }
    }

    let mut reference_netlist_sha256 = None;
    let mut reference_local_path = None;
    if validity_status == "benign_equivalent" {
        let equivalence = value.get("equivalence").ok_or("equivalence is required")?;
        if required_string(equivalence, "status")? != "proven" {
            return Err("benign_equivalent requires equivalence.status=proven".to_owned());
        }
        let proof_id = required_string(equivalence, "proof_evidence_id")?;
        let proof = evidence_by_id(&evidence_lookup, proof_id, "equivalence_report")?;
        if proof.formal_result.as_deref() != Some("equivalence_proved") {
            return Err("benign_equivalent requires equivalence_proved evidence".to_owned());
        }
        if !matches!(
            proof.proof_scope.as_deref(),
            Some("unbounded") | Some("assumption_scoped")
        ) {
            return Err(
                "benign equivalence requires unbounded or assumption_scoped proof metadata; bounded evidence is insufficient"
                    .to_owned(),
            );
        }
        if proof.property_id.is_none() {
            return Err("benign equivalence evidence requires property_id".to_owned());
        }

        let assumptions = required_string(equivalence, "assumptions_sha256")?.to_ascii_lowercase();
        if !valid_sha256(&assumptions) {
            return Err("equivalence.assumptions_sha256 is invalid".to_owned());
        }
        let assumptions_path = required_string(equivalence, "assumptions_local_path")?;
        safe_relative_path(assumptions_path, "equivalence.assumptions_local_path")?;
        if proof.assumptions_sha256.as_deref() != Some(assumptions.as_str())
            || proof.assumptions_local_path.as_deref() != Some(assumptions_path)
        {
            return Err(
                "equivalence proof must bind the exact assumptions digest and local bytes"
                    .to_owned(),
            );
        }
        if proof.subject_netlist_sha256.as_deref() != Some(netlist_sha256.as_str()) {
            return Err(
                "equivalence proof subject digest does not match the candidate netlist"
                    .to_owned(),
            );
        }

        let reference_digest =
            required_string(equivalence, "reference_netlist_sha256")?.to_ascii_lowercase();
        if !valid_sha256(&reference_digest) || reference_digest == netlist_sha256 {
            return Err(
                "equivalence.reference_netlist_sha256 must be a distinct valid digest"
                    .to_owned(),
            );
        }
        if proof.reference_netlist_sha256.as_deref() != Some(reference_digest.as_str()) {
            return Err(
                "equivalence proof reference digest does not match the declared golden netlist"
                    .to_owned(),
            );
        }
        let reference_path = required_string(equivalence, "reference_local_path")?.to_owned();
        safe_relative_path(&reference_path, "equivalence.reference_local_path")?;

        reference_netlist_sha256 = Some(reference_digest);
        reference_local_path = Some(reference_path);
    }

    let _ = profile_sha256;
    Ok(ProfileSummary {
        challenge_id,
        dataset_id,
        dataset_role,
        validity_status,
        netlist_sha256,
        subject_local_path,
        evidence,
        evidence_ids,
        active_trace_sha256,
        payload_effect_sha256,
        reference_netlist_sha256,
        reference_local_path,
    })
}

fn verify_file_under_root(
    canonical_root: &Path,
    relative_path: &str,
    expected_sha256: &str,
    label: &str,
) -> Result<(), String> {
    safe_relative_path(relative_path, label)?;
    let components = Path::new(relative_path).components().collect::<Vec<_>>();
    let mut candidate = canonical_root.to_path_buf();

    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err(format!("{label} contains a non-normal path component"));
        };
        candidate.push(name);
        let metadata = fs::symlink_metadata(&candidate)
            .map_err(|error| format!("{label} {}: {error}", candidate.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("{label} {} must not be a symlink", candidate.display()));
        }
        if index + 1 == components.len() {
            if !metadata.is_file() {
                return Err(format!("{label} {} must be a regular file", candidate.display()));
            }
        } else if !metadata.is_dir() {
            return Err(format!("{label} {} must be a directory", candidate.display()));
        }
    }

    let resolved = fs::canonicalize(&candidate)
        .map_err(|error| format!("canonicalize {label} {}: {error}", candidate.display()))?;
    if !resolved.starts_with(canonical_root) {
        return Err(format!("{label} {relative_path} escapes evidence root"));
    }
    let actual = sha256_file(&resolved)?;
    if actual != expected_sha256 {
        return Err(format!(
            "{label} {relative_path}: SHA-256 mismatch (expected {expected_sha256}, got {actual})"
        ));
    }
    Ok(())
}

fn verify_profile_bytes(root: &Path, summary: &ProfileSummary) -> Result<Value, String> {
    let root_meta = fs::symlink_metadata(root)
        .map_err(|error| format!("stat evidence root {}: {error}", root.display()))?;
    if root_meta.file_type().is_symlink() || !root_meta.is_dir() {
        return Err("evidence root must be a regular non-symlink directory".to_owned());
    }
    let canonical_root = fs::canonicalize(root)
        .map_err(|error| format!("canonicalize evidence root {}: {error}", root.display()))?;

    let mut verified = Vec::new();
    let mut unique = BTreeSet::new();

    let subject_key = format!("{}:{}", summary.subject_local_path, summary.netlist_sha256);
    if unique.insert(subject_key) {
        verify_file_under_root(
            &canonical_root,
            &summary.subject_local_path,
            &summary.netlist_sha256,
            "subject netlist",
        )?;
        verified.push(json!({
            "role": "subject_netlist",
            "path": summary.subject_local_path,
            "sha256": summary.netlist_sha256,
        }));
    }

    match (
        summary.reference_netlist_sha256.as_deref(),
        summary.reference_local_path.as_deref(),
    ) {
        (Some(digest), Some(path)) => {
            let key = format!("{path}:{digest}");
            if unique.insert(key) {
                verify_file_under_root(
                    &canonical_root,
                    path,
                    digest,
                    "equivalence reference netlist",
                )?;
                verified.push(json!({
                    "role": "equivalence_reference_netlist",
                    "path": path,
                    "sha256": digest,
                }));
            }
        }
        (None, None) => {}
        _ => {
            return Err(
                "equivalence reference digest/path must appear together".to_owned(),
            );
        }
    }

    for item in &summary.evidence {
        let path = item.local_path.as_deref().ok_or_else(|| {
            format!(
                "evidence {}: local_path is required with --evidence-root",
                item.id
            )
        })?;
        let key = format!("{path}:{}", item.sha256);
        if unique.insert(key) {
            verify_file_under_root(
                &canonical_root,
                path,
                &item.sha256,
                &format!("evidence {} artifact", item.id),
            )?;
            verified.push(json!({
                "role": item.kind,
                "evidence_id": item.id,
                "path": path,
                "sha256": item.sha256,
            }));
        }

        match (
            item.assumptions_sha256.as_deref(),
            item.assumptions_local_path.as_deref(),
        ) {
            (Some(digest), Some(path)) => {
                let key = format!("{path}:{digest}");
                if unique.insert(key) {
                    verify_file_under_root(
                        &canonical_root,
                        path,
                        digest,
                        &format!("evidence {} assumptions", item.id),
                    )?;
                    verified.push(json!({
                        "role": "assumptions",
                        "evidence_id": item.id,
                        "path": path,
                        "sha256": digest,
                    }));
                }
            }
            (None, None) => {}
            _ => {
                return Err(format!(
                    "evidence {}: assumptions digest/path must appear together",
                    item.id
                ));
            }
        }
    }

    Ok(json!({
        "status": "verified",
        "verified_file_count": verified.len(),
        "root_binding": "relative_paths_under_canonical_non_symlink_root",
        "files": verified,
        "limitations": [
            "Verifies bytes at recorded local paths only.",
            "Does not authenticate upstream publisher identity, tool execution or signatures.",
            "Not resistant to adversarial concurrent replacement between metadata checks and file open."
        ]
    }))
}

fn receipt_target(profile_path: &Path, receipt_path: &Path) -> Result<PathBuf, String> {
    let profile_parent = profile_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let profile_parent = fs::canonicalize(profile_parent)
        .map_err(|error| format!("canonicalize profile directory: {error}"))?;
    let receipt_parent = receipt_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let receipt_parent = fs::canonicalize(receipt_parent)
        .map_err(|error| format!("canonicalize receipt directory: {error}"))?;
    let filename = receipt_path
        .file_name()
        .ok_or("receipt path must include a filename")?;
    let target = receipt_parent.join(filename);
    if target.starts_with(&profile_parent) {
        return Err("receipt must be outside the profile directory".to_owned());
    }
    Ok(target)
}

fn run(args: Args) -> Result<(), String> {
    let profile_meta = fs::symlink_metadata(&args.profile)
        .map_err(|error| format!("stat profile {}: {error}", args.profile.display()))?;
    if profile_meta.file_type().is_symlink() || !profile_meta.is_file() {
        return Err("profile must be a regular non-symlink file".to_owned());
    }

    let profile_bytes = fs::read(&args.profile)
        .map_err(|error| format!("read profile {}: {error}", args.profile.display()))?;
    let profile_sha256 = sha256_bytes(&profile_bytes);
    let value: Value = serde_json::from_slice(&profile_bytes)
        .map_err(|error| format!("parse profile JSON: {error}"))?;
    let summary = validate_profile(&value, profile_sha256.clone())?;

    let byte_integrity = match args.evidence_root.as_deref() {
        Some(root) => verify_profile_bytes(root, &summary)?,
        None => json!({
            "status": "not_requested",
            "verified_file_count": 0,
            "files": []
        }),
    };

    let receipt_path = receipt_target(&args.profile, &args.receipt)?;
    let receipt = json!({
        "schema_version": "hardsec-challenge-profile-audit-receipt-v1",
        "result": "pass",
        "claim_scope": "challenge_profile_schema_semantics_and_optional_referenced_byte_integrity",
        "challenge_id": summary.challenge_id,
        "dataset_id": summary.dataset_id,
        "dataset_role": summary.dataset_role,
        "validity_status": summary.validity_status,
        "profile_sha256": profile_sha256,
        "netlist_sha256": summary.netlist_sha256,
        "evidence_count": summary.evidence.len(),
        "evidence_ids": summary.evidence_ids,
        "active_trace_sha256": summary.active_trace_sha256,
        "payload_effect_sha256": summary.payload_effect_sha256,
        "byte_integrity": byte_integrity,
        "security_verdict": null,
        "limitations": [
            "This is a metadata and evidence-binding audit, not a hardware security verdict.",
            "A witness-validated fixture is not necessarily a member of the source paper's evaluation corpus.",
            "Finite no-witness and bounded no-counterexample results do not establish global unreachability.",
            "Benign equivalence requires an independently verified whole-design equivalence result under explicit assumptions."
        ]
    });

    let serialized = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("serialize receipt: {error}"))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)
        .map_err(|error| format!("create receipt {}: {error}", receipt_path.display()))?;
    let write_result = output
        .write_all(&serialized)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all());
    drop(output);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&receipt_path);
        return Err(format!("write receipt {}: {error}", receipt_path.display()));
    }

    println!("HARDSEC challenge profile audit: PASS");
    println!("challenge: {}", summary.challenge_id);
    println!("role: {}", summary.dataset_role);
    println!("validity: {}", summary.validity_status);
    println!("evidence items: {}", summary.evidence.len());
    println!("receipt: {}", receipt_path.display());
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("HARDSEC challenge profile audit failed: {error}");
                ExitCode::FAILURE
            }
        },
        Ok(None) => {
            println!(
                "usage: hardsec_challenge_profile_audit --profile FILE --receipt NEW_FILE [--evidence-root DIR]"
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("HARDSEC challenge profile audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/hardsec/seqcube-challenge-profile.json"))
            .expect("valid fixture JSON")
    }

    #[test]
    fn accepts_seqcube_reference_fixture_with_explicit_unknown_metrics() {
        let value = fixture();
        let summary = validate_profile(&value, "0".repeat(64)).unwrap();
        assert_eq!(summary.validity_status, "witness_validated");
        assert_eq!(summary.dataset_role, "reference_fixture_only");
        assert_eq!(summary.evidence.len(), 5);
        assert_eq!(
            value["trigger"]["scoap"]["status"],
            json!("not_measured")
        );
        assert_eq!(
            value["structure"]["graph_metrics"]["status"],
            json!("not_measured")
        );
        assert_eq!(value["resources"]["status"], json!("not_measured"));
    }

    #[test]
    fn rejects_witness_validated_without_an_observed_trigger() {
        let mut value = fixture();
        value["trigger"]["activation"]["status"] = json!("not_observed_under_budget");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("activation.status=observed")
        );
    }

    #[test]
    fn rejects_witness_validated_without_observed_payload_effect() {
        let mut value = fixture();
        value["payload"]["status"] = json!("not_observed_under_budget");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("payload.status=observed")
        );
    }

    #[test]
    fn rejects_missing_paired_inactive_control() {
        let mut value = fixture();
        value["payload"]["paired_inactive_control"]["status"] = json!("unknown");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("passed paired inactive control")
        );
    }

    #[test]
    fn rejects_inactive_control_with_payload_effect() {
        let mut value = fixture();
        value["payload"]["paired_inactive_control"]["effect_observed"] = json!(true);
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("effect_observed=false")
        );
    }

    #[test]
    fn rejects_active_trace_digest_mismatch() {
        let mut value = fixture();
        value["trigger"]["activation"]["trace_sha256"] = json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("does not match its referenced evidence")
        );
    }

    #[test]
    fn rejects_payload_effect_digest_mismatch() {
        let mut value = fixture();
        value["payload"]["effect_sha256"] = json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("does not match its referenced evidence")
        );
    }

    #[test]
    fn rejects_missing_active_witness_reference() {
        let mut value = fixture();
        value["trigger"]["activation"]["witness_evidence_id"] = json!("missing-active-witness");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("unknown evidence_id")
        );
    }

    #[test]
    fn rejects_missing_bounded_formal_evidence() {
        let mut value = fixture();
        value["evidence"]
            .as_array_mut()
            .unwrap()
            .retain(|item| item["kind"] != "formal_proof");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("bounded_no_counterexample")
        );
    }

    #[test]
    fn rejects_bounded_formal_result_marked_unbounded() {
        let mut value = fixture();
        value["evidence"][4]["proof_scope"] = json!("unbounded");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("bounded scope")
        );
    }

    #[test]
    fn rejects_bounded_formal_evidence_without_positive_bound() {
        let mut value = fixture();
        value["evidence"][4]["bound_cycles"] = json!(0);
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("positive bound")
        );
    }

    #[test]
    fn rejects_bounded_formal_evidence_without_hashed_assumptions() {
        let mut value = fixture();
        value["evidence"][4].as_object_mut().unwrap().remove("assumptions_sha256");
        value["evidence"][4].as_object_mut().unwrap().remove("assumptions_local_path");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("hashed assumptions bytes")
        );
    }

    #[test]
    fn rejects_duplicate_evidence_ids() {
        let mut value = fixture();
        let duplicate = value["evidence"][0].clone();
        value["evidence"].as_array_mut().unwrap().push(duplicate);
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("duplicate evidence_id")
        );
    }

    #[test]
    fn rejects_invalid_netlist_digest() {
        let mut value = fixture();
        value["subject"]["netlist_sha256"] = json!("not-a-digest");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("subject.netlist_sha256")
        );
    }

    #[test]
    fn rejects_path_traversal_in_subject() {
        let mut value = fixture();
        value["subject"]["local_path"] = json!("../outside.v");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("subject.local_path")
        );
    }

    #[test]
    fn rejects_unknown_validity_status() {
        let mut value = fixture();
        value["validity_status"] = json!("secure");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("unsupported validity_status")
        );
    }

    #[test]
    fn rejects_benign_equivalent_without_equivalence_proof() {
        let mut value = fixture();
        value["validity_status"] = json!("benign_equivalent");
        value["equivalence"]["status"] = json!("unknown");
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("equivalence.status=proven")
        );
    }

    #[test]
    fn rejects_benign_equivalent_without_typed_equivalence_evidence() {
        let mut value = fixture();
        value["validity_status"] = json!("benign_equivalent");
        value["equivalence"] = json!({
            "status":"proven",
            "checker":"EQY",
            "tool_version":"pinned",
            "assumptions_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "proof_evidence_id":"seqcube-active-cover",
            "reference_netlist_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        });
        assert!(
            validate_profile(&value, "0".repeat(64))
                .unwrap_err()
                .contains("expected equivalence_report")
        );
    }

    #[test]
    fn accepts_unknown_invalid_as_a_non_confirmatory_profile() {
        let mut value = fixture();
        value["validity_status"] = json!("unknown_invalid");
        value["dataset_role"] = json!("exploratory_only");
        value["trigger"]["activation"]["status"] = json!("unknown");
        value["payload"]["status"] = json!("unknown");
        value["payload"]["paired_inactive_control"]["status"] = json!("unknown");
        let summary = validate_profile(&value, "0".repeat(64)).unwrap();
        assert_eq!(summary.validity_status, "unknown_invalid");
        assert_eq!(summary.dataset_role, "exploratory_only");
    }
}
