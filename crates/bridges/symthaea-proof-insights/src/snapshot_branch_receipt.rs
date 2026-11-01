// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared semantic validator and byte-identity verifier for PITB's R5
//! post-elaboration Lean snapshot receipts.

use std::{
    collections::BTreeSet,
    fs,
    path::Path,
};

use anyhow::{Context, Result, bail};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: &str = "pitb.sorrydb.snapshot-branch-receipt.v0";
const WIRE_SCHEMA: &str = "pitb.lean-expr-wire.v1";
const CACHE_POLICY: &str = "discard_and_reinitialize_per_branch";

#[derive(Debug, Clone)]
pub struct ReceiptSummary {
    pub receipt_sha256: String,
    pub task_id: String,
    pub repository_commit: String,
    pub environment_sha256: String,
    pub proof_state_sha256: String,
    pub snapshot_id_sha256: Option<String>,
    pub branch_count: u64,
    pub branch_ids: Vec<String>,
    pub run_status: String,
}

fn string<'a>(value: &'a Value, path: &str) -> Result<&'a str> {
    value
        .pointer(path)
        .and_then(Value::as_str)
        .with_context(|| format!("snapshot receipt field {path} must be a string"))
}

fn optional_string<'a>(value: &'a Value, path: &str) -> Result<Option<&'a str>> {
    let field = value
        .pointer(path)
        .with_context(|| format!("snapshot receipt field {path} is missing"))?;
    match field {
        Value::Null => Ok(None),
        Value::String(value) => Ok(Some(value)),
        _ => bail!("snapshot receipt field {path} must be a string or null"),
    }
}

fn integer(value: &Value, path: &str) -> Result<u64> {
    value
        .pointer(path)
        .and_then(Value::as_u64)
        .with_context(|| format!("snapshot receipt field {path} must be a nonnegative integer"))
}

fn boolean(value: &Value, path: &str) -> Result<bool> {
    value
        .pointer(path)
        .and_then(Value::as_bool)
        .with_context(|| format!("snapshot receipt field {path} must be boolean"))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_digest(value: &str, label: &str, length: usize) -> Result<()> {
    if !is_lower_hex(value, length) {
        bail!("{label} must be {length} lowercase hexadecimal characters");
    }
    Ok(())
}

fn safe_relative_path(value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || value.contains('\\')
        || value.contains('\0')
        || value.contains(':')
        || value.split('/').any(|part| part.is_empty() || part == "." || part == "..")
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::CurDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        bail!("snapshot receipt task source_path is unsafe");
    }
    Ok(())
}

pub fn compute_live_proof_state_sha256(value: &Value) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"PITB-LIVE-PROOF-STATE-V0\\0");
    for path in [
        "/environment/environment_sha256",
        "/proof_state/goal_metadata_sha256",
        "/proof_state/target_expr_sha256",
        "/proof_state/local_context_sha256",
        "/proof_state/metavar_context_sha256",
    ] {
        let digest = string(value, path)?;
        require_digest(digest, path, 64)?;
        let bytes = hex::decode(digest).context("decode live proof-state component digest")?;
        hasher.update(bytes);
    }
    Ok(hex::encode(hasher.finalize()))
}

pub fn validate_value(value: &Value, receipt_sha256: String) -> Result<ReceiptSummary> {
    if string(value, "/schema_version")? != SCHEMA_VERSION {
        bail!("unsupported snapshot receipt schema version");
    }
    let evidence_class = string(value, "/evidence_class")?;
    if !matches!(evidence_class, "synthetic_contract_fixture" | "measured_run") {
        bail!("unknown snapshot receipt evidence class");
    }
    let run_status = string(value, "/run_status")?;
    if !matches!(run_status, "not_attempted" | "measured" | "failed" | "unknown") {
        bail!("unknown snapshot receipt run_status");
    }

    let task_id = string(value, "/task/task_id")?.to_owned();
    let repository_commit = string(value, "/task/repository_commit")?.to_owned();
    require_digest(&task_id, "task_id", 64)?;
    require_digest(&repository_commit, "repository_commit", 40)?;
    let source_path = string(value, "/task/source_path")?;
    safe_relative_path(source_path)?;
    let source_sha256 = string(value, "/task/source_sha256")?;
    require_digest(source_sha256, "source_sha256", 64)?;

    let environment_sha256 = string(value, "/environment/environment_sha256")?.to_owned();
    require_digest(&environment_sha256, "environment_sha256", 64)?;
    let computed_live_state = compute_live_proof_state_sha256(value)?;
    let claimed_live_state = string(value, "/proof_state/proof_state_sha256")?;
    if claimed_live_state != computed_live_state {
        bail!("snapshot proof_state_sha256 does not bind environment, goal, target, local context and MetavarContext");
    }
    for path in [
        "/environment/lean_binary_sha256",
        "/environment/lean_runtime_manifest_sha256",
        "/environment/imported_olean_manifest_sha256",
        "/proof_state/goal_metadata_sha256",
        "/proof_state/target_expr_sha256",
        "/proof_state/local_context_sha256",
        "/proof_state/metavar_context_sha256",
        "/proof_state/proof_state_sha256",
        "/server/lean_server_binary_sha256",
        "/server/rpc_contract_sha256",
        "/snapshot/snapshot_payload_sha256",
    ] {
        require_digest(string(value, path)?, path, 64)?;
    }
    require_digest(
        string(value, "/server/snapshot_extension_commit")?,
        "snapshot_extension_commit",
        40,
    )?;
    if string(value, "/proof_state/wire_schema")? != WIRE_SCHEMA {
        bail!("snapshot proof state is not PITB Lean Expr wire v1");
    }
    if string(value, "/execution_contract/capture_stage")? != "post_elaboration_pre_tactic"
        || string(value, "/execution_contract/environment_policy")?
            != "immutable_shared_by_reference"
        || string(value, "/execution_contract/mutable_state_policy")?
            != "independent_metavar_context_copy_per_branch"
        || string(value, "/execution_contract/branch_workspace_policy")?
            != "private_workspace_per_branch"
        || string(value, "/execution_contract/typeclass_cache_policy")? != CACHE_POLICY
    {
        bail!("snapshot receipt execution contract differs from the qualified R5 policy");
    }

    let branch_rows = value
        .pointer("/branches")
        .and_then(Value::as_array)
        .context("snapshot receipt branches must be an array")?;
    let branch_count = integer(value, "/measurements/branch_count")?;
    let capture_wall_ms = integer(value, "/measurements/capture_wall_ms")?;
    let restore_p50 = integer(value, "/measurements/branch_restore_p50_ms")?;
    let restore_p95 = integer(value, "/measurements/branch_restore_p95_ms")?;
    let environment_reload_count = integer(value, "/measurements/environment_reload_count")?;
    let theorem_reelaboration_count =
        integer(value, "/measurements/theorem_reelaboration_count")?;
    let stale_cache_observations = integer(value, "/measurements/stale_cache_observations")?;
    let elapsed_wall_ms = integer(value, "/measurements/elapsed_wall_ms")?;
    let snapshot_status = string(value, "/snapshot/capture_status")?;
    let snapshot_id = optional_string(value, "/snapshot/snapshot_id_sha256")?.map(str::to_owned);

    if evidence_class == "synthetic_contract_fixture" {
        if run_status != "not_attempted"
            || snapshot_status != "not_attempted"
            || snapshot_id.is_some()
            || !branch_rows.is_empty()
            || branch_count != 0
            || capture_wall_ms != 0
            || restore_p50 != 0
            || restore_p95 != 0
            || environment_reload_count != 0
            || theorem_reelaboration_count != 0
            || stale_cache_observations != 0
            || elapsed_wall_ms != 0
        {
            bail!("synthetic snapshot receipt contains measured execution claims");
        }
    } else if run_status == "measured" {
        let snapshot_id = snapshot_id
            .as_deref()
            .context("measured R5 execution has no snapshot ID digest")?;
        require_digest(snapshot_id, "snapshot_id_sha256", 64)?;
        if snapshot_status != "captured" {
            bail!("measured R5 execution did not capture a proof-state snapshot");
        }
        if branch_rows.is_empty() || branch_count == 0 || branch_count as usize != branch_rows.len() {
            bail!("R5 branch_count does not equal the branch receipt count");
        }
        if capture_wall_ms == 0 || elapsed_wall_ms < capture_wall_ms {
            bail!("R5 timing is inconsistent with snapshot capture");
        }
        if restore_p50 > restore_p95 {
            bail!("R5 branch restore p50 exceeds p95");
        }
        if environment_reload_count != 0 || theorem_reelaboration_count != 0 {
            bail!("R5 branch execution reloaded the environment or re-elaborated the theorem body");
        }
        if stale_cache_observations != 0 {
            bail!("R5 observed stale cache state; result is not eligible");
        }
        let mut branch_ids = BTreeSet::new();
        for (index, branch) in branch_rows.iter().enumerate() {
            let branch_id = string(branch, &format!("/branches/{index}/branch_id"))?;
            if branch_id.is_empty()
                || branch_id.len() > 64
                || !branch_id.bytes().all(|byte| {
                    byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'
                })
                || !branch_ids.insert(branch_id)
            {
                bail!("R5 branch IDs must be unique lowercase kebab-case identifiers");
            }
            ordered_branch_ids.push(branch_id.to_owned());
            let branch_snapshot =
                string(branch, &format!("/branches/{index}/base_snapshot_sha256"))?;
            if branch_snapshot != snapshot_id {
                bail!("R5 branch does not descend from the captured snapshot");
            }
            if !boolean(branch, &format!("/branches/{index}/mutable_context_isolated"))? {
                bail!("R5 branch shares a mutable MetavarContext");
            }
            if string(branch, &format!("/branches/{index}/cache_policy"))? != CACHE_POLICY {
                bail!("R5 branch cache policy is not qualified");
            }
            let status = string(branch, &format!("/branches/{index}/status"))?;
            let candidate = optional_string(branch, &format!("/branches/{index}/candidate_sha256"))?;
            match (status, candidate) {
                ("candidate_available", Some(digest)) => {
                    require_digest(digest, "branch candidate SHA-256", 64)?;
                }
                ("candidate_available", None) => {
                    bail!("R5 available candidate has no digest");
                }
                (_, Some(_)) => bail!("R5 non-candidate branch unexpectedly has a candidate digest"),
                ("not_attempted" | "failed" | "cancelled" | "budget_exhausted", None) => {}
                _ => bail!("unknown R5 branch status"),
            }
        }
    } else if run_status == "not_attempted" {
        if snapshot_status != "not_attempted" || snapshot_id.is_some() || !branch_rows.is_empty() {
            bail!("not-attempted R5 receipt contains snapshot or branch execution");
        }
    } else if snapshot_status == "captured" && snapshot_id.is_none() {
        bail!("captured snapshot status has no snapshot identity");
    }

    Ok(ReceiptSummary {
        receipt_sha256,
        task_id,
        repository_commit,
        environment_sha256,
        proof_state_sha256: string(value, "/proof_state/proof_state_sha256")?.to_owned(),
        snapshot_id_sha256: snapshot_id,
        branch_count,
        branch_ids: ordered_branch_ids,
        run_status: run_status.to_owned(),
    })
}

pub fn verify_bytes(bytes: &[u8], expected_sha256: Option<&str>) -> Result<ReceiptSummary> {
    let actual_sha256 = hex::encode(Sha256::digest(bytes));
    if let Some(expected) = expected_sha256 {
        require_digest(expected, "expected snapshot receipt SHA-256", 64)?;
        if expected != actual_sha256 {
            bail!("snapshot receipt bytes do not match the branch-ledger SHA-256 link");
        }
    }
    let value: Value = serde_json::from_slice(bytes).context("parse snapshot receipt JSON")?;
    validate_value(&value, actual_sha256)
}

pub fn verify_file(path: &Path, expected_sha256: Option<&str>) -> Result<ReceiptSummary> {
    let bytes = fs::read(path).with_context(|| format!("read snapshot receipt {}", path.display()))?;
    verify_bytes(&bytes, expected_sha256)
}
