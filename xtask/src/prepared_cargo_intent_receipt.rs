use anyhow::{Context, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::cargo_execution_contract::validate_intent;
use crate::prepared_cargo_intent::{
    GitWorktreeStateBinding, PreparedCargoExecutionIntent,
};

const RECEIPT_SCHEMA: &str = "symthaea.prepared-cargo-execution-intent.v1";
const HASH_DOMAIN: &[u8] = b"symthaea.prepared-cargo-execution-intent.v1\0";

#[derive(Serialize)]
struct PreparedIntentIdentity<'a> {
    schema: &'static str,
    base_intent_id: &'a str,
    runtime_binding_id: &'a str,
    materialization_receipt_id: &'a str,
    tool_attestation_id: &'a str,
    git_worktree_state_binding: &'a GitWorktreeStateBinding,
}

pub(crate) fn verify_local(
    mut stored: PreparedCargoExecutionIntent,
) -> anyhow::Result<PreparedCargoExecutionIntent> {
    if stored.schema != RECEIPT_SCHEMA {
        bail!("unsupported prepared Cargo execution intent schema: {}", stored.schema);
    }
    normalize_digest("prepared_intent_id", &mut stored.prepared_intent_id)?;
    normalize_digest("runtime_binding_id", &mut stored.runtime_binding_id)?;
    normalize_digest(
        "materialization_receipt_id",
        &mut stored.materialization_receipt_id,
    )?;
    normalize_digest("tool_attestation_id", &mut stored.tool_attestation_id)?;
    validate_intent(&mut stored.base_intent)?;

    if stored.base_intent.git_worktree_state_before.is_some() {
        bail!("prepared intent v1 must not carry a Git worktree-state ID");
    }
    if stored.git_worktree_state_binding != GitWorktreeStateBinding::NotConverged {
        bail!("prepared intent v1 must record Git worktree-state as not converged");
    }

    let identity = PreparedIntentIdentity {
        schema: RECEIPT_SCHEMA,
        base_intent_id: &stored.base_intent.intent_id,
        runtime_binding_id: &stored.runtime_binding_id,
        materialization_receipt_id: &stored.materialization_receipt_id,
        tool_attestation_id: &stored.tool_attestation_id,
        git_worktree_state_binding: &stored.git_worktree_state_binding,
    };
    let bytes = serde_json::to_vec(&identity)
        .context("serialize stored prepared Cargo execution intent")?;
    let computed = domain_sha256(HASH_DOMAIN, &bytes);
    if computed != stored.prepared_intent_id {
        bail!(
            "prepared Cargo execution intent identity mismatch: declared {}, computed {}",
            stored.prepared_intent_id,
            computed
        );
    }
    Ok(stored)
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    value.make_ascii_lowercase();
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}
