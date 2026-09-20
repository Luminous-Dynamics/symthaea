use anyhow::{Context, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::cargo_execution_contract::validate_intent;
use crate::prepared_cargo_intent_v2::{
    PreparedCargoExecutionIntentV2, SourceGitStateObservationModel,
};

const RECEIPT_SCHEMA: &str = "symthaea.prepared-cargo-execution-intent.v2";
const HASH_DOMAIN: &[u8] = b"symthaea.prepared-cargo-execution-intent.v2\0";

#[derive(Serialize)]
struct PreparedIntentIdentity<'a> {
    schema: &'static str,
    base_intent_id: &'a str,
    runtime_binding_id: &'a str,
    materialization_receipt_id: &'a str,
    tool_attestation_id: &'a str,
    git_worktree_state_id: &'a str,
    source_git_state_observation: SourceGitStateObservationModel,
}

pub(crate) fn verify_local(
    mut stored: PreparedCargoExecutionIntentV2,
) -> anyhow::Result<PreparedCargoExecutionIntentV2> {
    if stored.schema != RECEIPT_SCHEMA {
        bail!("unsupported prepared Cargo execution intent v2 schema: {}", stored.schema);
    }
    normalize_digest("prepared_intent_id", &mut stored.prepared_intent_id)?;
    normalize_digest("runtime_binding_id", &mut stored.runtime_binding_id)?;
    normalize_digest(
        "materialization_receipt_id",
        &mut stored.materialization_receipt_id,
    )?;
    normalize_digest("tool_attestation_id", &mut stored.tool_attestation_id)?;
    normalize_digest(
        "git_worktree_state_id",
        &mut stored.git_worktree_state_id,
    )?;
    validate_intent(&mut stored.base_intent)?;

    if stored.base_intent.git_worktree_state_before.as_deref()
        != Some(stored.git_worktree_state_id.as_str())
    {
        bail!("prepared intent v2 base intent is not bound to its declared Git-state subject");
    }
    if stored.source_git_state_observation
        != SourceGitStateObservationModel::SequentialNotAtomic
    {
        bail!("unsupported source/Git-state observation model");
    }

    let identity = PreparedIntentIdentity {
        schema: RECEIPT_SCHEMA,
        base_intent_id: &stored.base_intent.intent_id,
        runtime_binding_id: &stored.runtime_binding_id,
        materialization_receipt_id: &stored.materialization_receipt_id,
        tool_attestation_id: &stored.tool_attestation_id,
        git_worktree_state_id: &stored.git_worktree_state_id,
        source_git_state_observation: stored.source_git_state_observation,
    };
    let bytes = serde_json::to_vec(&identity)
        .context("serialize stored Git-bound prepared Cargo execution intent")?;
    let computed = domain_sha256(HASH_DOMAIN, &bytes);
    if computed != stored.prepared_intent_id {
        bail!(
            "prepared Cargo execution intent v2 identity mismatch: declared {}, computed {}",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_execution_contract::{
        CargoExecutionIntentSpec, build_intent,
    };

    const LEGACY_INTENT_INPUT_SCHEMA: &str = "symthaea.cargo-execution-intent-input.v1";

    fn digest(seed: char) -> String {
        assert!(seed.is_ascii());
        format!("{:02x}", u32::from(seed)).repeat(32)
    }

    fn receipt() -> PreparedCargoExecutionIntentV2 {
        let git_worktree_state_id = digest('g');
        let base_intent = build_intent(
            CargoExecutionIntentSpec {
                schema: LEGACY_INTENT_INPUT_SCHEMA.into(),
                git_worktree_state_before: Some(git_worktree_state_id.clone()),
                build_context_id: digest('b'),
                invocation_id: digest('c'),
                plan_id: Some(digest('d')),
                transaction_id: Some(digest('e')),
                adapter_semantics_digest: digest('f'),
            },
            &digest('a'),
            &digest('1'),
        )
        .unwrap();
        let runtime_binding_id = digest('2');
        let materialization_receipt_id = digest('3');
        let tool_attestation_id = digest('4');
        let source_git_state_observation = SourceGitStateObservationModel::SequentialNotAtomic;
        let identity = PreparedIntentIdentity {
            schema: RECEIPT_SCHEMA,
            base_intent_id: &base_intent.intent_id,
            runtime_binding_id: &runtime_binding_id,
            materialization_receipt_id: &materialization_receipt_id,
            tool_attestation_id: &tool_attestation_id,
            git_worktree_state_id: &git_worktree_state_id,
            source_git_state_observation,
        };
        let prepared_intent_id = domain_sha256(
            HASH_DOMAIN,
            &serde_json::to_vec(&identity).unwrap(),
        );
        PreparedCargoExecutionIntentV2 {
            prepared_intent_id,
            schema: RECEIPT_SCHEMA.into(),
            base_intent,
            runtime_binding_id,
            materialization_receipt_id,
            tool_attestation_id,
            git_worktree_state_id,
            source_git_state_observation,
        }
    }

    #[test]
    fn canonical_v2_receipt_verifies() {
        let value = receipt();
        assert_eq!(verify_local(value.clone()).unwrap(), value);
    }

    #[test]
    fn identity_substitution_rejects() {
        let mut bad = receipt();
        bad.prepared_intent_id = digest('9');
        assert!(verify_local(bad).is_err());
    }

    #[test]
    fn explicit_git_subject_must_equal_base_intent_git_subject() {
        let mut bad = receipt();
        bad.git_worktree_state_id = digest('8');
        let identity = PreparedIntentIdentity {
            schema: RECEIPT_SCHEMA,
            base_intent_id: &bad.base_intent.intent_id,
            runtime_binding_id: &bad.runtime_binding_id,
            materialization_receipt_id: &bad.materialization_receipt_id,
            tool_attestation_id: &bad.tool_attestation_id,
            git_worktree_state_id: &bad.git_worktree_state_id,
            source_git_state_observation: bad.source_git_state_observation,
        };
        bad.prepared_intent_id = domain_sha256(
            HASH_DOMAIN,
            &serde_json::to_vec(&identity).unwrap(),
        );
        assert!(verify_local(bad).is_err());
    }
}
