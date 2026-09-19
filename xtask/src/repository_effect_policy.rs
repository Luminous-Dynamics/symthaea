use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

use crate::repository_snapshot_diff::{RepositorySourceDiff, compare};
use crate::repository_snapshot_receipt::ValidatedSnapshotReceipt;

const POLICY_SCHEMA: &str = "symthaea.repository-effect-policy.v1";
const POLICY_HASH_DOMAIN: &[u8] = b"symthaea.repository-effect-policy.v1\0";
const REPORT_SCHEMA: &str = "symthaea.repository-effect-evaluation.v1";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EffectMode {
    ReadOnlySource,
    ExactTransition,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EffectPolicySpec {
    pub schema: String,
    pub mode: EffectMode,
    #[serde(default)]
    pub expected_diff_id: Option<String>,
}

#[derive(Serialize)]
struct PolicyIdentity<'a> {
    schema: &'static str,
    mode: EffectMode,
    expected_diff_id: &'a Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EffectEvaluation {
    pub schema: &'static str,
    pub policy_id: String,
    pub mode: EffectMode,
    pub base_snapshot_id: String,
    pub head_snapshot_id: String,
    pub observed_diff_id: String,
    pub expected_diff_id: Option<String>,
    pub allowed: bool,
    pub reason: String,
}

pub fn run(
    base_path: &Path,
    head_path: &Path,
    policy_path: &Path,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let base = ValidatedSnapshotReceipt::load(base_path)?;
    let head = ValidatedSnapshotReceipt::load(head_path)?;
    let policy_bytes = fs::read(policy_path)
        .with_context(|| format!("read repository effect policy {}", policy_path.display()))?;
    let mut policy: EffectPolicySpec = serde_json::from_slice(&policy_bytes)
        .with_context(|| format!("parse repository effect policy {}", policy_path.display()))?;
    let policy_id = validate_and_identify_policy(&mut policy)?;
    let diff = compare(&base, &head)?;
    let evaluation = evaluate(policy_id, &policy, &diff);

    let mut rendered = serde_json::to_string_pretty(&evaluation)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create effect-evaluation output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write repository effect evaluation {}", path.display()))?;
        println!("Repository effect evaluation written to {}", path.display());
    } else {
        print!("{rendered}");
    }

    if !evaluation.allowed {
        bail!("repository effect policy rejected observed transition: {}", evaluation.reason);
    }
    Ok(())
}

pub(crate) fn validate_and_identify_policy(policy: &mut EffectPolicySpec) -> anyhow::Result<String> {
    if policy.schema != POLICY_SCHEMA {
        bail!("unsupported repository effect policy schema: {}", policy.schema);
    }

    match policy.mode {
        EffectMode::ReadOnlySource => {
            if policy.expected_diff_id.is_some() {
                bail!("read_only_source policy must not provide expected_diff_id");
            }
        }
        EffectMode::ExactTransition => {
            let expected = policy
                .expected_diff_id
                .as_mut()
                .context("exact_transition policy requires expected_diff_id")?;
            validate_sha256("expected_diff_id", expected)?;
            expected.make_ascii_lowercase();
        }
    }

    let identity = PolicyIdentity {
        schema: POLICY_SCHEMA,
        mode: policy.mode,
        expected_diff_id: &policy.expected_diff_id,
    };
    let bytes = serde_json::to_vec(&identity).context("serialize repository effect policy identity")?;
    Ok(domain_sha256(POLICY_HASH_DOMAIN, &bytes))
}

pub(crate) fn evaluate(
    policy_id: String,
    policy: &EffectPolicySpec,
    diff: &RepositorySourceDiff,
) -> EffectEvaluation {
    let (allowed, reason) = match policy.mode {
        EffectMode::ReadOnlySource => {
            if diff.identical {
                (true, "repository source subject remained exactly unchanged".to_string())
            } else {
                (
                    false,
                    format!(
                        "read-only policy requires identical source subjects; observed diff {}",
                        diff.diff_id
                    ),
                )
            }
        }
        EffectMode::ExactTransition => {
            let expected = policy
                .expected_diff_id
                .as_deref()
                .expect("validated exact-transition policy has expected_diff_id");
            if expected == diff.diff_id {
                (true, "observed transition matches preregistered exact diff".to_string())
            } else {
                (
                    false,
                    format!(
                        "observed diff {} does not match preregistered {}",
                        diff.diff_id, expected
                    ),
                )
            }
        }
    };

    EffectEvaluation {
        schema: REPORT_SCHEMA,
        policy_id,
        mode: policy.mode,
        base_snapshot_id: diff.base_snapshot_id.clone(),
        head_snapshot_id: diff.head_snapshot_id.clone(),
        observed_diff_id: diff.diff_id.clone(),
        expected_diff_id: policy.expected_diff_id.clone(),
        allowed,
        reason,
    }
}

fn validate_sha256(name: &str, value: &str) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repository_snapshot_diff::RepositorySourceDiff;

    fn diff(identical: bool) -> RepositorySourceDiff {
        RepositorySourceDiff {
            diff_id: if identical { "1".repeat(64) } else { "2".repeat(64) },
            schema: "symthaea.repository-source-diff.v1",
            base_snapshot_id: "a".repeat(64),
            head_snapshot_id: if identical { "a".repeat(64) } else { "b".repeat(64) },
            identical,
            source_bytes_changed: !identical,
            index_state_changed: false,
            subject_metadata_changed: false,
            changed_subject_fields: vec![],
            entry_changes: vec![],
        }
    }

    #[test]
    fn read_only_requires_exact_subject_equality() {
        let mut policy = EffectPolicySpec {
            schema: POLICY_SCHEMA.into(),
            mode: EffectMode::ReadOnlySource,
            expected_diff_id: None,
        };
        let policy_id = validate_and_identify_policy(&mut policy).unwrap();
        assert!(evaluate(policy_id.clone(), &policy, &diff(true)).allowed);
        assert!(!evaluate(policy_id, &policy, &diff(false)).allowed);
    }

    #[test]
    fn exact_transition_matches_diff_identity() {
        let observed = diff(false);
        let mut policy = EffectPolicySpec {
            schema: POLICY_SCHEMA.into(),
            mode: EffectMode::ExactTransition,
            expected_diff_id: Some(observed.diff_id.to_ascii_uppercase()),
        };
        let policy_id = validate_and_identify_policy(&mut policy).unwrap();
        assert_eq!(policy.expected_diff_id.as_deref(), Some(observed.diff_id.as_str()));
        assert!(evaluate(policy_id, &policy, &observed).allowed);
    }

    #[test]
    fn exact_transition_rejects_different_diff() {
        let observed = diff(false);
        let mut policy = EffectPolicySpec {
            schema: POLICY_SCHEMA.into(),
            mode: EffectMode::ExactTransition,
            expected_diff_id: Some("3".repeat(64)),
        };
        let policy_id = validate_and_identify_policy(&mut policy).unwrap();
        assert!(!evaluate(policy_id, &policy, &observed).allowed);
    }

    #[test]
    fn read_only_rejects_irrelevant_expected_diff_field() {
        let mut policy = EffectPolicySpec {
            schema: POLICY_SCHEMA.into(),
            mode: EffectMode::ReadOnlySource,
            expected_diff_id: Some("1".repeat(64)),
        };
        assert!(validate_and_identify_policy(&mut policy).is_err());
    }
}
