use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

use crate::cargo_adapter_semantics::RepositoryAccess;
use crate::repository_snapshot_receipt::ValidatedSnapshotReceipt;

const INPUT_SCHEMA: &str = "symthaea.repository-materialization-input.v1";
const RECEIPT_SCHEMA: &str = "symthaea.repository-materialization.v1";
const HASH_DOMAIN: &[u8] = b"symthaea.repository-materialization.v1\0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RepositoryMaterializationSpec {
    pub schema: String,
    pub runtime_instance_id: String,
    pub kind: RepositoryAccess,
    pub materialization_id: String,
    pub working_directory_subject_id: String,
    pub repository_setup_receipt_sha256: String,
    pub derivation_receipt_sha256: String,
    #[serde(default)]
    pub staged_content_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RepositoryMaterializationReceipt {
    pub materialization_receipt_id: String,
    pub schema: String,
    pub source_snapshot_id: String,
    pub runtime_instance_id: String,
    pub kind: RepositoryAccess,
    pub materialization_id: String,
    pub working_directory_subject_id: String,
    pub repository_setup_receipt_sha256: String,
    pub derivation_receipt_sha256: String,
    pub staged_content_id: Option<String>,
}

#[derive(Serialize)]
struct MaterializationIdentity<'a> {
    schema: &'static str,
    source_snapshot_id: &'a str,
    runtime_instance_id: &'a str,
    kind: &'a RepositoryAccess,
    materialization_id: &'a str,
    working_directory_subject_id: &'a str,
    repository_setup_receipt_sha256: &'a str,
    derivation_receipt_sha256: &'a str,
    staged_content_id: &'a Option<String>,
}

pub(crate) fn run(
    spec_path: &Path,
    pre_snapshot_path: &Path,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let spec_bytes = fs::read(spec_path)
        .with_context(|| format!("read repository materialization spec {}", spec_path.display()))?;
    let spec: RepositoryMaterializationSpec = serde_json::from_slice(&spec_bytes)
        .with_context(|| format!("parse repository materialization spec {}", spec_path.display()))?;
    let source = ValidatedSnapshotReceipt::load(pre_snapshot_path)?;
    let receipt = build_receipt(spec, source)?;
    write_json(&receipt, output)
}

pub(crate) fn run_verify(receipt_path: &Path, pre_snapshot_path: &Path) -> anyhow::Result<()> {
    let bytes = fs::read(receipt_path)
        .with_context(|| format!("read repository materialization receipt {}", receipt_path.display()))?;
    let stored: RepositoryMaterializationReceipt = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse repository materialization receipt {}", receipt_path.display()))?;
    let source = ValidatedSnapshotReceipt::load(pre_snapshot_path)?;
    verify_stored(stored, source)?;
    Ok(())
}

pub(crate) fn build_receipt(
    mut spec: RepositoryMaterializationSpec,
    mut source: ValidatedSnapshotReceipt,
) -> anyhow::Result<RepositoryMaterializationReceipt> {
    if spec.schema != INPUT_SCHEMA {
        bail!(
            "unsupported repository materialization input schema: {}",
            spec.schema
        );
    }
    source.validate()?;
    normalize_spec(&mut spec)?;
    validate_shape(&spec)?;

    let identity = MaterializationIdentity {
        schema: RECEIPT_SCHEMA,
        source_snapshot_id: &source.snapshot_id,
        runtime_instance_id: &spec.runtime_instance_id,
        kind: &spec.kind,
        materialization_id: &spec.materialization_id,
        working_directory_subject_id: &spec.working_directory_subject_id,
        repository_setup_receipt_sha256: &spec.repository_setup_receipt_sha256,
        derivation_receipt_sha256: &spec.derivation_receipt_sha256,
        staged_content_id: &spec.staged_content_id,
    };
    let canonical = serde_json::to_vec(&identity)
        .context("serialize repository materialization identity")?;
    let materialization_receipt_id = domain_sha256(HASH_DOMAIN, &canonical);

    Ok(RepositoryMaterializationReceipt {
        materialization_receipt_id,
        schema: RECEIPT_SCHEMA.into(),
        source_snapshot_id: source.snapshot_id,
        runtime_instance_id: spec.runtime_instance_id,
        kind: spec.kind,
        materialization_id: spec.materialization_id,
        working_directory_subject_id: spec.working_directory_subject_id,
        repository_setup_receipt_sha256: spec.repository_setup_receipt_sha256,
        derivation_receipt_sha256: spec.derivation_receipt_sha256,
        staged_content_id: spec.staged_content_id,
    })
}

pub(crate) fn verify_stored(
    stored: RepositoryMaterializationReceipt,
    source: ValidatedSnapshotReceipt,
) -> anyhow::Result<RepositoryMaterializationReceipt> {
    if stored.schema != RECEIPT_SCHEMA {
        bail!(
            "unsupported repository materialization receipt schema: {}",
            stored.schema
        );
    }
    validate_digest(
        "materialization_receipt_id",
        &stored.materialization_receipt_id,
    )?;

    let spec = RepositoryMaterializationSpec {
        schema: INPUT_SCHEMA.into(),
        runtime_instance_id: stored.runtime_instance_id.clone(),
        kind: stored.kind.clone(),
        materialization_id: stored.materialization_id.clone(),
        working_directory_subject_id: stored.working_directory_subject_id.clone(),
        repository_setup_receipt_sha256: stored.repository_setup_receipt_sha256.clone(),
        derivation_receipt_sha256: stored.derivation_receipt_sha256.clone(),
        staged_content_id: stored.staged_content_id.clone(),
    };
    let rebuilt = build_receipt(spec, source)?;
    if rebuilt != stored {
        bail!(
            "stored repository materialization receipt is not the canonical rebuild for the validated source subject"
        );
    }
    Ok(rebuilt)
}

fn normalize_spec(spec: &mut RepositoryMaterializationSpec) -> anyhow::Result<()> {
    for (name, value) in [
        ("runtime_instance_id", &mut spec.runtime_instance_id),
        ("materialization_id", &mut spec.materialization_id),
        (
            "working_directory_subject_id",
            &mut spec.working_directory_subject_id,
        ),
        (
            "repository_setup_receipt_sha256",
            &mut spec.repository_setup_receipt_sha256,
        ),
        (
            "derivation_receipt_sha256",
            &mut spec.derivation_receipt_sha256,
        ),
    ] {
        normalize_digest(name, value)?;
    }
    if let Some(staged) = &mut spec.staged_content_id {
        normalize_digest("staged_content_id", staged)?;
    }
    Ok(())
}

fn validate_shape(spec: &RepositoryMaterializationSpec) -> anyhow::Result<()> {
    if spec.working_directory_subject_id != spec.materialization_id {
        bail!(
            "repository materialization v1 requires working_directory_subject_id to equal materialization_id"
        );
    }
    if spec.runtime_instance_id == spec.materialization_id {
        bail!("runtime instance identity must be distinct from repository materialization identity");
    }

    match &spec.kind {
        RepositoryAccess::ReadOnlySource => {
            if spec.staged_content_id.is_some() {
                bail!("read_only_source materialization must not provide staged_content_id");
            }
        }
        RepositoryAccess::IsolatedStagingTree => {
            let staged = spec
                .staged_content_id
                .as_ref()
                .context("isolated_staging_tree materialization requires staged_content_id")?;
            if staged == &spec.materialization_id || staged == &spec.runtime_instance_id {
                bail!(
                    "staged content identity must be distinct from instance/materialization identities"
                );
            }
        }
    }
    Ok(())
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    validate_digest(name, value)?;
    value.make_ascii_lowercase();
    Ok(())
}

fn validate_digest(name: &str, value: &str) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
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

fn write_json(
    receipt: &RepositoryMaterializationReceipt,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let mut rendered = serde_json::to_string_pretty(receipt)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create materialization output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write repository materialization receipt {}", path.display()))?;
    } else {
        print!("{rendered}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repository_snapshot_receipt::{
        EntryKind, SnapshotEntry, SnapshotScope, SourceClass, UnknownSurface,
    };

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn source(content: char) -> ValidatedSnapshotReceipt {
        let mut source = ValidatedSnapshotReceipt {
            snapshot_id: String::new(),
            schema: "symthaea.repository-source-snapshot.v1".into(),
            git_head: "a".repeat(40),
            git_head_tree: "b".repeat(40),
            git_version: "git version 2.50.0".into(),
            scope: SnapshotScope {
                tracked_worktree: true,
                untracked_non_ignored: true,
                explicit_ignored_inputs: vec![],
                ignored_policy: "git_exclude_standard_plus_explicit_ignored_inputs".into(),
                symlink_policy: "hash_link_target_bytes_do_not_follow_referent".into(),
                submodule_policy: "record_gitlink_index_identity_mark_contents_unknown".into(),
                external_input_policy: "not_captured".into(),
            },
            entries: vec![SnapshotEntry {
                path: "src/lib.rs".into(),
                source_class: SourceClass::Tracked,
                kind: EntryKind::File,
                content_sha256: Some(digest(content)),
                size_bytes: Some(12),
                executable: Some(false),
                index_mode: Some("100644".into()),
                index_blob: Some("d".repeat(40)),
            }],
            unknown_surfaces: vec![UnknownSurface {
                kind: "environment_network_and_external_build_inputs_not_captured".into(),
                subject: None,
            }],
        };
        source.snapshot_id = source.computed_snapshot_id().unwrap();
        source.validate().unwrap();
        source
    }

    fn spec() -> RepositoryMaterializationSpec {
        RepositoryMaterializationSpec {
            schema: INPUT_SCHEMA.into(),
            runtime_instance_id: digest('1'),
            kind: RepositoryAccess::ReadOnlySource,
            materialization_id: digest('2'),
            working_directory_subject_id: digest('2'),
            repository_setup_receipt_sha256: digest('3'),
            derivation_receipt_sha256: digest('4'),
            staged_content_id: None,
        }
    }

    #[test]
    fn source_subject_is_derived_from_validated_snapshot() {
        let receipt = build_receipt(spec(), source('c')).unwrap();
        assert_eq!(receipt.source_snapshot_id, source('c').snapshot_id);
    }

    #[test]
    fn source_change_changes_materialization_receipt_identity() {
        let a = build_receipt(spec(), source('c')).unwrap();
        let b = build_receipt(spec(), source('e')).unwrap();
        assert_ne!(a.source_snapshot_id, b.source_snapshot_id);
        assert_ne!(a.materialization_receipt_id, b.materialization_receipt_id);
    }

    #[test]
    fn input_schema_cannot_mint_source_snapshot_id() {
        let json = format!(
            "{{\"schema\":\"{INPUT_SCHEMA}\",\"runtime_instance_id\":\"{}\",\"kind\":\"read_only_source\",\"materialization_id\":\"{}\",\"working_directory_subject_id\":\"{}\",\"repository_setup_receipt_sha256\":\"{}\",\"derivation_receipt_sha256\":\"{}\",\"source_snapshot_id\":\"{}\"}}",
            digest('1'),
            digest('2'),
            digest('2'),
            digest('3'),
            digest('4'),
            digest('9')
        );
        assert!(serde_json::from_str::<RepositoryMaterializationSpec>(&json).is_err());
    }

    #[test]
    fn staging_tree_requires_separate_staged_content_identity() {
        let mut missing = spec();
        missing.kind = RepositoryAccess::IsolatedStagingTree;
        assert!(build_receipt(missing, source('c')).is_err());

        let mut valid = spec();
        valid.kind = RepositoryAccess::IsolatedStagingTree;
        valid.staged_content_id = Some(digest('5'));
        assert!(build_receipt(valid, source('c')).is_ok());
    }

    #[test]
    fn read_only_source_rejects_staged_content_identity() {
        let mut invalid = spec();
        invalid.staged_content_id = Some(digest('5'));
        assert!(build_receipt(invalid, source('c')).is_err());
    }

    #[test]
    fn runtime_instance_is_identity_significant() {
        let a = build_receipt(spec(), source('c')).unwrap();
        let mut other = spec();
        other.runtime_instance_id = digest('6');
        let b = build_receipt(other, source('c')).unwrap();
        assert_ne!(a.materialization_receipt_id, b.materialization_receipt_id);
    }

    #[test]
    fn stored_receipt_cannot_be_reused_with_another_source() {
        let stored = build_receipt(spec(), source('c')).unwrap();
        assert!(verify_stored(stored, source('e')).is_err());
    }
}
