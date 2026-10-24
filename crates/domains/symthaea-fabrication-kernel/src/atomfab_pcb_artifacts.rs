// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hash actual files under a declared root instead of trusting package-manifest digests.
//!
//! This is an artifact-inventory contract only. It does not parse KiCad files or establish
//! DRC cleanliness, manufacturability, physical conformity, electrical safety, or release authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
};

pub const ARTIFACT_INVENTORY_SCHEMA: &str = "atomfab-pcb-artifact-inventory-v0.1";
pub const ARTIFACT_INVENTORY_AUTHORITY: &str =
    "artifact_presence_and_digest_inventory_only_no_design_or_production_authority";
pub const ARTIFACT_INVENTORY_CLAIM_CEILING: &str = "Artifact presence and SHA-256 inventory only; no KiCad DRC, design manufacturability, physical quality, safety, or production-release authority.";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactInventoryManifest {
    pub schema: String,
    pub design_id: String,
    pub design_revision: String,
    pub required_roles: Vec<String>,
    pub artifacts: Vec<ArtifactSpec>,
    pub authority: String,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSpec {
    pub role: String,
    pub path: String,
    #[serde(default)]
    pub expected_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ArtifactInventoryReceipt {
    pub schema: String,
    pub design_id: String,
    pub design_revision: String,
    pub manifest_sha256: String,
    pub status: String,
    pub required_role_count: usize,
    pub observed_artifact_count: usize,
    pub artifacts: Vec<ArtifactObservation>,
    pub missing_required_roles: Vec<String>,
    pub problems: Vec<String>,
    pub authority: String,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ArtifactObservation {
    pub role: String,
    pub path: String,
    pub status: String,
    pub byte_len: Option<u64>,
    pub sha256: Option<String>,
    pub expected_sha256_match: Option<bool>,
}

/// Evaluate a strict manifest against files that actually exist below `root`.
/// Digests are always computed from bytes read during this invocation.
pub fn evaluate_artifact_inventory(
    manifest_bytes: &[u8],
    root: impl AsRef<Path>,
) -> Result<ArtifactInventoryReceipt, String> {
    let manifest: ArtifactInventoryManifest = serde_json::from_slice(manifest_bytes)
        .map_err(|error| format!("invalid artifact inventory manifest: {error}"))?;

    if manifest.schema != ARTIFACT_INVENTORY_SCHEMA
        || manifest.design_id.trim().is_empty()
        || manifest.design_revision.trim().is_empty()
    {
        return Err("invalid artifact inventory schema, design ID, or revision".into());
    }
    if manifest.authority != ARTIFACT_INVENTORY_AUTHORITY
        || manifest.claim_ceiling != ARTIFACT_INVENTORY_CLAIM_CEILING
    {
        return Err("artifact inventory authority boundary weakened or missing".into());
    }
    if manifest.required_roles.is_empty()
        || manifest.required_roles.iter().any(|role| !valid_role(role))
    {
        return Err("required artifact roles must be nonempty, unique tokens".into());
    }

    let mut required_roles = BTreeSet::new();
    for role in &manifest.required_roles {
        if !required_roles.insert(role.clone()) {
            return Err(format!("duplicate required artifact role: {role}"));
        }
    }

    let canonical_root = fs::canonicalize(root.as_ref())
        .map_err(|error| format!("cannot canonicalize artifact root: {error}"))?;
    if !canonical_root.is_dir() {
        return Err("artifact root is not a directory".into());
    }

    let mut problems = Vec::new();
    let mut declared_roles = BTreeSet::new();
    let mut declared_paths = BTreeSet::new();
    let mut verified_roles = BTreeSet::new();
    let mut resolved_paths = BTreeSet::new();
    let mut observations = Vec::with_capacity(manifest.artifacts.len());

    for artifact in &manifest.artifacts {
        let mut observation = ArtifactObservation {
            role: artifact.role.clone(),
            path: artifact.path.clone(),
            status: "rejected".to_string(),
            byte_len: None,
            sha256: None,
            expected_sha256_match: None,
        };

        if !valid_role(&artifact.role) {
            problems.push(format!("INVALID_ROLE:{}", artifact.role));
            observations.push(observation);
            continue;
        }
        if !required_roles.contains(&artifact.role) {
            problems.push(format!("UNDECLARED_ROLE:{}", artifact.role));
        }
        if !declared_roles.insert(artifact.role.clone()) {
            problems.push(format!("DUPLICATE_ROLE:{}", artifact.role));
        }

        if !declared_paths.insert(artifact.path.clone()) {
            problems.push(format!("DUPLICATE_PATH:{}", artifact.path));
            observations.push(observation);
            continue;
        }

        if let Some(expected) = artifact.expected_sha256.as_deref() {
            if !is_sha256_hex(expected) {
                problems.push(format!("INVALID_EXPECTED_SHA256:{}", artifact.path));
                observation.expected_sha256_match = Some(false);
            }
        }

        let relative_path = match safe_relative_path(&artifact.path) {
            Ok(path) => path,
            Err(reason) => {
                problems.push(format!("{reason}:{}", artifact.path));
                observations.push(observation);
                continue;
            }
        };
        let candidate = canonical_root.join(relative_path);
        let canonical_file = match fs::canonicalize(&candidate) {
            Ok(path) => path,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                observation.status = "missing".to_string();
                problems.push(format!("MISSING_ARTIFACT:{}", artifact.path));
                observations.push(observation);
                continue;
            }
            Err(_) => {
                observation.status = "unreadable".to_string();
                problems.push(format!("ARTIFACT_UNREADABLE:{}", artifact.path));
                observations.push(observation);
                continue;
            }
        };

        if !canonical_file.starts_with(&canonical_root) {
            problems.push(format!("ARTIFACT_ESCAPES_ROOT:{}", artifact.path));
            observations.push(observation);
            continue;
        }
        if !canonical_file.is_file() {
            observation.status = "unreadable".to_string();
            problems.push(format!("ARTIFACT_NOT_REGULAR_FILE:{}", artifact.path));
            observations.push(observation);
            continue;
        }
        if !resolved_paths.insert(canonical_file.clone()) {
            problems.push(format!("DUPLICATE_RESOLVED_FILE:{}", artifact.path));
            observations.push(observation);
            continue;
        }

        let bytes = match fs::read(&canonical_file) {
            Ok(bytes) => bytes,
            Err(_) => {
                observation.status = "unreadable".to_string();
                problems.push(format!("ARTIFACT_UNREADABLE:{}", artifact.path));
                observations.push(observation);
                continue;
            }
        };
        let digest = sha256_hex(&bytes);
        observation.byte_len = Some(bytes.len() as u64);
        observation.sha256 = Some(digest.clone());

        if let Some(expected) = artifact.expected_sha256.as_deref() {
            let matches = is_sha256_hex(expected) && digest.eq_ignore_ascii_case(expected);
            observation.expected_sha256_match = Some(matches);
            if !matches {
                problems.push(format!("EXPECTED_SHA256_MISMATCH:{}", artifact.path));
            }
        }

        if !required_roles.contains(&artifact.role) {
            observation.status = "rejected".to_string();
        } else if observation.expected_sha256_match == Some(false) {
            observation.status = "rejected".to_string();
        } else {
            observation.status = "observed".to_string();
            verified_roles.insert(artifact.role.clone());
        }
        observations.push(observation);
    }

    let mut missing_required_roles: Vec<String> = required_roles
        .difference(&verified_roles)
        .cloned()
        .collect();
    missing_required_roles.sort();
    observations.sort_by(|left, right| {
        left.role
            .cmp(&right.role)
            .then_with(|| left.path.cmp(&right.path))
    });
    problems.sort();

    let status = if !problems.is_empty() {
        "rejected"
    } else if !missing_required_roles.is_empty() {
        "incomplete"
    } else {
        "complete"
    };

    Ok(ArtifactInventoryReceipt {
        schema: "atomfab-pcb-artifact-inventory-receipt-v0.1".to_string(),
        design_id: manifest.design_id,
        design_revision: manifest.design_revision,
        manifest_sha256: sha256_hex(manifest_bytes),
        status: status.to_string(),
        required_role_count: required_roles.len(),
        observed_artifact_count: observations
            .iter()
            .filter(|artifact| artifact.status == "observed")
            .count(),
        artifacts: observations,
        missing_required_roles,
        problems,
        authority: ARTIFACT_INVENTORY_AUTHORITY.to_string(),
        claim_ceiling: ARTIFACT_INVENTORY_CLAIM_CEILING.to_string(),
    })
}

fn valid_role(role: &str) -> bool {
    !role.is_empty()
        && role
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn safe_relative_path(raw: &str) -> Result<PathBuf, &'static str> {
    if raw.is_empty() || raw.contains('\\') || raw.contains(':') {
        return Err("UNSAFE_ARTIFACT_PATH");
    }
    let path = Path::new(raw);
    if path.is_absolute() {
        return Err("UNSAFE_ARTIFACT_PATH");
    }
    let mut has_normal_component = false;
    for component in path.components() {
        match component {
            Component::Normal(_) => has_normal_component = true,
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err("UNSAFE_ARTIFACT_PATH");
            }
        }
    }
    if !has_normal_component {
        return Err("UNSAFE_ARTIFACT_PATH");
    }
    Ok(path.to_path_buf())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_TEMP_ROOT: AtomicU64 = AtomicU64::new(1);

    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "atomfab-pcb-artifacts-{}-{}",
                std::process::id(),
                NEXT_TEMP_ROOT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).expect("create temporary artifact root");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn manifest(artifacts: serde_json::Value, required_roles: &[&str]) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema": ARTIFACT_INVENTORY_SCHEMA,
            "design_id": "ATOMFAB-SMOKE-BOARD",
            "design_revision": "r0",
            "required_roles": required_roles,
            "artifacts": artifacts,
            "authority": ARTIFACT_INVENTORY_AUTHORITY,
            "claim_ceiling": ARTIFACT_INVENTORY_CLAIM_CEILING
        }))
        .expect("serialize fixture manifest")
    }

    #[test]
    fn hashes_actual_bytes_and_completes_when_required_roles_exist() {
        let root = TempRoot::new();
        fs::write(root.path().join("board.fixture"), b"board bytes").expect("write board");
        fs::write(root.path().join("top.gbr.fixture"), b"gerber bytes").expect("write gerber");
        let input = manifest(
            serde_json::json!([
                {"role":"native_board","path":"board.fixture"},
                {"role":"gerber_top_copper","path":"top.gbr.fixture"}
            ]),
            &["native_board", "gerber_top_copper"],
        );

        let receipt =
            evaluate_artifact_inventory(&input, root.path()).expect("inventory evaluation");
        assert_eq!(receipt.status, "complete");
        assert_eq!(receipt.required_role_count, 2);
        assert_eq!(receipt.observed_artifact_count, 2);
        assert_eq!(
            receipt.artifacts[0].sha256.clone().unwrap(),
            sha256_hex(b"gerber bytes")
        );
        assert_eq!(
            receipt.artifacts[1].sha256.clone().unwrap(),
            sha256_hex(b"board bytes")
        );
        assert!(receipt.problems.is_empty());
    }

    #[test]
    fn absent_required_role_is_incomplete_and_never_silently_omitted() {
        let root = TempRoot::new();
        fs::write(root.path().join("board.fixture"), b"board bytes").expect("write board");
        let input = manifest(
            serde_json::json!([{"role":"native_board","path":"board.fixture"}]),
            &["native_board", "drc_report"],
        );

        let receipt =
            evaluate_artifact_inventory(&input, root.path()).expect("inventory evaluation");
        assert_eq!(receipt.status, "incomplete");
        assert_eq!(receipt.missing_required_roles, vec!["drc_report"]);
    }

    #[test]
    fn path_traversal_is_rejected_without_reading_outside_root() {
        let root = TempRoot::new();
        let outside = root
            .path()
            .parent()
            .expect("parent")
            .join("atomfab-outside-fixture");
        fs::write(&outside, b"must not be read").expect("write outside fixture");
        let input = manifest(
            serde_json::json!([{"role":"native_board","path":"../atomfab-outside-fixture"}]),
            &["native_board"],
        );

        let receipt =
            evaluate_artifact_inventory(&input, root.path()).expect("inventory evaluation");
        assert_eq!(receipt.status, "rejected");
        assert!(
            receipt
                .problems
                .iter()
                .any(|value| value.starts_with("UNSAFE_ARTIFACT_PATH:"))
        );
        assert_eq!(receipt.artifacts[0].sha256, None);
        let _ = fs::remove_file(outside);
    }

    #[test]
    fn expected_digest_mismatch_rejects_actual_file_bytes() {
        let root = TempRoot::new();
        fs::write(root.path().join("board.fixture"), b"actual bytes").expect("write board");
        let input = manifest(
            serde_json::json!([{
                "role":"native_board",
                "path":"board.fixture",
                "expected_sha256":"0000000000000000000000000000000000000000000000000000000000000000"
            }]),
            &["native_board"],
        );

        let receipt =
            evaluate_artifact_inventory(&input, root.path()).expect("inventory evaluation");
        assert_eq!(receipt.status, "rejected");
        assert_eq!(receipt.observed_artifact_count, 0);
        assert!(
            receipt
                .problems
                .iter()
                .any(|value| value.starts_with("EXPECTED_SHA256_MISMATCH:"))
        );
    }

    #[test]
    fn duplicate_roles_and_undeclared_roles_reject() {
        let root = TempRoot::new();
        fs::write(root.path().join("one.fixture"), b"one").expect("write first");
        fs::write(root.path().join("two.fixture"), b"two").expect("write second");
        let input = manifest(
            serde_json::json!([
                {"role":"native_board","path":"one.fixture"},
                {"role":"native_board","path":"two.fixture"},
                {"role":"untracked_extra","path":"two.fixture"}
            ]),
            &["native_board"],
        );

        let receipt =
            evaluate_artifact_inventory(&input, root.path()).expect("inventory evaluation");
        assert_eq!(receipt.status, "rejected");
        assert!(
            receipt
                .problems
                .iter()
                .any(|value| value.starts_with("DUPLICATE_ROLE:"))
        );
        assert!(
            receipt
                .problems
                .iter()
                .any(|value| value.starts_with("UNDECLARED_ROLE:"))
        );
    }

    #[test]
    fn repeated_inventory_is_byte_stable_and_receipt_is_sorted() {
        let root = TempRoot::new();
        fs::write(root.path().join("z.fixture"), b"z").expect("write z");
        fs::write(root.path().join("a.fixture"), b"a").expect("write a");
        let input = manifest(
            serde_json::json!([
                {"role":"gerber_top_copper","path":"z.fixture"},
                {"role":"native_board","path":"a.fixture"}
            ]),
            &["native_board", "gerber_top_copper"],
        );

        let first = evaluate_artifact_inventory(&input, root.path()).expect("first receipt");
        let second = evaluate_artifact_inventory(&input, root.path()).expect("second receipt");
        assert_eq!(first, second);
        assert_eq!(first.artifacts[0].role, "gerber_top_copper");
        assert_eq!(first.artifacts[1].role, "native_board");
    }

    #[test]
    fn weakened_authority_and_malformed_manifest_are_rejected() {
        let root = TempRoot::new();
        let mut value: serde_json::Value =
            serde_json::from_slice(&manifest(serde_json::json!([]), &["native_board"]))
                .expect("fixture JSON");
        value["authority"] = serde_json::Value::String("production_ready".to_string());
        let weakened = serde_json::to_vec(&value).expect("serialize weakened manifest");
        assert!(evaluate_artifact_inventory(&weakened, root.path()).is_err());
        assert!(evaluate_artifact_inventory(b"{not-json", root.path()).is_err());
    }
}
