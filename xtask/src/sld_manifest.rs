use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path};

const EXPECTED_FORMAT: &str = "SLD-ARTIFACT-MANIFEST-1";

#[derive(Debug, Deserialize)]
struct ArtifactManifest {
    manifest_format: String,
    hash_algorithm: String,
    source_commit_before_manifest: String,
    self_digest_excluded: bool,
    artifacts: Vec<ArtifactEntry>,
}

#[derive(Debug, Deserialize)]
struct ArtifactEntry {
    path: String,
    sha256: String,
    utf8_bytes: usize,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn is_hex_digest(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Verify a committed SLD artifact manifest against the current working tree.
///
/// The manifest itself is intentionally not hashed by the manifest, avoiding a
/// self-referential digest. This verifies artifact identity, not oracle validity,
/// scientific construct validity, or model behavior.
pub fn verify(root: &Path, manifest_path: &Path) -> Result<usize> {
    let root = root
        .canonicalize()
        .with_context(|| format!("resolve artifact root {}", root.display()))?;
    let manifest_bytes = fs::read(manifest_path)
        .with_context(|| format!("read manifest {}", manifest_path.display()))?;
    let manifest: ArtifactManifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parse manifest {}", manifest_path.display()))?;

    ensure!(
        manifest.manifest_format == EXPECTED_FORMAT,
        "unexpected manifest format: {}",
        manifest.manifest_format
    );
    ensure!(
        manifest.hash_algorithm == "SHA-256",
        "unsupported manifest hash algorithm: {}",
        manifest.hash_algorithm
    );
    ensure!(
        is_hex_digest(&manifest.source_commit_before_manifest, 40),
        "source_commit_before_manifest must be a 40-character hexadecimal commit ID"
    );
    ensure!(
        manifest.self_digest_excluded,
        "manifest must explicitly exclude its own bytes from the digest set"
    );
    ensure!(!manifest.artifacts.is_empty(), "manifest artifact list is empty");

    let mut seen_paths = BTreeSet::new();
    for artifact in &manifest.artifacts {
        ensure!(
            is_hex_digest(&artifact.sha256, 64),
            "{}: malformed SHA-256 digest",
            artifact.path
        );
        ensure!(
            seen_paths.insert(artifact.path.clone()),
            "duplicate artifact path: {}",
            artifact.path
        );

        let relative = Path::new(&artifact.path);
        ensure!(
            !relative.as_os_str().is_empty()
                && relative.components().all(|component| matches!(component, Component::Normal(_))),
            "unsafe artifact path: {}",
            artifact.path
        );

        let candidate = root.join(relative);
        let path = candidate
            .canonicalize()
            .with_context(|| format!("resolve artifact {}", artifact.path))?;
        ensure!(
            path.starts_with(&root),
            "artifact resolves outside root: {}",
            artifact.path
        );

        let bytes = fs::read(&path).with_context(|| format!("read artifact {}", artifact.path))?;
        std::str::from_utf8(&bytes)
            .with_context(|| format!("artifact is not UTF-8 text: {}", artifact.path))?;
        ensure!(
            bytes.len() == artifact.utf8_bytes,
            "{}: byte length mismatch: expected {}, got {}",
            artifact.path,
            artifact.utf8_bytes,
            bytes.len()
        );
        let actual = sha256_hex(&bytes);
        ensure!(
            actual == artifact.sha256,
            "{}: SHA-256 mismatch: expected {}, got {}",
            artifact.path,
            artifact.sha256,
            actual
        );
    }

    Ok(manifest.artifacts.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("sld-manifest-{}-{nonce}", std::process::id()))
    }

    fn write_fixture(root: &Path, artifact_path: &str, contents: &[u8]) -> PathBuf {
        let artifact = root.join(artifact_path);
        fs::create_dir_all(artifact.parent().expect("fixture parent")).expect("create fixture dirs");
        fs::write(&artifact, contents).expect("write fixture artifact");
        let manifest_path = root.join("manifest.json");
        let document = json!({
            "manifest_format": EXPECTED_FORMAT,
            "status": "DRAFT_CANDIDATE",
            "hash_algorithm": "SHA-256",
            "source_commit_before_manifest": "0123456789abcdef0123456789abcdef01234567",
            "self_digest_excluded": true,
            "artifacts": [{
                "path": artifact_path,
                "sha256": sha256_hex(contents),
                "utf8_bytes": contents.len(),
                "git_blob_sha1": "ignored-by-this-verifier"
            }]
        });
        fs::write(&manifest_path, serde_json::to_vec_pretty(&document).expect("serialize fixture"))
            .expect("write fixture manifest");
        manifest_path
    }

    #[test]
    fn verifies_exact_artifact_and_fails_after_tampering() {
        let root = temporary_root();
        fs::create_dir_all(&root).expect("create temporary root");
        let path = "docs/research/example.json";
        let manifest_path = write_fixture(&root, path, b"{\n  \"candidate\": true\n}\n");
        assert_eq!(verify(&root, &manifest_path).expect("valid manifest"), 1);

        fs::write(root.join(path), b"{\n  \"candidate\": false\n}\n").expect("tamper fixture");
        assert!(verify(&root, &manifest_path).is_err(), "tampered content must fail closed");
        fs::remove_dir_all(&root).expect("remove temporary root");
    }

    #[test]
    fn rejects_parent_traversal_paths() {
        let root = temporary_root();
        fs::create_dir_all(&root).expect("create temporary root");
        let manifest_path = root.join("manifest.json");
        let bytes = b"{}\n";
        let document = json!({
            "manifest_format": EXPECTED_FORMAT,
            "status": "DRAFT_CANDIDATE",
            "hash_algorithm": "SHA-256",
            "source_commit_before_manifest": "0123456789abcdef0123456789abcdef01234567",
            "self_digest_excluded": true,
            "artifacts": [{
                "path": "../outside.json",
                "sha256": sha256_hex(bytes),
                "utf8_bytes": bytes.len()
            }]
        });
        fs::write(&manifest_path, serde_json::to_vec_pretty(&document).expect("serialize manifest"))
            .expect("write manifest");
        assert!(verify(&root, &manifest_path).is_err(), "parent traversal must be rejected");
        fs::remove_dir_all(&root).expect("remove temporary root");
    }
}
