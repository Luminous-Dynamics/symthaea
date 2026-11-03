use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    env,
    fs,
    io::{self, Read},
    path::{Component, Path, PathBuf},
    process::ExitCode,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    benchmark: Benchmark,
    dataset: Dataset,
    task: Task,
    toolchains: Toolchains,
    artifacts: Artifacts,
    classification: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Benchmark {
    name: String,
    release: String,
    cohort_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    artifact_uri: String,
    license: String,
    archive_sha256: String,
    program_pair_count: u64,
    candidate_specification_count: u64,
    evaluation_task_count: Option<u64>,
    extraction_manifest_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Task {
    task_id: String,
    program_pair_id: String,
    specification_id: String,
    stratum: String,
    categories: Vec<String>,
    split: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Toolchains {
    rustc: String,
    aeneas: String,
    lean: String,
    lake_manifest_sha256: String,
    translator_config_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifacts {
    source_rust: Artifact,
    translated_lean: Artifact,
    statement: Artifact,
    lake_manifest: Option<Artifact>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: String,
    sha256: String,
}

fn fail(message: impl AsRef<str>) -> ExitCode {
    eprintln!("INVALID: {}", message.as_ref());
    ExitCode::FAILURE
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

// Resolve only normal relative paths. Reject symlinks in every component so an
// artifact cannot escape the supplied root through a symlink.
fn resolve_artifact(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let rel = Path::new(relative);
    if rel.as_os_str().is_empty() || rel.is_absolute() {
        return Err(format!("artifact path must be non-empty and relative: {relative}"));
    }

    let mut current = root.to_path_buf();
    for component in rel.components() {
        match component {
            Component::Normal(part) => current.push(part),
            _ => return Err(format!("artifact path contains traversal or prefix: {relative}")),
        }
        let metadata = fs::symlink_metadata(&current)
            .map_err(|e| format!("cannot inspect artifact {relative}: {e}"))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("symlink artifact paths are forbidden: {relative}"));
        }
    }
    if !current.is_file() {
        return Err(format!("artifact is not a regular file: {relative}"));
    }
    Ok(current)
}

fn verify_artifact(root: &Path, artifact: &Artifact, label: &str) -> Result<(), String> {
    if !valid_sha256(&artifact.sha256) {
        return Err(format!("{label}: malformed SHA-256 digest"));
    }
    let path = resolve_artifact(root, &artifact.path)?;
    let actual = sha256_file(&path).map_err(|e| format!("{label}: read failed: {e}"))?;
    if actual != artifact.sha256 {
        return Err(format!(
            "{label}: digest mismatch (expected {}, got {})",
            artifact.sha256, actual
        ));
    }
    println!("VERIFIED {label}: {} sha256={actual}", artifact.path);
    Ok(())
}

fn validate_metadata(m: &Manifest) -> Result<(), String> {
    if m.schema_version != "pitb-aegis-manifest/0.1" {
        return Err("unsupported schema_version".into());
    }
    if m.benchmark.name != "Aegis" || m.benchmark.release.trim().is_empty()
        || m.benchmark.cohort_id.trim().is_empty()
    {
        return Err("invalid benchmark identity".into());
    }
    if m.dataset.artifact_uri.trim().is_empty() || m.dataset.license.trim().is_empty()
        || m.dataset.program_pair_count == 0 || m.dataset.candidate_specification_count == 0
        || m.dataset.evaluation_task_count == Some(0)
    {
        return Err("invalid dataset provenance or corpus counts".into());
    }
    if !valid_sha256(&m.dataset.archive_sha256) {
        return Err("malformed dataset archive SHA-256".into());
    }
    if let Some(digest) = &m.dataset.extraction_manifest_sha256 {
        if !valid_sha256(digest) {
            return Err("malformed extraction manifest SHA-256".into());
        }
    }
    if m.task.task_id.trim().is_empty() || m.task.program_pair_id.trim().is_empty()
        || m.task.specification_id.trim().is_empty() || m.task.split.trim().is_empty()
    {
        return Err("task identity fields must be non-empty".into());
    }
    if !matches!(m.task.stratum.as_str(), "edge_case" | "explicit_contract" | "other" | "unknown") {
        return Err("unknown task stratum".into());
    }
    let allowed = ["fixed_width_arithmetic", "overflow", "explicit_failure", "loops",
        "casts", "mutation", "mixed", "other"];
    let mut seen = HashSet::new();
    for category in &m.task.categories {
        if !allowed.contains(&category.as_str()) || !seen.insert(category) {
            return Err(format!("unknown or duplicate task category: {category}"));
        }
    }
    if m.toolchains.rustc.trim().is_empty() || m.toolchains.aeneas.trim().is_empty()
        || m.toolchains.lean.trim().is_empty()
        || !valid_sha256(&m.toolchains.lake_manifest_sha256)
    {
        return Err("invalid or incomplete toolchain identity".into());
    }
    if let Some(digest) = &m.toolchains.translator_config_sha256 {
        if !valid_sha256(digest) {
            return Err("malformed translator config SHA-256".into());
        }
    }
    if !matches!(m.classification.as_str(), "proved" | "refuted" | "unresolved"
        | "invalid_task" | "translation_disagreement" | "specification_disagreement"
        | "execution_error")
    {
        return Err("unknown terminal classification".into());
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err(format!("usage: {} <manifest.json> <artifact-root>", args.first().map(String::as_str).unwrap_or("pitb-aegis-manifest")));
    }
    let manifest_path = Path::new(&args[1]);
    let root = fs::canonicalize(&args[2]).map_err(|e| format!("cannot resolve artifact root: {e}"))?;
    if !root.is_dir() {
        return Err("artifact root is not a directory".into());
    }
    let bytes = fs::read(manifest_path).map_err(|e| format!("cannot read manifest: {e}"))?;
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| format!("manifest JSON/schema decoding failed: {e}"))?;
    validate_metadata(&manifest)?;

    let mut paths = HashSet::new();
    let mut check = |artifact: &Artifact, label: &str| -> Result<(), String> {
        if !paths.insert(artifact.path.clone()) {
            return Err(format!("duplicate artifact path: {}", artifact.path));
        }
        verify_artifact(&root, artifact, label)
    };
    check(&manifest.artifacts.source_rust, "source_rust")?;
    check(&manifest.artifacts.translated_lean, "translated_lean")?;
    check(&manifest.artifacts.statement, "statement")?;
    if let Some(lake) = &manifest.artifacts.lake_manifest {
        check(lake, "lake_manifest")?;
        if lake.sha256 != manifest.toolchains.lake_manifest_sha256 {
            return Err("Lake manifest artifact digest differs from toolchain identity".into());
        }
    }
    println!(
        "MANIFEST_INTEGRITY_OK task={} program_pair={} specification={} classification={}",
        manifest.task.task_id, manifest.task.program_pair_id,
        manifest.task.specification_id, manifest.classification
    );
    println!("NOTE: digest integrity does not establish Rust/Lean semantic correspondence or proof validity.");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => fail(message),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, time::{SystemTime, UNIX_EPOCH}};

    #[test]
    fn accepts_lowercase_sha256_shape() {
        assert!(valid_sha256(&"a".repeat(64)));
        assert!(valid_sha256(&"0123456789abcdef".repeat(4)));
    }

    #[test]
    fn rejects_bad_sha256_shape() {
        assert!(!valid_sha256("abc"));
        assert!(!valid_sha256(&"A".repeat(64)));
        assert!(!valid_sha256(&"g".repeat(64)));
    }

    #[test]
    fn rejects_parent_and_absolute_paths() {
        let root = Path::new("/tmp");
        assert!(resolve_artifact(root, "../outside").is_err());
        assert!(resolve_artifact(root, "/etc/passwd").is_err());
        assert!(resolve_artifact(root, "").is_err());
    }

    #[test]
    fn verifies_regular_file_and_rejects_symlink() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = env::temp_dir().join(format!("pitb-aegis-test-{}-{nonce}", std::process::id()));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("source.rs"), b"fn main() {}").unwrap();
        assert!(resolve_artifact(&root, "source.rs").is_ok());

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.join("source.rs"), root.join("linked.rs")).unwrap();
            assert!(resolve_artifact(&root, "linked.rs").is_err());
        }

        fs::remove_dir_all(&root).unwrap();
    }
}
