use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const CANDIDATE_ONLY: &str = "reference_only_not_symthaea_qualified";

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u32,
    status: String,
    policy: Policy,
    entries: Vec<Entry>,
    retrieval_baselines: Vec<RetrievalBaseline>,
}

#[derive(Debug, Deserialize)]
struct Policy {
    upstream_repository: String,
    upstream_commit: String,
    upstream_lean_toolchain: String,
    default_acceptance_status: String,
    no_floating_main_refs: bool,
    do_not_vendor_entire_upstream_library: bool,
    require_existing_approved_symthaea_mathlib_environment: bool,
    accepted_proof_must_reject: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Entry {
    id: String,
    title: String,
    source_path: String,
    source_url: String,
    source_blob_sha: String,
    declarations: Vec<String>,
    relevance: String,
    limitations: Vec<String>,
    status: String,
    license: String,
}

#[derive(Debug, Deserialize)]
struct RetrievalBaseline {
    id: String,
    repository_url: String,
    commit: String,
    artifact_path: String,
    artifact_url: String,
    artifact_blob_sha: String,
    reported_scope: String,
    license: String,
    status: String,
    limitations: Vec<String>,
}

/// Validate the local shape and immutable upstream references of the candidate
/// manifest. This does not fetch upstream source or qualify a theorem in Lean.
pub fn validate(path: &Path) -> Result<()> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read proof corpus manifest {}", path.display()))?;
    let count = validate_manifest_text(&raw)?;
    println!(
        "proof corpus manifest structurally valid: {count} candidate entries; no theorem or verifier qualification is implied"
    );
    Ok(())
}

fn valid_git_sha(sha: &str) -> bool {
    sha.len() == 40
        && sha
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn validate_manifest_text(raw: &str) -> Result<usize> {
    let manifest: Manifest =
        serde_yaml::from_str(raw).context("parse proof corpus YAML manifest")?;

    if manifest.schema_version != 1 {
        bail!("unsupported proof corpus schema_version {}", manifest.schema_version);
    }
    if manifest.status != "candidate_index_only" {
        bail!("manifest status must remain candidate_index_only until qualification exists");
    }

    let policy = &manifest.policy;
    if policy.upstream_repository != "https://github.com/openai/math" {
        bail!("unexpected upstream repository {}", policy.upstream_repository);
    }
    if !valid_git_sha(&policy.upstream_commit) {
        bail!("upstream_commit must be a lowercase 40-hex Git SHA");
    }
    if policy.upstream_lean_toolchain.trim().is_empty() {
        bail!("upstream_lean_toolchain must not be empty");
    }
    if policy.default_acceptance_status != CANDIDATE_ONLY {
        bail!("default_acceptance_status must remain {CANDIDATE_ONLY}");
    }
    if !policy.no_floating_main_refs {
        bail!("no_floating_main_refs must be true");
    }
    if !policy.do_not_vendor_entire_upstream_library {
        bail!("do_not_vendor_entire_upstream_library must be true");
    }
    if !policy.require_existing_approved_symthaea_mathlib_environment {
        bail!("manifest must require Symthaea's approved Mathlib environment");
    }

    for required in [
        "sorry",
        "unexpected_axioms",
        "theorem_statement_mutation",
        "missing_domain_premises",
        "stale_environment_receipt",
        "timeout_or_incomplete_output",
    ] {
        if !policy.accepted_proof_must_reject.iter().any(|item| item == required) {
            bail!("proof policy omits required rejection control: {required}");
        }
    }

    if manifest.entries.is_empty() {
        bail!("manifest must include at least one candidate entry");
    }

    let mut ids = BTreeSet::new();
    let expected_prefix = format!(
        "{}/blob/{}/",
        policy.upstream_repository.trim_end_matches('/'),
        policy.upstream_commit
    );

    for entry in &manifest.entries {
        if entry.id.trim().is_empty() || !ids.insert(entry.id.as_str()) {
            bail!("entry IDs must be non-empty and unique: {:?}", entry.id);
        }
        if entry.title.trim().is_empty() || entry.relevance.trim().is_empty() {
            bail!("entry {} must have a title and relevance statement", entry.id);
        }
        if entry.source_path.is_empty()
            || entry.source_path.starts_with('/')
            || entry.source_path.split('/').any(|part| part == ".." || part.is_empty())
        {
            bail!("entry {} has an unsafe source_path", entry.id);
        }
        if !entry.source_url.starts_with(&expected_prefix)
            || entry.source_url != format!("{expected_prefix}{}", entry.source_path)
        {
            bail!(
                "entry {} source_url must bind the pinned commit and exact source_path",
                entry.id
            );
        }
        if !valid_git_sha(&entry.source_blob_sha) {
            bail!("entry {} source_blob_sha must be a lowercase 40-hex Git SHA", entry.id);
        }
        if entry.declarations.is_empty()
            || entry.declarations.iter().any(|name| name.trim().is_empty())
        {
            bail!("entry {} must name at least one non-empty declaration", entry.id);
        }
        if entry.limitations.is_empty()
            || entry.limitations.iter().any(|limitation| limitation.trim().is_empty())
        {
            bail!("entry {} must state limitations explicitly", entry.id);
        }
        if entry.status != CANDIDATE_ONLY {
            bail!("entry {} must remain candidate-only until independently qualified", entry.id);
        }
        if entry.license.trim().is_empty() {
            bail!("entry {} must retain license/attribution metadata", entry.id);
        }
    }

    if manifest.retrieval_baselines.is_empty() {
        bail!("manifest must name at least one retrieval baseline");
    }

    let mut baseline_ids = BTreeSet::new();
    for baseline in &manifest.retrieval_baselines {
        if baseline.id.trim().is_empty() || !baseline_ids.insert(baseline.id.as_str()) {
            bail!("retrieval baseline IDs must be non-empty and unique: {:?}", baseline.id);
        }
        if !baseline.repository_url.starts_with("https://github.com/") {
            bail!("baseline {} must use a pinned GitHub repository URL", baseline.id);
        }
        if !valid_git_sha(&baseline.commit) {
            bail!("baseline {} commit must be a lowercase 40-hex Git SHA", baseline.id);
        }
        if baseline.artifact_path.is_empty()
            || baseline.artifact_path.starts_with('/')
            || baseline.artifact_path.split('/').any(|part| part == ".." || part.is_empty())
        {
            bail!("baseline {} has an unsafe artifact_path", baseline.id);
        }
        let expected_url = format!(
            "{}/blob/{}/{}",
            baseline.repository_url.trim_end_matches('/'),
            baseline.commit,
            baseline.artifact_path
        );
        if baseline.artifact_url != expected_url {
            bail!(
                "baseline {} artifact_url must bind its pinned commit and exact artifact_path",
                baseline.id
            );
        }
        if !valid_git_sha(&baseline.artifact_blob_sha) {
            bail!("baseline {} artifact_blob_sha must be a lowercase 40-hex Git SHA", baseline.id);
        }
        if baseline.reported_scope.trim().is_empty() || baseline.license.trim().is_empty() {
            bail!("baseline {} must state its scope and license/provenance", baseline.id);
        }
        if baseline.status != "reference_baseline_only_not_a_prover_dependency" {
            bail!("baseline {} must remain reference-only, not a prover dependency", baseline.id);
        }
        if baseline.limitations.is_empty()
            || baseline.limitations.iter().any(|limitation| limitation.trim().is_empty())
        {
            bail!("baseline {} must state limitations explicitly", baseline.id);
        }
    }

    Ok(manifest.entries.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"
schema_version: 1
status: candidate_index_only
policy:
  upstream_repository: "https://github.com/openai/math"
  upstream_commit: "1581d4cbdd69a0faf34b2089614fe2a814316a6a"
  upstream_lean_toolchain: "leanprover/lean4:v4.34.1"
  default_acceptance_status: "reference_only_not_symthaea_qualified"
  no_floating_main_refs: true
  do_not_vendor_entire_upstream_library: true
  require_existing_approved_symthaea_mathlib_environment: true
  accepted_proof_must_reject:
    - sorry
    - unexpected_axioms
    - theorem_statement_mutation
    - missing_domain_premises
    - stale_environment_receipt
    - timeout_or_incomplete_output
retrieval_baselines:
  - id: "LEANSEARCH-V2-MPR-001"
    repository_url: "https://github.com/frenzymath/LeanSearch-v2"
    commit: "94f4888cbaf9f4322535755f86cbac690ec18080"
    artifact_path: "benchmark/MathlibMPR.json"
    artifact_url: "https://github.com/frenzymath/LeanSearch-v2/blob/94f4888cbaf9f4322535755f86cbac690ec18080/benchmark/MathlibMPR.json"
    artifact_blob_sha: "00f1539e5528fa9fb59d253259b4372c58bea09c"
    reported_scope: "69 research-level Mathlib theorems with premise-group ground truth"
    license: "Apache-2.0 repository license; check dataset-specific terms"
    status: "reference_baseline_only_not_a_prover_dependency"
    limitations:
      - "Not production dependency"
      - "Requires compatible benchmark environment"
entries:
  - id: "TEST-001"
    title: "Test theorem"
    source_path: "lean/OAI/Test.lean"
    source_url: "https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/OAI/Test.lean"
    source_blob_sha: "b2b32a01166f7990231a1d16240b1f80269d9114"
    declarations:
      - "OAI.Test.theorem"
    relevance: "Test-only reference"
    limitations:
      - "Not locally qualified"
    status: "reference_only_not_symthaea_qualified"
    license: "Apache-2.0"
"#;

    #[test]
    fn accepts_a_pinned_candidate_only_manifest() {
        assert_eq!(validate_manifest_text(VALID).unwrap(), 1);
    }

    #[test]
    fn rejects_floating_branch_urls() {
        let mutated = VALID.replace(
            "blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/",
            "blob/main/",
        );
        assert!(validate_manifest_text(&mutated).is_err());
    }

    #[test]
    fn rejects_floating_retrieval_baseline_urls() {
        let mutated = VALID.replace(
            "blob/94f4888cbaf9f4322535755f86cbac690ec18080/",
            "blob/main/",
        );
        assert!(validate_manifest_text(&mutated).is_err());
    }

    #[test]
    fn rejects_status_promotion_without_qualification() {
        let mutated = VALID.replace(
            "reference_only_not_symthaea_qualified",
            "formally_verified",
        );
        assert!(validate_manifest_text(&mutated).is_err());
    }

    #[test]
    fn rejects_missing_hostile_controls() {
        let mutated = VALID.replace("    - timeout_or_incomplete_output\n", "");
        assert!(validate_manifest_text(&mutated).is_err());
    }

    #[test]
    fn rejects_duplicate_ids() {
        // Add a second identical entry so validation has a duplicate identity.
        let entry = VALID.split("entries:\n").nth(1).unwrap();
        let mutated = format!("{VALID}{}", entry);
        assert!(validate_manifest_text(&mutated).is_err());
    }
}
