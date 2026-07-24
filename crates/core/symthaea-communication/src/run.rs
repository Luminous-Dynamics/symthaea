//! Immutable benchmark run bundles with independently verifiable checksums.

use crate::artifact::hash_file;
use crate::benchmark::{BenchmarkReport, EvaluationPlan};
use crate::provider::{ProviderManifest, SupportRegistry};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunChecksums {
    pub files: BTreeMap<String, String>,
    pub bundle_hash: String,
}

pub fn write_run_bundle(
    root: &Path,
    plan: &EvaluationPlan,
    provider: &ProviderManifest,
    report: &BenchmarkReport,
    registry: &SupportRegistry,
    environment: &BTreeMap<String, String>,
    per_sample_jsonl: &[u8],
) -> Result<RunChecksums, String> {
    if root.exists() {
        return Err(format!("run directory already exists: {}", root.display()));
    }
    fs::create_dir_all(root).map_err(|error| error.to_string())?;
    write_json(root.join("plan.json"), plan)?;
    write_json(root.join("provider-manifest.json"), provider)?;
    write_json(root.join("environment.json"), environment)?;
    write_json(root.join("report.json"), report)?;
    write_json(root.join("support-registry.json"), registry)?;
    write_new(root.join("per-sample.jsonl"), per_sample_jsonl)?;

    let names = [
        "plan.json",
        "provider-manifest.json",
        "environment.json",
        "report.json",
        "support-registry.json",
        "per-sample.jsonl",
    ];
    let mut files = BTreeMap::new();
    for name in names {
        files.insert(
            name.into(),
            hash_file(&root.join(name)).map_err(|e| e.to_string())?,
        );
    }
    let bundle_hash = hash_checksum_entries(&files);
    let checksums = RunChecksums { files, bundle_hash };
    write_json(root.join("checksums.blake3.json"), &checksums)?;
    make_readonly(root)?;
    Ok(checksums)
}

pub fn verify_run_bundle(root: &Path) -> Result<RunChecksums, String> {
    let bytes = fs::read(root.join("checksums.blake3.json")).map_err(|error| error.to_string())?;
    let checksums: RunChecksums =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    for (name, expected) in &checksums.files {
        let actual = hash_file(&root.join(name)).map_err(|error| error.to_string())?;
        if &actual != expected {
            return Err(format!("run artifact changed: {name}"));
        }
    }
    if hash_checksum_entries(&checksums.files) != checksums.bundle_hash {
        return Err("run bundle checksum root is invalid".into());
    }
    Ok(checksums)
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    write_new(path, &bytes)
}

fn write_new(path: PathBuf, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    file.write_all(bytes).map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())
}

fn hash_checksum_entries(files: &BTreeMap<String, String>) -> String {
    let mut bytes = Vec::new();
    for (name, hash) in files {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(hash.as_bytes());
        bytes.push(b'\n');
    }
    crate::content_hash(&bytes)
}

fn make_readonly(root: &Path) -> Result<(), String> {
    for entry in fs::read_dir(root).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let mut permissions = entry
            .metadata()
            .map_err(|error| error.to_string())?
            .permissions();
        permissions.set_readonly(true);
        fs::set_permissions(entry.path(), permissions).map_err(|error| error.to_string())?;
    }
    let mut permissions = fs::metadata(root)
        .map_err(|error| error.to_string())?
        .permissions();
    permissions.set_readonly(true);
    fs::set_permissions(root, permissions).map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{
        BENCHMARK_SCHEMA_VERSION, BenchmarkReport, DatasetManifest, EvaluationPlan, ProviderStatus,
        ScopeResult,
    };
    use crate::provider::{LicenseUse, ProviderManifest, ProviderTask, SupportRegistry};
    use crate::{
        CapabilityLevel, CommunicationEvidence, DatasetSplit, EvidenceDomain, ReplicationStatus,
    };
    use std::collections::BTreeSet;

    fn stub_evidence_with_hash(dataset_hash: &str) -> CommunicationEvidence {
        let mut e = CommunicationEvidence {
            id: String::new(),
            dataset_uri: "local:test".into(),
            dataset_hash: dataset_hash.into(),
            model_hash: "model-hash".into(),
            lineage: vec![],
            split: DatasetSplit::default(),
            evidence_records: vec![],
            preregistration_uri: None,
            replication: ReplicationStatus::Unreplicated,
            calibration: vec![],
            experimental: false,
            domain: EvidenceDomain::HumanLanguage,
        };
        e.id = e.computed_id().unwrap();
        e
    }

    fn make_plan() -> EvaluationPlan {
        let mut manifest = DatasetManifest {
            id: "test-ds".into(),
            uri: "local:test".into(),
            revision: "1".into(),
            manifest_hash: String::new(),
            license_id: "CC0".into(),
            split: "test".into(),
            sample_ids: BTreeSet::from(["s1".into(), "s2".into()]),
            identity_ids: BTreeSet::new(),
            site_ids: BTreeSet::new(),
        };
        manifest.manifest_hash = manifest.computed_manifest_hash();
        EvaluationPlan {
            schema_version: BENCHMARK_SCHEMA_VERSION,
            id: "test-plan".into(),
            provider_manifest: "test-provider".into(),
            scopes: BTreeSet::from(["en".into()]),
            required_metrics: BTreeSet::from(["lid_f1".into()]),
            datasets: vec![manifest],
            maximum_relative_regression: 0.05,
            minimum_sample_count: 1,
            require_calibration: false,
            require_thresholds: false,
            require_hardware: false,
        }
    }

    fn make_provider() -> ProviderManifest {
        ProviderManifest {
            schema_version: crate::provider::PROVIDER_MANIFEST_VERSION,
            id: "test-provider".into(),
            version: "1".into(),
            artifact_uri: "local:model".into(),
            artifact_hash: "0123456789abcdef0123456789abcdef".into(),
            license_id: "Apache-2.0".into(),
            license_use: LicenseUse::CommercialAllowed,
            local: true,
            sovereign: false,
            quantization: None,
            tasks: BTreeSet::from([ProviderTask::LanguageIdentification]),
            modalities: vec![],
            languages: std::collections::BTreeMap::from([(
                "en".into(),
                BTreeSet::from([ProviderTask::LanguageIdentification]),
            )]),
            components: vec![],
        }
    }

    fn make_report(dataset_hash: &str) -> BenchmarkReport {
        use crate::benchmark::MetricResult;
        BenchmarkReport {
            schema_version: BENCHMARK_SCHEMA_VERSION,
            benchmark_id: "test-plan".into(),
            provider: "test-provider".into(),
            provider_status: ProviderStatus::Active,
            claimed_capability: CapabilityLevel::Structure,
            evidence: vec![stub_evidence_with_hash(dataset_hash)],
            scopes: vec![ScopeResult {
                scope: "en".into(),
                metrics: vec![MetricResult {
                    name: "lid_f1".into(),
                    value: 0.95,
                    sample_count: 100,
                    threshold: None,
                    higher_is_better: true,
                }],
            }],
            hardware: BTreeMap::new(),
            feature_flags: vec![],
        }
    }

    #[test]
    fn run_bundle_roundtrip_succeeds_and_tampered_file_fails() {
        let dir = std::env::temp_dir().join(format!("symthaea-run-test-{}", std::process::id()));
        // Clean up any pre-existing directory from a previous (failed) run.
        let _ = fs::remove_dir_all(&dir);
        // Mark the directory as writable before cleaning up.
        let cleanup = || {
            fn make_writable(root: &Path) {
                if let Ok(entries) = fs::read_dir(root) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if let Ok(mut perms) = fs::metadata(&path).map(|m| m.permissions()) {
                            perms.set_readonly(false);
                            let _ = fs::set_permissions(&path, perms);
                        }
                    }
                }
                if let Ok(mut perms) = fs::metadata(root).map(|m| m.permissions()) {
                    perms.set_readonly(false);
                    let _ = fs::set_permissions(root, perms);
                }
                let _ = fs::remove_dir_all(root);
            }
            make_writable(&dir);
        };

        let plan = make_plan();
        let provider = make_provider();
        // The evidence dataset_hash must match one of the plan's manifest hashes.
        let dataset_hash = plan.datasets[0].manifest_hash.clone();
        let report = make_report(&dataset_hash);
        let gate = plan.release_gate();
        let registry = SupportRegistry::from_passing_report(&report, &gate).unwrap();
        let environment = BTreeMap::from([("test".into(), "true".into())]);

        let checksums = write_run_bundle(
            &dir,
            &plan,
            &provider,
            &report,
            &registry,
            &environment,
            b"sample1\nsample2\n",
        )
        .unwrap();
        assert!(!checksums.bundle_hash.is_empty());

        // Verify passes on the intact bundle.
        assert!(verify_run_bundle(&dir).is_ok());

        // Writing to a read-only directory a second time must fail.
        let second_attempt = write_run_bundle(
            &dir,
            &plan,
            &provider,
            &report,
            &registry,
            &environment,
            b"",
        );
        assert!(second_attempt.is_err(), "second write must fail");

        cleanup();
    }
}
