//! mycelix-attribution-scan — Scans lockfiles and generates attribution data.
//!
//! Reads Cargo.lock, package-lock.json, or flake.lock and outputs
//! DependencyIdentity + UsageReceipt JSON payloads for the Mycelix DHT.
//!
//! With the `submit` feature, can connect directly to a Holochain conductor
//! and submit dependencies + usage receipts via WebSocket.

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[cfg(feature = "submit")]
mod submit;

// ── CLI Arguments ────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "mycelix-attribution-scan")]
#[command(about = "Scan lockfiles and generate attribution data for the Mycelix DHT")]
struct Args {
    /// Path to lockfile (Cargo.lock, package-lock.json, or flake.lock)
    #[arg(short, long)]
    lockfile: PathBuf,

    /// Your DID (e.g. did:mycelix:abc123)
    #[arg(short, long)]
    did: String,

    /// Organization name (optional)
    #[arg(short, long)]
    organization: Option<String>,

    /// Output format: json (array), jsonl (one per line), or batch (bulk_register payload)
    #[arg(short, long, default_value = "jsonl")]
    format: OutputFormat,

    /// Submit directly to a running Holochain conductor (requires `submit` feature)
    /// Example: --submit ws://localhost:8888
    #[arg(long)]
    submit: Option<String>,

    /// hApp ID for conductor submission (default: "attribution")
    #[arg(long, default_value = "attribution")]
    app_id: String,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum OutputFormat {
    Json,
    Jsonl,
    /// Outputs a JSON array of DependencyIdentity objects for bulk_register_dependencies
    Batch,
}

// ── Output Types ─────────────────────────────────────────────────────

#[derive(Serialize, Debug)]
struct ScanOutput {
    dependencies: Vec<DependencyRecord>,
    usage_receipts: Vec<UsageReceiptRecord>,
}

#[derive(Serialize, Debug)]
struct DependencyRecord {
    id: String,
    name: String,
    ecosystem: String,
    version: Option<String>,
}

#[derive(Serialize, Debug)]
struct UsageReceiptRecord {
    dependency_id: String,
    user_did: String,
    organization: Option<String>,
    usage_type: String,
    version_range: Option<String>,
}

// ── Batch Output Type ───────────────────────────────────────────────

/// Matches the DependencyIdentity fields expected by bulk_register_dependencies.
/// Timestamps and booleans are omitted — the zome sets those on create.
#[derive(Serialize, Debug)]
struct BatchDependency {
    id: String,
    name: String,
    ecosystem: String,
    maintainer_did: String,
    repository_url: Option<String>,
    license: Option<String>,
    description: String,
    version: Option<String>,
}

// ── Cargo.lock Parser ────────────────────────────────────────────────

#[derive(Deserialize)]
struct CargoLock {
    package: Vec<CargoPackage>,
}

#[derive(Deserialize)]
struct CargoPackage {
    name: String,
    version: String,
    source: Option<String>,
}

fn parse_cargo_lock(content: &str) -> Vec<(DependencyRecord, UsageReceiptRecord)> {
    let lock: CargoLock = match toml::from_str(content) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to parse Cargo.lock: {}", e);
            return Vec::new();
        }
    };

    lock.package
        .into_iter()
        .filter(|p| {
            // Only include crates from crates.io (has source)
            p.source
                .as_ref()
                .is_some_and(|s| s.contains("crates.io") || s.starts_with("registry+"))
        })
        .map(|p| {
            let dep_id = format!("crate:{}:{}", p.name, p.version);
            let dep = DependencyRecord {
                id: dep_id.clone(),
                name: p.name,
                ecosystem: "RustCrate".into(),
                version: Some(p.version.clone()),
            };
            let receipt = UsageReceiptRecord {
                dependency_id: dep_id,
                user_did: String::new(), // filled by caller
                organization: None,
                usage_type: "DirectDependency".into(),
                version_range: Some(format!("={}", p.version)),
            };
            (dep, receipt)
        })
        .collect()
}

// ── package-lock.json Parser ─────────────────────────────────────────

#[derive(Deserialize)]
struct PackageLock {
    packages: Option<BTreeMap<String, NpmPackageEntry>>,
}

#[derive(Deserialize)]
struct NpmPackageEntry {
    version: Option<String>,
    resolved: Option<String>,
}

fn parse_package_lock(content: &str) -> Vec<(DependencyRecord, UsageReceiptRecord)> {
    let lock: PackageLock = match serde_json::from_str(content) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to parse package-lock.json: {}", e);
            return Vec::new();
        }
    };

    let packages = match lock.packages {
        Some(p) => p,
        None => return Vec::new(),
    };

    packages
        .into_iter()
        .filter(|(key, entry)| {
            // Skip root package (empty key) and packages without versions
            !key.is_empty()
                && entry.version.is_some()
                && entry
                    .resolved
                    .as_ref()
                    .is_some_and(|r| r.contains("registry.npmjs.org"))
        })
        .map(|(key, entry)| {
            let name = key
                .strip_prefix("node_modules/")
                .unwrap_or(&key)
                .to_string();
            let version = entry.version.unwrap_or_default();
            let dep_id = format!("npm:{}:{}", name, version);
            let dep = DependencyRecord {
                id: dep_id.clone(),
                name,
                ecosystem: "NpmPackage".into(),
                version: Some(version.clone()),
            };
            let receipt = UsageReceiptRecord {
                dependency_id: dep_id,
                user_did: String::new(),
                organization: None,
                usage_type: "DirectDependency".into(),
                version_range: Some(format!("={}", version)),
            };
            (dep, receipt)
        })
        .collect()
}

// ── flake.lock Parser ────────────────────────────────────────────────

#[derive(Deserialize)]
struct FlakeLock {
    nodes: BTreeMap<String, FlakeNode>,
}

#[derive(Deserialize)]
struct FlakeNode {
    locked: Option<FlakeLocked>,
    #[allow(dead_code)]
    original: Option<FlakeOriginal>,
}

#[derive(Deserialize)]
struct FlakeLocked {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    lock_type: Option<String>,
    owner: Option<String>,
    repo: Option<String>,
    rev: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct FlakeOriginal {
    #[serde(rename = "type")]
    orig_type: Option<String>,
    owner: Option<String>,
    repo: Option<String>,
    #[serde(rename = "ref")]
    git_ref: Option<String>,
}

fn parse_flake_lock(content: &str) -> Vec<(DependencyRecord, UsageReceiptRecord)> {
    let lock: FlakeLock = match serde_json::from_str(content) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to parse flake.lock: {}", e);
            return Vec::new();
        }
    };

    lock.nodes
        .into_iter()
        .filter(|(key, node)| {
            key != "root"
                && node
                    .locked
                    .as_ref()
                    .is_some_and(|l| l.owner.is_some() && l.repo.is_some())
        })
        .map(|(key, node)| {
            let locked = node.locked.unwrap();
            let owner = locked.owner.unwrap_or_default();
            let repo = locked.repo.unwrap_or_default();
            let rev = locked
                .rev
                .as_ref()
                .map(|r| r[..8.min(r.len())].to_string());
            let dep_id = format!("nix:{}/{}", owner, repo);
            let dep = DependencyRecord {
                id: dep_id.clone(),
                name: key,
                ecosystem: "NixFlake".into(),
                version: rev.clone(),
            };
            let receipt = UsageReceiptRecord {
                dependency_id: dep_id,
                user_did: String::new(),
                organization: None,
                usage_type: "DirectDependency".into(),
                version_range: rev,
            };
            (dep, receipt)
        })
        .collect()
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let content = match std::fs::read_to_string(&args.lockfile) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "Error reading {}: {}",
                args.lockfile.display(),
                e
            );
            std::process::exit(1);
        }
    };

    let filename = args
        .lockfile
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("");

    let pairs = match filename {
        "Cargo.lock" => parse_cargo_lock(&content),
        "package-lock.json" => parse_package_lock(&content),
        "flake.lock" => parse_flake_lock(&content),
        other => {
            eprintln!(
                "Unsupported lockfile: {}. Expected Cargo.lock, package-lock.json, or flake.lock",
                other
            );
            std::process::exit(1);
        }
    };

    let mut output = ScanOutput {
        dependencies: Vec::new(),
        usage_receipts: Vec::new(),
    };

    for (dep, mut receipt) in pairs {
        receipt.user_did = args.did.clone();
        receipt.organization = args.organization.clone();
        output.dependencies.push(dep);
        output.usage_receipts.push(receipt);
    }

    eprintln!(
        "Scanned {}: {} dependencies found",
        filename,
        output.dependencies.len()
    );

    // Submit to conductor if --submit is provided
    #[cfg(feature = "submit")]
    if let Some(ref ws_url) = args.submit {
        let now_micros = chrono::Utc::now().timestamp_micros();

        let submit_deps: Vec<submit::SubmitDependency> = output
            .dependencies
            .iter()
            .zip(output.usage_receipts.iter())
            .map(|(dep, receipt)| submit::SubmitDependency {
                id: dep.id.clone(),
                name: dep.name.clone(),
                ecosystem: dep.ecosystem.clone(),
                maintainer_did: receipt.user_did.clone(),
                repository_url: None,
                license: None,
                description: format!(
                    "{} {} ({})",
                    dep.ecosystem,
                    dep.name,
                    dep.version.as_deref().unwrap_or("*")
                ),
                version: dep.version.clone(),
                registered_at: now_micros,
                verified: false,
            })
            .collect();

        let submit_receipts: Vec<submit::SubmitUsageReceipt> = output
            .usage_receipts
            .iter()
            .zip(output.dependencies.iter())
            .map(|(receipt, dep)| submit::SubmitUsageReceipt {
                id: format!("scan:{}:{}", dep.id, now_micros),
                dependency_id: dep.id.clone(),
                user_did: receipt.user_did.clone(),
                organization: receipt.organization.clone(),
                usage_type: "DirectDependency".to_string(),
                scale: None,
                version_range: receipt.version_range.clone(),
                context: Some(format!("Scanned from {}", filename)),
                attested_at: now_micros,
            })
            .collect();

        let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        if let Err(e) = rt.block_on(submit::submit_to_conductor(
            ws_url,
            &args.app_id,
            submit_deps,
            submit_receipts,
        )) {
            eprintln!("Submit failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    #[cfg(not(feature = "submit"))]
    if args.submit.is_some() {
        eprintln!(
            "Error: --submit requires the `submit` feature. Rebuild with:\n  \
             cargo build --release --features submit"
        );
        std::process::exit(1);
    }

    match args.format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&output).unwrap()
            );
        }
        OutputFormat::Jsonl => {
            for dep in &output.dependencies {
                println!(
                    "{}",
                    serde_json::to_string(dep).unwrap()
                );
            }
            for receipt in &output.usage_receipts {
                println!(
                    "{}",
                    serde_json::to_string(receipt).unwrap()
                );
            }
        }
        OutputFormat::Batch => {
            // Output array of DependencyIdentity-shaped objects for bulk_register_dependencies
            let batch: Vec<_> = output
                .dependencies
                .iter()
                .zip(output.usage_receipts.iter())
                .map(|(dep, receipt)| {
                    BatchDependency {
                        id: dep.id.clone(),
                        name: dep.name.clone(),
                        ecosystem: dep.ecosystem.clone(),
                        maintainer_did: receipt.user_did.clone(),
                        repository_url: None,
                        license: None,
                        description: format!(
                            "{} {} ({})",
                            dep.ecosystem, dep.name, dep.version.as_deref().unwrap_or("*")
                        ),
                        version: dep.version.clone(),
                    }
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&batch).unwrap()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cargo_lock_sample() {
        let content = r#"
[[package]]
name = "serde"
version = "1.0.219"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "abc123"

[[package]]
name = "my-local-crate"
version = "0.1.0"
"#;
        let pairs = parse_cargo_lock(content);
        assert_eq!(pairs.len(), 1); // only crates.io, not local
        assert_eq!(pairs[0].0.id, "crate:serde:1.0.219");
        assert_eq!(pairs[0].0.ecosystem, "RustCrate");
    }

    #[test]
    fn test_parse_package_lock_sample() {
        let content = r#"{
  "packages": {
    "": { "name": "my-app", "version": "1.0.0" },
    "node_modules/react": {
      "version": "18.2.0",
      "resolved": "https://registry.npmjs.org/react/-/react-18.2.0.tgz"
    },
    "node_modules/local-pkg": {
      "version": "0.1.0"
    }
  }
}"#;
        let pairs = parse_package_lock(content);
        assert_eq!(pairs.len(), 1); // only npm registry, not root or local
        assert_eq!(pairs[0].0.id, "npm:react:18.2.0");
    }

    #[test]
    fn test_parse_flake_lock_sample() {
        let content = r#"{
  "nodes": {
    "root": { "inputs": { "nixpkgs": "nixpkgs" } },
    "nixpkgs": {
      "locked": {
        "type": "github",
        "owner": "NixOS",
        "repo": "nixpkgs",
        "rev": "abc12345deadbeef"
      },
      "original": {
        "type": "github",
        "owner": "NixOS",
        "repo": "nixpkgs",
        "ref": "nixos-unstable"
      }
    }
  }
}"#;
        let pairs = parse_flake_lock(content);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0.id, "nix:NixOS/nixpkgs");
        assert_eq!(pairs[0].0.ecosystem, "NixFlake");
    }

    #[test]
    fn test_empty_cargo_lock() {
        let content = "[[package]]\nname = \"root\"\nversion = \"0.0.0\"\n";
        let pairs = parse_cargo_lock(content);
        assert!(pairs.is_empty());
    }
}
