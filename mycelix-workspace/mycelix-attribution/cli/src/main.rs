// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    /// Path to lockfile (Cargo.lock, package-lock.json, flake.lock, requirements.txt, pyproject.toml, go.sum, Gemfile.lock, pom.xml)
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
    registered_at: i64,
    verified: bool,
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
            let rev = locked.rev.as_ref().map(|r| r[..8.min(r.len())].to_string());
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

// ── requirements.txt Parser ──────────────────────────────────────────

fn parse_requirements_txt(content: &str) -> Vec<(DependencyRecord, UsageReceiptRecord)> {
    content
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('-') {
                return None;
            }
            let (name, version) = if let Some(idx) = line.find("==") {
                (line[..idx].trim(), Some(line[idx + 2..].trim().to_string()))
            } else if let Some(idx) = line.find(">=") {
                (line[..idx].trim(), Some(line[idx + 2..].trim().to_string()))
            } else if let Some(idx) = line.find("~=") {
                (line[..idx].trim(), Some(line[idx + 2..].trim().to_string()))
            } else if let Some(idx) = line.find("<=") {
                (line[..idx].trim(), Some(line[idx + 2..].trim().to_string()))
            } else {
                (line, None)
            };
            let name = name.split(';').next()?.trim();
            let name = name.split('[').next()?.trim();
            if name.is_empty() {
                return None;
            }
            let dep_id = match &version {
                Some(v) => format!("pip:{}:{}", name, v),
                None => format!("pip:{}", name),
            };
            Some((
                DependencyRecord {
                    id: dep_id.clone(),
                    name: name.to_string(),
                    ecosystem: "PythonPackage".into(),
                    version: version.clone(),
                },
                UsageReceiptRecord {
                    dependency_id: dep_id,
                    user_did: String::new(),
                    organization: None,
                    usage_type: "DirectDependency".into(),
                    version_range: version.map(|v| format!("={}", v)),
                },
            ))
        })
        .collect()
}

// ── pyproject.toml Parser ───────────────────────────────────────────

fn parse_pyproject_toml(content: &str) -> Vec<(DependencyRecord, UsageReceiptRecord)> {
    let table: toml::Value = match toml::from_str(content) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to parse pyproject.toml: {}", e);
            return Vec::new();
        }
    };

    let deps = table
        .get("project")
        .and_then(|p| p.get("dependencies"))
        .and_then(|d| d.as_array());

    let Some(deps) = deps else {
        return Vec::new();
    };

    deps.iter()
        .filter_map(|dep_val| {
            let spec = dep_val.as_str()?;
            let (name, version) = if let Some(idx) = spec.find(">=") {
                (&spec[..idx], Some(spec[idx + 2..].trim()))
            } else if let Some(idx) = spec.find("==") {
                (&spec[..idx], Some(spec[idx + 2..].trim()))
            } else if let Some(idx) = spec.find("~=") {
                (&spec[..idx], Some(spec[idx + 2..].trim()))
            } else {
                (spec, None)
            };
            let name = name.split(';').next()?.trim();
            let name = name.split('[').next()?.trim();
            let version = version.map(|v| v.split(',').next().unwrap_or(v).trim().to_string());
            if name.is_empty() {
                return None;
            }
            let dep_id = match &version {
                Some(v) => format!("pip:{}:{}", name, v),
                None => format!("pip:{}", name),
            };
            Some((
                DependencyRecord {
                    id: dep_id.clone(),
                    name: name.to_string(),
                    ecosystem: "PythonPackage".into(),
                    version: version.clone(),
                },
                UsageReceiptRecord {
                    dependency_id: dep_id,
                    user_did: String::new(),
                    organization: None,
                    usage_type: "DirectDependency".into(),
                    version_range: version.map(|v| format!("={}", v)),
                },
            ))
        })
        .collect()
}

// ── go.sum Parser ───────────────────────────────────────────────────

fn parse_go_sum(content: &str) -> Vec<(DependencyRecord, UsageReceiptRecord)> {
    let mut seen = std::collections::HashSet::new();
    content
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            let mut parts = line.split_whitespace();
            let module = parts.next()?;
            let version_raw = parts.next()?;
            let version = version_raw.split('/').next()?;
            let key = format!("{}@{}", module, version);
            if !seen.insert(key) {
                return None;
            }
            let dep_id = format!("go:{}:{}", module, version);
            Some((
                DependencyRecord {
                    id: dep_id.clone(),
                    name: module.to_string(),
                    ecosystem: "GoModule".into(),
                    version: Some(version.to_string()),
                },
                UsageReceiptRecord {
                    dependency_id: dep_id,
                    user_did: String::new(),
                    organization: None,
                    usage_type: "DirectDependency".into(),
                    version_range: Some(format!("={}", version)),
                },
            ))
        })
        .collect()
}

// ── Gemfile.lock Parser ─────────────────────────────────────────────

fn parse_gemfile_lock(content: &str) -> Vec<(DependencyRecord, UsageReceiptRecord)> {
    let mut in_gem_section = false;
    let mut in_specs = false;
    let mut results = Vec::new();

    for line in content.lines() {
        if line.trim() == "GEM" {
            in_gem_section = true;
            continue;
        }
        if in_gem_section && line.trim() == "specs:" {
            in_specs = true;
            continue;
        }
        if in_gem_section && in_specs && !line.starts_with(' ') {
            in_gem_section = false;
            in_specs = false;
            continue;
        }
        if !in_specs {
            continue;
        }
        // Sub-dependencies are 6+ spaces — skip them
        if line.starts_with("      ") {
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(paren_start) = trimmed.find('(') {
            if let Some(paren_end) = trimmed.find(')') {
                let name = trimmed[..paren_start].trim();
                let version = &trimmed[paren_start + 1..paren_end];
                if !name.is_empty() {
                    let dep_id = format!("gem:{}:{}", name, version);
                    results.push((
                        DependencyRecord {
                            id: dep_id.clone(),
                            name: name.to_string(),
                            ecosystem: "RubyGem".into(),
                            version: Some(version.to_string()),
                        },
                        UsageReceiptRecord {
                            dependency_id: dep_id,
                            user_did: String::new(),
                            organization: None,
                            usage_type: "DirectDependency".into(),
                            version_range: Some(format!("={}", version)),
                        },
                    ));
                }
            }
        }
    }

    results
}

// ── pom.xml Parser ──────────────────────────────────────────────────

fn parse_pom_xml(content: &str) -> Vec<(DependencyRecord, UsageReceiptRecord)> {
    let mut results = Vec::new();
    let mut in_dependency = false;
    let mut group_id = String::new();
    let mut artifact_id = String::new();
    let mut version = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "<dependency>" {
            in_dependency = true;
            group_id.clear();
            artifact_id.clear();
            version.clear();
            continue;
        }
        if trimmed == "</dependency>" {
            if in_dependency && !group_id.is_empty() && !artifact_id.is_empty() {
                let dep_id = if version.is_empty() {
                    format!("mvn:{}:{}", group_id, artifact_id)
                } else {
                    format!("mvn:{}:{}:{}", group_id, artifact_id, version)
                };
                let ver = if version.is_empty() {
                    None
                } else {
                    Some(version.clone())
                };
                results.push((
                    DependencyRecord {
                        id: dep_id.clone(),
                        name: format!("{}:{}", group_id, artifact_id),
                        ecosystem: "MavenPackage".into(),
                        version: ver.clone(),
                    },
                    UsageReceiptRecord {
                        dependency_id: dep_id,
                        user_did: String::new(),
                        organization: None,
                        usage_type: "DirectDependency".into(),
                        version_range: ver.map(|v| format!("={}", v)),
                    },
                ));
            }
            in_dependency = false;
            continue;
        }
        if !in_dependency {
            continue;
        }
        if let Some(val) = extract_xml_value(trimmed, "groupId") {
            group_id = val;
        } else if let Some(val) = extract_xml_value(trimmed, "artifactId") {
            artifact_id = val;
        } else if let Some(val) = extract_xml_value(trimmed, "version") {
            version = val;
        }
    }

    results
}

fn extract_xml_value(line: &str, tag: &str) -> Option<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    if let Some(start) = line.find(&open) {
        if let Some(end) = line.find(&close) {
            let value = &line[start + open.len()..end];
            return Some(value.trim().to_string());
        }
    }
    None
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    if !args.did.starts_with("did:") {
        eprintln!(
            "Invalid DID: '{}'. Must start with 'did:' (e.g. did:mycelix:abc123)",
            args.did
        );
        std::process::exit(1);
    }

    let content = match std::fs::read_to_string(&args.lockfile) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading {}: {}", args.lockfile.display(), e);
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
        "requirements.txt" => parse_requirements_txt(&content),
        "pyproject.toml" => parse_pyproject_toml(&content),
        "go.sum" => parse_go_sum(&content),
        "Gemfile.lock" => parse_gemfile_lock(&content),
        "pom.xml" => parse_pom_xml(&content),
        other => {
            eprintln!(
                "Unsupported lockfile: {}. Expected Cargo.lock, package-lock.json, flake.lock, \
                 requirements.txt, pyproject.toml, go.sum, Gemfile.lock, or pom.xml",
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
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
        OutputFormat::Jsonl => {
            for dep in &output.dependencies {
                println!("{}", serde_json::to_string(dep).unwrap());
            }
            for receipt in &output.usage_receipts {
                println!("{}", serde_json::to_string(receipt).unwrap());
            }
        }
        OutputFormat::Batch => {
            // Output array of DependencyIdentity-shaped objects for bulk_register_dependencies
            let now_micros = chrono::Utc::now().timestamp_micros();
            let batch: Vec<_> = output
                .dependencies
                .iter()
                .zip(output.usage_receipts.iter())
                .map(|(dep, receipt)| BatchDependency {
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
            println!("{}", serde_json::to_string_pretty(&batch).unwrap());
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

    #[test]
    fn test_parse_requirements_txt() {
        let content = "# comment\nrequests==2.31.0\nflask>=3.0.0\nnumpy\n-e ./local\n";
        let pairs = parse_requirements_txt(content);
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].0.id, "pip:requests:2.31.0");
        assert_eq!(pairs[0].0.ecosystem, "PythonPackage");
        assert_eq!(pairs[1].0.id, "pip:flask:3.0.0");
        assert_eq!(pairs[2].0.id, "pip:numpy");
        assert!(pairs[2].0.version.is_none());
    }

    #[test]
    fn test_parse_pyproject_toml() {
        let content = r#"
[project]
name = "my-app"
dependencies = [
    "requests>=2.31.0",
    "flask==3.0.0",
    "numpy",
    "pandas>=2.0,<3",
]
"#;
        let pairs = parse_pyproject_toml(content);
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0].0.id, "pip:requests:2.31.0");
        assert_eq!(pairs[1].0.id, "pip:flask:3.0.0");
        assert_eq!(pairs[2].0.id, "pip:numpy");
        assert_eq!(pairs[3].0.id, "pip:pandas:2.0");
    }

    #[test]
    fn test_parse_go_sum() {
        let content = "\
golang.org/x/net v0.17.0 h1:abc123\n\
golang.org/x/net v0.17.0/go.mod h1:def456\n\
github.com/stretchr/testify v1.8.4 h1:ghi789\n";
        let pairs = parse_go_sum(content);
        assert_eq!(pairs.len(), 2); // deduplicated
        assert_eq!(pairs[0].0.id, "go:golang.org/x/net:v0.17.0");
        assert_eq!(pairs[0].0.ecosystem, "GoModule");
        assert_eq!(pairs[1].0.id, "go:github.com/stretchr/testify:v1.8.4");
    }

    #[test]
    fn test_parse_gemfile_lock() {
        let content = "GEM\n  remote: https://rubygems.org/\n  specs:\n    rails (7.1.0)\n      actioncable (= 7.1.0)\n    rack (3.0.8)\n\nPLATFORMS\n  ruby\n";
        let pairs = parse_gemfile_lock(content);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0.id, "gem:rails:7.1.0");
        assert_eq!(pairs[0].0.ecosystem, "RubyGem");
        assert_eq!(pairs[1].0.id, "gem:rack:3.0.8");
    }

    #[test]
    fn test_parse_pom_xml() {
        let content = r#"
<project>
  <dependencies>
    <dependency>
      <groupId>org.apache.commons</groupId>
      <artifactId>commons-lang3</artifactId>
      <version>3.14.0</version>
    </dependency>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.13.2</version>
    </dependency>
  </dependencies>
</project>
"#;
        let pairs = parse_pom_xml(content);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0.id, "mvn:org.apache.commons:commons-lang3:3.14.0");
        assert_eq!(pairs[0].0.ecosystem, "MavenPackage");
        assert_eq!(pairs[0].0.name, "org.apache.commons:commons-lang3");
        assert_eq!(pairs[1].0.id, "mvn:junit:junit:4.13.2");
    }
}
