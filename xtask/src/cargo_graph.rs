use anyhow::{Context, bail};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SCHEMA: &str = "symthaea.cargo-graph-snapshot.v1";
const HASH_DOMAIN: &[u8] = b"symthaea.cargo-graph-snapshot.v1\0";

#[derive(Debug, Clone, Serialize)]
pub struct CargoGraphSnapshot {
    pub snapshot_id: String,
    #[serde(flatten)]
    pub payload: CargoGraphPayload,
}

#[derive(Debug, Clone, Serialize)]
pub struct CargoGraphPayload {
    pub schema: &'static str,
    pub lock_sha256: String,
    pub workspace_manifest_sha256: String,
    pub workspace_members: Vec<String>,
    pub packages: Vec<PackageSnapshot>,
    pub resolve_nodes: Vec<ResolveNodeSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PackageSnapshot {
    pub id: String,
    pub origin: PackageOrigin,
    pub name: String,
    pub version: String,
    pub source: Option<String>,
    pub checksum: Option<String>,
    pub links: Option<String>,
    pub edition: Option<String>,
    pub rust_version: Option<String>,
    pub workspace_manifest_path: Option<String>,
    pub local_manifest_sha256: Option<String>,
    pub features: BTreeMap<String, Vec<String>>,
    pub targets: Vec<TargetSnapshot>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PackageOrigin {
    Workspace,
    LocalPath,
    External,
}

#[derive(Debug, Clone, Serialize)]
pub struct TargetSnapshot {
    pub name: String,
    pub kind: Vec<String>,
    pub crate_types: Vec<String>,
    pub required_features: Vec<String>,
    pub edition: Option<String>,
    pub doc: Option<bool>,
    pub doctest: Option<bool>,
    pub test: Option<bool>,
    pub workspace_src_path: Option<String>,
    pub local_src_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolveNodeSnapshot {
    pub package: String,
    pub features: Vec<String>,
    pub deps: Vec<ResolvedDependencySnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedDependencySnapshot {
    pub name: String,
    pub package: String,
    pub dep_kinds: Vec<DependencyKindSnapshot>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct DependencyKindSnapshot {
    pub kind: String,
    pub target: Option<String>,
}

pub fn run(root: &Path, output: Option<PathBuf>) -> anyhow::Result<()> {
    let root = root
        .canonicalize()
        .with_context(|| format!("canonicalize workspace root {}", root.display()))?;
    let snapshot = build_snapshot(&root)?;
    let mut rendered = serde_json::to_string_pretty(&snapshot)?;
    rendered.push('\n');

    if let Some(path) = output {
        let path = if path.is_absolute() {
            path
        } else {
            root.join(path)
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write cargo graph snapshot {}", path.display()))?;
        println!("Cargo graph snapshot written to {}", path.display());
    } else {
        print!("{rendered}");
    }

    Ok(())
}

pub fn build_snapshot(root: &Path) -> anyhow::Result<CargoGraphSnapshot> {
    let metadata = cargo_metadata_locked(root)?;
    let payload = payload_from_metadata(root, &metadata)?;
    let snapshot_id = payload_id(&payload)?;
    Ok(CargoGraphSnapshot {
        snapshot_id,
        payload,
    })
}

fn cargo_metadata_locked(root: &Path) -> anyhow::Result<Value> {
    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--format-version", "1"])
        .current_dir(root)
        .output()
        .context("execute cargo metadata --locked --format-version 1")?;

    if !output.status.success() {
        bail!(
            "cargo metadata --locked failed (status={}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    serde_json::from_slice(&output.stdout).context("parse cargo metadata JSON")
}

fn payload_from_metadata(root: &Path, metadata: &Value) -> anyhow::Result<CargoGraphPayload> {
    let packages = metadata
        .get("packages")
        .and_then(Value::as_array)
        .context("cargo metadata missing packages array")?;
    let workspace_members_raw: BTreeSet<String> = metadata
        .get("workspace_members")
        .and_then(Value::as_array)
        .context("cargo metadata missing workspace_members array")?
        .iter()
        .map(required_string)
        .collect::<anyhow::Result<_>>()?;

    let mut raw_to_stable = HashMap::with_capacity(packages.len());
    let mut package_snapshots = Vec::with_capacity(packages.len());

    for package in packages {
        let raw_id = required_field_string(package, "id")?;
        let name = required_field_string(package, "name")?;
        let version = required_field_string(package, "version")?;
        let source = optional_field_string(package, "source");
        let checksum = optional_field_string(package, "checksum");
        let links = optional_field_string(package, "links");
        let is_workspace = workspace_members_raw.contains(&raw_id);
        let manifest_path_raw = required_field_string(package, "manifest_path")?;
        let manifest_path = Path::new(&manifest_path_raw);
        let manifest_sha256 = sha256_file(manifest_path)?;
        let relative_manifest_path = relative_workspace_path(root, manifest_path).ok();

        let workspace_manifest_path = if is_workspace {
            Some(relative_manifest_path.clone().with_context(|| {
                format!(
                    "workspace package manifest {} is outside workspace root {}",
                    manifest_path.display(),
                    root.display()
                )
            })?)
        } else {
            None
        };

        let origin = if is_workspace {
            PackageOrigin::Workspace
        } else if source.is_none() {
            PackageOrigin::LocalPath
        } else {
            PackageOrigin::External
        };

        let stable_id = match origin {
            PackageOrigin::Workspace => format!(
                "workspace:{}#{name}@{version}",
                workspace_manifest_path
                    .as_deref()
                    .expect("workspace package has relative manifest path")
            ),
            PackageOrigin::LocalPath => match relative_manifest_path.as_deref() {
                Some(path) => format!("local-path:{path}#{name}@{version}#{manifest_sha256}"),
                None => format!("local-path-external:{manifest_sha256}#{name}@{version}"),
            },
            PackageOrigin::External => format!(
                "external:{}#{name}@{version}#{}",
                source.as_deref().expect("external package has source"),
                checksum.as_deref().unwrap_or("checksum:unavailable")
            ),
        };

        let features = canonical_features(package.get("features"))?;
        let targets = canonical_targets(root, package.get("targets"), source.is_none())?;

        raw_to_stable.insert(raw_id, stable_id.clone());
        package_snapshots.push(PackageSnapshot {
            id: stable_id,
            origin,
            name,
            version,
            source,
            checksum,
            links,
            edition: optional_field_string(package, "edition"),
            rust_version: optional_field_string(package, "rust_version"),
            workspace_manifest_path,
            local_manifest_sha256: (origin != PackageOrigin::External).then_some(manifest_sha256),
            features,
            targets,
        });
    }

    package_snapshots.sort_by(|a, b| a.id.cmp(&b.id));

    let mut workspace_members: Vec<String> = workspace_members_raw
        .iter()
        .map(|raw| {
            raw_to_stable
                .get(raw)
                .cloned()
                .with_context(|| format!("workspace member missing from packages: {raw}"))
        })
        .collect::<anyhow::Result<_>>()?;
    workspace_members.sort();

    let mut resolve_nodes = Vec::new();
    let resolve = metadata
        .get("resolve")
        .and_then(Value::as_object)
        .context("cargo metadata missing resolve object")?;
    let nodes = resolve
        .get("nodes")
        .and_then(Value::as_array)
        .context("cargo metadata resolve missing nodes array")?;

    for node in nodes {
        let raw_id = required_field_string(node, "id")?;
        let package = raw_to_stable
            .get(&raw_id)
            .cloned()
            .with_context(|| format!("resolved node package missing from packages: {raw_id}"))?;
        let mut features = string_array(node.get("features"))?;
        canonicalize_strings(&mut features);

        let mut deps = Vec::new();
        for dep in node
            .get("deps")
            .and_then(Value::as_array)
            .context("resolved node missing deps array")?
        {
            let name = required_field_string(dep, "name")?;
            let raw_pkg = required_field_string(dep, "pkg")?;
            let resolved_package = raw_to_stable.get(&raw_pkg).cloned().with_context(|| {
                format!("resolved dependency package missing from packages: {raw_pkg}")
            })?;
            let mut dep_kinds = Vec::new();
            for dep_kind in dep
                .get("dep_kinds")
                .and_then(Value::as_array)
                .context("resolved dependency missing dep_kinds array")?
            {
                dep_kinds.push(DependencyKindSnapshot {
                    kind: dep_kind
                        .get("kind")
                        .and_then(Value::as_str)
                        .unwrap_or("normal")
                        .to_string(),
                    target: dep_kind
                        .get("target")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                });
            }
            dep_kinds.sort();
            dep_kinds.dedup();
            deps.push(ResolvedDependencySnapshot {
                name,
                package: resolved_package,
                dep_kinds,
            });
        }
        deps.sort_by(|a, b| {
            (&a.name, &a.package, &a.dep_kinds).cmp(&(&b.name, &b.package, &b.dep_kinds))
        });

        resolve_nodes.push(ResolveNodeSnapshot {
            package,
            features,
            deps,
        });
    }
    resolve_nodes.sort_by(|a, b| a.package.cmp(&b.package));

    Ok(CargoGraphPayload {
        schema: SCHEMA,
        lock_sha256: sha256_file(&root.join("Cargo.lock"))?,
        workspace_manifest_sha256: sha256_file(&root.join("Cargo.toml"))?,
        workspace_members,
        packages: package_snapshots,
        resolve_nodes,
    })
}

fn canonical_features(value: Option<&Value>) -> anyhow::Result<BTreeMap<String, Vec<String>>> {
    let object = value
        .and_then(Value::as_object)
        .context("package features is not an object")?;
    let mut out = BTreeMap::new();
    for (name, members) in object {
        let mut values = string_array(Some(members))?;
        canonicalize_strings(&mut values);
        out.insert(name.clone(), values);
    }
    Ok(out)
}

fn canonical_targets(
    root: &Path,
    value: Option<&Value>,
    is_local: bool,
) -> anyhow::Result<Vec<TargetSnapshot>> {
    let targets = value
        .and_then(Value::as_array)
        .context("package targets is not an array")?;
    let mut out = Vec::with_capacity(targets.len());

    for target in targets {
        let mut kind = string_array(target.get("kind"))?;
        let mut crate_types = string_array(target.get("crate_types"))?;
        let mut required_features = optional_string_array(target.get("required-features"))?;
        canonicalize_strings(&mut kind);
        canonicalize_strings(&mut crate_types);
        canonicalize_strings(&mut required_features);

        let src_path = target.get("src_path").and_then(Value::as_str);
        let workspace_src_path = if is_local {
            src_path.and_then(|path| relative_workspace_path(root, Path::new(path)).ok())
        } else {
            None
        };
        let local_src_sha256 = if is_local {
            src_path.map(|path| sha256_file(Path::new(path))).transpose()?
        } else {
            None
        };

        out.push(TargetSnapshot {
            name: required_field_string(target, "name")?,
            kind,
            crate_types,
            required_features,
            edition: optional_field_string(target, "edition"),
            doc: optional_field_bool(target, "doc"),
            doctest: optional_field_bool(target, "doctest"),
            test: optional_field_bool(target, "test"),
            workspace_src_path,
            local_src_sha256,
        });
    }

    out.sort_by(|a, b| {
        (
            &a.name,
            &a.kind,
            &a.crate_types,
            &a.required_features,
            &a.edition,
            &a.doc,
            &a.doctest,
            &a.test,
            &a.workspace_src_path,
            &a.local_src_sha256,
        )
            .cmp(&(
                &b.name,
                &b.kind,
                &b.crate_types,
                &b.required_features,
                &b.edition,
                &b.doc,
                &b.doctest,
                &b.test,
                &b.workspace_src_path,
                &b.local_src_sha256,
            ))
    });
    Ok(out)
}

fn payload_id(payload: &CargoGraphPayload) -> anyhow::Result<String> {
    let canonical = serde_json::to_vec(payload).context("serialize canonical cargo graph payload")?;
    let mut hasher = Sha256::new();
    hasher.update(HASH_DOMAIN);
    hasher.update(canonical);
    Ok(hex_lower(&hasher.finalize()))
}

fn relative_workspace_path(root: &Path, path: &Path) -> anyhow::Result<String> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("canonicalize workspace path {}", path.display()))?;
    let relative = canonical.strip_prefix(root).with_context(|| {
        format!(
            "workspace path {} escapes canonical root {}",
            canonical.display(),
            root.display()
        )
    })?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let digest = Sha256::digest(bytes);
    Ok(hex_lower(&digest))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing into String cannot fail");
    }
    out
}

fn required_string(value: &Value) -> anyhow::Result<String> {
    value
        .as_str()
        .map(ToString::to_string)
        .context("expected JSON string")
}

fn required_field_string(value: &Value, field: &str) -> anyhow::Result<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .with_context(|| format!("missing string field {field}"))
}

fn optional_field_string(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn optional_field_bool(value: &Value, field: &str) -> Option<bool> {
    value.get(field).and_then(Value::as_bool)
}

fn string_array(value: Option<&Value>) -> anyhow::Result<Vec<String>> {
    let values = value
        .and_then(Value::as_array)
        .context("expected JSON array")?;
    values.iter().map(required_string).collect()
}

fn optional_string_array(value: Option<&Value>) -> anyhow::Result<Vec<String>> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(value) => string_array(Some(value)),
    }
}

fn canonicalize_strings(values: &mut Vec<String>) {
    values.sort();
    values.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_payload() -> CargoGraphPayload {
        CargoGraphPayload {
            schema: SCHEMA,
            lock_sha256: "aa".repeat(32),
            workspace_manifest_sha256: "bb".repeat(32),
            workspace_members: vec!["workspace:Cargo.toml#root@0.1.0".into()],
            packages: vec![PackageSnapshot {
                id: "workspace:Cargo.toml#root@0.1.0".into(),
                origin: PackageOrigin::Workspace,
                name: "root".into(),
                version: "0.1.0".into(),
                source: None,
                checksum: None,
                links: None,
                edition: Some("2024".into()),
                rust_version: None,
                workspace_manifest_path: Some("Cargo.toml".into()),
                local_manifest_sha256: Some("cc".repeat(32)),
                features: BTreeMap::from([("default".into(), vec!["feature-a".into()])]),
                targets: vec![TargetSnapshot {
                    name: "root".into(),
                    kind: vec!["lib".into()],
                    crate_types: vec!["lib".into()],
                    required_features: vec![],
                    edition: Some("2024".into()),
                    doc: Some(true),
                    doctest: Some(true),
                    test: Some(true),
                    workspace_src_path: Some("src/lib.rs".into()),
                    local_src_sha256: Some("dd".repeat(32)),
                }],
            }],
            resolve_nodes: vec![ResolveNodeSnapshot {
                package: "workspace:Cargo.toml#root@0.1.0".into(),
                features: vec!["default".into(), "feature-a".into()],
                deps: vec![],
            }],
        }
    }

    #[test]
    fn snapshot_id_is_deterministic() {
        let payload = fixture_payload();
        assert_eq!(payload_id(&payload).unwrap(), payload_id(&payload).unwrap());
    }

    #[test]
    fn feature_surface_changes_snapshot_id() {
        let a = fixture_payload();
        let mut b = a.clone();
        b.resolve_nodes[0].features.push("feature-b".into());
        b.resolve_nodes[0].features.sort();
        assert_ne!(payload_id(&a).unwrap(), payload_id(&b).unwrap());
    }

    #[test]
    fn omitted_required_features_are_empty() {
        assert!(optional_string_array(None).unwrap().is_empty());
        assert!(optional_string_array(Some(&Value::Null)).unwrap().is_empty());
    }

    #[test]
    fn dependency_kind_order_is_canonicalizable() {
        let mut kinds = [
            DependencyKindSnapshot {
                kind: "dev".into(),
                target: None,
            },
            DependencyKindSnapshot {
                kind: "normal".into(),
                target: Some("cfg(unix)".into()),
            },
        ];
        kinds.sort();
        assert_eq!(kinds[0].kind, "dev");
        assert_eq!(kinds[1].kind, "normal");
    }
}
