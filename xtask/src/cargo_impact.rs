use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

const SNAPSHOT_SCHEMA: &str = "symthaea.cargo-graph-snapshot.v1";
const SNAPSHOT_HASH_DOMAIN: &[u8] = b"symthaea.cargo-graph-snapshot.v1\0";
const IMPACT_SCHEMA: &str = "symthaea.cargo-impact-report.v1";
const CLOSURE_HASH_DOMAIN: &[u8] = b"symthaea.cargo-impact-closure.v1\0";

/// Strict reader for the flat JSON emitted by ENV-CARGO-001A.
///
/// Unknown fields fail closed. If the snapshot schema evolves, this reader
/// must be deliberately updated before it can make CI-routing decisions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    snapshot_id: String,
    schema: String,
    lock_sha256: String,
    workspace_manifest_sha256: String,
    workspace_members: Vec<String>,
    packages: Vec<PackageRecord>,
    resolve_nodes: Vec<ResolveNodeRecord>,
}

/// Exact payload shape hashed by `cargo_graph` (everything except snapshot_id).
#[derive(Serialize)]
struct SnapshotPayloadRef<'a> {
    schema: &'a str,
    lock_sha256: &'a str,
    workspace_manifest_sha256: &'a str,
    workspace_members: &'a [String],
    packages: &'a [PackageRecord],
    resolve_nodes: &'a [ResolveNodeRecord],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PackageRecord {
    id: String,
    origin: PackageOrigin,
    name: String,
    version: String,
    source: Option<String>,
    checksum: Option<String>,
    links: Option<String>,
    edition: Option<String>,
    rust_version: Option<String>,
    workspace_manifest_path: Option<String>,
    local_manifest_sha256: Option<String>,
    features: BTreeMap<String, Vec<String>>,
    targets: Vec<TargetRecord>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PackageOrigin {
    Workspace,
    LocalPath,
    External,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TargetRecord {
    name: String,
    kind: Vec<String>,
    crate_types: Vec<String>,
    required_features: Vec<String>,
    edition: Option<String>,
    doc: Option<bool>,
    doctest: Option<bool>,
    test: Option<bool>,
    workspace_src_path: Option<String>,
    local_src_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ResolveNodeRecord {
    package: String,
    features: Vec<String>,
    deps: Vec<DependencyRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct DependencyRecord {
    name: String,
    package: String,
    dep_kinds: Vec<DependencyKindRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct DependencyKindRecord {
    kind: String,
    target: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ImpactDocument {
    schema: &'static str,
    base_snapshot_id: String,
    head_snapshot_id: String,
    reports: Vec<PackageImpactReport>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PackageImpactReport {
    selector: String,
    state: ImpactState,
    changes: Vec<ImpactChange>,
    base_root_id: Option<String>,
    head_root_id: Option<String>,
    base_closure_fingerprint: Option<String>,
    head_closure_fingerprint: Option<String>,
    unknown_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImpactState {
    Unchanged,
    DependencyClosureChanged,
    PackageSurfaceChanged,
    FeatureSurfaceChanged,
    DependencyGraphChanged,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ImpactChange {
    DependencyClosure,
    PackageSurface,
    FeatureSurface,
    DependencyGraph,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ClosureSurface {
    packages: Vec<PackageRecord>,
    resolve_nodes: Vec<ResolveNodeRecord>,
}

struct SnapshotIndex<'a> {
    snapshot: &'a Snapshot,
    packages: HashMap<&'a str, &'a PackageRecord>,
    resolve_nodes: HashMap<&'a str, &'a ResolveNodeRecord>,
}

impl<'a> SnapshotIndex<'a> {
    fn new(snapshot: &'a Snapshot) -> Self {
        Self {
            snapshot,
            packages: snapshot
                .packages
                .iter()
                .map(|package| (package.id.as_str(), package))
                .collect(),
            resolve_nodes: snapshot
                .resolve_nodes
                .iter()
                .map(|node| (node.package.as_str(), node))
                .collect(),
        }
    }

    fn select_workspace_root(&self, selector: &str) -> Result<&'a str, String> {
        let matches: Vec<&PackageRecord> = self
            .snapshot
            .packages
            .iter()
            .filter(|package| {
                package.origin == PackageOrigin::Workspace
                    && (package.id == selector
                        || package.name == selector
                        || package.workspace_manifest_path.as_deref() == Some(selector))
            })
            .collect();

        match matches.as_slice() {
            [package] => Ok(package.id.as_str()),
            [] => Err(format!("workspace package selector not found: {selector}")),
            _ => Err(format!(
                "workspace package selector is ambiguous ({selector} matched {} packages)",
                matches.len()
            )),
        }
    }

    fn closure(&self, root: &str) -> Result<ClosureSurface, String> {
        let mut pending = vec![root.to_string()];
        let mut visited = BTreeSet::new();

        while let Some(package_id) = pending.pop() {
            if !visited.insert(package_id.clone()) {
                continue;
            }

            if !self.packages.contains_key(package_id.as_str()) {
                return Err(format!("resolved package missing package record: {package_id}"));
            }
            let node = self
                .resolve_nodes
                .get(package_id.as_str())
                .ok_or_else(|| format!("package missing resolve node: {package_id}"))?;

            for dependency in &node.deps {
                if !self.packages.contains_key(dependency.package.as_str()) {
                    return Err(format!(
                        "dependency {} references missing package {}",
                        dependency.name, dependency.package
                    ));
                }
                pending.push(dependency.package.clone());
            }
        }

        let mut packages = Vec::with_capacity(visited.len());
        let mut resolve_nodes = Vec::with_capacity(visited.len());
        for package_id in &visited {
            packages.push((*self
                .packages
                .get(package_id.as_str())
                .expect("validated closure package must exist"))
            .clone());
            resolve_nodes.push((*self
                .resolve_nodes
                .get(package_id.as_str())
                .expect("validated closure node must exist"))
            .clone());
        }
        packages.sort_by(|a, b| a.id.cmp(&b.id));
        canonicalize_resolve_nodes(&mut resolve_nodes);

        Ok(ClosureSurface {
            packages,
            resolve_nodes,
        })
    }
}

pub fn run(
    base_path: &Path,
    head_path: &Path,
    package_selectors: Vec<String>,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    if package_selectors.is_empty() {
        bail!("cargo-impact requires at least one --package selector");
    }

    let base = read_snapshot(base_path)?;
    let head = read_snapshot(head_path)?;
    let document = compare_snapshots(&base, &head, package_selectors);
    let mut rendered = serde_json::to_string_pretty(&document)?;
    rendered.push('\n');

    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create impact output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write Cargo impact report {}", path.display()))?;
        println!("Cargo impact report written to {}", path.display());
    } else {
        print!("{rendered}");
    }

    Ok(())
}

fn read_snapshot(path: &Path) -> anyhow::Result<Snapshot> {
    let bytes = fs::read(path)
        .with_context(|| format!("read Cargo graph snapshot {}", path.display()))?;
    let mut snapshot: Snapshot = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse Cargo graph snapshot {}", path.display()))?;
    canonicalize_snapshot(&mut snapshot);
    verify_snapshot(&snapshot)
        .with_context(|| format!("verify Cargo graph snapshot {}", path.display()))?;
    Ok(snapshot)
}

fn verify_snapshot(snapshot: &Snapshot) -> anyhow::Result<()> {
    if snapshot.schema != SNAPSHOT_SCHEMA {
        bail!("unsupported Cargo graph snapshot schema: {}", snapshot.schema);
    }
    let expected = snapshot_payload_id(snapshot)?;
    if snapshot.snapshot_id != expected {
        bail!(
            "Cargo graph snapshot identity mismatch: declared={}, computed={expected}",
            snapshot.snapshot_id
        );
    }
    Ok(())
}

fn compare_snapshots(
    base: &Snapshot,
    head: &Snapshot,
    mut selectors: Vec<String>,
) -> ImpactDocument {
    selectors.sort();
    selectors.dedup();

    let base_index = SnapshotIndex::new(base);
    let head_index = SnapshotIndex::new(head);
    let reports = selectors
        .into_iter()
        .map(|selector| compare_package(&base_index, &head_index, selector))
        .collect();

    ImpactDocument {
        schema: IMPACT_SCHEMA,
        base_snapshot_id: base.snapshot_id.clone(),
        head_snapshot_id: head.snapshot_id.clone(),
        reports,
    }
}

fn compare_package(
    base: &SnapshotIndex<'_>,
    head: &SnapshotIndex<'_>,
    selector: String,
) -> PackageImpactReport {
    let base_root = match base.select_workspace_root(&selector) {
        Ok(root) => root,
        Err(reason) => return unknown_report(selector, None, None, reason),
    };
    let head_root = match head.select_workspace_root(&selector) {
        Ok(root) => root,
        Err(reason) => {
            return unknown_report(selector, Some(base_root.to_string()), None, reason);
        }
    };

    let base_closure = match base.closure(base_root) {
        Ok(closure) => closure,
        Err(reason) => {
            return unknown_report(
                selector,
                Some(base_root.to_string()),
                Some(head_root.to_string()),
                format!("base closure incomplete: {reason}"),
            );
        }
    };
    let head_closure = match head.closure(head_root) {
        Ok(closure) => closure,
        Err(reason) => {
            return unknown_report(
                selector,
                Some(base_root.to_string()),
                Some(head_root.to_string()),
                format!("head closure incomplete: {reason}"),
            );
        }
    };

    let base_fingerprint = match closure_fingerprint(&base_closure) {
        Ok(value) => value,
        Err(error) => {
            return unknown_report(
                selector,
                Some(base_root.to_string()),
                Some(head_root.to_string()),
                format!("base closure fingerprint failed: {error}"),
            );
        }
    };
    let head_fingerprint = match closure_fingerprint(&head_closure) {
        Ok(value) => value,
        Err(error) => {
            return unknown_report(
                selector,
                Some(base_root.to_string()),
                Some(head_root.to_string()),
                format!("head closure fingerprint failed: {error}"),
            );
        }
    };

    let mut changes = BTreeSet::new();
    let base_ids: BTreeSet<&str> = base_closure
        .packages
        .iter()
        .map(|package| package.id.as_str())
        .collect();
    let head_ids: BTreeSet<&str> = head_closure
        .packages
        .iter()
        .map(|package| package.id.as_str())
        .collect();
    if base_ids != head_ids {
        changes.insert(ImpactChange::DependencyClosure);
    }

    let base_packages: BTreeMap<&str, &PackageRecord> = base_closure
        .packages
        .iter()
        .map(|package| (package.id.as_str(), package))
        .collect();
    let head_packages: BTreeMap<&str, &PackageRecord> = head_closure
        .packages
        .iter()
        .map(|package| (package.id.as_str(), package))
        .collect();
    if base_ids
        .intersection(&head_ids)
        .any(|id| base_packages.get(*id) != head_packages.get(*id))
    {
        changes.insert(ImpactChange::PackageSurface);
    }

    if feature_surface(&base_closure.resolve_nodes) != feature_surface(&head_closure.resolve_nodes) {
        changes.insert(ImpactChange::FeatureSurface);
    }
    if dependency_surface(&base_closure.resolve_nodes)
        != dependency_surface(&head_closure.resolve_nodes)
    {
        changes.insert(ImpactChange::DependencyGraph);
    }

    let changes: Vec<ImpactChange> = changes.into_iter().collect();
    PackageImpactReport {
        selector,
        state: primary_state(&changes),
        changes,
        base_root_id: Some(base_root.to_string()),
        head_root_id: Some(head_root.to_string()),
        base_closure_fingerprint: Some(base_fingerprint),
        head_closure_fingerprint: Some(head_fingerprint),
        unknown_reason: None,
    }
}

fn unknown_report(
    selector: String,
    base_root_id: Option<String>,
    head_root_id: Option<String>,
    reason: String,
) -> PackageImpactReport {
    PackageImpactReport {
        selector,
        state: ImpactState::Unknown,
        changes: Vec::new(),
        base_root_id,
        head_root_id,
        base_closure_fingerprint: None,
        head_closure_fingerprint: None,
        unknown_reason: Some(reason),
    }
}

fn primary_state(changes: &[ImpactChange]) -> ImpactState {
    if changes.contains(&ImpactChange::DependencyClosure) {
        ImpactState::DependencyClosureChanged
    } else if changes.contains(&ImpactChange::PackageSurface) {
        ImpactState::PackageSurfaceChanged
    } else if changes.contains(&ImpactChange::FeatureSurface) {
        ImpactState::FeatureSurfaceChanged
    } else if changes.contains(&ImpactChange::DependencyGraph) {
        ImpactState::DependencyGraphChanged
    } else {
        ImpactState::Unchanged
    }
}

fn feature_surface(nodes: &[ResolveNodeRecord]) -> BTreeMap<String, Vec<String>> {
    nodes
        .iter()
        .map(|node| (node.package.clone(), node.features.clone()))
        .collect()
}

fn dependency_surface(nodes: &[ResolveNodeRecord]) -> BTreeMap<String, Vec<DependencyRecord>> {
    nodes
        .iter()
        .map(|node| (node.package.clone(), node.deps.clone()))
        .collect()
}

fn closure_fingerprint(surface: &ClosureSurface) -> anyhow::Result<String> {
    let bytes = serde_json::to_vec(surface).context("serialize canonical dependency closure")?;
    Ok(domain_sha256(CLOSURE_HASH_DOMAIN, &bytes))
}

fn snapshot_payload_id(snapshot: &Snapshot) -> anyhow::Result<String> {
    let payload = SnapshotPayloadRef {
        schema: &snapshot.schema,
        lock_sha256: &snapshot.lock_sha256,
        workspace_manifest_sha256: &snapshot.workspace_manifest_sha256,
        workspace_members: &snapshot.workspace_members,
        packages: &snapshot.packages,
        resolve_nodes: &snapshot.resolve_nodes,
    };
    let bytes = serde_json::to_vec(&payload).context("serialize Cargo graph snapshot payload")?;
    Ok(domain_sha256(SNAPSHOT_HASH_DOMAIN, &bytes))
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

fn canonicalize_snapshot(snapshot: &mut Snapshot) {
    snapshot.workspace_members.sort();
    snapshot.workspace_members.dedup();
    snapshot.packages.sort_by(|a, b| a.id.cmp(&b.id));
    for package in &mut snapshot.packages {
        for members in package.features.values_mut() {
            members.sort();
            members.dedup();
        }
        for target in &mut package.targets {
            target.kind.sort();
            target.kind.dedup();
            target.crate_types.sort();
            target.crate_types.dedup();
            target.required_features.sort();
            target.required_features.dedup();
        }
        package.targets.sort_by(|a, b| {
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
    }
    canonicalize_resolve_nodes(&mut snapshot.resolve_nodes);
}

fn canonicalize_resolve_nodes(nodes: &mut Vec<ResolveNodeRecord>) {
    for node in nodes.iter_mut() {
        node.features.sort();
        node.features.dedup();
        for dependency in &mut node.deps {
            dependency.dep_kinds.sort();
            dependency.dep_kinds.dedup();
        }
        node.deps.sort();
        node.deps.dedup();
    }
    nodes.sort_by(|a, b| a.package.cmp(&b.package));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package(id: &str, name: &str) -> PackageRecord {
        let workspace = id.starts_with("workspace:");
        PackageRecord {
            id: id.into(),
            origin: if workspace {
                PackageOrigin::Workspace
            } else {
                PackageOrigin::External
            },
            name: name.into(),
            version: "1.0.0".into(),
            source: None,
            checksum: None,
            links: None,
            edition: Some("2024".into()),
            rust_version: None,
            workspace_manifest_path: workspace.then(|| format!("crates/{name}/Cargo.toml")),
            local_manifest_sha256: workspace.then(|| "aa".repeat(32)),
            features: BTreeMap::new(),
            targets: vec![TargetRecord {
                name: name.into(),
                kind: vec!["lib".into()],
                crate_types: vec!["lib".into()],
                required_features: vec![],
                edition: Some("2024".into()),
                doc: Some(true),
                doctest: Some(true),
                test: Some(true),
                workspace_src_path: workspace.then(|| format!("crates/{name}/src/lib.rs")),
                local_src_sha256: workspace.then(|| "bb".repeat(32)),
            }],
        }
    }

    fn dep(name: &str, package: &str) -> DependencyRecord {
        DependencyRecord {
            name: name.into(),
            package: package.into(),
            dep_kinds: vec![DependencyKindRecord {
                kind: "normal".into(),
                target: None,
            }],
        }
    }

    fn snapshot(packages: Vec<PackageRecord>, nodes: Vec<ResolveNodeRecord>) -> Snapshot {
        let mut snapshot = Snapshot {
            snapshot_id: String::new(),
            schema: SNAPSHOT_SCHEMA.into(),
            lock_sha256: "11".repeat(32),
            workspace_manifest_sha256: "22".repeat(32),
            workspace_members: packages
                .iter()
                .filter(|package| package.origin == PackageOrigin::Workspace)
                .map(|package| package.id.clone())
                .collect(),
            packages,
            resolve_nodes: nodes,
        };
        reseal(&mut snapshot);
        snapshot
    }

    fn reseal(snapshot: &mut Snapshot) {
        canonicalize_snapshot(snapshot);
        snapshot.snapshot_id = snapshot_payload_id(snapshot).unwrap();
    }

    fn root_node(snapshot: &mut Snapshot) -> &mut ResolveNodeRecord {
        snapshot
            .resolve_nodes
            .iter_mut()
            .find(|node| node.package.starts_with("workspace:root#"))
            .expect("root resolve node")
    }

    fn root_and_dep() -> Snapshot {
        snapshot(
            vec![
                package("workspace:root#root@1.0.0", "root"),
                package("external:dep#dep@1.0.0", "dep"),
            ],
            vec![
                ResolveNodeRecord {
                    package: "workspace:root#root@1.0.0".into(),
                    features: vec!["default".into()],
                    deps: vec![dep("dep", "external:dep#dep@1.0.0")],
                },
                ResolveNodeRecord {
                    package: "external:dep#dep@1.0.0".into(),
                    features: vec![],
                    deps: vec![],
                },
            ],
        )
    }

    fn report(base: &Snapshot, head: &Snapshot) -> PackageImpactReport {
        compare_snapshots(base, head, vec!["root".into()])
            .reports
            .into_iter()
            .next()
            .unwrap()
    }

    #[test]
    fn identical_closure_is_unchanged() {
        let base = root_and_dep();
        let head = base.clone();
        let report = report(&base, &head);
        assert_eq!(report.state, ImpactState::Unchanged);
        assert!(report.changes.is_empty());
        assert_eq!(
            report.base_closure_fingerprint,
            report.head_closure_fingerprint
        );
    }

    #[test]
    fn lock_hash_change_alone_does_not_impact_selected_closure() {
        let base = root_and_dep();
        let mut head = base.clone();
        head.lock_sha256 = "33".repeat(32);
        reseal(&mut head);

        assert_ne!(base.snapshot_id, head.snapshot_id);
        let report = report(&base, &head);
        assert_eq!(report.state, ImpactState::Unchanged);
        assert_eq!(
            report.base_closure_fingerprint,
            report.head_closure_fingerprint
        );
    }

    #[test]
    fn unrelated_package_change_does_not_impact_root() {
        let base = root_and_dep();
        let mut head = base.clone();
        head.packages
            .push(package("external:unrelated#unrelated@1.0.0", "unrelated"));
        head.resolve_nodes.push(ResolveNodeRecord {
            package: "external:unrelated#unrelated@1.0.0".into(),
            features: vec![],
            deps: vec![],
        });
        reseal(&mut head);

        assert_eq!(report(&base, &head).state, ImpactState::Unchanged);
    }

    #[test]
    fn transitive_package_identity_change_is_closure_change() {
        let base = root_and_dep();
        let mut head = base.clone();
        head.packages.retain(|package| package.name != "dep");
        head.packages
            .push(package("external:dep#dep@2.0.0", "dep"));
        root_node(&mut head).deps[0].package = "external:dep#dep@2.0.0".into();
        let dep_node = head
            .resolve_nodes
            .iter_mut()
            .find(|node| node.package.starts_with("external:dep#"))
            .unwrap();
        dep_node.package = "external:dep#dep@2.0.0".into();
        reseal(&mut head);

        let report = report(&base, &head);
        assert_eq!(report.state, ImpactState::DependencyClosureChanged);
        assert!(report.changes.contains(&ImpactChange::DependencyClosure));
    }

    #[test]
    fn enabled_feature_change_is_reported_separately() {
        let base = root_and_dep();
        let mut head = base.clone();
        root_node(&mut head).features.push("feature-a".into());
        reseal(&mut head);

        let report = report(&base, &head);
        assert_eq!(report.state, ImpactState::FeatureSurfaceChanged);
        assert_eq!(report.changes, vec![ImpactChange::FeatureSurface]);
    }

    #[test]
    fn dependency_kind_change_is_graph_change() {
        let base = root_and_dep();
        let mut head = base.clone();
        root_node(&mut head).deps[0].dep_kinds[0].kind = "build".into();
        reseal(&mut head);

        let report = report(&base, &head);
        assert_eq!(report.state, ImpactState::DependencyGraphChanged);
        assert!(report.changes.contains(&ImpactChange::DependencyGraph));
    }

    #[test]
    fn source_surface_change_is_not_hidden_by_stable_package_id() {
        let base = root_and_dep();
        let mut head = base.clone();
        let root = head
            .packages
            .iter_mut()
            .find(|package| package.name == "root")
            .unwrap();
        root.local_manifest_sha256 = Some("cc".repeat(32));
        reseal(&mut head);

        let report = report(&base, &head);
        assert_eq!(report.state, ImpactState::PackageSurfaceChanged);
        assert!(report.changes.contains(&ImpactChange::PackageSurface));
    }

    #[test]
    fn incomplete_graph_fails_toward_unknown() {
        let base = root_and_dep();
        let mut head = base.clone();
        head.resolve_nodes
            .retain(|node| !node.package.starts_with("external:dep#"));
        reseal(&mut head);

        let report = report(&base, &head);
        assert_eq!(report.state, ImpactState::Unknown);
        assert!(report
            .unknown_reason
            .as_deref()
            .unwrap()
            .contains("missing resolve node"));
    }

    #[test]
    fn missing_selector_is_unknown_not_unaffected() {
        let base = root_and_dep();
        let head = base.clone();
        let report = compare_snapshots(&base, &head, vec!["does-not-exist".into()])
            .reports
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(report.state, ImpactState::Unknown);
    }

    #[test]
    fn snapshot_identity_mismatch_is_rejected() {
        let mut snapshot = root_and_dep();
        snapshot.snapshot_id = "00".repeat(32);
        assert!(verify_snapshot(&snapshot).is_err());
    }

    #[test]
    fn selector_order_is_canonical() {
        let base = root_and_dep();
        let head = base.clone();
        let a = compare_snapshots(&base, &head, vec!["root".into(), "root".into()]);
        let b = compare_snapshots(&base, &head, vec!["root".into()]);
        assert_eq!(a, b);
    }
}
