//! Deterministic ownership audit for Cargo workspace dependency aliases.
//!
//! Cargo's unused-workspace-dependency lint only sees the workspace Cargo resolved. This
//! report maps each workspace declaration to inheriting member manifests and separately
//! records same-name declarations in nested, excluded, or standalone packages. It never
//! edits a manifest or lockfile and never authorizes dependency removal.

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use toml::Value;

const SKIP_DIRS: &[&str] = &[
    ".git", ".direnv", ".devenv", "target", "node_modules", "vendor",
    "result", "result-bin", ".venv", "__pycache__",
];

#[derive(Clone, Debug)]
struct Use {
    alias: String,
    package_name: Option<String>,
    section: String,
    inherited: bool,
    optional: bool,
    feature_only: bool,
}

#[derive(Clone, Debug)]
struct Manifest {
    path: String,
    hash: String,
    package: Option<String>,
    workspace_root: bool,
    uses: Vec<Use>,
    parse_error: Option<String>,
}

#[derive(Clone, Debug)]
struct Workspace {
    path: String,
    package: Option<String>,
    hash: String,
    members: Vec<String>,
    excludes: Vec<String>,
    dependencies: BTreeMap<String, Value>,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    git_commit: String,
    cargo_version: String,
    root_manifest_sha256: String,
    scanned_manifest_count: usize,
    parse_errors: Vec<ParseError>,
    workspaces: Vec<WorkspaceReport>,
    claim_ceiling: &'static str,
}

#[derive(Serialize)]
struct ParseError {
    manifest: String,
    sha256: String,
    error: String,
}

#[derive(Serialize)]
struct WorkspaceReport {
    workspace_root: String,
    package: Option<String>,
    manifest_sha256: String,
    member_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
    dependency_count: usize,
    categories: BTreeMap<String, usize>,
    dependencies: Vec<DependencyReport>,
}

#[derive(Serialize)]
struct DependencyReport {
    alias: String,
    package_name: String,
    classification: String,
    inheriting_manifests: Vec<Owner>,
    same_name_non_inheriting_manifests: Vec<Owner>,
    notes: Vec<String>,
}

#[derive(Serialize)]
struct Owner {
    manifest: String,
    package: Option<String>,
    section: String,
    manifest_sha256: String,
    optional: bool,
    feature_only: bool,
    inherits_this_entry: bool,
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn command_output(root: &Path, command: &str, args: &[&str]) -> Result<String> {
    let out = Command::new(command).args(args).current_dir(root).output()
        .with_context(|| format!("could not run {command}"))?;
    if !out.status.success() {
        anyhow::bail!("{command} {:?} failed: {}", args, String::from_utf8_lossy(&out.stderr));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn relative(root: &Path, path: &Path) -> Result<String> {
    let rel = path.strip_prefix(root)?;
    let parts: Vec<String> = rel.components()
        .filter_map(|c| c.as_os_str().to_str().map(str::to_string))
        .collect();
    Ok(if parts.is_empty() { ".".into() } else { parts.join("/") })
}

fn scan_paths(root: &Path) -> Result<Vec<PathBuf>> {
    fn walk(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_symlink() { continue; }
            if kind.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !SKIP_DIRS.contains(&name.as_str()) { walk(&path, paths)?; }
            } else if kind.is_file()
                && path.file_name().and_then(|s| s.to_str()) == Some("Cargo.toml")
            {
                paths.push(path);
            }
        }
        Ok(())
    }
    let mut paths = Vec::new();
    walk(root, &mut paths)?;
    paths.sort();
    Ok(paths)
}

fn default_feature_names(value: Option<&Value>) -> BTreeSet<String> {
    value.and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(|s| s.to_string()).collect())
        .unwrap_or_default()
}

fn collect_section(
    value: Option<&Value>,
    section: &str,
    defaults: &BTreeSet<String>,
) -> Vec<Use> {
    let mut out = Vec::new();
    let Some(table) = value.and_then(Value::as_table) else { return out };
    for (alias, config) in table {
        let cfg = config.as_table();
        let inherited = cfg.and_then(|t| t.get("workspace")).and_then(Value::as_bool).unwrap_or(false);
        let package_name = cfg.and_then(|t| t.get("package")).and_then(Value::as_str).map(str::to_string);
        let optional = cfg.and_then(|t| t.get("optional")).and_then(Value::as_bool).unwrap_or(false);
        let feature_only = optional && !defaults.contains(alias) && !defaults.contains(&format!("dep:{alias}"));
        out.push(Use {
            alias: alias.clone(), package_name, section: section.to_string(),
            inherited, optional, feature_only,
        });
    }
    out
}

fn parse_manifest(root: &Path, path: &Path) -> Result<Manifest> {
    let bytes = fs::read(path).with_context(|| format!("could not read {}", path.display()))?;
    let path_string = relative(root, path)?;
    let digest = hash(&bytes);
    let parsed: Value = match std::str::from_utf8(&bytes)
        .map_err(anyhow::Error::from)
        .and_then(|s| toml::from_str(s).map_err(anyhow::Error::from))
    {
        Ok(value) => value,
        Err(error) => return Ok(Manifest {
            path: path_string, hash: digest, package: None, workspace_root: false,
            uses: Vec::new(), parse_error: Some(error.to_string()),
        }),
    };
    let package = parsed.get("package").and_then(Value::as_table)
        .and_then(|t| t.get("name")).and_then(Value::as_str).map(str::to_string);
    let workspace_root = parsed.get("workspace").and_then(Value::as_table).is_some();
    let defaults = default_feature_names(parsed.get("features").and_then(Value::as_table).and_then(|t| t.get("default")));
    let mut uses = Vec::new();
    for (key, label) in [
        ("dependencies", "dependencies"),
        ("dev-dependencies", "dev-dependencies"),
        ("build-dependencies", "build-dependencies"),
    ] {
        uses.extend(collect_section(parsed.get(key), label, &defaults));
    }
    if let Some(targets) = parsed.get("target").and_then(Value::as_table) {
        for (target, config) in targets {
            for (key, suffix) in [
                ("dependencies", "dependencies"),
                ("dev-dependencies", "dev-dependencies"),
                ("build-dependencies", "build-dependencies"),
            ] {
                let section = config.as_table().and_then(|t| t.get(key));
                uses.extend(collect_section(section, &format!("target.{target}.{suffix}"), &defaults));
            }
        }
    }
    Ok(Manifest { path: path_string, hash: digest, package, workspace_root, uses, parse_error: None })
}

fn dir_of(manifest: &str) -> String {
    Path::new(manifest).parent().and_then(Path::to_str)
        .filter(|s| !s.is_empty()).unwrap_or(".").to_string()
}

fn is_within(parent: &str, child: &str) -> bool {
    parent == "." || child == parent || child.starts_with(&format!("{parent}/"))
}

fn pattern_segments(pattern: &str) -> Vec<&str> {
    pattern.trim_matches('/').split('/').filter(|s| !s.is_empty() && *s != ".").collect()
}

fn glob_segments(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((&"**", rest)) => glob_segments(rest, path)
            || (!path.is_empty() && glob_segments(pattern, &path[1..])),
        Some((&"*", rest)) => !path.is_empty() && glob_segments(rest, &path[1..]),
        Some((part, rest)) => !path.is_empty() && *part == path[0] && glob_segments(rest, &path[1..]),
    }
}

fn matches(pattern: &str, path: &str) -> bool {
    glob_segments(&pattern_segments(pattern), &pattern_segments(path))
}

fn nearest_workspace<'a>(manifest: &Manifest, workspaces: &'a [Workspace]) -> Option<&'a Workspace> {
    let candidate = dir_of(&manifest.path);
    workspaces.iter()
        .filter(|w| is_within(&dir_of(&w.path), &candidate))
        .max_by_key(|w| dir_of(&w.path).len())
}

fn relative_to_workspace(manifest: &str, workspace: &str) -> Option<String> {
    let member_dir = dir_of(manifest);
    let workspace_dir = dir_of(workspace);
    if member_dir == workspace_dir { return Some(".".into()); }
    if workspace_dir == "." { return Some(member_dir); }
    member_dir.strip_prefix(&format!("{workspace_dir}/")).map(str::to_string)
}

fn membership(manifest: &Manifest, workspace: &Workspace) -> (bool, bool) {
    let Some(path) = relative_to_workspace(&manifest.path, &workspace.path) else { return (false, false) };
    let excluded = workspace.excludes.iter().any(|p| matches(p, &path));
    if excluded { return (false, true); }
    let root_package = path == "." && manifest.package.is_some();
    (root_package || workspace.members.iter().any(|p| matches(p, &path)), false)
}

fn load_workspaces(root: &Path, manifests: &[Manifest]) -> Result<Vec<Workspace>> {
    let mut out = Vec::new();
    for m in manifests.iter().filter(|m| m.workspace_root && m.parse_error.is_none()) {
        let bytes = fs::read(root.join(&m.path))?;
        let doc: Value = toml::from_str(std::str::from_utf8(&bytes)?)?;
        let table = doc.get("workspace").and_then(Value::as_table).context("workspace table missing")?;
        let strings = |key: &str| table.get(key).and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
            .unwrap_or_else(Vec::new);
        let dependencies = table.get("dependencies").and_then(Value::as_table)
            .map(|t| t.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        out.push(Workspace {
            path: m.path.clone(), package: m.package.clone(), hash: m.hash.clone(),
            members: strings("members"), excludes: strings("exclude"), dependencies,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

fn classify(alias: &str, uses: &[&Use]) -> (String, Vec<String>) {
    if uses.is_empty() {
        return ("UNUSED_IN_ROOT_WORKSPACE".into(), vec![
            "No member manifest in this workspace inherits this dependency alias.".into()
        ]);
    }
    let normal = uses.iter().any(|u| u.section == "dependencies" || u.section.ends_with(".dependencies"));
    if !normal && uses.iter().all(|u| u.section.contains("dev-dependencies") || u.section.contains("build-dependencies")) {
        return ("USED_ONLY_BY_BUILD_OR_TEST_TARGET".into(), vec![
            "All inheritance sites are development or build dependencies.".into()
        ]);
    }
    if uses.iter().all(|u| u.optional && u.feature_only) {
        return ("USED_ONLY_UNDER_FEATURE".into(), vec![
            format!("All inheritance sites for {alias} are optional and excluded from default features.")
        ]);
    }
    ("USED_IN_WORKSPACE".into(), vec![
        "At least one member inherits this workspace dependency.".into()
    ])
}

fn dependency_report(alias: &str, config: &Value, ws: &Workspace, manifests: &[Manifest], workspaces: &[Workspace]) -> DependencyReport {
    let package_name = config.as_table().and_then(|t| t.get("package"))
        .and_then(Value::as_str).unwrap_or(alias).to_string();
    let mut inheriting = Vec::new();
    let mut external = Vec::new();
    let mut notes = Vec::new();

    for m in manifests.iter().filter(|m| m.parse_error.is_none()) {
        let nearest = nearest_workspace(m, workspaces);
        let same_ws = nearest.is_some_and(|w| w.path == ws.path);
        let (member, excluded) = membership(m, ws);
        for u in m.uses.iter().filter(|u| u.alias == alias || u.package_name.as_deref() == Some(package_name.as_str())) {
            let inherits = u.inherited && same_ws && member && !excluded;
            let owner = Owner {
                manifest: m.path.clone(), package: m.package.clone(), section: u.section.clone(),
                manifest_sha256: m.hash.clone(), optional: u.optional, feature_only: u.feature_only,
                inherits_this_entry: inherits,
            };
            if inherits {
                inheriting.push(owner);
            } else {
                external.push(owner);
                if excluded {
                    notes.push(format!("{} is under a workspace exclude pattern; this is a same-name candidate, not an inheritance site.", m.path));
                } else if nearest.is_some_and(|w| w.path != ws.path) {
                    notes.push(format!("{} belongs to a nested/independent workspace and does not inherit this workspace's version.", m.path));
                } else if !member {
                    notes.push(format!("{} is not matched by this workspace's member patterns.", m.path));
                }
            }
        }
    }

    let uses: Vec<&Use> = manifests.iter()
        .filter(|m| m.parse_error.is_none())
        .filter(|m| nearest_workspace(m, workspaces).is_some_and(|w| w.path == ws.path))
        .filter(|m| membership(m, ws).0 && !membership(m, ws).1)
        .flat_map(|m| m.uses.iter().filter(|u| u.inherited && (u.alias == alias || u.package_name.as_deref() == Some(package_name.as_str()))))
        .collect();
    let (mut classification, mut reasons) = classify(alias, &uses);
    if uses.is_empty() && ws.path != "Cargo.toml" {
        classification = "UNUSED_IN_NESTED_WORKSPACE".into();
    }

    // A malformed manifest can hide an inheritance edge. Keep this unresolved rather
    // than treating it as unused. Parse errors wholly inside a deeper workspace are separate.
    let workspace_dir = dir_of(&ws.path);
    let malformed_in_scope = manifests.iter().filter(|m| m.parse_error.is_some()).any(|m| {
        let candidate_dir = dir_of(&m.path);
        is_within(&workspace_dir, &candidate_dir)
            && !workspaces.iter().any(|nested| {
                nested.path != ws.path
                    && dir_of(&nested.path).len() > workspace_dir.len()
                    && is_within(&dir_of(&nested.path), &candidate_dir)
            })
    });
    if inheriting.is_empty() && malformed_in_scope {
        classification = "UNRESOLVED_OWNERSHIP".into();
        reasons.push("At least one manifest in this workspace scope could not be parsed; unused status is not established.".into());
    }

    if inheriting.is_empty() && !external.is_empty() {
        let has_excluded = external.iter().any(|o| manifests.iter().find(|m| m.path == o.manifest)
            .is_some_and(|m| membership(m, ws).1));
        let has_nested = external.iter().any(|o| manifests.iter().find(|m| m.path == o.manifest)
            .and_then(|m| nearest_workspace(m, workspaces)).is_some_and(|w| w.path != ws.path));
        if has_excluded {
            classification = "USED_BY_EXCLUDED_MEMBER".into();
            reasons.push("Same-name declarations exist under excluded paths, but they do not inherit this entry.".into());
        } else if has_nested {
            classification = "USED_BY_NESTED_WORKSPACE".into();
            reasons.push("Same-name declarations exist in nested workspaces, but they do not inherit this workspace entry.".into());
        } else {
            classification = "UNRESOLVED_OWNERSHIP".into();
            reasons.push("Same-name declarations exist but workspace inheritance is not proven.".into());
        }
    }
    if inheriting.is_empty() && external.is_empty() {
        reasons.push("No same-name declaration was found in scanned manifests; generated or external build contexts remain outside this observation.".into());
    }
    inheriting.sort_by(|a, b| a.manifest.cmp(&b.manifest).then(a.section.cmp(&b.section)));
    external.sort_by(|a, b| a.manifest.cmp(&b.manifest).then(a.section.cmp(&b.section)));
    notes.extend(reasons);
    notes.sort();
    notes.dedup();
    DependencyReport {
        alias: alias.to_string(), package_name, classification,
        inheriting_manifests: inheriting, same_name_non_inheriting_manifests: external,
        notes,
    }
}

fn build_report(root: &Path) -> Result<Report> {
    let paths = scan_paths(root)?;
    let mut manifests = Vec::new();
    let mut parse_errors = Vec::new();
    for path in paths {
        let m = parse_manifest(root, &path)?;
        if let Some(error) = &m.parse_error {
            parse_errors.push(ParseError { manifest: m.path.clone(), sha256: m.hash.clone(), error: error.clone() });
        }
        manifests.push(m);
    }
    manifests.sort_by(|a, b| a.path.cmp(&b.path));
    let workspaces = load_workspaces(root, &manifests)?;
    let mut workspace_reports = Vec::new();

    for ws in &workspaces {
        let mut dependencies = Vec::new();
        let mut categories = BTreeMap::new();
        for (alias, config) in &ws.dependencies {
            let d = dependency_report(alias, config, ws, &manifests, &workspaces);
            *categories.entry(d.classification.clone()).or_insert(0) += 1;
            dependencies.push(d);
        }
        dependencies.sort_by(|a, b| a.alias.cmp(&b.alias));
        workspace_reports.push(WorkspaceReport {
            workspace_root: ws.path.clone(), package: ws.package.clone(),
            manifest_sha256: ws.hash.clone(), member_patterns: ws.members.clone(),
            exclude_patterns: ws.excludes.clone(), dependency_count: dependencies.len(),
            categories, dependencies,
        });
    }

    let root_bytes = fs::read(root.join("Cargo.toml")).context("root Cargo.toml missing")?;
    Ok(Report {
        schema: "luminous.cargo-workspace-dependency-ownership.v1",
        git_commit: command_output(root, "git", &["rev-parse", "HEAD"] )?,
        cargo_version: command_output(root, &std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()), &["--version"] )?,
        root_manifest_sha256: hash(&root_bytes),
        scanned_manifest_count: manifests.len(),
        parse_errors,
        workspaces: workspace_reports,
        claim_ceiling: "Manifest ownership only. This report does not prove global non-use and does not authorize dependency removal.",
    })
}

fn render_markdown(report: &Report) -> String {
    let mut out = String::new();
    out.push_str("# Cargo workspace dependency ownership\n\n");
    out.push_str(&format!("- Commit: {}\n- Cargo: {}\n- Root manifest SHA-256: {}\n- Manifests scanned: {}\n- Workspace roots: {}\n\n",
        report.git_commit, report.cargo_version, report.root_manifest_sha256,
        report.scanned_manifest_count, report.workspaces.len()));
    out.push_str("> This is a manifest ownership census, not permission to prune dependencies.\n\n");
    for ws in &report.workspaces {
        out.push_str(&format!("## Workspace: {}\n\n", ws.workspace_root));
        out.push_str(&format!("- Manifest SHA-256: {}\n- Member patterns: {:?}\n- Exclude patterns: {:?}\n- Categories: {:?}\n\n",
            ws.manifest_sha256, ws.member_patterns, ws.exclude_patterns, ws.categories));
        out.push_str("| Alias | Package | Classification | Inheritors | Same-name candidates |\n|---|---|---|---:|---:|\n");
        for d in &ws.dependencies {
            out.push_str(&format!("| {} | {} | {} | {} | {} |\n",
                d.alias, d.package_name, d.classification,
                d.inheriting_manifests.len(), d.same_name_non_inheriting_manifests.len()));
        }
        out.push('\n');
        for d in &ws.dependencies {
            if d.inheriting_manifests.is_empty() && d.same_name_non_inheriting_manifests.is_empty() { continue; }
            out.push_str(&format!("### {}\n\n", d.alias));
            for owner in &d.inheriting_manifests {
                out.push_str(&format!("- INHERITS: {} — {} ({})\n",
                    owner.manifest, owner.package.as_deref().unwrap_or("?"), owner.section));
            }
            for owner in &d.same_name_non_inheriting_manifests {
                out.push_str(&format!("- SAME-NAME ONLY: {} — {} ({})\n",
                    owner.manifest, owner.package.as_deref().unwrap_or("?"), owner.section));
            }
            for note in &d.notes { out.push_str(&format!("- Note: {note}\n")); }
            out.push('\n');
        }
    }
    if !report.parse_errors.is_empty() {
        out.push_str("## Unresolved manifest parse errors\n\n");
        for e in &report.parse_errors { out.push_str(&format!("- {}: {}\n", e.manifest, e.error)); }
    }
    out
}

/// Build machine-readable JSON and a human-readable Markdown report; no manifests or locks are edited.
pub fn run(root: &Path, json_path: &Path, markdown_path: &Path) -> Result<()> {
    let report = build_report(root)?;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(parent) = json_path.parent() { fs::create_dir_all(parent)?; }
    if let Some(parent) = markdown_path.parent() { fs::create_dir_all(parent)?; }
    fs::write(json_path, format!("{json}\n"))?;
    fs::write(markdown_path, render_markdown(&report))?;
    println!("cargo-workspace-dependency-ownership: {} workspace roots, {} manifests, {} parse errors",
        report.workspaces.len(), report.scanned_manifest_count, report.parse_errors.len());
    println!("JSON: {}", json_path.display());
    println!("Markdown: {}", markdown_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_member_glob_supports_single_and_recursive_segments() {
        assert!(matches("crates/core/*", "crates/core/symthaea-core"));
        assert!(matches("crates/**", "crates/core/symthaea-core"));
        assert!(matches("**/tests", "crates/foo/tests"));
        assert!(!matches("crates/core/*", "crates/core/symthaea-core/tests"));
        assert!(!matches("crates/core/*", "crates/domains/foo"));
    }

    #[test]
    fn nearest_nested_workspace_takes_precedence() {
        let workspaces = vec![
            Workspace { path: "Cargo.toml".into(), package: Some("root".into()), hash: String::new(),
                members: vec!["nested".into(), "crates/*".into()], excludes: vec![], dependencies: BTreeMap::new() },
            Workspace { path: "nested/Cargo.toml".into(), package: None, hash: String::new(),
                members: vec!["crates/*".into()], excludes: vec![], dependencies: BTreeMap::new() },
        ];
        let manifest = Manifest { path: "nested/crates/demo/Cargo.toml".into(), hash: String::new(),
            package: Some("demo".into()), workspace_root: false, uses: vec![], parse_error: None };
        assert_eq!(nearest_workspace(&manifest, &workspaces).map(|w| w.path.as_str()), Some("nested/Cargo.toml"));
    }
    #[test]
    fn excludes_are_not_workspace_members() {
        let workspace = Workspace {
            path: "Cargo.toml".into(), package: Some("root".into()), hash: String::new(),
            members: vec!["crates/*".into()], excludes: vec!["crates/experimental".into()],
            dependencies: BTreeMap::new(),
        };
        let excluded = Manifest {
            path: "crates/experimental/Cargo.toml".into(), hash: String::new(),
            package: Some("experimental".into()), workspace_root: false, uses: vec![],
            parse_error: None,
        };
        assert_eq!(membership(&excluded, &workspace), (false, true));
    }

    #[test]
    fn build_and_test_uses_are_not_misclassified_as_runtime_dependencies() {
        let uses = vec![
            Use { alias: "serde".into(), package_name: None, section: "dev-dependencies".into(),
                inherited: true, optional: false, feature_only: false },
            Use { alias: "serde".into(), package_name: None, section: "build-dependencies".into(),
                inherited: true, optional: false, feature_only: false },
        ];
        assert_eq!(classify("serde", &uses).0, "USED_ONLY_BY_BUILD_OR_TEST_TARGET");
    }

    #[test]
    fn optional_dependency_can_be_feature_only() {
        let uses = vec![Use {
            alias: "simd".into(), package_name: None, section: "dependencies".into(),
            inherited: true, optional: true, feature_only: true,
        }];
        assert_eq!(classify("simd", &uses).0, "USED_ONLY_UNDER_FEATURE");
    }

    fn root_workspace() -> Workspace {
        Workspace {
            path: "Cargo.toml".into(),
            package: Some("symthaea".into()),
            hash: "root-manifest".into(),
            members: vec!["crates/*".into(), "xtask".into()],
            excludes: vec!["crates/archived".into()],
            dependencies: BTreeMap::new(),
        }
    }

    fn member_manifest(path: &str, package: &str, uses: Vec<Use>) -> Manifest {
        Manifest {
            path: path.into(),
            hash: format!("hash-{package}"),
            package: Some(package.into()),
            workspace_root: false,
            uses,
            parse_error: None,
        }
    }

    fn workspace_use(alias: &str, section: &str) -> Use {
        Use {
            alias: alias.into(),
            package_name: None,
            section: section.into(),
            inherited: true,
            optional: false,
            feature_only: false,
        }
    }

    fn direct_use(alias: &str) -> Use {
        Use {
            alias: alias.into(),
            package_name: None,
            section: "dependencies".into(),
            inherited: false,
            optional: false,
            feature_only: false,
        }
    }

    #[test]
    fn root_member_inheritance_is_counted_as_an_owner() {
        let mut ws = root_workspace();
        ws.dependencies.insert("serde".into(), Value::String("1".into()));
        let root = Manifest {
            path: "Cargo.toml".into(), hash: "root".into(), package: Some("symthaea".into()),
            workspace_root: true, uses: vec![], parse_error: None,
        };
        let member = member_manifest("crates/core/demo/Cargo.toml", "demo", vec![workspace_use("serde", "dependencies")]);
        let report = dependency_report("serde", &ws.dependencies["serde"], &ws, &[root, member], &[ws.clone()]);
        assert_eq!(report.classification, "USED_IN_WORKSPACE");
        assert_eq!(report.inheriting_manifests.len(), 1);
        assert_eq!(report.inheriting_manifests[0].manifest, "crates/core/demo/Cargo.toml");
    }

    #[test]
    fn direct_same_name_in_nested_workspace_does_not_inherit_root_version() {
        let mut ws = root_workspace();
        ws.dependencies.insert("serde".into(), Value::String("1".into()));
        let nested = Workspace {
            path: "crates/independent/Cargo.toml".into(), package: None,
            hash: "nested".into(), members: vec!["crates/*".into()], excludes: vec![],
            dependencies: BTreeMap::new(),
        };
        let root = Manifest {
            path: "Cargo.toml".into(), hash: "root".into(), package: Some("symthaea".into()),
            workspace_root: true, uses: vec![], parse_error: None,
        };
        let nested_member = member_manifest(
            "crates/independent/crates/demo/Cargo.toml",
            "demo",
            vec![direct_use("serde")],
        );
        let report = dependency_report(
            "serde", &ws.dependencies["serde"], &ws,
            &[root, nested_member], &[ws.clone(), nested],
        );
        assert_eq!(report.classification, "USED_BY_NESTED_WORKSPACE");
        assert!(report.inheriting_manifests.is_empty());
        assert_eq!(report.same_name_non_inheriting_manifests.len(), 1);
        assert!(!report.same_name_non_inheriting_manifests[0].inherits_this_entry);
    }

    #[test]
    fn excluded_member_is_not_counted_as_a_workspace_inheritor() {
        let mut ws = root_workspace();
        ws.dependencies.insert("serde".into(), Value::String("1".into()));
        let root = Manifest {
            path: "Cargo.toml".into(), hash: "root".into(), package: Some("symthaea".into()),
            workspace_root: true, uses: vec![], parse_error: None,
        };
        let excluded = member_manifest("crates/archived/Cargo.toml", "archived", vec![direct_use("serde")]);
        let report = dependency_report("serde", &ws.dependencies["serde"], &ws, &[root, excluded], &[ws.clone()]);
        assert_eq!(report.classification, "USED_BY_EXCLUDED_MEMBER");
        assert!(report.inheriting_manifests.is_empty());
    }

    #[test]
    fn root_dependency_without_inheritors_is_explicitly_unobserved() {
        let mut ws = root_workspace();
        ws.dependencies.insert("z3".into(), Value::String("0.12".into()));
        let root = Manifest {
            path: "Cargo.toml".into(), hash: "root".into(), package: Some("symthaea".into()),
            workspace_root: true, uses: vec![], parse_error: None,
        };
        let report = dependency_report("z3", &ws.dependencies["z3"], &ws, &[root], &[ws.clone()]);
        assert_eq!(report.classification, "UNUSED_IN_ROOT_WORKSPACE");
        assert!(report.inheriting_manifests.is_empty());
        assert!(report.notes.iter().any(|note| note.contains("not proof of global non-use")));
    }

}
