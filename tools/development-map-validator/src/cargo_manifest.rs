//! Pure, read-only Cargo manifest projection.
//!
//! This module parses Cargo.toml bytes as data. It never invokes Cargo, resolves
//! workspace globs, executes build scripts, expands procedural macros, or
//! promotes manifest declarations to compiler-resolved relationships.

use serde::Serialize;
use toml::Value;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ManifestAnchor {
    pub repository_full_name: String,
    pub revision: String,
    pub tree_oid: String,
    pub manifest_path: String,
    pub blob_oid: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PackageDeclaration {
    pub name: String,
    pub version: Option<String>,
    pub edition: Option<String>,
    pub rust_version: Option<String>,
    pub workspace_inherited_fields: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WorkspaceDeclaration {
    /// Literal member patterns from Cargo.toml; these are not glob-expanded.
    pub members: Vec<String>,
    pub default_members: Vec<String>,
    pub exclude: Vec<String>,
    pub resolver: Option<String>,
    pub shared_dependency_aliases: Vec<String>,
    pub shared_package_field_names: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DependencyDeclaration {
    pub alias: String,
    pub package: Option<String>,
    pub kind: String,
    pub target: Option<String>,
    pub version: Option<String>,
    pub path: Option<String>,
    pub git: Option<String>,
    pub branch: Option<String>,
    pub tag: Option<String>,
    pub rev: Option<String>,
    pub registry: Option<String>,
    pub optional: Option<bool>,
    pub default_features: Option<bool>,
    pub workspace_inherited: bool,
    pub features: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FeatureDeclaration {
    pub name: String,
    /// Cargo feature expressions are preserved, not expanded or interpreted.
    pub members: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TargetDeclaration {
    pub kind: String,
    pub name: Option<String>,
    pub path: Option<String>,
    pub required_features: Vec<String>,
    pub crate_types: Vec<String>,
    pub test: Option<bool>,
    pub doc: Option<bool>,
    pub bench: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BuildScriptDeclaration {
    /// One of: disabled, explicit_path, default_convention_unobserved, not_applicable.
    pub state: String,
    pub declared_path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CargoManifestProjection {
    pub schema_version: String,
    pub authority: String,
    pub source: ManifestAnchor,
    pub manifest_kind: String,
    pub package: Option<PackageDeclaration>,
    pub workspace: Option<WorkspaceDeclaration>,
    pub dependencies: Vec<DependencyDeclaration>,
    pub features: Vec<FeatureDeclaration>,
    pub targets: Vec<TargetDeclaration>,
    pub build_script: BuildScriptDeclaration,
    pub limitations: Vec<String>,
}

fn valid_git_oid(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_anchor(anchor: &ManifestAnchor) -> Result<(), String> {
    for (name, value) in [
        ("repository_full_name", anchor.repository_full_name.as_str()),
        ("manifest_path", anchor.manifest_path.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("source anchor {name} must not be empty"));
        }
    }
    for (name, value) in [
        ("revision", anchor.revision.as_str()),
        ("tree_oid", anchor.tree_oid.as_str()),
        ("blob_oid", anchor.blob_oid.as_str()),
    ] {
        if !valid_git_oid(value) {
            return Err(format!("source anchor {name} must be a 40- or 64-character hexadecimal Git object ID"));
        }
    }
    Ok(())
}

fn table<'a>(value: &'a Value, path: &str) -> Result<&'a toml::Table, String> {
    value.as_table().ok_or_else(|| format!("{path} must be a TOML table"))
}

fn optional_string(value: Option<&Value>, path: &str) -> Result<Option<String>, String> {
    match value {
        None => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(format!("{path} must be a string")),
    }
}

fn string_list(value: Option<&Value>, path: &str) -> Result<Vec<String>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let values = value
        .as_array()
        .ok_or_else(|| format!("{path} must be an array of strings"))?;
    let mut output = Vec::with_capacity(values.len());
    for (index, item) in values.iter().enumerate() {
        let item = item
            .as_str()
            .ok_or_else(|| format!("{path}[{index}] must be a string"))?;
        output.push(item.to_owned());
    }
    output.sort();
    Ok(output)
}

fn optional_bool(value: Option<&Value>, default: bool, path: &str) -> Result<bool, String> {
    match value {
        None => Ok(default),
        Some(Value::Boolean(value)) => Ok(*value),
        Some(_) => Err(format!("{path} must be a boolean")),
    }
}

fn package_string(value: Option<&Value>, path: &str) -> Result<Option<String>, String> {
    match value {
        None => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(Value::Table(table))
            if table.get("workspace").and_then(Value::as_bool) == Some(true) =>
        {
            Ok(None)
        }
        Some(_) => Err(format!("{path} must be a string or a workspace-inherited declaration")),
    }
}

fn parse_package(value: &Value) -> Result<PackageDeclaration, String> {
    let package = table(value, "package")?;
    let name = optional_string(package.get("name"), "package.name")?
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "package.name is required and must be a non-empty string".to_owned())?;

    let mut inherited = Vec::new();
    for key in ["version", "edition", "rust-version"] {
        if package
            .get(key)
            .and_then(Value::as_table)
            .and_then(|table| table.get("workspace"))
            .and_then(Value::as_bool)
            == Some(true)
        {
            inherited.push(key.to_owned());
        }
    }
    inherited.sort();

    Ok(PackageDeclaration {
        name,
        version: package_string(package.get("version"), "package.version")?,
        edition: package_string(package.get("edition"), "package.edition")?,
        rust_version: package_string(package.get("rust-version"), "package.rust-version")?,
        workspace_inherited_fields: inherited,
    })
}

fn parse_workspace(value: &Value) -> Result<WorkspaceDeclaration, String> {
    let workspace = table(value, "workspace")?;
    let members = string_list(workspace.get("members"), "workspace.members")?;
    let default_members = string_list(workspace.get("default-members"), "workspace.default-members")?;
    let exclude = string_list(workspace.get("exclude"), "workspace.exclude")?;
    let resolver = optional_string(workspace.get("resolver"), "workspace.resolver")?;
    let mut shared_dependency_aliases = workspace
        .get("dependencies")
        .map(|value| table(value, "workspace.dependencies"))
        .transpose()?
        .map(|value| value.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    shared_dependency_aliases.sort();
    let mut shared_package_field_names = workspace
        .get("package")
        .map(|value| table(value, "workspace.package"))
        .transpose()?
        .map(|value| value.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    shared_package_field_names.sort();

    Ok(WorkspaceDeclaration {
        members,
        default_members,
        exclude,
        resolver,
        shared_dependency_aliases,
        shared_package_field_names,
    })
}

fn parse_dependency_spec(
    alias: &str,
    spec: &Value,
    kind: &str,
    target: Option<&str>,
) -> Result<DependencyDeclaration, String> {
    if let Some(version) = spec.as_str() {
        return Ok(DependencyDeclaration {
            alias: alias.to_owned(),
            package: None,
            kind: kind.to_owned(),
            target: target.map(str::to_owned),
            version: Some(version.to_owned()),
            path: None,
            git: None,
            branch: None,
            tag: None,
            rev: None,
            registry: None,
            optional: Some(false),
            default_features: Some(true),
            workspace_inherited: false,
            features: Vec::new(),
        });
    }

    let dependency = table(spec, &format!("{kind}.{alias}"))?;
    let get_string = |key: &str| {
        optional_string(dependency.get(key), &format!("{kind}.{alias}.{key}"))
    };
    let optional = optional_bool(
        dependency.get("optional"),
        false,
        &format!("{kind}.{alias}.optional"),
    )?;
    let default_features = optional_bool(
        dependency.get("default-features"),
        true,
        &format!("{kind}.{alias}.default-features"),
    )?;
    let workspace_inherited = optional_bool(
        dependency.get("workspace"),
        false,
        &format!("{kind}.{alias}.workspace"),
    )?;
    let mut features = string_list(
        dependency.get("features"),
        &format!("{kind}.{alias}.features"),
    )?;
    features.sort();

    Ok(DependencyDeclaration {
        alias: alias.to_owned(),
        package: get_string("package")?,
        kind: kind.to_owned(),
        target: target.map(str::to_owned),
        version: get_string("version")?,
        path: get_string("path")?,
        git: get_string("git")?,
        branch: get_string("branch")?,
        tag: get_string("tag")?,
        rev: get_string("rev")?,
        registry: get_string("registry")?,
        optional: if dependency.get("optional").is_some() || !workspace_inherited { Some(optional) } else { None },
        default_features: if dependency.get("default-features").is_some() || !workspace_inherited { Some(default_features) } else { None },
        workspace_inherited,
        features,
    })
}

fn parse_dependency_table(
    value: Option<&Value>,
    kind: &str,
    target: Option<&str>,
    output: &mut Vec<DependencyDeclaration>,
) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    let dependencies = table(value, kind)?;
    for (alias, spec) in dependencies {
        output.push(parse_dependency_spec(alias, spec, kind, target)?);
    }
    Ok(())
}

fn parse_dependencies(root: &toml::Table) -> Result<Vec<DependencyDeclaration>, String> {
    let mut output = Vec::new();
    for (key, kind) in [
        ("dependencies", "normal"),
        ("dev-dependencies", "dev"),
        ("build-dependencies", "build"),
    ] {
        parse_dependency_table(root.get(key), kind, None, &mut output)?;
    }

    if let Some(targets) = root.get("target") {
        let targets = table(targets, "target")?;
        for (target, declarations) in targets {
            let declarations = table(declarations, &format!("target.{target}"))?;
            for (key, kind) in [
                ("dependencies", "normal"),
                ("dev-dependencies", "dev"),
                ("build-dependencies", "build"),
            ] {
                parse_dependency_table(
                    declarations.get(key),
                    kind,
                    Some(target),
                    &mut output,
                )?;
            }
        }
    }
    output.sort_by(|left, right| {
        (&left.target, &left.kind, &left.alias)
            .cmp(&(&right.target, &right.kind, &right.alias))
    });
    Ok(output)
}

fn parse_features(root: &toml::Table) -> Result<Vec<FeatureDeclaration>, String> {
    let Some(value) = root.get("features") else {
        return Ok(Vec::new());
    };
    let features = table(value, "features")?;
    let mut output = Vec::with_capacity(features.len());
    for (name, members) in features {
        output.push(FeatureDeclaration {
            name: name.clone(),
            members: string_list(Some(members), &format!("features.{name}"))?,
        });
    }
    output.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(output)
}

fn parse_target_entry(value: &Value, kind: &str, path: &str) -> Result<TargetDeclaration, String> {
    let target = table(value, path)?;
    Ok(TargetDeclaration {
        kind: kind.to_owned(),
        name: optional_string(target.get("name"), &format!("{path}.name"))?,
        path: optional_string(target.get("path"), &format!("{path}.path"))?,
        required_features: string_list(
            target.get("required-features"),
            &format!("{path}.required-features"),
        )?,
        crate_types: string_list(target.get("crate-type"), &format!("{path}.crate-type"))?,
        test: target.get("test").map(|value| optional_bool(Some(value), true, &format!("{path}.test"))).transpose()?,
        doc: target.get("doc").map(|value| optional_bool(Some(value), true, &format!("{path}.doc"))).transpose()?,
        bench: target.get("bench").map(|value| optional_bool(Some(value), true, &format!("{path}.bench"))).transpose()?,
    })
}

fn parse_targets(root: &toml::Table) -> Result<Vec<TargetDeclaration>, String> {
    let mut output = Vec::new();
    if let Some(lib) = root.get("lib") {
        output.push(parse_target_entry(lib, "lib", "lib")?);
    }
    for (key, kind) in [
        ("bin", "bin"),
        ("example", "example"),
        ("test", "test"),
        ("bench", "bench"),
    ] {
        let Some(entries) = root.get(key) else {
            continue;
        };
        let entries = entries
            .as_array()
            .ok_or_else(|| format!("{key} must be an array of target tables"))?;
        for (index, entry) in entries.iter().enumerate() {
            output.push(parse_target_entry(entry, kind, &format!("{key}[{index}]"))?);
        }
    }
    output.sort_by(|left, right| {
        (&left.kind, &left.name, &left.path).cmp(&(&right.kind, &right.name, &right.path))
    });
    Ok(output)
}

fn parse_build_script(package: Option<&Value>) -> Result<BuildScriptDeclaration, String> {
    let Some(package) = package else {
        return Ok(BuildScriptDeclaration {
            state: "not_applicable".to_owned(),
            declared_path: None,
        });
    };
    let package = table(package, "package")?;
    match package.get("build") {
        None => Ok(BuildScriptDeclaration {
            state: "default_convention_unobserved".to_owned(),
            declared_path: None,
        }),
        Some(Value::Boolean(false)) => Ok(BuildScriptDeclaration {
            state: "disabled".to_owned(),
            declared_path: None,
        }),
        Some(Value::String(path)) if !path.trim().is_empty() => Ok(BuildScriptDeclaration {
            state: "explicit_path".to_owned(),
            declared_path: Some(path.clone()),
        }),
        Some(_) => Err("package.build must be false or a non-empty path string".to_owned()),
    }
}

/// Produce a deterministic source-declared projection of one Cargo.toml.
///
/// The caller supplies exact Git source identity. This function validates the
/// anchor shape but does not authenticate Git objects or inspect other files.
pub fn project_cargo_manifest(
    raw_toml: &str,
    source: ManifestAnchor,
) -> Result<CargoManifestProjection, String> {
    validate_anchor(&source)?;
    let document: Value = toml::from_str(raw_toml)
        .map_err(|error| format!("Cargo.toml parse error: {error}"))?;
    let root = table(&document, "Cargo.toml")?;

    let package = root.get("package").map(parse_package).transpose()?;
    let workspace = root.get("workspace").map(parse_workspace).transpose()?;
    if package.is_none() && workspace.is_none() {
        return Err("Cargo.toml must declare [package], [workspace], or both".to_owned());
    }

    let mut inherited_fields = Vec::new();
    if let Some(package_value) = root.get("package") {
        let package_table = table(package_value, "package")?;
        for key in ["version", "edition", "rust-version"] {
            if package_table
                .get(key)
                .and_then(Value::as_table)
                .and_then(|value| value.get("workspace"))
                .and_then(Value::as_bool)
                == Some(true)
            {
                inherited_fields.push(key.to_owned());
            }
        }
    }
    inherited_fields.sort();

    let dependencies = parse_dependencies(root)?;
    let features = parse_features(root)?;
    let targets = parse_targets(root)?;
    let build_script = parse_build_script(root.get("package"))?;
    let manifest_kind = match (package.is_some(), workspace.is_some()) {
        (true, true) => "package_and_workspace",
        (true, false) => "package",
        (false, true) => "virtual_workspace",
        (false, false) => unreachable!("checked above"),
    };

    let mut limitations = vec![
        "Manifest declarations are not compiler-resolved dependency or API facts.".to_owned(),
        "Workspace member patterns are preserved literally and are not glob-expanded.".to_owned(),
        "Feature expressions are preserved as declarations and are not expanded into configuration worlds.".to_owned(),
        "Implicit lib/bin/example/test/bench target discovery requires a separate source-tree observation.".to_owned(),
        "Build-script state is a manifest declaration only; no build script was opened or executed.".to_owned(),
        "The supplied Git source anchor is recorded but not authenticated by this parser.".to_owned(),
    ];
    if !inherited_fields.is_empty() {
        limitations.push(format!(
            "Package fields inherited from [workspace.package] are declared but not resolved here: {}.",
            inherited_fields.join(", ")
        ));
    }
    limitations.sort();

    Ok(CargoManifestProjection {
        schema_version: "0.1.0".to_owned(),
        authority: "manifest_declared".to_owned(),
        source,
        manifest_kind: manifest_kind.to_owned(),
        package,
        workspace,
        dependencies,
        features,
        targets,
        build_script,
        limitations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const VIRTUAL_WORKSPACE: &str = include_str!("../fixtures/cargo-virtual-workspace.toml");
    const PACKAGE_MANIFEST: &str = include_str!("../fixtures/cargo-package.toml");

    fn anchor() -> ManifestAnchor {
        ManifestAnchor {
            repository_full_name: "example/repo".to_owned(),
            revision: "a".repeat(40),
            tree_oid: "b".repeat(40),
            manifest_path: "Cargo.toml".to_owned(),
            blob_oid: "c".repeat(40),
        }
    }

    #[test]
    fn projects_virtual_workspace_without_expanding_member_globs() {
        let projection = project_cargo_manifest(VIRTUAL_WORKSPACE, anchor()).expect("projection");
        assert_eq!(projection.manifest_kind, "virtual_workspace");
        let workspace = projection.workspace.expect("workspace");
        assert_eq!(workspace.members, vec!["apps/*".to_owned(), "crates/*".to_owned()]);
        assert_eq!(workspace.default_members, vec!["apps/cli".to_owned()]);
        assert_eq!(workspace.exclude, vec!["crates/experimental".to_owned()]);
        assert_eq!(workspace.resolver.as_deref(), Some("3"));
        assert_eq!(workspace.shared_dependency_aliases, vec!["serde".to_owned()]);
    }

    #[test]
    fn projects_package_identity_and_workspace_inheritance_without_resolving_it() {
        let projection = project_cargo_manifest(PACKAGE_MANIFEST, anchor()).expect("projection");
        let package = projection.package.expect("package");
        assert_eq!(package.name, "sample-engine");
        assert_eq!(package.version.as_deref(), Some("0.1.0"));
        assert_eq!(package.edition.as_deref(), Some("2024"));
        assert!(package.workspace_inherited_fields.is_empty());
    }

    #[test]
    fn records_inherited_package_fields_without_resolving_workspace_values() {
        let manifest = PACKAGE_MANIFEST.replace(
            "edition = \"2024\"",
            "edition = { workspace = true }",
        );
        let projection = project_cargo_manifest(&manifest, anchor()).expect("projection");
        let package = projection.package.expect("package");
        assert_eq!(package.edition, None);
        assert_eq!(package.workspace_inherited_fields, vec!["edition".to_owned()]);
        assert!(projection.limitations.iter().any(|limitation| {
            limitation.contains("edition")
        }));
    }

    #[test]
    fn separates_normal_dev_build_and_target_dependencies() {
        let projection = project_cargo_manifest(PACKAGE_MANIFEST, anchor()).expect("projection");
        let serde = projection.dependencies.iter().find(|dependency| dependency.alias == "serde").expect("serde");
        assert_eq!(serde.kind, "normal");
        assert!(serde.workspace_inherited);
        assert_eq!(serde.optional, None);
        assert_eq!(serde.default_features, None);
        assert!(projection.dependencies.iter().any(|dependency| {
            dependency.alias == "tempfile" && dependency.kind == "dev"
        }));
        assert!(projection.dependencies.iter().any(|dependency| {
            dependency.alias == "cc" && dependency.kind == "build"
        }));
        assert!(projection.dependencies.iter().any(|dependency| {
            dependency.alias == "nix"
                && dependency.target.as_deref() == Some("cfg(target_os = \"linux\")")
        }));
    }

    #[test]
    fn preserves_optional_and_default_feature_dependency_declarations() {
        let projection = project_cargo_manifest(PACKAGE_MANIFEST, anchor()).expect("projection");
        let tokio = projection.dependencies.iter().find(|dependency| dependency.alias == "tokio").expect("tokio");
        assert_eq!(tokio.optional, Some(true));
        assert_eq!(tokio.default_features, Some(false));
        assert_eq!(tokio.features, vec!["macros".to_owned(), "rt".to_owned()]);
        let network = projection.features.iter().find(|feature| feature.name == "network").expect("network");
        assert_eq!(network.members, vec!["dep:tokio".to_owned()]);
    }

    #[test]
    fn projects_only_explicit_targets_without_inventing_implicit_targets() {
        let projection = project_cargo_manifest(PACKAGE_MANIFEST, anchor()).expect("projection");
        assert!(projection.targets.iter().any(|target| target.kind == "lib" && target.name.as_deref() == Some("sample_engine")));
        assert!(projection.targets.iter().any(|target| target.kind == "bin" && target.name.as_deref() == Some("sample-cli")));
        assert_eq!(projection.targets.len(), 2);
    }

    #[test]
    fn records_build_script_declaration_without_running_it() {
        let projection = project_cargo_manifest(PACKAGE_MANIFEST, anchor()).expect("projection");
        assert_eq!(projection.build_script.state, "explicit_path");
        assert_eq!(projection.build_script.declared_path.as_deref(), Some("build.rs"));
        assert!(projection.limitations.iter().any(|limitation| limitation.contains("no build script was opened or executed")));
    }

    #[test]
    fn rejects_manifest_without_package_or_workspace() {
        let error = project_cargo_manifest("[dependencies]\nserde = \"1\"\n", anchor()).expect_err("invalid manifest");
        assert!(error.contains("[package]"));
    }

    #[test]
    fn rejects_malformed_toml() {
        let error = project_cargo_manifest("[package\nname = \"broken\"", anchor()).expect_err("parse error");
        assert!(error.contains("Cargo.toml parse error"));
    }

    #[test]
    fn rejects_unpinned_source_identity() {
        let mut invalid = anchor();
        invalid.revision = "main".to_owned();
        let error = project_cargo_manifest(VIRTUAL_WORKSPACE, invalid).expect_err("invalid anchor");
        assert!(error.contains("revision"));
    }
}
