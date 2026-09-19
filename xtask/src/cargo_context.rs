use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

const CONTEXT_SCHEMA: &str = "symthaea.cargo-build-context.v1";
const CONTEXT_HASH_DOMAIN: &[u8] = b"symthaea.cargo-build-context.v1\0";
const INVOCATION_HASH_DOMAIN: &[u8] = b"symthaea.cargo-invocation.v1\0";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CargoBuildContextSpec {
    pub schema: String,
    pub operation: CargoOperation,
    pub manifest_path: String,
    pub package_selection: PackageSelection,
    pub target_selection: TargetSelection,
    pub feature_selection: FeatureSelection,
    pub target_triples: Vec<String>,
    pub profile: String,
    pub toolchain: ToolchainIdentity,
    #[serde(default)]
    pub cargo_config_sha256: Option<String>,
    #[serde(default)]
    pub rustflags_sha256: Option<String>,
    #[serde(default)]
    pub rustdocflags_sha256: Option<String>,
    #[serde(default)]
    pub environment_fingerprints: BTreeMap<String, String>,
    pub raw_argv: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CargoOperation {
    Check,
    Build,
    Test,
    Clippy,
    Doc,
    Run,
    Bench,
    Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PackageSelection {
    #[serde(default)]
    pub workspace: bool,
    #[serde(default)]
    pub packages: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TargetSelection {
    #[serde(default)]
    pub lib: bool,
    #[serde(default)]
    pub bins: Vec<String>,
    #[serde(default)]
    pub examples: Vec<String>,
    #[serde(default)]
    pub tests: Vec<String>,
    #[serde(default)]
    pub benches: Vec<String>,
    #[serde(default)]
    pub all_targets: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FeatureSelection {
    #[serde(default)]
    pub requested: Vec<String>,
    #[serde(default)]
    pub no_default_features: bool,
    #[serde(default)]
    pub all_features: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ToolchainIdentity {
    pub cargo_version: String,
    pub rustc_version: String,
    pub host_triple: String,
    #[serde(default)]
    pub toolchain_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CargoBuildContextDocument {
    pub context_id: String,
    pub invocation_id: String,
    pub context: NormalizedCargoBuildContext,
    pub raw_argv: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct NormalizedCargoBuildContext {
    pub schema: &'static str,
    pub operation: CargoOperation,
    pub manifest_path: String,
    pub package_selection: PackageSelection,
    pub target_selection: TargetSelection,
    pub feature_selection: FeatureSelection,
    pub target_triples: Vec<String>,
    pub profile: String,
    pub toolchain: ToolchainIdentity,
    pub cargo_config_sha256: Option<String>,
    pub rustflags_sha256: Option<String>,
    pub rustdocflags_sha256: Option<String>,
    pub environment_fingerprints: BTreeMap<String, String>,
}

pub fn run(spec_path: &Path, output: Option<PathBuf>) -> anyhow::Result<()> {
    let bytes = fs::read(spec_path)
        .with_context(|| format!("read Cargo context spec {}", spec_path.display()))?;
    let spec: CargoBuildContextSpec = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse Cargo context spec {}", spec_path.display()))?;
    let document = build_document(spec)?;

    let mut rendered = serde_json::to_string_pretty(&document)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create context output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write Cargo context document {}", path.display()))?;
        println!("Cargo context document written to {}", path.display());
    } else {
        print!("{rendered}");
    }
    Ok(())
}

pub fn build_document(
    spec: CargoBuildContextSpec,
) -> anyhow::Result<CargoBuildContextDocument> {
    validate_spec(&spec)?;
    let raw_argv = spec.raw_argv.clone();
    let context = normalize(spec);
    let context_bytes =
        serde_json::to_vec(&context).context("serialize normalized Cargo build context")?;
    let argv_bytes = serde_json::to_vec(&raw_argv).context("serialize raw Cargo argv")?;

    Ok(CargoBuildContextDocument {
        context_id: domain_sha256(CONTEXT_HASH_DOMAIN, &context_bytes),
        invocation_id: domain_sha256(INVOCATION_HASH_DOMAIN, &argv_bytes),
        context,
        raw_argv,
    })
}

fn validate_spec(spec: &CargoBuildContextSpec) -> anyhow::Result<()> {
    if spec.schema != CONTEXT_SCHEMA {
        bail!("unsupported Cargo build-context schema: {}", spec.schema);
    }
    validate_relative_path(&spec.manifest_path)?;
    if spec.profile.trim().is_empty() {
        bail!("Cargo build context profile must not be empty");
    }
    if spec.raw_argv.is_empty() || spec.raw_argv.iter().any(|arg| arg.is_empty()) {
        bail!("raw_argv must contain the exact non-empty Cargo invocation arguments");
    }
    if !spec.package_selection.workspace && !spec.package_selection.exclude.is_empty() {
        bail!("Cargo --exclude semantics require workspace selection");
    }
    if spec.feature_selection.all_features && spec.feature_selection.no_default_features {
        bail!("all_features and no_default_features are contradictory context claims");
    }
    if spec.target_selection.all_targets
        && (spec.target_selection.lib
            || !spec.target_selection.bins.is_empty()
            || !spec.target_selection.examples.is_empty()
            || !spec.target_selection.tests.is_empty()
            || !spec.target_selection.benches.is_empty())
    {
        bail!("all_targets cannot be combined with specific target selectors in context v1");
    }
    if spec.toolchain.cargo_version.trim().is_empty()
        || spec.toolchain.rustc_version.trim().is_empty()
        || spec.toolchain.host_triple.trim().is_empty()
    {
        bail!("toolchain cargo_version, rustc_version, and host_triple are required");
    }
    for (name, digest) in [
        ("cargo_config_sha256", spec.cargo_config_sha256.as_deref()),
        ("rustflags_sha256", spec.rustflags_sha256.as_deref()),
        ("rustdocflags_sha256", spec.rustdocflags_sha256.as_deref()),
    ] {
        if let Some(digest) = digest {
            validate_sha256(name, digest)?;
        }
    }
    for (key, digest) in &spec.environment_fingerprints {
        if key.trim().is_empty() {
            bail!("environment fingerprint keys must not be empty");
        }
        validate_sha256(&format!("environment_fingerprints.{key}"), digest)?;
    }
    Ok(())
}

fn validate_relative_path(value: &str) -> anyhow::Result<()> {
    if value.trim().is_empty() {
        bail!("manifest_path must not be empty");
    }
    let path = Path::new(value);
    if path.is_absolute() {
        bail!("manifest_path must be workspace-relative, not absolute: {value}");
    }
    for component in path.components() {
        match component {
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("manifest_path must not escape the workspace: {value}")
            }
            Component::CurDir | Component::Normal(_) => {}
        }
    }
    Ok(())
}

fn validate_sha256(name: &str, digest: &str) -> anyhow::Result<()> {
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    Ok(())
}

fn normalize(mut spec: CargoBuildContextSpec) -> NormalizedCargoBuildContext {
    canonicalize_set(&mut spec.package_selection.packages);
    canonicalize_set(&mut spec.package_selection.exclude);
    canonicalize_set(&mut spec.target_selection.bins);
    canonicalize_set(&mut spec.target_selection.examples);
    canonicalize_set(&mut spec.target_selection.tests);
    canonicalize_set(&mut spec.target_selection.benches);
    canonicalize_set(&mut spec.feature_selection.requested);
    canonicalize_set(&mut spec.target_triples);

    // With all features requested, spelling additional individual features is
    // semantically redundant. The literal argv remains separately bound by
    // invocation_id, while context_id represents the normalized build context.
    if spec.feature_selection.all_features {
        spec.feature_selection.requested.clear();
    }

    NormalizedCargoBuildContext {
        schema: CONTEXT_SCHEMA,
        operation: spec.operation,
        manifest_path: normalize_relative_path(&spec.manifest_path),
        package_selection: spec.package_selection,
        target_selection: spec.target_selection,
        feature_selection: spec.feature_selection,
        target_triples: spec.target_triples,
        profile: spec.profile,
        toolchain: spec.toolchain,
        cargo_config_sha256: normalize_digest(spec.cargo_config_sha256),
        rustflags_sha256: normalize_digest(spec.rustflags_sha256),
        rustdocflags_sha256: normalize_digest(spec.rustdocflags_sha256),
        environment_fingerprints: spec
            .environment_fingerprints
            .into_iter()
            .map(|(key, value)| (key, value.to_ascii_lowercase()))
            .collect(),
    }
}

fn normalize_relative_path(value: &str) -> String {
    let parts: Vec<&str> = Path::new(value)
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => part.to_str(),
            Component::CurDir => None,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => None,
        })
        .collect();
    parts.join("/")
}

fn normalize_digest(value: Option<String>) -> Option<String> {
    value.map(|digest| digest.to_ascii_lowercase())
}

fn canonicalize_set(values: &mut Vec<String>) {
    values.sort();
    values.dedup();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> CargoBuildContextSpec {
        CargoBuildContextSpec {
            schema: CONTEXT_SCHEMA.into(),
            operation: CargoOperation::Check,
            manifest_path: "Cargo.toml".into(),
            package_selection: PackageSelection {
                workspace: false,
                packages: vec!["symthaea".into()],
                exclude: vec![],
            },
            target_selection: TargetSelection {
                lib: true,
                bins: vec![],
                examples: vec![],
                tests: vec![],
                benches: vec![],
                all_targets: false,
            },
            feature_selection: FeatureSelection {
                requested: vec!["geodesic_synthesis".into(), "code_generation".into()],
                no_default_features: false,
                all_features: false,
            },
            target_triples: vec!["x86_64-unknown-linux-gnu".into()],
            profile: "dev".into(),
            toolchain: ToolchainIdentity {
                cargo_version: "cargo 1.96.0".into(),
                rustc_version: "rustc 1.96.0".into(),
                host_triple: "x86_64-unknown-linux-gnu".into(),
                toolchain_name: Some("1.96.0".into()),
            },
            cargo_config_sha256: Some("AA".repeat(32)),
            rustflags_sha256: None,
            rustdocflags_sha256: None,
            environment_fingerprints: BTreeMap::new(),
            raw_argv: vec![
                "cargo".into(),
                "check".into(),
                "--lib".into(),
                "--features".into(),
                "geodesic_synthesis,code_generation".into(),
            ],
        }
    }

    #[test]
    fn set_order_changes_invocation_but_not_context() {
        let a = build_document(spec()).unwrap();
        let mut b_spec = spec();
        b_spec.package_selection.packages = vec!["symthaea".into(), "symthaea".into()];
        b_spec.feature_selection.requested =
            vec!["code_generation".into(), "geodesic_synthesis".into()];
        b_spec.raw_argv = vec![
            "cargo".into(),
            "check".into(),
            "--features".into(),
            "code_generation,geodesic_synthesis".into(),
            "--lib".into(),
        ];
        let b = build_document(b_spec).unwrap();
        assert_eq!(a.context_id, b.context_id);
        assert_ne!(a.invocation_id, b.invocation_id);
    }

    #[test]
    fn target_profile_and_default_feature_mode_are_identity_significant() {
        let base = build_document(spec()).unwrap();

        let mut target = spec();
        target.target_triples = vec!["wasm32-unknown-unknown".into()];
        assert_ne!(base.context_id, build_document(target).unwrap().context_id);

        let mut profile = spec();
        profile.profile = "release".into();
        assert_ne!(base.context_id, build_document(profile).unwrap().context_id);

        let mut defaults = spec();
        defaults.feature_selection.no_default_features = true;
        assert_ne!(base.context_id, build_document(defaults).unwrap().context_id);
    }

    #[test]
    fn all_features_normalizes_redundant_requested_features() {
        let mut a = spec();
        a.feature_selection.all_features = true;
        a.feature_selection.requested = vec!["code_generation".into()];
        let mut b = spec();
        b.feature_selection.all_features = true;
        b.feature_selection.requested.clear();
        assert_eq!(
            build_document(a).unwrap().context_id,
            build_document(b).unwrap().context_id
        );
    }

    #[test]
    fn contradictory_feature_modes_fail_closed() {
        let mut value = spec();
        value.feature_selection.all_features = true;
        value.feature_selection.no_default_features = true;
        assert!(build_document(value).is_err());
    }

    #[test]
    fn exclude_without_workspace_fails_closed() {
        let mut value = spec();
        value.package_selection.exclude.push("slow-domain".into());
        assert!(build_document(value).is_err());
    }

    #[test]
    fn target_selector_ambiguity_fails_closed() {
        let mut value = spec();
        value.target_selection.all_targets = true;
        assert!(build_document(value).is_err());
    }

    #[test]
    fn digest_case_is_semantically_canonical() {
        let a = build_document(spec()).unwrap();
        let mut b = spec();
        b.cargo_config_sha256 = Some("aa".repeat(32));
        assert_eq!(a.context_id, build_document(b).unwrap().context_id);
    }

    #[test]
    fn invalid_or_plaintext_environment_value_is_rejected() {
        let mut value = spec();
        value
            .environment_fingerprints
            .insert("RUSTFLAGS".into(), "-C target-cpu=native".into());
        assert!(build_document(value).is_err());
    }

    #[test]
    fn manifest_path_must_be_machine_independent_and_workspace_relative() {
        let mut absolute = spec();
        absolute.manifest_path = "/srv/symthaea/Cargo.toml".into();
        assert!(build_document(absolute).is_err());

        let mut traversal = spec();
        traversal.manifest_path = "../Cargo.toml".into();
        assert!(build_document(traversal).is_err());
    }

    #[test]
    fn unknown_spec_fields_are_rejected() {
        let json = r#"{
            "schema":"symthaea.cargo-build-context.v1",
            "operation":"check",
            "manifest_path":"Cargo.toml",
            "package_selection":{"workspace":false,"packages":[],"exclude":[]},
            "target_selection":{"lib":true,"bins":[],"examples":[],"tests":[],"benches":[],"all_targets":false},
            "feature_selection":{"requested":[],"no_default_features":false,"all_features":false},
            "target_triples":[],
            "profile":"dev",
            "toolchain":{"cargo_version":"cargo 1.96.0","rustc_version":"rustc 1.96.0","host_triple":"x86_64-unknown-linux-gnu"},
            "raw_argv":["cargo","check"],
            "future_field":true
        }"#;
        assert!(serde_json::from_str::<CargoBuildContextSpec>(json).is_err());
    }
}
