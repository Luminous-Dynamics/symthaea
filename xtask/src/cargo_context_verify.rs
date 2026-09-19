use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::cargo_context::{
    CargoBuildContextDocument, CargoBuildContextSpec, CargoOperation, FeatureSelection,
    PackageSelection, TargetSelection, ToolchainIdentity, build_document,
};

const CONTEXT_SCHEMA: &str = "symthaea.cargo-build-context.v1";

/// Owned representation of a stored normalized context. #4411's live
/// `NormalizedCargoBuildContext` deliberately carries a `&'static str` schema,
/// so verification uses an owned wire type and then reconstructs the canonical
/// #4411 input instead of introducing a second normalizer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StoredNormalizedCargoBuildContext {
    schema: String,
    operation: CargoOperation,
    manifest_path: String,
    package_selection: PackageSelection,
    target_selection: TargetSelection,
    feature_selection: FeatureSelection,
    target_triples: Vec<String>,
    profile: String,
    toolchain: ToolchainIdentity,
    cargo_config_sha256: Option<String>,
    rustflags_sha256: Option<String>,
    rustdocflags_sha256: Option<String>,
    environment_fingerprints: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StoredCargoBuildContextDocument {
    context_id: String,
    invocation_id: String,
    context: StoredNormalizedCargoBuildContext,
    raw_argv: Vec<String>,
}

pub(crate) fn run(document_path: &Path) -> anyhow::Result<()> {
    let bytes = fs::read(document_path)
        .with_context(|| format!("read Cargo context document {}", document_path.display()))?;
    verify_bytes(&bytes)?;
    Ok(())
}

/// Verify one stored document by reconstructing the original #4411 input and
/// passing it back through `build_document()`. This intentionally does not
/// duplicate context normalization or hashing semantics.
pub(crate) fn verify_bytes(bytes: &[u8]) -> anyhow::Result<CargoBuildContextDocument> {
    let stored: StoredCargoBuildContextDocument =
        serde_json::from_slice(bytes).context("parse stored Cargo build-context document")?;
    verify_stored(stored)
}

fn verify_stored(stored: StoredCargoBuildContextDocument) -> anyhow::Result<CargoBuildContextDocument> {
    if stored.context.schema != CONTEXT_SCHEMA {
        bail!(
            "unsupported stored Cargo build-context schema: {}",
            stored.context.schema
        );
    }

    let spec = CargoBuildContextSpec {
        schema: stored.context.schema.clone(),
        operation: stored.context.operation,
        manifest_path: stored.context.manifest_path.clone(),
        package_selection: stored.context.package_selection.clone(),
        target_selection: stored.context.target_selection.clone(),
        feature_selection: stored.context.feature_selection.clone(),
        target_triples: stored.context.target_triples.clone(),
        profile: stored.context.profile.clone(),
        toolchain: stored.context.toolchain.clone(),
        cargo_config_sha256: stored.context.cargo_config_sha256.clone(),
        rustflags_sha256: stored.context.rustflags_sha256.clone(),
        rustdocflags_sha256: stored.context.rustdocflags_sha256.clone(),
        environment_fingerprints: stored.context.environment_fingerprints.clone(),
        raw_argv: stored.raw_argv.clone(),
    };
    let rebuilt = build_document(spec).context("rebuild stored Cargo build-context document")?;

    if rebuilt.context_id != stored.context_id {
        bail!(
            "Cargo build-context identity mismatch: declared {}, computed {}",
            stored.context_id,
            rebuilt.context_id
        );
    }
    if rebuilt.invocation_id != stored.invocation_id {
        bail!(
            "Cargo invocation identity mismatch: declared {}, computed {}",
            stored.invocation_id,
            rebuilt.invocation_id
        );
    }
    if rebuilt.raw_argv != stored.raw_argv {
        bail!("stored raw Cargo argv is not identical to the canonical rebuilt document");
    }

    // Context identity already binds canonical JSON bytes, but comparing the
    // complete projection as JSON also rejects stored non-canonical orderings or
    // redundant set spellings that happen to reconstruct to the same semantic
    // context. Consumers therefore receive one canonical representation.
    let rebuilt_context =
        serde_json::to_value(&rebuilt.context).context("serialize rebuilt Cargo context")?;
    let stored_context =
        serde_json::to_value(&stored.context).context("serialize stored Cargo context")?;
    if rebuilt_context != stored_context {
        bail!("stored Cargo build-context projection is not canonical");
    }

    Ok(rebuilt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_context::{
        CargoBuildContextSpec, CargoOperation, FeatureSelection, PackageSelection,
        TargetSelection, ToolchainIdentity,
    };

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

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
                requested: vec!["code_generation".into(), "geodesic_synthesis".into()],
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
            cargo_config_sha256: Some(digest('a')),
            rustflags_sha256: None,
            rustdocflags_sha256: None,
            environment_fingerprints: BTreeMap::new(),
            raw_argv: vec![
                "cargo".into(),
                "check".into(),
                "--lib".into(),
                "--features".into(),
                "code_generation,geodesic_synthesis".into(),
            ],
        }
    }

    fn document_bytes() -> Vec<u8> {
        serde_json::to_vec(&build_document(spec()).unwrap()).unwrap()
    }

    #[test]
    fn canonical_stored_document_revalidates() {
        let expected = build_document(spec()).unwrap();
        let verified = verify_bytes(&document_bytes()).unwrap();
        assert_eq!(verified, expected);
    }

    #[test]
    fn normalized_context_tampering_rejects() {
        let mut value: serde_json::Value = serde_json::from_slice(&document_bytes()).unwrap();
        value["context"]["profile"] = serde_json::Value::String("release".into());
        assert!(verify_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn raw_argv_substitution_rejects() {
        let mut value: serde_json::Value = serde_json::from_slice(&document_bytes()).unwrap();
        value["raw_argv"][1] = serde_json::Value::String("build".into());
        assert!(verify_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn visible_id_substitution_rejects() {
        let mut context: serde_json::Value = serde_json::from_slice(&document_bytes()).unwrap();
        context["context_id"] = serde_json::Value::String(digest('8'));
        assert!(verify_bytes(&serde_json::to_vec(&context).unwrap()).is_err());

        let mut invocation: serde_json::Value = serde_json::from_slice(&document_bytes()).unwrap();
        invocation["invocation_id"] = serde_json::Value::String(digest('9'));
        assert!(verify_bytes(&serde_json::to_vec(&invocation).unwrap()).is_err());
    }

    #[test]
    fn unknown_fields_and_schema_substitution_reject() {
        let mut future: serde_json::Value = serde_json::from_slice(&document_bytes()).unwrap();
        future["future_field"] = serde_json::Value::Bool(true);
        assert!(verify_bytes(&serde_json::to_vec(&future).unwrap()).is_err());

        let mut schema: serde_json::Value = serde_json::from_slice(&document_bytes()).unwrap();
        schema["context"]["schema"] = serde_json::Value::String("future.context.v2".into());
        assert!(verify_bytes(&serde_json::to_vec(&schema).unwrap()).is_err());
    }

    #[test]
    fn noncanonical_set_projection_rejects_even_if_semantics_rebuild() {
        let mut value: serde_json::Value = serde_json::from_slice(&document_bytes()).unwrap();
        value["context"]["feature_selection"]["requested"] = serde_json::json!([
            "geodesic_synthesis",
            "code_generation"
        ]);
        assert!(verify_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
