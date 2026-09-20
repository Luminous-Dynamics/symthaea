use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

use crate::cargo_context::CargoBuildContextDocument;
use crate::cargo_context_verify::verify_bytes as verify_context_bytes;

const INPUT_SCHEMA: &str = "symthaea.tool-executable-attestation-input.v1";
const RECEIPT_SCHEMA: &str = "symthaea.tool-executable-attestation.v1";
const HASH_DOMAIN: &[u8] = b"symthaea.tool-executable-attestation.v1\0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExecutableProbeEvidence {
    pub executable_content_id: String,
    pub probe_transcript_sha256: String,
    pub probe_implementation_sha256: String,
    pub reported_version: String,
    pub reported_host_triple: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToolExecutableAttestationSpec {
    pub schema: String,
    pub cargo: ExecutableProbeEvidence,
    pub rustc: ExecutableProbeEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToolExecutableAttestationReceipt {
    pub tool_attestation_id: String,
    pub schema: String,
    pub context_id: String,
    pub invocation_id: String,
    pub cargo: ExecutableProbeEvidence,
    pub rustc: ExecutableProbeEvidence,
}

#[derive(Serialize)]
struct ToolAttestationIdentity<'a> {
    schema: &'static str,
    context_id: &'a str,
    invocation_id: &'a str,
    cargo: &'a ExecutableProbeEvidence,
    rustc: &'a ExecutableProbeEvidence,
}

pub(crate) fn run(
    spec_path: &Path,
    context_path: &Path,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let spec_bytes = fs::read(spec_path)
        .with_context(|| format!("read tool attestation spec {}", spec_path.display()))?;
    let spec: ToolExecutableAttestationSpec = serde_json::from_slice(&spec_bytes)
        .with_context(|| format!("parse tool attestation spec {}", spec_path.display()))?;
    let context_bytes = fs::read(context_path)
        .with_context(|| format!("read Cargo context document {}", context_path.display()))?;
    let context = verify_context_bytes(&context_bytes)?;
    let receipt = build_receipt(spec, context)?;
    write_json(&receipt, output)
}

pub(crate) fn run_verify(
    receipt_path: &Path,
    context_path: &Path,
) -> anyhow::Result<()> {
    let bytes = fs::read(receipt_path)
        .with_context(|| format!("read tool attestation receipt {}", receipt_path.display()))?;
    let stored: ToolExecutableAttestationReceipt = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse tool attestation receipt {}", receipt_path.display()))?;
    let context_bytes = fs::read(context_path)
        .with_context(|| format!("read Cargo context document {}", context_path.display()))?;
    let context = verify_context_bytes(&context_bytes)?;
    verify_stored(stored, context)?;
    Ok(())
}

pub(crate) fn build_receipt(
    mut spec: ToolExecutableAttestationSpec,
    context: CargoBuildContextDocument,
) -> anyhow::Result<ToolExecutableAttestationReceipt> {
    if spec.schema != INPUT_SCHEMA {
        bail!("unsupported tool attestation input schema: {}", spec.schema);
    }

    let context_bytes =
        serde_json::to_vec(&context).context("serialize Cargo context for tool attestation")?;
    let context = verify_context_bytes(&context_bytes)?;

    normalize_probe("cargo", &mut spec.cargo)?;
    normalize_probe("rustc", &mut spec.rustc)?;
    validate_distinct_executables(&spec)?;
    validate_against_context(&spec, &context)?;

    let identity = ToolAttestationIdentity {
        schema: RECEIPT_SCHEMA,
        context_id: &context.context_id,
        invocation_id: &context.invocation_id,
        cargo: &spec.cargo,
        rustc: &spec.rustc,
    };
    let canonical = serde_json::to_vec(&identity)
        .context("serialize tool executable attestation identity")?;
    let tool_attestation_id = domain_sha256(HASH_DOMAIN, &canonical);

    Ok(ToolExecutableAttestationReceipt {
        tool_attestation_id,
        schema: RECEIPT_SCHEMA.into(),
        context_id: context.context_id,
        invocation_id: context.invocation_id,
        cargo: spec.cargo,
        rustc: spec.rustc,
    })
}

pub(crate) fn verify_stored(
    stored: ToolExecutableAttestationReceipt,
    context: CargoBuildContextDocument,
) -> anyhow::Result<ToolExecutableAttestationReceipt> {
    if stored.schema != RECEIPT_SCHEMA {
        bail!("unsupported tool attestation receipt schema: {}", stored.schema);
    }
    validate_digest("tool_attestation_id", &stored.tool_attestation_id)?;

    let spec = ToolExecutableAttestationSpec {
        schema: INPUT_SCHEMA.into(),
        cargo: stored.cargo.clone(),
        rustc: stored.rustc.clone(),
    };
    let rebuilt = build_receipt(spec, context)?;
    if rebuilt != stored {
        bail!(
            "stored tool executable attestation is not the canonical rebuild for the validated Cargo context"
        );
    }
    Ok(rebuilt)
}

fn normalize_probe(name: &str, probe: &mut ExecutableProbeEvidence) -> anyhow::Result<()> {
    normalize_digest(
        &format!("{name}.executable_content_id"),
        &mut probe.executable_content_id,
    )?;
    normalize_digest(
        &format!("{name}.probe_transcript_sha256"),
        &mut probe.probe_transcript_sha256,
    )?;
    normalize_digest(
        &format!("{name}.probe_implementation_sha256"),
        &mut probe.probe_implementation_sha256,
    )?;
    validate_canonical_text(&format!("{name}.reported_version"), &probe.reported_version)?;
    validate_canonical_text(
        &format!("{name}.reported_host_triple"),
        &probe.reported_host_triple,
    )?;
    validate_host_triple(&format!("{name}.reported_host_triple"), &probe.reported_host_triple)?;
    Ok(())
}

fn validate_distinct_executables(spec: &ToolExecutableAttestationSpec) -> anyhow::Result<()> {
    if spec.cargo.executable_content_id == spec.rustc.executable_content_id {
        bail!("Cargo and rustc executable content identities must be distinct");
    }
    Ok(())
}

fn validate_against_context(
    spec: &ToolExecutableAttestationSpec,
    context: &CargoBuildContextDocument,
) -> anyhow::Result<()> {
    let toolchain = &context.context.toolchain;
    if spec.cargo.reported_version != toolchain.cargo_version {
        bail!(
            "attested Cargo version {:?} does not match validated context {:?}",
            spec.cargo.reported_version,
            toolchain.cargo_version
        );
    }
    if spec.rustc.reported_version != toolchain.rustc_version {
        bail!(
            "attested rustc version {:?} does not match validated context {:?}",
            spec.rustc.reported_version,
            toolchain.rustc_version
        );
    }
    if spec.cargo.reported_host_triple != toolchain.host_triple {
        bail!("attested Cargo host triple does not match validated context");
    }
    if spec.rustc.reported_host_triple != toolchain.host_triple {
        bail!("attested rustc host triple does not match validated context");
    }
    Ok(())
}

fn validate_canonical_text(name: &str, value: &str) -> anyhow::Result<()> {
    if value.is_empty() || value.trim() != value {
        bail!("{name} must be non-empty canonical text without surrounding whitespace");
    }
    if value.bytes().any(|byte| matches!(byte, b'\r' | b'\n' | 0)) {
        bail!("{name} must not contain control-line separators or NUL");
    }
    if value.len() > 256 {
        bail!("{name} exceeds the 256-byte v1 limit");
    }
    Ok(())
}

fn validate_host_triple(name: &str, value: &str) -> anyhow::Result<()> {
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!("{name} contains non-portable host-triple characters");
    }
    Ok(())
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    validate_digest(name, value)?;
    value.make_ascii_lowercase();
    Ok(())
}

fn validate_digest(name: &str, value: &str) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn write_json(
    receipt: &ToolExecutableAttestationReceipt,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let mut rendered = serde_json::to_string_pretty(receipt)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create tool-attestation output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write tool executable attestation {}", path.display()))?;
    } else {
        print!("{rendered}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_context::{
        CargoBuildContextSpec, CargoOperation, FeatureSelection, PackageSelection,
        TargetSelection, ToolchainIdentity, build_document,
    };
    use std::collections::BTreeMap;

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn context() -> CargoBuildContextDocument {
        build_document(CargoBuildContextSpec {
            schema: "symthaea.cargo-build-context.v1".into(),
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
                requested: vec!["code_generation".into()],
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
            cargo_config_sha256: None,
            rustflags_sha256: None,
            rustdocflags_sha256: None,
            environment_fingerprints: BTreeMap::new(),
            raw_argv: vec!["cargo".into(), "check".into(), "--lib".into()],
        })
        .unwrap()
    }

    fn spec() -> ToolExecutableAttestationSpec {
        ToolExecutableAttestationSpec {
            schema: INPUT_SCHEMA.into(),
            cargo: ExecutableProbeEvidence {
                executable_content_id: digest('1'),
                probe_transcript_sha256: digest('2'),
                probe_implementation_sha256: digest('3'),
                reported_version: "cargo 1.96.0".into(),
                reported_host_triple: "x86_64-unknown-linux-gnu".into(),
            },
            rustc: ExecutableProbeEvidence {
                executable_content_id: digest('4'),
                probe_transcript_sha256: digest('5'),
                probe_implementation_sha256: digest('3'),
                reported_version: "rustc 1.96.0".into(),
                reported_host_triple: "x86_64-unknown-linux-gnu".into(),
            },
        }
    }

    #[test]
    fn attestation_binds_validated_context_identity() {
        let expected = context();
        let receipt = build_receipt(spec(), expected.clone()).unwrap();
        assert_eq!(receipt.context_id, expected.context_id);
        assert_eq!(receipt.invocation_id, expected.invocation_id);
    }

    #[test]
    fn binary_identity_is_attested_separately_from_version_text() {
        let a = build_receipt(spec(), context()).unwrap();
        let mut changed = spec();
        changed.cargo.executable_content_id = digest('9');
        let b = build_receipt(changed, context()).unwrap();
        assert_eq!(a.cargo.reported_version, b.cargo.reported_version);
        assert_ne!(a.tool_attestation_id, b.tool_attestation_id);
    }

    #[test]
    fn cargo_or_rustc_version_substitution_rejects() {
        let mut bad_cargo = spec();
        bad_cargo.cargo.reported_version = "cargo 1.95.0".into();
        assert!(build_receipt(bad_cargo, context()).is_err());

        let mut bad_rustc = spec();
        bad_rustc.rustc.reported_version = "rustc 1.95.0".into();
        assert!(build_receipt(bad_rustc, context()).is_err());
    }

    #[test]
    fn host_substitution_rejects() {
        let mut bad = spec();
        bad.rustc.reported_host_triple = "aarch64-unknown-linux-gnu".into();
        assert!(build_receipt(bad, context()).is_err());
    }

    #[test]
    fn executable_roles_cannot_collapse_to_one_content_subject() {
        let mut bad = spec();
        bad.rustc.executable_content_id = bad.cargo.executable_content_id.clone();
        assert!(build_receipt(bad, context()).is_err());
    }

    #[test]
    fn transcript_and_probe_identity_are_attestation_material() {
        let a = build_receipt(spec(), context()).unwrap();
        let mut changed = spec();
        changed.rustc.probe_transcript_sha256 = digest('8');
        let b = build_receipt(changed, context()).unwrap();
        assert_ne!(a.tool_attestation_id, b.tool_attestation_id);
    }

    #[test]
    fn stored_receipt_cannot_be_rebound_to_another_context() {
        let stored = build_receipt(spec(), context()).unwrap();
        let mut other = context();
        other.context.toolchain.rustc_version = "rustc 1.97.0".into();
        assert!(verify_stored(stored, other).is_err());
    }
}
