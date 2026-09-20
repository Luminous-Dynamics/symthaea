use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

use crate::cargo_context::{CargoBuildContextDocument, ToolchainIdentity};
use crate::cargo_context_verify::verify_bytes as verify_context_bytes;
use crate::cargo_runtime_binding::CargoRuntimeBindingReceipt;

const INPUT_SCHEMA: &str = "symthaea.cargo-toolchain-attestation-input.v1";
const RECEIPT_SCHEMA: &str = "symthaea.cargo-toolchain-attestation.v1";
const HASH_DOMAIN: &[u8] = b"symthaea.cargo-toolchain-attestation.v1\0";

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CargoToolchainAttestationSpec {
    pub schema: String,
    pub probe_adapter_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToolProbeAttestation {
    pub role: String,
    pub executable_content_id: String,
    pub transcript_sha256: String,
    pub reported_version: String,
    pub host_triple: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CargoToolchainAttestationReceipt {
    pub toolchain_attestation_id: String,
    pub schema: String,
    pub runtime_binding_id: String,
    pub context_id: String,
    pub invocation_id: String,
    pub probe_adapter_sha256: String,
    pub cargo: ToolProbeAttestation,
    pub rustc: ToolProbeAttestation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedProbe {
    transcript_sha256: String,
    reported_version: String,
    host_triple: String,
}

#[derive(Serialize)]
struct ToolchainAttestationIdentity<'a> {
    schema: &'static str,
    runtime_binding_id: &'a str,
    context_id: &'a str,
    invocation_id: &'a str,
    probe_adapter_sha256: &'a str,
    cargo: &'a ToolProbeAttestation,
    rustc: &'a ToolProbeAttestation,
}

#[derive(Debug, Clone, Copy)]
enum ProbeKind {
    Cargo,
    Rustc,
}

impl ProbeKind {
    fn role(self) -> &'static str {
        match self {
            Self::Cargo => "cargo_executable",
            Self::Rustc => "rustc_executable",
        }
    }

    fn version_prefix(self) -> &'static str {
        match self {
            Self::Cargo => "cargo ",
            Self::Rustc => "rustc ",
        }
    }
}

pub(crate) fn run(
    spec_path: &Path,
    runtime_binding_path: &Path,
    pre_snapshot_path: &Path,
    context_path: &Path,
    adapter_semantics_path: &Path,
    effect_policy_path: &Path,
    cargo_transcript_path: &Path,
    rustc_transcript_path: &Path,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let bytes = fs::read(spec_path)
        .with_context(|| format!("read Cargo toolchain-attestation spec {}", spec_path.display()))?;
    let spec: CargoToolchainAttestationSpec = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse Cargo toolchain-attestation spec {}", spec_path.display()))?;

    let (runtime, context) = load_validated_runtime_and_context(
        runtime_binding_path,
        pre_snapshot_path,
        context_path,
        adapter_semantics_path,
        effect_policy_path,
    )?;
    let cargo_transcript = fs::read(cargo_transcript_path)
        .with_context(|| format!("read Cargo version transcript {}", cargo_transcript_path.display()))?;
    let rustc_transcript = fs::read(rustc_transcript_path)
        .with_context(|| format!("read rustc version transcript {}", rustc_transcript_path.display()))?;

    let receipt = build_receipt(spec, runtime, context, &cargo_transcript, &rustc_transcript)?;
    write_json(&receipt, output)
}

pub(crate) fn run_verify(
    receipt_path: &Path,
    runtime_binding_path: &Path,
    pre_snapshot_path: &Path,
    context_path: &Path,
    adapter_semantics_path: &Path,
    effect_policy_path: &Path,
    cargo_transcript_path: &Path,
    rustc_transcript_path: &Path,
) -> anyhow::Result<()> {
    let bytes = fs::read(receipt_path)
        .with_context(|| format!("read Cargo toolchain-attestation receipt {}", receipt_path.display()))?;
    let stored: CargoToolchainAttestationReceipt = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse Cargo toolchain-attestation receipt {}", receipt_path.display()))?;

    let (runtime, context) = load_validated_runtime_and_context(
        runtime_binding_path,
        pre_snapshot_path,
        context_path,
        adapter_semantics_path,
        effect_policy_path,
    )?;
    let cargo_transcript = fs::read(cargo_transcript_path)
        .with_context(|| format!("read Cargo version transcript {}", cargo_transcript_path.display()))?;
    let rustc_transcript = fs::read(rustc_transcript_path)
        .with_context(|| format!("read rustc version transcript {}", rustc_transcript_path.display()))?;

    verify_stored(stored, runtime, context, &cargo_transcript, &rustc_transcript)?;
    Ok(())
}

fn load_validated_runtime_and_context(
    runtime_binding_path: &Path,
    pre_snapshot_path: &Path,
    context_path: &Path,
    adapter_semantics_path: &Path,
    effect_policy_path: &Path,
) -> anyhow::Result<(CargoRuntimeBindingReceipt, CargoBuildContextDocument)> {
    crate::cargo_runtime_binding::run_verify(
        runtime_binding_path,
        pre_snapshot_path,
        context_path,
        adapter_semantics_path,
        effect_policy_path,
    )?;
    let runtime_bytes = fs::read(runtime_binding_path)
        .with_context(|| format!("read validated Cargo runtime binding {}", runtime_binding_path.display()))?;
    let runtime: CargoRuntimeBindingReceipt = serde_json::from_slice(&runtime_bytes)
        .with_context(|| format!("parse validated Cargo runtime binding {}", runtime_binding_path.display()))?;

    let context_bytes = fs::read(context_path)
        .with_context(|| format!("read Cargo context document {}", context_path.display()))?;
    let context = verify_context_bytes(&context_bytes)?;

    if runtime.context_id != context.context_id || runtime.invocation_id != context.invocation_id {
        bail!("runtime binding context/invocation does not match validated Cargo context document");
    }
    Ok((runtime, context))
}

pub(crate) fn build_receipt(
    mut spec: CargoToolchainAttestationSpec,
    runtime: CargoRuntimeBindingReceipt,
    context: CargoBuildContextDocument,
    cargo_transcript: &[u8],
    rustc_transcript: &[u8],
) -> anyhow::Result<CargoToolchainAttestationReceipt> {
    if spec.schema != INPUT_SCHEMA {
        bail!(
            "unsupported Cargo toolchain-attestation input schema: {}",
            spec.schema
        );
    }
    normalize_digest("probe_adapter_sha256", &mut spec.probe_adapter_sha256)?;
    validate_digest("runtime_binding_id", &runtime.runtime_binding_id)?;

    let context_bytes = serde_json::to_vec(&context)
        .context("serialize Cargo context for toolchain-attestation revalidation")?;
    let context = verify_context_bytes(&context_bytes)?;
    if runtime.context_id != context.context_id || runtime.invocation_id != context.invocation_id {
        bail!("runtime binding does not belong to the supplied validated Cargo context");
    }

    let cargo_content = bound_input_content(&runtime, ProbeKind::Cargo.role())?;
    let rustc_content = bound_input_content(&runtime, ProbeKind::Rustc.role())?;
    let cargo_probe = parse_probe(ProbeKind::Cargo, cargo_transcript)?;
    let rustc_probe = parse_probe(ProbeKind::Rustc, rustc_transcript)?;
    validate_probe_against_context(ProbeKind::Cargo, &cargo_probe, &context.context.toolchain)?;
    validate_probe_against_context(ProbeKind::Rustc, &rustc_probe, &context.context.toolchain)?;

    let cargo = ToolProbeAttestation {
        role: ProbeKind::Cargo.role().into(),
        executable_content_id: cargo_content,
        transcript_sha256: cargo_probe.transcript_sha256,
        reported_version: cargo_probe.reported_version,
        host_triple: cargo_probe.host_triple,
    };
    let rustc = ToolProbeAttestation {
        role: ProbeKind::Rustc.role().into(),
        executable_content_id: rustc_content,
        transcript_sha256: rustc_probe.transcript_sha256,
        reported_version: rustc_probe.reported_version,
        host_triple: rustc_probe.host_triple,
    };

    let identity = ToolchainAttestationIdentity {
        schema: RECEIPT_SCHEMA,
        runtime_binding_id: &runtime.runtime_binding_id,
        context_id: &context.context_id,
        invocation_id: &context.invocation_id,
        probe_adapter_sha256: &spec.probe_adapter_sha256,
        cargo: &cargo,
        rustc: &rustc,
    };
    let bytes = serde_json::to_vec(&identity)
        .context("serialize Cargo toolchain-attestation identity")?;
    let toolchain_attestation_id = domain_sha256(HASH_DOMAIN, &bytes);

    Ok(CargoToolchainAttestationReceipt {
        toolchain_attestation_id,
        schema: RECEIPT_SCHEMA.into(),
        runtime_binding_id: runtime.runtime_binding_id,
        context_id: context.context_id,
        invocation_id: context.invocation_id,
        probe_adapter_sha256: spec.probe_adapter_sha256,
        cargo,
        rustc,
    })
}

pub(crate) fn verify_stored(
    stored: CargoToolchainAttestationReceipt,
    runtime: CargoRuntimeBindingReceipt,
    context: CargoBuildContextDocument,
    cargo_transcript: &[u8],
    rustc_transcript: &[u8],
) -> anyhow::Result<CargoToolchainAttestationReceipt> {
    if stored.schema != RECEIPT_SCHEMA {
        bail!(
            "unsupported Cargo toolchain-attestation receipt schema: {}",
            stored.schema
        );
    }
    validate_digest(
        "toolchain_attestation_id",
        &stored.toolchain_attestation_id,
    )?;
    let spec = CargoToolchainAttestationSpec {
        schema: INPUT_SCHEMA.into(),
        probe_adapter_sha256: stored.probe_adapter_sha256.clone(),
    };
    let rebuilt = build_receipt(spec, runtime, context, cargo_transcript, rustc_transcript)?;
    if rebuilt != stored {
        bail!(
            "stored Cargo toolchain-attestation receipt is not the canonical rebuild for its validated runtime/context/transcripts"
        );
    }
    Ok(rebuilt)
}

fn bound_input_content(runtime: &CargoRuntimeBindingReceipt, role: &str) -> anyhow::Result<String> {
    let mut matches = runtime
        .evidence
        .bound_inputs
        .iter()
        .filter(|input| input.role == role);
    let first = matches
        .next()
        .with_context(|| format!("validated runtime binding lacks required role {role}"))?;
    if matches.next().is_some() {
        bail!("validated runtime binding contains duplicate role {role}");
    }
    validate_digest(&format!("{role}.content_id"), &first.content_id)?;
    Ok(first.content_id.to_ascii_lowercase())
}

fn parse_probe(kind: ProbeKind, bytes: &[u8]) -> anyhow::Result<ParsedProbe> {
    if bytes.is_empty() {
        bail!("{} probe transcript must not be empty", kind.role());
    }
    let text = std::str::from_utf8(bytes)
        .with_context(|| format!("{} probe transcript is not UTF-8", kind.role()))?;
    if text.contains('\0') {
        bail!("{} probe transcript contains NUL", kind.role());
    }

    let mut lines = text.lines().map(|line| line.strip_suffix('\r').unwrap_or(line));
    let version = lines
        .next()
        .context("tool probe transcript is missing version line")?
        .trim();
    if !version.starts_with(kind.version_prefix()) || version.len() <= kind.version_prefix().len() {
        bail!(
            "{} probe first line is not a canonical version line: {version:?}",
            kind.role()
        );
    }

    let hosts: Vec<&str> = lines
        .filter_map(|line| line.strip_prefix("host:"))
        .map(str::trim)
        .collect();
    if hosts.len() != 1 || hosts[0].is_empty() {
        bail!(
            "{} probe transcript must contain exactly one non-empty `host:` line",
            kind.role()
        );
    }

    Ok(ParsedProbe {
        transcript_sha256: sha256(bytes),
        reported_version: version.to_string(),
        host_triple: hosts[0].to_string(),
    })
}

fn validate_probe_against_context(
    kind: ProbeKind,
    probe: &ParsedProbe,
    expected: &ToolchainIdentity,
) -> anyhow::Result<()> {
    let expected_version = match kind {
        ProbeKind::Cargo => &expected.cargo_version,
        ProbeKind::Rustc => &expected.rustc_version,
    };
    if &probe.reported_version != expected_version {
        bail!(
            "{} probe version {:?} does not match validated Cargo context {:?}",
            kind.role(),
            probe.reported_version,
            expected_version
        );
    }
    if probe.host_triple != expected.host_triple {
        bail!(
            "{} probe host {:?} does not match validated Cargo context host {:?}",
            kind.role(),
            probe.host_triple,
            expected.host_triple
        );
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

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lower(&hasher.finalize())
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

fn write_json(
    receipt: &CargoToolchainAttestationReceipt,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let mut rendered = serde_json::to_string_pretty(receipt)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create toolchain-attestation output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write Cargo toolchain-attestation receipt {}", path.display()))?;
    } else {
        print!("{rendered}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_adapter_semantics::{
        BoundInputPolicy, CapturePolicy, CargoAdapterSemanticsSpec, CargoHomePolicy,
        DescendantPolicy, EnvironmentInheritancePolicy, EnvironmentPolicy,
        EphemeralDirectoryPolicy, NetworkPolicy, PlatformFamily, PointOfNoReturnPolicy,
        ProcessTreePolicy, RepositoryAccess, SandboxBackend, StdinPolicy,
        WorkingDirectoryPolicy, build_receipt as build_semantics,
    };
    use crate::cargo_context::{
        CargoBuildContextSpec, CargoOperation, FeatureSelection, PackageSelection,
        TargetSelection, build_document,
    };
    use crate::cargo_runtime_binding::{
        BoundRuntimeInput, CargoHomeEvidence, CargoRuntimeBindingSpec, EphemeralDirectoryEvidence,
        RealizedAdapterConfiguration, RuntimeContextRealization, RuntimeEnvironmentValue,
        RuntimePreparationEvidence, build_receipt as build_runtime,
    };
    use crate::repository_effect_policy::{EffectMode, EffectPolicySpec, validate_and_identify_policy};
    use crate::repository_snapshot_receipt::{
        EntryKind, SnapshotEntry, SnapshotScope, SourceClass, UnknownSurface,
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
                cargo_version: "cargo 1.96.0 (aaaaaaaaa 2026-01-01)".into(),
                rustc_version: "rustc 1.96.0 (bbbbbbbbb 2026-01-01)".into(),
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

    fn source() -> crate::repository_snapshot_receipt::ValidatedSnapshotReceipt {
        let mut source = crate::repository_snapshot_receipt::ValidatedSnapshotReceipt {
            snapshot_id: String::new(),
            schema: "symthaea.repository-source-snapshot.v1".into(),
            git_head: "a".repeat(40),
            git_head_tree: "b".repeat(40),
            git_version: "git version 2.50.0".into(),
            scope: SnapshotScope {
                tracked_worktree: true,
                untracked_non_ignored: true,
                explicit_ignored_inputs: vec![],
                ignored_policy: "git_exclude_standard_plus_explicit_ignored_inputs".into(),
                symlink_policy: "hash_link_target_bytes_do_not_follow_referent".into(),
                submodule_policy: "record_gitlink_index_identity_mark_contents_unknown".into(),
                external_input_policy: "not_captured".into(),
            },
            entries: vec![SnapshotEntry {
                path: "src/lib.rs".into(),
                source_class: SourceClass::Tracked,
                kind: EntryKind::File,
                content_sha256: Some(digest('c')),
                size_bytes: Some(12),
                executable: Some(false),
                index_mode: Some("100644".into()),
                index_blob: Some("d".repeat(40)),
            }],
            unknown_surfaces: vec![UnknownSurface {
                kind: "environment_network_and_external_build_inputs_not_captured".into(),
                subject: None,
            }],
        };
        source.snapshot_id = source.computed_snapshot_id().unwrap();
        source
    }

    fn semantics() -> crate::cargo_adapter_semantics::CargoAdapterSemanticsReceipt {
        build_semantics(CargoAdapterSemanticsSpec {
            schema: "symthaea.cargo-adapter-semantics-input.v1".into(),
            adapter_implementation_sha256: digest('1'),
            platform: PlatformFamily::Linux,
            sandbox_backend: SandboxBackend {
                name: "bubblewrap".into(),
                implementation_sha256: digest('2'),
            },
            repository_access: RepositoryAccess::ReadOnlySource,
            working_directory: WorkingDirectoryPolicy::RepositoryRoot,
            network: NetworkPolicy::Denied,
            environment: EnvironmentPolicy {
                inheritance: EnvironmentInheritancePolicy::ClearThenAllowlist,
                allowed_keys: vec![],
            },
            home: EphemeralDirectoryPolicy::EphemeralEmpty,
            cargo_home: CargoHomePolicy::ReadOnlyPrefetched,
            target_dir: EphemeralDirectoryPolicy::EphemeralEmpty,
            temp_dir: EphemeralDirectoryPolicy::EphemeralEmpty,
            external_inputs: BoundInputPolicy::ReadOnlyBoundClosures,
            descendants: DescendantPolicy::SameSandbox,
            stdin: StdinPolicy::Null,
            stdout: CapturePolicy::ExactBytesNoTruncation,
            stderr: CapturePolicy::ExactBytesNoTruncation,
            wall_clock_timeout_ms: 60_000,
            termination_grace_ms: 5_000,
            process_tree: ProcessTreePolicy::TerminateThenKillEntireSandboxTree,
            point_of_no_return: PointOfNoReturnPolicy::AfterEffectAdmissionBeforeSpawn,
        })
        .unwrap()
    }

    fn dir(byte: char) -> EphemeralDirectoryEvidence {
        EphemeralDirectoryEvidence {
            instance_id: digest(byte),
            empty_at_start_receipt_sha256: digest(char::from_u32(byte as u32 + 1).unwrap()),
            setup_receipt_sha256: digest(char::from_u32(byte as u32 + 2).unwrap()),
        }
    }

    fn runtime() -> CargoRuntimeBindingReceipt {
        let context = context();
        let semantics = semantics();
        let mut policy = EffectPolicySpec {
            schema: "symthaea.repository-effect-policy.v1".into(),
            mode: EffectMode::ReadOnlySource,
            expected_diff_id: None,
        };
        let policy_id = validate_and_identify_policy(&mut policy).unwrap();
        build_runtime(
            CargoRuntimeBindingSpec {
                schema: "symthaea.cargo-runtime-binding-input.v1".into(),
                realized: RealizedAdapterConfiguration {
                    platform: PlatformFamily::Linux,
                    adapter_implementation_sha256: digest('1'),
                    sandbox_backend: SandboxBackend {
                        name: "bubblewrap".into(),
                        implementation_sha256: digest('2'),
                    },
                    repository_access: RepositoryAccess::ReadOnlySource,
                    working_directory: WorkingDirectoryPolicy::RepositoryRoot,
                    network: NetworkPolicy::Denied,
                    environment_inheritance: EnvironmentInheritancePolicy::ClearThenAllowlist,
                    home: EphemeralDirectoryPolicy::EphemeralEmpty,
                    cargo_home: CargoHomePolicy::ReadOnlyPrefetched,
                    target_dir: EphemeralDirectoryPolicy::EphemeralEmpty,
                    temp_dir: EphemeralDirectoryPolicy::EphemeralEmpty,
                    external_inputs: BoundInputPolicy::ReadOnlyBoundClosures,
                    descendants: DescendantPolicy::SameSandbox,
                    stdin: StdinPolicy::Null,
                    stdout: CapturePolicy::ExactBytesNoTruncation,
                    stderr: CapturePolicy::ExactBytesNoTruncation,
                    wall_clock_timeout_ms: 60_000,
                    termination_grace_ms: 5_000,
                    process_tree: ProcessTreePolicy::TerminateThenKillEntireSandboxTree,
                    point_of_no_return: PointOfNoReturnPolicy::AfterEffectAdmissionBeforeSpawn,
                },
                context: RuntimeContextRealization {
                    toolchain: context.context.toolchain.clone(),
                    cargo_config_sha256: None,
                    rustflags_sha256: None,
                    rustdocflags_sha256: None,
                    environment_fingerprints: BTreeMap::new(),
                    configuration_setup_receipt_sha256: digest('3'),
                },
                evidence: RuntimePreparationEvidence {
                    runtime_instance_id: digest('4'),
                    sandbox_setup_receipt_sha256: digest('5'),
                    repository_materialization_id: digest('6'),
                    repository_setup_receipt_sha256: digest('7'),
                    working_directory_subject_id: digest('6'),
                    network_setup_receipt_sha256: digest('8'),
                    environment_clear_receipt_sha256: digest('9'),
                    ambient_environment: BTreeMap::new(),
                    home: dir('a'),
                    cargo_home: CargoHomeEvidence::ReadOnlyPrefetched {
                        content_id: digest('d'),
                        setup_receipt_sha256: digest('e'),
                    },
                    target_dir: dir('f'),
                    temp_dir: dir('i'),
                    bound_inputs: vec![
                        BoundRuntimeInput {
                            role: "cargo_executable".into(),
                            content_id: digest('l'),
                            read_only_setup_receipt_sha256: digest('m'),
                        },
                        BoundRuntimeInput {
                            role: "cargo_home_prefetch".into(),
                            content_id: digest('d'),
                            read_only_setup_receipt_sha256: digest('n'),
                        },
                        BoundRuntimeInput {
                            role: "rustc_executable".into(),
                            content_id: digest('o'),
                            read_only_setup_receipt_sha256: digest('p'),
                        },
                    ],
                    io_setup_receipt_sha256: digest('q'),
                    process_tree_setup_receipt_sha256: digest('r'),
                },
            },
            source(),
            context,
            semantics,
            policy_id,
            policy,
        )
        .unwrap()
    }

    fn cargo_transcript() -> Vec<u8> {
        b"cargo 1.96.0 (aaaaaaaaa 2026-01-01)\nrelease: 1.96.0\nhost: x86_64-unknown-linux-gnu\n".to_vec()
    }

    fn rustc_transcript() -> Vec<u8> {
        b"rustc 1.96.0 (bbbbbbbbb 2026-01-01)\nbinary: rustc\nhost: x86_64-unknown-linux-gnu\nrelease: 1.96.0\n".to_vec()
    }

    fn spec() -> CargoToolchainAttestationSpec {
        CargoToolchainAttestationSpec {
            schema: INPUT_SCHEMA.into(),
            probe_adapter_sha256: digest('s'),
        }
    }

    #[test]
    fn derives_executable_ids_from_runtime_and_parses_exact_transcripts() {
        let runtime = runtime();
        let receipt = build_receipt(
            spec(),
            runtime.clone(),
            context(),
            &cargo_transcript(),
            &rustc_transcript(),
        )
        .unwrap();
        assert_eq!(receipt.cargo.executable_content_id, digest('l'));
        assert_eq!(receipt.rustc.executable_content_id, digest('o'));
        assert_eq!(receipt.runtime_binding_id, runtime.runtime_binding_id);
        assert_eq!(receipt.cargo.transcript_sha256, sha256(&cargo_transcript()));
    }

    #[test]
    fn input_schema_cannot_mint_executable_or_reported_identity() {
        let json = format!(
            "{{\"schema\":\"{INPUT_SCHEMA}\",\"probe_adapter_sha256\":\"{}\",\"cargo_executable_content_id\":\"{}\"}}",
            digest('s'),
            digest('x')
        );
        assert!(serde_json::from_str::<CargoToolchainAttestationSpec>(&json).is_err());
    }

    #[test]
    fn version_or_host_substitution_rejects() {
        let bad_version = b"cargo 1.95.0 (aaaaaaaaa 2026-01-01)\nhost: x86_64-unknown-linux-gnu\n";
        assert!(
            build_receipt(spec(), runtime(), context(), bad_version, &rustc_transcript()).is_err()
        );

        let bad_host = b"rustc 1.96.0 (bbbbbbbbb 2026-01-01)\nhost: aarch64-unknown-linux-gnu\n";
        assert!(
            build_receipt(spec(), runtime(), context(), &cargo_transcript(), bad_host).is_err()
        );
    }

    #[test]
    fn transcript_substitution_changes_or_rejects_attestation() {
        let a = build_receipt(
            spec(),
            runtime(),
            context(),
            &cargo_transcript(),
            &rustc_transcript(),
        )
        .unwrap();
        let mut alternate = cargo_transcript();
        alternate.extend_from_slice(b"libgit2: 1.9.0\n");
        let b = build_receipt(spec(), runtime(), context(), &alternate, &rustc_transcript()).unwrap();
        assert_ne!(a.cargo.transcript_sha256, b.cargo.transcript_sha256);
        assert_ne!(a.toolchain_attestation_id, b.toolchain_attestation_id);
    }

    #[test]
    fn duplicate_or_missing_host_lines_reject() {
        let missing = b"cargo 1.96.0 (aaaaaaaaa 2026-01-01)\nrelease: 1.96.0\n";
        assert!(parse_probe(ProbeKind::Cargo, missing).is_err());
        let duplicate = b"cargo 1.96.0 (aaaaaaaaa 2026-01-01)\nhost: x86_64-unknown-linux-gnu\nhost: x86_64-unknown-linux-gnu\n";
        assert!(parse_probe(ProbeKind::Cargo, duplicate).is_err());
    }

    #[test]
    fn stored_receipt_rebuilds_only_for_exact_transcripts_and_runtime() {
        let stored = build_receipt(
            spec(),
            runtime(),
            context(),
            &cargo_transcript(),
            &rustc_transcript(),
        )
        .unwrap();
        verify_stored(
            stored.clone(),
            runtime(),
            context(),
            &cargo_transcript(),
            &rustc_transcript(),
        )
        .unwrap();

        let mut different_runtime = runtime();
        different_runtime.runtime_binding_id = digest('z');
        assert!(
            verify_stored(
                stored,
                different_runtime,
                context(),
                &cargo_transcript(),
                &rustc_transcript(),
            )
            .is_err()
        );
    }
}
