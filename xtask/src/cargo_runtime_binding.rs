use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::cargo_adapter_semantics::{
    BoundInputPolicy, CapturePolicy, CargoAdapterSemanticsReceipt, CargoHomePolicy,
    DescendantPolicy, EnvironmentInheritancePolicy, EphemeralDirectoryPolicy, NetworkPolicy,
    PlatformFamily, PointOfNoReturnPolicy, ProcessTreePolicy, RepositoryAccess, SandboxBackend,
    StdinPolicy, WorkingDirectoryPolicy,
};
use crate::cargo_context::{CargoBuildContextDocument, ToolchainIdentity};
use crate::cargo_context_verify::verify_bytes as verify_context_bytes;
use crate::repository_effect_policy::{
    EffectMode, EffectPolicySpec, validate_and_identify_policy,
};
use crate::repository_snapshot_receipt::ValidatedSnapshotReceipt;

const INPUT_SCHEMA: &str = "symthaea.cargo-runtime-binding-input.v1";
const RECEIPT_SCHEMA: &str = "symthaea.cargo-runtime-binding.v1";
const HASH_DOMAIN: &[u8] = b"symthaea.cargo-runtime-binding.v1\0";
const SOURCE_UNKNOWN_CLOSED_BY_RUNTIME: &str =
    "environment_network_and_external_build_inputs_not_captured";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum RuntimeEnvironmentValue {
    Absent,
    Present { value_sha256: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct EphemeralDirectoryEvidence {
    pub instance_id: String,
    pub empty_at_start_receipt_sha256: String,
    pub setup_receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum CargoHomeEvidence {
    EphemeralEmpty {
        instance_id: String,
        empty_at_start_receipt_sha256: String,
        setup_receipt_sha256: String,
    },
    ReadOnlyPrefetched {
        content_id: String,
        setup_receipt_sha256: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BoundRuntimeInput {
    pub role: String,
    pub content_id: String,
    pub read_only_setup_receipt_sha256: String,
}

/// The concrete adapter configuration that the trusted host claims it has
/// prepared. Every policy-shaped field is checked for exact equality with the
/// validated portable semantics receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RealizedAdapterConfiguration {
    pub platform: PlatformFamily,
    pub adapter_implementation_sha256: String,
    pub sandbox_backend: SandboxBackend,
    pub repository_access: RepositoryAccess,
    pub working_directory: WorkingDirectoryPolicy,
    pub network: NetworkPolicy,
    pub environment_inheritance: EnvironmentInheritancePolicy,
    pub home: EphemeralDirectoryPolicy,
    pub cargo_home: CargoHomePolicy,
    pub target_dir: EphemeralDirectoryPolicy,
    pub temp_dir: EphemeralDirectoryPolicy,
    pub external_inputs: BoundInputPolicy,
    pub descendants: DescendantPolicy,
    pub stdin: StdinPolicy,
    pub stdout: CapturePolicy,
    pub stderr: CapturePolicy,
    pub wall_clock_timeout_ms: u64,
    pub termination_grace_ms: u64,
    pub process_tree: ProcessTreePolicy,
    pub point_of_no_return: PointOfNoReturnPolicy,
}

/// Execution-specific values already represented semantically by the canonical
/// Cargo build-context document. The runtime binding must reproduce them
/// exactly rather than inventing a second context identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeContextRealization {
    pub toolchain: ToolchainIdentity,
    pub cargo_config_sha256: Option<String>,
    pub rustflags_sha256: Option<String>,
    pub rustdocflags_sha256: Option<String>,
    pub environment_fingerprints: BTreeMap<String, String>,
    pub configuration_setup_receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimePreparationEvidence {
    /// Execution-instance identity. Separate prepared sandboxes must not collapse
    /// to one runtime binding merely because all policy/content inputs match.
    pub runtime_instance_id: String,
    pub sandbox_setup_receipt_sha256: String,
    pub repository_materialization_id: String,
    pub repository_setup_receipt_sha256: String,
    pub working_directory_subject_id: String,
    pub network_setup_receipt_sha256: String,
    pub environment_clear_receipt_sha256: String,
    pub ambient_environment: BTreeMap<String, RuntimeEnvironmentValue>,
    pub home: EphemeralDirectoryEvidence,
    pub cargo_home: CargoHomeEvidence,
    pub target_dir: EphemeralDirectoryEvidence,
    pub temp_dir: EphemeralDirectoryEvidence,
    pub bound_inputs: Vec<BoundRuntimeInput>,
    pub io_setup_receipt_sha256: String,
    pub process_tree_setup_receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CargoRuntimeBindingSpec {
    pub schema: String,
    pub realized: RealizedAdapterConfiguration,
    pub context: RuntimeContextRealization,
    pub evidence: RuntimePreparationEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CargoRuntimeBindingReceipt {
    pub runtime_binding_id: String,
    pub schema: String,
    pub repository_source_before: String,
    pub context_id: String,
    pub invocation_id: String,
    pub adapter_semantics_id: String,
    pub effect_policy_id: String,
    pub realized: RealizedAdapterConfiguration,
    pub context: RuntimeContextRealization,
    pub evidence: RuntimePreparationEvidence,
}

#[derive(Serialize)]
struct RuntimeBindingIdentity<'a> {
    schema: &'static str,
    repository_source_before: &'a str,
    context_id: &'a str,
    invocation_id: &'a str,
    adapter_semantics_id: &'a str,
    effect_policy_id: &'a str,
    realized: &'a RealizedAdapterConfiguration,
    context: &'a RuntimeContextRealization,
    evidence: &'a RuntimePreparationEvidence,
}

pub(crate) fn run(
    spec_path: &Path,
    pre_snapshot_path: &Path,
    context_path: &Path,
    adapter_semantics_path: &Path,
    effect_policy_path: &Path,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let spec_bytes = fs::read(spec_path)
        .with_context(|| format!("read Cargo runtime-binding spec {}", spec_path.display()))?;
    let spec: CargoRuntimeBindingSpec = serde_json::from_slice(&spec_bytes)
        .with_context(|| format!("parse Cargo runtime-binding spec {}", spec_path.display()))?;
    let (source, context, semantics, policy_id, policy) = load_upstream(
        pre_snapshot_path,
        context_path,
        adapter_semantics_path,
        effect_policy_path,
    )?;
    let receipt = build_receipt(spec, source, context, semantics, policy_id, policy)?;
    write_json(&receipt, output)
}

pub(crate) fn run_verify(
    receipt_path: &Path,
    pre_snapshot_path: &Path,
    context_path: &Path,
    adapter_semantics_path: &Path,
    effect_policy_path: &Path,
) -> anyhow::Result<()> {
    let bytes = fs::read(receipt_path)
        .with_context(|| format!("read Cargo runtime-binding receipt {}", receipt_path.display()))?;
    let stored: CargoRuntimeBindingReceipt = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse Cargo runtime-binding receipt {}", receipt_path.display()))?;
    let (source, context, semantics, policy_id, policy) = load_upstream(
        pre_snapshot_path,
        context_path,
        adapter_semantics_path,
        effect_policy_path,
    )?;
    verify_stored(stored, source, context, semantics, policy_id, policy)?;
    Ok(())
}

fn load_upstream(
    pre_snapshot_path: &Path,
    context_path: &Path,
    adapter_semantics_path: &Path,
    effect_policy_path: &Path,
) -> anyhow::Result<(
    ValidatedSnapshotReceipt,
    CargoBuildContextDocument,
    CargoAdapterSemanticsReceipt,
    String,
    EffectPolicySpec,
)> {
    let source = ValidatedSnapshotReceipt::load(pre_snapshot_path)?;
    let context_bytes = fs::read(context_path)
        .with_context(|| format!("read Cargo context document {}", context_path.display()))?;
    let context = verify_context_bytes(&context_bytes)?;
    let semantics = CargoAdapterSemanticsReceipt::load(adapter_semantics_path)?;
    let policy_bytes = fs::read(effect_policy_path)
        .with_context(|| format!("read repository effect policy {}", effect_policy_path.display()))?;
    let mut policy: EffectPolicySpec = serde_json::from_slice(&policy_bytes)
        .with_context(|| format!("parse repository effect policy {}", effect_policy_path.display()))?;
    let policy_id = validate_and_identify_policy(&mut policy)?;
    Ok((source, context, semantics, policy_id, policy))
}

pub(crate) fn build_receipt(
    mut spec: CargoRuntimeBindingSpec,
    mut source: ValidatedSnapshotReceipt,
    context: CargoBuildContextDocument,
    mut semantics: CargoAdapterSemanticsReceipt,
    effect_policy_id: String,
    mut effect_policy: EffectPolicySpec,
) -> anyhow::Result<CargoRuntimeBindingReceipt> {
    if spec.schema != INPUT_SCHEMA {
        bail!("unsupported Cargo runtime-binding input schema: {}", spec.schema);
    }

    // Revalidate every upstream object again at the in-crate constructor
    // boundary so callers cannot mutate a previously validated object and then
    // mint stronger runtime evidence from it.
    source.validate()?;
    let context_bytes =
        serde_json::to_vec(&context).context("serialize Cargo context for runtime revalidation")?;
    let context = verify_context_bytes(&context_bytes)?;
    semantics.validate()?;
    let recomputed_policy_id = validate_and_identify_policy(&mut effect_policy)?;
    if recomputed_policy_id != effect_policy_id {
        bail!(
            "effect policy identity mismatch: supplied {}, recomputed {}",
            effect_policy_id,
            recomputed_policy_id
        );
    }

    validate_source_unknowns(&source)?;
    if effect_policy.mode != EffectMode::ReadOnlySource {
        bail!(
            "Cargo runtime-binding v1 requires read_only_source repository effect policy"
        );
    }

    normalize_spec(&mut spec)?;
    validate_realized_against_semantics(&spec.realized, &semantics)?;
    validate_context_realization(&spec.context, &context)?;
    validate_environment(&spec.evidence.ambient_environment, &semantics, &context)?;
    validate_cargo_home(&spec.evidence.cargo_home, &semantics, &spec.evidence.bound_inputs)?;
    validate_directory_isolation(&spec.evidence)?;
    validate_bound_inputs(&spec.evidence.bound_inputs)?;

    if spec.evidence.working_directory_subject_id != spec.evidence.repository_materialization_id {
        bail!(
            "working-directory subject must equal the prepared repository materialization subject"
        );
    }

    let identity = RuntimeBindingIdentity {
        schema: RECEIPT_SCHEMA,
        repository_source_before: &source.snapshot_id,
        context_id: &context.context_id,
        invocation_id: &context.invocation_id,
        adapter_semantics_id: &semantics.adapter_semantics_id,
        effect_policy_id: &effect_policy_id,
        realized: &spec.realized,
        context: &spec.context,
        evidence: &spec.evidence,
    };
    let canonical = serde_json::to_vec(&identity)
        .context("serialize Cargo runtime-binding identity")?;
    let runtime_binding_id = domain_sha256(HASH_DOMAIN, &canonical);

    Ok(CargoRuntimeBindingReceipt {
        runtime_binding_id,
        schema: RECEIPT_SCHEMA.into(),
        repository_source_before: source.snapshot_id,
        context_id: context.context_id,
        invocation_id: context.invocation_id,
        adapter_semantics_id: semantics.adapter_semantics_id,
        effect_policy_id,
        realized: spec.realized,
        context: spec.context,
        evidence: spec.evidence,
    })
}

fn verify_stored(
    stored: CargoRuntimeBindingReceipt,
    source: ValidatedSnapshotReceipt,
    context: CargoBuildContextDocument,
    semantics: CargoAdapterSemanticsReceipt,
    effect_policy_id: String,
    effect_policy: EffectPolicySpec,
) -> anyhow::Result<CargoRuntimeBindingReceipt> {
    if stored.schema != RECEIPT_SCHEMA {
        bail!("unsupported Cargo runtime-binding receipt schema: {}", stored.schema);
    }
    validate_digest("runtime_binding_id", &stored.runtime_binding_id)?;

    let spec = CargoRuntimeBindingSpec {
        schema: INPUT_SCHEMA.into(),
        realized: stored.realized.clone(),
        context: stored.context.clone(),
        evidence: stored.evidence.clone(),
    };
    let rebuilt = build_receipt(
        spec,
        source,
        context,
        semantics,
        effect_policy_id,
        effect_policy,
    )?;
    if rebuilt != stored {
        bail!(
            "stored Cargo runtime-binding receipt is not the canonical rebuild for its validated upstream subjects"
        );
    }
    Ok(rebuilt)
}

fn validate_source_unknowns(source: &ValidatedSnapshotReceipt) -> anyhow::Result<()> {
    for unknown in &source.unknown_surfaces {
        if unknown.kind != SOURCE_UNKNOWN_CLOSED_BY_RUNTIME || unknown.subject.is_some() {
            bail!(
                "repository source has an unknown surface not closed by runtime-binding v1: {}",
                unknown.kind
            );
        }
    }
    Ok(())
}

fn normalize_spec(spec: &mut CargoRuntimeBindingSpec) -> anyhow::Result<()> {
    normalize_digest(
        "realized.adapter_implementation_sha256",
        &mut spec.realized.adapter_implementation_sha256,
    )?;
    normalize_backend(&mut spec.realized.sandbox_backend)?;
    normalize_optional_digest(
        "context.cargo_config_sha256",
        &mut spec.context.cargo_config_sha256,
    )?;
    normalize_optional_digest(
        "context.rustflags_sha256",
        &mut spec.context.rustflags_sha256,
    )?;
    normalize_optional_digest(
        "context.rustdocflags_sha256",
        &mut spec.context.rustdocflags_sha256,
    )?;
    for (key, value) in &mut spec.context.environment_fingerprints {
        validate_env_key(key)?;
        normalize_digest("context.environment_fingerprints value", value)?;
    }
    normalize_digest(
        "context.configuration_setup_receipt_sha256",
        &mut spec.context.configuration_setup_receipt_sha256,
    )?;

    let evidence = &mut spec.evidence;
    for (name, value) in [
        ("runtime_instance_id", &mut evidence.runtime_instance_id),
        (
            "sandbox_setup_receipt_sha256",
            &mut evidence.sandbox_setup_receipt_sha256,
        ),
        (
            "repository_materialization_id",
            &mut evidence.repository_materialization_id,
        ),
        (
            "repository_setup_receipt_sha256",
            &mut evidence.repository_setup_receipt_sha256,
        ),
        (
            "working_directory_subject_id",
            &mut evidence.working_directory_subject_id,
        ),
        (
            "network_setup_receipt_sha256",
            &mut evidence.network_setup_receipt_sha256,
        ),
        (
            "environment_clear_receipt_sha256",
            &mut evidence.environment_clear_receipt_sha256,
        ),
        ("io_setup_receipt_sha256", &mut evidence.io_setup_receipt_sha256),
        (
            "process_tree_setup_receipt_sha256",
            &mut evidence.process_tree_setup_receipt_sha256,
        ),
    ] {
        normalize_digest(name, value)?;
    }

    for (key, state) in &mut evidence.ambient_environment {
        validate_env_key(key)?;
        if let RuntimeEnvironmentValue::Present { value_sha256 } = state {
            normalize_digest("ambient_environment value", value_sha256)?;
        }
    }
    normalize_ephemeral("home", &mut evidence.home)?;
    normalize_ephemeral("target_dir", &mut evidence.target_dir)?;
    normalize_ephemeral("temp_dir", &mut evidence.temp_dir)?;
    normalize_cargo_home(&mut evidence.cargo_home)?;
    normalize_bound_inputs(&mut evidence.bound_inputs)?;
    Ok(())
}

fn normalize_backend(backend: &mut SandboxBackend) -> anyhow::Result<()> {
    let trimmed = backend.name.trim();
    if trimmed.is_empty() || trimmed.len() > 64 {
        bail!("realized sandbox backend name must contain 1..=64 characters");
    }
    if !trimmed
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!("realized sandbox backend name contains non-portable characters");
    }
    backend.name = trimmed.to_string();
    normalize_digest(
        "realized.sandbox_backend.implementation_sha256",
        &mut backend.implementation_sha256,
    )
}

fn normalize_ephemeral(name: &str, value: &mut EphemeralDirectoryEvidence) -> anyhow::Result<()> {
    normalize_digest(&format!("{name}.instance_id"), &mut value.instance_id)?;
    normalize_digest(
        &format!("{name}.empty_at_start_receipt_sha256"),
        &mut value.empty_at_start_receipt_sha256,
    )?;
    normalize_digest(
        &format!("{name}.setup_receipt_sha256"),
        &mut value.setup_receipt_sha256,
    )
}

fn normalize_cargo_home(value: &mut CargoHomeEvidence) -> anyhow::Result<()> {
    match value {
        CargoHomeEvidence::EphemeralEmpty {
            instance_id,
            empty_at_start_receipt_sha256,
            setup_receipt_sha256,
        } => {
            normalize_digest("cargo_home.instance_id", instance_id)?;
            normalize_digest(
                "cargo_home.empty_at_start_receipt_sha256",
                empty_at_start_receipt_sha256,
            )?;
            normalize_digest("cargo_home.setup_receipt_sha256", setup_receipt_sha256)
        }
        CargoHomeEvidence::ReadOnlyPrefetched {
            content_id,
            setup_receipt_sha256,
        } => {
            normalize_digest("cargo_home.content_id", content_id)?;
            normalize_digest("cargo_home.setup_receipt_sha256", setup_receipt_sha256)
        }
    }
}

fn normalize_bound_inputs(values: &mut Vec<BoundRuntimeInput>) -> anyhow::Result<()> {
    for value in values.iter_mut() {
        let trimmed = value.role.trim();
        if trimmed.is_empty() || trimmed.len() > 64 {
            bail!("bound runtime input role must contain 1..=64 characters");
        }
        if !trimmed
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            bail!("bound runtime input role contains non-portable characters: {trimmed:?}");
        }
        value.role = trimmed.to_string();
        normalize_digest("bound_inputs.content_id", &mut value.content_id)?;
        normalize_digest(
            "bound_inputs.read_only_setup_receipt_sha256",
            &mut value.read_only_setup_receipt_sha256,
        )?;
    }
    values.sort_by(|left, right| {
        left.role
            .cmp(&right.role)
            .then_with(|| left.content_id.cmp(&right.content_id))
    });
    Ok(())
}

fn validate_realized_against_semantics(
    realized: &RealizedAdapterConfiguration,
    semantics: &CargoAdapterSemanticsReceipt,
) -> anyhow::Result<()> {
    if realized.platform != semantics.platform {
        bail!("realized platform does not match admitted adapter semantics");
    }
    if realized.adapter_implementation_sha256 != semantics.adapter_implementation_sha256 {
        bail!("realized adapter implementation does not match admitted semantics");
    }
    if realized.sandbox_backend != semantics.sandbox_backend {
        bail!("realized sandbox backend does not match admitted semantics");
    }
    if realized.repository_access != semantics.repository_access
        || realized.working_directory != semantics.working_directory
    {
        bail!("realized repository/working-directory policy does not match admitted semantics");
    }
    if realized.network != semantics.network {
        bail!("realized network policy does not match admitted semantics");
    }
    if realized.environment_inheritance != semantics.environment.inheritance {
        bail!("realized environment inheritance does not match admitted semantics");
    }
    if realized.home != semantics.home
        || realized.cargo_home != semantics.cargo_home
        || realized.target_dir != semantics.target_dir
        || realized.temp_dir != semantics.temp_dir
    {
        bail!("realized mutable-directory policy does not match admitted semantics");
    }
    if realized.external_inputs != semantics.external_inputs
        || realized.descendants != semantics.descendants
    {
        bail!("realized input/descendant policy does not match admitted semantics");
    }
    if realized.stdin != semantics.stdin
        || realized.stdout != semantics.stdout
        || realized.stderr != semantics.stderr
    {
        bail!("realized I/O policy does not match admitted semantics");
    }
    if realized.wall_clock_timeout_ms != semantics.wall_clock_timeout_ms
        || realized.termination_grace_ms != semantics.termination_grace_ms
    {
        bail!("realized timeout/grace does not match admitted semantics");
    }
    if realized.process_tree != semantics.process_tree
        || realized.point_of_no_return != semantics.point_of_no_return
    {
        bail!("realized process-tree/PONR policy does not match admitted semantics");
    }
    Ok(())
}

fn validate_context_realization(
    realized: &RuntimeContextRealization,
    context: &CargoBuildContextDocument,
) -> anyhow::Result<()> {
    if realized.toolchain != context.context.toolchain {
        bail!("realized toolchain identity does not match validated Cargo context");
    }
    if realized.cargo_config_sha256 != context.context.cargo_config_sha256
        || realized.rustflags_sha256 != context.context.rustflags_sha256
        || realized.rustdocflags_sha256 != context.context.rustdocflags_sha256
        || realized.environment_fingerprints != context.context.environment_fingerprints
    {
        bail!("realized Cargo configuration/environment fingerprints do not match validated context");
    }
    Ok(())
}

fn validate_environment(
    ambient: &BTreeMap<String, RuntimeEnvironmentValue>,
    semantics: &CargoAdapterSemanticsReceipt,
    context: &CargoBuildContextDocument,
) -> anyhow::Result<()> {
    let expected: BTreeSet<&str> = semantics
        .environment
        .allowed_keys
        .iter()
        .map(String::as_str)
        .collect();
    let actual: BTreeSet<&str> = ambient.keys().map(String::as_str).collect();
    if actual != expected {
        bail!(
            "ambient environment keys must exactly equal the admitted semantics allowlist"
        );
    }

    // If a context-semantic variable is also admitted as ambient input, both
    // evidence planes must agree on the same concrete value fingerprint.
    for (key, expected_digest) in &context.context.environment_fingerprints {
        if let Some(state) = ambient.get(key) {
            match state {
                RuntimeEnvironmentValue::Present { value_sha256 }
                    if value_sha256 == expected_digest => {}
                RuntimeEnvironmentValue::Present { .. } => bail!(
                    "ambient environment fingerprint for {key} disagrees with Cargo context"
                ),
                RuntimeEnvironmentValue::Absent => bail!(
                    "Cargo context fingerprints ambient key {key}, but runtime marks it absent"
                ),
            }
        }
    }
    Ok(())
}

fn validate_cargo_home(
    evidence: &CargoHomeEvidence,
    semantics: &CargoAdapterSemanticsReceipt,
    bound_inputs: &[BoundRuntimeInput],
) -> anyhow::Result<()> {
    match (&semantics.cargo_home, evidence) {
        (CargoHomePolicy::EphemeralEmpty, CargoHomeEvidence::EphemeralEmpty { .. }) => {
            if bound_inputs.iter().any(|input| input.role == "cargo_home_prefetch") {
                bail!("ephemeral-empty CARGO_HOME must not bind a cargo_home_prefetch input");
            }
        }
        (
            CargoHomePolicy::ReadOnlyPrefetched,
            CargoHomeEvidence::ReadOnlyPrefetched { content_id, .. },
        ) => {
            let Some(input) = bound_inputs
                .iter()
                .find(|input| input.role == "cargo_home_prefetch")
            else {
                bail!("read-only prefetched CARGO_HOME requires cargo_home_prefetch bound input");
            };
            if &input.content_id != content_id {
                bail!("CARGO_HOME content identity does not match bound cargo_home_prefetch input");
            }
        }
        _ => bail!("CARGO_HOME realization does not match admitted semantics"),
    }
    Ok(())
}

fn validate_directory_isolation(evidence: &RuntimePreparationEvidence) -> anyhow::Result<()> {
    let mut ids = BTreeSet::new();
    for (name, id) in [
        ("repository", evidence.repository_materialization_id.as_str()),
        ("home", evidence.home.instance_id.as_str()),
        ("target_dir", evidence.target_dir.instance_id.as_str()),
        ("temp_dir", evidence.temp_dir.instance_id.as_str()),
    ] {
        if !ids.insert(id) {
            bail!("runtime directory/materialization instance is aliased at {name}");
        }
    }
    if let CargoHomeEvidence::EphemeralEmpty { instance_id, .. } = &evidence.cargo_home {
        if !ids.insert(instance_id.as_str()) {
            bail!("ephemeral CARGO_HOME aliases another runtime directory/materialization");
        }
    }
    Ok(())
}

fn validate_bound_inputs(values: &[BoundRuntimeInput]) -> anyhow::Result<()> {
    let mut roles = BTreeSet::new();
    let mut contents = BTreeSet::new();
    for value in values {
        if !roles.insert(value.role.as_str()) {
            bail!("duplicate bound runtime input role: {}", value.role);
        }
        if !contents.insert(value.content_id.as_str()) {
            bail!("duplicate bound runtime input content identity: {}", value.content_id);
        }
    }
    for required in ["cargo_executable", "rustc_executable"] {
        if !roles.contains(required) {
            bail!("runtime binding requires bound input role {required}");
        }
    }
    Ok(())
}

fn validate_env_key(key: &str) -> anyhow::Result<()> {
    let mut bytes = key.bytes();
    let Some(first) = bytes.next() else {
        bail!("environment key must not be empty");
    };
    if !(first.is_ascii_alphabetic() || first == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        bail!("environment key is not a portable identifier: {key:?}");
    }
    Ok(())
}

fn normalize_optional_digest(name: &str, value: &mut Option<String>) -> anyhow::Result<()> {
    if let Some(value) = value {
        normalize_digest(name, value)?;
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

fn write_json(receipt: &CargoRuntimeBindingReceipt, output: Option<PathBuf>) -> anyhow::Result<()> {
    let mut rendered = serde_json::to_string_pretty(receipt)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create runtime-binding output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write Cargo runtime-binding receipt {}", path.display()))?;
    } else {
        print!("{rendered}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_adapter_semantics::{
        CargoAdapterSemanticsSpec, EnvironmentPolicy, build_receipt as build_semantics,
    };
    use crate::cargo_context::{
        CargoBuildContextSpec, CargoOperation, FeatureSelection, PackageSelection,
        TargetSelection, build_document,
    };
    use crate::repository_snapshot_receipt::{
        EntryKind, SnapshotEntry, SnapshotScope, SourceClass, UnknownSurface,
    };

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn source() -> ValidatedSnapshotReceipt {
        let mut source = ValidatedSnapshotReceipt {
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
                kind: SOURCE_UNKNOWN_CLOSED_BY_RUNTIME.into(),
                subject: None,
            }],
        };
        source.snapshot_id = source.computed_snapshot_id().unwrap();
        source.validate().unwrap();
        source
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
            cargo_config_sha256: Some(digest('4')),
            rustflags_sha256: Some(digest('5')),
            rustdocflags_sha256: None,
            environment_fingerprints: BTreeMap::from([("RUSTFLAGS".into(), digest('5'))]),
            raw_argv: vec![
                "cargo".into(),
                "check".into(),
                "--lib".into(),
                "--features".into(),
                "code_generation".into(),
            ],
        })
        .unwrap()
    }

    fn semantics() -> CargoAdapterSemanticsReceipt {
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
                allowed_keys: vec!["RUST_BACKTRACE".into()],
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
            wall_clock_timeout_ms: 1_200_000,
            termination_grace_ms: 5_000,
            process_tree: ProcessTreePolicy::TerminateThenKillEntireSandboxTree,
            point_of_no_return: PointOfNoReturnPolicy::AfterEffectAdmissionBeforeSpawn,
        })
        .unwrap()
    }

    fn policy() -> (String, EffectPolicySpec) {
        let mut policy = EffectPolicySpec {
            schema: "symthaea.repository-effect-policy.v1".into(),
            mode: EffectMode::ReadOnlySource,
            expected_diff_id: None,
        };
        let id = validate_and_identify_policy(&mut policy).unwrap();
        (id, policy)
    }

    fn dir(byte: char) -> EphemeralDirectoryEvidence {
        EphemeralDirectoryEvidence {
            instance_id: digest(byte),
            empty_at_start_receipt_sha256: digest(char::from_u32(byte as u32 + 1).unwrap()),
            setup_receipt_sha256: digest(char::from_u32(byte as u32 + 2).unwrap()),
        }
    }

    fn spec() -> CargoRuntimeBindingSpec {
        CargoRuntimeBindingSpec {
            schema: INPUT_SCHEMA.into(),
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
                wall_clock_timeout_ms: 1_200_000,
                termination_grace_ms: 5_000,
                process_tree: ProcessTreePolicy::TerminateThenKillEntireSandboxTree,
                point_of_no_return: PointOfNoReturnPolicy::AfterEffectAdmissionBeforeSpawn,
            },
            context: RuntimeContextRealization {
                toolchain: context().context.toolchain,
                cargo_config_sha256: Some(digest('4')),
                rustflags_sha256: Some(digest('5')),
                rustdocflags_sha256: None,
                environment_fingerprints: BTreeMap::from([("RUSTFLAGS".into(), digest('5'))]),
                configuration_setup_receipt_sha256: digest('6'),
            },
            evidence: RuntimePreparationEvidence {
                runtime_instance_id: digest('7'),
                sandbox_setup_receipt_sha256: digest('8'),
                repository_materialization_id: digest('9'),
                repository_setup_receipt_sha256: digest('a'),
                working_directory_subject_id: digest('9'),
                network_setup_receipt_sha256: digest('b'),
                environment_clear_receipt_sha256: digest('c'),
                ambient_environment: BTreeMap::from([(
                    "RUST_BACKTRACE".into(),
                    RuntimeEnvironmentValue::Present {
                        value_sha256: digest('d'),
                    },
                )]),
                home: dir('e'),
                cargo_home: CargoHomeEvidence::ReadOnlyPrefetched {
                    content_id: digest('h'),
                    setup_receipt_sha256: digest('i'),
                },
                target_dir: dir('j'),
                temp_dir: dir('m'),
                bound_inputs: vec![
                    BoundRuntimeInput {
                        role: "rustc_executable".into(),
                        content_id: digest('q'),
                        read_only_setup_receipt_sha256: digest('r'),
                    },
                    BoundRuntimeInput {
                        role: "cargo_home_prefetch".into(),
                        content_id: digest('h'),
                        read_only_setup_receipt_sha256: digest('s'),
                    },
                    BoundRuntimeInput {
                        role: "cargo_executable".into(),
                        content_id: digest('t'),
                        read_only_setup_receipt_sha256: digest('u'),
                    },
                ],
                io_setup_receipt_sha256: digest('v'),
                process_tree_setup_receipt_sha256: digest('w'),
            },
        }
    }

    fn build(spec: CargoRuntimeBindingSpec) -> anyhow::Result<CargoRuntimeBindingReceipt> {
        let (policy_id, policy) = policy();
        build_receipt(spec, source(), context(), semantics(), policy_id, policy)
    }

    #[test]
    fn canonical_runtime_binding_builds_and_canonicalizes_inputs() {
        let receipt = build(spec()).unwrap();
        assert_eq!(receipt.schema, RECEIPT_SCHEMA);
        let roles: Vec<&str> = receipt
            .evidence
            .bound_inputs
            .iter()
            .map(|input| input.role.as_str())
            .collect();
        assert_eq!(roles, vec!["cargo_executable", "cargo_home_prefetch", "rustc_executable"]);
    }

    #[test]
    fn policy_realization_mismatch_fails_closed() {
        let mut bad = spec();
        bad.realized.network = NetworkPolicy::LoopbackOnly;
        assert!(build(bad).is_err());
    }

    #[test]
    fn ambient_environment_must_exactly_cover_allowlist() {
        let mut missing = spec();
        missing.evidence.ambient_environment.clear();
        assert!(build(missing).is_err());

        let mut extra = spec();
        extra.evidence.ambient_environment.insert(
            "TERM".into(),
            RuntimeEnvironmentValue::Absent,
        );
        assert!(build(extra).is_err());
    }

    #[test]
    fn mutated_upstream_context_or_semantics_cannot_mint_binding() {
        let mut mutated_context = context();
        mutated_context.context.profile = "release".into();
        let (policy_id, policy) = policy();
        assert!(
            build_receipt(
                spec(),
                source(),
                mutated_context,
                semantics(),
                policy_id,
                policy,
            )
            .is_err()
        );

        let mut mutated_semantics = semantics();
        mutated_semantics.wall_clock_timeout_ms += 1;
        let (policy_id, policy) = policy();
        assert!(
            build_receipt(
                spec(),
                source(),
                context(),
                mutated_semantics,
                policy_id,
                policy,
            )
            .is_err()
        );
    }

    #[test]
    fn cargo_home_policy_and_prefetch_identity_are_cross_bound() {
        let mut missing_prefetch = spec();
        missing_prefetch
            .evidence
            .bound_inputs
            .retain(|input| input.role != "cargo_home_prefetch");
        assert!(build(missing_prefetch).is_err());

        let mut mismatched = spec();
        if let CargoHomeEvidence::ReadOnlyPrefetched { content_id, .. } =
            &mut mismatched.evidence.cargo_home
        {
            *content_id = digest('x');
        }
        assert!(build(mismatched).is_err());
    }

    #[test]
    fn duplicate_or_missing_executable_inputs_reject() {
        let mut duplicate = spec();
        duplicate.evidence.bound_inputs.push(BoundRuntimeInput {
            role: "cargo_executable".into(),
            content_id: digest('z'),
            read_only_setup_receipt_sha256: digest('y'),
        });
        assert!(build(duplicate).is_err());

        let mut missing = spec();
        missing
            .evidence
            .bound_inputs
            .retain(|input| input.role != "rustc_executable");
        assert!(build(missing).is_err());
    }

    #[test]
    fn runtime_instance_identity_prevents_preparation_collapse() {
        let a = build(spec()).unwrap();
        let mut other = spec();
        other.evidence.runtime_instance_id = digest('0');
        let b = build(other).unwrap();
        assert_ne!(a.runtime_binding_id, b.runtime_binding_id);
    }

    #[test]
    fn exact_transition_effect_policy_is_not_admitted_by_safe_runtime_v1() {
        let mut exact = EffectPolicySpec {
            schema: "symthaea.repository-effect-policy.v1".into(),
            mode: EffectMode::ExactTransition,
            expected_diff_id: Some(digest('3')),
        };
        let policy_id = validate_and_identify_policy(&mut exact).unwrap();
        assert!(
            build_receipt(spec(), source(), context(), semantics(), policy_id, exact).is_err()
        );
    }

    #[test]
    fn source_unknowns_must_be_closed_by_this_runtime_receipt() {
        let mut bad_source = source();
        bad_source.unknown_surfaces = vec![UnknownSurface {
            kind: "submodule_contents_unknown".into(),
            subject: Some("vendor/submodule".into()),
        }];
        bad_source.snapshot_id = bad_source.computed_snapshot_id().unwrap();
        let (policy_id, policy) = policy();
        assert!(
            build_receipt(spec(), bad_source, context(), semantics(), policy_id, policy).is_err()
        );
    }
}
