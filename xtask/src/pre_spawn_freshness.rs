use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cargo_runtime_binding::{
    CargoRuntimeBindingReceipt, RealizedAdapterConfiguration, RuntimeContextRealization,
    RuntimePreparationEvidence,
};
use crate::repository_materialization_receipt::RepositoryMaterializationReceipt;
use crate::tool_executable_attestation::ToolExecutableAttestationReceipt;

const OBSERVATION_SCHEMA: &str = "symthaea.pre-spawn-freshness-observation.v1";
const RECEIPT_SCHEMA: &str = "symthaea.pre-spawn-freshness.v1";
const HASH_DOMAIN: &[u8] = b"symthaea.pre-spawn-freshness.v1\0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FreshnessProjection {
    pub repository_source_before: String,
    pub context_id: String,
    pub invocation_id: String,
    pub adapter_semantics_id: String,
    pub effect_policy_id: String,
    pub realized: RealizedAdapterConfiguration,
    pub context: RuntimeContextRealization,
    pub evidence: RuntimePreparationEvidence,
    pub materialization_receipt_id: String,
    pub tool_attestation_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreSpawnFreshnessObservation {
    pub schema: String,
    pub probe_implementation_sha256: String,
    pub probe_transcript_sha256: String,
    pub observed: FreshnessProjection,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreSpawnFreshnessReceipt {
    pub freshness_receipt_id: String,
    pub schema: String,
    pub runtime_binding_id: String,
    pub materialization_receipt_id: String,
    pub tool_attestation_id: String,
    pub probe_implementation_sha256: String,
    pub probe_transcript_sha256: String,
    pub observed: FreshnessProjection,
}

#[derive(Serialize)]
struct FreshnessIdentity<'a> {
    schema: &'static str,
    runtime_binding_id: &'a str,
    materialization_receipt_id: &'a str,
    tool_attestation_id: &'a str,
    probe_implementation_sha256: &'a str,
    probe_transcript_sha256: &'a str,
    observed: &'a FreshnessProjection,
}

pub(crate) fn build_receipt(
    mut observation: PreSpawnFreshnessObservation,
    runtime: CargoRuntimeBindingReceipt,
    materialization: RepositoryMaterializationReceipt,
    tool_attestation: ToolExecutableAttestationReceipt,
) -> anyhow::Result<PreSpawnFreshnessReceipt> {
    if observation.schema != OBSERVATION_SCHEMA {
        bail!(
            "unsupported pre-spawn freshness observation schema: {}",
            observation.schema
        );
    }

    validate_digest("runtime_binding_id", &runtime.runtime_binding_id)?;
    validate_digest(
        "materialization_receipt_id",
        &materialization.materialization_receipt_id,
    )?;
    validate_digest("tool_attestation_id", &tool_attestation.tool_attestation_id)?;
    normalize_digest(
        "probe_implementation_sha256",
        &mut observation.probe_implementation_sha256,
    )?;
    normalize_digest(
        "probe_transcript_sha256",
        &mut observation.probe_transcript_sha256,
    )?;

    let expected = expected_projection(&runtime, &materialization, &tool_attestation)?;
    if observation.observed != expected {
        bail!(
            "pre-spawn freshness observation does not equal the exact admitted runtime subject"
        );
    }

    let identity = FreshnessIdentity {
        schema: RECEIPT_SCHEMA,
        runtime_binding_id: &runtime.runtime_binding_id,
        materialization_receipt_id: &materialization.materialization_receipt_id,
        tool_attestation_id: &tool_attestation.tool_attestation_id,
        probe_implementation_sha256: &observation.probe_implementation_sha256,
        probe_transcript_sha256: &observation.probe_transcript_sha256,
        observed: &expected,
    };
    let canonical = serde_json::to_vec(&identity)
        .context("serialize pre-spawn freshness identity")?;
    let freshness_receipt_id = domain_sha256(HASH_DOMAIN, &canonical);

    Ok(PreSpawnFreshnessReceipt {
        freshness_receipt_id,
        schema: RECEIPT_SCHEMA.into(),
        runtime_binding_id: runtime.runtime_binding_id,
        materialization_receipt_id: materialization.materialization_receipt_id,
        tool_attestation_id: tool_attestation.tool_attestation_id,
        probe_implementation_sha256: observation.probe_implementation_sha256,
        probe_transcript_sha256: observation.probe_transcript_sha256,
        observed: expected,
    })
}

pub(crate) fn expected_projection(
    runtime: &CargoRuntimeBindingReceipt,
    materialization: &RepositoryMaterializationReceipt,
    tool_attestation: &ToolExecutableAttestationReceipt,
) -> anyhow::Result<FreshnessProjection> {
    validate_cross_links(runtime, materialization, tool_attestation)?;

    Ok(FreshnessProjection {
        repository_source_before: runtime.repository_source_before.clone(),
        context_id: runtime.context_id.clone(),
        invocation_id: runtime.invocation_id.clone(),
        adapter_semantics_id: runtime.adapter_semantics_id.clone(),
        effect_policy_id: runtime.effect_policy_id.clone(),
        realized: runtime.realized.clone(),
        context: runtime.context.clone(),
        evidence: runtime.evidence.clone(),
        materialization_receipt_id: materialization.materialization_receipt_id.clone(),
        tool_attestation_id: tool_attestation.tool_attestation_id.clone(),
    })
}

fn validate_cross_links(
    runtime: &CargoRuntimeBindingReceipt,
    materialization: &RepositoryMaterializationReceipt,
    tool_attestation: &ToolExecutableAttestationReceipt,
) -> anyhow::Result<()> {
    if materialization.source_snapshot_id != runtime.repository_source_before {
        bail!("materialization source subject does not match runtime binding source subject");
    }
    if materialization.runtime_instance_id != runtime.evidence.runtime_instance_id {
        bail!("materialization runtime instance does not match runtime binding instance");
    }
    if materialization.kind != runtime.realized.repository_access {
        bail!("materialization access kind does not match runtime binding repository access");
    }
    if materialization.materialization_id != runtime.evidence.repository_materialization_id {
        bail!("materialization identity does not match runtime binding repository materialization");
    }
    if materialization.working_directory_subject_id != runtime.evidence.working_directory_subject_id {
        bail!("materialization working-directory subject does not match runtime binding");
    }
    if materialization.repository_setup_receipt_sha256
        != runtime.evidence.repository_setup_receipt_sha256
    {
        bail!("materialization setup evidence does not match runtime binding");
    }

    if tool_attestation.context_id != runtime.context_id
        || tool_attestation.invocation_id != runtime.invocation_id
    {
        bail!("tool executable attestation does not target the runtime binding Cargo context");
    }

    let cargo_input = runtime
        .evidence
        .bound_inputs
        .iter()
        .find(|input| input.role == "cargo_executable")
        .context("runtime binding has no cargo_executable bound input")?;
    let rustc_input = runtime
        .evidence
        .bound_inputs
        .iter()
        .find(|input| input.role == "rustc_executable")
        .context("runtime binding has no rustc_executable bound input")?;

    if cargo_input.content_id != tool_attestation.cargo.executable_content_id {
        bail!("attested Cargo executable does not match runtime bound input content identity");
    }
    if rustc_input.content_id != tool_attestation.rustc.executable_content_id {
        bail!("attested rustc executable does not match runtime bound input content identity");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_adapter_semantics::{
        BoundInputPolicy, CapturePolicy, CargoHomePolicy, DescendantPolicy,
        EnvironmentInheritancePolicy, EphemeralDirectoryPolicy, NetworkPolicy, PlatformFamily,
        PointOfNoReturnPolicy, ProcessTreePolicy, RepositoryAccess, SandboxBackend, StdinPolicy,
        WorkingDirectoryPolicy,
    };
    use crate::cargo_runtime_binding::{
        BoundRuntimeInput, CargoHomeEvidence, EphemeralDirectoryEvidence, RuntimeEnvironmentValue,
    };
    use crate::cargo_context::ToolchainIdentity;
    use crate::repository_materialization_receipt::RepositoryMaterializationReceipt;
    use crate::tool_executable_attestation::ExecutableProbeEvidence;
    use std::collections::BTreeMap;

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn dir(id: char) -> EphemeralDirectoryEvidence {
        EphemeralDirectoryEvidence {
            instance_id: digest(id),
            empty_at_start_receipt_sha256: digest('a'),
            setup_receipt_sha256: digest('b'),
        }
    }

    fn runtime() -> CargoRuntimeBindingReceipt {
        CargoRuntimeBindingReceipt {
            runtime_binding_id: digest('1'),
            schema: "symthaea.cargo-runtime-binding.v1".into(),
            repository_source_before: digest('2'),
            context_id: digest('3'),
            invocation_id: digest('4'),
            adapter_semantics_id: digest('5'),
            effect_policy_id: digest('6'),
            realized: RealizedAdapterConfiguration {
                platform: PlatformFamily::Linux,
                adapter_implementation_sha256: digest('7'),
                sandbox_backend: SandboxBackend {
                    name: "bubblewrap".into(),
                    implementation_sha256: digest('8'),
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
                toolchain: ToolchainIdentity {
                    cargo_version: "cargo 1.96.0".into(),
                    rustc_version: "rustc 1.96.0".into(),
                    host_triple: "x86_64-unknown-linux-gnu".into(),
                    toolchain_name: Some("1.96.0".into()),
                },
                cargo_config_sha256: Some(digest('9')),
                rustflags_sha256: None,
                rustdocflags_sha256: None,
                environment_fingerprints: BTreeMap::new(),
                configuration_setup_receipt_sha256: digest('c'),
            },
            evidence: RuntimePreparationEvidence {
                runtime_instance_id: digest('d'),
                sandbox_setup_receipt_sha256: digest('e'),
                repository_materialization_id: digest('f'),
                repository_setup_receipt_sha256: digest('a'),
                working_directory_subject_id: digest('f'),
                network_setup_receipt_sha256: digest('b'),
                environment_clear_receipt_sha256: digest('c'),
                ambient_environment: BTreeMap::new(),
                home: dir('g'),
                cargo_home: CargoHomeEvidence::ReadOnlyPrefetched {
                    content_id: digest('h'),
                    setup_receipt_sha256: digest('i'),
                },
                target_dir: dir('j'),
                temp_dir: dir('k'),
                bound_inputs: vec![
                    BoundRuntimeInput {
                        role: "cargo_executable".into(),
                        content_id: digest('l'),
                        read_only_setup_receipt_sha256: digest('m'),
                    },
                    BoundRuntimeInput {
                        role: "rustc_executable".into(),
                        content_id: digest('n'),
                        read_only_setup_receipt_sha256: digest('o'),
                    },
                ],
                io_setup_receipt_sha256: digest('p'),
                process_tree_setup_receipt_sha256: digest('q'),
            },
        }
    }

    fn materialization() -> RepositoryMaterializationReceipt {
        RepositoryMaterializationReceipt {
            materialization_receipt_id: digest('r'),
            schema: "symthaea.repository-materialization.v1".into(),
            source_snapshot_id: digest('2'),
            runtime_instance_id: digest('d'),
            kind: RepositoryAccess::ReadOnlySource,
            materialization_id: digest('f'),
            working_directory_subject_id: digest('f'),
            repository_setup_receipt_sha256: digest('a'),
            derivation_receipt_sha256: digest('s'),
            staged_content_id: None,
        }
    }

    fn tools() -> ToolExecutableAttestationReceipt {
        ToolExecutableAttestationReceipt {
            tool_attestation_id: digest('t'),
            schema: "symthaea.tool-executable-attestation.v1".into(),
            context_id: digest('3'),
            invocation_id: digest('4'),
            cargo: ExecutableProbeEvidence {
                executable_content_id: digest('l'),
                probe_transcript_sha256: digest('u'),
                probe_implementation_sha256: digest('v'),
                reported_version: "cargo 1.96.0".into(),
                reported_host_triple: "x86_64-unknown-linux-gnu".into(),
            },
            rustc: ExecutableProbeEvidence {
                executable_content_id: digest('n'),
                probe_transcript_sha256: digest('w'),
                probe_implementation_sha256: digest('v'),
                reported_version: "rustc 1.96.0".into(),
                reported_host_triple: "x86_64-unknown-linux-gnu".into(),
            },
        }
    }

    fn observation() -> PreSpawnFreshnessObservation {
        PreSpawnFreshnessObservation {
            schema: OBSERVATION_SCHEMA.into(),
            probe_implementation_sha256: digest('x'),
            probe_transcript_sha256: digest('y'),
            observed: expected_projection(&runtime(), &materialization(), &tools()).unwrap(),
        }
    }

    #[test]
    fn exact_runtime_subject_produces_freshness_receipt() {
        let receipt = build_receipt(observation(), runtime(), materialization(), tools()).unwrap();
        assert_eq!(receipt.runtime_binding_id, digest('1'));
        assert_eq!(receipt.materialization_receipt_id, digest('r'));
        assert_eq!(receipt.tool_attestation_id, digest('t'));
    }

    #[test]
    fn runtime_instance_drift_rejects() {
        let mut obs = observation();
        obs.observed.evidence.runtime_instance_id = digest('0');
        assert!(build_receipt(obs, runtime(), materialization(), tools()).is_err());
    }

    #[test]
    fn ambient_environment_drift_rejects() {
        let mut obs = observation();
        obs.observed.evidence.ambient_environment.insert(
            "TERM".into(),
            RuntimeEnvironmentValue::Present {
                value_sha256: digest('0'),
            },
        );
        assert!(build_receipt(obs, runtime(), materialization(), tools()).is_err());
    }

    #[test]
    fn setup_evidence_drift_rejects() {
        let mut obs = observation();
        obs.observed.evidence.network_setup_receipt_sha256 = digest('0');
        assert!(build_receipt(obs, runtime(), materialization(), tools()).is_err());
    }

    #[test]
    fn materialization_substitution_rejects_before_freshness() {
        let mut substituted = materialization();
        substituted.materialization_id = digest('0');
        assert!(expected_projection(&runtime(), &substituted, &tools()).is_err());
    }

    #[test]
    fn executable_substitution_rejects_before_freshness() {
        let mut substituted = tools();
        substituted.cargo.executable_content_id = digest('0');
        assert!(expected_projection(&runtime(), &materialization(), &substituted).is_err());
    }

    #[test]
    fn probe_transcript_is_freshness_identity_material() {
        let a = build_receipt(observation(), runtime(), materialization(), tools()).unwrap();
        let mut other = observation();
        other.probe_transcript_sha256 = digest('0');
        let b = build_receipt(other, runtime(), materialization(), tools()).unwrap();
        assert_ne!(a.freshness_receipt_id, b.freshness_receipt_id);
    }
}
