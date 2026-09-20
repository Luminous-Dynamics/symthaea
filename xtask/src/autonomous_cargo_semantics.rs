use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cargo_adapter_semantics::{
    BoundInputPolicy, CapturePolicy, CargoAdapterSemanticsReceipt, DescendantPolicy,
    EnvironmentInheritancePolicy, EphemeralDirectoryPolicy, NetworkPolicy,
    PointOfNoReturnPolicy, ProcessTreePolicy, RepositoryAccess, StdinPolicy,
    WorkingDirectoryPolicy,
};

const RECEIPT_SCHEMA: &str = "symthaea.autonomous-cargo-semantics-profile.v1";
const PROFILE_NAME: &str = "isolated_staging_no_network_v1";
const HASH_DOMAIN: &[u8] = b"symthaea.autonomous-cargo-semantics-profile.v1\0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AutonomousCargoSemanticsProfileReceipt {
    pub autonomous_semantics_profile_id: String,
    pub schema: String,
    pub profile: String,
    pub adapter_semantics_id: String,
}

#[derive(Serialize)]
struct ProfileIdentity<'a> {
    schema: &'static str,
    profile: &'static str,
    adapter_semantics_id: &'a str,
}

pub(crate) fn build_profile(
    mut semantics: CargoAdapterSemanticsReceipt,
) -> anyhow::Result<AutonomousCargoSemanticsProfileReceipt> {
    semantics.validate()?;
    validate_autonomous_profile(&semantics)?;

    let identity = ProfileIdentity {
        schema: RECEIPT_SCHEMA,
        profile: PROFILE_NAME,
        adapter_semantics_id: &semantics.adapter_semantics_id,
    };
    let bytes = serde_json::to_vec(&identity)
        .context("serialize autonomous Cargo semantics profile identity")?;
    let autonomous_semantics_profile_id = domain_sha256(HASH_DOMAIN, &bytes);

    Ok(AutonomousCargoSemanticsProfileReceipt {
        autonomous_semantics_profile_id,
        schema: RECEIPT_SCHEMA.into(),
        profile: PROFILE_NAME.into(),
        adapter_semantics_id: semantics.adapter_semantics_id,
    })
}

pub(crate) fn verify_stored(
    mut stored: AutonomousCargoSemanticsProfileReceipt,
    semantics: CargoAdapterSemanticsReceipt,
) -> anyhow::Result<AutonomousCargoSemanticsProfileReceipt> {
    if stored.schema != RECEIPT_SCHEMA {
        bail!(
            "unsupported autonomous Cargo semantics profile schema: {}",
            stored.schema
        );
    }
    if stored.profile != PROFILE_NAME {
        bail!("unsupported autonomous Cargo semantics profile: {}", stored.profile);
    }
    normalize_digest(
        "autonomous_semantics_profile_id",
        &mut stored.autonomous_semantics_profile_id,
    )?;
    normalize_digest("adapter_semantics_id", &mut stored.adapter_semantics_id)?;

    let rebuilt = build_profile(semantics)?;
    if rebuilt != stored {
        bail!(
            "stored autonomous Cargo semantics profile is not the canonical rebuild for the validated adapter semantics"
        );
    }
    Ok(rebuilt)
}

fn validate_autonomous_profile(semantics: &CargoAdapterSemanticsReceipt) -> anyhow::Result<()> {
    if semantics.repository_access != RepositoryAccess::IsolatedStagingTree {
        bail!("autonomous Cargo requires isolated_staging_tree repository access");
    }
    if semantics.working_directory != WorkingDirectoryPolicy::StagingRoot {
        bail!("autonomous Cargo requires staging_root as the working directory");
    }
    if semantics.network != NetworkPolicy::Denied {
        bail!("autonomous Cargo requires network denied");
    }
    if semantics.environment.inheritance != EnvironmentInheritancePolicy::ClearThenAllowlist {
        bail!("autonomous Cargo requires clear_then_allowlist environment handling");
    }
    if !semantics.environment.allowed_keys.is_empty() {
        bail!("autonomous Cargo v1 forbids ambient inherited environment keys");
    }
    if semantics.home != EphemeralDirectoryPolicy::EphemeralEmpty
        || semantics.target_dir != EphemeralDirectoryPolicy::EphemeralEmpty
        || semantics.temp_dir != EphemeralDirectoryPolicy::EphemeralEmpty
    {
        bail!("autonomous Cargo requires isolated ephemeral HOME/TARGET/TMP directories");
    }
    if semantics.external_inputs != BoundInputPolicy::ReadOnlyBoundClosures {
        bail!("autonomous Cargo requires read-only bound external inputs");
    }
    if semantics.descendants != DescendantPolicy::SameSandbox {
        bail!("autonomous Cargo requires all descendants to remain in the same sandbox");
    }
    if semantics.stdin != StdinPolicy::Null {
        bail!("autonomous Cargo requires null stdin");
    }
    if semantics.stdout != CapturePolicy::ExactBytesNoTruncation
        || semantics.stderr != CapturePolicy::ExactBytesNoTruncation
    {
        bail!("autonomous Cargo requires exact, untruncated stdout/stderr capture");
    }
    if semantics.process_tree != ProcessTreePolicy::TerminateThenKillEntireSandboxTree {
        bail!("autonomous Cargo requires whole-sandbox process-tree termination semantics");
    }
    if semantics.point_of_no_return != PointOfNoReturnPolicy::AfterEffectAdmissionBeforeSpawn {
        bail!("autonomous Cargo requires effect admission before process spawn");
    }
    Ok(())
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    value.make_ascii_lowercase();
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
        CargoAdapterSemanticsSpec, CargoHomePolicy, EnvironmentPolicy, PlatformFamily,
        SandboxBackend, build_receipt,
    };

    fn digest(seed: char) -> String {
        assert!(seed.is_ascii());
        format!("{:02x}", u32::from(seed)).repeat(32)
    }

    fn strict_spec() -> CargoAdapterSemanticsSpec {
        CargoAdapterSemanticsSpec {
            schema: "symthaea.cargo-adapter-semantics-input.v1".into(),
            adapter_implementation_sha256: digest('a'),
            platform: PlatformFamily::Linux,
            sandbox_backend: SandboxBackend {
                name: "bubblewrap".into(),
                implementation_sha256: digest('b'),
            },
            repository_access: RepositoryAccess::IsolatedStagingTree,
            working_directory: WorkingDirectoryPolicy::StagingRoot,
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
            wall_clock_timeout_ms: 20 * 60 * 1000,
            termination_grace_ms: 5_000,
            process_tree: ProcessTreePolicy::TerminateThenKillEntireSandboxTree,
            point_of_no_return: PointOfNoReturnPolicy::AfterEffectAdmissionBeforeSpawn,
        }
    }

    fn strict_semantics() -> CargoAdapterSemanticsReceipt {
        build_receipt(strict_spec()).unwrap()
    }

    #[test]
    fn strict_isolated_staging_profile_is_eligible() {
        let semantics = strict_semantics();
        let profile = build_profile(semantics.clone()).unwrap();
        assert_eq!(profile.adapter_semantics_id, semantics.adapter_semantics_id);
        assert_eq!(profile.profile, PROFILE_NAME);
    }

    #[test]
    fn direct_worktree_access_is_not_autonomous_eligible() {
        let mut spec = strict_spec();
        spec.repository_access = RepositoryAccess::ReadOnlySource;
        spec.working_directory = WorkingDirectoryPolicy::RepositoryRoot;
        let semantics = build_receipt(spec).unwrap();
        assert!(build_profile(semantics).is_err());
    }

    #[test]
    fn loopback_network_is_not_autonomous_eligible() {
        let mut spec = strict_spec();
        spec.network = NetworkPolicy::LoopbackOnly;
        let semantics = build_receipt(spec).unwrap();
        assert!(build_profile(semantics).is_err());
    }

    #[test]
    fn any_ambient_inherited_environment_key_rejects() {
        let mut spec = strict_spec();
        spec.environment.allowed_keys = vec!["TZ".into()];
        let semantics = build_receipt(spec).unwrap();
        assert!(build_profile(semantics).is_err());
    }

    #[test]
    fn profile_identity_changes_when_admitted_semantics_change() {
        let first = build_profile(strict_semantics()).unwrap();
        let mut spec = strict_spec();
        spec.adapter_implementation_sha256 = digest('c');
        let second = build_profile(build_receipt(spec).unwrap()).unwrap();
        assert_ne!(first.adapter_semantics_id, second.adapter_semantics_id);
        assert_ne!(
            first.autonomous_semantics_profile_id,
            second.autonomous_semantics_profile_id
        );
    }

    #[test]
    fn stored_profile_rebuilds_against_exact_semantics() {
        let semantics = strict_semantics();
        let profile = build_profile(semantics.clone()).unwrap();
        assert_eq!(verify_stored(profile.clone(), semantics).unwrap(), profile);
    }

    #[test]
    fn stored_profile_cannot_be_reused_for_other_semantics() {
        let profile = build_profile(strict_semantics()).unwrap();
        let mut other = strict_spec();
        other.adapter_implementation_sha256 = digest('d');
        assert!(verify_stored(profile, build_receipt(other).unwrap()).is_err());
    }
}
