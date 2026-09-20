#![allow(dead_code)]

#[path = "../src/repository_snapshot.rs"]
mod repository_snapshot;
#[path = "../src/repository_snapshot_receipt.rs"]
mod repository_snapshot_receipt;
#[path = "../src/git_worktree_state.rs"]
mod git_worktree_state;
#[path = "../src/git_worktree_state_receipt.rs"]
mod git_worktree_state_receipt;

#[cfg(test)]
mod contract_tests {
    use super::*;
    use git_worktree_state::{
        FsmonitorEnvironment, GitIndexFlags, GitWorktreeState, GitWorktreeStatePayload,
        IgnoreEnvironment, SparseCheckoutState,
    };
    use sha2::{Digest, Sha256};

    const HASH_DOMAIN: &[u8] = b"symthaea.git-worktree-state.v1\0";

    fn digest(seed: char) -> String {
        assert!(seed.is_ascii(), "fixture digest seed must be ASCII");
        format!("{:02x}", u32::from(seed)).repeat(32)
    }

    fn domain_sha256(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(HASH_DOMAIN);
        hasher.update(bytes);
        let digest = hasher.finalize();
        let mut out = String::with_capacity(digest.len() * 2);
        for byte in digest {
            use std::fmt::Write as _;
            write!(&mut out, "{byte:02x}").unwrap();
        }
        out
    }

    fn state() -> GitWorktreeState {
        let payload = GitWorktreeStatePayload {
            schema: "symthaea.git-worktree-state.v1",
            repository_source_snapshot_id: digest('a'),
            git_version: "git version 2.50.0".into(),
            index_flags: vec![
                GitIndexFlags {
                    path: "Cargo.toml".into(),
                    skip_worktree: false,
                    assume_unchanged: false,
                    fsmonitor_valid: true,
                },
                GitIndexFlags {
                    path: "src/lib.rs".into(),
                    skip_worktree: true,
                    assume_unchanged: false,
                    fsmonitor_valid: false,
                },
            ],
            sparse_checkout: SparseCheckoutState {
                enabled: true,
                cone_mode_configured: true,
                sparse_index_configured: true,
                specification_sha256: Some(digest('b')),
            },
            ignore_environment: IgnoreEnvironment {
                info_exclude_present: true,
                info_exclude_sha256: Some(digest('c')),
                global_excludes_configured: true,
                global_excludes_present: true,
                global_excludes_sha256: Some(digest('d')),
            },
            fsmonitor_environment: FsmonitorEnvironment {
                configured: true,
                config_value_sha256: Some(digest('e')),
            },
        };
        let state_id = domain_sha256(&serde_json::to_vec(&payload).unwrap());
        GitWorktreeState { state_id, payload }
    }

    fn resign(state: &mut GitWorktreeState) {
        state.state_id = domain_sha256(&serde_json::to_vec(&state.payload).unwrap());
    }

    fn verify(state: &GitWorktreeState) -> anyhow::Result<git_worktree_state_receipt::ValidatedGitWorktreeState> {
        git_worktree_state_receipt::verify_bytes(&serde_json::to_vec(state).unwrap())
    }

    #[test]
    fn exact_sensor_serialization_is_accepted_by_pure_verifier() {
        let sensor = state();
        let verified = verify(&sensor).unwrap();
        assert_eq!(verified.state_id, sensor.state_id);
        assert_eq!(
            verified.repository_source_snapshot_id,
            sensor.payload.repository_source_snapshot_id
        );
        assert_eq!(verified.index_flags.len(), sensor.payload.index_flags.len());
    }

    #[test]
    fn expected_source_is_cross_bound() {
        let verified = verify(&state()).unwrap();
        assert!(
            git_worktree_state_receipt::verify_for_source(verified, &digest('z')).is_err()
        );
    }

    #[test]
    fn index_paths_must_remain_canonical_sorted_and_unique() {
        let mut reordered = state();
        reordered.payload.index_flags.swap(0, 1);
        resign(&mut reordered);
        assert!(verify(&reordered).is_err());

        let mut duplicate = state();
        duplicate.payload.index_flags[1].path = "Cargo.toml".into();
        resign(&mut duplicate);
        assert!(verify(&duplicate).is_err());

        let mut noncanonical = state();
        noncanonical.payload.index_flags[1].path = "src/./lib.rs".into();
        resign(&mut noncanonical);
        assert!(verify(&noncanonical).is_err());
    }

    #[test]
    fn sparse_spec_presence_is_exact() {
        let mut missing = state();
        missing.payload.sparse_checkout.specification_sha256 = None;
        resign(&mut missing);
        assert!(verify(&missing).is_err());

        let mut dormant = state();
        dormant.payload.sparse_checkout.enabled = false;
        resign(&mut dormant);
        assert!(verify(&dormant).is_err());
    }

    #[test]
    fn ignore_environment_presence_and_digests_are_cross_bound() {
        let mut info = state();
        info.payload.ignore_environment.info_exclude_present = false;
        resign(&mut info);
        assert!(verify(&info).is_err());

        let mut global = state();
        global.payload.ignore_environment.global_excludes_configured = false;
        resign(&mut global);
        assert!(verify(&global).is_err());

        let mut missing_digest = state();
        missing_digest
            .payload
            .ignore_environment
            .global_excludes_sha256 = None;
        resign(&mut missing_digest);
        assert!(verify(&missing_digest).is_err());
    }

    #[test]
    fn fsmonitor_configuration_requires_exact_value_fingerprint() {
        let mut missing = state();
        missing.payload.fsmonitor_environment.config_value_sha256 = None;
        resign(&mut missing);
        assert!(verify(&missing).is_err());

        let mut unconfigured = state();
        unconfigured.payload.fsmonitor_environment.configured = false;
        resign(&mut unconfigured);
        assert!(verify(&unconfigured).is_err());
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut value = serde_json::to_value(state()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("invented_authority".into(), serde_json::Value::Bool(true));
        assert!(
            git_worktree_state_receipt::verify_bytes(&serde_json::to_vec(&value).unwrap()).is_err()
        );
    }

    #[test]
    fn identity_substitution_rejects_even_when_payload_is_unchanged() {
        let mut substituted = state();
        substituted.state_id = digest('f');
        assert!(verify(&substituted).is_err());
    }
}
