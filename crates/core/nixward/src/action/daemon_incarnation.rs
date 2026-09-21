// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Non-rehydratable daemon-process incarnation identity for local approval binding.
//!
//! A local interactive approval must not silently survive a daemon restart. This
//! module supplies a live process-incarnation object generated from OS randomness.
//! The live object deliberately implements neither Serialize/Deserialize nor Clone,
//! and there is no constructor from a persisted incarnation reference.
//!
//! The live daemon context also owns request-nonce generation. Runtime callers
//! should not provide either the daemon incarnation string or the request nonce.
//! Persisted references/nonces are evidence inputs, not live authority.

use super::authorization::NixActionIntentV1;
use super::local_approval::{LocalApprovalErrorV1, PendingNixApprovalRequestV1};
use super::temporal::UnixMillisV1;
use blake3::Hasher;
use thiserror::Error;

const DAEMON_INCARNATION_DOMAIN: &[u8] = b"nixward-daemon-incarnation-v1";

/// Live identity of one daemon process incarnation.
///
/// Generate this once during daemon startup and retain the object for the life of
/// that daemon. A restarted daemon must generate a new instance rather than
/// restoring this object from disk.
pub struct LiveDaemonIncarnationV1 {
    nonce: [u8; 32],
}

impl LiveDaemonIncarnationV1 {
    /// Generate one live daemon incarnation from the operating system RNG.
    pub fn generate() -> Result<Self, DaemonApprovalContextErrorV1> {
        Ok(Self {
            nonce: os_random_32()?,
        })
    }

    /// Stable, domain-separated public reference for this live incarnation.
    ///
    /// The raw random nonce is intentionally not exposed. The reference may be
    /// persisted in request/decision evidence but cannot recreate the live object.
    pub fn reference(&self) -> String {
        let mut h = Hasher::new();
        h.update(DAEMON_INCARNATION_DOMAIN);
        h.update(&self.nonce);
        format!("nixward-daemon-incarnation-v1:{}", h.finalize().to_hex())
    }

    /// Create a request-bound local approval request using this live incarnation.
    ///
    /// This is the preferred daemon-facing constructor. It owns both runtime-only
    /// uniqueness inputs:
    ///
    /// - daemon incarnation identity;
    /// - fresh 256-bit request nonce.
    ///
    /// The caller supplies semantic intent, review presentation/profile and time
    /// window only. It cannot accidentally reuse a persisted incarnation string or
    /// a previous request nonce through this API.
    pub fn create_approval_request(
        &self,
        intent: &NixActionIntentV1,
        displayed_action: &str,
        authority_profile_ref: impl Into<String>,
        created_at: UnixMillisV1,
        expires_at: UnixMillisV1,
    ) -> Result<PendingNixApprovalRequestV1, DaemonApprovalContextErrorV1> {
        self.create_approval_request_with_nonce(
            intent,
            displayed_action,
            authority_profile_ref,
            created_at,
            expires_at,
            os_random_32()?,
        )
    }

    fn create_approval_request_with_nonce(
        &self,
        intent: &NixActionIntentV1,
        displayed_action: &str,
        authority_profile_ref: impl Into<String>,
        created_at: UnixMillisV1,
        expires_at: UnixMillisV1,
        request_nonce: [u8; 32],
    ) -> Result<PendingNixApprovalRequestV1, DaemonApprovalContextErrorV1> {
        Ok(PendingNixApprovalRequestV1::from_intent(
            intent,
            self.reference(),
            displayed_action,
            authority_profile_ref,
            created_at,
            expires_at,
            request_nonce,
        )?)
    }

    #[cfg(test)]
    fn from_test_nonce(nonce: [u8; 32]) -> Self {
        Self { nonce }
    }
}

impl std::fmt::Debug for LiveDaemonIncarnationV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveDaemonIncarnationV1")
            .field("reference", &self.reference())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Error)]
pub enum DaemonApprovalContextErrorV1 {
    #[error("operating-system randomness unavailable for daemon approval context: {0}")]
    OsRandom(String),
    #[error(transparent)]
    LocalApproval(#[from] LocalApprovalErrorV1),
}

fn os_random_32() -> Result<[u8; 32], DaemonApprovalContextErrorV1> {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes)
        .map_err(|err| DaemonApprovalContextErrorV1::OsRandom(err.to_string()))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::executor::NixOSCommand;

    fn intent() -> NixActionIntentV1 {
        NixActionIntentV1::from_command(
            "machine:workstation",
            Some("generation:42".to_string()),
            &NixOSCommand::RebuildSwitch {
                flake: Some(".#workstation".to_string()),
                extra_args: vec![],
            },
        )
        .unwrap()
    }

    #[test]
    fn reference_is_deterministic_domain_separated_and_does_not_expose_nonce() {
        let incarnation = LiveDaemonIncarnationV1::from_test_nonce([7; 32]);
        let reference = incarnation.reference();

        assert_eq!(reference, incarnation.reference());
        assert!(reference.starts_with("nixward-daemon-incarnation-v1:"));
        assert_eq!(reference.len(), "nixward-daemon-incarnation-v1:".len() + 64);
        assert!(!reference.contains("07070707"));
        assert_ne!(
            reference,
            format!(
                "nixward-daemon-incarnation-v1:{}",
                blake3::hash(&[7; 32]).to_hex()
            ),
            "reference must include the explicit incarnation domain separator"
        );
    }

    #[test]
    fn different_process_incarnations_have_different_references() {
        let first = LiveDaemonIncarnationV1::from_test_nonce([1; 32]);
        let second = LiveDaemonIncarnationV1::from_test_nonce([2; 32]);
        assert_ne!(first.reference(), second.reference());
    }

    #[test]
    fn generated_incarnation_has_well_formed_reference() {
        let incarnation = LiveDaemonIncarnationV1::generate().unwrap();
        let reference = incarnation.reference();
        let digest = reference
            .strip_prefix("nixward-daemon-incarnation-v1:")
            .unwrap();
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn approval_request_is_bound_to_live_incarnation_and_runtime_nonce() {
        let incarnation = LiveDaemonIncarnationV1::from_test_nonce([9; 32]);
        let request = incarnation
            .create_approval_request(
                &intent(),
                "nixos-rebuild switch --flake .#workstation",
                "local-human-v1",
                UnixMillisV1::new(1_000),
                UnixMillisV1::new(2_000),
            )
            .unwrap();

        assert_eq!(request.daemon_incarnation_id, incarnation.reference());
        assert_ne!(request.request_nonce, [0; 32]);
    }

    #[test]
    fn request_nonce_deterministically_changes_request_identity() {
        let incarnation = LiveDaemonIncarnationV1::from_test_nonce([9; 32]);
        let first = incarnation
            .create_approval_request_with_nonce(
                &intent(),
                "nixos-rebuild switch --flake .#workstation",
                "local-human-v1",
                UnixMillisV1::new(1_000),
                UnixMillisV1::new(2_000),
                [3; 32],
            )
            .unwrap();
        let second = incarnation
            .create_approval_request_with_nonce(
                &intent(),
                "nixos-rebuild switch --flake .#workstation",
                "local-human-v1",
                UnixMillisV1::new(1_000),
                UnixMillisV1::new(2_000),
                [4; 32],
            )
            .unwrap();

        assert_ne!(first.request_nonce, second.request_nonce);
        assert_ne!(first.request_id().unwrap(), second.request_id().unwrap());
    }

    #[test]
    fn persisted_reference_is_not_a_live_constructor_input() {
        let incarnation = LiveDaemonIncarnationV1::from_test_nonce([4; 32]);
        let persisted = incarnation.reference();
        assert!(!persisted.is_empty());

        // There is intentionally no `from_reference`, Deserialize, or Clone
        // implementation for `LiveDaemonIncarnationV1`. Persisting this string is
        // audit/provenance evidence only and cannot revive the process incarnation.
    }
}
