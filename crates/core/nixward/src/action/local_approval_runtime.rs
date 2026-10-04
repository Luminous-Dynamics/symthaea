// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Daemon-local composition root for Nixward's local approval primitives.
//!
//! The lower layers intentionally prove separate facts. This module prevents the
//! normal daemon integration path from accidentally composing those facts from
//! different process/socket incarnations:
//!
//! ```text
//! one LiveDaemonIncarnationV1
//!   -> one LocalApprovalRequestStoreV1
//!   -> one LocalApprovalSocketServerV1
//!   -> requests minted only by that live incarnation
//!   -> exact atomic local decision consumption
//! ```
//!
//! The result is still approval evidence only. This module owns no Nix execution
//! authority, generic grant accounting, `DispatchPermitV2`, or privileged broker.

use super::authorization::{NixActionDescriptorV1, NixActionIntentV1};
use super::approver_evidence::RequiredApprovalProfileV1;
use super::executor::NixOSCommand;
use super::daemon_incarnation::{DaemonApprovalContextErrorV1, LiveDaemonIncarnationV1};
use super::local_approval::{
    PendingNixApprovalRequestV1, operator_visible_action_for_command,
};
use super::local_approval_projection::PendingNixApprovalProjectionV1;
use super::local_approval_socket::{
    LocalApprovalSocketErrorV1, LocalApprovalSocketServerV1,
    default_local_approval_runtime_dir_v1,
};
use super::local_approval_store::{
    ConsumedLocalApprovalDecisionV1, LocalApprovalRequestStoreErrorV1,
    LocalApprovalRequestStoreV1, PendingRequestCurrentnessV1, PendingRequestInstallV1,
};
use super::temporal::UnixMillisV1;
use std::path::Path;
use thiserror::Error;

/// One request minted by, and installed into, this exact live approval runtime.
///
/// This handle is deliberately not Serialize/Deserialize/Clone. The enclosed
/// request remains ordinary serializable evidence and can be projected explicitly
/// for the intended local UI, but the live-install fact is not recreated by serde.
pub struct InstalledLocalApprovalRequestV1 {
    request: PendingNixApprovalRequestV1,
    install: PendingRequestInstallV1,
    /// The exact disclosure text supplied at composition time. It is retained
    /// privately so projection cannot be reconstructed from an unrelated string.
    operator_visible_action: String,
}

impl InstalledLocalApprovalRequestV1 {
    pub fn request(&self) -> &PendingNixApprovalRequestV1 {
        &self.request
    }

    pub fn request_id(&self) -> &str {
        &self.install.request_id
    }

    /// Return the exact operator-visible action derived from the typed command
    /// when this live request was created. Callers receive a read-only view;
    /// they cannot replace the runtime-owned ceremony text.
    pub fn operator_visible_action(&self) -> &str {
        &self.operator_visible_action
    }

    pub fn superseded_request_ids(&self) -> &[String] {
        &self.install.superseded_request_ids
    }

    /// Project exactly what this runtime-owned request was created to display.
    ///
    /// Callers cannot substitute another display string: the original text is
    /// retained privately and the projection constructor re-checks its digest
    /// against the request's canonical display commitment.
    pub fn operator_projection(
        &self,
    ) -> Result<
        super::local_approval_projection::PendingNixApprovalProjectionV1,
        super::local_approval_projection::LocalApprovalProjectionErrorV1,
    > {
        super::local_approval_projection::PendingNixApprovalProjectionV1::from_request(
            &self.request,
            &self.operator_visible_action,
        )
    }
}

impl std::fmt::Debug for InstalledLocalApprovalRequestV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstalledLocalApprovalRequestV1")
            .field("request_id", &self.install.request_id)
            .field("action_intent_digest", &self.request.action_intent_digest)
            .field(
                "superseded_request_ids",
                &self.install.superseded_request_ids,
            )
            .field("operator_visible_action", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// One non-rehydratable local approval runtime for one daemon process.
///
/// Construction owns the live daemon-incarnation generation and derives both the
/// request store and socket listener from that exact same object. There is no
/// public constructor accepting pre-built/mismatched components.
pub struct LocalApprovalRuntimeV1 {
    daemon_incarnation: LiveDaemonIncarnationV1,
    request_store: LocalApprovalRequestStoreV1,
    socket_server: LocalApprovalSocketServerV1,
}

impl LocalApprovalRuntimeV1 {
    /// Bind using the runtime-only path policy from LOCAL-007.
    pub fn bind_default() -> Result<Self, LocalApprovalRuntimeErrorV1> {
        let runtime_dir = default_local_approval_runtime_dir_v1()?;
        Self::bind_in(&runtime_dir)
    }

    /// Bind one live runtime inside an explicit runtime-only directory.
    pub fn bind_in(runtime_dir: &Path) -> Result<Self, LocalApprovalRuntimeErrorV1> {
        let daemon_incarnation = LiveDaemonIncarnationV1::generate()?;
        let request_store = LocalApprovalRequestStoreV1::new(&daemon_incarnation);
        let socket_server = LocalApprovalSocketServerV1::bind_in(runtime_dir, &daemon_incarnation)?;

        debug_assert_eq!(
            request_store.daemon_incarnation_ref(),
            daemon_incarnation.reference()
        );

        Ok(Self {
            daemon_incarnation,
            request_store,
            socket_server,
        })
    }

    pub fn daemon_incarnation_ref(&self) -> String {
        self.daemon_incarnation.reference()
    }

    pub fn socket_path(&self) -> &Path {
        self.socket_server.socket_path()
    }

    pub fn transport_instance_ref(&self) -> &str {
        self.socket_server.transport_instance_ref()
    }

    /// Mint and atomically install one exact local approval request.
    ///
    /// The caller supplies the exact typed command, semantic intent, profile,
    /// and time window. The runtime derives the operator-visible ceremony text
    /// itself from that command before installation.
    /// Incarnation identity and request nonce remain owned by the live daemon context.
    pub fn create_pending_request(
        &self,
        intent: &NixActionIntentV1,
        command: &NixOSCommand,
        authority_profile: RequiredApprovalProfileV1,
        created_at: UnixMillisV1,
        expires_at: UnixMillisV1,
    ) -> Result<InstalledLocalApprovalRequestV1, LocalApprovalRuntimeErrorV1> {
        let expected_action = NixActionDescriptorV1::try_from(command)?;
        if intent.action != expected_action {
            return Err(LocalApprovalRuntimeErrorV1::IntentCommandMismatch);
        }
        let displayed_action = operator_visible_action_for_command(command);

        let request = self.daemon_incarnation.create_approval_request(
            intent,
            &displayed_action,
            authority_profile.as_ref(),
            created_at,
            expires_at,
        )?;
        let projection = PendingNixApprovalProjectionV1::from_request(&request, &displayed_action)?;
        let install = self
            .request_store
            .install_pending_with_projection(request.clone(), projection.projection_digest.clone())?;

        debug_assert_eq!(request.daemon_incarnation_id, self.daemon_incarnation.reference());
        debug_assert_eq!(request.request_id().ok().as_deref(), Some(install.request_id.as_str()));

        Ok(InstalledLocalApprovalRequestV1 {
            request,
            install,
            operator_visible_action: displayed_action,
        })
    }

    /// Observe projection currentness against this live runtime.
    ///
    /// This validates the projection commitment, binds it to this daemon incarnation,
    /// and then checks the live pending store at one instant. The result is deliberately
    /// non-authoritative: it is a UI/currentness observation and may become stale
    /// immediately after return. It never reserves, consumes, or authorizes an effect.
    pub fn observe_projection_currentness(
        &self,
        projection: &super::local_approval_projection::PendingNixApprovalProjectionV1,
        now: UnixMillisV1,
    ) -> Result<PendingRequestCurrentnessV1, LocalApprovalRuntimeErrorV1> {
        if projection.request_id.is_empty() {
            return Ok(PendingRequestCurrentnessV1::NotPending);
        }
        Ok(self
            .request_store
            .observe_projection_currentness(projection, now)?)
    }

    /// Observe whether a runtime-owned installed request is current at one instant.
    ///
    /// This is a non-authoritative snapshot for UI/currentness purposes. It validates
    /// the projection against the exact request retained by the installed handle and
    /// the live daemon incarnation before consulting pending state. The result can
    /// become stale immediately after return and never reserves or authorizes an effect.
    pub fn observe_installed_projection_currentness(
        &self,
        installed: &InstalledLocalApprovalRequestV1,
        projection: &super::local_approval_projection::PendingNixApprovalProjectionV1,
        now: UnixMillisV1,
    ) -> Result<PendingRequestCurrentnessV1, LocalApprovalRuntimeErrorV1> {
        projection.validate()?;
        if installed.request.daemon_incarnation_id != self.daemon_incarnation.reference()
            || projection.daemon_incarnation_ref != self.daemon_incarnation.reference()
            || projection.request_id != installed.request_id()
            || !projection.matches_request(&installed.request, &installed.operator_visible_action)?
        {
            return Ok(PendingRequestCurrentnessV1::NotPending);
        }
        Ok(self.request_store.observe_currentness(installed.request_id(), now)?)
    }

    /// Poll once for a local approval without blocking the daemon.
    ///
    /// A successful return is still approval evidence only; execution authority
    /// remains a separate downstream theorem.
    pub fn try_accept_and_consume(
        &self,
    ) -> Result<Option<ConsumedLocalApprovalDecisionV1>, LocalApprovalRuntimeErrorV1> {
        Ok(self
            .socket_server
            .try_accept_and_consume(&self.request_store)?)
    }
    /// Accept one LOCAL-007 socket submission and atomically consume its request.
    ///
    /// The returned affine-ish local token is still not execution authority.
    pub fn accept_and_consume(
        &self,
    ) -> Result<ConsumedLocalApprovalDecisionV1, LocalApprovalRuntimeErrorV1> {
        Ok(self
            .socket_server
            .accept_and_consume(&self.request_store)?)
    }

    pub fn pending_count(&self) -> Result<usize, LocalApprovalRuntimeErrorV1> {
        Ok(self.request_store.pending_count()?)
    }
}

impl std::fmt::Debug for LocalApprovalRuntimeV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalApprovalRuntimeV1")
            .field("daemon_incarnation_ref", &self.daemon_incarnation.reference())
            .field("socket_path", &self.socket_server.socket_path())
            .field(
                "transport_instance_ref",
                &self.socket_server.transport_instance_ref(),
            )
            .field("pending_count", &self.request_store.pending_count().ok())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Error)]
pub enum LocalApprovalRuntimeErrorV1 {
    #[error(transparent)]
    DaemonContext(#[from] DaemonApprovalContextErrorV1),
    #[error(transparent)]
    Authorization(#[from] super::authorization::NixAuthorizationErrorV1),
    #[error("approval intent action does not match the typed command")]
    IntentCommandMismatch,
    #[error(transparent)]
    Socket(#[from] LocalApprovalSocketErrorV1),
    #[error(transparent)]
    RequestStore(#[from] LocalApprovalRequestStoreErrorV1),
    #[error(transparent)]
    Projection(#[from] super::local_approval_projection::LocalApprovalProjectionErrorV1),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::executor::NixOSCommand;
    use crate::action::{
        LocalApprovalDecisionKindV1, LocalApprovalSubmissionV2, submit_local_approval_v2,
    };
    use std::sync::Arc;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn intent(service: &str) -> NixActionIntentV1 {
        NixActionIntentV1::from_command(
            "machine:workstation",
            Some("generation:42".to_string()),
            &NixOSCommand::Service {
                operation: crate::action::NixServiceOperationKindV1::Restart,
                unit: service.to_string(),
            },
        )
        .unwrap()
    }

    fn restart_command(service: &str) -> NixOSCommand {
        NixOSCommand::Service {
            operation: crate::action::NixServiceOperationKindV1::Restart,
            unit: service.to_string(),
        }
    }

    fn submission_for(
        installed: &InstalledLocalApprovalRequestV1,
        decision: LocalApprovalDecisionKindV1,
    ) -> LocalApprovalSubmissionV2 {
        let projection = installed.operator_projection().unwrap();
        LocalApprovalSubmissionV2::for_request_and_projection(
            installed.request(),
            &projection,
            decision,
            UnixMillisV1::new(wall_ms()),
        )
        .unwrap()
    }

    fn wall_ms() -> u64 {
        u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        )
        .unwrap()
    }

    #[test]
    fn request_creation_rejects_command_intent_substitution() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();
        let stop_command = NixOSCommand::Service {
            operation: crate::action::NixServiceOperationKindV1::Stop,
            unit: "nginx.service".to_string(),
        };

        assert_eq!(
            runtime
                .create_pending_request(
                    &intent("nginx.service"),
                    &stop_command,
                    RequiredApprovalProfileV1::SameUidProcessV1,
                    UnixMillisV1::new(now.saturating_sub(1_000)),
                    UnixMillisV1::new(now + 60_000),
                )
                .unwrap_err()
                .to_string(),
            "approval intent action does not match the typed command"
        );
        assert_eq!(runtime.pending_count().unwrap(), 0);
    }

    #[test]
    fn request_creation_derives_runtime_owned_operator_display() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();
        let command = restart_command("nginx.service");

        let installed = runtime
            .create_pending_request(
                &intent("nginx.service"),
                &command,
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();

        assert_eq!(
            installed.operator_visible_action(),
            "systemctl restart nginx.service"
        );
        let projection = installed.operator_projection().unwrap();
        assert_eq!(projection.operator_visible_action, installed.operator_visible_action());
    }
    
    #[test]
    fn config_patch_display_binds_expected_config_digest() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();
        let command_a = NixOSCommand::ConfigPatch {
            option_path: "services.nginx.enable".to_string(),
            value: "true".to_string(),
            expected_config_digest: "ab".repeat(32),
        };
        let command_b = NixOSCommand::ConfigPatch {
            option_path: "services.nginx.enable".to_string(),
            value: "true".to_string(),
            expected_config_digest: "cd".repeat(32),
        };
        let intent_a = NixActionIntentV1::from_command(
            "machine:workstation",
            Some("generation:42".to_string()),
            &command_a,
        )
        .unwrap();
        let intent_b = NixActionIntentV1::from_command(
            "machine:workstation",
            Some("generation:42".to_string()),
            &command_b,
        )
        .unwrap();

        let a = runtime
            .create_pending_request(
                &intent_a,
                &command_a,
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let b = runtime
            .create_pending_request(
                &intent_b,
                &command_b,
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(500)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();

        assert_ne!(a.operator_visible_action(), b.operator_visible_action());
        assert!(a.operator_visible_action().contains(&"ab".repeat(32)));
        assert!(b.operator_visible_action().contains(&"cd".repeat(32)));
    }

    #[test]
    fn projection_currentness_is_live_observation_not_authority() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();
        let installed = runtime
            .create_pending_request(
                &intent("nginx.service"),
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let projection = installed.operator_projection().unwrap();

        assert_eq!(
            runtime
                .observe_projection_currentness(&projection, UnixMillisV1::new(now))
                .unwrap(),
            PendingRequestCurrentnessV1::Current
        );
        assert_eq!(runtime.pending_count().unwrap(), 1);
    }

    #[test]
    fn projection_currentness_rejects_superseded_and_expired_snapshots() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();
        let action = intent("nginx.service");
        let first = runtime
            .create_pending_request(
                &action,
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let first_projection = first.operator_projection().unwrap();
        let second = runtime
            .create_pending_request(
                &action,
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(500)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let second_projection = second.operator_projection().unwrap();

        assert_eq!(
            runtime
                .observe_projection_currentness(&first_projection, UnixMillisV1::new(now))
                .unwrap(),
            PendingRequestCurrentnessV1::NotPending
        );
        assert_eq!(
            runtime
                .observe_projection_currentness(&second_projection, UnixMillisV1::new(now + 60_000))
                .unwrap(),
            PendingRequestCurrentnessV1::Expired
        );
        assert_eq!(runtime.pending_count().unwrap(), 1);
    }

    #[test]
    fn projection_currentness_rejects_projection_from_another_daemon_incarnation() {
        let parent = tempfile::tempdir().unwrap();
        let runtime_path = parent.path().join("runtime");
        let first = LocalApprovalRuntimeV1::bind_in(&runtime_path).unwrap();
        let now = wall_ms();
        let installed = first
            .create_pending_request(
                &intent("nginx.service"),
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let projection = installed.operator_projection().unwrap();
        drop(first);

        let second = LocalApprovalRuntimeV1::bind_in(&runtime_path).unwrap();
        assert_eq!(
            second
                .observe_projection_currentness(&projection, UnixMillisV1::new(now))
                .unwrap(),
            PendingRequestCurrentnessV1::NotPending
        );
        assert_eq!(second.pending_count().unwrap(), 0);
    }

    #[test]
    fn projection_currentness_requires_exact_runtime_owned_projection() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();
        let installed = runtime
            .create_pending_request(
                &intent("nginx.service"),
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let projection = installed.operator_projection().unwrap();

        assert_eq!(
            runtime
                .observe_installed_projection_currentness(
                    &installed,
                    &projection,
                    UnixMillisV1::new(now),
                )
                .unwrap(),
            PendingRequestCurrentnessV1::Current
        );

        let mut tampered = projection.clone();
        tampered.machine_target_ref = "machine:other".to_string();
        tampered.projection_digest = tampered.compute_digest().unwrap();
        assert!(tampered.validate().is_ok());

        assert_eq!(
            runtime
                .observe_projection_currentness(&tampered, UnixMillisV1::new(now))
                .unwrap(),
            PendingRequestCurrentnessV1::NotPending
        );

        assert!(matches!(
            runtime.observe_installed_projection_currentness(
                &installed,
                &tampered,
                UnixMillisV1::new(now),
            ),
            Err(LocalApprovalRuntimeErrorV1::Projection(_))
        ));
        assert_eq!(runtime.pending_count().unwrap(), 1);
    }

    #[test]
    fn projection_currentness_rejects_superseded_expired_and_restarted_requests() {
        let parent = tempfile::tempdir().unwrap();
        let runtime_path = parent.path().join("runtime");
        let runtime = LocalApprovalRuntimeV1::bind_in(&runtime_path).unwrap();
        let now = wall_ms();
        let action = intent("nginx.service");

        let first = runtime
            .create_pending_request(
                &action,
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let first_projection = first.operator_projection().unwrap();

        let second = runtime
            .create_pending_request(
                &action,
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(500)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let second_projection = second.operator_projection().unwrap();

        assert_eq!(
            runtime
                .observe_installed_projection_currentness(
                    &first,
                    &first_projection,
                    UnixMillisV1::new(now),
                )
                .unwrap(),
            PendingRequestCurrentnessV1::NotPending
        );
        assert_eq!(
            runtime
                .observe_installed_projection_currentness(
                    &second,
                    &second_projection,
                    UnixMillisV1::new(now + 60_000),
                )
                .unwrap(),
            PendingRequestCurrentnessV1::Expired
        );

        drop(runtime);
        let restarted = LocalApprovalRuntimeV1::bind_in(&runtime_path).unwrap();
        assert_eq!(
            restarted
                .observe_installed_projection_currentness(
                    &second,
                    &second_projection,
                    UnixMillisV1::new(now),
                )
                .unwrap(),
            PendingRequestCurrentnessV1::NotPending
        );
    }

    #[test]
    fn composition_root_binds_store_socket_and_requests_to_one_incarnation() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();

        let installed = runtime
            .create_pending_request(
                &intent("nginx.service"),
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();

        assert_eq!(
            installed.request().daemon_incarnation_id,
            runtime.daemon_incarnation_ref()
        );
        assert_eq!(
            installed.request_id(),
            installed.request().request_id().unwrap()
        );
        assert_eq!(runtime.pending_count().unwrap(), 1);
        assert!(runtime
            .transport_instance_ref()
            .starts_with("nixward-local-approval-socket-instance-v1:"));
    }

    #[test]
    fn installed_request_projection_is_derived_from_runtime_owned_display() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();
        let installed = runtime
            .create_pending_request(
                &intent("nginx.service"),
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();

        let projection = installed.operator_projection().unwrap();
        assert_eq!(projection.request_id, installed.request_id());
        assert_eq!(projection.operator_visible_action, "restart nginx.service");
        assert_eq!(
            projection.operator_visible_action_digest,
            installed.request().displayed_action_digest
        );
        assert_eq!(projection.projection_digest, projection.compute_digest().unwrap());
        projection.validate().unwrap();
    }

    #[test]
    fn installed_request_projection_cannot_be_rebound_to_external_display_text() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();
        let installed = runtime
            .create_pending_request(
                &intent("nginx.service"),
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();

        let projection = installed.operator_projection().unwrap();
        assert!(matches!(
            super::super::local_approval_projection::PendingNixApprovalProjectionV1::from_request(
                installed.request(),
                "restart sshd.service",
            ),
            Err(super::super::local_approval_projection::LocalApprovalProjectionErrorV1::DisplayedActionDigestMismatch)
        ));
        assert_eq!(projection.operator_visible_action, "restart nginx.service");
    }

    #[test]
    fn same_intent_reissue_supersedes_previous_request_inside_same_runtime() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();
        let action = intent("nginx.service");

        let first = runtime
            .create_pending_request(
                &action,
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let first_id = first.request_id().to_string();

        let second = runtime
            .create_pending_request(
                &action,
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(500)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();

        assert_ne!(first_id, second.request_id());
        assert_eq!(second.superseded_request_ids(), &[first_id]);
        assert_eq!(runtime.pending_count().unwrap(), 1);
    }

    #[test]
    fn restart_creates_new_incarnation_and_socket_instance() {
        let parent = tempfile::tempdir().unwrap();
        let runtime_path = parent.path().join("runtime");
        let first = LocalApprovalRuntimeV1::bind_in(&runtime_path).unwrap();
        let first_incarnation = first.daemon_incarnation_ref();
        let first_transport = first.transport_instance_ref().to_string();
        drop(first);

        let second = LocalApprovalRuntimeV1::bind_in(&runtime_path).unwrap();
        assert_ne!(first_incarnation, second.daemon_incarnation_ref());
        assert_ne!(first_transport, second.transport_instance_ref());
    }

    #[test]
    fn end_to_end_runtime_returns_consumed_decision_not_execution_authority() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = Arc::new(
            LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap(),
        );
        let now = wall_ms();
        let installed = runtime
            .create_pending_request(
                &intent("nginx.service"),
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let submission = submission_for(&installed, LocalApprovalDecisionKindV1::Approved);
        let socket_path = runtime.socket_path().to_path_buf();
        let client = thread::spawn(move || {
            submit_local_approval_v2(&socket_path, &submission).unwrap()
        });

        let consumed = runtime.accept_and_consume().unwrap();
        let ack = client.join().unwrap();

        assert_eq!(consumed.request_id(), installed.request_id());
        assert_eq!(consumed.decision_kind(), LocalApprovalDecisionKindV1::Approved);
        assert_eq!(ack.request_id, installed.request_id());
        assert_eq!(runtime.pending_count().unwrap(), 0);
        // There is intentionally no conversion here to LiveNixAuthorizationV1
        // or generic DispatchPermitV2. The returned type is the LOCAL-006 token.
    }

    #[test]
    fn restart_rejects_old_submission_after_new_runtime_takes_over() {
        let parent = tempfile::tempdir().unwrap();
        let runtime_path = parent.path().join("runtime");
        let first = LocalApprovalRuntimeV1::bind_in(&runtime_path).unwrap();
        let now = wall_ms();
        let installed = first
            .create_pending_request(
                &intent("nginx.service"),
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let submission = submission_for(&installed, LocalApprovalDecisionKindV1::Approved);
        let old_socket = first.socket_path().to_path_buf();
        let first_transport = first.transport_instance_ref().to_string();
        drop(first);

        let second = LocalApprovalRuntimeV1::bind_in(&runtime_path).unwrap();
        assert_ne!(second.daemon_incarnation_ref(), installed.request().daemon_incarnation_id);
        assert_eq!(second.socket_path(), old_socket.as_path());
        assert_ne!(second.transport_instance_ref(), first_transport);

        let new_socket = second.socket_path().to_path_buf();
        let client = thread::spawn(move || submit_local_approval_v2(&new_socket, &submission));
        let result = second.accept_and_consume();
        let client_result = client.join().unwrap();

        assert!(matches!(
            result,
            Err(LocalApprovalRuntimeErrorV1::Socket(
                LocalApprovalSocketErrorV1::RequestStore(
                    LocalApprovalRequestStoreErrorV1::RequestNotPending
                )
            ))
        ));
        assert!(client_result.is_err());
        assert_eq!(second.pending_count().unwrap(), 0);
    }

    #[test]
    fn superseded_submission_is_rejected_while_current_reissue_remains_consumable() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = Arc::new(
            LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap(),
        );
        let now = wall_ms();
        let action = intent("nginx.service");

        let first = runtime
            .create_pending_request(
                &action,
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let first_submission = submission_for(&first, LocalApprovalDecisionKindV1::Approved);

        let second = runtime
            .create_pending_request(
                &action,
                &restart_command("nginx.service"),
                RequiredApprovalProfileV1::SameUidProcessV1,
                UnixMillisV1::new(now.saturating_sub(500)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let second_submission = submission_for(&second, LocalApprovalDecisionKindV1::Approved);

        assert_eq!(second.superseded_request_ids(), &[first.request_id().to_string()]);
        assert_eq!(runtime.pending_count().unwrap(), 1);

        let stale_socket = runtime.socket_path().to_path_buf();
        let stale_client =
            thread::spawn(move || submit_local_approval_v2(&stale_socket, &first_submission));
        let stale_result = runtime.accept_and_consume();
        let stale_client_result = stale_client.join().unwrap();

        assert!(matches!(
            stale_result,
            Err(LocalApprovalRuntimeErrorV1::Socket(
                LocalApprovalSocketErrorV1::RequestStore(
                    LocalApprovalRequestStoreErrorV1::RequestNotPending
                )
            ))
        ));
        assert!(stale_client_result.is_err());
        assert_eq!(runtime.pending_count().unwrap(), 1);

        let current_socket = runtime.socket_path().to_path_buf();
        let current_client =
            thread::spawn(move || submit_local_approval_v2(&current_socket, &second_submission));
        let current_result = runtime.accept_and_consume().unwrap();
        let current_ack = current_client.join().unwrap().unwrap();

        assert_eq!(current_result.request_id(), second.request_id());
        assert_eq!(current_result.decision_kind(), LocalApprovalDecisionKindV1::Approved);
        assert_eq!(current_ack.request_id, second.request_id());
        assert_eq!(runtime.pending_count().unwrap(), 0);
    }

    #[test]
    fn two_independent_runtimes_do_not_share_incarnation_or_transport_identity() {
        let first_parent = tempfile::tempdir().unwrap();
        let second_parent = tempfile::tempdir().unwrap();
        let first =
            LocalApprovalRuntimeV1::bind_in(&first_parent.path().join("runtime")).unwrap();
        let second =
            LocalApprovalRuntimeV1::bind_in(&second_parent.path().join("runtime")).unwrap();

        assert_ne!(first.daemon_incarnation_ref(), second.daemon_incarnation_ref());
        assert_ne!(first.transport_instance_ref(), second.transport_instance_ref());
    }
}
