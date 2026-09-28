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

use super::authorization::NixActionIntentV1;
use super::daemon_incarnation::{DaemonApprovalContextErrorV1, LiveDaemonIncarnationV1};
use super::local_approval::PendingNixApprovalRequestV1;
use super::local_approval_socket::{
    LocalApprovalSocketErrorV1, LocalApprovalSocketServerV1,
    default_local_approval_runtime_dir_v1,
};
use super::local_approval_store::{
    ConsumedLocalApprovalDecisionV1, LocalApprovalRequestStoreErrorV1,
    LocalApprovalRequestStoreV1, PendingRequestInstallV1,
};
use super::temporal::UnixMillisV1;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// One request minted by, and installed into, this exact live approval runtime.
///
/// This handle is deliberately not Serialize/Deserialize/Clone. The enclosed
/// request remains ordinary serializable evidence and can be projected explicitly
/// for the intended local UI, but the live-install fact is not recreated by serde.
pub struct InstalledLocalApprovalRequestV1 {
    request: PendingNixApprovalRequestV1,
    install: PendingRequestInstallV1,
}

impl InstalledLocalApprovalRequestV1 {
    pub fn request(&self) -> &PendingNixApprovalRequestV1 {
        &self.request
    }

    pub fn request_id(&self) -> &str {
        &self.install.request_id
    }

    pub fn superseded_request_ids(&self) -> &[String] {
        &self.install.superseded_request_ids
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
    /// The caller supplies semantic intent/review/profile/time only. Incarnation
    /// identity and request nonce are owned by the live daemon context.
    pub fn create_pending_request(
        &self,
        intent: &NixActionIntentV1,
        displayed_action: &str,
        authority_profile_ref: impl Into<String>,
        created_at: UnixMillisV1,
        expires_at: UnixMillisV1,
    ) -> Result<InstalledLocalApprovalRequestV1, LocalApprovalRuntimeErrorV1> {
        let request = self.daemon_incarnation.create_approval_request(
            intent,
            displayed_action,
            authority_profile_ref,
            created_at,
            expires_at,
        )?;
        let install = self.request_store.install_pending(request.clone())?;

        debug_assert_eq!(request.daemon_incarnation_id, self.daemon_incarnation.reference());
        debug_assert_eq!(request.request_id().ok().as_deref(), Some(install.request_id.as_str()));

        Ok(InstalledLocalApprovalRequestV1 { request, install })
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
    Socket(#[from] LocalApprovalSocketErrorV1),
    #[error(transparent)]
    RequestStore(#[from] LocalApprovalRequestStoreErrorV1),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::executor::NixOSCommand;
    use crate::action::{
        LocalApprovalDecisionKindV1, LocalApprovalSubmissionV1, submit_local_approval_v1,
    };
    use std::sync::Arc;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn intent(service: &str) -> NixActionIntentV1 {
        NixActionIntentV1::from_command(
            "machine:workstation",
            Some("generation:42".to_string()),
            &NixOSCommand::Custom {
                command: "systemctl".to_string(),
                args: vec!["restart".to_string(), service.to_string()],
                safety_level: crate::action::SafetyLevel::SystemModify,
            },
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
    fn composition_root_binds_store_socket_and_requests_to_one_incarnation() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();

        let installed = runtime
            .create_pending_request(
                &intent("nginx.service"),
                "restart nginx.service",
                "local-human-v1",
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
    fn same_intent_reissue_supersedes_previous_request_inside_same_runtime() {
        let parent = tempfile::tempdir().unwrap();
        let runtime = LocalApprovalRuntimeV1::bind_in(&parent.path().join("runtime")).unwrap();
        let now = wall_ms();
        let action = intent("nginx.service");

        let first = runtime
            .create_pending_request(
                &action,
                "restart nginx.service",
                "local-human-v1",
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let first_id = first.request_id().to_string();

        let second = runtime
            .create_pending_request(
                &action,
                "restart nginx.service",
                "local-human-v1",
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
                "restart nginx.service",
                "local-human-v1",
                UnixMillisV1::new(now.saturating_sub(1_000)),
                UnixMillisV1::new(now + 60_000),
            )
            .unwrap();
        let submission = LocalApprovalSubmissionV1::for_request(
            installed.request(),
            LocalApprovalDecisionKindV1::Approved,
            UnixMillisV1::new(wall_ms()),
        )
        .unwrap();
        let socket_path = runtime.socket_path().to_path_buf();
        let client = thread::spawn(move || {
            submit_local_approval_v1(&socket_path, &submission).unwrap()
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
