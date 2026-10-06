// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Narrow typed systemd lifecycle mutation transport.
//!
//! This module is deliberately not an authority layer. It accepts only a
//! validated NixServiceOperationV1, emits no shell command, and returns the
//! exact systemd Job object path returned by D-Bus.
//!
//! Authorization, currentness checks, and admission remain the responsibility
//! of the caller. The purpose here is to make the mutation transport capable of
//! participating in CROSS-061's pre-armed JobRemoved protocol.

use super::service_domain::{NixServiceOperationKindV1, NixServiceOperationV1};
use thiserror::Error;
use zbus::zvariant::OwnedObjectPath;
use zbus::{Connection, Proxy};

const SYSTEMD_DESTINATION: &str = "org.freedesktop.systemd1";
const SYSTEMD_MANAGER_PATH: &str = "/org/freedesktop/systemd1";
const SYSTEMD_MANAGER_INTERFACE: &str = "org.freedesktop.systemd1.Manager";
const JOB_PATH_PREFIX: &str = "/org/freedesktop/systemd1/job/";
const JOB_MODE_REPLACE: &str = "replace";

#[derive(Debug, Error)]
pub enum NixSystemdMutationTransportErrorV1 {
    #[error("systemd D-Bus error: {0}")]
    Dbus(#[from] zbus::Error),
    #[error("invalid service operation: {0}")]
    InvalidServiceOperation(String),
    #[error("unsupported lifecycle operation for native D-Bus dispatch: {0}")]
    UnsupportedOperation(&'static str),
    #[error("invalid systemd manager unique D-Bus owner")]
    InvalidManagerOwner,
    #[error("systemd returned an invalid Job object path")]
    InvalidJobObjectPath,
}

/// Typed lifecycle mutation transport.
///
/// The connection is private and the dispatch method accepts only the semantic
/// service operation type. No free-form shell command can enter this boundary.
pub struct NixSystemdLifecycleMutationTransportV1 {
    connection: Connection,
}

impl NixSystemdLifecycleMutationTransportV1 {
    pub async fn connect_system()
        -> Result<Self, NixSystemdMutationTransportErrorV1>
    {
        Ok(Self {
            connection: Connection::system().await?,
        })
    }

    pub fn from_connection(connection: Connection) -> Self {
        Self { connection }
    }

    /// Dispatch one validated lifecycle operation to the captured systemd
    /// manager incarnation and return systemd's exact Job object path.
    ///
    /// This function deliberately does not accept or validate an authorization
    /// record. It is a transport primitive only and must remain downstream of
    /// the live execution-authority check.
    async fn dispatch_lifecycle(
        &self,
        operation: &NixServiceOperationV1,
    ) -> Result<OwnedObjectPath, NixSystemdMutationTransportErrorV1> {
        operation
            .validate_shape()
            .map_err(|error| {
                NixSystemdMutationTransportErrorV1::InvalidServiceOperation(error.to_string())
            })?;

        let (method, _job_type) = method_and_job_type(operation.operation())?;
        let manager = Proxy::new(
            &self.connection,
            SYSTEMD_DESTINATION,
            SYSTEMD_MANAGER_PATH,
            SYSTEMD_MANAGER_INTERFACE,
        )
        .await?;

        let job_path: OwnedObjectPath = manager
            .call(method, &(operation.unit(), JOB_MODE_REPLACE))
            .await?;

        validate_job_object_path(&job_path)?;
        Ok(job_path)
    }

    /// Dispatch only to the exact systemd manager connection captured by the
    /// read-only observer.
    ///
    /// The destination is the manager's unique D-Bus name rather than
    /// org.freedesktop.systemd1. If systemd restarts after observation, the
    /// old unique name has no owner and the call fails instead of being routed
    /// to the new manager incarnation.
    pub async fn dispatch_lifecycle_for_manager_owner(
        &self,
        operation: &NixServiceOperationV1,
        manager_owner: &str,
    ) -> Result<OwnedObjectPath, NixSystemdMutationTransportErrorV1> {
        operation
            .validate_shape()
            .map_err(|error| {
                NixSystemdMutationTransportErrorV1::InvalidServiceOperation(error.to_string())
            })?;
        validate_manager_owner(manager_owner)?;

        let (method, _job_type) = method_and_job_type(operation.operation())?;
        let manager = Proxy::new(
            &self.connection,
            manager_owner,
            SYSTEMD_MANAGER_PATH,
            SYSTEMD_MANAGER_INTERFACE,
        )
        .await?;

        let job_path: OwnedObjectPath = manager
            .call(method, &(operation.unit(), JOB_MODE_REPLACE))
            .await?;

        validate_job_object_path(&job_path)?;
        Ok(job_path)
    }

    pub fn method_name(
        operation: NixServiceOperationKindV1,
    ) -> Result<&'static str, NixSystemdMutationTransportErrorV1> {
        Ok(method_and_job_type(operation)?.0)
    }

    pub fn job_type_name(
        operation: NixServiceOperationKindV1,
    ) -> Result<&'static str, NixSystemdMutationTransportErrorV1> {
        Ok(method_and_job_type(operation)?.1)
    }
}

fn method_and_job_type(
    operation: NixServiceOperationKindV1,
) -> Result<(&'static str, &'static str), NixSystemdMutationTransportErrorV1> {
    let value = match operation {
        NixServiceOperationKindV1::Start => ("StartUnit", "start"),
        NixServiceOperationKindV1::Stop => ("StopUnit", "stop"),
        NixServiceOperationKindV1::Restart => ("RestartUnit", "restart"),
        NixServiceOperationKindV1::Reload => ("ReloadUnit", "reload"),
        NixServiceOperationKindV1::Enable => {
            return Err(NixSystemdMutationTransportErrorV1::UnsupportedOperation("enable"))
        }
        NixServiceOperationKindV1::Disable => {
            return Err(NixSystemdMutationTransportErrorV1::UnsupportedOperation("disable"))
        }
    };
    Ok(value)
}

fn validate_manager_owner(
    owner: &str,
) -> Result<(), NixSystemdMutationTransportErrorV1> {
    zbus::names::UniqueName::try_from(owner).map(|_| ()).map_err(|_| {
        NixSystemdMutationTransportErrorV1::InvalidManagerOwner
    })
}

fn validate_job_object_path(
    path: &OwnedObjectPath,
) -> Result<(), NixSystemdMutationTransportErrorV1> {
    let value = path.as_str();
    if value.is_empty() || value.len() > 4096 || !value.starts_with(JOB_PATH_PREFIX) {
        return Err(NixSystemdMutationTransportErrorV1::InvalidJobObjectPath);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_methods_are_exact() {
        assert_eq!(NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Start).unwrap(), "StartUnit");
        assert_eq!(NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Stop).unwrap(), "StopUnit");
        assert_eq!(NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Restart).unwrap(), "RestartUnit");
        assert_eq!(NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Reload).unwrap(), "ReloadUnit");
        assert!(NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Enable).is_err());
        assert!(NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Disable).is_err());
    }

    #[test]
    fn lifecycle_job_types_are_exact() {
        assert_eq!(NixSystemdLifecycleMutationTransportV1::job_type_name(NixServiceOperationKindV1::Restart).unwrap(), "restart");
        assert_eq!(NixSystemdLifecycleMutationTransportV1::job_type_name(NixServiceOperationKindV1::Reload).unwrap(), "reload");
    }

    #[test]
    fn manager_owner_must_be_a_unique_dbus_name() {
        assert!(validate_manager_owner(":1.42").is_ok());
        assert!(validate_manager_owner("org.freedesktop.systemd1").is_err());
        assert!(validate_manager_owner(":").is_err());
        assert!(validate_manager_owner("").is_err());
    }

    #[test]
    fn bound_dispatch_api_requires_an_explicit_manager_owner() {
        let _method = NixSystemdLifecycleMutationTransportV1::dispatch_lifecycle_for_manager_owner;
        assert!(validate_manager_owner(":1.42").is_ok());
    }

    #[test]
    fn returned_job_path_must_be_in_systemd_job_namespace() {
        let good = OwnedObjectPath::try_from("/org/freedesktop/systemd1/job/42").unwrap();
        let bad = OwnedObjectPath::try_from("/org/freedesktop/systemd1/unit/nginx_2eservice").unwrap();
        validate_job_object_path(&good).unwrap();
        assert!(validate_job_object_path(&bad).is_err());
    }
}
