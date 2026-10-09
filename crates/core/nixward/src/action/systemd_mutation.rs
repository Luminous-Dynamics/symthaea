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
const DBUS_DESTINATION: &str = "org.freedesktop.DBus";
const DBUS_PATH: &str = "/org/freedesktop/DBus";
const DBUS_INTERFACE: &str = "org.freedesktop.DBus";
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
    #[error("systemd manager incarnation changed before lifecycle dispatch")]
    ManagerOwnerChanged,
    #[error("D-Bus daemon incarnation changed before lifecycle dispatch")]
    BusIncarnationChanged,
    #[error("systemd returned an invalid Job object path")]
    InvalidJobObjectPath,
    #[error("systemd returned an invalid unit-file change record")]
    InvalidUnitFileChange,
}

/// Typed lifecycle mutation transport.
///
/// The connection is private and the dispatch method accepts only the semantic
/// service operation type. No free-form shell command can enter this boundary.
pub struct NixSystemdLifecycleMutationTransportV1 {
    connection: Connection,
}

/// Result returned by systemd's unit-file mutation APIs.
///
/// Enable/Disable are configuration mutations, not lifecycle jobs; their
/// return value is deliberately kept distinct from JobRemoved evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixSystemdUnitFileOperationResultV1 {
    pub carries_install_info: Option<bool>,
    pub changes: Vec<NixSystemdUnitFileChangeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixSystemdUnitFileChangeV1 {
    pub change_type: String,
    pub filename: String,
    pub destination: String,
}

impl NixSystemdLifecycleMutationTransportV1 {
    pub async fn connect_system() -> Result<Self, NixSystemdMutationTransportErrorV1> {
        Ok(Self {
            connection: Connection::system().await?,
        })
    }

    pub fn from_connection(connection: Connection) -> Self {
        Self { connection }
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
        operation.validate_shape().map_err(|error| {
            NixSystemdMutationTransportErrorV1::InvalidServiceOperation(error.to_string())
        })?;
        validate_manager_owner(manager_owner)?;
        let bus = Proxy::new(
            &self.connection,
            DBUS_DESTINATION,
            DBUS_PATH,
            DBUS_INTERFACE,
        )
        .await?;
        let current_owner: String = bus.call("GetNameOwner", &(SYSTEMD_DESTINATION,)).await?;
        if current_owner != manager_owner {
            return Err(NixSystemdMutationTransportErrorV1::ManagerOwnerChanged);
        }

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

    /// Governed dispatch bound to both the exact systemd manager owner and
    /// D-Bus daemon incarnation captured with the Service approval.
    pub async fn dispatch_lifecycle_for_manager_owner_and_bus_id(
        &self,
        operation: &NixServiceOperationV1,
        manager_owner: &str,
        expected_bus_id: &str,
    ) -> Result<OwnedObjectPath, NixSystemdMutationTransportErrorV1> {
        operation.validate_shape().map_err(|error| {
            NixSystemdMutationTransportErrorV1::InvalidServiceOperation(error.to_string())
        })?;
        validate_manager_owner(manager_owner)?;
        validate_bus_id(expected_bus_id)?;

        let bus = Proxy::new(
            &self.connection,
            DBUS_DESTINATION,
            DBUS_PATH,
            DBUS_INTERFACE,
        )
        .await?;

        let current_owner: String = bus.call("GetNameOwner", &(SYSTEMD_DESTINATION,)).await?;
        if current_owner != manager_owner {
            return Err(NixSystemdMutationTransportErrorV1::ManagerOwnerChanged);
        }

        let current_bus_id: String = bus.call("GetId", &()).await?;
        validate_bus_id(&current_bus_id)?;
        if current_bus_id != expected_bus_id {
            return Err(NixSystemdMutationTransportErrorV1::BusIncarnationChanged);
        }
        let confirmed_owner: String = bus.call("GetNameOwner", &(SYSTEMD_DESTINATION,)).await?;
        if confirmed_owner != manager_owner {
            return Err(NixSystemdMutationTransportErrorV1::ManagerOwnerChanged);
        }

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

        // A manager or bus rollover during the D-Bus RPC invalidates the dispatch
        // as qualifying evidence even if a Job path was returned.
        let final_owner: String = bus.call("GetNameOwner", &(SYSTEMD_DESTINATION,)).await?;
        if final_owner != manager_owner {
            return Err(NixSystemdMutationTransportErrorV1::ManagerOwnerChanged);
        }
        let final_bus_id: String = bus.call("GetId", &()).await?;
        validate_bus_id(&final_bus_id)?;
        if final_bus_id != expected_bus_id {
            return Err(NixSystemdMutationTransportErrorV1::BusIncarnationChanged);
        }
        let confirmed_final_owner: String =
            bus.call("GetNameOwner", &(SYSTEMD_DESTINATION,)).await?;
        if confirmed_final_owner != manager_owner {
            return Err(NixSystemdMutationTransportErrorV1::ManagerOwnerChanged);
        }

        validate_job_object_path(&job_path)?;
        Ok(job_path)
    }

    /// Enable the canonical unit through the exact approved systemd manager epoch.
    ///
    /// This returns unit-file change evidence, not a lifecycle Job handle.
    pub async fn enable_unit_file_for_manager_owner_and_bus_id(
        &self,
        unit: &str,
        manager_owner: &str,
        expected_bus_id: &str,
    ) -> Result<NixSystemdUnitFileOperationResultV1, NixSystemdMutationTransportErrorV1> {
        let unit = validate_unit_file_name(unit)?;
        self.verify_manager_epoch(manager_owner, expected_bus_id).await?;

        let manager = Proxy::new(
            &self.connection,
            manager_owner,
            SYSTEMD_MANAGER_PATH,
            SYSTEMD_MANAGER_INTERFACE,
        )
        .await?;

        let (carries_install_info, changes): (bool, Vec<(String, String, String)>) = manager
            .call("EnableUnitFiles", &(vec![unit], false, false))
            .await?;

        // A successful RPC is not qualifying evidence if systemd or the bus
        // rolled over while it was executing.
        self.verify_manager_epoch(manager_owner, expected_bus_id).await?;
        let changes = validate_unit_file_changes(changes)?;

        Ok(NixSystemdUnitFileOperationResultV1 {
            carries_install_info: Some(carries_install_info),
            changes,
        })
    }

    /// Disable the canonical unit through the exact approved systemd manager epoch.
    ///
    /// This returns unit-file change evidence, not a lifecycle Job handle.
    pub async fn disable_unit_file_for_manager_owner_and_bus_id(
        &self,
        unit: &str,
        manager_owner: &str,
        expected_bus_id: &str,
    ) -> Result<NixSystemdUnitFileOperationResultV1, NixSystemdMutationTransportErrorV1> {
        let unit = validate_unit_file_name(unit)?;
        self.verify_manager_epoch(manager_owner, expected_bus_id).await?;

        let manager = Proxy::new(
            &self.connection,
            manager_owner,
            SYSTEMD_MANAGER_PATH,
            SYSTEMD_MANAGER_INTERFACE,
        )
        .await?;

        let changes: Vec<(String, String, String)> = manager
            .call("DisableUnitFiles", &(vec![unit], false))
            .await?;

        self.verify_manager_epoch(manager_owner, expected_bus_id).await?;
        let changes = validate_unit_file_changes(changes)?;

        Ok(NixSystemdUnitFileOperationResultV1 {
            carries_install_info: None,
            changes,
        })
    }

    async fn verify_manager_epoch(
        &self,
        manager_owner: &str,
        expected_bus_id: &str,
    ) -> Result<(), NixSystemdMutationTransportErrorV1> {
        validate_manager_owner(manager_owner)?;
        validate_bus_id(expected_bus_id)?;

        let bus = Proxy::new(
            &self.connection,
            DBUS_DESTINATION,
            DBUS_PATH,
            DBUS_INTERFACE,
        )
        .await?;

        let first_owner: String = bus.call("GetNameOwner", &(SYSTEMD_DESTINATION,)).await?;
        let observed_bus_id: String = bus.call("GetId", &()).await?;
        let confirmed_owner: String = bus.call("GetNameOwner", &(SYSTEMD_DESTINATION,)).await?;

        validate_manager_epoch_observation(
            manager_owner,
            expected_bus_id,
            &first_owner,
            &observed_bus_id,
            &confirmed_owner,
        )
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
            return Err(NixSystemdMutationTransportErrorV1::UnsupportedOperation(
                "enable",
            ));
        }
        NixServiceOperationKindV1::Disable => {
            return Err(NixSystemdMutationTransportErrorV1::UnsupportedOperation(
                "disable",
            ));
        }
    };
    Ok(value)
}

fn validate_manager_owner(owner: &str) -> Result<(), NixSystemdMutationTransportErrorV1> {
    zbus::names::UniqueName::try_from(owner)
        .map(|_| ())
        .map_err(|_| NixSystemdMutationTransportErrorV1::InvalidManagerOwner)
}

fn validate_bus_id(bus_id: &str) -> Result<(), NixSystemdMutationTransportErrorV1> {
    if bus_id.len() != 32 || !bus_id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(NixSystemdMutationTransportErrorV1::BusIncarnationChanged);
    }
    Ok(())
}

/// Validate a bounded owner/bus/owner observation around a bus-incarnation read.
///
/// This narrows manager rollover ambiguity, but it is not an atomic snapshot of
/// external systemd state; consumers must retain the surrounding claim ceiling.
fn validate_manager_epoch_observation(
    expected_owner: &str,
    expected_bus_id: &str,
    first_owner: &str,
    observed_bus_id: &str,
    confirmed_owner: &str,
) -> Result<(), NixSystemdMutationTransportErrorV1> {
    validate_manager_owner(expected_owner)?;
    validate_bus_id(expected_bus_id)?;
    validate_manager_owner(first_owner)?;
    if first_owner != expected_owner {
        return Err(NixSystemdMutationTransportErrorV1::ManagerOwnerChanged);
    }
    validate_bus_id(observed_bus_id)?;
    if observed_bus_id != expected_bus_id {
        return Err(NixSystemdMutationTransportErrorV1::BusIncarnationChanged);
    }
    validate_manager_owner(confirmed_owner)?;
    if confirmed_owner != expected_owner {
        return Err(NixSystemdMutationTransportErrorV1::ManagerOwnerChanged);
    }
    Ok(())
}

fn validate_unit_file_name(
    unit: &str,
) -> Result<String, NixSystemdMutationTransportErrorV1> {
    let operation = NixServiceOperationV1::new(
        unit.to_string(),
        NixServiceOperationKindV1::Start,
    )
    .map_err(|error| {
        NixSystemdMutationTransportErrorV1::InvalidServiceOperation(error.to_string())
    })?;
    Ok(operation.unit().to_string())
}

fn validate_unit_file_changes(
    changes: Vec<(String, String, String)>,
) -> Result<Vec<NixSystemdUnitFileChangeV1>, NixSystemdMutationTransportErrorV1> {
    changes
        .into_iter()
        .map(|(change_type, filename, destination)| {
            if !matches!(change_type.as_str(), "symlink" | "unlink")
                || filename.is_empty()
                || destination.is_empty()
            {
                return Err(NixSystemdMutationTransportErrorV1::InvalidUnitFileChange);
            }
            Ok(NixSystemdUnitFileChangeV1 {
                change_type,
                filename,
                destination,
            })
        })
        .collect()
}

fn validate_job_object_path(
    path: &OwnedObjectPath,
) -> Result<(), NixSystemdMutationTransportErrorV1> {
    let value = path.as_str();
    if value.is_empty() || value.len() > 4096 || !value.starts_with(JOB_PATH_PREFIX) {
        return Err(NixSystemdMutationTransportErrorV1::InvalidJobObjectPath);
    }
    let suffix = &value[JOB_PATH_PREFIX.len()..];
    if suffix.is_empty() || suffix.contains('/') {
        return Err(NixSystemdMutationTransportErrorV1::InvalidJobObjectPath);
    }
    let id = suffix
        .parse::<u32>()
        .map_err(|_| NixSystemdMutationTransportErrorV1::InvalidJobObjectPath)?;
    if id == 0 {
        return Err(NixSystemdMutationTransportErrorV1::InvalidJobObjectPath);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manager_epoch_validation_rejects_owner_and_bus_rollover() {
        let owner = ":1.42";
        let bus_id = "0123456789abcdef0123456789abcdef";

        assert!(validate_manager_epoch_observation(owner, bus_id, owner, bus_id, owner).is_ok());
        assert_eq!(
            validate_manager_epoch_observation(
                owner,
                bus_id,
                ":1.43",
                bus_id,
                ":1.43",
            )
            .unwrap_err(),
            NixSystemdMutationTransportErrorV1::ManagerOwnerChanged
        );
        assert_eq!(
            validate_manager_epoch_observation(
                owner,
                bus_id,
                owner,
                "fedcba9876543210fedcba9876543210",
                owner,
            )
            .unwrap_err(),
            NixSystemdMutationTransportErrorV1::BusIncarnationChanged
        );
        assert_eq!(
            validate_manager_epoch_observation(owner, bus_id, owner, bus_id, ":1.43")
                .unwrap_err(),
            NixSystemdMutationTransportErrorV1::ManagerOwnerChanged
        );
    }

    #[test]
    fn unit_file_change_records_are_strict() {
        let changes = validate_unit_file_changes(vec![
            (
                "symlink".into(),
                "/etc/systemd/system/multi-user.target.wants/nginx.service".into(),
                "/nix/store/nginx.service".into(),
            ),
            (
                "unlink".into(),
                "/etc/systemd/system/multi-user.target.wants/nginx.service".into(),
                "/nix/store/nginx.service".into(),
            ),
        ])
        .unwrap();
        assert_eq!(changes.len(), 2);
        assert!(validate_unit_file_changes(vec![(
            "unknown".into(),
            "/etc/systemd/system/nginx.service".into(),
            "/nix/store/nginx.service".into(),
        )])
        .is_err());
        assert!(validate_unit_file_changes(vec![(
            "symlink".into(),
            String::new(),
            "/nix/store/nginx.service".into(),
        )])
        .is_err());
    }

    #[test]
    fn unit_file_name_reuses_typed_unit_validation() {
        assert_eq!(validate_unit_file_name("nginx.service").unwrap(), "nginx.service");
        assert!(validate_unit_file_name("nginx*.service").is_err());
        assert!(validate_unit_file_name("").is_err());
    }

    #[test]
    fn lifecycle_methods_are_exact() {
        assert_eq!(
            NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Start)
                .unwrap(),
            "StartUnit"
        );
        assert_eq!(
            NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Stop)
                .unwrap(),
            "StopUnit"
        );
        assert_eq!(
            NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Restart)
                .unwrap(),
            "RestartUnit"
        );
        assert_eq!(
            NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Reload)
                .unwrap(),
            "ReloadUnit"
        );
        assert!(
            NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Enable)
                .is_err()
        );
        assert!(
            NixSystemdLifecycleMutationTransportV1::method_name(NixServiceOperationKindV1::Disable)
                .is_err()
        );
    }

    #[test]
    fn lifecycle_job_types_are_exact() {
        assert_eq!(
            NixSystemdLifecycleMutationTransportV1::job_type_name(
                NixServiceOperationKindV1::Restart
            )
            .unwrap(),
            "restart"
        );
        assert_eq!(
            NixSystemdLifecycleMutationTransportV1::job_type_name(
                NixServiceOperationKindV1::Reload
            )
            .unwrap(),
            "reload"
        );
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
    fn governed_dispatch_api_requires_manager_owner_and_bus_epoch() {
        let _method =
            NixSystemdLifecycleMutationTransportV1::dispatch_lifecycle_for_manager_owner_and_bus_id;
        assert!(validate_manager_owner(":1.42").is_ok());
        assert!(validate_bus_id("0123456789abcdef0123456789abcdef").is_ok());
        assert!(validate_bus_id("").is_err());
        assert!(validate_bus_id("not-a-bus-id").is_err());
    }

    fn lifecycle_dispatch_api_is_owner_bound_not_well_known_name_bound() {
        let _method = NixSystemdLifecycleMutationTransportV1::dispatch_lifecycle_for_manager_owner;
        assert!(validate_manager_owner(":1.42").is_ok());
    }

    #[test]
    fn returned_job_path_must_be_in_systemd_job_namespace() {
        let good = OwnedObjectPath::try_from("/org/freedesktop/systemd1/job/42").unwrap();
        let bad_namespace =
            OwnedObjectPath::try_from("/org/freedesktop/systemd1/unit/nginx_2eservice").unwrap();
        let bad_id =
            OwnedObjectPath::try_from("/org/freedesktop/systemd1/job/not-a-number").unwrap();
        let zero = OwnedObjectPath::try_from("/org/freedesktop/systemd1/job/0").unwrap();
        let nested = OwnedObjectPath::try_from("/org/freedesktop/systemd1/job/42/extra").unwrap();
        validate_job_object_path(&good).unwrap();
        assert!(validate_job_object_path(&bad_namespace).is_err());
        assert!(validate_job_object_path(&bad_id).is_err());
        assert!(validate_job_object_path(&zero).is_err());
        assert!(validate_job_object_path(&nested).is_err());
    }
}
