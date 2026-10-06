// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent, read-only systemd D-Bus observation for governed Nixward effects.
//!
//! This module is deliberately narrower than the execution transport. It can
//! resolve units, read properties, capture a still-live job identity, consume
//! JobRemoved, and resolve invocation IDs. It exposes no systemd mutation method
//! and never mints execution authority.

use super::post_state::{
    NixPostStateStabilitySampleV1, NixPostStateStabilityEvidenceV1,
    NixServicePostStateObservationV1, NixSystemdJobEvidenceV1, NixSystemdJobTypeV1,
    NixSystemdUnitDefinitionIdentityV1, NixVerifiedPostStateObservationV1,
    NixVerifiedPostStateStabilityEvidenceV1,
};
use super::service_domain::{NixServiceOperationKindV1, NixServiceOperationV1};
use super::service_effect::{
    NixServiceDefinitionContentEvidenceV1, NixSystemdUnitDefinitionContentFileV1,
    NixVerifiedServiceDefinitionContentV1,
};
use super::service_state::{ServiceActiveStateV1, ServiceLoadStateV1, ServiceUnitFileStateV1};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom};
use std::time::Duration;
use thiserror::Error;
use zbus::export::futures_util::StreamExt;
use zbus::fdo::PropertiesProxy;
use zbus::zvariant::{OwnedObjectPath, OwnedValue};
use zbus::{Connection, Message, Proxy};

const SYSTEMD_DESTINATION: &str = "org.freedesktop.systemd1";
const SYSTEMD_MANAGER_PATH: &str = "/org/freedesktop/systemd1";
const SYSTEMD_MANAGER_INTERFACE: &str = "org.freedesktop.systemd1.Manager";
const DBUS_DESTINATION: &str = "org.freedesktop.DBus";
const DBUS_PATH: &str = "/org/freedesktop/DBus";
const DBUS_INTERFACE: &str = "org.freedesktop.DBus";
const SYSTEMD_UNIT_INTERFACE: &str = "org.freedesktop.systemd1.Unit";
const SYSTEMD_SERVICE_INTERFACE: &str = "org.freedesktop.systemd1.Service";
const SYSTEMD_JOB_INTERFACE: &str = "org.freedesktop.systemd1.Job";
const SYSTEMD_UNIT_PATH_PREFIX: &str = "/org/freedesktop/systemd1/unit/";
const SYSTEMD_JOB_PATH_PREFIX: &str = "/org/freedesktop/systemd1/job/";
const INVOCATION_ID_BYTES: usize = 16;

const REQUIRED_UNIT_PROPERTIES: &[&str] = &[
    "Id",
    "Names",
    "LoadState",
    "ActiveState",
    "SubState",
    "FragmentPath",
    "DropInPaths",
    "UnitFileState",
    "StateChangeTimestampMonotonic",
    "InvocationID",
];

#[derive(Debug, Error)]
pub enum NixSystemdObserverErrorV1 {
    #[error("systemd D-Bus error: {0}")]
    Dbus(#[from] zbus::Error),

    #[error("invalid service unit: {0}")]
    InvalidServiceUnit(String),

    #[error("missing required systemd property {interface}.{property}")]
    MissingProperty {
        interface: &'static str,
        property: &'static str,
    },

    #[error("invalid systemd property type for {interface}.{property}")]
    InvalidPropertyType {
        interface: &'static str,
        property: &'static str,
    },

    #[error("invalid systemd property value for {interface}.{property}")]
    InvalidPropertyValue {
        interface: &'static str,
        property: &'static str,
    },

    #[error("unknown systemd state vocabulary: {0}")]
    UnknownStateVocabulary(String),

    #[error("unit identity mismatch: requested {requested}, observed {observed}")]
    UnitIdentityMismatch { requested: String, observed: String },

    #[error("invalid systemd job identity: {0}")]
    InvalidJobIdentity(String),

    #[error("systemd job/unit mismatch: expected {expected}, observed {observed}")]
    JobUnitMismatch { expected: String, observed: String },

    #[error("systemd JobRemoved correlation mismatch")]
    JobCorrelationMismatch,

    #[error("systemd manager owner changed while arming JobRemoved watcher")]
    ManagerOwnerChanged,

    #[error("systemd JobRemoved watcher manager owner mismatch")]
    WatcherManagerOwnerMismatch,

    #[error("systemd JobRemoved signal timed out")]
    JobRemovedTimeout,

    #[error("systemd invocation ID has invalid shape")]
    InvalidInvocationId,

    #[error("systemd invocation ID is unavailable")]
    InvocationIdUnavailable,

    #[error("invalid verified post-state observation: {0}")]
    InvalidPostState(String),

    #[error("definition content capture I/O error: {0}")]
    DefinitionContentIo(String),

    #[error("definition content changed while being captured")]
    DefinitionContentMutationDetected,

    #[error("definition content source is not a regular file")]
    DefinitionContentNotRegular,

    #[error("definition content source uses a trailing symbolic link")]
    DefinitionContentSymlink,

    #[error("systemd definition identity changed during content capture")]
    DefinitionIdentityChanged,

    #[error("definition content exceeds capture size limit")]
    DefinitionContentTooLarge,

    #[error("systemd reports that this unit needs daemon reload")]
    DefinitionNeedsDaemonReload,
}

/// A one-shot, pre-armed watcher for the systemd Manager.JobRemoved signal.
///
/// Construction completes only after zbus has registered the Manager/JobRemoved
/// match rule. Consuming this value waits for exactly one correlated terminal
/// observation, so a caller cannot accidentally reuse the watcher for another
/// effect. The captured systemd manager unique owner is part of the watcher
/// epoch and must match the exact live Job handle.
#[must_use]
pub struct NixSystemdJobRemovedWatcherV1 {
    manager_owner: String,
    stream: zbus::SignalStream<'static>,
}

impl std::fmt::Debug for NixSystemdJobRemovedWatcherV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NixSystemdJobRemovedWatcherV1")
            .field("manager_owner", &self.manager_owner)
            .finish_non_exhaustive()
    }
}

impl NixSystemdJobRemovedWatcherV1 {
    pub fn manager_owner(&self) -> &str {
        &self.manager_owner
    }

    /// Consume this one-shot watcher and accept only the exact JobRemoved
    /// tuple belonging to the captured live Job and manager incarnation.
    pub async fn await_job_removed(
        mut self,
        expected: &NixSystemdJobHandleV1,
        timeout: Duration,
    ) -> Result<NixSystemdJobEvidenceV1, NixSystemdObserverErrorV1> {
        expected.validate()?;
        if expected.manager_owner != self.manager_owner {
            return Err(NixSystemdObserverErrorV1::WatcherManagerOwnerMismatch);
        }

        let result = tokio::time::timeout(timeout, async {
            while let Some(message) = self.stream.next().await {
                validate_manager_signal_sender(&message, &self.manager_owner)?;
                let removed = decode_job_removed(&message)?;
                if removed.id == expected.id
                    && removed.object_path.as_str() == expected.object_path.as_str()
                    && removed.unit == expected.unit
                {
                    return Ok(NixSystemdJobEvidenceV1 {
                        id: removed.id,
                        job_type: expected.job_type,
                        unit: removed.unit,
                        object_path: removed.object_path.as_str().to_string(),
                        result: removed.result,
                        manager_owner: self.manager_owner.clone(),
                    });
                }
            }
            Err(NixSystemdObserverErrorV1::JobRemovedTimeout)
        })
        .await
        .map_err(|_| NixSystemdObserverErrorV1::JobRemovedTimeout)??;

        Ok(result)
    }
}

/// A read-only systemd D-Bus observer.
///
/// The underlying connection is private so callers cannot obtain a general
/// D-Bus proxy from this type and accidentally reach a mutation method.
#[derive(Clone)]
pub struct NixSystemdReadOnlyObserverV1 {
    connection: Connection,
}

impl NixSystemdReadOnlyObserverV1 {
    pub async fn connect_system() -> Result<Self, NixSystemdObserverErrorV1> {
        Ok(Self {
            connection: Connection::system().await?,
        })
    }

    /// Construct an observer around an existing D-Bus connection.
    ///
    /// This is intended for controlled daemon composition and deterministic
    /// transport tests. The observer still exposes only its read-only API.
    pub fn from_connection(connection: Connection) -> Self {
        Self { connection }
    }

    /// Resolve one validated service unit through Manager.GetUnit.
    pub async fn resolve_service_unit(
        &self,
        unit: &str,
    ) -> Result<OwnedObjectPath, NixSystemdObserverErrorV1> {
        let unit = canonical_unit(unit)?;
        let manager = Proxy::new(
            &self.connection,
            SYSTEMD_DESTINATION,
            SYSTEMD_MANAGER_PATH,
            SYSTEMD_MANAGER_INTERFACE,
        )
        .await?;
        let object_path: OwnedObjectPath = manager.call("GetUnit", &(unit.as_str(),)).await?;
        validate_unit_object_path(&object_path)?;
        Ok(object_path)
    }

    /// Read the exact required Unit projection and Service.Result, then seal
    /// the observation so only the observer boundary can supply it to receipts.
    ///
    /// The generation is separate NixOS provenance supplied by the generation
    /// observer; systemd does not provide it.
    pub async fn observe_service_post_state(
        &self,
        operation: NixServiceOperationKindV1,
        unit: &str,
        generation: u64,
    ) -> Result<NixVerifiedPostStateObservationV1, NixSystemdObserverErrorV1> {
        self.observe_service_post_state_internal(operation, unit, generation, None)
            .await
    }

    /// Observe a completed effect using a watcher that was armed before
    /// mutation dispatch.
    ///
    /// This is the governed lifecycle path. The caller must arm the watcher
    /// before dispatching the effect, then capture the exact live Job handle,
    /// then pass both here.
    pub async fn observe_service_post_state_for_prearmed_job(
        &self,
        operation: NixServiceOperationKindV1,
        unit: &str,
        generation: u64,
        watcher: NixSystemdJobRemovedWatcherV1,
        job: &NixSystemdJobHandleV1,
        timeout: Duration,
    ) -> Result<NixVerifiedPostStateObservationV1, NixSystemdObserverErrorV1> {
        job.validate()?;
        let expected_job_type = NixSystemdJobTypeV1::for_operation(operation)
            .ok_or(NixSystemdObserverErrorV1::JobCorrelationMismatch)?;
        if job.job_type != expected_job_type {
            return Err(NixSystemdObserverErrorV1::JobCorrelationMismatch);
        }
        let completed_job = watcher.await_job_removed(job, timeout).await?;
        self.observe_service_post_state_internal(
            operation,
            unit,
            generation,
            Some(completed_job),
        )
        .await
    }

    /// Legacy convenience method.
    ///
    /// The watcher is armed only after the live Job handle is already known, so
    /// a sufficiently fast job can still emit JobRemoved before subscription.
    /// Governed callers must use arm_job_removed_watcher() before dispatch and
    /// observe_service_post_state_for_prearmed_job() afterward.
    pub async fn observe_service_post_state_for_completed_job(
        &self,
        operation: NixServiceOperationKindV1,
        unit: &str,
        generation: u64,
        job: &NixSystemdJobHandleV1,
        timeout: Duration,
    ) -> Result<NixVerifiedPostStateObservationV1, NixSystemdObserverErrorV1> {
        let watcher = self.arm_job_removed_watcher().await?;
        self.observe_service_post_state_for_prearmed_job(
            operation,
            unit,
            generation,
            watcher,
            job,
            timeout,
        )
        .await
    }

    /// Read Service.Result from the exact resolved unit object.
    pub async fn read_service_result(
        &self,
        object_path: &OwnedObjectPath,
    ) -> Result<String, NixSystemdObserverErrorV1> {
        validate_unit_object_path(object_path)?;
        let properties = self
            .get_all_properties(object_path, SYSTEMD_SERVICE_INTERFACE)
            .await?;
        let result = required_string(&properties, SYSTEMD_SERVICE_INTERFACE, "Result")?;
        if result.is_empty() {
            return Err(NixSystemdObserverErrorV1::InvalidPropertyValue {
                interface: SYSTEMD_SERVICE_INTERFACE,
                property: "Result",
            });
        }
        Ok(result)
    }

    /// Capture the exact bytes referenced by the current systemd definition identity.
    ///
    /// The result is observer-sealed and therefore suitable for construction of
    /// an authority-bound service-effect context. The capture deliberately does
    /// not persist raw bytes; it persists only per-file byte lengths and BLAKE3
    /// commitments.
    ///
    /// Each file is opened read-only with O_NOFOLLOW on Linux, hashed twice from
    /// the same descriptor, and checked with descriptor metadata before/after.
    /// The systemd definition identity and manager incarnation are re-read after
    /// capture. Any detected mutation or identity rollover fails closed.
    pub async fn capture_service_definition_content(
        &self,
        unit: &str,
    ) -> Result<NixVerifiedServiceDefinitionContentV1, NixSystemdObserverErrorV1> {
        let expected_unit = canonical_unit(unit)?;
        let manager_owner = self.systemd_manager_owner().await?;
        let bus_id = self.dbus_bus_id().await?;
        let object_path = self.resolve_service_unit(&expected_unit).await?;
        let (identity, need_daemon_reload) = self
            .read_definition_identity(&object_path, &expected_unit)
            .await?;
        if need_daemon_reload {
            return Err(NixSystemdObserverErrorV1::DefinitionNeedsDaemonReload);
        }

        let source_identity_digest = identity
            .digest(&expected_unit)
            .map_err(|error| NixSystemdObserverErrorV1::InvalidPostState(error.to_string()))?;

        let files = read_definition_content_files(&identity)?;
        if files.is_empty() {
            return Err(NixSystemdObserverErrorV1::DefinitionContentIo(
                "systemd definition has no readable source files".into(),
            ));
        }

        let post_owner = self.systemd_manager_owner().await?;
        let post_bus_id = self.dbus_bus_id().await?;
        let post_object_path = self.resolve_service_unit(&expected_unit).await?;
        let (post_identity, post_need_daemon_reload) = self
            .read_definition_identity(&post_object_path, &expected_unit)
            .await?;
        if post_need_daemon_reload {
            return Err(NixSystemdObserverErrorV1::DefinitionNeedsDaemonReload);
        }
        let post_identity_digest = post_identity
            .digest(&expected_unit)
            .map_err(|error| NixSystemdObserverErrorV1::InvalidPostState(error.to_string()))?;
        let final_owner = self.systemd_manager_owner().await?;

        if post_bus_id != bus_id {
            return Err(NixSystemdObserverErrorV1::ManagerOwnerChanged);
        }
        if post_owner != manager_owner || final_owner != manager_owner {
            return Err(NixSystemdObserverErrorV1::ManagerOwnerChanged);
        }
        if post_object_path.as_str() != object_path.as_str()
            || post_identity_digest != source_identity_digest
        {
            return Err(NixSystemdObserverErrorV1::DefinitionIdentityChanged);
        }

        let evidence = NixServiceDefinitionContentEvidenceV1 {
            unit: expected_unit,
            source_identity_digest,
            manager_owner,
            bus_id,
            files,
            captured_at_monotonic_us: monotonic_now_us()?,
        };

        NixVerifiedServiceDefinitionContentV1::from_observer(evidence)
            .map_err(|error| NixSystemdObserverErrorV1::InvalidPostState(error.to_string()))
    }

    async fn dbus_bus_id(&self) -> Result<String, NixSystemdObserverErrorV1> {
        let bus = Proxy::new(
            &self.connection,
            DBUS_DESTINATION,
            DBUS_PATH,
            DBUS_INTERFACE,
        )
        .await?;
        let id: String = bus.call("GetId", &()).await?;
        if id.len() != 32 || !id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(NixSystemdObserverErrorV1::InvalidPostState(
                "D-Bus GetId returned an invalid bus identifier".into(),
            ));
        }
        Ok(id)
    }

    async fn read_definition_identity(
        &self,
        object_path: &OwnedObjectPath,
        expected_unit: &str,
    ) -> Result<(NixSystemdUnitDefinitionIdentityV1, bool), NixSystemdObserverErrorV1> {
        validate_unit_object_path(object_path)?;
        let properties = self
            .get_all_properties(object_path, SYSTEMD_UNIT_INTERFACE)
            .await?;
        let identity = build_definition_identity_from_properties(&properties, expected_unit)?;
        let need_daemon_reload =
            required_bool(&properties, SYSTEMD_UNIT_INTERFACE, "NeedDaemonReload")?;
        Ok((identity, need_daemon_reload))
    }

    /// Arm the JobRemoved observation channel before any effect is dispatched.
    ///
    /// zbus registers a Manager/JobRemoved match rule before this method returns.
    /// The systemd unique owner is captured before registration and checked again
    /// afterward; an owner transition during arming invalidates the watcher.
    pub async fn arm_job_removed_watcher(
        &self,
    ) -> Result<NixSystemdJobRemovedWatcherV1, NixSystemdObserverErrorV1> {
        let manager_owner = self.systemd_manager_owner().await?;
        let manager = Proxy::new(
            &self.connection,
            SYSTEMD_DESTINATION,
            SYSTEMD_MANAGER_PATH,
            SYSTEMD_MANAGER_INTERFACE,
        )
        .await?;
        let stream: zbus::SignalStream<'static> = manager.receive_signal("JobRemoved").await?;

        let post_arm_owner = self.systemd_manager_owner().await?;
        if post_arm_owner != manager_owner {
            return Err(NixSystemdObserverErrorV1::ManagerOwnerChanged);
        }

        Ok(NixSystemdJobRemovedWatcherV1 {
            manager_owner,
            stream,
        })
    }

    /// Observe the service at least twice across a real monotonic window.
    ///
    /// The returned stability token is observer-sealed, so callers cannot
    /// construct proof evidence by merely populating matching timestamps.
    pub async fn observe_service_stability_window(
        &self,
        operation: NixServiceOperationKindV1,
        unit: &str,
        generation: u64,
        required_window_us: u64,
    ) -> Result<NixVerifiedPostStateStabilityEvidenceV1, NixSystemdObserverErrorV1> {
        if required_window_us == 0 {
            return Err(NixSystemdObserverErrorV1::InvalidPostState(
                "required stability window must be non-zero".to_string(),
            ));
        }

        let first = self
            .observe_service_post_state(operation, unit, generation)
            .await?;
        let first_at = first.as_ref().observed_at_monotonic_us;

        tokio::time::sleep(Duration::from_micros(required_window_us)).await;

        let second = self
            .observe_service_post_state(operation, unit, generation)
            .await?;
        let second_at = second.as_ref().observed_at_monotonic_us;

        let samples = vec![
            stability_sample_from_observation(first.as_ref())?,
            stability_sample_from_observation(second.as_ref())?,
        ];
        let sequence_digest = super::post_state::stability_sequence_digest(&samples)
            .map_err(|error| NixSystemdObserverErrorV1::InvalidPostState(error.to_string()))?;

        let evidence = NixPostStateStabilityEvidenceV1 {
            required_window_us,
            window_start_monotonic_us: first_at,
            window_end_monotonic_us: second_at,
            samples,
            sequence_digest,
        };

        NixVerifiedPostStateStabilityEvidenceV1::from_observer(evidence)
            .map_err(|error| NixSystemdObserverErrorV1::InvalidPostState(error.to_string()))
    }

    /// Capture a still-live Job identity from the exact Job object.
    ///
    /// JobType is not present in JobRemoved, so it must be captured before
    /// completion and carried forward as part of the correlation identity.
    pub async fn capture_job(
        &self,
        job_object_path: &OwnedObjectPath,
        expected_unit: &str,
    ) -> Result<NixSystemdJobHandleV1, NixSystemdObserverErrorV1> {
        let expected_unit = canonical_unit(expected_unit)?;
        validate_job_object_path(job_object_path)?;

        let manager_owner = self.systemd_manager_owner().await?;
        let properties = self
            .get_all_properties(job_object_path, SYSTEMD_JOB_INTERFACE)
            .await?;
        let id = required_u32(&properties, SYSTEMD_JOB_INTERFACE, "Id")?;
        if id == 0 {
            return Err(NixSystemdObserverErrorV1::InvalidJobIdentity(
                "job Id must be non-zero".to_string(),
            ));
        }

        let (job_unit, job_unit_path) = required_job_unit(&properties)?;
        let canonical_job_unit = canonical_unit(&job_unit)?;
        validate_unit_object_path(&job_unit_path)?;
        if canonical_job_unit != expected_unit {
            return Err(NixSystemdObserverErrorV1::JobUnitMismatch {
                expected: expected_unit,
                observed: canonical_job_unit,
            });
        }

        if job_unit_path.as_str() != self.resolve_service_unit(&expected_unit).await?.as_str() {
            return Err(NixSystemdObserverErrorV1::JobCorrelationMismatch);
        }

        let job_type = parse_job_type(required_string(
            &properties,
            SYSTEMD_JOB_INTERFACE,
            "JobType",
        )?)?;

        if !job_object_path.as_str().ends_with(&format!("/{id}")) {
            return Err(NixSystemdObserverErrorV1::InvalidJobIdentity(
                "job object path does not encode captured Id".to_string(),
            ));
        }

        let post_capture_owner = self.systemd_manager_owner().await?;
        if post_capture_owner != manager_owner {
            return Err(NixSystemdObserverErrorV1::ManagerOwnerChanged);
        }

        Ok(NixSystemdJobHandleV1 {
            id,
            job_type,
            unit: canonical_job_unit,
            object_path: job_object_path.clone(),
            unit_object_path: job_unit_path,
            manager_owner,
        })
    }

    /// Legacy convenience wrapper.
    ///
    /// New governed execution must call `arm_job_removed_watcher()` before
    /// mutation dispatch. This wrapper is retained only for compatibility and
    /// intentionally preserves the older post-hoc subscription race.
    pub async fn await_job_removed(
        &self,
        expected: &NixSystemdJobHandleV1,
        timeout: Duration,
    ) -> Result<NixSystemdJobEvidenceV1, NixSystemdObserverErrorV1> {
        let watcher = self.arm_job_removed_watcher().await?;
        watcher.await_job_removed(expected, timeout).await
    }

    /// Resolve a non-zero systemd InvocationID (D-Bus ay) to its exact unit.
    ///
    /// The caller supplies bytes so no string parsing is used at the D-Bus
    /// boundary. The returned object is re-observed and must name the expected
    /// canonical service unit.
    pub async fn resolve_invocation_id(
        &self,
        invocation_id: &[u8],
        expected_unit: &str,
    ) -> Result<OwnedObjectPath, NixSystemdObserverErrorV1> {
        if invocation_id.len() != INVOCATION_ID_BYTES {
            return Err(NixSystemdObserverErrorV1::InvalidInvocationId);
        }
        if invocation_id.iter().all(|byte| *byte == 0) {
            return Err(NixSystemdObserverErrorV1::InvocationIdUnavailable);
        }

        let expected_unit = canonical_unit(expected_unit)?;
        let manager = Proxy::new(
            &self.connection,
            SYSTEMD_DESTINATION,
            SYSTEMD_MANAGER_PATH,
            SYSTEMD_MANAGER_INTERFACE,
        )
        .await?;
        let path: OwnedObjectPath = manager
            .call("GetUnitByInvocationID", &(invocation_id.to_vec(),))
            .await?;
        validate_unit_object_path(&path)?;

        let properties = self
            .get_all_properties(&path, SYSTEMD_UNIT_INTERFACE)
            .await?;
        let observed_id = canonical_unit(&required_string(
            &properties,
            SYSTEMD_UNIT_INTERFACE,
            "Id",
        )?)?;
        if observed_id != expected_unit {
            return Err(NixSystemdObserverErrorV1::UnitIdentityMismatch {
                requested: expected_unit,
                observed: observed_id,
            });
        }
        let observed_invocation = required_invocation_id(&properties)?
            .ok_or(NixSystemdObserverErrorV1::InvocationIdUnavailable)?;
        let expected_invocation = hex::encode(invocation_id);
        if observed_invocation != expected_invocation {
            return Err(NixSystemdObserverErrorV1::InvalidInvocationId);
        }
        Ok(path)
    }

    async fn observe_service_post_state_internal(
        &self,
        operation: NixServiceOperationKindV1,
        unit: &str,
        generation: u64,
        job: Option<NixSystemdJobEvidenceV1>,
    ) -> Result<NixVerifiedPostStateObservationV1, NixSystemdObserverErrorV1> {
        if generation == 0 {
            return Err(NixSystemdObserverErrorV1::InvalidPropertyValue {
                interface: SYSTEMD_UNIT_INTERFACE,
                property: "NixOS generation",
            });
        }

        let expected_unit = canonical_unit(unit)?;
        let manager_owner = self.systemd_manager_owner().await?;
        let object_path = self.resolve_service_unit(&expected_unit).await?;
        let unit_properties = self
            .get_all_properties(&object_path, SYSTEMD_UNIT_INTERFACE)
            .await?;
        let service_properties = self
            .get_all_properties(&object_path, SYSTEMD_SERVICE_INTERFACE)
            .await?;
        let post_manager_owner = self.systemd_manager_owner().await?;
        if post_manager_owner != manager_owner {
            return Err(NixSystemdObserverErrorV1::ManagerOwnerChanged);
        }

        let service_result =
            required_string(&service_properties, SYSTEMD_SERVICE_INTERFACE, "Result")?;
        if service_result.trim().is_empty() {
            return Err(NixSystemdObserverErrorV1::InvalidPropertyValue {
                interface: SYSTEMD_SERVICE_INTERFACE,
                property: "Result",
            });
        }

        let observation = build_observation_from_properties(
            operation,
            &expected_unit,
            generation,
            &object_path,
            &manager_owner,
            &service_result,
            &unit_properties,
            job,
        )?;

        NixVerifiedPostStateObservationV1::from_observer(observation)
            .map_err(|error| NixSystemdObserverErrorV1::InvalidPostState(error.to_string()))
    }

    async fn systemd_manager_owner(&self) -> Result<String, NixSystemdObserverErrorV1> {
        let bus = Proxy::new(
            &self.connection,
            DBUS_DESTINATION,
            DBUS_PATH,
            DBUS_INTERFACE,
        )
        .await?;
        let owner: String = bus
            .call("GetNameOwner", &(SYSTEMD_DESTINATION,))
            .await?;
        validate_unique_owner(&owner)?;
        Ok(owner)
    }

    async fn get_all_properties(
        &self,
        object_path: &OwnedObjectPath,
        interface: &'static str,
    ) -> Result<HashMap<String, OwnedValue>, NixSystemdObserverErrorV1> {
        validate_unit_object_path(object_path)?;
        let properties =
            PropertiesProxy::new(&self.connection, SYSTEMD_DESTINATION, object_path.clone())
                .await?;
        Ok(properties.get_all(interface).await?)
    }
}

/// Immutable correlation identity for one systemd job captured while live.
///
/// Fields are private so callers cannot fabricate a handle and feed it into
/// the trusted JobRemoved correlation path.
pub struct NixSystemdJobHandleV1 {
    id: u32,
    job_type: NixSystemdJobTypeV1,
    unit: String,
    object_path: OwnedObjectPath,
    unit_object_path: OwnedObjectPath,
    manager_owner: String,
}

impl std::fmt::Debug for NixSystemdJobHandleV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NixSystemdJobHandleV1")
            .field("id", &self.id)
            .field("job_type", &self.job_type)
            .field("unit", &self.unit)
            .field("object_path", &self.object_path)
            .field("unit_object_path", &self.unit_object_path)
            .field("manager_owner", &self.manager_owner)
            .finish()
    }
}

impl NixSystemdJobHandleV1 {
    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn job_type(&self) -> NixSystemdJobTypeV1 {
        self.job_type
    }

    pub fn unit(&self) -> &str {
        &self.unit
    }

    pub fn object_path(&self) -> &str {
        self.object_path.as_str()
    }

    pub fn unit_object_path(&self) -> &str {
        self.unit_object_path.as_str()
    }

    pub fn manager_owner(&self) -> &str {
        &self.manager_owner
    }

    fn validate(&self) -> Result<(), NixSystemdObserverErrorV1> {
        if self.id == 0 {
            return Err(NixSystemdObserverErrorV1::InvalidJobIdentity(
                "job Id must be non-zero".to_string(),
            ));
        }
        let canonical = canonical_unit(&self.unit)?;
        if canonical != self.unit {
            return Err(NixSystemdObserverErrorV1::InvalidJobIdentity(
                "job unit is not canonical".to_string(),
            ));
        }
        validate_job_object_path(&self.object_path)?;
        validate_unit_object_path(&self.unit_object_path)?;
        validate_unique_owner(&self.manager_owner)?;
        if !self.object_path.as_str().ends_with(&format!("/{}", self.id)) {
            return Err(NixSystemdObserverErrorV1::InvalidJobIdentity(
                "job object path/Id mismatch".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_unique_owner(owner: &str) -> Result<(), NixSystemdObserverErrorV1> {
    zbus::names::UniqueName::try_from(owner)
        .map(|_| ())
        .map_err(|_| {
            NixSystemdObserverErrorV1::InvalidJobIdentity(
                "invalid systemd unique bus owner".to_string(),
            )
        })
}

fn validate_manager_signal_sender(
    message: &Message,
    expected_owner: &str,
) -> Result<(), NixSystemdObserverErrorV1> {
    let actual = message
        .header()
        .sender()
        .ok_or_else(|| NixSystemdObserverErrorV1::InvalidJobIdentity(
            "JobRemoved signal has no sender".to_string(),
        ))?
        .to_string();
    if actual != expected_owner {
        return Err(NixSystemdObserverErrorV1::JobCorrelationMismatch);
    }
    Ok(())
}

fn canonical_unit(unit: &str) -> Result<String, NixSystemdObserverErrorV1> {
    NixServiceOperationV1::new(unit, NixServiceOperationKindV1::Start)
        .map(|operation| operation.unit().to_string())
        .map_err(|error| NixSystemdObserverErrorV1::InvalidServiceUnit(error.to_string()))
}

fn validate_unit_object_path(
    path: &OwnedObjectPath,
) -> Result<(), NixSystemdObserverErrorV1> {
    let value = path.as_str();
    if value.is_empty() || value.len() > 4096 || !value.starts_with(SYSTEMD_UNIT_PATH_PREFIX) {
        return Err(NixSystemdObserverErrorV1::InvalidJobIdentity(
            "unexpected systemd unit object path".to_string(),
        ));
    }
    Ok(())
}

fn validate_job_object_path(
    path: &OwnedObjectPath,
) -> Result<(), NixSystemdObserverErrorV1> {
    let value = path.as_str();
    if value.is_empty() || value.len() > 4096 || !value.starts_with(SYSTEMD_JOB_PATH_PREFIX) {
        return Err(NixSystemdObserverErrorV1::InvalidJobIdentity(
            "unexpected systemd job object path".to_string(),
        ));
    }
    Ok(())
}

fn required_value<'a>(
    properties: &'a HashMap<String, OwnedValue>,
    interface: &'static str,
    property: &'static str,
) -> Result<&'a OwnedValue, NixSystemdObserverErrorV1> {
    properties
        .get(property)
        .ok_or(NixSystemdObserverErrorV1::MissingProperty { interface, property })
}

fn required_string(
    properties: &HashMap<String, OwnedValue>,
    interface: &'static str,
    property: &'static str,
) -> Result<String, NixSystemdObserverErrorV1> {
    required_value(properties, interface, property)?
        .clone()
        .try_into()
        .map_err(|_| NixSystemdObserverErrorV1::InvalidPropertyType {
            interface,
            property,
        })
}

fn required_bool(
    properties: &HashMap<String, OwnedValue>,
    interface: &'static str,
    property: &'static str,
) -> Result<bool, NixSystemdObserverErrorV1> {
    required_value(properties, interface, property)?
        .clone()
        .try_into()
        .map_err(|_| NixSystemdObserverErrorV1::InvalidPropertyType {
            interface,
            property,
        })
}

fn required_u64(
    properties: &HashMap<String, OwnedValue>,
    interface: &'static str,
    property: &'static str,
) -> Result<u64, NixSystemdObserverErrorV1> {
    required_value(properties, interface, property)?
        .clone()
        .try_into()
        .map_err(|_| NixSystemdObserverErrorV1::InvalidPropertyType {
            interface,
            property,
        })
}

fn required_u32(
    properties: &HashMap<String, OwnedValue>,
    interface: &'static str,
    property: &'static str,
) -> Result<u32, NixSystemdObserverErrorV1> {
    required_value(properties, interface, property)?
        .clone()
        .try_into()
        .map_err(|_| NixSystemdObserverErrorV1::InvalidPropertyType {
            interface,
            property,
        })
}

fn required_strings(
    properties: &HashMap<String, OwnedValue>,
    interface: &'static str,
    property: &'static str,
) -> Result<Vec<String>, NixSystemdObserverErrorV1> {
    required_value(properties, interface, property)?
        .clone()
        .try_into()
        .map_err(|_| NixSystemdObserverErrorV1::InvalidPropertyType {
            interface,
            property,
        })
}

fn required_bytes(
    properties: &HashMap<String, OwnedValue>,
    interface: &'static str,
    property: &'static str,
) -> Result<Vec<u8>, NixSystemdObserverErrorV1> {
    required_value(properties, interface, property)?
        .clone()
        .try_into()
        .map_err(|_| NixSystemdObserverErrorV1::InvalidPropertyType {
            interface,
            property,
        })
}

fn required_job_unit(
    properties: &HashMap<String, OwnedValue>,
) -> Result<(String, OwnedObjectPath), NixSystemdObserverErrorV1> {
    required_value(properties, SYSTEMD_JOB_INTERFACE, "Unit")?
        .clone()
        .try_into()
        .map_err(|_| NixSystemdObserverErrorV1::InvalidPropertyType {
            interface: SYSTEMD_JOB_INTERFACE,
            property: "Unit",
        })
}

fn required_invocation_id(
    properties: &HashMap<String, OwnedValue>,
) -> Result<Option<String>, NixSystemdObserverErrorV1> {
    invocation_id_to_string(required_bytes(
        properties,
        SYSTEMD_UNIT_INTERFACE,
        "InvocationID",
    )?)
}

fn invocation_id_to_string(
    bytes: Vec<u8>,
) -> Result<Option<String>, NixSystemdObserverErrorV1> {
    if bytes.len() != INVOCATION_ID_BYTES {
        return Err(NixSystemdObserverErrorV1::InvalidInvocationId);
    }
    if bytes.iter().all(|byte| *byte == 0) {
        Ok(None)
    } else {
        Ok(Some(hex::encode(bytes)))
    }
}

fn parse_job_type(value: String) -> Result<NixSystemdJobTypeV1, NixSystemdObserverErrorV1> {
    match value.as_str() {
        "start" => Ok(NixSystemdJobTypeV1::Start),
        "stop" => Ok(NixSystemdJobTypeV1::Stop),
        "restart" => Ok(NixSystemdJobTypeV1::Restart),
        "reload" => Ok(NixSystemdJobTypeV1::Reload),
        other => Err(NixSystemdObserverErrorV1::UnknownStateVocabulary(
            other.to_string(),
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DefinitionFileObjectIdentity {
    dev: u64,
    ino: u64,
    mode: u32,
    size: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ctime_sec: i64,
    ctime_nsec: i64,
}

#[cfg(unix)]
fn definition_file_identity(file: &std::fs::File) -> Result<DefinitionFileObjectIdentity, NixSystemdObserverErrorV1> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file
        .metadata()
        .map_err(|error| NixSystemdObserverErrorV1::DefinitionContentIo(error.to_string()))?;
    if !metadata.is_file() {
        return Err(NixSystemdObserverErrorV1::DefinitionContentNotRegular);
    }
    Ok(DefinitionFileObjectIdentity {
        dev: metadata.dev(),
        ino: metadata.ino(),
        mode: metadata.mode(),
        size: metadata.size(),
        mtime_sec: metadata.mtime(),
        mtime_nsec: metadata.mtime_nsec(),
        ctime_sec: metadata.ctime(),
        ctime_nsec: metadata.ctime_nsec(),
    })
}

#[cfg(unix)]
fn hash_open_definition_file(
    file: &mut std::fs::File,
) -> Result<(u64, String), NixSystemdObserverErrorV1> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| NixSystemdObserverErrorV1::DefinitionContentIo(error.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut length = 0u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| NixSystemdObserverErrorV1::DefinitionContentIo(error.to_string()))?;
        if read == 0 {
            break;
        }
        length = length
            .checked_add(read as u64)
            .ok_or(NixSystemdObserverErrorV1::DefinitionContentTooLarge)?;
        if length > 8 * 1024 * 1024 {
            return Err(NixSystemdObserverErrorV1::DefinitionContentTooLarge);
        }
        hasher.update(&buffer[..read]);
    }
    Ok((length, hasher.finalize().to_hex().to_string()))
}

#[cfg(unix)]
fn read_definition_content_file(
    path: &str,
) -> Result<NixSystemdUnitDefinitionContentFileV1, NixSystemdObserverErrorV1> {
    use std::os::unix::fs::OpenOptionsExt;

    let mut options = OpenOptions::new();
    options.read(true);
    let mut file = options
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| {
            if error.raw_os_error() == Some(libc::ELOOP) {
                NixSystemdObserverErrorV1::DefinitionContentSymlink
            } else {
                NixSystemdObserverErrorV1::DefinitionContentIo(error.to_string())
            }
        })?;

    let before = definition_file_identity(&file)?;
    let (first_len, first_digest) = hash_open_definition_file(&mut file)?;
    let middle = definition_file_identity(&file)?;
    let (second_len, second_digest) = hash_open_definition_file(&mut file)?;
    let after = definition_file_identity(&file)?;

    if before != middle || middle != after || first_len != second_len || first_digest != second_digest {
        return Err(NixSystemdObserverErrorV1::DefinitionContentMutationDetected);
    }

    Ok(NixSystemdUnitDefinitionContentFileV1 {
        path: path.to_string(),
        byte_len: second_len,
        content_digest: second_digest,
    })
}

#[cfg(not(unix))]
fn read_definition_content_file(
    _path: &str,
) -> Result<NixSystemdUnitDefinitionContentFileV1, NixSystemdObserverErrorV1> {
    Err(NixSystemdObserverErrorV1::DefinitionContentIo(
        "definition content capture requires a Unix file-descriptor API".into(),
    ))
}

fn read_definition_content_files(
    identity: &NixSystemdUnitDefinitionIdentityV1,
) -> Result<Vec<NixSystemdUnitDefinitionContentFileV1>, NixSystemdObserverErrorV1> {
    let mut paths = Vec::with_capacity(1 + identity.drop_in_paths.len());
    paths.push(identity.fragment_path.clone());
    paths.extend(identity.drop_in_paths.iter().cloned());

    let mut total = 0u64;
    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        let file = read_definition_content_file(&path)?;
        total = total
            .checked_add(file.byte_len)
            .ok_or(NixSystemdObserverErrorV1::DefinitionContentTooLarge)?;
        if total > 64 * 1024 * 1024 {
            return Err(NixSystemdObserverErrorV1::DefinitionContentTooLarge);
        }
        files.push(file);
    }
    Ok(files)
}

fn build_definition_identity_from_properties(
    properties: &HashMap<String, OwnedValue>,
    expected_unit: &str,
) -> Result<NixSystemdUnitDefinitionIdentityV1, NixSystemdObserverErrorV1> {
    let observed_id = canonical_unit(&required_string(
        properties,
        SYSTEMD_UNIT_INTERFACE,
        "Id",
    )?)?;
    let names = required_strings(properties, SYSTEMD_UNIT_INTERFACE, "Names")?;
    let canonical_names = names
        .iter()
        .map(|name| canonical_unit(name))
        .collect::<Result<Vec<_>, _>>()?;
    if observed_id != expected_unit && !canonical_names.iter().any(|name| name == expected_unit) {
        return Err(NixSystemdObserverErrorV1::UnitIdentityMismatch {
            requested: expected_unit.to_string(),
            observed: observed_id,
        });
    }

    let fragment_path = required_string(properties, SYSTEMD_UNIT_INTERFACE, "FragmentPath")?;
    let drop_in_paths = required_strings(properties, SYSTEMD_UNIT_INTERFACE, "DropInPaths")?;
    NixSystemdUnitDefinitionIdentityV1::new(fragment_path, drop_in_paths)
        .map_err(|error| NixSystemdObserverErrorV1::InvalidPostState(error.to_string()))
}

fn stability_sample_from_observation(
    observation: &NixServicePostStateObservationV1,
) -> Result<NixPostStateStabilitySampleV1, NixSystemdObserverErrorV1> {
    let manager_owner = observation
        .systemd_manager_owner
        .as_deref()
        .ok_or(NixSystemdObserverErrorV1::InvalidPostState(
            "stability observation has no systemd manager owner".to_string(),
        ))?;
    Ok(NixPostStateStabilitySampleV1 {
        operation: observation.operation,
        unit: observation.unit.clone(),
        unit_object_path: observation.unit_object_path.clone(),
        observed_generation: observation.observed_generation,
        definition_digest: observation.definition_digest().map_err(|error| {
            NixSystemdObserverErrorV1::InvalidPostState(error.to_string())
        })?,
        state_digest: observation.state_digest().map_err(|error| {
            NixSystemdObserverErrorV1::InvalidPostState(error.to_string())
        })?,
        manager_owner: manager_owner.to_string(),
        invocation_id: observation.invocation_id.clone(),
        state_change_at_monotonic_us: observation.state_change_at_monotonic_us,
        captured_at_monotonic_us: observation.observed_at_monotonic_us,
    })
}

fn build_observation_from_properties(
    operation: NixServiceOperationKindV1,
    expected_unit: &str,
    generation: u64,
    unit_object_path: &OwnedObjectPath,
    manager_owner: &str,
    service_result: &str,
    properties: &HashMap<String, OwnedValue>,
    job: Option<NixSystemdJobEvidenceV1>,
) -> Result<NixServicePostStateObservationV1, NixSystemdObserverErrorV1> {
    for property in REQUIRED_UNIT_PROPERTIES {
        if !properties.contains_key(*property) {
            return Err(NixSystemdObserverErrorV1::MissingProperty {
                interface: SYSTEMD_UNIT_INTERFACE,
                property,
            });
        }
    }

    let observed_id = canonical_unit(&required_string(
        properties,
        SYSTEMD_UNIT_INTERFACE,
        "Id",
    )?)?;
    let names = required_strings(properties, SYSTEMD_UNIT_INTERFACE, "Names")?;
    let canonical_names = names
        .iter()
        .map(|name| canonical_unit(name))
        .collect::<Result<Vec<_>, _>>()?;

    if observed_id != expected_unit && !canonical_names.iter().any(|name| name == expected_unit) {
        return Err(NixSystemdObserverErrorV1::UnitIdentityMismatch {
            requested: expected_unit.to_string(),
            observed: observed_id,
        });
    }

    let load_state = ServiceLoadStateV1::parse(&required_string(
        properties,
        SYSTEMD_UNIT_INTERFACE,
        "LoadState",
    )?)
    .map_err(|_| {
        NixSystemdObserverErrorV1::UnknownStateVocabulary("LoadState".to_string())
    })?;

    let active_state = ServiceActiveStateV1::parse(&required_string(
        properties,
        SYSTEMD_UNIT_INTERFACE,
        "ActiveState",
    )?)
    .map_err(|_| {
        NixSystemdObserverErrorV1::UnknownStateVocabulary("ActiveState".to_string())
    })?;

    let sub_state = required_string(properties, SYSTEMD_UNIT_INTERFACE, "SubState")?;
    if sub_state.is_empty() {
        return Err(NixSystemdObserverErrorV1::InvalidPropertyValue {
            interface: SYSTEMD_UNIT_INTERFACE,
            property: "SubState",
        });
    }

    let unit_file_state = ServiceUnitFileStateV1::parse(&required_string(
        properties,
        SYSTEMD_UNIT_INTERFACE,
        "UnitFileState",
    )?)
    .map_err(|_| {
        NixSystemdObserverErrorV1::UnknownStateVocabulary("UnitFileState".to_string())
    })?;

    let fragment_path = required_string(properties, SYSTEMD_UNIT_INTERFACE, "FragmentPath")?;
    let drop_in_paths =
        required_strings(properties, SYSTEMD_UNIT_INTERFACE, "DropInPaths")?;
    let state_change_at_monotonic_us = required_u64(
        properties,
        SYSTEMD_UNIT_INTERFACE,
        "StateChangeTimestampMonotonic",
    )?;
    let invocation_id = required_invocation_id(properties)?;
    let observed_at_monotonic_us = monotonic_now_us()?;

    let definition_identity =
        NixSystemdUnitDefinitionIdentityV1::new(fragment_path, drop_in_paths)
            .map_err(|error| NixSystemdObserverErrorV1::InvalidPostState(error.to_string()))?;

    if let Some(ref job) = job {
        let expected_job_type = NixSystemdJobTypeV1::for_operation(operation)
            .ok_or(NixSystemdObserverErrorV1::JobCorrelationMismatch)?;
        if job.job_type != expected_job_type || job.unit != expected_unit {
            return Err(NixSystemdObserverErrorV1::JobCorrelationMismatch);
        }
    }

    Ok(NixServicePostStateObservationV1 {
        operation,
        unit: expected_unit.to_string(),
        observed_generation: generation,
        unit_object_path: unit_object_path.as_str().to_string(),
        definition_identity,
        load_state,
        active_state,
        sub_state,
        unit_file_state,
        service_result,
        systemd_job: job,
        systemd_manager_owner: Some(manager_owner.to_string()),
        invocation_id,
        state_change_at_monotonic_us,
        observed_at_monotonic_us,
    })
}

#[derive(Debug, PartialEq, Eq)]
struct NixSystemdJobRemovedTuple {
    id: u32,
    object_path: OwnedObjectPath,
    unit: String,
    result: String,
}

fn decode_job_removed(
    message: &Message,
) -> Result<NixSystemdJobRemovedTuple, NixSystemdObserverErrorV1> {
    let (id, path, unit, result): (u32, OwnedObjectPath, String, String) =
        message.body().deserialize().map_err(|_| {
            NixSystemdObserverErrorV1::InvalidJobIdentity(
                "JobRemoved body is not native uoss".to_string(),
            )
        })?;

    validate_job_object_path(&path)?;
    if id == 0 || !path.as_str().ends_with(&format!("/{id}")) || result.is_empty() {
        return Err(NixSystemdObserverErrorV1::InvalidJobIdentity(
            "invalid JobRemoved tuple".to_string(),
        ));
    }

    Ok(NixSystemdJobRemovedTuple {
        id,
        object_path: path,
        unit: canonical_unit(&unit)?,
        result,
    })
}

fn monotonic_now_us() -> Result<u64, NixSystemdObserverErrorV1> {
    #[cfg(target_os = "linux")]
    {
        use nix::time::{clock_gettime, ClockId};
        let time = clock_gettime(ClockId::CLOCK_MONOTONIC).map_err(|_| {
            NixSystemdObserverErrorV1::InvalidPropertyValue {
                interface: SYSTEMD_UNIT_INTERFACE,
                property: "CLOCK_MONOTONIC",
            }
        })?;
        Ok((time.tv_sec() as u64)
            .saturating_mul(1_000_000)
            .saturating_add((time.tv_nsec() as u64) / 1_000))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err(NixSystemdObserverErrorV1::InvalidPropertyValue {
            interface: SYSTEMD_UNIT_INTERFACE,
            property: "CLOCK_MONOTONIC",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn definition_content_reader_hashes_exact_file_bytes() {
        use std::io::Write;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nginx.service");
        let mut file = std::fs::File::create(&path).unwrap();
        let bytes = b"[Service]\nExecStart=/usr/bin/nginx\n";
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();

        let captured = read_definition_content_file(path.to_str().unwrap()).unwrap();
        assert_eq!(captured.byte_len, bytes.len() as u64);
        assert_eq!(
            captured.content_digest,
            blake3::hash(bytes).to_hex().to_string()
        );
    }

    #[cfg(unix)]
    #[test]
    fn definition_content_reader_rejects_trailing_symlink() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("real.service");
        let link = directory.path().join("linked.service");
        std::fs::write(&target, b"[Service]\n").unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();

        assert!(matches!(
            read_definition_content_file(link.to_str().unwrap()),
            Err(NixSystemdObserverErrorV1::DefinitionContentSymlink)
        ));
    }

    #[test]
    fn invocation_id_is_exactly_16_bytes() {
        assert_eq!(invocation_id_to_string(vec![0; 16]).unwrap(), None);
        assert_eq!(
            invocation_id_to_string(vec![0xab; 16]).unwrap(),
            Some("abababababababababababababababab".into())
        );
        assert!(invocation_id_to_string(vec![0; 15]).is_err());
        assert!(invocation_id_to_string(vec![0; 17]).is_err());
    }

    #[test]
    fn job_type_vocabulary_is_strict() {
        assert_eq!(
            parse_job_type("start".into()).unwrap(),
            NixSystemdJobTypeV1::Start
        );
        assert_eq!(
            parse_job_type("stop".into()).unwrap(),
            NixSystemdJobTypeV1::Stop
        );
        assert_eq!(
            parse_job_type("restart".into()).unwrap(),
            NixSystemdJobTypeV1::Restart
        );
        assert_eq!(
            parse_job_type("reload".into()).unwrap(),
            NixSystemdJobTypeV1::Reload
        );
        assert!(parse_job_type("try-restart".into()).is_err());
    }

    #[test]
    fn job_path_must_encode_id() {
        let valid =
            OwnedObjectPath::try_from("/org/freedesktop/systemd1/job/42").unwrap();
        let bad =
            OwnedObjectPath::try_from("/org/freedesktop/systemd1/job/43").unwrap();
        validate_job_object_path(&valid).unwrap();
        assert!(valid.as_str().ends_with("/42"));
        assert!(!bad.as_str().ends_with("/42"));
    }

    #[test]
    fn unit_path_boundary_is_strict() {
        let valid = OwnedObjectPath::try_from(
            "/org/freedesktop/systemd1/unit/nginx_2eservice",
        )
        .unwrap();
        let bad = OwnedObjectPath::try_from("/org/freedesktop/systemd1/job/42").unwrap();
        validate_unit_object_path(&valid).unwrap();
        assert!(validate_unit_object_path(&bad).is_err());
    }

    #[test]
    fn unique_systemd_manager_owner_is_strict() {
        assert!(validate_unique_owner(":1.42").is_ok());
        assert!(validate_unique_owner(":").is_err());
        assert!(validate_unique_owner("org.freedesktop.systemd1").is_err());
        assert!(validate_unique_owner("").is_err());
        assert!(validate_unique_owner(&"x".repeat(256)).is_err());
    }

    #[test]
    fn job_handle_manager_owner_is_part_of_its_identity() {
        let object_path =
            OwnedObjectPath::try_from("/org/freedesktop/systemd1/job/42").unwrap();
        let unit_object_path =
            OwnedObjectPath::try_from("/org/freedesktop/systemd1/unit/nginx_2eservice").unwrap();

        let valid = NixSystemdJobHandleV1 {
            id: 42,
            job_type: NixSystemdJobTypeV1::Start,
            unit: "nginx.service".to_string(),
            object_path: object_path.clone(),
            unit_object_path: unit_object_path.clone(),
            manager_owner: ":1.42".to_string(),
        };
        valid.validate().unwrap();

        let invalid = NixSystemdJobHandleV1 {
            manager_owner: "org.freedesktop.systemd1".to_string(),
            ..valid
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn manager_signal_sender_mismatch_is_rejected() {
        let message = Message::signal(
            "/org/freedesktop/systemd1",
            "org.freedesktop.systemd1.Manager",
            "JobRemoved",
        )
        .unwrap()
        .build(&(42u32, "/org/freedesktop/systemd1/job/42", "nginx.service", "done"))
        .unwrap();
        // The sender field is absent on a locally constructed message. The
        // verifier must fail closed rather than treating absence as trusted.
        assert!(validate_manager_signal_sender(&message, ":1.42").is_err());
    }

    #[test]
    fn unknown_service_state_is_rejected() {
        assert!(ServiceActiveStateV1::parse("future-state").is_err());
        assert!(ServiceUnitFileStateV1::parse("future-state").is_err());
    }
}
