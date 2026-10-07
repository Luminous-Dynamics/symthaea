// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Ordered, manager/bus-bound systemd lifecycle transaction waist.
//!
//! Transport/orchestration only: this module does not authorize effects,
//! mint execution witnesses, or inspect arbitrary filesystem paths.

use super::post_state::{NixSystemdJobEvidenceV1, NixSystemdJobTypeV1};
use super::service_domain::{NixServiceOperationKindV1, NixServiceOperationV1};
use super::systemd_mutation::{
    NixSystemdLifecycleMutationTransportV1, NixSystemdMutationTransportErrorV1,
};
use super::systemd_observer::{
    NixSystemdJobRemovedWatcherV1, NixSystemdObserverErrorV1,
    NixSystemdReadOnlyObserverV1,
};
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixSystemdLifecycleEvidenceV2 {
    pub operation: NixServiceOperationKindV1,
    pub unit: String,
    pub job: NixSystemdJobEvidenceV1,
    pub manager_owner: String,
    pub bus_id: String,
}

pub struct NixSystemdLifecycleTransactionV2 {
    observer: NixSystemdReadOnlyObserverV1,
    mutation: NixSystemdLifecycleMutationTransportV1,
}

pub struct NixSystemdPreparedLifecycleTransactionV2 {
    observer: NixSystemdReadOnlyObserverV1,
    mutation: NixSystemdLifecycleMutationTransportV1,
    watcher: NixSystemdJobRemovedWatcherV1,
    operation: NixServiceOperationV1,
    expected_manager_owner: String,
    expected_bus_id: String,
    expected_job_type: NixSystemdJobTypeV1,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NixSystemdLifecycleTransactionErrorV2 {
    #[error("invalid service operation: {0}")]
    InvalidServiceOperation(String),
    #[error("operation is not supported by the native lifecycle Job transport")]
    UnsupportedOperation,
    #[error("systemd watcher manager owner does not match the bound Service epoch")]
    ManagerOwnerMismatch,
    #[error("systemd watcher D-Bus incarnation does not match the bound Service epoch")]
    BusIncarnationMismatch,
    #[error("systemd lifecycle JobType does not match the typed Service operation")]
    JobTypeMismatch,
    #[error("systemd observer failed: {0}")]
    Observer(#[from] NixSystemdObserverErrorV1),
    #[error("systemd mutation transport failed: {0}")]
    Mutation(#[from] NixSystemdMutationTransportErrorV1),
}

impl NixSystemdLifecycleTransactionV2 {
    pub fn new(
        observer: NixSystemdReadOnlyObserverV1,
        mutation: NixSystemdLifecycleMutationTransportV1,
    ) -> Self {
        Self { observer, mutation }
    }

    /// Prepare the lifecycle transaction by arming JobRemoved and binding the
    /// watcher to the exact manager/bus epoch before any mutation occurs.
    ///
    /// The returned prepared transaction owns the watcher, so dispatch cannot
    /// accidentally be attempted without a pre-armed observation path.
    pub async fn prepare_for_epoch(
        self,
        operation: NixServiceOperationKindV1,
        unit: &str,
        expected_manager_owner: &str,
        expected_bus_id: &str,
    ) -> Result<NixSystemdPreparedLifecycleTransactionV2, NixSystemdLifecycleTransactionErrorV2> {
        let operation = NixServiceOperationV1::new(unit.to_string(), operation).map_err(|error| {
            NixSystemdLifecycleTransactionErrorV2::InvalidServiceOperation(error.to_string())
        })?;
        let expected_job_type = NixSystemdJobTypeV1::for_operation(operation.operation())
            .ok_or(NixSystemdLifecycleTransactionErrorV2::UnsupportedOperation)?;

        let watcher = self.observer.arm_job_removed_watcher().await?;
        if watcher.manager_owner() != expected_manager_owner {
            return Err(NixSystemdLifecycleTransactionErrorV2::ManagerOwnerMismatch);
        }
        if watcher.bus_id() != expected_bus_id {
            return Err(NixSystemdLifecycleTransactionErrorV2::BusIncarnationMismatch);
        }

        Ok(NixSystemdPreparedLifecycleTransactionV2 {
            observer: self.observer,
            mutation: self.mutation,
            watcher,
            operation,
            expected_manager_owner: expected_manager_owner.to_string(),
            expected_bus_id: expected_bus_id.to_string(),
            expected_job_type,
        })
    }
}

impl NixSystemdPreparedLifecycleTransactionV2 {
    /// Dispatch only after preparation has sealed the pre-dispatch observation epoch.
    pub async fn dispatch_and_observe(
        self,
        timeout: Duration,
    ) -> Result<NixSystemdLifecycleEvidenceV2, NixSystemdLifecycleTransactionErrorV2> {
        let job_path = self
            .mutation
            .dispatch_lifecycle_for_manager_owner_and_bus_id(
                &self.operation,
                &self.expected_manager_owner,
                &self.expected_bus_id,
            )
            .await?;

        let job = self
            .observer
            .capture_dispatched_job(
                &job_path,
                self.operation.operation(),
                self.operation.unit(),
                &self.expected_manager_owner,
                &self.expected_bus_id,
            )
            .await?;

        if job.job_type() != self.expected_job_type {
            return Err(NixSystemdLifecycleTransactionErrorV2::JobTypeMismatch);
        }

        let job = self.watcher.await_job_removed(&job, timeout).await?;
        if job.manager_owner != self.expected_manager_owner {
            return Err(NixSystemdLifecycleTransactionErrorV2::ManagerOwnerMismatch);
        }

        Ok(NixSystemdLifecycleEvidenceV2 {
            operation: self.operation.operation(),
            unit: self.operation.unit().to_string(),
            job,
            manager_owner: self.expected_manager_owner,
            bus_id: self.expected_bus_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_transaction_rejects_unit_file_operations() {
        assert!(NixSystemdJobTypeV1::for_operation(NixServiceOperationKindV1::Enable).is_none());
        assert!(NixSystemdJobTypeV1::for_operation(NixServiceOperationKindV1::Disable).is_none());
    }

    #[test]
    fn lifecycle_transaction_requires_preparation_before_dispatch() {
        let _ = NixSystemdLifecycleTransactionV2::prepare_for_epoch;
        let _ = NixSystemdPreparedLifecycleTransactionV2::dispatch_and_observe;
    }
}
