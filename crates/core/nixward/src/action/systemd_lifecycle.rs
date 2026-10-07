// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Ordered, manager/bus-bound systemd lifecycle transaction waist.
//!
//! Transport/orchestration only: this module does not authorize effects, mint
//! execution witnesses, or inspect arbitrary filesystem paths.

use super::post_state::{NixSystemdJobEvidenceV1, NixSystemdJobTypeV1};
use super::service_domain::{NixServiceOperationKindV1, NixServiceOperationV1};
use super::systemd_mutation::{
    NixSystemdLifecycleMutationTransportV1, NixSystemdMutationTransportErrorV1,
};
use super::systemd_observer::{
    NixSystemdObserverErrorV1, NixSystemdReadOnlyObserverV1,
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

    /// Execute the only supported lifecycle transport order:
    /// watch -> epoch check -> dispatch -> capture -> await.
    pub async fn dispatch_and_observe_for_epoch(
        &self,
        operation: NixServiceOperationKindV1,
        unit: &str,
        expected_manager_owner: &str,
        expected_bus_id: &str,
        timeout: Duration,
    ) -> Result<NixSystemdLifecycleEvidenceV2, NixSystemdLifecycleTransactionErrorV2> {
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

        let job_path = self
            .mutation
            .dispatch_lifecycle_for_manager_owner_and_bus_id(
                &operation,
                expected_manager_owner,
                expected_bus_id,
            )
            .await?;

        let job = self
            .observer
            .capture_dispatched_job(
                &job_path,
                operation.operation(),
                operation.unit(),
                expected_manager_owner,
                expected_bus_id,
            )
            .await?;

        if job.job_type() != expected_job_type {
            return Err(NixSystemdLifecycleTransactionErrorV2::JobTypeMismatch);
        }

        let job = watcher.await_job_removed(&job, timeout).await?;
        if job.manager_owner != expected_manager_owner {
            return Err(NixSystemdLifecycleTransactionErrorV2::ManagerOwnerMismatch);
        }

        Ok(NixSystemdLifecycleEvidenceV2 {
            operation: operation.operation(),
            unit: operation.unit().to_string(),
            job,
            manager_owner: expected_manager_owner.to_string(),
            bus_id: expected_bus_id.to_string(),
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
    fn lifecycle_transaction_api_is_structurally_narrow() {
        let _ = NixSystemdLifecycleTransactionV2::dispatch_and_observe_for_epoch;
    }
}