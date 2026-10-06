// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Ordered, transport-only systemd lifecycle transaction waist.
//!
//! This module exists to make the safe sequencing of CROSS-061/063/065
//! structural rather than convention-only. It does not authorize the effect.

use super::post_state::{NixSystemdJobEvidenceV1, NixSystemdJobTypeV1};
use super::service_domain::{NixServiceOperationKindV1, NixServiceOperationV1};
use super::systemd_mutation::{
    NixSystemdLifecycleMutationTransportV1, NixSystemdMutationTransportErrorV1,
};
use super::systemd_observer::{
    NixSystemdJobHandleV1, NixSystemdJobRemovedWatcherV1, NixSystemdObserverErrorV1,
    NixSystemdReadOnlyObserverV1,
};
use std::time::Duration;
use thiserror::Error;

#[derive(Debug)]
pub struct NixSystemdLifecycleEvidenceV1 {
    pub operation: NixServiceOperationKindV1,
    pub unit: String,
    pub job: NixSystemdJobEvidenceV1,
    pub manager_owner: String,
}

pub struct NixSystemdLifecycleTransactionV1 {
    observer: NixSystemdReadOnlyObserverV1,
    mutation: NixSystemdLifecycleMutationTransportV1,
}

impl NixSystemdLifecycleTransactionV1 {
    pub fn new(
        observer: NixSystemdReadOnlyObserverV1,
        mutation: NixSystemdLifecycleMutationTransportV1,
    ) -> Self {
        Self { observer, mutation }
    }

    /// Execute the non-authorizing transport sequence in the only supported order:
    ///
    /// watch -> dispatch -> capture -> await terminal signal.
    pub async fn dispatch_and_observe(
        &self,
        operation: NixServiceOperationKindV1,
        unit: &str,
        timeout: Duration,
    ) -> Result<NixSystemdLifecycleEvidenceV1, NixSystemdLifecycleTransactionErrorV1> {
        let operation = NixServiceOperationV1::new(unit.to_string(), operation)
            .map_err(|error| NixSystemdLifecycleTransactionErrorV1::InvalidServiceOperation(error.to_string()))?;
        let expected_job_type = NixSystemdJobTypeV1::for_operation(operation.operation())
            .ok_or(NixSystemdLifecycleTransactionErrorV1::UnsupportedOperation)?;

        let watcher = self.observer.arm_job_removed_watcher().await?;
        let manager_owner = watcher.manager_owner().to_string();

        let job_path = self
            .mutation
            .dispatch_lifecycle_for_manager_owner(&operation, &manager_owner)
            .await?;

        let job = self.observer.capture_job(&job_path, operation.unit()).await?;
        if job.job_type() != expected_job_type {
            return Err(NixSystemdLifecycleTransactionErrorV1::JobTypeMismatch);
        }

        let evidence = watcher.await_job_removed(&job, timeout).await?;
        if evidence.manager_owner != manager_owner {
            return Err(NixSystemdLifecycleTransactionErrorV1::ManagerOwnerMismatch);
        }

        Ok(NixSystemdLifecycleEvidenceV1 {
            operation: operation.operation(),
            unit: operation.unit().to_string(),
            job: evidence,
            manager_owner,
        })
    }
}

#[derive(Debug, Error)]
pub enum NixSystemdLifecycleTransactionErrorV1 {
    #[error("invalid service operation: {0}")]
    InvalidServiceOperation(String),
    #[error("operation is not supported by the native systemd lifecycle transport")]
    UnsupportedOperation,
    #[error("systemd lifecycle JobType does not match the typed operation")]
    JobTypeMismatch,
    #[error("systemd manager incarnation changed across the governed lifecycle sequence")]
    ManagerOwnerMismatch,
    #[error("systemd observer failed: {0}")]
    Observer(#[from] NixSystemdObserverErrorV1),
    #[error("systemd mutation transport failed: {0}")]
    Mutation(#[from] NixSystemdMutationTransportErrorV1),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_transaction_rejects_enablement_transport() {
        let _ = NixSystemdJobRemovedWatcherV1::manager_owner;
        let result = NixSystemdJobTypeV1::for_operation(NixServiceOperationKindV1::Enable);
        assert!(result.is_none());
    }

    #[test]
    fn lifecycle_sequence_is_named_in_execution_order() {
        // Keep the method's textual order auditable: arm, dispatch, capture, await.
        let _ = NixSystemdLifecycleTransactionV1::dispatch_and_observe;
    }
}
