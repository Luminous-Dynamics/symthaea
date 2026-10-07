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
