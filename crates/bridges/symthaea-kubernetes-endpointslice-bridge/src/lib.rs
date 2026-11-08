// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only EndpointSlice topology augmentation for Kubernetes E1 replay.
//!
//! This crate is deliberately not a second Kubernetes discoverer. It augments
//! an already-built `KubernetesReplayDiscoverer` snapshot and retains the same
//! `kubernetes-object-replay` source identity.

#![forbid(unsafe_code)]

mod model;

pub use model::{
    ENDPOINTSLICE_API_VERSION, ENDPOINTSLICE_KIND, ENDPOINTSLICE_MANAGED_BY_LABEL,
    ENDPOINTSLICE_SERVICE_LABEL, augment_endpoint_slices,
};
