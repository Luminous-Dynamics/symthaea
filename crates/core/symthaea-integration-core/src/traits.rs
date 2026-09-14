// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only integration behavior contracts.
//!
//! The boxed-future shape is intentionally object-safe without adding an
//! `async-trait` dependency. v0.1 exposes observation and discovery only.

use crate::manifest::IntegrationManifest;
use crate::observation::{EntityRef, ObservationBatch};
use crate::topology::DiscoverySnapshot;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;

pub type IntegrationFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Canonical capability an adapter must explicitly declare before the registry
/// will honor `DiscoveryRequest::require_complete`.
///
/// Completeness means the adapter is qualified to treat absence inside the
/// requested/configured scope as meaningful evidence, not merely that it
/// returned every object present in one fixture, page, watch cache, or replay
/// corpus.
pub const COMPLETE_DISCOVERY_CAPABILITY: &str = "discover.snapshot.complete";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ObservationRequest {
    /// Empty means all entities visible within the integration's configured scope.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<EntityRef>,
    /// Empty means all declared observation signals.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub signals: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub since_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub until_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub filters: BTreeMap<String, String>,
}

impl ObservationRequest {
    pub fn validate(&self) -> Result<(), IntegrationError> {
        if let (Some(since), Some(until)) = (self.since_unix_ms, self.until_unix_ms) {
            if since > until {
                return Err(IntegrationError::InvalidRequest(format!(
                    "since_unix_ms ({since}) is after until_unix_ms ({until})"
                )));
            }
        }
        if self.signals.iter().any(|signal| signal.trim().is_empty()) {
            return Err(IntegrationError::InvalidRequest(
                "signal selectors may not contain empty strings".into(),
            ));
        }
        if self.filters.keys().any(|key| key.trim().is_empty()) {
            return Err(IntegrationError::InvalidRequest(
                "observation filter keys may not be empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DiscoveryRequest {
    /// Optional discovery root/scope. None means the adapter's configured scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<EntityRef>,
    /// Empty means all declared entity kinds.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entity_kinds: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub filters: BTreeMap<String, String>,
    /// Require a qualified exhaustive snapshot for the requested/configured
    /// scope. When false (the default), absence of an entity/relation must not
    /// be interpreted as evidence that it does not exist.
    #[serde(default)]
    pub require_complete: bool,
}

impl DiscoveryRequest {
    pub fn validate(&self) -> Result<(), IntegrationError> {
        if let Some(root) = &self.root {
            if root.namespace.trim().is_empty()
                || root.kind.trim().is_empty()
                || root.id.trim().is_empty()
            {
                return Err(IntegrationError::InvalidRequest(
                    "discovery root must contain non-empty namespace, kind, and id".into(),
                ));
            }
        }

        let mut kinds = BTreeSet::new();
        for kind in &self.entity_kinds {
            if kind.trim().is_empty() {
                return Err(IntegrationError::InvalidRequest(
                    "discovery entity-kind selectors may not contain empty strings".into(),
                ));
            }
            if !kinds.insert(kind) {
                return Err(IntegrationError::InvalidRequest(format!(
                    "duplicate discovery entity-kind selector `{kind}`"
                )));
            }
        }

        if self.filters.keys().any(|key| key.trim().is_empty()) {
            return Err(IntegrationError::InvalidRequest(
                "discovery filter keys may not be empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IntegrationError {
    #[error("invalid integration request: {0}")]
    InvalidRequest(String),
    #[error("integration authentication failed: {0}")]
    Authentication(String),
    #[error("integration permission denied: {0}")]
    PermissionDenied(String),
    #[error("integration transport failure: {0}")]
    Transport(String),
    #[error("integration protocol failure: {0}")]
    Protocol(String),
    #[error("integration produced invalid output: {0}")]
    InvalidOutput(String),
    #[error("integration operation is unsupported: {0}")]
    Unsupported(String),
}

/// Shared identity contract for every integration implementation.
pub trait IntegrationIdentity: Send + Sync {
    fn manifest(&self) -> &IntegrationManifest;
}

/// Read-only sensor contract.
pub trait Observer: IntegrationIdentity {
    fn observe<'a>(
        &'a self,
        request: ObservationRequest,
    ) -> IntegrationFuture<'a, Result<ObservationBatch, IntegrationError>>;
}

/// Read-only topology/inventory discovery contract.
pub trait Discoverer: IntegrationIdentity {
    fn discover<'a>(
        &'a self,
        request: DiscoveryRequest,
    ) -> IntegrationFuture<'a, Result<DiscoverySnapshot, IntegrationError>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverted_time_window_is_rejected() {
        let request = ObservationRequest {
            since_unix_ms: Some(10),
            until_unix_ms: Some(9),
            ..Default::default()
        };
        assert!(matches!(
            request.validate(),
            Err(IntegrationError::InvalidRequest(_))
        ));
    }

    #[test]
    fn empty_request_means_unrestricted_but_not_complete() {
        assert!(ObservationRequest::default().validate().is_ok());
        let discovery = DiscoveryRequest::default();
        assert!(discovery.validate().is_ok());
        assert!(!discovery.require_complete);
    }

    #[test]
    fn duplicate_discovery_kind_is_rejected() {
        let request = DiscoveryRequest {
            entity_kinds: vec!["pod".into(), "pod".into()],
            ..Default::default()
        };
        assert!(matches!(
            request.validate(),
            Err(IntegrationError::InvalidRequest(_))
        ));
    }

    #[test]
    fn missing_complete_flag_deserializes_conservatively() {
        let request: DiscoveryRequest = serde_json::from_str("{}").unwrap();
        assert!(!request.require_complete);
    }
}
