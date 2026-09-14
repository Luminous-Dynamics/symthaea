// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical semantic identifier vocabulary shared by integration adapters.
//!
//! Identifier scheme strings are part of the evidence contract: two adapters
//! can only resolve the same external entity when they mean the same thing by a
//! scheme and scope. Keep standards-backed names here rather than duplicating
//! string literals in each bridge.

use crate::{
    ExternalIdentifier, IdentifierStability, IdentifierUniqueness, IdentityValidationError,
};

pub const K8S_CLUSTER_UID_SCHEME: &str = "k8s.cluster.uid";
pub const K8S_NODE_UID_SCHEME: &str = "k8s.node.uid";
pub const K8S_POD_UID_SCHEME: &str = "k8s.pod.uid";
pub const K8S_SERVICE_UID_SCHEME: &str = "k8s.service.uid";
pub const K8S_DEPLOYMENT_UID_SCHEME: &str = "k8s.deployment.uid";
pub const K8S_REPLICASET_UID_SCHEME: &str = "k8s.replicaset.uid";
pub const K8S_STATEFULSET_UID_SCHEME: &str = "k8s.statefulset.uid";
pub const K8S_DAEMONSET_UID_SCHEME: &str = "k8s.daemonset.uid";
pub const K8S_JOB_UID_SCHEME: &str = "k8s.job.uid";
pub const K8S_CRONJOB_UID_SCHEME: &str = "k8s.cronjob.uid";

/// Symthaea fallback for Kubernetes object kinds without a standards-defined
/// kind-specific UID scheme in the semantic-convention surface used by v0.1.
/// It intentionally cannot collide with a standards-backed scheme.
pub const K8S_OBJECT_UID_SCHEME: &str = "symthaea.k8s.object.uid.v1";

/// Return the standards-backed UID scheme for an integration-core Kubernetes
/// entity kind when one is defined by the v0.1 semantic vocabulary.
pub fn kubernetes_uid_scheme(entity_kind: &str) -> Option<&'static str> {
    match entity_kind {
        "k8s_node" => Some(K8S_NODE_UID_SCHEME),
        "k8s_pod" => Some(K8S_POD_UID_SCHEME),
        "k8s_service" => Some(K8S_SERVICE_UID_SCHEME),
        "k8s_deployment" => Some(K8S_DEPLOYMENT_UID_SCHEME),
        "k8s_replicaset" => Some(K8S_REPLICASET_UID_SCHEME),
        "k8s_statefulset" => Some(K8S_STATEFULSET_UID_SCHEME),
        "k8s_daemonset" => Some(K8S_DAEMONSET_UID_SCHEME),
        "k8s_job" => Some(K8S_JOB_UID_SCHEME),
        "k8s_cronjob" => Some(K8S_CRONJOB_UID_SCHEME),
        _ => None,
    }
}

/// Canonical collision-safe scope for object UIDs when a real Kubernetes
/// cluster UID is known. The caller-supplied display/name identity of a cluster
/// must never be substituted for this value.
pub fn kubernetes_cluster_uid_scope(cluster_uid: &str) -> Result<String, IdentityValidationError> {
    require_non_blank("k8s.cluster.uid", cluster_uid)?;
    Ok(format!(
        "semantic-scope-v1|{}:{}|{}:{}",
        K8S_CLUSTER_UID_SCHEME.len(),
        K8S_CLUSTER_UID_SCHEME,
        cluster_uid.len(),
        cluster_uid
    ))
}

/// Canonical Kubernetes cluster identity. OpenTelemetry currently defines
/// `k8s.cluster.uid` as a pseudo-ID based on the kube-system Namespace UID.
pub fn kubernetes_cluster_uid_identifier(
    cluster_uid: &str,
) -> Result<ExternalIdentifier, IdentityValidationError> {
    require_non_blank("k8s.cluster.uid", cluster_uid)?;
    let identifier = ExternalIdentifier {
        scheme: K8S_CLUSTER_UID_SCHEME.into(),
        value: cluster_uid.into(),
        scope: None,
        uniqueness: IdentifierUniqueness::Global,
        stability: IdentifierStability::Persistent,
        case_sensitive: true,
    };
    identifier.validate()?;
    Ok(identifier)
}

/// Build an object UID identifier shared by Kubernetes-native and OTLP
/// adapters. A UID without a real cluster UID remains intentionally ambiguous:
/// matching it may produce a candidate, but must not become a strong scoped
/// equivalence merely because two sources copied the same string.
pub fn kubernetes_object_uid_identifier(
    entity_kind: &str,
    uid: &str,
    cluster_uid: Option<&str>,
) -> Result<ExternalIdentifier, IdentityValidationError> {
    require_non_blank("kubernetes object uid", uid)?;
    let scheme = kubernetes_uid_scheme(entity_kind).unwrap_or(K8S_OBJECT_UID_SCHEME);
    let (scope, uniqueness) = match cluster_uid.map(str::trim).filter(|value| !value.is_empty()) {
        Some(cluster_uid) => (
            Some(kubernetes_cluster_uid_scope(cluster_uid)?),
            IdentifierUniqueness::Scoped,
        ),
        None => (None, IdentifierUniqueness::Ambiguous),
    };
    let identifier = ExternalIdentifier {
        scheme: scheme.into(),
        value: uid.into(),
        scope,
        uniqueness,
        stability: IdentifierStability::Persistent,
        case_sensitive: true,
    };
    identifier.validate()?;
    Ok(identifier)
}

fn require_non_blank(field: &'static str, value: &str) -> Result<(), IdentityValidationError> {
    if value.trim().is_empty() {
        Err(IdentityValidationError::EmptyField(field))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pod_uid_uses_standard_scheme_and_real_cluster_scope() {
        let identifier = kubernetes_object_uid_identifier(
            "k8s_pod",
            "pod-uid",
            Some("cluster-uid"),
        )
        .unwrap();
        assert_eq!(identifier.scheme, K8S_POD_UID_SCHEME);
        assert_eq!(identifier.uniqueness, IdentifierUniqueness::Scoped);
        assert!(identifier.scope.unwrap().contains(K8S_CLUSTER_UID_SCHEME));
    }

    #[test]
    fn missing_cluster_uid_never_becomes_strongly_scoped_by_name() {
        let identifier =
            kubernetes_object_uid_identifier("k8s_pod", "pod-uid", None).unwrap();
        assert_eq!(identifier.scheme, K8S_POD_UID_SCHEME);
        assert_eq!(identifier.uniqueness, IdentifierUniqueness::Ambiguous);
        assert!(identifier.scope.is_none());
    }

    #[test]
    fn unsupported_kind_uses_nonstandard_fallback_scheme() {
        let identifier = kubernetes_object_uid_identifier(
            "k8s_object",
            "widget-uid",
            Some("cluster-uid"),
        )
        .unwrap();
        assert_eq!(identifier.scheme, K8S_OBJECT_UID_SCHEME);
    }

    #[test]
    fn cluster_scope_is_separator_collision_safe() {
        let left = kubernetes_cluster_uid_scope("a|b").unwrap();
        let right = kubernetes_cluster_uid_scope("a").unwrap();
        assert_ne!(left, right);
    }
}
