// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::{PROMETHEUS_INTEGRATION_ID, PrometheusTextObserver};
use symthaea_integration_core::{
    IdentityProvider, IdentityRequest, IdentitySnapshot, IntegrationError, IntegrationFuture,
};

impl PrometheusTextObserver {
    pub fn identity_snapshot_sync(
        &self,
        request: IdentityRequest,
    ) -> Result<IdentitySnapshot, IntegrationError> {
        request.validate()?;
        let claims = self
            .identity_claims()
            .iter()
            .filter(|claim| {
                request.entities.is_empty() || request.entities.contains(&claim.subject)
            })
            .filter(|claim| {
                request.schemes.is_empty()
                    || request.schemes.contains(&claim.identifier.scheme)
            })
            .filter(|claim| {
                request
                    .at_unix_ms
                    .is_none_or(|at| claim.is_active_at(at))
            })
            .cloned()
            .collect();

        let snapshot = IdentitySnapshot {
            integration_id: PROMETHEUS_INTEGRATION_ID.into(),
            collected_at_unix_ms: self.batch().collected_at_unix_ms,
            claims,
            separation_claims: vec![],
        };
        snapshot
            .validate()
            .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
        Ok(snapshot)
    }
}

impl IdentityProvider for PrometheusTextObserver {
    fn identity_snapshot<'a>(
        &'a self,
        request: IdentityRequest,
    ) -> IntegrationFuture<'a, Result<IdentitySnapshot, IntegrationError>> {
        Box::pin(async move { self.identity_snapshot_sync(request) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PrometheusFixtureContext, PrometheusIdentityMapping, PrometheusTextObserver,
    };

    fn observer() -> PrometheusTextObserver {
        PrometheusTextObserver::from_text(
            PrometheusFixtureContext {
                identity_mapping: PrometheusIdentityMapping::OtelPrometheusCompatibility,
                ..PrometheusFixtureContext::default()
            },
            "# TYPE up gauge\nup{job=\"shop/api\",instance=\"api-17\"} 1\n",
            100,
        )
        .unwrap()
    }

    #[test]
    fn identity_provider_filters_scheme() {
        let snapshot = observer()
            .identity_snapshot_sync(IdentityRequest {
                schemes: vec!["otel.service.instance.triplet".into()],
                ..IdentityRequest::default()
            })
            .unwrap();
        assert_eq!(snapshot.claims.len(), 1);

        let empty = observer()
            .identity_snapshot_sync(IdentityRequest {
                schemes: vec!["host.id".into()],
                ..IdentityRequest::default()
            })
            .unwrap();
        assert!(empty.claims.is_empty());
    }
}
