// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::{OTLP_METRICS_INTEGRATION_ID, OtlpMetricsObserver};
use symthaea_integration_core::{
    IdentityProvider, IdentityRequest, IdentitySnapshot, IntegrationError, IntegrationFuture,
};

impl OtlpMetricsObserver {
    pub fn identity_snapshot_sync(
        &self,
        request: IdentityRequest,
    ) -> Result<IdentitySnapshot, IntegrationError> {
        request.validate()?;
        let translated = self.translate()?;
        let claims = translated
            .identity_claims
            .into_iter()
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
            .collect();

        let snapshot = IdentitySnapshot {
            integration_id: OTLP_METRICS_INTEGRATION_ID.into(),
            collected_at_unix_ms: self.ingested_at_unix_ms,
            claims,
            separation_claims: vec![],
        };
        snapshot
            .validate()
            .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
        Ok(snapshot)
    }
}

impl IdentityProvider for OtlpMetricsObserver {
    fn identity_snapshot<'a>(
        &'a self,
        request: IdentityRequest,
    ) -> IntegrationFuture<'a, Result<IdentitySnapshot, IntegrationError>> {
        Box::pin(async move { self.identity_snapshot_sync(request) })
    }
}
