// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Thin runtime bridge from the zero-authority Component control host into the
//! runtime-neutral extension admission-policy interface.
//!
//! This crate deliberately owns no signer verification, trust policy, admission
//! policy, routing, or capability execution. It only converts a successful
//! `ControlPlaneHost` inspection into the `TechnicalInspection` evidence consumed
//! by the higher admission layer.

#![deny(unsafe_code)]

use symthaea_extension_admission::Sha256Digest;
use symthaea_extension_admission_policy::{TechnicalInspection, TechnicalInspector};
use symthaea_extension_host::{ControlHostError, ControlHostPolicy, ControlPlaneHost};

/// Adapter that delegates technical compatibility inspection to the hardened
/// zero-authority Component host.
#[derive(Debug, Clone, Copy)]
pub struct WasmControlTechnicalInspector {
    host: ControlPlaneHost,
}

impl WasmControlTechnicalInspector {
    pub const fn new(host: ControlPlaneHost) -> Self {
        Self { host }
    }

    pub fn with_policy(policy: ControlHostPolicy) -> Self {
        Self::new(ControlPlaneHost::new(policy))
    }

    pub const fn host(&self) -> &ControlPlaneHost {
        &self.host
    }
}

impl Default for WasmControlTechnicalInspector {
    fn default() -> Self {
        Self::new(ControlPlaneHost::default())
    }
}

impl TechnicalInspector for WasmControlTechnicalInspector {
    type Error = ControlHostError;

    fn inspect(
        &self,
        manifest_bytes: &[u8],
        payload_bytes: &[u8],
    ) -> Result<TechnicalInspection, Self::Error> {
        let inspection = self.host.inspect(manifest_bytes, payload_bytes)?;
        Ok(TechnicalInspection::new(
            inspection.manifest,
            Sha256Digest::new(inspection.manifest_sha256),
            Sha256Digest::new(inspection.component_sha256),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_rejection_propagates_before_admission() {
        let manifest = br#"{
            "id":"org.example.data",
            "name":"Example Data",
            "version":"1.0.0",
            "kind":"knowledge_pack",
            "runtime":"data_only",
            "provides":[{
                "id":"knowledge.example.read",
                "description":"Read example knowledge",
                "effect":"pure"
            }],
            "resources":{
                "memory_bytes":0,
                "fuel":0,
                "max_wall_time_ms":0,
                "max_output_bytes":0,
                "max_concurrency":0
            }
        }"#;

        let inspector = WasmControlTechnicalInspector::default();
        let error = TechnicalInspector::inspect(&inspector, manifest, b"payload").unwrap_err();
        assert!(matches!(error, ControlHostError::RuntimeNotWasm));
    }

    #[test]
    fn host_size_ceiling_is_preserved_by_adapter() {
        let mut policy = ControlHostPolicy::default();
        policy.max_manifest_bytes = 1;
        let inspector = WasmControlTechnicalInspector::with_policy(policy);

        let error = TechnicalInspector::inspect(&inspector, b"{}", b"").unwrap_err();
        assert!(matches!(error, ControlHostError::ManifestTooLarge));
    }
}
