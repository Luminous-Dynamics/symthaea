// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MFDI (Multi-Factor Delegated Identity) integration for CognitiveLoopService.
//!
//! This module provides identity verification, capability gating, and
//! cryptographic signing for cognitive loop outputs when the `identity`
//! feature is enabled.

#[cfg(feature = "identity")]
#[allow(unused_imports)]
use super::CognitiveLoopService;
#[cfg(feature = "identity")]
#[allow(unused_imports)]
use anyhow::Result;

#[cfg(feature = "identity")]
impl CognitiveLoopService {
    /// Get current agent ID (if identity is set)
    pub fn agent_id(&self) -> Option<&str> {
        self.mfdi_bridge.agent_id()
    }

    /// Get current assurance level
    pub fn assurance_level(&self) -> crate::identity::AssuranceLevel {
        self.mfdi_bridge.assurance_level()
    }

    /// Set identity from external verification
    pub fn set_identity(&mut self, identity: crate::identity::MfdiIdentity) {
        self.mfdi_bridge.set_identity(identity);
    }

    /// Check if a cognitive capability is allowed at current assurance level
    pub fn check_capability(&self, capability: crate::identity::CognitiveCapability) -> Result<()> {
        self.mfdi_bridge
            .check_capability(capability)
            .map_err(|e| anyhow::anyhow!("MFDI capability denied: {:?}", e))
    }

    /// Sign a cycle output
    pub fn sign_output(&mut self, output: &[f32]) -> Result<crate::identity::SignedOutput> {
        self.mfdi_bridge
            .sign_output(output)
            .map_err(|e| anyhow::anyhow!("MFDI signing failed: {:?}", e))
    }

    /// Verify a signed request
    pub fn verify_request(&mut self, request: &crate::identity::SignedRequest) -> Result<()> {
        self.mfdi_bridge
            .verify_request(request)
            .map_err(|e| anyhow::anyhow!("MFDI verification failed: {:?}", e))
    }

    /// Get mutable access to MFDI bridge for advanced operations
    pub fn mfdi_bridge_mut(&mut self) -> &mut crate::identity::MfdiBridge {
        &mut self.mfdi_bridge
    }
}
