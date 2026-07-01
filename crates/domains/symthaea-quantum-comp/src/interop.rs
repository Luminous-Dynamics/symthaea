//! Integration-boundary declarations for future Symthaea and Mycelix connectors.
//!
//! Alpha.9 keeps external integration explicit. These structs document what an
//! adapter is allowed to claim before it touches real QPUs, Symthaea substrate
//! registries, or Mycelix source chains.

use crate::experiment::ClaimBoundary;

/// Declared external integration target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationTarget {
    /// Symthaea substrate registry or cognition runtime.
    Symthaea,
    /// Mycelix source-chain or governance receipt layer.
    Mycelix,
    /// External quantum circuit or backend tooling.
    ExternalQuantumBackend,
    /// Local-only lab notebook or CI environment.
    LocalLab,
}

/// Adapter authority boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterAuthority {
    /// Adapter may export text, reports, or circuit artifacts only.
    ExportOnly,
    /// Adapter may observe external outputs, but not certify them.
    ObserveOnly,
    /// Adapter may request signing or attestation from another system.
    AttestationRequest,
    /// Adapter may never be used in alpha without explicit downstream review.
    BlockedInAlpha,
}

/// External backend or integration declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrationDeclaration {
    /// Integration target.
    pub target: IntegrationTarget,
    /// Adapter authority.
    pub authority: AdapterAuthority,
    /// Claim boundary allowed for reports using this integration.
    pub maximum_claim_boundary: ClaimBoundary,
    /// Human-readable adapter label.
    pub adapter_label: String,
    /// Required caveats.
    pub caveats: Vec<String>,
}

impl IntegrationDeclaration {
    /// Local lab declaration for alpha examples.
    pub fn local_lab() -> Self {
        Self {
            target: IntegrationTarget::LocalLab,
            authority: AdapterAuthority::ExportOnly,
            maximum_claim_boundary: ClaimBoundary::LocalSimulation,
            adapter_label: "local-alpha-lab".to_string(),
            caveats: vec![
                "local reports are not externally attested".to_string(),
                "fingerprints are not cryptographic receipts".to_string(),
            ],
        }
    }

    /// Future Mycelix bridge declaration.
    pub fn mycelix_receipt_request() -> Self {
        Self {
            target: IntegrationTarget::Mycelix,
            authority: AdapterAuthority::AttestationRequest,
            maximum_claim_boundary: ClaimBoundary::LocalSimulation,
            adapter_label: "mycelix-receipt-request-v0".to_string(),
            caveats: vec![
                "alpha crate does not sign source-chain entries itself".to_string(),
                "a real Mycelix connector must attach agent identity and signature metadata"
                    .to_string(),
            ],
        }
    }

    /// Future external quantum backend observation declaration.
    pub fn external_backend_observation(adapter_label: impl Into<String>) -> Self {
        Self {
            target: IntegrationTarget::ExternalQuantumBackend,
            authority: AdapterAuthority::ObserveOnly,
            maximum_claim_boundary: ClaimBoundary::ExternalBackendObservation,
            adapter_label: adapter_label.into(),
            caveats: vec![
                "requires backend name, device/simulator metadata, transpiler/circuit version, and raw result attachment".to_string(),
                "external observation is still not evidence of quantum consciousness".to_string(),
            ],
        }
    }

    /// Renders a compact text summary.
    pub fn to_text(&self) -> String {
        format!(
            "target={:?} authority={:?} maximum_claim_boundary={:?} adapter={} caveats={}",
            self.target,
            self.authority,
            self.maximum_claim_boundary,
            self.adapter_label,
            self.caveats.join(" | "),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interop_declarations_keep_caveats() {
        let decl = IntegrationDeclaration::mycelix_receipt_request();
        assert_eq!(decl.target, IntegrationTarget::Mycelix);
        assert!(decl.to_text().contains("does not sign"));
    }
}
