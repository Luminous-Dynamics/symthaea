// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ethics gate for cell manipulation protocols.
//!
//! Enforces consent, IRB approval, and protocol-specific ethical requirements
//! before allowing any cell manipulation to proceed.

use serde::{Deserialize, Serialize};

/// Ethics gate that must be satisfied before cell manipulations can proceed.
///
/// All cell foundry operations that modify living cells should pass through
/// this gate. IVG protocols have additional requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicsGate {
    /// True when informed consent from the donor / participants has been documented.
    pub consent_obtained: bool,
    /// True when the protocol has received Institutional Review Board (IRB) approval.
    pub irb_approved: bool,
    /// Optional additional requirements for specific protocol types.
    pub additional_approvals: Vec<String>,
}

impl EthicsGate {
    /// Create a gate with no approvals (all checks will fail).
    pub fn new() -> Self {
        Self {
            consent_obtained: false,
            irb_approved: false,
            additional_approvals: Vec::new(),
        }
    }

    /// Create a fully approved gate.
    pub fn fully_approved() -> Self {
        Self {
            consent_obtained: true,
            irb_approved: true,
            additional_approvals: Vec::new(),
        }
    }

    /// Check whether a general cell manipulation is ethically cleared.
    pub fn check_manipulation(&self, action: &str) -> Result<(), String> {
        if !self.consent_obtained {
            return Err(format!(
                "Manipulation '{}' requires informed consent",
                action
            ));
        }
        if !self.irb_approved {
            return Err(format!("Manipulation '{}' requires IRB approval", action));
        }
        Ok(())
    }

    /// Check whether an IVG protocol is ethically cleared.
    ///
    /// IVG has additional ethical requirements beyond standard cell manipulation:
    /// gamete creation from iPSCs raises unique consent and regulatory issues.
    pub fn check_ivg_protocol(&self) -> Result<(), String> {
        self.check_manipulation("IVG protocol")?;

        // IVG requires explicit gamete-creation consent
        if !self
            .additional_approvals
            .iter()
            .any(|a| a == "gamete_creation")
        {
            return Err("IVG protocol requires explicit gamete-creation consent approval".into());
        }

        Ok(())
    }

    /// Check whether an SCNT protocol is ethically cleared.
    pub fn check_scnt_protocol(&self) -> Result<(), String> {
        self.check_manipulation("SCNT protocol")?;

        if !self
            .additional_approvals
            .iter()
            .any(|a| a == "nuclear_transfer")
        {
            return Err("SCNT protocol requires nuclear transfer approval".into());
        }

        Ok(())
    }

    /// Add an additional approval.
    pub fn add_approval(&mut self, approval: &str) {
        if !self.additional_approvals.contains(&approval.to_string()) {
            self.additional_approvals.push(approval.to_string());
        }
    }

    /// Grant informed consent.
    pub fn grant_consent(&mut self) {
        self.consent_obtained = true;
    }

    /// Grant IRB approval.
    pub fn grant_irb(&mut self) {
        self.irb_approved = true;
    }
}

impl Default for EthicsGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_consent_fails() {
        let gate = EthicsGate::new();
        assert!(gate.check_manipulation("test").is_err());
    }

    #[test]
    fn test_no_irb_fails() {
        let mut gate = EthicsGate::new();
        gate.consent_obtained = true;
        assert!(gate.check_manipulation("test").is_err());
    }

    #[test]
    fn test_fully_approved_passes() {
        let gate = EthicsGate::fully_approved();
        assert!(gate.check_manipulation("test").is_ok());
    }

    #[test]
    fn test_ivg_requires_gamete_approval() {
        let gate = EthicsGate::fully_approved();
        assert!(
            gate.check_ivg_protocol().is_err(),
            "IVG should require gamete_creation approval"
        );
    }

    #[test]
    fn test_ivg_passes_with_gamete_approval() {
        let mut gate = EthicsGate::fully_approved();
        gate.add_approval("gamete_creation");
        assert!(gate.check_ivg_protocol().is_ok());
    }

    #[test]
    fn test_scnt_requires_nuclear_transfer_approval() {
        let gate = EthicsGate::fully_approved();
        assert!(
            gate.check_scnt_protocol().is_err(),
            "SCNT should require nuclear_transfer approval"
        );
    }

    #[test]
    fn test_scnt_passes_with_approval() {
        let mut gate = EthicsGate::fully_approved();
        gate.add_approval("nuclear_transfer");
        assert!(gate.check_scnt_protocol().is_ok());
    }

    #[test]
    fn test_add_approval_idempotent() {
        let mut gate = EthicsGate::new();
        gate.add_approval("test");
        gate.add_approval("test");
        assert_eq!(gate.additional_approvals.len(), 1);
    }

    #[test]
    fn test_grant_consent() {
        let mut gate = EthicsGate::new();
        assert!(!gate.consent_obtained);
        gate.grant_consent();
        assert!(gate.consent_obtained);
    }

    #[test]
    fn test_grant_irb() {
        let mut gate = EthicsGate::new();
        assert!(!gate.irb_approved);
        gate.grant_irb();
        assert!(gate.irb_approved);
    }

    #[test]
    fn test_error_message_contains_action() {
        let gate = EthicsGate::new();
        let err = gate.check_manipulation("custom_action").unwrap_err();
        assert!(
            err.contains("custom_action"),
            "Error should contain the action name: {}",
            err
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut gate = EthicsGate::fully_approved();
        gate.add_approval("gamete_creation");
        let json = serde_json::to_string(&gate).unwrap();
        let deser: EthicsGate = serde_json::from_str(&json).unwrap();
        assert_eq!(gate.consent_obtained, deser.consent_obtained);
        assert_eq!(gate.irb_approved, deser.irb_approved);
        assert_eq!(gate.additional_approvals, deser.additional_approvals);
    }
}
