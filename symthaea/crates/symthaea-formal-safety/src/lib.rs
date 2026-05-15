// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Formal safety case primitives for engineering workflows.
//!
//! This crate deliberately stores proof obligations and evidence references,
//! not proof assistant bindings. Lean, Coq, Isabelle, TLA+, SMT, and runtime
//! monitors can be adapter crates layered on top.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Type of evidence attached to a safety claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceKind {
    /// Static proof or theorem-prover output.
    FormalProof,
    /// Simulation result from an external solver.
    Simulation,
    /// Test, inspection, calibration, or commissioning result.
    Test,
    /// Field telemetry or digital twin observation.
    Telemetry,
    /// Engineering standard, code, or regulatory citation.
    Standard,
}

/// Domain templates for common engineering safety cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyCaseTemplate {
    /// Civil/structural concept or asset.
    CivilStructure,
    /// Mechanism, machine, or thermal/mechanical assembly.
    MechanicalSystem,
    /// Robot or embodied autonomous system.
    Robotics,
    /// Electrical, electronics, circuit, or power system.
    ElectricalSystem,
    /// Aerospace system with coupled aero/thermal/structural/control risk.
    AerospaceSystem,
    /// Chemical or process system.
    ChemicalProcess,
    /// Nuclear or radiation-adjacent system.
    NuclearSystem,
    /// Materials, degradation, or manufacturability case.
    Materials,
    /// Environmental/sustainability case.
    EnvironmentalSystem,
    /// Cross-domain system-of-systems case.
    SystemOfSystems,
}

/// Status of a proof obligation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObligationStatus {
    /// Identified but not yet attempted.
    Open,
    /// Work is in progress.
    InProgress,
    /// Satisfied by attached evidence.
    Discharged,
    /// Evidence failed or contradicted the claim.
    Failed,
    /// Needs human engineering review.
    ReviewRequired,
}

/// A verifiable safety obligation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofObligation {
    /// Stable obligation id.
    pub id: Uuid,
    /// Safety claim in controlled natural language.
    pub claim: String,
    /// Method expected to discharge the claim.
    pub expected_evidence: EvidenceKind,
    /// Current status.
    pub status: ObligationStatus,
    /// Evidence references, such as file paths, solver run ids, or proof names.
    pub evidence_refs: Vec<String>,
}

impl ProofObligation {
    /// Create an open proof obligation.
    pub fn new(claim: impl Into<String>, expected_evidence: EvidenceKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            claim: claim.into(),
            expected_evidence,
            status: ObligationStatus::Open,
            evidence_refs: Vec::new(),
        }
    }

    /// Attach evidence and discharge the obligation.
    pub fn discharge(mut self, evidence_ref: impl Into<String>) -> Self {
        self.evidence_refs.push(evidence_ref.into());
        self.status = ObligationStatus::Discharged;
        self
    }
}

/// Minimal safety case container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyCase {
    /// Stable safety case id.
    pub id: Uuid,
    /// System or concept being justified.
    pub subject: String,
    /// Proof obligations associated with the subject.
    pub obligations: Vec<ProofObligation>,
}

impl SafetyCase {
    /// Construct an empty safety case.
    pub fn new(subject: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            subject: subject.into(),
            obligations: Vec::new(),
        }
    }

    /// Construct a safety case from a domain template.
    pub fn from_template(subject: impl Into<String>, template: SafetyCaseTemplate) -> Self {
        let mut safety_case = Self::new(subject);
        for (claim, evidence) in template_obligations(template) {
            safety_case.add_obligation(ProofObligation::new(claim, evidence));
        }
        safety_case
    }

    /// Add an obligation to the case.
    pub fn add_obligation(&mut self, obligation: ProofObligation) {
        self.obligations.push(obligation);
    }

    /// Returns true when every obligation has been discharged.
    pub fn is_discharged(&self) -> bool {
        !self.obligations.is_empty()
            && self
                .obligations
                .iter()
                .all(|obligation| obligation.status == ObligationStatus::Discharged)
    }
}

fn template_obligations(template: SafetyCaseTemplate) -> Vec<(&'static str, EvidenceKind)> {
    match template {
        SafetyCaseTemplate::CivilStructure => vec![
            (
                "load cases and combinations are identified for the intended structure",
                EvidenceKind::Standard,
            ),
            (
                "service and ultimate stresses remain below allowable limits",
                EvidenceKind::Simulation,
            ),
            (
                "deflection, vibration, and stability limits are explicitly checked",
                EvidenceKind::Simulation,
            ),
            (
                "inspection, maintenance, and degradation assumptions are documented",
                EvidenceKind::Telemetry,
            ),
        ],
        SafetyCaseTemplate::MechanicalSystem => vec![
            (
                "kinematic limits and collision envelopes are identified",
                EvidenceKind::Simulation,
            ),
            (
                "thermal, fatigue, and wear assumptions are documented",
                EvidenceKind::Test,
            ),
            (
                "failure modes and safe-state behavior are identified",
                EvidenceKind::Standard,
            ),
        ],
        SafetyCaseTemplate::Robotics => vec![
            (
                "workspace, force, velocity, and human-proximity limits are enforced",
                EvidenceKind::FormalProof,
            ),
            (
                "sim-to-real assumptions and domain randomization coverage are documented",
                EvidenceKind::Simulation,
            ),
            (
                "emergency stop and safe fallback behavior are tested",
                EvidenceKind::Test,
            ),
        ],
        SafetyCaseTemplate::ElectricalSystem => vec![
            (
                "voltage, current, thermal, and isolation limits are checked",
                EvidenceKind::Simulation,
            ),
            (
                "fault, transient, and protection behavior is evaluated",
                EvidenceKind::Simulation,
            ),
            (
                "applicable electrical codes and standards are recorded",
                EvidenceKind::Standard,
            ),
        ],
        SafetyCaseTemplate::AerospaceSystem => vec![
            (
                "aero-thermal-structural-control coupling assumptions are documented",
                EvidenceKind::Simulation,
            ),
            (
                "safe-mode and abort/fallback behavior are verified",
                EvidenceKind::FormalProof,
            ),
            (
                "loads, margins, and operational envelopes are reviewed",
                EvidenceKind::Standard,
            ),
        ],
        SafetyCaseTemplate::ChemicalProcess => vec![
            (
                "reaction, pressure, temperature, and inventory hazards are identified",
                EvidenceKind::Standard,
            ),
            (
                "process upset and relief scenarios are simulated or tested",
                EvidenceKind::Simulation,
            ),
            (
                "containment, shutdown, and monitoring obligations are defined",
                EvidenceKind::Test,
            ),
        ],
        SafetyCaseTemplate::NuclearSystem => vec![
            (
                "radiological source term and exposure boundaries are documented",
                EvidenceKind::Standard,
            ),
            (
                "criticality, shielding, decay heat, or safeguards assumptions are checked",
                EvidenceKind::Simulation,
            ),
            (
                "human oversight, containment, and incident response obligations are explicit",
                EvidenceKind::FormalProof,
            ),
        ],
        SafetyCaseTemplate::Materials => vec![
            (
                "material properties, provenance, and uncertainty bounds are documented",
                EvidenceKind::Test,
            ),
            (
                "aging, fatigue, corrosion, or degradation models are checked",
                EvidenceKind::Simulation,
            ),
            (
                "manufacturability and inspection assumptions are recorded",
                EvidenceKind::Standard,
            ),
        ],
        SafetyCaseTemplate::EnvironmentalSystem => vec![
            (
                "environmental boundary conditions and affected stakeholders are identified",
                EvidenceKind::Standard,
            ),
            (
                "carbon, water, habitat, or pollution impacts are quantified",
                EvidenceKind::Simulation,
            ),
            (
                "monitoring and intervention thresholds are defined",
                EvidenceKind::Telemetry,
            ),
        ],
        SafetyCaseTemplate::SystemOfSystems => vec![
            (
                "interfaces, authorities, and cross-domain assumptions are documented",
                EvidenceKind::Standard,
            ),
            (
                "hazard propagation across subsystems is analyzed",
                EvidenceKind::Simulation,
            ),
            (
                "deployment is blocked until all critical obligations are discharged",
                EvidenceKind::FormalProof,
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safety_case_requires_all_obligations_discharged() {
        let mut safety_case = SafetyCase::new("concept beam");
        safety_case.add_obligation(ProofObligation::new(
            "stress remains below allowable under service load",
            EvidenceKind::Simulation,
        ));
        assert!(!safety_case.is_discharged());

        safety_case.obligations[0] = safety_case.obligations[0].clone().discharge("fea-run-42");
        assert!(safety_case.is_discharged());
    }

    #[test]
    fn template_creates_domain_obligations() {
        let safety_case =
            SafetyCase::from_template("pedestrian bridge", SafetyCaseTemplate::CivilStructure);
        assert!(safety_case.obligations.len() >= 3);
        assert!(!safety_case.is_discharged());
    }
}
