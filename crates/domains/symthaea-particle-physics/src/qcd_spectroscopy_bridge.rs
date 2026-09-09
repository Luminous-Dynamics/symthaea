// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! QCD ↔ spectroscopy semantic bridge.
//!
//! This module connects non-Abelian gluon self-interaction concepts to
//! spectroscopy hypotheses without claiming that Symthaea's HDC encodings are
//! a lattice-QCD solver. The bridge expresses *why* gluonic bound states are
//! admissible in QCD and which spectroscopy channels/hypotheses they inform.

use crate::exotic_hadrons::{CompositionModel, HadronBasisState};
use crate::hadron_spectroscopy::Jpc;
use serde::{Deserialize, Serialize};

/// QCD mechanism relevant to an emergent hadronic-state hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QcdMechanism {
    /// Three-gluon non-Abelian self interaction (ggg vertex).
    TripleGluonSelfInteraction,
    /// Four-gluon non-Abelian self interaction (gggg vertex).
    QuarticGluonSelfInteraction,
    /// Color confinement / color-singlet physical-state requirement.
    ColorConfinement,
    /// Gluonic excitation accompanying valence quarks.
    ExcitedGluonicField,
    /// Mixing between states with identical conserved quantum numbers.
    StateMixing,
}

/// What level of assertion a bridge statement is making.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BridgeAuthority {
    /// Structural fact already represented by the QCD model.
    QcdStructural,
    /// Qualitative physical implication of the structural model.
    QualitativeImplication,
    /// Requires an external quantitative solver/calculation for validation.
    RequiresExternalCalculation,
}

/// Semantic link from QCD structure to a spectroscopy composition hypothesis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QcdSpectroscopyLink {
    pub mechanism: QcdMechanism,
    pub target: CompositionModel,
    pub jpc: Option<Jpc>,
    pub authority: BridgeAuthority,
    pub rationale: String,
}

/// Canonical qualitative links for a glueball hypothesis.
///
/// These links do not predict masses, widths, matrix elements, or branching
/// fractions. Those require lattice QCD, phenomenology, or experiment.
pub fn glueball_links(jpc: Jpc) -> Vec<QcdSpectroscopyLink> {
    let target = CompositionModel::Pure(HadronBasisState::Glueball);
    vec![
        QcdSpectroscopyLink {
            mechanism: QcdMechanism::TripleGluonSelfInteraction,
            target: target.clone(),
            jpc: Some(jpc),
            authority: BridgeAuthority::QcdStructural,
            rationale: "QCD contains a non-Abelian three-gluon self-interaction, so purely gluonic interacting configurations are structurally allowed".into(),
        },
        QcdSpectroscopyLink {
            mechanism: QcdMechanism::QuarticGluonSelfInteraction,
            target: target.clone(),
            jpc: Some(jpc),
            authority: BridgeAuthority::QcdStructural,
            rationale: "QCD also contains a four-gluon self-interaction; gluons are not passive force carriers".into(),
        },
        QcdSpectroscopyLink {
            mechanism: QcdMechanism::ColorConfinement,
            target,
            jpc: Some(jpc),
            authority: BridgeAuthority::QualitativeImplication,
            rationale: "physical asymptotic states must be color singlets, motivating confined gluonic bound-state channels".into(),
        },
    ]
}

/// Canonical qualitative links for a hybrid-hadron hypothesis.
pub fn hybrid_links(jpc: Option<Jpc>) -> Vec<QcdSpectroscopyLink> {
    let target = CompositionModel::Pure(HadronBasisState::Hybrid { valence: None });
    vec![QcdSpectroscopyLink {
        mechanism: QcdMechanism::ExcitedGluonicField,
        target,
        jpc,
        authority: BridgeAuthority::QualitativeImplication,
        rationale: "hybrid spectroscopy treats gluonic excitation as an explicit degree of freedom in addition to valence quarks".into(),
    }]
}

/// Mark a proposed mass/width/branching-ratio calculation as outside the HDC
/// bridge's authority.
pub fn requires_quantitative_qcd(observable: &str) -> bool {
    matches!(
        observable,
        "mass" | "width" | "branching_fraction" | "matrix_element" | "form_factor" | "finite_volume_spectrum"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glueball_bridge_exposes_non_abelian_mechanisms() {
        let links = glueball_links(Jpc::PSEUDOSCALAR);
        assert!(links.iter().any(|l| l.mechanism == QcdMechanism::TripleGluonSelfInteraction));
        assert!(links.iter().any(|l| l.mechanism == QcdMechanism::QuarticGluonSelfInteraction));
        assert!(links.iter().any(|l| l.mechanism == QcdMechanism::ColorConfinement));
    }

    #[test]
    fn bridge_refuses_to_be_a_fake_lattice_solver() {
        assert!(requires_quantitative_qcd("mass"));
        assert!(requires_quantitative_qcd("branching_fraction"));
        assert!(!requires_quantitative_qcd("qualitative_composition"));
    }
}
