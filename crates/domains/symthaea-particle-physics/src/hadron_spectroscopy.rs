// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-aware hadron spectroscopy.
//!
//! This module separates measured resonance observables from theoretical
//! interpretations of internal composition. That distinction is essential for
//! frontier QCD states such as glueball candidates, hybrids, multiquarks, and
//! hadronic molecules.
//!
//! The core rule is deliberately strict:
//! **observation is data; composition is interpretation.**

use serde::{Deserialize, Serialize};

/// Spin/parity/charge-conjugation quantum numbers J^PC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Jpc {
    pub spin_j: u8,
    pub parity: i8,
    pub charge_conjugation: i8,
}

impl Jpc {
    pub const fn new(spin_j: u8, parity: i8, charge_conjugation: i8) -> Self {
        Self {
            spin_j,
            parity,
            charge_conjugation,
        }
    }

    pub const SCALAR: Self = Self::new(0, 1, 1); // 0++
    pub const TENSOR: Self = Self::new(2, 1, 1); // 2++
    pub const PSEUDOSCALAR: Self = Self::new(0, -1, 1); // 0-+
}

/// Value with asymmetric uncertainty.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub value: f64,
    pub err_minus: f64,
    pub err_plus: f64,
}

impl Measurement {
    pub const fn symmetric(value: f64, err: f64) -> Self {
        Self {
            value,
            err_minus: err,
            err_plus: err,
        }
    }

    pub const fn asymmetric(value: f64, err_minus: f64, err_plus: f64) -> Self {
        Self {
            value,
            err_minus,
            err_plus,
        }
    }
}

/// Broad epistemic status of a resonance observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservationStatus {
    Predicted,
    Candidate,
    Observed,
    Established,
}

/// Strength/status of a theoretical interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InterpretationStatus {
    Proposed,
    Supported,
    StronglySupported,
    Consensus,
    Contested,
}

/// QCD treatment used for a theoretical prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QcdApproximation {
    PureYangMills,
    QuenchedQcd,
    DynamicalQcd { flavors: u8 },
    Phenomenological,
}

/// Coarse-grained internal-composition hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HadronicComposition {
    QuarkAntiquark,
    ThreeQuark,
    Multiquark,
    Glueball,
    Hybrid,
    HadronicMolecule,
    Mixed,
    Unknown,
}

/// Source class for a piece of spectroscopy evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceKind {
    ExperimentalObservation,
    ExperimentalUpperLimit,
    LatticePrediction,
    PhenomenologicalModel,
}

/// How a record bears on an interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceEffect {
    Supports,
    Constrains,
    Challenges,
    Context,
}

/// Minimal immutable provenance record for a spectroscopy claim.
///
/// Static strings keep built-in fixtures const-friendly. A future ingestion
/// layer can own arbitrary external strings and normalize them into stable keys.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceRecord {
    pub key: &'static str,
    pub kind: EvidenceKind,
    pub effect: EvidenceEffect,
    pub source: &'static str,
    pub persistent_id: &'static str,
    pub year: u16,
}

/// A directly observed resonance, independent of its internal interpretation.
///
/// `Serialize` is intentional here; these built-in fixtures use static labels.
/// External ingestion can map owned source records into this canonical form.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ObservedResonance {
    pub name: &'static str,
    pub jpc: Option<Jpc>,
    /// Mass in MeV/c^2.
    pub mass_mev: Measurement,
    /// Width in MeV.
    pub width_mev: Measurement,
    pub status: ObservationStatus,
}

/// A hypothesis linking an observed resonance to an internal composition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompositionInterpretation {
    pub resonance: &'static str,
    pub composition: HadronicComposition,
    pub status: InterpretationStatus,
    /// Provenance key intended to resolve into the evidence graph rather than
    /// serve as a free-floating confidence scalar.
    pub evidence_key: &'static str,
}

/// A theoretical glueball level from a specified QCD calculation family.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GlueballLevel {
    pub jpc: Jpc,
    /// Predicted mass in GeV/c^2.
    pub mass_gev: Measurement,
    pub approximation: QcdApproximation,
}

/// Baseline pure-gauge glueball spectrum.
///
/// These approximate values are represented as theory predictions, not as
/// discovered particles. Physical resonances can mix with quarkonia once
/// dynamical quarks are included; callers must not equate spectrum proximity
/// with particle identity.
pub const BASELINE_GLUEBALL_LEVELS: [GlueballLevel; 3] = [
    GlueballLevel {
        jpc: Jpc::SCALAR,
        mass_gev: Measurement::symmetric(1.70, 0.10),
        approximation: QcdApproximation::PureYangMills,
    },
    GlueballLevel {
        jpc: Jpc::TENSOR,
        mass_gev: Measurement::symmetric(2.39, 0.12),
        approximation: QcdApproximation::PureYangMills,
    },
    GlueballLevel {
        jpc: Jpc::PSEUDOSCALAR,
        mass_gev: Measurement::symmetric(2.56, 0.15),
        approximation: QcdApproximation::PureYangMills,
    },
];

/// 2026 Yang-Mills lattice prediction for the scalar-glueball mass radius.
///
/// Source: Abbott et al., Phys. Rev. Lett. 136, 041901 (2026),
/// DOI 10.1103/67xg-qxhz. This is a theory prediction at a single lattice
/// spacing, not an experimental radius measurement.
pub const SCALAR_GLUEBALL_MASS_RADIUS_FM: Measurement = Measurement::symmetric(0.263, 0.031);

/// BESIII-era X(2370) resonance fixture.
///
/// The resonance and J^PC assignment are kept separate from its proposed
/// glueball-dominant composition. This prevents a future reinterpretation from
/// mutating the underlying experimental observation.
///
/// Combined 2026 BESIII values: m = 2359 +13/-14 MeV/c^2,
/// Gamma = 170 +44/-29 MeV.
pub const X2370: ObservedResonance = ObservedResonance {
    name: "X(2370)",
    jpc: Some(Jpc::PSEUDOSCALAR),
    mass_mev: Measurement::asymmetric(2359.0, 14.0, 13.0),
    width_mev: Measurement::asymmetric(170.0, 29.0, 44.0),
    status: ObservationStatus::Established,
};

/// Evidence records supporting the canonical X(2370) observation and the
/// glueball-dominant interpretation. The records do not themselves compute a
/// confidence score; that belongs to the epistemic layer.
pub const X2370_EVIDENCE: [EvidenceRecord; 4] = [
    EvidenceRecord {
        key: "besiii::x2370::jpc_0minusplus",
        kind: EvidenceKind::ExperimentalObservation,
        effect: EvidenceEffect::Supports,
        source: "BESIII Collaboration",
        persistent_id: "arXiv:2312.05324",
        year: 2024,
    },
    EvidenceRecord {
        key: "besiii::x2370::mass_width_decay_modes",
        kind: EvidenceKind::ExperimentalObservation,
        effect: EvidenceEffect::Supports,
        source: "BESIII Collaboration",
        persistent_id: "arXiv:2605.26495",
        year: 2026,
    },
    EvidenceRecord {
        key: "besiii::x2370::kstar_k_suppression",
        kind: EvidenceKind::ExperimentalUpperLimit,
        effect: EvidenceEffect::Supports,
        source: "BESIII Collaboration",
        persistent_id: "arXiv:2607.20366",
        year: 2026,
    },
    EvidenceRecord {
        key: "lattice::scalar_glueball::mass_radius",
        kind: EvidenceKind::LatticePrediction,
        effect: EvidenceEffect::Context,
        source: "Abbott et al.",
        persistent_id: "doi:10.1103/67xg-qxhz",
        year: 2026,
    },
];

/// BESIII's 2026 glueball-dominant interpretation represented conservatively.
pub const X2370_GLUEBALL_INTERPRETATION: CompositionInterpretation = CompositionInterpretation {
    resonance: "X(2370)",
    composition: HadronicComposition::Glueball,
    status: InterpretationStatus::StronglySupported,
    evidence_key: "besiii::x2370::pseudoscalar_glueball_dominant",
};

/// A retained molecular-family alternative. This does not assert equal support;
/// it exists so downstream epistemic reasoning can represent live model
/// alternatives rather than collapsing identity into a boolean.
pub const X2370_MOLECULAR_INTERPRETATION: CompositionInterpretation = CompositionInterpretation {
    resonance: "X(2370)",
    composition: HadronicComposition::HadronicMolecule,
    status: InterpretationStatus::Proposed,
    evidence_key: "theory::x2370::hadronic_molecule_family",
};

/// Does a resonance share the quantum-number channel of a predicted glueball?
pub fn glueball_channel_matches(resonance: &ObservedResonance, level: &GlueballLevel) -> bool {
    resonance.jpc == Some(level.jpc)
}

/// Distance between a resonance mass and a glueball-level central prediction.
///
/// Returns MeV/c^2. This is a descriptive diagnostic only; it is deliberately
/// not converted into an identity/confidence score because theory and
/// experimental uncertainties are not interchangeable.
pub fn glueball_mass_gap_mev(resonance: &ObservedResonance, level: &GlueballLevel) -> f64 {
    (resonance.mass_mev.value - level.mass_gev.value * 1000.0).abs()
}

/// Resolve a built-in evidence key.
pub fn x2370_evidence(key: &str) -> Option<&'static EvidenceRecord> {
    X2370_EVIDENCE.iter().find(|record| record.key == key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x2370_observation_is_not_the_same_as_interpretation() {
        assert_eq!(X2370.status, ObservationStatus::Established);
        assert_eq!(
            X2370_GLUEBALL_INTERPRETATION.status,
            InterpretationStatus::StronglySupported
        );
        assert_ne!(
            X2370_GLUEBALL_INTERPRETATION.composition,
            X2370_MOLECULAR_INTERPRETATION.composition
        );
    }

    #[test]
    fn pseudoscalar_glueball_channel_matches_x2370_quantum_numbers() {
        let pseudoscalar = BASELINE_GLUEBALL_LEVELS
            .iter()
            .find(|level| level.jpc == Jpc::PSEUDOSCALAR)
            .expect("pseudoscalar glueball baseline");
        assert!(glueball_channel_matches(&X2370, pseudoscalar));
    }

    #[test]
    fn mass_proximity_is_diagnostic_not_identity() {
        let pseudoscalar = BASELINE_GLUEBALL_LEVELS
            .iter()
            .find(|level| level.jpc == Jpc::PSEUDOSCALAR)
            .expect("pseudoscalar glueball baseline");
        let gap = glueball_mass_gap_mev(&X2370, pseudoscalar);
        assert!(gap > 0.0);
        assert_eq!(X2370.status, ObservationStatus::Established);
        assert_ne!(
            X2370_GLUEBALL_INTERPRETATION.status,
            InterpretationStatus::Consensus
        );
    }

    #[test]
    fn evidence_records_are_traceable() {
        let mass_width = x2370_evidence("besiii::x2370::mass_width_decay_modes")
            .expect("mass/width evidence");
        assert_eq!(mass_width.persistent_id, "arXiv:2605.26495");
        assert_eq!(mass_width.kind, EvidenceKind::ExperimentalObservation);

        let suppression = x2370_evidence("besiii::x2370::kstar_k_suppression")
            .expect("K* K suppression evidence");
        assert_eq!(suppression.persistent_id, "arXiv:2607.20366");
        assert_eq!(suppression.effect, EvidenceEffect::Supports);
    }

    #[test]
    fn glueball_levels_are_theory_predictions() {
        for level in BASELINE_GLUEBALL_LEVELS {
            assert!(level.mass_gev.value > 0.0);
            assert!(matches!(
                level.approximation,
                QcdApproximation::PureYangMills | QcdApproximation::QuenchedQcd
            ));
        }
    }

    #[test]
    fn scalar_radius_is_positive_and_compact() {
        assert!(SCALAR_GLUEBALL_MASS_RADIUS_FM.value > 0.0);
        assert!(SCALAR_GLUEBALL_MASS_RADIUS_FM.value < 0.5);
    }
}
