// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-aware hadron spectroscopy.
//!
//! This module separates measured resonance observables from theoretical
//! interpretations of internal composition. That distinction is essential for
//! frontier QCD states such as glueball candidates, hybrids, multiquarks, and
//! hadronic molecules.

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

/// A directly observed resonance, independent of its internal interpretation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionInterpretation {
    pub resonance: &'static str,
    pub composition: HadronicComposition,
    pub status: InterpretationStatus,
    /// Human-readable provenance key; intended to resolve into the evidence
    /// graph rather than serve as a free-floating confidence scalar.
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

/// Baseline pure-gauge/quenched glueball spectrum.
///
/// These values are intentionally represented as theory predictions, not as
/// discovered particles. Physical resonances can mix strongly with quarkonia
/// once dynamical quarks are included.
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

/// BESIII-era X(2370) resonance fixture.
///
/// The resonance and J^PC assignment are kept separate from its proposed
/// glueball-dominant composition. This prevents a future reinterpretation from
/// mutating the underlying experimental observation.
pub const X2370: ObservedResonance = ObservedResonance {
    name: "X(2370)",
    jpc: Some(Jpc::PSEUDOSCALAR),
    mass_mev: Measurement::asymmetric(2359.0, 14.0, 13.0),
    width_mev: Measurement::asymmetric(170.0, 29.0, 44.0),
    status: ObservationStatus::Established,
};

/// Current leading interpretation represented conservatively.
pub const X2370_GLUEBALL_INTERPRETATION: CompositionInterpretation = CompositionInterpretation {
    resonance: "X(2370)",
    composition: HadronicComposition::Glueball,
    status: InterpretationStatus::StronglySupported,
    evidence_key: "besiii::x2370::pseudoscalar_glueball_dominant",
};

/// Competing molecular interpretation retained explicitly so the model can
/// reason over disagreement rather than collapse it into a boolean flag.
pub const X2370_MOLECULAR_INTERPRETATION: CompositionInterpretation = CompositionInterpretation {
    resonance: "X(2370)",
    composition: HadronicComposition::HadronicMolecule,
    status: InterpretationStatus::Proposed,
    evidence_key: "theory::x2370::sigma_antisigma_molecule",
};

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
        assert_eq!(X2370.jpc, Some(Jpc::PSEUDOSCALAR));
        assert!(BASELINE_GLUEBALL_LEVELS
            .iter()
            .any(|level| level.jpc == Jpc::PSEUDOSCALAR));
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
}
