// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unified composition ontology for conventional and exotic hadrons.
//!
//! This is an interpretation model, not an assertion that constituent language
//! is exact at every scale. Mixed states are first-class so quarkonium/glueball
//! mixing and other superpositions are representable without forced identity.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HadronBasisState {
    QuarkAntiquark { flavor: Option<String> },
    ThreeQuark { flavor: Option<String> },
    Tetraquark { flavor: Option<String> },
    Pentaquark { flavor: Option<String> },
    Glueball,
    Hybrid { valence: Option<String> },
    HadronicMolecule { constituents: Vec<String> },
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixtureComponent {
    pub basis: HadronBasisState,
    /// Optional normalized weight. `None` means the component is proposed but
    /// its admixture is not yet quantitatively established.
    pub weight: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompositionModel {
    Pure(HadronBasisState),
    Mixed(Vec<MixtureComponent>),
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompositionError {
    EmptyMixture,
    InvalidWeight(f64),
    WeightsDoNotNormalize(f64),
}

impl CompositionModel {
    pub fn mixed(components: Vec<MixtureComponent>) -> Result<Self, CompositionError> {
        if components.is_empty() {
            return Err(CompositionError::EmptyMixture);
        }
        for component in &components {
            if let Some(weight) = component.weight {
                if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                    return Err(CompositionError::InvalidWeight(weight));
                }
            }
        }
        if components.iter().all(|component| component.weight.is_some()) {
            let sum: f64 = components.iter().filter_map(|component| component.weight).sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(CompositionError::WeightsDoNotNormalize(sum));
            }
        }
        Ok(Self::Mixed(components))
    }

    pub fn contains_basis(&self, target: &HadronBasisState) -> bool {
        match self {
            Self::Pure(basis) => basis == target,
            Self::Mixed(components) => components.iter().any(|component| &component.basis == target),
            Self::Unknown => false,
        }
    }

    pub fn is_mixed(&self) -> bool {
        matches!(self, Self::Mixed(_))
    }
}

pub fn quarkonium_glueball_mixture(
    flavor: Option<String>,
    quarkonium_weight: Option<f64>,
    glueball_weight: Option<f64>,
) -> Result<CompositionModel, CompositionError> {
    CompositionModel::mixed(vec![
        MixtureComponent {
            basis: HadronBasisState::QuarkAntiquark { flavor },
            weight: quarkonium_weight,
        },
        MixtureComponent {
            basis: HadronBasisState::Glueball,
            weight: glueball_weight,
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn represents_all_requested_exotic_families() {
        let states = [
            HadronBasisState::Tetraquark { flavor: None },
            HadronBasisState::Pentaquark { flavor: None },
            HadronBasisState::Glueball,
            HadronBasisState::Hybrid { valence: None },
            HadronBasisState::HadronicMolecule { constituents: vec!["Sigma".into(), "anti-Sigma".into()] },
        ];
        assert_eq!(states.len(), 5);
    }

    #[test]
    fn quarkonium_glueball_mixing_is_first_class() {
        let model = quarkonium_glueball_mixture(Some("light flavor singlet".into()), Some(0.35), Some(0.65)).unwrap();
        assert!(model.is_mixed());
        assert!(model.contains_basis(&HadronBasisState::Glueball));
    }

    #[test]
    fn arbitrary_unknown_mixture_weights_are_allowed() {
        let model = CompositionModel::mixed(vec![
            MixtureComponent { basis: HadronBasisState::Glueball, weight: None },
            MixtureComponent { basis: HadronBasisState::Hybrid { valence: None }, weight: None },
            MixtureComponent { basis: HadronBasisState::HadronicMolecule { constituents: vec!["A".into(), "B".into()] }, weight: None },
        ]).unwrap();
        assert!(model.is_mixed());
    }

    #[test]
    fn quantitative_weights_must_normalize() {
        let result = quarkonium_glueball_mixture(None, Some(0.8), Some(0.8));
        assert!(matches!(result, Err(CompositionError::WeightsDoNotNormalize(_))));
    }
}
