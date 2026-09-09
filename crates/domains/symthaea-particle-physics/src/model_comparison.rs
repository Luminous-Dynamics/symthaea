// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Candidate/model comparison for hadron spectroscopy.
//!
//! The engine identifies observables on which candidate models make different
//! qualitative predictions. It deliberately does not manufacture posterior
//! probabilities from qualitative signatures.

use crate::exotic_hadrons::CompositionModel;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExpectedSignature {
    Enhanced,
    Suppressed,
    Present,
    Absent,
    Compatible,
    Unspecified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservablePrediction {
    pub observable: String,
    pub expectation: ExpectedSignature,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateModel {
    pub id: String,
    pub label: String,
    pub composition: CompositionModel,
    pub predictions: Vec<ObservablePrediction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelExpectation {
    pub model_id: String,
    pub expectation: ExpectedSignature,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscriminatingObservable {
    pub observable: String,
    pub expectations: Vec<ModelExpectation>,
}

/// Return only observables where at least two specified model expectations differ.
pub fn discriminating_observables(models: &[CandidateModel]) -> Vec<DiscriminatingObservable> {
    let mut by_observable: BTreeMap<String, Vec<ModelExpectation>> = BTreeMap::new();
    for model in models {
        for prediction in &model.predictions {
            if prediction.expectation == ExpectedSignature::Unspecified {
                continue;
            }
            by_observable
                .entry(prediction.observable.clone())
                .or_default()
                .push(ModelExpectation {
                    model_id: model.id.clone(),
                    expectation: prediction.expectation,
                });
        }
    }

    by_observable
        .into_iter()
        .filter_map(|(observable, expectations)| {
            let distinct: BTreeSet<_> = expectations.iter().map(|e| e.expectation).collect();
            (distinct.len() > 1).then_some(DiscriminatingObservable {
                observable,
                expectations,
            })
        })
        .collect()
}

/// Rank by how many distinct qualitative signatures are predicted, then by
/// how many candidate models speak to the observable. This is an experiment-
/// selection heuristic, not a statistical confidence score.
pub fn rank_discriminators(models: &[CandidateModel]) -> Vec<DiscriminatingObservable> {
    let mut discriminators = discriminating_observables(models);
    discriminators.sort_by(|a, b| {
        let a_distinct: BTreeSet<_> = a.expectations.iter().map(|e| e.expectation).collect();
        let b_distinct: BTreeSet<_> = b.expectations.iter().map(|e| e.expectation).collect();
        b_distinct
            .len()
            .cmp(&a_distinct.len())
            .then_with(|| b.expectations.len().cmp(&a.expectations.len()))
            .then_with(|| a.observable.cmp(&b.observable))
    });
    discriminators
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exotic_hadrons::HadronBasisState;

    fn candidate(id: &str, basis: HadronBasisState, kstar: ExpectedSignature) -> CandidateModel {
        CandidateModel {
            id: id.into(),
            label: id.into(),
            composition: CompositionModel::Pure(basis),
            predictions: vec![
                ObservablePrediction {
                    observable: "decay::kstar_k".into(),
                    expectation: kstar,
                    rationale: "candidate-dependent flavor/threshold signature".into(),
                },
                ObservablePrediction {
                    observable: "jpc::0-+".into(),
                    expectation: ExpectedSignature::Compatible,
                    rationale: "all three candidates can occupy this channel".into(),
                },
            ],
        }
    }

    #[test]
    fn finds_glueball_qbarq_molecule_discriminator() {
        let models = vec![
            candidate("glueball", HadronBasisState::Glueball, ExpectedSignature::Suppressed),
            candidate(
                "qbarq",
                HadronBasisState::QuarkAntiquark { flavor: None },
                ExpectedSignature::Present,
            ),
            candidate(
                "molecule",
                HadronBasisState::HadronicMolecule {
                    constituents: vec!["Sigma".into(), "anti-Sigma".into()],
                },
                ExpectedSignature::Enhanced,
            ),
        ];
        let discriminators = rank_discriminators(&models);
        assert_eq!(discriminators.len(), 1);
        assert_eq!(discriminators[0].observable, "decay::kstar_k");
        assert_eq!(discriminators[0].expectations.len(), 3);
    }

    #[test]
    fn shared_predictions_are_not_fake_discriminators() {
        let models = vec![
            candidate("a", HadronBasisState::Glueball, ExpectedSignature::Suppressed),
            candidate("b", HadronBasisState::Glueball, ExpectedSignature::Suppressed),
        ];
        assert!(discriminating_observables(&models).is_empty());
    }
}
