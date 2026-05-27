// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Direct structured readout helpers for Mamba-bypass Broca decoding.

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::universal_semantics::{
    SemanticMolecule, SemanticMoleculeBasis, SemanticPrime, SyntacticRole,
};

use crate::encoder::ThoughtChannels;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredRoleFill {
    pub role: String,
    pub prime: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredReadout {
    pub decoder: String,
    pub intent: String,
    pub roles: Vec<StructuredRoleFill>,
    pub intensity: f32,
    pub confidence: f32,
    pub surface: String,
    pub molecule: SemanticMolecule,
}

/// Deterministic semantic molecule decoder.
///
/// This is not intended to be fluent English. It is the grounded "truth layer"
/// that Mamba or another translator can humanize later.
pub struct StructuredDecoder {
    basis: SemanticMoleculeBasis,
}

impl StructuredDecoder {
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            basis: SemanticMoleculeBasis::new(genesis),
        }
    }

    pub fn decode(&self, channels: &ThoughtChannels) -> StructuredReadout {
        let intent = active_intent_name(channels);
        let action = intent_prime(intent);
        let evaluator = if channels.valence() < -0.2 {
            SemanticPrime::Bad
        } else if channels.valence() > 0.2 {
            SemanticPrime::Good
        } else if channels.epistemic_ordinal() >= 2.5 {
            SemanticPrime::Maybe
        } else {
            SemanticPrime::True
        };
        let predicate = if channels.epistemic_ordinal() <= 0.5 {
            SemanticPrime::Know
        } else if channels.epistemic_ordinal() >= 2.5 {
            SemanticPrime::Maybe
        } else {
            SemanticPrime::Think
        };

        let mut structure = vec![
            (SyntacticRole::Agent, SemanticPrime::I),
            (SyntacticRole::Action, action),
            (SyntacticRole::Patient, SemanticPrime::Something),
            (SyntacticRole::Predicate, predicate),
            (SyntacticRole::Evaluator, evaluator),
        ];
        if channels.time_pressure() > 0.65 {
            structure.push((SyntacticRole::Time, SemanticPrime::Now));
        }
        if channels.domain_familiarity() < 0.35 {
            structure.push((SyntacticRole::Reason, SemanticPrime::Maybe));
        }

        let intensity =
            ((channels.psi() + channels.arousal() + channels.coherence()) / 3.0).clamp(0.0, 1.0);
        let molecule = SemanticMolecule::new(&self.basis, structure.clone(), intensity);
        let roles: Vec<_> = structure
            .iter()
            .map(|(role, prime)| StructuredRoleFill {
                role: role.name().to_string(),
                prime: prime.as_gate_name().to_string(),
            })
            .collect();

        StructuredReadout {
            decoder: "structured".to_string(),
            intent: intent.to_string(),
            surface: roles
                .iter()
                .map(|rf| format!("{}:{}", rf.role, rf.prime))
                .collect::<Vec<_>>()
                .join(" "),
            roles,
            intensity,
            confidence: ((channels.coherence() + (1.0 - channels.epistemic_ordinal() / 4.0)) / 2.0)
                .clamp(0.0, 1.0),
            molecule,
        }
    }
}

pub fn active_intent_name(channels: &ThoughtChannels) -> &'static str {
    let names = [
        "analyze", "create", "explain", "question", "answer", "reflect", "relate", "unknown",
    ];
    channels
        .channels
        .iter()
        .take(names.len())
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| names[idx])
        .unwrap_or("unknown")
}

fn intent_prime(intent: &str) -> SemanticPrime {
    match intent {
        "analyze" => SemanticPrime::Think,
        "create" => SemanticPrime::Do,
        "explain" => SemanticPrime::Say,
        "question" => SemanticPrime::Want,
        "answer" => SemanticPrime::Know,
        "reflect" => SemanticPrime::Feel,
        "relate" => SemanticPrime::With,
        _ => SemanticPrime::Maybe,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_decoder_reflects_active_intent() {
        let genesis = GenesisSeed::from_phrase("structured-decoder-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        assert_eq!(readout.intent, "explain");
        assert!(readout.surface.contains("ACTION:SAY"));
        assert!(!readout.roles.is_empty());
    }
}
