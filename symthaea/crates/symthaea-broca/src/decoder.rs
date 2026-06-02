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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredTranslation {
    pub translator: String,
    pub source_decoder: String,
    pub grounding_surface: String,
    pub text: String,
    pub confidence: f32,
    pub evidence_level: String,
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

/// Deterministic structured-to-prose translator.
///
/// This is deliberately conservative: it turns the role/filler readout into
/// short English without sampling, `<unk>`, or autoregressive drift. Mamba can
/// still be used later as an optional humanizer, but this is the grounded
/// baseline that must remain inspectable.
#[derive(Debug, Clone, Default)]
pub struct StructuredProseTranslator;

impl StructuredProseTranslator {
    pub fn new() -> Self {
        Self
    }

    pub fn translate(&self, readout: &StructuredReadout) -> StructuredTranslation {
        let action = role_prime(readout, "ACTION").unwrap_or("MAYBE");
        let predicate = role_prime(readout, "PREDICATE").unwrap_or("THINK");
        let evaluator = role_prime(readout, "EVALUATOR").unwrap_or("TRUE");
        let patient = role_prime(readout, "PATIENT").unwrap_or("SOMETHING");
        let urgent = role_prime(readout, "TIME") == Some("NOW");
        let uncertain_reason = role_prime(readout, "REASON") == Some("MAYBE");

        let mut clauses = vec![action_sentence(action, patient).to_string()];
        if let Some(clause) = predicate_clause(predicate) {
            clauses.push(clause.to_string());
        }
        if let Some(clause) = evaluator_clause(evaluator) {
            clauses.push(clause.to_string());
        }
        if uncertain_reason {
            clauses.push("I may need more context.".to_string());
        }
        if urgent {
            clauses.push("This should happen now.".to_string());
        }

        StructuredTranslation {
            translator: "structured-prose-v1".to_string(),
            source_decoder: readout.decoder.clone(),
            grounding_surface: readout.surface.clone(),
            text: clauses.join(" "),
            confidence: readout.confidence.clamp(0.0, 1.0),
            evidence_level: "deterministic".to_string(),
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

fn role_prime<'a>(readout: &'a StructuredReadout, role: &str) -> Option<&'a str> {
    readout
        .roles
        .iter()
        .find(|fill| fill.role == role)
        .map(|fill| fill.prime.as_str())
}

fn action_sentence(action: &str, patient: &str) -> &'static str {
    match action {
        "THINK" => "I am thinking about this.",
        "DO" => "I can act on this.",
        "SAY" => "I can explain this.",
        "WANT" => "I want to clarify this.",
        "KNOW" => "I know enough to answer.",
        "FEEL" => "I feel this state changing.",
        "WITH" => "I am relating this to nearby context.",
        "MAYBE" => "I am uncertain about this.",
        _ if patient == "SOMETHING" => "I am attending to something.",
        _ => "I am attending to this.",
    }
}

fn predicate_clause(predicate: &str) -> Option<&'static str> {
    match predicate {
        "KNOW" => Some("The state is treated as known."),
        "THINK" => Some("The state is still being reasoned through."),
        "MAYBE" => Some("The state is uncertain."),
        _ => None,
    }
}

fn evaluator_clause(evaluator: &str) -> Option<&'static str> {
    match evaluator {
        "TRUE" => Some("The signal is consistent."),
        "GOOD" => Some("The signal is favorable."),
        "BAD" => Some("The signal is unfavorable."),
        "MAYBE" => Some("The signal needs verification."),
        _ => None,
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

    #[test]
    fn structured_prose_translation_preserves_grounding() {
        let genesis = GenesisSeed::from_phrase("structured-prose-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(4);
        let readout = decoder.decode(&channels);
        let translation = StructuredProseTranslator::new().translate(&readout);

        assert_eq!(translation.grounding_surface, readout.surface);
        assert!(translation.text.contains("answer"));
        assert!(!translation.text.contains("<unk>"));
        assert_eq!(translation.evidence_level, "deterministic");
    }
}
