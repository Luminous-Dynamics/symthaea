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

/// The epistemic state supplied by the upstream thought channels.
///
/// These labels preserve the input disposition; they are not a probability
/// estimate and do not establish that a proposition is factually true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicDisposition {
    Certain,
    Probable,
    Uncertain,
    #[default]
    Unknown,
    OutOfDomain,
}

impl EpistemicDisposition {
    /// Map the encoder's ordered channel (0=Certain ... 4=OutOfDomain) to a
    /// stable categorical state. Non-finite input fails closed to Unknown.
    fn from_ordinal(ordinal: f32) -> Self {
        if !ordinal.is_finite() {
            return Self::Unknown;
        }
        if ordinal < 0.5 {
            Self::Certain
        } else if ordinal < 1.5 {
            Self::Probable
        } else if ordinal < 2.5 {
            Self::Uncertain
        } else if ordinal < 3.5 {
            Self::Unknown
        } else {
            Self::OutOfDomain
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Certain => "certain",
            Self::Probable => "probable",
            Self::Uncertain => "uncertain",
            Self::Unknown => "unknown",
            Self::OutOfDomain => "out_of_domain",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredRoleFill {
    pub role: String,
    pub prime: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredReadout {
    pub decoder: String,
    pub intent: String,
    /// The categorical disposition decoded from the upstream channel.
    /// Older serialized readouts default to Unknown rather than gaining certainty.
    #[serde(default)]
    pub epistemic_status: EpistemicDisposition,
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
    /// Status carried through from the readout without inference or upgrading.
    #[serde(default)]
    pub epistemic_status: EpistemicDisposition,
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
        let epistemic_status = EpistemicDisposition::from_ordinal(channels.epistemic_ordinal());
        let action = intent_prime(intent, epistemic_status);
        let evaluator = if channels.valence() < -0.2 {
            SemanticPrime::Bad
        } else if channels.valence() > 0.2 {
            SemanticPrime::Good
        } else if channels.epistemic_ordinal() >= 2.5 {
            SemanticPrime::Maybe
        } else {
            SemanticPrime::True
        };
        let predicate = match epistemic_status {
            EpistemicDisposition::Certain => SemanticPrime::Know,
            EpistemicDisposition::Probable => SemanticPrime::Think,
            EpistemicDisposition::Uncertain
            | EpistemicDisposition::Unknown
            | EpistemicDisposition::OutOfDomain => SemanticPrime::Maybe,
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
            surface: format!(
                "{} EPISTEMIC_STATUS:{}",
                roles
                    .iter()
                    .map(|rf| format!("{}:{}", rf.role, rf.prime))
                    .collect::<Vec<_>>()
                    .join(" "),
                epistemic_status.as_str()
            ),
            epistemic_status,
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
        let evaluator = role_prime(readout, "EVALUATOR").unwrap_or("TRUE");
        let patient = role_prime(readout, "PATIENT").unwrap_or("SOMETHING");
        let urgent = role_prime(readout, "TIME") == Some("NOW");
        let uncertain_reason = role_prime(readout, "REASON") == Some("MAYBE");

        let mut clauses = vec![action_sentence(
            readout.intent.as_str(),
            action,
            patient,
            readout.epistemic_status,
        )
        .to_string()];
        clauses.push(predicate_clause(readout.epistemic_status).to_string());
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
            epistemic_status: readout.epistemic_status,
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

fn intent_prime(intent: &str, epistemic_status: EpistemicDisposition) -> SemanticPrime {
    match intent {
        "analyze" => SemanticPrime::Think,
        "create" => SemanticPrime::Do,
        "explain" => SemanticPrime::Say,
        "question" => SemanticPrime::Want,
        "answer" => match epistemic_status {
            EpistemicDisposition::Certain => SemanticPrime::Know,
            EpistemicDisposition::Probable => SemanticPrime::Think,
            EpistemicDisposition::Uncertain
            | EpistemicDisposition::Unknown
            | EpistemicDisposition::OutOfDomain => SemanticPrime::Maybe,
        },
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

fn action_sentence(
    intent: &str,
    action: &str,
    patient: &str,
    epistemic_status: EpistemicDisposition,
) -> &'static str {
    // An Answer intent describes the requested act, not permission to claim
    // knowledge. The upstream epistemic disposition controls the wording.
    if intent == "answer" {
        return match epistemic_status {
            EpistemicDisposition::Certain => "The available evidence supports a direct answer.",
            EpistemicDisposition::Probable => "I can offer a qualified answer.",
            EpistemicDisposition::Uncertain => {
                "I cannot give a confident answer because the evidence is uncertain."
            }
            EpistemicDisposition::Unknown => {
                "I do not have enough information to answer reliably."
            }
            EpistemicDisposition::OutOfDomain => {
                "This question is outside the domain I can answer reliably."
            }
        };
    }

    match (intent, action) {
        ("analyze", "THINK") => "I am analyzing the situation.",
        ("create", "DO") => "I can create a concrete next step.",
        ("explain", "SAY") => "I can explain the situation.",
        ("question", "WANT") => "I want to ask a clarifying question.",
        ("reflect", "FEEL") => "I am reflecting on the state change.",
        ("relate", "WITH") => "I am relating this to nearby context.",
        (_, "THINK") => "I am thinking about this.",
        (_, "DO") => "I can act on this.",
        (_, "SAY") => "I can explain this.",
        (_, "WANT") => "I want to clarify this.",
        (_, "FEEL") => "I feel this state changing.",
        (_, "WITH") => "I am relating this to nearby context.",
        (_, "MAYBE") => "I am uncertain about this.",
        _ if patient == "SOMETHING" => "I am attending to something.",
        _ => "I am attending to this.",
    }
}

fn predicate_clause(epistemic_status: EpistemicDisposition) -> &'static str {
    match epistemic_status {
        EpistemicDisposition::Certain => "The state is treated as known.",
        EpistemicDisposition::Probable => "The state is supported, but not certain.",
        EpistemicDisposition::Uncertain => "The state remains uncertain.",
        EpistemicDisposition::Unknown => "The available information does not establish the state.",
        EpistemicDisposition::OutOfDomain => "The state is outside the supported domain.",
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
    fn answer_intent_never_overrides_epistemic_disposition() {
        let genesis = GenesisSeed::from_phrase("structured-epistemic-status-test");
        let decoder = StructuredDecoder::new(&genesis);
        let translator = StructuredProseTranslator::new();

        let cases = [
            (0.0, EpistemicDisposition::Certain, "direct answer"),
            (1.0, EpistemicDisposition::Probable, "qualified answer"),
            (2.0, EpistemicDisposition::Uncertain, "evidence is uncertain"),
            (3.0, EpistemicDisposition::Unknown, "not have enough information"),
            (4.0, EpistemicDisposition::OutOfDomain, "outside the domain"),
        ];

        for (ordinal, expected_status, expected_phrase) in cases {
            let mut channels = ThoughtChannels::with_intent(4);
            channels.set_epistemic(ordinal);
            let readout = decoder.decode(&channels);
            let translation = translator.translate(&readout);

            assert_eq!(readout.epistemic_status, expected_status);
            assert_eq!(translation.epistemic_status, expected_status);
            assert!(
                readout.surface.contains(&format!("EPISTEMIC_STATUS:{}", expected_status.as_str())),
                "surface must retain status {:?}",
                expected_status
            );
            assert!(
                translation.text.to_lowercase().contains(expected_phrase),
                "status {:?} should render '{expected_phrase}', got '{}'",
                expected_status,
                translation.text
            );
            if expected_status != EpistemicDisposition::Certain {
                assert!(
                    !translation.text.contains("I know enough to answer"),
                    "status {:?} must not claim knowledge: {}",
                    expected_status,
                    translation.text
                );
            }
        }
    }

    #[test]
    fn missing_serialized_epistemic_status_defaults_to_unknown() {
        let value = serde_json::json!({
            "decoder": "structured",
            "intent": "answer",
            "roles": [],
            "intensity": 0.0,
            "confidence": 0.0,
            "surface": ""
        });
        let readout: StructuredReadout = serde_json::from_value(value).unwrap();
        assert_eq!(readout.epistemic_status, EpistemicDisposition::Unknown);
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
