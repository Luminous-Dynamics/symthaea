// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing pronunciation lattice.
//!
//! A renderer must consume a resolved plan. It must not rerun G2P, silently
//! choose a pronunciation variant, or invent a phone when a word is OOV.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PronunciationPlan {
    pub schema_version: u32,
    pub language: String,
    pub source_text: String,
    pub words: Vec<WordPronunciation>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WordPronunciation {
    pub start_byte: usize,
    pub end_byte: usize,
    pub surface: String,
    pub state: ResolutionState,
    pub candidates: Vec<PronunciationCandidate>,
    pub selected_candidate_id: Option<String>,
    pub selection_receipt: Option<SelectionReceipt>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionState {
    NeedsReview,
    Selected,
    OutOfVocabulary,
    Unsupported,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PronunciationCandidate {
    pub candidate_id: String,
    pub provenance: CandidateProvenance,
    /// Raw provider score only. It is not a calibrated probability and must
    /// never be compared across providers without an explicit calibration.
    pub raw_provider_score: Option<f32>,
    pub syllables: Vec<PronunciationSyllable>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateProvenance {
    pub source: CandidateSource,
    pub provider_id: String,
    pub provider_version: String,
    /// Digest/revision of the exact lexicon, engine, annotation bundle, or
    /// override source. A mutable path or display name is not sufficient.
    pub source_digest: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateSource {
    Dictionary,
    RuleG2p,
    ModelG2p,
    HumanReviewed,
    UserOverride,
    CommunityReviewed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PronunciationSyllable {
    pub stress: u8,
    pub phones: Vec<Phone>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Phone {
    /// One canonical normalized IPA token. Stress, syllable boundary and
    /// length markers are represented in typed fields, not mixed into tokens.
    pub ipa: String,
    pub kind: PhoneKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneKind {
    Vowel,
    Consonant,
    SyllabicConsonant,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionReceipt {
    pub authority: SelectionAuthority,
    /// Non-empty reference to a review/override record; avoid storing personal
    /// reviewer identity in the pronunciation artifact itself.
    pub evidence_reference: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionAuthority {
    HumanReviewer,
    User,
    CommunityReview,
    ApprovedPolicy,
}

impl PronunciationPlan {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported pronunciation-plan schema version {}",
                self.schema_version
            ));
        }
        if self.language.trim().is_empty()
            || self.language.bytes().any(|byte| byte.is_ascii_whitespace())
        {
            return Err("language must be an explicit non-empty tag".into());
        }
        if self.source_text.trim().is_empty() {
            return Err("source_text must not be empty".into());
        }
        if self.words.is_empty() {
            return Err("at least one word span is required".into());
        }

        let mut previous_end = 0usize;
        let mut previous_surface_end = 0usize;
        for (word_index, word) in self.words.iter().enumerate() {
            if word.start_byte >= word.end_byte || word.end_byte > self.source_text.len() {
                return Err(format!("word {word_index} has an invalid byte span"));
            }
            if !self.source_text.is_char_boundary(word.start_byte)
                || !self.source_text.is_char_boundary(word.end_byte)
            {
                return Err(format!("word {word_index} span splits a UTF-8 code point"));
            }
            if word.start_byte < previous_end {
                return Err(format!("word {word_index} overlaps or is out of order"));
            }
            if word.surface != self.source_text[word.start_byte..word.end_byte] {
                return Err(format!("word {word_index} surface does not match source_text"));
            }
            if word.surface.chars().all(char::is_whitespace) {
                return Err(format!("word {word_index} contains only whitespace"));
            }
            validate_word_state(word_index, word)?;

            previous_end = word.end_byte;
            previous_surface_end = word.end_byte;
        }

        // Every non-whitespace source character must belong to exactly one
        // declared word span. Text cannot disappear between tokenization and G2P.
        for (byte_index, character) in self.source_text.char_indices() {
            if character.is_whitespace() {
                continue;
            }
            let covered = self.words.iter().any(|word| {
                byte_index >= word.start_byte && byte_index < word.end_byte
            });
            if !covered {
                return Err(format!(
                    "source character {character:?} at byte {byte_index} is not covered by a word span"
                ));
            }
        }

        if previous_surface_end > self.source_text.len() {
            return Err("word span exceeds source text".into());
        }
        Ok(())
    }

    /// A renderer may consume only fully selected pronunciations. Review,
    /// OOV, and unsupported words must be resolved upstream, not guessed
    /// during synthesis.
    pub fn validate_renderable(&self) -> Result<(), String> {
        self.validate()?;
        for (word_index, word) in self.words.iter().enumerate() {
            if word.state != ResolutionState::Selected {
                return Err(format!(
                    "word {word_index} is {:?}; rendering requires an explicit selected pronunciation",
                    word.state
                ));
            }
        }
        Ok(())
    }
}

fn validate_word_state(index: usize, word: &WordPronunciation) -> Result<(), String> {
    let mut ids = std::collections::BTreeSet::new();
    for candidate in &word.candidates {
        if candidate.candidate_id.trim().is_empty()
            || !ids.insert(candidate.candidate_id.as_str())
        {
            return Err(format!("word {index} has an empty or duplicate candidate ID"));
        }
        if candidate.provenance.provider_id.trim().is_empty()
            || candidate.provenance.provider_version.trim().is_empty()
            || candidate.provenance.source_digest.trim().is_empty()
        {
            return Err(format!("candidate {} has incomplete provenance", candidate.candidate_id));
        }
        if candidate
            .raw_provider_score
            .is_some_and(|score| !score.is_finite())
        {
            return Err(format!("candidate {} has a non-finite raw score", candidate.candidate_id));
        }
        if candidate.syllables.is_empty() {
            return Err(format!("candidate {} has no syllables", candidate.candidate_id));
        }
        for (syllable_index, syllable) in candidate.syllables.iter().enumerate() {
            if syllable.stress > 2 || syllable.phones.is_empty() {
                return Err(format!(
                    "candidate {} syllable {syllable_index} has invalid stress or no phones",
                    candidate.candidate_id
                ));
            }
            let has_nucleus = syllable.phones.iter().any(|phone| {
                matches!(phone.kind, PhoneKind::Vowel | PhoneKind::SyllabicConsonant)
            });
            if !has_nucleus {
                return Err(format!(
                    "candidate {} syllable {syllable_index} has no vowel or syllabic nucleus",
                    candidate.candidate_id
                ));
            }
            for (phone_index, phone) in syllable.phones.iter().enumerate() {
                if phone.ipa.trim().is_empty()
                    || phone.ipa.chars().any(char::is_whitespace)
                    || phone.ipa == "ˈ"
                    || phone.ipa == "ˌ"
                    || phone.ipa == "ː"
                {
                    return Err(format!(
                        "candidate {} syllable {syllable_index} phone {phone_index} is not one normalized IPA token",
                        candidate.candidate_id
                    ));
                }
            }
        }
    }

    match word.state {
        ResolutionState::NeedsReview => {
            if word.candidates.is_empty()
                || word.selected_candidate_id.is_some()
                || word.selection_receipt.is_some()
            {
                return Err(format!(
                    "word {index} needs review and must retain candidates without a selection"
                ));
            }
        }
        ResolutionState::Selected => {
            let Some(selected) = word.selected_candidate_id.as_deref() else {
                return Err(format!("word {index} is selected without a candidate ID"));
            };
            if !ids.contains(selected) {
                return Err(format!(
                    "word {index} selects unknown pronunciation candidate {selected}"
                ));
            }
            let Some(receipt) = &word.selection_receipt else {
                return Err(format!("word {index} selection has no evidence receipt"));
            };
            if receipt.evidence_reference.trim().is_empty() {
                return Err(format!("word {index} selection evidence reference is empty"));
            }
        }
        ResolutionState::OutOfVocabulary | ResolutionState::Unsupported => {
            if !word.candidates.is_empty()
                || word.selected_candidate_id.is_some()
                || word.selection_receipt.is_some()
            {
                return Err(format!(
                    "word {index} OOV/unsupported state must not invent or select a pronunciation"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance(source: CandidateSource, provider: &str) -> CandidateProvenance {
        CandidateProvenance {
            source,
            provider_id: provider.into(),
            provider_version: "1.0.0".into(),
            source_digest: "blake3:fixture".into(),
        }
    }

    fn candidate(id: &str, vowel: &str, source: CandidateSource) -> PronunciationCandidate {
        PronunciationCandidate {
            candidate_id: id.into(),
            provenance: provenance(source, "fixture-g2p"),
            raw_provider_score: Some(0.42),
            syllables: vec![PronunciationSyllable {
                stress: 1,
                phones: vec![
                    Phone { ipa: "ɹ".into(), kind: PhoneKind::Consonant },
                    Phone { ipa: vowel.into(), kind: PhoneKind::Vowel },
                    Phone { ipa: "d".into(), kind: PhoneKind::Consonant },
                ],
            }],
        }
    }

    fn word(
        surface: &str,
        start_byte: usize,
        state: ResolutionState,
        candidates: Vec<PronunciationCandidate>,
    ) -> WordPronunciation {
        WordPronunciation {
            start_byte,
            end_byte: start_byte + surface.len(),
            surface: surface.into(),
            state,
            candidates,
            selected_candidate_id: None,
            selection_receipt: None,
        }
    }

    #[test]
    fn contextual_heteronym_retains_multiple_candidates_until_review() {
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![word(
                "read",
                0,
                ResolutionState::NeedsReview,
                vec![
                    candidate("past", "ɛ", CandidateSource::Dictionary),
                    candidate("present", "iː", CandidateSource::ModelG2p),
                ],
            )],
        };
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn selected_candidate_requires_a_receipt() {
        let mut selected = word(
            "read",
            0,
            ResolutionState::Selected,
            vec![candidate("present", "iː", CandidateSource::Dictionary)],
        );
        selected.selected_candidate_id = Some("present".into());
        selected.selection_receipt = Some(SelectionReceipt {
            authority: SelectionAuthority::HumanReviewer,
            evidence_reference: "review:fixture-001".into(),
        });
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![selected],
        };
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn out_of_vocabulary_is_explicit_and_contains_no_guessed_phone() {
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "xqzpl".into(),
            words: vec![word("xqzpl", 0, ResolutionState::OutOfVocabulary, vec![])],
        };
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn unrepresented_source_text_cannot_be_silently_dropped() {
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "light bright".into(),
            words: vec![word(
                "light",
                0,
                ResolutionState::NeedsReview,
                vec![candidate("light-1", "aɪ", CandidateSource::Dictionary)],
            )],
        };
        let error = plan.validate().unwrap_err();
        assert!(error.contains("not covered"), "{error}");
    }

    #[test]
    fn spans_must_be_valid_utf8_boundaries_and_match_the_surface() {
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "café".into(),
            words: vec![word(
                "caf",
                0,
                ResolutionState::NeedsReview,
                vec![candidate("cafe", "æ", CandidateSource::Dictionary)],
            )],
        };
        assert!(plan.validate().is_err());

        let split = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "café".into(),
            words: vec![WordPronunciation {
                start_byte: 0,
                end_byte: 4,
                surface: "caf".into(),
                state: ResolutionState::NeedsReview,
                candidates: vec![candidate("cafe", "æ", CandidateSource::Dictionary)],
                selected_candidate_id: None,
                selection_receipt: None,
            }],
        };
        assert!(split.validate().is_err());
    }

    #[test]
    fn unresolved_or_unknown_candidate_cannot_be_marked_selected() {
        let mut unknown = word(
            "read",
            0,
            ResolutionState::Selected,
            vec![candidate("past", "ɛ", CandidateSource::Dictionary)],
        );
        unknown.selected_candidate_id = Some("missing".into());
        unknown.selection_receipt = Some(SelectionReceipt {
            authority: SelectionAuthority::User,
            evidence_reference: "user-choice:1".into(),
        });
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "read".into(),
            words: vec![unknown],
        };
        assert!(plan.validate().unwrap_err().contains("unknown pronunciation"));
    }

    #[test]
    fn phones_must_be_normalized_tokens_and_syllables_need_a_nucleus() {
        let mut invalid = candidate("bad", "ɛ", CandidateSource::Dictionary);
        invalid.syllables[0].phones[1].ipa = "ˈɛ".into();
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "read".into(),
            words: vec![word("read", 0, ResolutionState::NeedsReview, vec![invalid])],
        };
        assert!(plan.validate().unwrap_err().contains("normalized IPA"));

        let mut no_nucleus = candidate("no-nucleus", "ɛ", CandidateSource::Dictionary);
        no_nucleus.syllables[0].phones.retain(|phone| phone.kind == PhoneKind::Consonant);
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "read".into(),
            words: vec![word("read", 0, ResolutionState::NeedsReview, vec![no_nucleus])],
        };
        assert!(plan.validate().unwrap_err().contains("no vowel"));
    }

    #[test]
    fn non_finite_provider_scores_and_incomplete_provenance_fail_closed() {
        let mut bad_score = candidate("bad-score", "ɛ", CandidateSource::ModelG2p);
        bad_score.raw_provider_score = Some(f32::NAN);
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "read".into(),
            words: vec![word("read", 0, ResolutionState::NeedsReview, vec![bad_score])],
        };
        assert!(plan.validate().unwrap_err().contains("non-finite"));

        let mut bad_source = candidate("bad-source", "ɛ", CandidateSource::ModelG2p);
        bad_source.provenance.source_digest.clear();
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en".into(),
            source_text: "read".into(),
            words: vec![word("read", 0, ResolutionState::NeedsReview, vec![bad_source])],
        };
        assert!(plan.validate().unwrap_err().contains("incomplete provenance"));
    }
    #[test]
    fn unresolved_heteronym_is_valid_but_not_renderable() {
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![word(
                "read",
                0,
                ResolutionState::NeedsReview,
                vec![
                    candidate("past", "ɛ", CandidateSource::Dictionary),
                    candidate("present", "iː", CandidateSource::ModelG2p),
                ],
            )],
        };
        assert!(plan.validate().is_ok());
        assert!(plan.validate_renderable().unwrap_err().contains("rendering requires"));
    }

    #[test]
    fn resolved_pronunciation_is_renderable() {
        let mut selected = word(
            "read",
            0,
            ResolutionState::Selected,
            vec![candidate("past", "ɛ", CandidateSource::Dictionary)],
        );
        selected.selected_candidate_id = Some("past".into());
        selected.selection_receipt = Some(SelectionReceipt {
            authority: SelectionAuthority::HumanReviewer,
            evidence_reference: "review:read-past-001".into(),
        });
        let plan = PronunciationPlan {
            schema_version: 1,
            language: "en-US".into(),
            source_text: "read".into(),
            words: vec![selected],
        };
        assert!(plan.validate_renderable().is_ok());
    }

}
