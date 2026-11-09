// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Loss-preserving normalization of Piper Plus's raw English IPA stream.
//!
//! Piper Plus emits one IPA character token at a time, with separate stress
//! marker tokens and parallel A1/A2/A3 prosody entries. This adapter rebuilds
//! word-level IPA text and preserves stress evidence. It deliberately does
//! not infer canonical phone units, syllable boundaries, contextual variants,
//! or OOV state. Those require a language-profile/lexicon layer and review.

use piper_plus_g2p::ProsodyInfo;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StressKind {
    Primary,
    Secondary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StressEvidence {
    /// Position in the original Piper token array.
    pub raw_token_index: usize,
    /// Position of the following IPA character in this word's reconstructed
    /// stream. This is not claimed to be a phone or syllable boundary.
    pub following_ipa_character_index: usize,
    pub kind: StressKind,
    pub a1: Option<i32>,
    pub a2: Option<i32>,
    pub a3: Option<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RawWordHypothesis {
    pub word_index: usize,
    /// Concatenated IPA character stream; not yet segmented into phones.
    pub raw_ipa: String,
    /// One-entry-per-character raw tokens after explicit stress markers and
    /// word separators have been removed. Combining diacritics and affricates
    /// remain unsegmented for a later canonical phone inventory.
    pub ipa_character_tokens: Vec<String>,
    /// Stress marker evidence only. No syllable boundaries are asserted.
    pub stress_evidence: Vec<StressEvidence>,
    /// Null means no independently validated syllabification was supplied.
    pub syllable_boundaries: Option<Vec<usize>>,
    /// The backend does not expose an independently calibrated OOV decision.
    pub oov_status: &'static str,
    /// Raw Piper output is a candidate hypothesis, never ground truth.
    pub resolution_status: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RawPronunciationHypothesis {
    pub language: &'static str,
    pub provider: &'static str,
    pub provider_version: &'static str,
    pub words: Vec<RawWordHypothesis>,
}

fn validate_prosody_alignment(
    tokens: &[String],
    prosody: &[Option<ProsodyInfo>],
) -> Result<(), String> {
    if tokens.len() != prosody.len() {
        return Err(format!(
            "Piper token/prosody length mismatch: {} tokens, {} prosody entries",
            tokens.len(),
            prosody.len()
        ));
    }
    for (index, token) in tokens.iter().enumerate() {
        if token.is_empty() {
            return Err(format!("Piper token {index} is empty"));
        }
        if token.chars().any(char::is_whitespace) && token != " " {
            return Err(format!(
                "Piper token {index} contains whitespace other than the word separator"
            ));
        }
        if let Some(info) = prosody[index] {
            if !(0..=2).contains(&info.a2) {
                return Err(format!(
                    "Piper token {index} has an unsupported English A2 stress value {}",
                    info.a2
                ));
            }
        }
    }
    Ok(())
}

/// Convert Piper's raw character token stream into word-level IPA hypotheses
/// without pretending that character tokens are phoneme units.
pub fn normalize_piper_english(
    tokens: &[String],
    prosody: &[Option<ProsodyInfo>],
) -> Result<RawPronunciationHypothesis, String> {
    validate_prosody_alignment(tokens, prosody)?;

    let mut words = Vec::new();
    let mut current_tokens: Vec<String> = Vec::new();
    let mut stress_evidence = Vec::new();
    let mut pending_stress: Option<(usize, StressKind, Option<ProsodyInfo>)> = None;
    let mut raw_position = 0usize;

    let flush_word = |words: &mut Vec<RawWordHypothesis>,
                      current_tokens: &mut Vec<String>,
                      stress_evidence: &mut Vec<StressEvidence>| {
        if current_tokens.is_empty() {
            return;
        }
        let raw_ipa = current_tokens.concat();
        words.push(RawWordHypothesis {
            word_index: words.len(),
            raw_ipa,
            ipa_character_tokens: std::mem::take(current_tokens),
            stress_evidence: std::mem::take(stress_evidence),
            syllable_boundaries: None,
            oov_status: "not_assessed",
            resolution_status: "needs_review",
        });
    };

    for (raw_token_index, token) in tokens.iter().enumerate() {
        match token.as_str() {
            " " => {
                if pending_stress.is_some() {
                    return Err(format!(
                        "stress marker at Piper token {} is not followed by an IPA character before a word boundary",
                        pending_stress.as_ref().unwrap().0
                    ));
                }
                flush_word(&mut words, &mut current_tokens, &mut stress_evidence);
                raw_position = 0;
            }
            "ˈ" | "ˌ" => {
                if pending_stress.is_some() {
                    return Err(format!(
                        "consecutive Piper stress markers at raw token {raw_token_index}"
                    ));
                }
                let kind = if token == "ˈ" {
                    StressKind::Primary
                } else {
                    StressKind::Secondary
                };
                let info = prosody[raw_token_index];
                let expected_a2 = match kind {
                    StressKind::Primary => 2,
                    StressKind::Secondary => 1,
                };
                if info.is_none_or(|value| value.a2 != expected_a2) {
                    return Err(format!(
                        "Piper stress marker at token {raw_token_index} disagrees with its A2 prosody value"
                    ));
                }
                pending_stress = Some((raw_token_index, kind, info));
            }
            _ => {
                if let Some((marker_index, kind, marker_info)) = pending_stress.take() {
                    let character_index = raw_position;
                    let following_info = prosody[raw_token_index];
                    if let (Some(marker), Some(following)) = (marker_info, following_info) {
                        if marker.a2 != following.a2 || marker.a3 != following.a3 {
                            return Err(format!(
                                "Piper stress/prosody alignment mismatch between raw tokens {marker_index} and {raw_token_index}"
                            ));
                        }
                    }
                    stress_evidence.push(StressEvidence {
                        raw_token_index: marker_index,
                        following_ipa_character_index: character_index,
                        kind,
                        a1: marker_info.map(|value| value.a1),
                        a2: marker_info.map(|value| value.a2),
                        a3: marker_info.map(|value| value.a3),
                    });
                }
                current_tokens.push(token.clone());
                raw_position += token.chars().count();
            }
        }
    }

    if let Some((marker_index, _, _)) = pending_stress {
        return Err(format!(
            "Piper stress marker at raw token {marker_index} has no following IPA character"
        ));
    }
    flush_word(&mut words, &mut current_tokens, &mut stress_evidence);
    if words.is_empty() {
        return Err("Piper returned no lexical word hypotheses".into());
    }

    Ok(RawPronunciationHypothesis {
        language: "en",
        provider: "piper-plus-g2p",
        provider_version: "0.5.3",
        words,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(a2: i32, a3: i32) -> Option<ProsodyInfo> {
        Some(ProsodyInfo { a1: 0, a2, a3 })
    }

    #[test]
    fn rebuilds_word_ipa_and_preserves_primary_and_secondary_stress() {
        let tokens = vec![
            "h".into(),
            "ə".into(),
            "l".into(),
            "ˈ".into(),
            "o".into(),
            "ʊ".into(),
            " ".into(),
            "ˌ".into(),
            "ɪ".into(),
            "n".into(),
        ];
        let prosody = vec![
            info(0, 5),
            info(0, 5),
            info(0, 5),
            info(2, 5),
            info(2, 5),
            info(2, 5),
            None,
            info(1, 2),
            info(1, 2),
            info(1, 2),
        ];
        let normalized = normalize_piper_english(&tokens, &prosody).unwrap();
        assert_eq!(normalized.words.len(), 2);
        assert_eq!(normalized.words[0].raw_ipa, "həloʊ");
        assert_eq!(normalized.words[0].stress_evidence.len(), 1);
        assert_eq!(normalized.words[0].stress_evidence[0].kind, StressKind::Primary);
        assert_eq!(normalized.words[0].stress_evidence[0].following_ipa_character_index, 3);
        assert_eq!(normalized.words[1].raw_ipa, "ɪn");
        assert_eq!(normalized.words[1].stress_evidence[0].kind, StressKind::Secondary);
    }

    #[test]
    fn does_not_claim_phone_segmentation_syllabification_or_oov_status() {
        let tokens = vec!["t".into(), "ʃ".into(), " ".into(), "x".into()];
        let prosody = vec![info(0, 2), info(0, 2), None, info(0, 1)];
        let normalized = normalize_piper_english(&tokens, &prosody).unwrap();
        assert_eq!(normalized.words[0].raw_ipa, "tʃ");
        assert_eq!(normalized.words[0].syllable_boundaries, None);
        assert_eq!(normalized.words[0].oov_status, "not_assessed");
        assert_eq!(normalized.words[0].resolution_status, "needs_review");
    }

    #[test]
    fn malformed_stress_or_parallel_metadata_fails_closed() {
        let tokens = vec!["ˈ".into(), "a".into()];
        let mismatched = vec![info(1, 1), info(1, 1)];
        assert!(normalize_piper_english(&tokens, &mismatched).is_err());
        assert!(normalize_piper_english(&tokens, &[info(2, 1)]).is_err());
        assert!(normalize_piper_english(&tokens, &[info(1, 1), info(1, 1)]).is_err());
        assert!(normalize_piper_english(&["a".into()], &[]).is_err());
    }

    #[test]
    fn dangling_or_duplicate_stress_markers_are_rejected() {
        let tokens = vec!["ˈ".into(), "ˌ".into(), "a".into()];
        let prosody = vec![info(2, 1), info(1, 1), info(1, 1)];
        assert!(normalize_piper_english(&tokens, &prosody).is_err());

        let tokens = vec!["a".into(), "ˈ".into()];
        let prosody = vec![info(0, 1), info(2, 1)];
        assert!(normalize_piper_english(&tokens, &prosody).is_err());
    }
}
