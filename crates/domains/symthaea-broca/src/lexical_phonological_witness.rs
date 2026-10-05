// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit lexical-to-phonological realization witnesses.
//!
//! This module does not perform grapheme-to-phoneme conversion and does not claim that a
//! phoneme sequence is linguistically correct merely because a caller supplied it. Instead,
//! it makes the caller's realization claim auditable: every lexical constituent is mapped to
//! one or more exact phonological segment indices/symbols, the mapping preserves lexical order,
//! and the mapping is bound to the exact lexical-binding provenance token.

use serde::{Deserialize, Serialize};

use crate::lexical_binding::LexicalMorphosyntacticBinding;
use crate::phonological_plan::PhonemeSlot;

/// Stable identity for the explicit lexical-to-phonological witness contract.
pub const LEXICAL_PHONOLOGICAL_WITNESS_VERSION: &str =
    "broca-lexical-phonological-witness-v1";

/// One explicit realization claim for one final lexical constituent position.
///
/// `segment_indices` refer to positions in the candidate phonological segment stream.
/// `symbols` stores the exact caller-asserted symbols at those positions so the witness
/// can be independently checked against a persisted phonological plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalPhonologicalMapping {
    /// Zero-based position in the final lexical constituent stream.
    pub lexical_position: usize,
    /// Exact zero-based indices into the phonological segment stream.
    pub segment_indices: Vec<usize>,
    /// Exact phoneme symbols asserted for those segment indices, in the same order.
    pub symbols: Vec<String>,
}

/// Complete explicit realization witness for one lexical binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalPhonologicalWitness {
    pub version: String,
    /// Exact provenance token of the lexical/morphosyntactic binding being realized.
    pub lexical_binding_provenance: String,
    pub mappings: Vec<LexicalPhonologicalMapping>,
}

impl LexicalPhonologicalWitness {
    /// Construct a witness only from an already validated lexical binding.
    ///
    /// The caller must still supply the constituent-to-segment mapping explicitly.
    pub fn new(
        binding: &LexicalMorphosyntacticBinding,
        mappings: Vec<LexicalPhonologicalMapping>,
    ) -> Result<Self, LexicalPhonologicalWitnessError> {
        binding
            .validate()
            .map_err(|_| LexicalPhonologicalWitnessError::InvalidLexicalBinding)?;

        let witness = Self {
            version: LEXICAL_PHONOLOGICAL_WITNESS_VERSION.to_string(),
            lexical_binding_provenance: binding.provenance_token(),
            mappings,
        };
        witness.validate_against_binding(binding)?;
        Ok(witness)
    }

    /// Validate the witness independently of a segment stream.
    pub fn validate(&self) -> Result<(), LexicalPhonologicalWitnessError> {
        if self.version != LEXICAL_PHONOLOGICAL_WITNESS_VERSION {
            return Err(LexicalPhonologicalWitnessError::InvalidVersion);
        }
        if !is_blake3_token(&self.lexical_binding_provenance) {
            return Err(LexicalPhonologicalWitnessError::InvalidLexicalBindingProvenance);
        }
        if self.mappings.is_empty() {
            return Err(LexicalPhonologicalWitnessError::EmptyMappings);
        }

        for (expected_position, mapping) in self.mappings.iter().enumerate() {
            if mapping.lexical_position != expected_position {
                return Err(LexicalPhonologicalWitnessError::NonContiguousLexicalPositions);
            }
            if mapping.segment_indices.is_empty() {
                return Err(LexicalPhonologicalWitnessError::EmptySegmentMapping {
                    lexical_position: mapping.lexical_position,
                });
            }
            if mapping.segment_indices.len() != mapping.symbols.len() {
                return Err(LexicalPhonologicalWitnessError::SymbolCountMismatch {
                    lexical_position: mapping.lexical_position,
                });
            }

            for (index, symbol) in mapping.symbols.iter().enumerate() {
                if symbol.trim().is_empty() {
                    return Err(LexicalPhonologicalWitnessError::EmptySymbol {
                        lexical_position: mapping.lexical_position,
                        symbol_offset: index,
                    });
                }
                if symbol.eq_ignore_ascii_case("SIL") {
                    return Err(LexicalPhonologicalWitnessError::SilenceMappedAsLexical {
                        lexical_position: mapping.lexical_position,
                        segment_index: mapping.segment_indices[index],
                    });
                }
            }

            for pair in mapping.segment_indices.windows(2) {
                if pair[0] >= pair[1] {
                    return Err(LexicalPhonologicalWitnessError::SegmentIndicesNotIncreasing {
                        lexical_position: mapping.lexical_position,
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate that the witness covers the exact final lexical constituent stream.
    pub fn validate_against_binding(
        &self,
        binding: &LexicalMorphosyntacticBinding,
    ) -> Result<(), LexicalPhonologicalWitnessError> {
        binding
            .validate()
            .map_err(|_| LexicalPhonologicalWitnessError::InvalidLexicalBinding)?;
        self.validate()?;

        let expected_provenance = binding.provenance_token();
        if self.lexical_binding_provenance != expected_provenance {
            return Err(LexicalPhonologicalWitnessError::LexicalBindingProvenanceMismatch);
        }
        if self.mappings.len() != binding.constituents.len() {
            return Err(LexicalPhonologicalWitnessError::LexicalCoverageMismatch);
        }

        Ok(())
    }

    /// Validate the witness against the exact candidate phonological segment stream.
    ///
    /// Every non-silence segment must be covered exactly once by a lexical constituent.
    /// Explicit `SIL` segments remain realization-layer material and are intentionally outside
    /// the lexical witness, so pause insertion cannot masquerade as lexical content.
    pub fn validate_against_segments(
        &self,
        binding: &LexicalMorphosyntacticBinding,
        segments: &[PhonemeSlot],
    ) -> Result<(), LexicalPhonologicalWitnessError> {
        self.validate_against_binding(binding)?;

        let mut covered = vec![false; segments.len()];
        let mut previous_segment_index = None;

        for mapping in &self.mappings {
            for (offset, &segment_index) in mapping.segment_indices.iter().enumerate() {
                let segment = segments
                    .get(segment_index)
                    .ok_or(LexicalPhonologicalWitnessError::SegmentIndexOutOfRange {
                        lexical_position: mapping.lexical_position,
                        segment_index,
                    })?;
                if segment.symbol.eq_ignore_ascii_case("SIL") {
                    return Err(LexicalPhonologicalWitnessError::SilenceMappedAsLexical {
                        lexical_position: mapping.lexical_position,
                        segment_index,
                    });
                }
                if covered[segment_index] {
                    return Err(LexicalPhonologicalWitnessError::DuplicateSegmentCoverage {
                        segment_index,
                    });
                }
                if previous_segment_index.is_some_and(|previous| segment_index <= previous) {
                    return Err(LexicalPhonologicalWitnessError::LexicalMappingOrderMismatch {
                        lexical_position: mapping.lexical_position,
                    });
                }
                previous_segment_index = Some(segment_index);
                if mapping.symbols[offset] != segment.symbol {
                    return Err(LexicalPhonologicalWitnessError::SegmentSymbolMismatch {
                        lexical_position: mapping.lexical_position,
                        segment_index,
                    });
                }
                covered[segment_index] = true;
            }
        }

        if segments.iter().enumerate().any(|(index, segment)| {
            !segment.symbol.eq_ignore_ascii_case("SIL") && !covered[index]
        }) {
            return Err(LexicalPhonologicalWitnessError::UncoveredPhonologicalSegment);
        }

        Ok(())
    }
}

fn is_blake3_token(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexicalPhonologicalWitnessError {
    InvalidVersion,
    InvalidLexicalBinding,
    InvalidLexicalBindingProvenance,
    LexicalBindingProvenanceMismatch,
    EmptyMappings,
    NonContiguousLexicalPositions,
    EmptySegmentMapping { lexical_position: usize },
    SymbolCountMismatch { lexical_position: usize },
    EmptySymbol { lexical_position: usize, symbol_offset: usize },
    SilenceMappedAsLexical { lexical_position: usize, segment_index: usize },
    SegmentIndicesNotIncreasing { lexical_position: usize },
    SegmentIndexOutOfRange { lexical_position: usize, segment_index: usize },
    DuplicateSegmentCoverage { segment_index: usize },
    SegmentSymbolMismatch { lexical_position: usize, segment_index: usize },
    LexicalMappingOrderMismatch { lexical_position: usize },
    LexicalCoverageMismatch,
    UncoveredPhonologicalSegment,
}

impl std::fmt::Display for LexicalPhonologicalWitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidVersion => write!(f, "lexical-to-phonological witness version is unsupported"),
            Self::InvalidLexicalBinding => write!(f, "lexical-to-phonological witness references an invalid lexical binding"),
            Self::InvalidLexicalBindingProvenance => write!(f, "lexical binding provenance must be a 64-character BLAKE3 hexadecimal token"),
            Self::LexicalBindingProvenanceMismatch => write!(f, "lexical-to-phonological witness does not match the exact lexical binding"),
            Self::EmptyMappings => write!(f, "lexical-to-phonological witness mappings must be non-empty"),
            Self::NonContiguousLexicalPositions => write!(f, "lexical-to-phonological mappings must cover contiguous lexical positions from zero"),
            Self::EmptySegmentMapping { lexical_position } => write!(f, "lexical position {lexical_position} must map to at least one phonological segment"),
            Self::SymbolCountMismatch { lexical_position } => write!(f, "lexical position {lexical_position} has mismatched segment-index and symbol counts"),
            Self::EmptySymbol { lexical_position, symbol_offset } => write!(f, "lexical position {lexical_position} has an empty phoneme symbol at offset {symbol_offset}"),
            Self::SilenceMappedAsLexical { lexical_position, segment_index } => write!(f, "SIL segment {segment_index} cannot be claimed as lexical realization for position {lexical_position}"),
            Self::SegmentIndicesNotIncreasing { lexical_position } => write!(f, "lexical position {lexical_position} must map to strictly increasing segment indices"),
            Self::SegmentIndexOutOfRange { lexical_position, segment_index } => write!(f, "lexical position {lexical_position} references out-of-range segment {segment_index}"),
            Self::DuplicateSegmentCoverage { segment_index } => write!(f, "phonological segment {segment_index} is claimed by multiple lexical constituents"),
            Self::SegmentSymbolMismatch { lexical_position, segment_index } => write!(f, "lexical position {lexical_position} symbol does not match phonological segment {segment_index}"),
            Self::LexicalMappingOrderMismatch { lexical_position } => write!(f, "lexical position {lexical_position} maps to a segment before an earlier lexical position"),
            Self::LexicalCoverageMismatch => write!(f, "lexical-to-phonological witness does not cover every lexical constituent exactly once"),
            Self::UncoveredPhonologicalSegment => write!(f, "a non-silence phonological segment is not covered by the lexical witness"),
        }
    }
}

impl std::error::Error for LexicalPhonologicalWitnessError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical_binding::{
        ConstituentDependency, LanguageRuleBinding, LanguageRuleStatus, LexemeBinding,
        LexicalSource,
    };
    use crate::linguistic_frame::{FormulationStrategy, LinguisticFrame};
    use crate::{GrammaticalFunction, MorphologicalFeature, SpeechPlan, StructuredDecoder, ThoughtChannels};
    use symthaea_core::genesis::GenesisSeed;

    fn frame() -> LinguisticFrame {
        let genesis = GenesisSeed::from_phrase("lexical-phonological-witness-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(4);
        let readout = decoder.decode(&channels);
        LinguisticFrame::from_speech_plan(&SpeechPlan::from_readout(&channels, &readout))
    }

    fn binding() -> LexicalMorphosyntacticBinding {
        let frame = frame();
        let mut constituents = frame
            .constituents
            .iter()
            .map(|slot| LexemeBinding {
                position: slot.position,
                source: LexicalSource::SemanticConstituent {
                    role: slot.role.clone(),
                    prime: slot.prime.clone(),
                },
                lemma: slot.prime.to_ascii_lowercase(),
                lexeme_id: format!("en:fixture:{}", slot.position),
                grammatical_function: GrammaticalFunction::Other("fixture".into()),
                morphology: Vec::new(),
                morphophonological_form: Some(slot.prime.to_ascii_lowercase()),
                provenance: "fixture:v1".into(),
                semantic_payload: true,
            })
            .collect::<Vec<_>>();

        constituents.insert(
            1,
            LexemeBinding {
                position: 1,
                source: LexicalSource::InsertedFunctionWord {
                    insertion_reason: "fixture:article".into(),
                },
                lemma: "the".into(),
                lexeme_id: "en:det:the".into(),
                grammatical_function: GrammaticalFunction::FunctionWord,
                morphology: Vec::new(),
                morphophonological_form: Some("the".into()),
                provenance: "fixture:v1".into(),
                semantic_payload: false,
            },
        );
        for (position, constituent) in constituents.iter_mut().enumerate() {
            constituent.position = position;
        }

        LexicalMorphosyntacticBinding::new(
            &frame,
            LanguageRuleBinding {
                language_tag: "en".into(),
                status: LanguageRuleStatus::Bound,
                rule_id: Some("fixture-rules-v1".into()),
                provenance: Some("fixture-rules".into()),
                unbound_reason: None,
            },
            constituents,
            Vec::new(),
            Vec::new(),
        )
        .expect("fixture lexical binding")
    }

    fn segments(binding: &LexicalMorphosyntacticBinding) -> Vec<PhonemeSlot> {
        (0..binding.constituents.len())
            .map(|index| {
                PhonemeSlot::new(
                    format!("P{index}"),
                    index,
                    crate::SyllableStress::Primary,
                    true,
                    false,
                    true,
                )
            })
            .collect()
    }

    fn witness_for_segments(
        binding: &LexicalMorphosyntacticBinding,
    ) -> LexicalPhonologicalWitness {
        let segments = segments(binding);
        LexicalPhonologicalWitness::new(
            binding,
            segments
                .iter()
                .enumerate()
                .map(|(index, segment)| LexicalPhonologicalMapping {
                    lexical_position: index,
                    segment_indices: vec![index],
                    symbols: vec![segment.symbol.clone()],
                })
                .collect(),
        )
        .expect("explicit witness")
    }

    #[test]
    fn valid_witness_covers_semantic_and_inserted_function_word_positions() {
        let binding = binding();
        let witness = witness_for_segments(&binding);
        assert!(witness.validate_against_segments(&binding, &segments(&binding)).is_ok());
        assert_eq!(witness.mappings.len(), 3);
    }

    #[test]
    fn swapped_lexical_mapping_fails_order_and_coverage_contract() {
        let binding = binding();
        let witness = LexicalPhonologicalWitness {
            version: LEXICAL_PHONOLOGICAL_WITNESS_VERSION.into(),
            lexical_binding_provenance: binding.provenance_token(),
            mappings: {
                let mut mappings = witness_for_segments(&binding).mappings;
                mappings.swap(0, 1);
                mappings
            },
        };

        assert_eq!(
            witness
                .validate_against_segments(&binding, &segments())
                .expect_err("swapped lexical realization must fail"),
            LexicalPhonologicalWitnessError::LexicalMappingOrderMismatch {
                lexical_position: 1,
            }
        );
    }

    #[test]
    fn duplicate_segment_claim_fails_closed() {
        let binding = binding();
        let mut mappings = witness_for_segments(&binding).mappings;
        let duplicated_index = mappings[0].segment_indices[0];
        mappings[1].segment_indices[0] = duplicated_index;
        mappings[1].symbols[0] = mappings[0].symbols[0].clone();
        let witness = LexicalPhonologicalWitness {
            version: LEXICAL_PHONOLOGICAL_WITNESS_VERSION.into(),
            lexical_binding_provenance: binding.provenance_token(),
            mappings,
        };

        assert_eq!(
            witness
                .validate_against_segments(&binding, &segments(&binding))
                .expect_err("duplicate segment claim must fail"),
            LexicalPhonologicalWitnessError::DuplicateSegmentCoverage {
                segment_index: duplicated_index,
            }
        );
    }

    #[test]
    fn one_lexical_constituent_may_cover_multiple_segments() {
        let binding = binding();
        let segments = vec![
            PhonemeSlot::new("P0A", 0, crate::SyllableStress::Primary, true, false, true),
            PhonemeSlot::new("P0B", 1, crate::SyllableStress::Primary, false, false, false),
            PhonemeSlot::new("P1", 2, crate::SyllableStress::Primary, true, false, true),
            PhonemeSlot::new("P2", 3, crate::SyllableStress::Primary, true, false, true),
            PhonemeSlot::new("P3", 4, crate::SyllableStress::Primary, true, false, true),
            PhonemeSlot::new("P4", 5, crate::SyllableStress::Primary, true, false, true),
        ];

        let mut mappings = Vec::new();
        mappings.push(LexicalPhonologicalMapping {
            lexical_position: 0,
            segment_indices: vec![0, 1],
            symbols: vec!["P0A".into(), "P0B".into()],
        });
        for position in 1..binding.constituents.len() {
            mappings.push(LexicalPhonologicalMapping {
                lexical_position: position,
                segment_indices: vec![position + 1],
                symbols: vec![segments[position + 1].symbol.clone()],
            });
        }
        mappings.truncate(6);

        let witness = LexicalPhonologicalWitness::new(&binding, mappings)
            .expect("multi-segment lexical mapping should validate");
        assert!(witness.validate_against_segments(&binding, &segments).is_ok());
    }

    #[test]
    fn out_of_range_segment_reference_fails_closed() {
        let binding = binding();
        let mut witness = witness_for_segments(&binding);
        witness.mappings[0].segment_indices[0] = usize::MAX;

        assert_eq!(
            witness
                .validate_against_segments(&binding, &segments(&binding))
                .expect_err("out-of-range segment reference must fail"),
            LexicalPhonologicalWitnessError::SegmentIndexOutOfRange {
                lexical_position: 0,
                segment_index: usize::MAX,
            }
        );
    }

    #[test]
    fn provenance_mismatch_fails_closed() {
        let binding = binding();
        let mut witness = witness_for_segments(&binding);
        witness.lexical_binding_provenance =
            blake3::hash(b"other-binding").to_hex().to_string();

        assert_eq!(
            witness
                .validate_against_binding(&binding)
                .expect_err("different lexical binding must fail"),
            LexicalPhonologicalWitnessError::LexicalBindingProvenanceMismatch
        );
    }

    #[test]
    fn missing_mapping_fails_closed() {
        let binding = binding();
        let witness = LexicalPhonologicalWitness {
            version: LEXICAL_PHONOLOGICAL_WITNESS_VERSION.into(),
            lexical_binding_provenance: binding.provenance_token(),
            mappings: vec![
                LexicalPhonologicalMapping { lexical_position: 0, segment_indices: vec![0], symbols: vec!["AY".into()] },
                LexicalPhonologicalMapping { lexical_position: 1, segment_indices: vec![1], symbols: vec!["DH".into()] },
            ],
        };

        assert_eq!(
            witness
                .validate_against_binding(&binding)
                .expect_err("missing lexical constituent mapping must fail"),
            LexicalPhonologicalWitnessError::LexicalCoverageMismatch
        );
    }

    #[test]
    fn unmapped_silence_is_allowed_but_not_lexicalized() {
        let binding = binding();
        let witness = witness_for_segments(&binding);
        let mut segments = segments(&binding);
        let silence_index = segments.len();
        segments.push(PhonemeSlot::new(
            "SIL",
            silence_index,
            crate::SyllableStress::None,
            true,
            false,
            true,
        ));

        assert!(witness.validate_against_segments(&binding, &segments).is_ok());
    }

    #[test]
    fn symbol_tampering_fails_closed() {
        let binding = binding();
        let witness = witness_for_segments(&binding);
        let mut segments = segments();
        segments[1].symbol = "T".into();

        assert_eq!(
            witness
                .validate_against_segments(&binding, &segments)
                .expect_err("symbol tampering must fail"),
            LexicalPhonologicalWitnessError::SegmentSymbolMismatch {
                lexical_position: 1,
                segment_index: 1,
            }
        );
    }
}
