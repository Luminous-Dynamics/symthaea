// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative, Rust-native Humdrum **kern source adapter.
//!
//! This parser does not convert through MIDI or MusicXML. It preserves
//! Humdrum records and spine-local source identity, parses a deliberately
//! bounded note/rest/multiple-stop subset, and derives per-spine measure maps
//! only from explicit barline records. Spine split/join/exchange/add operations
//! are Unsupported in V1; they are never silently flattened.

use crate::barline_eval::{
    MeasureKind, MeasureMap, MeasureMapStatus, MeasureSpan, RationalBeat,
};
use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const HUMDRUM_KERN_SCHEMA_V1: &str = "muse-humdrum-kern-score-v1";
pub const HUMDRUM_KERN_CANONICALIZATION_V1: &str =
    "humdrum-kern-fixed-spines-rational-quarter-beats-v1";

const MAX_SOURCE_BYTES: usize = 32 * 1024 * 1024;
const MAX_SOURCE_LINES: usize = 1_000_000;
const MAX_SPINES: usize = 256;
const MAX_EVENTS_PER_SPINE: usize = 200_000;
const MAX_TOKEN_BYTES: usize = 4096;
const MAX_DOTS: u32 = 16;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumdrumRecordKindV1 {
    Blank,
    GlobalComment,
    LocalComment,
    ExclusiveInterpretation,
    Interpretation,
    Data,
    Termination,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumdrumSourceRecordV1 {
    pub line_number: usize,
    pub kind: HumdrumRecordKindV1,
    /// Original source record without its line terminator.
    pub raw: String,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumdrumInterpretationKindV1 {
    Clef,
    KeySignature,
    Meter,
    Tempo,
    Instrument,
    Transposition,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumdrumInterpretationV1 {
    pub source_line: usize,
    pub source_order: usize,
    pub kind: HumdrumInterpretationKindV1,
    pub raw_token: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumdrumKernPitchV1 {
    pub step: String,
    pub octave: i32,
    /// Explicit alteration only: -1 flat, 0 natural, +1 sharp, etc.
    /// Key-signature/accidental state is intentionally not inferred in V1.
    pub explicit_alter: Option<i8>,
    pub explicit_accidental: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumdrumKernEventKindV1 {
    Pitched(HumdrumKernPitchV1),
    Rest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumdrumKernEventV1 {
    pub source_line: usize,
    pub source_order: usize,
    pub spine_index: usize,
    /// Index within a whitespace-delimited multiple-stop token.
    pub subtoken_index: usize,
    pub score_offset: RationalBeat,
    pub performed_duration: RationalBeat,
    pub kind: HumdrumKernEventKindV1,
    /// Tie/slur/beam/articulation/ornament and editorial signifiers retained
    /// as opaque suffix data until a dedicated semantic decoder is qualified.
    pub signifiers: String,
    pub raw_subtoken: String,
    pub raw_spine_token: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumdrumKernBarlineV1 {
    pub source_line: usize,
    pub source_order: usize,
    pub spine_index: usize,
    pub score_offset: RationalBeat,
    /// The measure number and optional suffix following the equals sign(s).
    pub measure_label: Option<String>,
    pub raw_token: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumdrumKernSpineV1 {
    pub spine_index: usize,
    pub exclusive_interpretation: String,
    pub events: Vec<HumdrumKernEventV1>,
    pub barlines: Vec<HumdrumKernBarlineV1>,
    pub interpretations: Vec<HumdrumInterpretationV1>,
    pub measure_map: MeasureMap,
    pub final_cursor: RationalBeat,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumdrumKernScoreV1 {
    pub schema: String,
    pub canonicalization_version: String,
    pub source_sha256: String,
    pub score_clock_id: String,
    pub spines: Vec<HumdrumKernSpineV1>,
    pub source_records: Vec<HumdrumSourceRecordV1>,
    /// Canonical commitment excluding this field.
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HumdrumKernError {
    SourceTooLarge { bytes: usize, maximum: usize },
    InvalidUtf8,
    EmptyScoreClockId,
    TooManyLines { count: usize, maximum: usize },
    TooManySpines { count: usize, maximum: usize },
    MissingExclusiveInterpretation,
    NonKernSpine { spine_index: usize, interpretation: String },
    InconsistentSpineCount { line_number: usize, expected: usize, actual: usize },
    UnsupportedSpinePath { line_number: usize, token: String },
    EarlySpineTermination { line_number: usize },
    MissingSpineTermination,
    RecordAfterTermination { line_number: usize },
    LocalCommentBeforeExclusive { line_number: usize },
    EmptySpineToken { line_number: usize, spine_index: usize },
    TokenTooLarge { line_number: usize, bytes: usize },
    InvalidReciprocal { line_number: usize, token: String },
    InvalidPitch { line_number: usize, token: String },
    UnsupportedAccidental { line_number: usize, token: String },
    UnequalMultipleStopDurations { line_number: usize, spine_index: usize },
    EventBudgetExceeded { spine_index: usize, maximum: usize },
    ArithmeticOverflow,
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedKernAtom {
    duration: RationalBeat,
    kind: HumdrumKernEventKindV1,
    signifiers: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpineCursor {
    current: RationalBeat,
    maximum: RationalBeat,
}

impl SpineCursor {
    fn new() -> Self {
        Self {
            current: RationalBeat::new(0, 1).expect("zero rational is valid"),
            maximum: RationalBeat::new(0, 1).expect("zero rational is valid"),
        }
    }
}

pub fn parse_humdrum_kern(
    bytes: &[u8],
    score_clock_id: &str,
) -> Result<HumdrumKernScoreV1, HumdrumKernError> {
    if bytes.len() > MAX_SOURCE_BYTES {
        return Err(HumdrumKernError::SourceTooLarge {
            bytes: bytes.len(),
            maximum: MAX_SOURCE_BYTES,
        });
    }
    if score_clock_id.trim().is_empty() {
        return Err(HumdrumKernError::EmptyScoreClockId);
    }
    let source = std::str::from_utf8(bytes).map_err(|_| HumdrumKernError::InvalidUtf8)?;
    let lines: Vec<&str> = source.split('\n').collect();
    if lines.len() > MAX_SOURCE_LINES {
        return Err(HumdrumKernError::TooManyLines {
            count: lines.len(),
            maximum: MAX_SOURCE_LINES,
        });
    }

    let source_sha256 = sha256_hex(bytes);
    let mut records = Vec::with_capacity(lines.len());
    let mut spines: Vec<HumdrumKernSpineV1> = Vec::new();
    let mut cursors: Vec<SpineCursor> = Vec::new();
    let mut active = false;
    let mut terminated = false;
    let mut source_order = 0usize;

    for (zero_line, source_line) in lines.iter().enumerate() {
        let line_number = zero_line + 1;
        let raw = source_line.strip_suffix('\r').unwrap_or(source_line);
        if raw.is_empty() {
            records.push(HumdrumSourceRecordV1 {
                line_number,
                kind: HumdrumRecordKindV1::Blank,
                raw: raw.to_string(),
                fields: Vec::new(),
            });
            continue;
        }
        if raw.starts_with("!!") {
            records.push(HumdrumSourceRecordV1 {
                line_number,
                kind: HumdrumRecordKindV1::GlobalComment,
                raw: raw.to_string(),
                fields: Vec::new(),
            });
            continue;
        }
        if terminated {
            return Err(HumdrumKernError::RecordAfterTermination { line_number });
        }

        let fields: Vec<String> = raw.split('\t').map(str::to_string).collect();
        if raw.starts_with('!') {
            if !active {
                return Err(HumdrumKernError::LocalCommentBeforeExclusive { line_number });
            }
            check_field_count(line_number, spines.len(), fields.len())?;
            records.push(HumdrumSourceRecordV1 {
                line_number,
                kind: HumdrumRecordKindV1::LocalComment,
                raw: raw.to_string(),
                fields,
            });
            continue;
        }

        if raw.starts_with('*') {
            if !active {
                if fields.len() > MAX_SPINES {
                    return Err(HumdrumKernError::TooManySpines {
                        count: fields.len(),
                        maximum: MAX_SPINES,
                    });
                }
                for (spine_index, field) in fields.iter().enumerate() {
                    if field != "**kern" {
                        return Err(HumdrumKernError::NonKernSpine {
                            spine_index,
                            interpretation: field.clone(),
                        });
                    }
                }
                if fields.is_empty() {
                    return Err(HumdrumKernError::MissingExclusiveInterpretation);
                }
                for spine_index in 0..fields.len() {
                    spines.push(HumdrumKernSpineV1 {
                        spine_index,
                        exclusive_interpretation: "**kern".into(),
                        events: Vec::new(),
                        barlines: Vec::new(),
                        interpretations: Vec::new(),
                        measure_map: MeasureMap {
                            source_id: format!("humdrum-sha256:{source_sha256}#spine={spine_index}"),
                            score_clock_id: score_clock_id.to_string(),
                            canonicalization_version: HUMDRUM_KERN_CANONICALIZATION_V1.into(),
                            status: MeasureMapStatus::Unknown {
                                reason: "no explicit **kern barline boundaries have been parsed".into(),
                            },
                            measures: Vec::new(),
                        },
                        final_cursor: RationalBeat::new(0, 1)
                            .map_err(|_| HumdrumKernError::ArithmeticOverflow)?,
                    });
                    cursors.push(SpineCursor::new());
                }
                active = true;
                records.push(HumdrumSourceRecordV1 {
                    line_number,
                    kind: HumdrumRecordKindV1::ExclusiveInterpretation,
                    raw: raw.to_string(),
                    fields,
                });
                continue;
            }

            check_field_count(line_number, spines.len(), fields.len())?;
            if fields.iter().any(|token| {
                matches!(token.as_str(), "*^" | "*v" | "*x" | "*+")
            }) {
                let token = fields.iter()
                    .find(|token| matches!(token.as_str(), "*^" | "*v" | "*x" | "*+"))
                    .cloned()
                    .unwrap_or_default();
                return Err(HumdrumKernError::UnsupportedSpinePath {
                    line_number,
                    token,
                });
            }
            if fields.iter().any(|token| token == "*-") {
                if fields.iter().any(|token| token != "*-") {
                    return Err(HumdrumKernError::EarlySpineTermination { line_number });
                }
                terminated = true;
                records.push(HumdrumSourceRecordV1 {
                    line_number,
                    kind: HumdrumRecordKindV1::Termination,
                    raw: raw.to_string(),
                    fields,
                });
                continue;
            }
            if fields.iter().any(|token| token.starts_with("**")) {
                return Err(HumdrumKernError::UnsupportedSpinePath {
                    line_number,
                    token: fields.iter().find(|token| token.starts_with("**"))
                        .cloned().unwrap_or_default(),
                });
            }
            for (spine_index, token) in fields.iter().enumerate() {
                spines[spine_index].interpretations.push(HumdrumInterpretationV1 {
                    source_line: line_number,
                    source_order,
                    kind: interpretation_kind(token),
                    raw_token: token.clone(),
                });
                source_order = source_order.checked_add(1)
                    .ok_or(HumdrumKernError::ArithmeticOverflow)?;
            }
            records.push(HumdrumSourceRecordV1 {
                line_number,
                kind: HumdrumRecordKindV1::Interpretation,
                raw: raw.to_string(),
                fields,
            });
            continue;
        }

        if !active {
            return Err(HumdrumKernError::MissingExclusiveInterpretation);
        }
        check_field_count(line_number, spines.len(), fields.len())?;
        for (spine_index, field) in fields.iter().enumerate() {
            if field.is_empty() {
                return Err(HumdrumKernError::EmptySpineToken {
                    line_number,
                    spine_index,
                });
            }
            if field.len() > MAX_TOKEN_BYTES {
                return Err(HumdrumKernError::TokenTooLarge {
                    line_number,
                    bytes: field.len(),
                });
            }

            if field.starts_with('=') {
                let label = parse_barline_label(field);
                spines[spine_index].barlines.push(HumdrumKernBarlineV1 {
                    source_line: line_number,
                    source_order,
                    spine_index,
                    score_offset: cursors[spine_index].current,
                    measure_label: label,
                    raw_token: field.clone(),
                });
                source_order = source_order.checked_add(1)
                    .ok_or(HumdrumKernError::ArithmeticOverflow)?;
                continue;
            }
            if field == "." {
                continue;
            }

            let subtokens: Vec<&str> = field.split_whitespace().collect();
            if subtokens.is_empty() {
                return Err(HumdrumKernError::EmptySpineToken {
                    line_number,
                    spine_index,
                });
            }
            let mut parsed_atoms = Vec::with_capacity(subtokens.len());
            for subtoken in subtokens.iter().copied() {
                parsed_atoms.push(parse_kern_atom(subtoken, line_number)?);
            }
            let duration = parsed_atoms[0].duration;
            if parsed_atoms.iter().any(|atom| atom.duration != duration) {
                return Err(HumdrumKernError::UnequalMultipleStopDurations {
                    line_number,
                    spine_index,
                });
            }
            if spines[spine_index].events.len() + parsed_atoms.len() > MAX_EVENTS_PER_SPINE {
                return Err(HumdrumKernError::EventBudgetExceeded {
                    spine_index,
                    maximum: MAX_EVENTS_PER_SPINE,
                });
            }
            for (subtoken_index, atom) in parsed_atoms.into_iter().enumerate() {
                spines[spine_index].events.push(HumdrumKernEventV1 {
                    source_line: line_number,
                    source_order,
                    spine_index,
                    subtoken_index,
                    score_offset: cursors[spine_index].current,
                    performed_duration: atom.duration,
                    kind: atom.kind,
                    signifiers: atom.signifiers,
                    raw_subtoken: subtokens[subtoken_index].to_string(),
                    raw_spine_token: field.clone(),
                });
                source_order = source_order.checked_add(1)
                    .ok_or(HumdrumKernError::ArithmeticOverflow)?;
            }
            cursors[spine_index].current = cursors[spine_index].current
                .checked_add(duration)
                .map_err(|_| HumdrumKernError::ArithmeticOverflow)?;
            if cursors[spine_index].current.compare_exact(cursors[spine_index].maximum)
                == Ordering::Greater
            {
                cursors[spine_index].maximum = cursors[spine_index].current;
            }
        }
        records.push(HumdrumSourceRecordV1 {
            line_number,
            kind: HumdrumRecordKindV1::Data,
            raw: raw.to_string(),
            fields,
        });
    }

    if !active {
        return Err(HumdrumKernError::MissingExclusiveInterpretation);
    }
    if !terminated {
        return Err(HumdrumKernError::MissingSpineTermination);
    }
    for (spine_index, spine) in spines.iter_mut().enumerate() {
        spine.final_cursor = cursors[spine_index].maximum;
        spine.measure_map = build_spine_measure_map(
            &source_sha256,
            score_clock_id,
            spine_index,
            &spine.barlines,
            cursors[spine_index].maximum,
        )?;
    }

    let mut score = HumdrumKernScoreV1 {
        schema: HUMDRUM_KERN_SCHEMA_V1.into(),
        canonicalization_version: HUMDRUM_KERN_CANONICALIZATION_V1.into(),
        source_sha256,
        score_clock_id: score_clock_id.to_string(),
        spines,
        source_records: records,
        report_sha256: String::new(),
    };
    score.report_sha256 = humdrum_kern_score_commitment(&score)?;
    Ok(score)
}

fn check_field_count(
    line_number: usize,
    expected: usize,
    actual: usize,
) -> Result<(), HumdrumKernError> {
    if expected != actual {
        return Err(HumdrumKernError::InconsistentSpineCount {
            line_number,
            expected,
            actual,
        });
    }
    Ok(())
}

fn interpretation_kind(token: &str) -> HumdrumInterpretationKindV1 {
    if token.starts_with("*clef") {
        HumdrumInterpretationKindV1::Clef
    } else if token.starts_with("*k[") {
        HumdrumInterpretationKindV1::KeySignature
    } else if token.starts_with("*M") && !token.starts_with("*MM") {
        HumdrumInterpretationKindV1::Meter
    } else if token.starts_with("*MM") {
        HumdrumInterpretationKindV1::Tempo
    } else if token.starts_with("*ITr") {
        HumdrumInterpretationKindV1::Transposition
    } else if token.starts_with("*I") {
        HumdrumInterpretationKindV1::Instrument
    } else {
        HumdrumInterpretationKindV1::Other
    }
}

fn parse_barline_label(token: &str) -> Option<String> {
    let equals = token.bytes().take_while(|byte| *byte == b'=').count();
    let suffix = token.get(equals..)?;
    let label: String = suffix
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric())
        .collect();
    if label.is_empty() { None } else { Some(label) }
}

fn build_spine_measure_map(
    source_sha256: &str,
    score_clock_id: &str,
    spine_index: usize,
    barlines: &[HumdrumKernBarlineV1],
    final_cursor: RationalBeat,
) -> Result<MeasureMap, HumdrumKernError> {
    let source_id = format!("humdrum-sha256:{source_sha256}#spine={spine_index}");
    if barlines.is_empty() {
        return Ok(MeasureMap {
            source_id,
            score_clock_id: score_clock_id.into(),
            canonicalization_version: HUMDRUM_KERN_CANONICALIZATION_V1.into(),
            status: MeasureMapStatus::Unknown {
                reason: "no explicit **kern barline records are present in this spine".into(),
            },
            measures: Vec::new(),
        });
    }
    let zero = RationalBeat::new(0, 1).map_err(|_| HumdrumKernError::ArithmeticOverflow)?;
    let mut boundaries = vec![zero];
    for barline in barlines {
        if barline.score_offset.compare_exact(*boundaries.last().unwrap())
            == Ordering::Greater
        {
            boundaries.push(barline.score_offset);
        }
    }
    if boundaries.len() < 2 {
        return Ok(MeasureMap {
            source_id,
            score_clock_id: score_clock_id.into(),
            canonicalization_version: HUMDRUM_KERN_CANONICALIZATION_V1.into(),
            status: MeasureMapStatus::Unknown {
                reason: "explicit barlines do not bound a positive-duration measure".into(),
            },
            measures: Vec::new(),
        });
    }
    let last_boundary = *boundaries.last().unwrap();
    if final_cursor.compare_exact(last_boundary) == Ordering::Greater {
        return Ok(MeasureMap {
            source_id,
            score_clock_id: score_clock_id.into(),
            canonicalization_version: HUMDRUM_KERN_CANONICALIZATION_V1.into(),
            status: MeasureMapStatus::Unknown {
                reason: "final spine content extends beyond the last explicit barline".into(),
            },
            measures: Vec::new(),
        });
    }

    let mut measures = Vec::with_capacity(boundaries.len() - 1);
    for (measure_index, pair) in boundaries.windows(2).enumerate() {
        let start = pair[0];
        let end = pair[1];
        measures.push(MeasureSpan {
            start,
            end,
            kind: MeasureKind::Unclassified,
        });
        let _ = measure_index;
    }
    Ok(MeasureMap {
        source_id,
        score_clock_id: score_clock_id.into(),
        canonicalization_version: HUMDRUM_KERN_CANONICALIZATION_V1.into(),
        status: MeasureMapStatus::Supported,
        measures,
    })
}

fn parse_kern_atom(
    token: &str,
    line_number: usize,
) -> Result<ParsedKernAtom, HumdrumKernError> {
    if token.is_empty() {
        return Err(HumdrumKernError::InvalidPitch {
            line_number,
            token: token.into(),
        });
    }
    let bytes = token.as_bytes();
    let digit_count = bytes.iter().take_while(|byte| byte.is_ascii_digit()).count();
    if digit_count == 0 {
        return Err(HumdrumKernError::InvalidReciprocal {
            line_number,
            token: token.into(),
        });
    }
    let reciprocal = &token[..digit_count];
    let mut rest = &token[digit_count..];
    let dot_count = rest.bytes().take_while(|byte| *byte == b'.').count() as u32;
    if dot_count > MAX_DOTS {
        return Err(HumdrumKernError::InvalidReciprocal {
            line_number,
            token: token.into(),
        });
    }
    rest = &rest[dot_count..];
    let duration = reciprocal_duration(reciprocal, dot_count).ok_or_else(|| {
        HumdrumKernError::InvalidReciprocal {
            line_number,
            token: token.into(),
        }
    })?;

    if let Some(signifiers) = rest.strip_prefix('r') {
        return Ok(ParsedKernAtom {
            duration,
            kind: HumdrumKernEventKindV1::Rest,
            signifiers: signifiers.to_string(),
        });
    }

    let pitch_bytes = rest.as_bytes();
    let Some(first) = pitch_bytes.first().copied() else {
        return Err(HumdrumKernError::InvalidPitch {
            line_number,
            token: token.into(),
        });
    };
    if !first.is_ascii_alphabetic() || !matches!(first.to_ascii_lowercase(), b'a'..=b'g') {
        return Err(HumdrumKernError::InvalidPitch {
            line_number,
            token: token.into(),
        });
    }
    let letter_count = pitch_bytes
        .iter()
        .take_while(|byte| **byte == first)
        .count();
    let step = (first as char).to_ascii_uppercase().to_string();
    let octave = if first.is_ascii_lowercase() {
        4_i32.checked_add(i32::try_from(letter_count - 1).map_err(|_| {
            HumdrumKernError::InvalidPitch {
                line_number,
                token: token.into(),
            }
        })?)
    } else {
        3_i32.checked_sub(i32::try_from(letter_count - 1).map_err(|_| {
            HumdrumKernError::InvalidPitch {
                line_number,
                token: token.into(),
            }
        })?)
    }
    .ok_or_else(|| HumdrumKernError::InvalidPitch {
        line_number,
        token: token.into(),
    })?;

    let after_pitch = &rest[letter_count..];
    let accidental_length = after_pitch
        .bytes()
        .take_while(|byte| matches!(*byte, b'#' | b'-' | b'n'))
        .count();
    let accidental = &after_pitch[..accidental_length];
    let signifiers = &after_pitch[accidental_length..];
    let explicit_alter = if accidental.is_empty() {
        None
    } else if accidental == "n" {
        Some(0)
    } else if accidental.bytes().all(|byte| byte == b'#') {
        Some(i8::try_from(accidental.len()).map_err(|_| {
            HumdrumKernError::UnsupportedAccidental {
                line_number,
                token: token.into(),
            }
        })?)
    } else if accidental.bytes().all(|byte| byte == b'-') {
        Some(-i8::try_from(accidental.len()).map_err(|_| {
            HumdrumKernError::UnsupportedAccidental {
                line_number,
                token: token.into(),
            }
        })?)
    } else {
        return Err(HumdrumKernError::UnsupportedAccidental {
            line_number,
            token: token.into(),
        });
    };

    Ok(ParsedKernAtom {
        duration,
        kind: HumdrumKernEventKindV1::Pitched(HumdrumKernPitchV1 {
            step,
            octave,
            explicit_alter,
            explicit_accidental: if accidental.is_empty() {
                None
            } else {
                Some(accidental.into())
            },
        }),
        signifiers: signifiers.to_string(),
    })
}

/// Reciprocal rhythm is expressed in quarter-note beats. E.g. 4 = 1,
/// 8 = 1/2, 2 = 2, 1 = 4, 0 = breve = 8, 00 = longa = 16.
fn reciprocal_duration(reciprocal: &str, dots: u32) -> Option<RationalBeat> {
    let (base_numerator, base_denominator): (i128, i128) = match reciprocal {
        "0" => (8, 1),
        "00" => (16, 1),
        "000" => (32, 1),
        _ => {
            let value = reciprocal.parse::<i128>().ok()?;
            if value <= 0 {
                return None;
            }
            (4, value)
        }
    };
    let dot_denominator = 1_i128.checked_shl(dots)?;
    let dot_numerator = dot_denominator.checked_mul(2)?.checked_sub(1)?;
    let numerator = base_numerator.checked_mul(dot_numerator)?;
    let denominator = base_denominator.checked_mul(dot_denominator)?;
    let divisor = gcd_i128(numerator, denominator);
    let numerator = i64::try_from(numerator / divisor).ok()?;
    let denominator = i64::try_from(denominator / divisor).ok()?;
    RationalBeat::new(numerator, denominator).ok()
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.abs().max(1)
}

pub fn humdrum_kern_score_commitment(
    score: &HumdrumKernScoreV1,
) -> Result<String, HumdrumKernError> {
    let mut payload = score.clone();
    payload.report_sha256.clear();
    canonical_json_sha256(&payload).map_err(|_| HumdrumKernError::SerializationFailed)
}

pub fn validate_humdrum_kern_replay(
    bytes: &[u8],
    score_clock_id: &str,
    supplied: &HumdrumKernScoreV1,
) -> Result<bool, HumdrumKernError> {
    let replayed = parse_humdrum_kern(bytes, score_clock_id)?;
    Ok(replayed == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> HumdrumKernScoreV1 {
        parse_humdrum_kern(source.as_bytes(), "kern-test-clock").unwrap()
    }

    #[test]
    fn parses_single_spine_pitch_rhythm_and_explicit_measure_boundaries() {
        let source = "!!!OTL: Simple\n**kern\n=1\n4c\n8d\n8e\n4f\n=2\n*-\n";
        let score = parse(source);
        assert_eq!(score.spines.len(), 1);
        assert_eq!(score.spines[0].events.len(), 4);
        assert_eq!(score.spines[0].events[0].score_offset, RationalBeat::new(0, 1).unwrap());
        assert_eq!(score.spines[0].events[1].score_offset, RationalBeat::new(1, 1).unwrap());
        assert_eq!(score.spines[0].events[2].score_offset, RationalBeat::new(3, 2).unwrap());
        assert_eq!(score.spines[0].events[3].score_offset, RationalBeat::new(2, 1).unwrap());
        assert_eq!(score.spines[0].measure_map.status, MeasureMapStatus::Supported);
        assert_eq!(score.spines[0].measure_map.measures.len(), 1);
        assert_eq!(score.spines[0].measure_map.measures[0].end, RationalBeat::new(3, 1).unwrap());
    }

    #[test]
    fn parses_pitch_octave_mapping_and_explicit_accidentals() {
        let source = "**kern\n4C\n4B\n4c\n4cc#\n4DD-\n*-\n";
        let score = parse(source);
        let pitches: Vec<_> = score.spines[0].events.iter().map(|event| match &event.kind {
            HumdrumKernEventKindV1::Pitched(pitch) => (pitch.step.clone(), pitch.octave, pitch.explicit_alter),
            HumdrumKernEventKindV1::Rest => panic!("expected pitch"),
        }).collect();
        assert_eq!(pitches, vec![
            ("C".into(), 3, None),
            ("B".into(), 3, None),
            ("C".into(), 4, None),
            ("C".into(), 5, Some(1)),
            ("D".into(), 2, Some(-1)),
        ]);
    }

    #[test]
    fn reciprocal_durations_and_dots_are_exact() {
        assert_eq!(reciprocal_duration("4", 0), Some(RationalBeat::new(1, 1).unwrap()));
        assert_eq!(reciprocal_duration("8", 0), Some(RationalBeat::new(1, 2).unwrap()));
        assert_eq!(reciprocal_duration("2", 0), Some(RationalBeat::new(2, 1).unwrap()));
        assert_eq!(reciprocal_duration("1", 0), Some(RationalBeat::new(4, 1).unwrap()));
        assert_eq!(reciprocal_duration("0", 0), Some(RationalBeat::new(8, 1).unwrap()));
        assert_eq!(reciprocal_duration("4", 1), Some(RationalBeat::new(3, 2).unwrap()));
        assert_eq!(reciprocal_duration("8", 2), Some(RationalBeat::new(7, 8).unwrap()));
    }

    #[test]
    fn preserves_rests_and_multiple_stop_source_order() {
        let source = "**kern\n4c 4e 4g\n4r\n*-\n";
        let score = parse(source);
        assert_eq!(score.spines[0].events.len(), 4);
        assert_eq!(score.spines[0].events[0].subtoken_index, 0);
        assert_eq!(score.spines[0].events[1].subtoken_index, 1);
        assert_eq!(score.spines[0].events[2].subtoken_index, 2);
        assert_eq!(score.spines[0].events[3].kind, HumdrumKernEventKindV1::Rest);
        assert_eq!(score.spines[0].events[3].score_offset, RationalBeat::new(1, 1).unwrap());
    }

    #[test]
    fn unequal_multiple_stop_durations_are_rejected() {
        let source = "**kern\n4c 8e\n*-\n";
        assert!(matches!(
            parse_humdrum_kern(source.as_bytes(), "clock"),
            Err(HumdrumKernError::UnequalMultipleStopDurations { .. })
        ));
    }

    #[test]
    fn preserves_ties_slurs_beams_and_other_signifiers_as_raw_suffixes() {
        let score = parse("**kern\n4c[\n4c_\n4c]\n4d(\n4d)\n*-\n");
        assert_eq!(score.spines[0].events[0].signifiers, "[");
        assert_eq!(score.spines[0].events[1].signifiers, "_");
        assert_eq!(score.spines[0].events[2].signifiers, "]");
        assert_eq!(score.spines[0].events[3].signifiers, "(");
        assert_eq!(score.spines[0].events[4].signifiers, ")");
    }

    #[test]
    fn preserves_tandem_interpretations_and_measure_labels() {
        let score = parse("**kern\n*clefG2\n*M4/4\n*k[f#c#]\n*MM120\n*Iflute\n=12a\n4c\n=13\n*-\n");
        assert_eq!(score.spines[0].interpretations.len(), 5);
        assert_eq!(score.spines[0].interpretations[0].kind, HumdrumInterpretationKindV1::Clef);
        assert_eq!(score.spines[0].interpretations[1].kind, HumdrumInterpretationKindV1::Meter);
        assert_eq!(score.spines[0].interpretations[2].kind, HumdrumInterpretationKindV1::KeySignature);
        assert_eq!(score.spines[0].interpretations[3].kind, HumdrumInterpretationKindV1::Tempo);
        assert_eq!(score.spines[0].interpretations[4].kind, HumdrumInterpretationKindV1::Instrument);
        assert_eq!(score.spines[0].barlines[0].measure_label.as_deref(), Some("12a"));
    }

    #[test]
    fn multiple_spines_keep_independent_cursors_and_maps() {
        let source = "**kern\t**kern\n=1\t=1\n4c\t2C\n4d\t2D\n=2\t=2\n*-\t*-\n";
        let score = parse(source);
        assert_eq!(score.spines.len(), 2);
        assert_eq!(score.spines[0].final_cursor, RationalBeat::new(2, 1).unwrap());
        assert_eq!(score.spines[1].final_cursor, RationalBeat::new(4, 1).unwrap());
        assert_eq!(score.spines[0].measure_map.measures[0].end, RationalBeat::new(2, 1).unwrap());
        assert_eq!(score.spines[1].measure_map.measures[0].end, RationalBeat::new(4, 1).unwrap());
    }

    #[test]
    fn null_tokens_do_not_advance_and_global_local_comments_are_preserved() {
        let source = "!!!COM: Test\n**kern\t**kern\n!voice one\t!voice two\n=1\t=1\n4c\t.\n.\t4C\n=2\t=2\n*-\t*-\n";
        let score = parse(source);
        assert_eq!(score.spines[0].final_cursor, RationalBeat::new(1, 1).unwrap());
        assert_eq!(score.spines[1].final_cursor, RationalBeat::new(2, 1).unwrap());
        assert!(score.source_records.iter().any(|record| record.kind == HumdrumRecordKindV1::GlobalComment));
        assert!(score.source_records.iter().any(|record| record.kind == HumdrumRecordKindV1::LocalComment));
    }

    #[test]
    fn missing_explicit_barlines_leave_measure_map_unknown() {
        let score = parse("**kern\n4c\n4d\n*-\n");
        assert!(matches!(score.spines[0].measure_map.status, MeasureMapStatus::Unknown { .. }));
        assert!(score.spines[0].measure_map.measures.is_empty());
    }

    #[test]
    fn split_join_exchange_and_add_spine_operations_are_unsupported() {
        for operation in ["*^", "*v", "*x", "*+"] {
            let source = format!("**kern\n{operation}\n*-\n");
            assert!(matches!(
                parse_humdrum_kern(source.as_bytes(), "clock"),
                Err(HumdrumKernError::UnsupportedSpinePath { .. })
            ));
        }
    }

    #[test]
    fn malformed_spine_counts_non_kern_spines_and_early_termination_fail_closed() {
        assert!(matches!(
            parse_humdrum_kern(b"**kern\t**kern\n4c\n*-\n", "clock"),
            Err(HumdrumKernError::InconsistentSpineCount { .. })
        ));
        assert!(matches!(
            parse_humdrum_kern(b"**kern\t**text\n4c\tla\n*-\t*-\n", "clock"),
            Err(HumdrumKernError::NonKernSpine { .. })
        ));
        assert!(matches!(
            parse_humdrum_kern(b"**kern\t**kern\n*-\t4c\n", "clock"),
            Err(HumdrumKernError::EarlySpineTermination { .. })
        ));
    }

    #[test]
    fn source_commitment_replay_and_tampering_are_detected() {
        let source = "**kern\n=1\n4c\n4d\n=2\n*-\n";
        let score = parse(source);
        assert_eq!(score.report_sha256.len(), 64);
        assert_eq!(humdrum_kern_score_commitment(&score).unwrap(), score.report_sha256);
        assert!(validate_humdrum_kern_replay(source.as_bytes(), "kern-test-clock", &score).unwrap());
        let mut tampered = score.clone();
        tampered.spines[0].events[0].score_offset = RationalBeat::new(1, 1).unwrap();
        assert!(!validate_humdrum_kern_replay(source.as_bytes(), "kern-test-clock", &tampered).unwrap());
    }

    #[test]
    fn unsupported_or_missing_reciprocal_and_pitch_syntax_is_rejected() {
        for token in ["c", "x", "4h", "4c#-", "4cn#"] {
            let source = format!("**kern\\n{token}\\n*-\\n");
            assert!(
                parse_humdrum_kern(source.as_bytes(), "clock").is_err(),
                "token should be rejected by the conservative V1 parser: {token}"
            );
        }
    }

    #[test]
    fn incomplete_trailing_measure_does_not_get_inferred_from_meter_or_cursor() {
        let score = parse("**kern\\n*M4/4\\n=1\\n4c\\n4d\\n*-\\n");
        assert!(matches!(
            score.spines[0].measure_map.status,
            MeasureMapStatus::Unknown { .. }
        ));
        assert!(score.spines[0].measure_map.measures.is_empty());
    }

    #[test]
    fn unterminated_spines_and_records_after_termination_are_rejected() {
        assert_eq!(
            parse_humdrum_kern(b"**kern\\n4c\\n", "clock"),
            Err(HumdrumKernError::MissingSpineTermination)
        );
        assert_eq!(
            parse_humdrum_kern(b"**kern\\n*-\\n4c\\n", "clock"),
            Err(HumdrumKernError::RecordAfterTermination { line_number: 3 })
        );
    }

    #[test]
    fn strict_tab_delimited_spine_width_is_enforced_on_interpretations_and_comments() {
        for source in [
            "**kern\\t**kern\\n*M4/4\\n*-\\t*-\\n",
            "**kern\\t**kern\\n!only-one-local-comment\\n*-\\t*-\\n",
        ] {
            assert!(matches!(
                parse_humdrum_kern(source.as_bytes(), "clock"),
                Err(HumdrumKernError::InconsistentSpineCount { .. })
            ));
        }
    }

}
