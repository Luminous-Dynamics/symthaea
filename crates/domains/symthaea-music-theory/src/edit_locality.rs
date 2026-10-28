// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verifiable edit locality for symbolic monophonic scores.
//!
//! The contract keeps two questions separate:
//! 1. Did the requested property change as specified?
//! 2. Were protected properties preserved?
//!
//! This avoids a blended reward in which a successful requested edit masks
//! collateral damage. Version 1.0 deliberately rejects event-count changes and
//! does not claim instrumentation, polyphonic, or rendered-audio preservation.

use serde::{Deserialize, Serialize};

/// Version of the requested-change / protected-property contract.
pub const EDIT_LOCALITY_CONTRACT_VERSION: &str = "edit-locality-v1.0";
/// Bound O(n) inputs so evidence generation has predictable cost.
pub const MAX_EDIT_LOCALITY_EVENTS: usize = 4096;

/// A monophonic symbolic event: MIDI semitone, onset in beats and duration in beats.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalityEvent {
    pub pitch_midi: i32,
    pub onset_beats: f64,
    pub duration_beats: f64,
}

/// A single requested edit. The changed dimension and protected dimensions are
/// implied by the operation, avoiding caller-defined masks that hide violations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EditRequest {
    pub operation: EditOperation,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EditOperation {
    /// Translate every pitch by a constant number of semitones.
    Transpose { semitones: i32 },
    /// Multiply every note duration; preserve all pitches and onsets.
    ScaleDurations { factor: f64 },
    /// Scale onset offsets relative to the first source event and move the
    /// phrase start to target_start_beats; preserve pitches and durations.
    ScaleOnsets {
        factor: f64,
        target_start_beats: f64,
    },
    /// Change one pitch only; all other pitches and every onset/duration are protected.
    PitchAt { index: usize, semitones: i32 },
    /// Change one duration only; all other durations and every pitch/onset are protected.
    DurationAt { index: usize, factor: f64 },
}

/// Explicit numerical tolerances. Defaults are initial engineering values,
/// not calibrated against listener judgements or real score-edit data.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EditLocalityConfig {
    pub pitch_tolerance_semitones: f64,
    pub onset_tolerance_beats: f64,
    /// Absolute log ratio for durations, |ln(actual / expected)|.
    pub duration_log_ratio_tolerance: f64,
}

impl Default for EditLocalityConfig {
    fn default() -> Self {
        Self {
            pitch_tolerance_semitones: 0.25,
            onset_tolerance_beats: 0.05,
            duration_log_ratio_tolerance: 0.05,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditLocalityError {
    InvalidInput(&'static str),
    UnsupportedEdit(&'static str),
}

/// Independent, inspectable property evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyCheck {
    pub property: String,
    pub max_error: Option<f64>,
    pub mean_error: Option<f64>,
    pub tolerance: f64,
    pub checked_values: usize,
    pub passed: bool,
}

/// A replayable receipt: request, thresholds, requested-axis evidence and all
/// protected-property evidence are recorded independently.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditLocalityEvidence {
    pub contract_version: String,
    pub request: EditRequest,
    pub config: EditLocalityConfig,
    pub source_event_count: usize,
    pub target_event_count: usize,
    pub event_count_preserved: bool,
    pub requested_property: PropertyCheck,
    pub protected_properties: Vec<PropertyCheck>,
    pub requested_property_passed: bool,
    pub protected_properties_passed: bool,
    pub passed: bool,
    pub failure_reasons: Vec<String>,
}

/// Verify a request with the default configuration.
pub fn verify_edit_locality(
    source: &[LocalityEvent],
    target: &[LocalityEvent],
    request: &EditRequest,
) -> Result<EditLocalityEvidence, EditLocalityError> {
    verify_edit_locality_with_config(
        source,
        target,
        request,
        &EditLocalityConfig::default(),
    )
}

/// Verify the requested edit and all implicitly protected properties.
/// Event alignment is index-preserving by design; this contract does not
/// certify insertions, deletions, reordering, polyphony or instrumentation edits.
pub fn verify_edit_locality_with_config(
    source: &[LocalityEvent],
    target: &[LocalityEvent],
    request: &EditRequest,
    config: &EditLocalityConfig,
) -> Result<EditLocalityEvidence, EditLocalityError> {
    validate_config(config)?;
    validate_events(source, "source")?;
    validate_events(target, "target")?;
    if source.is_empty() || target.is_empty() {
        return Err(EditLocalityError::InvalidInput(
            "source and target must both contain events",
        ));
    }
    if source.len() > MAX_EDIT_LOCALITY_EVENTS || target.len() > MAX_EDIT_LOCALITY_EVENTS {
        return Err(EditLocalityError::UnsupportedEdit(
            "event count exceeds the edit-locality computation budget",
        ));
    }
    if source.len() != target.len() {
        return Err(EditLocalityError::UnsupportedEdit(
            "event insertion/deletion is not supported by edit-locality-v1.0",
        ));
    }

    let n = source.len();
    let mut protected = Vec::new();
    let (requested_property, requested_passed) = match request.operation {
        EditOperation::Transpose { semitones } => {
            validate_semitones(semitones)?;
            let errors = source
                .iter()
                .zip(target)
                .map(|(s, t)| {
                    let expected = s.pitch_midi + semitones;
                    if !(0..=127).contains(&expected) {
                        return Err(EditLocalityError::UnsupportedEdit(
                            "transposition produces a pitch outside MIDI range",
                        ));
                    }
                    Ok((t.pitch_midi - expected).abs() as f64)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let check = make_check(
                "requested_pitch_transposition",
                &errors,
                config.pitch_tolerance_semitones,
            );
            protected.push(pitch_preservation(source, target, config, None));
            protected.push(onset_preservation(source, target, config));
            protected.push(duration_preservation(source, target, config, None));
            let passed = check.passed;
            (check, passed)
        }
        EditOperation::ScaleDurations { factor } => {
            validate_factor(factor)?;
            let errors = source
                .iter()
                .zip(target)
                .map(|(s, t)| duration_log_error(t.duration_beats, s.duration_beats * factor))
                .collect::<Result<Vec<_>, _>>()?;
            let check = make_check(
                "requested_duration_scaling",
                &errors,
                config.duration_log_ratio_tolerance,
            );
            protected.push(pitch_preservation(source, target, config, None));
            protected.push(onset_preservation(source, target, config));
            let passed = check.passed;
            (check, passed)
        }
        EditOperation::ScaleOnsets {
            factor,
            target_start_beats,
        } => {
            validate_factor(factor)?;
            if !target_start_beats.is_finite() || target_start_beats < 0.0 {
                return Err(EditLocalityError::InvalidInput(
                    "target_start_beats must be finite and non-negative",
                ));
            }
            let source_origin = source[0].onset_beats;
            let errors = source
                .iter()
                .zip(target)
                .map(|(s, t)| {
                    let expected =
                        target_start_beats + factor * (s.onset_beats - source_origin);
                    (t.onset_beats - expected).abs()
                })
                .collect::<Vec<_>>();
            let check = make_check(
                "requested_onset_scaling",
                &errors,
                config.onset_tolerance_beats,
            );
            protected.push(pitch_preservation(source, target, config, None));
            protected.push(duration_preservation(source, target, config, None));
            let passed = check.passed;
            (check, passed)
        }
        EditOperation::PitchAt { index, semitones } => {
            validate_semitones(semitones)?;
            if index >= n {
                return Err(EditLocalityError::InvalidInput(
                    "pitch edit index must lie within the source sequence",
                ));
            }
            let expected = source[index].pitch_midi + semitones;
            if !(0..=127).contains(&expected) {
                return Err(EditLocalityError::UnsupportedEdit(
                    "local pitch edit produces a pitch outside MIDI range",
                ));
            }
            let check = make_check(
                "requested_local_pitch_change",
                &[(target[index].pitch_midi - expected).abs() as f64],
                config.pitch_tolerance_semitones,
            );
            protected.push(pitch_preservation(source, target, config, Some(index)));
            protected.push(onset_preservation(source, target, config));
            protected.push(duration_preservation(source, target, config, None));
            let passed = check.passed;
            (check, passed)
        }
        EditOperation::DurationAt { index, factor } => {
            validate_factor(factor)?;
            if index >= n {
                return Err(EditLocalityError::InvalidInput(
                    "duration edit index must lie within the source sequence",
                ));
            }
            let check = make_check(
                "requested_local_duration_change",
                &[duration_log_error(
                    target[index].duration_beats,
                    source[index].duration_beats * factor,
                )?],
                config.duration_log_ratio_tolerance,
            );
            protected.push(pitch_preservation(source, target, config, None));
            protected.push(onset_preservation(source, target, config));
            protected.push(duration_preservation(source, target, config, Some(index)));
            let passed = check.passed;
            (check, passed)
        }
    };

    let protected_properties_passed = protected.iter().all(|check| check.passed);
    let mut failure_reasons = Vec::new();
    if !requested_passed {
        failure_reasons.push(format!(
            "requested property '{}' did not meet its target",
            requested_property.property
        ));
    }
    for check in &protected {
        if !check.passed {
            failure_reasons.push(format!(
                "protected property '{}' changed beyond tolerance",
                check.property
            ));
        }
    }
    let passed = requested_passed && protected_properties_passed;
    Ok(EditLocalityEvidence {
        contract_version: EDIT_LOCALITY_CONTRACT_VERSION.to_owned(),
        request: request.clone(),
        config: *config,
        source_event_count: source.len(),
        target_event_count: target.len(),
        event_count_preserved: true,
        requested_property,
        protected_properties: protected,
        requested_property_passed: requested_passed,
        protected_properties_passed,
        passed,
        failure_reasons,
    })
}

fn validate_config(config: &EditLocalityConfig) -> Result<(), EditLocalityError> {
    let tolerances = [
        config.pitch_tolerance_semitones,
        config.onset_tolerance_beats,
        config.duration_log_ratio_tolerance,
    ];
    if tolerances.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err(EditLocalityError::InvalidInput(
            "tolerances must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_events(events: &[LocalityEvent], label: &'static str) -> Result<(), EditLocalityError> {
    let mut previous_onset: Option<f64> = None;
    for event in events {
        if !(0..=127).contains(&event.pitch_midi) {
            return Err(EditLocalityError::InvalidInput(
                "pitch must lie within MIDI range [0, 127]",
            ));
        }
        if !event.onset_beats.is_finite()
            || event.onset_beats < 0.0
            || !event.duration_beats.is_finite()
            || event.duration_beats <= 0.0
        {
            return Err(EditLocalityError::InvalidInput(
                "onsets must be finite/non-negative and durations finite/positive",
            ));
        }
        if previous_onset.is_some_and(|p| event.onset_beats <= p) {
            return Err(match label {
                "source" => EditLocalityError::InvalidInput(
                    "source events must be strictly ordered by onset",
                ),
                _ => EditLocalityError::InvalidInput(
                    "target events must be strictly ordered by onset",
                ),
            });
        }
        previous_onset = Some(event.onset_beats);
    }
    Ok(())
}

fn validate_semitones(semitones: i32) -> Result<(), EditLocalityError> {
    if !(-24..=24).contains(&semitones) {
        return Err(EditLocalityError::UnsupportedEdit(
            "semitone shift must be within [-24, 24]",
        ));
    }
    Ok(())
}

fn validate_factor(factor: f64) -> Result<(), EditLocalityError> {
    if !factor.is_finite() || factor <= 0.0 || factor > 8.0 {
        return Err(EditLocalityError::UnsupportedEdit(
            "scale factor must be finite and in (0, 8]",
        ));
    }
    Ok(())
}

fn duration_log_error(actual: f64, expected: f64) -> Result<f64, EditLocalityError> {
    let ratio = actual / expected;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(EditLocalityError::InvalidInput(
            "duration residual is outside the finite numeric contract",
        ));
    }
    let error = ratio.ln().abs();
    if !error.is_finite() {
        return Err(EditLocalityError::InvalidInput(
            "duration residual is not finite",
        ));
    }
    Ok(error)
}

fn make_check(property: &str, errors: &[f64], tolerance: f64) -> PropertyCheck {
    if errors.is_empty() || errors.iter().any(|x| !x.is_finite()) {
        return PropertyCheck {
            property: property.to_owned(),
            max_error: None,
            mean_error: None,
            tolerance,
            checked_values: errors.len(),
            passed: false,
        };
    }
    let max_error = errors.iter().copied().fold(0.0_f64, f64::max);
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
    PropertyCheck {
        property: property.to_owned(),
        max_error: Some(max_error),
        mean_error: Some(mean_error),
        tolerance,
        checked_values: errors.len(),
        passed: max_error <= tolerance,
    }
}

fn pitch_preservation(
    source: &[LocalityEvent],
    target: &[LocalityEvent],
    config: &EditLocalityConfig,
    excluded_index: Option<usize>,
) -> PropertyCheck {
    let errors: Vec<_> = source
        .iter()
        .zip(target)
        .enumerate()
        .filter(|(i, _)| Some(*i) != excluded_index)
        .map(|(_, (s, t))| (t.pitch_midi - s.pitch_midi).abs() as f64)
        .collect();
    make_check(
        if excluded_index.is_some() { "non_target_pitch_preservation" } else { "pitch_preservation" },
        &errors,
        config.pitch_tolerance_semitones,
    )
}

fn onset_preservation(
    source: &[LocalityEvent],
    target: &[LocalityEvent],
    config: &EditLocalityConfig,
) -> PropertyCheck {
    let errors: Vec<_> = source
        .iter()
        .zip(target)
        .map(|(s, t)| (t.onset_beats - s.onset_beats).abs())
        .collect();
    make_check("onset_preservation", &errors, config.onset_tolerance_beats)
}

fn duration_preservation(
    source: &[LocalityEvent],
    target: &[LocalityEvent],
    config: &EditLocalityConfig,
    excluded_index: Option<usize>,
) -> PropertyCheck {
    let errors: Vec<_> = source
        .iter()
        .zip(target)
        .enumerate()
        .filter(|(i, _)| Some(*i) != excluded_index)
        .map(|(_, (s, t))| duration_log_error(t.duration_beats, s.duration_beats).unwrap_or(f64::INFINITY))
        .collect();
    make_check(
        if excluded_index.is_some() { "non_target_duration_preservation" } else { "duration_preservation" },
        &errors,
        config.duration_log_ratio_tolerance,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const CASES: &str = include_str!("../mel-edit-locality-smoke-cases-v1.jsonl");

    #[derive(Debug, Deserialize)]
    struct Fixture {
        case_id: String,
        source_events: Vec<LocalityEvent>,
        target_events: Vec<LocalityEvent>,
        edit_request: EditRequest,
        expected_requested_property_passed: bool,
        expected_protected_properties_passed: bool,
        expected_verdict: String,
    }

    fn simple_source() -> Vec<LocalityEvent> {
        vec![
            LocalityEvent { pitch_midi: 60, onset_beats: 0.0, duration_beats: 0.5 },
            LocalityEvent { pitch_midi: 62, onset_beats: 1.0, duration_beats: 0.75 },
            LocalityEvent { pitch_midi: 65, onset_beats: 2.0, duration_beats: 0.5 },
            LocalityEvent { pitch_midi: 64, onset_beats: 3.0, duration_beats: 1.0 },
        ]
    }

    #[test]
    fn rejects_collateral_changes_even_when_requested_edit_succeeds() {
        let source = simple_source();
        let target = source.iter().enumerate().map(|(i, e)| LocalityEvent {
            pitch_midi: e.pitch_midi + 5,
            onset_beats: e.onset_beats + if i == 2 { 0.5 } else { 0.0 },
            duration_beats: e.duration_beats,
        }).collect::<Vec<_>>();
        let request = EditRequest { operation: EditOperation::Transpose { semitones: 5 } };
        let evidence = verify_edit_locality(&source, &target, &request).unwrap();
        assert!(evidence.requested_property_passed);
        assert!(!evidence.protected_properties_passed);
        assert!(!evidence.passed);
    }

    #[test]
    fn no_op_does_not_satisfy_requested_edit() {
        let source = simple_source();
        let request = EditRequest { operation: EditOperation::Transpose { semitones: 5 } };
        let evidence = verify_edit_locality(&source, &source, &request).unwrap();
        assert!(!evidence.requested_property_passed);
        assert!(evidence.protected_properties_passed);
        assert!(!evidence.passed);
    }

    #[test]
    fn event_count_changes_are_explicitly_unsupported() {
        let source = simple_source();
        let mut target = source.clone();
        target.push(LocalityEvent { pitch_midi: 67, onset_beats: 4.0, duration_beats: 0.5 });
        let request = EditRequest { operation: EditOperation::Transpose { semitones: 5 } };
        assert!(matches!(
            verify_edit_locality(&source, &target, &request),
            Err(EditLocalityError::UnsupportedEdit(_))
        ));
    }

    #[test]
    fn corpus_replays_and_receipts_round_trip() {
        let mut count = 0;
        let mut accepted = 0;
        let mut rejected = 0;
        let mut seen = std::collections::HashSet::new();
        for line in CASES.lines().filter(|line| !line.trim().is_empty()) {
            let value: Value = serde_json::from_str(line).expect("valid fixture JSON");
            let fixture: Fixture = serde_json::from_value(value.clone()).expect("fixture schema matches structs");
            assert!(seen.insert(fixture.case_id.clone()), "duplicate {}", fixture.case_id);
            let result = verify_edit_locality(
                &fixture.source_events,
                &fixture.target_events,
                &fixture.edit_request,
            );
            match result {
                Ok(evidence) => {
                    let receipt = serde_json::to_string(&evidence).expect("serialize receipt");
                    let replay: EditLocalityEvidence = serde_json::from_str(&receipt).expect("receipt round-trip");
                    assert_eq!(replay, evidence);
                    assert_eq!(evidence.requested_property_passed, fixture.expected_requested_property_passed,
                        "{} requested-property mismatch: {evidence:?}", fixture.case_id);
                    assert_eq!(evidence.protected_properties_passed, fixture.expected_protected_properties_passed,
                        "{} protected-property mismatch: {evidence:?}", fixture.case_id);
                    assert_eq!(evidence.passed, fixture.expected_verdict == "accept",
                        "{} verdict mismatch: {evidence:?}", fixture.case_id);
                    if evidence.passed { accepted += 1; } else { rejected += 1; }
                }
                Err(error) => {
                    assert_eq!(fixture.expected_verdict, "reject",
                        "{} unexpectedly unsupported: {error:?}", fixture.case_id);
                    rejected += 1;
                }
            }
            count += 1;
        }
        assert_eq!(count, 132);
        assert_eq!(accepted, 60);
        assert_eq!(rejected, 72);
    }
}
