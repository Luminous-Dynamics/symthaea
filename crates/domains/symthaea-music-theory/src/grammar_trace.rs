// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composer-owned structural assertions.
//!
//! These records deliberately reference score events by `(voice, index)`.
//! The public protocol assigns stable string IDs later, without making the
//! theory crate depend on a transport/UI crate. Analyst must still verify
//! every assertion against the exported symbolic score.

use serde::{Deserialize, Serialize};

use crate::form::Form;
use crate::grammar::{GrammarFamily, GrammarPlanEvidence};
use crate::motif::Motif;
use crate::score::{Emphasis, Score, ScoreNote, VoiceRole};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreEventRef {
    pub role: VoiceRole,
    pub role_index: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralAssertion {
    pub id: String,
    pub kind: String,
    pub parent_id: Option<String>,
    pub start: ScoreEventRef,
    pub end: ScoreEventRef,
    pub decision_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotifAssertion {
    pub occurrence_id: String,
    pub events: Vec<ScoreEventRef>,
    pub formal_region_id: String,
    pub transformation: String,
    pub preserved_invariants: Vec<String>,
    pub changed_dimensions: Vec<String>,
    pub literal_distance: f32,
    pub structural_distance: f32,
    pub role_binding: Option<String>,
    pub decision_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadenceAssertion {
    pub cadence_id: String,
    pub proposed_type: String,
    pub preparation: Vec<ScoreEventRef>,
    pub arrival: ScoreEventRef,
    pub fulfils_obligation_id: Option<String>,
    pub decision_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssertedObligationState {
    Created,
    Reinforced,
    Deferred,
    Transformed,
    Fulfilled,
    Abandoned,
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObligationAssertion {
    pub obligation_id: String,
    pub from: Option<AssertedObligationState>,
    pub to: AssertedObligationState,
    pub evidence: Vec<ScoreEventRef>,
    pub responsible_pass: String,
    pub transformation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GrammarStructuralTrace {
    pub structures: Vec<StructuralAssertion>,
    pub motif_occurrences: Vec<MotifAssertion>,
    pub cadences: Vec<CadenceAssertion>,
    pub obligation_transitions: Vec<ObligationAssertion>,
}

fn voice_with_refs(score: &Score, role: VoiceRole) -> Vec<(ScoreEventRef, ScoreNote)> {
    score
        .voice(role)
        .into_iter()
        .enumerate()
        .map(|(role_index, note)| (ScoreEventRef { role, role_index }, note))
        .collect()
}

fn region_endpoints(score: &Score, start: f64, end: f64) -> Option<(ScoreEventRef, ScoreEventRef)> {
    let melody: Vec<_> = voice_with_refs(score, VoiceRole::Melody)
        .into_iter()
        .filter(|(_, note)| note.onset.beats() >= start - 1e-7 && note.onset.beats() < end + 1e-7)
        .collect();
    if let (Some(first), Some(last)) = (melody.first(), melody.last()) {
        return Some((first.0, last.0));
    }
    let mut events: Vec<_> = [
        VoiceRole::Melody,
        VoiceRole::CounterMelody,
        VoiceRole::Harmony,
        VoiceRole::Bass,
    ]
    .into_iter()
    .flat_map(|role| voice_with_refs(score, role))
    .filter(|(_, note)| note.onset.beats() >= start - 1e-7 && note.onset.beats() < end + 1e-7)
    .collect();
    events.sort_by(|left, right| left.1.onset.beats().total_cmp(&right.1.onset.beats()));
    Some((events.first()?.0, events.last()?.0))
}

fn structure_for_region(
    score: &Score,
    id: String,
    kind: String,
    start: f64,
    end: f64,
    decision_id: &str,
) -> Option<StructuralAssertion> {
    let (start, end) = region_endpoints(score, start, end)?;
    Some(StructuralAssertion {
        id,
        kind,
        parent_id: None,
        start,
        end,
        decision_id: decision_id.into(),
    })
}

fn ratio_close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-6
}

fn motif_assertions(
    score: &Score,
    motif: &Motif,
    regions: &[StructuralAssertion],
) -> Vec<MotifAssertion> {
    let expected: Vec<_> = motif
        .notes
        .iter()
        .filter(|note| note.degree.is_some())
        .collect();
    let melody = voice_with_refs(score, VoiceRole::Melody);
    if expected.len() < 2 || melody.len() < expected.len() {
        return Vec::new();
    }
    let expected_intervals: Vec<_> = expected
        .windows(2)
        .map(|pair| pair[1].degree.unwrap() - pair[0].degree.unwrap())
        .collect();
    let expected_durations: Vec<_> = expected.iter().map(|note| note.duration.beats()).collect();
    let mut result = Vec::new();
    let mut last_end = 0usize;
    for start in 0..=melody.len() - expected.len() {
        if start < last_end {
            continue;
        }
        let window = &melody[start..start + expected.len()];
        let intervals: Vec<_> = window
            .windows(2)
            .map(|pair| pair[0].1.pitch.semitones_to(pair[1].1.pitch))
            .collect();
        // Diatonic degree intervals and semitone intervals are not identical,
        // so compare direction and repeated-tone identity here; Analyst later
        // checks the realized event sequence independently.
        let contour_similarity = intervals
            .iter()
            .zip(&expected_intervals)
            .filter(|(actual, expected)| {
                actual.signum() == expected.signum() && (**actual == 0) == (**expected == 0)
            })
            .count() as f64
            / expected_intervals.len().max(1) as f64;
        let actual_durations: Vec<_> = window
            .iter()
            .map(|(_, note)| note.duration.beats())
            .collect();
        let scale = actual_durations[0] / expected_durations[0].max(1e-9);
        let expected_total = expected_durations.iter().sum::<f64>().max(1e-9);
        let actual_total = actual_durations.iter().sum::<f64>().max(1e-9);
        let rhythm_similarity = 1.0
            - actual_durations
                .iter()
                .zip(&expected_durations)
                .map(|(actual, expected)| (actual / actual_total - expected / expected_total).abs())
                .sum::<f64>()
                / 2.0;
        let structural_similarity = 0.65 * contour_similarity + 0.35 * rhythm_similarity;
        // This is an intentionally permissive composer assertion. Analyst's
        // independent symbolic verifier remains responsible for accepting or
        // rejecting it against the exported score and motif definition.
        if structural_similarity < 0.68 {
            continue;
        }
        let transformation = if contour_similarity > 0.999
            && rhythm_similarity > 0.999
            && ratio_close(scale, 1.0)
        {
            "transposed"
        } else if contour_similarity > 0.999 && rhythm_similarity > 0.999 && ratio_close(scale, 2.0)
        {
            "augmented"
        } else if contour_similarity > 0.999 && rhythm_similarity > 0.999 && ratio_close(scale, 0.5)
        {
            "diminished"
        } else {
            "developed"
        };
        let onset = window[0].1.onset.beats();
        let formal_region_id = regions
            .iter()
            .find(|region| {
                let start = score.voice(region.start.role)[region.start.role_index]
                    .onset
                    .beats();
                let end_note = score.voice(region.end.role)[region.end.role_index];
                let end = (end_note.onset + end_note.duration).beats();
                onset >= start - 1e-6 && onset <= end + 1e-6
            })
            .map_or_else(|| "piece".into(), |region| region.id.clone());
        result.push(MotifAssertion {
            occurrence_id: format!("motif-primary-composer-occurrence-{}", result.len()),
            events: window.iter().map(|(reference, _)| *reference).collect(),
            formal_region_id,
            transformation: transformation.into(),
            preserved_invariants: [
                (contour_similarity >= 0.75).then_some("contour-direction".into()),
                (rhythm_similarity >= 0.75).then_some("duration-ratios".into()),
            ]
            .into_iter()
            .flatten()
            .collect(),
            changed_dimensions: (transformation == "developed")
                .then(|| vec!["interval-or-rhythm-detail".into()])
                .unwrap_or_default(),
            literal_distance: (1.0 - structural_similarity) as f32,
            structural_distance: (1.0 - structural_similarity) as f32,
            role_binding: Some("primary-material".into()),
            decision_id: "grammar-owner-motif-realization".into(),
        });
        last_end = start + expected.len();
    }
    result
}

fn cadence_assertions(score: &Score, family: GrammarFamily) -> Vec<CadenceAssertion> {
    let mut result = Vec::new();
    for (arrival, note) in voice_with_refs(score, VoiceRole::Melody) {
        if note.emphasis != Emphasis::Cadential {
            continue;
        }
        let prior = arrival
            .role_index
            .checked_sub(1)
            .map(|role_index| ScoreEventRef {
                role: VoiceRole::Melody,
                role_index,
            });
        result.push(CadenceAssertion {
            cadence_id: format!("composer-cadence-{}", result.len()),
            proposed_type: if family == GrammarFamily::PeriodSentence {
                "phrase-close"
            } else {
                "grammar-specific-close"
            }
            .into(),
            preparation: prior.into_iter().collect(),
            arrival,
            fulfils_obligation_id: None,
            decision_id: "grammar-owner-cadence".into(),
        });
    }
    result
}

/// Build the assertion trace at the grammar boundary, while the plan and
/// source motif are still available. This is an assertion, never verification.
pub fn build_grammar_trace(
    family: GrammarFamily,
    score: &Score,
    form: Option<&Form>,
    plan: &GrammarPlanEvidence,
    motif: &Motif,
) -> GrammarStructuralTrace {
    let mut structures = Vec::new();
    match plan {
        GrammarPlanEvidence::PeriodSentence(_) => {
            if let Some(form) = form {
                let mut start = 0.0;
                for (index, section) in form.sections.iter().enumerate() {
                    let end = start + section.period.total_duration().beats();
                    if let Some(assertion) = structure_for_region(
                        score,
                        format!("period-section-{index}"),
                        format!("period-{:?}", section.role).to_ascii_lowercase(),
                        start,
                        end,
                        "period-sentence-engine",
                    ) {
                        structures.push(assertion);
                    }
                    start = end;
                }
            }
        }
        GrammarPlanEvidence::Contrapuntal(_) => {
            let bar = score.meter as f64;
            for (index, (kind, start, end)) in [
                ("fugal-exposition", 0.0, 3.0 * bar),
                ("fugal-episodes-and-inversion", 3.0 * bar, 7.0 * bar),
                ("fugal-stretto", 7.0 * bar, 9.0 * bar),
                (
                    "fugal-augmented-final",
                    9.0 * bar,
                    score.total_beats.beats(),
                ),
            ]
            .into_iter()
            .enumerate()
            {
                if let Some(assertion) = structure_for_region(
                    score,
                    format!("contrapuntal-section-{index}"),
                    kind.into(),
                    start,
                    end,
                    "compact-fugue-engine",
                ) {
                    structures.push(assertion);
                }
            }
        }
        GrammarPlanEvidence::GrooveCycle(plan) => {
            for (index, section) in plan.sections.iter().enumerate() {
                let start = section.start_cycle as f64 * plan.cycle_beats;
                let end = (section.start_cycle + section.cycles) as f64 * plan.cycle_beats;
                if let Some(assertion) = structure_for_region(
                    score,
                    format!("groove-section-{index}"),
                    format!("groove-{:?}", section.role).to_ascii_lowercase(),
                    start,
                    end,
                    "groove-cycle-section-plan",
                ) {
                    structures.push(assertion);
                }
            }
        }
        GrammarPlanEvidence::AdditiveProcess(plan) => {
            for (index, start) in plan.cycle_starts.iter().enumerate() {
                let end = plan
                    .cycle_starts
                    .get(index + 1)
                    .map_or(score.total_beats.beats(), |value| value.beats());
                if let Some(assertion) = structure_for_region(
                    score,
                    format!("process-stage-{index}"),
                    format!("additive-prefix-{}", plan.prefix_lengths[index]),
                    start.beats(),
                    end,
                    "additive-process-cycle",
                ) {
                    structures.push(assertion);
                }
            }
        }
        GrammarPlanEvidence::ModalArc(plan) => {
            for (index, (kind, start, end)) in [
                ("modal-exposition", 0.0, plan.alap_end.beats()),
                ("modal-pulse", plan.alap_end.beats(), plan.jor_end.beats()),
                (
                    "modal-intensification",
                    plan.jor_end.beats(),
                    plan.jhala_end.beats(),
                ),
            ]
            .into_iter()
            .enumerate()
            {
                if let Some(assertion) = structure_for_region(
                    score,
                    format!("modal-stage-{index}"),
                    kind.into(),
                    start,
                    end,
                    "modal-arc-stage-plan",
                ) {
                    structures.push(assertion);
                }
            }
        }
        GrammarPlanEvidence::Compatibility { .. } => {
            if let Some(form) = form {
                let mut start = 0.0;
                for (index, section) in form.sections.iter().enumerate() {
                    let end = start + section.period.total_duration().beats();
                    if let Some(assertion) = structure_for_region(
                        score,
                        format!("period-section-{index}"),
                        format!("period-{:?}", section.role).to_ascii_lowercase(),
                        start,
                        end,
                        "period-form-section",
                    ) {
                        structures.push(assertion);
                    }
                    start = end;
                }
            }
        }
    }

    let mut obligations = Vec::new();
    match plan {
        GrammarPlanEvidence::PeriodSentence(_) => {
            obligations.push(ObligationAssertion {
                obligation_id: "thematic-return-and-closure".into(),
                from: None,
                to: AssertedObligationState::Created,
                evidence: Vec::new(),
                responsible_pass: "period-sentence-planner".into(),
                transformation: None,
            });
            obligations.push(ObligationAssertion {
                obligation_id: "thematic-return-and-closure".into(),
                from: Some(AssertedObligationState::Created),
                to: AssertedObligationState::Fulfilled,
                evidence: structures
                    .iter()
                    .flat_map(|region| [region.start, region.end])
                    .collect(),
                responsible_pass: "period-sentence-engine".into(),
                transformation: Some("return-and-cadential-liquidation".into()),
            });
        }
        GrammarPlanEvidence::Contrapuntal(_) => {
            obligations.push(ObligationAssertion {
                obligation_id: "subject-transformation-and-final-combination".into(),
                from: None,
                to: AssertedObligationState::Created,
                evidence: Vec::new(),
                responsible_pass: "compact-fugue-planner".into(),
                transformation: None,
            });
            obligations.push(ObligationAssertion {
                obligation_id: "subject-transformation-and-final-combination".into(),
                from: Some(AssertedObligationState::Created),
                to: AssertedObligationState::Fulfilled,
                evidence: structures
                    .iter()
                    .flat_map(|region| [region.start, region.end])
                    .collect(),
                responsible_pass: "compact-fugue-engine".into(),
                transformation: Some("inversion-stretto-augmentation".into()),
            });
        }
        GrammarPlanEvidence::GrooveCycle(plan) => {
            for item in &plan.obligations {
                obligations.push(ObligationAssertion {
                    obligation_id: item.code.clone(),
                    from: None,
                    to: AssertedObligationState::Created,
                    evidence: Vec::new(),
                    responsible_pass: "groove-cycle-planner".into(),
                    transformation: None,
                });
                obligations.push(ObligationAssertion {
                    obligation_id: item.code.clone(),
                    from: Some(AssertedObligationState::Created),
                    to: if item.fulfilled {
                        AssertedObligationState::Fulfilled
                    } else {
                        AssertedObligationState::Unresolved
                    },
                    evidence: structures
                        .iter()
                        .flat_map(|region| [region.start, region.end])
                        .collect(),
                    responsible_pass: "groove-cycle-realizer".into(),
                    transformation: None,
                });
            }
        }
        GrammarPlanEvidence::AdditiveProcess(_) => {
            obligations.push(ObligationAssertion {
                obligation_id: "audible-grow-shrink-rule".into(),
                from: None,
                to: AssertedObligationState::Created,
                evidence: Vec::new(),
                responsible_pass: "additive-process-planner".into(),
                transformation: None,
            });
            obligations.push(ObligationAssertion {
                obligation_id: "audible-grow-shrink-rule".into(),
                from: Some(AssertedObligationState::Created),
                to: AssertedObligationState::Fulfilled,
                evidence: structures
                    .iter()
                    .flat_map(|region| [region.start, region.end])
                    .collect(),
                responsible_pass: "additive-process-realizer".into(),
                transformation: Some("prefix-trajectory-realized".into()),
            });
        }
        GrammarPlanEvidence::ModalArc(_) => {
            obligations.push(ObligationAssertion {
                obligation_id: "ordered-modal-stages".into(),
                from: None,
                to: AssertedObligationState::Created,
                evidence: Vec::new(),
                responsible_pass: "modal-arc-planner".into(),
                transformation: None,
            });
            obligations.push(ObligationAssertion {
                obligation_id: "ordered-modal-stages".into(),
                from: Some(AssertedObligationState::Created),
                to: AssertedObligationState::Fulfilled,
                evidence: structures
                    .iter()
                    .flat_map(|region| [region.start, region.end])
                    .collect(),
                responsible_pass: "modal-arc-realizer".into(),
                transformation: Some("exposition-pulse-intensification".into()),
            });
        }
        GrammarPlanEvidence::Compatibility { .. } => {}
    }

    GrammarStructuralTrace {
        motif_occurrences: motif_assertions(score, motif, &structures),
        cadences: cadence_assertions(score, family),
        structures,
        obligation_transitions: obligations,
    }
}

#[cfg(test)]
mod tests {
    use crate::{MusicalIntent, Style, compose_with_grammar_plan};

    #[test]
    fn all_flagship_grammar_owners_emit_direct_structural_traces() {
        for style in [
            Style::Classical,
            Style::AfroCuban,
            Style::Minimalism,
            Style::HindustaniInspired,
        ] {
            let intent = MusicalIntent {
                bars: 8,
                seed: 11,
                ..MusicalIntent::default()
            };
            let realized =
                compose_with_grammar_plan(style.grammar_profile(), &intent, &style.spec());
            assert!(!realized.trace.structures.is_empty(), "{style:?}");
            assert!(!realized.trace.motif_occurrences.is_empty(), "{style:?}");
            if style == Style::Classical {
                assert!(!realized.trace.cadences.is_empty());
            } else if style != Style::AfroCuban {
                assert!(realized.trace.cadences.is_empty());
            }
        }
    }
}
