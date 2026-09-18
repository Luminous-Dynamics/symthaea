// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003C5A: real-music specificity controls for the frozen rendered-attack
//! gate. Controls are matched accompaniment attacks occurring well before the
//! first introduced attack, so detections expose nonlocal renderer/mastering
//! leakage rather than being silently treated as successful localization.

use symthaea_muse::evidence_digest::{
    canonical_json_sha256,
    rendered_attack_evidence::measure_rendered_attack,
    rendered_attack_gate::{RenderedAttackGateConfigV1, evaluate_rendered_attack_gate},
    rendered_attack_panel::{RenderedAttackPanelSampleV1, summarize_rendered_attack_panel},
    rendered_attack_provenance::{
        RENDERED_ATTACK_PROVENANCE_VERSION, RenderedAttackPanelProvenanceV1,
        RenderedAttackSubjectProvenanceV1, bind_rendered_attack_panel,
    },
    sha256_hex,
};
use symthaea_muse::theory_realize::{PerformedVoice, perform_with_spec, realize_with_spec};
use symthaea_muse::{AudioData, MusicalState};
use symthaea_music_theory::{
    Duration, MusicalIntent, Score, SonataSectionKind, Style, VoiceRole, compose_sonata_with_plan,
};

const SAMPLE_RATE: u32 = 44_100;
const SEEDS: [u64; 8] = [3, 11, 23, 41, 59, 79, 97, 127];
const START_MATCH_EPSILON_SECS: f32 = 0.000_1;
const FREQUENCY_MATCH_EPSILON_HZ: f32 = 0.001;
const PRE_WINDOW_SECS: f32 = 0.005;
const POST_WINDOW_SECS: f32 = 0.050;
const PRE_INTERVENTION_GUARD_SECS: f32 = 1.0;
const MAX_CONTROLS_PER_SEED: usize = 8;
const RENDERER_ID: &str = "symthaea-muse::theory_realize::realize_with_spec/native";

fn should_rearticulate(
    note: &symthaea_music_theory::ScoreNote,
    start: Duration,
    end: Duration,
) -> bool {
    let in_target = note.onset.beats() >= start.beats() && note.onset.beats() < end.beats();
    let accompaniment = matches!(note.role, VoiceRole::Harmony | VoiceRole::Bass);
    if !in_target || !accompaniment || note.duration.beats() < 1.0 {
        return false;
    }
    let split_onset = note.onset + note.duration.scale(1, 2);
    split_onset.beats() < end.beats()
}

fn rearticulate_accompaniment(score: &Score, start: Duration, end: Duration) -> Score {
    let mut candidate = Score::new(score.key, score.tempo_bpm, score.meter);
    for note in &score.notes {
        if should_rearticulate(note, start, end) {
            let first_duration = note.duration.scale(1, 2);
            let second_duration = note.duration.saturating_sub(first_duration);
            let mut first = *note;
            first.duration = first_duration;
            let mut second = *note;
            second.onset = note.onset + first_duration;
            second.duration = second_duration;
            candidate.push(first);
            candidate.push(second);
        } else {
            candidate.push(*note);
        }
    }
    candidate
}

fn stereo_frames(audio: &AudioData) -> &[[f32; 2]] {
    match audio {
        AudioData::StereoF32(frames) => frames,
        other => panic!("native Sonata renderer must emit StereoF32, got {other:?}"),
    }
}

fn stereo_sha256(frames: &[[f32; 2]]) -> String {
    let mut bytes = Vec::with_capacity(frames.len() * 2 * std::mem::size_of::<f32>());
    for frame in frames {
        for sample in frame {
            assert!(sample.is_finite(), "audio provenance cannot hash NaN/Inf");
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
    }
    sha256_hex(&bytes)
}

#[derive(Debug, Clone)]
struct PerformedAttack {
    voice: String,
    frequency_hz: f32,
    start_time: f32,
}

fn accompaniment_attacks(voices: &[PerformedVoice]) -> Vec<PerformedAttack> {
    voices
        .iter()
        .filter(|voice| matches!(voice.name.as_str(), "Harmony" | "Bass"))
        .flat_map(|voice| {
            voice.notes.iter().map(|note| PerformedAttack {
                voice: voice.name.clone(),
                frequency_hz: note.frequency,
                start_time: note.start_time,
            })
        })
        .collect()
}

fn partition_candidate_attacks(
    baseline: &[PerformedAttack],
    candidate: &[PerformedAttack],
) -> (Vec<PerformedAttack>, Vec<PerformedAttack>) {
    let mut used = vec![false; baseline.len()];
    let mut matched = Vec::new();
    let mut introduced = Vec::new();

    for attack in candidate {
        let matching = baseline.iter().enumerate().position(|(index, baseline)| {
            !used[index]
                && baseline.voice == attack.voice
                && (baseline.frequency_hz - attack.frequency_hz).abs()
                    <= FREQUENCY_MATCH_EPSILON_HZ
                && (baseline.start_time - attack.start_time).abs() <= START_MATCH_EPSILON_SECS
        });
        if let Some(index) = matching {
            used[index] = true;
            matched.push(attack.clone());
        } else {
            introduced.push(attack.clone());
        }
    }

    let stable_sort = |attacks: &mut Vec<PerformedAttack>| {
        attacks.sort_by(|left, right| {
            left.start_time
                .total_cmp(&right.start_time)
                .then_with(|| left.voice.cmp(&right.voice))
                .then_with(|| left.frequency_hz.total_cmp(&right.frequency_hz))
        });
    };
    stable_sort(&mut matched);
    stable_sort(&mut introduced);
    (matched, introduced)
}

/// Mark the matched first attack of every split pair. Re-articulation keeps
/// that first attack at the original performed onset and introduces a second
/// same-pitch attack later. A long split note can put the first attack more
/// than the one-second time guard before the midpoint, so time alone is not a
/// sufficient control-purity condition.
fn split_origin_mask(matched: &[PerformedAttack], introduced: &[PerformedAttack]) -> Vec<bool> {
    let mut origin = vec![false; matched.len()];
    for added in introduced {
        let nearest = matched
            .iter()
            .enumerate()
            .filter(|(_, attack)| {
                attack.voice == added.voice
                    && (attack.frequency_hz - added.frequency_hz).abs()
                        <= FREQUENCY_MATCH_EPSILON_HZ
                    && attack.start_time < added.start_time
            })
            .min_by(|(_, left), (_, right)| {
                (added.start_time - left.start_time)
                    .total_cmp(&(added.start_time - right.start_time))
            })
            .map(|(index, _)| index)
            .expect("every introduced split attack must have a preceding matched origin");
        origin[nearest] = true;
    }
    origin
}

fn pre_intervention_controls(
    matched: &[PerformedAttack],
    introduced: &[PerformedAttack],
) -> Vec<PerformedAttack> {
    let first_introduced = introduced
        .first()
        .expect("control panel requires an introduced attack")
        .start_time;
    let cutoff = first_introduced - PRE_INTERVENTION_GUARD_SECS;
    let split_origins = split_origin_mask(matched, introduced);
    let mut eligible: Vec<_> = matched
        .iter()
        .enumerate()
        .filter(|(index, attack)| {
            !split_origins[*index]
                && attack.start_time >= PRE_WINDOW_SECS
                && attack.start_time + POST_WINDOW_SECS < cutoff
        })
        .map(|(_, attack)| attack.clone())
        .collect();

    // Keep the latest controls before the intervention boundary so they remain
    // musically close while still satisfying the frozen one-second guard.
    if eligible.len() > MAX_CONTROLS_PER_SEED {
        eligible.drain(..eligible.len() - MAX_CONTROLS_PER_SEED);
    }
    eligible
}

#[test]
fn frozen_gate_records_real_music_pre_intervention_control_detections_without_hiding_them() {
    let config = RenderedAttackGateConfigV1::default();
    assert_eq!(config.ratio_floor_rms, 1.0e-6);
    assert_eq!(config.minimum_post_difference_rms, 1.0e-4);
    assert_eq!(config.minimum_post_to_pre_ratio, 2.0);

    let spec = Style::Sonata.spec();
    let state = MusicalState::default();
    let mut samples = Vec::new();
    let mut provenance_subjects = Vec::new();

    for seed in SEEDS {
        let intent = MusicalIntent {
            seed,
            bars: 4,
            ..MusicalIntent::default()
        };
        let realization = compose_sonata_with_plan(&intent, &spec).unwrap();
        let target = realization
            .plan
            .sections
            .iter()
            .find(|section| section.kind == SonataSectionKind::RecapitulationPrimary)
            .unwrap();
        let split_count = realization
            .score
            .notes
            .iter()
            .filter(|note| should_rearticulate(note, target.start, target.end))
            .count();
        assert!(split_count > 0, "seed {seed}: no bounded symbolic split available");

        let candidate_score =
            rearticulate_accompaniment(&realization.score, target.start, target.end);
        let baseline_performed = perform_with_spec(&realization.score, &spec, seed, &state);
        let candidate_performed = perform_with_spec(&candidate_score, &spec, seed, &state);
        let baseline_attacks = accompaniment_attacks(&baseline_performed);
        let candidate_attacks = accompaniment_attacks(&candidate_performed);
        let (matched, introduced) = partition_candidate_attacks(&baseline_attacks, &candidate_attacks);
        assert_eq!(
            introduced.len(),
            split_count,
            "seed {seed}: control selection requires the same performed-event conservation as C4"
        );
        let controls = pre_intervention_controls(&matched, &introduced);
        assert!(
            !controls.is_empty(),
            "seed {seed}: no uncontaminated pre-intervention controls satisfy the frozen guard"
        );

        let baseline_render =
            realize_with_spec(&realization.score, &spec, seed, &state, SAMPLE_RATE);
        let candidate_render =
            realize_with_spec(&candidate_score, &spec, seed, &state, SAMPLE_RATE);
        let baseline_frames = stereo_frames(&baseline_render.audio);
        let candidate_frames = stereo_frames(&candidate_render.audio);
        assert_eq!(baseline_frames.len(), candidate_frames.len());

        let subject_id = format!("sonata-seed-{seed}");
        provenance_subjects.push(RenderedAttackSubjectProvenanceV1 {
            subject_id: subject_id.clone(),
            seed,
            baseline_score_sha256: canonical_json_sha256(&realization.score).unwrap(),
            candidate_score_sha256: canonical_json_sha256(&candidate_score).unwrap(),
            baseline_audio_sha256: stereo_sha256(baseline_frames),
            candidate_audio_sha256: stereo_sha256(candidate_frames),
        });

        for (attack_ordinal, attack) in controls.iter().enumerate() {
            let evidence = measure_rendered_attack(
                baseline_frames,
                candidate_frames,
                SAMPLE_RATE,
                attack.start_time,
                PRE_WINDOW_SECS,
                POST_WINDOW_SECS,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "seed {seed} control {attack_ordinal}: malformed rendered-attack evidence: {error:?}"
                )
            });
            let gate = evaluate_rendered_attack_gate(&evidence, config).unwrap_or_else(|error| {
                panic!(
                    "seed {seed} control {attack_ordinal}: frozen gate could not evaluate valid evidence: {error:?}"
                )
            });
            samples.push(RenderedAttackPanelSampleV1 {
                subject_id: subject_id.clone(),
                attack_ordinal,
                attack_time_secs: attack.start_time,
                gate,
            });
        }
    }

    let panel = summarize_rendered_attack_panel(&samples).unwrap();
    assert_eq!(panel.subject_count, SEEDS.len());
    assert_eq!(panel.localized_change_count + panel.rejected_count, panel.attack_count);
    assert!(panel.subjects.iter().all(|subject| subject.attack_count > 0));

    let provenance = RenderedAttackPanelProvenanceV1 {
        provenance_version: RENDERED_ATTACK_PROVENANCE_VERSION.into(),
        renderer_id: RENDERER_ID.into(),
        sample_rate: SAMPLE_RATE,
        pre_window_seconds: PRE_WINDOW_SECS,
        post_window_seconds: POST_WINDOW_SECS,
        subjects: provenance_subjects,
    };
    let bound = bind_rendered_attack_panel(&panel, &provenance).unwrap();

    // Detections are NOT asserted to be zero. Any firing on these earlier
    // unchanged, non-split-origin attacks is retained as evidence of nonlocal
    // or global A/B change that a later specificity analysis must account for.
    println!(
        "MEL003C5A controls={} subjects={} detected={} rejected={} post_energy={} growth={} pre_rms_mean={:.8} post_rms_mean={:.8} ratio_mean={:.3} binding_sha256={}",
        panel.attack_count,
        panel.subject_count,
        panel.localized_change_count,
        panel.rejected_count,
        panel.post_energy_requirement_met_count,
        panel.growth_requirement_met_count,
        panel.pre_difference_rms.mean,
        panel.post_difference_rms.mean,
        panel.post_to_pre_ratio.mean,
        bound.binding_sha256
    );
}
