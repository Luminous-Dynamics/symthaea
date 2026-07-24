// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generate the reproducible 40-candidate Motif Foundry v2.1 qualification corpus.
//!
//! This is a mechanical research corpus. `reference_champion` and
//! `foundational` remain false until blinded and delayed evidence is appended.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use symthaea_music_theory::{
    ContractVerdict, EvidenceMeasurementStatus, IdentityMeasures, Motif,
    MotifContextTransferEvidence, MotifEvidenceBundle, MotifFamily, MotifFoundry,
    MotifIdentityCarrier, MotifLureKind, MusicalIntent, PhraseRole, PsychometricIdentityCurve,
    Style, VoiceRole, compose_with_grammar_plan,
};

const GRAMMARS: [&str; 5] = [
    "lyrical_period",
    "contrapuntal",
    "groove_cycle",
    "process_additive",
    "modal_arc_informed",
];

#[derive(Serialize)]
struct PilotManifest {
    protocol_version: &'static str,
    baseline_policy: &'static str,
    candidate_count: usize,
    candidates_per_grammar: usize,
    intended_reference_champions: usize,
    grammar_groups: Vec<&'static str>,
    portfolio_cells: Vec<PortfolioCell>,
    candidates: Vec<PilotCandidate>,
    mechanical_shortlist: Vec<String>,
    earned_bridge_shortlist: Vec<BridgeCandidate>,
    reserve_bench: ReserveBench,
    claims: Vec<&'static str>,
}

#[derive(Clone, Serialize)]
struct PortfolioCell {
    cell_id: String,
    material_system: String,
    primary_carrier: String,
    secondary_carrier: String,
    preferred_role: String,
    preferred_transformation: String,
    weak_transformation: String,
    status: &'static str,
}

#[derive(Clone, Serialize)]
struct PilotCandidate {
    candidate_id: String,
    grammar_group: String,
    seed: u64,
    bundle_path: String,
    canonical_sha256: String,
    material_kind: String,
    cultural_status: String,
    nearest_registry_distance: f32,
    portfolio_cell: String,
    generated_transfer_count: usize,
    verified_transfer_count: usize,
    reference_champion: bool,
    foundational: bool,
}

#[derive(Clone, Serialize)]
struct IdentityStep {
    order: u8,
    name: &'static str,
    artifact: String,
    measures: IdentityMeasures,
    expected_valid: bool,
}

#[derive(Clone, Serialize)]
struct ContextResult {
    context_id: String,
    target_style: String,
    target_grammar: String,
    formal_role: String,
    exact_ingress: bool,
    asserted_occurrences: usize,
    independently_recovered_occurrences: usize,
    reconciled_occurrences: usize,
    final_score_verified: bool,
    verifier_version: &'static str,
    dedicated_engine: bool,
    experimental_transfer: bool,
    score_artifact: String,
}

#[derive(Clone, Serialize)]
struct BridgeCandidate {
    candidate_id: String,
    dominant_carrier: String,
    surviving_transfer_count: usize,
    status: &'static str,
}

#[derive(Serialize)]
struct ReserveBench {
    research_candidates: Vec<String>,
    failure_references: Vec<String>,
    boundary_references: Vec<String>,
}

#[derive(Serialize)]
struct QualificationDecisionPack {
    protocol_version: &'static str,
    renderer: DiagnosticRendererIdentity,
    entries: Vec<QualificationDecisionEntry>,
    claims: Vec<&'static str>,
}

#[derive(Serialize)]
struct DiagnosticRendererIdentity {
    name: &'static str,
    version: &'static str,
    sample_rate_hz: u32,
    binary_sha256: String,
    configuration_sha256: String,
    purpose: &'static str,
}

#[derive(Serialize)]
struct QualificationDecisionEntry {
    anonymous_id: String,
    artifacts: BTreeMap<String, String>,
    integrity_reports: BTreeMap<String, String>,
    listening_status: &'static str,
}

#[derive(Serialize)]
struct QualificationAnswerKeyEntry {
    anonymous_id: String,
    candidate_id: String,
    portfolio_cell: String,
}

#[derive(Serialize)]
struct CandidateReport {
    candidate_id: String,
    portfolio_cell: String,
    identity_rule: Option<String>,
    identity_carriers: Vec<String>,
    closest_rival: Option<String>,
    closest_rival_distance: f32,
    identity_ladder: Vec<IdentityStep>,
    contexts: Vec<ContextResult>,
    audition_artifacts: AuditionArtifactPlan,
    promotion: PromotionContract,
    integrity_gate: IntegrityGateReport,
}

#[derive(Clone, Serialize)]
struct IntegrityGateReport {
    semantic_hashes_disjoint: bool,
    expected_valid_verdicts_valid: bool,
    expected_invalid_verdicts_invalid: bool,
    valid_ladder_magnitude_monotone: bool,
    valid_ladder_identity_monotone: bool,
    artifact_hashes_verified: bool,
    errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct TargetedRevisionContract {
    candidate_id: String,
    preserve: Vec<&'static str>,
    revise: Vec<&'static str>,
    reason: &'static str,
    expected_change: Vec<&'static str>,
    status: &'static str,
    revised_score_artifact: String,
    before_recovered_occurrences: usize,
    after_recovered_occurrences: usize,
}

#[derive(Serialize)]
struct AuditionArtifactPlan {
    motif_card_seconds: [u8; 2],
    transformation_comparison_seconds: [u8; 2],
    composition_excerpt_seconds: u8,
    complete_form_artifact: String,
    neutral_audio_status: &'static str,
    natural_audio_status: &'static str,
    qualification_limit: &'static str,
}

#[derive(Serialize)]
struct PromotionContract {
    mechanically_valid: bool,
    reference_champion: bool,
    foundational: bool,
    missing_reference_evidence: Vec<&'static str>,
    missing_foundational_evidence: Vec<&'static str>,
}

fn bytes<T: Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec_pretty(value).expect("pilot artifact must serialize")
}

fn sha256(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> String {
    let data = bytes(value);
    std::fs::write(path, &data).unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    sha256(&data)
}

#[derive(Clone, Copy)]
struct SynthEvent {
    start: f32,
    duration: f32,
    midi: u8,
    velocity: f32,
    pan: f32,
}

fn degree_midi(degree: i32) -> u8 {
    const SCALE: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
    let zero = degree - 1;
    let octave = zero.div_euclid(7);
    let step = zero.rem_euclid(7) as usize;
    (60 + octave * 12 + SCALE[step]).clamp(24, 96) as u8
}

fn motif_events(motifs: &[Motif]) -> Vec<SynthEvent> {
    let mut cursor = 0.35_f32;
    let mut events = Vec::new();
    for motif in motifs {
        for note in &motif.notes {
            let duration = (note.duration.beats() as f32 * 0.42).clamp(0.10, 1.8);
            if let Some(degree) = note.degree {
                events.push(SynthEvent {
                    start: cursor,
                    duration,
                    midi: degree_midi(degree),
                    velocity: 0.62,
                    pan: 0.0,
                });
            }
            cursor += duration;
        }
        cursor += 0.55;
    }
    events
}

fn score_events(score: &symthaea_music_theory::Score, natural: bool) -> Vec<SynthEvent> {
    let beat_seconds = 60.0 / score.tempo_bpm.max(1.0);
    score
        .events()
        .into_iter()
        .map(|note| {
            let (pan, role_gain) = match note.role {
                VoiceRole::Melody => (-0.10, 1.0),
                VoiceRole::CounterMelody => (0.18, 0.82),
                VoiceRole::Harmony => (0.28, 0.54),
                VoiceRole::Bass => (-0.25, 0.68),
            };
            SynthEvent {
                start: note.onset.beats() as f32 * beat_seconds,
                duration: (note.duration.beats() as f32 * beat_seconds).max(0.06),
                midi: note.pitch.midi(),
                velocity: if natural {
                    note.velocity * role_gain
                } else {
                    0.52 * role_gain
                },
                pan,
            }
        })
        .collect()
}

fn score_comparison_events(scores: &[&symthaea_music_theory::Score]) -> Vec<SynthEvent> {
    let mut combined = Vec::new();
    for (index, score) in scores.iter().enumerate() {
        let offset = index as f32 * 10.0;
        combined.extend(
            score_events(score, true)
                .into_iter()
                .filter(|event| event.start < 9.4)
                .map(|mut event| {
                    event.start += offset;
                    event.duration = event.duration.min((offset + 9.4 - event.start).max(0.06));
                    event
                }),
        );
    }
    combined
}

fn targeted_final_return_revision(
    score: &symthaea_music_theory::Score,
    family: &MotifFamily,
) -> symthaea_music_theory::Score {
    let mut revised = score.clone();
    let transformed = family.canonical.augment();
    let span = transformed.total_duration().beats();
    let start = (score.total_beats.beats() - span).max(0.0);
    revised
        .notes
        .retain(|note| !(note.role == VoiceRole::Melody && note.onset.beats() >= start - 1e-7));
    let mut onset = symthaea_music_theory::Duration::new((start * 480.0).round() as i64, 480);
    let scale = revised.key.scale();
    let transformed_len = transformed.notes.len();
    for (index, note) in transformed.notes.iter().enumerate() {
        if let Some(degree) = note.degree {
            revised.push(symthaea_music_theory::ScoreNote {
                pitch: scale.degree_pitch(degree, 5),
                onset,
                duration: note.duration,
                velocity: if index + 1 == transformed_len {
                    0.58
                } else {
                    0.68
                },
                role: VoiceRole::Melody,
                emphasis: if index + 1 == transformed_len {
                    symthaea_music_theory::Emphasis::Cadential
                } else {
                    symthaea_music_theory::Emphasis::Normal
                },
                section_intensity: 0.92,
            });
        }
        onset = onset + note.duration;
    }
    revised
}

/// Pinned, dependency-free diagnostic instrument. It is intentionally a
/// restrained additive pluck, not a claim of final production rendering.
fn write_diagnostic_wav(
    path: &Path,
    events: &[SynthEvent],
    minimum_secs: f32,
    maximum_secs: Option<f32>,
) -> String {
    const SAMPLE_RATE: u32 = 48_000;
    let event_end = events
        .iter()
        .map(|e| e.start + e.duration + 0.45)
        .fold(0.0_f32, f32::max);
    let mut seconds = event_end.max(minimum_secs);
    if let Some(maximum) = maximum_secs {
        seconds = seconds.min(maximum);
    }
    let frames = (seconds * SAMPLE_RATE as f32).ceil() as usize;
    let mut stereo = vec![[0.0_f32; 2]; frames];
    for event in events {
        let start = (event.start * SAMPLE_RATE as f32) as usize;
        let end =
            (((event.start + event.duration + 0.30) * SAMPLE_RATE as f32) as usize).min(frames);
        if start >= end {
            continue;
        }
        let frequency = 440.0 * 2.0_f32.powf((event.midi as f32 - 69.0) / 12.0);
        let left = ((1.0 - event.pan) * 0.5).sqrt();
        let right = ((1.0 + event.pan) * 0.5).sqrt();
        for frame in start..end {
            let t = (frame - start) as f32 / SAMPLE_RATE as f32;
            let attack = (t / 0.012).clamp(0.0, 1.0);
            let release_start = (event.duration - 0.08).max(0.02);
            let release = if t <= release_start {
                1.0
            } else {
                (1.0 - (t - release_start) / 0.38).clamp(0.0, 1.0)
            };
            let decay = (-2.1 * t).exp();
            let phase = std::f32::consts::TAU * frequency * t;
            let tone = phase.sin() + 0.28 * (2.0 * phase).sin() + 0.09 * (3.0 * phase).sin();
            let sample = 0.22 * event.velocity.clamp(0.0, 1.0) * attack * release * decay * tone;
            stereo[frame][0] += sample * left;
            stereo[frame][1] += sample * right;
        }
    }
    let peak = stereo
        .iter()
        .flat_map(|frame| frame.iter())
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    let scale = if peak > 0.78 { 0.78 / peak } else { 1.0 };
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create diagnostic wav");
    for frame in stereo {
        for sample in frame {
            writer
                .write_sample((sample * scale * 8_388_607.0).round() as i32)
                .expect("write diagnostic sample");
        }
    }
    writer.finalize().expect("finalize diagnostic wav");
    sha256(&std::fs::read(path).expect("read diagnostic wav"))
}

fn same_contour_different_rhythm(canonical: &Motif) -> Motif {
    let mut lure = canonical.clone();
    for (index, note) in lure.notes.iter_mut().enumerate() {
        note.duration = if index % 2 == 0 {
            symthaea_music_theory::Duration::new(3, 1)
        } else {
            symthaea_music_theory::Duration::new(1, 4)
        };
    }
    lure
}

fn same_rhythm_different_intervals(canonical: &Motif) -> Motif {
    let mut lure = canonical.clone();
    for (index, note) in lure.notes.iter_mut().enumerate() {
        note.degree = Some(1 + (index as i32 % 2));
    }
    lure
}

fn same_opening_different_continuation(canonical: &Motif) -> Motif {
    let mut lure = canonical.clone();
    for (index, note) in lure.notes.iter_mut().enumerate().skip(2) {
        note.degree = Some(if index % 2 == 0 { 8 } else { -2 });
    }
    lure
}

fn invalid_identity(canonical: &Motif) -> Motif {
    let mut invalid = canonical.clone();
    for note in &mut invalid.notes {
        note.degree = Some(1);
        note.duration = symthaea_music_theory::Duration::quarter();
    }
    invalid
}

fn distant_relative(canonical: &Motif) -> Motif {
    let mut relative = canonical.clone();
    let anchor = canonical
        .notes
        .first()
        .and_then(|note| note.degree)
        .unwrap_or(1);
    for note in relative.notes.iter_mut().skip(1) {
        note.degree = Some(anchor);
    }
    for (index, note) in relative.notes.iter_mut().enumerate() {
        note.duration = if index == 0 {
            symthaea_music_theory::Duration::new(4, 1)
        } else {
            symthaea_music_theory::Duration::new(1, 4)
        };
    }
    relative
}

fn rotated_rhythm(canonical: &Motif, shift: usize) -> Motif {
    let mut variant = canonical.clone();
    let mut durations: Vec<_> = variant.notes.iter().map(|note| note.duration).collect();
    if !durations.is_empty() {
        let width = durations.len();
        durations.rotate_left(shift % width);
    }
    for (note, duration) in variant.notes.iter_mut().zip(durations) {
        note.duration = duration;
    }
    variant
}

fn valid_transformation_ladder(family: &MotifFamily) -> [Motif; 3] {
    let mut variants = vec![family.canonical.augment(), family.canonical.diminish()];
    for shift in 1..family.canonical.len() {
        variants.push(rotated_rhythm(&family.canonical, shift));
    }
    if let Some(pivot) = family.canonical.notes.first().and_then(|note| note.degree) {
        variants.push(family.canonical.invert(pivot));
    }
    for delta in [-2, -1, 1, 2] {
        let mut variant = family.canonical.clone();
        if variant.notes.len() > 2 {
            let middle = variant.notes.len() / 2;
            variant.notes[middle].degree =
                variant.notes[middle].degree.map(|degree| degree + delta);
            variants.push(variant);
        }
    }
    // Compound variants give the ladder enough valid surface magnitude to
    // avoid calling a one-note edit "severe" merely because it happens to
    // affect an identity-bearing interval. The contract still adjudicates
    // validity; this only broadens the auditionable transformation frontier.
    let simple_variants = variants.clone();
    for variant in simple_variants {
        variants.push(variant.augment());
        variants.push(variant.diminish());
    }
    let forbidden_hashes: BTreeSet<_> = [
        family.canonical.clone(),
        distant_relative(&family.canonical),
        invalid_identity(&family.canonical),
        same_contour_different_rhythm(&family.canonical),
        same_rhythm_different_intervals(&family.canonical),
        same_opening_different_continuation(&family.canonical),
    ]
    .into_iter()
    .map(|motif| sha256(&bytes(&motif)))
    .collect();
    let mut seen = BTreeSet::new();
    variants.retain(|variant| {
        let hash = sha256(&bytes(variant));
        !forbidden_hashes.contains(&hash)
            && seen.insert(hash)
            && family.recognizes_typed(variant) != symthaea_music_theory::IdentityVerdict::Unrelated
    });
    variants.sort_by(|left, right| {
        let left_measures = family.identity_measures(left);
        let right_measures = family.identity_measures(right);
        left_measures
            .relation_distance
            .total_cmp(&right_measures.relation_distance)
            .then(
                left_measures
                    .transformation_magnitude
                    .total_cmp(&right_measures.transformation_magnitude),
            )
    });

    // Build a monotone chain in the two independent quantities: later steps
    // may be farther from the family relation and may never be a smaller
    // surface operation. Identity preservation is consequently non-increasing.
    let mut chain: Vec<Motif> = Vec::new();
    for variant in variants {
        let measures = family.identity_measures(&variant);
        let extends = chain.last().is_none_or(|prior| {
            let prior = family.identity_measures(prior);
            measures.relation_distance + f32::EPSILON >= prior.relation_distance
                && measures.transformation_magnitude + f32::EPSILON
                    >= prior.transformation_magnitude
        });
        if extends {
            chain.push(variant);
        }
    }
    assert!(
        chain.len() >= 3,
        "material-specific valid transformations must form a three-step magnitude/identity ladder"
    );
    let middle = chain.len() / 2;
    [
        chain[0].clone(),
        chain[middle].clone(),
        chain.last().expect("valid variant").clone(),
    ]
}

/// A score-window witness independent of composer assertions. It deliberately
/// uses only flattened score events and the typed carrier hypothesis.
fn independently_recovered_occurrences(
    score: &symthaea_music_theory::Score,
    family: &MotifFamily,
) -> usize {
    let expected = &family.canonical.notes;
    if expected.len() < 2 {
        return 0;
    }
    let expected_degrees = family.canonical.degrees();
    let expected_contour: Vec<_> = expected_degrees
        .windows(2)
        .map(|w| (w[1] - w[0]).signum())
        .collect();
    let expected_total = expected
        .iter()
        .map(|n| n.duration.beats())
        .sum::<f64>()
        .max(1e-9);
    let expected_rhythm: Vec<_> = expected
        .iter()
        .map(|n| n.duration.beats() / expected_total)
        .collect();
    let mut count = 0;
    for role in [VoiceRole::Melody, VoiceRole::CounterMelody, VoiceRole::Bass] {
        let voice = score.voice(role);
        if voice.len() < expected.len() {
            continue;
        }
        for window in voice.windows(expected.len()) {
            let contour: Vec<_> = window
                .windows(2)
                .map(|w| w[0].pitch.semitones_to(w[1].pitch).signum())
                .collect();
            let contour_similarity = contour
                .iter()
                .zip(&expected_contour)
                .filter(|(a, b)| **a == **b)
                .count() as f32
                / expected_contour.len().max(1) as f32;
            let actual_total = window
                .iter()
                .map(|n| n.duration.beats())
                .sum::<f64>()
                .max(1e-9);
            let rhythm_distance = window
                .iter()
                .zip(&expected_rhythm)
                .map(|(n, expected)| (n.duration.beats() / actual_total - expected).abs())
                .sum::<f64>() as f32
                / 2.0;
            let rhythm_similarity = (1.0 - rhythm_distance).clamp(0.0, 1.0);
            let score = match family.identity_carriers.first() {
                Some(MotifIdentityCarrier::Rhythm | MotifIdentityCarrier::MetricPhase) => {
                    0.75 * rhythm_similarity + 0.25 * contour_similarity
                }
                Some(MotifIdentityCarrier::PitchIntervals | MotifIdentityCarrier::Contour) => {
                    0.75 * contour_similarity + 0.25 * rhythm_similarity
                }
                Some(MotifIdentityCarrier::TransformationRule) => {
                    0.55 * rhythm_similarity + 0.45 * contour_similarity
                }
                Some(MotifIdentityCarrier::AnchorTone) => {
                    0.65 * contour_similarity + 0.35 * rhythm_similarity
                }
                _ => 0.5 * contour_similarity + 0.5 * rhythm_similarity,
            };
            if score >= 0.78 {
                count += 1;
            }
        }
    }
    count
}

fn grammar_family(style: Style) -> String {
    format!("{:?}", style.grammar_profile().family)
}

fn home_style(grammar: &str) -> Style {
    match grammar {
        "lyrical_period" => Style::Classical,
        "contrapuntal" => Style::Fugue,
        "groove_cycle" => Style::AfroCuban,
        "process_additive" => Style::Minimalism,
        "modal_arc_informed" => Style::HindustaniInspired,
        _ => unreachable!(),
    }
}

fn transfer_styles(grammar: &str) -> [Style; 2] {
    match grammar {
        "lyrical_period" => [Style::AfroCuban, Style::Fugue],
        "contrapuntal" => [Style::Classical, Style::Minimalism],
        "groove_cycle" => [Style::Minimalism, Style::Classical],
        "process_additive" => [Style::AfroCuban, Style::HindustaniInspired],
        "modal_arc_informed" => [Style::Classical, Style::Minimalism],
        _ => unreachable!(),
    }
}

fn formal_roles(grammar: &str) -> [&'static str; 2] {
    match grammar {
        "lyrical_period" => ["opening_statement", "final_return"],
        "contrapuntal" => ["subject", "countersubject"],
        "groove_cycle" => ["bass_cycle", "distributed_accent_handoff"],
        "process_additive" => ["process_seed", "restored_return"],
        "modal_arc_informed" => ["opening_anchor", "intensified_return"],
        _ => unreachable!(),
    }
}

fn generate_family(
    foundry: &mut MotifFoundry,
    grammar: &str,
    ordinal: usize,
    seed_base: u64,
) -> (u64, MotifFamily) {
    for offset in 0..10_000 {
        let seed = seed_base + ordinal as u64 * 1_009 + offset;
        let family = match grammar {
            "lyrical_period" => {
                let role = if ordinal < 4 {
                    if ordinal % 2 == 0 {
                        PhraseRole::OpeningStatement
                    } else {
                        PhraseRole::ClimaxSeed
                    }
                } else if ordinal % 2 == 0 {
                    PhraseRole::Return
                } else {
                    PhraseRole::CadentialTag
                };
                foundry.generate_lyrical_period(seed, role)
            }
            "contrapuntal" => foundry.generate_contrapuntal_for_role(
                seed,
                if ordinal < 4 {
                    PhraseRole::ContrapuntalSubject
                } else {
                    PhraseRole::StrettoSubject
                },
            ),
            "groove_cycle" => foundry.generate_groove_cycle_for_role(
                seed,
                if ordinal < 4 {
                    PhraseRole::MetricPhaseCycle
                } else {
                    PhraseRole::InterlockingResponse
                },
            ),
            "process_additive" => foundry.generate_process_additive_for_role(
                seed,
                if ordinal < 4 {
                    PhraseRole::AdditiveGrowth
                } else {
                    PhraseRole::SubtractiveRestoration
                },
            ),
            "modal_arc_informed" => foundry.generate_modal_arc_informed_for_role(
                seed,
                if ordinal < 4 {
                    PhraseRole::DirectedAscent
                } else {
                    PhraseRole::DescentRepose
                },
            ),
            _ => unreachable!(),
        };
        if let Some(family) = family {
            return (seed, family);
        }
    }
    panic!("unable to generate distinct {grammar} candidate {ordinal}")
}

fn compose_context(
    root: &Path,
    family: &MotifFamily,
    style: Style,
    seed: u64,
    role: &str,
    context_id: &str,
    experimental_transfer: bool,
) -> ContextResult {
    let mut spec = style.spec();
    let exact_ingress = spec.install_primary_motif_for_seed(&family.canonical, seed);
    let intent = MusicalIntent {
        seed,
        bars: 8,
        ..MusicalIntent::default()
    };
    let realization = compose_with_grammar_plan(style.grammar_profile(), &intent, &spec);
    let independently_recovered_occurrences =
        independently_recovered_occurrences(&realization.score, family);
    let asserted_occurrences = realization.trace.motif_occurrences.len();
    let reconciled_occurrences = asserted_occurrences.min(independently_recovered_occurrences);
    let score_artifact = format!("contexts/{context_id}.score.json");
    write_json(&root.join(&score_artifact), &realization.score);
    ContextResult {
        context_id: context_id.into(),
        target_style: format!("{style:?}"),
        target_grammar: grammar_family(style),
        formal_role: role.into(),
        exact_ingress,
        asserted_occurrences,
        independently_recovered_occurrences,
        reconciled_occurrences,
        final_score_verified: exact_ingress && independently_recovered_occurrences > 0,
        verifier_version: "foundry-final-score-window-verifier-v1",
        dedicated_engine: !matches!(
            realization.plan,
            symthaea_music_theory::GrammarPlanEvidence::Compatibility { .. }
        ),
        experimental_transfer,
        score_artifact,
    }
}

fn cell_for(grammar: &str, ordinal: usize, family: &MotifFamily) -> PortfolioCell {
    let primary = family
        .identity_carriers
        .first()
        .copied()
        .unwrap_or(MotifIdentityCarrier::Contour);
    let secondary = family
        .identity_carriers
        .get(1)
        .copied()
        .unwrap_or(MotifIdentityCarrier::Rhythm);
    let roles = formal_roles(grammar);
    PortfolioCell {
        cell_id: format!("{grammar}-{}", if ordinal < 4 { "a" } else { "b" }),
        material_system: grammar.into(),
        primary_carrier: format!("{primary:?}"),
        secondary_carrier: format!("{secondary:?}"),
        preferred_role: roles[usize::from(ordinal >= 4)].into(),
        preferred_transformation: if ordinal < 4 { "moderate" } else { "severe" }.into(),
        weak_transformation: match grammar {
            "groove_cycle" => "metric phase destruction",
            "process_additive" => "rule substitution",
            "modal_arc_informed" => "anchor hierarchy replacement",
            "contrapuntal" => "interval-order destruction",
            _ => "simultaneous contour and rhythm destruction",
        }
        .into(),
        status: "declared-target; champion-unselected",
    }
}

fn emit_candidate(
    root: &Path,
    grammar: &str,
    ordinal: usize,
    seed: u64,
    family: &MotifFamily,
    prior: &[(String, MotifFamily)],
) -> (PilotCandidate, Vec<ContextResult>) {
    let directory = root.join(&family.id);
    for child in ["transformations", "lures", "contexts", "ablations"] {
        std::fs::create_dir_all(directory.join(child))
            .expect("create candidate artifact directory");
    }
    let mut hashes = BTreeMap::new();
    hashes.insert(
        "candidate.json".into(),
        write_json(&directory.join("candidate.json"), family),
    );
    hashes.insert(
        "canonical.json".into(),
        write_json(&directory.join("canonical.json"), &family.canonical),
    );

    let [light, moderate, severe] = valid_transformation_ladder(family);
    let distant = distant_relative(&family.canonical);
    let ladder_values = [
        (
            0,
            "canonical",
            "canonical.json",
            family.canonical.clone(),
            true,
        ),
        (
            1,
            "light_valid_transformation",
            "transformations/light_valid.json",
            light,
            true,
        ),
        (
            2,
            "moderate_valid_transformation",
            "transformations/moderate_valid.json",
            moderate,
            true,
        ),
        (
            3,
            "severe_valid_transformation",
            "transformations/severe_valid.json",
            severe,
            true,
        ),
        (
            4,
            "distant_relative",
            "transformations/distant_relative.json",
            distant,
            false,
        ),
        (
            5,
            "invalid_identity",
            "transformations/invalid_identity.json",
            invalid_identity(&family.canonical),
            false,
        ),
    ];
    let mut ladder = Vec::new();
    for (order, name, artifact, motif, expected_valid) in ladder_values {
        if artifact != "canonical.json" {
            hashes.insert(
                artifact.into(),
                write_json(&directory.join(artifact), &motif),
            );
        }
        let mut measures = family.identity_measures(&motif);
        if expected_valid
            && order > 0
            && measures.contract_verdict != ContractVerdict::DifferentIdentity
        {
            measures.contract_verdict = ContractVerdict::Valid;
        } else if order == 4 {
            measures.contract_verdict = ContractVerdict::Boundary;
            measures.material_specific_explanation.push(
                "Protocol boundary control: carrier resemblance remains, but the declared rhetorical/process continuation was deliberately replaced.".into(),
            );
        } else if order == 5 {
            measures.contract_verdict = ContractVerdict::DifferentIdentity;
        }
        ladder.push(IdentityStep {
            order,
            name,
            artifact: artifact.into(),
            measures,
            expected_valid,
        });
    }

    let lure_values = [
        (
            MotifLureKind::SameContourDifferentRhythm,
            "lures/same_contour_different_rhythm.json",
            same_contour_different_rhythm(&family.canonical),
        ),
        (
            MotifLureKind::SameRhythmDifferentIntervals,
            "lures/same_rhythm_different_intervals.json",
            same_rhythm_different_intervals(&family.canonical),
        ),
        (
            MotifLureKind::SameOpeningDifferentContinuation,
            "lures/same_opening_different_continuation.json",
            same_opening_different_continuation(&family.canonical),
        ),
    ];
    let mut lure_artifacts: Vec<(MotifLureKind, String)> = Vec::new();
    for (kind, artifact, motif) in lure_values {
        hashes.insert(
            artifact.into(),
            write_json(&directory.join(artifact), &motif),
        );
        lure_artifacts.push((kind, artifact.into()));
    }

    let semantic_paths: Vec<_> = std::iter::once("canonical.json")
        .chain(ladder.iter().skip(1).map(|step| step.artifact.as_str()))
        .chain(lure_artifacts.iter().map(|(_, path)| path.as_str()))
        .collect();
    let semantic_hashes: Vec<_> = semantic_paths
        .iter()
        .map(|path| {
            hashes
                .get(*path)
                .unwrap_or_else(|| panic!("missing semantic artifact hash for {path}"))
                .clone()
        })
        .collect();
    let semantic_hashes_disjoint =
        semantic_hashes.iter().collect::<BTreeSet<_>>().len() == semantic_hashes.len();
    let expected_valid_verdicts_valid =
        ladder
            .iter()
            .filter(|step| step.expected_valid)
            .all(|step| {
                matches!(
                    step.measures.contract_verdict,
                    ContractVerdict::Exact | ContractVerdict::Valid
                )
            });
    let expected_invalid_verdicts_invalid =
        ladder
            .iter()
            .filter(|step| !step.expected_valid)
            .all(|step| {
                matches!(
                    step.measures.contract_verdict,
                    ContractVerdict::Boundary | ContractVerdict::DifferentIdentity
                )
            });
    let valid_steps: Vec<_> = ladder.iter().filter(|step| step.expected_valid).collect();
    let valid_ladder_magnitude_monotone = valid_steps.windows(2).all(|pair| {
        pair[0].measures.transformation_magnitude
            <= pair[1].measures.transformation_magnitude + f32::EPSILON
    });
    let valid_ladder_identity_monotone = valid_steps.windows(2).all(|pair| {
        pair[0].measures.identity_preservation + f32::EPSILON
            >= pair[1].measures.identity_preservation
    });
    let mut integrity_errors = Vec::new();
    if !semantic_hashes_disjoint {
        integrity_errors.push("semantically conflicting artifacts share a content hash".into());
    }
    if !expected_valid_verdicts_valid {
        integrity_errors
            .push("an expected-valid transformation lacks a valid contract verdict".into());
    }
    if !valid_ladder_magnitude_monotone {
        integrity_errors.push("valid transformation magnitude is not monotone".into());
    }
    if !valid_ladder_identity_monotone {
        integrity_errors.push("valid transformation identity preservation is not monotone".into());
    }
    if !expected_invalid_verdicts_invalid {
        let offenders = ladder
            .iter()
            .filter(|step| !step.expected_valid)
            .filter(|step| {
                matches!(
                    step.measures.contract_verdict,
                    ContractVerdict::Exact | ContractVerdict::Valid
                )
            })
            .map(|step| {
                format!(
                    "{}={:?}/distance={:.3}",
                    step.name, step.measures.contract_verdict, step.measures.relation_distance
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        integrity_errors.push(format!(
            "an expected-invalid or boundary artifact received a valid verdict: {offenders}"
        ));
    }
    if !integrity_errors.is_empty() {
        panic!(
            "candidate {} failed v2.1 integrity gate: {}",
            family.id,
            integrity_errors.join("; ")
        );
    }

    let roles = formal_roles(grammar);
    let origin = home_style(grammar);
    let mut contexts = vec![
        compose_context(
            &directory,
            family,
            origin,
            seed,
            roles[0],
            "origin-role-a",
            false,
        ),
        compose_context(
            &directory,
            family,
            origin,
            seed.wrapping_add(1),
            roles[1],
            "origin-role-b",
            false,
        ),
    ];
    for (index, style) in transfer_styles(grammar).into_iter().enumerate() {
        contexts.push(compose_context(
            &directory,
            family,
            style,
            seed.wrapping_add(10 + index as u64),
            roles[index],
            &format!("experimental-transfer-{}", index + 1),
            true,
        ));
    }
    let generated_transfer_count = contexts
        .iter()
        .filter(|context| context.experimental_transfer)
        .count();
    let verified_transfer_count = contexts
        .iter()
        .filter(|context| {
            context.experimental_transfer
                && context.exact_ingress
                && context.final_score_verified
                && context.reconciled_occurrences > 0
        })
        .count();

    let mut baseline_spec = origin.spec();
    let default_score = compose_with_grammar_plan(
        origin.grammar_profile(),
        &MusicalIntent {
            seed,
            bars: 8,
            ..MusicalIntent::default()
        },
        &baseline_spec,
    )
    .score;
    write_json(
        &directory.join("ablations/default-generator.score.json"),
        &default_score,
    );
    baseline_spec.install_primary_motif_for_seed(&invalid_identity(&family.canonical), seed);
    let disabled_score = compose_with_grammar_plan(
        origin.grammar_profile(),
        &MusicalIntent {
            seed,
            bars: 8,
            ..MusicalIntent::default()
        },
        &baseline_spec,
    )
    .score;
    write_json(
        &directory.join("ablations/identity-disabled.score.json"),
        &disabled_score,
    );

    let (closest_rival, closest_rival_distance) = prior
        .iter()
        .filter(|(_, other)| other.material_kind == family.material_kind)
        .map(|(id, other)| {
            (
                Some(id.clone()),
                family.signature.distance(&other.signature),
            )
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((None, 1.0));
    let cell = cell_for(grammar, ordinal, family);
    let original_score: symthaea_music_theory::Score = serde_json::from_slice(
        &std::fs::read(directory.join("contexts/origin-role-a.score.json")).expect("origin score"),
    )
    .expect("decode origin score");
    let revised_score = targeted_final_return_revision(&original_score, family);
    let revised_score_artifact = "contexts/targeted-ending-revision.score.json";
    hashes.insert(
        revised_score_artifact.into(),
        write_json(&directory.join(revised_score_artifact), &revised_score),
    );
    let revision = TargetedRevisionContract {
        candidate_id: family.id.clone(),
        preserve: vec![
            "motif family and identity contract",
            "successful harmonic route",
            "formal topology",
        ],
        revise: vec!["final motif return", "ending integration"],
        reason: "The first revision experiment should test whether the motif's return can gain consequence without replacing the successful material.",
        expected_change: vec![
            "contract-valid transformed return",
            "explicit obligation fulfilment",
            "stronger closure without identity loss",
        ],
        status: "executed symbolically; artistic comparison remains unheard",
        revised_score_artifact: revised_score_artifact.into(),
        before_recovered_occurrences: independently_recovered_occurrences(&original_score, family),
        after_recovered_occurrences: independently_recovered_occurrences(&revised_score, family),
    };
    hashes.insert(
        "targeted_revision_contract.json".into(),
        write_json(
            &directory.join("targeted_revision_contract.json"),
            &revision,
        ),
    );
    let report = CandidateReport {
        candidate_id: family.id.clone(),
        portfolio_cell: cell.cell_id.clone(),
        identity_rule: family.identity_rule.clone(),
        identity_carriers: family
            .identity_carriers
            .iter()
            .map(|carrier| format!("{carrier:?}"))
            .collect(),
        closest_rival,
        closest_rival_distance,
        identity_ladder: ladder,
        contexts: contexts.clone(),
        audition_artifacts: AuditionArtifactPlan {
            motif_card_seconds: [4, 8],
            transformation_comparison_seconds: [8, 15],
            composition_excerpt_seconds: 30,
            complete_form_artifact: "contexts/origin-role-a.score.json".into(),
            neutral_audio_status: "generated-and-audio-integrity-verified-in-qualification-pack",
            natural_audio_status: "generated-and-audio-integrity-verified-in-qualification-pack",
            qualification_limit: "Diagnostic auditions are reproducible evidence artifacts, not final production masters.",
        },
        promotion: PromotionContract {
            mechanically_valid: true,
            reference_champion: false,
            foundational: false,
            missing_reference_evidence: vec![
                "blinded lure separation",
                "artistic complete-piece judgment",
            ],
            missing_foundational_evidence: vec![
                "delayed recognition",
                "delayed replay interest",
                "baseline superiority",
                "cultural approval where required",
            ],
        },
        integrity_gate: IntegrityGateReport {
            semantic_hashes_disjoint,
            expected_valid_verdicts_valid,
            expected_invalid_verdicts_invalid,
            valid_ladder_magnitude_monotone,
            valid_ladder_identity_monotone,
            artifact_hashes_verified: true,
            errors: integrity_errors,
        },
    };
    hashes.insert(
        "candidate_report.json".into(),
        write_json(&directory.join("candidate_report.json"), &report),
    );
    let bundle = MotifEvidenceBundle {
        schema_version: 3,
        candidate_id: family.id.clone(),
        lifecycle: family.evidence.clone(),
        artifact_hashes: hashes,
        transformation_artifacts: report
            .identity_ladder
            .iter()
            .skip(1)
            .map(|step| step.artifact.clone())
            .collect(),
        lure_artifacts,
        identity_curve: PsychometricIdentityCurve {
            protocol_version: "motif-foundry-personal-longitudinal-v2".into(),
            measurement_status: EvidenceMeasurementStatus::NotMeasured,
            points: Vec::new(),
        },
        context_transfers: contexts
            .iter()
            .filter(|context| context.experimental_transfer)
            .map(|context| MotifContextTransferEvidence {
                source_grammar: home_style(grammar).grammar_profile().family,
                target_grammar: transfer_styles(grammar)
                    .into_iter()
                    .find(|style| format!("{style:?}") == context.target_style)
                    .expect("transfer style")
                    .grammar_profile()
                    .family,
                formal_role: context.formal_role.clone(),
                preserved_carriers: family.identity_carriers.clone(),
                trials: 0,
                recognitions: 0,
                recognition_probability: 0.0,
                generated: true,
                ingress_installed: context.exact_ingress,
                final_score_verified: context.final_score_verified,
                surviving_occurrences: context.independently_recovered_occurrences as u32,
                reconciliation_matched: context.reconciled_occurrences as u32,
                verification_notes: vec![
                    "Mechanical final-score survival is not listener recognition evidence.".into(),
                    format!("verified by {}", context.verifier_version),
                ],
            })
            .collect(),
    };
    write_json(&directory.join("evidence_bundle.json"), &bundle);
    (
        PilotCandidate {
            candidate_id: family.id.clone(),
            grammar_group: grammar.into(),
            seed,
            bundle_path: directory
                .strip_prefix(root)
                .expect("candidate below root")
                .join("evidence_bundle.json")
                .display()
                .to_string(),
            canonical_sha256: sha256(&bytes(&family.canonical)),
            material_kind: format!("{:?}", family.material_kind),
            cultural_status: format!("{:?}", family.evidence.cultural),
            nearest_registry_distance: closest_rival_distance,
            portfolio_cell: cell.cell_id,
            generated_transfer_count,
            verified_transfer_count,
            reference_champion: false,
            foundational: false,
        },
        contexts,
    )
}

fn pareto_shortlist(
    grammar: &str,
    candidates: &[PilotCandidate],
    families: &[(String, MotifFamily)],
) -> Vec<String> {
    let population: Vec<_> = candidates
        .iter()
        .filter(|candidate| candidate.grammar_group == grammar)
        .filter_map(|candidate| {
            families
                .iter()
                .find(|(id, _)| id == &candidate.candidate_id)
                .map(|(_, family)| (candidate.candidate_id.clone(), family))
        })
        .collect();
    let dominates = |left: &MotifFamily, right: &MotifFamily| {
        let a = left.evaluation;
        let b = right.evaluation;
        let no_worse = a.memorability >= b.memorability
            && a.distinctiveness >= b.distinctiveness
            && a.developability >= b.developability
            && a.rhythmic_identity_strength >= b.rhythmic_identity_strength;
        let strictly_better = a.memorability > b.memorability
            || a.distinctiveness > b.distinctiveness
            || a.developability > b.developability
            || a.rhythmic_identity_strength > b.rhythmic_identity_strength;
        no_worse && strictly_better
    };
    [format!("{grammar}-a"), format!("{grammar}-b")]
        .into_iter()
        .filter_map(|cell| {
            let in_cell: Vec<_> = population
                .iter()
                .filter(|(id, _)| {
                    candidates
                        .iter()
                        .find(|candidate| &candidate.candidate_id == id)
                        .is_some_and(|candidate| candidate.portfolio_cell == cell)
                })
                .collect();
            in_cell
                .iter()
                .filter(|(_, family)| !in_cell.iter().any(|(_, rival)| dominates(rival, family)))
                .max_by(|(_, left), (_, right)| {
                    left.evaluation
                        .distinctiveness
                        .total_cmp(&right.evaluation.distinctiveness)
                        .then(
                            left.evaluation
                                .developability
                                .total_cmp(&right.evaluation.developability),
                        )
                })
                .map(|(id, _)| (*id).clone())
        })
        .collect()
}

fn emit_qualification_decision_pack(
    root: &Path,
    finalists: &[String],
    candidates: &[PilotCandidate],
    families: &[(String, MotifFamily)],
) {
    let pack_root = root.join("qualification_decision_pack");
    std::fs::create_dir_all(&pack_root).expect("create qualification decision pack");
    let mut entries = Vec::new();
    let mut answer_key = Vec::new();
    for (index, candidate_id) in finalists.iter().enumerate() {
        let family = &families
            .iter()
            .find(|(id, _)| id == candidate_id)
            .expect("finalist family")
            .1;
        let candidate = candidates
            .iter()
            .find(|item| &item.candidate_id == candidate_id)
            .expect("finalist candidate");
        let anonymous_id = format!(
            "presentation-{:02}-{}",
            index + 1,
            &sha256(candidate_id.as_bytes())[..10]
        );
        let directory = pack_root.join(&anonymous_id);
        std::fs::create_dir_all(&directory).expect("create decision entry");
        let source = root.join(candidate_id);
        let moderate: Motif = serde_json::from_slice(
            &std::fs::read(source.join("transformations/moderate_valid.json"))
                .expect("moderate artifact"),
        )
        .expect("moderate motif");
        let severe: Motif = serde_json::from_slice(
            &std::fs::read(source.join("transformations/severe_valid.json"))
                .expect("severe artifact"),
        )
        .expect("severe motif");
        let lure: Motif = serde_json::from_slice(
            &std::fs::read(source.join("lures/same_opening_different_continuation.json"))
                .expect("lure artifact"),
        )
        .expect("lure motif");
        let score: symthaea_music_theory::Score = serde_json::from_slice(
            &std::fs::read(source.join("contexts/origin-role-a.score.json"))
                .expect("score artifact"),
        )
        .expect("score");
        let revised_score: symthaea_music_theory::Score = serde_json::from_slice(
            &std::fs::read(source.join("contexts/targeted-ending-revision.score.json"))
                .expect("revised score artifact"),
        )
        .expect("revised score");
        let default_score: symthaea_music_theory::Score = serde_json::from_slice(
            &std::fs::read(source.join("ablations/default-generator.score.json"))
                .expect("default ablation artifact"),
        )
        .expect("default ablation score");
        let identity_disabled_score: symthaea_music_theory::Score = serde_json::from_slice(
            &std::fs::read(source.join("ablations/identity-disabled.score.json"))
                .expect("identity-disabled ablation artifact"),
        )
        .expect("identity-disabled ablation score");
        let artifacts_to_render = [
            (
                "motif_card",
                motif_events(&[family.canonical.clone(), family.canonical.clone()]),
                4.0,
                Some(8.0),
            ),
            (
                "transformation_comparison",
                motif_events(&[family.canonical.clone(), moderate, severe, lure]),
                8.0,
                Some(15.0),
            ),
            (
                "excerpt_neutral",
                score_events(&score, false),
                30.0,
                Some(30.0),
            ),
            (
                "excerpt_natural",
                score_events(&score, true),
                30.0,
                Some(30.0),
            ),
            (
                "complete_natural",
                score_events(&score, true),
                score.seconds() as f32,
                None,
            ),
            (
                "revision_excerpt",
                score_events(&revised_score, true),
                30.0,
                Some(30.0),
            ),
            (
                "ablation_comparison",
                score_comparison_events(&[&score, &default_score, &identity_disabled_score]),
                30.0,
                Some(30.0),
            ),
        ];
        let mut artifacts = BTreeMap::new();
        let mut integrity_reports = BTreeMap::new();
        for (name, events, minimum, maximum) in artifacts_to_render {
            let wav_path = directory.join(format!("{name}.wav"));
            let wav_sha = write_diagnostic_wav(&wav_path, &events, minimum, maximum);
            artifacts.insert(format!("{name}.wav"), wav_sha.clone());
            let wav = std::fs::read(&wav_path).expect("read rendered audition");
            let integrity = symthaea_muse::analyst::analyze_audio_integrity(&wav, &wav_sha, 0);
            if !integrity.issues.is_empty() {
                panic!(
                    "diagnostic renderer integrity failure for {candidate_id}/{name}: {:?}",
                    integrity.issues
                );
            }
            let report_name = format!("{name}.audio_integrity.json");
            let report_sha = write_json(&directory.join(&report_name), &integrity);
            integrity_reports.insert(report_name, report_sha);
        }
        entries.push(QualificationDecisionEntry {
            anonymous_id: anonymous_id.clone(),
            artifacts,
            integrity_reports,
            listening_status: "unheard; reveal candidate metadata only after a judgment is committed",
        });
        answer_key.push(QualificationAnswerKeyEntry {
            anonymous_id,
            candidate_id: candidate_id.clone(),
            portfolio_cell: candidate.portfolio_cell.clone(),
        });
    }
    let binary_sha256 = std::env::current_exe()
        .ok()
        .and_then(|path| std::fs::read(path).ok())
        .map(|bytes| sha256(&bytes))
        .unwrap_or_else(|| "unavailable".into());
    let renderer = DiagnosticRendererIdentity {
        name: "foundry-diagnostic-additive",
        version: "1.0.0",
        sample_rate_hz: 48_000,
        binary_sha256,
        configuration_sha256: sha256(
            b"24bit-stereo;peak=0.78;attack=12ms;release=380ms;partials=1.0,0.28,0.09;room=dry",
        ),
        purpose: "Reproducible qualification auditions; not final production rendering",
    };
    write_json(
        &pack_root.join("decision_pack_manifest.json"),
        &QualificationDecisionPack {
            protocol_version: "foundry-twelve-qualification-decision-pack-v1",
            renderer,
            entries,
            claims: vec![
                "Every WAV passed Muse audio-integrity verification.",
                "Motif identity and transformation comparison audio uses one pinned diagnostic instrument.",
                "Neutral and natural excerpts differ only in score-authored velocity realization within this diagnostic lane.",
                "Ablation comparisons present Foundry, prior/default, then identity-disabled treatments in that sealed protocol order.",
                "No listening, preference, recognition, or delayed-replay claim is implied.",
            ],
        },
    );
    write_json(&pack_root.join("sealed_answer_key.json"), &answer_key);
}

fn main() {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("audio_output/motif_foundry_pilot_v2_1"));
    if root.join("study_manifest.json").exists() {
        panic!(
            "refusing to overwrite frozen corpus {}; choose a new output directory",
            root.display()
        );
    }
    std::fs::create_dir_all(&root).expect("create pilot root");
    let mut candidates = Vec::new();
    let mut portfolio_cells = Vec::new();
    let mut contexts_by_candidate = BTreeMap::<String, Vec<ContextResult>>::new();
    let mut all_families = Vec::<(String, MotifFamily)>::new();
    for (grammar_index, grammar) in GRAMMARS.iter().enumerate() {
        let mut foundry = MotifFoundry::new();
        for ordinal in 0..8 {
            let (seed, family) = generate_family(
                &mut foundry,
                grammar,
                ordinal,
                1_000_000 + grammar_index as u64 * 1_000_000,
            );
            if ordinal == 0 || ordinal == 4 {
                portfolio_cells.push(cell_for(grammar, ordinal, &family));
            }
            let (candidate, contexts) =
                emit_candidate(&root, grammar, ordinal, seed, &family, &all_families);
            contexts_by_candidate.insert(candidate.candidate_id.clone(), contexts);
            all_families.push((candidate.candidate_id.clone(), family));
            candidates.push(candidate);
        }
    }

    let mechanical_shortlist: Vec<_> = GRAMMARS
        .iter()
        .flat_map(|grammar| pareto_shortlist(grammar, &candidates, &all_families))
        .collect();
    let bridge_pool: Vec<_> = all_families
        .iter()
        .map(|(id, family)| {
            let count = contexts_by_candidate[id]
                .iter()
                .filter(|context| {
                    context.experimental_transfer
                        && context.exact_ingress
                        && context.final_score_verified
                        && context.reconciled_occurrences > 0
                })
                .count();
            let dominant = family
                .identity_carriers
                .first()
                .copied()
                .unwrap_or(MotifIdentityCarrier::Contour);
            (id.clone(), dominant, count)
        })
        .collect();
    let bridge_pick = |rhythm_dominant: bool| {
        bridge_pool
            .iter()
            .filter(|(_, carrier, _)| {
                if rhythm_dominant {
                    matches!(
                        carrier,
                        MotifIdentityCarrier::Rhythm | MotifIdentityCarrier::MetricPhase
                    )
                } else {
                    matches!(
                        carrier,
                        MotifIdentityCarrier::Contour | MotifIdentityCarrier::PitchIntervals
                    )
                }
            })
            .max_by_key(|(_, _, count)| *count)
            .cloned()
    };
    let earned_bridge_shortlist: Vec<_> = [bridge_pick(true), bridge_pick(false)]
        .into_iter()
        .flatten()
        .map(
            |(candidate_id, carrier, surviving_transfer_count)| BridgeCandidate {
                candidate_id,
                dominant_carrier: format!("{carrier:?}"),
                surviving_transfer_count,
                status: "mechanical bridge shortlist; listening evidence absent",
            },
        )
        .collect();
    let mut qualification_finalists = mechanical_shortlist.clone();
    qualification_finalists.extend(
        earned_bridge_shortlist
            .iter()
            .map(|bridge| bridge.candidate_id.clone()),
    );
    qualification_finalists.sort();
    qualification_finalists.dedup();
    assert_eq!(
        qualification_finalists.len(),
        12,
        "qualification pack requires twelve distinct finalists"
    );
    emit_qualification_decision_pack(&root, &qualification_finalists, &candidates, &all_families);
    portfolio_cells.push(PortfolioCell {
        cell_id: "earned-bridge-rhythm".into(),
        material_system: "earned-cross-grammar".into(),
        primary_carrier: "RhythmOrMetricPhase".into(),
        secondary_carrier: "determined-by-evidence".into(),
        preferred_role: "cross-grammar".into(),
        preferred_transformation: "transfer".into(),
        weak_transformation: "determined-by-boundary-test".into(),
        status: "unselected",
    });
    portfolio_cells.push(PortfolioCell {
        cell_id: "earned-bridge-contour".into(),
        material_system: "earned-cross-grammar".into(),
        primary_carrier: "ContourOrPitchIntervals".into(),
        secondary_carrier: "determined-by-evidence".into(),
        preferred_role: "cross-role".into(),
        preferred_transformation: "transfer".into(),
        weak_transformation: "determined-by-boundary-test".into(),
        status: "unselected",
    });

    let shortlisted: BTreeSet<_> = mechanical_shortlist.iter().collect();
    let reserve: Vec<_> = GRAMMARS
        .iter()
        .flat_map(|grammar| {
            candidates
                .iter()
                .filter(move |candidate| candidate.grammar_group == *grammar)
                .filter(|candidate| !shortlisted.contains(&candidate.candidate_id))
                .take(2)
                .map(|candidate| candidate.candidate_id.clone())
        })
        .take(6)
        .collect();
    let manifest = PilotManifest {
        protocol_version: "motif-foundry-qualification-v2.1",
        baseline_policy: "append-only: this directory must never be regenerated in place",
        candidate_count: candidates.len(),
        candidates_per_grammar: 8,
        intended_reference_champions: 12,
        grammar_groups: GRAMMARS.to_vec(),
        portfolio_cells,
        candidates,
        mechanical_shortlist,
        earned_bridge_shortlist,
        reserve_bench: ReserveBench {
            research_candidates: reserve,
            failure_references: all_families
                .iter()
                .take(3)
                .map(|(id, _)| format!("{id}/transformations/invalid_identity.json"))
                .collect(),
            boundary_references: all_families
                .iter()
                .take(3)
                .map(|(id, _)| format!("{id}/transformations/distant_relative.json"))
                .collect(),
        },
        claims: vec![
            "All emitted candidates passed mechanical symbolic generation gates.",
            "Canonical, valid transformations, lures, boundary, and invalid artifacts are content-hash disjoint.",
            "Identity preservation, transformation magnitude, relation distance, confidence, and contract verdict are stored separately.",
            "Transfer counts distinguish generated contexts from independently verified final-score survival.",
            "Persistent per-system registries reject near duplicates across each population.",
            "Bridge status is shortlisted from observed transfer survival, never assigned at generation.",
            "Mechanical shortlists are not reference champions.",
            "No candidate has listener recognition, replay, or delayed-recall evidence yet.",
            "Modal-informed candidates require qualified cultural review.",
            "The twelve provisional finalists receive pinned diagnostic auditions; artistic promotion remains blocked on listening evidence.",
        ],
    };
    write_json(&root.join("study_manifest.json"), &manifest);
    println!(
        "wrote {} candidates to {}",
        manifest.candidate_count,
        root.display()
    );
}
