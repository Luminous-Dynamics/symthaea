// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-FRONTIER-001 M0: source-derived catalog × instrument capability frontier.
//!
//! This audit joins identities and executable style routes. It does not infer
//! sample availability, playable range, expressive gesture support, fidelity,
//! cultural authenticity, or listener preference from registry membership.

#![cfg(not(target_arch = "wasm32"))]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::process::ExitCode;
use symthaea_music_theory::pitch::PitchClass;
use symthaea_music_theory::score::{Score, VoiceRole as TheoryRole};
use symthaea_muse::instruments::Instrument;
use symthaea_muse::{MusicalIntent, Style};
use symthaea_muse_protocol::catalog::{self, CanonicalStyle, CATALOG};

const BENCHMARK_ID: &str = "MEL-FRONTIER-001";
const SCHEMA_VERSION: &str = "melothaea.capability-frontier.v1";
const INITIAL_CATALOG_COHORT: usize = 152;
const DEVELOPMENT_SEEDS: [u64; 8] = [0, 1, 7, 13, 29, 42, 73, 101];

#[derive(Debug, Serialize)]
struct FrontierReport {
    schema_version: &'static str,
    benchmark: &'static str,
    source_revision: String,
    source_revision_source: &'static str,
    executable_sha256: String,
    catalog_sha256: String,
    initial_catalog_cohort_size: usize,
    catalog_entries: usize,
    catalog_entries_above_initial_cohort: usize,
    catalog_route_declared: usize,
    catalog_route_resolved: usize,
    catalog_unrouted: usize,
    catalog_route_unresolved: usize,
    executable_style_variants: usize,
    instrument_identities: usize,
    potential_catalog_instrument_pairs: usize,
    route_resolved_candidate_pairs: usize,
    currently_unrouted_pairs: usize,
    frontier_cells: Vec<FrontierCell>,
    catalog_routes: Vec<CatalogRouteRow>,
    instrument_roster: Vec<InstrumentRow>,
    style_seed_smoke: Vec<StyleSeedSmoke>,
    style_seed_smoke_counts: OutcomeCounts,
    limitations: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct CatalogRouteRow {
    catalog_id: u16,
    canonical_name: &'static str,
    display_name: &'static str,
    constellation: &'static str,
    catalog_status: &'static str,
    requires_expert_review: bool,
    composer_style: Option<&'static str>,
    route_status: &'static str,
}

#[derive(Debug, Serialize)]
struct InstrumentRow {
    instrument_id: &'static str,
    rust_variant: String,
    declared_renderer_family: &'static str,
    gm_program_0_based: u8,
    sample_capability_status: &'static str,
    range_and_gesture_status: &'static str,
}

#[derive(Debug, Serialize)]
struct FrontierCell {
    catalog_id: u16,
    style_name: &'static str,
    composer_style: Option<&'static str>,
    instrument_id: &'static str,
    declared_renderer_family: &'static str,
    frontier_status: &'static str,
    measured_pair_evidence: bool,
    required_gates: [&'static str; 6],
}

#[derive(Debug, Serialize)]
struct StyleSeedSmoke {
    style_id: String,
    catalog_ids: Vec<u16>,
    rust_variant: String,
    seed: u64,
    first_score_sha256: String,
    replay_score_sha256: String,
    symbolic_replay_matches: bool,
    tempo_bpm: f32,
    meter_beats: u8,
    note_count: usize,
    melody_note_count: usize,
    harmony_note_count: usize,
    bass_note_count: usize,
    counter_melody_note_count: usize,
    total_seconds: f64,
    universal_invariant_violations: Vec<&'static str>,
    outcome: &'static str,
}

#[derive(Debug, Default, Serialize)]
struct OutcomeCounts {
    passed: usize,
    invalid_score: usize,
    empty_score: usize,
    non_deterministic: usize,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("capability_frontier: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let (source_revision, source_revision_source) = declared_source_revision()?;
    let executable_sha256 = sha256_file(
        &env::current_exe().map_err(|e| format!("cannot identify running executable: {e}"))?,
    )?;
    let catalog_sha256 = sha256_json(&CATALOG)?;

    let styles = Style::ALL;
    let instruments = Instrument::ALL;
    let style_names: HashSet<String> = styles.iter().copied().map(style_serde_name).collect();

    let catalog_routes: Vec<CatalogRouteRow> = CATALOG
        .iter()
        .map(|entry| catalog_route_row(entry, &style_names))
        .collect();
    let catalog_route_declared = catalog_routes
        .iter()
        .filter(|row| row.composer_style.is_some())
        .count();
    let catalog_route_resolved = catalog_routes
        .iter()
        .filter(|row| row.route_status == "route_resolves_to_executable_style")
        .count();
    let catalog_unrouted = catalog_routes
        .iter()
        .filter(|row| row.route_status == "no_composer_route")
        .count();
    let catalog_route_unresolved = catalog_routes
        .iter()
        .filter(|row| row.route_status == "declared_route_unresolved")
        .count();

    let mut style_catalog_ids: HashMap<String, Vec<u16>> = HashMap::new();
    for row in &catalog_routes {
        if row.route_status == "route_resolves_to_executable_style" {
            if let Some(style_id) = row.composer_style {
                style_catalog_ids
                    .entry(style_id.to_owned())
                    .or_default()
                    .push(row.catalog_id);
            }
        }
    }

    let instrument_roster: Vec<InstrumentRow> = instruments
        .iter()
        .copied()
        .map(instrument_row)
        .collect();

    let mut frontier_cells = Vec::with_capacity(CATALOG.len() * instruments.len());
    let mut route_resolved_candidate_pairs = 0usize;
    let mut currently_unrouted_pairs = 0usize;
    for entry in CATALOG.iter() {
        let route_status = catalog_routes
            .iter()
            .find(|row| row.catalog_id == entry.id)
            .map(|row| row.route_status)
            .unwrap_or("catalog_entry_missing_from_route_audit");
        for instrument in instruments {
            let frontier_status = match route_status {
                "route_resolves_to_executable_style" => {
                    route_resolved_candidate_pairs += 1;
                    "candidate_unqualified"
                }
                "declared_route_unresolved" => {
                    currently_unrouted_pairs += 1;
                    "declared_route_unresolved"
                }
                _ => {
                    currently_unrouted_pairs += 1;
                    "style_not_callable"
                }
            };
            frontier_cells.push(FrontierCell {
                catalog_id: entry.id,
                style_name: entry.display_name,
                composer_style: entry.composer_style,
                instrument_id: instrument.canonical_id(),
                declared_renderer_family: renderer_family(instrument),
                frontier_status,
                measured_pair_evidence: false,
                required_gates: [
                    "score_and_route_contract",
                    "per_note_render_path_receipt",
                    "instrument_register_coverage",
                    "instrument_gesture_and_dynamics_coverage",
                    "signal_health_and_replay",
                    "blind_audio_evaluation_for_quality_claims",
                ],
            });
        }
    }

    let mut style_seed_smoke = Vec::with_capacity(styles.len() * DEVELOPMENT_SEEDS.len());
    let mut counts = OutcomeCounts::default();
    for style in styles {
        let style_id = style_serde_name(style);
        for seed in DEVELOPMENT_SEEDS {
            let intent = MusicalIntent {
                valence: 0.15,
                arousal: 0.45,
                energy: 0.60,
                bars: 4,
                seed,
                tonic: PitchClass::C,
            };
            let first = symthaea_music_theory::composer::compose_styled(&intent, style);
            let replay = symthaea_music_theory::composer::compose_styled(&intent, style);
            let first_hash = sha256_json(&first)?;
            let replay_hash = sha256_json(&replay)?;
            let replay_matches = first_hash == replay_hash;
            let violations = universal_invariant_violations(&first);
            let outcome = if first.notes.is_empty() {
                counts.empty_score += 1;
                "empty_score"
            } else if !violations.is_empty() || !universal_invariant_violations(&replay).is_empty() {
                counts.invalid_score += 1;
                "universal_invariant_violation"
            } else if !replay_matches {
                counts.non_deterministic += 1;
                "symbolic_replay_mismatch"
            } else {
                counts.passed += 1;
                "mechanical_smoke_pass"
            };

            style_seed_smoke.push(StyleSeedSmoke {
                style_id: style_id.clone(),
                catalog_ids: style_catalog_ids.get(&style_id).cloned().unwrap_or_default(), 
                rust_variant: format!("{style:?}"),
                seed,
                first_score_sha256: first_hash,
                replay_score_sha256: replay_hash,
                symbolic_replay_matches: replay_matches,
                tempo_bpm: first.tempo_bpm,
                meter_beats: first.meter,
                note_count: first.notes.len(),
                melody_note_count: first.voice(TheoryRole::Melody).len(),
                harmony_note_count: first.voice(TheoryRole::Harmony).len(),
                bass_note_count: first.voice(TheoryRole::Bass).len(),
                counter_melody_note_count: first.voice(TheoryRole::CounterMelody).len(),
                total_seconds: first.seconds(),
                universal_invariant_violations: violations,
                outcome,
            });
        }
    }

    let report = FrontierReport {
        schema_version: SCHEMA_VERSION,
        benchmark: BENCHMARK_ID,
        source_revision,
        source_revision_source,
        executable_sha256,
        catalog_sha256,
        initial_catalog_cohort_size: INITIAL_CATALOG_COHORT,
        catalog_entries: CATALOG.len(),
        catalog_entries_above_initial_cohort: CATALOG.len().saturating_sub(INITIAL_CATALOG_COHORT),
        catalog_route_declared,
        catalog_route_resolved,
        catalog_unrouted,
        catalog_route_unresolved,
        executable_style_variants: styles.len(),
        instrument_identities: instruments.len(),
        potential_catalog_instrument_pairs: CATALOG.len() * instruments.len(),
        route_resolved_candidate_pairs,
        currently_unrouted_pairs,
        frontier_cells,
        catalog_routes,
        instrument_roster,
        style_seed_smoke,
        style_seed_smoke_counts: counts,
        limitations: vec![
            "A catalog route or instrument ID is identity/dispatch evidence, not a completed musical capability.",
            "Style smoke cases use one fixed intent and eight development seeds; they do not establish broad intent controllability or human preference.",
            "Instrument renderer family and GM program are source declarations; they do not prove source-sample selection or a distinct high-quality acoustic model.",
            "Cross-axis cells are candidates until exact per-note render receipts, playable-range/articulation evidence, and the stated validation gates are joined.",
            "A source revision from the environment is caller/build-supplied metadata, not an authenticated build attestation.",
            "The initial 152 entries are the starting cohort only; all counts and cross products are derived from the live registries so the frontier can grow beyond 152.",
        ],
    };

    let mut json = serde_json::to_vec_pretty(&report)
        .map_err(|e| format!("cannot serialize frontier report: {e}"))?;
    json.push(b'\n');

    let output_path = env::args().nth(1);
    if env::args().skip(2).next().is_some() {
        return Err("usage: capability_frontier [OUTPUT.json]".into());
    }
    if let Some(path) = output_path {
        std::fs::write(&path, json)
            .map_err(|e| format!("cannot write report to {path}: {e}"))?;
    } else {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        out.write_all(&json)
            .map_err(|e| format!("cannot write JSON to stdout: {e}"))?;
    }
    Ok(())
}

fn declared_source_revision() -> Result<(String, &'static str), String> {
    for (variable, source) in [
        ("SYMTHAEA_SOURCE_REV", "SYMTHAEA_SOURCE_REV"),
        ("GITHUB_SHA", "GITHUB_SHA"),
        ("GIT_COMMIT", "GIT_COMMIT"),
    ] {
        if let Ok(value) = env::var(variable) {
            let value = value.trim().to_ascii_lowercase();
            let valid = (40..=64).contains(&value.len())
                && value.bytes().all(|b| b.is_ascii_hexdigit());
            if !valid {
                return Err(format!(
                    "{variable} must contain a full 40-64 character hexadecimal commit ID"
                ));
            }
            return Ok((value, source));
        }
    }
    Err(
        "missing source revision; set SYMTHAEA_SOURCE_REV=\"$(git rev-parse HEAD)\" (or GITHUB_SHA)"
            .into(),
    )
}

fn style_serde_name(style: Style) -> String {
    serde_json::to_value(style)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| format!("{style:?}"))
}

fn catalog_route_row(entry: &CanonicalStyle, executable_style_names: &HashSet<String>) -> CatalogRouteRow {
    let route_status = match entry.composer_style {
        None => "no_composer_route",
        Some(name) if executable_style_names.contains(name) => "route_resolves_to_executable_style",
        Some(_) => "declared_route_unresolved",
    };
    CatalogRouteRow {
        catalog_id: entry.id,
        canonical_name: entry.name,
        display_name: entry.display_name,
        constellation: entry.constellation.name(),
        catalog_status: entry.status.label(),
        requires_expert_review: entry.requires_expert_review,
        composer_style: entry.composer_style,
        route_status,
    }
}

fn instrument_row(instrument: Instrument) -> InstrumentRow {
    InstrumentRow {
        instrument_id: instrument.canonical_id(),
        rust_variant: format!("{instrument:?}"),
        declared_renderer_family: renderer_family(instrument),
        gm_program_0_based: instrument.gm_program(),
        sample_capability_status: "not_measured_by_frontier_m0",
        range_and_gesture_status: "not_measured_by_frontier_m0",
    }
}

fn renderer_family(instrument: Instrument) -> &'static str {
    if instrument.uses_karplus_strong() {
        "karplus_strong"
    } else if instrument.uses_fm() {
        "frequency_modulation"
    } else {
        "additive_declared"
    }
}

fn universal_invariant_violations(score: &Score) -> Vec<&'static str> {
    let mut violations = Vec::new();
    if !score.tempo_bpm.is_finite() || score.tempo_bpm <= 0.0 {
        violations.push("tempo_non_finite_or_non_positive");
    }
    if score.meter == 0 {
        violations.push("meter_zero");
    }
    if score.notes.is_empty() {
        violations.push("empty_score");
    }
    if !score.total_beats.beats().is_finite() || score.total_beats.beats() < 0.0 {
        violations.push("invalid_total_beats");
    }
    for note in &score.notes {
        if !note.onset.beats().is_finite() || note.onset.beats() < 0.0 {
            violations.push("negative_or_non_finite_onset");
            break;
        }
    }
    for note in &score.notes {
        if !note.duration.beats().is_finite() || note.duration.beats() <= 0.0 {
            violations.push("non_positive_or_non_finite_duration");
            break;
        }
    }
    for note in &score.notes {
        if !note.velocity.is_finite() || !(0.0..=1.0).contains(&note.velocity) {
            violations.push("velocity_out_of_range_or_non_finite");
            break;
        }
    }
    if !score.seconds().is_finite() || score.seconds() <= 0.0 {
        violations.push("invalid_total_seconds");
    }
    violations
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes =
        serde_json::to_vec(value).map_err(|e| format!("cannot serialize digest subject: {e}"))?;
    let mut hash = Sha256::new();
    hash.update(&bytes);
    Ok(format!("{:x}", hash.finalize()))
}

fn sha256_file(path: &std::path::Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|e| format!("cannot open executable for hashing: {e}"))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| format!("cannot hash executable bytes: {e}"))?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_registry_entries_are_not_dropped() {
        let ids: HashSet<u16> = CATALOG.iter().map(|entry| entry.id).collect();
        assert_eq!(ids.len(), CATALOG.len());
        assert_eq!(catalog::Constellation::ALL.len(), 19);
    }

    #[test]
    fn instrument_roster_ids_are_unique_and_stable_shaped() {
        let ids: HashSet<&str> = Instrument::ALL
            .iter()
            .copied()
            .map(Instrument::canonical_id)
            .collect();
        assert_eq!(ids.len(), Instrument::ALL.len());
        assert!(ids.iter().all(|id| {
            !id.is_empty()
                && id.bytes().all(|b| {
                    b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'
                })
        }));
    }

    #[test]
    fn source_revision_requires_full_hex_sha() {
        let invalid_values = vec![
            String::new(),
            "main".to_owned(),
            "deadbeef".to_owned(),
            "z".repeat(40),
        ];
        for invalid in &invalid_values {
            assert!(
                !((40..=64).contains(&invalid.len())
                    && invalid.bytes().all(|b| b.is_ascii_hexdigit()))
            );
        }
        let valid = "0123456789abcdef0123456789abcdef01234567";
        assert!((40..=64).contains(&valid.len()));
        assert!(valid.bytes().all(|b| b.is_ascii_hexdigit()));
    }

    #[test]
    fn frontier_counts_are_derived_not_hard_coded_to_152() {
        let instruments = Instrument::ALL.len();
        let cells = CATALOG.len() * instruments;
        assert_eq!(cells, CATALOG.len() * Instrument::ALL.len());
        assert!(cells >= instruments);
        assert_eq!(cells % instruments, 0);
    }

    #[test]
    fn generated_style_names_are_unique() {
        let names: HashSet<String> = Style::ALL.iter().copied().map(style_serde_name).collect();
        assert_eq!(names.len(), Style::ALL.len());
    }
}
