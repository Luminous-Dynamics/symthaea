// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Blind, backend-comparative Muse vocal release gate.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use symthaea::voice::diffsinger::DiffSingerEngine;
use symthaea::voice::functional_singing::FunctionalSingingEngine;
use symthaea::voice::kokoro_singing::KokoroSingingEngine;
use symthaea::voice::singing_engine::{SingingVoiceEngine, VocalPerformance, VocalStem};
use symthaea::voice::singing_quality::{VocalQualityMetrics, analyze_vocal_stem};
use symthaea_muse::{MusicalIntent, MusicalState, Style};

const PHRASES: &[(&str, &str)] = &[
    ("softly now the stars are waking", "legato_sustain"),
    ("the sun rises over the valley", "clear_consonants"),
    ("every silver river remembers", "liquid_consonants"),
    ("hold the light inside the silence", "soft_attack"),
    (
        "we are dancing through the morning",
        "rhythmic_articulation",
    ),
    ("consciousness shapes every note", "consonant_cluster"),
    ("bright birds scatter into blue", "plosive_cluster"),
    ("a quiet fire carries me home", "breath_control"),
    ("beautiful impossible harmony", "multisyllabic"),
    ("stay with me beyond the turning tide", "long_phrase"),
    ("soft breath clear heart open sky", "aspirates"),
    ("sing the world awake again", "strong_attack"),
    ("thin threads twist through the stillness", "fricatives"),
    ("joy joins every echo", "affricates"),
    ("round the wild horizon we wander", "diphthongs"),
    ("keep this little flame alive", "voiced_unvoiced"),
    ("moonlight moves across the water", "nasals"),
    ("rise and shimmer over me", "melisma"),
    ("hold on through the longest night", "sustain"),
    ("gently let the ending fall", "phrase_release"),
];

#[derive(Clone, Copy)]
struct GateCase {
    lyrics: &'static str,
    challenge: &'static str,
    category: &'static str,
    style: Style,
    valence: f32,
    arousal: f32,
    transpose_semitones: f32,
}

fn cases() -> Vec<GateCase> {
    PHRASES
        .iter()
        .flat_map(|&(lyrics, challenge)| {
            [
                ("intimate_low", Style::Folk, 0.35, 0.25, -5.0),
                ("center_legato", Style::Classical, 0.55, 0.48, 0.0),
                ("luminous_high", Style::Cinematic, 0.78, 0.72, 5.0),
            ]
            .into_iter()
            .map(
                move |(category, style, valence, arousal, transpose_semitones)| GateCase {
                    lyrics,
                    challenge,
                    category,
                    style,
                    valence,
                    arousal,
                    transpose_semitones,
                },
            )
        })
        .collect()
}

#[derive(Serialize)]
struct BlindClip {
    clip_id: String,
    case_id: usize,
    lyrics: String,
    audio: String,
    category: String,
    challenge: String,
}

#[derive(Serialize)]
struct KeyEntry {
    clip_id: String,
    backend: String,
    case_id: usize,
}

#[derive(Serialize)]
struct ObjectiveEntry {
    clip_id: String,
    backend: String,
    case_id: usize,
    metrics: VocalQualityMetrics,
}

#[derive(Deserialize)]
struct HumanRating {
    listener_id: String,
    clip_id: String,
    naturalness: f32,
    emotional_fit: f32,
    identity_consistency: f32,
    artifact_free: f32,
    lyric_words_understood: usize,
    lyric_words_total: usize,
}

/// Optional authorized human anchors. Paths remain private; imported copies
/// receive the same salted concealment as synthetic clips.
#[derive(Deserialize)]
struct ReferenceClip {
    case_id: usize,
    audio: PathBuf,
    lyrics: String,
    category: String,
    challenge: String,
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|x| x == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn save(path: &Path, stem: &VocalStem) -> Result<()> {
    let path = path.to_str().context("non-UTF8 output path")?;
    symthaea::voice::save_wav(&stem.samples, stem.sample_rate, path)
}

fn blind_id(salt: &str, backend: &str, case_id: usize) -> String {
    let digest = blake3::hash(format!("muse-vocal-v2:{salt}:{backend}:{case_id}").as_bytes());
    format!("vocal-{}", &digest.to_hex()[..12])
}

fn blind_salt(args: &[String], output: &Path) -> Result<String> {
    if let Some(seed) = arg(args, "--blind-seed") {
        return Ok(blake3::hash(seed.as_bytes()).to_hex().to_string());
    }
    let mut bytes = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .with_context(|| {
            format!(
                "secure blinding entropy unavailable for {}",
                output.display()
            )
        })?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn render_one(
    engine: &mut dyn SingingVoiceEngine,
    performance: &VocalPerformance,
    case_id: usize,
    case: &GateCase,
    salt: &str,
    output: &Path,
    clips: &mut Vec<BlindClip>,
    key: &mut Vec<KeyEntry>,
    objective: &mut Vec<ObjectiveEntry>,
) -> Result<()> {
    let stem = engine.render(performance)?;
    let id = blind_id(salt, &stem.backend, case_id);
    let filename = format!("{id}.wav");
    save(&output.join(&filename), &stem)?;
    let metrics = analyze_vocal_stem(performance, &stem);
    clips.push(BlindClip {
        clip_id: id.clone(),
        case_id,
        lyrics: performance.lyrics.clone(),
        audio: filename,
        category: case.category.to_string(),
        challenge: case.challenge.to_string(),
    });
    key.push(KeyEntry {
        clip_id: id.clone(),
        backend: stem.backend.clone(),
        case_id,
    });
    objective.push(ObjectiveEntry {
        clip_id: id,
        backend: stem.backend,
        case_id,
        metrics,
    });
    Ok(())
}

fn import_references(
    manifest: &Path,
    salt: &str,
    output: &Path,
    clips: &mut Vec<BlindClip>,
    key: &mut Vec<KeyEntry>,
) -> Result<()> {
    let references: Vec<ReferenceClip> = serde_json::from_slice(&std::fs::read(manifest)?)?;
    for reference in references {
        if !reference.audio.is_file() {
            anyhow::bail!("missing human reference: {}", reference.audio.display());
        }
        let id = blind_id(salt, "human-reference", reference.case_id);
        if clips.iter().any(|clip| clip.clip_id == id) {
            anyhow::bail!("duplicate human-reference case_id {}", reference.case_id);
        }
        let filename = format!("{id}.wav");
        std::fs::copy(&reference.audio, output.join(&filename))?;
        clips.push(BlindClip {
            clip_id: id.clone(),
            case_id: reference.case_id,
            lyrics: reference.lyrics,
            audio: filename,
            category: reference.category,
            challenge: reference.challenge,
        });
        key.push(KeyEntry {
            clip_id: id,
            backend: "human-reference".to_string(),
            case_id: reference.case_id,
        });
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let output = PathBuf::from(
        arg(&args, "--output").unwrap_or_else(|| "audio_output/muse_vocal_gate".into()),
    );
    let voice = arg(&args, "--voice").unwrap_or_else(|| "af_bella".into());
    let worker = arg(&args, "--diffsinger-worker").map(PathBuf::from);
    let ratings = arg(&args, "--ratings").map(PathBuf::from);
    let reference_manifest = arg(&args, "--reference-manifest").map(PathBuf::from);
    let minimum_listeners: usize = arg(&args, "--minimum-listeners")
        .as_deref()
        .unwrap_or("5")
        .parse()
        .context("invalid --minimum-listeners")?;
    std::fs::create_dir_all(&output)?;
    let salt = blind_salt(&args, &output)?;

    let mut kokoro = KokoroSingingEngine::load(&voice)?;
    let mut functional = FunctionalSingingEngine::default();
    let mut diffsinger = worker.as_deref().map(DiffSingerEngine::spawn).transpose()?;
    let mut clips = Vec::new();
    let mut key = Vec::new();
    let mut objective = Vec::new();

    let cases = cases();
    for (case_id, case) in cases.iter().enumerate() {
        let intent = MusicalIntent {
            valence: case.valence,
            arousal: case.arousal,
            energy: 0.55,
            bars: 2,
            seed: case_id as u64 + 9001,
            ..Default::default()
        };
        let state = MusicalState {
            valence: case.valence,
            arousal: case.arousal,
            consciousness_level: 0.72,
            ..Default::default()
        };
        let mut melody =
            symthaea_muse::theory_realize::compose_and_perform_melody(&intent, case.style, &state);
        let syllables = symthaea_muse::singing_bridge::syllable_count(case.lyrics).max(1);
        let wanted_notes = if case.challenge == "melisma" {
            syllables + 3
        } else {
            syllables
        };
        melody.truncate(wanted_notes);
        let ratio = 2.0f32.powf(case.transpose_semitones / 12.0);
        melody.iter_mut().for_each(|note| note.frequency *= ratio);
        let performance = VocalPerformance::from_melody(case.lyrics, &melody, "en")?;
        render_one(
            &mut kokoro,
            &performance,
            case_id,
            case,
            &salt,
            &output,
            &mut clips,
            &mut key,
            &mut objective,
        )?;
        render_one(
            &mut functional,
            &performance,
            case_id,
            case,
            &salt,
            &output,
            &mut clips,
            &mut key,
            &mut objective,
        )?;
        if let Some(engine) = diffsinger.as_mut() {
            render_one(
                engine,
                &performance,
                case_id,
                case,
                &salt,
                &output,
                &mut clips,
                &mut key,
                &mut objective,
            )?;
        }
    }
    if let Some(manifest) = reference_manifest {
        import_references(&manifest, &salt, &output, &mut clips, &mut key)?;
    }

    // Deterministic IDs conceal the backend; sort by ID so presentation order
    // does not reveal alternating render order.
    clips.sort_by(|a, b| a.clip_id.cmp(&b.clip_id));
    std::fs::write(
        output.join("blind_manifest.json"),
        serde_json::to_vec_pretty(&clips)?,
    )?;
    std::fs::write(
        output.join("private_backend_key.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "format_version": 2, "salt": salt, "entries": key
        }))?,
    )?;
    std::fs::write(
        output.join("objective_report.json"),
        serde_json::to_vec_pretty(&objective)?,
    )?;

    let rating_template: Vec<_> = clips.iter().map(|c| serde_json::json!({
        "listener_id": null, "clip_id": c.clip_id, "naturalness": null,
        "emotional_fit": null, "identity_consistency": null, "artifact_free": null,
        "lyric_words_understood": null, "lyric_words_total": c.lyrics.split_whitespace().count()
    })).collect();
    std::fs::write(
        output.join("ratings.template.json"),
        serde_json::to_vec_pretty(&rating_template)?,
    )?;

    let objective_pass = objective.iter().all(|entry| entry.metrics.objective_pass);
    let mut status = serde_json::json!({
        "objective_pass": objective_pass,
        "human_complete": false,
        "release_pass": false,
        "required": {"minimum_listeners": minimum_listeners, "naturalness_mean": 4.0,
                     "emotional_fit_mean": 4.0, "identity_consistency_mean": 4.0,
                     "artifact_free_mean": 4.0, "word_comprehension": 0.95}
    });
    if let Some(path) = ratings {
        let values: Vec<HumanRating> = serde_json::from_slice(&std::fs::read(path)?)?;
        let backend_by_id: BTreeMap<_, _> = key
            .iter()
            .map(|entry| (entry.clip_id.as_str(), entry.backend.as_str()))
            .collect();
        let listeners: std::collections::BTreeSet<_> =
            values.iter().map(|r| r.listener_id.clone()).collect();
        let mut by_clip_listener = BTreeMap::new();
        for rating in &values {
            if !(1.0..=5.0).contains(&rating.naturalness)
                || !(1.0..=5.0).contains(&rating.emotional_fit)
                || !(1.0..=5.0).contains(&rating.identity_consistency)
                || !(1.0..=5.0).contains(&rating.artifact_free)
                || rating.lyric_words_understood > rating.lyric_words_total
            {
                anyhow::bail!("invalid human rating for {}", rating.clip_id);
            }
            if by_clip_listener
                .insert((rating.clip_id.clone(), rating.listener_id.clone()), ())
                .is_some()
            {
                anyhow::bail!("duplicate listener/clip rating");
            }
        }
        let complete = listeners.len() >= minimum_listeners
            && clips.iter().all(|c| {
                listeners.iter().all(|listener| {
                    by_clip_listener.contains_key(&(c.clip_id.clone(), listener.clone()))
                })
            });
        let model_values: Vec<_> = values
            .iter()
            .filter(|rating| backend_by_id.get(rating.clip_id.as_str()) != Some(&"human-reference"))
            .collect();
        let reference_values: Vec<_> = values
            .iter()
            .filter(|rating| backend_by_id.get(rating.clip_id.as_str()) == Some(&"human-reference"))
            .collect();
        let naturalness = if model_values.is_empty() {
            0.0
        } else {
            model_values.iter().map(|r| r.naturalness).sum::<f32>() / model_values.len() as f32
        };
        let mean = |f: fn(&HumanRating) -> f32| {
            model_values.iter().map(|r| f(r)).sum::<f32>() / model_values.len().max(1) as f32
        };
        let emotional_fit = mean(|r| r.emotional_fit);
        let identity_consistency = mean(|r| r.identity_consistency);
        let artifact_free = mean(|r| r.artifact_free);
        let understood: usize = model_values.iter().map(|r| r.lyric_words_understood).sum();
        let total: usize = model_values.iter().map(|r| r.lyric_words_total).sum();
        let comprehension = understood as f32 / total.max(1) as f32;
        status = serde_json::json!({
            "objective_pass": objective_pass, "human_complete": complete,
            "listener_count": listeners.len(), "naturalness_mean": naturalness,
            "human_reference_rating_count": reference_values.len(),
            "emotional_fit_mean": emotional_fit, "identity_consistency_mean": identity_consistency,
            "artifact_free_mean": artifact_free, "word_comprehension": comprehension,
            "release_pass": objective_pass && complete && naturalness >= 4.0
                && emotional_fit >= 4.0 && identity_consistency >= 4.0
                && artifact_free >= 4.0 && comprehension >= 0.95
        });
    }
    std::fs::write(
        output.join("release_status.json"),
        serde_json::to_vec_pretty(&status)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&status)?);
    Ok(())
}
